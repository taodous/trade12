"""Données du dashboard, sans Streamlit : lecture seule du compte, du journal et des logs.

Tout ce que le dashboard affiche est calculé ici, pour être testé sans interface. Aucune fonction
n'envoie d'ordre ni n'écrit : le dashboard peut tourner à tout moment, bot actif ou arrêté.
"""

import json
import math
import sqlite3
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any, Protocol
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from alpaca.common.exceptions import APIError
from alpaca.trading.requests import GetPortfolioHistoryRequest
from requests.exceptions import RequestException

from audit.metrics import PerformanceMetrics, compute_metrics
from bot.config import PROJECT_ROOT, ConfigError, StrategyConfig, load_settings
from bot.data.market import (
    EXCHANGE_TIMEZONE,
    BarsClient,
    DataFetchError,
    SnapshotClient,
    create_data_client,
    fetch_bars,
    fetch_latest_prices,
)
from bot.execution.broker import Broker, BrokerError, EquitySample, MarketClock, PositionDetail
from bot.execution.client import AccountDataError, PaperModeError, create_trading_client
from bot.journal.store import EquityPoint, Journal
from bot.risk.state import RiskStateError
from bot.status import LOG_PATTERN, BotStatus, StatusClient, build_status

DAILY_TIMEFRAME = "1D"
INTRADAY_REPORTING = "market_hours"
# Une seule séance pour l'historique intraday : celle du jour, ou la dernière si le marché dort.
INTRADAY_PERIOD = "1D"
# Un point de départ, la clôture de la veille, et au moins une séance.
MIN_CURVE_POINTS = 2


class DashboardDataClient(BarsClient, SnapshotClient, Protocol):
    """Partie de l'API de données lue par le dashboard : barres et derniers prix de SPY."""


# Pannes attendues : chaque section du dashboard les affiche et réessaie au rafraîchissement.
DASHBOARD_ERRORS = (
    ConfigError,
    PaperModeError,
    APIError,
    RequestException,
    BrokerError,
    DataFetchError,
    AccountDataError,
    RiskStateError,
    OSError,
    sqlite3.Error,
)


def connect() -> tuple[StatusClient, DashboardDataClient]:
    """Crée les clients paper en lecture : trading (compte, positions) et données (SPY)."""
    settings = load_settings()
    return create_trading_client(settings), create_data_client(settings)


@dataclass(frozen=True)
class Heartbeat:
    """Dernier signe de vie du bot, d'après son fichier de log."""

    last_seen: datetime | None
    active: bool


@dataclass(frozen=True)
class LiveView:
    """Partie du dashboard rafraîchie en continu."""

    status: BotStatus
    clock: MarketClock
    positions: list[PositionDetail]
    intraday: pd.Series
    heartbeat: Heartbeat


@dataclass(frozen=True)
class HistoryView:
    """Courbes depuis le début du paper, qui ne changent qu'à la clôture."""

    start: date | None
    equity: pd.Series
    benchmark: pd.Series
    drawdown: pd.Series
    metrics: PerformanceMetrics | None
    benchmark_metrics: PerformanceMetrics | None
    bot_sessions: list[date]

    @property
    def total_return(self) -> float | None:
        """Rendement depuis la clôture qui précède le début du paper, non annualisé."""
        if len(self.equity) < MIN_CURVE_POINTS:
            return None
        return float(self.equity.iloc[-1] / self.equity.iloc[0] - 1.0)

    @property
    def benchmark_return(self) -> float | None:
        """Rendement du benchmark sur la même période."""
        values = self.benchmark.dropna()
        if len(values) < MIN_CURVE_POINTS:
            return None
        return float(values.iloc[-1] / values.iloc[0] - 1.0)


def last_log_time(log_dir: Path) -> datetime | None:
    """Retourne l'horodatage de la dernière ligne lisible du fichier de log le plus récent."""
    files = sorted(log_dir.glob(LOG_PATTERN)) if log_dir.is_dir() else []
    if not files:
        return None
    for line in reversed(files[-1].read_text(encoding="utf-8").splitlines()):
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        stamp = entry.get("timestamp") if isinstance(entry, dict) else None
        if isinstance(stamp, str):
            return datetime.fromisoformat(stamp)
    return None


def heartbeat(signals: Iterable[datetime | None], now: datetime, max_age: timedelta) -> Heartbeat:
    """Le bot est actif si son dernier signe de vie est assez récent.

    Avant l'ouverture, le bot logue à chaque tick. Une fois le cycle du jour terminé, il ne logue
    plus, mais relève le capital à chaque tick : les deux signes sont donc nécessaires.
    """
    known = [signal for signal in signals if signal is not None]
    last_seen = max(known) if known else None
    return Heartbeat(last_seen, last_seen is not None and now - last_seen <= max_age)


def session_equity(
    samples: Sequence[EquitySample], start: date, today: date, live_equity: float
) -> pd.Series:
    """Capital à la clôture de chaque séance depuis `start`, précédé de la clôture de la veille.

    Alpaca date chaque fenêtre journalière par son début, juste après la clôture précédente : sa
    date UTC est donc celle de la séance. La valeur du jour, encore provisoire, est remplacée par
    le capital en direct.
    """
    by_day = {sample.timestamp.astimezone(UTC).date(): sample.equity for sample in samples}
    if today in by_day:
        by_day[today] = live_equity
    before = [day for day in by_day if day < start]
    days = sorted([day for day in by_day if day >= start] + ([max(before)] if before else []))
    return pd.Series(
        [by_day[day] for day in days],
        index=pd.DatetimeIndex(pd.to_datetime(days)),
        dtype=float,
        name="equity",
    )


def intraday_equity(samples: Sequence[EquitySample]) -> pd.Series:
    """Capital de la séance, fenêtre par fenêtre, en heure de New York."""
    index = pd.DatetimeIndex([sample.timestamp for sample in samples]).tz_convert(EXCHANGE_TIMEZONE)
    return pd.Series([sample.equity for sample in samples], index=index, dtype=float, name="equity")


def normalized_benchmark(closes: pd.Series, index: pd.DatetimeIndex, base: float) -> pd.Series:
    """Valeur qu'aurait eue `base` investi dans le benchmark à la première date de `index`."""
    if index.empty:
        return pd.Series(dtype=float, name="benchmark")
    aligned = closes.reindex(closes.index.union(index)).sort_index().ffill().reindex(index)
    if aligned.isna().iloc[0]:
        return pd.Series(np.nan, index=index, name="benchmark")
    scaled: pd.Series = aligned * (base / float(aligned.iloc[0]))
    return scaled.rename("benchmark")


def drawdown_series(equity: pd.Series) -> pd.Series:
    """Baisse depuis le plus haut atteint, positive, à chaque date."""
    return (1.0 - equity / equity.cummax()).rename("drawdown")


def performance(equity: pd.Series) -> PerformanceMetrics | None:
    """Métriques des rendements journaliers, ou None avant la première séance complète."""
    returns = equity.pct_change().dropna()
    if returns.empty:
        return None
    return compute_metrics(returns)


def decision_table(decisions: Sequence[Mapping[str, Any]], limit: int) -> pd.DataFrame:
    """Les `limit` dernières décisions, de la plus récente à la plus ancienne."""
    rows = []
    for row in list(decisions)[-limit:][::-1]:
        inputs = json.loads(row["inputs"])
        rows.append(
            {
                "heure (UTC)": datetime.fromisoformat(row["timestamp"]).astimezone(UTC),
                "séance": row["session"],
                "symbole": row["symbol"],
                "décision": row["note"],
                "poids visé": inputs.get("target_weight"),
                "ordre": f"{row['side']} {row['quantity']:g}" if row["side"] else "",
                "risque": _risk_label(row),
                "statut": row["status"] or "",
                "prix exécuté": row["filled_price"],
                "slippage": row["slippage_pct"],
            }
        )
    return pd.DataFrame(rows)


def _risk_label(row: Mapping[str, Any]) -> str:
    if row["risk_approved"] is None:
        return ""
    return "accepté" if row["risk_approved"] else f"refusé : {row['risk_reason']}"


def mean_slippage(decisions: Iterable[Mapping[str, Any]]) -> tuple[float | None, int]:
    """Slippage moyen des ordres exécutés, et leur nombre. Positif quand il est défavorable."""
    values = [
        float(row["slippage_pct"])
        for row in decisions
        if row["slippage_pct"] is not None and math.isfinite(float(row["slippage_pct"]))
    ]
    return (float(np.mean(values)) if values else None), len(values)


@dataclass(frozen=True)
class JournalView:
    """Contenu du journal utile au dashboard. Vide si le bot n'a jamais tourné."""

    decisions: list[dict[str, Any]]
    equity: list[EquityPoint]

    @property
    def bot_sessions(self) -> list[date]:
        """Séances où le bot a tourné, marché ouvert."""
        return [point.session for point in self.equity]

    @property
    def last_record(self) -> datetime | None:
        """Dernier relevé du capital, fait à chaque tick marché ouvert."""
        return max((point.recorded_at for point in self.equity), default=None)


def read_journal(config: StrategyConfig) -> JournalView:
    """Lit le journal en lecture seule, sans jamais le créer."""
    path = PROJECT_ROOT / config.live.journal_file
    if not path.exists():
        return JournalView([], [])
    journal = Journal(path, read_only=True)
    try:
        return JournalView(journal.decisions(), journal.equity_history())
    finally:
        journal.close()


def load_live(config: StrategyConfig, client: StatusClient, now: datetime) -> LiveView:
    """Lit l'état courant : compte, risque, positions, séance du jour et activité du bot."""
    broker = Broker(client, config.api_retry)
    request = GetPortfolioHistoryRequest(
        period=INTRADAY_PERIOD,
        timeframe=config.dashboard.intraday_timeframe,
        intraday_reporting=INTRADAY_REPORTING,
    )
    max_age = timedelta(
        seconds=config.live.poll_interval_seconds * config.dashboard.heartbeat_intervals
    )
    return LiveView(
        status=build_status(config, client, now),
        clock=broker.clock(),
        positions=broker.position_details(),
        intraday=intraday_equity(broker.portfolio_history(request)),
        heartbeat=heartbeat(
            [last_log_time(PROJECT_ROOT / config.live.log_dir), read_journal(config).last_record],
            now,
            max_age,
        ),
    )


def load_history(
    config: StrategyConfig,
    client: StatusClient,
    data_client: DashboardDataClient,
    now: datetime,
) -> HistoryView:
    """Courbe du capital depuis le début du paper, comparée au benchmark sur la même période.

    Le début du paper est la première séance relevée par le bot dans son journal.
    """
    bot_sessions = read_journal(config).bot_sessions
    empty = pd.Series(dtype=float)
    if not bot_sessions:
        return HistoryView(None, empty, empty, empty, None, None, [])

    start = min(bot_sessions)
    broker = Broker(client, config.api_retry)
    today = now.astimezone(ZoneInfo(EXCHANGE_TIMEZONE)).date()
    # Quelques jours avant le début, pour avoir la clôture de la veille comme point de départ.
    lookback = datetime.combine(start - timedelta(days=7), datetime.min.time(), tzinfo=UTC)
    samples = broker.portfolio_history(
        GetPortfolioHistoryRequest(timeframe=DAILY_TIMEFRAME, start=lookback)
    )
    equity = session_equity(samples, start, today, float(broker.account().equity))
    if equity.empty:
        return HistoryView(start, empty, empty, empty, None, None, bot_sessions)

    symbol = config.backtest.benchmark
    # Séances terminées seulement : l'abonnement gratuit refuse les données SIP récentes. Le point
    # du jour vient du dernier prix.
    end = datetime.combine(today, datetime.min.time(), tzinfo=ZoneInfo(EXCHANGE_TIMEZONE))
    closes = fetch_bars(data_client, symbol, lookback.date(), end, config.data, config.api_retry)
    benchmark_closes = closes["close"][closes.index < pd.Timestamp(today)]
    latest = fetch_latest_prices(data_client, [symbol], config.live.price_feed, config.api_retry)
    if symbol in latest and pd.Timestamp(today) in equity.index:
        benchmark_closes = pd.concat(
            [benchmark_closes, pd.Series([latest[symbol].price], index=[pd.Timestamp(today)])]
        )
    benchmark = normalized_benchmark(
        benchmark_closes, pd.DatetimeIndex(equity.index), float(equity.iloc[0])
    )
    return HistoryView(
        start=start,
        equity=equity,
        benchmark=benchmark,
        drawdown=drawdown_series(equity),
        metrics=performance(equity),
        benchmark_metrics=performance(benchmark.dropna()),
        bot_sessions=bot_sessions,
    )
