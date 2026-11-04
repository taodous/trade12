"""Récupération des séances et des barres journalières auprès d'Alpaca.

Une barre est indexée par sa date de séance et contient le prix de clôture de cette séance : elle
n'est connue qu'après la clôture. Une décision prise pendant la séance t ne doit donc utiliser que
les barres antérieures à t.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from typing import Protocol
from zoneinfo import ZoneInfo

import pandas as pd
import structlog
from alpaca.common.types import RawData
from alpaca.data.enums import Adjustment, DataFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.models import BarSet, Snapshot
from alpaca.data.requests import StockBarsRequest, StockSnapshotRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.models import Calendar
from alpaca.trading.requests import GetCalendarRequest

from bot.config import DataConfig, RetryConfig, Settings, StrategyConfig
from bot.data.quality import clean_bars
from bot.retry import with_backoff

logger = structlog.get_logger()

EXCHANGE_TIMEZONE = "America/New_York"
BAR_COLUMNS = ["open", "high", "low", "close", "volume", "trade_count", "vwap"]


class DataFetchError(RuntimeError):
    """Réponse de l'API de données absente ou inexploitable."""


class BarsClient(Protocol):
    """Partie de l'API de données nécessaire à la récupération des barres."""

    def get_stock_bars(self, request_params: StockBarsRequest) -> BarSet | RawData:
        """Retourne les barres demandées."""
        ...


class CalendarClient(Protocol):
    """Partie de l'API nécessaire à la lecture du calendrier de marché."""

    def get_calendar(self, filters: GetCalendarRequest | None = None) -> list[Calendar] | RawData:
        """Retourne les séances demandées."""
        ...


class SnapshotClient(Protocol):
    """Partie de l'API de données nécessaire à la lecture des derniers prix."""

    def get_stock_snapshot(
        self, request_params: StockSnapshotRequest
    ) -> dict[str, Snapshot] | RawData:
        """Retourne les instantanés demandés."""
        ...


class LiveDataClient(BarsClient, SnapshotClient, Protocol):
    """Partie de l'API de données utilisée par le bot."""


@dataclass(frozen=True)
class LatestPrice:
    """Dernière transaction connue d'un symbole."""

    price: float
    as_of: datetime


@dataclass(frozen=True)
class Session:
    """Séance de marché."""

    date: date
    close: datetime


def create_data_client(settings: Settings) -> StockHistoricalDataClient:
    """Crée le client de données de marché, en lecture seule."""
    return StockHistoricalDataClient(
        api_key=settings.alpaca_api_key.get_secret_value(),
        secret_key=settings.alpaca_secret_key.get_secret_value(),
    )


def fetch_sessions(
    client: CalendarClient, start: date, end: date, retry: RetryConfig
) -> list[Session]:
    """Retourne les séances du calendrier entre `start` et `end`, incluses."""
    filters = GetCalendarRequest(start=start, end=end)
    calendar = with_backoff(lambda: client.get_calendar(filters), retry)
    if isinstance(calendar, dict):
        raise DataFetchError("Réponse brute reçue au lieu du calendrier")
    timezone = ZoneInfo(EXCHANGE_TIMEZONE)
    return [Session(day.date, day.close.replace(tzinfo=timezone)) for day in calendar]


def last_completed_session(
    sessions: Sequence[Session], now: datetime, min_bar_age: timedelta
) -> Session:
    """Retourne la dernière séance terminée depuis au moins `min_bar_age`.

    Écarter la séance en cours évite de stocker une barre incomplète comme si elle était définitive.
    """
    completed = [session for session in sessions if session.close + min_bar_age <= now]
    if not completed:
        raise DataFetchError("Aucune séance terminée sur la période demandée")
    return max(completed, key=lambda session: session.date)


def fetch_bars(
    client: BarsClient,
    symbol: str,
    start: date,
    end: datetime,
    config: DataConfig,
    retry: RetryConfig,
) -> pd.DataFrame:
    """Retourne les barres journalières de `symbol`, indexées par date de séance."""
    request = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame.Day,
        start=datetime.combine(start, time.min, tzinfo=UTC),
        end=end,
        feed=DataFeed(config.feed),
        adjustment=Adjustment(config.adjustment),
    )
    result = with_backoff(lambda: client.get_stock_bars(request), retry)
    if not isinstance(result, BarSet):
        raise DataFetchError(f"Réponse brute reçue au lieu des barres de {symbol}")

    frame = result.df
    if frame.empty:
        raise DataFetchError(f"Aucune barre reçue pour {symbol}")

    missing = [column for column in BAR_COLUMNS if column not in frame.columns]
    if missing:
        raise DataFetchError(f"Colonnes absentes pour {symbol} : {', '.join(missing)}")

    bars = frame.droplevel("symbol")[BAR_COLUMNS].copy()
    timestamps = pd.DatetimeIndex(bars.index)
    bars.index = timestamps.tz_convert(EXCHANGE_TIMEZONE).normalize().tz_localize(None)
    bars.index.name = "session"
    return bars.sort_index()


def fetch_latest_prices(
    client: SnapshotClient, symbols: Sequence[str], feed: str, retry: RetryConfig
) -> dict[str, LatestPrice]:
    """Retourne la dernière transaction de chaque symbole. Un symbole sans donnée est absent."""
    request = StockSnapshotRequest(symbol_or_symbols=list(symbols), feed=DataFeed(feed))
    snapshots = with_backoff(lambda: client.get_stock_snapshot(request), retry)
    prices = {}
    for symbol, snapshot in snapshots.items():
        if not isinstance(snapshot, Snapshot):
            raise DataFetchError("Réponse brute reçue au lieu des derniers prix")
        if snapshot.latest_trade is not None:
            trade = snapshot.latest_trade
            prices[symbol] = LatestPrice(trade.price, trade.timestamp)
    return prices


class AlpacaMarketData:
    """Données de marché du bot en paper trading."""

    def __init__(
        self,
        data_client: LiveDataClient,
        calendar_client: CalendarClient,
        config: StrategyConfig,
        retry: RetryConfig,
    ) -> None:
        """Associe la source aux clients Alpaca."""
        self._data_client = data_client
        self._calendar_client = calendar_client
        self._config = config
        self._retry = retry

    def sessions(self, start: date, end: date) -> list[Session]:
        """Retourne les séances de la période."""
        return fetch_sessions(self._calendar_client, start, end, self._retry)

    def closes(self, sessions: Sequence[Session]) -> pd.DataFrame:
        """Retourne les clôtures contrôlées, une colonne par symbole.

        Une séance manquante ou une barre retirée reste vide, et un symbole sans donnée donne une
        colonne vide : aucune valeur n'est reportée, le symbole est simplement exclu du cycle.
        """
        expected = [session.date for session in sessions]
        index = pd.DatetimeIndex(pd.to_datetime(expected), name="session")
        closes = pd.DataFrame(index=index, columns=self._config.data.symbols, dtype=float)
        for symbol in self._config.data.symbols:
            try:
                bars = fetch_bars(
                    self._data_client,
                    symbol,
                    expected[0],
                    sessions[-1].close,
                    self._config.data,
                    self._retry,
                )
            except DataFetchError as exc:
                logger.error("bars_unavailable", symbol=symbol, reason=str(exc))
                continue
            cleaned = clean_bars(bars, expected, self._config.risk.max_price_change_pct)
            for issue in cleaned.issues:
                logger.warning(
                    "data_issue",
                    symbol=symbol,
                    session=issue.session.isoformat(),
                    kind=issue.kind.value,
                )
            closes[symbol] = cleaned.bars["close"].reindex(index)
        return closes

    def latest_prices(self) -> dict[str, LatestPrice]:
        """Retourne le dernier prix connu de chaque symbole."""
        return fetch_latest_prices(
            self._data_client,
            self._config.data.symbols,
            self._config.live.price_feed,
            self._retry,
        )
