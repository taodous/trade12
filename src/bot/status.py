"""État du bot en paper trading, en lecture seule : aucun ordre, aucune écriture.

Affiche le capital, le kill switch, les positions et leur poids visé, le dernier cycle, les séances
sans relevé et les erreurs du dernier fichier de log. Retourne 2 si le kill switch est déclenché.
"""

import json
import math
import sqlite3
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any, Protocol
from zoneinfo import ZoneInfo

import structlog
from alpaca.common.exceptions import APIError
from requests.exceptions import RequestException

from bot.config import (
    PROJECT_ROOT,
    ConfigError,
    RiskLimits,
    StrategyConfig,
    load_settings,
    load_strategy_config,
)
from bot.data.market import EXCHANGE_TIMEZONE, CalendarClient, DataFetchError, fetch_sessions
from bot.execution.broker import Broker, BrokerClient, BrokerError
from bot.execution.client import AccountDataError, PaperModeError, create_trading_client
from bot.journal.store import Journal
from bot.logs import configure_logging
from bot.risk.models import Position
from bot.risk.state import JsonRiskStateStore, RiskState, RiskStateError

logger = structlog.get_logger()

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_HALTED = 2
LOG_PATTERN = "bot-*.jsonl"


@dataclass(frozen=True)
class PositionLine:
    """Position du compte, comparée au poids visé au dernier cycle."""

    symbol: str
    quantity: float
    market_value: float
    weight: float
    target_weight: float | None
    in_universe: bool


@dataclass(frozen=True)
class OrderLine:
    """Ordre proposé au dernier cycle et son sort."""

    symbol: str
    side: str
    quantity: float
    risk_approved: bool
    risk_reason: str | None
    status: str | None
    filled_price: float | None


@dataclass(frozen=True)
class LogSummary:
    """Résumé du dernier fichier de log."""

    path: Path | None
    errors: int = 0
    warnings: int = 0
    unreadable: int = 0
    last_error: str | None = None


@dataclass(frozen=True)
class BotStatus:
    """État complet du bot à un instant."""

    generated_at: datetime
    equity: float
    last_equity: float
    cash: float
    risk: RiskState
    limits: RiskLimits
    positions: list[PositionLine]
    journal_found: bool
    last_cycle: date | None
    recorded_sessions: int
    first_recorded: date | None
    missed_sessions: list[date]
    last_orders: list[OrderLine]
    log: LogSummary
    notes: list[str] = field(default_factory=list)

    @property
    def exposure(self) -> float:
        """Part du capital investie."""
        return 1.0 - self.cash / self.equity

    @property
    def day_change(self) -> float:
        """Variation du capital depuis la clôture précédente."""
        return self.equity / self.last_equity - 1.0

    @property
    def drawdown(self) -> float:
        """Baisse depuis le plus haut enregistré par le module risque, positive."""
        peak = self.risk.peak_equity or self.equity
        return max(0.0, 1.0 - self.equity / peak)


def summarize_log(log_dir: Path) -> LogSummary:
    """Compte les erreurs et avertissements du fichier de log le plus récent."""
    files = sorted(log_dir.glob(LOG_PATTERN)) if log_dir.is_dir() else []
    if not files:
        return LogSummary(path=None)
    path = files[-1]
    errors = warnings = unreadable = 0
    last_error = None
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            unreadable += 1
            continue
        level = entry.get("level") if isinstance(entry, dict) else None
        if level == "error":
            errors += 1
            reason = entry.get("reason") or entry.get("error") or ""
            last_error = f"{entry.get('timestamp', '?')} {entry.get('event', '?')} {reason}".strip()
        elif level == "warning":
            warnings += 1
    return LogSummary(path, errors, warnings, unreadable, last_error)


def last_target_weights(decisions: Iterable[Mapping[str, Any]], session: date) -> dict[str, float]:
    """Retourne le poids visé de chaque symbole lors du cycle de `session`."""
    weights: dict[str, float] = {}
    for row in decisions:
        if row["session"] != session.isoformat() or row["note"] == "kill_switch":
            continue
        weight = json.loads(row["inputs"]).get("target_weight")
        if isinstance(weight, int | float) and math.isfinite(weight):
            weights[row["symbol"]] = float(weight)
    return weights


def last_orders(decisions: Iterable[Mapping[str, Any]], session: date) -> list[OrderLine]:
    """Retourne les ordres proposés lors du cycle de `session`, acceptés ou refusés."""
    return [
        OrderLine(
            symbol=row["symbol"],
            side=row["side"],
            quantity=float(row["quantity"]),
            risk_approved=bool(row["risk_approved"]),
            risk_reason=row["risk_reason"],
            status=row["status"],
            filled_price=row["filled_price"],
        )
        for row in decisions
        if row["session"] == session.isoformat() and row["side"] is not None
    ]


def missed_sessions(trading_days: Sequence[date], recorded: Iterable[date]) -> list[date]:
    """Retourne les séances de bourse sans relevé du capital, donc sans bot actif."""
    seen = set(recorded)
    return [day for day in trading_days if day not in seen]


def position_lines(
    positions: Mapping[str, Position],
    equity: float,
    targets: Mapping[str, float],
    universe: Iterable[str],
) -> list[PositionLine]:
    """Décrit chaque position, y compris celles visées mais absentes du compte."""
    allowed = set(universe)
    symbols = sorted(set(positions) | {symbol for symbol, weight in targets.items() if weight > 0})
    lines = []
    for symbol in symbols:
        position = positions.get(symbol, Position(0.0, 0.0))
        lines.append(
            PositionLine(
                symbol=symbol,
                quantity=position.quantity,
                market_value=position.market_value,
                weight=position.market_value / equity,
                target_weight=targets.get(symbol),
                in_universe=symbol in allowed,
            )
        )
    return lines


def _percent(value: float | None) -> str:
    return "n/d" if value is None or not math.isfinite(value) else f"{value:.2%}"


def format_status(status: BotStatus) -> str:
    """Met l'état en forme pour la console."""
    risk = status.risk
    lines = [
        f"État du bot paper au {status.generated_at:%Y-%m-%d %H:%M} UTC",
        "",
        "Capital",
        f"  capital {status.equity:,.2f} $, cash {status.cash:,.2f} $, "
        f"exposition {_percent(status.exposure)}",
        f"  variation du jour {_percent(status.day_change)} "
        f"(kill switch à -{_percent(status.limits.max_daily_loss_pct)})",
        f"  drawdown {_percent(status.drawdown)} depuis le plus haut "
        f"{(risk.peak_equity or status.equity):,.2f} $ "
        f"(arrêt à {_percent(status.limits.max_drawdown_pct)})",
        "",
    ]
    if risk.halt_reason is not None:
        halted_at = f"{risk.halted_at:%Y-%m-%d %H:%M}" if risk.halted_at else "?"
        lines += [
            f"KILL SWITCH DÉCLENCHÉ : {risk.halt_reason.value}, le {halted_at} UTC",
            "  Le bot ferme les positions et refuse tout ordre. Reprise sur décision humaine :",
            "  uv run python -m bot.main --reset-kill-switch",
            "",
        ]
    else:
        lines += ["Kill switch armé, non déclenché", ""]

    lines.append("Positions (poids réel / poids visé au dernier cycle)")
    if not status.positions:
        lines.append("  aucune")
    for item in status.positions:
        flag = "" if item.in_universe else "  HORS UNIVERS"
        lines.append(
            f"  {item.symbol:<6} {item.quantity:>10.2f} parts {item.market_value:>14,.2f} $ "
            f"{_percent(item.weight):>8} / {_percent(item.target_weight):>8}{flag}"
        )
    lines.append("")

    if not status.journal_found:
        lines += ["Journal absent : le bot n'a pas encore tourné.", ""]
    else:
        last = status.last_cycle.isoformat() if status.last_cycle else "aucun"
        first = status.first_recorded.isoformat() if status.first_recorded else "aucun"
        lines += [
            "Journal",
            f"  dernier cycle de décision : {last}",
            f"  séances avec relevé du capital : {status.recorded_sessions} (depuis le {first})",
        ]
        if status.missed_sessions:
            days = ", ".join(day.isoformat() for day in status.missed_sessions)
            lines.append(f"  séances manquées (bot inactif) : {days}")
        if status.last_orders:
            lines.append("  ordres du dernier cycle :")
        for order in status.last_orders:
            verdict = "accepté" if order.risk_approved else f"refusé ({order.risk_reason})"
            fill = f" exécuté à {order.filled_price:.2f}" if order.filled_price else ""
            lines.append(
                f"    {order.side:<4} {order.quantity:>8.2f} {order.symbol:<6} {verdict}, "
                f"statut {order.status or 'n/d'}{fill}"
            )
        lines.append("")

    log = status.log
    if log.path is None:
        lines.append("Logs : aucun fichier")
    else:
        lines.append(
            f"Logs : {log.path.name}, {log.errors} erreurs, {log.warnings} avertissements"
            + (f", {log.unreadable} lignes illisibles" if log.unreadable else "")
        )
        if log.last_error:
            lines.append(f"  dernière erreur : {log.last_error}")
    lines += [f"Note : {note}" for note in status.notes]
    return "\n".join(lines)


def _read_journal(path: Path) -> tuple[list[dict[str, Any]], date | None, list[date]] | None:
    if not path.exists():
        return None
    journal = Journal(path, read_only=True)
    try:
        recorded = [point.session for point in journal.equity_history()]
        return journal.decisions(), journal.last_completed_cycle(), recorded
    finally:
        journal.close()


class StatusClient(BrokerClient, CalendarClient, Protocol):
    """Partie de l'API de trading lue par le suivi : compte, positions et calendrier."""


def build_status(config: StrategyConfig, trading_client: StatusClient, now: datetime) -> BotStatus:
    """Lit le compte paper, l'état du risque, le journal et les logs, sans rien modifier."""
    broker = Broker(trading_client, config.api_retry)
    account = broker.account()
    positions = broker.positions()
    risk = JsonRiskStateStore(PROJECT_ROOT / config.risk_state_file).load()
    equity = float(account.equity)

    journal = _read_journal(PROJECT_ROOT / config.live.journal_file)
    decisions, last_cycle, recorded = journal or ([], None, [])
    targets = last_target_weights(decisions, last_cycle) if last_cycle else {}
    missed: list[date] = []
    today = now.astimezone(ZoneInfo(EXCHANGE_TIMEZONE)).date()
    if recorded and min(recorded) < today:
        calendar = fetch_sessions(
            trading_client, min(recorded), today - timedelta(days=1), config.api_retry
        )
        missed = missed_sessions([session.date for session in calendar], recorded)

    return BotStatus(
        generated_at=now,
        equity=equity,
        last_equity=float(account.last_equity),
        cash=float(account.cash),
        risk=risk,
        limits=config.risk,
        positions=position_lines(positions, equity, targets, config.data.symbols),
        journal_found=journal is not None,
        last_cycle=last_cycle,
        recorded_sessions=len(recorded),
        first_recorded=min(recorded) if recorded else None,
        missed_sessions=missed,
        last_orders=last_orders(decisions, last_cycle) if last_cycle else [],
        log=summarize_log(PROJECT_ROOT / config.live.log_dir),
        notes=[] if account.can_trade else ["le compte ne peut pas trader"],
    )


def collect_status(now: datetime) -> BotStatus:
    """Charge la configuration et le client paper, puis lit l'état du bot."""
    return build_status(load_strategy_config(), create_trading_client(load_settings()), now)


def main() -> int:
    """Affiche l'état du bot et retourne le code de sortie du processus."""
    configure_logging()
    try:
        status = collect_status(datetime.now(UTC))
    except (
        ConfigError,
        PaperModeError,
        RiskStateError,
        AccountDataError,
        BrokerError,
        DataFetchError,
        APIError,
        RequestException,
        OSError,
        sqlite3.Error,
    ) as exc:
        logger.error("status_failed", error=type(exc).__name__, reason=str(exc))
        return EXIT_ERROR
    print(format_status(status))
    return EXIT_HALTED if status.risk.halt_reason is not None else EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
