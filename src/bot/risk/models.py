"""Types échangés avec le module de risque."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date, datetime
from enum import StrEnum


class OrderSide(StrEnum):
    """Sens d'un ordre."""

    BUY = "buy"
    SELL = "sell"


class RejectReason(StrEnum):
    """Motif de refus d'un ordre, enregistré au journal."""

    HALTED = "halted"
    INVALID_PORTFOLIO = "invalid_portfolio"
    INVALID_ORDER = "invalid_order"
    MISSING_DATA = "missing_data"
    STALE_DATA = "stale_data"
    ABERRANT_PRICE = "aberrant_price"
    STALE_SIGNAL = "stale_signal"
    SHORT_NOT_ALLOWED = "short_not_allowed"
    POSITION_LIMIT = "position_limit"
    EXPOSURE_LIMIT = "exposure_limit"
    ORDER_LIMIT = "order_limit"


class HaltReason(StrEnum):
    """Motif de déclenchement du kill switch."""

    DAILY_LOSS = "daily_loss"
    MAX_DRAWDOWN = "max_drawdown"


@dataclass(frozen=True)
class ProposedOrder:
    """Ordre proposé par la stratégie. `signal_session` est la séance dont provient le signal."""

    symbol: str
    side: OrderSide
    quantity: float
    signal_session: date


@dataclass(frozen=True)
class ClosingOrder:
    """Ordre de fermeture émis par le kill switch."""

    symbol: str
    side: OrderSide
    quantity: float


@dataclass(frozen=True)
class Position:
    """Position du compte. `pending_notional` est la valeur signée de ses ordres encore ouverts."""

    quantity: float
    market_value: float
    pending_notional: float = 0.0


@dataclass(frozen=True)
class PortfolioState:
    """État du compte. Le pouvoir d'achat en est absent : le risque se mesure sur le capital."""

    equity: float
    day_start_equity: float
    positions: Mapping[str, Position] = field(default_factory=dict)


@dataclass(frozen=True)
class PriceSnapshot:
    """Prix de référence d'un symbole au moment de l'ordre."""

    price: float
    previous_close: float
    as_of: datetime


@dataclass(frozen=True)
class RiskContext:
    """Repères temporels d'une décision. `last_completed_session` précède la séance en cours."""

    now: datetime
    session_date: date
    last_completed_session: date


@dataclass(frozen=True)
class RiskDecision:
    """Décision sur un ordre, accompagnée de son motif pour le journal."""

    approved: bool
    reason: RejectReason | None = None
    detail: str = ""
