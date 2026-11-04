"""Traduction des signaux en ordres proposés, pure et sans effet de bord."""

import math
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from datetime import date
from enum import StrEnum

from bot.risk.models import OrderSide, PortfolioState, ProposedOrder
from bot.strategy.base import LONG


class PlanNote(StrEnum):
    """Raison de l'action retenue pour un symbole."""

    ORDER = "order"
    HOLD = "hold"
    MISSING_DATA = "missing_data"
    OPEN_ORDER = "open_order"
    NO_PRICE = "no_price"
    TOO_SMALL = "too_small"


@dataclass(frozen=True)
class PlanItem:
    """Décision pour un symbole. `signal` est vide faute de données fiables."""

    symbol: str
    signal: int | None
    note: PlanNote
    order: ProposedOrder | None = None


def plan_orders(
    signals: Mapping[str, int | None],
    portfolio: PortfolioState,
    open_order_symbols: Collection[str],
    prices: Mapping[str, float],
    weight: float,
    signal_session: date,
) -> list[PlanItem]:
    """Retourne une décision par symbole, les ventes en premier pour libérer le capital.

    Une donnée manquante ne produit jamais d'ordre : sans signal fiable, la position reste telle
    quelle au lieu d'être vendue. Une position existante n'est pas rééquilibrée, comme dans le
    backtest. Un symbole avec un ordre encore ouvert est ignoré pour éviter un doublon.
    """
    items = [
        _plan_symbol(symbol, signal, portfolio, open_order_symbols, prices, weight, signal_session)
        for symbol, signal in signals.items()
    ]
    return sorted(items, key=lambda item: item.order is None or item.order.side is OrderSide.BUY)


def _plan_symbol(
    symbol: str,
    signal: int | None,
    portfolio: PortfolioState,
    open_order_symbols: Collection[str],
    prices: Mapping[str, float],
    weight: float,
    signal_session: date,
) -> PlanItem:
    if signal is None:
        return PlanItem(symbol, signal, PlanNote.MISSING_DATA)
    if symbol in open_order_symbols:
        return PlanItem(symbol, signal, PlanNote.OPEN_ORDER)

    position = portfolio.positions.get(symbol)
    quantity = position.quantity if position is not None else 0.0

    if signal != LONG:
        if quantity == 0:
            return PlanItem(symbol, signal, PlanNote.HOLD)
        side = OrderSide.SELL if quantity > 0 else OrderSide.BUY
        order = ProposedOrder(symbol, side, abs(quantity), signal_session)
        return PlanItem(symbol, signal, PlanNote.ORDER, order)

    if quantity != 0:
        return PlanItem(symbol, signal, PlanNote.HOLD)
    price = prices.get(symbol)
    if price is None or not math.isfinite(price) or price <= 0:
        return PlanItem(symbol, signal, PlanNote.NO_PRICE)
    shares = math.floor(weight * portfolio.equity / price)
    if shares < 1:
        return PlanItem(symbol, signal, PlanNote.TOO_SMALL)
    order = ProposedOrder(symbol, OrderSide.BUY, float(shares), signal_session)
    return PlanItem(symbol, signal, PlanNote.ORDER, order)
