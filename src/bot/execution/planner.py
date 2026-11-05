"""Traduction des poids visés en ordres proposés, pure et sans effet de bord."""

import math
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from datetime import date
from enum import StrEnum

from bot.config import RebalanceConfig
from bot.risk.models import OrderSide, PortfolioState, ProposedOrder
from bot.strategy.base import FLAT, LONG


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
    """Décision pour un symbole. `target` est vide faute de données fiables."""

    symbol: str
    target: float | None
    note: PlanNote
    order: ProposedOrder | None = None

    @property
    def signal(self) -> int | None:
        """Retourne le signal journalisé : long si un poids est visé, hors marché sinon."""
        if self.target is None:
            return None
        return LONG if self.target > 0 else FLAT


def plan_orders(
    targets: Mapping[str, float | None],
    portfolio: PortfolioState,
    open_order_symbols: Collection[str],
    prices: Mapping[str, float],
    signal_session: date,
    rebalance: RebalanceConfig | None,
) -> list[PlanItem]:
    """Retourne une décision par symbole, les ventes en premier pour libérer le capital.

    Une donnée manquante ne produit jamais d'ordre : sans poids fiable, la position reste telle
    quelle au lieu d'être vendue. Un symbole avec un ordre encore ouvert est ignoré pour éviter un
    doublon. Sans rééquilibrage, une position existante n'est pas modifiée tant que son poids visé
    reste positif, comme dans le backtest. Avec rééquilibrage, elle est ramenée à son poids visé si
    l'écart dépasse la bande, en parts entières.
    """
    items = [
        _plan_symbol(
            symbol, target, portfolio, open_order_symbols, prices, signal_session, rebalance
        )
        for symbol, target in targets.items()
    ]
    return sorted(items, key=lambda item: item.order is None or item.order.side is OrderSide.BUY)


def _valid_price(price: float | None) -> bool:
    return price is not None and math.isfinite(price) and price > 0


def _plan_symbol(
    symbol: str,
    target: float | None,
    portfolio: PortfolioState,
    open_order_symbols: Collection[str],
    prices: Mapping[str, float],
    signal_session: date,
    rebalance: RebalanceConfig | None,
) -> PlanItem:
    if target is None:
        return PlanItem(symbol, target, PlanNote.MISSING_DATA)
    if symbol in open_order_symbols:
        return PlanItem(symbol, target, PlanNote.OPEN_ORDER)

    position = portfolio.positions.get(symbol)
    quantity = position.quantity if position is not None else 0.0

    if target <= 0:
        if quantity == 0:
            return PlanItem(symbol, target, PlanNote.HOLD)
        side = OrderSide.SELL if quantity > 0 else OrderSide.BUY
        order = ProposedOrder(symbol, side, abs(quantity), signal_session)
        return PlanItem(symbol, target, PlanNote.ORDER, order)

    if rebalance is None and quantity != 0:
        return PlanItem(symbol, target, PlanNote.HOLD)
    price = prices.get(symbol)
    if price is None or not _valid_price(price):
        return PlanItem(symbol, target, PlanNote.NO_PRICE)
    desired = math.floor(target * portfolio.equity / price)
    if rebalance is None:
        if desired < 1:
            return PlanItem(symbol, target, PlanNote.TOO_SMALL)
        order = ProposedOrder(symbol, OrderSide.BUY, float(desired), signal_session)
        return PlanItem(symbol, target, PlanNote.ORDER, order)

    change = desired - quantity
    if abs(change) * price < rebalance.band_pct * portfolio.equity:
        return PlanItem(symbol, target, PlanNote.HOLD)
    if change > 0:
        shares = math.floor(change)
        if shares < 1:
            return PlanItem(symbol, target, PlanNote.TOO_SMALL)
        order = ProposedOrder(symbol, OrderSide.BUY, float(shares), signal_session)
    else:
        order = ProposedOrder(symbol, OrderSide.SELL, -change, signal_session)
    return PlanItem(symbol, target, PlanNote.ORDER, order)
