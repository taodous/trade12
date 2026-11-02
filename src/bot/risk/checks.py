"""Contrôles de risque purs et sans effet de bord.

Chaque contrôle refuse tout ce qu'il ne peut pas vérifier. Une comparaison avec NaN étant toujours
fausse, les valeurs sont testées explicitement avant d'être comparées aux limites.
"""

import math
from collections.abc import Mapping

from bot.config import RiskLimits
from bot.risk.models import (
    ClosingOrder,
    HaltReason,
    OrderSide,
    PortfolioState,
    Position,
    PriceSnapshot,
    ProposedOrder,
    RejectReason,
    RiskContext,
    RiskDecision,
)

_NO_POSITION = Position(quantity=0.0, market_value=0.0)
APPROVED = RiskDecision(approved=True)


def _reject(reason: RejectReason, detail: str = "") -> RiskDecision:
    return RiskDecision(approved=False, reason=reason, detail=detail)


def is_valid_portfolio(portfolio: PortfolioState) -> bool:
    """Indique si l'état du compte est exploitable pour mesurer le risque."""
    if not (math.isfinite(portfolio.equity) and portfolio.equity > 0):
        return False
    if not (math.isfinite(portfolio.day_start_equity) and portfolio.day_start_equity > 0):
        return False
    return all(
        math.isfinite(position.quantity)
        and math.isfinite(position.market_value)
        and math.isfinite(position.pending_notional)
        for position in portfolio.positions.values()
    )


def halt_reason(
    portfolio: PortfolioState, peak_equity: float, limits: RiskLimits
) -> HaltReason | None:
    """Retourne le motif de déclenchement du kill switch, ou None si les limites sont respectées.

    `portfolio` doit avoir été validé par `is_valid_portfolio`. Le capital est comparé à un seuil
    plutôt que la perte à la limite : `1 - 90000 / 100000` donne 0.09999999999999998 en flottant,
    ce qui laisserait passer une perte exactement égale à la limite.
    """
    if portfolio.equity <= portfolio.day_start_equity * (1 - limits.max_daily_loss_pct):
        return HaltReason.DAILY_LOSS
    if portfolio.equity <= peak_equity * (1 - limits.max_drawdown_pct):
        return HaltReason.MAX_DRAWDOWN
    return None


def closing_orders(portfolio: PortfolioState) -> list[ClosingOrder]:
    """Retourne les ordres qui ferment toutes les positions.

    Ils ne passent pas par `check_order` : une fermeture d'urgence ne doit être bloquée ni par des
    données périmées ni par le nombre d'ordres. Les ordres ouverts sont à annuler avant de les
    envoyer, sinon une vente en attente pourrait s'y ajouter et créer une position à découvert.
    """
    return [
        ClosingOrder(
            symbol=symbol,
            side=OrderSide.SELL if position.quantity > 0 else OrderSide.BUY,
            quantity=abs(position.quantity),
        )
        for symbol, position in sorted(portfolio.positions.items())
        if position.quantity != 0
    ]


def _check_market_data(
    snapshot: PriceSnapshot | None, context: RiskContext, limits: RiskLimits
) -> RiskDecision:
    if snapshot is None:
        return _reject(RejectReason.MISSING_DATA, "aucun prix")
    if snapshot.as_of.tzinfo is None:
        return _reject(RejectReason.MISSING_DATA, "horodatage sans fuseau")
    for price in (snapshot.price, snapshot.previous_close):
        if not (math.isfinite(price) and price > 0):
            return _reject(RejectReason.MISSING_DATA, f"prix invalide : {price}")

    age = (context.now - snapshot.as_of).total_seconds()
    if age < 0:
        return _reject(RejectReason.STALE_DATA, "prix daté du futur")
    if age > limits.max_data_age_seconds:
        return _reject(RejectReason.STALE_DATA, f"prix âgé de {age:.0f} s")

    change = snapshot.price / snapshot.previous_close - 1
    if abs(change) > limits.max_price_change_pct:
        return _reject(RejectReason.ABERRANT_PRICE, f"variation de {change:.4f}")
    return APPROVED


def _check_size(
    order: ProposedOrder,
    portfolio: PortfolioState,
    price: float,
    limits: RiskLimits,
) -> RiskDecision:
    position = portfolio.positions.get(order.symbol, _NO_POSITION)
    signed_quantity = order.quantity if order.side is OrderSide.BUY else -order.quantity
    current = position.quantity * price + position.pending_notional
    projected = (position.quantity + signed_quantity) * price + position.pending_notional
    increases_risk = abs(projected) > abs(current)

    if projected < 0 and (current >= 0 or increases_risk) and not limits.allow_short:
        return _reject(RejectReason.SHORT_NOT_ALLOWED, f"position visée : {projected:.2f}")
    # Un ordre qui réduit une position reste permis même si la limite est dépassée.
    if not increases_risk:
        return APPROVED

    position_limit = limits.max_position_pct * portfolio.equity
    if abs(projected) > position_limit:
        detail = f"position visée {abs(projected):.2f}, limite {position_limit:.2f}"
        return _reject(RejectReason.POSITION_LIMIT, detail)

    others = sum(
        abs(other.market_value + other.pending_notional)
        for symbol, other in portfolio.positions.items()
        if symbol != order.symbol
    )
    exposure_limit = limits.max_total_exposure_pct * portfolio.equity
    if others + abs(projected) > exposure_limit:
        detail = f"exposition visée {others + abs(projected):.2f}, limite {exposure_limit:.2f}"
        return _reject(RejectReason.EXPOSURE_LIMIT, detail)
    return APPROVED


def check_order(
    order: ProposedOrder,
    portfolio: PortfolioState,
    market: Mapping[str, PriceSnapshot],
    context: RiskContext,
    limits: RiskLimits,
) -> RiskDecision:
    """Vérifie un ordre contre les données, le signal et les limites de taille.

    Le signal doit provenir de la dernière séance terminée. Plus ancien, il est périmé. Plus
    récent, il utiliserait une clôture qui n'existe pas encore. L'état du compte et le kill switch
    sont contrôlés en amont par `RiskManager`.
    """
    if not (math.isfinite(order.quantity) and order.quantity > 0):
        return _reject(RejectReason.INVALID_ORDER, f"quantité invalide : {order.quantity}")

    snapshot = market.get(order.symbol)
    data_decision = _check_market_data(snapshot, context, limits)
    if not data_decision.approved or snapshot is None:
        return data_decision

    if order.signal_session != context.last_completed_session:
        detail = (
            f"signal du {order.signal_session}, "
            f"dernière séance terminée le {context.last_completed_session}"
        )
        return _reject(RejectReason.STALE_SIGNAL, detail)

    return _check_size(order, portfolio, snapshot.price, limits)
