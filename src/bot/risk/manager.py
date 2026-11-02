"""Point de passage obligé de tout ordre : applique les contrôles et tient l'état du risque."""

from collections.abc import Mapping
from datetime import date, datetime

import structlog

from bot.config import RiskLimits
from bot.risk.checks import check_order, halt_reason, is_valid_portfolio
from bot.risk.models import (
    PortfolioState,
    PriceSnapshot,
    ProposedOrder,
    RejectReason,
    RiskContext,
    RiskDecision,
)
from bot.risk.state import RiskState, RiskStateStore

logger = structlog.get_logger()


class RiskManager:
    """Valide les ordres, déclenche le kill switch et conserve leur état.

    Une fois déclenché, le kill switch reste actif, y compris après un redémarrage, jusqu'à
    `reset_halt`. Tout état enregistré l'est avant de retourner : si l'écriture échoue, l'exception
    remonte et aucun ordre n'est approuvé.
    """

    def __init__(self, limits: RiskLimits, store: RiskStateStore) -> None:
        """Charge l'état enregistré."""
        self._limits = limits
        self._store = store
        self._state = store.load()

    @property
    def state(self) -> RiskState:
        """Retourne l'état courant du risque."""
        return self._state

    @property
    def is_halted(self) -> bool:
        """Indique si le kill switch est déclenché."""
        return self._state.halt_reason is not None

    def _set_state(self, state: RiskState) -> None:
        if state != self._state:
            self._store.save(state)
            self._state = state

    def update(self, portfolio: PortfolioState, now: datetime) -> RiskState:
        """Met à jour le plus haut du capital et déclenche le kill switch si besoin.

        À appeler à chaque cycle, même sans ordre à envoyer. Un état de compte inexploitable ne
        change rien : il bloque les ordres dans `review`, sans fermer de position sur une donnée
        peut-être fausse.
        """
        if not is_valid_portfolio(portfolio):
            logger.warning("risk_invalid_portfolio")
            return self._state

        peak_equity = max(self._state.peak_equity or portfolio.equity, portfolio.equity)
        state = self._state.model_copy(update={"peak_equity": peak_equity})
        reason = halt_reason(portfolio, peak_equity, self._limits)
        if reason is not None and not self.is_halted:
            state = state.model_copy(update={"halt_reason": reason, "halted_at": now})
            logger.error(
                "kill_switch_triggered",
                reason=reason.value,
                equity=portfolio.equity,
                peak_equity=peak_equity,
                day_start_equity=portfolio.day_start_equity,
            )
        self._set_state(state)
        return self._state

    def _orders_sent_on(self, session_date: date) -> int:
        return self._state.orders_count if self._state.order_date == session_date else 0

    def review(
        self,
        order: ProposedOrder,
        portfolio: PortfolioState,
        market: Mapping[str, PriceSnapshot],
        context: RiskContext,
    ) -> RiskDecision:
        """Décide si un ordre est accepté.

        Un ordre accepté est compté immédiatement : il doit ensuite être envoyé ou journalisé comme
        abandonné, mais il ne peut plus être approuvé une seconde fois sans être recompté.
        """
        if not is_valid_portfolio(portfolio):
            return RiskDecision(approved=False, reason=RejectReason.INVALID_PORTFOLIO)

        self.update(portfolio, context.now)
        if self._state.halt_reason is not None:
            return RiskDecision(
                approved=False,
                reason=RejectReason.HALTED,
                detail=self._state.halt_reason.value,
            )

        decision = check_order(order, portfolio, market, context, self._limits)
        if not decision.approved:
            return decision

        sent = self._orders_sent_on(context.session_date)
        if sent >= self._limits.max_orders_per_day:
            detail = f"{sent} ordres sur la séance, limite {self._limits.max_orders_per_day}"
            return RiskDecision(approved=False, reason=RejectReason.ORDER_LIMIT, detail=detail)

        self._set_state(
            self._state.model_copy(
                update={"order_date": context.session_date, "orders_count": sent + 1}
            )
        )
        return decision

    def reset_halt(self, portfolio: PortfolioState) -> None:
        """Réarme le kill switch après une décision humaine, jamais automatiquement.

        Le plus haut repart du capital actuel, sinon la limite de drawdown se redéclencherait
        aussitôt. Une perte journalière encore dépassée le redéclenche au prochain `update`.
        """
        if not is_valid_portfolio(portfolio):
            raise ValueError("Réarmement impossible : état du compte inexploitable")
        logger.warning("kill_switch_reset", previous_reason=self._state.halt_reason)
        self._set_state(
            self._state.model_copy(
                update={"halt_reason": None, "halted_at": None, "peak_equity": portfolio.equity}
            )
        )
