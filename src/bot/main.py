"""Boucle principale du bot, en paper trading uniquement."""

import argparse
import sqlite3
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
from typing import Protocol
from zoneinfo import ZoneInfo

import pandas as pd
import structlog
from alpaca.common.exceptions import APIError
from requests.exceptions import RequestException

from bot.config import (
    PROJECT_ROOT,
    ConfigError,
    StrategyConfig,
    load_settings,
    load_strategy_config,
)
from bot.data.market import (
    EXCHANGE_TIMEZONE,
    AlpacaMarketData,
    DataFetchError,
    LatestPrice,
    Session,
    create_data_client,
    last_completed_session,
)
from bot.execution.broker import (
    Broker,
    BrokerError,
    make_client_order_id,
)
from bot.execution.client import AccountDataError, PaperModeError, create_trading_client
from bot.execution.planner import PlanItem, PlanNote, plan_orders
from bot.journal.store import STATUS_NOT_FOUND, Decision, Journal
from bot.logs import configure_logging
from bot.risk.checks import closing_orders
from bot.risk.manager import RiskManager
from bot.risk.models import (
    OrderSide,
    PortfolioState,
    Position,
    PriceSnapshot,
    ProposedOrder,
    RiskContext,
)
from bot.risk.state import InMemoryRiskStateStore, JsonRiskStateStore, RiskStateError
from bot.strategy.base import WeightStrategy
from bot.strategy.factory import build_strategy, rebalance_policy

logger = structlog.get_logger()

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_HALTED = 2
STATUS_DRY_RUN = "dry_run"
STATUS_UNKNOWN = "unknown"
ORDER_PREFIX = "bot"
KILL_PREFIX = "kill"
RECOVERABLE_ERRORS = (APIError, RequestException, DataFetchError, BrokerError, AccountDataError)


class MarketData(Protocol):
    """Données de marché nécessaires à un cycle."""

    def sessions(self, start: date, end: date) -> list[Session]:
        """Retourne les séances de la période."""
        ...

    def closes(self, sessions: Sequence[Session]) -> pd.DataFrame:
        """Retourne les clôtures de ces séances, vides quand une donnée manque."""
        ...

    def latest_prices(self) -> dict[str, LatestPrice]:
        """Retourne le dernier prix connu de chaque symbole."""
        ...


class TradingBot:
    """Enchaîne le kill switch, les signaux, le risque, l'envoi et le suivi des ordres."""

    def __init__(
        self,
        config: StrategyConfig,
        strategy: WeightStrategy,
        broker: Broker,
        market: MarketData,
        risk: RiskManager,
        journal: Journal,
        *,
        dry_run: bool = False,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        """Assemble le bot. Lève `ValueError` si le poids dépasse la limite par position."""
        self._config = config
        self._strategy = strategy
        self._broker = broker
        self._market = market
        self._risk = risk
        self._journal = journal
        self._dry_run = dry_run
        self._sleep = sleep
        self._rebalance = rebalance_policy(config.strategy)
        if strategy.max_weight > config.risk.max_position_pct:
            raise ValueError(
                f"Poids de {strategy.max_weight:.4f} supérieur à la limite par position"
            )

    def tick(self, now: datetime) -> None:
        """Exécute une itération : kill switch, suivi des ordres puis décisions de la séance."""
        clock = self._broker.clock()
        self._risk.update(self._broker.account_summary(), now)
        if self._risk.is_halted:
            self._liquidate(now)
            return

        self._reconcile()
        session = now.astimezone(ZoneInfo(EXCHANGE_TIMEZONE)).date()
        if not self._dry_run and self._journal.is_cycle_completed(session):
            return
        if not clock.is_open and not self._dry_run:
            logger.info("market_closed")
            return
        # Comme le backtest, une stratégie rééquilibrée ne décide qu'une fois par mois : à la
        # première séance du mois où le bot tourne.
        if (
            self._rebalance is not None
            and not self._dry_run
            and self._journal.has_completed_cycle_in_month(session)
        ):
            logger.info("not_a_rebalance_session")
            return
        self._trade(now, session)

    @property
    def is_halted(self) -> bool:
        """Indique si le kill switch est déclenché."""
        return self._risk.is_halted

    def _reconcile(self) -> None:
        for pending in self._journal.pending_orders():
            report = self._broker.find_order(pending.client_order_id)
            if report is None:
                self._journal.record_status(
                    pending.decision_id,
                    STATUS_NOT_FOUND,
                    is_final=True,
                    detail="ordre inconnu du broker",
                )
                logger.warning("order_not_found", client_order_id=pending.client_order_id)
                continue
            self._journal.record_report(pending.decision_id, report)
            if report.is_final:
                logger.info(
                    "order_final",
                    client_order_id=report.client_order_id,
                    status=report.status,
                    filled_quantity=report.filled_quantity,
                    filled_price=report.filled_price,
                )

    def _load_signals(
        self, now: datetime, session: date
    ) -> tuple[Session, pd.DataFrame, dict[str, float | None]]:
        lookback = timedelta(days=self._config.live.signal_lookback_days)
        calendar = self._market.sessions(session - lookback, session)
        min_age = timedelta(minutes=self._config.data.min_bar_age_minutes)
        last = last_completed_session(calendar, now, min_age)
        sessions = [item for item in calendar if item.date <= last.date]
        warm_up = self._strategy.warm_up
        if len(sessions) < warm_up:
            raise DataFetchError(
                f"{len(sessions)} séances disponibles pour un préchauffage de {warm_up}"
            )

        closes = self._market.closes(sessions)
        # Une clôture manquante rendrait le signal hors marché et déclencherait une vente à tort.
        complete = closes.columns[closes.tail(warm_up).notna().all()]
        targets: dict[str, float | None] = dict.fromkeys(closes.columns)
        # Sans un symbole indispensable, comme celui du filtre de régime, aucun poids n'est fiable.
        required = self._strategy.required_symbols
        if len(complete) and required <= {str(symbol) for symbol in complete}:
            generated = self._strategy.target_weights(closes[complete]).iloc[-1]
            targets.update({str(symbol): float(generated[symbol]) for symbol in complete})
        return last, closes, targets

    def _trade(self, now: datetime, session: date) -> None:
        account = self._broker.account()
        if not account.can_trade:
            logger.error("account_cannot_trade", status=account.status.value)
            return

        last, closes, targets = self._load_signals(now, session)
        latest = self._market.latest_prices()
        # Lu après les prix : une transaction reçue pendant le cycle ne doit pas paraître future.
        decided_at = self._broker.clock().timestamp
        snapshots = {
            symbol: PriceSnapshot(price.price, float(closes[symbol].iloc[-1]), price.as_of)
            for symbol, price in latest.items()
            if symbol in closes.columns
        }
        prices = {symbol: snapshot.price for symbol, snapshot in snapshots.items()}
        portfolio, open_symbols = self._broker.portfolio(account, prices)
        items = plan_orders(targets, portfolio, open_symbols, prices, last.date, self._rebalance)
        context = RiskContext(decided_at, session, last.date)

        for item in items:
            inputs = self._inputs(item.symbol, last.date, closes, snapshots, portfolio)
            inputs["target_weight"] = item.target
            portfolio = self._process(item, inputs, portfolio, snapshots, context)

        if self._risk.is_halted:
            return
        self._journal.complete_cycle(session, now)
        logger.info("cycle_completed", session=session.isoformat(), decisions=len(items))

    @staticmethod
    def _inputs(
        symbol: str,
        signal_session: date,
        closes: pd.DataFrame,
        snapshots: Mapping[str, PriceSnapshot],
        portfolio: PortfolioState,
    ) -> dict[str, object]:
        series = closes[symbol]
        snapshot = snapshots.get(symbol)
        return {
            "signal_session": signal_session,
            "bars": int(series.notna().sum()),
            "first_bar": pd.Timestamp(str(series.index[0])).date(),
            "last_close": float(series.iloc[-1]),
            "price": snapshot.price if snapshot else None,
            "price_as_of": snapshot.as_of if snapshot else None,
            "equity": portfolio.equity,
        }

    def _process(
        self,
        item: PlanItem,
        inputs: dict[str, object],
        portfolio: PortfolioState,
        snapshots: Mapping[str, PriceSnapshot],
        context: RiskContext,
    ) -> PortfolioState:
        order = item.order
        if order is None:
            self._journal.record_decision(
                Decision(
                    context.now, context.session_date, item.symbol, item.signal, item.note, inputs
                )
            )
            return portfolio

        decision = self._risk.review(order, portfolio, snapshots, context)
        client_order_id = make_client_order_id(
            ORDER_PREFIX, context.session_date, order.symbol, order.side
        )
        snapshot = snapshots.get(order.symbol)
        decision_id = self._journal.record_decision(
            Decision(
                timestamp=context.now,
                session=context.session_date,
                symbol=order.symbol,
                signal=item.signal,
                note=PlanNote.ORDER,
                inputs=inputs,
                side=order.side,
                quantity=order.quantity,
                risk_approved=decision.approved,
                risk_reason=decision.reason.value if decision.reason else None,
                risk_detail=decision.detail,
                client_order_id=client_order_id if decision.approved else None,
                reference_price=snapshot.price if snapshot else None,
            )
        )
        if not decision.approved:
            logger.warning(
                "order_rejected_by_risk",
                symbol=order.symbol,
                reason=decision.reason.value if decision.reason else None,
                detail=decision.detail,
            )
            return portfolio

        self._send(decision_id, order.symbol, order.side, order.quantity, client_order_id)
        # Le prochain ordre du cycle doit voir celui-ci, sinon l'exposition serait sous-estimée.
        price = snapshot.price if snapshot else 0.0
        return _with_pending(portfolio, order, price)

    def _send(
        self,
        decision_id: int,
        symbol: str,
        side: OrderSide,
        quantity: float,
        client_order_id: str,
    ) -> None:
        if self._dry_run:
            self._journal.record_status(
                decision_id, STATUS_DRY_RUN, is_final=True, detail="simulation, aucun envoi"
            )
            logger.info("order_simulated", symbol=symbol, side=side.value, quantity=quantity)
            return
        try:
            report = self._broker.submit(symbol, side, quantity, client_order_id)
        except (APIError, RequestException, BrokerError) as exc:
            # Sort inconnu : l'ordre est peut-être parti. Le suivi le retrouvera par identifiant.
            self._journal.record_status(
                decision_id, STATUS_UNKNOWN, is_final=False, detail=type(exc).__name__
            )
            logger.error("order_status_unknown", symbol=symbol, client_order_id=client_order_id)
            return
        self._journal.record_report(decision_id, report)
        logger.info(
            "order_submitted",
            symbol=symbol,
            side=side.value,
            quantity=quantity,
            status=report.status,
            client_order_id=client_order_id,
        )

    def _liquidate(self, now: datetime) -> None:
        """Annule les ordres ouverts et ferme toutes les positions, sans passer par `review`.

        L'identifiant contient l'heure : si une fermeture a échoué, un nouvel essai après
        redémarrage ne doit pas se heurter à l'identifiant de l'ordre refusé.
        """
        reason = self._risk.state.halt_reason
        logger.error("kill_switch_active", reason=reason.value if reason else None)
        if not self._dry_run:
            self._broker.cancel_all()
            self._wait_for_cancellations()

        session = now.astimezone(ZoneInfo(EXCHANGE_TIMEZONE)).date()
        account = self._broker.account_summary()
        portfolio = replace(account, positions=self._broker.positions())
        for order in closing_orders(portfolio):
            client_order_id = f"{KILL_PREFIX}-{now:%Y%m%d%H%M%S}-{order.symbol}"
            decision_id = self._journal.record_decision(
                Decision(
                    timestamp=now,
                    session=session,
                    symbol=order.symbol,
                    signal=None,
                    note="kill_switch",
                    inputs={"equity": account.equity, "peak": self._risk.state.peak_equity},
                    side=order.side,
                    quantity=order.quantity,
                    risk_approved=True,
                    risk_reason=reason.value if reason else None,
                    client_order_id=client_order_id,
                )
            )
            self._send(decision_id, order.symbol, order.side, order.quantity, client_order_id)

    def _wait_for_cancellations(self) -> None:
        for _ in range(self._config.live.cancel_wait_seconds):
            if not self._broker.open_orders():
                return
            self._sleep(1.0)
        logger.error("open_orders_remaining")

    def run(self, *, once: bool, now: Callable[[], datetime]) -> int:
        """Répète les itérations et retourne le code de sortie du processus.

        Une panne de l'API est retentée à l'itération suivante, puis arrête le bot après trop
        d'échecs consécutifs. Pendant une panne, le kill switch ne peut pas lire le capital.
        """
        failures = 0
        while True:
            try:
                self.tick(now())
                failures = 0
            except RECOVERABLE_ERRORS as exc:
                failures += 1
                logger.error("tick_failed", error=type(exc).__name__, failures=failures)
                if once or failures >= self._config.live.max_consecutive_errors:
                    return EXIT_ERROR
            if self.is_halted:
                logger.error("bot_stopped_by_kill_switch")
                return EXIT_HALTED
            if once:
                return EXIT_OK
            self._sleep(float(self._config.live.poll_interval_seconds))


def _with_pending(portfolio: PortfolioState, order: ProposedOrder, price: float) -> PortfolioState:
    signed = order.quantity if order.side is OrderSide.BUY else -order.quantity
    current = portfolio.positions.get(order.symbol, Position(0.0, 0.0))
    updated = replace(current, pending_notional=current.pending_notional + signed * price)
    return replace(portfolio, positions={**portfolio.positions, order.symbol: updated})


def main(argv: Sequence[str] | None = None) -> int:
    """Lance le bot et retourne le code de sortie du processus."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--once", action="store_true", help="exécute une seule itération")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="une seule itération, sans envoi d'ordre et sans modifier l'état réel du risque",
    )
    parser.add_argument(
        "--reset-kill-switch",
        action="store_true",
        help="réarme le kill switch après une décision humaine, puis s'arrête",
    )
    args = parser.parse_args(argv)

    configure_logging()
    journal = None
    try:
        config = load_strategy_config()
        settings = load_settings()
        trading_client = create_trading_client(settings)
        broker = Broker(trading_client, config.api_retry)
        state_store = JsonRiskStateStore(PROJECT_ROOT / config.risk_state_file)

        if args.reset_kill_switch:
            RiskManager(config.risk, state_store).reset_halt(broker.account_summary())
            return EXIT_OK

        journal_path = PROJECT_ROOT / config.live.journal_file
        if args.dry_run:
            journal_path = journal_path.with_suffix(".dry-run.db")
            risk = RiskManager(config.risk, InMemoryRiskStateStore(state_store.load()))
        else:
            risk = RiskManager(config.risk, state_store)
        journal = Journal(journal_path)
        market = AlpacaMarketData(
            create_data_client(settings), trading_client, config, config.api_retry
        )
        bot = TradingBot(
            config,
            build_strategy(config.strategy, config.data.symbols),
            broker,
            market,
            risk,
            journal,
            dry_run=args.dry_run,
        )
        return bot.run(once=args.once or args.dry_run, now=lambda: datetime.now(UTC))
    except KeyboardInterrupt:
        logger.info("bot_interrupted")
        return EXIT_OK
    except (
        ConfigError,
        PaperModeError,
        RiskStateError,
        ValueError,
        OSError,
        sqlite3.Error,
        *RECOVERABLE_ERRORS,
    ) as exc:
        logger.error("bot_failed", error=type(exc).__name__, reason=str(exc))
        return EXIT_ERROR
    finally:
        if journal is not None:
            journal.close()


if __name__ == "__main__":
    sys.exit(main())
