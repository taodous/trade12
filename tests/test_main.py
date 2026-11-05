import json
import sqlite3
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import uuid4
from zoneinfo import ZoneInfo

import pandas as pd
import pytest
from alpaca.common.exceptions import APIError
from alpaca.common.types import RawData
from alpaca.trading.enums import (
    AccountStatus,
    AssetClass,
    AssetExchange,
    OrderClass,
    OrderStatus,
    OrderType,
    PositionSide,
    TimeInForce,
)
from alpaca.trading.enums import OrderSide as AlpacaOrderSide
from alpaca.trading.models import Clock, Order, TradeAccount
from alpaca.trading.models import Position as AlpacaPosition
from alpaca.trading.requests import GetOrdersRequest, MarketOrderRequest, OrderRequest
from requests.exceptions import ConnectionError as RequestsConnectionError

from bot import main as main_module
from bot.config import (
    ConfigError,
    MovingAverageParams,
    MovingAverageSettings,
    StrategyConfig,
    load_strategy_config,
)
from bot.data.market import DataFetchError, LatestPrice, Session
from bot.execution.broker import REJECTED_BY_BROKER, Broker
from bot.journal.store import STATUS_NOT_FOUND, EquityPoint, Journal
from bot.logs import configure_logging
from bot.main import EXIT_ERROR, EXIT_HALTED, EXIT_OK, STATUS_DRY_RUN, STATUS_UNKNOWN, TradingBot
from bot.risk.manager import RiskManager
from bot.risk.models import HaltReason
from bot.risk.state import InMemoryRiskStateStore, JsonRiskStateStore, RiskState
from bot.strategy.base import FLAT, LONG

NEW_YORK = ZoneInfo("America/New_York")
NOW = datetime(2026, 11, 3, 15, 0, tzinfo=UTC)
TODAY = date(2026, 11, 3)
YESTERDAY = date(2026, 11, 2)
SYMBOLS = ["A", "B", "C", "D", "E"]
PRICE = 100.0


def api_error(status_code: int) -> APIError:
    http_error = type("E", (), {"response": type("R", (), {"status_code": status_code})()})()
    return APIError('{"code": 1, "message": "erreur"}', http_error)  # type: ignore[no-untyped-call]


def make_order(
    client_order_id: str,
    symbol: str,
    side: AlpacaOrderSide,
    quantity: float,
    status: OrderStatus = OrderStatus.ACCEPTED,
    filled_price: float | None = None,
) -> Order:
    return Order(
        id=uuid4(),
        client_order_id=client_order_id,
        created_at=NOW,
        updated_at=NOW,
        submitted_at=NOW,
        asset_id=uuid4(),
        symbol=symbol,
        asset_class=AssetClass.US_EQUITY,
        qty=str(quantity),
        filled_qty=str(quantity if filled_price is not None else 0),
        filled_avg_price=str(filled_price) if filled_price is not None else None,
        order_class=OrderClass.SIMPLE,
        order_type=OrderType.MARKET,
        type=OrderType.MARKET,
        side=side,
        time_in_force=TimeInForce.DAY,
        status=status,
        extended_hours=False,
    )


def make_position(symbol: str, quantity: float) -> AlpacaPosition:
    return AlpacaPosition(
        asset_id=uuid4(),
        symbol=symbol,
        exchange=AssetExchange.ARCA,
        asset_class=AssetClass.US_EQUITY,
        avg_entry_price=str(PRICE),
        qty=str(abs(quantity)),
        side=PositionSide.LONG if quantity > 0 else PositionSide.SHORT,
        cost_basis=str(abs(quantity) * PRICE),
        market_value=str(quantity * PRICE),
        current_price=str(PRICE),
    )


class FakeClient:
    """Client de trading simulé, qui refuse comme Alpaca un identifiant déjà utilisé."""

    def __init__(self) -> None:
        self.is_open = True
        self.clock_time = NOW
        self.equity = "100000"
        self.last_equity = "100000"
        self.status = AccountStatus.ACTIVE
        self.positions: list[AlpacaPosition] = []
        self.open_orders: list[Order] = []
        self.orders: dict[str, Order] = {}
        self.submitted: list[MarketOrderRequest] = []
        self.cancel_calls = 0
        self.submit_error: Exception | None = None
        self.submit_reaches_broker = False
        self.clock_error: Exception | None = None
        self.next_open = NOW

    def get_clock(self) -> Clock | RawData:
        if self.clock_error is not None:
            raise self.clock_error
        return Clock(
            timestamp=self.clock_time,
            is_open=self.is_open,
            next_open=self.next_open,
            next_close=NOW,
        )

    def get_account(self) -> TradeAccount | RawData:
        return TradeAccount(
            id=uuid4(),
            account_number="numero",
            status=self.status,
            currency="USD",
            equity=self.equity,
            last_equity=self.last_equity,
            cash=self.equity,
            buying_power="400000",
            trading_blocked=False,
            account_blocked=False,
        )

    def get_all_positions(self) -> list[AlpacaPosition] | RawData:
        return self.positions

    def get_orders(self, filter: GetOrdersRequest | None = None) -> list[Order] | RawData:
        return self.open_orders

    def submit_order(self, order_data: OrderRequest) -> Order | RawData:
        assert isinstance(order_data, MarketOrderRequest)
        assert order_data.client_order_id is not None
        assert order_data.symbol is not None
        assert order_data.qty is not None
        order = make_order(
            order_data.client_order_id,
            order_data.symbol,
            AlpacaOrderSide(order_data.side),
            order_data.qty,
        )
        if self.submit_error is not None:
            if self.submit_reaches_broker:
                self.orders[order_data.client_order_id] = order
            raise self.submit_error
        if order_data.client_order_id in self.orders:
            raise api_error(422)
        self.submitted.append(order_data)
        self.orders[order_data.client_order_id] = order
        return order

    def get_order_by_client_id(self, client_id: str) -> Order | RawData:
        if client_id not in self.orders:
            raise api_error(404)
        return self.orders[client_id]

    def cancel_orders(self) -> object:
        self.cancel_calls += 1
        self.open_orders = []
        return []

    def submitted_symbols(self) -> list[str]:
        return [str(order.symbol) for order in self.submitted]


def calendar() -> list[Session]:
    days = [date(2026, 10, 28), date(2026, 10, 29), date(2026, 10, 30), YESTERDAY, TODAY]
    close = datetime(2026, 1, 1, 16).time()
    return [Session(day, datetime.combine(day, close, tzinfo=NEW_YORK)) for day in days]


@dataclass
class FakeMarket:
    """Données de marché simulées, cohérentes avec le prix de référence."""

    missing_symbols: list[str] = field(default_factory=list)
    price_age: timedelta = timedelta(minutes=1)
    closes_error: Exception | None = None
    # Dernier prix, qui peut s'écarter de la clôture de la veille, comme lors d'un gap.
    latest_price: float = PRICE

    def sessions(self, start: date, end: date) -> list[Session]:
        return calendar()

    def closes(self, sessions: Sequence[Session]) -> pd.DataFrame:
        if self.closes_error is not None:
            raise self.closes_error
        index = pd.DatetimeIndex(pd.to_datetime([session.date for session in sessions]))
        closes = pd.DataFrame(PRICE, index=index, columns=SYMBOLS, dtype=float)
        for symbol in self.missing_symbols:
            closes.loc[index[-1], symbol] = float("nan")
        return closes

    def latest_prices(self) -> dict[str, LatestPrice]:
        return {symbol: LatestPrice(self.latest_price, NOW - self.price_age) for symbol in SYMBOLS}


@dataclass
class FixedStrategy:
    """Vise `weight` sur chaque symbole long, 0 sinon. Les poids peuvent être fixés à la main."""

    signals: dict[str, int]
    weight: float = 1 / len(SYMBOLS)
    warm_up: int = 3
    required_symbols: frozenset[str] = frozenset()
    weights: dict[str, float] | None = None
    seen_columns: list[list[str]] = field(default_factory=list)

    @property
    def max_weight(self) -> float:
        return self.weight

    def target_weights(self, closes: pd.DataFrame) -> pd.DataFrame:
        self.seen_columns.append([str(column) for column in closes.columns])
        values = self.weights or {
            symbol: self.weight if signal == LONG else 0.0
            for symbol, signal in self.signals.items()
        }
        return pd.DataFrame(
            {symbol: values[str(symbol)] for symbol in closes.columns}, index=closes.index
        )


def make_config(**risk_overrides: Any) -> StrategyConfig:
    base = load_strategy_config()
    return base.model_copy(
        update={
            "strategy": MovingAverageSettings(
                name="moving_average", params=MovingAverageParams(fast_window=2, slow_window=3)
            ),
            "data": base.data.model_copy(update={"symbols": SYMBOLS}),
            "risk": base.risk.model_copy(update=risk_overrides),
            "live": base.live.model_copy(
                update={"max_consecutive_errors": 2, "cancel_wait_seconds": 2}
            ),
        }
    )


@dataclass
class Harness:
    client: FakeClient
    market: FakeMarket
    strategy: FixedStrategy
    journal: Journal
    store: InMemoryRiskStateStore
    bot: TradingBot
    naps: list[float]

    def decisions(self) -> dict[str, dict[str, Any]]:
        return {row["symbol"]: row for row in self.journal.decisions()}


ALL_LONG = dict.fromkeys(SYMBOLS, LONG)
ALL_FLAT = dict.fromkeys(SYMBOLS, FLAT)


def make_harness(
    tmp_path: Path,
    signals: dict[str, int] | None = None,
    *,
    dry_run: bool = False,
    config: StrategyConfig | None = None,
    client: FakeClient | None = None,
    market: FakeMarket | None = None,
    state: RiskState | None = None,
) -> Harness:
    config = config or make_config()
    client = client or FakeClient()
    market = market or FakeMarket()
    strategy = FixedStrategy(signals or ALL_LONG)
    journal = Journal(tmp_path / "journal.db")
    store = InMemoryRiskStateStore(state or RiskState())
    naps: list[float] = []
    bot = TradingBot(
        config,
        strategy,
        Broker(client, config.api_retry),
        market,
        RiskManager(config.risk, store),
        journal,
        dry_run=dry_run,
        sleep=naps.append,
    )
    return Harness(client, market, strategy, journal, store, bot, naps)


def test_long_signals_become_orders_sized_from_equity(tmp_path: Path) -> None:
    harness = make_harness(tmp_path)
    harness.bot.tick(NOW)
    assert harness.client.submitted_symbols() == SYMBOLS
    assert {order.qty for order in harness.client.submitted} == {200.0}
    assert {order.side for order in harness.client.submitted} == {AlpacaOrderSide.BUY}
    assert {order.time_in_force for order in harness.client.submitted} == {TimeInForce.DAY}


def test_buying_power_is_not_used_for_sizing(tmp_path: Path) -> None:
    """Le compte a 400 000 $ de pouvoir d'achat : la taille se base sur les 100 000 $ de capital."""
    harness = make_harness(tmp_path)
    harness.bot.tick(NOW)
    assert sum(float(order.qty or 0) for order in harness.client.submitted) * PRICE == 100_000


def test_every_decision_is_journaled_with_its_inputs(tmp_path: Path) -> None:
    harness = make_harness(tmp_path, {**ALL_LONG, "B": FLAT})
    harness.bot.tick(NOW)

    rows = harness.decisions()

    assert set(rows) == set(SYMBOLS)
    assert rows["B"]["note"] == "hold"
    assert rows["B"]["signal"] == FLAT
    assert rows["A"]["risk_approved"] == 1
    assert rows["A"]["client_order_id"] == "bot-20261103-A-buy"
    assert rows["A"]["status"] == OrderStatus.ACCEPTED.value
    assert rows["A"]["reference_price"] == PRICE
    inputs = json.loads(rows["A"]["inputs"])
    assert inputs["signal_session"] == YESTERDAY.isoformat()
    assert inputs["last_close"] == PRICE
    assert inputs["equity"] == 100_000


def test_signal_comes_from_last_completed_session_not_today(tmp_path: Path) -> None:
    harness = make_harness(tmp_path)
    harness.bot.tick(NOW)
    inputs = json.loads(harness.decisions()["A"]["inputs"])
    assert inputs["signal_session"] == YESTERDAY.isoformat()
    assert inputs["bars"] == 4


@pytest.mark.scenario("order_rejected")
def test_risk_rejection_sends_nothing_and_records_why(tmp_path: Path) -> None:
    market = FakeMarket(price_age=timedelta(hours=2))
    harness = make_harness(tmp_path, market=market)
    harness.bot.tick(NOW)
    assert harness.client.submitted == []
    row = harness.decisions()["A"]
    assert row["risk_approved"] == 0
    assert row["risk_reason"] == "stale_data"
    assert row["client_order_id"] is None
    assert harness.journal.pending_orders() == []


def test_price_received_during_the_cycle_is_not_refused(tmp_path: Path) -> None:
    """Le cycle dure quelques secondes : une transaction récente suit le début de l'itération."""
    market = FakeMarket(price_age=timedelta(seconds=-20))
    harness = make_harness(tmp_path, market=market)
    harness.client.clock_time = NOW + timedelta(seconds=21)
    harness.bot.tick(NOW)
    assert harness.client.submitted_symbols() == SYMBOLS


def test_price_far_ahead_of_broker_clock_is_refused(tmp_path: Path) -> None:
    market = FakeMarket(price_age=timedelta(seconds=-20))
    harness = make_harness(tmp_path, market=market)
    harness.bot.tick(NOW)
    assert harness.client.submitted == []
    assert harness.decisions()["A"]["risk_reason"] == "stale_data"


def test_orders_of_a_cycle_count_towards_exposure(tmp_path: Path) -> None:
    """Sans suivi dans le cycle, chaque achat verrait un portefeuille vide et passerait."""
    config = make_config(max_total_exposure_pct=0.5)
    harness = make_harness(tmp_path, config=config)
    harness.bot.tick(NOW)
    assert harness.client.submitted_symbols() == ["A", "B"]
    assert harness.decisions()["C"]["risk_reason"] == "exposure_limit"


def test_order_limit_applies_across_a_cycle(tmp_path: Path) -> None:
    harness = make_harness(tmp_path, config=make_config(max_orders_per_day=3))
    harness.bot.tick(NOW)
    assert harness.client.submitted_symbols() == ["A", "B", "C"]
    assert harness.decisions()["D"]["risk_reason"] == "order_limit"


@pytest.mark.scenario("duplicate_order")
def test_second_tick_does_not_trade_again(tmp_path: Path) -> None:
    harness = make_harness(tmp_path)
    harness.bot.tick(NOW)
    harness.bot.tick(NOW + timedelta(minutes=1))
    assert len(harness.client.submitted) == len(SYMBOLS)
    assert len(harness.journal.decisions()) == len(SYMBOLS)


@pytest.mark.scenario("duplicate_order")
def test_restart_does_not_trade_again(tmp_path: Path) -> None:
    first = make_harness(tmp_path)
    first.bot.tick(NOW)
    second = make_harness(tmp_path, client=first.client)
    second.bot.tick(NOW + timedelta(minutes=5))
    assert len(first.client.submitted) == len(SYMBOLS)


@pytest.mark.scenario("duplicate_order")
def test_crash_mid_cycle_cannot_duplicate_orders(tmp_path: Path) -> None:
    """Le cycle n'est pas marqué terminé : la reprise renvoie les mêmes identifiants, refusés."""
    first = make_harness(tmp_path)
    first.bot.tick(NOW)
    with first.journal._connection:
        first.journal._connection.execute("DELETE FROM cycles")

    first.bot.tick(NOW + timedelta(minutes=1))

    assert len(first.client.submitted) == len(SYMBOLS)
    assert len(first.client.orders) == len(SYMBOLS)


def test_next_session_trades_again(tmp_path: Path) -> None:
    harness = make_harness(tmp_path)
    harness.bot.tick(NOW)
    assert harness.journal.is_cycle_completed(TODAY)
    assert not harness.journal.is_cycle_completed(TODAY + timedelta(days=1))


def test_market_closed_sends_nothing(tmp_path: Path) -> None:
    harness = make_harness(tmp_path)
    harness.client.is_open = False
    harness.bot.tick(NOW)
    assert harness.client.submitted == []
    assert harness.journal.decisions() == []
    assert not harness.journal.is_cycle_completed(TODAY)


def test_account_that_cannot_trade_sends_nothing(tmp_path: Path) -> None:
    harness = make_harness(tmp_path)
    harness.client.status = AccountStatus.ACCOUNT_UPDATED
    harness.bot.tick(NOW)
    assert harness.client.submitted == []
    assert not harness.journal.is_cycle_completed(TODAY)


def test_flat_signal_sells_existing_position(tmp_path: Path) -> None:
    harness = make_harness(tmp_path, ALL_FLAT)
    harness.client.positions = [make_position("A", 50.0)]
    harness.bot.tick(NOW)
    assert harness.client.submitted_symbols() == ["A"]
    assert harness.client.submitted[0].side is AlpacaOrderSide.SELL
    assert harness.client.submitted[0].qty == 50.0


def test_existing_long_position_is_not_rebalanced(tmp_path: Path) -> None:
    harness = make_harness(tmp_path, {**ALL_FLAT, "A": LONG})
    harness.client.positions = [make_position("A", 50.0)]
    harness.bot.tick(NOW)
    assert harness.client.submitted == []
    assert harness.decisions()["A"]["note"] == "hold"


def test_sells_are_sent_before_buys(tmp_path: Path) -> None:
    harness = make_harness(tmp_path, {**ALL_FLAT, "A": LONG, "E": FLAT})
    harness.client.positions = [make_position("E", 50.0)]
    harness.bot.tick(NOW)
    assert harness.client.submitted_symbols() == ["E", "A"]


@pytest.mark.scenario("missing_data")
def test_missing_close_does_not_sell_position_held(tmp_path: Path) -> None:
    """Un trou de données ne doit pas être confondu avec un signal de sortie."""
    market = FakeMarket(missing_symbols=["A"])
    harness = make_harness(tmp_path, ALL_LONG, market=market)
    harness.client.positions = [make_position("A", 50.0)]

    harness.bot.tick(NOW)

    assert "A" not in harness.client.submitted_symbols()
    row = harness.decisions()["A"]
    assert row["note"] == "missing_data"
    assert row["signal"] is None
    assert "A" not in harness.strategy.seen_columns[0]


@pytest.mark.scenario("missing_data")
def test_all_data_missing_still_completes_without_orders(tmp_path: Path) -> None:
    market = FakeMarket(missing_symbols=SYMBOLS)
    harness = make_harness(tmp_path, market=market)
    harness.bot.tick(NOW)
    assert harness.client.submitted == []
    assert harness.strategy.seen_columns == []


@pytest.mark.scenario("missing_data")
def test_too_few_sessions_raises_instead_of_trading(tmp_path: Path) -> None:
    harness = make_harness(tmp_path)
    harness.strategy.warm_up = 50
    with pytest.raises(DataFetchError, match="séances"):
        harness.bot.tick(NOW)
    assert harness.client.submitted == []


def test_open_order_skips_symbol(tmp_path: Path) -> None:
    harness = make_harness(tmp_path)
    harness.client.open_orders = [make_order("autre", "A", AlpacaOrderSide.BUY, 1.0)]
    harness.bot.tick(NOW)
    assert "A" not in harness.client.submitted_symbols()
    assert harness.decisions()["A"]["note"] == "open_order"


def test_open_order_counts_towards_exposure(tmp_path: Path) -> None:
    harness = make_harness(tmp_path, config=make_config(max_total_exposure_pct=0.5))
    harness.client.open_orders = [make_order("autre", "A", AlpacaOrderSide.BUY, 300.0)]
    harness.bot.tick(NOW)
    assert harness.client.submitted_symbols() == ["B"]


@pytest.mark.scenario("order_rejected")
def test_broker_rejection_is_journaled_and_cycle_continues(tmp_path: Path) -> None:
    harness = make_harness(tmp_path)
    harness.client.submit_error = api_error(403)
    harness.bot.tick(NOW)
    row = harness.decisions()["A"]
    assert row["status"] == REJECTED_BY_BROKER
    assert row["is_final"] == 1
    assert row["detail"] == "HTTP 403"
    assert len(harness.journal.decisions()) == len(SYMBOLS)
    assert harness.journal.is_cycle_completed(TODAY)


@pytest.mark.scenario("api_down")
def test_network_failure_leaves_order_unknown_then_finds_it(tmp_path: Path) -> None:
    """La réponse est perdue mais l'ordre est parti : le suivi le retrouve sans le renvoyer."""
    harness = make_harness(tmp_path, {**ALL_FLAT, "A": LONG})
    harness.client.submit_error = RequestsConnectionError("coupure")
    harness.client.submit_reaches_broker = True

    harness.bot.tick(NOW)

    assert harness.decisions()["A"]["status"] == STATUS_UNKNOWN
    assert len(harness.journal.pending_orders()) == 1

    harness.client.submit_error = None
    harness.bot.tick(NOW + timedelta(minutes=1))

    assert harness.decisions()["A"]["status"] == OrderStatus.ACCEPTED.value
    assert harness.client.submitted == []


def test_order_lost_before_reaching_broker_is_marked_not_found(tmp_path: Path) -> None:
    harness = make_harness(tmp_path, {**ALL_FLAT, "A": LONG})
    harness.client.submit_error = RequestsConnectionError("coupure")
    harness.bot.tick(NOW)
    harness.client.submit_error = None
    harness.bot.tick(NOW + timedelta(minutes=1))
    assert harness.decisions()["A"]["status"] == STATUS_NOT_FOUND
    assert harness.journal.pending_orders() == []


def test_fill_records_price_and_slippage(tmp_path: Path) -> None:
    harness = make_harness(tmp_path, {**ALL_FLAT, "A": LONG})
    harness.bot.tick(NOW)
    client_order_id = "bot-20261103-A-buy"
    harness.client.orders[client_order_id] = make_order(
        client_order_id, "A", AlpacaOrderSide.BUY, 200.0, OrderStatus.FILLED, 100.5
    )

    harness.bot.tick(NOW + timedelta(minutes=1))

    row = harness.decisions()["A"]
    assert row["status"] == OrderStatus.FILLED.value
    assert row["filled_price"] == 100.5
    assert row["filled_quantity"] == 200.0
    assert row["slippage_pct"] == pytest.approx(0.005)
    assert harness.journal.pending_orders() == []


def test_journal_failure_prevents_the_order(tmp_path: Path) -> None:
    harness = make_harness(tmp_path)
    harness.journal.close()
    with pytest.raises(sqlite3.ProgrammingError):
        harness.bot.tick(NOW)
    assert harness.client.submitted == []


@pytest.mark.scenario("crash")
def test_daily_loss_cancels_orders_closes_positions_and_stops(tmp_path: Path) -> None:
    harness = make_harness(tmp_path)
    harness.client.equity = "97000"
    harness.client.positions = [make_position("A", 50.0), make_position("B", -10.0)]
    harness.client.open_orders = [make_order("autre", "C", AlpacaOrderSide.BUY, 1.0)]

    code = harness.bot.run(once=False, now=lambda: NOW)

    assert code == EXIT_HALTED
    assert harness.client.cancel_calls == 1
    sides = {str(order.symbol): order.side for order in harness.client.submitted}
    assert sides == {"A": AlpacaOrderSide.SELL, "B": AlpacaOrderSide.BUY}
    assert harness.store.load().halt_reason is HaltReason.DAILY_LOSS
    assert harness.decisions()["A"]["note"] == "kill_switch"
    assert harness.decisions()["A"]["client_order_id"].startswith("kill-")


@pytest.mark.scenario("opening_gap")
def test_opening_gap_refuses_orders_on_aberrant_price(tmp_path: Path) -> None:
    """Un gap de -25 % dépasse la variation tolérée : aucun achat sur un prix douteux."""
    harness = make_harness(tmp_path, market=FakeMarket(latest_price=PRICE * 0.75))
    harness.bot.tick(NOW)
    assert harness.client.submitted == []
    assert {row["risk_reason"] for row in harness.decisions().values()} == {"aberrant_price"}


@pytest.mark.scenario("opening_gap")
def test_opening_gap_loss_triggers_kill_switch_before_any_order(tmp_path: Path) -> None:
    """Un gap de -10 % sur une position fait perdre 4 % : seul l'ordre de fermeture part."""
    harness = make_harness(tmp_path, market=FakeMarket(latest_price=PRICE * 0.9))
    harness.client.equity = "96000"
    harness.client.positions = [make_position("A", 400.0)]

    code = harness.bot.run(once=False, now=lambda: NOW)

    assert code == EXIT_HALTED
    assert harness.client.submitted_symbols() == ["A"]
    assert harness.client.submitted[0].side == AlpacaOrderSide.SELL
    assert harness.decisions()["A"]["note"] == "kill_switch"


@pytest.mark.scenario("crash")
def test_kill_switch_closes_even_with_stale_data_or_order_limit(tmp_path: Path) -> None:
    config = make_config(max_orders_per_day=1)
    market = FakeMarket(price_age=timedelta(days=3))
    state = RiskState(order_date=TODAY, orders_count=1)
    harness = make_harness(tmp_path, config=config, market=market, state=state)
    harness.client.equity = "97000"
    harness.client.positions = [make_position("A", 50.0)]
    harness.bot.tick(NOW)
    assert harness.client.submitted_symbols() == ["A"]


@pytest.mark.scenario("crash")
def test_kill_switch_works_when_market_is_closed(tmp_path: Path) -> None:
    harness = make_harness(tmp_path)
    harness.client.is_open = False
    harness.client.equity = "97000"
    harness.client.positions = [make_position("A", 50.0)]
    harness.bot.tick(NOW)
    assert harness.client.submitted_symbols() == ["A"]


def test_kill_switch_already_active_blocks_trading_after_restart(tmp_path: Path) -> None:
    state = RiskState(peak_equity=100_000.0, halt_reason=HaltReason.DAILY_LOSS, halted_at=NOW)
    harness = make_harness(tmp_path, state=state)
    assert harness.bot.run(once=False, now=lambda: NOW) == EXIT_HALTED
    assert harness.client.submitted == []
    assert harness.journal.decisions() == []


@pytest.mark.scenario("crash")
@pytest.mark.risk_limit("halted")
def test_kill_switch_triggered_mid_cycle_stops_buying(tmp_path: Path) -> None:
    """Le kill switch est évalué à nouveau par `review`, avec le capital lu pour le cycle."""
    harness = make_harness(tmp_path)

    original = harness.client.get_account
    calls = {"count": 0}

    def losing_after_first_read() -> TradeAccount | RawData:
        calls["count"] += 1
        harness.client.equity = "100000" if calls["count"] == 1 else "97000"
        return original()

    harness.client.get_account = losing_after_first_read  # type: ignore[method-assign]
    harness.bot.tick(NOW)

    assert harness.client.submitted == []
    assert harness.decisions()["A"]["risk_reason"] == "halted"
    assert not harness.journal.is_cycle_completed(TODAY)


def test_waits_for_cancellations_before_closing(tmp_path: Path) -> None:
    harness = make_harness(tmp_path)
    harness.client.equity = "97000"
    stuck = make_order("autre", "C", AlpacaOrderSide.BUY, 1.0)
    harness.client.open_orders = [stuck]
    harness.client.cancel_orders = lambda: []  # type: ignore[method-assign]
    harness.bot.tick(NOW)
    assert harness.naps == [1.0, 1.0]


def test_dry_run_sends_nothing_and_keeps_real_state(tmp_path: Path) -> None:
    harness = make_harness(tmp_path, dry_run=True)
    harness.bot.tick(NOW)
    assert harness.client.submitted == []
    assert harness.client.orders == {}
    assert harness.decisions()["A"]["status"] == STATUS_DRY_RUN
    assert harness.journal.pending_orders() == []


def test_dry_run_works_when_market_is_closed(tmp_path: Path) -> None:
    harness = make_harness(tmp_path, dry_run=True)
    harness.client.is_open = False
    harness.bot.tick(NOW)
    assert len(harness.journal.decisions()) == len(SYMBOLS)


def test_dry_run_can_be_repeated_in_a_session(tmp_path: Path) -> None:
    harness = make_harness(tmp_path, dry_run=True, config=make_config(max_orders_per_day=50))
    harness.bot.tick(NOW)
    harness.bot.tick(NOW + timedelta(minutes=1))
    assert len(harness.journal.decisions()) == 2 * len(SYMBOLS)


def test_dry_run_kill_switch_does_not_touch_the_account(tmp_path: Path) -> None:
    harness = make_harness(tmp_path, dry_run=True)
    harness.client.equity = "97000"
    harness.client.positions = [make_position("A", 50.0)]
    assert harness.bot.run(once=True, now=lambda: NOW) == EXIT_HALTED
    assert harness.client.cancel_calls == 0
    assert harness.client.submitted == []
    assert harness.decisions()["A"]["status"] == STATUS_DRY_RUN


@pytest.mark.scenario("api_down")
def test_run_stops_after_consecutive_failures(tmp_path: Path) -> None:
    harness = make_harness(tmp_path)
    harness.client.clock_error = RequestsConnectionError("panne")
    assert harness.bot.run(once=False, now=lambda: NOW) == EXIT_ERROR
    assert len(harness.naps) == 1


@pytest.mark.scenario("api_down")
def test_run_recovers_from_a_single_failure(tmp_path: Path) -> None:
    harness = make_harness(tmp_path)
    harness.client.clock_error = api_error(500)
    instants = iter([NOW, NOW + timedelta(minutes=1)])

    def now() -> datetime:
        instant = next(instants)
        if instant != NOW:
            harness.client.clock_error = None
            harness.client.is_open = False
        return instant

    with pytest.raises(StopIteration):
        harness.bot.run(once=False, now=now)
    assert harness.naps == [60.0, 60.0]


def test_run_once_returns_after_one_tick(tmp_path: Path) -> None:
    harness = make_harness(tmp_path)
    assert harness.bot.run(once=True, now=lambda: NOW) == EXIT_OK
    assert harness.naps == []


def test_run_once_reports_failure(tmp_path: Path) -> None:
    market = FakeMarket(closes_error=DataFetchError("panne"))
    harness = make_harness(tmp_path, market=market)
    assert harness.bot.run(once=True, now=lambda: NOW) == EXIT_ERROR


def test_weight_above_position_limit_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Poids"):
        make_harness(tmp_path, config=make_config(max_position_pct=0.1))


# Stratégie rééquilibrée chaque mois


def rebalanced_config() -> StrategyConfig:
    base = make_config()
    return base.model_copy(update={"strategy": base.candidates[0]})


def test_rebalanced_strategy_trades_on_first_session_of_month(tmp_path: Path) -> None:
    harness = make_harness(tmp_path, config=rebalanced_config())
    harness.journal.complete_cycle(date(2026, 10, 30), NOW)
    harness.bot.tick(NOW)
    assert harness.client.submitted_symbols() == SYMBOLS


def test_rebalanced_strategy_waits_for_next_month(tmp_path: Path) -> None:
    """Comme le backtest, la stratégie mensuelle ne décide qu'une fois par mois."""
    harness = make_harness(tmp_path, config=rebalanced_config())
    harness.journal.complete_cycle(YESTERDAY, NOW)
    harness.bot.tick(NOW)
    assert harness.client.submitted == []
    assert harness.journal.decisions() == []
    assert not harness.journal.is_cycle_completed(TODAY)


def test_dry_run_of_rebalanced_strategy_ignores_the_month(tmp_path: Path) -> None:
    harness = make_harness(tmp_path, config=rebalanced_config(), dry_run=True)
    harness.journal.complete_cycle(YESTERDAY, NOW)
    harness.bot.tick(NOW)
    assert len(harness.journal.decisions()) == len(SYMBOLS)


def test_rebalance_brings_position_back_to_its_weight(tmp_path: Path) -> None:
    """A vaut 10 % du capital pour un poids visé de 20 % : achat de la différence."""
    harness = make_harness(tmp_path, {**ALL_FLAT, "A": LONG}, config=rebalanced_config())
    harness.client.positions = [make_position("A", 100.0)]
    harness.bot.tick(NOW)
    assert harness.client.submitted_symbols() == ["A"]
    assert harness.client.submitted[0].qty == 100.0


def test_every_decision_records_its_target_weight(tmp_path: Path) -> None:
    harness = make_harness(tmp_path, {**ALL_LONG, "B": FLAT})
    harness.bot.tick(NOW)
    rows = harness.decisions()
    assert json.loads(rows["A"]["inputs"])["target_weight"] == pytest.approx(0.2)
    assert json.loads(rows["B"]["inputs"])["target_weight"] == 0.0
    assert rows["B"]["signal"] == FLAT


@pytest.mark.scenario("missing_data")
def test_missing_required_symbol_blocks_every_order(tmp_path: Path) -> None:
    """Sans le symbole du filtre de régime, aucun poids n'est fiable : rien n'est vendu."""
    market = FakeMarket(missing_symbols=["A"])
    harness = make_harness(tmp_path, ALL_FLAT, market=market)
    harness.strategy.required_symbols = frozenset({"A"})
    harness.client.positions = [make_position("B", 50.0)]

    harness.bot.tick(NOW)

    assert harness.client.submitted == []
    assert {row["note"] for row in harness.decisions().values()} == {"missing_data"}
    assert harness.strategy.seen_columns == []


def test_main_reports_invalid_config(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def fail() -> StrategyConfig:
        raise ConfigError("invalide")

    monkeypatch.setattr(main_module, "load_strategy_config", fail)
    assert main_module.main(["--once"]) == EXIT_ERROR
    assert "bot_failed" in capsys.readouterr().out


def wire_main(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, client: FakeClient) -> None:
    monkeypatch.setattr(main_module, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(main_module, "load_strategy_config", make_config)
    monkeypatch.setattr(main_module, "load_settings", lambda: None)
    monkeypatch.setattr(main_module, "create_trading_client", lambda _: client)
    monkeypatch.setattr(main_module, "create_data_client", lambda _: object())
    monkeypatch.setattr(main_module, "AlpacaMarketData", lambda *_: FakeMarket())


def test_main_once_uses_real_state_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    client = FakeClient()
    client.is_open = False
    wire_main(monkeypatch, tmp_path, client)
    assert main_module.main(["--once"]) == EXIT_OK
    assert (tmp_path / "data" / "risk_state.json").exists()
    assert (tmp_path / "data" / "journal.db").exists()


def test_main_dry_run_never_writes_real_state(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    client = FakeClient()
    wire_main(monkeypatch, tmp_path, client)
    assert main_module.main(["--dry-run"]) in {EXIT_OK, EXIT_ERROR}
    assert client.submitted == []
    assert not (tmp_path / "data" / "risk_state.json").exists()
    assert not (tmp_path / "data" / "journal.db").exists()
    assert (tmp_path / "data" / "journal.dry-run.db").exists()


def test_main_resets_kill_switch(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    client = FakeClient()
    wire_main(monkeypatch, tmp_path, client)
    state_path = tmp_path / make_config().risk_state_file
    store = JsonRiskStateStore(state_path)
    store.save(RiskState(peak_equity=200_000.0, halt_reason=HaltReason.MAX_DRAWDOWN, halted_at=NOW))

    assert main_module.main(["--reset-kill-switch"]) == EXIT_OK

    state = store.load()
    assert state.halt_reason is None
    assert state.peak_equity == 100_000.0
    assert client.submitted == []


def test_main_refuses_corrupt_risk_state(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    client = FakeClient()
    wire_main(monkeypatch, tmp_path, client)
    state_path = tmp_path / make_config().risk_state_file
    state_path.parent.mkdir(parents=True)
    state_path.write_text("corrompu")
    assert main_module.main(["--once"]) == EXIT_ERROR
    assert client.submitted == []


@pytest.fixture
def restore_logging() -> Iterator[None]:
    """`main` écrit ses logs dans un fichier : la configuration console est rétablie ensuite."""
    yield
    configure_logging()


def test_main_writes_a_daily_log_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, restore_logging: None
) -> None:
    client = FakeClient()
    client.is_open = False
    wire_main(monkeypatch, tmp_path, client)
    assert main_module.main(["--once"]) == EXIT_OK
    files = list((tmp_path / make_config().live.log_dir).glob("bot-*.jsonl"))
    assert len(files) == 1
    events = [json.loads(line)["event"] for line in files[0].read_text("utf-8").splitlines()]
    assert "market_closed" in events


def test_main_until_close_stops_without_session_today(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, restore_logging: None
) -> None:
    client = FakeClient()
    client.is_open = False
    client.next_open = NOW + timedelta(days=3)
    wire_main(monkeypatch, tmp_path, client)
    assert main_module.main(["--until-close", "--log-dir", "autres"]) == EXIT_OK
    assert list((tmp_path / "autres").glob("bot-*.jsonl"))
    assert client.submitted == []


OPENING = datetime(2026, 11, 3, 14, 30, tzinfo=UTC)


def test_until_close_waits_for_open_trades_then_stops_after_close(tmp_path: Path) -> None:
    harness = make_harness(tmp_path)
    client = harness.client
    # Avant l'ouverture, puis pendant la séance, puis après la clôture.
    states = iter([(False, OPENING), (True, OPENING), (False, OPENING + timedelta(days=1))])

    def now() -> datetime:
        client.is_open, client.next_open = next(states)
        return NOW

    assert harness.bot.run(once=False, now=now, until_close=True) == EXIT_OK
    assert client.submitted_symbols() == SYMBOLS
    assert harness.naps == [60.0, 60.0]


def test_until_close_stops_at_once_on_a_holiday(tmp_path: Path) -> None:
    harness = make_harness(tmp_path)
    harness.client.is_open = False
    harness.client.next_open = NOW + timedelta(days=1)
    assert harness.bot.run(once=False, now=lambda: NOW, until_close=True) == EXIT_OK
    assert harness.client.submitted == []
    assert harness.naps == []


def test_until_close_keeps_trying_after_an_api_failure(tmp_path: Path) -> None:
    """Une panne ne dit rien de l'horloge : le bot ne doit pas croire la séance terminée."""
    harness = make_harness(tmp_path)
    harness.client.clock_error = api_error(500)
    instants = iter([NOW, NOW])

    def now() -> datetime:
        instant = next(instants)
        if harness.naps:
            harness.client.clock_error = None
            harness.client.is_open = False
            harness.client.next_open = NOW + timedelta(days=1)
        return instant

    assert harness.bot.run(once=False, now=now, until_close=True) == EXIT_OK
    assert harness.naps == [60.0]


def test_until_close_still_reports_the_kill_switch(tmp_path: Path) -> None:
    harness = make_harness(tmp_path)
    harness.client.equity = "97000"
    harness.client.is_open = False
    harness.client.next_open = NOW + timedelta(days=1)
    assert harness.bot.run(once=False, now=lambda: NOW, until_close=True) == EXIT_HALTED


def test_equity_is_recorded_once_per_open_session(tmp_path: Path) -> None:
    harness = make_harness(tmp_path)
    harness.bot.tick(NOW)
    harness.client.equity = "101000"
    harness.bot.tick(NOW + timedelta(minutes=1))
    assert harness.journal.equity_history() == [
        EquityPoint(TODAY, 101_000.0, 101_000.0, NOW + timedelta(minutes=1))
    ]


def test_equity_is_not_recorded_when_market_is_closed(tmp_path: Path) -> None:
    harness = make_harness(tmp_path)
    harness.client.is_open = False
    harness.bot.tick(NOW)
    assert harness.journal.equity_history() == []
