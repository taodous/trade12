from collections.abc import Callable
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pytest
from alpaca.common.exceptions import APIError
from alpaca.trading.enums import OrderSide as AlpacaOrderSide
from alpaca.trading.enums import OrderStatus

from bot.config import RebalanceConfig, RetryConfig
from bot.execution.broker import (
    REJECTED_BY_BROKER,
    Broker,
    BrokerClient,
    BrokerError,
    OrderReport,
    make_client_order_id,
)
from bot.execution.planner import PlanItem, PlanNote, plan_orders
from bot.journal.store import STATUS_NOT_FOUND, Decision, Journal, observed_slippage
from bot.risk.models import OrderSide, PortfolioState, Position
from bot.risk.state import InMemoryRiskStateStore, RiskState
from bot.strategy.base import FLAT, LONG
from tests.test_main import FakeClient, api_error, make_order, make_position

SESSION = date(2026, 11, 2)
NOW = datetime(2026, 11, 3, 15, 0, tzinfo=UTC)
RETRY = RetryConfig(max_attempts=2, base_delay_seconds=0.001, max_delay_seconds=0.001)
PORTFOLIO = PortfolioState(equity=100_000.0, day_start_equity=100_000.0)


MONTHLY = RebalanceConfig(frequency="monthly", band_pct=0.02)


def plan(
    signals: dict[str, int | None],
    portfolio: PortfolioState = PORTFOLIO,
    open_orders: set[str] | None = None,
    prices: dict[str, float] | None = None,
    rebalance: RebalanceConfig | None = None,
) -> dict[str, Any]:
    """Planifie avec un poids de 10 % par signal long."""
    targets = {
        symbol: None if signal is None else (0.1 if signal == LONG else 0.0)
        for symbol, signal in signals.items()
    }
    items = plan_orders(
        targets,
        portfolio,
        open_orders or set(),
        {"A": 100.0, "B": 100.0} if prices is None else prices,
        SESSION,
        rebalance,
    )
    return {item.symbol: item for item in items}


def test_plan_buys_whole_shares_rounded_down() -> None:
    order = plan({"A": LONG}, prices={"A": 333.0})["A"].order
    assert order.side is OrderSide.BUY
    assert order.quantity == 30.0
    assert order.signal_session == SESSION


def test_plan_never_exceeds_target_weight_through_rounding() -> None:
    order = plan({"A": LONG}, prices={"A": 7.77})["A"].order
    assert order.quantity * 7.77 <= 10_000


def test_plan_holds_when_flat_without_position() -> None:
    assert plan({"A": FLAT})["A"].note is PlanNote.HOLD


def test_plan_holds_long_position_without_rebalancing() -> None:
    portfolio = PortfolioState(100_000.0, 100_000.0, {"A": Position(1.0, 100.0)})
    item = plan({"A": LONG}, portfolio)["A"]
    assert item.note is PlanNote.HOLD
    assert item.order is None


def test_plan_sells_whole_position_on_flat_signal() -> None:
    portfolio = PortfolioState(100_000.0, 100_000.0, {"A": Position(12.5, 1250.0)})
    order = plan({"A": FLAT}, portfolio)["A"].order
    assert order.side is OrderSide.SELL
    assert order.quantity == 12.5


def test_plan_covers_short_position_on_flat_signal() -> None:
    portfolio = PortfolioState(100_000.0, 100_000.0, {"A": Position(-5.0, -500.0)})
    order = plan({"A": FLAT}, portfolio)["A"].order
    assert order.side is OrderSide.BUY
    assert order.quantity == 5.0


@pytest.mark.scenario("missing_data")
def test_plan_missing_signal_never_produces_an_order() -> None:
    portfolio = PortfolioState(100_000.0, 100_000.0, {"A": Position(10.0, 1000.0)})
    item = plan({"A": None}, portfolio)["A"]
    assert item.note is PlanNote.MISSING_DATA
    assert item.order is None


def test_plan_skips_symbol_with_open_order() -> None:
    assert plan({"A": LONG}, open_orders={"A"})["A"].note is PlanNote.OPEN_ORDER


@pytest.mark.parametrize("price", [float("nan"), 0.0, -1.0, float("inf")])
def test_plan_refuses_to_size_with_invalid_price(price: float) -> None:
    assert plan({"A": LONG}, prices={"A": price})["A"].note is PlanNote.NO_PRICE


def test_plan_refuses_to_size_without_price() -> None:
    assert plan({"A": LONG}, prices={})["A"].note is PlanNote.NO_PRICE


def test_plan_sell_does_not_need_a_price() -> None:
    portfolio = PortfolioState(100_000.0, 100_000.0, {"A": Position(10.0, 1000.0)})
    assert plan({"A": FLAT}, portfolio, prices={})["A"].order is not None


def test_plan_skips_position_too_small() -> None:
    assert plan({"A": LONG}, prices={"A": 50_000.0})["A"].note is PlanNote.TOO_SMALL


def test_plan_orders_sells_first() -> None:
    portfolio = PortfolioState(100_000.0, 100_000.0, {"B": Position(10.0, 1000.0)})
    items = plan_orders(
        {"A": 0.1, "B": 0.0}, portfolio, set(), {"A": 100.0, "B": 100.0}, SESSION, None
    )
    assert [item.symbol for item in items] == ["B", "A"]


def holding(quantity: float) -> PortfolioState:
    return PortfolioState(100_000.0, 100_000.0, {"A": Position(quantity, quantity * 100.0)})


def test_rebalance_buys_the_missing_shares() -> None:
    order = plan({"A": LONG}, holding(50.0), rebalance=MONTHLY)["A"].order
    assert order.side is OrderSide.BUY
    assert order.quantity == 50.0


def test_rebalance_sells_the_excess_shares() -> None:
    order = plan({"A": LONG}, holding(150.0), rebalance=MONTHLY)["A"].order
    assert order.side is OrderSide.SELL
    assert order.quantity == 50.0


def test_rebalance_ignores_a_gap_within_the_band() -> None:
    """95 parts au lieu de 100 : 500 $ d'écart, sous la bande de 2 000 $."""
    item = plan({"A": LONG}, holding(95.0), rebalance=MONTHLY)["A"]
    assert item.note is PlanNote.HOLD
    assert item.order is None


def test_rebalance_closes_a_zero_weight_even_within_the_band() -> None:
    order = plan({"A": FLAT}, holding(5.0), rebalance=MONTHLY)["A"].order
    assert order.side is OrderSide.SELL
    assert order.quantity == 5.0


def test_rebalance_needs_a_price_to_resize() -> None:
    assert plan({"A": LONG}, holding(50.0), prices={}, rebalance=MONTHLY)["A"].note is (
        PlanNote.NO_PRICE
    )


def test_plan_item_signal_follows_its_target() -> None:
    assert PlanItem("A", None, PlanNote.MISSING_DATA).signal is None
    assert PlanItem("A", 0.0, PlanNote.HOLD).signal == FLAT
    assert PlanItem("A", 0.15, PlanNote.HOLD).signal == LONG


def make_broker(client: FakeClient) -> Broker:
    typed_client: BrokerClient = client
    return Broker(typed_client, RETRY)


def test_client_order_id_is_stable_and_within_alpaca_limit() -> None:
    first = make_client_order_id("bot", SESSION, "SPY", OrderSide.BUY)
    assert first == make_client_order_id("bot", SESSION, "SPY", OrderSide.BUY)
    assert first != make_client_order_id("bot", SESSION, "SPY", OrderSide.SELL)
    assert first != make_client_order_id("bot", date(2026, 11, 3), "SPY", OrderSide.BUY)
    assert len(first) <= 48


def test_broker_reads_clock_and_account() -> None:
    client = FakeClient()
    client.equity = "90000"
    client.last_equity = "91000"
    broker = make_broker(client)
    assert broker.clock().is_open is True
    summary = broker.account_summary()
    assert summary.equity == 90_000
    assert summary.day_start_equity == 91_000
    assert summary.positions == {}


def test_broker_positions_are_signed() -> None:
    client = FakeClient()
    client.positions = [make_position("A", 10.0), make_position("B", -5.0)]
    positions = make_broker(client).positions()
    assert positions["A"].quantity == 10.0
    assert positions["B"].quantity == -5.0
    assert positions["B"].market_value == -500.0


def test_broker_open_orders_keep_only_remaining_quantity() -> None:
    client = FakeClient()
    partial = make_order("x", "A", AlpacaOrderSide.SELL, 10.0).model_copy(
        update={"filled_qty": "4"}
    )
    client.open_orders = [partial]
    assert make_broker(client).open_orders()[0].signed_quantity == -6.0


def test_broker_portfolio_values_open_orders() -> None:
    client = FakeClient()
    client.positions = [make_position("A", 10.0)]
    client.open_orders = [
        make_order("x", "A", AlpacaOrderSide.SELL, 2.0),
        make_order("y", "B", AlpacaOrderSide.BUY, 3.0),
    ]
    broker = make_broker(client)
    portfolio, symbols = broker.portfolio(broker.account(), {"A": 50.0, "B": 20.0})
    assert symbols == {"A", "B"}
    assert portfolio.positions["A"].pending_notional == -100.0
    assert portfolio.positions["B"].pending_notional == 60.0
    assert portfolio.positions["B"].quantity == 0.0


def test_broker_open_order_without_price_invalidates_portfolio() -> None:
    client = FakeClient()
    client.open_orders = [make_order("y", "B", AlpacaOrderSide.BUY, 3.0)]
    broker = make_broker(client)
    portfolio, _ = broker.portfolio(broker.account(), {})
    assert portfolio.positions["B"].pending_notional != portfolio.positions["B"].pending_notional


def test_submit_returns_broker_report() -> None:
    report = make_broker(FakeClient()).submit("A", OrderSide.BUY, 2.0, "id-1")
    assert report.client_order_id == "id-1"
    assert report.status == OrderStatus.ACCEPTED.value
    assert report.is_final is False
    assert report.broker_order_id is not None


@pytest.mark.scenario("duplicate_order")
def test_submit_duplicate_returns_existing_order_without_second_order() -> None:
    client = FakeClient()
    broker = make_broker(client)
    first = broker.submit("A", OrderSide.BUY, 2.0, "id-1")
    second = broker.submit("A", OrderSide.BUY, 2.0, "id-1")
    assert second.broker_order_id == first.broker_order_id
    assert len(client.submitted) == 1


@pytest.mark.scenario("order_rejected")
def test_submit_refusal_is_reported_not_raised() -> None:
    client = FakeClient()
    client.submit_error = api_error(403)
    report = make_broker(client).submit("A", OrderSide.BUY, 2.0, "id-1")
    assert report.status == REJECTED_BY_BROKER
    assert report.is_final is True
    assert report.detail == "HTTP 403"


@pytest.mark.parametrize("status_code", [500, 503])
def test_submit_server_error_is_raised_as_unknown(status_code: int) -> None:
    client = FakeClient()
    client.submit_error = api_error(status_code)
    with pytest.raises(APIError):
        make_broker(client).submit("A", OrderSide.BUY, 2.0, "id-1")


@pytest.mark.scenario("api_down")
def test_submit_retries_rate_limit_with_same_identifier() -> None:
    client = FakeClient()
    calls: list[str] = []
    original = client.submit_order

    def limited(order_data: Any) -> Any:
        calls.append(order_data.client_order_id)
        if len(calls) == 1:
            raise api_error(429)
        return original(order_data)

    client.submit_order = limited  # type: ignore[method-assign]
    report = make_broker(client).submit("A", OrderSide.BUY, 2.0, "id-1")
    assert calls == ["id-1", "id-1"]
    assert report.status == OrderStatus.ACCEPTED.value


def test_find_order_unknown_returns_none() -> None:
    assert make_broker(FakeClient()).find_order("absent") is None


def test_find_order_propagates_other_errors() -> None:
    client = FakeClient()

    def fail(client_id: str) -> Any:
        raise api_error(500)

    client.get_order_by_client_id = fail  # type: ignore[method-assign]
    with pytest.raises(APIError):
        make_broker(client).find_order("id")


def test_find_order_reports_fill() -> None:
    client = FakeClient()
    client.orders["id"] = make_order("id", "A", AlpacaOrderSide.BUY, 2.0, OrderStatus.FILLED, 101.0)
    report = make_broker(client).find_order("id")
    assert report is not None
    assert report.is_final is True
    assert report.filled_price == 101.0
    assert report.filled_quantity == 2.0


@pytest.mark.parametrize(
    "method",
    ["get_clock", "get_all_positions", "get_orders", "submit_order", "get_order_by_client_id"],
)
def test_broker_rejects_raw_responses(method: str) -> None:
    client = FakeClient()
    setattr(client, method, lambda *_args, **_kwargs: {"raw": True})
    broker = make_broker(client)
    calls: dict[str, Callable[[], object]] = {
        "get_clock": broker.clock,
        "get_all_positions": broker.positions,
        "get_orders": broker.open_orders,
        "submit_order": lambda: broker.submit("A", OrderSide.BUY, 1.0, "id"),
        "get_order_by_client_id": lambda: broker.find_order("id"),
    }
    with pytest.raises(BrokerError):
        calls[method]()


def test_cancel_all_calls_broker() -> None:
    client = FakeClient()
    make_broker(client).cancel_all()
    assert client.cancel_calls == 1


@pytest.mark.parametrize(
    ("side", "fill", "expected"),
    [
        (OrderSide.BUY, 101.0, 0.01),
        (OrderSide.BUY, 99.0, -0.01),
        (OrderSide.SELL, 99.0, 0.01),
        (OrderSide.SELL, 101.0, -0.01),
    ],
)
def test_slippage_is_positive_when_unfavourable(
    side: OrderSide, fill: float, expected: float
) -> None:
    assert observed_slippage(side, 100.0, fill) == pytest.approx(expected)


def decision(**overrides: Any) -> Decision:
    values: dict[str, Any] = {
        "timestamp": NOW,
        "session": SESSION,
        "symbol": "A",
        "signal": LONG,
        "note": "order",
        "inputs": {"last_close": 100.0, "when": NOW},
        "side": OrderSide.BUY,
        "quantity": 2.0,
        "risk_approved": True,
        "client_order_id": "id-1",
        "reference_price": 100.0,
        **overrides,
    }
    return Decision(**values)


def test_journal_persists_across_reopen(tmp_path: Path) -> None:
    path = tmp_path / "sous" / "journal.db"
    journal = Journal(path)
    journal.record_decision(decision())
    journal.complete_cycle(SESSION, NOW)
    journal.close()

    reopened = Journal(path)
    assert reopened.decisions()[0]["symbol"] == "A"
    assert reopened.is_cycle_completed(SESSION)
    reopened.close()


def test_approved_order_stays_pending_until_final(tmp_path: Path) -> None:
    journal = Journal(tmp_path / "j.db")
    decision_id = journal.record_decision(decision())
    assert [order.client_order_id for order in journal.pending_orders()] == ["id-1"]

    journal.record_report(decision_id, OrderReport("id-1", "b", "accepted", False))
    assert len(journal.pending_orders()) == 1

    journal.record_report(decision_id, OrderReport("id-1", "b", "filled", True, 2.0, 100.2))
    assert journal.pending_orders() == []
    row = journal.decisions()[0]
    assert row["slippage_pct"] == pytest.approx(0.002)
    assert row["broker_order_id"] == "b"


def test_rejected_decision_is_not_pending(tmp_path: Path) -> None:
    journal = Journal(tmp_path / "j.db")
    journal.record_decision(
        decision(risk_approved=False, risk_reason="stale_data", client_order_id=None)
    )
    assert journal.pending_orders() == []
    assert journal.decisions()[0]["risk_reason"] == "stale_data"


def test_decision_without_order_is_recorded(tmp_path: Path) -> None:
    journal = Journal(tmp_path / "j.db")
    journal.record_decision(
        decision(
            side=None,
            quantity=None,
            risk_approved=None,
            client_order_id=None,
            reference_price=None,
            note="hold",
        )
    )
    row = journal.decisions()[0]
    assert row["note"] == "hold"
    assert row["risk_approved"] is None


def test_record_status_can_finish_an_order(tmp_path: Path) -> None:
    journal = Journal(tmp_path / "j.db")
    decision_id = journal.record_decision(decision())
    journal.record_status(decision_id, STATUS_NOT_FOUND, is_final=True, detail="perdu")
    assert journal.pending_orders() == []
    assert journal.decisions()[0]["detail"] == "perdu"


def test_record_report_for_unknown_decision_fails(tmp_path: Path) -> None:
    with pytest.raises(KeyError):
        Journal(tmp_path / "j.db").record_report(9, OrderReport("x", "y", "filled", True))


def test_cycle_is_per_session(tmp_path: Path) -> None:
    journal = Journal(tmp_path / "j.db")
    assert not journal.is_cycle_completed(SESSION)
    journal.complete_cycle(SESSION, NOW)
    journal.complete_cycle(SESSION, NOW)
    assert journal.is_cycle_completed(SESSION)
    assert not journal.is_cycle_completed(date(2026, 11, 3))


def test_in_memory_state_store() -> None:
    store = InMemoryRiskStateStore(RiskState(orders_count=1))
    assert store.load().orders_count == 1
    store.save(RiskState(orders_count=2))
    assert store.load().orders_count == 2
