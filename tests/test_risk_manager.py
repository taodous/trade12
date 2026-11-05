from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pytest

from bot.config import RiskLimits, load_strategy_config
from bot.risk.checks import closing_orders
from bot.risk.manager import RiskManager
from bot.risk.models import (
    HaltReason,
    OrderSide,
    PortfolioState,
    Position,
    PriceSnapshot,
    ProposedOrder,
    RejectReason,
    RiskContext,
)
from bot.risk.state import JsonRiskStateStore, RiskState, RiskStateError

NOW = datetime(2026, 11, 3, 15, 0, tzinfo=UTC)
LAST_SESSION = date(2026, 11, 2)
CONTEXT = RiskContext(now=NOW, session_date=date(2026, 11, 3), last_completed_session=LAST_SESSION)
NEXT_DAY = RiskContext(
    now=NOW + timedelta(days=1),
    session_date=date(2026, 11, 4),
    last_completed_session=date(2026, 11, 3),
)
LIMITS = RiskLimits(
    max_position_pct=0.2,
    max_total_exposure_pct=1.0,
    max_daily_loss_pct=0.02,
    max_drawdown_pct=0.1,
    max_orders_per_day=2,
    max_data_age_seconds=900,
    max_clock_skew_seconds=5,
    max_price_change_pct=0.2,
    allow_short=False,
    allow_leverage=False,
)
ORDER = ProposedOrder("SPY", OrderSide.BUY, 10.0, LAST_SESSION)
MARKET = {"SPY": PriceSnapshot(100.0, 100.0, NOW)}
NEXT_DAY_ORDER = replace(ORDER, signal_session=NEXT_DAY.last_completed_session)
NEXT_DAY_MARKET = {"SPY": PriceSnapshot(100.0, 100.0, NEXT_DAY.now)}


def portfolio(equity: float = 100_000.0, day_start: float | None = None) -> PortfolioState:
    return PortfolioState(equity=equity, day_start_equity=day_start or equity)


class MemoryStore:
    def __init__(self, state: RiskState | None = None) -> None:
        self.state = state or RiskState()
        self.saves = 0

    def load(self) -> RiskState:
        return self.state

    def save(self, state: RiskState) -> None:
        self.state = state
        self.saves += 1


class FailingStore(MemoryStore):
    def save(self, state: RiskState) -> None:
        raise OSError("disque plein")


def make_manager(store: MemoryStore | None = None) -> RiskManager:
    return RiskManager(LIMITS, store or MemoryStore())


def test_order_is_approved_and_counted() -> None:
    manager = make_manager()
    assert manager.review(ORDER, portfolio(), MARKET, CONTEXT).approved
    assert manager.state.orders_count == 1
    assert manager.state.order_date == CONTEXT.session_date


def test_rejected_order_is_not_counted() -> None:
    manager = make_manager()
    decision = manager.review(replace(ORDER, quantity=-1.0), portfolio(), MARKET, CONTEXT)
    assert decision.reason is RejectReason.INVALID_ORDER
    assert manager.state.orders_count == 0


@pytest.mark.risk_limit("order_limit")
def test_order_limit_is_enforced() -> None:
    manager = make_manager()
    assert manager.review(ORDER, portfolio(), MARKET, CONTEXT).approved
    assert manager.review(ORDER, portfolio(), MARKET, CONTEXT).approved
    decision = manager.review(ORDER, portfolio(), MARKET, CONTEXT)
    assert decision.reason is RejectReason.ORDER_LIMIT
    assert manager.state.orders_count == 2


def test_order_limit_resets_on_next_session() -> None:
    manager = make_manager()
    manager.review(ORDER, portfolio(), MARKET, CONTEXT)
    manager.review(ORDER, portfolio(), MARKET, CONTEXT)
    assert manager.review(NEXT_DAY_ORDER, portfolio(), NEXT_DAY_MARKET, NEXT_DAY).approved
    assert manager.state.orders_count == 1


def test_order_limit_survives_restart(tmp_path: Path) -> None:
    store = JsonRiskStateStore(tmp_path / "risk.json")
    first = RiskManager(LIMITS, store)
    first.review(ORDER, portfolio(), MARKET, CONTEXT)
    first.review(ORDER, portfolio(), MARKET, CONTEXT)

    restarted = RiskManager(LIMITS, store)

    decision = restarted.review(ORDER, portfolio(), MARKET, CONTEXT)
    assert decision.reason is RejectReason.ORDER_LIMIT


@pytest.mark.risk_limit("halted")
@pytest.mark.risk_limit("daily_loss")
def test_daily_loss_triggers_kill_switch_and_blocks_orders() -> None:
    manager = make_manager()
    losing = portfolio(equity=97_000.0, day_start=100_000.0)
    decision = manager.review(ORDER, losing, MARKET, CONTEXT)
    assert decision.reason is RejectReason.HALTED
    assert decision.detail == HaltReason.DAILY_LOSS.value
    assert manager.is_halted
    assert manager.state.halted_at == CONTEXT.now


def test_drawdown_is_measured_from_highest_equity_seen() -> None:
    manager = make_manager()
    manager.update(portfolio(equity=120_000.0), NOW)
    manager.update(portfolio(equity=110_000.0, day_start=110_000.0), NOW)
    assert manager.state.peak_equity == 120_000.0
    assert not manager.is_halted

    manager.update(portfolio(equity=108_000.0, day_start=108_000.0), NOW)

    assert manager.state.halt_reason is HaltReason.MAX_DRAWDOWN


@pytest.mark.risk_limit("max_drawdown")
def test_slow_decline_triggers_drawdown_without_any_daily_loss_trigger() -> None:
    manager = make_manager()
    equity = 100_000.0
    while not manager.is_halted:
        next_equity = equity * 0.99
        manager.update(portfolio(equity=next_equity, day_start=equity), NOW)
        equity = next_equity
    assert manager.state.halt_reason is HaltReason.MAX_DRAWDOWN
    assert equity <= 90_000.0


def test_peak_survives_restart(tmp_path: Path) -> None:
    store = JsonRiskStateStore(tmp_path / "risk.json")
    RiskManager(LIMITS, store).update(portfolio(equity=120_000.0), NOW)

    restarted = RiskManager(LIMITS, store)
    restarted.update(portfolio(equity=108_000.0, day_start=108_000.0), NOW)

    assert restarted.state.halt_reason is HaltReason.MAX_DRAWDOWN


def test_kill_switch_stays_active_when_equity_recovers() -> None:
    manager = make_manager()
    manager.update(portfolio(equity=97_000.0, day_start=100_000.0), NOW)
    decision = manager.review(ORDER, portfolio(equity=150_000.0), MARKET, CONTEXT)
    assert decision.reason is RejectReason.HALTED


def test_kill_switch_survives_restart(tmp_path: Path) -> None:
    store = JsonRiskStateStore(tmp_path / "risk.json")
    RiskManager(LIMITS, store).update(portfolio(equity=97_000.0, day_start=100_000.0), NOW)

    restarted = RiskManager(LIMITS, store)

    assert restarted.is_halted
    decision = restarted.review(ORDER, portfolio(), MARKET, CONTEXT)
    assert decision.reason is RejectReason.HALTED


def test_kill_switch_blocks_sells_through_review() -> None:
    manager = make_manager()
    manager.update(portfolio(equity=97_000.0, day_start=100_000.0), NOW)
    sell = replace(ORDER, side=OrderSide.SELL)
    assert manager.review(sell, portfolio(), MARKET, CONTEXT).reason is RejectReason.HALTED


def test_first_halt_reason_and_time_are_kept() -> None:
    manager = make_manager()
    manager.update(portfolio(equity=97_000.0, day_start=100_000.0), NOW)
    later = NOW + timedelta(hours=1)
    manager.update(portfolio(equity=50_000.0, day_start=50_000.0), later)
    assert manager.state.halt_reason is HaltReason.DAILY_LOSS
    assert manager.state.halted_at == NOW


def test_closing_orders_stay_available_after_halt() -> None:
    manager = make_manager()
    holdings = PortfolioState(
        equity=97_000.0,
        day_start_equity=100_000.0,
        positions={"SPY": Position(100.0, 10_000.0)},
    )
    manager.update(holdings, NOW)
    assert manager.is_halted
    assert [order.symbol for order in closing_orders(holdings)] == ["SPY"]


@pytest.mark.parametrize("equity", [float("nan"), float("inf"), 0.0, -1.0])
@pytest.mark.risk_limit("invalid_portfolio")
def test_invalid_portfolio_rejects_order_without_halting(equity: float) -> None:
    manager = make_manager()
    invalid = PortfolioState(equity=equity, day_start_equity=100_000.0)
    decision = manager.review(ORDER, invalid, MARKET, CONTEXT)
    assert decision.reason is RejectReason.INVALID_PORTFOLIO
    assert not manager.is_halted
    assert manager.state.peak_equity is None


def test_nan_equity_does_not_corrupt_peak() -> None:
    manager = make_manager()
    manager.update(portfolio(equity=100_000.0), NOW)
    manager.update(PortfolioState(equity=float("nan"), day_start_equity=100_000.0), NOW)
    assert manager.state.peak_equity == 100_000.0


def test_unsaved_approval_raises_instead_of_approving() -> None:
    store = FailingStore(RiskState(peak_equity=100_000.0))
    manager = RiskManager(LIMITS, store)
    with pytest.raises(OSError, match="disque plein"):
        manager.review(ORDER, portfolio(), MARKET, CONTEXT)


def test_unsaved_halt_raises_and_memory_state_is_unchanged() -> None:
    store = FailingStore(RiskState(peak_equity=100_000.0))
    manager = RiskManager(LIMITS, store)
    with pytest.raises(OSError, match="disque plein"):
        manager.update(portfolio(equity=97_000.0, day_start=100_000.0), NOW)
    assert not manager.is_halted


def test_state_is_saved_only_when_it_changes() -> None:
    store = MemoryStore()
    manager = make_manager(store)
    manager.update(portfolio(), NOW)
    manager.update(portfolio(), NOW)
    assert store.saves == 1


def test_reset_rearms_and_rebases_peak() -> None:
    manager = make_manager()
    manager.update(portfolio(equity=100_000.0), NOW)
    manager.update(portfolio(equity=85_000.0, day_start=85_000.0), NOW)
    assert manager.is_halted

    manager.reset_halt(portfolio(equity=85_000.0))

    assert not manager.is_halted
    assert manager.state.peak_equity == 85_000.0
    assert manager.state.halted_at is None
    assert manager.review(ORDER, portfolio(equity=85_000.0), MARKET, CONTEXT).approved


def test_reset_retriggers_if_daily_loss_is_still_exceeded() -> None:
    manager = make_manager()
    losing = portfolio(equity=97_000.0, day_start=100_000.0)
    manager.update(losing, NOW)
    manager.reset_halt(losing)
    manager.update(losing, NOW)
    assert manager.state.halt_reason is HaltReason.DAILY_LOSS


def test_reset_refuses_invalid_portfolio() -> None:
    manager = make_manager()
    manager.update(portfolio(equity=97_000.0, day_start=100_000.0), NOW)
    with pytest.raises(ValueError, match="inexploitable"):
        manager.reset_halt(PortfolioState(equity=float("nan"), day_start_equity=1.0))
    assert manager.is_halted


def test_store_starts_empty(tmp_path: Path) -> None:
    assert JsonRiskStateStore(tmp_path / "absent" / "risk.json").load() == RiskState()


def test_store_round_trip_creates_directory(tmp_path: Path) -> None:
    store = JsonRiskStateStore(tmp_path / "nouveau" / "risk.json")
    state = RiskState(
        peak_equity=1.5,
        halt_reason=HaltReason.MAX_DRAWDOWN,
        halted_at=NOW,
        order_date=date(2026, 11, 3),
        orders_count=4,
    )
    store.save(state)
    assert store.load() == state
    assert not (tmp_path / "nouveau" / "risk.tmp").exists()


@pytest.mark.parametrize(
    "content",
    ["pas du json", "", '{"champ_inconnu": 1}', '{"peak_equity": "abc"}', '{"peak_equity": NaN}'],
)
def test_corrupt_state_stops_instead_of_starting_blank(tmp_path: Path, content: str) -> None:
    path = tmp_path / "risk.json"
    path.write_text(content)
    with pytest.raises(RiskStateError):
        RiskManager(LIMITS, JsonRiskStateStore(path))


def test_project_risk_state_file_is_outside_git() -> None:
    assert load_strategy_config().risk_state_file.startswith("data/")
