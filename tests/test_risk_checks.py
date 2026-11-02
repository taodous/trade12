from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
from typing import Any

import pytest

from bot.config import RiskLimits
from bot.risk.checks import check_order, closing_orders, halt_reason, is_valid_portfolio
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

NOW = datetime(2026, 11, 3, 15, 0, tzinfo=UTC)
LAST_SESSION = date(2026, 11, 2)
CONTEXT = RiskContext(now=NOW, session_date=date(2026, 11, 3), last_completed_session=LAST_SESSION)
NAN = float("nan")
INF = float("inf")


def make_limits(**overrides: Any) -> RiskLimits:
    values: dict[str, Any] = {
        "max_position_pct": 0.2,
        "max_total_exposure_pct": 1.0,
        "max_daily_loss_pct": 0.02,
        "max_drawdown_pct": 0.1,
        "max_orders_per_day": 3,
        "max_data_age_seconds": 900,
        "max_price_change_pct": 0.2,
        "allow_short": False,
        "allow_leverage": False,
        **overrides,
    }
    return RiskLimits(**values)


LIMITS = make_limits()


def make_portfolio(
    positions: dict[str, Position] | None = None,
    equity: float = 100_000.0,
    day_start_equity: float = 100_000.0,
) -> PortfolioState:
    return PortfolioState(
        equity=equity, day_start_equity=day_start_equity, positions=positions or {}
    )


def buy(quantity: float = 100.0, symbol: str = "SPY") -> ProposedOrder:
    return ProposedOrder(symbol, OrderSide.BUY, quantity, LAST_SESSION)


def sell(quantity: float = 100.0, symbol: str = "SPY") -> ProposedOrder:
    return ProposedOrder(symbol, OrderSide.SELL, quantity, LAST_SESSION)


def make_market(
    price: float = 100.0, previous_close: float = 100.0, age_seconds: float = 60.0
) -> dict[str, PriceSnapshot]:
    return {"SPY": PriceSnapshot(price, previous_close, NOW - timedelta(seconds=age_seconds))}


def check(
    order: ProposedOrder,
    portfolio: PortfolioState | None = None,
    market: dict[str, PriceSnapshot] | None = None,
    limits: RiskLimits = LIMITS,
) -> RejectReason | None:
    decision = check_order(
        order,
        portfolio or make_portfolio(),
        make_market() if market is None else market,
        CONTEXT,
        limits,
    )
    assert decision.approved == (decision.reason is None)
    return decision.reason


def test_valid_order_is_approved() -> None:
    assert check(buy()) is None


@pytest.mark.parametrize("quantity", [0.0, -1.0, NAN, INF])
def test_invalid_quantity_is_rejected(quantity: float) -> None:
    assert check(buy(quantity)) is RejectReason.INVALID_ORDER


def test_missing_snapshot_is_rejected() -> None:
    assert check(buy(), market={}) is RejectReason.MISSING_DATA


@pytest.mark.parametrize(
    ("price", "previous_close"),
    [(NAN, 100.0), (100.0, NAN), (INF, 100.0), (0.0, 100.0), (100.0, 0.0), (-1.0, 100.0)],
)
def test_invalid_prices_are_rejected(price: float, previous_close: float) -> None:
    market = make_market(price, previous_close)
    assert check(buy(), market=market) is RejectReason.MISSING_DATA


def test_timestamp_without_timezone_is_rejected() -> None:
    market = {"SPY": PriceSnapshot(100.0, 100.0, NOW.replace(tzinfo=None))}
    assert check(buy(), market=market) is RejectReason.MISSING_DATA


@pytest.mark.parametrize(("age", "reason"), [(900.0, None), (901.0, RejectReason.STALE_DATA)])
def test_data_age_limit(age: float, reason: RejectReason | None) -> None:
    assert check(buy(), market=make_market(age_seconds=age)) is reason


def test_price_from_the_future_is_rejected() -> None:
    assert check(buy(), market=make_market(age_seconds=-1.0)) is RejectReason.STALE_DATA


@pytest.mark.parametrize(
    ("price", "reason"),
    [
        (119.0, None),
        (81.0, None),
        (121.0, RejectReason.ABERRANT_PRICE),
        (79.0, RejectReason.ABERRANT_PRICE),
    ],
)
def test_aberrant_price_limit(price: float, reason: RejectReason | None) -> None:
    order = buy(1.0)
    assert check(order, market=make_market(price=price)) is reason


def test_data_checks_also_apply_to_sells() -> None:
    portfolio = make_portfolio({"SPY": Position(100.0, 10_000.0)})
    market = make_market(age_seconds=901.0)
    assert check(sell(), portfolio, market) is RejectReason.STALE_DATA


@pytest.mark.parametrize(
    "signal_session",
    [date(2026, 10, 30), date(2026, 11, 3), date(2026, 11, 4)],
    ids=["old", "same_day_look_ahead", "future"],
)
def test_signal_must_come_from_last_completed_session(signal_session: date) -> None:
    order = replace(buy(), signal_session=signal_session)
    assert check(order) is RejectReason.STALE_SIGNAL


@pytest.mark.parametrize(
    ("quantity", "reason"),
    [(200.0, None), (200.01, RejectReason.POSITION_LIMIT)],
)
def test_position_limit_is_a_fraction_of_equity(
    quantity: float, reason: RejectReason | None
) -> None:
    assert check(buy(quantity)) is reason


def test_position_limit_counts_existing_position() -> None:
    portfolio = make_portfolio({"SPY": Position(150.0, 15_000.0)})
    assert check(buy(50.0), portfolio) is None
    assert check(buy(51.0), portfolio) is RejectReason.POSITION_LIMIT


def test_position_limit_counts_open_orders() -> None:
    portfolio = make_portfolio({"SPY": Position(0.0, 0.0, pending_notional=15_000.0)})
    assert check(buy(50.0), portfolio) is None
    assert check(buy(51.0), portfolio) is RejectReason.POSITION_LIMIT


def test_position_limit_follows_lower_equity() -> None:
    assert check(buy(150.0), make_portfolio(equity=50_000.0)) is RejectReason.POSITION_LIMIT


def invested_portfolio(invested: float) -> PortfolioState:
    return make_portfolio({"QQQ": Position(1.0, invested)})


@pytest.mark.parametrize(
    ("quantity", "reason"),
    [(100.0, None), (101.0, RejectReason.EXPOSURE_LIMIT)],
)
def test_exposure_limit_counts_all_positions(quantity: float, reason: RejectReason | None) -> None:
    assert check(buy(quantity), invested_portfolio(90_000.0)) is reason


def test_exposure_limit_counts_open_orders_on_other_symbols() -> None:
    portfolio = make_portfolio({"QQQ": Position(0.0, 0.0, pending_notional=90_000.0)})
    assert check(buy(101.0), portfolio) is RejectReason.EXPOSURE_LIMIT


def test_exposure_limit_counts_short_positions_as_exposure() -> None:
    assert check(buy(101.0), invested_portfolio(-90_000.0)) is RejectReason.EXPOSURE_LIMIT


def test_exposure_above_equity_is_refused_without_leverage() -> None:
    """Le compte offre 4 fois son capital en pouvoir d'achat : il ne doit jamais servir."""
    limits = make_limits(max_position_pct=1.0)
    portfolio = invested_portfolio(50_000.0)
    assert check(buy(500.0), portfolio, limits=limits) is None
    assert check(buy(501.0), portfolio, limits=limits) is RejectReason.EXPOSURE_LIMIT


def test_leverage_requires_explicit_activation() -> None:
    limits = make_limits(max_position_pct=1.0, max_total_exposure_pct=1.5, allow_leverage=True)
    portfolio = invested_portfolio(100_000.0)
    assert check(buy(500.0), portfolio, limits=limits) is None
    assert check(buy(501.0), portfolio, limits=limits) is RejectReason.EXPOSURE_LIMIT


def test_sell_without_position_is_short_and_rejected() -> None:
    assert check(sell()) is RejectReason.SHORT_NOT_ALLOWED


def test_sell_beyond_position_is_rejected() -> None:
    portfolio = make_portfolio({"SPY": Position(100.0, 10_000.0)})
    assert check(sell(100.01), portfolio) is RejectReason.SHORT_NOT_ALLOWED


def test_small_reversal_to_short_is_rejected() -> None:
    """La position diminue en valeur absolue, mais passe quand même à découvert."""
    portfolio = make_portfolio({"SPY": Position(100.0, 10_000.0)})
    assert check(sell(150.0), portfolio) is RejectReason.SHORT_NOT_ALLOWED


def test_sell_beyond_position_with_open_sell_is_rejected() -> None:
    portfolio = make_portfolio({"SPY": Position(100.0, 10_000.0, pending_notional=-10_000.0)})
    assert check(sell(1.0), portfolio) is RejectReason.SHORT_NOT_ALLOWED


def test_sell_of_whole_position_is_approved() -> None:
    assert check(sell(), make_portfolio({"SPY": Position(100.0, 10_000.0)})) is None


def test_short_is_allowed_only_when_enabled_and_within_limits() -> None:
    limits = make_limits(allow_short=True)
    assert check(sell(200.0), limits=limits) is None
    assert check(sell(201.0), limits=limits) is RejectReason.POSITION_LIMIT


def test_existing_short_can_be_reduced_when_shorting_is_disabled() -> None:
    portfolio = make_portfolio({"SPY": Position(-100.0, -10_000.0)})
    assert check(buy(50.0), portfolio) is None


def test_reducing_oversized_position_is_approved() -> None:
    """Après une hausse ou une baisse du capital, il faut pouvoir revenir sous la limite."""
    portfolio = make_portfolio({"SPY": Position(500.0, 50_000.0)})
    assert check(sell(100.0), portfolio) is None
    assert check(buy(1.0), portfolio) is RejectReason.POSITION_LIMIT


def test_reducing_position_is_approved_above_exposure_limit() -> None:
    portfolio = make_portfolio(
        {"SPY": Position(100.0, 10_000.0), "QQQ": Position(1.0, 100_000.0)},
    )
    assert check(sell(50.0), portfolio) is None


@pytest.mark.parametrize(
    "portfolio",
    [
        make_portfolio(equity=NAN),
        make_portfolio(equity=INF),
        make_portfolio(equity=0.0),
        make_portfolio(equity=-1.0),
        make_portfolio(day_start_equity=NAN),
        make_portfolio(day_start_equity=0.0),
        make_portfolio({"SPY": Position(NAN, 1.0)}),
        make_portfolio({"SPY": Position(1.0, NAN)}),
        make_portfolio({"SPY": Position(1.0, 1.0, pending_notional=INF)}),
    ],
)
def test_invalid_portfolio_is_detected(portfolio: PortfolioState) -> None:
    assert is_valid_portfolio(portfolio) is False


def test_valid_portfolio_is_accepted() -> None:
    assert is_valid_portfolio(make_portfolio({"SPY": Position(1.0, 100.0)}))


@pytest.mark.parametrize(
    ("equity", "reason"),
    [
        (98_001.0, None),
        (98_000.0, HaltReason.DAILY_LOSS),
        (90_000.0, HaltReason.DAILY_LOSS),
    ],
)
def test_daily_loss_limit(equity: float, reason: HaltReason | None) -> None:
    portfolio = make_portfolio(equity=equity)
    assert halt_reason(portfolio, 100_000.0, LIMITS) is reason


@pytest.mark.parametrize(
    ("equity", "reason"),
    [(90_001.0, None), (90_000.0, HaltReason.MAX_DRAWDOWN), (80_000.0, HaltReason.MAX_DRAWDOWN)],
)
def test_drawdown_limit_is_measured_from_peak(equity: float, reason: HaltReason | None) -> None:
    portfolio = make_portfolio(equity=equity, day_start_equity=equity)
    assert halt_reason(portfolio, 100_000.0, LIMITS) is reason


def test_gain_does_not_trigger_kill_switch() -> None:
    portfolio = make_portfolio(equity=120_000.0)
    assert halt_reason(portfolio, 120_000.0, LIMITS) is None


def test_closing_orders_close_every_position_and_skip_empty_ones() -> None:
    portfolio = make_portfolio(
        {
            "TLT": Position(-20.0, -2_000.0),
            "SPY": Position(10.5, 1_000.0),
            "QQQ": Position(0.0, 0.0),
        }
    )
    orders = closing_orders(portfolio)
    assert [(order.symbol, order.side, order.quantity) for order in orders] == [
        ("SPY", OrderSide.SELL, 10.5),
        ("TLT", OrderSide.BUY, 20.0),
    ]


def test_closing_orders_on_empty_portfolio() -> None:
    assert closing_orders(make_portfolio()) == []


def test_open_sell_does_not_hide_a_risk_increase() -> None:
    """La position réelle visée est de 5 000 : l'achat l'augmente, même sous sa valeur actuelle."""
    portfolio = make_portfolio({"SPY": Position(300.0, 30_000.0, pending_notional=-25_000.0)})
    assert check(buy(200.0), portfolio) is RejectReason.POSITION_LIMIT
