import math
from datetime import date

import numpy as np
import pandas as pd
import pytest

from audit.metrics import (
    PerformanceMetrics,
    Status,
    compute_metrics,
    compute_trade_statistics,
    evaluate_targets,
    replay_kill_switch,
    suspicious_results,
)
from bot.config import PerformanceTargets, RiskLimits

TARGETS = PerformanceTargets(
    min_annual_return=0.08,
    max_drawdown_alert=0.10,
    max_drawdown_reject=0.15,
    min_sharpe=1.0,
    min_calmar=1.0,
    min_positive_months=0.6,
    suspicious_annual_return=0.3,
    suspicious_sharpe=2.0,
)
LIMITS = RiskLimits(
    max_position_pct=0.2,
    max_total_exposure_pct=1.0,
    max_daily_loss_pct=0.02,
    max_drawdown_pct=0.1,
    max_orders_per_day=20,
    max_data_age_seconds=900,
    max_clock_skew_seconds=5,
    max_price_change_pct=0.2,
    allow_short=False,
    allow_leverage=False,
)


def make_returns(values: list[float], start: str = "2024-01-02") -> pd.Series:
    index = pd.bdate_range(start, periods=len(values), name="session")
    return pd.Series(values, index=index, dtype=float)


def random_returns() -> pd.Series:
    rng = np.random.default_rng(7)
    return make_returns(rng.normal(0.0005, 0.01, 252 * 3).tolist(), "2016-01-04")


def make_metrics(**overrides: float) -> PerformanceMetrics:
    values: dict[str, float] = {
        "annual_return": 0.10,
        "annual_volatility": 0.08,
        "sharpe": 1.2,
        "sortino": 1.6,
        "calmar": 1.25,
        "max_drawdown": -0.08,
        "max_drawdown_sessions": 50,
        "positive_months": 0.65,
        "sessions": 756,
        **overrides,
    }
    return PerformanceMetrics.model_validate(values)


BENCHMARK = make_metrics(annual_volatility=0.17, sharpe=0.8)


def test_annualization_uses_252_sessions() -> None:
    returns = random_returns()
    metrics = compute_metrics(returns)
    total = float(np.prod(1 + returns.to_numpy()))
    assert metrics.annual_return == pytest.approx(total ** (252 / len(returns)) - 1)
    assert metrics.annual_volatility == pytest.approx(returns.std() * math.sqrt(252))
    assert metrics.sharpe == pytest.approx(returns.mean() / returns.std() * math.sqrt(252))


def test_sortino_matches_hand_computation() -> None:
    returns = random_returns()
    downside = np.sqrt((np.minimum(returns, 0) ** 2).sum() / len(returns))
    assert compute_metrics(returns).sortino == pytest.approx(
        returns.mean() / downside * math.sqrt(252)
    )


def test_drawdown_and_calmar_on_known_path() -> None:
    # Capital : 1.10, 0.88, 0.99, 1.10. Plus bas de -20 % depuis 1.10.
    returns = make_returns([0.10, -0.20, 0.125, 1.10 / 0.99 - 1])
    metrics = compute_metrics(returns)
    assert metrics.max_drawdown == pytest.approx(-0.20)
    assert metrics.calmar == pytest.approx(metrics.annual_return / 0.20)
    assert metrics.sessions == 4


def test_drawdown_from_initial_capital_on_first_session() -> None:
    assert compute_metrics(make_returns([-0.10, 0.05, 0.01])).max_drawdown == pytest.approx(-0.10)
    assert compute_metrics(make_returns([-0.10, 0.05, 0.01])).max_drawdown_sessions == 3


def test_longest_drawdown_duration() -> None:
    returns = make_returns([0.10, -0.05, -0.05, 0.20, 0.01, -0.01, 0.05])
    assert compute_metrics(returns).max_drawdown_sessions == 2


def test_positive_months_compounds_within_each_month_and_counts_flat_as_negative() -> None:
    january = make_returns([0.10, -0.05], "2024-01-02")
    february = make_returns([-0.10, 0.05], "2024-02-01")
    march = make_returns([0.0, 0.0], "2024-03-01")
    april = make_returns([0.01, 0.01], "2024-04-01")
    returns = pd.concat([january, february, march, april])
    assert compute_metrics(returns).positive_months == pytest.approx(0.5)


@pytest.mark.parametrize("values", [[], [0.01, float("nan")]])
def test_invalid_returns_are_rejected(values: list[float]) -> None:
    with pytest.raises(ValueError, match="complets"):
        compute_metrics(make_returns(values))


def test_all_flat_returns_do_not_pass_ratio_targets() -> None:
    metrics = compute_metrics(make_returns([0.0] * 30))
    statuses = {c.name: c.status for c in evaluate_targets(metrics, BENCHMARK, TARGETS)}
    assert statuses["annual_return"] is Status.FAIL
    assert statuses["calmar"] is Status.FAIL
    assert statuses["sharpe"] is Status.FAIL


def test_trade_statistics() -> None:
    stats = compute_trade_statistics(np.array([100.0, 300.0, -100.0, -100.0, 0.0]))
    assert stats.closed_trades == 5
    assert stats.win_rate == pytest.approx(0.4)
    assert stats.profit_factor == pytest.approx(2.0)
    assert stats.average_win_over_loss == pytest.approx(2.0)


def test_trade_statistics_without_losses_or_trades() -> None:
    assert compute_trade_statistics(np.array([1.0])).profit_factor is None
    assert compute_trade_statistics(np.array([1.0])).average_win_over_loss is None
    empty = compute_trade_statistics(np.array([]))
    assert empty.closed_trades == 0
    assert empty.win_rate == 0.0


def test_replay_finds_daily_loss_at_exact_limit() -> None:
    replay = replay_kill_switch(make_returns([0.01, -0.019, -0.02, -0.03]), LIMITS)
    assert replay.first_daily_loss == date(2024, 1, 4)
    assert replay.daily_loss_sessions == 2


def test_replay_finds_slow_drawdown_without_daily_loss() -> None:
    replay = replay_kill_switch(make_returns([-0.015] * 10), LIMITS)
    assert replay.first_daily_loss is None
    assert replay.first_drawdown == date(2024, 1, 10)


def test_replay_counts_drawdown_at_exact_limit() -> None:
    assert replay_kill_switch(make_returns([-0.10]), LIMITS).first_drawdown == date(2024, 1, 2)


def test_replay_finds_nothing_on_calm_path() -> None:
    replay = replay_kill_switch(make_returns([0.001] * 10), LIMITS)
    assert replay.first_daily_loss is None
    assert replay.first_drawdown is None
    assert replay.drawdown_sessions == 0


def test_replay_measures_drawdown_from_initial_capital() -> None:
    assert replay_kill_switch(make_returns([-0.019] * 6), LIMITS).first_drawdown == date(2024, 1, 9)


def test_good_metrics_pass_every_criterion() -> None:
    criteria = evaluate_targets(make_metrics(), BENCHMARK, TARGETS)
    assert {criterion.status for criterion in criteria} == {Status.PASS}
    assert len(criteria) == 7


@pytest.mark.parametrize(
    ("drawdown", "status"),
    [
        (-0.10, Status.PASS),
        (-0.101, Status.ALERT),
        (-0.15, Status.ALERT),
        (-0.151, Status.FAIL),
    ],
)
def test_drawdown_thresholds(drawdown: float, status: Status) -> None:
    metrics = make_metrics(max_drawdown=drawdown)
    statuses = {c.name: c.status for c in evaluate_targets(metrics, BENCHMARK, TARGETS)}
    assert statuses["max_drawdown"] is status


@pytest.mark.parametrize(
    ("name", "overrides"),
    [
        ("annual_return", {"annual_return": 0.079}),
        ("sharpe", {"sharpe": 0.99}),
        ("calmar", {"calmar": 0.99}),
        ("positive_months", {"positive_months": 0.59}),
        ("volatility_below_benchmark", {"annual_volatility": 0.17}),
        ("sharpe_above_benchmark", {"sharpe": 0.79}),
        ("sharpe", {"sharpe": float("nan")}),
    ],
)
def test_each_criterion_can_fail(name: str, overrides: dict[str, float]) -> None:
    criteria = evaluate_targets(make_metrics(**overrides), BENCHMARK, TARGETS)
    assert {c.name: c.status for c in criteria}[name] is Status.FAIL


def test_suspicious_results_are_flagged() -> None:
    assert suspicious_results(make_metrics(), TARGETS) == []
    assert suspicious_results(make_metrics(annual_return=0.31, sharpe=2.1), TARGETS) == [
        "annual_return",
        "sharpe",
    ]
