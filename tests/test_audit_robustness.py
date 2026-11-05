import json
import math
from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from audit.backtest import BacktestError
from audit.metrics import deflated_sharpe
from audit.monte_carlo import (
    block_bootstrap,
    max_drawdowns,
    run_monte_carlo,
    trade_order_drawdowns,
)
from audit.registry import COST_STRESS, SENSITIVITY, load_registry, record_variant, trial_sharpes
from audit.sensitivity import neighbor_share, run_sensitivity
from audit.walk_forward import run_walk_forward, window_bounds
from bot.config import MonteCarloConfig, StrategyConfig
from tests.test_audit_backtest import make_config, random_prices

NOW = datetime(2026, 11, 4, tzinfo=UTC)


def audit_config() -> StrategyConfig:
    """Grille 2 x 2 autour de 5/20, walk-forward court, sur 300 séances synthétiques."""
    base = make_config(["A", "B"], date(2022, 12, 31))
    audit = base.audit
    return base.model_copy(
        update={
            "backtest": base.backtest.model_copy(update={"slippage_pct": 0.001}),
            "audit": audit.model_copy(
                update={
                    # Fenêtres rapides 3 et 5, lentes 12 et 20 autour de 5/20.
                    "sensitivity": audit.sensitivity.model_copy(
                        update={"relative_steps": [0.6, 1.0]}
                    ),
                    "walk_forward": audit.walk_forward.model_copy(
                        update={"train_sessions": 60, "test_sessions": 30}
                    ),
                }
            ),
        }
    )


# Sensibilité


def test_neighbor_share_counts_settings_close_to_the_chosen_one() -> None:
    assert neighbor_share(1.0, [0.9, 0.7, float("nan")], 0.8) == pytest.approx(1 / 3)
    assert neighbor_share(1.0, [0.8], 0.8) == 1.0


@pytest.mark.parametrize("chosen", [0.0, -0.5, float("nan")])
def test_neighbor_share_is_undefined_without_a_positive_sharpe(chosen: float) -> None:
    assert math.isnan(neighbor_share(chosen, [0.5], 0.8))


def test_neighbor_share_is_undefined_without_neighbors() -> None:
    assert math.isnan(neighbor_share(1.0, [], 0.8))


def test_sensitivity_evaluates_grid_and_registers_each_setting(tmp_path: Path) -> None:
    config = audit_config()
    opens, closes = random_prices()
    registry = tmp_path / "variants.jsonl"

    report = run_sensitivity(config, opens, closes, registry, NOW)

    assert report.parameters == ["fast_window", "slow_window"]
    assert [(p.params["fast_window"], p.params["slow_window"]) for p in report.grid] == [
        (3, 12),
        (3, 20),
        (5, 12),
        (5, 20),
    ]
    assert report.chosen_sharpe == report.grid[3].sharpe
    assert report.evaluation_start == pd.Timestamp(closes.index[20]).date().isoformat()
    purposes = [entry["purpose"] for entry in load_registry(registry)]
    assert purposes == [SENSITIVITY] * 4 + [COST_STRESS] * 2


def test_higher_slippage_lowers_the_return(tmp_path: Path) -> None:
    config = audit_config()
    opens, closes = random_prices()
    report = run_sensitivity(config, opens, closes, tmp_path / "v.jsonl", NOW)
    assert [cost.slippage_multiplier for cost in report.costs] == [2.0, 3.0]
    assert report.costs[1].annual_return < report.costs[0].annual_return
    assert report.costs[0].slippage_pct == pytest.approx(2 * config.backtest.slippage_pct)


# Registre


def test_trial_sharpes_exclude_cost_stress_and_read_legacy_entries(tmp_path: Path) -> None:
    path = tmp_path / "variants.jsonl"
    legacy = {"id": "ancien", "definition": {}, "results": {"sharpe": 0.4}}
    path.write_text(json.dumps(legacy) + "\n", encoding="utf-8")
    record_variant(path, {"a": 1}, {"sharpe": 0.6}, NOW, SENSITIVITY)
    record_variant(path, {"a": 2}, {"sharpe": 0.1}, NOW, COST_STRESS)
    assert trial_sharpes(path) == [0.4, 0.6]


def test_registry_is_empty_when_absent(tmp_path: Path) -> None:
    assert load_registry(tmp_path / "absent.jsonl") == []
    assert trial_sharpes(tmp_path / "absent.jsonl") == []


# Walk-forward


def test_window_bounds_slide_by_test_length_and_keep_full_windows() -> None:
    windows = window_bounds(sessions=100, first=10, train=50, test=20)
    assert [(w.train_start, w.test_start, w.test_end) for w in windows] == [
        (10, 60, 80),
        (30, 80, 100),
    ]
    assert window_bounds(sessions=79, first=10, train=50, test=20) == []


def test_walk_forward_reports_each_window() -> None:
    config = audit_config()
    opens, closes = random_prices()

    report = run_walk_forward(config, opens, closes)

    assert len(report.windows) == 7
    assert report.windows[0].train_start == pd.Timestamp(closes.index[20]).date().isoformat()
    assert report.windows[0].test_start == pd.Timestamp(closes.index[80]).date().isoformat()
    assert report.reoptimized.sessions == 7 * 30
    grid = {(3, 12), (3, 20), (5, 12), (5, 20)}
    for window in report.windows:
        assert (window.selected["fast_window"], window.selected["slow_window"]) in grid
    shares = [w.fixed_test_return > 0 for w in report.windows]
    assert report.fixed_positive_share == pytest.approx(sum(shares) / len(shares))


def test_walk_forward_selection_ignores_later_prices() -> None:
    """Changer les prix après le début d'une fenêtre de test ne change pas son choix de réglage."""
    config = audit_config()
    opens, closes = random_prices()
    cut = 140
    shocked_opens, shocked_closes = opens.copy(), closes.copy()
    shocked_opens.iloc[cut:] *= np.linspace(0.5, 2.0, len(opens) - cut)[:, None]
    shocked_closes.iloc[cut:] *= np.linspace(0.5, 2.0, len(closes) - cut)[:, None]

    first = run_walk_forward(config, opens, closes).windows
    second = run_walk_forward(config, shocked_opens, shocked_closes).windows

    # Les fenêtres de test commencent aux séances 80, 110 et 140 : leur choix ne voit que l'avant.
    for before, after in zip(first[:3], second[:3], strict=True):
        assert (before.selected, before.train_sharpe) == (after.selected, after.train_sharpe)
    assert first[:2] == second[:2]


def test_walk_forward_needs_one_full_window() -> None:
    config = audit_config()
    opens, closes = random_prices(100)
    with pytest.raises(BacktestError, match="walk-forward"):
        run_walk_forward(config, opens, closes)


# Monte Carlo

MC = MonteCarloConfig(seed=7, simulations=500, block_sessions=5)


def test_max_drawdowns_per_row() -> None:
    equity = np.array([[1.0, 1.2, 0.9, 1.3], [1.0, 1.1, 1.2, 1.3]])
    assert max_drawdowns(equity) == pytest.approx([0.25, 0.0])


def test_trade_order_drawdown_counts_a_loss_from_the_start() -> None:
    rng = np.random.default_rng(0)
    gains = trade_order_drawdowns(np.array([10.0, 20.0]), 100.0, 50, rng)
    assert gains == pytest.approx(np.zeros(50))
    losses = trade_order_drawdowns(np.array([-10.0, -20.0]), 100.0, 50, rng)
    assert losses == pytest.approx(np.full(50, 0.3))


def test_block_bootstrap_keeps_length_and_consecutive_blocks() -> None:
    returns = np.arange(10, dtype=float)
    samples = block_bootstrap(returns, 4, 20, np.random.default_rng(1))
    assert samples.shape == (20, 10)
    steps = np.diff(samples[:, :4], axis=1)
    assert set(np.unique(steps)) <= {1.0, -9.0}


def test_monte_carlo_is_reproducible_with_its_seed() -> None:
    returns = pd.Series(np.random.default_rng(3).normal(0.0005, 0.01, 300))
    pnl = np.array([100.0, -50.0, 80.0, -120.0, 60.0])
    first = run_monte_carlo(returns, pnl, 10_000.0, MC)
    assert first == run_monte_carlo(returns, pnl, 10_000.0, MC)
    assert first.trade_order_drawdown is not None
    assert first.worst_drawdown_p95 == max(first.trade_order_drawdown.p95, first.block_drawdown.p95)
    assert first.block_drawdown.p5 <= first.block_drawdown.p50 <= first.block_drawdown.p95


def test_monte_carlo_without_trades_uses_the_block_draw() -> None:
    returns = pd.Series(np.random.default_rng(3).normal(0.0, 0.01, 100))
    report = run_monte_carlo(returns, np.array([]), 10_000.0, MC)
    assert report.trade_order_drawdown is None
    assert report.worst_drawdown_p95 == report.block_drawdown.p95


def test_monte_carlo_refuses_incomplete_returns() -> None:
    with pytest.raises(ValueError, match="complets"):
        run_monte_carlo(pd.Series([0.01, float("nan")]), np.array([]), 1.0, MC)


# Sharpe dégonflé


def drifting(mean: float, sessions: int = 1000, seed: int = 0) -> pd.Series:
    return pd.Series(np.random.default_rng(seed).normal(mean, 0.01, sessions))


def test_single_trial_is_the_probabilistic_sharpe() -> None:
    result = deflated_sharpe(drifting(0.002), [1.5])
    assert result.trials == 1
    assert result.expected_max_sharpe == 0.0
    assert result.probability > 0.99


def test_more_trials_raise_the_bar_and_lower_the_probability() -> None:
    returns = drifting(0.0003)
    alone = deflated_sharpe(returns, [0.5])
    crowded = deflated_sharpe(returns, [0.5, -0.4, 1.2, 0.1, 0.9, -1.0, 0.3, 0.6])
    assert crowded.expected_max_sharpe > 0
    assert crowded.probability < alone.probability


def test_no_edge_gives_low_probability() -> None:
    assert deflated_sharpe(drifting(-0.0005), [0.0]).probability < 0.5


def test_flat_returns_have_no_probability() -> None:
    assert math.isnan(deflated_sharpe(pd.Series([0.0] * 50), [0.0]).probability)


@pytest.mark.parametrize(
    ("returns", "trials"),
    [
        (pd.Series([0.01]), [1.0]),
        (pd.Series([0.01, float("nan")]), [1.0]),
        (pd.Series([0.01, 0.02]), []),
        (pd.Series([0.01, 0.02]), [float("inf")]),
    ],
)
def test_deflated_sharpe_refuses_invalid_inputs(returns: pd.Series, trials: list[float]) -> None:
    with pytest.raises(ValueError):
        deflated_sharpe(returns, trials)
