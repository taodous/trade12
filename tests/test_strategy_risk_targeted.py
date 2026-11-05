import math
from typing import Any

import numpy as np
import pandas as pd
import pytest

from bot.config import (
    MovingAverageParams,
    MovingAverageSettings,
    RebalanceConfig,
    RiskTargetedParams,
    RiskTargetedSettings,
)
from bot.strategy.equal_weight import EqualWeight
from bot.strategy.factory import build_strategy, rebalance_policy
from bot.strategy.moving_average import MovingAverageCrossover
from bot.strategy.risk_targeted import TRADING_DAYS, RiskTargeted


def make_params(**overrides: Any) -> RiskTargetedParams:
    values: dict[str, Any] = {
        "trend": "sma",
        "trend_window": 5,
        "volatility_window": 5,
        "target_volatility": 0.08,
        "max_weight": 1.0,
        "max_exposure": 1.0,
        "regime_filter": False,
        "regime_symbol": "A",
        "regime_window": 5,
        "regime_scale": 0.5,
        "risky_symbols": ["A"],
        **overrides,
    }
    return RiskTargetedParams(**values)


def walk(drifts: dict[str, float], vols: dict[str, float], sessions: int = 60) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    index = pd.bdate_range("2024-01-02", periods=sessions, name="session")
    data = {
        symbol: 100 * np.exp(np.cumsum(drift + vols[symbol] * rng.standard_normal(sessions)))
        for symbol, drift in drifts.items()
    }
    return pd.DataFrame(data, index=index)


def annual_vol(closes: pd.Series, window: int) -> float:
    returns = closes.pct_change().iloc[-window:]
    return float(returns.std(ddof=1)) * math.sqrt(TRADING_DAYS)


def test_warm_up_covers_trend_volatility_and_regime() -> None:
    assert RiskTargeted(make_params(trend_window=200, volatility_window=63)).warm_up == 201
    assert RiskTargeted(make_params(trend_window=20, volatility_window=63)).warm_up == 64
    with_regime = make_params(trend_window=20, regime_filter=True, regime_window=300)
    assert RiskTargeted(with_regime).warm_up == 300


def test_properties_expose_cap_and_required_symbol() -> None:
    assert RiskTargeted(make_params(max_weight=0.2)).max_weight == 0.2
    assert RiskTargeted(make_params()).required_symbols == frozenset()
    assert RiskTargeted(make_params(regime_filter=True)).required_symbols == {"A"}


def test_no_weight_before_warm_up_and_weights_stay_within_bounds() -> None:
    closes = walk({"A": 0.01, "B": 0.01, "C": 0.0}, {"A": 0.01, "B": 0.02, "C": 0.01})
    strategy = RiskTargeted(make_params(max_weight=0.3, max_exposure=0.8, target_volatility=1.0))
    weights = strategy.target_weights(closes)
    assert (weights.iloc[: strategy.warm_up - 1] == 0).all().all()
    assert weights.notna().all().all()
    assert ((weights >= 0) & (weights <= 0.3 + 1e-12)).all().all()
    assert (weights.sum(axis=1) <= 0.8 + 1e-12).all()


def test_single_asset_is_scaled_to_the_target_volatility() -> None:
    closes = walk({"A": 0.01}, {"A": 0.01})
    weights = RiskTargeted(make_params()).target_weights(closes)
    expected = 0.08 / annual_vol(closes["A"], 5)
    assert weights["A"].iloc[-1] == pytest.approx(expected)


def test_weights_are_inversely_proportional_to_volatility() -> None:
    closes = walk({"A": 0.01, "B": 0.02}, {"A": 0.01, "B": 0.02})
    weights = RiskTargeted(make_params(target_volatility=0.01)).target_weights(closes).iloc[-1]
    ratio = annual_vol(closes["B"], 5) / annual_vol(closes["A"], 5)
    assert weights["A"] / weights["B"] == pytest.approx(ratio)


def test_exposure_cap_limits_a_very_calm_asset() -> None:
    closes = walk({"A": 0.001}, {"A": 0.0001})
    strategy = RiskTargeted(make_params(max_weight=0.7, max_exposure=0.7))
    weights = strategy.target_weights(closes)
    assert weights["A"].iloc[-1] == pytest.approx(0.7)


def test_position_cap_applies_after_scaling() -> None:
    closes = walk({"A": 0.001}, {"A": 0.0001})
    weights = RiskTargeted(make_params(max_weight=0.2)).target_weights(closes)
    assert weights["A"].iloc[-1] == pytest.approx(0.2)


def test_asset_below_its_average_is_not_held() -> None:
    closes = walk({"A": 0.01, "B": -0.02}, {"A": 0.005, "B": 0.005})
    weights = RiskTargeted(make_params()).target_weights(closes).iloc[-1]
    assert weights["A"] > 0
    assert weights["B"] == 0


def test_momentum_trend_uses_the_return_over_the_window() -> None:
    index = pd.bdate_range("2024-01-02", periods=8)
    # Rendement sur 5 séances : à la dernière séance, 101 contre 102 cinq séances plus tôt.
    values = [100.0, 101.0, 102.0, 103.0, 104.0, 103.0, 102.5, 101.0]
    closes = pd.DataFrame({"A": values}, index=index)
    strategy = RiskTargeted(make_params(trend="momentum", trend_window=5, volatility_window=3))
    weights = strategy.target_weights(closes)["A"]
    assert weights.iloc[5] > 0
    assert weights.iloc[7] == 0


def test_flat_prices_give_no_weight_instead_of_nan() -> None:
    index = pd.bdate_range("2024-01-02", periods=20)
    closes = pd.DataFrame({"A": np.linspace(100, 101, 20), "B": 50.0}, index=index)
    weights = RiskTargeted(make_params()).target_weights(closes)
    assert weights.notna().all().all()
    assert (weights["B"] == 0).all()


def test_regime_on_leaves_weights_unchanged() -> None:
    closes = walk({"A": 0.01, "B": 0.01}, {"A": 0.005, "B": 0.005})
    base = make_params(risky_symbols=["A", "B"])
    plain = RiskTargeted(base).target_weights(closes)
    filtered = RiskTargeted(base.model_copy(update={"regime_filter": True})).target_weights(closes)
    assert (plain.iloc[-1] > 0).all()
    pd.testing.assert_series_equal(plain.iloc[-1], filtered.iloc[-1])


def test_regime_off_halves_risky_weights() -> None:
    index = pd.bdate_range("2024-01-02", periods=30)
    rng = np.random.default_rng(1)
    falling_regime = 100 * np.exp(np.cumsum(-0.01 + 0.002 * rng.standard_normal(30)))
    rising_asset = 100 * np.exp(np.cumsum(0.01 + 0.005 * rng.standard_normal(30)))
    closes = pd.DataFrame({"R": falling_regime, "A": rising_asset}, index=index)
    base = make_params(regime_symbol="R", risky_symbols=["A"])

    plain = RiskTargeted(base).target_weights(closes)["A"]
    filtered = RiskTargeted(base.model_copy(update={"regime_filter": True})).target_weights(closes)[
        "A"
    ]

    assert plain.iloc[-1] > 0
    assert filtered.iloc[-1] == pytest.approx(plain.iloc[-1] * 0.5)


def test_regime_filter_without_its_symbol_is_refused() -> None:
    closes = walk({"B": 0.01}, {"B": 0.01})
    with pytest.raises(ValueError, match="régime"):
        RiskTargeted(make_params(regime_filter=True)).target_weights(closes)


def test_unsorted_sessions_are_refused() -> None:
    closes = walk({"A": 0.01}, {"A": 0.01}).iloc[::-1]
    with pytest.raises(ValueError, match="triées"):
        RiskTargeted(make_params()).target_weights(closes)


def test_weights_have_no_look_ahead() -> None:
    """Modifier les clôtures après t ne change aucun poids jusqu'à t."""
    closes = walk({"A": 0.01, "B": 0.005, "C": 0.0}, {"A": 0.01, "B": 0.02, "C": 0.015})
    strategy = RiskTargeted(make_params(regime_filter=True, risky_symbols=["A", "B"]))
    cut = 40
    shocked = closes.copy()
    shocked.iloc[cut + 1 :] *= np.linspace(0.3, 3.0, len(closes) - cut - 1)[:, None]

    before = strategy.target_weights(closes).iloc[: cut + 1]
    after = strategy.target_weights(shocked).iloc[: cut + 1]

    pd.testing.assert_frame_equal(before, after)


def test_equal_weight_targets_each_long_signal() -> None:
    closes = walk({"A": 0.01, "B": -0.01}, {"A": 0.002, "B": 0.002}, sessions=10)
    signals = MovingAverageCrossover(MovingAverageParams(fast_window=2, slow_window=3))
    strategy = EqualWeight(signals, weight=0.25, warm_up=3)
    weights = strategy.target_weights(closes)
    assert set(np.unique(weights.to_numpy())) <= {0.0, 0.25}
    assert weights["A"].iloc[-1] == 0.25
    assert weights["B"].iloc[-1] == 0.0
    assert strategy.max_weight == 0.25


def test_factory_builds_the_configured_strategy() -> None:
    moving = MovingAverageSettings(
        name="moving_average", params=MovingAverageParams(fast_window=5, slow_window=20)
    )
    built = build_strategy(moving, ["A", "B", "C", "D"])
    assert isinstance(built, EqualWeight)
    assert built.weight == 0.25
    assert built.warm_up == 20
    assert rebalance_policy(moving) is None

    rebalance = RebalanceConfig(frequency="monthly", band_pct=0.02)
    targeted = RiskTargetedSettings(name="risk_targeted", params=make_params(), rebalance=rebalance)
    assert isinstance(build_strategy(targeted, ["A"]), RiskTargeted)
    assert rebalance_policy(targeted) == rebalance
