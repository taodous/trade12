import numpy as np
import pandas as pd
import pytest

from bot.config import MovingAverageParams
from bot.strategy.base import FLAT, LONG, Strategy
from bot.strategy.moving_average import MovingAverageCrossover

STRATEGY: Strategy = MovingAverageCrossover(MovingAverageParams(fast_window=2, slow_window=3))
LONGER_STRATEGY: Strategy = MovingAverageCrossover(
    MovingAverageParams(fast_window=5, slow_window=20)
)


def make_closes(values: dict[str, list[float]]) -> pd.DataFrame:
    length = len(next(iter(values.values())))
    index = pd.bdate_range("2024-01-02", periods=length, name="session")
    return pd.DataFrame(values, index=index)


def random_walk(seed: int = 0, sessions: int = 300) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    returns = rng.normal(0, 0.01, (sessions, 2))
    index = pd.bdate_range("2024-01-02", periods=sessions, name="session")
    return pd.DataFrame(100 * np.exp(np.cumsum(returns, axis=0)), index=index, columns=["A", "B"])


def test_signals_match_hand_computed_values() -> None:
    # Rapide (2) : -, 1.5, 2.5, 3.5, 3.5, 2.5, 1.5. Lente (3) : -, -, 2, 3, 3.33, 3, 2.
    closes = make_closes({"A": [1, 2, 3, 4, 3, 2, 1]})
    assert list(STRATEGY.generate(closes)["A"]) == [
        FLAT,
        FLAT,
        LONG,
        LONG,
        LONG,
        FLAT,
        FLAT,
    ]


def test_flat_until_slow_average_is_available() -> None:
    closes = make_closes({"A": [float(i) for i in range(1, 31)]})
    signals = LONGER_STRATEGY.generate(closes)["A"]
    assert (signals.iloc[:19] == FLAT).all()
    assert (signals.iloc[19:] == LONG).all()


def test_flat_when_averages_are_equal() -> None:
    assert (STRATEGY.generate(make_closes({"A": [100.0] * 10}))["A"] == FLAT).all()


def test_flat_in_downtrend() -> None:
    closes = make_closes({"A": [float(i) for i in range(30, 0, -1)]})
    assert (LONGER_STRATEGY.generate(closes)["A"] == FLAT).all()


def test_symbols_are_independent() -> None:
    rising = [float(i) for i in range(1, 11)]
    falling = rising[::-1]
    signals = STRATEGY.generate(make_closes({"UP": rising, "DOWN": falling}))
    assert signals["UP"].iloc[-1] == LONG
    assert signals["DOWN"].iloc[-1] == FLAT
    pd.testing.assert_frame_equal(
        signals[["UP"]], STRATEGY.generate(make_closes({"UP": rising})), check_freq=False
    )


def test_output_has_same_shape_and_only_known_values() -> None:
    closes = random_walk()
    signals = LONGER_STRATEGY.generate(closes)
    assert signals.index.equals(closes.index)
    assert list(signals.columns) == list(closes.columns)
    assert set(np.unique(signals.to_numpy())) <= {FLAT, LONG}


@pytest.mark.parametrize("cut", [1, 19, 20, 21, 100, 299])
def test_past_signals_do_not_change_when_data_is_added(cut: int) -> None:
    closes = random_walk()
    full = LONGER_STRATEGY.generate(closes)
    truncated = LONGER_STRATEGY.generate(closes.iloc[:cut])
    pd.testing.assert_frame_equal(truncated, full.iloc[:cut], check_freq=False)


def test_past_signals_do_not_depend_on_future_prices() -> None:
    closes = random_walk()
    altered = closes.copy()
    altered.iloc[150:] = altered.iloc[150:] * 5
    original_signals = LONGER_STRATEGY.generate(closes)
    altered_signals = LONGER_STRATEGY.generate(altered)
    pd.testing.assert_frame_equal(
        altered_signals.iloc[:150], original_signals.iloc[:150], check_freq=False
    )
    assert not altered_signals.equals(original_signals)


@pytest.mark.parametrize("factor", [0.5, 0.849, 3.0])
def test_signals_do_not_depend_on_price_scale(factor: float) -> None:
    closes = random_walk()
    pd.testing.assert_frame_equal(
        LONGER_STRATEGY.generate(closes * factor),
        LONGER_STRATEGY.generate(closes),
        check_freq=False,
    )


def test_missing_price_gives_flat_until_windows_are_complete() -> None:
    values = [float(i) for i in range(1, 11)]
    values[5] = float("nan")
    signals = STRATEGY.generate(make_closes({"A": values}))["A"]
    # Le prix manquant sort de la fenêtre lente trois séances plus tard.
    assert list(signals.iloc[4:9]) == [LONG, FLAT, FLAT, FLAT, LONG]


def test_missing_prices_are_not_filled_from_other_sessions() -> None:
    closes = make_closes({"A": [1.0, 2.0, float("nan"), float("nan"), float("nan"), 6.0]})
    assert list(STRATEGY.generate(closes)["A"]) == [FLAT] * 6


def test_does_not_modify_input() -> None:
    closes = random_walk()
    before = closes.copy()
    LONGER_STRATEGY.generate(closes)
    pd.testing.assert_frame_equal(closes, before)


def test_is_deterministic() -> None:
    closes = random_walk()
    pd.testing.assert_frame_equal(
        LONGER_STRATEGY.generate(closes), LONGER_STRATEGY.generate(closes)
    )


def test_rejects_unsorted_sessions() -> None:
    with pytest.raises(ValueError, match="triées"):
        STRATEGY.generate(random_walk().iloc[::-1])


def test_rejects_duplicate_sessions() -> None:
    closes = random_walk()
    with pytest.raises(ValueError, match="uniques"):
        STRATEGY.generate(pd.concat([closes.iloc[:5], closes.iloc[4:6]]))
