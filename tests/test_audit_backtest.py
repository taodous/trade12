import json
from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from audit import backtest as backtest_module
from audit.backtest import (
    BacktestError,
    BacktestRun,
    evaluate_in_sample,
    evaluate_out_of_sample,
    executed_targets,
    load_in_sample,
    order_targets,
    run_buy_and_hold,
    run_in_sample,
    run_strategy,
    run_weights,
    variant_definition,
)
from audit.registry import record_variant, variant_id
from bot.config import (
    BacktestConfig,
    ConfigError,
    DataConfig,
    MovingAverageParams,
    MovingAverageSettings,
    RebalanceConfig,
    StrategyConfig,
    load_strategy_config,
)
from bot.data.store import BarStore
from bot.strategy.base import FLAT, LONG, Strategy
from bot.strategy.equal_weight import EqualWeight
from bot.strategy.moving_average import MovingAverageCrossover

NOW = datetime(2026, 11, 3, tzinfo=UTC)
NO_COSTS = BacktestConfig(
    in_sample_end=date(2022, 12, 31),
    initial_capital=1000.0,
    fees_pct=0.0,
    slippage_pct=0.0,
    benchmark="A",
    report_file="reports/report.json",
    registry_file="audit/variants.jsonl",
)
COSTS = NO_COSTS.model_copy(update={"fees_pct": 0.001, "slippage_pct": 0.01})


class ScriptedStrategy:
    """Stratégie dont les signaux sont écrits à la main."""

    def __init__(self, signals: pd.DataFrame) -> None:
        self.signals = signals

    def generate(self, closes: pd.DataFrame) -> pd.DataFrame:
        return self.signals.loc[closes.index]


MONTHLY = RebalanceConfig(frequency="monthly", band_pct=0.02)
REFERENCE = MovingAverageSettings(
    name="moving_average", params=MovingAverageParams(fast_window=50, slow_window=200)
)


def run_signals(
    strategy: Strategy,
    opens: pd.DataFrame,
    closes: pd.DataFrame,
    weight: float,
    start: int,
    config: BacktestConfig,
) -> BacktestRun:
    """Exécute une stratégie à signaux à poids égal, sans rééquilibrage."""
    equal = EqualWeight(strategy, weight=weight, warm_up=start)
    return run_strategy(equal, opens, closes, start, config, None)


def compound(returns: pd.Series) -> float:
    return float(np.prod(1 + returns.to_numpy()))


def frame(values: dict[str, list[float]]) -> pd.DataFrame:
    length = len(next(iter(values.values())))
    index = pd.bdate_range("2024-01-02", periods=length, name="session")
    return pd.DataFrame(values, index=index, dtype=float)


def random_prices(sessions: int = 300, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2016-01-04", periods=sessions, name="session")
    closes = pd.DataFrame(
        100 * np.exp(np.cumsum(rng.normal(0, 0.01, (sessions, 2)), axis=0)),
        index=index,
        columns=["A", "B"],
    )
    opens = closes.shift(1).fillna(closes.iloc[0]) * (1 + rng.normal(0, 0.002, (sessions, 2)))
    return opens, closes


def scripted_orders(signals: pd.DataFrame, weight: float) -> pd.Series:
    strategy = EqualWeight(ScriptedStrategy(signals), weight=weight, warm_up=0)
    return order_targets(executed_targets(strategy, signals), 0, None)["A"]


def test_target_weights_shift_signal_by_one_session() -> None:
    weights = scripted_orders(frame({"A": [FLAT, LONG, LONG, FLAT, FLAT]}), 0.5)
    assert weights.isna().tolist() == [True, True, False, True, False]
    assert weights.iloc[2] == 0.5
    assert weights.iloc[4] == 0.0


def test_target_weights_never_order_on_signal_session() -> None:
    weights = scripted_orders(frame({"A": [LONG, LONG, LONG]}), 1.0)
    assert weights.isna().tolist() == [True, False, True]


def test_fill_happens_at_next_open_and_matches_hand_computation() -> None:
    opens = frame({"A": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0]})
    closes = opens + 0.5
    signals = frame({"A": [LONG, LONG, FLAT, FLAT, FLAT, FLAT]})

    run = run_signals(ScriptedStrategy(signals), opens, closes, 0.5, 0, NO_COSTS)

    # Achat de 500 $ à 11 après le signal de la séance 0, vente à 13 après celui de la séance 2.
    expected_final = 1000 - 500 + 500 / 11 * 13
    assert compound(run.returns) * 1000 == pytest.approx(expected_final)
    assert run.orders == 2
    assert run.closed_trade_pnl.tolist() == pytest.approx([500 / 11 * 2])


def test_price_move_on_signal_session_is_not_captured() -> None:
    """Une hausse de 100 % le jour du signal ne doit rien rapporter : l'achat vient après."""
    opens = frame({"A": [10.0, 20.0, 20.0, 20.0]})
    closes = frame({"A": [20.0, 20.0, 20.0, 20.0]})
    signals = frame({"A": [LONG, LONG, LONG, LONG]})
    run = run_signals(ScriptedStrategy(signals), opens, closes, 1.0, 0, NO_COSTS)
    assert compound(run.returns) == pytest.approx(1.0)


def test_gap_at_execution_open_is_not_captured() -> None:
    """Le signal de la veille achète après le gap d'ouverture, pas avant."""
    opens = frame({"A": [10.0, 15.0, 15.0]})
    closes = frame({"A": [10.0, 15.0, 15.0]})
    signals = frame({"A": [LONG, LONG, LONG]})
    run = run_signals(ScriptedStrategy(signals), opens, closes, 1.0, 0, NO_COSTS)
    assert compound(run.returns) == pytest.approx(1.0)


def test_costs_match_hand_computation() -> None:
    opens = frame({"A": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0]})
    closes = opens + 0.5
    signals = frame({"A": [LONG, LONG, FLAT, FLAT, FLAT, FLAT]})

    run = run_signals(ScriptedStrategy(signals), opens, closes, 0.5, 0, COSTS)

    shares = 500 / 11
    cost_in = shares * 11 * 1.01 * 1.001
    proceeds = shares * 13 * 0.99 * 0.999
    assert compound(run.returns) * 1000 == pytest.approx(1000 - cost_in + proceeds)


def test_costs_reduce_returns() -> None:
    opens, closes = random_prices()
    strategy = MovingAverageCrossover(MovingAverageParams(fast_window=5, slow_window=20))
    gross = run_signals(strategy, opens, closes, 0.5, 20, NO_COSTS)
    net = run_signals(strategy, opens, closes, 0.5, 20, COSTS)
    assert gross.orders == net.orders
    assert compound(net.returns) < compound(gross.returns)


def test_past_results_do_not_change_when_data_is_added() -> None:
    opens, closes = random_prices()
    strategy = MovingAverageCrossover(MovingAverageParams(fast_window=5, slow_window=20))
    full = run_signals(strategy, opens, closes, 0.5, 20, COSTS)
    cut = run_signals(strategy, opens.iloc[:150], closes.iloc[:150], 0.5, 20, COSTS)
    pd.testing.assert_series_equal(
        cut.returns, full.returns.iloc[: len(cut.returns)], check_freq=False
    )


def test_exposure_never_exceeds_capital() -> None:
    opens, closes = random_prices()
    signals = pd.DataFrame(LONG, index=closes.index, columns=closes.columns)
    run = run_signals(ScriptedStrategy(signals), opens, closes, 0.5, 0, COSTS)
    assert run.max_exposure <= 1.0
    assert run.open_trades == 2


def test_buy_and_hold_starts_at_warm_up_end() -> None:
    # Un achat dès la première séance capterait en plus le gap de 40 à 50 de la séance 2.
    opens = frame({"A": [10.0, 20.0, 50.0, 50.0]})
    closes = frame({"A": [20.0, 40.0, 50.0, 100.0]})
    run = run_buy_and_hold(opens["A"], closes["A"], 2, NO_COSTS)
    assert len(run.returns) == 2
    assert compound(run.returns) == pytest.approx(2.0)


def with_hole(prices: pd.DataFrame) -> pd.DataFrame:
    holed = prices.copy()
    holed.iloc[3, 0] = np.nan
    return holed


def test_invalid_prices_are_rejected() -> None:
    opens, closes = random_prices(30)
    signals = pd.DataFrame(FLAT, index=closes.index, columns=closes.columns)
    strategy = ScriptedStrategy(signals)
    for bad_opens, bad_closes in [
        (with_hole(opens), closes),
        (opens, with_hole(closes)),
        (opens.iloc[1:], closes),
        (opens.iloc[:5], closes.iloc[:5]),
    ]:
        with pytest.raises(BacktestError):
            run_signals(strategy, bad_opens, bad_closes, 0.5, 20, NO_COSTS)


def make_config(symbols: list[str], in_sample_end: date) -> StrategyConfig:
    base = load_strategy_config()
    return base.model_copy(
        update={
            "risk": base.risk.model_copy(update={"max_position_pct": 0.5}),
            "strategy": MovingAverageSettings(
                name="moving_average", params=MovingAverageParams(fast_window=5, slow_window=20)
            ),
            "data": DataConfig(
                symbols=symbols,
                start_date=date(2016, 1, 4),
                end_date=None,
                feed="sip",
                adjustment="all",
                min_bar_age_minutes=15,
                directory="data/bars",
            ),
            "backtest": NO_COSTS.model_copy(
                update={"in_sample_end": in_sample_end, "benchmark": symbols[0]}
            ),
        }
    )


def fill_store(path: Path, sessions: int = 300) -> tuple[pd.DataFrame, pd.DataFrame]:
    opens, closes = random_prices(sessions)
    store = BarStore(path / "data" / "bars")
    for symbol in ["A", "B"]:
        bars = pd.DataFrame(
            {
                "open": opens[symbol],
                "high": np.maximum(opens[symbol], closes[symbol]),
                "low": np.minimum(opens[symbol], closes[symbol]),
                "close": closes[symbol],
                "volume": 1000.0,
            }
        )
        store.save(symbol, bars, feed="sip", adjustment="all", fetched_at=NOW, issues=[])
    return opens, closes


def test_load_in_sample_excludes_every_later_session(tmp_path: Path) -> None:
    _, closes = fill_store(tmp_path)
    end = pd.Timestamp(closes.index[200]).date()
    config = make_config(["A", "B"], end)

    opens_in, closes_in = load_in_sample(BarStore(tmp_path / "data" / "bars"), config)

    assert closes_in.index.max() == pd.Timestamp(end)
    assert opens_in.index.max() == pd.Timestamp(end)
    assert len(closes_in) == 201


def test_out_of_sample_prices_cannot_influence_the_report(tmp_path: Path) -> None:
    """Modifier toutes les séances après la coupure ne doit rien changer au rapport."""
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    _, closes = fill_store(first_root)
    end = pd.Timestamp(closes.index[200]).date()
    fill_store(second_root)
    store = BarStore(second_root / "data" / "bars")
    for symbol in ["A", "B"]:
        bars = store.load(symbol)
        bars.loc[bars.index > pd.Timestamp(end), ["open", "high", "low", "close"]] *= 3
        store.save(
            symbol, bars, feed="sip", adjustment="all", fetched_at=NOW, issues=[], overwrite=True
        )
    config = make_config(["A", "B"], end)

    first = run_in_sample(config, first_root, NOW)
    second = run_in_sample(config, second_root, NOW)

    assert first == second


def test_run_in_sample_writes_report_and_registry(tmp_path: Path) -> None:
    fill_store(tmp_path)
    config = make_config(["A", "B"], date(2022, 12, 31))

    report = run_in_sample(config, tmp_path, NOW)

    saved = json.loads((tmp_path / "reports" / "report.json").read_text(encoding="utf-8"))
    assert saved["sample"] == "in_sample"
    assert saved["variant"] == report.variant
    assert len(report.criteria) == 7
    lines = (tmp_path / "audit" / "variants.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["id"] == report.variant


def test_evaluation_starts_when_first_order_is_possible(tmp_path: Path) -> None:
    _, closes = fill_store(tmp_path)
    config = make_config(["A", "B"], date(2022, 12, 31))
    report = run_in_sample(config, tmp_path, NOW)
    assert report.start == pd.Timestamp(closes.index[20]).date()
    assert report.strategy.sessions == report.benchmark.sessions == len(closes) - 20


def test_weight_above_position_limit_is_refused(tmp_path: Path) -> None:
    config = make_config(["A", "B"], date(2022, 12, 31))
    fill_store(tmp_path)
    config = config.model_copy(
        update={"risk": config.risk.model_copy(update={"max_position_pct": 0.4})}
    )
    with pytest.raises(BacktestError, match="Poids"):
        run_in_sample(config, tmp_path, NOW)


def test_registry_counts_a_variant_once(tmp_path: Path) -> None:
    path = tmp_path / "variants.jsonl"
    definition = {"strategy": "X", "params": {"fast": 1}}
    assert record_variant(path, definition, {"sharpe": 1.0}, NOW) is True
    assert record_variant(path, definition, {"sharpe": 9.0}, NOW) is False
    assert record_variant(path, {**definition, "params": {"fast": 2}}, {"sharpe": 2.0}, NOW)
    entries = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(entries) == 2
    assert entries[0]["results"]["sharpe"] == 1.0


def test_variant_id_ignores_key_order() -> None:
    assert variant_id({"a": 1, "b": 2}) == variant_id({"b": 2, "a": 1})
    assert variant_id({"a": 1}) != variant_id({"a": 2})


def test_main_reports_missing_data(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(backtest_module, "PROJECT_ROOT", tmp_path)
    assert backtest_module.main() == 1
    assert "backtest_failed" in capsys.readouterr().out


def test_main_reports_invalid_config(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def fail() -> StrategyConfig:
        raise ConfigError("invalide")

    monkeypatch.setattr(backtest_module, "load_strategy_config", fail)
    assert backtest_module.main() == 1
    assert "invalide" in capsys.readouterr().out


def test_project_split_precedes_end_of_data() -> None:
    config = load_strategy_config()
    assert config.backtest.in_sample_end == date(2022, 12, 31)
    assert config.backtest.benchmark in config.data.symbols


def test_orders_start_with_current_positions() -> None:
    executed = frame({"A": [0.5, 0.5, 0.5, 0.0, 0.0], "B": [0.0, 0.5, 0.5, 0.5, 0.5]})
    weights = order_targets(executed, 2, None)
    assert weights.iloc[:2].isna().all().all()
    assert weights.iloc[2].tolist() == [0.5, 0.5]
    assert weights["A"].iloc[3] == 0.0
    assert weights["B"].iloc[3:].isna().all()


@pytest.mark.parametrize("start", [-1, 5])
def test_orders_refuse_start_outside_sessions(start: int) -> None:
    executed = frame({"A": [0.5, 0.5, 0.5, 0.0, 0.0]})
    with pytest.raises(BacktestError, match="départ"):
        order_targets(executed, start, None)


def test_monthly_rebalance_orders_every_symbol_on_first_session_of_month() -> None:
    index = pd.bdate_range("2024-01-29", "2024-03-05", name="session")
    executed = pd.DataFrame(0.3, index=index, columns=["A", "B"])
    weights = order_targets(executed, 0, MONTHLY)
    ordered = [day.date().isoformat() for day in weights.dropna().index]
    assert ordered == ["2024-01-29", "2024-02-01", "2024-03-01"]
    assert (weights.dropna() == 0.3).all().all()


def test_monthly_rebalance_restores_drifted_weights() -> None:
    """A double en janvier : le rééquilibrage de février le ramène à son poids visé."""
    index = pd.bdate_range("2024-01-02", "2024-02-29", name="session")
    prices = pd.DataFrame({"A": np.linspace(100, 200, len(index)), "B": 100.0}, index=index)
    targets = pd.DataFrame(0.4, index=index, columns=["A", "B"])
    without = run_weights(prices, prices, order_targets(targets, 0, None), 0, NO_COSTS)
    with_rebalance = run_weights(prices, prices, order_targets(targets, 0, MONTHLY), 0, NO_COSTS)
    assert with_rebalance.orders > without.orders


class FixedWeights:
    """Stratégie à poids écrits à la main, pour tester leur validation."""

    def __init__(self, weights: pd.DataFrame, max_weight: float = 1.0) -> None:
        self.weights = weights
        self.max_weight = max_weight
        self.warm_up = 0
        self.required_symbols: frozenset[str] = frozenset()

    def target_weights(self, closes: pd.DataFrame) -> pd.DataFrame:
        return self.weights


@pytest.mark.parametrize(
    ("values", "cap", "message"),
    [
        ([0.1, float("nan")], 1.0, "manquant"),
        ([0.1, -0.1], 1.0, "négatif"),
        ([0.1, 0.3], 0.2, "plafond"),
        ([0.6, 0.6], 1.0, "capital"),
    ],
)
def test_invalid_strategy_weights_are_refused(
    values: list[float], cap: float, message: str
) -> None:
    closes = frame({"A": [1.0], "B": [1.0]})
    weights = pd.DataFrame([values], index=closes.index, columns=closes.columns)
    with pytest.raises(BacktestError, match=message):
        executed_targets(FixedWeights(weights, cap), closes)


def test_misaligned_strategy_weights_are_refused() -> None:
    closes = frame({"A": [1.0, 2.0]})
    weights = pd.DataFrame({"A": [0.1]}, index=closes.index[:1])
    with pytest.raises(BacktestError, match="alignés"):
        executed_targets(FixedWeights(weights), closes)


def test_run_weights_refuses_misaligned_weights() -> None:
    opens, closes = random_prices(60)
    strategy = EqualWeight(
        MovingAverageCrossover(MovingAverageParams(fast_window=5, slow_window=20)), 0.5, 20
    )
    weights = order_targets(executed_targets(strategy, closes), 0, None)
    with pytest.raises(BacktestError, match="alignés"):
        run_weights(opens, closes, weights.iloc[:-1], 20, NO_COSTS)


def test_reference_variant_keeps_its_stage_7_definition() -> None:
    """Changer cette forme changerait l'identifiant des variantes déjà inscrites au registre."""
    config = load_strategy_config().model_copy(update={"strategy": REFERENCE})
    definition = variant_definition(config)
    assert set(definition) == {
        "strategy",
        "params",
        "symbols",
        "start_date",
        "in_sample_end",
        "fees_pct",
        "slippage_pct",
    }
    assert definition["strategy"] == "MovingAverageCrossover"
    assert variant_id(definition) == "9446033063a5"


def test_configured_strategy_is_the_selected_candidate() -> None:
    config = load_strategy_config()
    assert config.strategy == config.candidates[0]
    assert variant_id(variant_definition(config)) == "aa32f0c7ab3e"


def test_rebalanced_variant_includes_its_rebalance_policy() -> None:
    config = load_strategy_config()
    candidate = config.candidates[0]
    definition = variant_definition(config, candidate)
    assert definition["strategy"] == "RiskTargeted"
    assert definition["rebalance"] == {"frequency": "monthly", "band_pct": 0.02}


def test_in_sample_report_can_be_written_elsewhere(tmp_path: Path) -> None:
    fill_store(tmp_path)
    config = make_config(["A", "B"], date(2022, 12, 31))
    target = tmp_path / "ailleurs" / "report.json"
    evaluation = evaluate_in_sample(config, tmp_path, NOW, target)
    assert json.loads(target.read_text(encoding="utf-8"))["variant"] == evaluation.report.variant
    assert not (tmp_path / "reports" / "report.json").exists()


def test_out_of_sample_starts_empty_after_the_split_and_is_logged(tmp_path: Path) -> None:
    _, closes = fill_store(tmp_path)
    end = pd.Timestamp(closes.index[200]).date()
    config = make_config(["A", "B"], end)

    evaluation = evaluate_out_of_sample(config, tmp_path, NOW)

    report = evaluation.report
    assert report.sample == "out_of_sample"
    assert report.start == pd.Timestamp(closes.index[201]).date()
    assert report.strategy.sessions == report.benchmark.sessions == len(closes) - 201
    log = (tmp_path / config.audit.out_of_sample_log).read_text(encoding="utf-8").splitlines()
    assert json.loads(log[0]) == {"opened_at": NOW.isoformat(), "variant": report.variant}
    assert (tmp_path / "reports" / "backtest_out_of_sample.json").exists()
    assert not (tmp_path / "audit" / "variants.jsonl").exists()


def test_out_of_sample_needs_sessions_after_the_split(tmp_path: Path) -> None:
    fill_store(tmp_path)
    config = make_config(["A", "B"], date(2030, 1, 1))
    with pytest.raises(BacktestError, match="Aucune séance"):
        evaluate_out_of_sample(config, tmp_path, NOW)


def test_out_of_sample_needs_history_for_the_signal(tmp_path: Path) -> None:
    _, closes = fill_store(tmp_path)
    config = make_config(["A", "B"], pd.Timestamp(closes.index[5]).date())
    with pytest.raises(BacktestError, match="Historique"):
        evaluate_out_of_sample(config, tmp_path, NOW)
