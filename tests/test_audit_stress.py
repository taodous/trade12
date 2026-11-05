from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from audit.pytest_plugin import FAILED, PASSED
from audit.stress_tests import (
    MarkedTestResult,
    gap_stress,
    market_scenarios,
    operational_coverage,
    required_risk_limits,
    run_marked_tests,
)
from bot.config import StressScenario, load_strategy_config


def series(values: list[float], start: str = "2020-01-01") -> pd.Series:
    return pd.Series(values, index=pd.bdate_range(start, periods=len(values)), dtype=float)


def test_market_scenario_measures_return_drawdown_and_worst_day() -> None:
    returns = series([0.0, 0.1, -0.5, 0.2, 0.0])
    benchmark = series([0.0, -0.1, 0.0, 0.0, 0.0])
    scenario = StressScenario(name="crise", start=date(2020, 1, 2), end=date(2020, 1, 6))

    [result] = market_scenarios(returns, benchmark, [scenario])

    assert result.sessions == 3
    assert result.strategy_return == pytest.approx(1.1 * 0.5 * 1.2 - 1)
    assert result.strategy_drawdown == pytest.approx(1 - 0.55 / 1.1)
    assert result.strategy_worst_day == -0.5
    assert result.benchmark_return == pytest.approx(-0.1)
    assert result.benchmark_drawdown == pytest.approx(0.1)


def test_scenario_outside_the_period_stays_empty() -> None:
    returns = series([0.01, 0.02])
    scenario = StressScenario(name="avant", start=date(2008, 9, 1), end=date(2008, 12, 31))
    [result] = market_scenarios(returns, returns, [scenario])
    assert result.sessions == 0
    assert result.strategy_return is None
    assert result.benchmark_drawdown is None


def test_gap_adds_to_the_worst_drawdown() -> None:
    limits = load_strategy_config().risk
    gap = gap_stress(max_drawdown=-0.2, max_exposure=0.5, gap_pct=0.1, limits=limits)
    assert gap.immediate_loss == pytest.approx(0.05)
    assert gap.triggers_kill_switch
    assert gap.drawdown_with_gap == pytest.approx(1 - 0.8 * 0.95)


def test_small_gap_with_low_exposure_does_not_trigger_kill_switch() -> None:
    limits = load_strategy_config().risk
    gap = gap_stress(max_drawdown=0.0, max_exposure=0.1, gap_pct=0.1, limits=limits)
    assert not gap.triggers_kill_switch


def test_required_risk_limits_cover_every_reject_and_halt_reason() -> None:
    limits = required_risk_limits()
    assert len(limits) == len(set(limits)) == 13
    assert {"order_limit", "daily_loss", "max_drawdown", "halted"} <= set(limits)


def result(nodeid: str, outcome: str = PASSED, **marks: list[str]) -> MarkedTestResult:
    return MarkedTestResult(
        nodeid=nodeid,
        scenarios=marks.get("scenarios", []),
        risk_limits=marks.get("risk_limits", []),
        outcome=outcome,
    )


def test_coverage_requires_one_passing_test_per_item() -> None:
    outcomes = [
        result("t1", scenarios=["crash"]),
        result("t2", FAILED, scenarios=["api_down"]),
        result("t3", scenarios=["api_down"], risk_limits=["halted"]),
        result("t4", scenarios=["typo"]),
    ]

    report = operational_coverage(outcomes, ["crash", "api_down", "duplicate_order"], ["halted"])

    covered = {item.name: item.covered for item in report.scenarios}
    assert covered == {"crash": True, "api_down": False, "duplicate_order": False}
    assert report.scenarios[1].failed == ["t2"]
    assert report.risk_limits[0].tests == ["t3"]
    assert report.unknown_markers == ["typo"]


def test_marked_tests_are_run_and_reported(tmp_path: Path) -> None:
    """Lance pytest pour de vrai, avec le plugin, sur un petit projet de test."""
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "test_sample.py").write_text(
        "import pytest\n\n"
        "@pytest.mark.scenario('crash')\n"
        "def test_ok():\n    assert True\n\n"
        "@pytest.mark.risk_limit('halted')\n"
        "def test_ko():\n    assert False\n\n"
        "@pytest.mark.scenario('api_down')\n"
        "@pytest.mark.skip(reason='non lancé')\n"
        "def test_skipped():\n    pass\n\n"
        "def test_unmarked():\n    assert True\n",
        encoding="utf-8",
    )

    outcomes = {item.nodeid.split("::")[-1]: item for item in run_marked_tests(tmp_path)}

    assert set(outcomes) == {"test_ok", "test_ko", "test_skipped"}
    assert outcomes["test_ok"].outcome == PASSED
    assert outcomes["test_ok"].scenarios == ["crash"]
    assert outcomes["test_ko"].outcome == FAILED
    assert outcomes["test_ko"].risk_limits == ["halted"]
    assert outcomes["test_skipped"].outcome == "skipped"


def test_no_marked_test_leaves_every_scenario_uncovered(tmp_path: Path) -> None:
    """Sans test marqué, l'audit ne doit pas conclure à une couverture complète."""
    (tmp_path / "tests").mkdir()
    outcomes = run_marked_tests(tmp_path)
    assert outcomes == []
    assert not operational_coverage(outcomes, ["crash"], []).scenarios[0].covered
