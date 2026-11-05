import json
from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd
import pytest

from audit import report as report_module
from audit.backtest import run_in_sample
from audit.execution_gap import ExecutionGapReport
from audit.metrics import DeflatedSharpe, PerformanceMetrics, Status
from audit.monte_carlo import Distribution, MonteCarloReport
from audit.report import (
    AuditCriterion,
    AuditReport,
    Axis,
    Severity,
    Verdict,
    operational_criteria,
    out_of_sample_missing,
    performance_criteria,
    robustness_criteria,
    run_audit,
    worst_verdict,
)
from audit.sabotage import Outcome, SabotageReport, SabotageResult
from audit.sensitivity import CostStress, SensitivityReport
from audit.stress_tests import Coverage, MarkedTestResult, OperationalReport
from audit.walk_forward import WalkForwardReport
from bot.config import ConfigError, StrategyConfig, load_strategy_config
from tests.test_audit_backtest import fill_store, make_config

NOW = datetime(2026, 11, 4, tzinfo=UTC)
CONFIG = load_strategy_config()
THRESHOLDS = CONFIG.audit.thresholds
TARGETS = CONFIG.targets
# Sharpe minimal sous slippage majoré, dérivé des seuils : ne pas l'écrire en dur.
STRESSED_SHARPE = THRESHOLDS.min_stressed_sharpe_ratio * TARGETS.min_sharpe


def criterion(
    status: Status, severity: Severity = Severity.REWORK, *, blocks: bool = True
) -> AuditCriterion:
    return AuditCriterion(
        axis=Axis.ROBUSTNESS,
        name="c",
        label="c",
        value=1.0,
        target="",
        status=status,
        severity=severity,
        blocks_when_insufficient=blocks,
    )


@pytest.mark.parametrize(
    ("item", "verdict"),
    [
        (criterion(Status.PASS), Verdict.VALIDATED),
        (criterion(Status.ALERT, Severity.REJECT), Verdict.VALIDATED),
        (criterion(Status.FAIL), Verdict.REWORK),
        (criterion(Status.FAIL, Severity.REJECT), Verdict.REJECTED),
        (criterion(Status.INSUFFICIENT, Severity.REJECT), Verdict.REWORK),
        (criterion(Status.INSUFFICIENT, blocks=False), Verdict.VALIDATED),
    ],
)
def test_criterion_verdict_rules(item: AuditCriterion, verdict: Verdict) -> None:
    assert item.verdict is verdict


def test_worst_verdict_wins() -> None:
    assert worst_verdict([]) is Verdict.VALIDATED
    assert worst_verdict([criterion(Status.PASS), criterion(Status.FAIL)]) is Verdict.REWORK
    items = [criterion(Status.FAIL), criterion(Status.FAIL, Severity.REJECT)]
    assert worst_verdict(items) is Verdict.REJECTED


def test_out_of_sample_not_opened_blocks_validation() -> None:
    assert out_of_sample_missing().verdict is Verdict.REWORK


def test_performance_criteria_reject_on_drawdown_and_suspicious_results(tmp_path: Path) -> None:
    fill_store(tmp_path)
    backtest = run_in_sample(make_config(["A", "B"], date(2022, 12, 31)), tmp_path, NOW)
    suspicious = backtest.model_copy(update={"suspicious": ["sharpe"]})

    criteria = {item.name: item for item in performance_criteria(suspicious, TARGETS)}

    assert len(criteria) == 8
    assert criteria["in_sample.max_drawdown"].severity is Severity.REJECT
    assert criteria["in_sample.sharpe"].severity is Severity.REWORK
    assert criteria["in_sample.suspicious"].verdict is Verdict.REJECTED
    assert criteria["in_sample.suspicious"].detail == "sharpe"


def metrics(sharpe: float) -> PerformanceMetrics:
    return PerformanceMetrics(
        annual_return=0.1,
        annual_volatility=0.1,
        sharpe=sharpe,
        sortino=sharpe,
        calmar=1.0,
        max_drawdown=-0.1,
        max_drawdown_sessions=10,
        positive_months=0.6,
        sessions=500,
    )


def robustness(
    *,
    drawdown: float,
    positive: float,
    efficiency: float,
    share: float,
    stressed: float,
    dsr: float,
) -> dict[str, AuditCriterion]:
    distribution = Distribution(p5=0.0, p50=0.0, p95=drawdown)
    monte_carlo = MonteCarloReport(
        simulations=10,
        trades=5,
        trade_order_drawdown=None,
        block_drawdown=distribution,
        block_annual_return=distribution,
        worst_drawdown_p95=drawdown,
    )
    walk_forward = WalkForwardReport(
        windows=[],
        fixed_positive_share=positive,
        reoptimized=metrics(1.0),
        mean_train_sharpe=1.0,
        efficiency=efficiency,
    )
    sensitivity = SensitivityReport(
        evaluation_start="2017-01-02",
        parameters=["fast_window", "slow_window"],
        grid=[],
        chosen_sharpe=1.0,
        neighbor_share=share,
        costs=[
            CostStress(
                slippage_multiplier=multiplier,
                slippage_pct=0.001 * multiplier,
                annual_return=0.1,
                sharpe=stressed,
                max_drawdown=-0.1,
            )
            for multiplier in (2.0, 3.0)
        ],
    )
    deflated = DeflatedSharpe(sharpe=1.0, trials=3, expected_max_sharpe=0.2, probability=dsr)
    items = robustness_criteria(
        monte_carlo, walk_forward, sensitivity, deflated, THRESHOLDS, TARGETS
    )
    return {item.name: item for item in items}


def test_robustness_criteria_pass_at_their_thresholds() -> None:
    items = robustness(
        drawdown=0.15, positive=0.6, efficiency=0.5, share=0.7, stressed=STRESSED_SHARPE, dsr=0.95
    )
    assert {item.status for item in items.values()} == {Status.PASS}
    assert all(item.axis is Axis.ROBUSTNESS for item in items.values())


def test_robustness_criteria_fail_beyond_thresholds_or_on_nan() -> None:
    nan = float("nan")
    items = robustness(
        drawdown=0.151,
        positive=0.5,
        efficiency=nan,
        share=nan,
        stressed=STRESSED_SHARPE - 0.01,
        dsr=0.9,
    )
    assert {item.status for item in items.values()} == {Status.FAIL}
    assert worst_verdict(list(items.values())) is Verdict.REWORK


def operational(
    sabotage: SabotageReport | None,
    fills: int = 0,
    mean: float | None = None,
    covered: bool = True,
) -> dict[str, AuditCriterion]:
    tests = ["t"] if covered else []
    report = OperationalReport(
        scenarios=[Coverage(name="crash", tests=tests, failed=[])],
        risk_limits=[Coverage(name="halted", tests=tests, failed=[])],
        unknown_markers=[],
    )
    execution = ExecutionGapReport(
        journal_found=True,
        fills=fills,
        assumed_slippage_pct=0.001,
        mean_slippage_pct=mean,
        p95_slippage_pct=mean,
    )
    items = operational_criteria(report, sabotage, "hash", execution, THRESHOLDS)
    return {item.name: item for item in items}


def sabotage_report(outcome: Outcome, source: str = "hash") -> SabotageReport:
    return SabotageReport(
        ran_at=NOW,
        source_hash=source,
        baseline_seconds=1.0,
        timeout_seconds=30.0,
        results=[SabotageResult(name="s", path="p", description="d", outcome=outcome, seconds=1.0)],
    )


def test_operational_criteria_pass_when_everything_is_covered() -> None:
    items = operational(sabotage_report(Outcome.TIMEOUT), fills=25, mean=0.0005)
    assert all(item.status is Status.PASS for item in items.values())
    assert "limite de temps" in items["sabotage.survivors"].detail


def test_uncovered_scenario_or_limit_rejects() -> None:
    items = operational(sabotage_report(Outcome.DETECTED), covered=False)
    assert items["scenario.crash"].verdict is Verdict.REJECTED
    assert items["risk_limits.triggered"].verdict is Verdict.REJECTED
    assert "halted" in items["risk_limits.triggered"].detail


def test_surviving_sabotage_rejects() -> None:
    item = operational(sabotage_report(Outcome.SURVIVED))["sabotage.survivors"]
    assert item.verdict is Verdict.REJECTED
    assert "survivants : s" in item.detail


@pytest.mark.parametrize(
    ("sabotage", "word"), [(None, "Non lancé"), (sabotage_report(Outcome.DETECTED, "x"), "Périmé")]
)
def test_missing_or_stale_sabotage_blocks_validation(
    sabotage: SabotageReport | None, word: str
) -> None:
    item = operational(sabotage)["sabotage.survivors"]
    assert item.status is Status.INSUFFICIENT
    assert item.verdict is Verdict.REWORK
    assert word in item.detail


def test_execution_gap_without_enough_fills_does_not_block() -> None:
    item = operational(sabotage_report(Outcome.DETECTED), fills=3, mean=0.01)[
        "execution_gap.slippage"
    ]
    assert item.status is Status.INSUFFICIENT
    assert item.verdict is Verdict.VALIDATED


def test_execution_gap_above_backtest_assumption_fails() -> None:
    item = operational(sabotage_report(Outcome.DETECTED), fills=25, mean=0.002)[
        "execution_gap.slippage"
    ]
    assert item.verdict is Verdict.REWORK


# Audit complet sur données synthétiques


def synthetic_config(tmp_path: Path) -> StrategyConfig:
    _, closes = fill_store(tmp_path)
    base = make_config(["A", "B"], pd.Timestamp(closes.index[200]).date())
    audit = base.audit
    return base.model_copy(
        update={
            "audit": audit.model_copy(
                update={
                    # Fenêtres rapides 3 et 5, lentes 12 et 20 autour de 5/20.
                    "sensitivity": audit.sensitivity.model_copy(
                        update={"relative_steps": [0.6, 1.0]}
                    ),
                    "walk_forward": audit.walk_forward.model_copy(
                        update={"train_sessions": 60, "test_sessions": 30}
                    ),
                    "monte_carlo": audit.monte_carlo.model_copy(update={"simulations": 50}),
                }
            )
        }
    )


@pytest.fixture
def fast_audit(monkeypatch: pytest.MonkeyPatch) -> None:
    """Évite de relancer toute la suite de tests et de produire la fiche quantstats."""
    marked = [
        MarkedTestResult(nodeid=f"t::{name}", scenarios=[name], risk_limits=[], outcome="passed")
        for name in CONFIG.audit.stress.required_scenarios
    ]
    monkeypatch.setattr(report_module, "run_marked_tests", lambda root: marked)
    monkeypatch.setattr(report_module, "_tearsheet", lambda *args: None)


@pytest.mark.usefixtures("fast_audit")
def test_run_audit_writes_reports_without_opening_out_of_sample(tmp_path: Path) -> None:
    config = synthetic_config(tmp_path)

    report = run_audit(config, tmp_path, NOW, out_of_sample=False)

    assert report.out_of_sample is None
    assert not (tmp_path / config.audit.out_of_sample_log).exists()
    names = {item.name for item in report.criteria}
    assert "out_of_sample.opened" in names
    assert "risk_limits.triggered" in names
    # Aucun test ne déclenche les motifs de risque dans ce faux résultat : l'audit doit rejeter.
    assert report.axis_verdicts[Axis.OPERATIONS] is Verdict.REJECTED
    assert report.verdict is Verdict.REJECTED
    assert any("Out-of-sample non ouvert" in limit for limit in report.limits)

    saved = json.loads((tmp_path / "reports" / "audit.json").read_text(encoding="utf-8"))
    assert saved["verdict"] == Verdict.REJECTED.value
    page = (tmp_path / "reports" / "audit.html").read_text(encoding="utf-8")
    assert "Verdict : rejetée" in page
    assert "Limites" in page
    # Le rapport se relit, NaN compris : NaN n'étant égal à rien, on compare les statuts.
    reread = AuditReport.model_validate(saved)
    assert reread.verdict is report.verdict
    assert [item.status for item in reread.criteria] == [item.status for item in report.criteria]
    assert reread.sensitivity.grid == report.sensitivity.grid


@pytest.mark.usefixtures("fast_audit")
def test_run_audit_with_out_of_sample_logs_access_and_adds_criteria(tmp_path: Path) -> None:
    config = synthetic_config(tmp_path)

    report = run_audit(config, tmp_path, NOW, out_of_sample=True)

    assert report.out_of_sample is not None
    assert report.out_of_sample.start > report.in_sample.end
    log = (tmp_path / config.audit.out_of_sample_log).read_text(encoding="utf-8").splitlines()
    assert len(log) == 1
    assert json.loads(log[0])["variant"] == report.variant
    names = {item.name for item in report.criteria}
    assert "out_of_sample.opened" not in names
    assert "out_of_sample.sharpe" in names


def test_main_passes_out_of_sample_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[bool] = []

    def fake_run(
        config: StrategyConfig, root: Path, now: datetime, *, out_of_sample: bool
    ) -> AuditReport:
        calls.append(out_of_sample)
        raise ConfigError("arrêt")

    monkeypatch.setattr(report_module, "run_audit", fake_run)
    assert report_module.main([]) == 1
    assert report_module.main(["--out-of-sample"]) == 1
    assert calls == [False, True]


@pytest.mark.usefixtures("fast_audit")
def test_main_succeeds_on_a_complete_audit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = synthetic_config(tmp_path)
    monkeypatch.setattr(report_module, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(report_module, "load_strategy_config", lambda: config)
    assert report_module.main([]) == 0
    assert "audit_done" in capsys.readouterr().out
