"""Rapport d'audit : performance, robustesse et sûreté opérationnelle, avec verdict par critère.

Règles du verdict, fixées avant le premier résultat :

- rejetée : drawdown au-delà du seuil de rejet, résultat suspect non investigué, scénario
  opérationnel non couvert ou en échec, motif de risque jamais déclenché, sabotage non détecté ;
- à retravailler : tout autre critère en échec, ou un critère bloquant qui manque de données
  (out-of-sample non ouvert, test de sabotage absent ou périmé) ;
- validée : aucun des cas précédents. Une alerte ne bloque pas mais reste affichée.

L'écart d'exécution avec le paper trading ne bloque pas tant que les données manquent : il ne peut
être mesuré qu'à l'étape 9. L'out-of-sample n'est ouvert qu'avec `--out-of-sample`.
"""

import argparse
import html
import math
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

import pandas as pd
import quantstats as qs
import structlog
from pydantic import BaseModel, ConfigDict

from audit.backtest import (
    BacktestError,
    BacktestReport,
    evaluate_in_sample,
    evaluate_out_of_sample,
)
from audit.execution_gap import ExecutionGapReport, measure_execution_gap
from audit.metrics import DeflatedSharpe, Status, deflated_sharpe
from audit.monte_carlo import MonteCarloReport, run_monte_carlo
from audit.registry import load_registry, trial_sharpes
from audit.sabotage import Outcome, SabotageReport, load_sabotage_report, source_hash
from audit.sensitivity import SensitivityReport, run_sensitivity
from audit.stress_tests import (
    OperationalReport,
    StressReport,
    gap_stress,
    market_scenarios,
    operational_coverage,
    required_risk_limits,
    run_marked_tests,
)
from audit.walk_forward import WalkForwardReport, run_walk_forward
from bot.config import (
    PROJECT_ROOT,
    ConfigError,
    PerformanceTargets,
    RobustnessThresholds,
    StrategyConfig,
    load_strategy_config,
)
from bot.data.store import StoreError
from bot.logs import configure_logging
from bot.strategy.factory import rebalance_policy

logger = structlog.get_logger()

REPORT_JSON = "audit.json"
REPORT_HTML = "audit.html"
TEARSHEET_HTML = "tearsheet_in_sample.html"


class Axis(StrEnum):
    """Axe du protocole d'audit."""

    PERFORMANCE = "performance"
    ROBUSTNESS = "robustesse"
    OPERATIONS = "sûreté opérationnelle"


class Severity(StrEnum):
    """Conséquence d'un critère en échec."""

    REJECT = "reject"
    REWORK = "rework"


class Verdict(StrEnum):
    """Verdict d'un critère, d'un axe ou de l'audit."""

    VALIDATED = "validée"
    REWORK = "à retravailler"
    REJECTED = "rejetée"


_VERDICT_RANK = {Verdict.VALIDATED: 0, Verdict.REWORK: 1, Verdict.REJECTED: 2}


class AuditCriterion(BaseModel):
    """Critère évalué, avec sa conséquence en cas d'échec."""

    model_config = ConfigDict(frozen=True, ser_json_inf_nan="strings")

    axis: Axis
    name: str
    label: str
    value: float | None
    target: str
    status: Status
    severity: Severity
    blocks_when_insufficient: bool = True
    detail: str = ""

    @property
    def verdict(self) -> Verdict:
        """Retourne le verdict que ce critère impose à lui seul."""
        if self.status is Status.FAIL:
            return Verdict.REJECTED if self.severity is Severity.REJECT else Verdict.REWORK
        if self.status is Status.INSUFFICIENT and self.blocks_when_insufficient:
            return Verdict.REWORK
        return Verdict.VALIDATED


def worst_verdict(criteria: Sequence[AuditCriterion]) -> Verdict:
    """Retourne le verdict le plus sévère des critères, validée s'il n'y en a aucun."""
    verdicts = [criterion.verdict for criterion in criteria]
    return max(verdicts, key=_VERDICT_RANK.__getitem__, default=Verdict.VALIDATED)


def _at_least(value: float, target: float) -> Status:
    return Status.PASS if math.isfinite(value) and value >= target else Status.FAIL


def _at_most(value: float, target: float) -> Status:
    return Status.PASS if math.isfinite(value) and value <= target else Status.FAIL


_PERFORMANCE_LABELS = {
    "annual_return": "Rendement annualisé",
    "max_drawdown": "Drawdown maximal",
    "volatility_below_benchmark": "Volatilité inférieure au benchmark",
    "sharpe": "Sharpe",
    "calmar": "Calmar",
    "positive_months": "Mois positifs",
    "sharpe_above_benchmark": "Sharpe supérieur au benchmark",
}


def performance_criteria(
    report: BacktestReport, targets: PerformanceTargets
) -> list[AuditCriterion]:
    """Reprend les critères du backtest. Un drawdown en échec dépasse le seuil de rejet."""
    sample = "in-sample" if report.sample == "in_sample" else "out-of-sample"
    criteria = []
    for item in report.criteria:
        if item.name == "max_drawdown":
            alert, reject = targets.max_drawdown_alert, targets.max_drawdown_reject
            target = f"≤ {alert:.1%} (rejet > {reject:.1%})"
        else:
            target = f"≥ {item.target:.4g}"
        criteria.append(
            AuditCriterion(
                axis=Axis.PERFORMANCE,
                name=f"{report.sample}.{item.name}",
                label=f"{_PERFORMANCE_LABELS.get(item.name, item.name)} ({sample})",
                value=item.value,
                target=target,
                status=item.status,
                severity=Severity.REJECT if item.name == "max_drawdown" else Severity.REWORK,
            )
        )
    criteria.append(
        AuditCriterion(
            axis=Axis.PERFORMANCE,
            name=f"{report.sample}.suspicious",
            label=f"Aucun résultat suspect ({sample})",
            value=float(len(report.suspicious)),
            target=(
                f"rendement ≤ {targets.suspicious_annual_return:.0%}, "
                f"Sharpe ≤ {targets.suspicious_sharpe:g}"
            ),
            status=Status.FAIL if report.suspicious else Status.PASS,
            severity=Severity.REJECT,
            detail=", ".join(report.suspicious),
        )
    )
    return criteria


def out_of_sample_missing() -> AuditCriterion:
    """Critère bloquant tant que l'out-of-sample n'a pas été ouvert."""
    return AuditCriterion(
        axis=Axis.PERFORMANCE,
        name="out_of_sample.opened",
        label="Évaluation out-of-sample",
        value=None,
        target="ouverte avec --out-of-sample",
        status=Status.INSUFFICIENT,
        severity=Severity.REWORK,
        detail="Non ouverte : les objectifs se mesurent out-of-sample.",
    )


def robustness_criteria(
    monte_carlo: MonteCarloReport,
    walk_forward: WalkForwardReport,
    sensitivity: SensitivityReport,
    deflated: DeflatedSharpe,
    thresholds: RobustnessThresholds,
    targets: PerformanceTargets,
) -> list[AuditCriterion]:
    """Évalue les critères de robustesse statistique."""
    stressed = next(
        item
        for item in sensitivity.costs
        if item.slippage_multiplier == thresholds.stressed_slippage_multiplier
    )
    stressed_target = thresholds.min_stressed_sharpe_ratio * targets.min_sharpe

    def criterion(
        name: str, label: str, value: float, target: str, status: Status, detail: str = ""
    ) -> AuditCriterion:
        return AuditCriterion(
            axis=Axis.ROBUSTNESS,
            name=name,
            label=label,
            value=value,
            target=target,
            status=status,
            severity=Severity.REWORK,
            detail=detail,
        )

    return [
        criterion(
            "monte_carlo.drawdown_p95",
            "Drawdown Monte Carlo, percentile 95",
            monte_carlo.worst_drawdown_p95,
            f"≤ {thresholds.max_monte_carlo_drawdown_p95:.1%}",
            _at_most(monte_carlo.worst_drawdown_p95, thresholds.max_monte_carlo_drawdown_p95),
            f"pire des deux tirages, {monte_carlo.trades} trades",
        ),
        criterion(
            "walk_forward.positive_windows",
            "Fenêtres de test positives (réglage fixe)",
            walk_forward.fixed_positive_share,
            f"≥ {thresholds.min_walk_forward_positive_windows:.0%}",
            _at_least(
                walk_forward.fixed_positive_share, thresholds.min_walk_forward_positive_windows
            ),
            f"{len(walk_forward.windows)} fenêtres",
        ),
        criterion(
            "walk_forward.efficiency",
            "Efficacité du walk-forward (réglage réoptimisé)",
            walk_forward.efficiency,
            f"≥ {thresholds.min_walk_forward_efficiency:g}",
            _at_least(walk_forward.efficiency, thresholds.min_walk_forward_efficiency),
            f"Sharpe d'entraînement moyen {walk_forward.mean_train_sharpe:.2f}",
        ),
        criterion(
            "sensitivity.neighbor_share",
            "Réglages voisins stables",
            sensitivity.neighbor_share,
            f"≥ {thresholds.min_neighbor_share:.0%} à {thresholds.neighbor_sharpe_ratio:.0%} "
            "du Sharpe choisi",
            _at_least(sensitivity.neighbor_share, thresholds.min_neighbor_share),
            f"Sharpe choisi {sensitivity.chosen_sharpe:.2f}",
        ),
        criterion(
            "sensitivity.stressed_slippage",
            f"Sharpe avec slippage x{thresholds.stressed_slippage_multiplier:g}",
            stressed.sharpe,
            f"≥ {stressed_target:g}",
            _at_least(stressed.sharpe, stressed_target),
        ),
        criterion(
            "deflated_sharpe.probability",
            "Sharpe dégonflé (probabilité)",
            deflated.probability,
            f"≥ {thresholds.min_deflated_sharpe_probability:g}",
            _at_least(deflated.probability, thresholds.min_deflated_sharpe_probability),
            f"{deflated.trials} essais, Sharpe attendu du hasard "
            f"{deflated.expected_max_sharpe:.2f}",
        ),
    ]


def operational_criteria(
    operational: OperationalReport,
    sabotage: SabotageReport | None,
    current_hash: str,
    execution: ExecutionGapReport,
    thresholds: RobustnessThresholds,
) -> list[AuditCriterion]:
    """Évalue la couverture des scénarios, des limites de risque, le sabotage et l'exécution."""
    criteria = [
        AuditCriterion(
            axis=Axis.OPERATIONS,
            name=f"scenario.{item.name}",
            label=f"Scénario : {item.name}",
            value=float(len(item.tests)),
            target="≥ 1 test, tous réussis",
            status=Status.PASS if item.covered else Status.FAIL,
            severity=Severity.REJECT,
            detail=", ".join(item.failed) if item.failed else "",
        )
        for item in operational.scenarios
    ]
    missing = [item.name for item in operational.risk_limits if not item.covered]
    covered = len(operational.risk_limits) - len(missing)
    criteria.append(
        AuditCriterion(
            axis=Axis.OPERATIONS,
            name="risk_limits.triggered",
            label="Motifs de refus et d'arrêt déclenchés par un test",
            value=float(covered),
            target=f"{len(operational.risk_limits)} sur {len(operational.risk_limits)}",
            status=Status.FAIL if missing else Status.PASS,
            severity=Severity.REJECT,
            detail=f"non couverts : {', '.join(missing)}" if missing else "",
        )
    )
    criteria.append(_sabotage_criterion(sabotage, current_hash))

    enough = execution.fills >= thresholds.min_paper_fills
    mean = execution.mean_slippage_pct
    if not enough or mean is None:
        status = Status.INSUFFICIENT
    else:
        status = _at_most(mean, execution.assumed_slippage_pct)
    criteria.append(
        AuditCriterion(
            axis=Axis.OPERATIONS,
            name="execution_gap.slippage",
            label="Slippage paper moyen inférieur à celui du backtest",
            value=mean,
            target=f"≤ {execution.assumed_slippage_pct:.2%}, sur ≥ {thresholds.min_paper_fills} "
            "exécutions",
            status=status,
            severity=Severity.REWORK,
            blocks_when_insufficient=False,
            detail=f"{execution.fills} exécutions au journal",
        )
    )
    return criteria


def _sabotage_criterion(sabotage: SabotageReport | None, current_hash: str) -> AuditCriterion:
    def criterion(value: float | None, status: Status, detail: str) -> AuditCriterion:
        return AuditCriterion(
            axis=Axis.OPERATIONS,
            name="sabotage.survivors",
            label="Sabotages des contrôles détectés par les tests",
            value=value,
            target="0 survivant",
            status=status,
            severity=Severity.REJECT,
            detail=detail,
        )

    if sabotage is None:
        return criterion(None, Status.INSUFFICIENT, "Non lancé : uv run python -m audit.sabotage")
    survivors = sabotage.survivors
    if sabotage.source_hash != current_hash:
        return criterion(
            float(len(survivors)),
            Status.INSUFFICIENT,
            "Périmé : le code a changé depuis, relancer uv run python -m audit.sabotage",
        )
    timeouts = [item.name for item in sabotage.results if item.outcome is Outcome.TIMEOUT]
    details = [f"{len(sabotage.results)} sabotages"]
    if survivors:
        details.append(f"survivants : {', '.join(survivors)}")
    if timeouts:
        details.append(f"détectés par limite de temps : {', '.join(timeouts)}")
    return criterion(
        float(len(survivors)), Status.FAIL if survivors else Status.PASS, " ; ".join(details)
    )


class AuditReport(BaseModel):
    """Rapport d'audit complet.

    Comme tous les modèles de l'audit, une valeur indéfinie est écrite "NaN" en JSON, et non
    null, pour que le rapport se relise.
    """

    model_config = ConfigDict(frozen=True, ser_json_inf_nan="strings")

    generated_at: datetime
    variant: str
    in_sample: BacktestReport
    out_of_sample: BacktestReport | None
    sensitivity: SensitivityReport
    walk_forward: WalkForwardReport
    monte_carlo: MonteCarloReport
    deflated_sharpe: DeflatedSharpe
    stress: StressReport
    sabotage: SabotageReport | None
    execution_gap: ExecutionGapReport
    criteria: list[AuditCriterion]
    axis_verdicts: dict[Axis, Verdict]
    verdict: Verdict
    limits: list[str]


def build_limits(
    config: StrategyConfig,
    in_sample: BacktestReport,
    out_of_sample: BacktestReport | None,
    registered: int,
    execution: ExecutionGapReport,
) -> list[str]:
    """Énonce les limites du rapport, qui doivent accompagner tout verdict."""
    oos = (
        f"Out-of-sample évalué du {out_of_sample.start} au {out_of_sample.end}."
        if out_of_sample
        else "Out-of-sample non ouvert : aucun objectif n'est encore mesuré hors échantillon."
    )
    rebalance = rebalance_policy(config.strategy)
    rebalancing = (
        "aucun rééquilibrage"
        if rebalance is None
        else f"rééquilibrage {rebalance.frequency} complet, sans la bande de "
        f"{rebalance.band_pct:.0%} du paper (plus de frais, donc prudent)"
    )
    return [
        f"Période in-sample du {in_sample.start} au {in_sample.end}, soit "
        f"{in_sample.strategy.sessions} séances. Une seule période historique, un seul régime "
        "de taux et de marché.",
        oos,
        f"Échantillon : {in_sample.trades.closed_trades} trades clôturés. Les statistiques de "
        "trades et le Monte Carlo sur leur ordre sont grossiers avec si peu de trades.",
        f"{registered} variantes au registre. Le Sharpe dégonflé en tient compte, mais seulement "
        "des variantes inscrites.",
        f"Univers de {len(config.data.symbols)} ETF choisi en 2026 : ce sont des ETF qui ont "
        "survécu jusqu'à aujourd'hui (biais du survivant, limité pour des ETF aussi larges).",
        "Modèle de backtest : exécution à l'ouverture avec un slippage fixe, cash non rémunéré, "
        f"ni kill switch ni contrôles du module risque appliqués, {rebalancing}. Le Sharpe est "
        "calculé avec un taux sans risque nul.",
        "Paper trading : les exécutions d'Alpaca sont simulées et sous-estiment le slippage "
        "réel. Les prix temps réel viennent du flux IEX, qui ne couvre qu'une partie du marché.",
        f"Écart d'exécution mesuré sur {execution.fills} exécutions paper. La comparaison "
        "complète backtest et paper sur une même période attend l'étape 9.",
        "Les scénarios opérationnels sont vérifiés avec un faux broker. La fermeture des "
        "positions par le kill switch n'a jamais été exercée sur l'API réelle.",
    ]


def _tearsheet(report_dir: Path, returns: pd.Series, benchmark: pd.Series, title: str) -> None:
    # quantstats ne déclare pas les types de ses rapports.
    qs.reports.html(  # type: ignore[no-untyped-call]
        returns.rename("Stratégie"),
        benchmark=benchmark.rename("SPY"),
        output=str(report_dir / TEARSHEET_HTML),
        title=title,
    )


def run_audit(
    config: StrategyConfig, root: Path, now: datetime, *, out_of_sample: bool
) -> AuditReport:
    """Lance tout le protocole d'audit et écrit les rapports JSON et HTML."""
    audit = config.audit
    registry = root / config.backtest.registry_file
    evaluation = evaluate_in_sample(config, root, now)
    in_sample = evaluation.report
    logger.info("audit_step", step="backtest")

    sensitivity = run_sensitivity(config, evaluation.opens, evaluation.closes, registry, now)
    logger.info("audit_step", step="sensitivity")
    walk_forward = run_walk_forward(config, evaluation.opens, evaluation.closes)
    logger.info("audit_step", step="walk_forward")
    monte_carlo = run_monte_carlo(
        evaluation.run.returns,
        evaluation.run.closed_trade_pnl,
        config.backtest.initial_capital,
        audit.monte_carlo,
    )
    deflated = deflated_sharpe(evaluation.run.returns, trial_sharpes(registry))

    stress = StressReport(
        market=market_scenarios(
            evaluation.run.returns, evaluation.benchmark_run.returns, audit.stress.scenarios
        ),
        gap=gap_stress(
            in_sample.strategy.max_drawdown,
            in_sample.max_exposure,
            audit.stress.synthetic_gap_pct,
            config.risk,
        ),
        operational=operational_coverage(
            run_marked_tests(root), audit.stress.required_scenarios, required_risk_limits()
        ),
    )
    logger.info("audit_step", step="stress_tests")
    sabotage = load_sabotage_report(root / audit.sabotage.report_file)
    execution = measure_execution_gap(root / config.live.journal_file, config.backtest.slippage_pct)

    oos_report = None
    if out_of_sample:
        oos_report = evaluate_out_of_sample(config, root, now).report
        logger.warning("out_of_sample_opened", variant=oos_report.variant)

    criteria = performance_criteria(in_sample, config.targets)
    if oos_report is None:
        criteria.append(out_of_sample_missing())
    else:
        criteria += performance_criteria(oos_report, config.targets)
    criteria += robustness_criteria(
        monte_carlo, walk_forward, sensitivity, deflated, audit.thresholds, config.targets
    )
    criteria += operational_criteria(
        stress.operational, sabotage, source_hash(root), execution, audit.thresholds
    )

    report = AuditReport(
        generated_at=now,
        variant=in_sample.variant,
        in_sample=in_sample,
        out_of_sample=oos_report,
        sensitivity=sensitivity,
        walk_forward=walk_forward,
        monte_carlo=monte_carlo,
        deflated_sharpe=deflated,
        stress=stress,
        sabotage=sabotage,
        execution_gap=execution,
        criteria=criteria,
        axis_verdicts={
            axis: worst_verdict([item for item in criteria if item.axis is axis]) for axis in Axis
        },
        verdict=worst_verdict(criteria),
        limits=build_limits(config, in_sample, oos_report, len(load_registry(registry)), execution),
    )

    report_dir = root / audit.report_dir
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / REPORT_JSON).write_text(report.model_dump_json(indent=2), encoding="utf-8")
    (report_dir / REPORT_HTML).write_text(render_html(report), encoding="utf-8")
    _tearsheet(
        report_dir,
        evaluation.run.returns,
        evaluation.benchmark_run.returns,
        f"Stratégie {report.variant}, in-sample, nette de coûts",
    )
    return report


_STATUS_TEXT = {
    Status.PASS: "réussi",
    Status.ALERT: "alerte",
    Status.FAIL: "échec",
    Status.INSUFFICIENT: "données insuffisantes",
}
_VERDICT_CLASS = {
    Verdict.VALIDATED: "pass",
    Verdict.REWORK: "alert",
    Verdict.REJECTED: "fail",
}
_STATUS_CLASS = {
    Status.PASS: "pass",
    Status.ALERT: "alert",
    Status.FAIL: "fail",
    Status.INSUFFICIENT: "alert",
}

_STYLE = """
body { font-family: system-ui, sans-serif; max-width: 1100px; margin: 2rem auto; color: #1d1d1f;
       padding: 0 1rem; line-height: 1.45; }
h1 { margin-bottom: 0.2rem; }
table { border-collapse: collapse; width: 100%; margin: 0.8rem 0 1.6rem; font-size: 0.92rem; }
th, td { border-bottom: 1px solid #ddd; padding: 0.35rem 0.5rem; text-align: left;
         vertical-align: top; }
th { background: #f4f4f6; }
td.num { text-align: right; font-variant-numeric: tabular-nums; }
.pass { color: #126b2f; font-weight: 600; }
.alert { color: #8a5a00; font-weight: 600; }
.fail { color: #b3261e; font-weight: 600; }
.verdict { font-size: 1.4rem; padding: 0.6rem 1rem; border: 2px solid currentColor;
           display: inline-block; border-radius: 6px; margin: 0.6rem 0 1rem; }
.muted { color: #666; font-size: 0.9rem; }
"""


def _e(value: object) -> str:
    return html.escape(str(value))


def _number(value: float | None, digits: int = 3) -> str:
    if value is None or not math.isfinite(value):
        return "n/d"
    return f"{value:.{digits}f}"


def _pct(value: float | None) -> str:
    if value is None or not math.isfinite(value):
        return "n/d"
    return f"{value:.1%}"


def _table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    head = "".join(f"<th>{_e(header)}</th>" for header in headers)
    body = "".join("<tr>" + "".join(rows_cell for rows_cell in row) + "</tr>" for row in rows)
    return f"<table><tr>{head}</tr>{body}</table>"


def _cell(value: str, css: str = "") -> str:
    return f'<td class="{css}">{_e(value)}</td>' if css else f"<td>{_e(value)}</td>"


def _criteria_section(report: AuditReport) -> str:
    parts = []
    for axis in Axis:
        verdict = report.axis_verdicts[axis]
        rows = [
            [
                _cell(item.label),
                _cell(_number(item.value), "num"),
                _cell(item.target),
                _cell(_STATUS_TEXT[item.status], _STATUS_CLASS[item.status]),
                _cell(item.detail),
            ]
            for item in report.criteria
            if item.axis is axis
        ]
        parts.append(
            f'<h3>{_e(axis.value.capitalize())} : <span class="{_VERDICT_CLASS[verdict]}">'
            f"{_e(verdict.value)}</span></h3>"
            + _table(["Critère", "Valeur", "Cible", "Statut", "Détail"], rows)
        )
    return "".join(parts)


def _performance_section(report: AuditReport) -> str:
    samples = [report.in_sample] + ([report.out_of_sample] if report.out_of_sample else [])
    rows = []
    for sample in samples:
        for name, metrics in (("Stratégie", sample.strategy), ("SPY", sample.benchmark)):
            rows.append(
                [
                    _cell(f"{name} ({sample.sample})"),
                    _cell(_pct(metrics.annual_return), "num"),
                    _cell(_pct(metrics.annual_volatility), "num"),
                    _cell(_number(metrics.sharpe, 2), "num"),
                    _cell(_number(metrics.sortino, 2), "num"),
                    _cell(_number(metrics.calmar, 2), "num"),
                    _cell(_pct(metrics.max_drawdown), "num"),
                    _cell(str(metrics.max_drawdown_sessions), "num"),
                    _cell(_pct(metrics.positive_months), "num"),
                ]
            )
    trades = report.in_sample.trades
    return (
        _table(
            [
                "Série",
                "Rendement/an",
                "Volatilité",
                "Sharpe",
                "Sortino",
                "Calmar",
                "Max DD",
                "Durée DD (séances)",
                "Mois positifs",
            ],
            rows,
        )
        + f'<p class="muted">In-sample : {trades.closed_trades} trades clôturés, taux de réussite '
        f"{_pct(trades.win_rate)}, profit factor {_number(trades.profit_factor, 2)}, gain moyen "
        f"sur perte moyenne {_number(trades.average_win_over_loss, 2)}. Exposition moyenne "
        f"{_pct(report.in_sample.mean_exposure)}. Fiche détaillée : "
        f'<a href="{TEARSHEET_HTML}">{TEARSHEET_HTML}</a>.</p>'
    )


def _settings_label(params: dict[str, float]) -> str:
    return ", ".join(f"{name} {value:g}" for name, value in params.items())


def _grid_table(sensitivity: SensitivityReport) -> str:
    """Matrice du Sharpe pour deux paramètres, liste des réglages sinon."""
    if len(sensitivity.parameters) != 2:
        rows = [
            [_cell(_settings_label(point.params)), _cell(_number(point.sharpe, 2), "num")]
            for point in sensitivity.grid
        ]
        return _table(["Réglage", "Sharpe"], rows)
    row_name, column_name = sensitivity.parameters
    row_values = sorted({point.params[row_name] for point in sensitivity.grid})
    column_values = sorted({point.params[column_name] for point in sensitivity.grid})
    sharpe = {
        (point.params[row_name], point.params[column_name]): point.sharpe
        for point in sensitivity.grid
    }
    rows = [
        [_cell(f"{row_name} {row:g}")]
        + [_cell(_number(sharpe[(row, column)], 2), "num") for column in column_values]
        for row in row_values
    ]
    return _table([""] + [f"{column_name} {column:g}" for column in column_values], rows)


def _robustness_section(report: AuditReport) -> str:
    sensitivity = report.sensitivity
    costs = [
        [
            _cell(f"x{item.slippage_multiplier:g} ({item.slippage_pct:.2%})"),
            _cell(_pct(item.annual_return), "num"),
            _cell(_number(item.sharpe, 2), "num"),
            _cell(_pct(item.max_drawdown), "num"),
        ]
        for item in sensitivity.costs
    ]
    windows = [
        [
            _cell(f"{item.test_start} au {item.test_end}"),
            _cell(_settings_label(item.selected)),
            _cell(_number(item.train_sharpe, 2), "num"),
            _cell(_pct(item.test_return), "num"),
            _cell(_pct(item.fixed_test_return), "num"),
            _cell(_number(item.fixed_test_sharpe, 2), "num"),
        ]
        for item in report.walk_forward.windows
    ]
    mc = report.monte_carlo
    mc_rows = [
        [_cell(name)] + [_cell(_pct(getattr(dist, p)), "num") for p in ("p5", "p50", "p95")]
        for name, dist in (
            ("Drawdown, ordre des trades", mc.trade_order_drawdown),
            ("Drawdown, blocs de séances", mc.block_drawdown),
            ("Rendement annualisé, blocs de séances", mc.block_annual_return),
        )
        if dist is not None
    ]
    return (
        f"<h3>Sensibilité : Sharpe par réglage, depuis le {_e(sensitivity.evaluation_start)}</h3>"
        + _grid_table(sensitivity)
        + "<h3>Coûts majorés (réglage choisi)</h3>"
        + _table(["Slippage", "Rendement/an", "Sharpe", "Max DD"], costs)
        + "<h3>Walk-forward</h3>"
        + _table(
            [
                "Fenêtre de test",
                "Réglage retenu",
                "Sharpe entraînement",
                "Rendement réoptimisé",
                "Rendement fixe",
                "Sharpe fixe",
            ],
            windows,
        )
        + f"<h3>Monte Carlo ({mc.simulations} tirages)</h3>"
        + _table(["Grandeur", "p5", "p50", "p95"], mc_rows)
    )


def _operations_section(report: AuditReport) -> str:
    market = [
        [
            _cell(f"{item.name} ({item.start} au {item.end})"),
            _cell(_pct(item.strategy_return), "num"),
            _cell(_pct(item.benchmark_return), "num"),
            _cell(_pct(item.strategy_drawdown), "num"),
            _cell(_pct(item.benchmark_drawdown), "num"),
            _cell(_pct(item.strategy_worst_day), "num"),
        ]
        for item in report.stress.market
    ]
    gap = report.stress.gap
    coverage = [
        [
            _cell(item.name),
            _cell(str(len(item.tests)), "num"),
            _cell("oui" if item.covered else "non", "pass" if item.covered else "fail"),
        ]
        for item in report.stress.operational.scenarios + report.stress.operational.risk_limits
    ]
    sabotage_rows = (
        [
            [
                _cell(item.name),
                _cell(item.description),
                _cell(item.outcome.value, "fail" if item.outcome is Outcome.SURVIVED else "pass"),
            ]
            for item in report.sabotage.results
        ]
        if report.sabotage
        else []
    )
    return (
        "<h3>Crises de marché (informatif)</h3>"
        + _table(["Crise", "Stratégie", "SPY", "DD stratégie", "DD SPY", "Pire séance"], market)
        + f'<p class="muted">Gap synthétique de {_pct(gap.gap_pct)} à l\'ouverture, avec '
        f"l'exposition maximale de {_pct(gap.max_exposure)} : perte immédiate de "
        f"{_pct(gap.immediate_loss)}, kill switch "
        f"{'déclenché' if gap.triggers_kill_switch else 'non déclenché'} après coup. Ajouté "
        f"au pire drawdown, le drawdown atteindrait {_pct(gap.drawdown_with_gap)}.</p>"
        + "<h3>Couverture des scénarios et des motifs de risque</h3>"
        + _table(["Scénario ou motif", "Tests", "Couvert"], coverage)
        + (
            "<h3>Sabotages</h3>" + _table(["Sabotage", "Effet", "Résultat"], sabotage_rows)
            if sabotage_rows
            else ""
        )
    )


def render_html(report: AuditReport) -> str:
    """Retourne le rapport d'audit en HTML autonome."""
    limits = "".join(f"<li>{_e(item)}</li>" for item in report.limits)
    return (
        '<!DOCTYPE html><html lang="fr"><head><meta charset="utf-8">'
        f"<title>Audit {_e(report.variant)}</title><style>{_STYLE}</style></head><body>"
        f"<h1>Rapport d'audit</h1>"
        f'<p class="muted">Variante {_e(report.variant)}, généré le '
        f"{_e(report.generated_at.isoformat(timespec='seconds'))}</p>"
        f'<div class="verdict {_VERDICT_CLASS[report.verdict]}">Verdict : '
        f"{_e(report.verdict.value)}</div>"
        "<h2>Verdict critère par critère</h2>"
        + _criteria_section(report)
        + "<h2>Performance ajustée du risque</h2>"
        + _performance_section(report)
        + "<h2>Robustesse statistique</h2>"
        + _robustness_section(report)
        + "<h2>Sûreté opérationnelle</h2>"
        + _operations_section(report)
        + f"<h2>Limites</h2><ul>{limits}</ul></body></html>"
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Lance l'audit et retourne le code de sortie : 0 si l'audit a pu être produit."""
    parser = argparse.ArgumentParser(description="Rapport d'audit de la stratégie.")
    parser.add_argument(
        "--out-of-sample",
        action="store_true",
        help="Ouvre l'out-of-sample. Chaque ouverture est journalisée et le contamine.",
    )
    arguments = parser.parse_args(argv)
    configure_logging()
    try:
        report = run_audit(
            load_strategy_config(),
            PROJECT_ROOT,
            datetime.now(UTC),
            out_of_sample=arguments.out_of_sample,
        )
    except (ConfigError, StoreError, BacktestError, RuntimeError) as exc:
        logger.error("audit_failed", reason=str(exc))
        return 1
    logger.info(
        "audit_done",
        verdict=report.verdict.value,
        axes={axis.value: verdict.value for axis, verdict in report.axis_verdicts.items()},
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
