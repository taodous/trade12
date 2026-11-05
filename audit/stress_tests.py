"""Tests de stress : crises de marché sur le backtest et scénarios opérationnels du bot.

Les scénarios de marché sont informatifs : la performance pendant une crise est déjà comprise
dans le drawdown in-sample. Les scénarios opérationnels sont bloquants : chacun doit être couvert
par au moins un test marqué, et tous ces tests doivent passer. De même, chaque motif de refus et
chaque motif d'arrêt du module risque doit être déclenché par au moins un test.

Un marqueur prouve qu'un test vise un scénario, pas qu'il le vérifie bien : c'est le rôle du test
de sabotage (`audit.sabotage`).
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict

from audit.pytest_plugin import OUTPUT_OPTION, PASSED, RISK_LIMIT_MARKER, SCENARIO_MARKER
from bot.config import RiskLimits, StressScenario
from bot.risk.models import HaltReason, RejectReason


class ScenarioResult(BaseModel):
    """Comportement de la stratégie et du benchmark pendant une crise historique."""

    model_config = ConfigDict(frozen=True, ser_json_inf_nan="strings")

    name: str
    start: str
    end: str
    sessions: int
    strategy_return: float | None
    benchmark_return: float | None
    strategy_drawdown: float | None
    benchmark_drawdown: float | None
    strategy_worst_day: float | None


class GapStress(BaseModel):
    """Gap synthétique sur tout l'univers, avec l'exposition et le drawdown maximaux."""

    model_config = ConfigDict(frozen=True, ser_json_inf_nan="strings")

    gap_pct: float
    max_exposure: float
    immediate_loss: float
    triggers_kill_switch: bool
    drawdown_with_gap: float


class MarkedTestResult(BaseModel):
    """Résultat d'un test marqué."""

    model_config = ConfigDict(frozen=True, ser_json_inf_nan="strings")

    nodeid: str
    scenarios: list[str]
    risk_limits: list[str]
    outcome: str


class Coverage(BaseModel):
    """Couverture d'un scénario ou d'un motif de risque par les tests."""

    model_config = ConfigDict(frozen=True, ser_json_inf_nan="strings")

    name: str
    tests: list[str]
    failed: list[str]

    @property
    def covered(self) -> bool:
        """Indique si au moins un test couvre l'élément et si tous ses tests passent."""
        return bool(self.tests) and not self.failed


class OperationalReport(BaseModel):
    """Couverture des scénarios opérationnels et des limites de risque."""

    model_config = ConfigDict(frozen=True, ser_json_inf_nan="strings")

    scenarios: list[Coverage]
    risk_limits: list[Coverage]
    unknown_markers: list[str]


class StressReport(BaseModel):
    """Résultat des tests de stress."""

    model_config = ConfigDict(frozen=True, ser_json_inf_nan="strings")

    market: list[ScenarioResult]
    gap: GapStress
    operational: OperationalReport


def _drawdown(returns: pd.Series) -> float:
    equity = np.concatenate([[1.0], np.cumprod(1 + returns.to_numpy())])
    return float(np.max(1 - equity / np.maximum.accumulate(equity)))


def _total(returns: pd.Series) -> float:
    return float(np.prod(1 + returns.to_numpy()) - 1)


def market_scenarios(
    returns: pd.Series, benchmark: pd.Series, scenarios: list[StressScenario]
) -> list[ScenarioResult]:
    """Mesure chaque crise sur les rendements fournis. Une crise hors période reste vide."""
    results = []
    for scenario in scenarios:
        window = returns.loc[pd.Timestamp(scenario.start) : pd.Timestamp(scenario.end)]
        reference = benchmark.loc[window.index]
        filled = not window.empty
        results.append(
            ScenarioResult(
                name=scenario.name,
                start=scenario.start.isoformat(),
                end=scenario.end.isoformat(),
                sessions=len(window),
                strategy_return=_total(window) if filled else None,
                benchmark_return=_total(reference) if filled else None,
                strategy_drawdown=_drawdown(window) if filled else None,
                benchmark_drawdown=_drawdown(reference) if filled else None,
                strategy_worst_day=float(window.min()) if filled else None,
            )
        )
    return results


def gap_stress(
    max_drawdown: float, max_exposure: float, gap_pct: float, limits: RiskLimits
) -> GapStress:
    """Ajoute un gap sur tout l'univers au pire drawdown, avec l'exposition maximale.

    Les deux pires moments ne coïncident pas forcément : le résultat est un majorant. Un gap à
    l'ouverture survient avant toute réaction possible, le kill switch ne peut que le constater.
    """
    loss = gap_pct * max_exposure
    return GapStress(
        gap_pct=gap_pct,
        max_exposure=max_exposure,
        immediate_loss=loss,
        triggers_kill_switch=loss >= limits.max_daily_loss_pct,
        drawdown_with_gap=1 - (1 - abs(max_drawdown)) * (1 - loss),
    )


def required_risk_limits() -> list[str]:
    """Retourne chaque motif de refus et d'arrêt que les tests doivent déclencher."""
    return [reason.value for reason in RejectReason] + [reason.value for reason in HaltReason]


def _coverage(name: str, outcomes: list[MarkedTestResult], key: str) -> Coverage:
    tests = [item for item in outcomes if name in getattr(item, key)]
    return Coverage(
        name=name,
        tests=[item.nodeid for item in tests],
        failed=[item.nodeid for item in tests if item.outcome != PASSED],
    )


def operational_coverage(
    outcomes: list[MarkedTestResult], scenarios: list[str], limits: list[str]
) -> OperationalReport:
    """Associe chaque scénario et chaque motif exigé aux tests qui le couvrent."""
    known = set(scenarios) | set(limits)
    used = {name for item in outcomes for name in item.scenarios + item.risk_limits}
    return OperationalReport(
        scenarios=[_coverage(name, outcomes, "scenarios") for name in scenarios],
        risk_limits=[_coverage(name, outcomes, "risk_limits") for name in limits],
        unknown_markers=sorted(used - known),
    )


def run_marked_tests(root: Path) -> list[MarkedTestResult]:
    """Lance les tests marqués dans un processus séparé et retourne leurs résultats."""
    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory) / "outcomes.json"
        command = [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-p",
            "no:cacheprovider",
            "-o",
            "addopts=",
            "-p",
            "audit.pytest_plugin",
            "-m",
            f"{SCENARIO_MARKER} or {RISK_LIMIT_MARKER}",
            OUTPUT_OPTION,
            str(output),
            "tests",
        ]
        # Le code de retour n'est pas lu : un test en échec apparaît dans les résultats.
        subprocess.run(command, cwd=root, capture_output=True, check=False)
        if not output.exists():
            raise RuntimeError("pytest n'a produit aucun résultat pour les tests marqués")
        raw = json.loads(output.read_text(encoding="utf-8"))
    return [MarkedTestResult.model_validate(item) for item in raw]
