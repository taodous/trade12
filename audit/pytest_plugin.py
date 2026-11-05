"""Plugin pytest qui exporte le résultat des tests marqués `scenario` ou `risk_limit`.

Activé par `-p audit.pytest_plugin --audit-output <fichier>`, il écrit pour chaque test marqué
ses scénarios, les motifs de risque qu'il déclenche et son résultat. Sans l'option, il ne fait
rien.
"""

import json
from pathlib import Path

import pytest

OUTPUT_OPTION = "--audit-output"
SCENARIO_MARKER = "scenario"
RISK_LIMIT_MARKER = "risk_limit"
PASSED = "passed"
FAILED = "failed"
SKIPPED = "skipped"


def pytest_addoption(parser: pytest.Parser) -> None:
    """Déclare l'option qui donne le fichier de sortie."""
    parser.addoption(OUTPUT_OPTION, action="store", default=None, help="Sortie JSON de l'audit")


class _Collector:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._markers: dict[str, tuple[list[str], list[str]]] = {}
        self._outcomes: dict[str, str] = {}

    def pytest_collection_modifyitems(self, items: list[pytest.Item]) -> None:
        for item in items:
            scenarios = [str(mark.args[0]) for mark in item.iter_markers(SCENARIO_MARKER)]
            limits = [str(mark.args[0]) for mark in item.iter_markers(RISK_LIMIT_MARKER)]
            if scenarios or limits:
                self._markers[item.nodeid] = (scenarios, limits)

    def pytest_runtest_logreport(self, report: pytest.TestReport) -> None:
        if report.nodeid not in self._markers:
            return
        if report.failed:
            self._outcomes[report.nodeid] = FAILED
        elif report.skipped and self._outcomes.get(report.nodeid) != FAILED:
            self._outcomes[report.nodeid] = SKIPPED
        elif report.when == "call" and self._outcomes.get(report.nodeid) is None:
            self._outcomes[report.nodeid] = PASSED

    def pytest_sessionfinish(self) -> None:
        records = [
            {
                "nodeid": nodeid,
                "scenarios": scenarios,
                "risk_limits": limits,
                # Un test collecté mais jamais exécuté jusqu'au bout n'a rien prouvé.
                "outcome": self._outcomes.get(nodeid, FAILED),
            }
            for nodeid, (scenarios, limits) in sorted(self._markers.items())
        ]
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(records, indent=2), encoding="utf-8")


def pytest_configure(config: pytest.Config) -> None:
    """Enregistre le collecteur si un fichier de sortie est demandé."""
    output = config.getoption(OUTPUT_OPTION)
    if output:
        config.pluginmanager.register(_Collector(Path(str(output))), "audit-collector")
