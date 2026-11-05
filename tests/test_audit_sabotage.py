from datetime import UTC, datetime
from pathlib import Path

import pytest

from audit import sabotage as sabotage_module
from audit.sabotage import (
    SABOTAGES,
    Outcome,
    Sabotage,
    SabotageError,
    SabotageReport,
    SabotageResult,
    apply_sabotage,
    load_sabotage_report,
    run_sabotage,
    source_hash,
)
from bot.config import PROJECT_ROOT, SabotageConfig

NOW = datetime(2026, 11, 4, tzinfo=UTC)
SETTINGS = SabotageConfig(
    report_file="reports/sabotage.json",
    test_paths=["tests"],
    min_timeout_seconds=5,
    baseline_timeout_seconds=120,
    timeout_multiplier=1.0,
)


def test_apply_sabotage_replaces_a_unique_fragment() -> None:
    item = Sabotage("x", "f.py", "a > b", "a >= b", "")
    assert apply_sabotage("if a > b:\n", item) == "if a >= b:\n"


@pytest.mark.parametrize("text", ["rien\n", "a > b\na > b\n"])
def test_apply_sabotage_refuses_missing_or_ambiguous_fragment(text: str) -> None:
    with pytest.raises(SabotageError, match="trouvé"):
        apply_sabotage(text, Sabotage("x", "f.py", "a > b", "a >= b", ""))


def test_every_sabotage_still_applies_to_the_current_code() -> None:
    """Un contrôle renommé ou déplacé doit faire échouer ce test, pas disparaître de l'audit."""
    assert len({item.name for item in SABOTAGES}) == len(SABOTAGES)
    for item in SABOTAGES:
        text = (PROJECT_ROOT / item.path).read_text(encoding="utf-8")
        assert apply_sabotage(text, item) != text


def test_source_hash_follows_content_but_not_line_endings(tmp_path: Path) -> None:
    for directory in ("src", "tests"):
        (tmp_path / directory).mkdir()
    module = tmp_path / "src" / "m.py"
    module.write_bytes(b"x = 1\n")
    first = source_hash(tmp_path)
    module.write_bytes(b"x = 1\r\n")
    assert source_hash(tmp_path) == first
    module.write_bytes(b"x = 2\n")
    assert source_hash(tmp_path) != first


def make_project(root: Path) -> None:
    """Projet minimal avec ses paquets `bot` et `audit`, testé par un seul fichier."""
    (root / "src" / "bot").mkdir(parents=True)
    (root / "src" / "bot" / "__init__.py").write_text("", encoding="utf-8")
    (root / "src" / "bot" / "rules.py").write_text(
        "def limit(x):\n"
        "    return min(x, 10)\n\n"
        "def untested(x):\n"
        "    return x + 1\n\n"
        "def wait():\n"
        "    return True\n",
        encoding="utf-8",
    )
    (root / "audit").mkdir()
    (root / "audit" / "__init__.py").write_text("", encoding="utf-8")
    (root / "config").mkdir()
    (root / "tests").mkdir()
    (root / "tests" / "test_rules.py").write_text(
        "from bot.rules import limit, wait\n\n"
        "def test_limit():\n    assert limit(50) == 10\n\n"
        "def test_wait():\n"
        "    while not wait():\n        pass\n",
        encoding="utf-8",
    )
    (root / "pyproject.toml").write_text("[tool.pytest.ini_options]\n", encoding="utf-8")


PLANTED = (
    Sabotage("caught", "src/bot/rules.py", "min(x, 10)", "x", "limite retirée"),
    Sabotage("missed", "src/bot/rules.py", "x + 1", "x + 2", "fonction non testée"),
    Sabotage("hangs", "src/bot/rules.py", "return True", "return False", "boucle sans fin"),
)


def test_run_sabotage_classifies_each_outcome_on_a_copy(tmp_path: Path) -> None:
    make_project(tmp_path)
    before = (tmp_path / "src" / "bot" / "rules.py").read_text(encoding="utf-8")

    report = run_sabotage(tmp_path, SETTINGS, NOW, PLANTED)

    outcomes = {item.name: item.outcome for item in report.results}
    assert outcomes == {
        "caught": Outcome.DETECTED,
        "missed": Outcome.SURVIVED,
        "hangs": Outcome.TIMEOUT,
    }
    assert report.survivors == ["missed"]
    assert report.timeout_seconds >= SETTINGS.min_timeout_seconds
    assert report.source_hash == source_hash(tmp_path)
    assert (tmp_path / "src" / "bot" / "rules.py").read_text(encoding="utf-8") == before


def test_report_fingerprints_the_tested_copy_not_later_changes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Un changement du code pendant le test ne doit pas passer pour testé."""
    make_project(tmp_path)
    before = source_hash(tmp_path)
    original_run = sabotage_module._run_tests

    def run_then_edit(
        workdir: Path, settings: SabotageConfig, timeout: float | None
    ) -> tuple[int, float]:
        (tmp_path / "src" / "bot" / "rules.py").write_text("changed = 1\n", encoding="utf-8")
        return original_run(workdir, settings, timeout)

    monkeypatch.setattr(sabotage_module, "_run_tests", run_then_edit)
    report = run_sabotage(tmp_path, SETTINGS, NOW, PLANTED[:1])

    assert report.source_hash == before
    assert report.source_hash != source_hash(tmp_path)


def test_run_sabotage_requires_passing_tests_first(tmp_path: Path) -> None:
    make_project(tmp_path)
    test_file = tmp_path / "tests" / "test_rules.py"
    test_file.write_text("def test_broken():\n    assert False\n", encoding="utf-8")
    with pytest.raises(SabotageError, match="sans sabotage"):
        run_sabotage(tmp_path, SETTINGS, NOW, PLANTED[:1])


def test_run_sabotage_refuses_an_inapplicable_sabotage_before_running(tmp_path: Path) -> None:
    make_project(tmp_path)
    absent = Sabotage("absent", "src/bot/rules.py", "introuvable", "x", "")
    with pytest.raises(SabotageError, match="absent"):
        run_sabotage(tmp_path, SETTINGS, NOW, (absent,))


def sample_report(outcome: Outcome = Outcome.DETECTED) -> SabotageReport:
    return SabotageReport(
        ran_at=NOW,
        source_hash="abc",
        baseline_seconds=1.0,
        timeout_seconds=30.0,
        results=[SabotageResult(name="s", path="p", description="d", outcome=outcome, seconds=1)],
    )


def test_report_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "sabotage.json"
    assert load_sabotage_report(path) is None
    path.write_text(sample_report().model_dump_json(), encoding="utf-8")
    assert load_sabotage_report(path) == sample_report()


@pytest.mark.parametrize(("outcome", "code"), [(Outcome.DETECTED, 0), (Outcome.SURVIVED, 1)])
def test_main_writes_report_and_fails_on_survivors(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, outcome: Outcome, code: int
) -> None:
    def fake_run(root: Path, settings: SabotageConfig, now: datetime) -> SabotageReport:
        return sample_report(outcome)

    monkeypatch.setattr(sabotage_module, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(sabotage_module, "run_sabotage", fake_run)
    assert sabotage_module.main() == code
    assert load_sabotage_report(tmp_path / "reports" / "sabotage.json") == sample_report(outcome)


def test_main_reports_sabotage_error(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def fail(root: Path, settings: SabotageConfig, now: datetime) -> SabotageReport:
        raise SabotageError("copie impossible")

    monkeypatch.setattr(sabotage_module, "run_sabotage", fail)
    assert sabotage_module.main() == 1
    assert "copie impossible" in capsys.readouterr().out
