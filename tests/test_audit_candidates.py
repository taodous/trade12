import json
from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from audit import candidates as candidates_module
from audit.candidates import CandidateResult, compare_candidates, select_candidate
from audit.registry import load_registry
from bot.config import (
    ConfigError,
    RebalanceConfig,
    RiskTargetedParams,
    RiskTargetedSettings,
    StrategyConfig,
)
from tests.test_audit_backtest import fill_store, make_config

NOW = datetime(2026, 11, 5, tzinfo=UTC)


def result(index: int, sharpe: float, drawdown: float, eligible: bool) -> CandidateResult:
    return CandidateResult(
        index=index,
        variant=f"v{index}",
        settings={},
        annual_return=0.05,
        annual_volatility=0.08,
        sharpe=sharpe,
        max_drawdown=drawdown,
        calmar=0.5,
        positive_months=0.6,
        eligible=eligible,
    )


def test_best_sharpe_among_eligible_variants_wins() -> None:
    results = [
        result(0, 1.4, -0.20, eligible=False),
        result(1, 0.9, -0.12, eligible=True),
        result(2, 0.7, -0.08, eligible=True),
    ]
    assert select_candidate(results) == (1, False)


def test_without_eligible_variant_best_sharpe_is_kept_and_flagged() -> None:
    results = [result(0, 0.4, -0.2, eligible=False), result(1, 0.6, -0.3, eligible=False)]
    assert select_candidate(results) == (1, True)


def test_undefined_sharpe_ranks_last_and_ties_keep_the_first() -> None:
    results = [
        result(0, float("nan"), -0.1, eligible=True),
        result(1, 0.5, -0.1, eligible=True),
        result(2, 0.5, -0.1, eligible=True),
    ]
    assert select_candidate(results) == (1, False)


def test_selection_needs_at_least_one_variant() -> None:
    with pytest.raises(ValueError, match="Aucune variante"):
        select_candidate([])


def candidates_config(tmp_path: Path) -> StrategyConfig:
    fill_store(tmp_path)
    base = make_config(["A", "B"], date(2022, 12, 31))
    params = RiskTargetedParams(
        trend="sma",
        trend_window=20,
        volatility_window=10,
        target_volatility=0.08,
        max_weight=0.5,
        max_exposure=1.0,
        regime_filter=False,
        regime_symbol="A",
        regime_window=20,
        regime_scale=0.5,
        risky_symbols=["A"],
    )
    rebalance = RebalanceConfig(frequency="monthly", band_pct=0.02)
    variants = [
        RiskTargetedSettings(name="risk_targeted", params=params, rebalance=rebalance),
        RiskTargetedSettings(
            name="risk_targeted",
            params=params.model_copy(update={"regime_filter": True}),
            rebalance=rebalance,
        ),
    ]
    return base.model_copy(update={"candidates": variants})


def test_compare_candidates_registers_each_variant_and_writes_reports(tmp_path: Path) -> None:
    config = candidates_config(tmp_path)

    report = compare_candidates(config, tmp_path, NOW)

    assert [item.index for item in report.results] == [0, 1]
    assert report.results[0].variant != report.results[1].variant
    assert report.selected in {0, 1}
    registry = load_registry(tmp_path / config.backtest.registry_file)
    assert [entry["id"] for entry in registry] == [item.variant for item in report.results]
    for item in report.results:
        assert (tmp_path / "reports" / "candidates" / f"{item.variant}.json").exists()
        assert item.eligible == (abs(item.max_drawdown) <= config.targets.max_drawdown_reject)
    saved = json.loads((tmp_path / "reports" / "candidates.json").read_text(encoding="utf-8"))
    assert saved["selected"] == report.selected
    # Le backtest de la stratégie configurée n'est pas écrasé par les variantes.
    assert not (tmp_path / config.backtest.report_file).exists()


def test_main_reports_invalid_config(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def fail() -> StrategyConfig:
        raise ConfigError("invalide")

    monkeypatch.setattr(candidates_module, "load_strategy_config", fail)
    assert candidates_module.main() == 1
    assert "invalide" in capsys.readouterr().out


def test_main_compares_the_configured_candidates(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = candidates_config(tmp_path)
    monkeypatch.setattr(candidates_module, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(candidates_module, "load_strategy_config", lambda: config)
    assert candidates_module.main() == 0
    assert "candidates_done" in capsys.readouterr().out
