from datetime import date
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from bot.config import (
    ConfigError,
    MovingAverageParams,
    MovingAverageSettings,
    RiskLimits,
    RiskTargetedSettings,
    Settings,
    StrategyConfig,
    load_settings,
    load_strategy_config,
    sensitivity_grid,
)

SECRET = "valeur-tres-secrete"


def valid_limits(**overrides: Any) -> dict[str, Any]:
    return {
        "max_position_pct": 0.2,
        "max_total_exposure_pct": 1.0,
        "max_daily_loss_pct": 0.02,
        "max_drawdown_pct": 0.1,
        "max_orders_per_day": 20,
        "max_data_age_seconds": 900,
        "max_clock_skew_seconds": 5,
        "max_price_change_pct": 0.2,
        "allow_short": False,
        "allow_leverage": False,
        **overrides,
    }


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ALPACA_API_KEY", raising=False)
    monkeypatch.delenv("ALPACA_SECRET_KEY", raising=False)


def write_env(path: Path, api_key: str, secret_key: str) -> Path:
    env_path = path / ".env"
    env_path.write_text(f"ALPACA_API_KEY={api_key}\nALPACA_SECRET_KEY={secret_key}\n")
    return env_path


def test_load_settings_reads_env_file(tmp_path: Path) -> None:
    settings = load_settings(write_env(tmp_path, "key", SECRET))
    assert settings.alpaca_api_key.get_secret_value() == "key"
    assert settings.alpaca_secret_key.get_secret_value() == SECRET


def test_settings_repr_hides_secrets(tmp_path: Path) -> None:
    settings = load_settings(write_env(tmp_path, SECRET, SECRET))
    assert SECRET not in repr(settings)
    assert SECRET not in str(settings)


def test_load_settings_missing_keys_names_fields(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="alpaca_api_key, alpaca_secret_key"):
        load_settings(tmp_path / "absent.env")


def test_load_settings_rejects_empty_keys(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="alpaca_api_key"):
        load_settings(write_env(tmp_path, "", SECRET))


def test_load_settings_error_does_not_leak_secret(tmp_path: Path) -> None:
    with pytest.raises(ConfigError) as excinfo:
        load_settings(write_env(tmp_path, "", SECRET))
    assert SECRET not in str(excinfo.value)


def test_settings_is_frozen(tmp_path: Path) -> None:
    settings: Settings = load_settings(write_env(tmp_path, "key", SECRET))
    with pytest.raises(ValidationError):
        settings.alpaca_api_key = "autre"  # type: ignore[assignment,misc]


def test_project_strategy_file_is_valid() -> None:
    config = load_strategy_config()
    assert config.risk.allow_short is False
    assert config.risk.allow_leverage is False
    assert config.risk.max_total_exposure_pct <= 1


def test_risk_limits_accepts_valid_values() -> None:
    assert RiskLimits.model_validate(valid_limits()).max_orders_per_day == 20


@pytest.mark.parametrize(
    "overrides",
    [
        {"max_position_pct": 0.0},
        {"max_position_pct": 1.5},
        {"max_daily_loss_pct": -0.01},
        {"max_drawdown_pct": 0.0},
        {"max_orders_per_day": 0},
        {"max_data_age_seconds": 0},
        {"max_price_change_pct": 0.0},
        {"max_position_pct": "0.2"},
        {"allow_short": "false"},
        {"champ_inconnu": 1},
        {"max_total_exposure_pct": 1.5},
        {"max_position_pct": 0.5, "max_total_exposure_pct": 0.3},
    ],
)
def test_risk_limits_rejects_invalid_values(overrides: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        RiskLimits.model_validate(valid_limits(**overrides))


def test_risk_limits_exposure_above_one_requires_leverage() -> None:
    limits = valid_limits(max_total_exposure_pct=1.5, allow_leverage=True)
    assert RiskLimits.model_validate(limits).max_total_exposure_pct == 1.5


def test_risk_limits_requires_every_field() -> None:
    limits = valid_limits()
    del limits["max_drawdown_pct"]
    with pytest.raises(ValidationError):
        RiskLimits.model_validate(limits)


@pytest.mark.parametrize(
    ("fast_window", "slow_window"),
    [(0, 200), (50, 0), (-1, 200), (200, 200), (200, 50)],
)
def test_moving_average_params_rejects_invalid_windows(fast_window: int, slow_window: int) -> None:
    with pytest.raises(ValidationError):
        MovingAverageParams(fast_window=fast_window, slow_window=slow_window)


def test_project_strategy_respects_the_risk_limits() -> None:
    config = load_strategy_config()
    assert isinstance(config.strategy, RiskTargetedSettings)
    assert config.strategy.params.max_weight <= config.risk.max_position_pct
    assert config.strategy.params.max_exposure <= config.risk.max_total_exposure_pct


def project_config_dict(**overrides: Any) -> dict[str, Any]:
    return {**load_strategy_config().model_dump(), **overrides}


def test_benchmark_must_be_in_universe() -> None:
    raw = project_config_dict()
    raw["backtest"]["benchmark"] = "ABC"
    with pytest.raises(ValidationError, match="benchmark"):
        StrategyConfig.model_validate(raw)


def test_in_sample_end_must_follow_data_start() -> None:
    raw = project_config_dict()
    raw["backtest"]["in_sample_end"] = raw["data"]["start_date"]
    with pytest.raises(ValidationError, match="in_sample_end"):
        StrategyConfig.model_validate(raw)


def audit_dict(raw: dict[str, Any]) -> dict[str, Any]:
    audit: dict[str, Any] = raw["audit"]
    return audit


@pytest.mark.parametrize(
    ("section", "key", "value", "message"),
    [
        ("sensitivity", "relative_steps", [1.0, 0.8], "croissantes"),
        ("sensitivity", "relative_steps", [0.8, 1.2], "1.0"),
        # Un pas minuscule ramène les fenêtres à 0 séance : combinaison invalide.
        ("sensitivity", "relative_steps", [0.001, 1.0], "invalide"),
        ("sensitivity", "relative_steps", [1.0, 1.001], "même valeur"),
        (
            "sensitivity",
            "parameters",
            {"moving_average": ["fast_window", "fast_window"]},
            "doublon",
        ),
        ("sensitivity", "parameters", {"risk_targeted": ["absent"]}, "numérique"),
        ("sensitivity", "parameters", {"risk_targeted": ["regime_filter"]}, "numérique"),
        ("sensitivity", "parameters", {"moving_average": ["fast_window"]}, "Aucune grille"),
        ("sensitivity", "slippage_multipliers", [0.5, 2.0], "greater than 1"),
        ("thresholds", "stressed_slippage_multiplier", 4.0, "grille"),
    ],
)
def test_audit_grid_is_validated(section: str, key: str, value: object, message: str) -> None:
    raw = project_config_dict()
    audit_dict(raw)[section][key] = value
    with pytest.raises(ValidationError, match=message):
        StrategyConfig.model_validate(raw)


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("max_weight", 0.5, "limite par position"),
        ("regime_symbol", "XYZ", "hors de l'univers"),
        ("risky_symbols", ["SPY", "XYZ"], "hors de l'univers"),
    ],
)
def test_candidates_must_respect_risk_limits_and_universe(
    key: str, value: object, message: str
) -> None:
    raw = project_config_dict()
    raw["candidates"][0]["params"][key] = value
    with pytest.raises(ValidationError, match=message):
        StrategyConfig.model_validate(raw)


def test_candidate_exposure_cannot_exceed_risk_limit() -> None:
    raw = project_config_dict()
    raw["risk"]["max_total_exposure_pct"] = 0.9
    raw["risk"]["max_position_pct"] = 0.2
    with pytest.raises(ValidationError, match="exposition maximale"):
        StrategyConfig.model_validate(raw)


def test_pre_registered_candidates_are_the_three_planned_variants() -> None:
    """Variantes fixées le 2026-11-05 avant tout résultat : les changer exige une décision."""
    candidates = [
        candidate
        for candidate in load_strategy_config().candidates
        if isinstance(candidate, RiskTargetedSettings)
    ]
    assert [
        (c.params.trend, c.params.trend_window, c.params.regime_filter) for c in candidates
    ] == [
        ("sma", 200, False),
        ("sma", 200, True),
        ("momentum", 252, False),
    ]
    for candidate in candidates:
        assert candidate.params.volatility_window == 63
        assert candidate.params.target_volatility == 0.08
        assert candidate.rebalance.band_pct == 0.02


def test_grid_of_the_reference_strategy_is_the_stage_7_grid() -> None:
    config = load_strategy_config()
    reference = MovingAverageSettings(
        name="moving_average", params=MovingAverageParams(fast_window=50, slow_window=200)
    )
    grid = sensitivity_grid(reference, config.audit.sensitivity)
    windows = [
        (item.params.fast_window, item.params.slow_window)
        for item in grid
        if isinstance(item, MovingAverageSettings)
    ]
    fasts = [30, 40, 50, 60, 70]
    slows = [120, 160, 200, 240, 280]
    assert windows == [(fast, slow) for fast in fasts for slow in slows]


def test_stress_scenario_dates_are_ordered() -> None:
    raw = project_config_dict()
    scenario = audit_dict(raw)["stress"]["scenarios"][0]
    scenario["end"] = date(2000, 1, 1)
    with pytest.raises(ValidationError, match="end"):
        StrategyConfig.model_validate(raw)


def test_project_audit_thresholds_match_the_validated_plan() -> None:
    """Seuils validés le 2026-11-04, avant tout résultat : les changer exige une décision."""
    thresholds = load_strategy_config().audit.thresholds
    assert thresholds.max_monte_carlo_drawdown_p95 == 0.15
    assert thresholds.min_walk_forward_positive_windows == 0.60
    assert thresholds.min_walk_forward_efficiency == 0.5
    assert thresholds.neighbor_sharpe_ratio == 0.80
    assert thresholds.min_neighbor_share == 0.70
    assert thresholds.stressed_slippage_multiplier == 2.0
    assert thresholds.min_stressed_sharpe_ratio == 0.80
    assert thresholds.min_deflated_sharpe_probability == 0.95


def test_project_targets_match_the_documented_revision() -> None:
    """Cibles révisées le 2026-11-05 (CLAUDE.md) : les changer exige une nouvelle décision."""
    targets = load_strategy_config().targets
    assert targets.min_sharpe == 0.6
    assert targets.min_annual_return == 0.05
    assert targets.min_calmar == 0.5
    # Seuils de risque et de résultats suspects inchangés.
    assert (targets.max_drawdown_alert, targets.max_drawdown_reject) == (0.10, 0.15)
    assert targets.min_positive_months == 0.60
    assert (targets.suspicious_annual_return, targets.suspicious_sharpe) == (0.30, 2.0)


def test_load_strategy_config_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="Lecture impossible"):
        load_strategy_config(tmp_path / "absent.yaml")


def test_load_strategy_config_malformed_yaml(tmp_path: Path) -> None:
    path = tmp_path / "strategy.yaml"
    path.write_text("risk: [non ferme")
    with pytest.raises(ConfigError, match="YAML invalide"):
        load_strategy_config(path)


@pytest.mark.parametrize("content", ["", "risk: {}", "autre: 1"])
def test_load_strategy_config_invalid_content(tmp_path: Path, content: str) -> None:
    path = tmp_path / "strategy.yaml"
    path.write_text(content)
    with pytest.raises(ConfigError, match="Configuration invalide"):
        load_strategy_config(path)
