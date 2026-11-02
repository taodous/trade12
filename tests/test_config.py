from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from bot.config import (
    ConfigError,
    MovingAverageParams,
    RiskLimits,
    Settings,
    load_settings,
    load_strategy_config,
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


def test_project_strategy_windows_are_valid() -> None:
    params = load_strategy_config().strategy
    assert params.fast_window < params.slow_window


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
