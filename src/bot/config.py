"""Chargement et validation de la configuration du bot."""

from pathlib import Path
from typing import Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, SecretStr, ValidationError, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ENV_PATH = PROJECT_ROOT / ".env"
DEFAULT_STRATEGY_CONFIG_PATH = PROJECT_ROOT / "config" / "strategy.yaml"


class ConfigError(Exception):
    """Configuration absente ou invalide. Le message ne contient jamais de secret."""


class Settings(BaseSettings):
    """Clés API du compte paper Alpaca, lues depuis l'environnement ou le fichier `.env`."""

    model_config = SettingsConfigDict(frozen=True, extra="ignore", hide_input_in_errors=True)

    alpaca_api_key: SecretStr = Field(min_length=1)
    alpaca_secret_key: SecretStr = Field(min_length=1)


class RiskLimits(BaseModel):
    """Limites de risque appliquées par le module `risk`."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    max_position_pct: float = Field(gt=0, le=1)
    max_total_exposure_pct: float = Field(gt=0)
    max_daily_loss_pct: float = Field(gt=0, le=1)
    max_drawdown_pct: float = Field(gt=0, le=1)
    max_orders_per_day: int = Field(gt=0)
    max_data_age_seconds: int = Field(gt=0)
    max_price_change_pct: float = Field(gt=0)
    allow_short: bool
    allow_leverage: bool

    @model_validator(mode="after")
    def _check_exposure(self) -> Self:
        if self.max_total_exposure_pct > 1 and not self.allow_leverage:
            raise ValueError("max_total_exposure_pct supérieur à 1 exige allow_leverage")
        if self.max_position_pct > self.max_total_exposure_pct:
            raise ValueError("max_position_pct ne peut pas dépasser max_total_exposure_pct")
        return self


class StrategyConfig(BaseModel):
    """Contenu validé de `config/strategy.yaml`."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    risk: RiskLimits


def load_settings(env_path: Path = DEFAULT_ENV_PATH) -> Settings:
    """Charge les clés API.

    Lève `ConfigError` en citant uniquement les champs en cause, jamais leur valeur.
    """
    try:
        return Settings(_env_file=env_path)
    except ValidationError as exc:
        fields = ", ".join(str(error["loc"][0]) for error in exc.errors())
        raise ConfigError(f"Clés API absentes ou invalides : {fields}") from None


def load_strategy_config(path: Path = DEFAULT_STRATEGY_CONFIG_PATH) -> StrategyConfig:
    """Charge et valide le fichier de stratégie, ou lève `ConfigError`."""
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ConfigError(f"Lecture impossible de {path} : {exc}") from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"YAML invalide dans {path} : {exc}") from exc

    try:
        return StrategyConfig.model_validate(raw)
    except ValidationError as exc:
        raise ConfigError(f"Configuration invalide dans {path} : {exc}") from exc
