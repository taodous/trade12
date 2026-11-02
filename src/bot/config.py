"""Chargement et validation de la configuration du bot."""

from datetime import date
from pathlib import Path
from typing import Annotated, Literal, Self

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


class MovingAverageParams(BaseModel):
    """Fenêtres de la stratégie de croisement de moyennes mobiles, en séances."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    fast_window: int = Field(gt=0)
    slow_window: int = Field(gt=0)

    @model_validator(mode="after")
    def _check_windows(self) -> Self:
        if self.fast_window >= self.slow_window:
            raise ValueError("fast_window doit être inférieur à slow_window")
        return self


class DataConfig(BaseModel):
    """Paramètres des données historiques."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    symbols: list[Annotated[str, Field(pattern=r"^[A-Z][A-Z.]*$")]] = Field(min_length=1)
    start_date: date
    end_date: date | None
    feed: Literal["sip", "iex"]
    adjustment: Literal["raw", "split", "dividend", "all"]
    min_bar_age_minutes: int = Field(ge=0)
    directory: str = Field(min_length=1)

    @model_validator(mode="after")
    def _check_values(self) -> Self:
        if len(set(self.symbols)) != len(self.symbols):
            raise ValueError("symbols contient des doublons")
        if self.end_date is not None and self.end_date < self.start_date:
            raise ValueError("end_date ne peut pas précéder start_date")
        return self


class RetryConfig(BaseModel):
    """Backoff exponentiel appliqué aux appels API limités en débit."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    max_attempts: int = Field(gt=0)
    base_delay_seconds: float = Field(gt=0)
    max_delay_seconds: float = Field(gt=0)


class BacktestConfig(BaseModel):
    """Paramètres du backtest. Les séances après `in_sample_end` sont l'out-of-sample."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    in_sample_end: date
    initial_capital: float = Field(gt=0)
    fees_pct: float = Field(ge=0, lt=1)
    slippage_pct: float = Field(ge=0, lt=1)
    benchmark: str = Field(min_length=1)
    report_file: str = Field(min_length=1)
    registry_file: str = Field(min_length=1)


class PerformanceTargets(BaseModel):
    """Objectifs de performance et seuils de résultats suspects."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    min_annual_return: float
    max_drawdown_alert: float = Field(gt=0, le=1)
    max_drawdown_reject: float = Field(gt=0, le=1)
    min_sharpe: float
    min_calmar: float
    min_positive_months: float = Field(ge=0, le=1)
    suspicious_annual_return: float
    suspicious_sharpe: float


class StrategyConfig(BaseModel):
    """Contenu validé de `config/strategy.yaml`."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    risk: RiskLimits
    risk_state_file: str = Field(min_length=1)
    strategy: MovingAverageParams
    data: DataConfig
    backtest: BacktestConfig
    targets: PerformanceTargets
    api_retry: RetryConfig

    @model_validator(mode="after")
    def _check_backtest(self) -> Self:
        if self.backtest.benchmark not in self.data.symbols:
            raise ValueError("benchmark doit faire partie de data.symbols")
        if self.backtest.in_sample_end <= self.data.start_date:
            raise ValueError("in_sample_end doit suivre data.start_date")
        return self


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
