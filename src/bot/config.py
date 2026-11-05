"""Chargement et validation de la configuration du bot."""

from datetime import date
from itertools import product
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
    max_clock_skew_seconds: int = Field(ge=0)
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


class RiskTargetedParams(BaseModel):
    """Stratégie à risque ciblé : filtre de tendance, poids en inverse de la volatilité, cible.

    `trend` vaut `sma` (clôture au-dessus de sa moyenne sur `trend_window` séances) ou `momentum`
    (rendement positif sur `trend_window` séances). Le filtre de régime réduit l'exposition des
    `risky_symbols` quand `regime_symbol` clôture sous sa moyenne sur `regime_window` séances.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    trend: Literal["sma", "momentum"]
    trend_window: int = Field(gt=1)
    volatility_window: int = Field(gt=1)
    target_volatility: float = Field(gt=0, le=1)
    max_weight: float = Field(gt=0, le=1)
    max_exposure: float = Field(gt=0, le=1)
    regime_filter: bool
    regime_symbol: str = Field(pattern=r"^[A-Z][A-Z.]*$")
    regime_window: int = Field(gt=1)
    regime_scale: float = Field(ge=0, le=1)
    risky_symbols: list[Annotated[str, Field(pattern=r"^[A-Z][A-Z.]*$")]]

    @model_validator(mode="after")
    def _check_weights(self) -> Self:
        if self.max_weight > self.max_exposure:
            raise ValueError("max_weight ne peut pas dépasser max_exposure")
        return self


class RebalanceConfig(BaseModel):
    """Rééquilibrage périodique. En paper, un écart sous `band_pct` du capital reste tel quel."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    frequency: Literal["monthly"]
    band_pct: float = Field(ge=0, lt=1)


class MovingAverageSettings(BaseModel):
    """Stratégie de référence : positions égales, sans rééquilibrage."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    name: Literal["moving_average"]
    params: MovingAverageParams


class RiskTargetedSettings(BaseModel):
    """Stratégie à risque ciblé, rééquilibrée périodiquement."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    name: Literal["risk_targeted"]
    params: RiskTargetedParams
    rebalance: RebalanceConfig


StrategySettings = Annotated[
    MovingAverageSettings | RiskTargetedSettings, Field(discriminator="name")
]


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


class LiveConfig(BaseModel):
    """Paramètres de la boucle de paper trading."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    journal_file: str = Field(min_length=1)
    log_dir: str = Field(min_length=1)
    poll_interval_seconds: int = Field(gt=0)
    max_consecutive_errors: int = Field(gt=0)
    signal_lookback_days: int = Field(gt=0)
    price_feed: Literal["sip", "iex"]
    cancel_wait_seconds: int = Field(ge=0)


class WalkForwardConfig(BaseModel):
    """Fenêtres glissantes du walk-forward, en séances."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    train_sessions: int = Field(gt=0)
    test_sessions: int = Field(gt=0)


class MonteCarloConfig(BaseModel):
    """Tirages Monte Carlo, reproductibles grâce à la graine."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    seed: int = Field(ge=0)
    simulations: int = Field(gt=0)
    block_sessions: int = Field(gt=0)


def _strictly_increasing(values: list[int] | list[float]) -> bool:
    return all(left < right for left, right in zip(values, values[1:], strict=False))


class SensitivityConfig(BaseModel):
    """Grille de réglages autour des valeurs choisies, et multiplicateurs de slippage.

    `parameters` donne, pour chaque stratégie, les paramètres qui varient. Chacun prend sa valeur
    choisie multipliée par chaque pas de `relative_steps`, arrondie à l'entier pour un paramètre
    entier. La grille est le produit de ces valeurs. Définie en relatif, elle ne dépend pas de la
    variante retenue.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    parameters: dict[str, list[str]] = Field(min_length=1)
    relative_steps: list[Annotated[float, Field(gt=0)]] = Field(min_length=1)
    slippage_multipliers: list[Annotated[float, Field(gt=1)]] = Field(min_length=1)

    @model_validator(mode="after")
    def _check_grid(self) -> Self:
        for names in self.parameters.values():
            if not names or len(set(names)) != len(names):
                raise ValueError("Chaque liste de parameters doit être non vide et sans doublon")
        for values in (self.relative_steps, self.slippage_multipliers):
            if not _strictly_increasing(values):
                raise ValueError("Les valeurs de la grille doivent être uniques et croissantes")
        if 1.0 not in self.relative_steps:
            raise ValueError("relative_steps doit contenir 1.0, le réglage choisi")
        return self


def sensitivity_grid(
    settings: MovingAverageSettings | RiskTargetedSettings, sensitivity: SensitivityConfig
) -> list[MovingAverageSettings | RiskTargetedSettings]:
    """Retourne chaque réglage de la grille, dans l'ordre du produit des paramètres.

    Lève `ValueError` si la stratégie n'a pas de grille, si un paramètre n'est pas numérique, si
    deux pas donnent la même valeur ou si une combinaison est invalide.
    """
    names = sensitivity.parameters.get(settings.name)
    if names is None:
        raise ValueError(f"Aucune grille de sensibilité pour la stratégie {settings.name}")
    params = settings.params
    values: list[list[int | float]] = []
    for name in names:
        chosen = getattr(params, name, None)
        if isinstance(chosen, bool) or not isinstance(chosen, int | float):
            raise ValueError(f"{name} n'est pas un paramètre numérique de la stratégie")
        steps = [chosen * step for step in sensitivity.relative_steps]
        column: list[int | float] = (
            [round(value) for value in steps] if isinstance(chosen, int) else list(steps)
        )
        if not _strictly_increasing(column):
            raise ValueError(f"La grille de {name} donne deux fois la même valeur")
        values.append(column)

    grid = []
    for combination in product(*values):
        update = dict(zip(names, combination, strict=True))
        try:
            variant = type(params).model_validate({**params.model_dump(), **update})
        except ValidationError as exc:
            raise ValueError(f"Réglage invalide dans la grille : {update}") from exc
        grid.append(settings.model_copy(update={"params": variant}))
    return grid


class StressScenario(BaseModel):
    """Période de crise historique, bornes incluses."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    name: str = Field(min_length=1)
    start: date
    end: date

    @model_validator(mode="after")
    def _check_dates(self) -> Self:
        if self.end < self.start:
            raise ValueError("end ne peut pas précéder start")
        return self


class StressConfig(BaseModel):
    """Scénarios de marché et scénarios opérationnels exigés."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    synthetic_gap_pct: float = Field(gt=0, lt=1)
    scenarios: list[StressScenario]
    required_scenarios: list[str] = Field(min_length=1)


class SabotageConfig(BaseModel):
    """Test de sabotage : chaque contrôle cassé doit faire échouer au moins un test."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    report_file: str = Field(min_length=1)
    test_paths: list[str] = Field(min_length=1)
    min_timeout_seconds: int = Field(gt=0)
    baseline_timeout_seconds: int = Field(gt=0)
    timeout_multiplier: float = Field(ge=1)


class RobustnessThresholds(BaseModel):
    """Seuils de robustesse, fixés avant de voir les résultats."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    max_monte_carlo_drawdown_p95: float = Field(gt=0, le=1)
    min_walk_forward_positive_windows: float = Field(ge=0, le=1)
    min_walk_forward_efficiency: float
    neighbor_sharpe_ratio: float = Field(gt=0, le=1)
    min_neighbor_share: float = Field(ge=0, le=1)
    stressed_slippage_multiplier: float = Field(gt=1)
    min_stressed_sharpe_ratio: float = Field(gt=0)
    min_deflated_sharpe_probability: float = Field(gt=0, lt=1)
    min_paper_fills: int = Field(gt=0)


class AuditConfig(BaseModel):
    """Paramètres du protocole d'audit."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    report_dir: str = Field(min_length=1)
    out_of_sample_log: str = Field(min_length=1)
    walk_forward: WalkForwardConfig
    monte_carlo: MonteCarloConfig
    sensitivity: SensitivityConfig
    stress: StressConfig
    sabotage: SabotageConfig
    thresholds: RobustnessThresholds

    @model_validator(mode="after")
    def _check_multiplier(self) -> Self:
        multiplier = self.thresholds.stressed_slippage_multiplier
        if multiplier not in self.sensitivity.slippage_multipliers:
            raise ValueError("stressed_slippage_multiplier doit faire partie de la grille")
        return self


class StrategyConfig(BaseModel):
    """Contenu validé de `config/strategy.yaml`."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    risk: RiskLimits
    risk_state_file: str = Field(min_length=1)
    strategy: StrategySettings
    # Variantes pré-enregistrées avant tout résultat, comparées par `audit.candidates`.
    candidates: list[StrategySettings]
    data: DataConfig
    backtest: BacktestConfig
    targets: PerformanceTargets
    live: LiveConfig
    api_retry: RetryConfig
    audit: AuditConfig

    @model_validator(mode="after")
    def _check_backtest(self) -> Self:
        if self.backtest.benchmark not in self.data.symbols:
            raise ValueError("benchmark doit faire partie de data.symbols")
        if self.backtest.in_sample_end <= self.data.start_date:
            raise ValueError("in_sample_end doit suivre data.start_date")
        for settings in [self.strategy, *self.candidates]:
            self._check_strategy(settings)
            sensitivity_grid(settings, self.audit.sensitivity)
        return self

    def _check_strategy(self, settings: MovingAverageSettings | RiskTargetedSettings) -> None:
        if not isinstance(settings, RiskTargetedSettings):
            return
        params = settings.params
        if params.max_weight > self.risk.max_position_pct:
            raise ValueError("max_weight dépasse la limite par position du module risque")
        if params.max_exposure > self.risk.max_total_exposure_pct:
            raise ValueError("max_exposure dépasse l'exposition maximale du module risque")
        unknown = {params.regime_symbol, *params.risky_symbols} - set(self.data.symbols)
        if unknown:
            raise ValueError(f"Symboles hors de l'univers : {sorted(unknown)}")


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
