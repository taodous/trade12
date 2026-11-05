"""Sensibilité de la stratégie à ses réglages et aux coûts d'exécution.

La grille vient de `audit.sensitivity` : chaque paramètre listé pour la stratégie varie autour de
sa valeur choisie. Tous les réglages sont évalués sur la même période, portefeuille vide à la
séance où le plus exigeant d'entre eux devient calculable. Le Sharpe du réglage choisi peut donc
différer légèrement de celui du backtest in-sample, qui commence dès sa propre séance. Chaque
réglage est inscrit au registre des variantes : il compte comme un essai pour le biais de tests
multiples.
"""

import math
from datetime import datetime
from pathlib import Path

import pandas as pd
from pydantic import BaseModel, ConfigDict

from audit.backtest import configured_strategy, run_strategy, variant_definition
from audit.metrics import compute_metrics
from audit.registry import COST_STRESS, SENSITIVITY, record_variant
from bot.config import StrategyConfig, sensitivity_grid
from bot.strategy.factory import rebalance_policy


class GridPoint(BaseModel):
    """Résultat d'un réglage de la grille. `params` ne contient que les paramètres variés."""

    model_config = ConfigDict(frozen=True, ser_json_inf_nan="strings")

    params: dict[str, float]
    annual_return: float
    sharpe: float
    max_drawdown: float


class CostStress(BaseModel):
    """Résultat du réglage choisi avec un slippage majoré."""

    model_config = ConfigDict(frozen=True, ser_json_inf_nan="strings")

    slippage_multiplier: float
    slippage_pct: float
    annual_return: float
    sharpe: float
    max_drawdown: float


class SensitivityReport(BaseModel):
    """Stabilité de la performance autour du réglage choisi."""

    model_config = ConfigDict(frozen=True, ser_json_inf_nan="strings")

    evaluation_start: str
    parameters: list[str]
    grid: list[GridPoint]
    chosen_sharpe: float
    neighbor_share: float
    costs: list[CostStress]


def neighbor_share(chosen: float, neighbors: list[float], ratio: float) -> float:
    """Retourne la part des réglages voisins dont le Sharpe atteint `ratio` fois celui choisi.

    Un Sharpe choisi négatif, nul ou indéfini ne permet pas de parler de stabilité : la part
    est alors NaN, ce qui fait échouer le critère.
    """
    if not neighbors or not math.isfinite(chosen) or chosen <= 0:
        return float("nan")
    stable = [value for value in neighbors if math.isfinite(value) and value >= ratio * chosen]
    return len(stable) / len(neighbors)


def _results(returns: pd.Series) -> dict[str, float]:
    metrics = compute_metrics(returns)
    return {
        "annual_return": metrics.annual_return,
        "sharpe": metrics.sharpe,
        "max_drawdown": metrics.max_drawdown,
    }


def run_sensitivity(
    config: StrategyConfig,
    opens: pd.DataFrame,
    closes: pd.DataFrame,
    registry: Path,
    now: datetime,
) -> SensitivityReport:
    """Évalue la grille de réglages, puis le réglage choisi sous des coûts majorés."""
    settings_grid = sensitivity_grid(config.strategy, config.audit.sensitivity)
    parameters = config.audit.sensitivity.parameters[config.strategy.name]
    strategies = [configured_strategy(config, settings) for settings in settings_grid]
    start = max(strategy.warm_up for strategy in strategies)

    points = []
    for settings, strategy in zip(settings_grid, strategies, strict=True):
        run = run_strategy(
            strategy, opens, closes, start, config.backtest, rebalance_policy(settings)
        )
        results = _results(run.returns)
        points.append(
            GridPoint(
                params={name: float(getattr(settings.params, name)) for name in parameters},
                **results,
            )
        )
        record_variant(registry, variant_definition(config, settings), results, now, SENSITIVITY)

    chosen_index = settings_grid.index(config.strategy)
    chosen_sharpe = points[chosen_index].sharpe
    others = [point.sharpe for index, point in enumerate(points) if index != chosen_index]
    return SensitivityReport(
        evaluation_start=pd.Timestamp(closes.index[start]).date().isoformat(),
        parameters=parameters,
        grid=points,
        chosen_sharpe=chosen_sharpe,
        neighbor_share=neighbor_share(
            chosen_sharpe, others, config.audit.thresholds.neighbor_sharpe_ratio
        ),
        costs=[
            _cost_stress(config, opens, closes, multiplier, registry, now)
            for multiplier in config.audit.sensitivity.slippage_multipliers
        ],
    )


def _cost_stress(
    config: StrategyConfig,
    opens: pd.DataFrame,
    closes: pd.DataFrame,
    multiplier: float,
    registry: Path,
    now: datetime,
) -> CostStress:
    slippage = config.backtest.slippage_pct * multiplier
    stressed = config.model_copy(
        update={"backtest": config.backtest.model_copy(update={"slippage_pct": slippage})}
    )
    strategy = configured_strategy(config)
    run = run_strategy(
        strategy,
        opens,
        closes,
        strategy.warm_up,
        stressed.backtest,
        rebalance_policy(config.strategy),
    )
    results = _results(run.returns)
    record_variant(registry, variant_definition(stressed), results, now, COST_STRESS)
    return CostStress(slippage_multiplier=multiplier, slippage_pct=slippage, **results)
