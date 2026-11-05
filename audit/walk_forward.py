"""Validation walk-forward, à l'intérieur de l'in-sample.

Deux lectures des mêmes fenêtres :

- réglage fixe : la stratégie telle qu'elle sera déployée, mesurée fenêtre de test par fenêtre de
  test. C'est la régularité de la performance dans le temps.
- réglage réoptimisé : sur chaque fenêtre d'entraînement, on retient le réglage de la grille de
  sensibilité au meilleur Sharpe, puis on le mesure sur la fenêtre de test suivante. L'efficacité
  compare le Sharpe obtenu en test à celui obtenu en entraînement : une efficacité faible signale
  qu'optimiser sur le passé ne se transmet pas à la suite, donc un risque de surajustement.

Le choix d'une fenêtre n'utilise que des séances antérieures à sa fenêtre de test. Le réglage
retenu s'applique dès la première ouverture de la fenêtre de test, à partir de la dernière clôture
de l'entraînement, ce qui reproduit une décision prise ce soir-là. Avec un rééquilibrage mensuel,
le changement de réglage ne produit d'ordres qu'au premier rééquilibrage suivant.
"""

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict

from audit.backtest import (
    BacktestError,
    configured_strategy,
    executed_targets,
    order_targets,
    run_weights,
)
from audit.metrics import PerformanceMetrics, compute_metrics
from bot.config import StrategyConfig, sensitivity_grid
from bot.strategy.factory import rebalance_policy


class WindowResult(BaseModel):
    """Résultat d'une fenêtre de test."""

    model_config = ConfigDict(frozen=True, ser_json_inf_nan="strings")

    train_start: str
    test_start: str
    test_end: str
    selected: dict[str, float]
    train_sharpe: float
    test_return: float
    test_sharpe: float
    fixed_test_return: float
    fixed_test_sharpe: float


class WalkForwardReport(BaseModel):
    """Résultat du walk-forward."""

    model_config = ConfigDict(frozen=True, ser_json_inf_nan="strings")

    windows: list[WindowResult]
    fixed_positive_share: float
    reoptimized: PerformanceMetrics
    mean_train_sharpe: float
    efficiency: float


@dataclass(frozen=True)
class Window:
    """Bornes d'une fenêtre, en positions de séance. La fin est exclue."""

    train_start: int
    test_start: int
    test_end: int


def window_bounds(sessions: int, first: int, train: int, test: int) -> list[Window]:
    """Découpe les séances en fenêtres glissantes complètes, décalées de la durée du test."""
    windows = []
    start = first
    while start + train + test <= sessions:
        windows.append(Window(start, start + train, start + train + test))
        start += test
    return windows


def _total_return(returns: pd.Series) -> float:
    return float(np.prod(1 + returns.to_numpy()) - 1)


def _sharpe(returns: pd.Series) -> float:
    return compute_metrics(returns).sharpe


def _rank(value: float) -> float:
    return value if math.isfinite(value) else -math.inf


def run_walk_forward(
    config: StrategyConfig, opens: pd.DataFrame, closes: pd.DataFrame
) -> WalkForwardReport:
    """Lance le walk-forward sur les prix in-sample fournis."""
    settings = config.audit.walk_forward
    candidates = sensitivity_grid(config.strategy, config.audit.sensitivity)
    parameters = config.audit.sensitivity.parameters[config.strategy.name]
    strategies = [configured_strategy(config, candidate) for candidate in candidates]
    first = max(strategy.warm_up for strategy in strategies)
    windows = window_bounds(len(closes), first, settings.train_sessions, settings.test_sessions)
    if not windows:
        raise BacktestError("Trop peu de séances pour une seule fenêtre de walk-forward")

    last = windows[-1].test_end
    opens = opens.iloc[:last]
    closes = closes.iloc[:last]
    rebalance = rebalance_policy(config.strategy)
    executed = [executed_targets(strategy, closes) for strategy in strategies]
    returns = [
        run_weights(
            opens, closes, order_targets(held, first, rebalance), first, config.backtest
        ).returns
        for held in executed
    ]

    stitched = pd.DataFrame(0.0, index=closes.index, columns=closes.columns)
    selections = []
    for window in windows:
        train_dates = closes.index[window.train_start : window.test_start]
        train_sharpes = [_sharpe(series.loc[train_dates]) for series in returns]
        best = int(np.argmax([_rank(value) for value in train_sharpes]))
        selections.append((best, train_sharpes[best]))
        rows = slice(window.test_start, window.test_end)
        stitched.iloc[rows] = executed[best].iloc[rows].to_numpy()

    first_test = windows[0].test_start
    reoptimized = run_weights(
        opens,
        closes,
        order_targets(stitched, first_test, rebalance),
        first_test,
        config.backtest,
    ).returns

    chosen = candidates.index(config.strategy)
    results = []
    for window, (best, train_sharpe) in zip(windows, selections, strict=True):
        test_dates = closes.index[window.test_start : window.test_end]
        fixed = returns[chosen].loc[test_dates]
        results.append(
            WindowResult(
                train_start=_iso(closes.index[window.train_start]),
                test_start=_iso(test_dates[0]),
                test_end=_iso(test_dates[-1]),
                selected={
                    name: float(getattr(candidates[best].params, name)) for name in parameters
                },
                train_sharpe=train_sharpe,
                test_return=_total_return(reoptimized.loc[test_dates]),
                test_sharpe=_sharpe(reoptimized.loc[test_dates]),
                fixed_test_return=_total_return(fixed),
                fixed_test_sharpe=_sharpe(fixed),
            )
        )

    metrics = compute_metrics(reoptimized)
    mean_train = float(np.mean([train for _, train in selections]))
    efficiency = metrics.sharpe / mean_train if mean_train > 0 else float("nan")
    return WalkForwardReport(
        windows=results,
        fixed_positive_share=float(np.mean([item.fixed_test_return > 0 for item in results])),
        reoptimized=metrics,
        mean_train_sharpe=mean_train,
        efficiency=efficiency,
    )


def _iso(timestamp: object) -> str:
    return pd.Timestamp(str(timestamp)).date().isoformat()
