"""Robustesse Monte Carlo : distribution des drawdowns et des rendements possibles.

Deux tirages complémentaires, reproductibles grâce à la graine de la configuration :

- l'ordre des trades clôturés est mélangé. La courbe ne contient que leurs résultats, sans la
  valeur latente des positions ouvertes, et ignore que plusieurs trades se chevauchent : elle
  sous-estime le drawdown réel. Avec peu de trades, la distribution est grossière.
- les rendements journaliers sont tirés par blocs consécutifs, avec remise. Les blocs gardent la
  persistance de la volatilité sur quelques semaines, mais coupent les tendances plus longues.

Le critère retient le pire des deux percentiles 95.
"""

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict

from audit.metrics import TRADING_DAYS
from bot.config import MonteCarloConfig

PERCENTILES = (5, 50, 95)


class Distribution(BaseModel):
    """Percentiles 5, 50 et 95 d'une grandeur simulée."""

    model_config = ConfigDict(frozen=True, ser_json_inf_nan="strings")

    p5: float
    p50: float
    p95: float


class MonteCarloReport(BaseModel):
    """Résultat des deux tirages. Les drawdowns sont des pertes positives (0.12 = 12 %)."""

    model_config = ConfigDict(frozen=True, ser_json_inf_nan="strings")

    simulations: int
    trades: int
    trade_order_drawdown: Distribution | None
    block_drawdown: Distribution
    block_annual_return: Distribution
    worst_drawdown_p95: float


def _distribution(values: np.ndarray) -> Distribution:
    p5, p50, p95 = np.percentile(values, PERCENTILES)
    return Distribution(p5=float(p5), p50=float(p50), p95=float(p95))


def max_drawdowns(equity: np.ndarray) -> np.ndarray:
    """Retourne le drawdown maximal de chaque ligne d'une matrice de courbes de capital."""
    peaks = np.maximum.accumulate(equity, axis=1)
    return np.asarray(np.max(1 - equity / peaks, axis=1))


def trade_order_drawdowns(
    pnl: np.ndarray, initial_capital: float, simulations: int, rng: np.random.Generator
) -> np.ndarray:
    """Retourne le drawdown maximal de chaque permutation des résultats des trades.

    Le capital de départ fait partie de la courbe : une perte dès le premier trade compte.
    """
    orders = np.array([rng.permutation(pnl) for _ in range(simulations)])
    start = np.full((simulations, 1), initial_capital)
    equity = np.hstack([start, initial_capital + np.cumsum(orders, axis=1)])
    return max_drawdowns(equity)


def block_bootstrap(
    returns: np.ndarray, block: int, simulations: int, rng: np.random.Generator
) -> np.ndarray:
    """Retourne des séries de rendements de même longueur, faites de blocs tirés avec remise.

    Les blocs sont pris sur la série vue comme circulaire, pour que les dernières séances aient
    autant de chances d'être tirées que les autres.
    """
    length = len(returns)
    blocks = -(-length // block)
    starts = rng.integers(0, length, size=(simulations, blocks))
    offsets = np.arange(block)
    indices = (starts[:, :, None] + offsets) % length
    return np.asarray(returns[indices.reshape(simulations, -1)[:, :length]], dtype=float)


def run_monte_carlo(
    returns: pd.Series, pnl: np.ndarray, initial_capital: float, settings: MonteCarloConfig
) -> MonteCarloReport:
    """Lance les deux tirages sur les rendements et les trades du backtest."""
    if returns.empty or returns.isna().any():
        raise ValueError("Les rendements doivent être non vides et complets")
    rng = np.random.default_rng(settings.seed)

    trade_distribution = None
    if len(pnl):
        drawdowns = trade_order_drawdowns(pnl, initial_capital, settings.simulations, rng)
        trade_distribution = _distribution(drawdowns)

    samples = block_bootstrap(
        returns.to_numpy(), settings.block_sessions, settings.simulations, rng
    )
    equity = np.cumprod(1 + samples, axis=1)
    equity = np.hstack([np.ones((settings.simulations, 1)), equity])
    block_drawdowns = max_drawdowns(equity)
    years = samples.shape[1] / TRADING_DAYS
    annual = equity[:, -1] ** (1 / years) - 1

    block_distribution = _distribution(block_drawdowns)
    worst = block_distribution.p95
    if trade_distribution is not None:
        worst = max(worst, trade_distribution.p95)
    return MonteCarloReport(
        simulations=settings.simulations,
        trades=len(pnl),
        trade_order_drawdown=trade_distribution,
        block_drawdown=block_distribution,
        block_annual_return=_distribution(annual),
        worst_drawdown_p95=worst,
    )
