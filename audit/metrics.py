"""Métriques de performance et comparaison aux objectifs, à partir de rendements journaliers.

L'annualisation repose sur 252 séances par an (`quantstats`), et non sur les 365 jours utilisés par
défaut par `vectorbt`, qui gonfleraient le Sharpe d'environ 20 %. Le taux sans risque est de 0 :
le Sharpe est donc flatteur, à stratégie comme à benchmark égaux, et le cash ne rapporte rien non
plus dans le backtest.
"""

import math
from datetime import date
from enum import StrEnum
from statistics import NormalDist
from typing import Any

import numpy as np
import pandas as pd
import quantstats as qs
from pydantic import BaseModel, ConfigDict

from bot.config import PerformanceTargets, RiskLimits


class PerformanceMetrics(BaseModel):
    """Performance ajustée du risque sur une série de rendements."""

    model_config = ConfigDict(frozen=True, ser_json_inf_nan="strings")

    annual_return: float
    annual_volatility: float
    sharpe: float
    sortino: float
    calmar: float
    max_drawdown: float
    max_drawdown_sessions: int
    positive_months: float
    sessions: int


class TradeStatistics(BaseModel):
    """Statistiques des trades clôturés. `profit_factor` est vide sans trade perdant."""

    model_config = ConfigDict(frozen=True, ser_json_inf_nan="strings")

    closed_trades: int
    win_rate: float
    profit_factor: float | None
    average_win_over_loss: float | None


class KillSwitchReplay(BaseModel):
    """Déclenchements qu'aurait eus le kill switch, le backtest ne l'appliquant pas."""

    model_config = ConfigDict(frozen=True, ser_json_inf_nan="strings")

    first_daily_loss: date | None
    first_drawdown: date | None
    daily_loss_sessions: int
    drawdown_sessions: int


class Status(StrEnum):
    """Résultat d'un critère."""

    PASS = "pass"
    ALERT = "alert"
    FAIL = "fail"
    INSUFFICIENT = "insufficient"


class Criterion(BaseModel):
    """Critère de performance évalué."""

    model_config = ConfigDict(frozen=True, ser_json_inf_nan="strings")

    name: str
    value: float
    target: float
    status: Status


def _scalar(value: Any) -> float:
    return float(value)


def _longest_underwater(returns: pd.Series) -> int:
    equity = (1 + returns).cumprod()
    underwater = (equity < equity.cummax().clip(lower=1.0)).to_numpy()
    longest = current = 0
    for is_underwater in underwater:
        current = current + 1 if is_underwater else 0
        longest = max(longest, current)
    return longest


def compute_metrics(returns: pd.Series) -> PerformanceMetrics:
    """Retourne les métriques d'une série de rendements journaliers nets."""
    if returns.empty or returns.isna().any():
        raise ValueError("Les rendements doivent être non vides et complets")

    monthly = (1 + returns).groupby(returns.index.to_period("M")).prod() - 1  # type: ignore[attr-defined]
    annual_return = _scalar(qs.stats.cagr(returns))
    max_drawdown = _scalar(qs.stats.max_drawdown(returns))
    return PerformanceMetrics(
        annual_return=annual_return,
        annual_volatility=_scalar(qs.stats.volatility(returns)),
        sharpe=_scalar(qs.stats.sharpe(returns)),
        sortino=_scalar(qs.stats.sortino(returns)),
        calmar=annual_return / abs(max_drawdown) if max_drawdown < 0 else float("nan"),
        max_drawdown=max_drawdown,
        max_drawdown_sessions=_longest_underwater(returns),
        positive_months=float((monthly > 0).mean()),
        sessions=len(returns),
    )


def compute_trade_statistics(pnl: np.ndarray) -> TradeStatistics:
    """Retourne les statistiques à partir du résultat net de chaque trade clôturé."""
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    has_both = len(wins) > 0 and len(losses) > 0
    return TradeStatistics(
        closed_trades=len(pnl),
        win_rate=float(len(wins) / len(pnl)) if len(pnl) else 0.0,
        profit_factor=float(wins.sum() / -losses.sum()) if len(losses) else None,
        average_win_over_loss=float(wins.mean() / -losses.mean()) if has_both else None,
    )


def replay_kill_switch(returns: pd.Series, limits: RiskLimits) -> KillSwitchReplay:
    """Repère les séances où les limites du kill switch auraient été atteintes.

    Le contrôle se fait à la clôture : une limite franchie en cours de séance puis effacée
    n'apparaît pas, le résultat est donc un minimum.
    """
    equity = (1 + returns).cumprod()
    peak = equity.cummax().clip(lower=1.0)
    daily_loss = returns <= -limits.max_daily_loss_pct
    drawdown = equity <= peak * (1 - limits.max_drawdown_pct)

    def first(mask: pd.Series) -> date | None:
        return pd.Timestamp(str(mask.idxmax())).date() if mask.any() else None

    return KillSwitchReplay(
        first_daily_loss=first(daily_loss),
        first_drawdown=first(drawdown),
        daily_loss_sessions=int(daily_loss.sum()),
        drawdown_sessions=int(drawdown.sum()),
    )


def _minimum(name: str, value: float, target: float) -> Criterion:
    return Criterion(
        name=name,
        value=value,
        target=target,
        status=Status.PASS if value >= target else Status.FAIL,
    )


def evaluate_targets(
    metrics: PerformanceMetrics, benchmark: PerformanceMetrics, targets: PerformanceTargets
) -> list[Criterion]:
    """Compare la stratégie aux objectifs, critère par critère. Une valeur NaN échoue."""
    drawdown = abs(metrics.max_drawdown)
    if drawdown <= targets.max_drawdown_alert:
        drawdown_status = Status.PASS
    elif drawdown <= targets.max_drawdown_reject:
        drawdown_status = Status.ALERT
    else:
        drawdown_status = Status.FAIL
    volatility_ok = metrics.annual_volatility < benchmark.annual_volatility
    return [
        _minimum("annual_return", metrics.annual_return, targets.min_annual_return),
        Criterion(
            name="max_drawdown",
            value=drawdown,
            target=targets.max_drawdown_alert,
            status=drawdown_status,
        ),
        Criterion(
            name="volatility_below_benchmark",
            value=metrics.annual_volatility,
            target=benchmark.annual_volatility,
            status=Status.PASS if volatility_ok else Status.FAIL,
        ),
        _minimum("sharpe", metrics.sharpe, targets.min_sharpe),
        _minimum("calmar", metrics.calmar, targets.min_calmar),
        _minimum("positive_months", metrics.positive_months, targets.min_positive_months),
        _minimum("sharpe_above_benchmark", metrics.sharpe, benchmark.sharpe),
    ]


def suspicious_results(metrics: PerformanceMetrics, targets: PerformanceTargets) -> list[str]:
    """Retourne les résultats trop bons pour être crus sans enquête."""
    flags = []
    if metrics.annual_return > targets.suspicious_annual_return:
        flags.append("annual_return")
    if metrics.sharpe > targets.suspicious_sharpe:
        flags.append("sharpe")
    return flags


TRADING_DAYS = 252
_EULER_GAMMA = 0.5772156649015329


class DeflatedSharpe(BaseModel):
    """Sharpe dégonflé : probabilité que le vrai Sharpe dépasse le meilleur Sharpe dû au hasard.

    `expected_max_sharpe` est le Sharpe annualisé que le meilleur de `trials` essais sans aucun
    pouvoir prédictif atteindrait en moyenne, compte tenu de la dispersion des essais.
    """

    model_config = ConfigDict(frozen=True, ser_json_inf_nan="strings")

    sharpe: float
    trials: int
    expected_max_sharpe: float
    probability: float


def deflated_sharpe(returns: pd.Series, trial_sharpes: list[float]) -> DeflatedSharpe:
    """Retourne le Sharpe dégonflé de Bailey et López de Prado (2014).

    `trial_sharpes` contient le Sharpe annualisé de chaque variante testée, celle-ci comprise. Avec
    un seul essai, le seuil est 0 et le résultat est le Sharpe probabiliste. L'asymétrie et
    l'épaisseur des queues des rendements réduisent la confiance accordée au Sharpe observé.
    """
    if returns.empty or returns.isna().any() or len(returns) < 2:
        raise ValueError("Les rendements doivent être complets et compter au moins deux séances")
    if not trial_sharpes or not all(math.isfinite(value) for value in trial_sharpes):
        raise ValueError("Les Sharpe des essais doivent être renseignés et finis")

    normal = NormalDist()
    scale = math.sqrt(TRADING_DAYS)
    std = float(returns.std())
    sharpe = float(returns.mean()) / std if std > 0 else float("nan")
    trials = len(trial_sharpes)
    if trials > 1:
        dispersion = float(np.std(np.asarray(trial_sharpes) / scale, ddof=1))
        expected_max = dispersion * (
            (1 - _EULER_GAMMA) * normal.inv_cdf(1 - 1 / trials)
            + _EULER_GAMMA * normal.inv_cdf(1 - 1 / (trials * math.e))
        )
    else:
        expected_max = 0.0

    skew = _scalar(returns.skew())
    kurtosis = _scalar(returns.kurt()) + 3
    variance = 1 - skew * sharpe + (kurtosis - 1) / 4 * sharpe**2
    if not (math.isfinite(sharpe) and math.isfinite(variance) and variance > 0):
        probability = float("nan")
    else:
        z = (sharpe - expected_max) * math.sqrt(len(returns) - 1) / math.sqrt(variance)
        probability = normal.cdf(z)
    return DeflatedSharpe(
        sharpe=sharpe * scale,
        trials=trials,
        expected_max_sharpe=expected_max * scale,
        probability=probability,
    )
