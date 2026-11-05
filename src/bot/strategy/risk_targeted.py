"""Stratégie à risque ciblé : filtre de tendance, inverse de la volatilité, volatilité cible.

À chaque clôture, pour les symboles en tendance haussière :

1. poids de base en inverse de la volatilité récente, normalisés à 1 ;
2. mise à l'échelle pour que la volatilité estimée du portefeuille, covariances comprises, vise
   `target_volatility`, sans dépasser `max_exposure` ;
3. plafond de `max_weight` par symbole, qui ne peut que réduire le risque ;
4. si le filtre de régime est actif et que `regime_symbol` clôture sous sa moyenne, les poids des
   `risky_symbols` sont multipliés par `regime_scale`.

Tout est calculé avec les clôtures jusqu'à t. Un symbole sans historique complet sur la fenêtre de
volatilité reçoit un poids nul.
"""

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from bot.config import RiskTargetedParams

# Convention d'annualisation des volatilités journalières, la même que l'audit.
TRADING_DAYS = 252


@dataclass(frozen=True)
class RiskTargeted:
    """Stratégie pure, mêmes clôtures en entrée, mêmes poids en sortie."""

    params: RiskTargetedParams

    @property
    def warm_up(self) -> int:
        """Retourne le nombre de séances nécessaires à la tendance, à la volatilité et au régime."""
        params = self.params
        regime = params.regime_window if params.regime_filter else 0
        return max(params.trend_window + 1, params.volatility_window + 1, regime)

    @property
    def max_weight(self) -> float:
        """Retourne le plafond par symbole."""
        return self.params.max_weight

    @property
    def required_symbols(self) -> frozenset[str]:
        """Retourne le symbole du filtre de régime, indispensable quand le filtre est actif."""
        return frozenset({self.params.regime_symbol}) if self.params.regime_filter else frozenset()

    def _trend(self, closes: pd.DataFrame) -> pd.DataFrame:
        window = self.params.trend_window
        if self.params.trend == "sma":
            return closes > closes.rolling(window, min_periods=window).mean()
        return closes / closes.shift(window) - 1 > 0

    def _scaled(self, window: np.ndarray) -> np.ndarray:
        """Retourne les poids des symboles actifs à partir de leurs rendements récents."""
        params = self.params
        volatility = window.std(axis=0, ddof=1) * math.sqrt(TRADING_DAYS)
        base = np.where(volatility > 0, 1 / np.where(volatility > 0, volatility, 1.0), 0.0)
        if base.sum() <= 0:
            return np.zeros(window.shape[1])
        base = base / base.sum()
        covariance = np.atleast_2d(np.cov(window, rowvar=False)) * TRADING_DAYS
        portfolio_volatility = math.sqrt(max(float(base @ covariance @ base), 0.0))
        if portfolio_volatility <= 0:
            return np.zeros(window.shape[1])
        scale = min(params.target_volatility / portfolio_volatility, params.max_exposure)
        return np.asarray(np.minimum(base * scale, params.max_weight))

    def target_weights(self, closes: pd.DataFrame) -> pd.DataFrame:
        """Retourne le poids visé après chaque clôture, par séance et par symbole.

        Lève `ValueError` si les séances ne sont pas uniques et triées, ou si le symbole du filtre
        de régime manque alors que le filtre est actif.
        """
        if not (closes.index.is_unique and closes.index.is_monotonic_increasing):
            raise ValueError("Les séances doivent être uniques et triées")
        params = self.params
        if params.regime_filter and params.regime_symbol not in closes.columns:
            raise ValueError(f"Filtre de régime sans données pour {params.regime_symbol}")

        returns = closes.pct_change(fill_method=None).to_numpy()
        trend = self._trend(closes).to_numpy()
        weights = np.zeros(closes.shape)
        window_length = params.volatility_window
        for row in range(self.warm_up - 1, len(closes)):
            window = returns[row - window_length + 1 : row + 1]
            active = trend[row] & np.isfinite(window).all(axis=0)
            if active.any():
                weights[row, active] = self._scaled(window[:, active])

        result = pd.DataFrame(weights, index=closes.index, columns=closes.columns)
        if params.regime_filter:
            regime = closes[params.regime_symbol]
            average = regime.rolling(params.regime_window, min_periods=params.regime_window)
            risk_off = (regime < average.mean()).to_numpy()
            risky = [symbol for symbol in params.risky_symbols if symbol in result.columns]
            result.loc[risk_off, risky] *= params.regime_scale
        return result
