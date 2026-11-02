"""Stratégie de référence : croisement de moyennes mobiles, à l'achat ou hors marché."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from bot.config import MovingAverageParams
from bot.strategy.base import FLAT, LONG


@dataclass(frozen=True)
class MovingAverageCrossover:
    """Vise une position longue tant que la moyenne rapide est au-dessus de la moyenne lente.

    Le signal dépend de l'état des moyennes et non de l'instant du croisement : le même historique
    donne la même position, même après un redémarrage du bot. Le signal reste hors marché tant
    qu'une moyenne n'est pas calculable, pendant le préchauffage ou après un prix manquant.

    La comparaison ne dépend pas de l'échelle des prix. Un ajustement futur qui multiplie tout
    l'historique par un même facteur ne change donc aucun signal passé.
    """

    params: MovingAverageParams

    def generate(self, closes: pd.DataFrame) -> pd.DataFrame:
        """Retourne la position souhaitée après chaque clôture, par séance et par symbole.

        Lève `ValueError` si les séances ne sont pas uniques et triées, car un calcul glissant sur
        des séances dans le désordre pourrait utiliser des données futures.
        """
        if not (closes.index.is_unique and closes.index.is_monotonic_increasing):
            raise ValueError("Les séances doivent être uniques et triées")

        fast_window = self.params.fast_window
        slow_window = self.params.slow_window
        fast = closes.rolling(fast_window, min_periods=fast_window).mean()
        slow = closes.rolling(slow_window, min_periods=slow_window).mean()
        positions = np.where(fast > slow, LONG, FLAT)
        return pd.DataFrame(positions, index=closes.index, columns=closes.columns)
