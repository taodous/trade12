"""Adaptation d'une stratégie à signaux en stratégie à poids."""

from dataclasses import dataclass

import pandas as pd

from bot.strategy.base import LONG, Strategy


@dataclass(frozen=True)
class EqualWeight:
    """Vise le même poids pour chaque symbole en position longue, et 0 sinon.

    Sans rééquilibrage, c'est le comportement de la stratégie de référence : une position est prise
    au poids visé, puis évolue librement jusqu'au signal de sortie.
    """

    signals: Strategy
    weight: float
    warm_up: int
    required_symbols: frozenset[str] = frozenset()

    @property
    def max_weight(self) -> float:
        """Retourne le poids de chaque position."""
        return self.weight

    def target_weights(self, closes: pd.DataFrame) -> pd.DataFrame:
        """Retourne `weight` pour chaque signal long, 0 sinon."""
        return self.signals.generate(closes).eq(LONG).astype(float) * self.weight
