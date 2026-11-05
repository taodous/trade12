"""Interface commune aux stratégies, utilisée à l'identique en backtest et en paper trading."""

from typing import Protocol

import pandas as pd

LONG = 1
FLAT = 0


class Strategy(Protocol):
    """Stratégie pure : mêmes données en entrée, mêmes signaux en sortie, sans effet de bord."""

    def generate(self, closes: pd.DataFrame) -> pd.DataFrame:
        """Retourne la position souhaitée après chaque clôture, par séance et par symbole.

        `closes` a une ligne par séance et une colonne par symbole. Le résultat a la même forme et
        contient `LONG` ou `FLAT`. Le signal de la séance t utilise la clôture de t : il ne peut
        être exécuté qu'à partir de la séance suivante.
        """
        ...


class WeightStrategy(Protocol):
    """Stratégie pure qui vise un poids par symbole, en fraction du capital.

    `warm_up` est le nombre de séances nécessaires avant le premier poids fiable. `max_weight` est
    le plus grand poids qu'elle puisse viser pour un symbole. `required_symbols` liste les symboles
    sans lesquels aucun poids n'est fiable, comme celui d'un filtre de régime.
    """

    @property
    def warm_up(self) -> int:
        """Retourne le nombre de séances nécessaires avant le premier poids fiable."""
        ...

    @property
    def max_weight(self) -> float:
        """Retourne le plus grand poids visé pour un symbole."""
        ...

    @property
    def required_symbols(self) -> frozenset[str]:
        """Retourne les symboles dont les données sont indispensables."""
        ...

    def target_weights(self, closes: pd.DataFrame) -> pd.DataFrame:
        """Retourne le poids visé après chaque clôture, par séance et par symbole.

        Le poids de la séance t n'utilise que les clôtures jusqu'à t : il ne peut être exécuté
        qu'à partir de la séance suivante. Chaque poids est entre 0 et `max_weight`, sans valeur
        manquante, et leur somme ne dépasse pas 1.
        """
        ...
