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
