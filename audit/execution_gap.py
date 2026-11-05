"""Écart d'exécution entre le backtest et le paper trading.

Compare le slippage observé sur les ordres exécutés, enregistré au journal, au slippage supposé
par le backtest. Les deux ne mesurent pas tout à fait la même chose : le backtest exécute à
l'ouverture, le bot au prix du moment, comparé au dernier prix IEX. Le paper trading d'Alpaca
sous-estime aussi le slippage réel, car ses exécutions sont simulées.

La comparaison complète, backtest et paper sur la même période, demande une période de paper
trading avec la stratégie finale : c'est l'étape 9.
"""

import math
import sqlite3
from pathlib import Path

import numpy as np
from pydantic import BaseModel, ConfigDict

# Les décisions du kill switch n'ont pas de prix de référence : elles sont exclues d'office.
_FILLS_QUERY = """
SELECT slippage_pct FROM decisions
WHERE filled_price IS NOT NULL AND slippage_pct IS NOT NULL AND filled_quantity > 0
"""


class ExecutionGapReport(BaseModel):
    """Slippage observé en paper trading. Positif quand l'exécution est défavorable."""

    model_config = ConfigDict(frozen=True, ser_json_inf_nan="strings")

    journal_found: bool
    fills: int
    assumed_slippage_pct: float
    mean_slippage_pct: float | None
    p95_slippage_pct: float | None


def read_fill_slippages(journal: Path) -> list[float] | None:
    """Retourne le slippage de chaque ordre exécuté, ou None si le journal n'existe pas.

    Le journal est ouvert en lecture seule : l'audit ne doit jamais le modifier.
    """
    if not journal.exists():
        return None
    connection = sqlite3.connect(f"{journal.resolve().as_uri()}?mode=ro", uri=True)
    try:
        rows = connection.execute(_FILLS_QUERY).fetchall()
    finally:
        connection.close()
    return [float(row[0]) for row in rows if math.isfinite(float(row[0]))]


def measure_execution_gap(journal: Path, assumed_slippage_pct: float) -> ExecutionGapReport:
    """Mesure le slippage observé et le compare à celui du backtest."""
    slippages = read_fill_slippages(journal)
    values = np.asarray(slippages or [], dtype=float)
    filled = len(values) > 0
    return ExecutionGapReport(
        journal_found=slippages is not None,
        fills=len(values),
        assumed_slippage_pct=assumed_slippage_pct,
        mean_slippage_pct=float(values.mean()) if filled else None,
        p95_slippage_pct=float(np.percentile(values, 95)) if filled else None,
    )
