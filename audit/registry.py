"""Registre des variantes testées, pour mesurer le biais de tests multiples."""

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

# Variante évaluée sous des coûts majorés : la même stratégie, pas un essai de plus.
COST_STRESS = "cost_stress"
BACKTEST = "backtest"
SENSITIVITY = "sensitivity"


def variant_id(definition: dict[str, Any]) -> str:
    """Retourne un identifiant stable pour une définition de variante."""
    canonical = json.dumps(definition, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]


def load_registry(path: Path) -> list[dict[str, Any]]:
    """Retourne les entrées du registre, dans l'ordre d'inscription."""
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def record_variant(
    path: Path,
    definition: dict[str, Any],
    results: dict[str, float],
    now: datetime,
    purpose: str = BACKTEST,
) -> bool:
    """Ajoute une variante au registre et indique si elle était nouvelle.

    Une variante déjà présente n'est pas réinscrite : relancer le même test n'est pas un test de
    plus, et son premier résultat reste en place. `purpose` ne fait pas partie de l'identifiant.
    """
    identifier = variant_id(definition)
    if any(entry["id"] == identifier for entry in load_registry(path)):
        return False

    entry = {
        "id": identifier,
        "first_run": now.isoformat(),
        "purpose": purpose,
        "definition": definition,
        "results": results,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as registry:
        registry.write(json.dumps(entry, sort_keys=True, default=str) + "\n")
    return True


def trial_sharpes(path: Path) -> list[float]:
    """Retourne le Sharpe de chaque essai distinct, hors variantes à coûts majorés.

    Les entrées antérieures à l'ajout de `purpose` sont des backtests.
    """
    return [
        float(entry["results"]["sharpe"])
        for entry in load_registry(path)
        if entry.get("purpose", BACKTEST) != COST_STRESS
    ]
