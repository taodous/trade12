"""Registre des variantes testées, pour mesurer le biais de tests multiples."""

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any


def variant_id(definition: dict[str, Any]) -> str:
    """Retourne un identifiant stable pour une définition de variante."""
    canonical = json.dumps(definition, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]


def record_variant(
    path: Path, definition: dict[str, Any], results: dict[str, float], now: datetime
) -> bool:
    """Ajoute une variante au registre et indique si elle était nouvelle.

    Une variante déjà présente n'est pas réinscrite : relancer le même test n'est pas un test de
    plus, et son premier résultat reste en place.
    """
    identifier = variant_id(definition)
    if path.exists():
        lines = path.read_text(encoding="utf-8").splitlines()
        if any(json.loads(line)["id"] == identifier for line in lines if line.strip()):
            return False

    entry = {
        "id": identifier,
        "first_run": now.isoformat(),
        "definition": definition,
        "results": results,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as registry:
        registry.write(json.dumps(entry, sort_keys=True, default=str) + "\n")
    return True
