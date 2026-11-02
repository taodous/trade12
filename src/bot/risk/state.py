"""État du risque conservé entre deux démarrages du bot.

Un redémarrage ne doit ni réarmer le kill switch, ni oublier le plus haut du capital, ni remettre à
zéro le compteur d'ordres : une boucle folle pourrait sinon les contourner en plantant.
"""

from datetime import date, datetime
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, ConfigDict, ValidationError

from bot.risk.models import HaltReason


class RiskStateError(RuntimeError):
    """État du risque illisible. Le bot doit s'arrêter plutôt que repartir d'un état vierge."""


class RiskState(BaseModel):
    """État persistant du risque."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    peak_equity: float | None = None
    halt_reason: HaltReason | None = None
    halted_at: datetime | None = None
    order_date: date | None = None
    orders_count: int = 0


class RiskStateStore(Protocol):
    """Lecture et écriture de l'état du risque."""

    def load(self) -> RiskState:
        """Retourne l'état enregistré, ou un état vierge au premier lancement."""
        ...

    def save(self, state: RiskState) -> None:
        """Enregistre l'état."""
        ...


class JsonRiskStateStore:
    """Conserve l'état du risque dans un fichier JSON, écrit de façon atomique."""

    def __init__(self, path: Path) -> None:
        """Associe le stockage à `path`, créé à la première écriture."""
        self._path = path

    def load(self) -> RiskState:
        """Retourne l'état enregistré, ou lève `RiskStateError` s'il est illisible."""
        try:
            return RiskState.model_validate_json(self._path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return RiskState()
        except (OSError, ValidationError) as exc:
            raise RiskStateError(f"État du risque illisible dans {self._path}") from exc

    def save(self, state: RiskState) -> None:
        """Enregistre l'état, sans jamais laisser de fichier partiellement écrit."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self._path.with_suffix(".tmp")
        temporary.write_text(state.model_dump_json(indent=2), encoding="utf-8")
        temporary.replace(self._path)
