"""Configuration des logs structurés."""

import sys
from pathlib import Path
from typing import Any, TextIO

import structlog


class TeeLogger:
    """Écrit chaque ligne de log sur la console et, si besoin, dans des fichiers.

    Une tâche planifiée n'a pas de console : sans fichier, ses logs seraient perdus.
    """

    def __init__(self, files: list[TextIO]) -> None:
        """Associe le logger à ses fichiers, en plus de la console."""
        self._files = files

    def msg(self, message: str) -> None:
        """Écrit une ligne partout, aussitôt vidée pour survivre à un arrêt brutal.

        La console est relue à chaque ligne : elle peut avoir été remplacée depuis la configuration.
        """
        for stream in [sys.stdout, *self._files]:
            stream.write(message + "\n")
            stream.flush()

    debug = info = warning = warn = error = critical = exception = fatal = log = msg


_files: list[TextIO] = []


def configure_logging(log_file: Path | None = None) -> None:
    """Configure structlog pour écrire des logs JSON horodatés en UTC.

    Avec `log_file`, chaque ligne est aussi ajoutée à ce fichier, créé avec son dossier si besoin.
    Un nouvel appel remplace la configuration précédente et ferme l'ancien fichier.
    """
    for stream in _files:
        stream.close()
    _files.clear()
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        _files.append(log_file.open("a", encoding="utf-8"))
    tee = TeeLogger(list(_files))

    def factory(*args: Any) -> TeeLogger:
        return tee

    structlog.configure(
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.JSONRenderer(),
        ],
        logger_factory=factory,
        cache_logger_on_first_use=False,
    )
