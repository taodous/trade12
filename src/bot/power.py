"""Empêche la mise en veille du PC pendant une séance planifiée.

Sous Windows, `SetThreadExecutionState` garde le système éveillé tant que le processus le demande.
L'écran peut s'éteindre. La demande est retirée à la sortie, même sur erreur. Ailleurs, rien ne
change : la gestion de l'énergie relève alors de la machine qui héberge le bot.
"""

import sys
from collections.abc import Iterator
from contextlib import contextmanager

import structlog

logger = structlog.get_logger()

# Drapeaux de l'API Windows SetThreadExecutionState.
ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001


@contextmanager
def keep_awake(*, enabled: bool = True) -> Iterator[bool]:
    """Garde le système éveillé pendant le bloc. Retourne True si la demande est active."""
    if not enabled or sys.platform != "win32":
        yield False
        return

    import ctypes

    kernel32 = ctypes.windll.kernel32
    if kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED) == 0:
        logger.warning("keep_awake_failed")
        yield False
        return
    logger.info("keep_awake_enabled")
    try:
        yield True
    finally:
        kernel32.SetThreadExecutionState(ES_CONTINUOUS)
