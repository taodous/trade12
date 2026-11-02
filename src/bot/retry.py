"""Backoff exponentiel sur les erreurs de limite de débit de l'API."""

import time
from collections.abc import Callable
from http import HTTPStatus

import structlog
from alpaca.common.exceptions import APIError

from bot.config import RetryConfig

logger = structlog.get_logger()


def with_backoff[T](
    call: Callable[[], T],
    config: RetryConfig,
    *,
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    """Exécute `call` en le relançant avec un délai croissant sur erreur 429.

    Toute autre erreur est propagée immédiatement, comme la dernière erreur 429.
    """
    attempt = 1
    while True:
        try:
            return call()
        except APIError as exc:
            if exc.status_code != HTTPStatus.TOO_MANY_REQUESTS or attempt >= config.max_attempts:
                raise
            delay = min(config.base_delay_seconds * 2 ** (attempt - 1), config.max_delay_seconds)
            logger.warning("api_rate_limited", attempt=attempt, delay_seconds=delay)
            sleep(delay)
            attempt += 1
