"""Vérifie la configuration et la connexion au compte paper, sans envoyer d'ordre."""

import sys

import structlog
from alpaca.common.exceptions import APIError
from requests.exceptions import RequestException

from bot.config import ConfigError, load_settings, load_strategy_config
from bot.execution.client import (
    AccountDataError,
    PaperModeError,
    create_trading_client,
    read_account,
)
from bot.logs import configure_logging

logger = structlog.get_logger()


def main() -> int:
    """Lance la vérification et retourne le code de sortie du processus."""
    configure_logging()
    try:
        load_strategy_config()
        account = read_account(create_trading_client(load_settings()))
    except (ConfigError, PaperModeError, AccountDataError) as exc:
        logger.error("connection_check_failed", reason=str(exc))
        return 1
    except APIError as exc:
        logger.error("connection_check_failed", reason="api_error", status_code=exc.status_code)
        return 1
    except RequestException as exc:
        logger.error("connection_check_failed", reason="network_error", error=type(exc).__name__)
        return 1

    logger.info(
        "connection_check_ok",
        status=account.status.value,
        currency=account.currency,
        equity=str(account.equity),
        cash=str(account.cash),
        buying_power=str(account.buying_power),
        can_trade=account.can_trade,
    )
    return 0 if account.can_trade else 1


if __name__ == "__main__":
    sys.exit(main())
