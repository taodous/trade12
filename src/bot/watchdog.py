"""Chien de garde : alerte si le bot ne donne plus signe de vie pendant une séance ouverte.

Lancé par la tâche planifiée `TradeBotWatchdog`, toutes les 30 minutes en semaine. Il couvre ce que
le bot ne peut pas signaler lui-même : un plantage, une fenêtre fermée, un PC redémarré. Il ne fait
que lire : l'horloge Alpaca, les logs et le journal.
"""

import sys
from datetime import UTC, datetime

import structlog
from alpaca.common.exceptions import APIError
from requests.exceptions import RequestException

from bot.alerts import Alert, Alerter, Severity, build_alerter
from bot.config import ConfigError, StrategyConfig, load_settings, load_strategy_config
from bot.execution.broker import Broker, BrokerClient, BrokerError, MarketClock
from bot.execution.client import PaperModeError, create_trading_client
from bot.heartbeat import Heartbeat, bot_heartbeat
from bot.logs import configure_logging

logger = structlog.get_logger()

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_BOT_INACTIVE = 2


def check(clock: MarketClock, beat: Heartbeat) -> Alert | None:
    """Retourne l'alerte à envoyer, ou None si tout va bien ou si le marché est fermé."""
    if not clock.is_open or beat.active:
        return None
    seen = f"{beat.last_seen:%d/%m %H:%M} UTC" if beat.last_seen else "jamais"
    return Alert(
        "Bot inactif pendant la séance",
        f"Marché ouvert, mais aucun signe de vie du bot depuis : {seen}. Le kill switch n'est pas "
        "surveillé. Vérifier la tâche TradeBotPaper et logs/stderr.log.",
        Severity.CRITICAL,
    )


def watch(config: StrategyConfig, client: BrokerClient, alerter: Alerter, now: datetime) -> int:
    """Vérifie le bot une fois et retourne le code de sortie."""
    clock = Broker(client, config.api_retry).clock()
    alert = check(clock, bot_heartbeat(config, now))
    if alert is None:
        logger.info("watchdog_ok", market_open=clock.is_open)
        return EXIT_OK
    alerter.send(alert)
    return EXIT_BOT_INACTIVE


def main() -> int:
    """Lance une vérification."""
    configure_logging()
    alerter = Alerter([])
    try:
        config = load_strategy_config()
        settings = load_settings()
        alerter = build_alerter(config.alerts, settings)
        return watch(config, create_trading_client(settings), alerter, datetime.now(UTC))
    except (ConfigError, PaperModeError, OSError) as exc:
        logger.error("watchdog_failed", error=type(exc).__name__, reason=str(exc))
        return EXIT_ERROR
    except (APIError, RequestException, BrokerError) as exc:
        # Sans horloge, impossible de savoir si le marché est ouvert : on prévient quand même.
        logger.error("watchdog_failed", error=type(exc).__name__)
        alerter.send(
            Alert(
                "Chien de garde : API Alpaca injoignable",
                f"{type(exc).__name__}. Si le marché est ouvert, le bot est sans doute en panne.",
                Severity.WARNING,
            )
        )
        return EXIT_ERROR


if __name__ == "__main__":
    sys.exit(main())
