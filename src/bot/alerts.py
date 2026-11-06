"""Alertes : notification Windows sur ce PC et notification sur le téléphone via ntfy.

Une alerte qui échoue est journalisée, mais ne bloque jamais le bot : le module risque et le kill
switch ne dépendent pas des alertes. Le canal ntfy est un secret, jamais affiché ni journalisé.

Test des canaux : `uv run python -m bot.alerts --test`.
"""

import argparse
import json
import os
import subprocess
import sys
import urllib.request
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from http import HTTPStatus
from typing import Any, Protocol

import structlog
from pydantic import SecretStr

from bot.config import (
    AlertsConfig,
    ConfigError,
    Settings,
    load_settings,
    load_strategy_config,
)
from bot.logs import configure_logging

logger = structlog.get_logger()

# Rappelle sur le téléphone que l'alerte vient du compte paper.
TITLE_PREFIX = "[PAPER] "
# Identifiant d'application de Windows PowerShell, connu de Windows : le toast s'affiche sans
# installer d'application.
POWERSHELL_APP_ID = r"{1AC14E77-02E7-4E5D-B744-2EB1AE5198B7}\WindowsPowerShell\v1.0\powershell.exe"
# Le texte passe par l'environnement : il n'est jamais interprété comme du code PowerShell.
_NOTIFICATIONS = "Windows.UI.Notifications"
TOAST_SCRIPT = "\n".join(
    [
        f"$null = [{_NOTIFICATIONS}.ToastNotificationManager, {_NOTIFICATIONS}, "
        "ContentType = WindowsRuntime]",
        f"$manager = [{_NOTIFICATIONS}.ToastNotificationManager]",
        f"$kind = [{_NOTIFICATIONS}.ToastTemplateType]::ToastText02",
        "$template = $manager::GetTemplateContent($kind)",
        "$texts = $template.GetElementsByTagName('text')",
        "$null = $texts.Item(0).AppendChild($template.CreateTextNode($env:BOT_ALERT_TITLE))",
        "$null = $texts.Item(1).AppendChild($template.CreateTextNode($env:BOT_ALERT_MESSAGE))",
        f"$toast = [{_NOTIFICATIONS}.ToastNotification]::new($template)",
        f"$manager::CreateToastNotifier('{POWERSHELL_APP_ID}').Show($toast)",
    ]
)
# Pas de fenêtre console pour le processus PowerShell du toast.
CREATE_NO_WINDOW = 0x08000000
# Pannes attendues d'un canal : réseau, serveur, PowerShell. Elles sont journalisées.
ALERT_ERRORS = (OSError, subprocess.SubprocessError, ValueError)


class Severity(StrEnum):
    """Gravité d'une alerte."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


# Priorités ntfy, de 1 à 5 : une alerte critique sonne même en mode silencieux.
NTFY_PRIORITY = {Severity.INFO: 3, Severity.WARNING: 4, Severity.CRITICAL: 5}
NTFY_TAGS = {
    Severity.INFO: ["chart_with_upwards_trend"],
    Severity.WARNING: ["warning"],
    Severity.CRITICAL: ["rotating_light"],
}


@dataclass(frozen=True)
class Alert:
    """Message envoyé à chaque canal."""

    title: str
    message: str
    severity: Severity


class Notifier(Protocol):
    """Canal d'alerte."""

    @property
    def name(self) -> str:
        """Nom du canal, pour les logs."""
        ...

    def send(self, alert: Alert) -> None:
        """Envoie l'alerte, ou lève une erreur de `ALERT_ERRORS`."""
        ...


Opener = Callable[..., Any]


class NtfyNotifier:
    """Notification sur le téléphone via un serveur ntfy, publiée en JSON (texte UTF-8)."""

    name = "ntfy"

    def __init__(
        self,
        server: str,
        topic: SecretStr,
        timeout: float,
        opener: Opener = urllib.request.urlopen,
    ) -> None:
        """Associe le canal à son serveur et à son nom secret."""
        self._server = server
        self._topic = topic
        self._timeout = timeout
        self._opener = opener

    def payload(self, alert: Alert) -> dict[str, Any]:
        """Corps de la publication. Contient le nom du canal : à ne jamais journaliser."""
        return {
            "topic": self._topic.get_secret_value(),
            "title": TITLE_PREFIX + alert.title,
            "message": alert.message,
            "priority": NTFY_PRIORITY[alert.severity],
            "tags": NTFY_TAGS[alert.severity],
        }

    def send(self, alert: Alert) -> None:
        """Publie l'alerte. Le canal est dans le corps, jamais dans l'URL."""
        request = urllib.request.Request(
            self._server,
            data=json.dumps(self.payload(alert)).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with self._opener(request, timeout=self._timeout) as response:
            if response.status != HTTPStatus.OK:
                raise OSError(f"ntfy a répondu {response.status}")


class WindowsToastNotifier:
    """Notification Windows, affichée dans le coin de l'écran et le centre de notifications."""

    name = "windows"

    def __init__(self, timeout: float, run: Callable[..., Any] = subprocess.run) -> None:
        """Associe le canal à la commande qui lance PowerShell."""
        self._timeout = timeout
        self._run = run

    def send(self, alert: Alert) -> None:
        """Affiche le toast avec Windows PowerShell, sans fenêtre."""
        environment = {
            **os.environ,
            "BOT_ALERT_TITLE": TITLE_PREFIX + alert.title,
            "BOT_ALERT_MESSAGE": alert.message,
        }
        self._run(
            ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", TOAST_SCRIPT],
            env=environment,
            timeout=self._timeout,
            check=True,
            capture_output=True,
            creationflags=CREATE_NO_WINDOW,
        )


class Alerter:
    """Envoie chaque alerte à tous les canaux. Un canal en panne n'empêche pas les autres."""

    def __init__(self, notifiers: Sequence[Notifier]) -> None:
        """Associe l'alerteur à ses canaux. Sans canal, il ne fait que journaliser."""
        self._notifiers = list(notifiers)

    @property
    def channels(self) -> list[str]:
        """Noms des canaux actifs."""
        return [notifier.name for notifier in self._notifiers]

    def send(self, alert: Alert) -> list[str]:
        """Envoie l'alerte et retourne les canaux en échec."""
        logger.info("alert", title=alert.title, severity=alert.severity.value)
        failed = []
        for notifier in self._notifiers:
            try:
                notifier.send(alert)
            except ALERT_ERRORS as exc:
                failed.append(notifier.name)
                logger.warning("alert_failed", channel=notifier.name, error=type(exc).__name__)
        return failed


def build_alerter(config: AlertsConfig, settings: Settings | None) -> Alerter:
    """Crée les canaux configurés. Sans paramètres chargés, seul le toast Windows est possible."""
    notifiers: list[Notifier] = []
    if config.windows_toast and sys.platform == "win32":
        notifiers.append(WindowsToastNotifier(config.timeout_seconds))
    if settings is not None and settings.ntfy_topic is not None:
        notifiers.append(
            NtfyNotifier(config.ntfy_server, settings.ntfy_topic, config.timeout_seconds)
        )
    return Alerter(notifiers)


def main(argv: Sequence[str] | None = None) -> int:
    """Envoie une alerte de test sur chaque canal configuré."""
    parser = argparse.ArgumentParser(description="Teste les canaux d'alerte.")
    parser.add_argument("--test", action="store_true", help="envoie une alerte de test")
    args = parser.parse_args(argv)
    configure_logging()
    if not args.test:
        parser.print_help()
        return 0
    try:
        alerter = build_alerter(load_strategy_config().alerts, load_settings())
    except ConfigError as exc:
        print(f"Configuration illisible : {exc}")
        return 1
    if not alerter.channels:
        print("Aucun canal configuré : activer windows_toast ou renseigner NTFY_TOPIC dans .env.")
        return 1
    failed = alerter.send(
        Alert("Test des alertes", "Si tu lis ceci, ce canal fonctionne.", Severity.INFO)
    )
    for channel in alerter.channels:
        print(f"{channel} : {'échec' if channel in failed else 'envoyée'}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
