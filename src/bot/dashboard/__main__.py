"""Lance le dashboard : `uv run python -m bot.dashboard`.

Le serveur n'écoute que sur localhost, sans authentification : le dashboard n'est pas accessible
depuis le réseau. Il est indépendant du bot et peut être arrêté (Ctrl+C) à tout moment.
"""

import sys
import threading
import webbrowser
from pathlib import Path

from streamlit.web import cli as streamlit_cli

from bot.config import ConfigError, load_strategy_config

HOST = "localhost"
# Délai laissé au serveur pour démarrer avant d'ouvrir le navigateur.
BROWSER_DELAY_SECONDS = 2.0


def streamlit_args(port: int) -> list[str]:
    """Arguments de `streamlit run` : local uniquement, sans télémétrie."""
    app = Path(__file__).with_name("app.py")
    return [
        "streamlit",
        "run",
        str(app),
        "--server.address",
        HOST,
        "--server.port",
        str(port),
        "--server.headless",
        "true",
        "--browser.gatherUsageStats",
        "false",
    ]


def main() -> int:
    """Démarre le serveur Streamlit et ouvre le dashboard dans le navigateur."""
    try:
        port = load_strategy_config().dashboard.port
    except ConfigError as exc:
        print(f"Configuration illisible : {exc}")
        return 1
    url = f"http://{HOST}:{port}"
    print(f"Dashboard sur {url} (Ctrl+C pour l'arrêter, sans effet sur le bot)")
    threading.Timer(BROWSER_DELAY_SECONDS, webbrowser.open, [url]).start()
    sys.argv = streamlit_args(port)
    streamlit_cli.main()
    return 0


if __name__ == "__main__":
    sys.exit(main())
