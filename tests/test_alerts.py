import json
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.error import URLError

import pytest
from pydantic import SecretStr

from bot import alerts as alerts_module
from bot import heartbeat as heartbeat_module
from bot import watchdog
from bot.alerts import (
    Alert,
    Alerter,
    NtfyNotifier,
    Severity,
    WindowsToastNotifier,
    build_alerter,
)
from bot.config import Settings, load_settings
from bot.execution.broker import MarketClock
from bot.heartbeat import Heartbeat
from bot.journal.store import EquityPoint, Journal
from tests.test_main import FakeClient, make_config

ALERT = Alert("Kill switch déclenché", "Motif : daily_loss. Été à 98 000 $", Severity.CRITICAL)
NOW = datetime(2026, 11, 9, 18, 0, tzinfo=UTC)
SECRET_TOPIC = "canal-secret-123"


class FakeResponse:
    def __init__(self, status: int) -> None:
        self.status = status

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *_: object) -> None:
        return None


class Recorder:
    """Canal qui garde les alertes reçues, ou qui échoue."""

    def __init__(self, name: str = "test", error: Exception | None = None) -> None:
        self.name = name
        self.error = error
        self.alerts: list[Alert] = []

    def send(self, alert: Alert) -> None:
        if self.error is not None:
            raise self.error
        self.alerts.append(alert)


def test_ntfy_publishes_utf8_json_with_the_topic_in_the_body() -> None:
    requests: list[Any] = []

    def opener(request: Any, timeout: float) -> FakeResponse:
        requests.append((request, timeout))
        return FakeResponse(200)

    NtfyNotifier("https://ntfy.sh", SecretStr(SECRET_TOPIC), 5.0, opener).send(ALERT)

    request, timeout = requests[0]
    body = json.loads(request.data.decode("utf-8"))
    assert request.full_url == "https://ntfy.sh"
    assert SECRET_TOPIC not in request.full_url
    assert body["topic"] == SECRET_TOPIC
    assert body["title"] == "[PAPER] Kill switch déclenché"
    assert body["message"] == ALERT.message
    assert (body["priority"], body["tags"]) == (5, ["rotating_light"])
    assert timeout == 5.0


def test_ntfy_rejects_an_error_status() -> None:
    notifier = NtfyNotifier(
        "https://ntfy.sh", SecretStr("x"), 5.0, lambda *_, **__: FakeResponse(429)
    )
    with pytest.raises(OSError, match="429"):
        notifier.send(ALERT)


def test_windows_toast_passes_text_through_the_environment() -> None:
    calls: list[tuple[list[str], dict[str, Any]]] = []

    def run(command: list[str], **options: Any) -> None:
        calls.append((command, options))

    WindowsToastNotifier(3.0, run).send(ALERT)

    command, options = calls[0]
    assert command[0] == "powershell.exe"
    # Le texte n'apparaît pas dans le script : il ne peut pas être interprété comme du code.
    assert ALERT.message not in command[-1]
    assert options["env"]["BOT_ALERT_TITLE"] == "[PAPER] Kill switch déclenché"
    assert options["env"]["BOT_ALERT_MESSAGE"] == ALERT.message
    assert (options["timeout"], options["check"]) == (3.0, True)


def test_alerter_survives_a_failing_channel(capsys: pytest.CaptureFixture[str]) -> None:
    broken = Recorder("cassé", URLError("réseau"))
    working = Recorder("ok")
    timeout = Recorder("lent", subprocess.TimeoutExpired("powershell", 3))

    failed = Alerter([broken, working, timeout]).send(ALERT)

    assert failed == ["cassé", "lent"]
    assert working.alerts == [ALERT]
    output = capsys.readouterr().out
    assert "alert_failed" in output
    assert SECRET_TOPIC not in output


def settings(topic: str | None) -> Settings:
    return Settings(
        alpaca_api_key=SecretStr("k"),
        alpaca_secret_key=SecretStr("s"),
        ntfy_topic=SecretStr(topic) if topic else None,
    )


@pytest.mark.parametrize(
    ("toast", "platform", "topic", "channels"),
    [
        (True, "win32", SECRET_TOPIC, ["windows", "ntfy"]),
        (True, "linux", SECRET_TOPIC, ["ntfy"]),
        (False, "win32", None, []),
        (True, "win32", None, ["windows"]),
    ],
)
def test_build_alerter_picks_configured_channels(
    monkeypatch: pytest.MonkeyPatch,
    toast: bool,
    platform: str,
    topic: str | None,
    channels: list[str],
) -> None:
    monkeypatch.setattr(sys, "platform", platform)
    config = make_config().alerts.model_copy(update={"windows_toast": toast})
    assert build_alerter(config, settings(topic)).channels == channels
    assert "ntfy" not in build_alerter(config, None).channels


def test_settings_ignore_an_empty_topic(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    env.write_text("ALPACA_API_KEY=k\nALPACA_SECRET_KEY=s\nNTFY_TOPIC=\n", encoding="utf-8")
    assert load_settings(env).ntfy_topic is None


def test_cli_test_reports_each_channel(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    recorder = Recorder("ok")
    monkeypatch.setattr(alerts_module, "load_settings", lambda: settings(None))
    monkeypatch.setattr(alerts_module, "build_alerter", lambda *_: Alerter([recorder]))
    assert alerts_module.main(["--test"]) == 0
    assert recorder.alerts[0].severity is Severity.INFO
    assert "ok : envoyée" in capsys.readouterr().out


def test_cli_without_channel_fails(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(alerts_module, "load_settings", lambda: settings(None))
    monkeypatch.setattr(alerts_module, "build_alerter", lambda *_: Alerter([]))
    assert alerts_module.main(["--test"]) == 1
    assert alerts_module.main([]) == 0
    assert "Aucun canal" in capsys.readouterr().out


# Chien de garde

OPEN = MarketClock(True, NOW, NOW, NOW)
CLOSED = MarketClock(False, NOW, NOW + timedelta(days=1), NOW)


@pytest.mark.parametrize(
    ("clock", "beat", "alerted"),
    [
        (CLOSED, Heartbeat(None, False), False),
        (OPEN, Heartbeat(NOW, True), False),
        (OPEN, Heartbeat(NOW - timedelta(hours=1), False), True),
        (OPEN, Heartbeat(None, False), True),
    ],
)
def test_watchdog_alerts_only_when_market_is_open_and_bot_silent(
    clock: MarketClock, beat: Heartbeat, alerted: bool
) -> None:
    alert = watchdog.check(clock, beat)
    assert (alert is not None) is alerted
    if alert is not None:
        assert alert.severity is Severity.CRITICAL


def test_watch_reads_the_journal_heartbeat(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(heartbeat_module, "PROJECT_ROOT", tmp_path)
    config = make_config()
    client = FakeClient()
    recorder = Recorder()
    clock_time = client.clock_time

    assert (
        watchdog.watch(config, client, Alerter([recorder]), clock_time)
        == watchdog.EXIT_BOT_INACTIVE
    )
    assert len(recorder.alerts) == 1

    journal = Journal(tmp_path / config.live.journal_file)
    journal.record_equity(
        EquityPoint(clock_time.date(), 1.0, 0.0, clock_time - timedelta(minutes=1))
    )
    journal.close()
    assert watchdog.watch(config, client, Alerter([recorder]), clock_time) == watchdog.EXIT_OK
    assert len(recorder.alerts) == 1
