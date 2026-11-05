"""Briques du fonctionnement sans surveillance : horloge, journal, logs et mise en veille."""

import json
import sqlite3
import sys
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pytest
import structlog

from bot import power
from bot.execution.broker import MarketClock
from bot.journal.store import EquityPoint, Journal
from bot.logs import configure_logging

# 16h05 à New York, mais déjà le lendemain en UTC.
EVENING = datetime(2026, 11, 4, 1, 5, tzinfo=UTC)
MORNING = datetime(2026, 11, 3, 13, 0, tzinfo=UTC)
OPEN_TODAY = datetime(2026, 11, 3, 14, 30, tzinfo=UTC)
OPEN_TOMORROW = datetime(2026, 11, 4, 14, 30, tzinfo=UTC)


@pytest.mark.parametrize(
    ("is_open", "timestamp", "next_open", "over"),
    [
        (False, MORNING, OPEN_TODAY, False),
        (True, OPEN_TODAY, OPEN_TOMORROW, False),
        (False, datetime(2026, 11, 3, 21, 5, tzinfo=UTC), OPEN_TOMORROW, True),
        # Le jour de la bourse est celui de New York, pas celui d'UTC.
        (False, EVENING, OPEN_TOMORROW, True),
        (False, MORNING, OPEN_TODAY + timedelta(days=4), True),
    ],
)
def test_session_over_follows_the_exchange_day(
    is_open: bool, timestamp: datetime, next_open: datetime, over: bool
) -> None:
    clock = MarketClock(is_open, timestamp, next_open, next_open + timedelta(hours=7))
    assert clock.is_session_over is over


def test_equity_point_exposure() -> None:
    point = EquityPoint(date(2026, 11, 3), 100_000.0, 25_000.0, MORNING)
    assert point.exposure == pytest.approx(0.75)


def test_journal_last_cycle_and_equity_history(tmp_path: Path) -> None:
    journal = Journal(tmp_path / "journal.db")
    assert journal.last_completed_cycle() is None
    journal.complete_cycle(date(2026, 11, 2), MORNING)
    journal.complete_cycle(date(2026, 11, 3), MORNING)
    later = EquityPoint(date(2026, 11, 3), 101.0, 1.0, MORNING)
    earlier = EquityPoint(date(2026, 11, 2), 100.0, 1.0, MORNING)
    journal.record_equity(later)
    journal.record_equity(earlier)
    assert journal.last_completed_cycle() == date(2026, 11, 3)
    assert journal.equity_history() == [earlier, later]
    journal.close()


def test_read_only_journal_cannot_write_nor_create(tmp_path: Path) -> None:
    path = tmp_path / "journal.db"
    with pytest.raises(sqlite3.OperationalError):
        Journal(path, read_only=True)
    assert not path.exists()

    Journal(path).close()
    journal = Journal(path, read_only=True)
    with pytest.raises(sqlite3.OperationalError, match="readonly"):
        journal.complete_cycle(date(2026, 11, 3), MORNING)
    journal.close()


def test_logs_go_to_console_and_file(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = tmp_path / "logs" / "bot.jsonl"
    try:
        configure_logging(path)
        structlog.get_logger().error("boom", reason="test")
    finally:
        configure_logging()
    entry = json.loads(path.read_text(encoding="utf-8"))
    assert (entry["event"], entry["level"], entry["reason"]) == ("boom", "error", "test")
    assert "boom" in capsys.readouterr().out


def test_reconfiguring_logs_closes_the_previous_file(tmp_path: Path) -> None:
    first, second = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    try:
        configure_logging(first)
        configure_logging(second)
        structlog.get_logger().info("seconde")
    finally:
        configure_logging()
    assert first.read_text(encoding="utf-8") == ""
    assert "seconde" in second.read_text(encoding="utf-8")


def test_keep_awake_does_nothing_when_disabled() -> None:
    with power.keep_awake(enabled=False) as active:
        assert active is False


def test_keep_awake_does_nothing_outside_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    with power.keep_awake() as active:
        assert active is False


@pytest.mark.skipif(sys.platform != "win32", reason="API Windows")
def test_keep_awake_requests_then_releases_on_windows() -> None:
    import ctypes

    kernel32 = ctypes.windll.kernel32
    with power.keep_awake() as active:
        assert active is True
        # L'appel retourne l'état précédent : la demande du bloc est bien en place.
        current = kernel32.SetThreadExecutionState(power.ES_CONTINUOUS | power.ES_SYSTEM_REQUIRED)
        assert current & power.ES_SYSTEM_REQUIRED
    released = kernel32.SetThreadExecutionState(power.ES_CONTINUOUS)
    assert not released & power.ES_SYSTEM_REQUIRED
