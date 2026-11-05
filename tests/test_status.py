import json
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pytest

from bot import status as status_module
from bot.data.market import Session
from bot.journal.store import Decision, EquityPoint, Journal
from bot.risk.models import HaltReason, OrderSide, Position
from bot.risk.state import JsonRiskStateStore, RiskState
from bot.status import (
    EXIT_ERROR,
    EXIT_HALTED,
    EXIT_OK,
    BotStatus,
    LogSummary,
    OrderLine,
    collect_status,
    format_status,
    last_orders,
    last_target_weights,
    missed_sessions,
    position_lines,
    summarize_log,
)
from tests.test_main import NEW_YORK, FakeClient, make_config, make_position

NOW = datetime(2026, 11, 5, 21, 30, tzinfo=UTC)
CYCLE = date(2026, 11, 3)


def decision(symbol: str, weight: float, **extra: Any) -> Decision:
    return Decision(
        timestamp=NOW,
        session=CYCLE,
        symbol=symbol,
        signal=1,
        note="order",
        inputs={"target_weight": weight},
        **extra,
    )


def test_target_weights_come_from_the_requested_cycle(tmp_path: Path) -> None:
    journal = Journal(tmp_path / "journal.db")
    journal.record_decision(decision("A", 0.2))
    journal.record_decision(decision("B", 0.0))
    other = Decision(NOW, date(2026, 10, 1), "A", 1, "order", {"target_weight": 0.9})
    journal.record_decision(other)
    kill = Decision(NOW, CYCLE, "C", None, "kill_switch", {"equity": 1.0})
    journal.record_decision(kill)
    assert last_target_weights(journal.decisions(), CYCLE) == {"A": 0.2, "B": 0.0}
    journal.close()


def test_last_orders_list_accepted_and_rejected_orders(tmp_path: Path) -> None:
    journal = Journal(tmp_path / "journal.db")
    journal.record_decision(decision("A", 0.2))
    journal.record_decision(
        decision("B", 0.2, side=OrderSide.BUY, quantity=3.0, risk_approved=True)
    )
    journal.record_decision(
        decision(
            "C", 0.2, side=OrderSide.BUY, quantity=1.0, risk_approved=False, risk_reason="stale"
        )
    )
    orders = last_orders(journal.decisions(), CYCLE)
    assert [(order.symbol, order.risk_approved) for order in orders] == [
        ("B", True),
        ("C", False),
    ]
    assert orders[1].risk_reason == "stale"
    journal.close()


def test_missed_sessions_are_trading_days_without_equity() -> None:
    days = [date(2026, 11, 2), date(2026, 11, 3), date(2026, 11, 4)]
    assert missed_sessions(days, [date(2026, 11, 2), date(2026, 11, 4)]) == [date(2026, 11, 3)]


def test_position_lines_flag_foreign_and_missing_positions() -> None:
    positions = {"A": Position(10.0, 20_000.0), "ZZZ": Position(1.0, 500.0)}
    lines = position_lines(positions, 100_000.0, {"A": 0.25, "B": 0.1, "C": 0.0}, ["A", "B", "C"])
    assert [(line.symbol, line.in_universe) for line in lines] == [
        ("A", True),
        ("B", True),
        ("ZZZ", False),
    ]
    assert lines[0].weight == pytest.approx(0.2)
    assert lines[1].quantity == 0.0
    assert lines[2].target_weight is None


def test_summarize_log_reads_the_latest_file(tmp_path: Path) -> None:
    assert summarize_log(tmp_path / "absent").path is None
    (tmp_path / "bot-2026-11-04.jsonl").write_text('{"level": "error"}\n', encoding="utf-8")
    entries = [
        {"level": "info", "event": "ok"},
        {"level": "warning", "event": "order_rejected_by_risk"},
        {"level": "error", "event": "tick_failed", "error": "APIError", "timestamp": "t"},
    ]
    text = "\n".join(json.dumps(entry) for entry in entries) + "\npas du json\n\n"
    (tmp_path / "bot-2026-11-05.jsonl").write_text(text, encoding="utf-8")

    summary = summarize_log(tmp_path)

    assert summary.path == tmp_path / "bot-2026-11-05.jsonl"
    assert (summary.errors, summary.warnings, summary.unreadable) == (1, 1, 1)
    assert summary.last_error == "t tick_failed APIError"


def sample_status(**overrides: Any) -> BotStatus:
    values: dict[str, Any] = {
        "generated_at": NOW,
        "equity": 95_000.0,
        "last_equity": 100_000.0,
        "cash": 19_000.0,
        "risk": RiskState(peak_equity=100_000.0),
        "limits": make_config().risk,
        "positions": position_lines(
            {"A": Position(10.0, 76_000.0), "ZZZ": Position(1.0, 0.0)}, 95_000.0, {"A": 0.2}, ["A"]
        ),
        "journal_found": True,
        "last_cycle": CYCLE,
        "recorded_sessions": 3,
        "first_recorded": date(2026, 11, 2),
        "missed_sessions": [date(2026, 11, 4)],
        "last_orders": [OrderLine("A", "buy", 2.0, False, "stale_price", None, None)],
        "log": LogSummary(Path("bot-2026-11-05.jsonl"), 1, 2, 0, "t tick_failed APIError"),
    }
    values.update(overrides)
    return BotStatus(**values)


def test_status_measures_exposure_day_change_and_drawdown() -> None:
    status = sample_status()
    assert status.exposure == pytest.approx(0.8)
    assert status.day_change == pytest.approx(-0.05)
    assert status.drawdown == pytest.approx(0.05)


def test_format_status_shows_what_needs_attention() -> None:
    text = format_status(sample_status())
    for expected in (
        "exposition 80.00%",
        "drawdown 5.00%",
        "Kill switch armé",
        "HORS UNIVERS",
        "séances manquées (bot inactif) : 2026-11-04",
        "refusé (stale_price)",
        "1 erreurs, 2 avertissements",
        "dernière erreur : t tick_failed APIError",
    ):
        assert expected in text


def test_format_status_explains_a_triggered_kill_switch() -> None:
    risk = RiskState(peak_equity=100_000.0, halt_reason=HaltReason.DAILY_LOSS, halted_at=NOW)
    text = format_status(
        sample_status(risk=risk, journal_found=False, positions=[], log=LogSummary(None))
    )
    assert "KILL SWITCH DÉCLENCHÉ : daily_loss" in text
    assert "--reset-kill-switch" in text
    assert "Journal absent" in text
    assert "aucune" in text
    assert "Logs : aucun fichier" in text


def wire_status(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, client: FakeClient) -> None:
    monkeypatch.setattr(status_module, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(status_module, "load_strategy_config", make_config)
    monkeypatch.setattr(status_module, "load_settings", lambda: None)
    monkeypatch.setattr(status_module, "create_trading_client", lambda _: client)

    def sessions(client: object, start: date, end: date, retry: object) -> list[Session]:
        days = [date(2026, 11, 2), date(2026, 11, 3), date(2026, 11, 4)]
        close = datetime(2026, 1, 1, 16).time()
        return [
            Session(day, datetime.combine(day, close, tzinfo=NEW_YORK))
            for day in days
            if start <= day <= end
        ]

    monkeypatch.setattr(status_module, "fetch_sessions", sessions)


def test_collect_status_reads_without_writing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    client = FakeClient()
    client.positions = [make_position("A", 50.0)]
    wire_status(monkeypatch, tmp_path, client)
    config = make_config()
    journal = Journal(tmp_path / config.live.journal_file)
    journal.record_decision(decision("A", 0.2))
    journal.complete_cycle(CYCLE, NOW)
    for day in (date(2026, 11, 2), date(2026, 11, 4)):
        journal.record_equity(EquityPoint(day, 100_000.0, 90_000.0, NOW))
    journal.close()
    before = (tmp_path / config.live.journal_file).read_bytes()

    status = collect_status(NOW)

    assert status.journal_found
    assert status.last_cycle == CYCLE
    assert status.missed_sessions == [CYCLE]
    assert [(line.symbol, line.target_weight) for line in status.positions] == [("A", 0.2)]
    assert (tmp_path / config.live.journal_file).read_bytes() == before
    assert not (tmp_path / config.risk_state_file).exists()
    assert client.submitted == []


def test_collect_status_without_journal(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    client = FakeClient()
    wire_status(monkeypatch, tmp_path, client)
    status = collect_status(NOW)
    assert not status.journal_found
    assert status.positions == []
    assert not (tmp_path / make_config().live.journal_file).exists()


def test_main_returns_halted_code(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    wire_status(monkeypatch, tmp_path, FakeClient())
    store = JsonRiskStateStore(tmp_path / make_config().risk_state_file)
    store.save(RiskState(peak_equity=1.0, halt_reason=HaltReason.MAX_DRAWDOWN, halted_at=NOW))
    assert status_module.main() == EXIT_HALTED
    assert "KILL SWITCH" in capsys.readouterr().out


def test_main_prints_status(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    wire_status(monkeypatch, tmp_path, FakeClient())
    assert status_module.main() == EXIT_OK
    assert "Kill switch armé" in capsys.readouterr().out


def test_main_reports_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    wire_status(monkeypatch, tmp_path, FakeClient())
    state = tmp_path / make_config().risk_state_file
    state.parent.mkdir(parents=True)
    state.write_text("corrompu", encoding="utf-8")
    assert status_module.main() == EXIT_ERROR
    assert "status_failed" in capsys.readouterr().out
