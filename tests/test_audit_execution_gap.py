from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from audit.execution_gap import measure_execution_gap, read_fill_slippages
from bot.execution.broker import OrderReport
from bot.journal.store import Decision, Journal
from bot.risk.models import OrderSide

NOW = datetime(2026, 11, 4, 15, 0, tzinfo=UTC)
SESSION = date(2026, 11, 4)


def record(
    journal: Journal, symbol: str, reference: float | None, filled: float | None, quantity: float
) -> None:
    decision_id = journal.record_decision(
        Decision(
            timestamp=NOW,
            session=SESSION,
            symbol=symbol,
            signal=1,
            note="order",
            inputs={},
            side=OrderSide.BUY,
            quantity=quantity,
            risk_approved=True,
            client_order_id=f"id-{symbol}",
            reference_price=reference,
        )
    )
    journal.record_report(
        decision_id,
        OrderReport(
            client_order_id=f"id-{symbol}",
            broker_order_id="b",
            status="filled" if filled else "rejected",
            is_final=True,
            filled_quantity=quantity if filled else 0.0,
            filled_price=filled,
        ),
    )


def test_missing_journal_is_reported_without_creating_it(tmp_path: Path) -> None:
    path = tmp_path / "journal.db"
    report = measure_execution_gap(path, 0.001)
    assert not report.journal_found
    assert report.fills == 0
    assert report.mean_slippage_pct is None
    assert not path.exists()


def test_only_filled_orders_with_a_reference_price_are_measured(tmp_path: Path) -> None:
    path = tmp_path / "journal.db"
    journal = Journal(path)
    record(journal, "A", 100.0, 100.5, 2.0)
    record(journal, "B", 100.0, 99.9, 1.0)
    record(journal, "C", 100.0, None, 1.0)
    record(journal, "D", None, 50.0, 1.0)
    journal.close()

    assert read_fill_slippages(path) == pytest.approx([0.005, -0.001])
    report = measure_execution_gap(path, 0.001)
    assert report.journal_found
    assert report.fills == 2
    assert report.mean_slippage_pct == pytest.approx(0.002)
    assert report.assumed_slippage_pct == 0.001
