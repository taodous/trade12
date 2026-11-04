"""Journal SQLite de chaque décision, pour reconstituer pourquoi chaque trade a eu lieu.

Une décision est écrite avant l'envoi de l'ordre : si l'écriture échoue, l'exception remonte et
l'ordre ne part pas. Si le bot s'arrête juste après l'envoi, l'identifiant client déjà enregistré
permet de retrouver l'ordre chez le broker.
"""

import json
import sqlite3
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

from bot.execution.broker import OrderReport
from bot.risk.models import OrderSide

_SCHEMA = """
CREATE TABLE IF NOT EXISTS decisions (
    id INTEGER PRIMARY KEY,
    timestamp TEXT NOT NULL,
    session TEXT NOT NULL,
    symbol TEXT NOT NULL,
    signal INTEGER,
    note TEXT NOT NULL,
    inputs TEXT NOT NULL,
    side TEXT,
    quantity REAL,
    risk_approved INTEGER,
    risk_reason TEXT,
    risk_detail TEXT,
    client_order_id TEXT,
    broker_order_id TEXT,
    status TEXT,
    is_final INTEGER NOT NULL DEFAULT 1,
    reference_price REAL,
    filled_quantity REAL,
    filled_price REAL,
    slippage_pct REAL,
    detail TEXT
);
CREATE TABLE IF NOT EXISTS cycles (
    session TEXT PRIMARY KEY,
    completed_at TEXT NOT NULL
);
"""
STATUS_NOT_FOUND = "not_found"


@dataclass(frozen=True)
class Decision:
    """Décision à enregistrer pour un symbole."""

    timestamp: datetime
    session: date
    symbol: str
    signal: int | None
    note: str
    inputs: dict[str, Any]
    side: OrderSide | None = None
    quantity: float | None = None
    risk_approved: bool | None = None
    risk_reason: str | None = None
    risk_detail: str | None = None
    client_order_id: str | None = None
    reference_price: float | None = None


@dataclass(frozen=True)
class PendingOrder:
    """Ordre envoyé dont l'issue n'est pas encore connue."""

    decision_id: int
    client_order_id: str


def observed_slippage(side: OrderSide, reference_price: float, filled_price: float) -> float:
    """Retourne le coût d'exécution par rapport au prix de référence, positif si défavorable."""
    difference = filled_price - reference_price
    if side is OrderSide.SELL:
        difference = -difference
    return difference / reference_price


class Journal:
    """Enregistre les décisions et le sort des ordres."""

    def __init__(self, path: Path) -> None:
        """Ouvre le journal et crée ses tables si besoin."""
        path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(path)
        self._connection.row_factory = sqlite3.Row
        self._connection.executescript(_SCHEMA)

    def close(self) -> None:
        """Ferme le journal."""
        self._connection.close()

    def record_decision(self, decision: Decision) -> int:
        """Enregistre une décision et retourne son identifiant."""
        is_awaiting_order = decision.risk_approved is True
        with self._connection:
            cursor = self._connection.execute(
                """
                INSERT INTO decisions (
                    timestamp, session, symbol, signal, note, inputs, side, quantity,
                    risk_approved, risk_reason, risk_detail, client_order_id, is_final,
                    reference_price
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    decision.timestamp.isoformat(),
                    decision.session.isoformat(),
                    decision.symbol,
                    decision.signal,
                    decision.note,
                    json.dumps(decision.inputs, sort_keys=True, default=str),
                    decision.side.value if decision.side else None,
                    decision.quantity,
                    None if decision.risk_approved is None else int(decision.risk_approved),
                    decision.risk_reason,
                    decision.risk_detail,
                    decision.client_order_id,
                    0 if is_awaiting_order else 1,
                    decision.reference_price,
                ),
            )
        if cursor.lastrowid is None:
            raise sqlite3.DatabaseError("Décision non enregistrée")
        return cursor.lastrowid

    def record_report(self, decision_id: int, report: OrderReport) -> None:
        """Enregistre l'état d'un ordre et, s'il est exécuté, le slippage observé."""
        row = self._connection.execute(
            "SELECT side, reference_price FROM decisions WHERE id = ?", (decision_id,)
        ).fetchone()
        if row is None:
            raise KeyError(decision_id)
        slippage = None
        if report.filled_price is not None and row["reference_price"] and row["side"]:
            slippage = observed_slippage(
                OrderSide(row["side"]), row["reference_price"], report.filled_price
            )
        with self._connection:
            self._connection.execute(
                """
                UPDATE decisions SET broker_order_id = ?, status = ?, is_final = ?,
                    filled_quantity = ?, filled_price = ?, slippage_pct = ?, detail = ?
                WHERE id = ?
                """,
                (
                    report.broker_order_id,
                    report.status,
                    int(report.is_final),
                    report.filled_quantity,
                    report.filled_price,
                    slippage,
                    report.detail,
                    decision_id,
                ),
            )

    def record_status(self, decision_id: int, status: str, *, is_final: bool, detail: str) -> None:
        """Enregistre un état sans réponse du broker : simulation, sort inconnu ou ordre perdu."""
        with self._connection:
            self._connection.execute(
                "UPDATE decisions SET status = ?, is_final = ?, detail = ? WHERE id = ?",
                (status, int(is_final), detail, decision_id),
            )

    def pending_orders(self) -> list[PendingOrder]:
        """Retourne les ordres dont l'issue reste à connaître."""
        rows = self._connection.execute(
            """
            SELECT id, client_order_id FROM decisions
            WHERE is_final = 0 AND client_order_id IS NOT NULL ORDER BY id
            """
        ).fetchall()
        return [PendingOrder(row["id"], row["client_order_id"]) for row in rows]

    def is_cycle_completed(self, session: date) -> bool:
        """Indique si les décisions de cette séance ont déjà été prises."""
        row = self._connection.execute(
            "SELECT 1 FROM cycles WHERE session = ?", (session.isoformat(),)
        ).fetchone()
        return row is not None

    def complete_cycle(self, session: date, completed_at: datetime) -> None:
        """Marque les décisions de cette séance comme prises."""
        with self._connection:
            self._connection.execute(
                "INSERT OR REPLACE INTO cycles (session, completed_at) VALUES (?, ?)",
                (session.isoformat(), completed_at.isoformat()),
            )

    def decisions(self) -> list[dict[str, Any]]:
        """Retourne toutes les décisions, de la plus ancienne à la plus récente."""
        rows = self._connection.execute("SELECT * FROM decisions ORDER BY id").fetchall()
        return [dict(row) for row in rows]
