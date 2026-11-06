"""Signe de vie du bot, lu sans jamais écrire : par le dashboard et par le chien de garde.

Avant l'ouverture, le bot logue à chaque tick. Une fois le cycle du jour terminé, il ne logue plus,
mais relève le capital dans le journal à chaque tick, marché ouvert. Les deux signes sont donc lus.
"""

import json
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

from bot.config import PROJECT_ROOT, StrategyConfig
from bot.journal.store import EquityPoint, Journal
from bot.status import LOG_PATTERN


@dataclass(frozen=True)
class Heartbeat:
    """Dernier signe de vie du bot."""

    last_seen: datetime | None
    active: bool


@dataclass(frozen=True)
class JournalView:
    """Contenu du journal utile au suivi. Vide si le bot n'a jamais tourné."""

    decisions: list[dict[str, Any]]
    equity: list[EquityPoint]

    @property
    def bot_sessions(self) -> list[date]:
        """Séances où le bot a tourné, marché ouvert."""
        return [point.session for point in self.equity]

    @property
    def last_record(self) -> datetime | None:
        """Dernier relevé du capital, fait à chaque tick marché ouvert."""
        return max((point.recorded_at for point in self.equity), default=None)


def read_journal(config: StrategyConfig) -> JournalView:
    """Lit le journal en lecture seule, sans jamais le créer."""
    path = PROJECT_ROOT / config.live.journal_file
    if not path.exists():
        return JournalView([], [])
    journal = Journal(path, read_only=True)
    try:
        return JournalView(journal.decisions(), journal.equity_history())
    finally:
        journal.close()


def last_log_time(log_dir: Path) -> datetime | None:
    """Retourne l'horodatage de la dernière ligne lisible du fichier de log le plus récent."""
    files = sorted(log_dir.glob(LOG_PATTERN)) if log_dir.is_dir() else []
    if not files:
        return None
    for line in reversed(files[-1].read_text(encoding="utf-8").splitlines()):
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        stamp = entry.get("timestamp") if isinstance(entry, dict) else None
        if isinstance(stamp, str):
            return datetime.fromisoformat(stamp)
    return None


def heartbeat(signals: Iterable[datetime | None], now: datetime, max_age: timedelta) -> Heartbeat:
    """Le bot est actif si son dernier signe de vie date de moins de `max_age`."""
    known = [signal for signal in signals if signal is not None]
    last_seen = max(known) if known else None
    return Heartbeat(last_seen, last_seen is not None and now - last_seen <= max_age)


def bot_heartbeat(config: StrategyConfig, now: datetime) -> Heartbeat:
    """Signe de vie du bot d'après son dernier log et son dernier relevé du capital."""
    max_age = timedelta(
        seconds=config.live.poll_interval_seconds * config.dashboard.heartbeat_intervals
    )
    signals = [last_log_time(PROJECT_ROOT / config.live.log_dir), read_journal(config).last_record]
    return heartbeat(signals, now, max_age)
