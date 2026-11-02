"""Contrôle qualité des barres journalières, pur et sans effet de bord.

Aucune valeur n'est réparée ni interpolée. Une barre structurellement fausse est retirée. Une forte
variation reste en place et est seulement signalée : un krach réel est une donnée qu'il ne faut pas
effacer du backtest.
"""

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date
from enum import StrEnum

import pandas as pd
from pydantic import BaseModel, ConfigDict


class IssueKind(StrEnum):
    """Nature d'une anomalie de données."""

    DUPLICATE = "duplicate"
    UNEXPECTED_SESSION = "unexpected_session"
    INVALID_BAR = "invalid_bar"
    MISSING_SESSION = "missing_session"
    ABNORMAL_RETURN = "abnormal_return"


class DataIssue(BaseModel):
    """Anomalie détectée sur une séance."""

    model_config = ConfigDict(frozen=True)

    session: date
    kind: IssueKind
    removed: bool
    detail: str = ""


@dataclass(frozen=True)
class CleanedBars:
    """Barres conservées et anomalies détectées."""

    bars: pd.DataFrame
    issues: list[DataIssue]


def _invalid_bars(bars: pd.DataFrame) -> pd.Series:
    """Retourne le masque des barres aux prix ou au volume incohérents."""
    prices = bars[["open", "high", "low", "close"]]
    missing_or_negative = prices.isna().any(axis=1) | (prices <= 0).any(axis=1)
    bad_range = (prices.max(axis=1) > bars["high"]) | (prices.min(axis=1) < bars["low"])
    bad_volume = bars["volume"].isna() | (bars["volume"] <= 0)
    return missing_or_negative | bad_range | bad_volume


def _issues(sessions: Iterable[pd.Timestamp], kind: IssueKind, *, removed: bool) -> list[DataIssue]:
    return [DataIssue(session=day.date(), kind=kind, removed=removed) for day in sessions]


def clean_bars(
    bars: pd.DataFrame, expected_sessions: Iterable[date], max_price_change_pct: float
) -> CleanedBars:
    """Retire les barres fausses et signale les anomalies.

    `bars` est indexé par date de séance. Le contrôle des variations ne compare une clôture qu'à
    la clôture précédente et n'utilise donc aucune donnée future.
    """
    expected = pd.DatetimeIndex(pd.to_datetime(list(expected_sessions)))

    duplicate = pd.Series(bars.index.duplicated(keep="first"), index=bars.index)
    unexpected = pd.Series(~bars.index.isin(expected), index=bars.index)
    invalid = _invalid_bars(bars) & ~duplicate & ~unexpected

    issues = [
        *_issues(bars.index[duplicate], IssueKind.DUPLICATE, removed=True),
        *_issues(bars.index[unexpected & ~duplicate], IssueKind.UNEXPECTED_SESSION, removed=True),
        *_issues(bars.index[invalid], IssueKind.INVALID_BAR, removed=True),
    ]

    kept = bars[~(duplicate | unexpected | invalid)].sort_index()
    issues += _issues(
        expected.difference(pd.DatetimeIndex(kept.index)), IssueKind.MISSING_SESSION, removed=False
    )

    changes = kept["close"].pct_change(fill_method=None)
    for day, change in changes[changes.abs() > max_price_change_pct].items():
        session = pd.Timestamp(str(day)).date()
        issues.append(
            DataIssue(
                session=session,
                kind=IssueKind.ABNORMAL_RETURN,
                removed=False,
                detail=f"variation de clôture de {change:.4f}",
            )
        )

    return CleanedBars(bars=kept, issues=sorted(issues, key=lambda issue: issue.session))
