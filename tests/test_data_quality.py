from datetime import date

import numpy as np
import pandas as pd
import pytest

from bot.data.market import BAR_COLUMNS
from bot.data.quality import DataIssue, IssueKind, clean_bars

MAX_CHANGE = 0.2


def make_bars(days: list[str], closes: list[float] | None = None) -> pd.DataFrame:
    closes = closes or [100.0 + i for i in range(len(days))]
    return pd.DataFrame(
        {
            "open": closes,
            "high": [close + 1 for close in closes],
            "low": [close - 1 for close in closes],
            "close": closes,
            "volume": 1000.0,
            "trade_count": 10.0,
            "vwap": closes,
        },
        index=pd.DatetimeIndex(pd.to_datetime(days), name="session"),
    )[BAR_COLUMNS]


DAYS = ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]
EXPECTED = [date.fromisoformat(day) for day in DAYS]


def kinds(issues: list[DataIssue]) -> list[IssueKind]:
    return [issue.kind for issue in issues]


def test_clean_data_is_unchanged() -> None:
    bars = make_bars(DAYS)
    result = clean_bars(bars, EXPECTED, MAX_CHANGE)
    assert result.issues == []
    pd.testing.assert_frame_equal(result.bars, bars)


def test_does_not_modify_input() -> None:
    bars = make_bars(DAYS)
    bars.loc[pd.Timestamp("2024-01-03"), "close"] = np.nan
    before = bars.copy()
    clean_bars(bars, EXPECTED, MAX_CHANGE)
    pd.testing.assert_frame_equal(bars, before)


def test_missing_session_is_flagged_and_not_filled() -> None:
    bars = make_bars([DAYS[0], DAYS[1], DAYS[3]])
    result = clean_bars(bars, EXPECTED, MAX_CHANGE)
    assert [(i.session, i.kind, i.removed) for i in result.issues] == [
        (date(2024, 1, 4), IssueKind.MISSING_SESSION, False)
    ]
    assert len(result.bars) == 3


def test_last_session_absent_is_flagged_as_stale() -> None:
    result = clean_bars(make_bars(DAYS[:-1]), EXPECTED, MAX_CHANGE)
    assert result.issues[-1].session == date(2024, 1, 5)
    assert result.issues[-1].kind is IssueKind.MISSING_SESSION


def test_duplicate_keeps_first_value() -> None:
    bars = make_bars([DAYS[0], DAYS[0], DAYS[1], DAYS[2], DAYS[3]], [100, 999, 101, 102, 103])
    result = clean_bars(bars, EXPECTED, MAX_CHANGE)
    assert kinds(result.issues) == [IssueKind.DUPLICATE]
    assert result.bars.loc[pd.Timestamp(DAYS[0]), "close"] == 100
    assert result.bars.index.is_unique


def test_session_outside_calendar_is_removed() -> None:
    bars = make_bars([*DAYS, "2024-01-06"])
    result = clean_bars(bars, EXPECTED, MAX_CHANGE)
    assert kinds(result.issues) == [IssueKind.UNEXPECTED_SESSION]
    assert pd.Timestamp("2024-01-06") not in result.bars.index


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("close", np.nan),
        ("open", np.nan),
        ("close", 0.0),
        ("low", -1.0),
        ("high", 1.0),
        ("low", 500.0),
        ("close", 500.0),
        ("open", 1.0),
        ("volume", 0.0),
        ("volume", np.nan),
    ],
)
def test_invalid_bar_is_removed_then_reported_missing(column: str, value: float) -> None:
    bars = make_bars(DAYS)
    bars.loc[pd.Timestamp(DAYS[1]), column] = value
    result = clean_bars(bars, EXPECTED, MAX_CHANGE)
    assert pd.Timestamp(DAYS[1]) not in result.bars.index
    assert set(kinds(result.issues)) >= {IssueKind.INVALID_BAR, IssueKind.MISSING_SESSION}


def test_large_move_is_flagged_but_kept() -> None:
    bars = make_bars(DAYS, [100.0, 70.0, 71.0, 72.0])
    result = clean_bars(bars, EXPECTED, MAX_CHANGE)
    assert [(i.session, i.kind, i.removed) for i in result.issues] == [
        (date(2024, 1, 3), IssueKind.ABNORMAL_RETURN, False)
    ]
    assert result.bars.loc[pd.Timestamp(DAYS[1]), "close"] == 70.0


def test_move_at_threshold_is_not_flagged() -> None:
    bars = make_bars(DAYS, [100.0, 110.0, 111.0, 112.0])
    assert clean_bars(bars, EXPECTED, 0.2).issues == []


def test_quality_check_has_no_look_ahead() -> None:
    """Ajouter une séance future ne change aucune anomalie des séances passées."""
    past = make_bars(DAYS[:3], [100.0, 70.0, 71.0])
    full = make_bars(DAYS, [100.0, 70.0, 71.0, 200.0])
    past_issues = clean_bars(past, EXPECTED[:3], MAX_CHANGE).issues
    full_issues = clean_bars(full, EXPECTED, MAX_CHANGE).issues
    assert [i for i in full_issues if i.session <= EXPECTED[2]] == past_issues
