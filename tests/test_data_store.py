from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd
import pytest

from bot.data.quality import DataIssue, IssueKind
from bot.data.store import BarStore, StoreError

FETCHED_AT = datetime(2026, 11, 2, 12, 0, tzinfo=UTC)
ISSUE = DataIssue(session=date(2024, 1, 4), kind=IssueKind.MISSING_SESSION, removed=False)


def make_bars(close: float = 100.0) -> pd.DataFrame:
    return pd.DataFrame(
        {"close": [close, close + 1]},
        index=pd.DatetimeIndex(pd.to_datetime(["2024-01-02", "2024-01-03"]), name="session"),
    )


def save(store: BarStore, bars: pd.DataFrame, *, overwrite: bool = False) -> None:
    store.save(
        "SPY",
        bars,
        feed="sip",
        adjustment="all",
        fetched_at=FETCHED_AT,
        issues=[ISSUE],
        overwrite=overwrite,
    )


def test_round_trip_preserves_bars(tmp_path: Path) -> None:
    store = BarStore(tmp_path / "bars")
    save(store, make_bars())
    pd.testing.assert_frame_equal(store.load("SPY"), make_bars(), check_freq=False)


def test_metadata_records_provenance(tmp_path: Path) -> None:
    store = BarStore(tmp_path)
    save(store, make_bars())
    metadata = store.load_metadata("SPY")
    assert metadata.feed == "sip"
    assert metadata.adjustment == "all"
    assert metadata.start == date(2024, 1, 2)
    assert metadata.end == date(2024, 1, 3)
    assert metadata.fetched_at == FETCHED_AT
    assert metadata.rows == 2
    assert metadata.issues == [ISSUE]


def test_refuses_to_replace_without_request(tmp_path: Path) -> None:
    store = BarStore(tmp_path)
    save(store, make_bars())
    with pytest.raises(StoreError, match="existent"):
        save(store, make_bars(200.0))
    assert store.load("SPY")["close"].iloc[0] == 100.0


def test_overwrite_replaces_data(tmp_path: Path) -> None:
    store = BarStore(tmp_path)
    save(store, make_bars())
    save(store, make_bars(200.0), overwrite=True)
    assert store.load("SPY")["close"].iloc[0] == 200.0


def test_load_detects_modified_file(tmp_path: Path) -> None:
    store = BarStore(tmp_path)
    save(store, make_bars())
    make_bars(200.0).to_parquet(tmp_path / "SPY.parquet")
    with pytest.raises(StoreError, match="Empreinte"):
        store.load("SPY")


def test_load_missing_symbol(tmp_path: Path) -> None:
    with pytest.raises(StoreError, match="Provenance"):
        BarStore(tmp_path).load("SPY")


def test_load_missing_bars_file(tmp_path: Path) -> None:
    store = BarStore(tmp_path)
    save(store, make_bars())
    (tmp_path / "SPY.parquet").unlink()
    with pytest.raises(StoreError, match="illisibles"):
        store.load("SPY")


def test_corrupt_metadata(tmp_path: Path) -> None:
    store = BarStore(tmp_path)
    save(store, make_bars())
    (tmp_path / "SPY.json").write_text("pas du json")
    with pytest.raises(StoreError, match="Provenance"):
        store.load("SPY")


def test_refuses_empty_bars(tmp_path: Path) -> None:
    with pytest.raises(StoreError, match="Aucune barre"):
        save(BarStore(tmp_path), make_bars().iloc[0:0])


def save_symbol(store: BarStore, symbol: str, bars: pd.DataFrame) -> None:
    store.save(symbol, bars, feed="sip", adjustment="all", fetched_at=FETCHED_AT, issues=[])


def test_load_closes_aligns_symbols_without_filling(tmp_path: Path) -> None:
    store = BarStore(tmp_path)
    save_symbol(store, "SPY", make_bars(100.0))
    save_symbol(store, "TLT", make_bars(50.0).iloc[1:])

    closes = store.load_closes(["SPY", "TLT"])

    assert list(closes.columns) == ["SPY", "TLT"]
    assert closes.index.is_monotonic_increasing
    assert closes.loc[pd.Timestamp("2024-01-02"), "SPY"] == 100.0
    assert closes["TLT"].isna().tolist() == [True, False]
    assert closes.loc[pd.Timestamp("2024-01-03"), "TLT"] == 51.0


def test_load_closes_requires_symbols(tmp_path: Path) -> None:
    with pytest.raises(StoreError, match="Aucun symbole"):
        BarStore(tmp_path).load_closes([])


def test_load_closes_detects_modified_file(tmp_path: Path) -> None:
    store = BarStore(tmp_path)
    save_symbol(store, "SPY", make_bars())
    make_bars(200.0).to_parquet(tmp_path / "SPY.parquet")
    with pytest.raises(StoreError, match="Empreinte"):
        store.load_closes(["SPY"])


def test_exists(tmp_path: Path) -> None:
    store = BarStore(tmp_path)
    assert store.exists("SPY") is False
    save(store, make_bars())
    assert store.exists("SPY") is True
