from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import pytest
from alpaca.common.exceptions import APIError
from alpaca.common.types import RawData
from alpaca.data.enums import Adjustment, DataFeed
from alpaca.data.models import BarSet, Snapshot
from alpaca.data.requests import StockBarsRequest, StockSnapshotRequest
from alpaca.trading.models import Calendar
from alpaca.trading.requests import GetCalendarRequest
from requests.exceptions import ConnectionError as RequestsConnectionError

from bot.config import DataConfig, RetryConfig, Settings, StrategyConfig, load_strategy_config
from bot.data import download
from bot.data.market import (
    BAR_COLUMNS,
    AlpacaMarketData,
    DataFetchError,
    Session,
    fetch_bars,
    fetch_latest_prices,
    fetch_sessions,
    last_completed_session,
)
from bot.data.quality import IssueKind
from bot.data.store import BarStore, StoreError

NEW_YORK = ZoneInfo("America/New_York")
RETRY = RetryConfig(max_attempts=1, base_delay_seconds=1.0, max_delay_seconds=1.0)
DATA = DataConfig(
    symbols=["SPY"],
    start_date=date(2024, 3, 8),
    end_date=None,
    feed="sip",
    adjustment="all",
    min_bar_age_minutes=15,
    directory="data/bars",
)
# Le 8 mars 2024 est en heure d'hiver à New York, le 11 mars en heure d'été.
SESSIONS = [date(2024, 3, 8), date(2024, 3, 11)]
SESSION_TIMESTAMPS = {
    date(2024, 3, 8): "2024-03-08T05:00:00Z",
    date(2024, 3, 11): "2024-03-11T04:00:00Z",
}


def raw_bar(timestamp: str, close: float = 100.0) -> dict[str, object]:
    return {
        "t": timestamp,
        "o": close,
        "h": close + 1,
        "l": close - 1,
        "c": close,
        "v": 1000,
        "n": 10,
        "vw": close,
    }


def make_barset(symbol: str, days: list[date]) -> BarSet:
    return BarSet({symbol: [raw_bar(SESSION_TIMESTAMPS[day]) for day in days]})


def make_calendar(days: list[date]) -> list[Calendar]:
    return [Calendar(date=day.isoformat(), open="09:30", close="16:00") for day in days]


class FakeBarsClient:
    def __init__(self, result: BarSet | RawData | None = None) -> None:
        self.result = result
        self.requests: list[StockBarsRequest] = []

    def get_stock_bars(self, request_params: StockBarsRequest) -> BarSet | RawData:
        self.requests.append(request_params)
        if self.result is not None:
            return self.result
        symbol = str(request_params.symbol_or_symbols)
        return make_barset(symbol, SESSIONS)


class FakeCalendarClient:
    def __init__(self, result: list[Calendar] | RawData | None = None) -> None:
        self.result = make_calendar(SESSIONS) if result is None else result
        self.filters: GetCalendarRequest | None = None

    def get_calendar(self, filters: GetCalendarRequest | None = None) -> list[Calendar] | RawData:
        self.filters = filters
        return self.result


def close_of(day: date) -> datetime:
    return datetime.combine(day, time(16), tzinfo=NEW_YORK)


def test_fetch_sessions_localizes_close() -> None:
    client = FakeCalendarClient()
    sessions = fetch_sessions(client, SESSIONS[0], SESSIONS[1], RETRY)
    assert [session.date for session in sessions] == SESSIONS
    assert sessions[0].close == close_of(SESSIONS[0])
    assert sessions[0].close.astimezone(UTC).hour == 21
    assert sessions[1].close.astimezone(UTC).hour == 20
    assert client.filters is not None
    assert client.filters.start == SESSIONS[0]


def test_fetch_sessions_rejects_raw_response() -> None:
    with pytest.raises(DataFetchError):
        fetch_sessions(FakeCalendarClient({"raw": 1}), SESSIONS[0], SESSIONS[1], RETRY)


@pytest.mark.parametrize(
    ("offset", "expected"),
    [
        (timedelta(minutes=14), SESSIONS[0]),
        (timedelta(minutes=15), SESSIONS[1]),
        (timedelta(days=2), SESSIONS[1]),
    ],
)
def test_last_completed_session_waits_after_close(offset: timedelta, expected: date) -> None:
    sessions = [Session(day, close_of(day)) for day in SESSIONS]
    now = close_of(SESSIONS[1]) + offset
    assert last_completed_session(sessions, now, timedelta(minutes=15)).date == expected


def test_last_completed_session_excludes_session_in_progress() -> None:
    sessions = [Session(day, close_of(day)) for day in SESSIONS]
    during_session = datetime.combine(SESSIONS[1], time(12), tzinfo=NEW_YORK)
    assert last_completed_session(sessions, during_session, timedelta(0)).date == SESSIONS[0]


def test_last_completed_session_requires_one() -> None:
    sessions = [Session(SESSIONS[0], close_of(SESSIONS[0]))]
    with pytest.raises(DataFetchError):
        last_completed_session(sessions, close_of(SESSIONS[0]), timedelta(minutes=15))


def test_fetch_bars_indexes_by_session_date_across_dst() -> None:
    bars = fetch_bars(FakeBarsClient(), "SPY", SESSIONS[0], close_of(SESSIONS[1]), DATA, RETRY)
    assert list(bars.index) == [pd.Timestamp(day) for day in SESSIONS]
    assert bars.index.name == "session"
    assert pd.DatetimeIndex(bars.index).tz is None
    assert list(bars.columns) == BAR_COLUMNS


def test_fetch_bars_builds_request_from_config() -> None:
    client = FakeBarsClient()
    end = close_of(SESSIONS[1])
    fetch_bars(client, "SPY", SESSIONS[0], end, DATA, RETRY)
    request = client.requests[0]
    assert request.feed is DataFeed.SIP
    assert request.adjustment is Adjustment.ALL
    assert request.end == end.astimezone(UTC).replace(tzinfo=None)


def test_fetch_bars_rejects_raw_response() -> None:
    with pytest.raises(DataFetchError, match="brute"):
        fetch_bars(
            FakeBarsClient({"SPY": []}), "SPY", SESSIONS[0], close_of(SESSIONS[1]), DATA, RETRY
        )


def test_fetch_bars_rejects_empty_response() -> None:
    with pytest.raises(DataFetchError, match="Aucune barre"):
        fetch_bars(
            FakeBarsClient(BarSet({"SPY": []})),
            "SPY",
            SESSIONS[0],
            close_of(SESSIONS[1]),
            DATA,
            RETRY,
        )


def make_config() -> StrategyConfig:
    return load_strategy_config().model_copy(update={"data": DATA})


def test_download_universe_stores_bars_and_quality_report(tmp_path: Path) -> None:
    store = BarStore(tmp_path)
    now = close_of(SESSIONS[1]) + timedelta(hours=1)
    bars_client = FakeBarsClient(make_barset("SPY", SESSIONS[:1]))

    download.download_universe(make_config(), bars_client, FakeCalendarClient(), store, now=now)

    assert list(store.load("SPY").index) == [pd.Timestamp(SESSIONS[0])]
    metadata = store.load_metadata("SPY")
    assert metadata.fetched_at == now
    assert [issue.kind for issue in metadata.issues] == [IssueKind.MISSING_SESSION]
    assert bars_client.requests[0].end == close_of(SESSIONS[1]).astimezone(UTC).replace(tzinfo=None)


def test_download_universe_stops_at_last_completed_session(tmp_path: Path) -> None:
    store = BarStore(tmp_path)
    now = close_of(SESSIONS[0]) + timedelta(hours=1)
    bars_client = FakeBarsClient(make_barset("SPY", SESSIONS[:1]))

    download.download_universe(make_config(), bars_client, FakeCalendarClient(), store, now=now)

    assert bars_client.requests[0].end == close_of(SESSIONS[0]).astimezone(UTC).replace(tzinfo=None)
    assert store.load_metadata("SPY").issues == []


def test_download_universe_refuses_existing_data_before_any_request(tmp_path: Path) -> None:
    store = BarStore(tmp_path)
    now = close_of(SESSIONS[1]) + timedelta(hours=1)
    download.download_universe(
        make_config(), FakeBarsClient(), FakeCalendarClient(), store, now=now
    )
    bars_client = FakeBarsClient()

    with pytest.raises(StoreError, match="SPY"):
        download.download_universe(make_config(), bars_client, FakeCalendarClient(), store, now=now)

    assert bars_client.requests == []


def test_download_universe_overwrite(tmp_path: Path) -> None:
    store = BarStore(tmp_path)
    now = close_of(SESSIONS[1]) + timedelta(hours=1)
    config = make_config()
    download.download_universe(config, FakeBarsClient(), FakeCalendarClient(), store, now=now)
    download.download_universe(
        config, FakeBarsClient(), FakeCalendarClient(), store, now=now, overwrite=True
    )
    assert store.load_metadata("SPY").rows == 2


def test_main_reports_store_error(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def fail(*_: object, **__: object) -> None:
        raise StoreError("Des barres existent déjà pour : SPY")

    monkeypatch.setattr(
        download,
        "load_settings",
        lambda: Settings(alpaca_api_key="cle", alpaca_secret_key="secret", _env_file=None),
    )
    monkeypatch.setattr(download, "download_universe", fail)

    assert download.main([]) == 1
    assert "download_failed" in capsys.readouterr().out


def test_fetch_bars_rejects_missing_column() -> None:
    # Alpaca retire du DataFrame toute colonne entièrement vide.
    bars = [{**raw_bar(SESSION_TIMESTAMPS[SESSIONS[0]]), "vw": None}]
    with pytest.raises(DataFetchError, match="vwap"):
        fetch_bars(
            FakeBarsClient(BarSet({"SPY": bars})),
            "SPY",
            SESSIONS[0],
            close_of(SESSIONS[1]),
            DATA,
            RETRY,
        )


@pytest.mark.parametrize(
    ("error", "reason"),
    [
        (APIError('{"code": 1, "message": "interdit"}'), "api_error"),  # type: ignore[no-untyped-call]
        (RequestsConnectionError("hors ligne"), "network_error"),
    ],
)
def test_main_reports_remote_errors(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    error: Exception,
    reason: str,
) -> None:
    def fail(*_: object, **__: object) -> None:
        raise error

    settings = Settings(alpaca_api_key="cle", alpaca_secret_key="secret", _env_file=None)
    monkeypatch.setattr(download, "load_settings", lambda: settings)
    monkeypatch.setattr(download, "download_universe", fail)

    assert download.main([]) == 1
    assert reason in capsys.readouterr().out


TRADE_TIME = "2024-03-11T14:00:00Z"


def make_snapshot(symbol: str, trade: dict[str, object] | None) -> Snapshot:
    bar = raw_bar(TRADE_TIME)
    quote = {
        "t": TRADE_TIME,
        "ax": "V",
        "ap": 100.0,
        "as": 1,
        "bx": "V",
        "bp": 99.0,
        "bs": 1,
        "c": [],
        "z": "A",
    }
    raw: Any = {
        "latestTrade": trade,
        "latestQuote": quote,
        "minuteBar": bar,
        "dailyBar": bar,
        "prevDailyBar": bar,
    }
    return Snapshot(symbol, raw)


TRADE = {"t": TRADE_TIME, "x": "V", "p": 101.5, "s": 10, "c": [], "i": 1, "z": "A"}


class FakeLiveClient(FakeBarsClient):
    def __init__(
        self,
        snapshots: dict[str, Snapshot] | RawData | None = None,
        empty_symbols: tuple[str, ...] = (),
        available: list[date] | None = None,
    ) -> None:
        super().__init__()
        self.available = SESSIONS if available is None else available
        self.snapshots = snapshots or {}
        self.empty_symbols = empty_symbols
        self.snapshot_requests: list[StockSnapshotRequest] = []

    def get_stock_bars(self, request_params: StockBarsRequest) -> BarSet | RawData:
        self.requests.append(request_params)
        symbol = str(request_params.symbol_or_symbols)
        if symbol in self.empty_symbols:
            return BarSet({symbol: []})
        return make_barset(symbol, self.available)

    def get_stock_snapshot(
        self, request_params: StockSnapshotRequest
    ) -> dict[str, Snapshot] | RawData:
        self.snapshot_requests.append(request_params)
        return self.snapshots


def session_list() -> list[Session]:
    return [Session(day, close_of(day)) for day in SESSIONS]


def test_fetch_latest_prices_reads_last_trade() -> None:
    client = FakeLiveClient({"SPY": make_snapshot("SPY", TRADE)})
    prices = fetch_latest_prices(client, ["SPY"], "iex", RETRY)
    assert prices["SPY"].price == 101.5
    assert prices["SPY"].as_of == datetime(2024, 3, 11, 14, tzinfo=UTC)
    assert client.snapshot_requests[0].feed is DataFeed.IEX


def test_fetch_latest_prices_omits_symbol_without_trade() -> None:
    client = FakeLiveClient({"SPY": make_snapshot("SPY", None)})
    assert fetch_latest_prices(client, ["SPY"], "iex", RETRY) == {}


def test_fetch_latest_prices_rejects_raw_response() -> None:
    client = FakeLiveClient({"SPY": {"raw": 1}})
    with pytest.raises(DataFetchError, match="brute"):
        fetch_latest_prices(client, ["SPY"], "iex", RETRY)


def live_config(symbols: list[str]) -> StrategyConfig:
    config = make_config()
    return config.model_copy(update={"data": DATA.model_copy(update={"symbols": symbols})})


def test_market_data_closes_keeps_every_symbol_on_expected_sessions() -> None:
    client = FakeLiveClient()
    market = AlpacaMarketData(client, FakeCalendarClient(), live_config(["SPY", "TLT"]), RETRY)

    closes = market.closes(session_list())

    assert list(closes.columns) == ["SPY", "TLT"]
    assert list(closes.index) == [pd.Timestamp(day) for day in SESSIONS]
    assert closes.notna().all().all()
    assert client.requests[0].end == close_of(SESSIONS[1]).astimezone(UTC).replace(tzinfo=None)


def test_market_data_closes_leaves_gap_for_missing_session() -> None:
    client = FakeLiveClient(available=SESSIONS[:1])
    market = AlpacaMarketData(client, FakeCalendarClient(), live_config(["SPY"]), RETRY)

    closes = market.closes(session_list())

    assert closes["SPY"].notna().tolist() == [True, False]


def test_market_data_closes_survives_symbol_without_data() -> None:
    client = FakeLiveClient(empty_symbols=("TLT",))
    market = AlpacaMarketData(client, FakeCalendarClient(), live_config(["SPY", "TLT"]), RETRY)

    closes = market.closes(session_list())

    assert closes["SPY"].notna().all()
    assert closes["TLT"].isna().all()


def test_market_data_sessions_and_latest_prices() -> None:
    client = FakeLiveClient({"SPY": make_snapshot("SPY", TRADE)})
    market = AlpacaMarketData(client, FakeCalendarClient(), live_config(["SPY"]), RETRY)
    assert [item.date for item in market.sessions(SESSIONS[0], SESSIONS[1])] == SESSIONS
    assert market.latest_prices()["SPY"].price == 101.5
