import json
import math
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import streamlit as st
from alpaca.trading.models import Calendar, PortfolioHistory
from alpaca.trading.requests import GetCalendarRequest, GetPortfolioHistoryRequest
from streamlit.testing.v1 import AppTest

from bot.config import PROJECT_ROOT
from bot.dashboard import __main__ as launcher
from bot.dashboard import data
from bot.data.market import LatestPrice
from bot.execution.broker import Broker, BrokerError, EquitySample, MarketClock
from bot.journal.store import Decision, EquityPoint, Journal
from bot.risk.models import HaltReason, OrderSide
from bot.risk.state import RiskState
from tests.test_main import FakeClient, make_config, make_position
from tests.test_status import sample_status, wire_status

NOW = datetime(2026, 11, 9, 18, 0, tzinfo=UTC)
TODAY = date(2026, 11, 9)
APP = PROJECT_ROOT / "src" / "bot" / "dashboard" / "app.py"


def stamp(day: date, hour: int = 5) -> datetime:
    return datetime(day.year, day.month, day.day, hour, tzinfo=UTC)


def history(timeframe: str, points: list[tuple[datetime, float]]) -> PortfolioHistory:
    return PortfolioHistory(
        timestamp=[int(moment.timestamp()) for moment, _ in points],
        equity=[value for _, value in points],
        profit_loss=[0.0] * len(points),
        profit_loss_pct=[0.0] * len(points),
        timeframe=timeframe,
    )


# Broker


def test_position_details_keep_cost_and_unrealized_result() -> None:
    client = FakeClient()
    position = make_position("A", 10.0).model_copy(
        update={"unrealized_pl": "12.5", "unrealized_plpc": "0.0125", "change_today": "-0.01"}
    )
    client.positions = [make_position("B", -2.0), position]
    details = Broker(client, make_config().api_retry).position_details()
    assert [(item.symbol, item.quantity) for item in details] == [("A", 10.0), ("B", -2.0)]
    assert (details[0].unrealized_pl, details[0].change_today) == (12.5, -0.01)
    assert details[0].market_value == 1000.0


def test_portfolio_history_drops_unknown_windows() -> None:
    client = FakeClient()
    client.history["1D"] = history(
        "1D",
        [
            (stamp(date(2026, 11, 4)), 0.0),
            (stamp(date(2026, 11, 5)), 100.0),
            (stamp(date(2026, 11, 6)), math.nan),
        ],
    )
    request = GetPortfolioHistoryRequest(timeframe="1D")
    samples = Broker(client, make_config().api_retry).portfolio_history(request)
    assert samples == [EquitySample(stamp(date(2026, 11, 5)), 100.0)]


def test_portfolio_history_refuses_raw_response() -> None:
    class RawClient(FakeClient):
        def get_portfolio_history(self, history_filter: Any = None) -> Any:
            return {"equity": []}

    broker = Broker(RawClient(), make_config().api_retry)
    with pytest.raises(BrokerError, match="historique"):
        broker.portfolio_history(GetPortfolioHistoryRequest(timeframe="1D"))


# Calculs


def test_last_log_time_reads_the_last_valid_line(tmp_path: Path) -> None:
    assert data.last_log_time(tmp_path / "absent") is None
    lines = ['{"timestamp": "2026-11-09T17:58:00+00:00"}', "illisible", ""]
    (tmp_path / "bot-2026-11-09.jsonl").write_text("\n".join(lines), encoding="utf-8")
    assert data.last_log_time(tmp_path) == datetime(2026, 11, 9, 17, 58, tzinfo=UTC)


@pytest.mark.parametrize(
    ("signals", "active"),
    [
        ([None, None], False),
        ([NOW - timedelta(minutes=10), None], False),
        ([NOW - timedelta(minutes=10), NOW - timedelta(minutes=1)], True),
    ],
)
def test_heartbeat_uses_the_latest_sign_of_life(
    signals: list[datetime | None], active: bool
) -> None:
    assert data.heartbeat(signals, NOW, timedelta(minutes=3)).active is active


def test_session_equity_starts_from_previous_close_and_uses_live_value_today() -> None:
    samples = [
        EquitySample(stamp(date(2026, 11, 3)), 99.0),
        EquitySample(stamp(date(2026, 11, 5)), 100.0),
        EquitySample(stamp(date(2026, 11, 6)), 101.0),
        EquitySample(stamp(TODAY), 101.0),
    ]
    equity = data.session_equity(samples, date(2026, 11, 6), TODAY, 103.0)
    days = list(pd.DatetimeIndex(equity.index).date)
    assert days == [date(2026, 11, 5), date(2026, 11, 6), TODAY]
    assert equity.tolist() == [100.0, 101.0, 103.0]


def test_session_equity_without_any_prior_close() -> None:
    samples = [EquitySample(stamp(TODAY), 100.0)]
    assert data.session_equity(samples, TODAY, TODAY, 104.0).tolist() == [104.0]


def test_intraday_equity_is_in_new_york_time() -> None:
    series = data.intraday_equity([EquitySample(datetime(2026, 11, 9, 14, 30, tzinfo=UTC), 1.0)])
    assert series.index[0].hour == 9


def test_normalized_benchmark_aligns_and_fills_holidays() -> None:
    closes = pd.Series(
        [10.0, 11.0, 12.0], index=pd.to_datetime(["2026-11-05", "2026-11-06", "2026-11-09"])
    )
    index = pd.DatetimeIndex(pd.to_datetime(["2026-11-05", "2026-11-07", "2026-11-09"]))
    benchmark = data.normalized_benchmark(closes, index, 1000.0)
    assert benchmark.tolist() == pytest.approx([1000.0, 1100.0, 1200.0])


def test_normalized_benchmark_without_starting_price() -> None:
    closes = pd.Series([10.0], index=pd.to_datetime(["2026-11-09"]))
    index = pd.DatetimeIndex(pd.to_datetime(["2026-11-06", "2026-11-09"]))
    assert data.normalized_benchmark(closes, index, 1.0).isna().all()
    assert data.normalized_benchmark(closes, pd.DatetimeIndex([]), 1.0).empty


def test_drawdown_and_performance() -> None:
    equity = pd.Series(
        [100.0, 110.0, 99.0, 104.5],
        index=pd.to_datetime(["2026-11-04", "2026-11-05", "2026-11-06", "2026-11-09"]),
    )
    assert data.drawdown_series(equity).tolist() == pytest.approx([0.0, 0.0, 0.1, 0.05])
    metrics = data.performance(equity)
    assert metrics is not None and metrics.sessions == 3
    assert metrics.max_drawdown == pytest.approx(-0.1)
    assert data.performance(equity.iloc[:1]) is None


def test_history_view_returns() -> None:
    index = pd.to_datetime(["2026-11-06", "2026-11-09"])
    view = data.HistoryView(
        start=TODAY,
        equity=pd.Series([100.0, 102.0], index=index),
        benchmark=pd.Series([math.nan, 100.0], index=index),
        drawdown=pd.Series(dtype=float),
        metrics=None,
        benchmark_metrics=None,
        bot_sessions=[TODAY],
    )
    assert view.total_return == pytest.approx(0.02)
    assert view.benchmark_return is None


def write_journal(path: Path) -> None:
    journal = Journal(path)
    journal.record_decision(
        Decision(NOW, TODAY, "A", 1, "hold", {"target_weight": 0.1}, risk_approved=None)
    )
    decision_id = journal.record_decision(
        Decision(
            datetime(2026, 11, 9, 9, 31, tzinfo=datetime.now().astimezone().tzinfo),
            TODAY,
            "B",
            1,
            "order",
            {"target_weight": 0.2},
            side=OrderSide.BUY,
            quantity=3.0,
            risk_approved=False,
            risk_reason="stale_price",
        )
    )
    journal.record_status(decision_id, "", is_final=True, detail="")
    journal.record_equity(EquityPoint(TODAY, 100.0, 20.0, NOW - timedelta(minutes=1)))
    journal.close()


def test_decision_table_is_most_recent_first(tmp_path: Path) -> None:
    write_journal(tmp_path / "journal.db")
    journal = Journal(tmp_path / "journal.db", read_only=True)
    rows = journal.decisions()
    journal.close()
    rows[0]["slippage_pct"] = 0.001
    table = data.decision_table(rows, 5)
    assert table["symbole"].tolist() == ["B", "A"]
    assert table["risque"].tolist() == ["refusé : stale_price", ""]
    assert table["ordre"].tolist() == ["buy 3", ""]
    assert table["heure (UTC)"].iloc[1] == NOW
    assert data.decision_table(rows, 1)["symbole"].tolist() == ["B"]
    assert data.mean_slippage(rows) == (0.001, 1)
    assert data.mean_slippage([]) == (None, 0)


def test_read_journal_never_creates_it(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(data, "PROJECT_ROOT", tmp_path)
    config = make_config()
    view = data.read_journal(config)
    assert (view.decisions, view.bot_sessions, view.last_record) == ([], [], None)
    assert not (tmp_path / config.live.journal_file).exists()

    write_journal(tmp_path / config.live.journal_file)
    view = data.read_journal(config)
    assert view.bot_sessions == [TODAY]
    assert view.last_record == NOW - timedelta(minutes=1)


# Chargement


class CalendarFakeClient(FakeClient):
    """Client complet pour le suivi : le calendrier est remplacé par `wire_status`."""

    def get_calendar(self, filters: GetCalendarRequest | None = None) -> list[Calendar]:
        return []


def wire(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, client: FakeClient) -> None:
    wire_status(monkeypatch, tmp_path, client)
    monkeypatch.setattr(data, "PROJECT_ROOT", tmp_path)


def test_load_live_reads_account_positions_and_heartbeat(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    client = CalendarFakeClient()
    client.positions = [make_position("A", 10.0)]
    client.history["5Min"] = history("5Min", [(NOW - timedelta(minutes=5), 100_000.0)])
    wire(monkeypatch, tmp_path, client)
    write_journal(tmp_path / make_config().live.journal_file)

    live = data.load_live(make_config(), client, NOW)

    assert live.status.equity == 100_000.0
    assert [item.symbol for item in live.positions] == ["A"]
    assert live.intraday.tolist() == [100_000.0]
    assert live.heartbeat.active
    request = client.history_requests[-1]
    assert (request.period, request.timeframe) == ("1D", "5Min")
    assert client.submitted == []


def test_load_history_without_journal(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    client = CalendarFakeClient()
    wire(monkeypatch, tmp_path, client)
    view = data.load_history(make_config(), client, object(), NOW)  # type: ignore[arg-type]
    assert view.start is None and view.equity.empty
    assert client.history_requests == []


def test_load_history_compares_to_the_benchmark(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    client = CalendarFakeClient()
    client.equity = "102000"
    client.history["1D"] = history(
        "1D", [(stamp(date(2026, 11, 6)), 100_000.0), (stamp(TODAY), 100_500.0)]
    )
    wire(monkeypatch, tmp_path, client)
    write_journal(tmp_path / make_config().live.journal_file)
    requested: list[datetime] = []

    def bars(client: object, symbol: str, start: date, end: datetime, *_: object) -> pd.DataFrame:
        requested.append(end)
        index = pd.to_datetime(["2026-11-06", "2026-11-09"])
        return pd.DataFrame({"close": [500.0, 999.0]}, index=index)

    def latest(*_: object) -> dict[str, LatestPrice]:
        return {"SPY": LatestPrice(505.0, NOW)}

    monkeypatch.setattr(data, "fetch_bars", bars)
    monkeypatch.setattr(data, "fetch_latest_prices", latest)
    config = make_config()

    view = data.load_history(config, client, object(), NOW)  # type: ignore[arg-type]

    assert view.start == TODAY
    assert view.equity.tolist() == [100_000.0, 102_000.0]
    # La barre du jour, encore partielle, est remplacée par le dernier prix.
    assert view.benchmark.tolist() == pytest.approx([100_000.0, 101_000.0])
    assert view.total_return == pytest.approx(0.02)
    assert view.benchmark_return == pytest.approx(0.01)
    assert requested[0] < NOW - timedelta(hours=1)


# Interface


def fake_live(**status: Any) -> data.LiveView:
    return data.LiveView(
        status=sample_status(**status),
        clock=MarketClock(True, NOW, NOW, NOW),
        positions=Broker(FakeClient(), make_config().api_retry).position_details(),
        intraday=data.intraday_equity([EquitySample(NOW, 95_000.0)]),
        heartbeat=data.Heartbeat(NOW, True),
    )


def fake_history() -> data.HistoryView:
    equity = pd.Series([100_000.0, 95_000.0], index=pd.to_datetime(["2026-11-06", "2026-11-09"]))
    return data.HistoryView(
        start=TODAY,
        equity=equity,
        benchmark=equity * 1.01,
        drawdown=data.drawdown_series(equity),
        metrics=data.performance(equity),
        benchmark_metrics=data.performance(equity * 1.01),
        bot_sessions=[TODAY],
    )


@pytest.fixture
def app(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> AppTest:
    st.cache_data.clear()
    st.cache_resource.clear()
    monkeypatch.setattr(data, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(data, "connect", lambda: (FakeClient(), object()))
    monkeypatch.setattr(data, "load_live", lambda *_: fake_live())
    monkeypatch.setattr(data, "load_history", lambda *_: fake_history())
    return AppTest.from_file(str(APP), default_timeout=60)


def test_app_renders_live_and_history(app: AppTest) -> None:
    app.run()
    assert not app.exception
    labels = {metric.label: metric.value for metric in app.metric}
    assert labels["Bot"] == "actif"
    assert labels["Exposition"] == "80.0%"
    assert any("Échantillon" in warning.value for warning in app.warning)
    assert [tab.label for tab in app.tabs] == ["Aujourd'hui", "Positions", "Décisions", "Logs"]


def test_app_shows_a_triggered_kill_switch(app: AppTest, monkeypatch: pytest.MonkeyPatch) -> None:
    risk = RiskState(peak_equity=100_000.0, halt_reason=HaltReason.DAILY_LOSS, halted_at=NOW)
    monkeypatch.setattr(data, "load_live", lambda *_: fake_live(risk=risk))
    app.run()
    assert not app.exception
    assert any("Kill switch déclenché" in error.value for error in app.error)


def test_app_survives_api_failures_and_empty_history(
    app: AppTest, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(*_: object) -> data.LiveView:
        raise BrokerError("panne")

    monkeypatch.setattr(data, "load_live", fail)
    monkeypatch.setattr(
        data,
        "load_history",
        lambda *_: data.HistoryView(None, pd.Series(), pd.Series(), pd.Series(), None, None, []),
    )
    app.run()
    assert not app.exception
    assert any("BrokerError" in error.value for error in app.error)
    assert any("Pas encore d'historique" in info.value for info in app.info)


def test_launcher_listens_on_localhost_only() -> None:
    args = launcher.streamlit_args(8501)
    assert args[:2] == ["streamlit", "run"]
    assert Path(args[2]) == APP
    options = dict(zip(args[3::2], args[4::2], strict=True))
    assert options["--server.address"] == "localhost"
    assert options["--browser.gatherUsageStats"] == "false"


def test_status_journal_round_trip_is_json(tmp_path: Path) -> None:
    """Les poids visés du journal restent lisibles par le dashboard."""
    write_journal(tmp_path / "journal.db")
    journal = Journal(tmp_path / "journal.db", read_only=True)
    inputs = [json.loads(row["inputs"]) for row in journal.decisions()]
    journal.close()
    assert [item["target_weight"] for item in inputs] == [0.1, 0.2]
