import json
from decimal import Decimal
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

import pytest
from alpaca.common.enums import BaseURL
from alpaca.common.exceptions import APIError
from alpaca.common.types import RawData
from alpaca.trading.enums import AccountStatus
from alpaca.trading.models import TradeAccount
from requests.exceptions import ConnectionError as RequestsConnectionError

from bot import check_connection
from bot.config import ConfigError, Settings
from bot.execution import client as client_module
from bot.execution.client import (
    AccountDataError,
    PaperModeError,
    create_trading_client,
    read_account,
)

API_KEY = "cle-api-factice"
SECRET_KEY = "cle-secrete-factice"
ACCOUNT_NUMBER = "numero-de-compte-factice"


def make_settings() -> Settings:
    return Settings(alpaca_api_key=API_KEY, alpaca_secret_key=SECRET_KEY, _env_file=None)


def make_account(**overrides: Any) -> TradeAccount:
    fields: dict[str, Any] = {
        "id": uuid4(),
        "account_number": ACCOUNT_NUMBER,
        "status": AccountStatus.ACTIVE,
        "currency": "USD",
        "equity": "100000.50",
        "cash": "100000.50",
        "buying_power": "200001",
        "trading_blocked": False,
        "account_blocked": False,
        **overrides,
    }
    return TradeAccount(**fields)


class FakeClient:
    def __init__(self, account: TradeAccount | RawData) -> None:
        self.account = account

    def get_account(self) -> TradeAccount | RawData:
        return self.account


def test_client_targets_paper_endpoint() -> None:
    client = create_trading_client(make_settings())
    assert client._base_url == BaseURL.TRADING_PAPER
    assert client._sandbox is True


def test_client_refuses_live_mode() -> None:
    with pytest.raises(PaperModeError):
        create_trading_client(make_settings(), paper=False)


def test_client_refuses_endpoint_other_than_paper(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        client_module,
        "TradingClient",
        lambda **_: SimpleNamespace(_base_url=BaseURL.TRADING_LIVE),
    )
    with pytest.raises(PaperModeError):
        create_trading_client(make_settings())


def test_read_account_returns_snapshot() -> None:
    snapshot = read_account(FakeClient(make_account()))
    assert snapshot.equity == Decimal("100000.50")
    assert snapshot.cash == Decimal("100000.50")
    assert snapshot.buying_power == Decimal("200001")
    assert snapshot.currency == "USD"
    assert snapshot.can_trade is True


def test_snapshot_excludes_account_number() -> None:
    snapshot = read_account(FakeClient(make_account()))
    assert ACCOUNT_NUMBER not in repr(snapshot)


@pytest.mark.parametrize(
    "overrides",
    [
        {"status": AccountStatus.ACCOUNT_UPDATED},
        {"trading_blocked": True},
        {"account_blocked": True},
    ],
)
def test_snapshot_cannot_trade(overrides: dict[str, Any]) -> None:
    assert read_account(FakeClient(make_account(**overrides))).can_trade is False


def test_read_account_rejects_raw_response() -> None:
    with pytest.raises(AccountDataError, match="brute"):
        read_account(FakeClient({"equity": "1"}))


@pytest.mark.parametrize(
    ("field", "value"),
    [("equity", None), ("equity", "pas-un-nombre"), ("trading_blocked", None)],
)
def test_read_account_rejects_invalid_field(field: str, value: Any) -> None:
    with pytest.raises(AccountDataError, match=field):
        read_account(FakeClient(make_account(**{field: value})))


def patch_check(monkeypatch: pytest.MonkeyPatch, get_account: Any) -> None:
    monkeypatch.setattr(check_connection, "load_settings", make_settings)
    monkeypatch.setattr(
        check_connection,
        "create_trading_client",
        lambda _: SimpleNamespace(get_account=get_account),
    )


def test_check_connection_succeeds(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    patch_check(monkeypatch, make_account)

    assert check_connection.main() == 0

    output = capsys.readouterr().out
    event = json.loads(output)
    assert event["event"] == "connection_check_ok"
    assert event["equity"] == "100000.50"
    assert event["level"] == "info"
    assert ACCOUNT_NUMBER not in output
    assert API_KEY not in output
    assert SECRET_KEY not in output


def test_check_connection_fails_when_account_cannot_trade(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patch_check(monkeypatch, lambda: make_account(trading_blocked=True))
    assert check_connection.main() == 1


def test_check_connection_reports_missing_keys(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def fail() -> Settings:
        raise ConfigError("Clés API absentes ou invalides : alpaca_api_key")

    monkeypatch.setattr(check_connection, "load_settings", fail)

    assert check_connection.main() == 1
    assert "alpaca_api_key" in capsys.readouterr().out


def test_check_connection_reports_api_status(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def fail() -> TradeAccount:
        http_error = SimpleNamespace(response=SimpleNamespace(status_code=401))
        raise APIError(  # type: ignore[no-untyped-call]
            '{"code": 40110000, "message": "unauthorized"}', http_error
        )

    patch_check(monkeypatch, fail)

    assert check_connection.main() == 1
    assert json.loads(capsys.readouterr().out)["status_code"] == 401


def test_check_connection_reports_network_error(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def fail() -> TradeAccount:
        raise RequestsConnectionError("hors ligne")

    patch_check(monkeypatch, fail)

    assert check_connection.main() == 1
    assert json.loads(capsys.readouterr().out)["reason"] == "network_error"
