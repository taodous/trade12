from types import SimpleNamespace

import pytest
from alpaca.common.exceptions import APIError

from bot.config import RetryConfig
from bot.retry import with_backoff

CONFIG = RetryConfig(max_attempts=5, base_delay_seconds=1.0, max_delay_seconds=3.0)


def api_error(status_code: int) -> APIError:
    http_error = SimpleNamespace(response=SimpleNamespace(status_code=status_code))
    return APIError('{"code": 1, "message": "erreur"}', http_error)  # type: ignore[no-untyped-call]


class FlakyCall:
    def __init__(self, failures: int, status_code: int = 429) -> None:
        self.failures = failures
        self.status_code = status_code
        self.calls = 0

    def __call__(self) -> str:
        self.calls += 1
        if self.calls <= self.failures:
            raise api_error(self.status_code)
        return "ok"


def test_returns_without_waiting_on_success() -> None:
    delays: list[float] = []
    assert with_backoff(FlakyCall(0), CONFIG, sleep=delays.append) == "ok"
    assert delays == []


def test_delays_grow_exponentially_up_to_the_cap() -> None:
    delays: list[float] = []
    assert with_backoff(FlakyCall(4), CONFIG, sleep=delays.append) == "ok"
    assert delays == [1.0, 2.0, 3.0, 3.0]


def test_raises_after_last_attempt() -> None:
    call = FlakyCall(10)
    delays: list[float] = []
    with pytest.raises(APIError):
        with_backoff(call, CONFIG, sleep=delays.append)
    assert call.calls == CONFIG.max_attempts
    assert len(delays) == CONFIG.max_attempts - 1


@pytest.mark.parametrize("status_code", [401, 403, 500])
def test_other_errors_are_not_retried(status_code: int) -> None:
    call = FlakyCall(1, status_code)
    delays: list[float] = []
    with pytest.raises(APIError):
        with_backoff(call, CONFIG, sleep=delays.append)
    assert call.calls == 1
    assert delays == []
