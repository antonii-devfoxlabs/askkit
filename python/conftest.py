# Here, at the top of python/, not in tests/: evals/ has a conftest of
# its own, and mypy refuses two modules both named conftest.

from datetime import UTC, datetime
from itertools import count
from pathlib import Path

import httpx2
import pytest


@pytest.fixture(autouse=True)
def steady_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    # A real clock would print a different time on every run. This one
    # moves 0.42 s each time AskKit reads it, so a streamed answer shows
    # its first token at 0.42 s and ends at 0.84 s. The CLI's clock,
    # which askkit inbox times each call on, moves the same steps on a
    # count of its own, so the two never take each other's readings.
    readings = count(start=0.0, step=0.42)
    monkeypatch.setattr("askkit.ask.perf_counter", lambda: next(readings))
    monkeypatch.setattr(
        "askkit.structured.perf_counter", lambda: next(readings)
    )
    cli_readings = count(start=0.0, step=0.42)
    monkeypatch.setattr("askkit.cli.perf_counter", lambda: next(cli_readings))


@pytest.fixture
def labelled_now(monkeypatch: pytest.MonkeyPatch) -> None:
    # The CLI's time of day, fixed, for a test that reads the time a
    # label says its JSON was written (askkit/label.py).
    moment = datetime(2026, 11, 21, 9, 30, tzinfo=UTC)
    monkeypatch.setattr("askkit.cli.now", lambda: moment)


@pytest.fixture(autouse=True)
def own_cache_folder(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # A test never reads or writes the response cache of the folder it
    # runs from: an answer kept by one test would answer the next.
    monkeypatch.setattr("askkit.cli.CACHE_FOLDER", tmp_path / ".askkit-cache")


@pytest.fixture(autouse=True)
def own_log_folder(
    tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A test's calls are logged in a folder of their own, never in the
    # log of the folder it runs from, nor in the test's tmp_path, which
    # some tests expect to stay empty; a run a test started ends with it.
    log_folder = tmp_path_factory.mktemp("log") / ".askkit-log"
    monkeypatch.setattr("askkit.cli.LOG_FOLDER", log_folder)
    monkeypatch.setattr("askkit.calllog.current", None)


class ModelAPIInTests(AssertionError):
    pass


# The model APIs AskKit calls. A test reaching one would send a real,
# maybe billed, request and pass or fail on the network.
MODEL_API_HOSTS = (
    "api.anthropic.com",
    "api.openai.com",
    "generativelanguage.googleapis.com",
)
TESTS = Path(__file__).parent / "tests"


@pytest.fixture(autouse=True)
def no_model_api(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A test that forgets to give AskKit a stand-in would call the real
    # API: lesson 11.2's first draft of the async inbox did exactly that,
    # and got a real 401. Real transports refuse the model APIs here;
    # httpx2.MockTransport is a class of its own, so stand-ins still
    # answer. Only for tests/: an eval answers at the network with a
    # plugin of its own, which this would undo.
    if not request.node.path.is_relative_to(TESTS):
        return
    sync_send = httpx2.HTTPTransport.handle_request
    async_send = httpx2.AsyncHTTPTransport.handle_async_request

    def guarded(
        transport: httpx2.HTTPTransport, sent: httpx2.Request
    ) -> httpx2.Response:
        if sent.url.host in MODEL_API_HOSTS:
            raise ModelAPIInTests(f"a test called {sent.url.host}")
        return sync_send(transport, sent)

    async def guarded_async(
        transport: httpx2.AsyncHTTPTransport, sent: httpx2.Request
    ) -> httpx2.Response:
        if sent.url.host in MODEL_API_HOSTS:
            raise ModelAPIInTests(f"a test called {sent.url.host}")
        return await async_send(transport, sent)

    monkeypatch.setattr(httpx2.HTTPTransport, "handle_request", guarded)
    monkeypatch.setattr(
        httpx2.AsyncHTTPTransport, "handle_async_request", guarded_async
    )
