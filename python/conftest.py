# Here, at the top of python/, not in tests/: evals/ has a conftest of
# its own, and mypy refuses two modules both named conftest.

from itertools import count
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def steady_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    # A real clock would print a different time on every run. This one
    # moves 0.42 s each time AskKit reads it, so a streamed answer shows
    # its first token at 0.42 s and ends at 0.84 s.
    readings = count(start=0.0, step=0.42)
    monkeypatch.setattr("askkit.ask.perf_counter", lambda: next(readings))
    monkeypatch.setattr(
        "askkit.structured.perf_counter", lambda: next(readings)
    )


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
