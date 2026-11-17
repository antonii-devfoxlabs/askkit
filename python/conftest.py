# Here, at the top of python/, not in tests/: evals/ has a conftest of
# its own, and mypy refuses two modules both named conftest.

from itertools import count

import pytest


@pytest.fixture(autouse=True)
def steady_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    # A real clock would print a different time on every run. This one
    # moves 0.42 s each time AskKit reads it, so a streamed answer shows
    # its first token at 0.42 s and ends at 0.84 s.
    readings = count(start=0.0, step=0.42)
    monkeypatch.setattr("askkit.ask.perf_counter", lambda: next(readings))
