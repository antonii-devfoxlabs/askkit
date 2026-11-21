from collections.abc import Iterator
from pathlib import Path

import httpx2
import pytest
from anthropic import Anthropic

from askkit.attacks import attacks_line
from askkit.config import KEY_NAME, load_api_key
from askkit.prompts import IN_USE
from askkit.recordings import recording, replaying
from askkit.runs import add_run

RECORDING = Path(__file__).parent / "extract" / "recording.json"


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--record",
        action="store_true",
        help="call the model and save its answers in extract/recording.json",
    )
    parser.addoption(
        "--replay",
        action="store_true",
        help="answer from extract/recording.json; no key, no call, no cost",
    )
    parser.addoption(
        "--prompt-version",
        type=int,
        help="the extract_ticket version to score; IN_USE's when not given",
    )
    parser.addoption(
        "--model",
        default="claude-haiku-5-5",
        help="the model to score the prompt on",
    )
    parser.addoption(
        "--escalate-to",
        help="ask this larger model again when an answer fails extract's"
        " checks, as extract --escalate-to does",
    )
    parser.addoption(
        "--results",
        type=Path,
        help="add this run's faults to a results file, for askkit diff",
    )


# One client for the whole run: pytest makes it before the first case and
# hands the same one to every case after. Without a key every case is
# skipped, not failed: a missing key says nothing about the prompt.
@pytest.fixture(scope="session")
def client(request: pytest.FixtureRequest) -> Anthropic:
    if request.config.getoption("--replay"):
        if not RECORDING.is_file():
            pytest.skip(f"no {RECORDING.name} yet: run once with --record")
        return Anthropic(
            api_key="replay",
            http_client=httpx2.Client(transport=replaying(RECORDING)),
        )
    key = load_api_key()
    if key is None:
        pytest.skip(f"{KEY_NAME} is not set: the eval calls the model")
    if request.config.getoption("--record"):
        return Anthropic(
            api_key=key,
            http_client=httpx2.Client(transport=recording(RECORDING)),
        )
    return Anthropic(api_key=key)


@pytest.fixture(scope="session")
def model(request: pytest.FixtureRequest) -> str:
    chosen: str = request.config.getoption("--model")
    return chosen


@pytest.fixture(scope="session")
def escalate_to(request: pytest.FixtureRequest) -> str | None:
    chosen: str | None = request.config.getoption("--escalate-to")
    return chosen


# Every case of a routed run, and whether its first answer failed
# extract's checks and went to the larger model, gathered while the run
# goes and counted at its end.
ESCALATED = pytest.StashKey[dict[str, bool]]()


@pytest.fixture(scope="session")
def escalated(request: pytest.FixtureRequest) -> dict[str, bool]:
    cases: dict[str, bool] = {}
    request.config.stash[ESCALATED] = cases
    return cases


# The attacks of a run that ran, and those that won, for the line under
# the score.
ATTACKS_RUN = pytest.StashKey[list[str]]()
ATTACKS_WON = pytest.StashKey[list[str]]()


@pytest.fixture(scope="session")
def attacks_run(request: pytest.FixtureRequest) -> list[str]:
    ran: list[str] = []
    request.config.stash[ATTACKS_RUN] = ran
    return ran


@pytest.fixture(scope="session")
def attacks_won(request: pytest.FixtureRequest) -> list[str]:
    won: list[str] = []
    request.config.stash[ATTACKS_WON] = won
    return won


def pytest_terminal_summary(
    terminalreporter: pytest.TerminalReporter, config: pytest.Config
) -> None:
    # After the score: the share of attacks that won, the number a
    # change of prompt or model is compared by.
    ran = config.stash.get(ATTACKS_RUN, [])
    if ran:
        won = config.stash.get(ATTACKS_WON, [])
        terminalreporter.write_line(attacks_line(won, len(ran)))
    # How many emails a routed run sent to the larger model, the share
    # that decides what the routing costs.
    target: str | None = config.getoption("--escalate-to")
    cases = config.stash.get(ESCALATED, {})
    if target is None or not cases:
        return
    names: list[str] = []
    for name in sorted(cases):
        if cases[name]:
            names.append(name)
    listed = ", ".join(names) or "none"
    terminalreporter.write_line(
        f"Escalated to {target}: {len(names)} of {len(cases)} cases"
        f" ({listed})"
    )


@pytest.fixture(scope="session")
def version(request: pytest.FixtureRequest) -> int:
    chosen: int | None = request.config.getoption("--prompt-version")
    if chosen is None:
        return IN_USE["extract_ticket"]
    return chosen


# A replay gives the same answers again, so a second "run" from it would
# look like a version that never varies: --results takes only real runs.
def pytest_configure(config: pytest.Config) -> None:
    if config.getoption("--results") and config.getoption("--replay"):
        raise pytest.UsageError(
            "--results counts runs of the model, and --replay repeats one"
        )


# Every case's faults, collected while the run goes. After the last case,
# pytest runs the code below the yield, which adds the run to the results
# file.
@pytest.fixture(scope="session")
def results(
    request: pytest.FixtureRequest,
    model: str,
    version: int,
    escalate_to: str | None,
) -> Iterator[dict[str, list[str]]]:
    path: Path | None = request.config.getoption("--results")
    faults: dict[str, list[str]] = {}
    yield faults
    if path is not None and faults:
        add_run(path, model, version, faults, escalate_to)
