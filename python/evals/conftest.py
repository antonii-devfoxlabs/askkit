from collections.abc import Iterator
from pathlib import Path

import httpx2
import pytest
from anthropic import Anthropic

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
    request: pytest.FixtureRequest, model: str, version: int
) -> Iterator[dict[str, list[str]]]:
    path: Path | None = request.config.getoption("--results")
    faults: dict[str, list[str]] = {}
    yield faults
    if path is not None and faults:
        add_run(path, model, version, faults)
