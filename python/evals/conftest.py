from pathlib import Path

import httpx2
import pytest
from anthropic import Anthropic

from askkit.config import KEY_NAME, load_api_key
from askkit.recordings import recording, replaying

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


# One client for the whole run: pytest makes it before the first case and
# hands the same one to every case after. Without a key every case is
# skipped, not failed: a missing key says nothing about the prompt.
@pytest.fixture(scope="session")
def client(request: pytest.FixtureRequest) -> Anthropic:
    if request.config.getoption("--replay"):
        if not RECORDING.is_file():
            pytest.skip(f"no {RECORDING.name} yet: run once with --record")
        # No retries: a request the recording lacks fails at once.
        return Anthropic(
            api_key="replay",
            max_retries=0,
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
