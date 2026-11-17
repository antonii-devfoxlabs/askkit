import pytest
from anthropic import Anthropic

from askkit.config import KEY_NAME, load_api_key


# One client for the whole run: pytest makes it before the first case and
# hands the same one to every case after. Without a key every case is
# skipped, not failed: a missing key says nothing about the prompt.
@pytest.fixture(scope="session")
def client() -> Anthropic:
    key = load_api_key()
    if key is None:
        pytest.skip(f"{KEY_NAME} is not set: the eval calls the model")
    return Anthropic(api_key=key)
