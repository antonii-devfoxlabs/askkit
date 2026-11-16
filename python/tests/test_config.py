import os
from collections.abc import Iterator
from pathlib import Path

import pytest

from askkit.config import KEY_NAME, load_api_key, masked

KEY_IN_FILE = "sk-ant-api03-from-the-env-file-0001"
KEY_IN_SHELL = "sk-ant-api03-from-the-shell-00002"


@pytest.fixture(autouse=True)
def no_key_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[None]:
    # load_dotenv writes into os.environ itself, so clean up after it.
    saved = os.environ.pop(KEY_NAME, None)
    monkeypatch.chdir(tmp_path)
    yield
    os.environ.pop(KEY_NAME, None)
    if saved is not None:
        os.environ[KEY_NAME] = saved


def test_key_comes_from_the_env_file(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text(
        f"# AskKit's settings\n{KEY_NAME}={KEY_IN_FILE}\n", encoding="utf-8"
    )
    assert load_api_key() == KEY_IN_FILE


def test_env_file_in_a_folder_above_is_found(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text(
        f"{KEY_NAME}={KEY_IN_FILE}\n", encoding="utf-8"
    )
    below = tmp_path / "src"
    below.mkdir()
    os.chdir(below)
    assert load_api_key() == KEY_IN_FILE


def test_key_in_the_environment_wins(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text(
        f"{KEY_NAME}={KEY_IN_FILE}\n", encoding="utf-8"
    )
    os.environ[KEY_NAME] = KEY_IN_SHELL
    assert load_api_key() == KEY_IN_SHELL


def test_no_key_anywhere_gives_none() -> None:
    assert load_api_key() is None


def test_an_empty_key_gives_none(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text(f"{KEY_NAME}=\n", encoding="utf-8")
    assert load_api_key() is None


def test_masked_shows_only_both_ends() -> None:
    assert masked(KEY_IN_FILE) == "sk-ant...0001"


def test_masked_hides_a_short_key_entirely() -> None:
    assert masked("sk-ant-12") == "*********"
