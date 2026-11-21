import pytest

from askkit.keys import KEY_NAMES, known_keys, without_keys


@pytest.fixture(autouse=True)
def no_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in KEY_NAMES:
        monkeypatch.delenv(name, raising=False)


def test_every_key_askkit_takes_is_replaced(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api03-first")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-proj-second")
    monkeypatch.setenv("GEMINI_API_KEY", "AIza-third")
    text = "sk-ant-api03-first, sk-proj-second and AIza-third"
    assert without_keys(text) == "[KEY], [KEY] and [KEY]"


def test_a_key_is_replaced_wherever_it_stands(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api03-first")
    assert without_keys("key=sk-ant-api03-first;") == "key=[KEY];"


def test_a_key_set_to_nothing_changes_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # "".replace would put the placeholder between every two characters.
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("GEMINI_API_KEY", "   ")
    assert known_keys() == []
    assert without_keys("Where is order 4417?") == "Where is order 4417?"


def test_a_key_is_found_without_the_spaces_around_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", " sk-ant-api03-first\n")
    assert known_keys() == ["sk-ant-api03-first"]
