from datetime import date
from pathlib import Path
from typing import Any

import httpx2
import pytest
from anthropic import Anthropic
from openai import OpenAI

from askkit.ask import MODELS
from askkit.audio import TRANSCRIBE_MODEL
from askkit.cli import main
from askkit.fallback import OPENAI_MODELS
from askkit.models import (
    DEFAULT_MODEL,
    LIFECYCLES,
    WARN_DAYS,
    Lifecycle,
    claude_lifecycle,
    lifecycle_change,
    openai_lifecycle,
    retirement_fault,
    retirement_warning,
    retires_text,
)
from askkit.routing import LARGE_MODEL, SMALL_MODEL

HAIKU = "claude-haiku-5-5"
SONNET = "claude-sonnet-5-5"
LUNA = "gpt-6-luna"
# Claude Sonnet 4.5's real schedule: announced on 30 September 2026,
# retired on 30 November 2026, Claude Sonnet 5.5 recommended instead.
SONNET_45 = Lifecycle("Anthropic", date(2026, 11, 30), True, SONNET)


def test_every_model_askkit_can_call_has_a_lifecycle() -> None:
    # A model with no row could be retired without AskKit ever saying.
    called = {*MODELS, *OPENAI_MODELS, TRANSCRIBE_MODEL, DEFAULT_MODEL}
    assert called <= set(LIFECYCLES)
    assert {SMALL_MODEL, LARGE_MODEL} <= set(LIFECYCLES)


def test_every_model_is_a_pinned_id_not_an_alias() -> None:
    # An alias such as gemini-flash-latest moves to each new release.
    for model in LIFECYCLES:
        assert not model.endswith("-latest")
        assert "preview" not in model


def test_the_default_is_the_model_every_command_starts_with() -> None:
    assert DEFAULT_MODEL == HAIKU
    assert SMALL_MODEL == DEFAULT_MODEL


def test_an_earliest_date_is_never_a_retirement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # "Not sooner than" is not a day requests fail, even once it passes.
    monkeypatch.setitem(
        LIFECYCLES,
        HAIKU,
        Lifecycle("Anthropic", date(2027, 10, 7), False, None),
    )
    long_after = date(2028, 1, 1)
    assert retirement_fault(HAIKU, long_after) is None
    assert retirement_warning(HAIKU, date(2027, 10, 1)) is None


def test_an_announced_day_warns_inside_the_notice_and_stops_on_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(LIFECYCLES, "claude-sonnet-4-5", SONNET_45)
    model = "claude-sonnet-4-5"
    assert retirement_warning(model, date(2026, 9, 30)) is None
    assert retirement_warning(model, date(2026, 10, 1)) == (
        "claude-sonnet-4-5 retires on 2026-11-30, in 60 days;"
        " move to claude-sonnet-5-5"
    )
    assert retirement_warning(model, date(2026, 11, 29)) == (
        "claude-sonnet-4-5 retires on 2026-11-30, in 1 day;"
        " move to claude-sonnet-5-5"
    )
    assert retirement_fault(model, date(2026, 11, 29)) is None
    assert retirement_fault(model, date(2026, 11, 30)) == (
        "claude-sonnet-4-5 was retired on 2026-11-30;"
        " move to claude-sonnet-5-5"
    )
    assert retirement_warning(model, date(2026, 11, 30)) is None
    assert WARN_DAYS == 60


def test_the_retirement_column_says_how_sure_each_date_is() -> None:
    assert retires_text(SONNET_45) == "2026-11-30"
    assert retires_text(LIFECYCLES[HAIKU]) == "not before 2027-10-07"
    assert retires_text(LIFECYCLES[LUNA]) == "no date announced"
    deprecated = Lifecycle("Anthropic", None, True, None)
    assert retires_text(deprecated) == "retiring, day not given yet"


def claude_models_client(info: dict[str, Any]) -> Anthropic:
    def reply(request: httpx2.Request) -> httpx2.Response:
        assert request.url.path == f"/v1/models/{info['id']}"
        return httpx2.Response(200, json=info)

    return Anthropic(
        api_key="sk-ant-api03-a-test-key-Ab12",
        max_retries=0,
        http_client=httpx2.Client(transport=httpx2.MockTransport(reply)),
    )


def claude_info(
    model: str, lifecycle: str, retires_at: str | None
) -> dict[str, Any]:
    return {
        "type": "model",
        "id": model,
        "display_name": "Claude",
        "created_at": "2026-10-07T00:00:00Z",
        "lifecycle": lifecycle,
        "deprecated_at": None if lifecycle == "active" else retires_at,
        "retires_at": retires_at,
    }


def openai_models_client(model: str, shutdown_date: str | None) -> OpenAI:
    def reply(request: httpx2.Request) -> httpx2.Response:
        assert request.url.path == f"/v1/models/{model}"
        return httpx2.Response(
            200,
            json={
                "id": model,
                "object": "model",
                "created": 1780000000,
                "owned_by": "openai",
                "shutdown_date": shutdown_date,
            },
        )

    return OpenAI(
        api_key="sk-proj-a-test-key-Cd34",
        max_retries=0,
        http_client=httpx2.Client(transport=httpx2.MockTransport(reply)),
    )


def test_an_active_claude_model_matches_its_earliest_date() -> None:
    client = claude_models_client(claude_info(HAIKU, "active", None))
    live = claude_lifecycle(client, HAIKU)
    assert live == Lifecycle("Anthropic", None, False, None)
    assert lifecycle_change(HAIKU, LIFECYCLES[HAIKU], live) is None


def test_a_deprecation_the_table_does_not_have_is_reported() -> None:
    info = claude_info(HAIKU, "deprecated", "2027-10-07T00:00:00Z")
    live = claude_lifecycle(claude_models_client(info), HAIKU)
    assert live == Lifecycle("Anthropic", date(2027, 10, 7), True, None)
    assert lifecycle_change(HAIKU, LIFECYCLES[HAIKU], live) == (
        "claude-haiku-5-5: Anthropic says 2027-10-07, and models.py says"
        " not before 2027-10-07; update its row"
    )


def test_openai_s_shutdown_date_is_read() -> None:
    live = openai_lifecycle(openai_models_client(LUNA, None), LUNA)
    assert lifecycle_change(LUNA, LIFECYCLES[LUNA], live) is None
    later = openai_lifecycle(openai_models_client(LUNA, "2027-06-01"), LUNA)
    assert later == Lifecycle("OpenAI", date(2027, 6, 1), True, None)
    assert lifecycle_change(LUNA, LIFECYCLES[LUNA], later) is not None


def test_models_prints_the_table(capsys: pytest.CaptureFixture[str]) -> None:
    main(["models"])
    out = capsys.readouterr().out
    assert out.splitlines() == [
        "Model               Provider   Retires",
        "claude-haiku-5-5    Anthropic  not before 2027-10-07",
        "claude-sonnet-5-5   Anthropic  not before 2027-09-28",
        "gpt-6-luna          OpenAI     no date announced",
        "gpt-transcribe      OpenAI     no date announced",
        "",
        "Default:       claude-haiku-5-5",
        "Checked:       2026-11-20, on each provider's deprecations page",
    ]


def test_models_warns_and_then_stops_on_the_day(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setitem(LIFECYCLES, "claude-sonnet-4-5", SONNET_45)
    monkeypatch.setattr("askkit.cli.today", lambda: date(2026, 11, 20))
    main(["models"])
    assert (
        "Warning:       claude-sonnet-4-5 retires on 2026-11-30, in 10 days;"
        " move to claude-sonnet-5-5"
    ) in capsys.readouterr().out
    monkeypatch.setattr("askkit.cli.today", lambda: date(2026, 11, 30))
    with pytest.raises(SystemExit) as stopped:
        main(["models"])
    assert stopped.value.code == (
        "askkit: claude-sonnet-4-5 was retired on 2026-11-30;"
        " move to claude-sonnet-5-5"
    )


def test_models_live_reports_a_row_to_update(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api03-a-test-key-Ab12")
    monkeypatch.setenv("OPENAI_API_KEY", "")
    answers = {
        HAIKU: claude_info(HAIKU, "active", None),
        SONNET: claude_info(SONNET, "deprecated", "2027-09-28T00:00:00Z"),
    }

    def reply(request: httpx2.Request) -> httpx2.Response:
        model = request.url.path.rsplit("/", 1)[-1]
        return httpx2.Response(200, json=answers[model])

    def make_claude(**options: Any) -> Anthropic:
        return Anthropic(
            api_key=options["api_key"],
            max_retries=0,
            http_client=httpx2.Client(transport=httpx2.MockTransport(reply)),
        )

    monkeypatch.setattr("askkit.cli.Anthropic", make_claude)
    with pytest.raises(SystemExit) as stopped:
        main(["models", "--live"])
    out = capsys.readouterr().out
    assert "Live:          claude-haiku-5-5, no date announced" in out
    assert "Not checked:   gpt-6-luna (OPENAI_API_KEY is not set)" in out
    assert stopped.value.code == (
        "askkit: claude-sonnet-5-5: Anthropic says 2027-09-28, and"
        " models.py says not before 2027-09-28; update its row"
    )


def test_a_chain_with_a_retired_model_stops_before_any_call(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # The last model of a chain too: a fallback to a retired model would
    # fail with a 404, which does not fall back.
    monkeypatch.setitem(
        LIFECYCLES,
        LUNA,
        Lifecycle("OpenAI", date(2027, 6, 1), True, "gpt-6.1-terra"),
    )
    monkeypatch.setattr("askkit.cli.today", lambda: date(2027, 6, 1))

    def no_calls(**options: Any) -> Anthropic:
        raise AssertionError("no client may be made")

    monkeypatch.setattr("askkit.cli.Anthropic", no_calls)
    with pytest.raises(SystemExit) as stopped:
        main(
            [
                "answer",
                str(tmp_path),
                "--out",
                str(tmp_path / "answers.json"),
                "--fallback",
                LUNA,
            ]
        )
    assert stopped.value.code == (
        "askkit: gpt-6-luna was retired on 2027-06-01; move to gpt-6.1-terra"
    )


def test_a_model_retiring_soon_is_named_and_the_command_runs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setitem(
        LIFECYCLES,
        HAIKU,
        Lifecycle("Anthropic", date(2027, 10, 7), True, "claude-haiku-6"),
    )
    monkeypatch.setattr("askkit.cli.today", lambda: date(2027, 9, 7))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")
    # The key check stops ask after the warning: no call is made.
    with pytest.raises(SystemExit):
        main(["ask", "Who pays postage?"])
    assert capsys.readouterr().err == (
        "askkit: warning: claude-haiku-5-5 retires on 2027-10-07, in 30"
        " days; move to claude-haiku-6\n"
    )
