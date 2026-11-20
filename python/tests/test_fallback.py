import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import anthropic
import httpx2
import pytest
from anthropic import Anthropic
from anthropic.types import Message
from openai import OpenAI

from askkit.ask import ask
from askkit.cli import main
from askkit.cost import usage_cost
from askkit.fallback import (
    COOL_DOWN,
    FirstAnswer,
    Handover,
    ask_openai,
    fallbacks_line,
    falls_back,
    first_answer,
)
from askkit.retry import ANSWER_POLICY, Retry, RetryPolicy, with_retries
from askkit.rubric import load_answers

HAIKU = "claude-haiku-5-5"
SONNET = "claude-sonnet-5-5"
LUNA = "gpt-6-luna"
REQUEST_ID = "req_011CAskKitFallback"
MAIN = Path(__file__).parent.parent / "evals" / "ask" / "main"


def error_body(error_type: str, message: str) -> dict[str, Any]:
    return {
        "type": "error",
        "error": {"type": error_type, "message": message},
        "request_id": REQUEST_ID,
    }


OVERLOADED = error_body("overloaded_error", "Overloaded")
BAD_REQUEST = error_body("invalid_request_error", "max_tokens: too large")


def claude_body(
    model: str,
    text: str = "<answer>We do.</answer>",
    stop_reason: str = "end_turn",
    category: str | None = None,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "id": "msg_01AskKitFallback",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [{"type": "text", "text": text}],
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {"input_tokens": 410, "output_tokens": 52},
    }
    if stop_reason == "refusal":
        body["content"] = []
        body["stop_details"] = {
            "type": "refusal",
            "category": category,
            "explanation": None,
        }
    return body


def luna_body(
    text: str = "<answer>We do.</answer>",
    status: str = "completed",
    reason: str | None = None,
    refusal: str | None = None,
) -> dict[str, Any]:
    parts: list[dict[str, Any]] = [
        {"type": "output_text", "text": text, "annotations": []}
    ]
    if refusal is not None:
        parts = [{"type": "refusal", "refusal": refusal}]
    details = None
    if reason is not None:
        details = {"reason": reason}
    return {
        "id": "resp_01AskKitFallback",
        "object": "response",
        "created_at": 1795000000,
        "model": LUNA,
        "status": status,
        "incomplete_details": details,
        "output": [
            {
                "type": "message",
                "id": "msg_01Luna",
                "role": "assistant",
                "status": "completed",
                "content": parts,
            }
        ],
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
        "usage": {
            "input_tokens": 1200,
            "input_tokens_details": {
                "cached_tokens": 1024,
                "cache_write_tokens": 0,
            },
            "output_tokens": 60,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": 1260,
        },
    }


Step = tuple[int, dict[str, Any]]


def claude_client(steps: list[Step], sent: list[str]) -> Anthropic:
    # Each request gets the next step; the last one repeats. sent keeps
    # the model each request named.
    def reply(request: httpx2.Request) -> httpx2.Response:
        sent.append(json.loads(request.content)["model"])
        status, body = steps[min(len(sent), len(steps)) - 1]
        return httpx2.Response(
            status, json=body, headers={"request-id": REQUEST_ID}
        )

    return Anthropic(
        api_key="sk-ant-api03-a-test-key-Ab12",
        max_retries=0,
        http_client=httpx2.Client(transport=httpx2.MockTransport(reply)),
    )


def openai_client(
    bodies: list[dict[str, Any]], sent: list[dict[str, Any]]
) -> OpenAI:
    def reply(request: httpx2.Request) -> httpx2.Response:
        sent.append(json.loads(request.content))
        body = bodies[min(len(sent), len(bodies)) - 1]
        return httpx2.Response(200, json=body)

    return OpenAI(
        api_key="sk-proj-a-test-key-Cd34",
        max_retries=0,
        http_client=httpx2.Client(transport=httpx2.MockTransport(reply)),
    )


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def by_model(
    clients: dict[str, Anthropic],
) -> Callable[[str], Message]:
    # One try per model: these tests are about the chain, not retries.
    def send(model: str) -> Message:
        return ask(clients[model], "Who pays?", model, 1000, "Be brief.")

    return send


def test_an_overloaded_model_hands_the_question_to_the_next() -> None:
    sent: list[str] = []
    clients = {
        HAIKU: claude_client([(529, OVERLOADED)], sent),
        SONNET: claude_client([(200, claude_body(SONNET))], sent),
    }
    down: dict[str, float] = {}
    clock = FakeClock()
    outcome = first_answer([HAIKU, SONNET], by_model(clients), down, clock)
    assert outcome.model == SONNET
    assert outcome.handovers == [Handover(HAIKU, "529")]
    assert sent == [HAIKU, SONNET]
    # Haiku is passed over until the cool-down ends.
    assert down == {HAIKU: COOL_DOWN}


def test_a_rejected_request_is_not_handed_on() -> None:
    # A 400 fails the same way on every model: falling back would hide it.
    sent: list[str] = []
    clients = {
        HAIKU: claude_client([(400, BAD_REQUEST)], sent),
        SONNET: claude_client([(200, claude_body(SONNET))], sent),
    }
    with pytest.raises(anthropic.BadRequestError):
        first_answer([HAIKU, SONNET], by_model(clients), {}, FakeClock())
    assert sent == [HAIKU]


def test_the_last_model_s_error_is_raised_as_it_is() -> None:
    sent: list[str] = []
    clients = {
        HAIKU: claude_client([(529, OVERLOADED)], sent),
        SONNET: claude_client([(529, OVERLOADED)], sent),
    }
    with pytest.raises(anthropic.OverloadedError):
        first_answer([HAIKU, SONNET], by_model(clients), {}, FakeClock())
    assert sent == [HAIKU, SONNET]


def test_a_failed_model_is_passed_over_while_it_cools_down() -> None:
    sent: list[str] = []
    clients = {
        HAIKU: claude_client(
            [(529, OVERLOADED), (200, claude_body(HAIKU))], sent
        ),
        SONNET: claude_client([(200, claude_body(SONNET))], sent),
    }
    down: dict[str, float] = {}
    clock = FakeClock()
    send = by_model(clients)
    first_answer([HAIKU, SONNET], send, down, clock)
    clock.now = COOL_DOWN - 1
    second = first_answer([HAIKU, SONNET], send, down, clock)
    assert second.handovers == [Handover(HAIKU, "cooling down")]
    assert second.model == SONNET
    # Once the cool-down is over, Haiku is asked first again.
    clock.now = COOL_DOWN
    third = first_answer([HAIKU, SONNET], send, down, clock)
    assert third.model == HAIKU
    assert sent == [HAIKU, SONNET, SONNET, HAIKU]


def test_the_last_model_is_asked_even_while_it_cools_down() -> None:
    sent: list[str] = []
    clients = {HAIKU: claude_client([(200, claude_body(HAIKU))], sent)}
    outcome = first_answer(
        [HAIKU], by_model(clients), {HAIKU: COOL_DOWN}, FakeClock()
    )
    assert outcome.model == HAIKU


def test_a_declined_answer_goes_on_to_another_claude_model() -> None:
    sent: list[str] = []
    declined = claude_body(HAIKU, stop_reason="refusal", category="cyber")
    clients = {
        HAIKU: claude_client([(200, declined)], sent),
        SONNET: claude_client([(200, claude_body(SONNET))], sent),
    }
    down: dict[str, float] = {}
    outcome = first_answer(
        [HAIKU, SONNET], by_model(clients), down, FakeClock()
    )
    assert outcome.model == SONNET
    assert outcome.handovers == [Handover(HAIKU, "refusal")]
    # The declined call may be billed, so it is kept to be priced.
    assert [model for _, model in outcome.declined] == [HAIKU]
    # A decline is about this question: Haiku stays in the chain.
    assert down == {}


def test_a_declined_answer_is_not_sent_to_another_provider() -> None:
    sent: list[str] = []
    declined = claude_body(HAIKU, stop_reason="refusal", category="cyber")
    clients = {HAIKU: claude_client([(200, declined)], sent)}

    def send(model: str) -> Message:
        if model == LUNA:
            raise AssertionError("the declined question reached OpenAI")
        return ask(clients[model], "Who pays?", model, 1000, "Be brief.")

    outcome = first_answer([HAIKU, LUNA], send, {}, FakeClock())
    assert outcome.model == HAIKU
    assert outcome.response.stop_reason == "refusal"


def test_reasoning_extraction_is_not_handed_on() -> None:
    # Anthropic: no recommended fallback model; change the prompt instead.
    sent: list[str] = []
    declined = claude_body(
        HAIKU, stop_reason="refusal", category="reasoning_extraction"
    )
    clients = {
        HAIKU: claude_client([(200, declined)], sent),
        SONNET: claude_client([(200, claude_body(SONNET))], sent),
    }
    outcome = first_answer(
        [HAIKU, SONNET], by_model(clients), {}, FakeClock()
    )
    assert outcome.model == HAIKU
    assert sent == [HAIKU]


def test_what_falls_back_at_openai() -> None:
    request = httpx2.Request("POST", "https://api.openai.com/v1/responses")

    def status_error(status: int) -> Exception:
        response = httpx2.Response(status, request=request)
        client = openai_client([], [])
        return client._make_status_error("busy", body=None, response=response)

    assert falls_back(status_error(503))  # type: ignore[arg-type]
    assert falls_back(status_error(429))  # type: ignore[arg-type]
    assert not falls_back(status_error(401))  # type: ignore[arg-type]


def test_openai_s_answer_is_read_in_claude_s_shape() -> None:
    sent: list[dict[str, Any]] = []
    client = openai_client([luna_body()], sent)
    message = ask_openai(client, "Who pays?", LUNA, 1000, "Be brief.")
    assert message.stop_reason == "end_turn"
    assert message.content[0].type == "text"
    assert message.content[0].text == "<answer>We do.</answer>"
    # OpenAI's 1,200 input tokens include the 1,024 read from its cache.
    assert message.usage.input_tokens == 176
    assert message.usage.cache_read_input_tokens == 1024
    assert message.usage.cache_creation_input_tokens == 0
    assert message.usage.output_tokens == 60
    # The same question: the system prompt as instructions, no copy kept.
    assert sent[0]["instructions"] == "Be brief."
    assert sent[0]["input"] == "Who pays?"
    assert sent[0]["max_output_tokens"] == 1000
    assert sent[0]["store"] is False


def test_openai_s_answer_is_priced_at_its_own_prices() -> None:
    message = ask_openai(
        openai_client([luna_body()], []), "Who pays?", LUNA, 1000, "x"
    )
    # 176 plain at $0.10, 1,024 read at $0.01, 60 out at $0.50 a million.
    expected = (176 * 0.10 + 1024 * 0.01 + 60 * 0.50) / 1_000_000
    assert usage_cost(LUNA, message.usage) == pytest.approx(expected)


def test_openai_s_cut_off_and_declined_answers_keep_their_meaning() -> None:
    cut = luna_body(status="incomplete", reason="max_output_tokens")
    filtered = luna_body(status="incomplete", reason="content_filter")
    declined = luna_body(refusal="I can't help with that.")
    stops = []
    for body in (cut, filtered, declined):
        client = openai_client([body], [])
        stops.append(ask_openai(client, "Q", LUNA, 1000, "x").stop_reason)
    assert stops == ["max_tokens", "refusal", "refusal"]
    message = ask_openai(openai_client([declined], []), "Q", LUNA, 1000, "x")
    # Of a declined answer no part is kept.
    assert message.content == []


def test_an_openai_answer_with_no_reading_is_refused() -> None:
    body = luna_body(status="failed")
    client = openai_client([body], [])
    with pytest.raises(ValueError) as raised:
        ask_openai(client, "Q", LUNA, 1000, "x")
    assert str(raised.value) == (
        "gpt-6-luna stopped as failed, which AskKit cannot read as an answer"
    )


def test_the_fallbacks_line_names_each_question_and_why() -> None:
    def one(handovers: list[Handover]) -> FirstAnswer:
        response = Message.model_validate(claude_body(LUNA))
        return FirstAnswer(response, LUNA, handovers, [])

    line = fallbacks_line(
        [
            ("postage", one([])),
            ("gift", one([Handover(HAIKU, "529")])),
            ("canada", one([Handover(HAIKU, "cooling down")])),
        ]
    )
    assert line == "Fallbacks:     2 (gift: 529; canada: cooling down)"
    assert fallbacks_line([("postage", one([]))]) is None


@pytest.fixture
def ask_set(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api03-a-test-key-Ab12")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-proj-a-test-key-Cd34")
    part = tmp_path / "main"
    part.mkdir()
    cases = json.loads((MAIN / "cases.json").read_text(encoding="utf-8"))
    (part / "cases.json").write_text(json.dumps(cases[:2]), encoding="utf-8")
    documents = tmp_path / "documents"
    documents.mkdir()
    for path in (MAIN.parent / "documents").iterdir():
        (documents / path.name).write_bytes(path.read_bytes())
    return part


def run_with(
    monkeypatch: pytest.MonkeyPatch,
    claude: list[Step],
    luna: list[dict[str, Any]],
    claude_sent: list[str],
    luna_sent: list[dict[str, Any]],
) -> None:
    def make_claude(**options: Any) -> Anthropic:
        return claude_client(claude, claude_sent)

    def make_openai(**options: Any) -> OpenAI:
        return openai_client(luna, luna_sent)

    def no_waiting(
        send: Callable[[], Message], policy: RetryPolicy, retries: list[Retry]
    ) -> Message:
        return with_retries(
            send, policy, retries, lambda seconds: None, lambda: 0.0
        )

    monkeypatch.setattr("askkit.cli.Anthropic", make_claude)
    monkeypatch.setattr("askkit.cli.OpenAI", make_openai)
    monkeypatch.setattr("askkit.cli.with_retries", no_waiting)


def test_answer_falls_back_to_openai_and_stays_there_while_haiku_cools(
    ask_set: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    claude_sent: list[str] = []
    luna_sent: list[dict[str, Any]] = []
    run_with(
        monkeypatch,
        [(529, OVERLOADED)],
        [luna_body()],
        claude_sent,
        luna_sent,
    )
    main(
        [
            "answer",
            str(ask_set),
            "--out",
            "answers.json",
            "--fallback",
            LUNA,
        ]
    )
    lines = capsys.readouterr().out.splitlines()
    # Haiku used its five tries on the first question only; the second
    # went straight to GPT-6 Luna.
    assert claude_sent == [HAIKU] * ANSWER_POLICY.tries
    assert len(luna_sent) == 2
    assert lines[0].endswith("(gpt-6-luna)")
    assert lines[1].endswith("(gpt-6-luna)")
    assert "Model:         gpt-6-luna" in lines
    assert "Tokens:        2,400 in, 120 out" in lines
    assert "Cache:         2,048 read, 0 written" in lines
    assert "Cost:          $0.000116" in lines
    assert any(line.startswith("Retries:       4 (529") for line in lines)
    assert (
        f"Fallbacks:     2 ({ask_set_cases(ask_set)[0]}: 529;"
        f" {ask_set_cases(ask_set)[1]}: cooling down)"
    ) in lines
    answers = load_answers(Path("answers.json"))
    assert [answer.model for answer in answers] == [LUNA, LUNA]


def ask_set_cases(part: Path) -> list[str]:
    cases = json.loads((part / "cases.json").read_text(encoding="utf-8"))
    return [case["name"] for case in cases]


def test_answer_without_a_fallback_saves_haiku_as_the_model(
    ask_set: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    claude_sent: list[str] = []
    run_with(monkeypatch, [(200, claude_body(HAIKU))], [], claude_sent, [])
    main(["answer", str(ask_set), "--out", "answers.json"])
    answers = load_answers(Path("answers.json"))
    assert [answer.model for answer in answers] == [HAIKU, HAIKU]


def test_answer_needs_the_openai_key_before_any_call(
    ask_set: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "")
    claude_sent: list[str] = []
    run_with(monkeypatch, [(529, OVERLOADED)], [], claude_sent, [])
    with pytest.raises(SystemExit) as stopped:
        main(["answer", str(ask_set), "--out", "a.json", "--fallback", LUNA])
    assert stopped.value.code == (
        "askkit: OPENAI_API_KEY is not set (a fallback in the chain runs at"
        " OpenAI); put your OpenAI key after the = in .env"
    )
    assert claude_sent == []


def test_answer_refuses_a_model_twice_in_the_chain(
    ask_set: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(SystemExit) as stopped:
        main(["answer", str(ask_set), "--out", "a.json", "--fallback", HAIKU])
    assert stopped.value.code == (
        "askkit: --fallback claude-haiku-5-5 is already in the chain;"
        " fall back to another model"
    )


def test_answer_s_last_error_still_names_the_question(
    ask_set: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    claude_sent: list[str] = []
    run_with(monkeypatch, [(529, OVERLOADED)], [], claude_sent, [])
    with pytest.raises(SystemExit) as stopped:
        main(
            [
                "answer",
                str(ask_set),
                "--out",
                "a.json",
                "--fallback",
                SONNET,
                "--tries",
                "1",
            ]
        )
    first = ask_set_cases(ask_set)[0]
    assert stopped.value.code == (
        f"askkit: {first}: Anthropic's servers are overloaded (529) after 1"
        " try;\ntry again in a few minutes"
        f" (request ID {REQUEST_ID})"
    )
    assert claude_sent == [HAIKU, SONNET]


def test_answer_counts_only_the_last_model_s_tries(
    ask_set: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    claude_sent: list[str] = []
    run_with(monkeypatch, [(529, OVERLOADED)], [], claude_sent, [])
    with pytest.raises(SystemExit) as stopped:
        main(
            ["answer", str(ask_set), "--out", "a.json", "--fallback", SONNET]
        )
    # Five tries on Haiku, then five on Sonnet: the message names five.
    assert len(claude_sent) == 2 * ANSWER_POLICY.tries
    assert "overloaded (529) after 5 tries;" in str(stopped.value.code)


def test_answer_can_ask_the_fallback_model_first_to_grade_it(
    ask_set: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    claude_sent: list[str] = []
    luna_sent: list[dict[str, Any]] = []
    run_with(monkeypatch, [], [luna_body()], claude_sent, luna_sent)
    main(["answer", str(ask_set), "--out", "luna.json", "-m", LUNA])
    answers = load_answers(Path("luna.json"))
    assert [answer.model for answer in answers] == [LUNA, LUNA]
    assert claude_sent == []
