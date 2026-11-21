import json
from typing import Any

import httpx
import httpx2
import pytest
from any_llm import AnyLLM
from any_llm.exceptions import AnyLLMError
from google.genai import types

from askkit.cost import usage_cost
from askkit.fallback import failure, falls_back
from askkit.layer import (
    LAYER_MODELS,
    LAYER_PROVIDERS,
    LAYER_RETRIES,
    Layer,
    ask_layer,
    layer_client,
    layer_error,
)
from askkit.models import LIFECYCLES
from askkit.prices import CACHE_PRICES, PRICES

LUNA = "gpt-6-luna"
OPENAI_ID = "req_AskKitLayerOpenAI"
FLASH_LITE = "gemini-3.5-flash-lite"

Reply = tuple[int, dict[str, Any]]

# The layer's own create(), kept before any test replaces it.
REAL_CREATE = AnyLLM.create


def luna_chat_body(
    finish: str = "stop",
    content: str | None = "<answer>We do.</answer>",
    refusal: str | None = None,
) -> dict[str, Any]:
    # A Chat Completions answer, as OpenAI sends it to the layer: its
    # prompt_tokens count the whole prompt, the cache read included.
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if refusal is not None:
        message["refusal"] = refusal
    return {
        "id": "chatcmpl-AskKit",
        "object": "chat.completion",
        "created": 1_795_000_000,
        "model": LUNA,
        "choices": [
            {"index": 0, "finish_reason": finish, "message": message}
        ],
        "usage": {
            "prompt_tokens": 1200,
            "completion_tokens": 60,
            "total_tokens": 1260,
            "prompt_tokens_details": {
                "cached_tokens": 1024,
                "cache_write_tokens": 0,
            },
        },
    }


def flash_lite_body() -> dict[str, Any]:
    # A generateContent answer, as Google sends it: promptTokenCount, like
    # OpenAI's count, includes the tokens read from its cache.
    return {
        "candidates": [
            {
                "content": {
                    "role": "model",
                    "parts": [{"text": "<answer>We do.</answer>"}],
                },
                "finishReason": "STOP",
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 1200,
            "cachedContentTokenCount": 1024,
            "candidatesTokenCount": 60,
            "totalTokenCount": 1260,
        },
        "modelVersion": FLASH_LITE,
        "responseId": "AskKit-gemini",
    }


def error_reply(status: int) -> Reply:
    return status, {"error": {"message": "test", "type": "server_error"}}


class StandIn:
    # Answers each request with the next reply, the last one repeating,
    # and keeps every request it got.
    def __init__(self, replies: list[Reply]) -> None:
        self.replies = replies
        self.requests: list[tuple[str, dict[str, Any]]] = []

    def answer(self, path: str, content: bytes) -> Reply:
        self.requests.append((path, json.loads(content)))
        return self.replies[min(len(self.requests), len(self.replies)) - 1]


def use_stand_in(monkeypatch: pytest.MonkeyPatch, stand_in: StandIn) -> None:
    # layer_client builds its client with the layer's real create(), with
    # the arguments AskKit gives it; only the network under the provider's
    # SDK is the stand-in. OpenAI's SDK runs on httpx2, Google's on httpx.

    def openai_reply(request: httpx2.Request) -> httpx2.Response:
        status, body = stand_in.answer(request.url.path, request.content)
        headers = {"retry-after-ms": "0", "x-request-id": OPENAI_ID}
        return httpx2.Response(status, json=body, headers=headers)

    def google_reply(request: httpx.Request) -> httpx.Response:
        status, body = stand_in.answer(request.url.path, request.content)
        return httpx.Response(status, json=body)

    def create(provider: str, **options: Any) -> AnyLLM:
        if provider == "openai":
            # The client layer_client made, with its response hook, on the
            # stand-in's transport.
            made = options["http_client"]
            options["http_client"] = httpx2.AsyncClient(
                transport=httpx2.MockTransport(openai_reply),
                event_hooks=made.event_hooks,
            )
        else:
            # No wait between Google's tries: these tests count them.
            options["http_options"] = types.HttpOptions(
                httpx_async_client=httpx.AsyncClient(
                    transport=httpx.MockTransport(google_reply)
                ),
                retry_options=types.HttpRetryOptions(
                    attempts=options["max_retries"] + 1,
                    initial_delay=0.0,
                    max_delay=0.0,
                ),
            )
        return REAL_CREATE(provider, **options)

    monkeypatch.setattr("askkit.layer.AnyLLM.create", create)


def asked(model: str, replies: list[Reply], monkeypatch: Any) -> Any:
    stand_in = StandIn(replies)
    use_stand_in(monkeypatch, stand_in)
    llm = layer_client(model, "a-test-key-Ef56")
    message = ask_layer(llm, "Who pays?", model, 1000, "Be brief.", 30.0)
    return message, stand_in


def test_every_layer_model_has_its_rows() -> None:
    for model in LAYER_MODELS:
        assert LIFECYCLES[model].provider in LAYER_PROVIDERS
        assert model in PRICES
        assert model in CACHE_PRICES


def test_openai_s_answer_comes_back_in_claude_s_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    message, stand_in = asked(LUNA, [(200, luna_chat_body())], monkeypatch)
    assert message.stop_reason == "end_turn"
    assert message.content[0].type == "text"
    assert message.content[0].text == "<answer>We do.</answer>"
    # OpenAI's 1,200 prompt tokens include the 1,024 read from its cache;
    # in Claude's shape input_tokens counts only the other 176.
    assert message.usage.input_tokens == 176
    assert message.usage.cache_read_input_tokens == 1024
    assert message.usage.output_tokens == 60
    # The layer's Message has no request ID; AskKit reads OpenAI's header.
    assert message._request_id == OPENAI_ID
    path, sent = stand_in.requests[0]
    assert path == "/v1/chat/completions"
    assert sent["model"] == LUNA
    assert sent["messages"][0] == {"role": "system", "content": "Be brief."}
    assert sent["messages"][1]["role"] == "user"


def test_google_s_answer_comes_back_in_the_same_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    message, stand_in = asked(
        FLASH_LITE, [(200, flash_lite_body())], monkeypatch
    )
    assert message.stop_reason == "end_turn"
    assert message.content[0].text == "<answer>We do.</answer>"
    assert message.usage.input_tokens == 176
    assert message.usage.cache_read_input_tokens == 1024
    path, _ = stand_in.requests[0]
    assert path.endswith(f"/models/{FLASH_LITE}:generateContent")


def test_a_layer_answer_is_priced_at_its_own_model_s_prices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    luna, _ = asked(LUNA, [(200, luna_chat_body())], monkeypatch)
    flash, _ = asked(FLASH_LITE, [(200, flash_lite_body())], monkeypatch)
    # 176 plain, 1,024 read from the cache, 60 out, a million at a time.
    assert usage_cost(LUNA, luna.usage) == pytest.approx(
        (176 * 0.10 + 1024 * 0.01 + 60 * 0.50) / 1_000_000
    )
    assert usage_cost(FLASH_LITE, flash.usage) == pytest.approx(
        (176 * 0.30 + 1024 * 0.03 + 60 * 2.50) / 1_000_000
    )


def test_cut_off_and_declined_answers_keep_their_meaning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bodies = [
        luna_chat_body(finish="length", content="<answer>We"),
        luna_chat_body(finish="content_filter", content=None),
        luna_chat_body(content=None, refusal="I can't help with that."),
    ]
    stops = []
    for body in bodies:
        message, _ = asked(LUNA, [(200, body)], monkeypatch)
        stops.append(message.stop_reason)
    assert stops == ["max_tokens", "refusal", "refusal"]


def test_google_gets_the_same_retries_as_openai(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Google's SDK makes no retries unless told to (lesson 13.1), and the
    # layer keeps that default when it is given no number. layer_client
    # gives it LAYER_RETRIES, so two 503s are tried again at both.
    for model, body in ((LUNA, luna_chat_body()), (FLASH_LITE, None)):
        answer = flash_lite_body() if body is None else body
        replies = [error_reply(503), error_reply(503), (200, answer)]
        message, stand_in = asked(model, replies, monkeypatch)
        assert message.stop_reason == "end_turn"
        assert len(stand_in.requests) == LAYER_RETRIES + 1


def layer_failure_of(status: int, monkeypatch: Any) -> AnyLLMError:
    with pytest.raises(AnyLLMError) as raised:
        asked(LUNA, [error_reply(status)], monkeypatch)
    return raised.value


def test_what_falls_back_through_the_layer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    busy = layer_failure_of(503, monkeypatch)
    limited = layer_failure_of(429, monkeypatch)
    rejected = layer_failure_of(401, monkeypatch)
    assert falls_back(busy)
    assert falls_back(limited)
    assert not falls_back(rejected)
    assert failure(busy) == "503"


def test_no_answer_falls_back_and_reads_as_no_answer() -> None:
    llm = AnyLLM.create(
        "openai",
        api_key="a-test-key-Ef56",
        api_base="http://127.0.0.1:9/v1",
        unified_exceptions=True,
        max_retries=0,
    )
    with pytest.raises(AnyLLMError) as raised:
        ask_layer(Layer(llm, []), "Who pays?", LUNA, 1000, "x", 5.0)
    assert raised.value.status_code is None
    assert falls_back(raised.value)
    assert failure(raised.value) == "no answer"


def test_layer_errors_name_the_provider_and_its_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert layer_error(layer_failure_of(401, monkeypatch)) == (
        "OpenAI rejected the key; check OPENAI_API_KEY in .env"
    )
    assert layer_error(layer_failure_of(400, monkeypatch)) == (
        "OpenAI rejected the request (400)"
    )
    assert layer_error(layer_failure_of(503, monkeypatch)) == (
        "OpenAI's servers failed (503) after its retries"
    )
