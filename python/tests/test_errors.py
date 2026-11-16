from typing import Any

import httpx2
import pytest
from anthropic import Anthropic, APIError

from askkit.cli import main
from askkit.errors import error_message

REQUEST_ID = "req_011CAskKitTest"


def error_body(error_type: str, message: str) -> dict[str, Any]:
    return {
        "type": "error",
        "error": {"type": error_type, "message": message},
        "request_id": REQUEST_ID,
    }


def failing_client(
    status: int, body: dict[str, Any], calls: list[int]
) -> Anthropic:
    def answer(request: httpx2.Request) -> httpx2.Response:
        calls.append(status)
        return httpx2.Response(
            status, json=body, headers={"request-id": REQUEST_ID}
        )

    # No retries here, so the tests do not wait; AskKit itself keeps
    # the SDK's default of two.
    return Anthropic(
        api_key="sk-ant-api03-a-test-key-Ab12",
        max_retries=0,
        http_client=httpx2.Client(transport=httpx2.MockTransport(answer)),
    )


def unreachable_client(failure: type[httpx2.TransportError]) -> Anthropic:
    def answer(request: httpx2.Request) -> httpx2.Response:
        raise failure("no route", request=request)

    return Anthropic(
        api_key="sk-ant-api03-a-test-key-Ab12",
        max_retries=0,
        http_client=httpx2.Client(transport=httpx2.MockTransport(answer)),
    )


def raised(client: Anthropic) -> APIError:
    with pytest.raises(APIError) as failure:
        client.messages.create(
            model="claude-haiku-5-5",
            max_tokens=1000,
            messages=[{"role": "user", "content": "Hi"}],
        )
    return failure.value


@pytest.mark.parametrize(
    ("status", "error_type", "message"),
    [
        (
            401,
            "authentication_error",
            "the API key was rejected (401); put a working key in .env",
        ),
        (
            403,
            "permission_error",
            "this key may not make this request (403); check its workspace",
        ),
        (
            404,
            "not_found_error",
            "the model was not found (404); it may be retired or not yours",
        ),
        (
            413,
            "request_too_large",
            "the request is too large (413); send less text",
        ),
        (
            429,
            "rate_limit_error",
            "the rate limit was reached (429) after the SDK's retries;"
            "\nwait a minute and try again",
        ),
        (
            529,
            "overloaded_error",
            "Anthropic's servers are overloaded (529) after the SDK's"
            " retries;\ntry again in a few minutes"
            f" (request ID {REQUEST_ID})",
        ),
        (
            500,
            "api_error",
            "Anthropic's side failed (500) after the SDK's retries;"
            f"\ntry again later (request ID {REQUEST_ID})",
        ),
    ],
)
def test_each_error_says_what_to_do(
    status: int, error_type: str, message: str
) -> None:
    body = error_body(error_type, "From the API.")
    error = raised(failing_client(status, body, []))
    assert error_message(error) == message


def test_a_bad_request_shows_the_api_message() -> None:
    body = error_body("invalid_request_error", "max_tokens: Field required")
    error = raised(failing_client(400, body, []))
    assert error_message(error) == (
        "the API rejected the request (400): max_tokens: Field required"
    )


def test_an_unknown_status_shows_its_code_type_and_message() -> None:
    body = error_body("billing_error", "Check your payment details.")
    error = raised(failing_client(402, body, []))
    assert error_message(error) == (
        "the API answered 402 (billing_error): Check your payment details."
    )


def test_no_connection_says_so() -> None:
    error = raised(unreachable_client(httpx2.ConnectError))
    assert error_message(error) == (
        "could not reach the API after the SDK's retries;"
        "\ncheck your connection"
    )


def test_a_timeout_says_so() -> None:
    error = raised(unreachable_client(httpx2.ReadTimeout))
    assert error_message(error) == (
        "no answer in time after the SDK's retries; try again"
    )


def test_ask_stops_with_the_message_instead_of_a_traceback(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api03-a-test-key-Ab12")
    calls: list[int] = []
    options_seen: list[dict[str, Any]] = []

    def make(**options: Any) -> Anthropic:
        options_seen.append(options)
        body = error_body("authentication_error", "invalid x-api-key")
        return failing_client(401, body, calls)

    monkeypatch.setattr("askkit.cli.Anthropic", make)
    with pytest.raises(SystemExit) as stopped:
        main(["ask", "Hi"])
    assert stopped.value.code == (
        "askkit: the API key was rejected (401); put a working key in .env"
    )
    assert calls == [401]
    assert capsys.readouterr().out == ""
    # AskKit leaves the SDK's retries as they are.
    assert "max_retries" not in options_seen[0]
