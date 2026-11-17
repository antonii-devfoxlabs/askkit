import json
from pathlib import Path
from typing import Any

import httpx2

# A recording is a JSON list of exchanges, each the body of a request
# AskKit sent and the body of the answer the API gave it. Only bodies are
# kept: the headers carry the API key, and a recording goes into Git.
Exchange = dict[str, dict[str, Any]]


def load_exchanges(path: Path) -> list[Exchange]:
    if not path.is_file():
        return []
    exchanges: list[Exchange] = json.loads(path.read_text(encoding="utf-8"))
    return exchanges


def save_exchanges(path: Path, exchanges: list[Exchange]) -> None:
    text = json.dumps(exchanges, indent=2, ensure_ascii=False)
    path.write_text(text + "\n", encoding="utf-8")


def recording(
    path: Path, upstream: httpx2.BaseTransport | None = None
) -> httpx2.MockTransport:
    # Sends every request on to the API and saves each answer it gives,
    # in place of an older answer to the same request. An error is not
    # saved: a test played back from it would test the error, not the
    # answer.
    if upstream is None:
        upstream = httpx2.HTTPTransport()
    exchanges = load_exchanges(path)

    def handle(request: httpx2.Request) -> httpx2.Response:
        response = upstream.handle_request(request)
        response.read()
        if response.status_code == 200:
            sent = json.loads(request.content)
            for exchange in exchanges:
                if exchange["request"] == sent:
                    exchanges.remove(exchange)
                    break
            exchanges.append({"request": sent, "response": response.json()})
            save_exchanges(path, exchanges)
        return response

    return httpx2.MockTransport(handle)


def replaying(path: Path) -> httpx2.MockTransport:
    # Answers every request from the recording, and only a request that
    # is the same, field for field, as one that was recorded. Anything
    # else (another prompt, another email, another model) was never
    # answered, so it fails instead of getting an old answer.
    exchanges = load_exchanges(path)

    def handle(request: httpx2.Request) -> httpx2.Response:
        sent = json.loads(request.content)
        for exchange in exchanges:
            if exchange["request"] == sent:
                return httpx2.Response(200, json=exchange["response"])
        raise LookupError(
            f"{path.name} has no answer to this request; record it again"
            " with --record"
        )

    return httpx2.MockTransport(handle)
