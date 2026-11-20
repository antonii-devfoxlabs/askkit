import json
import sys
from time import perf_counter
from typing import Any

from anthropic import Anthropic
from anthropic.types import Message, MessageParam

from askkit.ask import Timing
from askkit.calllog import log_call


def settled_fields(text: str) -> dict[str, Any]:
    # The fields of a JSON object that the text has already moved past: a
    # comma or the closing brace comes after each one, so its value will
    # not change. The field still being written is left out, since its
    # value may still grow ("44" on the way to "4417").
    settled: dict[str, Any] = {}
    for end, character in enumerate(text):
        if character not in ",}":
            continue
        try:
            fields = json.loads(text[:end] + "}")
        except json.JSONDecodeError:
            # A comma or a brace inside a text value: cut there, the text
            # is left open, so it is not JSON.
            continue
        if isinstance(fields, dict):
            settled = fields
    return settled


def json_request(
    model: str,
    max_tokens: int,
    system: str,
    messages: list[MessageParam],
    schema: dict[str, Any],
) -> dict[str, Any]:
    # Every field stream_json sends, in one dict. The response cache makes
    # its key from this same dict, so nothing sent is left out of the key.
    return {
        "model": model,
        "max_tokens": max_tokens,
        "system": system,
        "messages": messages,
        "output_config": {
            "format": {"type": "json_schema", "schema": schema}
        },
    }


def stream_json(
    client: Anthropic,
    model: str,
    max_tokens: int,
    system: str,
    messages: list[MessageParam],
    schema: dict[str, Any],
) -> tuple[Message | None, Timing]:
    # An answer held to a JSON schema, streamed: the SDK refuses an
    # ordinary call whose max_tokens could take over ten minutes. Each
    # field is shown once it has settled, on the error output, as a
    # preview for a person; code uses the answer only from the final
    # Message, after its stop reason and its checks. No Message when you
    # cancel it with Ctrl-C.
    first_token: float | None = None
    text = ""
    shown: list[str] = []
    start = perf_counter()
    try:
        with client.messages.stream(
            **json_request(model, max_tokens, system, messages, schema)
        ) as stream:
            for piece in stream.text_stream:
                if first_token is None and piece:
                    first_token = perf_counter() - start
                text += piece
                for name, value in settled_fields(text).items():
                    if name not in shown:
                        shown.append(name)
                        print(
                            f"{'Settled:':<15}{name} {json.dumps(value)}",
                            file=sys.stderr,
                        )
            response = stream.get_final_message()
            total = perf_counter() - start
            log_call(
                response,
                model,
                total,
                system,
                messages,
                stream.request_id,
            )
            return response, Timing(first_token, total)
    except KeyboardInterrupt:
        # Leaving the with block closed the connection: no more comes.
        return None, Timing(first_token, perf_counter() - start)
