import json
import sys
from time import perf_counter
from typing import Any

from anthropic import Anthropic, AsyncAnthropic
from anthropic.types import Message, MessageParam

from askkit.ask import Timing
from askkit.calllog import log_call, request_id_of
from askkit.sdk_retries import RetryCount


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


def settled_value(value: Any) -> str:
    # A field as the preview shows it. A list, such as an invoice's rows,
    # is shown by its length: its JSON would fill a line per field with
    # every row, and the rows are printed whole once they are checked.
    if isinstance(value, list):
        return f"(a list of {len(value)})"
    return json.dumps(value)


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
    preview: bool = True,
) -> tuple[Message | None, Timing]:
    # An answer held to a JSON schema, streamed: the SDK refuses an
    # ordinary call whose max_tokens could take over ten minutes. Each
    # field is shown once it has settled, on the error output, as a
    # preview for a person; code uses the answer only from the final
    # Message, after its stop reason and its checks. No Message when you
    # cancel it with Ctrl-C. preview=False shows nothing: a run of many
    # emails prints one line for each instead.
    first_token: float | None = None
    text = ""
    shown: list[str] = []
    count = RetryCount()
    start = perf_counter()
    try:
        with client.with_middleware(count).messages.stream(
            **json_request(model, max_tokens, system, messages, schema)
        ) as stream:
            for piece in stream.text_stream:
                if first_token is None and piece:
                    first_token = perf_counter() - start
                text += piece
                if not preview:
                    continue
                for name, value in settled_fields(text).items():
                    if name not in shown:
                        shown.append(name)
                        print(
                            f"{'Settled:':<15}{name} {settled_value(value)}",
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
                count.retries,
            )
            return response, Timing(first_token, total)
    except KeyboardInterrupt:
        # Leaving the with block closed the connection: no more comes.
        return None, Timing(first_token, perf_counter() - start)


async def ask_json(
    client: AsyncAnthropic,
    model: str,
    max_tokens: int,
    system: str,
    messages: list[MessageParam],
    schema: dict[str, Any],
) -> Message:
    # The request stream_json sends, on the async client and not
    # streamed: while this call waits for its answer, the program can
    # send and wait on others. Nobody reads a preview in a run of many
    # emails, so the answer comes whole. Each call is timed and logged
    # on its own, so the log shows its own wait, not the whole run's.
    start = perf_counter()
    # The raw response knows the retries the SDK made (retries_taken).
    raw = await client.messages.with_raw_response.create(
        **json_request(model, max_tokens, system, messages, schema)
    )
    response: Message = await raw.parse()
    log_call(
        response,
        model,
        perf_counter() - start,
        system,
        messages,
        request_id_of(response),
        raw.retries_taken,
    )
    return response
