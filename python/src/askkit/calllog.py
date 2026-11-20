import json
import statistics
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

from anthropic.types import Message, MessageParam
from pydantic import BaseModel, ConfigDict

from askkit.cost import usage_cost
from askkit.prices import SEARCH_PRICE
from askkit.server_tools import searches

# Where AskKit writes down its calls: one file of numbers a day, and,
# only when you ask for it, one file of what was sent and what came
# back. .gitignore leaves the folder out.
LOG_FOLDER = Path(".askkit-log")

# The numbers hold no one's words, so they are kept. What was sent can
# hold what a customer wrote: it is kept for a week, long enough to
# look into a bad answer, and removed after.
CONTENT_MAX_AGE = timedelta(days=7)


# One call, as the log keeps it. Every field is a number or a name
# AskKit chose, never a word the model read or wrote.
class LoggedCall(BaseModel):
    model_config = ConfigDict(extra="forbid")

    at: datetime
    command: str
    model: str
    input_tokens: int
    cache_read: int
    cache_written: int
    output_tokens: int
    seconds: float
    cost: float | None
    stop_reason: str | None
    retries: int
    request_id: str | None


@dataclass
class Run:
    # What one run of the CLI logs: the folder, the command that makes
    # the calls, and whether what was sent is kept too (--log-content).
    folder: Path
    command: str
    content: bool


# The run being logged. The CLI sets it before its first call; outside
# the CLI, in the eval and in tests that call a function directly, it is
# None and nothing is written.
current: Run | None = None


def start_log(folder: Path, command: str, content: bool) -> None:
    # global: the assignment changes the module's current, not a new
    # variable inside this function.
    global current
    current = Run(folder, command, content)


def clock() -> datetime:
    # The one place the log reads the time, in UTC, so a day is the same
    # day wherever AskKit runs, and a test can set it.
    return datetime.now(UTC)


def text_of(content: object) -> str:
    # The words of a message's content. A block that is not text, such as
    # an image or a PDF, is named, never copied into the log; a request
    # for a tool is named with its tool.
    if isinstance(content, str):
        return content
    parts: list[str] = []
    if isinstance(content, Iterable):
        for block in content:
            if isinstance(block, BaseModel):
                # The model's own turn, sent back as the SDK returned it.
                block = block.model_dump()
            if not isinstance(block, dict):
                continue
            kind = block.get("type")
            if kind == "text":
                parts.append(str(block["text"]))
            elif kind == "tool_result":
                parts.append(text_of(block.get("content", "")))
            elif kind == "tool_use":
                parts.append(f"[tool_use: {block['name']}]")
            else:
                parts.append(f"[{kind}]")
    return "\n".join(parts)


def prompt_text(system: str, messages: list[MessageParam]) -> str:
    lines = [f"system: {system}"]
    for message in messages:
        lines.append(f"{message['role']}: {text_of(message['content'])}")
    return "\n".join(lines)


def request_id_of(response: object) -> str | None:
    # The SDK sets _request_id (public, despite the underscore) on a
    # response it received; a Message AskKit built itself, such as an
    # OpenAI answer in Claude's shape, has none.
    value = getattr(response, "_request_id", None)
    return value if isinstance(value, str) else None


def logged_call(
    command: str,
    response: Message,
    model: str,
    seconds: float,
    retries: int,
    request_id: str | None,
    at: datetime,
) -> LoggedCall:
    # The cost is worked out now, at today's prices, and kept: when a
    # price changes, the days before it still add up to what they cost.
    # Searches are billed on top of their tokens.
    usage = response.usage
    cost = usage_cost(model, usage) + searches([response]) * SEARCH_PRICE
    # Rounded to a ten-thousandth of a cent: enough for any sum of days,
    # without the float's last digits (0.00043270000000000006).
    cost = round(cost, 10)
    return LoggedCall(
        at=at,
        command=command,
        # The model AskKit asked and prices the call at, so the name and
        # the cost always agree.
        model=model,
        input_tokens=usage.input_tokens,
        cache_read=usage.cache_read_input_tokens or 0,
        cache_written=usage.cache_creation_input_tokens or 0,
        output_tokens=usage.output_tokens,
        seconds=round(seconds, 3),
        cost=cost,
        stop_reason=response.stop_reason,
        retries=retries,
        request_id=request_id,
    )


def day_file(folder: Path, kind: str, day: date) -> Path:
    return folder / f"{kind}-{day.isoformat()}.jsonl"


def append_line(path: Path, data: str) -> None:
    # "a" adds to the end of the file: lines already written stay as
    # they are, and two runs on the same day share one file.
    path.parent.mkdir(exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(data + "\n")


def remove_old_content(folder: Path, today: date) -> None:
    # The day is in the file's name, so its age needs no reading.
    for path in folder.glob("content-*.jsonl"):
        day = date.fromisoformat(path.stem.removeprefix("content-"))
        if today - day > CONTENT_MAX_AGE:
            path.unlink()


def write_call(call: LoggedCall, prompt: str, answer: str) -> None:
    # Called by each function that makes a billed call, once the answer
    # is whole, before AskKit checks it: a call that fails a check was
    # still paid for.
    if current is None:
        return
    day = call.at.date()
    append_line(
        day_file(current.folder, "calls", day), call.model_dump_json()
    )
    if current.content:
        kept = {
            # The same text as the calls file writes, to match them up.
            "at": call.model_dump(mode="json")["at"],
            "request_id": call.request_id,
            "prompt": prompt,
            "answer": answer,
        }
        append_line(
            day_file(current.folder, "content", day), json.dumps(kept)
        )
        remove_old_content(current.folder, day)


def log_call(
    response: Message,
    model: str,
    seconds: float,
    system: str,
    messages: list[MessageParam],
    request_id: str | None,
    retries: int = 0,
) -> None:
    # request_id: the API's ID of the request, from its request-id
    # header: on the stream for a streamed answer, request_id_of(response)
    # for any other.
    if current is None:
        return
    call = logged_call(
        current.command,
        response,
        model,
        seconds,
        retries,
        request_id,
        clock(),
    )
    answer = text_of(response.content)
    write_call(call, prompt_text(system, messages), answer)


def day_calls(folder: Path, day: date) -> list[LoggedCall]:
    path = day_file(folder, "calls", day)
    calls: list[LoggedCall] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        calls.append(LoggedCall.model_validate_json(line))
    return calls


@dataclass
class Group:
    # The calls of one command, or of one model, added up.
    name: str
    calls: int
    retries: int
    cache_read: int
    median_seconds: float
    cost: float


def grouped(calls: list[LoggedCall], by_model: bool) -> list[Group]:
    names: dict[str, list[LoggedCall]] = {}
    for call in calls:
        name = call.model if by_model else call.command
        names.setdefault(name, []).append(call)
    groups: list[Group] = []
    for name, members in names.items():
        groups.append(
            Group(
                name=name,
                calls=len(members),
                retries=sum(call.retries for call in members),
                cache_read=sum(call.cache_read for call in members),
                median_seconds=statistics.median(
                    call.seconds for call in members
                ),
                cost=sum(call.cost or 0.0 for call in members),
            )
        )
    return sorted(groups, key=lambda group: group.cost, reverse=True)


def day_lines(calls: list[LoggedCall], day: date) -> list[str]:
    total = sum(call.cost or 0.0 for call in calls)
    lines = [
        f"Day:           {day.isoformat()} (UTC)",
        f"Calls:         {len(calls)}, ${total:.6f}",
    ]
    unpriced = [call for call in calls if call.cost is None]
    if unpriced:
        lines.append(f"Not priced:    {len(unpriced)} (length unknown)")
    slowest = max(calls, key=lambda call: call.seconds)
    lines.append(
        f"Slowest:       {slowest.command}, {slowest.seconds:.2f} s"
        f" ({slowest.request_id or 'no request ID'})"
    )
    lines.append("")
    lines.append(
        f"{'Command':<12}{'Calls':>6}{'Retries':>9}{'Cache read':>12}"
        f"{'Median':>9}{'Cost':>12}"
    )
    for group in grouped(calls, by_model=False):
        lines.append(
            f"{group.name:<12}{group.calls:>6}{group.retries:>9}"
            f"{group.cache_read:>12,}{group.median_seconds:>8.2f}s"
            f"{'$' + format(group.cost, '.6f'):>12}"
        )
    lines.append("")
    lines.append(f"{'Model':<21}{'Calls':>6}{'Cost':>12}")
    for group in grouped(calls, by_model=True):
        lines.append(
            f"{group.name:<21}{group.calls:>6}"
            f"{'$' + format(group.cost, '.6f'):>12}"
        )
    return lines
