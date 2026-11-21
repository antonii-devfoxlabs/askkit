import csv
import math
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter, process_time

from askkit.classify import QUEUES

# The two columns of an inbox file: an email's own name, and its text.
COLUMNS = ["id", "email"]

# What an email's line says when no queue can be trusted: a person
# sorts it, as lesson 4.5 decided for a ticket.
TO_A_PERSON = "to a person"


@dataclass(frozen=True)
class InboxEmail:
    id: str
    email: str


@dataclass(frozen=True)
class JobTimes:
    # wall: the seconds a person waits for the whole job, read on
    # perf_counter. cpu: the seconds the processor spent running
    # AskKit's own code meanwhile, read on process_time, which does not
    # count the time spent waiting.
    wall: float
    cpu: float


def read_inbox(path: Path) -> list[InboxEmail]:
    # The csv module, not split(","): an email holds commas and line
    # breaks of its own, which the file keeps inside double quotes.
    # newline="" leaves those line breaks for csv to read.
    with path.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames != COLUMNS:
            found = ",".join(reader.fieldnames or [])
            raise ValueError(
                f"{path}: the columns must be id,email; this file has "
                f"{found or 'none'}"
            )
        emails = [InboxEmail(row["id"], row["email"]) for row in reader]
    if not emails:
        raise ValueError(f"{path} holds no emails")
    return emails


class Clocks:
    # Two clocks started together: read() says how long the job has
    # taken, and how much of that the processor spent on AskKit's code.
    def __init__(self) -> None:
        self.wall_start = perf_counter()
        self.cpu_start = process_time()

    def read(self) -> JobTimes:
        return JobTimes(
            perf_counter() - self.wall_start,
            process_time() - self.cpu_start,
        )


def waiting_share(times: JobTimes) -> float:
    # The share of the job's time when AskKit's code was not running:
    # it was waiting for the network and the model.
    if times.wall <= 0:
        return 0.0
    return max(times.wall - times.cpu, 0.0) / times.wall


def whole_job_seconds(
    call_seconds: list[float], total: int, at_once: int
) -> float:
    # at_once calls wait at all times, so each of the at_once places
    # answers its share of the file one email after another, each about
    # as long as an average call. One at a time, that is every email.
    average = sum(call_seconds) / len(call_seconds)
    return average * math.ceil(total / at_once)


@dataclass(frozen=True)
class Rate:
    # What a long run at this --at-once sends a minute, in the units the
    # API's rate limits count: requests, and input tokens.
    requests: float
    input_tokens: float


def run_rate(
    call_seconds: list[float], counted_tokens: list[int], at_once: int
) -> Rate:
    # at_once places, each sending its next request the moment its call
    # answers: a place sends 60 / average seconds requests a minute.
    # counted_tokens are the input tokens a rate limit counts per call.
    average_seconds = sum(call_seconds) / len(call_seconds)
    requests = at_once * 60 / average_seconds
    average_tokens = sum(counted_tokens) / len(counted_tokens)
    return Rate(requests, requests * average_tokens)


def rate_text(rate: Rate) -> str:
    # Rounded the way a rate limit is written: whole requests, and input
    # tokens to the nearest hundred.
    tokens = round(rate.input_tokens, -2)
    return (
        f"about {rate.requests:,.0f} requests, {tokens:,.0f} input tokens"
        " a minute"
    )


def pace_text(at_once: int) -> str:
    # How the emails were sent, for the lines under a run.
    if at_once == 1:
        return "one at a time"
    return f"{at_once:,} at once"


def duration_text(seconds: float) -> str:
    # Rounded the way a person plans: "about 24 minutes".
    if seconds < 1.5:
        return "about 1 second"
    if seconds < 90:
        return f"about {round(seconds)} seconds"
    minutes = seconds / 60
    if minutes < 90:
        return f"about {round(minutes)} minutes"
    return f"about {minutes / 60:.1f} hours"


def queue_counts(queues: list[str]) -> str:
    # Every queue in the order of QUEUES, then the emails left to a
    # person; a queue no email went to is left out.
    parts: list[str] = []
    for queue in [*QUEUES, TO_A_PERSON]:
        count = queues.count(queue)
        if count:
            parts.append(f"{queue} {count}")
    return ", ".join(parts)


def progress_step(total: int) -> int:
    # A progress line after each tenth of the emails: about ten lines for
    # a long run, whatever its size.
    return max(1, math.ceil(total / 10))


def progress_text(
    finished: int, total: int, not_sorted: int, seconds: float
) -> str:
    # finished: the emails answered or failed so far, of total, in
    # seconds. The time left assumes the emails still to come take as
    # long as those so far did.
    left = seconds / finished * (total - finished)
    text = f"{finished:,} of {total:,} done"
    if not_sorted:
        text += f", {not_sorted:,} not sorted"
    return f"{text}, {duration_text(left)} left"
