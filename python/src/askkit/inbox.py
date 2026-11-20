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


def rounds(emails: list[InboxEmail], at_once: int) -> list[list[InboxEmail]]:
    # The emails in groups of at_once, in the order of the file: a group
    # is sent together, and the next one starts once all its answers are
    # in. The last group holds what is left.
    groups: list[list[InboxEmail]] = []
    for start in range(0, len(emails), at_once):
        groups.append(emails[start : start + at_once])
    return groups


def whole_job_seconds(
    times: JobTimes, done: int, total: int, at_once: int
) -> float:
    # The rounds done so far, at the pace they went, for every round of
    # the file: a round takes about one wait, however many it holds.
    rounds_done = math.ceil(done / at_once)
    rounds_total = math.ceil(total / at_once)
    return times.wall / rounds_done * rounds_total


def pace_text(at_once: int) -> str:
    # How the emails were sent, for the lines under a run.
    if at_once == 1:
        return "one at a time"
    return f"{at_once:,} at once"


def duration_text(seconds: float) -> str:
    # Rounded the way a person plans: "about 24 minutes".
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
