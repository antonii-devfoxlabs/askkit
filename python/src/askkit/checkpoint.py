import json
from pathlib import Path

# A checkpoint is a JSON Lines file (lesson 10.8) with one line for each
# email whose queue is known, written the moment it is known:
#
#     {"id": "e0001", "queue": "damaged"}
#
# A job that stops, however it stops, leaves every line it wrote. The
# next run reads them and sends only the emails they do not name, so no
# email is paid for twice, except the few whose answer was on its way
# when the job stopped.


def whole_record(line: str) -> tuple[str, str] | None:
    # The id and queue of one line, or None for a line that is not a
    # whole record: the end of a line cut off when the job stopped while
    # writing it.
    try:
        record = json.loads(line)
    except json.JSONDecodeError:
        return None
    if not isinstance(record, dict):
        return None
    email_id = record.get("id")
    queue = record.get("queue")
    if not isinstance(email_id, str) or not isinstance(queue, str):
        return None
    return email_id, queue


def read_checkpoint(path: Path) -> dict[str, str]:
    # Each sorted email's queue, by its id. No file yet: nothing is
    # sorted. A cut-off line is skipped, so its email is sent again:
    # sorting an email twice costs a call, losing one costs a customer.
    if not path.exists():
        return {}
    done: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        record = whole_record(line)
        if record is not None:
            email_id, queue = record
            done[email_id] = queue
    return done


def start_checkpoint(path: Path) -> dict[str, str]:
    # What the checkpoint already holds, read before a run sends
    # anything. A file whose last line was cut off gets a line break
    # first, so the next record starts on a line of its own and the
    # cut-off part stays one line to skip.
    done = read_checkpoint(path)
    if path.exists() and path.stat().st_size > 0:
        with path.open("rb") as file:
            file.seek(-1, 2)
            cut_off = file.read(1) != b"\n"
        if cut_off:
            with path.open("a", encoding="utf-8") as file:
                file.write("\n")
    return done


def record_sorted(path: Path, email_id: str, queue: str) -> None:
    # One line, appended ("a" keeps what is there). Closing the file at
    # the end of the with block hands the line to the operating system,
    # which keeps it even if AskKit is killed the next moment.
    line = json.dumps({"id": email_id, "queue": queue})
    with path.open("a", encoding="utf-8") as file:
        file.write(line + "\n")
