import random
import time
from collections.abc import Callable
from dataclasses import dataclass

import anthropic
from anthropic.types import Message


@dataclass(frozen=True)
class RetryPolicy:
    # tries: every request one call may send, the first one included.
    # first_wait and max_wait: the longest wait before the first retry,
    # doubled for each retry after it, never over max_wait (seconds).
    # deadline: seconds from the first send after which no retry starts.
    tries: int
    first_wait: float
    max_wait: float
    deadline: float


# askkit answer works through a whole set of questions, and a question
# that fails loses its place in the run, so it waits longer than the
# SDK's three tries in about a second and a half: five tries, spread
# over up to two minutes.
ANSWER_POLICY = RetryPolicy(
    tries=5, first_wait=1.0, max_wait=30.0, deadline=120.0
)

# Seconds to reach the API, as the SDK's default, and the longest wait
# for one answer. The SDK waits ten minutes for an answer; askkit answer
# asks for --max-tokens 1000 unless told otherwise, which at the SDK's
# slowest pace below takes under half a minute.
CONNECT_TIMEOUT = 5.0
ANSWER_TIMEOUT = 60.0

# The SDK's own slowest pace for an answer that is not streamed: 128,000
# output tokens in an hour. A timeout shorter than max_tokens at that
# pace could cut off an answer that is on its way.
SLOWEST_TOKENS_AN_HOUR = 128_000

# Anthropic asks for streaming when a call may take over ten minutes.
LONGEST_TIMEOUT = 600.0

# The statuses worth another try, as the SDK decides (lesson 2.6): a
# request that took too long to arrive (408), a clash with another
# request (409), too much too fast (429), and Anthropic's side (500
# and above).
RETRY_STATUSES = (408, 409, 429)


@dataclass(frozen=True)
class Retry:
    # One failed try that was tried again: why it failed, as a status
    # code or a word, and how long AskKit waited before the next one.
    reason: str
    wait: float


def worth_retrying(error: anthropic.APIError) -> bool:
    if isinstance(error, anthropic.APIStatusError):
        # The API may say itself whether a retry can help; it decides.
        header = error.response.headers.get("x-should-retry")
        if header == "true":
            return True
        if header == "false":
            return False
        if isinstance(error, anthropic.EnforcedSpendLimitReachedError):
            # A spent monthly cap: only next month helps (lesson 2.6).
            return False
        status = error.status_code
        return status in RETRY_STATUSES or status >= 500
    # No answer at all: the network failed, or the timeout ran out.
    return isinstance(error, anthropic.APIConnectionError)


def reason(error: anthropic.APIError) -> str:
    if isinstance(error, anthropic.APIStatusError):
        return str(error.status_code)
    if isinstance(error, anthropic.APITimeoutError):
        return "timeout"
    return "no connection"


def retry_after(error: anthropic.APIError) -> float | None:
    # The seconds the API asks to wait, from its retry-after header.
    # Not every error carries one, and an error with no response at all
    # has no headers: then there is nothing to obey.
    if not isinstance(error, anthropic.APIStatusError):
        return None
    value = error.response.headers.get("retry-after")
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def wait_before(
    retry: int,
    policy: RetryPolicy,
    error: anthropic.APIError,
    draw: Callable[[], float] = random.random,
) -> float:
    # retry counts from 0. The API's own retry-after wins. Otherwise the
    # longest wait doubles with each retry up to max_wait, and the wait
    # is a random share of it, anywhere from nothing to all of it ("full
    # jitter"), so many clients that failed together come back apart.
    asked = retry_after(error)
    if asked is not None:
        return asked
    longest = min(policy.first_wait * 2.0**retry, policy.max_wait)
    return draw() * longest


def with_retries(
    send: Callable[[], Message],
    policy: RetryPolicy,
    retries: list[Retry],
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
    draw: Callable[[], float] = random.random,
) -> Message:
    # Calls send() until it returns, and appends each failed try that is
    # tried again to retries. It stops and raises the last error when the
    # error is not worth another try, when every try is used, or when the
    # wait would end past the deadline.
    start = clock()
    retried = 0
    while True:
        try:
            return send()
        except anthropic.APIError as error:
            if not worth_retrying(error) or retried + 1 >= policy.tries:
                raise
            wait = wait_before(retried, policy, error, draw)
            if clock() - start + wait > policy.deadline:
                raise
            retries.append(Retry(reason(error), wait))
            sleep(wait)
            retried += 1


def slowest_seconds(max_tokens: int) -> float:
    return max_tokens * 3600 / SLOWEST_TOKENS_AN_HOUR


def timeout_fault(tries: int, timeout: float, max_tokens: int) -> str | None:
    # Checked before any call, so a setting that cannot work costs nothing.
    if tries < 1:
        return f"--tries {tries} is below one; give 1 to never try again"
    if not 0 < timeout <= LONGEST_TIMEOUT:
        return (
            f"--timeout {timeout:g} is not from 1 to {LONGEST_TIMEOUT:g}"
            " seconds; a call that may take longer than ten minutes"
            " should be streamed"
        )
    slowest = slowest_seconds(max_tokens)
    if slowest > timeout:
        return (
            f"--max-tokens {max_tokens:,} may take up to {slowest:,.0f} s,"
            f" longer than the --timeout of {timeout:g} s;\ngive a smaller"
            " --max-tokens or a longer --timeout"
        )
    return None


def after_tries(count: int) -> str:
    if count == 1:
        return "after 1 try"
    return f"after {count} tries"


def retries_line(retries: list[Retry]) -> str:
    # One line for the run: how many tries were made again, why, and how
    # long AskKit waited in all.
    reasons: list[str] = []
    waited = 0.0
    for retry in retries:
        reasons.append(retry.reason)
        waited += retry.wait
    return (
        f"Retries:       {len(retries)} ({', '.join(reasons)}),"
        f" {waited:.1f} s waiting"
    )
