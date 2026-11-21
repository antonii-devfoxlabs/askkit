import random
import time
from collections.abc import Callable
from dataclasses import dataclass

import anthropic
from anthropic.types import Message
from any_llm.exceptions import AnyLLMError
from tenacity import (
    RetryCallState,
    Retrying,
    retry_if_exception,
    stop_after_attempt,
    stop_before_delay,
)


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


def worth_retrying(error: BaseException) -> bool:
    # The one rule for which failures are worth another try, for every
    # provider: AskKit's retries and its fallbacks (lesson 10.5) both
    # read it, so a failure is never retried by one and refused by the
    # other.
    if isinstance(error, AnyLLMError):
        # The same rule in the layer's terms: no status means no answer
        # at all.
        status = error.status_code
        return status is None or status in RETRY_STATUSES or status >= 500
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


def reason(error: BaseException) -> str:
    # The reason for the Retries line. The layer gives a timeout and a
    # lost connection the same class and no status, so both read as
    # "no answer".
    if isinstance(error, AnyLLMError):
        if error.status_code is None:
            return "no answer"
        return str(error.status_code)
    if isinstance(error, anthropic.APIStatusError):
        return str(error.status_code)
    if isinstance(error, anthropic.APITimeoutError):
        return "timeout"
    return "no connection"


def retry_after(error: BaseException) -> float | None:
    # The seconds the API asks to wait, from its retry-after header.
    # Not every error carries one, and an error with no response at all
    # has no headers: then there is nothing to obey. A layer error keeps
    # the provider SDK's own error, whose response has the headers.
    source: BaseException | None = error
    if isinstance(error, AnyLLMError):
        source = error.original_exception
    response = getattr(source, "response", None)
    headers = getattr(response, "headers", None)
    if headers is None:
        return None
    value = headers.get("retry-after")
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def wait_before(
    retry: int,
    policy: RetryPolicy,
    error: BaseException,
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


def failed_with(state: RetryCallState) -> BaseException:
    # The error of the try that just failed. tenacity calls wait and
    # before_sleep only after a failure, so there always is one.
    if state.outcome is None:
        raise RuntimeError("no try has ended yet")
    error = state.outcome.exception()
    if error is None:
        raise RuntimeError("the last try did not fail")
    return error


def with_retries(
    send: Callable[[], Message],
    policy: RetryPolicy,
    retries: list[Retry],
    sleep: Callable[[float], None] = time.sleep,
    draw: Callable[[], float] = random.random,
) -> Message:
    # Calls send() until it returns, and appends each failed try that is
    # tried again to retries. The loop is tenacity's (lesson 13.5); the
    # policy's answers are AskKit's: which failures (worth_retrying), how
    # long to wait (wait_before), and when to stop: every try used, or a
    # wait that would end at the deadline or past it. reraise=True raises
    # the provider's own error after the last try, as lesson 10.4's loop
    # did, not tenacity's RetryError, so every except that catches an
    # API error still catches it. sleep and draw are there for the tests,
    # which wait no time and draw the same share every time.
    def wait(state: RetryCallState) -> float:
        return wait_before(
            state.attempt_number - 1, policy, failed_with(state), draw
        )

    def note(state: RetryCallState) -> None:
        error = failed_with(state)
        retries.append(Retry(reason(error), state.upcoming_sleep))

    retrying = Retrying(
        retry=retry_if_exception(worth_retrying),
        wait=wait,
        stop=(
            stop_after_attempt(policy.tries)
            | stop_before_delay(policy.deadline)
        ),
        before_sleep=note,
        sleep=sleep,
        reraise=True,
    )
    return retrying(send)


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
