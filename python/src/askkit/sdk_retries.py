from typing import Any

from anthropic import APIRequest, APIResponse, CallNext, Middleware


class RetryCount(Middleware):
    # The SDK sends a failed request again by itself (lesson 2.6), and the
    # answer comes back as if it had been sent once. A call that is not
    # streamed can read the count from its raw response (retries_taken);
    # the stream helper, messages.stream, has none. Middleware runs on
    # every try, inside the SDK's retry loop, so this one notes how many
    # retries came before the try that answered: the SDK's retries_taken,
    # 0 on the first try. One RetryCount per call, given to that call's
    # client with client.with_middleware(count).
    def __init__(self) -> None:
        self.retries = 0

    def handle(
        self, request: APIRequest, call_next: CallNext
    ) -> APIResponse[Any]:
        self.retries = request.retries_taken
        return call_next(request)
