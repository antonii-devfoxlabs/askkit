import anthropic

AFTER_RETRIES = "after the SDK's retries"


def error_message(
    error: anthropic.APIError, after: str = AFTER_RETRIES
) -> str:
    # after says what was tried before giving up: the SDK's retries, or
    # a command's own (askkit answer). Most specific first: the spend
    # cap's error is a kind of RateLimitError, OverloadedError a kind of
    # InternalServerError, and APITimeoutError a kind of
    # APIConnectionError.
    if isinstance(error, anthropic.AuthenticationError):
        return "the API key was rejected (401); put a working key in .env"
    if isinstance(error, anthropic.PermissionDeniedError):
        return "this key may not make this request (403); check its workspace"
    if isinstance(error, anthropic.NotFoundError):
        return "the model was not found (404); it may be retired or not yours"
    if isinstance(error, anthropic.BadRequestError):
        # The API's own message names what is wrong with the request.
        return f"the API rejected the request (400): {error.api_message}"
    if isinstance(error, anthropic.ContentTooLargeError):
        return "the request is too large (413); send less text"
    if isinstance(error, anthropic.EnforcedSpendLimitReachedError):
        # Waiting a minute will not help; the API names the date.
        return f"the monthly spend cap was reached (429): {error.api_message}"
    if isinstance(error, anthropic.RateLimitError):
        return (
            f"the rate limit was reached (429) {after};"
            "\nwait a minute and try again"
        )
    if isinstance(error, anthropic.OverloadedError):
        return (
            f"Anthropic's servers are overloaded (529) {after};"
            "\ntry again in a few minutes"
            f" (request ID {error.request_id})"
        )
    if isinstance(error, anthropic.InternalServerError):
        return (
            f"Anthropic's side failed ({error.status_code})"
            f" {after};"
            f"\ntry again later (request ID {error.request_id})"
        )
    if (
        isinstance(error, anthropic.APIStatusError)
        and error.status_code == 200
    ):
        # An error event inside a stream: the answer had begun (200), so
        # the SDK does not send the request again.
        return (
            f"the answer broke off midway ({error.type}):"
            f" {error.api_message};\nask again"
        )
    if isinstance(error, anthropic.APIStatusError):
        return (
            f"the API answered {error.status_code} ({error.type}):"
            f" {error.api_message}"
        )
    if isinstance(error, anthropic.APITimeoutError):
        return f"no answer in time {after}; try again"
    return f"could not reach the API {after};\ncheck your connection"


# The start of the 400 Anthropic answers once the monthly spend limit
# you set yourself (lesson 2.1) is reached: every request gets it.
SPEND_LIMIT_MESSAGE = "You have reached your specified API usage limits"


def stops_the_run(error: anthropic.APIError) -> bool:
    # An error every other request of a run would get as well: the key,
    # its permissions, the model, billing or the month's spending. The
    # run stops, since no other email can get through. Any other error
    # is about one request or one moment (a 400 for this email's text,
    # a 529 after the SDK's retries): that email is left for the next
    # run.
    if isinstance(
        error,
        (
            anthropic.AuthenticationError,
            anthropic.PermissionDeniedError,
            anthropic.NotFoundError,
            anthropic.EnforcedSpendLimitReachedError,
        ),
    ):
        return True
    if isinstance(error, anthropic.BadRequestError):
        message = error.api_message or ""
        return message.startswith(SPEND_LIMIT_MESSAGE)
    if isinstance(error, anthropic.APIStatusError):
        return error.status_code == 402
    return False


def error_name(error: anthropic.APIError) -> str:
    # A few words for one failed request among many, as a run lists it:
    # the status code and the API's name for the error.
    if isinstance(error, anthropic.APIStatusError):
        if error.type is None:
            return str(error.status_code)
        return f"{error.status_code} {error.type}"
    if isinstance(error, anthropic.APITimeoutError):
        return "no answer in time"
    return "could not reach the API"
