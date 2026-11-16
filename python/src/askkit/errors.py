import anthropic

AFTER_RETRIES = "after the SDK's retries"


def error_message(error: anthropic.APIError) -> str:
    # Most specific first: the spend cap's error is a kind of
    # RateLimitError, OverloadedError a kind of InternalServerError,
    # and APITimeoutError a kind of APIConnectionError.
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
            f"the rate limit was reached (429) {AFTER_RETRIES};"
            "\nwait a minute and try again"
        )
    if isinstance(error, anthropic.OverloadedError):
        return (
            f"Anthropic's servers are overloaded (529) {AFTER_RETRIES};"
            "\ntry again in a few minutes"
            f" (request ID {error.request_id})"
        )
    if isinstance(error, anthropic.InternalServerError):
        return (
            f"Anthropic's side failed ({error.status_code})"
            f" {AFTER_RETRIES};"
            f"\ntry again later (request ID {error.request_id})"
        )
    if isinstance(error, anthropic.APIStatusError):
        return (
            f"the API answered {error.status_code} ({error.type}):"
            f" {error.api_message}"
        )
    if isinstance(error, anthropic.APITimeoutError):
        return f"no answer in time {AFTER_RETRIES}; try again"
    return f"could not reach the API {AFTER_RETRIES};\ncheck your connection"
