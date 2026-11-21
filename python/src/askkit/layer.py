from dataclasses import dataclass
from typing import Any

import httpx2
from anthropic.types import Message
from any_llm import AnyLLM
from any_llm.exceptions import AnyLLMError, AuthenticationError
from openai import DefaultAsyncHttpxClient

from askkit.config import OPENAI_KEY_NAME
from askkit.models import LIFECYCLES


@dataclass(frozen=True)
class LayerProvider:
    # name: the layer's own name for the provider, as any-llm spells it.
    # key: the environment variable that holds that provider's key.
    name: str
    key: str


# The providers askkit answer reaches through one provider-agnostic
# layer, any-llm, keyed by the provider column of models.py. Claude is
# not here: AskKit calls it through Anthropic's own SDK, which keeps
# what the layer leaves out, such as the request ID, the retry count and
# the spend cap's error class (lesson 13.2). A model of another provider
# joins with a row here and its rows in models.py and prices.py.
LAYER_PROVIDERS: dict[str, LayerProvider] = {
    "Google": LayerProvider("gemini", "GEMINI_API_KEY"),
    "OpenAI": LayerProvider("openai", OPENAI_KEY_NAME),
}

# The models askkit answer can send through the layer. Each needs its
# rows in models.py and prices.py (a test checks them); its provider is
# read from models.py.
LAYER_MODELS: tuple[str, ...] = ("gemini-3.5-flash-lite", "gpt-6-luna")

# No retries under the layer. Each provider's SDK has a default of its
# own, two at OpenAI and none at Google (lesson 13.1), and the layer
# passes a number on. askkit answer retries every model of its chain
# with one policy of its own (lesson 13.5), so a provider SDK that
# retried too would multiply its tries.
LAYER_RETRIES = 0


def layer_provider(model: str) -> LayerProvider:
    return LAYER_PROVIDERS[LIFECYCLES[model].provider]


@dataclass
class Layer:
    # One provider's client in the layer, and the request IDs its
    # provider sent, the newest last. The layer's Message has no
    # _request_id, so AskKit reads the header under the layer (lesson
    # 13.2): only OpenAI's, the one provider here that sends a request ID
    # AskKit keeps.
    llm: AnyLLM
    request_ids: list[str]


def layer_client(model: str, key: str) -> Layer:
    # unified_exceptions: the layer's own error classes for every
    # provider, the original kept in error.original_exception.
    provider = layer_provider(model).name
    request_ids: list[str] = []
    options: dict[str, Any] = {}
    if provider == "openai":

        async def note_request_id(response: httpx2.Response) -> None:
            request_id = response.headers.get("x-request-id")
            if request_id is not None:
                request_ids.append(request_id)

        # OpenAI's own default client, with one hook that runs on every
        # response, retries included.
        options["http_client"] = DefaultAsyncHttpxClient(
            event_hooks={"response": [note_request_id]}
        )
    llm = AnyLLM.create(
        provider,
        api_key=key,
        unified_exceptions=True,
        max_retries=LAYER_RETRIES,
        **options,
    )
    return Layer(llm, request_ids)


def ask_layer(
    layer: Layer,
    question: str,
    model: str,
    max_tokens: int,
    system: str,
    timeout: float,
) -> Message:
    # The same question as ask() sends Claude. messages() takes it in
    # Claude's shape and gives the answer back as the SDK's own Message,
    # whichever provider wrote it: stop reason, blocks and token counts
    # in Claude's words, cache reads apart from input_tokens.
    layer.request_ids.clear()
    response = layer.llm.messages(
        model=model,
        messages=[{"role": "user", "content": question}],
        max_tokens=max_tokens,
        system=system,
        timeout=timeout,
    )
    # messages() gives a stream only when asked for one, and this call
    # asks for none.
    if not isinstance(response, Message):
        raise TypeError(f"the layer sent {model}'s answer as a stream")
    if layer.request_ids:
        # The ID of the response that answered, the last one of its tries.
        response._request_id = layer.request_ids[-1]
    return response


def layer_error(error: AnyLLMError, tries: str) -> str:
    # error.provider_name is the layer's name for the provider that
    # failed, such as "gemini"; the message names the provider as
    # models.py does. tries: how many tries AskKit made, such as "after 5
    # tries", for the failures it retries.
    provider, key = "the provider", "its key"
    for name, layer in LAYER_PROVIDERS.items():
        if layer.name == error.provider_name:
            provider, key = name, layer.key
    if isinstance(error, AuthenticationError):
        return f"{provider} rejected the key; check {key} in .env"
    status = error.status_code
    if status is None:
        return f"no answer from {provider} {tries}; check your connection"
    if status == 429:
        return (
            f"{provider} says too many requests (429) {tries}; wait and retry"
        )
    if status >= 500:
        return f"{provider}'s servers failed ({status}) {tries}"
    return f"{provider} rejected the request ({status})"
