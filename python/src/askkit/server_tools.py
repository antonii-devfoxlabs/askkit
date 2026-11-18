from anthropic.types import (
    ContentBlock,
    Message,
    WebSearchTool20250305Param,
)

# The most searches one call may run. Each one is billed, and each one's
# results are read as input tokens.
MAX_SEARCHES = 3


def web_search(sites: tuple[str, ...]) -> WebSearchTool20250305Param:
    # Anthropic runs this tool on its own servers: the model asks, the
    # API searches, and the results come back in the same reply. AskKit
    # never sees the request before it runs, so its limits go into the
    # definition: results only from the sites you name, and a few
    # searches at most. The basic version, with no code run behind it.
    return {
        "type": "web_search_20250305",
        "name": "web_search",
        "max_uses": MAX_SEARCHES,
        "allowed_domains": list(sites),
    }


def server_line(block: ContentBlock) -> str | None:
    # A line for a server tool's request or its failure, so you can see
    # what ran on Anthropic's servers; None for every other block.
    if block.type == "server_tool_use" and block.name == "web_search":
        return f"Search:        {block.input.get('query', '')}"
    if block.type == "web_search_tool_result" and not isinstance(
        block.content, list
    ):
        # A failed search is not an exception: the reply is a 200, and
        # the result holds an error code instead of a list of results.
        return f"Search error:  {block.content.error_code}"
    return None


def searches(responses: list[Message]) -> int:
    # The searches the API ran and billed, as its usage counts them.
    count = 0
    for response in responses:
        used = response.usage.server_tool_use
        if used is not None:
            count += used.web_search_requests
    return count


def cited_urls(responses: list[Message]) -> list[str]:
    # Every page the answer cites, each once, in the order cited.
    # Anthropic asks that an answer shown to people shows its sources.
    urls: list[str] = []
    for response in responses:
        for block in response.content:
            if block.type != "text" or block.citations is None:
                continue
            for citation in block.citations:
                if citation.type != "web_search_result_location":
                    continue
                if citation.url not in urls:
                    urls.append(citation.url)
    return urls
