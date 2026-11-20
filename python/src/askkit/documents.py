from anthropic.types import TextBlockParam

from askkit.caching import CACHE
from askkit.prompts import render

# The words both prompts tell the model to write when it has no answer.
WAY_OUT = "I don't know"


def documents_block(documents: dict[str, str]) -> str:
    # Anthropic's layout: each document numbered, with its source.
    lines = ["<documents>"]
    index = 1
    for source, content in documents.items():
        lines.append(f'<document index="{index}">')
        lines.append(f"<source>{source}</source>")
        lines.append("<document_content>")
        lines.append(content.strip())
        lines.append("</document_content>")
        lines.append("</document>")
        index += 1
    lines.append("</documents>")
    return "\n".join(lines)


def documents_request(documents: dict[str, str]) -> str:
    return render("quote_request", {"documents": documents_block(documents)})


def question_part(question: str) -> str:
    return f"<question>\n{question}\n</question>"


def with_documents(question: str, documents: dict[str, str]) -> str:
    # The whole message as one text, for a history file and the eval: the
    # same words, in the same order, as the two blocks below.
    return documents_request(documents) + "\n\n" + question_part(question)


def document_question(
    question: str, documents: dict[str, str]
) -> list[TextBlockParam]:
    # The documents and the request to quote stay the same while you ask
    # about them, so they end in a breakpoint and the next question reads
    # them from the cache. Only the question changes; it goes last.
    return [
        {
            "type": "text",
            "text": documents_request(documents),
            "cache_control": CACHE,
        },
        {"type": "text", "text": question_part(question)},
    ]


def flat(text: str) -> str:
    # A file breaks lines inside sentences; a quote runs on one line.
    return " ".join(text.split())


def quotes_in(answer: str) -> list[str]:
    quotes = []
    for piece in answer.split("<quote>")[1:]:
        quotes.append(piece.split("</quote>")[0].strip())
    return quotes


def missing_quotes(answer: str, documents: dict[str, str]) -> list[str]:
    missing = []
    for quote in quotes_in(answer):
        found = False
        for content in documents.values():
            if flat(quote) in flat(content):
                found = True
        if not found:
            missing.append(quote)
    return missing


def unchecked(answer: str) -> bool:
    # An answer that quotes nothing gives missing_quotes nothing to check,
    # so it would pass unchecked; only the prompt's way out may do that.
    return not quotes_in(answer) and WAY_OUT not in answer
