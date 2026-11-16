from askkit.prompts import render


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


def with_documents(question: str, documents: dict[str, str]) -> str:
    values = {"documents": documents_block(documents), "question": question}
    return render("quote_request", values)


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
