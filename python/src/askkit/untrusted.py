import json


def marked(source: str, content: str) -> str:
    # Text someone else wrote, as one JSON object with its source.
    # json.dumps escapes every quote and line break inside content, so
    # the writer can neither end the object nor add a field to it.
    return json.dumps(
        {"source": source, "content": content}, ensure_ascii=False
    )
