import json


def marked(source: str, content: str) -> str:
    # Text someone else wrote, as one JSON object that says where it came
    # from. json.dumps escapes every quote, backslash and line break
    # inside content, so nothing the writer types can end the object or
    # add a field of its own: the object's border is not a tag anyone can
    # type again. ensure_ascii=False keeps an accent as itself, not as
    # é, which costs tokens and is harder for the model to read.
    return json.dumps(
        {"source": source, "content": content}, ensure_ascii=False
    )
