import hashlib
import re
from pathlib import Path

import pytest

from askkit.prompts import ASK_SYSTEM, IN_USE, TEMPLATES, render

# Every version that has been sent, with a fingerprint of its bytes. A
# sent version is never edited, so its fingerprint never changes: a new
# version is a new file, added here and to CHANGES.md.
RELEASED = {
    "ask_system.v1.txt": (
        "fb2b2715e90c45f95754b141992dadf42aa90655698fd7eae60c12b0d9d79346"
    ),
    "ask_system.v2.txt": (
        "70bbce54f01c9a82d943f0875bfee892ef480a276516ac0ad0d1349ce8719761"
    ),
    "classify_email.v1.txt": (
        "9fc584cd96a95a585ccddacc8a2453835979df96d90b8895525985f5cd3cf9ee"
    ),
    "classify_email.v2.txt": (
        "b71e0d97eb161dc86df9c899202f5f29dd5fd509f20ebc20eff2aa5a6557eccd"
    ),
    "extract_ticket.v1.txt": (
        "9c52494b304b836c87ed89cf78160ae10d6128d372911b6bf21574eb7a3635f1"
    ),
    "extract_ticket.v2.txt": (
        "a42d1f762451c925326027eed710ab5bff1a29790c34542ce97ca18c2faea50e"
    ),
    "extract_ticket.v3.txt": (
        "833b51c321b0a1e8328dd9028aee1d4855d8869e5662d186abf8a7fb47666e01"
    ),
    "extract_ticket.v4.txt": (
        "0d32d6a38dc3ac4e72f525731a2d9190143a80ab159d4d5e191a2c99ab1d0896"
    ),
    "quote_request.v1.txt": (
        "754bbdfd562c8534f7eb354092da4ddf3c0c43c85f62c7ea7c5937986364d903"
    ),
    "quote_request.v2.txt": (
        "476a48c7d1346e43e4472aec0f61f4f032f606829d9095a07a3372959fe4614c"
    ),
    "quote_request.v3.txt": (
        "b7d00a753696ced0df90d46e3dc506e674bfbb093bfac0f6ee4a71712497b19c"
    ),
    "retry_ticket.v1.txt": (
        "a88920196f5db7dd08eaee247474631854b76edfcd95408222ecbeb014b6af8c"
    ),
    "summarise_turns.v1.txt": (
        "f933d9595272db575d30ef4890c6871efa9d1aa3e0572bf90c976a1748c7f7ff"
    ),
    "fold_turns.v1.txt": (
        "05aa2042426c91896a81c57311942f2a4440cfb58e5ca92e766c51b1640ee2e8"
    ),
    "earlier_turns.v1.txt": (
        "336fc685957a278165c49287f580eebd05700aeb50db9f18e0e99fa8253a73b5"
    ),
    "remembered.v1.txt": (
        "2f61f208d01e317554cd227af33e132a4f33f7e55f3da1f1aa2f3ac7ebae2d80"
    ),
    "read_invoice.v1.txt": (
        "e71a36521ee912dfc1e225bd506c60e9d9582a87ea3067b4b6d84e8c0c5c0f4c"
    ),
    "read_invoice.v2.txt": (
        "55f0cf403ff2bffae196cd712839e81728110b76231f215b273e8504560c60a3"
    ),
}


def template_files() -> list[Path]:
    return sorted(TEMPLATES.glob("*.txt"))


def test_a_sent_version_never_changes() -> None:
    for path in template_files():
        fingerprint = hashlib.sha256(path.read_bytes()).hexdigest()
        assert RELEASED.get(path.name) == fingerprint, path.name


def test_every_template_file_is_released_and_described() -> None:
    changes = (TEMPLATES / "CHANGES.md").read_text(encoding="utf-8")
    assert sorted(RELEASED) == [path.name for path in template_files()]
    for file_name in RELEASED:
        name, version = file_name.removesuffix(".txt").split(".v")
        assert f"## {name}\n" in changes
        assert f"- v{version}: " in changes


def test_the_versions_in_use_exist() -> None:
    for name, version in IN_USE.items():
        assert f"{name}.v{version}.txt" in RELEASED


def test_every_tag_in_a_template_is_closed() -> None:
    for path in template_files():
        text = path.read_text(encoding="utf-8")
        for name in re.findall(r"<([a-z_]+)>", text):
            assert text.count(f"<{name}>") == text.count(f"</{name}>"), (
                f"{path.name}: <{name}>"
            )


def test_render_fills_every_placeholder() -> None:
    message = render(
        "quote_request", {"documents": "<documents/>", "question": "Who?"}, 2
    )
    assert message.startswith("<documents/>\n\nAnswer the question below")
    assert message.endswith("<question>\nWho?\n</question>")
    assert "$" not in message


def test_render_reads_an_older_version_when_asked() -> None:
    values = {"documents": "", "question": ""}
    assert "Before you write the" not in render("quote_request", values, 1)
    assert "Before you write the" in render("quote_request", values, 2)


def test_render_stops_on_a_missing_value() -> None:
    with pytest.raises(ValueError) as error:
        render("quote_request", {"documents": "<documents/>"}, 2)
    assert str(error.value) == (
        "quote_request v2 needs ['documents', 'question'],"
        " but was given ['documents']"
    )


def test_render_stops_on_a_value_with_no_placeholder() -> None:
    # Template.substitute() would drop it without a word.
    with pytest.raises(ValueError) as error:
        render("ask_system", {"question": "Who pays?"})
    assert str(error.value) == (
        "ask_system v2 needs [], but was given ['question']"
    )


def test_the_system_prompt_is_the_version_in_use() -> None:
    assert ASK_SYSTEM == render("ask_system", {}, IN_USE["ask_system"])
    assert not ASK_SYSTEM.endswith("\n")
