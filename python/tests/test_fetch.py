import socket
from collections.abc import Callable, Iterator
from itertools import count
from typing import Any

import httpx2
import pytest
from test_tools import ANSWERED, asked_for, call, use_answers

from askkit.cli import main
from askkit.fetch import (
    FETCH,
    MAX_PAGE_BYTES,
    TOO_LARGE_PAGE,
    TOO_SLOW,
    addresses_of,
    allowed_host,
    fetch_page,
    open_client,
    public_address,
)
from askkit.tools import CALCULATOR, CLOCK, run_tool

SITES = ("docs.python.org",)
PAGE_URL = "https://docs.python.org/3/library/secrets.html"
PUBLIC = "151.101.0.223"

NOT_ALLOWED = (
    "fetch_page fetches only from the sites the user allowed, and this"
    " address is not on one of them; do not try other addresses: tell the"
    " user they can allow the site with --allow-site"
)
PRIVATE = (
    "docs.python.org leads to a private network address, which fetch_page"
    " never reaches; do not call it again: tell the user the page could"
    " not be fetched"
)

Handler = Callable[[httpx2.Request], httpx2.Response]


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    # No test may look a name up or open a connection for real.
    def refuse(*arguments: object, **options: object) -> object:
        raise AssertionError("a test tried to reach the network")

    monkeypatch.setattr("askkit.fetch.socket.getaddrinfo", refuse)
    monkeypatch.setattr("askkit.fetch.open_client", refuse)


def lookup_gives(
    monkeypatch: pytest.MonkeyPatch, addresses: list[str]
) -> None:
    monkeypatch.setattr("askkit.fetch.addresses_of", lambda host: addresses)


def site_answers(
    monkeypatch: pytest.MonkeyPatch,
    handler: Handler,
    sent: list[httpx2.Request] | None = None,
) -> None:
    # A stand-in site: every request goes to handler, never the network.
    def record(request: httpx2.Request) -> httpx2.Response:
        if sent is not None:
            sent.append(request)
        return handler(request)

    def make() -> httpx2.Client:
        return httpx2.Client(
            transport=httpx2.MockTransport(record), follow_redirects=False
        )

    monkeypatch.setattr("askkit.fetch.open_client", make)


def page(text: str = "<p>A page</p>") -> Handler:
    def answer(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(
            200,
            headers={"content-type": "text/html; charset=utf-8"},
            content=text.encode(),
        )

    return answer


def fetched(tool_input: dict[str, object]) -> str:
    return fetch_page(tool_input, SITES)


def refusal(tool_input: dict[str, object]) -> str:
    with pytest.raises(ValueError) as refused:
        fetched(tool_input)
    return str(refused.value)


# The definition


def test_the_fetch_takes_one_url() -> None:
    assert FETCH["name"] == "fetch_page"
    schema: Any = FETCH["input_schema"]
    assert schema["required"] == ["url"]
    assert list(schema["properties"]) == ["url"]
    assert schema["properties"]["url"]["type"] == "string"


def test_the_description_says_never_to_get_round_a_refusal() -> None:
    assert "never try another address" in FETCH["description"]


# Allowed destinations


@pytest.mark.parametrize(
    "url",
    [
        "https://docs.python.org/3/",
        "https://DOCS.python.org/3/",
        "https://docs.python.org:443/3/",
    ],
)
def test_an_https_address_on_an_allowed_site_passes(url: str) -> None:
    assert allowed_host(url, SITES) == "docs.python.org"


def test_plain_http_is_refused() -> None:
    with pytest.raises(ValueError) as refused:
        allowed_host("http://docs.python.org/3/", SITES)
    assert str(refused.value) == (
        "fetch_page fetches only https:// addresses; call it again with the"
        " page's https:// address"
    )


@pytest.mark.parametrize(
    "url",
    [
        "https://www.python.org/",
        "https://docs.python.org.evil.example/3/",
        "https://docs.python.org@127.0.0.1/",
        "https://127.0.0.1/",
        "https:///3/",
    ],
)
def test_a_site_you_did_not_allow_is_refused(url: str) -> None:
    with pytest.raises(ValueError) as refused:
        allowed_host(url, SITES)
    assert str(refused.value) == NOT_ALLOWED


def test_with_no_site_allowed_every_address_is_refused() -> None:
    with pytest.raises(ValueError) as refused:
        allowed_host("https://docs.python.org/3/", ())
    assert str(refused.value) == NOT_ALLOWED


@pytest.mark.parametrize(
    "url",
    [
        "https://docs.python.org:8443/3/",
        "https://ana:secret@docs.python.org/3/",
        "https://ana@docs.python.org/3/",
    ],
)
def test_a_port_or_a_user_name_is_refused(url: str) -> None:
    with pytest.raises(ValueError) as refused:
        allowed_host(url, SITES)
    assert str(refused.value) == (
        "the address must name the site alone, with no user name, password"
        " or port; call it again with the plain address"
    )


# Private-network blocking


@pytest.mark.parametrize(
    "address",
    [
        "127.0.0.1",
        "10.0.0.5",
        "172.16.0.1",
        "192.168.1.1",
        "169.254.169.254",
        "100.64.0.1",
        "0.0.0.0",
        "::1",
        "fd00::1",
        "::ffff:127.0.0.1",
    ],
)
def test_a_private_address_is_refused(
    monkeypatch: pytest.MonkeyPatch, address: str
) -> None:
    lookup_gives(monkeypatch, [address])
    with pytest.raises(ValueError) as refused:
        public_address("docs.python.org")
    assert str(refused.value) == PRIVATE
    # The address itself stays on your computer.
    assert address not in str(refused.value)


def test_one_private_address_among_public_ones_is_enough_to_refuse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lookup_gives(monkeypatch, [PUBLIC, "10.0.0.5"])
    with pytest.raises(ValueError) as refused:
        public_address("docs.python.org")
    assert str(refused.value) == PRIVATE


@pytest.mark.parametrize("address", [PUBLIC, "2a04:4e42::223"])
def test_a_public_address_passes(
    monkeypatch: pytest.MonkeyPatch, address: str
) -> None:
    lookup_gives(monkeypatch, [address, "151.101.64.223"])
    assert public_address("docs.python.org") == address


def test_a_name_that_leads_nowhere_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fails(host: str) -> list[str]:
        raise socket.gaierror(8, "nodename nor servname provided")

    monkeypatch.setattr("askkit.fetch.addresses_of", fails)
    with pytest.raises(ValueError) as refused:
        public_address("docs.python.org")
    assert str(refused.value) == (
        "docs.python.org could not be found; tell the user, instead of"
        " calling again"
    )


def test_every_address_of_a_name_is_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    found = [
        (socket.AF_INET, socket.SOCK_STREAM, 6, "", (PUBLIC, 443)),
        (socket.AF_INET6, socket.SOCK_STREAM, 6, "", ("::1", 443, 0, 0)),
    ]
    monkeypatch.setattr(
        "askkit.fetch.socket.getaddrinfo", lambda *arguments, **o: found
    )
    assert addresses_of("docs.python.org") == [PUBLIC, "::1"]


# The download


def test_the_client_follows_no_redirect_and_waits_at_most_five_seconds() -> (
    None
):
    # The real client, before any test replaces it.
    with open_client() as client:
        assert client.follow_redirects is False
        assert client.timeout == httpx2.Timeout(5.0)
        assert client.trust_env is False


def test_a_page_is_fetched_from_the_checked_address(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lookup_gives(monkeypatch, [PUBLIC])
    sent: list[httpx2.Request] = []
    site_answers(monkeypatch, page(), sent)
    assert fetched({"url": PAGE_URL + "?x=1"}) == "<p>A page</p>"
    (request,) = sent
    # The connection goes to the address that was checked, not to a
    # second look-up of the name; the site still sees its own name.
    assert str(request.url) == f"https://{PUBLIC}/3/library/secrets.html?x=1"
    assert request.headers["host"] == "docs.python.org"
    assert request.extensions["sni_hostname"] == "docs.python.org"


def test_an_ipv6_address_goes_in_square_brackets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lookup_gives(monkeypatch, ["2a04:4e42::223"])
    sent: list[httpx2.Request] = []
    site_answers(monkeypatch, page(), sent)
    fetched({"url": "https://docs.python.org"})
    assert str(sent[0].url) == "https://[2a04:4e42::223]/"


def test_nothing_is_sent_to_a_private_address(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lookup_gives(monkeypatch, ["192.168.1.1"])
    sent: list[httpx2.Request] = []
    site_answers(monkeypatch, page(), sent)
    assert refusal({"url": PAGE_URL}) == PRIVATE
    assert sent == []


def test_a_redirect_is_not_followed(monkeypatch: pytest.MonkeyPatch) -> None:
    lookup_gives(monkeypatch, [PUBLIC])
    sent: list[httpx2.Request] = []

    def moved(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(301, headers={"location": "/3/library/"})

    site_answers(monkeypatch, moved, sent)
    assert refusal({"url": "https://docs.python.org/3/library"}) == (
        "the page has moved to https://docs.python.org/3/library/; call"
        " fetch_page again with that address if it is on an allowed site"
    )
    assert len(sent) == 1


def test_a_redirect_to_a_private_address_never_reaches_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lookup_gives(monkeypatch, [PUBLIC])
    sent: list[httpx2.Request] = []

    def moved(request: httpx2.Request) -> httpx2.Response:
        location = "http://192.168.1.1/admin"
        return httpx2.Response(302, headers={"location": location})

    site_answers(monkeypatch, moved, sent)
    assert "moved to http://192.168.1.1/admin" in refusal({"url": PAGE_URL})
    # The model's next request, for that address, is refused before any
    # look-up.
    with pytest.raises(ValueError):
        fetched({"url": "http://192.168.1.1/admin"})
    assert len(sent) == 1


def test_a_page_that_is_not_there_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lookup_gives(monkeypatch, [PUBLIC])
    site_answers(monkeypatch, lambda request: httpx2.Response(404))
    assert refusal({"url": PAGE_URL}) == (
        "the site answered 404, so there is no page to read; tell the user,"
        " instead of calling again"
    )


def test_a_page_that_is_not_text_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lookup_gives(monkeypatch, [PUBLIC])

    def picture(request: httpx2.Request) -> httpx2.Response:
        headers = {"content-type": "image/png"}
        return httpx2.Response(200, headers=headers, content=b"\x89PNG")

    site_answers(monkeypatch, picture)
    assert refusal({"url": PAGE_URL}) == (
        "the page is image/png, not text; tell the user fetch_page cannot"
        " read it"
    )


def test_a_page_of_exactly_the_limit_is_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lookup_gives(monkeypatch, [PUBLIC])
    site_answers(monkeypatch, page("a" * MAX_PAGE_BYTES))
    assert len(fetched({"url": PAGE_URL})) == MAX_PAGE_BYTES


def chunks_of(size: int, chunk: int, read: list[int]) -> Iterator[bytes]:
    while size > 0:
        read.append(min(chunk, size))
        yield b"a" * min(chunk, size)
        size -= chunk


def test_a_page_over_the_limit_is_refused_while_it_arrives(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lookup_gives(monkeypatch, [PUBLIC])
    read: list[int] = []

    def endless(request: httpx2.Request) -> httpx2.Response:
        # A site that states no size and keeps sending.
        return httpx2.Response(
            200,
            headers={"content-type": "text/plain"},
            content=chunks_of(10 * MAX_PAGE_BYTES, 10_000, read),
        )

    site_answers(monkeypatch, endless)
    assert refusal({"url": PAGE_URL}) == TOO_LARGE_PAGE
    # It stopped reading just past the limit, not at the end.
    assert sum(read) == MAX_PAGE_BYTES + 10_000


def test_a_stated_size_over_the_limit_is_refused_before_reading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lookup_gives(monkeypatch, [PUBLIC])
    read: list[int] = []

    def large(request: httpx2.Request) -> httpx2.Response:
        headers = {"content-type": "text/plain", "content-length": "2000000"}
        content = chunks_of(2_000_000, 10_000, read)
        return httpx2.Response(200, headers=headers, content=content)

    site_answers(monkeypatch, large)
    assert refusal({"url": PAGE_URL}) == TOO_LARGE_PAGE
    assert read == []


def test_a_stated_size_that_lies_is_still_counted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lookup_gives(monkeypatch, [PUBLIC])
    read: list[int] = []

    def lies(request: httpx2.Request) -> httpx2.Response:
        headers = {"content-type": "text/plain", "content-length": "500"}
        content = chunks_of(10 * MAX_PAGE_BYTES, 10_000, read)
        return httpx2.Response(200, headers=headers, content=content)

    site_answers(monkeypatch, lies)
    assert refusal({"url": PAGE_URL}) == TOO_LARGE_PAGE
    assert sum(read) == MAX_PAGE_BYTES + 10_000


def test_a_site_that_does_not_answer_in_time_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lookup_gives(monkeypatch, [PUBLIC])

    def silent(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ReadTimeout("The read operation timed out")

    site_answers(monkeypatch, silent)
    assert refusal({"url": PAGE_URL}) == TOO_SLOW
    assert "timed out" not in TOO_SLOW


def test_a_slow_page_stops_at_the_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lookup_gives(monkeypatch, [PUBLIC])
    # Each read of the clock is 3 seconds later: a site that sends a
    # little at a time, never slow enough for one wait to time out.
    readings = count(start=0.0, step=3.0)
    monkeypatch.setattr("askkit.fetch.perf_counter", lambda: next(readings))
    read: list[int] = []

    def trickle(request: httpx2.Request) -> httpx2.Response:
        headers = {"content-type": "text/plain"}
        content = chunks_of(50_000, 1_000, read)
        return httpx2.Response(200, headers=headers, content=content)

    site_answers(monkeypatch, trickle)
    assert refusal({"url": PAGE_URL}) == TOO_SLOW
    assert len(read) == 4


def test_a_site_that_cannot_be_reached_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lookup_gives(monkeypatch, [PUBLIC])

    def unreachable(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ConnectError("[Errno 61] Connection refused")

    site_answers(monkeypatch, unreachable)
    assert refusal({"url": PAGE_URL}) == (
        "could not connect to docs.python.org; do not call it again now:"
        " tell the user the page could not be fetched"
    )


@pytest.mark.parametrize(
    "tool_input", [{}, {"url": 3}, {"url": PAGE_URL, "max_bytes": 10}]
)
def test_a_wrong_input_is_refused(tool_input: dict[str, object]) -> None:
    assert refusal(tool_input) == (
        "fetch_page takes exactly url, as text; call it again with the"
        " page's address"
    )


# In AskKit


def test_run_tool_fetches_from_the_sites_it_is_given(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lookup_gives(monkeypatch, [PUBLIC])
    site_answers(monkeypatch, page())
    request = call("fetch_page", {"url": PAGE_URL})
    assert run_tool(request, SITES)["content"] == "<p>A page</p>"
    assert run_tool(request)["content"] == NOT_ALLOWED


def test_askkit_ask_offers_the_fetch_only_with_a_site(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    sent: list[dict[str, Any]] = []
    use_answers(monkeypatch, sent, [ANSWERED, ANSWERED])
    main(["ask", "Hi"])
    main(["ask", "Hi", "--allow-site", "docs.python.org"])
    assert sent[0]["tools"] == [CLOCK, CALCULATOR]
    assert sent[1]["tools"] == [CLOCK, CALCULATOR, FETCH]


def test_askkit_ask_shows_a_fetched_page_as_its_size(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    lookup_gives(monkeypatch, [PUBLIC])
    site_answers(monkeypatch, page("x" * 36_415))
    sent: list[dict[str, Any]] = []
    asked = asked_for("fetch_page", {"url": PAGE_URL})
    use_answers(monkeypatch, sent, [asked, ANSWERED])
    main(["ask", "Read it", "--allow-site", "docs.python.org"])
    result = sent[1]["messages"][2]["content"][0]
    assert result["content"] == "x" * 36_415
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == "Tool:          fetch_page: 36,415 characters"


def test_askkit_ask_refuses_a_private_address(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    lookup_gives(monkeypatch, ["192.168.1.1"])
    sent: list[dict[str, Any]] = []
    asked = asked_for("fetch_page", {"url": PAGE_URL})
    use_answers(monkeypatch, sent, [asked, ANSWERED])
    main(["ask", "Read it", "--allow-site", "docs.python.org"])
    result = sent[1]["messages"][2]["content"][0]
    assert result["is_error"] is True
    assert result["content"] == PRIVATE
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == f"Tool error:    fetch_page: {PRIVATE}"
