import ipaddress
import socket
from time import perf_counter
from urllib.parse import urljoin, urlsplit

import httpx2
from anthropic.types import ToolParam

# AskKit makes the request, from your computer, so a fetch can reach
# whatever your computer can: your router, a service on your own
# machine, a server at work. The model only asks; these checks decide.
FETCH: ToolParam = {
    "name": "fetch_page",
    "description": (
        "Downloads one web page and returns its text, up to 100,000"
        " bytes. Call it when the user gives you a web address, or asks"
        " what a page says, such as a page of the Python documentation."
        " It fetches only https addresses on the sites the user allowed;"
        " never call it with an address you made up. If a fetch is"
        " refused, tell the user why; never try another address to get"
        " round it."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": (
                    "The page's full address, starting with https://,"
                    " such as https://docs.python.org/3/library/"
                    "secrets.html."
                ),
            },
        },
        "required": ["url"],
        "additionalProperties": False,
    },
    "strict": True,
}

# A larger page is refused: its text goes into the context, is paid for
# as input, and is sent again with every later round of the loop.
MAX_PAGE_BYTES = 100_000
# The longest AskKit waits for any one thing from the site.
WAIT_SECONDS = 5.0
# The longest the whole download may take, first byte to last.
FETCH_SECONDS = 10.0

TOO_LARGE_PAGE = (
    f"the page is larger than {MAX_PAGE_BYTES:,} bytes, the most"
    " fetch_page reads; tell the user it is too large, instead of"
    " calling again"
)
TOO_SLOW = (
    f"the site did not send the page within {FETCH_SECONDS:.0f} seconds;"
    " do not call it again now: tell the user the page could not be"
    " fetched"
)


def allowed_host(url: str, sites: tuple[str, ...]) -> str:
    # The site the address names, if you allowed it. Raises ValueError,
    # with words for the model, for any other address.
    parts = urlsplit(url)
    if parts.scheme != "https":
        raise ValueError(
            "fetch_page fetches only https:// addresses; call it again"
            " with the page's https:// address"
        )
    host = parts.hostname
    if host is None or host not in sites:
        raise ValueError(
            "fetch_page fetches only from the sites the user allowed, and"
            " this address is not on one of them; do not try other"
            " addresses: tell the user they can allow the site with"
            " --allow-site"
        )
    # Only the site's name: no user name, password or port before or
    # after it.
    if parts.netloc.lower() not in (host, f"{host}:443"):
        raise ValueError(
            "the address must name the site alone, with no user name,"
            " password or port; call it again with the plain address"
        )
    return host


def addresses_of(host: str) -> list[str]:
    # Every address the name leads to, as this computer looks it up.
    found = socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM)
    addresses: list[str] = []
    for entry in found:
        addresses.append(str(entry[4][0]))
    return addresses


def public_address(host: str) -> str:
    # An address of the site that is on the public internet. Raises
    # ValueError if the name leads nowhere, or anywhere private.
    try:
        addresses = addresses_of(host)
    except OSError:
        addresses = []
    if not addresses:
        raise ValueError(
            f"{host} could not be found; tell the user, instead of calling"
            " again"
        )
    # Every address is checked, not only the first: a name can lead to
    # several, and one private one is enough to refuse it.
    for address in addresses:
        if not ipaddress.ip_address(address).is_global:
            raise ValueError(
                f"{host} leads to a private network address, which"
                " fetch_page never reaches; do not call it again: tell the"
                " user the page could not be fetched"
            )
    return addresses[0]


def open_client() -> httpx2.Client:
    # Redirects are not followed: a new address comes back to the model,
    # and its next request passes every check again. No proxy and no
    # saved passwords from your environment go along.
    return httpx2.Client(
        timeout=WAIT_SECONDS, follow_redirects=False, trust_env=False
    )


def fetch_page(tool_input: dict[str, object], sites: tuple[str, ...]) -> str:
    # The page's text. Raises ValueError, saying what is wrong and what to
    # do instead, for anything it refuses or cannot fetch.
    url = tool_input.get("url")
    if set(tool_input) != {"url"} or not isinstance(url, str):
        raise ValueError(
            "fetch_page takes exactly url, as text; call it again with the"
            " page's address"
        )
    host = allowed_host(url, sites)
    address = public_address(host)
    parts = urlsplit(url)
    path = parts.path or "/"
    if parts.query:
        path += "?" + parts.query
    if ":" in address:
        # An IPv6 address goes in square brackets in a URL.
        address = f"[{address}]"
    # Connect to the address just checked, never to a second look-up of
    # the name, while the site still sees its own name: in the Host
    # header, and in the TLS handshake that checks its certificate.
    target = f"https://{address}{path}"
    started = perf_counter()
    try:
        with open_client() as client:
            with client.stream(
                "GET",
                target,
                headers={"Host": host},
                extensions={"sni_hostname": host},
            ) as response:
                if response.is_redirect:
                    moved_to = urljoin(url, response.headers["location"])
                    raise ValueError(
                        f"the page has moved to {moved_to}; call fetch_page"
                        " again with that address if it is on an allowed"
                        " site"
                    )
                if response.status_code != 200:
                    raise ValueError(
                        f"the site answered {response.status_code}, so"
                        " there is no page to read; tell the user, instead"
                        " of calling again"
                    )
                kind = response.headers.get("content-type", "")
                if not kind.startswith(("text/", "application/json")):
                    raise ValueError(
                        f"the page is {kind or 'of no stated type'}, not"
                        " text; tell the user fetch_page cannot read it"
                    )
                # The size the site states is checked first, but the bytes
                # are counted anyway: a site can state any size.
                stated = response.headers.get("content-length", "")
                if stated.isdecimal() and int(stated) > MAX_PAGE_BYTES:
                    raise ValueError(TOO_LARGE_PAGE)
                body = b""
                for chunk in response.iter_bytes():
                    body += chunk
                    if len(body) > MAX_PAGE_BYTES:
                        raise ValueError(TOO_LARGE_PAGE)
                    if perf_counter() - started > FETCH_SECONDS:
                        raise ValueError(TOO_SLOW)
                return body.decode(response.encoding or "utf-8", "replace")
    except httpx2.TimeoutException:
        # The library's own words stay out: they were written for you.
        raise ValueError(TOO_SLOW) from None
    except httpx2.TransportError:
        raise ValueError(
            f"could not connect to {host}; do not call it again now: tell"
            " the user the page could not be fetched"
        ) from None
