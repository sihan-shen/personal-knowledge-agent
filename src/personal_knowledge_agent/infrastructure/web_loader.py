"""Safely load one public, static HTML page."""

from __future__ import annotations

import socket
from collections.abc import Callable, Sequence
from html.parser import HTMLParser
from ipaddress import ip_address
from typing import Protocol, Self, cast
from urllib.error import HTTPError, URLError
from urllib.parse import SplitResult, urlsplit, urlunsplit
from urllib.request import HTTPRedirectHandler, OpenerDirector, Request, build_opener

from personal_knowledge_agent.domain.errors import (
    DocumentReadError,
    EmptyDocumentError,
    UnsupportedDocumentError,
)
from personal_knowledge_agent.domain.models import Document, make_document_id

_USER_AGENT = "personal-knowledge-agent/0.1"
_TIMEOUT_SECONDS = 10.0
_MAXIMUM_BYTES = 2_000_000
_IGNORED_TAGS = {"script", "style", "noscript", "svg", "template"}
_BLOCK_TAGS = {"h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "pre", "blockquote"}


class _Response(Protocol):
    """The small response surface needed by the loader."""

    status: int
    headers: object

    def __enter__(self) -> Self: ...

    def __exit__(self, *args: object) -> None: ...

    def read(self, size: int = -1) -> bytes: ...


Resolver = Callable[[str], Sequence[str]]
Opener = Callable[[Request, float], _Response]


class _NoRedirectHandler(HTTPRedirectHandler):
    """Turn every redirect into a normal HTTP error rather than following it."""

    def redirect_request(self, *args: object, **kwargs: object) -> None:
        return None


class _HtmlTextExtractor(HTMLParser):
    """Extract title and visible block text from a page without dependencies."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.title: str | None = None
        self._title_parts: list[str] = []
        self._in_title = False
        self._ignored_depth = 0
        self._body_depth = 0
        self._preferred_depth = 0
        self._preferred_seen = False
        self._blocks: list[tuple[bool, str, list[str]]] = []
        self._active_blocks: list[int] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        del attrs
        tag = tag.lower()
        if tag in _IGNORED_TAGS:
            self._ignored_depth += 1
            return
        if self._ignored_depth:
            return
        if tag == "title":
            self._in_title = True
            self._title_parts = []
            return
        if tag == "body":
            self._body_depth += 1
        if tag in {"main", "article"}:
            self._preferred_seen = True
            self._preferred_depth += 1
        if tag in _BLOCK_TAGS and self._is_visible_content():
            self._blocks.append((self._preferred_depth > 0, tag, []))
            self._active_blocks.append(len(self._blocks) - 1)

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in _IGNORED_TAGS:
            self._ignored_depth = max(0, self._ignored_depth - 1)
            return
        if self._ignored_depth:
            return
        if tag == "title":
            self._in_title = False
            candidate = _normalize_text("".join(self._title_parts))
            if self.title is None and candidate:
                self.title = candidate
            return
        for position in range(len(self._active_blocks) - 1, -1, -1):
            block_index = self._active_blocks[position]
            if self._blocks[block_index][1] == tag:
                del self._active_blocks[position]
                break
        if tag == "body":
            self._body_depth = max(0, self._body_depth - 1)
        if tag in {"main", "article"}:
            self._preferred_depth = max(0, self._preferred_depth - 1)

    def handle_data(self, data: str) -> None:
        if self._ignored_depth:
            return
        if self._in_title:
            self._title_parts.append(data)
        if self._is_visible_content() and self._active_blocks:
            self._blocks[self._active_blocks[-1]][2].append(data)

    def markdown(self) -> str:
        """Return title plus the preferred visible block sequence."""
        sections: list[str] = []
        if self.title:
            sections.append(f"# {self.title}")
        for preferred, tag, parts in self._blocks:
            if self._preferred_seen and not preferred:
                continue
            text = _normalize_text("".join(parts))
            if not text:
                continue
            if tag.startswith("h"):
                text = f"{'#' * int(tag[1])} {text}"
            sections.append(text)
        return "\n\n".join(sections)

    def _is_visible_content(self) -> bool:
        if self._preferred_seen:
            return self._preferred_depth > 0
        return self._body_depth > 0


def _normalize_text(value: str) -> str:
    return " ".join(value.split())


def _default_resolver(host: str) -> Sequence[str]:
    """Resolve a hostname immediately before opening the single connection."""
    return tuple(
        sorted(
            {
                str(address[4][0])
                for address in socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)
            }
        )
    )


def _default_opener(request: Request, timeout: float) -> _Response:
    opener: OpenerDirector = build_opener(_NoRedirectHandler())
    return cast(_Response, opener.open(request, timeout=timeout))


class WebLoader:
    """Load a public HTTP(S) HTML document with deterministic safety seams."""

    def __init__(
        self,
        *,
        resolver: Resolver = _default_resolver,
        opener: Opener = _default_opener,
        timeout_seconds: float = _TIMEOUT_SECONDS,
        maximum_bytes: int = _MAXIMUM_BYTES,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if maximum_bytes <= 0:
            raise ValueError("maximum_bytes must be positive")
        self._resolver = resolver
        self._opener = opener
        self._timeout_seconds = timeout_seconds
        self._maximum_bytes = maximum_bytes

    def load(self, url: str) -> Document:
        """Fetch one safe public HTML page and return its extracted document."""
        source, parsed = self._validate_url(url)
        self._assert_public_host(parsed.hostname)
        request = Request(source, headers={"User-Agent": _USER_AGENT})
        try:
            response = self._opener(request, self._timeout_seconds)
            with response:
                if not 200 <= response.status < 300:
                    raise DocumentReadError(
                        "Web server returned an unsuccessful status"
                    )
                if not _is_html_response(response.headers):
                    raise DocumentReadError("Web response is not HTML")
                body = response.read(self._maximum_bytes + 1)
        except DocumentReadError:
            raise
        except (HTTPError, URLError, OSError, TimeoutError) as error:
            raise DocumentReadError("Could not fetch web document") from error

        if len(body) > self._maximum_bytes:
            raise DocumentReadError("Web response exceeds the configured size limit")
        try:
            extractor = _HtmlTextExtractor()
            extractor.feed(body.decode(_response_charset(response.headers)))
            extractor.close()
        except (LookupError, UnicodeDecodeError) as error:
            raise DocumentReadError(
                "Web response could not be decoded as HTML"
            ) from error

        content = extractor.markdown()
        if not content.strip():
            raise EmptyDocumentError("Web page has no visible text")
        return Document(make_document_id(source), source, content)

    def _validate_url(self, url: str) -> tuple[str, SplitResult]:
        try:
            parsed = urlsplit(url)
            host = parsed.hostname
            _ = parsed.port
        except ValueError as error:
            raise UnsupportedDocumentError("Web URL is invalid") from error
        if (
            parsed.scheme.lower() not in {"http", "https"}
            or not parsed.netloc
            or host is None
            or parsed.username is not None
            or parsed.password is not None
        ):
            raise UnsupportedDocumentError(
                "Only credential-free absolute HTTP(S) URLs are supported"
            )
        normalized = urlunsplit(
            (
                parsed.scheme.lower(),
                parsed.netloc.lower(),
                parsed.path or "/",
                parsed.query,
                "",
            )
        )
        return normalized, urlsplit(normalized)

    def _assert_public_host(self, host: str | None) -> None:
        assert host is not None
        try:
            addresses = self._resolver(host)
        except (OSError, ValueError) as error:
            raise DocumentReadError("Could not resolve web host") from error
        if not addresses:
            raise DocumentReadError("Could not resolve web host")
        try:
            public = all(ip_address(address).is_global for address in addresses)
        except ValueError as error:
            raise DocumentReadError("Could not resolve web host") from error
        if not public:
            raise DocumentReadError(
                "Web host does not resolve to a safe public address"
            )


def _is_html_response(headers: object) -> bool:
    get_content_type = getattr(headers, "get_content_type", None)
    return callable(get_content_type) and get_content_type().lower() == "text/html"


def _response_charset(headers: object) -> str:
    get_content_charset = getattr(headers, "get_content_charset", None)
    if not callable(get_content_charset):
        return "utf-8"
    return get_content_charset() or "utf-8"
