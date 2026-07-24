"""Unit tests for safe, single-page static HTML loading."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from email.message import Message
from io import BytesIO
from urllib.error import URLError
from urllib.request import Request

import pytest

from personal_knowledge_agent.domain.errors import (
    DocumentReadError,
    EmptyDocumentError,
    UnsupportedDocumentError,
)
from personal_knowledge_agent.domain.models import make_document_id
from personal_knowledge_agent.infrastructure.web_loader import WebLoader


class FakeResponse:
    """Minimal deterministic HTTP response."""

    def __init__(
        self,
        body: bytes,
        *,
        status: int = 200,
        content_type: str = "text/html; charset=utf-8",
    ) -> None:
        self._body = BytesIO(body)
        self.status = status
        self.headers = Message()
        self.headers["Content-Type"] = content_type

    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, *_: object) -> None:
        self._body.close()

    def read(self, size: int = -1) -> bytes:
        return self._body.read(size)


Resolver = Callable[[str], Sequence[str]]
Opener = Callable[[Request, float], FakeResponse]


def _loader(
    response: FakeResponse | Exception | None = None,
    *,
    resolver: Resolver | None = None,
    maximum_bytes: int = 1_024,
) -> tuple[WebLoader, list[Request]]:
    requests: list[Request] = []

    def open_request(request: Request, timeout: float) -> FakeResponse:
        assert timeout > 0
        requests.append(request)
        if isinstance(response, Exception):
            raise response
        assert response is not None
        return response

    return (
        WebLoader(
            resolver=resolver or (lambda _: ("93.184.216.34",)),
            opener=open_request,
            maximum_bytes=maximum_bytes,
        ),
        requests,
    )


@pytest.mark.parametrize(
    "url",
    [
        "ftp://public.example/page",
        "file:///etc/passwd",
        "https://user:pass@public.example/",
    ],
)
def test_rejects_unsupported_or_credentialed_urls_before_network_access(
    url: str,
) -> None:
    loader, requests = _loader(FakeResponse(b"<p>unused</p>"))

    with pytest.raises(UnsupportedDocumentError):
        loader.load(url)

    assert requests == []


def test_rejects_private_resolved_address_before_network_access() -> None:
    loader, requests = _loader(
        FakeResponse(b"<p>unused</p>"), resolver=lambda _: ("127.0.0.1",)
    )

    with pytest.raises(DocumentReadError, match="safe public"):
        loader.load("https://public.example/page")

    assert requests == []


def test_maps_dns_failure_to_safe_document_read_error() -> None:
    def fail_resolution(_: str) -> Sequence[str]:
        raise OSError("private resolver detail")

    loader, _ = _loader(FakeResponse(b"<p>unused</p>"), resolver=fail_resolution)

    with pytest.raises(DocumentReadError, match="resolve"):
        loader.load("https://public.example/page")


@pytest.mark.parametrize(
    "response",
    [
        FakeResponse(b"failure", status=404),
        FakeResponse(b"not HTML", content_type="application/json"),
    ],
)
def test_rejects_unsuitable_http_responses(response: FakeResponse) -> None:
    loader, _ = _loader(response)

    with pytest.raises(DocumentReadError):
        loader.load("https://public.example/page")


def test_rejects_oversized_response_without_extracting_content() -> None:
    loader, _ = _loader(FakeResponse(b"x" * 20), maximum_bytes=10)

    with pytest.raises(DocumentReadError, match="size limit"):
        loader.load("https://public.example/page")


def test_maps_network_open_failure_to_safe_document_read_error() -> None:
    loader, _ = _loader(URLError("private connection detail"))

    with pytest.raises(DocumentReadError, match="fetch"):
        loader.load("https://public.example/page")


def test_extracts_title_and_visible_main_content_as_markdown() -> None:
    html = b"""
    <html><head><title>  RAG Guide  </title><style>hidden style</style></head>
    <body><p>Outside text</p><main><h2>Getting started</h2><p>Use retrieval.</p>
    <ul><li>Keep sources</li></ul><script>ignore this()</script></main></body></html>
    """
    loader, requests = _loader(FakeResponse(html))

    document = loader.load("https://PUBLIC.example/guide#fragment")

    source = "https://public.example/guide"
    assert document.source == source
    assert document.document_id == make_document_id(source)
    assert (
        document.content
        == "# RAG Guide\n\n## Getting started\n\nUse retrieval.\n\nKeep sources"
    )
    assert requests[0].get_header("User-agent") == "personal-knowledge-agent/0.1"


def test_rejects_html_without_visible_text() -> None:
    loader, _ = _loader(
        FakeResponse(b"<html><body><script>ignore()</script></body></html>")
    )

    with pytest.raises(EmptyDocumentError, match="no visible text"):
        loader.load("https://public.example/page")
