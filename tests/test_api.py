"""HTTP contract tests for the local grounded RAG API."""

from __future__ import annotations

import io
import json
import logging
import logging.config
import re
import sqlite3
from contextlib import closing
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from uvicorn.config import LOGGING_CONFIG

from personal_knowledge_agent.api import create_app
from personal_knowledge_agent.application.ingest import IngestService
from personal_knowledge_agent.domain.errors import LanguageModelError
from personal_knowledge_agent.domain.models import (
    AnswerStatus,
    Evidence,
    GroundedAnswer,
    SearchMode,
)
from personal_knowledge_agent.infrastructure.markdown_chunker import MarkdownChunker
from personal_knowledge_agent.infrastructure.markdown_loader import MarkdownLoader
from personal_knowledge_agent.infrastructure.sqlite_keyword_index import (
    SQLiteKeywordIndex,
)
from tests.helpers import FakeEmbedder

ROOT = Path(__file__).resolve().parents[1]
API_LOGGER_NAME = "uvicorn.error.personal_knowledge_agent.api"


def test_readme_documents_local_api_request_diagnostics() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert "X-Request-ID" in readme
    assert "api.request.completed" in readme
    assert "question text" in readme


def _request_record(caplog: pytest.LogCaptureFixture) -> dict[str, object]:
    records = [record for record in caplog.records if record.name == API_LOGGER_NAME]
    assert len(records) == 1
    decoded = json.loads(records[0].message)
    assert isinstance(decoded, dict)
    return decoded


def test_api_request_log_uses_default_uvicorn_error_logger(tmp_path: Path) -> None:
    database = tmp_path / "demo.db"
    _create_indexed_database(database)
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    logging.config.dictConfig(LOGGING_CONFIG)
    uvicorn_logger = logging.getLogger("uvicorn")
    uvicorn_logger.addHandler(handler)
    try:
        client = TestClient(
            create_app(database=database, environment={"PKA_LLM_API_KEY": "test-key"})
        )
        response = client.get("/health")
    finally:
        uvicorn_logger.removeHandler(handler)

    assert response.status_code == 200
    assert "api.request.completed" in stream.getvalue()


class FakeAnswerer:
    """Return one stable grounded answer without invoking external infrastructure."""

    def answer(
        self,
        question: str,
        *,
        limit: int,
        mode: SearchMode,
    ) -> GroundedAnswer:
        assert question == "What is hybrid retrieval?"
        assert limit == 5
        assert mode is SearchMode.HYBRID
        evidence = Evidence(
            "S1",
            "chunk-1",
            "retrieval.md",
            ("Hybrid retrieval",),
            "Evidence text",
            1.0,
        )
        return GroundedAnswer(
            AnswerStatus.ANSWERED,
            "Hybrid retrieval combines rankings. [S1]",
            ("S1",),
            (evidence,),
            "fake-model",
        )


class FailingAnswerer:
    """Raise a provider-shaped failure without making a real request."""

    def answer(
        self,
        question: str,
        *,
        limit: int,
        mode: SearchMode,
    ) -> GroundedAnswer:
        raise LanguageModelError("provider is unavailable")


def _create_indexed_database(database: Path) -> None:
    with closing(sqlite3.connect(database)) as connection:
        with connection:
            connection.execute("CREATE TABLE chunks (chunk_id TEXT PRIMARY KEY)")
            connection.execute("INSERT INTO chunks VALUES ('chunk-1')")


def test_health_reports_degraded_when_model_configuration_is_missing(
    tmp_path: Path,
) -> None:
    database = tmp_path / "demo.db"
    _create_indexed_database(database)
    client = TestClient(create_app(database=database, environment={}))

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "degraded",
        "knowledge_base_ready": True,
        "model_configured": False,
    }


def test_health_reports_degraded_when_database_has_no_chunks(tmp_path: Path) -> None:
    database = tmp_path / "empty.db"
    with closing(sqlite3.connect(database)):
        pass
    client = TestClient(
        create_app(database=database, environment={"PKA_LLM_API_KEY": "test-key"})
    )

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "degraded",
        "knowledge_base_ready": False,
        "model_configured": True,
    }


def test_health_reports_ready_when_database_and_model_are_available(
    tmp_path: Path,
) -> None:
    database = tmp_path / "demo.db"
    _create_indexed_database(database)
    client = TestClient(
        create_app(database=database, environment={"PKA_LLM_API_KEY": "test-key"})
    )

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ready",
        "knowledge_base_ready": True,
        "model_configured": True,
    }


def test_api_preserves_a_valid_request_id_and_logs_safe_completion(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    database = tmp_path / "demo.db"
    _create_indexed_database(database)
    client = TestClient(
        create_app(database=database, environment={"PKA_LLM_API_KEY": "test-key"})
    )
    caplog.set_level(logging.INFO, logger=API_LOGGER_NAME)
    uvicorn_logger = logging.getLogger("uvicorn")
    uvicorn_logger.addHandler(caplog.handler)

    try:
        response = client.get("/health", headers={"X-Request-ID": "demo-123"})
    finally:
        uvicorn_logger.removeHandler(caplog.handler)

    assert response.headers["x-request-id"] == "demo-123"
    record = _request_record(caplog)
    assert record["event"] == "api.request.completed"
    assert record["request_id"] == "demo-123"
    assert record["method"] == "GET"
    assert record["path"] == "/health"
    assert record["status_code"] == 200
    assert record["outcome"] == "success"
    assert isinstance(record["duration_ms"], float)
    assert record["duration_ms"] >= 0


def test_api_replaces_an_invalid_request_id(tmp_path: Path) -> None:
    database = tmp_path / "demo.db"
    _create_indexed_database(database)
    client = TestClient(
        create_app(database=database, environment={"PKA_LLM_API_KEY": "test-key"})
    )

    response = client.get("/health", headers={"X-Request-ID": "invalid value"})

    assert response.headers["x-request-id"] != "invalid value"
    assert re.fullmatch(r"[0-9a-f]{32}", response.headers["x-request-id"])


def test_api_logs_not_ready_outcome_and_omits_question(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    client = TestClient(
        create_app(
            database=tmp_path / "missing.db",
            environment={"PKA_LLM_API_KEY": "test-key"},
            answer_service_factory=FakeAnswerer,
        )
    )
    caplog.set_level(logging.INFO, logger=API_LOGGER_NAME)
    uvicorn_logger = logging.getLogger("uvicorn")
    uvicorn_logger.addHandler(caplog.handler)
    question = "credential-like-value-must-not-appear"

    try:
        response = client.post("/ask", json={"question": question})
    finally:
        uvicorn_logger.removeHandler(caplog.handler)

    assert response.status_code == 503
    assert _request_record(caplog)["outcome"] == "not_ready"
    assert question not in caplog.text


def test_api_logs_model_error_outcome(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    database = tmp_path / "demo.db"
    _create_indexed_database(database)
    client = TestClient(
        create_app(
            database=database,
            environment={"PKA_LLM_API_KEY": "test-key"},
            answer_service_factory=FailingAnswerer,
        )
    )
    caplog.set_level(logging.INFO, logger=API_LOGGER_NAME)
    uvicorn_logger = logging.getLogger("uvicorn")
    uvicorn_logger.addHandler(caplog.handler)

    try:
        response = client.post("/ask", json={"question": "What is hybrid retrieval?"})
    finally:
        uvicorn_logger.removeHandler(caplog.handler)

    assert response.status_code == 502
    assert _request_record(caplog)["outcome"] == "model_error"


def test_api_adds_request_id_to_validation_errors(tmp_path: Path) -> None:
    database = tmp_path / "demo.db"
    _create_indexed_database(database)
    client = TestClient(
        create_app(database=database, environment={"PKA_LLM_API_KEY": "test-key"})
    )

    response = client.post("/ask", json={"question": "   "})

    assert response.status_code == 422
    assert re.fullmatch(r"[0-9a-f]{32}", response.headers["x-request-id"])


def test_ask_serializes_validated_grounded_citations(tmp_path: Path) -> None:
    database = tmp_path / "demo.db"
    _create_indexed_database(database)
    client = TestClient(
        create_app(
            database=database,
            environment={"PKA_LLM_API_KEY": "test-key"},
            answer_service_factory=FakeAnswerer,
        )
    )

    response = client.post("/ask", json={"question": "What is hybrid retrieval?"})

    assert response.status_code == 200
    assert response.json() == {
        "status": "answered",
        "answer": "Hybrid retrieval combines rankings. [S1]",
        "citations": [
            {
                "id": "S1",
                "source": "retrieval.md",
                "heading_path": ["Hybrid retrieval"],
            }
        ],
        "model_id": "fake-model",
    }


def test_ask_rejects_a_whitespace_only_question(tmp_path: Path) -> None:
    database = tmp_path / "demo.db"
    _create_indexed_database(database)
    client = TestClient(
        create_app(
            database=database,
            environment={"PKA_LLM_API_KEY": "test-key"},
            answer_service_factory=FakeAnswerer,
        )
    )

    response = client.post("/ask", json={"question": "   "})

    assert response.status_code == 422


def test_ask_returns_503_when_the_database_is_not_ready(tmp_path: Path) -> None:
    client = TestClient(
        create_app(
            database=tmp_path / "missing.db",
            environment={"PKA_LLM_API_KEY": "test-key"},
            answer_service_factory=FakeAnswerer,
        )
    )

    response = client.post("/ask", json={"question": "What is hybrid retrieval?"})

    assert response.status_code == 503
    assert response.json() == {"detail": {"code": "knowledge_base_not_ready"}}


def test_ask_returns_503_when_the_model_is_not_configured(tmp_path: Path) -> None:
    database = tmp_path / "demo.db"
    _create_indexed_database(database)
    client = TestClient(
        create_app(
            database=database,
            environment={},
            answer_service_factory=FakeAnswerer,
        )
    )

    response = client.post("/ask", json={"question": "What is hybrid retrieval?"})

    assert response.status_code == 503
    assert response.json() == {"detail": {"code": "model_not_configured"}}


def test_ask_returns_502_for_a_provider_failure(tmp_path: Path) -> None:
    database = tmp_path / "demo.db"
    _create_indexed_database(database)
    client = TestClient(
        create_app(
            database=database,
            environment={"PKA_LLM_API_KEY": "test-key"},
            answer_service_factory=FailingAnswerer,
        )
    )

    response = client.post("/ask", json={"question": "What is hybrid retrieval?"})

    assert response.status_code == 502
    assert response.json() == {"detail": {"code": "model_request_failed"}}


@pytest.mark.parametrize(
    "filename",
    ["retrieval.md", "grounded-answers.md", "prompt-engineering.md"],
)
def test_demo_documents_can_be_ingested(filename: str, tmp_path: Path) -> None:
    source = ROOT / "docs" / "demo-knowledge" / filename
    index = SQLiteKeywordIndex(
        tmp_path / "demo.db",
        FakeEmbedder({}, default_vector=(1.0, 0.0)),
    )

    result = IngestService(MarkdownLoader(), MarkdownChunker(), index).ingest(source)

    assert result.chunk_count >= 2
