"""HTTP contract tests for the local grounded RAG API."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from fastapi.testclient import TestClient

from personal_knowledge_agent.api import create_app
from personal_knowledge_agent.domain.errors import LanguageModelError
from personal_knowledge_agent.domain.models import (
    AnswerStatus,
    Evidence,
    GroundedAnswer,
    SearchMode,
)


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
    with sqlite3.connect(database) as connection:
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
    with sqlite3.connect(database):
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
