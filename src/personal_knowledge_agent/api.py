"""Local HTTP adapter for the grounded RAG workflow."""

from __future__ import annotations

import os
import sqlite3
from collections.abc import Callable, Mapping
from contextlib import closing
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from personal_knowledge_agent.composition import create_grounded_answer_service
from personal_knowledge_agent.domain.errors import (
    ConfigurationError,
    IndexStorageError,
    KnowledgeBaseEmptyError,
    LanguageModelError,
    StructuredOutputError,
)
from personal_knowledge_agent.domain.models import (
    AnswerStatus,
    GroundedAnswer,
    SearchMode,
)
from personal_knowledge_agent.domain.protocols import GroundedQuestionAnswerer
from personal_knowledge_agent.infrastructure.llm_settings import load_llm_settings

DEFAULT_DEMO_DATABASE = Path(".pka/demo-knowledge.db")


class AskRequest(BaseModel):
    """The fixed-scope question accepted by the local API."""

    question: str = Field(min_length=1, max_length=4_000)

    @field_validator("question")
    @classmethod
    def normalize_question(cls, value: str) -> str:
        """Reject questions that become empty after normalization."""
        normalized = value.strip()
        if not normalized:
            raise ValueError("question must not be blank")
        return normalized


class CitationResponse(BaseModel):
    """A safe public projection of one request-local evidence citation."""

    id: str
    source: str
    heading_path: list[str]


class AskResponse(BaseModel):
    """The validated grounded answer exposed over HTTP."""

    status: Literal["answered", "insufficient_evidence"]
    answer: str
    citations: list[CitationResponse]
    model_id: str | None


class HealthResponse(BaseModel):
    """Readiness details that never expose runtime configuration values."""

    status: Literal["ready", "degraded"]
    knowledge_base_ready: bool
    model_configured: bool


def create_app(
    *,
    database: Path = DEFAULT_DEMO_DATABASE,
    environment: Mapping[str, str] | None = None,
    answer_service_factory: Callable[[], GroundedQuestionAnswerer] | None = None,
) -> FastAPI:
    """Create a local-only grounded RAG API with explicit dependencies."""
    selected_environment = os.environ if environment is None else environment
    selected_database = database.expanduser()
    app = FastAPI(title="Personal Knowledge Agent API", version="0.1.0")

    def model_configured() -> bool:
        try:
            load_llm_settings(selected_environment)
        except ConfigurationError:
            return False
        return True

    def knowledge_base_ready() -> bool:
        if not selected_database.is_file():
            return False
        try:
            database_uri = f"{selected_database.resolve().as_uri()}?mode=ro"
            with closing(sqlite3.connect(database_uri, uri=True)) as connection:
                row = connection.execute("SELECT COUNT(*) FROM chunks").fetchone()
        except (OSError, sqlite3.Error):
            return False
        return row is not None and int(row[0]) > 0

    def health_response() -> HealthResponse:
        indexed = knowledge_base_ready()
        configured = model_configured()
        return HealthResponse(
            status="ready" if indexed and configured else "degraded",
            knowledge_base_ready=indexed,
            model_configured=configured,
        )

    def require_answerer() -> GroundedQuestionAnswerer:
        if not knowledge_base_ready():
            raise HTTPException(
                status_code=503,
                detail={"code": "knowledge_base_not_ready"},
            )
        if not model_configured():
            raise HTTPException(
                status_code=503,
                detail={"code": "model_not_configured"},
            )
        if answer_service_factory is not None:
            return answer_service_factory()
        return create_grounded_answer_service(selected_database, selected_environment)

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        """Return readiness without attempting a provider call."""
        return health_response()

    @app.post("/ask", response_model=AskResponse)
    def ask(request: AskRequest) -> AskResponse:
        """Answer a question with fixed hybrid retrieval settings."""
        try:
            answer = require_answerer().answer(
                request.question,
                limit=5,
                mode=SearchMode.HYBRID,
            )
        except HTTPException:
            raise
        except (KnowledgeBaseEmptyError, IndexStorageError) as error:
            raise HTTPException(
                status_code=503,
                detail={"code": "knowledge_base_not_ready"},
            ) from error
        except (LanguageModelError, StructuredOutputError) as error:
            raise HTTPException(
                status_code=502,
                detail={"code": "model_request_failed"},
            ) from error
        return _serialize_answer(answer)

    return app


def _serialize_answer(answer: GroundedAnswer) -> AskResponse:
    """Project validated domain evidence into the public response contract."""
    status: Literal["answered", "insufficient_evidence"]
    if answer.status is AnswerStatus.ANSWERED:
        status = "answered"
    else:
        status = "insufficient_evidence"
    return AskResponse(
        status=status,
        answer=answer.answer,
        citations=[
            CitationResponse(
                id=evidence.citation_id,
                source=evidence.source,
                heading_path=list(evidence.heading_path),
            )
            for evidence in answer.evidence
        ],
        model_id=answer.model_id,
    )
