"""Tests for the outer grounded-answer composition root."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from personal_knowledge_agent.application.answer import AnswerService
from personal_knowledge_agent.composition import create_grounded_answer_service
from personal_knowledge_agent.domain.models import ModelOutput
from tests.helpers import FakeEmbedder, FakeLanguageModel


def test_create_grounded_answer_service_wires_existing_dependencies(
    tmp_path: Path,
) -> None:
    with (
        patch(
            "personal_knowledge_agent.composition.E5Embedder",
            return_value=FakeEmbedder({}, default_vector=(1.0, 0.0)),
        ),
        patch(
            "personal_knowledge_agent.composition.OpenAICompatibleChatModel",
            return_value=FakeLanguageModel(ModelOutput("fake-model", "{}", "stop")),
        ),
    ):
        service = create_grounded_answer_service(
            tmp_path / "knowledge.db",
            {"PKA_LLM_API_KEY": "test-key"},
        )

    assert isinstance(service, AnswerService)
