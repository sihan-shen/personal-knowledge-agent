"""Tests for the structured language-model memory extractor adapter."""

import pytest

from personal_knowledge_agent.domain.errors import (
    InvalidMemoryError,
    InvalidMemorySchemaError,
    LanguageModelError,
)
from personal_knowledge_agent.domain.memory import (
    MAX_MEMORY_EXTRACTION_INPUT_LENGTH,
    MemoryProposal,
    MemoryType,
)
from personal_knowledge_agent.domain.models import ModelOutput
from personal_knowledge_agent.infrastructure.llm_memory_extractor import (
    LLMMemoryExtractor,
)
from personal_knowledge_agent.infrastructure.memory_prompt_builder import (
    build_memory_extraction_messages,
)
from tests.helpers import FakeLanguageModel


def test_extractor_forwards_builder_messages_and_returns_parser_tuple() -> None:
    model = FakeLanguageModel(
        ModelOutput(
            "extractor",
            '{"candidates":[{"type":"goal","key":"goal.ai",'
            '"content":"Build AI systems"}]}',
            "stop",
        )
    )

    result = LLMMemoryExtractor(model).extract("My goal is AI")

    assert result == (MemoryProposal(MemoryType.GOAL, "goal.ai", "Build AI systems"),)
    assert model.calls == [build_memory_extraction_messages("My goal is AI")]


@pytest.mark.parametrize(
    "message",
    ["", "   ", "x" * (MAX_MEMORY_EXTRACTION_INPUT_LENGTH + 1)],
)
def test_extractor_rejects_invalid_message_before_model_call(message: str) -> None:
    model = FakeLanguageModel(RuntimeError("must not be called"))

    with pytest.raises(InvalidMemoryError):
        LLMMemoryExtractor(model).extract(message)

    assert model.calls == []


def test_extractor_propagates_language_model_error() -> None:
    model = FakeLanguageModel(LanguageModelError("provider unavailable"))

    with pytest.raises(LanguageModelError, match="provider unavailable"):
        LLMMemoryExtractor(model).extract("I prefer concise answers")


def test_extractor_propagates_invalid_memory_schema_error() -> None:
    model = FakeLanguageModel(ModelOutput("extractor", "not JSON", "stop"))

    with pytest.raises(InvalidMemorySchemaError):
        LLMMemoryExtractor(model).extract("I prefer concise answers")


def test_extractor_propagates_unexpected_error() -> None:
    model = FakeLanguageModel(RuntimeError("programming failure"))

    with pytest.raises(RuntimeError, match="programming failure"):
        LLMMemoryExtractor(model).extract("I prefer concise answers")
