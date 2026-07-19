"""Tests for the LLM knowledge graph extractor adapter."""

from __future__ import annotations

from personal_knowledge_agent.domain.errors import (
    InvalidGraphSchemaError,
    LanguageModelError,
)
from personal_knowledge_agent.domain.graph import (
    EntityType,
    GraphChunkExtraction,
    GraphEntity,
    make_entity_id,
)
from personal_knowledge_agent.domain.models import Chunk, ModelOutput
from personal_knowledge_agent.infrastructure.llm_knowledge_graph_extractor import (
    LLMKnowledgeGraphExtractor,
)
from tests.helpers import FakeLanguageModel


def _make_chunk() -> Chunk:
    return Chunk("c1", "d1", 0, "RAG uses vector retrieval.", ("Architecture",))


class TestLLMExtractor:
    def test_single_call_and_return(self) -> None:
        chunk = _make_chunk()
        valid_json = (
            '{"entities":[{"local_id":"E1","name":"RAG","type":"method"},'
            '{"local_id":"E2","name":"Vector Retrieval","type":"technology"}],'
            '"relations":[{"source_id":"E1","type":"uses","target_id":"E2",'
            '"evidence_quote":"RAG uses vector retrieval."}]}'
        )
        model = FakeLanguageModel(ModelOutput("test-model", valid_json, "stop"))
        extractor = LLMKnowledgeGraphExtractor(model)
        result = extractor.extract(chunk)

        assert len(result.entities) == 2
        assert len(result.relation_evidence) == 1
        assert result.chunk == chunk
        assert len(model.calls) == 1

    def test_propagates_language_model_error(self) -> None:
        chunk = _make_chunk()
        model = FakeLanguageModel(LanguageModelError("Provider error"))
        extractor = LLMKnowledgeGraphExtractor(model)
        try:
            extractor.extract(chunk)
            assert False, "Expected LanguageModelError"
        except LanguageModelError:
            pass
        assert len(model.calls) == 1

    def test_propagates_invalid_schema_error(self) -> None:
        chunk = _make_chunk()
        invalid_json = '{"entities":[],"relations":[],"extra":"x"}'
        model = FakeLanguageModel(ModelOutput("test-model", invalid_json, "stop"))
        extractor = LLMKnowledgeGraphExtractor(model)
        try:
            extractor.extract(chunk)
            assert False, "Expected InvalidGraphSchemaError"
        except InvalidGraphSchemaError:
            pass
        assert len(model.calls) == 1
