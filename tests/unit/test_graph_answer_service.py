"""Tests for explicit graph-grounded answer orchestration."""

from __future__ import annotations

import pytest

from personal_knowledge_agent.application.graph_answer import GraphAnswerService
from personal_knowledge_agent.domain.errors import (
    GraphStorageError,
    InvalidAnswerSchemaError,
    InvalidCitationError,
    InvalidGraphError,
    LanguageModelError,
)
from personal_knowledge_agent.domain.graph import (
    EntityNeighborhood,
    EntitySearchResult,
    EntityType,
    GraphBuildResult,
    GraphEntity,
    GraphNeighbor,
    GraphProvenance,
    RelationType,
    make_entity_id,
)
from personal_knowledge_agent.domain.models import AnswerStatus, ModelOutput
from personal_knowledge_agent.infrastructure.graph_answer_prompt_builder import (
    build_graph_answer_messages,
)
from personal_knowledge_agent.infrastructure.graph_evidence_builder import (
    build_graph_evidence,
)
from tests.helpers import FakeLanguageModel

_LOCAL_REFUSAL = "The knowledge graph does not contain enough relationship evidence."


class FakeGraphStore:
    def __init__(
        self,
        results: list[EntitySearchResult] | Exception,
        neighborhoods: dict[str, EntityNeighborhood | Exception],
    ) -> None:
        self.results = results
        self.neighborhoods = neighborhoods
        self.search_calls: list[tuple[str, int]] = []
        self.neighborhood_calls: list[str] = []

    def search_entities(
        self,
        query: str,
        *,
        limit: int = 10,
    ) -> list[EntitySearchResult]:
        self.search_calls.append((query, limit))
        if isinstance(self.results, Exception):
            raise self.results
        return self.results

    def get_neighborhood(self, entity_id: str) -> EntityNeighborhood:
        self.neighborhood_calls.append(entity_id)
        result = self.neighborhoods[entity_id]
        if isinstance(result, Exception):
            raise result
        return result

    def replace_document(self, *args: object, **kwargs: object) -> GraphBuildResult:
        del args, kwargs
        raise AssertionError("answering must not write graph state")


class RecordingModelFactory:
    def __init__(self, model: FakeLanguageModel) -> None:
        self.model = model
        self.calls = 0

    def __call__(self) -> FakeLanguageModel:
        self.calls += 1
        return self.model


def entity(entity_type: EntityType, name: str) -> GraphEntity:
    return GraphEntity(make_entity_id(entity_type, name), entity_type, name)


def search_result(item: GraphEntity, score: float = 1.0) -> EntitySearchResult:
    return EntitySearchResult(
        item.entity_id,
        item.entity_type,
        item.display_name,
        score,
    )


def neighborhood(
    selected: GraphEntity,
    related: GraphEntity,
    quote: str,
    *,
    with_provenance: bool = True,
) -> EntityNeighborhood:
    provenance = (
        GraphProvenance(
            "/notes/rag.md",
            f"chunk-{selected.display_name}",
            0,
            ("RAG", "Graph"),
            quote,
        ),
    )
    return EntityNeighborhood(
        selected,
        (
            GraphNeighbor(
                related.entity_id,
                related.display_name,
                related.entity_type,
                RelationType.USES,
                provenance if with_provenance else (),
            ),
        ),
        (),
    )


def make_service(
    results: list[EntitySearchResult] | None = None,
    neighborhoods: dict[str, EntityNeighborhood | Exception] | None = None,
    model_result: ModelOutput | Exception | None = None,
) -> tuple[
    GraphAnswerService,
    FakeGraphStore,
    RecordingModelFactory,
    FakeLanguageModel,
]:
    store = FakeGraphStore(results or [], neighborhoods or {})
    model = FakeLanguageModel(
        model_result
        or ModelOutput(
            "fake-model",
            '{"status":"answered","answer":"Supported. [G1]","citation_ids":["G1"]}',
            "stop",
        )
    )
    factory = RecordingModelFactory(model)
    return GraphAnswerService(store, factory), store, factory, model


@pytest.mark.parametrize(
    ("question", "limit"),
    [("", 3), ("   ", 3), ("question", 0), ("question", 6)],
)
def test_rejects_invalid_input_before_dependencies(
    question: str,
    limit: int,
) -> None:
    service, store, factory, model = make_service()

    with pytest.raises(InvalidGraphError):
        service.answer(question, limit=limit)

    assert store.search_calls == []
    assert store.neighborhood_calls == []
    assert factory.calls == 0
    assert model.calls == []


def test_no_seed_entities_returns_local_refusal_without_model_construction() -> None:
    service, store, factory, model = make_service()

    answer = service.answer("unknown relationship")

    assert answer.status is AnswerStatus.INSUFFICIENT_EVIDENCE
    assert answer.answer == _LOCAL_REFUSAL
    assert answer.citation_ids == ()
    assert answer.evidence == ()
    assert answer.model_id is None
    assert store.search_calls == [("unknown relationship", 3)]
    assert store.neighborhood_calls == []
    assert factory.calls == 0
    assert model.calls == []


def test_seed_without_provenance_returns_local_refusal_after_one_expansion() -> None:
    rag = entity(EntityType.METHOD, "RAG")
    vectors = entity(EntityType.TECHNOLOGY, "Vector Retrieval")
    rag_neighborhood = neighborhood(
        rag,
        vectors,
        "unused",
        with_provenance=False,
    )
    service, store, factory, model = make_service(
        [search_result(rag)],
        {rag.entity_id: rag_neighborhood},
    )

    answer = service.answer("How is RAG related to vectors?")

    assert answer.status is AnswerStatus.INSUFFICIENT_EVIDENCE
    assert answer.answer == _LOCAL_REFUSAL
    assert answer.citation_ids == ()
    assert answer.evidence == ()
    assert answer.model_id is None
    assert store.neighborhood_calls == [rag.entity_id]
    assert factory.calls == 0
    assert model.calls == []


def test_expands_ranked_seeds_once_then_lazily_generates_a_validated_answer() -> None:
    rag = entity(EntityType.METHOD, "RAG")
    vectors = entity(EntityType.TECHNOLOGY, "Vector Retrieval")
    agents = entity(EntityType.CONCEPT, "Agents")
    rag_neighborhood = neighborhood(rag, vectors, "RAG uses vector retrieval.")
    agents_neighborhood = neighborhood(agents, rag, "Agents use RAG.")
    results = [search_result(rag, 2.0), search_result(agents, 1.0)]
    neighborhoods = {
        rag.entity_id: rag_neighborhood,
        agents.entity_id: agents_neighborhood,
    }
    service, store, factory, model = make_service(results, neighborhoods)
    question = "How does RAG support my agents? Include every relevant link."

    answer = service.answer(question, limit=2)

    expected_evidence = build_graph_evidence((rag_neighborhood, agents_neighborhood))
    expected_messages = build_graph_answer_messages(question, expected_evidence)
    assert store.search_calls == [(question, 2)]
    assert store.neighborhood_calls == [rag.entity_id, agents.entity_id]
    assert factory.calls == 1
    assert model.calls == [expected_messages]
    assert answer.status is AnswerStatus.ANSWERED
    assert answer.answer == "Supported. [G1]"
    assert answer.citation_ids == ("G1",)
    assert tuple(item.citation_id for item in answer.evidence) == ("G1",)
    assert answer.model_id == "fake-model"


@pytest.mark.parametrize(
    ("model_result", "error_type", "message"),
    [
        (LanguageModelError("provider failed"), LanguageModelError, "provider failed"),
        (
            ModelOutput("model", "not-json", "stop"),
            InvalidAnswerSchemaError,
            "valid JSON",
        ),
        (
            ModelOutput(
                "model",
                '{"status":"answered","answer":"wrong [G99]","citation_ids":["G99"]}',
                "stop",
            ),
            InvalidCitationError,
            "unknown citation",
        ),
    ],
)
def test_does_not_hide_generation_or_parser_errors(
    model_result: ModelOutput | Exception,
    error_type: type[Exception],
    message: str,
) -> None:
    rag = entity(EntityType.METHOD, "RAG")
    vectors = entity(EntityType.TECHNOLOGY, "Vector Retrieval")
    service, store, factory, model = make_service(
        [search_result(rag)],
        {rag.entity_id: neighborhood(rag, vectors, "RAG uses vectors.")},
        model_result,
    )

    with pytest.raises(error_type, match=message) as error_info:
        service.answer("question")

    if isinstance(model_result, Exception):
        assert error_info.value is model_result
    assert store.search_calls == [("question", 3)]
    assert store.neighborhood_calls == [rag.entity_id]
    assert factory.calls == 1
    assert len(model.calls) == 1


@pytest.mark.parametrize("failure_stage", ["search", "neighborhood"])
def test_propagates_graph_storage_errors_without_model_construction(
    failure_stage: str,
) -> None:
    rag = entity(EntityType.METHOD, "RAG")
    error = GraphStorageError(f"{failure_stage} failed")
    if failure_stage == "search":
        store = FakeGraphStore(error, {})
    else:
        store = FakeGraphStore(
            [search_result(rag)],
            {rag.entity_id: error},
        )
    model = FakeLanguageModel(
        ModelOutput(
            "fake-model",
            '{"status":"answered","answer":"Supported. [G1]","citation_ids":["G1"]}',
            "stop",
        )
    )
    factory = RecordingModelFactory(model)
    service = GraphAnswerService(store, factory)

    with pytest.raises(GraphStorageError) as error_info:
        service.answer("question")

    assert error_info.value is error
    assert factory.calls == 0
    assert model.calls == []
