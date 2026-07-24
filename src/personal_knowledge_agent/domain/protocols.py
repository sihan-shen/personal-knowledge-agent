"""Protocols that define replaceable retrieval components."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Protocol, TypeVar

from personal_knowledge_agent.domain.evaluation import (
    FaithfulnessJudgeRequest,
    JudgeMetricScore,
    ReferenceQualityJudgeRequest,
    ReferenceQualityScores,
)
from personal_knowledge_agent.domain.graph import (
    EntityNeighborhood,
    EntitySearchResult,
    GraphBuildResult,
    GraphChunkExtraction,
    GraphSourceDocument,
)
from personal_knowledge_agent.domain.memory import (
    MemoryActivationAction,
    MemoryProposal,
    MemoryRecallReport,
    MemoryRecord,
    MemorySearchResult,
    MemoryStatus,
    MemoryType,
)
from personal_knowledge_agent.domain.models import (
    ChatMessage,
    Chunk,
    Document,
    Embedding,
    GroundedAnswer,
    MemoryAwareAnswerResult,
    ModelOutput,
    RouteDecision,
    SearchMode,
    SearchResult,
)


class QuestionRouter(Protocol):
    """Choose the safe response path for one user question."""

    def route(self, question: str) -> RouteDecision:
        """Return one validated route decision."""
        ...


class LanguageModel(Protocol):
    """Generate provider-neutral output from an ordered message sequence."""

    def generate(self, messages: Sequence[ChatMessage]) -> ModelOutput:
        """Return one complete model response."""
        ...


DocumentSource = TypeVar("DocumentSource", contravariant=True)


class DocumentLoader(Protocol[DocumentSource]):
    """Load one source into a document."""

    def load(self, source: DocumentSource) -> Document:
        """Load and validate a document."""
        ...


class Chunker(Protocol):
    """Split a document into ordered searchable chunks."""

    def split(self, document: Document) -> list[Chunk]:
        """Create deterministic chunks from a document."""
        ...


class EmbeddingModel(Protocol):
    """Encode documents and questions in a shared vector space."""

    @property
    def model_id(self) -> str:
        """Return the stable identifier for vectors this model creates."""
        ...

    def embed_documents(self, texts: Sequence[str]) -> list[Embedding]:
        """Encode document passages for persistent retrieval."""
        ...

    def embed_query(self, query: str) -> Embedding:
        """Encode one user question for retrieval."""
        ...


class KnowledgeIndex(Protocol):
    """Persist documents and return ranked chunks."""

    def replace_document(
        self,
        document: Document,
        chunks: Sequence[Chunk],
    ) -> bool:
        """Replace a source atomically and report whether it existed."""
        ...

    def search(
        self,
        query: str,
        *,
        limit: int = 5,
        mode: SearchMode = SearchMode.LEXICAL,
    ) -> list[SearchResult]:
        """Return the highest-ranked chunks for a query."""
        ...


class MemoryExtractor(Protocol):
    """Extract normalized memory proposals from one user message."""

    def extract(self, message: str) -> tuple[MemoryProposal, ...]:
        """Return validated candidate memories."""
        ...


class MemorySafetyPolicy(Protocol):
    """Check local text against prohibited memory categories."""

    def allows(self, *values: str) -> bool:
        """Return false when any supplied local text matches a prohibited category."""
        ...


class MemoryRelevancePolicy(Protocol):
    """Decide whether a ranked active memory is relevant enough to disclose."""

    def allows(self, query: str, result: MemorySearchResult) -> bool:
        """Return whether the candidate passes the local relevance gate."""
        ...


class MemoryRecaller(Protocol):
    """Build bounded request-local context from confirmed memories."""

    def recall(self, query: str, *, limit: int = 3) -> MemoryRecallReport:
        """Return accepted memories and safe warnings for one query."""
        ...


class GroundedQuestionAnswerer(Protocol):
    """Answer with document evidence only."""

    def answer(
        self,
        question: str,
        *,
        limit: int = 5,
        mode: SearchMode = SearchMode.HYBRID,
    ) -> GroundedAnswer:
        """Return one validated document-grounded answer."""
        ...


class FaithfulnessJudge(Protocol):
    """Score whether an answer is supported by supplied evidence only."""

    def judge_faithfulness(
        self,
        request: FaithfulnessJudgeRequest,
    ) -> JudgeMetricScore:
        """Return one valid faithfulness score."""
        ...


class ReferenceQualityJudge(Protocol):
    """Score relevance and completeness against supplied reference answers only."""

    def judge_reference_quality(
        self,
        request: ReferenceQualityJudgeRequest,
    ) -> ReferenceQualityScores:
        """Return valid relevance and completeness scores."""
        ...


class MemoryAwareQuestionAnswerer(Protocol):
    """Answer with document and confirmed memory context."""

    def answer(
        self,
        question: str,
        *,
        limit: int = 5,
        mode: SearchMode = SearchMode.HYBRID,
    ) -> MemoryAwareAnswerResult:
        """Return one validated answer and its recall report."""
        ...


class MemoryStore(Protocol):
    """Persist and retrieve long-term memories through their lifecycle."""

    def create_pending(
        self, proposals: Sequence[MemoryProposal], *, source_hash: str
    ) -> tuple[MemoryRecord, ...]:
        """Persist automatic proposals as pending candidates."""
        ...

    def remember(
        self,
        memory_type: MemoryType,
        memory_key: str,
        content: str,
        *,
        action: MemoryActivationAction = MemoryActivationAction.ACTIVATE,
        replace_id: str | None = None,
    ) -> MemoryRecord:
        """Create one explicit active memory."""
        ...

    def approve(
        self,
        candidate_id: str,
        *,
        action: MemoryActivationAction,
        replace_id: str | None = None,
    ) -> MemoryRecord:
        """Activate one pending memory, applying conflict behavior atomically."""
        ...

    def reject(self, candidate_id: str) -> None:
        """Delete one pending candidate and retain only its fingerprint."""
        ...

    def forget(self, memory_id: str) -> int:
        """Delete one memory and report the number of records removed."""
        ...

    def list_memories(
        self,
        *,
        status: MemoryStatus | None = None,
        memory_type: MemoryType | None = None,
    ) -> list[MemoryRecord]:
        """List memories matching optional lifecycle filters."""
        ...

    def search(
        self,
        query: str,
        *,
        limit: int = 5,
        memory_type: MemoryType | None = None,
    ) -> list[MemorySearchResult]:
        """Return ranked active memories matching a query."""
        ...


# ---------------------------------------------------------------------------
# Graph protocols
# ---------------------------------------------------------------------------


class GraphSourceReader(Protocol):
    """Load one indexed source document for building a graph."""

    def load(self, source: Path) -> GraphSourceDocument:
        """Return the current source metadata and ordered chunks."""
        ...


class KnowledgeGraphExtractor(Protocol):
    """Extract validated entities and relations from one chunk."""

    def extract(self, chunk: Chunk) -> GraphChunkExtraction:
        """Return one validated chunk extraction."""
        ...


class KnowledgeGraphStore(Protocol):
    """Persist graph facts and support local queries."""

    def replace_document(
        self,
        source_document: GraphSourceDocument,
        extractions: Sequence[GraphChunkExtraction],
    ) -> GraphBuildResult:
        """Atomically replace one document's graph contribution."""
        ...

    def search_entities(
        self,
        query: str,
        *,
        limit: int = 10,
    ) -> list[EntitySearchResult]:
        """Return BM25-ranked entity matches."""
        ...

    def get_neighborhood(
        self,
        entity_id: str,
    ) -> EntityNeighborhood:
        """Return one entity with its inbound and outbound relations."""
        ...
