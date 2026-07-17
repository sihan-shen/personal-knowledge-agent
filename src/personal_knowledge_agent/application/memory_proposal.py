"""Application coordination for automatic pending-memory proposals."""

from __future__ import annotations

from collections.abc import Callable

from personal_knowledge_agent.domain.errors import (
    EmbeddingError,
    InvalidMemoryError,
    InvalidMemorySchemaError,
    LanguageModelError,
    MemoryStorageError,
)
from personal_knowledge_agent.domain.memory import (
    MAX_MEMORY_EXTRACTION_INPUT_LENGTH,
    MemoryExtractionReport,
    MemoryExtractionSkipReason,
    memory_source_hash,
)
from personal_knowledge_agent.domain.protocols import (
    MemoryExtractor,
    MemorySafetyPolicy,
    MemoryStore,
)


def _skipped(reason: MemoryExtractionSkipReason) -> MemoryExtractionReport:
    return MemoryExtractionReport((), reason)


class MemoryProposalService:
    """Coordinate guarded extraction and atomic pending-memory persistence."""

    def __init__(
        self,
        extractor: MemoryExtractor,
        store: MemoryStore,
        safety_policy: MemorySafetyPolicy,
        prefilter: Callable[[str], bool],
    ) -> None:
        self._extractor = extractor
        self._store = store
        self._safety_policy = safety_policy
        self._prefilter = prefilter

    def propose(self, message: str) -> MemoryExtractionReport:
        """Create pending candidates or return one safe no-op reason."""
        if not message.strip():
            raise InvalidMemoryError("Memory proposal message must not be blank")
        if len(message) > MAX_MEMORY_EXTRACTION_INPUT_LENGTH:
            return _skipped(MemoryExtractionSkipReason.INPUT_TOO_LONG)
        if not self._prefilter(message):
            return _skipped(MemoryExtractionSkipReason.PREFILTER_MISS)
        if not self._safety_policy.allows(message):
            return _skipped(MemoryExtractionSkipReason.SENSITIVE_INPUT)

        try:
            proposals = self._extractor.extract(message)
            if not proposals:
                return _skipped(MemoryExtractionSkipReason.NO_NEW_CANDIDATES)

            safety_values = tuple(
                value
                for proposal in proposals
                for value in (proposal.memory_key, proposal.content)
            )
            if not self._safety_policy.allows(*safety_values):
                return _skipped(MemoryExtractionSkipReason.UNSAFE_OUTPUT)

            candidates = self._store.create_pending(
                proposals,
                source_hash=memory_source_hash(message),
            )
        except LanguageModelError:
            return _skipped(MemoryExtractionSkipReason.PROVIDER_ERROR)
        except InvalidMemorySchemaError:
            return _skipped(MemoryExtractionSkipReason.INVALID_OUTPUT)
        except EmbeddingError:
            return _skipped(MemoryExtractionSkipReason.EMBEDDING_ERROR)
        except MemoryStorageError:
            return _skipped(MemoryExtractionSkipReason.STORAGE_ERROR)

        if not candidates:
            return _skipped(MemoryExtractionSkipReason.NO_NEW_CANDIDATES)
        return MemoryExtractionReport(candidates, None)
