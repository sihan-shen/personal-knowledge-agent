"""Coordinate conservative request-local recall of confirmed memories."""

from __future__ import annotations

from personal_knowledge_agent.domain.errors import (
    EmbeddingError,
    InvalidMemoryError,
    InvalidQueryError,
    MemoryStorageError,
)
from personal_knowledge_agent.domain.memory import (
    MAX_RECALLED_MEMORIES,
    MemoryEvidence,
    MemoryRecallReport,
    MemoryRecallWarning,
    MemoryRecord,
    MemorySearchResult,
    MemoryStatus,
    MemoryType,
)
from personal_knowledge_agent.domain.protocols import (
    MemoryRelevancePolicy,
    MemorySafetyPolicy,
    MemoryStore,
)

MEMORY_RECALL_CANDIDATE_LIMIT = 12

type _MemorySlot = tuple[MemoryType, str]


class MemoryRecallService:
    """Recall bounded active-memory slots without partial unsafe disclosure."""

    def __init__(
        self,
        store: MemoryStore,
        relevance_policy: MemoryRelevancePolicy,
        safety_policy: MemorySafetyPolicy,
    ) -> None:
        self._store = store
        self._relevance_policy = relevance_policy
        self._safety_policy = safety_policy

    def recall(self, query: str, *, limit: int = 3) -> MemoryRecallReport:
        """Return safe relevant memory context and typed local warnings."""
        if not query.strip():
            raise InvalidQueryError("Memory recall query must not be blank")
        if not 1 <= limit <= MAX_RECALLED_MEMORIES:
            raise InvalidQueryError("Memory recall limit must be between 1 and 3")

        try:
            candidates = self._store.search(
                query,
                limit=MEMORY_RECALL_CANDIDATE_LIMIT,
            )
        except EmbeddingError:
            return MemoryRecallReport((), (MemoryRecallWarning.EMBEDDING_ERROR,))
        except MemoryStorageError:
            return MemoryRecallReport((), (MemoryRecallWarning.STORAGE_ERROR,))
        except InvalidMemoryError:
            return MemoryRecallReport((), ())
        if not candidates:
            return MemoryRecallReport((), ())

        try:
            active = self._store.list_memories(status=MemoryStatus.ACTIVE)
        except MemoryStorageError:
            return MemoryRecallReport((), (MemoryRecallWarning.STORAGE_ERROR,))

        return self._select(query, candidates, active, limit=limit)

    def _select(
        self,
        query: str,
        candidates: list[MemorySearchResult],
        active: list[MemoryRecord],
        *,
        limit: int,
    ) -> MemoryRecallReport:
        slots: dict[_MemorySlot, list[MemoryRecord]] = {}
        active_ids: set[str] = set()
        for memory in active:
            slot = (memory.memory_type, memory.memory_key)
            slots.setdefault(slot, []).append(memory)
            active_ids.add(memory.memory_id)

        candidate_scores = {item.memory.memory_id: item.score for item in candidates}
        selected: list[tuple[MemoryRecord, float]] = []
        considered: set[_MemorySlot] = set()
        warnings: set[MemoryRecallWarning] = set()

        for candidate in candidates:
            slot = (candidate.memory.memory_type, candidate.memory.memory_key)
            if slot in considered or not self._relevance_policy.allows(
                query, candidate
            ):
                continue
            considered.add(slot)
            peers = slots.get(slot, [])
            if candidate.memory.memory_id not in active_ids or not peers:
                return MemoryRecallReport((), (MemoryRecallWarning.STORAGE_ERROR,))
            peers = sorted(peers, key=lambda item: (item.created_at, item.memory_id))
            if len(peers) > MAX_RECALLED_MEMORIES:
                warnings.add(MemoryRecallWarning.AMBIGUOUS_MEMORY_SLOT)
                continue
            if not all(
                self._safety_policy.allows(
                    peer.memory_type.value,
                    peer.memory_key,
                    peer.content,
                )
                for peer in peers
            ):
                warnings.add(MemoryRecallWarning.UNSAFE_MEMORY)
                continue
            if len(selected) + len(peers) > limit:
                continue
            selected.extend(
                (peer, candidate_scores.get(peer.memory_id, candidate.score))
                for peer in peers
            )
            if len(selected) == limit:
                break

        evidence = tuple(
            MemoryEvidence(f"M{index}", memory, score)
            for index, (memory, score) in enumerate(selected, start=1)
        )
        ordered_warnings = tuple(
            warning for warning in MemoryRecallWarning if warning in warnings
        )
        return MemoryRecallReport(evidence, ordered_warnings)
