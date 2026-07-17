"""Application coordination for explicit long-term memory administration."""

from __future__ import annotations

import builtins

from personal_knowledge_agent.domain.errors import (
    InvalidMemoryError,
    UnsafeMemoryError,
)
from personal_knowledge_agent.domain.memory import (
    MemoryActivationAction,
    MemoryProposal,
    MemoryRecord,
    MemorySearchResult,
    MemoryStatus,
    MemoryType,
)
from personal_knowledge_agent.domain.protocols import MemorySafetyPolicy, MemoryStore


def _require_nonblank(value: str, field_name: str) -> None:
    if not value.strip():
        raise InvalidMemoryError(f"{field_name} must not be blank")


def _validate_activation(
    action: MemoryActivationAction,
    replace_id: str | None,
) -> None:
    if not isinstance(action, MemoryActivationAction):
        raise InvalidMemoryError("action must be a MemoryActivationAction")
    if action is MemoryActivationAction.REPLACE:
        if replace_id is None or not replace_id.strip():
            raise InvalidMemoryError("REPLACE requires a non-blank replace_id")
    elif replace_id is not None:
        raise InvalidMemoryError(f"{action.name} does not accept replace_id")


class MemoryAdminService:
    """Validate explicit memory operations and delegate them to a store."""

    def __init__(
        self,
        store: MemoryStore,
        safety_policy: MemorySafetyPolicy,
    ) -> None:
        self._store = store
        self._safety_policy = safety_policy

    def remember(
        self,
        memory_type: MemoryType,
        memory_key: str,
        content: str,
        *,
        action: MemoryActivationAction = MemoryActivationAction.ACTIVATE,
        replace_id: str | None = None,
    ) -> MemoryRecord:
        """Create one validated, safe active memory explicitly."""
        _validate_activation(action, replace_id)
        try:
            proposal = MemoryProposal(memory_type, memory_key, content)
        except ValueError as error:
            raise InvalidMemoryError(str(error)) from error
        if not self._safety_policy.allows(proposal.memory_key, proposal.content):
            raise UnsafeMemoryError("Memory key or content is unsafe")
        return self._store.remember(
            proposal.memory_type,
            proposal.memory_key,
            proposal.content,
            action=action,
            replace_id=replace_id,
        )

    def list(
        self,
        *,
        status: MemoryStatus | None = None,
        memory_type: MemoryType | None = None,
    ) -> list[MemoryRecord]:
        """Return memories matching the requested lifecycle filters."""
        return self._store.list_memories(status=status, memory_type=memory_type)

    def search(
        self,
        query: str,
        *,
        limit: int = 5,
        memory_type: MemoryType | None = None,
    ) -> builtins.list[MemorySearchResult]:
        """Return ranked active memories within the supported result limit."""
        _require_nonblank(query, "query")
        if not 1 <= limit <= 20:
            raise InvalidMemoryError("Memory search limit must be between 1 and 20")
        return self._store.search(query, limit=limit, memory_type=memory_type)

    def approve(
        self,
        candidate_id: str,
        *,
        action: MemoryActivationAction = MemoryActivationAction.ACTIVATE,
        replace_id: str | None = None,
    ) -> MemoryRecord:
        """Activate one pending memory with explicit conflict behavior."""
        _require_nonblank(candidate_id, "candidate_id")
        _validate_activation(action, replace_id)
        return self._store.approve(
            candidate_id,
            action=action,
            replace_id=replace_id,
        )

    def reject(self, candidate_id: str) -> None:
        """Reject one pending candidate."""
        _require_nonblank(candidate_id, "candidate_id")
        return self._store.reject(candidate_id)

    def forget(self, memory_id: str) -> int:
        """Delete one complete memory replacement chain."""
        _require_nonblank(memory_id, "memory_id")
        return self._store.forget(memory_id)
