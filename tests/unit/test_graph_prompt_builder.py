"""Tests for the per-chunk graph extraction prompt builder."""

from __future__ import annotations

import json

from personal_knowledge_agent.domain.graph import (
    EntityType,
    MAX_GRAPH_ENTITIES_PER_CHUNK,
    MAX_GRAPH_EVIDENCE_QUOTE_LENGTH,
    MAX_GRAPH_ENTITY_NAME_LENGTH,
    MAX_GRAPH_RELATIONS_PER_CHUNK,
    RelationType,
)
from personal_knowledge_agent.domain.models import Chunk, MessageRole
from personal_knowledge_agent.infrastructure.graph_prompt_builder import (
    build_graph_extraction_messages,
)


def _make_chunk() -> Chunk:
    return Chunk(
        "c1",
        "doc1",
        0,
        "RAG uses vector retrieval.",
        ("Architecture", "Retrieval"),
    )


class TestPromptMessages:
    def test_returns_two_messages(self) -> None:
        sys_msg, user_msg = build_graph_extraction_messages(_make_chunk())
        assert sys_msg.role is MessageRole.SYSTEM
        assert user_msg.role is MessageRole.USER

    def test_user_message_contains_heading_path(self) -> None:
        _, user_msg = build_graph_extraction_messages(_make_chunk())
        assert '["Architecture", "Retrieval"]' in user_msg.content

    def test_user_message_contains_chunk_text(self) -> None:
        _, user_msg = build_graph_extraction_messages(_make_chunk())
        assert "RAG uses vector retrieval." in user_msg.content

    def test_system_contains_all_entity_types(self) -> None:
        sys_msg, _ = build_graph_extraction_messages(_make_chunk())
        for et in EntityType:
            assert et.value in sys_msg.content

    def test_system_contains_all_relation_types(self) -> None:
        sys_msg, _ = build_graph_extraction_messages(_make_chunk())
        for rt in RelationType:
            assert rt.value in sys_msg.content

    def test_system_contains_limits(self) -> None:
        sys_msg, _ = build_graph_extraction_messages(_make_chunk())
        assert str(MAX_GRAPH_ENTITIES_PER_CHUNK) in sys_msg.content
        assert str(MAX_GRAPH_RELATIONS_PER_CHUNK) in sys_msg.content
        assert str(MAX_GRAPH_ENTITY_NAME_LENGTH) in sys_msg.content
        assert str(MAX_GRAPH_EVIDENCE_QUOTE_LENGTH) in sys_msg.content

    def test_system_mentions_evidence_quote_requirement(self) -> None:
        sys_msg, _ = build_graph_extraction_messages(_make_chunk())
        assert "evidence_quote" in sys_msg.content
        assert "exact" in sys_msg.content.lower()

    def test_system_mentions_local_ids(self) -> None:
        sys_msg, _ = build_graph_extraction_messages(_make_chunk())
        assert "E1" in sys_msg.content or "local_id" in sys_msg.content

    def test_system_rejects_self_loops(self) -> None:
        sys_msg, _ = build_graph_extraction_messages(_make_chunk())
        assert "self-loop" in sys_msg.content or "self" in sys_msg.content.lower()

    def test_privacy_omits_chunk_id(self) -> None:
        sys_msg, user_msg = build_graph_extraction_messages(_make_chunk())
        combined = sys_msg.content + user_msg.content
        assert "c1" not in combined

    def test_privacy_omits_document_id(self) -> None:
        sys_msg, user_msg = build_graph_extraction_messages(_make_chunk())
        combined = sys_msg.content + user_msg.content
        assert "doc1" not in combined

    def test_privacy_omits_source_path(self) -> None:
        sys_msg, user_msg = build_graph_extraction_messages(_make_chunk())
        combined = sys_msg.content + user_msg.content
        assert ".md" not in combined

    def test_privacy_omits_graph_db(self) -> None:
        sys_msg, user_msg = build_graph_extraction_messages(_make_chunk())
        combined = sys_msg.content + user_msg.content
        assert "graph.db" not in combined.lower()
