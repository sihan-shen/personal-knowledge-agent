"""Command-line interface for Personal Knowledge Agent."""

from __future__ import annotations

import argparse
import os
import re
import shlex
from collections.abc import Sequence
from pathlib import Path

from personal_knowledge_agent.application.agent import AgentService
from personal_knowledge_agent.application.answer import AnswerService
from personal_knowledge_agent.application.ingest import IngestService
from personal_knowledge_agent.application.memory_admin import MemoryAdminService
from personal_knowledge_agent.application.memory_aware_answer import (
    MemoryAwareAnswerService,
)
from personal_knowledge_agent.application.memory_proposal import MemoryProposalService
from personal_knowledge_agent.application.memory_recall import MemoryRecallService
from personal_knowledge_agent.application.search import SearchService
from personal_knowledge_agent.domain.errors import (
    EmbeddingError,
    KnowledgeError,
    MemoryConflictError,
    MemoryStorageError,
)
from personal_knowledge_agent.domain.memory import (
    MAX_MEMORY_EXTRACTION_INPUT_LENGTH,
    MemoryActivationAction,
    MemoryExtractionSkipReason,
    MemoryRecallReport,
    MemoryRecallWarning,
    MemoryRecord,
    MemorySearchResult,
    MemoryStatus,
    MemoryType,
)
from personal_knowledge_agent.domain.models import (
    AgentResponse,
    AnswerStatus,
    GroundedAnswer,
    RouteAction,
    RouteDecision,
    RouterFallbackReason,
    RouteSource,
    SearchMode,
    SearchResult,
)
from personal_knowledge_agent.domain.protocols import MemoryRecaller
from personal_knowledge_agent.infrastructure.e5_embedder import E5Embedder
from personal_knowledge_agent.infrastructure.hybrid_question_router import (
    HybridQuestionRouter,
)
from personal_knowledge_agent.infrastructure.llm_memory_extractor import (
    LLMMemoryExtractor,
)
from personal_knowledge_agent.infrastructure.llm_settings import load_llm_settings
from personal_knowledge_agent.infrastructure.markdown_chunker import MarkdownChunker
from personal_knowledge_agent.infrastructure.markdown_loader import MarkdownLoader
from personal_knowledge_agent.infrastructure.memory_prefilter import (
    should_extract_memory,
)
from personal_knowledge_agent.infrastructure.memory_relevance import (
    ConservativeMemoryRelevancePolicy,
)
from personal_knowledge_agent.infrastructure.memory_safety import (
    LocalMemorySafetyPolicy,
)
from personal_knowledge_agent.infrastructure.openai_compatible_chat_model import (
    OpenAICompatibleChatModel,
)
from personal_knowledge_agent.infrastructure.sqlite_keyword_index import (
    SQLiteKeywordIndex,
)
from personal_knowledge_agent.infrastructure.sqlite_memory_store import (
    SQLiteMemoryStore,
)

_DEFAULT_DATABASE = Path(".pka/knowledge.db")
_DEFAULT_MEMORY_DATABASE = Path(".pka/memory.db")
from personal_knowledge_agent.application.graph_build import GraphBuildService
from personal_knowledge_agent.application.graph_query import GraphQueryService
from personal_knowledge_agent.domain.errors import GraphFeatureError, GraphSourceNotFoundError, InvalidGraphError, GraphEmptyError, GraphEntityNotFoundError
from personal_knowledge_agent.infrastructure.llm_knowledge_graph_extractor import LLMKnowledgeGraphExtractor
from personal_knowledge_agent.infrastructure.sqlite_graph_source_reader import SQLiteGraphSourceReader
from personal_knowledge_agent.infrastructure.sqlite_knowledge_graph_store import SQLiteKnowledgeGraphStore
_DEFAULT_GRAPH_DATABASE = Path(".pka/graph.db")
_SNIPPET_LENGTH = 160
_SILENT_MEMORY_SKIP_REASONS = {
    MemoryExtractionSkipReason.PREFILTER_MISS,
    MemoryExtractionSkipReason.NO_NEW_CANDIDATES,
}
_VISIBLE_MEMORY_SKIP_LABELS = {
    MemoryExtractionSkipReason.INPUT_TOO_LONG: "input too long",
    MemoryExtractionSkipReason.SENSITIVE_INPUT: "sensitive input",
    MemoryExtractionSkipReason.PROVIDER_ERROR: "provider error",
    MemoryExtractionSkipReason.INVALID_OUTPUT: "invalid extractor output",
    MemoryExtractionSkipReason.UNSAFE_OUTPUT: "unsafe extractor output",
    MemoryExtractionSkipReason.EMBEDDING_ERROR: "embedding error",
    MemoryExtractionSkipReason.STORAGE_ERROR: "storage error",
}
_MEMORY_RECALL_WARNING_LABELS = {
    MemoryRecallWarning.EMBEDDING_ERROR: "embedding error",
    MemoryRecallWarning.STORAGE_ERROR: "storage error",
    MemoryRecallWarning.UNSAFE_MEMORY: "unsafe stored memory",
    MemoryRecallWarning.AMBIGUOUS_MEMORY_SLOT: "ambiguous memory slot",
}


class _UnavailableMemoryRecaller:
    """Return one safe warning when optional recall cannot be constructed."""

    def __init__(self, warning: MemoryRecallWarning) -> None:
        self._warning = warning

    def recall(self, query: str, *, limit: int = 3) -> MemoryRecallReport:
        del query, limit
        return MemoryRecallReport((), (self._warning,))


def _search_limit(value: str) -> int:
    try:
        limit = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("limit must be an integer") from error
    if not 1 <= limit <= 20:
        raise argparse.ArgumentTypeError("limit must be between 1 and 20")
    return limit


def _answer_limit(value: str) -> int:
    try:
        limit = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("limit must be an integer") from error
    if not 1 <= limit <= 10:
        raise argparse.ArgumentTypeError("limit must be between 1 and 10")
    return limit


def _add_memory_database_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--memory-db",
        type=Path,
        default=_DEFAULT_MEMORY_DATABASE,
    )


def build_parser() -> argparse.ArgumentParser:
    """Create the parser for supported user commands."""
    parser = argparse.ArgumentParser(prog="pka")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("health", help="Check that the CLI is available.")

    ingest_parser = subparsers.add_parser(
        "ingest",
        help="Index one local Markdown document.",
    )
    ingest_parser.add_argument("path", type=Path)
    ingest_parser.add_argument("--db", type=Path, default=_DEFAULT_DATABASE)

    search_parser = subparsers.add_parser(
        "search",
        help="Search the persistent local knowledge base.",
    )
    search_parser.add_argument("query")
    search_parser.add_argument("--limit", type=_search_limit, default=5)
    search_parser.add_argument(
        "--mode",
        choices=[mode.value for mode in SearchMode],
        default=SearchMode.HYBRID.value,
    )
    search_parser.add_argument("--db", type=Path, default=_DEFAULT_DATABASE)

    ask_parser = subparsers.add_parser(
        "ask",
        help="Answer a question using evidence from the knowledge base.",
    )
    ask_parser.add_argument("query")
    ask_parser.add_argument("--limit", type=_answer_limit, default=5)
    ask_parser.add_argument(
        "--mode",
        choices=[mode.value for mode in SearchMode],
        default=SearchMode.HYBRID.value,
    )
    ask_parser.add_argument("--db", type=Path, default=_DEFAULT_DATABASE)

    agent_parser = subparsers.add_parser(
        "agent",
        help="Choose whether to answer directly or retrieve grounded evidence.",
    )
    agent_parser.add_argument("query")
    agent_parser.add_argument("--limit", type=_answer_limit, default=5)
    agent_parser.add_argument(
        "--mode",
        choices=[mode.value for mode in SearchMode],
        default=SearchMode.HYBRID.value,
    )
    agent_parser.add_argument("--db", type=Path, default=_DEFAULT_DATABASE)
    _add_memory_database_argument(agent_parser)

    memory_parser = subparsers.add_parser(
        "memory",
        help="Manage confirmation-gated long-term memories.",
    )
    memory_subparsers = memory_parser.add_subparsers(
        dest="memory_command",
        required=True,
    )

    remember_parser = memory_subparsers.add_parser(
        "remember",
        help="Create one explicit active memory.",
    )
    remember_parser.add_argument(
        "--type",
        dest="memory_type",
        choices=[memory_type.value for memory_type in MemoryType],
        required=True,
    )
    remember_parser.add_argument("--key", dest="memory_key", required=True)
    activation = remember_parser.add_mutually_exclusive_group()
    activation.add_argument("--replace", dest="replace_id")
    activation.add_argument("--keep-both", action="store_true")
    remember_parser.add_argument("content")
    _add_memory_database_argument(remember_parser)

    list_parser = memory_subparsers.add_parser(
        "list",
        help="List stored memories newest first.",
    )
    list_parser.add_argument(
        "--status",
        choices=[status.value for status in MemoryStatus],
    )
    list_parser.add_argument(
        "--type",
        dest="memory_type",
        choices=[memory_type.value for memory_type in MemoryType],
    )
    _add_memory_database_argument(list_parser)

    memory_search_parser = memory_subparsers.add_parser(
        "search",
        help="Search active memories.",
    )
    memory_search_parser.add_argument("query")
    memory_search_parser.add_argument("--limit", type=_search_limit, default=5)
    memory_search_parser.add_argument(
        "--type",
        dest="memory_type",
        choices=[memory_type.value for memory_type in MemoryType],
    )
    _add_memory_database_argument(memory_search_parser)

    approve_parser = memory_subparsers.add_parser(
        "approve",
        help="Activate one pending memory candidate.",
    )
    approve_parser.add_argument("candidate_id")
    approval = approve_parser.add_mutually_exclusive_group()
    approval.add_argument("--replace", dest="replace_id")
    approval.add_argument("--keep-both", action="store_true")
    _add_memory_database_argument(approve_parser)

    reject_parser = memory_subparsers.add_parser(
        "reject",
        help="Reject one pending memory candidate.",
    )
    reject_parser.add_argument("candidate_id")
    _add_memory_database_argument(reject_parser)

    forget_parser = memory_subparsers.add_parser(
        "forget",
        help="Hard-delete one complete memory replacement chain.",
    )
    forget_parser.add_argument("memory_id")
    _add_memory_database_argument(forget_parser)
    return parser


def _ingest(path: Path, database: Path) -> None:
    index = SQLiteKeywordIndex(database, E5Embedder())
    service = IngestService(MarkdownLoader(), MarkdownChunker(), index)
    result = service.ingest(path)
    action = "Reindexed" if result.replaced else "Indexed"
    print(f"{action} {result.chunk_count} chunks from {result.source}")


def _format_snippet(text: str) -> str:
    collapsed = re.sub(r"\s+", " ", text).strip()
    if len(collapsed) <= _SNIPPET_LENGTH:
        return collapsed
    return f"{collapsed[: _SNIPPET_LENGTH - 3]}..."


def _format_result(position: int, result: SearchResult) -> str:
    location = result.source
    if result.chunk.heading_path:
        location = f"{location} > {' > '.join(result.chunk.heading_path)}"
    return (
        f"{position}. [score={result.score:.3f}] {location}\n"
        f"   {_format_snippet(result.chunk.text)}"
    )


def _search(query: str, limit: int, mode: SearchMode, database: Path) -> None:
    index = SQLiteKeywordIndex(database, E5Embedder())
    results = SearchService(index).search(query, limit=limit, mode=mode)
    print(f"Search mode: {mode.value}")
    if not results:
        print("No relevant chunks found.")
        return
    for position, result in enumerate(results, start=1):
        print(_format_result(position, result))


def _memory_store(database: Path) -> SQLiteMemoryStore:
    return SQLiteMemoryStore(database, E5Embedder())


def _memory_database_argument(database: Path) -> str:
    return f"--memory-db {shlex.quote(str(database))}"


def _memory_activation_action(
    replace_id: str | None,
    keep_both: bool,
) -> MemoryActivationAction:
    if replace_id is not None:
        return MemoryActivationAction.REPLACE
    if keep_both:
        return MemoryActivationAction.KEEP_BOTH
    return MemoryActivationAction.ACTIVATE


def _format_memory_record(record: MemoryRecord) -> str:
    return (
        f"[{record.memory_id}] {record.status.value} "
        f"{record.memory_type.value} {record.memory_key}\n"
        f"  {record.content}"
    )


def _format_memory_search_result(
    position: int,
    result: MemorySearchResult,
) -> str:
    record = result.memory
    return (
        f"{position}. [score={result.score:.3f}] [{record.memory_id}] "
        f"{record.memory_type.value} {record.memory_key}\n"
        f"  {_format_snippet(record.content)}"
    )


def _active_memory_ids(
    service: MemoryAdminService,
    memory_type: MemoryType,
    memory_key: str,
) -> list[str]:
    return [
        record.memory_id
        for record in service.list(
            status=MemoryStatus.ACTIVE,
            memory_type=memory_type,
        )
        if record.memory_key == memory_key
    ]


def _memory_remember(
    memory_type: MemoryType,
    memory_key: str,
    content: str,
    *,
    action: MemoryActivationAction,
    replace_id: str | None,
    database: Path,
) -> None:
    service = MemoryAdminService(_memory_store(database), LocalMemorySafetyPolicy())
    try:
        record = service.remember(
            memory_type,
            memory_key,
            content,
            action=action,
            replace_id=replace_id,
        )
    except MemoryConflictError as error:
        if action is not MemoryActivationAction.ACTIVATE:
            raise
        active_ids = _active_memory_ids(service, memory_type, memory_key)
        if not active_ids:
            raise
        joined = ", ".join(active_ids)
        database_argument = _memory_database_argument(database)
        raise MemoryConflictError(
            f"Active memory conflict. Active IDs: {joined}\n"
            "Retry with one of:\n"
            f"  pka memory remember --type {memory_type.value} "
            f"--key {memory_key} --replace {active_ids[0]} "
            f"{database_argument} CONTENT\n"
            f"  pka memory remember --type {memory_type.value} "
            f"--key {memory_key} --keep-both {database_argument} CONTENT"
        ) from error
    print(f"Remembered {_format_memory_record(record)}")


def _memory_list(
    status: MemoryStatus | None,
    memory_type: MemoryType | None,
    database: Path,
) -> None:
    service = MemoryAdminService(_memory_store(database), LocalMemorySafetyPolicy())
    for record in service.list(status=status, memory_type=memory_type):
        print(_format_memory_record(record))


def _memory_search(
    query: str,
    limit: int,
    memory_type: MemoryType | None,
    database: Path,
) -> None:
    service = MemoryAdminService(_memory_store(database), LocalMemorySafetyPolicy())
    results = service.search(query, limit=limit, memory_type=memory_type)
    if not results:
        print("No active memories found.")
        return
    for position, result in enumerate(results, start=1):
        print(_format_memory_search_result(position, result))


def _memory_approve(
    candidate_id: str,
    *,
    action: MemoryActivationAction,
    replace_id: str | None,
    database: Path,
) -> None:
    service = MemoryAdminService(_memory_store(database), LocalMemorySafetyPolicy())
    try:
        record = service.approve(
            candidate_id,
            action=action,
            replace_id=replace_id,
        )
    except MemoryConflictError as error:
        if action is not MemoryActivationAction.ACTIVATE:
            raise
        candidates = service.list(status=MemoryStatus.PENDING)
        candidate = next(
            (item for item in candidates if item.memory_id == candidate_id),
            None,
        )
        if candidate is None:
            raise
        active_ids = _active_memory_ids(
            service,
            candidate.memory_type,
            candidate.memory_key,
        )
        if not active_ids:
            raise
        joined = ", ".join(active_ids)
        database_argument = _memory_database_argument(database)
        raise MemoryConflictError(
            f"Active memory conflict. Active IDs: {joined}\n"
            "Retry with one of:\n"
            f"  pka memory approve {candidate_id} --replace {active_ids[0]} "
            f"{database_argument}\n"
            f"  pka memory approve {candidate_id} --keep-both {database_argument}"
        ) from error
    print(f"Approved [{record.memory_id}] {record.status.value}")


def _memory_reject(candidate_id: str, database: Path) -> None:
    service = MemoryAdminService(_memory_store(database), LocalMemorySafetyPolicy())
    service.reject(candidate_id)
    print(f"Rejected [{candidate_id}]")


def _memory_forget(memory_id: str, database: Path) -> None:
    service = MemoryAdminService(_memory_store(database), LocalMemorySafetyPolicy())
    deleted = service.forget(memory_id)
    noun = "record" if deleted == 1 else "records"
    print(f"Forgot {deleted} memory {noun}.")


def _format_evidence_source(answer: GroundedAnswer, citation_id: str) -> str:
    evidence = next(item for item in answer.evidence if item.citation_id == citation_id)
    if not evidence.heading_path:
        return evidence.source
    return f"{evidence.source} > {' > '.join(evidence.heading_path)}"


def _print_answer(answer: GroundedAnswer) -> None:
    if answer.status is AnswerStatus.INSUFFICIENT_EVIDENCE:
        print("Insufficient evidence:")
        print(answer.answer)
        return

    print("Answer:")
    print(answer.answer)
    print("\nSources:")
    for citation_id in answer.citation_ids:
        print(f"[{citation_id}] {_format_evidence_source(answer, citation_id)}")


def _ask(query: str, limit: int, mode: SearchMode, database: Path) -> None:
    settings = load_llm_settings(os.environ)
    index = SQLiteKeywordIndex(database, E5Embedder())
    language_model = OpenAICompatibleChatModel(
        api_key=settings.api_key,
        base_url=settings.base_url,
        model_id=settings.model_id,
    )
    answer = AnswerService(SearchService(index), language_model).answer(
        query,
        limit=limit,
        mode=mode,
    )
    _print_answer(answer)


def _format_route(decision: RouteDecision) -> str:
    if decision.source.value == "fallback":
        labels = {
            RouterFallbackReason.PROVIDER_ERROR: "provider error",
            RouterFallbackReason.INVALID_OUTPUT: "invalid router output",
        }
        assert decision.fallback_reason is not None
        return f"Route: retrieve (fallback: {labels[decision.fallback_reason]})"
    if decision.action is RouteAction.DIRECT:
        assert decision.direct_intent is not None
        return (
            f"Route: direct ({decision.direct_intent.value}, {decision.source.value})"
        )
    return f"Route: retrieve ({decision.source.value})"


def _print_route(decision: RouteDecision) -> None:
    print(_format_route(decision))


def _print_agent_answer(response: AgentResponse) -> None:
    if response.direct_answer is not None:
        print(response.direct_answer)
        return
    assert response.grounded_answer is not None
    if response.memory_recall is not None:
        _print_memory_recall_warnings(response.memory_recall)
    _print_contextual_answer(response.grounded_answer, response.memory_recall)


def _print_memory_recall_warnings(report: MemoryRecallReport) -> None:
    for warning in MemoryRecallWarning:
        if warning in report.warnings:
            print(f"Memory recall skipped: {_MEMORY_RECALL_WARNING_LABELS[warning]}")


def _print_contextual_answer(
    answer: GroundedAnswer,
    recall: MemoryRecallReport | None,
) -> None:
    if answer.status is AnswerStatus.INSUFFICIENT_EVIDENCE:
        print("Insufficient evidence:")
        print(answer.answer)
        return

    print("Answer:")
    print(answer.answer)
    source_ids = tuple(
        citation_id
        for citation_id in answer.citation_ids
        if citation_id.startswith("S")
    )
    memory_ids = tuple(
        citation_id
        for citation_id in answer.citation_ids
        if citation_id.startswith("M")
    )
    if source_ids:
        print("\nSources:")
        for citation_id in source_ids:
            print(f"[{citation_id}] {_format_evidence_source(answer, citation_id)}")
    if memory_ids:
        assert recall is not None
        recalled = {item.citation_id: item.memory for item in recall.memories}
        print("\nMemories used:")
        for citation_id in memory_ids:
            memory = recalled[citation_id]
            print(
                f"[{citation_id}] {memory.memory_type.value}.{memory.memory_key} "
                f"— {memory.content}"
            )


def _memory_database_suffix(database: Path) -> str:
    if database == _DEFAULT_MEMORY_DATABASE:
        return ""
    return f" {_memory_database_argument(database)}"


def _print_memory_candidates(
    candidates: Sequence[MemoryRecord],
    memory_service: MemoryAdminService,
    database: Path,
) -> None:
    database_suffix = _memory_database_suffix(database)
    candidates_with_conflicts = [
        (
            candidate,
            _active_memory_ids(
                memory_service,
                candidate.memory_type,
                candidate.memory_key,
            ),
        )
        for candidate in candidates
    ]
    print("\nMemory candidates:")
    for candidate, active_ids in candidates_with_conflicts:
        print(_format_memory_record(candidate))
        if active_ids:
            print(f"Conflicts with active IDs: {', '.join(active_ids)}")
        print("Approve with:")
        for active_id in active_ids:
            print(
                f"  pka memory approve {candidate.memory_id} "
                f"--replace {active_id}{database_suffix}"
            )
        if active_ids:
            print(
                f"  pka memory approve {candidate.memory_id} "
                f"--keep-both{database_suffix}"
            )
        else:
            print(f"  pka memory approve {candidate.memory_id}{database_suffix}")


def _print_memory_extraction_skip(reason: MemoryExtractionSkipReason) -> None:
    if reason in _SILENT_MEMORY_SKIP_REASONS:
        return
    print(f"\nMemory extraction skipped: {_VISIBLE_MEMORY_SKIP_LABELS[reason]}")


def _memory_preflight_skip_reason(
    query: str,
    safety_policy: LocalMemorySafetyPolicy,
) -> MemoryExtractionSkipReason | None:
    if len(query) > MAX_MEMORY_EXTRACTION_INPUT_LENGTH:
        return MemoryExtractionSkipReason.INPUT_TOO_LONG
    if not should_extract_memory(query):
        return MemoryExtractionSkipReason.PREFILTER_MISS
    if not safety_policy.allows(query):
        return MemoryExtractionSkipReason.SENSITIVE_INPUT
    return None


def _agent(
    query: str,
    limit: int,
    mode: SearchMode,
    database: Path,
    memory_database: Path,
) -> None:
    settings = load_llm_settings(os.environ)
    index = SQLiteKeywordIndex(database, E5Embedder())
    language_model = OpenAICompatibleChatModel(
        api_key=settings.api_key,
        base_url=settings.base_url,
        model_id=settings.model_id,
    )
    answer_service = AnswerService(SearchService(index), language_model)

    def memory_answerer() -> MemoryAwareAnswerService:
        recaller: MemoryRecaller
        try:
            memory_store = _memory_store(memory_database)
        except EmbeddingError:
            recaller = _UnavailableMemoryRecaller(MemoryRecallWarning.EMBEDDING_ERROR)
        except MemoryStorageError:
            recaller = _UnavailableMemoryRecaller(MemoryRecallWarning.STORAGE_ERROR)
        else:
            recaller = MemoryRecallService(
                memory_store,
                ConservativeMemoryRelevancePolicy(),
                LocalMemorySafetyPolicy(),
            )
        return MemoryAwareAnswerService(
            SearchService(index),
            recaller,
            language_model,
        )

    response = AgentService(
        HybridQuestionRouter(language_model),
        answer_service,
        memory_answerer,
    ).respond(query, limit=limit, mode=mode, on_route=_print_route)
    _print_agent_answer(response)
    if response.route.source is RouteSource.FALLBACK:
        return
    safety_policy = LocalMemorySafetyPolicy()
    preflight_skip_reason = _memory_preflight_skip_reason(query, safety_policy)
    if preflight_skip_reason is not None:
        _print_memory_extraction_skip(preflight_skip_reason)
        return
    try:
        memory_store = _memory_store(memory_database)
        report = MemoryProposalService(
            LLMMemoryExtractor(language_model),
            memory_store,
            safety_policy,
            should_extract_memory,
        ).propose(query)
        if report.candidates:
            _print_memory_candidates(
                report.candidates,
                MemoryAdminService(memory_store, safety_policy),
                memory_database,
            )
        else:
            assert report.skipped_reason is not None
            _print_memory_extraction_skip(report.skipped_reason)
    except EmbeddingError:
        _print_memory_extraction_skip(MemoryExtractionSkipReason.EMBEDDING_ERROR)
    except MemoryStorageError:
        _print_memory_extraction_skip(MemoryExtractionSkipReason.STORAGE_ERROR)


def _graph_build(source: Path, database: Path, graph_database: Path) -> None:
    import os
    from personal_knowledge_agent.infrastructure.llm_settings import load_llm_settings
    from personal_knowledge_agent.infrastructure.openai_compatible_chat_model import OpenAICompatibleChatModel
    settings = load_llm_settings(os.environ)
    language_model = OpenAICompatibleChatModel(api_key=settings.api_key, base_url=settings.base_url, model_id=settings.model_id)
    source_reader = SQLiteGraphSourceReader(database)
    store = SQLiteKnowledgeGraphStore(graph_database)
    service = GraphBuildService(source_reader, LLMKnowledgeGraphExtractor(language_model), store)
    result = service.build(source)
    print(f"Built graph for {result.source}")
    print(f"Chunks processed: {result.chunks}")
    print(f"Entities contributed: {result.entities}")
    print(f"Relations contributed: {result.relations}")
    print(f"Evidence records: {result.evidence}")


def _graph_search(query: str, limit: int, graph_database: Path) -> None:
    service = GraphQueryService(SQLiteKnowledgeGraphStore(graph_database))
    results = service.search(query, limit=limit)
    if not results:
        print("No graph entities found.")
        return
    for result in results:
        print(f"[{result.entity_id}] {result.entity_type.value} {result.display_name} score={result.score:.4f}")


def _graph_show(entity_id: str, graph_database: Path) -> None:
    service = GraphQueryService(SQLiteKnowledgeGraphStore(graph_database))
    neighborhood = service.show(entity_id)
    print(f"Entity: {neighborhood.entity.display_name}")
    print(f"  Type: {neighborhood.entity.entity_type.value}")
    print(f"  ID: {neighborhood.entity.entity_id}")
    if neighborhood.outgoing:
        print("Outgoing:")
        for n in neighborhood.outgoing:
            print(f"  {n.relation_type.value} -> {n.display_name} [{n.entity_type.value}]")
    if neighborhood.incoming:
        print("Incoming:")
        for n in neighborhood.incoming:
            print(f"  {n.relation_type.value} <- {n.display_name} [{n.entity_type.value}]")



def main(arguments: Sequence[str] | None = None) -> None:
    """Run the CLI and print the result of the requested command."""
    parser = build_parser()
    parsed = parser.parse_args(arguments)

    try:
        if parsed.command == "health":
            print("Personal Knowledge Agent is healthy.")
        elif parsed.command == "ingest":
            _ingest(parsed.path, parsed.db)
        elif parsed.command == "search":
            _search(parsed.query, parsed.limit, SearchMode(parsed.mode), parsed.db)
        elif parsed.command == "ask":
            _ask(parsed.query, parsed.limit, SearchMode(parsed.mode), parsed.db)
        elif parsed.command == "agent":
            _agent(
                parsed.query,
                parsed.limit,
                SearchMode(parsed.mode),
                parsed.db,
                parsed.memory_db,
            )
        elif parsed.command == "graph":
            if parsed.graph_command == "build":
                _graph_build(parsed.source, parsed.db, parsed.graph_db)
            elif parsed.graph_command == "search":
                _graph_search(parsed.query, parsed.limit, parsed.graph_db)
            elif parsed.graph_command == "show":
                _graph_show(parsed.entity_id, parsed.graph_db)
        elif parsed.command == "memory":
            memory_type = (
                MemoryType(parsed.memory_type)
                if hasattr(parsed, "memory_type") and parsed.memory_type is not None
                else None
            )
            if parsed.memory_command == "remember":
                assert memory_type is not None
                _memory_remember(
                    memory_type,
                    parsed.memory_key,
                    parsed.content,
                    action=_memory_activation_action(
                        parsed.replace_id,
                        parsed.keep_both,
                    ),
                    replace_id=parsed.replace_id,
                    database=parsed.memory_db,
                )
            elif parsed.memory_command == "list":
                status = (
                    MemoryStatus(parsed.status) if parsed.status is not None else None
                )
                _memory_list(status, memory_type, parsed.memory_db)
            elif parsed.memory_command == "search":
                _memory_search(
                    parsed.query,
                    parsed.limit,
                    memory_type,
                    parsed.memory_db,
                )
            elif parsed.memory_command == "approve":
                _memory_approve(
                    parsed.candidate_id,
                    action=_memory_activation_action(
                        parsed.replace_id,
                        parsed.keep_both,
                    ),
                    replace_id=parsed.replace_id,
                    database=parsed.memory_db,
                )
            elif parsed.memory_command == "reject":
                _memory_reject(parsed.candidate_id, parsed.memory_db)
            elif parsed.memory_command == "forget":
                _memory_forget(parsed.memory_id, parsed.memory_db)
    except KnowledgeError as error:
        parser.exit(1, f"error: {error}\n")


if __name__ == "__main__":
    main()
