"""Command-line interface for Personal Knowledge Agent."""

from __future__ import annotations

import argparse
import os
import re
from collections.abc import Sequence
from pathlib import Path

from personal_knowledge_agent.application.agent import AgentService
from personal_knowledge_agent.application.answer import AnswerService
from personal_knowledge_agent.application.ingest import IngestService
from personal_knowledge_agent.application.memory_admin import MemoryAdminService
from personal_knowledge_agent.application.search import SearchService
from personal_knowledge_agent.domain.errors import KnowledgeError, MemoryConflictError
from personal_knowledge_agent.domain.memory import (
    MemoryActivationAction,
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
    SearchMode,
    SearchResult,
)
from personal_knowledge_agent.infrastructure.e5_embedder import E5Embedder
from personal_knowledge_agent.infrastructure.hybrid_question_router import (
    HybridQuestionRouter,
)
from personal_knowledge_agent.infrastructure.llm_settings import load_llm_settings
from personal_knowledge_agent.infrastructure.markdown_chunker import MarkdownChunker
from personal_knowledge_agent.infrastructure.markdown_loader import MarkdownLoader
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
_SNIPPET_LENGTH = 160


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
        f"  {_format_snippet(record.content)}"
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
        raise MemoryConflictError(
            f"Active memory conflict. Active IDs: {joined}\n"
            "Retry with one of:\n"
            f"  pka memory remember --type {memory_type.value} "
            f"--key {memory_key} --replace {active_ids[0]} CONTENT\n"
            f"  pka memory remember --type {memory_type.value} "
            f"--key {memory_key} --keep-both CONTENT"
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
        raise MemoryConflictError(
            f"Active memory conflict. Active IDs: {joined}\n"
            "Retry with one of:\n"
            f"  pka memory approve {candidate_id} --replace {active_ids[0]}\n"
            f"  pka memory approve {candidate_id} --keep-both"
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
    _print_answer(response.grounded_answer)


def _agent(query: str, limit: int, mode: SearchMode, database: Path) -> None:
    settings = load_llm_settings(os.environ)
    index = SQLiteKeywordIndex(database, E5Embedder())
    language_model = OpenAICompatibleChatModel(
        api_key=settings.api_key,
        base_url=settings.base_url,
        model_id=settings.model_id,
    )
    answer_service = AnswerService(SearchService(index), language_model)
    response = AgentService(
        HybridQuestionRouter(language_model),
        answer_service,
    ).respond(query, limit=limit, mode=mode, on_route=_print_route)
    _print_agent_answer(response)


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
            _agent(parsed.query, parsed.limit, SearchMode(parsed.mode), parsed.db)
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
