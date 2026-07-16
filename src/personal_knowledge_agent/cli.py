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
from personal_knowledge_agent.application.search import SearchService
from personal_knowledge_agent.domain.errors import KnowledgeError
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
from personal_knowledge_agent.infrastructure.openai_compatible_chat_model import (
    OpenAICompatibleChatModel,
)
from personal_knowledge_agent.infrastructure.sqlite_keyword_index import (
    SQLiteKeywordIndex,
)

_DEFAULT_DATABASE = Path(".pka/knowledge.db")
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


def _print_agent_response(response: AgentResponse) -> None:
    print(_format_route(response.route))
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
    ).respond(query, limit=limit, mode=mode)
    _print_agent_response(response)


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
    except KnowledgeError as error:
        parser.exit(1, f"error: {error}\n")


if __name__ == "__main__":
    main()
