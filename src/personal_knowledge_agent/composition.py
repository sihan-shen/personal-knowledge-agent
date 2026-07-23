"""Outer-layer factories shared by executable adapters."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from personal_knowledge_agent.application.answer import AnswerService
from personal_knowledge_agent.application.search import SearchService
from personal_knowledge_agent.infrastructure.e5_embedder import E5Embedder
from personal_knowledge_agent.infrastructure.llm_settings import load_llm_settings
from personal_knowledge_agent.infrastructure.openai_compatible_chat_model import (
    OpenAICompatibleChatModel,
)
from personal_knowledge_agent.infrastructure.sqlite_keyword_index import (
    SQLiteKeywordIndex,
)


def create_grounded_answer_service(
    database: Path,
    environment: Mapping[str, str],
) -> AnswerService:
    """Construct the existing grounded-answer service from explicit runtime inputs."""
    settings = load_llm_settings(environment)
    index = SQLiteKeywordIndex(database, E5Embedder())
    language_model = OpenAICompatibleChatModel(
        api_key=settings.api_key,
        base_url=settings.base_url,
        model_id=settings.model_id,
    )
    return AnswerService(SearchService(index), language_model)
