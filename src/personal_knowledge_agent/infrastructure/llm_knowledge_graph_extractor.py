"""Provider-neutral LLM adapter for per-chunk graph extraction."""

from __future__ import annotations

from personal_knowledge_agent.domain.graph import GraphChunkExtraction
from personal_knowledge_agent.domain.models import Chunk
from personal_knowledge_agent.domain.protocols import (
    KnowledgeGraphExtractor,
    LanguageModel,
)
from personal_knowledge_agent.infrastructure.graph_prompt_builder import (
    build_graph_extraction_messages,
)
from personal_knowledge_agent.infrastructure.structured_graph_parser import (
    parse_graph_extraction,
)


class LLMKnowledgeGraphExtractor:
    """Extract entities and relations from one chunk via an LLM.

    This is a deliberately thin adapter that builds the prompt, calls the
    language model, and delegates strict validation to the parser.
    """

    def __init__(self, language_model: LanguageModel) -> None:
        self._language_model = language_model

    def extract(self, chunk: Chunk) -> GraphChunkExtraction:
        """Extract and validate graph facts from one chunk."""
        messages = build_graph_extraction_messages(chunk)
        output = self._language_model.generate(messages)
        return parse_graph_extraction(output, chunk)
