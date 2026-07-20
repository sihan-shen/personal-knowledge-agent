"""Application boundaries for evaluation corpora."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Protocol

from personal_knowledge_agent.domain.evaluation import EvaluationSuite


@dataclass(frozen=True, slots=True)
class EvaluationCorpus:
    """A suite paired with its exact, existing local document files."""

    suite: EvaluationSuite
    document_paths: Mapping[str, Path]

    def __post_init__(self) -> None:
        expected_document_ids = {
            document.document_id for document in self.suite.documents
        }
        if set(self.document_paths) != expected_document_ids:
            raise ValueError("document paths must match suite document IDs exactly")
        paths = dict(self.document_paths)
        if any(not path.is_file() for path in paths.values()):
            raise ValueError("document paths must refer to an existing file")
        object.__setattr__(self, "document_paths", MappingProxyType(paths))


class EvaluationCorpusLoader(Protocol):
    """Loads an evaluation suite together with local fixture paths."""

    def load(self) -> EvaluationCorpus:
        """Return the complete evaluation corpus."""
