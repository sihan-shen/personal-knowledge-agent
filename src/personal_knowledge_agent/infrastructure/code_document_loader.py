"""Load small local source and development-document files."""

from __future__ import annotations

import re
from pathlib import Path

from personal_knowledge_agent.domain.errors import (
    DocumentReadError,
    EmptyDocumentError,
    UnsupportedDocumentError,
)
from personal_knowledge_agent.domain.models import Document, make_document_id

_SUPPORTED = {".py", ".json", ".yaml", ".yml", ".toml", ".txt", ".rst"}
_DECLARATION = re.compile(r"^(?:async[ \t]+)?(?:def|class)[ \t]+([^(:]+)")


class CodeDocumentLoader:
    """Load explicit UTF-8 code documents without execution or AST parsing."""

    def load(self, path: Path) -> Document:
        resolved = path.expanduser().resolve()
        if resolved.suffix.lower() not in _SUPPORTED:
            raise UnsupportedDocumentError(f"Unsupported code document: {resolved}")
        if not resolved.is_file():
            raise DocumentReadError(f"Path is not a readable file: {resolved}")
        try:
            payload = resolved.read_bytes()
        except OSError as error:
            raise DocumentReadError(f"Could not read document: {resolved}") from error
        if len(payload) > 1_048_576 or b"\0" in payload:
            raise DocumentReadError(f"Document is binary or exceeds 1 MiB: {resolved}")
        try:
            content = payload.decode("utf-8")
        except UnicodeDecodeError as error:
            raise DocumentReadError(
                f"Document is not valid UTF-8: {resolved}"
            ) from error
        if not content.strip():
            raise EmptyDocumentError(f"Document is empty: {resolved}")
        source = str(resolved)
        return Document(
            make_document_id(source), source, self._content(resolved, content)
        )

    @staticmethod
    def _content(path: Path, content: str) -> str:
        if path.suffix.lower() != ".py":
            return f"# {path.name}\n\n{content.strip()}"
        lines = [f"# {path.name}"]
        for line in content.splitlines():
            match = _DECLARATION.match(line)
            if match is not None and line == line.lstrip():
                lines.extend(("", f"## {match.group(1).strip()}", ""))
            lines.append(line)
        return "\n".join(lines).strip()
