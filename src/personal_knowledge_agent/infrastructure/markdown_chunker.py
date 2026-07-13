"""Heading-aware Markdown chunking with fenced-code preservation."""

from __future__ import annotations

import re
from dataclasses import dataclass

from personal_knowledge_agent.domain.models import Chunk, Document, make_chunk_id

_HEADING_PATTERN = re.compile(
    r"^\s{0,3}(#{1,6})[ \t]+(.+?)(?:[ \t]+#+[ \t]*)?$"
)
_FENCE_PATTERN = re.compile(r"^\s{0,3}(`{3,}|~{3,})")


@dataclass(frozen=True, slots=True)
class _Block:
    heading_path: tuple[str, ...]
    text: str


class MarkdownChunker:
    """Split Markdown at semantic block boundaries."""

    def __init__(self, target_chars: int = 800) -> None:
        if target_chars <= 0:
            raise ValueError("target_chars must be positive")
        self._target_chars = target_chars

    def split(self, document: Document) -> list[Chunk]:
        """Create deterministic chunks in document order."""
        grouped = self._aggregate(self._scan_blocks(document.content))
        return [
            Chunk(
                chunk_id=make_chunk_id(document.document_id, position),
                document_id=document.document_id,
                position=position,
                text=text,
                heading_path=heading_path,
            )
            for position, (heading_path, text) in enumerate(grouped)
        ]

    def _scan_blocks(self, content: str) -> list[_Block]:
        blocks: list[_Block] = []
        heading_path: tuple[str, ...] = ()
        paragraph_lines: list[str] = []
        code_lines: list[str] = []
        fence: str | None = None

        def flush_paragraph() -> None:
            text = "\n".join(paragraph_lines).strip()
            if text:
                blocks.append(_Block(heading_path, text))
            paragraph_lines.clear()

        def flush_code() -> None:
            text = "\n".join(code_lines).strip()
            if text:
                blocks.append(_Block(heading_path, text))
            code_lines.clear()

        for line in content.splitlines():
            if fence is not None:
                code_lines.append(line)
                if self._closes_fence(line, fence):
                    flush_code()
                    fence = None
                continue

            fence_match = _FENCE_PATTERN.match(line)
            if fence_match is not None:
                flush_paragraph()
                fence = fence_match.group(1)
                code_lines.append(line)
                continue

            heading_match = _HEADING_PATTERN.match(line)
            if heading_match is not None:
                flush_paragraph()
                level = len(heading_match.group(1))
                title = heading_match.group(2).strip()
                heading_path = heading_path[: level - 1] + (title,)
                continue

            if not line.strip():
                flush_paragraph()
                continue

            paragraph_lines.append(line)

        if fence is not None:
            flush_code()
        flush_paragraph()
        return blocks

    @staticmethod
    def _closes_fence(line: str, opening: str) -> bool:
        stripped = line.strip()
        return (
            len(stripped) >= len(opening)
            and set(stripped) == {opening[0]}
            and stripped[0] == opening[0]
        )

    def _aggregate(
        self,
        blocks: list[_Block],
    ) -> list[tuple[tuple[str, ...], str]]:
        chunks: list[tuple[tuple[str, ...], str]] = []
        current_heading: tuple[str, ...] | None = None
        current_text = ""

        for block in blocks:
            candidate = (
                f"{current_text}\n\n{block.text}" if current_text else block.text
            )
            must_split = current_text and (
                block.heading_path != current_heading
                or len(candidate) > self._target_chars
            )
            if must_split:
                assert current_heading is not None
                chunks.append((current_heading, current_text))
                current_text = block.text
            else:
                current_text = candidate
            current_heading = block.heading_path

        if current_text:
            assert current_heading is not None
            chunks.append((current_heading, current_text))

        return chunks
