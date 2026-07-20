"""Strict loader for the packaged XQuAD-mini-v1 evaluation corpus."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Final, cast

from personal_knowledge_agent.application.evaluate import EvaluationCorpus
from personal_knowledge_agent.domain.errors import InvalidEvaluationError
from personal_knowledge_agent.domain.evaluation import (
    EvaluationCase,
    EvaluationDocument,
    EvaluationLanguage,
    EvaluationSuite,
)

_SUITE_DIRECTORY: Final[Path] = (
    Path(__file__).parent.parent / "evaluation_suites" / "xquad_mini"
)
_MANIFEST_KEYS: Final[frozenset[str]] = frozenset(
    {"suite_id", "version", "attribution", "documents", "cases"}
)
_DOCUMENT_KEYS: Final[frozenset[str]] = frozenset({"document_id", "resource_name"})
_CASE_KEYS: Final[frozenset[str]] = frozenset(
    {
        "case_id",
        "language",
        "question",
        "reference_answers",
        "gold_document_id",
    }
)
_LOGICAL_ID_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"^xquad-mini-v1:([0-9a-f]{24}):(en|zh)$"
)
_REVISION: Final[str] = "7d30520c717524000f0d9d2f9c10a069acd9d285"
_ENGLISH_SHA256: Final[str] = (
    "e4c57d1c9143aaa1c5d265ba5987a65f4e69528d2a98f29d6e75019b10344f29"
)
_CHINESE_SHA256: Final[str] = (
    "691d0b3359bc6b8faa8de931dfdfe21d50a65861ae348e32a0d1a0190b0c8835"
)
_CORPUS_SHA256: Final[str] = (
    "4cfa30a46d8c3a065f298fda2d41018df2902dfccba1c78532d6b628f3a34d43"
)
_LICENSE_SHA256: Final[str] = (
    "23ee78c8bae49cf08ea2f0c84945c66b987ebe4520881fb51b3dad4fb43d07c2"
)


def _invalid(detail: str) -> InvalidEvaluationError:
    return InvalidEvaluationError(f"invalid XQuAD-mini manifest: {detail}")


def _object_without_duplicate_keys(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise _invalid(f"duplicate JSON key {key!r}")
        value[key] = item
    return value


def _mapping(value: object, field_name: str) -> Mapping[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise _invalid(f"{field_name} must be an object")
    return cast(dict[str, object], value)


def _list(value: object, field_name: str) -> list[object]:
    if not isinstance(value, list):
        raise _invalid(f"{field_name} must be an array")
    return cast(list[object], value)


def _text(mapping: Mapping[str, object], key: str, owner: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise _invalid(f"{owner}.{key} must be non-blank text")
    return value


def _require_exact_keys(
    mapping: Mapping[str, object], expected: frozenset[str], owner: str
) -> None:
    if set(mapping) != expected:
        raise _invalid(f"{owner} must contain exactly {sorted(expected)!r}")


def _read_manifest(path: Path) -> tuple[Mapping[str, object], bytes]:
    try:
        serialized_bytes = path.read_bytes()
        serialized = serialized_bytes.decode("utf-8")
    except (OSError, UnicodeError) as exc:
        raise _invalid("cases.json cannot be read") from exc
    try:
        value: object = json.loads(
            serialized, object_pairs_hook=_object_without_duplicate_keys
        )
    except json.JSONDecodeError as exc:
        raise _invalid("cases.json is not valid JSON") from exc
    return _mapping(value, "root"), serialized_bytes


def _read_attribution(suite_directory: Path, manifest_attribution: str) -> None:
    manifest_fragments = (
        "XQuAD",
        "CC BY-SA 4.0",
        "ATTRIBUTION.md",
        "LICENSE.txt",
        _REVISION,
    )
    if any(fragment not in manifest_attribution for fragment in manifest_fragments):
        raise _invalid("attribution metadata is incomplete")

    try:
        attribution = (suite_directory / "ATTRIBUTION.md").read_text(encoding="utf-8")
        license_bytes = (suite_directory / "LICENSE.txt").read_bytes()
        license_text = license_bytes.decode("utf-8")
    except (OSError, UnicodeError) as exc:
        raise _invalid("attribution or license resource cannot be read") from exc

    attribution_fragments = (
        "https://github.com/google-deepmind/xquad",
        f"xquad/{_REVISION}/xquad.en.json",
        f"xquad/{_REVISION}/xquad.zh.json",
        _REVISION,
        _ENGLISH_SHA256,
        _CHINESE_SHA256,
        _CORPUS_SHA256,
        "Mikel Artetxe",
        "Sebastian Ruder",
        "Dani Yogatama",
        "derivative",
        "CC BY-SA 4.0",
    )
    if any(fragment not in attribution for fragment in attribution_fragments):
        raise _invalid("ATTRIBUTION.md is incomplete")
    if (
        "Attribution-ShareAlike 4.0 International" not in license_text
        or hashlib.sha256(license_bytes).hexdigest() != _LICENSE_SHA256
    ):
        raise _invalid("LICENSE.txt is not CC BY-SA 4.0")


def _validate_resource_integrity(
    suite_directory: Path,
    manifest_bytes: bytes,
    documents: tuple[EvaluationDocument, ...],
    document_paths: Mapping[str, Path],
) -> None:
    documents_directory = suite_directory / "documents"
    try:
        actual_resources = {
            path.relative_to(suite_directory).as_posix()
            for path in documents_directory.rglob("*.md")
            if path.is_file()
        }
    except OSError as exc:
        raise _invalid("documents directory cannot be read") from exc
    declared_resources = {document.resource_name for document in documents}
    if actual_resources != declared_resources:
        raise _invalid("documents directory must match the manifest exactly")

    digest = hashlib.sha256()
    digest.update(b"cases.json\0")
    digest.update(manifest_bytes)
    digest.update(b"\0")
    try:
        for document in documents:
            digest.update(document.resource_name.encode("utf-8"))
            digest.update(b"\0")
            digest.update(document_paths[document.document_id].read_bytes())
            digest.update(b"\0")
    except OSError as exc:
        raise _invalid("document integrity cannot be verified") from exc
    if digest.hexdigest() != _CORPUS_SHA256:
        raise _invalid("packaged corpus integrity check failed")


def _document_and_path(
    value: object, index: int, suite_directory: Path
) -> tuple[EvaluationDocument, Path, str]:
    owner = f"documents[{index}]"
    item = _mapping(value, owner)
    _require_exact_keys(item, _DOCUMENT_KEYS, owner)
    document_id = _text(item, "document_id", owner)
    resource_name = _text(item, "resource_name", owner)

    match = _LOGICAL_ID_PATTERN.fullmatch(document_id)
    if match is None:
        raise _invalid(f"{owner}.document_id has an invalid format")
    question_id, language = match.groups()
    expected_resource = f"documents/{question_id}.{language}.md"
    if resource_name != expected_resource:
        raise _invalid(f"{owner}.resource_name does not match its document ID")

    path = suite_directory / "documents" / f"{question_id}.{language}.md"
    try:
        resolved_path = path.resolve(strict=True)
        resolved_path.relative_to(suite_directory.resolve(strict=True))
        serialized = resolved_path.read_bytes().decode("utf-8")
    except (OSError, UnicodeError, ValueError) as exc:
        raise _invalid(f"{owner}.resource_name is not a readable suite file") from exc
    if not resolved_path.is_file():
        raise _invalid(f"{owner}.resource_name is not a file")

    lines = serialized.splitlines()
    if (
        len(lines) != 3
        or lines[0] != f"# {document_id}"
        or lines[1] != ""
        or not lines[2].strip()
        or serialized != f"# {document_id}\n\n{lines[2]}\n"
    ):
        raise _invalid(f"{owner}.resource_name is not one title plus one context")

    try:
        document = EvaluationDocument(document_id, resource_name)
    except ValueError as exc:
        raise _invalid(f"{owner} violates the evaluation document schema") from exc
    return document, resolved_path, lines[2]


def _case(value: object, index: int) -> EvaluationCase:
    owner = f"cases[{index}]"
    item = _mapping(value, owner)
    _require_exact_keys(item, _CASE_KEYS, owner)
    case_id = _text(item, "case_id", owner)
    language_value = _text(item, "language", owner)
    question = _text(item, "question", owner)
    gold_document_id = _text(item, "gold_document_id", owner)

    match = _LOGICAL_ID_PATTERN.fullmatch(case_id)
    if match is None:
        raise _invalid(f"{owner}.case_id has an invalid format")
    _, id_language = match.groups()
    if language_value != id_language:
        raise _invalid(f"{owner}.language does not match its case ID")
    try:
        language = EvaluationLanguage(language_value)
    except ValueError as exc:
        raise _invalid(f"{owner}.language must be en or zh") from exc

    answer_values = _list(item.get("reference_answers"), f"{owner}.reference_answers")
    answers: list[str] = []
    for answer_index, answer in enumerate(answer_values):
        if not isinstance(answer, str) or not answer.strip():
            raise _invalid(
                f"{owner}.reference_answers[{answer_index}] must be non-blank text"
            )
        answers.append(answer)
    if not answers:
        raise _invalid(f"{owner}.reference_answers must not be empty")
    if len(set(answers)) != len(answers):
        raise _invalid(f"{owner}.reference_answers must be distinct")
    if gold_document_id != case_id:
        raise _invalid(f"{owner}.gold_document_id must be the same-language case ID")

    try:
        return EvaluationCase(
            case_id,
            language,
            question,
            tuple(answers),
            gold_document_id,
        )
    except ValueError as exc:
        raise _invalid(f"{owner} violates the evaluation case schema") from exc


def _validate_alignment(
    documents: tuple[EvaluationDocument, ...],
    cases: tuple[EvaluationCase, ...],
    contexts: Mapping[str, str],
) -> None:
    if len(documents) != 24:
        raise _invalid("documents must contain exactly 24 entries")
    if len(cases) != 24:
        raise _invalid("cases must contain exactly 24 entries")
    document_ids = tuple(document.document_id for document in documents)
    case_ids = tuple(case.case_id for case in cases)
    if len(set(document_ids)) != 24:
        raise _invalid("document IDs must be unique")
    if len(set(case_ids)) != 24:
        raise _invalid("case IDs must be unique")
    if document_ids != case_ids:
        raise _invalid("documents and cases must have the same ordered IDs")
    if Counter(case.language for case in cases) != {
        EvaluationLanguage.EN: 12,
        EvaluationLanguage.ZH: 12,
    }:
        raise _invalid("cases must contain exactly 12 English and 12 Chinese entries")

    for pair_index in range(0, 24, 2):
        english, chinese = cases[pair_index : pair_index + 2]
        english_match = _LOGICAL_ID_PATTERN.fullmatch(english.case_id)
        chinese_match = _LOGICAL_ID_PATTERN.fullmatch(chinese.case_id)
        if english_match is None or chinese_match is None:
            raise _invalid("case IDs have an invalid format")
        if (
            english.language is not EvaluationLanguage.EN
            or chinese.language is not EvaluationLanguage.ZH
            or english_match.group(1) != chinese_match.group(1)
        ):
            raise _invalid("cases must be ordered as aligned English/Chinese pairs")

    for case in cases:
        context = contexts.get(case.gold_document_id)
        if context is None:
            raise _invalid("every gold document must be present")
        if case.reference_answers[0] not in context:
            raise _invalid("every primary answer must occur in its gold context")


class XQuadMiniSuiteLoader:
    """Load and validate the packaged XQuAD-mini-v1 suite."""

    def __init__(self, suite_directory: Path | None = None) -> None:
        self._suite_directory = suite_directory or _SUITE_DIRECTORY

    def load(self) -> EvaluationCorpus:
        """Return the complete packaged corpus or reject any malformed resource."""
        manifest, manifest_bytes = _read_manifest(self._suite_directory / "cases.json")
        _require_exact_keys(manifest, _MANIFEST_KEYS, "root")
        suite_id = _text(manifest, "suite_id", "root")
        version = _text(manifest, "version", "root")
        attribution = _text(manifest, "attribution", "root")
        if suite_id != "xquad-mini" or version != "v1":
            raise _invalid("suite_id and version must be xquad-mini v1")
        _read_attribution(self._suite_directory, attribution)

        document_values = _list(manifest.get("documents"), "documents")
        loaded_documents = tuple(
            _document_and_path(value, index, self._suite_directory)
            for index, value in enumerate(document_values)
        )
        documents = tuple(item[0] for item in loaded_documents)
        document_paths = {item[0].document_id: item[1] for item in loaded_documents}
        contexts = {item[0].document_id: item[2] for item in loaded_documents}
        _validate_resource_integrity(
            self._suite_directory,
            manifest_bytes,
            documents,
            document_paths,
        )

        case_values = _list(manifest.get("cases"), "cases")
        cases = tuple(_case(value, index) for index, value in enumerate(case_values))
        _validate_alignment(documents, cases, contexts)

        try:
            suite = EvaluationSuite(
                suite_id,
                version,
                documents,
                cases,
                attribution,
            )
            return EvaluationCorpus(suite, document_paths)
        except ValueError as exc:
            raise _invalid("suite violates the evaluation corpus schema") from exc
