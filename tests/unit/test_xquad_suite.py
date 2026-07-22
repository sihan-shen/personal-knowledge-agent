"""Tests for the packaged bilingual XQuAD-mini evaluation suite."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from collections import Counter
from pathlib import Path

import pytest

from personal_knowledge_agent.application.ingest import IngestService
from personal_knowledge_agent.application.search import SearchService
from personal_knowledge_agent.domain.errors import InvalidEvaluationError
from personal_knowledge_agent.domain.evaluation import EvaluationLanguage
from personal_knowledge_agent.domain.models import SearchMode
from personal_knowledge_agent.infrastructure import xquad_suite
from personal_knowledge_agent.infrastructure.markdown_chunker import MarkdownChunker
from personal_knowledge_agent.infrastructure.markdown_loader import MarkdownLoader
from personal_knowledge_agent.infrastructure.sqlite_keyword_index import (
    SQLiteKeywordIndex,
)
from personal_knowledge_agent.infrastructure.xquad_suite import XQuadMiniSuiteLoader
from tests.helpers import FakeEmbedder

_EXPECTED_QUESTION_IDS = (
    "56beb4343aeaaa14008c925b",
    "56beb7953aeaaa14008c92ab",
    "56beb86b3aeaaa14008c92bd",
    "56bec6ac3aeaaa14008c93fd",
    "56beca913aeaaa14008c946d",
    "57339c16d058e614000b5ec5",
    "5733a32bd058e614000b5f32",
    "57338007d058e614000b5bda",
    "573380e0d058e614000b5be9",
    "5733834ed058e614000b5c26",
    "56dde1d966d3e219004dad8d",
    "56de0daecffd8e1900b4b595",
)
_EXPECTED_REFERENCE_ANSWERS = (
    ("308",),
    ("308",),
    ("Pittsburgh Steelers",),
    ("匹兹堡钢人队",),
    ("39",),
    ("39",),
    ("Six",),
    ("六",),
    ("24",),
    ("24",),
    ("Ogród Saski",),
    ("Ogród Saski",),
    ("Polonia Warsaw",),
    ("华沙波兰人队",),
    ("multi-cultural",),
    ("多元文化",),
    ("commune",),
    ("市镇",),
    ("1817",),
    ("1817 年",),
    ("Rollo",),
    ("罗洛",),
    ("Drogo",),
    ("卓戈 (Drogo)",),
)


def _copy_packaged_suite(tmp_path: Path) -> Path:
    corpus = XQuadMiniSuiteLoader().load()
    source = next(iter(corpus.document_paths.values())).parent.parent
    destination = tmp_path / "xquad_mini"
    shutil.copytree(source, destination)
    return destination


def _corpus_digest(suite_directory: Path) -> str:
    manifest_path = suite_directory / "cases.json"
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    digest = hashlib.sha256()
    digest.update(b"cases.json\0")
    digest.update(manifest_bytes)
    digest.update(b"\0")
    for document in manifest["documents"]:
        resource_name = document["resource_name"]
        digest.update(resource_name.encode("utf-8"))
        digest.update(b"\0")
        digest.update((suite_directory / resource_name).read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def test_xquad_mini_loads_a_balanced_bilingual_public_corpus() -> None:
    corpus = XQuadMiniSuiteLoader().load()

    assert (corpus.suite.suite_id, corpus.suite.version) == ("xquad-mini", "v1")
    assert len(corpus.suite.cases) == 24
    assert Counter(case.language for case in corpus.suite.cases) == {
        EvaluationLanguage.EN: 12,
        EvaluationLanguage.ZH: 12,
    }
    assert len(corpus.suite.documents) == 24
    assert len(corpus.document_paths) == 24
    assert all(path.is_file() for path in corpus.document_paths.values())


def test_each_question_retrieves_its_gold_document_lexically(tmp_path: Path) -> None:
    corpus = XQuadMiniSuiteLoader().load()
    index = SQLiteKeywordIndex(
        tmp_path / "knowledge.db",
        FakeEmbedder({}, default_vector=(1.0, 0.0)),
    )
    ingester = IngestService(MarkdownLoader(), MarkdownChunker(), index)
    searcher = SearchService(index)
    for document in corpus.suite.documents:
        ingester.ingest(corpus.document_paths[document.document_id])

    for case in corpus.suite.cases:
        results = searcher.search(case.question, limit=5, mode=SearchMode.LEXICAL)
        gold_source = str(corpus.document_paths[case.gold_document_id].resolve())

        assert gold_source in {result.source for result in results}, case.case_id


def test_xquad_mini_preserves_aligned_id_order_and_distinct_answer_aliases() -> None:
    corpus = XQuadMiniSuiteLoader().load()
    cases = corpus.suite.cases

    for pair_index in range(0, len(cases), 2):
        english, chinese = cases[pair_index : pair_index + 2]
        english_prefix, english_language = english.case_id.rsplit(":", 1)
        chinese_prefix, chinese_language = chinese.case_id.rsplit(":", 1)

        assert english_prefix == chinese_prefix
        assert (english_language, chinese_language) == ("en", "zh")
        assert english.case_id == english.gold_document_id
        assert chinese.case_id == chinese.gold_document_id

    assert len({case.case_id for case in cases}) == 24
    assert (
        tuple(case.case_id.split(":", 2)[1] for case in cases[::2])
        == _EXPECTED_QUESTION_IDS
    )
    assert tuple(case.reference_answers for case in cases) == (
        _EXPECTED_REFERENCE_ANSWERS
    )
    assert tuple(document.document_id for document in corpus.suite.documents) == tuple(
        case.case_id for case in cases
    )
    assert all(
        case.reference_answers == tuple(dict.fromkeys(case.reference_answers))
        for case in cases
    )


def test_xquad_mini_has_complete_public_attribution() -> None:
    corpus = XQuadMiniSuiteLoader().load()
    root = next(iter(corpus.document_paths.values())).parent.parent
    attribution_path = root / "ATTRIBUTION.md"
    license_path = root / "LICENSE.txt"

    attribution = attribution_path.read_text(encoding="utf-8")
    license_text = license_path.read_text(encoding="utf-8")

    assert "XQuAD" in corpus.suite.attribution
    assert "CC BY-SA 4.0" in corpus.suite.attribution
    assert "ATTRIBUTION.md" in corpus.suite.attribution
    assert "LICENSE.txt" in corpus.suite.attribution
    assert "https://github.com/google-deepmind/xquad" in attribution
    assert "xquad.en.json" in attribution
    assert "xquad.zh.json" in attribution
    assert "Artetxe" in attribution and "Ruder" in attribution
    assert "Yogatama" in attribution
    assert "derivative" in attribution.lower()
    assert "CC BY-SA 4.0" in attribution
    assert "7d30520c717524000f0d9d2f9c10a069acd9d285" in attribution
    assert (
        "e4c57d1c9143aaa1c5d265ba5987a65f4e69528d2a98f29d6e75019b10344f29"
        in attribution
    )
    assert (
        "691d0b3359bc6b8faa8de931dfdfe21d50a65861ae348e32a0d1a0190b0c8835"
        in attribution
    )
    assert (
        "c255777a49cee10455ec20fa58046df9b325ea41f0dbafff146e10b630ec9ccb"
        in attribution
    )
    assert len(re.findall(r"\b[0-9a-f]{64}\b", attribution)) == 3
    assert "Attribution-ShareAlike 4.0 International" in license_text


def test_every_gold_document_exists_and_contains_only_title_and_context() -> None:
    corpus = XQuadMiniSuiteLoader().load()

    for case in corpus.suite.cases:
        path = corpus.document_paths[case.gold_document_id]
        lines = path.read_text(encoding="utf-8").splitlines()

        assert lines[0] == f"# {case.gold_document_id}"
        assert lines[1] == ""
        assert len(lines) == 3
        assert lines[2].strip()
        assert not lines[2].startswith("#")
        assert any(answer in lines[2] for answer in case.reference_answers)


def test_xquad_mini_rejects_malformed_json(tmp_path: Path) -> None:
    (tmp_path / "cases.json").write_text("{not-json", encoding="utf-8")

    with pytest.raises(InvalidEvaluationError, match="manifest"):
        XQuadMiniSuiteLoader(tmp_path).load()


def test_xquad_mini_rejects_an_incomplete_manifest(tmp_path: Path) -> None:
    incomplete_manifest = {
        "suite_id": "xquad-mini",
        "version": "v1",
        "attribution": "XQuAD derivative, CC BY-SA 4.0",
        "documents": [],
    }
    (tmp_path / "cases.json").write_text(
        json.dumps(incomplete_manifest), encoding="utf-8"
    )

    with pytest.raises(InvalidEvaluationError, match="manifest"):
        XQuadMiniSuiteLoader(tmp_path).load()


def test_xquad_mini_rejects_a_changed_question(tmp_path: Path) -> None:
    suite_directory = _copy_packaged_suite(tmp_path)
    manifest_path = suite_directory / "cases.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["cases"][0]["question"] = "A substituted question?"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    with pytest.raises(InvalidEvaluationError, match="integrity"):
        XQuadMiniSuiteLoader(suite_directory).load()


def test_xquad_mini_rejects_a_changed_context_that_still_has_the_answer(
    tmp_path: Path,
) -> None:
    suite_directory = _copy_packaged_suite(tmp_path)
    document_path = next((suite_directory / "documents").glob("*.en.md"))
    serialized = document_path.read_text(encoding="utf-8")
    document_path.write_text(serialized[:-1] + " Tampered.\n", encoding="utf-8")

    with pytest.raises(InvalidEvaluationError, match="integrity"):
        XQuadMiniSuiteLoader(suite_directory).load()


def test_xquad_mini_rejects_a_truncated_license(tmp_path: Path) -> None:
    suite_directory = _copy_packaged_suite(tmp_path)
    (suite_directory / "LICENSE.txt").write_text(
        "Attribution-ShareAlike 4.0 International\n", encoding="utf-8"
    )

    with pytest.raises(InvalidEvaluationError, match="LICENSE"):
        XQuadMiniSuiteLoader(suite_directory).load()


def test_xquad_mini_rejects_an_undeclared_markdown_document(tmp_path: Path) -> None:
    suite_directory = _copy_packaged_suite(tmp_path)
    (suite_directory / "documents" / "stale.md").write_text(
        "# stale\n\nStale context.\n", encoding="utf-8"
    )

    with pytest.raises(InvalidEvaluationError, match="documents"):
        XQuadMiniSuiteLoader(suite_directory).load()


def test_xquad_mini_rejects_duplicate_same_language_contexts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    suite_directory = _copy_packaged_suite(tmp_path)
    manifest = json.loads((suite_directory / "cases.json").read_text(encoding="utf-8"))
    english_documents = [
        document
        for document in manifest["documents"]
        if document["document_id"].endswith(":en")
    ]
    first_path = suite_directory / english_documents[0]["resource_name"]
    duplicate_path = suite_directory / english_documents[1]["resource_name"]
    duplicate_id = english_documents[1]["document_id"]
    first_context = first_path.read_text(encoding="utf-8").splitlines()[2]
    duplicate_path.write_text(
        f"# {duplicate_id}\n\n{first_context}\n",
        encoding="utf-8",
    )
    digest = _corpus_digest(suite_directory)
    attribution_path = suite_directory / "ATTRIBUTION.md"
    attribution = attribution_path.read_text(encoding="utf-8")
    attribution_path.write_text(
        attribution.replace(xquad_suite._CORPUS_SHA256, digest),
        encoding="utf-8",
    )
    monkeypatch.setattr(xquad_suite, "_CORPUS_SHA256", digest)

    with pytest.raises(InvalidEvaluationError, match="duplicate English context"):
        XQuadMiniSuiteLoader(suite_directory).load()
