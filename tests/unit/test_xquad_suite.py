"""Tests for the packaged bilingual XQuAD-mini evaluation suite."""

from __future__ import annotations

import json
import re
import shutil
from collections import Counter
from pathlib import Path

import pytest

from personal_knowledge_agent.domain.errors import InvalidEvaluationError
from personal_knowledge_agent.domain.evaluation import EvaluationLanguage
from personal_knowledge_agent.infrastructure.xquad_suite import XQuadMiniSuiteLoader

_EXPECTED_QUESTION_IDS = (
    "56beb4343aeaaa14008c925b",
    "56beb4343aeaaa14008c925c",
    "56beb4343aeaaa14008c925d",
    "56beb4343aeaaa14008c925e",
    "56beb4343aeaaa14008c925f",
    "56d6f3500d65d21400198290",
    "56d6f3500d65d21400198291",
    "56d6f3500d65d21400198292",
    "56d6f3500d65d21400198293",
    "56d6f3500d65d21400198294",
    "56d9992fdc89441400fdb59c",
    "56d9992fdc89441400fdb59e",
)
_EXPECTED_REFERENCE_ANSWERS = (
    ("308",),
    ("308",),
    ("136",),
    ("136 次",),
    ("118",),
    ("118",),
    ("four",),
    ("四次",),
    ("Kawann Short",),
    ("卡万·肖特",),
    ("24",),
    ("24 次",),
    ("Kawann Short",),
    ("卡万·肖特",),
    ("four",),
    ("四次",),
    ("four",),
    ("四次",),
    ("Kurt Coleman",),
    ("科特·科尔曼",),
    ("24",),
    ("24 次",),
    ("Kony Ealy",),
    ("科尼·伊利",),
)


def _copy_packaged_suite(tmp_path: Path) -> Path:
    corpus = XQuadMiniSuiteLoader().load()
    source = next(iter(corpus.document_paths.values())).parent.parent
    destination = tmp_path / "xquad_mini"
    shutil.copytree(source, destination)
    return destination


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
        "4cfa30a46d8c3a065f298fda2d41018df2902dfccba1c78532d6b628f3a34d43"
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
