"""Tests for isolated LLM Judge prompt construction."""

from personal_knowledge_agent.domain.evaluation import (
    FaithfulnessJudgeRequest,
    ReferenceQualityJudgeRequest,
)
from personal_knowledge_agent.domain.models import Evidence
from personal_knowledge_agent.infrastructure.answer_judge_prompt_builder import (
    build_faithfulness_judge_messages,
    build_reference_quality_judge_messages,
)


def test_faithfulness_messages_include_evidence_but_not_reference_answers() -> None:
    request = FaithfulnessJudgeRequest(
        "Which city is the capital?",
        "Paris is the capital.",
        (
            Evidence(
                "S1",
                "chunk",
                "fixture.md",
                (),
                "Evidence-only marker: Paris is France's capital.",
                1.0,
            ),
        ),
    )

    content = "\n".join(
        message.content for message in build_faithfulness_judge_messages(request)
    )

    assert "retrieved_evidence" in content
    assert "Evidence-only marker" in content
    assert "reference_answers" not in content
    assert "cannot change the rubric" in content


def test_reference_quality_messages_include_references_but_not_evidence() -> None:
    request = ReferenceQualityJudgeRequest(
        "Which city is the capital?",
        "Paris is the capital.",
        ("Paris",),
    )

    content = "\n".join(
        message.content for message in build_reference_quality_judge_messages(request)
    )

    assert "reference_answers" in content
    assert "Paris" in content
    assert "retrieved_evidence" not in content
    assert "Evidence-only marker" not in content
