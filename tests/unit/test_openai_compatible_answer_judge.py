"""Tests for the provider-neutral LLM Judge adapter."""

from personal_knowledge_agent.domain.evaluation import (
    FaithfulnessJudgeRequest,
    ReferenceQualityJudgeRequest,
)
from personal_knowledge_agent.domain.models import Evidence, ModelOutput
from personal_knowledge_agent.infrastructure.openai_compatible_answer_judge import (
    OpenAICompatibleAnswerJudge,
)
from tests.helpers import ScriptedLanguageModel


def faithfulness_request() -> FaithfulnessJudgeRequest:
    return FaithfulnessJudgeRequest(
        "Which city is the capital?",
        "Paris is the capital.",
        (Evidence("S1", "chunk", "fixture.md", (), "Evidence-only text.", 1.0),),
    )


def reference_quality_request() -> ReferenceQualityJudgeRequest:
    return ReferenceQualityJudgeRequest(
        "Which city is the capital?",
        "Paris is the capital.",
        ("Paris",),
    )


def test_adapter_parses_both_judge_contracts_and_preserves_input_isolation() -> None:
    model = ScriptedLanguageModel(
        (
            ModelOutput(
                "judge-model",
                '{"faithfulness":{"score":2,"reason":"Supported."}}',
                "stop",
            ),
            ModelOutput(
                "judge-model",
                '{"relevance":{"score":2,"reason":"Direct."},'
                '"completeness":{"score":2,"reason":"Complete."}}',
                "stop",
            ),
        )
    )
    judge = OpenAICompatibleAnswerJudge(model)

    faithfulness = judge.judge_faithfulness(faithfulness_request())
    quality = judge.judge_reference_quality(reference_quality_request())

    assert faithfulness.score == 2
    assert quality.relevance.score == 2
    faithfulness_content = "\n".join(message.content for message in model.calls[0])
    quality_content = "\n".join(message.content for message in model.calls[1])
    assert "reference_answers" not in faithfulness_content
    assert "retrieved_evidence" not in quality_content
