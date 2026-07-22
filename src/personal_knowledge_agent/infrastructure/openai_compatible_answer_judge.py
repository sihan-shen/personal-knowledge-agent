"""Adapt structured Chat Completions output to the two LLM Judge protocols."""

from __future__ import annotations

from personal_knowledge_agent.domain.evaluation import (
    FaithfulnessJudgeRequest,
    JudgeMetricScore,
    ReferenceQualityJudgeRequest,
    ReferenceQualityScores,
)
from personal_knowledge_agent.domain.protocols import LanguageModel
from personal_knowledge_agent.infrastructure.answer_judge_prompt_builder import (
    build_faithfulness_judge_messages,
    build_reference_quality_judge_messages,
)
from personal_knowledge_agent.infrastructure.structured_answer_judge_parser import (
    parse_faithfulness_judge_output,
    parse_reference_quality_judge_output,
)


class OpenAICompatibleAnswerJudge:
    """Judge answers through the existing OpenAI-compatible model boundary."""

    def __init__(self, language_model: LanguageModel) -> None:
        self._language_model = language_model

    def judge_faithfulness(
        self,
        request: FaithfulnessJudgeRequest,
    ) -> JudgeMetricScore:
        """Return one strict faithfulness assessment from evidence-only input."""
        output = self._language_model.generate(
            build_faithfulness_judge_messages(request)
        )
        return parse_faithfulness_judge_output(output)

    def judge_reference_quality(
        self,
        request: ReferenceQualityJudgeRequest,
    ) -> ReferenceQualityScores:
        """Return strict relevance and completeness scores from references only."""
        output = self._language_model.generate(
            build_reference_quality_judge_messages(request)
        )
        return parse_reference_quality_judge_output(output)
