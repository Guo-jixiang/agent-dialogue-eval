"""Tests for InstructionParser constraint category robustness and grounding."""
from __future__ import annotations

import pytest

from core.llm_client import LLMClient
from core.models import (
    Constraint,
    ConstraintCategory,
    ConstraintType,
    FlowNode,
    InstructionSpec,
    KnowledgePoint,
    Transition,
    ConditionType,
)
from parser.instruction_parser import InstructionParser


def _make_parser() -> InstructionParser:
    """Create a parser instance with a dummy (unused) LLM client for unit tests.

    ``_validate_and_normalize`` and ``_verify_grounding`` are pure functions
    that don't call the LLM, so any LLMClient instance works.
    """
    return InstructionParser(llm_client=LLMClient(api_key="dummy", base_url="http://localhost", model="dummy"))


def _make_raw(category: str, description: str = "测试约束", threshold: float | None = None) -> dict:
    return {
        "role": "测试角色",
        "objective": "测试目标",
        "flow_graph": [],
        "constraints": [
            {
                "id": "c_1",
                "type": "soft",
                "category": category,
                "description": description,
                "threshold": threshold,
            }
        ],
        "knowledge_points": [],
    }


class TestConstraintCategoryFallback:
    """Verify that unknown/hallucinated categories don't crash the parser."""

    def test_behavior_category_with_digits_does_not_crash(self):
        """The exact bug scenario: 'behavior' + description with digits."""
        parser = _make_parser()
        raw = _make_raw(category="behavior", description="控制行为，避免打断2次以上")
        spec = parser._validate_and_normalize(raw, "test instruction")
        assert len(spec.constraints) == 1
        assert spec.constraints[0].category == ConstraintCategory.tone

    def test_completely_unknown_category_defaults_to_tone(self):
        """An entirely bogus category should default to tone without crashing."""
        parser = _make_parser()
        raw = _make_raw(category="bogus_category", description="some constraint")
        spec = parser._validate_and_normalize(raw, "test instruction")
        assert len(spec.constraints) == 1
        assert spec.constraints[0].category == ConstraintCategory.tone

    def test_existing_fallback_word_maps_to_forbidden_word(self):
        """Existing fallback: 'word' -> forbidden_word."""
        parser = _make_parser()
        raw = _make_raw(category="word", description="禁止说保证")
        spec = parser._validate_and_normalize(raw, "test instruction")
        assert len(spec.constraints) == 1
        assert spec.constraints[0].category == ConstraintCategory.forbidden_word

    def test_existing_fallback_limit_maps_to_length(self):
        """Existing fallback: 'limit' -> length."""
        parser = _make_parser()
        raw = _make_raw(category="limit", description="每次回复不超过50字")
        spec = parser._validate_and_normalize(raw, "test instruction")
        assert len(spec.constraints) == 1
        assert spec.constraints[0].category == ConstraintCategory.length

    def test_valid_category_passes_through_unchanged(self):
        """A valid category should pass through unchanged."""
        parser = _make_parser()
        raw = _make_raw(category="pacing", description="控制语速和节奏")
        spec = parser._validate_and_normalize(raw, "test instruction")
        assert len(spec.constraints) == 1
        assert spec.constraints[0].category == ConstraintCategory.pacing


class TestThresholdExtraction:
    """Verify that threshold extraction still works after the refactor."""

    def test_length_category_extracts_threshold_from_description(self):
        """When category=length and no explicit threshold, extract digits from description."""
        parser = _make_parser()
        raw = _make_raw(category="length", description="每次回复不超过50个字")
        spec = parser._validate_and_normalize(raw, "test instruction")
        assert len(spec.constraints) == 1
        assert spec.constraints[0].threshold == 50.0

    def test_length_category_uses_explicit_threshold(self):
        """When explicit threshold is provided, use it directly."""
        parser = _make_parser()
        raw = _make_raw(category="length", description="长度限制", threshold=30)
        spec = parser._validate_and_normalize(raw, "test instruction")
        assert len(spec.constraints) == 1
        assert spec.constraints[0].threshold == 30.0

    def test_non_length_category_does_not_extract_digits(self):
        """Digits in a tone constraint description should not become a threshold."""
        parser = _make_parser()
        raw = _make_raw(category="tone", description="使用3种以上的语气变化")
        spec = parser._validate_and_normalize(raw, "test instruction")
        assert len(spec.constraints) == 1
        assert spec.constraints[0].threshold is None


# ---------------------------------------------------------------------------
# Grounding verification helpers
# ---------------------------------------------------------------------------

SAMPLE_SOURCE = """角色：你是外卖平台骑手服务专员。

约束要求：
1. 每次回复不超过80个字，语言简洁清晰
2. 禁止使用"保证"、"承诺"等绝对性词汇
3. 语气亲切专业，不得使用生硬或命令式语气

知识点：
- 意外险保障金额：最高50万元人民币
- 客服热线：400-888-xxxx"""


def _make_spec_with_constraints(
    *descriptions: str, category: ConstraintCategory = ConstraintCategory.tone
) -> InstructionSpec:
    """Build a minimal InstructionSpec with given constraint descriptions."""
    constraints = [
        Constraint(
            id=f"c_{i+1}",
            type=ConstraintType.soft,
            category=category,
            description=desc,
            threshold=None,
        )
        for i, desc in enumerate(descriptions)
    ]
    return InstructionSpec(
        role="测试",
        objective="测试",
        flow_graph=[],
        constraints=constraints,
        knowledge_points=[],
        raw_text=SAMPLE_SOURCE,
    )


def _make_spec_with_flow_nodes(*descriptions: str) -> InstructionSpec:
    """Build a minimal InstructionSpec with given flow node descriptions."""
    nodes = [
        FlowNode(
            id=f"node_{i+1}",
            description=desc,
            required=True,
            content_keywords=[],
            transitions=[],
        )
        for i, desc in enumerate(descriptions)
    ]
    return InstructionSpec(
        role="测试",
        objective="测试",
        flow_graph=nodes,
        constraints=[],
        knowledge_points=[],
        raw_text=SAMPLE_SOURCE,
    )


def _make_spec_with_knowledge(
    descriptions_and_contents: list[tuple[str, str]],
) -> InstructionSpec:
    """Build a minimal InstructionSpec with given knowledge points."""
    kps = [
        KnowledgePoint(
            id=f"k_{i+1}",
            description=desc,
            correct_content=content,
            common_errors=[],
        )
        for i, (desc, content) in enumerate(descriptions_and_contents)
    ]
    return InstructionSpec(
        role="测试",
        objective="测试",
        flow_graph=[],
        constraints=[],
        knowledge_points=kps,
        raw_text=SAMPLE_SOURCE,
    )


class TestGroundingVerification:
    """Verify that _verify_grounding correctly scores extracted items."""

    def test_constraint_matching_source_has_high_score(self):
        """Constraint text that appears verbatim in source → high score."""
        score = InstructionParser._grounding_score(
            "每次回复不超过80个字", SAMPLE_SOURCE
        )
        assert score >= 0.4

    def test_constraint_not_in_source_has_low_score(self):
        """Completely fabricated constraint → low score."""
        score = InstructionParser._grounding_score(
            "控制对话节奏保持每分钟200字语速", SAMPLE_SOURCE
        )
        assert score < 0.4

    def test_flow_node_matching_source_has_high_score(self):
        """Flow node that references source content → high score."""
        score = InstructionParser._grounding_score(
            "自我介绍为外卖平台骑手服务专员并问候用户", SAMPLE_SOURCE
        )
        # Contains "外卖平台骑手服务专员" which appears verbatim in source
        assert score >= 0.4

    def test_flow_node_fabricated_has_low_score(self):
        """Flow node completely unrelated to source → low score."""
        score = InstructionParser._grounding_score(
            "量子计算芯片性能评估与对比分析", SAMPLE_SOURCE
        )
        assert score < 0.4

    def test_knowledge_point_matching_source_has_high_score(self):
        """Knowledge point with content from source → high score."""
        score = InstructionParser._grounding_score(
            "意外险保障金额最高50万元人民币", SAMPLE_SOURCE
        )
        assert score >= 0.4

    def test_knowledge_point_fabricated_has_low_score(self):
        """Knowledge point with fabricated content → low score."""
        score = InstructionParser._grounding_score(
            "2028年首批火星移民费用为500万元每人", SAMPLE_SOURCE
        )
        assert score < 0.4

    def test_empty_description_scores_1(self):
        """Empty descriptions get vacuous score of 1.0."""
        score = InstructionParser._grounding_score("", SAMPLE_SOURCE)
        assert score == 1.0

    def test_spec_unchanged_after_verification(self):
        """Grounding never removes items — only logs warnings."""
        parser = _make_parser()
        spec = _make_spec_with_constraints("完全捏造的约束内容没有依据")
        result = parser._verify_grounding(spec, SAMPLE_SOURCE)
        assert result is spec
        assert len(result.constraints) == 1

    def test_ngram_extraction_chinese(self):
        """Character 3-grams for Chinese text."""
        ngrams = InstructionParser._char_ngrams("不超过80个", n=3)
        assert "不超过" in ngrams
        assert "超过8" in ngrams
        assert "80个" in ngrams
        # 6 chars → 4 trigrams
        assert len(ngrams) == 4
