"""Tests for evidence validation."""
from __future__ import annotations

from core.models import (
    Checkpoint,
    DialogueTurn,
    Evidence,
    EvaluationDimension,
    EvaluationType,
    InstructionSpec,
    Persona,
    PersonaArchetype,
    SimulatedDialogue,
    TerminationReason,
)
from evaluator.evidence import build_validated_result, validate_evidence


def _make_dialogue(turns: list[tuple[int, str, str]]) -> SimulatedDialogue:
    """Helper to create a dialogue from (turn_id, role, content) tuples."""
    return SimulatedDialogue(
        id="test_dialogue",
        persona=Persona(
            archetype=PersonaArchetype.cooperative,
            behavior_description="test",
            response_style="normal",
            emotional_state="neutral",
            domain_knowledge="basic",
        ),
        instruction_spec=InstructionSpec(
            role="test agent",
            objective="test",
            flow_graph=[],
            constraints=[],
            knowledge_points=[],
            raw_text="test",
        ),
        turns=[DialogueTurn(turn_id=tid, role=role, content=content) for tid, role, content in turns],
        termination_reason=TerminationReason.natural_end,
    )


def _make_checkpoint(cp_id: str = "cp_001") -> Checkpoint:
    return Checkpoint(
        id=cp_id,
        description="test checkpoint",
        dimension=EvaluationDimension.flow,
        evaluation_type=EvaluationType.binary,
    )


class TestValidateEvidence:
    def test_valid_turn_ids(self):
        dialogue = _make_dialogue([(1, "agent", "hello"), (2, "user", "hi")])
        evidence = Evidence(turn_ids=[1, 2], quoted_text="hello", reasoning="ok")
        warnings = validate_evidence(evidence, dialogue, _make_checkpoint())
        assert warnings == []

    def test_invalid_turn_id(self):
        dialogue = _make_dialogue([(1, "agent", "hello")])
        evidence = Evidence(turn_ids=[99], quoted_text="hello", reasoning="ok")
        warnings = validate_evidence(evidence, dialogue, _make_checkpoint())
        assert len(warnings) == 1
        assert "turn_id 99" in warnings[0]

    def test_quoted_text_found(self):
        dialogue = _make_dialogue([(1, "agent", "您好，我是XX公司的小李")])
        evidence = Evidence(turn_ids=[1], quoted_text="我是XX公司", reasoning="ok")
        warnings = validate_evidence(evidence, dialogue, _make_checkpoint())
        assert warnings == []

    def test_quoted_text_not_found(self):
        dialogue = _make_dialogue([(1, "agent", "您好")])
        evidence = Evidence(turn_ids=[1], quoted_text="完全不存在的内容片段", reasoning="ok")
        warnings = validate_evidence(evidence, dialogue, _make_checkpoint())
        assert len(warnings) == 1
        assert "quoted_text" in warnings[0]

    def test_empty_quoted_text(self):
        dialogue = _make_dialogue([(1, "agent", "hello")])
        evidence = Evidence(turn_ids=[1], quoted_text="", reasoning="ok")
        warnings = validate_evidence(evidence, dialogue, _make_checkpoint())
        assert warnings == []


class TestBuildValidatedResult:
    def test_basic_construction(self):
        dialogue = _make_dialogue([(1, "agent", "hello world")])
        cp = _make_checkpoint()
        raw = {"passed": True, "score": 1.0, "turn_ids": [1], "quoted_text": "hello", "reasoning": "ok"}
        result = build_validated_result(raw, cp, dialogue)
        assert result.passed is True
        assert result.score == 1.0
        assert result.evidence.quoted_text == "hello"
        assert result.validation_warnings == []

    def test_confidence_from_raw(self):
        dialogue = _make_dialogue([(1, "agent", "hello")])
        cp = _make_checkpoint()
        raw = {"passed": True, "score": 1.0, "confidence": 0.85}
        result = build_validated_result(raw, cp, dialogue)
        assert result.evidence.confidence == 0.85

    def test_validation_warnings_attached(self):
        dialogue = _make_dialogue([(1, "agent", "hello")])
        cp = _make_checkpoint()
        raw = {"passed": True, "score": 1.0, "turn_ids": [99], "quoted_text": "不存在的文本", "reasoning": "ok"}
        result = build_validated_result(raw, cp, dialogue)
        assert len(result.validation_warnings) == 2  # invalid turn_id + unmatched quote
