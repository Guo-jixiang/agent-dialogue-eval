"""Tests for self-consistency and confidence computation."""
from __future__ import annotations

from core.models import (
    Checkpoint,
    CheckpointResult,
    EvaluationDimension,
    EvaluationType,
    Evidence,
)
from evaluator.judges.consistency_wrapper import _compute_agreement_confidence, SelfConsistentJudge


def _make_checkpoint(eval_type: EvaluationType = EvaluationType.score_1_5) -> Checkpoint:
    return Checkpoint(
        id="cp_001",
        description="test",
        dimension=EvaluationDimension.style,
        evaluation_type=eval_type,
    )


def _make_result(checkpoint: Checkpoint, score: float, passed: bool = True) -> CheckpointResult:
    return CheckpointResult(
        checkpoint=checkpoint,
        passed=passed,
        score=score,
        evidence=Evidence(turn_ids=[], quoted_text="", reasoning=""),
    )


class TestComputeAgreementConfidence:
    def test_binary_unanimous_pass(self):
        cp = _make_checkpoint(EvaluationType.binary)
        scores = [1.0, 1.0, 1.0]
        conf = _compute_agreement_confidence(scores, cp)
        assert conf == 1.0

    def test_binary_unanimous_fail(self):
        cp = _make_checkpoint(EvaluationType.binary)
        scores = [0.0, 0.0, 0.0]
        conf = _compute_agreement_confidence(scores, cp)
        assert conf == 1.0

    def test_binary_split_2_1(self):
        cp = _make_checkpoint(EvaluationType.binary)
        scores = [1.0, 1.0, 0.0]
        conf = _compute_agreement_confidence(scores, cp)
        assert abs(conf - 2 / 3) < 0.01

    def test_numeric_identical(self):
        cp = _make_checkpoint(EvaluationType.score_1_5)
        scores = [0.8, 0.8, 0.8]
        conf = _compute_agreement_confidence(scores, cp)
        assert abs(conf - 1.0) < 1e-9

    def test_numeric_spread(self):
        cp = _make_checkpoint(EvaluationType.score_1_5)
        scores = [0.2, 0.5, 0.8]
        conf = _compute_agreement_confidence(scores, cp)
        assert 0.0 < conf < 1.0

    def test_single_run(self):
        cp = _make_checkpoint()
        scores = [0.5]
        conf = _compute_agreement_confidence(scores, cp)
        assert conf == 1.0


class TestSelfConsistentJudgeAllIdentical:
    def test_identical_runs(self):
        cp = _make_checkpoint()
        run1 = [_make_result(cp, 0.8)]
        run2 = [_make_result(cp, 0.8)]
        assert SelfConsistentJudge._all_identical([run1, run2]) is True

    def test_different_runs(self):
        cp = _make_checkpoint()
        run1 = [_make_result(cp, 0.8)]
        run2 = [_make_result(cp, 0.6)]
        assert SelfConsistentJudge._all_identical([run1, run2]) is False

    def test_single_run(self):
        cp = _make_checkpoint()
        run1 = [_make_result(cp, 0.8)]
        assert SelfConsistentJudge._all_identical([run1]) is True
