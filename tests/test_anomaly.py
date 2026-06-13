"""Tests for anomaly detection."""
from __future__ import annotations

from core.models import (
    Checkpoint,
    CheckpointResult,
    DimensionScore,
    EvaluationDimension,
    EvaluationType,
    Evidence,
)
from evaluator.anomaly import detect_anomalies, detect_dimension_anomalies


def _make_result(checkpoint: Checkpoint, score: float, confidence: float = 1.0, warnings: list[str] | None = None, run_scores: list[float] | None = None) -> CheckpointResult:
    return CheckpointResult(
        checkpoint=checkpoint,
        passed=score >= 0.5,
        score=score,
        evidence=Evidence(turn_ids=[], quoted_text="", reasoning=""),
        confidence=confidence,
        validation_warnings=warnings or [],
        run_scores=run_scores or [],
    )


def _make_checkpoint(cp_id: str = "cp_001", eval_type: EvaluationType = EvaluationType.binary) -> Checkpoint:
    return Checkpoint(
        id=cp_id,
        description="test",
        dimension=EvaluationDimension.flow,
        evaluation_type=eval_type,
    )


def _make_dim_score(dimension: str = "flow", score: float = 0.8, results: list | None = None) -> DimensionScore:
    return DimensionScore(
        dimension=EvaluationDimension(dimension),
        score=score,
        weighted_score=score * 0.3,
        checkpoint_results=results or [],
    )


class TestDetectAnomalies:
    def test_no_anomalies_normal_case(self):
        cp = _make_checkpoint()
        results = [
            _make_result(cp, 0.9),
            _make_result(cp, 0.7),
            _make_result(cp, 0.8),
        ]
        dim_scores = [_make_dim_score(results=results)]
        flags = detect_anomalies(results, dim_scores)
        assert flags == []

    def test_suspicious_full_score(self):
        cp = _make_checkpoint()
        results = [_make_result(cp, 1.0) for _ in range(10)]
        dim_scores = [_make_dim_score(results=results)]
        flags = detect_anomalies(results, dim_scores)
        assert any("SUSPICIOUS_FULL_SCORE" in f for f in flags)

    def test_suspicious_zero_score(self):
        cp = _make_checkpoint()
        results = [_make_result(cp, 0.0) for _ in range(10)]
        dim_scores = [_make_dim_score(results=results)]
        flags = detect_anomalies(results, dim_scores)
        assert any("SUSPICIOUS_ZERO_SCORE" in f for f in flags)

    def test_low_confidence(self):
        cp = _make_checkpoint()
        results = [_make_result(cp, 0.8, confidence=0.3)]
        dim_scores = [_make_dim_score(results=results)]
        flags = detect_anomalies(results, dim_scores)
        assert any("LOW_CONFIDENCE" in f for f in flags)

    def test_evidence_warnings(self):
        cp = _make_checkpoint()
        results = [_make_result(cp, 0.8, warnings=["turn_id 99 does not exist"])]
        dim_scores = [_make_dim_score(results=results)]
        flags = detect_anomalies(results, dim_scores)
        assert any("EVIDENCE_WARNINGS" in f for f in flags)

    def test_bimodal_distribution(self):
        cp = _make_checkpoint()
        results = [_make_result(cp, 0.8, run_scores=[0.1, 0.9, 0.1])]
        dim_scores = [_make_dim_score(results=results)]
        flags = detect_anomalies(results, dim_scores)
        assert any("BIMODAL" in f for f in flags)

    def test_empty_results(self):
        flags = detect_anomalies([], [])
        assert flags == []


class TestDetectDimensionAnomalies:
    def test_all_perfect(self):
        cp = _make_checkpoint()
        results = [_make_result(cp, 1.0), _make_result(cp, 1.0)]
        flags = detect_dimension_anomalies(results, "flow")
        assert any("ALL_PERFECT" in f for f in flags)

    def test_all_zero(self):
        cp = _make_checkpoint()
        results = [_make_result(cp, 0.0), _make_result(cp, 0.0)]
        flags = detect_dimension_anomalies(results, "flow")
        assert any("ALL_ZERO" in f for f in flags)

    def test_mixed_scores(self):
        cp = _make_checkpoint()
        results = [_make_result(cp, 0.8), _make_result(cp, 0.3)]
        flags = detect_dimension_anomalies(results, "flow")
        assert flags == []
