from __future__ import annotations

from config.settings import settings
from core.models import CheckpointResult, DimensionScore


def detect_anomalies(
    results: list[CheckpointResult],
    dimension_scores: list[DimensionScore],
) -> list[str]:
    """Detect anomalous scoring patterns. Returns list of anomaly descriptions."""
    flags: list[str] = []

    if not results:
        return flags

    # 1. Full-score clustering
    full_count = sum(1 for r in results if r.score >= 0.99)
    ratio = full_count / len(results)
    if ratio > settings.ANOMALY_FULL_SCORE_RATIO:
        flags.append(
            f"SUSPICIOUS_FULL_SCORE: {ratio:.0%} 检查点接近满分（阈值 {settings.ANOMALY_FULL_SCORE_RATIO:.0%}），建议人工复核"
        )

    # 2. Zero-score clustering
    zero_count = sum(1 for r in results if r.score <= 0.01)
    ratio = zero_count / len(results)
    if ratio > settings.ANOMALY_ZERO_SCORE_RATIO:
        flags.append(
            f"SUSPICIOUS_ZERO_SCORE: {ratio:.0%} 检查点为零分（阈值 {settings.ANOMALY_ZERO_SCORE_RATIO:.0%}），建议人工复核"
        )

    # 3. Low confidence clustering
    low_conf = [r for r in results if r.confidence < settings.CONFIDENCE_THRESHOLD]
    if low_conf:
        flags.append(
            f"LOW_CONFIDENCE: {len(low_conf)} 个检查点置信度低于 {settings.CONFIDENCE_THRESHOLD}，建议人工复核"
        )

    # 4. Evidence validation warnings
    warn_count = sum(len(r.validation_warnings) for r in results)
    if warn_count > 0:
        flags.append(f"EVIDENCE_WARNINGS: {warn_count} 条证据验证警告")

    # 5. Bimodal distribution in run_scores
    for r in results:
        if len(r.run_scores) >= 3:
            low = sum(1 for s in r.run_scores if s < 0.3)
            high = sum(1 for s in r.run_scores if s > 0.7)
            if low > 0 and high > 0 and low + high == len(r.run_scores):
                flags.append(
                    f"BIMODAL: 检查点 {r.checkpoint.id} 评分呈双峰分布（{r.run_scores}），评判一致性低"
                )

    return flags


def detect_dimension_anomalies(
    results: list[CheckpointResult],
    dimension: str,
) -> list[str]:
    """Detect anomalies for a single dimension."""
    dim_results = [r for r in results if r.checkpoint.dimension.value == dimension]
    if not dim_results:
        return []

    flags: list[str] = []
    scores = [r.score for r in dim_results]

    if all(s >= 0.99 for s in scores):
        flags.append(f"DIM_{dimension}_ALL_PERFECT")
    if all(s <= 0.01 for s in scores):
        flags.append(f"DIM_{dimension}_ALL_ZERO")

    return flags
