from __future__ import annotations

import asyncio

from loguru import logger

from config.settings import settings
from core.llm_client import LLMClient
from core.models import Checkpoint, CheckpointResult, EvaluationType, SimulatedDialogue
from evaluator.judges.base import BaseJudge
from evaluator.judges.meta_judge import MetaJudge


class SelfConsistentJudge:
    """Wraps any BaseJudge with multi-run self-consistency and arbitration.

    Based on Self-Consistency (Wang et al., 2022): multiple samples + majority
    voting significantly improves reasoning accuracy. Applied here to reduce
    variance in LLM-as-judge scoring (G-Eval, Liu et al., 2023).
    """

    def __init__(
        self,
        base_judge: BaseJudge,
        llm_client: LLMClient,
        num_runs: int = settings.SELF_CONSISTENCY_RUNS,
        divergence_threshold: float = settings.DIVERGENCE_THRESHOLD,
    ):
        self._judge = base_judge
        self._num_runs = num_runs
        self._meta = MetaJudge(llm_client, num_runs=num_runs)
        self._meta.DIVERGENCE_THRESHOLD = divergence_threshold

    async def judge(
        self,
        dialogue: SimulatedDialogue,
        checkpoints: list[Checkpoint],
    ) -> list[CheckpointResult]:
        relevant_cps = [cp for cp in checkpoints if cp.dimension == self._judge.dimension]
        if not relevant_cps:
            return []

        # Run the judge N times in parallel
        async def _run(i: int) -> list[CheckpointResult]:
            logger.debug(f"SelfConsistentJudge({self._judge.dimension.value}): run {i+1}/{self._num_runs}")
            return await self._judge.judge(dialogue, checkpoints)

        runs: list[list[CheckpointResult]] = await asyncio.gather(
            *(_run(i) for i in range(self._num_runs))
        )

        # Check if all runs are identical (deterministic checks)
        if self._all_identical(runs):
            logger.debug(f"SelfConsistentJudge({self._judge.dimension.value}): all runs identical, skipping arbitration")
            return self._attach_metadata(runs[0], runs, force_confidence=1.0)

        # Arbitrate divergent results
        final = await self._meta.arbitrate(dialogue, runs)
        return self._attach_metadata(final, runs)

    @staticmethod
    def _all_identical(runs: list[list[CheckpointResult]]) -> bool:
        """Check if all runs produce identical score tuples."""
        if len(runs) < 2:
            return True
        first_scores = {(r.checkpoint.id, r.score, r.passed) for r in runs[0]}
        for run in runs[1:]:
            run_scores = {(r.checkpoint.id, r.score, r.passed) for r in run}
            if first_scores != run_scores:
                return False
        return True

    @staticmethod
    def _attach_metadata(
        results: list[CheckpointResult],
        runs: list[list[CheckpointResult]],
        force_confidence: float | None = None,
    ) -> list[CheckpointResult]:
        """Attach num_runs, run_scores, and confidence to each result."""
        # Build lookup: checkpoint_id -> list of scores across runs
        scores_by_cp: dict[str, list[float]] = {}
        for run in runs:
            for r in run:
                scores_by_cp.setdefault(r.checkpoint.id, []).append(r.score)

        for result in results:
            cp_id = result.checkpoint.id
            result.num_runs = len(runs)
            result.run_scores = scores_by_cp.get(cp_id, [result.score])

            if force_confidence is not None:
                result.confidence = force_confidence
            else:
                # Confidence = LLM self-reported * agreement across runs
                llm_conf = result.evidence.confidence if result.evidence else 1.0
                agreement_conf = _compute_agreement_confidence(result.run_scores, result.checkpoint)
                result.confidence = round(llm_conf * agreement_conf, 3)

        return results


def _compute_agreement_confidence(scores: list[float], checkpoint: Checkpoint) -> float:
    """Compute confidence from inter-run agreement.

    For binary checkpoints: fraction agreeing with majority vote.
    For numeric checkpoints: 1 - normalized_std_dev.
    """
    if len(scores) < 2:
        return 1.0

    if checkpoint.evaluation_type == EvaluationType.binary:
        # Scores are 0.0 or 1.0 for binary
        passed_count = sum(1 for s in scores if s >= 0.5)
        majority = passed_count > len(scores) / 2
        agree_count = passed_count if majority else len(scores) - passed_count
        return agree_count / len(scores)
    else:
        mean = sum(scores) / len(scores)
        variance = sum((s - mean) ** 2 for s in scores) / len(scores)
        std_dev = variance ** 0.5
        return max(0.0, 1.0 - std_dev)
