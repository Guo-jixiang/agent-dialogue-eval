from __future__ import annotations

from dataclasses import dataclass

from jinja2 import Environment, FileSystemLoader
from loguru import logger

from config.settings import settings
from core.llm_client import LLMClient
from core.models import (
    Checkpoint,
    CheckpointResult,
    EvaluationType,
    Evidence,
    SimulatedDialogue,
)
from evaluator.evidence import build_validated_result


@dataclass
class _RunResult:
    passed: bool
    score: float
    reasoning: str


class MetaJudge:
    """Arbitrates checkpoints where multiple runs diverge.

    Handles all evaluation types:
    - binary: disagreement triggers majority-vote arbitration
    - score_1_5 / percentage: score spread > threshold triggers arbitration
    """

    DIVERGENCE_THRESHOLD = settings.DIVERGENCE_THRESHOLD

    def __init__(self, llm_client: LLMClient, num_runs: int = 3):
        self._llm = llm_client
        self._num_runs = num_runs
        self._jinja = Environment(
            loader=FileSystemLoader(str(settings.PROMPTS_DIR)),
            trim_blocks=True,
            lstrip_blocks=True,
        )

    async def arbitrate(
        self,
        dialogue: SimulatedDialogue,
        results_per_run: list[list[CheckpointResult]],
    ) -> list[CheckpointResult]:
        """
        Takes multiple runs of results for the same checkpoints.
        Returns final results, arbitrating where there's divergence.
        """
        if not results_per_run:
            return []

        # Group by checkpoint id
        by_cp: dict[str, list[CheckpointResult]] = {}
        for run in results_per_run:
            for r in run:
                by_cp.setdefault(r.checkpoint.id, []).append(r)

        final: list[CheckpointResult] = []
        disputed: list[tuple[Checkpoint, list[CheckpointResult]]] = []

        for cp_id, runs in by_cp.items():
            if self._is_divergent(runs):
                disputed.append((runs[0].checkpoint, runs))
            else:
                avg_score = sum(r.score for r in runs) / len(runs)
                avg_passed = avg_score >= 0.5
                final.append(
                    CheckpointResult(
                        checkpoint=runs[0].checkpoint,
                        passed=avg_passed,
                        score=avg_score,
                        evidence=runs[0].evidence,
                        explanation=f"均值 ({len(runs)}次): {avg_score:.2f}",
                    )
                )

        if disputed:
            arbitrated = await self._arbitrate_disputed(dialogue, disputed)
            final.extend(arbitrated)

        return final

    @staticmethod
    def _is_divergent(runs: list[CheckpointResult]) -> bool:
        """Check if runs diverge enough to warrant arbitration."""
        if len(runs) < 2:
            return False

        cp = runs[0].checkpoint
        if cp.evaluation_type == EvaluationType.binary:
            passed_count = sum(1 for r in runs if r.passed)
            return 0 < passed_count < len(runs)
        else:
            scores = [r.score for r in runs]
            spread = max(scores) - min(scores)
            return spread > settings.DIVERGENCE_THRESHOLD

    async def _arbitrate_disputed(
        self,
        dialogue: SimulatedDialogue,
        disputed: list[tuple[Checkpoint, list[CheckpointResult]]],
    ) -> list[CheckpointResult]:
        dialogue_text = "\n".join(
            f"[第{t.turn_id}轮] {'外呼专员' if t.role == 'agent' else '用户'}: {t.content}"
            for t in dialogue.turns
        )

        disputed_items = []
        for cp, runs in disputed:
            disputed_items.append({
                "checkpoint": cp,
                "runs": [{"passed": r.passed, "score": r.score, "reasoning": r.explanation} for r in runs],
            })

        template = self._jinja.get_template("meta_judge.j2")
        prompt = template.render(
            dialogue_text=dialogue_text,
            disputed_items=disputed_items,
        )

        logger.debug(f"MetaJudge: arbitrating {len(disputed)} disputed checkpoints")
        raw = await self._llm.chat_json(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
        )

        cp_map = {cp.id: cp for cp, _ in disputed}
        results = []
        for r in raw.get("arbitrations", []):
            cp = cp_map.get(r.get("checkpoint_id", ""))
            if not cp:
                continue
            mapped = {
                "turn_ids": r.get("turn_ids", []),
                "quoted_text": r.get("quoted_text", ""),
                "reasoning": r.get("reasoning", ""),
                "passed": r.get("final_passed", False),
                "score": r.get("final_score", 0.5),
            }
            result = build_validated_result(mapped, cp, dialogue, default_score=0.5)
            result.explanation = f"[仲裁] {r.get('reasoning', '')}"
            results.append(result)
        return results
