from __future__ import annotations


from jinja2 import Environment, FileSystemLoader
from loguru import logger

from config.settings import settings
from core.llm_client import LLMClient
from core.models import (
    Checkpoint,
    CheckpointResult,
    EvaluationDimension,
    Evidence,
    SimulatedDialogue,
)
from evaluator.evidence import build_validated_result
from evaluator.judges.base import BaseJudge, build_batch_results


class KnowledgeJudge(BaseJudge):
    dimension = EvaluationDimension.knowledge

    def __init__(self, llm_client: LLMClient):
        self._llm = llm_client
        self._jinja = Environment(
            loader=FileSystemLoader(str(settings.PROMPTS_DIR)),
            trim_blocks=True,
            lstrip_blocks=True,
        )

    async def judge(
        self,
        dialogue: SimulatedDialogue,
        checkpoints: list[Checkpoint],
    ) -> list[CheckpointResult]:
        knowledge_cps = [cp for cp in checkpoints if cp.dimension == self.dimension]
        if not knowledge_cps:
            return []

        dialogue_text = self._dialogue_text(dialogue)
        kps = dialogue.instruction_spec.knowledge_points

        def _on_item(item, cp):
            if item.get("status") == "not_triggered":
                return CheckpointResult(
                    checkpoint=cp, passed=True, score=1.0,
                    evidence=Evidence(turn_ids=[], quoted_text="",
                                      reasoning=item.get("reasoning", "按需传达类型，用户未询问"),
                                      confidence=1.0),
                    explanation=f"[未触发] {item.get('reasoning', '按需传达类型，用户未询问相关问题')}",
                )
            return build_validated_result(item, cp, dialogue)

        pairs = await self._batch_call(knowledge_cps, "knowledge_judge.j2",
                                       {"dialogue_text": dialogue_text, "knowledge_points": kps})
        return build_batch_results(pairs, dialogue, on_item=_on_item)
