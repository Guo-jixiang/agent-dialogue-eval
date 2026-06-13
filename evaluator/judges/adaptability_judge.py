from __future__ import annotations


from jinja2 import Environment, FileSystemLoader
from loguru import logger

from config.settings import settings
from core.llm_client import LLMClient
from core.models import (
    Checkpoint,
    CheckpointResult,
    Evidence,
    EvaluationDimension,
    SimulatedDialogue,
)
from evaluator.evidence import build_validated_result
from evaluator.judges.base import BaseJudge, build_batch_results


class AdaptabilityJudge(BaseJudge):
    dimension = EvaluationDimension.adaptability

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
        cps = [cp for cp in checkpoints if cp.dimension == self.dimension]
        if not cps:
            return []

        dialogue_text = self._dialogue_text(dialogue)
        archetype = dialogue.persona.archetype.value
        persona_desc = dialogue.persona.behavior_description

        def _on_item(item, cp):
            if item.get("status") == "not_triggered":
                return CheckpointResult(
                    checkpoint=cp, passed=True, score=1.0,
                    evidence=Evidence(turn_ids=[], quoted_text="",
                                      reasoning=item.get("reasoning", "用户未触发异常场景"),
                                      confidence=1.0),
                    explanation=f"[未触发] {item.get('reasoning', '用户未触发异常场景')}",
                )
            return build_validated_result(item, cp, dialogue)

        pairs = await self._batch_call(cps, "adaptability_judge.j2",
                                       {"dialogue_text": dialogue_text,
                                        "persona_archetype": archetype,
                                        "persona_description": persona_desc})
        return build_batch_results(pairs, dialogue, on_item=_on_item)
