from __future__ import annotations

from jinja2 import Environment, FileSystemLoader
from loguru import logger

from config.settings import settings
from core.llm_client import LLMClient
from core.models import (
    Checkpoint,
    CheckpointResult,
    EvaluationDimension,
    SimulatedDialogue,
)
from evaluator.evidence import build_validated_result
from evaluator.judges.base import BaseJudge, build_batch_results


class SafetyJudge(BaseJudge):
    dimension = EvaluationDimension.safety

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

        template = self._jinja.get_template("safety_judge.j2")
        dialogue_text = self._dialogue_text(dialogue)
        kps = dialogue.instruction_spec.knowledge_points
        pairs = await self._batch_call(cps, "safety_judge.j2",
                                       {"dialogue_text": dialogue_text, "knowledge_points": kps})
        return build_batch_results(pairs, dialogue)
