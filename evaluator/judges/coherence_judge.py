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


class CoherenceJudge(BaseJudge):
    dimension = EvaluationDimension.coherence

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
        pairs = await self._batch_call(cps, "coherence_judge.j2", {"dialogue_text": dialogue_text})
        return build_batch_results(pairs, dialogue)
