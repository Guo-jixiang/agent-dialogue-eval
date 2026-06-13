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
from evaluator.judges.base import BaseJudge, build_batch_results


class FlowJudge(BaseJudge):
    dimension = EvaluationDimension.flow

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
        flow_cps = [cp for cp in checkpoints if cp.dimension == self.dimension]
        if not flow_cps:
            return []

        dialogue_text = self._dialogue_text(dialogue)

        def _on_item(item, cp):
            if item.get("status") == "not_applicable":
                return CheckpointResult(
                    checkpoint=cp, passed=True, score=1.0,
                    evidence=Evidence(turn_ids=[], quoted_text="",
                                      reasoning=item.get("reasoning", "该分支未触发")),
                    explanation=f"[未触发] {item.get('reasoning', '该跳转分支条件未触发')}",
                )
            evidence = Evidence(
                turn_ids=item.get("turn_ids", []),
                quoted_text=item.get("quoted_text", ""),
                reasoning=item.get("reasoning", ""),
                confidence=float(item.get("confidence") or 1.0),
            )
            return CheckpointResult(
                checkpoint=cp, passed=bool(item.get("passed", False)),
                score=float(item.get("score") or 0.0), evidence=evidence,
                explanation=item.get("reasoning", ""),
            )

        pairs = await self._batch_call(flow_cps, "flow_judge.j2", {"dialogue_text": dialogue_text})
        return build_batch_results(pairs, dialogue, on_item=_on_item)



