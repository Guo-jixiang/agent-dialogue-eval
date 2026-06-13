from __future__ import annotations

from abc import ABC, abstractmethod

from loguru import logger

from core.models import Checkpoint, CheckpointResult, EvaluationDimension, SimulatedDialogue

# Max checkpoints per LLM batch call — prevents JSON truncation under high concurrency.
_MAX_PER_BATCH = 8


class BaseJudge(ABC):
    dimension: EvaluationDimension

    @abstractmethod
    async def judge(
        self,
        dialogue: SimulatedDialogue,
        checkpoints: list[Checkpoint],
    ) -> list[CheckpointResult]:
        """Evaluate all checkpoints in this judge's dimension."""
        ...

    def _dialogue_text(self, dialogue: SimulatedDialogue) -> str:
        lines = []
        for turn in dialogue.turns:
            role_label = "外呼专员" if turn.role == "agent" else "用户"
            lines.append(f"[第{turn.turn_id}轮] {role_label}: {turn.content}")
        return "\n".join(lines)

    async def _batch_call(
        self,
        cps: list[Checkpoint],
        template_name: str,
        render_kwargs: dict,
        temperature: float = 0.1,
    ) -> list[tuple[dict | None, Checkpoint]]:
        """Split checkpoints into chunks, call LLM per chunk, return
        ``(raw_item_or_None, checkpoint)`` pairs.

        If a chunk call fails or the LLM omits a checkpoint, the pair is
        ``(None, checkpoint)`` — callers should produce a fallback result.
        """
        template = self._jinja.get_template(template_name)
        all_pairs: list[tuple[dict | None, Checkpoint]] = []

        for i in range(0, len(cps), _MAX_PER_BATCH):
            chunk = cps[i : i + _MAX_PER_BATCH]
            logger.debug(
                f"{self.dimension.value}: batch chunk {i // _MAX_PER_BATCH + 1} "
                f"({len(chunk)} checkpoints)"
            )
            prompt = template.render(**render_kwargs, checkpoints=chunk)
            try:
                raw = await self._llm.chat_json(
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                )
            except Exception as e:
                logger.warning(f"{self.dimension.value} chunk failed: {e}")
                for cp in chunk:
                    all_pairs.append((None, cp))
                continue

            cp_map = {cp.id: cp for cp in chunk}
            seen: set[str] = set()
            items = raw if isinstance(raw, list) else raw.get("results", [])
            for item in items:
                cp_id = item.get("checkpoint_id", "")
                cp = cp_map.get(cp_id)
                if cp and cp_id not in seen:
                    seen.add(cp_id)
                    all_pairs.append((item, cp))

            for cp in chunk:
                if cp.id not in seen:
                    all_pairs.append((None, cp))

        return all_pairs


def build_batch_results(
    pairs: list[tuple[dict | None, Checkpoint]],
    dialogue: SimulatedDialogue,
    *,
    default_score: float = 0.0,
    default_passed: bool = False,
    on_item: callable | None = None,
) -> list[CheckpointResult]:
    """Convert ``_batch_call`` pairs into ``CheckpointResult`` list.

    ``on_item(item, cp)`` is called for each successfully-parsed pair and
    should return a ``CheckpointResult``.  If ``None``, ``build_validated_result``
    is used with the given *default_score* / *default_passed*.
    """
    from evaluator.evidence import build_validated_result  # deferred import

    results: list[CheckpointResult] = []
    for item, cp in pairs:
        if item is None:
            results.append(CheckpointResult(
                checkpoint=cp, passed=default_passed, score=float(default_score),
                explanation="LLM未返回此检查点结果",
            ))
        elif on_item is not None:
            results.append(on_item(item, cp))
        else:
            results.append(build_validated_result(
                item, cp, dialogue,
                default_score=default_score, default_passed=default_passed,
            ))
    return results
