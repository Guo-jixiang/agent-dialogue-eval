from __future__ import annotations

import re

from jinja2 import Environment, FileSystemLoader
from loguru import logger

from config.settings import settings
from core.llm_client import LLMClient
from core.models import (
    Checkpoint,
    CheckpointResult,
    ConstraintCategory,
    EvaluationDimension,
    EvaluationType,
    Evidence,
    InstructionSpec,
    SimulatedDialogue,
)
from evaluator.evidence import build_validated_result
from evaluator.judges.base import BaseJudge, build_batch_results

# Quote characters via \u escapes to avoid Python 3.12 confusable-char SyntaxError
_OPEN_QUOTES = "“‘「『"   # left double/single/CJK quotes
_CLOSE_QUOTES = "”’」』"  # right double/single/CJK quotes
_ALL_QUOTES = _OPEN_QUOTES + _CLOSE_QUOTES
_QUOTE_PAT = re.compile(
    f"[{re.escape(_OPEN_QUOTES)}]([^{re.escape(_CLOSE_QUOTES)}]+)[{re.escape(_CLOSE_QUOTES)}]"
)
_CJK_QUOTE_PAT = re.compile(
    f"[「『]([一-鿿]+)[」』]"
)


class ConstraintJudge(BaseJudge):
    dimension = EvaluationDimension.constraint

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
        target_dims = {EvaluationDimension.constraint}
        cps = [cp for cp in checkpoints if cp.dimension in target_dims]
        if not cps:
            return []

        results: list[CheckpointResult] = []
        soft_cps: list[Checkpoint] = []

        spec = dialogue.instruction_spec
        constraint_map = {c.id: c for c in spec.constraints}

        for cp in cps:
            constraint = constraint_map.get(cp.source_constraint_id or "")
            if not constraint:
                soft_cps.append(cp)
                continue

            if constraint.category == ConstraintCategory.length:
                result = self._check_length(cp, constraint, dialogue)
                results.append(result)
            elif constraint.category == ConstraintCategory.forbidden_word:
                result = self._check_forbidden_words(cp, constraint, dialogue)
                results.append(result)
            elif constraint.category == ConstraintCategory.repetition:
                result = self._check_repetition(cp, dialogue)
                results.append(result)
            else:
                soft_cps.append(cp)

        # LLM evaluation for soft/semantic constraints
        if soft_cps:
            llm_results = await self._llm_judge(dialogue, soft_cps)
            results.extend(llm_results)

        return results

    def _check_length(
        self, cp: Checkpoint, constraint, dialogue: SimulatedDialogue
    ) -> CheckpointResult:
        threshold = constraint.threshold
        if threshold is None:
            return CheckpointResult(checkpoint=cp, passed=True, score=1.0, explanation="no numeric threshold")

        agent_turns = [t for t in dialogue.turns if t.role == "agent"]
        violations = [t for t in agent_turns if len(t.content) > threshold]
        compliance_rate = 1.0 - len(violations) / max(len(agent_turns), 1)

        violation_examples = []
        violation_turn_ids = []
        for t in violations[:3]:
            violation_turn_ids.append(t.turn_id)
            violation_examples.append(f"第{t.turn_id}轮({len(t.content)}字): {t.content[:40]}...")

        evidence = Evidence(
            turn_ids=violation_turn_ids,
            quoted_text="; ".join(violation_examples) if violation_examples else "全部符合字数要求",
            reasoning=f"共{len(agent_turns)}轮，违规{len(violations)}轮，合规率{compliance_rate:.0%}",
        )
        return CheckpointResult(
            checkpoint=cp,
            passed=compliance_rate >= 0.8,
            score=compliance_rate,
            evidence=evidence,
            explanation=evidence.reasoning,
        )

    def _check_forbidden_words(
        self, cp: Checkpoint, constraint, dialogue: SimulatedDialogue
    ) -> CheckpointResult:
        desc = constraint.description

        # Extract quoted words from description (handles Chinese and Unicode quotes)
        quoted = _QUOTE_PAT.findall(desc)
        word_matches = _CJK_QUOTE_PAT.findall(desc)
        forbidden = list(set(quoted + word_matches))

        if not forbidden:
            # Fallback: comma-separated list
            parts = re.split(r"[、，,]", desc)
            forbidden = [p.strip().strip("\"'") for p in parts if 2 <= len(p.strip()) <= 6]

        agent_turns = [t for t in dialogue.turns if t.role == "agent"]
        violations = []
        for t in agent_turns:
            for word in forbidden:
                if word and word in t.content:
                    violations.append((t.turn_id, word, t.content[:50]))

        passed = len(violations) == 0
        evidence = Evidence(
            turn_ids=[v[0] for v in violations],
            quoted_text="; ".join(f"第{v[0]}轮含'{v[1]}'" for v in violations[:3]) if violations else "未检测到禁用词",
            reasoning=f"检测词汇: {forbidden}; 违规{len(violations)}次",
        )
        return CheckpointResult(
            checkpoint=cp,
            passed=passed,
            score=0.0 if violations else 1.0,
            evidence=evidence,
            explanation=evidence.reasoning,
        )

    def _check_repetition(self, cp: Checkpoint, dialogue: SimulatedDialogue) -> CheckpointResult:
        agent_turns = [t for t in dialogue.turns if t.role == "agent"]
        if len(agent_turns) < 2:
            return CheckpointResult(checkpoint=cp, passed=True, score=1.0, explanation="too few turns")

        repeated = 0
        for i in range(1, len(agent_turns)):
            prev, curr = agent_turns[i - 1].content, agent_turns[i].content
            overlap = len(set(prev) & set(curr)) / max(len(set(curr)), 1)
            if overlap > 0.7 and len(curr) > 10:
                repeated += 1

        compliance_rate = 1.0 - repeated / max(len(agent_turns) - 1, 1)
        evidence = Evidence(
            turn_ids=[],
            quoted_text=f"检测到{repeated}次高重复轮次",
            reasoning=f"非重复率{compliance_rate:.0%}",
        )
        return CheckpointResult(
            checkpoint=cp,
            passed=compliance_rate >= 0.8,
            score=compliance_rate,
            evidence=evidence,
            explanation=evidence.reasoning,
        )

    async def _llm_judge(
        self, dialogue: SimulatedDialogue, checkpoints: list[Checkpoint]
    ) -> list[CheckpointResult]:
        dialogue_text = self._dialogue_text(dialogue)
        pairs = await self._batch_call(checkpoints, "constraint_judge.j2",
                                       {"dialogue_text": dialogue_text})
        return build_batch_results(pairs, dialogue)
