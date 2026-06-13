"""Standalone evaluation entry point.

``evaluate_dialogue()`` can be called with any dialogue (internally
simulated or externally provided) and an ``InstructionSpec``.  It is
the same logic that was previously inlined inside
``run_full_pipeline()``.
"""

from __future__ import annotations

import asyncio

from loguru import logger

from core.llm_client import LLMClient
from core.models import (
    Checkpoint,
    EvaluationReport,
    InstructionSpec,
    SimulatedDialogue,
)
from evaluator.decomposer import decompose
from evaluator.judges.adaptability_judge import AdaptabilityJudge
from evaluator.judges.coherence_judge import CoherenceJudge
from evaluator.judges.consistency_wrapper import SelfConsistentJudge
from evaluator.judges.constraint_judge import ConstraintJudge
from evaluator.judges.flow_judge import FlowJudge
from evaluator.judges.knowledge_judge import KnowledgeJudge
from evaluator.judges.safety_judge import SafetyJudge
from evaluator.judges.style_judge import StyleJudge
from evaluator.scorer import score_dialogue


async def evaluate_dialogue(
    dialogue: SimulatedDialogue,
    spec: InstructionSpec,
    eval_client: LLMClient,
) -> EvaluationReport:
    """Evaluate a single dialogue against an instruction specification.

    Parameters
    ----------
    dialogue:
        The dialogue to evaluate.
    spec:
        The parsed instruction specification.
    eval_client:
        LLM client used by all judges.

    Returns
    -------
    EvaluationReport
    """
    checkpoints = decompose(spec)

    # Wrap each judge with self-consistency (multi-run + arbitration)
    sc_flow = SelfConsistentJudge(FlowJudge(eval_client), eval_client)
    sc_constraint = SelfConsistentJudge(ConstraintJudge(eval_client), eval_client)
    sc_knowledge = SelfConsistentJudge(KnowledgeJudge(eval_client), eval_client)
    sc_style = SelfConsistentJudge(StyleJudge(eval_client), eval_client)
    sc_coherence = SelfConsistentJudge(CoherenceJudge(eval_client), eval_client)
    sc_safety = SelfConsistentJudge(SafetyJudge(eval_client), eval_client)
    sc_adaptability = SelfConsistentJudge(AdaptabilityJudge(eval_client), eval_client)

    # Run all 7 judges in parallel with per-judge timing
    import time as _time
    judges = [sc_flow, sc_constraint, sc_knowledge, sc_style, sc_coherence, sc_safety, sc_adaptability]

    async def _timed_judge(j):
        _tj = _time.time()
        r = await j.judge(dialogue, checkpoints)
        logger.debug(f"[TIMING] Judge {j._judge.dimension.value}: {_time.time() - _tj:.1f}s ({len(r)} results)")
        return r

    _t_judges = _time.time()
    all_results = await asyncio.gather(*(_timed_judge(j) for j in judges))
    logger.debug(f"[TIMING] All 7 judges total: {_time.time() - _t_judges:.1f}s")

    final_results = []
    for results in all_results:
        final_results.extend(results)

    return score_dialogue(dialogue, final_results)
