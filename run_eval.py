"""Main orchestration pipeline with checkpoint/resume and batch processing."""
from __future__ import annotations

import asyncio
import hashlib
import time
import json
import os
import uuid
from pathlib import Path
from typing import Any

from loguru import logger

from config.settings import settings
from core.agent import BaseAgent
from core.llm_client import LLMClient, make_eval_client
from core.models import PersonaArchetype, Persona, SimulatedDialogue, EvaluationReport
from evaluator.decomposer import decompose
from evaluator.eval_pipeline import evaluate_dialogue
from evaluator.scorer import aggregate_reports
from parser.instruction_parser import InstructionParser
from report.generator import ReportGenerator
from simulator.persona import generate_personas
from simulator.scenario import run_scenario_matrix, _resolve_personas


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _instruction_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _checkpoint_path(output_dir: Path) -> Path:
    return output_dir / "checkpoint.json"


def _load_checkpoint(output_dir: Path, instruction_text: str) -> dict[str, Any] | None:
    """Load checkpoint if it exists and matches the current instruction."""
    cp_path = _checkpoint_path(output_dir)
    if not cp_path.exists():
        return None
    try:
        state = json.loads(cp_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        logger.warning("Checkpoint file corrupted, starting fresh")
        return None
    if state.get("instruction_hash") != _instruction_hash(instruction_text):
        logger.info("Instruction changed — checkpoint invalid, starting fresh")
        return None
    completed = len(state.get("completed_dialogue_ids", []))
    if state.get("phase") == "done":
        logger.info(f"Checkpoint: run already complete ({completed} dialogues)")
    else:
        logger.info(f"Resuming from checkpoint: {completed} dialogues already completed")
    return state


def _save_checkpoint(output_dir: Path, state: dict[str, Any]) -> None:
    """Atomically write checkpoint (tmp → rename)."""
    cp_path = _checkpoint_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = cp_path.with_suffix(".json.tmp")
    tmp_path.write_text(
        json.dumps(state, ensure_ascii=False, default=str), encoding="utf-8"
    )
    os.replace(tmp_path, cp_path)


# ---------------------------------------------------------------------------
# Client factory
# ---------------------------------------------------------------------------

def _make_clients(
    eval_api_key: str = "",
    eval_base_url: str = "",
    eval_model: str = "",
    sim_api_key: str = "",
    sim_base_url: str = "",
    sim_model: str = "",
    agent_api_key: str = "",
    agent_base_url: str = "",
    agent_model: str = "",
) -> tuple[LLMClient, LLMClient, LLMClient]:
    eval_client = make_eval_client(
        api_key=eval_api_key,
        base_url=eval_base_url,
        model=eval_model,
    )
    sim_client = LLMClient(
        api_key=sim_api_key or settings.SIM_LLM_API_KEY,
        base_url=sim_base_url or settings.SIM_LLM_BASE_URL,
        model=sim_model or settings.SIM_LLM_MODEL,
    )
    agent_client = LLMClient(
        api_key=agent_api_key or settings.AGENT_LLM_API_KEY,
        base_url=agent_base_url or settings.AGENT_LLM_BASE_URL,
        model=agent_model or settings.AGENT_LLM_MODEL,
    )
    return eval_client, sim_client, agent_client


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

async def run_full_pipeline(
    instruction_text: str,
    output_dir: Path | str = "reports",
    archetypes: list[PersonaArchetype] | None = None,
    dimensions: dict[str, list[str]] | None = None,
    agent: BaseAgent | None = None,
    personas: list[Persona] | None = None,
    batch_size: int | None = None,
    resume: bool = True,
    on_progress: callable | None = None,
    # LLM config overrides (empty = use .env)
    eval_api_key: str = "",
    eval_base_url: str = "",
    eval_model: str = "",
    sim_api_key: str = "",
    sim_base_url: str = "",
    sim_model: str = "",
    agent_api_key: str = "",
    agent_base_url: str = "",
    agent_model: str = "",
) -> dict:
    """Run the complete eval pipeline with checkpoint/resume and batching.

    Parameters
    ----------
    batch_size:
        Number of dialogues to simulate + evaluate per batch.
        Defaults to ``settings.EVAL_BATCH_SIZE``.
    resume:
        If True, load existing checkpoint and skip already-completed dialogues.
    on_progress:
        Optional async callback(dict) for streaming progress. Called with:
        ``{"type": "phase", "phase": "parsing"|"enriching"|"simulating"|"evaluating"|"report"}``
        ``{"type": "dialogue", "dialogue_id": ..., "persona_label": ..., "score": ...}``
    """
    output_dir = Path(output_dir)
    batch_size = batch_size or settings.EVAL_BATCH_SIZE

    async def _emit(event: dict) -> None:
        if on_progress:
            await on_progress(event)

    eval_client, sim_client, agent_client = _make_clients(
        eval_api_key=eval_api_key,
        eval_base_url=eval_base_url,
        eval_model=eval_model,
        sim_api_key=sim_api_key,
        sim_base_url=sim_base_url,
        sim_model=sim_model,
        agent_api_key=agent_api_key,
        agent_base_url=agent_base_url,
        agent_model=agent_model,
    )

    # ── Checkpoint ─────────────────────────────────────────────────────
    checkpoint: dict[str, Any] | None = None
    skip_ids: set[str] = set()
    completed_reports: list[dict] = []
    completed_dialogues: list[dict] = []
    start_batch = 0

    if resume:
        checkpoint = _load_checkpoint(output_dir, instruction_text)
        if checkpoint is not None:
            skip_ids = set(checkpoint.get("completed_dialogue_ids", []))
            completed_reports = checkpoint.get("dialogue_reports", [])
            completed_dialogues = checkpoint.get("dialogues", [])
            start_batch = checkpoint.get("current_batch", 0)
            if checkpoint.get("phase") == "done":
                # Always regenerate reports to apply latest templates and fixes,
                # even if checkpoint was completed. This ensures all output formats
                # (JSON, MD, HTML) are up-to-date with current code.
                logger.info("Run checkpoint found as 'done' — regenerating report files with latest templates")
                # Fall through to Phase 4 to regenerate report files
                # but skip simulation/evaluation (start_batch = total_batches)
                start_batch = 999  # skip all batches

    run_id = checkpoint.get("run_id", uuid.uuid4().hex[:12]) if checkpoint else uuid.uuid4().hex[:12]

    # ── Phase 1: Parse instruction ─────────────────────────────────────
    _t_total = time.time()
    logger.info("=== Phase 1: Parsing instruction ===")
    await _emit({"type": "phase", "phase": "parsing", "message": "解析指令中…"})
    _t1 = time.time()
    parser = InstructionParser(eval_client)
    spec = await parser.parse(instruction_text)
    logger.info(f"[TIMING] Phase 1 parse: {time.time() - _t1:.1f}s")
    logger.info(f"Parsed: {len(spec.flow_graph)} nodes, {len(spec.constraints)} "
                f"constraints, {len(spec.knowledge_points)} KPs")

    # ── Resolve all personas once ──────────────────────────────────────
    await _emit({"type": "phase", "phase": "enriching", "message": "生成用户角色中…"})
    _t2 = time.time()
    if personas:
        all_personas = personas
        logger.info(f"Using {len(all_personas)} pre-generated personas")
    else:
        all_personas = await _resolve_personas(dimensions, archetypes, eval_client)
    logger.info(f"[TIMING] Phase 2 personas: {time.time() - _t2:.1f}s, {len(all_personas)} personas")
    total_batches = (len(all_personas) + batch_size - 1) // batch_size
    logger.info(f"Total: {len(all_personas)} personas → {total_batches} batches "
                f"(batch_size={batch_size})")
    await _emit({"type": "phase", "phase": "simulating", "message": "模拟对话中…",
                 "total": len(all_personas)})

    # Decompose checkpoints once (shared across all dialogues)
    checkpoints = decompose(spec)
    logger.info(f"Decomposed into {len(checkpoints)} checkpoints")

    dialogue_reports: list[dict] = list(completed_reports)
    saved_dialogues: list[dict] = list(completed_dialogues)

    # ── Phase 2+3: Simulate + Evaluate, batch by batch ─────────────────
    for batch_idx in range(start_batch, total_batches):
        batch_start = batch_idx * batch_size
        batch_personas = all_personas[batch_start:batch_start + batch_size]

        # Filter out already-completed
        pending = [
            p for p in batch_personas
            if f"{p.label}" not in skip_ids
        ]
        if not pending:
            logger.info(f"Batch {batch_idx + 1}/{total_batches}: all done, skipping")
            continue

        logger.info(f"=== Batch {batch_idx + 1}/{total_batches}: "
                    f"simulating {len(pending)} dialogues ===")
        _t_sim = time.time()
        batch_dialogues = await run_scenario_matrix(
            spec=spec,
            agent_client=agent_client,
            simulator_client=sim_client,
            max_turns=settings.MAX_TURNS,
            eval_client=eval_client,
            agent=agent,
            personas=pending,
        )

        logger.info(f"[TIMING] Batch {batch_idx + 1} simulate: {time.time() - _t_sim:.1f}s, {len(batch_dialogues)} dialogues")
        logger.info(f"=== Batch {batch_idx + 1}/{total_batches}: "
                    f"evaluating {len(batch_dialogues)} dialogues ===")
        _t_eval = time.time()
        eval_sem = asyncio.Semaphore(settings.EVAL_CONCURRENCY)

        async def _eval_one(dialogue: SimulatedDialogue) -> tuple[str, dict, dict]:
            dialogue_id = f"{dialogue.persona.label}"
            async with eval_sem:
                logger.info(f"Evaluating {dialogue_id}")
                report = await evaluate_dialogue(dialogue, spec, eval_client)
                logger.info(f"  Score {dialogue_id}: {report.overall_score}")
                # Emit progress immediately when each dialogue completes
                await _emit({
                    "type": "dialogue",
                    "dialogue_id": dialogue_id,
                    "persona_label": dialogue.persona.label,
                    "score": report.overall_score,
                    # Feed: full turn list for real-time display in the UI
                    "turns": [
                        {"role": t.role, "content": t.content}
                        for t in dialogue.turns
                    ],
                    "termination_reason": (
                        dialogue.termination_reason.value
                        if hasattr(dialogue.termination_reason, "value")
                        else str(dialogue.termination_reason)
                    ),
                })
                return dialogue_id, report.model_dump(), dialogue.model_dump()

        # Filter out already-completed dialogues
        pending_dialogues = [
            d for d in batch_dialogues
            if f"{d.persona.label}" not in skip_ids
        ]
        if pending_dialogues:
            eval_results = await asyncio.gather(
                *(_eval_one(d) for d in pending_dialogues), return_exceptions=True
            )
            failed_count = 0
            for result in eval_results:
                if isinstance(result, Exception):
                    failed_count += 1
                    err_msg = f"{type(result).__name__}: {result}"
                    logger.error(f"Evaluation failed: {err_msg}")
                    await _emit({"type": "eval_error", "error": str(result)[:300]})
                    continue
                dialogue_id, report_dict, dialogue_dict = result
                dialogue_reports.append(report_dict)
                saved_dialogues.append(dialogue_dict)
                skip_ids.add(dialogue_id)

            # If ALL evaluations failed, bail out early
            if failed_count > 0 and failed_count == len(eval_results):
                msg = f"全部 {failed_count} 个对话评测失败，请检查评判模型配置（key/url/model）是否正确"
                logger.error(msg)
                await _emit({"type": "error", "error": msg})
                raise RuntimeError(msg)

        logger.info(f"[TIMING] Batch {batch_idx + 1} evaluate: {time.time() - _t_eval:.1f}s")

        # ── Checkpoint after batch ─────────────────────────────────────
        _save_checkpoint(output_dir, {
            "run_id": run_id,
            "instruction_hash": _instruction_hash(instruction_text),
            "completed_dialogue_ids": sorted(skip_ids),
            "dialogue_reports": dialogue_reports,
            "dialogues": saved_dialogues,
            "current_batch": batch_idx + 1,
            "phase": "evaluating",
        })

    # ── Phase 4: Analysis Agents (Layer 4 → 5 → 6) ────────────────────
    logger.info("=== Phase 4: Analysis agents ===")
    await _emit({"type": "phase", "phase": "analyzing", "message": "分析评测结果中…"})
    _t4 = time.time()

    from evaluator.report_agent import ReportAgent
    from evaluator.judge_agent import JudgeAgent
    from evaluator.attribution_agent import AttributionAgent

    report_agent = ReportAgent(eval_client)
    reports = [EvaluationReport(**r) for r in dialogue_reports]
    dialogues_list = [SimulatedDialogue(**d) for d in saved_dialogues]

    # Layer 4: per-dialogue analysis (parallel)
    async def _analyze_one(report, dia):
        try:
            return await report_agent.analyze(report, dia)
        except Exception as e:
            logger.warning(f"ReportAgent failed for {report.persona_label}: {e}")
            return None

    _t_l4 = time.time()
    l4_results = await asyncio.gather(
        *(_analyze_one(r, d) for r, d in zip(reports, dialogues_list)),
        return_exceptions=False,
    )
    analyses = [a for a in l4_results if a is not None]
    logger.info(f"[TIMING] Layer 4 (report agent x{len(analyses)}): {time.time() - _t_l4:.1f}s")

    # Aggregate (needed for Layer 5 dimension scores)
    agg_report = aggregate_reports(spec, reports, instruction_text)

    # Layer 5: judge — cross-comparison
    judge_agent = JudgeAgent(eval_client)
    _t_l5 = time.time()
    try:
        judge_result = await judge_agent.analyze(analyses, agg_report.score_by_dimension)
    except Exception as e:
        logger.warning(f"JudgeAgent failed: {e}")
        judge_result = None
    logger.info(f"[TIMING] Layer 5 (judge): {time.time() - _t_l5:.1f}s")

    # Layer 6: attribution — root cause analysis
    _t_l6 = time.time()
    bottom_labels = {r.persona_label for r in (judge_result.bottom5 if judge_result else [])}
    bottom_analyses = [a for a in analyses if a.persona_label in bottom_labels] if bottom_labels else analyses[:5]

    failed_cps: dict[str, list[str]] = {}
    for r in reports:
        cps = []
        for ds in r.dimension_scores:
            for cr in ds.checkpoint_results:
                if not cr.passed:
                    cps.append(cr.checkpoint.description)
        if cps:
            failed_cps[r.persona_label] = cps

    attr_agent = AttributionAgent(eval_client)
    attr_result = None
    try:
        attr_result = await attr_agent.analyze(bottom_analyses, failed_cps)
    except Exception as e:
        logger.warning(f"AttributionAgent failed: {e}")
    logger.info(f"[TIMING] Layer 6 (attribution): {time.time() - _t_l6:.1f}s")

    # Attach analysis to aggregated report BEFORE rendering
    agg_report.per_dialogue_analyses = [a.model_dump() for a in analyses]
    agg_report.judge_analysis = judge_result.model_dump() if judge_result else None
    agg_report.attribution = attr_result.model_dump() if attr_result else None

    # Build structured 3-layer summary (cross-dialogue aggregation)
    try:
        agg_report.structured_summary = report_agent.build_structured_summary(
            agg_report, analyses
        )
    except Exception as e:
        logger.warning(f"build_structured_summary failed: {e}")
        agg_report.structured_summary = None

    # Now generate reports (analysis included in rendered output)
    generator = ReportGenerator(output_dir)
    paths = generator.generate(agg_report, dialogues_list)

    logger.info(f"[TIMING] Phase 4 analysis: {time.time() - _t4:.1f}s")
    logger.info(f"[TIMING] TOTAL pipeline: {time.time() - _t_total:.1f}s")
    logger.info("Report generated:")
    for fmt, path in paths.items():
        logger.info(f"  {fmt}: {path}")

    # ── Mark done ──────────────────────────────────────────────────────
    _save_checkpoint(output_dir, {
        "run_id": run_id,
        "instruction_hash": _instruction_hash(instruction_text),
        "completed_dialogue_ids": sorted(skip_ids),
        "dialogue_reports": dialogue_reports,
        "dialogues": saved_dialogues,
        "current_batch": total_batches,
        "phase": "done",
    })

    return {
        "report_id": agg_report.id,
        "overall_score": agg_report.overall_score,
        "paths": {k: str(v) for k, v in paths.items()},
        "report": agg_report,
        "dialogues": dialogues_list,
    }


def _result_from_checkpoint(
    output_dir: Path,
    reports_data: list[dict],
    dialogues_data: list[dict],
) -> dict:
    """Reconstruct result when run is already complete."""
    reports = [EvaluationReport(**r) for r in reports_data]
    dialogues = [SimulatedDialogue(**d) for d in dialogues_data]

    report_id = reports[0].id if reports else output_dir.name
    overall = reports[0].overall_score if reports else 0.0

    paths = {}
    for ext in ("md", "html", "json"):
        p = output_dir / f"report_{report_id}.{ext}"
        if p.exists():
            paths[ext] = str(p)

    return {
        "report_id": report_id,
        "overall_score": overall,
        "paths": paths,
        "report": reports,
        "dialogues": dialogues,
    }


if __name__ == "__main__":
    import sys

    instruction_file = sys.argv[1] if len(sys.argv) > 1 else "tests/fixtures/rider_instruction.txt"
    text = Path(instruction_file).read_text(encoding="utf-8")
    asyncio.run(run_full_pipeline(text))
