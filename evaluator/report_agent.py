"""Layer 4: Report Agent — per-dialogue deep analysis + cross-dialogue aggregation."""

from __future__ import annotations

from collections import defaultdict
from jinja2 import Environment, FileSystemLoader

from config.settings import settings
from core.llm_client import LLMClient, RoundRobinLLMClient
from core.models import AggregatedReport, EvaluationReport, PerDialogueAnalysis, SimulatedDialogue

_DIM_LABELS = {
    "flow": "流程完整性",
    "constraint": "话术规范",
    "knowledge": "知识准确性",
    "style": "风格语气",
    "coherence": "对话连贯性",
    "safety": "安全合规",
    "adaptability": "异常应对",
}

_SCORE_GRADE = {
    90: ("优秀", "强项，建议保持"),
    75: ("良好", "基本满足要求，可进一步优化"),
    60: ("及格", "存在明显不足，需要改进"),
    0:  ("不及格", "严重问题，必须修复"),
}

def _grade(score_pct: float) -> tuple[str, str]:
    for threshold, (label, hint) in sorted(_SCORE_GRADE.items(), reverse=True):
        if score_pct >= threshold:
            return label, hint
    return "不及格", "严重问题，必须修复"


def _overall_grade(score: float) -> tuple[str, str]:
    """Return (label, deploy_advice) for overall score."""
    if score >= 90:
        return "优秀", "可直接上线，建议定期巡检"
    if score >= 75:
        return "良好", "建议修复薄弱点后上线"
    if score >= 60:
        return "及格", "建议优化后再上线，重点修复红色问题"
    return "不及格", "暂不建议上线，需要系统性优化"


class ReportAgent:
    """Generate per-dialogue analysis from evaluation results."""

    def __init__(self, llm_client: LLMClient | RoundRobinLLMClient):
        self._llm = llm_client
        self._jinja = Environment(
            loader=FileSystemLoader(str(settings.PROMPTS_DIR)),
            trim_blocks=True, lstrip_blocks=True,
        )

    async def analyze(
        self, report: EvaluationReport, dialogue: SimulatedDialogue,
    ) -> PerDialogueAnalysis:
        template = self._jinja.get_template("report_agent.j2")

        turns = dialogue.turns if hasattr(dialogue, 'turns') else []
        turn_count = len(turns) if turns else 0
        dialogue_text = "\n".join(
            f"[{t.turn_id}] {'外呼专员' if t.role == 'agent' else '用户'}: {t.content}"
            for t in turns[-20:]
        ) if turns else ""

        dim_scores = []
        for ds in report.dimension_scores:
            dim_scores.append({
                "key": ds.dimension.value,
                "label": _DIM_LABELS.get(ds.dimension.value, ds.dimension.value),
                "score_pct": round(ds.score * 100),
            })

        checkpoints = []
        for cr in report.dimension_scores:
            for r in cr.checkpoint_results:
                checkpoints.append({
                    "passed": r.passed,
                    "desc": r.checkpoint.description,
                    "score_pct": round(r.score * 100),
                    "reason": r.explanation[:120],
                })

        prompt = template.render(
            persona_label=report.persona_label,
            overall_score=round(report.overall_score),
            turn_count=turn_count,
            termination_reason=(dialogue.termination_reason.value
                                if hasattr(dialogue, 'termination_reason')
                                and dialogue.termination_reason else "unknown"),
            dimension_scores=dim_scores,
            checkpoints=checkpoints[:15],
            dialogue_text=dialogue_text,
        )

        raw = await self._llm.chat_json(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )

        return PerDialogueAnalysis(
            dialogue_id=dialogue.id if hasattr(dialogue, 'id') else "",
            persona_label=report.persona_label,
            overall_score=report.overall_score,
            success=raw.get("success", False),
            strengths=raw.get("strengths", []),
            weaknesses=raw.get("weaknesses", []),
            suggestions=raw.get("suggestions", []),
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Cross-dialogue aggregation (produces the structured_summary field)
    # ──────────────────────────────────────────────────────────────────────────

    def build_structured_summary(
        self,
        agg_report: AggregatedReport,
        per_analyses: list[PerDialogueAnalysis],
    ) -> dict:
        """
        Build a structured, cross-dialogue summary for the new three-layer report.

        Returns a dict with keys:
          - overall_grade: str
          - deploy_advice: str
          - core_conclusion: {strengths: str, top_weakness: str, top_priority: str}
          - grouped_issues: list of issue groups, each with:
              {severity: 'critical'|'warning'|'ok',
               title: str, frequency: str, example: str, suggestion: str}
          - dim_summary: list of {key, label, score_pct, grade, color}
          - persona_summary: list of {label, score, grade, key_issues: list[str]}
        """
        reports = agg_report.dialogue_reports
        n = len(reports)

        # ── Overall grade ──────────────────────────────────────────────
        overall_grade, deploy_advice = _overall_grade(agg_report.overall_score)

        # ── Dimension summary ──────────────────────────────────────────
        _DIM_COLORS = {
            "flow": "#a78bfa", "constraint": "#fbbf24", "knowledge": "#34d399",
            "style": "#60a5fa", "coherence": "#f472b6", "safety": "#f87171",
            "adaptability": "#2dd4bf",
        }
        dim_summary = []
        for k, v in agg_report.score_by_dimension.items():
            pct = round(v * 100)
            grade, _ = _grade(pct)
            dim_summary.append({
                "key": k,
                "label": _DIM_LABELS.get(k, k),
                "score_pct": pct,
                "grade": grade,
                "color": _DIM_COLORS.get(k, "#888"),
            })
        dim_summary.sort(key=lambda d: d["score_pct"])

        # ── Cross-dialogue issue aggregation ──────────────────────────
        # Group checkpoint failures by checkpoint description across all dialogues
        failure_counts: dict[str, dict] = {}  # desc → {count, examples, dim, is_hard}
        for dr in reports:
            hard_ids = set()
            if agg_report.instruction_spec:
                from core.models import ConstraintType
                hard_ids = {
                    c.id for c in agg_report.instruction_spec.constraints
                    if c.type == ConstraintType.hard
                }
            for ds in dr.dimension_scores:
                for cr in ds.checkpoint_results:
                    if cr.passed:
                        continue
                    desc = cr.checkpoint.description
                    if desc not in failure_counts:
                        failure_counts[desc] = {
                            "count": 0,
                            "dim": _DIM_LABELS.get(ds.dimension.value, ds.dimension.value),
                            "dim_key": ds.dimension.value,
                            "is_hard": cr.checkpoint.source_constraint_id in hard_ids
                                        if cr.checkpoint.source_constraint_id else False,
                            "examples": [],
                        }
                    failure_counts[desc]["count"] += 1
                    # Collect one short example
                    if len(failure_counts[desc]["examples"]) < 2:
                        evidence = ""
                        if cr.evidence and cr.evidence.quoted_text:
                            evidence = cr.evidence.quoted_text[:80]
                        elif cr.explanation:
                            evidence = cr.explanation[:80]
                        if evidence:
                            failure_counts[desc]["examples"].append(evidence)

        # Build grouped issues
        grouped_issues = []
        for desc, info in sorted(
            failure_counts.items(),
            key=lambda x: (-x[1]["is_hard"], -x[1]["count"])
        ):
            freq = info["count"]
            freq_str = f"{freq}/{n} 次对话" if n > 1 else "出现"
            severity = "critical" if info["is_hard"] or freq >= n else (
                "warning" if freq >= max(2, n // 2) else "info"
            )
            example = info["examples"][0] if info["examples"] else ""
            grouped_issues.append({
                "severity": severity,
                "dim": info["dim"],
                "dim_key": info["dim_key"],
                "title": desc[:80],
                "frequency": freq_str,
                "freq_count": freq,
                "total": n,
                "example": example,
                "is_hard": info["is_hard"],
                "suggestion": "",  # will be filled by LLM or rule-based
            })

        # Rule-based suggestions for common issues
        _SUGGESTION_RULES = [
            ("字", "回复字数超限", "建议精简话术，删除冗余称谓，把首轮开场白控制在约30字以内"),
            ("流程节点", "流程节点缺失", "建议在对话脚本中明确加入该节点的触发条件和示例话术"),
            ("知识点", "知识传达不准确", "建议在指令中补充该知识点的准确表述和反例"),
            ("挽留", "挽留环节缺失", "建议在骑手表示不愿跑单时，添加挽留话术分支"),
            ("排名", "报名规则说明缺失", "建议在鼓励骑手后，主动说明排名竞争规则"),
            ("语气", "语气风格问题", "建议调整语气，使对话更自然流畅，像真实通话"),
            ("重复", "话术重复", "建议增加话术多样性，换不同措辞表达同一意思"),
        ]
        for issue in grouped_issues:
            for keyword, _, suggestion in _SUGGESTION_RULES:
                if keyword in issue["title"]:
                    issue["suggestion"] = suggestion
                    break
            if not issue["suggestion"]:
                issue["suggestion"] = f"建议重点优化「{issue['dim']}」维度，排查该问题的根本原因"

        # ── Core conclusion (3 sentences) ─────────────────────────────
        strong_dims = [d for d in dim_summary if d["score_pct"] >= 80]
        weak_dims = [d for d in dim_summary if d["score_pct"] < 60]
        top_issue = grouped_issues[0] if grouped_issues else None

        strengths_str = "、".join(d["label"] for d in strong_dims[-3:]) if strong_dims else "暂无明显强项"
        top_weakness_str = weak_dims[0]["label"] + f"（{weak_dims[0]['score_pct']}%）" if weak_dims else (
            dim_summary[0]["label"] + f"（{dim_summary[0]['score_pct']}%）" if dim_summary else "无"
        )
        top_priority_str = top_issue["title"][:40] if top_issue else "继续保持现有水平"

        core_conclusion = {
            "strengths": f"✅ 强项：{strengths_str}，表现稳定",
            "top_weakness": f"❌ 最大短板：{top_weakness_str}，是当前最影响体验的问题",
            "top_priority": f"🔧 最高优先级：{top_priority_str}",
        }

        # ── Persona summary ────────────────────────────────────────────
        persona_summary = []
        analysis_map = {a.persona_label: a for a in per_analyses}
        for dr in sorted(reports, key=lambda r: r.overall_score):
            grade, _ = _overall_grade(dr.overall_score)
            analysis = analysis_map.get(dr.persona_label)
            key_issues: list[str] = []
            if analysis:
                key_issues = analysis.weaknesses[:3]
            elif dr.critical_failures:
                key_issues = dr.critical_failures[:3]
            persona_summary.append({
                "label": dr.persona_label,
                "score": dr.overall_score,
                "grade": grade,
                "hard_failed": dr.hard_constraint_failed,
                "key_issues": key_issues,
                "strengths": analysis.strengths[:2] if analysis else [],
                "suggestions": analysis.suggestions[:2] if analysis else [],
            })

        return {
            "overall_score": round(agg_report.overall_score, 1),
            "overall_grade": overall_grade,
            "deploy_advice": deploy_advice,
            "core_conclusion": core_conclusion,
            "grouped_issues": grouped_issues,
            "dim_summary": dim_summary,
            "persona_summary": persona_summary,
            "total_dialogues": n,
            "hard_failed_count": sum(1 for dr in reports if dr.hard_constraint_failed),
            "critical_issue_count": len([i for i in grouped_issues if i["severity"] == "critical"]),
        }
