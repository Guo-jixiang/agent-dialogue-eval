"""Layer 5: Judge Agent — cross-comparison of all dialogue analyses."""

from __future__ import annotations

from jinja2 import Environment, FileSystemLoader

from config.settings import settings
from core.llm_client import LLMClient, RoundRobinLLMClient
from core.models import (
    CommonIssue, DimensionStat, JudgeAnalysis, PerDialogueAnalysis, RankedDialogue,
)


class JudgeAgent:
    """Global cross-comparison: top/bottom, common issues, dimension stats."""

    def __init__(self, llm_client: LLMClient | RoundRobinLLMClient):
        self._llm = llm_client
        self._jinja = Environment(
            loader=FileSystemLoader(str(settings.PROMPTS_DIR)),
            trim_blocks=True, lstrip_blocks=True,
        )

    async def analyze(
        self,
        analyses: list[PerDialogueAnalysis],
        dimension_scores: dict[str, float],
    ) -> JudgeAnalysis:
        template = self._jinja.get_template("judge_agent.j2")

        dialogues = [
            {
                "persona_label": a.persona_label,
                "overall_score": a.overall_score,
                "success": a.success,
                "strengths": a.strengths,
                "weaknesses": a.weaknesses,
            }
            for a in analyses
        ]

        prompt = template.render(
            total=len(analyses),
            dialogues=dialogues,
            dimension_scores=dimension_scores,
        )

        raw = await self._llm.chat_json(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )

        return JudgeAnalysis(
            top5=[RankedDialogue(**r) for r in raw.get("top5", [])],
            bottom5=[RankedDialogue(**r) for r in raw.get("bottom5", [])],
            common_issues=[CommonIssue(**i) for i in raw.get("common_issues", [])],
            dimension_summary={
                k: DimensionStat(**v) if isinstance(v, dict) else DimensionStat(avg=v, min_val=v, max_val=v)
                for k, v in raw.get("dimension_summary", {}).items()
            },
        )
