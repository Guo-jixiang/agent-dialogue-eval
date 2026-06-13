"""Layer 6: Attribution Agent — root cause analysis of low-score cases."""

from __future__ import annotations

from jinja2 import Environment, FileSystemLoader

from config.settings import settings
from core.llm_client import LLMClient, RoundRobinLLMClient
from core.models import AttributionReport, PerDialogueAnalysis, RootCause


class AttributionAgent:
    """Analyze low-score cases to identify root causes."""

    def __init__(self, llm_client: LLMClient | RoundRobinLLMClient):
        self._llm = llm_client
        self._jinja = Environment(
            loader=FileSystemLoader(str(settings.PROMPTS_DIR)),
            trim_blocks=True, lstrip_blocks=True,
        )

    async def analyze(
        self,
        analyses: list[PerDialogueAnalysis],
        failed_checkpoints: dict[str, list[str]],
    ) -> AttributionReport:
        """Identify root causes from bottom-performing dialogues.

        Parameters
        ----------
        analyses:
            Bottom N per-dialogue analyses (already filtered by caller).
        failed_checkpoints:
            Map of persona_label → list of failed checkpoint descriptions.
        """
        template = self._jinja.get_template("attribution_agent.j2")

        low_score_dialogues = [
            {
                "persona_label": a.persona_label,
                "overall_score": a.overall_score,
                "weaknesses": a.weaknesses,
                "failed_checkpoints": failed_checkpoints.get(a.persona_label, []),
            }
            for a in analyses
        ]

        prompt = template.render(
            total=len(low_score_dialogues),
            low_score_dialogues=low_score_dialogues,
        )

        raw = await self._llm.chat_json(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )

        return AttributionReport(
            root_causes=[RootCause(**rc) for rc in raw.get("root_causes", [])],
            summary=raw.get("summary", ""),
        )
