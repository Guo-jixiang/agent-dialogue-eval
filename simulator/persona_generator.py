"""Persona generation agent — creates targeted test personas per evaluation dimension."""

from __future__ import annotations

from jinja2 import Environment, FileSystemLoader

from config.settings import settings
from core.llm_client import LLMClient
from core.models import InstructionSpec, Persona, PersonaArchetype

_DIM_META: dict[str, dict[str, str]] = {
    "flow":         {"label": "流程遵循", "desc": "是否按指令规定的对话流程节点推进，完成必须步骤"},
    "constraint":   {"label": "约束合规", "desc": "是否严格遵守硬性约束（禁用词/长度上限等）和软性约束"},
    "knowledge":    {"label": "知识准确", "desc": "提供的业务信息是否与知识点一致，有无编造或错误"},
    "style":        {"label": "风格语气", "desc": "表达风格是否与角色定义一致，措辞是否得体"},
    "coherence":    {"label": "连贯性",   "desc": "对话上下文是否逻辑一致，有无前后矛盾"},
    "safety":       {"label": "安全合规", "desc": "是否泄露信息、产生有害内容或编造不存在的政策"},
    "adaptability": {"label": "应变能力", "desc": "面对用户异常行为时的应对质量"},
}



class PersonaGenerator:
    """Generate test personas dynamically based on evaluation dimensions."""

    def __init__(self, llm_client: LLMClient):
        self._llm = llm_client
        self._jinja = Environment(
            loader=FileSystemLoader(str(settings.PROMPTS_DIR)),
            trim_blocks=True,
            lstrip_blocks=True,
        )

    async def generate(
        self,
        spec: InstructionSpec,
        dims: list[dict],  # [{"key": "flow", "count": 5 or 0}, ...]
        mode: str = "multi",
        total: int = 7,
    ) -> list[Persona]:
        """Generate personas, splitting into parallel chunks when count >= 10."""
        template = self._jinja.get_template("persona_generator.j2")

        selected = []
        for d in dims:
            key = d["key"]
            if key not in _DIM_META:
                raise ValueError(
                    f"Unknown dimension key: {key}. "
                    f"Valid keys: {list(_DIM_META.keys())}"
                )
            selected.append({
                "key": key,
                "label": _DIM_META[key]["label"],
                "desc": _DIM_META[key]["desc"],
                "count": d.get("count", 0),
            })

        if mode == "single":
            count = dims[0].get("count", 5) if dims else 5
        else:
            count = total

        # Split large batches into parallel chunks (max 8 per chunk)
        import asyncio
        if count > 8:
            chunks = []
            remaining = count
            while remaining > 0:
                chunk_size = min(remaining, 8)
                chunks.append(chunk_size)
                remaining -= chunk_size

            async def _one_chunk(sz: int) -> list[dict]:
                prompt = template.render(
                    spec=spec, selected_dims=selected, count=sz, mode=mode,
                )
                raw = await self._llm.chat_json(
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.8,
                )
                return raw.get("personas", [])

            chunk_results = await asyncio.gather(*(
                _one_chunk(sz) for sz in chunks
            ))
            raw_personas = [p for chunk in chunk_results for p in chunk]
        else:
            prompt = template.render(
                spec=spec, selected_dims=selected, count=count, mode=mode,
            )
            raw = await self._llm.chat_json(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.8,
            )
            raw_personas = raw.get("personas", [])

        personas: list[Persona] = []
        for item in raw_personas:
            tags = item.get("tags", {})
            personas.append(Persona(
                archetype=PersonaArchetype.cooperative,
                label_override=item.get("label", ""),
                behavior_description=item.get("behavior_description", ""),
                response_style=item.get("response_style", ""),
                emotional_state=item.get("emotional_state", ""),
                domain_knowledge=item.get("domain_knowledge", ""),
                tags={
                    "cooperation": tags.get("cooperation", "cooperative"),
                    "verbosity": tags.get("verbosity", "verbose"),
                    "familiarity": tags.get("familiarity", "novice"),
                    "urgency": tags.get("urgency", "relaxed"),
                },
                test_dimension=item.get("test_dimension", ""),
            ))

        return personas
