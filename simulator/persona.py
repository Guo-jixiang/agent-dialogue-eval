from __future__ import annotations

import itertools

from jinja2 import Environment, FileSystemLoader
from loguru import logger

from config.settings import settings
from core.llm_client import LLMClient
from core.models import (
    PERSONA_DIMENSIONS,
    Persona,
    PersonaArchetype,
    derive_archetype,
)


# ---------------------------------------------------------------------------
# Seed persona definitions (kept for CLI backward compat & as LLM reference)
# ---------------------------------------------------------------------------

PERSONA_DEFINITIONS: dict[PersonaArchetype, Persona] = {
    PersonaArchetype.cooperative: Persona(
        archetype=PersonaArchetype.cooperative,
        behavior_description="积极配合，按预期流程回应，愿意提供信息，不主动刁难",
        response_style="简短友好，肯定性回应为主",
        emotional_state="平静，略带好奇",
        domain_knowledge="普通用户，了解基本情况",
    ),
    PersonaArchetype.resistant: Persona(
        archetype=PersonaArchetype.resistant,
        behavior_description="对来电目的持怀疑或拒绝态度，倾向于拒绝，但不会立即挂断",
        response_style="简短，带拒绝意味，使用'不需要''不想''算了'等词",
        emotional_state="轻微不满或警惕",
        domain_knowledge="了解基本情况，对推销有防备",
    ),
    PersonaArchetype.confused: Persona(
        archetype=PersonaArchetype.confused,
        behavior_description="对对话内容频繁表示不理解，需要多次解释，容易跑偏",
        response_style="提问多，说'什么意思''没听懂''你说什么'",
        emotional_state="困惑，但无明显负面情绪",
        domain_knowledge="对业务知识了解较少",
    ),
    PersonaArchetype.impatient: Persona(
        archetype=PersonaArchetype.impatient,
        behavior_description="明显不耐烦，催促对方说重点，对冗长回复反应强烈",
        response_style="非常简短，使用'说重点''我很忙''快说'",
        emotional_state="急躁，时间压力感强",
        domain_knowledge="了解基本情况，只想快速获取关键信息",
    ),
    PersonaArchetype.off_topic: Persona(
        archetype=PersonaArchetype.off_topic,
        behavior_description="频繁将话题引向与业务无关的方向，提出无关问题",
        response_style="随意，话题发散，有时问天气、问时间等无关问题",
        emotional_state="散漫，不专注",
        domain_knowledge="对业务不感兴趣",
    ),
    PersonaArchetype.silent: Persona(
        archetype=PersonaArchetype.silent,
        behavior_description="回应极少，只说'嗯''哦''知道了'，不主动提供信息",
        response_style="1-3个字的极简回应",
        emotional_state="冷漠或被动",
        domain_knowledge="了解基本情况，但不愿主动表达",
    ),
    PersonaArchetype.detail_seeking: Persona(
        archetype=PersonaArchetype.detail_seeking,
        behavior_description="对业务细节极度关注，反复追问边界情况和例外条款",
        response_style="问题多且具体，追问细节，会说'那如果...''具体是怎么...''有没有例外'",
        emotional_state="理性，有些怀疑，但保持礼貌",
        domain_knowledge="对该领域有一定了解，善于发现漏洞",
    ),
}


# ---------------------------------------------------------------------------
# Dimension → persona trait mapping (deterministic base, LLM enriches later)
# ---------------------------------------------------------------------------

_COOPERATION_TRAITS: dict[str, dict[str, str]] = {
    "cooperative": {
        "behavior": "积极配合，按预期流程回应，愿意提供信息",
        "emotional": "平静友好",
    },
    "neutral": {
        "behavior": "态度不明确，有时配合有时走神，需要对方引导",
        "emotional": "中性，略有心不在焉",
    },
    "resistant": {
        "behavior": "对来电持怀疑或拒绝态度，倾向于推脱或婉拒",
        "emotional": "轻微不满或警惕",
    },
}

_VERBOSITY_TRAITS: dict[str, dict[str, str]] = {
    "verbose": {
        "style": "回复较长，喜欢展开说，偶尔跑题",
        "emotional_boost": "",
    },
    "terse": {
        "style": "极简短回复，每次1-2句，不爱多说",
        "emotional_boost": "",
    },
    "inquisitive": {
        "style": "频繁提问，追问细节和边界情况",
        "emotional_boost": "",
    },
    "perfunctory": {
        "style": "敷衍式回应，用'嗯''哦''知道了'应付",
        "emotional_boost": "",
    },
}

_FAMILIARITY_TRAITS: dict[str, dict[str, str]] = {
    "novice": {
        "knowledge": "第一次接触此类业务，对术语和流程不熟悉",
        "behavior_add": "可能问'这是什么意思'、'我不太懂'",
    },
    "expert": {
        "knowledge": "经历过多次类似场景，对流程和规则很熟悉",
        "behavior_add": "可能主动提及过往经验或与其他家的对比",
    },
    "partial": {
        "knowledge": "了解一些但不全面，容易产生误解或片面判断",
        "behavior_add": "可能基于片面理解提出质疑",
    },
}

_URGENCY_MODIFIERS: dict[str, dict[str, str]] = {
    "relaxed": {
        "emotional_add": "，时间充裕，心态从容",
        "behavior_add": "不着急，有耐心听完",
    },
    "rushed": {
        "emotional_add": "，正在忙或赶时间，耐心有限",
        "behavior_add": "催促对方说重点，想尽快结束",
    },
}


def _build_persona_from_tags(tags: dict[str, str]) -> Persona:
    """Deterministically build a Persona from dimension value tags."""
    coop = _COOPERATION_TRAITS.get(tags.get("cooperation", ""), _COOPERATION_TRAITS["cooperative"])
    verb = _VERBOSITY_TRAITS.get(tags.get("verbosity", ""), _VERBOSITY_TRAITS["terse"])
    fam = _FAMILIARITY_TRAITS.get(tags.get("familiarity", ""), _FAMILIARITY_TRAITS["novice"])
    urg = _URGENCY_MODIFIERS.get(tags.get("urgency", ""), _URGENCY_MODIFIERS["relaxed"])

    behavior = f"{coop['behavior']}，{verb['style']}。{urg['behavior_add']}。{fam['behavior_add']}"
    emotional = f"{coop['emotional']}{urg['emotional_add']}"
    archetype = derive_archetype(tags)

    return Persona(
        archetype=archetype,
        behavior_description=behavior,
        response_style=verb["style"],
        emotional_state=emotional,
        domain_knowledge=fam["knowledge"],
        tags=tags,
        variant_seed=archetype.value,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_persona(archetype: PersonaArchetype) -> Persona:
    return PERSONA_DEFINITIONS[archetype]


def get_all_personas() -> list[Persona]:
    return list(PERSONA_DEFINITIONS.values())


def dimension_combinations(
    selected: dict[str, list[str]] | None = None,
) -> list[dict[str, str]]:
    """Return the cartesian product of selected dimension values.

    Args:
        selected: e.g. {"cooperation": ["cooperative", "resistant"], "verbosity": ["verbose", "terse"]}
                  If None or empty, uses all values for all dimensions.

    Returns:
        List of tag dicts, e.g. [{"cooperation": "cooperative", "verbosity": "verbose", ...}, ...]
    """
    dims = []
    values = []
    for dim_key in ("cooperation", "verbosity", "familiarity", "urgency"):
        all_vals = list(PERSONA_DIMENSIONS[dim_key].keys())
        all_vals.remove("label")
        picked = (selected or {}).get(dim_key, all_vals)
        if not picked:
            picked = all_vals
        dims.append(dim_key)
        values.append([v for v in picked if v in all_vals])

    combos = []
    for combo in itertools.product(*values):
        combos.append(dict(zip(dims, combo)))
    return combos


async def generate_personas(
    selected_dimensions: dict[str, list[str]] | None,
    eval_client: LLMClient | None = None,
) -> list[Persona]:
    """Generate personas from dimension combinations, optionally enriched by LLM.

    This is the main entry point replacing the old diversify_personas + seed flow.
    """
    combos = dimension_combinations(selected_dimensions)
    logger.info(f"Building {len(combos)} personas from dimension combinations")

    # Deterministic base personas
    personas = [_build_persona_from_tags(tags) for tags in combos]

    # LLM enrichment pass — rephrase descriptions for more natural variety
    if eval_client is not None and len(personas) >= 4:
        personas = await _enrich_personas(personas, eval_client)

    return personas


# ---------------------------------------------------------------------------
# LLM enrichment
# ---------------------------------------------------------------------------

async def _enrich_personas(
    personas: list[Persona],
    eval_client: LLMClient,
    batch_size: int = 24,
) -> list[Persona]:
    """Use LLM to enrich personas in batches to avoid output truncation.

    Each batch is sent as a separate LLM call, then results are merged.
    """
    jinja = Environment(
        loader=FileSystemLoader(str(settings.PROMPTS_DIR)),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    template = jinja.get_template("persona_diversify.j2")

    enriched_map: dict[str, Persona] = {}
    total_batches = (len(personas) + batch_size - 1) // batch_size

    try:
        for batch_idx in range(total_batches):
            batch_start = batch_idx * batch_size
            batch = personas[batch_start:batch_start + batch_size]

            compact = []
            for p in batch:
                compact.append({
                    "archetype": p.archetype.value,
                    "tags": p.tags,
                    "label": p.label,
                    "behavior_description": p.behavior_description,
                    "response_style": p.response_style,
                    "emotional_state": p.emotional_state,
                    "domain_knowledge": p.domain_knowledge,
                })

            logger.info(
                f"Enriching personas batch {batch_idx + 1}/{total_batches} "
                f"({len(batch)} personas)"
            )
            result = await eval_client.chat_json(
                messages=[{
                    "role": "user",
                    "content": template.render(personas=compact),
                }],
                temperature=0.85,
                max_tokens=8192,
            )

            items = result if isinstance(result, list) else result.get("personas", [])
            if not items:
                logger.warning(
                    f"Batch {batch_idx + 1}: enrichment returned empty, "
                    f"keeping deterministic personas"
                )
                continue

            for item in items:
                tags = item.get("tags", {})
                key = _tags_key(tags)
                if not key:
                    continue
                enriched_map[key] = Persona(
                    archetype=derive_archetype(tags),
                    behavior_description=item.get("behavior_description", ""),
                    response_style=item.get("response_style", ""),
                    emotional_state=item.get("emotional_state", ""),
                    domain_knowledge=item.get("domain_knowledge", ""),
                    tags=tags,
                    variant_seed=item.get("archetype", ""),
                )

        # Merge: use enriched where available, keep deterministic for rest
        result_personas = []
        for p in personas:
            key = _tags_key(p.tags)
            result_personas.append(enriched_map.get(key, p))

        enriched_count = sum(
            1 for p in result_personas if _tags_key(p.tags) in enriched_map
        )
        logger.info(
            f"LLM enriched {enriched_count}/{len(result_personas)} personas "
            f"across {total_batches} batches"
        )
        return result_personas

    except Exception as e:
        logger.warning(f"Persona enrichment failed (non-fatal), using deterministic: {e}")
        return personas


def _tags_key(tags: dict[str, str]) -> str:
    parts = [tags.get(d, "") for d in ("cooperation", "verbosity", "familiarity", "urgency")]
    return "|".join(parts)


# ---------------------------------------------------------------------------
# Legacy compat — used by CLI when --personas flag is passed
# ---------------------------------------------------------------------------

async def diversify_personas(
    seeds: list[Persona],
    eval_client: LLMClient,
    count: int | None = None,
) -> list[Persona]:
    """Legacy: generate variants from archetype seeds using the old LLM path.

    Redirects to dimension-based generation when called with 7 seeds (all archetypes).
    """
    count = count or settings.PERSONA_VARIANT_COUNT
    if len(seeds) >= 7 and count >= 36:
        logger.info("Legacy diversify → redirecting to dimension-based generation")
        return await generate_personas(selected_dimensions=None, eval_client=eval_client)

    # Small selection — use old LLM diversification logic
    jinja = Environment(
        loader=FileSystemLoader(str(settings.PROMPTS_DIR)),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    template = jinja.get_template("persona_diversify.j2")
    prompt = template.render(seeds=seeds, count=count)

    try:
        logger.info(f"Diversifying {len(seeds)} seed personas into {count} variants")
        result = await eval_client.chat_json(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.9,
            max_tokens=4096,
        )

        variants: list[Persona] = []
        archetype_map = {a.value: a for a in PersonaArchetype}
        for item in result if isinstance(result, list) else result.get("personas", []):
            arch_raw = item.get("archetype", "")
            archetype = archetype_map.get(arch_raw)
            if archetype is None:
                logger.warning(f"Unknown archetype '{arch_raw}' in variant, skipping")
                continue
            variants.append(Persona(
                archetype=archetype,
                behavior_description=item.get("behavior_description", ""),
                response_style=item.get("response_style", ""),
                emotional_state=item.get("emotional_state", ""),
                domain_knowledge=item.get("domain_knowledge", ""),
                tags=item.get("tags", {}),
                variant_seed=arch_raw,
            ))

        if len(variants) >= len(seeds):
            logger.info(f"Generated {len(variants)} persona variants")
            return variants
        else:
            logger.warning(f"Only {len(variants)} valid variants (expected >={len(seeds)}), using seeds")
            return list(seeds)

    except Exception as e:
        logger.warning(f"Persona diversification failed (non-fatal), using seeds: {e}")
        return list(seeds)
