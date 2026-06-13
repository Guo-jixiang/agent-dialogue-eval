from __future__ import annotations

import asyncio

from jinja2 import Environment, FileSystemLoader
from loguru import logger

from config.settings import settings
from core.agent import BaseAgent
from core.models import InstructionSpec, PersonaArchetype, Persona, SimulatedDialogue
from core.llm_client import LLMClient
from simulator.engine import DialogueEngine
from simulator.persona import generate_personas, get_all_personas, get_persona
from simulator.runner import ConversationRunner
from simulator.user_simulator import UserSimulator

_sem: asyncio.Semaphore | None = None


def _get_sem() -> asyncio.Semaphore:
    global _sem
    if _sem is None:
        _sem = asyncio.Semaphore(settings.SIM_CONCURRENCY)
    return _sem


async def _resolve_personas(
    dimensions: dict[str, list[str]] | None,
    archetypes: list[PersonaArchetype] | None,
    eval_client: LLMClient | None,
) -> list[Persona]:
    """Resolve persona list from dimensions, archetypes, or defaults."""
    if dimensions is not None:
        personas = await generate_personas(dimensions, eval_client)
        logger.info(f"Generated {len(personas)} personas from dimension combinations")
        return personas

    if archetypes:
        seeds = [get_persona(a) for a in archetypes]
        if eval_client is not None and len(seeds) >= 3:
            from simulator.persona import diversify_personas
            personas = await diversify_personas(seeds, eval_client)
            logger.info(f"Diversified {len(seeds)} seeds -> {len(personas)} variants")
            return personas
        return list(seeds)

    # Default: all personas
    seeds = get_all_personas()
    if eval_client is not None:
        personas = await generate_personas(None, eval_client)
        logger.info(f"Generated {len(personas)} personas from all dimensions")
        return personas
    return list(seeds)


async def run_scenario_matrix(
    spec: InstructionSpec,
    agent_client: LLMClient,
    simulator_client: LLMClient,
    archetypes: list[PersonaArchetype] | None = None,
    dimensions: dict[str, list[str]] | None = None,
    max_turns: int = 30,
    eval_client: LLMClient | None = None,
    agent: BaseAgent | None = None,
    personas: list[Persona] | None = None,
) -> list[SimulatedDialogue]:
    """Run persona x instruction simulation matrix, concurrently.

    Precedence:
    1. If *personas* is provided, use them directly (batch mode).
    2. If *dimensions* is provided, generate personas from dimension combinations.
    3. If *archetypes* is provided (legacy CLI path), use those seeds.
    4. Otherwise use all 7 seeds.

    If *agent* is provided it overrides *agent_client* — use this to
    inject a ``FridayAgent`` or any custom ``BaseAgent`` implementation.
    """
    if personas is None:
        personas = await _resolve_personas(dimensions, archetypes, eval_client)
    sem = _get_sem()

    if agent is not None:
        # ── New decoupled path: ConversationRunner + UserSimulator ──
        jinja_env = Environment(
            loader=FileSystemLoader(str(settings.PROMPTS_DIR)),
            trim_blocks=True,
            lstrip_blocks=True,
        )
        instruction_context = {
            "role": spec.role,
            "user_role": spec.user_role,
            "objective": spec.objective,
        }

        async def _run_one(persona: Persona) -> SimulatedDialogue:
            async with sem:
                user_sim = UserSimulator(
                    llm_client=simulator_client,
                    persona=persona,
                    instruction_context=instruction_context,
                    jinja_env=jinja_env,
                )
                runner = ConversationRunner(
                    agent=agent,
                    user_simulator=user_sim,
                    spec=spec,
                    max_turns=max_turns,
                    eval_client=eval_client,
                )
                return await runner.run()

        dialogues = await asyncio.gather(*(_run_one(p) for p in personas))
        return list(dialogues)

    # ── Legacy path (backward compatible) ──
    engine = DialogueEngine(
        agent_client=agent_client,
        simulator_client=simulator_client,
        max_turns=max_turns,
        eval_client=eval_client,
    )

    async def _run_one(persona: Persona) -> SimulatedDialogue:
        async with sem:
            return await engine.run(spec, persona)

    dialogues = await asyncio.gather(*(_run_one(p) for p in personas))
    return list(dialogues)
