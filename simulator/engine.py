"""Backward-compatible DialogueEngine — thin wrapper around the decoupled classes.

.. deprecated::
    Prefer using ``ConversationRunner`` + ``LLMAgent`` + ``UserSimulator``
    directly for new code.  This class is kept for backward compatibility
    and delegates all work to the new abstractions.
"""

from __future__ import annotations

import warnings

from jinja2 import Environment, FileSystemLoader
from loguru import logger

from config.settings import settings
from core.agent import LLMAgent
from core.llm_client import LLMClient
from core.models import (
    InstructionSpec,
    Persona,
    SimulatedDialogue,
)
from simulator.runner import ConversationRunner
from simulator.user_simulator import UserSimulator


class DialogueEngine:
    """Backward-compatible wrapper.

    Behaves identically to the pre-decoupling ``DialogueEngine`` but
    internally creates an ``LLMAgent``, ``UserSimulator``, and
    ``ConversationRunner``, then delegates ``run()`` to the runner.
    """

    def __init__(
        self,
        agent_client: LLMClient,
        simulator_client: LLMClient,
        max_turns: int = settings.MAX_TURNS,
        eval_client: LLMClient | None = None,
    ):
        self._agent_client = agent_client
        self._sim_client = simulator_client
        self._max_turns = max_turns
        self._eval_client = eval_client
        self._jinja = Environment(
            loader=FileSystemLoader(str(settings.PROMPTS_DIR)),
            trim_blocks=True,
            lstrip_blocks=True,
        )

    # ==================================================================
    # Main loop (delegates to runner)
    # ==================================================================

    async def run(self, spec: InstructionSpec, persona: Persona) -> SimulatedDialogue:
        """Run one dialogue end-to-end.

        Internally creates an ``LLMAgent``, ``UserSimulator``, and
        ``ConversationRunner``, then delegates.
        """
        instruction_context = {
            "role": spec.role,
            "user_role": spec.user_role,
            "objective": spec.objective,
        }

        agent = LLMAgent(self._agent_client, spec)
        user_sim = UserSimulator(
            llm_client=self._sim_client,
            persona=persona,
            instruction_context=instruction_context,
            jinja_env=self._jinja,
        )
        runner = ConversationRunner(
            agent=agent,
            user_simulator=user_sim,
            spec=spec,
            max_turns=self._max_turns,
            eval_client=self._eval_client,
        )
        return await runner.run()

