"""Standalone user-persona simulator.

``UserSimulator`` is self-contained: it owns its system prompt, manages
its own message history, and can be used independently of any agent or
conversation runner.
"""

from __future__ import annotations

from core.llm_client import LLMClient
from core.models import Persona
from simulator.constants import HINT_PREFIXES

from jinja2 import Environment


class UserSimulator:
    """Self-contained simulated user driven by an LLM.

    Usage::

        sim = UserSimulator(llm_client, persona, instruction_context, jinja_env)
        reply = await sim.respond("外呼专员: 您好，我是...")
        reply = await sim.respond("外呼专员: 请问您考虑得怎么样？", end_hint_level=1)
    """

    def __init__(
        self,
        llm_client: LLMClient,
        persona: Persona,
        instruction_context: dict[str, str],
        jinja_env: Environment,
    ) -> None:
        """
        Parameters
        ----------
        llm_client:
            The LLM backend used to generate user replies.
        persona:
            The user persona to embody.
        instruction_context:
            Dict with keys ``"role"``, ``"user_role"``, ``"objective"``
            (extracted from ``InstructionSpec``).
        jinja_env:
            Jinja2 environment that can load ``user_simulator_system.j2``.
        """
        self._llm = llm_client
        self._persona = persona
        self._jinja = jinja_env
        self._system_prompt = self._build_system_prompt(instruction_context)
        self._history: list[dict[str, str]] = [
            {"role": "system", "content": self._system_prompt}
        ]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def respond(self, agent_message: str, end_hint_level: int = 0) -> tuple[str, bool]:
        """Feed the agent's latest message and return the simulated user's reply.

        Returns
        -------
        tuple[str, bool]
            The reply text and whether the user wants to end the conversation.
            ``True`` when the reply contains ``[END]`` (stripped from text).
        """
        prefix = HINT_PREFIXES.get(end_hint_level, "")
        if prefix:
            formatted = f"{prefix}\n外呼专员: {agent_message}"
        else:
            formatted = f"外呼专员: {agent_message}"

        self._history.append({"role": "user", "content": formatted})
        raw = await self._llm.chat(messages=self._history, temperature=0.85)
        self._history.append({"role": "assistant", "content": raw})

        should_end = "[END]" in raw
        reply = raw.replace("[END]", "").strip()
        return reply, should_end

    async def kickoff(self) -> str:
        """Let the simulator 'start' — used when the agent sends the opening
        and we need the first user reply.  Equivalent to calling
        ``respond`` with an empty agent message that carries only the
        hint prefix.

        This method exists for logical clarity: the very first call
        after the agent's opening is just ``respond(agent_opening)``.
        """
        return await self.respond("")

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def persona(self) -> Persona:
        return self._persona

    @property
    def history(self) -> list[dict[str, str]]:
        """Return a shallow copy of the internal message history."""
        return list(self._history)

    # ------------------------------------------------------------------
    # System prompt
    # ------------------------------------------------------------------

    def _build_system_prompt(self, ctx: dict[str, str]) -> str:
        """Render ``user_simulator_system.j2`` with persona + instruction context.

        Extracted from ``DialogueEngine._build_sim_system()``.
        """
        template = self._jinja.get_template("user_simulator_system.j2")
        return template.render(
            persona=self._persona,
            instruction_context=ctx,
        )
