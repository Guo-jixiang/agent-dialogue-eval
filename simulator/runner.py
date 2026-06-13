"""Conversation runner — orchestrates a dialogue between any agent and user simulator.

This is the decoupled replacement for the monolithic ``DialogueEngine``.
It injects a ``BaseAgent`` + ``UserSimulator`` via constructor and runs
the conversation loop, handling coverage tracking and termination.
It does **not** know whether the agent is LLM-simulated or an external API.
"""

from __future__ import annotations

import re
import uuid

from loguru import logger

from config.settings import settings
from core.agent import BaseAgent
from core.llm_client import LLMClient
from core.models import (
    DialogueTurn,
    FlowNode,
    InstructionSpec,
    SimulatedDialogue,
    TerminationReason,
)
from simulator.constants import (
    HINT_PREFIXES,
    SEMANTIC_CHECK_FIRST,
    SEMANTIC_CHECK_INTERVAL,
)
from simulator.user_simulator import UserSimulator

# ── Agent farewell detection ─────────────────────────────────────────────
# Only checked on *agent* replies (never user), so false positives are rare.
_AGENT_FAREWELL = re.compile(
    r"(再见|拜拜|通话结束|感谢您的?配?合|祝您|不打扰|先这样|今天就到这"
    r"|那就不打扰|有问题随时|保持联系|下次再聊|回头再联系)"
)


class ConversationRunner:
    """Orchestrate a conversation between a ``BaseAgent`` and a ``UserSimulator``.

    Termination (in priority order):
    1. **User end**: user simulator outputs ``[END]`` → ``user_hangup``
    2. **Agent farewell**: agent says a goodbye phrase → ``natural_end``
    3. **Max turns**: exceeded ``max_turns`` → ``max_turns``
    """

    def __init__(
        self,
        agent: BaseAgent,
        user_simulator: UserSimulator,
        spec: InstructionSpec,
        max_turns: int = settings.MAX_TURNS,
        eval_client: LLMClient | None = None,
    ) -> None:
        self._agent = agent
        self._user = user_simulator
        self._spec = spec
        self._max_turns = max_turns
        self._eval_client = eval_client

    # ------------------------------------------------------------------
    async def run(self) -> SimulatedDialogue:
        dialogue_id = str(uuid.uuid4())[:8]
        persona = self._user.persona
        logger.info(f"[{dialogue_id}] Starting dialogue: persona={persona.archetype}")

        turns: list[DialogueTurn] = []
        termination = TerminationReason.max_turns

        # --- Coverage tracking ---
        required_nodes = [n for n in self._spec.flow_graph if n.required]
        covered_required: set[str] = set()
        round_all_covered: int | None = None

        # --- Agent message history (no system prompt injected) ---
        agent_messages: list[dict[str, str]] = []

        # --- Opening: agent sends first ---
        opening = await self._agent.reply(
            agent_messages + [{"role": "user", "content": "请开始外呼通话。"}]
        )
        turns.append(DialogueTurn(
            turn_id=1, role="agent", content=opening,
            metadata={"char_count": len(opening)},
        ))
        agent_messages.append({"role": "assistant", "content": opening})
        self._update_coverage(opening, covered_required)
        if self._all_required_covered(covered_required, required_nodes):
            round_all_covered = 1

        # --- Alternating turns ---
        for turn_num in range(2, self._max_turns * 2 + 2, 2):
            end_hint = self._compute_hint_level(
                covered_required, required_nodes, round_all_covered, turn_num,
            )

            # ---- User turn ----
            user_reply, user_should_end = await self._user.respond(
                agent_message=opening if turn_num == 2 else agent_reply,
                end_hint_level=end_hint,
            )
            turns.append(DialogueTurn(
                turn_id=turn_num, role="user", content=user_reply,
                metadata={"char_count": len(user_reply)},
            ))

            # Layer 1: user explicitly ends
            if user_should_end:
                termination = TerminationReason.user_hangup
                logger.info(f"[{dialogue_id}] User end at turn {turn_num}")
                break

            # ---- Agent turn ----
            agent_messages.append({"role": "user", "content": user_reply})
            agent_reply = await self._agent.reply(agent_messages)
            turns.append(DialogueTurn(
                turn_id=turn_num + 1, role="agent", content=agent_reply,
                metadata={"char_count": len(agent_reply)},
            ))
            agent_messages.append({"role": "assistant", "content": agent_reply})

            # Update coverage
            self._update_coverage(agent_reply, covered_required)

            # Periodic semantic coverage check
            agent_reply_count = turn_num // 2
            if (
                self._eval_client is not None
                and agent_reply_count >= SEMANTIC_CHECK_FIRST
                and (agent_reply_count - SEMANTIC_CHECK_FIRST) % SEMANTIC_CHECK_INTERVAL == 0
                and not self._all_required_covered(covered_required, required_nodes)
            ):
                await self._semantic_coverage_check(turns, covered_required)

            if self._all_required_covered(covered_required, required_nodes) and round_all_covered is None:
                round_all_covered = turn_num + 1

            # Layer 2: agent says farewell → natural end
            if _AGENT_FAREWELL.search(agent_reply):
                if self._all_required_covered(covered_required, required_nodes) or len(turns) >= 6:
                    termination = TerminationReason.natural_end
                    logger.info(
                        f"[{dialogue_id}] Agent farewell at turn {turn_num + 1} "
                        f"({len(covered_required)}/{len(required_nodes)} required covered)"
                    )
                    break

        logger.info(
            f"[{dialogue_id}] Done: {termination.value}, {len(turns)} turns, "
            f"covered {len(covered_required)}/{len(required_nodes)} required nodes"
        )

        return SimulatedDialogue(
            id=f"dialogue_{dialogue_id}",
            persona=persona,
            instruction_spec=self._spec,
            turns=turns,
            termination_reason=termination,
        )

    # ------------------------------------------------------------------
    # Coverage tracking
    # ------------------------------------------------------------------

    def _update_coverage(self, reply: str, covered: set[str]) -> None:
        for node in self._spec.flow_graph:
            if node.id in covered:
                continue
            for kw in node.content_keywords:
                if kw and kw in reply:
                    covered.add(node.id)
                    break

    @staticmethod
    def _all_required_covered(covered: set[str], required: list[FlowNode]) -> bool:
        return all(n.id in covered for n in required)

    # ------------------------------------------------------------------
    # End hints (nudge user toward natural ending)
    # ------------------------------------------------------------------

    def _compute_hint_level(
        self,
        covered: set[str],
        required: list[FlowNode],
        round_all_covered: int | None,
        turn: int,
    ) -> int:
        total = len(required)
        covered_count = len(covered)
        if round_all_covered is not None:
            elapsed = max((turn - round_all_covered) // 2, 0)
            if elapsed >= 6:
                return 3
            if elapsed >= 4:
                return 2
            if elapsed >= 2:
                return 1
        if covered_count >= total - 1 and turn >= 12:
            return 1
        return 0

    # ------------------------------------------------------------------
    # Semantic coverage
    # ------------------------------------------------------------------

    async def _semantic_coverage_check(
        self, turns: list[DialogueTurn], covered: set[str],
    ) -> None:
        from jinja2 import Environment, FileSystemLoader

        try:
            jinja = Environment(
                loader=FileSystemLoader(str(settings.PROMPTS_DIR)),
                trim_blocks=True, lstrip_blocks=True,
            )
            template = jinja.get_template("semantic_coverage.j2")
            dialogue_text = "\n".join(
                f"[{t.turn_id}] {'Agent' if t.role == 'agent' else '用户'}: {t.content}"
                for t in turns
            )
            required = [n for n in self._spec.flow_graph if n.required]
            node_descs = "\n".join(
                f"  - {n.id}: {n.description}" for n in required if n.id not in covered
            )
            prompt = template.render(
                dialogue_text=dialogue_text, uncovered_nodes=node_descs,
            )
            raw = await self._eval_client.chat_json(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
            )
            for item in raw.get("covered", []):
                node_id = item.get("node_id", "")
                if node_id:
                    covered.add(node_id)
            logger.debug(f"Semantic check: added {list(raw.get('covered', []))}, "
                         f"now {len(covered)}/{len(required)} required")
        except Exception as e:
            logger.warning(f"Semantic coverage check failed: {e}")


