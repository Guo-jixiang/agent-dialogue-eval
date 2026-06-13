"""Agent abstraction layer — pluggable digital-human backends.

Provides:
- BaseAgent: abstract interface for any agent backend
- LLMAgent: self-simulated agent via local LLMClient (OpenAI SDK)
- FridayAgent: external LLM call via raw HTTP API (generic REST format)
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import httpx
from loguru import logger

from core.llm_client import LLMClient
from core.models import InstructionSpec


class BaseAgent(ABC):
    """Abstract agent interface.

    Implementations can be LLM-simulated or wrappers around external
    digital-human APIs (e.g. Friday).  The runner only calls
    ``reply(messages)`` — it does not know or care about the backend.
    """

    @abstractmethod
    async def reply(self, messages: list[dict[str, str]]) -> str:
        """Return the agent's next utterance given the conversation *messages*.

        Parameters
        ----------
        messages:
            Accumulated conversation history in OpenAI-style format
            ``[{"role": "...", "content": "..."}, ...]``, where the
            *first* message is the agent's opening and subsequent
            messages alternate between ``"user"`` (the real user or
            simulated user) and ``"assistant"`` (the agent).

        Returns
        -------
        str
            The agent's reply (raw text).
        """
        ...

    @property
    @abstractmethod
    def label(self) -> str:
        """Human-readable label for logging and reports.

        Examples: ``"llm-gpt-4o"``, ``"friday-v2"``.
        """
        ...


class LLMAgent(BaseAgent):
    """Self-simulated agent backed by an LLM (current default behaviour).

    This is the agent that was previously embedded directly inside
    ``DialogueEngine``.  It owns its system prompt (derived from the
    *spec*) and delegates message generation to an ``LLMClient``.
    """

    def __init__(self, llm_client: LLMClient, spec: InstructionSpec) -> None:
        self._llm = llm_client
        self._spec = spec
        self._system_prompt = self._build_system_prompt(spec)

    # ------------------------------------------------------------------
    # BaseAgent interface
    # ------------------------------------------------------------------

    @property
    def label(self) -> str:
        return f"llm-{self._llm.model}"

    async def reply(self, messages: list[dict[str, str]]) -> str:
        full_messages = [{"role": "system", "content": self._system_prompt}] + messages
        return await self._llm.chat(messages=full_messages, temperature=0.7)

    # ------------------------------------------------------------------
    # System prompt (extracted from DialogueEngine._build_agent_system)
    # ------------------------------------------------------------------

    @staticmethod
    def _build_system_prompt(spec: InstructionSpec) -> str:
        flow_text = "\n".join(f"- {node.description}" for node in spec.flow_graph)
        constraint_text = "\n".join(
            f"- [{c.type.value.upper()}] {c.description}" for c in spec.constraints
        )
        knowledge_text = "\n".join(
            f"- {k.description}: {k.correct_content}" for k in spec.knowledge_points
        )
        return (
            f"你是{spec.role}。\n\n"
            f"本次外呼目标：{spec.objective}\n\n"
            f"对话流程：\n{flow_text}\n\n"
            f"约束要求：\n{constraint_text}\n\n"
            f"知识点：\n{knowledge_text}\n\n"
            "重要：当所有核心步骤完成后，如果用户已给出明确确认，请自然引导结束对话"
            "（例如'好的，那就不打扰您了'），不要主动开启新话题或强行挽留。"
        )


def resolve_agent(
    agent_type: str,
    *,
    spec: InstructionSpec,
    llm_client: LLMClient | None = None,
    friday_api_url: str = "",
    friday_api_key: str = "",
    friday_model: str = "",
) -> BaseAgent:
    """Create the appropriate agent with automatic fallback.

    If *agent_type* is ``"friday"`` but the Friday API URL is not
    configured, falls back to ``LLMAgent`` (requires *llm_client*).

    Returns
    -------
    BaseAgent
        ``FridayAgent`` or ``LLMAgent``.
    """
    if agent_type == "friday" and friday_api_url:
        logger.info(f"Using FridayAgent: url={friday_api_url}, model={friday_model or '(none)'}")
        return FridayAgent(
            api_url=friday_api_url,
            api_key=friday_api_key,
            spec=spec,
            model=friday_model,
        )

    if agent_type == "friday":
        logger.warning(
            "Friday API requested but FRIDAY_API_URL is not configured. "
            "Falling back to LLMAgent."
        )

    if llm_client is None:
        raise ValueError(
            "LLMAgent requires an llm_client, but none was provided. "
            "Either configure FRIDAY_API_URL or provide an LLM client."
        )

    logger.info(f"Using LLMAgent: model={llm_client.model}")
    return LLMAgent(llm_client, spec)


class FridayAgent(BaseAgent):
    """Agent backed by an external digital-human API (generic REST format).

    **Does NOT build or send a system prompt** — the external service
    (Friday) manages its own digital-human behaviour and prompt
    configuration.  We only forward the raw conversation messages plus
    the structured ``instruction_spec`` so the downstream service has
    the task context it needs.

    Usage::

        agent = FridayAgent(
            api_url="https://your-api.example.com/chat",
            api_key="sk-...",
            spec=spec,
            model="your-model",      # optional — sent as "model" field
        )

    **Request format** (sent to *api_url*)::

        POST {api_url}
        Authorization: Bearer {api_key}
        Content-Type: application/json

        {
            "model": "...",            // only if *model* is configured
            "messages": [              // raw conversation, no system prompt
                {"role": "user", "content": "请开始外呼通话。"},
                {"role": "assistant", "content": "您好，我是..."},
                {"role": "user", "content": "什么事？"},
                ...
            ],
            "temperature": 0.7,
            "instruction_spec": { ... }  // structured spec (role, objective, etc.)
        }

    **Response parsing** — tries these paths in order:

    1. ``{"reply": "..."}``  (simple key)
    2. ``{"choices": [{"message": {"content": "..."}}]}``  (OpenAI-compatible)
    3. ``{"content": "..."}``  (direct content field)
    4. Raw text body (plain string response)
    """

    def __init__(
        self,
        api_url: str,
        api_key: str,
        spec: InstructionSpec,
        model: str = "",
        timeout: float = 60.0,
    ) -> None:
        self._url = api_url
        self._key = api_key
        self._spec = spec
        self._model = model
        self._timeout = timeout

    # ------------------------------------------------------------------
    # BaseAgent interface
    # ------------------------------------------------------------------

    @property
    def label(self) -> str:
        if self._model:
            return f"friday-{self._model}"
        return "friday"

    async def reply(self, messages: list[dict[str, str]]) -> str:
        """Call the external API and return the agent's reply.

        Sends raw conversation messages **without** a system prompt —
        the external service manages its own prompt configuration.
        """
        body: dict = {
            "messages": messages,
            "temperature": 0.7,
        }
        if self._model:
            body["model"] = self._model

        headers = {
            "Authorization": f"Bearer {self._key}",
            "Content-Type": "application/json",
        }

        logger.debug(f"[{self.label}] POST {self._url}  ({len(messages)} messages)")

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.post(self._url, json=body, headers=headers)
            resp.raise_for_status()

        return self._parse_response(resp)

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_response(resp: httpx.Response) -> str:
        """Extract the agent's reply from the HTTP response.

        Tries several common formats (best-effort).
        """
        content_type = resp.headers.get("content-type", "")

        # Plain text response — use as-is
        if "text/plain" in content_type or not content_type.startswith("application/json"):
            text = resp.text.strip()
            if text:
                logger.debug(f"FridayAgent: received plain text reply ({len(text)} chars)")
                return text
            raise ValueError("FridayAgent: empty response body")

        # JSON response — try known shapes
        data = resp.json()

        # 1. Simple key: {"reply": "..."}
        if isinstance(data, dict) and "reply" in data:
            return str(data["reply"])

        # 2. OpenAI-compatible: {"choices": [{"message": {"content": "..."}}]}
        if isinstance(data, dict) and "choices" in data:
            choices = data["choices"]
            if isinstance(choices, list) and len(choices) > 0:
                msg = choices[0].get("message", {})
                if isinstance(msg, dict) and "content" in msg:
                    return str(msg["content"])

        # 3. Direct content field: {"content": "..."} or {"text": "..."}
        if isinstance(data, dict):
            if "content" in data:
                return str(data["content"])
            if "text" in data:
                return str(data["text"])

        # 4. The whole response is a plain string
        if isinstance(data, str):
            return data

        # 5. Fallback — stringify the whole JSON (last resort)
        logger.warning(
            f"FridayAgent: unrecognised response shape, keys={list(data.keys()) if isinstance(data, dict) else type(data).__name__}. "
            "Returning raw JSON as fallback."
        )
        return str(data)
