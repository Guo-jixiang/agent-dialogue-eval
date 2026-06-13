from __future__ import annotations

import asyncio
import itertools
import json
import re
from typing import Any

import httpx
from openai import AsyncOpenAI
from loguru import logger

from config.settings import settings
from core.rate_limit import get_rate_limiter, get_concurrency_sem


class LLMClient:
    def __init__(self, api_key: str, base_url: str, model: str, max_retries: int = 5):
        self.model = model
        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=300.0,
            max_retries=max_retries,
            http_client=httpx.AsyncClient(
                limits=httpx.Limits(
                    max_connections=1000,
                    max_keepalive_connections=400,
                ),
                timeout=httpx.Timeout(30.0, connect=10.0),
            ),
        )

    async def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2048,
        response_format: dict[str, str] | None = None,
    ) -> str:
        # Acquire rate-limit token before hitting the API
        limiter = get_rate_limiter(
            rate=settings.LLM_RATE_LIMIT_RPS,
            burst=settings.LLM_RATE_LIMIT_BURST,
        )
        if limiter is not None:
            await limiter.acquire()

        sem = get_concurrency_sem(settings.LLM_MAX_CONCURRENCY)
        if sem is not None:
            async with sem:
                return await self._do_chat(messages, temperature, max_tokens, response_format)
        return await self._do_chat(messages, temperature, max_tokens, response_format)

    async def _do_chat(
        self,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
        response_format: dict[str, str] | None,
    ) -> str:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format:
            kwargs["response_format"] = response_format

        logger.debug(f"LLM call: model={self.model} messages={len(messages)}")

        # Retry connection errors with exponential backoff.
        # The OpenAI SDK's max_retries only covers HTTP 5xx/429 —
        # TCP-level connection drops need our own retry loop.
        last_exc: Exception | None = None
        for attempt in range(settings.LLM_MAX_RETRIES):
            try:
                response = await self._client.chat.completions.create(**kwargs)
                content = response.choices[0].message.content or ""
                logger.debug(f"LLM response: {len(content)} chars")
                return content
            except Exception as e:
                last_exc = e
                # Only retry connection/transport errors, not auth/billing errors
                err_str = str(e).lower()
                is_retryable = (
                    isinstance(e, (httpx.ConnectError, httpx.RemoteProtocolError,
                                   httpx.ReadTimeout, httpx.WriteTimeout,
                                   httpx.PoolTimeout, httpx.ConnectTimeout))
                    or "connection" in err_str
                    or "timeout" in err_str
                    or "reset" in err_str
                )
                if not is_retryable or attempt >= settings.LLM_MAX_RETRIES - 1:
                    raise
                delay = min(0.5 * (2 ** attempt), 8)
                logger.warning(
                    f"LLM connection error (attempt {attempt + 1}/{settings.LLM_MAX_RETRIES}), "
                    f"retrying in {delay}s: {e}"
                )
                await asyncio.sleep(delay)

        raise last_exc  # type: ignore[misc]

    async def chat_json(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        if max_tokens is None:
            max_tokens = settings.LLM_JSON_MAX_TOKENS

        raw = await self._try_chat_json(messages, temperature, max_tokens)
        result = _parse_json(raw)
        if result is not None:
            return result

        # Retry once — the model sometimes returns empty or garbled output
        logger.warning("JSON parse failed on first attempt, retrying…")
        raw = await self._try_chat_json(messages, temperature, max_tokens)
        result = _parse_json(raw)
        if result is not None:
            return result

        raise ValueError(f"Failed to parse JSON from LLM response: {raw[:500]}")

    async def _try_chat_json(
        self,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> str:
        return await self.chat(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
        )


def _parse_json(raw: str) -> dict[str, Any] | None:
    """5-stage JSON parse chain. Returns parsed dict or None."""
    # Stage 1: direct parse
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Stage 2: strip markdown fences and surrounding text
    stripped = _extract_json(raw)
    if stripped:
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            pass

    # Stage 3: brace-matching walk to find a valid JSON object
    obj = _find_json_object(raw)
    if obj is not None:
        return obj

    # Stage 4: auto-complete truncated JSON
    repaired = _repair_truncated_json(raw)
    if repaired is not None:
        logger.warning("JSON was truncated — repaired via bracket auto-completion")
        return repaired

    # Stage 5: json_repair library
    try:
        from json_repair import repair_json
        return json.loads(repair_json(raw))
    except Exception:
        pass

    return None


def _extract_json(text: str) -> str | None:
    """Strip markdown fences, code blocks, and surrounding text to isolate JSON."""
    text = text.strip()
    # Remove ```json ... ``` blocks
    text = text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    # Find the first { and last }
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        return text[start : end + 1]
    return None


def _find_json_object(text: str) -> dict | None:
    """Find the first valid JSON object by brace matching."""
    depth = 0
    start = -1
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start >= 0:
                candidate = text[start : i + 1]
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    start = -1
    return None


def _repair_truncated_json(text: str) -> dict | None:
    """Auto-complete truncated JSON by closing unclosed brackets and strings.

    Handles the common LLM failure mode where ``max_tokens`` cuts off the
    response mid-JSON, leaving dangling ``{`` / ``[`` or unterminated strings.

    Uses a stack to track bracket nesting so that closes are emitted in the
    correct LIFO order (objects before arrays, etc.).

    Returns the parsed ``dict`` on success, or ``None`` if repair fails.
    """
    # Strip markdown fences and leading/trailing whitespace
    cleaned = text.removeprefix("```json").removeprefix("```").strip()

    # Find the start of the JSON object
    start = cleaned.find("{")
    if start < 0:
        return None
    body = cleaned[start:]

    # Walk the string, tracking the stack of unclosed brackets and string state
    stack: list[str] = []  # tracks '{' and '[' openers
    in_string = False
    escape_next = False

    for ch in body:
        if escape_next:
            escape_next = False
            continue
        if ch == "\\":
            escape_next = True
            continue
        if ch == '"' and not escape_next:
            in_string = not in_string
        elif not in_string:
            if ch == "{":
                stack.append("{")
            elif ch == "}":
                if stack and stack[-1] == "{":
                    stack.pop()
            elif ch == "[":
                stack.append("[")
            elif ch == "]":
                if stack and stack[-1] == "[":
                    stack.pop()

    # Remove trailing comma (common in truncated arrays/objects)
    body = body.rstrip()
    if body.endswith(","):
        body = body[:-1]

    # Close unclosed string
    if in_string:
        body += '"'

    # Close brackets in LIFO order
    for opener in reversed(stack):
        body += "}" if opener == "{" else "]"

    try:
        return json.loads(body)
    except json.JSONDecodeError:
        return None


# ── Multi-key round-robin client ──────────────────────────────────────────


class RoundRobinLLMClient:
    """Distributes ``chat()`` / ``chat_json()`` calls across multiple
    ``LLMClient`` instances in round-robin order, spreading rate-limit
    pressure across independent API keys.

    Thread-safe for concurrent asyncio usage via an internal lock.
    """

    def __init__(self, clients: list[LLMClient]) -> None:
        if not clients:
            raise ValueError("RoundRobinLLMClient requires at least 1 LLMClient")
        self._clients = clients
        self._cycle = itertools.cycle(range(len(clients)))
        self._lock = asyncio.Lock()

    async def _next(self) -> LLMClient:
        async with self._lock:
            return self._clients[next(self._cycle)]

    async def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2048,
        response_format: dict[str, str] | None = None,
    ) -> str:
        return await (await self._next()).chat(
            messages, temperature, max_tokens, response_format
        )

    async def chat_json(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        return await (await self._next()).chat_json(
            messages, temperature, max_tokens
        )


def make_eval_client(
    api_key: str = "",
    base_url: str = "",
    model: str = "",
    max_retries: int | None = None,
) -> LLMClient | RoundRobinLLMClient:
    """Create the eval LLM client(s).

    If ``EVAL_LLM_API_KEY_2`` and ``EVAL_LLM_API_KEY_3`` are both set in
    settings, returns a ``RoundRobinLLMClient`` that distributes calls
    across all three keys.  Otherwise returns a single ``LLMClient``.

    Parameters
    ----------
    api_key:
        Override for the primary key (empty = use ``settings.EVAL_LLM_API_KEY``).
    base_url / model:
        Overrides forwarded to every underlying ``LLMClient``.
    """
    key1 = api_key or settings.EVAL_LLM_API_KEY
    key2 = settings.EVAL_LLM_API_KEY_2
    key3 = settings.EVAL_LLM_API_KEY_3
    url = base_url or settings.EVAL_LLM_BASE_URL
    mdl = model or settings.EVAL_LLM_MODEL
    retries = max_retries if max_retries is not None else settings.LLM_MAX_RETRIES

    if key2 and key3:
        all_keys = [key1, key2, key3]
        clients = [
            LLMClient(k if k else key1, url, mdl, max_retries=retries)
            for k in all_keys
        ]
        logger.info(
            "RoundRobinLLMClient: distributing across {} eval API keys",
            len(clients),
        )
        return RoundRobinLLMClient(clients)

    return LLMClient(key1, url, mdl, max_retries=retries)
