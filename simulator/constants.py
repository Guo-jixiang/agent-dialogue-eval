"""Shared constants used by the simulator subsystem."""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Semantic coverage check scheduling
# ---------------------------------------------------------------------------
SEMANTIC_CHECK_FIRST = 3    # first semantic check after this many agent replies
SEMANTIC_CHECK_INTERVAL = 4  # re-check every N agent replies thereafter

# ---------------------------------------------------------------------------
# Progressive end hints injected into simulator prompt
# ---------------------------------------------------------------------------
HINT_PREFIXES: dict[int, str] = {
    1: "[对话已接近尾声，你可以开始考虑自然收尾]",
    2: "[现在是合适结束对话的时机，你可以表现出准备挂断的态度]",
    3: "[对话已完成核心内容，请尽快自然结束本次通话]",
}
