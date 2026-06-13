from __future__ import annotations

from difflib import SequenceMatcher

from config.settings import settings
from core.models import Checkpoint, CheckpointResult, Evidence, SimulatedDialogue


def extract_turn_context(dialogue: SimulatedDialogue, turn_ids: list[int], context_window: int = 1) -> str:
    """Return dialogue excerpt around specified turn IDs with optional context."""
    turn_map = {t.turn_id: t for t in dialogue.turns}
    all_ids = sorted(set(turn_ids))

    expanded: set[int] = set()
    for tid in all_ids:
        for offset in range(-context_window, context_window + 1):
            expanded.add(tid + offset)

    lines = []
    for turn in dialogue.turns:
        if turn.turn_id in expanded:
            role_label = "外呼专员" if turn.role == "agent" else "用户"
            marker = "→ " if turn.turn_id in set(all_ids) else "  "
            lines.append(f"{marker}[第{turn.turn_id}轮] {role_label}: {turn.content}")

    return "\n".join(lines)


def summarize_failures(results: list[CheckpointResult]) -> list[str]:
    """Return human-readable failure summaries for critical failures."""
    failures = []
    for r in results:
        if not r.passed and r.score == 0.0:
            detail = r.evidence.quoted_text[:60] if r.evidence else ""
            failures.append(f"【{r.checkpoint.description}】{detail}")
    return failures


def validate_evidence(
    evidence: Evidence,
    dialogue: SimulatedDialogue,
    checkpoint: Checkpoint,
) -> list[str]:
    """Validate turn_ids and quoted_text against actual dialogue. Returns warnings."""
    warnings = []
    valid_turn_ids = {t.turn_id for t in dialogue.turns}

    for tid in evidence.turn_ids:
        if tid not in valid_turn_ids:
            warnings.append(f"turn_id {tid} 不存在于对话中")

    if evidence.quoted_text and not _fuzzy_contains_dialogue(dialogue, evidence.quoted_text):
        # Skip validation warning for descriptive/analytical text that is clearly
        # not intended as a direct quote (e.g., arbitration summaries, "not applicable" notes).
        # Heuristics: no turn_ids referenced, or text looks like analysis rather than dialogue.
        if not _is_descriptive_text(evidence):
            warnings.append("quoted_text 未在对话原文中找到匹配")

    return warnings


def _is_descriptive_text(evidence: Evidence) -> bool:
    """Detect if quoted_text is descriptive/analytical rather than a direct dialogue quote.

    Common patterns from MetaJudge arbitration or "not_applicable" results:
    - Empty turn_ids with non-empty text (no specific turn referenced)
    - Text contains analytical markers like "未", "全程", "从未", "直接跳转" etc.
    - Text starts with formatting like "[第X轮]" (multi-turn composite, handled separately)
    """
    text = evidence.quoted_text.strip()
    if not text:
        return True

    # If no turn_ids are referenced, the text is likely a summary/observation
    if not evidence.turn_ids:
        return True

    # Analytical/descriptive markers commonly produced by MetaJudge
    descriptive_markers = [
        "全程", "从未", "未提及", "未涉及", "未触发", "不适用",
        "直接跳转", "完全没有", "未执行", "未检测到",
    ]
    for marker in descriptive_markers:
        if marker in text:
            return True

    return False


def _fuzzy_contains_dialogue(dialogue: SimulatedDialogue, quoted: str, threshold: float = 0.6) -> bool:
    """Check if quoted text approximately appears in any dialogue turn."""
    if not quoted.strip():
        return True

    # Handle multi-turn composite quotes like "[第7轮] 外呼专员: ..."
    # Strip turn markers and check each segment
    import re
    segments = re.split(r'\[第\d+轮\]\s*(?:外呼专员|用户)\s*[:：]\s*', quoted)
    segments = [s.strip() for s in segments if s.strip()]

    # If we extracted segments from formatted quotes, validate each segment
    if segments and len(segments) != len(quoted.strip().splitlines()):
        for segment in segments:
            if _single_segment_matches(dialogue, segment, threshold):
                continue
            # If any segment doesn't match, fall through to full-text check
            break
        else:
            return True  # All segments matched

    # Single-text matching
    return _single_segment_matches(dialogue, quoted, threshold)


def _single_segment_matches(dialogue: SimulatedDialogue, text: str, threshold: float = 0.6) -> bool:
    """Check if a single text segment matches any dialogue turn."""
    if not text.strip():
        return True

    for turn in dialogue.turns:
        if text in turn.content:
            return True
        ratio = SequenceMatcher(None, text, turn.content).ratio()
        if ratio > threshold:
            return True
        # Sliding window: check if any 15-char substring of text appears in turn
        window = min(15, len(text))
        if window < 4:
            continue
        for i in range(len(text) - window + 1):
            snippet = text[i : i + window]
            if snippet in turn.content:
                return True
    return False


def build_validated_result(
    raw_item: dict,
    checkpoint: Checkpoint,
    dialogue: SimulatedDialogue,
    default_score: float = 0.0,
    default_passed: bool = False,
) -> CheckpointResult:
    """Build CheckpointResult from raw LLM output with evidence validation."""
    evidence = Evidence(
        turn_ids=raw_item.get("turn_ids", []),
        quoted_text=raw_item.get("quoted_text", ""),
        reasoning=raw_item.get("reasoning", ""),
        confidence=float(raw_item.get("confidence") or 1.0),
    )
    warnings = validate_evidence(evidence, dialogue, checkpoint) if settings.EVIDENCE_VALIDATION_ENABLED else []
    return CheckpointResult(
        checkpoint=checkpoint,
        passed=bool(raw_item.get("passed", default_passed)),
        score=float(raw_item.get("score") or default_score),
        evidence=evidence,
        explanation=raw_item.get("reasoning", ""),
        validation_warnings=warnings,
    )
