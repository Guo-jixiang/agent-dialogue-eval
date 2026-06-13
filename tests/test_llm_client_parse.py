"""Tests for JSON parse resilience: _repair_truncated_json and the 4-stage chain."""
from __future__ import annotations

import json

import pytest

from core.llm_client import _repair_truncated_json, _extract_json, _find_json_object


# ---------------------------------------------------------------------------
# Sample data
# ---------------------------------------------------------------------------

VALID_RESULTS = {
    "results": [
        {
            "checkpoint_id": "cp_001",
            "passed": True,
            "score": 1.0,
            "turn_ids": [1],
            "quoted_text": "您好，请问是机构负责人吗？",
            "reasoning": "专员在第1轮明确询问，用户在第2轮确认。",
            "confidence": 1.0,
        },
        {
            "checkpoint_id": "cp_002",
            "passed": True,
            "score": 0.8,
            "turn_ids": [2, 3],
            "quoted_text": "您知道后台已走低延迟线路吗？",
            "reasoning": "用户确认后，专员继续询问。",
            "confidence": 1.0,
        },
    ]
}


def _valid_json_str() -> str:
    return json.dumps(VALID_RESULTS, ensure_ascii=False)


def _truncated_at(results_json: str, char_count: int) -> str:
    """Simulate truncation at a specific character position."""
    return results_json[:char_count]


# ---------------------------------------------------------------------------
# Existing stages — regression
# ---------------------------------------------------------------------------

class TestExistingStages:
    def test_extract_json_strips_markdown_fences(self):
        raw = '```json\n{"a": 1}\n```'
        result = _extract_json(raw)
        assert result == '{"a": 1}'
        assert json.loads(result) == {"a": 1}

    def test_extract_json_handles_surrounding_text(self):
        raw = 'Here is your JSON: {"key": "value"} Hope this helps!'
        result = _extract_json(raw)
        assert json.loads(result) == {"key": "value"}

    def test_find_json_object_finds_first_valid(self):
        raw = 'extra {"a": 1} more {"b": 2}'
        result = _find_json_object(raw)
        assert result == {"a": 1}


# ---------------------------------------------------------------------------
# Stage 4: truncated JSON repair
# ---------------------------------------------------------------------------

class TestRepairTruncatedJson:

    # ── Unclosed object brace ──────────────────────────────────────────

    def test_missing_closing_brace(self):
        """Truncated at mid-object — missing final }."""
        full = _valid_json_str()
        # Cut off right after the last quoted_text value
        pos = full.rfind('"您知道后台已走低延迟线路吗？"') + len('"您知道后台已走低延迟线路吗？"')
        truncated = full[:pos]  # ends with ...线路吗？"  → no closing braces
        result = _repair_truncated_json(truncated)
        assert result is not None
        assert len(result["results"]) == 2

    def test_missing_closing_brace_and_bracket(self):
        """Truncated inside results array — missing ] and }."""
        full = _valid_json_str()
        # Cut after the first checkpoint object
        first_end = full.find('},') + 1  # after first checkpoint's }
        truncated = full[:first_end]  # ends with }  → missing ] and outer }
        result = _repair_truncated_json(truncated)
        assert result is not None
        assert len(result["results"]) == 1
        assert result["results"][0]["checkpoint_id"] == "cp_001"

    def test_mid_key_truncation(self):
        """Truncated in the middle of a key name — rare edge case.

        Mid-key truncation is extremely unlikely in practice (LLMs don't cut
        off mid-token like this). The repair correctly returns None rather than
        producing a corrupt parse. The common case — truncation after complete
        values with missing closing brackets — is the important one.
        """
        raw = '{"results": [{"checkpoint_id": "cp_001", "passed": true, "score": 1.0, "turn_'
        result = _repair_truncated_json(raw)
        # Mid-key without value produces invalid JSON even after bracket
        # completion. Returning None is the safe choice.
        assert result is None

    def test_mid_string_truncation(self):
        """Truncated inside a string value."""
        raw = '{"results": [{"checkpoint_id": "cp_001", "quoted_text": "您好，请问是'
        result = _repair_truncated_json(raw)
        assert result is not None

    # ── Valid JSON passes through ──────────────────────────────────────

    def test_valid_json_parses_correctly(self):
        raw = _valid_json_str()
        result = _repair_truncated_json(raw)
        assert result == VALID_RESULTS

    def test_valid_simple_object(self):
        result = _repair_truncated_json('{"a": 1, "b": [2, 3]}')
        assert result == {"a": 1, "b": [2, 3]}

    # ── Edge cases ─────────────────────────────────────────────────────

    def test_nested_objects_all_closed(self):
        raw = '{"outer": {"inner": {"deep": "value"}}}'
        result = _repair_truncated_json(raw)
        assert result == {"outer": {"inner": {"deep": "value"}}}

    def test_nested_objects_unclosed(self):
        raw = '{"outer": {"inner": {"deep": "value"'
        result = _repair_truncated_json(raw)
        assert result is not None
        assert result["outer"]["inner"]["deep"] == "value"

    def test_array_of_strings_unclosed(self):
        raw = '{"items": ["a", "b", "c"'
        result = _repair_truncated_json(raw)
        assert result == {"items": ["a", "b", "c"]}

    def test_no_json_found(self):
        result = _repair_truncated_json("just some text, no JSON here")
        assert result is None

    def test_empty_string(self):
        result = _repair_truncated_json("")
        assert result is None

    def test_markdown_fence_truncated(self):
        raw = '```json\n{"results": [{"id": "cp_001", "passed": true}'
        result = _repair_truncated_json(raw)
        assert result is not None
        assert result["results"][0]["id"] == "cp_001"

    # ── The exact failure from the user's error ────────────────────────

    def test_exact_user_error_pattern(self):
        """The exact truncated JSON from the user's error message."""
        raw = (
            '{ "results": [ '
            '{ "checkpoint_id": "cp_001", "passed": true, "score": 1.0, '
            '"turn_ids": [1], "quoted_text": "您好，请问是机构负责人吗？", '
            '"reasoning": "专员在第1轮明确询问是否是机构负责人，用户在第2轮确认，身份确认完成。", '
            '"confidence": 1.0, "status": "applicable" }, '
            '{ "checkpoint_id": "cp_002", "passed": true, "score": 1.0, '
            '"turn_ids": [2, 3], "quoted_text": "您知道后台已走低延迟线路吗？", '
            '"reasoning": "用户确认后，专员继续询问知情情况，跳转正确。", '
            '"confidence": 1.0,'
        )
        result = _repair_truncated_json(raw)
        assert result is not None
        assert len(result["results"]) == 2
        assert result["results"][0]["checkpoint_id"] == "cp_001"
        assert result["results"][1]["checkpoint_id"] == "cp_002"
        # Second object should have confidence repaired
        assert result["results"][1]["confidence"] == 1.0


# ---------------------------------------------------------------------------
# Full 4-stage chain integration
# ---------------------------------------------------------------------------

def _full_parse_chain(raw: str) -> dict | None:
    """Replicate the 4-stage chain from chat_json()."""
    # Stage 1
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    # Stage 2
    stripped = _extract_json(raw)
    if stripped:
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            pass
    # Stage 3
    obj = _find_json_object(raw)
    if obj is not None:
        return obj
    # Stage 4
    return _repair_truncated_json(raw)


class TestFullParseChain:
    def test_valid_json_parsed_at_stage_1(self):
        result = _full_parse_chain('{"status": "ok"}')
        assert result == {"status": "ok"}

    def test_markdown_fence_parsed_at_stage_2(self):
        result = _full_parse_chain('```json\n{"a": 1}\n```')
        assert result == {"a": 1}

    def test_surrounding_text_parsed_at_stage_2_or_3(self):
        result = _full_parse_chain('The answer is {"key": 42}. Done.')
        assert result == {"key": 42}

    def test_truncated_repaired_at_stage_4(self):
        result = _full_parse_chain('{"items": [1, 2, 3')
        assert result == {"items": [1, 2, 3]}

    def test_user_error_repaired_at_stage_4(self):
        """The exact truncated JSON from the user's error should be repaired."""
        raw = (
            '{ "results": [ '
            '{ "checkpoint_id": "cp_001", "passed": true, "score": 1.0, '
            '"turn_ids": [1], "quoted_text": "您好，请问是机构负责人吗？", '
            '"reasoning": "专员在第1轮明确询问，用户在第2轮确认。", '
            '"confidence": 1.0, "status": "applicable" }, '
            '{ "checkpoint_id": "cp_002", "passed": true, "score": 1.0, '
            '"turn_ids": [2, 3], "quoted_text": "您知道后台已走低延迟线路吗？", '
            '"reasoning": "用户确认后，专员继续询问。", '
            '"confidence": 1.0,'
        )
        result = _full_parse_chain(raw)
        assert result is not None
        assert len(result["results"]) == 2
