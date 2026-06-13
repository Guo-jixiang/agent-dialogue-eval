from __future__ import annotations

import re
from pathlib import Path

from jinja2 import Environment, FileSystemLoader
from loguru import logger

from config.settings import settings
from core.llm_client import LLMClient
from core.models import (
    Constraint,
    ConstraintCategory,
    ConstraintType,
    FlowNode,
    InstructionSpec,
    KnowledgePoint,
    Transition,
    ConditionType,
)


class InstructionParser:
    def __init__(self, llm_client: LLMClient):
        self._llm = llm_client
        self._jinja = Environment(
            loader=FileSystemLoader(str(settings.PROMPTS_DIR)),
            trim_blocks=True,
            lstrip_blocks=True,
        )

    async def parse(self, instruction_text: str) -> InstructionSpec:
        logger.info("Phase 1: LLM structured extraction")
        raw = await self._llm_extract(instruction_text)

        logger.info("Phase 2: deterministic validation and normalization")
        spec = self._validate_and_normalize(raw, instruction_text)

        logger.info("Phase 3: grounding verification")
        spec = self._verify_grounding(spec, instruction_text)

        return spec

    async def _llm_extract(self, instruction_text: str) -> dict:
        template = self._jinja.get_template("instruction_parser.j2")
        prompt = template.render(instruction_text=instruction_text)
        result = await self._llm.chat_json(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=16384,  # instruction parsing produces the largest JSON output
        )
        return result

    def _validate_and_normalize(self, raw: dict, instruction_text: str) -> InstructionSpec:
        # Parse flow nodes
        flow_graph = []
        for i, node_data in enumerate(raw.get("flow_graph", [])):
            transitions = []
            for t in node_data.get("transitions", []):
                raw_ctype = t.get("condition_type", "intent")
                try:
                    ctype = ConditionType(raw_ctype)
                except ValueError:
                    logger.warning(f"Unknown condition_type '{raw_ctype}', defaulting to 'intent'")
                    ctype = ConditionType.intent
                transitions.append(Transition(
                    target_node_id=t.get("target_node_id", ""),
                    condition=t.get("condition", ""),
                    condition_type=ctype,
                ))
            flow_graph.append(
                FlowNode(
                    id=node_data.get("id", f"node_{i}"),
                    description=node_data.get("description", ""),
                    required=node_data.get("required", True),
                    content_keywords=node_data.get("content_keywords", []),
                    transitions=transitions,
                )
            )

        # Fallback mapping for LLM-generated category values not in our enum
        _category_fallback = {
            "content": "tone",
            "style": "tone",
            "language": "tone",
            "word": "forbidden_word",
            "words": "forbidden_word",
            "limit": "length",
            "count": "length",
            "repeat": "repetition",
            "speed": "pacing",
            "pace": "pacing",
            "behavior": "tone",
        }

        # Parse constraints with regex normalization for numeric thresholds
        constraints = []
        for c_data in raw.get("constraints", []):
            # Resolve category first, using fallback map and try/except safety net
            raw_cat = c_data.get("category", "tone")
            category_str = _category_fallback.get(raw_cat, raw_cat)
            try:
                category = ConstraintCategory(category_str)
            except ValueError:
                logger.warning(f"Unknown constraint category '{raw_cat}', defaulting to 'tone'")
                category = ConstraintCategory.tone

            # Extract numeric threshold from description for length constraints
            threshold = c_data.get("threshold")
            if threshold is None:
                nums = re.findall(r"\d+", c_data.get("description", ""))
                if nums and category == ConstraintCategory.length:
                    threshold = float(nums[0])

            raw_type = c_data.get("type", "soft")
            try:
                ctype = ConstraintType(raw_type)
            except ValueError:
                logger.warning(f"Unknown constraint type '{raw_type}', defaulting to 'soft'")
                ctype = ConstraintType.soft

            constraints.append(
                Constraint(
                    id=c_data.get("id", f"c{len(constraints)}"),
                    type=ctype,
                    category=category,
                    description=c_data.get("description", ""),
                    threshold=threshold,
                )
            )

        # Parse knowledge points
        knowledge_points = [
            KnowledgePoint(
                id=k.get("id", f"kp{i}"),
                description=k.get("description", ""),
                correct_content=k.get("correct_content", ""),
                common_errors=k.get("common_errors", []),
            )
            for i, k in enumerate(raw.get("knowledge_points", []))
        ]

        # Validate flow graph connectivity: warn on orphan nodes
        if flow_graph:
            reachable = {flow_graph[0].id}
            for node in flow_graph:
                for t in node.transitions:
                    reachable.add(t.target_node_id)
            for node in flow_graph:
                if node.id not in reachable and node != flow_graph[0]:
                    logger.warning(f"Flow node '{node.id}' may be unreachable")

        return InstructionSpec(
            role=raw.get("role", "外呼专员"),
            objective=raw.get("objective", ""),
            flow_graph=flow_graph,
            constraints=constraints,
            knowledge_points=knowledge_points,
            raw_text=instruction_text,
        )

    @staticmethod
    def _char_ngrams(text: str, n: int = 3) -> set[str]:
        """Extract character n-grams from text, stripping whitespace."""
        cleaned = re.sub(r"\s+", "", text)
        if len(cleaned) < n:
            return {cleaned} if cleaned else set()
        return {cleaned[i : i + n] for i in range(len(cleaned) - n + 1)}

    @staticmethod
    def _grounding_score(item_text: str, source_text: str) -> float:
        """Fraction of item's character 3-grams found in source text."""
        if not item_text or not item_text.strip():
            return 1.0
        item_ng = InstructionParser._char_ngrams(item_text)
        source_ng = InstructionParser._char_ngrams(source_text)
        if not item_ng:
            return 1.0
        overlap = item_ng & source_ng
        return len(overlap) / len(item_ng)

    def _verify_grounding(self, spec: InstructionSpec, raw_text: str) -> InstructionSpec:
        """Hardcoded grounding verification using character n-gram overlap.

        For each extracted item (constraint, flow node, knowledge point),
        checks whether its description text has sufficient character n-gram
        overlap with the source instruction. Items with low overlap are likely
        LLM hallucinations — a warning is logged but the item is NOT removed
        (to avoid false positives deleting real constraints).

        Uses character 3-grams: works for Chinese without any tokenizer dependency.
        """
        GROUNDING_THRESHOLD = 0.4

        # Verify constraints
        for c in spec.constraints:
            score = self._grounding_score(c.description, raw_text)
            if score < GROUNDING_THRESHOLD:
                logger.warning(
                    f"Constraint '{c.id}' may be ungrounded "
                    f"(score={score:.2f}): {c.description[:80]}"
                )

        # Verify flow nodes
        for node in spec.flow_graph:
            score = self._grounding_score(node.description, raw_text)
            if score < GROUNDING_THRESHOLD:
                logger.warning(
                    f"Flow node '{node.id}' may be ungrounded "
                    f"(score={score:.2f}): {node.description[:80]}"
                )

        # Verify knowledge points (check both description and correct_content)
        for kp in spec.knowledge_points:
            desc_score = self._grounding_score(kp.description, raw_text)
            content_score = self._grounding_score(kp.correct_content, raw_text)
            score = min(desc_score, content_score)
            if score < GROUNDING_THRESHOLD:
                logger.warning(
                    f"Knowledge point '{kp.id}' may be ungrounded "
                    f"(score={score:.2f}): {kp.description[:80]}"
                )

        return spec
