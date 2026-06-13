from __future__ import annotations

from enum import Enum
from typing import Any
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Instruction Spec models
# ---------------------------------------------------------------------------

class ConditionType(str, Enum):
    keyword = "keyword"
    intent = "intent"
    always = "always"


class Transition(BaseModel):
    target_node_id: str
    condition: str
    condition_type: ConditionType = ConditionType.intent


class FlowNode(BaseModel):
    id: str
    description: str
    required: bool = True
    content_keywords: list[str] = Field(default_factory=list)
    transitions: list[Transition] = Field(default_factory=list)


class ConstraintType(str, Enum):
    hard = "hard"
    soft = "soft"


class ConstraintCategory(str, Enum):
    length = "length"
    forbidden_word = "forbidden_word"
    tone = "tone"
    repetition = "repetition"
    pacing = "pacing"


class Constraint(BaseModel):
    id: str
    type: ConstraintType
    category: ConstraintCategory
    description: str
    threshold: float | None = None  # numeric limit when applicable


class KnowledgePoint(BaseModel):
    id: str
    description: str
    correct_content: str
    common_errors: list[str] = Field(default_factory=list)


class InstructionSpec(BaseModel):
    role: str
    user_role: str = ""
    objective: str
    flow_graph: list[FlowNode]
    constraints: list[Constraint]
    knowledge_points: list[KnowledgePoint]
    raw_text: str


# ---------------------------------------------------------------------------
# Simulation models
# ---------------------------------------------------------------------------

class PersonaArchetype(str, Enum):
    cooperative = "cooperative"
    resistant = "resistant"
    confused = "confused"
    impatient = "impatient"
    off_topic = "off_topic"
    silent = "silent"
    detail_seeking = "detail_seeking"


# ---------------------------------------------------------------------------
# Persona dimension system — 4 independent axes, 72 combinations
# ---------------------------------------------------------------------------

PERSONA_DIMENSIONS: dict[str, dict[str, str]] = {
    "cooperation": {
        "label": "配合意愿",
        "cooperative": "积极配合",
        "neutral": "态度中立",
        "resistant": "抵触抗拒",
    },
    "verbosity": {
        "label": "话量风格",
        "verbose": "健谈多话",
        "terse": "寡言少语",
        "inquisitive": "刨根问底",
        "perfunctory": "敷衍应付",
    },
    "familiarity": {
        "label": "业务熟悉度",
        "novice": "完全新手",
        "expert": "资深老手",
        "partial": "一知半解",
    },
    "urgency": {
        "label": "时间紧迫度",
        "relaxed": "从容不迫",
        "rushed": "急迫匆忙",
    },
}


def derive_archetype(tags: dict[str, str]) -> PersonaArchetype:
    """Map dimension combination → closest archetype (derived label, not a multiplier)."""
    c = tags.get("cooperation", "")
    v = tags.get("verbosity", "")
    f = tags.get("familiarity", "")
    u = tags.get("urgency", "")

    if c == "cooperative" and v == "inquisitive" and f in ("expert", "partial"):
        return PersonaArchetype.detail_seeking
    if c == "cooperative":
        return PersonaArchetype.cooperative
    if c == "resistant" and v == "terse" and u == "rushed":
        return PersonaArchetype.impatient
    if c == "resistant":
        return PersonaArchetype.resistant
    if c == "neutral" and v == "inquisitive":
        return PersonaArchetype.confused
    if c == "neutral" and v == "perfunctory":
        return PersonaArchetype.off_topic
    if v == "terse" and c == "neutral":
        return PersonaArchetype.silent
    if v == "terse" and c == "resistant":
        return PersonaArchetype.impatient
    return PersonaArchetype.cooperative


def persona_label(tags: dict[str, str]) -> str:
    """Human-readable label from dimension values, e.g. '积极配合·健谈多话·完全新手·从容不迫'."""
    parts = []
    for dim_key in ("cooperation", "verbosity", "familiarity", "urgency"):
        val = tags.get(dim_key, "")
        label = PERSONA_DIMENSIONS.get(dim_key, {}).get(val, val)
        parts.append(label)
    return "·".join(parts)


class Persona(BaseModel):
    archetype: PersonaArchetype
    behavior_description: str
    response_style: str
    emotional_state: str
    domain_knowledge: str
    tags: dict[str, str] = Field(default_factory=dict)
    variant_seed: str | None = None
    test_dimension: str = ""  # targeted eval dimension key, empty for comprehensive
    label_override: str = ""  # custom label from persona generator, takes precedence

    @property
    def label(self) -> str:
        if self.label_override:
            return self.label_override
        return persona_label(self.tags) if self.tags else self.archetype.value


class DialogueTurn(BaseModel):
    turn_id: int
    role: str  # "agent" | "user"
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def char_count(self) -> int:
        return len(self.content)


class TerminationReason(str, Enum):
    natural_end = "natural_end"
    max_turns = "max_turns"
    user_hangup = "user_hangup"


class DialogueSource(str, Enum):
    """Origin of a dialogue — simulated internally or provided externally."""
    simulated = "simulated"
    external = "external"
    production = "production"


class SimulatedDialogue(BaseModel):
    id: str
    persona: Persona
    instruction_spec: InstructionSpec
    turns: list[DialogueTurn]
    termination_reason: TerminationReason


class Dialogue(BaseModel):
    """A conversation between an agent and a user.

    Superset of ``SimulatedDialogue`` that can also represent
    externally-provided dialogues (where persona / instruction_spec /
    termination_reason may not be available).

    ``SimulatedDialogue`` is kept as a backward-compatible alias —
    existing code that constructs ``SimulatedDialogue`` directly
    continues to work, and code that reads the fields will see the
    same values.
    """
    id: str
    source: DialogueSource = DialogueSource.simulated
    persona: Persona | None = None
    instruction_spec: InstructionSpec | None = None
    turns: list[DialogueTurn]
    termination_reason: TerminationReason | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Evaluation models
# ---------------------------------------------------------------------------

class EvaluationType(str, Enum):
    binary = "binary"
    score_1_5 = "score_1_5"
    percentage = "percentage"


class EvaluationDimension(str, Enum):
    flow = "flow"                    # 流程遵循 — 任务完成度
    constraint = "constraint"        # 约束合规 — 硬性规则遵守
    knowledge = "knowledge"          # 知识准确 — 信息正确性
    style = "style"                  # 风格语气 — 自然度与得体性
    coherence = "coherence"          # 连贯性 — 上下文逻辑一致
    safety = "safety"                # 安全合规 — 无有害/越界内容
    adaptability = "adaptability"    # 应变能力 — 异常处理与错误恢复


class Checkpoint(BaseModel):
    id: str
    description: str
    dimension: EvaluationDimension
    evaluation_type: EvaluationType
    weight: float = 1.0
    source_constraint_id: str | None = None
    source_flow_node_id: str | None = None
    source_knowledge_id: str | None = None


class Evidence(BaseModel):
    turn_ids: list[int]
    quoted_text: str
    reasoning: str
    confidence: float = 1.0


class CheckpointResult(BaseModel):
    checkpoint: Checkpoint
    passed: bool
    score: float  # 0-1 normalized
    evidence: Evidence | None = None
    explanation: str = ""
    confidence: float = 1.0
    validation_warnings: list[str] = Field(default_factory=list)
    num_runs: int = 1
    run_scores: list[float] = Field(default_factory=list)


class DimensionScore(BaseModel):
    dimension: EvaluationDimension
    score: float  # 0-1
    weighted_score: float
    checkpoint_results: list[CheckpointResult]
    summary: str = ""
    anomaly_flags: list[str] = Field(default_factory=list)


class EvaluationReport(BaseModel):
    id: str
    dialogue_id: str
    persona_archetype: PersonaArchetype
    persona_label: str = ""
    dimension_tags: dict[str, str] = Field(default_factory=dict)
    overall_score: float  # 0-100
    hard_constraint_failed: bool
    dimension_scores: list[DimensionScore]
    critical_failures: list[str]
    strengths: list[str]
    weaknesses: list[str]
    anomaly_flags: list[str] = Field(default_factory=list)


class AggregatedReport(BaseModel):
    id: str
    instruction_raw: str
    instruction_spec: InstructionSpec
    dialogue_reports: list[EvaluationReport]
    overall_score: float
    score_by_persona: dict[str, float]
    score_by_dimension: dict[str, float]
    score_by_dimension_value: dict[str, dict[str, float]] = Field(default_factory=dict)
    critical_failures: list[str]
    strengths: list[str]
    weaknesses: list[str]
    improvement_suggestions: list[str]
    # Layer 4/5/6 analysis results (populated after evaluation)
    per_dialogue_analyses: list[dict] = Field(default_factory=list)
    judge_analysis: dict | None = None
    attribution: dict | None = None
    # Structured 3-layer summary (built by ReportAgent.build_structured_summary)
    structured_summary: dict | None = None


# ── Layer 4: Per-dialogue analysis ──────────────────────────────────────

class PerDialogueAnalysis(BaseModel):
    dialogue_id: str
    persona_label: str
    overall_score: float
    success: bool
    strengths: list[str]
    weaknesses: list[str]
    suggestions: list[str]


# ── Layer 5: Judge cross-comparison ─────────────────────────────────────

class RankedDialogue(BaseModel):
    rank: int
    dialogue_id: str
    persona_label: str
    score: float
    key_reason: str = ""

class CommonIssue(BaseModel):
    issue: str
    count: int
    pct: float
    examples: list[str] = Field(default_factory=list)

class DimensionStat(BaseModel):
    avg: float
    min_val: float
    max_val: float
    below_60_count: int = 0

class JudgeAnalysis(BaseModel):
    top5: list[RankedDialogue] = Field(default_factory=list)
    bottom5: list[RankedDialogue] = Field(default_factory=list)
    common_issues: list[CommonIssue] = Field(default_factory=list)
    dimension_summary: dict[str, DimensionStat] = Field(default_factory=dict)


# ── Layer 6: Root cause attribution ─────────────────────────────────────

class RootCause(BaseModel):
    category: str  # 知识库缺失 | Prompt问题 | 工具失效 | 流程缺陷 | 约束过严 | 其他
    description: str
    affected_checkpoints: list[str] = Field(default_factory=list)
    affected_personas: list[str] = Field(default_factory=list)
    fix_suggestion: str = ""
    confidence: float = 0.8

class AttributionReport(BaseModel):
    root_causes: list[RootCause] = Field(default_factory=list)
    summary: str = ""
