"""Pydantic v2 data models for agentic evaluation agent configuration."""

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field, field_validator, model_validator


class CheckConfig(BaseModel):
    """Objective, test-like check under one dimension."""

    id: str = Field(..., min_length=1, description="Unique check identifier")
    name: str = Field(..., min_length=1, description="Human-readable check name")
    instruction: str = Field(..., min_length=1, description="How to test this check")
    weight: float = Field(default=1.0, gt=0, description="Relative weight for this check")


class SubcriterionConfig(BaseModel):
    """Subjective subcriterion under one dimension."""

    id: str = Field(..., min_length=1, description="Unique subcriterion identifier")
    name: str = Field(..., min_length=1, description="Human-readable subcriterion name")
    instruction: str = Field(..., min_length=1, description="How to assess this subcriterion")
    weight: float = Field(default=1.0, gt=0, description="Relative weight for this subcriterion")


class DimensionScoringConfig(BaseModel):
    """How one dimension should be scored."""

    method: str = Field(
        default="verdict_only",
        description="Scoring method: verdict_only | test_based | rubric_based | hybrid",
    )
    objective_weight: float = Field(
        default=1.0, ge=0, le=1, description="Weight of objective checks in final score"
    )
    subjective_weight: float = Field(
        default=0.0, ge=0, le=1, description="Weight of subjective subcriteria in final score"
    )
    checks: List[CheckConfig] = Field(
        default_factory=list, description="Objective checks for pass/fail style scoring"
    )
    subcriteria: List[SubcriterionConfig] = Field(
        default_factory=list, description="Subjective subcriteria for rubric-based scoring"
    )
    rating_to_score: Dict[str, float] = Field(
        default_factory=lambda: {"good": 100.0, "ok": 70.0, "poor": 40.0},
        description="Score mapping used by subjective ratings",
    )

    @model_validator(mode="after")
    def validate_consistency(self) -> "DimensionScoringConfig":
        if self.method not in {"verdict_only", "test_based", "rubric_based", "hybrid"}:
            raise ValueError(f"Unknown scoring method: {self.method}")
        if self.objective_weight + self.subjective_weight <= 0:
            raise ValueError("objective_weight + subjective_weight must be > 0")

        if self.method == "test_based" and not self.checks:
            raise ValueError("test_based scoring requires non-empty checks")
        if self.method == "rubric_based" and not self.subcriteria:
            raise ValueError("rubric_based scoring requires non-empty subcriteria")
        if self.method == "hybrid" and (not self.checks or not self.subcriteria):
            raise ValueError("hybrid scoring requires both checks and subcriteria")
        return self


class DimensionConfig(BaseModel):
    """A single rubric dimension that an agent evaluates."""

    id: str = Field(..., min_length=1, description="Unique dimension identifier")
    name: str = Field(..., min_length=1, description="Human-readable dimension name")
    instruction: str = Field(
        ..., min_length=1, description="Evaluation instruction for this dimension"
    )
    weight: float = Field(
        default=1.0,
        description=(
            "Weight of this dimension in agent score. Must be > 0, or the sentinel "
            "-1 meaning 'auto-share': split (1 - sum of explicit weights) evenly with "
            "any other -1 dimensions and the query-specific task pool."
        ),
    )
    scoring: Optional[DimensionScoringConfig] = Field(
        default=None, description="Optional scoring strategy for this dimension"
    )

    @field_validator("weight")
    @classmethod
    def _validate_weight(cls, v: float) -> float:
        if v == -1 or v > 0:
            return v
        raise ValueError(
            f"DimensionConfig.weight must be > 0 or the sentinel -1, got {v}"
        )


class AgentScoringConfig(BaseModel):
    """Agent-level scoring configuration."""

    max_score: float = Field(
        default=20.0,
        ge=0,
        description=(
            "Maximum total score for this agent. Set to 0 for agents that gate the "
            "pipeline but should not contribute to the rubric total (e.g. build_engineer)."
        ),
    )
    task_aggregation_mode: Literal["strict", "cumulative"] = Field(
        default="strict",
        description=(
            "How subtask results are aggregated into a main-task score. "
            "'strict' → any failed subtask makes main task score = 0. "
            "'cumulative' → pass-ratio of subtasks."
        ),
    )


class RubricConfig(BaseModel):
    """Rubric configuration containing scoring mode and dimensions."""

    scoring_mode: str = Field(
        default="pass_fail_na",
        description="Scoring mode for this rubric",
    )
    dimensions: List[DimensionConfig] = Field(
        ..., min_length=1, description="List of evaluation dimensions"
    )

    @model_validator(mode="after")
    def validate_unique_dimension_ids(self) -> "RubricConfig":
        ids = [d.id for d in self.dimensions]
        if len(ids) != len(set(ids)):
            seen = set()
            dupes = []
            for dim_id in ids:
                if dim_id in seen:
                    dupes.append(dim_id)
                seen.add(dim_id)
            raise ValueError(f"Duplicate dimension IDs: {dupes}")
        return self


class RuntimeConfig(BaseModel):
    """Runtime limits for an agent."""

    max_steps: int = Field(
        default=10, ge=1, le=100, description="Maximum agent iterations"
    )
    max_step_seconds: float = Field(
        default=60.0, gt=0, description="Max seconds per step"
    )
    max_total_seconds: float = Field(
        default=300.0, gt=0, description="Max total seconds for agent run"
    )

    @model_validator(mode="after")
    def validate_total_gte_step(self) -> "RuntimeConfig":
        if self.max_total_seconds < self.max_step_seconds:
            raise ValueError(
                f"max_total_seconds ({self.max_total_seconds}) must be >= "
                f"max_step_seconds ({self.max_step_seconds})"
            )
        return self


class OutputConfig(BaseModel):
    """Output configuration for agent verdicts."""

    require_evidence: bool = Field(
        default=False, description="Whether evidence is required for each dimension"
    )
    allow_not_applicable: bool = Field(
        default=True, description="Whether not_applicable verdict is allowed"
    )


class AgentConfig(BaseModel):
    """Complete configuration for a persona-driven evaluation agent."""

    version: str = Field(default="1.0.0", description="Config schema version")
    id: str = Field(..., min_length=1, description="Unique agent identifier")
    name: str = Field(..., min_length=1, description="Human-readable agent name")
    description: str = Field(default="", description="Agent description")
    system_prompt: str = Field(
        ..., min_length=1, description="Persona system prompt for the agent"
    )
    rubric: RubricConfig = Field(..., description="Evaluation rubric")
    allowed_tools: List[str] = Field(
        ..., min_length=1, description="Tools this agent may use"
    )
    runtime: RuntimeConfig = Field(
        default_factory=RuntimeConfig, description="Runtime limits"
    )
    output: OutputConfig = Field(
        default_factory=OutputConfig, description="Output configuration"
    )
    role: Literal["builder", "evaluator"] = Field(
        default="evaluator",
        description="'builder' runs setup phase; 'evaluator' runs test phase",
    )
    stage: Literal["build", "discover", "evaluate", "review"] | None = Field(
        default=None,
        description="Execution stage used by dependency-aware pipelines",
    )
    depends_on: List[str] = Field(
        default_factory=list,
        description="List of upstream agent ids that must finish before this agent runs",
    )
    enabled: bool = Field(default=True, description="Whether agent is active")
    disabled: Optional[bool] = Field(
        default=None, description="Inverse of 'enabled'; deprecated alias for enabled=False"
    )
    tags: List[str] = Field(default_factory=list, description="Optional tags")
    scoring: Optional[AgentScoringConfig] = Field(
        default=None, description="Agent-level scoring config (max_score, task_aggregation_mode)"
    )
    free_task_count: Optional[int] = Field(
        default=None,
        ge=0,
        description="Optional override for query-specific free task count; ignored for build-stage agents.",
    )
    subtask_decomposition_policy: str = Field(
        default="",
        description=(
            "Optional free-text policy that overrides the planner's default 'split by "
            "UI region / domain area' rule. Code-test-style agents typically want "
            "'one subtask per rubric check' / 'one subtask per unit test'; "
            "DOM-interaction agents typically want 'one subtask per interactive element'. "
            "If empty, the planner uses its default decomposition heuristic."
        ),
    )

    @model_validator(mode="after")
    def validate_pipeline_fields(self) -> "AgentConfig":
        if self.stage is None:
            self.stage = "build" if self.role == "builder" else "evaluate"

        deduped: List[str] = []
        seen: set[str] = set()
        for agent_id in self.depends_on:
            normalized = str(agent_id).strip()
            if not normalized or normalized == self.id or normalized in seen:
                continue
            seen.add(normalized)
            deduped.append(normalized)
        self.depends_on = deduped
        return self

    @model_validator(mode="before")
    @classmethod
    def _normalize_disabled(cls, data: Any) -> Any:
        if isinstance(data, dict) and data.get("disabled") is True:
            data = dict(data)
            data["enabled"] = False
            del data["disabled"]
        return data
