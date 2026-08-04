"""Function-first task planner — two-round planning for interaction_visual phase."""

import asyncio
import json
import re
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from langchain_core.messages import HumanMessage, SystemMessage

from ..utils.logger import logger
from .query_generator import BehaviorQuery, _ainvoke_with_rate_limit_retry
from .rubric_config import get_interaction_visual_execution_groups, get_interaction_visual_standards


PROMPT_CONFIG_PATH = (
    Path(__file__).resolve().parents[3] / "configs" / "planner_prompt_config.json"
)


@lru_cache(maxsize=1)
def _load_prompt_config() -> Dict[str, str]:
    with PROMPT_CONFIG_PATH.open("r", encoding="utf-8") as fh:
        return json.load(fh)


@dataclass
class PlannedTask:
    """Intermediate planning object representing a functional task."""
    task_id: str
    title: str
    task_text: str

    # Classification
    phase: str = "interaction_visual"
    task_type: str = "functional_core"  # functional_core | functional_edge | rubric_gap_fill
    generated_from: str = "round1_functional"  # round1_functional | round2_rubric_alignment
    parent_task_id: Optional[str] = None

    # Requirement mapping
    source_requirement_refs: List[str] = field(default_factory=list)
    source_requirement_texts: List[str] = field(default_factory=list)

    # Rubric mapping
    covers_standard_ids: List[str] = field(default_factory=list)
    rubric_gap_only: bool = False

    # Execution
    scenario_id: Optional[str] = None
    scenario_weight: float = 1.0
    multi_step: bool = True

    # Observability
    expected_signals: List[str] = field(default_factory=list)
    preconditions: List[str] = field(default_factory=list)

    # Provenance
    based_on: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "title": self.title,
            "task_text": self.task_text,
            "phase": self.phase,
            "task_type": self.task_type,
            "generated_from": self.generated_from,
            "parent_task_id": self.parent_task_id,
            "source_requirement_refs": self.source_requirement_refs,
            "source_requirement_texts": self.source_requirement_texts,
            "covers_standard_ids": self.covers_standard_ids,
            "rubric_gap_only": self.rubric_gap_only,
            "scenario_id": self.scenario_id,
            "scenario_weight": self.scenario_weight,
            "multi_step": self.multi_step,
            "expected_signals": self.expected_signals,
            "preconditions": self.preconditions,
            "based_on": self.based_on,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PlannedTask":
        return cls(
            task_id=d["task_id"],
            title=d.get("title", ""),
            task_text=d.get("task_text", ""),
            phase=d.get("phase", "interaction_visual"),
            task_type=d.get("task_type", "functional_core"),
            generated_from=d.get("generated_from", "round1_functional"),
            parent_task_id=d.get("parent_task_id"),
            source_requirement_refs=d.get("source_requirement_refs", []),
            source_requirement_texts=d.get("source_requirement_texts", []),
            covers_standard_ids=d.get("covers_standard_ids", []),
            rubric_gap_only=d.get("rubric_gap_only", False),
            scenario_id=d.get("scenario_id"),
            scenario_weight=float(d.get("scenario_weight", 1.0) or 1.0),
            multi_step=d.get("multi_step", True),
            expected_signals=d.get("expected_signals", []),
            preconditions=d.get("preconditions", []),
            based_on=d.get("based_on", {}),
        )


@dataclass
class SubTaskSpec:
    """Executable task node in tree synthesis mode."""

    sub_task_id: str
    parent_main_task_id: str
    title: str
    goal: str
    task_text: str
    kind: str = "independent"  # independent | staged_flow | system
    stage_index: int = 1
    depends_on_subtask_ids: List[str] = field(default_factory=list)
    needs_clean_state: bool = True
    can_run_parallel: bool = False
    expected_signals: List[str] = field(default_factory=list)
    evidence_requirements: List[str] = field(default_factory=lambda: ["screenshot", "text"])
    dimension_ids: List[str] = field(default_factory=list)
    preconditions: List[str] = field(default_factory=list)
    success_criteria: List[str] = field(default_factory=list)
    failure_policy: str = "continue_parent"
    estimated_cost: str = "medium"
    source_task_id: Optional[str] = None
    based_on: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "sub_task_id": self.sub_task_id,
            "parent_main_task_id": self.parent_main_task_id,
            "title": self.title,
            "goal": self.goal,
            "task_text": self.task_text,
            "kind": self.kind,
            "stage_index": self.stage_index,
            "depends_on_subtask_ids": self.depends_on_subtask_ids,
            "needs_clean_state": self.needs_clean_state,
            "can_run_parallel": self.can_run_parallel,
            "expected_signals": self.expected_signals,
            "evidence_requirements": self.evidence_requirements,
            "dimension_ids": self.dimension_ids,
            "preconditions": self.preconditions,
            "success_criteria": self.success_criteria,
            "failure_policy": self.failure_policy,
            "estimated_cost": self.estimated_cost,
            "source_task_id": self.source_task_id,
            "based_on": self.based_on,
        }

    def to_planned_task(self) -> PlannedTask:
        payload_based_on = dict(self.based_on)
        payload_based_on.update({
            "task_level": "sub_task",
            "task_kind": self.kind,
            "parent_main_task_id": self.parent_main_task_id,
            "subtask_goal": self.goal,
            "constraint_dimension_ids": list(self.dimension_ids),
            "success_criteria": list(self.success_criteria),
            "failure_policy": self.failure_policy,
            "evidence_requirements": list(self.evidence_requirements),
            "estimated_cost": self.estimated_cost,
            "depends_on_subtask_ids": list(self.depends_on_subtask_ids),
            "can_run_parallel": bool(self.can_run_parallel),
            "needs_clean_state": bool(self.needs_clean_state),
            "stage_index": int(self.stage_index),
        })
        if self.source_task_id:
            payload_based_on["source_task_id"] = self.source_task_id
        return PlannedTask(
            task_id=self.sub_task_id,
            title=self.title,
            task_text=self.task_text,
            parent_task_id=self.parent_main_task_id,
            covers_standard_ids=[],
            expected_signals=list(self.expected_signals),
            preconditions=list(self.preconditions),
            based_on=payload_based_on,
            multi_step=self.kind != "independent",
        )


@dataclass
class MainTaskSpec:
    """Planning and aggregation node in tree synthesis mode."""

    main_task_id: str
    title: str
    goal: str
    origin: str = "query_specific"  # fixed_dimension | query_specific | system
    owner_agent_id: Optional[str] = None
    dimension_ids: List[str] = field(default_factory=list)
    query_template_id: Optional[str] = None
    decomposition_policy: str = "single"  # single | independent_set | staged_flow
    max_subtasks: int = 1
    parallelism_hint: str = "serial"
    priority: str = "normal"
    expected_outputs: List[str] = field(default_factory=lambda: ["verdict", "evidence"])
    aggregation_method: str = "mean_completion"
    notes: str = ""
    source_task_ids: List[str] = field(default_factory=list)
    subtasks: List[SubTaskSpec] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "main_task_id": self.main_task_id,
            "title": self.title,
            "goal": self.goal,
            "origin": self.origin,
            "owner_agent_id": self.owner_agent_id,
            "dimension_ids": self.dimension_ids,
            "query_template_id": self.query_template_id,
            "decomposition_policy": self.decomposition_policy,
            "max_subtasks": self.max_subtasks,
            "parallelism_hint": self.parallelism_hint,
            "priority": self.priority,
            "expected_outputs": self.expected_outputs,
            "aggregation_method": self.aggregation_method,
            "notes": self.notes,
            "source_task_ids": self.source_task_ids,
            "subtasks": [subtask.to_dict() for subtask in self.subtasks],
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "MainTaskSpec":
        return cls(
            main_task_id=str(payload.get("main_task_id", "")),
            title=str(payload.get("title", "")),
            goal=str(payload.get("goal", "")),
            origin=str(payload.get("origin", "query_specific") or "query_specific"),
            owner_agent_id=payload.get("owner_agent_id"),
            dimension_ids=[str(item) for item in (payload.get("dimension_ids") or [])],
            query_template_id=payload.get("query_template_id"),
            decomposition_policy=str(payload.get("decomposition_policy", "single") or "single"),
            max_subtasks=max(0, int(payload.get("max_subtasks", 1) or 0)),
            parallelism_hint=str(payload.get("parallelism_hint", "serial") or "serial"),
            priority=str(payload.get("priority", "normal") or "normal"),
            expected_outputs=[str(item) for item in (payload.get("expected_outputs") or ["verdict", "evidence"])],
            aggregation_method=str(payload.get("aggregation_method", "mean_completion") or "mean_completion"),
            notes=str(payload.get("notes", "") or ""),
            source_task_ids=[str(item) for item in (payload.get("source_task_ids") or [])],
            subtasks=[SubTaskSpec(**deepcopy(item)) for item in (payload.get("subtasks") or [])],
        )


@dataclass
class TaskTreePlan:
    """Hierarchical task plan used by tree synthesis mode."""

    synthesis_mode: str = "tree"
    owner_agent_id: Optional[str] = None
    main_tasks: List[MainTaskSpec] = field(default_factory=list)

    def flatten_for_execution(self) -> List[PlannedTask]:
        return build_execution_plan(self)

    def to_dict(self) -> dict:
        return {
            "synthesis_mode": self.synthesis_mode,
            "owner_agent_id": self.owner_agent_id,
            "main_tasks": [main_task.to_dict() for main_task in self.main_tasks],
        }


def _main_task_origin_for_planned_task(task: PlannedTask) -> str:
    if task.task_type == "rubric_gap_fill" or task.rubric_gap_only:
        return "fixed_dimension"
    if task.based_on.get("builder_prerequisite"):
        return "system"
    return "query_specific"


def _normalize_text_label(value: str, fallback: str) -> str:
    normalized = re.sub(r"\s+", " ", str(value or "").strip())
    return normalized or fallback


def _slugify_fragment(value: str, fallback: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", str(value or "").lower()).strip("-")
    return slug or fallback


def _looks_like_flow(task: PlannedTask) -> bool:
    if task.multi_step:
        return True
    text = f"{task.title} {task.task_text}".lower()
    flow_markers = [
        "flow",
        "journey",
        "checkout",
        "submit",
        "sign in",
        "signup",
        "log in",
        "complete",
        "navigate",
        "continue",
        "multi-step",
    ]
    return any(marker in text for marker in flow_markers)


def _task_text_for_dimension(
    dimension_name: str,
    dimension_instruction: str,
    user_query: str,
    scoring: Optional[Any] = None,
) -> str:
    parts = [
        f"Evaluate the current implementation for the dimension '{dimension_name}'.",
        dimension_instruction,
    ]

    if scoring is not None:
        checks = getattr(scoring, "checks", None) or []
        subcriteria = getattr(scoring, "subcriteria", None) or []
        if checks:
            check_lines = "\n".join(
                f"  - {chk.id}: {chk.instruction}" for chk in checks
            )
            parts.append(f"Analyze from the following aspects:\n{check_lines}")
        if subcriteria:
            sub_lines = "\n".join(
                f"  - {sub.id}: {sub.instruction}" for sub in subcriteria
            )
            parts.append(f"Required subcriteria:\n{sub_lines}")

    parts.append(f"Keep the original query intent in scope: {user_query}")
    return "\n\n".join(parts)


def _query_specific_candidates(
    user_query: str,
    requirements: Sequence[Dict[str, Any]],
    count: int,
) -> List[Dict[str, str]]:
    candidates: List[Dict[str, str]] = []
    seen_labels: set[str] = set()
    for requirement in requirements:
        text = _normalize_text_label(requirement.get("text", ""), user_query)
        requirement_id = str(requirement.get("requirement_id") or f"req-{len(candidates)+1}")
        key = text.lower()
        if key in seen_labels:
            continue
        seen_labels.add(key)
        candidates.append({
            "requirement_id": requirement_id,
            "title": text[:80],
            "goal": text,
        })
        if len(candidates) >= count:
            break

    if not candidates:
        candidates.append({
            "requirement_id": "query-1",
            "title": _normalize_text_label(user_query, "Core query scenario")[:80],
            "goal": _normalize_text_label(user_query, "Core query scenario"),
        })

    while len(candidates) < count:
        index = len(candidates) + 1
        candidates.append({
            "requirement_id": f"query-{index}",
            "title": f"Extended scenario {index}",
            "goal": f"Validate an additional implementation-specific scenario for: {user_query}",
        })

    return candidates[:count]


def _strip_json_fence(content: str) -> str:
    normalized = content.strip()
    if normalized.startswith("```"):
        normalized = re.sub(r"^```[a-z]*\n?", "", normalized)
        normalized = re.sub(r"\n?```$", "", normalized)
    return normalized.strip()


_CJK_TEXT_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")


def _contains_cjk_text(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, (list, tuple, set)):
        return any(_contains_cjk_text(item) for item in value)
    if isinstance(value, dict):
        return any(_contains_cjk_text(item) for item in value.values())
    return bool(_CJK_TEXT_RE.search(str(value)))


def _round1_payload_needs_english_rewrite(items: Sequence[Dict[str, Any]]) -> bool:
    user_facing_fields = (
        "title",
        "task_text",
        "expected_signals",
        "preconditions",
        "source_requirement_texts",
    )
    for item in items:
        if not isinstance(item, dict):
            continue
        if any(_contains_cjk_text(item.get(field)) for field in user_facing_fields):
            return True
    return False


async def _rewrite_round1_payload_to_english(
    *,
    llm: Any,
    raw_items: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    rewrite_prompt = (
        "Rewrite this JSON array so every user-facing text field is English only.\n"
        "Preserve the array length, task_id values, task_type values, numeric fields, booleans, refs, and overall semantics.\n"
        "Translate these fields into concise natural English: title, task_text, source_requirement_texts, expected_signals, preconditions.\n"
        "Do not add or remove keys. Return only valid JSON.\n\n"
        f"Input JSON:\n{json.dumps(list(raw_items), ensure_ascii=False, indent=2)}"
    )
    response = await _ainvoke_with_rate_limit_retry(llm, [
        SystemMessage(content="You rewrite browser test task JSON into English-only text. Output only valid JSON."),
        HumanMessage(content=rewrite_prompt),
    ])
    content = _strip_json_fence(str(response.content or ""))
    rewritten = json.loads(content)
    if not isinstance(rewritten, list):
        raise ValueError("Round1 English rewrite must return a JSON array")
    return [item for item in rewritten if isinstance(item, dict)]


def _requirements_by_priority(requirements: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    priority_order = {"primary": 0, "high": 0, "secondary": 1, "medium": 1, "low": 2}
    return sorted(
        [req for req in requirements if isinstance(req, dict)],
        key=lambda req: (
            priority_order.get(str(req.get("priority") or "secondary").lower(), 1),
            str(req.get("requirement_id") or ""),
        ),
    )


_TOOL_CAPABILITY_HINTS = {
    "navigate": "open application pages and move between in-app states",
    "navigate_back": "move backward through browser history when needed",
    "navigate_forward": "move forward through browser history when needed",
    "reload_page": "refresh the current page to verify reload behavior",
    "get_page_context": "inspect visible structure, semantics, and interactive elements",
    "get_global_dom_summary": "inspect a structured summary of major rendered DOM regions and interactive elements",
    "click_element": "trigger element-based click interactions",
    "dblclick_element": "trigger double-click interactions when explicitly required",
    "right_click_element": "open context menus or alternate actions",
    "drag_and_drop": "exercise drag-and-drop interactions",
    "hover_element": "inspect hover-triggered UI states",
    "type_text": "enter text into editable fields",
    "clear_text": "clear inputs before retries or edge-case checks",
    "select_option": "change select/dropdown values",
    "check_element": "toggle checkboxes or switches on",
    "uncheck_element": "toggle checkboxes or switches off",
    "focus_element": "move focus to a target element",
    "press_key": "send keyboard shortcuts or key presses",
    "press_key_on": "send keyboard actions to a specific element",
    "action_sequence": "execute a rapid sequence of browser actions (key presses, clicks, hovers, scrolls) with optional mid-sequence screenshots for before/after comparison",
    "scroll_by": "scroll the page to reach off-screen content",
    "scroll_to_element": "bring a target element into view",
    "wait_for_milliseconds": "wait briefly for async UI state changes",
    "preview_click_at": "inspect coordinate-based click targets before committing the click",
    "click_at": "execute coordinate-based clicks after preview confirmation",
    "dblclick_at": "execute coordinate-based double-clicks after preview confirmation",
    "inspect_last_screenshot": "analyze the latest screenshot for visual evidence",
    "read_file": "inspect source files or logs for implementation evidence",
    "search_source": "search source code for relevant implementation details",
    "local_fs_write": "write or patch local workspace files",
    "local_fs_read": "read local workspace files directly",
    "local_exec_run": "run one-off local commands like installs or builds",
    "local_exec_start": "start long-running local processes like dev servers",
    "runtime_detect": "inspect local runtime availability and versions",
    "local_port_allocate": "allocate reserved local ports for services",
    "local_port_release": "release allocated local ports",
    "protocol_write_artifacts": "write structured build or run artifacts for downstream steps",
    "submit_verdict": "submit a single final verdict",
    "submit_group_verdict": "submit grouped verdicts across rubric dimensions",
}


def _summarize_allowed_tool_capabilities(allowed_tools: Sequence[str]) -> str:
    lines: List[str] = []
    seen: set[str] = set()
    for tool_name in allowed_tools:
        hint = _TOOL_CAPABILITY_HINTS.get(str(tool_name))
        if not hint or hint in seen:
            continue
        seen.add(hint)
        lines.append(f"- {hint}")
    if not lines:
        return "- No capability summary available; rely strictly on the allowed tool names."
    return "\n".join(lines)


def _format_agent_identity_block(agent_profile: Optional[Dict[str, Any]]) -> str:
    if not agent_profile:
        return "Current agent identity: not provided. Plan conservatively for a generic browser evaluator."

    agent_name = _normalize_text_label(agent_profile.get("agent_name", ""), "Unnamed agent")
    agent_id = _normalize_text_label(agent_profile.get("agent_id", ""), agent_name)
    role = _normalize_text_label(agent_profile.get("role", "evaluator"), "evaluator")
    stage = _normalize_text_label(agent_profile.get("stage", ""), role)
    description = _normalize_text_label(agent_profile.get("description", ""), "No description provided.")
    system_prompt = _normalize_text_label(agent_profile.get("system_prompt", ""), "No persona instruction provided.")
    allowed_tools = [str(tool) for tool in (agent_profile.get("allowed_tools") or []) if str(tool).strip()]
    tools_block = "\n".join(f"- {tool}" for tool in allowed_tools) or "- No tools declared"
    capability_block = _summarize_allowed_tool_capabilities(allowed_tools)
    decomposition_policy = str(agent_profile.get("subtask_decomposition_policy", "") or "").strip()
    decomposition_block = (
        f"Subtask decomposition policy (HARD OVERRIDE — supersedes the generic 'split by UI region / aggregate per check' default below):\n{decomposition_policy}\n"
        if decomposition_policy
        else ""
    )

    return (
        "Current agent identity and capability boundary:\n"
        f"- Agent name: {agent_name}\n"
        f"- Agent id: {agent_id}\n"
        f"- Role: {role}\n"
        f"- Stage: {stage}\n"
        f"- Mission: {description}\n"
        f"- Persona instruction: {system_prompt[:1200]}\n"
        "Allowed tools:\n"
        f"{tools_block}\n"
        "Capability summary:\n"
        f"{capability_block}\n"
        f"{decomposition_block}"
        "Planning guardrails:\n"
        "- Keep planned tasks and subtasks inside this role boundary.\n"
        "- Do not create work that requires unavailable tools.\n"
        "- If multiple decompositions are possible, prefer the one that best matches this agent's persona and toolset."
    )


def _format_taskability_block(agent_profile: Optional[Dict[str, Any]]) -> str:
    if not agent_profile:
        return (
            "Taskability routing rules:\n"
            "- Plan conservatively and avoid checks that depend on unavailable observation or implementation evidence.\n"
            "- Do not assign subtle animation, motion smoothness, or fine-grained visual-diff checks unless the agent clearly has the tools to validate them."
        )

    allowed_tools = {str(tool).strip() for tool in (agent_profile.get("allowed_tools") or []) if str(tool).strip()}
    stage = _normalize_text_label(agent_profile.get("stage", ""), _normalize_text_label(agent_profile.get("role", "evaluator"), "evaluator"))

    has_source_access = bool({"read_file", "search_source"} & allowed_tools)
    has_visual_evidence = "inspect_last_screenshot" in allowed_tools
    has_page_context = bool({"get_page_context", "get_global_dom_summary"} & allowed_tools)
    has_hover = "hover_element" in allowed_tools
    has_precision_click = bool({"preview_click_at", "click_at", "dblclick_at"} & allowed_tools)
    has_browser_interaction = bool(
        {
            "click_element", "type_text", "hover_element", "select_option",
            "click_at", "dblclick_at", "press_key", "press_key_on", "scroll_by",
            "scroll_to_element", "check_element", "uncheck_element",
            "drag_and_drop", "focus_element", "clear_text",
            "get_page_context", "get_global_dom_summary",
            "preview_click_at",
        } & allowed_tools
    )
    has_local_exec = bool({"local_exec_run", "local_fs_read", "local_fs_write"} & allowed_tools)

    rules: List[str] = [
        "Taskability routing rules:",
        "- Only assign checks that this agent can validate reliably with its allowed tools.",
        "- If a requirement is hard to verify with this agent's tools, prefer a simpler observable proxy or leave that requirement for a better-suited agent instead of forcing a brittle task.",
    ]

    if has_local_exec and not has_browser_interaction and not has_visual_evidence:
        rules.append(
            "- This is a static-analysis / unit-test evaluator (no browser, no dev server, no screenshots, no playwright). "
            "Sub tasks MUST be expressed as: read source files, run a test command, parse build/test logs. "
            "Never describe sub tasks in terms of DOM rendering, visible regions, hover/scroll/click flows, "
            "screenshots, layout, or visual prominence — even if the page_context or user query suggests them. "
            "Translate any user-facing requirement into the equivalent unit-test or source-inspection check. "
            "If a requirement cannot be re-expressed as a unit test or source inspection, drop it rather than "
            "inventing a DOM-style sub task this agent cannot execute."
        )

    if has_source_access:
        rules.append(
            "- This agent can inspect source code. It may validate whether hard-to-observe behavior is implemented in code, especially for subtle animation, shimmer, motion, or smoothness requirements that are not reliably judgeable from static screenshots alone."
        )
    else:
        rules.append(
            "- This agent cannot inspect source code during execution. Do not assign tasks whose primary success condition depends on source-only facts or implementation details that are not directly observable in the page."
        )

    if has_visual_evidence:
        rules.append(
            "- This agent can inspect screenshots, but static screenshots are not enough for nuanced motion quality, subtle shimmer, or fine-grained animation smoothness. Do not assign those as primary pass/fail checks unless the rendered evidence is obvious."
        )
    else:
        rules.append(
            "- This agent lacks screenshot inspection. Avoid appearance-first judgments that require direct visual confirmation."
        )

    if has_page_context and not has_visual_evidence:
        rules.append(
            "- Structural DOM context tools alone are not enough to prove visual prominence, visibility, or overlay blockage. Avoid tasks that would require that leap."
        )

    if has_hover and not has_source_access:
        rules.append(
            "- Hover checks are allowed only when the visual change should be obvious. Avoid tasks that depend on very subtle lift, shadow, or transient styling differences."
        )

    if has_precision_click:
        rules.append(
            "- Coordinate-based clicks are available but still noisy. Avoid long precision sequences, exact board-game constructions, or negative tests whose conclusion depends on repeated pixel-perfect targeting."
        )

    if stage == "review":
        rules.append(
            "- Review-stage agents should prefer implementation-quality, maintainability, robustness, and engineering-risk questions over pure interaction choreography."
        )
    else:
        rules.append(
            "- Evaluation-stage agents should prefer robust user-observable checks over fragile source-only or frame-perfect validations."
        )

    return "\n".join(rules)


def build_query_specific_main_tasks(
    user_query: str,
    requirements: Sequence[Dict[str, Any]],
    *,
    owner_agent_id: Optional[str],
    count: int,
) -> List[MainTaskSpec]:
    if count <= 0:
        return []

    main_tasks: List[MainTaskSpec] = []
    for index, candidate in enumerate(
        _query_specific_candidates(user_query, _requirements_by_priority(requirements), count),
        start=1,
    ):
        main_tasks.append(MainTaskSpec(
            main_task_id=f"query::{index:02d}",
            title=candidate["title"],
            goal=candidate["goal"],
            origin="query_specific",
            owner_agent_id=owner_agent_id,
            query_template_id=candidate["requirement_id"],
            decomposition_policy="single",
            max_subtasks=3,
            parallelism_hint="mixed",
            priority="normal",
            notes="Query-shared main task template.",
        ))
    return main_tasks


def build_query_main_tasks_from_round1_tasks(
    round1_tasks: Sequence[PlannedTask],
    *,
    owner_agent_id: Optional[str],
    count: Optional[int] = None,
) -> List[MainTaskSpec]:
    selected_tasks = list(round1_tasks)
    if count is not None:
        selected_tasks = selected_tasks[: max(0, int(count))]

    main_tasks: List[MainTaskSpec] = []
    for index, task in enumerate(selected_tasks, start=1):
        requirement_refs = [str(ref) for ref in (task.source_requirement_refs or []) if str(ref).strip()]
        query_template_id = requirement_refs[0] if requirement_refs else str(task.scenario_id or task.task_id)
        notes = f"Query main task synthesized from round1 planned task {task.task_id}."
        if task.source_requirement_texts:
            notes = f"{notes} Source requirements: {' | '.join(task.source_requirement_texts[:3])}"
        main_tasks.append(MainTaskSpec(
            main_task_id=f"query::{index:02d}",
            title=_normalize_text_label(task.title, f"Query task {index}"),
            goal=_normalize_text_label(task.task_text, task.title or f"Query task {index}"),
            origin="query_specific",
            owner_agent_id=owner_agent_id,
            query_template_id=query_template_id,
            decomposition_policy="single",
            max_subtasks=3,
            parallelism_hint="mixed",
            priority="normal",
            notes=notes,
            source_task_ids=[task.task_id],
        ))
    return main_tasks


def build_fixed_dimension_main_tasks(
    dimensions: Sequence[Any],
    *,
    owner_agent_id: Optional[str],
    user_query: str,
) -> List[MainTaskSpec]:
    main_tasks: List[MainTaskSpec] = []
    for dimension in dimensions:
        main_tasks.append(MainTaskSpec(
            main_task_id=f"dimension::{dimension.id}",
            title=f"Assess {dimension.name}",
            goal=_task_text_for_dimension(
                dimension.name, dimension.instruction, user_query,
                scoring=getattr(dimension, "scoring", None),
            ),
            origin="fixed_dimension",
            owner_agent_id=owner_agent_id,
            dimension_ids=[dimension.id],
            decomposition_policy="single",
            max_subtasks=5,
            parallelism_hint="mixed",
            priority="high",
            notes=f"Fixed dimension task for rubric dimension {dimension.id}.",
        ))
    return main_tasks


def _adjust_main_task_decomposition_for_agent(
    main_task: MainTaskSpec,
    *,
    agent_profile: Optional[Dict[str, Any]] = None,
) -> MainTaskSpec:
    adjusted = MainTaskSpec.from_dict(main_task.to_dict())
    role = str((agent_profile or {}).get("role") or "").strip().lower()
    stage = str((agent_profile or {}).get("stage") or "").strip().lower()
    agent_id = str((agent_profile or {}).get("agent_id") or "").strip().lower()
    dimension_ids = {str(item).strip().lower() for item in adjusted.dimension_ids if str(item).strip()}

    # Build-path review dimensions are evidence-centric and usually do not
    # benefit from regional page decomposition. One focused pass over logs +
    # source is typically enough to reach a verdict.
    if role == "evaluator" and stage == "review" and (
        agent_id == "implementation_reviewer"
        or {"deployment_effort", "build_code_health"} & dimension_ids
    ):
        if {"deployment_effort", "build_code_health"} & dimension_ids:
            adjusted.max_subtasks = 1
            adjusted.parallelism_hint = "serial"
            adjusted.notes = (adjusted.notes + " Single evidence pass preferred for build-path review dimensions.").strip()

    return adjusted


def build_fixed_dimension_planned_tasks(
    dimensions: Sequence[Any],
    *,
    generated_from: str = "fixed_dimension_planning",
) -> List[PlannedTask]:
    planned_tasks: List[PlannedTask] = []
    for dimension in dimensions:
        planned_tasks.append(PlannedTask(
            task_id=f"task_dimension_{dimension.id}",
            title=f"Assess {dimension.name}",
            task_text=_task_text_for_dimension(
                dimension.name, dimension.instruction, dimension.name,
                scoring=getattr(dimension, "scoring", None),
            ),
            phase="interaction_visual",
            task_type="rubric_gap_fill",
            generated_from=generated_from,
            covers_standard_ids=[dimension.id],
            rubric_gap_only=True,
            scenario_id=f"task_dimension_{dimension.id}_s1",
            scenario_weight=1.0,
            multi_step=True,
            expected_signals=[f"observable evidence collected for {dimension.id}"],
            preconditions=[],
            based_on={
                "fixed_dimension": True,
                "dimension_id": dimension.id,
                "task_level": "task",
            },
        ))
    return planned_tasks


def _make_single_subtask(
    *,
    parent_main_task_id: str,
    task: PlannedTask,
    index: int,
    kind: str,
) -> SubTaskSpec:
    sub_task_id = f"{parent_main_task_id}::sub::{index:02d}"
    return SubTaskSpec(
        sub_task_id=sub_task_id,
        parent_main_task_id=parent_main_task_id,
        title=task.title,
        goal=task.task_text,
        task_text=task.task_text,
        kind=kind,
        stage_index=index,
        depends_on_subtask_ids=[],
        needs_clean_state=True,
        can_run_parallel=kind == "independent",
        expected_signals=list(task.expected_signals),
        dimension_ids=list(task.covers_standard_ids),
        preconditions=list(task.preconditions),
        success_criteria=list(task.expected_signals),
        failure_policy="continue_parent" if kind == "independent" else "stop_parent",
        source_task_id=task.task_id,
        based_on=dict(task.based_on),
    )


def _make_flow_subtasks(
    *,
    parent_main_task_id: str,
    task: PlannedTask,
) -> List[SubTaskSpec]:
    enter_id = f"{parent_main_task_id}::sub::01"
    verify_id = f"{parent_main_task_id}::sub::02"
    signals = list(task.expected_signals)
    return [
        SubTaskSpec(
            sub_task_id=enter_id,
            parent_main_task_id=parent_main_task_id,
            title=f"Reach flow state: {task.title}",
            goal=f"Complete the setup or entry path for: {task.title}",
            task_text=(
                f"Execute the core flow needed for this scenario without final judgment yet. "
                f"Scenario: {task.task_text}"
            ),
            kind="staged_flow",
            stage_index=1,
            needs_clean_state=True,
            can_run_parallel=False,
            expected_signals=signals,
            dimension_ids=list(task.covers_standard_ids),
            preconditions=list(task.preconditions),
            success_criteria=signals[:1] or ["target flow becomes reachable"],
            failure_policy="stop_parent",
            source_task_id=task.task_id,
            based_on={**task.based_on, "flow_stage": "enter"},
        ),
        SubTaskSpec(
            sub_task_id=verify_id,
            parent_main_task_id=parent_main_task_id,
            title=f"Verify outcome: {task.title}",
            goal=f"Validate the resulting state and observable evidence for: {task.title}",
            task_text=(
                f"Starting from the flow state established by the previous stage, verify the final user-visible outcome. "
                f"Scenario: {task.task_text}"
            ),
            kind="staged_flow",
            stage_index=2,
            depends_on_subtask_ids=[enter_id],
            needs_clean_state=False,
            can_run_parallel=False,
            expected_signals=signals,
            dimension_ids=list(task.covers_standard_ids),
            preconditions=list(task.preconditions),
            success_criteria=signals or ["final state matches expected scenario outcome"],
            failure_policy="stop_parent",
            source_task_id=task.task_id,
            based_on={**task.based_on, "flow_stage": "verify"},
        ),
    ]


_SUBTASK_SYNTHESIS_MAX_ATTEMPTS = 3
_ROUND1_PLANNING_MAX_ATTEMPTS = 3


async def _generate_subtasks_with_llm(
    *,
    llm: Any,
    user_query: str,
    main_task: MainTaskSpec,
    candidate_tasks: Sequence[PlannedTask],
    requirements: Sequence[Dict[str, Any]],
    page_context: Optional[str],
    source_files: Dict[str, str],
    agent_profile: Optional[Dict[str, Any]] = None,
) -> List[SubTaskSpec]:
    if llm is None:
        raise RuntimeError(f"Subtask synthesis requires an LLM for main task {main_task.main_task_id}")

    cfg = _load_prompt_config()
    candidate_tasks_block = json.dumps([task.to_dict() for task in candidate_tasks], ensure_ascii=False, indent=2)
    requirements_block = "\n".join(
        f"- [{req.get('requirement_id', '?')}] {req.get('text', '')}"
        for req in requirements
        if isinstance(req, dict)
    ) or "- [none] No explicit requirements available"
    main_task_block = json.dumps(main_task.to_dict(), ensure_ascii=False, indent=2)
    page_context_block = (page_context or "No page context available")[:3000]
    source_snippet = "\n".join(source_files.values())[:5000]
    prompt = cfg["subtask_user_template"].format(
        agent_identity_block=_format_agent_identity_block(agent_profile),
        taskability_block=_format_taskability_block(agent_profile),
        user_query=user_query,
        main_task_block=main_task_block,
        candidate_tasks_block=candidate_tasks_block or "[]",
        requirements_block=requirements_block,
        page_context_block=page_context_block,
        source_snippet=source_snippet,
    )

    last_error: Optional[Exception] = None
    for attempt in range(1, _SUBTASK_SYNTHESIS_MAX_ATTEMPTS + 1):
        attempt_prompt = prompt
        if attempt > 1:
            attempt_prompt = (
                f"{prompt}\n\n"
                "Your previous response was invalid for subtask synthesis.\n"
                f"Validation error: {last_error}\n"
                "Retry and return only a valid JSON array. Do not include markdown fences, comments, or trailing commas.\n"
                "Every item must contain: title, task_text, area_label, kind."
            )

        try:
            response = await _ainvoke_with_rate_limit_retry(llm, [
                SystemMessage(content=cfg["subtask_system_prompt"]),
                HumanMessage(content=attempt_prompt),
            ])
            raw_items = json.loads(_strip_json_fence(str(response.content or "[]")))
            if not isinstance(raw_items, list):
                raise ValueError(f"Subtask synthesis returned non-list payload for {main_task.main_task_id}")
            if not raw_items:
                raise ValueError(f"Subtask synthesis returned no subtasks for {main_task.main_task_id}")

            subtasks: List[SubTaskSpec] = []
            existing_ids: List[str] = []
            source_task = candidate_tasks[0] if candidate_tasks else None
            dimension_ids = list(source_task.covers_standard_ids) if source_task is not None else list(main_task.dimension_ids)

            for index, item in enumerate(raw_items[: max(1, main_task.max_subtasks)], start=1):
                if not isinstance(item, dict):
                    raise ValueError(f"Subtask item {index} is not an object for {main_task.main_task_id}")
                kind = str(item.get("kind", "independent") or "independent").strip().lower()
                if kind not in {"independent", "staged_flow"}:
                    raise ValueError(f"Subtask item {index} has invalid kind '{kind}' for {main_task.main_task_id}")
                subtask_id = f"{main_task.main_task_id}::sub::{index:02d}"
                depends_on_indexes = [
                    int(dep_index)
                    for dep_index in (item.get("depends_on_indexes") or [])
                    if isinstance(dep_index, int) and 1 <= dep_index < index
                ]
                depends_on_ids = [existing_ids[dep_index - 1] for dep_index in depends_on_indexes if dep_index - 1 < len(existing_ids)]
                title = _normalize_text_label(item.get("title", ""), "")
                task_text = _normalize_text_label(item.get("task_text", ""), "")
                area_label = _normalize_text_label(item.get("area_label", ""), "")
                if not title or not task_text or not area_label:
                    raise ValueError(
                        f"Subtask item {index} missing required title/task_text/area_label for {main_task.main_task_id}"
                    )
                based_on = {
                    "task_origin": main_task.origin,
                    "area_label": area_label,
                    "llm_subtask": True,
                }
                if source_task is not None:
                    based_on["source_task_id"] = source_task.task_id
                subtasks.append(SubTaskSpec(
                    sub_task_id=subtask_id,
                    parent_main_task_id=main_task.main_task_id,
                    title=title,
                    goal=area_label,
                    task_text=task_text,
                    kind=kind,
                    stage_index=index,
                    depends_on_subtask_ids=depends_on_ids,
                    needs_clean_state=bool(item.get("needs_clean_state", kind != "staged_flow" or index == 1)),
                    can_run_parallel=bool(item.get("can_run_parallel", kind == "independent" and not depends_on_ids)),
                    expected_signals=[str(signal) for signal in (item.get("expected_signals") or []) if signal is not None][:4],
                    dimension_ids=dimension_ids,
                    preconditions=[str(condition) for condition in (item.get("preconditions") or []) if condition is not None][:3],
                    success_criteria=[str(rule) for rule in (item.get("success_criteria") or []) if rule is not None][:4],
                    failure_policy="stop_parent" if kind == "staged_flow" else "continue_parent",
                    source_task_id=source_task.task_id if source_task is not None else None,
                    based_on=based_on,
                ))
                existing_ids.append(subtask_id)

            if not subtasks:
                raise ValueError(f"Subtask synthesis produced no valid subtasks for {main_task.main_task_id}")

            return subtasks
        except Exception as exc:
            last_error = exc
            if attempt >= _SUBTASK_SYNTHESIS_MAX_ATTEMPTS:
                raise ValueError(
                    f"Subtask synthesis failed after {_SUBTASK_SYNTHESIS_MAX_ATTEMPTS} attempts for {main_task.main_task_id}: {exc}"
                ) from exc
            logger.warning(
                "[TaskPlanner] Subtask synthesis attempt %s/%s failed for %s: %s",
                attempt,
                _SUBTASK_SYNTHESIS_MAX_ATTEMPTS,
                main_task.main_task_id,
                exc,
            )


async def _elaborate_main_task_subtasks(
    *,
    main_task: MainTaskSpec,
    candidate_tasks: Sequence[PlannedTask],
    llm: Any,
    user_query: str,
    requirements: Sequence[Dict[str, Any]],
    page_context: Optional[str],
    source_files: Dict[str, str],
    agent_profile: Optional[Dict[str, Any]] = None,
) -> List[SubTaskSpec]:
    return await _generate_subtasks_with_llm(
        llm=llm,
        user_query=user_query,
        main_task=main_task,
        candidate_tasks=candidate_tasks,
        requirements=requirements,
        page_context=page_context,
        source_files=source_files,
        agent_profile=agent_profile,
    )


async def _generate_subtasks_for_all_main_tasks(
    *,
    llm: Any,
    user_query: str,
    main_tasks: Sequence[MainTaskSpec],
    requirements: Sequence[Dict[str, Any]],
    page_context: Optional[str],
    source_files: Dict[str, str],
    agent_profile: Optional[Dict[str, Any]] = None,
    generate_query_count: int = 0,
) -> Tuple[Dict[str, List[SubTaskSpec]], List[MainTaskSpec]]:
    """Batch subtask synthesis: one LLM call for all main tasks.

    When generate_query_count > 0, the LLM is also asked to generate additional
    query-specific main tasks with their own subtasks, avoiding overlap with the
    fixed-dimension tasks passed in main_tasks.

    Returns a (subtasks_by_id, new_query_main_tasks) tuple.
    subtasks_by_id maps main_task_id -> list of SubTaskSpec for every key
    returned by the LLM (including newly generated query task IDs).
    new_query_main_tasks contains the MainTaskSpec objects created from newly
    generated query task keys.
    """
    if llm is None:
        raise RuntimeError("Batched subtask synthesis requires an LLM")
    if not main_tasks:
        return {}, []

    cfg = _load_prompt_config()
    fixed_main_tasks_block = json.dumps(
        [mt.to_dict() for mt in main_tasks],
        ensure_ascii=False, indent=2,
    )
    requirements_block = "\n".join(
        f"- [{req.get('requirement_id', '?')}] {req.get('text', '')}"
        for req in requirements
        if isinstance(req, dict)
    ) or "- [none] No explicit requirements available"
    page_context_block = (page_context or "No page context available")[:3000]
    source_snippet = "\n".join(source_files.values())[:5000]

    # `generate_query_count` 3 cases, distinguished by whether pre-built query::
    # main tasks are already in `main_tasks` (shared/pre-built path vs NoQST ablation):
    #   >0               : LLM ALSO creates new query:: main tasks + subtasks.
    #   =0 + pre-built   : LLM returns subtasks for the query:: keys already listed
    #                      (do NOT forbid — strict models like kimi-k2.6 would then
    #                      omit query:: keys, leaving subtasks empty/pending forever).
    #   =0 + no pre-built: NoQST ablation — forbid query:: outright (hard gate below).
    has_prebuilt_query = any(m.main_task_id.startswith("query::") for m in main_tasks)
    query_generation_block = ""
    if (generate_query_count or 0) > 0:
        query_generation_block = (
            f"\nAdditionally, generate **{generate_query_count} query-specific main "
            f"tasks** that cover aspects of the user query NOT already covered by "
            f"the fixed rubric dimensions above. These query tasks must have distinct "
            f"focus areas and each must include its own sub tasks.\n"
            f"Use main_task_id values like \"query::01\", \"query::02\", etc.\n"
            f"The returned JSON dict MUST include keys for all fixed-dimension "
            f"main_task_ids above AND the {generate_query_count} new query task keys."
        )
    elif has_prebuilt_query:
        query_generation_block = (
            "\nThe query::XX main_task_ids already listed above are pre-built "
            "query-specific tasks — return subtask arrays for EACH of them. Do NOT "
            "omit any main_task_id listed above, and do not invent extra query:: tasks.\n"
        )
    else:
        # NoQST ablation (query_specific_main_task_count=0, no pre-built query tasks):
        # explicitly forbid query-specific tasks. The hard gate below ALSO drops any
        # query:: keys the LLM emits despite this instruction, so the ablation is
        # enforced deterministically regardless of LLM compliance.
        query_generation_block = (
            "\nDo NOT generate any query-specific (query::) main tasks. "
            "Return subtasks ONLY for the fixed-dimension main_task_ids listed "
            "above. Any query::XX keys you emit will be discarded.\n"
        )

    prompt = cfg["subtask_user_template"].format(
        agent_identity_block=_format_agent_identity_block(agent_profile),
        taskability_block=_format_taskability_block(agent_profile),
        user_query=user_query,
        fixed_main_tasks_block=fixed_main_tasks_block,
        query_generation_block=query_generation_block,
        requirements_block=requirements_block,
        page_context_block=page_context_block,
        source_snippet=source_snippet,
    )

    existing_ids = {mt.main_task_id for mt in main_tasks}

    last_error: Optional[Exception] = None
    for attempt in range(1, _SUBTASK_SYNTHESIS_MAX_ATTEMPTS + 1):
        attempt_prompt = prompt
        if attempt > 1:
            attempt_prompt = (
                f"{prompt}\n\n"
                "Your previous response was invalid for batch subtask synthesis.\n"
                f"Validation error: {last_error}\n"
                "Retry and return only a valid JSON object (dict). "
                "Every main_task_id from the input must appear as a key. "
                "When generating query-specific tasks, also include the new "
                "query::XX keys with their sub task arrays. "
                "Each value must be a valid array of sub task objects. "
                "Do not include markdown fences, comments, or trailing commas.\n"
                "Every sub task must contain: title, task_text, area_label, kind."
            )

        try:
            response = await _ainvoke_with_rate_limit_retry(llm, [
                SystemMessage(content=cfg["subtask_system_prompt"]),
                HumanMessage(content=attempt_prompt),
            ])
            raw = json.loads(_strip_json_fence(str(response.content or "{}")))
            if not isinstance(raw, dict):
                raise ValueError(f"Batched subtask synthesis returned non-dict payload")

            result: Dict[str, List[SubTaskSpec]] = {}
            new_query_main_tasks: List[MainTaskSpec] = []

            # Process all keys from the LLM response
            for mid, items in raw.items():
                if not isinstance(items, list):
                    raise ValueError(f"Entry for {mid} is not an array")

                is_new = mid not in existing_ids

                if is_new:
                    if (generate_query_count or 0) == 0:
                        # NoQST hard gate: count=0 forbids query-specific tasks.
                        # Silently drop any query:: key the LLM emitted despite the
                        # prohibition prompt above, so the ablation is enforced
                        # deterministically. (count>0 path unchanged below.)
                        continue
                    # This is a newly generated query-specific main task
                    if not mid.startswith("query::"):
                        raise ValueError(
                            f"Newly generated main task ID '{mid}' must use query::XX format"
                        )
                    # Determine title and goal from the first subtask
                    first_item = items[0] if items else {}
                    if isinstance(first_item, dict):
                        query_title = _normalize_text_label(
                            first_item.get("title", ""), mid
                        ) or mid
                        query_goal = _normalize_text_label(
                            first_item.get("area_label", ""), query_title
                        ) or query_title
                    else:
                        query_title = mid
                        query_goal = mid

                    # Create MainTaskSpec for the new query task
                    query_main_task = MainTaskSpec(
                        main_task_id=mid,
                        title=query_title,
                        goal=query_goal,
                        origin="query_specific",
                        owner_agent_id=main_tasks[0].owner_agent_id if main_tasks else None,
                        dimension_ids=[],
                        decomposition_policy="single",
                        max_subtasks=len(items),
                        parallelism_hint="mixed",
                        priority="normal",
                        notes="LLM-generated query-specific main task (merged synthesis).",
                    )
                    new_query_main_tasks.append(query_main_task)
                    # Inherit owner_agent_id for its subtasks
                    parent_id = mid
                    parent_origin = "query_specific"
                    parent_dim_ids: List[str] = []
                else:
                    # Existing fixed-dimension main task
                    mt = next((m for m in main_tasks if m.main_task_id == mid), None)
                    if mt is None:
                        raise ValueError(f"Missing input main task for key {mid}")
                    parent_id = mt.main_task_id
                    parent_origin = mt.origin
                    parent_dim_ids = list(mt.dimension_ids)

                subtasks: List[SubTaskSpec] = []
                existing_sub_ids: List[str] = []
                max_sub = 5 if not is_new else 3

                for index, item in enumerate(items[: max(1, max_sub)], start=1):
                    if not isinstance(item, dict):
                        raise ValueError(f"Subtask item {index} for {mid} is not an object")
                    kind = str(item.get("kind", "independent") or "independent").strip().lower()
                    if kind not in {"independent", "staged_flow"}:
                        raise ValueError(f"Subtask item {index} for {mid} has invalid kind '{kind}'")
                    subtask_id = f"{mid}::sub::{index:02d}"
                    depends_on_indexes = [
                        int(dep_index)
                        for dep_index in (item.get("depends_on_indexes") or [])
                        if isinstance(dep_index, int) and 1 <= dep_index < index
                    ]
                    depends_on_ids = [
                        existing_sub_ids[dep_index - 1]
                        for dep_index in depends_on_indexes
                        if dep_index - 1 < len(existing_sub_ids)
                    ]
                    title = _normalize_text_label(item.get("title", ""), "")
                    task_text = _normalize_text_label(item.get("task_text", ""), "")
                    area_label = _normalize_text_label(item.get("area_label", ""), "")
                    if not title or not task_text or not area_label:
                        raise ValueError(
                            f"Subtask item {index} for {mid} missing required title/task_text/area_label"
                        )
                    based_on: Dict[str, Any] = {
                        "task_origin": parent_origin,
                        "area_label": area_label,
                        "llm_subtask": True,
                    }
                    subtasks.append(SubTaskSpec(
                        sub_task_id=subtask_id,
                        parent_main_task_id=parent_id,
                        title=title,
                        goal=area_label,
                        task_text=task_text,
                        kind=kind,
                        stage_index=index,
                        depends_on_subtask_ids=depends_on_ids,
                        needs_clean_state=bool(item.get("needs_clean_state", True)),
                        can_run_parallel=bool(
                            item.get("can_run_parallel", kind == "independent" and not depends_on_ids)
                        ),
                        expected_signals=[str(s) for s in (item.get("expected_signals") or []) if s is not None][:4],
                        dimension_ids=parent_dim_ids,
                        preconditions=[str(c) for c in (item.get("preconditions") or []) if c is not None][:3],
                        success_criteria=[str(r) for r in (item.get("success_criteria") or []) if r is not None][:4],
                        failure_policy="stop_parent" if kind == "staged_flow" else "continue_parent",
                        based_on=based_on,
                    ))
                    existing_sub_ids.append(subtask_id)

                if not subtasks:
                    raise ValueError(f"No valid subtasks generated for {mid}")
                result[mid] = subtasks

            return result, new_query_main_tasks

        except Exception as exc:
            last_error = exc
            if attempt >= _SUBTASK_SYNTHESIS_MAX_ATTEMPTS:
                raise ValueError(
                    f"Batched subtask synthesis failed after {_SUBTASK_SYNTHESIS_MAX_ATTEMPTS} attempts: {exc}"
                ) from exc
            logger.warning(
                "[TaskPlanner] Batched subtask synthesis attempt %s/%s failed; retrying: %s",
                attempt,
                _SUBTASK_SYNTHESIS_MAX_ATTEMPTS,
                exc,
            )

    raise RuntimeError("Unreachable — batched subtask synthesis retry loop exhausted")


async def synthesize_task_tree_for_agent(
    *,
    owner_agent_id: Optional[str],
    user_query: str,
    requirements: Sequence[Dict[str, Any]],
    dimensions: Sequence[Any],
    round1_tasks: Sequence[PlannedTask],
    aligned_tasks: Sequence[PlannedTask],
    query_specific_main_task_count: int,
    llm: Any = None,
    page_context: Optional[str] = None,
    source_files: Optional[Dict[str, str]] = None,
    agent_profile: Optional[Dict[str, Any]] = None,
    shared_query_main_tasks: Optional[Sequence[MainTaskSpec]] = None,
) -> TaskTreePlan:
    """Create a hierarchical plan with fixed-dimension and query-specific main tasks.

    Fixed-dimension main tasks are built from the rubric. Query-specific main
    tasks and their subtasks are generated together with fixed-dimension subtasks
    in a single batched LLM call, allowing the LLM to avoid overlap.
    """
    # Build fixed-dimension main tasks from rubric
    fixed_main_tasks = build_fixed_dimension_main_tasks(
        dimensions,
        owner_agent_id=owner_agent_id,
        user_query=user_query,
    )

    # Build query-specific main tasks
    if shared_query_main_tasks:
        # Pre-built from shared query DB (cross-agent)
        query_main_tasks = [MainTaskSpec.from_dict(task.to_dict()) for task in shared_query_main_tasks]
    else:
        # Query tasks will be generated during the batched subtask call
        query_main_tasks = []

    for mt in query_main_tasks:
        mt.owner_agent_id = owner_agent_id

    # Apply agent-profile adjustments to fixed main tasks
    all_adjusted: List[MainTaskSpec] = []
    for mt in fixed_main_tasks:
        adjusted = _adjust_main_task_decomposition_for_agent(mt, agent_profile=agent_profile)
        adjusted.source_task_ids = []
        all_adjusted.append(adjusted)
    for mt in query_main_tasks:
        adjusted = _adjust_main_task_decomposition_for_agent(mt, agent_profile=agent_profile)
        all_adjusted.append(adjusted)

    # Single batched LLM call: generates subtasks for fixed tasks AND
    # creates new query-specific main tasks with their subtasks
    generate_query = query_specific_main_task_count if not shared_query_main_tasks else 0
    subtasks_by_id, new_query_main_tasks = await _generate_subtasks_for_all_main_tasks(
        llm=llm,
        user_query=user_query,
        main_tasks=all_adjusted,
        requirements=requirements,
        page_context=page_context,
        source_files=source_files or {},
        agent_profile=agent_profile,
        generate_query_count=generate_query,
    )

    # Apply agent-profile adjustments to newly generated query tasks
    for mt in new_query_main_tasks:
        mt.owner_agent_id = owner_agent_id
        _adjust_main_task_decomposition_for_agent(mt, agent_profile=agent_profile)

    # Distribute subtasks back and fill post-processing fields
    def _finalise_main_task(mt: MainTaskSpec) -> MainTaskSpec:
        mt.subtasks = list(subtasks_by_id.get(mt.main_task_id, []))
        if any(s.kind == "staged_flow" for s in mt.subtasks):
            mt.decomposition_policy = "staged_flow"
        elif len(mt.subtasks) > 1:
            mt.decomposition_policy = "independent_set"
        else:
            mt.decomposition_policy = "single"
        return mt

    final_fixed = [_finalise_main_task(mt) for mt in all_adjusted[:len(fixed_main_tasks)]]

    # For pre-built shared query tasks: bind source_task_id from round1 tasks
    # (metadata used for traceability, does not affect LLM output)
    round1_candidates = list(round1_tasks)[: max(0, len(query_main_tasks))]
    query_source_task_map: Dict[str, Optional[PlannedTask]] = {}
    for index, mt in enumerate(query_main_tasks):
        if round1_candidates:
            source_task_ids = {tid for tid in (mt.source_task_ids or []) if tid}
            task = next((item for item in round1_candidates if item.task_id in source_task_ids), None)
            if task is None and index < len(round1_candidates):
                task = round1_candidates[index]
            query_source_task_map[mt.main_task_id] = task
        else:
            query_source_task_map[mt.main_task_id] = None

    final_query = [_finalise_main_task(mt) for mt in all_adjusted[len(fixed_main_tasks):]]
    final_new_query = [_finalise_main_task(mt) for mt in new_query_main_tasks]

    # Bind source_task_id for shared query subtasks (metadata only)
    for mt in final_query:
        source_task = query_source_task_map.get(mt.main_task_id)
        if source_task is not None:
            mt.source_task_ids = [source_task.task_id]
            for subtask in mt.subtasks:
                subtask.source_task_id = source_task.task_id
                subtask.based_on["source_task_id"] = source_task.task_id

    return TaskTreePlan(
        synthesis_mode="tree",
        owner_agent_id=owner_agent_id,
        main_tasks=[*final_fixed, *final_query, *final_new_query],
    )


def build_execution_plan(task_tree_plan: TaskTreePlan) -> List[PlannedTask]:
    """Flatten a task tree into dependency-respecting execution order."""
    ordered: List[PlannedTask] = []
    for main_task in task_tree_plan.main_tasks:
        subtasks = sorted(main_task.subtasks, key=lambda subtask: (subtask.stage_index, subtask.sub_task_id))
        pending = {subtask.sub_task_id: subtask for subtask in subtasks}
        emitted: set[str] = set()
        while pending:
            ready = [
                subtask for subtask in pending.values()
                if all(dep_id in emitted for dep_id in subtask.depends_on_subtask_ids)
            ]
            if not ready:
                ready = [pending[min(pending.keys())]]
            for subtask in sorted(ready, key=lambda item: (item.stage_index, item.sub_task_id)):
                emitted.add(subtask.sub_task_id)
                pending.pop(subtask.sub_task_id, None)
                ordered.append(subtask.to_planned_task())
    return ordered


def synthesize_task_tree_from_planned_tasks(
    planned_tasks: List[PlannedTask],
    *,
    owner_agent_id: Optional[str] = None,
    query_specific_main_task_count: Optional[int] = None,
) -> TaskTreePlan:
    """Build a minimal tree plan from existing flat tasks.

    This compatibility layer preserves current execution semantics by mapping
    each flat planned task to a main task with exactly one executable sub task.
    Later iterations can replace this synthesis with richer decomposition.
    """
    main_tasks: List[MainTaskSpec] = []
    for task in planned_tasks:
        main_task_id = task.parent_task_id or f"main::{task.task_id}"
        subtask = SubTaskSpec(
            sub_task_id=task.task_id,
            parent_main_task_id=main_task_id,
            title=task.title,
            goal=task.task_text,
            task_text=task.task_text,
            kind="system" if task.based_on.get("builder_prerequisite") else "independent",
            needs_clean_state=True,
            can_run_parallel=False,
            expected_signals=list(task.expected_signals),
            dimension_ids=list(task.covers_standard_ids),
            preconditions=list(task.preconditions),
            success_criteria=list(task.expected_signals),
            failure_policy="continue_parent",
            source_task_id=task.task_id,
            based_on=dict(task.based_on),
        )
        origin = _main_task_origin_for_planned_task(task)
        main_tasks.append(MainTaskSpec(
            main_task_id=main_task_id,
            title=task.title,
            goal=task.task_text,
            origin=origin,
            owner_agent_id=owner_agent_id,
            dimension_ids=list(task.covers_standard_ids),
            decomposition_policy="single",
            max_subtasks=(
                max(1, int(query_specific_main_task_count or 1))
                if origin == "query_specific"
                else 1
            ),
            parallelism_hint="serial",
            notes=f"Compat tree synthesized from planned task {task.task_id}.",
            subtasks=[subtask],
        ))
    return TaskTreePlan(synthesis_mode="tree", owner_agent_id=owner_agent_id, main_tasks=main_tasks)


def build_task_tree_payload(
    task_tree_plan: TaskTreePlan,
    executed_tasks: List[Dict[str, Any]],
    *,
    aggregation_mode: str = "strict",
) -> List[Dict[str, Any]]:
    """Attach execution results to a synthesized task tree plan.

    Args:
        aggregation_mode: ``"strict"`` (default) — any failed subtask → main
            task completion_score = 0. ``"cumulative"`` — mean of subtask scores.
    """
    executed_by_id = {
        str(task.get("task_id")): task
        for task in executed_tasks
        if isinstance(task, dict) and task.get("task_id") is not None
    }
    payload: List[Dict[str, Any]] = []

    for main_task in task_tree_plan.main_tasks:
        subtask_payloads: List[Dict[str, Any]] = []
        completion_values: List[float] = []
        verdicts: List[str] = []
        for subtask in main_task.subtasks:
            executed = dict(executed_by_id.get(subtask.sub_task_id, {}))
            completion_score = executed.get("completion_score")
            if isinstance(completion_score, (int, float)):
                completion_values.append(float(completion_score))
            verdict = str(executed.get("verdict", "pending") or "pending")
            verdicts.append(verdict)
            subtask_payloads.append({
                **subtask.to_dict(),
                "dimension_ids": [],
                "status": str(executed.get("status", verdict) or verdict),
                "verdict": verdict,
                "reason": str(executed.get("reason", "") or ""),
                "completion_score": completion_score,
                "dimension_results": executed.get("rubric_results") or executed.get("dimensions") or [],
                "evidence": [],
                "trajectory": executed.get("trajectory") if isinstance(executed.get("trajectory"), dict) else None,
                "blocked_by": executed.get("blocked_by") if isinstance(executed.get("blocked_by"), list) else [],
                "duration_ms": (
                    int(((executed.get("trajectory") or {}).get("duration_ms", 0)) or 0)
                    if isinstance(executed.get("trajectory"), dict)
                    else 0
                ),
            })

        normalized_verdicts = {value.lower() for value in verdicts if value}
        if normalized_verdicts == {"passed"}:
            main_verdict = "passed"
        elif normalized_verdicts == {"failed"}:
            main_verdict = "failed"
        elif normalized_verdicts:
            main_verdict = "partial"
        else:
            main_verdict = "pending"

        if aggregation_mode == "strict" and completion_values:
            has_failure = any(v <= 0.0 for v in completion_values)
            main_completion = 0.0 if has_failure else 1.0
        else:
            main_completion = (
                sum(completion_values) / len(completion_values)
                if completion_values else None
            )

        payload.append({
            **main_task.to_dict(),
            "status": main_verdict,
            "verdict": main_verdict,
            "completion_score": main_completion,
            "subtask_count": len(main_task.subtasks),
            "completed_subtask_count": len([s for s in subtask_payloads if s.get("status") != "pending"]),
            "dimension_coverage": list(main_task.dimension_ids),
            "key_findings": [
                subtask_payload.get("reason", "")
                for subtask_payload in subtask_payloads
                if subtask_payload.get("reason")
            ][:3],
            "failure_stage": next(
                (subtask_payload.get("title") for subtask_payload in subtask_payloads if subtask_payload.get("verdict") == "failed"),
                None,
            ),
            "aggregated_evidence_refs": [],
            "subtasks": subtask_payloads,
        })

    return payload


async def plan_round1(
    user_query: str,
    requirements: List[Dict],
    files: Dict[str, str],
    llm: Any,
    initial_screenshot_b64: Optional[str] = None,
    page_context: Optional[str] = None,
    planning_notes: Optional[str] = None,
    agent_profile: Optional[Dict[str, Any]] = None,
) -> List[PlannedTask]:
    """Round 1: function-first task planning.

    Generates PlannedTask list based on user query, requirements, source code,
    and initial page context. Does NOT use rubric as input.
    """
    behavior_reqs = [r for r in requirements if r.get("phase") in ("behavior", "visual")]
    if not behavior_reqs:
        behavior_reqs = requirements

    req_block = "\n".join(
        f"- [{r.get('requirement_id', '?')}] ({r.get('priority', 'secondary')}) {r.get('text', '')}"
        for r in behavior_reqs
    )

    all_source = "\n".join(files.values())
    source_snippet = all_source[:4000]

    context_section = ""
    if page_context:
        context_section += f"\nPage accessibility context (interactive elements):\n{page_context[:2000]}\n"
    if initial_screenshot_b64:
        context_section += "\n[Initial screenshot provided — use it to understand the UI layout and entry points.]\n"
    if planning_notes:
        context_section += f"\nAdditional planning context from upstream personas:\n{planning_notes[:3000]}\n"

    cfg = _load_prompt_config()
    prompt = cfg["round1_user_template"].format(
        agent_identity_block=_format_agent_identity_block(agent_profile),
        taskability_block=_format_taskability_block(agent_profile),
        user_query=user_query,
        req_block=req_block,
        context_section=context_section,
        source_snippet=source_snippet,
    )

    last_error: Optional[Exception] = None
    for attempt in range(1, _ROUND1_PLANNING_MAX_ATTEMPTS + 1):
        attempt_prompt = prompt
        if attempt > 1:
            attempt_prompt = (
                f"{prompt}\n\n"
                "Your previous response was invalid for round1 task planning.\n"
                f"Validation error: {last_error}\n"
                "Retry and return only a valid JSON array. Do not include markdown fences, comments, trailing commas, or unescaped quotes inside JSON strings.\n"
                "Every item must contain: task_id, title, task_text, task_type, source_requirement_refs, source_requirement_texts, covers_standard_ids, scenario_id, scenario_weight, multi_step, expected_signals, preconditions."
            )

        try:
            response = await _ainvoke_with_rate_limit_retry(llm, [
                SystemMessage(content=cfg["round1_system_prompt"]),
                HumanMessage(content=attempt_prompt),
            ])
            content = _strip_json_fence(str(response.content or ""))
            raw = json.loads(content)
            if not isinstance(raw, list):
                raise ValueError("Round 1 planning must return a JSON array")
            if _round1_payload_needs_english_rewrite(raw):
                logger.warning("[TaskPlanner] Round 1 returned non-English task text; rewriting to English")
                raw = await _rewrite_round1_payload_to_english(llm=llm, raw_items=raw)
            tasks = []
            for item in raw:
                if not isinstance(item, dict) or not item.get("task_id"):
                    continue
                tasks.append(PlannedTask(
                    task_id=str(item["task_id"]),
                    title=str(item.get("title", "")),
                    task_text=str(item.get("task_text", "")),
                    phase="interaction_visual",
                    task_type=str(item.get("task_type", "functional_core")),
                    generated_from="round1_functional",
                    parent_task_id=None,
                    source_requirement_refs=list(item.get("source_requirement_refs") or []),
                    source_requirement_texts=list(item.get("source_requirement_texts") or []),
                    covers_standard_ids=[],
                    rubric_gap_only=False,
                    scenario_id=str(item.get("scenario_id") or f"{item['task_id']}_s1"),
                    scenario_weight=float(item.get("scenario_weight", 1.0) or 1.0),
                    multi_step=bool(item.get("multi_step", True)),
                    expected_signals=list(item.get("expected_signals") or []),
                    preconditions=list(item.get("preconditions") or []),
                    based_on={
                        "query": True,
                        "code": True,
                        "screenshot": initial_screenshot_b64 is not None,
                        "a11y_tree": page_context is not None,
                    },
                ))
            if not tasks:
                logger.warning(
                    "[TaskPlanner] Round 1 returned an empty task list (LLM returned %s); "
                    "no query-specific tasks will be executed",
                    "[]" if raw == [] else "items with missing task_id",
                )
            else:
                logger.info("[TaskPlanner] Round 1: generated %d tasks", len(tasks))
            return tasks
        except Exception as exc:
            last_error = exc
            if attempt < _ROUND1_PLANNING_MAX_ATTEMPTS:
                logger.warning(
                    "[TaskPlanner] Round 1 attempt %d/%d returned invalid payload; retrying: %s",
                    attempt,
                    _ROUND1_PLANNING_MAX_ATTEMPTS,
                    exc,
                )
                continue
            logger.exception("[TaskPlanner] Round 1 failed; returning empty task list")
            return []

    logger.error("[TaskPlanner] Round 1 exhausted retries unexpectedly; returning empty task list")
    return []


async def _plan_round2_with_standards(
    tasks_v1: List[PlannedTask],
    standards: List[Dict],
    requirements: List[Dict],
    llm: Any,
    agent_profile: Optional[Dict[str, Any]] = None,
) -> List[PlannedTask]:
    """Core round-2 rubric alignment logic, parameterised by a standards list."""
    if not standards:
        return tasks_v1

    standards_block = "\n".join(
        f"- {s['standard_id']}: {s['description']}"
        for s in standards
    )

    tasks_block = json.dumps([t.to_dict() for t in tasks_v1], ensure_ascii=False, indent=2)

    req_block = "\n".join(
        f"- [{r.get('requirement_id', '?')}] {r.get('text', '')}"
        for r in requirements
    )

    cfg = _load_prompt_config()
    prompt = cfg["round2_user_template"].format(
        agent_identity_block=_format_agent_identity_block(agent_profile),
        taskability_block=_format_taskability_block(agent_profile),
        req_block=req_block,
        tasks_block=tasks_block,
        standards_block=standards_block,
    )

    try:
        response = await _ainvoke_with_rate_limit_retry(llm, [
            SystemMessage(content=cfg["round2_system_prompt"]),
            HumanMessage(content=prompt),
        ])
        content = response.content.strip()
        if content.startswith("```"):
            content = re.sub(r"^```[a-z]*\n?", "", content)
            content = re.sub(r"\n?```$", "", content)
        raw = json.loads(content)

        valid_standard_ids = {s["standard_id"] for s in standards}
        tasks_v2 = []
        for item in raw:
            if not isinstance(item, dict) or not item.get("task_id"):
                continue
            covers = [sid for sid in (item.get("covers_standard_ids") or []) if sid in valid_standard_ids]
            item["covers_standard_ids"] = covers
            tasks_v2.append(PlannedTask.from_dict(item))

        covered_ids = {sid for t in tasks_v2 for sid in t.covers_standard_ids}
        missing = [s["standard_id"] for s in standards if s["standard_id"] not in covered_ids]
        if missing:
            raise ValueError(
                "Round 2 planning did not cover required standards: " + ", ".join(sorted(missing))
            )

        logger.info("[TaskPlanner] Round 2: %d tasks (was %d)", len(tasks_v2), len(tasks_v1))
        return tasks_v2
    except Exception:
        logger.exception("[TaskPlanner] Round 2 failed")
        raise


async def plan_round2(
    tasks_v1: List[PlannedTask],
    requirements: List[Dict],
    llm: Any,
    agent_profile: Optional[Dict[str, Any]] = None,
) -> List[PlannedTask]:
    """Round 2: rubric alignment using interaction_visual standards from rubric_config.json."""
    standards = get_interaction_visual_standards()
    execution_groups = get_interaction_visual_execution_groups()

    # Append execution group info to standards descriptions when available
    if execution_groups:
        groups_by_sid: Dict[str, List[str]] = {}
        for g in execution_groups:
            for sid in g.get("standard_ids", []):
                groups_by_sid.setdefault(sid, []).append(g["group_id"])
        standards = [
            {**s, "description": s["description"] + (
                f" [groups: {', '.join(groups_by_sid[s['standard_id']])}]"
                if s["standard_id"] in groups_by_sid else ""
            )}
            for s in standards
        ]

    return await _plan_round2_with_standards(tasks_v1, standards, requirements, llm, agent_profile=agent_profile)


async def plan_round2_agentic(
    tasks_v1: List[PlannedTask],
    dimensions: List[Any],
    requirements: List[Dict],
    llm: Any,
    agent_profile: Optional[Dict[str, Any]] = None,
) -> List[PlannedTask]:
    """Round 2 for agentic mode: align tasks with agent rubric dimensions.

    Args:
        tasks_v1: Tasks from round 1.
        dimensions: List of DimensionConfig from the agent's rubric.
        requirements: Decomposed requirements list.
        llm: LLM instance.

    Returns:
        tasks_v2 with covers_standard_ids populated using dimension IDs.
    """
    standards = [
        {
            "standard_id": d.id,
            "description": d.instruction,
            "guideline_ref": d.name,
            "category": "behavior",
            "check_nodes": [],
        }
        for d in dimensions
    ]
    return await _plan_round2_with_standards(tasks_v1, standards, requirements, llm, agent_profile=agent_profile)


def compile_to_behavior_queries(
    tasks: List[PlannedTask],
    standards: Optional[List[Dict]] = None,
) -> List[BehaviorQuery]:
    """Compile PlannedTask list into BehaviorQuery list.

    Each PlannedTask expands into N BehaviorQuery objects (one per covered standard),
    all sharing the same execution_group_id (task_id) so the agent runs the task once.
    """
    if standards is None:
        standards = get_interaction_visual_standards()
    standard_map = {s["standard_id"]: s for s in standards}

    queries: List[BehaviorQuery] = []
    for task in tasks:
        covers = task.covers_standard_ids
        if not covers:
            # Task has no standard coverage — skip (shouldn't happen after round2 guardrail)
            continue

        valid_covers = [sid for sid in covers if sid in standard_map]
        if not valid_covers:
            continue

        scenario_id = task.scenario_id or f"{task.task_id}_s1"
        execution_group_id = task.task_id  # All queries from this task share the same group

        for sid in valid_covers:
            std = standard_map[sid]
            default_nodes = []
            for idx, n in enumerate(std.get("check_nodes", []) or [], start=1):
                if not isinstance(n, dict):
                    continue
                default_nodes.append({
                    "node_id": str(n.get("node_id", f"{sid}_{scenario_id}_n{idx}")),
                    "standard_id": sid,
                    "assertion": str(n.get("assertion", std["description"])),
                    "evidence_required": n.get("evidence_required", ["screenshot", "text"]),
                })
            if not default_nodes:
                default_nodes = [{
                    "node_id": f"{sid}_{scenario_id}_n1",
                    "standard_id": sid,
                    "assertion": std["description"],
                    "evidence_required": ["screenshot", "text"],
                }]

            queries.append(BehaviorQuery(
                query_id=f"{sid}__{scenario_id}",
                standard_id=sid,
                guideline_ref=std.get("guideline_ref", ""),
                query_text=task.task_text,
                category=std.get("category", "behavior"),
                scenario_id=scenario_id,
                scenario_index=1,
                scenario_weight=task.scenario_weight,
                check_nodes=default_nodes,
                execution_group_id=execution_group_id,
                execution_standard_ids=valid_covers,
            ))

    logger.info("[TaskPlanner] Compiled %d tasks → %d BehaviorQuery objects", len(tasks), len(queries))
    return queries
