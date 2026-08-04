"""System prompt rendering for the frontend evaluator agent.

Prompt text and tool-usage guidance are stored in
configs/agent_prompt_config.json. This module only composes and renders
those config blocks.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .config import AgentConfig

# Lazy import helper to avoid circular dependencies at module level
_TOOL_REGISTRY: Any = None
_GET_TOOL_DEFS: Any = None


def _ensure_tool_registry_loaded() -> None:
    global _TOOL_REGISTRY, _GET_TOOL_DEFS
    if _TOOL_REGISTRY is not None:
        return
    from ..tools.registry import TOOL_REGISTRY as _reg, get_tool_definitions as _gtd
    _TOOL_REGISTRY = _reg
    _GET_TOOL_DEFS = _gtd


PROMPT_CONFIG_PATH = (
    Path(__file__).resolve().parents[3] / "configs" / "agent_prompt_config.json"
)
COORDINATE_CLICK_CONFIRM_TOOLS = {"click_at", "dblclick_at"}
COORDINATE_CLICK_PREVIEW_TOOLS = {"preview_click_at"}


@lru_cache(maxsize=1)
def _load_prompt_config() -> Dict[str, Any]:
    with PROMPT_CONFIG_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def _tools_include_coordinate_click(allowed_tools: Optional[List[str]]) -> bool:
    if allowed_tools is None:
        return True
    allowed = set(allowed_tools)
    return bool(COORDINATE_CLICK_CONFIRM_TOOLS.intersection(allowed) and COORDINATE_CLICK_PREVIEW_TOOLS.intersection(allowed))


def _render_tool_guidelines(
    section: str,
    *,
    max_iterations: Optional[int] = None,
    allowed_tools: Optional[List[str]] = None,
) -> str:
    cfg = _load_prompt_config()
    rules = cfg["tool_guidelines"][section]

    rendered: List[str] = []
    for line in rules.get("always", []):
        if "{max_iterations}" in line:
            rendered.append(line.format(max_iterations=max_iterations))
        else:
            rendered.append(line)

    if _tools_include_coordinate_click(allowed_tools):
        rendered.append(rules["coordinate_click_when_available"])
    elif rules.get("coordinate_click_when_unavailable"):
        rendered.append(rules["coordinate_click_when_unavailable"])

    return "\n".join(rendered)


def _build_available_tools_block(
    allowed_tools: Optional[List[str]],
) -> str:
    """Build the available-tools section of the system prompt, filtered by allowed_tools."""
    _ensure_tool_registry_loaded()

    if allowed_tools is not None:
        tool_defs = _GET_TOOL_DEFS(allowed_tools)
    else:
        tool_defs = [
            {"name": t["name"], "description": t["description"], "input_schema": t["input_schema"]}
            for t in _TOOL_REGISTRY.values()
        ]

    lines: List[str] = []
    for i, tool in enumerate(tool_defs, 1):
        name = tool["name"]
        params = tool.get("input_schema", {}).get("properties", {})
        param_str = ", ".join(params.keys()) if params else ""
        desc = (tool["description"] or "").split("\n")[0].strip()
        lines.append(f"{i}. **{name}({param_str})** - {desc}")

    return "\n".join(lines)


def _build_check_example(check_id: str) -> Dict[str, Any]:
    return {
        "check_id": check_id,
        "status": "passed",
        "reason": f"Observed outcome for {check_id}.",
        "evidence": [f"Tool result supporting {check_id}"],
    }


def _build_subcriterion_example(subcriterion_id: str, rating_values: List[str]) -> Dict[str, Any]:
    rating = rating_values[0] if rating_values else "good"
    return {
        "subcriterion_id": subcriterion_id,
        "rating": rating,
        "reason": f"Assessment for {subcriterion_id}.",
        "evidence": [f"Evidence supporting {subcriterion_id}"],
    }


def _render_submission_examples(agent_config: "AgentConfig") -> str:
    """Generate a simple submit_verdict example for the agent."""
    for dim in agent_config.rubric.dimensions:
        payload: Dict[str, Any] = {
            "verdict": "passed",
            "reason": f"Concrete conclusion for {dim.id}.",
        }
        return (
            f"Example:\n```json\n{json.dumps(payload, indent=2, ensure_ascii=True)}\n```"
        )
    return ""


def format_system_prompt(
    user_query: str,
    app_url: str,
    max_iterations: int,
    standard_ids: Optional[List[str]] = None,
    agent_config: Optional["AgentConfig"] = None,
    allowed_tools: Optional[List[str]] = None,
) -> str:
    """Format the default evaluator prompt from config.

    When agent_config is provided, the agent's persona (system_prompt) is
    injected so the LLM adopts the correct role for this evaluation task.

    When allowed_tools is provided, only those tools are listed in the prompt
    and the coordinate-click guidelines are included only when the
    corresponding tools are available.
    """
    cfg = _load_prompt_config()

    standard_ids_block = ""
    if standard_ids:
        standard_ids_block = cfg["system_standard_ids_rule_template"].format(
            joined_ids=", ".join(standard_ids)
        )

    persona_block = ""
    if agent_config is not None:
        persona_block = (
            f"\n## Your Role\n\n"
            f"You are **{agent_config.name}**. {agent_config.system_prompt}\n"
        )
        if allowed_tools is None:
            allowed_tools = list(agent_config.allowed_tools) if agent_config.allowed_tools else None

    return cfg["system_template"].format(
        user_query=user_query,
        app_url=app_url,
        available_tools_block=_build_available_tools_block(allowed_tools),
        tool_usage_guidelines=_render_tool_guidelines(
            "system", max_iterations=max_iterations, allowed_tools=allowed_tools
        ),
        standard_ids_block=f"\n{standard_ids_block}" if standard_ids_block else "",
        example_flow="\n".join(cfg["system_example_flow"]),
        persona_block=persona_block,
    )


def format_agentic_prompt(
    agent_config: "AgentConfig",
    app_url: str,
    max_iterations: Optional[int] = None,
) -> str:
    """Format the agentic system prompt from config and role toolset."""
    cfg = _load_prompt_config()

    dim_lines: List[str] = []
    for dim in agent_config.rubric.dimensions:
        scoring_lines: List[str] = []
        if dim.scoring:
            scoring_lines.append(
                f"Scoring method: {dim.scoring.method}; "
                f"objective_weight={dim.scoring.objective_weight:.2f}; "
                f"subjective_weight={dim.scoring.subjective_weight:.2f}"
            )
            if dim.scoring.checks:
                scoring_lines.append("Objective checks:")
                for chk in dim.scoring.checks:
                    scoring_lines.append(
                        f"- {chk.id} ({chk.name}, weight={chk.weight:.2f}): {chk.instruction}"
                    )
            if dim.scoring.subcriteria:
                scoring_lines.append("Subjective subcriteria:")
                for sub in dim.scoring.subcriteria:
                    scoring_lines.append(
                        f"- {sub.id} ({sub.name}, weight={sub.weight:.2f}): {sub.instruction}"
                    )
        scoring_block = ("\n" + "\n".join(scoring_lines)) if scoring_lines else ""
        dim_lines.append(
            f"### {dim.id} — {dim.name} (dimension_weight={dim.weight:.2f})\n"
            f"{dim.instruction}{scoring_block}"
        )

    allow_na = agent_config.output.allow_not_applicable
    na_policy = (
        "The `not_applicable` verdict is allowed when a dimension is genuinely "
        "out of scope for this application."
        if allow_na
        else "The `not_applicable` verdict is **not** permitted for this agent."
    )

    rendered = cfg["agentic_template"].format(
        agent_name=agent_config.name,
        system_prompt=agent_config.system_prompt,
        app_url=app_url,
        scoring_mode=agent_config.rubric.scoring_mode,
        na_policy=na_policy,
        dimensions_block="\n\n".join(dim_lines),
        tools_list="\n".join(f"- `{t}`" for t in agent_config.allowed_tools),
        tool_usage_guidelines=_render_tool_guidelines(
            "agentic",
            max_iterations=(
                max_iterations
                if max_iterations is not None
                else agent_config.runtime.max_steps
            ),
            allowed_tools=agent_config.allowed_tools,
        ),
        submission_examples_block=_render_submission_examples(agent_config),
        na_verdict_option=", `not_applicable`" if allow_na else "",
        max_steps=agent_config.runtime.max_steps,
    )

    # Replace submit_group_verdict instructions with submit_verdict.
    rendered = rendered.replace(
        "call **submit_group_verdict** with one entry per dimension.",
        "call **submit_verdict** with your verdict and reason.",
    )
    # Remove multi-dimension fields that don't apply to submit_verdict.
    rendered = rendered.replace(
        "Each entry must include:\n"
        "- `standard_id`: the dimension **id** from the rubric above\n"
        "- `verdict`: one of `passed`, `failed`",
        "Your submission must include:\n"
        "- `verdict`: one of `passed`, `failed`",
    )
    rendered = rendered.replace(
        "- `evidence`: array of concrete evidence items (tool output snippets, observed state changes)\n"
        "- `checks`: required whenever the dimension defines objective checks\n"
        "- `subcriteria`: required whenever the dimension defines subjective subcriteria\n\n"
        "Every dimension must have an explicit `verdict`. Do not omit it, even in score-first mode.\n\n",
        "\n",
    )
    rendered = rendered.replace(
        "For better downstream analysis, also include these optional fields per dimension when available:\n"
        "- `confidence`: `high` | `medium` | `low`\n"
        "- `severity`: `critical` | `major` | `minor` | `none`\n"
        "- `observations`: short bullet-style strings of what you observed\n"
        "- `recommendation`: one short fix suggestion\n\n"
        "Required item shapes when present:\n"
        "- `checks`: `{{\"check_id\":\"...\",\"status\":\"passed|failed|not_applicable\",\"reason\":\"...\",\"evidence\":[]}}`\n"
        "- `subcriteria`: `{{\"subcriterion_id\":\"...\",\"rating\":\"good|ok|poor\",\"reason\":\"...\",\"evidence\":[]}}`",
        "",
    )

    return rendered
