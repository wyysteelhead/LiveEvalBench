"""Query generator — produces behavior test queries from fixed standards."""

import asyncio
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage

from ..llm.retry import invoke_with_llm_api_retry, is_transient_llm_error
from ..utils.logger import logger
from .rubric_config import (
    get_interaction_visual_execution_groups,
    get_interaction_visual_standards,
)


RATE_LIMIT_RETRIES = 5
RATE_LIMIT_BACKOFF_BASE_SECONDS = 1.0
RATE_LIMIT_JITTER_SECONDS = 0.35

@dataclass
class BehaviorQuery:
    """A single behavior test query."""
    query_id: str
    standard_id: str
    guideline_ref: str
    query_text: str
    category: str
    scenario_id: Optional[str] = None
    scenario_index: Optional[int] = None
    scenario_weight: float = 1.0
    check_nodes: Optional[List[Dict[str, Any]]] = None
    execution_group_id: Optional[str] = None
    execution_standard_ids: Optional[List[str]] = None

    def to_dict(self) -> dict:
        return {
            "query_id": self.query_id,
            "standard_id": self.standard_id,
            "guideline_ref": self.guideline_ref,
            "query_text": self.query_text,
            "category": self.category,
            "scenario_id": self.scenario_id,
            "scenario_index": self.scenario_index,
            "scenario_weight": self.scenario_weight,
            "check_nodes": self.check_nodes or [],
            "execution_group_id": self.execution_group_id,
            "execution_standard_ids": self.execution_standard_ids or [self.standard_id],
        }


async def generate(
    files: Dict[str, str],
    llm: Optional[Any] = None,
    user_query: Optional[str] = None,
) -> List[BehaviorQuery]:
    """Generate behavior test queries from unified rubric JSON.

    For each standard in rubric.interaction_visual.standards, the LLM writes
    one concrete UI-specific query. Every standard must produce at least one
    query — the web agent determines not_applicable at runtime.

    Args:
        files: Dict mapping filename to file content
        llm: LangChain LLM instance. If None, returns an empty list.
        user_query: Original prompt used to generate the UI.

    Returns:
        List of BehaviorQuery instances (≥1 per standard, no upper limit).
    """
    if llm is None:
        return []
    standards = get_interaction_visual_standards()
    if not standards:
        return []

    all_source = "\n".join(files.values())
    try:
        return await _generate_from_standards(
            all_source,
            llm,
            user_query=user_query,
            standards_override=standards,
        )
    except Exception:
        logger.exception(
            "Behavior query generation failed; falling back to deterministic queries."
        )
        return _build_fallback_queries(standards)


async def _generate_from_standards(
    source: str,
    llm: Any,
    user_query: Optional[str] = None,
    standards_override: Optional[List[Dict]] = None,
) -> List[BehaviorQuery]:
    standards = (
        standards_override
        if standards_override is not None
        else get_interaction_visual_standards()
    )
    if not standards:
        return []
    selected_ids = {s["standard_id"] for s in standards}
    groups = _build_groups_for_selected_standards(
        selected_ids,
        execution_groups=get_interaction_visual_execution_groups(),
    )
    intent_section = (
        f"The UI was built from this user request: \"{user_query}\"\n\n"
        if user_query
        else ""
    )

    standards_block = "\n\n".join(
        f"standard_id: {s['standard_id']}\n"
        f"type: {s['type']}\n"
        f"description: {s['description']}"
        for s in standards
    )
    groups_block = "\n\n".join(
        f"group_id: {g['group_id']}\n"
        f"description: {g['description']}\n"
        f"scenario_count: {g.get('scenario_count', 1)}\n"
        f"standard_ids: {', '.join(g['standard_ids'])}"
        for g in groups
    )

    prompt = f"""{intent_section}You are a frontend QA engineer writing test tasks for an autonomous browser agent.

The agent can: navigate to URLs, click elements, type text, press keys (Tab, Enter, Space, arrow keys), take screenshots, and read the accessibility tree. It cannot directly inspect CSS computed styles — to verify visual properties like focus rings or hover effects, it must take a screenshot and describe what it sees.

For each execution group below, write the requested number of scenario tasks (scenario_count).
Each scenario task must:
- Start with a specific action ("Press Tab...", "Click...", "Type... then press Enter", "Hover over...")
- End with what the agent should observe or screenshot to verify ("take a screenshot and check if...", "observe whether...", "verify that the page...")
- Reference actual element names or roles found in the source code (not CSS class names)
- Explicitly cover all standard_ids listed in that execution group
- Include at least one branching condition when useful ("if A is absent, try B path and verify ...")
- Be 2-5 sentences

Rules:
- Every execution group must produce exactly scenario_count tasks — do NOT skip any group.
- "generic" standards apply to any UI. Name the specific interactive elements found in this UI.
- "ui_specific" standards: write a task even if you are unsure the feature exists. The agent will return not_applicable at runtime if the feature is absent.
- Do NOT invent features that are not in the source code.

Return ONLY a JSON array. Each element must have:
  group_id
  scenarios: array of scenario objects. Each scenario object must have:
    - scenario_id
    - query_text
    - check_nodes: array. Each node must include:
      - node_id
      - standard_id
      - assertion
      - evidence_required (array of strings like screenshot/dom/text/timing)

Execution groups:
{groups_block}

Standards:
{standards_block}

Source code (first 4000 chars):
```
{source[:4000]}
```

Return only valid JSON array, no explanation."""

    response = await _ainvoke_with_rate_limit_retry(llm, [
        SystemMessage(content="You are a frontend QA engineer. Output only valid JSON."),
        HumanMessage(content=prompt),
    ])

    content = response.content.strip()
    if content.startswith("```"):
        content = re.sub(r"^```[a-z]*\n?", "", content)
        content = re.sub(r"\n?```$", "", content)

    raw = json.loads(content)

    # Build lookups
    standard_map = {s["standard_id"]: s for s in standards}
    group_map = {g["group_id"]: g for g in groups}

    queries: List[BehaviorQuery] = []
    covered_ids = set()
    for item in raw:
        gid = item.get("group_id")
        if not gid:
            continue
        group = group_map.get(gid)
        if not group:
            continue
        scenarios = item.get("scenarios")
        if not isinstance(scenarios, list):
            # Backward compatibility with old shape: one query_text per group.
            fallback_text = str(item.get("query_text", "")).strip()
            if fallback_text:
                scenarios = [{
                    "scenario_id": f"{gid}_s1",
                    "query_text": fallback_text,
                    "check_nodes": [
                        {
                            "node_id": f"{sid}_n1",
                            "standard_id": sid,
                            "assertion": standard_map[sid]["description"],
                            "evidence_required": ["screenshot", "text"],
                        }
                        for sid in group["standard_ids"] if sid in standard_map
                    ],
                }]
            else:
                scenarios = []

        for scenario_index, sc in enumerate(scenarios, start=1):
            query_text = str(sc.get("query_text", "")).strip()
            if not query_text:
                continue

            scenario_id = str(sc.get("scenario_id", "")).strip() or f"{gid}_s{scenario_index}"
            raw_nodes = sc.get("check_nodes")
            if not isinstance(raw_nodes, list):
                raw_nodes = []

            node_standard_ids = [
                str(n.get("standard_id", "")).strip()
                for n in raw_nodes
                if isinstance(n, dict)
            ]
            model_ids = [sid for sid in node_standard_ids if sid in set(group["standard_ids"])]
            group_ids = [sid for sid in group["standard_ids"] if sid in set(model_ids)] or group["standard_ids"]
            group_ids = [sid for sid in group_ids if sid in standard_map]
            if not group_ids:
                continue

            node_map = {}
            for node in raw_nodes:
                if not isinstance(node, dict):
                    continue
                sid = str(node.get("standard_id", "")).strip()
                if sid in group_ids:
                    node_map.setdefault(sid, []).append(node)

            for sid in group_ids:
                std = standard_map[sid]
                covered_ids.add(sid)
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
                check_nodes = node_map.get(sid) or default_nodes
                queries.append(BehaviorQuery(
                    query_id=f"{sid}__{scenario_id}",
                    standard_id=sid,
                    guideline_ref=std["guideline_ref"],
                    query_text=query_text,
                    category=std["category"],
                    scenario_id=scenario_id,
                    scenario_index=scenario_index,
                    scenario_weight=float(sc.get("scenario_weight", 1.0) or 1.0),
                    check_nodes=check_nodes,
                    execution_group_id=group["group_id"],
                    execution_standard_ids=group_ids,
                ))

    # Guardrail: if LLM output is incomplete, backfill missing standards with single-standard tasks.
    missing_ids = [sid for sid in selected_ids if sid not in covered_ids]
    for sid in missing_ids:
        std = standard_map[sid]
        default_nodes = []
        for idx, n in enumerate(std.get("check_nodes", []) or [], start=1):
            if not isinstance(n, dict):
                continue
            default_nodes.append({
                "node_id": str(n.get("node_id", f"{sid}_fallback_n{idx}")),
                "standard_id": sid,
                "assertion": str(n.get("assertion", std["description"])),
                "evidence_required": n.get("evidence_required", ["screenshot", "text"]),
            })
        if not default_nodes:
            default_nodes = [{
                "node_id": f"{sid}_fallback_n1",
                "standard_id": sid,
                "assertion": std["description"],
                "evidence_required": ["screenshot", "text"],
            }]
        queries.append(BehaviorQuery(
            query_id=f"{sid}__fallback_s1",
            standard_id=sid,
            guideline_ref=std["guideline_ref"],
            query_text=_fallback_query_text(std),
            category=std["category"],
            scenario_id="fallback_s1",
            scenario_index=1,
            scenario_weight=1.0,
            check_nodes=default_nodes,
            execution_group_id=sid,
            execution_standard_ids=[sid],
        ))

    return queries


async def generate_for_standards(
    files: Dict[str, str],
    llm: Any,
    standard_ids: List[str],
    user_query: Optional[str] = None,
) -> List[BehaviorQuery]:
    """Generate queries for a specific subset of standard_ids only.

    Used when re-running a task after some queries were deleted — regenerates
    fresh queries only for the deleted standards.
    """
    if llm is None or not standard_ids:
        return []

    standards = get_interaction_visual_standards()
    requested_standard_ids = {
        str(sid).split("__", 1)[0] for sid in standard_ids
    }
    subset = [s for s in standards if s["standard_id"] in requested_standard_ids]
    if not subset:
        return []

    all_source = "\n".join(files.values())
    try:
        return await _generate_from_standards(all_source, llm, user_query=user_query, standards_override=subset)
    except Exception:
        logger.exception(
            "Behavior query regeneration failed; falling back to deterministic queries."
        )
        return _build_fallback_queries(subset)


def save_queries(queries: List[BehaviorQuery], path: Path) -> None:
    """Persist queries to a JSON sidecar file."""
    path.write_text(
        json.dumps([q.to_dict() for q in queries], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def load_queries_from_list(raw_list: List[Dict]) -> List[BehaviorQuery]:
    """Deserialize a list of query dicts into BehaviorQuery objects."""
    return [
        BehaviorQuery(
            query_id=item["query_id"],
            standard_id=(
                item.get("standard_id")
                or str(item["query_id"]).split("__", 1)[0]
            ),
            guideline_ref=item.get("guideline_ref", ""),
            query_text=item["query_text"],
            category=item.get("category", "behavior"),
            scenario_id=item.get("scenario_id"),
            scenario_index=item.get("scenario_index"),
            scenario_weight=float(item.get("scenario_weight", 1.0) or 1.0),
            check_nodes=item.get("check_nodes") if isinstance(item.get("check_nodes"), list) else [],
            execution_group_id=item.get("execution_group_id"),
            execution_standard_ids=item.get("execution_standard_ids"),
        )
        for item in raw_list
        if isinstance(item, dict) and item.get("query_id")
    ]


def load_queries(path: Path) -> Optional[List[BehaviorQuery]]:
    """Load queries from a JSON sidecar file. Returns None if file doesn't exist."""
    if not path.exists():
        return None
    raw = json.loads(path.read_text(encoding="utf-8"))
    # New sidecar format: dict with behavior_queries key
    if isinstance(raw, dict):
        raw = raw.get("behavior_queries", [])
    if not isinstance(raw, list):
        return None
    return load_queries_from_list(raw)


def _build_groups_for_selected_standards(
    selected_ids: set[str],
    execution_groups: Optional[List[Dict]] = None,
) -> List[Dict]:
    groups: List[Dict] = []
    covered_ids = set()
    for group in execution_groups or []:
        ids = [sid for sid in group["standard_ids"] if sid in selected_ids]
        if not ids:
            continue
        covered_ids.update(ids)
        groups.append({
            "group_id": group["group_id"],
            "description": group["description"],
            "scenario_count": int(group.get("scenario_count", 1) or 1),
            "standard_ids": ids,
        })

    # Any standard not explicitly grouped falls back to a single-standard group.
    for sid in sorted(selected_ids - covered_ids):
        groups.append({
            "group_id": sid,
            "description": f"Single-standard execution group for {sid}.",
            "scenario_count": 1,
            "standard_ids": [sid],
        })
    return groups


def _fallback_query_text(standard: Dict) -> str:
    return (
        f"Test this standard in the current UI: {standard['description']} "
        "Use concrete interactions and state clear observable evidence for pass/fail/not_applicable."
    )


def _build_fallback_queries(standards: List[Dict]) -> List[BehaviorQuery]:
    """Build deterministic queries when LLM generation is unavailable."""
    standard_map = {s["standard_id"]: s for s in standards}
    groups = _build_groups_for_selected_standards(
        set(standard_map.keys()),
        execution_groups=get_interaction_visual_execution_groups(),
    )

    queries: List[BehaviorQuery] = []
    for group in groups:
        group_ids = [sid for sid in group["standard_ids"] if sid in standard_map]
        if not group_ids:
            continue

        scenario_count = max(1, int(group.get("scenario_count", 1) or 1))
        for scenario_index in range(1, scenario_count + 1):
            group_descriptions = "; ".join(standard_map[sid]["description"] for sid in group_ids)
            group_query_text = (
                f"Scenario {scenario_index}: run one end-to-end behavior test flow in the current UI that covers "
                f"all of these standards: {group_descriptions} "
                "Use concrete interactions and include explicit observable evidence for pass/fail/not_applicable."
            )
            scenario_id = f"{group['group_id']}_fallback_s{scenario_index}"

            for sid in group_ids:
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
                    guideline_ref=std["guideline_ref"],
                    query_text=group_query_text,
                    category=std["category"],
                    scenario_id=scenario_id,
                    scenario_index=scenario_index,
                    scenario_weight=1.0,
                    check_nodes=default_nodes,
                    execution_group_id=group["group_id"],
                    execution_standard_ids=group_ids,
                ))
    return queries


def _is_rate_limit_error(error: Exception) -> bool:
    return is_transient_llm_error(error)


def _planner_retry_settings() -> Dict[str, float | int]:
    return {
        "retries": max(0, int(os.getenv("LLM_API_RETRY_COUNT", str(RATE_LIMIT_RETRIES)))),
        "base_delay_seconds": max(
            0.0,
            float(os.getenv("LLM_API_RETRY_BASE_DELAY_SECONDS", str(RATE_LIMIT_BACKOFF_BASE_SECONDS))),
        ),
        "jitter_seconds": max(
            0.0,
            float(os.getenv("LLM_API_RETRY_JITTER_SECONDS", str(RATE_LIMIT_JITTER_SECONDS))),
        ),
    }


async def _ainvoke_with_rate_limit_retry(
    llm: Any,
    messages: List[Any],
    retries: Optional[int] = None,
) -> Any:
    """Invoke LLM with retry-on-rate-limit behavior."""
    retry_settings = _planner_retry_settings()
    return await invoke_with_llm_api_retry(
        lambda: llm.ainvoke(messages),
        retries=retry_settings["retries"] if retries is None else max(0, int(retries)),
        base_delay_seconds=float(retry_settings["base_delay_seconds"]),
        jitter_seconds=float(retry_settings["jitter_seconds"]),
        operation_name="LLM query generation",
        should_retry=_is_rate_limit_error,
    )
