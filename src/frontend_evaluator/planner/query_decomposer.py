"""User-query decomposition into planner phases."""

import json
import re
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage


_EMPTY_DECOMPOSITION = {
    "source_requirements": [],
    "dom_requirements": [],
    "behavior_requirements": [],
    "visual_requirements": [],
}


async def decompose(user_query: Optional[str], llm: Optional[Any]) -> Dict:
    """Decompose user query into source/dom/behavior/visual requirement lists."""
    if not user_query:
        return dict(_EMPTY_DECOMPOSITION)
    if llm is None:
        return _fallback_from_text(user_query)

    prompt = f"""
You are decomposing a frontend request into four evaluation phases.

User query:
{user_query}

Return ONLY valid JSON object with exactly these keys:
- source_requirements: list
- dom_requirements: list
- behavior_requirements: list
- visual_requirements: list

Each list item must be an object:
{{"requirement_id": "phase_N", "phase": "source|dom|behavior|visual", "text": "...", "priority": "primary" or "secondary"}}

Where requirement_id follows the pattern: phase abbreviation + underscore + sequential number (e.g. "behavior_1", "source_2").

LITERAL FIDELITY (most important):
- Extract ONLY requirements that the user explicitly stated or that are clearly implied by an explicit value in the query. Do NOT invent, "fill in", or upgrade requirements based on what a typical implementation would have.
- Preserve every concrete value from the user query EXACTLY as written: colors, sizes, copy text, framework/library choices, component names, counts, animations, ranges, etc. Never substitute, generalize, or paraphrase a value (e.g. "white" must stay "white" — not "light wood", "cream", "near-white", or "pale").
- If the user query is short or vague, returning a small number of requirements is correct. Prefer empty lists over invented requirements. Do NOT fabricate visual style, color, layout, framework, or accessibility constraints that the user did not mention.
- Do NOT pull constraints from frontend best-practice norms (e.g. "should be centered", "should be responsive", "should use SVG", "should be red") unless the user query says so.
- Each `text` field should faithfully restate the user's own intent. Quote or near-quote the user's wording rather than rewriting it in your own design language.

Phase routing:
- Put tech stack, code quality, maintainability, semantic/source constraints into source_requirements.
- Put runtime DOM/accessibility structure constraints into dom_requirements.
- Put interaction/functional flow requirements into behavior_requirements.
- Put visual style/layout/look-and-feel requirements into visual_requirements.
- If a phase has no clear requirement, return [] for that phase.
- Keep items concise and non-duplicated.
"""

    response = await llm.ainvoke([
        SystemMessage(content="Output only valid JSON."),
        HumanMessage(content=prompt),
    ])
    content = str(response.content).strip()
    if content.startswith("```"):
        content = re.sub(r"^```[a-zA-Z]*\n?", "", content)
        content = re.sub(r"\n?```$", "", content)

    try:
        raw = json.loads(content)
    except Exception:
        return _fallback_from_text(user_query)
    return _normalize(raw)


def compute_phase_requirement_scores(
    query_decomposition: Dict,
    source_results: List[Any],
    dom_results: List[Any],
    behavior_results: List[Dict],
    visual_results: List[Any],
) -> Dict:
    """Compute phase-level proxy scores with merged interaction+visual phase."""
    source_score = _score_check_results(source_results)
    dom_score = _score_check_results(dom_results)
    behavior_score = _score_behavior_results(behavior_results)
    legacy_visual_score = _score_check_results(visual_results)

    # New flow: visual rubric is executed in the behavior phase.
    merged_interaction_visual_score = (
        behavior_score if behavior_score is not None else legacy_visual_score
    )

    source_reqs = query_decomposition.get("source_requirements") or []
    dom_reqs = query_decomposition.get("dom_requirements") or []
    behavior_reqs = query_decomposition.get("behavior_requirements") or []
    visual_reqs = query_decomposition.get("visual_requirements") or []
    merged_interaction_visual_reqs = behavior_reqs + visual_reqs

    def _phase_entry(reqs: List[Any], score: Optional[float]) -> Dict[str, Any]:
        if not reqs:
            return {"requirements": 0, "score": None}
        return {
            "requirements": len(reqs),
            "score": round(score, 3) if score is not None else None,
        }

    output: Dict[str, Any] = {
        "source": _phase_entry(source_reqs, source_score),
        "dom": _phase_entry(dom_reqs, dom_score),
        "interaction_visual": _phase_entry(
            merged_interaction_visual_reqs, merged_interaction_visual_score
        ),
        # Backward-compatible keys for old consumers.
        "behavior": _phase_entry(behavior_reqs, merged_interaction_visual_score),
        "visual": _phase_entry(visual_reqs, merged_interaction_visual_score),
    }

    active_scores = []
    if source_reqs and source_score is not None:
        active_scores.append(source_score)
    if dom_reqs and dom_score is not None:
        active_scores.append(dom_score)
    if merged_interaction_visual_reqs and merged_interaction_visual_score is not None:
        active_scores.append(merged_interaction_visual_score)

    output["query_fulfillment_score"] = (
        round(sum(active_scores) / len(active_scores), 3) if active_scores else None
    )
    return output


def _normalize(raw: Dict) -> Dict:
    out = dict(_EMPTY_DECOMPOSITION)
    phase_counters: Dict[str, int] = {}
    phase_map = {
        "source_requirements": "source",
        "dom_requirements": "dom",
        "behavior_requirements": "behavior",
        "visual_requirements": "visual",
    }
    for key in out.keys():
        items = raw.get(key) if isinstance(raw, dict) else None
        if not isinstance(items, list):
            continue
        phase = phase_map.get(key, key.replace("_requirements", ""))
        norm_items = []
        for item in items:
            if isinstance(item, str):
                text = item.strip()
                if text:
                    phase_counters[phase] = phase_counters.get(phase, 0) + 1
                    norm_items.append({
                        "requirement_id": f"{phase}_{phase_counters[phase]}",
                        "phase": phase,
                        "text": text,
                        "priority": "secondary",
                    })
                continue
            if not isinstance(item, dict):
                continue
            text = str(item.get("text", "")).strip()
            if not text:
                continue
            priority = str(item.get("priority", "secondary")).lower().strip()
            if priority not in ("primary", "secondary"):
                priority = "secondary"
            phase_counters[phase] = phase_counters.get(phase, 0) + 1
            req_id = str(item.get("requirement_id", "")).strip() or f"{phase}_{phase_counters[phase]}"
            norm_items.append({
                "requirement_id": req_id,
                "phase": phase,
                "text": text,
                "priority": priority,
            })
        out[key] = norm_items
    return out


def _fallback_from_text(user_query: str) -> Dict:
    q = user_query.lower()
    out = dict(_EMPTY_DECOMPOSITION)
    if any(k in q for k in ("typescript", "react", "vue", "next", "maintain", "组件", "技术栈")):
        out["source_requirements"].append({"requirement_id": "source_1", "phase": "source", "text": user_query, "priority": "primary"})
    if any(k in q for k in ("accessibility", "aria", "label", "无障碍", "dom")):
        out["dom_requirements"].append({"requirement_id": "dom_1", "phase": "dom", "text": user_query, "priority": "secondary"})
    if any(k in q for k in ("submit", "login", "search", "flow", "交互", "功能", "点击", "输入")):
        out["behavior_requirements"].append({"requirement_id": "behavior_1", "phase": "behavior", "text": user_query, "priority": "primary"})
    if any(k in q for k in ("style", "visual", "design", "layout", "color", "风格", "视觉", "配色")):
        out["visual_requirements"].append({"requirement_id": "visual_1", "phase": "visual", "text": user_query, "priority": "secondary"})
    if not any(out.values()):
        out["behavior_requirements"].append({"requirement_id": "behavior_1", "phase": "behavior", "text": user_query, "priority": "primary"})
    return out


def _score_check_results(results: List[Any]) -> Optional[float]:
    if not results:
        return None
    total = len(results)
    if total == 0:
        return None
    passed = sum(1 for r in results if getattr(r, "passed", False))
    return passed / total


def _score_behavior_results(results: List[Dict]) -> Optional[float]:
    if not results:
        return None
    # Scenario-aware scoring: aggregate to standard_id first.
    by_standard: Dict[str, List[bool]] = {}
    for r in results:
        sid = str(r.get("standard_id") or r.get("query_id") or "").strip()
        if not sid:
            continue
        v = r.get("verdict", {}) or {}
        vt = str(v.get("verdict", "passed" if v.get("passed") else "failed")).lower()
        if vt == "not_applicable":
            continue
        by_standard.setdefault(sid, []).append(vt == "passed")

    if not by_standard:
        return None

    per_standard_scores = [
        (sum(1 for ok in checks if ok) / len(checks))
        for checks in by_standard.values()
        if checks
    ]
    if not per_standard_scores:
        return None
    return sum(per_standard_scores) / len(per_standard_scores)
