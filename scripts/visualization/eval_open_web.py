#!/usr/bin/env python3
"""run_web_open_tree.py - tree-aware web dashboard for eval_open.py reports.

The dashboard is file-backed and understands both the current eval_open output
and older ad-hoc JSON arrays. It renders task status, build metadata, preview
links, agent results, and audit activity from the optional JSONL audit log.
"""
from __future__ import annotations

import argparse
import asyncio
import ast
import html
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List
from urllib.parse import quote

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse

from frontend_evaluator.utils.logger import logger
from frontend_evaluator.web.dashboard_shared import SHARED_AGENT_DETAIL_CSS, SHARED_AGENT_DETAIL_JS


def _read_report(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {
            "mode": "open",
            "results": [],
            "summary": {"total": 0, "passed": 0, "partial": 0, "failed": 0},
            "_meta": {"report_path": str(path), "exists": False, "results_count": 0},
        }

    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        data = {"mode": "open", "results": data}
    if not isinstance(data, dict):
        raise RuntimeError("report json must be an object or an array")

    results = data.get("results")
    if not isinstance(results, list):
        data["results"] = []

    data.setdefault("mode", "open")
    data.setdefault("summary", {})
    data.setdefault("_meta", {})
    data["_meta"].update({
        "report_path": str(path),
        "exists": True,
        "results_count": len(data.get("results", [])),
    })
    return data


def _rows(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = payload.get("results")
    return [row for row in rows if isinstance(row, dict)] if isinstance(rows, list) else []


def _read_audit(path: Path | None) -> List[Dict[str, Any]]:
    if not path or not path.exists():
        return []

    entries: List[Dict[str, Any]] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(entry, dict):
            entries.append(entry)
    return entries


def _normalize_status(value: Any) -> str | None:
    if isinstance(value, bool):
        return "passed" if value else "failed"
    if value is None:
        return None

    normalized = str(value).strip().lower()
    mapping = {
        "true": "passed",
        "false": "failed",
        "completed": "passed",
        "success": "passed",
        "error": "failed",
        "compile_error": "failed",
    }
    if normalized in {"passed", "partial", "failed"}:
        return normalized
    return mapping.get(normalized)


def _report_from_row(row: Dict[str, Any]) -> Dict[str, Any]:
    report = row.get("result")
    if isinstance(report, dict):
        return report
    fallback = row.get("report")
    return fallback if isinstance(fallback, dict) else {}


def _parse_verdicts(raw: Any) -> list | None:
    if isinstance(raw, list):
        return raw
    if isinstance(raw, str):
        stripped = raw.strip()
        for parser in (_try_parse_json, _try_parse_python):
            parsed = parser(stripped)
            if isinstance(parsed, list):
                return parsed
    return None


def _try_parse_json(text: str) -> Any:
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None


def _try_parse_python(text: str) -> Any:
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return None


def _extract_build_info(build_result: Dict[str, Any]) -> Dict[str, Any]:
    verdict = build_result.get("verdict")
    if isinstance(verdict, dict):
        return verdict

    verdicts_raw = build_result.get("verdicts")
    items = _parse_verdicts(verdicts_raw)
    if isinstance(items, list):
        for item in items:
            if isinstance(item, dict) and str(item.get("standard_id") or "").strip() == "build_success":
                item_verdict = str(item.get("verdict") or "").strip().lower()
                return {
                    "status": item_verdict,
                    "passed": item_verdict == "passed",
                    "reason": str(item.get("reason") or ""),
                }

    return {}


def _format_fact_value(value: Any) -> str:
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    if isinstance(value, list):
        if value and all(isinstance(item, (str, int, float, bool)) or item is None for item in value):
            return ", ".join(str(item) for item in value)
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _normalize_pipeline_handoffs(report: Dict[str, Any]) -> List[Dict[str, Any]]:
    handoffs = report.get("pipeline_handoffs")
    if not isinstance(handoffs, list):
        return []

    normalized_handoffs: List[Dict[str, Any]] = []
    for index, handoff in enumerate(handoffs):
        if not isinstance(handoff, dict):
            continue
        facts = handoff.get("facts") if isinstance(handoff.get("facts"), dict) else {}
        warnings = handoff.get("warnings") if isinstance(handoff.get("warnings"), list) else []
        blockers = handoff.get("blockers") if isinstance(handoff.get("blockers"), list) else []
        evidence_refs = handoff.get("evidence_refs") if isinstance(handoff.get("evidence_refs"), list) else []
        normalized_handoffs.append({
            "agent_id": handoff.get("producer_agent_id") or handoff.get("agent_id") or f"handoff-{index + 1}",
            "stage": handoff.get("stage") or "unknown",
            "status": handoff.get("status") or "unknown",
            "summary": handoff.get("summary") or "",
            "facts": facts,
            "warnings": [str(item) for item in warnings if item is not None],
            "blockers": [str(item) for item in blockers if item is not None],
            "evidence_refs": [str(item) for item in evidence_refs if item is not None],
        })
    return normalized_handoffs


def _handoff_evidence_lines(handoff: Dict[str, Any]) -> List[str]:
    facts = handoff.get("facts") if isinstance(handoff.get("facts"), dict) else {}
    lines: List[str] = []
    ordered_fact_keys = [
        "app_url",
        "preview_url",
        "cdp_url",
        "artifacts_path",
        "process_log_dir",
        "build_status",
        "install_status",
        "startup_status",
        "task_completion_score",
        "score",
    ]
    for key in ordered_fact_keys:
        value = facts.get(key)
        if value in (None, "", [], {}):
            continue
        lines.append(f"{key}: {_format_fact_value(value)}")

    ports = facts.get("ports")
    if isinstance(ports, list):
        for port in ports:
            if not isinstance(port, dict):
                continue
            port_name = port.get("name") or "port"
            port_value = port.get("url") or port.get("port") or "-"
            lines.append(f"{port_name}: {port_value}")

    for warning in handoff.get("warnings") if isinstance(handoff.get("warnings"), list) else []:
        lines.append(f"warning: {warning}")
    for blocker in handoff.get("blockers") if isinstance(handoff.get("blockers"), list) else []:
        lines.append(f"blocker: {blocker}")
    for ref in handoff.get("evidence_refs") if isinstance(handoff.get("evidence_refs"), list) else []:
        lines.append(f"evidence_ref: {ref}")
    return lines


def _looks_like_path_key(key: str) -> bool:
    normalized = key.strip().lower()
    return any(token in normalized for token in (
        "path",
        "dir",
        "cwd",
        "workspace",
        "session",
        "url",
        "cdp",
        "file",
    ))


def _looks_like_path_value(value: str) -> bool:
    text = value.strip()
    if not text:
        return False
    lowered = text.lower()
    if lowered.startswith(("http://", "https://", "ws://", "wss://")):
        return True
    return any(token in text for token in (
        "/tmp/",
        "/artifacts/",
        "artifacts.json",
        "process_logs",
        "eval_open_",
        "/Users/",
        "/osfs/",
        "localhost:",
    ))


def _collect_path_evidence(value: Any, prefix: str = "") -> List[str]:
    lines: List[str] = []
    if isinstance(value, dict):
        for key, nested in value.items():
            key_text = str(key)
            next_prefix = f"{prefix}.{key_text}" if prefix else key_text
            if isinstance(nested, str) and (_looks_like_path_key(key_text) or _looks_like_path_value(nested)):
                lines.append(f"{next_prefix}: {nested}")
            elif isinstance(nested, (dict, list)):
                lines.extend(_collect_path_evidence(nested, next_prefix))
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            next_prefix = f"{prefix}[{index}]" if prefix else f"[{index}]"
            if isinstance(nested, str) and _looks_like_path_value(nested):
                lines.append(f"{next_prefix}: {nested}")
            elif isinstance(nested, (dict, list)):
                lines.extend(_collect_path_evidence(nested, next_prefix))
    elif isinstance(value, str) and prefix and _looks_like_path_value(value):
        lines.append(f"{prefix}: {value}")
    return lines


def _extract_agent_path_evidence(agent: Dict[str, Any], handoff: Dict[str, Any] | None) -> List[str]:
    lines: List[str] = []
    if handoff:
        lines.extend(_handoff_evidence_lines(handoff))

    trajectory = agent.get("trajectory") if isinstance(agent.get("trajectory"), dict) else {}
    steps = trajectory.get("steps") if isinstance(trajectory.get("steps"), list) else []
    for index, step in enumerate(steps):
        if not isinstance(step, dict):
            continue
        step_prefix = f"trajectory.steps[{index}]"
        args = step.get("args") if isinstance(step.get("args"), dict) else step.get("tool_args")
        if isinstance(args, (dict, list)):
            lines.extend(_collect_path_evidence(args, f"{step_prefix}.args"))
        for field_name in ("result", "observation"):
            field_value = step.get(field_name)
            if isinstance(field_value, str) and _looks_like_path_value(field_value):
                lines.append(f"{step_prefix}.{field_name}: {field_value}")
            elif isinstance(field_value, (dict, list)):
                lines.extend(_collect_path_evidence(field_value, f"{step_prefix}.{field_name}"))

    trajectory_ref = agent.get("trajectory_ref")
    if isinstance(trajectory_ref, str) and trajectory_ref.strip():
        lines.append(f"trajectory_ref: {trajectory_ref}")

    deduped: List[str] = []
    seen: set[str] = set()
    for line in lines:
        normalized = line.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(normalized)
    return deduped


def _augment_agent_with_handoff(agent: Dict[str, Any], handoff: Dict[str, Any] | None) -> Dict[str, Any]:
    augmented = dict(agent)
    if handoff:
        augmented["pipeline_handoff"] = handoff
        augmented.setdefault("pipeline_stage", handoff.get("stage"))
        if not augmented.get("trajectory_ref"):
            evidence_refs = handoff.get("evidence_refs") if isinstance(handoff.get("evidence_refs"), list) else []
            if evidence_refs:
                augmented["trajectory_ref"] = evidence_refs[0]

        dimensions = [dimension for dimension in augmented.get("dimensions", []) if isinstance(dimension, dict)]
        evidence_lines = _handoff_evidence_lines(handoff)
        dimension_payload = {
            "dimension_id": "pipeline_handoff",
            "score": handoff.get("facts", {}).get("score") if isinstance(handoff.get("facts"), dict) else None,
            "verdict": handoff.get("status"),
            "reason": handoff.get("summary") or f"stage={handoff.get('stage') or 'unknown'}",
            "evidence": evidence_lines,
        }
        dimensions = [dimension for dimension in dimensions if dimension.get("dimension_id") != "pipeline_handoff"]
        dimensions.append(dimension_payload)
        augmented["dimensions"] = dimensions

        existing_end_reason = str(augmented.get("end_reason") or "").strip()
        if handoff.get("summary"):
            augmented["end_reason"] = existing_end_reason or str(handoff.get("summary"))

    augmented["path_evidence"] = _extract_agent_path_evidence(augmented, handoff)
    return augmented


def _synthetic_agent_from_handoff(handoff: Dict[str, Any]) -> Dict[str, Any]:
    facts = handoff.get("facts") if isinstance(handoff.get("facts"), dict) else {}
    return _augment_agent_with_handoff(
        {
            "agent_id": handoff.get("agent_id"),
            "status": handoff.get("status"),
            "end_reason": handoff.get("summary"),
            "score": facts.get("score"),
            "task_completion_score": facts.get("task_completion_score"),
            "planned_tasks": [],
            "task_results": [],
            "dimensions": [],
            "trajectory_ref": None,
            "trajectory": {
                "status": handoff.get("status") or "unknown",
                "steps": [],
                "dom_elements": [],
                "initial_diagnostics": {},
                "duration_ms": 0,
            },
        },
        handoff,
    )


def _extract_agents(report: Dict[str, Any]) -> List[Dict[str, Any]]:
    handoffs = _normalize_pipeline_handoffs(report)
    handoff_by_id = {
        str(handoff.get("agent_id") or ""): handoff for handoff in handoffs if handoff.get("agent_id")
    }

    direct = report.get("agents")
    if isinstance(direct, list):
        agents = [agent for agent in direct if isinstance(agent, dict)]
    else:
        fallback = report.get("agent_results")
        agents = [agent for agent in fallback if isinstance(agent, dict)] if isinstance(fallback, list) else []

    if agents:
        augmented_agents: List[Dict[str, Any]] = []
        seen_agent_ids: set[str] = set()
        for agent in agents:
            agent_id = str(agent.get("agent_id") or agent.get("role") or "").strip()
            if agent_id:
                seen_agent_ids.add(agent_id)
            augmented_agents.append(_augment_agent_with_handoff(agent, handoff_by_id.get(agent_id)))
        for handoff in handoffs:
            agent_id = str(handoff.get("agent_id") or "").strip()
            if agent_id and agent_id not in seen_agent_ids:
                augmented_agents.append(_synthetic_agent_from_handoff(handoff))
        return augmented_agents

    return [_synthetic_agent_from_handoff(handoff) for handoff in handoffs]


def _extract_link(row: Dict[str, Any]) -> str | None:
    return _extract_link_from_report(_report_from_row(row), row)


def _extract_link_from_report(report: Dict[str, Any], row: Dict[str, Any] | None = None) -> str | None:
    # Optimized: avoid repeated _report_from_row calls
    candidates = []
    if row:
        candidates.append(row.get("link"))
    candidates.extend([
        report.get("link"),
        report.get("app_link"),
        report.get("preview_url"),
    ])

    artifacts = report.get("artifacts")
    if isinstance(artifacts, dict):
        ports = artifacts.get("ports")
        if isinstance(ports, list):
            for port in ports:
                if isinstance(port, dict) and port.get("url"):
                    candidates.append(port.get("url"))

    for candidate in candidates:
        if candidate:
            return str(candidate)
    return None


def _derive_status(row: Dict[str, Any], report: Dict[str, Any] | None = None) -> str:
    if row.get("error"):
        return "failed"

    explicit = _normalize_status(row.get("status"))
    if report is None:
        report = _report_from_row(row)

    verdict = report.get("verdict")
    if isinstance(verdict, dict):
        verdict_status = _normalize_status(verdict.get("status"))
        if verdict_status:
            return verdict_status
        if isinstance(verdict.get("passed"), bool):
            return "passed" if verdict.get("passed") else "failed"
    else:
        verdict_status = _normalize_status(verdict)
        if verdict_status:
            return verdict_status

    build_result = report.get("build_result")
    if isinstance(build_result, dict):
        build_info = _extract_build_info(build_result)
        if build_info.get("passed") is False:
            return "failed"

    task_total = 0
    task_passed = 0
    has_pass = False
    has_partial = False
    has_failed = False

    for agent in _extract_agents(report):
            if not isinstance(agent, dict):
                continue

            task_results = agent.get("task_results")
            if isinstance(task_results, list):
                for task in task_results:
                    if not isinstance(task, dict):
                        continue
                    task_total += 1
                    if task.get("passed") is True:
                        task_passed += 1
                        has_pass = True
                    elif task.get("passed") is False:
                        has_failed = True
            else:
                tasks = agent.get("tasks")
                if isinstance(tasks, list):
                    for task in tasks:
                        if not isinstance(task, dict):
                            continue
                        task_total += 1
                        if task.get("passed") is True:
                            task_passed += 1
                            has_pass = True
                        elif task.get("passed") is False:
                            has_failed = True

            dimensions = agent.get("dimensions")
            if isinstance(dimensions, list):
                for dim in dimensions:
                    if not isinstance(dim, dict):
                        continue
                    dim_status = _normalize_status(dim.get("verdict"))
                    if dim_status == "passed":
                        has_pass = True
                    elif dim_status == "partial":
                        has_partial = True
                    elif dim_status == "failed":
                        has_failed = True

    if task_total:
        if task_passed == task_total and not has_failed and not has_partial:
            return "passed"
        if task_passed > 0 or has_pass or has_partial:
            return "partial"
        return "failed"

    if has_pass and not has_failed and not has_partial:
        return "passed"
    if has_pass or has_partial:
        return "partial"

    return explicit or "failed"


def _extract_reason(row: Dict[str, Any], report: Dict[str, Any] | None = None) -> str:
    if row.get("error"):
        return str(row.get("error"))

    if report is None:
        report = _report_from_row(row)
    verdict = report.get("verdict")
    if isinstance(verdict, dict) and verdict.get("reason"):
        return str(verdict.get("reason"))

    build_result = report.get("build_result")
    if isinstance(build_result, dict):
        build_info = _extract_build_info(build_result)
        if build_info.get("reason"):
            return str(build_info.get("reason"))

    for agent in _extract_agents(report):
            if not isinstance(agent, dict):
                continue
            task_results = agent.get("task_results")
            if isinstance(task_results, list):
                for task in task_results:
                    if isinstance(task, dict) and task.get("passed") is False and task.get("reason"):
                        return str(task.get("reason"))
            else:
                tasks = agent.get("tasks")
                if isinstance(tasks, list):
                    for task in tasks:
                        if isinstance(task, dict) and task.get("passed") is False and task.get("reason"):
                            return str(task.get("reason"))
            dimensions = agent.get("dimensions")
            if isinstance(dimensions, list):
                for dim in dimensions:
                    if not isinstance(dim, dict):
                        continue
                    if _normalize_status(dim.get("verdict")) in {"failed", "partial"} and dim.get("reason"):
                        return str(dim.get("reason"))
            if agent.get("end_reason"):
                return str(agent.get("end_reason"))

    return ""


def _runtime_paths_from_report(report: Dict[str, Any]) -> Dict[str, str]:
    direct_paths = report.get("runtime_paths") if isinstance(report.get("runtime_paths"), dict) else {}
    runtime_paths = {
        "workspace_root": str(direct_paths.get("workspace_root") or report.get("workspace_root") or "").strip(),
        "artifacts_path": str(direct_paths.get("artifacts_path") or report.get("artifacts_path") or "").strip(),
        "process_log_dir": str(direct_paths.get("process_log_dir") or report.get("process_log_dir") or "").strip(),
    }

    handoffs = report.get("pipeline_handoffs") if isinstance(report.get("pipeline_handoffs"), list) else []
    for handoff in handoffs:
        if not isinstance(handoff, dict):
            continue
        facts = handoff.get("facts") if isinstance(handoff.get("facts"), dict) else {}
        if not runtime_paths["artifacts_path"] and facts.get("artifacts_path"):
            runtime_paths["artifacts_path"] = str(facts.get("artifacts_path"))
        if not runtime_paths["process_log_dir"] and facts.get("process_log_dir"):
            runtime_paths["process_log_dir"] = str(facts.get("process_log_dir"))
        if runtime_paths["artifacts_path"] and runtime_paths["process_log_dir"]:
            break

    if not runtime_paths["workspace_root"] and runtime_paths["artifacts_path"]:
        runtime_paths["workspace_root"] = str(Path(runtime_paths["artifacts_path"]).parent)

    return {key: value for key, value in runtime_paths.items() if value}


def _runtime_paths_text(report: Dict[str, Any]) -> str:
    runtime_paths = _runtime_paths_from_report(report)
    if not runtime_paths:
        return "No runtime paths recorded."

    lines: List[str] = []
    if runtime_paths.get("workspace_root"):
        lines.append(f"Workspace: {runtime_paths['workspace_root']}")
    if runtime_paths.get("artifacts_path"):
        lines.append(f"Artifacts JSON: {runtime_paths['artifacts_path']}")
    if runtime_paths.get("process_log_dir"):
        lines.append(f"Process Logs: {runtime_paths['process_log_dir']}")
    return "\n".join(lines) if lines else "No runtime paths recorded."


def _parse_path_evidence_line(line: str) -> Dict[str, str] | None:
    text = str(line or "").strip()
    if not text:
        return None
    key, separator, value = text.partition(": ")
    if not separator:
        key, value = "detail", text
    lower_key = key.lower()
    lower_value = value.lower()
    group = "Runtime"
    source = "runtime"
    label = "detail"
    if lower_key.startswith("trajectory.steps["):
        group = "Trajectory"
        source = "trajectory"
    elif lower_key == "trajectory_ref":
        group = "Reference"
        source = "reference"
    elif lower_key == "evidence_ref":
        group = "Reference"
        source = "handoff"
    elif "url" in lower_key or lower_value.startswith(("http://", "https://", "ws://", "wss://")):
        group = "URLs"
        source = "trajectory" if lower_key.startswith("trajectory") else "runtime"
    elif any(token in lower_key for token in ("artifacts", "process_log_dir", "workspace_root", "path", "dir")):
        group = "Runtime"
        source = "trajectory" if lower_key.startswith("trajectory") else "runtime"
    if lower_key == "workspace_root":
        label = "workspace"
    elif lower_key == "artifacts_path":
        label = "artifacts json"
    elif lower_key == "process_log_dir":
        label = "process logs"
    elif lower_key == "app_url":
        label = "app url"
    elif lower_key == "preview_url":
        label = "preview url"
    elif lower_key == "cdp_url":
        label = "cdp url"
    elif lower_key == "trajectory_ref":
        label = "trajectory ref"
    elif lower_key == "evidence_ref":
        label = "evidence ref"
    elif lower_key.startswith("trajectory.steps["):
        match = re.match(r"^trajectory\.steps\[(\d+)\]\.(.+)$", key)
        if match:
            step_index = int(match.group(1)) + 1
            tail = match.group(2)
            tail = tail.removeprefix("args.").removeprefix("observation.").removeprefix("result.")
            tail_label = tail.split(".")[-1].split("[")[0].replace("_", " ")
            label = f"step {step_index} · {tail_label}"
        else:
            label = "trajectory"
    else:
        label = key.split(".")[-1].split("[")[0].replace("_", " ")
    return {
        "group": group,
        "source": source,
        "key": key,
        "label": label,
        "value": value,
    }


def _render_path_value(value: str) -> str:
    escaped = html.escape(str(value))
    if str(value).lower().startswith(("http://", "https://", "ws://", "wss://")):
        return f'<a href="{escaped}" target="_blank" rel="noopener">{escaped}</a>'
    return escaped


def _render_agent_paths_initial(path_evidence: List[str]) -> str:
    entries = [entry for entry in (_parse_path_evidence_line(line) for line in path_evidence) if entry]
    if not entries:
        return ""

    source_counts: Dict[str, int] = {}
    for entry in entries:
        source = entry["source"]
        source_counts[source] = source_counts.get(source, 0) + 1

    summary_parts = [f'<span class="path-chip">{len(entries)} entries</span>']
    summary_parts.extend(
        f'<span class="path-chip">{html.escape(source)} {count}</span>'
        for source, count in source_counts.items()
    )

    groups_html: List[str] = []
    for group_name in ("Runtime", "URLs", "Reference", "Trajectory"):
        group_items = [entry for entry in entries if entry["group"] == group_name]
        if not group_items:
            continue
        rows = "".join(
            '<div class="path-row">'
            f'<div class="path-key">{html.escape(item["label"] or item["key"])}</div>'
            f'<div class="path-value">{_render_path_value(item["value"])}</div>'
            '</div>'
            for item in group_items
        )
        groups_html.append(
            '<div class="path-group">'
            f'<div class="path-group-title">{html.escape(group_name)}</div>'
            f'{rows}'
            '</div>'
        )

    return (
        '<div class="section agent-paths"><div class="label">Agent Paths</div>'
        '<div class="path-panel">'
        f'<div class="path-summary">{"".join(summary_parts)}</div>'
        f'<div class="path-groups">{"".join(groups_html)}</div>'
        '</div></div>'
    )


def _render_pipeline_handoffs_initial(report: Dict[str, Any]) -> str:
    handoffs = _normalize_pipeline_handoffs(report)
    if not handoffs:
        return ""

    parts: List[str] = ['<div class="section"><div class="label">Pipeline Flow</div><div class="flow">']
    for index, handoff in enumerate(handoffs):
        if index:
            parts.append('<div class="flow-arrow">&rarr;</div>')
        facts = handoff.get("facts") if isinstance(handoff.get("facts"), dict) else {}
        subtitle_parts = [f"stage {handoff.get('stage') or '-'}"]
        if facts.get("artifacts_path"):
            subtitle_parts.append(f"artifacts {facts.get('artifacts_path')}")
        if facts.get("process_log_dir"):
            subtitle_parts.append(f"logs {facts.get('process_log_dir')}")
        parts.append(
            '<div class="flow-node">'
            f'<div class="flow-node-top"><span class="flow-node-id">{html.escape(str(handoff.get("agent_id") or "unknown"))}</span>{_initial_status_badge(handoff.get("status"))}</div>'
            f'<div class="flow-node-summary">{html.escape(str(handoff.get("summary") or "No summary recorded."))}</div>'
            f'<div class="flow-node-meta">{html.escape(" | ".join(subtitle_parts))}</div>'
            '</div>'
        )
    parts.append('</div></div>')
    return ''.join(parts)


def _extract_model(row: Dict[str, Any], report: Dict[str, Any] | None = None) -> str:
    model = row.get("model")
    if isinstance(model, str) and model.strip():
        return model.strip()
    if report is None:
        report = _report_from_row(row)
    meta = report.get("_meta") if isinstance(report.get("_meta"), dict) else {}
    model = meta.get("model") or report.get("model")
    if isinstance(model, str) and model.strip():
        return model.strip()
    agents = report.get("agents") if isinstance(report.get("agents"), list) else []
    for agent in agents:
        if isinstance(agent, dict):
            agent_model = agent.get("model")
            if isinstance(agent_model, str) and agent_model.strip():
                return agent_model.strip()
    return ""


def _preview(row: Dict[str, Any], idx: int, report: Dict[str, Any] | None = None) -> Dict[str, Any]:
    # Avoid repeated _report_from_row calls - pass pre-computed report if available
    if report is None:
        report = _report_from_row(row)
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    link = _extract_link_from_report(report)
    runtime_paths = _runtime_paths_from_report(report)
    return {
        "index": idx,
        "id": row.get("id") or row.get("sample_id") or f"row-{idx}",
        "status": _derive_status(row, report),
        "query": row.get("query"),
        "model": _extract_model(row, report),
        "reason": _extract_reason(row, report),
        "error": row.get("error"),
        "link": link,
        "has_preview": bool(link),
        "agents_total": summary.get("agents_total"),
        "agents_completed": summary.get("agents_completed"),
        "overall_score": summary.get("overall_score"),
        "overall_max_score": summary.get("overall_max_score"),
        "avg_task_completion_score": summary.get("avg_task_completion_score"),
        "avg_main_task_pass_rate": summary.get("avg_main_task_pass_rate"),
        "process_log_dir": runtime_paths.get("process_log_dir"),
        "has_runtime_paths": bool(runtime_paths),
    }


def _normalized_summary(rows: List[Dict[str, Any]]) -> Dict[str, int]:
    summary = {"total": len(rows), "passed": 0, "partial": 0, "failed": 0}
    for row in rows:
        status = _derive_status(row)
        if status == "passed":
            summary["passed"] += 1
        elif status == "partial":
            summary["partial"] += 1
        else:
            summary["failed"] += 1
    return summary


def _audit_summary(entries: List[Dict[str, Any]]) -> Dict[str, int]:
    summary = {"total": len(entries), "auto_approve": 0, "require_confirm": 0, "other": 0}
    for entry in entries:
        risk = str(entry.get("risk") or "").strip().lower()
        if risk == "auto_approve":
            summary["auto_approve"] += 1
        elif risk == "require_confirm":
            summary["require_confirm"] += 1
        else:
            summary["other"] += 1
    return summary


def _stats_payload(rows: List[Dict[str, Any]], audit_entries: List[Dict[str, Any]]) -> Dict[str, int]:
    summary = _normalized_summary(rows)
    return {
        **summary,
        "completed": summary["total"],
        "running": 0,
        "pending": 0,
        "failed_verdict": summary["failed"],
        "audit_total": len(audit_entries),
    }


def _initial_status_badge(status: Any) -> str:
    normalized = _normalize_status(status) or "failed"
    badge_class = "ok" if normalized == "passed" else "warn" if normalized == "partial" else "bad"
    return f'<span class="badge {badge_class}">{html.escape(normalized)}</span>'


def _initial_score(value: Any, max_value: Any = None) -> str:
    if isinstance(value, (int, float)):
        if isinstance(max_value, (int, float)):
            max_text = f"{max_value:.0f}" if float(max_value).is_integer() else f"{max_value:.2f}"
            return f"{value:.2f}/{max_text}"
        return f"{value:.2f}"
    return "-"


def _summary_score_value(summary: Dict[str, Any]) -> Any:
    if isinstance(summary.get("overall_score"), (int, float)):
        return summary.get("overall_score")
    if isinstance(summary.get("avg_task_completion_score"), (int, float)):
        return summary.get("avg_task_completion_score")
    return summary.get("avg_main_task_pass_rate")


def _summary_score_max(summary: Dict[str, Any]) -> Any:
    return summary.get("overall_max_score")


def _summary_score_label(summary: Dict[str, Any]) -> str:
    if isinstance(summary.get("overall_score"), (int, float)):
        return "total score"
    return "score"


def _render_initial_tasks(previews: List[Dict[str, Any]]) -> str:
    if not previews:
        return '<div class="empty">No tasks found in this report.</div>'

    parts: List[str] = []
    for task in previews:
        task_id = html.escape(str(task.get("id") or "-"))
        query = html.escape(str(task.get("query") or task.get("reason") or "No query provided."))
        agents_completed = html.escape(str(task.get("agents_completed") if task.get("agents_completed") is not None else "-"))
        agents_total = html.escape(str(task.get("agents_total") if task.get("agents_total") is not None else "-"))
        preview_flag = '<span>preview</span>' if task.get("has_preview") else ""
        score_value = task.get("overall_score") if isinstance(task.get("overall_score"), (int, float)) else (task.get("avg_task_completion_score") if isinstance(task.get("avg_task_completion_score"), (int, float)) else task.get("avg_main_task_pass_rate"))
        score_max = task.get("overall_max_score")
        score_label = "total score" if isinstance(task.get("overall_score"), (int, float)) else "score"
        parts.append(
            """
        <div class="task-item" data-index="{index}" onclick="window.__openSelectTask && window.__openSelectTask({index})">
      <div class="task-top"><div class="task-id">{task_id}</div>{badge}</div>
      <div class="task-query">{query}</div>
      <div class="task-meta">
        <span>agents {agents_completed}/{agents_total}</span>
                <span>{task_label} {task_pct}</span>
        {preview_flag}
      </div>
    </div>""".format(
                index=task.get("index", 0),
                task_id=task_id,
                badge=_initial_status_badge(task.get("status")),
                query=query,
                agents_completed=agents_completed,
                agents_total=agents_total,
                task_pct=html.escape(_initial_score(score_value, score_max)),
                task_label=html.escape(score_label),
                preview_flag=preview_flag,
            )
        )
    return "".join(parts)


def _render_initial_details(row: Dict[str, Any], index: int) -> str:
    report = _report_from_row(row)
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    artifacts = report.get("artifacts") if isinstance(report.get("artifacts"), dict) else {}
    build_result = report.get("build_result") if isinstance(report.get("build_result"), dict) else {}
    build_info = _extract_build_info(build_result)
    link = _extract_link(row)
    runtime_paths_text = _runtime_paths_text(report)

    chips: List[str] = []
    if isinstance(summary.get("agents_total"), (int, float)):
        chips.append(f"agents {summary.get('agents_completed', '-')} / {summary.get('agents_total', '-')}")
    score_value = _summary_score_value(summary)
    if isinstance(score_value, (int, float)):
        chips.append(f"{_summary_score_label(summary)} {_initial_score(score_value, _summary_score_max(summary))}")
    runtime_paths = _runtime_paths_from_report(report)
    if runtime_paths.get("process_log_dir"):
        chips.append(f"logs {runtime_paths['process_log_dir']}")

    ports = artifacts.get("ports") if isinstance(artifacts.get("ports"), list) else []
    port_lines: List[str] = []
    for port in ports:
        if isinstance(port, dict):
            port_lines.append(f"{port.get('name') or 'app'}: {port.get('url') or port.get('port') or '-'}")
    port_text = "\n".join(port_lines) if port_lines else "No declared ports."

    build_status = build_info.get("status")
    if not build_status:
        build_status = "failed" if build_info.get("passed") is False else "passed"

    chips_html = "".join(
        f'<span class="chip">{html.escape(str(chip))}</span>' for chip in chips
    ) or '<span class="chip">No summary metadata</span>'
    link_html = (
        f'<a class="link" href="{html.escape(str(link))}" target="_blank" rel="noopener">{html.escape(str(link))}</a>'
        if link
        else 'Unavailable'
    )
    agents_html = _render_initial_agents(report, index)
    pipeline_html = _render_pipeline_handoffs_initial(report)

    return """<div class="section"><div class="label">Task</div><div class="value mono">{task_id}</div></div>
    <div class="section"><div class="label">Status</div><div class="value">{status_badge}</div></div>
    <div class="section"><div class="label">Query</div><div class="quote">{query}</div></div>
    <div class="section"><div class="label">Reason</div><div class="quote">{reason}</div></div>
    <div class="section"><div class="label">Build</div><div class="value">{build_badge}</div><div class="quote">{build_reason}</div></div>
    {pipeline_html}
    <div class="section"><div class="label">Runtime Metadata</div><div class="chips">{chips}</div></div>
    <div class="section"><div class="label">Runtime Paths</div><div class="quote code">{runtime_paths}</div></div>
    <div class="section"><div class="label">Preview Link</div><div class="value">{link}</div></div>
    <div class="section"><div class="label">Artifacts</div><div class="quote">{artifacts}\nCDP: {cdp}</div></div>
    <div class="section"><div class="label">Agents</div></div>{agents_html}""".format(
        task_id=html.escape(str(row.get("id") or row.get("sample_id") or f"row-{index}")),
        status_badge=_initial_status_badge(_derive_status(row)),
        query=html.escape(str(row.get("query") or "No query provided.")),
        reason=html.escape(_extract_reason(row) or "No explicit reason recorded."),
        build_badge=_initial_status_badge(build_status),
        build_reason=html.escape(str(build_info.get("reason") or "No build verdict available.")),
        pipeline_html=pipeline_html,
        chips=chips_html,
        runtime_paths=html.escape(runtime_paths_text),
        link=link_html,
        artifacts=html.escape(port_text),
        cdp=html.escape(str(artifacts.get("cdp_url") or "-")),
        agents_html=agents_html,
    )


def _render_traj_step_initial(step: Dict[str, Any], step_index: int, agent_index: int, task_index: int) -> str:
    tool_name = html.escape(str(step.get("tool_name") or "unknown"))
    success = step.get("success")
    if success is True:
        success_badge = '<span class="ts-success ok">✓</span>'
    elif success is False:
        success_badge = '<span class="ts-success fail">✗</span>'
    else:
        success_badge = ""
    task_label = ""
    task_title = step.get("task_title") or step.get("task_id")
    if task_title:
        task_label = f'<span class="ts-task">{html.escape(str(task_title))}</span>'
    ts = step.get("timestamp")
    time_badge = f'<span class="ts-time">{html.escape(str(ts))}</span>' if ts else ""
    args = step.get("args") or step.get("tool_args") or {}
    args_html = ""
    if args and isinstance(args, dict) and args:
        args_id = f"ssr-args-{task_index}-{agent_index}-{step_index}"
        args_str = html.escape(json.dumps(args, indent=2, ensure_ascii=False))
        args_html = (
            f'<div class="ts-args-toggle" onclick="document.getElementById(\'{args_id}\').classList.toggle(\'collapsed\')">▼ args</div>'
            f'<pre id="{args_id}" class="ts-args collapsed">{args_str}</pre>'
        )
    result_val = step.get("result") or step.get("observation")
    result_html = ""
    if result_val:
        result_html = f'<div class="ts-result"><b>Result:</b> {html.escape(str(result_val))}</div>'
    screenshot = step.get("screenshot")
    screenshot_path = step.get("screenshot_path")
    screenshot_html = ""
    if screenshot_path:
        src = f"/api/screenshot?path={quote(str(screenshot_path), safe='')}"
        escaped_src = html.escape(src)
        screenshot_html = (
            f'<div class="ts-screenshot">'
            f'<img src="{escaped_src}" alt="step {step_index + 1} screenshot" '
            f'onclick="window.open(this.src)" title="Click to open full size" />'
            f'</div>'
        )
    elif screenshot:
        src = str(screenshot)
        if not src.startswith("data:") and not src.startswith("http"):
            src = f"data:image/png;base64,{src}"
        escaped_src = html.escape(src)
        screenshot_html = (
            f'<div class="ts-screenshot">'
            f'<img src="{escaped_src}" alt="step {step_index + 1} screenshot" '
            f'onclick="window.open(this.src)" title="Click to open full size" />'
            f'</div>'
        )
    return (
        f'<div class="traj-step">'
        f'<div class="ts-hdr">'
        f'<span class="ts-idx">#{step_index + 1}</span>'
        f'<span class="ts-tool">{tool_name}</span>'
        f'{success_badge}'
        f'{task_label}'
        f'{time_badge}'
        f'</div>'
        f'{args_html}'
        f'{result_html}'
        f'{screenshot_html}'
        f'</div>'
    )


def _render_initial_agents(report: Dict[str, Any], task_index: int = 0) -> str:
    agents = _extract_agents(report)
    if not agents:
        return '<div class="value" style="font-size:13px;color:var(--muted)">No agent report available.</div>'

    parts: List[str] = []
    for agent_index, agent in enumerate(agents):
        agent_id = html.escape(str(agent.get("agent_id") or agent.get("role") or "unknown"))
        status = html.escape(str(agent.get("status") or "unknown"))
        score_raw = agent.get("score")
        score = html.escape(str(round(score_raw, 1)) if isinstance(score_raw, float) else (str(score_raw) if score_raw is not None else "-"))
        end_reason = html.escape(str(agent.get("end_reason") or ""))
        end_label = f" · {end_reason}" if end_reason else ""

        tasks = agent.get("tasks") if isinstance(agent.get("tasks"), list) else []
        task_step_count = 0
        for task in tasks:
            if not isinstance(task, dict):
                continue
            task_trajectory = task.get("trajectory") if isinstance(task.get("trajectory"), dict) else {}
            task_steps = task_trajectory.get("steps") if isinstance(task_trajectory.get("steps"), list) else []
            task_step_count += len([s for s in task_steps if isinstance(s, dict)])
        trajectory = agent.get("trajectory") if isinstance(agent.get("trajectory"), dict) else {}
        traj_steps = trajectory.get("steps") if isinstance(trajectory.get("steps"), list) else []
        step_count = task_step_count or len([s for s in traj_steps if isinstance(s, dict)])
        task_count = len([task for task in tasks if isinstance(task, dict)])
        step_count_label = f"{task_count} tasks" if task_count else (f"{step_count} steps" if step_count else "no task")

        # Dimensions as collapsible accordions
        dimensions = agent.get("dimensions") if isinstance(agent.get("dimensions"), list) else []
        dims_html = ""
        if dimensions:
            acc_items = ""
            for di, dim in enumerate(dimensions):
                if not isinstance(dim, dict):
                    continue
                dim_id_val = html.escape(str(dim.get("dimension_id") or "dim"))
                dim_score_raw = dim.get("score")
                dim_score_str = f"{round(dim_score_raw, 1)}" if isinstance(dim_score_raw, float) else (str(dim_score_raw) if dim_score_raw is not None else "-")
                dim_verdict = html.escape(str(dim.get("verdict") or ""))
                dim_reason = html.escape(str(dim.get("reason") or "-"))
                acc_id = f"ssr-dim-{task_index}-{agent_index}-{di}"
                acc_items += (
                    f'<div class="dim-acc" id="{acc_id}">'
                    f'<div class="dim-acc-hdr" onclick="event.stopPropagation();document.getElementById(\'{acc_id}\').classList.toggle(\'open\');">'
                    f'<span class="dim-id">[{dim_id_val}]</span>'
                    f'<span class="dim-score">{html.escape(dim_score_str)}{f" · {dim_verdict}" if dim_verdict else ""}</span>'
                    f'<span class="dim-expand">▶</span>'
                    f'</div>'
                    f'<div class="dim-acc-body"><div class="dim-reason">{dim_reason}</div></div>'
                    f'</div>'
                )
            dims_html = f'<div style="margin-top:4px"><div class="label" style="margin-bottom:4px">Dimensions</div>{acc_items}</div>'
        else:
            dims_html = '<div class="value" style="font-size:12px;color:var(--muted)">No dimension results.</div>'

        ag_ov_id = f"ag-ov-{task_index}-{agent_index}"
        onclick = f"selectAgent({task_index},{agent_index})"
        parts.append(
            f'<div class="ag-ov-card" id="{ag_ov_id}" onclick="{onclick}">'
            f'<div class="ag-ov-hdr">'
            f'<div>'
            f'<div class="ag-id">{agent_id}</div>'
            f'<div class="ag-role">status {status}{end_label}</div>'
            f'</div>'
            f'<div style="display:flex;align-items:center;gap:8px">'
            f'{_initial_status_badge(agent.get("status"))}'
            f'<span class="ag-score">{score}</span>'
            f'<span class="ag-expand-btn" style="font-size:11px">{step_count_label}</span>'
            f'</div>'
            f'</div>'
            f'<div class="ag-ov-body">'
            f'{dims_html}'
            f'</div>'
            f'</div>'
        )
    return ''.join(parts)


def _render_initial_preview(row: Dict[str, Any]) -> str:
    link = _extract_link(row)
    if not link:
        return '<div class="empty">This task does not expose a preview URL.</div>'
    escaped_link = html.escape(str(link))
    return f'''<div class="preview-wrap">
    <div class="preview-toolbar">
      <div class="preview-url">{escaped_link}</div>
      <a class="preview-button" href="{escaped_link}" target="_blank" rel="noopener">Open</a>
    </div>
    <iframe src="{escaped_link}" loading="lazy" referrerpolicy="no-referrer"></iframe>
  </div>'''


_DB_SCHEMA_VERSION = "3"  # bump when raw_json storage format changes (e.g., TEXT -> gzip BLOB -> separate files)


def _ensure_db(report_path: Path) -> str:
    """Ensure SQLite DB exists, creating it from the JSONL if needed.

    If the DB already exists it is used as-is — no rebuild from source,
    even on errors.  Returns db_path as a string.
    """
    db_path = Path(str(report_path) + ".db")
    if db_path.exists():
        logger.info("Using existing DB: %s", db_path)
        return str(db_path)

    if not report_path.exists():
        logger.warning("Report not found: %s", report_path)
        return str(db_path)

    logger.info("Importing JSONL report to SQLite (one-time, may take a while for large files)...")
    _import_jsonl_to_db(report_path, db_path)
    logger.info("SQLite import complete: %s", db_path)
    return str(db_path)


def _import_jsonl_to_db(jsonl_path: Path, db_path: Path) -> None:
    """Read a JSONL report line by line and build the SQLite database."""
    import gzip
    import sqlite3
    import time

    db_path.parent.mkdir(parents=True, exist_ok=True)

    # Remove stale DB files and per-task raw_json directory
    for suffix in ("", "-wal", "-shm"):
        p = Path(str(db_path) + suffix)
        if p.exists():
            p.unlink()
    raw_json_dir = Path(str(db_path) + ".tasks")
    if raw_json_dir.exists():
        import shutil
        shutil.rmtree(raw_json_dir)
    raw_json_dir.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=OFF")
    conn.execute("PRAGMA cache_size=-200000")
    conn.execute("PRAGMA mmap_size=1073741824")
    conn.execute("PRAGMA busy_timeout=30000")

    conn.executescript("""
        CREATE TABLE tasks (
            idx INTEGER PRIMARY KEY,
            id TEXT,
            status TEXT,
            query TEXT,
            model TEXT DEFAULT '',
            reason TEXT,
            error TEXT,
            link TEXT,
            has_preview INTEGER,
            agents_total REAL,
            agents_completed REAL,
            overall_score REAL,
            overall_max_score REAL,
            avg_task_completion_score REAL,
            avg_main_task_pass_rate REAL,
            process_log_dir TEXT,
            has_runtime_paths INTEGER,
            raw_json_path TEXT
        );
        CREATE TABLE meta (
            key TEXT PRIMARY KEY,
            value TEXT
        );
    """)

    import_start = time.monotonic()
    last_log_at = import_start
    total = 0
    COMMIT_INTERVAL = 100
    LOG_INTERVAL = 3.0
    GZIP_BATCH_SIZE = 50
    gzip_batch: list[tuple[int, str]] = []
    conn.execute("BEGIN TRANSACTION")

    with open(jsonl_path, "r", encoding="utf-8") as fh:
        for line_no, raw in enumerate(fh, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError:
                logger.warning("Skipping invalid JSONL line %d", line_no)
                continue
            if not isinstance(obj, dict):
                continue

            report = _report_from_row(obj)
            preview = _preview(obj, total, report)

            obj_str = raw
            gzip_batch.append((total, obj_str))
            raw_json_file = str(raw_json_dir / f"{total}.json.gz")
            if len(gzip_batch) >= GZIP_BATCH_SIZE:
                for batch_idx, batch_obj_str in gzip_batch:
                    batch_path = raw_json_dir / f"{batch_idx}.json.gz"
                    with gzip.open(batch_path, "wt", encoding="utf-8") as gf:
                        gf.write(batch_obj_str)
                gzip_batch.clear()

            conn.execute(
                "INSERT INTO tasks VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    total,
                    preview.get("id"),
                    preview.get("status"),
                    preview.get("query"),
                    preview.get("model"),
                    preview.get("reason"),
                    preview.get("error"),
                    preview.get("link"),
                    1 if preview.get("has_preview") else 0,
                    preview.get("agents_total"),
                    preview.get("agents_completed"),
                    preview.get("overall_score"),
                    preview.get("overall_max_score"),
                    preview.get("avg_task_completion_score"),
                    preview.get("avg_main_task_pass_rate"),
                    preview.get("process_log_dir"),
                    1 if preview.get("has_runtime_paths") else 0,
                    str(raw_json_file),
                ),
            )
            total += 1

            if total - (total // COMMIT_INTERVAL) * COMMIT_INTERVAL == 0:
                conn.execute("COMMIT")
                conn.execute("BEGIN TRANSACTION")

            now = time.monotonic()
            if now - last_log_at >= LOG_INTERVAL:
                elapsed = now - import_start
                rate = total / max(elapsed, 0.001)
                logger.info("  %d tasks | %.0f tasks/s", total, rate)
                last_log_at = now

    # Flush remaining gzip batch
    if gzip_batch:
        for batch_idx, batch_obj_str in gzip_batch:
            batch_path = raw_json_dir / f"{batch_idx}.json.gz"
            with gzip.open(batch_path, "wt", encoding="utf-8") as gf:
                gf.write(batch_obj_str)
        gzip_batch.clear()

    conn.execute("COMMIT")

    # Write metadata
    if jsonl_path.exists():
        conn.execute(
            "INSERT OR REPLACE INTO meta VALUES ('source_mtime', ?)",
            (str(jsonl_path.stat().st_mtime),),
        )
    conn.execute("INSERT OR REPLACE INTO meta VALUES ('total_tasks', ?)", (str(total),))
    conn.execute("INSERT OR REPLACE INTO meta VALUES ('report_path', ?)", (str(jsonl_path),))
    conn.execute("INSERT OR REPLACE INTO meta VALUES ('schema_version', ?)", (_DB_SCHEMA_VERSION,))
    conn.commit()
    conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    conn.close()

    elapsed = time.monotonic() - import_start
    logger.info("Imported %d tasks to %s (%.1f s)", total, db_path, elapsed)


def create_app(report_path: Path, audit_path: Path | None = None, page_size: int = 20) -> FastAPI:
    import sqlite3

    db_path_str = _ensure_db(report_path)

    app = FastAPI(title="Open Evaluation Dashboard", version="2.0.0")
    repo_root = Path(__file__).resolve().parents[1]

    def _db() -> sqlite3.Connection:
        conn = sqlite3.connect(db_path_str)
        conn.row_factory = sqlite3.Row
        return conn

    def _db_summary() -> Dict[str, int]:
        conn = _db()
        row = conn.execute(
            "SELECT COUNT(*) as total,"
            " SUM(CASE WHEN status='passed' THEN 1 ELSE 0 END) as passed,"
            " SUM(CASE WHEN status='partial' THEN 1 ELSE 0 END) as partial,"
            " SUM(CASE WHEN status='failed' THEN 1 ELSE 0 END) as failed"
            " FROM tasks"
        ).fetchone()
        conn.close()
        return {
            "total": row["total"] or 0,
            "passed": row["passed"] or 0,
            "partial": row["partial"] or 0,
            "failed": row["failed"] or 0,
        }

    def _resolve_screenshot_path(path_value: str) -> Path:
        candidate = Path(path_value).expanduser().resolve()
        if not candidate.exists() or not candidate.is_file():
            raise HTTPException(status_code=404, detail="screenshot not found")
        if candidate.suffix.lower() != ".png":
            raise HTTPException(status_code=400, detail="only .png screenshots are supported")
        if not candidate.is_relative_to(repo_root):
            raise HTTPException(status_code=403, detail="screenshot path outside workspace root")
        return candidate

    @app.get("/api/tasks")
    async def api_tasks(
        offset: int = Query(0, ge=0),
        limit: int | None = Query(None, ge=1),
        sort_by: str = Query("index", regex="^(index|score)$"),
        sort_dir: str = Query("asc", regex="^(asc|desc)$"),
        filter_model: str | None = Query(None),
        filter_score_min: float | None = Query(None, ge=0),
        filter_score_max: float | None = Query(None, ge=0),
        filter_query: str | None = Query(None),
    ) -> Dict[str, Any]:
        try:
            audit_entries = _read_audit(audit_path)
            conn = _db()

            where_clauses: List[str] = []
            where_params: List[Any] = []
            if filter_query:
                where_clauses.append("query LIKE ?")
                where_params.append(f"%{filter_query}%")
            if filter_model:
                where_clauses.append("model=?")
                where_params.append(filter_model)
            if filter_score_min is not None:
                where_clauses.append("COALESCE(overall_score, avg_task_completion_score, avg_main_task_pass_rate, -1) >= ?")
                where_params.append(filter_score_min)
            if filter_score_max is not None:
                where_clauses.append("COALESCE(overall_score, avg_task_completion_score, avg_main_task_pass_rate, -1) <= ?")
                where_params.append(filter_score_max)

            where_sql = (" WHERE " + " AND ".join(where_clauses)) if where_clauses else ""

            score_expr = "COALESCE(overall_score, avg_task_completion_score, avg_main_task_pass_rate, -1)"
            order_col = score_expr if sort_by == "score" else "idx"
            order_sql = f" ORDER BY {order_col} {sort_dir.upper()}, idx"

            count_row = conn.execute(f"SELECT COUNT(*) as cnt FROM tasks{where_sql}", where_params).fetchone()
            total = count_row["cnt"]

            select_cols = (
                "idx,id,status,query,model,reason,error,link,has_preview,"
                "agents_total,agents_completed,overall_score,overall_max_score,"
                "avg_task_completion_score,avg_main_task_pass_rate,process_log_dir,has_runtime_paths"
            )
            if limit is not None:
                rows = conn.execute(
                    f"SELECT {select_cols} FROM tasks{where_sql}{order_sql} LIMIT ? OFFSET ?",
                    (*where_params, limit, offset),
                ).fetchall()
            else:
                rows = conn.execute(
                    f"SELECT {select_cols} FROM tasks{where_sql}{order_sql}"
                ).fetchall()
            conn.close()
            tasks = [dict(r) for r in rows]
            for t in tasks:
                t["index"] = t.pop("idx")
                t["has_preview"] = bool(t.get("has_preview"))
                t["has_runtime_paths"] = bool(t.get("has_runtime_paths"))
            summary = _db_summary()
            return {
                "tasks": tasks,
                "total": total,
                "summary": summary,
                "audit_summary": _audit_summary(audit_entries),
                "meta": {
                    "report_path": str(report_path),
                    "exists": report_path.exists(),
                    "results_count": total,
                    "audit_path": str(audit_path) if audit_path else None,
                },
            }
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

    @app.get("/api/models")
    async def api_models() -> Dict[str, Any]:
        try:
            conn = _db()
            rows = conn.execute(
                "SELECT DISTINCT model FROM tasks WHERE model IS NOT NULL AND model != '' ORDER BY model"
            ).fetchall()
            conn.close()
            models = [r["model"] for r in rows]
            return {"models": models}
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

    @app.get("/api/stats")
    async def api_stats() -> Dict[str, int]:
        try:
            summary = _db_summary()
            audit_entries = _read_audit(audit_path)
            return {
                **summary,
                "completed": summary["total"],
                "running": 0,
                "pending": 0,
                "failed_verdict": summary["failed"],
                "audit_total": len(audit_entries),
            }
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

    @app.get("/api/tasks/{index}")
    async def api_task(index: int) -> Dict[str, Any]:
        try:
            conn = _db()
            row = conn.execute("SELECT * FROM tasks WHERE idx=?", (index,)).fetchone()
            if row is None:
                conn.close()
                raise HTTPException(status_code=404, detail="task index out of range")
            raw_json = row["raw_json_path"]
            conn.close()
            # raw_json is stored in a separate gzip file
            import gzip as _gz
            raw_json_path = Path(raw_json)
            if raw_json_path.exists():
                with _gz.open(raw_json_path, "rt", encoding="utf-8") as gf:
                    raw_json = gf.read()
            else:
                raw_json = "{}"
            task_obj = json.loads(raw_json)
            report = _report_from_row(task_obj)
            report = dict(report)
            report["pipeline_handoffs"] = _normalize_pipeline_handoffs(report)
            report["agents"] = _extract_agents(report)
            report["runtime_paths"] = _runtime_paths_from_report(report)
            build_result = report.get("build_result") if isinstance(report.get("build_result"), dict) else {}
            build_info = _extract_build_info(build_result)
            _build_status = build_info.get("status")
            if not _build_status:
                _build_status = "failed" if build_info.get("passed") is False else "passed"
            return {
                "index": index,
                "row": task_obj,
                "report": report,
                "link": _extract_link(task_obj),
                "reason": _extract_reason(task_obj),
                "derived_status": _derive_status(task_obj),
                "_build_status": _build_status,
                "_build_reason": str(build_info.get("reason") or "No build verdict available."),
            }
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

    @app.get("/api/audit")
    async def api_audit() -> List[Dict[str, Any]]:
        try:
            return _read_audit(audit_path)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

    @app.get("/api/agents-config")
    async def api_agents_config() -> Dict[str, Any]:
        agents_dir = repo_root / "agents"
        configs: Dict[str, Dict[str, Any]] = {}
        if agents_dir.is_dir():
            for p in sorted(agents_dir.glob("*.json")):
                try:
                    data = json.loads(p.read_text(encoding="utf-8"))
                    agent_id = str(data.get("id", "") or p.stem)
                    configs[agent_id] = {
                        "id": agent_id,
                        "name": str(data.get("name", agent_id)),
                        "role": str(data.get("role", "evaluator")),
                        "stage": str(data.get("stage", "")),
                        "enabled": bool(data.get("enabled", True)),
                        "system_prompt": str(data.get("system_prompt", "")),
                        "allowed_tools": [
                            str(t) for t in (data.get("allowed_tools") or [])
                        ],
                    }
                except Exception:
                    pass
        return {"agents": configs}

    @app.get("/api/screenshot")
    async def api_screenshot(path: str = Query(..., description="Absolute screenshot file path")) -> FileResponse:
        try:
            resolved = _resolve_screenshot_path(path)
            return FileResponse(resolved, media_type="image/png")
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

    @app.get("/", response_class=HTMLResponse)
    async def root() -> HTMLResponse:
        summary = _db_summary()
        results_count = summary["total"]
        audit_entries = _read_audit(audit_path)
        audit_summary = _audit_summary(audit_entries)
        report_exists = report_path.exists()
        report_path_text = str(report_path)
        initial_meta = (
            f"report {'ok' if report_exists else 'missing'} | rows {results_count} | {report_path_text} | "
            f"audit {str(audit_path) if audit_path else 'disabled'}"
        )
        initial_error = ""
        if not report_exists:
            initial_error = f"report file not found: {report_path_text}"
        elif results_count == 0:
            initial_error = f"report loaded but contains 0 results: {report_path_text}"
        initial_tasks = '<div class="empty">Loading tasks...</div>'
        initial_agents = '<div class="empty">Select a task to inspect agent results.</div>'
        initial_trajectory = '<div class="empty">Click an agent to view its trajectory.</div>'
        initial_raw = 'Raw JSON is lazy-loaded for the selected task.'

        content = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Open Evaluation Dashboard</title>
<style>
:root{--bg:#f4f6f1;--surface:#fffdf7;--surface-2:#eff5eb;--ink:#1f3322;--muted:#697b6d;--border:#d7e0d3;--accent:#2f6b47;--accent-2:#194f34;--ok:#1f8f4e;--warn:#b7791f;--bad:#c2412d;--shadow:0 10px 30px rgba(23,35,28,.08);--radius:14px;--radius-sm:10px}
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:Georgia,'Times New Roman',serif;background:radial-gradient(circle at top right,#d9ead6 0%,#f4f6f1 38%,#e9efe4 100%);color:var(--ink);min-height:100vh}
.header{display:flex;justify-content:space-between;align-items:center;padding:18px 24px;background:linear-gradient(120deg,#173825,#2f6b47);color:#f6f9f2;box-shadow:var(--shadow)}
.title{font-size:20px;font-weight:700;letter-spacing:.02em}
.subtitle{font-size:11px;color:rgba(246,249,242,.72);margin-top:4px}
.meta{font-size:11px;color:rgba(246,249,242,.7);text-align:right}
.error-banner{display:none;margin:16px 24px 0;padding:12px 14px;border-radius:12px;border:1px solid #efc2b8;background:#fff1ed;color:#9a3412;font-size:12px;line-height:1.5;white-space:pre-wrap}
.error-banner.visible{display:block}
.stats{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:12px;padding:18px 24px}
.stat{background:rgba(255,253,247,.92);border:1px solid var(--border);border-radius:var(--radius-sm);padding:14px 16px;box-shadow:var(--shadow)}
.stat-value{font-size:24px;font-weight:700}
.stat-label{font-size:11px;text-transform:uppercase;letter-spacing:.08em;color:var(--muted);margin-top:4px}
.layout{display:grid;grid-template-columns:300px minmax(260px,.72fr) minmax(620px,1.78fr);gap:14px;padding:0 24px 18px;align-items:start}
.panel{background:rgba(255,253,247,.95);border:1px solid var(--border);border-radius:var(--radius);box-shadow:var(--shadow);overflow:hidden;display:flex;flex-direction:column;min-height:520px}
.layout .panel{height:clamp(520px,calc(100vh - 220px),860px)}
.panel-head{display:flex;justify-content:space-between;align-items:center;padding:12px 16px;background:var(--surface-2);border-bottom:1px solid var(--border)}
.panel-title{font-size:11px;font-weight:700;letter-spacing:.08em;text-transform:uppercase;color:var(--muted)}
.panel-count{background:var(--accent);color:#fff;border-radius:999px;padding:3px 8px;font-size:10px;font-weight:700}
.panel-toolbar{display:flex;flex-wrap:wrap;gap:6px;padding:8px 12px;border-bottom:1px solid var(--border);background:var(--surface);font-size:11px}
.panel-toolbar select,.panel-toolbar input{padding:4px 6px;border:1px solid var(--border);border-radius:6px;font-size:11px;background:var(--surface);color:var(--ink);font-family:inherit;outline:none;max-width:100px}
.panel-toolbar select:focus,.panel-toolbar input:focus{border-color:var(--accent)}
.panel-toolbar label{display:flex;align-items:center;gap:4px;color:var(--muted);font-size:10px;white-space:nowrap}
.panel-toolbar .filter-clear{background:none;border:none;color:var(--muted);cursor:pointer;font-size:13px;padding:0 2px;line-height:1}
.panel-toolbar .filter-clear:hover{color:var(--accent-2)}
.panel-body{flex:1;overflow:auto}
#agents.panel-body,#trajectory.panel-body{overscroll-behavior:contain}
.task-item{padding:12px 16px;border-bottom:1px solid var(--border);cursor:pointer;transition:background .15s ease,border-color .15s ease}
.task-item:hover{background:#f2f8ef}
.task-item.active{background:#e6f1e6;border-left:3px solid var(--accent)}
.task-top{display:flex;justify-content:space-between;gap:8px;align-items:center}
.task-id{font-size:13px;font-weight:700;max-width:170px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.task-query{margin-top:6px;font-size:12px;color:var(--muted);line-height:1.45;display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden}
.task-meta{margin-top:6px;font-size:11px;color:var(--muted);display:flex;gap:10px;flex-wrap:wrap}
.badge{display:inline-flex;align-items:center;gap:6px;padding:3px 9px;border-radius:999px;font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:.05em}
.badge::before{content:'';width:6px;height:6px;border-radius:50%}
.badge.ok{background:#dff3e6;color:var(--ok)}
.badge.ok::before{background:var(--ok)}
.badge.warn{background:#fff2da;color:var(--warn)}
.badge.warn::before{background:var(--warn)}
.badge.bad{background:#fde4df;color:var(--bad)}
.badge.bad::before{background:var(--bad)}
.badge.t0{background:#fde4df;color:#9a3412}.badge.t0::before{background:#9a3412}
.badge.t1{background:#fff0e8;color:#c2412d}.badge.t1::before{background:#c2412d}
.badge.t2{background:#fff8e6;color:#b7791f}.badge.t2::before{background:#b7791f}
.badge.t3{background:#e6f5e8;color:#2d7d4f}.badge.t3::before{background:#2d7d4f}
.badge.t4{background:#d4f0dc;color:#1f6e3a}.badge.t4::before{background:#1f6e3a}
.section{padding:14px 16px;border-bottom:1px solid var(--border)}
.section:last-child{border-bottom:none}
.label{font-size:11px;text-transform:uppercase;letter-spacing:.08em;color:var(--muted);margin-bottom:6px}
.value{font-size:13px;line-height:1.6}
.value.mono,.code,.link{font-family:ui-monospace,SFMono-Regular,Menlo,monospace}
.link{color:var(--accent);text-decoration:none;word-break:break-all}
.link:hover{text-decoration:underline}
.quote{padding:10px 12px;background:#f8fbf5;border:1px solid var(--border);border-radius:10px;font-size:12px;line-height:1.6;white-space:pre-wrap;max-height:120px;overflow-y:auto}
.chips{display:flex;gap:6px;flex-wrap:wrap}
.chip{background:#eef6ea;border:1px solid var(--border);border-radius:999px;padding:3px 8px;font-size:11px;color:var(--muted)}
.agent-card{border:1px solid var(--border);border-radius:12px;margin-top:10px;background:#fff}
.agent-head{display:flex;justify-content:space-between;align-items:flex-start;gap:10px;padding:12px 14px;border-bottom:1px solid var(--border);background:#f8fbf5}
.agent-id{font-size:13px;font-weight:700}
.agent-meta{font-size:11px;color:var(--muted);margin-top:4px}
.agent-score{font-size:18px;font-weight:700;color:var(--accent)}
.list{display:flex;flex-direction:column;gap:8px}
.item{padding:10px 12px;border:1px solid var(--border);border-radius:10px;background:#fffdfb}
.item-top{display:flex;justify-content:space-between;gap:8px;align-items:center;margin-bottom:6px}
.item-title{font-size:12px;font-weight:700}
.item-body{font-size:12px;line-height:1.55;color:var(--muted)}
.ag-ov-card{margin:8px 12px;border:1px solid var(--border);border-radius:10px;overflow:hidden;background:#fff;cursor:pointer;transition:border-color .15s,box-shadow .15s}
.ag-ov-card:hover{border-color:var(--accent)}
.ag-ov-card.selected{border-color:var(--accent);box-shadow:0 0 0 2px rgba(47,107,71,.18)}
.ag-ov-hdr{display:flex;align-items:center;justify-content:space-between;gap:8px;padding:10px 12px;background:#f8fbf5}
.ag-ov-hdr:hover{background:#eef6ea}
.ag-ov-body{padding:6px 12px 10px}
.ag-rate{font-size:26px;font-weight:800;color:var(--accent);line-height:1}
.ag-rate-meta{font-size:11px;color:var(--muted);text-align:right;line-height:1.45}
.ag-pass-caption{font-size:10px;text-transform:uppercase;letter-spacing:.08em;color:var(--muted)}
.ag-summary-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px;margin-top:4px}
.ag-summary-cell{padding:8px 9px;border:1px solid var(--border);border-radius:8px;background:#fffdf9}
.ag-summary-label{font-size:10px;text-transform:uppercase;letter-spacing:.06em;color:var(--muted);margin-bottom:4px}
.ag-summary-value{font-size:12px;font-weight:700;color:var(--ink)}
.dim-acc{margin-bottom:5px;border:1px solid var(--border);border-radius:7px;overflow:hidden}
.dim-acc-hdr{display:flex;align-items:center;gap:8px;padding:7px 9px;cursor:pointer;background:#f8fbf5;font-size:12px}
.dim-acc-hdr:hover{background:#eef6ea}
.dim-expand{margin-left:auto;font-size:9px;color:var(--muted);transition:transform .15s}
.dim-acc.open .dim-expand{transform:rotate(90deg)}
.dim-acc-body{display:none;padding:8px 10px;border-top:1px solid var(--border);background:#fffdf9}
.dim-acc.open .dim-acc-body{display:block}
.traj-group{border-bottom:1px solid var(--border)}
.traj-group:last-child{border-bottom:none}
.traj-group-hdr{display:flex;align-items:center;gap:8px;padding:10px 14px;cursor:pointer;background:#f8fbf5;font-size:12px;font-weight:700}
.traj-group-hdr:hover{background:#eef6ea}
.traj-group-count{font-size:11px;color:var(--muted);font-weight:400;margin-left:auto}
.traj-expand{font-size:9px;color:var(--muted);transition:transform .15s}
.traj-group.open .traj-expand{transform:rotate(90deg)}
.traj-group-body{display:none}
.traj-group.open .traj-group-body{display:block}
.traj-sticky{position:sticky;top:0;z-index:6;background:rgba(255,253,247,.97);backdrop-filter:blur(6px);border-bottom:1px solid var(--border);box-shadow:0 8px 20px rgba(23,35,28,.06)}
.traj-summary{padding:14px 16px 10px}
.traj-summary-top{display:flex;justify-content:space-between;align-items:flex-start;gap:12px}
.traj-summary-main{min-width:0;display:flex;flex-direction:column;gap:6px}
.traj-agent-name{font-size:16px;font-weight:700;line-height:1.25}
.traj-task-name{font-size:13px;line-height:1.45;color:var(--ink);word-break:break-word}
.traj-meta-row{display:flex;flex-wrap:wrap;gap:8px;margin-top:2px}
.traj-task-switcher{display:flex;gap:8px;overflow:auto;padding:0 16px 12px;scrollbar-width:thin}
.traj-task-pill{flex:0 0 auto;min-width:170px;max-width:260px;padding:10px 12px;border:1px solid var(--border);border-radius:12px;background:#fff;cursor:pointer;transition:border-color .15s ease,box-shadow .15s ease,background .15s ease}
.traj-task-pill:hover{border-color:var(--accent)}
.traj-task-pill.active{border-color:var(--accent);background:#eef6ea;box-shadow:0 0 0 2px rgba(47,107,71,.14)}
.traj-task-pill-title{font-size:12px;font-weight:700;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.traj-task-pill-meta{margin-top:5px;font-size:11px;color:var(--muted);display:flex;gap:8px;flex-wrap:wrap}
.traj-detail{padding-bottom:8px}
.tree-layout{display:grid;grid-template-columns:minmax(240px,290px) minmax(0,1fr);gap:12px;padding:12px 16px 16px}
.tree-rail{display:flex;flex-direction:column;gap:8px}
.tree-main-card{border:1px solid var(--border);border-radius:12px;background:#fff;padding:10px 11px;cursor:pointer;transition:border-color .15s ease,box-shadow .15s ease,background .15s ease}
.tree-main-card:hover{border-color:var(--accent)}
.tree-main-card.active{border-color:var(--accent);background:#eef6ea;box-shadow:0 0 0 2px rgba(47,107,71,.14)}
.tree-main-title{font-size:12px;font-weight:700;line-height:1.4;color:var(--ink)}
.tree-main-meta{margin-top:6px;display:flex;flex-wrap:wrap;gap:6px;font-size:11px;color:var(--muted)}
.tree-main-origin{font-size:10px;text-transform:uppercase;letter-spacing:.06em;color:var(--muted)}
.tree-stage-shell{display:flex;flex-direction:column;gap:12px;min-width:0}
.tree-stage-header{padding:12px 14px;border:1px solid var(--border);border-radius:12px;background:#fff}
.tree-stage-title{font-size:16px;font-weight:800;line-height:1.25;color:var(--ink)}
.tree-stage-goal{margin-top:8px;font-size:12px;line-height:1.6;color:var(--muted)}
.tree-subtasks{display:flex;flex-wrap:wrap;gap:8px}
.tree-subtask-pill{min-width:190px;max-width:280px;padding:10px 12px;border:1px solid var(--border);border-radius:12px;background:#fff;cursor:pointer;transition:border-color .15s ease,box-shadow .15s ease,background .15s ease}
.tree-subtask-pill:hover{border-color:var(--accent)}
.tree-subtask-pill.active{border-color:var(--accent);background:#eef6ea;box-shadow:0 0 0 2px rgba(47,107,71,.12)}
.tree-subtask-title{font-size:12px;font-weight:700;line-height:1.4;color:var(--ink)}
.tree-subtask-meta{margin-top:5px;display:flex;gap:6px;flex-wrap:wrap;font-size:11px;color:var(--muted)}
.tree-detail-grid{display:grid;grid-template-columns:minmax(0,1fr);gap:10px}
.tree-info-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:10px}
.tree-info-card{padding:10px 12px;border:1px solid var(--border);border-radius:10px;background:#fffdf9}
.tree-info-label{font-size:10px;text-transform:uppercase;letter-spacing:.06em;color:var(--muted);margin-bottom:5px}
.tree-info-value{font-size:12px;line-height:1.55;color:var(--ink)}
.tree-check-list{display:flex;flex-direction:column;gap:5px}
.tree-check-item{font-size:11px;color:var(--muted);padding:6px 8px;border:1px solid var(--border);border-radius:8px;background:#fff}
.empty{display:flex;align-items:center;justify-content:center;height:100%;padding:24px;color:var(--muted);font-size:13px;text-align:center}
.bottom{display:grid;grid-template-columns:1fr 1fr;gap:14px;padding:0 24px 24px}
.bottom .panel{min-height:260px}
pre{margin:0;padding:14px 16px;background:#101614;color:#d0e1d0;font-size:11px;line-height:1.5;overflow:auto;max-height:320px}
.raw-load-btn{border:1px solid var(--border);background:#fff;color:var(--accent-2);border-radius:999px;padding:5px 10px;font-size:11px;font-weight:700;cursor:pointer}
.raw-load-btn:hover{background:#f2f8ef}
.pagination-bar{display:flex;align-items:center;justify-content:center;gap:8px;padding:8px 12px;border-top:1px solid var(--border);background:var(--surface-2);font-size:11px}
.pagination-bar.hidden{display:none}
.pagination-btn{padding:4px 10px;border:1px solid var(--border);border-radius:6px;font-size:11px;font-weight:600;cursor:pointer;background:var(--surface);color:var(--ink);transition:all .15s ease}
.pagination-btn:hover{background:var(--accent);color:#fff;border-color:var(--accent)}
.pagination-btn:disabled{opacity:.4;cursor:not-allowed}
.pagination-info{color:var(--muted);font-size:11px}
.raw-load-btn:disabled{opacity:.5;cursor:not-allowed}
.audit-entry{padding:11px 14px;border-bottom:1px solid var(--border)}
.audit-entry:last-child{border-bottom:none}
.audit-ts{font-size:11px;color:var(--muted)}
.audit-cmd{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px;margin-top:5px;line-height:1.5;word-break:break-all}
.audit-meta{margin-top:6px;font-size:11px;color:var(--muted);display:flex;gap:10px;flex-wrap:wrap}
.flow-collapse-hdr{display:flex;align-items:center;gap:6px;cursor:pointer;user-select:none;padding:2px 0 6px}
.flow-collapse-hdr:hover .label{color:var(--accent)}
.flow-toggle{font-size:9px;color:var(--muted);transition:transform .15s}
.flow-section.open .flow-toggle{transform:rotate(90deg)}
.flow-body{display:none}
.flow-section.open .flow-body{display:block}
.flow{display:flex;align-items:stretch;gap:8px;flex-wrap:wrap}
.flow-arrow{display:flex;align-items:center;justify-content:center;color:var(--muted);font-size:16px;padding:0 2px}
.flow-node{min-width:160px;flex:1 1 180px;border:1px solid var(--border);border-radius:10px;background:#fff;padding:10px 12px}
.flow-node-top{display:flex;justify-content:space-between;align-items:center;gap:8px;margin-bottom:6px}
.flow-node-id{font-size:12px;font-weight:700;font-family:ui-monospace,SFMono-Regular,Menlo,monospace}
.flow-node-summary{font-size:12px;line-height:1.5;color:var(--ink)}
.flow-node-meta{margin-top:6px;font-size:11px;line-height:1.5;color:var(--muted);white-space:pre-wrap;word-break:break-word}
.ag-ov-actions{display:flex;gap:6px;margin-top:8px}
.ag-ov-btn{padding:4px 10px;border:1px solid var(--border);border-radius:6px;font-size:11px;font-weight:600;cursor:pointer;background:var(--surface-2);color:var(--ink);transition:all .15s ease}
.ag-ov-btn:hover{background:var(--accent);color:#fff;border-color:var(--accent)}
.modal-overlay{position:fixed;inset:0;background:rgba(0,0,0,.45);z-index:999;display:flex;align-items:center;justify-content:center}
.modal-box{background:var(--surface);border-radius:12px;box-shadow:0 8px 32px rgba(0,0,0,.18);max-width:700px;width:90vw;max-height:80vh;display:flex;flex-direction:column}
.modal-head{display:flex;justify-content:space-between;align-items:center;padding:14px 18px;border-bottom:1px solid var(--border)}
.modal-title{font-size:13px;font-weight:700}
.modal-close{padding:4px 10px;border:none;background:none;font-size:18px;cursor:pointer;color:var(--muted)}
.modal-body{padding:16px 18px;overflow-y:auto;font-size:12px;line-height:1.7;white-space:pre-wrap;word-break:break-word;color:var(--ink)}
.modal-body code{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;background:var(--surface-2);padding:1px 5px;border-radius:4px;font-size:11px}
.modal-section{margin-bottom:14px}
.modal-section-title{font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:.05em;color:var(--muted);margin-bottom:6px}
.tool-tag{display:inline-block;padding:2px 8px;margin:2px 4px 2px 0;border-radius:999px;font-size:11px;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;background:var(--surface-2);border:1px solid var(--border)}
__SHARED_AGENT_DETAIL_CSS__
@media (max-width:1200px){.layout{grid-template-columns:1fr}.layout .panel{height:auto;max-height:none}.bottom{grid-template-columns:1fr}.panel{min-height:360px}.stats{grid-template-columns:repeat(2,minmax(0,1fr))}.tree-layout{grid-template-columns:1fr}.tree-info-grid,.ag-summary-grid{grid-template-columns:1fr}}
</style>
</head>
<body>
<div class="header">
  <div>
    <div class="title">Open Evaluation Tree Dashboard</div>
    <div class="subtitle">Legacy-compatible reports with agent pass-rate cards and tree-aware task trajectory</div>
  </div>
    <div class="meta" id="meta">__INITIAL_META__</div>
</div>
<div class="error-banner __INITIAL_ERROR_CLASS__" id="error-banner">__INITIAL_ERROR__</div>
<div class="stats">
    <div class="stat"><div class="stat-value" id="s-total">__INITIAL_TOTAL__</div><div class="stat-label">Total</div></div>
    <div class="stat"><div class="stat-value" id="s-pass" style="color:var(--ok)">__INITIAL_PASSED__</div><div class="stat-label">Passed</div></div>
    <div class="stat"><div class="stat-value" id="s-partial" style="color:var(--warn)">__INITIAL_PARTIAL__</div><div class="stat-label">Partial</div></div>
    <div class="stat"><div class="stat-value" id="s-fail" style="color:var(--bad)">__INITIAL_FAILED__</div><div class="stat-label">Failed</div></div>
    <div class="stat"><div class="stat-value" id="s-audit">__INITIAL_AUDIT_TOTAL__</div><div class="stat-label">Audit Entries</div></div>
</div>
<div class="layout">
  <div class="panel">
        <div class="panel-head"><div class="panel-title">Tasks</div><div class="panel-count" id="task-count">__INITIAL_TASK_COUNT__</div></div>
        <div class="panel-toolbar" id="task-toolbar">
          <label>Sort
            <select id="sort-by" onchange="applySortFilter()">
              <option value="index">Index</option>
              <option value="score">Score</option>
            </select>
            <select id="sort-dir" onchange="applySortFilter()">
              <option value="asc">↑</option>
              <option value="desc">↓</option>
            </select>
          </label>
          <label>Score
            <input type="number" id="filter-score-min" placeholder="min" min="0" step="0.01" onchange="applySortFilter()" style="width:52px">
            <input type="number" id="filter-score-max" placeholder="max" min="0" step="0.01" onchange="applySortFilter()" style="width:52px">
          </label>
          <label>Search
            <input type="text" id="filter-query" placeholder="query text..." onchange="applySortFilter()" style="width:90px">
          </label>
          <label>Model
            <select id="filter-model" onchange="applySortFilter()">
              <option value="">All</option>
            </select>
          </label>
          <button class="filter-clear" onclick="clearSortFilter()" title="Clear filters">↺</button>
        </div>
        <div class="panel-body" id="tasks">__INITIAL_TASKS__</div>
        <div class="pagination-bar" id="tasks-pagination"></div>
  </div>
  <div class="panel">
    <div class="panel-head"><div class="panel-title">Agents</div></div>
        <div class="panel-body" id="agents">__INITIAL_AGENTS__</div>
  </div>
  <div class="panel">
    <div class="panel-head"><div class="panel-title">Task Tree / Trajectory</div><div class="panel-count" id="traj-header"></div></div>
        <div class="panel-body" id="trajectory">__INITIAL_TRAJECTORY__</div>
  </div>
</div>
<div class="bottom">
  <div class="panel">
        <div class="panel-head"><div class="panel-title">Raw JSON</div><button class="raw-load-btn" id="raw-load-btn" type="button" onclick="loadRawJson()" disabled>Load</button></div>
    <div class="panel-body"><pre id="raw">__INITIAL_RAW__</pre></div>
  </div>
  <div class="panel">
    <div class="panel-head"><div class="panel-title">Audit Log</div><div class="panel-count" id="audit-count">0</div></div>
    <div class="panel-body" id="audit"><div class="empty">No audit log loaded.</div></div>
  </div>
</div>
<script>
const state={selected:null,audit:[],tasksPayload:null,currentAgents:[],currentTaskIndex:null,selectedAgent:null,selectedAgentTask:0,selectedAgentMainTask:0,selectedAgentSubtask:0,currentTaskPayload:null,rawLoadedIndex:null,taskOffset:0,taskTotal:0,sortBy:'index',sortDir:'asc',filterModel:'',filterScoreMin:'',filterScoreMax:'',filterQuery:''};
const PAGE_SIZE=__PAGE_SIZE__;
const esc=value=>String(value??'').replace(/[&<>"']/g,m=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[m]));
function setUiError(message){
    const banner=document.getElementById('error-banner');
    banner.textContent=String(message||'Unknown error');
    banner.classList.add('visible');
    document.getElementById('meta').textContent='Error';
    document.getElementById('agents').innerHTML=`<div class="empty">${esc(message||'Unknown error')}</div>`;
    document.getElementById('trajectory').innerHTML='<div class="empty">Preview unavailable.</div>';
    document.getElementById('raw').textContent=String(message||'Unknown error');
    const rawBtn=document.getElementById('raw-load-btn');
    if(rawBtn)rawBtn.disabled=true;
}
function clearUiError(){
    const banner=document.getElementById('error-banner');
    banner.textContent='';
    banner.classList.remove('visible');
}
function setRawPlaceholder(message='Raw JSON is lazy-loaded for the selected task.'){
    document.getElementById('raw').textContent=String(message);
    state.rawLoadedIndex=null;
}
function loadRawJson(){
    const rawBtn=document.getElementById('raw-load-btn');
    if(!state.currentTaskPayload){
        setRawPlaceholder('Select a task first to inspect raw JSON.');
        if(rawBtn)rawBtn.disabled=true;
        return;
    }
    document.getElementById('raw').textContent=JSON.stringify(state.currentTaskPayload,null,2);
    state.rawLoadedIndex=state.selected;
}
function normalizeTasksPayload(payload){
    if(Array.isArray(payload)){
        return {tasks:payload, summary:{total:payload.length}, audit_summary:{}, meta:{source_shape:'array'}};
    }
    if(!payload || typeof payload!=='object'){
        return {tasks:[], summary:{}, audit_summary:{}, meta:{source_shape:typeof payload}};
    }
    if(Array.isArray(payload.tasks)){
        return {
            tasks:payload.tasks,
            summary:payload.summary||{},
            audit_summary:payload.audit_summary||{},
            meta:payload.meta||{},
        };
    }
    if(Array.isArray(payload.results)){
        return {
            tasks:payload.results,
            summary:payload.summary||{total:payload.results.length},
            audit_summary:payload.audit_summary||{},
            meta:{...(payload.meta||{}), source_shape:'results'},
        };
    }
    if(Array.isArray(payload.rows)){
        return {
            tasks:payload.rows,
            summary:payload.summary||{total:payload.rows.length},
            audit_summary:payload.audit_summary||{},
            meta:{...(payload.meta||{}), source_shape:'rows'},
        };
    }
    return {
        tasks:[],
        summary:payload.summary||{},
        audit_summary:payload.audit_summary||{},
        meta:{...(payload.meta||{}), source_shape:'object-without-tasks'},
    };
}
async function fetchJson(url){
    const res=await fetch(url);
    const text=await res.text();
    let payload;
    try{payload=text?JSON.parse(text):null;}catch(_err){payload=text;}
    if(!res.ok){
        const detail=payload&&typeof payload==='object'&&payload.detail?payload.detail:text||res.statusText;
        throw new Error(`${url} -> HTTP ${res.status}: ${detail}`);
    }
    return payload;
}
const statusBadge=status=>{
    if(status==='passed'||status==='completed')return `<span class="badge ok">${esc(status==='completed'?'completed':'passed')}</span>`;
    if(status==='partial'||status==='blocked')return `<span class="badge warn">${esc(status||'partial')}</span>`;
    return `<span class="badge bad">${esc(status||'failed')}</span>`;
};
const resolveScore100=(score,maxScore)=>{
    if(typeof score!=='number')return null;
    if(typeof maxScore==='number'&&maxScore>1)return Math.min(100,Math.max(0,(score/maxScore)*100));
    if(score<=1&&score>=0)return Math.min(100,Math.max(0,score*100));
    return Math.min(100,Math.max(0,score));
};
const taskScoreBadge=task=>{
    if(task.status==='passed'||task.status==='completed')return statusBadge(task.status);
    const score=getTaskSummaryScore(task);
    const maxScore=getTaskSummaryScoreMax(task);
    const pct=resolveScore100(score,maxScore);
    if(pct===null)return statusBadge(task.status);
    const tier=Math.min(4,Math.floor(pct/20));
    return `<span class="badge t${tier}">${esc(scoreText(score, maxScore))}</span>`;
};
const fmt=value=>value==null?'-':(typeof value==='number'?value.toFixed(2):String(value));
const rateText=value=>typeof value==='number'?value.toFixed(2):'-';
const scoreText=(value,maxValue=null)=>{
    if(typeof value!=='number')return '-';
    if(typeof maxValue==='number'){
        const maxLabel=Number.isInteger(maxValue)?maxValue.toFixed(0):maxValue.toFixed(2);
        return `${value.toFixed(2)}/${maxLabel}`;
    }
    return value.toFixed(2);
};
const averageScore=values=>values.length?(values.reduce((sum,value)=>sum+value,0)/values.length):null;
const getTaskSummaryScore=task=>typeof task?.overall_score==='number'?task.overall_score:(typeof task?.avg_task_completion_score==='number'?task.avg_task_completion_score:(typeof task?.avg_main_task_pass_rate==='number'?task.avg_main_task_pass_rate:null));
const getTaskSummaryScoreMax=task=>typeof task?.overall_max_score==='number'?task.overall_max_score:null;
function buildScreenshotSrc(step){
    const screenshotPath=step&&step.screenshot_path?String(step.screenshot_path).trim():'';
    if(screenshotPath)return `/api/screenshot?path=${encodeURIComponent(screenshotPath)}`;
    const screenshot=step&&step.screenshot?String(step.screenshot).trim():'';
    if(!screenshot)return '';
    if(screenshot.startsWith('data:')||screenshot.startsWith('http://')||screenshot.startsWith('https://'))return screenshot;
    return `data:image/png;base64,${screenshot}`;
}
const toImgSrc=value=>{
    const text=String(value??'').trim();
    if(!text)return '';
    if(text.startsWith('data:')||text.startsWith('http://')||text.startsWith('https://'))return text;
    return `data:image/png;base64,${text}`;
};
function getAgentTasks(agent){
    return Array.isArray(agent&&agent.tasks)?agent.tasks.filter(task=>task&&typeof task==='object'):[];
}
function getAgentTaskResults(agent){
    const taskResults=Array.isArray(agent&&agent.task_results)?agent.task_results.filter(task=>task&&typeof task==='object'):[];
    if(taskResults.length)return taskResults;
    return getAgentTasks(agent).map(task=>({
        task_id:task.task_id,
        title:task.title,
        passed:task.passed,
        verdict:task.verdict||task.status,
        reason:task.reason,
        steps:task.steps,
        covers_standard_ids:Array.isArray(task.covers_standard_ids)?task.covers_standard_ids:[],
    }));
}
function getTaskTrajectorySteps(task){
    const trajectory=task&&typeof task.trajectory==='object'?task.trajectory:{};
    return Array.isArray(trajectory.steps)?trajectory.steps.filter(step=>step&&typeof step==='object'):[];
}
function getAgentStepCount(agent){
    const tasks=getAgentTasks(agent);
    if(tasks.length){
        return tasks.reduce((total,task)=>total+getTaskTrajectorySteps(task).length,0);
    }
    const trajectory=agent&&typeof agent.trajectory==='object'?agent.trajectory:{};
    return Array.isArray(trajectory.steps)?trajectory.steps.filter(step=>step&&typeof step==='object').length:0;
}
function getAgentTaskTree(agent){
    return Array.isArray(agent&&agent.task_tree)?agent.task_tree.filter(main=>main&&typeof main==='object'):[];
}
function getTaskLikeStatus(task){
    if(!task||typeof task!=='object')return 'unknown';
    if(task.status)return String(task.status);
    if(task.verdict)return String(task.verdict);
    if(task.passed===true)return 'passed';
    if(task.passed===false)return 'failed';
    return 'unknown';
}
function getAgentPassSummary(agent){
    const tree=getAgentTaskTree(agent);
    if(tree.length){
        const terminal=tree.map(main=>getTaskLikeStatus(main)).filter(status=>!['pending','running','unknown'].includes(status));
        const passed=terminal.filter(status=>status==='passed').length;
        const evaluated=terminal.length;
        const scoreValues=tree.map(main=>main&&typeof main.score==='number'?main.score:null).filter(value=>typeof value==='number');
        const passRateValues=tree.map(main=>main&&typeof main.pass_rate==='number'?main.pass_rate:(main&&typeof main.completion_score==='number'?main.completion_score:null)).filter(value=>typeof value==='number');
        const subtaskCount=tree.reduce((total,main)=>total+(Array.isArray(main.subtasks)?main.subtasks.length:0),0);
        const mainTaskWeight=typeof agent?.main_task_weight==='number'?agent.main_task_weight:(tree.length?20/tree.length:null);
        const score=typeof agent?.score==='number'?agent.score:scoreValues.reduce((sum,value)=>sum+value,0);
        const maxScore=typeof agent?.score_max==='number'?agent.score_max:20;
        return {
            score,
            maxScore,
            passRate:typeof agent?.main_task_completion_score==='number'?agent.main_task_completion_score:averageScore(passRateValues),
            passed,
            evaluated,
            total:tree.length,
            mainTaskCount:tree.length,
            subtaskCount,
            mainTaskWeight,
            mode:'tree',
        };
    }
    const taskResults=getAgentTaskResults(agent);
    const terminal=taskResults.filter(task=>['passed','failed','partial'].includes(getTaskLikeStatus(task)));
    const passed=terminal.filter(task=>getTaskLikeStatus(task)==='passed').length;
    const evaluated=terminal.length;
    const scoreValues=taskResults.map(task=>task&&typeof task.completion_score==='number'?task.completion_score:null).filter(value=>typeof value==='number');
    return {
        score:averageScore(scoreValues),
        maxScore:null,
        passRate:averageScore(scoreValues),
        passed,
        evaluated,
        total:taskResults.length,
        mainTaskCount:taskResults.length,
        subtaskCount:0,
        mainTaskWeight:null,
        mode:'flat',
    };
}
function renderCheckItems(items){
    const list=Array.isArray(items)?items.filter(Boolean):[];
    if(!list.length)return '<div class="tree-check-item">None recorded.</div>';
    return list.map(item=>`<div class="tree-check-item">${esc(item)}</div>`).join('');
}
function renderPipelineHandoffs(report){
    const handoffs=Array.isArray(report.pipeline_handoffs)?report.pipeline_handoffs:[];
    if(!handoffs.length)return '';
    const flowId='flow-'+Math.random().toString(36).slice(2);
    const passCount=handoffs.filter(h=>h.status==='passed'||h.status==='completed').length;
    const chip=`${passCount}/${handoffs.length} passed`;
    return `<div class="section flow-section" id="${flowId}">
      <div class="flow-collapse-hdr" onclick="document.getElementById('${flowId}').classList.toggle('open')">
        <div class="label" style="margin-bottom:0">Pipeline Flow</div>
        <span style="font-size:11px;color:var(--muted);margin-left:4px">${esc(chip)}</span>
        <span class="flow-toggle" style="margin-left:auto">▶</span>
      </div>
      <div class="flow-body"><div class="flow">${handoffs.map((handoff,index)=>{
        const facts=handoff&&typeof handoff.facts==='object'&&handoff.facts?handoff.facts:{};
        const meta=[`stage ${handoff.stage||'-'}`];
        if(facts.artifacts_path)meta.push(`artifacts ${facts.artifacts_path}`);
        if(facts.process_log_dir)meta.push(`logs ${facts.process_log_dir}`);
        const node=`<div class="flow-node">
            <div class="flow-node-top"><span class="flow-node-id">${esc(handoff.agent_id||'unknown')}</span>${statusBadge(handoff.status||'unknown')}</div>
            <div class="flow-node-summary">${esc(handoff.summary||'No summary recorded.')}</div>
            <div class="flow-node-meta">${esc(meta.join(' | '))}</div>
        </div>`;
        return index===0?node:`<div class="flow-arrow">&rarr;</div>${node}`;
    }).join('')}</div></div></div>`;
}
__SHARED_AGENT_DETAIL_JS__
// --- Agent config cache (loaded once from /api/agents-config) ---
state.agentsConfig=null;
async function loadAgentsConfig(){
  if(state.agentsConfig)return state.agentsConfig;
  try{const payload=await fetchJson('/api/agents-config');state.agentsConfig=payload.agents||{};}catch(_){state.agentsConfig={};}
  return state.agentsConfig;
}
function openAgentModal(agentId,kind){
  const cfg=(state.agentsConfig||{})[agentId];
  if(!cfg)return;
  const overlay=document.createElement('div');overlay.className='modal-overlay';
  const title=kind==='tools'?`${esc(cfg.name)} — Allowed Tools`:kind==='prompt'?`${esc(cfg.name)} — Persona`:'';
  let body='';
  if(kind==='tools'){
    const tools=cfg.allowed_tools||[];
    body=tools.length?tools.map(t=>`<span class="tool-tag">${esc(t)}</span>`).join(''):'<div class="empty">No tools declared.</div>';
  }else if(kind==='prompt'){
    body=`<div class="modal-section"><div class="modal-section-title">Persona / System Prompt</div><p>${esc(cfg.system_prompt||'No persona configured.')}</p></div>`;
  }
  overlay.innerHTML=`<div class="modal-box"><div class="modal-head"><div class="modal-title">${title}</div><button class="modal-close" onclick="this.closest('.modal-overlay').remove()">&times;</button></div><div class="modal-body">${body}</div></div>`;
  overlay.addEventListener('click',e=>{if(e.target===overlay)overlay.remove();});
  document.body.appendChild(overlay);
}
function renderAgentOverview(agents,taskIndex){
  if(!Array.isArray(agents)||!agents.length)return '<div class="section"><div class="label">Agents</div><div class="value">No agent report available.</div></div>';
  // eagerly load configs on first render
  loadAgentsConfig();
  return agents.map((agent,agentIndex)=>{
    const agOvId=`ag-ov-${taskIndex}-${agentIndex}`;
        const summary=getAgentPassSummary(agent);
        const score=summary.score==null?'-':scoreText(summary.score,summary.maxScore);
    const passRate=summary.passRate==null?'-':rateText(summary.passRate);
        const modeLabel=esc(agent.task_synthesis_mode||summary.mode||'flat');
        const stepCount=getAgentStepCount(agent);
        const agentId=esc(agent.agent_id||'');
    return `<div class="ag-ov-card" id="${agOvId}" onclick="selectAgent(${taskIndex},${agentIndex})">
      <div class="ag-ov-hdr">
        <div>
          <div class="ag-id">${esc(agent.agent_id||agent.role||'unknown')}</div>
          <div class="ag-role">status ${esc(agent.status||'-')}${agent.end_reason?` · ${esc(agent.end_reason)}`:''}</div>
        </div>
        <div style="display:flex;align-items:center;gap:8px">
          ${statusBadge(agent.status||'unknown')}
                    <div style="display:flex;flex-direction:column;align-items:flex-end;gap:3px">
                                                <div class="ag-pass-caption">agent score</div>
                                                <div class="ag-rate">${esc(score)}</div>
                        <div class="ag-rate-meta">${summary.passed}/${summary.evaluated||0} passed</div>
                    </div>
        </div>
      </div>
      <div class="ag-ov-body">
                <div class="ag-summary-grid">
                    <div class="ag-summary-cell"><div class="ag-summary-label">Main Tasks</div><div class="ag-summary-value">${esc(String(summary.mainTaskCount||0))}</div></div>
                    <div class="ag-summary-cell"><div class="ag-summary-label">Subtasks</div><div class="ag-summary-value">${esc(String(summary.subtaskCount||0))}</div></div>
                    <div class="ag-summary-cell"><div class="ag-summary-label">Pass Rate</div><div class="ag-summary-value">${esc(passRate)}</div></div>
                    <div class="ag-summary-cell"><div class="ag-summary-label">Mode</div><div class="ag-summary-value">${modeLabel}</div></div>
                    <div class="ag-summary-cell"><div class="ag-summary-label">Steps</div><div class="ag-summary-value">${esc(String(stepCount||0))}</div></div>
                </div>
        <div class="ag-ov-actions" onclick="event.stopPropagation()">
          <button class="ag-ov-btn" onclick="event.stopPropagation();openAgentModal('${agentId}','tools')">Tools</button>
          <button class="ag-ov-btn" onclick="event.stopPropagation();openAgentModal('${agentId}','prompt')">Persona</button>
        </div>
      </div>
    </div>`;
  }).join('');
}
function selectAgent(taskIndex,agentIndex){
  document.querySelectorAll('.ag-ov-card').forEach(el=>el.classList.remove('selected'));
  const card=document.getElementById(`ag-ov-${taskIndex}-${agentIndex}`);
  if(card)card.classList.add('selected');
  const agents=Array.isArray(state.currentAgents)?state.currentAgents:[];
  const agent=agents[agentIndex];
  state.selectedAgent=agentIndex;
    state.selectedAgentTask=0;
    state.selectedAgentMainTask=0;
    state.selectedAgentSubtask=0;
  const header=document.getElementById('traj-header');
  if(header)header.textContent=agent?(agent.agent_id||agent.role||'agent'):'';
  if(!agent){
    document.getElementById('trajectory').innerHTML=`<div class="empty">Agent not found. (pool size: ${agents.length}, requested index: ${agentIndex})</div>`;
    return;
  }
    const tree=getAgentTaskTree(agent);
    const tasks=getAgentTasks(agent);
  const steps=(agent.trajectory&&Array.isArray(agent.trajectory.steps))?agent.trajectory.steps:[];
    if(!tree.length&&!tasks.length&&!steps.length&&!agent.trajectory_ref){
    document.getElementById('trajectory').innerHTML=`<div class="section"><div class="label">Agent: ${esc(agent.agent_id||agent.role||'unknown')}</div><div class="value" style="color:var(--muted)">No trajectory steps recorded for this agent.</div></div>`;
    return;
  }
  document.getElementById('trajectory').innerHTML=renderAgentTrajectory(agent,taskIndex,agentIndex,state.selectedAgentTask||0);
}
function selectAgentMainTask(taskIndex,agentIndex,mainTaskPosition){
    state.selectedAgent=agentIndex;
    state.selectedAgentMainTask=mainTaskPosition;
    state.selectedAgentSubtask=0;
    const agent=(Array.isArray(state.currentAgents)?state.currentAgents:[])[agentIndex];
    if(!agent)return;
    document.getElementById('trajectory').innerHTML=renderAgentTrajectory(agent,taskIndex,agentIndex,state.selectedAgentTask||0);
}
function selectAgentSubtask(taskIndex,agentIndex,mainTaskPosition,subtaskPosition){
    state.selectedAgent=agentIndex;
    state.selectedAgentMainTask=mainTaskPosition;
    state.selectedAgentSubtask=subtaskPosition;
    const agent=(Array.isArray(state.currentAgents)?state.currentAgents:[])[agentIndex];
    if(!agent)return;
    document.getElementById('trajectory').innerHTML=renderAgentTrajectory(agent,taskIndex,agentIndex,state.selectedAgentTask||0);
}
function selectAgentTask(taskIndex,agentIndex,taskPosition){
  state.selectedAgent=agentIndex;
  state.selectedAgentTask=taskPosition;
  const agent=(Array.isArray(state.currentAgents)?state.currentAgents:[])[agentIndex];
  if(!agent)return;
  document.getElementById('trajectory').innerHTML=renderAgentTrajectory(agent,taskIndex,agentIndex,taskPosition);
}
function renderTreeSubtaskTrajectory(mainTask,subtask,taskIndex,agentIndex,mainPosition,subPosition){
        const taskSteps=getTaskTrajectorySteps(subtask);
        const subtaskStatus=getTaskLikeStatus(subtask);
        const basedOn=subtask&&typeof subtask.based_on==='object'?subtask.based_on:{};
        const areaLabel=basedOn.area_label||subtask.goal||'-';
        const dimensions=Array.isArray(subtask.rubric_results)?subtask.rubric_results:(Array.isArray(subtask.dimensions)?subtask.dimensions:[]);
        const dimensionHtml=dimensions.length?dimensions.map(dim=>`<span class="chip">${esc(dim.dimension_id||'dimension')} ${esc(fmt(dim.score))}${dim.verdict?` · ${esc(dim.verdict)}`:''}</span>`).join(''):'<span class="chip">No rubric data</span>';
        const dependencyText=Array.isArray(subtask.depends_on_subtask_ids)&&subtask.depends_on_subtask_ids.length?subtask.depends_on_subtask_ids.join(', '):'None';
        const reasonHtml=subtask.reason?`<div class="tree-info-card"><div class="tree-info-label">Reason</div><div class="tree-info-value">${esc(String(subtask.reason))}</div></div>`:'';
        const stepsHtml=taskSteps.length?taskSteps.map((step,si)=>{
                const args=step.args||step.tool_args||{};
                const argsId=`tree-a-${taskIndex}-${agentIndex}-${mainPosition}-${subPosition}-${si}`;
                const hasArgs=args&&typeof args==='object'&&Object.keys(args).length>0;
                const successIcon=step.success===true?`<span class="ts-success ok">✓</span>`:step.success===false?`<span class="ts-success fail">✗</span>`:'';
                const screenshotSrc=buildScreenshotSrc(step);
                return `<div class="traj-step">
                        <div class="ts-hdr">
                                <span class="ts-idx">#${si+1}</span>
                                <span class="ts-tool">${esc(step.tool_name||'unknown')}</span>
                                ${successIcon}
                                ${step.timestamp?`<span class="ts-time">${esc(step.timestamp)}</span>`:''}
                        </div>
                        ${hasArgs?`<div class="ts-args-toggle" onclick="document.getElementById('${argsId}').classList.toggle('collapsed')">▼ args</div><pre id="${argsId}" class="ts-args collapsed">${esc(JSON.stringify(args,null,2))}</pre>`:''}
                        ${step.result||step.observation?`<div class="ts-result"><b>Result:</b> ${esc(String(step.result||step.observation))}</div>`:''}
                        ${screenshotSrc?`<div class="ts-screenshot"><img src="${esc(screenshotSrc)}" alt="screenshot" onclick="window.open(this.src)" title="Click for full size"/></div>`:''}
                </div>`;
        }).join(''):'<div class="traj-step"><div class="ts-result">No recorded steps for this subtask.</div></div>';
        return `<div class="tree-detail-grid">
            <div class="tree-info-grid">
                <div class="tree-info-card"><div class="tree-info-label">Main Task</div><div class="tree-info-value">${esc(mainTask.title||mainTask.main_task_id||'-')}</div></div>
                <div class="tree-info-card"><div class="tree-info-label">Subtask</div><div class="tree-info-value">${esc(subtask.title||subtask.task_id||'-')}</div></div>
                <div class="tree-info-card"><div class="tree-info-label">Area</div><div class="tree-info-value">${esc(areaLabel)}</div></div>
                <div class="tree-info-card"><div class="tree-info-label">Dependencies</div><div class="tree-info-value">${esc(dependencyText)}</div></div>
            </div>
            <div class="tree-info-card"><div class="tree-info-label">Status</div><div class="chips">${statusBadge(subtaskStatus)}${subtask.task_id?`<span class="chip">${esc(subtask.task_id)}</span>`:''}${subtask.kind?`<span class="chip">${esc(subtask.kind)}</span>`:''}</div></div>
            <div class="tree-info-card"><div class="tree-info-label">Task Text</div><div class="tree-info-value">${esc(subtask.task_text||subtask.goal||'-')}</div></div>
            ${reasonHtml}
            <div class="tree-info-card"><div class="tree-info-label">Success Criteria</div><div class="tree-check-list">${renderCheckItems(subtask.success_criteria)}</div></div>
            <div class="tree-info-card"><div class="tree-info-label">Expected Signals</div><div class="tree-check-list">${renderCheckItems(subtask.expected_signals)}</div></div>
            <div class="tree-info-card"><div class="tree-info-label">Rubric Coverage</div><div class="chips">${dimensionHtml}</div></div>
            <div><div class="section" style="padding-left:0;padding-right:0"><div class="label">Trajectory Steps</div></div>${stepsHtml}</div>
        </div>`;
}
function renderSingleTaskTrajectory(task,taskIndex,agentIndex,taskPosition){
    const taskSteps=getTaskTrajectorySteps(task);
    const taskStatus=task.status||task.verdict||(task.passed===true?'passed':(task.passed===false?'failed':'partial'));
    const taskBadge=statusBadge(taskStatus);
    const completion=typeof task.completion_score==='number'?`<span class="chip">score ${scoreText(task.completion_score)}</span>`:'';
    const dimensions=Array.isArray(task.rubric_results)?task.rubric_results:(Array.isArray(task.dimensions)?task.dimensions:[]);
    const dimsHtml=dimensions.length?`<div class="section"><div class="label">Rubrics</div><div class="chips">${dimensions.map(dim=>`<span class="chip">${esc(dim.dimension_id||'dimension')} ${esc(fmt(dim.score))}${dim.verdict?` · ${esc(dim.verdict)}`:''}</span>`).join('')}</div></div>`:'';
    const reasonHtml=task.reason?`<div class="section"><div class="label">Reason</div><div class="quote">${esc(String(task.reason))}</div></div>`:'';
    const stepsHtml=taskSteps.length?taskSteps.map((step,si)=>{
        const args=step.args||step.tool_args||{};
        const argsId=`traj-a-${taskIndex}-${agentIndex}-${taskPosition}-${si}`;
        const hasArgs=args&&typeof args==='object'&&Object.keys(args).length>0;
        const successIcon=step.success===true?`<span class="ts-success ok">✓</span>`:step.success===false?`<span class="ts-success fail">✗</span>`:'';
        return `<div class="traj-step">
            <div class="ts-hdr">
                <span class="ts-idx">#${si+1}</span>
                <span class="ts-tool">${esc(step.tool_name||'unknown')}</span>
                ${successIcon}
                ${step.timestamp?`<span class="ts-time">${esc(step.timestamp)}</span>`:''}
            </div>
            ${hasArgs?`<div class="ts-args-toggle" onclick="document.getElementById('${argsId}').classList.toggle('collapsed')">▼ args</div><pre id="${argsId}" class="ts-args collapsed">${esc(JSON.stringify(args,null,2))}</pre>`:''}
            ${step.result||step.observation?`<div class="ts-result"><b>Result:</b> ${esc(String(step.result||step.observation))}</div>`:''}
            ${(()=>{const screenshotSrc=buildScreenshotSrc(step);return screenshotSrc?`<div class="ts-screenshot"><img src="${esc(screenshotSrc)}" alt="screenshot" onclick="window.open(this.src)" title="Click for full size"/></div>`:'';})()}
        </div>`;
    }).join(''):'<div class="traj-step"><div class="ts-result">No recorded steps for this task.</div></div>';
    return `<div class="section"><div class="label">Selected Task</div><div class="value">${esc(task.title||task.task_id||`Task ${taskPosition+1}`)}</div><div class="chips" style="margin-top:8px">${completion}${taskBadge}</div></div>${reasonHtml}${dimsHtml}<div>${stepsHtml}</div>`;
}
function renderAgentTrajectory(agent,taskIndex,agentIndex,selectedTaskPosition){
    const taskTree=getAgentTaskTree(agent);
    if(taskTree.length){
        const summary=getAgentPassSummary(agent);
        const mainIndex=(Number.isInteger(state.selectedAgentMainTask)&&state.selectedAgentMainTask>=0&&state.selectedAgentMainTask<taskTree.length)?state.selectedAgentMainTask:0;
        const mainTask=taskTree[mainIndex]||taskTree[0];
        const subtasks=Array.isArray(mainTask&&mainTask.subtasks)?mainTask.subtasks:[];
        const subIndex=(Number.isInteger(state.selectedAgentSubtask)&&state.selectedAgentSubtask>=0&&state.selectedAgentSubtask<subtasks.length)?state.selectedAgentSubtask:0;
        const selectedSubtask=subtasks[subIndex]||null;
        const mainNav=taskTree.map((item,index)=>{
            const itemStatus=getTaskLikeStatus(item);
            const subCount=Array.isArray(item.subtasks)?item.subtasks.length:0;
            const coverage=Array.isArray(item.dimension_coverage)&&item.dimension_coverage.length?item.dimension_coverage.join(', '):'';
            return `<div class="tree-main-card${index===mainIndex?' active':''}" onclick="selectAgentMainTask(${taskIndex},${agentIndex},${index})">
              <div class="tree-main-origin">${esc(item.origin||'query_specific')}</div>
              <div class="tree-main-title">${esc(item.title||item.main_task_id||`Main Task ${index+1}`)}</div>
                                                        <div class="tree-main-meta">${statusBadge(itemStatus)}<span class="chip">${subCount} subtasks</span>${typeof item.pass_rate==='number'?`<span class="chip">pass rate ${rateText(item.pass_rate)}</span>`:''}${typeof item.score==='number'?`<span class="chip">score ${scoreText(item.score,item.max_score)}</span>`:''}${coverage?`<span class="chip">${esc(coverage)}</span>`:''}</div>
            </div>`;
        }).join('');
        const subtaskNav=subtasks.length?subtasks.map((subtask,index)=>{
            const subStatus=getTaskLikeStatus(subtask);
            const stepCount=getTaskTrajectorySteps(subtask).length;
            const basedOn=subtask&&typeof subtask.based_on==='object'?subtask.based_on:{};
            const areaLabel=basedOn.area_label||subtask.goal||'';
            return `<div class="tree-subtask-pill${index===subIndex?' active':''}" onclick="selectAgentSubtask(${taskIndex},${agentIndex},${mainIndex},${index})">
              <div class="tree-subtask-title">${esc(subtask.title||subtask.task_id||`Subtask ${index+1}`)}</div>
                                                        <div class="tree-subtask-meta">${statusBadge(subStatus)}<span>${stepCount} steps</span>${areaLabel?`<span>${esc(areaLabel)}</span>`:''}</div>
            </div>`;
        }).join(''):'<div class="empty">No subtasks recorded for this main task.</div>';
                                return `<div class="traj-sticky"><div class="traj-summary"><div class="traj-summary-top"><div class="traj-summary-main"><div class="label">Agent</div><div class="traj-agent-name">${esc(agent.agent_id||agent.role||'unknown')}</div><div class="traj-meta-row"><span class="chip">agent score ${summary.score==null?'-':scoreText(summary.score,summary.maxScore)}</span>${summary.passRate==null?'':`<span class="chip">pass rate ${rateText(summary.passRate)}</span>`}<span class="chip">${summary.passed}/${summary.evaluated||0} passed</span><span class="chip">${summary.mainTaskCount} main tasks</span><span class="chip">${summary.subtaskCount} subtasks</span></div></div></div></div></div><div class="tree-layout"><div class="tree-rail">${mainNav}</div><div class="tree-stage-shell"><div class="tree-stage-header"><div class="label">Selected Main Task</div><div class="tree-stage-title">${esc(mainTask.title||mainTask.main_task_id||'-')}</div><div class="traj-meta-row">${statusBadge(getTaskLikeStatus(mainTask))}${typeof mainTask.pass_rate==='number'?`<span class="chip">pass rate ${rateText(mainTask.pass_rate)}</span>`:''}${typeof mainTask.score==='number'?`<span class="chip">score ${scoreText(mainTask.score,mainTask.max_score)}</span>`:''}${mainTask.main_task_id?`<span class="chip">${esc(mainTask.main_task_id)}</span>`:''}${mainTask.origin?`<span class="chip">${esc(mainTask.origin)}</span>`:''}</div><div class="tree-stage-goal">${esc(mainTask.goal||'No goal recorded.')}</div></div><div><div class="section" style="padding-left:0;padding-right:0"><div class="label">Subtasks</div></div><div class="tree-subtasks">${subtaskNav}</div></div>${selectedSubtask?renderTreeSubtaskTrajectory(mainTask,selectedSubtask,taskIndex,agentIndex,mainIndex,subIndex):'<div class="empty">Select a subtask to inspect trajectory.</div>'}</div></div>`;
    }
    const tasks=getAgentTasks(agent);
    if(tasks.length){
        const agentLabel=esc(agent.agent_id||agent.role||'unknown');
        const normalizedIndex=(Number.isInteger(selectedTaskPosition)&&selectedTaskPosition>=0&&selectedTaskPosition<tasks.length)?selectedTaskPosition:0;
        const selectedTask=tasks[normalizedIndex]||tasks[0];
        const selectedStatus=selectedTask.status||selectedTask.verdict||(selectedTask.passed===true?'passed':(selectedTask.passed===false?'failed':'partial'));
        const selectedSteps=getTaskTrajectorySteps(selectedTask).length;
        const taskNav=tasks.map((task,taskPosition)=>{
            const taskStatus=task.status||task.verdict||(task.passed===true?'passed':(task.passed===false?'failed':'partial'));
            const taskSteps=getTaskTrajectorySteps(task);
                        return `<div class="traj-task-pill${taskPosition===normalizedIndex?' active':''}" onclick="selectAgentTask(${taskIndex},${agentIndex},${taskPosition})">
                            <div class="traj-task-pill-title">${esc(task.title||task.task_id||`Task ${taskPosition+1}`)}</div>
                            <div class="traj-task-pill-meta"><span>${taskSteps.length} steps</span>${typeof task.completion_score==='number'?`<span>score ${scoreText(task.completion_score)}</span>`:''}<span>${esc(taskStatus)}</span></div>
            </div>`;
        }).join('');
                return `<div class="traj-sticky"><div class="traj-summary"><div class="traj-summary-top"><div class="traj-summary-main"><div class="label">Agent</div><div class="traj-agent-name">${agentLabel}</div><div class="label" style="margin-top:4px">Current Task</div><div class="traj-task-name">${esc(selectedTask.title||selectedTask.task_id||`Task ${normalizedIndex+1}`)}</div><div class="traj-meta-row">${statusBadge(selectedStatus)}${typeof selectedTask.completion_score==='number'?`<span class="chip">score ${scoreText(selectedTask.completion_score)}</span>`:''}<span class="chip">${selectedSteps} steps</span>${selectedTask.task_id?`<span class="chip">${esc(selectedTask.task_id)}</span>`:''}</div></div></div></div><div class="traj-task-switcher">${taskNav}</div></div><div class="traj-detail">${renderSingleTaskTrajectory(selectedTask,taskIndex,agentIndex,normalizedIndex)}</div>`;
    }
  const trajectory=agent.trajectory&&typeof agent.trajectory==='object'?agent.trajectory:{};
  const steps=Array.isArray(trajectory.steps)?trajectory.steps:[];
  const agentLabel=esc(agent.agent_id||agent.role||'unknown');
  if(!steps.length){
    const ref=agent.trajectory_ref;
    return `<div class="section"><div class="label">Agent: ${agentLabel}</div><div class="value">${ref?`Ref: ${esc(ref)}`:'No trajectory steps recorded.'}</div></div>`;
  }
  const groups=[];
  const groupMap=new Map();
  for(const step of steps){
    const key=step.task_id||'__default__';
    const label=step.task_title||step.task_id||'Steps';
    if(!groupMap.has(key)){const g={key,label,steps:[]};groups.push(g);groupMap.set(key,g);}
    groupMap.get(key).steps.push(step);
  }
  let html=`<div class="section"><div class="label">Agent: ${agentLabel} · ${groups.length} task group${groups.length!==1?'s':''}</div></div>`;
  groups.forEach((group,gi)=>{
    const groupId=`traj-g-${taskIndex}-${agentIndex}-${gi}`;
    const stepsHtml=group.steps.map((step,si)=>{
      const args=step.args||step.tool_args||{};
      const argsId=`traj-a-${taskIndex}-${agentIndex}-${gi}-${si}`;
      const hasArgs=args&&typeof args==='object'&&Object.keys(args).length>0;
      const successIcon=step.success===true?`<span class="ts-success ok">✓</span>`:step.success===false?`<span class="ts-success fail">✗</span>`:'';
    const screenshotSrc=buildScreenshotSrc(step);
    const screenshotHtml=screenshotSrc?`<div class="ts-screenshot"><img src="${esc(screenshotSrc)}" alt="screenshot" onclick="window.open(this.src)" title="Click for full size"/></div>`:'';
      return `<div class="traj-step">
        <div class="ts-hdr">
          <span class="ts-idx">#${si+1}</span>
          <span class="ts-tool">${esc(step.tool_name||'unknown')}</span>
          ${successIcon}
          ${step.timestamp?`<span class="ts-time">${esc(step.timestamp)}</span>`:''}
        </div>
        ${hasArgs?`<div class="ts-args-toggle" onclick="document.getElementById('${argsId}').classList.toggle('collapsed')">▼ args</div><pre id="${argsId}" class="ts-args collapsed">${esc(JSON.stringify(args,null,2))}</pre>`:''}
        ${step.result||step.observation?`<div class="ts-result"><b>Result:</b> ${esc(String(step.result||step.observation))}</div>`:''}
        ${screenshotHtml}
      </div>`;
    }).join('');
    html+=`<div class="traj-group open" id="${groupId}">
      <div class="traj-group-hdr" onclick="document.getElementById('${groupId}').classList.toggle('open')">
        <span>${esc(group.label)}</span>
        <span class="traj-group-count">${group.steps.length} steps</span>
        <span class="traj-expand">▶</span>
      </div>
      <div class="traj-group-body">${stepsHtml}</div>
    </div>`;
  });
  return html;
}
function renderTasks(tasks){
  const container=document.getElementById('tasks');
  if(!tasks.length){container.innerHTML='<div class="empty">No tasks found in this report.</div>';return;}
  container.innerHTML=tasks.map(task=>`
    <div class="task-item${state.selected===task.index?' active':''}" data-index="${task.index}">
      <div class="task-top"><div class="task-id">${esc(task.id)}</div>${taskScoreBadge(task)}</div>
      <div class="task-query">${esc(task.query||task.reason||'No query provided.')}</div>
      <div class="task-meta">
        <span>agents ${esc(task.agents_completed??'-')}/${esc(task.agents_total??'-')}</span>
        ${task.has_preview?'<span>preview</span>':''}
      </div>
    </div>`).join('');
}
function renderAudit(entries){
  const container=document.getElementById('audit');
  document.getElementById('audit-count').textContent=entries.length;
  if(!entries.length){container.innerHTML='<div class="empty">No audit log loaded.</div>';return;}
  const recent=entries.slice(-40).reverse();
  container.innerHTML=recent.map(entry=>`
    <div class="audit-entry">
      <div class="audit-ts">${esc(entry.ts||'-')}</div>
      <div class="audit-cmd">${esc(entry.cmd||'-')}</div>
      <div class="audit-meta"><span>risk ${esc(entry.risk||'-')}</span><span>cwd ${esc(entry.cwd||'-')}</span></div>
    </div>`).join('');
}
function renderPagination(){
  const bar=document.getElementById('tasks-pagination');
  if(!bar)return;
  const total=state.taskTotal||0;
  if(total<=PAGE_SIZE){
    bar.classList.add('hidden');
    return;
  }
  bar.classList.remove('hidden');
  const offset=state.taskOffset||0;
  const curPage=Math.floor(offset/PAGE_SIZE)+1;
  const totalPages=Math.ceil(total/PAGE_SIZE);
  const prevDisabled=offset<=0?' disabled':'';
  const nextDisabled=offset+PAGE_SIZE>=total?' disabled':'';
  bar.innerHTML=`
    <button class="pagination-btn" onclick="loadTaskPage(0)"${prevDisabled} title="First">&laquo;</button>
    <button class="pagination-btn" onclick="loadTaskPage(${Math.max(0,offset-PAGE_SIZE)})"${prevDisabled}>&lsaquo; Prev</button>
    <span class="pagination-info">Page ${curPage}/${totalPages} (${total} tasks)</span>
    <button class="pagination-btn" onclick="loadTaskPage(${offset+PAGE_SIZE})"${nextDisabled}>Next &rsaquo;</button>
    <button class="pagination-btn" onclick="loadTaskPage(${(totalPages-1)*PAGE_SIZE})"${nextDisabled} title="Last">&raquo;</button>`;
}
function buildTaskUrl(offset){
  let url=`/api/tasks?offset=${offset}&limit=${PAGE_SIZE}`;
  if(state.sortBy!=='index')url+=`&sort_by=${state.sortBy}`;
  if(state.sortDir!=='asc')url+=`&sort_dir=${state.sortDir}`;
  if(state.filterQuery)url+=`&filter_query=${encodeURIComponent(state.filterQuery)}`;
  if(state.filterModel)url+=`&filter_model=${encodeURIComponent(state.filterModel)}`;
  if(state.filterScoreMin!=='')url+=`&filter_score_min=${state.filterScoreMin}`;
  if(state.filterScoreMax!=='')url+=`&filter_score_max=${state.filterScoreMax}`;
  return url;
}
function syncToolbarFromState(){
  document.getElementById('sort-by').value=state.sortBy;
  document.getElementById('sort-dir').value=state.sortDir;
  document.getElementById('filter-query').value=state.filterQuery;
  document.getElementById('filter-model').value=state.filterModel;
  document.getElementById('filter-score-min').value=state.filterScoreMin;
  document.getElementById('filter-score-max').value=state.filterScoreMax;
}
function applySortFilter(){
  state.sortBy=document.getElementById('sort-by').value;
  state.sortDir=document.getElementById('sort-dir').value;
  state.filterQuery=document.getElementById('filter-query').value;
  state.filterModel=document.getElementById('filter-model').value;
  state.filterScoreMin=document.getElementById('filter-score-min').value;
  state.filterScoreMax=document.getElementById('filter-score-max').value;
  loadTaskPage(0);
}
function clearSortFilter(){
  state.sortBy='index';state.sortDir='asc';state.filterQuery='';state.filterModel='';state.filterScoreMin='';state.filterScoreMax='';
  syncToolbarFromState();
  loadTaskPage(0);
}
async function loadTaskPage(offset){
  const url=buildTaskUrl(offset);
  const data=await fetchJson(url);
  const normalized=normalizeTasksPayload(data);
  const tasks=normalized.tasks;
  state.taskOffset=offset;
  state.taskTotal=data.total??tasks.length;
  state.tasksPayload=data;
  renderTasks(tasks);
  renderPagination();
  if(state.selected!==null){
    const inPage=tasks.some(task=>task.index===state.selected);
    if(!inPage){
      document.getElementById('agents').innerHTML='<div class="empty">Select a task to inspect agent results.</div>';
      document.getElementById('trajectory').innerHTML='<div class="empty">Click an agent to view its trajectory.</div>';
      document.getElementById('traj-header').textContent='';
    }
  }
  if(state.selected===null&&tasks.length){
    await selectTask(tasks[0].index);
  }
}
async function loadModels(){
  try{
    const payload=await fetchJson('/api/models');
    const models=Array.isArray(payload.models)?payload.models:[];
    const sel=document.getElementById('filter-model');
    if(!sel)return;
    sel.innerHTML='<option value="">All</option>'+models.map(m=>`<option value="${esc(m)}">${esc(m)}</option>`).join('');
  }catch(_){}
}
// renderPreview removed: preview column replaced by trajectory column
async function refresh(){
        clearUiError();
    const rawBtn=document.getElementById('raw-load-btn');
    if(rawBtn)rawBtn.disabled=state.selected===null && !state.currentTaskPayload;

    let auditEntries=[];
    try{
        const auditPayload=await fetchJson('/api/audit');
        auditEntries=Array.isArray(auditPayload)?auditPayload:[];
    }catch(_err){
        auditEntries=[];
    }
    state.audit=auditEntries;
    renderAudit(state.audit);

    // Refresh current page to get updated stats without losing position
    const data=await fetchJson(buildTaskUrl(state.taskOffset||0));
    const normalized=normalizeTasksPayload(data);
    const tasks=normalized.tasks;
    const summary=normalized.summary||{};
    const auditSummary=normalized.audit_summary||{};
    const meta=normalized.meta||{};
    state.taskTotal=data.total??tasks.length;
    state.tasksPayload=data;

    const existsLabel=meta.exists===false?'missing':'ok';
    const resultsCount=meta.results_count ?? state.taskTotal;
    document.getElementById('meta').textContent=`report ${existsLabel} | rows ${resultsCount} | ${meta.report_path||'-'} | audit ${meta.audit_path||'disabled'} | ${new Date().toLocaleTimeString()}`;
  document.getElementById('s-total').textContent=summary.total??state.taskTotal;
  document.getElementById('s-pass').textContent=summary.passed??'-';
  document.getElementById('s-partial').textContent=summary.partial??'-';
  document.getElementById('s-fail').textContent=summary.failed??'-';
  document.getElementById('s-audit').textContent=auditSummary.total??state.audit.length;
  document.getElementById('task-count').textContent=state.taskTotal;
  renderTasks(tasks);
  renderPagination();

    if(meta.exists===false){
        setUiError(`report file not found: ${meta.report_path||'-'}`);
        return;
    }

    if(!tasks.length && state.taskTotal===0){
        document.getElementById('agents').innerHTML=`<div class="empty">Report loaded, but it contains 0 results.\nPath: ${esc(meta.report_path||'-')}</div>`;
        document.getElementById('trajectory').innerHTML='<div class="empty">No trajectory available.</div>';
        state.currentTaskPayload=null;
        setRawPlaceholder('Raw JSON unavailable because the report contains 0 results.');
        if(rawBtn)rawBtn.disabled=true;
        return;
    }

    const hasSelectedTask=state.selected!==null;
    if(tasks.length && !hasSelectedTask){
        await selectTask(tasks[0].index);
    } else if(state.selected!==null && !hasSelectedTask){
        state.selected=null;
        state.selectedAgent=null;
        state.currentTaskPayload=null;
        document.getElementById('agents').innerHTML='<div class="empty">Select a task to inspect agent results.</div>';
        document.getElementById('trajectory').innerHTML='<div class="empty">Click an agent to view its trajectory.</div>';
        document.getElementById('traj-header').textContent='';
        setRawPlaceholder();
        if(rawBtn)rawBtn.disabled=true;
    }
}

async function selectTask(index,scroll=true){
  if(state.selected!==index){state.selectedAgent=null;}
  state.selected=index;
  document.querySelectorAll('.task-item').forEach(el=>el.classList.toggle('active',Number(el.dataset.index)===index));
  const data=await fetchJson(`/api/tasks/${index}`);
  state.currentTaskPayload=data;
  state.rawLoadedIndex=null;
  const row=data.row||{};
  const report=data.report||{};
  const summary=report.summary||{};
  const artifacts=report.artifacts||{};
  const buildResult=report.build_result||{};
  const buildVerdict=buildResult.verdict||buildResult.verdicts||{};
  const link=data.link||null;
  const runtimePaths=report.runtime_paths||{};
  const runtimePathLines=[];
  if(runtimePaths.workspace_root)runtimePathLines.push(`Workspace: ${runtimePaths.workspace_root}`);
  if(runtimePaths.artifacts_path)runtimePathLines.push(`Artifacts JSON: ${runtimePaths.artifacts_path}`);
  if(runtimePaths.process_log_dir)runtimePathLines.push(`Process Logs: ${runtimePaths.process_log_dir}`);
  const chips=[];
  if(typeof summary.agents_total==='number')chips.push(`agents ${summary.agents_completed??'-'}/${summary.agents_total}`);
        const scoreValue=typeof summary.overall_score==='number'?summary.overall_score:(typeof summary.avg_task_completion_score==='number'?summary.avg_task_completion_score:summary.avg_main_task_pass_rate);
        const scoreMax=typeof summary.overall_max_score==='number'?summary.overall_max_score:null;
        if(typeof scoreValue==='number')chips.push(`${typeof summary.overall_score==='number'?'total score':'score'} ${scoreText(scoreValue,scoreMax)}`);
  if(runtimePaths.process_log_dir)chips.push(`logs ${runtimePaths.process_log_dir}`);
  const ports=Array.isArray(artifacts.ports)?artifacts.ports:[];
    const portList=ports.length?ports.map(port=>`${port.name||'app'}: ${port.url||port.port||'-'}`).join('\\n'):'No declared ports.';
  const pipelineHtml=renderPipelineHandoffs(report);
  // Store agents for trajectory panel
  state.currentAgents=Array.isArray(report.agents)?report.agents:[];
  state.currentTaskIndex=index;
  // Col 2: task header + agent overview
  const taskHeader=`
    <div class="section"><div class="label">Task</div><div class="value mono">${esc(row.id||row.sample_id||`row-${index}`)}</div></div>
    <div class="section"><div class="label">Status</div>${statusBadge(data.derived_status||'failed')}</div>
    <div class="section"><div class="label">Query</div><div class="quote">${esc(row.query||'No query provided.')}</div></div>
    <div class="section"><div class="label">Reason</div><div class="quote">${esc(data.reason||'No explicit reason recorded.')}</div></div>
    <div class="section"><div class="label">Preview</div><div class="value">${link?`<a class="link" href="${esc(link)}" target="_blank" rel="noopener">${esc(link)}</a>`:'Unavailable'}</div></div>
    <div class="section"><div class="label">Build</div>${statusBadge(data._build_status||'passed')}<div class="quote">${esc(data._build_reason||'No build verdict available.')}</div></div>
    <div class="section"><div class="label">Agents</div></div>`;
  document.getElementById('agents').innerHTML=taskHeader+renderAgentOverview(report.agents,index);
  // If a previous agent was selected, restore it; otherwise clear trajectory
  if(state.selectedAgent!==null&&state.currentAgents.length>state.selectedAgent){
    selectAgent(index,state.selectedAgent);
  } else {
    document.getElementById('trajectory').innerHTML='<div class="empty">Click an agent card to view its trajectory.</div>';
    document.getElementById('traj-header').textContent='';
  }
    setRawPlaceholder(`Raw JSON ready for task ${index}. Click Load to render it.`);
    const rawBtn=document.getElementById('raw-load-btn');
    if(rawBtn)rawBtn.disabled=false;
  if(scroll){document.getElementById('agents').scrollTop=0;}
    document.getElementById('trajectory').scrollTop=0;
}
window.__openSelectTask=index=>selectTask(index).catch(err=>setUiError(err instanceof Error?err.message:String(err)));
document.getElementById('tasks').addEventListener('click',event=>{
    const item=event.target.closest('.task-item');
    if(!item){
        return;
    }
    const index=Number(item.dataset.index);
    if(Number.isNaN(index)){
        setUiError('frontend error: clicked task is missing a numeric data-index');
        return;
    }
    window.__openSelectTask(index);
});
window.addEventListener('error',event=>{
    setUiError(`frontend error: ${event.message}`);
});
window.addEventListener('unhandledrejection',event=>{
    const reason=event.reason instanceof Error?event.reason.message:String(event.reason);
    setUiError(`request failed: ${reason}`);
});
loadModels();
refresh().catch(err=>setUiError(err instanceof Error?err.message:String(err)));
setInterval(()=>refresh().catch(err=>setUiError(err instanceof Error?err.message:String(err))),5000);
</script>
</body>
</html>"""
        content = content.replace("__INITIAL_META__", html.escape(initial_meta), 1)
        content = content.replace("__INITIAL_ERROR_CLASS__", "visible" if initial_error else "", 1)
        content = content.replace("__INITIAL_ERROR__", html.escape(initial_error), 1)
        content = content.replace("__INITIAL_TOTAL__", str(summary.get("total", 0)), 1)
        content = content.replace("__INITIAL_PASSED__", str(summary.get("passed", 0)), 1)
        content = content.replace("__INITIAL_PARTIAL__", str(summary.get("partial", 0)), 1)
        content = content.replace("__INITIAL_FAILED__", str(summary.get("failed", 0)), 1)
        content = content.replace("__INITIAL_AUDIT_TOTAL__", str(audit_summary.get("total", 0)), 1)
        content = content.replace("__INITIAL_TASK_COUNT__", str(results_count), 1)
        content = content.replace("__INITIAL_TASKS__", initial_tasks, 1)
        content = content.replace("__INITIAL_AGENTS__", initial_agents, 1)
        content = content.replace("__INITIAL_TRAJECTORY__", initial_trajectory, 1)
        content = content.replace("__SHARED_AGENT_DETAIL_CSS__", SHARED_AGENT_DETAIL_CSS, 1)
        content = content.replace("__SHARED_AGENT_DETAIL_JS__", SHARED_AGENT_DETAIL_JS, 1)
        content = content.replace("__PAGE_SIZE__", str(page_size), 1)
        content = content.replace(
            "__INITIAL_RAW__",
            html.escape(initial_raw),
            1,
        )

        return HTMLResponse(
            content=content,
            headers={
                "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
                "Pragma": "no-cache",
                "Expires": "0",
            },
        )

    return app


async def main() -> int:
    import uvicorn

    parser = argparse.ArgumentParser(description="Open evaluation tree-aware web dashboard")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8022)
    parser.add_argument("--reload", action="store_true")
    parser.add_argument(
        "--report-path",
        default="artifacts/reports/open_report.json",
        help="Path to eval_open.py --output JSONL (default .db path: <report_path>.db)",
    )
    parser.add_argument(
        "--audit-log",
        metavar="PATH",
        help="Path to audit log JSONL produced by eval_open.py --audit-log",
    )
    parser.add_argument(
        "--page-size",
        type=int,
        default=20,
        help="Number of tasks to load per page (default: 20)",
    )
    args = parser.parse_args()

    report_path = Path(args.report_path)
    audit_path = Path(args.audit_log) if args.audit_log else None
    app = create_app(report_path, audit_path, page_size=args.page_size)

    logger.info("Starting open dashboard at http://%s:%d", args.host, args.port)
    logger.info("Reading report from: %s", report_path)
    if audit_path:
        logger.info("Reading audit log from: %s", audit_path)

    try:
        cfg = uvicorn.Config(app, host=args.host, port=args.port, reload=args.reload, log_level="info")
        await uvicorn.Server(cfg).serve()
        return 0
    except KeyboardInterrupt:
        return 0
    except Exception as exc:
        logger.error("Web server failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
