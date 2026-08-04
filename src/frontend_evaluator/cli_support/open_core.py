from __future__ import annotations

import asyncio
import json
import os
import random
import re
import shutil
import signal
import subprocess
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

from frontend_evaluator.agent.agent_registry import AgentRegistry
from frontend_evaluator.agent.config import AgentConfig
from frontend_evaluator.agent.orchestrator import AgenticOrchestrator
from frontend_evaluator.agent.pipeline import AgentHandoff, build_dependency_layers
from frontend_evaluator.agent.report import build_agentic_report, normalize_agent_result
from frontend_evaluator.events.emitter import EventEmitter
from frontend_evaluator.open_monitor import build_monitor_task_descriptors
from frontend_evaluator.planner.task_planner import (
    MainTaskSpec,
    build_fixed_dimension_planned_tasks,
)
from frontend_evaluator.sandbox.cdp_playwright_executor import CdpPlaywrightExecutor
from frontend_evaluator.tools import session as _tool_session
from frontend_evaluator.tools.page_inspector import get_page_context
from frontend_evaluator.llm.retry import acquire_cdp_permit
from frontend_evaluator.utils.fixed_task_db import FixedTaskDB
from frontend_evaluator.utils.logger import setup_logger
from frontend_evaluator.utils.subprocess import close_subprocess_transports

from .common import maybe_await
from .row_priority_gate import RowPriorityGate

if TYPE_CHECKING:
    from frontend_evaluator.planner.task_planner import PlannedTask
    from frontend_evaluator.utils.config import Config


logger = setup_logger("eval_open")


def _terminate_process_group(pid: int, *, name: str = "process", grace_s: float = 2.0) -> None:
    """Kill *pid* and ALL its children by signalling the process group.

    Chromium spawns ~7-10 child processes (zygote, renderer, GPU, network, etc.).
    A plain ``os.kill(pid, SIGTERM)`` leaves those children as orphans which
    keep their file descriptors and inotify watches open until the kernel
    reaps them — quickly exhausting ``fs.inotify.max_user_instances`` (default
    128) across a batch of evaluations.

    Requires the process to have been started with ``start_new_session=True``
    so that pid == pgid.
    """
    try:
        os.killpg(pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    except PermissionError:
        # Not in our session — fall back to single-process kill
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            return
    # Give the process a moment to flush; chromium otherwise leaves zombie children.
    deadline = time.monotonic() + grace_s
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)  # probe: still alive?
        except ProcessLookupError:
            return
        time.sleep(0.1)
    # Still alive — SIGKILL the group
    try:
        os.killpg(pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    except PermissionError:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            return
    logger.debug("[cleanup] Force-killed %s pgid=%d (SIGKILL after %.0fs)", name, pid, grace_s)


# ---------------------------------------------------------------------------
# Health-monitoring helpers used by the batched-processing loop below.
# ---------------------------------------------------------------------------


def _slim_build_result_for_report(build_result: Dict[str, Any]) -> Dict[str, Any]:
    """Return a copy of *build_result* with trajectory/tasks/task_tree removed.

    ``build_result`` is the same agent object that already appears in
    ``combined_report["agents"]`` (as the build_engineer entry). The web
    visualizer (`scripts/visualization/eval_open_web.py`) only reads
    ``build_result.verdict`` / ``build_result.verdicts`` — never the
    trajectory/tasks/task_tree under it. Duplicating those fields across
    six paths (build_result.trajectory, build_result.tasks[*].trajectory,
    build_result.task_tree[*].subtasks[*].trajectory, plus the same three
    under agents[0]) inflates results 6x; for builds with MB-scale npm
    install stdout in step results this exploded single rows to 290-440 MB.

    Returns a new dict; the original ``build_result`` is unchanged (it's
    still the canonical entry in the agents array).
    """
    if not isinstance(build_result, dict):
        return build_result
    # Fields the web UI consumes via _extract_build_info — keep verbatim.
    keep = {
        "agent_id", "status", "runtime", "end_reason",
        "verdict", "verdicts",
        "score", "score_scale", "score_max", "agent_max_score",
        "task_synthesis_mode", "aggregation_mode",
        "task_completion_score", "legacy_task_completion_score",
        "main_task_pass_rate", "main_task_completion_score",
        "main_task_weight", "main_task_score_total",
        "main_tasks_total", "main_tasks_evaluated", "main_tasks_passed",
        "main_task_status_counts", "legacy_score",
        "dimensions", "trajectory_ref",
    }
    return {k: v for k, v in build_result.items() if k in keep}


def _slim_agent_for_tree_mode(agent: Dict[str, Any]) -> Dict[str, Any]:
    """Drop redundant trajectory copies inside an agent that ran in tree mode.

    In tree mode (`TASK_SYNTHESIS_MODE=tree`) every executed step lives under
    ``agent.task_tree[*].subtasks[*].trajectory.steps`` — that is what the web
    UI's tree view actually renders (``renderTreeSubtaskTrajectory``).

    Two other fields hold the *same* steps as fallbacks for non-tree mode:

      - ``agent.tasks[*].trajectory.steps``     ← used by the flat task view
      - ``agent.trajectory.steps``              ← derived in ``report.py`` from
                                                  tasks; legacy fallback only

    In tree mode the frontend never enters those branches, so we keep each
    container (status/duration_ms/etc.) but null out ``.steps`` to save ~2/3
    of the agent payload.
    """
    if not isinstance(agent, dict):
        return agent
    mode = str(agent.get("task_synthesis_mode") or "").strip().lower()
    if mode != "tree":
        return agent

    slimmed = dict(agent)
    redirect_hint = "task_tree[*].subtasks[*].trajectory.steps"

    tasks = slimmed.get("tasks")
    if isinstance(tasks, list):
        new_tasks = []
        for t in tasks:
            if isinstance(t, dict):
                t2 = dict(t)
                traj = t2.get("trajectory")
                if isinstance(traj, dict) and traj.get("steps"):
                    traj2 = dict(traj)
                    traj2["steps"] = []
                    traj2["_steps_in"] = redirect_hint
                    t2["trajectory"] = traj2
                new_tasks.append(t2)
            else:
                new_tasks.append(t)
        slimmed["tasks"] = new_tasks

    traj = slimmed.get("trajectory")
    if isinstance(traj, dict) and traj.get("steps"):
        traj2 = dict(traj)
        traj2["steps"] = []
        traj2["_steps_in"] = redirect_hint
        slimmed["trajectory"] = traj2

    return slimmed


def _slim_combined_report(combined_report: Dict[str, Any]) -> Dict[str, Any]:
    """Apply all trajectory dedup transforms to a finalized combined report.

    Called right before the report is returned for persistence. Mutates the
    dict in place and returns it for convenience.
    """
    if not isinstance(combined_report, dict):
        return combined_report
    if "build_result" in combined_report:
        combined_report["build_result"] = _slim_build_result_for_report(combined_report["build_result"])
    agents = combined_report.get("agents")
    if isinstance(agents, list):
        combined_report["agents"] = [_slim_agent_for_tree_mode(a) for a in agents]
    return combined_report




def _get_process_memory_percent() -> float:
    """RSS memory of this process as a percentage of total physical RAM."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    rss_kb = int(line.split()[1])
                    break
            else:
                return 0.0
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    total_kb = int(line.split()[1])
                    break
            else:
                return 0.0
        return (rss_kb / total_kb) * 100.0
    except (OSError, ValueError, IndexError):
        return 0.0


class _noop_semaphore:
    """Async context manager that does nothing — fallback when no semaphore."""

    async def __aenter__(self) -> None:
        pass

    async def __aexit__(self, *_: Any) -> None:
        pass


# Threshold (chars). Above this, raw_markdown is written to a workspace file
# and the build_engineer reads it via local_fs_read(start_line=…, end_line=…)
# instead of receiving the whole blob inline. ~8K tokens at the ~4 chars/token
# ratio we saw on the rebuilt fixture — beyond that the LLM starts truncating
# when it has to retype code through local_fs_write tool args.
_BUILD_RAW_MARKDOWN_FILE_THRESHOLD_CHARS = 32_000


def _write_raw_markdown_pointer(
    *,
    workspace: str,
    raw_markdown: str,
) -> Optional[Dict[str, Any]]:
    """Materialise raw_markdown into the workspace and emit a small index.

    Returns a dict with {path, index_path, total_lines, size_bytes, header_hints,
    fence_lines} when the file-pointer path is engaged; returns None when the
    body is short enough to stay inline.

    Index format (.model_reply.index.json):
      {
        "size_bytes":     int,
        "total_lines":    int,
        "header_hints":   [{"line": int, "text": str}, ...],  # ## / ### lines
        "fence_lines":    [int, ...],                          # ``` opening lines
      }
    The index is intentionally observational — it records WHERE plausible file
    boundaries appear, NOT how to slice them. The build_engineer is expected to
    read around those positions and decide for itself. This is what makes the
    approach robust to token-streaming stutter that would defeat hard slicing.
    """
    if not raw_markdown or len(raw_markdown) < _BUILD_RAW_MARKDOWN_FILE_THRESHOLD_CHARS:
        return None
    ws = Path(workspace)
    ws.mkdir(parents=True, exist_ok=True)
    reply_path = ws / ".model_reply.md"
    index_path = ws / ".model_reply.index.json"
    reply_path.write_text(raw_markdown, encoding="utf-8")

    header_hints: List[Dict[str, Any]] = []
    fence_lines: List[int] = []
    raw_lines = raw_markdown.splitlines()
    for i, line in enumerate(raw_lines, start=1):
        stripped = line.lstrip()
        if stripped.startswith("## ") or stripped.startswith("### "):
            header_hints.append({"line": i, "text": line.rstrip()[:200]})
        if stripped.startswith("```"):
            fence_lines.append(i)

    # Pair adjacent fence lines into (open, close). When the LLM should write a
    # file from a fence, it can read content_start..content_end straight from
    # disk via local_fs_extract_block — no retyping. nearest_header is the most
    # recent "##"/"###" line within 10 lines above the fence (usually a path
    # hint like "### src/components/App.tsx"), and prev_lines gives the 3
    # lines immediately above the open fence for cases where the path is
    # written inline (e.g. "**File: src/App.tsx**").
    fence_pairs: List[Dict[str, Any]] = []
    for j in range(0, len(fence_lines) - 1, 2):
        open_line = fence_lines[j]
        close_line = fence_lines[j + 1]
        open_text = raw_lines[open_line - 1] if open_line - 1 < len(raw_lines) else ""
        lang_tail = open_text.lstrip().lstrip("`").strip()
        lang = lang_tail.split()[0] if lang_tail else ""
        nearest_header = ""
        for h in reversed(header_hints):
            delta = open_line - h["line"]
            if 0 < delta <= 10:
                nearest_header = h["text"]
                break
        prev_start = max(1, open_line - 3)
        prev_lines = [
            raw_lines[k - 1].rstrip()[:200]
            for k in range(prev_start, open_line)
            if k - 1 < len(raw_lines)
        ]
        fence_pairs.append({
            "open_line": open_line,
            "close_line": close_line,
            "lang": lang,
            "content_start": open_line + 1,
            "content_end": close_line - 1,
            "nearest_header": nearest_header,
            "prev_lines": prev_lines,
        })

    total_lines = raw_markdown.count("\n") + (0 if raw_markdown.endswith("\n") else 1)
    size_bytes = len(raw_markdown.encode("utf-8"))

    index = {
        "size_bytes": size_bytes,
        "total_lines": total_lines,
        "header_hints": header_hints,
        "fence_lines": fence_lines,
        "fence_pairs": fence_pairs,
    }
    index_path.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "path": str(reply_path),
        "index_path": str(index_path),
        "total_lines": total_lines,
        "size_bytes": size_bytes,
        "header_hints": header_hints,
        "fence_lines": fence_lines,
        "fence_pairs": fence_pairs,
    }


def _build_build_phase_task_context(
    *,
    workspace: str,
    artifacts_path: str,
    process_log_dir: Path,
    raw_markdown: str,
) -> str:
    lines = [
        f"workspace: {workspace}",
        f"artifacts_path: {artifacts_path}",
        f"session_id: {artifacts_path}",
        "",
        f"process_log_dir: {process_log_dir}",
        "",
        "=== Build Engineer Extraction Rules ===",
        "Do not rely on any pre-parsed file manifest.",
        "Read the original model markdown reply below and extract the code/file snippets yourself.",
        "If the reply contains multiple files or mixed prose and code, segment it mentally and write files only from the code snippets you can justify from the original reply.",
        "If the reply is ambiguous, inspect the original markdown directly before deciding whether a required file is missing.",
    ]
    chromium_bin = str(os.getenv("CHROMIUM_BIN", "") or "").strip()
    if chromium_bin:
        lines.extend([
            "",
            f"browser_cmd: {chromium_bin}",
        ])

    pointer = _write_raw_markdown_pointer(
        workspace=workspace,
        raw_markdown=raw_markdown,
    )
    if pointer is None:
        lines.extend([
            "",
            f"=== Original model markdown reply ===\n{raw_markdown}",
        ])
    else:
        approx_tokens = pointer["size_bytes"] // 4
        fence_pair_count = len(pointer.get("fence_pairs") or [])
        lines.extend([
            "",
            "=== Original model markdown reply (large; stored as file) ===",
            f"path:        {pointer['path']}",
            f"index_path:  {pointer['index_path']}",
            f"total_lines: {pointer['total_lines']}",
            f"size_bytes:  {pointer['size_bytes']} (~{approx_tokens} tokens)",
            f"fence_pairs: {fence_pair_count} (paired ``` open/close offsets in the index)",
            "",
            "Read the file in chunks via local_fs_read with start_line / end_line "
            "(1-based, inclusive). The response appends a truncation hint with the "
            "next start_line when more lines remain — keep calling until you have "
            "covered the file end-to-end.",
            "The index file (JSON) lists header_hints (## / ### lines), fence_lines "
            "(``` opening lines), and fence_pairs — each pair has open_line, "
            "close_line, lang, content_start, content_end, nearest_header, and "
            "prev_lines (the 3 lines immediately above the open fence). Use "
            "fence_pairs to enumerate every code block deterministically.",
            "",
            "WRITING FILES FROM THIS REPLY — MANDATORY:",
            "Do NOT pipe long code through local_fs_write content= (the LLM tool "
            "args silently truncate / paraphrase long blocks). Instead, for every "
            "fence_pair, call:",
            "  local_fs_extract_block(",
            "    source_path=<path above>,",
            "    start_line=<fence_pair.content_start>,",
            "    end_line=<fence_pair.content_end>,",
            "    dest_path=<workspace-relative target file>,",
            "    strip_fence=false,    # content_start/end already exclude fences",
            "    session_id=<session_id>,",
            "  )",
            "Decide the dest_path from nearest_header and prev_lines (look for "
            '"### path/to/File.tsx", "**File: path**", "// path", or a path comment '
            "on the first line of the block). Reserve local_fs_write for small "
            "config tweaks (<30 lines) that you actually need to author or edit "
            "yourself — never use it to re-emit code that already exists verbatim "
            "in the reply.",
            "You MUST cover every fence_pair before deciding the project is "
            "complete. After extraction, list the workspace and confirm every "
            "implied path is on disk before running npm install.",
        ])
    return "\n".join(lines)


def _build_build_repair_task_context(
    *,
    base_context: str,
    build_result: Dict[str, Any],
    artifacts: Dict[str, Any],
) -> str:
    dimension_lines: List[str] = []
    for item in build_result.get("dimensions") if isinstance(build_result.get("dimensions"), list) else []:
        if not isinstance(item, dict):
            continue
        dim_id = str(item.get("dimension_id") or "").strip()
        verdict = str(item.get("verdict") or "unknown").strip()
        reason = str(item.get("reason") or "").strip()
        if dim_id:
            dimension_lines.append(f"- {dim_id}: {verdict} :: {reason}")

    artifacts_block = json.dumps(artifacts or {}, ensure_ascii=False, indent=2)
    lines = [
        base_context,
        "",
        "=== Repair Tail Instructions ===",
        "Build verdicts have already been decided from the original build path. Do not revise or reinterpret those verdicts.",
        "Your only goal now is bounded repair: make a small number of targeted fixes, retry startup, and try to produce ready artifacts for downstream agents.",
        "Fail fast if the project remains fundamentally unrecoverable. You may retry a step after inspecting logs, but do not fall back to invented alternate workflows.",
        "If repair still fails, write failed artifacts with a precise error.",
        "",
        "=== Frozen Build Verdict Summary ===",
        *(dimension_lines or ["- none recorded"]),
        "",
        "=== Latest Artifacts Snapshot ===",
        artifacts_block,
    ]
    return "\n".join(lines)


def _read_artifacts_snapshot(artifacts_path: str) -> Dict[str, Any]:
    try:
        payload = json.loads(Path(artifacts_path).read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {"state": "failed", "error": "artifacts payload is not an object"}
    except Exception as exc:
        return {"state": "failed", "error": f"artifacts.json missing: {exc}"}



def _builder_planned_tasks(agent: AgentConfig) -> List["PlannedTask"]:
    tasks = build_fixed_dimension_planned_tasks(
        agent.rubric.dimensions,
        generated_from="build_fixed_dimension_planning",
    )
    # Build-stage agents must *deploy* code, not evaluate.  The default
    # task_text uses "Evaluate ..." framing which misleads the build
    # engineer into assessing an empty workspace instead of writing files.
    # Override to action-oriented text.  This function is only called for
    # the build-phase agent (line ~1282), so evaluate-stage agents are
    # unaffected.
    for t in tasks:
        dim_id = t.based_on.get("dimension_id", "build_success") if t.based_on else "build_success"
        t.title = f"Deploy for {dim_id}"
        t.task_text = (
            f"Deploy the frontend project described in the original model reply "
            f"into the workspace.  Follow your system_prompt: extract code from "
            f"the markdown, write files with local_fs_write, install deps, start "
            f"the dev server and CDP browser.  Dimension '{dim_id}' passes only "
            f"when the app is served, the page is not blank, CDP is reachable, "
            f"and ready artifacts have been written."
        )
    return tasks


def _resolve_query_specific_main_task_count(agent: AgentConfig, config: "Config") -> int:
    if (agent.stage or "evaluate") == "build":
        return 0
    # NoQST ablation: an explicit FREE_TASK_COUNT=0 env forces zero query-specific
    # tasks even when the agent config pins a non-zero free_task_count (e.g.
    # ui_tester.json's free_task_count=5). Without this, the per-agent value
    # shadows the env and `_load_agent_fixed_query_main_tasks` still injects shared
    # query tasks → the ablation never takes effect. Default behaviour (env unset or
    # non-zero) is unchanged. See experiment_results_coverage_humanagreement.md Part F.
    env_free = os.environ.get("FREE_TASK_COUNT")
    if env_free is not None and env_free.strip() == "0":
        return 0
    if agent.free_task_count is not None:
        return max(0, int(agent.free_task_count))
    return config.free_task_count


def _parse_main_task_specs(payload: Optional[List[Dict[str, Any]]]) -> List[MainTaskSpec]:
    specs: List[MainTaskSpec] = []
    for item in payload or []:
        if not isinstance(item, dict):
            continue
        try:
            spec = MainTaskSpec.from_dict(item)
        except Exception:
            continue
        if not spec.main_task_id.strip() or not spec.title.strip() or not spec.goal.strip():
            continue
        specs.append(spec)
    return specs


def _open_fixed_task_db(config: "Config") -> Optional[FixedTaskDB]:
    db_path = config.fixed_task_db_path
    if not db_path:
        return None
    return FixedTaskDB(db_path)


async def _load_agent_fixed_query_main_tasks(
    *,
    query: str,
    agent: AgentConfig,
    config: "Config",
    fixed_task_db: Optional[FixedTaskDB] = None,
) -> List[MainTaskSpec]:
    if (agent.stage or "evaluate") == "build":
        return []

    normalized_query = str(query or "").strip()
    if not normalized_query:
        return []

    expected_count = _resolve_query_specific_main_task_count(agent, config)
    if expected_count <= 0:
        return []

    if fixed_task_db is not None:
        # Mode A: Try existing agent-specific tasks from DB
        agent_payload = fixed_task_db.get_query_agent_tasks(normalized_query, agent.id)
        agent_specs = _parse_main_task_specs(agent_payload) if agent_payload else []
        if agent_specs:
            return agent_specs
        if agent_payload is not None:
            logger.warning(
                "[OpenCore] Ignoring invalid/empty DB agent query tasks for query='%s', agent='%s'; will generate from scratch",
                normalized_query,
                agent.id,
            )

    # Mode B: No DB or no tasks in DB — return empty list.
    # The tree-based task planner will generate both query-specific main tasks
    # and their subtasks in a single one-shot call.
    return []


async def _gather_planning_page_context(executor: Any) -> Optional[str]:
    try:
        page_context = await get_page_context(executor)
    except Exception:
        logger.exception("Failed to gather planning page context")
        return None

    if not isinstance(page_context, str):
        return None
    normalized = page_context.strip()
    if not normalized or normalized.startswith("Failed to get page context"):
        return None
    return normalized


async def _emit_task_progress(
    callback: Optional[Callable[[str, Dict[str, Any]], Any]],
    task_id: str,
    payload: Dict[str, Any],
) -> None:
    if callback is None:
        return
    awaited = maybe_await(callback(task_id, payload))
    if awaited is not None:
        await awaited


def _build_agent_progress_emitter(
    *,
    task_id: str,
    agent_id: str,
    steps_total: int,
    current_task_title: Optional[str],
    current_task_index: int,
    current_task_total: int,
    callback: Optional[Callable[[str, Dict[str, Any]], Any]],
) -> Optional[EventEmitter]:
    if callback is None:
        return None

    emitter = EventEmitter(task_id=task_id, enabled=True)

    async def _on_event(event: Any) -> None:
        await _emit_task_progress(
            callback,
            task_id,
            {
                "kind": "agent",
                "agent_id": agent_id,
                "status": "running",
                "steps_completed": max(0, int(getattr(event, "iteration", 0) or 0)),
                "steps_total": steps_total,
                "current_task_title": current_task_title,
                "current_task_index": current_task_index,
                "current_task_total": current_task_total,
            },
        )

    emitter.on(_on_event)
    return emitter


def normalize_open_status(value: Any) -> str | None:
    if isinstance(value, bool):
        return "passed" if value else "failed"
    if value is None:
        return None

    normalized = str(value).strip().lower()
    mapping = {
        "true": "passed",
        "false": "failed",
        "success": "passed",
        "completed": "passed",
        "error": "failed",
        "compile_error": "failed",
    }
    if normalized in {"passed", "partial", "failed"}:
        return normalized
    return mapping.get(normalized)


def _summary_progress_ratio(summary: Dict[str, Any]) -> Optional[float]:
    overall_score = summary.get("overall_score")
    overall_max_score = summary.get("overall_max_score")
    if isinstance(overall_score, (int, float)) and isinstance(overall_max_score, (int, float)):
        max_score = float(overall_max_score)
        if max_score > 0:
            return float(overall_score) / max_score

    avg_main_task_pass_rate = summary.get("avg_main_task_pass_rate")
    if isinstance(avg_main_task_pass_rate, (int, float)):
        return float(avg_main_task_pass_rate)

    avg_task_completion = summary.get("avg_task_completion_score")
    if isinstance(avg_task_completion, (int, float)):
        return float(avg_task_completion)

    return None


def _get_render_link(row: Dict[str, Any]) -> str:
    ext_info = row.get("ext_info")
    if isinstance(ext_info, str):
        try:
            parsed = json.loads(ext_info)
            if isinstance(parsed, dict):
                link = parsed.get("render_link", "")
                return str(link) if link else ""
        except (json.JSONDecodeError, TypeError):
            pass
    elif isinstance(ext_info, dict):
        link = ext_info.get("render_link", "")
        return str(link) if link else ""
    return ""


def _build_result_failure_reason(build_result: Dict[str, Any]) -> Optional[str]:
    """Extract a failure reason from a build agent result when the top-level
    ``verdict`` field is missing or ambiguous.  Falls back to inspecting
    per-task results and per-dimension verdicts so that a build agent that
    explicitly marks its own tasks as failed is treated as a build failure."""
    # 1) Per-task passed flags (flat tasks from _builder_planned_tasks).
    task_results = build_result.get("task_results") if isinstance(build_result.get("task_results"), list) else []
    for task in task_results:
        if not isinstance(task, dict):
            continue
        if task.get("passed") is False:
            reason = str(task.get("reason") or "").strip()
            title = str(task.get("title") or task.get("task_id") or "build task")
            return f"Build task '{title}' failed: {reason}" if reason else f"Build task '{title}' failed."

    # 2) Per-dimension verdicts.
    dimensions = build_result.get("dimensions") if isinstance(build_result.get("dimensions"), list) else []
    for dim in dimensions:
        if not isinstance(dim, dict):
            continue
        dim_verdict = str(dim.get("verdict") or "").strip().lower()
        if dim_verdict == "failed":
            dim_id = str(dim.get("dimension_id") or "unknown")
            dim_reason = str(dim.get("reason") or "").strip()
            return f"Build dimension '{dim_id}' failed: {dim_reason}" if dim_reason else f"Build dimension '{dim_id}' failed."

    return None


def derive_open_report_verdict(report: Dict[str, Any]) -> Dict[str, Any]:
    build_result = report.get("build_result")
    if isinstance(build_result, dict):
        build_verdict = build_result.get("verdict")
        if isinstance(build_verdict, dict) and build_verdict.get("passed") is False:
            return {
                "passed": False,
                "status": "failed",
                "reason": str(build_verdict.get("reason") or "Build failed."),
            }
        # Fallback: check task-level and dimension-level failure signals when
        # the agent result is missing a top-level verdict (e.g. legacy
        # orchestrator results before the aggregated verdict was added).
        fallback_reason = _build_result_failure_reason(build_result)
        if fallback_reason:
            return {
                "passed": False,
                "status": "failed",
                "reason": fallback_reason,
            }

    task_total = 0
    task_passed = 0
    has_pass = False
    has_partial = False
    has_failed = False

    agents = report.get("agents")
    if isinstance(agents, list):
        for agent in agents:
            if not isinstance(agent, dict):
                continue

            task_results = agent.get("task_results")
            if isinstance(task_results, list):
                for task in task_results:
                    if not isinstance(task, dict):
                        continue
                    passed = task.get("passed")
                    task_total += 1
                    if passed is True:
                        task_passed += 1
                        has_pass = True
                    elif passed is False:
                        has_failed = True

            dimensions = agent.get("dimensions")
            if isinstance(dimensions, list):
                for dim in dimensions:
                    if not isinstance(dim, dict):
                        continue
                    verdict = normalize_open_status(dim.get("verdict"))
                    if verdict == "passed":
                        has_pass = True
                    elif verdict == "partial":
                        has_partial = True
                    elif verdict == "failed":
                        has_failed = True

    # If build_engineer PASSED but every evaluate agent hit an infra error
    # (status=error, score=None) — e.g. CDP connection lost — no real
    # evaluation ran.  Return "failed" so the whole task is re-run, not
    # "partial" (which would skip re-running the evaluate phase).
    if isinstance(agents, list):
        build_agent = next(
            (ag for ag in agents if isinstance(ag, dict) and ag.get("agent_id") == "build_engineer"),
            None,
        )
        build_passed = (
            build_agent is not None
            and build_agent.get("status") == "completed"
            and build_agent.get("score") is not None
        )
        evaluate_agents = [
            ag for ag in agents
            if isinstance(ag, dict) and ag.get("agent_id") != "build_engineer"
        ]
        if build_passed and evaluate_agents and all(
            ag.get("score") is None
            for ag in evaluate_agents
        ):
            return {
                "passed": False,
                "status": "failed",
                "reason": "Build succeeded but all evaluation agents produced no score (infrastructure error, timeout, or crash). No evaluation was performed.",
            }

    if task_total:
        if task_passed == task_total and not has_failed and not has_partial:
            return {
                "passed": True,
                "status": "passed",
                "reason": "All planned tasks passed.",
            }
        if task_passed > 0 or has_pass or has_partial:
            return {
                "passed": False,
                "status": "partial",
                "reason": "Some planned tasks passed, but the evaluation was not fully successful.",
            }
        return {
            "passed": False,
            "status": "failed",
            "reason": "Planned tasks did not pass.",
        }

    if has_pass and not has_failed and not has_partial:
        return {
            "passed": True,
            "status": "passed",
            "reason": "All reported checks passed.",
        }
    if has_pass or has_partial:
        return {
            "passed": False,
            "status": "partial",
            "reason": "Reported checks are mixed.",
        }

    summary = report.get("summary")
    if isinstance(summary, dict):
        progress_ratio = _summary_progress_ratio(summary)
        if isinstance(progress_ratio, float):
            if progress_ratio >= 0.999:
                return {
                    "passed": True,
                    "status": "passed",
                    "reason": "Summary score reached 100%.",
                }
            if progress_ratio > 0:
                return {
                    "passed": False,
                    "status": "partial",
                    "reason": "Summary score is between 0% and 100%.",
                }

    return {
        "passed": False,
        "status": "failed",
        "reason": "No passing signals were found in the evaluation report.",
    }


class _NullExecutor:
    """Phase 1 stub: build_tools don't need a real executor."""

    app_url = ""

    async def get_context(self) -> Dict[str, Any]:
        return {"diagnostics": {}, "accessibility_tree": {}}

    async def navigate(self, url: str, **kw: Any) -> Dict[str, Any]:
        return {"success": True}

    async def screenshot(self, **kw: Any) -> str:
        return ""


class _LocalCdpStub:
    """Minimal sandbox stub for CdpPlaywrightExecutor when using a local CDP."""

    def __init__(self, cdp_url: str) -> None:
        self._cdp_url = cdp_url

    def get_cdp_url(self) -> str:
        return self._cdp_url

    async def start_browser(self) -> None:
        pass

    def stop(self) -> None:
        pass


def collect_workspace_files(workspace_root: str) -> Dict[str, str]:
    """Collect text source files from the temporary workspace for read-only inspection tools."""
    root = Path(workspace_root)
    collected: Dict[str, str] = {}
    ignored_dirs = {"node_modules", ".next", ".git", "dist", "build", "coverage"}

    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if any(part in ignored_dirs for part in path.parts):
            continue
        if path.stat().st_size > 256_000:
            continue
        try:
            collected[str(path.relative_to(root)).replace(os.sep, "/")] = path.read_text(
                encoding="utf-8"
            )
        except (OSError, UnicodeDecodeError):
            continue

    return collected


def _build_source_inventory(source_files: Dict[str, str]) -> Dict[str, Any]:
    paths = sorted(source_files.keys())
    framework_hint = "unknown"
    if "package.json" in source_files:
        package_json = source_files["package.json"].lower()
        if '"next"' in package_json or "next" in package_json:
            framework_hint = "nextjs"
        elif '"react"' in package_json or "react" in package_json:
            framework_hint = "react"
        else:
            framework_hint = "node"
    elif any(path.endswith(".html") for path in paths):
        framework_hint = "static_html"

    return {
        "file_count": len(source_files),
        "has_package_json": "package.json" in source_files,
        "framework_hint": framework_hint,
        "entry_candidates": [
            path for path in paths if path in {"app/page.tsx", "src/main.tsx", "src/main.jsx", "index.html"}
        ],
    }


def _get_available_memory_mib() -> float:
    """Return available memory in MiB from /proc/meminfo, or a large default."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return float(line.split()[1]) / 1024.0
    except (OSError, ValueError):
        pass
    return 65536.0  # fallback: assume plenty


def _get_inotify_headroom() -> tuple[int, int]:
    """Return (instances_in_use_by_us, max_user_instances) for inotify.

    Inotify is rate-limited per UID by ``/proc/sys/fs/inotify/max_user_instances``
    (default 128). Each chromium / node process opens 1-3 instances; running
    out of headroom crashes them with EMFILE on ``inotify_init1()``.
    """
    try:
        with open("/proc/sys/fs/inotify/max_user_instances") as f:
            max_instances = int(f.read().strip())
    except (OSError, ValueError):
        return (0, 1 << 30)  # unknown — pretend infinite

    try:
        my_uid = os.getuid()
    except AttributeError:
        return (0, max_instances)

    in_use = 0
    try:
        for proc_entry in Path("/proc").iterdir():
            if not proc_entry.name.isdigit():
                continue
            try:
                if proc_entry.stat().st_uid != my_uid:
                    continue
                fd_dir = proc_entry / "fd"
                if not fd_dir.is_dir():
                    continue
                for fd_link in fd_dir.iterdir():
                    try:
                        if os.readlink(str(fd_link)) == "anon_inode:inotify":
                            in_use += 1
                    except (OSError, ValueError):
                        continue
            except (PermissionError, OSError):
                continue
    except OSError:
        return (0, max_instances)
    return (in_use, max_instances)


def _get_system_fd_headroom() -> tuple[int, int]:
    """Return (in_use, max) for kernel-wide file descriptors via /proc/sys/fs/file-nr."""
    try:
        with open("/proc/sys/fs/file-nr") as f:
            parts = f.read().split()
        if len(parts) >= 3:
            return (int(parts[0]), int(parts[2]))
    except (OSError, ValueError):
        pass
    return (0, 1 << 30)


async def _wait_for_available_resources(
    worker_count: int,
    is_ssr: bool,
    check_interval_s: float = 3.0,
    max_wait_s: float = 120.0,
) -> bool:
    """Poll until sufficient memory, inotify, and fd headroom is available.

    Each non-SSR worker is estimated at 256 MiB (Chromium only).
    Each SSR worker is estimated at 768 MiB (Chromium + dev server).
    Each chromium also opens ~3 inotify instances and ~150 fds; we keep a
    20%-of-cap safety margin so unrelated processes (logging, NFS, monitor)
    don't get starved.
    Returns True if all resources are adequate, False if timed out.
    """
    per_worker_mib = 768 if is_ssr else 256
    required_mib = worker_count * per_worker_mib
    per_worker_inotify = 3
    per_worker_fds = 150
    deadline = time.monotonic() + max_wait_s

    while True:
        available_mib = _get_available_memory_mib()
        inotify_used, inotify_max = _get_inotify_headroom()
        fd_used, fd_max = _get_system_fd_headroom()

        # Reserve 20% of each pool as safety margin
        inotify_budget = max(0, int(inotify_max * 0.8) - inotify_used)
        fd_budget = max(0, int(fd_max * 0.8) - fd_used)
        inotify_need = worker_count * per_worker_inotify
        fd_need = worker_count * per_worker_fds

        bottleneck = None
        if available_mib < required_mib:
            bottleneck = f"memory ({available_mib:.0f}<{required_mib:.0f} MiB)"
        elif inotify_budget < inotify_need:
            bottleneck = f"inotify ({inotify_used}/{inotify_max}, need {inotify_need} more, budget {inotify_budget})"
        elif fd_budget < fd_need:
            bottleneck = f"system fds ({fd_used}/{fd_max}, need {fd_need} more, budget {fd_budget})"

        if bottleneck is None:
            return True

        remaining = deadline - time.monotonic()
        if remaining <= 1:
            return False
        wait = min(check_interval_s, remaining)
        logger.warning(
            "[scale_up] Resource pressure (%s) for %d workers. Waiting %.0fs...",
            bottleneck, worker_count, wait,
        )
        await asyncio.sleep(wait)


def _is_ssr_project(workspace_root: str) -> bool:
    """Detect whether the project is an SSR framework (Next.js, Nuxt, Remix, etc.).

    Returns True if the project uses a framework with server-side rendering
    where starting multiple dev server instances helps reduce contention.
    """
    root = Path(workspace_root)
    pkg_json_path = root / "package.json"
    if not pkg_json_path.exists():
        return False
    try:
        pkg_json = json.loads(pkg_json_path.read_text(encoding="utf-8"))
        all_deps = {
            *(pkg_json.get("dependencies") or {}),
            *(pkg_json.get("devDependencies") or {}),
        }
        ssr_frameworks = {"next", "nuxt", "remix", "gatsby", "sveltekit", "astro", "solid-js"}
        return bool(all_deps & ssr_frameworks)
    except (json.JSONDecodeError, OSError):
        return False


async def _scale_up_workers(
    *,
    workspace_root: str,
    worker_count: int,
    task_id: str,
    session_id: str = "",
    primary_app_url: str = "",
) -> List[Dict[str, Any]]:
    """Scale up a single-worker row to *worker_count* workers.

    Each worker gets its own Chromium instance with an independent CDP
    connection. For SSR projects (Next.js, Nuxt, etc.), each worker also
    gets a dedicated dev server on its own port.

    Ports are allocated from the shared SQLite-backed port pool
    (``tools.ports.allocate``) to avoid conflicts between concurrent rows.

    Args:
        primary_app_url: The actual app URL from the primary build executor.
            Used as the app URL for non-SSR workers (which share the primary
            dev server).

    Returns a list of worker dicts:
        [{
            "app_url": str,
            "cdp_url": str,
            "app_port": int,
            "cdp_port": int,
            "app_lease_id": str,
            "cdp_lease_id": str,
            "dev_pid": int | None,
            "chromium_pid": int | None,
        }, ...]
    """
    if worker_count <= 1:
        return []

    resolved_root = Path(workspace_root).resolve(strict=True)
    effective_session_id = session_id

    from tools.ports import allocate as _allocate_port
    from tools.exec import start as _exec_start

    is_ssr = _is_ssr_project(workspace_root)
    logger.info(
        "[scale_up] Scaling row %s to %d workers (SSR=%s, project=%s)",
        task_id, worker_count, is_ssr, resolved_root.name,
    )

    chromium_bin = str(os.getenv("CHROMIUM_BIN", "") or "").strip() or "chromium-browser"

    # --- Wait for sufficient system resources before scaling ---
    resources_ok = await _wait_for_available_resources(worker_count, is_ssr)
    if not resources_ok:
        logger.warning(
            "[scale_up] Resource wait timed out for %s (SSR=%s), "
            "proceeding anyway — workers may fail",
            task_id, is_ssr,
        )

    workers: List[Dict[str, Any]] = []
    # Stagger delay between workers to avoid fork storm.
    # 16 workers × 0.3s ≈ 5s total extra wait, well worth avoiding EAGAIN.
    _stagger_s = 0.3
    for i in range(worker_count):
        if i > 0:
            await asyncio.sleep(_stagger_s)

        # --- Allocate ports from shared pool ---
        app_lease = _allocate_port(
            name=f"app_worker_{i}",
            holder=f"worker_{task_id}_{i}",
        )
        cdp_lease = _allocate_port(
            name=f"cdp_worker_{i}",
            holder=f"worker_{task_id}_{i}",
        )
        app_port = app_lease["port"]
        cdp_port = cdp_lease["port"]

        if effective_session_id:
            _tool_session.record_port_allocated(effective_session_id, app_port)
            _tool_session.record_port_allocated(effective_session_id, cdp_port)

        # --- Start Chromium with CDP (with retry) ---
        browser_cmd = (
            f"{chromium_bin} --headless --no-sandbox "
            f"--remote-debugging-port={cdp_port} "
            f"--remote-debugging-address=0.0.0.0 "
            f"--disable-logging --log-level=3 "
            f"--disable-gpu-sandbox --disable-software-rasterizer "
            f"--renderer-process-limit=2 "
            f"--disable-features=site-per-process,IsolateOrigins "
            f"about:blank"
        )
        browser_result = None
        _chromium_retries = 2
        for attempt in range(1 + _chromium_retries):
            if attempt > 0:
                _backoff = 2.0 * attempt
                logger.info(
                    "[scale_up] Retrying Chromium start for worker %d "
                    "(attempt %d/%d) after %.0fs...",
                    i, attempt + 1, 1 + _chromium_retries, _backoff,
                )
                await asyncio.sleep(_backoff)
            browser_result = await asyncio.to_thread(
                _exec_start,
                browser_cmd,
                cwd=str(resolved_root),
                readiness={"url": f"http://localhost:{cdp_port}/json", "timeout_s": 30, "interval_s": 0.5},
                log_dir=str(resolved_root / ".process_logs"),
            )
            if "error" not in (browser_result or {}):
                break
        if not browser_result or "error" in browser_result:
            logger.error(
                "[scale_up] Worker %d browser start failed after all retries: %s",
                i, browser_result,
            )
            # Release the allocated ports so the port pool doesn't get fragmented.
            try:
                from tools.ports import release as _release_port
                _release_port(app_lease["lease_id"])
                _release_port(cdp_lease["lease_id"])
            except Exception:
                pass
            continue

        worker: Dict[str, Any] = {
            "app_url": primary_app_url or f"http://localhost:{app_port}",
            "cdp_url": f"http://localhost:{cdp_port}",
            "app_port": app_port,
            "cdp_port": cdp_port,
            "app_lease_id": app_lease["lease_id"],
            "cdp_lease_id": cdp_lease["lease_id"],
            "dev_pid": None,
            "chromium_pid": browser_result.get("pid"),
        }

        # --- Start dev server (SSR projects only) ---
        dev_pid = None
        if is_ssr:
            pkg_json = json.loads((resolved_root / "package.json").read_text(encoding="utf-8"))
            scripts = pkg_json.get("scripts", {})
            dev_script_key = next(
                (k for k in ("dev", "start", "serve") if k in scripts),
                None,
            )
            if dev_script_key:
                app_cmd = f"npm run {dev_script_key} -- --port {app_port}"
            elif "next" in (pkg_json.get("dependencies") or {}):
                app_cmd = f"npx next dev --port {app_port}"
            elif "vite" in (pkg_json.get("dependencies") or {}) or "vite" in (pkg_json.get("devDependencies") or {}):
                app_cmd = f"npx vite --port {app_port}"
            else:
                # Fallback — try the original dev script or skip
                app_cmd = f"npx vite --port {app_port}"

            readiness_url = f"http://localhost:{app_port}"
            server_result = await asyncio.to_thread(
                _exec_start,
                app_cmd,
                cwd=str(resolved_root),
                readiness={"url": readiness_url, "timeout_s": 60, "interval_s": 1.0},
                log_dir=str(resolved_root / ".process_logs"),
            )
            if "error" in (server_result or {}):
                logger.warning(
                    "[scale_up] Worker %d dev server start failed (will use shared): %s",
                    i, server_result.get("error", server_result),
                )
                # Fall back to the primary app URL
                worker["app_url"] = primary_app_url or f"http://localhost:{app_port}"
            else:
                worker["dev_pid"] = server_result.get("pid")

        workers.append(worker)

    if not workers:
        logger.warning("[scale_up] No workers were successfully started for %s", task_id)
        return []

    logger.info(
        "[scale_up] Successfully started %d/%d workers for %s",
        len(workers), worker_count, task_id,
    )
    return workers


def _empty_runtime_payload(status: str, error: str | None = None) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "status": status,
        "runtime": {"steps_used": 0, "duration_ms": 0},
        "dimensions": [],
        "planned_tasks": [],
        "task_results": [],
        "task_completion_score": None,
        "trajectory_ref": None,
        "trajectory": {
            "status": status,
            "steps": [],
            "dom_elements": [],
            "initial_diagnostics": {},
            "duration_ms": 0,
        },
    }
    if error:
        payload["error"] = error
    return payload


def _blocked_agent_payload(agent: AgentConfig, blocked_by: List[str]) -> Dict[str, Any]:
    reason = f"Blocked by upstream agents: {', '.join(blocked_by)}"
    return normalize_agent_result({
        "agent_id": agent.id,
        "score": None,
        "score_scale": "0-100",
        "end_reason": reason,
        **_empty_runtime_payload("blocked", error=reason),
    })


def _infra_agent_payload(agent: AgentConfig, reason: str) -> Dict[str, Any]:
    """Build an infra-error payload so the task is picked up by rerun."""
    return normalize_agent_result({
        "agent_id": agent.id,
        "score": None,
        "score_scale": "0-100",
        "end_reason": reason,
        **_empty_runtime_payload("infra_error", error=reason),
    })


def _invalid_agent_payload(agent: AgentConfig, reason: str) -> Dict[str, Any]:
    return normalize_agent_result({
        "agent_id": agent.id,
        "score": None,
        "score_scale": "0-100",
        "end_reason": reason,
        **_empty_runtime_payload("failed", error=reason),
    })


def _skipped_agent_payload(agent: AgentConfig, reason: str) -> Dict[str, Any]:
    """Build a completed status payload with score=0 for an agent that was
    skipped due to build failure.  Unlike _blocked_agent_payload (score=None),
    this contributes score 0 to aggregation — the task could not be evaluated
    because the build was broken."""
    return normalize_agent_result({
        "agent_id": agent.id,
        "score": 0,
        "score_scale": "0-100",
        "end_reason": reason,
        "aggregation_mode": "strict",
        "agent_max_score": 20,
        **_empty_runtime_payload("completed", error=reason),
    })


def _force_build_failure(build_result: Dict[str, Any], reason: str) -> Dict[str, Any]:
    normalized = dict(build_result or {})
    dimensions = normalized.get("dimensions") if isinstance(normalized.get("dimensions"), list) else []
    updated_dimensions: List[Dict[str, Any]] = []
    found_build_success = False
    for item in dimensions:
        if not isinstance(item, dict):
            continue
        copied = dict(item)
        if str(copied.get("dimension_id") or "").strip() == "build_success":
            copied["verdict"] = "failed"
            copied["reason"] = reason
            copied["score"] = 0.0
            found_build_success = True
        updated_dimensions.append(copied)
    if not found_build_success:
        updated_dimensions.insert(0, {
            "dimension_id": "build_success",
            "verdict": "failed",
            "reason": reason,
            "score": 0.0,
        })
    normalized["dimensions"] = updated_dimensions
    normalized["status"] = "failed"
    normalized.pop("task_tree", None)
    normalized["score"] = 0.0
    normalized["task_completion_score"] = 0.0
    existing_end_reason = str(normalized.get("end_reason") or "").strip()
    normalized["end_reason"] = reason if not existing_end_reason else f"{existing_end_reason}; {reason}"
    return normalized


def _validate_build_phase_execution(
    *,
    source_files: Dict[str, str],
    build_result: Dict[str, Any],
) -> Optional[str]:
    trajectory = build_result.get("trajectory") if isinstance(build_result.get("trajectory"), dict) else {}
    steps = trajectory.get("steps") if isinstance(trajectory.get("steps"), list) else []
    tool_names = {
        str(step.get("tool_name") or step.get("tool") or step.get("name") or "").strip()
        for step in steps
        if isinstance(step, dict)
    }
    tool_names.discard("")
    start_commands = [
        str(
            ((step.get("args") if isinstance(step.get("args"), dict) else step.get("tool_args") or {}) or {}).get("cmd")
            or ((step.get("args") if isinstance(step.get("args"), dict) else step.get("tool_args") or {}) or {}).get("command")
            or ""
        ).strip()
        for step in steps
        if isinstance(step, dict)
        and str(step.get("tool_name") or step.get("tool") or step.get("name") or "").strip() == "local_exec_start"
    ]
    lowered_start_commands = [command.lower() for command in start_commands if command]

    if not source_files:
        return "Build engineer ended with an empty workspace. It must extract code snippets from the original model reply and write files before concluding."

    if not ({"local_exec_run", "local_exec_start", "protocol_write_artifacts"} & tool_names):
        return (
            "Build engineer ended without any build actions. Expected at least one of "
            "local_exec_run, local_exec_start, or protocol_write_artifacts."
        )

    return None


def _build_path_review_task_context(build_handoff: AgentHandoff) -> str:
    facts = build_handoff.facts if isinstance(build_handoff.facts, dict) else {}
    trajectory_log_paths = [str(item) for item in (facts.get("trajectory_log_paths") or []) if str(item).strip()]
    path_review_hints = [str(item) for item in (facts.get("path_review_hints") or []) if str(item).strip()]
    build_dimension_outcomes = [str(item) for item in (facts.get("build_dimension_outcomes") or []) if str(item).strip()]
    repair_actions = [str(item) for item in (facts.get("repair_actions") or []) if str(item).strip()]
    failure_snippets = [str(item) for item in (facts.get("failure_snippets") or []) if str(item).strip()]
    repair_event_count = int(facts.get("repair_event_count") or 0)
    transient_retry_count = int(facts.get("transient_retry_count") or 0)
    repair_events = facts.get("repair_events") if isinstance(facts.get("repair_events"), list) else []
    artifacts_path = str(facts.get("artifacts_path") or "").strip()
    process_log_dir = str(facts.get("process_log_dir") or "").strip()
    evidence_refs = [str(item) for item in (build_handoff.evidence_refs or []) if str(item).strip()]

    required_path_targets = trajectory_log_paths[:] or ([artifacts_path] if artifacts_path else [])
    require_log_read = bool(trajectory_log_paths or process_log_dir)
    lines = [
        "=== Build Path Review Protocol ===",
        "This reviewer must use the upstream build path before reaching any verdict.",
        "A verdict is invalid if you skip path evidence and rely only on source inspection or general reasoning.",
        "",
        "CRITICAL — only the files explicitly listed below exist. Do NOT invent or assume files",
        "named 'trajectory.log' or 'step_summary.log' — those names are field identifiers, not real",
        "files.  The actual log files are stdout/stderr captures with timestamp-prefixed names.",
        "If a file path is not in the list below, it DOES NOT EXIST. Do not waste steps looking for it.",
        "",
        "Before calling submit_verdict, you must call local_fs_read on the required",
        "build-path evidence listed below.  Use local_fs_read (not read_file) for these absolute paths.",
        "You may inspect source code with read_file/search_source afterwards, but source-only review",
        "is not allowed.",
        "",
        "IMPORTANT: You are in READ-ONLY mode. No browser or runtime-testing tools",
        "(navigate, click_element, type_text, get_page_context, etc.) are available.",
        "Only local_fs_read, read_file, search_source, and submit_verdict will work.",
        "Do not attempt to use browser tools — they are not available in this mode.",
    ]
    if artifacts_path:
        lines.append(f"\nartifacts_path: {artifacts_path}")
        lines.append("  This is the primary build artifact — read it first. It contains ports, URLs,")
        lines.append("  environment metadata, and the CDP endpoint for the running application.")
    if process_log_dir:
        lines.append(f"\nprocess_log_dir: {process_log_dir}")
        lines.append("  Directory containing timestamped stdout/stderr logs from build steps")
        lines.append("  (e.g. npm install, npm run dev).  Call local_fs_read on this directory")
        lines.append("  path first to list available log files, then read individual files.")
    if required_path_targets:
        lines.append("\n=== REQUIRED PATH TARGETS (only these files exist) ===")
        lines.extend(f"  - {path}" for path in required_path_targets)
        lines.append("\nThese are the ONLY build-path files available.  If you need to read source files")
        lines.append("(such as index.html, app/page.tsx, etc.), use read_file with the relative filename.")
    if require_log_read:
        lines.append("")
        lines.append("CRITICAL VALIDATION RULE:")
        lines.append("Reading artifacts_path alone is insufficient when log files or process_log_dir are provided.")
        lines.append("You must inspect at least one timestamped log file via local_fs_read before submitting a verdict.")
    if evidence_refs:
        lines.append("evidence_refs:")
        lines.extend(f"- {path}" for path in evidence_refs)
    if build_dimension_outcomes:
        lines.append("build_dimension_outcomes:")
        lines.extend(f"- {item}" for item in build_dimension_outcomes)
    lines.append(f"repair_event_count: {repair_event_count}  (modify-then-restart events — real repair)")
    lines.append(f"transient_retry_count: {transient_retry_count}  (port/network retries with no intervening edit — harmless)")
    if repair_events:
        lines.append("repair_events:")
        for ev in repair_events[:8]:
            if not isinstance(ev, dict):
                continue
            kind = str(ev.get("kind") or "")
            idx = ev.get("step_index")
            writes = ev.get("writes_between") or []
            writes_str = ", ".join(str(w) for w in writes[:4]) if writes else ""
            lines.append(f"- {kind} @ step[{idx}] writes_between=[{writes_str}]")
    if repair_actions:
        lines.append("files_written (informational — NOT repair evidence by itself):")
        lines.extend(f"- {item}" for item in repair_actions)
    if failure_snippets:
        lines.append("failure_snippets:")
        lines.extend(f"- {item}" for item in failure_snippets[:5])
    if path_review_hints:
        lines.append("path_review_hints:")
        lines.extend(f"- {item}" for item in path_review_hints)
    return "\n".join(lines)


def _extract_local_fs_read_paths(agent_payload: Dict[str, Any]) -> List[str]:
    trajectory = agent_payload.get("trajectory") if isinstance(agent_payload.get("trajectory"), dict) else {}
    steps = trajectory.get("steps") if isinstance(trajectory.get("steps"), list) else []
    read_paths: List[str] = []
    for step in steps:
        if not isinstance(step, dict):
            continue
        tool_name = str(step.get("tool_name") or step.get("tool") or step.get("name") or "").strip()
        if tool_name != "local_fs_read":
            continue
        args = step.get("args") if isinstance(step.get("args"), dict) else step.get("tool_args")
        if not isinstance(args, dict):
            continue
        path = str(args.get("path") or "").strip()
        if path:
            read_paths.append(path)
    return read_paths


def _build_path_review_used_handoff_paths(agent_payload: Dict[str, Any], build_handoff: AgentHandoff) -> bool:
    consulted_paths = _extract_local_fs_read_paths(agent_payload)
    if not consulted_paths:
        return False

    facts = build_handoff.facts if isinstance(build_handoff.facts, dict) else {}
    required_paths = [str(item) for item in (facts.get("trajectory_log_paths") or []) if str(item).strip()]
    artifacts_path = str(facts.get("artifacts_path") or "").strip()
    process_log_dir = str(facts.get("process_log_dir") or "").strip()
    require_log_read = bool(required_paths or process_log_dir)
    consulted_required_log = False
    consulted_artifact = False

    for consulted_path in consulted_paths:
        if consulted_path in required_paths:
            consulted_required_log = True
        if process_log_dir and consulted_path.startswith(process_log_dir.rstrip("/") + "/"):
            consulted_required_log = True
        if artifacts_path and consulted_path == artifacts_path:
            consulted_artifact = True

    if require_log_read:
        return consulted_required_log
    if artifacts_path:
        return consulted_artifact
    return False


def _make_build_handoff(
    agent: AgentConfig,
    build_result: Dict[str, Any],
    artifacts: Dict[str, Any],
    process_log_dir: Path,
    source_files: Dict[str, str],
    artifacts_path: str,
) -> AgentHandoff:
    build_ready = artifacts.get("state") == "ready"
    ports = artifacts.get("ports") if isinstance(artifacts.get("ports"), list) else []
    first_port = ports[0] if ports and isinstance(ports[0], dict) else {}
    app_url = str(first_port.get("url", "")).strip()
    cdp_url = str(artifacts.get("cdp_url", "")).strip()
    warnings: List[str] = []
    if build_result.get("status") != "completed":
        warnings.append(f"build_agent_status={build_result.get('status')}")
    if artifacts.get("error"):
        warnings.append(str(artifacts.get("error")))
    trajectory_summary = _extract_build_trajectory_summary(build_result)

    status = "completed" if build_ready else "failed"
    summary = (
        f"App is running at {app_url} and CDP is ready."
        if build_ready else f"Build failed: {artifacts.get('error') or 'unknown error'}"
    )
    blockers = [] if build_ready else [summary]
    return AgentHandoff(
        producer_agent_id=agent.id,
        stage=agent.stage or "build",
        status=status,
        summary=summary,
        facts={
            "build_status": artifacts.get("state", "failed"),
            "app_url": app_url,
            "preview_url": app_url,
            "cdp_url": cdp_url,
            "ports": ports,
            "install_status": "success" if build_ready else "failed",
            "startup_status": "success" if build_ready else "failed",
            "fix_count": artifacts.get("fix_count", 0),
            "runtime_warnings": warnings,
            "startup_observations": [summary],
            "source_inventory": _build_source_inventory(source_files),
            "artifacts_path": artifacts_path,
            "process_log_dir": str(process_log_dir),
            "build_dimension_outcomes": _summarize_dimension_outcomes(build_result),
            "trajectory_step_summaries": trajectory_summary["step_summaries"],
            "trajectory_log_paths": trajectory_summary["log_paths"],
            "trajectory_error_signals": trajectory_summary["error_signals"],
            "repair_actions": trajectory_summary["repair_actions"],
            "failure_snippets": trajectory_summary["failure_snippets"],
            "repair_event_count": trajectory_summary["repair_event_count"],
            "transient_retry_count": trajectory_summary["transient_retry_count"],
            "repair_events": trajectory_summary["repair_events"],
            "path_review_hints": [
                "Inspect trajectory_log_paths with read_file before concluding deployment-effort or build-code-health dimensions.",
                "Search for keywords such as npm ERR, warning, module not found, can't resolve, failed to compile, syntax error, missing script, and ENOENT.",
                "no_extra_repair_needed is decided from repair_event_count (real repair = modify-then-restart) — repair_actions only lists files written and is NOT a repair count.",
                "transient_retry_count (port conflicts, transient install retries with no intervening edit) is harmless and must not lower the verdict.",
            ],
        },
        warnings=warnings,
        blockers=blockers,
        evidence_refs=[str(process_log_dir)],
    )


def _make_runtime_handoff(
    agent: AgentConfig,
    agent_payload: Dict[str, Any],
    runtime_paths: Dict[str, str],
    app_url: str,
    cdp_url: str,
) -> AgentHandoff:
    completion = agent_payload.get("task_completion_score")
    main_task_pass_rate = agent_payload.get("main_task_pass_rate")
    if isinstance(main_task_pass_rate, (int, float)):
        summary = f"main_task_pass_rate={main_task_pass_rate:.3f}; status={agent_payload.get('status', 'unknown')}"
    elif isinstance(completion, (int, float)):
        summary = f"task_completion_score={completion:.3f}; status={agent_payload.get('status', 'unknown')}"
    else:
        summary = str(agent_payload.get("end_reason") or agent_payload.get("status") or "completed")
    warnings: List[str] = []
    if agent_payload.get("error"):
        warnings.append(str(agent_payload["error"]))
    return AgentHandoff(
        producer_agent_id=agent.id,
        stage=agent.stage or "evaluate",
        status=str(agent_payload.get("status", "unknown")),
        summary=summary,
        facts={
            "task_completion_score": completion,
            "main_task_pass_rate": main_task_pass_rate,
            "score": agent_payload.get("score"),
            "end_reason": agent_payload.get("end_reason"),
            "workspace_root": runtime_paths.get("workspace_root"),
            "artifacts_path": runtime_paths.get("artifacts_path"),
            "process_log_dir": runtime_paths.get("process_log_dir"),
            "app_url": app_url,
            "preview_url": app_url,
            "cdp_url": cdp_url,
        },
        warnings=warnings,
        blockers=[],
        evidence_refs=[
            ref for ref in [
                runtime_paths.get("process_log_dir"),
                runtime_paths.get("artifacts_path"),
            ] if ref
        ],
    )


def _build_runtime_paths(tmpdir: str, artifacts_path: str, process_log_dir: Path) -> Dict[str, str]:
    return {
        "workspace_root": tmpdir,
        "artifacts_path": artifacts_path,
        "process_log_dir": str(process_log_dir),
    }


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


def _stringify_nested_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, (dict, list)):
        try:
            return json.dumps(value, ensure_ascii=False)
        except Exception:
            return str(value)
    return str(value)


def _collect_log_paths_from_value(value: Any) -> List[str]:
    paths: List[str] = []
    if isinstance(value, dict):
        for key, nested in value.items():
            normalized_key = str(key).strip().lower()
            if isinstance(nested, str) and normalized_key in {
                "stdout_path",
                "stderr_path",
                "log_path",
                "artifacts_path",
                "process_log_dir",
            }:
                paths.append(nested)
            elif isinstance(nested, (dict, list)):
                paths.extend(_collect_log_paths_from_value(nested))
    elif isinstance(value, list):
        for nested in value:
            if isinstance(nested, (dict, list)):
                paths.extend(_collect_log_paths_from_value(nested))
    return paths


_BUILD_ERROR_SIGNAL_PATTERNS: tuple[tuple[str, str], ...] = (
    ("npm err", "npm_err"),
    ("warning", "warning"),
    ("warn", "warning"),
    ("module not found", "module_not_found"),
    ("cannot find module", "cannot_find_module"),
    ("can't resolve", "cannot_resolve"),
    ("failed to compile", "failed_to_compile"),
    ("compile error", "compile_error"),
    ("syntax error", "syntax_error"),
    ("unexpected token", "unexpected_token"),
    ("missing script", "missing_script"),
    ("enoent", "enoent"),
    ("eaddrinuse", "eaddrinuse"),
)


_DEV_SERVER_CMD_RE = re.compile(
    r"\b(vite|next(?!\.config)|nuxt|astro|http\.server|serve\b|parcel|webpack-dev-server|rollup|"
    r"ng\s+serve|gatsby\s+develop)\b",
    re.IGNORECASE,
)
_INSTALL_CMD_RE = re.compile(
    r"\b(npm|pnpm|yarn|bun)\s+(install|i|ci|add)\b",
    re.IGNORECASE,
)
_CDP_CMD_RE = re.compile(r"\b(chrom|google-chrome|chromium-browser)\b", re.IGNORECASE)


def _classify_build_step(tool_name: str, command: str, args_dict: Dict[str, Any]) -> str | None:
    """Return one of {WRITE, INSTALL, START_APP, START_CDP, ARTIFACT} or None."""
    if tool_name in ("local_fs_write", "local_fs_extract_block"):
        return "WRITE"
    if tool_name == "local_exec_run" and _INSTALL_CMD_RE.search(command):
        return "INSTALL"
    if tool_name == "local_exec_start":
        if _CDP_CMD_RE.search(command):
            return "START_CDP"
        if _DEV_SERVER_CMD_RE.search(command):
            return "START_APP"
    if tool_name == "protocol_write_artifacts":
        return "ARTIFACT"
    return None


def _artifact_state_from_args(args_dict: Dict[str, Any]) -> str | None:
    state = args_dict.get("state")
    if isinstance(state, str) and state.strip():
        return state.strip().lower()
    payload = args_dict.get("artifacts") or args_dict.get("payload")
    if isinstance(payload, dict):
        s = payload.get("state")
        if isinstance(s, str) and s.strip():
            return s.strip().lower()
    return None


def _extract_build_trajectory_summary(build_result: Dict[str, Any]) -> Dict[str, Any]:
    trajectory = build_result.get("trajectory") if isinstance(build_result.get("trajectory"), dict) else {}
    steps = trajectory.get("steps") if isinstance(trajectory.get("steps"), list) else []

    step_summaries: List[str] = []
    log_paths: List[str] = []
    error_signals: List[str] = []
    failure_snippets: List[str] = []

    # Records used by the modify-then-restart classifier.
    # Each entry: {"index", "kind", "path"(write only), "state"(artifact only)}
    events: List[Dict[str, Any]] = []
    write_paths: List[str] = []

    for index, step in enumerate(steps):
        if not isinstance(step, dict):
            continue
        tool_name = str(
            step.get("tool_name")
            or step.get("tool")
            or step.get("name")
            or step.get("action")
            or "unknown"
        ).strip()
        args = step.get("args") if isinstance(step.get("args"), dict) else step.get("tool_args")
        args_dict = args if isinstance(args, dict) else {}
        result_payload = step.get("result")
        observation = step.get("observation")
        step_text = "\n".join(
            part for part in [
                _stringify_nested_value(args_dict),
                _stringify_nested_value(result_payload),
                _stringify_nested_value(observation),
            ]
            if part and part not in {"{}", "[]"}
        )
        command = str(
            args_dict.get("command")
            or args_dict.get("cmd")
            or args_dict.get("shell_command")
            or args_dict.get("path")
            or args_dict.get("file_path")
            or ""
        ).strip()

        status = "error" if "error" in step_text.lower() else "ok"
        summary = f"step[{index}] {tool_name}"
        if command:
            summary += f" :: {command}"
        summary += f" :: {status}"
        step_summaries.append(summary)

        for candidate in _collect_log_paths_from_value(step):
            if candidate and candidate not in log_paths:
                log_paths.append(candidate)

        normalized_text = re.sub(r"\s+", " ", step_text).strip()
        lowered = normalized_text.lower()
        for needle, label in _BUILD_ERROR_SIGNAL_PATTERNS:
            if needle in lowered and label not in error_signals:
                error_signals.append(label)

        kind = _classify_build_step(tool_name, command, args_dict)
        if kind == "WRITE":
            wp = str(args_dict.get("path") or args_dict.get("file_path") or args_dict.get("dest_path") or "").strip()
            events.append({"index": index, "kind": "WRITE", "path": wp})
            if wp and wp not in write_paths:
                write_paths.append(wp)
        elif kind == "ARTIFACT":
            events.append({"index": index, "kind": "ARTIFACT", "state": _artifact_state_from_args(args_dict)})
        elif kind in {"INSTALL", "START_APP", "START_CDP"}:
            events.append({"index": index, "kind": kind})

        if any(token in lowered for token in ("error", "failed", "warning", "warn")):
            snippet = normalized_text[:240]
            if snippet and snippet not in failure_snippets:
                failure_snippets.append(snippet)

    # Modify-then-restart classifier: walk same-kind repetitions, count
    # WRITEs between consecutive non-WRITE events of the same kind.
    repair_events: List[Dict[str, Any]] = []
    transient_retry_count = 0

    def _last_idx_in_events(target_kind: str, before_pos: int) -> int:
        for j in range(before_pos - 1, -1, -1):
            if events[j].get("kind") == target_kind:
                return j
        return -1

    for pos, ev in enumerate(events):
        kind = ev.get("kind")
        if kind in {"INSTALL", "START_APP", "START_CDP"}:
            prev = _last_idx_in_events(kind, pos)
            if prev < 0:
                continue  # first occurrence — not a retry
            writes_between = [
                e for e in events[prev + 1 : pos] if e.get("kind") == "WRITE"
            ]
            if writes_between:
                repair_events.append({
                    "kind": f"{kind.lower()}_after_write",
                    "step_index": ev["index"],
                    "writes_between": [w.get("path") or "" for w in writes_between][:8],
                })
            else:
                transient_retry_count += 1
        elif kind == "ARTIFACT" and ev.get("state") == "ready":
            # Look for a previous ARTIFACT(state=failed) → counts as repair if
            # any WRITE happened in between (a failed artifact followed by edits
            # then a successful artifact == build was patched into working order).
            for j in range(pos - 1, -1, -1):
                prev_ev = events[j]
                if prev_ev.get("kind") == "ARTIFACT" and prev_ev.get("state") == "failed":
                    writes_between = [
                        e for e in events[j + 1 : pos] if e.get("kind") == "WRITE"
                    ]
                    if writes_between:
                        repair_events.append({
                            "kind": "artifact_ready_after_failed_with_write",
                            "step_index": ev["index"],
                            "writes_between": [w.get("path") or "" for w in writes_between][:8],
                        })
                    else:
                        transient_retry_count += 1
                    break

    # Step summaries: keep head-4 + tail-16 so install/start/artifact-write at
    # the tail are always visible even on long extract-heavy runs.
    if len(step_summaries) > 20:
        clipped = (
            step_summaries[:4]
            + [f"... ({len(step_summaries) - 20} steps elided) ..."]
            + step_summaries[-16:]
        )
    else:
        clipped = step_summaries

    return {
        "step_summaries": clipped,
        "log_paths": log_paths[:12],
        "error_signals": error_signals[:16],
        "repair_actions": write_paths[:16],  # kept for back-compat; now means "files written" (not "files repaired")
        "failure_snippets": failure_snippets[:8],
        "repair_event_count": len(repair_events),
        "transient_retry_count": transient_retry_count,
        "repair_events": repair_events[:8],
    }


def _summarize_dimension_outcomes(build_result: Dict[str, Any]) -> List[str]:
    lines: List[str] = []
    for item in build_result.get("dimensions") if isinstance(build_result.get("dimensions"), list) else []:
        if not isinstance(item, dict):
            continue
        dim_id = str(item.get("dimension_id") or "").strip()
        verdict = str(item.get("verdict") or "unknown").strip()
        reason = str(item.get("reason") or "").strip()
        if dim_id:
            lines.append(f"{dim_id}: {verdict} :: {reason}")
    return lines


def _is_build_path_review_agent(agent: AgentConfig) -> bool:
    return any(str(tag).strip().lower() in {"build_path_review", "build-path-review"} for tag in agent.tags)


def _handoff_path_evidence_lines(handoff: Dict[str, Any]) -> List[str]:
    facts = handoff.get("facts") if isinstance(handoff.get("facts"), dict) else {}
    lines: List[str] = []
    for key in ("workspace_root", "artifacts_path", "process_log_dir", "app_url", "preview_url", "cdp_url"):
        value = facts.get(key)
        if value not in (None, "", [], {}):
            lines.append(f"{key}: {value}")
    for ref in handoff.get("evidence_refs") if isinstance(handoff.get("evidence_refs"), list) else []:
        lines.append(f"evidence_ref: {ref}")
    return lines


def _attach_report_path_evidence(report: Dict[str, Any]) -> None:
    agents = report.get("agents")
    if not isinstance(agents, list):
        return

    handoff_by_id: Dict[str, Dict[str, Any]] = {}
    for handoff in report.get("pipeline_handoffs") if isinstance(report.get("pipeline_handoffs"), list) else []:
        if not isinstance(handoff, dict):
            continue
        handoff_agent_id = str(handoff.get("producer_agent_id") or handoff.get("agent_id") or "").strip()
        if handoff_agent_id:
            handoff_by_id[handoff_agent_id] = handoff

    for agent in agents:
        if not isinstance(agent, dict):
            continue
        agent_id = str(agent.get("agent_id") or agent.get("role") or "").strip()
        handoff = handoff_by_id.get(agent_id)
        lines: List[str] = []
        if handoff:
            lines.extend(_handoff_path_evidence_lines(handoff))

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
        agent["path_evidence"] = deduped


async def run_open_single(
    *,
    config: "Config",
    task_id: str,
    raw_markdown: str,
    query: str,
    agents_dir: str = "agents",
    max_parallel: Optional[int] = None,
    subtask_semaphore: Optional[asyncio.Semaphore] = None,
    npm_install_semaphore: Optional[asyncio.Semaphore] = None,
    build_semaphore: Optional[asyncio.Semaphore] = None,
    planned_tasks: Optional[List["PlannedTask"]] = None,
    planned_tasks_by_agent: Optional[Dict[str, List["PlannedTask"]]] = None,
    fixed_task_db: Optional[FixedTaskDB] = None,
    evaluate_agent_gate: Optional[RowPriorityGate] = None,
    agent_ids: Optional[List[str]] = None,
    task_progress_callback: Optional[Callable[[str, Dict[str, Any]], Any]] = None,
    total_chromium_semaphore: Optional[asyncio.Semaphore] = None,
    heavy_exec_semaphore: Optional[asyncio.Semaphore] = None,
) -> Dict[str, Any]:
    """Two-phase open-mode evaluation: build then evaluate."""
    _workspace_base = os.path.join(tempfile.gettempdir(), "eval_open_workspaces")
    os.makedirs(_workspace_base, exist_ok=True)
    safe_task_id = str(task_id).replace("/", "_").replace("..", "_") or "default"
    tmpdir = os.path.join(_workspace_base, safe_task_id)
    os.makedirs(tmpdir, exist_ok=True)
    artifacts_path = os.path.join(tmpdir, "artifacts.json")
    process_log_dir = Path(tmpdir) / ".process_logs"
    process_log_dir.mkdir(parents=True, exist_ok=True)
    _tool_session.register_session(artifacts_path, workspace_root=tmpdir)
    active_session_token = _tool_session.activate_session(artifacts_path)
    runtime_paths = _build_runtime_paths(tmpdir, artifacts_path, process_log_dir)

    # --- Pre-install common npm dependencies ---
    # Populate node_modules in the workspace with a comprehensive set of
    # common packages so the build agent's npm install only fetches the
    # delta (usually 0-3 packages).  Serialized with a semaphore to avoid
    # disk/cache thrashing when many rows start concurrently.
    # Started as a background task so agent loading + setup overlap with it.
    #
    # Timeout: 10 min. The full preinstall_package.json (~75 packages, Radix,
    # firebase, supabase, etc.) takes 3-8 min on a cold cache + concurrent
    # rows. The previous 180s ceiling caused 100% of rows to time out and
    # fall back to per-row npm install, defeating the cache.
    _NPM_PREINSTALL_TIMEOUT_S = 600

    async def _run_npm_preinstall() -> Dict[str, Any]:
        metrics: Dict[str, Any] = {
            "npm_wait_ms": 0,
            "npm_duration_ms": 0,
            "npm_timed_out": False,
            "npm_returncode": None,
        }
        preinstall_json_path = (
            Path(__file__).resolve().parents[3] / "configs" / "preinstall_package.json"
        )
        if not preinstall_json_path.exists():
            logger.warning(
                "[OpenCore] Preinstall package.json not found at %s; skipping preinstall",
                preinstall_json_path,
            )
            return metrics
        try:
            workspace_pkg = Path(tmpdir) / "package.json"
            workspace_pkg.write_text(
                preinstall_json_path.read_text(encoding="utf-8"), encoding="utf-8"
            )
            if npm_install_semaphore is not None:
                npm_wait_start = time.monotonic()
                async with npm_install_semaphore:
                    metrics["npm_wait_ms"] = int((time.monotonic() - npm_wait_start) * 1000)
                    proc_start = time.monotonic()
                    proc = await asyncio.create_subprocess_exec(
                        "npm", "install",
                        "--no-audit", "--no-fund", "--prefer-offline",
                        cwd=tmpdir,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                    )
                    stdout, stderr = await asyncio.wait_for(
                        proc.communicate(), timeout=_NPM_PREINSTALL_TIMEOUT_S,
                    )
                    close_subprocess_transports(proc)
                    metrics["npm_duration_ms"] = int((time.monotonic() - proc_start) * 1000)
            else:
                proc_start = time.monotonic()
                proc = await asyncio.create_subprocess_exec(
                    "npm", "install",
                    "--no-audit", "--no-fund", "--prefer-offline",
                    cwd=tmpdir,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(), timeout=_NPM_PREINSTALL_TIMEOUT_S,
                )
                close_subprocess_transports(proc)
                metrics["npm_duration_ms"] = int((time.monotonic() - proc_start) * 1000)
            metrics["npm_returncode"] = proc.returncode
            if proc.returncode != 0:
                logger.warning(
                    "[OpenCore] npm preinstall exited %d (stderr: %s); "
                    "build agent will run its own install",
                    proc.returncode,
                    (stderr or b"").decode("utf-8", errors="replace")[:500],
                )
            else:
                logger.debug("[OpenCore] npm preinstall complete in %s", tmpdir)
        except asyncio.TimeoutError:
            metrics["npm_timed_out"] = True
            logger.warning(
                "[OpenCore] npm preinstall timed out in %s; "
                "build agent will run its own install",
                tmpdir,
            )
        except Exception:
            logger.exception(
                "[OpenCore] npm preinstall failed in %s; "
                "build agent will run its own install",
                tmpdir,
            )
        return metrics

    _preinstall_task = asyncio.create_task(_run_npm_preinstall())

    try:
        # Initialize worker tracking before any early-return paths that
        # lead to the cleanup finally block below.
        worker_executors: List[CdpPlaywrightExecutor] = []
        worker_info: List[Dict[str, Any]] = []
        # Track Chromium permits acquired so the finally block can release them.
        # Must be initialized before any early-return path that hits the outer
        # finally block.  Stored as a single-element list so _scale_workers_bg
        # can write into it.
        _chromium_permits_acquired: list[int] = [0]
        registry = AgentRegistry(agents_dir)
        agents = registry.load()
        dependency_layers = build_dependency_layers(agents)
        builders = [a for a in agents if (a.stage or "evaluate") == "build"]
        for agent in agents:
            planned_count = 1
            if planned_tasks_by_agent and agent.id in planned_tasks_by_agent:
                planned_count = max(1, len(planned_tasks_by_agent[agent.id]))
            elif planned_tasks and (agent_ids is None or len(agent_ids) == 1):
                planned_count = max(1, len(planned_tasks))
            await _emit_task_progress(
                task_progress_callback,
                task_id,
                {
                    "kind": "agent",
                    "agent_id": agent.id,
                    "status": "pending",
                    "steps_completed": 0,
                    "steps_total": agent.runtime.max_steps * planned_count,
                    "current_task_title": None,
                    "current_task_index": 0,
                    "current_task_total": max(0, planned_count if planned_count > 1 else 0),
                },
            )

        if not builders:
            raise RuntimeError("No build-stage agent configured in agents/")

        builder = builders[0]
        build_phase_agent = builder.model_copy(
            deep=True,
        )
        task_context = _build_build_phase_task_context(
            workspace=tmpdir,
            artifacts_path=artifacts_path,
            process_log_dir=process_log_dir,
            raw_markdown=raw_markdown,
        )
        orchestrator = AgenticOrchestrator(
            config,
            agents_dir=agents_dir,
            max_parallel=max_parallel,
            subtask_semaphore=subtask_semaphore,
        )
        await _emit_task_progress(
            task_progress_callback,
            task_id,
            {
                "kind": "task",
                "status": "running",
                "phase": "build",
            },
        )
        await _emit_task_progress(
            task_progress_callback,
            task_id,
            {
                "kind": "agent",
                "agent_id": builder.id,
                "status": "running",
                "steps_completed": 0,
                "steps_total": builder.runtime.max_steps,
                "current_task_title": "build phase",
                "current_task_index": 0,
                "current_task_total": 0,
                "queue_reason": None,
            },
        )
        # Ensure preinstall is complete before the build agent starts writing
        # files — otherwise agent-written content could conflict with the
        # preinstall package.json.
        if not _preinstall_task.done():
            await _emit_task_progress(
                task_progress_callback,
                task_id,
                {
                    "kind": "agent",
                    "agent_id": builder.id,
                    "status": "running",
                    "steps_completed": 0,
                    "steps_total": builder.runtime.max_steps,
                    "current_task_title": "build phase",
                    "current_task_index": 0,
                    "current_task_total": 0,
                    "queue_reason": "waiting_npm_preinstall",
                },
            )
        preinstall_metrics = await _preinstall_task
        # Gate build-agent execution with build_semaphore to avoid
        # overwhelming the LLM API and local resources when many rows
        # enter the build phase concurrently.
        _build_gate = build_semaphore or _noop_semaphore()
        build_wait_start = time.monotonic()
        await _emit_task_progress(
            task_progress_callback,
            task_id,
            {
                "kind": "agent",
                "agent_id": builder.id,
                "status": "running",
                "steps_completed": 0,
                "steps_total": builder.runtime.max_steps,
                "current_task_title": "build phase",
                "current_task_index": 0,
                "current_task_total": 0,
                "queue_reason": "waiting_build_slot",
            },
        )
        async with _build_gate:
            build_wait_ms = int((time.monotonic() - build_wait_start) * 1000)
            await _emit_task_progress(
                task_progress_callback,
                task_id,
                {
                    "kind": "agent",
                    "agent_id": builder.id,
                    "status": "running",
                    "steps_completed": 0,
                    "steps_total": builder.runtime.max_steps,
                    "current_task_title": "build phase",
                    "current_task_index": 0,
                    "current_task_total": 0,
                    "build_wait_ms": build_wait_ms,
                    "npm_wait_ms": int(preinstall_metrics.get("npm_wait_ms", 0) or 0),
                    "npm_duration_ms": int(preinstall_metrics.get("npm_duration_ms", 0) or 0),
                    "queue_reason": None,
                },
            )
            build_report = await orchestrator.run(
            "",
            _NullExecutor(),
            source_files={},
            user_query="",
            planned_tasks_by_agent={builder.id: _builder_planned_tasks(build_phase_agent)},
            shared_task_context_by_agent={builder.id: task_context},
            agent_ids=[builder.id],
            query_specific_main_task_count=0,
            extra_context={"workspace_root": tmpdir, "artifacts_path": artifacts_path},
            progress_callback=lambda agent_id, payload, current_task_id=task_id: _emit_task_progress(
                task_progress_callback,
                current_task_id,
                {"kind": "agent", "agent_id": agent_id, **payload},
            ),
        )
        build_payloads = (
            build_report.get("agents")
            if isinstance(build_report, dict) and isinstance(build_report.get("agents"), list)
            else []
        )
        if not build_payloads:
            return {
                "verdict": {"passed": False, "reason": "Build orchestration produced no builder result."},
                **runtime_paths,
                "runtime_paths": runtime_paths,
            }
        build_result = build_payloads[0]
        if build_result is None:
            return {
                "verdict": {"passed": False, "reason": "Build orchestration produced a null builder result."},
                **runtime_paths,
                "runtime_paths": runtime_paths,
            }
        build_runtime = build_result.get("runtime") if isinstance(build_result.get("runtime"), dict) else {}
        if not isinstance(build_runtime, dict):
            build_runtime = {}
            build_result["runtime"] = build_runtime
        build_runtime["build_wait_ms"] = build_wait_ms
        build_runtime["npm_wait_ms"] = int(preinstall_metrics.get("npm_wait_ms", 0) or 0)
        build_runtime["npm_duration_ms"] = int(preinstall_metrics.get("npm_duration_ms", 0) or 0)
        await _emit_task_progress(
            task_progress_callback,
            task_id,
            {
                "kind": "agent",
                "agent_id": builder.id,
                "status": str(build_result.get("status", "completed") or "completed"),
                "steps_completed": int(build_runtime.get("steps_used", 0) or 0),
                "steps_total": builder.runtime.max_steps,
                "current_task_title": None,
                "current_task_index": 0,
                "current_task_total": 0,
                "build_wait_ms": build_wait_ms,
                "npm_wait_ms": int(preinstall_metrics.get("npm_wait_ms", 0) or 0),
                "npm_duration_ms": int(preinstall_metrics.get("npm_duration_ms", 0) or 0),
                "end_reason": str(build_result.get("end_reason", "") or ""),
                "queue_reason": None,
            },
        )

        artifacts = _read_artifacts_snapshot(artifacts_path)

        source_files = collect_workspace_files(tmpdir)
        build_integrity_error = _validate_build_phase_execution(
            source_files=source_files,
            build_result=build_result,
        )
        if build_integrity_error:
            build_result = _force_build_failure(build_result, build_integrity_error)
            artifacts = {"state": "failed", "error": build_integrity_error}
            Path(artifacts_path).write_text(
                json.dumps(artifacts, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        # 兜底：agent 被超时/外部终止时，框架自动写 failed artifacts，
        # 确保 finally 清理逻辑能获取到残留进程的 PID 和端口信息。
        if artifacts.get("state") != "ready" and not Path(artifacts_path).exists():
            Path(artifacts_path).write_text(
                json.dumps(artifacts, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        build_handoff = _make_build_handoff(
            builder,
            build_result,
            artifacts,
            process_log_dir,
            source_files,
            artifacts_path,
        )
        pipeline_handoffs: Dict[str, AgentHandoff] = {builder.id: build_handoff}
        runtime_agent_payloads: List[Dict[str, Any]] = [build_result]

        if evaluate_agent_gate is not None:
            await evaluate_agent_gate.mark_build_complete(task_id)

        if artifacts.get("state") != "ready":
            # --- Surface the real build error ---
            build_status = str(build_result.get("status") or "")
            build_end_reason = str(build_result.get("end_reason") or "").strip()
            build_runtime = build_result.get("runtime") if isinstance(build_result.get("runtime"), dict) else {}
            build_steps_used = int(build_runtime.get("steps_used", 0) or 0)
            artifacts_error = str(artifacts.get("error") or "").strip()
            # Build a single-line diagnostic summary that includes the most
            # useful signal (end_reason is richest for timeouts/step-exhaustion,
            # artifacts.error for explicit failure).
            diagnostic = build_end_reason or artifacts_error or f"build agent status={build_status}"
            logger.error(
                "[OpenCore] Build failed. status=%s end_reason=%s artifacts_error=%s steps_used=%d/%d",
                build_status,
                build_end_reason or "(none)",
                artifacts_error or "(none)",
                build_steps_used,
                builder.runtime.max_steps,
            )
            # ---
            selected_agent_ids = set(agent_ids or [agent.id for agent in agents])
            path_review_agents = [
                agent for agent in agents
                if agent.id != builder.id
                and agent.id in selected_agent_ids
                and _is_build_path_review_agent(agent)
            ]
            if path_review_agents:
                await _emit_task_progress(
                    task_progress_callback,
                    task_id,
                    {
                        "kind": "task",
                        "status": "running",
                        "phase": "review",
                        "reason": "Build failed; running build-path review agents on logs and source only.",
                        "error": diagnostic,
                    },
                )

                async def _run_path_review_agent(agent: AgentConfig) -> tuple[AgentConfig, Dict[str, Any]]:
                    shared_task_context = _build_path_review_task_context(build_handoff)
                    evaluate_wait_start = time.monotonic()
                    await _emit_task_progress(
                        task_progress_callback,
                        task_id,
                        {
                            "kind": "agent",
                            "agent_id": agent.id,
                            "status": "running",
                            "steps_completed": 0,
                            "steps_total": agent.runtime.max_steps,
                            "current_task_title": None,
                            "current_task_index": 0,
                            "current_task_total": 0,
                            "queue_reason": "waiting_evaluate_slot",
                        },
                    )
                    if evaluate_agent_gate is not None:
                        await evaluate_agent_gate.acquire(task_id)
                    try:
                        evaluate_agent_wait_ms = int((time.monotonic() - evaluate_wait_start) * 1000)
                        await _emit_task_progress(
                            task_progress_callback,
                            task_id,
                            {
                                "kind": "agent",
                                "agent_id": agent.id,
                                "status": "running",
                                "steps_completed": 0,
                                "steps_total": agent.runtime.max_steps,
                                "current_task_title": None,
                                "current_task_index": 0,
                                "current_task_total": 0,
                                "evaluate_agent_wait_ms": evaluate_agent_wait_ms,
                                "queue_reason": None,
                            },
                        )
                        # Build a focused read-only system prompt for path review
                        path_review_allowed_tools = [
                            t for t in agent.allowed_tools
                            if t in {"local_fs_read", "read_file", "search_source", "submit_verdict"}
                        ]
                        agent_report = await orchestrator.run(
                            "",
                            _NullExecutor(),
                            source_files=source_files,
                            user_query=query,
                            agent_ids=[agent.id],
                            shared_task_context_by_agent={agent.id: shared_task_context},
                            dependency_handoffs={agent.id: [build_handoff.to_dict()]},
                            extra_context={"workspace_root": tmpdir, "artifacts_path": artifacts_path},
                            progress_callback=lambda agent_id, payload, current_task_id=task_id: _emit_task_progress(
                                task_progress_callback,
                                current_task_id,
                                {"kind": "agent", "agent_id": agent_id, **payload},
                            ),
                            agent_overrides={
                                agent.id: {
                                    "allowed_tools": path_review_allowed_tools,
                                },
                            },
                        )
                    finally:
                        if evaluate_agent_gate is not None:
                            await evaluate_agent_gate.release(task_id)
                    agent_payloads = agent_report.get("agents", []) or []
                    if not agent_payloads:
                        return agent, _blocked_agent_payload(agent, [builder.id])
                    agent_payload = agent_payloads[0]
                    agent_runtime = agent_payload.get("runtime") if isinstance(agent_payload.get("runtime"), dict) else {}
                    if not isinstance(agent_runtime, dict):
                        agent_runtime = {}
                        agent_payload["runtime"] = agent_runtime
                    agent_runtime["evaluate_agent_wait_ms"] = evaluate_agent_wait_ms
                    if not _build_path_review_used_handoff_paths(agent_payload, build_handoff):
                        agent_payload = _infra_agent_payload(
                            agent,
                            "Build-path review is invalid: reviewer did not inspect any required build-path file with local_fs_read before submitting a verdict.",
                        )
                    return agent, agent_payload

                path_review_results = await asyncio.gather(*[_run_path_review_agent(agent) for agent in path_review_agents])
                for agent, agent_payload in path_review_results:
                    pipeline_handoffs[agent.id] = _make_runtime_handoff(
                        agent,
                        agent_payload,
                        runtime_paths,
                        "",
                        "",
                    )
                    runtime_agent_payloads.append(agent_payload)
            else:
                await _emit_task_progress(
                    task_progress_callback,
                    task_id,
                    {
                        "kind": "task",
                        "status": "running",
                        "phase": "review",
                        "reason": f"Build failed: {diagnostic}",
                    },
                )

            blocked_agents: List[Dict[str, Any]] = []
            for agent in agents:
                if agent.id == builder.id or agent.id in {review_agent.id for review_agent in path_review_agents}:
                    continue
                await _emit_task_progress(
                    task_progress_callback,
                    task_id,
                    {
                        "kind": "agent",
                        "agent_id": agent.id,
                        "status": "blocked",
                        "steps_completed": 0,
                        "steps_total": agent.runtime.max_steps,
                        "current_task_title": None,
                        "current_task_index": 0,
                        "current_task_total": 0,
                        "end_reason": f"Blocked by {builder.id}",
                    },
                )
                blocked_agents.append(_blocked_agent_payload(agent, [builder.id]))
            combined_report = build_agentic_report(
                [*runtime_agent_payloads, *blocked_agents],
                task_id=task_id,
            )
            combined_report["build_result"] = build_result
            combined_report["artifacts"] = artifacts
            combined_report.update(runtime_paths)
            combined_report["runtime_paths"] = runtime_paths
            combined_report["pipeline_handoffs"] = [handoff.to_dict() for handoff in pipeline_handoffs.values()]
            _attach_report_path_evidence(combined_report)
            combined_report["verdict"] = {
                "passed": False,
                "status": "failed",
                "reason": f"Build failed: {artifacts.get('error', 'unknown')}",
            }
            return _slim_combined_report(combined_report)

        # --- build_success=failed check: skip evaluation agents with score 0 ---
        # Even when artifacts.state == "ready", the build agent may have scored
        # build_success=0 (e.g. blank page, broken UI).  In that case downstream
        # evaluation agents must not run and get score 0 automatically.
        _build_success_verdict: str = ""
        for _dim in (build_result.get("dimensions") if isinstance(build_result.get("dimensions"), list) else []):
            if isinstance(_dim, dict) and str(_dim.get("dimension_id") or "").strip() == "build_success":
                _build_success_verdict = str(_dim.get("verdict") or "").strip().lower()
                break
        if _build_success_verdict == "failed":
            logger.warning(
                "[OpenCore] build_success=failed (score=0) but artifacts state=ready — "
                "skipping evaluation agents with score 0 for %s",
                task_id,
            )
            _failure_reason = _build_result_failure_reason(build_result) or "build_success=failed"
            selected_agent_ids = set(agent_ids or [agent.id for agent in agents])
            _skipped_agents: List[Dict[str, Any]] = []
            for _agent in agents:
                if _agent.id == builder.id or _agent.id not in selected_agent_ids:
                    continue
                _skipped_agents.append(_skipped_agent_payload(
                    _agent,
                    f"Skipped: {_failure_reason}",
                ))
            combined_report = build_agentic_report(
                [build_result, *_skipped_agents],
                task_id=task_id,
            )
            combined_report["build_result"] = build_result
            combined_report["artifacts"] = artifacts
            combined_report.update(runtime_paths)
            combined_report["runtime_paths"] = runtime_paths
            combined_report["pipeline_handoffs"] = [handoff.to_dict() for handoff in pipeline_handoffs.values()]
            _attach_report_path_evidence(combined_report)
            combined_report["verdict"] = {
                "passed": False,
                "status": "failed",
                "reason": _failure_reason,
            }
            return _slim_combined_report(combined_report)

        ports = artifacts.get("ports", [])
        if not isinstance(ports, list) or not ports:
            return {
                "verdict": {
                    "passed": False,
                    "reason": "Build artifacts missing required ports[0].url",
                },
                "build_result": build_result,
                "artifacts": artifacts,
                **runtime_paths,
                "runtime_paths": runtime_paths,
            }

        first_port = ports[0] if isinstance(ports[0], dict) else None
        app_url = str((first_port or {}).get("url", "")).strip()
        if not app_url:
            return {
                "verdict": {
                    "passed": False,
                    "reason": "Build artifacts missing required app_url at ports[0].url",
                },
                "build_result": build_result,
                "artifacts": artifacts,
                **runtime_paths,
                "runtime_paths": runtime_paths,
            }

        cdp_url = str(artifacts.get("cdp_url", "")).strip()
        if not cdp_url:
            return {
                "verdict": {
                    "passed": False,
                    "reason": "Build artifacts missing required cdp_url",
                },
                "build_result": build_result,
                "artifacts": artifacts,
                **runtime_paths,
                "runtime_paths": runtime_paths,
            }

        # Gate memory + CDP connection rate before starting the primary executor.
        # _wait_for_available_resources blocks until at least 256 MiB is free;
        # acquire_cdp_permit throttles concurrent connect_over_cdp calls so a
        # burst of builds completing simultaneously doesn't spike memory or
        # overwhelm Playwright's internal connection pool.
        await _wait_for_available_resources(1, False, max_wait_s=120.0)
        await acquire_cdp_permit()
        executor = CdpPlaywrightExecutor(_LocalCdpStub(cdp_url), app_url)
        _executor_startup_error: Optional[str] = None
        try:
            await executor.start_service(timeout=config.cdp_connection_timeout)
        except Exception as _start_exc:
            _executor_startup_error = str(_start_exc)
            logger.error(
                "[OpenCore] Evaluate executor start_service failed for %s: %s",
                app_url, _start_exc,
            )

        if _executor_startup_error is not None:
            # Executor failed to load the app page — surface the error
            # while preserving the full build context so the web dashboard
            # can show what the build produced and why evaluate couldn't start.
            selected_agent_ids = set(agent_ids or [agent.id for agent in agents])
            startup_failed_agents: List[Dict[str, Any]] = []
            for agent in agents:
                if agent.id == builder.id:
                    continue
                if agent.id not in selected_agent_ids:
                    continue
                startup_failed_agents.append(normalize_agent_result({
                    "agent_id": agent.id,
                    "score": None,
                    "score_scale": "0-100",
                    "end_reason": f"Evaluate executor failed to load app: {_executor_startup_error}",
                    **_empty_runtime_payload("failed", error=_executor_startup_error),
                }))
            combined_report = build_agentic_report(
                [build_result, *startup_failed_agents],
                task_id=task_id,
            )
            combined_report["build_result"] = build_result
            combined_report["artifacts"] = artifacts
            combined_report.update(runtime_paths)
            combined_report["runtime_paths"] = runtime_paths
            combined_report["app_link"] = app_url
            combined_report["preview_url"] = app_url
            combined_report["pipeline_handoffs"] = [handoff.to_dict() for handoff in pipeline_handoffs.values()]
            _attach_report_path_evidence(combined_report)
            combined_report["verdict"] = {
                "passed": False,
                "status": "failed",
                "reason": f"Evaluate executor failed to load app: {_executor_startup_error}",
            }
            return _slim_combined_report(combined_report)

        # --- Scale up additional worker instances for subtask distribution ---
        # Each worker gets its own Chromium (and optionally dev server), so
        # CDP operations across concurrent subtasks don't contend on a single
        # browser process.
        worker_count = config.per_row_worker_count
        worker_future: Optional[asyncio.Task] = None
        if worker_count > 1:

            async def _scale_workers_bg() -> List[CdpPlaywrightExecutor]:
                """Background task: scale up workers concurrently with evaluation setup."""
                # Acquire global Chromium permits before creating workers.
                # Blocks until enough permits are available (i.e. another row
                # has finished and released its permits).  This prevents total
                # Chromium processes across all rows from exceeding
                # MAX_TOTAL_CHROMIUM_WORKERS.
                if total_chromium_semaphore:
                    for _ in range(worker_count):
                        await total_chromium_semaphore.acquire()
                    _chromium_permits_acquired[0] = worker_count
                try:
                    _wi = await _scale_up_workers(
                        workspace_root=tmpdir,
                        worker_count=worker_count,
                        task_id=task_id,
                        session_id=artifacts_path,
                        primary_app_url=app_url,
                    )
                except BaseException:
                    # Release permits if scale-up itself threw (e.g. port allocation failed)
                    _released = 0
                    if total_chromium_semaphore and _chromium_permits_acquired[0]:
                        for _ in range(_chromium_permits_acquired[0]):
                            total_chromium_semaphore.release()
                            _released += 1
                        logger.info(
                            "[scale_up] Released %d Chromium permits after scale-up failure for %s",
                            _released, task_id,
                        )
                        _chromium_permits_acquired[0] = 0
                    raise
                # Store worker info for cleanup (PID killing + port lease release).
                # worker_info is consumed in the outer finally block below.
                worker_info.extend(_wi)
                _executors: List[CdpPlaywrightExecutor] = []
                for _w in _wi:
                    _we = CdpPlaywrightExecutor(_LocalCdpStub(_w["cdp_url"]), _w["app_url"])
                    await acquire_cdp_permit()
                    await _we.start_service(timeout=config.cdp_connection_timeout)
                    _executors.append(_we)
                if _executors:
                    logger.info(
                        "[OpenCore] Scaled up %d worker(s) for row %s",
                        len(_executors), task_id,
                    )
                return _executors

            worker_future = asyncio.ensure_future(_scale_workers_bg())

        try:
            orchestrator = AgenticOrchestrator(
                config,
                agents_dir=agents_dir,
                max_parallel=1 if config.task_synthesis_mode == "tree" else max_parallel,
                # subtask_semaphore deliberately NOT passed here —
                # evaluate-agent concurrency is already controlled by
                # evaluate_agent_gate (RowPriorityGate).  Passing the shared
                # global pool creates a double-gating deadlock where agents
                # hold gate slots while their subtasks wait for pool slots.
            )
            selected_agent_ids = set(agent_ids or [agent.id for agent in agents])
            page_context = await _gather_planning_page_context(executor)
            await _emit_task_progress(
                task_progress_callback,
                task_id,
                {
                    "kind": "task",
                    "status": "running",
                    "phase": "evaluate",
                },
            )

            for layer in dependency_layers:
                runnable = [
                    agent for agent in layer
                    if agent.id != builder.id and agent.id in selected_agent_ids
                ]

                async def _run_one_agent(agent: Any) -> Any:
                    blocked_by = [
                        dep_id for dep_id in agent.depends_on
                        if dep_id not in pipeline_handoffs or pipeline_handoffs[dep_id].status != "completed"
                    ]
                    if blocked_by:
                        await _emit_task_progress(
                            task_progress_callback,
                            task_id,
                            {
                                "kind": "agent",
                                "agent_id": agent.id,
                                "status": "blocked",
                                "steps_completed": 0,
                                "steps_total": agent.runtime.max_steps,
                                "current_task_title": None,
                                "current_task_index": 0,
                                "current_task_total": 0,
                                "end_reason": f"Blocked by {', '.join(blocked_by)}",
                            },
                        )
                        return _blocked_agent_payload(agent, blocked_by)

                    agent_dependency_handoffs = {
                        agent.id: [pipeline_handoffs[dep_id].to_dict() for dep_id in agent.depends_on if dep_id in pipeline_handoffs]
                    }
                    planned_task_list = planned_tasks if planned_tasks and len(runnable) == 1 else None
                    agent_planned_task_map = (
                        {agent.id: planned_tasks_by_agent[agent.id]}
                        if planned_tasks_by_agent and agent.id in planned_tasks_by_agent
                        else None
                    )
                    shared_query_main_tasks = await _load_agent_fixed_query_main_tasks(
                        query=query,
                        agent=agent,
                        config=config,
                        fixed_task_db=fixed_task_db,
                    )
                    agent_query_task_count = (
                        len(shared_query_main_tasks) if shared_query_main_tasks
                        else _resolve_query_specific_main_task_count(agent, config)
                    )
                    agent_executor = CdpPlaywrightExecutor(_LocalCdpStub(cdp_url), app_url)
                    try:
                        shared_task_context = None
                        if _is_build_path_review_agent(agent) and builder.id in pipeline_handoffs:
                            shared_task_context = _build_path_review_task_context(pipeline_handoffs[builder.id])
                        evaluate_wait_start = time.monotonic()
                        await _emit_task_progress(
                            task_progress_callback,
                            task_id,
                            {
                                "kind": "agent",
                                "agent_id": agent.id,
                                "status": "running",
                                "steps_completed": 0,
                                "steps_total": agent.runtime.max_steps,
                                "current_task_title": None,
                                "current_task_index": 0,
                                "current_task_total": 0,
                                "queue_reason": "waiting_evaluate_slot",
                            },
                        )
                        _is_heavy_exec = heavy_exec_semaphore is not None and "heavy_exec" in (agent.tags or [])
                        if evaluate_agent_gate is not None:
                            await evaluate_agent_gate.acquire(task_id)
                        if _is_heavy_exec:
                            await heavy_exec_semaphore.acquire()
                        try:
                            evaluate_agent_wait_ms = int((time.monotonic() - evaluate_wait_start) * 1000)
                            await acquire_cdp_permit()
                            cdp_connect_start = time.monotonic()
                            await agent_executor.start_service(timeout=config.cdp_connection_timeout)
                            cdp_connect_ms = int((time.monotonic() - cdp_connect_start) * 1000)
                            await _emit_task_progress(
                                task_progress_callback,
                                task_id,
                                {
                                    "kind": "agent",
                                    "agent_id": agent.id,
                                    "status": "running",
                                    "steps_completed": 0,
                                    "steps_total": agent.runtime.max_steps,
                                    "current_task_title": None,
                                    "current_task_index": 0,
                                    "current_task_total": 0,
                                    "evaluate_agent_wait_ms": evaluate_agent_wait_ms,
                                    "cdp_connect_ms": cdp_connect_ms,
                                    "queue_reason": None,
                                },
                            )
                            agent_report = await orchestrator.run(
                                app_url,
                                agent_executor,
                                source_files=source_files,
                                user_query=query,
                                page_context=page_context,
                                planned_tasks=planned_task_list,
                                planned_tasks_by_agent=agent_planned_task_map,
                                shared_task_context_by_agent=({agent.id: shared_task_context} if shared_task_context else None),
                                shared_query_main_tasks=shared_query_main_tasks,
                                agent_ids=[agent.id],
                                dependency_handoffs=agent_dependency_handoffs,
                                extra_context={"workspace_root": str(tmpdir) if tmpdir else "", "artifacts_path": artifacts_path},
                                progress_callback=lambda agent_id, payload, current_task_id=task_id: _emit_task_progress(
                                    task_progress_callback,
                                    current_task_id,
                                    {"kind": "agent", "agent_id": agent_id, **payload},
                                ),
                                query_specific_main_task_count=agent_query_task_count,
                                worker_executors=worker_future,
                            )
                        finally:
                            if _is_heavy_exec:
                                heavy_exec_semaphore.release()
                            if evaluate_agent_gate is not None:
                                await evaluate_agent_gate.release(task_id)
                    finally:
                        await agent_executor.shutdown()
                    agent_payloads = agent_report.get("agents", []) or []
                    if not agent_payloads:
                        return _blocked_agent_payload(agent, ["pipeline_execution_error"])
                    agent_payload = agent_payloads[0]
                    agent_runtime = agent_payload.get("runtime") if isinstance(agent_payload.get("runtime"), dict) else {}
                    if not isinstance(agent_runtime, dict):
                        agent_runtime = {}
                        agent_payload["runtime"] = agent_runtime
                    agent_runtime["evaluate_agent_wait_ms"] = evaluate_agent_wait_ms
                    agent_runtime["cdp_connect_ms"] = cdp_connect_ms
                    if _is_build_path_review_agent(agent) and builder.id in pipeline_handoffs:
                        if not _build_path_review_used_handoff_paths(agent_payload, pipeline_handoffs[builder.id]):
                            agent_payload = _infra_agent_payload(
                                agent,
                                "Build-path review is invalid: reviewer did not inspect any required build-path file with local_fs_read before submitting a verdict.",
                            )
                    pipeline_handoffs[agent.id] = _make_runtime_handoff(
                        agent,
                        agent_payload,
                        runtime_paths,
                        app_url,
                        cdp_url,
                    )
                    return agent_payload

                layer_payloads = await asyncio.gather(*[_run_one_agent(agent) for agent in runnable])
                runtime_agent_payloads.extend(layer_payloads)

            combined_report = build_agentic_report(
                runtime_agent_payloads,
                task_id=task_id,
            )
            combined_report["build_result"] = build_result
            combined_report["artifacts"] = artifacts
            combined_report.update(runtime_paths)
            combined_report["runtime_paths"] = runtime_paths
            combined_report["app_link"] = app_url
            combined_report["preview_url"] = app_url
            combined_report["pipeline_handoffs"] = [handoff.to_dict() for handoff in pipeline_handoffs.values()]
            _attach_report_path_evidence(combined_report)
            combined_report["verdict"] = derive_open_report_verdict(combined_report)
            return _slim_combined_report(combined_report)
        finally:
            await executor.shutdown()
            # Resolve and shutdown worker executors (may be a background task)
            _worker_list: List[Any] = []
            if isinstance(worker_future, asyncio.Task):
                try:
                    _resolved = await worker_future
                    _worker_list = _resolved or []
                except Exception:
                    _worker_list = []
            else:
                _worker_list = worker_executors
            for we in _worker_list:
                try:
                    await we.shutdown()
                except Exception:
                    pass

    finally:
        if not _preinstall_task.done():
            _preinstall_task.cancel()
            try:
                await _preinstall_task
            except asyncio.CancelledError:
                pass
            except Exception:
                pass
        try:
            from tools.ports import release as _release_port
        except ImportError:
            _release_port = lambda _: None

        # Cleanup primary build artifacts (dev server, Chromium, ports)
        try:
            arts = json.loads(Path(artifacts_path).read_text(encoding="utf-8"))
            for pid_key in ("dev_pid", "chromium_pid"):
                pid = arts.get(pid_key)
                if pid:
                    _terminate_process_group(int(pid), name=f"primary_{pid_key}")
            for lease_key in ("port_lease_id", "cdp_port_lease_id"):
                lease_id = arts.get(lease_key)
                if lease_id:
                    try:
                        _release_port(lease_id)
                    except Exception:
                        pass
        except Exception:
            pass

        # Cleanup worker instances: kill processes and release ports
        if worker_info:
            for wi in worker_info:
                for pid_key in ("dev_pid", "chromium_pid"):
                    pid = wi.get(pid_key)
                    if pid:
                        _terminate_process_group(int(pid), name=f"worker_{pid_key}")
                for lease_key in ("app_lease_id", "cdp_lease_id"):
                    lease_id = wi.get(lease_key)
                    if lease_id:
                        try:
                            _release_port(lease_id)
                        except Exception:
                            pass
        # --- Fallback: 扫描 /proc 杀掉残留 chromium / dev server 进程 ---
        # 双重匹配:
        #   1) cwd 落在 tmpdir 下(适用 npm/node 等);
        #   2) cmdline 引用 tmpdir 或本行分配的 CDP 端口(chromium renderer 的 cwd 通常不在 tmpdir,
        #      所以光看 cwd 会漏)。
        cdp_port_strs: set[str] = set()
        try:
            arts2 = json.loads(Path(artifacts_path).read_text(encoding="utf-8"))
            for k in ("cdp_url",):
                v = str(arts2.get(k, "") or "")
                m = re.search(r":(\d{2,5})", v)
                if m:
                    cdp_port_strs.add(m.group(1))
        except Exception:
            pass
        if worker_info:
            for wi in worker_info:
                p = wi.get("cdp_port")
                if p:
                    cdp_port_strs.add(str(p))
        try:
            killed_orphans = 0
            for proc_entry in Path("/proc").iterdir():
                if not proc_entry.name.isdigit():
                    continue
                pid = int(proc_entry.name)
                match = False
                # cwd match
                try:
                    cwd_link = proc_entry / "cwd"
                    if cwd_link.is_symlink():
                        resolved = os.readlink(str(cwd_link))
                        if resolved.startswith(tmpdir):
                            match = True
                except (OSError, ValueError):
                    pass
                # cmdline match — catches chromium subprocesses whose cwd is /
                if not match:
                    try:
                        cmdline_bytes = (proc_entry / "cmdline").read_bytes()
                        cmdline = cmdline_bytes.replace(b"\x00", b" ").decode("utf-8", errors="ignore")
                        if tmpdir in cmdline:
                            match = True
                        elif cdp_port_strs:
                            for port in cdp_port_strs:
                                if f"--remote-debugging-port={port}" in cmdline:
                                    match = True
                                    break
                    except (OSError, ValueError):
                        pass
                if match:
                    try:
                        os.kill(pid, signal.SIGKILL)
                        killed_orphans += 1
                    except (OSError, ProcessLookupError):
                        pass
            if killed_orphans:
                logger.info("[cleanup] /proc fallback killed %d orphan processes for %s",
                            killed_orphans, task_id)
        except Exception:
            pass
        # ---------------------------------------------------------------
        # Release global Chromium permits now that all worker processes
        # have been killed and port leases released.
        _p_cnt = _chromium_permits_acquired[0]
        if total_chromium_semaphore and _p_cnt:
            for _ in range(_p_cnt):
                total_chromium_semaphore.release()
            logger.debug(
                "[cleanup] Released %d Chromium permits for %s", _p_cnt, task_id,
            )
        if os.environ.get("EVAL_KEEP_WORKSPACE") == "1":
            logger.info("[cleanup] EVAL_KEEP_WORKSPACE=1: kept workspace %s",
                        tmpdir)
        else:
            shutil.rmtree(tmpdir, ignore_errors=True)
        _tool_session.reset_active_session(active_session_token)
        _tool_session.clear_session(artifacts_path)


async def run_open_batch(
    *,
    config: "Config",
    rows: List[Dict[str, Any]],
    max_parallel: int = 1,
    agents_dir: str = "agents",
    existing_index: Optional[Dict[str, Dict[str, Any]]] = None,
    rerun_nonpassed: bool = False,
    audit_log_fn: Optional[Callable[[str, str, str], None]] = None,
    planned_tasks: Optional[List["PlannedTask"]] = None,
    planned_tasks_by_agent: Optional[Dict[str, List["PlannedTask"]]] = None,
    agent_ids: Optional[List[str]] = None,
    task_descriptors: Optional[List[Dict[str, Any]]] = None,
    task_progress_callback: Optional[Callable[[str, Dict[str, Any]], Any]] = None,
    on_result: Any = None,
    suppress_skipped_on_result: bool = False,
    save_interval: int = 5,
    force_rerun_agents: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Run open-mode evaluation in batch for already loaded JSONL rows.

    Each completed row is reported via *on_result(index, row_result)*
    immediately (no batching).  *save_interval* is accepted for backward
    compatibility but ignored — callers should write results individually
    to a JSONL file for incremental checkpointing.
    """
    existing = existing_index or {}
    # Row-level semaphore — limits how many query-model pairs (sandboxes/builds)
    # can be in-flight at once. This is intentionally independent from the
    # global subtask budget so high subtask parallelism does not automatically
    # imply high outer-row concurrency.
    row_semaphore = asyncio.Semaphore(max(1, config.row_parallelism))
    _actual_row_par = max(1, config.row_parallelism)
    _actual_subtask_par = max(1, max_parallel)
    # Global Chromium worker cap — prevents total Chromium processes across
    # Chromium worker cap. 0 = disabled (no limit). Env: MAX_TOTAL_CHROMIUM_WORKERS
    _max_total_chromium = int(os.getenv("MAX_TOTAL_CHROMIUM_WORKERS", "0"))
    _eval_agent_par = config.evaluate_agent_parallelism  # 0 = disabled
    _heavy_exec_conc = config.heavy_exec_parallelism      # 0 = disabled
    logger.info(
        "Parallelism config: row=%d build=%d npm=%d eval_agent=%s heavy_exec=%s chromium_cap=%s"
        " | rate limits: llm=%.0f/s cdp=%.0f/s",
        _actual_row_par,
        max(1, config.build_parallelism),
        max(1, config.npm_preinstall_parallelism),
        _eval_agent_par if _eval_agent_par > 0 else "disabled",
        _heavy_exec_conc if _heavy_exec_conc > 0 else "disabled",
        _max_total_chromium if _max_total_chromium > 0 else "disabled",
        float(os.getenv("LLM_CALLS_PER_SECOND", "20")),
        float(os.getenv("CDP_CONNECTS_PER_SECOND", "5")),
    )
    # Global subtask semaphore — shared across all rows/agents so at most
    # max_parallel subtasks execute concurrently in the entire batch.
    global_subtask_semaphore = asyncio.Semaphore(max(1, max_parallel))
    # Serialize npm install at the start of each build so concurrent rows
    # don't compete for the npm cache and disk I/O simultaneously.
    _npm_concurrency = max(1, config.npm_preinstall_parallelism)
    npm_install_semaphore = asyncio.Semaphore(_npm_concurrency)
    # Build-agent execution semaphore — limits how many build agent
    # workflows run concurrently (env: BUILD_PARALLELISM).
    _build_conc = max(1, config.build_parallelism)
    build_semaphore = asyncio.Semaphore(_build_conc)
    # evaluate_agent_gate: None = disabled (rely on LLM/CDP rate limiters only)
    evaluate_agent_gate = RowPriorityGate(_eval_agent_par) if _eval_agent_par > 0 else None
    # total_chromium_semaphore: None = disabled
    total_chromium_semaphore = asyncio.Semaphore(_max_total_chromium) if _max_total_chromium > 0 else None
    # heavy_exec_semaphore: None = disabled
    heavy_exec_semaphore = asyncio.Semaphore(_heavy_exec_conc) if _heavy_exec_conc > 0 else None
    # 3000ms default — at 8+ row parallelism, 0 stagger has every row hitting
    # `npm install` simultaneously, bursting NFS reads and CPU.  3s spread
    # smooths the startup wave with negligible total-time cost.
    start_jitter_ms = max(0, int(os.getenv("EVAL_OPEN_TASK_START_JITTER_MS", "3000")))
    descriptors = task_descriptors or build_monitor_task_descriptors(rows)
    fixed_task_db = _open_fixed_task_db(config)

    # Track completed indices for resume - only store indices, not full results in memory
    # This avoids loading all previous results into memory on resume
    completed_indices: set[int] = set()
    skipped_indices: set[int] = set()

    # Track partial-rows that have agent-level failures and need selective rerun
    # sample_id -> {"good_agents": list[dict], "failed_ids": list[str]}
    _partial_rerun_info: dict[str, dict] = {}

    # Pre-scan to identify which rows should be skipped (without loading full results)
    for index, row in enumerate(rows):
        sample_id = str(row.get("id", "")).strip()
        if sample_id and sample_id in existing:
            existing_status = str(existing[sample_id].get("status", "")).lower()

            # Orphan-blocked detection: build_engineer.verdict.passed=True yet a
            # downstream agent ended in status='blocked' is contradictory — if
            # build passed, the downstream agent should have been allowed to
            # run. Force a selective rerun of those blocked agents regardless
            # of the row's top-level status (including 'passed' rows that would
            # otherwise be skipped).
            if rerun_nonpassed:
                existing_result_orphan = existing[sample_id].get("result", {})
                existing_agents_orphan = (
                    existing_result_orphan.get("agents", [])
                    if isinstance(existing_result_orphan, dict)
                    else []
                )
                build_passed_orphan = False
                for agent in existing_agents_orphan:
                    if (
                        isinstance(agent, dict)
                        and str(agent.get("agent_id") or "").strip() == "build_engineer"
                    ):
                        verdict = agent.get("verdict")
                        if isinstance(verdict, dict) and verdict.get("passed") is True:
                            build_passed_orphan = True
                        break
                if build_passed_orphan:
                    spurious_blocked = [
                        str(a.get("agent_id"))
                        for a in existing_agents_orphan
                        if isinstance(a, dict)
                        and str(a.get("agent_id") or "").strip() != "build_engineer"
                        and str(a.get("status", "")).lower() == "blocked"
                    ]
                    if spurious_blocked:
                        good_agents_orphan = [
                            a for a in existing_agents_orphan
                            if isinstance(a, dict)
                            and str(a.get("agent_id") or "").strip() not in spurious_blocked
                        ]
                        _partial_rerun_info[sample_id] = {
                            "good_agents": good_agents_orphan,
                            "failed_ids": spurious_blocked,
                        }
                        logger.info(
                            "Row %s: build_engineer passed but downstream agents %s "
                            "were blocked (row.status=%s); scheduling them for rerun.",
                            sample_id,
                            spurious_blocked,
                            existing_status,
                        )
                        continue

            # ── force-rerun-agents: override normal skip/resume logic ──
            # When set, every row that has any named agent reruns only those
            # agents, reusing saved results for all others.  Works regardless
            # of the row's top-level status (passed, partial, failed — even if
            # the agent's own status was "completed" with a poor verdict).
            if force_rerun_agents:
                existing_result_fr = existing[sample_id].get("result", {})
                existing_agents_fr = (
                    existing_result_fr.get("agents", [])
                    if isinstance(existing_result_fr, dict)
                    else []
                )
                force_set = set(force_rerun_agents)
                good_agents_fr: list[dict] = []
                failed_ids_fr: list[str] = []
                for agent in existing_agents_fr:
                    if isinstance(agent, dict):
                        aid = str(agent.get("agent_id") or "").strip()
                        if aid in force_set:
                            failed_ids_fr.append(aid)
                        else:
                            good_agents_fr.append(agent)
                if failed_ids_fr:
                    _partial_rerun_info[sample_id] = {
                        "good_agents": good_agents_fr,
                        "failed_ids": failed_ids_fr,
                    }
                    logger.info(
                        "force-rerun '%s': re-running agents %s, reusing %d good agent(s)",
                        sample_id, failed_ids_fr, len(good_agents_fr),
                    )
                    continue  # bypass normal should_skip / partial logic below
                # No matching agents in this row — fall through to normal logic

            should_skip = existing_status in {"passed"} if rerun_nonpassed else True
            if should_skip:
                skipped_indices.add(index)
                completed_indices.add(index)
                # Only store minimal info for checkpoint writing - just enough to identify
                # We don't need the full result in memory
            elif existing_status == "partial" and rerun_nonpassed:
                # Agent-level selective rerun: identify which agents failed
                existing_result = existing[sample_id].get("result", {})
                existing_agents = existing_result.get("agents", []) if isinstance(existing_result, dict) else []
                good_agents: list[dict] = []
                failed_ids: list[str] = []
                for agent in existing_agents:
                    if isinstance(agent, dict):
                        aid = agent.get("agent_id")
                        astat = str(agent.get("status", "")).lower()
                        if astat == "completed":
                            good_agents.append(agent)
                        elif aid:
                            failed_ids.append(aid)
                if failed_ids:
                    _partial_rerun_info[sample_id] = {"good_agents": good_agents, "failed_ids": failed_ids}
                    # Don't add to completed_indices — will re-run build + failed agents
                else:
                    # All agents completed (e.g. all tasks passed), skip
                    skipped_indices.add(index)
                    completed_indices.add(index)

    async def run_row(index: int, row: Dict[str, Any]) -> Dict[str, Any]:
        descriptor = descriptors[index]
        task_id = str(descriptor["task_id"])
        sample_id = str(row.get("id", "")).strip()

        # Check if this index was already completed in a previous run
        if index in completed_indices:
            existing_entry = existing.get(sample_id, {})
            existing_status = str(existing_entry.get("status", "")).lower() if existing_entry else "unknown"
            logger.info("Skipping '%s' (status=%s)", sample_id, existing_status)
            await _emit_task_progress(
                task_progress_callback,
                task_id,
                {
                    "kind": "task",
                    "status": "completed",
                    "phase": "finished",
                    "reason": f"Skipped from resume report (status={existing_status})",
                    "verdict_status": existing_status or None,
                },
            )
            # Carry forward the full previous result so new checkpoints retain
            # the detailed agent-task data from the original run.
            return {
                "id": sample_id,
                "status": existing_status,
                "query": existing_entry.get("query") or row.get("query", ""),
                "model": row.get("model", ""),
                "result": existing_entry.get("result"),
                "error": existing_entry.get("error"),
                "link": existing_entry.get("link") or _get_render_link(row),
            }

        raw_markdown = str(row.get("code") or row.get("markdown") or "").strip()
        query = str(row.get("query", "")).strip()
        if audit_log_fn is not None:
            audit_log_fn(f"evaluate:{sample_id or '?'}", "auto_approve", ".")

        # Agent-level selective rerun: run only build agent + failed agents
        _partial_good_agents: list[dict] | None = None
        _rerun_agent_ids: list[str] | None = agent_ids
        if sample_id in _partial_rerun_info:
            info = _partial_rerun_info.pop(sample_id)
            _rerun_agent_ids = info["failed_ids"]
            _partial_good_agents = info["good_agents"]
            logger.info(
                "Partial rerun for '%s': running %d failed agent(s), reusing %d good agent(s)",
                sample_id, len(_rerun_agent_ids), len(_partial_good_agents),
            )

        row_wait_start = time.monotonic()
        async with row_semaphore:
            row_wait_ms = int((time.monotonic() - row_wait_start) * 1000)
            if start_jitter_ms > 0:
                jitter_delay = random.uniform(0.0, start_jitter_ms / 1000.0)
                if jitter_delay > 0:
                    logger.debug("Task %s applying startup jitter %.3fs", task_id, jitter_delay)
                    await asyncio.sleep(jitter_delay)
            await _emit_task_progress(
                task_progress_callback,
                task_id,
                {
                    "kind": "task",
                    "status": "running",
                    "phase": "queued",
                    "row_wait_ms": row_wait_ms,
                },
            )
            try:
                result = await run_open_single(
                    config=config,
                    task_id=task_id,
                    raw_markdown=raw_markdown,
                    query=query,
                    agents_dir=agents_dir,
                    max_parallel=max_parallel,
                    subtask_semaphore=global_subtask_semaphore,
                    npm_install_semaphore=npm_install_semaphore,
                    build_semaphore=build_semaphore,
                    planned_tasks=planned_tasks,
                    planned_tasks_by_agent=planned_tasks_by_agent,
                    fixed_task_db=fixed_task_db,
                    evaluate_agent_gate=evaluate_agent_gate,
                    agent_ids=_rerun_agent_ids,
                    task_progress_callback=task_progress_callback,
                    total_chromium_semaphore=total_chromium_semaphore,
                    heavy_exec_semaphore=heavy_exec_semaphore,
                )
                verdict = result.get("verdict") if isinstance(result, dict) else None

                # Merge partial rerun: combine old good agent results with new run,
                # deduplicating by agent_id (new result takes precedence for any
                # agent that was re-run, e.g. the build agent).
                if _partial_good_agents is not None and isinstance(result, dict):
                    new_agents = list(result.get("agents", []))
                    new_ids = {a.get("agent_id") for a in new_agents if isinstance(a, dict)}
                    deduped_good = [a for a in _partial_good_agents if a.get("agent_id") not in new_ids]
                    merged_report = build_agentic_report(
                        new_agents + deduped_good,
                        task_id=task_id,
                    )
                    for key in ("build_result", "artifacts", "runtime_paths", "app_link", "preview_url", "pipeline_handoffs"):
                        if key in result:
                            merged_report[key] = result[key]
                    _attach_report_path_evidence(merged_report)
                    merged_report["verdict"] = derive_open_report_verdict(merged_report)
                    logger.info(
                        "Merged partial rerun for '%s': %d new agents + %d unique good agents → overall status=%s",
                        sample_id,
                        len(new_agents),
                        len(deduped_good),
                        merged_report["verdict"].get("status", "?"),
                    )
                    result = merged_report
                    verdict = result.get("verdict")

                status = None
                if isinstance(verdict, dict):
                    status = normalize_open_status(verdict.get("status"))
                    if status is None and isinstance(verdict.get("passed"), bool):
                        status = "passed" if verdict.get("passed") else "failed"
                if status is None and isinstance(result, dict):
                    status = normalize_open_status(result.get("status") or result.get("passed"))
                if status is None:
                    status = "failed"
                reason = ""
                if isinstance(verdict, dict):
                    reason = str(verdict.get("reason") or "")
                elif isinstance(result, dict):
                    reason = str(result.get("reason") or "")
                await _emit_task_progress(
                    task_progress_callback,
                    task_id,
                    {
                        "kind": "task_final",
                        "status": "completed",
                        "phase": "finished",
                        "verdict_status": status,
                        "reason": reason,
                        "report": result,
                    },
                )
                return {"id": sample_id, "status": status, "query": query, "model": row.get("model", ""), "result": result, "link": _get_render_link(row)}
            except Exception as exc:
                logger.error("Row '%s' failed: %s", sample_id, exc, exc_info=True)
                await _emit_task_progress(
                    task_progress_callback,
                    task_id,
                    {
                        "kind": "task_final",
                        "status": "failed",
                        "phase": "finished",
                        "verdict_status": "failed",
                        "reason": str(exc),
                        "error": str(exc),
                    },
                )
                return {"id": sample_id, "status": "failed", "error": str(exc), "model": row.get("model", ""), "link": _get_render_link(row)}

    async def run_indexed(index: int, row: Dict[str, Any]) -> tuple[int, Dict[str, Any]]:
        return index, await run_row(index, row)

    # ------------------------------------------------------------------
    # Batched processing with automatic drain on three triggers:
    #
    #   1. Periodic — every N completed rows (env DRAIN_ROW_WINDOW).
    #   2. Consecutive LLM errors — M+ in a row (env DRAIN_LLM_ERRORS).
    #   3. Process memory — RSS exceeds P% of RAM (env DRAIN_MEMORY_PCT).
    #
    # After each drain we wait for all in-flight rows, reset the shared
    # httpx client (clears stale connections), and force GC — preventing
    # slow resource accumulation without requiring a full restart.
    # ------------------------------------------------------------------
    import os as _os
    _DRAIN_WINDOW = int(_os.getenv("DRAIN_ROW_WINDOW", "5"))
    _MAX_CONSECUTIVE_LLM_ERRORS = int(_os.getenv("DRAIN_LLM_ERRORS", "5"))
    _MAX_MEMORY_PERCENT = float(_os.getenv("DRAIN_MEMORY_PCT", "50.0"))
    _drain_count = 0             # non-skipped completions since last drain
    _drain_counters: dict[str, int] = {"periodic": 0, "llm_errors": 0, "memory": 0}
    _pending: dict[int, asyncio.Task] = {}
    _row_iter = iter(enumerate(rows))

    async def _launch_one() -> bool:
        """Launch the next non-skipped row; return False when exhausted."""
        for idx, row in _row_iter:
            if idx in skipped_indices:
                # Skipped rows complete instantly (no semaphore).  Process
                # them synchronously here so the caller never sees them.
                result = await run_indexed(idx, row)
                if on_result is not None and not suppress_skipped_on_result:
                    aw = maybe_await(on_result(idx, result))
                    if aw is not None:
                        await aw
                continue
            t = asyncio.create_task(run_indexed(idx, row))
            _pending[idx] = t
            return True
        return False

    async def _process_one(t: asyncio.Task) -> None:
        nonlocal _drain_count
        idx, row_result = t.result()
        if on_result is not None:
            aw = maybe_await(on_result(idx, row_result))
            if aw is not None:
                await aw
        _drain_count += 1

    async def _run_drain(reason: str = "periodic") -> None:
        """Force GC and reset error counters.

        We do NOT reset the httpx connection pool here — doing so would
        close the shared client and break *all* existing ChatOpenAI
        instances (they hold the old client reference and would fail on
        every subsequent LLM call).  Each ChatOpenAI manages its own
        connection pool; httpx handles stale connections transparently
        via keepalive expiry and retry.
        """
        nonlocal _drain_count
        _drain_counters[reason] = _drain_counters.get(reason, 0) + 1
        logger.info(
            "Drain [%s] at %d completed rows (%d pending run in background) …",
            reason, _drain_count, len(_pending),
        )

        import gc
        n = gc.collect()
        logger.info("Drain [%s]: done (gc freed %d objects)", reason, n)

        from frontend_evaluator.llm.retry import reset_llm_connection_error_count as _reset_err
        _reset_err()

        # Cooldown: give the upstream API time to recover from rate limits.
        if reason in ("llm_errors", "memory"):
            cooldown = 5.0
            logger.info("Drain [%s]: cooldown %.1f s …", reason, cooldown)
            await asyncio.sleep(cooldown)

    # Seed the initial batch
    for _ in range(max(1, config.row_parallelism)):
        if not await _launch_one():
            break

    while _pending:
        done, _ = await asyncio.wait(
            list(_pending.values()),
            return_when=asyncio.FIRST_COMPLETED,
        )
        for t in done:
            idx = next(i for i, task in _pending.items() if task is t)
            del _pending[idx]
            await _process_one(t)

        from frontend_evaluator.llm.retry import get_llm_connection_error_count as _get_err

        def _write_diag_stats(completed: int, llm_errs: int) -> None:
            path = os.getenv("EVAL_OPEN_DIAG_STATS_PATH", "")
            if not path:
                return
            try:
                stats = {
                    "timestamp": time.time(),
                    "pid": os.getpid(),
                    "rss_kb": 0,
                    "mem_percent": _get_process_memory_percent(),
                    "active_sessions": _tool_session.get_session_count(),
                    "pending_tasks": len(_pending),
                    "completed_count": completed,
                    "drain_counters": dict(_drain_counters),
                    "llm_errs": llm_errs,
                }
                # Read RSS from /proc/self/status for consistency
                try:
                    with open("/proc/self/status") as f:
                        for line in f:
                            if line.startswith("VmRSS:"):
                                stats["rss_kb"] = int(line.split()[1])
                                break
                except OSError:
                    pass
                tmp = path + ".tmp"
                with open(tmp, "w") as f:
                    f.write(json.dumps(stats) + "\n")
                os.replace(tmp, path)
            except OSError:
                pass

        # Write diagnostic stats every 5 rows (same cadence as health log)
        if _drain_count > 0 and _drain_count % 5 == 0:
            _write_diag_stats(_drain_count, _get_err())

        # Log memory health every 5 rows so the operator can see trends
        if _drain_count > 0 and _drain_count % 5 == 0:
            logger.info(
                "Health: mem=%.1f%% pending=%d llm_errs=%d completed=%d",
                _get_process_memory_percent(),
                len(_pending),
                _get_err(),
                _drain_count,
            )

        # --- Drain trigger 1: periodic row-count threshold -------------
        if _drain_count > 0 and _drain_count % _DRAIN_WINDOW == 0:
            await _run_drain("periodic")

        # --- Drain trigger 2: global LLM connection error count --------
        _llm_err_count = _get_err()
        if _llm_err_count >= _MAX_CONSECUTIVE_LLM_ERRORS:
            logger.warning(
                "Drain triggered: %d LLM connection errors since last drain",
                _llm_err_count,
            )
            await _run_drain("llm_errors")

        # --- Drain trigger 3: process memory exceeds threshold ---------
        mem_pct = _get_process_memory_percent()
        if mem_pct >= _MAX_MEMORY_PERCENT:
            logger.warning(
                "Drain triggered: process memory at %.1f%% (≥ %.0f%%)",
                mem_pct, _MAX_MEMORY_PERCENT,
            )
            await _run_drain("memory")

        # Refill the pending pool — target exactly row_parallelism in-flight
        if await _launch_one():
            need = max(0, config.row_parallelism - len(_pending))
            for _ in range(need):
                if not await _launch_one():
                    break

    # Results are persisted incrementally by the caller via on_result;
    # no in-memory accumulation needed.
    return []
