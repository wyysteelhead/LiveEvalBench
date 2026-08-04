from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set

from .cli_support.common import write_json_output


logger = logging.getLogger(__name__)
_UNSET = object()


def build_monitor_task_descriptors(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    descriptors: List[Dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        sample_id = str(row.get("id", "") or "").strip()
        query = str(row.get("query", "") or "").strip()
        task_id = f"task-{index:04d}"
        descriptors.append(
            {
                "task_id": task_id,
                "index": index - 1,
                "position": index,
                "sample_id": sample_id,
                "query": query,
            }
        )
    return descriptors


def build_monitor_attach_command(state_path: Path) -> str:
    return f"python scripts/eval_open_monitor.py --state {state_path}"


def _task_sort_key(task: Dict[str, Any]) -> tuple[int, int, int]:
    """Sort key: active agents first, waiting agents next, finished last.

    Within the "running" status, agents that are actually executing
    subtasks (steps_completed > 0 and no waiting queue_reason) sort
    ahead of agents that are queued for evaluate/subtask slots.
    """
    status = str(task.get("status", "pending") or "pending").strip().lower()
    if status in {"completed", "failed"}:
        return (2, 0, int(task.get("position", 0)))
    if status != "running":
        return (1, 0, int(task.get("position", 0)))

    agents = task.get("agents", []) if isinstance(task.get("agents"), list) else []
    active = 0
    waiting = 0
    for a in agents:
        if not isinstance(a, dict):
            continue
        q = str(a.get("queue_reason") or "").strip()
        steps = int(a.get("steps_completed", 0) or 0)
        if q.startswith("waiting"):
            waiting += 1
        elif steps > 0 and a.get("status") == "running":
            active += 1

    if active > 0:
        sub = 0  # has actively executing agents
    elif waiting > 0:
        sub = 1  # only waiting agents
    else:
        sub = 2  # no agents registered yet
    return (0, sub, int(task.get("position", 0)))


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _agent_progress_summary(agent_payloads: Iterable[Dict[str, Any]]) -> Dict[str, int]:
    summary = {"total": 0, "pending": 0, "running": 0, "finished": 0}
    for agent in agent_payloads:
        if not isinstance(agent, dict):
            continue
        summary["total"] += 1
        status = str(agent.get("status", "pending") or "pending").strip().lower()
        if status == "running":
            summary["running"] += 1
        elif status in {"completed", "failed", "blocked", "max_steps_exceeded", "timeout", "error", "no_verdict_submitted"}:
            summary["finished"] += 1
        else:
            summary["pending"] += 1
    return summary


def _normalize_tree_status(value: Any) -> str:
    normalized = str(value or "pending").strip().lower()
    if normalized in {"completed", "success"}:
        return "passed"
    if normalized in {"passed", "failed", "partial", "running", "blocked", "pending"}:
        return normalized
    if normalized in {"error", "timeout", "max_steps_exceeded"}:
        return "failed"
    return "pending"


def _empty_task_tree_state(mode: str = "flat") -> Dict[str, Any]:
    return {
        "mode": mode,
        "main_tasks": [],
        "current_main_task_id": None,
        "current_task_id": None,
        "current_subtask_title": None,
        "running_subtasks": [],
        "summary": {
            "main_tasks_total": 0,
            "main_tasks_pending": 0,
            "main_tasks_running": 0,
            "main_tasks_finished": 0,
            "subtasks_total": 0,
            "subtasks_pending": 0,
            "subtasks_running": 0,
            "subtasks_finished": 0,
        },
    }


def _refresh_main_task_status(main_task: Dict[str, Any]) -> None:
    subtasks = main_task.get("subtasks") if isinstance(main_task.get("subtasks"), list) else []
    if not subtasks:
        main_task["status"] = _normalize_tree_status(main_task.get("status"))
        return

    statuses = [_normalize_tree_status(subtask.get("status")) for subtask in subtasks]
    if any(status == "running" for status in statuses):
        main_task["status"] = "running"
    elif all(status == "passed" for status in statuses):
        main_task["status"] = "passed"
    elif all(status in {"failed", "blocked"} for status in statuses):
        main_task["status"] = "failed"
    elif any(status in {"passed", "failed", "partial", "blocked"} for status in statuses):
        main_task["status"] = "partial"
    else:
        main_task["status"] = "pending"


def _refresh_tree_summary(tree_state: Dict[str, Any]) -> None:
    main_pending = 0
    main_running = 0
    main_finished = 0
    sub_pending = 0
    sub_running = 0
    sub_finished = 0

    for main_task in tree_state.get("main_tasks", []):
        status = _normalize_tree_status(main_task.get("status"))
        if status == "running":
            main_running += 1
        elif status in {"passed", "failed", "partial", "blocked"}:
            main_finished += 1
        else:
            main_pending += 1

        for subtask in main_task.get("subtasks", []):
            sub_status = _normalize_tree_status(subtask.get("status"))
            if sub_status == "running":
                sub_running += 1
            elif sub_status in {"passed", "failed", "partial", "blocked"}:
                sub_finished += 1
            else:
                sub_pending += 1

    tree_state["summary"] = {
        "main_tasks_total": len(tree_state.get("main_tasks", [])),
        "main_tasks_pending": main_pending,
        "main_tasks_running": main_running,
        "main_tasks_finished": main_finished,
        "subtasks_total": sub_pending + sub_running + sub_finished,
        "subtasks_pending": sub_pending,
        "subtasks_running": sub_running,
        "subtasks_finished": sub_finished,
    }


def _build_tree_state_from_plan(plan: Any, *, mode: str = "tree") -> Dict[str, Any]:
    tree_state = _empty_task_tree_state(mode)
    if not isinstance(plan, dict):
        return tree_state

    main_tasks = plan.get("main_tasks") if isinstance(plan.get("main_tasks"), list) else []
    normalized_main_tasks: List[Dict[str, Any]] = []
    for main_task in main_tasks:
        if not isinstance(main_task, dict):
            continue
        subtasks = main_task.get("subtasks") if isinstance(main_task.get("subtasks"), list) else []
        normalized_subtasks: List[Dict[str, Any]] = []
        for subtask in subtasks:
            if not isinstance(subtask, dict):
                continue
            normalized_subtasks.append(
                {
                    "task_id": str(subtask.get("sub_task_id") or subtask.get("task_id") or ""),
                    "title": str(subtask.get("title") or ""),
                    "goal": str(subtask.get("goal") or subtask.get("task_text") or ""),
                    "kind": str(subtask.get("kind") or "independent"),
                    "stage_index": int(subtask.get("stage_index", 0) or 0),
                    "status": "pending",
                    "completion_score": None,
                    "reason": "",
                    "steps": 0,
                    "steps_total": 0,
                    "is_current": False,
                }
            )
        normalized_main_task = {
            "main_task_id": str(main_task.get("main_task_id") or ""),
            "title": str(main_task.get("title") or ""),
            "goal": str(main_task.get("goal") or ""),
            "origin": str(main_task.get("origin") or "query_specific"),
            "dimension_ids": [str(item) for item in (main_task.get("dimension_ids") or []) if item is not None],
            "status": "pending",
            "subtasks": normalized_subtasks,
        }
        _refresh_main_task_status(normalized_main_task)
        normalized_main_tasks.append(normalized_main_task)

    tree_state["main_tasks"] = normalized_main_tasks
    _refresh_tree_summary(tree_state)
    return tree_state


def _merge_task_results_into_tree(tree_state: Dict[str, Any], tasks: Any) -> Dict[str, Any]:
    if not isinstance(tasks, list):
        return tree_state

    task_index = {
        str(task.get("task_id") or ""): task
        for task in tasks
        if isinstance(task, dict) and str(task.get("task_id") or "")
    }
    for main_task in tree_state.get("main_tasks", []):
        for subtask in main_task.get("subtasks", []):
            task_payload = task_index.get(str(subtask.get("task_id") or ""))
            if not task_payload:
                continue
            subtask["status"] = _normalize_tree_status(task_payload.get("status") or task_payload.get("verdict"))
            subtask["completion_score"] = task_payload.get("completion_score")
            subtask["reason"] = str(task_payload.get("reason") or "")
            subtask["steps"] = int(task_payload.get("steps", 0) or 0)
            subtask["is_current"] = False
        _refresh_main_task_status(main_task)

    _refresh_tree_summary(tree_state)
    return tree_state


def _build_tree_state_from_runtime(task_tree: Any, *, mode: str = "tree") -> Dict[str, Any]:
    tree_state = _empty_task_tree_state(mode)
    if not isinstance(task_tree, list):
        return tree_state

    normalized_main_tasks: List[Dict[str, Any]] = []
    for main_task in task_tree:
        if not isinstance(main_task, dict):
            continue
        subtasks = main_task.get("subtasks") if isinstance(main_task.get("subtasks"), list) else []
        normalized_subtasks: List[Dict[str, Any]] = []
        for subtask in subtasks:
            if not isinstance(subtask, dict):
                continue
            normalized_subtasks.append(
                {
                    "task_id": str(subtask.get("task_id") or subtask.get("sub_task_id") or ""),
                    "title": str(subtask.get("title") or ""),
                    "goal": str(subtask.get("goal") or subtask.get("task_text") or ""),
                    "kind": str(subtask.get("task_kind") or subtask.get("kind") or "independent"),
                    "stage_index": int(subtask.get("stage_index", 0) or 0),
                    "status": _normalize_tree_status(subtask.get("status") or subtask.get("verdict")),
                    "completion_score": subtask.get("completion_score"),
                    "reason": str(subtask.get("reason") or ""),
                    "steps": int(subtask.get("steps", 0) or 0),
                    "steps_total": int(subtask.get("steps_total", 0) or 0),
                    "is_current": False,
                }
            )
        normalized_main_task = {
            "main_task_id": str(main_task.get("main_task_id") or ""),
            "title": str(main_task.get("title") or ""),
            "goal": str(main_task.get("goal") or ""),
            "origin": str(main_task.get("origin") or "query_specific"),
            "dimension_ids": [str(item) for item in (main_task.get("dimension_ids") or []) if item is not None],
            "status": _normalize_tree_status(main_task.get("status") or main_task.get("verdict")),
            "subtasks": normalized_subtasks,
        }
        _refresh_main_task_status(normalized_main_task)
        normalized_main_tasks.append(normalized_main_task)

    tree_state["main_tasks"] = normalized_main_tasks
    _refresh_tree_summary(tree_state)
    return tree_state


def _build_tree_state_from_agent_payload(agent_payload: Dict[str, Any]) -> Dict[str, Any]:
    mode = str(agent_payload.get("task_synthesis_mode") or "flat")
    task_tree = agent_payload.get("task_tree")
    if isinstance(task_tree, list) and task_tree:
        return _build_tree_state_from_runtime(task_tree, mode=mode)

    planned_task_tree = agent_payload.get("planned_task_tree")
    tasks = agent_payload.get("tasks")
    if isinstance(planned_task_tree, dict):
        return _merge_task_results_into_tree(_build_tree_state_from_plan(planned_task_tree, mode=mode), tasks)

    return _empty_task_tree_state(mode)


def _normalize_running_subtasks(items: Any) -> List[Dict[str, Any]]:
    if not isinstance(items, list):
        return []

    normalized: List[Dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        task_id = str(item.get("task_id") or "")
        title = str(item.get("title") or "")
        if not task_id and not title:
            continue
        normalized.append(
            {
                "task_id": task_id,
                "title": title,
                "main_task_id": str(item.get("main_task_id") or "") or None,
                "main_task_title": str(item.get("main_task_title") or "") or None,
                "status": str(item.get("status") or "running") or "running",
                "steps_completed": max(0, int(item.get("steps_completed", 0) or 0)),
                "steps_total": max(0, int(item.get("steps_total", 0) or 0)),
            }
        )
    return normalized

def _set_current_tree_pointer(
    agent: Dict[str, Any],
    *,
    current_main_task_id: Optional[str],
    current_task_id: Optional[str],
    current_subtask_title: Optional[str],
    running_subtasks: Optional[List[Dict[str, Any]]],
    current_task_status: Optional[str],
    current_task_completion_score: Optional[float],
    current_task_steps_completed: Optional[int],
    current_task_steps_total: Optional[int],
) -> None:
    tree_state = agent.get("task_tree") if isinstance(agent.get("task_tree"), dict) else None
    if not tree_state:
        return

    tree_state["current_main_task_id"] = current_main_task_id
    tree_state["current_task_id"] = current_task_id
    tree_state["current_subtask_title"] = current_subtask_title
    tree_state["running_subtasks"] = _normalize_running_subtasks(running_subtasks)

    for main_task in tree_state.get("main_tasks", []):
        for subtask in main_task.get("subtasks", []):
            subtask["is_current"] = False

    if tree_state["running_subtasks"]:
        for main_task in tree_state.get("main_tasks", []):
            for subtask in main_task.get("subtasks", []):
                for running in tree_state["running_subtasks"]:
                    title_matches = running.get("title") and str(subtask.get("title") or "") == str(running.get("title"))
                    id_matches = running.get("task_id") and str(subtask.get("task_id") or "") == str(running.get("task_id"))
                    if not title_matches and not id_matches:
                        continue
                    subtask["status"] = _normalize_tree_status(running.get("status") or "running")
                    subtask["is_current"] = True
                    subtask["steps"] = max(0, int(running.get("steps_completed", 0) or 0))
                    subtask["steps_total"] = max(0, int(running.get("steps_total", 0) or 0))
                _refresh_main_task_status(main_task)
        _refresh_tree_summary(tree_state)
        return

    if not current_main_task_id and not current_task_id and not current_subtask_title:
        _refresh_tree_summary(tree_state)
        return

    normalized_status = _normalize_tree_status(current_task_status or "running")
    for main_task in tree_state.get("main_tasks", []):
        if current_main_task_id and str(main_task.get("main_task_id") or "") != str(current_main_task_id):
            continue
        for subtask in main_task.get("subtasks", []):
            title_matches = current_subtask_title and str(subtask.get("title") or "") == str(current_subtask_title)
            id_matches = current_task_id and str(subtask.get("task_id") or "") == str(current_task_id)
            if not title_matches and not id_matches:
                continue
            subtask["status"] = normalized_status
            subtask["is_current"] = normalized_status == "running"
            if current_task_completion_score is not None:
                subtask["completion_score"] = float(current_task_completion_score)
            if current_task_steps_completed is not None:
                subtask["steps"] = max(0, int(current_task_steps_completed))
            if current_task_steps_total is not None:
                subtask["steps_total"] = max(0, int(current_task_steps_total))
            _refresh_main_task_status(main_task)
            _refresh_tree_summary(tree_state)
            return

    _refresh_tree_summary(tree_state)


class OpenEvalMonitorWriter:
    def __init__(
        self,
        state_path: Path,
        *,
        max_parallel: int,
        source_jsonl: Optional[str] = None,
        agents_dir: Optional[str] = None,
    ) -> None:
        self.state_path = state_path
        self._lock = asyncio.Lock()
        self._state: Dict[str, Any] = {
            "version": 2,
            "mode": "eval_open_monitor",
            "updated_at": _now_iso(),
            "meta": {
                "max_parallel": max(1, int(max_parallel)),
                "source_jsonl": source_jsonl,
                "agents_dir": agents_dir,
            },
            "summary": {
                "tasks_total": 0,
                "tasks_pending": 0,
                "tasks_running": 0,
                "tasks_finished": 0,
            },
            "tasks": [],
        }

    async def initialize(
        self, descriptors: List[Dict[str, Any]], *, resume_from: Optional[Path] = None,
        reset_sample_ids: Optional[Set[str]] = None,
    ) -> None:
        async with self._lock:
            tasks = []
            for descriptor in descriptors:
                tasks.append(
                    {
                        "task_id": descriptor["task_id"],
                        "index": int(descriptor["index"]),
                        "position": int(descriptor["position"]),
                        "sample_id": descriptor.get("sample_id") or "",
                        "query": descriptor.get("query") or "",
                        "status": "pending",
                        "phase": "pending",
                        "started_at": None,
                        "completed_at": None,
                        "verdict_status": None,
                        "reason": "",
                        "error": None,
                        "row_wait_ms": 0,
                        "agents": [],
                        "agent_summary": {"total": 0, "pending": 0, "running": 0, "finished": 0},
                    }
                )
            if resume_from and resume_from.exists():
                prev_status: Dict[str, Dict[str, Any]] = {}
                try:
                    prev = json.loads(resume_from.read_text(encoding="utf-8"))
                    for t in prev.get("tasks") if isinstance(prev, dict) else []:
                        if isinstance(t, dict) and t.get("task_id"):
                            prev_status[str(t["task_id"])] = t
                except Exception:
                    pass
                if prev_status:
                    reset_set = {str(s).strip() for s in (reset_sample_ids or []) if str(s or "").strip()}
                    reset_count = 0
                    for task in tasks:
                        prev_task = prev_status.get(task["task_id"])
                        if prev_task is None:
                            continue
                        # Force-rerun: skip carry-forward for these sample_ids so
                        # the new run starts them from a clean pending state.
                        if reset_set and str(task.get("sample_id") or "") in reset_set:
                            reset_count += 1
                            continue
                        old_status = str(prev_task.get("status", "pending") or "pending").strip().lower()
                        # On restart, "running" is stale — the previous process
                        # was killed and this task is not actually executing.
                        if old_status == "running":
                            old_status = "pending"
                        task["status"] = old_status
                        task["phase"] = prev_task.get("phase", "pending")
                        task["verdict_status"] = prev_task.get("verdict_status")
                        task["reason"] = prev_task.get("reason", "")
                        task["error"] = prev_task.get("error")
                        task["row_wait_ms"] = prev_task.get("row_wait_ms", 0)
                        task["started_at"] = prev_task.get("started_at")
                        task["completed_at"] = prev_task.get("completed_at")
                        task["agents"] = prev_task.get("agents", []) if isinstance(prev_task.get("agents"), list) else []
                        task["agent_summary"] = prev_task.get(
                            "agent_summary", {"total": 0, "pending": 0, "running": 0, "finished": 0}
                        )
                    if reset_set:
                        logger.info(
                            "monitor initialize: reset %d sample_ids to pending (force-rerun set size=%d)",
                            reset_count, len(reset_set),
                        )
            self._state["tasks"] = tasks
            self._refresh_summary_locked()
            await self._flush_locked()

    async def mark_task_status(
        self,
        task_id: str,
        *,
        status: str,
        phase: Optional[str] = None,
        reason: Optional[str] = None,
        error: Optional[str] = None,
        verdict_status: Optional[str] = None,
        row_wait_ms: Optional[int] = None,
    ) -> None:
        async with self._lock:
            task = self._task_locked(task_id)
            if phase is not None:
                task["phase"] = phase
            task["status"] = status
            if reason is not None:
                task["reason"] = reason
            if error is not None:
                task["error"] = error
            if verdict_status is not None:
                task["verdict_status"] = verdict_status
            if row_wait_ms is not None:
                task["row_wait_ms"] = max(0, int(row_wait_ms))
            if status == "running" and not task.get("started_at"):
                task["started_at"] = _now_iso()
            if status in {"completed", "failed"}:
                task["completed_at"] = _now_iso()
            self._refresh_task_summary_locked(task)
            self._refresh_summary_locked()
            await self._flush_locked()

    async def upsert_agent(
        self,
        task_id: str,
        agent_id: str,
        *,
        status: Any = _UNSET,
        steps_completed: Any = _UNSET,
        steps_total: Any = _UNSET,
        current_task_title: Any = _UNSET,
        current_task_id: Any = _UNSET,
        current_task_status: Any = _UNSET,
        current_task_completion_score: Any = _UNSET,
        current_task_steps_completed: Any = _UNSET,
        current_task_steps_total: Any = _UNSET,
        current_task_index: Any = _UNSET,
        current_task_total: Any = _UNSET,
        current_main_task_id: Any = _UNSET,
        current_main_task_title: Any = _UNSET,
        current_subtask_title: Any = _UNSET,
        running_subtasks: Any = _UNSET,
        task_synthesis_mode: Any = _UNSET,
        planned_task_tree: Any = _UNSET,
        build_wait_ms: Any = _UNSET,
        npm_wait_ms: Any = _UNSET,
        npm_duration_ms: Any = _UNSET,
        evaluate_agent_wait_ms: Any = _UNSET,
        cdp_connect_ms: Any = _UNSET,
        subtask_wait_ms: Any = _UNSET,
        end_reason: Any = _UNSET,
        queue_reason: Any = _UNSET,
    ) -> None:
        async with self._lock:
            task = self._task_locked(task_id)
            agent = self._agent_locked(task, agent_id)
            if status is not _UNSET:
                agent["status"] = status
                if status == "running" and not agent.get("started_at"):
                    agent["started_at"] = _now_iso()
                if status in {"completed", "failed", "blocked", "timeout", "error", "max_steps_exceeded", "no_verdict_submitted"}:
                    agent["completed_at"] = _now_iso()
            if steps_total is not _UNSET:
                agent["steps_total"] = max(0, int(steps_total)) if steps_total is not None else 0
            if steps_completed is not _UNSET:
                normalized_completed = max(0, int(steps_completed)) if steps_completed is not None else 0
                total = int(agent.get("steps_total") or 0)
                agent["steps_completed"] = min(normalized_completed, total) if total > 0 else normalized_completed
            if task_synthesis_mode is not _UNSET:
                agent["task_synthesis_mode"] = str(task_synthesis_mode or "flat")
            if planned_task_tree is not _UNSET and planned_task_tree is not None:
                agent["task_tree"] = _build_tree_state_from_plan(
                    planned_task_tree,
                    mode=str(task_synthesis_mode or agent.get("task_synthesis_mode") or "tree"),
                )
            if current_task_title is not _UNSET:
                agent["current_task_title"] = current_task_title
            if current_task_id is not _UNSET:
                agent["current_task_id"] = current_task_id
            if current_task_status is not _UNSET:
                agent["current_task_status"] = current_task_status
            if current_task_completion_score is not _UNSET:
                agent["current_task_completion_score"] = (
                    float(current_task_completion_score)
                    if current_task_completion_score is not None
                    else None
                )
            if current_task_steps_completed is not _UNSET:
                agent["current_task_steps_completed"] = (
                    max(0, int(current_task_steps_completed))
                    if current_task_steps_completed is not None
                    else 0
                )
            if current_task_steps_total is not _UNSET:
                agent["current_task_steps_total"] = (
                    max(0, int(current_task_steps_total))
                    if current_task_steps_total is not None
                    else 0
                )
            if current_task_index is not _UNSET:
                agent["current_task_index"] = int(current_task_index) if current_task_index is not None else 0
            if current_task_total is not _UNSET:
                agent["current_task_total"] = int(current_task_total) if current_task_total is not None else 0
            if current_main_task_id is not _UNSET:
                agent["current_main_task_id"] = current_main_task_id
            if current_main_task_title is not _UNSET:
                agent["current_main_task_title"] = current_main_task_title
            if current_subtask_title is not _UNSET:
                agent["current_subtask_title"] = current_subtask_title
            if running_subtasks is not _UNSET:
                agent["running_subtasks"] = _normalize_running_subtasks(running_subtasks or [])
            if build_wait_ms is not _UNSET:
                agent["build_wait_ms"] = max(0, int(build_wait_ms)) if build_wait_ms is not None else 0
            if npm_wait_ms is not _UNSET:
                agent["npm_wait_ms"] = max(0, int(npm_wait_ms)) if npm_wait_ms is not None else 0
            if npm_duration_ms is not _UNSET:
                agent["npm_duration_ms"] = max(0, int(npm_duration_ms)) if npm_duration_ms is not None else 0
            if evaluate_agent_wait_ms is not _UNSET:
                agent["evaluate_agent_wait_ms"] = max(0, int(evaluate_agent_wait_ms)) if evaluate_agent_wait_ms is not None else 0
            if cdp_connect_ms is not _UNSET:
                agent["cdp_connect_ms"] = max(0, int(cdp_connect_ms)) if cdp_connect_ms is not None else 0
            if subtask_wait_ms is not _UNSET:
                agent["subtask_wait_ms"] = max(0, int(subtask_wait_ms)) if subtask_wait_ms is not None else 0
            if end_reason is not _UNSET:
                agent["end_reason"] = end_reason
            if queue_reason is not _UNSET:
                agent["queue_reason"] = str(queue_reason or "") or None

            _set_current_tree_pointer(
                agent,
                current_main_task_id=agent.get("current_main_task_id"),
                current_task_id=agent.get("current_task_id"),
                current_subtask_title=agent.get("current_subtask_title"),
                running_subtasks=agent.get("running_subtasks"),
                current_task_status=agent.get("current_task_status"),
                current_task_completion_score=(
                    float(agent.get("current_task_completion_score"))
                    if isinstance(agent.get("current_task_completion_score"), (int, float))
                    else None
                ),
                current_task_steps_completed=(
                    int(agent.get("current_task_steps_completed"))
                    if isinstance(agent.get("current_task_steps_completed"), int)
                    else None
                ),
                current_task_steps_total=(
                    int(agent.get("current_task_steps_total"))
                    if isinstance(agent.get("current_task_steps_total"), int)
                    else None
                ),
            )
            self._refresh_task_summary_locked(task)
            self._refresh_summary_locked()
            await self._flush_locked()

    async def apply_final_result(
        self,
        task_id: str,
        *,
        lifecycle_status: str,
        verdict_status: Optional[str],
        reason: Optional[str] = None,
        error: Optional[str] = None,
        report: Optional[Dict[str, Any]] = None,
    ) -> None:
        async with self._lock:
            task = self._task_locked(task_id)
            task["status"] = lifecycle_status
            task["phase"] = "finished" if lifecycle_status != "running" else task.get("phase")
            task["verdict_status"] = verdict_status
            task["reason"] = reason or task.get("reason") or ""
            task["error"] = error
            task["completed_at"] = _now_iso()
            if isinstance(report, dict):
                agents = report.get("agents")
                if isinstance(agents, list):
                    for agent_payload in agents:
                        if not isinstance(agent_payload, dict):
                            continue
                        agent_id = str(agent_payload.get("agent_id", "") or "").strip()
                        if not agent_id:
                            continue
                        agent = self._agent_locked(task, agent_id)
                        agent["status"] = str(agent_payload.get("status", "completed") or "completed")
                        runtime = agent_payload.get("runtime") if isinstance(agent_payload.get("runtime"), dict) else {}
                        steps_used = int(runtime.get("steps_used", 0) or 0)
                        total = int(agent.get("steps_total") or 0)
                        agent["steps_completed"] = min(steps_used, total) if total > 0 else steps_used
                        agent["build_wait_ms"] = max(0, int(runtime.get("build_wait_ms", agent.get("build_wait_ms", 0)) or 0))
                        agent["npm_wait_ms"] = max(0, int(runtime.get("npm_wait_ms", agent.get("npm_wait_ms", 0)) or 0))
                        agent["npm_duration_ms"] = max(0, int(runtime.get("npm_duration_ms", agent.get("npm_duration_ms", 0)) or 0))
                        agent["evaluate_agent_wait_ms"] = max(0, int(runtime.get("evaluate_agent_wait_ms", agent.get("evaluate_agent_wait_ms", 0)) or 0))
                        agent["cdp_connect_ms"] = max(0, int(runtime.get("cdp_connect_ms", agent.get("cdp_connect_ms", 0)) or 0))
                        agent["subtask_wait_ms"] = max(0, int(runtime.get("subtask_wait_ms", agent.get("subtask_wait_ms", 0)) or 0))
                        agent["end_reason"] = str(agent_payload.get("end_reason", "") or "")
                        agent["completed_at"] = _now_iso()
                        agent["task_synthesis_mode"] = str(agent_payload.get("task_synthesis_mode") or agent.get("task_synthesis_mode") or "flat")
                        agent["task_tree"] = _build_tree_state_from_agent_payload(agent_payload)
                        agent["current_task_id"] = None
                        agent["current_task_title"] = None
                        agent["current_task_status"] = None
                        agent["current_task_completion_score"] = None
                        agent["current_task_steps_completed"] = 0
                        agent["current_task_steps_total"] = 0
                        agent["current_main_task_id"] = None
                        agent["current_main_task_title"] = None
                        agent["current_subtask_title"] = None
                        agent["running_subtasks"] = []
                        agent["queue_reason"] = None
            self._refresh_task_summary_locked(task)
            self._refresh_summary_locked()
            await self._flush_locked()

    def _task_locked(self, task_id: str) -> Dict[str, Any]:
        for task in self._state["tasks"]:
            if task.get("task_id") == task_id:
                return task
        raise KeyError(f"Unknown monitor task_id: {task_id}")

    @staticmethod
    def _agent_locked(task: Dict[str, Any], agent_id: str) -> Dict[str, Any]:
        agents = task.setdefault("agents", [])
        if not isinstance(agents, list):
            task["agents"] = agents = []
        for agent in agents:
            if agent.get("agent_id") == agent_id:
                return agent
        created = {
            "agent_id": agent_id,
            "status": "pending",
            "steps_completed": 0,
            "steps_total": 0,
            "current_task_title": None,
            "current_task_id": None,
            "current_task_status": None,
            "current_task_completion_score": None,
            "current_task_steps_completed": 0,
            "current_task_steps_total": 0,
            "current_task_index": 0,
            "current_task_total": 0,
            "current_main_task_id": None,
            "current_main_task_title": None,
            "current_subtask_title": None,
            "running_subtasks": [],
            "task_synthesis_mode": "flat",
            "task_tree": _empty_task_tree_state(),
            "build_wait_ms": 0,
            "npm_wait_ms": 0,
            "npm_duration_ms": 0,
            "evaluate_agent_wait_ms": 0,
            "cdp_connect_ms": 0,
            "subtask_wait_ms": 0,
            "started_at": None,
            "completed_at": None,
            "end_reason": "",
            "queue_reason": None,
        }
        agents.append(created)
        return created

    def _refresh_task_summary_locked(self, task: Dict[str, Any]) -> None:
        agents_raw = task.get("agents", [])
        task["agent_summary"] = _agent_progress_summary(agents_raw if isinstance(agents_raw, list) else [])

    def _refresh_summary_locked(self) -> None:
        pending = 0
        running = 0
        finished = 0
        for task in self._state["tasks"]:
            status = str(task.get("status", "pending") or "pending").strip().lower()
            if status == "running":
                running += 1
            elif status in {"completed", "failed"}:
                finished += 1
            else:
                pending += 1
        self._state["updated_at"] = _now_iso()
        self._state["summary"] = {
            "tasks_total": len(self._state["tasks"]),
            "tasks_pending": pending,
            "tasks_running": running,
            "tasks_finished": finished,
        }

    def _flush_locked_sync(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self._state["tasks"].sort(key=_task_sort_key)
        write_json_output(self.state_path, self._state)

    async def _flush_locked(self) -> None:
        await asyncio.to_thread(self._flush_locked_sync)
