#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import select
import sys
import termios
import tty
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from rich.console import Console, Group
    from rich.live import Live
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    from rich.tree import Tree
except ImportError as exc:  # pragma: no cover - runtime dependency guard
    print(f"rich is required for eval_open_monitor.py: {exc}", file=sys.stderr)
    print("Install dependencies with: pip install rich", file=sys.stderr)
    raise SystemExit(1) from exc


PAGE_SIZE = 10


def _ellipsize(text: Any, limit: int) -> str:
    value = str(text or "")
    if limit <= 0 or len(value) <= limit:
        return value
    if limit == 1:
        return "…"
    return value[: limit - 1] + "…"


def _load_state(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {
            "updated_at": None,
            "summary": {"tasks_total": 0, "tasks_pending": 0, "tasks_running": 0, "tasks_finished": 0},
            "tasks": [],
        }
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {
            "updated_at": None,
            "summary": {"tasks_total": 0, "tasks_pending": 0, "tasks_running": 0, "tasks_finished": 0},
            "tasks": [],
        }
    return data if isinstance(data, dict) else {"summary": {}, "tasks": []}


def _status_style(status: str) -> str:
    normalized = str(status or "").lower()
    if normalized in {"completed", "passed"}:
        return "green"
    if normalized in {"running"}:
        return "yellow"
    if normalized in {"failed", "error", "timeout", "max_steps_exceeded"}:
        return "red"
    if normalized in {"blocked"}:
        return "magenta"
    return "bright_black"


def _queue_reason_label(queue_reason: Any) -> str:
    normalized = str(queue_reason or "").strip().lower()
    if normalized == "waiting_build_slot":
        return "queued(build)"
    if normalized == "waiting_evaluate_slot":
        return "queued(cdp)"
    if normalized == "waiting_npm_preinstall":
        return "queued(npm)"
    if normalized == "waiting_subtask_slot":
        return "queued(subtask)"
    return ""


def _queue_reason_detail(queue_reason: Any) -> str:
    normalized = str(queue_reason or "").strip().lower()
    if normalized == "waiting_build_slot":
        return "waiting build parallelism slot"
    if normalized == "waiting_evaluate_slot":
        return "waiting evaluate/CDP parallelism slot"
    if normalized == "waiting_npm_preinstall":
        return "waiting npm preinstall gate"
    if normalized == "waiting_subtask_slot":
        return "waiting subtask parallelism slot"
    return ""


def _format_task_phase(task: Dict[str, Any]) -> str:
    phase = str(task.get("phase") or "-")
    if phase == "queued":
        row_wait_ms = int(task.get("row_wait_ms", 0) or 0)
        return f"queued(row {row_wait_ms}ms)" if row_wait_ms > 0 else "queued(row)"
    return phase


def _format_agent_status(agent: Dict[str, Any]) -> str:
    queue_label = _queue_reason_label(agent.get("queue_reason"))
    return queue_label or str(agent.get("status") or "pending")


def _segment_bar(width: int, parts: List[tuple[int, str]]) -> Text:
    total = sum(max(0, count) for count, _ in parts)
    width = max(10, width)
    text = Text("[")
    if total <= 0:
        text.append("·" * width, style="bright_black")
        text.append("]")
        return text

    used = 0
    for index, (count, style) in enumerate(parts):
        span = int(round((max(0, count) / total) * width)) if total else 0
        if index == len(parts) - 1:
            span = width - used
        used += span
        if span > 0:
            fill = "#"
            if style == "yellow":
                fill = ">"
            elif style == "bright_black":
                fill = "·"
            text.append(fill * span, style=style)
    if used < width:
        text.append("·" * (width - used), style="bright_black")
    text.append("]")
    return text


def _summary_bar(summary: Dict[str, Any], width: int = 50) -> Text:
    finished = int(summary.get("tasks_finished", 0) or 0)
    running = int(summary.get("tasks_running", 0) or 0)
    pending = int(summary.get("tasks_pending", 0) or 0)
    bar = _segment_bar(width, [(finished, "green"), (running, "yellow"), (pending, "bright_black")])
    bar.append(f"  done {finished} | running {running} | pending {pending}")
    return bar


def _agent_bar(agent_summary: Dict[str, Any], width: int = 24) -> Text:
    finished = int(agent_summary.get("finished", 0) or 0)
    running = int(agent_summary.get("running", 0) or 0)
    pending = int(agent_summary.get("pending", 0) or 0)
    total = int(agent_summary.get("total", 0) or 0)
    bar = _segment_bar(width, [(finished, "green"), (running, "yellow"), (pending, "bright_black")])
    bar.append(f"  {finished}/{total} done, {running} running")
    return bar


def _step_bar(agent: Dict[str, Any], width: int = 36) -> Text:
    completed, total = _overall_step_counts(agent)
    running = 1 if str(agent.get("status", "")).lower() == "running" and completed < total else 0
    remaining = max(0, total - completed - running)
    bar = _segment_bar(width, [(completed, "green"), (running, "yellow"), (remaining, "bright_black")])
    bar.append(f"  {completed}/{total}")
    return bar


def _task_step_bar(completed: int, total: int, width: int = 18) -> Text:
    total = max(0, int(total or 0))
    completed = max(0, int(completed or 0))
    running = 1 if total > 0 and completed < total else 0
    remaining = max(0, total - completed - running)
    bar = _segment_bar(width, [(completed, "green"), (running, "yellow"), (remaining, "bright_black")])
    bar.append(f"  {completed}/{total}")
    return bar


def _running_subtasks_summary(agent: Dict[str, Any], limit: int = 2) -> str:
    items = agent.get("running_subtasks") if isinstance(agent.get("running_subtasks"), list) else []
    if not items:
        return "-"

    titles = [
        str(item.get("title") or item.get("task_id") or "").strip()
        for item in items
        if isinstance(item, dict)
    ]
    titles = [title for title in titles if title]
    if not titles:
        return f"{len(items)} running"
    if len(titles) <= limit:
        return f"{len(items)} running: {', '.join(titles)}"
    return f"{len(items)} running: {', '.join(titles[:limit])} +{len(titles) - limit}"


def _overall_step_counts(agent: Dict[str, Any]) -> tuple[int, int]:
    raw_completed = max(0, int(agent.get("steps_completed", 0) or 0))
    raw_total = max(0, int(agent.get("steps_total", 0) or 0))

    task_tree = agent.get("task_tree") if isinstance(agent.get("task_tree"), dict) else {}
    main_tasks = task_tree.get("main_tasks") if isinstance(task_tree.get("main_tasks"), list) else []
    subtasks = [
        subtask
        for main_task in main_tasks
        if isinstance(main_task, dict)
        for subtask in (main_task.get("subtasks") if isinstance(main_task.get("subtasks"), list) else [])
        if isinstance(subtask, dict)
    ]
    running_subtasks = agent.get("running_subtasks") if isinstance(agent.get("running_subtasks"), list) else []
    running_subtasks = [item for item in running_subtasks if isinstance(item, dict)]

    tree_completed = sum(max(0, int(subtask.get("steps", 0) or 0)) for subtask in subtasks)
    tree_total_sum = sum(max(0, int(subtask.get("steps_total", 0) or 0)) for subtask in subtasks)
    running_completed = sum(max(0, int(item.get("steps_completed", 0) or 0)) for item in running_subtasks)

    inferred_task_total = max(
        0,
        int(agent.get("current_task_total", 0) or 0),
        len(subtasks),
        int((task_tree.get("summary") or {}).get("subtasks_total", 0) or 0)
        if isinstance(task_tree.get("summary"), dict)
        else 0,
    )
    inferred_unit_total = max(
        0,
        int(agent.get("current_task_steps_total", 0) or 0),
        max((max(0, int(subtask.get("steps_total", 0) or 0)) for subtask in subtasks), default=0),
        max((max(0, int(item.get("steps_total", 0) or 0)) for item in running_subtasks), default=0),
    )
    derived_total = max(tree_total_sum, inferred_task_total * inferred_unit_total)
    derived_completed = max(raw_completed, tree_completed, running_completed)

    if derived_total > 0:
        return derived_completed, max(raw_total, derived_total)
    return raw_completed, raw_total


def _render_agent_task_tree(agent: Dict[str, Any]) -> Optional[Panel]:
    tree_state = agent.get("task_tree") if isinstance(agent.get("task_tree"), dict) else None
    if not tree_state:
        return None

    main_tasks = tree_state.get("main_tasks") if isinstance(tree_state.get("main_tasks"), list) else []
    if not main_tasks:
        return None

    summary = tree_state.get("summary") if isinstance(tree_state.get("summary"), dict) else {}
    root = Tree(
        Text(
            f"{agent.get('agent_id') or '-'}  main {summary.get('main_tasks_finished', 0)}/{summary.get('main_tasks_total', 0)}"
            f" | sub {summary.get('subtasks_finished', 0)}/{summary.get('subtasks_total', 0)}",
            style="cyan",
        )
    )

    for main_task in main_tasks:
        main_status = str(main_task.get("status") or "pending")
        main_text = Text()
        main_text.append(f"[{main_status}] ", style=_status_style(main_status))
        main_text.append(str(main_task.get("title") or "(untitled)"), style="bold")
        dimension_ids = ", ".join(main_task.get("dimension_ids") or [])
        if dimension_ids:
            main_text.append(f"  ({dimension_ids})", style="bright_black")
        main_node = root.add(main_text)

        for subtask in main_task.get("subtasks", []):
            sub_status = str(subtask.get("status") or "pending")
            sub_text = Text()
            sub_text.append("• ", style="bright_black")
            sub_text.append(f"[{sub_status}] ", style=_status_style(sub_status))
            sub_style = "bold yellow" if subtask.get("is_current") else ""
            sub_text.append(str(subtask.get("title") or "(untitled)"), style=sub_style)
            completion = subtask.get("completion_score")
            if isinstance(completion, (int, float)):
                sub_text.append(f"  score={completion:.2f}", style="bright_black")
            steps = int(subtask.get("steps", 0) or 0)
            steps_total = int(subtask.get("steps_total", 0) or 0)
            if steps or steps_total:
                sub_text.append("  ", style="bright_black")
                sub_text.append_text(_task_step_bar(steps, steps_total if steps_total > 0 else steps))
            reason = str(subtask.get("reason") or "")
            if reason:
                sub_text.append(f"  {_ellipsize(reason, 80)}", style="bright_black")
            main_node.add(sub_text)

    return Panel(root, title="Task Tree Progress", border_style="magenta")


def _sorted_agents(task: Dict[str, Any]) -> List[Dict[str, Any]]:
    agents = task.get("agents") if isinstance(task.get("agents"), list) else []
    return sorted(
        [agent for agent in agents if isinstance(agent, dict)],
        key=lambda item: (str(item.get("status", "")).lower() != "running", item.get("agent_id") or ""),
    )


def _render_task_meta(task: Dict[str, Any]) -> Panel:
    meta = Table.grid(expand=True)
    meta.add_column(ratio=1)
    meta.add_row(Text(f"Task: {task.get('sample_id') or task.get('task_id')}", style="cyan"))
    meta.add_row(Text(f"Status: {task.get('status')} | Phase: {_format_task_phase(task)} | Verdict: {task.get('verdict_status') or '-'}"))
    meta.add_row(Text(f"Query: {_ellipsize(task.get('query') or '', 140)}", no_wrap=True, overflow="ellipsis"))
    if task.get("reason"):
        meta.add_row(Text(f"Reason: {task.get('reason')}", style="bright_black"))
    if task.get("error"):
        meta.add_row(Text(f"Error: {task.get('error')}", style="red"))
    return Panel(meta, title="Task Detail", border_style="blue")


def _running_tasks(state: Dict[str, Any]) -> List[Dict[str, Any]]:
    tasks = state.get("tasks")
    rows = [task for task in tasks if isinstance(task, dict)] if isinstance(tasks, list) else []
    running = [task for task in rows if str(task.get("status", "")).lower() == "running"]
    if running:
        return sorted(running, key=lambda item: int(item.get("position", 0) or 0))
    return sorted(rows, key=lambda item: int(item.get("position", 0) or 0))


def _find_task(state: Dict[str, Any], task_id: str) -> Optional[Dict[str, Any]]:
    for task in state.get("tasks", []):
        if isinstance(task, dict) and task.get("task_id") == task_id:
            return task
    return None


def _render_overview(state: Dict[str, Any], selected_index: int, page: int, state_path: Path) -> Group:
    summary = state.get("summary") if isinstance(state.get("summary"), dict) else {}
    tasks = _running_tasks(state)
    page_count = max(1, (len(tasks) + PAGE_SIZE - 1) // PAGE_SIZE)
    page = max(0, min(page, page_count - 1))
    start = page * PAGE_SIZE
    visible = tasks[start:start + PAGE_SIZE]

    header = Table.grid(expand=True)
    header.add_column(ratio=1)
    header.add_row(Text(f"State file: {state_path}", style="cyan"))
    header.add_row(_summary_bar(summary))
    header.add_row(Text(f"Updated: {state.get('updated_at') or 'waiting'} | Page {page + 1}/{page_count}", style="bright_black"))

    table = Table(expand=True, box=None, padding=(0, 1))
    table.add_column("Sel", width=3)
    table.add_column("Task", width=14)
    table.add_column("Status", width=10)
    table.add_column("Phase", width=18)
    table.add_column("Agents", ratio=2)
    table.add_column("Query", ratio=4)

    if not visible:
        table.add_row("", "-", "-", "-", "No running tasks", "Waiting for updates")
    else:
        for idx, task in enumerate(visible):
            absolute_index = start + idx
            pointer = ">" if absolute_index == selected_index else " "
            task_label = str(task.get("sample_id") or task.get("task_id") or "-")
            status = str(task.get("status") or "pending")
            phase = _format_task_phase(task)
            agent_summary = task.get("agent_summary") if isinstance(task.get("agent_summary"), dict) else {}
            query = str(task.get("query") or "")
            style = "bold reverse" if absolute_index == selected_index else ""
            table.add_row(
                Text(pointer, style=style),
                Text(task_label[:14], style=style),
                Text(status[:10], style=_status_style(status) if not style else style),
                Text(phase[:18], style=style),
                _agent_bar(agent_summary),
                Text(query[:120], style=style),
            )

    footer = Text("j/k or Up/Down move | n/p or Left/Right page | Enter open task | r refresh | q quit", style="bright_black")
    return Group(
        Panel(header, title="Batch Progress", border_style="blue"),
        Panel(table, title="Running Tasks", border_style="green"),
        Panel(footer, border_style="bright_black"),
    )


def _render_task_detail(task: Optional[Dict[str, Any]], selected_agent_index: int) -> Group:
    if not task:
        return Group(Panel(Text("Task no longer exists in state file.", style="red"), title="Task Detail"))

    agents_sorted = _sorted_agents(task)
    selected_agent_index = max(0, min(selected_agent_index, max(0, len(agents_sorted) - 1)))

    table = Table(expand=True, box=None, padding=(0, 1))
    table.add_column("Sel", width=3)
    table.add_column("Agent", width=18)
    table.add_column("Status", width=16)
    table.add_column("Steps", ratio=2)
    table.add_column("Local", width=14)
    table.add_column("Total", width=14)
    table.add_column("Parallel", width=36)
    table.add_column("Current", ratio=2)
    table.add_column("Tree", width=8)

    if not agents_sorted:
        table.add_row("", "-", "-", "No agent progress", "-", "-", "", "-")
    else:
        for index, agent in enumerate(agents_sorted):
            current_label = str(agent.get("current_task_title") or agent.get("end_reason") or "")
            if agent.get("current_task_total"):
                current_label = f"{agent.get('current_task_index', 0)}/{agent.get('current_task_total', 0)} {current_label}".strip()
            task_steps_completed = int(agent.get("current_task_steps_completed", 0) or 0)
            task_steps_total = int(agent.get("current_task_steps_total", 0) or 0)
            total_steps_completed, total_steps_total = _overall_step_counts(agent)
            task_status = str(agent.get("current_task_status") or "")
            current = current_label
            local_steps_text = "-"
            if task_steps_total > 0:
                local_steps_text = f"{task_status or 'running'} {task_steps_completed}/{task_steps_total}"
            total_steps_text = f"total {total_steps_completed}/{total_steps_total}"
            parallel_text = _running_subtasks_summary(agent)
            queue_detail = _queue_reason_detail(agent.get("queue_reason"))
            if queue_detail:
                current = queue_detail if not current else f"{queue_detail} | {current}"
            running_subtasks = agent.get("running_subtasks") if isinstance(agent.get("running_subtasks"), list) else []
            if len(running_subtasks) > 1:
                current = f"{current_label} [{len(running_subtasks)} subtasks running]".strip()
            status = _format_agent_status(agent)
            has_tree = "yes" if isinstance(agent.get("task_tree"), dict) and (agent.get("task_tree") or {}).get("main_tasks") else "-"
            style = "bold reverse" if index == selected_agent_index else ""
            table.add_row(
                Text(">" if index == selected_agent_index else " ", style=style),
                str(agent.get("agent_id") or "-"),
                Text(status, style=_status_style(status)),
                _step_bar(agent),
                local_steps_text[:14],
                total_steps_text[:14],
                parallel_text[:36],
                current[:160],
                Text(has_tree, style=style),
            )

    footer = Text("j/k move | Enter open agent tree | b back | q quit | r refresh", style="bright_black")
    return Group(
        _render_task_meta(task),
        Panel(table, title="Agent Summary", border_style="green"),
        Panel(footer, border_style="bright_black"),
    )


def _render_agent_tree_detail(task: Optional[Dict[str, Any]], selected_agent_index: int) -> Group:
    if not task:
        return Group(Panel(Text("Task no longer exists in state file.", style="red"), title="Agent Tree Detail"))

    agents_sorted = _sorted_agents(task)
    if not agents_sorted:
        return Group(
            _render_task_meta(task),
            Panel(Text("No agent progress available.", style="red"), title="Agent Tree Detail", border_style="red"),
        )

    selected_agent_index = max(0, min(selected_agent_index, len(agents_sorted) - 1))
    agent = agents_sorted[selected_agent_index]
    total_steps_completed, total_steps_total = _overall_step_counts(agent)
    summary = Table.grid(expand=True)
    summary.add_column(ratio=1)
    summary.add_row(Text(f"Agent: {agent.get('agent_id') or '-'}", style="cyan"))
    summary.add_row(Text(f"Status: {_format_agent_status(agent) or '-'} | Steps: {total_steps_completed}/{total_steps_total}"))
    current_label = str(agent.get("current_task_title") or agent.get("end_reason") or "")
    running_subtasks_text = _running_subtasks_summary(agent, limit=3)
    if running_subtasks_text != "-":
        summary.add_row(Text(f"Parallel: {_ellipsize(running_subtasks_text, 120)}", style="bright_black"))
    queue_detail = _queue_reason_detail(agent.get("queue_reason"))
    if queue_detail:
        summary.add_row(Text(f"Queue: {_ellipsize(queue_detail, 120)}", style="bright_black"))
    current = current_label
    if current_label:
        task_steps_completed = int(agent.get("current_task_steps_completed", 0) or 0)
        task_steps_total = int(agent.get("current_task_steps_total", 0) or 0)
        if task_steps_total > 0:
            current = (
                f"[local {task_steps_completed}/{task_steps_total} | total {total_steps_completed}/{total_steps_total}] {current_label}"
            )
        summary.add_row(Text(f"Current: {_ellipsize(current, 120)}", style="bright_black"))

    tree_panel = _render_agent_task_tree(agent) or Panel(
        Text("This agent has no task tree data.", style="red"),
        title="Task Tree Progress",
        border_style="red",
    )
    footer = Text("b back to agent list | q quit | r refresh", style="bright_black")
    return Group(
        _render_task_meta(task),
        Panel(summary, title="Agent Detail", border_style="green"),
        tree_panel,
        Panel(footer, border_style="bright_black"),
    )


@contextmanager
def _raw_input_mode() -> Any:
    if not sys.stdin.isatty():
        yield
        return
    fd = sys.stdin.fileno()
    original = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        yield
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, original)


def _read_key(timeout: float) -> Optional[str]:
    if not sys.stdin.isatty():
        return None
    ready, _, _ = select.select([sys.stdin], [], [], timeout)
    if not ready:
        return None
    first = os.read(sys.stdin.fileno(), 1).decode("utf-8", errors="ignore")
    if first != "\x1b":
        return first
    suffix = os.read(sys.stdin.fileno(), 2).decode("utf-8", errors="ignore")
    return first + suffix


def main() -> int:
    parser = argparse.ArgumentParser(description="Terminal monitor for scripts/eval_open.py")
    parser.add_argument("--state", required=True, help="Path to eval_open monitor state JSON.")
    parser.add_argument("--refresh", type=float, default=0.5, help="Refresh interval in seconds.")
    args = parser.parse_args()

    state_path = Path(args.state)
    console = Console()
    selected_index = 0
    page = 0
    detail_task_id: Optional[str] = None
    detail_agent_index = 0
    detail_agent_tree_open = False

    with _raw_input_mode():
        with Live(console=console, screen=True, auto_refresh=False) as live:
            while True:
                state = _load_state(state_path)
                tasks = _running_tasks(state)
                if tasks:
                    selected_index = max(0, min(selected_index, len(tasks) - 1))
                    page = selected_index // PAGE_SIZE
                else:
                    selected_index = 0
                    page = 0

                if detail_task_id is not None:
                    task = _find_task(state, detail_task_id)
                    if detail_agent_tree_open:
                        live.update(_render_agent_tree_detail(task, detail_agent_index), refresh=True)
                    else:
                        live.update(_render_task_detail(task, detail_agent_index), refresh=True)
                else:
                    live.update(_render_overview(state, selected_index, page, state_path), refresh=True)

                key = _read_key(max(0.05, float(args.refresh)))
                if key is None:
                    continue
                if key in {"q", "Q"}:
                    return 0
                if key in {"r", "R"}:
                    continue
                if detail_task_id is not None:
                    if key in {"b", "B", "\x7f"}:
                        if detail_agent_tree_open:
                            detail_agent_tree_open = False
                        else:
                            detail_task_id = None
                            detail_agent_index = 0
                    elif not detail_agent_tree_open and key in {"j", "\x1b[B"}:
                        task = _find_task(state, detail_task_id)
                        agents_sorted = _sorted_agents(task) if task else []
                        if agents_sorted:
                            detail_agent_index = min(len(agents_sorted) - 1, detail_agent_index + 1)
                    elif not detail_agent_tree_open and key in {"k", "\x1b[A"}:
                        task = _find_task(state, detail_task_id)
                        agents_sorted = _sorted_agents(task) if task else []
                        if agents_sorted:
                            detail_agent_index = max(0, detail_agent_index - 1)
                    elif not detail_agent_tree_open and key in {"\r", "\n"}:
                        task = _find_task(state, detail_task_id)
                        agents_sorted = _sorted_agents(task) if task else []
                        if agents_sorted:
                            detail_agent_index = max(0, min(detail_agent_index, len(agents_sorted) - 1))
                            agent = agents_sorted[detail_agent_index]
                            has_tree = isinstance(agent.get("task_tree"), dict) and (agent.get("task_tree") or {}).get("main_tasks")
                            if has_tree:
                                detail_agent_tree_open = True
                    continue

                if key in {"j", "\x1b[B"} and tasks:
                    selected_index = min(len(tasks) - 1, selected_index + 1)
                    page = selected_index // PAGE_SIZE
                elif key in {"k", "\x1b[A"} and tasks:
                    selected_index = max(0, selected_index - 1)
                    page = selected_index // PAGE_SIZE
                elif key in {"n", "\x1b[C"} and tasks:
                    max_page = max(0, (len(tasks) - 1) // PAGE_SIZE)
                    page = min(max_page, page + 1)
                    selected_index = min(len(tasks) - 1, page * PAGE_SIZE)
                elif key in {"p", "\x1b[D"} and tasks:
                    page = max(0, page - 1)
                    selected_index = page * PAGE_SIZE
                elif key in {"\r", "\n"} and tasks:
                    detail_task_id = str(tasks[selected_index].get("task_id") or "") or None
                    detail_agent_index = 0
                    detail_agent_tree_open = False


if __name__ == "__main__":
    raise SystemExit(main())