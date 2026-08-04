from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Sequence

from ..agent.runtime_factory import create_evaluator_runtime

if TYPE_CHECKING:
    from ..utils.config import Config


@dataclass(slots=True)
class ChecklistItem:
    """One externally defined checklist item executed by a single evaluator."""

    instruction: str
    title: str | None = None
    item_id: str | None = None
    covers_standard_ids: list[str] = field(default_factory=list)
    expected_signals: list[str] = field(default_factory=list)
    preconditions: list[str] = field(default_factory=list)
    multi_step: bool = True
    scenario_weight: float = 1.0


def coerce_checklist_item(
    item: ChecklistItem | Mapping[str, Any] | str,
    *,
    index: int,
    prefix: str,
) -> ChecklistItem:
    if isinstance(item, ChecklistItem):
        if not item.instruction.strip():
            raise ValueError(f"Checklist item {index} is missing instruction")
        return item

    if isinstance(item, str):
        instruction = item.strip()
        if not instruction:
            raise ValueError(f"Checklist item {index} is empty")
        return ChecklistItem(instruction=instruction, item_id=f"{prefix}_{index}")

    if isinstance(item, Mapping):
        instruction = str(
            item.get("instruction") or item.get("task_text") or item.get("text") or ""
        ).strip()
        if not instruction:
            raise ValueError(f"Checklist item {index} is missing instruction/task_text/text")
        title = str(item.get("title") or item.get("name") or "").strip() or None
        item_id = str(item.get("item_id") or item.get("id") or "").strip() or f"{prefix}_{index}"
        return ChecklistItem(
            instruction=instruction,
            title=title,
            item_id=item_id,
            covers_standard_ids=[str(value) for value in item.get("covers_standard_ids", []) or []],
            expected_signals=[str(value) for value in item.get("expected_signals", []) or []],
            preconditions=[str(value) for value in item.get("preconditions", []) or []],
            multi_step=bool(item.get("multi_step", True)),
            scenario_weight=float(item.get("scenario_weight", 1.0) or 1.0),
        )

    raise TypeError(f"Unsupported checklist item type at index {index}: {type(item)!r}")


def build_checklist_task_prompt(task: ChecklistItem) -> str:
    prompt = task.instruction.strip()
    sections: list[str] = [prompt]

    if task.preconditions:
        preconditions = "\n".join(f"- {item}" for item in task.preconditions)
        sections.append(f"Preconditions:\n{preconditions}")

    if task.expected_signals:
        expected_signals = "\n".join(f"- {item}" for item in task.expected_signals)
        sections.append(f"Expected signals:\n{expected_signals}")

    return "\n\n".join(section for section in sections if section)


def summarize_checklist_results(task_results: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    total = len(task_results)
    passed = sum(1 for row in task_results if row.get("passed") is True)
    failed = sum(1 for row in task_results if row.get("passed") is False)
    total_iterations = sum(int(row.get("iterations", 0) or 0) for row in task_results)
    evaluated = passed + failed
    avg_task_completion = (passed / evaluated) if evaluated else None
    return {
        "tasks_total": total,
        "tasks_passed": passed,
        "tasks_failed": failed,
        "tasks_evaluated": evaluated,
        "avg_task_completion_score": avg_task_completion,
        "total_steps_used": total_iterations,
    }


def _create_sandbox(config: "Config", sandbox_provider: str):
    """Sandbox creation is NOT supported in the public (local-only) release.

    The benchmark runs locally via Playwright (EXECUTOR_BACKEND=playwright);
    `eval_open.py` uses CdpPlaywrightExecutor directly and never calls this.
    Remote sandbox providers (E2B/Docker/Remote/OpenSandbox) were removed.
    """
    raise NotImplementedError(
        "Remote sandbox providers are not included in the public release. "
        "Use eval_open.py with EXECUTOR_BACKEND=playwright (local)."
    )


async def run_checklist(
    *,
    config: "Config",
    logger: Any,
    files: Dict[str, str],
    checklist: Sequence[ChecklistItem],
    query: str,
    sandbox_provider: str,
) -> Dict[str, Any]:
    from ..sandbox.agent_browser_executor import get_executor

    sandbox = None
    executor = None
    try:
        sandbox = _create_sandbox(config, sandbox_provider)
        sandbox.start(timeout=config.sandbox_startup_timeout)
        sandbox.write_files(files)
        app_url = await sandbox.start_app(
            install_timeout=config.sandbox_startup_timeout,
            startup_timeout=config.app_startup_timeout,
        )

        executor = get_executor(config.executor_backend, sandbox, app_url)
        await executor.start_service(timeout=config.cdp_connection_timeout)

        evaluator = create_evaluator_runtime(
            config,
            max_iterations=config.max_agent_steps,
            log_level=config.log_level,
        )

        task_results: List[Dict[str, Any]] = []
        for index, task in enumerate(checklist, start=1):
            try:
                await executor.reset(app_url)
            except Exception as exc:
                logger.warning(
                    "Checklist task reset failed before task %s: %s",
                    task.item_id or index,
                    exc,
                )

            task_prompt = build_checklist_task_prompt(task)
            if query:
                task_prompt = f"Context: {query}\n\nTask:\n{task_prompt}"

            try:
                result = await evaluator.evaluate(
                    user_query=task_prompt,
                    app_url=app_url,
                    executor=executor,
                )
                verdict = result.get("verdict", {})
                task_results.append(
                    {
                        "task_id": task.item_id or f"checklist_{index}",
                        "title": task.title or f"Checklist item {index}",
                        "instruction": task.instruction,
                        "passed": bool(verdict.get("passed")),
                        "verdict": verdict.get("verdict", "failed"),
                        "reason": verdict.get("reason", ""),
                        "iterations": result.get("iterations", 0),
                        "steps": result.get("steps", []),
                        "dom_elements": result.get("dom_elements", []),
                        "expected_signals": list(task.expected_signals),
                        "preconditions": list(task.preconditions),
                    }
                )
            except Exception as exc:
                logger.error(
                    "Checklist task %s failed: %s",
                    task.item_id or index,
                    exc,
                    exc_info=True,
                )
                task_results.append(
                    {
                        "task_id": task.item_id or f"checklist_{index}",
                        "title": task.title or f"Checklist item {index}",
                        "instruction": task.instruction,
                        "passed": False,
                        "verdict": "failed",
                        "reason": str(exc),
                        "iterations": 0,
                        "steps": [],
                        "dom_elements": [],
                        "expected_signals": list(task.expected_signals),
                        "preconditions": list(task.preconditions),
                    }
                )

        return {
            "evaluation_mode": "checklist",
            "query": query,
            "app_url": app_url,
            "summary": summarize_checklist_results(task_results),
            "tasks": task_results,
        }
    finally:
        if executor:
            await executor.shutdown()
        if sandbox:
            sandbox.stop()