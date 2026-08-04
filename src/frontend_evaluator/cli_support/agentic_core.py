from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from .common import maybe_await

if TYPE_CHECKING:
    from frontend_evaluator.utils.config import Config


def _summary_progress_ratio(summary: dict[str, Any]) -> float | None:
    overall_score = summary.get("overall_score")
    overall_max_score = summary.get("overall_max_score")
    if isinstance(overall_score, (int, float)) and isinstance(overall_max_score, (int, float)):
        max_score = float(overall_max_score)
        if max_score > 0:
            return float(overall_score) / max_score

    avg_score = summary.get("avg_main_task_pass_rate")
    if isinstance(avg_score, (int, float)):
        return float(avg_score)

    avg_score = summary.get("avg_task_completion_score")
    if isinstance(avg_score, (int, float)):
        return float(avg_score)

    return None


def derive_agentic_status(report: dict[str, Any]) -> str:
    summary = report.get("summary", {})
    avg_score = _summary_progress_ratio(summary)
    if avg_score is not None:
        if avg_score >= 0.7:
            return "passed"
        if avg_score >= 0.4:
            return "partial"
        return "failed"

    statuses = [agent.get("status") for agent in report.get("agents", [])]
    if not statuses:
        return "failed"

    completed = sum(1 for status in statuses if status == "completed")
    if completed == len(statuses):
        return "passed"
    if completed > 0:
        return "partial"
    return "failed"


def _create_sandbox(config: "Config", sandbox_provider: str):
    """Sandbox creation is NOT supported in the public (local-only) release.

    eval_open.py runs locally via CdpPlaywrightExecutor and never calls this.
    Remote sandbox providers (E2B/Docker/Remote/OpenSandbox) were removed.
    """
    raise NotImplementedError(
        "Remote sandbox providers are not included in the public release. "
        "Use eval_open.py with EXECUTOR_BACKEND=playwright (local)."
    )


async def run_agentic_single(
    *,
    config: "Config",
    logger: Any,
    files: dict[str, str],
    query: str,
    sandbox_provider: str,
    agents_dir: str = "agents",
    max_parallel: int | None = None,
    planned_tasks: list[Any] | None = None,
    planned_tasks_by_agent: dict[str, list[Any]] | None = None,
    agent_ids: list[str] | None = None,
) -> dict[str, Any]:
    from frontend_evaluator.agent.orchestrator import AgenticOrchestrator
    from frontend_evaluator.llm.factory import LLMFactory
    from frontend_evaluator.planner.query_decomposer import decompose
    from frontend_evaluator.sandbox.agent_browser_executor import get_executor

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

        initial_screenshot_b64 = None
        page_context = None
        query_decomposition = None
        if query and planned_tasks is None and planned_tasks_by_agent is None:
            try:
                llm = LLMFactory.create_llm(
                    provider=config.model_provider,
                    api_key=config.get_llm_api_key(),
                    model=config.model_name,
                    base_url=config.custom_base_url if config.model_provider == "custom" else None,
                    temperature=config.temperature,
                    disable_thinking=config.disable_thinking,
                )
                initial_screenshot_b64 = await executor.screenshot()
                ctx = await executor.get_context()
                page_context = str(ctx.get("accessibility_tree", "")) if ctx else None
                query_decomposition = await decompose(query, llm)
                logger.info("Initial context collected; query decomposed for planning")
            except Exception as exc:
                logger.warning("Initial context collection failed (planning will be skipped): %s", exc)

        orchestrator = AgenticOrchestrator(
            config,
            agents_dir=agents_dir,
            max_parallel=max_parallel,
        )
        return await orchestrator.run(
            app_url,
            executor,
            source_files=files,
            user_query=query or None,
            query_decomposition=query_decomposition,
            initial_screenshot_b64=initial_screenshot_b64,
            page_context=page_context,
            planned_tasks=planned_tasks,
            planned_tasks_by_agent=planned_tasks_by_agent,
            agent_ids=agent_ids,
        )
    finally:
        if executor:
            await executor.shutdown()
        if sandbox:
            sandbox.stop()


async def run_agentic_batch(
    *,
    rows: list[dict[str, Any]],
    args: Any,
    config: "Config",
    logger: Any,
    sandbox_provider: str,
    max_parallel: int | None,
    existing_index: dict[str, dict[str, Any]],
    files_from_row_fn: Any,
    on_result: Any = None,
) -> list[dict[str, Any]]:
    semaphore = asyncio.Semaphore(max(1, max_parallel or 1))

    async def run_row(row: dict[str, Any]) -> dict[str, Any]:
        sample_id = str(row.get("id", "")).strip()
        if sample_id and sample_id in existing_index:
            existing_row = existing_index[sample_id]
            existing_status = str(existing_row.get("status", "")).lower()
            should_skip = existing_status in {"passed", "partial"} if args.rerun_nonpassed else True
            if should_skip:
                logger.info("Skipping '%s' (status=%s)", sample_id, existing_status)
                return existing_row

        files = files_from_row_fn(row)
        query = str(row.get("query", args.query or "")).strip()
        async with semaphore:
            try:
                report = await run_agentic_single(
                    config=config,
                    logger=logger,
                    files=files,
                    query=query,
                    sandbox_provider=sandbox_provider,
                    agents_dir=args.agents_dir,
                    max_parallel=max_parallel,
                )
                return {
                    "id": sample_id,
                    "status": derive_agentic_status(report),
                    "query": query,
                    "result": report,
                }
            except Exception as exc:
                logger.error("Row '%s' failed: %s", sample_id, exc, exc_info=True)
                return {"id": sample_id, "status": "failed", "error": str(exc)}

    results: list[dict[str, Any] | None] = [None] * len(rows)

    async def run_indexed(index: int, row: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        return index, await run_row(row)

    tasks = [asyncio.create_task(run_indexed(index, row)) for index, row in enumerate(rows)]
    for completed_task in asyncio.as_completed(tasks):
        index, row_result = await completed_task
        results[index] = row_result
        if on_result is not None:
            awaited = maybe_await(on_result(index, row_result, results))
            if awaited is not None:
                await awaited

    return [row for row in results if row is not None]
