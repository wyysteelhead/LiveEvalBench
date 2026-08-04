"""Subprocess-based isolation for task tree synthesis.

Runs ``synthesize_task_tree_for_agent`` in a child process so that
D-state (uninterruptible NFS sleep) in the child can be killed via
``kill -9`` without freezing the main event loop.

Usage
-----
    plan = await run_synthesis_in_subprocess(
        owner_agent_id="code_tester",
        user_query="...",
        dimensions=[...],
        llm_config={"provider": "custom", "api_key": "...", ...},
        timeout=300,
    )
"""

from __future__ import annotations

import asyncio
import multiprocessing
import multiprocessing.queues
import time
from typing import Any, Dict, List, Optional, Sequence

from frontend_evaluator.agent.config import DimensionConfig
from frontend_evaluator.llm.factory import LLMFactory
from frontend_evaluator.planner.task_planner import (
    MainTaskSpec,
    PlannedTask,
    TaskTreePlan,
    synthesize_task_tree_for_agent,
)

# ---------------------------------------------------------------------------
# Subprocess entry point
# ---------------------------------------------------------------------------


def _synthesis_worker(
    result_queue: multiprocessing.queues.Queue,
    error_queue: multiprocessing.queues.Queue,
    owner_agent_id: Optional[str],
    user_query: str,
    requirements: List[Dict[str, Any]],
    dimensions_data: List[Dict[str, Any]],
    round1_tasks_data: List[Dict[str, Any]],
    aligned_tasks_data: List[Dict[str, Any]],
    query_specific_main_task_count: int,
    page_context: Optional[str],
    source_files: Optional[Dict[str, str]],
    agent_profile: Optional[Dict[str, Any]],
    shared_query_main_tasks_data: Optional[List[Dict[str, Any]]],
    llm_config: Dict[str, Any],
) -> None:
    """Run inside a child process — do not call directly."""
    try:
        # Reconstruct domain objects from serialized dicts
        dimensions = [DimensionConfig.model_validate(d) for d in dimensions_data]

        round1_tasks = (
            [PlannedTask.from_dict(t) for t in round1_tasks_data]
            if round1_tasks_data
            else []
        )
        aligned_tasks = (
            [PlannedTask.from_dict(t) for t in aligned_tasks_data]
            if aligned_tasks_data
            else []
        )
        shared_query_main_tasks = (
            [MainTaskSpec.from_dict(t) for t in shared_query_main_tasks_data]
            if shared_query_main_tasks_data
            else None
        )

        # Create a fresh LLM inside the child process.
        # Environment variables (API keys, base URLs) are inherited from the
        # parent via fork/spawn so reading them here works correctly.
        llm = LLMFactory.create_llm(
            provider=llm_config["provider"],
            api_key=llm_config["api_key"],
            model=llm_config.get("model"),
            base_url=llm_config.get("base_url"),
            temperature=0,
            disable_thinking=False,
        )

        result = asyncio.run(
            synthesize_task_tree_for_agent(
                owner_agent_id=owner_agent_id,
                user_query=user_query,
                requirements=requirements,
                dimensions=dimensions,
                round1_tasks=round1_tasks,
                aligned_tasks=aligned_tasks,
                query_specific_main_task_count=query_specific_main_task_count,
                llm=llm,
                page_context=page_context,
                source_files=source_files or {},
                agent_profile=agent_profile,
                shared_query_main_tasks=shared_query_main_tasks,
            )
        )
        result_queue.put(result.to_dict())
    except Exception as exc:  # noqa: BLE001
        import traceback

        error_queue.put(f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def run_synthesis_in_subprocess(
    *,
    owner_agent_id: Optional[str],
    user_query: str,
    requirements: Sequence[Dict[str, Any]],
    dimensions: Sequence[Any],
    round1_tasks: Sequence[Any],
    aligned_tasks: Sequence[Any],
    query_specific_main_task_count: int,
    page_context: Optional[str],
    source_files: Optional[Dict[str, str]],
    agent_profile: Optional[Dict[str, Any]],
    shared_query_main_tasks: Optional[Sequence[Any]],
    llm_config: Dict[str, Any],
    timeout: float = 300,
) -> TaskTreePlan:
    """Run ``synthesize_task_tree_for_agent`` in a child process.

    Unlike ``asyncio.wait_for``, this provides a **real wall-clock timeout**
    enforced by ``kill -9`` on the child process.  If the child is stuck in
    D state (uninterruptible NFS sleep) the parent can still terminate it.

    Parameters
    ----------
    llm_config : dict
        Provider configuration used to create the LLM inside the child::

            {
                "provider": "custom",
                "api_key": "...",
                "model": "...",
                "base_url": "...",
            }

    timeout : float
        Maximum wall-clock seconds to wait before killing the child.
        Default 300 (5 minutes, matching the original ``asyncio.wait_for``).

    Returns
    -------
    TaskTreePlan
        The synthesised task plan.

    Raises
    ------
    asyncio.TimeoutError
        If the child process does not complete within *timeout* seconds.
    RuntimeError
        If the child process raises an exception during synthesis.
    """
    # ------------------------------------------------------------------
    # Serialise complex parameters to JSON-compatible dicts
    # ------------------------------------------------------------------
    dimensions_data: List[Dict[str, Any]] = [
        d.model_dump() if hasattr(d, "model_dump") else dict(d) for d in dimensions
    ]
    round1_tasks_data: List[Dict[str, Any]] = [t.to_dict() for t in (round1_tasks or [])]
    aligned_tasks_data: List[Dict[str, Any]] = [t.to_dict() for t in (aligned_tasks or [])]

    shared_query_main_tasks_data: Optional[List[Dict[str, Any]]] = (
        [t.to_dict() for t in shared_query_main_tasks]
        if shared_query_main_tasks
        else None
    )

    # ------------------------------------------------------------------
    # Start child process
    # ------------------------------------------------------------------
    ctx = multiprocessing.get_context("fork")
    result_queue: multiprocessing.queues.Queue = ctx.Queue()
    error_queue: multiprocessing.queues.Queue = ctx.Queue()

    process = ctx.Process(
        target=_synthesis_worker,
        args=(
            result_queue,
            error_queue,
            owner_agent_id,
            user_query,
            list(requirements),
            dimensions_data,
            round1_tasks_data,
            aligned_tasks_data,
            query_specific_main_task_count,
            page_context,
            dict(source_files) if source_files else None,
            agent_profile,
            shared_query_main_tasks_data,
            llm_config,
        ),
    )
    process.start()

    # ------------------------------------------------------------------
    # Monitor with wall-clock timeout
    # ------------------------------------------------------------------
    deadline = time.monotonic() + timeout
    try:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise asyncio.TimeoutError(
                    f"Task synthesis timed out after {timeout:.0f}s "
                    f"(subprocess PID {process.pid})"
                )

            # Propagate child-side exceptions
            if not error_queue.empty():
                error_text: str = error_queue.get_nowait()
                raise RuntimeError(
                    f"Task synthesis subprocess error:\n{error_text}"
                )

            # Success
            if not result_queue.empty():
                result_data: Dict[str, Any] = result_queue.get_nowait()
                plan = TaskTreePlan()
                plan.synthesis_mode = result_data.get("synthesis_mode", "tree")
                plan.owner_agent_id = result_data.get("owner_agent_id")
                plan.main_tasks = [
                    MainTaskSpec.from_dict(mt)
                    for mt in result_data.get("main_tasks", [])
                ]
                return plan

            await asyncio.sleep(0.5)
    finally:
        # Ensure the child is cleaned up
        if process.is_alive():
            process.kill()
            process.join(timeout=10)