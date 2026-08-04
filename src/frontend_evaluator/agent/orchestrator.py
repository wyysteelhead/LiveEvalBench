"""Multi-agent concurrent orchestrator for agentic evaluation mode."""

import asyncio
import inspect
import json
import traceback
import os
import time
import uuid
from typing import Any, Awaitable, Callable, Dict, List, Optional, Union

from frontend_evaluator.sandbox import apply_font_load_timeout
from .agent_registry import AgentRegistry
from .config import AgentConfig
from .evaluator import InfraError
from .pipeline import AgentHandoff, format_dependency_handoffs
from .report import build_agentic_report, normalize_agent_result
from .runtime_factory import create_evaluator_runtime
from .runtime_interface import EvaluatorRuntime
from ..events.emitter import EventEmitter
from ..tools.registry import get_tool_function
from ..tools.source_reader import set_source_files
from ..utils.config import Config
from ..utils.logger import logger
from ..planner.synthesis_isolation import run_synthesis_in_subprocess
from ..planner.task_planner import (
    MainTaskSpec,
    PlannedTask,
    build_task_tree_payload,
    build_execution_plan,
    synthesize_task_tree_for_agent,
    synthesize_task_tree_from_planned_tasks,
)


def _tree_subtask_constraint_dimension_ids(task: PlannedTask) -> List[str]:
    based_on = task.based_on if isinstance(task.based_on, dict) else {}
    return [
        str(item)
        for item in (based_on.get("constraint_dimension_ids") or [])
        if item is not None
    ]


def _tree_subtask_list_field(task: PlannedTask, key: str) -> List[str]:
    based_on = task.based_on if isinstance(task.based_on, dict) else {}
    return [
        str(item)
        for item in (based_on.get(key) or [])
        if item is not None and str(item).strip()
    ]


def _tree_subtask_depends_on_ids(task: PlannedTask) -> List[str]:
    based_on = task.based_on if isinstance(task.based_on, dict) else {}
    return [
        str(item)
        for item in (based_on.get("depends_on_subtask_ids") or [])
        if item is not None and str(item).strip()
    ]


def _tree_subtask_can_run_parallel(task: PlannedTask) -> bool:
    based_on = task.based_on if isinstance(task.based_on, dict) else {}
    return bool(based_on.get("can_run_parallel", False))


def _build_tree_execution_batches(
    planned_tasks: List[PlannedTask],
    *,
    parallel_budget: int,
) -> List[List[PlannedTask]]:
    if parallel_budget <= 1:
        return [[task] for task in planned_tasks]

    batches: List[List[PlannedTask]] = []
    pending: List[PlannedTask] = list(planned_tasks)
    completed: set[str] = set()

    while pending:
        ready = [
            task
            for task in pending
            if all(dep_id in completed for dep_id in _tree_subtask_depends_on_ids(task))
        ]
        if not ready:
            ready = [pending[0]]

        parallel_ready = [task for task in ready if _tree_subtask_can_run_parallel(task)]
        batch = parallel_ready[:parallel_budget] if parallel_ready else [ready[0]]
        batches.append(batch)

        for task in batch:
            completed.add(task.task_id)
            pending.remove(task)

    return batches


def _classify_task_exception(exc: Exception) -> tuple[str, str]:
    message = str(exc).lower()
    infra_markers = (
        "api key",
        "authentication",
        "unauthorized",
        "rate limit",
        "quota",
        "provider api",
        "error calling anthropic api",
        "error calling openai api",
        "error calling google api",
        "bad request",
        "invalid_request_error",
        "tool call",
        "tool arguments",
        "missing 1 required positional argument",
        "connection reset",
        "connection refused",
        "connection aborted",
        "connection error",
        "service unavailable",
        "gateway timeout",
        "timed out while",
        "timeout while",
    )
    if isinstance(exc, (InfraError, asyncio.TimeoutError, TimeoutError)) or type(exc).__name__ in ("TimeoutError", "InfraError"):
        return "inconclusive", "infra_failure"
    if any(marker in message for marker in infra_markers):
        return "inconclusive", "infra_failure"
    return "failed", "execution_failure"


def _is_inconclusive_infra(outcome: Dict[str, Any]) -> bool:
    """Check whether a task outcome is inconclusive due to infrastructure failure
    and should be retried. Returns True for timeout, infra_error, and
    inconclusive verdicts caused by infra failures."""
    tr = outcome.get("task_result") or {}
    status = str(tr.get("status", "")).lower()
    verdict = str(tr.get("verdict", "")).lower()
    error_cat = str(tr.get("error_category", "")).lower()
    return (
        status in ("infra_error", "timeout")
        or verdict == "inconclusive"
        or error_cat == "infra_failure"
    )


def _detect_artifacts_write_failure(
    executed_task: Dict[str, Any],
) -> Optional[str]:
    """Scan a build agent task's trajectory for a failed protocol_write_artifacts call.

    When the build engineer calls ``protocol_write_artifacts`` with ``state=ready``
    and the app/cdp URL is not reachable, the tool returns an ``"error"`` key.
    This function returns the error message if found, otherwise ``None``.

    Only the **last** protocol_write_artifacts call is considered — if the agent
    retried and the final attempt succeeded, earlier transient errors are ignored.
    """
    steps = executed_task.get("trajectory", {}).get("steps", [])
    if not isinstance(steps, list):
        return None

    # Find the last protocol_write_artifacts step
    last_idx = -1
    for idx, step in enumerate(steps):
        if isinstance(step, dict) and step.get("tool_name") == "protocol_write_artifacts":
            last_idx = idx

    if last_idx < 0:
        return None

    step = steps[last_idx]
    result = str(step.get("result", ""))
    if '"error":' not in result:
        return None
    import re as _re
    m = _re.search(r'"error":\s*"([^"]*)"', result)
    if m:
        msg = m.group(1).replace("\\n", " ").replace("\\t", " ")
        return msg
    return None


def _tree_subtask_prompt(
    task: PlannedTask,
    constraint_dimension_ids: List[str],
    agent_config: Optional[Any] = None,
) -> str:
    based_on = task.based_on if isinstance(task.based_on, dict) else {}
    prompt_parts = [task.task_text]

    subtask_goal = str(based_on.get("subtask_goal", "") or "").strip()
    if subtask_goal:
        prompt_parts.append(f"Goal: {subtask_goal}")

    success_criteria = _tree_subtask_list_field(task, "success_criteria")
    if success_criteria:
        prompt_parts.append("Success criteria:\n- " + "\n- ".join(success_criteria))

    expected_signals = [str(item) for item in (task.expected_signals or []) if str(item).strip()]
    if expected_signals:
        prompt_parts.append("Expected visible signals:\n- " + "\n- ".join(expected_signals))

    evidence_requirements = _tree_subtask_list_field(task, "evidence_requirements")
    if evidence_requirements:
        prompt_parts.append("Collect evidence from:\n- " + "\n- ".join(evidence_requirements))

    failure_policy = str(based_on.get("failure_policy", "") or "").strip()
    estimated_cost = str(based_on.get("estimated_cost", "") or "").strip()
    execution_rules: List[str] = [
        "Return a single overall binary verdict for this subtask.",
        "Do not conclude failure from one short unsuccessful attempt when the feature is stateful or requires sequencing.",
        "For games, canvases, editors, drag/drop flows, or other stateful interactions, try alternative action sequences until the target condition is achieved or repeated evidence shows it cannot be achieved.",
        "For visibility, blockage, overlay, or prominence claims, prioritize direct rendered evidence such as screenshots or clearly observable page state over DOM or accessibility-tree text alone. Hidden DOM/a11y content by itself is not enough to prove the user can see it.",
        "Do not fail a task solely because subtle animation, shimmer, hover styling, or motion smoothness could not be confidently judged from static visual evidence. If implementation evidence is unavailable and the visual evidence is inconclusive, avoid turning that uncertainty into a product failure.",
        "Do not fail a task solely because a high-precision coordinate sequence, long board-game construction, or negative click test may have been affected by tool precision limits. Only conclude failure when the broken behavior is reproducible from stable observable evidence.",
        "Do not treat a functionally reasonable implementation deviation as a failure just because it is not written exactly as the user query or spec text. Only fail when the deviation breaks a core requirement, produces incorrect behavior, or creates a meaningful engineering risk for this subtask.",
        "If you conclude failure, explain what sequences you tried and what observable blocker prevented success.",
    ]
    if failure_policy:
        execution_rules.append(f"Failure policy: {failure_policy}.")
    if estimated_cost:
        execution_rules.append(f"Expected effort for this subtask: {estimated_cost}.")
    prompt_parts.append("Execution rules:\n- " + "\n- ".join(execution_rules))

    if constraint_dimension_ids:
        dim_refs: List[str] = []
        for dim_id in constraint_dimension_ids:
            ref = dim_id
            if agent_config is not None:
                rubric = getattr(agent_config, "rubric", None)
                if rubric is not None:
                    dimensions = getattr(rubric, "dimensions", []) or []
                    for dim in dimensions:
                        if getattr(dim, "id", None) == dim_id:
                            scoring = getattr(dim, "scoring", None)
                            if scoring is not None:
                                checks = getattr(scoring, "checks", None) or []
                                subcriteria = getattr(scoring, "subcriteria", None) or []
                                check_items = [f"check '{chk.id}': {chk.instruction}" for chk in checks]
                                sub_items = [f"subcriterion '{sub.id}': {sub.instruction}" for sub in subcriteria]
                                if check_items or sub_items:
                                    ref = f"{dim_id} — focus areas:\n  - " + "\n  - ".join(check_items + sub_items)
                            break
            dim_refs.append(ref)
        prompt_parts.append(
            "Planning constraints for this subtask: use these as coverage reminders while testing, "
            f"but do not return per-dimension verdicts. Relevant dimensions: {', '.join(constraint_dimension_ids)}."
        )
        if any("\n" in ref for ref in dim_refs):
            prompt_parts.append(
                "Dimension check references:\n" + "\n".join(dim_refs)
            )

    return "\n\n".join(prompt_parts)


def _build_dimensions_from_main_task_tree(
    *,
    agent_config: AgentConfig,
    task_tree_payload: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    dimensions_from_main_tasks: List[Dict[str, Any]] = []
    for dim_cfg in agent_config.rubric.dimensions:
        covering = [
            main_task
            for main_task in task_tree_payload
            if isinstance(main_task, dict) and dim_cfg.id in (main_task.get("dimension_ids") or [])
        ]
        if not covering:
            dimensions_from_main_tasks.append({
                "dimension_id": dim_cfg.id,
                "verdict": None,
                "reason": "No main task covered this dimension.",
                "evidence": [],
                "score": None,
                "score_breakdown": {
                    "method": "main_task_aggregation",
                    "final_score": None,
                    "main_task_count": 0,
                    "main_task_scores": [],
                },
                "weight": float(dim_cfg.weight),
                "verdict_source": "main_task_aggregation",
            })
            continue

        scores = [
            float(main_task.get("completion_score")) * 100.0
            for main_task in covering
            if isinstance(main_task.get("completion_score"), (int, float))
        ]
        statuses = {
            str(main_task.get("status") or main_task.get("verdict") or "").strip().lower()
            for main_task in covering
            if str(main_task.get("status") or main_task.get("verdict") or "").strip()
        }
        statuses.discard("")
        if statuses == {"passed"}:
            dim_verdict = "passed"
        elif statuses == {"failed"}:
            dim_verdict = "failed"
        elif statuses:
            dim_verdict = "partial"
        else:
            dim_verdict = None

        reason_parts: List[str] = []
        for main_task in covering:
            if main_task.get("failure_stage"):
                reason_parts.append(f"{main_task.get('title')}: failed at {main_task.get('failure_stage')}")
            elif main_task.get("key_findings"):
                key_findings = main_task.get("key_findings") or []
                if key_findings:
                    reason_parts.append(f"{main_task.get('title')}: {key_findings[0]}")

        dimensions_from_main_tasks.append({
            "dimension_id": dim_cfg.id,
            "verdict": dim_verdict,
            "reason": "; ".join(reason_parts[:3]) or f"Aggregated from {len(covering)} main task results.",
            "evidence": [],
            "score": (sum(scores) / len(scores)) if scores else None,
            "score_breakdown": {
                "method": "main_task_aggregation",
                "final_score": (sum(scores) / len(scores)) if scores else None,
                "main_task_count": len(covering),
                "main_task_scores": [
                    {
                        "main_task_id": main_task.get("main_task_id"),
                        "score": (float(main_task.get("completion_score")) * 100.0) if isinstance(main_task.get("completion_score"), (int, float)) else None,
                        "verdict": main_task.get("status") or main_task.get("verdict"),
                    }
                    for main_task in covering
                ],
            },
            "weight": float(dim_cfg.weight),
            "verdict_source": "main_task_aggregation",
        })

    return dimensions_from_main_tasks


class AgenticOrchestrator:
    """Runs multiple persona agents concurrently against the same app.

    Each agent gets its own isolated browser context (when using
    CdpPlaywrightExecutor) so that cookies, localStorage, and DOM state
    are independent.  For other executors the agents run sequentially,
    calling executor.reset(app_url) between runs.
    """

    def __init__(
        self,
        config: Config,
        agents_dir: str = "agents",
        max_parallel: Optional[int] = None,
        trajectory_dir: str = "artifacts/agentic_trajectories",
        subtask_semaphore: Optional["asyncio.Semaphore"] = None,
    ):
        self.config = config
        self.agents_dir = agents_dir
        self.max_parallel = max_parallel
        self.subtask_semaphore = subtask_semaphore
        # Kept for backward-compatible constructor signature; trajectories are now
        # embedded in report payloads for easier debugging in web views.
        self.trajectory_dir = trajectory_dir

    async def run(
        self,
        app_url: str,
        executor: Any,
        source_files: Optional[Dict[str, str]] = None,
        task_id: Optional[str] = None,
        user_query: Optional[str] = None,
        query_decomposition: Optional[Dict] = None,
        initial_screenshot_b64: Optional[str] = None,
        page_context: Optional[str] = None,
        role_filter: Optional[str] = None,
        planned_tasks: Optional[List[PlannedTask]] = None,
        planned_tasks_by_agent: Optional[Dict[str, List[PlannedTask]]] = None,
        shared_task_context_by_agent: Optional[Dict[str, str]] = None,
        shared_query_main_tasks: Optional[List[MainTaskSpec]] = None,
        query_specific_main_task_count: Optional[int] = None,
        agent_ids: Optional[List[str]] = None,
        dependency_handoffs: Optional[Dict[str, List[Dict[str, Any]]]] = None,
        progress_callback: Optional[Callable[[str, Dict[str, Any]], Any]] = None,
        extra_context: Optional[Dict[str, Any]] = None,
        agent_overrides: Optional[Dict[str, Dict[str, Any]]] = None,
        worker_executors: Optional[Union[List[Any], asyncio.Task]] = None,
    ) -> Dict[str, Any]:
        """Run all enabled agents and return the aggregated task-level report.

        Args:
            app_url: URL of the running Next.js application.
            executor: Shared browser executor (CdpPlaywrightExecutor or similar).
            task_id: Optional task identifier propagated into the report.
            user_query: User query for function-first task planning.
            query_decomposition: Pre-computed query decomposition dict.
            initial_screenshot_b64: Base64 screenshot of initial app state.
            page_context: Accessibility tree string of initial app state.
            worker_executors: Optional list of additional CdpPlaywrightExecutor
                instances, or an ``asyncio.Task`` that resolves to one.  When
                provided, subtasks within each agent are round-robined across
                all workers to reduce CDP contention.

        Returns:
            Aggregated report dict (see report.build_agentic_report).
        """
        # Resolve background worker scale-up task if still running
        if isinstance(worker_executors, asyncio.Task):
            try:
                worker_executors = await worker_executors
            except Exception as _wex:
                logger.warning(
                    "Background worker scale-up task failed, "
                    "falling back to single-worker mode: %s",
                    _wex,
                )
                worker_executors = None

        effective_task_id = task_id or str(uuid.uuid4())
        set_source_files(source_files or {})
        registry = AgentRegistry(self.agents_dir)
        agents = registry.load()

        if role_filter:
            agents = [a for a in agents if getattr(a, "role", "evaluator") == role_filter]

        if agent_ids:
            allowed_agent_ids = {agent_id for agent_id in agent_ids if agent_id}
            agents = [a for a in agents if a.id in allowed_agent_ids]

        if agent_overrides:
            for agent in agents:
                override = agent_overrides.get(agent.id)
                if override is None:
                    continue
                if "allowed_tools" in override:
                    agent.allowed_tools = override["allowed_tools"]
                if "system_prompt" in override:
                    agent.system_prompt = override["system_prompt"]

        if not agents:
            logger.warning("No enabled agents found — returning empty report")
            return build_agentic_report([], task_id=effective_task_id)

        logger.info(f"Running {len(agents)} agent(s) in agentic mode")
        per_task_budget = self._compute_per_task_agent_parallelism(agent_count=len(agents))
        if self.subtask_semaphore is not None:
            per_task_budget = 1
        logger.info(
            "Agentic concurrency budget: total=%s, agents=%s, per_task_parallelism=%s",
            self._get_total_parallel_budget(),
            len(agents),
            per_task_budget,
        )

        # Determine whether we can run concurrently (CDP executor has a shared
        # Browser object we can fork new contexts from).
        is_cdp = self._is_cdp_executor(executor)

        # Build planning context to pass to each agent
        planning_ctx = {
            "user_query": user_query,
            "query_decomposition": query_decomposition,
            "initial_screenshot_b64": initial_screenshot_b64,
            "page_context": page_context,
            "source_files": source_files or {},
            "planned_tasks": planned_tasks,
            "planned_tasks_by_agent": planned_tasks_by_agent or {},
            "shared_task_context_by_agent": shared_task_context_by_agent or {},
            "shared_query_main_tasks": shared_query_main_tasks,
            "query_specific_main_task_count": query_specific_main_task_count,
            "dependency_handoffs": dependency_handoffs or {},
            "tree_subtask_parallelism": per_task_budget if is_cdp else 1,
            "subtask_semaphore": self.subtask_semaphore,
        }
        if isinstance(worker_executors, list) and len(worker_executors) > 1:
            extra_context = dict(extra_context or {})
            extra_context["worker_executors"] = worker_executors

        if extra_context:
            planning_ctx.update(extra_context)

        if is_cdp:
            semaphore = asyncio.Semaphore(per_task_budget)

            async def run_one(agent: AgentConfig):
                async with semaphore:
                    return await self._run_single_agent_cdp(
                        agent,
                        app_url,
                        executor,
                        self._planning_context_for_agent(planning_ctx, agent.id),
                        progress_callback=progress_callback,
                        worker_executors=worker_executors,
                    )

            tasks = [run_one(agent) for agent in agents]
            results = await asyncio.gather(*tasks, return_exceptions=True)
        else:
            results = []
            for agent in agents:
                result = await self._run_single_agent_sequential(
                    agent,
                    app_url,
                    executor,
                    self._planning_context_for_agent(planning_ctx, agent.id),
                    progress_callback=progress_callback,
                )
                results.append(result)

        # Normalise any exceptions returned by gather into error dicts
        clean_results: List[Dict[str, Any]] = []
        for agent, result in zip(agents, results):
            if isinstance(result, Exception):
                logger.error(f"Agent '{agent.id}' raised an exception: {result}")
                clean_results.append(self._error_result(agent, str(result)))
            else:
                clean_results.append(result)

        return build_agentic_report(clean_results, task_id=effective_task_id)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _planning_context_for_agent(planning_ctx: Dict[str, Any], agent_id: str) -> Dict[str, Any]:
        agent_specific = dict(planning_ctx)
        planned_tasks_by_agent = planning_ctx.get("planned_tasks_by_agent") or {}
        shared_task_context_by_agent = planning_ctx.get("shared_task_context_by_agent") or {}
        agent_specific["planned_tasks_override"] = (
            planned_tasks_by_agent.get(agent_id)
            if agent_id in planned_tasks_by_agent
            else planning_ctx.get("planned_tasks")
        )
        if agent_id in shared_task_context_by_agent:
            agent_specific["shared_task_context"] = shared_task_context_by_agent.get(agent_id)
        return agent_specific

    @staticmethod
    def _is_cdp_executor(executor: Any) -> bool:
        """Return True if executor exposes a shared Browser for context isolation."""
        # CdpPlaywrightExecutor: expose browser via connector.browser
        if hasattr(executor, "connector"):
            connector = getattr(executor, "connector", None)
            if hasattr(connector, "browser") and connector.browser is not None:
                return True

        # LocalPlaywrightExecutor: expose browser directly
        if hasattr(executor, "browser") and executor.browser is not None:
            return True

        return False

    @staticmethod
    async def _ensure_cdp_health(executor: Any, app_url: str, agent_id: str) -> bool:
        """Verify CDP connection is healthy before agent starts; reconnect if stale.

        Level 1 check: probes the CDP WebSocket with a lightweight call.  If
        it fails, attempts a reconnection without restarting the browser process
        or sandbox.  Returns True if the connection is (or became) healthy.
        """
        check = getattr(executor, "check_cdp_health", None)
        if not check:
            return True  # Non-CDP executor, nothing to check

        if await check():
            return True

        logger.warning("[%s] CDP connection stale — attempting reconnect", agent_id)
        reconnect = getattr(executor, "reconnect_cdp", None)
        if not reconnect:
            return False

        try:
            await reconnect()
        except Exception as exc:
            logger.error("[%s] CDP reconnection failed: %s", agent_id, exc)
            return False

        # Re-navigate to the app URL with the fresh connection
        navigate = getattr(executor, "navigate", None)
        if navigate:
            try:
                nav_result = await navigate(app_url)
                if nav_result.get("success"):
                    logger.info("[%s] CDP reconnected and re-navigated successfully", agent_id)
                    return True
                logger.warning("[%s] CDP reconnected but navigation failed: %s",
                               agent_id, nav_result.get("error"))
                # Navigation failure is non-fatal after reconnect — the
                # agent can navigate itself when it starts.  The CDP
                # connection is healthy even though the page didn't load.
            except Exception as exc:
                logger.warning("[%s] CDP reconnected but navigation raised: %s", agent_id, exc)

        return True

    def _get_total_parallel_budget(self) -> int:
        """Get task-level parallel budget from environment or config fallback."""
        if self.max_parallel is not None:
            return max(1, int(self.max_parallel))
        value = os.getenv("MAX_PARALLEL", str(self.config.max_parallel))
        try:
            budget = int(value)
        except ValueError:
            budget = self.config.max_parallel
        return max(1, budget)

    def _compute_per_task_agent_parallelism(self, agent_count: int) -> int:
        """Compute agent parallelism within one task from global budget.

        Formula: floor(total_budget / agent_count), minimum 1.
        """
        if agent_count <= 0:
            return 1
        total_budget = self._get_total_parallel_budget()
        return max(1, total_budget // agent_count)

    @staticmethod
    def _prepend_builder_prerequisite_task(
        agent_config: AgentConfig,
        planned_tasks: List[PlannedTask],
    ) -> List[PlannedTask]:
        """Ensure builder agents validate build readiness before other tasks."""
        if (agent_config.stage or "evaluate") == "build":
            return planned_tasks
        if agent_config.role != "builder":
            return planned_tasks

        dimension_ids = {dimension.id for dimension in agent_config.rubric.dimensions}
        if "build_success" not in dimension_ids:
            return planned_tasks

        normalized_tasks: List[PlannedTask] = []
        for task in planned_tasks:
            task_dict = task.to_dict()
            covers = [sid for sid in task_dict.get("covers_standard_ids", []) if sid != "build_success"]
            task_dict["covers_standard_ids"] = covers
            normalized_tasks.append(PlannedTask.from_dict(task_dict))

        prerequisite_task = PlannedTask(
            task_id="task_build_success_gate",
            title="Validate build readiness first",
            task_text=(
                "Open the running application at the provided app_url and verify the build is truly ready "
                "for deeper testing. Confirm the page renders visible content, the environment is reachable, "
                "and there are no obvious startup failure signals such as a blank page or immediate runtime errors."
            ),
            phase="interaction_visual",
            task_type="rubric_gap_fill",
            generated_from="round2_rubric_alignment",
            covers_standard_ids=["build_success"],
            rubric_gap_only=True,
            scenario_id="task_build_success_gate_s1",
            scenario_weight=1.0,
            multi_step=True,
            expected_signals=[
                "application URL responds with visible UI",
                "page is not blank after navigation",
                "no obvious startup or hydration error signal blocks testing",
            ],
            preconditions=["the builder phase has already started the app and browser"],
            based_on={"builder_prerequisite": True},
        )
        return [prerequisite_task, *normalized_tasks]

    async def _run_single_agent_cdp(
        self,
        agent_config: AgentConfig,
        app_url: str,
        shared_executor: Any,
        planning_ctx: Optional[Dict[str, Any]] = None,
        progress_callback: Optional[Callable[[str, Dict[str, Any]], Any]] = None,
        worker_executors: Optional[List[Any]] = None,
    ) -> Dict[str, Any]:
        """Run a single agent with an isolated BrowserContext forked from the shared Browser.

        When *worker_executors* is provided, each worker's browser is wrapped
        in a ``_WorkerPool`` so that subtasks are round-robined across
        independent Chromium instances, reducing CDP contention.
        """
        # Support both shared executor shapes:
        # - CdpPlaywrightExecutor via connector.browser
        # - LocalPlaywrightExecutor via browser
        if hasattr(shared_executor, "connector"):
            browser = shared_executor.connector.browser
        else:
            browser = shared_executor.browser
        context = None
        page = None
        agent_executor = None

        async def _resolve_browser():
            """Resolve the current shared Browser object, handling CDP executor shapes."""
            if hasattr(shared_executor, "connector"):
                return shared_executor.connector.browser
            return shared_executor.browser

        async def _isolated_executor_factory() -> Dict[str, Any]:
            current_browser = await _resolve_browser()
            child_context = await current_browser.new_context(
                viewport={"width": 1280, "height": 720},
                user_agent=(
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                ),
                ignore_https_errors=True,
            )
            await apply_font_load_timeout(child_context)
            child_page = await child_context.new_page()
            child_executor = _IsolatedPageExecutor(child_page, nav_url)
            await child_executor.navigate(nav_url)
            return {
                "executor": child_executor,
                "page": child_page,
                "context": child_context,
            }

        # CDP health check: if the WebSocket connection went stale (e.g. after
        # a long build phase), reconnect before forking the agent context.
        if not await self._ensure_cdp_health(shared_executor, app_url, agent_config.id):
            return self._error_result(agent_config, "CDP connection lost and recovery failed")

        # Refresh browser reference — CDP reconnection above may have replaced it.
        browser = await _resolve_browser()

        # Resolve the URL the CDP browser should navigate to.  When the
        # executor has a sandbox-provided browser URL (e.g. a sandbox-provided internal
        # address), use that to avoid TLS/proxy issues.
        nav_url = getattr(shared_executor, "browser_url", app_url)

        try:
            # Create isolated browser context
            context = await browser.new_context(
                viewport={"width": 1280, "height": 720},
                user_agent=(
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                ),
                ignore_https_errors=True,
            )
            await apply_font_load_timeout(context)
            page = await context.new_page()

            # Lightweight wrapper that delegates to the isolated page
            import functools
            agent_executor = _IsolatedPageExecutor(
                page, nav_url,
                reconnect_fn=functools.partial(self._ensure_cdp_health, shared_executor, app_url, agent_config.id),
                page_factory=_isolated_executor_factory,
            )
            await agent_executor.navigate(nav_url)

            enriched_planning_ctx = dict(planning_ctx or {})
            enriched_planning_ctx["isolated_executor_factory"] = _isolated_executor_factory
            # Pass a CDP reconnection function so the task pool can recover
            # the connection if it goes stale between tasks.
            from functools import partial as _partial
            enriched_planning_ctx["cdp_reconnect_fn"] = _partial(
                self._ensure_cdp_health,
                shared_executor,
                app_url,
                agent_config.id,
            )

            # Build per-worker isolated executor factories for multi-worker
            # subtask distribution.  Each worker has its own Chromium (and
            # optionally its own dev server), so CDP operations across
            # subtasks don't contend on a single browser process.
            if worker_executors:
                worker_factories: List[Callable[[], Awaitable[Dict[str, Any]]]] = []
                for we in worker_executors:
                    async def _make_worker_factory(we: Any) -> Callable:
                        if hasattr(we, "connector"):
                            wb = we.connector.browser
                        else:
                            wb = we.browser
                        wnav = getattr(we, "browser_url", app_url)

                        async def _wf() -> Dict[str, Any]:
                            wctx = await wb.new_context(
                                viewport={"width": 1280, "height": 720},
                                user_agent=(
                                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                                    "Chrome/120.0.0.0 Safari/537.36"
                                ),
                                ignore_https_errors=True,
                            )
                            await apply_font_load_timeout(wctx)
                            wpage = await wctx.new_page()
                            wexec = _IsolatedPageExecutor(wpage, wnav)
                            await wexec.navigate(wnav)
                            return {"executor": wexec, "page": wpage, "context": wctx}

                        return _wf

                    factory = await _make_worker_factory(we)
                    worker_factories.append(factory)

                enriched_planning_ctx["worker_pool"] = _WorkerPool(worker_factories)

            return await self._run_agent_with_tasks(
                agent_config,
                app_url,
                agent_executor,
                enriched_planning_ctx,
                progress_callback=progress_callback,
            )

        except Exception as exc:
            logger.error(f"[{agent_config.id}] CDP agent run failed: {exc}")
            return self._error_result(agent_config, str(exc))

        finally:
            if page:
                try:
                    await page.close()
                except Exception:
                    pass
            if context:
                try:
                    await context.close()
                except Exception:
                    pass

    async def _run_single_agent_sequential(
        self,
        agent_config: AgentConfig,
        app_url: str,
        executor: Any,
        planning_ctx: Optional[Dict[str, Any]] = None,
        progress_callback: Optional[Callable[[str, Dict[str, Any]], Any]] = None,
    ) -> Dict[str, Any]:
        """Run a single agent on the shared executor, resetting before each run."""
        try:
            if hasattr(executor, "reset"):
                await executor.reset(app_url)

            return await self._run_agent_with_tasks(
                agent_config,
                app_url,
                executor,
                planning_ctx or {},
                progress_callback=progress_callback,
            )

        except Exception as exc:
            logger.error(f"[{agent_config.id}] Sequential agent run failed: {exc}\n{traceback.format_exc()}")
            return self._error_result(agent_config, str(exc))

    async def _run_agent_with_tasks(
        self,
        agent_config: AgentConfig,
        app_url: str,
        executor: Any,
        planning_ctx: Dict[str, Any],
        progress_callback: Optional[Callable[[str, Dict[str, Any]], Any]] = None,
    ) -> Dict[str, Any]:
        """Run function-first planning then execute each planned task.

        Tree mode fails fast if planning produces no tasks.
        """
        user_query = planning_ctx.get("user_query") or ""
        query_decomposition = planning_ctx.get("query_decomposition") or {}
        initial_screenshot_b64 = planning_ctx.get("initial_screenshot_b64")
        page_context = planning_ctx.get("page_context")
        source_files = planning_ctx.get("source_files") or {}
        planned_tasks_override = planning_ctx.get("planned_tasks")
        planned_tasks_by_agent = planning_ctx.get("planned_tasks_by_agent") or {}
        dependency_handoffs = planning_ctx.get("dependency_handoffs") or {}
        query_specific_main_task_count = planning_ctx.get(
            "query_specific_main_task_count",
            self.config.query_specific_main_task_count,
        )
        shared_query_main_tasks = planning_ctx.get("shared_query_main_tasks")
        shared_task_context = str(planning_ctx.get("shared_task_context") or "").strip()

        if not planned_tasks_override:
            planned_tasks_override = planned_tasks_by_agent.get(agent_config.id)

        upstream_handoffs = [
            AgentHandoff(**handoff)
            for handoff in dependency_handoffs.get(agent_config.id, [])
            if isinstance(handoff, dict)
        ]
        planning_notes = format_dependency_handoffs(upstream_handoffs)
        agent_profile = {
            "agent_id": agent_config.id,
            "agent_name": agent_config.name,
            "role": agent_config.role,
            "stage": agent_config.stage,
            "description": agent_config.description,
            "system_prompt": agent_config.system_prompt,
            "allowed_tools": list(agent_config.allowed_tools),
            "subtask_decomposition_policy": agent_config.subtask_decomposition_policy,
        }

        # Build requirements list from decomposition
        requirements: List[Dict[str, Any]] = []
        for phase_key in ("behavior_requirements", "visual_requirements",
                          "source_requirements", "dom_requirements"):
            requirements.extend(query_decomposition.get(phase_key) or [])

        planned_tasks: List[PlannedTask] = list(planned_tasks_override or [])
        tasks_v1: List[PlannedTask] = []
        llm = None
        _planning_evaluator = None
        # Create LLM when planning may be needed: either for round1 or for
        # tree-mode fixed-dimension subtask synthesis.
        if user_query and not shared_query_main_tasks:
            _planning_evaluator = self._make_evaluator(agent_config)
            llm = _planning_evaluator.llm
            # Shutdown planning evaluator immediately — it was only needed to
            # extract the LLM reference. The llm variable retains a reference
            # to the ChatOpenAI instance so LLM calls continue to work.
            await _planning_evaluator.shutdown()
        # Skip round1 — query-specific main tasks are generated together with subtasks
        # in a single batched LLM call during tree synthesis.
        if not planned_tasks and user_query and not shared_query_main_tasks and llm is not None:
            logger.info("[%s] Tree mode: skipping round1; query tasks generated during subtask synthesis", agent_config.id)
            tasks_v1 = []

        planned_tasks = self._prepend_builder_prerequisite_task(agent_config, planned_tasks)
        task_tree_plan = None
        if planned_tasks or user_query:
            effective_query_count = query_specific_main_task_count or 0
            if user_query:
                # LLM-based tree synthesis runs in an isolated child process
                # so that D-state (uninterruptible NFS sleep) can be killed
                # without freezing the main event loop.  If the child is
                # killed a TimeoutError propagates and is handled as a normal
                # agent error by the caller.
                llm_config = {
                    "provider": self.config.model_provider,
                    "api_key": self.config.get_llm_api_key(),
                    "model": self.config.model_name,
                    "base_url": self.config.custom_base_url,
                }
                task_tree_plan = await run_synthesis_in_subprocess(
                    owner_agent_id=agent_config.id,
                    user_query=user_query,
                    requirements=requirements,
                    dimensions=agent_config.rubric.dimensions,
                    round1_tasks=tasks_v1,
                    aligned_tasks=planned_tasks,
                    query_specific_main_task_count=effective_query_count,
                    page_context=page_context,
                    source_files=source_files,
                    agent_profile=agent_profile,
                    shared_query_main_tasks=shared_query_main_tasks,
                    llm_config=llm_config,
                    timeout=300,
                )
                planned_tasks = build_execution_plan(task_tree_plan)
            else:
                task_tree_plan = synthesize_task_tree_from_planned_tasks(
                    planned_tasks,
                    owner_agent_id=agent_config.id,
                    query_specific_main_task_count=query_specific_main_task_count,
                )
                planned_tasks = task_tree_plan.flatten_for_execution()
        planned_task_count = len(planned_tasks)
        steps_total = agent_config.runtime.max_steps * max(1, planned_task_count)

        main_task_title_by_id = {
            main_task.main_task_id: main_task.title
            for main_task in (task_tree_plan.main_tasks if task_tree_plan is not None else [])
        }

        await self._emit_progress(
            progress_callback,
            agent_config.id,
            {
                "event": "agent_started",
                "status": "running",
                "steps_completed": 0,
                "steps_total": steps_total,
                "task_synthesis_mode": self.config.task_synthesis_mode,
                "planned_task_tree": task_tree_plan.to_dict() if task_tree_plan is not None else None,
                "current_task_index": 0,
                "current_task_total": planned_task_count,
                "current_task_title": None,
            },
        )

        if not planned_tasks:
            error_msg = "Task planning produced no executable tasks in tree mode."
            logger.error("[%s] %s", agent_config.id, error_msg)
            result = self._error_result(agent_config, error_msg)
            result["task_synthesis_mode"] = self.config.task_synthesis_mode
            result["task_tree"] = []
            runtime = result.get("runtime") if isinstance(result.get("runtime"), dict) else {}
            await self._emit_progress(
                progress_callback,
                agent_config.id,
                {
                    "event": "agent_completed",
                    "status": str(result.get("status", "completed") or "completed"),
                    "steps_completed": int(runtime.get("steps_used", 0) or 0),
                    "steps_total": steps_total,
                    "current_task_index": 0,
                    "current_task_total": 0,
                    "current_task_title": None,
                    "end_reason": str(result.get("end_reason", "") or ""),
                },
            )
            return result

        # Execute each planned task
        logger.info("[%s] Executing %d planned tasks", agent_config.id, len(planned_tasks))
        executed_tasks: List[Dict[str, Any]] = []
        task_results: List[Dict[str, Any]] = []
        trajectory_steps: List[Dict[str, Any]] = []
        dom_elements: List[Dict[str, Any]] = []
        initial_diagnostics: Dict[str, Any] = {}
        start_ts = time.monotonic()
        steps_completed_so_far = 0
        tree_subtask_parallelism = max(1, int(planning_ctx.get("tree_subtask_parallelism", 1) or 1))
        isolated_executor_factory = planning_ctx.get("isolated_executor_factory")
        cdp_reconnect_fn = planning_ctx.get("cdp_reconnect_fn")
        pool_wait_ms_total = 0

        def _aggregate_task_verdict(
            rubric_results: List[Dict[str, Any]],
            *,
            binary_only: bool = False,
        ) -> tuple[Optional[float], str]:
            applicable = [
                item for item in rubric_results
                if isinstance(item.get("score"), (int, float)) and float(item.get("weight", 0) or 0) > 0
            ]
            if applicable:
                denom = sum(float(item.get("weight", 0) or 0) for item in applicable)
                weighted_score = (
                    sum(float(item.get("score", 0) or 0) * float(item.get("weight", 0) or 0) for item in applicable) / denom
                    if denom > 0 else None
                )
            else:
                weighted_score = None

            verdicts = {str(item.get("verdict") or "").strip().lower() for item in rubric_results if item.get("verdict") is not None}
            verdicts.discard("")
            if binary_only:
                task_verdict = "passed" if verdicts == {"passed"} else "failed"
                return (1.0 if task_verdict == "passed" else 0.0), task_verdict

            if verdicts == {"passed"}:
                task_verdict = "passed"
            elif verdicts == {"failed"}:
                task_verdict = "failed"
            elif verdicts:
                task_verdict = "partial"
            elif isinstance(weighted_score, (int, float)):
                if weighted_score >= 99.5:
                    task_verdict = "passed"
                elif weighted_score <= 0.5:
                    task_verdict = "failed"
                else:
                    task_verdict = "partial"
            else:
                task_verdict = "failed"
            return (weighted_score / 100.0 if isinstance(weighted_score, (int, float)) else None), task_verdict

        async def _execute_task(
            task: PlannedTask,
            *,
            current_task_index: int,
            task_executor: Any,
            shared_task_progress: Optional[Dict[str, int]] = None,
            running_subtasks_ref: Optional[List[Dict[str, Any]]] = None,
        ) -> Dict[str, Any]:
            event_emitter = self._build_progress_emitter(
                agent_id=agent_config.id,
                steps_offset=steps_completed_so_far,
                steps_total=steps_total,
                current_task_steps_total=agent_config.runtime.max_steps,
                current_task_id=task.task_id,
                current_task_index=current_task_index,
                current_task_total=planned_task_count,
                current_task_title=task.title,
                shared_task_progress=shared_task_progress,
                running_subtasks_ref=running_subtasks_ref,
                progress_callback=progress_callback,
            )
            current_main_task_id = str((task.based_on or {}).get("parent_main_task_id") or task.parent_task_id or "")
            current_main_task_title = main_task_title_by_id.get(current_main_task_id)
            evaluator = self._make_evaluator(agent_config, event_emitter=event_emitter)
            is_tree_subtask = str((task.based_on or {}).get("task_level", "") or "") == "sub_task"
            is_build_agent = (agent_config.stage or "evaluate") == "build"

            effective_url = getattr(task_executor, "browser_url", getattr(task_executor, "app_url", app_url))
            nav_success = False
            nav_error: Optional[str] = None
            recovery_resource: Optional[Dict[str, Any]] = None
            for nav_attempt in range(1, 4):
                try:
                    nav_result = await task_executor.navigate(effective_url)
                    if nav_result.get("success"):
                        nav_success = True
                        break
                    nav_error = str(nav_result.get("error", "navigation returned success=False"))
                except Exception as exc:
                    nav_error = str(exc)
                if nav_attempt < 3:
                    await asyncio.sleep(2.0 * nav_attempt)

            # If all navigate attempts failed and we have an isolated executor
            # factory, rebuild the page/context and give it one final try.
            if not nav_success and callable(isolated_executor_factory):
                logger.warning(
                    "[%s] Navigation failed for task %s after 3 attempts (%s) — "
                    "rebuilding isolated page/context",
                    agent_config.id, task.task_id, nav_error,
                )
                factory_retried = False
                for _ in range(2):  # first try, then CDP reconnect + retry
                    try:
                        recovery_resource = await isolated_executor_factory()
                        task_executor = recovery_resource["executor"]
                        nav_result = await task_executor.navigate(effective_url)
                        if nav_result.get("success"):
                            nav_success = True
                            break
                        nav_error = str(nav_result.get("error", "recovery navigation failed"))
                    except Exception as exc:
                        nav_error = f"recovery navigation raised: {exc}"
                    if factory_retried or not callable(cdp_reconnect_fn):
                        break
                    factory_retried = True
                    logger.warning(
                        "[%s] Factory failed, attempting CDP reconnect for task %s",
                        agent_config.id, task.task_id,
                    )
                    try:
                        await cdp_reconnect_fn()
                    except Exception:
                        pass

            if not nav_success:
                logger.warning("[%s] Page reset failed before task %s: %s",
                               agent_config.id, task.task_id, nav_error)

            try:
                task_start_ts = time.monotonic()
                task_constraint_dimension_ids = _tree_subtask_constraint_dimension_ids(task)
                task_prompt = (
                    _tree_subtask_prompt(task, task_constraint_dimension_ids, agent_config=agent_config)
                    if is_tree_subtask
                    else task.task_text
                )
                if shared_task_context:
                    task_prompt = f"{shared_task_context}\n\n=== Current Task ===\n{task_prompt}"

                if is_build_agent:
                    # Build agents must use evaluate_agentic so their
                    # agent_config.system_prompt, allowed_tools, and rubric
                    # are injected — a generic "evaluator" prompt would tell
                    # the build engineer to evaluate instead of deploy.
                    agentic_result = await evaluator.evaluate_agentic(
                        agent_config=agent_config,
                        app_url=app_url,
                        executor=task_executor,
                        task_context=task_prompt,
                    )
                    # Adapt evaluate_agentic return shape to the format
                    # expected by the downstream task-result pipeline.
                    trajectory = (
                        agentic_result.get("trajectory")
                        if isinstance(agentic_result.get("trajectory"), dict)
                        else {}
                    )
                    dimensions = (
                        agentic_result.get("dimensions")
                        if isinstance(agentic_result.get("dimensions"), list)
                        else []
                    )
                    agentic_reason = str(agentic_result.get("end_reason") or "")
                    group_verdicts: list = []
                    for dim in dimensions:
                        dim_verdict = str(dim.get("verdict") or "").strip().lower()
                        group_verdicts.append({
                            "dimension_id": dim.get("dimension_id", ""),
                            "verdict": dim_verdict,
                            "passed": dim_verdict == "passed",
                            "reason": str(dim.get("reason") or ""),
                            "score": dim.get("score"),
                        })
                    # Binary verdict: only passed when both (a) the agent
                    # completed its run and (b) the build_success dimension
                    # was not explicitly reported as failed.
                    bd_success_failed = any(
                        gv.get("dimension_id") == "build_success"
                        and gv.get("verdict") == "failed"
                        for gv in group_verdicts
                    )
                    agentic_verdict = "passed" if (
                        agentic_result.get("status") == "completed"
                        and not bd_success_failed
                    ) else "failed"
                    result = {
                        "status": agentic_result.get("status", ""),
                        "verdict": {
                            "verdict": agentic_verdict,
                            "passed": agentic_verdict == "passed",
                            "reason": agentic_reason,
                            "group_verdicts": group_verdicts,
                        },
                        "group_verdicts": group_verdicts,
                        "iterations": int(agentic_result.get("runtime", {}).get("steps_used", 0) or 0),
                        "steps": trajectory.get("steps", []),
                        "dom_elements": trajectory.get("dom_elements", []),
                        "initial_diagnostics": trajectory.get("initial_diagnostics", {}),
                        "duration_ms": int(
                            trajectory.get("duration_ms")
                            or agentic_result.get("runtime", {}).get("duration_ms", 0)
                            or 0
                        ),
                    }
                else:
                    result = await evaluator.evaluate(
                        user_query=task_prompt,
                        app_url=app_url,
                        executor=task_executor,
                        standard_ids=([] if is_tree_subtask else list(task.covers_standard_ids or [])),
                        task_id=task.task_id,
                        task_title=task.title,
                    )
                verdict = result.get("verdict", {})
                group_verdicts = result.get("group_verdicts") if isinstance(result.get("group_verdicts"), list) else []
                task_steps = result.get("steps") if isinstance(result.get("steps"), list) else []
                task_dom_elements = (
                    [item for item in result.get("dom_elements", []) if isinstance(item, dict)]
                    if isinstance(result.get("dom_elements"), list)
                    else []
                )
                task_initial_diagnostics = (
                    dict(result.get("initial_diagnostics") or {})
                    if isinstance(result.get("initial_diagnostics"), dict)
                    else {}
                )
                task_duration_ms = int(result.get("duration_ms", 0) or 0) or int((time.monotonic() - task_start_ts) * 1000)
                if is_tree_subtask:
                    task_rubric_results = []
                    task_verdict = "passed" if str(verdict.get("verdict", "") or "").strip().lower() == "passed" else "failed"
                    task_completion_score = 1.0 if task_verdict == "passed" else 0.0
                else:
                    task_rubric_results = evaluator.build_dimension_results(
                        agent_config=agent_config,
                        dimension_ids=list(task.covers_standard_ids or []),
                        raw_group_verdicts=group_verdicts,
                        raw_summary_verdict=str(verdict.get("verdict", "") or "").strip().lower() or None,
                        raw_summary_reason=str(verdict.get("reason", "") or ""),
                        status=str(result.get("status", "completed") or "completed"),
                    )
                    task_completion_score, task_verdict = _aggregate_task_verdict(task_rubric_results)
                task_reason = str(verdict.get("reason", "") or "")
                if not task_reason:
                    task_reason = "; ".join(
                        str(item.get("reason", ""))
                        for item in task_rubric_results[:3]
                        if item.get("reason")
                    )
                task_result = {
                    "task_id": task.task_id,
                    "parent_task_id": task.parent_task_id,
                    "title": task.title,
                    "task_text": task.task_text,
                    "covers_standard_ids": task.covers_standard_ids,
                    "passed": task_verdict == "passed",
                    "verdict": task_verdict,
                    "reason": task_reason,
                    "steps": len(task_steps) if task_steps else int(result.get("iterations", 0) or 0),
                }
                # Reclassify timeout/infra evaluator results as inconclusive so
                # the retry loop picks them up instead of permanently marking failed.
                _eval_status = str(result.get("status", "") or "").lower()
                if _eval_status in ("timeout", "infra_error", "max_steps_exceeded", "no_verdict_submitted") and task_verdict != "passed":
                    logger.warning(
                        "[%s] Task %s evaluator returned status=%s — "
                        "reclassifying from %s to inconclusive for retry",
                        agent_config.id, task.task_id, _eval_status, task_verdict,
                    )
                    task_result["verdict"] = "inconclusive"
                    task_result["error_category"] = "infra_failure"
                    task_verdict = "inconclusive"
                    task_completion_score = 0.0
                executed_task = {
                    **task_result,
                    "task_level": str((task.based_on or {}).get("task_level", "task") or "task"),
                    "task_kind": str((task.based_on or {}).get("task_kind", "task") or "task"),
                    "status": task_result["verdict"],
                    "completion_score": task_completion_score,
                    "rubric_results": task_rubric_results,
                    "dimensions": task_rubric_results,
                    "trajectory": {
                        "status": task_result["verdict"],
                        "steps": list(task_steps),
                        "dom_elements": task_dom_elements,
                        "initial_diagnostics": task_initial_diagnostics,
                        "duration_ms": task_duration_ms,
                    },
                }
                return {
                    "event": "task_completed",
                    "task": task,
                    "task_result": task_result,
                    "executed_task": executed_task,
                    "trajectory_steps": [
                        {**step, "task_id": task.task_id, "task_title": task.title}
                        for step in task_steps
                        if isinstance(step, dict)
                    ],
                    "steps_used": int(result.get("iterations", 0) or 0),
                    "current_main_task_id": current_main_task_id,
                    "current_main_task_title": current_main_task_title,
                    "current_task_index": current_task_index,
                    "task_completion_score": task_completion_score,
                    "task_initial_diagnostics": task_initial_diagnostics,
                    "task_dom_elements": task_dom_elements,
                    "recovery_resource": recovery_resource,
                }
            except Exception as exc:
                logger.error("[%s] Task %s failed: %s", agent_config.id, task.task_id, exc)
                task_status, error_category = _classify_task_exception(exc)
                task_reason = str(exc)
                rescued_steps = list(getattr(evaluator, "_steps", []) or [])
                rescued_dom = list(getattr(evaluator, "_dom_elements", []) or [])
                rescued_diag = dict(getattr(evaluator, "_initial_diagnostics", {}) or {})
                task_result = {
                    "task_id": task.task_id,
                    "parent_task_id": task.parent_task_id,
                    "title": task.title,
                    "task_text": task.task_text,
                    "covers_standard_ids": task.covers_standard_ids,
                    "passed": False if task_status == "failed" else None,
                    "verdict": task_status,
                    "reason": task_reason,
                    "steps": len(rescued_steps),
                    "error_category": error_category,
                }
                task_rubric_results = [] if is_tree_subtask else evaluator.build_dimension_results(
                    agent_config=agent_config,
                    dimension_ids=list(task.covers_standard_ids or []),
                    raw_group_verdicts=[{
                        "reason": task_reason,
                        "error_category": error_category,
                    }],
                    raw_summary_verdict=None,
                    raw_summary_reason=task_reason,
                    status="completed",
                )
                return {
                    "event": "task_failed",
                    "task": task,
                    "task_result": task_result,
                    "executed_task": {
                        **task_result,
                        "task_level": str((task.based_on or {}).get("task_level", "task") or "task"),
                        "task_kind": str((task.based_on or {}).get("task_kind", "task") or "task"),
                        "status": task_status,
                        "completion_score": 0.0 if task_status == "failed" else None,
                        "rubric_results": task_rubric_results,
                        "dimensions": task_rubric_results,
                        "trajectory": {
                            "status": task_status,
                            "steps": rescued_steps,
                            "dom_elements": rescued_dom,
                            "initial_diagnostics": rescued_diag,
                            "duration_ms": 0,
                        },
                    },
                    "trajectory_steps": rescued_steps,
                    "steps_used": len(rescued_steps),
                    "current_main_task_id": current_main_task_id,
                    "current_main_task_title": current_main_task_title,
                    "current_task_index": current_task_index,
                    "task_completion_score": 0.0 if task_status == "failed" else None,
                    "task_initial_diagnostics": {},
                    "task_dom_elements": [],
                    "recovery_resource": recovery_resource,
                }
            finally:
                # Release LLM/HTTP resources held by this per-task evaluator
                # so the GC can collect it promptly and pooled connections
                # are not held open longer than necessary.
                await evaluator.shutdown()

        # --- Dynamic work pool (semaphore-based) ---
        # Tasks are executed concurrently with up to tree_subtask_parallelism
        # slots.  When a task finishes its slot is immediately freed for the
        # next pending task, so no slot sits idle waiting for a slow sibling.
        # If a shared semaphore was passed from the batch runner it is used
        # instead so the limit applies globally across all concurrent rows.
        pool_sem: asyncio.Semaphore = (
            planning_ctx.get("subtask_semaphore")
            or asyncio.Semaphore(tree_subtask_parallelism)
        )
        pool_completed: set[str] = set()
        pool_progress: Dict[str, int] = {}
        pool_running: List[Dict[str, Any]] = []

        # Recovery tracker: counts consecutive timeout/infra failures per task
        # so we can rebuild the browser page/context before retrying.
        _MAX_RECOVERY_ATTEMPTS = 1
        _recovery_tracker: Dict[str, int] = {}
        _TIMEOUT_FAILURE_MARKERS = ("timeout", "timed out", "infra_failure")

        async def _run_task_with_pool(
            task: PlannedTask,
            task_index: int,
        ) -> Dict[str, Any]:
            nonlocal steps_completed_so_far, pool_wait_ms_total
            # Wait for dependencies (preserved for future staged_flow support)
            dep_ids = _tree_subtask_depends_on_ids(task)
            while dep_ids and not all(dep_id in pool_completed for dep_id in dep_ids):
                await asyncio.sleep(0.1)

            subtask_wait_start = time.monotonic()
            queued_main_id = str((task.based_on or {}).get("parent_main_task_id") or task.parent_task_id or "")
            await self._emit_progress(
                progress_callback,
                agent_config.id,
                {
                    "event": "task_queued",
                    "status": "running",
                    "steps_completed": steps_completed_so_far,
                    "steps_total": steps_total,
                    "current_task_status": "queued",
                    "current_task_steps_completed": 0,
                    "current_task_steps_total": agent_config.runtime.max_steps,
                    "current_task_id": task.task_id,
                    "current_task_index": task_index,
                    "current_task_total": planned_task_count,
                    "current_task_title": task.title,
                    "current_main_task_id": queued_main_id or None,
                    "current_main_task_title": main_task_title_by_id.get(queued_main_id),
                    "current_subtask_title": task.title,
                    "queue_reason": "waiting_subtask_slot",
                },
            )
            async with pool_sem:
                pool_wait_ms_total += int((time.monotonic() - subtask_wait_start) * 1000)
                # Allocate isolated executor for this task
                task_exec = executor
                resource = None
                outcome = None
                worker_pool: Optional[_WorkerPool] = planning_ctx.get("worker_pool")

                # --- Multi-worker: round-robin across worker browsers ---
                if worker_pool is not None:
                    factory_failed = False
                    try:
                        resource = await worker_pool.get_worker()
                        task_exec = resource["executor"]
                    except Exception as exc:
                        factory_failed = True
                        logger.warning(
                            "[%s] Worker pool allocation failed for %s: %s — "
                            "falling back to primary executor",
                            agent_config.id, task.task_id, exc,
                        )
                    if factory_failed and callable(isolated_executor_factory):
                        try:
                            resource = await isolated_executor_factory()
                            task_exec = resource["executor"]
                            factory_failed = False
                        except Exception as exc:
                            logger.warning(
                                "[%s] Primary factory fallback also failed for %s: %s",
                                agent_config.id, task.task_id, exc,
                            )

                elif callable(isolated_executor_factory):
                    factory_failed = False
                    try:
                        resource = await isolated_executor_factory()
                        task_exec = resource["executor"]
                    except Exception as exc:
                        factory_failed = True
                        logger.warning(
                            "[%s] Failed to allocate isolated executor for %s: %s — "
                            "attempting CDP reconnect and retry",
                            agent_config.id, task.task_id, exc,
                        )

                    # If the factory failed (likely CDP connection stale),
                    # try CDP reconnect via the orchestrator's health check,
                    # then retry the factory once.
                    if factory_failed and callable(cdp_reconnect_fn):
                        try:
                            cdp_ok = await cdp_reconnect_fn()
                        except Exception:
                            cdp_ok = False
                        if cdp_ok:
                            try:
                                resource = await isolated_executor_factory()
                                task_exec = resource["executor"]
                                factory_failed = False
                            except Exception as exc:
                                logger.warning(
                                    "[%s] Retry still failed after CDP reconnect for %s: %s",
                                    agent_config.id, task.task_id, exc,
                                )

                try:
                    main_id = str((task.based_on or {}).get("parent_main_task_id") or task.parent_task_id or "")
                    running_entry = {
                        "task_id": task.task_id,
                        "title": task.title,
                        "main_task_id": main_id,
                        "main_task_title": main_task_title_by_id.get(main_id),
                        "status": "running",
                        "steps_completed": 0,
                        "steps_total": agent_config.runtime.max_steps,
                    }
                    pool_running.append(running_entry)
                    pool_progress[str(task.task_id)] = 0

                    await self._emit_progress(
                        progress_callback, agent_config.id,
                        {
                            "event": "task_started",
                            "status": "running",
                            "steps_completed": steps_completed_so_far,
                            "steps_total": steps_total,
                            "current_task_status": "running",
                            "current_task_steps_completed": 0,
                            "current_task_steps_total": agent_config.runtime.max_steps,
                            "current_task_id": task.task_id,
                            "current_task_index": task_index,
                            "current_task_total": planned_task_count,
                            "current_task_title": task.title,
                            "current_main_task_id": main_id or None,
                            "current_main_task_title": main_task_title_by_id.get(main_id),
                            "current_subtask_title": task.title,
                            "running_subtasks": list(pool_running),
                            "queue_reason": None,
                        },
                    )

                    outcome = await _execute_task(
                        task,
                        current_task_index=task_index,
                        task_executor=task_exec,
                        shared_task_progress=pool_progress,
                        running_subtasks_ref=pool_running,
                    )

                    # --- Recovery: detect consecutive timeout/infra failures ---
                    # If the task failed with a timeout-like error and we have an
                    # isolated executor factory, rebuild the page/context and retry
                    # once.  This prevents a stuck page from poisoning subsequent tasks.
                    task_verdict = str(outcome.get("task_result", {}).get("verdict", "")).lower()
                    error_cat = str(outcome.get("task_result", {}).get("error_category", "")).lower()
                    is_timeout_failure = any(
                        marker in task_verdict or marker in error_cat
                        for marker in _TIMEOUT_FAILURE_MARKERS
                    )
                    is_timeout_eligible = (
                        worker_pool is not None or callable(isolated_executor_factory)
                    )
                    if is_timeout_failure and is_timeout_eligible:
                        prev_count = _recovery_tracker.get(task.task_id, 0)
                        if prev_count < _MAX_RECOVERY_ATTEMPTS:
                            _recovery_tracker[task.task_id] = prev_count + 1
                            logger.warning(
                                "[%s] Recovering from timeout for task %s (attempt %d/%d) — "
                                "rebuilding page/context and retrying",
                                agent_config.id, task.task_id, prev_count + 1, _MAX_RECOVERY_ATTEMPTS,
                            )
                            # Emit recovery event
                            await self._emit_progress(
                                progress_callback, agent_config.id,
                                {
                                    "event": "task_recovery",
                                    "status": "running",
                                    "steps_completed": steps_completed_so_far,
                                    "steps_total": steps_total,
                                    "current_task_id": task.task_id,
                                    "current_task_index": task_index,
                                    "current_task_total": planned_task_count,
                                    "current_task_title": task.title,
                                    "recovery_attempt": prev_count + 1,
                                    "recovery_max": _MAX_RECOVERY_ATTEMPTS,
                                },
                            )
                            # Close the stale resource
                            if resource is not None:
                                try:
                                    await resource["page"].close()
                                except Exception:
                                    pass
                                try:
                                    await resource["context"].close()
                                except Exception:
                                    pass
                            # Allocate a fresh resource (use worker pool if available)
                            try:
                                if worker_pool is not None:
                                    resource = await worker_pool.get_worker()
                                else:
                                    resource = await isolated_executor_factory()
                                task_exec = resource["executor"]
                            except Exception as exc:
                                logger.error("[%s] Recovery allocation failed for %s: %s",
                                             agent_config.id, task.task_id, exc)
                            else:
                                # Retry the task on the fresh page/context
                                outcome = await _execute_task(
                                    task,
                                    current_task_index=task_index,
                                    task_executor=task_exec,
                                    shared_task_progress=pool_progress,
                                    running_subtasks_ref=pool_running,
                                )
                    return outcome
                finally:
                    if resource is not None:
                        try:
                            await resource["page"].close()
                        except Exception:
                            pass
                        try:
                            await resource["context"].close()
                        except Exception:
                            pass
                    # Clean up any page/context rebuilt by the task-level navigate recovery
                    recovery = outcome.get("recovery_resource") if outcome is not None else None
                    if recovery is not None:
                        try:
                            await recovery["page"].close()
                        except Exception:
                            pass
                        try:
                            await recovery["context"].close()
                        except Exception:
                            pass
                    pool_running[:] = [e for e in pool_running if e.get("task_id") != task.task_id]
                    pool_completed.add(task.task_id)
                    pool_progress.pop(str(task.task_id), None)

        pool_tasks = [
            _run_task_with_pool(task, len(task_results) + idx + 1)
            for idx, task in enumerate(planned_tasks)
        ]
        pool_outcomes = list(await asyncio.gather(*pool_tasks))

        # Pre-process: detect artifact write failures BEFORE the retry loop and
        # reclassify as inconclusive so the retry mechanism rebuilds the workspace
        # and re-runs the build engineer, instead of permanently marking as failed.
        if agent_config.role == "builder":
            for _pre_idx, _pre_outcome in enumerate(pool_outcomes):
                _pre_error = _detect_artifacts_write_failure(_pre_outcome["executed_task"])
                if _pre_error:
                    _pre_tid = _pre_outcome["task"].task_id
                    logger.warning(
                        "[%s] Pre-retry: protocol_write_artifacts failure for %s: %s "
                        "— classifying as inconclusive for retry",
                        agent_config.id, _pre_tid, _pre_error,
                    )
                    _pre_outcome["task_result"]["verdict"] = "inconclusive"
                    _pre_outcome["task_result"]["error_category"] = "infra_failure"
                    _pre_outcome["task_result"]["status"] = "infra_error"
                    _pre_outcome["executed_task"]["verdict"] = "inconclusive"
                    _pre_outcome["executed_task"]["status"] = "infra_error"

        # Retry inconclusive / infra-failed tasks (up to 3 rounds, agent-level)
        _MAX_TASK_RETRY_ROUNDS = 3
        _task_retry_count: Dict[str, int] = {}
        workspace_root = str(planning_ctx.get("workspace_root") or "").strip()
        w_artifacts_path = str(planning_ctx.get("artifacts_path") or "").strip()

        for retry_round in range(_MAX_TASK_RETRY_ROUNDS):
            retry_indices = [
                idx for idx, outcome in enumerate(pool_outcomes)
                if _is_inconclusive_infra(outcome)
                and _task_retry_count.get(planned_tasks[idx].task_id, 0) < _MAX_TASK_RETRY_ROUNDS
            ]
            if not retry_indices:
                break

            logger.warning(
                "[%s] Retry round %d/%d: %d inconclusive task(s) — %s",
                agent_config.id,
                retry_round + 1,
                _MAX_TASK_RETRY_ROUNDS,
                len(retry_indices),
                ", ".join(str(planned_tasks[i].task_id) for i in retry_indices),
            )

            # Try workspace rebuild before retrying
            if workspace_root and w_artifacts_path:
                try:
                    rebuild_fn = get_tool_function("protocol_rebuild_workspace")
                    rebuild_json = await rebuild_fn(
                        workspace_root=workspace_root,
                        session_id=w_artifacts_path,
                        artifacts_path=w_artifacts_path,
                    )
                    rebuild_result = json.loads(rebuild_json) if isinstance(rebuild_json, str) else rebuild_json
                    if rebuild_result.get("success"):
                        logger.info(
                            "[%s] Workspace rebuilt — app_url=%s cdp_url=%s",
                            agent_config.id,
                            rebuild_result.get("app_url"),
                            rebuild_result.get("cdp_url"),
                        )
                    else:
                        logger.warning(
                            "[%s] Workspace rebuild failed: %s — marking retry-exhausted",
                            agent_config.id,
                            rebuild_result.get("error", "unknown"),
                        )
                        for idx in retry_indices:
                            pool_outcomes[idx]["task_result"]["retry_exhausted"] = True
                            pool_outcomes[idx]["executed_task"]["retry_exhausted"] = True
                        break
                except Exception as rebuild_exc:
                    logger.warning("[%s] Workspace rebuild error: %s", agent_config.id, rebuild_exc)
                    for idx in retry_indices:
                        pool_outcomes[idx]["task_result"]["retry_exhausted"] = True
                        pool_outcomes[idx]["executed_task"]["retry_exhausted"] = True
                    break

            # Re-execute each inconclusive task
            for idx in retry_indices:
                task = planned_tasks[idx]
                tid = task.task_id
                _task_retry_count[tid] = _task_retry_count.get(tid, 0) + 1
                round_num = _task_retry_count[tid]

                pool_completed.discard(tid)
                await self._emit_progress(
                    progress_callback,
                    agent_config.id,
                    {
                        "event": "task_retry",
                        "status": "running",
                        "task_id": tid,
                        "retry_round": round_num,
                        "max_retries": _MAX_TASK_RETRY_ROUNDS,
                    },
                )
                try:
                    fresh_outcome = await _run_task_with_pool(task, idx + 1)
                    fresh_outcome["task_result"]["retry_count"] = round_num
                    fresh_outcome["executed_task"]["retry_count"] = round_num
                    pool_outcomes[idx] = fresh_outcome
                except Exception as retry_exc:
                    logger.error(
                        "[%s] Retry %d/%d for task %s failed: %s",
                        agent_config.id, round_num, _MAX_TASK_RETRY_ROUNDS, tid, retry_exc,
                    )
                    pool_outcomes[idx]["task_result"]["retry_count"] = round_num
                    pool_outcomes[idx]["executed_task"]["retry_count"] = round_num

        for outcome in pool_outcomes:
            task = outcome["task"]
            if not initial_diagnostics and outcome["task_initial_diagnostics"]:
                initial_diagnostics = dict(outcome["task_initial_diagnostics"])
            if not dom_elements and outcome["task_dom_elements"]:
                dom_elements = list(outcome["task_dom_elements"])
            trajectory_steps.extend(outcome["trajectory_steps"])
            # Post-process: for build agents, scan for protocol_write_artifacts
            # errors that the LLM may have ignored when issuing its verdict.
            if agent_config.role == "builder":
                artifacts_error = _detect_artifacts_write_failure(outcome["executed_task"])
                if artifacts_error:
                    logger.warning(
                        "[%s] Detected protocol_write_artifacts failure for task %s: %s",
                        agent_config.id, task.task_id, artifacts_error,
                    )
                    # Force the task and its build_success dimension to failed
                    outcome["task_result"]["passed"] = False
                    outcome["task_result"]["verdict"] = "failed"
                    outcome["task_result"]["reason"] = (
                        f"Artifacts write failed: {artifacts_error}"
                    )
                    outcome["executed_task"]["passed"] = False
                    outcome["executed_task"]["verdict"] = "failed"
                    outcome["executed_task"]["status"] = "failed"
                    outcome["executed_task"]["completion_score"] = 0.0
                    outcome["task_completion_score"] = 0.0
                    # Correct the build_success dimension in rubric_results
                    rubric = outcome["executed_task"].get("rubric_results")
                    if isinstance(rubric, list):
                        for item in rubric:
                            if isinstance(item, dict) and item.get("dimension_id") == "build_success":
                                item["verdict"] = "failed"
                                item["reason"] = (
                                    f"Artifacts write failed: {artifacts_error}"
                                )
                                item["score"] = 0.0
                                if isinstance(item.get("score_breakdown"), dict):
                                    item["score_breakdown"]["final_score"] = 0.0
                                break
                    # Also correct dimensions list if present
                    dims = outcome["executed_task"].get("dimensions")
                    if isinstance(dims, list):
                        for d in dims:
                            if isinstance(d, dict) and d.get("dimension_id") == "build_success":
                                d["verdict"] = "failed"
                                d["reason"] = (
                                    f"Artifacts write failed: {artifacts_error}"
                                )
                                d["score"] = 0.0
                                break

            task_results.append(outcome["task_result"])
            executed_tasks.append(outcome["executed_task"])
            steps_completed_so_far += int(outcome["steps_used"])
            await self._emit_progress(
                progress_callback,
                agent_config.id,
                {
                    "event": outcome["event"],
                    "status": "running",
                    "steps_completed": steps_completed_so_far,
                    "steps_total": steps_total,
                    "current_task_id": task.task_id,
                    "current_task_status": outcome["task_result"]["verdict"],
                    "current_task_completion_score": outcome["task_completion_score"],
                    "current_task_steps_completed": outcome["task_result"]["steps"],
                    "current_task_steps_total": agent_config.runtime.max_steps,
                    "current_task_index": outcome["current_task_index"],
                    "current_task_total": planned_task_count,
                    "current_task_title": task.title,
                    "current_main_task_id": outcome["current_main_task_id"] or None,
                    "current_main_task_title": outcome["current_main_task_title"],
                    "current_subtask_title": task.title,
                    "running_subtasks": [],
                    "queue_reason": None,
                },
            )

        duration_ms = int((time.monotonic() - start_ts) * 1000)

        # Compute task_completion_score
        task_completion_values = [
            float(task.get("completion_score"))
            for task in executed_tasks
            if isinstance(task.get("completion_score"), (int, float))
        ]
        if task_completion_values:
            task_completion_score = sum(task_completion_values) / len(task_completion_values)
        else:
            task_completion_score = None

        total_steps = sum(t.get("steps", 0) for t in task_results)

        task_tree_payload = (
            build_task_tree_payload(
                task_tree_plan, executed_tasks,
                aggregation_mode=self.config.aggregation_mode,
            )
            if task_tree_plan is not None
            else []
        )

        # Derive agent dimensions from main-task results (tree mode).
        if task_tree_plan is not None:
            dimensions_from_tasks = _build_dimensions_from_main_task_tree(
                agent_config=agent_config,
                task_tree_payload=task_tree_payload,
            )
        else:
            dimensions_from_tasks = []
            for dim_cfg in agent_config.rubric.dimensions:
                covering = [
                    rubric
                    for task in executed_tasks
                    if isinstance(task, dict)
                    for rubric in (task.get("rubric_results") or [])
                    if isinstance(rubric, dict) and rubric.get("dimension_id") == dim_cfg.id
                ]
                if not covering:
                    dim_verdict: Optional[str] = None
                    dim_reason = "No tasks covered this dimension."
                    dim_score = None
                    score_breakdown = {
                        "method": "task_first_aggregation",
                        "final_score": None,
                        "task_count": 0,
                        "task_scores": [],
                    }
                else:
                    verdicts = {str(item.get("verdict") or "").strip().lower() for item in covering if item.get("verdict") is not None}
                    verdicts.discard("")
                    if verdicts == {"passed"}:
                        dim_verdict = "passed"
                    elif verdicts == {"failed"}:
                        dim_verdict = "failed"
                    else:
                        dim_verdict = "partial"

                    scores = [float(item.get("score")) for item in covering if isinstance(item.get("score"), (int, float))]
                    dim_score = (sum(scores) / len(scores)) if scores else None
                    dim_reason = "; ".join(
                        str(item.get("reason", ""))
                        for item in covering[:3]
                        if item.get("reason")
                    ) or f"Aggregated from {len(covering)} task rubric results."
                    score_breakdown = {
                        "method": "task_first_aggregation",
                        "final_score": dim_score,
                        "task_count": len(covering),
                        "task_scores": [
                            {
                                "task_id": item.get("task_id"),
                                "score": item.get("score"),
                                "verdict": item.get("verdict"),
                            }
                            for item in covering
                        ],
                    }
                dimensions_from_tasks.append({
                    "dimension_id": dim_cfg.id,
                    "verdict": dim_verdict,
                    "reason": dim_reason,
                    "evidence": [],
                    "score": dim_score,
                    "score_breakdown": score_breakdown,
                    "weight": float(dim_cfg.weight),
                    "verdict_source": "task_first_aggregation",
                })

        applicable_dims = [
            item for item in dimensions_from_tasks
            if isinstance(item.get("score"), (int, float)) and float(item.get("weight", 0) or 0) > 0
        ]
        if applicable_dims:
            denom = sum(float(item.get("weight", 0) or 0) for item in applicable_dims)
            agent_score = (
                sum(float(item.get("score", 0) or 0) * float(item.get("weight", 0) or 0) for item in applicable_dims) / denom
                if denom > 0 else None
            )
        else:
            agent_score = None

        # Aggregate a top-level verdict from task results so downstream
        # verdict derivation (e.g. derive_open_report_verdict) can
        # short-circuit on build/agent failure without relying solely on
        # the per-task passed flags buried inside task_results.
        task_passed = sum(1 for t in task_results if t.get("passed") is True)
        task_failed = sum(1 for t in task_results if t.get("passed") is False)
        if task_failed and not task_passed:
            agent_verdict = {"verdict": "failed", "passed": False, "reason": f"{task_failed}/{len(task_results)} tasks failed."}
        elif task_failed:
            agent_verdict = {"verdict": "partial", "passed": False, "reason": f"{task_passed} passed, {task_failed} failed."}
        elif task_passed == len(task_results) and task_results:
            agent_verdict = {"verdict": "passed", "passed": True, "reason": f"All {len(task_results)} tasks passed."}
        else:
            agent_verdict = {"verdict": "inconclusive", "passed": False, "reason": f"0/{len(task_results)} tasks produced a definitive result."}

        agent_scoring_cfg = agent_config.scoring
        agent_max_score = agent_scoring_cfg.max_score if agent_scoring_cfg else None
        task_aggregation_mode = (
            agent_scoring_cfg.task_aggregation_mode
            if agent_scoring_cfg and agent_scoring_cfg.task_aggregation_mode
            else self.config.aggregation_mode
        )
        dimension_weights = {
            str(dim.id): float(dim.weight)
            for dim in agent_config.rubric.dimensions
        }

        result = {
            "agent_id": agent_config.id,
            "status": "completed",
            "runtime": {
                "steps_used": total_steps,
                "duration_ms": duration_ms,
                "subtask_wait_ms": pool_wait_ms_total,
            },
            "end_reason": f"Executed {len(task_results)} planned tasks.",
            "verdict": agent_verdict,
            "score": agent_score,
            "score_scale": "0-100",
            "agent_max_score": agent_max_score,
            "dimension_weights": dimension_weights,
            "planned_tasks": [t.to_dict() for t in planned_tasks],
            "planned_task_tree": task_tree_plan.to_dict() if task_tree_plan is not None else None,
            "tasks": executed_tasks,
            "task_results": task_results,
            "task_synthesis_mode": self.config.task_synthesis_mode,
            "aggregation_mode": task_aggregation_mode,
            "task_tree": task_tree_payload,
            "task_completion_score": task_completion_score,
            "dimensions": dimensions_from_tasks,
            "trajectory_ref": None,
            "trajectory": {
                "status": "completed",
                "steps": trajectory_steps,
                "dom_elements": dom_elements,
                "initial_diagnostics": initial_diagnostics,
                "duration_ms": duration_ms,
            },
        }
        await self._emit_progress(
            progress_callback,
            agent_config.id,
            {
                "event": "agent_completed",
                "status": "completed",
                "steps_completed": total_steps,
                "steps_total": steps_total,
                "current_task_index": planned_task_count,
                "current_task_total": planned_task_count,
                "current_task_title": None,
                "subtask_wait_ms": pool_wait_ms_total,
                "end_reason": f"Executed {len(task_results)} planned tasks.",
                "queue_reason": None,
            },
        )
        return result

    def _make_evaluator(
        self,
        agent_config: AgentConfig,
        event_emitter: Optional[EventEmitter] = None,
    ) -> EvaluatorRuntime:
        """Create an evaluator runtime configured for the given agent."""
        evaluator = create_evaluator_runtime(
            self.config,
            max_iterations=agent_config.runtime.max_steps,
            allowed_tools=list(agent_config.allowed_tools) if agent_config.allowed_tools else None,
            log_level=self.config.log_level,
            event_emitter=event_emitter,
        )
        evaluator._standard_scoring_requirements = evaluator._build_scoring_requirements(agent_config)
        return evaluator

    @staticmethod
    async def _emit_progress(
        progress_callback: Optional[Callable[[str, Dict[str, Any]], Any]],
        agent_id: str,
        payload: Dict[str, Any],
    ) -> None:
        if progress_callback is None:
            return
        callback_result = progress_callback(agent_id, payload)
        if inspect.isawaitable(callback_result):
            await callback_result

    def _build_progress_emitter(
        self,
        *,
        agent_id: str,
        steps_offset: int,
        steps_total: int,
        current_task_steps_total: int,
        current_task_id: Optional[str],
        current_task_index: int,
        current_task_total: int,
        current_task_title: Optional[str],
        shared_task_progress: Optional[Dict[str, int]] = None,
        running_subtasks_ref: Optional[List[Dict[str, Any]]] = None,
        progress_callback: Optional[Callable[[str, Dict[str, Any]], Any]],
    ) -> Optional[EventEmitter]:
        if progress_callback is None:
            return None

        emitter = EventEmitter(task_id=f"agent-progress:{agent_id}", enabled=True)

        async def _on_event(event: Any) -> None:
            event_data = getattr(event, "data", {})
            if not isinstance(event_data, dict):
                event_data = {}
            event_step_count = max(0, int(event_data.get("step_count", 0) or 0))
            iteration_count = max(0, int(getattr(event, "iteration", 0) or 0))
            current_task_steps_completed = max(event_step_count, iteration_count)
            if shared_task_progress is not None and current_task_id:
                shared_task_progress[current_task_id] = current_task_steps_completed
                steps_completed = steps_offset + sum(shared_task_progress.values())
            else:
                steps_completed = steps_offset + current_task_steps_completed
            running_subtasks_payload = None
            if running_subtasks_ref is not None:
                running_subtasks_payload = []
                for item in running_subtasks_ref:
                    task_id = str(item.get("task_id") or "")
                    running_subtasks_payload.append({
                        **item,
                        "steps_completed": max(0, int(shared_task_progress.get(task_id, 0) if shared_task_progress is not None and task_id else item.get("steps_completed", 0) or 0)),
                        "steps_total": max(0, int(item.get("steps_total", 0) or 0)),
                    })
            await self._emit_progress(
                progress_callback,
                agent_id,
                {
                    "event": getattr(event, "event_type", "progress"),
                    "status": "running",
                    "steps_completed": steps_completed,
                    "steps_total": steps_total,
                    "current_task_steps_completed": current_task_steps_completed,
                    "current_task_steps_total": current_task_steps_total,
                    "current_task_id": current_task_id,
                    "current_task_index": current_task_index,
                    "current_task_total": current_task_total,
                    "current_task_title": current_task_title,
                    "running_subtasks": running_subtasks_payload,
                },
            )

        emitter.on(_on_event)
        return emitter

    @staticmethod
    def _error_result(agent_config: AgentConfig, error_msg: str) -> Dict[str, Any]:
        """Build an error result dict for an agent that failed to run."""
        return normalize_agent_result({
            "agent_id": agent_config.id,
            "status": "error",
            "error": error_msg,
            "runtime": {"steps_used": 0, "duration_ms": 0},
            "end_reason": f"Agent encountered an error: {error_msg}",
            "score": None,
            "score_scale": "0-100",
            "planned_tasks": [],
            "task_results": [],
            "task_completion_score": None,
            "dimensions": [
                {
                    "dimension_id": d.id,
                    "verdict": "failed",
                    "reason": f"Agent encountered an error: {error_msg}",
                    "evidence": [],
                }
                for d in agent_config.rubric.dimensions
            ],
            "trajectory_ref": None,
            "trajectory": {
                "status": "error",
                "steps": [],
                "dom_elements": [],
                "initial_diagnostics": {},
                "duration_ms": 0,
            },
        })


class _WorkerPool:
    """Round-robin pool of isolated executor factories (one per worker browser).

    Distributes ``get_worker()`` calls across N workers so that CDP
    operations (screenshot, click, evaluate, etc.) are spread across
    independent Chromium instances instead of contending on a single
    CDP connection.
    """

    def __init__(self, factories: List[Callable[[], Awaitable[Dict[str, Any]]]]):
        self._factories = list(factories)
        self._counter = 0

    @property
    def worker_count(self) -> int:
        return len(self._factories)

    async def get_worker(self) -> Dict[str, Any]:
        """Return the next worker's isolated executor + page + context.

        Round-robins across the worker factories so that concurrent
        subtasks are spread evenly.
        """
        factory = self._factories[self._counter % len(self._factories)]
        self._counter += 1
        return await factory()

    async def get_worker_at(self, index: int) -> Dict[str, Any]:
        """Return a specific worker's executor (for retry/recovery)."""
        return await self._factories[index % len(self._factories)]()


class _IsolatedPageExecutor:
    """Thin executor wrapper around a Playwright Page for isolated contexts.

    Delegates all browser actions to the page, mimicking the interface
    expected by FrontendEvaluator / tool functions.

    When *reconnect_fn* and *page_factory* are provided, the executor can
    recover from a stale CDP connection (e.g. WebSocket timed out during
    LLM inference) by reconnecting and creating a fresh page.
    """

    def __init__(self, page: Any, app_url: str,
                 reconnect_fn: Optional[Callable[[], Awaitable[bool]]] = None,
                 page_factory: Optional[Callable[[], Awaitable[Dict[str, Any]]]] = None):
        self._page = page
        self.app_url = app_url
        self._reconnect_fn = reconnect_fn
        self._page_factory = page_factory
        self._console_messages: list = []
        self._page_errors: list = []
        self._request_failures: list = []
        self._http_errors: list = []
        self._attach_listeners()

    def _attach_listeners(self) -> None:
        def on_console(msg):
            self._console_messages.append(
                {"type": msg.type, "text": str(msg.text)[:300]}
            )

        def on_error(err):
            self._page_errors.append({"message": str(err)[:300]})

        def on_request_failed(req):
            failure = req.failure
            error_text = (
                failure.get("errorText", "unknown")
                if isinstance(failure, dict)
                else str(failure or "unknown")
            )
            self._request_failures.append(
                {"url": req.url[:500], "error": error_text[:300]}
            )

        def on_response(resp):
            if resp.status >= 400:
                self._http_errors.append(
                    {"url": resp.url[:500], "status": resp.status}
                )

        self._page.on("console", on_console)
        self._page.on("pageerror", on_error)
        self._page.on("requestfailed", on_request_failed)
        self._page.on("response", on_response)

    async def check_health(self) -> bool:
        """Check whether the underlying Page is still alive.

        Performs a lightweight evaluate call.  Returns False if the CDP
        WebSocket has been closed (e.g. after a long idle period).
        """
        try:
            await self._page.evaluate("1+1")
            return True
        except Exception:
            return False

    async def recover(self) -> bool:
        """Recover from a stale CDP connection.

        1. Reconnects the shared CDP connection via *_reconnect_fn*.
        2. Creates a fresh isolated page via *_page_factory*.
        3. Re-attaches console / error / request listeners.

        Returns True if recovery succeeded.
        """
        if not self._reconnect_fn or not self._page_factory:
            return False
        try:
            ok = await self._reconnect_fn()
            if not ok:
                return False
            result = await self._page_factory()
            new_page = result.get("page")
            if new_page is None:
                return False
            self._page = new_page
            self._console_messages = []
            self._page_errors = []
            self._request_failures = []
            self._http_errors = []
            self._attach_listeners()
            # Navigate to the app URL so subsequent tool calls work
            await self._page.goto(self.app_url, wait_until="domcontentloaded", timeout=30000)
            await self._page.wait_for_timeout(500)
            return True
        except Exception:
            return False

    async def navigate(self, url: str, timeout: int = None) -> Dict[str, Any]:
        if timeout is None:
            timeout = int(os.getenv("APP_NAVIGATION_TIMEOUT", "30")) * 1000
        last_error = None
        base_backoff_s = 2.0
        for attempt in (1, 2, 3):
            try:
                # Primary: domcontentloaded — avoids hanging on SPAs with
                # long-lived connections (HMR, polling, WebSocket).
                await self._page.goto(url, wait_until="domcontentloaded", timeout=timeout)
                await self._page.wait_for_timeout(500)

                # Best-effort app-ready check
                ready = await self._check_app_ready()
                if ready.get("success"):
                    return {"success": True, "url": self._page.url}

                last_error = ready.get("error", "app not ready")
                # App loaded but not ready — give it a short stabilisation window
                await self._page.wait_for_timeout(1500)
                ready = await self._check_app_ready()
                if ready.get("success"):
                    return {"success": True, "url": self._page.url}

            except Exception as e:
                last_error = e
                if attempt <= 2:
                    backoff = base_backoff_s * (2 ** (attempt - 1))
                    await asyncio.sleep(backoff)
                    continue
                # All 3 retries exhausted — try CDP recovery once more
                error_str = str(e).lower()
                if ("connection closed" in error_str or "target closed" in error_str) \
                   and await self.recover():
                    try:
                        await self._page.goto(url, wait_until="domcontentloaded", timeout=timeout)
                        await self._page.wait_for_timeout(500)
                        return {"success": True, "url": self._page.url}
                    except Exception as retry_e:
                        last_error = retry_e
                return {"success": False, "error": str(last_error)}

            # Fallback via networkidle for edge cases where domcontentloaded
            # returned before the app is genuinely usable.
            if attempt < 3:
                try:
                    await self._page.goto(url, wait_until="networkidle", timeout=max(timeout // 2, 10000))
                    await self._page.wait_for_timeout(500)
                    return {"success": True, "url": self._page.url}
                except Exception:
                    backoff = base_backoff_s * (2 ** (attempt - 1))
                    await asyncio.sleep(backoff)
                    continue

        return {"success": False, "error": str(last_error)}

    async def _check_app_ready(self) -> Dict[str, Any]:
        """Verify the page is in an interactive state after navigation."""
        try:
            diag = await self._page.evaluate("""() => {
                const body = document.body;
                const root = document.getElementById('root');
                const readyState = document.readyState;
                const bodyHtmlLen = body ? body.innerHTML.length : 0;
                const bodyText = (body ? body.innerText : '').trim();
                const childCount = body ? body.childElementCount : 0;
                const hasScripts = document.scripts.length > 0;
                return {
                    ready_state: readyState,
                    body_html_length: bodyHtmlLen,
                    body_text_preview: bodyText.slice(0, 200),
                    body_child_count: childCount,
                    has_scripts: hasScripts,
                };
            }""")
        except Exception as exc:
            return {"success": False, "error": f"app-ready eval failed: {exc}"}

        if diag.get("ready_state") == "loading":
            return {"success": False, "error": "page still loading"}

        body_html = int(diag.get("body_html_length", 0) or 0)
        body_children = int(diag.get("body_child_count", 0) or 0)
        if body_html < 50 and body_children == 0:
            return {"success": False, "error": f"body appears empty (html_len={body_html}, children={body_children})"}

        # Check for critical resource failures on first load
        critical_failures = [f for f in self._request_failures
                             if f.get("url", "").endswith((".js", ".css"))]
        if critical_failures:
            return {"success": False, "error": f"critical resource failures: {critical_failures[0]}"}

        return {"success": True}

    async def click(self, selector: str, timeout: int = 5000) -> Dict[str, Any]:
        await self._page.click(selector, timeout=timeout)
        await self._page.wait_for_timeout(350)
        return {"success": True, "message": f"Clicked: {selector}"}

    async def click_at(self, x: int, y: int, timeout: int = 5000) -> Dict[str, Any]:
        await self._page.mouse.click(int(x), int(y))
        await self._page.wait_for_timeout(350)
        return {"success": True, "message": f"Clicked at: ({int(x)}, {int(y)})"}

    async def hover(self, selector: str, timeout: int = 5000) -> Dict[str, Any]:
        await self._page.hover(selector, timeout=timeout)
        await self._page.wait_for_timeout(300)
        return {"success": True, "message": f"Hovered: {selector}"}

    async def type_text(self, selector: str, text: str, timeout: int = 5000) -> Dict[str, Any]:
        await self._page.fill(selector, text, timeout=timeout)
        await self._page.wait_for_timeout(150)
        return {"success": True, "message": f"Typed into: {selector}"}

    async def get_context(self) -> Dict[str, Any]:
        await self._page.wait_for_timeout(100)
        snapshot = await self._page.accessibility.snapshot()
        diagnostics = await self._collect_diagnostics()
        return {
            "success": True,
            "title": await self._page.title(),
            "url": self._page.url,
            "accessibility_tree": snapshot,
            "diagnostics": diagnostics,
        }

    async def evaluate_js(self, expression: str) -> Dict[str, Any]:
        result = await self._page.evaluate(expression)
        return {"success": True, "result": result}

    async def keyboard_down(self, key: str) -> Dict[str, Any]:
        await self._page.keyboard.down(key)
        return {"success": True, "message": f"Key down: {key}"}

    async def keyboard_up(self, key: str) -> Dict[str, Any]:
        await self._page.keyboard.up(key)
        return {"success": True, "message": f"Key up: {key}"}

    async def keyboard_press(self, key: str, duration: int = 0) -> Dict[str, Any]:
        if duration > 0:
            await self._page.keyboard.down(key)
            await self._page.wait_for_timeout(duration)
            await self._page.keyboard.up(key)
        else:
            await self._page.keyboard.press(key)
        return {"success": True, "message": f"Key press: {key}" + (f" (hold {duration}ms)" if duration > 0 else "")}

    async def screenshot(self, full_page: bool = False) -> str:
        import base64
        data = await self._page.screenshot(type="png", full_page=full_page)
        return base64.b64encode(data).decode()

    async def _collect_diagnostics(self) -> Dict[str, Any]:
        try:
            diag = await self._page.evaluate(
                """() => {
                    const root = document.getElementById('root');
                    const scripts = Array.from(document.scripts)
                        .map(s => s.src).filter(Boolean);
                    return {
                        ready_state: document.readyState,
                        root_exists: !!root,
                        root_html_length: root ? root.innerHTML.length : 0,
                        body_html_length: document.body ? document.body.innerHTML.length : 0,
                        body_text_preview: (document.body?.innerText || '').trim().slice(0, 200),
                        script_sources: scripts.slice(0, 10),
                    };
                }"""
            )
        except Exception:
            diag = {}
        diag["console_messages"] = list(self._console_messages)
        diag["page_errors"] = list(self._page_errors)
        diag["request_failures"] = list(self._request_failures)
        diag["http_errors"] = list(self._http_errors)
        return diag
