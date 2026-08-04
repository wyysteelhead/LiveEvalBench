"""Report aggregation for agentic evaluation runs."""

from datetime import datetime, timezone
import uuid
from typing import Any, Dict, List, Optional


INCONCLUSIVE_TASK_STATUSES = {"inconclusive", "infra_failed", "tool_error", "provider_error"}
_DEFAULT_AGENT_MAX_SCORE = 20.0  # overridable per agent via scoring.max_score


def _normalize_runtime(runtime: Any) -> Dict[str, int]:
    payload = runtime if isinstance(runtime, dict) else {}
    return {
        "steps_used": int(payload.get("steps_used", 0) or 0),
        "duration_ms": int(payload.get("duration_ms", 0) or 0),
    }


def _normalize_dimensions(dimensions: Any) -> List[Dict[str, Any]]:
    if not isinstance(dimensions, list):
        return []

    normalized: List[Dict[str, Any]] = []
    for raw in dimensions:
        if not isinstance(raw, dict):
            continue
        dimension = dict(raw)
        evidence = dimension.get("evidence")
        dimension["evidence"] = evidence if isinstance(evidence, list) else []
        dimension["dimension_id"] = str(dimension.get("dimension_id") or "dimension")
        verdict = dimension.get("verdict")
        dimension["verdict"] = None if verdict is None else str(verdict)
        dimension["reason"] = str(dimension.get("reason") or "")
        score = dimension.get("score")
        dimension["score"] = float(score) if isinstance(score, (int, float)) else None
        weight = dimension.get("weight")
        dimension["weight"] = float(weight) if isinstance(weight, (int, float)) else None
        normalized.append(dimension)
    return normalized


def _normalize_trajectory(trajectory: Any, *, status: str, runtime: Dict[str, int]) -> Dict[str, Any]:
    payload = trajectory if isinstance(trajectory, dict) else {}
    steps = payload.get("steps") if isinstance(payload.get("steps"), list) else []
    dom_elements = payload.get("dom_elements") if isinstance(payload.get("dom_elements"), list) else []
    initial_diagnostics = (
        payload.get("initial_diagnostics")
        if isinstance(payload.get("initial_diagnostics"), dict) else {}
    )
    return {
        "status": str(payload.get("status") or status),
        "steps": [step for step in steps if isinstance(step, dict)],
        "dom_elements": dom_elements,
        "initial_diagnostics": initial_diagnostics,
        "duration_ms": int(payload.get("duration_ms", runtime["duration_ms"]) or runtime["duration_ms"] or 0),
    }


def _normalize_tasks(tasks: Any) -> List[Dict[str, Any]]:
    if not isinstance(tasks, list):
        return []

    normalized: List[Dict[str, Any]] = []
    for raw in tasks:
        if not isinstance(raw, dict):
            continue
        task = dict(raw)
        passed = task.get("passed")
        task_status = str(
            task.get("status")
            or task.get("verdict")
            or ("passed" if passed is True else "failed" if passed is False else "unknown")
        ).strip().lower()
        trajectory = _normalize_trajectory(
            task.get("trajectory"),
            status=task_status,
            runtime={"steps_used": int(task.get("steps", 0) or 0), "duration_ms": 0},
        )
        steps_count = int(task.get("steps", 0) or 0) or len(trajectory["steps"])
        completion_score = task.get("completion_score")
        if not isinstance(completion_score, (int, float)):
            if passed is True:
                completion_score = 1.0
            elif passed is False:
                completion_score = 0.0
            else:
                completion_score = None
        normalized_passed: Optional[bool]
        if task_status in INCONCLUSIVE_TASK_STATUSES:
            normalized_passed = None
            completion_score = None
        else:
            normalized_passed = passed if isinstance(passed, bool) else None
        normalized.append({
            **task,
            "task_id": str(task.get("task_id") or task.get("id") or "task"),
            "parent_task_id": (
                str(task.get("parent_task_id"))
                if task.get("parent_task_id") is not None
                else None
            ),
            "title": str(task.get("title") or task.get("task_title") or ""),
            "task_text": str(task.get("task_text") or task.get("prompt") or ""),
            "covers_standard_ids": [
                str(item) for item in (task.get("covers_standard_ids") or []) if item is not None
            ],
            "passed": normalized_passed,
            "verdict": str(task.get("verdict") or task_status).strip().lower(),
            "reason": str(task.get("reason") or ""),
            "status": task_status,
            "steps": steps_count,
            "completion_score": float(completion_score) if isinstance(completion_score, (int, float)) else None,
            "rubric_results": _normalize_dimensions(task.get("rubric_results") or task.get("dimensions")),
            "dimensions": _normalize_dimensions(task.get("dimensions")),
            "trajectory": trajectory,
        })
    return normalized


def _normalize_task_tree(task_tree: Any) -> List[Dict[str, Any]]:
    if not isinstance(task_tree, list):
        return []

    normalized: List[Dict[str, Any]] = []
    for raw in task_tree:
        if not isinstance(raw, dict):
            continue
        node = dict(raw)
        node["main_task_id"] = str(node.get("main_task_id") or "main-task")
        node["title"] = str(node.get("title") or "")
        node["goal"] = str(node.get("goal") or "")
        node["origin"] = str(node.get("origin") or "query_specific")
        node["status"] = str(node.get("status") or node.get("verdict") or "pending")
        node["verdict"] = str(node.get("verdict") or node.get("status") or "pending")
        node["dimension_ids"] = [
            str(item) for item in (node.get("dimension_ids") or []) if item is not None
        ]
        node["dimension_coverage"] = [
            str(item) for item in (node.get("dimension_coverage") or []) if item is not None
        ]
        raw_subtasks = []
        for raw_subtask in (node.get("subtasks") or []):
            if not isinstance(raw_subtask, dict):
                continue
            normalized_subtask = dict(raw_subtask)
            if "task_id" not in normalized_subtask and normalized_subtask.get("sub_task_id") is not None:
                normalized_subtask["task_id"] = normalized_subtask.get("sub_task_id")
            if (
                "parent_task_id" not in normalized_subtask
                and normalized_subtask.get("parent_main_task_id") is not None
            ):
                normalized_subtask["parent_task_id"] = normalized_subtask.get("parent_main_task_id")
            raw_subtasks.append(normalized_subtask)
        node["subtasks"] = _normalize_tasks(raw_subtasks)
        for subtask in node["subtasks"]:
            subtask_status = str(subtask.get("status") or subtask.get("verdict") or "pending").strip().lower()
            if subtask_status == "passed":
                subtask["passed"] = True
                subtask["verdict"] = "passed"
                subtask["status"] = "passed"
                subtask["completion_score"] = 1.0
            elif subtask_status in {"pending", "running"}:
                subtask["completion_score"] = 0.0
            elif subtask_status in INCONCLUSIVE_TASK_STATUSES:
                subtask["passed"] = None
                subtask["verdict"] = subtask_status
                subtask["status"] = subtask_status
                subtask["completion_score"] = None
            else:
                subtask["passed"] = False
                subtask["verdict"] = "failed"
                subtask["status"] = "failed"
                subtask["completion_score"] = 0.0
        normalized.append(node)
    return normalized


def _bare_dimension_id_from_main_task_id(main_task_id: Any) -> str:
    """Strip the ``dimension::`` / ``main::task_dimension_`` prefix.

    Mirrors ``scripts/reweight_report.py::_bare_dim_id`` so runtime scoring
    and post-hoc reweighting resolve the same dimension key.
    """
    text = str(main_task_id or "")
    if "::" in text:
        text = text.split("::", 1)[1]
    if text.startswith("task_dimension_"):
        text = text[len("task_dimension_") :]
    return text


def _resolve_main_task_weight_share(
    main_task: Dict[str, Any],
    *,
    dimension_weights: Dict[str, float],
    per_unspec_share: float,
    per_query_share: float,
) -> float:
    """Pick the per-main-task weight share (fraction of agent_max_score).

    Mirrors ``scripts/reweight_report.py``: each fixed_dimension main task
    contributes the rubric weight of its dimension(s); a dimension carrying
    the sentinel ``weight == -1`` contributes ``per_unspec_share`` instead,
    meaning it auto-shares the residual ``1 - sum(explicit_weights)`` evenly
    with any other ``-1`` dimensions and the query-specific task pool.
    Query-specific tasks each contribute ``per_query_share``. Unknown
    origins map to 0.
    """
    origin = str(main_task.get("origin") or "").strip().lower()
    if origin == "fixed_dimension":
        dim_ids: List[str] = []
        for raw in main_task.get("dimension_ids") or []:
            text = str(raw).strip()
            if text:
                dim_ids.append(text)
        if not dim_ids:
            bare = _bare_dimension_id_from_main_task_id(main_task.get("main_task_id"))
            if bare:
                dim_ids.append(bare)
        total = 0.0
        for did in dim_ids:
            w = float(dimension_weights.get(did, 0.0))
            if w == -1:
                total += per_unspec_share
            elif w > 0:
                total += w
        return total
    if origin == "query_specific":
        return per_query_share
    return 0.0


def _compute_main_task_metrics(
    task_tree: List[Dict[str, Any]],
    *,
    aggregation_mode: str = "strict",
    agent_max_score: float = _DEFAULT_AGENT_MAX_SCORE,
    dimension_weights: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    status_counts: Dict[str, int] = {}
    passed = 0
    evaluated = 0
    completion_scores: List[float] = []
    uniform_weight = (agent_max_score / len(task_tree)) if task_tree else None

    per_query_share = 0.0
    per_unspec_share = 0.0
    if dimension_weights is not None and task_tree:
        explicit_share = sum(
            float(w) for w in dimension_weights.values()
            if float(w) != -1 and float(w) > 0
        )
        unspec_dim_ids = {
            str(did) for did, w in dimension_weights.items()
            if float(w) == -1
        }
        unspec_dim_appearances = 0
        query_count = 0
        for main_task in task_tree:
            origin = str(main_task.get("origin") or "").strip().lower()
            if origin == "fixed_dimension":
                dim_ids = [str(item).strip() for item in (main_task.get("dimension_ids") or []) if str(item).strip()]
                if not dim_ids:
                    bare = _bare_dimension_id_from_main_task_id(main_task.get("main_task_id"))
                    if bare:
                        dim_ids = [bare]
                for did in dim_ids:
                    if did in unspec_dim_ids:
                        unspec_dim_appearances += 1
            elif origin == "query_specific":
                query_count += 1
        unspec_units = unspec_dim_appearances + (1 if query_count > 0 else 0)
        remaining = max(0.0, 1.0 - explicit_share)
        per_unspec_share = (remaining / unspec_units) if unspec_units > 0 else 0.0
        per_query_share = (per_unspec_share / query_count) if query_count > 0 else 0.0

    main_task_score_total = 0.0

    for main_task in task_tree:
        subtasks = main_task.get("subtasks") if isinstance(main_task.get("subtasks"), list) else []
        if subtasks:
            subtask_statuses = [str(subtask.get("status") or subtask.get("verdict") or "pending").lower() for subtask in subtasks]
            passed_subtasks = sum(1 for status in subtask_statuses if status == "passed")
            failed_subtasks = sum(1 for status in subtask_statuses if status == "failed")
            inconclusive_subtasks = sum(1 for status in subtask_statuses if status in INCONCLUSIVE_TASK_STATUSES)
            total_subtasks = len(subtasks)
            main_task["subtasks_total"] = total_subtasks
            main_task["subtasks_passed"] = passed_subtasks
            main_task["subtasks_inconclusive"] = inconclusive_subtasks
            conclusive_subtasks = passed_subtasks + failed_subtasks
            if aggregation_mode == "strict" and conclusive_subtasks:
                main_task["completion_score"] = 0.0 if failed_subtasks > 0 else 1.0
            else:
                main_task["completion_score"] = (
                    passed_subtasks / conclusive_subtasks if conclusive_subtasks else None
                )
            if total_subtasks and passed_subtasks == total_subtasks:
                status = "passed"
            elif any(status == "running" for status in subtask_statuses):
                status = "running"
            elif all(status == "pending" for status in subtask_statuses):
                status = "pending"
            elif failed_subtasks > 0:
                status = "partial" if (passed_subtasks > 0 or inconclusive_subtasks > 0) else "failed"
            elif passed_subtasks > 0 and inconclusive_subtasks > 0:
                status = "partial"
            elif inconclusive_subtasks > 0:
                status = "inconclusive"
            elif passed_subtasks > 0:
                status = "partial"
            else:
                status = "failed"
            main_task["status"] = status
            main_task["verdict"] = status
        else:
            status = str(main_task.get("status") or main_task.get("verdict") or "pending").lower()

        completion_score = main_task.get("completion_score")
        pass_rate = float(completion_score) if isinstance(completion_score, (int, float)) else None
        main_task["pass_rate"] = pass_rate
        if dimension_weights is not None and agent_max_score:
            share = _resolve_main_task_weight_share(
                main_task,
                dimension_weights=dimension_weights,
                per_unspec_share=per_unspec_share,
                per_query_share=per_query_share,
            )
            main_task_weight_value: Optional[float] = float(share) * float(agent_max_score)
        else:
            main_task_weight_value = uniform_weight
        main_task["task_weight"] = main_task_weight_value
        main_task["max_score"] = main_task_weight_value
        main_task["score"] = (
            float(main_task_weight_value) * pass_rate
            if main_task_weight_value is not None and pass_rate is not None
            else 0.0 if main_task_weight_value is not None
            else None
        )
        if isinstance(main_task.get("score"), (int, float)):
            main_task_score_total += float(main_task["score"])

        status_counts[status] = status_counts.get(status, 0) + 1
        if status in {"passed", "failed", "partial"}:
            evaluated += 1
            if status == "passed":
                passed += 1
        if isinstance(completion_score, (int, float)):
            completion_scores.append(float(completion_score))

    return {
        "main_tasks_total": len(task_tree),
        "main_tasks_evaluated": evaluated,
        "main_tasks_passed": passed,
        "main_task_status_counts": status_counts,
        "main_task_pass_rate": (passed / evaluated) if evaluated else None,
        "main_task_completion_score": (
            sum(completion_scores) / len(completion_scores)
            if completion_scores else None
        ),
        "main_task_weight": (
            None if dimension_weights is not None else uniform_weight
        ),
        "main_task_score_total": main_task_score_total if task_tree else None,
        "agent_score": main_task_score_total if task_tree else None,
        "agent_max_score": agent_max_score if task_tree else None,
    }


def _derive_task_results_from_tasks(tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        {
            "task_id": task["task_id"],
            "title": task.get("title", ""),
            "task_text": task.get("task_text", ""),
            "covers_standard_ids": task.get("covers_standard_ids", []),
            "passed": task.get("passed"),
            "verdict": task.get("verdict"),
            "reason": task.get("reason", ""),
            "steps": task.get("steps", 0),
        }
        for task in tasks
    ]


def _derive_dimensions_from_tasks(tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_dimension: Dict[str, Dict[str, Any]] = {}
    for task in tasks:
        for dim in task.get("rubric_results", []) or task.get("dimensions", []) or []:
            dim_id = str(dim.get("dimension_id") or "dimension")
            bucket = by_dimension.setdefault(
                dim_id,
                {
                    "dimension_id": dim_id,
                    "reason_parts": [],
                    "evidence": [],
                    "scores": [],
                    "weights": [],
                    "task_ids": [],
                    "verdicts": [],
                },
            )
            if dim.get("reason"):
                bucket["reason_parts"].append(str(dim.get("reason")))
            if isinstance(dim.get("score"), (int, float)):
                bucket["scores"].append(float(dim.get("score")))
            if isinstance(dim.get("weight"), (int, float)):
                bucket["weights"].append(float(dim.get("weight")))
            if task.get("task_id"):
                bucket["task_ids"].append(str(task.get("task_id")))
            if dim.get("verdict") is not None:
                bucket["verdicts"].append(str(dim.get("verdict")))

    normalized: List[Dict[str, Any]] = []
    for dim_id, bucket in by_dimension.items():
        verdicts = {value.lower() for value in bucket["verdicts"]}
        if not verdicts:
            verdict: Optional[str] = None
        elif verdicts == {"passed"}:
            verdict = "passed"
        elif verdicts == {"failed"}:
            verdict = "failed"
        elif verdicts == {"inconclusive"}:
            verdict = "inconclusive"
        else:
            verdict = "partial"
        scores = bucket["scores"]
        normalized.append({
            "dimension_id": dim_id,
            "verdict": verdict,
            "reason": "; ".join(bucket["reason_parts"][:3]),
            "evidence": bucket["evidence"],
            "score": (sum(scores) / len(scores)) if scores else None,
            "weight": (bucket["weights"][0] if bucket["weights"] else None),
            "score_breakdown": {
                "method": "task_first_aggregation",
                "final_score": (sum(scores) / len(scores)) if scores else None,
                "task_count": len(bucket["task_ids"]),
                "task_scores": [
                    {"task_id": task_id}
                    for task_id in bucket["task_ids"]
                ],
            },
            "verdict_source": "task_first_aggregation",
            "covered_task_ids": bucket["task_ids"],
        })
    return normalized


def _derive_dimensions_from_task_tree(task_tree: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_dimension: Dict[str, Dict[str, Any]] = {}
    for main_task in task_tree:
        if not isinstance(main_task, dict):
            continue
        for dim_id in (main_task.get("dimension_ids") or []):
            dim_key = str(dim_id or "dimension")
            bucket = by_dimension.setdefault(
                dim_key,
                {
                    "dimension_id": dim_key,
                    "reason_parts": [],
                    "scores": [],
                    "main_task_ids": [],
                    "verdicts": [],
                },
            )
            main_status = str(main_task.get("status") or main_task.get("verdict") or "").lower().strip()
            if main_status:
                bucket["verdicts"].append(main_status)
            completion_score = main_task.get("completion_score")
            if isinstance(completion_score, (int, float)):
                bucket["scores"].append(float(completion_score) * 100.0)
            main_task_id = main_task.get("main_task_id")
            if main_task_id:
                bucket["main_task_ids"].append(str(main_task_id))
            failure_stage = str(main_task.get("failure_stage") or "").strip()
            if failure_stage:
                bucket["reason_parts"].append(f"Failed at {failure_stage}")
            else:
                key_findings = main_task.get("key_findings") or []
                if key_findings:
                    bucket["reason_parts"].append(str(key_findings[0]))

    normalized: List[Dict[str, Any]] = []
    for dim_id, bucket in by_dimension.items():
        verdicts = set(bucket["verdicts"])
        if not verdicts:
            verdict: Optional[str] = None
        elif verdicts == {"passed"}:
            verdict = "passed"
        elif verdicts == {"failed"}:
            verdict = "failed"
        elif verdicts == {"inconclusive"}:
            verdict = "inconclusive"
        else:
            verdict = "partial"
        scores = bucket["scores"]
        normalized.append({
            "dimension_id": dim_id,
            "verdict": verdict,
            "reason": "; ".join(bucket["reason_parts"][:3]),
            "evidence": [],
            "score": (sum(scores) / len(scores)) if scores else None,
            "weight": None,
            "score_breakdown": {
                "method": "main_task_aggregation",
                "final_score": (sum(scores) / len(scores)) if scores else None,
                "main_task_count": len(bucket["main_task_ids"]),
                "main_task_scores": [
                    {"main_task_id": main_task_id}
                    for main_task_id in bucket["main_task_ids"]
                ],
            },
            "verdict_source": "main_task_aggregation",
            "covered_main_task_ids": bucket["main_task_ids"],
        })
    return normalized


def _derive_trajectory_from_tasks(tasks: List[Dict[str, Any]], *, status: str, runtime: Dict[str, int]) -> Dict[str, Any]:
    steps: List[Dict[str, Any]] = []
    dom_elements: List[Dict[str, Any]] = []
    initial_diagnostics: Dict[str, Any] = {}
    duration_ms = 0
    for task in tasks:
        trajectory = task.get("trajectory") if isinstance(task.get("trajectory"), dict) else {}
        steps.extend([step for step in trajectory.get("steps", []) if isinstance(step, dict)])
        if not dom_elements and isinstance(trajectory.get("dom_elements"), list):
            dom_elements = [item for item in trajectory.get("dom_elements", []) if isinstance(item, dict)]
        if not initial_diagnostics and isinstance(trajectory.get("initial_diagnostics"), dict):
            initial_diagnostics = dict(trajectory.get("initial_diagnostics") or {})
        duration_ms += int(trajectory.get("duration_ms", 0) or 0)
    return {
        "status": status,
        "steps": steps,
        "dom_elements": dom_elements,
        "initial_diagnostics": initial_diagnostics,
        "duration_ms": duration_ms or runtime["duration_ms"],
    }


def normalize_agent_result(agent_result: Dict[str, Any]) -> Dict[str, Any]:
    payload = dict(agent_result)
    runtime = _normalize_runtime(payload.get("runtime"))
    status = str(payload.get("status") or "unknown")
    score = payload.get("score")
    task_completion_score = payload.get("task_completion_score")
    trajectory_ref = payload.get("trajectory_ref")
    tasks = _normalize_tasks(payload.get("tasks"))
    task_results = payload.get("task_results") if isinstance(payload.get("task_results"), list) else []
    if not task_results and tasks:
        task_results = _derive_task_results_from_tasks(tasks)
    dimensions = _normalize_dimensions(payload.get("dimensions"))
    if not dimensions and tasks:
        dimensions = _derive_dimensions_from_tasks(tasks)
    trajectory = payload.get("trajectory")
    if not isinstance(trajectory, dict) and tasks:
        trajectory = _derive_trajectory_from_tasks(tasks, status=status, runtime=runtime)
    task_tree = _normalize_task_tree(payload.get("task_tree"))

    legacy_score = float(score) if isinstance(score, (int, float)) else None
    legacy_task_completion_score = (
        float(task_completion_score) if isinstance(task_completion_score, (int, float)) else None
    )

    agent_failed = status not in ("completed", "running", "pending", "unknown")

    if not isinstance(task_completion_score, (int, float)) and tasks and not agent_failed:
        completions = [task.get("completion_score") for task in tasks if isinstance(task.get("completion_score"), (int, float))]
        task_completion_score = (sum(completions) / len(completions)) if completions else None
    normalized_task_completion_score = (
        float(task_completion_score) if isinstance(task_completion_score, (int, float)) and not agent_failed else None
    )

    aggregation_mode = str(payload.get("aggregation_mode") or "").strip().lower() or "strict"
    agent_max_score_raw = payload.get("agent_max_score")
    agent_max_score = float(agent_max_score_raw) if isinstance(agent_max_score_raw, (int, float)) else _DEFAULT_AGENT_MAX_SCORE

    dimension_weights_raw = payload.get("dimension_weights")
    dimension_weights: Optional[Dict[str, float]] = None
    if isinstance(dimension_weights_raw, dict) and dimension_weights_raw:
        parsed: Dict[str, float] = {}
        for key, value in dimension_weights_raw.items():
            try:
                parsed[str(key)] = float(value)
            except (TypeError, ValueError):
                continue
        if parsed:
            dimension_weights = parsed

    main_task_metrics = _compute_main_task_metrics(
        task_tree,
        aggregation_mode=aggregation_mode,
        agent_max_score=agent_max_score,
        dimension_weights=dimension_weights,
    )
    if not dimensions and task_tree:
        dimensions = _derive_dimensions_from_task_tree(task_tree)
    main_task_pass_rate = main_task_metrics["main_task_pass_rate"]
    main_task_completion_score = main_task_metrics["main_task_completion_score"]
    if main_task_completion_score is not None and not agent_failed:
        normalized_task_completion_score = main_task_completion_score

    normalized_score = legacy_score
    score_max = None
    score_scale = str(payload.get("score_scale") or "0-100")
    if agent_failed:
        normalized_score = None
        score_max = None
        score_scale = "0-100"
    elif main_task_completion_score is not None:
        normalized_score = main_task_metrics["agent_score"]
        score_max = main_task_metrics["agent_max_score"]
        score_scale = "0-20"

    normalized = dict(payload)
    normalized.update({
        "agent_id": str(payload.get("agent_id") or payload.get("role") or ""),
        "status": status,
        "runtime": runtime,
        "end_reason": str(payload.get("end_reason") or payload.get("error") or ""),
        "score": normalized_score,
        "score_scale": score_scale,
        "score_max": score_max,
        "planned_tasks": payload.get("planned_tasks") if isinstance(payload.get("planned_tasks"), list) else [],
        "tasks": tasks,
        "task_results": task_results,
        "task_synthesis_mode": str(payload.get("task_synthesis_mode") or "flat"),
        "task_tree": task_tree,
        "task_completion_score": normalized_task_completion_score,
        "legacy_task_completion_score": legacy_task_completion_score,
        "main_task_pass_rate": main_task_pass_rate,
        "main_task_completion_score": main_task_completion_score,
        "main_task_weight": main_task_metrics["main_task_weight"],
        "main_task_score_total": main_task_metrics["main_task_score_total"],
        "main_tasks_total": main_task_metrics["main_tasks_total"],
        "main_tasks_evaluated": main_task_metrics["main_tasks_evaluated"],
        "main_tasks_passed": main_task_metrics["main_tasks_passed"],
        "main_task_status_counts": main_task_metrics["main_task_status_counts"],
        "legacy_score": legacy_score,
        "dimensions": dimensions,
        "trajectory_ref": str(trajectory_ref).strip() if trajectory_ref else None,
        "trajectory": _normalize_trajectory(trajectory, status=status, runtime=runtime),
    })
    if "path_evidence" in payload:
        normalized["path_evidence"] = [
            str(item) for item in payload.get("path_evidence", []) if item is not None
        ]
    return normalized


def build_agentic_report(
    agent_results: List[Dict[str, Any]],
    task_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Aggregate per-agent results into a task-level report.

    Args:
        agent_results: List of result dicts returned by evaluate_agentic().
        task_id: Optional stable task identifier; generated if not provided.

    Returns:
        Task-level report dict.
    """
    normalized_results = [normalize_agent_result(agent) for agent in agent_results if isinstance(agent, dict)]

    status_counts: Dict[str, int] = {}
    dimension_counts = {"passed": 0, "partial": 0, "failed": 0, "inconclusive": 0, "not_applicable": 0, "scored_only": 0}
    total_dimensions = 0
    total_steps = 0
    total_duration_ms = 0
    scored_agents = 0
    total_agent_score = 0.0
    scored_dimensions = 0
    total_dimension_score = 0.0
    main_task_status_counts: Dict[str, int] = {}
    total_main_tasks = 0
    total_main_tasks_evaluated = 0
    total_main_tasks_passed = 0
    total_max_score = 0.0

    for agent in normalized_results:
        status = str(agent.get("status", "unknown"))
        status_counts[status] = status_counts.get(status, 0) + 1
        runtime = agent.get("runtime", {}) or {}
        total_steps += int(runtime.get("steps_used", 0) or 0)
        total_duration_ms += int(runtime.get("duration_ms", 0) or 0)

        agent_ok = status == "completed"

        agent_score = agent.get("score")
        if isinstance(agent_score, (int, float)) and agent_ok:
            scored_agents += 1
            total_agent_score += float(agent_score)
        agent_score_max = agent.get("score_max")
        if isinstance(agent_score_max, (int, float)) and agent_ok:
            total_max_score += float(agent_score_max)

        if agent_ok:
            total_main_tasks += int(agent.get("main_tasks_total", 0) or 0)
            total_main_tasks_evaluated += int(agent.get("main_tasks_evaluated", 0) or 0)
            total_main_tasks_passed += int(agent.get("main_tasks_passed", 0) or 0)
            for main_status, count in (agent.get("main_task_status_counts") or {}).items():
                key = str(main_status)
                main_task_status_counts[key] = main_task_status_counts.get(key, 0) + int(count or 0)

        for dim in agent.get("dimensions", []) or []:
            total_dimensions += 1
            verdict_raw = dim.get("verdict")
            verdict = str(verdict_raw).lower() if verdict_raw is not None else "scored_only"
            if verdict not in dimension_counts:
                verdict = "scored_only"
            dimension_counts[verdict] += 1
            if agent_ok:
                dim_score = dim.get("score")
                if isinstance(dim_score, (int, float)):
                    scored_dimensions += 1
                    total_dimension_score += float(dim_score)

    completed_agents = status_counts.get("completed", 0)
    total_agents = len(normalized_results)

    task_completion_by_agent = {
        r["agent_id"]: r.get("task_completion_score")
        for r in normalized_results
        if r.get("task_completion_score") is not None and r.get("status") == "completed"
    }
    avg_task_completion = (
        sum(task_completion_by_agent.values()) / len(task_completion_by_agent)
        if task_completion_by_agent else None
    )
    main_task_pass_rate_by_agent = {
        r["agent_id"]: r.get("main_task_pass_rate")
        for r in normalized_results
        if r.get("main_task_pass_rate") is not None and r.get("status") == "completed"
    }
    avg_main_task_pass_rate = (
        sum(main_task_pass_rate_by_agent.values()) / len(main_task_pass_rate_by_agent)
        if main_task_pass_rate_by_agent else None
    )
    agent_score_by_agent = {
        r["agent_id"]: r.get("score")
        for r in normalized_results
        if r.get("score") is not None and r.get("status") == "completed"
    }

    return {
        "evaluation_mode": "agentic",
        "task_id": task_id or str(uuid.uuid4()),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "agents_total": total_agents,
            "agents_completed": completed_agents,
            "agent_status_counts": status_counts,
            "agents_scored": scored_agents,
            "agent_score_by_agent": agent_score_by_agent,
            "agent_score_average": (total_agent_score / scored_agents) if scored_agents else None,
            "overall_score": total_agent_score if scored_agents else None,
            "overall_max_score": total_max_score if total_max_score > 0 else None,
            "dimensions_total": total_dimensions,
            "dimension_verdict_counts": dimension_counts,
            "dimensions_scored": scored_dimensions,
            "dimension_score_average": (
                total_dimension_score / scored_dimensions
            ) if scored_dimensions else None,
            "total_steps_used": total_steps,
            "total_duration_ms": total_duration_ms,
            "main_tasks_total": total_main_tasks,
            "main_tasks_evaluated": total_main_tasks_evaluated,
            "main_tasks_passed": total_main_tasks_passed,
            "main_task_status_counts": main_task_status_counts,
            "main_task_pass_rate_by_agent": main_task_pass_rate_by_agent,
            "avg_main_task_pass_rate": avg_main_task_pass_rate,
            "task_completion_by_agent": task_completion_by_agent,
            "avg_task_completion_score": avg_task_completion,
        },
        "agents": normalized_results,
    }
