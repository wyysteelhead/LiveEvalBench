"""Main planner orchestrator."""

from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional

from ..agent.runtime_factory import create_evaluator_runtime
from ..events.emitter import EventEmitter
from ..llm.factory import LLMFactory
from ..sandbox.agent_browser_executor import get_executor
from ..utils.config import Config
from ..utils.logger import logger
from . import dom_agent, dom_auditor, query_decomposer, query_generator, source_agent, source_scanner, task_planner
from .report import PlannerReport, aggregate
from .source_scanner import CheckResult


class Planner:
    """Autonomous planner that runs all guideline checks in a single sandbox."""

    async def run(
        self,
        files: Dict[str, str],
        config: Config,
        weights: Optional[Dict[str, float]] = None,
        event_emitter: Optional[EventEmitter] = None,
        user_query: Optional[str] = None,
        queries_path: Optional[Path] = None,
        regenerate_queries: bool = False,
    ) -> PlannerReport:
        """Run the full evaluation pipeline.

        Args:
            files: Dict mapping filename to file content
            config: Configuration instance
            weights: Optional per-category score weights
            event_emitter: Optional event emitter for web visualization
            user_query: Original prompt used to generate the UI (e.g. "制作一个咖啡宣传网站").
                        Passed to the query generator to produce relevant behavior tests.
            queries_path: Path to a .queries.json sidecar file. If the file exists and
                          regenerate_queries is False, queries are loaded from it instead of
                          being regenerated. After generation the queries are saved here.
            regenerate_queries: Force re-generation even if a queries file already exists.

        Returns:
            PlannerReport with all results
        """
        async def _emit(phase: str, data: dict):
            if event_emitter:
                await event_emitter.emit("planner_phase", {"phase": phase, **data})

        # 1. Static source analysis (no sandbox needed)
        logger.info("[Planner] Running source scanner...")
        await _emit("source_scan", {"status": "running"})
        src_results = source_scanner.scan(files)
        await _emit("source_scan", {"status": "done", "checks": len(src_results)})
        logger.info(f"[Planner] Source scan: {len(src_results)} checks")

        # Build LLM instance
        llm = LLMFactory.create_llm(
            provider=config.model_provider,
            api_key=config.get_llm_api_key(),
            model=config.model_name,
            base_url=config.custom_base_url if config.model_provider == "custom" else None,
            temperature=config.temperature,
            disable_thinking=config.disable_thinking,
        )

        # 1a. Decompose user query into phase requirements
        logger.info("[Planner] Decomposing user query...")
        await _emit("query_decomposition", {"status": "running"})
        query_decomposition = await query_decomposer.decompose(user_query, llm)
        await _emit("query_decomposition", {
            "status": "done",
            "source_requirements": len(query_decomposition.get("source_requirements", [])),
            "dom_requirements": len(query_decomposition.get("dom_requirements", [])),
            "behavior_requirements": len(query_decomposition.get("behavior_requirements", [])),
            "visual_requirements": len(query_decomposition.get("visual_requirements", [])),
        })

        # 1b. Source agent (LLM checks ambiguous source patterns)
        logger.info("[Planner] Running source agent...")
        await _emit("source_agent", {"status": "running"})
        src_agent_results = await source_agent.run(files, llm)
        src_results = src_results + src_agent_results
        await _emit("source_agent", {"status": "done", "checks": len(src_agent_results)})
        logger.info(f"[Planner] Source agent: {len(src_agent_results)} checks")

        # 3. Start sandbox
        logger.info("[Planner] Starting sandbox...")
        await _emit("sandbox_start", {"status": "running"})
        sandbox = self._create_sandbox(config)

        dom_results = []
        behavior_results = []
        visual_results = []
        standard_scores = {}
        build_error = None
        queries: List[query_generator.BehaviorQuery] = []
        planned_tasks_v1: List[task_planner.PlannedTask] = []
        planned_tasks_v2: List[task_planner.PlannedTask] = []

        try:
            sandbox.start()
            sandbox.write_files(files)

            # 4. Build gate: start app
            logger.info("[Planner] Starting app (npm install + dev server)...")
            await _emit("build", {"status": "running"})
            try:
                await sandbox.start_app(
                    install_timeout=config.sandbox_startup_timeout,
                    startup_timeout=config.app_startup_timeout,
                )
            except Exception as e:
                build_error = str(e)
                logger.error(f"[Planner] Build failed: {e}")
                await _emit("build", {"status": "failed", "error": build_error})
                return aggregate(
                    source_results=src_results,
                    dom_results=[],
                    behavior_results=[],
                    visual_results=[],
                    build_error=build_error,
                    weights=weights,
                    query_decomposition=query_decomposition,
                )

            await _emit("build", {"status": "done"})
            app_url = sandbox.get_app_url()
            logger.info(f"[Planner] App running at {app_url}")

            # 5. Start executor
            executor = get_executor(config.executor_backend, sandbox, app_url)
            logger.info(f"[Planner] Starting executor ({config.executor_backend})...")
            await executor.start_service(timeout=max(config.cdp_connection_timeout, 90))

            try:
                # 5a. Collect initial page context for task planning
                logger.info("[Planner] Collecting initial page context...")
                initial_screenshot_b64: Optional[str] = None
                page_context: Optional[str] = None
                try:
                    initial_screenshot_b64 = await executor.screenshot()
                    ctx = await executor.get_context()
                    page_context = str(ctx) if ctx else None
                except Exception as e:
                    logger.warning("[Planner] Could not collect initial context: %s", e)

                # 5b. Two-round task planning (function-first)
                all_requirements = (
                    query_decomposition.get("source_requirements", [])
                    + query_decomposition.get("dom_requirements", [])
                    + query_decomposition.get("behavior_requirements", [])
                    + query_decomposition.get("visual_requirements", [])
                )

                sidecar_data = self._load_sidecar(queries_path) if queries_path and not regenerate_queries else None

                if sidecar_data is not None and sidecar_data.get("behavior_queries"):
                    queries = query_generator.load_queries_from_list(sidecar_data["behavior_queries"])
                    logger.info("[Planner] Loaded %d queries from sidecar", len(queries))
                else:
                    logger.warning("[Planner] No sidecar data; skipping behavior query generation (tree mode only)")

                await _emit("query_generation", {"status": "done", "queries": len(queries)})
                logger.info("[Planner] %d behavior queries ready", len(queries))
                # 6. DOM audit
                logger.info("[Planner] Running DOM audit...")
                await _emit("dom_audit", {"status": "running"})
                dom_results = await dom_auditor.audit(executor, app_url)
                await _emit("dom_audit", {"status": "done", "checks": len(dom_results)})
                logger.info(f"[Planner] DOM audit: {len(dom_results)} checks")

                # 6b. DOM agent (LLM checks ambiguous DOM patterns)
                logger.info("[Planner] Running DOM agent...")
                await _emit("dom_agent", {"status": "running"})
                dom_agent_results = await dom_agent.run(executor, llm)
                dom_results = dom_results + dom_agent_results
                await _emit("dom_agent", {"status": "done", "checks": len(dom_agent_results)})
                logger.info(f"[Planner] DOM agent: {len(dom_agent_results)} checks")

                # 7. Interaction + visual rubric tests (single merged execution phase)
                evaluator = create_evaluator_runtime(
                    config,
                    max_iterations=config.max_agent_steps,
                )

                grouped_queries = self._build_execution_groups(queries)
                await _emit("behavior_tests", {"status": "running", "total": len(grouped_queries)})
                for i, group in enumerate(grouped_queries):
                    logger.info(
                        "[Planner] Running behavior group: %s (%s)",
                        group["group_id"],
                        ", ".join(group["standard_ids"]),
                    )
                    await _emit("behavior_tests", {
                        "status": "query_start",
                        "index": i,
                        "total": len(grouped_queries),
                        "query_id": group["group_id"],
                        "query_text": group["query_text"],
                    })
                    await executor.reset(app_url)
                    try:
                        result = await evaluator.evaluate(
                            user_query=group["query_text"],
                            app_url=app_url,
                            executor=executor,
                            standard_ids=group["standard_ids"],
                        )
                        per_standard_verdicts = self._normalize_group_verdicts(
                            result=result,
                            standard_ids=group["standard_ids"],
                        )
                        behavior_results.extend(
                            self._materialize_group_results(
                                group_queries=group["queries"],
                                per_standard_verdicts=per_standard_verdicts,
                                result=result,
                            )
                        )
                        summary = per_standard_verdicts.get(group["standard_ids"][0], {})
                        await _emit("behavior_tests", {
                            "status": "query_done",
                            "index": i,
                            "query_id": group["group_id"],
                            "passed": summary.get("passed", False),
                            "reason": summary.get("reason", "")[:120],
                        })
                    except Exception as e:
                        logger.error(f"[Planner] Behavior group {group['group_id']} failed: {e}")
                        failed = {
                            sid: {"verdict": "failed", "passed": False, "reason": str(e)}
                            for sid in group["standard_ids"]
                        }
                        behavior_results.extend(
                            self._materialize_group_results(
                                group_queries=group["queries"],
                                per_standard_verdicts=failed,
                                result={"iterations": 0, "steps": [], "dom_elements": []},
                            )
                        )
                        await _emit("behavior_tests", {
                            "status": "query_error",
                            "index": i,
                            "query_id": group["group_id"],
                            "error": str(e)[:120],
                        })

                # Visual rubric is now covered by behavior query execution.
                # Keep visual_results for backward report compatibility.
                visual_results = []

            finally:
                await executor.shutdown()

        finally:
            try:
                sandbox.stop()
            except Exception:
                pass

        # 9. Aggregate report
        phase_scores = query_decomposer.compute_phase_requirement_scores(
            query_decomposition=query_decomposition,
            source_results=src_results,
            dom_results=dom_results,
            behavior_results=behavior_results,
            visual_results=visual_results,
        )
        standard_scores = self._aggregate_standard_scores(behavior_results)
        task_results_list = self._aggregate_task_results(planned_tasks_v2, behavior_results)
        task_completion = self._compute_task_completion_score(task_results_list)
        report = aggregate(
            source_results=src_results,
            dom_results=dom_results,
            behavior_results=behavior_results,
            visual_results=visual_results,
            interaction_visual_standard_scores=standard_scores,
            build_error=build_error,
            weights=weights,
            query_decomposition=query_decomposition,
            phase_requirement_scores=phase_scores,
            query_fulfillment_score=phase_scores.get("query_fulfillment_score"),
            planned_tasks=[t.to_dict() for t in planned_tasks_v2],
            task_results=task_results_list,
            task_completion_score=task_completion,
        )

        await _emit("done", {
            "overall_score": round(report.overall_score, 3),
            "categories": {k: v.to_dict() for k, v in report.category_scores.items()},
        })

        logger.info(
            f"[Planner] Done. Overall score: {report.overall_score:.1%} "
            f"({sum(1 for c in report.category_scores.values() if c.score == 1.0)}/"
            f"{len(report.category_scores)} categories passed)"
        )
        return report

    async def rerun(
        self,
        files: Dict[str, str],
        original_report: Dict,
        queries_path: Path,
        config: Config,
        event_emitter: Optional[EventEmitter] = None,
    ) -> PlannerReport:
        """Re-run only the behavior checks that were edited or deleted in the sidecar.

        Diff logic:
        - Deleted (in original but not in sidecar): regenerate a new query for that standard_id
        - Modified (text changed): re-run with the new query_text
        - Unchanged: keep the original result as-is

        Source / DOM / visual checks are taken unchanged from original_report.
        """
        async def _emit(phase: str, data: dict):
            if event_emitter:
                await event_emitter.emit("planner_phase", {"phase": phase, **data})

        # Load sidecar (new format with behavior_queries key; fallback to legacy flat list)
        sidecar_data = self._load_sidecar(queries_path)
        if sidecar_data and sidecar_data.get("behavior_queries"):
            sidecar_queries = query_generator.load_queries_from_list(sidecar_data["behavior_queries"])
        else:
            sidecar_queries = query_generator.load_queries(queries_path) or []
        sidecar_by_id = {q.query_id: q for q in sidecar_queries}

        # Original behavior results
        original_behavior: List[Dict] = original_report.get("behavior_checks", [])
        original_by_id = {c["query_id"]: c for c in original_behavior}

        # Classify each sidecar query
        to_rerun: List[query_generator.BehaviorQuery] = []
        unchanged_results: List[Dict] = []

        for q in sidecar_queries:
            orig = original_by_id.get(q.query_id)
            if orig is None or orig.get("query_text") != q.query_text:
                to_rerun.append(q)
            else:
                unchanged_results.append(orig)

        # Deleted: in original but not in sidecar → regenerate
        deleted_ids = [qid for qid in original_by_id if qid not in sidecar_by_id]
        if deleted_ids:
            deleted_standard_ids = sorted({
                (original_by_id[qid].get("standard_id") or qid.split("__", 1)[0])
                for qid in deleted_ids
            })
            llm = LLMFactory.create_llm(
                provider=config.model_provider,
                api_key=config.get_llm_api_key(),
                model=config.model_name,
                base_url=config.custom_base_url if config.model_provider == "custom" else None,
                temperature=config.temperature,
                disable_thinking=config.disable_thinking,
            )
            new_queries = await query_generator.generate_for_standards(
                files, llm, deleted_standard_ids
            )
            to_rerun.extend(new_queries)
            # Persist the regenerated queries back to the sidecar
            all_queries = sidecar_queries + new_queries
            if sidecar_data:
                sidecar_data["behavior_queries"] = [q.to_dict() for q in all_queries]
                self._save_sidecar(queries_path, sidecar_data)
            else:
                query_generator.save_queries(all_queries, queries_path)
            logger.info(
                "[Planner.rerun] Regenerated %s queries for deleted standards: %s",
                len(new_queries),
                deleted_standard_ids,
            )

        logger.info(
            f"[Planner.rerun] {len(unchanged_results)} unchanged, "
            f"{len(to_rerun)} to re-run, {len(deleted_ids)} deleted/regenerated"
        )

        if not to_rerun:
            # Nothing changed — rebuild report from original data
            return self._rebuild_report_from_original(original_report)

        # Start sandbox and re-run only the changed queries
        llm = LLMFactory.create_llm(
            provider=config.model_provider,
            api_key=config.get_llm_api_key(),
            model=config.model_name,
            base_url=config.custom_base_url if config.model_provider == "custom" else None,
            temperature=config.temperature,
            disable_thinking=config.disable_thinking,
        )

        sandbox = self._create_sandbox(config)
        new_behavior_results: List[Dict] = []

        try:
            sandbox.start()
            sandbox.write_files(files)
            await _emit("build", {"status": "running"})
            try:
                await sandbox.start_app(
                    install_timeout=config.sandbox_startup_timeout,
                    startup_timeout=config.app_startup_timeout,
                )
            except Exception as e:
                logger.error(f"[Planner.rerun] Build failed: {e}")
                await _emit("build", {"status": "failed", "error": str(e)})
                return self._rebuild_report_from_original(original_report, build_error=str(e))

            await _emit("build", {"status": "done"})
            app_url = sandbox.get_app_url()

            executor = get_executor(config.executor_backend, sandbox, app_url)
            await executor.start_service(timeout=max(config.cdp_connection_timeout, 90))

            try:
                evaluator = create_evaluator_runtime(
                    config,
                    max_iterations=config.max_agent_steps,
                )

                grouped_queries = self._build_execution_groups(to_rerun)
                await _emit("behavior_tests", {"status": "running", "total": len(grouped_queries)})
                for i, group in enumerate(grouped_queries):
                    logger.info(f"[Planner.rerun] Running group: {group['group_id']}")
                    await executor.reset(app_url)
                    try:
                        result = await evaluator.evaluate(
                            user_query=group["query_text"],
                            app_url=app_url,
                            executor=executor,
                            standard_ids=group["standard_ids"],
                        )
                        per_standard_verdicts = self._normalize_group_verdicts(
                            result=result,
                            standard_ids=group["standard_ids"],
                        )
                        new_behavior_results.extend(
                            self._materialize_group_results(
                                group_queries=group["queries"],
                                per_standard_verdicts=per_standard_verdicts,
                                result=result,
                            )
                        )
                    except Exception as e:
                        logger.error(f"[Planner.rerun] Group {group['group_id']} failed: {e}")
                        failed = {
                            sid: {"verdict": "failed", "passed": False, "reason": str(e)}
                            for sid in group["standard_ids"]
                        }
                        new_behavior_results.extend(
                            self._materialize_group_results(
                                group_queries=group["queries"],
                                per_standard_verdicts=failed,
                                result={"iterations": 0, "steps": [], "dom_elements": []},
                            )
                        )
            finally:
                await executor.shutdown()
        finally:
            try:
                sandbox.stop()
            except Exception:
                pass

        # Merge: preserve sidecar order, then append any newly regenerated queries
        rerun_by_id = {r["query_id"]: r for r in new_behavior_results}
        merged_behavior: List[Dict] = []
        seen_ids: set = set()
        for q in sidecar_queries:
            seen_ids.add(q.query_id)
            if q.query_id in rerun_by_id:
                merged_behavior.append(rerun_by_id[q.query_id])
            elif q.query_id in original_by_id:
                merged_behavior.append(original_by_id[q.query_id])
        # Append regenerated queries for deleted standards (not in sidecar)
        for r in new_behavior_results:
            if r["query_id"] not in seen_ids:
                merged_behavior.append(r)

        # Reconstruct source/dom/visual CheckResult lists from original
        src_results = [
            CheckResult(check_id=c["check_id"], category=c["category"],
                        passed=c["passed"], detail=c["detail"], score=c.get("score", 1.0))
            for c in original_report.get("source_checks", [])
        ]
        dom_results = [
            CheckResult(check_id=c["check_id"], category=c["category"],
                        passed=c["passed"], detail=c["detail"], score=c.get("score", 1.0))
            for c in original_report.get("dom_checks", [])
        ]
        visual_results = [
            CheckResult(check_id=c["check_id"], category=c["category"],
                        passed=c["passed"], detail=c["detail"], score=c.get("score", 1.0),
                        screenshot=c.get("screenshot"))
            for c in original_report.get("visual_checks", [])
        ]

        rerun_phase_scores = query_decomposer.compute_phase_requirement_scores(
            query_decomposition=original_report.get("query_decomposition", {}),
            source_results=src_results,
            dom_results=dom_results,
            behavior_results=merged_behavior,
            visual_results=visual_results,
        )
        rerun_standard_scores = self._aggregate_standard_scores(merged_behavior)
        rerun_planned_tasks = original_report.get("planned_tasks", [])
        rerun_planned_task_objs = [task_planner.PlannedTask.from_dict(t) for t in rerun_planned_tasks]
        rerun_task_results = self._aggregate_task_results(rerun_planned_task_objs, merged_behavior)
        rerun_task_completion = self._compute_task_completion_score(rerun_task_results)
        report = aggregate(
            source_results=src_results,
            dom_results=dom_results,
            behavior_results=merged_behavior,
            visual_results=visual_results,
            interaction_visual_standard_scores=rerun_standard_scores,
            query_decomposition=original_report.get("query_decomposition", {}),
            phase_requirement_scores=rerun_phase_scores,
            query_fulfillment_score=rerun_phase_scores.get("query_fulfillment_score"),
            planned_tasks=rerun_planned_tasks,
            task_results=rerun_task_results,
            task_completion_score=rerun_task_completion,
        )
        logger.info(f"[Planner.rerun] Done. New overall score: {report.overall_score:.1%}")
        return report

    def _build_execution_groups(self, queries: List[query_generator.BehaviorQuery]) -> List[Dict]:
        grouped: Dict[str, Dict] = {}
        for q in queries:
            group_id = q.execution_group_id or q.query_id
            scenario_id = q.scenario_id or "scenario"
            key = f"{group_id}::{scenario_id}::{q.query_text}"
            if key not in grouped:
                grouped[key] = {
                    "group_id": group_id,
                    "scenario_id": scenario_id,
                    "query_text": q.query_text,
                    "standard_ids": [],
                    "queries": [],
                }
            if q.standard_id not in grouped[key]["standard_ids"]:
                grouped[key]["standard_ids"].append(q.standard_id)
            grouped[key]["queries"].append(q)
        return list(grouped.values())

    def _create_sandbox(self, config: Config):
        """Sandbox creation is NOT supported in the public (local-only) release.

        eval_open.py runs locally via CdpPlaywrightExecutor and never calls this.
        Remote sandbox providers (E2B/Docker/Remote/OpenSandbox) were removed.
        """
        raise NotImplementedError(
            "Remote sandbox providers are not included in the public release. "
            "Use eval_open.py with EXECUTOR_BACKEND=playwright (local)."
        )

    def _normalize_group_verdicts(
        self,
        result: Dict,
        standard_ids: List[str],
    ) -> Dict[str, Dict]:
        parsed: Dict[str, Dict] = {}
        for item in result.get("group_verdicts", []) or []:
            sid = str(item.get("standard_id", "")).strip()
            if sid in standard_ids:
                verdict = str(item.get("verdict", "failed")).lower()
                reason = str(item.get("reason", ""))
                raw_score = item.get("score")
                score_100 = None
                if isinstance(raw_score, (int, float)):
                    s = float(raw_score)
                    score_100 = s if s > 1.0 else s * 100.0
                passed = (verdict == "passed") or (
                    verdict == "scored" and score_100 is not None and score_100 >= 60.0
                )
                row = {
                    "verdict": verdict,
                    "passed": passed,
                    "reason": reason,
                }
                for extra_key in (
                    "score",
                    "rating",
                    "confidence",
                    "severity",
                    "observations",
                    "recommendation",
                    "checks",
                    "subcriteria",
                    "evidence",
                ):
                    if extra_key in item:
                        row[extra_key] = item.get(extra_key)
                parsed[sid] = row

        # Backward compatibility: if model used single submit_verdict, apply to whole group.
        if not parsed:
            v = result.get("verdict", {})
            verdict = str(v.get("verdict", "failed")).lower()
            reason = str(v.get("reason", ""))
            malformed = "malformed group verdict payload" in reason.lower()
            for sid in standard_ids:
                fallback_reason = reason
                if malformed:
                    fallback_reason = (
                        f"No valid per-standard verdict returned for {sid}; "
                        "agent submitted malformed grouped payload"
                    )
                parsed[sid] = {
                    "verdict": verdict,
                    "passed": verdict == "passed",
                    "reason": fallback_reason,
                }

        # Ensure all standards in this group have a verdict.
        for sid in standard_ids:
            if sid not in parsed:
                parsed[sid] = {
                    "verdict": "failed",
                    "passed": False,
                    "reason": "No verdict returned for this standard in grouped run",
                }
        return parsed

    def _materialize_group_results(
        self,
        group_queries: List[query_generator.BehaviorQuery],
        per_standard_verdicts: Dict[str, Dict],
        result: Dict,
    ) -> List[Dict]:
        rows: List[Dict] = []
        for q in group_queries:
            rows.append({
                "query_id": q.query_id,
                "standard_id": q.standard_id,
                "guideline_ref": q.guideline_ref,
                "category": q.category,
                "query_text": q.query_text,
                "scenario_id": q.scenario_id,
                "scenario_index": q.scenario_index,
                "scenario_weight": q.scenario_weight,
                "check_nodes": q.check_nodes or [],
                "execution_group_id": q.execution_group_id or q.query_id,
                "execution_standard_ids": q.execution_standard_ids or [q.standard_id],
                "verdict": per_standard_verdicts.get(q.standard_id, {
                    "verdict": "failed",
                    "passed": False,
                    "reason": "Missing verdict",
                }),
                "iterations": result.get("iterations", 0),
                "steps": result.get("steps", []),
                "dom_elements": result.get("dom_elements", []),
            })
        return rows

    def _aggregate_standard_scores(self, behavior_results: List[Dict]) -> Dict[str, Dict]:
        by_standard: Dict[str, List[Dict]] = defaultdict(list)
        for row in behavior_results:
            sid = str(row.get("standard_id") or row.get("query_id") or "").strip()
            if sid:
                by_standard[sid].append(row)

        summary: Dict[str, Dict] = {}
        for sid, rows in by_standard.items():
            weighted_sum = 0.0
            total_weight = 0.0
            evaluated = 0
            reasons: List[str] = []
            scenario_ids: List[str] = []

            for r in rows:
                scenario_ids.append(str(r.get("scenario_id") or ""))
                verdict = r.get("verdict", {}) or {}
                vt = str(verdict.get("verdict", "failed")).lower().strip()
                if vt == "not_applicable":
                    continue
                score = verdict.get("score")
                if isinstance(score, (int, float)):
                    s = float(score)
                    score_100 = s if s > 1.0 else s * 100.0
                elif vt == "passed":
                    score_100 = 100.0
                elif vt == "failed":
                    score_100 = 0.0
                else:
                    score_100 = 0.0

                w = float(r.get("scenario_weight", 1.0) or 1.0)
                weighted_sum += score_100 * w
                total_weight += w
                evaluated += 1
                reason = str(verdict.get("reason", "")).strip()
                if reason:
                    reasons.append(reason)

            final_score = (weighted_sum / total_weight) if total_weight > 0 else None
            if final_score is None:
                final_rating = "not_applicable"
                final_verdict = "not_applicable"
            elif final_score >= 85:
                final_rating = "good"
                final_verdict = "passed"
            elif final_score >= 60:
                final_rating = "ok"
                final_verdict = "passed"
            else:
                final_rating = "poor"
                final_verdict = "failed"

            summary[sid] = {
                "standard_id": sid,
                "final_score": round(final_score, 2) if final_score is not None else None,
                "final_rating": final_rating,
                "final_verdict": final_verdict,
                "scenario_count": len(rows),
                "evaluated_scenarios": evaluated,
                "scenario_ids": [s for s in scenario_ids if s],
                "summary_reason": " | ".join(reasons[:3]) if reasons else "",
            }
        return summary

    def _rebuild_report_from_original(
        self, original_report: Dict, build_error: Optional[str] = None
    ) -> PlannerReport:
        """Reconstruct a PlannerReport from a stored report dict."""
        src_results = [
            CheckResult(check_id=c["check_id"], category=c["category"],
                        passed=c["passed"], detail=c["detail"], score=c.get("score", 1.0))
            for c in original_report.get("source_checks", [])
        ]
        dom_results = [
            CheckResult(check_id=c["check_id"], category=c["category"],
                        passed=c["passed"], detail=c["detail"], score=c.get("score", 1.0))
            for c in original_report.get("dom_checks", [])
        ]
        visual_results = [
            CheckResult(check_id=c["check_id"], category=c["category"],
                        passed=c["passed"], detail=c["detail"], score=c.get("score", 1.0),
                        screenshot=c.get("screenshot"))
            for c in original_report.get("visual_checks", [])
        ]
        return aggregate(
            source_results=src_results,
            dom_results=dom_results,
            behavior_results=original_report.get("behavior_checks", []),
            visual_results=visual_results,
            interaction_visual_standard_scores=original_report.get("interaction_visual_standard_scores", {}),
            build_error=build_error,
            query_decomposition=original_report.get("query_decomposition", {}),
            phase_requirement_scores=original_report.get("phase_requirement_scores", {}),
            query_fulfillment_score=original_report.get("query_fulfillment_score"),
            planned_tasks=original_report.get("planned_tasks", []),
            task_results=original_report.get("task_results", []),
            task_completion_score=original_report.get("task_completion_score"),
        )

    def _load_sidecar(self, queries_path: Optional[Path]) -> Optional[Dict]:
        """Load sidecar JSON. Returns None if file doesn't exist or is legacy flat list."""
        if not queries_path or not queries_path.exists():
            return None
        import json as _json
        try:
            raw = _json.loads(queries_path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                return raw
            # Legacy flat list — wrap it
            return {"behavior_queries": raw}
        except Exception:
            return None

    def _save_sidecar(self, queries_path: Path, data: Dict) -> None:
        """Save sidecar JSON with full planning chain."""
        import json as _json
        queries_path.write_text(
            _json.dumps(data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def _aggregate_task_results(
        self,
        planned_tasks: List["task_planner.PlannedTask"],
        behavior_results: List[Dict],
    ) -> List[Dict]:
        """Aggregate behavior results up to task level."""
        # Build lookup: scenario_id -> list of behavior result rows
        by_scenario: Dict[str, List[Dict]] = defaultdict(list)
        for row in behavior_results:
            sid = str(row.get("scenario_id") or "")
            if sid:
                by_scenario[sid].append(row)

        task_results = []
        for t in planned_tasks:
            scenario_id = t.scenario_id or f"{t.task_id}_s1"
            rows = by_scenario.get(scenario_id, [])

            linked_query_ids = [r.get("query_id", "") for r in rows]

            # Determine task_verdict: all not_applicable → not_applicable; any fail → failed; else passed
            verdicts = []
            for r in rows:
                v = r.get("verdict", {}) or {}
                vt = str(v.get("verdict", "failed")).lower()
                verdicts.append(vt)

            if not verdicts:
                task_verdict = "not_applicable"
                task_score = None
            elif all(vt == "not_applicable" for vt in verdicts):
                task_verdict = "not_applicable"
                task_score = None
            else:
                applicable = [vt for vt in verdicts if vt != "not_applicable"]
                passed_count = sum(1 for vt in applicable if vt == "passed")
                task_score = round(passed_count / len(applicable), 3) if applicable else 0.0
                task_verdict = "passed" if all(vt == "passed" for vt in applicable) else "failed"

            reasons = [
                str((r.get("verdict") or {}).get("reason", "")).strip()
                for r in rows
                if (r.get("verdict") or {}).get("reason")
            ]
            summary_reason = " | ".join(reasons[:2]) if reasons else ""

            task_results.append({
                "task_id": t.task_id,
                "title": t.title,
                "task_type": t.task_type,
                "generated_from": t.generated_from,
                "covers_standard_ids": t.covers_standard_ids,
                "linked_query_ids": linked_query_ids,
                "task_verdict": task_verdict,
                "task_score": task_score,
                "summary_reason": summary_reason,
            })
        return task_results

    def _compute_task_completion_score(self, task_results: List[Dict]) -> Optional[float]:
        """Compute task_completion_score = passed_tasks / evaluated_tasks (excludes not_applicable)."""
        evaluated = [t for t in task_results if t.get("task_verdict") != "not_applicable"]
        if not evaluated:
            return None
        passed = sum(1 for t in evaluated if t.get("task_verdict") == "passed")
        return round(passed / len(evaluated), 3)
