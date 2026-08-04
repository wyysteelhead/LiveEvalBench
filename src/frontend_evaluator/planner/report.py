"""Report aggregator — combines all check results into a structured report."""

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .source_scanner import CheckResult


@dataclass
class CategoryScore:
    """Score summary for a single category."""
    category: str
    total: int
    passed: int
    score: float  # 0.0 – 1.0

    def to_dict(self) -> dict:
        return {
            "category": self.category,
            "total": self.total,
            "passed": self.passed,
            "score": round(self.score, 3),
        }


@dataclass
class PlannerReport:
    """Aggregated report from all planner checks."""
    source_results: List[CheckResult] = field(default_factory=list)
    dom_results: List[CheckResult] = field(default_factory=list)
    behavior_results: List[Dict] = field(default_factory=list)
    visual_results: List[CheckResult] = field(default_factory=list)
    build_error: Optional[str] = None
    category_scores: Dict[str, CategoryScore] = field(default_factory=dict)
    overall_score: float = 0.0
    query_decomposition: Dict = field(default_factory=dict)
    phase_requirement_scores: Dict = field(default_factory=dict)
    query_fulfillment_score: Optional[float] = None
    interaction_visual_standard_scores: Dict = field(default_factory=dict)
    # New: function-first planning results
    planned_tasks: List[Dict] = field(default_factory=list)
    task_results: List[Dict] = field(default_factory=list)
    task_completion_score: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "overall_score": round(self.overall_score, 3),
            "build_error": self.build_error,
            "category_scores": {
                k: v.to_dict() for k, v in self.category_scores.items()
            },
            "source_checks": [r.to_dict() for r in self.source_results],
            "dom_checks": [r.to_dict() for r in self.dom_results],
            "behavior_checks": self.behavior_results,
            "visual_checks": [r.to_dict() for r in self.visual_results],
            "query_decomposition": self.query_decomposition,
            "phase_requirement_scores": self.phase_requirement_scores,
            "query_fulfillment_score": (
                round(self.query_fulfillment_score, 3)
                if self.query_fulfillment_score is not None
                else None
            ),
            "interaction_visual_standard_scores": self.interaction_visual_standard_scores,
            "planned_tasks": self.planned_tasks,
            "task_results": self.task_results,
            "task_completion_score": (
                round(self.task_completion_score, 3)
                if self.task_completion_score is not None
                else None
            ),
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


def aggregate(
    source_results: List[CheckResult],
    dom_results: List[CheckResult],
    behavior_results: List[Dict],
    visual_results: List[CheckResult],
    build_error: Optional[str] = None,
    weights: Optional[Dict[str, float]] = None,
    query_decomposition: Optional[Dict] = None,
    phase_requirement_scores: Optional[Dict] = None,
    query_fulfillment_score: Optional[float] = None,
    interaction_visual_standard_scores: Optional[Dict] = None,
    planned_tasks: Optional[List[Dict]] = None,
    task_results: Optional[List[Dict]] = None,
    task_completion_score: Optional[float] = None,
) -> PlannerReport:
    """Aggregate all check results into a PlannerReport.

    Args:
        source_results: Results from static source analysis
        dom_results: Results from DOM auditor
        behavior_results: Results from behavior tests (list of verdict dicts)
        visual_results: Results from visual checker
        build_error: Build error message if app failed to start
        weights: Optional per-category weight overrides (default: equal weights)

    Returns:
        PlannerReport with scores computed
    """
    report = PlannerReport(
        source_results=source_results,
        dom_results=dom_results,
        behavior_results=behavior_results,
        visual_results=visual_results,
        build_error=build_error,
        query_decomposition=query_decomposition or {},
        phase_requirement_scores=phase_requirement_scores or {},
        query_fulfillment_score=query_fulfillment_score,
        interaction_visual_standard_scores=interaction_visual_standard_scores or {},
        planned_tasks=planned_tasks or [],
        task_results=task_results or [],
        task_completion_score=task_completion_score,
    )

    if build_error:
        report.overall_score = 0.0
        return report

    # Collect all CheckResult objects
    all_check_results: List[CheckResult] = (
        source_results + dom_results + visual_results
    )

    # Convert behavior verdicts to CheckResult-like objects for scoring.
    # If standard-level aggregated scores are available, score by standard_id.
    # N/A items are excluded from both numerator and denominator.
    behavior_check_results: List[CheckResult] = []
    standard_scores = interaction_visual_standard_scores or {}
    if standard_scores:
        for sid, row in standard_scores.items():
            if not isinstance(row, dict):
                continue
            verdict_type = str(row.get("final_verdict", "")).strip().lower()
            if verdict_type == "not_applicable":
                continue
            passed = verdict_type == "passed"
            behavior_check_results.append(CheckResult(
                check_id=str(sid),
                category="behavior",
                passed=passed,
                detail=str(row.get("summary_reason", "")),
                score=1.0 if passed else 0.0,
            ))
    else:
        for i, r in enumerate(behavior_results):
            v = r.get("verdict", {})
            verdict_type = v.get("verdict", "failed" if not v.get("passed") else "passed")
            if verdict_type == "not_applicable":
                continue   # excluded from scoring
            behavior_check_results.append(CheckResult(
                check_id=r.get("query_id", f"behavior_{i}"),
                category=r.get("category", "behavior"),
                passed=(verdict_type == "passed"),
                detail=v.get("reason", ""),
                score=1.0 if verdict_type == "passed" else 0.0,
            ))
    all_check_results.extend(behavior_check_results)

    # Group by category
    by_category: Dict[str, List[CheckResult]] = {}
    for result in all_check_results:
        by_category.setdefault(result.category, []).append(result)

    # Compute per-category scores
    category_scores: Dict[str, CategoryScore] = {}
    for cat, checks in by_category.items():
        total = len(checks)
        passed = sum(1 for c in checks if c.passed)
        score = passed / total if total > 0 else 1.0
        category_scores[cat] = CategoryScore(
            category=cat,
            total=total,
            passed=passed,
            score=score,
        )

    report.category_scores = category_scores

    # Compute overall score (weighted average, equal weights by default)
    if category_scores:
        effective_weights = weights or {}
        total_weight = 0.0
        weighted_sum = 0.0
        for cat, cs in category_scores.items():
            w = effective_weights.get(cat, 1.0)
            weighted_sum += cs.score * w
            total_weight += w
        report.overall_score = weighted_sum / total_weight if total_weight > 0 else 0.0
    else:
        report.overall_score = 1.0

    return report
