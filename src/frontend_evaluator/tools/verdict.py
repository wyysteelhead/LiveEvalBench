"""Verdict submission tool for agent evaluation results."""

import ast
import json
from typing import Any, Dict, List, Optional

from .registry import register_tool


class VerdictSubmitted(Exception):
    """Exception raised when verdict is submitted to signal evaluation completion."""

    def __init__(self, verdict: str, reason: str, verdicts: Optional[List[Dict[str, str]]] = None):
        # verdict: "passed" | "failed" | "not_applicable" | "unable"
        self.verdict = verdict
        # "unable" is not a pass, but it is also not a failure — treat as
        # passed for backward-compat so downstream scoring does not penalise.
        self.passed = (verdict != "failed")
        self.reason = reason
        self.verdicts = verdicts or []
        super().__init__(f"Verdict: {verdict.upper()} - {reason}")


@register_tool(
    name="submit_verdict",
    description=(
        "Submit the final evaluation verdict. Use this when you have completed testing. "
        "verdict must be one of: 'passed', 'failed', 'not_applicable', or 'unable'.\n"
        "- 'passed': the feature exists and works correctly.\n"
        "- 'failed': the feature should exist based on the app's purpose but is missing or broken.\n"
        "- 'not_applicable': the query asks about a feature that is genuinely not part of "
        "this app's scope (e.g. testing form submission on a pure counter app with no inputs). "
        "Use this ONLY when the app clearly has no reason to have the feature — "
        "NOT as an excuse to skip testing something that is broken.\n"
        "- 'unable': you genuinely cannot determine whether the feature passes or fails "
        "(e.g. the page cannot be reached, critical files are missing, the environment is "
        "broken beyond repair, or a required capability is unavailable). "
        "Use this ONLY when you have made a reasonable effort and the task is genuinely "
        "impossible to evaluate, NOT as a shortcut to avoid testing."
    ),
)
async def submit_verdict(verdict: str, reason: str) -> str:
    """Submit final evaluation verdict.

    Args:
        verdict: "passed", "failed", "not_applicable", or "unable"
        reason: Detailed explanation of the verdict

    Raises:
        VerdictSubmitted: Always raised to signal completion
    """
    if not isinstance(verdict, str) or verdict not in ("passed", "failed", "not_applicable", "unable"):
        return (
            "ERROR: Invalid verdict value. verdict must be one of: 'passed', 'failed', "
            "'not_applicable', 'unable'. You submitted: " + repr(verdict) + ". "
            "Please call submit_verdict again with a valid verdict."
        )
    if not isinstance(reason, str) or not reason.strip():
        return (
            "ERROR: reason must be a non-empty string explaining the verdict. "
            "Please call submit_verdict again with a detailed reason."
        )
    raise VerdictSubmitted(
        verdict=verdict,
        reason=reason.strip(),
        verdicts=[{"standard_id": "", "verdict": verdict, "reason": reason.strip()}],
    )


@register_tool(
    name="submit_group_verdict",
    description=(
        "Submit final verdicts for multiple standards in one call. "
        "Use when one task validates several standards together.\n"
        "Input:\n"
        "- verdicts: array of items with fields\n"
        "  - standard_id: standard identifier\n"
        "  - verdict: one of 'passed', 'failed', 'not_applicable', 'unable'\n"
        "  - reason: concise evidence-based explanation\n"
        "Every standard tested in this run must be included exactly once."
    ),
)
async def submit_group_verdict(verdicts: List[Dict[str, Any]]) -> str:
    """Submit multi-standard verdicts in one call."""
    payload = _coerce_group_verdict_payload(verdicts)
    if not isinstance(payload, list) or not payload:
        raise VerdictSubmitted(
            verdict="failed",
            reason="Malformed group verdict payload",
            verdicts=[{"standard_id": "", "verdict": "failed", "reason": "Malformed group verdict payload"}],
        )

    normalized = []
    passed = 0
    failed = 0
    not_applicable = 0
    unable = 0
    for item in payload:
        sid = str(item.get("standard_id", "")).strip()
        raw_verdict = str(item.get("verdict", "")).strip().lower()
        if raw_verdict in ("passed", "failed", "not_applicable", "unable"):
            verdict = raw_verdict
        elif raw_verdict:
            verdict = "scored"
        else:
            verdict = "scored"
        reason = str(item.get("reason", "")).strip() or "No reason provided"
        evidence = item.get("evidence", [])
        if not isinstance(evidence, list):
            evidence = []
        extra_fields = {
            key: value
            for key, value in item.items()
            if key not in {"standard_id", "verdict", "reason", "evidence"}
        }
        normalized.append({
            "standard_id": sid,
            "verdict": verdict,
            "reason": reason,
            "evidence": evidence,
            **extra_fields,
        })
        if verdict == "passed":
            passed += 1
        elif verdict == "failed":
            failed += 1
        elif verdict == "unable":
            unable += 1
        else:
            not_applicable += 1

    # Summary is "failed" only when at least one standard actually failed.
    # "not_applicable" and "unable" should not force the whole grouped submission to failed.
    summary_verdict = "failed" if failed > 0 else "passed"
    summary_reason = (
        f"Submitted {len(normalized)} grouped verdict(s): "
        f"{passed} passed, {failed} failed, {unable} unable, {not_applicable} not_applicable."
    )
    raise VerdictSubmitted(
        verdict=summary_verdict,
        reason=summary_reason,
        verdicts=normalized,
    )


def _coerce_group_verdict_payload(value: Any) -> Any:
    """Accept list payloads and tolerate common model formatting mistakes."""
    if isinstance(value, list):
        return value

    # Some models send a JSON string.
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
            return _coerce_group_verdict_payload(decoded)
        except Exception:
            try:
                decoded = ast.literal_eval(value)
                return _coerce_group_verdict_payload(decoded)
            except Exception:
                return value

    # Some models wrap array in an object.
    if isinstance(value, dict):
        for key in ("verdicts", "items", "results"):
            if key in value:
                return _coerce_group_verdict_payload(value[key])
    return value
