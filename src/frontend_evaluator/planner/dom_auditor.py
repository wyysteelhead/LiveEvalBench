"""DOM auditor — checks against the rendered page via JavaScript evaluation."""

from typing import Any, List

from .rubric_config import get_dom_auditor_checks
from .source_scanner import CheckResult


async def audit(executor: Any, app_url: str) -> List[CheckResult]:
    """Run DOM-level checks against the rendered page.

    Args:
        executor: Executor instance implementing ExecutorInterface
        app_url: URL of the running application (used for navigation if needed)

    Returns:
        List of CheckResult for each DOM check
    """
    results: List[CheckResult] = []
    check_defs = get_dom_auditor_checks()

    async def js(expr: str) -> Any:
        res = await executor.evaluate_js(expr)
        return res.get("result")

    for c in check_defs:
        check_id = str(c.get("check_id", "")).strip() or "dom_unknown"
        category = str(c.get("category", "")).strip() or "dom"
        check_type = str(c.get("type", "")).strip()
        expression = str(c.get("expression", ""))
        pass_detail = str(c.get("pass_detail", "Passed"))
        fail_detail = str(c.get("fail_detail", "Failed"))

        try:
            value = await js(expression)
            passed = _dom_check_passed(check_type, value)
            detail = pass_detail.format(value=value) if passed else fail_detail.format(value=value)
            results.append(CheckResult(
                check_id=check_id,
                category=category,
                passed=passed,
                detail=detail,
                score=1.0 if passed else 0.0,
            ))
        except Exception as e:
            results.append(CheckResult(
                check_id=check_id,
                category=category,
                passed=False,
                detail=f"Check failed: {e}",
                score=0.0,
            ))

    return results


def _dom_check_passed(check_type: str, value: Any) -> bool:
    if check_type == "js_count_zero":
        return int(value or 0) == 0
    if check_type == "js_count_gt_zero":
        return int(value or 0) > 0
    if check_type == "js_truthy":
        if isinstance(value, str):
            return bool(value.strip())
        return bool(value)
    if check_type == "js_falsy":
        return not bool(value)
    return False
