"""DOM LLM agent — checks ambiguous DOM patterns via JavaScript evaluation."""

from typing import Any, Dict, List

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from .rubric_config import get_dom_agent_standards
from .source_scanner import CheckResult

_TOOLS = [
    {
        "name": "evaluate_js",
        "description": "Evaluate a JavaScript expression in the browser and return the result.",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "JavaScript expression to evaluate. Must return a JSON-serializable value.",
                }
            },
            "required": ["expression"],
        },
    },
    {
        "name": "submit_results",
        "description": "Submit the final check results. Call this once you have checked ALL standards.",
        "parameters": {
            "type": "object",
            "properties": {
                "results": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "check_id": {"type": "string"},
                            "passed": {"type": "boolean"},
                            "detail": {"type": "string"},
                        },
                        "required": ["check_id", "passed", "detail"],
                    },
                }
            },
            "required": ["results"],
        },
    },
]


async def run(executor: Any, llm: Any) -> List[CheckResult]:
    """Run DOM agent checks against the rendered page.

    Args:
        executor: Executor instance implementing ExecutorInterface
        llm: LangChain LLM instance

    Returns:
        List of CheckResult for each DOM standard
    """
    standards = get_dom_agent_standards()
    if not standards:
        return []

    llm_with_tools = llm.bind_tools(_TOOLS)

    standards_block = "\n\n".join(
        f"check_id: {s['check_id']}\ncategory: {s['category']}\ndescription: {s['description']}"
        for s in standards
    )

    messages = [
        SystemMessage(content=(
            "You are a frontend DOM auditor. Inspect the rendered page via JavaScript "
            "to check specific frontend standards. Use evaluate_js to run checks, "
            "then call submit_results with your findings for ALL standards listed."
        )),
        HumanMessage(content=(
            f"Check each of the following standards and submit results for ALL of them:\n\n"
            f"{standards_block}\n\n"
            "Use evaluate_js to inspect the DOM and computed styles, then call submit_results."
        )),
    ]

    for _ in range(15):
        response = await llm_with_tools.ainvoke(messages)
        messages.append(response)

        if not hasattr(response, "tool_calls") or not response.tool_calls:
            break

        tool_messages = []
        submitted = None

        for tc in response.tool_calls:
            name = tc["name"]
            args = tc["args"]
            tid = tc["id"]

            if name == "evaluate_js":
                expression = args.get("expression", "")
                try:
                    res = await executor.evaluate_js(expression)
                    result = str(res.get("result", ""))
                except Exception as e:
                    result = f"JS error: {e}"
            elif name == "submit_results":
                submitted = args.get("results", [])
                result = "Results submitted."
            else:
                result = f"Unknown tool: {name}"

            tool_messages.append(ToolMessage(content=str(result), tool_call_id=tid))

        messages.extend(tool_messages)

        if submitted is not None:
            return _build_check_results(submitted, standards=standards)

    return _all_failed("DOM agent did not complete checks", standards=standards)


def _build_check_results(raw: List[Dict], standards: List[Dict]) -> List[CheckResult]:
    """Convert raw submitted results to CheckResult objects, filling gaps."""
    standard_map = {s["check_id"]: s for s in standards}
    results = []
    seen: set = set()

    for item in raw:
        cid = item.get("check_id", "")
        std = standard_map.get(cid)
        if not std or cid in seen:
            continue
        seen.add(cid)
        passed = bool(item.get("passed", False))
        results.append(CheckResult(
            check_id=cid,
            category=std["category"],
            passed=passed,
            detail=item.get("detail", ""),
            score=1.0 if passed else 0.0,
        ))

    for s in standards:
        if s["check_id"] not in seen:
            results.append(CheckResult(
                check_id=s["check_id"],
                category=s["category"],
                passed=False,
                detail="Check not completed by agent",
                score=0.0,
            ))

    return results


def _all_failed(reason: str, standards: List[Dict]) -> List[CheckResult]:
    return [
        CheckResult(
            check_id=s["check_id"],
            category=s["category"],
            passed=False,
            detail=reason,
            score=0.0,
        )
        for s in standards
    ]
