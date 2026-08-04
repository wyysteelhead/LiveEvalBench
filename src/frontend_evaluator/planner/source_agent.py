"""Source code LLM agent — checks ambiguous source patterns."""

import re
from typing import Any, Dict, List

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from .rubric_config import get_source_agent_standards
from .source_scanner import CheckResult

_TOOLS = [
    {
        "name": "read_file",
        "description": "Read the full content of a source file.",
        "parameters": {
            "type": "object",
            "properties": {
                "filename": {
                    "type": "string",
                    "description": "The filename to read (as it appears in the file list).",
                }
            },
            "required": ["filename"],
        },
    },
    {
        "name": "search_source",
        "description": "Search all source files for a regex pattern. Returns matching lines with context.",
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Regex pattern to search for.",
                },
                "context_lines": {
                    "type": "integer",
                    "description": "Number of lines of context around each match (default: 2).",
                },
            },
            "required": ["pattern"],
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
                    "description": "List of check results, one per standard.",
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


async def run(files: Dict[str, str], llm: Any) -> List[CheckResult]:
    """Run source agent checks on the provided files.

    Args:
        files: Dict mapping filename to file content
        llm: LangChain LLM instance

    Returns:
        List of CheckResult for each source standard
    """
    standards = get_source_agent_standards()
    if not files:
        return _all_failed("No source files provided", standards=standards)
    if not standards:
        return []

    llm_with_tools = llm.bind_tools(_TOOLS)

    file_list = "\n".join(f"- {name}" for name in files)
    standards_block = "\n\n".join(
        f"check_id: {s['check_id']}\ncategory: {s['category']}\ndescription: {s['description']}"
        for s in standards
    )

    messages = [
        SystemMessage(content=(
            "You are a frontend code quality auditor. Inspect source files to check "
            "specific frontend standards. Use read_file and search_source to gather evidence, "
            "then call submit_results with your findings for ALL standards listed. "
            "Only fail a check if you find clear evidence of a violation."
        )),
        HumanMessage(content=(
            f"Available files:\n{file_list}\n\n"
            f"Check each of the following standards and submit results for ALL of them:\n\n"
            f"{standards_block}\n\n"
            "Use read_file and search_source to inspect the code, then call submit_results."
        )),
    ]

    for _ in range(30):
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

            if name == "read_file":
                filename = args.get("filename", "")
                content = files.get(filename)
                if content is None:
                    result = f"File not found: {filename}. Available: {list(files.keys())}"
                else:
                    result = content[:8000]
            elif name == "search_source":
                pattern = args.get("pattern", "")
                ctx_lines = int(args.get("context_lines", 2))
                result = _search_source(files, pattern, ctx_lines)
            elif name == "submit_results":
                submitted = args.get("results", [])
                result = "Results submitted."
            else:
                result = f"Unknown tool: {name}"

            tool_messages.append(ToolMessage(content=str(result), tool_call_id=tid))

        messages.extend(tool_messages)

        if submitted is not None:
            return _build_check_results(submitted, standards=standards)

    return _all_failed("Source agent did not complete checks", standards=standards)


def _search_source(files: Dict[str, str], pattern: str, context_lines: int = 2) -> str:
    """Search all source files for a regex pattern."""
    try:
        compiled = re.compile(pattern, re.IGNORECASE)
    except re.error as e:
        return f"Invalid regex: {e}"

    matches = []
    for filename, content in files.items():
        lines = content.splitlines()
        for i, line in enumerate(lines):
            if compiled.search(line):
                start = max(0, i - context_lines)
                end = min(len(lines), i + context_lines + 1)
                ctx = "\n".join(
                    f"{'>>>' if j == i else '   '} {j + 1}: {lines[j]}"
                    for j in range(start, end)
                )
                matches.append(f"[{filename}:{i + 1}]\n{ctx}")

    if not matches:
        return "No matches found."
    return "\n\n".join(matches[:20])


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
