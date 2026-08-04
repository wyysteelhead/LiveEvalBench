"""Read-only source inspection tools for agentic evaluators."""

from __future__ import annotations

from contextvars import ContextVar
import re
from typing import Dict

from .registry import register_tool


_SOURCE_FILES: ContextVar[Dict[str, str]] = ContextVar("source_reader_source_files", default={})


def set_source_files(files: Dict[str, str]) -> None:
    """Set in-memory source files for read-only source inspection tools."""
    _SOURCE_FILES.set(dict(files or {}))


def clear_source_files() -> None:
    """Clear in-memory source files."""
    _SOURCE_FILES.set({})


def _current_source_files() -> Dict[str, str]:
    return _SOURCE_FILES.get()


@register_tool(
    name="read_file",
    description=(
        "Read a source file from the current evaluation input (read-only). "
        "Use this to inspect implementation details and verify behavior."
    ),
)
async def read_file(path: str, max_chars: int = 12000) -> str:
    """Read a file by path from the in-memory source bundle.

    Supports both relative paths (e.g. ``"app/page.tsx"``) and absolute
    paths (e.g. ``"/tmp/eval_open_xxx/index.html"``).  When the exact path
    is not found the tool falls back to basename matching.
    """
    source_files = _current_source_files()
    content = source_files.get(path)
    if content is None:
        # Fallback 1: try just the filename (basename)
        basename = path.rsplit("/", 1)[-1]
        content = source_files.get(basename)
    if content is None:
        # Fallback 2: try matching any key that ends with this path
        for key, value in source_files.items():
            if key.endswith(path) or path.endswith(key):
                content = value
                break
    if content is None:
        return f"File not found: {path}"

    limit = max(200, min(max_chars, 50000))
    if len(content) <= limit:
        return content
    return content[:limit] + f"\n\n... [truncated, total_chars={len(content)}]"


@register_tool(
    name="search_source",
    description=(
        "Search source files using a regex pattern or literal substring (read-only). "
        "If the pattern is not a valid regex it is treated as a literal search. "
        "Returns matching lines with file path and line number."
    ),
)
async def search_source(pattern: str, max_results: int = 30) -> str:
    """Search all source files for regex matches."""
    source_files = _current_source_files()
    if not source_files:
        return "No source files available in context."

    try:
        regex = re.compile(pattern)
    except re.error as exc:
        return f"Invalid regex pattern: {exc}"

    limit = max(1, min(max_results, 200))
    results = []
    for path, content in source_files.items():
        for i, line in enumerate(content.splitlines(), start=1):
            if regex.search(line):
                results.append(f"{path}:{i}: {line[:300]}")
                if len(results) >= limit:
                    break
        if len(results) >= limit:
            break

    if not results:
        return f"No matches found for pattern: {pattern}"
    return "\n".join(results)

