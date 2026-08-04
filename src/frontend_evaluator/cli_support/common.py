from __future__ import annotations

import inspect
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, Sequence

from frontend_evaluator.parser.artifact_parser import ArtifactParser


def load_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw.strip():
            continue
        try:
            row = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL at line {line_no}: {exc}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"Invalid JSONL row at line {line_no}: expected object")
        rows.append(row)
    return rows


def files_from_row(row: dict[str, Any]) -> dict[str, str]:
    code_blob = row.get("code")
    if isinstance(code_blob, str) and code_blob.strip():
        files = ArtifactParser.parse(code_blob)
        if files:
            return files

    files_obj = row.get("files")
    if isinstance(files_obj, dict):
        return {str(path): str(content) for path, content in files_obj.items() if isinstance(path, str)}

    return {}


def files_from_fixture_dir(fixture_dir: Path) -> dict[str, str]:
    return {
        str(path.relative_to(fixture_dir)): path.read_text(encoding="utf-8")
        for path in fixture_dir.rglob("*")
        if path.is_file()
    }


def files_from_markdown(path: Path) -> dict[str, str]:
    return ArtifactParser.parse(path.read_text(encoding="utf-8"))


def configure_logger_levels(logger: Any, log_level: str) -> None:
    logger.setLevel(log_level.upper())
    for handler in logger.handlers:
        handler.setLevel(log_level.upper())


def default_output_path_for_mode(mode: str) -> Path:
    if mode == "agentic":
        return Path("artifacts/reports/agentic_batch_report.json")
    return Path("artifacts/reports/legacy_batch_report.json")


def summarize_status_rows(rows: Sequence[dict[str, Any]]) -> dict[str, int]:
    summary = {"total": len(rows), "passed": 0, "partial": 0, "failed": 0}
    for row in rows:
        status = str(row.get("status", "")).lower()
        if status == "passed":
            summary["passed"] += 1
        elif status == "partial":
            summary["partial"] += 1
        else:
            summary["failed"] += 1
    return summary


def build_batch_report(
    *,
    mode: str,
    source: str,
    results: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    rows = list(results)
    return {
        "mode": mode,
        "source": source,
        "summary": summarize_status_rows(rows),
        "results": rows,
    }


def _extract_resume_rows(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]

    if isinstance(payload, dict):
        rows = payload.get("results")
        if isinstance(rows, list):
            return [row for row in rows if isinstance(row, dict)]

    return []


def _looks_like_jsonl(path: Path) -> bool:
    """Heuristic: if the file starts with ``{`` and contains multiple lines
    each starting with ``{``, it is likely JSONL rather than a JSON array."""
    try:
        with open(path, "rb") as fh:
            head = fh.read(4096)
        text = head.decode("utf-8")
        lines = [ln for ln in text.splitlines() if ln.strip()]
        return len(lines) >= 2 and all(ln.strip().startswith("{") for ln in lines[:5])
    except Exception:
        return False


def _build_index_from_jsonl_stream(
    path: Path,
    keys: list[str],
) -> dict[str, dict[str, Any]]:
    """Build a resume index from a JSONL file by streaming line-by-line.

    Unlike ``_read_resume_rows_jsonl`` this never materialises the full
    file in memory — only the *unique* completed rows are kept (in the
    returned index dict).  For reports with embedded source code the
    per-line JSON can be 10–50 MB, so this is critical for large files.
    """
    index: dict[str, dict[str, Any]] = {}
    with path.open("rb") as fh:
        for raw_bytes in fh:
            raw = raw_bytes.decode("utf-8").strip()
            if not raw:
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, dict):
                continue
            for key in keys:
                row_id = str(row.get(key, "")).strip()
                if row_id:
                    index[row_id] = row  # last occurrence wins
                    break
    return index


def load_resume_index(
    path: Path,
    id_keys: str | Iterable[str] = "id",
    mode: str | None = None,
) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}

    # JSONL format — one JSON object per line, each is a completed row
    if path.suffix == ".jsonl" or _looks_like_jsonl(path):
        keys = [id_keys] if isinstance(id_keys, str) else list(id_keys)
        return _build_index_from_jsonl_stream(path, keys)

    # Legacy JSON array / object format
    payload = json.loads(path.read_text(encoding="utf-8"))
    if mode and isinstance(payload, dict) and payload.get("mode") not in {None, mode}:
        return {}

    keys = [id_keys] if isinstance(id_keys, str) else list(id_keys)
    rows = _extract_resume_rows(payload)

    index: dict[str, dict[str, Any]] = {}
    for row in rows:
        for key in keys:
            row_id = str(row.get(key, "")).strip()
            if row_id:
                index[row_id] = row
                break
    return index


def maybe_await(callback_result: Any) -> Any:
    if inspect.isawaitable(callback_result):
        return callback_result
    return None


def write_json_output(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=path.parent, prefix=".tmp_")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def find_latest_checkpoint(output_path: Path) -> Path | None:
    """Return the newest checkpoint file.

    The canonical checkpoint is now a JSONL file written directly at
    *output_path*.  If it exists and has data it is returned immediately;
    otherwise fall back to legacy ``{stem}.ckpt-*.{suffix}`` files.
    """
    if output_path.exists() and output_path.stat().st_size > 0:
        return output_path
    if output_path.suffix != ".jsonl":
        jsonl = output_path.with_suffix(".jsonl")
        if jsonl.exists() and jsonl.stat().st_size > 0:
            return jsonl

    candidates = sorted(
        output_path.parent.glob(f"{output_path.stem}.ckpt-*{output_path.suffix}"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for candidate in candidates:
        if candidate.stat().st_size > 0:
            return candidate
    return None
