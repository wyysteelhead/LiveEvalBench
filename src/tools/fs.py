"""File-system helpers with atomic writes and snapshot support."""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path


def read(path: str | Path, start: int | None = None, end: int | None = None) -> str:
    """Read a file, optionally slicing by 1-based line numbers [start, end]."""
    text = Path(path).read_text(encoding="utf-8")
    if start is None and end is None:
        return text
    lines = text.splitlines(keepends=True)
    s = (start - 1) if start else 0
    e = end if end else len(lines)
    return "".join(lines[s:e])


def write(
    path: str | Path,
    content: str,
    create: bool = True,
    overwrite: bool = True,
) -> None:
    """Atomically write *content* to *path* via a temp file + os.replace."""
    p = Path(path)
    if not overwrite and p.exists():
        raise FileExistsError(f"{path} already exists and overwrite=False")
    if create:
        p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=p.parent, prefix=".tmp_")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(content)
        os.replace(tmp, p)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def apply_patch(patch_text: str) -> dict:
    """Apply a unified diff via the system `patch` command.

    Returns {'applied': bool, 'conflicts': list[str]}.
    """
    result = subprocess.run(
        ["patch", "-p1", "--batch", "--no-backup-if-mismatch"],
        input=patch_text,
        capture_output=True,
        text=True,
    )
    conflicts: list[str] = []
    if result.returncode != 0:
        conflicts = [
            line for line in result.stderr.splitlines() if "FAILED" in line or "reject" in line.lower()
        ]
    return {"applied": result.returncode == 0, "conflicts": conflicts}


def snapshot(label: str, base_dir: str | Path = ".runs/snapshots") -> str:
    """Create a timestamped metadata directory for *label*, return its path."""
    import datetime

    ts = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    safe_label = label.replace("/", "_").replace(" ", "_")
    snap_dir = Path(base_dir) / f"{ts}_{safe_label}"
    snap_dir.mkdir(parents=True, exist_ok=True)
    meta = snap_dir / "meta.json"
    import json

    write(
        meta,
        json.dumps({"label": label, "created_at": ts}, indent=2) + "\n",
    )
    return str(snap_dir)
