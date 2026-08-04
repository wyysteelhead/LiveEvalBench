"""Protocol helpers: manifest/artifacts validation, atomic JSON I/O, state polling."""
from __future__ import annotations

import copy
import json
import time
from pathlib import Path
from typing import Any

from .fs import write as _atomic_write


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_manifest(data: dict) -> list[str]:
    """Return list of validation error strings (empty = valid)."""
    errors: list[str] = []
    required_top = ["name", "engines", "package_manager", "commands", "env", "ports_pref"]
    for field in required_top:
        if field not in data:
            errors.append(f"missing required field: {field}")
    if "engines" in data and "node" not in data["engines"]:
        errors.append("missing required field: engines.node")
    if "commands" in data and "dev" not in data["commands"]:
        errors.append("missing required field: commands.dev")
    return errors


def validate_artifacts(data: dict) -> list[str]:
    """Return list of validation error strings (empty = valid)."""
    errors: list[str] = []
    required_top = ["state", "revision", "last_updated"]
    for field in required_top:
        if field not in data:
            errors.append(f"missing required field: {field}")
    state = data.get("state")
    if state == "ready":
        ports = data.get("ports", [])
        if not ports:
            errors.append("state=ready requires at least one entry in ports[]")
        for i, p in enumerate(ports):
            if "port" not in p:
                errors.append(f"ports[{i}] missing required field: port")
            if "url" not in p:
                errors.append(f"ports[{i}] missing required field: url")
    return errors


# ---------------------------------------------------------------------------
# Atomic JSON I/O
# ---------------------------------------------------------------------------

def write_json_atomic(
    path: str | Path,
    data: dict,
    bump_revision: bool = True,
) -> None:
    """Write *data* to *path* atomically, optionally incrementing revision."""
    import datetime

    out = copy.deepcopy(data)
    if bump_revision:
        out["revision"] = int(out.get("revision", 0)) + 1
    out["last_updated"] = datetime.datetime.utcnow().isoformat() + "Z"
    _atomic_write(path, json.dumps(out, indent=2) + "\n")


def read_json(path: str | Path) -> dict:
    """Read JSON from *path*. Returns {} if file does not exist."""
    p = Path(path)
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def wait_for_state(
    path: str | Path,
    desired: str,
    timeout_s: float = 120,
    poll_ms: int = 500,
) -> dict:
    """Poll *path* until data['state'] == *desired*.

    Raises RuntimeError if state becomes 'failed'.
    Raises TimeoutError on timeout.
    """
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        data = read_json(path)
        state = data.get("state")
        if state == desired:
            return data
        if state == "failed":
            raise RuntimeError(
                f"Artifacts reached failed state: {data.get('error', 'unknown')}"
            )
        time.sleep(poll_ms / 1000)
    raise TimeoutError(
        f"Timed out waiting for state={desired!r} in {path} after {timeout_s}s"
    )


def _deep_merge(base: dict, patch: dict) -> dict:
    result = copy.deepcopy(base)
    for k, v in patch.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = copy.deepcopy(v)
    return result


def update_artifacts(
    path: str | Path,
    patch: dict,
    bump_revision: bool = True,
) -> dict:
    """Read artifacts JSON, deep-merge *patch*, atomically write back. Returns merged dict."""
    existing = read_json(path)
    merged = _deep_merge(existing, patch)
    write_json_atomic(path, merged, bump_revision=bump_revision)
    return read_json(path)
