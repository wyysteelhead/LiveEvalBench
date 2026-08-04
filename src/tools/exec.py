"""Command execution with guardrail enforcement."""
from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import time
from contextlib import suppress
from pathlib import Path
from typing import Any, Sequence

from .guardrail import audit_log_entry, check_command


def _ensure_str(value: Any, default: str = "") -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, str):
        return value
    return str(value) if value is not None else default


def _command_label(cmd: str | Sequence[str]) -> str:
    raw = cmd if isinstance(cmd, str) else " ".join(str(part) for part in cmd)
    label = re.sub(r"[^a-zA-Z0-9]+", "_", raw).strip("_").lower()
    return label[:48] or "process"


def _prepare_log_paths(
    cmd: str | Sequence[str],
    cwd: str,
    log_dir: str | None,
) -> tuple[Path, Path]:
    root = Path(log_dir) if log_dir else Path(cwd) / ".process_logs"
    root.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    prefix = f"{stamp}_{_command_label(cmd)}"
    return root / f"{prefix}.stdout.log", root / f"{prefix}.stderr.log"


def _with_log_hint(message: str, stdout_path: Path, stderr_path: Path) -> str:
    return f"{message}. stdout_log={stdout_path} stderr_log={stderr_path}"


def run(
    cmd: str | Sequence[str],
    cwd: str = ".",
    env: dict[str, str] | None = None,
    timeout_s: int = 120,
    pty: bool = False,
) -> dict[str, Any]:
    """Run *cmd* synchronously after guardrail check.

    Returns {stdout, stderr, exit_code, duration_ms}.
    Writes a JSON audit line to stderr of the *calling* process.
    """
    risk = check_command(cmd)  # raises ValueError on REJECT
    entry = audit_log_entry(cmd, risk, cwd)
    print(json.dumps(entry), file=sys.stderr)

    merged_env = {**os.environ, **(env or {})}
    # Suppress ANSI color codes from child processes so stdout/stderr captured
    # in tool results stay readable for the LLM. Caller-supplied env wins.
    for _key, _val in (("NO_COLOR", "1"), ("FORCE_COLOR", "0")):
        merged_env.setdefault(_key, _val)
    shell = isinstance(cmd, str)

    t0 = time.monotonic()
    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        env=merged_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        shell=shell,
        start_new_session=True,  # isolate process group so killpg reaches all children
    )
    try:
        stdout, stderr = proc.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        # Kill the entire process group so child processes don't keep the pipes open,
        # which would cause communicate() to block forever after the parent is killed.
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except OSError:
            with suppress(OSError):
                proc.kill()
        with suppress(Exception):
            stdout, stderr = proc.communicate(timeout=5)
        stdout = stdout or ""
        stderr = stderr or ""
        duration_ms = int((time.monotonic() - t0) * 1000)
        return {
            "stdout": _ensure_str(stdout),
            "stderr": _ensure_str(stderr),
            "exit_code": -1,
            "duration_ms": duration_ms,
            "error": "timeout",
        }
    duration_ms = int((time.monotonic() - t0) * 1000)
    return {
        "stdout": _ensure_str(stdout),
        "stderr": _ensure_str(stderr),
        "exit_code": proc.returncode,
        "duration_ms": duration_ms,
    }


def start(
    cmd: str | Sequence[str],
    cwd: str = ".",
    env: dict[str, str] | None = None,
    readiness: dict | None = None,
    log_dir: str | None = None,
) -> dict[str, Any]:
    """Start *cmd* non-blocking (Popen). Optionally poll an HTTP readiness URL.

    readiness = {"url": str, "timeout_s": int, "interval_s": float}
    Returns {pid, started_at, stdout_path, stderr_path}.
    """
    risk = check_command(cmd)
    entry = audit_log_entry(cmd, risk, cwd)
    print(json.dumps(entry), file=sys.stderr)

    merged_env = {**os.environ, **(env or {})}
    for _key, _val in (("NO_COLOR", "1"), ("FORCE_COLOR", "0")):
        merged_env.setdefault(_key, _val)
    shell = isinstance(cmd, str)
    stdout_path, stderr_path = _prepare_log_paths(cmd, cwd, log_dir)
    stdout_handle = stdout_path.open("ab")
    stderr_handle = stderr_path.open("ab")
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=cwd,
            env=merged_env,
            shell=shell,
            stdout=stdout_handle,
            stderr=stderr_handle,
            start_new_session=True,  # isolate process group so killpg reaches all children (chromium zygote/renderers/etc.)
        )
    finally:
        stdout_handle.close()
        stderr_handle.close()
    started_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    if readiness:
        import urllib.request

        url = readiness["url"]
        deadline = time.monotonic() + readiness.get("timeout_s", 60)
        interval = readiness.get("interval_s", 1.0)
        last_error: Exception | None = None
        ready = False
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                raise RuntimeError(
                    _with_log_hint(
                        f"process exited before readiness check completed with exit code {proc.returncode}",
                        stdout_path,
                        stderr_path,
                    )
                )
            try:
                with urllib.request.urlopen(url, timeout=2):
                    ready = True
                    break
            except Exception as exc:
                last_error = exc
                time.sleep(interval)

        if not ready:
            # Kill the whole process group — chromium spawns many children that
            # would otherwise leak as orphans and hold inotify/fd indefinitely.
            with suppress(Exception):
                os.killpg(proc.pid, signal.SIGTERM)
                proc.wait(timeout=3)
            if proc.poll() is None:
                with suppress(Exception):
                    os.killpg(proc.pid, signal.SIGKILL)
            message = f"timed out waiting for readiness URL {url}"
            if last_error is not None:
                message = f"{message}: {last_error}"
            raise TimeoutError(_with_log_hint(message, stdout_path, stderr_path))

    return {
        "pid": proc.pid,
        "started_at": started_at,
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
    }


def plan(steps: list[dict]) -> dict[str, Any]:
    """Dry-run validate a list of step dicts without executing anything.

    Each step: {"cmd": str | list, "cwd": str (optional)}
    Returns {normalized_plan, guardrail_report}.
    """
    normalized: list[dict] = []
    report: list[dict] = []
    for i, step in enumerate(steps):
        cmd = step.get("cmd", "")
        cwd = step.get("cwd", ".")
        try:
            risk = check_command(cmd)
            status = "ok"
            message = ""
        except ValueError as exc:
            risk = "reject"
            status = "rejected"
            message = str(exc)
        normalized.append({"index": i, "cmd": cmd, "cwd": cwd, "risk": risk})
        report.append({"index": i, "status": status, "risk": risk, "message": message})
    return {"normalized_plan": normalized, "guardrail_report": report}
