"""Tool-layer session state and constraint enforcement.

A session is keyed by artifacts_path (unique per evaluation run).
Tools call get_session() to read/update state and enforce constraints.

Build-phase constraints (enforced in build_tools.py):
  - local_exec_start with "npm run dev" requires install_success=True
  - local_exec_start port flags must come from allocated_ports
  - protocol_write_artifacts(state=ready) requires HTTP verification of app_url + cdp_url

Future browser-phase constraints can be added here without touching tool files.
"""
from __future__ import annotations

import contextvars
import re
import subprocess
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Dict, Optional, Set


@dataclass
class SessionState:
    """Mutable state tracked across tool calls within one evaluation run."""
    # Build phase
    workspace_root: str = ""
    install_success: bool = False
    allocated_ports: Set[int] = field(default_factory=set)
    # Extensible: browser phase state can be added here


_sessions: Dict[str, SessionState] = {}
_lock = threading.Lock()
_active_session_id: contextvars.ContextVar[str] = contextvars.ContextVar(
    "frontend_evaluator_active_session_id",
    default="",
)


def register_session(session_id: str, workspace_root: str = "") -> None:
    """Create a fresh session. Call before Phase 1 starts."""
    with _lock:
        _sessions[session_id] = SessionState(workspace_root=workspace_root)


def activate_session(session_id: str) -> contextvars.Token[str]:
    """Mark a session as active for the current async context."""
    return _active_session_id.set(session_id)


def reset_active_session(token: contextvars.Token[str]) -> None:
    """Restore the previous active session for the current async context."""
    _active_session_id.reset(token)


def get_active_session_id() -> str:
    """Return the current active session id, if any."""
    return _active_session_id.get()


def get_effective_session_id(session_id: str = "") -> str:
    """Prefer explicit session_id, otherwise fall back to the active session."""
    return session_id or get_active_session_id()


def clear_session(session_id: str) -> None:
    """Remove session. Call in finally block after evaluation."""
    with _lock:
        _sessions.pop(session_id, None)


def get_session_count() -> int:
    """Return the number of currently registered sessions (thread-safe)."""
    with _lock:
        return len(_sessions)


def get_session(session_id: str) -> Optional[SessionState]:
    """Return session state, or None if not registered."""
    return _sessions.get(session_id)


def get_effective_session(session_id: str = "") -> Optional[SessionState]:
    """Return the explicit session, or the active session when omitted."""
    effective_session_id = get_effective_session_id(session_id)
    if not effective_session_id:
        return None
    return get_session(effective_session_id)


def get_workspace_root(session_id: str = "") -> str:
    """Return the workspace root for a session, if available."""
    session = get_effective_session(session_id)
    return session.workspace_root if session is not None else ""


# ---------------------------------------------------------------------------
# Constraint checkers (called by tool functions)
# ---------------------------------------------------------------------------

def check_exec_start(session_id: str, cmd: str) -> Optional[str]:
    """Return an error string if cmd violates session constraints, else None.

    Constraints:
    1. 'npm run dev/start' requires install_success=True
       (exempted for static servers: python -m http.server, npx serve, etc.)
    2. --port / --remote-debugging-port values must be in allocated_ports
    """
    session = get_effective_session(session_id)
    if session is None:
        return None  # no session registered, no constraints applied

    cmd_lower = cmd.lower()

    # Constraint 1: npm run dev/start requires prior successful install
    # Static servers (python http.server, npx serve) are exempt
    _static_servers = ("python -m http.server", "python3 -m http.server", "npx serve", "npx http-server")
    is_static = any(s in cmd_lower for s in _static_servers)
    if not is_static and ("npm run dev" in cmd_lower or "npm run start" in cmd_lower):
        if not session.install_success:
            return (
                "constraint_violation: cannot start dev server before "
                "a successful 'npm install' (exit_code=0)"
            )

    # Constraint 2: port flags must come from allocated_ports
    import re
    for flag in ("--port", "--remote-debugging-port"):
        m = re.search(rf"{re.escape(flag)}[=\s]+(\d+)", cmd)
        if m:
            port = int(m.group(1))
            if port not in session.allocated_ports:
                return (
                    f"constraint_violation: port {port} was not allocated via "
                    f"local_port_allocate. Call local_port_allocate first."
                )

    return None


def check_write_artifacts_ready(
    app_url: str,
    cdp_url: str,
    session_id: str = "",
) -> Optional[str]:
    """Return an error string if app_url or cdp_url are not reachable, else None.

    On failure, gathers diagnostics (port liveness, process status) so the
    agent can distinguish "server still starting" from "server crashed" and
    take appropriate action without guesswork.
    """
    for label, url in (("app_url", app_url), ("cdp_url", cdp_url)):
        if not url:
            return f"constraint_violation: {label} is empty"

        # Force IPv4 — localhost can resolve to ::1 and fail when the
        # kernel has IPv6 disabled (common in containers).
        url = url.replace("//localhost:", "//127.0.0.1:")

        last_error: Exception | None = None
        http_error_status: int | None = None
        for attempt in range(1, 6):
            try:
                with urllib.request.urlopen(url, timeout=5):
                    last_error = None
                    http_error_status = None
                    break
            except urllib.error.HTTPError as exc:
                # Server IS running and responding — even a 4xx/5xx proves
                # connectivity.  Don't fail the check; the evaluator agents
                # will test the actual app content.
                http_error_status = exc.code
                last_error = None
                break
            except Exception as exc:
                last_error = exc
                if attempt < 6:
                    time.sleep(1.0)

        if last_error is not None:
            diag = _gather_unreachable_diagnostics(url, session_id)
            return (
                f"constraint_violation: {label} ({url}) is not reachable after "
                f"5 retries (5s each): {last_error}.\n"
                f"{diag}"
            )

    return None


def _gather_unreachable_diagnostics(url: str, session_id: str = "") -> str:
    """Build a diagnostic summary when *url* is unreachable.

    Checks port liveness and node/npm process status so the agent gets
    actionable information instead of a raw connection-refused error.
    """
    lines: list[str] = []
    port_listening: str | None = None  # cached result

    # 1. Extract port from the URL
    m = re.search(r":(\d{4,5})(?:/|$)", url)
    port = int(m.group(1)) if m else 0
    if port:
        port_listening = _check_port_listening(port) or None
        if port_listening:
            lines.append(f"- Port {port}: LISTENING ({port_listening})")
        else:
            lines.append(f"- Port {port}: NOT LISTENING (no process bound to this port)")

    # 2. Look for npm/node processes tied to this session's workspace
    workspace_root = get_workspace_root(session_id) if session_id else ""
    proc_info = _check_workspace_processes(workspace_root, port) if workspace_root else ""
    has_processes = bool(proc_info)
    if proc_info:
        lines.append(proc_info)
    elif workspace_root:
        lines.append("- npm/node processes in workspace: NONE FOUND")

    # 3. Suggest next action based on cached findings
    if port and port_listening is None:
        if workspace_root and not has_processes:
            lines.append(
                "- SUGGESTION: The dev server appears to have exited. "
                "Restart it with 'npm run dev' (or equivalent), then retry "
                "protocol_write_artifacts with state=ready."
            )
        elif workspace_root:
            lines.append(
                "- SUGGESTION: A build process is running but not yet listening "
                "on the port. Wait 10-20s and retry protocol_write_artifacts."
            )
    elif not lines:
        lines.append("- Unable to gather diagnostics. Check the dev server manually.")

    return "\n".join(lines) if lines else ""


def _check_port_listening(port: int) -> str:
    """Return a short description of what is listening on *port*, or ''."""
    try:
        out = subprocess.check_output(
            ["ss", "-tlnp", f"sport = :{port}"],
            stderr=subprocess.DEVNULL,
            timeout=3,
        )
        text = out.decode("utf-8", errors="replace").strip()
        if not text:
            return ""
        # Extract process info from the last column (users:((\"proc\",pid,fd)))
        proc_match = re.search(r'users:.*?"([^"]+)"', text)
        if proc_match:
            return proc_match.group(1)
        return text.splitlines()[-1].split()[-1] if text.splitlines() else ""
    except Exception:
        return ""


def _check_workspace_processes(workspace_root: str, port: int = 0) -> str:
    """Check for running npm/node processes tied to *workspace_root*.

    Returns a human-readable summary, or empty string if none found.
    """
    try:
        ps_out = subprocess.check_output(
            ["ps", "aux"],
            stderr=subprocess.DEVNULL,
            timeout=3,
        ).decode("utf-8", errors="replace")
    except Exception:
        return ""

    lines = ps_out.splitlines()
    matches: list[str] = []
    for line in lines:
        if workspace_root not in line:
            continue
        if not any(kw in line for kw in ("node", "npm", "npx", "pnpm", "yarn")):
            continue
        # Extract PID and command
        parts = line.split()
        if len(parts) >= 11:
            pid = parts[1]
            cmd = " ".join(parts[10:])
            matches.append(f"PID {pid}: {cmd[:120]}")

    if not matches:
        return ""

    count = len(matches)
    suffix = f" (port {port})" if port else ""
    header = f"- npm/node processes in workspace{suffix}: {count} running"
    details = "\n".join(f"    {m}" for m in matches[-5:])  # show last 5 at most
    return f"{header}\n{details}"


# ---------------------------------------------------------------------------
# State updaters (called by tool functions after successful operations)
# ---------------------------------------------------------------------------

def record_install_success(session_id: str) -> None:
    session = get_effective_session(session_id)
    if session is not None:
        session.install_success = True


def record_port_allocated(session_id: str, port: int) -> None:
    session = get_effective_session(session_id)
    if session is not None:
        session.allocated_ports.add(port)


def record_port_released(session_id: str, port: int) -> None:
    session = get_effective_session(session_id)
    if session is not None:
        session.allocated_ports.discard(port)
