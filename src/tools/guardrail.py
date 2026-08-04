"""Command guardrail — risk classification and domain allow-listing."""
from __future__ import annotations

import datetime
import shlex
from typing import Sequence

# ---------------------------------------------------------------------------
# Risk levels
# ---------------------------------------------------------------------------
AUTO_APPROVE = "auto_approve"
REQUIRE_CONFIRM = "require_confirm"
REJECT = "reject"

# ---------------------------------------------------------------------------
# Forbidden commands (always rejected regardless of arguments)
# ---------------------------------------------------------------------------
FORBIDDEN_COMMANDS: frozenset[str] = frozenset(
    {"sudo", "apt", "apt-get", "dnf", "brew", "docker"}
)

# ---------------------------------------------------------------------------
# Command policy: first token → risk level
# ---------------------------------------------------------------------------
COMMAND_POLICY: dict[str, str] = {
    # package managers — safe read operations
    "npm": AUTO_APPROVE,
    "npx": AUTO_APPROVE,
    "pnpm": AUTO_APPROVE,
    "yarn": AUTO_APPROVE,
    "corepack": AUTO_APPROVE,
    "node": AUTO_APPROVE,
    # version probes
    "which": AUTO_APPROVE,
    "env": AUTO_APPROVE,
    # file inspection
    "cat": AUTO_APPROVE,
    "ls": AUTO_APPROVE,
    "find": AUTO_APPROVE,
    "grep": AUTO_APPROVE,
    "head": AUTO_APPROVE,
    "tail": AUTO_APPROVE,
    "wc": AUTO_APPROVE,
    "stat": AUTO_APPROVE,
    "echo": AUTO_APPROVE,
    "pwd": AUTO_APPROVE,
    # build / test runners
    "next": AUTO_APPROVE,
    "tsc": AUTO_APPROVE,
    "eslint": AUTO_APPROVE,
    "prettier": AUTO_APPROVE,
    "jest": AUTO_APPROVE,
    "vitest": AUTO_APPROVE,
    "playwright": AUTO_APPROVE,
    # network probes (read-only)
    "curl": REQUIRE_CONFIRM,
    "wget": REQUIRE_CONFIRM,
    "ping": REQUIRE_CONFIRM,
    # process management
    "kill": REQUIRE_CONFIRM,
    "pkill": REQUIRE_CONFIRM,
    # shell builtins that can be risky
    "rm": REQUIRE_CONFIRM,
    "mv": REQUIRE_CONFIRM,
    "cp": REQUIRE_CONFIRM,
    "chmod": REQUIRE_CONFIRM,
    "chown": REQUIRE_CONFIRM,
    # always forbidden
    "sudo": REJECT,
    "apt": REJECT,
    "apt-get": REJECT,
    "dnf": REJECT,
    "brew": REJECT,
    "docker": REJECT,
}


_INSTALL_SUBCOMMANDS: frozenset[str] = frozenset(
    {"install", "i", "ci", "add", "install-test", "install-ci-test"}
)
_INSTALL_MANAGERS: frozenset[str] = frozenset({"npm", "pnpm", "yarn", "bun"})


def is_install_command(cmd: str | Sequence[str]) -> bool:
    """Return True if cmd is a package-manager install/add operation."""
    if isinstance(cmd, str):
        try:
            parts = shlex.split(cmd)
        except ValueError:
            parts = cmd.split()
    else:
        parts = [str(p) for p in cmd]
    parts = [p for p in parts if p]
    if len(parts) < 2:
        return False
    if parts[0].lower() not in _INSTALL_MANAGERS:
        return False
    # Skip leading flags to find the subcommand
    subcmd = next((p for p in parts[1:] if not p.startswith("-")), "")
    return subcmd.lower() in _INSTALL_SUBCOMMANDS


def _first_token(cmd: str | Sequence[str]) -> str:
    if isinstance(cmd, str):
        parts = shlex.split(cmd)
    else:
        parts = list(cmd)
    return parts[0] if parts else ""


def check_command(cmd: str | Sequence[str]) -> str:
    """Return risk level string, or raise ValueError for REJECT commands."""
    token = _first_token(cmd)
    if token in FORBIDDEN_COMMANDS:
        raise ValueError(f"Command '{token}' is forbidden by guardrail policy")
    level = COMMAND_POLICY.get(token, REQUIRE_CONFIRM)
    if level == REJECT:
        raise ValueError(f"Command '{token}' is rejected by guardrail policy")
    return level


# ---------------------------------------------------------------------------
# Domain allow-listing
# ---------------------------------------------------------------------------
_BENCHMARK_ALLOWLIST: frozenset[str] = frozenset(
    {
        "registry.npmjs.org",
        "registry.yarnpkg.com",
        "dl.yarnpkg.com",
        "github.com",
        "raw.githubusercontent.com",
        "objects.githubusercontent.com",
    }
)


def is_domain_allowed(
    url: str,
    mode: str = "benchmark",
    allowlist: frozenset[str] | None = None,
) -> bool:
    """Return True if the URL's host is permitted under the given mode.

    mode='benchmark' — strict: only _BENCHMARK_ALLOWLIST (or custom allowlist)
    mode='open'      — permissive: everything allowed
    """
    if mode == "open":
        return True
    from urllib.parse import urlparse

    host = urlparse(url).hostname or ""
    effective = allowlist if allowlist is not None else _BENCHMARK_ALLOWLIST
    return host in effective


# ---------------------------------------------------------------------------
# Audit log
# ---------------------------------------------------------------------------
def audit_log_entry(
    cmd: str | Sequence[str],
    risk: str,
    cwd: str = "",
) -> dict:
    """Return a structured audit log entry dict."""
    return {
        "ts": datetime.datetime.utcnow().isoformat() + "Z",
        "cmd": cmd if isinstance(cmd, str) else list(cmd),
        "risk": risk,
        "cwd": cwd,
    }
