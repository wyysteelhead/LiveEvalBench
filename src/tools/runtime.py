"""Runtime detection and environment preparation."""
from __future__ import annotations

import shutil
from typing import Any

from .exec import run

# ---------------------------------------------------------------------------
# Default CI-friendly environment variables
# ---------------------------------------------------------------------------
DEFAULT_ENV: dict[str, str] = {
    "CI": "1",
    "NEXT_TELEMETRY_DISABLED": "1",
    "PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD": "1",
    "PUPPETEER_SKIP_DOWNLOAD": "1",
}


def _probe_version(binary: str) -> str | None:
    """Return version string for *binary*, or None if not found."""
    if not shutil.which(binary):
        return None
    result = run([binary, "--version"], timeout_s=10)
    if result["exit_code"] == 0:
        return result["stdout"].strip().lstrip("v")
    return None


def detect() -> dict[str, Any]:
    """Probe installed runtimes and return a detection dict."""
    node = _probe_version("node")
    npm = _probe_version("npm")
    pnpm = _probe_version("pnpm")
    yarn = _probe_version("yarn")

    if pnpm:
        package_manager = "pnpm"
    elif yarn:
        package_manager = "yarn"
    elif npm:
        package_manager = "npm"
    else:
        package_manager = None

    return {
        "node": node,
        "npm": npm,
        "pnpm": pnpm,
        "yarn": yarn,
        "package_manager": package_manager,
    }


def ensure(spec: dict) -> dict[str, Any]:
    """Ensure the runtime matches *spec*.

    spec keys (all optional):
      - node_version: str  (semver prefix, e.g. "18")
      - package_manager: str  ("npm" | "pnpm" | "yarn")

    If the requested package manager is missing, attempts `corepack enable`.
    Does NOT install Node (sudo/apt are forbidden).
    Returns {ok, detected, env, warnings}.
    """
    detected = detect()
    warnings: list[str] = []
    ok = True

    # Node check
    req_node = spec.get("node_version")
    if req_node and detected["node"]:
        if not detected["node"].startswith(req_node.lstrip("v")):
            warnings.append(
                f"node {detected['node']} does not match requested {req_node}"
            )
    elif req_node and not detected["node"]:
        warnings.append("node not found; cannot install without sudo/apt")
        ok = False

    # Package manager check
    req_pm = spec.get("package_manager")
    if req_pm and not detected.get(req_pm):
        # Try corepack enable
        result = run(["corepack", "enable"], timeout_s=30)
        if result["exit_code"] == 0:
            detected = detect()
            if not detected.get(req_pm):
                warnings.append(
                    f"{req_pm} still not found after corepack enable"
                )
                ok = False
        else:
            warnings.append(
                f"{req_pm} not found and corepack enable failed: "
                + result["stderr"][:200]
            )
            ok = False

    return {
        "ok": ok,
        "detected": detected,
        "env": {**DEFAULT_ENV},
        "warnings": warnings,
    }
