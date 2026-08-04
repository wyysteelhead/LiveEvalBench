"""Sandbox factory — STUB (public release, local-only).

The public benchmark runs locally via Playwright (EXECUTOR_BACKEND=playwright,
the default) and does not use a sandbox service. Remote sandbox providers
(E2B / Docker / Remote / OpenSandbox) were removed from the public release.
This factory is kept only so legacy `from ..sandbox.factory import
get_sandbox_environment` imports (in the api/batch/agentic/planner modules)
do not break at import time. Calling it raises — the local eval flow
(`eval_open.py` → `open_core` → `CdpPlaywrightExecutor`) never calls it.
"""

from typing import Optional

from .interface import SandboxInterface


def get_sandbox_environment(
    provider: Optional[str] = None,
    **kwargs,
) -> SandboxInterface:
    """Removed in the public (local-only) release.

    The local eval flow uses EXECUTOR_BACKEND=playwright and never calls this.
    If you need a remote sandbox, vendor the removed provider modules back in.
    """
    raise NotImplementedError(
        "Remote sandbox providers (E2B/Docker/Remote/OpenSandbox) are not "
        "included in the public release. Run the benchmark locally with "
        "EXECUTOR_BACKEND=playwright (default); 'eval_open.py' does not use "
        "a sandbox service."
    )