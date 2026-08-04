"""Sandbox & local execution module.

The public benchmark runs LOCALLY via Playwright (EXECUTOR_BACKEND=playwright,
the default) — no sandbox service required. Remote sandbox providers (E2B /
Docker / Remote / OpenSandbox) were removed from the public release; the
`get_sandbox_environment` factory is kept as a stub that raises if called.
"""

from .interface import SandboxInterface
from .factory import get_sandbox_environment
from .browser_connector import BrowserConnector
from .executor_interface import ExecutorInterface
from .agent_browser_executor import AgentBrowserExecutor, get_executor

__all__ = [
    "SandboxInterface",
    "get_sandbox_environment",
    "BrowserConnector",
    "ExecutorInterface",
    "AgentBrowserExecutor",
    "get_executor",
    "apply_font_load_timeout",
]

# JavaScript that patches document.fonts.ready to resolve within 30 seconds
# regardless of font loading status.  This prevents Playwright's
# page.screenshot() from hanging for the default 30s waiting for fonts
# that may never load (e.g. from a slow or unreachable CDN).
# The patch shadows the FontFaceSet.ready getter on the specific instance
# so the rest of the FontFaceSet API remains intact.
_FONT_LOAD_TIMEOUT_SCRIPT = """
(() => {
  const fonts = document.fonts;
  if (!fonts) return;
  const origReady = fonts.ready;
  const timeout = new Promise(resolve => setTimeout(resolve, 30000));
  Object.defineProperty(fonts, 'ready', {
    get: () => Promise.race([origReady, timeout]),
    configurable: true,
  });
})();
"""


async def apply_font_load_timeout(context) -> None:
    """Patch document.fonts.ready on the given context so screenshots won't
    hang on font loading.  Call right after BrowserContext creation."""
    await context.add_init_script(_FONT_LOAD_TIMEOUT_SCRIPT)
