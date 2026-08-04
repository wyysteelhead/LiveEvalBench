"""AgentBrowser CLI executor backend.

Uses the `agent-browser` CLI tool via CDP to control a browser.
This is an alternative to PlaywrightExecutor for environments where
agent-browser is available.

Limitation: reset() uses storage clear + navigate rather than a new
browser context, providing weaker isolation than PlaywrightExecutor.
"""

import asyncio
import base64
import json
import os
import tempfile
from typing import Any, Dict, Optional

from ..utils.subprocess import close_subprocess_transports


class AgentBrowserExecutor:
    """Executor that drives a browser via the agent-browser CLI over CDP.

    Requires `agent-browser` to be installed and accessible in PATH.
    The sandbox must expose a CDP endpoint (Chromium started with
    --remote-debugging-port=9222 --remote-debugging-address=0.0.0.0).
    """

    def __init__(self, sandbox: Any, app_url: str):
        """Initialize the executor.

        Args:
            sandbox: Sandbox instance implementing SandboxInterface
            app_url: URL of the application to test
        """
        self.sandbox = sandbox
        self.app_url = app_url
        self._cdp_url: Optional[str] = None
        self.service_started = False

    async def start_service(self, timeout: int = 60) -> None:
        """Start Chromium via sandbox and obtain the CDP URL.

        Args:
            timeout: Timeout in seconds for browser startup

        Raises:
            RuntimeError: If browser fails to start or CDP is unreachable
        """
        if self.service_started:
            return

        await self.sandbox.start_browser()
        self._cdp_url = self.sandbox.get_cdp_url()

        # Verify connectivity with a snapshot command
        try:
            await self.get_context()
        except Exception as e:
            raise RuntimeError(
                f"agent-browser CDP connection failed ({self._cdp_url}): {e}"
            )

        self.service_started = True

    async def _run(self, *args: str) -> Dict[str, Any]:
        """Run an agent-browser command and return parsed JSON output.

        Args:
            *args: Command arguments after `agent-browser --cdp <url>`

        Returns:
            Parsed JSON result dict

        Raises:
            RuntimeError: If the command fails or returns non-zero exit code
        """
        cmd = ["agent-browser", "--cdp", self._cdp_url, *args, "--json"]
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        close_subprocess_transports(proc)

        if proc.returncode != 0:
            raise RuntimeError(
                f"agent-browser command failed (exit {proc.returncode}): "
                f"{stderr.decode().strip()}"
            )

        try:
            return json.loads(stdout.decode())
        except json.JSONDecodeError as e:
            raise RuntimeError(
                f"agent-browser returned non-JSON output: {stdout.decode()[:500]}"
            ) from e

    async def navigate(self, url: str, timeout: int = 30000) -> Dict[str, Any]:
        """Navigate to a URL.

        Args:
            url: URL to navigate to
            timeout: Navigation timeout in milliseconds (unused by CLI, kept for interface compat)

        Returns:
            Result dict with url key
        """
        return await self._run("open", url)

    async def click(self, selector: str, timeout: int = 5000) -> Dict[str, Any]:
        """Click an element.

        Args:
            selector: Element selector
            timeout: Click timeout in milliseconds (unused by CLI)

        Returns:
            Result dict
        """
        return await self._run("click", selector)

    async def click_at(self, x: int, y: int, timeout: int = 5000) -> Dict[str, Any]:
        """Click at viewport coordinates in CSS pixels.

        Uses JS dispatch fallback because agent-browser CLI has no stable
        coordinate-click subcommand across versions.
        """
        x = int(x)
        y = int(y)
        js = f"""
(() => {{
  const x = {x};
  const y = {y};
  const target = document.elementFromPoint(x, y);
  if (!target) return {{ ok: false, error: 'no element at point' }};
  const events = ['pointerdown', 'mousedown', 'pointerup', 'mouseup', 'click'];
  for (const type of events) {{
    const evt = type.startsWith('pointer')
      ? new PointerEvent(type, {{ bubbles: true, cancelable: true, clientX: x, clientY: y, pointerType: 'mouse' }})
      : new MouseEvent(type, {{ bubbles: true, cancelable: true, clientX: x, clientY: y, view: window }});
    target.dispatchEvent(evt);
  }}
  return {{ ok: true }};
}})()
"""
        evaluated = await self.evaluate_js(js)
        result = evaluated.get("result", {})
        if isinstance(result, dict) and result.get("ok"):
            return {"success": True, "message": f"Clicked at: ({x}, {y})"}
        raise RuntimeError(f"Coordinate click failed: {result}")

        async def dblclick_at(self, x: int, y: int, timeout: int = 5000) -> Dict[str, Any]:
                """Double-click at viewport coordinates in CSS pixels."""
                x = int(x)
                y = int(y)
                js = f"""
(() => {{
    const x = {x};
    const y = {y};
    const target = document.elementFromPoint(x, y);
    if (!target) return {{ ok: false, error: 'no element at point' }};
    const events = ['pointerdown', 'mousedown', 'pointerup', 'mouseup', 'click', 'pointerdown', 'mousedown', 'pointerup', 'mouseup', 'click', 'dblclick'];
    for (const type of events) {{
        const evt = type.startsWith('pointer')
            ? new PointerEvent(type, {{ bubbles: true, cancelable: true, clientX: x, clientY: y, pointerType: 'mouse' }})
            : new MouseEvent(type, {{ bubbles: true, cancelable: true, clientX: x, clientY: y, detail: type === 'dblclick' ? 2 : 1, view: window }});
        target.dispatchEvent(evt);
    }}
    return {{ ok: true }};
}})()
"""
                evaluated = await self.evaluate_js(js)
                result = evaluated.get("result", {})
                if isinstance(result, dict) and result.get("ok"):
                        return {"success": True, "message": f"Double-clicked at: ({x}, {y})"}
                raise RuntimeError(f"Coordinate double-click failed: {result}")

    async def hover(self, selector: str, timeout: int = 5000) -> Dict[str, Any]:
        """Hover an element.

        Tries native agent-browser hover first, then falls back to dispatching
        mouse events in page JS if the CLI subcommand is unavailable.
        """
        try:
            return await self._run("hover", selector)
        except Exception:
            escaped = selector.replace("\\", "\\\\").replace("'", "\\'")
            js = f"""
(() => {{
  const sel = '{escaped}';
  let el = null;
  if (sel.startsWith('text=')) {{
    const text = sel.slice(5);
    const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_ELEMENT);
    while (walker.nextNode()) {{
      const node = walker.currentNode;
      if ((node.innerText || '').trim() === text.trim()) {{
        el = node;
        break;
      }}
    }}
  }} else {{
    el = document.querySelector(sel);
  }}
  if (!el) return {{ ok: false, error: 'selector not found' }};
  ['mouseover','mouseenter','mousemove'].forEach(type =>
    el.dispatchEvent(new MouseEvent(type, {{ bubbles: true, cancelable: true, view: window }}))
  );
  return {{ ok: true }};
}})()
"""
            eval_result = await self.evaluate_js(js)
            result = eval_result.get("result", {})
            if isinstance(result, dict) and result.get("ok"):
                return {"success": True, "message": f"Hovered: {selector} (js fallback)"}
            raise RuntimeError(f"Hover failed: {result}")

    async def type_text(self, selector: str, text: str, timeout: int = 5000) -> Dict[str, Any]:
        """Type text into an input field.

        Args:
            selector: Input element selector
            text: Text to type
            timeout: Timeout in milliseconds (unused by CLI)

        Returns:
            Result dict
        """
        return await self._run("fill", selector, text)

    async def get_context(self) -> Dict[str, Any]:
        """Get page context via accessibility snapshot.

        Returns:
            Result dict with accessibility tree
        """
        return await self._run("snapshot")

    async def evaluate_js(self, expression: str) -> Dict[str, Any]:
        """Execute JavaScript in the page context.

        Args:
            expression: JavaScript expression to evaluate

        Returns:
            Result dict with evaluation result
        """
        return await self._run("eval", expression)

    async def screenshot(self, full_page: bool = False) -> str:
        """Take a screenshot and return base64-encoded PNG.

        Args:
            full_page: Whether to capture the full scrollable page (not supported by CLI)

        Returns:
            Base64-encoded PNG string
        """
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            tmp_path = f.name

        try:
            await self._run("screenshot", tmp_path)
            with open(tmp_path, "rb") as f:
                return base64.b64encode(f.read()).decode()
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    async def reset(self, app_url: str) -> None:
        """Reset browser state by clearing storage and navigating to app_url.

        Note: This provides weaker isolation than PlaywrightExecutor's new-context
        approach. Cookies and service workers are not cleared.

        Args:
            app_url: URL to navigate to after clearing storage
        """
        await self.evaluate_js("localStorage.clear(); sessionStorage.clear();")
        await self.navigate(app_url)

    async def shutdown(self) -> None:
        """Close the browser via agent-browser CLI."""
        if not self.service_started:
            return
        try:
            await self._run("close")
        except Exception:
            pass
        self.service_started = False


def get_executor(executor_backend: str, sandbox: Any, app_url: str):
    """Factory function to create the appropriate executor.

    Args:
        executor_backend: "playwright" or "agent-browser"
        sandbox: Sandbox instance
        app_url: Application URL

    Returns:
        Executor instance implementing ExecutorInterface
    """
    if executor_backend == "agent-browser":
        return AgentBrowserExecutor(sandbox, app_url)

    # Prefer commands.run-backed Playwright service when available
    # (sandbox-style API), otherwise use CDP-based Playwright.
    core_sandbox = sandbox.sandbox if hasattr(sandbox, "sandbox") else sandbox
    commands = getattr(core_sandbox, "commands", None)
    supports_playwright_service = callable(getattr(commands, "run", None))

    if supports_playwright_service:
        print("[Executor] Using Playwright service backend (commands.run)")
        from .playwright_executor import PlaywrightExecutor

        return PlaywrightExecutor(sandbox, app_url)

    has_cdp_capabilities = callable(getattr(core_sandbox, "start_browser", None)) and callable(
        getattr(core_sandbox, "get_cdp_url", None)
    )
    if has_cdp_capabilities:
        print("[Executor] Using Playwright CDP backend (start_browser/get_cdp_url)")
        from .cdp_playwright_executor import CdpPlaywrightExecutor

        return CdpPlaywrightExecutor(sandbox, app_url)

    raise RuntimeError(
        "EXECUTOR_BACKEND=playwright requires either sandbox.commands.run "
        "(Playwright service mode) or start_browser/get_cdp_url "
        "(Playwright CDP mode), but current sandbox backend provides neither."
    )
