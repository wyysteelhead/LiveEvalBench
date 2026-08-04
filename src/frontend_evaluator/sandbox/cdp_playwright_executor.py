"""Playwright executor that connects to existing Chromium via CDP."""

import base64
import json
import os
import queue
import threading
from datetime import datetime
from typing import Any, Dict

from frontend_evaluator.utils.logger import logger
from . import apply_font_load_timeout
from .browser_connector import BrowserConnector

# ---------------------------------------------------------------------------
# Background log writer — keeps NFS writes off the asyncio event loop.
# Sync callbacks (Playwright page event listeners) call _debug_log() which
# only does put_nowait() into this queue; a daemon thread handles the actual
# file I/O so the event loop is never blocked.
# ---------------------------------------------------------------------------
_log_queue: queue.Queue = queue.Queue()
_writer_thread: threading.Thread | None = None
_writer_lock = threading.Lock()


def _log_writer_loop() -> None:
    while True:
        item = _log_queue.get()
        if item is None:
            break
        path, record = item
        try:
            log_dir = os.path.dirname(path)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=True) + "\n")
        except Exception:
            pass
        finally:
            _log_queue.task_done()


def _ensure_writer_started() -> None:
    global _writer_thread
    if _writer_thread is not None and _writer_thread.is_alive():
        return
    with _writer_lock:
        if _writer_thread is not None and _writer_thread.is_alive():
            return
        _writer_thread = threading.Thread(target=_log_writer_loop, daemon=True, name="cdp-log-writer")
        _writer_thread.start()


class CdpPlaywrightExecutor:
    """Executor that drives browser actions with Playwright over CDP."""

    def __init__(self, sandbox: Any, app_url: str):
        self.sandbox = sandbox
        self.app_url = app_url
        # Resolve the URL the CDP browser should actually navigate to.
        # When the sandbox runs the browser and dev server in the same
        # environment (e.g. a sandbox), the browser should use an
        # internal address to avoid TLS/proxy issues with the external URL.
        self._browser_url = (
            sandbox.get_browser_app_url()
            if hasattr(sandbox, "get_browser_app_url")
            else app_url
        )
        self.connector = BrowserConnector()
        self.page = None
        self.service_started = False

        self._console_messages = []
        self._page_errors = []
        self._request_failures = []
        self._http_errors = []
        self._debug_log_path = os.getenv("CDP_DEBUG_LOG_PATH", "logs/cdp_executor.log")
        self._startup_diagnostics: Dict[str, Any] = {}

    @property
    def browser_url(self) -> str:
        """The URL the CDP browser should navigate to (may differ from app_url)."""
        return self._browser_url

    def _debug_log(self, event: str, payload: Dict[str, Any]) -> None:
        """Enqueue a diagnostic record for async writing by the background log thread."""
        try:
            _ensure_writer_started()
            record = {
                "ts": datetime.utcnow().isoformat(timespec="milliseconds") + "Z",
                "event": event,
                "app_url": self.app_url,
                "payload": payload,
            }
            _log_queue.put_nowait((self._debug_log_path, record))
        except Exception:
            pass

    async def start_service(self, timeout: int = 60) -> None:
        if self.service_started:
            return

        await self.sandbox.start_browser()
        cdp_url = self.sandbox.get_cdp_url()
        self._debug_log("start_service", {"cdp_url": cdp_url})
        self.page = await self.connector.connect(cdp_url, timeout=timeout * 1000)
        self._attach_page_listeners()
        self.service_started = True  # CDP is connected — we are operational

        # Best-effort initial navigation with progressive backoff.
        # Failure is NON-FATAL: CDP is connected, the page may just be
        # compiling (Next.js first-request delay) or under heavy load.
        # The agent can inspect diagnostics, wait, and retry or restart
        # the dev server — this is a step-level issue, not a task-level one.
        base_nav_timeout = int(os.getenv("APP_NAVIGATION_TIMEOUT", "30")) * 1000
        nav_timeouts = [
            base_nav_timeout,
            max(base_nav_timeout // 2, 8000),
            max(base_nav_timeout // 4, 5000),
        ]

        last_error = None
        for attempt, nav_timeout in enumerate(nav_timeouts, 1):
            nav_result = await self.navigate(self._browser_url, timeout=nav_timeout)
            if nav_result.get("success"):
                return
            last_error = nav_result.get("error", "unknown navigation error")
            self._debug_log("start_service_navigate_failed", {
                "attempt": attempt, "timeout_ms": nav_timeout, "error": last_error,
            })

        # Page didn't load in any attempt — store diagnostics for the agent.
        try:
            self._startup_diagnostics = await self._collect_runtime_diagnostics()
        except Exception:
            self._startup_diagnostics = {}
        self._startup_diagnostics["navigation_error"] = last_error or "unknown"
        logger.warning(
            "CdpPlaywrightExecutor.start_service: initial navigation to %s "
            "failed after %d attempts (%s). CDP is connected — agents may "
            "attempt recovery (e.g. check dev server, wait for compilation, retry).",
            self.app_url, len(nav_timeouts), last_error,
        )

    @staticmethod
    def _truncate(value: Any, limit: int = 300) -> str:
        text = str(value)
        if len(text) <= limit:
            return text
        return text[: limit - 3] + "..."

    @staticmethod
    def _push(bucket: list, event: Dict[str, Any], max_events: int = 20) -> None:
        bucket.append(event)
        if len(bucket) > max_events:
            del bucket[:-max_events]

    def _attach_page_listeners(self) -> None:
        if not self.page:
            return

        def on_console(message):
            event = {"type": message.type, "text": self._truncate(message.text)}
            self._push(self._console_messages, event)
            if str(message.type).lower() == "error":
                self._debug_log("console_error", event)

        def on_page_error(error):
            event = {"message": self._truncate(error)}
            self._push(self._page_errors, event)
            self._debug_log("page_error", event)

        def on_request_failed(request):
            failure = request.failure
            if isinstance(failure, dict):
                error_text = failure.get("errorText", "unknown")
            elif isinstance(failure, str):
                error_text = failure
            else:
                error_text = str(failure) if failure else "unknown"
            event = {
                "url": self._truncate(request.url, 500),
                "method": request.method,
                "resource_type": request.resource_type,
                "error": self._truncate(error_text),
            }
            self._push(self._request_failures, event)
            self._debug_log("request_failed", event)

        def on_response(response):
            if response.status >= 400:
                event = {
                    "url": self._truncate(response.url, 500),
                    "status": response.status,
                    "resource_type": response.request.resource_type,
                }
                self._push(self._http_errors, event)
                self._debug_log("http_error_response", event)

        self.page.on("console", on_console)
        self.page.on("pageerror", on_page_error)
        self.page.on("requestfailed", on_request_failed)
        self.page.on("response", on_response)

    async def _wait_page_stable(
        self,
        network_idle_timeout_ms: int = 5000,
        post_wait_ms: int = 300,
    ) -> None:
        """Best-effort wait for page/network stabilization."""
        if not self.page:
            return
        try:
            await self.page.wait_for_load_state("networkidle", timeout=network_idle_timeout_ms)
        except Exception:
            # Some SPAs keep long-lived connections and never reach networkidle.
            pass
        if post_wait_ms > 0:
            await self.page.wait_for_timeout(post_wait_ms)

    async def _check_app_ready(self) -> Dict[str, Any]:
        """Verify the page is in an interactive state after navigation."""
        if not self.page:
            return {"success": False, "error": "no page"}
        try:
            diag = await self.page.evaluate("""() => {
                const body = document.body;
                const root = document.getElementById('root');
                const readyState = document.readyState;
                const bodyHtmlLen = body ? body.innerHTML.length : 0;
                const bodyText = (body ? body.innerText : '').trim();
                const childCount = body ? body.childElementCount : 0;
                const hasScripts = document.scripts.length > 0;
                return {
                    ready_state: readyState,
                    body_html_length: bodyHtmlLen,
                    body_text_preview: bodyText.slice(0, 200),
                    body_child_count: childCount,
                    has_scripts: hasScripts,
                };
            }""")
        except Exception as exc:
            return {"success": False, "error": f"app-ready eval failed: {exc}"}

        if diag.get("ready_state") == "loading":
            return {"success": False, "error": "page still loading"}

        body_html = int(diag.get("body_html_length", 0) or 0)
        body_children = int(diag.get("body_child_count", 0) or 0)
        if body_html < 50 and body_children == 0:
            return {"success": False, "error": f"body appears empty (html_len={body_html}, children={body_children})"}

        critical_failures = [f for f in self._request_failures
                             if f.get("url", "").endswith((".js", ".css"))]
        if critical_failures:
            return {"success": False, "error": f"critical resource failures: {critical_failures[0]}"}

        return {"success": True}

    async def _collect_runtime_diagnostics(self) -> Dict[str, Any]:
        if not self.page:
            return {}

        diagnostics = await self.page.evaluate(
            """() => {
                const root = document.getElementById('root');
                const scripts = Array.from(document.scripts).map((s) => s.src).filter(Boolean);
                const stylesheets = Array.from(document.querySelectorAll('link[rel="stylesheet"]'))
                  .map((l) => l.href)
                  .filter(Boolean);
                return {
                  ready_state: document.readyState,
                  root_exists: !!root,
                  root_child_count: root ? root.childElementCount : 0,
                  root_html_length: root ? root.innerHTML.length : 0,
                  body_html_length: document.body ? document.body.innerHTML.length : 0,
                  body_text_preview: (document.body?.innerText || '').trim().slice(0, 200),
                  script_sources: scripts.slice(0, 10),
                  stylesheet_hrefs: stylesheets.slice(0, 10),
                  has_vite_client: scripts.some((src) => src.includes('@vite/client')),
                };
            }"""
        )
        diagnostics["console_messages"] = list(self._console_messages)
        diagnostics["page_errors"] = list(self._page_errors)
        diagnostics["request_failures"] = list(self._request_failures)
        diagnostics["http_errors"] = list(self._http_errors)
        return diagnostics

    async def navigate(self, url: str, timeout: int = None) -> Dict[str, Any]:
        if timeout is None:
            timeout = int(os.getenv("APP_NAVIGATION_TIMEOUT", "30")) * 1000
        last_error = None
        base_backoff_s = 2.0
        self._debug_log("navigate_start", {"url": url, "timeout_ms": timeout})
        for attempt in (1, 2, 3):
            try:
                # Primary: domcontentloaded — avoids hanging on SPAs with
                # long-lived connections (HMR, polling, WebSocket).
                await self.page.goto(url, wait_until="domcontentloaded", timeout=timeout)
                await self.page.wait_for_timeout(500)

                # Best-effort app-ready check
                ready = await self._check_app_ready()
                if ready.get("success"):
                    out = {"success": True, "url": self.page.url}
                    self._debug_log("navigate_ok", {"attempt": attempt, "url": self.page.url, "method": "domcontentloaded+appready"})
                    return out

                last_error = ready.get("error", "app not ready")
                # App loaded but not ready — give it a short stabilisation window
                await self.page.wait_for_timeout(2000)
                ready = await self._check_app_ready()
                if ready.get("success"):
                    out = {"success": True, "url": self.page.url}
                    self._debug_log("navigate_ok", {"attempt": attempt, "url": self.page.url, "method": "domcontentloaded+appready+stabilised"})
                    return out

            except Exception as e:
                last_error = e
                self._debug_log("navigate_error", {"attempt": attempt, "error": self._truncate(e, 500)})
                if attempt <= 2:
                    backoff = base_backoff_s * (2 ** (attempt - 1))
                    await self.page.wait_for_timeout(int(backoff * 1000))
                    continue
                return {"success": False, "error": str(last_error)}

            # Fallback via networkidle for edge cases where domcontentloaded
            # returned before the app is genuinely usable.
            if attempt < 3:
                try:
                    self._debug_log("navigate_fallback_networkidle", {"attempt": attempt})
                    await self.page.goto(url, wait_until="networkidle", timeout=max(timeout // 2, 10000))
                    await self._wait_page_stable(network_idle_timeout_ms=3000, post_wait_ms=500)
                    out = {"success": True, "url": self.page.url}
                    self._debug_log("navigate_ok", {"attempt": attempt, "url": self.page.url, "method": "networkidle_fallback"})
                    return out
                except Exception:
                    backoff = base_backoff_s * (2 ** (attempt - 1))
                    await self.page.wait_for_timeout(int(backoff * 1000))
                    continue

        return {"success": False, "error": str(last_error)}

    async def click(self, selector: str, timeout: int = 5000) -> Dict[str, Any]:
        await self.page.click(selector, timeout=timeout)
        await self._wait_page_stable(network_idle_timeout_ms=3000, post_wait_ms=350)
        return {"success": True, "message": f"Clicked: {selector}"}

    async def click_at(self, x: int, y: int, timeout: int = 5000) -> Dict[str, Any]:
        await self.page.mouse.click(int(x), int(y))
        await self._wait_page_stable(network_idle_timeout_ms=3000, post_wait_ms=350)
        return {"success": True, "message": f"Clicked at: ({int(x)}, {int(y)})"}

    async def dblclick_at(self, x: int, y: int, timeout: int = 5000) -> Dict[str, Any]:
        await self.page.mouse.dblclick(int(x), int(y))
        await self._wait_page_stable(network_idle_timeout_ms=3000, post_wait_ms=350)
        return {"success": True, "message": f"Double-clicked at: ({int(x)}, {int(y)})"}

    async def hover(self, selector: str, timeout: int = 5000) -> Dict[str, Any]:
        await self.page.hover(selector, timeout=timeout)
        await self.page.wait_for_timeout(300)
        return {"success": True, "message": f"Hovered: {selector}"}

    async def type_text(self, selector: str, text: str, timeout: int = 5000) -> Dict[str, Any]:
        await self.page.fill(selector, text, timeout=timeout)
        await self._wait_page_stable(network_idle_timeout_ms=1500, post_wait_ms=150)
        return {"success": True, "message": f"Typed into: {selector}"}

    async def get_context(self) -> Dict[str, Any]:
        await self._wait_page_stable(network_idle_timeout_ms=2000, post_wait_ms=100)
        snapshot = await self.page.accessibility.snapshot()
        diagnostics = await self._collect_runtime_diagnostics()
        return {
            "success": True,
            "title": await self.page.title(),
            "url": self.page.url,
            "accessibility_tree": snapshot,
            "diagnostics": diagnostics,
        }

    async def evaluate_js(self, expression: str) -> Dict[str, Any]:
        result = await self.page.evaluate(expression)
        return {"success": True, "result": result}

    async def screenshot(self, full_page: bool = False) -> str:
        screenshot_bytes = await self.page.screenshot(type="png", full_page=full_page)
        return base64.b64encode(screenshot_bytes).decode()

    async def reset(self, app_url: str) -> None:
        if not self.connector.browser:
            raise RuntimeError("Browser not connected")

        if self.connector.page:
            try:
                await self.connector.page.close()
            except Exception:
                pass
        if self.connector.context:
            try:
                await self.connector.context.close()
            except Exception:
                pass

        self.connector.context = await self.connector.browser.new_context(
            viewport={"width": 1280, "height": 720},
            user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36",
            ignore_https_errors=True,
        )
        await apply_font_load_timeout(self.connector.context)
        self.connector.page = await self.connector.context.new_page()
        self.page = self.connector.page

        self._console_messages = []
        self._page_errors = []
        self._request_failures = []
        self._http_errors = []
        self._attach_page_listeners()
        await self.navigate(self._browser_url, timeout=int(os.getenv("APP_NAVIGATION_TIMEOUT", "30")) * 1000)

    async def keyboard_down(self, key: str) -> Dict[str, Any]:
        await self.page.keyboard.down(key)
        return {"success": True, "message": f"Key down: {key}"}

    async def keyboard_up(self, key: str) -> Dict[str, Any]:
        await self.page.keyboard.up(key)
        return {"success": True, "message": f"Key up: {key}"}

    async def keyboard_press(self, key: str, duration: int = 0) -> Dict[str, Any]:
        if duration > 0:
            await self.page.keyboard.down(key)
            await self.page.wait_for_timeout(duration)
            await self.page.keyboard.up(key)
        else:
            await self.page.keyboard.press(key)
        await self._wait_page_stable(network_idle_timeout_ms=500, post_wait_ms=50)
        return {"success": True, "message": f"Key press: {key}" + (f" (hold {duration}ms)" if duration > 0 else "")}

    async def shutdown(self) -> None:
        if not self.service_started:
            return
        await self.connector.close()
        self.page = None
        self.service_started = False

    async def check_cdp_health(self) -> bool:
        """Check whether the CDP connection is still alive.

        Probes the connection by creating and immediately closing a temporary
        browser context.  If the connection is stale (e.g. WebSocket timed out
        after long idle) this returns False.
        """
        if not self.connector or not self.connector.browser:
            return False
        try:
            ctx = await self.connector.browser.new_context(timeout=5000)
            await ctx.close()
            return True
        except Exception:
            return False

    async def reconnect_cdp(self, timeout: int = 60) -> None:
        """Reconnect CDP to the existing browser process.

        Closes the stale Playwright connection and creates a fresh one.
        Does NOT restart the browser process or the sandbox.
        Callers should re-navigate to the app URL after this succeeds.
        """
        try:
            await self.connector.close()
        except Exception:
            pass

        self.connector = BrowserConnector()
        cdp_url = self.sandbox.get_cdp_url()
        self.page = await self.connector.connect(cdp_url, timeout=timeout * 1000)
        self._attach_page_listeners()
        self.service_started = True
