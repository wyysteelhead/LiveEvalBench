"""Playwright service that runs inside the container.

This service maintains a long-running browser and page instance,
receives commands via JSON files, executes them, and returns results.
"""

import asyncio
import base64
import json
import os
import sys
import traceback
from pathlib import Path


COMMAND_FILE = "/tmp/pw_command.json"
RESULT_FILE = "/tmp/pw_result.json"
POLL_INTERVAL = 0.1  # seconds
MAX_DEBUG_EVENTS = 20


def _truncate(value, limit=300):
    text = str(value)
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _push_runtime_event(bucket, event):
    bucket.append(event)
    if len(bucket) > MAX_DEBUG_EVENTS:
        del bucket[:-MAX_DEBUG_EVENTS]


def _new_runtime_state():
    return {
        "console_messages": [],
        "page_errors": [],
        "request_failures": [],
        "http_errors": [],
    }


def _attach_page_listeners(page, runtime_state):
    def on_console(message):
        try:
            _push_runtime_event(
                runtime_state["console_messages"],
                {
                    "type": message.type,
                    "text": _truncate(message.text),
                },
            )
        except Exception as exc:
            _push_runtime_event(
                runtime_state["console_messages"],
                {
                    "type": "listener_error",
                    "text": _truncate(exc),
                },
            )

    def on_page_error(error):
        try:
            _push_runtime_event(
                runtime_state["page_errors"],
                {"message": _truncate(error)},
            )
        except Exception as exc:
            _push_runtime_event(
                runtime_state["page_errors"],
                {"message": _truncate(exc)},
            )

    def on_request_failed(request):
        try:
            failure = request.failure
            if isinstance(failure, dict):
                error_text = failure.get("errorText", "unknown")
            elif isinstance(failure, str):
                error_text = failure
            else:
                error_text = str(failure) if failure else "unknown"

            _push_runtime_event(
                runtime_state["request_failures"],
                {
                    "url": _truncate(request.url, 500),
                    "method": request.method,
                    "resource_type": request.resource_type,
                    "error": _truncate(error_text),
                },
            )
        except Exception as exc:
            _push_runtime_event(
                runtime_state["request_failures"],
                {
                    "url": "listener_error",
                    "method": "unknown",
                    "resource_type": "unknown",
                    "error": _truncate(exc),
                },
            )

    def on_response(response):
        try:
            if response.status >= 400:
                _push_runtime_event(
                    runtime_state["http_errors"],
                    {
                        "url": _truncate(response.url, 500),
                        "status": response.status,
                        "resource_type": response.request.resource_type,
                    },
                )
        except Exception as exc:
            _push_runtime_event(
                runtime_state["http_errors"],
                {
                    "url": "listener_error",
                    "status": -1,
                    "resource_type": _truncate(exc),
                },
            )

    page.on("console", on_console)
    page.on("pageerror", on_page_error)
    page.on("requestfailed", on_request_failed)
    page.on("response", on_response)


async def _collect_runtime_diagnostics(page, runtime_state):
    diagnostics = await page.evaluate(
        """() => {
            const root = document.getElementById('root');
            const scripts = Array.from(document.scripts).map((script) => script.src).filter(Boolean);
            const stylesheets = Array.from(document.querySelectorAll('link[rel="stylesheet"]'))
              .map((link) => link.href)
              .filter(Boolean);
            const resources = performance.getEntriesByType('resource')
              .slice(0, 20)
              .map((entry) => ({
                name: entry.name,
                initiatorType: entry.initiatorType,
              }));

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
              resources,
            };
        }"""
    )

    diagnostics["console_messages"] = list(runtime_state["console_messages"])
    diagnostics["page_errors"] = list(runtime_state["page_errors"])
    diagnostics["request_failures"] = list(runtime_state["request_failures"])
    diagnostics["http_errors"] = list(runtime_state["http_errors"])
    return diagnostics


async def main():
    """Main service loop."""
    from playwright.async_api import async_playwright

    playwright = None
    browser = None
    page = None
    context = None
    runtime_state = _new_runtime_state()

    print("[Playwright Service] Starting...", flush=True)

    try:
        # Initialize Playwright
        playwright = await async_playwright().start()
        print("[Playwright Service] Playwright started", flush=True)

        # Wait for init command
        while True:
            if os.path.exists(COMMAND_FILE):
                with open(COMMAND_FILE, 'r') as f:
                    command = json.load(f)

                if command.get("type") == "init":
                    print(f"[Playwright Service] Initializing browser with URL: {command.get('url')}", flush=True)

                    try:
                        # Launch browser with explicit path to Chromium
                        print("[Playwright Service] Launching Chromium...", flush=True)
                        browser = await playwright.chromium.launch(
                            headless=True,
                            executable_path="/opt/chromium/chrome",
                            args=[
                                '--no-sandbox',
                                '--disable-dev-shm-usage',
                                '--font-render-hinting=none',
                                '--disable-font-subpixel-positioning',
                            ]
                        )
                        print("[Playwright Service] Chromium launched successfully", flush=True)

                        # Create page
                        print("[Playwright Service] Creating new page...", flush=True)
                        context = await browser.new_context(
                            viewport={"width": 1280, "height": 720}
                        )
                        from . import apply_font_load_timeout
                        await apply_font_load_timeout(context)
                        page = await context.new_page()
                        runtime_state = _new_runtime_state()
                        _attach_page_listeners(page, runtime_state)
                        print("[Playwright Service] Page created", flush=True)

                        # Navigate to initial URL if provided
                        initial_url = command.get("url")
                        if initial_url:
                            print(f"[Playwright Service] Navigating to {initial_url}...", flush=True)
                            await page.goto(initial_url, wait_until="domcontentloaded", timeout=int(os.getenv("APP_NAVIGATION_TIMEOUT", "30")) * 1000)
                            await page.wait_for_timeout(1000)
                            print("[Playwright Service] Navigation complete", flush=True)

                        result = {
                            "success": True,
                            "message": "Browser initialized",
                            "url": page.url if initial_url else None
                        }

                    except Exception as e:
                        print(f"[Playwright Service] ERROR during init: {e}", flush=True)
                        traceback.print_exc()
                        result = {
                            "success": False,
                            "error": str(e),
                            "traceback": traceback.format_exc()
                        }

                    with open(RESULT_FILE, 'w') as f:
                        json.dump(result, f)

                    os.remove(COMMAND_FILE)
                    print("[Playwright Service] Browser initialized", flush=True)
                    break

            await asyncio.sleep(POLL_INTERVAL)

        # Main command loop
        print("[Playwright Service] Ready to receive commands", flush=True)

        while True:
            if os.path.exists(COMMAND_FILE):
                try:
                    # Read command
                    with open(COMMAND_FILE, 'r') as f:
                        command = json.load(f)

                    command_type = command.get("type")
                    print(f"[Playwright Service] Executing command: {command_type}", flush=True)

                    result = None

                    # Execute command
                    if command_type == "navigate":
                        url = command.get("url")
                        timeout = command.get("timeout", int(os.getenv("APP_NAVIGATION_TIMEOUT", "30")) * 1000)
                        last_error = None
                        base_backoff_s = 2.0
                        for attempt in (1, 2, 3):
                            try:
                                await page.goto(url, wait_until="networkidle", timeout=timeout)
                                await page.wait_for_timeout(1000)
                                result = {
                                    "success": True,
                                    "url": page.url
                                }
                                break
                            except Exception as e:
                                last_error = e
                                if attempt <= 2:
                                    try:
                                        await page.goto(url, wait_until="domcontentloaded", timeout=timeout)
                                        await page.wait_for_timeout(1000)
                                        result = {
                                            "success": True,
                                            "url": page.url
                                        }
                                        break
                                    except Exception as fallback_error:
                                        last_error = fallback_error
                                        backoff = base_backoff_s * (2 ** (attempt - 1))
                                        await page.wait_for_timeout(int(backoff * 1000))
                                        continue
                                result = {
                                    "success": False,
                                    "error": str(last_error)
                                }

                    elif command_type == "click":
                        selector = command.get("selector")
                        timeout = command.get("timeout", 5000)
                        await page.click(selector, timeout=timeout)
                        # Wait for React state updates to reflect in DOM
                        await page.wait_for_timeout(500)
                        result = {
                            "success": True,
                            "message": f"Clicked: {selector}"
                        }

                    elif command_type == "click_at":
                        x = int(command.get("x"))
                        y = int(command.get("y"))
                        await page.mouse.click(x, y)
                        # Wait for React state updates to reflect in DOM
                        await page.wait_for_timeout(500)
                        result = {
                            "success": True,
                            "message": f"Clicked at: ({x}, {y})"
                        }

                    elif command_type == "hover":
                        selector = command.get("selector")
                        timeout = command.get("timeout", 5000)
                        await page.hover(selector, timeout=timeout)
                        # Allow CSS :hover transitions/effects to apply.
                        await page.wait_for_timeout(300)
                        result = {
                            "success": True,
                            "message": f"Hovered: {selector}"
                        }

                    elif command_type == "type":
                        selector = command.get("selector")
                        text = command.get("text")
                        timeout = command.get("timeout", 5000)
                        await page.fill(selector, text, timeout=timeout)
                        result = {
                            "success": True,
                            "message": f"Typed into: {selector}"
                        }

                    elif command_type == "get_context":
                        title = await page.title()
                        url = page.url
                        snapshot = await page.accessibility.snapshot()
                        diagnostics = await _collect_runtime_diagnostics(page, runtime_state)

                        result = {
                            "success": True,
                            "title": title,
                            "url": url,
                            "accessibility_tree": snapshot,
                            "diagnostics": diagnostics,
                        }

                    elif command_type == "evaluate_js":
                        # Execute JavaScript in the page context
                        expression = command.get("expression")
                        js_result = await page.evaluate(expression)
                        result = {
                            "success": True,
                            "result": js_result
                        }

                    elif command_type == "keyboard_down":
                        key = command.get("key")
                        await page.keyboard.down(key)
                        result = {
                            "success": True,
                            "message": f"Key down: {key}"
                        }

                    elif command_type == "keyboard_up":
                        key = command.get("key")
                        await page.keyboard.up(key)
                        result = {
                            "success": True,
                            "message": f"Key up: {key}"
                        }

                    elif command_type == "keyboard_press":
                        key = command.get("key")
                        duration = command.get("duration", 0)
                        if duration > 0:
                            await page.keyboard.down(key)
                            await page.wait_for_timeout(duration)
                            await page.keyboard.up(key)
                        else:
                            await page.keyboard.press(key)
                        result = {
                            "success": True,
                            "message": f"Key press: {key}" + (f" (hold {duration}ms)" if duration > 0 else "")
                        }

                    elif command_type == "reset":
                        url = command.get("url")
                        # Close current page and create a fresh context + page
                        if page:
                            try:
                                await page.close()
                            except:
                                pass
                        if context:
                            try:
                                await context.close()
                            except:
                                pass
                        context = await browser.new_context(
                            viewport={"width": 1280, "height": 720}
                        )
                        from . import apply_font_load_timeout
                        await apply_font_load_timeout(context)
                        page = await context.new_page()
                        runtime_state = _new_runtime_state()
                        _attach_page_listeners(page, runtime_state)
                        if url:
                            await page.goto(url, wait_until="domcontentloaded", timeout=int(os.getenv("APP_NAVIGATION_TIMEOUT", "30")) * 1000)
                            await page.wait_for_timeout(1000)
                        result = {
                            "success": True,
                            "message": "Browser context reset",
                            "url": page.url
                        }

                    elif command_type == "screenshot":
                        full_page = command.get("full_page", False)
                        screenshot_bytes = await page.screenshot(
                            type="png", full_page=full_page
                        )
                        screenshot_b64 = base64.b64encode(screenshot_bytes).decode()
                        result = {
                            "success": True,
                            "screenshot": screenshot_b64
                        }

                    elif command_type == "shutdown":
                        result = {
                            "success": True,
                            "message": "Shutting down"
                        }

                        # Write result before shutting down
                        with open(RESULT_FILE, 'w') as f:
                            json.dump(result, f)
                        os.remove(COMMAND_FILE)

                        print("[Playwright Service] Shutdown requested", flush=True)
                        break

                    else:
                        result = {
                            "success": False,
                            "error": f"Unknown command type: {command_type}"
                        }

                    # Write result
                    if result:
                        with open(RESULT_FILE, 'w') as f:
                            json.dump(result, f)

                    # Remove command file
                    os.remove(COMMAND_FILE)

                    print(f"[Playwright Service] Command completed: {command_type}", flush=True)

                except Exception as e:
                    # Write error result
                    error_result = {
                        "success": False,
                        "error": str(e),
                        "traceback": traceback.format_exc()
                    }

                    with open(RESULT_FILE, 'w') as f:
                        json.dump(error_result, f)

                    # Remove command file
                    if os.path.exists(COMMAND_FILE):
                        os.remove(COMMAND_FILE)

                    print(f"[Playwright Service] Error: {e}", flush=True)

            await asyncio.sleep(POLL_INTERVAL)

    finally:
        # Cleanup
        print("[Playwright Service] Cleaning up...", flush=True)

        if page:
            try:
                await page.close()
            except:
                pass

        if context:
            try:
                await context.close()
            except:
                pass

        if browser:
            try:
                await browser.close()
            except:
                pass

        if playwright:
            try:
                await playwright.stop()
            except:
                pass

        print("[Playwright Service] Stopped", flush=True)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("[Playwright Service] Interrupted", flush=True)
    except Exception as e:
        print(f"[Playwright Service] Fatal error: {e}", flush=True)
        traceback.print_exc()
        sys.exit(1)
