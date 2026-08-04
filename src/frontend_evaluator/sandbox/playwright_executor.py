"""Playwright executor for running Playwright commands inside sandbox containers.

This executor manages a long-running Playwright service inside the container,
communicating via JSON files to maintain browser and page state across multiple
tool invocations.
"""

import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional


class PlaywrightExecutor:
    """Manages Playwright execution inside sandbox container.

    Starts a long-running Playwright service in the container and communicates
    with it via JSON files to execute browser automation commands.
    """

    def __init__(self, sandbox: Any, app_url: str):
        """Initialize the executor.

        Args:
            sandbox: sandbox instance (wrapper class)
            app_url: URL of the application to test
        """
        # Handle both sandbox wrapper and raw sandbox
        if hasattr(sandbox, 'sandbox'):
            # sandbox wrapper - use the inner sandbox
            self.sandbox_wrapper = sandbox
            self.sandbox = sandbox.sandbox
        else:
            # Raw sandbox object
            self.sandbox_wrapper = None
            self.sandbox = sandbox

        self.app_url = app_url
        self.service_started = False

        # File paths in container
        self.command_file = "/tmp/pw_command.json"
        self.result_file = "/tmp/pw_result.json"
        self.service_script = "/tmp/playwright_service.py"
        self.service_log = "/tmp/pw_service.log"

    async def start_service(self, timeout: int = 60) -> None:
        """Start the Playwright service in the container.

        Args:
            timeout: Timeout in seconds for Playwright init (default: 60)

        Raises:
            RuntimeError: If service fails to start
        """
        if self.service_started:
            return

        print("[Executor] Starting Playwright service in container...")

        try:
            # 1. Upload service script using commands.run instead of files.write_files
            # (files.write_files has issues with sandbox proxy)
            service_code = Path(__file__).parent / "playwright_service.py"
            with open(service_code, 'r') as f:
                script_content = f.read()

            # Write file using cat and heredoc
            write_cmd = f"cat > {self.service_script} << 'PLAYWRIGHT_SERVICE_EOF'\n{script_content}\nPLAYWRIGHT_SERVICE_EOF"
            await self.sandbox.commands.run(write_cmd)

            # Make executable
            await self.sandbox.commands.run(f"chmod +x {self.service_script}")
            print("[Executor] Service script uploaded")

            # 2. Start service in background
            await self.sandbox.commands.run(
                f"nohup python3 {self.service_script} > {self.service_log} 2>&1 &"
            )
            print("[Executor] Service process started")

            # 3. Wait a moment for service to start
            await asyncio.sleep(1)

            # 4. Send init command with configurable timeout
            await self._send_command({
                "type": "init",
                "url": self.app_url
            }, timeout=timeout)

            self.service_started = True
            print("[Executor] ✓ Playwright service ready")

        except Exception as e:
            # Try to read service log for debugging
            try:
                log_result = await self.sandbox.commands.run(f"cat {self.service_log}")
                log_content = log_result.logs.stdout[0].text if log_result.logs.stdout else "No logs"
                raise RuntimeError(f"Failed to start Playwright service: {e}\nService log:\n{log_content}")
            except:
                raise RuntimeError(f"Failed to start Playwright service: {e}")

    async def _send_command(self, command: Dict, timeout: int = 30) -> Dict:
        """Send a command to the service and wait for result.

        Args:
            command: Command dictionary with 'type' and parameters
            timeout: Timeout in seconds

        Returns:
            Result dictionary from service

        Raises:
            TimeoutError: If command times out
            RuntimeError: If command execution fails
        """
        # Write command file using echo (avoid files.write_files API issue)
        command_json = json.dumps(command)
        # Escape for shell - replace single quotes
        command_json_escaped = command_json.replace("'", "'\"'\"'")
        await self.sandbox.commands.run(f"echo '{command_json_escaped}' > {self.command_file}")

        # Poll for result
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                # Check if result file exists and read it
                result_check = await self.sandbox.commands.run(f"cat {self.result_file}")
                if result_check.logs.stdout:
                    result_content = result_check.logs.stdout[0].text
                    result = json.loads(result_content)

                    # Delete result file
                    await self.sandbox.commands.run(f"rm -f {self.result_file}")

                    # Check for errors
                    if not result.get("success"):
                        error_msg = result.get("error", "Unknown error")
                        traceback_msg = result.get("traceback", "")
                        raise RuntimeError(f"{error_msg}\n{traceback_msg}")

                    return result

            except (FileNotFoundError, json.JSONDecodeError, IndexError):
                # Result not ready yet or file doesn't exist
                pass
            except Exception as e:
                # If it's a command error (file not found), continue polling
                if "No such file" in str(e) or "cannot open" in str(e):
                    pass
                else:
                    raise

            await asyncio.sleep(0.2)

        # Timeout - try to read service log
        try:
            log_result = await self.sandbox.commands.run(f"tail -50 {self.service_log}")
            log_content = log_result.logs.stdout[0].text if log_result.logs.stdout else "No logs"
            raise TimeoutError(
                f"Command timeout after {timeout}s. Command: {command.get('type')}\n"
                f"Service log:\n{log_content}"
            )
        except:
            raise TimeoutError(f"Command timeout after {timeout}s. Command: {command.get('type')}")

    async def navigate(self, url: str, timeout: int = None) -> Dict:
        if timeout is None:
            timeout = int(os.getenv("APP_NAVIGATION_TIMEOUT", "30")) * 1000
        """Navigate to a URL.

        Args:
            url: URL to navigate to
            timeout: Navigation timeout in milliseconds

        Returns:
            Result with final URL
        """
        return await self._send_command({
            "type": "navigate",
            "url": url,
            "timeout": timeout
        })

    async def click(self, selector: str, timeout: int = 5000) -> Dict:
        """Click an element.

        Args:
            selector: Element selector
            timeout: Click timeout in milliseconds

        Returns:
            Result dictionary
        """
        return await self._send_command({
            "type": "click",
            "selector": selector,
            "timeout": timeout
        })

    async def click_at(self, x: int, y: int, timeout: int = 5000) -> Dict:
        """Click at viewport coordinates in CSS pixels.

        Args:
            x: Viewport X coordinate
            y: Viewport Y coordinate
            timeout: Timeout in milliseconds

        Returns:
            Result dictionary
        """
        return await self._send_command({
            "type": "click_at",
            "x": int(x),
            "y": int(y),
            "timeout": timeout,
        })

    async def dblclick_at(self, x: int, y: int, timeout: int = 5000) -> Dict:
        """Double-click at viewport coordinates in CSS pixels."""
        return await self._send_command({
            "type": "dblclick_at",
            "x": int(x),
            "y": int(y),
            "timeout": timeout,
        })

    async def hover(self, selector: str, timeout: int = 5000) -> Dict:
        """Hover an element.

        Args:
            selector: Element selector
            timeout: Hover timeout in milliseconds

        Returns:
            Result dictionary
        """
        return await self._send_command({
            "type": "hover",
            "selector": selector,
            "timeout": timeout
        })

    async def type_text(self, selector: str, text: str, timeout: int = 5000) -> Dict:
        """Type text into an input field.

        Args:
            selector: Input element selector
            text: Text to type
            timeout: Type timeout in milliseconds

        Returns:
            Result dictionary
        """
        return await self._send_command({
            "type": "type",
            "selector": selector,
            "text": text,
            "timeout": timeout
        })

    async def get_context(self) -> Dict:
        """Get page context including accessibility tree.

        Returns:
            Result with title, URL, and accessibility tree
        """
        return await self._send_command({
            "type": "get_context"
        })

    async def evaluate_js(self, expression: str) -> Dict:
        """Execute JavaScript in the page context.

        Args:
            expression: JavaScript expression to evaluate

        Returns:
            Result dictionary with the evaluation result
        """
        return await self._send_command({
            "type": "evaluate_js",
            "expression": expression
        })

    async def screenshot(self, full_page: bool = False) -> str:
        """Take a screenshot of the current page.

        Args:
            full_page: Whether to capture the full scrollable page

        Returns:
            Base64-encoded PNG screenshot
        """
        result = await self._send_command({
            "type": "screenshot",
            "full_page": full_page
        })
        return result["screenshot"]

    async def reset(self, app_url: str) -> None:
        """Reset browser state by creating a new context and navigating to app_url.

        Args:
            app_url: URL to navigate to after reset
        """
        await self._send_command({
            "type": "reset",
            "url": app_url
        })

    async def keyboard_down(self, key: str) -> Dict:
        return await self._send_command({
            "type": "keyboard_down",
            "key": key
        })

    async def keyboard_up(self, key: str) -> Dict:
        return await self._send_command({
            "type": "keyboard_up",
            "key": key
        })

    async def keyboard_press(self, key: str, duration: int = 0) -> Dict:
        return await self._send_command({
            "type": "keyboard_press",
            "key": key,
            "duration": duration
        })

    async def shutdown(self) -> None:
        """Shutdown the Playwright service.

        Raises:
            RuntimeError: If shutdown fails
        """
        if not self.service_started:
            return

        print("[Executor] Shutting down Playwright service...")

        try:
            await self._send_command({"type": "shutdown"}, timeout=10)
            self.service_started = False
            print("[Executor] ✓ Service shutdown complete")
        except Exception as e:
            print(f"[Executor] Warning: Shutdown error: {e}")
            # Force kill the service process
            try:
                await self.sandbox.commands.run("pkill -f playwright_service.py")
            except:
                pass
