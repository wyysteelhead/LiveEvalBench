"""Abstract interface for sandbox implementations.

This module defines the protocol that all sandbox implementations must follow,
enabling pluggable sandbox implementations.
"""

from typing import Dict, Protocol, runtime_checkable


@runtime_checkable
class SandboxInterface(Protocol):
    """Protocol defining the contract for sandbox implementations.

    All sandbox implementations must implement these methods
    to ensure compatibility with the evaluation workflow.
    """

    def start(self, timeout: int = 120) -> None:
        """Start the sandbox environment.

        Args:
            timeout: Maximum time in seconds to wait for sandbox startup

        Raises:
            TimeoutError: If sandbox fails to start within timeout
            RuntimeError: If sandbox creation fails
        """
        ...

    def write_files(self, files: Dict[str, str]) -> None:
        """Write application files to the sandbox.

        Args:
            files: Dictionary mapping file paths to content

        Raises:
            RuntimeError: If sandbox is not running or file write fails
        """
        ...

    async def start_app(self, install_timeout: int, startup_timeout: int) -> str:
        """Install dependencies and start the Next.js application.

        This method should:
        1. Run npm install
        2. Start npm run dev in background
        3. Poll the app URL until it returns 200 status

        Args:
            install_timeout: Maximum time in seconds for npm install
            startup_timeout: Maximum time in seconds for server startup

        Returns:
            URL of the running application

        Raises:
            RuntimeError: If npm install fails
            TimeoutError: If server doesn't become ready within timeout
        """
        ...

    async def start_browser(self) -> None:
        """Start Chromium browser with CDP enabled.

        The browser must be started with:
        - --remote-debugging-port=9222
        - --remote-debugging-address=0.0.0.0
        - --headless --no-sandbox --disable-gpu

        Raises:
            RuntimeError: If browser fails to start
        """
        ...

    def get_app_url(self) -> str:
        """Get the URL where the Next.js application is accessible.

        Returns:
            Full URL (e.g., "http://<sandbox-host>:3000" or "http://localhost:3000")
        """
        ...

    def get_cdp_url(self) -> str:
        """Get the WebSocket URL for Chrome DevTools Protocol connection.

        Returns:
            WebSocket URL (e.g., "ws://<sandbox-host>:9222" or "ws://localhost:9222")
        """
        ...

    def stop(self) -> None:
        """Stop and cleanup the sandbox environment.

        This should gracefully shutdown all processes and release resources.
        """
        ...

    def __enter__(self):
        """Context manager entry - start the sandbox."""
        ...

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - stop the sandbox."""
        ...
