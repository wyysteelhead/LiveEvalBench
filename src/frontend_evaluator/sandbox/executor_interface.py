"""Protocol interface for browser executor backends."""

from typing import Any, Dict, Protocol, runtime_checkable


@runtime_checkable
class ExecutorInterface(Protocol):
    """Protocol that all executor backends must satisfy."""

    async def start_service(self, timeout: int = 60) -> None:
        """Start the browser service."""
        ...

    async def navigate(self, url: str, timeout: int = 30000) -> Dict[str, Any]:
        """Navigate to a URL."""
        ...

    async def click(self, selector: str, timeout: int = 5000) -> Dict[str, Any]:
        """Click an element."""
        ...

    async def click_at(self, x: int, y: int, timeout: int = 5000) -> Dict[str, Any]:
        """Click at viewport coordinates in CSS pixels."""
        ...

    async def dblclick_at(self, x: int, y: int, timeout: int = 5000) -> Dict[str, Any]:
        """Double-click at viewport coordinates in CSS pixels."""
        ...

    async def hover(self, selector: str, timeout: int = 5000) -> Dict[str, Any]:
        """Hover an element."""
        ...

    async def type_text(self, selector: str, text: str, timeout: int = 5000) -> Dict[str, Any]:
        """Type text into an input field."""
        ...

    async def get_context(self) -> Dict[str, Any]:
        """Get page context including accessibility tree."""
        ...

    async def evaluate_js(self, expression: str) -> Dict[str, Any]:
        """Execute JavaScript in the page context."""
        ...

    async def screenshot(self, full_page: bool = False) -> str:
        """Take a screenshot and return base64-encoded PNG."""
        ...

    async def reset(self, app_url: str) -> None:
        """Reset browser state and navigate to app_url."""
        ...

    async def keyboard_down(self, key: str) -> Dict[str, Any]:
        """Press and hold a keyboard key (dispatches keydown only)."""
        ...

    async def keyboard_up(self, key: str) -> Dict[str, Any]:
        """Release a held keyboard key (dispatches keyup)."""
        ...

    async def keyboard_press(self, key: str, duration: int = 0) -> Dict[str, Any]:
        """Press and release a keyboard key, optionally holding for duration ms."""
        ...

    async def shutdown(self) -> None:
        """Shutdown the browser service."""
        ...
