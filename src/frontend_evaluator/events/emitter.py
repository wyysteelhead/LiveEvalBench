"""Event emitter for broadcasting evaluation events."""

from typing import Callable, List, Optional, Dict, Any
import asyncio
from ..utils.logger import logger


class EventEmitter:
    """Emits events during evaluation for monitoring and visualization."""

    def __init__(self, task_id: str, enabled: bool = True):
        """Initialize event emitter.

        Args:
            task_id: ID of the task being evaluated
            enabled: Whether event emission is enabled
        """
        self.task_id = task_id
        self.enabled = enabled
        self.callbacks: List[Callable] = []
        self._iteration = 0

    def on(self, callback: Callable):
        """Register a callback for events.

        Args:
            callback: Async function to call with each event
        """
        self.callbacks.append(callback)

    def set_iteration(self, iteration: int):
        """Update current iteration number."""
        self._iteration = iteration

    async def emit(
        self,
        event_type: str,
        data: Dict[str, Any],
        screenshot: Optional[str] = None,
    ):
        """Emit an event to all registered callbacks.

        Args:
            event_type: Type of event
            data: Event data
            screenshot: Optional base64-encoded screenshot
        """
        if not self.enabled:
            return

        from .models import EvaluationEvent
        import time

        event = EvaluationEvent(
            task_id=self.task_id,
            timestamp=time.time(),
            iteration=self._iteration,
            event_type=event_type,
            data=data,
            screenshot=screenshot,
        )

        # Call all callbacks
        for callback in self.callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(event)
                else:
                    callback(event)
            except Exception as e:
                logger.warning(f"Event callback error: {e}")
