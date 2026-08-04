from __future__ import annotations

from typing import Any, Dict, Optional, Protocol, TYPE_CHECKING

if TYPE_CHECKING:
    from .config import AgentConfig


class EvaluatorRuntime(Protocol):
    """Common protocol for evaluator runtime backends."""

    async def evaluate(
        self,
        user_query: str,
        app_url: str,
        executor: Any,
        standard_ids: Optional[list[str]] = None,
        task_id: Optional[str] = None,
        task_title: Optional[str] = None,
        agent_config: Optional["AgentConfig"] = None,
    ) -> Dict[str, Any]:
        ...

    async def evaluate_agentic(
        self,
        agent_config: "AgentConfig",
        app_url: str,
        executor: Any,
        task_context: Optional[str] = None,
    ) -> Dict[str, Any]:
        ...
