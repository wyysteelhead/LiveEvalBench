"""LangGraph state definition for the frontend evaluator agent."""

import operator
from typing import TypedDict, List, Optional, Any, Annotated
from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages


class AgentState(TypedDict):
    """State for the frontend evaluator agent.

    This state is passed through the LangGraph workflow and tracks
    the agent's progress through the evaluation.
    """

    # Conversation messages (agent reasoning and tool results)
    # Use add_messages to append new messages instead of replacing
    messages: Annotated[List[BaseMessage], add_messages]

    # User's evaluation query (what to test)
    user_query: str

    # URL of the running application
    app_url: str

    # Final verdict (set when evaluation completes)
    verdict: Optional[dict]

    # Current iteration count
    iteration: int

    # Maximum iterations allowed
    max_iterations: int

    # PlaywrightExecutor instance (passed through for tool execution)
    executor: Any  # PlaywrightExecutor

    # Tool-call steps collected during evaluation
    steps: Annotated[List[dict], operator.add]
