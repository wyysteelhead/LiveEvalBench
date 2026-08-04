"""Tool registry system for agent tools.

Provides decorator-based registration and automatic schema generation.
"""

from typing import Callable, Dict, List, Any, Optional, Iterable, get_type_hints, get_origin, get_args, Union
from inspect import signature, Parameter
import json


# Global registry of tools
TOOL_REGISTRY: Dict[str, Dict[str, Any]] = {}


def register_tool(name: str, description: str):
    """Decorator to register a tool function.

    Args:
        name: Tool name (used by LLM to invoke)
        description: Human-readable description of what the tool does

    Example:
        @register_tool("navigate", "Navigate to a URL")
        async def navigate(url: str) -> str:
            ...
    """

    def decorator(func: Callable) -> Callable:
        # Get function signature
        sig = signature(func)
        type_hints = get_type_hints(func)

        # Build parameter schema
        parameters = {
            "type": "object",
            "properties": {},
            "required": [],
        }

        for param_name, param in sig.parameters.items():
            # Skip self/cls/executor parameters (executor is injected at runtime)
            if param_name in ("self", "cls", "executor"):
                continue

            param_type = type_hints.get(param_name, str)
            param_schema = _type_to_schema(param_type)

            parameters["properties"][param_name] = param_schema

            # Mark as required if no default value
            if param.default == Parameter.empty:
                parameters["required"].append(param_name)

        # Register tool
        TOOL_REGISTRY[name] = {
            "name": name,
            "description": description,
            "input_schema": parameters,
            "function": func,
        }

        return func

    return decorator


def _type_to_schema(python_type: type) -> Dict[str, Any]:
    """Convert Python type to JSON schema type.

    Args:
        python_type: Python type annotation

    Returns:
        JSON schema dictionary
    """
    # Handle basic types
    type_mapping = {
        str: {"type": "string"},
        int: {"type": "integer"},
        float: {"type": "number"},
        bool: {"type": "boolean"},
    }

    if python_type in type_mapping:
        return type_mapping[python_type]

    origin = get_origin(python_type)
    args = get_args(python_type)

    # Handle Optional/Union[T, None]
    if origin is Union and args:
        non_none = [a for a in args if a is not type(None)]  # noqa: E721
        if len(non_none) == 1:
            return _type_to_schema(non_none[0])
        return {"type": "string"}

    # Handle list[T]
    if origin is list:
        item_schema = _type_to_schema(args[0]) if args else {"type": "string"}
        return {"type": "array", "items": item_schema}

    # Handle dict[K, V]
    if origin is dict:
        if len(args) == 2:
            return {"type": "object", "additionalProperties": _type_to_schema(args[1])}
        return {"type": "object"}

    # Default to string
    return {"type": "string"}


def get_tool_definitions(allowed_names: Optional[Iterable[str]] = None) -> List[Dict[str, Any]]:
    """Get all registered tools in Anthropic API format.

    Args:
        allowed_names: Optional allowlist of tool names. If provided, only
            tools in this set are returned.

    Returns:
        List of tool definitions for Claude API
    """
    allowed = set(allowed_names) if allowed_names is not None else None
    return [
        {
            "name": tool["name"],
            "description": tool["description"],
            "input_schema": tool["input_schema"],
        }
        for tool in TOOL_REGISTRY.values()
        if allowed is None or tool["name"] in allowed
    ]


def get_tool_function(name: str) -> Callable:
    """Get the function for a registered tool.

    Args:
        name: Tool name

    Returns:
        Tool function

    Raises:
        KeyError: If tool not found
    """
    if name not in TOOL_REGISTRY:
        raise KeyError(f"Tool '{name}' not found in registry")

    return TOOL_REGISTRY[name]["function"]


def clear_registry() -> None:
    """Clear all registered tools (useful for testing)."""
    TOOL_REGISTRY.clear()
