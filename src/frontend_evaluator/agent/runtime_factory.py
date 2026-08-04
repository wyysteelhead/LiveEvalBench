from __future__ import annotations

import os
from typing import List, Optional

from ..events.emitter import EventEmitter
from ..utils.config import Config
from .claude_sdk_runtime import ClaudeSDKRuntime
from .evaluator import FrontendEvaluator
from .runtime_interface import EvaluatorRuntime


def create_evaluator_runtime(
    config: Config,
    *,
    max_iterations: Optional[int] = None,
    allowed_tools: Optional[List[str]] = None,
    log_level: Optional[str] = None,
    event_emitter: Optional[EventEmitter] = None,
) -> EvaluatorRuntime:
    """Create the configured evaluator runtime backend."""
    runtime = config.agent_runtime

    if runtime == "claude_sdk" and config.model_provider != "anthropic":
        raise ValueError(
            "AGENT_RUNTIME=claude_sdk requires MODEL_PROVIDER=anthropic"
        )

    common_kwargs = {
        "api_key": config.get_llm_api_key(),
        "model": config.model_name,
        "provider": config.model_provider,
        "base_url": config.custom_base_url if config.model_provider == "custom" else None,
        "max_iterations": max_iterations or config.max_agent_steps,
        "max_verdict_validation_retries": config.max_verdict_validation_retries,
        "llm_api_retry_count": max(0, int(os.getenv("LLM_API_RETRY_COUNT", "5"))),
        "llm_api_retry_base_delay_seconds": max(0.0, float(os.getenv("LLM_API_RETRY_BASE_DELAY_SECONDS", "1.0"))),
        "llm_api_retry_jitter_seconds": max(0.0, float(os.getenv("LLM_API_RETRY_JITTER_SECONDS", "0.35"))),
        "allowed_tools": allowed_tools,
        "log_level": log_level or config.log_level,
        "event_emitter": event_emitter,
        "disable_thinking": config.disable_thinking,
        "temperature": config.temperature,
    }

    if runtime == "react":
        return FrontendEvaluator(**common_kwargs)
    if runtime == "claude_sdk":
        return ClaudeSDKRuntime(
            **common_kwargs,
            cli_path=config.claude_sdk_cli_path,
            system_tools_enabled=config.claude_sdk_system_tools_enabled,
        )

    raise ValueError(f"Unknown AGENT_RUNTIME: {runtime}")