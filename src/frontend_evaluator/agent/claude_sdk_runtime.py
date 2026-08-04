from __future__ import annotations

import asyncio
import time
from inspect import Parameter, signature
from typing import Any, Callable, Dict, List, Optional, get_type_hints

from langchain_core.messages import HumanMessage, SystemMessage

from ..tools import browser_actions
from ..llm.retry import invoke_with_llm_api_retry
from ..tools.registry import TOOL_REGISTRY, get_tool_function
from ..tools.verdict import VerdictSubmitted
from ..utils.logger import logger
from ..llm.message_utils import get_provider_from_llm, normalize_multimodal_content_for_provider
from .config import AgentConfig
from .evaluator import FrontendEvaluator, InfraError, _is_infra_error_message, _INFRA_CONSECUTIVE_THRESHOLD
from .prompts import format_agentic_prompt, format_system_prompt


class ClaudeSDKRuntime(FrontendEvaluator):
    """Evaluator runtime backed by claude-agent-sdk."""

    _SDK_SERVER_NAME = "frontend_eval"

    def __init__(
        self,
        api_key: str,
        model: Optional[str] = None,
        provider: str = "anthropic",
        base_url: Optional[str] = None,
        max_iterations: int = 10,
        max_verdict_validation_retries: int = 6,
        allowed_tools: Optional[List[str]] = None,
        log_level: Optional[str] = None,
        event_emitter: Any = None,
        cli_path: Optional[str] = None,
        system_tools_enabled: bool = False,
        disable_thinking: bool = False,
    ):
        super().__init__(
            api_key=api_key,
            model=model,
            provider=provider,
            base_url=base_url,
            max_iterations=max_iterations,
            max_verdict_validation_retries=max_verdict_validation_retries,
            allowed_tools=allowed_tools,
            log_level=log_level,
            event_emitter=event_emitter,
            disable_thinking=disable_thinking,
        )
        self.cli_path = cli_path
        self.system_tools_enabled = system_tools_enabled
        self._sdk_verdict_validation_failures = 0
        self._sdk_verdict_payload: Optional[Dict[str, Any]] = None

    @staticmethod
    def _load_sdk_symbols() -> Dict[str, Any]:
        try:
            from claude_agent_sdk import (
                ClaudeAgentOptions,
                ClaudeSDKClient,
                create_sdk_mcp_server,
                tool,
            )
        except ImportError as exc:
            raise RuntimeError(
                "claude-agent-sdk is required for AGENT_RUNTIME=claude_sdk. "
                "Install the 'claude-agent-sdk' package first."
            ) from exc

        return {
            "ClaudeAgentOptions": ClaudeAgentOptions,
            "ClaudeSDKClient": ClaudeSDKClient,
            "create_sdk_mcp_server": create_sdk_mcp_server,
            "tool": tool,
        }

    @staticmethod
    def _tool_arg_schema(tool_func: Callable[..., Any]) -> Dict[str, Any]:
        hints = get_type_hints(tool_func)
        params: Dict[str, Any] = {}
        for param_name, param in signature(tool_func).parameters.items():
            if param_name in {"self", "cls", "executor"}:
                continue
            annotation = hints.get(param_name, str)
            if param.default is not Parameter.empty and annotation is None:
                annotation = str
            params[param_name] = annotation
        return params

    @classmethod
    def _mcp_tool_name(cls, tool_name: str) -> str:
        return f"mcp__{cls._SDK_SERVER_NAME}__{tool_name}"

    @staticmethod
    def _sdk_text_result(text: str) -> Dict[str, Any]:
        return {"content": [{"type": "text", "text": text}]}

    @staticmethod
    def _sdk_content_result(content: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {"content": content}

    def _build_sdk_tool_wrappers(self, executor: Any) -> List[Callable[..., Any]]:
        sdk = self._load_sdk_symbols()
        sdk_tool = sdk["tool"]
        wrappers: List[Callable[..., Any]] = []

        allowed_names = self.allowed_tools if self.allowed_tools is not None else set(TOOL_REGISTRY.keys())

        for tool_name, tool_meta in TOOL_REGISTRY.items():
            if tool_name not in allowed_names:
                continue
            tool_func = get_tool_function(tool_name)
            arg_schema = self._tool_arg_schema(tool_func)
            description = tool_meta["description"]

            async def _invoke(args, _tool_name=tool_name, _tool_func=tool_func):
                return await self._execute_sdk_tool(
                    tool_name=_tool_name,
                    tool_func=_tool_func,
                    tool_args=args,
                    executor=executor,
                )

            wrappers.append(sdk_tool(tool_name, description, arg_schema)(_invoke))

        return wrappers

    def _build_sdk_options(
        self,
        *,
        system_prompt: str,
        tool_names: List[str],
        max_turns: int,
    ) -> Any:
        sdk = self._load_sdk_symbols()
        ClaudeAgentOptions = sdk["ClaudeAgentOptions"]
        create_sdk_mcp_server = sdk["create_sdk_mcp_server"]

        server = create_sdk_mcp_server(
            name=self._SDK_SERVER_NAME,
            version="1.0.0",
            tools=self._build_sdk_tool_wrappers(executor=self._sdk_executor),
        )

        option_kwargs: Dict[str, Any] = {
            "system_prompt": system_prompt,
            "max_turns": max_turns,
            "mcp_servers": {self._SDK_SERVER_NAME: server},
            "allowed_tools": [self._mcp_tool_name(name) for name in tool_names],
        }
        if self.model:
            option_kwargs["model"] = self.model
        if self.cli_path:
            option_kwargs["cli_path"] = self.cli_path
        if not self.system_tools_enabled:
            option_kwargs["tools"] = []
        return ClaudeAgentOptions(**option_kwargs)

    async def _execute_sdk_tool(
        self,
        *,
        tool_name: str,
        tool_func: Callable[..., Any],
        tool_args: Dict[str, Any],
        executor: Any,
    ) -> Dict[str, Any]:
        tool_name = tool_name.strip()
        step = {
            "tool_name": tool_name,
            "args": {k: str(v) for k, v in tool_args.items()},
            "timestamp": time.time(),
        }
        if self._current_task_id:
            step["task_id"] = self._current_task_id
        if self._current_task_title:
            step["task_title"] = self._current_task_title

        if self.event_emitter:
            self.event_emitter.set_iteration(len(self._steps))
            await self.event_emitter.emit(
                "tool_executing",
                {
                    "tool_name": tool_name,
                    "tool_args": tool_args,
                },
            )

        if not self._is_tool_allowed(tool_name):
            message = f"Permission denied: tool '{tool_name}' is not allowed for this agent"
            step["result"] = message
            step["success"] = False
            self._steps.append(step)
            if self.event_emitter:
                await self.event_emitter.emit("tool_failed", {"tool_name": tool_name, "error": message})
            return self._sdk_text_result(message)

        wrong_verdict_tool_error = self._should_block_wrong_verdict_tool(tool_name)
        if wrong_verdict_tool_error:
            step["result"] = wrong_verdict_tool_error
            step["success"] = False
            self._steps.append(step)
            if self.event_emitter:
                await self.event_emitter.emit("tool_failed", {"tool_name": tool_name, "error": wrong_verdict_tool_error})
            return self._sdk_text_result(wrong_verdict_tool_error)

        blind_retry_error = self._should_block_blind_retry(tool_name, tool_args)
        if blind_retry_error:
            step["result"] = blind_retry_error
            step["success"] = False
            step["semantic_action_signature"] = self._semantic_action_signature(tool_name, tool_args)
            self._steps.append(step)
            if self.event_emitter:
                await self.event_emitter.emit("tool_failed", {"tool_name": tool_name, "error": blind_retry_error})
            return self._sdk_text_result(blind_retry_error)

        try:
            if "executor" in signature(tool_func).parameters:
                result = await tool_func(executor, **tool_args)
            else:
                result = await tool_func(**tool_args)

            result_text = str(result.get("display_text") or result.get("result") or result) if isinstance(result, dict) else str(result)
            sdk_response = self._sdk_text_result(result_text)
            if isinstance(result, dict) and "tool_message_content" in result:
                multimodal_content = result.get("tool_message_content")
                if isinstance(multimodal_content, list) and multimodal_content:
                    sdk_response = self._sdk_content_result(
                        normalize_multimodal_content_for_provider(
                            multimodal_content,
                            get_provider_from_llm(self.llm),
                        )
                    )

            step["result"] = result_text
            step["success"] = True
            step["semantic_action_signature"] = self._semantic_action_signature(tool_name, tool_args)
            self._consecutive_infra_failures = 0

            selector_arg_name = browser_actions.SELECTOR_ARG_BY_TOOL.get(tool_name)
            if selector_arg_name:
                step["element_id"] = self._parse_selector_id(tool_args.get(selector_arg_name, ""))

            screenshot = None
            if tool_name in browser_actions.PAGE_MUTATION_TOOLS:
                try:
                    screenshot = await executor.screenshot()
                except Exception as exc:
                    logger.warning("Screenshot failed: %s", exc)
            if screenshot:
                step["screenshot"] = screenshot

            self._steps.append(step)
            if self.event_emitter:
                await self.event_emitter.emit(
                    "tool_completed",
                    {"tool_name": tool_name, "result": result_text},
                    screenshot=screenshot,
                )
            return sdk_response

        except VerdictSubmitted as exc:
            if tool_name == "submit_group_verdict":
                validation_errors = self._validate_group_verdict_submission(exc.verdicts)
                if validation_errors:
                    self._sdk_verdict_validation_failures += 1
                    retry_message = self._build_group_verdict_retry_message(
                        validation_errors,
                        exc.verdicts,
                        attempt=self._sdk_verdict_validation_failures,
                    )
                    step["result"] = retry_message
                    step["success"] = False
                    step["verdict_validation_errors"] = validation_errors
                    step["submitted_group_verdicts"] = exc.verdicts
                    self._steps.append(step)
                    if self.event_emitter:
                        await self.event_emitter.emit("tool_failed", {"tool_name": tool_name, "error": retry_message})
                    if self._sdk_verdict_validation_failures > self._max_verdict_validation_retries:
                        final_reason = (
                            "submit_group_verdict validation failed after "
                            f"{self._sdk_verdict_validation_failures} attempts. "
                            f"Last errors: {'; '.join(validation_errors)}"
                        )
                        self._sdk_verdict_payload = {
                            "verdict": "failed",
                            "passed": False,
                            "reason": final_reason,
                            "group_verdicts": [],
                        }
                        return self._sdk_text_result(final_reason)
                    return self._sdk_text_result(retry_message)

            reason = self._sanitize_verdict_reason(exc.reason)
            group_verdicts = [
                {**item, "reason": self._sanitize_verdict_reason(str(item.get("reason", "")))}
                for item in exc.verdicts
            ]
            self._sdk_verdict_payload = {
                "verdict": exc.verdict,
                "passed": exc.passed,
                "reason": reason,
                "group_verdicts": group_verdicts,
            }
            step["result"] = f"{exc.verdict.upper()}: {reason}"
            step["success"] = exc.passed
            self._steps.append(step)
            if self.event_emitter:
                await self.event_emitter.emit("verdict_submitted", {"passed": exc.passed, "reason": reason})
            return self._sdk_text_result(step["result"])

        except Exception as exc:
            self._consecutive_infra_failures += 1
            step["result"] = str(exc)
            step["success"] = False
            step["semantic_action_signature"] = self._semantic_action_signature(tool_name, tool_args)
            self._steps.append(step)
            if (self._consecutive_infra_failures >= _INFRA_CONSECUTIVE_THRESHOLD
                    and _is_infra_error_message(str(exc))):
                logger.warning("Persistent infra failures (%d consecutive)", self._consecutive_infra_failures)
                raise InfraError(
                    f"Persistent infrastructure failures ({self._consecutive_infra_failures} consecutive): "
                    f"{exc}"
                ) from exc
            if self.event_emitter:
                await self.event_emitter.emit("tool_failed", {"tool_name": tool_name, "error": str(exc)})
            return self._sdk_text_result(f"Error executing tool: {exc}")

    async def _run_sdk_query(
        self,
        *,
        system_prompt: str,
        user_message: str,
        tool_names: List[str],
        max_turns: int,
        timeout_seconds: Optional[float] = None,
    ) -> Optional[Dict[str, Any]]:
        sdk = self._load_sdk_symbols()
        ClaudeSDKClient = sdk["ClaudeSDKClient"]

        self._sdk_verdict_payload = None
        self._sdk_verdict_validation_failures = 0

        options = self._build_sdk_options(
            system_prompt=system_prompt,
            tool_names=tool_names,
            max_turns=max_turns,
        )

        async def _run_session(client: Any, initial_message: str) -> None:
            """Run one agent session; returns normally on ResultMessage."""
            await client.query(initial_message)
            interrupt_sent = False
            async for message in client.receive_response():
                if self._sdk_verdict_payload and not interrupt_sent:
                    try:
                        await client.interrupt()
                        interrupt_sent = True
                    except Exception:
                        interrupt_sent = True
                if type(message).__name__ == "ResultMessage":
                    break

        async def _consume() -> None:
            async with ClaudeSDKClient(options=options) as client:
                await _run_session(client, user_message)

                # If the agent stopped without a verdict, force it to
                # submit one.  The SDK may end the session at any time
                # (e.g. the model returns end_turn without calling
                # any tool), so we re-engage the agent until either a
                # verdict arrives or we have clearly exhausted the
                # turn budget.
                verdict_tool = "submit_group_verdict" if self._required_standard_ids else "submit_verdict"
                force_prompt = (
                    "You previously stopped without submitting a final verdict. "
                    "Based only on the evidence already collected, call "
                    f"{verdict_tool} now with your best assessment."
                )
                max_force = max(1, max_turns // 2)
                for _ in range(max_force):
                    if self._sdk_verdict_payload:
                        break
                    remaining = len(self._steps)
                    if remaining >= max_turns:
                        break
                    await _run_session(client, force_prompt)

        async def _run_once() -> None:
            if timeout_seconds is not None:
                await asyncio.wait_for(_consume(), timeout=timeout_seconds)
            else:
                await _consume()

        await invoke_with_llm_api_retry(
            _run_once,
            retries=self.llm_api_retry_count,
            base_delay_seconds=self.llm_api_retry_base_delay_seconds,
            jitter_seconds=self.llm_api_retry_jitter_seconds,
            operation_name="Claude SDK query",
        )
        return self._sdk_verdict_payload

    async def evaluate(
        self,
        user_query: str,
        app_url: str,
        executor: Any,
        standard_ids: Optional[List[str]] = None,
        task_id: Optional[str] = None,
        task_title: Optional[str] = None,
        agent_config: Optional["AgentConfig"] = None,
    ) -> Dict[str, Any]:
        logger.info("Starting Claude SDK evaluation: %s", user_query)

        self._steps = []
        self._dom_elements = []
        self._initial_diagnostics = {}
        self._current_task_id = task_id
        self._current_task_title = task_title
        []
        self._verdict_validation_failures = 0
        self._sdk_executor = executor
        eval_start_ts = time.monotonic()

        try:
            ctx = await executor.get_context()
            self._initial_diagnostics = ctx.get("diagnostics", {})
            self._dom_elements = self._extract_interactive_elements(ctx.get("accessibility_tree", {}))
        except Exception as exc:
            logger.warning("Failed to capture initial DOM elements: %s", exc)

        system_prompt = format_system_prompt(
            user_query=user_query,
            app_url=app_url,
            max_iterations=self.max_iterations,
            standard_ids=standard_ids,
            agent_config=agent_config,
            allowed_tools=list(self.allowed_tools) if self.allowed_tools is not None else None,
        )
        tool_names = list(self.allowed_tools) if self.allowed_tools is not None else list(TOOL_REGISTRY.keys())
        verdict = await self._run_sdk_query(
            system_prompt=system_prompt,
            user_message="Begin the evaluation.",
            tool_names=tool_names,
            max_turns=self.max_iterations,
        )

        if not verdict:
            verdict = {
                "verdict": "failed",
                "passed": False,
                "reason": "Evaluation incomplete: Claude SDK session ended without verdict",
                "group_verdicts": [],
            }

        return {
            "verdict": verdict,
            "group_verdicts": verdict.get("group_verdicts", []),
            "iterations": len(self._steps),
            "steps": self._steps,
            "dom_elements": self._dom_elements,
            "initial_diagnostics": self._initial_diagnostics,
            "duration_ms": int((time.monotonic() - eval_start_ts) * 1000),
        }

    async def evaluate_agentic(
        self,
        agent_config: AgentConfig,
        app_url: str,
        executor: Any,
        task_context: Optional[str] = None,
    ) -> Dict[str, Any]:
        agent_id = agent_config.id
        dimension_ids = [d.id for d in agent_config.rubric.dimensions]
        start_ts = time.monotonic()

        logger.info("[%s] Starting Claude SDK agentic evaluation at %s", agent_id, app_url)

        self._steps = []
        self._dom_elements = []
        self._initial_diagnostics = {}
        []
        self._standard_scoring_requirements = self._build_scoring_requirements(agent_config)
        self._verdict_validation_failures = 0
        self._consecutive_infra_failures = 0
        self._sdk_executor = executor

        try:
            ctx = await executor.get_context()
            self._initial_diagnostics = ctx.get("diagnostics", {})
            self._dom_elements = self._extract_interactive_elements(ctx.get("accessibility_tree", {}))
        except Exception as exc:
            logger.warning("[%s] Failed to capture initial DOM: %s", agent_id, exc)

        system_prompt = format_agentic_prompt(
            agent_config,
            app_url,
            max_iterations=agent_config.runtime.max_steps,
        )
        saved_allowed = self.allowed_tools
        saved_max = self.max_iterations
        saved_step_timeout = self.max_step_seconds
        self.allowed_tools = set(agent_config.allowed_tools)
        self.max_iterations = agent_config.runtime.max_steps if agent_config.runtime.max_steps is not None else self.max_iterations
        self.max_step_seconds = agent_config.runtime.max_step_seconds

        status = "error"
        raw_group_verdicts: List[Dict[str, Any]] = []
        raw_summary_verdict: Optional[str] = None
        raw_summary_reason = ""
        iterations_used = 0
        end_reason = ""

        try:
            raw_verdict = await self._run_sdk_query(
                system_prompt=system_prompt,
                user_message=task_context or "Begin your evaluation.",
                tool_names=list(self.allowed_tools),
                max_turns=agent_config.runtime.max_steps,
                timeout_seconds=agent_config.runtime.max_total_seconds,
            )
            iterations_used = len(self._steps)
            if raw_verdict:
                raw_group_verdicts = raw_verdict.get("group_verdicts", [])
                raw_summary_verdict = str(raw_verdict.get("verdict", "")).strip().lower()
                raw_summary_reason = str(raw_verdict.get("reason", "")).strip()
                status = "completed"
                end_reason = "Agent submitted a verdict."
            else:
                if iterations_used >= agent_config.runtime.max_steps:
                    status = "max_steps_exceeded"
                    end_reason = "Agent exhausted its step budget without submitting a verdict."
                else:
                    status = "no_verdict_submitted"
                    end_reason = "Agent stopped without calling submit_verdict/submit_group_verdict."
        except asyncio.TimeoutError:
            logger.warning("[%s] Timed out after %ss", agent_id, agent_config.runtime.max_total_seconds)
            status = "timeout"
            iterations_used = len(self._steps)
            end_reason = "Agent timed out before submitting a verdict."
        except InfraError as exc:
            logger.warning("[%s] Infrastructure error: %s", agent_id, exc)
            status = "infra_error"
            iterations_used = len(self._steps)
            end_reason = f"Persistent infrastructure failures: {str(exc)}"
        except Exception as exc:
            logger.error("[%s] Evaluation error: %s", agent_id, exc)
            status = "error"
            iterations_used = len(self._steps)
            end_reason = f"Internal evaluator error: {str(exc)}"
        finally:
            self.allowed_tools = saved_allowed
            self.max_iterations = saved_max
            self.max_step_seconds = saved_step_timeout

        duration_ms = int((time.monotonic() - start_ts) * 1000)

        dimensions_out = self.build_dimension_results(
            agent_config=agent_config,
            dimension_ids=dimension_ids,
            raw_group_verdicts=raw_group_verdicts,
            raw_summary_verdict=raw_summary_verdict,
            raw_summary_reason=raw_summary_reason,
            status=status,
        )

        applicable = [
            d for d in dimensions_out
            if isinstance(d.get("score"), (int, float)) and float(d.get("weight", 0)) > 0
        ]
        if applicable:
            denom = sum(float(d["weight"]) for d in applicable)
            weighted_sum = sum(float(d["score"]) * float(d["weight"]) for d in applicable)
            agent_score = weighted_sum / denom if denom > 0 else None
        else:
            agent_score = None

        return {
            "agent_id": agent_id,
            "status": status,
            "runtime": {
                "steps_used": iterations_used,
                "duration_ms": duration_ms,
            },
            "end_reason": end_reason,
            "score": agent_score,
            "score_scale": "0-100",
            "dimensions": dimensions_out,
            "trajectory_ref": None,
            "trajectory": {
                "status": status,
                "steps": self._steps,
                "dom_elements": self._dom_elements,
                "initial_diagnostics": self._initial_diagnostics,
                "duration_ms": duration_ms,
            },
        }
