"""Main frontend evaluator agent using LangGraph ReAct pattern."""

import asyncio
import json
import os
import re
import time
from inspect import Parameter, signature
from typing import TYPE_CHECKING, Dict, Any, List, Optional, Set, Tuple
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from langgraph.graph import StateGraph, END

from .state import AgentState
from .prompts import format_system_prompt, format_agentic_prompt
from ..tools.registry import get_tool_definitions, get_tool_function
from ..tools.verdict import VerdictSubmitted
from ..tools.vision_inspector import set_vision_llm
from ..llm.factory import LLMFactory
from ..llm.exceptions import handle_provider_error
from ..llm.retry import invoke_with_llm_api_retry
from ..llm.message_utils import (
    get_model_name_from_llm,
    get_provider_from_llm,
    normalize_messages_for_provider,
    normalize_multimodal_content_for_provider,
)
from ..utils.logger import logger
from ..events.emitter import EventEmitter

# Import tools to register them
from ..tools import browser_actions, page_inspector, source_reader, verdict, vision_inspector, build_tools

if TYPE_CHECKING:
    from .config import AgentConfig


class InfraError(Exception):
    """Raised when persistent infrastructure failures prevent task completion.
    Signals the orchestrator that this task should be retried, not failed."""


_INFRA_ERROR_MARKERS = (
    "rate limit", "rate-limited", "ratelimit", "too many requests", "quota",
    "overloaded", "overload", "temporarily unavailable", "service unavailable",
    "server error", "internal server error", "bad gateway", "gateway timeout",
    "timeout", "timed out", "connection reset", "connection aborted",
    "connection error", "apiconnectionerror", "apitimeouterror",
    "429", "500", "502", "503", "504",
)


def _is_infra_error_message(message: str) -> bool:
    """Check if an error message indicates a transient infrastructure failure."""
    lowered = (message or "").lower()
    return any(marker in lowered for marker in _INFRA_ERROR_MARKERS)


OBSERVATION_TOOL_NAMES = {
    "get_page_context",
    "get_global_dom_summary",
    "inspect_last_screenshot",
    "preview_click_at",
}
BLIND_RETRY_GUARDED_TOOLS = {
    "click_element",
    "click_at",
    "dblclick_at",
    "dblclick_element",
    "right_click_element",
    "reload_page",
    "navigate_back",
    "navigate_forward",
}
MAX_SAME_ACTION_RETRIES_WITHOUT_OBSERVATION = 3

# Number of most recent agent-tool rounds to preserve in full fidelity.
# Older rounds are compressed by stripping image payloads (screenshots)
# while keeping textual content, so the conversation history stays
# within the model's context window.
CONTEXT_KEEP_ROUNDS = int(os.getenv("CONTEXT_KEEP_ROUNDS", "15"))
_INFRA_CONSECUTIVE_THRESHOLD = 3

# Max chars per JSON string field when truncating tool results (0 = disabled).
_TOOL_RESULT_FIELD_MAX_CHARS = int(os.getenv("TOOL_RESULT_FIELD_MAX_CHARS", "8000"))
# Max chars for the entire tool result string (0 = disabled).
_TOOL_RESULT_TOTAL_MAX_CHARS = int(os.getenv("TOOL_RESULT_TOTAL_MAX_CHARS", "20000"))
# Max chars persisted in trajectory step["result"]. Distinct from the LLM-bound
# truncation above — this caps what gets written to the JSONL report so single
# rows can't explode to hundreds of MB when tool output (npm install stdout etc.)
# is megabytes. Default 16 KB. Set 0 to disable.
_TRAJECTORY_STEP_RESULT_MAX_CHARS = int(os.getenv("TRAJECTORY_STEP_RESULT_MAX_CHARS", "16000"))


def _cap_step_result_for_persistence(value: Any) -> Any:
    """Cap string fields stored in ``step['result']`` to keep reports small."""
    limit = _TRAJECTORY_STEP_RESULT_MAX_CHARS
    if limit <= 0:
        return value
    if isinstance(value, str):
        if len(value) > limit:
            return value[:limit] + f"\n...[truncated {len(value) - limit} chars for report storage]"
        return value
    return value


class FrontendEvaluator:
    """Frontend evaluator agent using LangGraph ReAct pattern.

    Orchestrates the evaluation workflow:
    1. Agent reasons and decides which tools to use
    2. Tools are executed
    3. Results feed back to agent
    4. Loop continues until verdict submitted or max iterations reached
    """

    def __init__(
        self,
        api_key: str,
        model: Optional[str] = None,
        provider: str = "anthropic",
        base_url: Optional[str] = None,
        max_iterations: int = 10,
        max_verdict_validation_retries: int = 6,
        llm_api_retry_count: int = 5,
        llm_api_retry_base_delay_seconds: float = 1.0,
        llm_api_retry_jitter_seconds: float = 0.35,
        allowed_tools: Optional[List[str]] = None,
        log_level: Optional[str] = None,
        event_emitter: Optional[EventEmitter] = None,
        disable_thinking: bool = False,
        temperature: float = 0,
    ):
        """Initialize the evaluator agent.

        Args:
            api_key: API key for the LLM provider
            model: Model name (optional, uses provider default if not specified)
            provider: LLM provider (anthropic, openai, google, custom)
            base_url: Custom base URL (required for custom provider, optional for others)
            max_iterations: Maximum number of agent iterations
            max_verdict_validation_retries: Maximum retries after invalid grouped verdict submissions
            allowed_tools: Optional list of allowed tool names for this agent
            log_level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            event_emitter: Optional event emitter for monitoring
            disable_thinking: Disable extended thinking/reasoning for the LLM
            temperature: LLM temperature (only sent to models that support it)
        """
        self.api_key = api_key
        self.model = model
        self.provider = provider
        self.base_url = base_url
        self.max_iterations = max_iterations
        self.max_step_seconds: Optional[float] = None
        self.allowed_tools: Optional[Set[str]] = set(allowed_tools) if allowed_tools else None
        self.event_emitter = event_emitter
        self.llm_api_retry_count = max(0, int(llm_api_retry_count))
        self.llm_api_retry_base_delay_seconds = max(0.0, float(llm_api_retry_base_delay_seconds))
        self.llm_api_retry_jitter_seconds = max(0.0, float(llm_api_retry_jitter_seconds))
        self._steps: List[dict] = []
        self._dom_elements: List[dict] = []
        self._initial_diagnostics: Dict[str, Any] = {}
        self._current_task_id: Optional[str] = None
        self._current_task_title: Optional[str] = None
        self._required_standard_ids: List[str] = []
        self._standard_scoring_requirements: Dict[str, Dict[str, List[str]]] = {}
        self._verdict_validation_failures: int = 0
        self._max_verdict_validation_retries: int = max(0, int(max_verdict_validation_retries))
        self._consecutive_infra_failures: int = 0

        # Update logger level if provided
        if log_level:
            logger.setLevel(log_level.upper())
            for handler in logger.handlers:
                handler.setLevel(log_level.upper())

        # Initialize LLM using factory
        self.llm = LLMFactory.create_llm(
            provider=provider,
            api_key=api_key,
            model=model,
            base_url=base_url,
            temperature=temperature,
            disable_thinking=disable_thinking,
        )

        # Initialize vision LLM — use a separate multimodal model if configured
        vision_model_name = os.getenv("VISION_MODEL_NAME")
        if vision_model_name and vision_model_name != model:
            vision_llm = LLMFactory.create_llm(
                provider=provider,
                api_key=api_key,
                model=vision_model_name,
                base_url=base_url,
                temperature=temperature,
                disable_thinking=True,
            )
            set_vision_llm(vision_llm)
            self._vision_supported = True
        else:
            set_vision_llm(self.llm)
            self._vision_supported = True

        # Build LangGraph workflow
        self.workflow = self._build_workflow()

    def _is_tool_allowed(self, tool_name: str) -> bool:
        """Check whether a tool is allowed for this evaluator."""
        if self.allowed_tools is None:
            return True
        return tool_name.strip() in self.allowed_tools

    @staticmethod
    def _extract_interactive_elements(tree: dict) -> list:
        """Extract interactive elements from an accessibility tree snapshot."""
        elements = []
        seen_ids: set = set()
        interactive_roles = {
            "button", "link", "textbox", "checkbox", "radio",
            "combobox", "menuitem", "tab", "spinbutton", "slider",
        }

        def traverse(node):
            if not node:
                return
            role = node.get("role", "")
            name = node.get("name", "")
            if role in interactive_roles:
                elem_id = f"{role}:{name}" if name else f"{role}:{len(elements)}"
                if elem_id not in seen_ids:
                    seen_ids.add(elem_id)
                    elements.append({
                        "id": elem_id,
                        "role": role,
                        "name": name,
                        "value": node.get("value", ""),
                    })
            for child in node.get("children", []):
                traverse(child)

        traverse(tree)
        return elements

    @staticmethod
    def _parse_selector_id(selector: str) -> str:
        """Map a Playwright selector string to a canonical element ID."""
        s = selector.strip()
        m = re.match(r'^text=(.+)$', s)
        if m:
            return f"text:{m.group(1).strip()}"
        m = re.match(r'^role=(\w+)\[name=[\'"](.+?)[\'"]\]', s)
        if m:
            return f"{m.group(1)}:{m.group(2)}"
        m = re.match(r'^\[data-testid=[\'"](.+?)[\'"]\]', s)
        if m:
            return f"testid:{m.group(1)}"
        return s

    @staticmethod
    def _result_is_failure(result_text: str) -> bool:
        """Infer whether a tool result indicates failure."""
        text = (result_text or "").strip().lower()
        if not text:
            return False
        failure_markers = (
            " failed:",
            "timed out",
            "not found",
            "not clickable",
            "not hoverable",
            "error executing tool",
            "application not ready",
            "unable",
        )
        return any(marker in text for marker in failure_markers)

    @staticmethod
    def _reason_is_speculative(reason: str) -> bool:
        """Detect speculative verdict language that is not directly evidence-based."""
        text = (reason or "").strip().lower()
        if not text:
            return True
        speculative_markers = (
            "highly improbable",
            "likely",
            "probably",
            "suggests",
            "strongly suggests",
            "i was unable to directly test",
            "unable to directly test",
            "it is possible",
            "might be",
            "may be",
        )
        return any(marker in text for marker in speculative_markers)

    def _build_evidence_reason(self, fallback_prefix: str = "") -> str:
        """Build a concise factual reason from recorded tool steps."""
        failures: List[str] = []
        successes: List[str] = []
        for step in self._steps:
            tool = str(step.get("tool_name", "")).strip()
            result = str(step.get("result", "")).strip()
            if not result:
                continue
            normalized = f"{tool}: {result}" if tool else result
            if self._result_is_failure(result):
                failures.append(normalized)
            elif result.lower().startswith("successfully"):
                successes.append(normalized)

        if failures:
            detail = "; ".join(failures[:3])
            base = f"Verdict based on observed tool failures: {detail}."
            return f"{fallback_prefix}{base}".strip()

        if successes:
            detail = "; ".join(successes[:3])
            base = f"Verdict based on observed tool outcomes: {detail}."
            return f"{fallback_prefix}{base}".strip()

        return (fallback_prefix + "Verdict recorded without usable tool evidence.").strip()

    def _sanitize_verdict_reason(self, reason: str) -> str:
        """Ensure verdict reasons are factual and grounded in executed tool outputs."""
        text = (reason or "").strip()
        if self._reason_is_speculative(text):
            return self._build_evidence_reason()
        return text

    def _build_budget_reminder(self, remaining_turns: int) -> Optional[str]:
        if remaining_turns <= 0:
            return None

        verdict_tool = "submit_group_verdict" if self._required_standard_ids else "submit_verdict"
        if remaining_turns == 1:
            return (
                "This is your final reasoning turn within the step budget. "
                "Do not start new exploration. Based only on the evidence already collected, "
                f"your next response must call {verdict_tool}."
            )
        if remaining_turns == 2:
            return (
                "You have 2 reasoning turns remaining including this one. "
                "If current evidence is already sufficient, submit your verdict now. "
                "Otherwise use at most one final observation, then submit a verdict on the next turn."
            )
        return None

    def _should_block_wrong_verdict_tool(self, tool_name: str) -> Optional[str]:
        stripped = tool_name.strip()
        if self._required_standard_ids and stripped == "submit_verdict":
            joined_ids = ", ".join(self._required_standard_ids)
            return (
                "Blocked: this run requires per-standard verdicts for "
                f"[{joined_ids}]. You MUST use submit_group_verdict (not submit_verdict). "
                "On your next turn, call submit_group_verdict with one entry per standard_id."
            )
        if not self._required_standard_ids and stripped == "submit_group_verdict":
            return (
                "Blocked: this run does NOT use per-standard verdicts. "
                "You MUST use submit_verdict (not submit_group_verdict). "
                "On your next turn, call submit_verdict with a single verdict and reason."
            )
        return None

    @staticmethod
    def _is_verdict_tool_call(tool_call: Dict[str, Any]) -> bool:
        return str(tool_call.get("name", "")).strip() in {"submit_verdict", "submit_group_verdict"}

    @staticmethod
    def _build_scoring_requirements(agent_config: "AgentConfig") -> Dict[str, Dict[str, Any]]:
        requirements: Dict[str, Dict[str, Any]] = {}
        for dim in agent_config.rubric.dimensions:
            scoring = getattr(dim, "scoring", None)
            requirements[str(dim.id)] = {
                "check_ids": [str(chk.id) for chk in (getattr(scoring, "checks", None) or [])],
                "subcriterion_ids": [str(sub.id) for sub in (getattr(scoring, "subcriteria", None) or [])],
                "subcriterion_rating_values": list((getattr(scoring, "rating_to_score", None) or {}).keys()) or ["good", "ok", "poor"],
            }
        return requirements

    @staticmethod
    def _build_check_example_item(check_id: str) -> Dict[str, Any]:
        return {
            "check_id": check_id,
            "status": "passed",
            "reason": f"Observed outcome for {check_id}.",
            "evidence": [f"Tool result supporting {check_id}"],
        }

    @staticmethod
    def _build_subcriterion_example_item(subcriterion_id: str, rating_values: List[str]) -> Dict[str, Any]:
        rating = rating_values[0] if rating_values else "good"
        return {
            "subcriterion_id": subcriterion_id,
            "rating": rating,
            "reason": f"Assessment for {subcriterion_id}.",
            "evidence": [f"Evidence supporting {subcriterion_id}"],
        }

    def _build_group_verdict_example_payload(self) -> str:
        payload: List[Dict[str, Any]] = []
        for sid in self._required_standard_ids:
            scoring_req = self._standard_scoring_requirements.get(str(sid), {})
            item: Dict[str, Any] = {
                "standard_id": sid,
                "verdict": "passed",
                "reason": f"Concrete conclusion for {sid}.",
                "evidence": [f"Tool output or observed state for {sid}"],
            }
            check_ids = list(scoring_req.get("check_ids") or [])
            if check_ids:
                item["checks"] = [self._build_check_example_item(check_id) for check_id in check_ids]
            subcriterion_ids = list(scoring_req.get("subcriterion_ids") or [])
            if subcriterion_ids:
                rating_values = list(scoring_req.get("subcriterion_rating_values") or [])
                item["subcriteria"] = [
                    self._build_subcriterion_example_item(sub_id, rating_values)
                    for sub_id in subcriterion_ids
                ]
            payload.append(item)
        return json.dumps(payload, indent=2, ensure_ascii=True)

    @staticmethod
    def _is_observation_tool(tool_name: str) -> bool:
        return tool_name in OBSERVATION_TOOL_NAMES

    @staticmethod
    def _semantic_action_signature(tool_name: str, tool_args: Dict[str, Any]) -> Optional[str]:
        if tool_name not in BLIND_RETRY_GUARDED_TOOLS:
            return None

        normalized_args = {
            key: value
            for key, value in tool_args.items()
            if key not in {"token", "timeout", "reason"}
        }
        return f"{tool_name}:{json.dumps(normalized_args, sort_keys=True, ensure_ascii=True, default=str)}"

    def _analyze_retry_window(self, signature: str) -> Tuple[int, Optional[dict]]:
        attempts = 0
        last_matching_step: Optional[dict] = None
        for step in reversed(self._steps):
            step_tool = str(step.get("tool_name", ""))
            if self._is_observation_tool(step_tool):
                break

            if step.get("semantic_action_signature") != signature:
                continue

            attempts += 1
            if last_matching_step is None:
                last_matching_step = step

        return attempts, last_matching_step

    def _should_block_blind_retry(self, tool_name: str, tool_args: Dict[str, Any]) -> Optional[str]:
        signature = self._semantic_action_signature(tool_name, tool_args)
        if not signature:
            return None

        attempts_without_observation, last_matching_step = self._analyze_retry_window(signature)
        if attempts_without_observation == 0 or last_matching_step is None:
            return None

        if bool(last_matching_step.get("success")):
            if tool_name == "click_at":
                return (
                    "Blocked blind retry: the same coordinate click already succeeded without any new page observation. "
                    "If you intended a double-click, call dblclick_at instead. Otherwise inspect the page state before repeating this click."
                )
            return (
                "Blocked blind retry: the same action already succeeded without any new page observation. "
                "Inspect the page state before repeating the identical action."
            )

        if attempts_without_observation >= MAX_SAME_ACTION_RETRIES_WITHOUT_OBSERVATION:
            return (
                "Blocked blind retry: the same action has already been retried "
                f"{attempts_without_observation} times without any new page observation. "
                "Inspect the page state before trying again."
            )

        return None

    @staticmethod
    def _missing_required_tool_args(tool_func: Any, tool_args: Dict[str, Any]) -> List[str]:
        missing: List[str] = []
        for param_name, param in signature(tool_func).parameters.items():
            if param_name in {"self", "cls", "executor"}:
                continue
            if param.default is not Parameter.empty:
                continue
            if param_name not in tool_args:
                missing.append(param_name)
        return missing

    @staticmethod
    def _format_missing_tool_args_error(
        tool_name: str,
        missing_args: List[str],
        provided_args: Dict[str, Any],
    ) -> str:
        missing_list = ", ".join(missing_args)
        provided_list = ", ".join(sorted(provided_args.keys())) or "<none>"
        return (
            f"Invalid tool call for {tool_name}: missing required argument(s): {missing_list}. "
            f"Provided argument(s): {provided_list}. "
            "Retry the same tool with all required arguments present."
        )

    def _validate_group_verdict_submission(
        self,
        verdicts: List[Dict[str, Any]],
    ) -> List[str]:
        """Validate grouped verdict coverage and required fields for the current run."""
        if not self._required_standard_ids:
            return []

        errors: List[str] = []
        seen: Dict[str, int] = {}
        required = list(dict.fromkeys(str(item).strip() for item in self._required_standard_ids if str(item).strip()))
        required_set = set(required)

        if not isinstance(verdicts, list) or not verdicts:
            return ["no grouped verdict items were submitted"]

        for index, item in enumerate(verdicts, start=1):
            sid = str(item.get("standard_id", "")).strip()
            verdict = str(item.get("verdict", "")).strip().lower()
            if not sid:
                errors.append(f"item {index} is missing standard_id")
                continue
            seen[sid] = seen.get(sid, 0) + 1
            if sid not in required_set:
                errors.append(f"unexpected standard_id: {sid}")
            if verdict not in ("passed", "failed", "not_applicable"):
                errors.append(
                    f"item for {sid} must include verdict=passed|failed|not_applicable"
                )
            reason = item.get("reason")
            if not isinstance(reason, str) or not reason.strip():
                errors.append(f"item for {sid} must include a non-empty reason")
            evidence = item.get("evidence")
            if not isinstance(evidence, list) or not evidence:
                errors.append(f"item for {sid} must include a non-empty evidence array")

            scoring_req = self._standard_scoring_requirements.get(sid, {})
            required_check_ids = list(scoring_req.get("check_ids") or [])
            required_subcriterion_ids = list(scoring_req.get("subcriterion_ids") or [])

            if required_check_ids:
                checks = item.get("checks", [])
                if isinstance(checks, dict):
                    errors.append(
                        f"item for {sid} has invalid checks format: expected a list of objects like {{\"check_id\":\"...\",\"status\":\"passed|failed|not_applicable\",\"reason\":\"...\",\"evidence\":[]}}, got an object/dict"
                    )
                elif not isinstance(checks, list) or not checks:
                    errors.append(f"item for {sid} must include checks for: {', '.join(required_check_ids)}")
                else:
                    seen_check_ids = [
                        str(check.get("check_id", "")).strip()
                        for check in checks
                        if isinstance(check, dict) and str(check.get("check_id", "")).strip()
                    ]
                    for check in checks:
                        if not isinstance(check, dict):
                            errors.append(f"item for {sid} has invalid check entry: each check must be an object")
                            continue
                        status = str(check.get("status", "")).strip().lower()
                        if status not in ("passed", "failed", "not_applicable"):
                            check_id = str(check.get("check_id", "")).strip() or "<missing-check-id>"
                            errors.append(
                                f"item for {sid} has invalid checks status for {check_id}: expected passed|failed|not_applicable"
                            )
                        if not isinstance(check.get("reason"), str) or not str(check.get("reason", "")).strip():
                            check_id = str(check.get("check_id", "")).strip() or "<missing-check-id>"
                            errors.append(f"item for {sid} check {check_id} must include a non-empty reason")
                        if not isinstance(check.get("evidence"), list) or not check.get("evidence"):
                            check_id = str(check.get("check_id", "")).strip() or "<missing-check-id>"
                            errors.append(f"item for {sid} check {check_id} must include a non-empty evidence array")
                    missing_check_ids = [check_id for check_id in required_check_ids if check_id not in seen_check_ids]
                    duplicate_check_ids = sorted({check_id for check_id in seen_check_ids if seen_check_ids.count(check_id) > 1})
                    if missing_check_ids:
                        errors.append(f"item for {sid} is missing checks: {', '.join(missing_check_ids)}")
                    for check_id in duplicate_check_ids:
                        errors.append(f"item for {sid} has duplicate check_id: {check_id}")

            if required_subcriterion_ids:
                subcriteria = item.get("subcriteria", [])
                if isinstance(subcriteria, dict):
                    errors.append(
                        f"item for {sid} has invalid subcriteria format: expected a list of objects like {{\"subcriterion_id\":\"...\",\"rating\":\"good|ok|poor\",\"reason\":\"...\",\"evidence\":[]}}, got an object/dict"
                    )
                elif not isinstance(subcriteria, list) or not subcriteria:
                    errors.append(
                        f"item for {sid} must include subcriteria for: {', '.join(required_subcriterion_ids)}"
                    )
                else:
                    allowed_ratings = list(scoring_req.get("subcriterion_rating_values") or ["good", "ok", "poor"])
                    seen_subcriterion_ids = [
                        str(sub.get("subcriterion_id", "")).strip()
                        for sub in subcriteria
                        if isinstance(sub, dict) and str(sub.get("subcriterion_id", "")).strip()
                    ]
                    for sub in subcriteria:
                        if not isinstance(sub, dict):
                            errors.append(f"item for {sid} has invalid subcriteria entry: each subcriteria item must be an object")
                            continue
                        sub_id = str(sub.get("subcriterion_id", "")).strip() or "<missing-subcriterion-id>"
                        rating = str(sub.get("rating", "")).strip().lower()
                        if rating not in allowed_ratings:
                            errors.append(
                                f"item for {sid} has invalid subcriteria rating for {sub_id}: expected one of {', '.join(allowed_ratings)}"
                            )
                        if not isinstance(sub.get("reason"), str) or not str(sub.get("reason", "")).strip():
                            errors.append(f"item for {sid} subcriteria {sub_id} must include a non-empty reason")
                        if not isinstance(sub.get("evidence"), list) or not sub.get("evidence"):
                            errors.append(f"item for {sid} subcriteria {sub_id} must include a non-empty evidence array")
                    missing_subcriterion_ids = [
                        sub_id for sub_id in required_subcriterion_ids if sub_id not in seen_subcriterion_ids
                    ]
                    duplicate_subcriterion_ids = sorted(
                        {sub_id for sub_id in seen_subcriterion_ids if seen_subcriterion_ids.count(sub_id) > 1}
                    )
                    if missing_subcriterion_ids:
                        errors.append(
                            f"item for {sid} is missing subcriteria: {', '.join(missing_subcriterion_ids)}"
                        )
                    for sub_id in duplicate_subcriterion_ids:
                        errors.append(f"item for {sid} has duplicate subcriterion_id: {sub_id}")

        missing = [sid for sid in required if sid not in seen]
        for sid in missing:
            errors.append(f"missing standard_id: {sid}")

        duplicates = [sid for sid, count in seen.items() if count > 1]
        for sid in duplicates:
            errors.append(f"duplicate standard_id: {sid}")

        return errors

    @staticmethod
    def _summarize_group_verdict_submission(verdicts: List[Dict[str, Any]]) -> str:
        """Format grouped verdict items for retry guidance."""
        if not verdicts:
            return "- <no parsed verdict items>"

        lines: List[str] = []
        for index, item in enumerate(verdicts, start=1):
            sid = str(item.get("standard_id", "")).strip() or f"<missing-standard-id-{index}>"
            verdict = str(item.get("verdict", "")).strip() or "<missing-verdict>"
            reason = str(item.get("reason", "")).strip() or "<missing-reason>"
            reason_compact = re.sub(r"\s+", " ", reason)
            if len(reason_compact) > 160:
                reason_compact = reason_compact[:157] + "..."
            lines.append(f"- {sid}: verdict={verdict}; reason={reason_compact}")
        return "\n".join(lines)

    def _build_group_verdict_retry_message(
        self,
        errors: List[str],
        verdicts: List[Dict[str, Any]],
        *,
        attempt: int,
    ) -> str:
        """Build retry guidance for an invalid grouped verdict submission."""
        problems = "\n".join(f"- {item}" for item in errors)
        previous = self._summarize_group_verdict_submission(verdicts)
        required_schema_lines: List[str] = []
        for sid in self._required_standard_ids:
            scoring_req = self._standard_scoring_requirements.get(str(sid), {})
            required_check_ids = list(scoring_req.get("check_ids") or [])
            required_subcriterion_ids = list(scoring_req.get("subcriterion_ids") or [])
            parts = ["standard_id", "verdict", "reason", "evidence[]"]
            if required_check_ids:
                parts.append(f"checks[{', '.join(required_check_ids)}]")
            if required_subcriterion_ids:
                parts.append(f"subcriteria[{', '.join(required_subcriterion_ids)}]")
            required_schema_lines.append(f"- {sid}: " + "; ".join(parts))
        required_schema = "\n".join(required_schema_lines) or "- <no standard schema available>"
        example_payload = self._build_group_verdict_example_payload()
        return (
            "Validation failed for submit_group_verdict. You must resubmit the complete verdict list.\n\n"
            f"Attempt: {attempt}/{self._max_verdict_validation_retries + 1}\n"
            "Problems found:\n"
            f"{problems}\n\n"
            "Required schema for this run:\n"
            f"{required_schema}\n\n"
            "Copy this exact shape and replace the placeholder text with your actual conclusions and evidence:\n"
            f"```json\n{example_payload}\n```\n\n"
            "Your previous normalized submission:\n"
            f"{previous}\n\n"
            "Keep valid items unchanged unless you have a concrete reason to revise them. "
            "Fix only the invalid items and resubmit the full verdict list with one item per required standard_id."
        )

    @staticmethod
    def _score_from_verdict(verdict: str) -> Optional[float]:
        verdict_norm = str(verdict or "").strip().lower()
        if not verdict_norm:
            return None
        if verdict_norm == "passed":
            return 100.0
        if verdict_norm == "failed":
            return 0.0
        if verdict_norm == "unable":
            return 100.0  # not a failure — agent genuinely cannot evaluate
        if verdict_norm == "not_applicable":
            return None
        return None

    @staticmethod
    def _objective_signal_from_breakdown(score_breakdown: Dict[str, Any]) -> Optional[str]:
        objective_details = score_breakdown.get("objective_details")
        if not isinstance(objective_details, list) or not objective_details:
            return None

        applicable_statuses = [
            str(item.get("status", "")).strip().lower()
            for item in objective_details
            if isinstance(item, dict) and str(item.get("status", "")).strip().lower() != "not_applicable"
        ]
        if not applicable_statuses:
            return None

        passed_count = sum(1 for status in applicable_statuses if status == "passed")
        if passed_count == len(applicable_statuses):
            return "passed"
        if passed_count == 0:
            return "failed"
        return "partial"

    @classmethod
    def _downgrade_fallback_verdict(
        cls,
        verdict_value: Optional[str],
        score_breakdown: Dict[str, Any],
    ) -> Optional[str]:
        objective_signal = cls._objective_signal_from_breakdown(score_breakdown)
        if verdict_value == "passed" and objective_signal in {"failed", "partial"}:
            return objective_signal
        if verdict_value is None and objective_signal in {"failed", "partial", "passed"}:
            return objective_signal
        return verdict_value

    @staticmethod
    def _normalize_check_status(value: str) -> str:
        v = str(value or "").strip().lower()
        if v in {"pass", "passed", "ok", "success"}:
            return "passed"
        if v in {"na", "n/a", "not_applicable", "not-applicable"}:
            return "not_applicable"
        if v in {"fail", "failed", "error"}:
            return "failed"
        return "failed"

    def _compute_dimension_score(
        self,
        dimension_cfg: Any,
        verdict_value: Optional[str],
        payload: Dict[str, Any],
    ) -> Tuple[Optional[float], Dict[str, Any]]:
        """Compute weighted score for one dimension from config and model payload."""
        scoring_cfg = getattr(dimension_cfg, "scoring", None)
        verdict_score = self._score_from_verdict(verdict_value)

        # If dimension is explicitly NA, exclude from weighted score.
        if verdict_value == "not_applicable":
            return None, {
                "method": getattr(scoring_cfg, "method", "verdict_only") if scoring_cfg else "verdict_only",
                "final_score": None,
                "excluded": True,
                "reason": "Dimension marked not_applicable",
            }

        if not scoring_cfg:
            return verdict_score, {
                "method": "verdict_only",
                "final_score": verdict_score,
                "verdict_score": verdict_score,
            }

        checks_payload = payload.get("checks", [])
        if not isinstance(checks_payload, list):
            checks_payload = []
        checks_by_id = {
            str(item.get("check_id", "")).strip(): item
            for item in checks_payload
            if isinstance(item, dict)
        }

        subcriteria_payload = payload.get("subcriteria", [])
        if not isinstance(subcriteria_payload, list):
            subcriteria_payload = []
        subcriteria_by_id = {
            str(item.get("subcriterion_id", "")).strip(): item
            for item in subcriteria_payload
            if isinstance(item, dict)
        }

        objective_score: Optional[float] = None
        objective_details: List[Dict[str, Any]] = []
        if getattr(scoring_cfg, "checks", None):
            # When no checks were submitted at all (e.g. submit_verdict was used),
            # fall back to verdict-based scoring instead of defaulting all to failed.
            if not checks_payload:
                objective_score = verdict_score
                for chk in scoring_cfg.checks:
                    objective_details.append({
                        "check_id": chk.id,
                        "status": "not_applicable",
                        "weight": float(chk.weight),
                        "reported": False,
                        "note": "No checks submitted (single-verdict mode)",
                    })
            else:
                passed_weight = 0.0
                applicable_weight = 0.0
                for chk in scoring_cfg.checks:
                    chk_id = chk.id
                    reported = checks_by_id.get(chk_id, {})
                    status = self._normalize_check_status(reported.get("status", "failed"))
                    weight = float(chk.weight)
                    detail = {
                        "check_id": chk_id,
                        "status": status,
                        "weight": weight,
                        "reported": bool(reported),
                    }
                    objective_details.append(detail)
                    if status == "not_applicable":
                        continue
                    applicable_weight += weight
                    if status == "passed":
                        passed_weight += weight
                if applicable_weight > 0:
                    objective_score = (passed_weight / applicable_weight) * 100.0
                elif verdict_score is not None:
                    objective_score = verdict_score

        subjective_score: Optional[float] = None
        subjective_details: List[Dict[str, Any]] = []
        if getattr(scoring_cfg, "subcriteria", None):
            weighted_total = 0.0
            total_weight = 0.0
            mapping = dict(scoring_cfg.rating_to_score or {})
            for sub in scoring_cfg.subcriteria:
                sub_id = sub.id
                reported = subcriteria_by_id.get(sub_id, {})
                rating = str(reported.get("rating", "poor")).strip().lower()
                score = mapping.get(rating, mapping.get("poor", 40.0))
                weight = float(sub.weight)
                total_weight += weight
                weighted_total += weight * float(score)
                subjective_details.append({
                    "subcriterion_id": sub_id,
                    "rating": rating,
                    "score": float(score),
                    "weight": weight,
                    "reported": bool(reported),
                })
            if total_weight > 0:
                subjective_score = weighted_total / total_weight
            elif verdict_score is not None:
                subjective_score = verdict_score

        method = str(scoring_cfg.method)
        objective_weight = float(scoring_cfg.objective_weight)
        subjective_weight = float(scoring_cfg.subjective_weight)

        if method == "test_based":
            final_score = objective_score if objective_score is not None else verdict_score
        elif method == "rubric_based":
            final_score = subjective_score if subjective_score is not None else verdict_score
        elif method == "hybrid":
            used = []
            if objective_score is not None and objective_weight > 0:
                used.append(("objective", objective_score, objective_weight))
            if subjective_score is not None and subjective_weight > 0:
                used.append(("subjective", subjective_score, subjective_weight))
            if used:
                denom = sum(w for _, _, w in used)
                final_score = sum(score * w for _, score, w in used) / denom
            else:
                final_score = verdict_score
        else:
            final_score = verdict_score

        breakdown = {
            "method": method,
            "verdict_score": verdict_score,
            "objective_score": objective_score,
            "subjective_score": subjective_score,
            "weights": {
                "objective": objective_weight,
                "subjective": subjective_weight,
            },
            "objective_details": objective_details,
            "subjective_details": subjective_details,
            "final_score": final_score,
        }
        if final_score is not None:
            final_score = max(0.0, min(100.0, float(final_score)))
            breakdown["final_score"] = final_score
        return final_score, breakdown

    def build_dimension_results(
        self,
        *,
        agent_config: "AgentConfig",
        dimension_ids: List[str],
        raw_group_verdicts: List[Dict[str, Any]],
        raw_summary_verdict: Optional[str],
        raw_summary_reason: str,
        status: str,
    ) -> List[Dict[str, Any]]:
        """Build normalized dimension results for the requested dimensions.

        This is used by both full agent evaluation and task-level evaluation so that
        task rubric scoring follows the same verdict/scoring logic as agent scoring.
        """
        verdict_by_sid: dict[str, dict] = {}
        for item in raw_group_verdicts:
            sid = str(item.get("standard_id", "")).strip()
            if sid and sid in dimension_ids:
                verdict_by_sid[sid] = item

        fallback_reason = {
            "timeout": "Agent timed out before submitting a verdict.",
            "infra_error": "Task incomplete due to infrastructure failures.",
            "max_steps_exceeded": "Agent exhausted its step budget without submitting a verdict.",
            "no_verdict_submitted": "Agent stopped without submitting any verdict.",
            "error": "An internal error occurred during evaluation.",
            "completed": "Dimension not reported by agent.",
        }.get(status, "No verdict submitted.")

        default_item: Optional[dict] = None
        if not verdict_by_sid and raw_summary_verdict in ("passed", "failed", "not_applicable"):
            default_item = {
                "verdict": raw_summary_verdict,
                "reason": raw_summary_reason or "Agent submitted only an overall verdict.",
                "evidence": [],
            }
        if not default_item:
            for item in raw_group_verdicts:
                if not str(item.get("standard_id", "")).strip():
                    default_item = dict(item)
                    break

        dimensions_out: list[dict] = []
        dimension_cfg_by_id = {d.id: d for d in agent_config.rubric.dimensions}
        for dim_id in dimension_ids:
            dim_cfg = dimension_cfg_by_id[dim_id]
            if dim_id in verdict_by_sid:
                item = verdict_by_sid[dim_id]
                raw_verdict = str(item.get("verdict", "")).lower().strip()
                if raw_verdict in ("passed", "failed", "not_applicable"):
                    verdict_value: Optional[str] = raw_verdict
                else:
                    verdict_value = None

                evidence = item.get("evidence", [])
                if not isinstance(evidence, list):
                    evidence = []

                if verdict_value == "not_applicable" and not agent_config.output.allow_not_applicable:
                    verdict_value = None
                    reason = (
                        "Agent returned not_applicable, but this agent does not allow "
                        "not_applicable verdicts."
                    )
                    evidence = []
                else:
                    reason = self._sanitize_verdict_reason(str(item.get("reason", "")))

                if (
                    agent_config.output.require_evidence
                    and verdict_value != "not_applicable"
                    and not evidence
                ):
                    reason = (
                        "Evidence is required for this agent, but no evidence was provided "
                        "for this dimension."
                    )

                extra_fields = {
                    key: value
                    for key, value in item.items()
                    if key not in {"standard_id", "verdict", "reason", "evidence"}
                }
                score, score_breakdown = self._compute_dimension_score(
                    dimension_cfg=dim_cfg,
                    verdict_value=verdict_value,
                    payload=item,
                )
                dimensions_out.append({
                    "dimension_id": dim_id,
                    "verdict": verdict_value,
                    "reason": reason,
                    "evidence": evidence,
                    "score": score,
                    "score_breakdown": score_breakdown,
                    "weight": float(dim_cfg.weight),
                    "verdict_source": "dimension_specific",
                    **extra_fields,
                })
            elif default_item:
                raw_verdict = str(default_item.get("verdict", "")).lower().strip()
                if raw_verdict in ("passed", "failed", "not_applicable"):
                    verdict_value = raw_verdict
                else:
                    verdict_value = None
                if verdict_value == "not_applicable" and not agent_config.output.allow_not_applicable:
                    verdict_value = None
                evidence = default_item.get("evidence", [])
                if not isinstance(evidence, list):
                    evidence = []
                reason = self._sanitize_verdict_reason(
                    str(default_item.get("reason", "Agent submitted only an overall verdict."))
                )
                if (
                    agent_config.output.require_evidence
                    and verdict_value != "not_applicable"
                    and not evidence
                ):
                    reason = (
                        "Evidence is required for this agent, but no evidence was provided "
                        "for this dimension."
                    )
                extra_fields = {
                    key: value
                    for key, value in default_item.items()
                    if key not in {"standard_id", "verdict", "reason", "evidence"}
                }
                score, score_breakdown = self._compute_dimension_score(
                    dimension_cfg=dim_cfg,
                    verdict_value=verdict_value,
                    payload=default_item,
                )
                downgraded_verdict = self._downgrade_fallback_verdict(verdict_value, score_breakdown)
                if downgraded_verdict != verdict_value:
                    verdict_value = downgraded_verdict
                    score, score_breakdown = self._compute_dimension_score(
                        dimension_cfg=dim_cfg,
                        verdict_value=verdict_value,
                        payload=default_item,
                    )
                dimensions_out.append({
                    "dimension_id": dim_id,
                    "verdict": verdict_value,
                    "reason": reason,
                    "evidence": evidence,
                    "score": score,
                    "score_breakdown": score_breakdown,
                    "weight": float(dim_cfg.weight),
                    "verdict_source": "agent_level_fallback",
                    **extra_fields,
                })
            else:
                fallback_verdict: Optional[str]
                if status in {"timeout", "max_steps_exceeded", "no_verdict_submitted", "error"}:
                    fallback_verdict = "failed"
                else:
                    fallback_verdict = None
                score, score_breakdown = self._compute_dimension_score(
                    dimension_cfg=dim_cfg,
                    verdict_value=fallback_verdict,
                    payload={},
                )
                dimensions_out.append({
                    "dimension_id": dim_id,
                    "verdict": fallback_verdict,
                    "reason": fallback_reason,
                    "evidence": [],
                    "score": score,
                    "score_breakdown": score_breakdown,
                    "weight": float(dim_cfg.weight),
                    "verdict_source": "system_fallback",
                })
        return dimensions_out

    @staticmethod
    def _truncate_tool_result(content: Any) -> Any:
        """Truncate tool result content to prevent oversized LLM requests.

        If the content looks like a JSON object, each string value is truncated
        individually so the model still sees the field structure.  Plain strings
        are truncated as a whole.  Non-string/non-dict content is returned as-is.
        """
        field_max = _TOOL_RESULT_FIELD_MAX_CHARS
        total_max = _TOOL_RESULT_TOTAL_MAX_CHARS

        def _trunc(s: str, limit: int) -> str:
            if limit <= 0 or len(s) <= limit:
                return s
            return s[:limit] + "...[truncated]"

        if isinstance(content, str):
            # Try to parse as JSON for field-level truncation.
            if field_max > 0 and content.lstrip().startswith("{"):
                try:
                    data = json.loads(content)
                    if isinstance(data, dict):
                        changed = False
                        for key, val in data.items():
                            if isinstance(val, str):
                                truncated = _trunc(val, field_max)
                                if truncated is not val:
                                    data[key] = truncated
                                    changed = True
                        if changed:
                            content = json.dumps(data, ensure_ascii=False)
                except (json.JSONDecodeError, ValueError):
                    pass
            # Apply total cap after field-level truncation (or if not JSON).
            return _trunc(content, total_max)

        return content

    @staticmethod
    def _compress_old_messages(messages: List) -> List:
        """Compress old conversation rounds to stay within the context window.

        Keeps the last ``CONTEXT_KEEP_ROUNDS`` agent-tool rounds intact.
        For older rounds, strips image payloads from ``HumanMessage``
        followups while preserving textual content — the image is already
        redundant with the ``ToolMessage`` text (e.g. ``inspect_last_screenshot``
        returns a textual description).

        A "round" is one ``AIMessage`` with tool calls plus the ``ToolMessage``
        and optional ``HumanMessage`` followups that immediately follow it.
        Messages before the first tool-calling ``AIMessage`` (the initial
        instruction) are always preserved unchanged.
        """
        if len(messages) <= 3:
            return messages

        # Locate rounds: each AIMessage with tool_calls starts a new round.
        rounds: List[Tuple[int, int]] = []  # (start_idx, end_idx_exclusive)
        round_start: int | None = None
        for i, msg in enumerate(messages):
            if isinstance(msg, AIMessage) and getattr(msg, 'tool_calls', None):
                if round_start is not None:
                    rounds.append((round_start, i))
                round_start = i
        if round_start is not None:
            rounds.append((round_start, len(messages)))

        if len(rounds) <= CONTEXT_KEEP_ROUNDS:
            return messages

        # Cutoff: messages from this index onward are kept intact.
        rounds_to_keep = len(rounds) - CONTEXT_KEEP_ROUNDS
        cutoff_idx = rounds[rounds_to_keep][0]

        compressed: List = []
        for i, msg in enumerate(messages):
            if i >= cutoff_idx:
                compressed.append(msg)
            elif i < rounds[0][0]:
                compressed.append(msg)
            elif isinstance(msg, HumanMessage) and isinstance(msg.content, list):
                text_only = [
                    item for item in msg.content
                    if isinstance(item, dict) and item.get("type") == "text"
                ]
                if text_only:
                    compressed.append(HumanMessage(content=text_only))
            else:
                compressed.append(msg)

        return compressed

    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow.

        Returns:
            Compiled StateGraph
        """
        # Create graph
        graph = StateGraph(AgentState)

        # Add nodes
        graph.add_node("agent", self._agent_node)
        graph.add_node("tools", self._tools_node)

        # Set entry point
        graph.set_entry_point("agent")

        # Add edges
        graph.add_conditional_edges(
            "agent",
            self._should_continue,
            {
                "continue": "tools",
                "end": END,
            },
        )
        graph.add_edge("tools", "agent")

        return graph.compile()

    async def _agent_node(self, state: AgentState) -> Dict[str, Any]:
        """Agent reasoning node - calls Claude to decide next action.

        Args:
            state: Current agent state

        Returns:
            Updated state with new message
        """
        logger.info(f"Agent iteration {state['iteration']}/{state['max_iterations']}")

        # Get tool definitions
        tools = get_tool_definitions(self.allowed_tools)

        # Bind tools to LLM
        llm_with_tools = self.llm.bind_tools(tools)

        # Debug: Log message history being sent to LLM
        logger.debug(f"Message history length: {len(state['messages'])}")
        for i, msg in enumerate(state['messages']):
            msg_type = type(msg).__name__
            content_preview = str(msg.content)[:100] if hasattr(msg, 'content') else 'N/A'
            logger.debug(f"  Message {i}: {msg_type} - {content_preview}...")

        # Call LLM with tools
        try:
            llm_messages = list(state["messages"])
            llm_messages = self._compress_old_messages(llm_messages)
            remaining_turns = max(0, int(state["max_iterations"]) - int(state["iteration"]))
            budget_reminder = self._build_budget_reminder(remaining_turns)
            if budget_reminder:
                llm_messages.append(HumanMessage(content=budget_reminder))
            llm_messages = normalize_messages_for_provider(
                llm_messages,
                self.provider,
                model_name=get_model_name_from_llm(self.llm) or self.model,
            )

            # --- LLM call instrumentation (for reproducing failures) ---
            _call_log_path = os.getenv(
                "LLM_CALL_LOG_PATH", "llm_calls.log"
            )
            _call_id = f"call_{int(time.time()*1000)}_{id(llm_messages) & 0xffffff:x}"

            def _safe_serialize_messages(msgs):
                out = []
                for m in msgs:
                    try:
                        content = m.content if hasattr(m, "content") else None
                        # content 可能是 str / list (multimodal) / None
                        if isinstance(content, list):
                            content_repr = []
                            for part in content:
                                if isinstance(part, dict):
                                    if part.get("type") == "image_url":
                                        url = part.get("image_url", {}).get("url", "")
                                        # 截断 base64，只留前缀和长度
                                        if isinstance(url, str) and url.startswith("data:"):
                                            content_repr.append({
                                                "type": "image_url",
                                                "data_len": len(url),
                                                "prefix": url[:80],
                                            })
                                        else:
                                            content_repr.append({
                                                "type": "image_url",
                                                "url_len": len(str(url)),
                                            })
                                    else:
                                        content_repr.append(part)
                                else:
                                    content_repr.append(part)
                        else:
                            content_repr = content
                        out.append({
                            "type": type(m).__name__,
                            "content": content_repr,
                            "tool_calls": getattr(m, "tool_calls", None),
                            "tool_call_id": getattr(m, "tool_call_id", None),
                            "id": getattr(m, "id", None),
                        })
                    except Exception as _e:
                        out.append({"type": type(m).__name__, "_serialize_error": str(_e)})
                return out

            _callable_tools = [t.get("function", {}).get("name", "?") for t in tools] if tools else []
            _t0 = time.time()
            try:
                # 落盘输入快照（请求发出前）
                os.makedirs(os.path.dirname(_call_log_path), exist_ok=True)
                with open(_call_log_path, "a") as _f:
                    _f.write(json.dumps({
                        "call_id": _call_id,
                        "phase": "request",
                        "ts": _t0,
                        "ts_iso": time.strftime("%Y-%m-%dT%H:%M:%S"),
                        "model": get_model_name_from_llm(self.llm) or self.model,
                        "provider": self.provider,
                        "iteration": state["iteration"],
                        "max_iterations": state["max_iterations"],
                        "tools": _callable_tools,
                        "messages": _safe_serialize_messages(llm_messages),
                    }, ensure_ascii=False, default=str) + "\n")
            except Exception:
                pass

            _llm_exc = None
            try:
                response = await invoke_with_llm_api_retry(
                    lambda: llm_with_tools.ainvoke(llm_messages),
                    retries=self.llm_api_retry_count,
                    base_delay_seconds=self.llm_api_retry_base_delay_seconds,
                    jitter_seconds=self.llm_api_retry_jitter_seconds,
                    operation_name="Evaluator LLM request",
                )
            except Exception as _e:
                _llm_exc = _e
                response = None
            _dt = time.time() - _t0

            try:
                with open(_call_log_path, "a") as _f:
                    _resp_rec = {"call_id": _call_id, "phase": "response",
                                 "ts": time.time(), "elapsed_s": round(_dt, 2)}
                    if _llm_exc is not None:
                        _resp_rec["status"] = "failed"
                        _resp_rec["error_type"] = type(_llm_exc).__name__
                        _resp_rec["error"] = str(_llm_exc)[:2000]
                    else:
                        _resp_rec["status"] = "ok"
                        _resp_rec["resp_content_len"] = len(response.content) if (response is not None and hasattr(response, "content") and response.content) else 0
                        _resp_rec["resp_tool_calls"] = ([
                            {"name": tc.get("name"), "args_len": len(json.dumps(tc.get("args", {}), default=str))}
                            for tc in (response.tool_calls or [])
                        ] if hasattr(response, "tool_calls") else None)
                        # token usage tally (gemini等返回 usage；超 TOKEN_LIMIT_KILL 则写 stop 标记)
                        try:
                            _tu = None
                            if hasattr(response, "usage_metadata") and response.usage_metadata:
                                _tu = response.usage_metadata
                            elif hasattr(response, "response_metadata"):
                                _rm = response.response_metadata or {}
                                _tu = (_rm.get("token_usage") or _rm.get("usage"))
                            if _tu and isinstance(_tu, dict):
                                _pt = _tu.get("input_tokens", _tu.get("prompt_tokens", 0)) or 0
                                _ct = _tu.get("output_tokens", _tu.get("completion_tokens", 0)) or 0
                                _resp_rec["usage"] = {"input": _pt, "output": _ct}
                        except Exception:
                            pass
                    _f.write(json.dumps(_resp_rec, ensure_ascii=False, default=str) + "\n")
            except Exception:
                pass

            if _llm_exc is not None:
                raise _llm_exc

            # Debug: Log response details
            logger.debug(f"LLM response type: {type(response)}")
            logger.debug(f"LLM response content: {response.content if hasattr(response, 'content') else 'N/A'}")
            logger.debug(f"LLM response has tool_calls: {hasattr(response, 'tool_calls')}")
            if hasattr(response, 'tool_calls'):
                logger.debug(f"LLM response tool_calls: {response.tool_calls}")

        except Exception as e:
            error_msg = handle_provider_error(e, self.provider)
            logger.error(f"LLM API error: {error_msg}")
            self._consecutive_infra_failures += 1
            is_infra = _is_infra_error_message(error_msg)
            prefix = "[INFRA] " if is_infra else ""
            if self._consecutive_infra_failures >= _INFRA_CONSECUTIVE_THRESHOLD and is_infra:
                raise InfraError(
                    f"Persistent infrastructure failures ({self._consecutive_infra_failures} consecutive): "
                    f"{error_msg}"
                ) from e
            raise RuntimeError(f"{prefix}{error_msg}") from e

        # Increment iteration
        return {
            "messages": [response],
            "iteration": state["iteration"] + 1,
        }

    async def _tools_node(self, state: AgentState) -> Dict[str, Any]:
        """Tools execution node - executes tool calls from agent.

        Args:
            state: Current agent state

        Returns:
            Updated state with tool results
        """
        last_message = state["messages"][-1]
        tool_calls = last_message.tool_calls

        if not tool_calls:
            verdict_tool = "submit_group_verdict" if self._required_standard_ids else "submit_verdict"
            # Nudge the model to either submit a verdict or propose a next tool call.
            nudge = (
                "You returned no tool calls. If you have enough evidence, call "
                f"{verdict_tool} now. Otherwise, propose and execute "
                "at least one next tool call (e.g., get_page_context, click_element, "
                "inspect_last_screenshot) to gather evidence toward a verdict."
            )
            return {"messages": [HumanMessage(content=nudge)]}
        tool_messages = []
        deferred_followup_messages = []
        executor = state["executor"]

        # Update event emitter iteration
        if self.event_emitter:
            self.event_emitter.set_iteration(state["iteration"])

        for tool_call in tool_calls:
            tool_name = tool_call["name"].strip()
            tool_args = tool_call["args"]
            tool_id = tool_call["id"]

            _args_preview = str(tool_args)
            if len(_args_preview) > 300:
                _args_preview = _args_preview[:300] + "…"
            logger.info(f"Executing tool: {tool_name} with args: {_args_preview}")

            # Emit tool executing event
            if self.event_emitter:
                await self.event_emitter.emit(
                    "tool_executing",
                    {
                        "tool_name": tool_name,
                        "tool_args": tool_args,
                        "step_count": len(self._steps) + 1,
                    },
                )

            step = {
                "tool_name": tool_name,
                "args": {k: str(v) for k, v in tool_args.items()},
                "timestamp": time.time(),
            }
            if self._current_task_id:
                step["task_id"] = self._current_task_id
            if self._current_task_title:
                step["task_title"] = self._current_task_title

            if not self._is_tool_allowed(tool_name):
                error_msg = (
                    f"Permission denied: tool '{tool_name}' is not allowed for this agent"
                )
                step["result"] = error_msg
                step["success"] = False
                logger.warning(error_msg)

                if self.event_emitter:
                    await self.event_emitter.emit(
                        "tool_failed",
                        {
                            "tool_name": tool_name,
                            "error": error_msg,
                            "step_count": len(self._steps) + 1,
                        },
                    )

                tool_messages.append(
                    ToolMessage(
                        content=f"Error executing tool: {error_msg}",
                        tool_call_id=tool_id,
                    )
                )
                self._steps.append(step)
                continue

            wrong_verdict_tool_error = self._should_block_wrong_verdict_tool(tool_name)
            if wrong_verdict_tool_error:
                step["result"] = wrong_verdict_tool_error
                step["success"] = False
                logger.warning(wrong_verdict_tool_error)

                if self.event_emitter:
                    await self.event_emitter.emit(
                        "tool_failed",
                        {
                            "tool_name": tool_name,
                            "error": wrong_verdict_tool_error,
                            "step_count": len(self._steps) + 1,
                        },
                    )

                tool_messages.append(
                    ToolMessage(
                        content=wrong_verdict_tool_error,
                        tool_call_id=tool_id,
                    )
                )
                self._steps.append(step)
                continue

            blind_retry_error = self._should_block_blind_retry(tool_name, tool_args)
            if blind_retry_error:
                step["result"] = blind_retry_error
                step["success"] = False
                step["semantic_action_signature"] = self._semantic_action_signature(tool_name, tool_args)
                logger.warning(blind_retry_error)

                if self.event_emitter:
                    await self.event_emitter.emit(
                        "tool_failed",
                        {
                            "tool_name": tool_name,
                            "error": blind_retry_error,
                            "step_count": len(self._steps) + 1,
                        },
                    )

                tool_messages.append(
                    ToolMessage(
                        content=blind_retry_error,
                        tool_call_id=tool_id,
                    )
                )
                self._steps.append(step)
                continue

            try:
                # Get tool function
                tool_func = get_tool_function(tool_name)

                missing_args = self._missing_required_tool_args(tool_func, tool_args)
                if missing_args:
                    error_msg = self._format_missing_tool_args_error(tool_name, missing_args, tool_args)
                    step["result"] = error_msg
                    step["success"] = False
                    step["semantic_action_signature"] = self._semantic_action_signature(tool_name, tool_args)

                    logger.warning(error_msg)

                    if self.event_emitter:
                        await self.event_emitter.emit(
                            "tool_failed",
                            {
                                "tool_name": tool_name,
                                "error": error_msg,
                                "step_count": len(self._steps) + 1,
                            },
                        )

                    tool_messages.append(
                        ToolMessage(
                            content=error_msg,
                            tool_call_id=tool_id,
                        )
                    )
                    self._steps.append(step)
                    continue

                # CDP health check: before injecting the executor into a tool
                # call, verify the underlying CDP WebSocket is still alive.
                # If it went stale (e.g. during LLM inference), attempt recovery.
                if "executor" in signature(tool_func).parameters:
                    health_check = getattr(executor, "check_health", None)
                    if health_check is not None and not await health_check():
                        recover = getattr(executor, "recover", None)
                        recovered = await recover() if recover else False
                        if not recovered:
                            error_msg = (
                                "Browser CDP connection was lost and could not be "
                                "recovered automatically. Please retry this action."
                            )
                            step["result"] = error_msg
                            step["success"] = False
                            logger.warning("CDP connection dead for tool '%s', recovery failed", tool_name)
                            if self.event_emitter:
                                await self.event_emitter.emit("tool_failed", {
                                    "tool_name": tool_name, "error": error_msg,
                                    "step_count": len(self._steps) + 1,
                                })
                            tool_messages.append(ToolMessage(content=error_msg, tool_call_id=tool_id))
                            self._steps.append(step)
                            # CDP recovery failed — the browser session is dead and
                            # retrying the same action will not succeed.  Raise
                            # InfraError so the evaluator reclassifies this task as
                            # infra_error / inconclusive and the orchestrator's retry
                            # loop rebuilds the workspace (dev server + CDP) before
                            # re-attempting the task.
                            raise InfraError(
                                f"CDP connection lost and recovery failed for tool '{tool_name}': {error_msg}"
                            )

                # Inject executor based on function signature to avoid name allowlists.
                call_coro = None
                if "executor" in signature(tool_func).parameters:
                    call_coro = tool_func(executor, **tool_args)
                else:
                    call_coro = tool_func(**tool_args)

                if self.max_step_seconds is not None:
                    result = await asyncio.wait_for(call_coro, timeout=self.max_step_seconds)
                else:
                    result = await call_coro

                tool_message_content: Any = str(result)
                followup_messages: List[Any] = []
                result_text = str(result)
                if isinstance(result, dict) and "tool_message_content" in result:
                    multimodal_content = result.get("tool_message_content")
                    result_text = str(result.get("display_text") or result.get("result") or "").strip() or str(result)

                    image_parts: List[Dict[str, Any]] = []
                    if isinstance(multimodal_content, list):
                        for item in multimodal_content:
                            if isinstance(item, dict) and item.get("type") in ("image_url", "text"):
                                image_parts.append(item)

                    tool_message_content = result_text
                    has_images = any(p.get("type") == "image_url" for p in image_parts)
                    if has_images and os.getenv("VISION_MODEL_NAME"):
                        image_parts = [p for p in image_parts if p.get("type") != "image_url"]
                    if image_parts:
                        followup_text = (
                            f"Visual attachment from the just-completed tool '{tool_name}'. "
                            "Use this image as additional observation for your next step. "
                            "The preceding tool message contains the textual summary."
                        )
                        if tool_name == "preview_click_at":
                            followup_text = (
                                f"Visual attachment from {tool_name}. Inspect the red marker position in this preview image yourself before deciding "
                                "whether to call click_at, dblclick_at, inspect_last_screenshot, or another tool. "
                                "The preceding tool message contains the textual summary."
                            )
                        followup_content = [
                            {"type": "text", "text": followup_text},
                            *image_parts,
                        ]
                        followup_messages.append(
                            HumanMessage(
                                content=normalize_multimodal_content_for_provider(
                                    followup_content,
                                    get_provider_from_llm(self.llm),
                                )
                            )
                        )

                step["result"] = _cap_step_result_for_persistence(result_text)
                step["success"] = True
                step["semantic_action_signature"] = self._semantic_action_signature(tool_name, tool_args)
                self._consecutive_infra_failures = 0

                # For interaction tools, record element ID
                selector_arg_name = browser_actions.SELECTOR_ARG_BY_TOOL.get(tool_name)
                if selector_arg_name:
                    step["element_id"] = self._parse_selector_id(
                        tool_args.get(selector_arg_name, "")
                    )

                _rt_preview = result_text[:300] + "…" if len(result_text) > 300 else result_text
                logger.info(f"Tool result: {_rt_preview}")

                # Capture screenshot and store it inline in the report step payload.
                screenshot = None
                if tool_name in browser_actions.PAGE_MUTATION_TOOLS:
                    try:
                        screenshot = await executor.screenshot()
                    except Exception as e:
                        logger.warning(f"Screenshot failed: {e}")
                if screenshot:
                    step["screenshot"] = screenshot

                # Emit tool completed event
                if self.event_emitter:
                    await self.event_emitter.emit(
                        "tool_completed",
                        {
                            "tool_name": tool_name,
                            "result": result_text,
                            "step_count": len(self._steps) + 1,
                        },
                        screenshot=screenshot,
                    )

                # Create tool message
                tool_messages.append(
                    ToolMessage(
                        content=self._truncate_tool_result(tool_message_content),
                        tool_call_id=tool_id,
                    )
                )
                if followup_messages:
                    deferred_followup_messages.extend(followup_messages)

            except VerdictSubmitted as e:
                if tool_name == "submit_group_verdict":
                    validation_errors = self._validate_group_verdict_submission(e.verdicts)
                    if validation_errors:
                        self._verdict_validation_failures += 1
                        retry_message = self._build_group_verdict_retry_message(
                            validation_errors,
                            e.verdicts,
                            attempt=self._verdict_validation_failures,
                        )
                        step["result"] = retry_message
                        step["success"] = False
                        step["verdict_validation_errors"] = validation_errors
                        step["submitted_group_verdicts"] = e.verdicts
                        self._steps.append(step)

                        logger.warning(
                            "Grouped verdict validation failed: %s",
                            "; ".join(validation_errors),
                        )

                        if self.event_emitter:
                            await self.event_emitter.emit(
                                "tool_failed",
                                {
                                    "tool_name": tool_name,
                                    "error": retry_message,
                                    "step_count": len(self._steps) + 1,
                                },
                            )

                        if self._verdict_validation_failures > self._max_verdict_validation_retries:
                            final_reason = (
                                "submit_group_verdict validation failed after "
                                f"{self._verdict_validation_failures} attempts. "
                                f"Last errors: {'; '.join(validation_errors)}"
                            )
                            step["result"] = final_reason
                            state["verdict"] = {
                                "verdict": "failed",
                                "passed": False,
                                "reason": final_reason,
                                "group_verdicts": [],
                            }
                            tool_messages.append(
                                ToolMessage(
                                    content=final_reason,
                                    tool_call_id=tool_id,
                                )
                            )
                        else:
                            tool_messages.append(
                                ToolMessage(
                                    content=retry_message,
                                    tool_call_id=tool_id,
                                )
                            )
                        continue

                step["result"] = f"{e.verdict.upper()}: {e.reason}"
                step["success"] = e.passed
                self._steps.append(step)

                logger.info(f"Verdict submitted: {e.verdict} - {e.reason}")

                state["verdict"] = {
                    "verdict": e.verdict,
                    "passed": e.passed,
                    "reason": e.reason,
                    "group_verdicts": e.verdicts,
                }

                # Emit verdict submitted event
                if self.event_emitter:
                    await self.event_emitter.emit(
                        "verdict_submitted",
                        {
                            "passed": e.passed,
                            "reason": e.reason,
                            "step_count": len(self._steps) + 1,
                        },
                    )

                # Create tool message indicating verdict submission
                tool_messages.append(
                    ToolMessage(
                        content=f"Verdict submitted: {'PASSED' if e.passed else 'FAILED'} - {e.reason}",
                        tool_call_id=tool_id,
                    )
                )

                # Re-raise to be caught by workflow
                raise

            except asyncio.TimeoutError:
                timeout_msg = (
                    "Tool execution timed out"
                    if self.max_step_seconds is None
                    else f"Tool execution timed out after {self.max_step_seconds:.1f}s"
                )
                step["result"] = timeout_msg
                step["success"] = False
                step["semantic_action_signature"] = self._semantic_action_signature(tool_name, tool_args)

                logger.error(timeout_msg)

                self._consecutive_infra_failures += 1
                if self._consecutive_infra_failures >= _INFRA_CONSECUTIVE_THRESHOLD:
                    raise InfraError(
                        f"Persistent tool timeouts ({self._consecutive_infra_failures} consecutive)"
                    )

                if self.event_emitter:
                    await self.event_emitter.emit(
                        "tool_failed",
                        {
                            "tool_name": tool_name,
                            "error": timeout_msg,
                            "step_count": len(self._steps) + 1,
                        },
                    )

                tool_messages.append(
                    ToolMessage(
                        content=f"Error executing tool: {timeout_msg}",
                        tool_call_id=tool_id,
                    )
                )

            except InfraError:
                # Let InfraError propagate through the tool node unmodified so
                # evaluate() / evaluate_agentic() can catch it and set status
                # to "infra_error", which triggers orchestrator retry.
                raise

            except Exception as e:
                step["result"] = str(e)
                step["success"] = False
                step["semantic_action_signature"] = self._semantic_action_signature(tool_name, tool_args)

                logger.error(f"Tool execution error: {e}")

                # Emit tool failed event
                if self.event_emitter:
                    await self.event_emitter.emit(
                        "tool_failed",
                        {
                            "tool_name": tool_name,
                            "error": str(e),
                            "step_count": len(self._steps) + 1,
                        },
                    )

                tool_messages.append(
                    ToolMessage(
                        content=f"Error executing tool: {str(e)}",
                        tool_call_id=tool_id,
                    )
                )

            self._steps.append(step)

        return {"messages": [*tool_messages, *deferred_followup_messages]}

    def _should_continue(self, state: AgentState) -> str:
        """Determine if workflow should continue or end.

        Args:
            state: Current agent state

        Returns:
            "continue" or "end"
        """
        # End if verdict submitted
        if state.get("verdict"):
            return "end"

        iteration = state.get("iteration", 0) or 0
        max_iterations = state.get("max_iterations", 0) or 0

        last_message = state["messages"][-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            if iteration >= max_iterations:
                # Allow exactly one grace turn at the iteration boundary so the
                # agent can submit a verdict without starting new exploration.
                # Use == (not >=) so a blocked or retried verdict tool cannot
                # trigger an unbounded loop.
                if iteration == max_iterations and any(
                    self._is_verdict_tool_call(call) for call in last_message.tool_calls
                ):
                    return "continue"
                logger.warning("Max iterations reached without a final verdict tool call")
                return "end"
            return "continue"

        # End if max iterations reached
        if iteration >= max_iterations:
            logger.warning("Max iterations reached without verdict")
            return "end"

        # No tool calls: keep going and let _tools_node inject a nudge message
        return "continue"

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
        """Run the evaluation workflow.

        Args:
            user_query: User's requirement to test
            app_url: URL of the running application
            executor: PlaywrightExecutor instance
            agent_config: Optional AgentConfig to inject the agent's persona
                into the system prompt.

        Returns:
            Evaluation result with verdict
        """
        logger.info(f"Starting evaluation: {user_query}")
        logger.info(f"App URL: {app_url}")

        self._steps = []
        self._dom_elements = []
        self._initial_diagnostics = {}
        self._current_task_id = task_id
        self._current_task_title = task_title
        []
        self._verdict_validation_failures = 0
        self._consecutive_infra_failures = 0
        eval_start_ts = time.monotonic()

        # Capture initial interactive DOM elements (page already loaded by caller)
        try:
            ctx = await executor.get_context()
            self._initial_diagnostics = ctx.get("diagnostics", {})
            self._dom_elements = self._extract_interactive_elements(
                ctx.get("accessibility_tree", {})
            )
            logger.debug(f"Captured {len(self._dom_elements)} interactive DOM elements")
            if not self._dom_elements:
                logger.warning(
                    "Initial page appears empty or unmounted: "
                    f"root_html_length={self._initial_diagnostics.get('root_html_length', 'n/a')}, "
                    f"body_text_preview={self._initial_diagnostics.get('body_text_preview', '')!r}, "
                    f"script_sources={self._initial_diagnostics.get('script_sources', [])}, "
                    f"http_errors={self._initial_diagnostics.get('http_errors', [])}, "
                    f"request_failures={self._initial_diagnostics.get('request_failures', [])}, "
                    f"page_errors={self._initial_diagnostics.get('page_errors', [])}, "
                    f"console_messages={self._initial_diagnostics.get('console_messages', [])}"
                )
        except Exception as e:
            logger.warning(f"Failed to capture initial DOM elements: {e}")

        # Format system prompt (include agent persona when available)
        system_prompt = format_system_prompt(
            user_query=user_query,
            app_url=app_url,
            max_iterations=self.max_iterations,
            standard_ids=standard_ids,
            agent_config=agent_config,
            allowed_tools=list(self.allowed_tools) if self.allowed_tools is not None else None,
        )

        # Initialize state
        initial_state: AgentState = {
            "messages": [
                SystemMessage(content=system_prompt),
                HumanMessage(content="Begin the evaluation."),
            ],
            "user_query": user_query,
            "app_url": app_url,
            "verdict": None,
            "iteration": 0,
            "max_iterations": self.max_iterations,
            "executor": executor,
            "steps": [],
        }

        # Run workflow
        try:
            final_state = await self.workflow.ainvoke(initial_state)

            # Extract verdict
            verdict = final_state.get("verdict")

            if not verdict:
                verdict = {
                    "verdict": "inconclusive",
                    "passed": None,
                    "reason": "Evaluation incomplete: Max iterations reached without verdict",
                }
            verdict["reason"] = self._sanitize_verdict_reason(str(verdict.get("reason", "")))
            if verdict.get("group_verdicts"):
                sanitized_group = []
                for item in verdict["group_verdicts"]:
                    copied = dict(item)
                    copied["reason"] = self._sanitize_verdict_reason(str(copied.get("reason", "")))
                    sanitized_group.append(copied)
                verdict["group_verdicts"] = sanitized_group

            logger.info(f"Evaluation complete: {verdict}")

            return {
                "verdict": verdict,
                "group_verdicts": verdict.get("group_verdicts", []),
                "iterations": final_state["iteration"],
                "steps": self._steps,
                "dom_elements": self._dom_elements,
                "initial_diagnostics": self._initial_diagnostics,
                "duration_ms": int((time.monotonic() - eval_start_ts) * 1000),
            }

        except VerdictSubmitted as e:
            logger.info(f"Evaluation complete via exception: {e.verdict} - {e.reason}")

            return {
                "verdict": {
                    "verdict": e.verdict,
                    "passed": e.passed,
                    "reason": self._sanitize_verdict_reason(e.reason),
                    "group_verdicts": [
                        {
                            **item,
                            "reason": self._sanitize_verdict_reason(str(item.get("reason", ""))),
                        }
                        for item in e.verdicts
                    ],
                },
                "group_verdicts": [
                    {
                        **item,
                        "reason": self._sanitize_verdict_reason(str(item.get("reason", ""))),
                    }
                    for item in e.verdicts
                ],
                "iterations": initial_state["iteration"],
                "steps": self._steps,
                "dom_elements": self._dom_elements,
                "initial_diagnostics": self._initial_diagnostics,
                "duration_ms": int((time.monotonic() - eval_start_ts) * 1000),
            }

        except Exception as e:
            logger.error(f"Evaluation error: {e}")
            raise

    async def evaluate_agentic(
        self,
        agent_config: "AgentConfig",
        app_url: str,
        executor: Any,
        task_context: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run evaluation for a single agentic persona.

        Args:
            agent_config: Loaded AgentConfig for this persona.
            app_url: URL of the running application.
            executor: Browser executor instance.

        Returns:
            Structured result dict with agent_id, status, runtime, dimensions.
        """
        agent_id = agent_config.id
        dimension_ids = [d.id for d in agent_config.rubric.dimensions]
        start_ts = time.monotonic()

        logger.info(f"[{agent_id}] Starting agentic evaluation at {app_url}")

        self._steps = []
        self._dom_elements = []
        self._initial_diagnostics = {}
        []
        self._standard_scoring_requirements = self._build_scoring_requirements(agent_config)
        self._verdict_validation_failures = 0
        self._consecutive_infra_failures = 0

        # Capture initial DOM state
        try:
            ctx = await executor.get_context()
            self._initial_diagnostics = ctx.get("diagnostics", {})
            self._dom_elements = self._extract_interactive_elements(
                ctx.get("accessibility_tree", {})
            )
        except Exception as exc:
            logger.warning(f"[{agent_id}] Failed to capture initial DOM: {exc}")

        # Build system prompt from agent config
        system_prompt = format_agentic_prompt(
            agent_config,
            app_url,
            max_iterations=agent_config.runtime.max_steps,
        )

        # Use agent-specific tool allowlist and step budget
        saved_allowed = self.allowed_tools
        saved_max = self.max_iterations
        saved_step_timeout = self.max_step_seconds
        self.allowed_tools = set(agent_config.allowed_tools)
        self.max_iterations = agent_config.runtime.max_steps if agent_config.runtime.max_steps is not None else self.max_iterations
        self.max_step_seconds = agent_config.runtime.max_step_seconds

        initial_state: AgentState = {
            "messages": [
                SystemMessage(content=system_prompt),
                HumanMessage(content=task_context or "Begin your evaluation."),
            ],
            "user_query": agent_config.description or agent_config.name,
            "app_url": app_url,
            "verdict": None,
            "iteration": 0,
            "max_iterations": agent_config.runtime.max_steps,
            "executor": executor,
            "steps": [],
        }

        status = "error"
        raw_group_verdicts: list = []
        raw_summary_verdict: Optional[str] = None
        raw_summary_reason: str = ""
        iterations_used = 0
        end_reason: str = ""

        try:
            # Primary evaluation run.
            final_state = await asyncio.wait_for(
                self.workflow.ainvoke(initial_state),
                timeout=agent_config.runtime.max_total_seconds,
            )
            iterations_used = final_state.get("iteration", 0)
            raw_verdict = final_state.get("verdict")

            # If the agent stopped early (iterations < budget) without
            # submitting a verdict — re-engage and let it continue with
            # its remaining step budget.
            if not raw_verdict and iterations_used < agent_config.runtime.max_steps:
                logger.warning(
                    "[%s] Agent ended early (step %d/%d) without verdict — forcing continuation.",
                    agent_id, iterations_used, agent_config.runtime.max_steps,
                )
                force_state = dict(final_state)
                force_state["messages"] = list(final_state["messages"])
                force_state["messages"].append(HumanMessage(
                    content=(
                        "You stopped without submitting a final verdict. "
                        "Continue your evaluation — you still have steps "
                        "remaining. When you are ready, call "
                        f"{'submit_group_verdict' if self._required_standard_ids else 'submit_verdict'}."
                    )
                ))
                force_state["verdict"] = None
                try:
                    final_state = await asyncio.wait_for(
                        self.workflow.ainvoke(force_state),
                        timeout=agent_config.runtime.max_total_seconds,
                    )
                    iterations_used = final_state.get("iteration", 0)
                    raw_verdict = final_state.get("verdict")
                except Exception as exc:
                    logger.warning(
                        "[%s] Forced-verdict continuation failed: %s",
                        agent_id, exc,
                    )
                    # Keep the original final_state / raw_verdict from
                    # the primary run; the fallback logic below applies.

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
            logger.warning(f"[{agent_id}] Timed out after {agent_config.runtime.max_total_seconds}s")
            status = "timeout"
            iterations_used = agent_config.runtime.max_steps
            end_reason = "Agent timed out before submitting a verdict."

        except VerdictSubmitted as exc:
            logger.info(f"[{agent_id}] Verdict submitted via exception")
            raw_group_verdicts = exc.verdicts or []
            raw_summary_verdict = str(exc.verdict or "").strip().lower()
            raw_summary_reason = str(exc.reason or "").strip()
            status = "completed"
            iterations_used = len(self._steps)
            end_reason = "Agent submitted a verdict."

        except InfraError as exc:
            logger.warning(f"[{agent_id}] Infrastructure error: {exc}")
            status = "infra_error"
            iterations_used = len(self._steps)
            end_reason = f"Persistent infrastructure failures: {str(exc)}"

        except Exception as exc:
            exc_msg = str(exc)
            if _is_infra_error_message(exc_msg):
                logger.warning(f"[{agent_id}] Infrastructure error reclassified from generic exception: {exc_msg}")
                status = "infra_error"
                end_reason = f"Persistent infrastructure failures: {exc_msg}"
            else:
                logger.error(f"[{agent_id}] Evaluation error: {exc_msg}")
                status = "error"
                end_reason = f"Internal evaluator error: {exc_msg}"

        finally:
            # Restore evaluator state
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

    async def shutdown(self) -> None:
        """Release LLM and workflow resources.

        Breaks reference cycles (evaluator → workflow → evaluator) so the
        GC can collect this instance promptly, and releases the ChatOpenAI
        / ChatAnthropic reference that holds an internal httpx client.
        Call when the evaluator is no longer needed.
        """
        self._steps.clear()
        self._dom_elements.clear()
        self._initial_diagnostics.clear()
        self.llm = None
        self.workflow = None
        self.event_emitter = None
