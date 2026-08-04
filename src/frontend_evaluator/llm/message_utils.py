"""Helpers for provider-specific multimodal and message normalization."""

from __future__ import annotations

import copy
import json
from typing import Any, List

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage


_GEMINI_FUNCTION_CALL_SIGNATURES_MAP_KEY = "__gemini_function_call_thought_signatures__"


def normalize_image_url_for_provider(url: Any, provider: str | None) -> Any:
    """Normalize image payloads for providers with non-standard data URL handling.

    For the ``"custom"`` provider (which uses ``ChatOpenAI``), the standard
    ``data:image/…;base64,…`` URI format is the correct representation and
    must be preserved — it is the format every OpenAI-compatible API expects.
    """
    return url


def normalize_multimodal_content_for_provider(content: Any, provider: str | None) -> Any:
    """Normalize LangChain multimodal content blocks for the target provider."""
    if not isinstance(content, list):
        return content

    changed = False
    normalized: List[Any] = []
    for item in content:
        if not isinstance(item, dict):
            normalized.append(item)
            continue

        if item.get("type") != "image_url":
            normalized.append(item)
            continue

        image_url = item.get("image_url") if isinstance(item.get("image_url"), dict) else None
        if image_url is None:
            normalized.append(item)
            continue

        original_url = image_url.get("url")
        normalized_url = normalize_image_url_for_provider(original_url, provider)
        if normalized_url == original_url:
            normalized.append(item)
            continue

        changed = True
        normalized.append({
            **item,
            "image_url": {
                **image_url,
                "url": normalized_url,
            },
        })

    return normalized if changed else content


def _normalize_google_content_blocks(content: Any) -> Any:
    """Downgrade unsigned Gemini thinking blocks to plain text for safe history replay."""
    if not isinstance(content, list):
        return content

    changed = False
    normalized: List[Any] = []
    for item in content:
        if not isinstance(item, dict):
            normalized.append(item)
            continue

        block_type = str(item.get("type") or "").strip().lower()
        if block_type not in {"thinking", "reasoning"}:
            normalized.append(item)
            continue

        if block_type == "thinking":
            text_value = str(item.get("thinking") or "")
            signature = item.get("signature")
        else:
            text_value = str(item.get("reasoning") or "")
            extras = item.get("extras") if isinstance(item.get("extras"), dict) else {}
            signature = extras.get("signature")

        if isinstance(signature, str) and signature.strip():
            normalized.append(item)
            continue

        changed = True
        if text_value:
            normalized.append({"type": "text", "text": text_value})

    return normalized if changed else content


def normalize_content_for_provider(content: Any, provider: str | None) -> Any:
    """Apply provider-specific content normalization while preserving message semantics."""
    normalized = normalize_multimodal_content_for_provider(content, provider)
    if str(provider or "").strip().lower() == "google":
        normalized = _normalize_google_content_blocks(normalized)
    return normalized


def _is_gemini_compatible_provider(provider: str | None, model_name: str | None) -> bool:
    normalized_provider = str(provider or "").strip().lower()
    normalized_model = str(model_name or "").strip().lower()
    if normalized_provider == "google":
        return True
    if normalized_provider == "custom" and "gemini" in normalized_model:
        return True
    return False


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content or "")

    parts: List[str] = []
    for item in content:
        if isinstance(item, str):
            if item:
                parts.append(item)
            continue
        if not isinstance(item, dict):
            value = str(item or "")
            if value:
                parts.append(value)
            continue
        block_type = str(item.get("type") or "").strip().lower()
        if block_type == "text":
            text = str(item.get("text") or "")
        elif block_type == "thinking":
            text = str(item.get("thinking") or "")
        elif block_type == "reasoning":
            text = str(item.get("reasoning") or "")
        else:
            text = ""
        if text:
            parts.append(text)
    return "\n".join(part for part in parts if part).strip()


def _rewrite_unsigned_tool_call_ai_message(message: AIMessage, provider: str | None) -> AIMessage:
    normalized_content = normalize_content_for_provider(message.content, provider)
    content_text = _content_to_text(normalized_content)
    lines: List[str] = []
    if content_text:
        lines.append(content_text)
    lines.append("Previous assistant tool calls:")
    for tool_call in message.tool_calls:
        tool_name = str(tool_call.get("name") or "<unknown-tool>")
        tool_args = tool_call.get("args")
        try:
            args_text = json.dumps(tool_args, ensure_ascii=True, sort_keys=True)
        except Exception:
            args_text = str(tool_args)
        lines.append(f"- {tool_name} args={args_text}")

    return message.model_copy(
        update={
            "content": "\n".join(lines),
            "tool_calls": [],
            "invalid_tool_calls": [],
            "additional_kwargs": {
                key: value
                for key, value in (message.additional_kwargs or {}).items()
                if key not in {_GEMINI_FUNCTION_CALL_SIGNATURES_MAP_KEY, "function_call"}
            },
        }
    )


def _rewrite_tool_message_as_human(message: ToolMessage) -> HumanMessage:
    content_text = _content_to_text(message.content)
    if not content_text:
        content_text = "<empty tool result>"
    return HumanMessage(
        content=(
            f"Tool result for previous call {message.tool_call_id or '<unknown-call-id>'}:\n"
            f"{content_text}"
        )
    )


def normalize_messages_for_provider(
    messages: List[Any],
    provider: str | None,
    model_name: str | None = None,
) -> List[Any]:
    """Clone LangChain messages with provider-normalized multimodal content when needed."""
    if _is_gemini_compatible_provider(provider, model_name):
        normalized_messages: List[Any] = []
        index = 0
        while index < len(messages):
            message = messages[index]
            if isinstance(message, AIMessage) and message.tool_calls:
                signature_map = message.additional_kwargs.get(_GEMINI_FUNCTION_CALL_SIGNATURES_MAP_KEY, {})
                tool_ids = [
                    str(tool_call.get("id") or "").strip()
                    for tool_call in message.tool_calls
                    if str(tool_call.get("id") or "").strip()
                ]
                missing_signatures = not isinstance(signature_map, dict) or any(
                    not str(signature_map.get(tool_id) or "").strip()
                    for tool_id in tool_ids
                )
                if missing_signatures:
                    normalized_messages.append(_rewrite_unsigned_tool_call_ai_message(message, provider))
                    index += 1
                    while index < len(messages) and isinstance(messages[index], ToolMessage):
                        normalized_messages.append(_rewrite_tool_message_as_human(messages[index]))
                        index += 1
                    continue

            content = getattr(message, "content", None)
            normalized_content = normalize_content_for_provider(content, provider)
            if normalized_content is content:
                normalized_messages.append(message)
            elif hasattr(message, "model_copy"):
                normalized_messages.append(message.model_copy(update={"content": normalized_content}))
            else:
                cloned = copy.copy(message)
                try:
                    cloned.content = normalized_content
                    normalized_messages.append(cloned)
                except Exception:
                    normalized_messages.append(message)
            index += 1
        return normalized_messages

    normalized_messages: List[Any] = []
    for message in messages:
        content = getattr(message, "content", None)
        normalized_content = normalize_content_for_provider(content, provider)
        if normalized_content is content:
            normalized_messages.append(message)
            continue

        if hasattr(message, "model_copy"):
            normalized_messages.append(message.model_copy(update={"content": normalized_content}))
            continue

        cloned = copy.copy(message)
        try:
            cloned.content = normalized_content
            normalized_messages.append(cloned)
        except Exception:
            normalized_messages.append(message)

    return normalized_messages


def get_provider_from_llm(llm: Any) -> str | None:
    """Return the provider tag injected by LLMFactory when available."""
    return getattr(llm, "_frontend_evaluator_provider", None)


def get_model_name_from_llm(llm: Any) -> str | None:
    """Return the model name configured on the LLM when available."""
    for attr in ("model_name", "model"):
        value = getattr(llm, attr, None)
        if isinstance(value, str) and value.strip():
            return value
    return None