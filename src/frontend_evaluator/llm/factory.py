"""LLM factory for creating provider-specific chat models."""
import os
from typing import Optional

from langchain_core.language_models import BaseChatModel

from .exceptions import UnsupportedProviderError, InvalidModelError

import httpx
_shared_http_client: httpx.Client | None = None
_shared_async_client: httpx.AsyncClient | None = None

def _llm_limits() -> httpx.Limits:
    """Build httpx connection limits.

    NOTE: keepalive connection reuse is disabled here as a diagnostic measure.
    Symptom: identical 35KB LLM requests take ~3s from a fresh process but
    ~287s (and frequently ReadTimeout) from the long-lived eval process that
    reuses a shared httpx client. The huge discrepancy is consistent with a
    poisoned/keeped-alive connection being reused. Setting
    max_keepalive_connections=0 forces a fresh TCP+TLS connection per request,
    which is the fastest way to confirm/refute the connection-pool hypothesis.
    Re-enable keepalive once the root cause is confirmed.
    """
    max_conn = int(os.getenv("LLM_HTTP_MAX_CONNECTIONS", "30"))
    max_keepalive = int(os.getenv("LLM_HTTP_MAX_KEEPALIVE", "0"))
    keepalive_expiry = float(os.getenv("LLM_HTTP_KEEPALIVE_EXPIRY", "5"))
    return httpx.Limits(
        max_connections=max_conn,
        max_keepalive_connections=max_keepalive,
        keepalive_expiry=keepalive_expiry,
    )


def _get_http_client() -> httpx.Client:
    global _shared_http_client
    if _shared_http_client is None:
        _shared_http_client = httpx.Client(
            verify=False,
            limits=_llm_limits(),
            event_hooks={"response": [_sync_log_response]},
        )
    return _shared_http_client

def _get_async_http_client() -> httpx.AsyncClient:
    global _shared_async_client
    if _shared_async_client is None:
        _shared_async_client = httpx.AsyncClient(
            verify=False,
            limits=_llm_limits(),
            event_hooks={"response": [_async_log_response]},
        )
    return _shared_async_client


# ---------------------------------------------------------------------------
# Per-request diagnostic logging (httpx event hooks).
# Writes one line per completed request to LLM_CALL_LOG_PATH (same file as the
# evaluator instrumentation) so slow/poisoned-connection requests are visible.
# ---------------------------------------------------------------------------
import time as _time
import json as _json


def _log_response_common(response: httpx.Response) -> dict:
    try:
        req = response.request
        elapsed = None
        # httpx Response has .elapsed (timedelta) since 0.18
        try:
            elapsed = response.elapsed.total_seconds()
        except Exception:
            elapsed = None
        host = ""
        try:
            host = req.url.host or ""
        except Exception:
            host = ""
        return {
            "phase": "http_response",
            "ts": _time.time(),
            "ts_iso": _time.strftime("%Y-%m-%dT%H:%M:%S"),
            "method": req.method,
            "host": host,
            "path": str(req.url.path),
            "status_code": response.status_code,
            "elapsed_s": round(elapsed, 2) if elapsed is not None else None,
            "req_content_len": len(req.content) if req.content else 0,
            "resp_content_len": len(response.content) if response.content else 0,
        }
    except Exception as e:
        return {"phase": "http_response", "error": str(e)}


def _sync_log_response(response: httpx.Response) -> None:
    if os.getenv("LLM_CALL_LOG_PATH"):
        try:
            with open(os.environ["LLM_CALL_LOG_PATH"], "a") as _f:
                _f.write(_json.dumps(_log_response_common(response), ensure_ascii=False, default=str) + "\n")
        except Exception:
            pass


async def _async_log_response(response: httpx.Response) -> None:
    if os.getenv("LLM_CALL_LOG_PATH"):
        try:
            with open(os.environ["LLM_CALL_LOG_PATH"], "a") as _f:
                _f.write(_json.dumps(_log_response_common(response), ensure_ascii=False, default=str) + "\n")
        except Exception:
            pass

# Default models for each provider
PROVIDER_DEFAULTS = {
    "anthropic": "claude-sonnet-4-5-20250929",
    "openai": "gpt-4o",
    "google": "gemini-2.0-flash-exp",
    "custom": None,  # Custom provider requires explicit model specification
}

# Supported models per provider (for validation)
# Note: 'custom' provider does not validate models
SUPPORTED_MODELS = {
    "anthropic": [
        "claude-sonnet-4-5-20250929",
        "claude-opus-4-5-20251101",
        "claude-3-5-sonnet-20241022",
        "claude-3-opus-20240229",
        "claude-3-sonnet-20240229",
        "claude-3-haiku-20240307",
    ],
    "openai": [
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-4-turbo",
        "gpt-4-turbo-preview",
        "gpt-4",
        "gpt-3.5-turbo",
    ],
    "google": [
        "gemini-2.0-flash-exp",
        "gemini-1.5-pro",
        "gemini-1.5-flash",
        "gemini-pro",
    ],
    "custom": [],  # No validation for custom provider
}

# Known model name prefixes that accept the temperature parameter.
# Models not in this list (e.g. Claude 4.x) will not receive temperature,
# as their APIs reject it.
_TEMPERATURE_SUPPORTED_MODELS = frozenset({
    "gpt-",
    "gemini-",
    "claude-3-",    # Claude 3.x (3-opus, 3-sonnet, 3-haiku)
    "claude-3-5-",  # Claude 3.5 (3-5-sonnet, 3-5-haiku)
})

# Provider-specific "disable thinking" configurations, keyed by model prefix.
# When disable_thinking=True, the matching config dict is merged into the
# LLM kwargs for the custom provider path. Unknown models default to no-op.
#
# Note: These are passed via ChatOpenAI(extra_body=...) so they end up in
# the request body, not as SDK keyword arguments.
_THINKING_DISABLE_CONFIGS: dict[str, dict] = {
    "claude-": {"thinking": {"type": "disabled"}},
    "gemini-": {"thinking": {"type": "disabled"}},
}


def _supports_temperature(model: str) -> bool:
    """Check whether a model name accepts the temperature parameter."""
    return any(model.startswith(prefix) for prefix in _TEMPERATURE_SUPPORTED_MODELS)


def _get_thinking_disable_config(model: str) -> Optional[dict]:
    """Get kwargs needed to disable extended thinking for a given model.

    Args:
        model: The model name to look up (e.g. ``"claude-opus-4-7"``).

    Returns:
        A dict of kwargs to merge into LLM init params, or ``None`` if
        the model has no known thinking-disable mechanism.
    """
    for prefix, config in _THINKING_DISABLE_CONFIGS.items():
        if model.startswith(prefix):
            return config
    return None


class LLMFactory:
    """Factory for creating LLM instances from different providers."""

    @staticmethod
    def create_llm(
        provider: str,
        api_key: str,
        model: Optional[str] = None,
        temperature: float = 0,
        base_url: Optional[str] = None,
        disable_thinking: bool = False,
        **kwargs
    ) -> BaseChatModel:
        """
        Create an LLM instance for the specified provider.

        Args:
            provider: Provider name (anthropic, openai, google, custom)
            api_key: API key for the provider
            model: Model name (optional for built-in providers, required for custom)
            temperature: Temperature for generation (default: 0).
                         Only sent to known models that support the parameter.
            base_url: Custom base URL (for custom provider or to override default)
            disable_thinking: Disable extended thinking/reasoning (provider-specific)
            **kwargs: Additional provider-specific arguments

        Returns:
            BaseChatModel instance

        Raises:
            UnsupportedProviderError: If provider is not supported
            InvalidModelError: If model is not valid for the provider
            ValueError: If required parameters are missing
        """
        provider = provider.lower()

        # Validate provider
        if provider not in PROVIDER_DEFAULTS:
            supported = ", ".join(PROVIDER_DEFAULTS.keys())
            raise UnsupportedProviderError(
                f"Unsupported provider: {provider}. Supported providers: {supported}"
            )

        # Custom provider
        if provider == "custom":
            if not model:
                raise ValueError("MODEL_NAME is required when using custom provider")
            if not base_url:
                raise ValueError("CUSTOM_BASE_URL is required when using custom provider")

            from langchain_openai import ChatOpenAI
            import httpx

            _timeout = float(os.getenv("LLM_REQUEST_TIMEOUT_SECONDS", "120"))
            # 不再传 custom http_client：实测 shared httpx client 长期复用会让请求被
            # 服务端 hang（同请求新建 client ~2s，shared 内 hang 到 timeout）；每次
            # 新建 client 又会连接泄漏。改用 SDK 默认 client（连接管理健康），SSL
            # 信任通过环境变量 SSL_CERT_FILE 指向系统 CA bundle 解决（certifi 不含
            # 内网 Ant Financial CA，故必须显式指定系统 CA）。
            # 仅当显式要求旧 shared client 时才回退（预留环境开关）。
            llm_kwargs: dict = {
                "api_key": api_key,
                "model": model,
                "base_url": base_url,
                "request_timeout": _timeout,
                **kwargs
            }
            if os.getenv("LLM_USE_SHARED_HTTPX_CLIENT", "0") == "1":
                llm_kwargs["http_client"] = _get_http_client()
                llm_kwargs["http_async_client"] = _get_async_http_client()
            if _supports_temperature(model):
                llm_kwargs["temperature"] = temperature
            if disable_thinking:
                thinking_config = _get_thinking_disable_config(model)
                if thinking_config is not None:
                    llm_kwargs["extra_body"] = thinking_config
            llm = ChatOpenAI(**llm_kwargs)
            setattr(llm, "_frontend_evaluator_provider", provider)
            return llm

        # Use default model if not specified (for built-in providers)
        if model is None:
            model = PROVIDER_DEFAULTS[provider]

        # Validate model for built-in providers
        if model not in SUPPORTED_MODELS[provider]:
            supported = ", ".join(SUPPORTED_MODELS[provider])
            raise InvalidModelError(
                f"Invalid model '{model}' for provider '{provider}'. "
                f"Supported models: {supported}"
            )

        # Create provider-specific LLM
        if provider == "anthropic":
            from langchain_anthropic import ChatAnthropic
            llm_kwargs = {
                "api_key": api_key,
                "model": model,
                **kwargs
            }
            # Temperature: only send when model supports it AND thinking is disabled.
            # When thinking is enabled, the Anthropic API rejects temperature.
            if disable_thinking:
                llm_kwargs["thinking"] = {"type": "disabled"}
                if _supports_temperature(model):
                    llm_kwargs["temperature"] = temperature
            llm = ChatAnthropic(**llm_kwargs)
            setattr(llm, "_frontend_evaluator_provider", provider)
            return llm

        elif provider == "openai":
            from langchain_openai import ChatOpenAI
            _timeout = float(os.getenv("LLM_REQUEST_TIMEOUT_SECONDS", "120"))
            llm_kwargs = {
                "api_key": api_key,
                "model": model,
                "request_timeout": _timeout,
                **kwargs
            }
            if _supports_temperature(model):
                llm_kwargs["temperature"] = temperature
            # Allow overriding base_url for OpenAI
            if base_url:
                llm_kwargs["base_url"] = base_url
            llm = ChatOpenAI(**llm_kwargs)
            setattr(llm, "_frontend_evaluator_provider", provider)
            return llm

        elif provider == "google":
            from langchain_google_genai import ChatGoogleGenerativeAI
            llm_kwargs = {
                "google_api_key": api_key,
                "model": model,
                **kwargs
            }
            if _supports_temperature(model):
                llm_kwargs["temperature"] = temperature
            llm = ChatGoogleGenerativeAI(**llm_kwargs)
            setattr(llm, "_frontend_evaluator_provider", provider)
            return llm

        # Should never reach here due to validation above
        raise UnsupportedProviderError(f"Provider {provider} not implemented")