"""Configuration management for the frontend evaluator."""

import os
from typing import Optional
from dotenv import load_dotenv


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


class Config:
    """Configuration manager for the frontend evaluator.

    Loads configuration from environment variables with sensible defaults.
    """

    def __init__(self, env_file: Optional[str] = None):
        """Initialize configuration.

        Args:
            env_file: Path to .env file (defaults to .env in current directory)
        """
        if env_file:
            load_dotenv(env_file)
        else:
            load_dotenv()

    @property
    def model_provider(self) -> str:
        """Get LLM provider (anthropic, openai, google, custom)."""
        return os.getenv("MODEL_PROVIDER", "anthropic").lower()

    @property
    def agent_runtime(self) -> str:
        """Get evaluator runtime backend (react or claude_sdk)."""
        return os.getenv("AGENT_RUNTIME", "react").lower()

    @property
    def model_name(self) -> Optional[str]:
        """Get model name (optional, uses provider default if not specified)."""
        return os.getenv("MODEL_NAME")

    @property
    def anthropic_api_key(self) -> Optional[str]:
        """Get Anthropic API key (required only when MODEL_PROVIDER=anthropic)."""
        key = os.getenv("ANTHROPIC_API_KEY")
        if self.model_provider == "anthropic" and not key:
            raise ValueError("ANTHROPIC_API_KEY required when MODEL_PROVIDER=anthropic")
        return key

    @property
    def openai_api_key(self) -> Optional[str]:
        """Get OpenAI API key (required only when MODEL_PROVIDER=openai)."""
        key = os.getenv("OPENAI_API_KEY")
        if self.model_provider == "openai" and not key:
            raise ValueError("OPENAI_API_KEY required when MODEL_PROVIDER=openai")
        return key

    @property
    def google_api_key(self) -> Optional[str]:
        """Get Google API key (required only when MODEL_PROVIDER=google)."""
        key = os.getenv("GOOGLE_API_KEY")
        if self.model_provider == "google" and not key:
            raise ValueError("GOOGLE_API_KEY required when MODEL_PROVIDER=google")
        return key

    @property
    def custom_api_key(self) -> Optional[str]:
        """Get custom API key (required only when MODEL_PROVIDER=custom)."""
        key = os.getenv("CUSTOM_API_KEY")
        if self.model_provider == "custom" and not key:
            raise ValueError("CUSTOM_API_KEY required when MODEL_PROVIDER=custom")
        return key

    @property
    def custom_base_url(self) -> Optional[str]:
        """Get custom base URL (required only when MODEL_PROVIDER=custom)."""
        url = os.getenv("CUSTOM_BASE_URL")
        if self.model_provider == "custom" and not url:
            raise ValueError("CUSTOM_BASE_URL required when MODEL_PROVIDER=custom")
        return url

    def get_llm_api_key(self) -> str:
        """Get API key for the configured provider.

        Returns:
            API key for the current provider

        Raises:
            ValueError: If provider is unknown or API key is missing
        """
        provider = self.model_provider
        if provider == "anthropic":
            return self.anthropic_api_key
        elif provider == "openai":
            return self.openai_api_key
        elif provider == "google":
            return self.google_api_key
        elif provider == "custom":
            return self.custom_api_key
        else:
            raise ValueError(f"Unknown provider: {provider}")

    @property
    def vision_model_name(self) -> Optional[str]:
        """Get vision model name for multimodal tasks (screenshot analysis).

        Falls back to MODEL_NAME if not set. Set to a VL model (e.g. qwen3-vl-plus)
        when the primary MODEL_NAME is text-only.
        """
        return os.getenv("VISION_MODEL_NAME")

    @property
    def sandbox_provider(self) -> str:
        """Sandbox provider env var (unused in the local-only public release).

        The benchmark runs locally via Playwright and never reads this; kept
        only because the dormant api/batch/agentic paths reference it.
        """
        return os.getenv("SANDBOX_PROVIDER", "").lower()

    @property
    def log_level(self) -> str:
        """Get log level."""
        return os.getenv("LOG_LEVEL", "INFO")

    @property
    def max_agent_steps(self) -> int:
        """Get maximum agent steps."""
        return int(os.getenv("MAX_AGENT_STEPS", "10"))

    @property
    def max_verdict_validation_retries(self) -> int:
        """Get maximum retries after invalid submit_group_verdict payloads."""
        return max(0, int(os.getenv("MAX_VERDICT_VALIDATION_RETRIES", "6")))

    @property
    def sandbox_startup_timeout(self) -> int:
        """Get sandbox startup timeout in seconds."""
        return int(os.getenv("SANDBOX_STARTUP_TIMEOUT", "120"))

    @property
    def app_startup_timeout(self) -> int:
        """Get app startup timeout in seconds."""
        return int(os.getenv("APP_STARTUP_TIMEOUT", "60"))

    @property
    def cdp_connection_timeout(self) -> int:
        """Get CDP connection timeout in seconds."""
        return int(os.getenv("CDP_CONNECTION_TIMEOUT", "30"))

    @property
    def app_navigation_timeout(self) -> int:
        """Get page.goto navigation timeout in seconds."""
        return int(os.getenv("APP_NAVIGATION_TIMEOUT", "30"))

    @property
    def executor_backend(self) -> str:
        """Get executor backend (playwright or agent-browser)."""
        return os.getenv("EXECUTOR_BACKEND", "playwright").lower()

    @property
    def max_parallel(self) -> int:
        """Get global max parallel task budget used by batch/agentic modes."""
        return int(os.getenv("MAX_PARALLEL", "1"))

    @property
    def per_row_worker_count(self) -> int:
        """Number of worker instances (dev server + Chromium) per evaluation row.

        When > 1, subtasks within a row are distributed across multiple
        workers round-robin, reducing CDP connection contention and dev
        server pressure. Each worker runs its own Chromium instance with
        an independent CDP connection.

        For static / Vite SPA projects, only Chromium instances are scaled
        (dev server is shared). For SSR projects (Next.js etc.), each worker
        gets both a dedicated dev server and Chromium.

        Default: 1 (current behavior, single worker per row). Set to 2 or
        more to enable multi-worker distribution. Memory budget: ~300 MB
        per Chromium instance.
        """
        return max(1, int(os.getenv("PER_ROW_WORKER_COUNT", "1")))

    @property
    def row_parallelism(self) -> int:
        """Maximum concurrent row/task executions across the batch.

        Rows are the coarse outer unit in eval_open/open-mode. Separating this
        from subtask parallelism prevents the batch runner from immediately
        flooding all task slots just because the global subtask budget is high.

        Default: inherit max_parallel. Set to 0 to inherit max_parallel.
        """
        raw = os.getenv("ROW_PARALLELISM", "").strip()
        if not raw:
            return self.max_parallel
        value = int(raw)
        if value <= 0:
            return self.max_parallel
        return value

    @property
    def build_parallelism(self) -> int:
        """Maximum concurrent build-phase agent executions.

        Build agents run npm install, dev-server startup, and Chromium
        launch — all resource-heavy.  Restricting build concurrency
        independent of evaluate-phase subtask parallelism prevents
        resource exhaustion and event-loop blocking at scale.

        Default: min(4, max_parallel).  Set to 0 to inherit max_parallel.
        """
        raw = os.getenv("BUILD_PARALLELISM", "").strip()
        if not raw:
            return max(1, min(4, self.max_parallel))
        value = int(raw)
        if value <= 0:
            return self.max_parallel
        return value

    @property
    def evaluate_agent_parallelism(self) -> int:
        """Maximum concurrent evaluate-stage agent executions across all rows.

        Tree-mode runs one orchestrator per agent from open_core, so this
        limit gates the expensive CDP + LLM main loop globally.
        Default: inherit max_parallel. Set to 0 to disable (no limit).
        """
        raw = os.getenv("EVALUATE_AGENT_PARALLELISM", "").strip()
        if not raw:
            return self.max_parallel
        value = int(raw)
        if value <= 0:
            return 0  # 0 = disabled, open_core will pass None gate
        return value

    @property
    def npm_preinstall_parallelism(self) -> int:
        """Maximum concurrent npm preinstall warmups across all rows.

        Default: min(4, build_parallelism). Set to 0 to inherit
        build_parallelism directly.
        """
        raw = os.getenv("NPM_PREINSTALL_PARALLELISM", "").strip()
        if not raw:
            return max(1, min(4, self.build_parallelism))
        value = int(raw)
        if value <= 0:
            return self.build_parallelism
        return value

    @property
    def row_timeout(self) -> Optional[float]:
        """Per-row wall-clock timeout in seconds.

        Rows that exceed this limit are cancelled and marked as failed.
        Default: 7200 (2 hours). Set to 0 to disable timeout.
        """
        raw = os.getenv("ROW_TIMEOUT", "").strip()
        if not raw:
            return 7200.0
        value = float(raw)
        if value <= 0:
            return None
        return value

    @property
    def chunk_size(self) -> int:
        """Number of results per chunk file when writing sharded JSONL output.

        Results are written to {output_stem}.shards/NNNN.jsonl, rotating to a
        new file every chunk_size rows. 0 = disabled (write to a single file).
        Default: 0.
        """
        raw = os.getenv("CHUNK_SIZE", "").strip()
        if not raw:
            return 0
        value = int(raw)
        return max(0, value)

    @property
    def claude_sdk_cli_path(self) -> Optional[str]:
        """Get optional Claude CLI path override for Claude SDK runtime."""
        value = os.getenv("CLAUDE_SDK_CLI_PATH", "").strip()
        return value or None

    @property
    def claude_sdk_system_tools_enabled(self) -> bool:
        """Whether Claude SDK built-in system tools are enabled."""
        return _env_bool("CLAUDE_SDK_SYSTEM_TOOLS_ENABLED", default=False)

    @property
    def task_synthesis_mode(self) -> str:
        """Get task synthesis mode (flat or tree)."""
        return os.getenv("TASK_SYNTHESIS_MODE", "flat").strip().lower() or "flat"

    @property
    def aggregation_mode(self) -> str:
        """Get subtask-to-main-task aggregation mode.

        - ``strict`` (default): any failed subtask → main task score = 0.
        - ``cumulative``: weighted average / pass ratio of subtasks.
        """
        return os.getenv("AGGREGATION_MODE", "strict").strip().lower() or "strict"

    @property
    def heavy_exec_parallelism(self) -> int:
        """Maximum concurrent heavy-exec agents (agents tagged 'heavy_exec').

        These agents spawn expensive subprocesses (npm install, vitest, jest)
        that each use 200-500 MB of RAM.  Limiting their concurrency independent
        of evaluate_agent_parallelism prevents memory exhaustion when many rows
        reach the evaluate phase simultaneously.

        Default: min(3, evaluate_agent_parallelism). Set to 0 to inherit
        evaluate_agent_parallelism (no separate cap).
        """
        raw = os.getenv("HEAVY_EXEC_PARALLELISM", "").strip()
        if not raw:
            return 0  # default: disabled (rely on evaluate_agent_parallelism or rate limits)
        value = int(raw)
        if value <= 0:
            return 0  # 0 = disabled
        return value

    @property
    def free_task_count(self) -> int:
        """Get default free task count for query-specific main tasks."""
        raw_value = os.getenv("FREE_TASK_COUNT")
        if raw_value is None:
            raw_value = os.getenv("QUERY_SPECIFIC_MAIN_TASK_COUNT", "3")
        return max(0, int(raw_value))

    @property
    def fixed_task_db_path(self) -> Optional[str]:
        """Get SQLite path for persistent query fixed task storage."""
        value = os.getenv("FIXED_TASK_DB_PATH", "artifacts/cache/fixed_tasks.db").strip()
        return value or None

    @property
    def llm_api_retry_count(self) -> int:
        """Get maximum retry count for transient LLM API failures."""
        return max(0, int(os.getenv("LLM_API_RETRY_COUNT", "5")))

    @property
    def llm_api_retry_base_delay_seconds(self) -> float:
        """Get base exponential backoff delay for transient LLM API failures."""
        return max(0.0, float(os.getenv("LLM_API_RETRY_BASE_DELAY_SECONDS", "1.0")))

    @property
    def llm_api_retry_jitter_seconds(self) -> float:
        """Get max jitter added to LLM API retry delays."""
        return max(0.0, float(os.getenv("LLM_API_RETRY_JITTER_SECONDS", "0.35")))

    @property
    def eval_open_task_start_jitter_ms(self) -> int:
        """Get startup jitter for each eval_open task in milliseconds."""
        return max(0, int(os.getenv("EVAL_OPEN_TASK_START_JITTER_MS", "250")))

    @property
    def disable_thinking(self) -> bool:
        """Disable extended thinking/reasoning across all LLM providers.

        Maps to provider-specific mechanisms:
        - Anthropic: thinking={"type": "disabled"}
        - OpenAI: no-op (GPT-4o has no separate thinking mode)
        - Google: no-op (reactive handling in message_utils)
        - Custom: no-op (provider-dependent)
        """
        return _env_bool("LLM_DISABLE_THINKING", default=False)

    @property
    def temperature(self) -> float:
        """LLM temperature / creativity.

        Only sent to known models that support the parameter (see LLMFactory).
        """
        return float(os.getenv("TEMPERATURE", "0"))

    @property
    def query_specific_main_task_count(self) -> int:
        """Get desired count for query-specific main tasks in tree mode."""
        return self.free_task_count
