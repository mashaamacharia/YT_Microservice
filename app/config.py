"""
config.py
─────────────────────────────────────────────────────────────
Central configuration for the LLM microservice.
All environment variables are loaded and validated here.
Every other module imports from here — nothing reads .env directly.

Usage:
    from app.config import settings
    print(settings.anthropic_api_key)
"""

import os
from functools import lru_cache
from dotenv import load_dotenv

load_dotenv()


class Settings:
    """
    Loads all config from environment variables.
    Provides helper methods used by provider_router.
    """

    # ─── Provider order ───────────────────────────────────────────
    # Comma-separated list defining which provider to try first.
    # Example: "anthropic,openai,gemini"
    # Only providers with a configured API key will actually be used.
    provider_order: list[str]

    # ─── API Keys ─────────────────────────────────────────────────
    anthropic_api_key: str
    openai_api_key: str
    gemini_api_key: str

    # ─── Models per task per provider ─────────────────────────────
    # RESEARCH task
    research_model_anthropic: str
    research_model_openai: str
    research_model_gemini: str

    # SCRIPT task
    script_model_anthropic: str
    script_model_openai: str
    script_model_gemini: str

    # ANALYTICS task
    analytics_model_anthropic: str
    analytics_model_openai: str
    analytics_model_gemini: str

    # KEYWORDS task
    keywords_model_anthropic: str
    keywords_model_openai: str
    keywords_model_gemini: str

    # ─── Token limits per task ────────────────────────────────────
    research_max_tokens: int
    script_max_tokens: int
    analytics_max_tokens: int
    keywords_max_tokens: int

    # ─── Timeouts ─────────────────────────────────────────────────
    llm_timeout_seconds: int

    # ─── Service ──────────────────────────────────────────────────
    port: int
    environment: str     # "development" | "production"
    log_level: str       # "debug" | "info" | "warning" | "error"

    def __init__(self):
        # Provider order
        raw_order = os.getenv("LLM_PROVIDER_ORDER", "anthropic,openai,gemini")
        self.provider_order = [p.strip() for p in raw_order.split(",") if p.strip()]

        # API Keys — default to empty string (missing = provider skipped)
        self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
        self.openai_api_key = os.getenv("OPENAI_API_KEY", "").strip()
        self.gemini_api_key = os.getenv("GEMINI_API_KEY", "").strip()

        # Models — RESEARCH
        self.research_model_anthropic = os.getenv(
            "RESEARCH_MODEL_ANTHROPIC", "claude-sonnet-4-6"
        ).strip()
        self.research_model_openai = os.getenv(
            "RESEARCH_MODEL_OPENAI", "gpt-4o-mini"
        ).strip()
        self.research_model_gemini = os.getenv(
            "RESEARCH_MODEL_GEMINI", "gemini-1.5-flash"
        ).strip()

        # Models — SCRIPT
        self.script_model_anthropic = os.getenv(
            "SCRIPT_MODEL_ANTHROPIC", "claude-sonnet-4-6"
        ).strip()
        self.script_model_openai = os.getenv(
            "SCRIPT_MODEL_OPENAI", "gpt-4o"
        ).strip()
        self.script_model_gemini = os.getenv(
            "SCRIPT_MODEL_GEMINI", "gemini-1.5-pro"
        ).strip()

        # Models — ANALYTICS
        self.analytics_model_anthropic = os.getenv(
            "ANALYTICS_MODEL_ANTHROPIC", "claude-haiku-4-5-20251001"
        ).strip()
        self.analytics_model_openai = os.getenv(
            "ANALYTICS_MODEL_OPENAI", "gpt-4o-mini"
        ).strip()
        self.analytics_model_gemini = os.getenv(
            "ANALYTICS_MODEL_GEMINI", "gemini-1.5-flash"
        ).strip()

        # Models — KEYWORDS
        self.keywords_model_anthropic = os.getenv(
            "KEYWORDS_MODEL_ANTHROPIC", "claude-haiku-4-5-20251001"
        ).strip()
        self.keywords_model_openai = os.getenv(
            "KEYWORDS_MODEL_OPENAI", "gpt-4o-mini"
        ).strip()
        self.keywords_model_gemini = os.getenv(
            "KEYWORDS_MODEL_GEMINI", "gemini-1.5-flash"
        ).strip()

        # Token limits
        self.research_max_tokens = int(os.getenv("RESEARCH_MAX_TOKENS", "1000"))
        self.script_max_tokens = int(os.getenv("SCRIPT_MAX_TOKENS", "4000"))
        self.analytics_max_tokens = int(os.getenv("ANALYTICS_MAX_TOKENS", "500"))
        self.keywords_max_tokens = int(os.getenv("KEYWORDS_MAX_TOKENS", "200"))

        # Timeouts
        self.llm_timeout_seconds = int(os.getenv("LLM_TIMEOUT_SECONDS", "30"))

        # Service
        self.port = int(os.getenv("PORT", "8001"))
        self.environment = os.getenv("ENVIRONMENT", "development").strip()
        self.log_level = os.getenv("LOG_LEVEL", "info").strip()

    def get_model_for_task(self, task: str, provider: str) -> str:
        """
        Returns the model string for a given task + provider combination.

        Example:
            settings.get_model_for_task("script", "anthropic")
            → "claude-sonnet-4-6"
        """
        key = f"{task.lower()}_model_{provider.lower()}"
        return getattr(self, key, "").strip()

    def get_api_key_for_provider(self, provider: str) -> str:
        """
        Returns the API key for a given provider.
        Returns empty string if not configured.

        Example:
            settings.get_api_key_for_provider("anthropic")
            → "sk-ant-xxx" or ""
        """
        key = f"{provider.lower()}_api_key"
        return getattr(self, key, "").strip()

    def get_max_tokens_for_task(self, task: str) -> int:
        """
        Returns the max token limit for a given task.

        Example:
            settings.get_max_tokens_for_task("script")
            → 4000
        """
        key = f"{task.lower()}_max_tokens"
        return getattr(self, key, 1000)

    def get_configured_providers(self) -> dict:
        """
        Returns a status dict of all providers.
        Used by the /health endpoint.

        Returns:
            {
                "anthropic": "configured",
                "openai": "not configured",
                "gemini": "not configured"
            }
        """
        all_providers = ["anthropic", "openai", "gemini"]
        return {
            provider: (
                "configured"
                if self.get_api_key_for_provider(provider)
                else "not configured"
            )
            for provider in all_providers
        }

    def validate(self) -> list[str]:
        """
        Validates the configuration and returns a list of warnings.
        Does NOT raise errors — the service starts regardless.
        Warnings are logged at startup so you know what's missing.

        Returns:
            List of warning strings. Empty list = all good.
        """
        warnings = []

        # Check at least one provider is configured
        configured = [
            p for p in self.provider_order
            if self.get_api_key_for_provider(p)
        ]
        if not configured:
            warnings.append(
                "⚠️  No LLM provider API keys found. "
                "Add at least one key to .env. "
                "All LLM calls will fail until a key is configured."
            )

        # Check provider_order only contains known providers
        known = {"anthropic", "openai", "gemini"}
        unknown = [p for p in self.provider_order if p not in known]
        if unknown:
            warnings.append(
                f"⚠️  Unknown providers in LLM_PROVIDER_ORDER: {unknown}. "
                f"These will be ignored."
            )

        # Check models exist for configured providers
        tasks = ["research", "script", "analytics", "keywords"]
        for provider in configured:
            for task in tasks:
                model = self.get_model_for_task(task, provider)
                if not model:
                    warnings.append(
                        f"⚠️  No model defined for task='{task}' "
                        f"provider='{provider}'. "
                        f"Add {task.upper()}_MODEL_{provider.upper()} to .env"
                    )

        return warnings


@lru_cache()
def get_settings() -> Settings:
    """
    Returns a cached Settings instance.
    Use this everywhere instead of instantiating Settings() directly.

    Usage:
        from app.config import get_settings
        settings = get_settings()
    """
    return Settings()


# Module-level singleton for convenience
# Both import styles work:
#   from app.config import settings
#   from app.config import get_settings; settings = get_settings()
settings = get_settings()