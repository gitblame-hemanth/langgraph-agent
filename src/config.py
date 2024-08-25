"""Application configuration with environment variable loading."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Literal


@dataclass(frozen=True)
class AppConfig:
    """Central application configuration."""

    # LLM settings
    llm_provider: Literal["openai", "bedrock", "azure", "azure_openai", "vertex", "vertex_ai", "google"] = "openai"
    model_name: str = "gpt-4o"
    temperature: float = 0.0
    max_tokens: int = 4096

    # Infrastructure
    redis_url: str = "redis://localhost:6379/0"

    # API server
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # LangSmith observability
    langsmith_enabled: bool = False
    langsmith_api_key: str = ""

    @classmethod
    def from_env(cls) -> AppConfig:
        """Build config from environment variables with sensible defaults."""
        return cls(
            llm_provider=os.getenv("LLM_PROVIDER", "openai"),  # type: ignore[arg-type]
            model_name=os.getenv("MODEL_NAME", "gpt-4o"),
            temperature=float(os.getenv("TEMPERATURE", "0.0")),
            max_tokens=int(os.getenv("MAX_TOKENS", "4096")),
            redis_url=os.getenv("REDIS_URL", "redis://localhost:6379/0"),
            api_host=os.getenv("API_HOST", "0.0.0.0"),
            api_port=int(os.getenv("API_PORT", "8000")),
            langsmith_enabled=os.getenv("LANGSMITH_ENABLED", "false").lower() == "true",
            langsmith_api_key=os.getenv("LANGSMITH_API_KEY", ""),
        )


@lru_cache(maxsize=1)
def get_config() -> AppConfig:
    """Return a cached singleton config instance."""
    return AppConfig.from_env()
