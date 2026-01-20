"""LLM provider factory."""

from __future__ import annotations

from typing import Any

import structlog

from src.llm.base import BaseLLMProvider

logger = structlog.get_logger(__name__)


def _get_config_value(config: Any, key: str, default: Any = None) -> Any:
    """Extract a value supporting both attribute and dict access."""
    val = getattr(config, key, None)
    if val is not None:
        return val
    if isinstance(config, dict):
        return config.get(key, default)
    return default


def _get_nested(config: Any, *keys: str, default: Any = None) -> Any:
    """Traverse nested config with mixed attribute/dict access."""
    current = config
    for key in keys:
        if current is None:
            return default
        current = _get_config_value(current, key)
    return current if current is not None else default


def get_llm_provider(config: Any = None) -> BaseLLMProvider:
    """Create an LLM provider based on application config.

    Supports config objects with attribute access (e.g., AppConfig dataclass)
    and plain dicts. Provider is resolved from:
      - config.llm_provider
      - config["llm"]["provider"]
      - config.llm.provider

    Args:
        config: Configuration object or dict. Uses get_config() if not provided.

    Returns:
        A configured BaseLLMProvider instance.

    Raises:
        ValueError: If the provider is unknown or not configured.
    """
    if config is None:
        from src.config import get_config

        config = get_config()

    # Resolve provider name
    provider = _get_config_value(config, "llm_provider") or _get_nested(config, "llm", "provider")

    if not provider:
        raise ValueError("LLM provider not configured. Set config.llm_provider or config['llm']['provider'].")

    provider = provider.lower().strip()

    # For nested config, pull LLM sub-config; otherwise use top-level
    llm_config = _get_config_value(config, "llm") or config

    logger.info("llm_factory_creating", provider=provider)

    if provider == "openai":
        from src.llm.openai_provider import OpenAIProvider

        return OpenAIProvider(
            model=_get_config_value(llm_config, "model") or _get_config_value(config, "model_name") or "gpt-4o",
            api_key=_get_config_value(llm_config, "api_key"),
            temperature=_get_config_value(llm_config, "temperature") or 0.1,
            max_tokens=_get_config_value(llm_config, "max_tokens") or 4096,
        )

    if provider == "bedrock":
        from src.llm.bedrock_provider import BedrockProvider

        return BedrockProvider(
            model=_get_config_value(llm_config, "model")
            or _get_config_value(config, "model_name")
            or "anthropic.claude-3-sonnet-20240229-v1:0",
            region=_get_config_value(llm_config, "region"),
            temperature=_get_config_value(llm_config, "temperature") or 0.1,
            max_tokens=_get_config_value(llm_config, "max_tokens") or 4096,
        )

    if provider in ("azure", "azure_openai"):
        from src.llm.azure_provider import AzureOpenAIProvider

        deployment = _get_config_value(llm_config, "deployment_name")
        endpoint = _get_config_value(llm_config, "endpoint")
        if not deployment or not endpoint:
            raise ValueError("Azure OpenAI requires 'deployment_name' and 'endpoint' in config.")

        return AzureOpenAIProvider(
            deployment_name=deployment,
            endpoint=endpoint,
            api_key=_get_config_value(llm_config, "api_key"),
            api_version=_get_config_value(llm_config, "api_version") or "2024-02-01",
            temperature=_get_config_value(llm_config, "temperature") or 0.1,
            max_tokens=_get_config_value(llm_config, "max_tokens") or 4096,
        )

    if provider in ("vertex", "vertex_ai", "google"):
        from src.llm.vertex_provider import VertexAIProvider

        return VertexAIProvider(
            model=_get_config_value(llm_config, "model") or _get_config_value(config, "model_name") or "gemini-1.5-pro",
            project=_get_config_value(llm_config, "project"),
            location=_get_config_value(llm_config, "location") or "us-central1",
            temperature=_get_config_value(llm_config, "temperature") or 0.1,
            max_tokens=_get_config_value(llm_config, "max_tokens") or 4096,
        )

    raise ValueError(
        f"Unknown LLM provider: '{provider}'. "
        f"Supported: openai, bedrock, azure, azure_openai, vertex, vertex_ai, google."
    )
