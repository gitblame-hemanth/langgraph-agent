"""LLM provider abstractions and factory."""

from src.llm.base import BaseLLMProvider
from src.llm.factory import get_llm_provider
from src.llm.openai_provider import OpenAIProvider


def __getattr__(name: str):
    """Lazy imports for optional providers (avoids import errors when SDKs aren't installed)."""
    if name == "BedrockProvider":
        from src.llm.bedrock_provider import BedrockProvider

        return BedrockProvider
    if name == "AzureOpenAIProvider":
        from src.llm.azure_provider import AzureOpenAIProvider

        return AzureOpenAIProvider
    if name == "VertexAIProvider":
        from src.llm.vertex_provider import VertexAIProvider

        return VertexAIProvider
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "AzureOpenAIProvider",
    "BaseLLMProvider",
    "BedrockProvider",
    "OpenAIProvider",
    "VertexAIProvider",
    "get_llm_provider",
]
