"""LLM provider abstraction — returns a provider based on config.

This module is kept as a fallback for any code that still calls get_llm().
New code should get the provider from state["_llm_provider"] instead.
"""

from __future__ import annotations

from functools import lru_cache

from src.llm.base import BaseLLMProvider
from src.llm.factory import get_llm_provider


@lru_cache(maxsize=1)
def get_llm() -> BaseLLMProvider:
    """Return a configured LLM provider instance based on application config.

    Uses the new factory. The returned instance is cached for the process lifetime.
    """
    return get_llm_provider()
