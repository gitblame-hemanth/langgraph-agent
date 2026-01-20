"""Abstract base class for LLM providers."""

import json
import re
from abc import ABC, abstractmethod
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


class BaseLLMProvider(ABC):
    """Abstract base for all LLM providers."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_message: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Generate a text completion from the LLM.

        Args:
            prompt: The user prompt.
            system_message: Optional system message.
            temperature: Override default temperature.
            max_tokens: Override default max tokens.

        Returns:
            The generated text.
        """
        ...

    def generate_json(
        self,
        prompt: str,
        system_message: str | None = None,
    ) -> dict[str, Any]:
        """Generate a response and parse it as JSON.

        Strips markdown code fences (```json ... ```) before parsing.
        This is the canonical method all nodes should use instead of
        raw generate() + json.loads().

        Args:
            prompt: The user prompt (should request JSON output).
            system_message: Optional system message.

        Returns:
            Parsed JSON as a dict.

        Raises:
            json.JSONDecodeError: If response is not valid JSON after stripping.
        """
        raw = self.generate(prompt, system_message=system_message)

        # Strip markdown code fences: ```json ... ``` or ``` ... ```
        cleaned = re.sub(
            r"```(?:json)?\s*\n?(.*?)\n?\s*```",
            r"\1",
            raw,
            flags=re.DOTALL,
        )
        cleaned = cleaned.strip()

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            logger.error(
                "json_parse_failed",
                raw_response=raw[:500],
                cleaned=cleaned[:500],
            )
            raise

    @abstractmethod
    def get_model_info(self) -> dict[str, Any]:
        """Return metadata about the configured model.

        Returns:
            Dict with at minimum 'provider' and 'model' keys.
        """
        ...
