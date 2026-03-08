"""Google Cloud Vertex AI LLM provider."""

import os
from typing import Any

import structlog

from src.llm.base import BaseLLMProvider

logger = structlog.get_logger(__name__)


class VertexAIProvider(BaseLLMProvider):
    """Vertex AI provider with lazy import of google-cloud-aiplatform."""

    def __init__(
        self,
        model: str = "gemini-1.5-pro",
        project: str | None = None,
        location: str = "us-central1",
        temperature: float = 0.1,
        max_tokens: int = 4096,
    ) -> None:
        self.model = model
        self.project = project or os.getenv("GOOGLE_CLOUD_PROJECT")
        self.location = location
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._generative_model = None
        logger.info(
            "vertex_provider_init",
            model=model,
            project=self.project,
            location=location,
        )

    def _get_model(self):
        """Lazy-initialize the Vertex AI generative model."""
        if self._generative_model is None:
            import vertexai
            from vertexai.generative_models import GenerativeModel

            vertexai.init(project=self.project, location=self.location)
            self._generative_model = GenerativeModel(self.model)
            logger.info("vertex_model_initialized", model=self.model)
        return self._generative_model

    def generate(
        self,
        prompt: str,
        system_message: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        from vertexai.generative_models import GenerationConfig

        model = self._get_model()

        generation_config = GenerationConfig(
            temperature=temperature if temperature is not None else self.temperature,
            max_output_tokens=max_tokens or self.max_tokens,
        )

        full_prompt = prompt
        if system_message:
            full_prompt = f"{system_message}\n\n{prompt}"

        logger.debug(
            "vertex_generate",
            model=self.model,
            prompt_len=len(full_prompt),
        )

        max_retries = 5
        for attempt in range(1, max_retries + 1):
            try:
                response = model.generate_content(
                    full_prompt,
                    generation_config=generation_config,
                )
                break
            except Exception as e:
                error_str = str(e).lower()
                if ("429" in error_str or "resource exhausted" in error_str) and attempt < max_retries:
                    import time

                    wait = min(2**attempt, 60)
                    logger.warning(
                        "vertex_rate_limited",
                        attempt=attempt,
                        wait_seconds=wait,
                    )
                    time.sleep(wait)
                    continue
                raise

        content = response.text
        logger.debug(
            "vertex_response",
            model=self.model,
            response_len=len(content),
        )
        return content

    def get_model_info(self) -> dict[str, Any]:
        return {
            "provider": "vertex_ai",
            "model": self.model,
            "project": self.project,
            "location": self.location,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
