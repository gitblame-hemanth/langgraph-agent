"""Azure OpenAI LLM provider."""

import os
from typing import Any

import structlog
from openai import APIConnectionError, APITimeoutError, AzureOpenAI, RateLimitError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.llm.base import BaseLLMProvider

logger = structlog.get_logger(__name__)

RETRYABLE_EXCEPTIONS = (RateLimitError, APITimeoutError, APIConnectionError)


class AzureOpenAIProvider(BaseLLMProvider):
    """Azure OpenAI chat completions provider."""

    def __init__(
        self,
        deployment_name: str,
        endpoint: str,
        api_key: str | None = None,
        api_version: str = "2024-02-01",
        temperature: float = 0.1,
        max_tokens: int = 4096,
    ) -> None:
        self.deployment_name = deployment_name
        self.endpoint = endpoint
        self.api_version = api_version
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = AzureOpenAI(
            api_key=api_key or os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=api_version,
            azure_endpoint=endpoint,
        )
        logger.info(
            "azure_openai_provider_init",
            deployment=deployment_name,
            endpoint=endpoint,
        )

    @retry(
        retry=retry_if_exception_type(RETRYABLE_EXCEPTIONS),
        wait=wait_exponential(multiplier=1, min=1, max=60),
        stop=stop_after_attempt(5),
        before_sleep=lambda retry_state: structlog.get_logger().warning(
            "azure_retry",
            attempt=retry_state.attempt_number,
            exception=str(retry_state.outcome.exception()),
        ),
    )
    def generate(
        self,
        prompt: str,
        system_message: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        messages: list[dict[str, str]] = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})

        logger.debug(
            "azure_generate",
            deployment=self.deployment_name,
            prompt_len=len(prompt),
        )

        response = self._client.chat.completions.create(
            model=self.deployment_name,
            messages=messages,
            temperature=temperature if temperature is not None else self.temperature,
            max_tokens=max_tokens or self.max_tokens,
        )

        content = response.choices[0].message.content or ""
        logger.debug(
            "azure_response",
            deployment=self.deployment_name,
            usage=dict(response.usage) if response.usage else None,
            response_len=len(content),
        )
        return content

    def get_model_info(self) -> dict[str, Any]:
        return {
            "provider": "azure_openai",
            "deployment": self.deployment_name,
            "endpoint": self.endpoint,
            "api_version": self.api_version,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
