"""AWS Bedrock LLM provider (Claude Messages API)."""

import json
import os
from typing import Any

import boto3
import structlog
from botocore.exceptions import ClientError

from src.llm.base import BaseLLMProvider

logger = structlog.get_logger(__name__)


class BedrockProvider(BaseLLMProvider):
    """AWS Bedrock provider using the Claude Messages API."""

    def __init__(
        self,
        model: str = "anthropic.claude-3-sonnet-20240229-v1:0",
        region: str | None = None,
        temperature: float = 0.1,
        max_tokens: int = 4096,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._region = region or os.getenv("AWS_DEFAULT_REGION", "us-east-1")
        self._client = boto3.client(
            "bedrock-runtime",
            region_name=self._region,
        )
        logger.info("bedrock_provider_init", model=model, region=self._region)

    def generate(
        self,
        prompt: str,
        system_message: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        body: dict[str, Any] = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens or self.max_tokens,
            "temperature": temperature if temperature is not None else self.temperature,
            "messages": [
                {"role": "user", "content": prompt},
            ],
        }

        if system_message:
            body["system"] = system_message

        logger.debug(
            "bedrock_generate",
            model=self.model,
            prompt_len=len(prompt),
        )

        max_retries = 5
        for attempt in range(1, max_retries + 1):
            try:
                response = self._client.invoke_model(
                    modelId=self.model,
                    contentType="application/json",
                    accept="application/json",
                    body=json.dumps(body),
                )
                break
            except ClientError as e:
                error_code = e.response["Error"]["Code"]
                if error_code == "ThrottlingException" and attempt < max_retries:
                    import time

                    wait = min(2**attempt, 60)
                    logger.warning(
                        "bedrock_throttled",
                        attempt=attempt,
                        wait_seconds=wait,
                    )
                    time.sleep(wait)
                    continue
                raise

        result = json.loads(response["body"].read())
        content = result["content"][0]["text"]

        logger.debug(
            "bedrock_response",
            model=self.model,
            input_tokens=result.get("usage", {}).get("input_tokens"),
            output_tokens=result.get("usage", {}).get("output_tokens"),
            response_len=len(content),
        )
        return content

    def get_model_info(self) -> dict[str, Any]:
        return {
            "provider": "bedrock",
            "model": self.model,
            "region": self._region,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
