"""Real Anthropic API client with rate limiting and retry."""

from __future__ import annotations

import asyncio
import logging
import os

import anthropic

from testbench.llm.protocol import LLMResponse

logger = logging.getLogger("testbench.llm")

DEFAULT_MODEL = "claude-sonnet-4-20250514"


class AnthropicLLMClient:
    """Wraps the Anthropic SDK for real LLM calls.

    Features:
    - Async API calls
    - Concurrency limiting via semaphore
    - Retry with exponential backoff on rate limits
    - Token counting from API response
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        concurrency_limit: int = 4,
        max_retries: int = 3,
        api_key: str | None = None,
    ) -> None:
        self.model = model
        self._semaphore = asyncio.Semaphore(concurrency_limit)
        self.max_retries = max_retries
        self._client = anthropic.AsyncAnthropic(
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"),
        )
        self._call_count = 0

    async def complete(
        self,
        messages: list[dict[str, str]],
        system: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> LLMResponse:
        """Make an API call with rate limiting and retry."""
        async with self._semaphore:
            return await self._call_with_retry(
                messages=messages,
                system=system,
                max_tokens=max_tokens,
                temperature=temperature,
            )

    async def _call_with_retry(
        self,
        messages: list[dict[str, str]],
        system: str | None,
        max_tokens: int,
        temperature: float,
    ) -> LLMResponse:
        last_error: Exception | None = None

        for attempt in range(self.max_retries):
            try:
                kwargs: dict = {
                    "model": self.model,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                }
                if system:
                    kwargs["system"] = system

                response = await self._client.messages.create(**kwargs)
                self._call_count += 1

                content = ""
                for block in response.content:
                    if block.type == "text":
                        content += block.text

                return LLMResponse(
                    content=content,
                    tokens_in=response.usage.input_tokens,
                    tokens_out=response.usage.output_tokens,
                    model=response.model,
                )

            except anthropic.RateLimitError as e:
                last_error = e
                wait = 2 ** attempt
                logger.warning(
                    "Rate limited (attempt %d/%d), waiting %ds",
                    attempt + 1, self.max_retries, wait,
                )
                await asyncio.sleep(wait)

            except anthropic.APIError as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    wait = 2 ** attempt
                    logger.warning(
                        "API error (attempt %d/%d): %s, retrying in %ds",
                        attempt + 1, self.max_retries, e, wait,
                    )
                    await asyncio.sleep(wait)
                else:
                    raise

        raise last_error  # type: ignore[misc]

    @property
    def call_count(self) -> int:
        return self._call_count
