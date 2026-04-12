"""LLM client protocol — interface for real and mock backends."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass
class LLMResponse:
    """Response from an LLM call."""

    content: str
    tokens_in: int
    tokens_out: int
    model: str = ""


class LLMClient(Protocol):
    """Interface that both real and mock LLM clients implement."""

    async def complete(
        self,
        messages: list[dict[str, str]],
        system: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> LLMResponse: ...
