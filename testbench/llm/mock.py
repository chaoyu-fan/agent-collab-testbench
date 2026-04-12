"""Mock LLM client for testing the framework without API calls."""

from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass, field
from typing import Any

from testbench.llm.protocol import LLMResponse


@dataclass
class MockLLMClient:
    """Deterministic mock that returns canned responses based on prompt content.

    Behavior:
    - If the prompt mentions "decompose" or "break down", returns a JSON
      subtask decomposition.
    - If the prompt mentions "synthesize" or "merge", returns a merged summary.
    - Otherwise, returns a generic specialist response seeded by prompt hash.
    """

    default_subtask_count: int = 3
    failure_rate: float = 0.0  # Fraction of calls that return error responses
    _call_count: int = field(default=0, init=False)

    async def complete(
        self,
        messages: list[dict[str, str]],
        system: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> LLMResponse:
        self._call_count += 1
        prompt_text = " ".join(m.get("content", "") for m in messages)
        prompt_lower = prompt_text.lower()

        if "decompose" in prompt_lower or "break down" in prompt_lower:
            content = self._decomposition_response(prompt_text)
        elif "synthesize" in prompt_lower or "merge" in prompt_lower:
            content = self._synthesis_response(prompt_text)
        else:
            content = self._specialist_response(prompt_text)

        tokens_in = len(prompt_text.split())
        tokens_out = len(content.split())

        return LLMResponse(
            content=content,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            model="mock",
        )

    def _decomposition_response(self, prompt: str) -> str:
        subtasks = []
        for i in range(self.default_subtask_count):
            subtasks.append({
                "scope": f"subtask_{i + 1}",
                "description": f"Handle aspect {i + 1} of the problem",
                "priority": i + 1,
            })
        return json.dumps({"subtasks": subtasks})

    def _synthesis_response(self, prompt: str) -> str:
        h = hashlib.md5(prompt.encode()).hexdigest()[:8]
        return (
            f"[Synthesis {h}] After reviewing all subagent outputs, "
            f"the unified solution addresses all identified aspects. "
            f"Key findings have been merged and conflicts resolved."
        )

    def _specialist_response(self, prompt: str) -> str:
        h = hashlib.md5(prompt.encode()).hexdigest()[:8]
        return (
            f"[Agent {h}] Analysis complete. "
            f"Identified the relevant code section and proposed a fix. "
            f"The issue stems from incorrect handling of edge cases."
        )

    @property
    def call_count(self) -> int:
        return self._call_count
