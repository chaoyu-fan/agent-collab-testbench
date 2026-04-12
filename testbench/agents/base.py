"""Base agent abstract class."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from testbench.llm.protocol import LLMClient, LLMResponse
from testbench.models.agent import AgentIdentity
from testbench.models.trace import ExecutionTrace


@dataclass
class AgentResult:
    """Output from an agent's run."""

    agent_name: str
    output: str
    tokens_in: int = 0
    tokens_out: int = 0


class BaseAgent(ABC):
    """Abstract base for all agent types."""

    def __init__(
        self,
        identity: AgentIdentity,
        llm: LLMClient,
        trace: ExecutionTrace,
    ) -> None:
        self.identity = identity
        self.llm = llm
        self.trace = trace

    @property
    def name(self) -> str:
        return self.identity.name

    @abstractmethod
    async def run(self) -> AgentResult:
        """Execute the agent's task and return the result."""
        ...

    async def _call_llm(
        self,
        messages: list[dict[str, str]],
        system: str | None = None,
        phase: str = "EXECUTION",
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> LLMResponse:
        """Call the LLM and record the event in the trace."""
        response = await self.llm.complete(
            messages=messages,
            system=system,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        self.trace.add_event(
            phase=phase,
            agent_name=self.name,
            event_type="llm_call",
            content=response.content[:200],  # Truncate for trace readability
            tokens_in=response.tokens_in,
            tokens_out=response.tokens_out,
        )
        return response
