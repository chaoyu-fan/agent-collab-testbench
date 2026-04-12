"""Solo agent — handles the full task without decomposition."""

from __future__ import annotations

from testbench.agents.base import BaseAgent, AgentResult
from testbench.agents.prompts import SOLO_SYSTEM, SOLO_USER
from testbench.llm.protocol import LLMClient
from testbench.models.agent import AgentIdentity
from testbench.models.trace import ExecutionTrace


class SoloAgent(BaseAgent):
    """Single agent that handles the entire task."""

    def __init__(
        self,
        task_description: str,
        llm: LLMClient,
        trace: ExecutionTrace,
    ) -> None:
        identity = AgentIdentity(name="solo", role="solo", task_scope="full_task")
        super().__init__(identity, llm, trace)
        self.task_description = task_description

    async def run(self) -> AgentResult:
        self.trace.add_event(
            phase="INIT",
            agent_name=self.name,
            event_type="spawn",
            content="Solo agent started",
        )

        user_msg = SOLO_USER.format(description=self.task_description)
        response = await self._call_llm(
            messages=[{"role": "user", "content": user_msg}],
            system=SOLO_SYSTEM,
            phase="EXECUTION",
        )

        self.trace.agent_snapshots[self.name] = [response.content]

        return AgentResult(
            agent_name=self.name,
            output=response.content,
            tokens_in=response.tokens_in,
            tokens_out=response.tokens_out,
        )
