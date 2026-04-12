"""Synthesizer agent — merges outputs from specialist agents."""

from __future__ import annotations

from testbench.agents.base import BaseAgent, AgentResult
from testbench.agents.prompts import SYNTHESIZER_SYSTEM, SYNTHESIZER_USER
from testbench.llm.protocol import LLMClient
from testbench.models.agent import AgentIdentity
from testbench.models.trace import ExecutionTrace


class SynthesizerAgent(BaseAgent):
    """Reviews and merges all subagent outputs into a unified solution."""

    def __init__(
        self,
        task_description: str,
        agent_outputs: dict[str, str],  # agent_name -> output
        llm: LLMClient,
        trace: ExecutionTrace,
    ) -> None:
        identity = AgentIdentity(name="synthesizer", role="synthesizer", task_scope="synthesis")
        super().__init__(identity, llm, trace)
        self.task_description = task_description
        self.agent_outputs = agent_outputs

    async def run(self) -> AgentResult:
        self.trace.add_event(
            phase="SUBAGENT_RESULT_COMPLETE",
            agent_name=self.name,
            event_type="spawn",
            content="Synthesizer agent started",
        )

        outputs_text = "\n\n---\n\n".join(
            f"### {name}\n{output}" for name, output in self.agent_outputs.items()
        )

        user_msg = SYNTHESIZER_USER.format(
            description=self.task_description,
            agent_outputs=outputs_text,
        )

        response = await self._call_llm(
            messages=[{"role": "user", "content": user_msg}],
            system=SYNTHESIZER_SYSTEM,
            phase="SUBAGENT_RESULT_COMPLETE",
        )

        self.trace.agent_snapshots[self.name] = [response.content]

        return AgentResult(
            agent_name=self.name,
            output=response.content,
            tokens_in=response.tokens_in,
            tokens_out=response.tokens_out,
        )
