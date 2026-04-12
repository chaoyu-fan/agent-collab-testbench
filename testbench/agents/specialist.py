"""Specialist agent — handles a specific subtask."""

from __future__ import annotations

from testbench.agents.base import BaseAgent, AgentResult
from testbench.agents.prompts import (
    SPECIALIST_SYSTEM,
    SPECIALIST_USER,
    SPECIALIST_USER_WITH_PRIOR,
    SPECIALIST_USER_WITH_BLACKBOARD,
)
from testbench.llm.protocol import LLMClient
from testbench.models.agent import AgentIdentity
from testbench.models.trace import ExecutionTrace


class SpecialistAgent(BaseAgent):
    """Handles a specific subtask scope."""

    def __init__(
        self,
        task_description: str,
        scope: str,
        scope_description: str,
        llm: LLMClient,
        trace: ExecutionTrace,
        prior_outputs: list[str] | None = None,
        blackboard: object | None = None,
    ) -> None:
        identity = AgentIdentity(
            name=f"specialist_{scope}",
            role="specialist",
            task_scope=scope,
        )
        super().__init__(identity, llm, trace)
        self.task_description = task_description
        self.scope = scope
        self.scope_description = scope_description
        self.prior_outputs = prior_outputs
        self.blackboard = blackboard

    async def run(self) -> AgentResult:
        self.trace.add_event(
            phase="PARALLEL_EXECUTION",
            agent_name=self.name,
            event_type="spawn",
            content=f"Specialist agent started for scope: {self.scope}",
        )

        user_msg = self._build_prompt()
        response = await self._call_llm(
            messages=[{"role": "user", "content": user_msg}],
            system=SPECIALIST_SYSTEM,
            phase="PARALLEL_EXECUTION",
        )

        # Post to blackboard if available
        if self.blackboard is not None and hasattr(self.blackboard, "post"):
            await self.blackboard.post(self.name, response.content[:500])

        self.trace.agent_snapshots.setdefault(self.name, []).append(response.content)

        return AgentResult(
            agent_name=self.name,
            output=response.content,
            tokens_in=response.tokens_in,
            tokens_out=response.tokens_out,
        )

    def _build_prompt(self) -> str:
        if self.prior_outputs:
            prior_text = "\n\n---\n\n".join(
                f"Agent {i+1}:\n{out}" for i, out in enumerate(self.prior_outputs)
            )
            return SPECIALIST_USER_WITH_PRIOR.format(
                description=self.task_description,
                scope=self.scope,
                scope_description=self.scope_description,
                prior_outputs=prior_text,
            )

        if self.blackboard is not None and hasattr(self.blackboard, "snapshot_sync"):
            bb_content = self.blackboard.snapshot_sync(exclude_agent=self.name)
            if bb_content:
                bb_text = "\n\n".join(
                    f"[{name}]: {'; '.join(summaries)}"
                    for name, summaries in bb_content.items()
                )
                return SPECIALIST_USER_WITH_BLACKBOARD.format(
                    description=self.task_description,
                    scope=self.scope,
                    scope_description=self.scope_description,
                    blackboard_content=bb_text,
                )

        return SPECIALIST_USER.format(
            description=self.task_description,
            scope=self.scope,
            scope_description=self.scope_description,
        )
