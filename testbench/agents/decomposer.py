"""Decomposer agent — breaks a task into subtasks."""

from __future__ import annotations

import json
from dataclasses import dataclass

from testbench.agents.base import BaseAgent, AgentResult
from testbench.agents.prompts import DECOMPOSER_SYSTEM, DECOMPOSER_USER
from testbench.llm.protocol import LLMClient
from testbench.models.agent import AgentIdentity
from testbench.models.trace import ExecutionTrace


@dataclass
class Subtask:
    scope: str
    description: str
    priority: int


class DecomposerAgent(BaseAgent):
    """Decomposes a task into subtasks for specialist agents."""

    def __init__(
        self,
        task_description: str,
        llm: LLMClient,
        trace: ExecutionTrace,
        max_subtasks: int = 5,
    ) -> None:
        identity = AgentIdentity(name="decomposer", role="decomposer", task_scope="decomposition")
        super().__init__(identity, llm, trace)
        self.task_description = task_description
        self.max_subtasks = max_subtasks

    async def run(self) -> AgentResult:
        self.trace.add_event(
            phase="SPAWN_SUBAGENTS",
            agent_name=self.name,
            event_type="spawn",
            content="Decomposer agent started",
        )

        user_msg = DECOMPOSER_USER.format(description=self.task_description)
        response = await self._call_llm(
            messages=[{"role": "user", "content": user_msg}],
            system=DECOMPOSER_SYSTEM,
            phase="SPAWN_SUBAGENTS",
        )

        self.trace.agent_snapshots[self.name] = [response.content]

        return AgentResult(
            agent_name=self.name,
            output=response.content,
            tokens_in=response.tokens_in,
            tokens_out=response.tokens_out,
        )

    def parse_subtasks(self, raw_output: str) -> list[Subtask]:
        """Parse the decomposer's JSON output into Subtask objects."""
        data = json.loads(raw_output)
        subtasks = []
        for item in data.get("subtasks", [])[:self.max_subtasks]:
            subtasks.append(
                Subtask(
                    scope=item["scope"],
                    description=item["description"],
                    priority=item.get("priority", len(subtasks) + 1),
                )
            )
        subtasks.sort(key=lambda s: s.priority)
        return subtasks
