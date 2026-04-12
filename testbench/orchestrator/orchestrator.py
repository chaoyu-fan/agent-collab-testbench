"""Main orchestrator — runs a task under a given collaboration strategy."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from testbench.agents.base import AgentResult
from testbench.agents.decomposer import DecomposerAgent
from testbench.agents.solo_agent import SoloAgent
from testbench.agents.specialist import SpecialistAgent
from testbench.agents.synthesizer import SynthesizerAgent
from testbench.config import RunConfig
from testbench.llm.protocol import LLMClient
from testbench.models.strategy import CollaborationStrategy
from testbench.models.task import Task
from testbench.models.trace import ExecutionTrace
from testbench.orchestrator.lifecycle import LifecycleManager
from testbench.orchestrator.shared_state import SharedBlackboard

logger = logging.getLogger("testbench.orchestrator")


@dataclass
class TaskRunResult:
    """Complete result of running a task under a strategy."""

    trace: ExecutionTrace
    final_output: str
    agent_count: int


class Orchestrator:
    """Coordinates task execution under a collaboration strategy.

    The orchestrator implements the 4-phase decision flow:
    1. SPAWN_SUBAGENTS: decompose or solo?
    2. PARALLEL_EXECUTION: concurrent or sequential?
    3. SUBAGENT_RESULT_PARTIAL: share via blackboard or isolate?
    4. SUBAGENT_RESULT_COMPLETE: synthesize or concatenate?
    """

    def __init__(
        self,
        llm: LLMClient,
        strategy: CollaborationStrategy,
        config: RunConfig,
    ) -> None:
        self.llm = llm
        self.strategy = strategy
        self.config = config

    async def run_task(self, task: Task) -> TaskRunResult:
        """Execute a single task under this strategy."""
        trace = ExecutionTrace(task_id=task.task_id, strategy=self.strategy)

        trace.add_event(
            phase="INIT",
            agent_name="orchestrator",
            event_type="start",
            content=f"Starting task {task.task_id} with strategy {self.strategy.config_id}",
        )

        if self.strategy.is_solo:
            result = await self._run_solo(task, trace)
        else:
            result = await self._run_collaborative(task, trace)

        trace.add_event(
            phase="DONE",
            agent_name="orchestrator",
            event_type="complete",
            content=f"Task completed. Output length: {len(result.final_output)}",
        )

        return result

    async def _run_solo(self, task: Task, trace: ExecutionTrace) -> TaskRunResult:
        """Solo path — single agent, no collaboration."""
        agent = SoloAgent(
            task_description=task.description,
            llm=self.llm,
            trace=trace,
        )
        result = await agent.run()
        trace.finalize(result.output)

        return TaskRunResult(
            trace=trace,
            final_output=result.output,
            agent_count=1,
        )

    async def _run_collaborative(self, task: Task, trace: ExecutionTrace) -> TaskRunResult:
        """Collaborative path — decompose, execute, optionally share and synthesize."""
        lifecycle = LifecycleManager(timeout_seconds=self.config.agent_timeout_seconds)

        # Phase 1: SPAWN_SUBAGENTS — decompose the task
        decomposer = DecomposerAgent(
            task_description=task.description,
            llm=self.llm,
            trace=trace,
            max_subtasks=self.config.max_subtasks,
        )
        decomp_result = await decomposer.run()
        subtasks = decomposer.parse_subtasks(decomp_result.output)

        if not subtasks:
            logger.warning("Decomposer produced no subtasks, falling back to solo")
            return await self._run_solo(task, trace)

        trace.add_event(
            phase="SPAWN_SUBAGENTS",
            agent_name="orchestrator",
            event_type="decomposition_complete",
            content=f"Decomposed into {len(subtasks)} subtasks: {[s.scope for s in subtasks]}",
        )

        # Phase 2 & 3: Execute specialists (parallel or sequential, with or without sharing)
        blackboard = SharedBlackboard() if self.strategy.share_partial_results else None

        specialists = [
            SpecialistAgent(
                task_description=task.description,
                scope=st.scope,
                scope_description=st.description,
                llm=self.llm,
                trace=trace,
                blackboard=blackboard if self.strategy.share_partial_results else None,
            )
            for st in subtasks
        ]

        if self.strategy.parallel_execution:
            # Parallel: spawn all and gather
            for agent in specialists:
                lifecycle.spawn(agent)
            agent_results = await lifecycle.await_all()
        else:
            # Sequential: each sees prior outputs
            agent_results: list[AgentResult] = []
            for i, agent in enumerate(specialists):
                if agent_results:
                    agent.prior_outputs = [r.output for r in agent_results]
                result = await agent.run()
                agent_results.append(result)

        agent_count = len(agent_results) + 1  # +1 for decomposer

        # Phase 4: SUBAGENT_RESULT_COMPLETE — synthesize or concatenate
        if self.strategy.synthesis_review and len(agent_results) > 0:
            agent_outputs = {r.agent_name: r.output for r in agent_results}
            synthesizer = SynthesizerAgent(
                task_description=task.description,
                agent_outputs=agent_outputs,
                llm=self.llm,
                trace=trace,
            )
            synth_result = await synthesizer.run()
            final_output = synth_result.output
            agent_count += 1  # +1 for synthesizer
        else:
            # Concatenate directly
            final_output = "\n\n---\n\n".join(r.output for r in agent_results)

        trace.finalize(final_output)

        return TaskRunResult(
            trace=trace,
            final_output=final_output,
            agent_count=agent_count,
        )
