"""Tests for the orchestrator — solo and collaborative paths."""

import pytest

from testbench.config import RunConfig
from testbench.llm.mock import MockLLMClient
from testbench.models.strategy import CollaborationStrategy
from testbench.models.task import Task
from testbench.orchestrator.orchestrator import Orchestrator


@pytest.fixture
def config() -> RunConfig:
    return RunConfig(mock_mode=True)


@pytest.fixture
def task() -> Task:
    return Task(
        task_id="test_001",
        description="Fix the off-by-one error in the pagination function.",
        ground_truth="Change the slice to use min().",
    )


class TestOrchestratorSolo:
    @pytest.mark.asyncio
    async def test_solo_produces_output(self, task: Task, config: RunConfig):
        llm = MockLLMClient()
        strategy = CollaborationStrategy()  # solo
        orch = Orchestrator(llm, strategy, config)

        result = await orch.run_task(task)

        assert result.final_output != ""
        assert result.agent_count == 1
        assert result.trace.task_id == "test_001"
        assert result.trace.strategy.config_id == "0000"
        assert result.trace.end_time is not None

    @pytest.mark.asyncio
    async def test_solo_trace_has_events(self, task: Task, config: RunConfig):
        llm = MockLLMClient()
        strategy = CollaborationStrategy()
        orch = Orchestrator(llm, strategy, config)

        result = await orch.run_task(task)
        events = result.trace.events

        # Should have: INIT start, INIT spawn, EXECUTION llm_call, DONE complete
        assert len(events) >= 3
        assert events[0].phase == "INIT"
        assert events[-1].phase == "DONE"

    @pytest.mark.asyncio
    async def test_solo_llm_called_once(self, task: Task, config: RunConfig):
        llm = MockLLMClient()
        strategy = CollaborationStrategy()
        orch = Orchestrator(llm, strategy, config)

        await orch.run_task(task)
        assert llm.call_count == 1


class TestOrchestratorCollaborative:
    @pytest.mark.asyncio
    async def test_spawn_only(self, task: Task, config: RunConfig):
        """Strategy 1000: spawn + sequential + no sharing + no synthesis."""
        llm = MockLLMClient()
        strategy = CollaborationStrategy(spawn_subagents=True)
        orch = Orchestrator(llm, strategy, config)

        result = await orch.run_task(task)

        assert result.final_output != ""
        assert result.agent_count >= 2  # decomposer + at least 1 specialist
        # No synthesis agent
        assert "synthesizer" not in result.trace.agent_snapshots

    @pytest.mark.asyncio
    async def test_full_collab(self, task: Task, config: RunConfig):
        """Strategy 1111: spawn + parallel + sharing + synthesis."""
        llm = MockLLMClient()
        strategy = CollaborationStrategy(
            spawn_subagents=True,
            parallel_execution=True,
            share_partial_results=True,
            synthesis_review=True,
        )
        orch = Orchestrator(llm, strategy, config)

        result = await orch.run_task(task)

        assert result.final_output != ""
        assert result.agent_count >= 3  # decomposer + specialists + synthesizer
        assert "synthesizer" in result.trace.agent_snapshots
        assert "[Synthesis" in result.final_output

    @pytest.mark.asyncio
    async def test_parallel_no_synthesis(self, task: Task, config: RunConfig):
        """Strategy 1100: spawn + parallel + no sharing + no synthesis."""
        llm = MockLLMClient()
        strategy = CollaborationStrategy(
            spawn_subagents=True,
            parallel_execution=True,
        )
        orch = Orchestrator(llm, strategy, config)

        result = await orch.run_task(task)

        # Output should be concatenated specialist results
        assert "---" in result.final_output
        assert "synthesizer" not in result.trace.agent_snapshots

    @pytest.mark.asyncio
    async def test_sequential_with_synthesis(self, task: Task, config: RunConfig):
        """Strategy 1001: spawn + sequential + no sharing + synthesis."""
        llm = MockLLMClient()
        strategy = CollaborationStrategy(
            spawn_subagents=True,
            synthesis_review=True,
        )
        orch = Orchestrator(llm, strategy, config)

        result = await orch.run_task(task)

        assert "synthesizer" in result.trace.agent_snapshots
        assert "[Synthesis" in result.final_output

    @pytest.mark.asyncio
    async def test_trace_records_all_phases(self, task: Task, config: RunConfig):
        """Full collab should record events across all phases."""
        llm = MockLLMClient()
        strategy = CollaborationStrategy(
            spawn_subagents=True,
            parallel_execution=True,
            share_partial_results=True,
            synthesis_review=True,
        )
        orch = Orchestrator(llm, strategy, config)

        result = await orch.run_task(task)
        phases = {e.phase for e in result.trace.events}

        assert "INIT" in phases
        assert "SPAWN_SUBAGENTS" in phases
        assert "PARALLEL_EXECUTION" in phases
        assert "SUBAGENT_RESULT_COMPLETE" in phases
        assert "DONE" in phases

    @pytest.mark.asyncio
    async def test_different_strategies_produce_different_traces(
        self, task: Task, config: RunConfig
    ):
        """Solo vs full collab should have materially different traces."""
        llm_solo = MockLLMClient()
        llm_collab = MockLLMClient()

        solo_orch = Orchestrator(llm_solo, CollaborationStrategy(), config)
        collab_orch = Orchestrator(
            llm_collab,
            CollaborationStrategy(
                spawn_subagents=True,
                parallel_execution=True,
                synthesis_review=True,
            ),
            config,
        )

        solo_result = await solo_orch.run_task(task)
        collab_result = await collab_orch.run_task(task)

        # Collab should have more events and agents
        assert len(collab_result.trace.events) > len(solo_result.trace.events)
        assert collab_result.agent_count > solo_result.agent_count
        assert llm_collab.call_count > llm_solo.call_count
