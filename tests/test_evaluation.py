"""Tests for the evaluation layer."""

import pytest

from testbench.config import RunConfig
from testbench.evaluation.patch_match import PatchMatchEvaluator
from testbench.llm.mock import MockLLMClient
from testbench.models.strategy import CollaborationStrategy
from testbench.models.task import Task
from testbench.orchestrator.orchestrator import Orchestrator


@pytest.fixture
def evaluator() -> PatchMatchEvaluator:
    return PatchMatchEvaluator(pass_threshold=0.3)


class TestPatchMatchEvaluator:
    def test_identical_texts(self, evaluator: PatchMatchEvaluator):
        from testbench.models.trace import ExecutionTrace
        from testbench.orchestrator.orchestrator import TaskRunResult

        trace = ExecutionTrace(task_id="t1", strategy=CollaborationStrategy())
        trace.finalize("exact match output")
        run_result = TaskRunResult(trace=trace, final_output="exact match output", agent_count=1)

        task = Task(task_id="t1", description="test", ground_truth="exact match output")
        result = evaluator.evaluate(task, run_result)

        assert result.passed is True
        assert result.patch_similarity == 1.0

    def test_completely_different(self, evaluator: PatchMatchEvaluator):
        from testbench.models.trace import ExecutionTrace
        from testbench.orchestrator.orchestrator import TaskRunResult

        trace = ExecutionTrace(task_id="t1", strategy=CollaborationStrategy())
        trace.finalize("aaaa bbbb cccc")
        run_result = TaskRunResult(trace=trace, final_output="aaaa bbbb cccc", agent_count=1)

        task = Task(task_id="t1", description="test", ground_truth="xxxx yyyy zzzz")
        result = evaluator.evaluate(task, run_result)

        assert result.patch_similarity < 0.3
        assert result.passed is False

    def test_partial_match(self, evaluator: PatchMatchEvaluator):
        from testbench.models.trace import ExecutionTrace
        from testbench.orchestrator.orchestrator import TaskRunResult

        trace = ExecutionTrace(task_id="t1", strategy=CollaborationStrategy())
        output = "fix the bug by changing the slice to use min function"
        trace.finalize(output)
        run_result = TaskRunResult(trace=trace, final_output=output, agent_count=1)

        task = Task(
            task_id="t1",
            description="test",
            ground_truth="change the slice to use min",
        )
        result = evaluator.evaluate(task, run_result)

        assert 0.0 < result.patch_similarity < 1.0

    @pytest.mark.asyncio
    async def test_evaluate_with_real_run(self, evaluator: PatchMatchEvaluator):
        """End-to-end: run a task through the orchestrator, then evaluate."""
        task = Task(
            task_id="e2e_001",
            description="Fix the pagination bug",
            ground_truth="Fix the pagination function",
        )
        llm = MockLLMClient()
        config = RunConfig(mock_mode=True)
        strategy = CollaborationStrategy()
        orch = Orchestrator(llm, strategy, config)

        run_result = await orch.run_task(task)
        result = evaluator.evaluate(task, run_result)

        assert result.task_id == "e2e_001"
        assert result.strategy_config_id == "0000"
        assert result.tokens_consumed > 0
        assert result.wall_time_seconds >= 0
        assert result.agent_count == 1
