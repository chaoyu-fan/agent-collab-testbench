"""End-to-end test: full pipeline with mock LLM across all strategies."""

import pytest

from testbench.analysis.comparator import Comparator
from testbench.analysis.report import generate_markdown, generate_csv, generate_json
from testbench.config import RunConfig
from testbench.evaluation.metrics import compute_strategy_metrics
from testbench.evaluation.patch_match import PatchMatchEvaluator
from testbench.llm.mock import MockLLMClient
from testbench.models.result import EvaluationResult
from testbench.models.strategy import CollaborationStrategy
from testbench.models.task import Task
from testbench.orchestrator.orchestrator import Orchestrator


@pytest.fixture
def tasks() -> list[Task]:
    return [
        Task(
            task_id="t1",
            description="Fix the off-by-one error in the pagination function.",
            ground_truth="Change the slice to use min() for boundary checking.",
        ),
        Task(
            task_id="t2",
            description="Fix the JWT token expiration check in auth middleware.",
            ground_truth="Add expiration check and return 401 for expired tokens.",
        ),
    ]


class TestFullPipelineMock:
    @pytest.mark.asyncio
    async def test_all_strategies_on_all_tasks(self, tasks: list[Task]):
        """Run all 9 strategies on 2 tasks, 1 run each. Verify traces and analysis."""
        strategies = CollaborationStrategy.all_strategies()
        config = RunConfig(runs_per_config=1, mock_mode=True)
        evaluator = PatchMatchEvaluator(pass_threshold=0.2)

        all_results: list[EvaluationResult] = []

        for task in tasks:
            for strategy in strategies:
                llm = MockLLMClient()
                orch = Orchestrator(llm, strategy, config)
                run_result = await orch.run_task(task)

                # Verify trace structure
                assert run_result.trace.task_id == task.task_id
                assert run_result.trace.strategy == strategy
                assert run_result.trace.end_time is not None
                assert len(run_result.trace.events) >= 3

                # Evaluate
                eval_result = evaluator.evaluate(task, run_result)
                all_results.append(eval_result)

        # Should have 2 tasks * 9 strategies = 18 results
        assert len(all_results) == 18

        # Verify all config IDs are represented
        config_ids = {r.strategy_config_id for r in all_results}
        assert len(config_ids) == 9

        # Run comparison analysis
        comparator = Comparator()
        report = comparator.analyze(all_results)

        assert len(report.strategy_metrics) == 9
        assert len(report.phase_ablations) == 4
        assert report.best_strategy != ""
        assert report.worst_strategy != ""

        # Verify report generation doesn't crash
        md = generate_markdown(report)
        assert "Strategy Results" in md
        assert "Phase Ablation" in md
        assert "0000" in md

        csv_out = generate_csv(report)
        assert "config_id" in csv_out

        json_out = generate_json(report)
        assert "strategy_metrics" in json_out

    @pytest.mark.asyncio
    async def test_solo_vs_collab_agent_counts(self, tasks: list[Task]):
        """Verify that solo uses 1 agent and collab uses more."""
        config = RunConfig(runs_per_config=1, mock_mode=True)
        task = tasks[0]

        # Solo
        solo_orch = Orchestrator(MockLLMClient(), CollaborationStrategy(), config)
        solo_result = await solo_orch.run_task(task)
        assert solo_result.agent_count == 1

        # Full collab
        collab_orch = Orchestrator(
            MockLLMClient(),
            CollaborationStrategy(
                spawn_subagents=True,
                parallel_execution=True,
                share_partial_results=True,
                synthesis_review=True,
            ),
            config,
        )
        collab_result = await collab_orch.run_task(task)
        assert collab_result.agent_count >= 3  # decomposer + specialists + synthesizer

    @pytest.mark.asyncio
    async def test_metrics_computation(self, tasks: list[Task]):
        """Verify metrics aggregate correctly."""
        config = RunConfig(runs_per_config=3, mock_mode=True)
        evaluator = PatchMatchEvaluator(pass_threshold=0.2)
        strategy = CollaborationStrategy()

        results: list[EvaluationResult] = []
        for task in tasks:
            for _ in range(config.runs_per_config):
                llm = MockLLMClient()
                orch = Orchestrator(llm, strategy, config)
                run_result = await orch.run_task(task)
                results.append(evaluator.evaluate(task, run_result))

        metrics = compute_strategy_metrics(results)
        assert metrics.config_id == "0000"
        assert metrics.num_runs == 6  # 2 tasks * 3 runs
        assert 0.0 <= metrics.pass_rate <= 1.0
        assert metrics.mean_tokens > 0

    @pytest.mark.asyncio
    async def test_trace_serialization(self, tasks: list[Task]):
        """Verify trace can be serialized to dict without errors."""
        config = RunConfig(mock_mode=True)
        llm = MockLLMClient()
        strategy = CollaborationStrategy(
            spawn_subagents=True,
            parallel_execution=True,
            synthesis_review=True,
        )
        orch = Orchestrator(llm, strategy, config)
        result = await orch.run_task(tasks[0])

        trace_dict = result.trace.to_dict()
        assert trace_dict["task_id"] == "t1"
        assert trace_dict["strategy_config_id"] == "1101"
        assert isinstance(trace_dict["events"], list)
        assert len(trace_dict["events"]) > 0
        assert trace_dict["total_tokens_in"] > 0
