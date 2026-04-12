"""Diff/patch-based correctness evaluator."""

from __future__ import annotations

import difflib

from testbench.evaluation.evaluator import Evaluator
from testbench.models.result import EvaluationResult
from testbench.models.task import Task
from testbench.orchestrator.orchestrator import TaskRunResult


class PatchMatchEvaluator(Evaluator):
    """Evaluates output by comparing against ground truth text.

    Uses SequenceMatcher for similarity scoring.
    A configurable threshold determines pass/fail.
    """

    def __init__(self, pass_threshold: float = 0.5) -> None:
        self.pass_threshold = pass_threshold

    def evaluate(self, task: Task, run_result: TaskRunResult) -> EvaluationResult:
        similarity = self._compute_similarity(
            run_result.final_output,
            task.ground_truth,
        )
        passed = similarity >= self.pass_threshold

        trace = run_result.trace
        return EvaluationResult(
            task_id=task.task_id,
            strategy_config_id=trace.strategy.config_id,
            run_id=trace.run_id,
            passed=passed,
            patch_similarity=similarity,
            tokens_consumed=trace.total_tokens_in + trace.total_tokens_out,
            wall_time_seconds=trace.wall_time_seconds,
            agent_count=run_result.agent_count,
            phases_active=trace.strategy.active_phases,
            metrics={
                "tokens_in": float(trace.total_tokens_in),
                "tokens_out": float(trace.total_tokens_out),
                "event_count": float(len(trace.events)),
            },
        )

    def _compute_similarity(self, output: str, ground_truth: str) -> float:
        """Compute text similarity using SequenceMatcher."""
        if not output and not ground_truth:
            return 1.0
        if not output or not ground_truth:
            return 0.0

        # Normalize whitespace for comparison
        out_normalized = " ".join(output.lower().split())
        gt_normalized = " ".join(ground_truth.lower().split())

        return difflib.SequenceMatcher(None, out_normalized, gt_normalized).ratio()
