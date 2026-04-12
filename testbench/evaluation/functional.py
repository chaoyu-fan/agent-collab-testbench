"""Test-suite-based evaluator — runs a command to validate the output."""

from __future__ import annotations

import asyncio
import logging

from testbench.evaluation.evaluator import Evaluator
from testbench.evaluation.patch_match import PatchMatchEvaluator
from testbench.models.result import EvaluationResult
from testbench.models.task import Task
from testbench.orchestrator.orchestrator import TaskRunResult

logger = logging.getLogger("testbench.evaluation")


class FunctionalEvaluator(Evaluator):
    """Evaluates by running a test command in a subprocess.

    Falls back to PatchMatchEvaluator if no test_command is provided.
    """

    def __init__(self, timeout_seconds: float = 60.0, pass_threshold: float = 0.5) -> None:
        self.timeout_seconds = timeout_seconds
        self._patch_evaluator = PatchMatchEvaluator(pass_threshold=pass_threshold)

    def evaluate(self, task: Task, run_result: TaskRunResult) -> EvaluationResult:
        if not task.test_command:
            return self._patch_evaluator.evaluate(task, run_result)

        # Run the test command synchronously (asyncio.run would conflict with event loop)
        passed = self._run_test_sync(task.test_command)

        # Also compute similarity for the metrics
        patch_result = self._patch_evaluator.evaluate(task, run_result)
        patch_result.passed = passed
        patch_result.metrics["test_command_passed"] = 1.0 if passed else 0.0
        return patch_result

    def _run_test_sync(self, command: str) -> bool:
        """Run a test command and return whether it passed (exit code 0)."""
        import subprocess

        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
            )
            if result.returncode != 0:
                logger.info("Test command failed (exit %d): %s", result.returncode, result.stderr[:200])
            return result.returncode == 0
        except subprocess.TimeoutExpired:
            logger.warning("Test command timed out after %.1fs", self.timeout_seconds)
            return False
        except Exception as e:
            logger.error("Test command error: %s", e)
            return False
