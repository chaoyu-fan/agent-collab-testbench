"""Evaluator ABC and registry."""

from __future__ import annotations

from abc import ABC, abstractmethod

from testbench.models.result import EvaluationResult
from testbench.models.task import Task
from testbench.orchestrator.orchestrator import TaskRunResult


class Evaluator(ABC):
    """Base class for evaluating task execution results."""

    @abstractmethod
    def evaluate(self, task: Task, run_result: TaskRunResult) -> EvaluationResult:
        """Evaluate the output against the ground truth."""
        ...
