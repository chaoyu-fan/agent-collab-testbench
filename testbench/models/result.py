"""Evaluation result data model."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class EvaluationResult:
    """Outcome of evaluating one (task, strategy, run) execution."""

    task_id: str
    strategy_config_id: str
    run_id: str
    passed: bool
    patch_similarity: float  # 0.0 - 1.0
    tokens_consumed: int
    wall_time_seconds: float
    agent_count: int
    phases_active: list[str]
    metrics: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "strategy_config_id": self.strategy_config_id,
            "run_id": self.run_id,
            "passed": self.passed,
            "patch_similarity": self.patch_similarity,
            "tokens_consumed": self.tokens_consumed,
            "wall_time_seconds": self.wall_time_seconds,
            "agent_count": self.agent_count,
            "phases_active": self.phases_active,
            "metrics": self.metrics,
        }
