"""Metric computation from evaluation results."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from testbench.models.result import EvaluationResult


@dataclass
class StrategyMetrics:
    """Aggregated metrics for a single strategy across multiple runs/tasks."""

    config_id: str
    num_runs: int
    pass_rate: float
    mean_similarity: float
    mean_tokens: float
    mean_wall_time: float
    mean_agent_count: float
    std_similarity: float = 0.0

    def to_dict(self) -> dict[str, float | str | int]:
        return {
            "config_id": self.config_id,
            "num_runs": self.num_runs,
            "pass_rate": round(self.pass_rate, 4),
            "mean_similarity": round(self.mean_similarity, 4),
            "mean_tokens": round(self.mean_tokens, 1),
            "mean_wall_time": round(self.mean_wall_time, 3),
            "mean_agent_count": round(self.mean_agent_count, 2),
        }


def compute_strategy_metrics(
    results: Sequence[EvaluationResult],
) -> StrategyMetrics:
    """Compute aggregated metrics for a set of results (same strategy)."""
    if not results:
        raise ValueError("Cannot compute metrics for empty results")

    config_id = results[0].strategy_config_id
    n = len(results)

    pass_count = sum(1 for r in results if r.passed)
    similarities = [r.patch_similarity for r in results]
    tokens = [r.tokens_consumed for r in results]
    times = [r.wall_time_seconds for r in results]
    agents = [r.agent_count for r in results]

    mean_sim = sum(similarities) / n
    std_sim = (sum((s - mean_sim) ** 2 for s in similarities) / n) ** 0.5 if n > 1 else 0.0

    return StrategyMetrics(
        config_id=config_id,
        num_runs=n,
        pass_rate=pass_count / n,
        mean_similarity=mean_sim,
        mean_tokens=sum(tokens) / n,
        mean_wall_time=sum(times) / n,
        mean_agent_count=sum(agents) / n,
        std_similarity=std_sim,
    )
