"""Statistical tests for comparing strategies."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class PairedComparison:
    """Result of comparing two strategies."""

    strategy_a: str
    strategy_b: str
    metric: str
    mean_a: float
    mean_b: float
    delta: float
    effect_size: float  # Cohen's d
    n_a: int
    n_b: int


def cohens_d(values_a: list[float], values_b: list[float]) -> float:
    """Compute Cohen's d effect size between two groups."""
    if not values_a or not values_b:
        return 0.0

    mean_a = sum(values_a) / len(values_a)
    mean_b = sum(values_b) / len(values_b)

    var_a = sum((x - mean_a) ** 2 for x in values_a) / max(len(values_a) - 1, 1)
    var_b = sum((x - mean_b) ** 2 for x in values_b) / max(len(values_b) - 1, 1)

    pooled_std = math.sqrt((var_a + var_b) / 2)
    if pooled_std == 0:
        return 0.0

    return (mean_a - mean_b) / pooled_std


def paired_comparison(
    strategy_a: str,
    values_a: list[float],
    strategy_b: str,
    values_b: list[float],
    metric: str = "pass_rate",
) -> PairedComparison:
    """Compare two strategies on a given metric."""
    mean_a = sum(values_a) / len(values_a) if values_a else 0.0
    mean_b = sum(values_b) / len(values_b) if values_b else 0.0

    return PairedComparison(
        strategy_a=strategy_a,
        strategy_b=strategy_b,
        metric=metric,
        mean_a=mean_a,
        mean_b=mean_b,
        delta=mean_a - mean_b,
        effect_size=cohens_d(values_a, values_b),
        n_a=len(values_a),
        n_b=len(values_b),
    )
