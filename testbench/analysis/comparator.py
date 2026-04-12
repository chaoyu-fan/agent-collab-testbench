"""Cross-strategy comparison and ablation analysis."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Sequence

from testbench.evaluation.metrics import StrategyMetrics, compute_strategy_metrics
from testbench.models.result import EvaluationResult


PHASE_NAMES = [
    "SPAWN_SUBAGENTS",
    "PARALLEL_EXECUTION",
    "SUBAGENT_RESULT_PARTIAL",
    "SUBAGENT_RESULT_COMPLETE",
]

PHASE_TO_BIT = {name: i for i, name in enumerate(PHASE_NAMES)}


@dataclass
class PhaseAblationResult:
    """Effect of toggling a single phase ON vs OFF."""

    phase: str
    pass_rate_on: float
    pass_rate_off: float
    delta: float
    similarity_on: float
    similarity_off: float
    tokens_on: float
    tokens_off: float
    num_configs_on: int
    num_configs_off: int


@dataclass
class ComparisonReport:
    """Full comparison across all strategies."""

    strategy_metrics: dict[str, StrategyMetrics]
    phase_ablations: list[PhaseAblationResult]
    best_strategy: str
    worst_strategy: str


class Comparator:
    """Analyzes results across strategies to identify collaboration effects."""

    def analyze(self, all_results: Sequence[EvaluationResult]) -> ComparisonReport:
        """Run the full comparison analysis."""
        # Group results by strategy
        by_strategy: dict[str, list[EvaluationResult]] = defaultdict(list)
        for r in all_results:
            by_strategy[r.strategy_config_id].append(r)

        # Compute per-strategy metrics
        strategy_metrics = {}
        for config_id, results in by_strategy.items():
            strategy_metrics[config_id] = compute_strategy_metrics(results)

        # Phase ablation analysis
        phase_ablations = []
        for phase in PHASE_NAMES:
            ablation = self._compute_phase_ablation(phase, strategy_metrics)
            if ablation is not None:
                phase_ablations.append(ablation)

        # Best/worst
        if strategy_metrics:
            best = max(strategy_metrics.values(), key=lambda m: m.pass_rate)
            worst = min(strategy_metrics.values(), key=lambda m: m.pass_rate)
        else:
            best = worst = None

        return ComparisonReport(
            strategy_metrics=strategy_metrics,
            phase_ablations=phase_ablations,
            best_strategy=best.config_id if best else "",
            worst_strategy=worst.config_id if worst else "",
        )

    def _compute_phase_ablation(
        self,
        phase: str,
        metrics: dict[str, StrategyMetrics],
    ) -> PhaseAblationResult | None:
        """Compute the marginal effect of toggling a phase ON vs OFF.

        Groups strategies where this phase's bit differs but all other bits
        are the same, then averages the effect.
        """
        bit_index = PHASE_TO_BIT[phase]

        on_metrics: list[StrategyMetrics] = []
        off_metrics: list[StrategyMetrics] = []

        for config_id, m in metrics.items():
            if len(config_id) != 4:
                continue
            if config_id[bit_index] == "1":
                on_metrics.append(m)
            else:
                off_metrics.append(m)

        if not on_metrics or not off_metrics:
            return None

        def avg(vals: list[float]) -> float:
            return sum(vals) / len(vals) if vals else 0.0

        pr_on = avg([m.pass_rate for m in on_metrics])
        pr_off = avg([m.pass_rate for m in off_metrics])
        sim_on = avg([m.mean_similarity for m in on_metrics])
        sim_off = avg([m.mean_similarity for m in off_metrics])
        tok_on = avg([m.mean_tokens for m in on_metrics])
        tok_off = avg([m.mean_tokens for m in off_metrics])

        return PhaseAblationResult(
            phase=phase,
            pass_rate_on=pr_on,
            pass_rate_off=pr_off,
            delta=pr_on - pr_off,
            similarity_on=sim_on,
            similarity_off=sim_off,
            tokens_on=tok_on,
            tokens_off=tok_off,
            num_configs_on=len(on_metrics),
            num_configs_off=len(off_metrics),
        )
