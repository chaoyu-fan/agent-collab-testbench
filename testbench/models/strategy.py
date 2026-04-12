"""Collaboration strategy — the 4-bit configuration under test."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product


@dataclass(frozen=True)
class CollaborationStrategy:
    """Controls which collaboration phases are active for a run.

    When spawn_subagents is False, the other three flags are forced to False
    (a solo agent has no one to collaborate with).
    """

    spawn_subagents: bool = False
    parallel_execution: bool = False
    share_partial_results: bool = False
    synthesis_review: bool = False

    def __post_init__(self) -> None:
        if not self.spawn_subagents:
            object.__setattr__(self, "parallel_execution", False)
            object.__setattr__(self, "share_partial_results", False)
            object.__setattr__(self, "synthesis_review", False)

    @property
    def config_id(self) -> str:
        """Compact identifier, e.g. '1010'."""
        bits = [
            self.spawn_subagents,
            self.parallel_execution,
            self.share_partial_results,
            self.synthesis_review,
        ]
        return "".join("1" if b else "0" for b in bits)

    @property
    def is_solo(self) -> bool:
        return not self.spawn_subagents

    @property
    def active_phases(self) -> list[str]:
        phases = []
        if self.spawn_subagents:
            phases.append("SPAWN_SUBAGENTS")
        if self.parallel_execution:
            phases.append("PARALLEL_EXECUTION")
        if self.share_partial_results:
            phases.append("SUBAGENT_RESULT_PARTIAL")
        if self.synthesis_review:
            phases.append("SUBAGENT_RESULT_COMPLETE")
        return phases

    @classmethod
    def from_config_id(cls, config_id: str) -> CollaborationStrategy:
        if len(config_id) != 4 or not all(c in "01" for c in config_id):
            raise ValueError(f"config_id must be 4 binary digits, got '{config_id}'")
        bits = [c == "1" for c in config_id]
        return cls(
            spawn_subagents=bits[0],
            parallel_execution=bits[1],
            share_partial_results=bits[2],
            synthesis_review=bits[3],
        )

    @classmethod
    def all_strategies(cls) -> list[CollaborationStrategy]:
        """Generate all 9 effective configurations."""
        seen: set[str] = set()
        strategies: list[CollaborationStrategy] = []
        for bits in product([False, True], repeat=4):
            s = cls(*bits)
            if s.config_id not in seen:
                seen.add(s.config_id)
                strategies.append(s)
        return strategies

    def describe(self) -> str:
        if self.is_solo:
            return "Solo (single agent, no collaboration)"
        parts = [f"Spawn: ON"]
        parts.append(f"Parallel: {'ON' if self.parallel_execution else 'OFF'}")
        parts.append(f"Sharing: {'ON' if self.share_partial_results else 'OFF'}")
        parts.append(f"Synthesis: {'ON' if self.synthesis_review else 'OFF'}")
        return " | ".join(parts)
