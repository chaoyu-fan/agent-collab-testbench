"""Execution trace recording."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from testbench.models.strategy import CollaborationStrategy


@dataclass
class TraceEvent:
    """A single event in an execution trace."""

    timestamp: float
    phase: str  # INIT, SPAWN_SUBAGENTS, PARALLEL_EXECUTION, etc.
    agent_name: str
    event_type: str  # llm_call, tool_use, message_sent, spawn, result
    content: str
    tokens_in: int = 0
    tokens_out: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionTrace:
    """Full trace of a single (task, strategy, run) execution."""

    task_id: str
    strategy: CollaborationStrategy
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    events: list[TraceEvent] = field(default_factory=list)
    start_time: float = field(default_factory=time.time)
    end_time: float | None = None
    final_output: str = ""
    agent_snapshots: dict[str, list[str]] = field(default_factory=dict)

    @property
    def total_tokens_in(self) -> int:
        return sum(e.tokens_in for e in self.events)

    @property
    def total_tokens_out(self) -> int:
        return sum(e.tokens_out for e in self.events)

    @property
    def wall_time_seconds(self) -> float:
        if self.end_time is None:
            return time.time() - self.start_time
        return self.end_time - self.start_time

    def add_event(
        self,
        phase: str,
        agent_name: str,
        event_type: str,
        content: str,
        tokens_in: int = 0,
        tokens_out: int = 0,
        **metadata: Any,
    ) -> TraceEvent:
        event = TraceEvent(
            timestamp=time.time(),
            phase=phase,
            agent_name=agent_name,
            event_type=event_type,
            content=content,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            metadata=metadata,
        )
        self.events.append(event)
        return event

    def finalize(self, output: str) -> None:
        self.final_output = output
        self.end_time = time.time()

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "strategy_config_id": self.strategy.config_id,
            "run_id": self.run_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "wall_time_seconds": self.wall_time_seconds,
            "total_tokens_in": self.total_tokens_in,
            "total_tokens_out": self.total_tokens_out,
            "final_output": self.final_output,
            "agent_snapshots": self.agent_snapshots,
            "events": [
                {
                    "timestamp": e.timestamp,
                    "phase": e.phase,
                    "agent_name": e.agent_name,
                    "event_type": e.event_type,
                    "content": e.content,
                    "tokens_in": e.tokens_in,
                    "tokens_out": e.tokens_out,
                    "metadata": e.metadata,
                }
                for e in self.events
            ],
        }
