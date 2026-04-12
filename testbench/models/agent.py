"""Agent identity and inter-agent message models."""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class AgentIdentity:
    """Identifies an agent and its role."""

    name: str
    role: str  # orchestrator, solo, decomposer, specialist, synthesizer
    task_scope: str = ""  # What part of the problem this agent handles


@dataclass
class AgentMessage:
    """A message passed between agents."""

    sender: str
    content: str
    message_type: str  # partial_result, final_result, query, broadcast
    receiver: str | None = None  # None = broadcast to all
    timestamp: float = field(default_factory=time.time)
