"""SharedBlackboard for partial result cross-pollination between agents."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field


@dataclass
class SharedBlackboard:
    """Async-safe shared state for broadcasting partial results.

    Agents post work summaries here. Other agents can read snapshots
    to incorporate into their context (cross-pollination).
    """

    _state: dict[str, list[str]] = field(default_factory=dict)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    _version: int = 0

    async def post(self, agent_name: str, summary: str) -> None:
        """Post a work summary for the given agent."""
        async with self._lock:
            self._state.setdefault(agent_name, []).append(summary)
            self._version += 1

    async def snapshot(self, exclude_agent: str | None = None) -> dict[str, list[str]]:
        """Get a snapshot of all posted summaries, optionally excluding one agent."""
        async with self._lock:
            if exclude_agent:
                return {k: list(v) for k, v in self._state.items() if k != exclude_agent}
            return {k: list(v) for k, v in self._state.items()}

    def snapshot_sync(self, exclude_agent: str | None = None) -> dict[str, list[str]]:
        """Synchronous snapshot (for use in prompt building)."""
        if exclude_agent:
            return {k: list(v) for k, v in self._state.items() if k != exclude_agent}
        return {k: list(v) for k, v in self._state.items()}

    @property
    def version(self) -> int:
        return self._version

    def clear(self) -> None:
        self._state.clear()
        self._version = 0
