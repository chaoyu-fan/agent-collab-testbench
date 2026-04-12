"""Agent lifecycle management — spawn, track, timeout, cancel."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from testbench.agents.base import AgentResult, BaseAgent

logger = logging.getLogger("testbench.lifecycle")


class LifecycleManager:
    """Manages agent coroutines with timeout and cancellation support."""

    def __init__(self, timeout_seconds: float = 300.0) -> None:
        self.timeout_seconds = timeout_seconds
        self._tasks: dict[str, asyncio.Task[AgentResult]] = {}

    def spawn(self, agent: BaseAgent) -> asyncio.Task[AgentResult]:
        """Spawn an agent as an asyncio task."""
        task = asyncio.create_task(agent.run(), name=agent.name)
        self._tasks[agent.name] = task
        logger.debug("Spawned agent: %s", agent.name)
        return task

    async def await_all(self) -> list[AgentResult]:
        """Wait for all spawned agents to complete, with timeout."""
        if not self._tasks:
            return []
        try:
            results = await asyncio.wait_for(
                asyncio.gather(*self._tasks.values(), return_exceptions=True),
                timeout=self.timeout_seconds,
            )
        except asyncio.TimeoutError:
            logger.warning("Agent timeout after %.1fs, cancelling remaining", self.timeout_seconds)
            for task in self._tasks.values():
                if not task.done():
                    task.cancel()
            # Gather again to collect results and exceptions
            results = await asyncio.gather(*self._tasks.values(), return_exceptions=True)

        agent_results: list[AgentResult] = []
        for result in results:
            if isinstance(result, BaseException):
                logger.error("Agent failed: %s", result)
            else:
                agent_results.append(result)

        self._tasks.clear()
        return agent_results

    async def run_sequential(self, agents: list[BaseAgent]) -> list[AgentResult]:
        """Run agents one at a time, sequentially."""
        results: list[AgentResult] = []
        for agent in agents:
            try:
                result = await asyncio.wait_for(
                    agent.run(),
                    timeout=self.timeout_seconds,
                )
                results.append(result)
            except asyncio.TimeoutError:
                logger.warning("Agent %s timed out", agent.name)
            except Exception as e:
                logger.error("Agent %s failed: %s", agent.name, e)
        return results

    def cancel(self, agent_name: str) -> bool:
        """Cancel a specific agent."""
        task = self._tasks.get(agent_name)
        if task and not task.done():
            task.cancel()
            return True
        return False

    @property
    def active_count(self) -> int:
        return sum(1 for t in self._tasks.values() if not t.done())
