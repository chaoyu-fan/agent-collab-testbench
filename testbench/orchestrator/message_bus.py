"""Message bus for directed inter-agent communication."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from testbench.models.agent import AgentMessage


@dataclass
class MessageBus:
    """Simple async message bus for agent-to-agent communication."""

    _messages: list[AgentMessage] = field(default_factory=list)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def send(self, message: AgentMessage) -> None:
        async with self._lock:
            self._messages.append(message)

    async def get_messages(
        self,
        receiver: str | None = None,
        since_index: int = 0,
    ) -> list[AgentMessage]:
        """Get messages for a specific receiver (or broadcasts) since an index."""
        async with self._lock:
            result = []
            for msg in self._messages[since_index:]:
                if msg.receiver is None or msg.receiver == receiver:
                    result.append(msg)
            return result

    @property
    def message_count(self) -> int:
        return len(self._messages)

    def clear(self) -> None:
        self._messages.clear()
