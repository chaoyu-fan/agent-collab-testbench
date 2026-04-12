"""Tests for SharedBlackboard and MessageBus."""

import pytest

from testbench.orchestrator.shared_state import SharedBlackboard
from testbench.orchestrator.message_bus import MessageBus
from testbench.models.agent import AgentMessage


class TestSharedBlackboard:
    @pytest.mark.asyncio
    async def test_post_and_snapshot(self):
        bb = SharedBlackboard()
        await bb.post("agent_1", "Found a bug in auth module")
        await bb.post("agent_2", "Investigating DB layer")

        snap = await bb.snapshot()
        assert "agent_1" in snap
        assert "agent_2" in snap
        assert snap["agent_1"] == ["Found a bug in auth module"]

    @pytest.mark.asyncio
    async def test_snapshot_exclude(self):
        bb = SharedBlackboard()
        await bb.post("agent_1", "my work")
        await bb.post("agent_2", "other work")

        snap = await bb.snapshot(exclude_agent="agent_1")
        assert "agent_1" not in snap
        assert "agent_2" in snap

    @pytest.mark.asyncio
    async def test_version_increments(self):
        bb = SharedBlackboard()
        assert bb.version == 0
        await bb.post("a", "x")
        assert bb.version == 1
        await bb.post("a", "y")
        assert bb.version == 2

    def test_snapshot_sync(self):
        bb = SharedBlackboard()
        bb._state = {"agent_1": ["work1"], "agent_2": ["work2"]}
        snap = bb.snapshot_sync(exclude_agent="agent_1")
        assert "agent_1" not in snap
        assert snap["agent_2"] == ["work2"]

    def test_clear(self):
        bb = SharedBlackboard()
        bb._state = {"a": ["x"]}
        bb._version = 5
        bb.clear()
        assert bb._state == {}
        assert bb.version == 0


class TestMessageBus:
    @pytest.mark.asyncio
    async def test_send_and_receive(self):
        bus = MessageBus()
        msg = AgentMessage(sender="a", content="hello", message_type="query", receiver="b")
        await bus.send(msg)

        msgs = await bus.get_messages(receiver="b")
        assert len(msgs) == 1
        assert msgs[0].content == "hello"

    @pytest.mark.asyncio
    async def test_broadcast(self):
        bus = MessageBus()
        msg = AgentMessage(sender="a", content="update", message_type="broadcast")
        await bus.send(msg)

        # Both receivers should see broadcasts
        msgs_b = await bus.get_messages(receiver="b")
        msgs_c = await bus.get_messages(receiver="c")
        assert len(msgs_b) == 1
        assert len(msgs_c) == 1

    @pytest.mark.asyncio
    async def test_since_index(self):
        bus = MessageBus()
        await bus.send(AgentMessage(sender="a", content="old", message_type="info"))
        await bus.send(AgentMessage(sender="a", content="new", message_type="info"))

        msgs = await bus.get_messages(since_index=1)
        assert len(msgs) == 1
        assert msgs[0].content == "new"

    @pytest.mark.asyncio
    async def test_message_count(self):
        bus = MessageBus()
        assert bus.message_count == 0
        await bus.send(AgentMessage(sender="a", content="x", message_type="info"))
        assert bus.message_count == 1
