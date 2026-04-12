"""Tests for mock LLM client and agent models."""

import pytest

from testbench.llm.mock import MockLLMClient
from testbench.models.agent import AgentIdentity, AgentMessage


class TestMockLLMClient:
    @pytest.mark.asyncio
    async def test_decomposition_response(self, mock_llm: MockLLMClient):
        import json
        resp = await mock_llm.complete(
            messages=[{"role": "user", "content": "Please decompose this task into subtasks."}]
        )
        data = json.loads(resp.content)
        assert "subtasks" in data
        assert len(data["subtasks"]) == 3
        assert resp.tokens_in > 0
        assert resp.tokens_out > 0
        assert resp.model == "mock"

    @pytest.mark.asyncio
    async def test_synthesis_response(self, mock_llm: MockLLMClient):
        resp = await mock_llm.complete(
            messages=[{"role": "user", "content": "Please synthesize and merge these results."}]
        )
        assert "[Synthesis" in resp.content

    @pytest.mark.asyncio
    async def test_specialist_response(self, mock_llm: MockLLMClient):
        resp = await mock_llm.complete(
            messages=[{"role": "user", "content": "Fix the bug in this function."}]
        )
        assert "[Agent" in resp.content

    @pytest.mark.asyncio
    async def test_call_count(self, mock_llm: MockLLMClient):
        assert mock_llm.call_count == 0
        await mock_llm.complete(messages=[{"role": "user", "content": "hello"}])
        assert mock_llm.call_count == 1
        await mock_llm.complete(messages=[{"role": "user", "content": "world"}])
        assert mock_llm.call_count == 2

    @pytest.mark.asyncio
    async def test_deterministic(self, mock_llm: MockLLMClient):
        """Same prompt produces same output."""
        msgs = [{"role": "user", "content": "Fix the bug please."}]
        r1 = await mock_llm.complete(messages=msgs)
        r2 = await mock_llm.complete(messages=msgs)
        assert r1.content == r2.content


class TestAgentModels:
    def test_agent_identity(self):
        a = AgentIdentity(name="agent_1", role="specialist", task_scope="auth module")
        assert a.name == "agent_1"
        assert a.role == "specialist"

    def test_agent_message(self):
        m = AgentMessage(
            sender="agent_1",
            content="Found the bug",
            message_type="partial_result",
        )
        assert m.receiver is None  # broadcast
        assert m.timestamp > 0
