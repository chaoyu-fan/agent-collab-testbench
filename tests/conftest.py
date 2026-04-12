"""Shared test fixtures."""

from __future__ import annotations

import pytest

from testbench.models.task import Task, TaskSuite
from testbench.models.strategy import CollaborationStrategy
from testbench.llm.mock import MockLLMClient


@pytest.fixture
def mock_llm() -> MockLLMClient:
    return MockLLMClient()


@pytest.fixture
def sample_task() -> Task:
    return Task(
        task_id="test_001",
        description="Fix the off-by-one error in the pagination function.",
        ground_truth="Change items[start:start + page_size] to items[start:min(start + page_size, len(items))].",
    )


@pytest.fixture
def sample_suite(sample_task: Task) -> TaskSuite:
    return TaskSuite(name="test_suite", tasks=[sample_task])


@pytest.fixture
def solo_strategy() -> CollaborationStrategy:
    return CollaborationStrategy()  # All False


@pytest.fixture
def full_collab_strategy() -> CollaborationStrategy:
    return CollaborationStrategy(
        spawn_subagents=True,
        parallel_execution=True,
        share_partial_results=True,
        synthesis_review=True,
    )
