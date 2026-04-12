"""Task and TaskSuite definitions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Task:
    """A single problem for the testbench to solve."""

    task_id: str
    description: str
    ground_truth: str  # Expected output or patch
    repository: str | None = None
    base_commit: str | None = None
    test_command: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskSuite:
    """A collection of tasks to run."""

    name: str
    tasks: list[Task]
    source_format: str = "json"
