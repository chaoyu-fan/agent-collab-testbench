"""Load tasks from JSON files."""

from __future__ import annotations

import json
from pathlib import Path

from testbench.models.task import Task, TaskSuite


def load_tasks_json(path: Path | str) -> TaskSuite:
    """Load a task suite from a JSON file.

    Expected format:
    {
        "name": "suite name",
        "tasks": [
            {
                "task_id": "...",
                "description": "...",
                "ground_truth": "...",
                "repository": null,
                "base_commit": null,
                "test_command": null,
                "metadata": {}
            }
        ]
    }
    """
    path = Path(path)
    with open(path) as f:
        data = json.load(f)

    tasks = []
    for t in data["tasks"]:
        tasks.append(
            Task(
                task_id=t["task_id"],
                description=t["description"],
                ground_truth=t["ground_truth"],
                repository=t.get("repository"),
                base_commit=t.get("base_commit"),
                test_command=t.get("test_command"),
                metadata=t.get("metadata", {}),
            )
        )

    return TaskSuite(
        name=data.get("name", path.stem),
        tasks=tasks,
        source_format="json",
    )
