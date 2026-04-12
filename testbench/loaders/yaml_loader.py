"""Load tasks from YAML files."""

from __future__ import annotations

from pathlib import Path

import yaml

from testbench.models.task import Task, TaskSuite


def load_tasks_yaml(path: Path | str) -> TaskSuite:
    """Load a task suite from a YAML file. Same schema as JSON loader."""
    path = Path(path)
    with open(path) as f:
        data = yaml.safe_load(f)

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
        source_format="yaml",
    )
