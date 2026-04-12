"""JSON serialization helpers for dataclasses."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def save_json(data: dict[str, Any] | list[Any], path: Path | str) -> None:
    """Write data to a JSON file, creating parent directories as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def load_json(path: Path | str) -> Any:
    """Read a JSON file."""
    with open(path) as f:
        return json.load(f)
