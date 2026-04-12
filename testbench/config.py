"""Run configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class RunConfig:
    """Controls how the testbench executes experiments."""

    runs_per_config: int = 3
    temperature: float = 0.0
    max_tokens_per_agent: int = 4096
    agent_timeout_seconds: float = 300.0
    max_subtasks: int = 5
    concurrency_limit: int = 4
    output_dir: Path = field(default_factory=lambda: Path("output"))
    mock_mode: bool = False
