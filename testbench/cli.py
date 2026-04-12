"""CLI interface for the testbench."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

from testbench.analysis.comparator import Comparator
from testbench.analysis.report import generate_markdown, save_report
from testbench.config import RunConfig
from testbench.evaluation.metrics import compute_strategy_metrics
from testbench.evaluation.patch_match import PatchMatchEvaluator
from testbench.llm.mock import MockLLMClient
from testbench.loaders.json_loader import load_tasks_json
from testbench.models.result import EvaluationResult
from testbench.models.strategy import CollaborationStrategy
from testbench.orchestrator.orchestrator import Orchestrator
from testbench.utils.logging import setup_logging
from testbench.utils.serialization import save_json


async def run_experiment(args: argparse.Namespace) -> None:
    """Run the testbench experiment."""
    logger = setup_logging(logging.DEBUG if args.verbose else logging.INFO)

    # Load tasks
    suite = load_tasks_json(args.tasks)
    logger.info("Loaded %d tasks from '%s'", len(suite.tasks), suite.name)

    # Parse strategies
    if args.strategies == "all":
        strategies = CollaborationStrategy.all_strategies()
    else:
        strategies = [
            CollaborationStrategy.from_config_id(s.strip())
            for s in args.strategies.split(",")
        ]
    logger.info("Testing %d strategies", len(strategies))

    # Config
    config = RunConfig(
        runs_per_config=args.runs,
        output_dir=Path(args.output),
        mock_mode=args.mock,
    )

    # LLM client
    if args.mock:
        llm = MockLLMClient()
    else:
        from testbench.llm.client import AnthropicLLMClient
        model = getattr(args, "model", None) or "claude-sonnet-4-20250514"
        llm = AnthropicLLMClient(
            model=model,
            concurrency_limit=config.concurrency_limit,
        )

    # Evaluator
    evaluator = PatchMatchEvaluator(pass_threshold=args.threshold)

    # Run experiments
    all_results: list[EvaluationResult] = []
    total_runs = len(suite.tasks) * len(strategies) * config.runs_per_config
    completed = 0

    for task in suite.tasks:
        for strategy in strategies:
            for run_idx in range(config.runs_per_config):
                orch = Orchestrator(llm, strategy, config)
                run_result = await orch.run_task(task)

                # Save trace
                trace_path = (
                    config.output_dir / "traces"
                    / f"{task.task_id}_{strategy.config_id}_{run_idx}.json"
                )
                save_json(run_result.trace.to_dict(), trace_path)

                # Evaluate
                eval_result = evaluator.evaluate(task, run_result)
                all_results.append(eval_result)

                completed += 1
                logger.info(
                    "[%d/%d] task=%s strategy=%s run=%d passed=%s similarity=%.3f",
                    completed, total_runs,
                    task.task_id, strategy.config_id, run_idx,
                    eval_result.passed, eval_result.patch_similarity,
                )

    # Save raw results
    results_path = config.output_dir / "results.json"
    save_json([r.to_dict() for r in all_results], results_path)
    logger.info("Saved %d results to %s", len(all_results), results_path)

    # Generate analysis report
    comparator = Comparator()
    report = comparator.analyze(all_results)
    saved_paths = save_report(report, config.output_dir)
    for p in saved_paths:
        logger.info("Report saved: %s", p)

    # Print summary to stdout
    print(generate_markdown(report))


async def analyze_results(args: argparse.Namespace) -> None:
    """Analyze previously saved results."""
    logger = setup_logging(logging.INFO)

    results_path = Path(args.input) / "results.json"
    if not results_path.exists():
        logger.error("No results.json found in %s", args.input)
        sys.exit(1)

    with open(results_path) as f:
        raw = json.load(f)

    results = [
        EvaluationResult(
            task_id=r["task_id"],
            strategy_config_id=r["strategy_config_id"],
            run_id=r["run_id"],
            passed=r["passed"],
            patch_similarity=r["patch_similarity"],
            tokens_consumed=r["tokens_consumed"],
            wall_time_seconds=r["wall_time_seconds"],
            agent_count=r["agent_count"],
            phases_active=r["phases_active"],
            metrics=r.get("metrics", {}),
        )
        for r in raw
    ]

    comparator = Comparator()
    report = comparator.analyze(results)

    output_dir = Path(args.output) if args.output else Path(args.input)
    fmt = args.format

    if fmt == "markdown":
        print(generate_markdown(report))
    else:
        save_report(report, output_dir, formats=[fmt])
        logger.info("Report saved to %s", output_dir)


def list_strategies(args: argparse.Namespace) -> None:
    """Print all effective strategy configurations."""
    strategies = CollaborationStrategy.all_strategies()
    print(f"{'Config':<8} {'Description'}")
    print(f"{'------':<8} {'-----------'}")
    for s in strategies:
        print(f"{s.config_id:<8} {s.describe()}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="testbench",
        description="Agent Collaboration Ablation Testbench",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # run
    run_parser = subparsers.add_parser("run", help="Run experiments")
    run_parser.add_argument("--tasks", required=True, help="Path to task suite JSON")
    run_parser.add_argument("--strategies", default="all", help="Comma-separated config IDs or 'all'")
    run_parser.add_argument("--runs", type=int, default=3, help="Runs per (task, strategy) pair")
    run_parser.add_argument("--output", default="output", help="Output directory")
    run_parser.add_argument("--mock", action="store_true", help="Use mock LLM")
    run_parser.add_argument("--threshold", type=float, default=0.3, help="Pass threshold for similarity")
    run_parser.add_argument("--model", default=None, help="Model ID for real API (default: claude-sonnet-4-20250514)")
    run_parser.add_argument("-v", "--verbose", action="store_true")

    # analyze
    analyze_parser = subparsers.add_parser("analyze", help="Analyze saved results")
    analyze_parser.add_argument("--input", required=True, help="Directory with results.json")
    analyze_parser.add_argument("--format", default="markdown", choices=["markdown", "csv", "json"])
    analyze_parser.add_argument("--output", default=None, help="Output directory")

    # list-strategies
    subparsers.add_parser("list-strategies", help="List all strategy configurations")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "run":
        asyncio.run(run_experiment(args))
    elif args.command == "analyze":
        asyncio.run(analyze_results(args))
    elif args.command == "list-strategies":
        list_strategies(args)
