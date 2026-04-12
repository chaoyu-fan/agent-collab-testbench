"""Report generation — markdown, CSV, and JSON output."""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path

from testbench.analysis.comparator import ComparisonReport


def generate_markdown(report: ComparisonReport) -> str:
    """Generate a markdown report from the comparison analysis."""
    lines: list[str] = []
    lines.append("# Agent Collaboration Ablation Report\n")

    # Strategy comparison table
    lines.append("## Strategy Results\n")
    lines.append("| Config | Pass Rate | Similarity | Tokens | Time (s) | Agents | Runs |")
    lines.append("|--------|-----------|------------|--------|----------|--------|------|")
    for cid in sorted(report.strategy_metrics):
        m = report.strategy_metrics[cid]
        lines.append(
            f"| {cid} | {m.pass_rate:.2%} | {m.mean_similarity:.3f} | "
            f"{m.mean_tokens:.0f} | {m.mean_wall_time:.3f} | "
            f"{m.mean_agent_count:.1f} | {m.num_runs} |"
        )

    lines.append("")
    lines.append(f"**Best strategy**: `{report.best_strategy}`")
    lines.append(f"**Worst strategy**: `{report.worst_strategy}`")
    lines.append("")

    # Phase ablation table
    lines.append("## Phase Ablation Analysis\n")
    lines.append("| Phase | Pass Rate (ON) | Pass Rate (OFF) | Delta | Tokens (ON) | Tokens (OFF) |")
    lines.append("|-------|---------------|-----------------|-------|-------------|--------------|")
    for a in report.phase_ablations:
        lines.append(
            f"| {a.phase} | {a.pass_rate_on:.2%} | {a.pass_rate_off:.2%} | "
            f"{a.delta:+.2%} | {a.tokens_on:.0f} | {a.tokens_off:.0f} |"
        )

    lines.append("")

    # Interpretation
    lines.append("## Interpretation\n")
    for a in report.phase_ablations:
        if a.delta > 0.05:
            lines.append(f"- **{a.phase}**: Collaboration HELPS (+{a.delta:.1%} pass rate)")
        elif a.delta < -0.05:
            lines.append(f"- **{a.phase}**: Collaboration HURTS ({a.delta:.1%} pass rate)")
        else:
            lines.append(f"- **{a.phase}**: Negligible effect ({a.delta:+.1%})")

    return "\n".join(lines)


def generate_csv(report: ComparisonReport) -> str:
    """Generate CSV output for external analysis."""
    output = io.StringIO()
    writer = csv.writer(output)

    writer.writerow([
        "config_id", "pass_rate", "mean_similarity", "mean_tokens",
        "mean_wall_time", "mean_agent_count", "num_runs",
    ])
    for cid in sorted(report.strategy_metrics):
        m = report.strategy_metrics[cid]
        writer.writerow([
            cid, f"{m.pass_rate:.4f}", f"{m.mean_similarity:.4f}",
            f"{m.mean_tokens:.1f}", f"{m.mean_wall_time:.3f}",
            f"{m.mean_agent_count:.2f}", m.num_runs,
        ])

    return output.getvalue()


def generate_json(report: ComparisonReport) -> str:
    """Generate JSON output for programmatic consumption."""
    data = {
        "strategy_metrics": {
            cid: m.to_dict() for cid, m in report.strategy_metrics.items()
        },
        "phase_ablations": [
            {
                "phase": a.phase,
                "pass_rate_on": a.pass_rate_on,
                "pass_rate_off": a.pass_rate_off,
                "delta": a.delta,
                "tokens_on": a.tokens_on,
                "tokens_off": a.tokens_off,
            }
            for a in report.phase_ablations
        ],
        "best_strategy": report.best_strategy,
        "worst_strategy": report.worst_strategy,
    }
    return json.dumps(data, indent=2)


def save_report(
    report: ComparisonReport,
    output_dir: Path,
    formats: list[str] | None = None,
) -> list[Path]:
    """Save reports in multiple formats. Returns paths of saved files."""
    if formats is None:
        formats = ["markdown", "csv", "json"]

    output_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    generators = {
        "markdown": (generate_markdown, "report.md"),
        "csv": (generate_csv, "report.csv"),
        "json": (generate_json, "report.json"),
    }

    for fmt in formats:
        if fmt in generators:
            gen_fn, filename = generators[fmt]
            path = output_dir / filename
            path.write_text(gen_fn(report))
            saved.append(path)

    return saved
