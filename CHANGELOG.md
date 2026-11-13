# Changelog

## v0.1.0 — 2026-11-13

- Launched an isolated research project on the weightspace-agent-lab branch.
- Defined two tracks: CPU Blind Repair (tested) and Qwen3.5-0.8B weight editing (pilot only).
- Added reproducible CPU benchmark generator, participant/organizer separation, immutable artifact hashes, patch validity checks, and held-out grading.
- Added 7 local Python unittest checks and no-op end-to-end smoke test; no-op accuracy gain = 0 as expected.
- Added optional Qwen inference evaluator with last boxed math grading, truncation and token statistics.
- Added Agent prompts, protocol, citations, roadmap and an automated CPU smoke-test workflow.
- Published GitHub issue #1 for research milestones and experiment updates.

**Important:** no Qwen GPU evaluation or actual agent intelligence comparison has been run; the CPU track has no trusted call-budget enforcement. The workflow was committed but its GitHub-hosted run result was not verified here.
