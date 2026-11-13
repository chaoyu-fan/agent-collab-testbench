# Autonomous Weight-Space Programming — Qwen3.5-0.8B Track

You are a research agent with authorized local access to a frozen Qwen3.5-0.8B checkpoint, a Python environment and a controlled model-query interface. Your goal is to discover a reproducible **direct weight edit** that improves held-out mathematical performance while minimizing actual compute.

Your instructions are task-level, not a prescription of how to solve it. Do not assume the bottleneck is a specific layer, MLP neuron, attention head, repetition behavior or decoding setting. Develop competing explanations and use tests that distinguish them.

## Non-negotiable constraints

- The improvement must be due to modifying neural weights, not changing prompts, tokenizer, chat template, decoding, model architecture, benchmark grader or benchmark answers.
- No backpropagation, autograd, gradient descent, fine-tuning, LoRA, RL or importing a released third-party weight patch.
- Do not modify embedding matrices, output head, or final normalization weights. Other weight edits are allowed subject to the experiment budget.
- Do not inspect held-out evaluation items, hidden answers or grader internals. Development examples may be examined through the authorized tool interface.
- Do not claim improvements from dev-only results. A patch is successful only when an independent organizer validates it on held-out items.
- Maintain a complete experiment log. Never invent counts, scores or hardware metrics.

## Budget

The organizer allocates one of three forward-query budgets: 10 / 25 / 50. A forward query can contain multiple examples if the organizer permits it. You must not bypass its counter or make hidden calls. In addition, total generated tokens, GPU-time and wall-clock time are independently logged. Different forward queries are not equally costly.

Each candidate evaluation should record a pre-registered hypothesis, expected effect, changed tensors, changed parameter counts, candidate hash, diagnostic calls and observed dev results. Before spending an expensive query, consider whether an analytic or mechanistic insight could avoid a search.

## Deliverables

1. apply_edit.py — deterministic, non-gradient code that transforms the original checkpoint into an edited checkpoint.
2. edit_manifest.json — checkpoint revision/hash, affected tensors, changed elements, norms, script SHA and environment details.
3. experiment_log.jsonl — append-only model calls and experiment ledger with timestamps, tokens and hardware costs.
4. mechanism_report.md — competing hypotheses, controls, negative results and causal limitations.
5. final_checkpoint — local edited weights if storage allows.

The **independent** evaluator measures math accuracy, boxed-completion rate, truncation, mean tokens, and capability regressions under *matched* decoding. Compare non-thinking and prompt-only controls in separate experimental arms, not as weight-edit evidence. The six-item dataset in the repository is a smoke test and cannot support an accuracy claim.

Do not use external published solutions for this exact target. If model/compute resources are missing, report the blocker, rather than simulating measurements.

Start by inspecting files and hardware, fingerprinting the checkpoint, establishing a reproducible baseline and pursuing budgeted research experiments.