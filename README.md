# WeightSpace Agent Lab — v0.2 source-grounded redesign

**Research question:** Can an AI research agent discover, mathematically characterize and repair neural failure mechanisms more efficiently than brute-force search and strong non-LLM optimizers, given identical information and compute?

Status (2026-11-13): source-audited design, CPU-side broker prototype, conditional LP baseline. **No real Qwen GPU experiment or verified Agent capability gain.** The original small NumPy classifier is now designated a plumbing smoke test ONLY.

## Read first
- [MIT original audit and previous errors](methodology/RESEARCH_AUDIT.md)
- [Controlled experiment matrix](methodology/EXPERIMENT_MATRIX.md)
- [Honest implementation status](methodology/STATUS.md)
- [MIT Experiment 8 logs](https://github.com/RikiyaT/lm_understand-logs/tree/main/exp8_math)
- [Nathan Barry independent LP analysis](https://nathan.rs/posts/weight-space-lp/)
- [Existing WSP-Bench](https://github.com/GPT-AGI/Weight-Space-Programing) — seeded Qwen repair is ALREADY being studied; do not claim it as novel.

## Three distinct tracks
1. **R — reproduction:** rebuild source-logged weight edit and pin exact vLLM think35 settings, model revision, grader and samples. This is not Agent discovery.
2. **D — algorithmic insight:** identical initial observations for Agent and no-LLM pipelines, matched model-query and token/GPU budget, isolate diagnosis, localization, direction estimation, conditional optimizer choice, and validation. LP with privileged layers/direction is an ablation, not a fair end-to-end competitor.
3. **G — unseen repair:** hidden fault mechanisms/architectures, behavior preservation, negative controls including no-edit, and held-out evaluation. Published original Qwen loop fix is an open-book demonstration, NOT eligible for blind-discovery ranking.

## Actual code status

Install numerical-method dependencies and run unit tests:

~~~bash
pip install -r requirements-methods.txt
PYTHONPATH=. python -m unittest discover -s tests -v
~~~

The historical weightlab.py pipeline is a toy smoke test, not the main experiment. qwen_eval.py is an *unverified Transformers pilot* and is NOT equivalent to original vLLM sampling.

New: wsp_broker.py implements MIT-shaped batch/pass quotas and hash-chained audit records with mockable inference backend, for single-process validation only. New: baseline/lp_conditional.py implements the mathematical downstream LP requiring known behavior direction, relevant layers and labeled activations. Neither implements a live isolated Qwen run. Tests validate software components, not scientific claims.

## Findings must be stratified

Report finish-with-box rate, accuracy conditional on finishing, total correctness, truncation/loop failures, tokens and GPU time, mechanism evidence, edit norm/size, general capability regression and paired confidence intervals. Pass count alone is not compute. Do not publish invented intelligence scores or compare differently sampled MATH-500 figures.

Project evolution: [Issue #1](https://github.com/chaoyu-fan/agent-collab-testbench/issues/1). Original code license MIT; cite MIT logs and Nathan Barry, preserving their independent licenses.
