# Source audit — 2026-11-13

## Direct primary sources
- [MIT CSAIL, Nov 2 2026](https://rikiyat.github.io/blog/weight-space-programming/)
- [Experiment 8 setup, logs and original score provenance](https://github.com/RikiyaT/lm_understand-logs/blob/main/exp8_math/README.md)
- [The actual p50c first-turn Agent task](https://github.com/RikiyaT/lm_understand-logs/blob/main/exp8_math/runs/p50c/turns/t001/prompt.txt)
- [Agent final choice](https://github.com/RikiyaT/lm_understand-logs/blob/main/exp8_math/runs/p50c/final_answer.md)
- [Released checkpoint with SHA checks](https://huggingface.co/RikiyaT/Qwen3.5-0.8B-weight-space-programming)
- [Nathan Barry, Nov 9 2026, mathematical critique](https://nathan.rs/posts/weight-space-lp/)
- [Prior WSP-Bench](https://github.com/GPT-AGI/Weight-Space-Programing)

## Original environment

GPT-6 Astra through restricted Codex CLI, maximal reasoning. Target Qwen3.5-0.8B text model, 24 layers; persistent CPU Python kernel with up to 300 executions; access to load, load_original, save_edit, drop_edit but NOT a local model inference function. Practice data: GSM8K train 7469; MATH train 7275; historical AIME 714 answer-only. Forward-query budgets: 10, 25 or 50. Generation per query up to 512 sequences (paired original and edited up to 256 EACH). Teacher-forced up to 1024 texts, 524288 input tokens, with optional hidden states, MLP activations, attention and token logprobs. No autograd, gradient training, or output embedding/final norm edits. Original Agent never saw held-out benchmarks.

Exact released evaluation: vLLM, thinking on, boxed math prompt, temperature 0.6, top_p 0.95, top_k 20, presence_penalty 1.5, 32768 max new tokens, last boxed answer with math-verify. Samples/problem: MATH-500 4, GSM8K 2, AMC23 16, Minerva 4, OlympiadBench 4, MMLU-Pro 1, GPQA Diamond 4, IFEval 2. Base revision in published patch card: 2fc06364715b967f1860aea9cf38778875588b17.

Reported original results ONLY, not ours: baseline MATH-500 7.9%, GSM8K 13.8%, IFEval 21.0%, MATH-500 cut-off share 93.0%; 10-pass MATH 12.3%, 25-pass MATH 14.1%, 50-pass MATH 48.1%, GSM8K 67.1%, IFEval 62.3%, cut-off 33.1%. p50c took 158 Agent turns, 80 Python calls, 20.3 wall hours including server outage; final change altered 64 down_proj output columns each in MLP layers 15 and 19 (131072 scalar weights).

## What the LP critique really proves

For output Wh, rank-one edit delta W = u v^T shifts residual by u (v^T h). Given known layer, behavior direction u and looping/productive activations h, finding v is a linear program with L1 sparsity and slack-penalized inequalities. Nathan Barry reports within HIS one-sample MATH-500 protocol: original Agent patch 50.9%, LP with original u 51.7%, heuristic-u LP 47.8%, about 8x fewer scalar parameter edits. The LP still relies on the original Agent localization, and in its strongest arm the original direction. It does NOT independently solve the full scientific task. His scores are not directly comparable to the MIT 4-sample MATH-500 48.1%.

## Where v0.1 was wrong

- The tiny 16→32→4 toy classifier is unrelated to genuine Qwen math/loop behavior.
- Artificial 24-coordinate/20-trial constraints do not reflect MIT tools or budgets.
- No independently metered forward interface with activation/MLP probes was implemented.
- The HF Qwen evaluation pilot differs from the vLLM sampling protocol, token budget and sample count.
- No full non-LLM baseline, conditional LP or compute-matched trial was included.
- Toy unit tests were presented too closely to claims of Agent readiness; NO Qwen GPU runs were made.
- Seeded Qwen fault repair is not unclaimed novelty; prior WSP-Bench exists.

Next step is truthful reproducibility, then controlled discovery — not a leaderboard based on a fabricated IQ score.
