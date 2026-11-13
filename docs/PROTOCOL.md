# Benchmark Protocol v0.1

Research objectives: (1) discover weight-space edits without gradient training; (2) distinguish analytical shortcuts from expensive brute force; (3) separate improved reasoning from reduced truncation/completion failures; (4) measure held-out transfer and robustness.

## Evidence tiers

1. Unit tests and no-op patch: establishes only that the benchmark plumbing works.
2. CPU blind repair: test several secret unseen seeds with hidden labels stored outside Agent's workspace. Measures diagnosis and algorithmic search, not LLM math.
3. Qwen direct editing: frozen Qwen revision, frozen math datasets, pre-registered decoding and evaluation, repeat sampling, held-out test. Requires real GPU verification.

## Private organizer boundary

Only copy participant/ to the contestant environment. Keep hidden_test.npz, teacher.npz and manifest.json isolated and inaccessible. Do not reveal secret seeds. This open-source generator does NOT constitute secure blind testing if organizer files or seeds are exposed. Hold-out grade once after patch freeze.

## Controls

- CPU: no-op, random sparse patch, fixed-cost coordinate search and an organizer-side analytical baseline.
- Qwen: base and edited under identical decoding; thinking-on vs thinking-off and short vs long context reported separately; prompt-only baseline must be labeled as a *different treatment*.
- Use paired per-example outcomes with confidence intervals and fixed dataset revisions.
- Test on unseen seeds and unknown task variants. Avoid benchmark answer contamination.

## Cost accounting

Report hidden Accuracy, Gain_pp, Accuracy@Budget, best-so-far improvement vs verified forward passes, generation tokens, actual GPU seconds, wall clock, time-to-first-positive candidate, changed entries and edit norm.

Do not consider a teacher-forced forward pass and generation of 32,768 tokens equivalent just because each is called one pass. Log both effective model work and calls. Do not create a synthetic IQ score without validation.

## Enforcement gap

This prototype cannot enforce Agent's own “20 evaluations” limit. A production runner must own the evaluator, intercept tool calls, write append-only traces, cap per-Agent resources, prohibit access to organizer data and independently collect GPU metrics. Until implemented, rankings of experiment efficiency are **unverified**.

## Real-model validity

The optional qwen_eval.py is an HF Transformers pilot requiring GPU and math-verify. It grades the last balanced boxed answer and records truncation and token counts. This is **not** MIT's 32k-token presence-penalty vLLM protocol. No comparable MATH-500 claims without reproducing inference and grader settings.

Threats: dev overfitting, leaked seeds, benchmark memorization, evaluator tampering, model swapping, sampling variance, false truncation interpretation, unmeasured compute and mechanism over-interpretation.