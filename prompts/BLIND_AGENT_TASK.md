# Blind Weight-Space Research Challenge — participant-facing v0.2

You are the research agent responsible for improving a provided neural model through **direct changes to its weights**. No one has told you which component, if any, is defective. Treat all failure explanations as falsifiable hypotheses.

## Objective

Increase **hidden held-out task accuracy**, preserving general behavior, without gradient descent, backpropagation, fine-tuning, LoRA, RL, leaked test results or known published weight patches. Evidence-based no-edit conclusions are permitted when proposed changes consistently cause harm.

## Tooling (to be provided by the organizer)

1. A persistent CPU Python kernel over read-only original weights and reversible edited tensors: NAMES, load(name), load_original(name), save_edit(name,tensor), drop_edit(name), edited(), tokenizer tok. You must not run the model yourself.
2. An independently controlled forward tool: 10, 25 or 50 successful queries (your assigned budget), either:
   - generation up to 512 sequences on original OR edited, or 256 sequences PER model for paired original/edited, with fixed per-item random seeds;
   - teacher-forced read of up to 1024 texts, at most 524288 total input tokens and at most 32768 per text; hidden states, MLP activations, attention or log-probs may be requested if supported.
3. An authorized practice set with answers, disjoint from the organizer-only hidden test. The organizer controls queries, final scoring, model/token budgets and GPU cost logs.

## Rules

- Preserve architecture, vocabulary, chat template and sampling. Only change organizer-allowed weight values. Respect restrictions on embeddings, final normalization, output head and uniform scaling when the manifest sets them.
- No direct model inference except through the forward tool; no secret data, external network, optimizer-based training or published target-specific patches.
- Numerical linear algebra, optimization theory and direct analytically justified weight transformations are allowed. Do not exclude an efficient mathematical algorithm just to satisfy a narrow guess about how the edit ought to be found.
- Before each costly experiment, record the mechanism hypothesis, predicted observation, falsifying observation, estimated cost and reason this experiment beats alternatives.
- Paired evaluations must reuse the same practice questions and seeds. Do not use a held-out score to select among candidate edits.
- Never assume the failure involves repeated reasoning, any particular layer or a rank-one projection. Discover the mechanism or report failure.
- Do not merely force early termination or skip all reasoning to increase completion rate; correctness and general-capability preservation matter.

## Required artifacts

Submit a deterministic patch script or complete edited tensors; affected tensor names and exact element changes; full chronological log of ALL candidate queries including failures, tokens and GPU cost as measured by the organizer; negative and dose-response controls; mechanism report distinguishing evidence from interpretation; and one final **frozen** candidate.

The organizer conducts independent hidden tests and reports pass budget, actual generated tokens, GPU seconds, finished-with-valid-box rate, correct-given-finished, overall accuracy, general capability regressions and the confidence interval on matched tasks.

Start by inspecting the model and environment. Conduct independent research through the provided tools. Do not substitute unexecuted plans or fabricated scores for actual experiments.

**Important:** this is a participant task contract for a forthcoming isolated runner. The current repository's Python mock broker is not itself an executable end-to-end Agent environment.
