# Protocol v0.2 — planned study, NOT results

## R: faithful MIT reproduction
R0 pin official Qwen checkpoint revision, build official p50c edit and verify BF16 weight hashes. R1 evaluate no-op/base and edited checkpoint with the official vLLM think35 sampling, 32768-token limit, presence penalty 1.5, correct grader and exact split/sample counts. R2 reconstruct original trace and resource use. Divergent model hashes or score distributions are blockers, not numbers to conceal. Published patch reuse is forbidden as an eligible autonomous Agent discovery entry.

## D: fixed-budget experimental arms

Common inputs: unknown target weights, practice pool and organizer-controlled forward/activation tools. Hidden test available to organizer only. Budget 10 / 25 / 50 passes AND accounted tokens, GPU time, wallclock. No Agent may run unrestricted model inference. All edits freeze before hidden testing.

| Arm | Starting information | Interpretation |
|---|---|---|
| D0 unchanged | base checkpoint | reference |
| D1 random/grid | identical tool budget | brute-force reference |
| D2 scripted causal probes | same raw data | conventional interpretability |
| D3 conditional LP | GIVEN oracle layers, u and labeled activations | privileged downstream upper bound, NOT fair end-to-end |
| D4 non-LLM discovery + LP | same inputs as Agent | fair automated baseline |
| D5 full autonomous Agent | same raw inputs and query budget | scientific research ability |
| D6 Agent + hint about LP | same plus method hint | isolate insight from prompted recipe |
| D7 Agent without activations | weights and black-box responses | value of mechanism tools |

Three experimental phases must be logged separately: diagnosis/localization; representation/optimization discovery; candidate edit and causal validation. We expect performance to vary by phase, not assume the Agent always wins.

## G: unseen generalization
Avoid direct reuse of published Qwen loop patch as hidden discovery. Include new faults, behavior-specific repair, no-edit correct-choice cases and model-family transfer. Hide organizer model anchor and test labels; do not allow comparison with public original weights. Existing WSP-Bench must be compared and credited, not silently reinvented.

## Metrics and statistical validity
With F meaning completed valid answer and C meaning correct, report P(F), P(C given F), P(C), cutoff/loop fraction; when incomplete responses cannot be correct: P(C) = P(F) * P(C given F). Also edited scalar count, edited weight norm, inference tokens, token-level FLOP proxy, actual GPU seconds, tool calls, wall-clock, cost and damage to other capabilities. Plot best-so-far accuracy lift vs measured compute; count forward passes separately because a single query can perform 512 generations or 1024 teacher-forced reads. Compare paired same problems/seeds; bootstrap by PROBLEM to avoid pseudoreplication over samples.

## Evidence gates
G0 original sources read — DONE. G1 unchanged Qwen evaluation match — NOT DONE. G2 official patch hash-verify — NOT DONE. G3 trusted isolated vLLM forward and edit broker — NOT DONE. G4 matched fully automated LP/no-LLM reference — NOT DONE. G5 blinded repeated transfer study — NOT DONE.
