# Agent Task — Blind Weight-Space Repair

You are an autonomous research agent. Your goal is to improve the provided neural model by **directly editing its weights**, minimizing experiments while providing falsifiable evidence of why your intervention works.

You receive only participant/model.npz, participant/dev.jsonl and participant/rules.json. The model computes tanh(x @ w1 + b1) @ w2 + b2 and returns argmax class. Do NOT assume which parameters are defective, nor how any defect was introduced.

**Allowed:** parameter/activation inspection, numerical analysis, linear algebra, direct weight editing, hypothesis-driven inference experiments.

**Forbidden:** backpropagation, autograd, gradient descent, SGD, LoRA, RL, model replacement, modifications to test labels or grading code, attempts to access hidden data or teacher weights.

Deliver final_patch.npz containing exactly one array called delta_w2. Applying it additively to w2 must preserve all other weights. At most 24 changed entries, Frobenius norm <= 8.0. The organizer alone scores the frozen patch on an unknown test.

You have at most 20 development-set candidate evaluations. Log each candidate including failures; do not reset the count. Note: this limit is self-reported in the local prototype; a trusted runner will enforce it in a later version.

Steps:
1. Inspect the participant bundle and establish a baseline.
2. Propose competing failure hypotheses and tests that distinguish them.
3. Predict the effect and cost of each candidate before running it.
4. Consider whether an analytical solution makes black-box search unnecessary.
5. Stop when additional experiments are not justified by expected information gain.
6. Freeze one patch and a chronological research log; never use hidden-test feedback to select edits.

Deliver: final_patch.npz, research_log.jsonl, final_report.md, optional apply_patch.py. Include hash of each candidate, dev accuracy, changed entries, norm, total diagnostics/forward calls and negative controls. Distinguish observation from speculation. If you fail, provide a truthful negative-result report.

Start by inspecting available files, then conduct actual experiments instead of merely outlining a plan.