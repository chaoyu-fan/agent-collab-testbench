# Roadmap (2026-11-13)

| Milestone | Artifact | Status |
|---|---|---|
| P0 | Scientific protocol, task prompt, baseline/controls policy | Done, draft |
| P1 | CPU seedable blind repair, patch validation, organizer grader, tests | Code complete, local 7 tests passed |
| P2 | Qwen3.5-0.8B GPU inference pilot, boxed math grading | Code written, GPU execution unverified |
| P3 | Trusted isolated Agent runner and verified experiment budget | Planned |
| P4 | Random/greedy/analytical baselines and 10/25/50 cost curves | Planned |
| P5 | Frozen GSM8K/MATH-500 revisions, multi-seed validation | Planned |
| P6 | Independent blinded comparative study and public report | Planned |

Next: give two different Coding Agents participant-only bundles with different hidden seeds, compare frozen patches in an organizer container; then build P3 before publishing any leaderboard. For the Qwen track calibrate an unchanged base and a no-op checkpoint under identical inference settings first. One issue per milestone; keep experiment manifests and negative results under version control (without hidden labels or private seeds).