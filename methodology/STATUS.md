# Evidence ledger — 2026-11-13

Verified: audited MIT experiment 8 README, first Agent task, final answer and official checkpoint record; Nathan Barry's independent analysis and WSP-Bench overview. CPU-side v0.1 toy tests 7/7 run and pass, **not evidence of LLM mathematical intelligence**. New separate CPU broker and conditional LP math baseline have synthetic unit tests (total local test suite 19/19 pass).

Designed, not done: live Qwen loading, real Agent run, GPU verification, bit-exact MIT patch reconstruction, trusted isolated vLLM service, fair matched compute comparisons, hidden mechanism transfer tests and public leaderboard.

Caution: wsp_broker.py is a mockable in-process prototype with hash-chained ledger, NOT a production security boundary. baseline/lp_conditional.py assumes oracle layers, u and labeled activations. Legacy weightlab.py remains only for smoke testing; qwen_eval.py is an unverified Transformers pilot, NOT original MIT inference.
