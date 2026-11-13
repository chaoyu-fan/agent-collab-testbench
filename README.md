# WeightSpace Agent Lab

**Autonomous Weight-Space Repair & Research-Intelligence Benchmark**  
Status: v0.1 (research prototype, 2026-11-13)

目标：让 Coding Agent 在受控环境中诊断模型缺陷、自主提出并验证干预方案、直接修改权重；衡量最终效果，也衡量算法洞察相对于暴力试错的计算效率。

## Tracks

| Track | 任务 | 状态 |
|---|---|---|
| A: Blind Repair | CPU NumPy 神经网络隐藏故障修复（稀疏编辑） | 生成器、合法性检查、隐藏集评分和 6 项单元测试已在本地通过 |
| B: Real WSP | Qwen/Qwen3.5-0.8B 数学任务权重编辑 | HF Transformers 评估脚本已经编写，但尚未 GPU 验证 |

## Quick start (CPU)

~~~bash
python -m pip install -r requirements.txt
python weightlab.py prepare --out runs/trial01
python weightlab.py demo-patch --model runs/trial01/participant/model.npz --out runs/trial01/noop.npz
python weightlab.py evaluate --participant runs/trial01/participant --organizer runs/trial01/organizer --patch runs/trial01/noop.npz --output runs/trial01/noop_result.json
PYTHONPATH=. python -m unittest discover -s tests -v
~~~

no-op 是管线验证，不是性能改进。正式盲测只能将 participant 文件夹交给 Agent；organizer 必须隔离到另一台主机或受控环境，且测试阶段只能在冻结提交后执行。CLI 可使用私有随机 seed；绝不能公开 organizer 的隐藏数据。

## Agent instructions

See [AGENT_PROMPT.md](AGENT_PROMPT.md). The agent receives model.npz, dev.jsonl, rules.json and submits final_patch.npz with delta_w2. No gradient descent, no LoRA, no RL. At most 24 changed entries and norm <= 8.0; the organizer grades on a hidden test.

## Qwen pilot (GPU, optional)

~~~bash
python -m pip install -r requirements-qwen.txt
python qwen_eval.py --model Qwen/Qwen3.5-0.8B --dataset examples/math_pilot.jsonl --out outputs/base.json --max-new-tokens 2048
# run an identical command with a separately saved, edited checkpoint
python qwen_eval.py --model /path/to/edited-checkpoint --dataset examples/math_pilot.jsonl --out outputs/edited.json --max-new-tokens 2048
~~~

The six-example dataset is a plumbing test, NOT a benchmark. Current Qwen script has not been run on a GPU here. Its Transformers sampling is NOT identical to the MIT vLLM experiment, especially around the presence penalty and long generation budget; scores must never be compared across protocols.

## Research questions and metrics

Measure hidden-set accuracy improvement, completion/truncation, output tokens, wall/GPU time, verified forward/tool calls, changed-parameter count, generalization and falsifiable mechanism hypotheses. The key result is **accuracy-gain-vs-compute** under matched budgets (e.g. 10/25/50 passes), not just a peak score. Code-level CPU success does not establish an Agent IQ score. Read [docs/PROTOCOL.md](docs/PROTOCOL.md), [docs/ROADMAP.md](docs/ROADMAP.md), [docs/REFERENCES.md](docs/REFERENCES.md).

## Isolation warning

The local demo holds participant and organizer folders on one machine. That is not secure isolation. v0.1 does not cryptographically enforce the Agent's self-reported experiment count; the organizer must later implement a trusted gateway for formal leaderboard claims.

License: MIT for original code and docs; external models/data retain their licenses.
