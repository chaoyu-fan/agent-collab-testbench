# Agent Collaboration Ablation Testbench

一个用于系统性评估「多智能体协作」在复杂任务执行各阶段中是否有效的实验框架。

## 动机

在 SWE-bench 等复杂任务中，AI Agent 系统面临关键决策：**是否应该让多个 Agent 协作？在哪个阶段协作？** 本框架通过消融实验（Ablation Study）量化回答这个问题——在每个协作阶段独立开关协作，对比任务完成效果。

## 核心概念

### 四个协作阶段

| 阶段 | 开启（ON） | 关闭（OFF） |
|------|-----------|------------|
| **SPAWN_SUBAGENTS** | 将任务分解为子任务，生成专家子 Agent | 单个 Agent 独立完成全部工作 |
| **PARALLEL_EXECUTION** | 子 Agent 并发执行（独立探索） | 子 Agent 串行执行（后者可看到前者输出） |
| **SUBAGENT_RESULT_PARTIAL** | 通过共享黑板广播中间结果（交叉授粉） | Agent 之间完全隔离 |
| **SUBAGENT_RESULT_COMPLETE** | 由合成 Agent 审查和合并所有输出 | 直接拼接各 Agent 输出 |

### 有效配置数

每个阶段可以 ON/OFF，理论上 2⁴ = 16 种组合。但当 `SPAWN_SUBAGENTS=OFF` 时，其余三个阶段无意义（强制为 OFF）。因此实际有效配置为 **9 种**：

```
0000  Solo（单 Agent，无协作）
1000  仅分解子任务
1001  分解 + 合成审查
1010  分解 + 中间结果共享
1011  分解 + 共享 + 合成
1100  分解 + 并行执行
1101  分解 + 并行 + 合成
1110  分解 + 并行 + 共享
1111  全协作（分解 + 并行 + 共享 + 合成）
```

## 安装

### 环境要求

- Python 3.10+
- pip

### 步骤

```bash
# 克隆仓库
git clone https://github.com/JakeFn123/agent-collab-testbench.git
cd agent-collab-testbench

# 创建虚拟环境并安装依赖
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# 或者直接安装依赖运行（不安装包）
pip install pytest pytest-asyncio pyyaml anthropic
```

## 快速开始

### 1. 查看所有策略配置

```bash
PYTHONPATH=. python -m testbench list-strategies
```

输出：
```
Config   Description
------   -----------
0000     Solo (single agent, no collaboration)
1000     Spawn: ON | Parallel: OFF | Sharing: OFF | Synthesis: OFF
1001     Spawn: ON | Parallel: OFF | Sharing: OFF | Synthesis: ON
...
1111     Spawn: ON | Parallel: ON | Sharing: ON | Synthesis: ON
```

### 2. 使用 Mock LLM 运行实验（不消耗 API 额度）

```bash
PYTHONPATH=. python -m testbench run \
    --tasks tasks/example_suite.json \
    --strategies all \
    --runs 3 \
    --mock \
    --output output/mock_experiment
```

### 3. 使用真实 Claude API 运行

```bash
export ANTHROPIC_API_KEY=sk-ant-...

PYTHONPATH=. python -m testbench run \
    --tasks tasks/example_suite.json \
    --strategies all \
    --runs 3 \
    --output output/real_experiment
```

### 4. 分析已有结果

```bash
# 输出 Markdown 报告到终端
PYTHONPATH=. python -m testbench analyze \
    --input output/mock_experiment \
    --format markdown

# 导出为 CSV（可在 Excel 或 Pandas 中进一步分析）
PYTHONPATH=. python -m testbench analyze \
    --input output/mock_experiment \
    --format csv

# 导出为 JSON（供程序消费）
PYTHONPATH=. python -m testbench analyze \
    --input output/mock_experiment \
    --format json
```

## CLI 参考

### `testbench run`

运行消融实验。

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--tasks` | 任务集 JSON 文件路径（必填） | - |
| `--strategies` | 策略配置 ID（逗号分隔）或 `all` | `all` |
| `--runs` | 每个 (任务, 策略) 组合的运行次数 | `3` |
| `--output` | 输出目录 | `output` |
| `--mock` | 使用 Mock LLM（不消耗 API） | `false` |
| `--model` | Claude 模型 ID | `claude-sonnet-4-20250514` |
| `--threshold` | 相似度通过阈值 | `0.3` |
| `-v` | 详细日志 | `false` |

**示例：只测试 Solo vs 全协作**
```bash
PYTHONPATH=. python -m testbench run \
    --tasks tasks/example_suite.json \
    --strategies 0000,1111 \
    --runs 5 \
    --mock
```

### `testbench analyze`

分析已保存的实验结果。

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--input` | 包含 `results.json` 的目录（必填） | - |
| `--format` | 输出格式：`markdown`、`csv`、`json` | `markdown` |
| `--output` | 报告保存目录 | 同 `--input` |

### `testbench list-strategies`

打印所有 9 种有效策略配置及描述。无额外参数。

## 自定义任务集

任务集为 JSON 文件，格式如下：

```json
{
  "name": "my_task_suite",
  "tasks": [
    {
      "task_id": "unique_id",
      "description": "问题描述，Agent 会基于此描述来解决问题",
      "ground_truth": "期望的正确输出/方案，用于评估 Agent 的输出",
      "repository": null,
      "base_commit": null,
      "test_command": null,
      "metadata": {
        "difficulty": "easy",
        "category": "bug_fix"
      }
    }
  ]
}
```

### 字段说明

| 字段 | 必填 | 说明 |
|------|------|------|
| `task_id` | 是 | 任务唯一标识 |
| `description` | 是 | 问题描述（输入给 Agent 的提示） |
| `ground_truth` | 是 | 期望的正确答案（用于评估相似度） |
| `repository` | 否 | 关联的代码仓库 |
| `base_commit` | 否 | 基准 commit |
| `test_command` | 否 | 验证命令（使用 FunctionalEvaluator 时会执行） |
| `metadata` | 否 | 自定义元数据 |

## 输出与报告

每次实验运行后，`output/` 目录结构如下：

```
output/
├── traces/                              # 每次运行的详细执行轨迹
│   ├── bug_fix_001_0000_0.json         # task_id + strategy + run_index
│   ├── bug_fix_001_1111_0.json
│   └── ...
├── results.json                         # 所有评估结果的汇总
├── report.md                            # Markdown 格式报告
├── report.csv                           # CSV 格式（可导入 Excel/Pandas）
└── report.json                          # JSON 格式（程序可读）
```

### 消融分析表（核心输出）

```
| Phase                    | Pass Rate (ON) | Pass Rate (OFF) | Delta  |
|--------------------------|---------------|-----------------|--------|
| SPAWN_SUBAGENTS          | 72%           | 58%             | +14%   |
| PARALLEL_EXECUTION       | 68%           | 71%             | -3%    |
| SUBAGENT_RESULT_PARTIAL  | 75%           | 65%             | +10%   |
| SUBAGENT_RESULT_COMPLETE | 78%           | 62%             | +16%   |
```

**解读方式**：
- **Delta > 0**：该阶段的协作有正面效果
- **Delta < 0**：该阶段的协作反而有害（增加了噪声/开销）
- **Delta ≈ 0**：该阶段的协作效果不显著

## 架构概览

```
testbench/
├── models/           # 数据模型
│   ├── strategy.py   #   CollaborationStrategy —— 4-bit 协作配置
│   ├── task.py       #   Task, TaskSuite —— 任务定义
│   ├── trace.py      #   ExecutionTrace, TraceEvent —— 执行轨迹
│   ├── result.py     #   EvaluationResult —— 评估结果
│   └── agent.py      #   AgentIdentity, AgentMessage —— Agent 身份与消息
├── agents/           # Agent 实现
│   ├── base.py       #   BaseAgent 抽象基类
│   ├── solo_agent.py #   SoloAgent —— 单 Agent 独立完成任务
│   ├── decomposer.py #   DecomposerAgent —— 任务分解
│   ├── specialist.py #   SpecialistAgent —— 子任务专家
│   ├── synthesizer.py#   SynthesizerAgent —— 结果合成
│   └── prompts.py    #   各角色的 Prompt 模板
├── orchestrator/     # 执行编排
│   ├── orchestrator.py  # 主编排器 —— 实现 4 阶段决策流
│   ├── lifecycle.py     # Agent 生命周期管理（spawn/join/cancel/timeout）
│   ├── shared_state.py  # SharedBlackboard —— 中间结果共享
│   └── message_bus.py   # MessageBus —— Agent 间消息传递
├── llm/              # LLM 后端
│   ├── protocol.py   #   LLMClient 协议（接口定义）
│   ├── mock.py       #   MockLLMClient —— 确定性 Mock（测试用）
│   └── client.py     #   AnthropicLLMClient —— 真实 Claude API
├── evaluation/       # 评估层
│   ├── evaluator.py  #   Evaluator 抽象基类
│   ├── patch_match.py#   PatchMatchEvaluator —— 文本相似度评估
│   ├── functional.py #   FunctionalEvaluator —— 运行测试命令评估
│   └── metrics.py    #   StrategyMetrics —— 聚合指标计算
├── analysis/         # 分析层
│   ├── comparator.py #   Comparator —— 跨策略对比与消融分析
│   ├── statistics.py #   统计检验（Cohen's d 效应量）
│   └── report.py     #   报告生成（Markdown / CSV / JSON）
├── loaders/          # 数据加载
│   ├── json_loader.py#   JSON 格式任务集加载
│   └── yaml_loader.py#   YAML 格式任务集加载
├── cli.py            # CLI 入口
└── __main__.py       # python -m testbench 入口
```

### 执行流程

对于每个 `(任务, 策略, 运行次数)` 三元组：

```
1. INIT：加载任务，创建 ExecutionTrace

2. SPAWN_SUBAGENTS 决策
   ├─ ON:  DecomposerAgent 分解任务 → 生成 N 个 SpecialistAgent
   └─ OFF: 创建 SoloAgent → 跳到步骤 6

3. PARALLEL_EXECUTION 决策
   ├─ ON:  asyncio.gather() 并发执行所有 Specialist
   └─ OFF: 串行执行，后续 Agent 可看到前序 Agent 的输出

4. SUBAGENT_RESULT_PARTIAL 决策（仅 PARALLEL=ON 时有意义）
   ├─ ON:  SharedBlackboard 激活 —— Agent 发布进度，读取他人进展
   └─ OFF: Agent 之间完全隔离

5. SUBAGENT_RESULT_COMPLETE 决策
   ├─ ON:  SynthesizerAgent 合并审查所有输出
   └─ OFF: 直接拼接各 Agent 输出

6. EVALUATE：对比 final_output 与 ground_truth → EvaluationResult

7. RECORD：序列化 ExecutionTrace 为 JSON
```

## 运行测试

```bash
source .venv/bin/activate
PYTHONPATH=. python -m pytest tests/ -v
```

测试覆盖：
- `test_strategy.py` — 策略配置逻辑（约束强制、序列化、枚举）
- `test_agents.py` — Mock LLM 和 Agent 模型
- `test_message_bus.py` — SharedBlackboard 和 MessageBus 的异步安全性
- `test_orchestrator.py` — Solo 和协作路径的编排逻辑
- `test_evaluation.py` — 评估器（相似度计算、端到端评估）
- `test_e2e_mock.py` — 全流程集成测试（9 策略 × 多任务 × 分析报告）

## 扩展指南

### 添加新的协作阶段

1. 在 `models/strategy.py` 的 `CollaborationStrategy` 中添加新的布尔字段
2. 在 `orchestrator/orchestrator.py` 中添加对应的决策分支
3. 更新 `analysis/comparator.py` 中的 `PHASE_NAMES`

### 添加新的评估器

1. 继承 `evaluation/evaluator.py` 中的 `Evaluator` 基类
2. 实现 `evaluate(task, run_result) -> EvaluationResult` 方法
3. 在 CLI 中注册新评估器

### 使用其他 LLM 后端

1. 实现 `llm/protocol.py` 中的 `LLMClient` 协议
2. 在 CLI 中添加对应的初始化逻辑

## License

MIT
