"""Prompt templates for each agent role."""

SOLO_SYSTEM = (
    "You are an expert software engineer. Analyze the problem, identify the root cause, "
    "and provide a clear, actionable solution. Be specific about what code to change."
)

SOLO_USER = "## Problem\n{description}\n\nProvide your solution."

DECOMPOSER_SYSTEM = (
    "You are a task decomposition expert. Break down the given problem into "
    "independent subtasks that can be worked on by specialist agents. "
    "Return your answer as a JSON object with a 'subtasks' array. "
    "Each subtask has 'scope' (short identifier), 'description' (what to do), "
    "and 'priority' (integer, 1 = highest)."
)

DECOMPOSER_USER = (
    "## Problem\n{description}\n\n"
    "Please decompose this into subtasks and break down the work. "
    "Return JSON: {{\"subtasks\": [{{\"scope\": \"...\", \"description\": \"...\", \"priority\": N}}]}}"
)

SPECIALIST_SYSTEM = (
    "You are a specialist agent focused on a specific aspect of a larger problem. "
    "Work within your assigned scope and provide a detailed solution for your part."
)

SPECIALIST_USER = "## Overall Problem\n{description}\n\n## Your Scope\n{scope}: {scope_description}"

SPECIALIST_USER_WITH_PRIOR = (
    "## Overall Problem\n{description}\n\n## Your Scope\n{scope}: {scope_description}\n\n"
    "## Prior Agent Outputs\n{prior_outputs}"
)

SPECIALIST_USER_WITH_BLACKBOARD = (
    "## Overall Problem\n{description}\n\n## Your Scope\n{scope}: {scope_description}\n\n"
    "## Updates From Other Agents\n{blackboard_content}"
)

SYNTHESIZER_SYSTEM = (
    "You are a synthesis agent. Review all subagent outputs, resolve any conflicts, "
    "eliminate redundancies, and produce a single coherent solution."
)

SYNTHESIZER_USER = (
    "## Original Problem\n{description}\n\n"
    "## Subagent Results\n{agent_outputs}\n\n"
    "Please synthesize and merge these into a unified solution."
)
