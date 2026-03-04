"""
Claude Agent Teams: Orchestration Example

Demonstrates a multi-agent team pattern where a lead agent
coordinates specialized sub-agents for a complex task.

This is a conceptual simulation — in practice, Claude Code's
Agent tool handles orchestration automatically.

Requirements:
    No external dependencies (pure simulation)
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any


# --- Team Configuration ---

@dataclass
class AgentConfig:
    """Configuration for a team member agent."""
    name: str
    role: str
    system_prompt: str
    allowed_tools: list[str] = field(default_factory=list)


TEAM = [
    AgentConfig(
        "explorer", "researcher",
        "Search the codebase thoroughly. Report file paths, "
        "function signatures, and architectural patterns.",
        allowed_tools=["Read", "Glob", "Grep"],
    ),
    AgentConfig(
        "planner", "architect",
        "Design an implementation plan based on exploration results. "
        "Consider edge cases and backward compatibility.",
        allowed_tools=["Read", "Glob", "Grep"],
    ),
    AgentConfig(
        "implementer", "developer",
        "Implement the planned changes. Write clean, tested code.",
        allowed_tools=["Read", "Write", "Edit", "Bash"],
    ),
    AgentConfig(
        "reviewer", "qa",
        "Review the implementation for bugs, security issues, "
        "and code quality. Run tests.",
        allowed_tools=["Read", "Grep", "Bash"],
    ),
]


# --- Orchestration Logic ---

@dataclass
class AgentResult:
    agent_name: str
    output: str
    files_touched: list[str] = field(default_factory=list)


async def run_agent(config: AgentConfig, task: str) -> AgentResult:
    """Simulate running a single agent."""
    await asyncio.sleep(0.01)  # simulate work
    return AgentResult(
        agent_name=config.name,
        output=f"[{config.role}] Completed: {task}",
        files_touched=["src/example.py"],
    )


async def orchestrate(task: str) -> list[AgentResult]:
    """Orchestrate the team through sequential phases."""
    results: list[AgentResult] = []

    # Phase 1: Explore (can run multiple explorers in parallel)
    explorer = TEAM[0]
    result = await run_agent(explorer, f"Explore context for: {task}")
    results.append(result)
    print(f"  Phase 1: {result.output}")

    # Phase 2: Plan
    planner = TEAM[1]
    result = await run_agent(planner, f"Plan approach for: {task}")
    results.append(result)
    print(f"  Phase 2: {result.output}")

    # Phase 3: Implement
    implementer = TEAM[2]
    result = await run_agent(implementer, f"Implement: {task}")
    results.append(result)
    print(f"  Phase 3: {result.output}")

    # Phase 4: Review
    reviewer = TEAM[3]
    result = await run_agent(reviewer, f"Review changes for: {task}")
    results.append(result)
    print(f"  Phase 4: {result.output}")

    return results


# --- Main ---

if __name__ == "__main__":
    print("Agent Team Orchestration Example")
    print("=" * 50)

    print("\nTeam members:")
    for agent in TEAM:
        print(f"  {agent.name} ({agent.role}): "
              f"tools={agent.allowed_tools}")

    print("\nRunning orchestration...")
    results = asyncio.run(orchestrate("Add JWT authentication"))

    print(f"\nCompleted: {len(results)} phases")
