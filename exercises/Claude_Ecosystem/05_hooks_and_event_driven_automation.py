"""
Exercises for Lesson 05: Hooks and Event-Driven Automation
Topic: Claude_Ecosystem

Solutions to practice problems from the lesson.
"""

from __future__ import annotations

import json
import fnmatch
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# === Exercise 1: Hook Configuration Parser ===
# Problem: Parse and validate hook JSON configurations.

class HookType(Enum):
    PRE_TOOL_USE = "PreToolUse"
    POST_TOOL_USE = "PostToolUse"
    NOTIFICATION = "Notification"
    STOP = "Stop"


@dataclass
class HookMatcher:
    """Matcher for filtering which tool calls trigger a hook."""
    tool_name: str | None = None
    file_pattern: str | None = None
    command_pattern: str | None = None


@dataclass
class HookConfig:
    """Parsed hook configuration."""
    hook_type: HookType
    command: str
    matchers: list[HookMatcher] = field(default_factory=list)
    timeout_ms: int = 60_000


def parse_hook_config(raw: dict[str, Any]) -> HookConfig | str:
    """Parse a raw JSON hook configuration.

    Returns HookConfig on success, error string on failure.
    """
    try:
        hook_type = HookType(raw["type"])
    except (KeyError, ValueError):
        return f"Invalid or missing hook type: {raw.get('type')}"

    command = raw.get("command", "")
    if not command:
        return "Hook must have a non-empty command"

    matchers = []
    for m in raw.get("matchers", []):
        matchers.append(HookMatcher(
            tool_name=m.get("tool_name"),
            file_pattern=m.get("file_pattern"),
            command_pattern=m.get("command_pattern"),
        ))

    return HookConfig(
        hook_type=hook_type,
        command=command,
        matchers=matchers,
        timeout_ms=raw.get("timeout_ms", 60_000),
    )


def exercise_1():
    """Demonstrate hook configuration parsing."""
    configs = [
        {"type": "PostToolUse", "command": "prettier --write $CLAUDE_FILE_PATH",
         "matchers": [{"tool_name": "Edit", "file_pattern": "*.ts"}]},
        {"type": "PreToolUse", "command": "echo 'blocked'",
         "matchers": [{"tool_name": "Bash",
                       "command_pattern": "rm -rf*"}]},
        {"type": "InvalidType", "command": "test"},
        {"type": "Notification", "command": ""},
    ]
    for raw in configs:
        result = parse_hook_config(raw)
        if isinstance(result, str):
            print(f"  ERROR: {result}")
        else:
            matchers_desc = ", ".join(
                m.tool_name or m.file_pattern or "any" for m in result.matchers
            ) or "none"
            print(f"  {result.hook_type.value}: '{result.command}' "
                  f"(matchers: {matchers_desc})")


# === Exercise 2: Event Dispatcher ===
# Problem: Simulate the hook event dispatch system. When a tool is used,
#   check all registered hooks and fire matching ones.

@dataclass
class ToolEvent:
    """An event triggered by a Claude Code tool call."""
    tool_name: str
    file_path: str | None = None
    command: str | None = None


@dataclass
class HookResult:
    """Result of executing a hook."""
    hook_command: str
    matched_by: str
    output: str


def dispatch_hooks(
    event: ToolEvent,
    hooks: list[HookConfig],
    phase: HookType,
) -> list[HookResult]:
    """Find and execute matching hooks for a tool event."""
    results: list[HookResult] = []

    for hook in hooks:
        if hook.hook_type != phase:
            continue

        if not hook.matchers:
            results.append(HookResult(
                hook_command=hook.command,
                matched_by="no matchers (matches all)",
                output=f"[simulated] {hook.command}",
            ))
            continue

        for matcher in hook.matchers:
            matched = True
            match_reason = []

            if matcher.tool_name and matcher.tool_name != event.tool_name:
                matched = False
            elif matcher.tool_name:
                match_reason.append(f"tool={matcher.tool_name}")

            if matched and matcher.file_pattern and event.file_path:
                if not fnmatch.fnmatch(event.file_path, matcher.file_pattern):
                    matched = False
                else:
                    match_reason.append(f"file={matcher.file_pattern}")

            if matched and matcher.command_pattern and event.command:
                if not fnmatch.fnmatch(event.command, matcher.command_pattern):
                    matched = False
                else:
                    match_reason.append(f"cmd={matcher.command_pattern}")

            if matched:
                results.append(HookResult(
                    hook_command=hook.command,
                    matched_by=", ".join(match_reason) or "default",
                    output=f"[simulated] {hook.command}",
                ))
                break

    return results


def exercise_2():
    """Simulate hook event dispatching."""
    hooks = [
        HookConfig(HookType.POST_TOOL_USE, "black $CLAUDE_FILE_PATH",
                    [HookMatcher(tool_name="Edit", file_pattern="*.py")]),
        HookConfig(HookType.POST_TOOL_USE, "prettier --write $CLAUDE_FILE_PATH",
                    [HookMatcher(tool_name="Edit", file_pattern="*.ts")]),
        HookConfig(HookType.PRE_TOOL_USE, "echo 'BLOCKED'",
                    [HookMatcher(tool_name="Bash", command_pattern="rm*")]),
    ]
    events = [
        ToolEvent("Edit", file_path="src/main.py"),
        ToolEvent("Edit", file_path="src/app.ts"),
        ToolEvent("Bash", command="rm -rf /tmp"),
        ToolEvent("Read", file_path="README.md"),
    ]
    for event in events:
        label = f"{event.tool_name}({event.file_path or event.command})"
        for phase in [HookType.PRE_TOOL_USE, HookType.POST_TOOL_USE]:
            results = dispatch_hooks(event, hooks, phase)
            for r in results:
                print(f"  {label} [{phase.value}] → {r.hook_command} "
                      f"(matched: {r.matched_by})")
        if not any(dispatch_hooks(event, hooks, p)
                   for p in [HookType.PRE_TOOL_USE, HookType.POST_TOOL_USE]):
            print(f"  {label} → no hooks matched")


# === Exercise 3: Hook vs CLAUDE.md Decision Guide ===
# Problem: Given a requirement, determine whether it should be implemented
#   as a hook (deterministic) or a CLAUDE.md instruction (suggestive).

@dataclass
class AutomationRequirement:
    """A requirement for automating behavior in Claude Code."""
    description: str
    needs_determinism: bool
    needs_shell_execution: bool
    needs_context_awareness: bool


def recommend_mechanism(req: AutomationRequirement) -> str:
    """Recommend hook vs CLAUDE.md based on requirement characteristics."""
    if req.needs_determinism and req.needs_shell_execution:
        return "Hook"
    elif req.needs_context_awareness and not req.needs_determinism:
        return "CLAUDE.md"
    elif req.needs_determinism:
        return "Hook"
    else:
        return "CLAUDE.md"


def exercise_3():
    """Demonstrate hook vs CLAUDE.md decision making."""
    requirements = [
        AutomationRequirement(
            "Auto-format Python files after every edit",
            needs_determinism=True, needs_shell_execution=True,
            needs_context_awareness=False),
        AutomationRequirement(
            "Use PEP 8 naming conventions in new code",
            needs_determinism=False, needs_shell_execution=False,
            needs_context_awareness=True),
        AutomationRequirement(
            "Block rm -rf commands",
            needs_determinism=True, needs_shell_execution=False,
            needs_context_awareness=False),
        AutomationRequirement(
            "Prefer functional style over OOP when possible",
            needs_determinism=False, needs_shell_execution=False,
            needs_context_awareness=True),
    ]
    for req in requirements:
        mechanism = recommend_mechanism(req)
        print(f"  {req.description}")
        print(f"    → {mechanism}\n")


if __name__ == "__main__":
    print("=== Exercise 1: Hook Config Parser ===")
    exercise_1()

    print("\n=== Exercise 2: Event Dispatcher ===")
    exercise_2()

    print("\n=== Exercise 3: Hook vs CLAUDE.md ===")
    exercise_3()

    print("\nAll exercises completed!")
