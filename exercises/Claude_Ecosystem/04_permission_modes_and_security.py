"""
Exercises for Lesson 04: Permission Modes and Security
Topic: Claude_Ecosystem

Solutions to practice problems from the lesson.
"""

import fnmatch
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any


# === Exercise 1: Permission Mode Simulator ===
# Problem: Implement the five Claude Code permission modes and determine
#   whether a tool call is allowed, denied, or requires user approval.

class PermissionMode(Enum):
    DEFAULT = "default"       # Ask for write/execute, auto-allow read
    AUTO_ACCEPT = "auto"      # Accept all with brief delay
    PLAN = "plan"             # Read-only, no edits or execution
    DONT_ASK = "dont_ask"     # Accept common, ask destructive
    BYPASS = "bypass"         # Accept all (Docker only)


class ToolAction(Enum):
    ALLOWED = "allowed"
    DENIED = "denied"
    ASK_USER = "ask_user"


READ_TOOLS = {"Read", "Glob", "Grep"}
WRITE_TOOLS = {"Write", "Edit", "NotebookEdit"}
EXEC_TOOLS = {"Bash"}

DESTRUCTIVE_PATTERNS = [
    r"rm\s+-rf",
    r"git\s+push\s+--force",
    r"git\s+reset\s+--hard",
    r"DROP\s+TABLE",
]


def check_permission(
    mode: PermissionMode,
    tool_name: str,
    command: str = "",
) -> ToolAction:
    """Determine whether a tool call is allowed under the given mode."""
    if mode == PermissionMode.BYPASS:
        return ToolAction.ALLOWED

    if mode == PermissionMode.PLAN:
        if tool_name in READ_TOOLS:
            return ToolAction.ALLOWED
        return ToolAction.DENIED

    if mode == PermissionMode.AUTO_ACCEPT:
        return ToolAction.ALLOWED

    is_destructive = any(re.search(p, command) for p in DESTRUCTIVE_PATTERNS)

    if mode == PermissionMode.DONT_ASK:
        if is_destructive:
            return ToolAction.ASK_USER
        return ToolAction.ALLOWED

    # DEFAULT mode
    if tool_name in READ_TOOLS:
        return ToolAction.ALLOWED
    if is_destructive:
        return ToolAction.ASK_USER
    if tool_name in WRITE_TOOLS or tool_name in EXEC_TOOLS:
        return ToolAction.ASK_USER
    return ToolAction.ASK_USER


def exercise_1():
    """Demonstrate permission mode behavior."""
    test_cases = [
        ("Read", ""),
        ("Edit", ""),
        ("Bash", "python -m pytest"),
        ("Bash", "rm -rf /tmp/old"),
        ("Bash", "git push --force"),
    ]
    for mode in PermissionMode:
        print(f"  {mode.value}:")
        for tool, cmd in test_cases:
            result = check_permission(mode, tool, cmd)
            label = f"{tool}({cmd})" if cmd else tool
            print(f"    {label:<30} → {result.value}")
        print()


# === Exercise 2: Allow/Deny Rule Matcher ===
# Problem: Implement glob-based allow/deny rules for Bash commands.

@dataclass
class PermissionRule:
    """An allow or deny rule for tool permissions."""
    action: str  # "allow" or "deny"
    tool: str
    pattern: str  # glob pattern for command matching


def match_rules(
    rules: list[PermissionRule],
    tool_name: str,
    command: str = "",
) -> ToolAction:
    """Match a tool call against allow/deny rules.

    Rules are evaluated in order. First match wins.
    Deny rules take priority over allow rules at the same specificity.
    """
    deny_matched = False
    allow_matched = False

    for rule in rules:
        if rule.tool != tool_name:
            continue

        if not rule.pattern or fnmatch.fnmatch(command, rule.pattern):
            if rule.action == "deny":
                deny_matched = True
                break
            elif rule.action == "allow":
                allow_matched = True

    if deny_matched:
        return ToolAction.DENIED
    if allow_matched:
        return ToolAction.ALLOWED
    return ToolAction.ASK_USER


def exercise_2():
    """Demonstrate allow/deny rule matching."""
    rules = [
        PermissionRule("deny", "Bash", "rm -rf *"),
        PermissionRule("allow", "Bash", "npm test*"),
        PermissionRule("allow", "Bash", "python -m pytest*"),
        PermissionRule("allow", "Edit", ""),
        PermissionRule("deny", "Bash", "git push --force*"),
    ]
    test_commands = [
        ("Bash", "npm test"),
        ("Bash", "python -m pytest tests/"),
        ("Bash", "rm -rf /important"),
        ("Bash", "git push --force origin main"),
        ("Bash", "curl https://example.com"),
        ("Edit", ""),
    ]
    for tool, cmd in test_commands:
        result = match_rules(rules, tool, cmd)
        label = f"{tool}({cmd})" if cmd else tool
        print(f"  {label:<40} → {result.value}")


# === Exercise 3: Security Policy Checker ===
# Problem: Analyze a set of permission rules for potential security issues.

def audit_security_policy(
    mode: PermissionMode,
    rules: list[PermissionRule],
) -> list[dict[str, str]]:
    """Audit a permission configuration for security issues."""
    findings: list[dict[str, str]] = []

    if mode == PermissionMode.BYPASS:
        findings.append({
            "severity": "critical",
            "message": "Bypass mode disables all permission checks. "
                       "Only use inside a container.",
        })

    if mode == PermissionMode.AUTO_ACCEPT:
        findings.append({
            "severity": "high",
            "message": "Auto-accept mode allows all tool calls without review.",
        })

    allow_all_bash = any(
        r.action == "allow" and r.tool == "Bash" and not r.pattern
        for r in rules
    )
    if allow_all_bash:
        findings.append({
            "severity": "high",
            "message": "Blanket allow for all Bash commands. "
                       "Consider restricting to specific patterns.",
        })

    has_deny_destructive = any(
        r.action == "deny" and r.tool == "Bash"
        and any(d in r.pattern for d in ["rm", "force", "reset"])
        for r in rules if r.pattern
    )
    if not has_deny_destructive and mode not in (
        PermissionMode.PLAN, PermissionMode.BYPASS
    ):
        findings.append({
            "severity": "medium",
            "message": "No explicit deny rules for destructive commands.",
        })

    if not findings:
        findings.append({
            "severity": "ok",
            "message": "Policy looks reasonable.",
        })

    return findings


def exercise_3():
    """Demonstrate security policy auditing."""
    configs = [
        ("Bypass + no rules", PermissionMode.BYPASS, []),
        ("Default + safe rules", PermissionMode.DEFAULT, [
            PermissionRule("allow", "Bash", "npm test*"),
            PermissionRule("deny", "Bash", "rm -rf*"),
        ]),
        ("Default + blanket allow", PermissionMode.DEFAULT, [
            PermissionRule("allow", "Bash", ""),
        ]),
    ]
    for label, mode, rules in configs:
        print(f"  {label}:")
        for f in audit_security_policy(mode, rules):
            print(f"    [{f['severity']}] {f['message']}")
        print()


if __name__ == "__main__":
    print("=== Exercise 1: Permission Modes ===")
    exercise_1()

    print("=== Exercise 2: Allow/Deny Rules ===")
    exercise_2()

    print("\n=== Exercise 3: Security Audit ===")
    exercise_3()

    print("All exercises completed!")
