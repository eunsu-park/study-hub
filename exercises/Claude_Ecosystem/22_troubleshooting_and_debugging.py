"""
Exercises for Lesson 22: Troubleshooting and Debugging
Topic: Claude_Ecosystem

Solutions to practice problems from the lesson.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# === Exercise 1: Error Classifier ===
# Problem: Classify Claude Code errors by category and suggest
#   resolution steps.

class ErrorCategory(Enum):
    PERMISSION = "permission"
    CONTEXT = "context_limit"
    HOOK = "hook_failure"
    MCP = "mcp_error"
    API = "api_error"
    NETWORK = "network"
    CONFIG = "configuration"
    UNKNOWN = "unknown"


@dataclass
class ErrorInfo:
    """Classified error with resolution guidance."""
    category: ErrorCategory
    message: str
    resolution_steps: list[str]
    severity: str  # "low", "medium", "high", "critical"


ERROR_PATTERNS: list[tuple[str, ErrorCategory, str, list[str]]] = [
    (r"permission denied|not allowed|requires approval",
     ErrorCategory.PERMISSION, "medium",
     ["Check permission mode (current vs needed)",
      "Add allow rule for the specific tool/command",
      "Consider using a less restrictive mode for this task"]),
    (r"context.*limit|too (long|many tokens)|truncat",
     ErrorCategory.CONTEXT, "high",
     ["Run /compact to compress conversation history",
      "Start a new session with /clear",
      "Break the task into smaller parts",
      "Reduce system prompt length in CLAUDE.md"]),
    (r"hook.*fail|hook.*error|hook.*exit",
     ErrorCategory.HOOK, "medium",
     ["Check hook command syntax in settings.json",
      "Test the hook command manually in terminal",
      "Check file path patterns in matchers",
      "Review hook timeout settings"]),
    (r"mcp.*error|server.*disconnect|transport.*fail",
     ErrorCategory.MCP, "high",
     ["Verify MCP server is running",
      "Check mcpServers config in settings.json",
      "Test server command manually",
      "Check environment variables (API keys, URLs)"]),
    (r"api.*error|rate.limit|429|503|overloaded",
     ErrorCategory.API, "medium",
     ["Wait and retry (API may be temporarily overloaded)",
      "Check API key validity",
      "Review rate limit headers for retry timing",
      "Switch to a less busy model (Haiku)"]),
    (r"connect|timeout|ECONNREFUSED|network",
     ErrorCategory.NETWORK, "high",
     ["Check internet connection",
      "Verify proxy settings if applicable",
      "Check firewall rules",
      "Try again after a brief wait"]),
    (r"config|settings|invalid.*json|parse.*error",
     ErrorCategory.CONFIG, "medium",
     ["Validate JSON syntax in settings files",
      "Check for trailing commas in JSON",
      "Compare with documented settings format",
      "Try resetting to default settings"]),
]


def classify_error(error_message: str) -> ErrorInfo:
    """Classify an error message and provide resolution guidance."""
    msg_lower = error_message.lower()
    for pattern, category, severity, steps in ERROR_PATTERNS:
        if re.search(pattern, msg_lower):
            return ErrorInfo(category, error_message, steps, severity)
    return ErrorInfo(
        ErrorCategory.UNKNOWN, error_message,
        ["Check Claude Code documentation",
         "Search GitHub issues for similar errors",
         "Try restarting Claude Code"],
        "low",
    )


def exercise_1():
    """Demonstrate error classification."""
    errors = [
        "Permission denied: Bash tool requires approval",
        "Context limit exceeded: conversation too long (195K/200K tokens)",
        "Hook failed: prettier command not found (exit code 127)",
        "MCP server disconnected: postgres server timeout",
        "API error: 429 Too Many Requests",
        "Something weird happened with the flux capacitor",
    ]
    for error in errors:
        info = classify_error(error)
        print(f"  [{info.category.value}] ({info.severity}) {error}")
        print(f"    Steps: {info.resolution_steps[0]}")
        print()


# === Exercise 2: Debug Checklist Runner ===
# Problem: Implement a systematic debugging checklist that guides
#   through common troubleshooting steps.

@dataclass
class ChecklistItem:
    """A single item in a debug checklist."""
    step: str
    check_command: str
    expected: str
    category: str


@dataclass
class CheckResult:
    """Result of running a checklist item."""
    item: ChecklistItem
    passed: bool
    actual: str


DEBUG_CHECKLIST = [
    ChecklistItem("Claude Code version", "claude --version",
                  "1.0+", "setup"),
    ChecklistItem("Node.js version", "node --version",
                  "v18+", "setup"),
    ChecklistItem("API key set", "echo $ANTHROPIC_API_KEY | head -c4",
                  "sk-a", "auth"),
    ChecklistItem("CLAUDE.md exists", "test -f CLAUDE.md && echo yes",
                  "yes", "config"),
    ChecklistItem("Settings valid JSON", "python -m json.tool .claude/settings.json",
                  "valid", "config"),
    ChecklistItem("Git repo", "git rev-parse --is-inside-work-tree",
                  "true", "environment"),
    ChecklistItem("Disk space", "df -h . | tail -1 | awk '{print $5}'",
                  "<90%", "environment"),
]


def run_checklist(
    simulated_results: dict[str, bool] | None = None,
) -> list[CheckResult]:
    """Run the debug checklist (simulated for exercise purposes)."""
    if simulated_results is None:
        simulated_results = {}

    results: list[CheckResult] = []
    for item in DEBUG_CHECKLIST:
        passed = simulated_results.get(item.step, True)
        actual = item.expected if passed else "FAILED"
        results.append(CheckResult(item, passed, actual))
    return results


def summarize_checklist(results: list[CheckResult]) -> dict[str, Any]:
    """Summarize checklist results."""
    passed = [r for r in results if r.passed]
    failed = [r for r in results if not r.passed]
    return {
        "total": len(results),
        "passed": len(passed),
        "failed": len(failed),
        "failed_items": [
            {"step": r.item.step, "category": r.item.category,
             "command": r.item.check_command}
            for r in failed
        ],
    }


def exercise_2():
    """Demonstrate debug checklist."""
    results = run_checklist({
        "Claude Code version": True,
        "Node.js version": True,
        "API key set": True,
        "CLAUDE.md exists": False,
        "Settings valid JSON": False,
        "Git repo": True,
        "Disk space": True,
    })
    summary = summarize_checklist(results)
    print(f"  Checklist: {summary['passed']}/{summary['total']} passed")
    if summary["failed_items"]:
        print(f"  Failed:")
        for item in summary["failed_items"]:
            print(f"    [{item['category']}] {item['step']}")
            print(f"      Run: {item['command']}")


# === Exercise 3: Log Analyzer ===
# Problem: Parse and analyze Claude Code session logs to identify
#   patterns, bottlenecks, and issues.

@dataclass
class LogEntry:
    """A parsed log entry."""
    timestamp: str
    level: str      # "info", "warn", "error", "debug"
    component: str  # "tool", "api", "hook", "mcp", "session"
    message: str
    duration_ms: int = 0


def parse_log_entries(raw_logs: list[str]) -> list[LogEntry]:
    """Parse raw log lines into structured entries."""
    entries: list[LogEntry] = []
    pattern = re.compile(
        r"\[(\d{2}:\d{2}:\d{2})\]\s+(\w+)\s+\[(\w+)\]\s+(.+?)(?:\s+\((\d+)ms\))?$"
    )
    for line in raw_logs:
        match = pattern.match(line)
        if match:
            entries.append(LogEntry(
                timestamp=match.group(1),
                level=match.group(2),
                component=match.group(3),
                message=match.group(4),
                duration_ms=int(match.group(5)) if match.group(5) else 0,
            ))
    return entries


def analyze_logs(entries: list[LogEntry]) -> dict[str, Any]:
    """Analyze log entries for patterns and issues."""
    errors = [e for e in entries if e.level == "error"]
    warnings = [e for e in entries if e.level == "warn"]
    slow_ops = [e for e in entries if e.duration_ms > 5000]

    by_component: dict[str, int] = {}
    for e in entries:
        by_component[e.component] = by_component.get(e.component, 0) + 1

    total_duration = sum(e.duration_ms for e in entries if e.duration_ms > 0)

    return {
        "total_entries": len(entries),
        "errors": len(errors),
        "warnings": len(warnings),
        "slow_operations": len(slow_ops),
        "total_duration_ms": total_duration,
        "by_component": by_component,
        "error_messages": [e.message for e in errors],
        "slowest": max(entries, key=lambda e: e.duration_ms).message
        if entries else "N/A",
    }


def exercise_3():
    """Demonstrate log analysis."""
    raw_logs = [
        "[10:15:01] info [session] Session started",
        "[10:15:02] info [tool] Read src/main.py (150ms)",
        "[10:15:03] info [tool] Grep 'TODO' in src/ (320ms)",
        "[10:15:05] info [api] Messages API call (2500ms)",
        "[10:15:08] warn [hook] Hook 'format' slow execution (8000ms)",
        "[10:15:10] info [tool] Edit src/main.py (80ms)",
        "[10:15:11] error [mcp] postgres server: connection refused",
        "[10:15:12] info [api] Messages API call (1800ms)",
        "[10:15:15] info [tool] Bash pytest tests/ (6200ms)",
        "[10:15:16] info [session] Task completed",
    ]
    entries = parse_log_entries(raw_logs)
    analysis = analyze_logs(entries)

    print(f"  Entries: {analysis['total_entries']}")
    print(f"  Errors: {analysis['errors']}, Warnings: {analysis['warnings']}")
    print(f"  Slow operations: {analysis['slow_operations']}")
    print(f"  By component: {analysis['by_component']}")
    print(f"  Slowest: {analysis['slowest']}")
    if analysis["error_messages"]:
        print(f"  Error messages:")
        for msg in analysis["error_messages"]:
            print(f"    - {msg}")


if __name__ == "__main__":
    print("=== Exercise 1: Error Classifier ===")
    exercise_1()

    print("=== Exercise 2: Debug Checklist ===")
    exercise_2()

    print("\n=== Exercise 3: Log Analyzer ===")
    exercise_3()

    print("\nAll exercises completed!")
