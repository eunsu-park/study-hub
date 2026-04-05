"""
Exercises for Lesson 21: Best Practices and Patterns
Topic: Claude_Ecosystem

Solutions to practice problems from the lesson.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


# === Exercise 1: Prompt Pattern Library ===
# Problem: Build a library of effective prompt patterns for Claude Code
#   with templates and usage guidance.

@dataclass
class PromptPattern:
    """A reusable prompt pattern for Claude Code."""
    name: str
    category: str       # "code", "analysis", "workflow", "debug"
    template: str
    variables: list[str]
    description: str
    example: str


PROMPT_PATTERNS = [
    PromptPattern(
        "Targeted Fix",
        "debug",
        "Fix the {error_type} in {file_path}. The error is: {error_message}. "
        "Run tests after fixing.",
        ["error_type", "file_path", "error_message"],
        "Direct Claude to fix a specific error with context.",
        "Fix the TypeError in src/utils.py. The error is: 'NoneType' "
        "has no attribute 'strip'. Run tests after fixing.",
    ),
    PromptPattern(
        "Incremental Refactor",
        "code",
        "Refactor {target} in {file_path} to use {pattern}. "
        "Keep the public API unchanged. Update tests if needed.",
        ["target", "file_path", "pattern"],
        "Safely refactor with clear constraints.",
        "Refactor the UserService class in src/services.py to use "
        "dependency injection. Keep the public API unchanged.",
    ),
    PromptPattern(
        "Code Review",
        "analysis",
        "Review {file_path} for: 1) Security issues 2) Performance "
        "3) Error handling 4) Test coverage gaps. "
        "Suggest fixes with code.",
        ["file_path"],
        "Structured code review with actionable output.",
        "Review src/auth.py for: 1) Security issues ...",
    ),
    PromptPattern(
        "Explore-Then-Act",
        "workflow",
        "First, understand how {feature} works in this codebase. "
        "Read the relevant files. Then {action}.",
        ["feature", "action"],
        "Two-phase approach: explore before modifying.",
        "First, understand how authentication works. Read the relevant "
        "files. Then add JWT token refresh support.",
    ),
]


def fill_template(pattern: PromptPattern,
                  values: dict[str, str]) -> str | None:
    """Fill a prompt template with provided values."""
    missing = [v for v in pattern.variables if v not in values]
    if missing:
        return None
    result = pattern.template
    for var, val in values.items():
        result = result.replace(f"{{{var}}}", val)
    return result


def exercise_1():
    """Demonstrate prompt pattern library."""
    print("  Available patterns:")
    for p in PROMPT_PATTERNS:
        print(f"    [{p.category}] {p.name}: {p.description}")

    print("\n  Filled template:")
    prompt = fill_template(PROMPT_PATTERNS[0], {
        "error_type": "KeyError",
        "file_path": "src/config.py",
        "error_message": "KeyError: 'database_url'",
    })
    print(f"    {prompt}")


# === Exercise 2: Anti-Pattern Detector ===
# Problem: Detect common anti-patterns in Claude Code usage and
#   suggest improvements.

@dataclass
class AntiPattern:
    """A known anti-pattern in Claude Code usage."""
    name: str
    detection_pattern: str  # regex to match in user prompts
    severity: str           # "low", "medium", "high"
    suggestion: str


ANTI_PATTERNS = [
    AntiPattern(
        "Vague request",
        r"^(fix|update|change|improve)\s+(it|this|the code|everything)\.?$",
        "high",
        "Be specific: mention the file, function, or error message.",
    ),
    AntiPattern(
        "No test instruction",
        r"^(add|implement|create|build)\b(?!.*test)",
        "medium",
        "Add 'and write tests' or 'run existing tests after'.",
    ),
    AntiPattern(
        "Overloaded request",
        r"\band\b.*\band\b.*\band\b",
        "medium",
        "Break into separate requests. Claude works better on focused tasks.",
    ),
    AntiPattern(
        "Missing context",
        r"^(why|how come|what happened)",
        "low",
        "Include the error message or file path for faster resolution.",
    ),
]


def detect_anti_patterns(prompt: str) -> list[dict[str, str]]:
    """Detect anti-patterns in a user prompt."""
    findings: list[dict[str, str]] = []
    for ap in ANTI_PATTERNS:
        if re.search(ap.detection_pattern, prompt, re.IGNORECASE):
            findings.append({
                "pattern": ap.name,
                "severity": ap.severity,
                "suggestion": ap.suggestion,
            })
    return findings


def exercise_2():
    """Demonstrate anti-pattern detection."""
    prompts = [
        "Fix it.",
        "Add user authentication and payment processing and email notifications and admin dashboard",
        "Why is it broken?",
        "Fix the TypeError in src/auth.py:42. The error is 'NoneType' has no attribute 'id'. Run tests after.",
        "Implement the caching layer",
    ]
    for prompt in prompts:
        findings = detect_anti_patterns(prompt)
        if findings:
            print(f"  '{prompt}'")
            for f in findings:
                print(f"    [{f['severity']}] {f['pattern']}: {f['suggestion']}")
        else:
            print(f"  '{prompt}' → No issues detected")
        print()


# === Exercise 3: Context Management Strategy ===
# Problem: Implement strategies for managing context window usage
#   effectively across long sessions.

@dataclass
class ContextEntry:
    """An entry consuming context window space."""
    source: str    # "system", "user", "assistant", "tool_result"
    tokens: int
    importance: str  # "critical", "high", "medium", "low"
    compressible: bool = True


class ContextManager:
    """Manages context window allocation and compression."""

    def __init__(self, max_tokens: int = 200_000) -> None:
        self.max_tokens = max_tokens
        self.entries: list[ContextEntry] = []

    def add(self, entry: ContextEntry) -> None:
        self.entries.append(entry)

    @property
    def used_tokens(self) -> int:
        return sum(e.tokens for e in self.entries)

    @property
    def available(self) -> int:
        return max(0, self.max_tokens - self.used_tokens)

    def utilization(self) -> float:
        return (self.used_tokens / self.max_tokens) * 100

    def should_compact(self, threshold_pct: float = 80.0) -> bool:
        return self.utilization() >= threshold_pct

    def simulate_compact(self) -> dict[str, Any]:
        """Simulate /compact: compress low-importance entries."""
        before = self.used_tokens
        compressible = [
            e for e in self.entries
            if e.compressible and e.importance in ("low", "medium")
        ]
        saved = sum(int(e.tokens * 0.7) for e in compressible)
        for e in compressible:
            e.tokens = int(e.tokens * 0.3)
        after = self.used_tokens
        return {
            "before_tokens": before,
            "after_tokens": after,
            "saved_tokens": before - after,
            "entries_compressed": len(compressible),
            "utilization_before": round(before / self.max_tokens * 100, 1),
            "utilization_after": round(after / self.max_tokens * 100, 1),
        }

    def breakdown(self) -> dict[str, int]:
        by_source: dict[str, int] = {}
        for e in self.entries:
            by_source[e.source] = by_source.get(e.source, 0) + e.tokens
        return by_source


def exercise_3():
    """Demonstrate context management."""
    ctx = ContextManager(max_tokens=200_000)

    ctx.add(ContextEntry("system", 3_000, "critical", False))
    ctx.add(ContextEntry("user", 500, "high"))
    ctx.add(ContextEntry("tool_result", 50_000, "medium"))
    ctx.add(ContextEntry("assistant", 10_000, "high"))
    ctx.add(ContextEntry("user", 800, "high"))
    ctx.add(ContextEntry("tool_result", 80_000, "low"))
    ctx.add(ContextEntry("assistant", 20_000, "medium"))

    print(f"  Used: {ctx.used_tokens:,}/{ctx.max_tokens:,} "
          f"({ctx.utilization():.1f}%)")
    print(f"  Breakdown: {ctx.breakdown()}")
    print(f"  Should compact: {ctx.should_compact()}")

    if ctx.should_compact():
        result = ctx.simulate_compact()
        print(f"\n  After /compact:")
        print(f"    {result['utilization_before']}% → "
              f"{result['utilization_after']}%")
        print(f"    Saved: {result['saved_tokens']:,} tokens "
              f"({result['entries_compressed']} entries compressed)")


if __name__ == "__main__":
    print("=== Exercise 1: Prompt Patterns ===")
    exercise_1()

    print("\n=== Exercise 2: Anti-Pattern Detector ===")
    exercise_2()

    print("=== Exercise 3: Context Management ===")
    exercise_3()

    print("\nAll exercises completed!")
