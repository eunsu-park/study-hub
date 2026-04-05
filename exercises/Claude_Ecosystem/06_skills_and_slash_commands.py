"""
Exercises for Lesson 06: Skills and Slash Commands
Topic: Claude_Ecosystem

Solutions to practice problems from the lesson.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


# === Exercise 1: SKILL.md Parser ===
# Problem: Parse a SKILL.md file with YAML frontmatter and Markdown body.

@dataclass
class SkillDefinition:
    """Parsed skill definition from a SKILL.md file."""
    name: str
    description: str
    trigger: str           # "user" or "auto"
    auto_trigger: str      # regex or keyword for auto-invoke
    allowed_tools: list[str]
    instructions: str


def parse_skill_md(text: str) -> SkillDefinition | str:
    """Parse SKILL.md content into a SkillDefinition.

    Expected format:
    ---
    name: skill-name
    description: What the skill does
    trigger: user | auto
    auto_trigger: pattern (optional)
    allowed_tools: [Tool1, Tool2]
    ---
    # Instructions body in Markdown
    """
    frontmatter_match = re.match(
        r"^---\s*\n(.*?)\n---\s*\n(.*)$", text, re.DOTALL
    )
    if not frontmatter_match:
        return "Invalid SKILL.md: missing YAML frontmatter"

    yaml_text = frontmatter_match.group(1)
    body = frontmatter_match.group(2).strip()

    fields: dict[str, str] = {}
    for line in yaml_text.split("\n"):
        match = re.match(r"^(\w+):\s*(.+)$", line.strip())
        if match:
            fields[match.group(1)] = match.group(2).strip()

    name = fields.get("name", "")
    if not name:
        return "SKILL.md must have a 'name' field"

    tools_raw = fields.get("allowed_tools", "[]")
    tools = [t.strip().strip("'\"") for t in
             tools_raw.strip("[]").split(",") if t.strip()]

    return SkillDefinition(
        name=name,
        description=fields.get("description", ""),
        trigger=fields.get("trigger", "user"),
        auto_trigger=fields.get("auto_trigger", ""),
        allowed_tools=tools,
        instructions=body,
    )


def exercise_1():
    """Demonstrate SKILL.md parsing."""
    skill_text = """---
name: commit
description: Create a well-formatted git commit
trigger: user
allowed_tools: [Bash, Read, Glob]
---
# Commit Skill

1. Run `git status` to see changes
2. Run `git diff --staged` to review staged changes
3. Write a commit message following Conventional Commits
4. Create the commit with `git commit`
"""
    result = parse_skill_md(skill_text)
    if isinstance(result, str):
        print(f"  ERROR: {result}")
    else:
        print(f"  Name:    {result.name}")
        print(f"  Trigger: {result.trigger}")
        print(f"  Tools:   {result.allowed_tools}")
        print(f"  Body:    {len(result.instructions)} chars")


# === Exercise 2: Command Router ===
# Problem: Build a router that maps slash commands to skills.

@dataclass
class SkillRegistry:
    """Registry of available skills with routing."""
    skills: dict[str, SkillDefinition] = field(default_factory=dict)

    def register(self, skill: SkillDefinition) -> None:
        self.skills[f"/{skill.name}"] = skill

    def route(self, user_input: str) -> tuple[str, SkillDefinition | None, str]:
        """Route user input to a skill.

        Returns: (input_type, matched_skill, remaining_args)
        """
        stripped = user_input.strip()
        if not stripped.startswith("/"):
            # Check auto-trigger skills
            for skill in self.skills.values():
                if (skill.trigger == "auto" and skill.auto_trigger
                        and re.search(skill.auto_trigger, stripped, re.I)):
                    return ("auto_triggered", skill, stripped)
            return ("message", None, stripped)

        parts = stripped.split(maxsplit=1)
        command = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""

        skill = self.skills.get(command)
        if skill:
            return ("skill", skill, args)
        return ("unknown_command", None, command)

    def list_skills(self) -> list[dict[str, str]]:
        return [
            {"command": cmd, "description": s.description,
             "trigger": s.trigger}
            for cmd, s in sorted(self.skills.items())
        ]


def exercise_2():
    """Demonstrate skill routing."""
    registry = SkillRegistry()

    commit_skill = SkillDefinition(
        "commit", "Create a git commit", "user", "", ["Bash", "Read"], "...")
    review_skill = SkillDefinition(
        "review", "Review code changes", "user", "", ["Read", "Grep"], "...")
    test_auto = SkillDefinition(
        "auto-test", "Run tests on test file changes", "auto",
        r"test_\w+\.py", ["Bash"], "Run pytest on the mentioned test file.")

    registry.register(commit_skill)
    registry.register(review_skill)
    registry.register(test_auto)

    inputs = [
        "/commit -m 'fix bug'",
        "/review",
        "/deploy",
        "Please fix test_utils.py",
        "Refactor the auth module",
    ]
    for text in inputs:
        kind, skill, args = registry.route(text)
        skill_name = skill.name if skill else "none"
        print(f"  '{text}' → {kind}, skill={skill_name}, args='{args}'")


# === Exercise 3: Skill Comparison Matrix ===
# Problem: Compare hooks, skills, and CLAUDE.md across key dimensions.

@dataclass
class MechanismComparison:
    """Comparison of Claude Code configuration mechanisms."""
    name: str
    deterministic: bool
    configurable_by: str    # "JSON", "Markdown", "YAML+Markdown"
    invocation: str         # "automatic", "manual", "always loaded"
    scope: str              # "tool-level", "task-level", "project-level"
    versioned: bool


MECHANISMS = [
    MechanismComparison(
        "Hooks", True, "JSON",
        "automatic (on tool events)", "tool-level", True),
    MechanismComparison(
        "Skills", False, "YAML+Markdown",
        "manual (/command) or auto-trigger", "task-level", True),
    MechanismComparison(
        "CLAUDE.md", False, "Markdown",
        "always loaded into context", "project-level", True),
]


def recommend_for_use_case(use_case: str) -> str:
    """Recommend the best mechanism for a given use case."""
    use_case_lower = use_case.lower()
    if any(w in use_case_lower for w in ["format", "lint", "block", "notify"]):
        return "Hooks"
    elif any(w in use_case_lower for w in
             ["commit", "review", "deploy", "workflow"]):
        return "Skills"
    else:
        return "CLAUDE.md"


def exercise_3():
    """Display mechanism comparison matrix."""
    print(f"  {'Mechanism':<12} {'Deterministic':<14} {'Config':<16} "
          f"{'Invocation':<35} {'Scope'}")
    print("  " + "-" * 90)
    for m in MECHANISMS:
        det = "Yes" if m.deterministic else "No"
        print(f"  {m.name:<12} {det:<14} {m.configurable_by:<16} "
              f"{m.invocation:<35} {m.scope}")

    print()
    use_cases = [
        "Auto-format Python files on save",
        "Standardized commit message workflow",
        "Use tabs instead of spaces",
        "Block dangerous git commands",
        "Run deployment checklist",
    ]
    for uc in use_cases:
        rec = recommend_for_use_case(uc)
        print(f"  {uc} → {rec}")


if __name__ == "__main__":
    print("=== Exercise 1: SKILL.md Parser ===")
    exercise_1()

    print("\n=== Exercise 2: Command Router ===")
    exercise_2()

    print("\n=== Exercise 3: Mechanism Comparison ===")
    exercise_3()

    print("\nAll exercises completed!")
