"""
Exercises for Lesson 03: CLAUDE.md and Project Setup
Topic: Claude_Ecosystem

Solutions to practice problems from the lesson.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# === Exercise 1: CLAUDE.md Section Parser ===
# Problem: Parse a CLAUDE.md file into structured sections (headings + content).

@dataclass
class MarkdownSection:
    """A section from a Markdown file."""
    level: int
    heading: str
    content: str
    line_number: int


def parse_claude_md(text: str) -> list[MarkdownSection]:
    """Parse CLAUDE.md content into structured sections.

    Splits on Markdown headings (## or ###) and captures content
    between them.
    """
    sections: list[MarkdownSection] = []
    lines = text.split("\n")
    current_heading = ""
    current_level = 0
    current_content: list[str] = []
    heading_line = 0

    for i, line in enumerate(lines, 1):
        match = re.match(r"^(#{1,6})\s+(.+)$", line)
        if match:
            if current_heading:
                sections.append(MarkdownSection(
                    level=current_level,
                    heading=current_heading,
                    content="\n".join(current_content).strip(),
                    line_number=heading_line,
                ))
            current_level = len(match.group(1))
            current_heading = match.group(2)
            current_content = []
            heading_line = i
        else:
            current_content.append(line)

    if current_heading:
        sections.append(MarkdownSection(
            level=current_level,
            heading=current_heading,
            content="\n".join(current_content).strip(),
            line_number=heading_line,
        ))
    return sections


def exercise_1():
    """Demonstrate CLAUDE.md parsing."""
    sample_claude_md = """# CLAUDE.md

## Code Style
- Python: PEP 8
- Use type hints everywhere

## Testing
Run tests with: `pytest tests/`
Coverage threshold: 80%

### Unit Tests
Located in tests/unit/

## Deployment
Use `make deploy` for production.
"""
    sections = parse_claude_md(sample_claude_md)
    for s in sections:
        print(f"  L{s.line_number} {'#' * s.level} {s.heading} "
              f"({len(s.content)} chars)")


# === Exercise 2: Settings Hierarchy Resolver ===
# Problem: Implement the Claude Code settings merge order:
#   global < project < local (later values override earlier ones).

def merge_settings(
    global_settings: dict[str, Any],
    project_settings: dict[str, Any],
    local_settings: dict[str, Any],
) -> dict[str, Any]:
    """Merge settings from three levels with increasing priority.

    Simulates Claude Code's settings hierarchy:
    - ~/.claude/settings.json (global)
    - .claude/settings.json (project, committed)
    - .claude/settings.local.json (local, gitignored)
    """
    merged: dict[str, Any] = {}

    for settings in [global_settings, project_settings, local_settings]:
        for key, value in settings.items():
            if (isinstance(value, list)
                    and isinstance(merged.get(key), list)):
                merged[key] = merged[key] + value
            elif (isinstance(value, dict)
                  and isinstance(merged.get(key), dict)):
                merged[key] = {**merged[key], **value}
            else:
                merged[key] = value

    return merged


def exercise_2():
    """Demonstrate settings hierarchy merge."""
    global_s = {
        "permissions": {"allow": ["Read"]},
        "model": "sonnet",
        "verbose": False,
    }
    project_s = {
        "permissions": {"allow": ["Bash(npm test)"]},
        "model": "sonnet",
    }
    local_s = {
        "model": "opus",
        "verbose": True,
    }
    merged = merge_settings(global_s, project_s, local_s)
    print(f"  model:   {merged['model']}  (local overrides)")
    print(f"  verbose: {merged['verbose']}  (local overrides)")
    print(f"  allow:   {merged['permissions']['allow']}  (lists merged)")


# === Exercise 3: CLAUDE.md Validator ===
# Problem: Validate that a CLAUDE.md file contains recommended sections
#   and follows best practices.

RECOMMENDED_SECTIONS = [
    "Code Style",
    "Testing",
    "Project Structure",
]

BEST_PRACTICES = [
    ("has_heading", "File should start with a top-level heading"),
    ("not_too_long", "File should be under 500 lines"),
    ("has_code_blocks", "Include code examples for commands"),
    ("no_secrets", "Should not contain API keys or passwords"),
]

SECRET_PATTERNS = [
    r"(?i)(api[_-]?key|password|secret)\s*[:=]\s*\S+",
    r"sk-[a-zA-Z0-9]{20,}",
    r"ghp_[a-zA-Z0-9]{36}",
]


def validate_claude_md(text: str) -> list[dict[str, Any]]:
    """Validate a CLAUDE.md file and return findings."""
    findings: list[dict[str, Any]] = []
    lines = text.split("\n")
    sections = parse_claude_md(text)
    section_headings = [s.heading for s in sections]

    if not lines or not lines[0].startswith("# "):
        findings.append({"severity": "warning",
                         "message": "Missing top-level heading"})

    if len(lines) > 500:
        findings.append({"severity": "warning",
                         "message": f"File is {len(lines)} lines (>500)"})

    if "```" not in text:
        findings.append({"severity": "info",
                         "message": "No code blocks found"})

    for section in RECOMMENDED_SECTIONS:
        if not any(section.lower() in h.lower() for h in section_headings):
            findings.append({"severity": "info",
                             "message": f"Missing recommended section: {section}"})

    for pattern in SECRET_PATTERNS:
        if re.search(pattern, text):
            findings.append({"severity": "error",
                             "message": "Potential secret detected!"})
            break

    if not findings:
        findings.append({"severity": "ok",
                         "message": "All checks passed"})

    return findings


def exercise_3():
    """Demonstrate CLAUDE.md validation."""
    good_md = """# CLAUDE.md

## Code Style
Python: PEP 8, type hints required.

## Testing
```bash
pytest tests/ --cov=src
```

## Project Structure
- src/ — main source code
- tests/ — test files
"""
    bad_md = "Just some text without structure.\napi_key: sk-abc123secret"

    print("  Good CLAUDE.md:")
    for f in validate_claude_md(good_md):
        print(f"    [{f['severity']}] {f['message']}")

    print("  Bad CLAUDE.md:")
    for f in validate_claude_md(bad_md):
        print(f"    [{f['severity']}] {f['message']}")


# === Exercise 4: .claude/ Directory Mapper ===
# Problem: Model the expected .claude/ directory structure and determine
#   which files should be committed vs gitignored.

@dataclass
class ClaudeFile:
    """A file in the .claude/ directory."""
    path: str
    purpose: str
    should_commit: bool


CLAUDE_DIR_FILES = [
    ClaudeFile("settings.json", "Project settings (shared)", True),
    ClaudeFile("settings.local.json", "Personal settings (local)", False),
    ClaudeFile("CLAUDE.md", "Project instructions (shared)", True),
    ClaudeFile("commands/", "Custom slash commands", True),
    ClaudeFile("plans/", "Generated plan files", False),
    ClaudeFile("worktrees/", "Worktree branches", False),
]


def generate_gitignore() -> str:
    """Generate .claude/.gitignore content based on commit rules."""
    ignored = [f.path for f in CLAUDE_DIR_FILES if not f.should_commit]
    return "\n".join(ignored)


def exercise_4():
    """Demonstrate .claude/ directory structure."""
    print("  .claude/ directory:")
    for f in CLAUDE_DIR_FILES:
        status = "commit" if f.should_commit else "gitignore"
        print(f"    {f.path:<25} [{status}] — {f.purpose}")
    print(f"\n  Generated .gitignore:\n{generate_gitignore()}")


if __name__ == "__main__":
    print("=== Exercise 1: CLAUDE.md Parser ===")
    exercise_1()

    print("\n=== Exercise 2: Settings Hierarchy ===")
    exercise_2()

    print("\n=== Exercise 3: CLAUDE.md Validator ===")
    exercise_3()

    print("\n=== Exercise 4: .claude/ Directory ===")
    exercise_4()

    print("\nAll exercises completed!")
