"""
Exercises for Lesson 09: IDE Integration
Topic: Claude_Ecosystem

Solutions to practice problems from the lesson.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


# === Exercise 1: Keybinding Configuration Manager ===
# Problem: Parse, validate, and manage IDE keybinding configurations
#   for Claude Code's VS Code and JetBrains integrations.

@dataclass
class Keybinding:
    """A keyboard shortcut binding."""
    key: str
    command: str
    when: str = ""
    description: str = ""


DEFAULT_KEYBINDINGS = [
    Keybinding("Ctrl+L", "claude.openPanel",
               description="Open Claude Code panel"),
    Keybinding("Ctrl+Shift+L", "claude.addToChat",
               when="editorHasSelection",
               description="Add selection to Claude chat"),
    Keybinding("Ctrl+I", "claude.inlineEdit",
               when="editorTextFocus",
               description="Inline edit with Claude"),
    Keybinding("Escape", "claude.cancel",
               when="claudeIsRunning",
               description="Cancel current Claude operation"),
]


def parse_keybindings(raw_json: str) -> list[Keybinding] | str:
    """Parse keybindings from a JSON string."""
    try:
        data = json.loads(raw_json)
    except json.JSONDecodeError as e:
        return f"Invalid JSON: {e}"

    if not isinstance(data, list):
        return "Keybindings must be a JSON array"

    bindings: list[Keybinding] = []
    for entry in data:
        if "key" not in entry or "command" not in entry:
            return f"Missing required fields in: {entry}"
        bindings.append(Keybinding(
            key=entry["key"],
            command=entry["command"],
            when=entry.get("when", ""),
            description=entry.get("description", ""),
        ))
    return bindings


def detect_conflicts(bindings: list[Keybinding]) -> list[tuple[Keybinding, Keybinding]]:
    """Detect keybinding conflicts (same key, overlapping conditions)."""
    conflicts: list[tuple[Keybinding, Keybinding]] = []
    for i, a in enumerate(bindings):
        for b in bindings[i + 1:]:
            if a.key == b.key and (not a.when or not b.when or a.when == b.when):
                conflicts.append((a, b))
    return conflicts


def exercise_1():
    """Demonstrate keybinding management."""
    print("  Default keybindings:")
    for kb in DEFAULT_KEYBINDINGS:
        when = f" (when: {kb.when})" if kb.when else ""
        print(f"    {kb.key:<18} → {kb.command}{when}")

    # Test conflict detection
    test_bindings = DEFAULT_KEYBINDINGS + [
        Keybinding("Ctrl+L", "editor.action.formatDocument",
                   description="Format document"),
    ]
    conflicts = detect_conflicts(test_bindings)
    print(f"\n  Conflicts found: {len(conflicts)}")
    for a, b in conflicts:
        print(f"    {a.key}: '{a.command}' vs '{b.command}'")


# === Exercise 2: Workspace Configuration ===
# Problem: Model VS Code workspace settings for Claude Code integration.

@dataclass
class WorkspaceConfig:
    """VS Code workspace configuration for Claude Code."""
    claude_enabled: bool = True
    claude_model: str = "sonnet"
    claude_auto_format: bool = True
    editor_font_size: int = 14
    editor_theme: str = "Default Dark+"
    extensions: list[str] = field(default_factory=list)


def generate_vscode_settings(config: WorkspaceConfig) -> dict[str, Any]:
    """Generate .vscode/settings.json content for Claude Code."""
    settings: dict[str, Any] = {
        "claude.enabled": config.claude_enabled,
        "claude.model": config.claude_model,
        "claude.autoFormat": config.claude_auto_format,
        "editor.fontSize": config.editor_font_size,
        "workbench.colorTheme": config.editor_theme,
    }
    if config.extensions:
        settings["claude.recommendedExtensions"] = config.extensions
    return settings


def validate_workspace(settings: dict[str, Any]) -> list[str]:
    """Validate workspace settings for common issues."""
    issues: list[str] = []
    valid_models = {"opus", "sonnet", "haiku"}

    model = settings.get("claude.model", "")
    if model and model not in valid_models:
        issues.append(f"Unknown model: {model}")

    if not settings.get("claude.enabled", True):
        issues.append("Claude is disabled in workspace settings")

    font_size = settings.get("editor.fontSize", 14)
    if not isinstance(font_size, int) or font_size < 8 or font_size > 72:
        issues.append(f"Unusual font size: {font_size}")

    return issues


def exercise_2():
    """Demonstrate workspace configuration."""
    config = WorkspaceConfig(
        claude_model="sonnet",
        extensions=["ms-python.python", "esbenp.prettier-vscode"],
    )
    settings = generate_vscode_settings(config)
    print("  Generated settings.json:")
    print(f"    {json.dumps(settings, indent=4)}")

    # Validate a bad config
    bad_settings = {"claude.model": "gpt-4", "claude.enabled": False,
                    "editor.fontSize": 200}
    issues = validate_workspace(bad_settings)
    print(f"\n  Validation issues: {issues}")


# === Exercise 3: IDE Feature Compatibility Matrix ===
# Problem: Track feature availability across IDEs (VS Code, JetBrains, Terminal).

@dataclass
class IDEFeature:
    """A Claude Code feature and its IDE support status."""
    name: str
    vscode: bool
    jetbrains: bool
    terminal: bool
    description: str


IDE_FEATURES = [
    IDEFeature("Inline editing", True, True, False,
               "Edit code inline with Ctrl+I"),
    IDEFeature("Chat panel", True, True, False,
               "Side panel for conversation"),
    IDEFeature("Add selection to chat", True, True, False,
               "Send selected code to Claude"),
    IDEFeature("Terminal integration", True, False, True,
               "Run Claude in integrated terminal"),
    IDEFeature("Slash commands", True, True, True,
               "Use /commit, /review, etc."),
    IDEFeature("Multi-file editing", True, True, True,
               "Edit multiple files in one session"),
    IDEFeature("Git integration", True, True, True,
               "Commit, push, create PRs"),
    IDEFeature("MCP support", True, True, True,
               "Connect to MCP servers"),
]


def get_support_summary() -> dict[str, dict[str, int]]:
    """Summarize feature support by IDE."""
    ides = {"vscode": 0, "jetbrains": 0, "terminal": 0}
    total = len(IDE_FEATURES)
    for f in IDE_FEATURES:
        if f.vscode:
            ides["vscode"] += 1
        if f.jetbrains:
            ides["jetbrains"] += 1
        if f.terminal:
            ides["terminal"] += 1
    return {ide: {"supported": count, "total": total,
                  "pct": round(count / total * 100)}
            for ide, count in ides.items()}


def exercise_3():
    """Display IDE feature compatibility matrix."""
    print(f"  {'Feature':<25} {'VS Code':>8} {'JetBrains':>10} "
          f"{'Terminal':>9}")
    print("  " + "-" * 55)
    for f in IDE_FEATURES:
        vs = "Yes" if f.vscode else "—"
        jb = "Yes" if f.jetbrains else "—"
        tm = "Yes" if f.terminal else "—"
        print(f"  {f.name:<25} {vs:>8} {jb:>10} {tm:>9}")

    print()
    summary = get_support_summary()
    for ide, stats in summary.items():
        print(f"  {ide}: {stats['supported']}/{stats['total']} "
              f"({stats['pct']}%)")


if __name__ == "__main__":
    print("=== Exercise 1: Keybinding Manager ===")
    exercise_1()

    print("\n=== Exercise 2: Workspace Config ===")
    exercise_2()

    print("\n=== Exercise 3: IDE Compatibility ===")
    exercise_3()

    print("\nAll exercises completed!")
