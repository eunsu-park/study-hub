"""
Exercises for Lesson 02: Claude Code — Getting Started
Topic: Claude_Ecosystem

Solutions to practice problems from the lesson.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# === Exercise 1: Tool System Simulator ===
# Problem: Model the core Claude Code tools (Read, Write, Edit, Bash, Glob, Grep)
#   and simulate a tool dispatch system.

class ToolCategory(Enum):
    FILE_READ = "file_read"
    FILE_WRITE = "file_write"
    SEARCH = "search"
    EXECUTION = "execution"


@dataclass
class Tool:
    """Represents a Claude Code built-in tool."""
    name: str
    category: ToolCategory
    description: str
    requires_permission: bool
    parameters: list[str]


CLAUDE_CODE_TOOLS = [
    Tool("Read", ToolCategory.FILE_READ,
         "Read file contents", False, ["file_path", "offset", "limit"]),
    Tool("Write", ToolCategory.FILE_WRITE,
         "Create or overwrite a file", True, ["file_path", "content"]),
    Tool("Edit", ToolCategory.FILE_WRITE,
         "Replace text in a file", True, ["file_path", "old_string", "new_string"]),
    Tool("Bash", ToolCategory.EXECUTION,
         "Execute a shell command", True, ["command", "timeout"]),
    Tool("Glob", ToolCategory.SEARCH,
         "Find files by pattern", False, ["pattern", "path"]),
    Tool("Grep", ToolCategory.SEARCH,
         "Search file contents", False, ["pattern", "path", "type"]),
]


def dispatch_tool(tool_name: str, params: dict[str, Any]) -> dict[str, Any]:
    """Simulate dispatching a tool call and returning a result."""
    tool = next((t for t in CLAUDE_CODE_TOOLS if t.name == tool_name), None)
    if tool is None:
        return {"error": f"Unknown tool: {tool_name}"}

    missing = [p for p in tool.parameters[:1] if p not in params]
    if missing:
        return {"error": f"Missing required parameter: {missing[0]}"}

    return {
        "tool": tool.name,
        "category": tool.category.value,
        "requires_permission": tool.requires_permission,
        "params": params,
        "status": "simulated_success",
    }


def exercise_1():
    """Demonstrate tool dispatch simulation."""
    calls = [
        ("Read", {"file_path": "src/main.py"}),
        ("Edit", {"file_path": "src/main.py",
                  "old_string": "def old()", "new_string": "def new()"}),
        ("Grep", {"pattern": "TODO", "path": "src/"}),
        ("Bash", {"command": "python -m pytest"}),
        ("UnknownTool", {"arg": "value"}),
    ]
    for name, params in calls:
        result = dispatch_tool(name, params)
        status = result.get("status", result.get("error"))
        perm = result.get("requires_permission", "N/A")
        print(f"  {name}: {status} (permission: {perm})")


# === Exercise 2: Conversation State Manager ===
# Problem: Model the Claude Code conversation as a sequence of
#   user messages, assistant responses, and tool calls.

@dataclass
class Message:
    """A single message in a Claude Code session."""
    role: str          # "user", "assistant", "tool_result"
    content: str
    tool_name: str | None = None
    tool_params: dict[str, Any] | None = None


@dataclass
class Session:
    """Tracks state for a Claude Code session."""
    messages: list[Message] = field(default_factory=list)
    tools_used: dict[str, int] = field(default_factory=dict)
    files_read: set[str] = field(default_factory=set)
    files_modified: set[str] = field(default_factory=set)

    def add_user_message(self, content: str) -> None:
        self.messages.append(Message("user", content))

    def add_tool_call(self, tool_name: str, params: dict[str, Any],
                      result: str) -> None:
        self.messages.append(
            Message("assistant", f"[tool_use: {tool_name}]",
                    tool_name=tool_name, tool_params=params))
        self.messages.append(Message("tool_result", result))
        self.tools_used[tool_name] = self.tools_used.get(tool_name, 0) + 1

        if tool_name == "Read":
            self.files_read.add(params.get("file_path", ""))
        elif tool_name in ("Write", "Edit"):
            self.files_modified.add(params.get("file_path", ""))

    def add_assistant_message(self, content: str) -> None:
        self.messages.append(Message("assistant", content))

    def summary(self) -> dict[str, Any]:
        return {
            "total_messages": len(self.messages),
            "tools_used": dict(self.tools_used),
            "files_read": sorted(self.files_read),
            "files_modified": sorted(self.files_modified),
        }


def exercise_2():
    """Simulate a Claude Code debugging session."""
    session = Session()
    session.add_user_message("Fix the failing test in test_utils.py")
    session.add_tool_call("Read", {"file_path": "tests/test_utils.py"},
                          "def test_add(): assert add(1, 2) == 4")
    session.add_tool_call("Read", {"file_path": "src/utils.py"},
                          "def add(a, b): return a + b")
    session.add_tool_call("Edit", {
        "file_path": "tests/test_utils.py",
        "old_string": "assert add(1, 2) == 4",
        "new_string": "assert add(1, 2) == 3",
    }, "File updated")
    session.add_tool_call("Bash", {"command": "python -m pytest tests/test_utils.py"},
                          "1 passed")
    session.add_assistant_message("Fixed: the test expected 4 but 1+2=3.")

    summary = session.summary()
    print(f"  Messages: {summary['total_messages']}")
    print(f"  Tools:    {summary['tools_used']}")
    print(f"  Read:     {summary['files_read']}")
    print(f"  Modified: {summary['files_modified']}")


# === Exercise 3: Slash Command Parser ===
# Problem: Parse and route Claude Code slash commands like
#   /help, /clear, /compact, /model, /cost.

SLASH_COMMANDS = {
    "/help": "Show available commands and usage information",
    "/clear": "Clear conversation history and start fresh",
    "/compact": "Summarize conversation to free context space",
    "/model": "Switch the active Claude model (opus, sonnet, haiku)",
    "/cost": "Show token usage and cost for this session",
    "/init": "Initialize CLAUDE.md for the current project",
    "/review": "Review recent code changes",
}


def parse_slash_command(input_text: str) -> dict[str, Any]:
    """Parse a slash command and extract command name and arguments."""
    input_text = input_text.strip()
    if not input_text.startswith("/"):
        return {"type": "message", "content": input_text}

    parts = input_text.split(maxsplit=1)
    command = parts[0].lower()
    args = parts[1] if len(parts) > 1 else ""

    if command not in SLASH_COMMANDS:
        return {"type": "error", "message": f"Unknown command: {command}"}

    return {
        "type": "command",
        "command": command,
        "args": args,
        "description": SLASH_COMMANDS[command],
    }


def exercise_3():
    """Demonstrate slash command parsing."""
    inputs = [
        "/help",
        "/model sonnet",
        "/cost",
        "/unknown",
        "Just a normal message",
    ]
    for text in inputs:
        result = parse_slash_command(text)
        if result["type"] == "command":
            print(f"  '{text}' → {result['command']} "
                  f"(args: '{result.get('args', '')}')")
        elif result["type"] == "error":
            print(f"  '{text}' → ERROR: {result['message']}")
        else:
            print(f"  '{text}' → regular message")


if __name__ == "__main__":
    print("=== Exercise 1: Tool Dispatch ===")
    exercise_1()

    print("\n=== Exercise 2: Session State ===")
    exercise_2()

    print("\n=== Exercise 3: Slash Commands ===")
    exercise_3()

    print("\nAll exercises completed!")
