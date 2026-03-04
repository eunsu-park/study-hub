"""
Exercises for Lesson 17: Claude Agent SDK
Topic: Claude_Ecosystem

Solutions to practice problems from the lesson.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# === Exercise 1: SDK Query Options Builder ===
# Problem: Build and validate QueryOptions configurations for the
#   Claude Agent SDK (claude_code_sdk).

class ModelChoice(Enum):
    OPUS = "opus"
    SONNET = "sonnet"
    HAIKU = "haiku"


@dataclass
class QueryOptions:
    """Configuration options for sdk.query()."""
    max_turns: int = 10
    system_prompt: str = ""
    model: ModelChoice = ModelChoice.SONNET
    allowed_tools: list[str] = field(default_factory=list)
    permission_mode: str = "default"
    cwd: str = "."

    def validate(self) -> list[str]:
        errors: list[str] = []
        if self.max_turns < 1:
            errors.append("max_turns must be at least 1")
        if self.max_turns > 100:
            errors.append("max_turns should not exceed 100")
        valid_modes = {"default", "auto", "plan", "bypass"}
        if self.permission_mode not in valid_modes:
            errors.append(f"Invalid permission_mode: {self.permission_mode}")
        valid_tools = {"Read", "Write", "Edit", "Bash", "Glob", "Grep",
                       "Agent", "NotebookEdit", "WebFetch", "WebSearch"}
        for tool in self.allowed_tools:
            base_tool = tool.split("(")[0]
            if base_tool not in valid_tools:
                errors.append(f"Unknown tool: {tool}")
        return errors

    def to_sdk_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "max_turns": self.max_turns,
        }
        if self.system_prompt:
            kwargs["system_prompt"] = self.system_prompt
        if self.model != ModelChoice.SONNET:
            kwargs["model"] = self.model.value
        if self.allowed_tools:
            kwargs["allowed_tools"] = self.allowed_tools
        if self.permission_mode != "default":
            kwargs["permission_mode"] = self.permission_mode
        if self.cwd != ".":
            kwargs["cwd"] = self.cwd
        return kwargs


def exercise_1():
    """Demonstrate QueryOptions configuration."""
    configs = [
        ("Read-only explorer", QueryOptions(
            max_turns=5,
            system_prompt="Explore the codebase and report findings.",
            allowed_tools=["Read", "Glob", "Grep"],
            permission_mode="plan",
        )),
        ("Code modifier", QueryOptions(
            max_turns=20,
            system_prompt="Implement the requested changes.",
            model=ModelChoice.OPUS,
        )),
        ("Invalid config", QueryOptions(
            max_turns=-1,
            permission_mode="turbo",
            allowed_tools=["FakeToolName"],
        )),
    ]
    for label, opts in configs:
        errors = opts.validate()
        print(f"  {label}:")
        if errors:
            for e in errors:
                print(f"    ERROR: {e}")
        else:
            kwargs = opts.to_sdk_kwargs()
            print(f"    SDK kwargs: {kwargs}")
        print()


# === Exercise 2: Agent Event Processor ===
# Problem: Process the different message types returned by sdk.query()
#   and extract structured information.

class MessageType(Enum):
    TEXT = "text"
    TOOL_USE = "tool_use"
    TOOL_RESULT = "tool_result"
    SYSTEM = "system"
    ERROR = "error"


@dataclass
class AgentMessage:
    """A message from the Agent SDK event stream."""
    msg_type: MessageType
    content: str = ""
    tool_name: str = ""
    tool_input: dict[str, Any] = field(default_factory=dict)
    tool_result: str = ""


@dataclass
class AgentRunSummary:
    """Summary of an agent run extracted from its message stream."""
    text_responses: list[str] = field(default_factory=list)
    tools_called: list[dict[str, Any]] = field(default_factory=list)
    files_read: set[str] = field(default_factory=set)
    files_modified: set[str] = field(default_factory=set)
    commands_run: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def process_agent_messages(messages: list[AgentMessage]) -> AgentRunSummary:
    """Process agent messages into a structured summary."""
    summary = AgentRunSummary()

    for msg in messages:
        if msg.msg_type == MessageType.TEXT:
            summary.text_responses.append(msg.content)

        elif msg.msg_type == MessageType.TOOL_USE:
            summary.tools_called.append({
                "tool": msg.tool_name,
                "input": msg.tool_input,
            })
            if msg.tool_name == "Read":
                path = msg.tool_input.get("file_path", "")
                if path:
                    summary.files_read.add(path)
            elif msg.tool_name in ("Write", "Edit"):
                path = msg.tool_input.get("file_path", "")
                if path:
                    summary.files_modified.add(path)
            elif msg.tool_name == "Bash":
                cmd = msg.tool_input.get("command", "")
                if cmd:
                    summary.commands_run.append(cmd)

        elif msg.msg_type == MessageType.ERROR:
            summary.errors.append(msg.content)

    return summary


def exercise_2():
    """Simulate and process agent messages."""
    messages = [
        AgentMessage(MessageType.TEXT, "Let me investigate the issue."),
        AgentMessage(MessageType.TOOL_USE, tool_name="Read",
                     tool_input={"file_path": "src/auth.py"}),
        AgentMessage(MessageType.TOOL_RESULT, tool_result="class AuthManager: ..."),
        AgentMessage(MessageType.TOOL_USE, tool_name="Grep",
                     tool_input={"pattern": "def login", "path": "src/"}),
        AgentMessage(MessageType.TOOL_USE, tool_name="Edit",
                     tool_input={"file_path": "src/auth.py",
                                 "old_string": "pass", "new_string": "return True"}),
        AgentMessage(MessageType.TOOL_USE, tool_name="Bash",
                     tool_input={"command": "python -m pytest tests/test_auth.py"}),
        AgentMessage(MessageType.TEXT, "Fixed the authentication bug."),
    ]
    summary = process_agent_messages(messages)
    print(f"  Text responses: {len(summary.text_responses)}")
    print(f"  Tools called:   {len(summary.tools_called)}")
    print(f"  Files read:     {sorted(summary.files_read)}")
    print(f"  Files modified: {sorted(summary.files_modified)}")
    print(f"  Commands run:   {summary.commands_run}")
    print(f"  Errors:         {summary.errors or 'none'}")


# === Exercise 3: Multi-Turn Agent Conversation ===
# Problem: Simulate a multi-turn agent conversation where each turn
#   may involve multiple tool calls.

@dataclass
class AgentTurn:
    """A single turn in a multi-turn agent conversation."""
    turn_number: int
    tool_calls: int = 0
    tokens_used: int = 0


class MultiTurnAgent:
    """Simulates a multi-turn agent execution."""

    def __init__(self, max_turns: int = 10) -> None:
        self.max_turns = max_turns
        self.turns: list[AgentTurn] = []
        self.total_tokens = 0

    def execute_turn(self, tool_calls: int = 1,
                     tokens: int = 500) -> bool:
        """Execute a single agent turn. Returns False if max_turns reached."""
        if len(self.turns) >= self.max_turns:
            return False
        turn = AgentTurn(
            turn_number=len(self.turns) + 1,
            tool_calls=tool_calls,
            tokens_used=tokens,
        )
        self.turns.append(turn)
        self.total_tokens += tokens
        return True

    def summary(self) -> dict[str, Any]:
        return {
            "turns_used": len(self.turns),
            "max_turns": self.max_turns,
            "remaining": self.max_turns - len(self.turns),
            "total_tool_calls": sum(t.tool_calls for t in self.turns),
            "total_tokens": self.total_tokens,
            "avg_tokens_per_turn": (
                self.total_tokens // max(len(self.turns), 1)
            ),
        }


def exercise_3():
    """Simulate a multi-turn agent conversation."""
    agent = MultiTurnAgent(max_turns=5)

    # Simulate turns with varying tool usage
    scenarios = [
        (2, 800),   # Exploration: 2 tool calls
        (3, 1200),  # Analysis: 3 tool calls
        (1, 600),   # Planning: 1 tool call
        (4, 1500),  # Implementation: 4 tool calls
        (2, 400),   # Verification: 2 tool calls
    ]
    for tool_calls, tokens in scenarios:
        success = agent.execute_turn(tool_calls, tokens)
        if not success:
            print(f"  Turn rejected: max_turns reached")
            break

    stats = agent.summary()
    print(f"  Turns: {stats['turns_used']}/{stats['max_turns']} "
          f"({stats['remaining']} remaining)")
    print(f"  Tool calls: {stats['total_tool_calls']}")
    print(f"  Tokens: {stats['total_tokens']:,} "
          f"(avg {stats['avg_tokens_per_turn']:,}/turn)")


if __name__ == "__main__":
    print("=== Exercise 1: Query Options ===")
    exercise_1()

    print("=== Exercise 2: Event Processor ===")
    exercise_2()

    print("\n=== Exercise 3: Multi-Turn Agent ===")
    exercise_3()

    print("\nAll exercises completed!")
