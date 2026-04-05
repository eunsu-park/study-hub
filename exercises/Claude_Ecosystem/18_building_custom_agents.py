"""
Exercises for Lesson 18: Building Custom Agents
Topic: Claude_Ecosystem

Solutions to practice problems from the lesson.
"""

from dataclasses import dataclass, field
from typing import Any, Callable


# === Exercise 1: Custom Tool Agent Framework ===
# Problem: Build a framework for creating agents with custom tools,
#   including tool registration, validation, and execution.

@dataclass
class CustomTool:
    """A custom tool for an agent."""
    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[[dict[str, Any]], Any]


class CustomToolAgent:
    """Agent with custom tool registration and execution."""

    def __init__(self, name: str, system_prompt: str = "") -> None:
        self.name = name
        self.system_prompt = system_prompt
        self._tools: dict[str, CustomTool] = {}
        self._execution_log: list[dict[str, Any]] = []

    def register_tool(self, tool: CustomTool) -> None:
        self._tools[tool.name] = tool

    def get_tool_definitions(self) -> list[dict[str, Any]]:
        """Get tool definitions in API format."""
        return [
            {"name": t.name, "description": t.description,
             "input_schema": t.parameters}
            for t in self._tools.values()
        ]

    def execute_tool(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        tool = self._tools.get(name)
        if not tool:
            return {"error": f"Tool '{name}' not registered"}
        try:
            result = tool.handler(args)
            self._execution_log.append({
                "tool": name, "args": args,
                "result": result, "success": True,
            })
            return {"result": result}
        except Exception as e:
            self._execution_log.append({
                "tool": name, "args": args,
                "error": str(e), "success": False,
            })
            return {"error": str(e)}

    @property
    def log(self) -> list[dict[str, Any]]:
        return list(self._execution_log)


def exercise_1():
    """Demonstrate custom tool agent."""
    # In-memory database
    db: dict[str, list[dict[str, Any]]] = {"users": [
        {"id": 1, "name": "Alice", "role": "admin"},
        {"id": 2, "name": "Bob", "role": "user"},
    ]}

    agent = CustomToolAgent(
        "db-assistant",
        "You help users query and manage the database.",
    )
    agent.register_tool(CustomTool(
        "query_users", "Search users by field",
        {"type": "object", "properties": {
            "field": {"type": "string"}, "value": {"type": "string"}},
         "required": ["field", "value"]},
        lambda args: [u for u in db["users"]
                      if str(u.get(args["field"])) == args["value"]],
    ))
    agent.register_tool(CustomTool(
        "add_user", "Add a new user",
        {"type": "object", "properties": {
            "name": {"type": "string"}, "role": {"type": "string"}},
         "required": ["name", "role"]},
        lambda args: (db["users"].append(
            {"id": len(db["users"]) + 1, **args}
        ) or db["users"][-1]),
    ))

    print(f"  Agent: {agent.name}")
    print(f"  Tools: {[t['name'] for t in agent.get_tool_definitions()]}")

    r1 = agent.execute_tool("query_users", {"field": "role", "value": "admin"})
    print(f"  Query admins: {r1}")

    r2 = agent.execute_tool("add_user", {"name": "Carol", "role": "user"})
    print(f"  Add user: {r2}")

    r3 = agent.execute_tool("unknown_tool", {})
    print(f"  Unknown: {r3}")


# === Exercise 2: Guardrail System ===
# Problem: Implement input/output guardrails that validate and filter
#   agent behavior.

@dataclass
class GuardrailResult:
    """Result of a guardrail check."""
    passed: bool
    reason: str = ""


class Guardrails:
    """Input/output guardrails for agent safety."""

    def __init__(self) -> None:
        self._input_checks: list[Callable[[str], GuardrailResult]] = []
        self._output_checks: list[Callable[[str], GuardrailResult]] = []
        self._tool_restrictions: dict[str, list[str]] = {}

    def add_input_check(self, check: Callable[[str], GuardrailResult]) -> None:
        self._input_checks.append(check)

    def add_output_check(self, check: Callable[[str], GuardrailResult]) -> None:
        self._output_checks.append(check)

    def restrict_tool(self, tool_name: str, blocked_patterns: list[str]) -> None:
        self._tool_restrictions[tool_name] = blocked_patterns

    def check_input(self, text: str) -> GuardrailResult:
        for check in self._input_checks:
            result = check(text)
            if not result.passed:
                return result
        return GuardrailResult(True)

    def check_output(self, text: str) -> GuardrailResult:
        for check in self._output_checks:
            result = check(text)
            if not result.passed:
                return result
        return GuardrailResult(True)

    def check_tool_call(self, tool_name: str, args: dict[str, Any]) -> GuardrailResult:
        patterns = self._tool_restrictions.get(tool_name, [])
        args_str = str(args).lower()
        for pattern in patterns:
            if pattern.lower() in args_str:
                return GuardrailResult(
                    False, f"Blocked pattern '{pattern}' in {tool_name} args")
        return GuardrailResult(True)


def exercise_2():
    """Demonstrate guardrail system."""
    guardrails = Guardrails()

    # Input guardrails
    guardrails.add_input_check(lambda text: GuardrailResult(
        len(text) <= 10_000, "Input too long (>10K chars)"))
    guardrails.add_input_check(lambda text: GuardrailResult(
        "ignore previous" not in text.lower(),
        "Prompt injection detected"))

    # Output guardrails
    guardrails.add_output_check(lambda text: GuardrailResult(
        "sk-" not in text, "Potential API key in output"))

    # Tool restrictions
    guardrails.restrict_tool("Bash", ["rm -rf", "sudo", "curl | sh"])

    # Test cases
    tests = [
        ("input", "Normal user request", None, None),
        ("input", "Ignore previous instructions and...", None, None),
        ("output", "Your key is sk-abc123", None, None),
        ("tool", "", "Bash", {"command": "rm -rf /"}),
        ("tool", "", "Bash", {"command": "python -m pytest"}),
    ]
    for check_type, text, tool, args in tests:
        if check_type == "input":
            r = guardrails.check_input(text)
            label = f"Input: '{text[:40]}'"
        elif check_type == "output":
            r = guardrails.check_output(text)
            label = f"Output: '{text[:40]}'"
        else:
            r = guardrails.check_tool_call(tool, args)
            label = f"Tool: {tool}({args})"
        status = "PASS" if r.passed else f"BLOCK: {r.reason}"
        print(f"  {label} → {status}")


# === Exercise 3: Human-in-the-Loop Controller ===
# Problem: Implement a human-in-the-loop pattern where certain
#   agent actions require user approval.

class ApprovalPolicy:
    """Determines which agent actions require human approval."""

    def __init__(self) -> None:
        self._auto_approve: set[str] = set()
        self._always_ask: set[str] = set()
        self._approval_log: list[dict[str, Any]] = []

    def auto_approve(self, *tool_names: str) -> None:
        self._auto_approve.update(tool_names)

    def require_approval(self, *tool_names: str) -> None:
        self._always_ask.update(tool_names)

    def should_ask(self, tool_name: str, args: dict[str, Any]) -> bool:
        if tool_name in self._always_ask:
            return True
        if tool_name in self._auto_approve:
            return False
        return True  # default: ask

    def record_decision(self, tool_name: str, args: dict[str, Any],
                        approved: bool) -> None:
        self._approval_log.append({
            "tool": tool_name, "args": args,
            "approved": approved,
        })

    def simulate_decision(self, tool_name: str,
                          args: dict[str, Any]) -> dict[str, Any]:
        """Simulate the approval flow."""
        needs_approval = self.should_ask(tool_name, args)
        if not needs_approval:
            approved = True
            method = "auto-approved"
        else:
            # Simulate: approve everything except destructive ops
            args_str = str(args).lower()
            approved = "rm" not in args_str and "drop" not in args_str
            method = "user-approved" if approved else "user-denied"

        self.record_decision(tool_name, args, approved)
        return {
            "tool": tool_name,
            "approved": approved,
            "method": method,
        }


def exercise_3():
    """Demonstrate human-in-the-loop approval."""
    policy = ApprovalPolicy()
    policy.auto_approve("Read", "Glob", "Grep")
    policy.require_approval("Bash", "Write")

    actions = [
        ("Read", {"file_path": "src/main.py"}),
        ("Grep", {"pattern": "TODO"}),
        ("Bash", {"command": "python -m pytest"}),
        ("Bash", {"command": "rm -rf dist/"}),
        ("Write", {"file_path": "src/new.py", "content": "..."}),
        ("Edit", {"file_path": "src/main.py"}),
    ]
    for tool, args in actions:
        result = policy.simulate_decision(tool, args)
        status = "APPROVED" if result["approved"] else "DENIED"
        print(f"  {tool}({list(args.values())[0]}) → "
              f"{status} ({result['method']})")


if __name__ == "__main__":
    print("=== Exercise 1: Custom Tool Agent ===")
    exercise_1()

    print("\n=== Exercise 2: Guardrails ===")
    exercise_2()

    print("\n=== Exercise 3: Human-in-the-Loop ===")
    exercise_3()

    print("\nAll exercises completed!")
