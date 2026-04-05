"""
Exercises for Lesson 13: Building Custom MCP Servers
Topic: Claude_Ecosystem

Solutions to practice problems from the lesson.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable


# === Exercise 1: MCP Server Builder ===
# Problem: Build a fluent API for constructing MCP server definitions
#   with resources, tools, and prompts.

@dataclass
class ResourceDef:
    uri: str
    name: str
    description: str
    handler: Callable[[], str] | None = None


@dataclass
class ToolDef:
    name: str
    description: str
    input_schema: dict[str, Any] = field(default_factory=dict)
    handler: Callable[[dict[str, Any]], Any] | None = None


@dataclass
class PromptDef:
    name: str
    description: str
    template: str
    arguments: list[dict[str, str]] = field(default_factory=list)


class MCPServerBuilder:
    """Fluent builder for MCP server definitions."""

    def __init__(self, name: str) -> None:
        self.name = name
        self._resources: list[ResourceDef] = []
        self._tools: list[ToolDef] = []
        self._prompts: list[PromptDef] = []

    def add_resource(self, uri: str, name: str, description: str,
                     handler: Callable[[], str] | None = None) -> "MCPServerBuilder":
        self._resources.append(ResourceDef(uri, name, description, handler))
        return self

    def add_tool(self, name: str, description: str,
                 schema: dict[str, Any] | None = None,
                 handler: Callable[[dict[str, Any]], Any] | None = None,
                 ) -> "MCPServerBuilder":
        self._tools.append(ToolDef(name, description, schema or {}, handler))
        return self

    def add_prompt(self, name: str, description: str, template: str,
                   arguments: list[dict[str, str]] | None = None,
                   ) -> "MCPServerBuilder":
        self._prompts.append(PromptDef(name, description, template,
                                       arguments or []))
        return self

    def build(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "capabilities": {
                "resources": len(self._resources) > 0,
                "tools": len(self._tools) > 0,
                "prompts": len(self._prompts) > 0,
            },
            "resources": [{"uri": r.uri, "name": r.name,
                           "description": r.description}
                          for r in self._resources],
            "tools": [{"name": t.name, "description": t.description,
                       "inputSchema": t.input_schema}
                      for t in self._tools],
            "prompts": [{"name": p.name, "description": p.description,
                         "arguments": p.arguments}
                        for p in self._prompts],
        }


def exercise_1():
    """Demonstrate MCP server builder."""
    server = (MCPServerBuilder("todo-manager")
              .add_resource("todo://list", "Todo List",
                           "Current list of todo items")
              .add_tool("add_todo", "Add a new todo item", {
                  "type": "object",
                  "properties": {
                      "title": {"type": "string"},
                      "priority": {"type": "string",
                                   "enum": ["low", "medium", "high"]},
                  },
                  "required": ["title"],
              })
              .add_tool("complete_todo", "Mark a todo as complete", {
                  "type": "object",
                  "properties": {"id": {"type": "integer"}},
                  "required": ["id"],
              })
              .add_prompt("daily_summary", "Generate daily summary",
                         "Summarize todos: {{todos}}", [
                             {"name": "todos", "description": "Todo list"}
                         ])
              .build())

    print(f"  Server: {server['name']}")
    print(f"  Capabilities: {server['capabilities']}")
    print(f"  Resources: {len(server['resources'])}")
    print(f"  Tools: {len(server['tools'])}")
    print(f"  Prompts: {len(server['prompts'])}")


# === Exercise 2: Tool Handler with Validation ===
# Problem: Implement tool handlers with input validation and error handling.

class ToolRegistry:
    """Registry for MCP tool handlers with validation."""

    def __init__(self) -> None:
        self._handlers: dict[str, Callable[[dict], Any]] = {}
        self._schemas: dict[str, dict[str, Any]] = {}

    def register(self, name: str, schema: dict[str, Any],
                 handler: Callable[[dict], Any]) -> None:
        self._handlers[name] = handler
        self._schemas[name] = schema

    def validate(self, name: str, arguments: dict[str, Any]) -> list[str]:
        """Validate arguments against the tool's input schema."""
        errors: list[str] = []
        schema = self._schemas.get(name)
        if not schema:
            return [f"Unknown tool: {name}"]

        required = schema.get("required", [])
        properties = schema.get("properties", {})

        for req in required:
            if req not in arguments:
                errors.append(f"Missing required field: {req}")

        for key, value in arguments.items():
            if key not in properties:
                errors.append(f"Unknown field: {key}")
                continue
            prop = properties[key]
            expected_type = prop.get("type")
            if expected_type == "string" and not isinstance(value, str):
                errors.append(f"{key}: expected string, got {type(value).__name__}")
            elif expected_type == "integer" and not isinstance(value, int):
                errors.append(f"{key}: expected integer, got {type(value).__name__}")
            if "enum" in prop and value not in prop["enum"]:
                errors.append(f"{key}: must be one of {prop['enum']}")

        return errors

    def call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Call a tool with validation."""
        errors = self.validate(name, arguments)
        if errors:
            return {"error": True, "messages": errors}

        handler = self._handlers.get(name)
        if not handler:
            return {"error": True, "messages": [f"No handler for: {name}"]}

        try:
            result = handler(arguments)
            return {"error": False, "result": result}
        except Exception as e:
            return {"error": True, "messages": [str(e)]}


def exercise_2():
    """Demonstrate tool handler registration and invocation."""
    registry = ToolRegistry()

    # Mock todo storage
    todos: list[dict[str, Any]] = []

    def add_todo(args: dict[str, Any]) -> dict[str, Any]:
        todo = {"id": len(todos) + 1, "title": args["title"],
                "priority": args.get("priority", "medium"), "done": False}
        todos.append(todo)
        return todo

    registry.register("add_todo", {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "priority": {"type": "string", "enum": ["low", "medium", "high"]},
        },
        "required": ["title"],
    }, add_todo)

    test_calls = [
        ("add_todo", {"title": "Write docs", "priority": "high"}),
        ("add_todo", {"title": "Review PR"}),
        ("add_todo", {"priority": "low"}),           # missing title
        ("add_todo", {"title": "Test", "priority": "urgent"}),  # invalid enum
        ("unknown_tool", {"arg": "value"}),
    ]
    for name, args in test_calls:
        result = registry.call(name, args)
        if result.get("error"):
            print(f"  {name}({args}) → ERROR: {result['messages']}")
        else:
            print(f"  {name}({args}) → {result['result']}")


# === Exercise 3: MCP Error Code Handler ===
# Problem: Implement MCP standard error codes and responses.

MCP_ERRORS = {
    -32700: "Parse error",
    -32600: "Invalid request",
    -32601: "Method not found",
    -32602: "Invalid params",
    -32603: "Internal error",
    -32000: "Server error",
}


def create_error_response(request_id: int, code: int,
                          detail: str = "") -> dict[str, Any]:
    """Create a standard MCP error response."""
    message = MCP_ERRORS.get(code, "Unknown error")
    if detail:
        message = f"{message}: {detail}"
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "error": {"code": code, "message": message},
    }


def handle_request(request: dict[str, Any],
                   supported_methods: set[str]) -> dict[str, Any]:
    """Handle an incoming MCP request with proper error responses."""
    if "jsonrpc" not in request or request.get("jsonrpc") != "2.0":
        return create_error_response(
            request.get("id", 0), -32600, "Missing jsonrpc field")

    method = request.get("method")
    if not method:
        return create_error_response(request.get("id", 0), -32600, "No method")

    if method not in supported_methods:
        return create_error_response(request.get("id", 0), -32601, method)

    return {
        "jsonrpc": "2.0",
        "id": request.get("id", 0),
        "result": {"status": "ok", "method": method},
    }


def exercise_3():
    """Demonstrate MCP error handling."""
    supported = {"initialize", "tools/list", "tools/call", "resources/read"}

    requests = [
        {"jsonrpc": "2.0", "method": "initialize", "id": 1},
        {"jsonrpc": "2.0", "method": "tools/list", "id": 2},
        {"jsonrpc": "2.0", "method": "unknown/method", "id": 3},
        {"method": "tools/list", "id": 4},  # missing jsonrpc
        {"jsonrpc": "2.0", "id": 5},        # missing method
    ]
    for req in requests:
        resp = handle_request(req, supported)
        status = "OK" if "result" in resp else f"ERROR {resp['error']['code']}"
        method = req.get("method", "N/A")
        print(f"  {method} → {status}")


if __name__ == "__main__":
    print("=== Exercise 1: Server Builder ===")
    exercise_1()

    print("\n=== Exercise 2: Tool Handlers ===")
    exercise_2()

    print("\n=== Exercise 3: Error Handling ===")
    exercise_3()

    print("\nAll exercises completed!")
