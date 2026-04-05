"""
Exercises for Lesson 12: Model Context Protocol (MCP)
Topic: Claude_Ecosystem

Solutions to practice problems from the lesson.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


# === Exercise 1: MCP Message Format Builder ===
# Problem: Construct valid MCP JSON-RPC messages for different operations
#   (initialize, list resources, call tool).

@dataclass
class MCPMessage:
    """A JSON-RPC 2.0 message for the MCP protocol."""
    method: str
    params: dict[str, Any] = field(default_factory=dict)
    id: int = 1

    def to_json(self) -> str:
        msg = {
            "jsonrpc": "2.0",
            "method": self.method,
            "id": self.id,
        }
        if self.params:
            msg["params"] = self.params
        return json.dumps(msg, indent=2)

    @staticmethod
    def response(id: int, result: Any) -> str:
        return json.dumps({
            "jsonrpc": "2.0",
            "id": id,
            "result": result,
        }, indent=2)

    @staticmethod
    def error(id: int, code: int, message: str) -> str:
        return json.dumps({
            "jsonrpc": "2.0",
            "id": id,
            "error": {"code": code, "message": message},
        }, indent=2)


def exercise_1():
    """Demonstrate MCP message construction."""
    messages = [
        MCPMessage("initialize", {
            "protocolVersion": "2025-03-26",
            "capabilities": {"tools": {}, "resources": {}},
            "clientInfo": {"name": "claude-code", "version": "1.0"},
        }, id=1),
        MCPMessage("tools/list", id=2),
        MCPMessage("tools/call", {
            "name": "get_weather",
            "arguments": {"city": "Seoul", "units": "celsius"},
        }, id=3),
        MCPMessage("resources/read", {
            "uri": "file:///data/config.json",
        }, id=4),
    ]
    for msg in messages:
        print(f"  {msg.method}:")
        print(f"    {msg.to_json()}\n")

    # Example response and error
    print("  Success response:")
    print(f"    {MCPMessage.response(3, {'temperature': 15, 'condition': 'cloudy'})}")
    print("  Error response:")
    print(f"    {MCPMessage.error(4, -32601, 'Resource not found')}")


# === Exercise 2: MCP Resource and Tool Schema Builder ===
# Problem: Define MCP resources and tools with proper JSON Schema definitions.

@dataclass
class MCPResource:
    """An MCP resource definition."""
    uri: str
    name: str
    description: str
    mime_type: str = "text/plain"

    def to_dict(self) -> dict[str, str]:
        return {
            "uri": self.uri,
            "name": self.name,
            "description": self.description,
            "mimeType": self.mime_type,
        }


@dataclass
class MCPToolParam:
    """A parameter for an MCP tool."""
    name: str
    param_type: str  # "string", "number", "boolean", "array", "object"
    description: str
    required: bool = True
    enum: list[str] | None = None


@dataclass
class MCPTool:
    """An MCP tool definition."""
    name: str
    description: str
    params: list[MCPToolParam] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        properties: dict[str, Any] = {}
        required: list[str] = []
        for p in self.params:
            prop: dict[str, Any] = {
                "type": p.param_type,
                "description": p.description,
            }
            if p.enum:
                prop["enum"] = p.enum
            properties[p.name] = prop
            if p.required:
                required.append(p.name)

        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        }


def exercise_2():
    """Demonstrate MCP resource and tool schema building."""
    resources = [
        MCPResource("db://users/schema", "Users Schema",
                    "Database schema for users table", "application/json"),
        MCPResource("file:///docs/api.md", "API Docs",
                    "API documentation", "text/markdown"),
    ]
    print("  Resources:")
    for r in resources:
        print(f"    {json.dumps(r.to_dict())}")

    tools = [
        MCPTool("query_database", "Execute a read-only SQL query", [
            MCPToolParam("sql", "string", "SQL SELECT query"),
            MCPToolParam("limit", "number", "Max rows to return",
                         required=False),
        ]),
        MCPTool("get_weather", "Get current weather for a city", [
            MCPToolParam("city", "string", "City name"),
            MCPToolParam("units", "string", "Temperature units",
                         enum=["celsius", "fahrenheit"]),
        ]),
    ]
    print("\n  Tools:")
    for t in tools:
        print(f"    {json.dumps(t.to_dict(), indent=4)}")


# === Exercise 3: MCP Server Registry ===
# Problem: Manage a registry of MCP server configurations for
#   Claude Code's settings.

@dataclass
class MCPServerConfig:
    """Configuration for connecting to an MCP server."""
    name: str
    transport: str  # "stdio" or "sse"
    command: str | None = None      # for stdio
    args: list[str] = field(default_factory=list)
    url: str | None = None          # for sse
    env: dict[str, str] = field(default_factory=dict)


class MCPRegistry:
    """Registry of MCP server configurations."""

    def __init__(self) -> None:
        self._servers: dict[str, MCPServerConfig] = {}

    def register(self, config: MCPServerConfig) -> None:
        self._servers[config.name] = config

    def remove(self, name: str) -> bool:
        if name in self._servers:
            del self._servers[name]
            return True
        return False

    def get(self, name: str) -> MCPServerConfig | None:
        return self._servers.get(name)

    def to_settings_json(self) -> dict[str, Any]:
        """Generate the mcpServers section for Claude Code settings."""
        servers: dict[str, Any] = {}
        for name, config in self._servers.items():
            entry: dict[str, Any] = {}
            if config.transport == "stdio":
                entry["command"] = config.command
                entry["args"] = config.args
            else:
                entry["url"] = config.url
            if config.env:
                entry["env"] = config.env
            servers[name] = entry
        return {"mcpServers": servers}

    def list_servers(self) -> list[dict[str, str]]:
        return [
            {"name": c.name, "transport": c.transport,
             "endpoint": c.command or c.url or ""}
            for c in self._servers.values()
        ]


def exercise_3():
    """Demonstrate MCP server registry management."""
    registry = MCPRegistry()

    registry.register(MCPServerConfig(
        "filesystem", "stdio",
        command="npx",
        args=["-y", "@modelcontextprotocol/server-filesystem", "/data"],
    ))
    registry.register(MCPServerConfig(
        "postgres", "stdio",
        command="npx",
        args=["-y", "@modelcontextprotocol/server-postgres"],
        env={"DATABASE_URL": "postgresql://localhost/mydb"},
    ))
    registry.register(MCPServerConfig(
        "custom-api", "sse",
        url="http://localhost:8080/mcp",
    ))

    print("  Registered servers:")
    for s in registry.list_servers():
        print(f"    {s['name']} ({s['transport']}): {s['endpoint']}")

    settings = registry.to_settings_json()
    print(f"\n  Settings JSON:")
    print(f"    {json.dumps(settings, indent=4)}")


if __name__ == "__main__":
    print("=== Exercise 1: MCP Messages ===")
    exercise_1()

    print("\n=== Exercise 2: Resource & Tool Schemas ===")
    exercise_2()

    print("\n=== Exercise 3: Server Registry ===")
    exercise_3()

    print("\nAll exercises completed!")
