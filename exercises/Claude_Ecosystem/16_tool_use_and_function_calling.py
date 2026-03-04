"""
Exercises for Lesson 16: Tool Use and Function Calling
Topic: Claude_Ecosystem

Solutions to practice problems from the lesson.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable


# === Exercise 1: Tool Schema Definition ===
# Problem: Define tools with proper JSON Schema input specifications
#   and generate the tool definition array for the API.

@dataclass
class ToolParam:
    name: str
    param_type: str
    description: str
    required: bool = True
    enum: list[str] | None = None
    default: Any = None


@dataclass
class ToolDefinition:
    name: str
    description: str
    params: list[ToolParam] = field(default_factory=list)

    def to_api_format(self) -> dict[str, Any]:
        properties: dict[str, Any] = {}
        required: list[str] = []
        for p in self.params:
            prop: dict[str, Any] = {"type": p.param_type,
                                     "description": p.description}
            if p.enum:
                prop["enum"] = p.enum
            if p.default is not None:
                prop["default"] = p.default
            properties[p.name] = prop
            if p.required:
                required.append(p.name)
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        }


def exercise_1():
    """Demonstrate tool schema definition."""
    tools = [
        ToolDefinition("get_stock_price", "Get current stock price", [
            ToolParam("symbol", "string", "Stock ticker symbol (e.g., AAPL)"),
            ToolParam("currency", "string", "Currency for price",
                      required=False, enum=["USD", "EUR", "KRW"],
                      default="USD"),
        ]),
        ToolDefinition("search_database", "Search records in database", [
            ToolParam("query", "string", "Search query"),
            ToolParam("table", "string", "Table name",
                      enum=["users", "products", "orders"]),
            ToolParam("limit", "integer", "Max results",
                      required=False, default=10),
        ]),
    ]
    for tool in tools:
        api_fmt = tool.to_api_format()
        print(f"  {api_fmt['name']}:")
        schema = api_fmt["input_schema"]
        print(f"    Required: {schema['required']}")
        for name, prop in schema["properties"].items():
            extras = []
            if "enum" in prop:
                extras.append(f"enum={prop['enum']}")
            if "default" in prop:
                extras.append(f"default={prop['default']}")
            extra_str = f" ({', '.join(extras)})" if extras else ""
            print(f"    - {name}: {prop['type']}{extra_str}")
        print()


# === Exercise 2: Agentic Tool Loop ===
# Problem: Implement the core agentic loop that processes tool calls
#   from Claude's responses and feeds results back.

# Mock tool implementations
MOCK_DATA: dict[str, Any] = {
    "AAPL": {"price": 178.50, "change": -1.2},
    "GOOGL": {"price": 141.80, "change": 0.5},
    "weather:Seoul": {"temp": 15, "condition": "cloudy"},
    "weather:Paris": {"temp": 22, "condition": "sunny"},
}

TOOL_HANDLERS: dict[str, Callable[[dict], Any]] = {
    "get_stock_price": lambda args: MOCK_DATA.get(
        args["symbol"], {"error": "Symbol not found"}),
    "get_weather": lambda args: MOCK_DATA.get(
        f"weather:{args['city']}", {"error": "City not found"}),
}


@dataclass
class ToolUseBlock:
    """A tool_use content block from Claude's response."""
    tool_id: str
    name: str
    input_args: dict[str, Any]


@dataclass
class SimulatedResponse:
    """Simulated Claude API response."""
    stop_reason: str  # "end_turn" or "tool_use"
    content: list[dict[str, Any]]


def process_tool_calls(
    tool_uses: list[ToolUseBlock],
) -> list[dict[str, Any]]:
    """Execute tool calls and format results for the API."""
    results: list[dict[str, Any]] = []
    for tool_use in tool_uses:
        handler = TOOL_HANDLERS.get(tool_use.name)
        if handler:
            result = handler(tool_use.input_args)
            content = json.dumps(result)
        else:
            content = json.dumps({"error": f"Unknown tool: {tool_use.name}"})
        results.append({
            "type": "tool_result",
            "tool_use_id": tool_use.tool_id,
            "content": content,
        })
    return results


def simulate_agentic_loop(user_query: str, max_turns: int = 5) -> list[str]:
    """Simulate the agentic loop with tool calls.

    In a real implementation, each iteration calls the Claude API.
    Here we simulate Claude's tool-calling behavior.
    """
    log: list[str] = []
    log.append(f"User: {user_query}")

    # Simulate: Claude decides to call tools based on the query
    simulated_tool_calls = []
    if "stock" in user_query.lower() or "price" in user_query.lower():
        simulated_tool_calls.append(
            ToolUseBlock("call_1", "get_stock_price", {"symbol": "AAPL"}))
    if "weather" in user_query.lower():
        simulated_tool_calls.append(
            ToolUseBlock("call_2", "get_weather", {"city": "Seoul"}))

    if simulated_tool_calls:
        log.append(f"Claude: [tool_use] {len(simulated_tool_calls)} call(s)")
        results = process_tool_calls(simulated_tool_calls)
        for tc, res in zip(simulated_tool_calls, results):
            log.append(f"  → {tc.name}({tc.input_args}) = {res['content']}")
        log.append("Claude: Based on the tool results, here is your answer.")
    else:
        log.append("Claude: I can answer this directly without tools.")

    return log


def exercise_2():
    """Demonstrate the agentic tool loop."""
    queries = [
        "What is the current stock price of AAPL?",
        "What's the weather like in Seoul?",
        "Tell me a joke",
    ]
    for query in queries:
        log = simulate_agentic_loop(query)
        for entry in log:
            print(f"  {entry}")
        print()


# === Exercise 3: Parallel Tool Execution ===
# Problem: Handle parallel tool calls where Claude requests
#   multiple tools simultaneously.

def execute_parallel_tools(
    tool_uses: list[ToolUseBlock],
) -> dict[str, Any]:
    """Execute multiple tool calls and aggregate results.

    In production, these could be executed concurrently with asyncio.
    """
    results: dict[str, Any] = {}
    errors: list[str] = []

    for tool_use in tool_uses:
        handler = TOOL_HANDLERS.get(tool_use.name)
        if handler:
            try:
                result = handler(tool_use.input_args)
                results[tool_use.tool_id] = {
                    "tool": tool_use.name,
                    "input": tool_use.input_args,
                    "output": result,
                    "success": True,
                }
            except Exception as e:
                errors.append(f"{tool_use.name}: {e}")
                results[tool_use.tool_id] = {
                    "tool": tool_use.name,
                    "output": str(e),
                    "success": False,
                }
        else:
            errors.append(f"Unknown tool: {tool_use.name}")

    return {
        "total": len(tool_uses),
        "succeeded": sum(1 for r in results.values() if r["success"]),
        "failed": len(errors),
        "results": results,
        "errors": errors,
    }


def exercise_3():
    """Demonstrate parallel tool execution."""
    parallel_calls = [
        ToolUseBlock("p1", "get_stock_price", {"symbol": "AAPL"}),
        ToolUseBlock("p2", "get_stock_price", {"symbol": "GOOGL"}),
        ToolUseBlock("p3", "get_weather", {"city": "Seoul"}),
        ToolUseBlock("p4", "get_weather", {"city": "Paris"}),
    ]
    summary = execute_parallel_tools(parallel_calls)
    print(f"  Parallel execution: {summary['succeeded']}/{summary['total']} "
          f"succeeded")
    for call_id, result in summary["results"].items():
        print(f"  {call_id} [{result['tool']}]: {result['output']}")


if __name__ == "__main__":
    print("=== Exercise 1: Tool Schema ===")
    exercise_1()

    print("=== Exercise 2: Agentic Loop ===")
    exercise_2()

    print("=== Exercise 3: Parallel Execution ===")
    exercise_3()

    print("\nAll exercises completed!")
