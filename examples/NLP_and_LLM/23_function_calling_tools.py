"""
23. Function Calling and Tool Use Example

Tool registry, schema generation, parallel execution, and error handling
"""

import json
import inspect
import time
from typing import Any, Callable, NamedTuple
from dataclasses import dataclass
from functools import wraps

print("=" * 60)
print("Function Calling and Tool Use")
print("=" * 60)


# ============================================
# 1. Tool Definition and Registry
# ============================================
print("\n[1] Tool Registry")
print("-" * 40)


@dataclass
class ToolDefinition:
    name: str
    description: str
    parameters: dict
    handler: Callable


class ToolRegistry:
    """Registry for managing and auto-generating tool schemas."""

    def __init__(self):
        self._tools: dict[str, ToolDefinition] = {}

    def tool(self, description: str):
        """Decorator to register a function as a tool."""
        def decorator(func: Callable) -> Callable:
            hints = func.__annotations__
            sig = inspect.signature(func)

            properties = {}
            required = []
            type_map = {str: "string", int: "integer", float: "number", bool: "boolean"}

            for param_name, param in sig.parameters.items():
                if param_name == "return":
                    continue
                hint = hints.get(param_name, str)
                json_type = type_map.get(hint, "string")
                properties[param_name] = {"type": json_type, "description": f"Parameter: {param_name}"}

                if param.default is inspect.Parameter.empty:
                    required.append(param_name)

            self._tools[func.__name__] = ToolDefinition(
                name=func.__name__,
                description=description,
                parameters={"type": "object", "properties": properties, "required": required},
                handler=func,
            )

            @wraps(func)
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)
            return wrapper
        return decorator

    def get_openai_tools(self) -> list[dict]:
        return [
            {"type": "function", "function": {"name": t.name, "description": t.description, "parameters": t.parameters}}
            for t in self._tools.values()
        ]

    def get_anthropic_tools(self) -> list[dict]:
        return [
            {"name": t.name, "description": t.description, "input_schema": t.parameters}
            for t in self._tools.values()
        ]

    def execute(self, name: str, arguments: dict) -> Any:
        if name not in self._tools:
            raise ValueError(f"Unknown tool: {name}")
        return self._tools[name].handler(**arguments)

    def list_tools(self) -> list[str]:
        return list(self._tools.keys())


registry = ToolRegistry()


@registry.tool("Get current weather for a city")
def get_weather(city: str, unit: str) -> dict:
    """Simulated weather API."""
    temps = {"Tokyo": 22, "New York": 18, "London": 15, "Seoul": 20}
    temp = temps.get(city, 20)
    if unit == "fahrenheit":
        temp = int(temp * 9 / 5 + 32)
    return {"city": city, "temperature": temp, "unit": unit, "condition": "partly cloudy"}


@registry.tool("Search for flights between two cities")
def search_flights(origin: str, destination: str, date: str) -> dict:
    """Simulated flight search."""
    return {
        "flights": [
            {"airline": "United", "departure": "08:30", "price": 320},
            {"airline": "Delta", "departure": "14:15", "price": 285},
        ],
        "origin": origin, "destination": destination, "date": date,
    }


@registry.tool("Calculate the total cost of items in a shopping cart")
def calculate_total(items: str, tax_rate: float) -> dict:
    """Calculate shopping cart total."""
    # items is a JSON string of [{name, price, quantity}]
    item_list = json.loads(items) if isinstance(items, str) else items
    subtotal = sum(i.get("price", 0) * i.get("quantity", 1) for i in item_list)
    tax = round(subtotal * tax_rate, 2)
    return {"subtotal": round(subtotal, 2), "tax": tax, "total": round(subtotal + tax, 2)}


print("Registered tools:")
for tool_name in registry.list_tools():
    print(f"  - {tool_name}")

print("\nOpenAI format (first tool):")
print(json.dumps(registry.get_openai_tools()[0], indent=2))


# ============================================
# 2. Tool Execution
# ============================================
print("\n[2] Tool Execution")
print("-" * 40)

result = registry.execute("get_weather", {"city": "Tokyo", "unit": "celsius"})
print(f"Weather in Tokyo: {result}")

result = registry.execute("search_flights", {
    "origin": "NYC", "destination": "Tokyo", "date": "2026-04-15"
})
print(f"Flights found: {len(result['flights'])}")

result = registry.execute("calculate_total", {
    "items": json.dumps([
        {"name": "Widget", "price": 29.99, "quantity": 2},
        {"name": "Gadget", "price": 49.99, "quantity": 1},
    ]),
    "tax_rate": 0.085,
})
print(f"Cart total: ${result['total']}")


# ============================================
# 3. Simulated Function Calling Loop
# ============================================
print("\n[3] Function Calling Loop (simulated)")
print("-" * 40)


def simulate_function_calling(user_query: str, registry: ToolRegistry) -> str:
    """Simulate the LLM function calling loop without actual LLM."""
    # Simulate LLM deciding which tool to call based on keywords
    query_lower = user_query.lower()

    tool_calls = []
    if "weather" in query_lower:
        cities = []
        for city in ["Tokyo", "New York", "London", "Seoul"]:
            if city.lower() in query_lower:
                cities.append(city)
        if not cities:
            cities = ["Tokyo"]
        for city in cities:
            tool_calls.append(("get_weather", {"city": city, "unit": "celsius"}))

    if "flight" in query_lower:
        tool_calls.append(("search_flights", {
            "origin": "NYC", "destination": "Tokyo", "date": "2026-04-15"
        }))

    if not tool_calls:
        return "I can help with weather and flights. What would you like to know?"

    # Execute tools
    results = []
    for func_name, args in tool_calls:
        print(f"  Calling {func_name}({args})")
        result = registry.execute(func_name, args)
        results.append(f"{func_name}: {json.dumps(result)}")

    return f"Based on {len(results)} tool call(s): " + "; ".join(results)


queries = [
    "What's the weather in Tokyo?",
    "Compare weather in Tokyo and New York",
    "Find me flights from NYC to Tokyo",
    "What's the weather in Seoul and find flights to Tokyo",
]

for q in queries:
    print(f"\nQuery: {q}")
    answer = simulate_function_calling(q, registry)
    print(f"Answer: {answer[:120]}...")


# ============================================
# 4. Error Handling
# ============================================
print("\n\n[4] Robust Tool Execution")
print("-" * 40)


class ToolResult(NamedTuple):
    success: bool
    data: Any
    error: str | None


class RobustToolExecutor:
    """Execute tools with error handling and retries."""

    def __init__(self, max_retries: int = 2):
        self.max_retries = max_retries

    def execute(self, name: str, arguments: dict,
                registry_tools: dict[str, Callable]) -> ToolResult:
        if name not in registry_tools:
            return ToolResult(False, None, f"Unknown tool: {name}")

        func = registry_tools[name]
        for attempt in range(self.max_retries + 1):
            try:
                result = func(**arguments)
                return ToolResult(True, result, None)
            except TypeError as e:
                return ToolResult(False, None, f"Invalid arguments: {e}")
            except Exception as e:
                if attempt < self.max_retries:
                    time.sleep(0.01)
                    continue
                return ToolResult(False, None, f"Failed after {self.max_retries + 1} attempts: {e}")

    def format_for_llm(self, result: ToolResult) -> str:
        if result.success:
            return json.dumps(result.data)
        return json.dumps({"error": result.error, "suggestion": "Try different parameters."})


executor = RobustToolExecutor()

# Successful call
r = executor.execute("get_weather", {"city": "Seoul", "unit": "celsius"},
                      {"get_weather": get_weather})
print(f"Success: {r.success}, Data: {r.data}")

# Unknown tool
r = executor.execute("unknown_tool", {}, {"get_weather": get_weather})
print(f"Unknown tool: {executor.format_for_llm(r)}")

# Wrong arguments
r = executor.execute("get_weather", {"wrong_param": "value"},
                      {"get_weather": get_weather})
print(f"Bad args: {executor.format_for_llm(r)}")


# ============================================
# 5. Tool Chain
# ============================================
print("\n[5] Sequential Tool Chain")
print("-" * 40)


class ToolChain:
    """Execute tools in sequence, passing context."""

    def __init__(self):
        self.steps: list[tuple[str, dict]] = []

    def add_step(self, tool_name: str, args: dict):
        self.steps.append((tool_name, args))
        return self

    def execute(self, registry: ToolRegistry) -> list[dict]:
        results = []
        for i, (tool_name, args) in enumerate(self.steps):
            print(f"  Step {i+1}: {tool_name}")
            result = registry.execute(tool_name, args)
            results.append({"step": i + 1, "tool": tool_name, "result": result})
        return results


chain = ToolChain()
chain.add_step("get_weather", {"city": "Tokyo", "unit": "celsius"})
chain.add_step("search_flights", {"origin": "NYC", "destination": "Tokyo", "date": "2026-04-15"})
chain.add_step("calculate_total", {
    "items": json.dumps([{"name": "Flight", "price": 285, "quantity": 1}]),
    "tax_rate": 0.0,
})

chain_results = chain.execute(registry)
for r in chain_results:
    print(f"  Step {r['step']} ({r['tool']}): {json.dumps(r['result'])[:80]}")

print("\n" + "=" * 60)
print("Function Calling and Tool Use example complete!")
print("=" * 60)
