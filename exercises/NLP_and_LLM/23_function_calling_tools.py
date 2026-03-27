"""
Exercises for Lesson 23: Function Calling and Tool Use
Topic: NLP_and_LLM

Practice problems for tool design, schema generation, and error handling.
"""

import json
import inspect
from typing import Any, Callable
from dataclasses import dataclass
from functools import wraps


# === Exercise 1: Auto Schema Generator ===
# Problem: Build a tool that auto-generates JSON Schema from Python
# function signatures, including type hints and docstrings.

def exercise_1():
    """Auto-generate JSON Schema from function signatures."""
    print("=" * 60)
    print("Exercise 1: Auto Schema Generator")
    print("=" * 60)

    # TODO: Implement schema generation from function signature
    def generate_schema(func: Callable) -> dict:
        """Generate OpenAI-compatible tool schema from a Python function."""
        sig = inspect.signature(func)
        hints = func.__annotations__
        doc = func.__doc__ or ""

        type_map = {
            str: "string", int: "integer", float: "number",
            bool: "boolean", list: "array", dict: "object",
        }

        properties = {}
        required = []

        for name, param in sig.parameters.items():
            if name == "return":
                continue

            hint = hints.get(name, str)
            json_type = type_map.get(hint, "string")
            properties[name] = {
                "type": json_type,
                "description": f"Parameter '{name}'",
            }

            if param.default is inspect.Parameter.empty:
                required.append(name)
            elif param.default is not None:
                properties[name]["default"] = param.default

        return {
            "type": "function",
            "function": {
                "name": func.__name__,
                "description": doc.strip().split("\n")[0] if doc else "",
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }

    # Test functions
    def get_weather(city: str, unit: str = "celsius") -> dict:
        """Get current weather for a city."""
        return {"city": city, "temp": 22, "unit": unit}

    def search_products(query: str, category: str, max_results: int = 10,
                        in_stock: bool = True) -> list:
        """Search products in the catalog by query and category."""
        return []

    def calculate_shipping(weight: float, destination: str, express: bool = False) -> dict:
        """Calculate shipping cost based on weight and destination."""
        return {}

    for func in [get_weather, search_products, calculate_shipping]:
        schema = generate_schema(func)
        print(f"\n{func.__name__}:")
        print(json.dumps(schema, indent=2))


# === Exercise 2: Tool Validation Decorator ===
# Problem: Create a decorator that validates tool inputs against
# constraints before execution.

def exercise_2():
    """Build input validation decorator for tools."""
    print("\n" + "=" * 60)
    print("Exercise 2: Tool Validation Decorator")
    print("=" * 60)

    @dataclass
    class ParamConstraint:
        min_val: float | None = None
        max_val: float | None = None
        min_length: int | None = None
        max_length: int | None = None
        choices: list | None = None

    # TODO: Implement validation decorator
    def validated(constraints: dict[str, ParamConstraint]):
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(**kwargs):
                errors = []
                for param, constraint in constraints.items():
                    if param not in kwargs:
                        continue
                    value = kwargs[param]

                    if constraint.min_val is not None and isinstance(value, (int, float)):
                        if value < constraint.min_val:
                            errors.append(f"{param}: {value} < min {constraint.min_val}")
                    if constraint.max_val is not None and isinstance(value, (int, float)):
                        if value > constraint.max_val:
                            errors.append(f"{param}: {value} > max {constraint.max_val}")
                    if constraint.min_length is not None and isinstance(value, str):
                        if len(value) < constraint.min_length:
                            errors.append(f"{param}: length {len(value)} < min {constraint.min_length}")
                    if constraint.max_length is not None and isinstance(value, str):
                        if len(value) > constraint.max_length:
                            errors.append(f"{param}: length {len(value)} > max {constraint.max_length}")
                    if constraint.choices is not None:
                        if value not in constraint.choices:
                            errors.append(f"{param}: '{value}' not in {constraint.choices}")

                if errors:
                    return {"error": "Validation failed", "details": errors}
                return func(**kwargs)
            return wrapper
        return decorator

    @validated({
        "query": ParamConstraint(min_length=1, max_length=500),
        "limit": ParamConstraint(min_val=1, max_val=100),
        "sort_by": ParamConstraint(choices=["relevance", "date", "price"]),
    })
    def search(query: str, limit: int = 10, sort_by: str = "relevance") -> dict:
        """Search with validated parameters."""
        return {"results": [f"Result for: {query}"], "limit": limit, "sort": sort_by}

    # Valid call
    result = search(query="python tutorial", limit=5, sort_by="relevance")
    print(f"Valid: {result}")

    # Invalid calls
    result = search(query="", limit=200, sort_by="popularity")
    print(f"Invalid: {result}")

    result = search(query="x" * 600, limit=0)
    print(f"Invalid: {result}")


# === Exercise 3: Multi-Tool Router ===
# Problem: Build a router that selects the best tool based on
# the user's query using keyword matching.

def exercise_3():
    """Build a keyword-based tool router."""
    print("\n" + "=" * 60)
    print("Exercise 3: Multi-Tool Router")
    print("=" * 60)

    @dataclass
    class ToolConfig:
        name: str
        keywords: list[str]
        handler: Callable

    class ToolRouter:
        def __init__(self):
            self.tools: list[ToolConfig] = []

        def register(self, name: str, keywords: list[str], handler: Callable):
            self.tools.append(ToolConfig(name, keywords, handler))

        # TODO: Route query to best matching tool(s)
        def route(self, query: str) -> list[tuple[str, float]]:
            """Return list of (tool_name, score) sorted by match score."""
            query_words = set(query.lower().split())
            scores = []
            for tool in self.tools:
                keyword_set = set(k.lower() for k in tool.keywords)
                overlap = len(query_words & keyword_set)
                if overlap > 0:
                    score = overlap / len(keyword_set)
                    scores.append((tool.name, round(score, 3)))
            return sorted(scores, key=lambda x: -x[1])

        # TODO: Execute the top-matching tool
        def execute_best(self, query: str) -> dict:
            matches = self.route(query)
            if not matches:
                return {"error": "No matching tool found"}
            best_name = matches[0][0]
            for tool in self.tools:
                if tool.name == best_name:
                    return {"tool": best_name, "result": tool.handler(query)}
            return {"error": f"Tool {best_name} not found"}

    router = ToolRouter()
    router.register("weather", ["weather", "temperature", "forecast", "rain", "sunny"],
                     lambda q: {"weather": "sunny", "temp": 22})
    router.register("flights", ["flight", "fly", "airline", "travel", "booking"],
                     lambda q: {"flights": [{"airline": "UA", "price": 300}]})
    router.register("hotels", ["hotel", "accommodation", "stay", "booking", "room"],
                     lambda q: {"hotels": [{"name": "Grand Hotel", "price": 150}]})
    router.register("calculator", ["calculate", "math", "sum", "total", "compute"],
                     lambda q: {"result": 42})

    queries = [
        "What is the weather forecast for Tokyo?",
        "Find me a flight to London",
        "I need a hotel room and booking",
        "Calculate the total cost",
        "Tell me about quantum physics",
    ]

    for q in queries:
        matches = router.route(q)
        result = router.execute_best(q)
        print(f"  Query: '{q}'")
        print(f"    Matches: {matches}")
        print(f"    Result: {result}")
        print()


# === Exercise 4: Conversation with Tool Loop ===
# Problem: Implement a complete conversation loop that simulates
# an LLM deciding when to use tools vs responding directly.

def exercise_4():
    """Implement a simulated tool-using conversation loop."""
    print("=" * 60)
    print("Exercise 4: Conversation with Tools")
    print("=" * 60)

    tools = {
        "get_time": lambda: {"time": "2026-03-16 14:30:00"},
        "get_weather": lambda city="NYC": {"city": city, "temp": 20, "condition": "sunny"},
        "calculate": lambda expr="0": {"result": eval(expr, {"__builtins__": {}})},
    }

    TOOL_KEYWORDS = {
        "get_time": ["time", "clock", "date", "now"],
        "get_weather": ["weather", "temperature", "forecast"],
        "calculate": ["calculate", "compute", "math", "sum", "plus", "minus"],
    }

    # TODO: Implement the conversation loop
    def chat(user_message: str) -> str:
        """Process a user message, using tools if needed."""
        msg_lower = user_message.lower()

        # Decide if we need tools
        tool_to_use = None
        best_score = 0
        for tool_name, keywords in TOOL_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in msg_lower)
            if score > best_score:
                best_score = score
                tool_to_use = tool_name

        if tool_to_use and best_score > 0:
            # Extract arguments if possible
            if tool_to_use == "get_weather":
                cities = ["tokyo", "london", "nyc", "seoul", "paris"]
                city = next((c for c in cities if c in msg_lower), "NYC")
                result = tools[tool_to_use](city)
            elif tool_to_use == "calculate":
                # Extract math expression
                import re
                nums = re.findall(r'\d+', user_message)
                if len(nums) >= 2:
                    result = tools[tool_to_use](f"{nums[0]}+{nums[1]}")
                else:
                    result = tools[tool_to_use]()
            else:
                result = tools[tool_to_use]()

            return f"[Used {tool_to_use}] {json.dumps(result)}"
        else:
            return f"[Direct response] I can help with weather, time, and calculations."

    conversations = [
        "What time is it now?",
        "What's the weather in Tokyo?",
        "Calculate 15 plus 27",
        "Tell me about Python programming",
        "What's the temperature forecast?",
    ]

    for msg in conversations:
        response = chat(msg)
        print(f"  User: {msg}")
        print(f"  Bot:  {response}")
        print()


if __name__ == "__main__":
    exercise_1()
    exercise_2()
    exercise_3()
    exercise_4()
