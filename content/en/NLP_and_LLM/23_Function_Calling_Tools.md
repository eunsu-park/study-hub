# 23. Function Calling and Tool Use

## Learning Objectives

- Master OpenAI function calling and Anthropic tool use APIs
- Design effective tool definition schemas with proper descriptions
- Implement parallel tool calling and tool choice strategies
- Build custom tools and integrate with the Model Context Protocol (MCP)
- Handle errors gracefully and orchestrate complex multi-tool workflows

---

## Theory & Principles

Function calling (also called "tool use") is the protocol by which an LLM **decides to invoke external code** instead of just generating text. Without it, an LLM is limited to its training data and cannot interact with the world. With it, the LLM becomes an *agent*: a system that can search the web, query databases, run computations, send emails, control APIs. The protocol itself is simple — JSON-schema-based — but the design space around it (tool description quality, parallel calls, choice strategy, error handling) is large.

This section covers:

- **(A) The protocol** — what messages flow between LLM and host, the loop structure.
- **(B) JSON Schema as the contract** — how a function description becomes a constrained generation grammar.
- **(C) Tool choice strategies** — auto, required, none, named tool; trade-offs.
- **(D) Parallel tool calling** — multiple simultaneous calls, dependency analysis.
- **(E) Provider variations** — OpenAI vs Anthropic vs Gemini, what's portable and what isn't.
- **(F) Tool description quality** — why writing the description well is more important than writing the function code.
- **(G) MCP (Model Context Protocol)** — Anthropic's emerging standard for tool servers.
- **(H) Error handling** — how to communicate tool failures back to the model so it can recover.

### A. The Protocol

The conversation between LLM and host has structured messages:

```
1. User → Host:  "What's the weather in Tokyo?"
2. Host → LLM:   user message + tool definitions
3. LLM → Host:   tool_calls = [{id: "1", name: "get_weather", args: {city: "Tokyo"}}]
4. Host → tool:  execute get_weather("Tokyo")
5. tool → Host:  {"temp": 22, "condition": "cloudy"}
6. Host → LLM:   previous + tool_result message {id: "1", content: "..."}
7. LLM → Host:   "It's 22°C and cloudy in Tokyo."
8. Host → User:  "It's 22°C and cloudy in Tokyo."
```

Steps 3-6 form the **tool loop**. The LLM may emit tool calls multiple times in a single conversation; each tool result feeds back as new context for the next LLM call. Termination: the LLM emits a normal text response with no tool calls.

### B. JSON Schema as the Contract

Each available tool is described with a JSON schema for its parameters:

```json
{
  "name": "get_weather",
  "description": "Get current weather for a city.",
  "parameters": {
    "type": "object",
    "properties": {
      "city": {"type": "string", "description": "City name, e.g., 'Tokyo'"},
      "units": {"type": "string", "enum": ["celsius", "fahrenheit"], "default": "celsius"}
    },
    "required": ["city"]
  }
}
```

This serves two purposes:
- **For the LLM**: tells it what tools exist and what arguments are valid. The model fine-tuning has internalized the JSON-schema convention, so it understands how to interpret the schema.
- **For constrained decoding** (in strict-mode APIs): the schema is compiled into a token mask that prevents the model from emitting invalid tool calls.

The strict-mode guarantee is significant: with strict mode on, the API will *never* return a tool call with the wrong arguments. The model can still pick the wrong tool or hallucinate values, but the structure is enforced.

### C. Tool Choice Strategies

How the LLM decides whether to use a tool:

- **`auto`** (default): LLM decides per-message whether to call a tool or respond directly. Standard for chatbots.
- **`required`** / **`any`**: LLM must call at least one tool. Useful when you know the response requires external data.
- **`none`**: LLM cannot call tools. Useful when you want to force a final text response.
- **`{"name": "tool_name"}`**: LLM must call this specific tool. Useful for forced extraction (using the function-calling API as structured output, lesson 22).

Choice of strategy is per-call, not per-conversation. A typical agent flow uses `auto` initially, then `none` to force a final summary.

### D. Parallel Tool Calling

Modern APIs (GPT-4 Turbo+, Claude 3.5+) can return *multiple* tool calls in a single response:

```json
{
  "tool_calls": [
    {"id": "1", "name": "get_weather", "args": {"city": "Tokyo"}},
    {"id": "2", "name": "get_weather", "args": {"city": "Paris"}}
  ]
}
```

The host executes both in parallel, returns both results in the next user message. Two benefits:
- **Latency**: parallel I/O instead of serial.
- **LLM efficiency**: one round-trip instead of two.

The model is trained to identify when calls are independent (Tokyo and Paris weather can be parallel) vs dependent (search → click first result must be serial). Quality varies — assume the model gets this mostly right but verify on your specific tools.

### E. Provider Variations

Three large providers have implemented function calling slightly differently:

- **OpenAI**: `tools` array of function definitions; `tool_calls` in the response. Strict mode available since 2024.
- **Anthropic**: `tools` array similar to OpenAI; response uses `tool_use` content blocks. Tool result fed back as `tool_result` content blocks.
- **Google Gemini**: similar `tools` definition; uses `function_call` in response. Slightly different message format.

Wrappers like LangChain, instructor, and `litellm` abstract these differences. For native code, pick one provider and stick with their format; cross-provider is doable but adds complexity.

### F. Tool Description Quality

The most important code in your function-calling system is the **string description** of each tool. The LLM sees the descriptions and decides which tool to call based on them. Quality there directly drives quality everywhere.

Best practices:
- **Specify what the tool does, not how it works.** "Get current weather" not "Call the weatherapi.com /v1/current.json endpoint".
- **Specify when to use it.** "Use when the user asks about current weather conditions."
- **Specify when NOT to use it.** "Do not use for weather forecasts; use `get_forecast` instead."
- **Show example inputs/outputs.** A line of example usage in the description significantly improves model selection.
- **Be specific about argument formats.** "City name in English, e.g., 'Tokyo' (not 'JP')."

A tool with a vague description gets called incorrectly; a tool with a precise description gets called when appropriate and skipped otherwise. This is leverage: 30 minutes spent on tool descriptions can outweigh hours of prompt engineering.

### G. MCP (Model Context Protocol)

Anthropic's MCP (2024) is an emerging standard for **tool servers**: external processes that expose tools to LLMs over a standard protocol. Instead of bundling tool implementations into your application, you connect to an MCP server that provides them.

Benefits:
- **Reusability**: write a tool once (e.g., "search GitHub repos"), use it from any MCP-compatible client.
- **Separation of concerns**: tool implementation lives in the server, not in your application code.
- **Security**: the server can enforce its own auth and rate limits; the LLM client doesn't see secrets.

Conceptually similar to LSP (Language Server Protocol) for code editors. Early but gaining traction.

### H. Error Handling

Tool calls fail. The right pattern: **catch errors and feed them back as tool results**, so the LLM can react.

```python
try:
    result = execute_tool(name, args)
except Exception as e:
    result = {"error": str(e)}
# Feed `result` back as the tool_result message
```

The LLM sees the error message and typically either retries with corrected arguments, switches to a different tool, or asks the user for clarification. Throwing the exception up to the user breaks the loop and produces a worse experience.

For unrecoverable errors (auth failures, billing issues), you may want to terminate the loop with a user-facing message. For recoverable errors (timeout, rate limit, malformed args), feeding back to the LLM is the right move.

### From Theory to the Functions Below

- §1 (overview) — frames §A's protocol and §F's tool-description importance.
- §2 (OpenAI) — implements §A-§E with OpenAI's API.
- §3 (Anthropic) — implements the same with Anthropic's tool-use blocks (§E variation).
- §4 (parallel calling) — §D's parallel tool calls with dependency reasoning.
- §5 (custom tools) — §B's JSON-schema authoring + §F's description best practices.
- §6 (MCP) — §G's protocol and example servers.
- §7 (orchestration patterns) — §H error handling, retry, multi-tool workflows that build on the lesson 14 agent loop.

---

## 1. Function Calling Overview

### Why Function Calling?

> **Function Calling / Tool Use**
>
> - **Bridge**: Connects LLM reasoning to real-world actions (APIs, databases, code execution)
> - **Structured Output**: Forces the LLM to produce valid JSON matching a schema
> - **Reliability**: Eliminates regex parsing of free-form text
> - **Composability**: Build complex workflows by chaining tool calls
> - **Safety**: The application controls execution; the LLM only decides what to call

### Provider Comparison

| Feature | OpenAI | Anthropic | Google (Gemini) |
|---------|--------|-----------|-----------------|
| API Name | Function Calling / Tools | Tool Use | Function Calling |
| Max Tools | 128 | 64 | 128 |
| Parallel Calls | Yes | Yes | Yes |
| Streaming | Yes | Yes | Yes |
| Forced Tool Use | `tool_choice: required` | `tool_choice: {"type": "tool", ...}` | `tool_config` |
| Stop Reason | `tool_calls` | `tool_use` | `FUNCTION_CALL` |
| Strict Schema | Yes (structured outputs) | No | No |

---

## 2. OpenAI Function Calling API

### Basic Function Calling

```python
from openai import OpenAI
import json

client = OpenAI()

# Define tools
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city. Use this when the user asks about weather conditions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "City name, e.g., 'San Francisco' or 'Tokyo'",
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "Temperature unit",
                        "default": "celsius",
                    },
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_flights",
            "description": "Search for available flights between two cities on a given date.",
            "parameters": {
                "type": "object",
                "properties": {
                    "origin": {"type": "string", "description": "Departure city or airport code"},
                    "destination": {"type": "string", "description": "Arrival city or airport code"},
                    "date": {"type": "string", "description": "Travel date in YYYY-MM-DD format"},
                    "passengers": {"type": "integer", "minimum": 1, "maximum": 9, "default": 1},
                },
                "required": ["origin", "destination", "date"],
            },
        },
    },
]

# Tool implementations
def get_weather(city: str, unit: str = "celsius") -> dict:
    """Simulated weather API."""
    return {
        "city": city,
        "temperature": 22 if unit == "celsius" else 72,
        "unit": unit,
        "condition": "partly cloudy",
        "humidity": 65,
    }

def search_flights(origin: str, destination: str, date: str,
                   passengers: int = 1) -> dict:
    """Simulated flight search API."""
    return {
        "flights": [
            {"airline": "United", "departure": "08:30", "arrival": "11:45",
             "price": 320 * passengers},
            {"airline": "Delta", "departure": "14:15", "arrival": "17:30",
             "price": 285 * passengers},
        ],
        "origin": origin,
        "destination": destination,
        "date": date,
    }

TOOL_REGISTRY = {
    "get_weather": get_weather,
    "search_flights": search_flights,
}

def chat_with_tools(user_message: str) -> str:
    """Complete function calling loop."""
    messages = [
        {"role": "system", "content": "You are a helpful travel assistant."},
        {"role": "user", "content": user_message},
    ]

    # Step 1: Send message with tools
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        tools=tools,
        tool_choice="auto",
    )
    message = response.choices[0].message

    # Step 2: Check if model wants to call tools
    if message.tool_calls:
        messages.append(message)  # Add assistant message with tool calls

        # Step 3: Execute each tool call
        for tool_call in message.tool_calls:
            func_name = tool_call.function.name
            func_args = json.loads(tool_call.function.arguments)

            print(f"Calling {func_name}({func_args})")

            # Execute the function
            if func_name in TOOL_REGISTRY:
                result = TOOL_REGISTRY[func_name](**func_args)
            else:
                result = {"error": f"Unknown function: {func_name}"}

            # Step 4: Send tool result back
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(result),
            })

        # Step 5: Get final response
        final_response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            tools=tools,
        )
        return final_response.choices[0].message.content

    return message.content

# Usage
print(chat_with_tools("What's the weather in Tokyo and find me flights from NYC to Tokyo on 2026-04-15?"))
```

### Tool Choice Strategies

```python
# AUTO: Model decides whether to use tools (default)
response = client.chat.completions.create(
    model="gpt-4o",
    messages=messages,
    tools=tools,
    tool_choice="auto",
)

# REQUIRED: Model must call at least one tool
response = client.chat.completions.create(
    model="gpt-4o",
    messages=messages,
    tools=tools,
    tool_choice="required",
)

# SPECIFIC: Force a specific tool
response = client.chat.completions.create(
    model="gpt-4o",
    messages=messages,
    tools=tools,
    tool_choice={"type": "function", "function": {"name": "get_weather"}},
)

# NONE: Disable tool use for this request
response = client.chat.completions.create(
    model="gpt-4o",
    messages=messages,
    tools=tools,
    tool_choice="none",
)
```

---

## 3. Anthropic Tool Use

### Claude Tool Use API

```python
from anthropic import Anthropic

anthropic = Anthropic()

# Anthropic tool definitions
tools = [
    {
        "name": "get_stock_price",
        "description": (
            "Get the current stock price for a given ticker symbol. "
            "Returns the latest price, change, and volume."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Stock ticker symbol, e.g., 'AAPL', 'GOOGL'",
                },
                "include_history": {
                    "type": "boolean",
                    "description": "Include 5-day price history",
                    "default": False,
                },
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "calculate_portfolio_value",
        "description": "Calculate the total value of a stock portfolio given holdings.",
        "input_schema": {
            "type": "object",
            "properties": {
                "holdings": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "ticker": {"type": "string"},
                            "shares": {"type": "number"},
                        },
                        "required": ["ticker", "shares"],
                    },
                },
            },
            "required": ["holdings"],
        },
    },
]

def get_stock_price(ticker: str, include_history: bool = False) -> dict:
    """Simulated stock API."""
    prices = {"AAPL": 198.50, "GOOGL": 175.30, "MSFT": 425.80, "NVDA": 890.10}
    price = prices.get(ticker.upper(), 100.00)
    result = {
        "ticker": ticker.upper(),
        "price": price,
        "change": 2.35,
        "change_percent": 1.2,
        "volume": 45_000_000,
    }
    if include_history:
        result["history"] = [price - i * 1.5 for i in range(5, 0, -1)]
    return result

def calculate_portfolio_value(holdings: list[dict]) -> dict:
    """Calculate portfolio value."""
    total = 0
    details = []
    for h in holdings:
        price_data = get_stock_price(h["ticker"])
        value = price_data["price"] * h["shares"]
        total += value
        details.append({
            "ticker": h["ticker"],
            "shares": h["shares"],
            "price": price_data["price"],
            "value": round(value, 2),
        })
    return {"total_value": round(total, 2), "holdings": details}

TOOL_REGISTRY = {
    "get_stock_price": get_stock_price,
    "calculate_portfolio_value": calculate_portfolio_value,
}

def chat_with_claude_tools(user_message: str) -> str:
    """Complete Anthropic tool use loop."""
    messages = [{"role": "user", "content": user_message}]

    while True:
        response = anthropic.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            tools=tools,
            messages=messages,
        )

        # Check if model wants to use tools
        if response.stop_reason == "tool_use":
            # Add assistant response (contains tool_use blocks)
            messages.append({"role": "assistant", "content": response.content})

            # Execute each tool call and collect results
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    func_name = block.name
                    func_args = block.input

                    print(f"Calling {func_name}({json.dumps(func_args)})")

                    try:
                        result = TOOL_REGISTRY[func_name](**func_args)
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": json.dumps(result),
                        })
                    except Exception as e:
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": json.dumps({"error": str(e)}),
                            "is_error": True,
                        })

            messages.append({"role": "user", "content": tool_results})

        elif response.stop_reason == "end_turn":
            # Extract final text response
            for block in response.content:
                if hasattr(block, "text"):
                    return block.text
            return "No text response generated."
        else:
            return f"Unexpected stop reason: {response.stop_reason}"

# Usage
result = chat_with_claude_tools(
    "What's the value of my portfolio? I have 100 shares of AAPL, "
    "50 shares of GOOGL, and 25 shares of NVDA."
)
print(result)
```

---

## 4. Parallel Tool Calling

### OpenAI Parallel Calls

```python
def handle_parallel_tool_calls(user_message: str) -> str:
    """Handle multiple tool calls in a single turn."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Use tools when needed."},
        {"role": "user", "content": user_message},
    ]

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        tools=tools,
        parallel_tool_calls=True,  # Enable parallel (default in most models)
    )

    message = response.choices[0].message

    if message.tool_calls:
        messages.append(message)

        # The model may issue multiple tool calls in one turn
        print(f"Model requested {len(message.tool_calls)} parallel tool calls")

        # Execute all tool calls (can be parallelized in production)
        for tool_call in message.tool_calls:
            func_name = tool_call.function.name
            func_args = json.loads(tool_call.function.arguments)

            result = TOOL_REGISTRY.get(func_name, lambda **k: {"error": "unknown"})(**func_args)

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(result),
            })

        # Get final response after all tools complete
        final = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            tools=tools,
        )
        return final.choices[0].message.content

    return message.content

# The model will call get_weather twice in parallel
result = handle_parallel_tool_calls(
    "Compare the weather in Tokyo and New York right now"
)
print(result)
```

### Async Parallel Execution

```python
import asyncio
from openai import AsyncOpenAI

async_client = AsyncOpenAI()

async def execute_tools_parallel(tool_calls: list) -> list[dict]:
    """Execute multiple tool calls concurrently."""
    async def run_tool(tool_call) -> dict:
        func_name = tool_call.function.name
        func_args = json.loads(tool_call.function.arguments)

        # In production: async API calls, async DB queries, etc.
        result = TOOL_REGISTRY.get(
            func_name, lambda **k: {"error": "unknown"}
        )(**func_args)

        return {
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": json.dumps(result),
        }

    # Run all tool calls concurrently
    results = await asyncio.gather(*[run_tool(tc) for tc in tool_calls])
    return list(results)

async def async_chat_with_tools(user_message: str) -> str:
    """Async function calling with parallel tool execution."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": user_message},
    ]

    response = await async_client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        tools=tools,
    )
    message = response.choices[0].message

    if message.tool_calls:
        messages.append(message)

        # Execute all tools in parallel
        tool_results = await execute_tools_parallel(message.tool_calls)
        messages.extend(tool_results)

        final = await async_client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            tools=tools,
        )
        return final.choices[0].message.content

    return message.content
```

---

## 5. Building Custom Tools

### Tool Design Patterns

```python
from typing import Any, Callable
from dataclasses import dataclass
from functools import wraps
import inspect

@dataclass
class ToolDefinition:
    name: str
    description: str
    parameters: dict
    handler: Callable

class ToolRegistry:
    """Registry for managing and auto-generating tool definitions."""

    def __init__(self):
        self._tools: dict[str, ToolDefinition] = {}

    def tool(self, description: str):
        """Decorator to register a function as a tool."""
        def decorator(func: Callable) -> Callable:
            # Auto-generate JSON schema from type hints
            hints = func.__annotations__
            sig = inspect.signature(func)

            properties = {}
            required = []

            for param_name, param in sig.parameters.items():
                if param_name == "return":
                    continue

                hint = hints.get(param_name, str)
                prop = self._type_to_schema(hint)

                # Use docstring parsing or parameter description
                prop["description"] = f"Parameter: {param_name}"
                properties[param_name] = prop

                if param.default is inspect.Parameter.empty:
                    required.append(param_name)

            tool_def = ToolDefinition(
                name=func.__name__,
                description=description,
                parameters={
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
                handler=func,
            )
            self._tools[func.__name__] = tool_def

            @wraps(func)
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)
            return wrapper

        return decorator

    def _type_to_schema(self, hint) -> dict:
        """Convert Python type hint to JSON Schema type."""
        type_map = {
            str: {"type": "string"},
            int: {"type": "integer"},
            float: {"type": "number"},
            bool: {"type": "boolean"},
            list: {"type": "array", "items": {"type": "string"}},
        }
        return type_map.get(hint, {"type": "string"})

    def get_openai_tools(self) -> list[dict]:
        """Export tools in OpenAI format."""
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                },
            }
            for t in self._tools.values()
        ]

    def get_anthropic_tools(self) -> list[dict]:
        """Export tools in Anthropic format."""
        return [
            {
                "name": t.name,
                "description": t.description,
                "input_schema": t.parameters,
            }
            for t in self._tools.values()
        ]

    def execute(self, name: str, arguments: dict) -> Any:
        """Execute a tool by name."""
        if name not in self._tools:
            raise ValueError(f"Unknown tool: {name}")
        return self._tools[name].handler(**arguments)

# Usage
registry = ToolRegistry()

@registry.tool("Search a database of products by name or category")
def search_products(query: str, category: str, max_results: int) -> dict:
    return {"results": [{"name": f"Product matching '{query}'", "category": category}]}

@registry.tool("Create a new support ticket in the help desk system")
def create_ticket(title: str, description: str, priority: str) -> dict:
    return {"ticket_id": "TKT-12345", "status": "created"}

# Auto-generated tool definitions
print(json.dumps(registry.get_openai_tools(), indent=2))
```

### Validated Tool with Pydantic

```python
from pydantic import BaseModel, Field
from typing import Literal

class SearchParams(BaseModel):
    query: str = Field(min_length=1, max_length=500, description="Search query text")
    filters: dict[str, str] = Field(
        default_factory=dict,
        description="Key-value filters to narrow results",
    )
    sort_by: Literal["relevance", "date", "price"] = Field(
        default="relevance", description="Sort order for results"
    )
    limit: int = Field(default=10, ge=1, le=100, description="Max results to return")

def validated_tool(params_model: type[BaseModel]):
    """Decorator that validates tool input with a Pydantic model."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(**kwargs):
            # Validate and coerce inputs
            try:
                validated = params_model(**kwargs)
                return func(**validated.model_dump())
            except Exception as e:
                return {"error": f"Validation failed: {str(e)}"}
        wrapper._params_model = params_model
        return wrapper
    return decorator

@validated_tool(SearchParams)
def search_knowledge_base(query: str, filters: dict, sort_by: str, limit: int) -> dict:
    """Search the internal knowledge base."""
    return {
        "results": [{"title": f"Result for: {query}", "score": 0.95}],
        "total": 1,
        "query": query,
        "sort_by": sort_by,
    }

# Validation catches bad inputs before hitting the tool
result = search_knowledge_base(query="", limit=200)
print(result)  # {"error": "Validation failed: ..."}

result = search_knowledge_base(query="LLM patterns", limit=5)
print(result)  # {"results": [...], "total": 1, ...}
```

---

## 6. Model Context Protocol (MCP)

### MCP Overview

> **Model Context Protocol (MCP)**
>
> - Open standard by Anthropic for connecting LLMs to external data and tools
> - Client-server architecture: LLM applications (clients) connect to MCP servers
> - Servers expose **tools**, **resources**, and **prompts** over a standardized protocol
> - Transport: stdio (local) or HTTP+SSE (remote)
> - Enables tool reuse across different LLM applications

### MCP Server Implementation

```python
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent
import json

# Create an MCP server
server = Server("weather-service")

@server.list_tools()
async def list_tools() -> list[Tool]:
    """Expose available tools."""
    return [
        Tool(
            name="get_weather",
            description="Get current weather for a location",
            inputSchema={
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "City name",
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "default": "celsius",
                    },
                },
                "required": ["city"],
            },
        ),
        Tool(
            name="get_forecast",
            description="Get 5-day weather forecast for a location",
            inputSchema={
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "days": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 7,
                        "default": 5,
                    },
                },
                "required": ["city"],
            },
        ),
    ]

@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Handle tool calls."""
    if name == "get_weather":
        city = arguments["city"]
        unit = arguments.get("unit", "celsius")
        # In production: call real weather API
        result = {
            "city": city,
            "temperature": 22 if unit == "celsius" else 72,
            "condition": "sunny",
            "humidity": 55,
        }
        return [TextContent(type="text", text=json.dumps(result))]

    elif name == "get_forecast":
        city = arguments["city"]
        days = arguments.get("days", 5)
        result = {
            "city": city,
            "forecast": [
                {"day": i + 1, "high": 22 + i, "low": 15 + i, "condition": "clear"}
                for i in range(days)
            ],
        }
        return [TextContent(type="text", text=json.dumps(result))]

    else:
        return [TextContent(type="text", text=f"Unknown tool: {name}")]

# Run the server
async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
```

### MCP Client Integration

```python
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def use_mcp_tools():
    """Connect to an MCP server and use its tools."""
    # Connect to a local MCP server via stdio
    server_params = StdioServerParameters(
        command="python",
        args=["weather_mcp_server.py"],
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # List available tools
            tools = await session.list_tools()
            print("Available tools:")
            for tool in tools.tools:
                print(f"  - {tool.name}: {tool.description}")

            # Call a tool
            result = await session.call_tool(
                "get_weather",
                arguments={"city": "Seoul", "unit": "celsius"},
            )
            print(f"Weather result: {result.content[0].text}")

            # Use tools with Claude
            anthropic = Anthropic()

            # Convert MCP tools to Anthropic format
            anthropic_tools = [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.inputSchema,
                }
                for tool in tools.tools
            ]

            response = anthropic.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1024,
                tools=anthropic_tools,
                messages=[
                    {"role": "user", "content": "What's the weather and 3-day forecast for Tokyo?"}
                ],
            )

            # Execute tool calls via MCP
            if response.stop_reason == "tool_use":
                for block in response.content:
                    if block.type == "tool_use":
                        mcp_result = await session.call_tool(
                            block.name, arguments=block.input
                        )
                        print(f"Tool {block.name}: {mcp_result.content[0].text}")
```

### MCP Architecture

| Component | Role | Examples |
|-----------|------|---------|
| **Client** | LLM application that consumes tools | Claude Desktop, custom apps |
| **Server** | Provides tools, resources, prompts | Weather service, DB connector |
| **Transport** | Communication layer | stdio, HTTP+SSE |
| **Resources** | Read-only data exposed by server | Files, DB records, API data |
| **Tools** | Executable actions | search, create, update, delete |
| **Prompts** | Reusable prompt templates | Analysis templates, extraction prompts |

---

## 7. Tool Orchestration Patterns

### Sequential Tool Chain

```python
class ToolChain:
    """Execute tools in sequence, passing results between steps."""

    def __init__(self):
        self.client = OpenAI()
        self.steps: list[dict] = []

    def add_step(self, tool_name: str, description: str):
        self.steps.append({"tool": tool_name, "description": description})
        return self

    def execute(self, initial_input: str, tools: list[dict],
                tool_registry: dict) -> list[dict]:
        """Execute the chain step by step."""
        results = []
        context = initial_input

        for i, step in enumerate(self.steps):
            print(f"Step {i+1}/{len(self.steps)}: {step['description']}")

            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": (
                        f"You are executing step {i+1}: {step['description']}\n"
                        f"Previous context:\n{context}\n\n"
                        f"Use the '{step['tool']}' tool to complete this step."
                    )},
                    {"role": "user", "content": initial_input},
                ],
                tools=tools,
                tool_choice={"type": "function", "function": {"name": step["tool"]}},
            )

            msg = response.choices[0].message
            if msg.tool_calls:
                tc = msg.tool_calls[0]
                args = json.loads(tc.function.arguments)
                result = tool_registry[tc.function.name](**args)
                results.append({"step": step["description"], "result": result})
                context = f"{context}\n\nStep {i+1} result: {json.dumps(result)}"

        return results
```

### Error Handling in Tool Calls

```python
from typing import NamedTuple

class ToolResult(NamedTuple):
    success: bool
    data: Any
    error: str | None

class RobustToolExecutor:
    """Execute tools with comprehensive error handling."""

    def __init__(self, timeout: float = 30.0, max_retries: int = 2):
        self.timeout = timeout
        self.max_retries = max_retries

    def execute(self, name: str, arguments: dict,
                registry: dict[str, Callable]) -> ToolResult:
        """Execute a tool with timeout, retry, and error handling."""
        import signal

        if name not in registry:
            return ToolResult(
                success=False,
                data=None,
                error=f"Unknown tool: {name}. Available: {list(registry.keys())}",
            )

        func = registry[name]

        for attempt in range(self.max_retries + 1):
            try:
                result = func(**arguments)
                return ToolResult(success=True, data=result, error=None)

            except TypeError as e:
                # Wrong arguments — don't retry, return error for LLM to fix
                return ToolResult(
                    success=False,
                    data=None,
                    error=f"Invalid arguments for {name}: {e}",
                )

            except ConnectionError as e:
                if attempt < self.max_retries:
                    import time
                    time.sleep(2 ** attempt)  # Exponential backoff
                    continue
                return ToolResult(
                    success=False,
                    data=None,
                    error=f"Connection failed after {self.max_retries + 1} attempts: {e}",
                )

            except Exception as e:
                return ToolResult(
                    success=False,
                    data=None,
                    error=f"Tool execution error: {type(e).__name__}: {e}",
                )

    def format_for_llm(self, result: ToolResult) -> str:
        """Format tool result as a string message for the LLM."""
        if result.success:
            return json.dumps(result.data) if not isinstance(result.data, str) else result.data
        else:
            return json.dumps({
                "error": result.error,
                "suggestion": "Please try a different approach or different parameters.",
            })

# Integration with the agentic loop
executor = RobustToolExecutor(timeout=30.0, max_retries=2)

def agentic_loop(user_message: str, tools: list[dict],
                 registry: dict[str, Callable], max_turns: int = 10) -> str:
    """Robust agentic loop with error handling."""
    messages = [
        {"role": "system", "content": (
            "You are a helpful assistant with tools. "
            "If a tool call fails, try a different approach. "
            "Don't repeat the same failed call."
        )},
        {"role": "user", "content": user_message},
    ]

    for turn in range(max_turns):
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            tools=tools,
        )
        msg = response.choices[0].message

        if not msg.tool_calls:
            return msg.content

        messages.append(msg)

        for tc in msg.tool_calls:
            args = json.loads(tc.function.arguments)
            result = executor.execute(tc.function.name, args, registry)

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": executor.format_for_llm(result),
            })

    return "Reached maximum turns without a final answer."
```

### Real-World Integration Examples

```python
import httpx

# Example: Database tool
def query_database(sql: str) -> dict:
    """Execute a read-only SQL query."""
    # Safety: only allow SELECT statements
    normalized = sql.strip().upper()
    if not normalized.startswith("SELECT"):
        return {"error": "Only SELECT queries are allowed"}

    # In production: connect to actual database
    return {"columns": ["id", "name"], "rows": [[1, "Alice"], [2, "Bob"]], "row_count": 2}

# Example: HTTP API tool
def call_api(url: str, method: str = "GET", body: dict | None = None) -> dict:
    """Make an HTTP API call."""
    # Safety: allowlist of permitted domains
    ALLOWED_DOMAINS = ["api.github.com", "api.openweathermap.org"]
    from urllib.parse import urlparse
    domain = urlparse(url).netloc
    if domain not in ALLOWED_DOMAINS:
        return {"error": f"Domain not allowed: {domain}"}

    try:
        with httpx.Client(timeout=10.0) as http_client:
            if method.upper() == "GET":
                resp = http_client.get(url)
            elif method.upper() == "POST":
                resp = http_client.post(url, json=body)
            else:
                return {"error": f"Unsupported method: {method}"}
            return {"status": resp.status_code, "body": resp.json()}
    except Exception as e:
        return {"error": str(e)}

# Example: File system tool (sandboxed)
def read_file(path: str) -> dict:
    """Read a file from the allowed directory."""
    from pathlib import Path
    ALLOWED_DIR = Path("/data/shared")
    target = (ALLOWED_DIR / path).resolve()

    # Security: prevent path traversal
    if not str(target).startswith(str(ALLOWED_DIR)):
        return {"error": "Access denied: path traversal detected"}

    try:
        content = target.read_text()
        return {"path": str(target), "content": content[:5000], "size": len(content)}
    except FileNotFoundError:
        return {"error": f"File not found: {path}"}
```

### Tool Design Best Practices

| Practice | Description |
|----------|-------------|
| Clear Descriptions | Tool descriptions are part of the prompt; make them specific |
| Constrained Inputs | Use enums, min/max, and required fields to guide the model |
| Idempotent Reads | GET/read operations should be safe to retry |
| Error Messages for LLMs | Return errors as strings the model can reason about |
| Domain Allow-listing | Restrict API calls to approved endpoints |
| Input Validation | Validate all arguments before execution |
| Timeout Boundaries | Set reasonable timeouts on all external calls |
| Audit Logging | Log every tool call with arguments and results |

---

## Next Steps

In [24_Production_LLM_Patterns.md](./24_Production_LLM_Patterns.md), we cover production deployment patterns including caching, cost optimization, observability, and multi-model routing.
