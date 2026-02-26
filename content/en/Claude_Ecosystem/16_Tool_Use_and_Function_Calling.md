# Tool Use and Function Calling

**Previous**: [15. Claude API Fundamentals](./15_Claude_API_Fundamentals.md) | **Next**: [17. Claude Agent SDK](./17_Claude_Agent_SDK.md)

---

Tool use (also called function calling) is the mechanism that lets Claude call functions you define. Instead of generating text about what it *would* do, Claude can actually *do* things -- look up data, query databases, call APIs, perform calculations, and interact with external systems. This lesson covers the complete tool use lifecycle: defining tools, handling the multi-step conversation flow, managing parallel tool calls, and applying best practices for production systems.

**Difficulty**: ⭐⭐

**Prerequisites**:
- Completion of Lesson 15 (Claude API Fundamentals)
- Understanding of JSON Schema basics
- Python 3.9+ or Node.js 18+

## Learning Objectives

After completing this lesson, you will be able to:

1. Define tools with name, description, and JSON Schema input specifications
2. Implement the full tool use conversation flow (request, tool_use, tool_result, response)
3. Handle parallel tool execution when Claude requests multiple tools simultaneously
4. Use the `tool_choice` parameter to control when and how Claude uses tools
5. Return errors from tools and handle edge cases
6. Apply advanced patterns: chaining, structured output, image returns
7. Design effective tools following best practices

---

## Table of Contents

1. [What Is Tool Use?](#1-what-is-tool-use)
2. [Tool Definition Format](#2-tool-definition-format)
3. [The Tool Use Conversation Flow](#3-the-tool-use-conversation-flow)
4. [Python Example: Weather Lookup](#4-python-example-weather-lookup)
5. [TypeScript Example: Database Query](#5-typescript-example-database-query)
6. [Multiple Tools](#6-multiple-tools)
7. [Parallel Tool Execution](#7-parallel-tool-execution)
8. [Controlling Tool Use: tool_choice](#8-controlling-tool-use-tool_choice)
9. [Error Handling in Tools](#9-error-handling-in-tools)
10. [Advanced Patterns](#10-advanced-patterns)
11. [Best Practices](#11-best-practices)
12. [Exercises](#12-exercises)
13. [References](#13-references)

---

## 1. What Is Tool Use?

Without tool use, Claude can only generate text. It can describe how to check the weather, but it cannot actually check it. Tool use bridges this gap:

```
┌────────────────────────────────────────────────────────────────────┐
│                Without Tool Use                                     │
│                                                                     │
│  User: "What is the weather in Tokyo?"                             │
│  Claude: "I don't have access to real-time data. You can check     │
│           weather.com or use a weather app."                        │
│                                                                     │
├────────────────────────────────────────────────────────────────────┤
│                With Tool Use                                        │
│                                                                     │
│  User: "What is the weather in Tokyo?"                             │
│  Claude: [calls get_weather(city="Tokyo")]                         │
│  System: [returns {"temp": 22, "condition": "sunny"}]              │
│  Claude: "It's currently 22°C and sunny in Tokyo."                 │
│                                                                     │
└────────────────────────────────────────────────────────────────────┘
```

**Why tool use matters:**
- **Real-time data**: Look up current information (weather, stock prices, database records)
- **Side effects**: Create files, send emails, update databases, deploy code
- **Computation**: Perform precise calculations, run code, process data
- **Integration**: Connect Claude to any API, service, or system
- **Structured output**: Force Claude to return data in a specific format (via tool schemas)

---

## 2. Tool Definition Format

Each tool is defined with three components:

```json
{
  "name": "get_weather",
  "description": "Get the current weather for a given city. Returns temperature, conditions, and humidity.",
  "input_schema": {
    "type": "object",
    "properties": {
      "city": {
        "type": "string",
        "description": "The city name, e.g., 'Tokyo' or 'New York'"
      },
      "units": {
        "type": "string",
        "enum": ["celsius", "fahrenheit"],
        "description": "Temperature unit. Defaults to celsius."
      }
    },
    "required": ["city"]
  }
}
```

### 2.1 Name

- Must be unique within a request.
- Use `snake_case` (convention).
- Be descriptive: `get_stock_price` not `gsp`.

### 2.2 Description

The description is **critical** -- Claude uses it to decide when to call the tool. Write it as if explaining the tool to a colleague:

```
Good:  "Search the company knowledge base for articles matching a query.
        Returns up to 10 results with title, snippet, and URL.
        Use this when the user asks about company policies, procedures,
        or internal documentation."

Bad:   "Search KB"
```

### 2.3 Input Schema (JSON Schema)

The `input_schema` follows the [JSON Schema](https://json-schema.org/) specification:

```json
{
  "type": "object",
  "properties": {
    "query": {
      "type": "string",
      "description": "Search query string",
      "minLength": 1,
      "maxLength": 500
    },
    "limit": {
      "type": "integer",
      "description": "Maximum number of results to return",
      "minimum": 1,
      "maximum": 50,
      "default": 10
    },
    "filters": {
      "type": "object",
      "description": "Optional filters to narrow results",
      "properties": {
        "category": {
          "type": "string",
          "enum": ["engineering", "hr", "finance", "legal"]
        },
        "date_after": {
          "type": "string",
          "format": "date",
          "description": "Only return articles published after this date (YYYY-MM-DD)"
        }
      }
    }
  },
  "required": ["query"]
}
```

**Key JSON Schema features for tool definitions:**
- `type`: string, integer, number, boolean, array, object
- `enum`: Restrict to specific values
- `description`: Explain each field (Claude reads these)
- `required`: Mark which fields are mandatory
- `default`: Specify default values
- `minimum` / `maximum`: Numeric constraints
- `minLength` / `maxLength`: String length constraints
- `items`: Schema for array elements

---

## 3. The Tool Use Conversation Flow

Tool use involves a multi-step exchange between your application and Claude:

```
┌──────────────────────────────────────────────────────────────────┐
│              Tool Use Conversation Flow                           │
│                                                                   │
│  Step 1: Send user message + tools definition                     │
│  ┌──────────┐         ┌──────────────┐                           │
│  │   Your   │ ──────▶ │    Claude    │                           │
│  │   App    │  tools   │    API      │                           │
│  └──────────┘ + msg    └──────┬───────┘                          │
│                               │                                   │
│  Step 2: Claude responds with tool_use block                      │
│                               │                                   │
│  ┌──────────┐         ┌──────▼───────┐                           │
│  │   Your   │ ◀────── │   Response:  │                           │
│  │   App    │  tool_   │   tool_use   │                           │
│  └────┬─────┘  use     └──────────────┘                          │
│       │                                                           │
│  Step 3: Execute the tool locally                                 │
│       │  Your app calls the actual function                       │
│       ▼                                                           │
│  ┌──────────┐                                                     │
│  │  Tool    │  get_weather("Tokyo")                               │
│  │ Function │  → {"temp": 22, "condition": "sunny"}              │
│  └────┬─────┘                                                     │
│       │                                                           │
│  Step 4: Send tool_result back to Claude                          │
│       │                                                           │
│  ┌────▼─────┐         ┌──────────────┐                           │
│  │   Your   │ ──────▶ │    Claude    │                           │
│  │   App    │  tool_   │    API      │                           │
│  └──────────┘  result  └──────┬───────┘                          │
│                               │                                   │
│  Step 5: Claude generates final response                          │
│                               │                                   │
│  ┌──────────┐         ┌──────▼───────┐                           │
│  │   Your   │ ◀────── │  "It's 22°C  │                           │
│  │   App    │  text    │  and sunny   │                           │
│  └──────────┘          │  in Tokyo."  │                           │
│                        └──────────────┘                           │
└──────────────────────────────────────────────────────────────────┘
```

The key insight: **Claude does not execute the tool itself**. It tells you *which* tool to call and *what arguments* to pass. Your application executes the tool and sends the result back.

---

## 4. Python Example: Weather Lookup

Here is a complete, working example of tool use in Python:

```python
import anthropic
import json

client = anthropic.Anthropic()

# Step 1: Define tools
tools = [
    {
        "name": "get_weather",
        "description": (
            "Get the current weather for a city. "
            "Returns temperature (Celsius), condition, and humidity."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "City name, e.g., 'London' or 'San Francisco'",
                },
            },
            "required": ["city"],
        },
    }
]

# Step 2: Define tool implementation
def get_weather(city: str) -> dict:
    """Simulate a weather API call."""
    # In production, call a real API like OpenWeatherMap
    weather_data = {
        "tokyo": {"temp": 22, "condition": "Sunny", "humidity": 55},
        "london": {"temp": 12, "condition": "Cloudy", "humidity": 78},
        "new york": {"temp": 18, "condition": "Partly Cloudy", "humidity": 62},
    }
    data = weather_data.get(city.lower())
    if data is None:
        return {"error": f"No weather data available for '{city}'"}
    return {"city": city, **data}

# Step 3: Tool use loop
def chat_with_tools(user_message: str) -> str:
    """Send a message and handle any tool use requests."""
    messages = [{"role": "user", "content": user_message}]

    # Initial API call with tools
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        tools=tools,
        messages=messages,
    )

    # Loop until Claude gives a final text response
    while response.stop_reason == "tool_use":
        # Extract tool use blocks from the response
        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                print(f"  [Tool call: {block.name}({json.dumps(block.input)})]")

                # Execute the tool
                if block.name == "get_weather":
                    result = get_weather(**block.input)
                else:
                    result = {"error": f"Unknown tool: {block.name}"}

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": json.dumps(result),
                })

        # Add Claude's response and tool results to the conversation
        messages.append({"role": "assistant", "content": response.content})
        messages.append({"role": "user", "content": tool_results})

        # Call the API again with the tool results
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            tools=tools,
            messages=messages,
        )

    # Extract the final text response
    return "".join(
        block.text for block in response.content if block.type == "text"
    )

# Usage
answer = chat_with_tools("What is the weather like in Tokyo right now?")
print(f"\nClaude: {answer}")
```

Output:
```
  [Tool call: get_weather({"city": "Tokyo"})]

Claude: It's currently 22°C and sunny in Tokyo with 55% humidity.
```

---

## 5. TypeScript Example: Database Query

```typescript
// database-tool.ts
import Anthropic from "@anthropic-ai/sdk";

const client = new Anthropic();

// Define tools
const tools: Anthropic.Tool[] = [
  {
    name: "query_database",
    description:
      "Execute a read-only SQL query against the application database. " +
      "Only SELECT queries are allowed. Returns results as JSON rows.",
    input_schema: {
      type: "object" as const,
      properties: {
        query: {
          type: "string",
          description: "SQL SELECT query to execute",
        },
        database: {
          type: "string",
          enum: ["users", "orders", "products"],
          description: "Which database to query",
        },
      },
      required: ["query", "database"],
    },
  },
  {
    name: "get_table_schema",
    description: "Get the column names and types for a database table.",
    input_schema: {
      type: "object" as const,
      properties: {
        table_name: {
          type: "string",
          description: "Name of the table",
        },
      },
      required: ["table_name"],
    },
  },
];

// Mock tool implementations
function queryDatabase(query: string, database: string): object {
  // Simulate database results
  if (query.toLowerCase().includes("count")) {
    return { rows: [{ count: 1247 }], rowCount: 1 };
  }
  return {
    rows: [
      { id: 1, name: "Alice", email: "alice@example.com" },
      { id: 2, name: "Bob", email: "bob@example.com" },
    ],
    rowCount: 2,
  };
}

function getTableSchema(tableName: string): object {
  const schemas: Record<string, object> = {
    users: {
      columns: [
        { name: "id", type: "INTEGER", nullable: false },
        { name: "name", type: "TEXT", nullable: false },
        { name: "email", type: "TEXT", nullable: false },
        { name: "created_at", type: "TIMESTAMP", nullable: false },
      ],
    },
  };
  return schemas[tableName] || { error: `Table '${tableName}' not found` };
}

// Execute a tool call
function executeTool(name: string, input: Record<string, string>): string {
  switch (name) {
    case "query_database":
      return JSON.stringify(queryDatabase(input.query, input.database));
    case "get_table_schema":
      return JSON.stringify(getTableSchema(input.table_name));
    default:
      return JSON.stringify({ error: `Unknown tool: ${name}` });
  }
}

// Main tool use loop
async function chatWithTools(userMessage: string): Promise<string> {
  const messages: Anthropic.MessageParam[] = [
    { role: "user", content: userMessage },
  ];

  let response = await client.messages.create({
    model: "claude-sonnet-4-20250514",
    max_tokens: 2048,
    tools,
    messages,
  });

  // Handle tool use loop
  while (response.stop_reason === "tool_use") {
    const toolResults: Anthropic.ToolResultBlockParam[] = [];

    for (const block of response.content) {
      if (block.type === "tool_use") {
        console.log(`  [Tool: ${block.name}(${JSON.stringify(block.input)})]`);
        const result = executeTool(
          block.name,
          block.input as Record<string, string>
        );
        toolResults.push({
          type: "tool_result",
          tool_use_id: block.id,
          content: result,
        });
      }
    }

    messages.push({ role: "assistant", content: response.content });
    messages.push({ role: "user", content: toolResults });

    response = await client.messages.create({
      model: "claude-sonnet-4-20250514",
      max_tokens: 2048,
      tools,
      messages,
    });
  }

  // Extract final text
  return response.content
    .filter((b): b is Anthropic.TextBlock => b.type === "text")
    .map((b) => b.text)
    .join("");
}

// Usage
async function main() {
  const answer = await chatWithTools(
    "How many users do we have? Also, what columns does the users table have?"
  );
  console.log(`\nClaude: ${answer}`);
}

main();
```

---

## 6. Multiple Tools

You can provide multiple tools in a single request. Claude will choose which ones to call based on the user's message:

```python
tools = [
    {
        "name": "get_weather",
        "description": "Get current weather for a city.",
        "input_schema": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"}
            },
            "required": ["city"],
        },
    },
    {
        "name": "get_stock_price",
        "description": "Get the current stock price for a ticker symbol.",
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Stock ticker symbol, e.g., 'AAPL'"
                }
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "calculate",
        "description": "Evaluate a mathematical expression. Supports +, -, *, /, **, sqrt, sin, cos, etc.",
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Math expression to evaluate, e.g., '(42 * 3.14) / 2'"
                }
            },
            "required": ["expression"],
        },
    },
]
```

Claude will intelligently select the appropriate tool:
- "What's the weather in Paris?" -> calls `get_weather`
- "What's Apple's stock price?" -> calls `get_stock_price`
- "What's 15% of 250?" -> calls `calculate`
- "Compare the weather in Tokyo and London" -> calls `get_weather` twice (parallel)

---

## 7. Parallel Tool Execution

Claude can request multiple tool calls in a single response. When it does, all `tool_use` blocks appear in the same `content` array:

```python
# Claude's response might contain multiple tool_use blocks:
# response.content = [
#   TextBlock(type="text", text="Let me check both cities..."),
#   ToolUseBlock(type="tool_use", id="toolu_01A", name="get_weather", input={"city": "Tokyo"}),
#   ToolUseBlock(type="tool_use", id="toolu_01B", name="get_weather", input={"city": "London"}),
# ]

def handle_parallel_tools(response, execute_fn):
    """Handle a response that may contain multiple tool_use blocks."""
    tool_results = []

    for block in response.content:
        if block.type == "tool_use":
            # Execute each tool
            result = execute_fn(block.name, block.input)
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,       # Must match the tool_use id
                "content": json.dumps(result),
            })

    return tool_results
```

**Important:** When returning results for parallel tool calls, you must include a `tool_result` for every `tool_use` block, matched by `tool_use_id`. If you skip one, the API will return an error.

For truly parallel execution (e.g., calling multiple APIs simultaneously):

```python
import asyncio
import anthropic

async def execute_tools_parallel(tool_calls: list) -> list:
    """Execute multiple tool calls concurrently."""
    async def execute_one(call):
        # Run the tool (simulate async I/O)
        if call.name == "get_weather":
            result = await async_get_weather(**call.input)
        elif call.name == "get_stock_price":
            result = await async_get_stock(**call.input)
        else:
            result = {"error": f"Unknown tool: {call.name}"}

        return {
            "type": "tool_result",
            "tool_use_id": call.id,
            "content": json.dumps(result),
        }

    # Execute all tool calls concurrently
    results = await asyncio.gather(
        *[execute_one(call) for call in tool_calls]
    )
    return list(results)
```

---

## 8. Controlling Tool Use: tool_choice

The `tool_choice` parameter controls how Claude uses tools:

```python
# auto (default): Claude decides whether to use tools
response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    tools=tools,
    tool_choice={"type": "auto"},     # Claude decides
    messages=[{"role": "user", "content": "Hello!"}],
)

# any: Force Claude to use at least one tool
response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    tools=tools,
    tool_choice={"type": "any"},      # Must use a tool
    messages=[{"role": "user", "content": "What's 2+2?"}],
)

# specific tool: Force Claude to use a particular tool
response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    tools=tools,
    tool_choice={"type": "tool", "name": "calculate"},  # Must use 'calculate'
    messages=[{"role": "user", "content": "What's 2+2?"}],
)
```

**When to use each mode:**

| Mode | Use Case |
|------|----------|
| `auto` | General conversation where tools are optional |
| `any` | When you always want structured output or action |
| `tool` (specific) | When you need a specific tool called (e.g., structured extraction) |

### Using tool_choice for Structured Output

A powerful pattern: define a tool whose schema matches your desired output format, then force Claude to call it:

```python
# Force structured output using tool_choice
tools = [
    {
        "name": "extract_contact_info",
        "description": "Extract structured contact information from text.",
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "email": {"type": "string", "format": "email"},
                "phone": {"type": "string"},
                "company": {"type": "string"},
                "role": {"type": "string"},
            },
            "required": ["name"],
        },
    }
]

response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    tools=tools,
    tool_choice={"type": "tool", "name": "extract_contact_info"},
    messages=[
        {
            "role": "user",
            "content": "Extract contact info from: Hi, I'm Jane Smith, "
                       "CTO at TechCorp. Reach me at jane@techcorp.io "
                       "or 555-0123.",
        }
    ],
)

# Claude is forced to return structured data via the tool
for block in response.content:
    if block.type == "tool_use":
        print(json.dumps(block.input, indent=2))
        # {
        #   "name": "Jane Smith",
        #   "email": "jane@techcorp.io",
        #   "phone": "555-0123",
        #   "company": "TechCorp",
        #   "role": "CTO"
        # }
```

---

## 9. Error Handling in Tools

When a tool execution fails, return an error in the `tool_result` rather than crashing:

```python
def execute_tool(name: str, input_data: dict) -> dict:
    """Execute a tool and handle errors gracefully."""
    try:
        if name == "get_weather":
            return get_weather(**input_data)
        elif name == "query_database":
            return query_database(**input_data)
        else:
            return {"error": f"Unknown tool: {name}"}

    except ValueError as e:
        return {"error": f"Invalid input: {str(e)}"}
    except TimeoutError:
        return {"error": "Operation timed out. Please try again."}
    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"}

# When building tool_results, set is_error for failures
def build_tool_result(tool_use_id: str, result: dict) -> dict:
    """Build a tool_result message, flagging errors."""
    content = json.dumps(result)
    is_error = "error" in result

    return {
        "type": "tool_result",
        "tool_use_id": tool_use_id,
        "content": content,
        "is_error": is_error,   # Tells Claude the tool failed
    }
```

The `is_error` flag is important: it tells Claude that the tool call failed, so Claude can inform the user or try a different approach rather than treating error text as valid data.

---

## 10. Advanced Patterns

### 10.1 Tool Call Chaining

Sometimes Claude needs to call multiple tools in sequence, using the output of one as input to the next:

```
User: "What's the weather in the city with the most users?"

Claude → call get_top_city()
     ← result: {"city": "Tokyo", "users": 5420}
Claude → call get_weather(city="Tokyo")
     ← result: {"temp": 22, "condition": "Sunny"}
Claude: "Your most popular city is Tokyo (5,420 users).
         It's currently 22°C and sunny there."
```

This happens automatically -- your tool use loop (the `while response.stop_reason == "tool_use"` pattern) handles chains of arbitrary length.

### 10.2 Tools That Return Images

Tools can return images to Claude for visual analysis:

```python
import base64

def take_screenshot() -> dict:
    """Capture a screenshot and return as base64."""
    # ... capture screenshot ...
    with open("screenshot.png", "rb") as f:
        image_data = base64.standard_b64encode(f.read()).decode("utf-8")
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": "image/png",
            "data": image_data,
        },
    }

# Return image content in tool_result
tool_result = {
    "type": "tool_result",
    "tool_use_id": block.id,
    "content": [
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": image_data,
            },
        },
        {
            "type": "text",
            "text": "Screenshot captured at 2026-02-22 10:30:00",
        },
    ],
}
```

### 10.3 Streaming with Tool Use

Tool use works with streaming, but requires careful event handling:

```python
import anthropic

client = anthropic.Anthropic()

def stream_with_tools(user_message: str, tools: list):
    """Stream responses while handling tool use."""
    messages = [{"role": "user", "content": user_message}]

    while True:
        # Use stream for the API call
        with client.messages.stream(
            model="claude-sonnet-4-20250514",
            max_tokens=2048,
            tools=tools,
            messages=messages,
        ) as stream:
            # Print text tokens as they arrive
            for text in stream.text_stream:
                print(text, end="", flush=True)

            # Get the final message to check for tool use
            response = stream.get_final_message()

        # If no tool use, we are done
        if response.stop_reason != "tool_use":
            print()  # Final newline
            break

        # Handle tool calls
        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                print(f"\n  [Calling {block.name}...]")
                result = execute_tool(block.name, block.input)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": json.dumps(result),
                })

        # Add to conversation and continue
        messages.append({"role": "assistant", "content": response.content})
        messages.append({"role": "user", "content": tool_results})
```

### 10.4 Structured Output Without Tool Execution

You can use tools purely for structured output extraction without actually executing anything:

```python
# Define a "tool" that is just a schema for structured output
sentiment_tool = {
    "name": "classify_sentiment",
    "description": "Classify the sentiment of a text.",
    "input_schema": {
        "type": "object",
        "properties": {
            "sentiment": {
                "type": "string",
                "enum": ["positive", "negative", "neutral", "mixed"],
            },
            "confidence": {
                "type": "number",
                "minimum": 0,
                "maximum": 1,
            },
            "key_phrases": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Key phrases that influenced the classification",
            },
        },
        "required": ["sentiment", "confidence", "key_phrases"],
    },
}

# Force Claude to use this tool
response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    tools=[sentiment_tool],
    tool_choice={"type": "tool", "name": "classify_sentiment"},
    messages=[
        {"role": "user", "content": "Analyze: 'The product is great but shipping was terrible.'"}
    ],
)

# Extract structured data -- no need to "execute" the tool
for block in response.content:
    if block.type == "tool_use":
        result = block.input
        print(f"Sentiment: {result['sentiment']}")
        print(f"Confidence: {result['confidence']}")
        print(f"Key phrases: {result['key_phrases']}")
```

---

## 11. Best Practices

### 11.1 Tool Design Principles

```
┌─────────────────────────────────────────────────────────────────┐
│                Tool Design Best Practices                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. Single Responsibility                                        │
│     Each tool does ONE thing well.                               │
│     BAD:  "manage_database" (too broad)                          │
│     GOOD: "query_users", "insert_order", "get_schema"           │
│                                                                  │
│  2. Clear Descriptions                                           │
│     Write descriptions as if for a human colleague.              │
│     Include: what it does, what it returns, when to use it.      │
│                                                                  │
│  3. Descriptive Parameters                                       │
│     Every parameter needs a description.                         │
│     Use enums for fixed sets of valid values.                    │
│     Use constraints (min/max) for numeric parameters.            │
│                                                                  │
│  4. Safe by Default                                              │
│     Read-only tools should not have write side effects.          │
│     Destructive tools should require explicit confirmation.      │
│     Validate inputs before executing.                            │
│                                                                  │
│  5. Informative Errors                                           │
│     Return actionable error messages.                            │
│     Use is_error flag so Claude knows the call failed.           │
│     BAD:  {"error": "failed"}                                    │
│     GOOD: {"error": "City 'Atlantis' not found. Available       │
│            cities: London, Tokyo, New York, Seoul."}             │
│                                                                  │
│  6. Concise Results                                              │
│     Return only what Claude needs. Large payloads waste tokens.  │
│     Paginate long results.                                       │
│     Summarize rather than dumping raw data.                      │
│                                                                  │
│  7. Minimize Tool Count                                          │
│     More tools = more tokens = higher cost and latency.          │
│     Combine related tools if they share parameters.              │
│     Remove tools that are rarely used.                           │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 11.2 Common Pitfalls

| Pitfall | Problem | Solution |
|---------|---------|----------|
| Vague descriptions | Claude calls wrong tool or wrong parameters | Write detailed, specific descriptions |
| No error handling | Application crashes on tool failure | Wrap every tool execution in try/except |
| Missing tool_result | API error when tool results are incomplete | Always return one result per tool_use block |
| Too many tools | High token usage, slow responses | Limit to 10-20 tools; use tool_choice for subsets |
| Sensitive operations | Claude might call destructive tools | Add confirmation steps for write/delete operations |
| Unbounded output | Tool returns 10,000 rows | Paginate, limit, or summarize results |

---

## 12. Exercises

### Exercise 1: Calculator Tool (Beginner)

Build a tool use application with a `calculate` tool that safely evaluates mathematical expressions. Test it with: "What's 15% of 340?", "What's the square root of 144?", "If I invest $1000 at 5% annual interest for 10 years, what's the final amount?"

### Exercise 2: Multi-Tool Assistant (Intermediate)

Create an assistant with three tools:
1. `search_web` -- simulates a web search (return mock results)
2. `get_current_time` -- returns the current date and time
3. `take_notes` -- saves a note to a list

Test with: "Search for the latest Python release, note the version number, and tell me what time it is." Verify that Claude chains the tools correctly.

### Exercise 3: Structured Data Extraction (Intermediate)

Use the `tool_choice` forced-tool pattern to extract structured data from unstructured text. Process a list of 5 job posting descriptions and extract: title, company, location, salary range, required skills, and experience level. Output the results as a JSON array.

### Exercise 4: Conversational Database Agent (Advanced)

Build a conversational agent that can query a SQLite database. Provide three tools: `list_tables`, `describe_table`, `run_query`. Implement full error handling, input validation (block non-SELECT queries), and result pagination. Test with natural language questions like "Show me all orders from last month" and "Which customer has spent the most?"

### Exercise 5: Tool Use with Streaming (Advanced)

Extend Exercise 4 with streaming. Display Claude's thinking text in real time, show a spinner while tools execute, and present the final answer with token usage statistics. Handle edge cases: what happens when a tool times out? What happens when Claude requests a tool that does not exist?

---

## 13. References

- Tool Use Documentation - https://docs.anthropic.com/en/docs/build-with-claude/tool-use
- Anthropic Cookbook: Tool Use - https://github.com/anthropics/anthropic-cookbook/tree/main/tool_use
- JSON Schema Reference - https://json-schema.org/understanding-json-schema/
- Anthropic Python SDK - https://github.com/anthropics/anthropic-sdk-python
- Anthropic TypeScript SDK - https://github.com/anthropics/anthropic-sdk-typescript

---

## Next Lesson

[17. Claude Agent SDK](./17_Claude_Agent_SDK.md) introduces the Agent SDK -- a higher-level abstraction that wraps the tool use loop, context management, and agent orchestration into a programmable interface. You will learn how to create agents that autonomously solve multi-step problems.
