# Claude Agent SDK

**Previous**: [16. Tool Use and Function Calling](./16_Tool_Use_and_Function_Calling.md) | **Next**: [18. Building Custom Agents](./18_Building_Custom_Agents.md)

---

The Claude Agent SDK (`claude-code-sdk`) provides programmatic access to the same agent capabilities that power the Claude Code CLI. Instead of the low-level tool use loop you built manually in Lesson 16, the Agent SDK gives you a high-level interface: define a task, configure the agent's tools and permissions, and let the agent loop handle the rest -- thinking, acting, observing, and iterating until the task is complete. This lesson covers the SDK's architecture, core concepts, configuration, and practical usage patterns.

**Difficulty**: ⭐⭐⭐

**Prerequisites**:
- Understanding of tool use from Lesson 16
- Python 3.10+ or Node.js 18+
- Claude Code CLI installed (the SDK depends on it)
- Familiarity with async programming (Python asyncio or TypeScript async/await)

**Learning Objectives**:
- Understand the Agent SDK's architecture and relationship to Claude Code
- Install and configure the SDK in Python and TypeScript
- Create agents with custom system prompts and tool configurations
- Process streaming events from agent execution
- Configure model selection, turn limits, and permission settings
- Integrate MCP servers with SDK-based agents
- Handle errors, retries, and edge cases in agent workflows

---

## Table of Contents

1. [What Is the Agent SDK?](#1-what-is-the-agent-sdk)
2. [Architecture](#2-architecture)
3. [Installation and Setup](#3-installation-and-setup)
4. [Core Concepts](#4-core-concepts)
5. [Creating an Agent (Python)](#5-creating-an-agent-python)
6. [Creating an Agent (TypeScript)](#6-creating-an-agent-typescript)
7. [Configuration Options](#7-configuration-options)
8. [Working with Agent Responses](#8-working-with-agent-responses)
9. [Error Handling and Retries](#9-error-handling-and-retries)
10. [Hooks in the SDK Context](#10-hooks-in-the-sdk-context)
11. [MCP Server Integration](#11-mcp-server-integration)
12. [Exercises](#12-exercises)
13. [References](#13-references)

---

## 1. What Is the Agent SDK?

The Agent SDK is a library that wraps the Claude Code agent loop into a programmable interface. It lets you build applications that leverage the same capabilities as the Claude Code CLI -- reading files, writing code, running commands, searching the web -- but controlled from your own code.

```
┌─────────────────────────────────────────────────────────────────┐
│                   Abstraction Levels                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Level 3: Claude Code CLI                                        │
│  ┌─────────────────────────────────────────────────────┐        │
│  │  Interactive terminal interface                      │        │
│  │  Human-in-the-loop (permission prompts)             │        │
│  │  Session management, conversation history            │        │
│  └─────────────────────────────────────────────────────┘        │
│                                                                  │
│  Level 2: Claude Agent SDK    ◀── THIS LESSON                   │
│  ┌─────────────────────────────────────────────────────┐        │
│  │  Programmatic agent control                          │        │
│  │  Built-in tools (Read, Write, Bash, Glob, etc.)     │        │
│  │  Automatic agent loop (think → act → observe)        │        │
│  │  Context window management and compaction            │        │
│  └─────────────────────────────────────────────────────┘        │
│                                                                  │
│  Level 1: Claude API (Messages + Tool Use)                       │
│  ┌─────────────────────────────────────────────────────┐        │
│  │  Raw API calls with manual tool use loop            │        │
│  │  You manage conversation state, tools, retries      │        │
│  │  Maximum control, maximum complexity                 │        │
│  └─────────────────────────────────────────────────────┘        │
│                                                                  │
│  Level 0: HTTP / REST                                            │
│  ┌─────────────────────────────────────────────────────┐        │
│  │  Raw HTTP requests to api.anthropic.com              │        │
│  │  No SDK, manual JSON construction                    │        │
│  └─────────────────────────────────────────────────────┘        │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**Key differences from the raw API:**
- **Built-in tools**: File operations (Read, Write, Edit, Glob, Grep), shell (Bash), web (WebFetch, WebSearch) are included automatically.
- **Agent loop**: The SDK automatically handles the think-act-observe cycle. You don't write the `while stop_reason == "tool_use"` loop.
- **Context management**: The SDK handles context window limits, compacting old messages when the window fills up.
- **Permission model**: Configurable permissions control which tools the agent can use and what it can access.

---

## 2. Architecture

The Agent SDK communicates with the Claude Code CLI process, which in turn calls the Claude API:

```
┌─────────────────────────────────────────────────────────────────┐
│                   SDK Architecture                               │
│                                                                   │
│  Your Application                                                 │
│  ┌──────────────────┐                                            │
│  │  Python / TS     │                                            │
│  │  Application     │                                            │
│  │  Code            │                                            │
│  └────────┬─────────┘                                            │
│           │  SDK API calls                                        │
│           ▼                                                       │
│  ┌──────────────────┐                                            │
│  │  claude-code-sdk │  Agent SDK library                         │
│  │  ┌────────────┐  │                                            │
│  │  │ Agent Loop │  │  Think → Act → Observe → Repeat           │
│  │  └────────────┘  │                                            │
│  │  ┌────────────┐  │                                            │
│  │  │ Tool Mgr   │  │  Built-in + custom tools                  │
│  │  └────────────┘  │                                            │
│  │  ┌────────────┐  │                                            │
│  │  │ Context Mgr│  │  Window management, compaction            │
│  │  └────────────┘  │                                            │
│  └────────┬─────────┘                                            │
│           │  Subprocess communication                             │
│           ▼                                                       │
│  ┌──────────────────┐                                            │
│  │  Claude Code CLI │  claude binary                             │
│  └────────┬─────────┘                                            │
│           │  HTTPS                                                │
│           ▼                                                       │
│  ┌──────────────────┐                                            │
│  │  Claude API      │  api.anthropic.com                         │
│  └──────────────────┘                                            │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

The agent loop follows a fixed cycle:

```
┌──────────┐     ┌──────────┐     ┌──────────┐
│  THINK   │────▶│   ACT    │────▶│ OBSERVE  │──┐
│          │     │          │     │          │  │
│ Analyze  │     │ Use tool │     │ Process  │  │
│ the task │     │ (Read,   │     │ tool     │  │
│ and plan │     │  Write,  │     │ results  │  │
│ next step│     │  Bash,   │     │          │  │
│          │     │  etc.)   │     │          │  │
└──────────┘     └──────────┘     └────┬─────┘  │
     ▲                                 │        │
     │                                 │        │
     │     ┌──────────────┐            │        │
     │     │   COMPLETE   │◀───────────┘        │
     │     │              │  (if task done)     │
     │     │ Return final │                     │
     │     │ result       │                     │
     │     └──────────────┘                     │
     │                                          │
     └──────────────────────────────────────────┘
              (if more work needed)
```

---

## 3. Installation and Setup

### 3.1 Prerequisites

The Agent SDK requires the Claude Code CLI to be installed:

```bash
# Install Claude Code CLI (if not already installed)
npm install -g @anthropic-ai/claude-code

# Verify installation
claude --version
```

### 3.2 Python SDK

```bash
# Install the Python SDK
pip install claude-code-sdk

# Or with uv
uv add claude-code-sdk
```

### 3.3 TypeScript SDK

```bash
# Install the TypeScript SDK
npm install @anthropic-ai/claude-code-sdk

# Or with your preferred package manager
pnpm add @anthropic-ai/claude-code-sdk
yarn add @anthropic-ai/claude-code-sdk
```

### 3.4 Authentication

The SDK uses the same authentication as Claude Code CLI. Ensure you are authenticated:

```bash
# Option 1: Log in interactively
claude login

# Option 2: Set API key directly
export ANTHROPIC_API_KEY="sk-ant-..."
```

---

## 4. Core Concepts

### 4.1 The Agent Loop

The agent loop is the core execution model. You provide a task (a natural language prompt), and the agent iterates through think-act-observe cycles until the task is complete or a limit is reached.

Each iteration:
1. **Think**: Claude analyzes the current state and decides what to do next.
2. **Act**: Claude calls a tool (read a file, run a command, search, etc.).
3. **Observe**: The tool result is returned to Claude for analysis.
4. **Decide**: Claude either performs another action or declares the task complete.

### 4.2 Built-In Tools

The Agent SDK includes all of Claude Code's built-in tools:

```
┌─────────────────────────────────────────────────────────────────┐
│                   Built-In Tools                                 │
├──────────────────┬──────────────────────────────────────────────┤
│ Category         │ Tools                                        │
├──────────────────┼──────────────────────────────────────────────┤
│ File System      │ Read     - Read file contents                │
│                  │ Write    - Create or overwrite files          │
│                  │ Edit     - Make targeted edits to files       │
│                  │ Glob     - Find files by pattern              │
│                  │ Grep     - Search file contents (ripgrep)     │
├──────────────────┼──────────────────────────────────────────────┤
│ Shell            │ Bash     - Execute shell commands             │
├──────────────────┼──────────────────────────────────────────────┤
│ Web              │ WebFetch - Fetch and parse web pages          │
│                  │ WebSearch - Search the web                    │
├──────────────────┼──────────────────────────────────────────────┤
│ Jupyter          │ NotebookEdit - Edit Jupyter notebook cells    │
├──────────────────┼──────────────────────────────────────────────┤
│ MCP              │ Any tools exposed by connected MCP servers    │
└──────────────────┴──────────────────────────────────────────────┘
```

### 4.3 Context Window Management

The SDK automatically manages the context window:
- As the conversation grows, older messages are **compacted** (summarized) to make room for new content.
- File contents read earlier may be summarized if they are no longer immediately relevant.
- The system prompt and recent messages are always preserved in full.

### 4.4 Streaming Events

Agent execution produces a stream of events that your application can process:

```
┌─────────────────────────────────────────────────────────────────┐
│                   Event Types                                    │
├──────────────────┬──────────────────────────────────────────────┤
│ Event            │ Description                                   │
├──────────────────┼──────────────────────────────────────────────┤
│ assistant        │ Text output from the agent (thinking/answer) │
│ tool_use         │ Agent is calling a tool                       │
│ tool_result      │ Result from a tool execution                  │
│ result           │ Final result of the agent task                │
│ error            │ Error during execution                        │
└──────────────────┴──────────────────────────────────────────────┘
```

---

## 5. Creating an Agent (Python)

### 5.1 Basic Usage

```python
import asyncio
from claude_code_sdk import query, ClaudeCodeOptions, Message

async def main():
    # Simple one-shot task
    messages: list[Message] = []

    async for message in query(
        prompt="Read the file README.md and summarize its contents.",
        options=ClaudeCodeOptions(
            max_turns=10,  # Maximum agent loop iterations
        ),
    ):
        if message.type == "assistant":
            # Agent's text output (thinking, explanations, final answer)
            for block in message.content:
                if hasattr(block, "text"):
                    print(block.text, end="")
        elif message.type == "tool_use":
            # Agent is using a tool
            print(f"\n  [Tool: {message.tool_name}]")
        elif message.type == "result":
            print(f"\n\n--- Task Complete ---")

asyncio.run(main())
```

### 5.2 With System Prompt

```python
import asyncio
from claude_code_sdk import query, ClaudeCodeOptions

async def code_review(file_path: str):
    """Run an automated code review on a file."""
    system_prompt = """You are a senior code reviewer. Analyze the given file for:
1. Code quality and readability
2. Potential bugs or edge cases
3. Performance considerations
4. Security vulnerabilities
5. Adherence to best practices

Provide your review in a structured format with severity levels
(critical, warning, suggestion) for each finding."""

    async for message in query(
        prompt=f"Review the code in {file_path} and provide detailed feedback.",
        options=ClaudeCodeOptions(
            system_prompt=system_prompt,
            max_turns=5,
        ),
    ):
        if message.type == "assistant":
            for block in message.content:
                if hasattr(block, "text"):
                    print(block.text, end="")

    print()  # Final newline

asyncio.run(code_review("src/app.py"))
```

### 5.3 Processing Structured Results

```python
import asyncio
import json
from claude_code_sdk import query, ClaudeCodeOptions

async def analyze_codebase(directory: str) -> dict:
    """Analyze a codebase and return structured results."""
    prompt = f"""Analyze the codebase in {directory}. Return a JSON object with:
{{
    "total_files": <number>,
    "languages": {{"python": <count>, "javascript": <count>, ...}},
    "largest_files": [
        {{"path": "<path>", "lines": <count>}},
        ...
    ],
    "potential_issues": [
        {{"file": "<path>", "issue": "<description>", "severity": "high|medium|low"}},
        ...
    ]
}}

Use the Glob and Read tools to explore the codebase. Return ONLY the JSON object."""

    full_response = ""
    async for message in query(
        prompt=prompt,
        options=ClaudeCodeOptions(max_turns=20),
    ):
        if message.type == "assistant":
            for block in message.content:
                if hasattr(block, "text"):
                    full_response += block.text

    # Extract JSON from the response
    try:
        # Find JSON in the response
        json_start = full_response.index("{")
        json_end = full_response.rindex("}") + 1
        return json.loads(full_response[json_start:json_end])
    except (ValueError, json.JSONDecodeError):
        return {"error": "Failed to parse structured response", "raw": full_response}

result = asyncio.run(analyze_codebase("/path/to/project"))
print(json.dumps(result, indent=2))
```

---

## 6. Creating an Agent (TypeScript)

### 6.1 Basic Usage

```typescript
import { query, ClaudeCodeOptions, Message } from "@anthropic-ai/claude-code-sdk";

async function main() {
  const options: ClaudeCodeOptions = {
    maxTurns: 10,
  };

  for await (const message of query({
    prompt: "List all Python files in the current directory and count the total lines of code.",
    options,
  })) {
    switch (message.type) {
      case "assistant":
        for (const block of message.content) {
          if ("text" in block) {
            process.stdout.write(block.text);
          }
        }
        break;

      case "tool_use":
        console.log(`\n  [Tool: ${message.toolName}]`);
        break;

      case "result":
        console.log("\n\n--- Task Complete ---");
        break;
    }
  }
}

main();
```

### 6.2 With Configuration

```typescript
import { query, ClaudeCodeOptions } from "@anthropic-ai/claude-code-sdk";

async function generateTests(filePath: string) {
  const systemPrompt = `You are a testing expert. Generate comprehensive unit tests
for the given code. Use pytest for Python, vitest for TypeScript.
Cover: happy paths, edge cases, error conditions, and boundary values.`;

  for await (const message of query({
    prompt: `Read ${filePath} and write comprehensive unit tests for it. Save the tests to a file.`,
    options: {
      systemPrompt,
      maxTurns: 15,
      allowedTools: ["Read", "Write", "Glob", "Bash"],
    },
  })) {
    if (message.type === "assistant") {
      for (const block of message.content) {
        if ("text" in block) {
          process.stdout.write(block.text);
        }
      }
    }
  }
}

generateTests("src/utils.ts");
```

---

## 7. Configuration Options

### 7.1 Available Options

```python
from claude_code_sdk import ClaudeCodeOptions

options = ClaudeCodeOptions(
    # Model selection
    model="claude-sonnet-4-20250514",        # Which model to use

    # Turn limits
    max_turns=25,                             # Max agent loop iterations

    # System prompt
    system_prompt="You are a helpful assistant.",

    # Working directory
    cwd="/path/to/project",                   # Set the working directory

    # Permission settings
    permission_mode="default",                # "default", "acceptEdits",
                                              # "bypassPermissions", "plan"

    # Tool restrictions
    allowed_tools=["Read", "Write", "Bash"],  # Whitelist specific tools
    disallowed_tools=["WebFetch"],            # Blacklist specific tools

    # MCP servers
    mcp_servers=[                             # Connect MCP servers
        {
            "name": "my-server",
            "command": "node",
            "args": ["/path/to/server.js"],
        }
    ],
)
```

### 7.2 Model Selection

```python
# Use the most capable model for complex tasks
options = ClaudeCodeOptions(model="claude-opus-4-20250514")

# Use Sonnet for balanced performance/cost
options = ClaudeCodeOptions(model="claude-sonnet-4-20250514")

# Use Haiku for simple, fast tasks
options = ClaudeCodeOptions(model="claude-haiku-3-5-20241022")
```

### 7.3 Permission Modes

```
┌─────────────────┬─────────────────────────────────────────────────┐
│ Mode            │ Description                                      │
├─────────────────┼─────────────────────────────────────────────────┤
│ default         │ Standard permissions with approval for           │
│                 │ file writes and shell commands                    │
├─────────────────┼─────────────────────────────────────────────────┤
│ acceptEdits     │ Auto-approve file edits; still prompt for        │
│                 │ shell commands                                    │
├─────────────────┼─────────────────────────────────────────────────┤
│ bypassPermissions│ Auto-approve everything (use in sandboxed       │
│                 │ environments only)                                │
├─────────────────┼─────────────────────────────────────────────────┤
│ plan            │ Read-only; no writes or shell commands            │
│                 │ (for exploration and analysis)                    │
└─────────────────┴─────────────────────────────────────────────────┘
```

For automated CI/CD or sandboxed environments:

```python
# In a Docker container or CI environment where safety is managed by isolation
options = ClaudeCodeOptions(
    permission_mode="bypassPermissions",
    max_turns=50,
)
```

### 7.4 Tool Restrictions

Control which tools the agent can use:

```python
# Read-only analysis: no writes, no shell
options = ClaudeCodeOptions(
    allowed_tools=["Read", "Glob", "Grep"],
)

# Full development: all tools except web
options = ClaudeCodeOptions(
    disallowed_tools=["WebFetch", "WebSearch"],
)
```

---

## 8. Working with Agent Responses

### 8.1 Event-Driven Processing

```python
import asyncio
from claude_code_sdk import query, ClaudeCodeOptions

async def process_events():
    """Demonstrate processing different event types."""
    tool_calls = []
    text_output = []
    errors = []

    async for message in query(
        prompt="Find all TODO comments in the codebase and list them.",
        options=ClaudeCodeOptions(max_turns=15),
    ):
        if message.type == "assistant":
            for block in message.content:
                if hasattr(block, "text"):
                    text_output.append(block.text)

        elif message.type == "tool_use":
            tool_calls.append({
                "tool": message.tool_name,
                "input": getattr(message, "tool_input", None),
            })

        elif message.type == "error":
            errors.append(str(message))

        elif message.type == "result":
            pass  # Final result

    # Summary
    print(f"Tool calls made: {len(tool_calls)}")
    for tc in tool_calls:
        print(f"  - {tc['tool']}")
    print(f"Errors: {len(errors)}")
    print(f"Response length: {sum(len(t) for t in text_output)} characters")

asyncio.run(process_events())
```

### 8.2 Progress Tracking

```python
import asyncio
import time
from claude_code_sdk import query, ClaudeCodeOptions

async def run_with_progress(prompt: str):
    """Run an agent task with progress tracking."""
    start_time = time.time()
    turn_count = 0
    tool_count = 0

    print(f"Task: {prompt}")
    print("-" * 60)

    async for message in query(
        prompt=prompt,
        options=ClaudeCodeOptions(max_turns=20),
    ):
        if message.type == "assistant":
            turn_count += 1
            elapsed = time.time() - start_time
            print(f"  [Turn {turn_count} | {elapsed:.1f}s]", end="")

            for block in message.content:
                if hasattr(block, "text"):
                    # Print first 100 chars of each text block
                    preview = block.text[:100].replace("\n", " ")
                    print(f" {preview}...")

        elif message.type == "tool_use":
            tool_count += 1
            print(f"    -> Tool: {message.tool_name}")

    total_time = time.time() - start_time
    print("-" * 60)
    print(f"Completed in {total_time:.1f}s | {turn_count} turns | {tool_count} tool calls")

asyncio.run(run_with_progress(
    "Read package.json and suggest dependency updates"
))
```

---

## 9. Error Handling and Retries

### 9.1 Handling Agent Errors

```python
import asyncio
from claude_code_sdk import query, ClaudeCodeOptions

async def safe_agent_run(prompt: str, max_retries: int = 3) -> str:
    """Run an agent task with error handling and retries."""
    for attempt in range(max_retries):
        try:
            result_text = ""
            async for message in query(
                prompt=prompt,
                options=ClaudeCodeOptions(max_turns=10),
            ):
                if message.type == "assistant":
                    for block in message.content:
                        if hasattr(block, "text"):
                            result_text += block.text

                elif message.type == "error":
                    raise RuntimeError(f"Agent error: {message}")

            return result_text

        except RuntimeError as e:
            if attempt < max_retries - 1:
                wait = 2 ** attempt
                print(f"Attempt {attempt + 1} failed: {e}. Retrying in {wait}s...")
                await asyncio.sleep(wait)
            else:
                raise

        except Exception as e:
            print(f"Unexpected error: {type(e).__name__}: {e}")
            raise

    return ""  # Should not reach here

result = asyncio.run(safe_agent_run("Summarize the project structure"))
print(result)
```

### 9.2 Turn Limit Handling

When the agent reaches `max_turns`, it stops even if the task is not complete. Handle this:

```python
async def run_with_continuation(prompt: str, max_total_turns: int = 50):
    """Run an agent task, continuing if the turn limit is hit."""
    turns_used = 0
    batch_size = 10
    full_result = ""
    continuation_prompt = prompt

    while turns_used < max_total_turns:
        remaining = min(batch_size, max_total_turns - turns_used)
        task_complete = False

        async for message in query(
            prompt=continuation_prompt,
            options=ClaudeCodeOptions(max_turns=remaining),
        ):
            if message.type == "assistant":
                for block in message.content:
                    if hasattr(block, "text"):
                        full_result += block.text
                turns_used += 1

            elif message.type == "result":
                task_complete = True

        if task_complete:
            break

        # If not complete, continue with context
        continuation_prompt = (
            "Continue the previous task. Here is what you have done so far:\n"
            f"{full_result[-500:]}\n\n"  # Last 500 chars of context
            "Please continue and complete the task."
        )

    return full_result
```

---

## 10. Hooks in the SDK Context

Hooks (covered in Lesson 5) also work with the Agent SDK. They let you intercept and modify agent behavior:

```python
# .claude/settings.json (in the project directory)
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Bash",
        "hooks": [
          {
            "type": "command",
            "command": "python /path/to/validate_command.py"
          }
        ]
      }
    ],
    "PostToolUse": [
      {
        "matcher": "Write",
        "hooks": [
          {
            "type": "command",
            "command": "python /path/to/lint_file.py"
          }
        ]
      }
    ]
  }
}
```

When the agent runs in a directory with these hook configurations, the hooks fire automatically. This is useful for:
- **Validation**: Check commands before execution
- **Linting**: Auto-lint files after the agent writes them
- **Logging**: Track all tool invocations for audit
- **Security**: Block dangerous operations

### Programmatic Hook-Like Behavior

If you want hook-like behavior without the settings file, process events in your streaming loop:

```python
async def agent_with_guards(prompt: str):
    """Run an agent with programmatic safety checks."""
    blocked_patterns = ["rm -rf", "DROP TABLE", "sudo"]

    async for message in query(
        prompt=prompt,
        options=ClaudeCodeOptions(max_turns=15),
    ):
        if message.type == "tool_use":
            # Check tool inputs for dangerous patterns
            tool_input = str(getattr(message, "tool_input", ""))
            for pattern in blocked_patterns:
                if pattern in tool_input:
                    print(f"  [BLOCKED] Tool {message.tool_name} "
                          f"contains blocked pattern: {pattern}")
                    # Note: this only logs; the SDK manages actual execution.
                    # Use hooks or permission_mode for actual blocking.

        if message.type == "assistant":
            for block in message.content:
                if hasattr(block, "text"):
                    print(block.text, end="")
```

---

## 11. MCP Server Integration

The Agent SDK can connect to MCP servers, extending the agent's capabilities with custom tools and resources:

```python
import asyncio
from claude_code_sdk import query, ClaudeCodeOptions

async def agent_with_mcp():
    """Run an agent with MCP server integration."""
    options = ClaudeCodeOptions(
        max_turns=15,
        mcp_servers=[
            {
                "name": "database",
                "command": "python",
                "args": ["/path/to/db-mcp-server/server.py"],
                "env": {
                    "DB_PATH": "/path/to/production.db",
                },
            },
            {
                "name": "weather",
                "command": "node",
                "args": ["/path/to/weather-server/dist/index.js"],
                "env": {
                    "WEATHER_API_KEY": "your-key-here",
                },
            },
        ],
    )

    async for message in query(
        prompt="Query the database for all users in Tokyo, "
               "then check the current weather there.",
        options=options,
    ):
        if message.type == "assistant":
            for block in message.content:
                if hasattr(block, "text"):
                    print(block.text, end="")
        elif message.type == "tool_use":
            print(f"\n  [Tool: {message.tool_name}]")

asyncio.run(agent_with_mcp())
```

The agent can use both built-in tools (Read, Bash, etc.) and MCP tools (database queries, weather lookups) in the same task, chaining them as needed.

### Remote MCP Servers

For remote MCP servers using Streamable HTTP transport:

```python
options = ClaudeCodeOptions(
    mcp_servers=[
        {
            "name": "remote-db",
            "type": "url",
            "url": "https://mcp.internal.company.com/database",
            "headers": {
                "Authorization": "Bearer your-token-here",
            },
        },
    ],
)
```

---

## 12. Exercises

### Exercise 1: Simple Agent Task (Beginner)

Write a Python script that uses the Agent SDK to read a Python file, count the number of functions defined in it, and print a summary. Test it on one of your own files.

### Exercise 2: Code Analysis Pipeline (Intermediate)

Build an agent pipeline that:
1. Scans a project directory for all Python files.
2. For each file, checks for missing docstrings, type hints, and test coverage.
3. Generates a report in Markdown format.
4. Saves the report to `code_analysis_report.md`.

Use the Agent SDK with appropriate tool restrictions (Read, Glob, Grep, Write).

### Exercise 3: Event-Driven Dashboard (Intermediate)

Create a script that runs an agent and displays a real-time dashboard showing:
- Current turn number
- Tools used so far (with counts)
- Elapsed time
- Characters of text generated
- Current status (thinking/tool_use/complete)

Use terminal control characters or a library like `rich` for the display.

### Exercise 4: Multi-Agent Coordination (Advanced)

Build a system that uses multiple Agent SDK calls in sequence:
1. Agent 1 (analysis): Analyze a codebase and identify areas that need refactoring.
2. Agent 2 (planning): Take Agent 1's output and create a detailed refactoring plan.
3. Agent 3 (execution): Execute the refactoring plan.
4. Agent 4 (review): Review the changes and report any issues.

Pass each agent's output as context to the next agent's prompt.

### Exercise 5: Agent with MCP Integration (Advanced)

Create an Agent SDK application that connects to a custom MCP server you build (from Lesson 13). The agent should:
1. Use an MCP tool to fetch data from an external source.
2. Process the data using built-in tools (Bash for computations, Write for output).
3. Generate a summary report.

Test with both stdio and HTTP transport for the MCP server.

---

## 13. References

- Claude Code SDK Documentation - https://docs.anthropic.com/en/docs/claude-code/sdk
- Claude Code SDK Python Package - https://pypi.org/project/claude-code-sdk/
- Claude Code SDK TypeScript Package - https://www.npmjs.com/package/@anthropic-ai/claude-code-sdk
- Claude Code Architecture - https://docs.anthropic.com/en/docs/claude-code/overview
- Agent Loop Documentation - https://docs.anthropic.com/en/docs/claude-code/sdk#the-agent-loop
- MCP Integration - https://docs.anthropic.com/en/docs/claude-code/mcp

---

## Next Lesson

[18. Building Custom Agents](./18_Building_Custom_Agents.md) takes the Agent SDK further, covering custom tool development, system prompt engineering for agents, practical agent examples (code review, documentation, database migration, customer support), testing strategies, and production deployment patterns.
