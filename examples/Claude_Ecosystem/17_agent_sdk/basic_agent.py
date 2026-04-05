"""
Claude Agent SDK: Basic Agent Example

Demonstrates creating a simple agent with the Claude Agent SDK
that can read files, search the web, and execute commands.

Requirements:
    pip install claude-agent-sdk
"""

import claude_code_sdk as sdk
import asyncio


async def run_basic_agent():
    """Run a basic Claude agent that analyzes a Python project."""

    # Simple single-turn invocation
    result = await sdk.query(
        prompt="What Python version is required by this project?",
        options=sdk.QueryOptions(
            max_turns=3,
            system_prompt="You are a helpful project analyzer.",
        ),
    )

    # Process the response
    for message in result:
        if message.type == "text":
            print(f"Agent: {message.content}")
        elif message.type == "tool_use":
            print(f"  [Tool] {message.tool}: {message.input}")
        elif message.type == "tool_result":
            print(f"  [Result] {message.content[:200]}...")


async def run_streaming_agent():
    """Run an agent with streaming output."""

    print("=== Streaming Agent ===\n")

    # Streaming invocation — process events as they arrive
    async for event in sdk.query(
        prompt="List all Python files in the current directory and summarize what each does.",
        options=sdk.QueryOptions(
            max_turns=5,
            system_prompt="You are a code analysis assistant. Be concise.",
        ),
    ):
        if event.type == "text":
            print(event.content, end="", flush=True)
        elif event.type == "tool_use":
            print(f"\n  [Calling {event.tool}...]")
        elif event.type == "tool_result":
            pass  # Tool results are processed internally

    print()  # Final newline


async def run_multi_turn_agent():
    """Run an agent that performs multiple turns of reasoning."""

    print("=== Multi-Turn Agent ===\n")

    # The agent will autonomously use tools across multiple turns
    result = await sdk.query(
        prompt=(
            "Analyze this project's test coverage. "
            "Find all test files, check what modules they test, "
            "and identify any modules that lack test coverage."
        ),
        options=sdk.QueryOptions(
            max_turns=10,  # Allow up to 10 tool-use cycles
            system_prompt=(
                "You are a thorough code quality analyst. "
                "Use tools to explore the codebase systematically. "
                "Provide actionable recommendations."
            ),
        ),
    )

    for message in result:
        if message.type == "text":
            print(message.content)


# --- Main ---

if __name__ == "__main__":
    print("Claude Agent SDK — Basic Examples\n")

    print("=" * 60)
    print("Example 1: Single Query")
    print("=" * 60)
    asyncio.run(run_basic_agent())

    print("\n" + "=" * 60)
    print("Example 2: Streaming Output")
    print("=" * 60)
    asyncio.run(run_streaming_agent())

    print("\n" + "=" * 60)
    print("Example 3: Multi-Turn Analysis")
    print("=" * 60)
    asyncio.run(run_multi_turn_agent())
