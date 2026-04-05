# Exercise: Lesson 16 — Agent Prompting Patterns
# Complete the TODO items below.
#
# Run: python 16_agent_prompting.py

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable


# === Exercise 1: ReAct Prompt Builder ===
# Build a Reasoning + Acting (ReAct) prompt that interleaves
# Thought, Action, and Observation steps.

AVAILABLE_TOOLS = {
    "search": "Search the web for information. Input: query string.",
    "calculator": "Evaluate a math expression. Input: expression string.",
    "lookup": "Look up a fact in the knowledge base. Input: topic string.",
}


def build_react_prompt(
    question: str,
    tools: dict[str, str],
    max_steps: int = 5,
) -> str:
    """Build a ReAct-style system prompt.

    The prompt should instruct the model to follow this loop:
      Thought: <reasoning about what to do next>
      Action: <tool_name>[<input>]
      Observation: <result from tool>
      ... repeat up to max_steps ...
      Thought: I have enough information.
      Final Answer: <answer>

    Hint: List each tool with its name and description. Include a
    worked example showing one full Thought/Action/Observation cycle.
    """
    # TODO: List available tools and their descriptions
    # TODO: Write the ReAct loop format specification
    # TODO: Include a brief example cycle
    # TODO: Append the question
    pass


def exercise_1():
    """Verify the ReAct prompt has all required components."""
    prompt = build_react_prompt("What is 15% of 280?", AVAILABLE_TOOLS)
    assert prompt is not None, "Must return a string"
    lower = prompt.lower()
    assert "thought" in lower, "Must include Thought step"
    assert "action" in lower, "Must include Action step"
    assert "observation" in lower, "Must include Observation step"
    assert "final answer" in lower, "Must include Final Answer step"
    assert "calculator" in lower, "Must list available tools"
    print(f"  ReAct prompt: {len(prompt)} chars")
    print("  All ReAct components present")


# === Exercise 2: Tool Dispatcher ===
# Parse tool calls from model output and dispatch to handlers.

TOOL_HANDLERS: dict[str, Callable[[str], str]] = {
    "search": lambda q: f"Results for '{q}': [Wikipedia article found]",
    "calculator": lambda expr: str(eval(expr)),  # Safe for exercise only
    "lookup": lambda topic: f"Fact: {topic} is a common CS concept.",
}


def parse_action(text: str) -> tuple[str, str] | None:
    """Parse an Action line from model output.

    Expected format: 'Action: tool_name[input_text]'

    Returns (tool_name, input_text) or None if no action found.

    Hint: Use a regex like r'Action:\\s*(\\w+)\\[(.+?)\\]'
    """
    # TODO: Use regex to extract tool name and input
    # TODO: Return the tuple or None
    pass


def dispatch_tool(tool_name: str, tool_input: str) -> str:
    """Execute the specified tool and return its output.

    Hint: Look up the tool in TOOL_HANDLERS. Return an error string
    if the tool is not found.
    """
    # TODO: Look up handler and execute
    # TODO: Handle unknown tools gracefully
    pass


def exercise_2():
    """Verify tool parsing and dispatch."""
    result = parse_action("Thought: I need to calculate.\nAction: calculator[15 * 280 / 100]")
    assert result is not None, "Must parse the action"
    tool_name, tool_input = result
    assert tool_name == "calculator"
    assert "15" in tool_input

    output = dispatch_tool(tool_name, tool_input)
    assert output == "42.0", f"Expected '42.0', got '{output}'"

    none_result = parse_action("Thought: Just thinking, no action.")
    assert none_result is None, "Should return None when no action present"

    error = dispatch_tool("unknown_tool", "test")
    assert "error" in error.lower() or "unknown" in error.lower()

    print(f"  Parsed: {tool_name}[{tool_input}]")
    print(f"  Result: {output}")


# === Exercise 3: Simulated ReAct Loop ===
# Simulate a multi-step ReAct execution loop.

@dataclass
class ReActStep:
    step_num: int
    thought: str
    action: str | None = None
    action_input: str | None = None
    observation: str | None = None


def simulate_react_loop(question: str, max_steps: int = 3) -> list[ReActStep]:
    """Simulate a ReAct loop for a simple question.

    For this exercise, simulate the model's behavior:
      Step 1: Think about what tool to use, call it
      Step 2: Process the observation, decide if more info needed
      Step 3: Provide final answer

    Hint: Hardcode a plausible reasoning trace. In production, each
    step would be an API call with the accumulated context.
    Create at least 2 steps where one has an action/observation
    and the last one has thought only (the final answer).
    """
    # TODO: Create step 1 with thought + action + observation
    # TODO: Create step 2 (or more) with final reasoning
    # TODO: Return the list of steps
    pass


def exercise_3():
    """Verify the simulated loop runs correctly."""
    steps = simulate_react_loop("What is 15% of 280?")
    assert steps is not None and len(steps) >= 2, "Need at least 2 steps"
    assert steps[0].action is not None, "First step should have an action"
    assert steps[0].observation is not None, "First step should have observation"
    assert steps[-1].thought is not None, "Last step should have a thought"
    for step in steps:
        action_str = f" -> {step.action}[{step.action_input}]" if step.action else ""
        obs_str = f" => {step.observation}" if step.observation else ""
        print(f"  Step {step.step_num}: {step.thought[:50]}{action_str}{obs_str}")


# === Exercise 4: Tool-Use Schema Builder ===
# Build Anthropic-style tool definitions for an agent.

def build_tool_definitions() -> list[dict[str, Any]]:
    """Build a list of tool definitions in Anthropic API format.

    Create at least 3 tools with proper JSON Schema input_schema.
    Each tool needs: name, description, input_schema.

    Required tools:
      1. web_search: query (string, required)
      2. get_weather: city (string, required), units (string, enum, optional)
      3. run_code: language (string, enum, required), code (string, required)

    Hint: Follow the Anthropic tool definition format:
    {"name": "...", "description": "...",
     "input_schema": {"type": "object", "properties": {...}, "required": [...]}}
    """
    # TODO: Define web_search tool schema
    # TODO: Define get_weather tool schema
    # TODO: Define run_code tool schema
    # TODO: Return list of all tool definitions
    pass


def validate_tool_schema(tool: dict) -> list[str]:
    """Validate a tool definition and return a list of issues.

    Check for: name, description, input_schema, properties, required.
    Return an empty list if valid.
    """
    # TODO: Check required top-level keys
    # TODO: Check input_schema structure
    # TODO: Return list of validation issues
    pass


def exercise_4():
    """Verify tool definitions are well-formed."""
    tools = build_tool_definitions()
    assert tools is not None and len(tools) >= 3, "Need at least 3 tools"
    for tool in tools:
        issues = validate_tool_schema(tool)
        assert len(issues) == 0, f"Tool '{tool.get('name')}' has issues: {issues}"
        props = list(tool["input_schema"]["properties"].keys())
        print(f"  {tool['name']}: params={props}")
    print(f"  All {len(tools)} tool schemas valid")


if __name__ == "__main__":
    print("=== Exercise 1: ReAct Prompt ===")
    exercise_1()

    print("=== Exercise 2: Tool Dispatcher ===")
    exercise_2()

    print("=== Exercise 3: Simulated ReAct Loop ===")
    exercise_3()

    print("=== Exercise 4: Tool-Use Schemas ===")
    exercise_4()

    print("\nAll exercises completed!")
