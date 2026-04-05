# Exercise: Lesson 01 — Prompt Fundamentals
# Complete the TODO items below.
#
# Run: python 01_prompt_fundamentals.py

import anthropic

client = anthropic.Anthropic()  # expects ANTHROPIC_API_KEY env var


# === Exercise 1: Build a Complete Prompt ===
# A well-structured prompt has 4 components:
#   1. Role/Context  2. Task  3. Constraints  4. Output format
# Hint: Use a system message for role, and user message for the rest.

def build_complete_prompt(topic: str) -> dict:
    """Return a dict with 'system' and 'user' keys containing a
    well-structured prompt that asks Claude to explain `topic`.

    Requirements:
      - system: assign a role (e.g., "expert instructor")
      - user: state the task, add at least one constraint
              (e.g., word limit), and specify output format (e.g., bullet list)
    """
    # TODO: Build and return the prompt dict
    # Example structure: {"system": "...", "user": "..."}
    pass


def exercise_1():
    prompt = build_complete_prompt("recursion in programming")
    assert isinstance(prompt, dict), "Must return a dict"
    assert "system" in prompt and "user" in prompt, "Must have system and user keys"
    assert len(prompt["system"]) > 10, "System prompt too short"
    assert len(prompt["user"]) > 20, "User prompt too short"
    print("[Ex1] Prompt built successfully")
    print(f"  system: {prompt['system'][:80]}...")
    print(f"  user:   {prompt['user'][:80]}...")


# === Exercise 2: Send a Prompt to the API ===
# Hint: Use client.messages.create() with model, max_tokens, system, messages.

def send_prompt(system: str, user: str, max_tokens: int = 300) -> str:
    """Send a prompt to Claude and return the text response."""
    # TODO: Call the Anthropic API and return response text
    # Use model="claude-sonnet-4-20250514"
    pass


def exercise_2():
    result = send_prompt(
        system="You are a helpful assistant.",
        user="What are the 3 primary colors? Reply in one sentence.",
    )
    assert isinstance(result, str) and len(result) > 5, "Expected a text response"
    print(f"[Ex2] Response: {result[:120]}")


# === Exercise 3: Compare Temperature Settings ===
# Hint: The temperature parameter (0.0-1.0) controls randomness.
# Lower = more deterministic, higher = more creative.

def compare_temperatures(prompt: str) -> dict[str, list[str]]:
    """Send the same prompt 3 times at temperature 0.0 and 3 times at 1.0.
    Return {"low": [resp1, resp2, resp3], "high": [resp1, resp2, resp3]}.
    """
    # TODO: Make 6 API calls total (3 at temp 0.0, 3 at temp 1.0)
    # Each call: model="claude-sonnet-4-20250514", max_tokens=100
    pass


def exercise_3():
    results = compare_temperatures("Name one unusual fruit.")
    assert "low" in results and "high" in results
    assert len(results["low"]) == 3 and len(results["high"]) == 3
    low_unique = len(set(results["low"]))
    high_unique = len(set(results["high"]))
    print(f"[Ex3] Low-temp unique responses:  {low_unique}/3")
    print(f"[Ex3] High-temp unique responses: {high_unique}/3")


# === Exercise 4: Prompt Component Ablation ===
# Test how removing each component affects response quality.
# Hint: Send 4 variants — full, no-role, no-constraint, no-format —
#       and compare response lengths.

def ablation_study(topic: str) -> dict[str, str]:
    """Return responses for 4 prompt variants:
      "full"          — all 4 components
      "no_role"       — remove system role
      "no_constraint" — remove constraint from user message
      "no_format"     — remove output format instruction
    """
    # TODO: Build 4 prompt variants and send each to the API
    # Return dict mapping variant name to response text
    pass


def exercise_4():
    results = ablation_study("linked lists")
    assert set(results.keys()) == {"full", "no_role", "no_constraint", "no_format"}
    for variant, text in results.items():
        print(f"[Ex4] {variant:>15}: {len(text):>5} chars | {text[:60]}...")


# === Exercise 5: Prompt Rewriter ===
# Given a vague prompt, programmatically add missing components.
# This is a pure Python exercise (no API call needed).

def rewrite_prompt(vague_prompt: str, role: str, constraint: str,
                   output_format: str) -> dict:
    """Take a vague user prompt and return a structured dict with:
      - "system": the role
      - "user": original prompt + constraint + output format instruction
    """
    # TODO: Combine the pieces into a well-structured prompt
    pass


def exercise_5():
    result = rewrite_prompt(
        vague_prompt="Tell me about Python",
        role="You are a senior software engineer and educator.",
        constraint="Keep your answer under 150 words.",
        output_format="Format as a numbered list of key points.",
    )
    assert "system" in result and "user" in result
    assert "Python" in result["user"]
    assert "150" in result["user"] or "numbered" in result["user"]
    print(f"[Ex5] Rewritten system: {result['system'][:60]}")
    print(f"[Ex5] Rewritten user:   {result['user'][:80]}...")


if __name__ == "__main__":
    print("=== Exercise 1: Build a Complete Prompt ===")
    exercise_1()

    print("\n=== Exercise 2: Send a Prompt to the API ===")
    exercise_2()

    print("\n=== Exercise 3: Compare Temperature Settings ===")
    exercise_3()

    print("\n=== Exercise 4: Prompt Component Ablation ===")
    exercise_4()

    print("\n=== Exercise 5: Prompt Rewriter (no API) ===")
    exercise_5()

    print("\nAll exercises completed!")
