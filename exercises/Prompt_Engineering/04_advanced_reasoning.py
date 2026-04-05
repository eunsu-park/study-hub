# Exercise: Lesson 04 — Advanced Reasoning
# Complete the TODO items below.
#
# Run: python 04_advanced_reasoning.py

import anthropic
import json

client = anthropic.Anthropic()  # expects ANTHROPIC_API_KEY env var

MODEL = "claude-sonnet-4-20250514"


# === Exercise 1: Tree of Thoughts — Idea Generation ===
# Generate multiple independent solution approaches for a problem.
# Hint: Ask Claude to brainstorm N distinct approaches as a numbered list.

def generate_thoughts(problem: str, n_thoughts: int = 3) -> list[str]:
    """Generate N distinct solution approaches for the given problem.
    Return a list of thought strings.
    """
    # TODO: Prompt Claude to brainstorm n_thoughts distinct approaches
    # TODO: Parse the numbered list into individual thought strings
    # Hint: Use a system prompt like "You are a creative problem solver.
    #        Generate exactly N distinct approaches, numbered 1 to N."
    pass


def exercise_1():
    thoughts = generate_thoughts(
        "How can we reduce latency in a distributed microservices system?",
        n_thoughts=3,
    )
    assert len(thoughts) == 3, f"Expected 3 thoughts, got {len(thoughts)}"
    for i, t in enumerate(thoughts, 1):
        print(f"[Ex1] Thought {i}: {t[:80]}...")


# === Exercise 2: Tree of Thoughts — Evaluation ===
# Evaluate each thought branch and score its promise.
# Hint: Ask Claude to rate each approach on feasibility and impact.

def evaluate_thought(problem: str, thought: str) -> dict:
    """Evaluate a single thought/approach for the given problem.
    Return {"score": float (1-10), "strengths": str, "weaknesses": str}.
    """
    # TODO: Prompt Claude to evaluate this approach, asking for:
    #   - A numeric score from 1-10
    #   - Key strengths (1 sentence)
    #   - Key weaknesses (1 sentence)
    # TODO: Parse the response into the dict format
    # Hint: Ask Claude to respond in JSON format for easy parsing
    pass


def exercise_2():
    problem = "How to make a Python web app handle 10x more traffic?"
    thoughts = [
        "Add a caching layer (Redis) in front of the database.",
        "Rewrite performance-critical paths in Rust via PyO3.",
        "Switch to an async framework and use horizontal scaling.",
    ]
    for thought in thoughts:
        result = evaluate_thought(problem, thought)
        assert "score" in result and "strengths" in result
        print(f"[Ex2] Score={result['score']:>4} | {thought[:50]}")
        print(f"       + {result['strengths'][:60]}")
        print(f"       - {result['weaknesses'][:60]}")


# === Exercise 3: Tree of Thoughts — Selection ===
# Pick the best thought and elaborate on it.

def select_and_elaborate(problem: str, thoughts: list[str]) -> dict:
    """Evaluate all thoughts, select the best, and elaborate on it.
    Return {"selected": str, "score": float, "plan": str}.
    """
    # TODO: Use evaluate_thought() on each thought
    # TODO: Select the one with the highest score
    # TODO: Ask Claude to elaborate the selected thought into a concrete plan
    pass


def exercise_3():
    problem = "Design a study schedule for learning 3 programming languages."
    thoughts = generate_thoughts(problem, n_thoughts=3)
    result = select_and_elaborate(problem, thoughts)
    assert "selected" in result and "plan" in result
    print(f"[Ex3] Selected: {result['selected'][:80]}...")
    print(f"[Ex3] Score: {result['score']}")
    print(f"[Ex3] Plan: {result['plan'][:150]}...")


# === Exercise 4: Self-Refine — Iterative Improvement ===
# Generate a draft, critique it, then refine. Repeat.
# Hint: Use 3 distinct prompts — generate, critique, refine.

def self_refine(task: str, max_iterations: int = 2) -> dict:
    """Implement the Self-Refine loop: generate -> critique -> refine.
    Return {"drafts": list[str], "critiques": list[str], "final": str}.
    """
    drafts = []
    critiques = []

    # TODO: Step 1 — Generate an initial draft for the task
    # TODO: Step 2 — Loop max_iterations times:
    #   a. Critique the current draft (ask Claude for specific improvements)
    #   b. Refine the draft based on the critique
    #   c. Append to drafts and critiques lists
    # TODO: Return the collected drafts, critiques, and final version
    pass


def exercise_4():
    result = self_refine(
        "Write a Python function that checks if a string is a palindrome.",
        max_iterations=2,
    )
    assert len(result["drafts"]) >= 2, "Need at least 2 drafts"
    assert len(result["critiques"]) >= 1, "Need at least 1 critique"
    print(f"[Ex4] Iterations: {len(result['drafts'])}")
    for i, d in enumerate(result["drafts"]):
        print(f"  Draft {i+1}: {d[:80]}...")
    print(f"[Ex4] Final: {result['final'][:100]}...")


# === Exercise 5: Full ToT Pipeline ===
# Combine generation, evaluation, and self-refine into one pipeline.
# This is a pure orchestration exercise (calls functions from above).

def tot_pipeline(problem: str, n_thoughts: int = 3,
                 refine_iterations: int = 1) -> dict:
    """Full Tree-of-Thoughts + Self-Refine pipeline.
    Steps:
      1. Generate n_thoughts approaches
      2. Evaluate and select the best
      3. Self-refine the selected approach
    Return {"thoughts": list, "selected": str, "refined": str}.
    """
    # TODO: Call generate_thoughts()
    # TODO: Call select_and_elaborate()
    # TODO: Call self_refine() on the elaborated plan
    # TODO: Return the combined results
    pass


def exercise_5():
    result = tot_pipeline(
        "Design a CLI tool that summarizes long PDF documents using an LLM.",
        n_thoughts=3,
        refine_iterations=1,
    )
    assert "thoughts" in result and "selected" in result and "refined" in result
    print(f"[Ex5] Generated {len(result['thoughts'])} thoughts")
    print(f"[Ex5] Selected: {result['selected'][:80]}...")
    print(f"[Ex5] Refined:  {result['refined'][:100]}...")


if __name__ == "__main__":
    print("=== Exercise 1: Tree of Thoughts — Generation ===")
    exercise_1()

    print("\n=== Exercise 2: Tree of Thoughts — Evaluation ===")
    exercise_2()

    print("\n=== Exercise 3: Tree of Thoughts — Selection ===")
    exercise_3()

    print("\n=== Exercise 4: Self-Refine ===")
    exercise_4()

    print("\n=== Exercise 5: Full ToT Pipeline ===")
    exercise_5()

    print("\nAll exercises completed!")
