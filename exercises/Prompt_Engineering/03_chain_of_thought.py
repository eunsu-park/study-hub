# Exercise: Lesson 03 — Chain of Thought
# Complete the TODO items below.
#
# Run: python 03_chain_of_thought.py

import anthropic
import re

client = anthropic.Anthropic()  # expects ANTHROPIC_API_KEY env var

MODEL = "claude-sonnet-4-20250514"


# === Exercise 1: Manual Chain-of-Thought Prompting ===
# Provide step-by-step reasoning examples in the prompt.
# Hint: Show a worked example, then ask for the same pattern on a new problem.

def cot_math_solver(problem: str) -> dict:
    """Solve a math word problem using manual CoT prompting.
    Return {"reasoning": str, "answer": str}.
    """
    # TODO: Build a prompt with a worked CoT example, e.g.:
    #   "Q: If a train travels 60 mph for 2.5 hours, how far does it go?
    #    Step 1: distance = speed x time
    #    Step 2: distance = 60 x 2.5 = 150
    #    Answer: 150 miles"
    # TODO: Ask Claude to solve `problem` following the same step-by-step format
    # TODO: Parse the response to extract reasoning and final answer
    pass


def exercise_1():
    result = cot_math_solver(
        "A store sells notebooks for $4 each. Maria buys 7 notebooks "
        "and pays with a $50 bill. How much change does she receive?"
    )
    assert "reasoning" in result and "answer" in result
    assert len(result["reasoning"]) > 20, "Reasoning too short"
    print(f"[Ex1] Reasoning: {result['reasoning'][:100]}...")
    print(f"[Ex1] Answer: {result['answer']}")


# === Exercise 2: Zero-Shot CoT ===
# Simply append "Let's think step by step" to trigger reasoning.
# Hint: Compare the response with and without the magic phrase.

def zero_shot_cot(question: str) -> str:
    """Ask a question with zero-shot CoT (append 'Let's think step by step').
    Return the full response text.
    """
    # TODO: Append the CoT trigger phrase to the question
    # TODO: Call the API and return the response
    pass


def direct_answer(question: str) -> str:
    """Ask the same question without CoT. Return the response text."""
    # TODO: Call the API with the question as-is (no CoT trigger)
    pass


def exercise_2():
    question = (
        "If you have 3 shirts and 4 pairs of pants, how many different "
        "outfits can you make?"
    )
    cot_resp = zero_shot_cot(question)
    direct_resp = direct_answer(question)
    print(f"[Ex2] Direct:   {direct_resp[:80]}")
    print(f"[Ex2] Zero-CoT: {cot_resp[:120]}")
    assert len(cot_resp) > len(direct_resp), "CoT should produce longer response"
    print("[Ex2] CoT response is longer (more reasoning) -- PASS")


# === Exercise 3: Self-Consistency (Majority Voting) ===
# Sample multiple CoT paths and take the majority answer.
# Hint: Use temperature > 0 for diversity, then extract and vote on answers.

def extract_number(text: str) -> int | None:
    """Extract the last number from a text response."""
    # TODO: Use regex to find all integers in the text
    # TODO: Return the last one found, or None if no number found
    pass


def self_consistency(question: str, n_samples: int = 5) -> dict:
    """Run self-consistency: sample n CoT responses, extract answers, vote.
    Return {"samples": list[str], "answers": list, "majority_answer": any}.
    """
    # TODO: Call zero_shot_cot() n_samples times with temperature=0.7
    #   (you'll need to modify or re-implement the call with temperature)
    # TODO: Extract the numeric answer from each sample
    # TODO: Find the majority answer (most common)
    # Hint: Use collections.Counter
    pass


def exercise_3():
    result = self_consistency(
        "A farmer has 15 apples. He gives 1/3 to his neighbor and eats 2. "
        "How many apples does he have left?",
        n_samples=5,
    )
    assert len(result["samples"]) == 5
    assert result["majority_answer"] is not None
    print(f"[Ex3] Answers extracted: {result['answers']}")
    print(f"[Ex3] Majority answer: {result['majority_answer']}")


# === Exercise 4: CoT for Logical Reasoning ===
# Use CoT to solve a logic puzzle that typically trips up direct answering.

LOGIC_PUZZLE = (
    "All roses are flowers. Some flowers fade quickly. "
    "Can we conclude that some roses fade quickly?"
)

def cot_logic(puzzle: str) -> dict:
    """Solve a logic puzzle with explicit CoT prompting.
    Return {"reasoning": str, "conclusion": str}.
    """
    # TODO: Build a system prompt that instructs step-by-step logical analysis
    # TODO: Ask Claude to identify premises, check validity, then conclude
    # TODO: Parse the response into reasoning and conclusion
    pass


def exercise_4():
    result = cot_logic(LOGIC_PUZZLE)
    assert "reasoning" in result and "conclusion" in result
    assert len(result["reasoning"]) > 30
    print(f"[Ex4] Reasoning: {result['reasoning'][:120]}...")
    print(f"[Ex4] Conclusion: {result['conclusion']}")


# === Exercise 5: CoT Evaluation Harness ===
# Compare direct vs CoT vs self-consistency on a small benchmark.
# This is a pure Python aggregation exercise (reuses functions above).

BENCHMARK = [
    ("What is 17 * 23?", 391),
    ("If 5 machines make 5 widgets in 5 minutes, how many minutes "
     "do 100 machines take to make 100 widgets?", 5),
    ("A bat and ball cost $1.10. The bat costs $1 more than the ball. "
     "How much does the ball cost in cents?", 5),
]


def exercise_5():
    """Evaluate direct, CoT, and self-consistency on BENCHMARK."""
    # TODO: For each problem, get answers via:
    #   1. direct_answer() -> extract_number()
    #   2. zero_shot_cot() -> extract_number()
    #   3. self_consistency() -> majority_answer
    # TODO: Print a results table and count correct answers per method
    methods = {"direct": 0, "cot": 0, "self_consistency": 0}
    for question, expected in BENCHMARK:
        # TODO: Run each method, compare to expected, increment if correct
        pass
    print("[Ex5] Results:")
    for method, correct in methods.items():
        print(f"  {method:>18}: {correct}/{len(BENCHMARK)} correct")


if __name__ == "__main__":
    print("=== Exercise 1: Manual Chain-of-Thought ===")
    exercise_1()

    print("\n=== Exercise 2: Zero-Shot CoT ===")
    exercise_2()

    print("\n=== Exercise 3: Self-Consistency ===")
    exercise_3()

    print("\n=== Exercise 4: CoT for Logic ===")
    exercise_4()

    print("\n=== Exercise 5: CoT Evaluation Harness ===")
    exercise_5()

    print("\nAll exercises completed!")
