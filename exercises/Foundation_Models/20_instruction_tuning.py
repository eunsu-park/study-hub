"""
Exercises for Lesson 20: Instruction Tuning
Topic: Foundation_Models

Solutions to practice problems from the lesson.
"""

import json


# === Exercise 1: Instruction Format Design ===
# Problem: Convert a raw QA pair into Alpaca, ShareGPT, and JSON formats.

def exercise_1():
    """Solution: Instruction format design"""
    # Alpaca format
    alpaca = {
        "instruction": "Write a Python one-liner to check if a list is sorted in ascending order.",
        "input": "",
        "output": (
            "You can check if a list is sorted using: "
            "`all(lst[i] <= lst[i+1] for i in range(len(lst)-1))` "
            "or more concisely with `lst == sorted(lst)`. The first "
            "approach is O(n) and stops early if unsorted; the second "
            "creates a new sorted copy (O(n log n))."
        ),
    }

    # ShareGPT format
    sharegpt = {
        "conversations": [
            {
                "from": "human",
                "value": "Hey, I'm trying to check if my Python list is sorted. "
                         "Is there a quick way to do this?"
            },
            {
                "from": "gpt",
                "value": (
                    "Yes! There are a few ways:\n\n"
                    "1. **One-liner** (O(n), early termination):\n"
                    "```python\n"
                    "is_sorted = all(lst[i] <= lst[i+1] for i in range(len(lst)-1))\n"
                    "```\n\n"
                    "2. **Simple comparison** (O(n log n)):\n"
                    "```python\nis_sorted = lst == sorted(lst)\n```"
                ),
            },
            {"from": "human", "value": "What's the difference between them?"},
            {
                "from": "gpt",
                "value": (
                    "The key difference is efficiency: the first stops as soon "
                    "as it finds an out-of-order pair (best case O(1)), while "
                    "the second always sorts the entire list."
                ),
            },
        ],
    }

    # JSON structured output format
    json_structured = {
        "instruction": "How do I check if a Python list is sorted? Provide a structured response.",
        "input": "list = [1, 2, 3, 4, 5]",
        "output": json.dumps({
            "solution": "all(lst[i] <= lst[i+1] for i in range(len(lst)-1))",
            "complexity": "O(n)",
            "alternative": "lst == sorted(lst)",
            "explanation": "Iterates through pairs, returns False on first out-of-order element",
        }),
    }

    print("  1. Alpaca format (single-turn):")
    print(f"     {json.dumps(alpaca, indent=6)[:200]}...")
    print("     Best for: Single-turn QA, instruction-following tasks.")
    print()

    print("  2. ShareGPT format (multi-turn):")
    print(f"     {len(sharegpt['conversations'])} turns")
    print("     Best for: Conversational models, follow-up questions.")
    print()

    print("  3. JSON structured output:")
    print(f"     Output schema: solution, complexity, alternative, explanation")
    print("     Best for: Models that need to output structured data, tool use.")


# === Exercise 2: Dataset Diversity vs Quality ===
# Problem: Compare FLAN (1,836 tasks) vs Alpaca (52K self-instruct).

def exercise_2():
    """Solution: Dataset diversity vs quality"""
    print("  FLAN strengths:")
    print("    - 1,836 diverse tasks (classification, summarization, translation)")
    print("    - Human-validated from academic NLP benchmarks")
    print("    - Tests zero-shot generalization through format variations")
    print("    Weakness: Academic/benchmark style, not conversational")
    print()

    print("  Alpaca strengths:")
    print("    - 52K natural instruction style examples")
    print("    - Diverse types from 175 seed examples via Self-Instruct")
    print("    - Matches real-world usage patterns")
    print("    Weakness: GPT-3/4 generated, inherited biases, variable quality")
    print()

    print("  Evaluation to determine which weakness matters more:")
    print("    Test set 1 (FLAN weakness): 100 informal conversational queries")
    print("    Test set 2 (Alpaca weakness): 100 edge cases with constraints")
    print("    Test set 3 (both): Multi-turn with mid-conversation task switches")
    print("    The bigger gap reveals which approach is more appropriate.")


# === Exercise 3: Loss Masking Implementation ===
# Problem: Implement masking so only response tokens contribute to loss.

def create_labels_with_masking(input_ids: list, instruction_length: int) -> list:
    """
    Create labels for instruction tuning where instruction tokens
    are masked (set to -100) and only response tokens contribute to loss.
    """
    IGNORE_INDEX = -100
    labels = list(input_ids)  # Copy to avoid modifying original

    # Mask all instruction tokens
    for i in range(instruction_length):
        labels[i] = IGNORE_INDEX

    return labels


def exercise_3():
    """Solution: Loss masking implementation"""
    # Example
    input_ids = [101, 234, 567, 789, 12, 345, 678]
    instruction_length = 3

    labels = create_labels_with_masking(input_ids, instruction_length)

    print("  Loss masking implementation:")
    print(f"    input_ids:          {input_ids}")
    print(f"    instruction_length: {instruction_length}")
    print(f"    labels:             {labels}")
    print()

    # Verify
    expected = [-100, -100, -100, 789, 12, 345, 678]
    assert labels == expected, f"Expected {expected}, got {labels}"
    print("    Verification: PASSED")
    print()

    # Extended example: multi-turn conversation masking
    conversation = [
        {"role": "system", "value": "You are a helpful assistant.", "tokens": [1, 2, 3]},
        {"role": "user", "value": "What is Python?", "tokens": [4, 5, 6]},
        {"role": "assistant", "value": "Python is a programming language.", "tokens": [7, 8, 9, 10]},
        {"role": "user", "value": "Tell me more.", "tokens": [11, 12]},
        {"role": "assistant", "value": "It supports OOP and FP.", "tokens": [13, 14, 15]},
    ]

    all_tokens = []
    all_labels = []
    IGNORE_INDEX = -100

    for turn in conversation:
        tokens = turn["tokens"]
        all_tokens.extend(tokens)
        if turn["role"] == "assistant":
            all_labels.extend(tokens)  # Only assistant turns in loss
        else:
            all_labels.extend([IGNORE_INDEX] * len(tokens))

    print("  Multi-turn conversation masking:")
    print(f"    All tokens: {all_tokens}")
    print(f"    Labels:     {all_labels}")
    print()
    print("  Why masking matters:")
    print("    Without: model trained to predict instructions AND responses")
    print("      -> may learn to repeat instructions verbatim")
    print("    With: model only learns to predict correct responses")
    print("      -> instructions are inputs to attend to, not outputs to generate")


# === Exercise 4: Evol-Instruct Quality Analysis ===
# Problem: Trace 3 levels of evolution and identify where complexity hurts.

def exercise_4():
    """Solution: Evol-Instruct quality analysis"""
    levels = [
        {
            "level": "0 (Seed)",
            "instruction": "Write a function to sort a list",
            "quality": "Simple, clear, fundamental Python task.",
            "training_value": "High",
        },
        {
            "level": "1 (Add constraints)",
            "instruction": (
                "Write a function to sort a list of dictionaries by a "
                "specific key, handling missing keys gracefully"
            ),
            "quality": "Good complexity increase: realistic, tests error handling.",
            "training_value": "High",
        },
        {
            "level": "2 (Multiple requirements)",
            "instruction": (
                "Sort list of dicts by multiple keys in specified sort orders, "
                "handle missing keys, guarantee O(n log n), include unit test"
            ),
            "quality": "Moderate: realistic but may be too long for context.",
            "training_value": "Medium",
        },
        {
            "level": "3 (Potentially harmful)",
            "instruction": (
                "Implement generic sort: any iterable, custom comparison with "
                "memoization, QuickSort AND TimSort switching, threading locks, "
                "stability proof, 10 unit tests"
            ),
            "quality": (
                "PROBLEM: unreasonably complex. Contains contradictions "
                "(QuickSort is not stable). Teaches over-engineering."
            ),
            "training_value": "Negative -- may degrade model quality",
        },
    ]

    for l in levels:
        print(f"  Level {l['level']}:")
        print(f"    Instruction: {l['instruction']}")
        print(f"    Quality: {l['quality']}")
        print(f"    Training value: {l['training_value']}")
        print()

    print("  Conclusion: Evol-Instruct is valuable for levels 1-2,")
    print("  but auto-generated level 3+ examples often become incoherent")
    print("  or require expert validation before training.")


if __name__ == "__main__":
    print("=== Exercise 1: Instruction Format Design ===")
    exercise_1()
    print("\n=== Exercise 2: Dataset Diversity vs Quality ===")
    exercise_2()
    print("\n=== Exercise 3: Loss Masking Implementation ===")
    exercise_3()
    print("\n=== Exercise 4: Evol-Instruct Quality Analysis ===")
    exercise_4()
    print("\nAll exercises completed!")
