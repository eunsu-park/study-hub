# Exercise: Lesson 02 — Zero-Shot and Few-Shot Prompting
# Complete the TODO items below.
#
# Run: python 02_zero_shot_and_few_shot.py

import anthropic

client = anthropic.Anthropic()  # expects ANTHROPIC_API_KEY env var

MODEL = "claude-sonnet-4-20250514"


# === Exercise 1: Zero-Shot Classifier ===
# Classify text sentiment without any examples.
# Hint: Be explicit about the allowed labels in your prompt.

SENTIMENTS = ["positive", "negative", "neutral"]

def zero_shot_classify(text: str) -> str:
    """Classify the sentiment of `text` using a zero-shot prompt.
    Must return one of: "positive", "negative", "neutral".
    """
    # TODO: Build a prompt that instructs Claude to classify sentiment
    # TODO: Call the API and parse the response to one of the three labels
    # Hint: Ask Claude to reply with exactly one word.
    pass


def exercise_1():
    cases = [
        ("I absolutely love this product!", "positive"),
        ("This is the worst experience ever.", "negative"),
        ("The meeting is scheduled for 3 PM.", "neutral"),
    ]
    for text, expected in cases:
        result = zero_shot_classify(text)
        assert result in SENTIMENTS, f"Invalid label: {result}"
        status = "PASS" if result == expected else "MISS"
        print(f"[Ex1] {status} | expected={expected}, got={result} | {text[:40]}")


# === Exercise 2: Few-Shot Classifier ===
# Provide examples to improve classification accuracy.
# Hint: Format examples clearly with a consistent pattern.

EXAMPLES = [
    ("The sunset was breathtaking and peaceful.", "positive"),
    ("I'm frustrated with the constant delays.", "negative"),
    ("Water boils at 100 degrees Celsius.", "neutral"),
    ("What a delightful surprise this turned out to be!", "positive"),
    ("The service was rude and unprofessional.", "negative"),
]

def few_shot_classify(text: str, examples: list[tuple[str, str]]) -> str:
    """Classify sentiment using few-shot examples in the prompt.
    Must return one of: "positive", "negative", "neutral".
    """
    # TODO: Format the examples into a prompt (e.g., "Text: ... -> Label: ...")
    # TODO: Append the new text and ask Claude to classify it
    # TODO: Parse and return the label
    pass


def exercise_2():
    test_cases = [
        ("This restaurant has the best pasta in town!", "positive"),
        ("I regret buying this, total waste of money.", "negative"),
        ("The capital of France is Paris.", "neutral"),
    ]
    for text, expected in test_cases:
        result = few_shot_classify(text, EXAMPLES)
        assert result in SENTIMENTS, f"Invalid label: {result}"
        status = "PASS" if result == expected else "MISS"
        print(f"[Ex2] {status} | expected={expected}, got={result} | {text[:40]}")


# === Exercise 3: Dynamic Example Selection ===
# Choose the most relevant examples for a given input.
# Hint: Use simple keyword overlap to measure similarity.

EXAMPLE_POOL = [
    ("The hotel room was spotless and cozy.", "positive"),
    ("Terrible customer support, waited 2 hours.", "negative"),
    ("Python 3.12 was released in October 2023.", "neutral"),
    ("Amazing concert, the band was incredible!", "positive"),
    ("The food arrived cold and tasted stale.", "negative"),
    ("Trains depart every 30 minutes.", "neutral"),
    ("Best purchase I've made all year.", "positive"),
    ("Completely disappointed with the quality.", "negative"),
]

def compute_similarity(text_a: str, text_b: str) -> float:
    """Compute a simple similarity score between two texts.
    Return a float between 0.0 and 1.0.
    """
    # TODO: Tokenize both texts into lowercase word sets
    # TODO: Return Jaccard similarity = |intersection| / |union|
    pass


def select_examples(text: str, pool: list[tuple[str, str]],
                     k: int = 3) -> list[tuple[str, str]]:
    """Select the top-k most similar examples from the pool."""
    # TODO: Score each example using compute_similarity
    # TODO: Return top-k examples sorted by descending similarity
    pass


def exercise_3():
    query = "The delivery was late and the package was damaged."
    selected = select_examples(query, EXAMPLE_POOL, k=3)
    assert len(selected) == 3, f"Expected 3 examples, got {len(selected)}"
    print(f"[Ex3] Query: {query}")
    for ex_text, ex_label in selected:
        sim = compute_similarity(query.lower(), ex_text.lower())
        print(f"  sim={sim:.3f} | {ex_label:>8} | {ex_text[:50]}")


# === Exercise 4: Few-Shot with Dynamic Selection (End-to-End) ===
# Combine exercises 2 and 3: select examples, then classify.

def dynamic_few_shot_classify(text: str) -> str:
    """Select top-3 examples dynamically, then classify with few-shot."""
    # TODO: Use select_examples() to pick relevant examples
    # TODO: Pass them to few_shot_classify()
    pass


def exercise_4():
    test_inputs = [
        "The staff went above and beyond to help us.",
        "Broken on arrival, what a scam.",
        "The library opens at 9 AM on weekdays.",
    ]
    for text in test_inputs:
        label = dynamic_few_shot_classify(text)
        assert label in SENTIMENTS
        print(f"[Ex4] {label:>8} | {text}")


# === Exercise 5: Zero-Shot vs Few-Shot Comparison ===
# Run both classifiers on the same inputs and compare accuracy.

EVAL_SET = [
    ("Absolutely wonderful, exceeded expectations!", "positive"),
    ("I'm never coming back to this place.", "negative"),
    ("The meeting will be held in room 204.", "neutral"),
    ("So grateful for the quick response!", "positive"),
    ("Unacceptable quality for the price.", "negative"),
]

def exercise_5():
    """Compare zero-shot and few-shot accuracy on the eval set."""
    # TODO: Run zero_shot_classify on each item in EVAL_SET
    # TODO: Run dynamic_few_shot_classify on each item in EVAL_SET
    # TODO: Compute and print accuracy for both methods
    zero_correct = 0
    few_correct = 0
    for text, expected in EVAL_SET:
        # TODO: Classify with both methods and count correct predictions
        pass
    total = len(EVAL_SET)
    print(f"[Ex5] Zero-shot accuracy: {zero_correct}/{total}")
    print(f"[Ex5] Few-shot accuracy:  {few_correct}/{total}")


if __name__ == "__main__":
    print("=== Exercise 1: Zero-Shot Classifier ===")
    exercise_1()

    print("\n=== Exercise 2: Few-Shot Classifier ===")
    exercise_2()

    print("\n=== Exercise 3: Dynamic Example Selection ===")
    exercise_3()

    print("\n=== Exercise 4: Dynamic Few-Shot (End-to-End) ===")
    exercise_4()

    print("\n=== Exercise 5: Zero-Shot vs Few-Shot Comparison ===")
    exercise_5()

    print("\nAll exercises completed!")
