"""
Exercises for Lesson 03: Emergent Abilities
Topic: Foundation_Models

Solutions to practice problems from the lesson.
"""

import re
from collections import Counter


# === Exercise 1: Metric Choice and the Emergence Mirage ===
# Problem: Explain why accuracy produces phase transitions while
# token-level accuracy produces smooth curves.

def exercise_1():
    """Solution: Metric choice and emergence mirage"""
    print("  Part 1: Accuracy creates phase transitions")
    print("    Accuracy (exact match) is a threshold metric: all-or-nothing.")
    print("    Even going from 2/5 to 4/5 digits correct stays at accuracy=0.")
    print("    Only when ALL digits are correct does accuracy jump to 1.")
    print("    This masks gradual improvement and looks like 'sudden emergence.'")
    print()

    # Demonstration with a simple model of improving digit accuracy
    import random
    random.seed(42)

    scales = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    num_trials = 100
    num_digits = 5

    print("  Simulation: 5-digit addition at increasing scale")
    print(f"  {'Scale':<8} {'Exact Match':<15} {'Per-Digit Accuracy':<20}")
    print("  " + "-" * 43)

    for scale in scales:
        # Per-digit accuracy increases linearly with scale
        per_digit_prob = min(0.1 * scale, 1.0)

        exact_matches = 0
        total_correct_digits = 0

        for _ in range(num_trials):
            correct_digits = sum(
                1 for _ in range(num_digits)
                if random.random() < per_digit_prob
            )
            total_correct_digits += correct_digits
            if correct_digits == num_digits:
                exact_matches += 1

        exact_acc = exact_matches / num_trials
        digit_acc = total_correct_digits / (num_trials * num_digits)
        print(f"  {scale:<8} {exact_acc:<15.2f} {digit_acc:<20.2f}")

    print()
    print("  Part 2: Token-level accuracy rises smoothly (see column above)")
    print("  Part 3: For research on emergence -> use continuous metrics")
    print("          For deployment decisions -> use task-level accuracy")


# === Exercise 2: Implementing Self-Consistency ===
# Problem: Implement self_consistency function that generates multiple
# CoT responses and returns majority vote answer.

def extract_numeric_answer(response: str) -> str:
    """Extract the last number mentioned in the response."""
    numbers = re.findall(r'\b\d+(?:\.\d+)?\b', response)
    return numbers[-1] if numbers else "unknown"


class MockModel:
    """Mock model that simulates CoT responses with some noise."""
    def __init__(self, correct_answer: int, accuracy: float = 0.7):
        self.correct_answer = correct_answer
        self.accuracy = accuracy
        self._call_count = 0

    def generate(self, prompt: str, temperature: float = 0.7) -> str:
        import random
        self._call_count += 1
        # Simulate: with probability 'accuracy', return correct answer
        if random.random() < self.accuracy:
            return (
                f"Let's think step by step. "
                f"First, we compute the intermediate value. "
                f"Then we get the final answer: {self.correct_answer}"
            )
        else:
            # Return a wrong answer
            wrong = self.correct_answer + random.choice([-2, -1, 1, 2, 3])
            return (
                f"Let's think step by step. "
                f"I think the answer is {wrong}"
            )


def self_consistency(
    question: str,
    model,
    n_samples: int = 5,
    temperature: float = 0.7,
) -> str:
    """
    Self-consistency decoding:
    Generate multiple CoT paths and take majority vote.
    """
    # Step 1: Build CoT prompt
    cot_prompt = f"Q: {question}\nA: Let's think step by step."

    # Step 2: Generate multiple responses
    answers = []
    for _ in range(n_samples):
        response = model.generate(cot_prompt, temperature=temperature)
        answer = extract_numeric_answer(response)
        answers.append(answer)

    # Step 3: Majority vote
    counter = Counter(answers)
    majority_answer = counter.most_common(1)[0][0]

    return majority_answer


def exercise_2():
    """Solution: Self-consistency implementation"""
    import random
    random.seed(42)

    model = MockModel(correct_answer=42, accuracy=0.7)

    question = "If a store has 6 boxes with 7 items each, how many items total?"
    n_samples = 11

    answer = self_consistency(question, model, n_samples=n_samples)

    print(f"  Question: {question}")
    print(f"  Number of samples: {n_samples}")
    print(f"  Majority vote answer: {answer}")
    print(f"  Expected correct answer: 42")
    print()
    print("  Why self-consistency works:")
    print("  - Incorrect paths produce diverse wrong answers")
    print("  - Correct paths converge on the same right answer")
    print("  - Majority vote filters out idiosyncratic errors")


# === Exercise 3: Capability Elicitation Experiment Design ===
# Problem: Design an experiment to test if role assignment improves
# accuracy on logic puzzles.

def exercise_3():
    """Solution: Experiment design for role assignment"""
    experiment = {
        "independent_variable": (
            "The role/persona assigned to the model in the system prompt"
        ),
        "dependent_variable": (
            "Accuracy (% correct) on a standardized set of 50 logic puzzles"
        ),
        "conditions": {
            "Control": (
                'No persona: "Solve the following logic puzzle: {puzzle}"'
            ),
            "Treatment A (Expert)": (
                '"You are a world-class logician with 30 years of experience. '
                'Solve the following logic puzzle: {puzzle}"'
            ),
            "Treatment B (Teacher)": (
                '"You are a patient teacher explaining logical reasoning. '
                'Solve the following logic puzzle step by step: {puzzle}"'
            ),
        },
        "confounds": [
            (
                "Temperature/sampling: Use temperature=0 for all conditions "
                "or average over multiple seeds"
            ),
            (
                "Problem ordering: Randomize puzzle order across conditions "
                "to prevent position bias"
            ),
            (
                "CoT vs no-CoT: 'Teacher' prompt implicitly encourages "
                "step-by-step reasoning. Ensure all conditions either include "
                "or exclude 'step by step' uniformly."
            ),
        ],
    }

    print(f"  Independent variable: {experiment['independent_variable']}")
    print(f"  Dependent variable: {experiment['dependent_variable']}")
    print()
    print("  Conditions:")
    for name, desc in experiment["conditions"].items():
        print(f"    {name}: {desc}")
    print()
    print("  Potential confounds to control:")
    for i, c in enumerate(experiment["confounds"], 1):
        print(f"    {i}. {c}")


# === Exercise 4: Tree of Thoughts vs Chain-of-Thought ===
# Problem: Compare CoT and ToT on 4 dimensions.

def exercise_4():
    """Solution: CoT vs ToT comparison"""
    comparison = {
        "Search strategy": {
            "CoT": (
                "Single linear path -- greedy left-to-right generation. "
                "Fast but commits early to potentially wrong directions."
            ),
            "ToT": (
                "Tree-structured BFS/DFS -- explores multiple branches "
                "and backtracks. More thorough but costly."
            ),
        },
        "Computational cost": {
            "CoT": "Low -- 1x cost, proportional to single response length.",
            "ToT": (
                "High -- 10-100x more expensive. "
                "Proportional to branching factor x depth x evaluation cost."
            ),
        },
        "Suitable problem types": {
            "CoT": (
                "Sequential reasoning where each step follows naturally "
                "(arithmetic, straightforward QA)."
            ),
            "ToT": (
                "Combinatorial/planning problems where early decisions "
                "strongly constrain later options (puzzles, code debugging)."
            ),
        },
        "Failure mode": {
            "CoT": (
                "Cascading error -- one wrong early step propagates "
                "through the entire chain."
            ),
            "ToT": (
                "Evaluation bottleneck -- quality depends on accuracy "
                "of the node evaluation function."
            ),
        },
    }

    for dim, vals in comparison.items():
        print(f"  {dim}:")
        print(f"    CoT: {vals['CoT']}")
        print(f"    ToT: {vals['ToT']}")
        print()


if __name__ == "__main__":
    print("=== Exercise 1: Metric Choice and the Emergence Mirage ===")
    exercise_1()
    print("\n=== Exercise 2: Implementing Self-Consistency ===")
    exercise_2()
    print("\n=== Exercise 3: Capability Elicitation Experiment Design ===")
    exercise_3()
    print("\n=== Exercise 4: Tree of Thoughts vs Chain-of-Thought ===")
    exercise_4()
    print("\nAll exercises completed!")
