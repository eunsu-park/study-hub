"""
Exercises for Lesson 01: Introduction to Claude
Topic: Claude_Ecosystem

Solutions to practice problems from the lesson.
"""

from dataclasses import dataclass
from typing import Optional


# === Exercise 1: Model Family Data Structure ===
# Problem: Define a data structure for Claude model specifications
#   and write a function to select the best model for a given task.

@dataclass
class ClaudeModel:
    """Specification for a Claude model variant."""
    name: str
    tier: str               # "opus", "sonnet", "haiku"
    context_window: int      # tokens
    max_output: int          # tokens
    input_cost_per_mtok: float   # USD per million input tokens
    output_cost_per_mtok: float  # USD per million output tokens
    speed: str               # "slow", "fast", "fastest"
    intelligence: str        # "highest", "high", "good"
    extended_thinking: bool


CLAUDE_MODELS = [
    ClaudeModel("Claude Opus 4", "opus", 200_000, 32_000,
                15.0, 75.0, "slow", "highest", True),
    ClaudeModel("Claude Sonnet 4", "sonnet", 200_000, 16_000,
                3.0, 15.0, "fast", "high", True),
    ClaudeModel("Claude Haiku", "haiku", 200_000, 8_000,
                0.25, 1.25, "fastest", "good", False),
]


def select_model(
    priority: str,
    needs_extended_thinking: bool = False,
    min_output_tokens: int = 0,
) -> ClaudeModel:
    """Select the best Claude model based on requirements.

    Args:
        priority: "quality", "speed", or "cost"
        needs_extended_thinking: whether extended thinking is required
        min_output_tokens: minimum output token capacity needed
    """
    candidates = [
        m for m in CLAUDE_MODELS
        if (not needs_extended_thinking or m.extended_thinking)
        and m.max_output >= min_output_tokens
    ]
    if not candidates:
        raise ValueError("No model meets the requirements")

    if priority == "quality":
        order = {"highest": 0, "high": 1, "good": 2}
        return min(candidates, key=lambda m: order[m.intelligence])
    elif priority == "speed":
        order = {"fastest": 0, "fast": 1, "slow": 2}
        return min(candidates, key=lambda m: order[m.speed])
    elif priority == "cost":
        return min(candidates, key=lambda m: m.input_cost_per_mtok)
    else:
        raise ValueError(f"Unknown priority: {priority}")


def exercise_1():
    """Demonstrate model selection logic."""
    print(f"Quality priority: {select_model('quality').name}")
    print(f"Speed priority:   {select_model('speed').name}")
    print(f"Cost priority:    {select_model('cost').name}")
    print(f"Extended thinking + speed: "
          f"{select_model('speed', needs_extended_thinking=True).name}")
    print(f"Need 32K output: "
          f"{select_model('cost', min_output_tokens=32_000).name}")


# === Exercise 2: Token Estimation ===
# Problem: Implement a simple token estimator that approximates
#   token count from text (roughly 1 token per 4 characters for English).

def estimate_tokens(text: str, chars_per_token: float = 4.0) -> int:
    """Estimate the number of tokens in a text string.

    This is a rough approximation. Real tokenizers (e.g., tiktoken)
    are more accurate but require external dependencies.
    """
    return max(1, int(len(text) / chars_per_token))


def estimate_cost(
    input_text: str,
    output_tokens: int,
    model: ClaudeModel,
) -> dict[str, float]:
    """Estimate the cost of an API call."""
    input_tokens = estimate_tokens(input_text)
    input_cost = (input_tokens / 1_000_000) * model.input_cost_per_mtok
    output_cost = (output_tokens / 1_000_000) * model.output_cost_per_mtok
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "input_cost_usd": round(input_cost, 6),
        "output_cost_usd": round(output_cost, 6),
        "total_cost_usd": round(input_cost + output_cost, 6),
    }


def exercise_2():
    """Demonstrate token estimation and cost calculation."""
    sample = "Explain the theory of relativity in simple terms."
    for model in CLAUDE_MODELS:
        cost = estimate_cost(sample, output_tokens=500, model=model)
        print(f"{model.name}: ~{cost['input_tokens']} input tokens, "
              f"${cost['total_cost_usd']:.6f}")


# === Exercise 3: Context Window Budget Planner ===
# Problem: Given a context window size, allocate budget for
#   system prompt, conversation history, and output.

@dataclass
class ContextBudget:
    """Token budget allocation for a Claude conversation."""
    system_prompt: int
    conversation_history: int
    reserved_output: int
    available_for_input: int
    total: int
    utilization_pct: float


def plan_context_budget(
    model: ClaudeModel,
    system_prompt_tokens: int = 2_000,
    output_reserve_pct: float = 0.15,
    history_tokens: int = 0,
) -> ContextBudget:
    """Plan token budget allocation within a model's context window."""
    reserved_output = min(
        model.max_output,
        int(model.context_window * output_reserve_pct),
    )
    used = system_prompt_tokens + history_tokens
    available = model.context_window - used - reserved_output
    utilization = (used / model.context_window) * 100

    return ContextBudget(
        system_prompt=system_prompt_tokens,
        conversation_history=history_tokens,
        reserved_output=reserved_output,
        available_for_input=max(0, available),
        total=model.context_window,
        utilization_pct=round(utilization, 1),
    )


def exercise_3():
    """Demonstrate context window budget planning."""
    model = CLAUDE_MODELS[1]  # Sonnet
    budget = plan_context_budget(model, system_prompt_tokens=3_000,
                                 history_tokens=10_000)
    print(f"Model: {model.name} ({model.context_window:,} tokens)")
    print(f"  System prompt:  {budget.system_prompt:>8,} tokens")
    print(f"  History:        {budget.conversation_history:>8,} tokens")
    print(f"  Output reserve: {budget.reserved_output:>8,} tokens")
    print(f"  Available:      {budget.available_for_input:>8,} tokens")
    print(f"  Utilization:    {budget.utilization_pct}%")


# === Exercise 4: Product Ecosystem Router ===
# Problem: Given a user's task description, recommend the appropriate
#   Claude product (Claude.ai, Claude Code, API, Desktop).

PRODUCT_KEYWORDS: dict[str, list[str]] = {
    "Claude Code": ["code", "debug", "refactor", "test", "commit", "terminal",
                     "cli", "edit file", "git", "build"],
    "Claude API": ["api", "integrate", "automate", "batch", "programmatic",
                    "sdk", "production", "pipeline"],
    "Claude.ai": ["chat", "write", "summarize", "brainstorm", "explain",
                   "analyze", "conversation", "essay"],
    "Claude Desktop": ["desktop", "mcp", "local files", "artifacts",
                        "preview", "app"],
}


def recommend_product(task: str) -> list[tuple[str, int]]:
    """Score and rank Claude products by relevance to a task description."""
    task_lower = task.lower()
    scores: dict[str, int] = {}
    for product, keywords in PRODUCT_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in task_lower)
        if score > 0:
            scores[product] = score
    if not scores:
        scores["Claude.ai"] = 1  # default recommendation
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def exercise_4():
    """Demonstrate product recommendation."""
    tasks = [
        "Debug a failing test in my Python project",
        "Write a blog post about machine learning",
        "Build an automated customer support pipeline",
        "Explore local CSV files and create a chart",
    ]
    for task in tasks:
        recs = recommend_product(task)
        best = recs[0][0]
        print(f"Task: {task}")
        print(f"  → Recommended: {best}\n")


if __name__ == "__main__":
    print("=== Exercise 1: Model Selection ===")
    exercise_1()

    print("\n=== Exercise 2: Token Estimation ===")
    exercise_2()

    print("\n=== Exercise 3: Context Budget ===")
    exercise_3()

    print("\n=== Exercise 4: Product Recommendation ===")
    exercise_4()

    print("\nAll exercises completed!")
