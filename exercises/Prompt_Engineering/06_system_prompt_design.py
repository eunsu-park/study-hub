# Exercise: Lesson 06 — System Prompt Design
# Complete the TODO items below.
#
# Run: python 06_system_prompt_design.py

import anthropic
import json

client = anthropic.Anthropic()  # expects ANTHROPIC_API_KEY env var

MODEL = "claude-sonnet-4-20250514"


# === Exercise 1: Role-Based System Prompt ===
# Design a system prompt that gives Claude a specific persona and expertise.
# Hint: Include role, expertise area, communication style, and audience.

def build_expert_prompt(role: str, expertise: str, audience: str,
                        style: str) -> str:
    """Build a system prompt that defines an expert persona.
    The prompt should include the role, expertise, target audience,
    and communication style.
    """
    # TODO: Compose a system prompt string incorporating all 4 parameters
    # Example: "You are a {role} specializing in {expertise}. ..."
    pass


def exercise_1():
    prompt = build_expert_prompt(
        role="senior database architect",
        expertise="PostgreSQL performance tuning",
        audience="junior developers",
        style="patient and example-driven",
    )
    assert isinstance(prompt, str) and len(prompt) > 50
    # Test that the persona actually influences the response
    response = client.messages.create(
        model=MODEL, max_tokens=200,
        system=prompt,
        messages=[{"role": "user", "content": "How do I speed up a slow query?"}],
    )
    text = response.content[0].text
    print(f"[Ex1] System prompt length: {len(prompt)} chars")
    print(f"[Ex1] Response preview: {text[:120]}...")


# === Exercise 2: Add Behavioral Constraints ===
# Add rules the model must follow (do's and don'ts).
# Hint: Use numbered rules and explicit prohibitions.

def build_constrained_prompt(base_role: str,
                             must_do: list[str],
                             must_not: list[str]) -> str:
    """Build a system prompt with explicit behavioral constraints.
    Args:
        base_role: the persona description
        must_do: list of required behaviors (e.g., "Always cite sources")
        must_not: list of prohibited behaviors (e.g., "Never give medical advice")
    """
    # TODO: Format the role, must_do rules, and must_not rules
    #       into a clear, structured system prompt
    # Hint: Use sections like "## Rules" and "## Restrictions"
    pass


def exercise_2():
    prompt = build_constrained_prompt(
        base_role="You are a helpful coding tutor.",
        must_do=[
            "Always explain WHY, not just HOW",
            "Include at least one code example in every answer",
            "End each response with a practice question",
        ],
        must_not=[
            "Never give complete homework solutions",
            "Never use jargon without defining it first",
        ],
    )
    assert "WHY" in prompt or "why" in prompt
    assert "Never" in prompt or "never" in prompt
    print(f"[Ex2] Constrained prompt ({len(prompt)} chars):")
    for line in prompt.split("\n"):
        if line.strip():
            print(f"  {line.strip()[:80]}")


# === Exercise 3: Output Guardrails ===
# Design a system prompt that enforces output format and safety.
# Hint: Test that the guardrails actually work by probing edge cases.

def build_guardrailed_prompt() -> str:
    """Build a system prompt for a customer service bot with guardrails:
    1. Must always respond in valid JSON with keys: response, confidence, escalate
    2. Must refuse to discuss competitor products
    3. Must escalate (escalate=true) if the user sounds angry or threatens
    4. Confidence must be "high", "medium", or "low"
    """
    # TODO: Write the system prompt with all 4 guardrails above
    pass


def test_guardrail(system_prompt: str, user_msg: str) -> dict:
    """Send a message and parse the JSON response."""
    # TODO: Call the API with the system prompt and user message
    # TODO: Parse the JSON response; return {} on parse failure
    pass


def exercise_3():
    system_prompt = build_guardrailed_prompt()
    test_cases = [
        ("What are your return policies?", False),
        ("How does your product compare to CompetitorX?", False),
        ("This is UNACCEPTABLE! I want a refund NOW!", True),
    ]
    for msg, expect_escalate in test_cases:
        result = test_guardrail(system_prompt, msg)
        assert "response" in result, f"Missing 'response' key for: {msg[:30]}"
        escalated = result.get("escalate", False)
        status = "PASS" if escalated == expect_escalate else "MISS"
        print(f"[Ex3] {status} | escalate={escalated} | {msg[:50]}")


# === Exercise 4: Multi-Capability System Prompt ===
# Design a prompt that routes between different capabilities.
# Hint: Define capabilities as named "modes" in the system prompt.

CAPABILITIES = {
    "translate": "Translate text between languages",
    "summarize": "Summarize text concisely",
    "analyze": "Analyze sentiment and key themes",
}

def build_multi_capability_prompt(capabilities: dict[str, str]) -> str:
    """Build a system prompt that supports multiple capabilities.
    The user will prefix their message with a mode tag like [translate].
    The bot should route to the appropriate capability.
    If no tag is given, default to general assistance.
    """
    # TODO: List each capability with its tag and description
    # TODO: Include instructions for routing based on the tag prefix
    # TODO: Include a fallback for unrecognized or missing tags
    pass


def exercise_4():
    system_prompt = build_multi_capability_prompt(CAPABILITIES)
    test_messages = [
        "[translate] Hola, como estas?",
        "[summarize] The quick brown fox jumps over the lazy dog. "
        "This sentence is famous for containing every letter of the alphabet.",
        "[analyze] I'm extremely happy with the service but the price was too high.",
        "What's the weather like today?",  # no tag -- should use default
    ]
    for msg in test_messages:
        response = client.messages.create(
            model=MODEL, max_tokens=200,
            system=system_prompt,
            messages=[{"role": "user", "content": msg}],
        )
        text = response.content[0].text
        print(f"[Ex4] Input: {msg[:50]}")
        print(f"       Output: {text[:80]}...\n")


# === Exercise 5: System Prompt Evaluator ===
# Score a system prompt on key quality criteria.
# This is a pure Python exercise (no API call needed).

QUALITY_CRITERIA = [
    ("has_role", "Defines a clear role or persona"),
    ("has_constraints", "Includes behavioral constraints or rules"),
    ("has_format", "Specifies output format"),
    ("has_examples", "Contains examples or demonstrations"),
    ("has_fallback", "Handles edge cases or unknown inputs"),
]

def evaluate_system_prompt(prompt: str) -> dict:
    """Score a system prompt on quality criteria.
    Return {"scores": dict[str, bool], "total": int, "max": int, "grade": str}.
    """
    # TODO: Check each criterion using keyword heuristics:
    #   has_role: contains words like "you are", "role", "act as"
    #   has_constraints: contains "must", "always", "never", "do not"
    #   has_format: contains "format", "JSON", "respond with", "output"
    #   has_examples: contains "example", "e.g.", "for instance", "like this"
    #   has_fallback: contains "otherwise", "if not", "default", "unknown"
    # TODO: Assign a grade: 5=A, 4=B, 3=C, 2=D, 0-1=F
    pass


def exercise_5():
    good_prompt = (
        "You are an expert financial advisor. You must always disclose risks. "
        "Never recommend specific stocks. Respond in JSON format with keys: "
        "advice, risk_level, disclaimer. For example: "
        '{\"advice\": \"...\", \"risk_level\": \"medium\", \"disclaimer\": \"...\"}. '
        "If the question is outside finance, default to a polite redirect."
    )
    result = evaluate_system_prompt(good_prompt)
    assert "scores" in result and "grade" in result
    print(f"[Ex5] Scores: {result['scores']}")
    print(f"[Ex5] Total: {result['total']}/{result['max']} -> Grade: {result['grade']}")


if __name__ == "__main__":
    print("=== Exercise 1: Role-Based System Prompt ===")
    exercise_1()

    print("\n=== Exercise 2: Behavioral Constraints ===")
    exercise_2()

    print("\n=== Exercise 3: Output Guardrails ===")
    exercise_3()

    print("\n=== Exercise 4: Multi-Capability Routing ===")
    exercise_4()

    print("\n=== Exercise 5: System Prompt Evaluator (no API) ===")
    exercise_5()

    print("\nAll exercises completed!")
