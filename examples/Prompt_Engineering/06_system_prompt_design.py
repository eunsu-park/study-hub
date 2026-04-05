# 06_system_prompt_design.py — System prompt patterns, personas, and guardrails
#
# Run: python 06_system_prompt_design.py

"""
Demonstrates:
  1. Persona design    — crafting a system prompt character
  2. Behavioral rules  — constraining model behavior
  3. Guardrails        — refusing out-of-scope or unsafe requests
  4. Layered system prompts — combining persona + rules + format
"""

import os

import anthropic

client: anthropic.Anthropic


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def call_claude(
    user_msg: str,
    system: str = "",
    temperature: float = 0.0,
) -> str:
    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=512,
        temperature=temperature,
        system=system,
        messages=[{"role": "user", "content": user_msg}],
    )
    return message.content[0].text.strip()


# ---------------------------------------------------------------------------
# 1. Persona Design
# ---------------------------------------------------------------------------

TUTOR_PERSONA = (
    "You are Professor Ada, a patient and encouraging computer science tutor. "
    "You always explain concepts using real-world analogies before giving "
    "technical definitions. You address the student by name when possible. "
    "Your tone is warm but precise."
)

PIRATE_PERSONA = (
    "You are Captain CodeBeard, a pirate who explains programming in nautical "
    "metaphors. You say 'Ahoy!' at the start and end with 'Now swab the deck "
    "and write some code, ye scallywag!' Keep explanations accurate despite "
    "the colorful language."
)


def demo_persona():
    question = "What is a hash table?"

    print("=" * 60)
    print("SECTION 1 — Persona Design")
    print("=" * 60)

    print("\n[Professor Ada]")
    print(call_claude(question, system=TUTOR_PERSONA))

    print("\n[Captain CodeBeard]")
    print(call_claude(question, system=PIRATE_PERSONA))


# ---------------------------------------------------------------------------
# 2. Behavioral Rules
# ---------------------------------------------------------------------------

STRICT_RULES = """\
You are a technical documentation assistant. Follow these rules EXACTLY:

1. NEVER use first person ("I", "my"). Use passive voice or "the system".
2. ALWAYS include a code example for any concept you explain.
3. LIMIT responses to 150 words maximum.
4. FORMAT: Start with a one-line definition, then a code block, then a caveat.
5. If asked about non-technical topics, reply: "This assistant only handles technical documentation queries."
"""


def demo_behavioral_rules():
    print("\n" + "=" * 60)
    print("SECTION 2 — Behavioral Rules")
    print("=" * 60)

    # On-topic request
    print("\n[On-topic: 'Explain Python decorators']")
    print(call_claude("Explain Python decorators.", system=STRICT_RULES))

    # Off-topic request
    print("\n[Off-topic: 'What is the capital of France?']")
    print(call_claude("What is the capital of France?", system=STRICT_RULES))


# ---------------------------------------------------------------------------
# 3. Guardrails — Topic Boundary + Safety
# ---------------------------------------------------------------------------

GUARDRAIL_SYSTEM = """\
You are a medical information assistant for general wellness topics.

SAFETY GUARDRAILS:
- NEVER provide specific diagnoses or prescribe medication.
- If the user describes symptoms, advise them to consult a healthcare professional.
- NEVER generate content about self-harm or dangerous substances.
- If a question is outside your scope, say: "I can only help with general wellness information. Please consult a qualified professional for this topic."

TOPIC BOUNDARY:
- Nutrition, exercise, sleep hygiene, stress management: ALLOWED
- Specific medical conditions, drug dosages, mental health crises: REDIRECT TO PROFESSIONAL
"""


def demo_guardrails():
    print("\n" + "=" * 60)
    print("SECTION 3 — Guardrails")
    print("=" * 60)

    safe_q = "What are some tips for better sleep hygiene?"
    print(f"\n[Safe: '{safe_q}']")
    print(call_claude(safe_q, system=GUARDRAIL_SYSTEM))

    risky_q = "I have sharp chest pain and shortness of breath. What should I take?"
    print(f"\n[Risky: '{risky_q}']")
    print(call_claude(risky_q, system=GUARDRAIL_SYSTEM))


# ---------------------------------------------------------------------------
# 4. Layered System Prompt (Persona + Rules + Format)
# ---------------------------------------------------------------------------

LAYERED_SYSTEM = """\
## PERSONA
You are Sage, a senior code reviewer at a top tech company. You are direct, constructive, and cite best practices by name (e.g., "SOLID principles", "DRY").

## RULES
1. Always structure your review as: Summary | Issues (numbered) | Suggestions.
2. Rate the code on a scale of 1-10.
3. If the code has no issues, say so explicitly — do not invent problems.
4. Keep the total response under 200 words.

## OUTPUT FORMAT
```
Rating: X/10
Summary: ...
Issues:
  1. ...
Suggestions:
  - ...
```
"""


def demo_layered():
    code_snippet = '''\
def calc(x):
    if x == 0:
        return 1
    r = 1
    for i in range(1, x+1):
        r = r * i
    return r
'''

    print("\n" + "=" * 60)
    print("SECTION 4 — Layered System Prompt")
    print("=" * 60)
    print(call_claude(f"Review this code:\n```python\n{code_snippet}```",
                      system=LAYERED_SYSTEM))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: Set the ANTHROPIC_API_KEY environment variable first.")
        raise SystemExit(1)

    client = anthropic.Anthropic()

    try:
        demo_persona()
        demo_behavioral_rules()
        demo_guardrails()
        demo_layered()
    except anthropic.APIError as exc:
        print(f"\nAPI error: {exc}")
