# 07_multi_turn_conversation.py — Multi-turn context management and memory
#
# Run: python 07_multi_turn_conversation.py

"""
Demonstrates:
  1. Basic multi-turn conversation — maintaining message history
  2. Context window management    — summarizing older messages
  3. Memory injection             — inserting external facts mid-conversation
  4. Conversation branching       — forking a conversation into two paths
"""

import os
from copy import deepcopy

import anthropic

client: anthropic.Anthropic
MODEL = "claude-sonnet-4-20250514"


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def chat(
    messages: list[dict],
    system: str = "",
    max_tokens: int = 512,
) -> str:
    """Send a multi-turn conversation and return the assistant reply."""
    response = client.messages.create(
        model=MODEL,
        max_tokens=max_tokens,
        temperature=0.0,
        system=system,
        messages=messages,
    )
    return response.content[0].text.strip()


# ---------------------------------------------------------------------------
# 1. Basic Multi-Turn Conversation
# ---------------------------------------------------------------------------

def demo_basic_multi_turn():
    """Build a conversation turn by turn, accumulating history."""

    system = "You are a helpful Python tutor. Keep answers concise."
    history: list[dict] = []

    turns = [
        "What is a list comprehension in Python?",
        "Can you give me a one-line example that squares even numbers from 1 to 10?",
        "How would I add a condition to also exclude numbers greater than 6?",
    ]

    print("=" * 60)
    print("SECTION 1 — Basic Multi-Turn Conversation")
    print("=" * 60)

    for user_msg in turns:
        history.append({"role": "user", "content": user_msg})
        reply = chat(history, system=system)
        history.append({"role": "assistant", "content": reply})
        print(f"\nUser: {user_msg}")
        print(f"Assistant: {reply}")

    print(f"\n  [Total turns: {len(history) // 2}]")


# ---------------------------------------------------------------------------
# 2. Context Window Management — Sliding Window + Summary
# ---------------------------------------------------------------------------

def summarize_history(history: list[dict], system: str) -> str:
    """Ask Claude to compress older conversation into a summary."""
    transcript = "\n".join(
        f"{m['role'].upper()}: {m['content']}" for m in history
    )
    prompt = (
        f"Summarize this conversation in 2-3 sentences, preserving "
        f"key facts and decisions:\n\n{transcript}"
    )
    return chat([{"role": "user", "content": prompt}], system=system)


def demo_context_management():
    """When history grows too long, summarize older turns."""

    system = "You are a project planning assistant."
    history: list[dict] = []
    max_turns_before_summary = 3

    turns = [
        "We need to build a REST API for a bookstore.",
        "Let's use FastAPI with PostgreSQL.",
        "Add JWT authentication and rate limiting.",
        "What's our tech stack summary so far?",
        "Now plan the database schema for books and users.",
    ]

    print("\n" + "=" * 60)
    print("SECTION 2 — Context Window Management")
    print("=" * 60)

    for i, user_msg in enumerate(turns):
        # Summarize if history is getting long
        if len(history) >= max_turns_before_summary * 2:
            summary = summarize_history(history, system)
            print(f"\n  [Summarizing {len(history)//2} turns]")
            print(f"  Summary: {summary[:120]}...")
            # Replace history with summary + last turn
            history = [
                {"role": "user", "content": f"[Context summary: {summary}]"},
                {"role": "assistant", "content": "Understood, I have the context."},
            ]

        history.append({"role": "user", "content": user_msg})
        reply = chat(history, system=system)
        history.append({"role": "assistant", "content": reply})
        print(f"\nTurn {i+1} — User: {user_msg}")
        print(f"  Assistant: {reply[:200]}...")


# ---------------------------------------------------------------------------
# 3. Memory Injection
# ---------------------------------------------------------------------------

def demo_memory_injection():
    """Inject external knowledge into the conversation mid-flow."""

    system = "You are a travel planning assistant."
    history: list[dict] = []

    # Normal turn
    history.append({"role": "user", "content": "Plan a 3-day trip to Tokyo."})
    reply = chat(history, system=system)
    history.append({"role": "assistant", "content": reply})
    print("\n" + "=" * 60)
    print("SECTION 3 — Memory Injection")
    print("=" * 60)
    print(f"\nUser: Plan a 3-day trip to Tokyo.")
    print(f"Assistant: {reply[:200]}...")

    # Inject user preferences as a memory block
    memory = (
        "[USER PREFERENCES — retrieved from profile database]\n"
        "- Budget: moderate ($150/day)\n"
        "- Interests: street food, temples, anime culture\n"
        "- Dietary: vegetarian\n"
        "- Mobility: prefers walking, no car rental\n"
        "[END PREFERENCES]"
    )
    history.append({
        "role": "user",
        "content": f"{memory}\n\nRevise the plan using my preferences above.",
    })
    reply = chat(history, system=system)
    history.append({"role": "assistant", "content": reply})
    print(f"\nUser: [injected preferences] Revise the plan.")
    print(f"Assistant: {reply[:300]}...")


# ---------------------------------------------------------------------------
# 4. Conversation Branching
# ---------------------------------------------------------------------------

def demo_branching():
    """Fork a conversation to explore two different directions."""

    system = "You are a software architect."
    shared_history = [
        {"role": "user", "content": "We need a message queue for our microservices."},
    ]
    reply = chat(shared_history, system=system)
    shared_history.append({"role": "assistant", "content": reply})

    print("\n" + "=" * 60)
    print("SECTION 4 — Conversation Branching")
    print("=" * 60)
    print(f"\nShared: {reply[:150]}...")

    # Branch A — RabbitMQ path
    branch_a = deepcopy(shared_history)
    branch_a.append({"role": "user", "content": "Let's go with RabbitMQ. What's the setup?"})
    reply_a = chat(branch_a, system=system)
    print(f"\n[Branch A — RabbitMQ]\n  {reply_a[:200]}...")

    # Branch B — Kafka path
    branch_b = deepcopy(shared_history)
    branch_b.append({"role": "user", "content": "Let's go with Kafka. What's the setup?"})
    reply_b = chat(branch_b, system=system)
    print(f"\n[Branch B — Kafka]\n  {reply_b[:200]}...")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: Set the ANTHROPIC_API_KEY environment variable first.")
        raise SystemExit(1)

    client = anthropic.Anthropic()

    try:
        demo_basic_multi_turn()
        demo_context_management()
        demo_memory_injection()
        demo_branching()
    except anthropic.APIError as exc:
        print(f"\nAPI error: {exc}")
