# Exercise: Lesson 07 — Multi-Turn Conversation
# Complete the TODO items below.
#
# Run: python 07_multi_turn_conversation.py

import anthropic
import json

client = anthropic.Anthropic()  # expects ANTHROPIC_API_KEY env var

MODEL = "claude-sonnet-4-20250514"


# === Exercise 1: Basic Multi-Turn Conversation ===
# Maintain a message history and send it with each API call.
# Hint: The messages list alternates user/assistant roles.

def chat(history: list[dict], user_message: str,
         system: str = "You are a helpful assistant.") -> str:
    """Append user_message to history, call the API, append assistant
    response to history, and return the response text.
    Mutates `history` in place.
    """
    # TODO: Append {"role": "user", "content": user_message} to history
    # TODO: Call client.messages.create() with system and full history
    # TODO: Extract the assistant's text response
    # TODO: Append {"role": "assistant", "content": response_text} to history
    # TODO: Return the response text
    pass


def exercise_1():
    history = []
    r1 = chat(history, "My name is Alex. Remember it.")
    r2 = chat(history, "What is my name?")
    assert len(history) == 4, f"Expected 4 messages, got {len(history)}"
    assert "Alex" in r2, "Model should remember the name"
    print(f"[Ex1] Turn 1: {r1[:60]}")
    print(f"[Ex1] Turn 2: {r2[:60]}")
    print(f"[Ex1] History length: {len(history)} messages")


# === Exercise 2: Sliding Window Memory ===
# Keep only the most recent N turns to stay within token limits.
# Hint: A "turn" is one user message + one assistant message (2 entries).

def sliding_window_chat(history: list[dict], user_message: str,
                        max_turns: int = 5,
                        system: str = "You are a helpful assistant.") -> str:
    """Like chat(), but trims history to the most recent max_turns turns
    before sending to the API. Still appends to the full history.
    """
    # TODO: Append the user message to history
    # TODO: Build a trimmed copy: keep only the last (max_turns * 2) messages
    # TODO: Call the API with the trimmed history
    # TODO: Append assistant response to history (the full one)
    # TODO: Return the response text
    pass


def exercise_2():
    history = []
    # Simulate 8 turns, window = 3
    for i in range(8):
        resp = sliding_window_chat(
            history, f"This is message number {i + 1}.", max_turns=3
        )
    assert len(history) == 16, f"Full history should have 16, got {len(history)}"
    # The model should NOT remember message 1 (outside window)
    check = sliding_window_chat(history, "What was message number 1?", max_turns=3)
    print(f"[Ex2] Full history: {len(history)} messages")
    print(f"[Ex2] Window check: {check[:80]}")


# === Exercise 3: Conversation Summarizer ===
# Summarize older messages to compress history while retaining key info.
# Hint: Ask Claude to summarize the conversation so far.

def summarize_history(history: list[dict]) -> str:
    """Summarize a conversation history into a concise paragraph.
    Return the summary string.
    """
    # TODO: Format the history into a readable transcript
    # TODO: Ask Claude to summarize the key points in 2-3 sentences
    # TODO: Return the summary text
    pass


def summarized_chat(history: list[dict], user_message: str,
                    max_turns: int = 4, system: str = "") -> str:
    """Chat with automatic summarization: when history exceeds max_turns,
    summarize the oldest half and prepend the summary as context.
    """
    # TODO: Append user message
    # TODO: If len(history) > max_turns * 2:
    #   a. Split history into old_half and recent_half
    #   b. Summarize old_half
    #   c. Replace history with: [summary as user msg, ack as assistant] + recent_half
    # TODO: Call the API and append response
    pass


def exercise_3():
    history = []
    topics = [
        "I'm learning Python. I know variables and loops.",
        "Now teach me about functions.",
        "What about lambda functions?",
        "How do decorators work?",
        "Explain generators and yield.",
        "What are context managers?",
    ]
    for msg in topics:
        resp = summarized_chat(history, msg, max_turns=3)
    print(f"[Ex3] History after 6 turns: {len(history)} messages (compressed)")
    print(f"[Ex3] Last response: {resp[:80]}...")


# === Exercise 4: Conversation State Tracker ===
# Track extracted entities and facts across a conversation.
# This is a pure Python exercise (no API call needed).

class ConversationState:
    """Track key-value facts extracted from conversation turns."""

    def __init__(self):
        self.facts: dict[str, str] = {}
        self.turn_count: int = 0

    def update(self, message: str) -> None:
        """Parse a user message and extract key facts.
        Look for patterns like "My name is X", "I live in X",
        "I work as X", "I like X".
        """
        # TODO: Use simple string matching or regex to extract facts
        # TODO: Store them in self.facts with keys like "name", "city", etc.
        # TODO: Increment turn_count
        pass

    def get_context_string(self) -> str:
        """Return a summary string of all known facts."""
        # TODO: Format self.facts into a readable string
        # Example: "Known facts: name=Alex, city=Boston, ..."
        pass


def exercise_4():
    state = ConversationState()
    messages = [
        "Hi, my name is Alex.",
        "I live in Berlin and work as a data scientist.",
        "I like hiking and photography.",
    ]
    for msg in messages:
        state.update(msg)
    assert state.turn_count == 3
    assert "name" in state.facts or len(state.facts) > 0
    context = state.get_context_string()
    print(f"[Ex4] Facts: {state.facts}")
    print(f"[Ex4] Context: {context}")


# === Exercise 5: Stateful Chat with Context Injection ===
# Combine the state tracker with the chat function.

def stateful_chat(history: list[dict], state: ConversationState,
                  user_message: str) -> str:
    """Chat that injects tracked state into the system prompt."""
    # TODO: Update state with the user message
    # TODO: Build a system prompt that includes state.get_context_string()
    # TODO: Use sliding_window_chat (or chat) with this dynamic system prompt
    # TODO: Return the response
    pass


def exercise_5():
    history = []
    state = ConversationState()
    r1 = stateful_chat(history, state, "My name is Jordan. I live in Tokyo.")
    r2 = stateful_chat(history, state, "What city do I live in?")
    print(f"[Ex5] Turn 1: {r1[:60]}")
    print(f"[Ex5] Turn 2: {r2[:60]}")
    print(f"[Ex5] Tracked state: {state.facts}")


if __name__ == "__main__":
    print("=== Exercise 1: Basic Multi-Turn ===")
    exercise_1()

    print("\n=== Exercise 2: Sliding Window ===")
    exercise_2()

    print("\n=== Exercise 3: Conversation Summarizer ===")
    exercise_3()

    print("\n=== Exercise 4: State Tracker (no API) ===")
    exercise_4()

    print("\n=== Exercise 5: Stateful Chat ===")
    exercise_5()

    print("\nAll exercises completed!")
