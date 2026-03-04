"""
Claude API: Messages API Examples

Demonstrates basic and streaming Messages API calls
using the Anthropic Python SDK.

Requirements:
    pip install anthropic
    export ANTHROPIC_API_KEY=sk-ant-...
"""

from __future__ import annotations

# --- Example 1: Basic Messages API Call ---

def basic_message():
    """Send a simple message and get a response."""
    import anthropic

    client = anthropic.Anthropic()

    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        system="You are a helpful assistant. Be concise.",
        messages=[
            {"role": "user", "content": "What is the capital of South Korea?"}
        ],
    )

    print(f"Response: {message.content[0].text}")
    print(f"Tokens: input={message.usage.input_tokens}, "
          f"output={message.usage.output_tokens}")
    print(f"Stop reason: {message.stop_reason}")


# --- Example 2: Multi-Turn Conversation ---

def multi_turn():
    """Demonstrate a multi-turn conversation."""
    import anthropic

    client = anthropic.Anthropic()

    messages = [
        {"role": "user", "content": "What is 2 + 2?"},
        {"role": "assistant", "content": "2 + 2 = 4."},
        {"role": "user", "content": "Now multiply that by 3."},
    ]

    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=256,
        messages=messages,
    )

    print(f"Response: {message.content[0].text}")


# --- Example 3: Streaming Response ---

def streaming():
    """Demonstrate streaming (Server-Sent Events) response."""
    import anthropic

    client = anthropic.Anthropic()

    print("Streaming: ", end="", flush=True)
    with client.messages.stream(
        model="claude-sonnet-4-20250514",
        max_tokens=256,
        messages=[{"role": "user", "content": "Count from 1 to 5."}],
    ) as stream:
        for text in stream.text_stream:
            print(text, end="", flush=True)
    print()  # newline


# --- Example 4: Temperature and Stop Sequences ---

def with_parameters():
    """Demonstrate advanced parameters."""
    import anthropic

    client = anthropic.Anthropic()

    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=512,
        temperature=0.0,  # deterministic output
        stop_sequences=["END"],
        messages=[
            {"role": "user",
             "content": "List 3 programming languages, then write END."}
        ],
    )

    print(f"Response: {message.content[0].text}")
    print(f"Stop reason: {message.stop_reason}")


# --- Main ---

if __name__ == "__main__":
    print("Claude API Messages Examples")
    print("=" * 40)
    print("\nNote: These examples require ANTHROPIC_API_KEY.")
    print("Set the environment variable before running.\n")

    # Uncomment to run (requires API key):
    # basic_message()
    # multi_turn()
    # streaming()
    # with_parameters()

    print("Examples defined. Uncomment the calls in __main__ to run.")
