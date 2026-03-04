"""
Exercises for Lesson 15: Claude API Fundamentals
Topic: Claude_Ecosystem

Solutions to practice problems from the lesson.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Generator


# === Exercise 1: Messages API Request Builder ===
# Problem: Build valid Messages API requests with proper structure
#   including system prompts, multi-turn conversations, and parameters.

@dataclass
class APIMessage:
    """A message in the Messages API format."""
    role: str      # "user" or "assistant"
    content: str


@dataclass
class MessagesRequest:
    """A Claude Messages API request."""
    model: str
    messages: list[APIMessage]
    max_tokens: int = 1024
    system: str = ""
    temperature: float = 1.0
    stop_sequences: list[str] = field(default_factory=list)
    stream: bool = False

    def to_dict(self) -> dict[str, Any]:
        request: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content}
                         for m in self.messages],
            "max_tokens": self.max_tokens,
        }
        if self.system:
            request["system"] = self.system
        if self.temperature != 1.0:
            request["temperature"] = self.temperature
        if self.stop_sequences:
            request["stop_sequences"] = self.stop_sequences
        if self.stream:
            request["stream"] = True
        return request

    def validate(self) -> list[str]:
        errors: list[str] = []
        valid_models = [
            "claude-opus-4-20250514", "claude-sonnet-4-20250514",
            "claude-haiku-4-5-20251001",
        ]
        if self.model not in valid_models:
            errors.append(f"Unknown model: {self.model}")
        if not self.messages:
            errors.append("At least one message required")
        if self.messages and self.messages[0].role != "user":
            errors.append("First message must be from user")
        for i, msg in enumerate(self.messages[1:], 1):
            if msg.role == self.messages[i - 1].role:
                errors.append(f"Consecutive {msg.role} messages at index {i}")
        if self.max_tokens < 1:
            errors.append("max_tokens must be positive")
        if not 0.0 <= self.temperature <= 1.0:
            errors.append("temperature must be between 0.0 and 1.0")
        return errors


def exercise_1():
    """Demonstrate API request construction and validation."""
    # Valid multi-turn request
    req = MessagesRequest(
        model="claude-sonnet-4-20250514",
        messages=[
            APIMessage("user", "What is the capital of France?"),
            APIMessage("assistant", "The capital of France is Paris."),
            APIMessage("user", "What about Germany?"),
        ],
        system="You are a geography expert. Be concise.",
        max_tokens=256,
        temperature=0.3,
    )
    errors = req.validate()
    print(f"  Valid request: {len(errors) == 0}")
    print(f"  Payload:\n    {json.dumps(req.to_dict(), indent=4)}")

    # Invalid request
    bad_req = MessagesRequest(
        model="gpt-4",
        messages=[
            APIMessage("assistant", "Hello"),  # wrong first role
            APIMessage("assistant", "Again"),   # consecutive
        ],
        temperature=2.0,
    )
    errors = bad_req.validate()
    print(f"\n  Invalid request errors:")
    for e in errors:
        print(f"    - {e}")


# === Exercise 2: Streaming Response Simulator ===
# Problem: Simulate the Server-Sent Events (SSE) streaming format
#   used by the Claude API.

@dataclass
class StreamEvent:
    """A single SSE event from the streaming API."""
    event_type: str
    data: dict[str, Any]


def simulate_stream(text: str, chunk_size: int = 5) -> Generator[StreamEvent, None, None]:
    """Simulate streaming a Claude API response.

    Yields SSE events matching the real API format:
    1. message_start
    2. content_block_start
    3. content_block_delta (repeated)
    4. content_block_stop
    5. message_delta
    6. message_stop
    """
    yield StreamEvent("message_start", {
        "type": "message_start",
        "message": {
            "id": "msg_sim_001",
            "type": "message",
            "role": "assistant",
            "model": "claude-sonnet-4-20250514",
        },
    })
    yield StreamEvent("content_block_start", {
        "type": "content_block_start",
        "index": 0,
        "content_block": {"type": "text", "text": ""},
    })

    for i in range(0, len(text), chunk_size):
        chunk = text[i:i + chunk_size]
        yield StreamEvent("content_block_delta", {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": chunk},
        })

    yield StreamEvent("content_block_stop", {
        "type": "content_block_stop", "index": 0,
    })
    yield StreamEvent("message_delta", {
        "type": "message_delta",
        "delta": {"stop_reason": "end_turn"},
        "usage": {"output_tokens": max(1, len(text) // 4)},
    })
    yield StreamEvent("message_stop", {"type": "message_stop"})


def exercise_2():
    """Demonstrate streaming response simulation."""
    text = "Paris is the capital of France."
    full_text = ""
    for event in simulate_stream(text, chunk_size=8):
        if event.event_type == "content_block_delta":
            chunk = event.data["delta"]["text"]
            full_text += chunk
            print(f"  [delta] '{chunk}'")
        elif event.event_type == "message_delta":
            tokens = event.data["usage"]["output_tokens"]
            print(f"  [done] stop_reason={event.data['delta']['stop_reason']}, "
                  f"tokens={tokens}")

    print(f"  Full text: '{full_text}'")


# === Exercise 3: Retry Logic with Exponential Backoff ===
# Problem: Implement retry logic for API calls with proper handling
#   of rate limits, overloaded errors, and exponential backoff.

@dataclass
class APIResponse:
    """Simulated API response."""
    status_code: int
    body: dict[str, Any]
    retry_after: int | None = None


def simulate_api_call(attempt: int) -> APIResponse:
    """Simulate an API call that fails initially then succeeds."""
    if attempt == 1:
        return APIResponse(529, {"error": {"type": "overloaded",
                                           "message": "API is overloaded"}})
    elif attempt == 2:
        return APIResponse(429, {"error": {"type": "rate_limit",
                                           "message": "Rate limited"}},
                           retry_after=2)
    else:
        return APIResponse(200, {
            "content": [{"type": "text", "text": "Success!"}],
            "usage": {"input_tokens": 10, "output_tokens": 5},
        })


def retry_with_backoff(
    max_retries: int = 5,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
) -> dict[str, Any]:
    """Execute an API call with exponential backoff retry logic."""
    retryable_codes = {429, 500, 502, 503, 529}
    attempts: list[dict[str, Any]] = []

    for attempt in range(1, max_retries + 1):
        response = simulate_api_call(attempt)
        delay = min(base_delay * (2 ** (attempt - 1)), max_delay)

        if response.retry_after:
            delay = max(delay, response.retry_after)

        attempts.append({
            "attempt": attempt,
            "status": response.status_code,
            "delay": round(delay, 1) if response.status_code in retryable_codes else 0,
        })

        if response.status_code == 200:
            return {"success": True, "attempts": attempts,
                    "result": response.body}

        if response.status_code not in retryable_codes:
            return {"success": False, "attempts": attempts,
                    "error": response.body}

    return {"success": False, "attempts": attempts,
            "error": "Max retries exceeded"}


def exercise_3():
    """Demonstrate retry with exponential backoff."""
    result = retry_with_backoff(max_retries=5, base_delay=1.0)
    print(f"  Success: {result['success']}")
    for a in result["attempts"]:
        status = a["status"]
        delay = f", retry in {a['delay']}s" if a["delay"] else ""
        print(f"  Attempt {a['attempt']}: HTTP {status}{delay}")
    if result["success"]:
        print(f"  Result: {result['result']['content'][0]['text']}")


if __name__ == "__main__":
    print("=== Exercise 1: Request Builder ===")
    exercise_1()

    print("\n=== Exercise 2: Streaming Simulator ===")
    exercise_2()

    print("\n=== Exercise 3: Retry Logic ===")
    exercise_3()

    print("\nAll exercises completed!")
