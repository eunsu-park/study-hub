# Claude API Fundamentals

**Previous**: [14. Claude Projects and Artifacts](./14_Claude_Projects_and_Artifacts.md) | **Next**: [16. Tool Use and Function Calling](./16_Tool_Use_and_Function_Calling.md)

---

The Claude API provides programmatic access to Claude models, enabling you to integrate Claude's capabilities into your applications, workflows, and automated systems. This lesson covers everything you need to start building with the Claude API: authentication, the Messages API, streaming, token management, error handling, and complete working examples in both Python and TypeScript.

**Difficulty**: ⭐⭐

**Prerequisites**:
- Python 3.9+ or Node.js 18+ installed
- An Anthropic API key (from console.anthropic.com)
- Basic familiarity with REST APIs and HTTP

## Learning Objectives

After completing this lesson, you will be able to:

1. Set up authentication and client SDKs in Python and TypeScript
2. Construct and send Messages API requests with system prompts
3. Process API responses including content blocks and usage metadata
4. Implement streaming for real-time response delivery
5. Manage tokens effectively (counting, budgeting, optimizing)
6. Handle errors with proper retry logic and exponential backoff
7. Build complete applications using the Claude API

---

## Table of Contents

1. [API Overview](#1-api-overview)
2. [Getting Started](#2-getting-started)
3. [Client SDKs](#3-client-sdks)
4. [Messages API](#4-messages-api)
5. [Streaming Responses](#5-streaming-responses)
6. [Token Counting and Management](#6-token-counting-and-management)
7. [Error Handling](#7-error-handling)
8. [Complete Working Examples](#8-complete-working-examples)
9. [Exercises](#9-exercises)
10. [References](#10-references)

---

## 1. API Overview

The Claude API is a REST API hosted at `https://api.anthropic.com`. The primary endpoint is the **Messages API**, which handles all text generation tasks.

```
┌────────────────────────────────────────────────────────────────┐
│                    Claude API Architecture                      │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Your Application                                               │
│  ┌─────────────────┐                                           │
│  │  Python / TS    │                                           │
│  │  Client SDK     │                                           │
│  └────────┬────────┘                                           │
│           │  HTTPS (TLS 1.2+)                                  │
│           │  x-api-key: sk-ant-...                             │
│           ▼                                                    │
│  ┌─────────────────────────────────────────┐                   │
│  │  https://api.anthropic.com              │                   │
│  │                                         │                   │
│  │  POST /v1/messages        ← Primary     │                   │
│  │  POST /v1/messages/count_tokens         │                   │
│  │  POST /v1/messages/batches  ← Batch API │                   │
│  │  GET  /v1/models           ← List       │                   │
│  └─────────────────────────────────────────┘                   │
│                                                                 │
│  Models Available:                                              │
│  ├── claude-opus-4-20250514     (most capable)                 │
│  ├── claude-sonnet-4-20250514   (balanced)                     │
│  └── claude-haiku-3-5-20241022  (fastest)                      │
│                                                                 │
└────────────────────────────────────────────────────────────────┘
```

**Key characteristics:**
- **Stateless**: Each API call is independent. There is no session state on the server.
- **Synchronous or streaming**: Choose between waiting for the full response or receiving it token by token.
- **Pay-per-token**: You are billed based on input tokens (your messages) and output tokens (Claude's response).

---

## 2. Getting Started

### 2.1 API Key Generation

1. Go to [console.anthropic.com](https://console.anthropic.com).
2. Create an account or sign in.
3. Navigate to **API Keys** in the dashboard.
4. Click **Create Key** and give it a descriptive name.
5. Copy the key immediately -- it will not be shown again.

Your API key looks like: `sk-ant-api03-...`

### 2.2 Authentication

Every API request must include the API key in the `x-api-key` header:

```bash
curl https://api.anthropic.com/v1/messages \
  -H "content-type: application/json" \
  -H "x-api-key: $ANTHROPIC_API_KEY" \
  -H "anthropic-version: 2023-06-01" \
  -d '{
    "model": "claude-sonnet-4-20250514",
    "max_tokens": 1024,
    "messages": [
      {"role": "user", "content": "Hello, Claude!"}
    ]
  }'
```

**Security best practices:**
- Never commit API keys to source control.
- Use environment variables: `export ANTHROPIC_API_KEY="sk-ant-..."`.
- Rotate keys periodically.
- Use separate keys for development and production.
- Set spending limits in the Anthropic console.

### 2.3 API Versioning

The `anthropic-version` header specifies the API version. Always include it:

```
anthropic-version: 2023-06-01
```

The SDKs handle this automatically, but if you make raw HTTP calls, you must include it.

---

## 3. Client SDKs

### 3.1 Python SDK

```bash
pip install anthropic
```

```python
import anthropic

# The SDK automatically reads ANTHROPIC_API_KEY from the environment
client = anthropic.Anthropic()

# Or provide the key explicitly
client = anthropic.Anthropic(api_key="sk-ant-...")
```

### 3.2 TypeScript SDK

```bash
npm install @anthropic-ai/sdk
```

```typescript
import Anthropic from "@anthropic-ai/sdk";

// Reads ANTHROPIC_API_KEY from the environment
const client = new Anthropic();

// Or provide explicitly
const client = new Anthropic({ apiKey: "sk-ant-..." });
```

### 3.3 Other Language SDKs

Official and community SDKs are available for many languages:

```
┌────────────────┬──────────────────────────────────────────────┐
│ Language       │ Package                                      │
├────────────────┼──────────────────────────────────────────────┤
│ Python         │ pip install anthropic         (official)     │
│ TypeScript/JS  │ npm install @anthropic-ai/sdk (official)     │
│ Java / Kotlin  │ com.anthropic:anthropic-java  (official)     │
│ Go             │ github.com/anthropics/anthropic-sdk-go       │
│ Ruby           │ gem install anthropic                        │
│ C# / .NET     │ NuGet: Anthropic                             │
│ PHP            │ composer require anthropic/anthropic         │
│ Rust           │ crates.io: anthropic (community)             │
└────────────────┴──────────────────────────────────────────────┘
```

All official SDKs follow the same design patterns, so concepts transfer across languages.

---

## 4. Messages API

The Messages API is the core endpoint for all Claude interactions.

### 4.1 Request Structure

```python
import anthropic

client = anthropic.Anthropic()

message = client.messages.create(
    model="claude-sonnet-4-20250514",     # Required: which model to use
    max_tokens=1024,                       # Required: max output tokens
    system="You are a helpful assistant.",  # Optional: system prompt
    messages=[                             # Required: conversation messages
        {
            "role": "user",
            "content": "What is the capital of France?"
        }
    ],
    temperature=0.7,                       # Optional: randomness (0-1)
    top_p=0.9,                             # Optional: nucleus sampling
    stop_sequences=["END"],                # Optional: stop generation
    metadata={"user_id": "user-123"},      # Optional: tracking metadata
)
```

### 4.2 Message Roles

The `messages` array uses two roles in a strict alternating pattern:

```python
messages = [
    # User message (your input)
    {"role": "user", "content": "Explain recursion."},

    # Assistant message (Claude's response — for multi-turn conversations)
    {"role": "assistant", "content": "Recursion is when a function calls itself..."},

    # User follow-up
    {"role": "user", "content": "Can you show me a Python example?"},
]
```

Rules:
- Messages must alternate between `user` and `assistant`.
- The first message must be from `user`.
- The last message must be from `user` (since you are asking Claude to respond).

### 4.3 System Prompt

The system prompt sets Claude's behavior for the entire conversation. It is separate from the messages array:

```python
message = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    system=[
        {
            "type": "text",
            "text": "You are a Python expert. Always provide type-annotated code. "
                    "Use descriptive variable names and include docstrings.",
        }
    ],
    messages=[
        {"role": "user", "content": "Write a function to find the nth Fibonacci number."}
    ],
)
```

The system prompt can also be a simple string: `system="You are a Python expert."`.

### 4.4 Response Structure

```python
# The response object
print(message.id)              # "msg_01XFDUDYJgAACzvnptvVoYEL"
print(message.type)            # "message"
print(message.role)            # "assistant"
print(message.model)           # "claude-sonnet-4-20250514"
print(message.stop_reason)     # "end_turn" | "max_tokens" | "stop_sequence" | "tool_use"
print(message.stop_sequence)   # The matched stop sequence, if any

# Content blocks (can contain text and tool_use blocks)
for block in message.content:
    if block.type == "text":
        print(block.text)      # The actual response text

# Token usage
print(message.usage.input_tokens)   # Tokens in your request
print(message.usage.output_tokens)  # Tokens in Claude's response
```

A typical response:

```json
{
  "id": "msg_01XFDUDYJgAACzvnptvVoYEL",
  "type": "message",
  "role": "assistant",
  "model": "claude-sonnet-4-20250514",
  "content": [
    {
      "type": "text",
      "text": "The capital of France is Paris."
    }
  ],
  "stop_reason": "end_turn",
  "stop_sequence": null,
  "usage": {
    "input_tokens": 25,
    "output_tokens": 12,
    "cache_creation_input_tokens": 0,
    "cache_read_input_tokens": 0
  }
}
```

### 4.5 Multi-Turn Conversations

To maintain conversation history, include all previous messages in each request:

```python
import anthropic

client = anthropic.Anthropic()
conversation: list[dict] = []

def chat(user_message: str) -> str:
    """Send a message and maintain conversation history."""
    conversation.append({"role": "user", "content": user_message})

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2048,
        system="You are a knowledgeable coding tutor.",
        messages=conversation,
    )

    assistant_message = response.content[0].text
    conversation.append({"role": "assistant", "content": assistant_message})

    return assistant_message

# Multi-turn conversation
print(chat("What is a hash table?"))
print(chat("How does collision resolution work?"))
print(chat("Show me a Python implementation."))
```

### 4.6 Multi-Modal Input (Images)

Claude can process images alongside text:

```python
import anthropic
import base64
from pathlib import Path

client = anthropic.Anthropic()

# Method 1: Base64-encoded image
image_data = base64.standard_b64encode(
    Path("screenshot.png").read_bytes()
).decode("utf-8")

message = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": image_data,
                    },
                },
                {
                    "type": "text",
                    "text": "Describe what you see in this image.",
                },
            ],
        }
    ],
)

# Method 2: URL-referenced image
message = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "url",
                        "url": "https://example.com/photo.jpg",
                    },
                },
                {
                    "type": "text",
                    "text": "What is shown in this photo?",
                },
            ],
        }
    ],
)
```

---

## 5. Streaming Responses

Streaming delivers Claude's response token by token as it is generated, rather than waiting for the complete response. This dramatically improves perceived latency for long responses.

### 5.1 Why Streaming?

```
Non-streaming:
  Request ──────── [wait 5 seconds] ──────── Full Response
  User sees nothing until the entire response is ready.

Streaming:
  Request ── token ── token ── token ── token ── ... ── done
  User sees the first token within milliseconds.
```

### 5.2 Python Streaming

```python
import anthropic

client = anthropic.Anthropic()

# Basic streaming
with client.messages.stream(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    messages=[
        {"role": "user", "content": "Write a short story about a robot learning to paint."}
    ],
) as stream:
    for text in stream.text_stream:
        print(text, end="", flush=True)

print()  # Final newline

# Access the final message after streaming completes
final_message = stream.get_final_message()
print(f"\nTokens used: {final_message.usage.input_tokens} in, "
      f"{final_message.usage.output_tokens} out")
```

**Streaming with event handling:**

```python
import anthropic

client = anthropic.Anthropic()

with client.messages.stream(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    messages=[
        {"role": "user", "content": "Explain quantum computing in simple terms."}
    ],
) as stream:
    for event in stream:
        # Each event has a type
        if event.type == "content_block_start":
            print("[START]", end="")
        elif event.type == "content_block_delta":
            if event.delta.type == "text_delta":
                print(event.delta.text, end="", flush=True)
        elif event.type == "content_block_stop":
            print("\n[END]")
        elif event.type == "message_stop":
            print("[MESSAGE COMPLETE]")
```

### 5.3 TypeScript Streaming

```typescript
import Anthropic from "@anthropic-ai/sdk";

const client = new Anthropic();

async function streamResponse() {
  // Basic streaming
  const stream = client.messages.stream({
    model: "claude-sonnet-4-20250514",
    max_tokens: 1024,
    messages: [
      { role: "user", content: "Write a haiku about programming." },
    ],
  });

  // Stream text to console
  stream.on("text", (text) => {
    process.stdout.write(text);
  });

  // Get the final message when done
  const finalMessage = await stream.finalMessage();
  console.log(`\nTokens: ${finalMessage.usage.input_tokens} in, ` +
              `${finalMessage.usage.output_tokens} out`);
}

streamResponse();
```

**Async iteration pattern:**

```typescript
import Anthropic from "@anthropic-ai/sdk";

const client = new Anthropic();

async function streamWithAsyncIterator() {
  const stream = await client.messages.create({
    model: "claude-sonnet-4-20250514",
    max_tokens: 1024,
    stream: true,  // Enable streaming at the request level
    messages: [
      { role: "user", content: "List the planets in our solar system." },
    ],
  });

  for await (const event of stream) {
    if (
      event.type === "content_block_delta" &&
      event.delta.type === "text_delta"
    ) {
      process.stdout.write(event.delta.text);
    }
  }
  console.log();
}

streamWithAsyncIterator();
```

### 5.4 Server-Sent Events Format

Under the hood, streaming uses Server-Sent Events (SSE). The raw format looks like:

```
event: message_start
data: {"type": "message_start", "message": {"id": "msg_...", "model": "claude-sonnet-4-20250514", ...}}

event: content_block_start
data: {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}

event: content_block_delta
data: {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "The"}}

event: content_block_delta
data: {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": " capital"}}

event: content_block_stop
data: {"type": "content_block_stop", "index": 0}

event: message_delta
data: {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 12}}

event: message_stop
data: {"type": "message_stop"}
```

The SDKs parse this format automatically, but understanding it helps when debugging or building custom integrations.

---

## 6. Token Counting and Management

### 6.1 What Are Tokens?

Tokens are the units Claude uses to process text. One token is roughly 3-4 English characters or about 0.75 words. Some examples:

```
"Hello"             → 1 token
"Hello, world!"     → 4 tokens
"antidisestablish-  → 4 tokens
 mentarianism"
"def fibonacci(n):" → 5 tokens
"こんにちは"         → 3 tokens (non-English may use more tokens)
```

### 6.2 Token Counting Before Sending

Use the count tokens endpoint to estimate cost before sending a request:

```python
import anthropic

client = anthropic.Anthropic()

# Count tokens without making a generation request
token_count = client.messages.count_tokens(
    model="claude-sonnet-4-20250514",
    messages=[
        {"role": "user", "content": "Write a comprehensive guide to Python decorators."}
    ],
    system="You are a Python expert.",
)

print(f"Input tokens: {token_count.input_tokens}")
# Use this to estimate cost before sending the actual request
```

### 6.3 Managing max_tokens

The `max_tokens` parameter controls the maximum number of tokens Claude will generate:

```python
# Short response (quick answers)
message = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=100,
    messages=[{"role": "user", "content": "What is 2+2?"}],
)

# Long response (detailed explanations, code)
message = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=4096,
    messages=[{"role": "user", "content": "Write a full REST API in FastAPI."}],
)
```

If Claude's response hits `max_tokens`, the `stop_reason` will be `"max_tokens"` and the response will be truncated. Handle this:

```python
if message.stop_reason == "max_tokens":
    print("Warning: Response was truncated. Consider increasing max_tokens.")
```

### 6.4 Token Budget Strategy

```python
def estimate_cost(input_tokens: int, output_tokens: int, model: str) -> float:
    """Estimate the cost of an API call in USD."""
    # Pricing as of early 2026 (check docs for current prices)
    pricing = {
        "claude-opus-4-20250514":   {"input": 15.00, "output": 75.00},
        "claude-sonnet-4-20250514": {"input": 3.00,  "output": 15.00},
        "claude-haiku-3-5-20241022": {"input": 0.80,  "output": 4.00},
    }

    rates = pricing.get(model, pricing["claude-sonnet-4-20250514"])
    cost = (
        (input_tokens / 1_000_000) * rates["input"] +
        (output_tokens / 1_000_000) * rates["output"]
    )
    return cost

# Example
cost = estimate_cost(
    input_tokens=1500,
    output_tokens=800,
    model="claude-sonnet-4-20250514"
)
print(f"Estimated cost: ${cost:.4f}")
```

---

## 7. Error Handling

### 7.1 Common Error Codes

```
┌──────────┬───────────────────────┬──────────────────────────────┐
│ Code     │ Error                 │ What to Do                   │
├──────────┼───────────────────────┼──────────────────────────────┤
│ 400      │ Invalid request       │ Check request format, params │
│ 401      │ Authentication error  │ Check API key                │
│ 403      │ Permission denied     │ Check API key permissions    │
│ 404      │ Not found             │ Check endpoint URL           │
│ 429      │ Rate limited          │ Wait and retry with backoff  │
│ 500      │ API error (server)    │ Retry after a delay          │
│ 529      │ API overloaded        │ Retry with backoff           │
└──────────┴───────────────────────┴──────────────────────────────┘
```

### 7.2 Python Error Handling

```python
import anthropic

client = anthropic.Anthropic()

try:
    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        messages=[{"role": "user", "content": "Hello!"}],
    )
    print(message.content[0].text)

except anthropic.AuthenticationError:
    print("Invalid API key. Check your ANTHROPIC_API_KEY.")

except anthropic.RateLimitError:
    print("Rate limited. Wait before retrying.")

except anthropic.BadRequestError as e:
    print(f"Invalid request: {e.message}")

except anthropic.InternalServerError:
    print("Anthropic server error. Retry later.")

except anthropic.APIStatusError as e:
    print(f"API error (status {e.status_code}): {e.message}")

except anthropic.APIConnectionError:
    print("Could not connect to the API. Check your network.")
```

### 7.3 TypeScript Error Handling

```typescript
import Anthropic from "@anthropic-ai/sdk";

const client = new Anthropic();

async function callClaude() {
  try {
    const message = await client.messages.create({
      model: "claude-sonnet-4-20250514",
      max_tokens: 1024,
      messages: [{ role: "user", content: "Hello!" }],
    });
    console.log(message.content[0].type === "text" ? message.content[0].text : "");

  } catch (error) {
    if (error instanceof Anthropic.AuthenticationError) {
      console.error("Invalid API key.");
    } else if (error instanceof Anthropic.RateLimitError) {
      console.error("Rate limited. Retry later.");
    } else if (error instanceof Anthropic.BadRequestError) {
      console.error(`Bad request: ${error.message}`);
    } else if (error instanceof Anthropic.InternalServerError) {
      console.error("Server error. Retry later.");
    } else if (error instanceof Anthropic.APIError) {
      console.error(`API error (${error.status}): ${error.message}`);
    } else {
      throw error;  // Re-throw unexpected errors
    }
  }
}

callClaude();
```

### 7.4 Retry with Exponential Backoff

```python
import time
import random
import anthropic

client = anthropic.Anthropic()

def call_with_retry(
    messages: list,
    max_retries: int = 5,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
) -> anthropic.types.Message:
    """Call the API with exponential backoff on retryable errors."""
    for attempt in range(max_retries):
        try:
            return client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1024,
                messages=messages,
            )
        except (anthropic.RateLimitError, anthropic.InternalServerError,
                anthropic.APIConnectionError) as e:
            if attempt == max_retries - 1:
                raise  # Final attempt: re-raise

            # Exponential backoff with jitter
            delay = min(base_delay * (2 ** attempt), max_delay)
            jitter = random.uniform(0, delay * 0.1)
            wait_time = delay + jitter

            print(f"Attempt {attempt + 1} failed ({type(e).__name__}). "
                  f"Retrying in {wait_time:.1f}s...")
            time.sleep(wait_time)

        except anthropic.BadRequestError:
            raise  # Don't retry client errors

    raise RuntimeError("Should not reach here")


# Usage
response = call_with_retry([
    {"role": "user", "content": "What is the meaning of life?"}
])
print(response.content[0].text)
```

Note: The official Python SDK has built-in retry logic. By default, it retries up to 2 times on `429`, `500`, and connection errors. You can configure this:

```python
client = anthropic.Anthropic(
    max_retries=5,       # Maximum retries (default: 2)
    timeout=60.0,        # Request timeout in seconds (default: 600)
)
```

---

## 8. Complete Working Examples

### 8.1 Python: Interactive Chat Application

```python
#!/usr/bin/env python3
"""Interactive chat application using the Claude API."""

import anthropic
import sys

def main():
    client = anthropic.Anthropic()
    conversation: list[dict] = []
    system_prompt = (
        "You are a helpful assistant. Be concise but thorough. "
        "When providing code, include brief comments explaining key parts."
    )

    print("Claude Chat (type 'quit' to exit, 'clear' to reset)")
    print("=" * 50)

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() == "quit":
            print("Goodbye!")
            break
        if user_input.lower() == "clear":
            conversation.clear()
            print("[Conversation cleared]")
            continue

        conversation.append({"role": "user", "content": user_input})

        try:
            print("\nClaude: ", end="", flush=True)
            full_response = ""

            with client.messages.stream(
                model="claude-sonnet-4-20250514",
                max_tokens=2048,
                system=system_prompt,
                messages=conversation,
            ) as stream:
                for text in stream.text_stream:
                    print(text, end="", flush=True)
                    full_response += text

            final = stream.get_final_message()
            print(f"\n  [{final.usage.input_tokens} in / "
                  f"{final.usage.output_tokens} out tokens]")

            conversation.append({
                "role": "assistant",
                "content": full_response,
            })

        except anthropic.APIError as e:
            print(f"\n[Error: {e.message}]")
            # Remove the failed user message to keep conversation valid
            conversation.pop()

if __name__ == "__main__":
    main()
```

### 8.2 TypeScript: Code Review Assistant

```typescript
// code-review.ts — Automated code review using Claude API
import Anthropic from "@anthropic-ai/sdk";
import { readFileSync } from "fs";

const client = new Anthropic();

interface ReviewResult {
  summary: string;
  issues: Array<{
    severity: "error" | "warning" | "info";
    line: number | null;
    message: string;
    suggestion: string;
  }>;
  score: number; // 1-10
}

async function reviewCode(
  filePath: string,
  language: string
): Promise<ReviewResult> {
  const code = readFileSync(filePath, "utf-8");

  const message = await client.messages.create({
    model: "claude-sonnet-4-20250514",
    max_tokens: 4096,
    system: `You are an expert code reviewer. Analyze the provided ${language} code
and respond with a JSON object containing:
- "summary": A 1-2 sentence summary of the code quality.
- "issues": An array of objects, each with "severity" (error/warning/info),
  "line" (line number or null), "message" (description), and "suggestion" (fix).
- "score": An integer 1-10 rating (10 = excellent).

Respond with ONLY the JSON object, no other text.`,
    messages: [
      {
        role: "user",
        content: `Review this ${language} code:\n\n\`\`\`${language}\n${code}\n\`\`\``,
      },
    ],
  });

  const responseText =
    message.content[0].type === "text" ? message.content[0].text : "";

  // Parse the JSON response
  const jsonMatch = responseText.match(/\{[\s\S]*\}/);
  if (!jsonMatch) {
    throw new Error("Failed to parse review response as JSON");
  }

  return JSON.parse(jsonMatch[0]) as ReviewResult;
}

// Main execution
async function main() {
  const filePath = process.argv[2];
  const language = process.argv[3] || "python";

  if (!filePath) {
    console.error("Usage: npx tsx code-review.ts <file-path> [language]");
    process.exit(1);
  }

  console.log(`Reviewing ${filePath} (${language})...`);
  const review = await reviewCode(filePath, language);

  console.log(`\nScore: ${review.score}/10`);
  console.log(`Summary: ${review.summary}\n`);

  if (review.issues.length === 0) {
    console.log("No issues found!");
  } else {
    console.log(`Issues (${review.issues.length}):`);
    for (const issue of review.issues) {
      const line = issue.line ? `L${issue.line}` : "—";
      const icon =
        issue.severity === "error" ? "[!]" :
        issue.severity === "warning" ? "[~]" : "[i]";
      console.log(`  ${icon} ${line}: ${issue.message}`);
      console.log(`      Fix: ${issue.suggestion}`);
    }
  }
}

main().catch(console.error);
```

### 8.3 Python: Document Summarizer with Token Management

```python
#!/usr/bin/env python3
"""Summarize long documents with automatic chunking for token limits."""

import anthropic
from pathlib import Path

client = anthropic.Anthropic()

# Model context limits (approximate input token budgets)
MODEL_LIMITS = {
    "claude-opus-4-20250514": 190_000,
    "claude-sonnet-4-20250514": 190_000,
    "claude-haiku-3-5-20241022": 190_000,
}

def count_tokens(text: str, model: str = "claude-sonnet-4-20250514") -> int:
    """Count tokens in a text string."""
    result = client.messages.count_tokens(
        model=model,
        messages=[{"role": "user", "content": text}],
    )
    return result.input_tokens

def summarize_text(
    text: str,
    model: str = "claude-sonnet-4-20250514",
    max_output_tokens: int = 2048,
) -> str:
    """Summarize a text, chunking if necessary."""
    token_count = count_tokens(text, model)
    max_input = MODEL_LIMITS.get(model, 190_000) - max_output_tokens - 500  # margin

    print(f"Document tokens: {token_count:,}")
    print(f"Max input tokens: {max_input:,}")

    if token_count <= max_input:
        # Document fits in a single request
        return _summarize_single(text, model, max_output_tokens)
    else:
        # Document is too long: split into chunks and summarize hierarchically
        return _summarize_chunked(text, model, max_input, max_output_tokens)

def _summarize_single(text: str, model: str, max_tokens: int) -> str:
    """Summarize a text that fits in a single context window."""
    message = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system="You are an expert summarizer. Create clear, comprehensive summaries "
               "that capture all key points, data, and conclusions.",
        messages=[
            {
                "role": "user",
                "content": f"Summarize the following document:\n\n{text}",
            }
        ],
    )
    return message.content[0].text

def _summarize_chunked(
    text: str, model: str, max_chunk_tokens: int, max_output_tokens: int
) -> str:
    """Summarize a long text by splitting into chunks."""
    # Simple split by paragraphs (production code should use better chunking)
    paragraphs = text.split("\n\n")
    chunks: list[str] = []
    current_chunk: list[str] = []
    current_tokens = 0

    for para in paragraphs:
        para_tokens = count_tokens(para, model)
        if current_tokens + para_tokens > max_chunk_tokens and current_chunk:
            chunks.append("\n\n".join(current_chunk))
            current_chunk = [para]
            current_tokens = para_tokens
        else:
            current_chunk.append(para)
            current_tokens += para_tokens

    if current_chunk:
        chunks.append("\n\n".join(current_chunk))

    print(f"Split into {len(chunks)} chunks")

    # Summarize each chunk
    chunk_summaries: list[str] = []
    for i, chunk in enumerate(chunks):
        print(f"  Summarizing chunk {i + 1}/{len(chunks)}...")
        summary = _summarize_single(chunk, model, max_output_tokens // len(chunks))
        chunk_summaries.append(summary)

    # Combine chunk summaries into a final summary
    combined = "\n\n---\n\n".join(
        f"Section {i + 1} Summary:\n{s}" for i, s in enumerate(chunk_summaries)
    )

    print("  Generating final summary...")
    return _summarize_single(
        f"Combine these section summaries into a single coherent summary:\n\n{combined}",
        model,
        max_output_tokens,
    )

# Usage
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python summarizer.py <file_path>")
        sys.exit(1)

    file_path = Path(sys.argv[1])
    text = file_path.read_text()
    summary = summarize_text(text)
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print('=' * 60)
    print(summary)
```

---

## 9. Exercises

### Exercise 1: First API Call (Beginner)

Set up the Python or TypeScript SDK and make your first API call. Ask Claude to explain a programming concept of your choice. Print the response text and the token usage. Try different `temperature` values (0.0, 0.5, 1.0) and observe how the responses change.

### Exercise 2: Multi-Turn Calculator (Beginner)

Build a conversational calculator that maintains a running total. Use multi-turn conversation to let the user say things like "add 5", "multiply by 3", "what is the current total?" and have Claude track the state.

### Exercise 3: Streaming Progress (Intermediate)

Build a Python script that streams Claude's response and shows a progress indicator. Display: the number of tokens generated so far, elapsed time, and tokens per second. Print a final summary with total cost estimate.

### Exercise 4: Batch Processing (Intermediate)

Write a script that takes a directory of text files, sends each to Claude for summarization using the Messages API, and writes the summaries to output files. Include error handling, retry logic, and a progress bar. Respect rate limits by adding delays between requests.

### Exercise 5: API Wrapper Library (Advanced)

Create a small wrapper library around the Anthropic SDK that provides:
- Automatic retry with exponential backoff
- Token usage tracking across multiple calls
- Cost estimation and budget enforcement (refuse to make calls over a dollar threshold)
- Conversation history management with automatic pruning when context gets too long
- Structured logging of all API calls

---

## 10. References

- Anthropic API Reference - https://docs.anthropic.com/en/api/messages
- Anthropic Python SDK - https://github.com/anthropics/anthropic-sdk-python
- Anthropic TypeScript SDK - https://github.com/anthropics/anthropic-sdk-typescript
- Anthropic Cookbook - https://github.com/anthropics/anthropic-cookbook
- API Pricing - https://www.anthropic.com/pricing
- Rate Limits - https://docs.anthropic.com/en/api/rate-limits
- Token Counting - https://docs.anthropic.com/en/docs/build-with-claude/token-counting
- Streaming - https://docs.anthropic.com/en/api/messages-streaming

---

## Next Lesson

[16. Tool Use and Function Calling](./16_Tool_Use_and_Function_Calling.md) extends your API knowledge with tool use -- letting Claude call functions you define. You will learn how to define tools, handle the tool use conversation flow, manage parallel tool calls, and build practical integrations.
