# 07. Multi-Turn Conversation

**Previous**: [System Prompt Design](./06_System_Prompt_Design.md) | **Next**: [Multimodal Prompting](./08_Multimodal_Prompting.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Design multi-turn conversation flows that maintain coherence and context across turns
2. Implement context window management strategies including sliding window, summarization, and entity memory
3. Build conversation state tracking systems that persist relevant information across turns
4. Handle topic switches, error recovery, and conversation branching gracefully
5. Apply context compression techniques for long conversations that exceed token limits

---

A single prompt-response exchange is the simplest form of LLM interaction, but real applications are conversational. Users ask follow-up questions, change their minds, refer back to earlier context, and expect the assistant to maintain a coherent understanding of the entire dialogue. Multi-turn conversation design is the art of managing this complexity: deciding what context to keep, what to compress, and how to steer the conversation toward productive outcomes -- all while working within fixed context window limits.

This lesson covers the full lifecycle of multi-turn conversation: from the mechanics of how LLM APIs handle message history, through memory management patterns, to advanced techniques for long-running dialogues.

## Table of Contents

1. [Multi-Turn Conversation Design](#1-multi-turn-conversation-design)
2. [Context Window Management](#2-context-window-management)
3. [Conversation State Tracking](#3-conversation-state-tracking)
4. [Memory Injection Patterns](#4-memory-injection-patterns)
5. [Turn-Taking and Conversation Steering](#5-turn-taking-and-conversation-steering)
6. [Handling Topic Switches](#6-handling-topic-switches)
7. [Error Recovery in Conversations](#7-error-recovery-in-conversations)
8. [Conversation Branching](#8-conversation-branching)
9. [Long Conversations and Context Compression](#9-long-conversations-and-context-compression)
10. [Designing Conversation Flows](#10-designing-conversation-flows)

---

## 1. Multi-Turn Conversation Design

### 1.1 How Multi-Turn Works in LLM APIs

LLMs are stateless. They do not "remember" previous conversations. What appears as memory is actually the full conversation history being sent with every request. Understanding this is fundamental to multi-turn design.

```python
import anthropic

client = anthropic.Anthropic()

# Turn 1: The model has no context
response1 = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    system="You are a helpful programming tutor.",
    messages=[
        {"role": "user", "content": "What is a Python decorator?"}
    ]
)

# Turn 2: We send the ENTIRE conversation history
response2 = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    system="You are a helpful programming tutor.",
    messages=[
        {"role": "user", "content": "What is a Python decorator?"},
        {"role": "assistant", "content": response1.content[0].text},
        {"role": "user", "content": "Can you show me an example with arguments?"}
    ]
)

# Turn 3: Full history again
response3 = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    system="You are a helpful programming tutor.",
    messages=[
        {"role": "user", "content": "What is a Python decorator?"},
        {"role": "assistant", "content": response1.content[0].text},
        {"role": "user", "content": "Can you show me an example with arguments?"},
        {"role": "assistant", "content": response2.content[0].text},
        {"role": "user", "content": "How does functools.wraps help?"}
    ]
)
```

### 1.2 The Conversation Manager Pattern

A basic conversation manager encapsulates history management:

```python
import anthropic
from dataclasses import dataclass, field


@dataclass
class Conversation:
    """Manages a multi-turn conversation with Claude."""
    system: str = ""
    model: str = "claude-sonnet-4-20250514"
    max_tokens: int = 1024
    messages: list[dict] = field(default_factory=list)
    _client: anthropic.Anthropic = field(
        default_factory=anthropic.Anthropic, repr=False
    )

    def say(self, user_message: str) -> str:
        """Send a message and get a response."""
        self.messages.append({"role": "user", "content": user_message})

        response = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=self.system,
            messages=self.messages
        )

        assistant_message = response.content[0].text
        self.messages.append({"role": "assistant", "content": assistant_message})

        return assistant_message

    def reset(self) -> None:
        """Clear conversation history."""
        self.messages.clear()

    @property
    def turn_count(self) -> int:
        """Number of user-assistant turn pairs."""
        return len(self.messages) // 2

    @property
    def total_chars(self) -> int:
        """Total character count of all messages."""
        return sum(len(m["content"]) for m in self.messages)


# Usage
conv = Conversation(system="You are a helpful math tutor.")
print(conv.say("What is the quadratic formula?"))
print(conv.say("Can you derive it from ax^2 + bx + c = 0?"))
print(f"Turns: {conv.turn_count}, Characters: {conv.total_chars}")
```

### 1.3 Conversation Cost Awareness

Every turn resends the full history, which means costs grow quadratically with conversation length:

```
Turn 1: Send ~100 tokens, receive ~200 tokens
Turn 2: Send ~400 tokens, receive ~200 tokens  (re-sent: 300)
Turn 3: Send ~800 tokens, receive ~200 tokens  (re-sent: 600)
Turn 4: Send ~1200 tokens, receive ~200 tokens (re-sent: 1000)
...
Turn N: Send ~(N*200 + 100) tokens, receive ~200 tokens
```

Total input tokens over N turns: approximately `N^2 * 100` -- quadratic growth. This is why context management is critical for long conversations.

---

## 2. Context Window Management

### 2.1 Understanding Context Windows

Every model has a fixed context window -- the maximum number of tokens it can process in a single request. This window must contain the system prompt, the full message history, and leave room for the response.

| Model | Context Window | Effective (minus system + response) |
|-------|---------------|-------------------------------------|
| Claude 3.5 Sonnet | 200K tokens | ~190K tokens |
| Claude Opus 4 | 200K tokens | ~190K tokens |
| GPT-4o | 128K tokens | ~120K tokens |
| GPT-4 Turbo | 128K tokens | ~120K tokens |

### 2.2 Token Counting

Before managing context, you need to count tokens:

```python
import anthropic


def count_tokens(
    messages: list[dict],
    system: str = "",
    model: str = "claude-sonnet-4-20250514"
) -> int:
    """Count tokens in a conversation using Anthropic's API."""
    client = anthropic.Anthropic()

    response = client.messages.count_tokens(
        model=model,
        system=system,
        messages=messages
    )

    return response.input_tokens


# For OpenAI models, use tiktoken
import tiktoken


def count_tokens_openai(
    messages: list[dict],
    model: str = "gpt-4o"
) -> int:
    """Count tokens for OpenAI models using tiktoken."""
    encoding = tiktoken.encoding_for_model(model)
    total = 0
    for msg in messages:
        # Each message has overhead for role and formatting
        total += 4  # message overhead
        total += len(encoding.encode(msg["content"]))
    total += 2  # reply priming
    return total


# Usage
messages = [
    {"role": "user", "content": "Hello, how are you?"},
    {"role": "assistant", "content": "I'm doing well! How can I help?"},
    {"role": "user", "content": "Tell me about Python generators."}
]

print(f"Anthropic tokens: {count_tokens(messages)}")
print(f"OpenAI tokens: {count_tokens_openai(messages)}")
```

### 2.3 Sliding Window Strategy

The simplest context management strategy: keep the most recent N turns:

```python
from dataclasses import dataclass, field
import anthropic


@dataclass
class SlidingWindowConversation:
    """Conversation with sliding window context management."""
    system: str = ""
    model: str = "claude-sonnet-4-20250514"
    max_tokens: int = 1024
    max_turns: int = 10  # Keep last N turn pairs
    messages: list[dict] = field(default_factory=list)
    _client: anthropic.Anthropic = field(
        default_factory=anthropic.Anthropic, repr=False
    )

    def _trim_history(self) -> list[dict]:
        """Return messages trimmed to max_turns pairs."""
        if len(self.messages) <= self.max_turns * 2:
            return self.messages

        # Keep the last max_turns pairs (user + assistant)
        trimmed = self.messages[-(self.max_turns * 2):]

        # Ensure we start with a user message
        while trimmed and trimmed[0]["role"] != "user":
            trimmed = trimmed[1:]

        return trimmed

    def say(self, user_message: str) -> str:
        """Send a message with sliding window context."""
        self.messages.append({"role": "user", "content": user_message})

        # Use trimmed history for the API call
        active_messages = self._trim_history()

        response = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=self.system,
            messages=active_messages
        )

        assistant_message = response.content[0].text
        self.messages.append({"role": "assistant", "content": assistant_message})
        return assistant_message


# Usage
conv = SlidingWindowConversation(
    system="You are a Python tutor.",
    max_turns=5
)

# After turn 6, turns 1 will be dropped
for i in range(8):
    response = conv.say(f"Question {i+1}: Tell me about Python feature #{i+1}")
    print(f"Turn {i+1}: {response[:80]}...")
    print(f"  Total messages: {len(conv.messages)}, Active: {len(conv._trim_history())}")
```

### 2.4 Token Budget Strategy

Instead of a fixed number of turns, manage by token budget:

```python
import tiktoken
from dataclasses import dataclass, field
import anthropic


@dataclass
class TokenBudgetConversation:
    """Conversation that respects a token budget."""
    system: str = ""
    model: str = "claude-sonnet-4-20250514"
    max_response_tokens: int = 1024
    context_budget: int = 50000  # Max tokens for history
    messages: list[dict] = field(default_factory=list)
    _client: anthropic.Anthropic = field(
        default_factory=anthropic.Anthropic, repr=False
    )

    def _estimate_tokens(self, text: str) -> int:
        """Rough token estimate: ~4 chars per token for English."""
        return len(text) // 4

    def _get_budget_messages(self) -> list[dict]:
        """Select messages that fit within the token budget."""
        total = self._estimate_tokens(self.system)
        selected = []

        # Walk backward from most recent, adding messages until budget exhausted
        for msg in reversed(self.messages):
            msg_tokens = self._estimate_tokens(msg["content"])
            if total + msg_tokens > self.context_budget:
                break
            selected.insert(0, msg)
            total += msg_tokens

        # Ensure we start with a user message
        while selected and selected[0]["role"] != "user":
            selected = selected[1:]

        return selected

    def say(self, user_message: str) -> str:
        """Send a message within the token budget."""
        self.messages.append({"role": "user", "content": user_message})
        active = self._get_budget_messages()

        response = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_response_tokens,
            system=self.system,
            messages=active
        )

        reply = response.content[0].text
        self.messages.append({"role": "assistant", "content": reply})
        return reply
```

---

## 3. Conversation State Tracking

### 3.1 Why Track State Separately?

When old messages drop out of the context window, the model loses information. State tracking extracts and persists key facts so they survive context trimming.

```python
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ConversationState:
    """Tracks structured state extracted from conversation."""
    user_info: dict[str, Any] = field(default_factory=dict)
    task_info: dict[str, Any] = field(default_factory=dict)
    decisions: list[str] = field(default_factory=list)
    open_questions: list[str] = field(default_factory=list)
    topic_history: list[str] = field(default_factory=list)

    def to_context_string(self) -> str:
        """Format state for injection into system prompt or context."""
        parts = []

        if self.user_info:
            info = ", ".join(f"{k}: {v}" for k, v in self.user_info.items())
            parts.append(f"User info: {info}")

        if self.task_info:
            info = ", ".join(f"{k}: {v}" for k, v in self.task_info.items())
            parts.append(f"Current task: {info}")

        if self.decisions:
            decs = "; ".join(self.decisions[-5:])  # Last 5 decisions
            parts.append(f"Decisions made: {decs}")

        if self.open_questions:
            qs = "; ".join(self.open_questions)
            parts.append(f"Open questions: {qs}")

        return "\n".join(parts)


# Usage
state = ConversationState()
state.user_info["name"] = "Alice"
state.user_info["project"] = "E-commerce API"
state.task_info["phase"] = "database design"
state.decisions.append("Use PostgreSQL for main store")
state.decisions.append("Use Redis for caching")
state.open_questions.append("Which ORM to use?")

print(state.to_context_string())
```

### 3.2 Automatic State Extraction

Use the LLM itself to extract state from each turn:

```python
import anthropic
import json
from dataclasses import dataclass, field
from typing import Any


@dataclass
class StatefulConversation:
    """Conversation with automatic state extraction."""
    system: str = ""
    model: str = "claude-sonnet-4-20250514"
    messages: list[dict] = field(default_factory=list)
    state: dict[str, Any] = field(default_factory=dict)
    _client: anthropic.Anthropic = field(
        default_factory=anthropic.Anthropic, repr=False
    )

    def _extract_state_update(
        self, user_msg: str, assistant_msg: str
    ) -> dict:
        """Use LLM to extract state updates from the latest exchange."""
        extraction_prompt = f"""Given this conversation exchange, extract any
new facts, decisions, or state changes as a JSON object.

User said: {user_msg}
Assistant said: {assistant_msg[:500]}

Current state: {json.dumps(self.state, indent=2)}

Return a JSON object with keys to ADD or UPDATE in the state.
Only include keys that changed or are new. Use null to delete a key.
If nothing changed, return {{}}.

Return ONLY the JSON object."""

        response = self._client.messages.create(
            model=self.model,
            max_tokens=512,
            messages=[
                {"role": "user", "content": extraction_prompt},
                {"role": "assistant", "content": "{"}
            ]
        )

        try:
            update = json.loads("{" + response.content[0].text)
            return update
        except json.JSONDecodeError:
            return {}

    def _build_system_with_state(self) -> str:
        """Inject current state into the system prompt."""
        if not self.state:
            return self.system

        state_str = json.dumps(self.state, indent=2)
        return (
            f"{self.system}\n\n"
            f"## Conversation State (persisted facts)\n"
            f"```json\n{state_str}\n```\n"
            f"Use this state to maintain context. Reference it naturally."
        )

    def say(self, user_message: str) -> str:
        """Send a message with automatic state tracking."""
        self.messages.append({"role": "user", "content": user_message})

        response = self._client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=self._build_system_with_state(),
            messages=self.messages[-20:]  # Sliding window of 10 turns
        )

        reply = response.content[0].text
        self.messages.append({"role": "assistant", "content": reply})

        # Extract state updates asynchronously
        update = self._extract_state_update(user_message, reply)
        for key, value in update.items():
            if value is None:
                self.state.pop(key, None)
            else:
                self.state[key] = value

        return reply


# Usage
conv = StatefulConversation(
    system="You are a project planning assistant."
)

conv.say("I'm building a todo app with React and FastAPI.")
conv.say("Let's use PostgreSQL for the database.")
conv.say("Actually, let's keep it simple with SQLite for now.")

print("Extracted state:", json.dumps(conv.state, indent=2))
```

### 3.3 Slot-Based State Tracking

For task-oriented conversations (booking, ordering, support), use a slot-filling pattern:

```python
from dataclasses import dataclass, field
from typing import Optional
import anthropic
import json


@dataclass
class BookingSlots:
    """Slots for a restaurant booking conversation."""
    restaurant: Optional[str] = None
    date: Optional[str] = None
    time: Optional[str] = None
    party_size: Optional[int] = None
    special_requests: Optional[str] = None

    @property
    def missing_slots(self) -> list[str]:
        """Return names of unfilled required slots."""
        required = ["restaurant", "date", "time", "party_size"]
        return [s for s in required if getattr(self, s) is None]

    @property
    def is_complete(self) -> bool:
        return len(self.missing_slots) == 0

    def to_prompt_context(self) -> str:
        lines = ["Current booking details:"]
        for s in ["restaurant", "date", "time", "party_size", "special_requests"]:
            val = getattr(self, s)
            status = val if val is not None else "[not yet provided]"
            lines.append(f"  - {s}: {status}")
        if self.missing_slots:
            lines.append(f"\nStill needed: {', '.join(self.missing_slots)}")
        return "\n".join(lines)


def run_booking_conversation():
    """Demonstrate slot-based conversation tracking."""
    client = anthropic.Anthropic()
    slots = BookingSlots()
    messages = []

    system_base = """You are a restaurant booking assistant.

Your job is to collect booking details from the user. Ask for missing
information one question at a time. Be conversational, not robotic.

When all required fields are filled, confirm the booking details and
ask for final confirmation.

Required: restaurant name, date, time, party size
Optional: special requests (allergies, occasion, seating preference)"""

    tools = [{
        "name": "update_booking",
        "description": "Update booking slots with information from the user",
        "input_schema": {
            "type": "object",
            "properties": {
                "restaurant": {"type": "string"},
                "date": {"type": "string", "description": "YYYY-MM-DD format"},
                "time": {"type": "string", "description": "HH:MM format"},
                "party_size": {"type": "integer", "minimum": 1},
                "special_requests": {"type": "string"}
            }
        }
    }]

    print("Booking Assistant: Hi! I'd love to help you make a reservation. What restaurant are you interested in?\n")

    while True:
        user_input = input("You: ").strip()
        if not user_input or user_input.lower() in ["quit", "exit"]:
            break

        messages.append({"role": "user", "content": user_input})

        system = f"{system_base}\n\n{slots.to_prompt_context()}"

        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=512,
            system=system,
            tools=tools,
            messages=messages
        )

        # Process tool calls to update slots
        assistant_text = ""
        for block in response.content:
            if block.type == "tool_use":
                for key, value in block.input.items():
                    if value is not None:
                        setattr(slots, key, value)
            elif block.type == "text":
                assistant_text = block.text

        # If tool was called, get the text response
        if response.stop_reason == "tool_use":
            # Send tool result and get natural language response
            messages.append({"role": "assistant", "content": response.content})
            messages.append({
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": next(
                        b.id for b in response.content if b.type == "tool_use"
                    ),
                    "content": "Booking updated successfully."
                }]
            })

            followup = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=512,
                system=system,
                tools=tools,
                messages=messages
            )
            assistant_text = next(
                (b.text for b in followup.content if b.type == "text"),
                ""
            )
            messages.append({"role": "assistant", "content": assistant_text})
        else:
            messages.append({"role": "assistant", "content": assistant_text})

        print(f"\nAssistant: {assistant_text}\n")

        if slots.is_complete:
            print("--- Booking Complete ---")
            print(slots.to_prompt_context())
            break
```

---

## 4. Memory Injection Patterns

### 4.1 Overview of Memory Patterns

When conversations exceed the context window, you need a memory system. There are three primary patterns:

| Pattern | Mechanism | Best For |
|---------|-----------|----------|
| **Summary memory** | Compress old turns into a summary | General conversations |
| **Sliding window + summary** | Recent turns verbatim + older turns summarized | Balanced approach |
| **Entity memory** | Extract and maintain a knowledge graph of entities | Fact-heavy conversations |

### 4.2 Summary Memory

Periodically summarize old messages and inject the summary:

```python
import anthropic
from dataclasses import dataclass, field


@dataclass
class SummaryMemoryConversation:
    """Conversation with summary-based memory."""
    system: str = ""
    model: str = "claude-sonnet-4-20250514"
    messages: list[dict] = field(default_factory=list)
    summary: str = ""
    summarize_every: int = 6  # Summarize after every N turn pairs
    _client: anthropic.Anthropic = field(
        default_factory=anthropic.Anthropic, repr=False
    )

    def _generate_summary(self, messages_to_summarize: list[dict]) -> str:
        """Summarize a batch of messages."""
        conversation_text = ""
        for msg in messages_to_summarize:
            role = msg["role"].capitalize()
            content = msg["content"] if isinstance(msg["content"], str) else str(msg["content"])
            conversation_text += f"{role}: {content}\n\n"

        prompt = (
            "Summarize this conversation segment concisely. "
            "Preserve: key facts, decisions, user preferences, "
            "unresolved questions, and any code/data discussed. "
            "Use bullet points. Be specific.\n\n"
            f"Previous summary: {self.summary or '(none)'}\n\n"
            f"New conversation:\n{conversation_text}"
        )

        response = self._client.messages.create(
            model=self.model,
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}]
        )

        return response.content[0].text

    def _build_context(self) -> tuple[str, list[dict]]:
        """Build system prompt with summary and trimmed messages."""
        system = self.system
        if self.summary:
            system += (
                f"\n\n## Conversation History Summary\n"
                f"{self.summary}\n\n"
                f"(Use this summary to maintain context about earlier discussion)"
            )

        # Keep recent messages (last summarize_every * 2 messages)
        recent = self.messages[-(self.summarize_every * 2):]

        # Ensure starts with user message
        while recent and recent[0]["role"] != "user":
            recent = recent[1:]

        return system, recent

    def say(self, user_message: str) -> str:
        """Send a message with summary-based memory."""
        self.messages.append({"role": "user", "content": user_message})

        # Check if we need to summarize
        recent_count = len(self.messages)
        if recent_count > self.summarize_every * 2 and recent_count % (self.summarize_every * 2) == 1:
            # Summarize older messages
            older = self.messages[:-(self.summarize_every * 2)]
            self.summary = self._generate_summary(older)

        system, active_messages = self._build_context()

        response = self._client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=system,
            messages=active_messages
        )

        reply = response.content[0].text
        self.messages.append({"role": "assistant", "content": reply})
        return reply


# Usage
conv = SummaryMemoryConversation(
    system="You are a software design consultant.",
    summarize_every=4  # Summarize after 4 turn pairs
)

# Simulate a long conversation
questions = [
    "I'm building an e-commerce platform. What architecture should I use?",
    "Let's go with microservices. What services do I need?",
    "How should the product catalog service work?",
    "What about the order processing service?",
    "Now tell me about payment integration.",
    "What database should each service use?",
    "How do the services communicate?",
    "What about monitoring and logging?",
]

for q in questions:
    response = conv.say(q)
    print(f"Q: {q[:60]}...")
    print(f"A: {response[:100]}...")
    print(f"  [Messages: {len(conv.messages)}, Summary: {len(conv.summary)} chars]")
    print()
```

### 4.3 Entity Memory

Extract and maintain entities (people, projects, decisions) across the conversation:

```python
import anthropic
import json
from dataclasses import dataclass, field


@dataclass
class Entity:
    """A tracked entity in the conversation."""
    name: str
    entity_type: str  # person, project, technology, decision, etc.
    attributes: dict = field(default_factory=dict)
    first_mentioned: int = 0  # Turn number
    last_mentioned: int = 0

    def to_string(self) -> str:
        attrs = ", ".join(f"{k}: {v}" for k, v in self.attributes.items())
        return f"{self.name} ({self.entity_type}): {attrs}"


@dataclass
class EntityMemory:
    """Entity-based conversation memory."""
    entities: dict[str, Entity] = field(default_factory=dict)

    def update_from_extraction(self, extracted: dict, turn: int) -> None:
        """Update entities from LLM extraction results."""
        for entity_data in extracted.get("entities", []):
            name = entity_data["name"]
            if name in self.entities:
                # Update existing entity
                self.entities[name].attributes.update(
                    entity_data.get("attributes", {})
                )
                self.entities[name].last_mentioned = turn
            else:
                # New entity
                self.entities[name] = Entity(
                    name=name,
                    entity_type=entity_data.get("type", "unknown"),
                    attributes=entity_data.get("attributes", {}),
                    first_mentioned=turn,
                    last_mentioned=turn
                )

    def to_context_string(self) -> str:
        """Format entities for prompt injection."""
        if not self.entities:
            return ""

        lines = ["## Known Entities"]
        for entity in sorted(
            self.entities.values(),
            key=lambda e: e.last_mentioned,
            reverse=True
        ):
            lines.append(f"- {entity.to_string()}")

        return "\n".join(lines)


def extract_entities(
    client: anthropic.Anthropic,
    user_msg: str,
    assistant_msg: str,
    existing_entities: dict
) -> dict:
    """Use LLM to extract entities from a conversation turn."""
    existing = json.dumps(
        {name: e.to_string() for name, e in existing_entities.items()},
        indent=2
    )

    prompt = f"""Extract entities from this conversation exchange.

Existing entities: {existing}

User: {user_msg}
Assistant: {assistant_msg[:500]}

Return JSON with this structure:
{{
  "entities": [
    {{
      "name": "entity name",
      "type": "person|project|technology|decision|location|organization",
      "attributes": {{"key": "value"}}
    }}
  ]
}}

Rules:
- Include both NEW entities and UPDATED existing ones
- Only include entities that are meaningfully discussed
- Attributes should capture specific facts, not general descriptions
- Return {{"entities": []}} if no notable entities found"""

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=512,
        messages=[
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": "{"}
        ]
    )

    try:
        return json.loads("{" + response.content[0].text)
    except json.JSONDecodeError:
        return {"entities": []}
```

### 4.4 Hybrid Memory: Sliding Window + Summary + Entities

The most robust approach combines all three patterns:

```python
import anthropic
from dataclasses import dataclass, field


@dataclass
class HybridMemoryConversation:
    """Conversation with hybrid memory management."""
    system: str = ""
    model: str = "claude-sonnet-4-20250514"
    messages: list[dict] = field(default_factory=list)
    summary: str = ""
    entity_memory: EntityMemory = field(default_factory=EntityMemory)
    max_recent_turns: int = 6
    _client: anthropic.Anthropic = field(
        default_factory=anthropic.Anthropic, repr=False
    )
    _turn_count: int = 0

    def _build_augmented_system(self) -> str:
        """Build system prompt with memory context."""
        parts = [self.system]

        # Add entity memory
        entity_ctx = self.entity_memory.to_context_string()
        if entity_ctx:
            parts.append(entity_ctx)

        # Add conversation summary
        if self.summary:
            parts.append(
                f"## Earlier Conversation Summary\n{self.summary}"
            )

        return "\n\n".join(parts)

    def say(self, user_message: str) -> str:
        """Send a message with hybrid memory."""
        self.messages.append({"role": "user", "content": user_message})
        self._turn_count += 1

        # Get recent messages for context
        recent = self.messages[-(self.max_recent_turns * 2):]
        while recent and recent[0]["role"] != "user":
            recent = recent[1:]

        # API call
        response = self._client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=self._build_augmented_system(),
            messages=recent
        )

        reply = response.content[0].text
        self.messages.append({"role": "assistant", "content": reply})

        # Update entity memory
        extracted = extract_entities(
            self._client, user_message, reply,
            self.entity_memory.entities
        )
        self.entity_memory.update_from_extraction(
            extracted, self._turn_count
        )

        # Update summary if we have enough old messages
        if len(self.messages) > self.max_recent_turns * 2 + 4:
            old = self.messages[:-(self.max_recent_turns * 2)]
            self.summary = self._summarize(old)

        return reply

    def _summarize(self, messages: list[dict]) -> str:
        """Generate a summary of older messages."""
        text = "\n".join(
            f"{m['role']}: {m['content'][:200]}" for m in messages
        )
        response = self._client.messages.create(
            model=self.model,
            max_tokens=300,
            messages=[{
                "role": "user",
                "content": f"Summarize this conversation concisely:\n\n{text}"
            }]
        )
        return response.content[0].text
```

---

## 5. Turn-Taking and Conversation Steering

### 5.1 Proactive Conversation Steering

Sometimes the assistant should guide the conversation rather than passively answering:

```python
STEERING_SYSTEM = """You are a requirements gathering assistant for software projects.

## Conversation Strategy
You are NOT a passive Q&A bot. You LEAD the conversation through these phases:

### Phase 1: Problem Understanding (turns 1-3)
- Ask about the problem the user is trying to solve
- Understand who the users/stakeholders are
- Clarify the scale and constraints

### Phase 2: Requirements Elicitation (turns 4-8)
- Walk through functional requirements systematically
- Ask about non-functional requirements (performance, security, scalability)
- Identify integrations and dependencies

### Phase 3: Prioritization (turns 9-11)
- Help categorize into must-have / nice-to-have / future
- Identify potential conflicts or trade-offs
- Suggest phased delivery

### Phase 4: Summary (turns 12+)
- Produce a structured requirements document
- Highlight gaps and open questions
- Suggest next steps

## Rules
- Ask ONE question at a time (don't overwhelm)
- Acknowledge the user's answer before asking the next question
- If the user goes off-topic, gently steer back
- Track which phase you're in based on the conversation history"""
```

### 5.2 Guided Follow-Up Questions

```python
import anthropic


def ask_with_followup(
    question: str,
    max_followups: int = 3
) -> dict:
    """Ask a question and automatically probe for more detail."""
    client = anthropic.Anthropic()

    system = """You are an interviewer gathering detailed information.
After each user response, do ONE of:
1. Ask a specific follow-up to deepen understanding
2. Ask about a related aspect not yet covered
3. Summarize what you've learned (only when you have enough detail)

When summarizing, start with "SUMMARY:" to signal completion."""

    messages = [{"role": "user", "content": question}]
    collected_info = []

    for i in range(max_followups + 1):
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=512,
            system=system,
            messages=messages
        )

        reply = response.content[0].text
        messages.append({"role": "assistant", "content": reply})

        if reply.startswith("SUMMARY:"):
            return {
                "summary": reply[8:].strip(),
                "turns": i + 1,
                "messages": messages
            }

        # Simulate user response (in practice, this would be real user input)
        print(f"Follow-up {i+1}: {reply}")
        user_response = input("Your answer: ").strip()
        if not user_response:
            break
        messages.append({"role": "user", "content": user_response})
        collected_info.append(user_response)

    return {
        "summary": "Conversation ended before summary",
        "turns": len(messages) // 2,
        "collected_info": collected_info
    }
```

### 5.3 Conversation Pacing

Control how much information the model delivers per turn:

```python
PACING_SYSTEM = """You are a coding tutor teaching Python.

## Pacing Rules

1. ONE concept per response. Don't jump ahead.
2. After explaining a concept, ask the student to try it:
   "Try writing a function that [exercise]. Let me know when you're ready."
3. Wait for the student's attempt before giving the next concept.
4. If the student's code has errors:
   - First: give a HINT, not the answer
   - Second attempt: give a more specific hint
   - Third attempt: show the solution with explanation
5. If the student says "skip" or "next": move on without judgment.
6. Track progress: "We've covered X and Y. Next up: Z."

## Concept Sequence
1. Variables and types
2. If/else statements
3. For loops
4. While loops
5. Functions
6. Lists
7. Dictionaries

Do NOT skip ahead in the sequence unless the student demonstrates mastery."""
```

---

## 6. Handling Topic Switches

### 6.1 Detecting Topic Switches

Users naturally change topics mid-conversation. The assistant should handle this gracefully:

```python
import anthropic
import json


def detect_topic_switch(
    client: anthropic.Anthropic,
    messages: list[dict],
    new_message: str
) -> dict:
    """Detect if a new message represents a topic switch."""
    recent_context = "\n".join(
        f"{m['role']}: {m['content'][:100]}"
        for m in messages[-6:]
    )

    tools = [{
        "name": "classify_topic",
        "description": "Classify whether the new message continues the current topic or switches",
        "input_schema": {
            "type": "object",
            "properties": {
                "is_topic_switch": {"type": "boolean"},
                "previous_topic": {"type": "string"},
                "new_topic": {"type": "string"},
                "switch_type": {
                    "type": "string",
                    "enum": ["none", "gradual", "abrupt", "return_to_previous"]
                }
            },
            "required": ["is_topic_switch", "switch_type"]
        }
    }]

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=256,
        tools=tools,
        tool_choice={"type": "tool", "name": "classify_topic"},
        messages=[{
            "role": "user",
            "content": (
                f"Recent conversation:\n{recent_context}\n\n"
                f"New message: {new_message}\n\n"
                f"Is this a topic switch?"
            )
        }]
    )

    for block in response.content:
        if block.type == "tool_use":
            return block.input

    return {"is_topic_switch": False, "switch_type": "none"}
```

### 6.2 Topic Switch Strategies

```python
TOPIC_AWARE_SYSTEM = """You are a versatile assistant that handles topic switches gracefully.

## When the user switches topics:

### Abrupt switch (unrelated new topic):
- Acknowledge the switch briefly: "Switching gears —"
- Answer the new question fully
- Do NOT reference the old topic unless asked

### Gradual switch (related topic):
- Bridge naturally: "Building on what we discussed about X..."
- Connect the topics if there's a meaningful link

### Return to previous topic:
- Recall the context: "Going back to our earlier discussion about X..."
- Pick up where you left off
- Reference specific details from the earlier discussion

### Ambiguous (could be either):
- Ask: "Just to make sure — are you asking about this in the context of
  [previous topic], or is this a new question?"

## Topic State
Mentally track the last 3 topics discussed. If the user references
"the earlier thing" or "what we talked about before," use this mental
log to resolve the reference."""
```

### 6.3 Topic Stack Management

```python
from dataclasses import dataclass, field


@dataclass
class TopicStack:
    """Manage a stack of conversation topics."""
    topics: list[dict] = field(default_factory=list)

    def push(self, topic: str, context: str = "") -> None:
        """Push a new topic onto the stack."""
        self.topics.append({
            "topic": topic,
            "context": context,
            "turn_started": len(self.topics)
        })

    def pop(self) -> dict | None:
        """Pop the current topic (return to previous)."""
        if self.topics:
            return self.topics.pop()
        return None

    @property
    def current_topic(self) -> str | None:
        return self.topics[-1]["topic"] if self.topics else None

    def to_context_string(self) -> str:
        if not self.topics:
            return "No active topic."
        lines = ["Topic stack (most recent first):"]
        for t in reversed(self.topics):
            lines.append(f"  - {t['topic']}: {t['context'][:80]}")
        return "\n".join(lines)
```

---

## 7. Error Recovery in Conversations

### 7.1 Types of Conversational Errors

| Error Type | Cause | Recovery Strategy |
|-----------|-------|-------------------|
| Misunderstanding | Ambiguous user input | Ask for clarification |
| Hallucination | Model generates false info | Correct when caught |
| Context loss | Important info dropped from window | Re-inject from memory |
| Instruction drift | Model gradually ignores system prompt | Reinforce constraints |
| Loop | Model repeats itself | Detect and break loop |

### 7.2 Graceful Misunderstanding Recovery

```python
RECOVERY_SYSTEM = """You are a technical assistant.

## Error Recovery Protocol

### If you're unsure what the user means:
"I want to make sure I understand correctly. Are you asking about:
a) [interpretation 1]
b) [interpretation 2]
Which is closer to what you need?"

### If the user says you're wrong:
1. Do NOT argue or defend your previous answer
2. Say: "Thank you for the correction. Let me reconsider."
3. Ask what specifically was wrong
4. Provide a corrected response

### If you realize your own mistake:
1. Acknowledge immediately: "I need to correct something I said earlier."
2. Clearly state what was wrong and what is right
3. Don't over-apologize — just fix it

### If you don't know:
"I'm not confident enough to give you a reliable answer on [topic].
I'd recommend checking [specific resource] for accurate information."

NEVER double down on an incorrect answer to save face."""
```

### 7.3 Loop Detection and Breaking

```python
from dataclasses import dataclass, field
import hashlib


@dataclass
class LoopDetector:
    """Detect when a conversation enters a repetitive loop."""
    response_hashes: list[str] = field(default_factory=list)
    response_snippets: list[str] = field(default_factory=list)
    loop_threshold: int = 3  # Consecutive similar responses

    def _hash_response(self, text: str) -> str:
        """Create a normalized hash of a response."""
        # Normalize: lowercase, strip whitespace, remove punctuation
        normalized = text.lower().strip()
        normalized = "".join(c for c in normalized if c.isalnum() or c.isspace())
        # Use first 200 chars to avoid hash being too specific
        return hashlib.md5(normalized[:200].encode()).hexdigest()

    def check(self, response: str) -> bool:
        """Check if the response indicates a loop. Returns True if looping."""
        current_hash = self._hash_response(response)
        self.response_hashes.append(current_hash)
        self.response_snippets.append(response[:100])

        if len(self.response_hashes) < self.loop_threshold:
            return False

        # Check if last N responses are similar
        recent = self.response_hashes[-self.loop_threshold:]
        if len(set(recent)) == 1:
            return True

        # Check for alternating pattern (A-B-A-B)
        if len(self.response_hashes) >= 4:
            h = self.response_hashes
            if h[-1] == h[-3] and h[-2] == h[-4]:
                return True

        return False

    def get_loop_breaker_message(self) -> str:
        """Generate a message to break the conversation loop."""
        return (
            "I notice I may be repeating myself. Let me approach this "
            "differently. Could you rephrase your question or tell me "
            "specifically what part isn't clear? I want to make sure "
            "I'm actually addressing what you need."
        )
```

### 7.4 Context Loss Recovery

```python
import anthropic


def recover_lost_context(
    client: anthropic.Anthropic,
    current_messages: list[dict],
    full_history: list[dict],
    user_reference: str
) -> str:
    """When a user references something outside the current context window."""
    # Search full history for relevant context
    search_prompt = f"""The user said: "{user_reference}"

This seems to reference earlier conversation that isn't in the current
context window. Search these older messages for relevant context:

{chr(10).join(f"{m['role']}: {m['content'][:200]}" for m in full_history[:-len(current_messages)])}

Return the most relevant excerpt (1-3 sentences) that the user is likely
referring to. If nothing matches, say "NO_MATCH"."""

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=256,
        messages=[{"role": "user", "content": search_prompt}]
    )

    recovered = response.content[0].text
    if "NO_MATCH" in recovered:
        return (
            "I apologize, but I don't have that earlier context available "
            "right now. Could you remind me what you're referring to?"
        )

    return f"[Recovered context: {recovered}]\n\n"
```

---

## 8. Conversation Branching

### 8.1 What is Conversation Branching?

Conversation branching allows you to explore multiple paths from a single point in the conversation, like a "what if" scenario. This is useful for:

- Comparing different approaches to a problem
- Testing how the model responds to different follow-ups
- A/B testing conversation strategies

### 8.2 Implementing Conversation Branches

```python
import anthropic
import copy
from dataclasses import dataclass, field


@dataclass
class ConversationTree:
    """Manage a tree of conversation branches."""
    system: str = ""
    model: str = "claude-sonnet-4-20250514"
    branches: dict[str, list[dict]] = field(default_factory=lambda: {"main": []})
    current_branch: str = "main"
    _client: anthropic.Anthropic = field(
        default_factory=anthropic.Anthropic, repr=False
    )

    def say(self, user_message: str) -> str:
        """Send a message on the current branch."""
        messages = self.branches[self.current_branch]
        messages.append({"role": "user", "content": user_message})

        response = self._client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=self.system,
            messages=messages
        )

        reply = response.content[0].text
        messages.append({"role": "assistant", "content": reply})
        return reply

    def create_branch(self, branch_name: str, from_branch: str = None) -> None:
        """Create a new branch from the current or specified branch."""
        source = from_branch or self.current_branch
        if source not in self.branches:
            raise ValueError(f"Source branch '{source}' not found")

        # Deep copy the source branch's messages
        self.branches[branch_name] = copy.deepcopy(self.branches[source])

    def switch_branch(self, branch_name: str) -> None:
        """Switch to a different branch."""
        if branch_name not in self.branches:
            raise ValueError(f"Branch '{branch_name}' not found")
        self.current_branch = branch_name

    def list_branches(self) -> dict[str, int]:
        """List all branches with their message counts."""
        return {
            name: len(msgs) for name, msgs in self.branches.items()
        }

    def compare_branches(self, branch_a: str, branch_b: str) -> dict:
        """Compare the last response of two branches."""
        last_a = self.branches[branch_a][-1]["content"] if self.branches[branch_a] else ""
        last_b = self.branches[branch_b][-1]["content"] if self.branches[branch_b] else ""
        return {
            "branch_a": {"name": branch_a, "last_response": last_a[:200]},
            "branch_b": {"name": branch_b, "last_response": last_b[:200]}
        }


# Usage
tree = ConversationTree(
    system="You are an architecture advisor."
)

# Main conversation
tree.say("I need to build a real-time chat application.")
tree.say("What database should I use?")

# Branch to explore PostgreSQL path
tree.create_branch("postgres_path")
tree.switch_branch("postgres_path")
pg_response = tree.say("Tell me more about using PostgreSQL with pub/sub for this.")

# Branch to explore MongoDB path
tree.create_branch("mongo_path", from_branch="main")
tree.switch_branch("mongo_path")
mongo_response = tree.say("What about using MongoDB with change streams?")

# Compare the two paths
comparison = tree.compare_branches("postgres_path", "mongo_path")
print("PostgreSQL path:", comparison["branch_a"]["last_response"])
print("MongoDB path:", comparison["branch_b"]["last_response"])
```

### 8.3 Checkpoint and Restore

```python
import json
import copy
from datetime import datetime


class ConversationCheckpoint:
    """Save and restore conversation states."""

    def __init__(self):
        self.checkpoints: dict[str, dict] = {}

    def save(
        self,
        name: str,
        messages: list[dict],
        metadata: dict = None
    ) -> None:
        """Save a conversation checkpoint."""
        self.checkpoints[name] = {
            "messages": copy.deepcopy(messages),
            "metadata": metadata or {},
            "timestamp": datetime.now().isoformat(),
            "turn_count": len(messages) // 2
        }

    def restore(self, name: str) -> list[dict]:
        """Restore a conversation from checkpoint."""
        if name not in self.checkpoints:
            raise ValueError(f"Checkpoint '{name}' not found")
        return copy.deepcopy(self.checkpoints[name]["messages"])

    def list_checkpoints(self) -> list[dict]:
        """List all checkpoints with metadata."""
        return [
            {
                "name": name,
                "turns": cp["turn_count"],
                "time": cp["timestamp"],
                **cp["metadata"]
            }
            for name, cp in self.checkpoints.items()
        ]

    def export_json(self, name: str) -> str:
        """Export a checkpoint as JSON for storage."""
        return json.dumps(self.checkpoints[name], indent=2)

    def import_json(self, name: str, json_str: str) -> None:
        """Import a checkpoint from JSON."""
        self.checkpoints[name] = json.loads(json_str)
```

---

## 9. Long Conversations and Context Compression

### 9.1 The Long Conversation Challenge

As conversations grow, several problems compound:

1. **Token limits**: Eventually the full history exceeds the context window
2. **Cost**: Input tokens grow quadratically (resending old turns)
3. **Attention degradation**: Models struggle to attend to information in the middle of very long contexts
4. **Latency**: More input tokens means slower time-to-first-token

### 9.2 Progressive Summarization

Compress older parts of the conversation in tiers:

```python
import anthropic
from dataclasses import dataclass, field


@dataclass
class ProgressiveSummary:
    """Multi-tier progressive summarization for long conversations."""
    tier1_summary: str = ""  # Very compressed, high-level
    tier2_summary: str = ""  # Moderate detail
    recent_messages: list[dict] = field(default_factory=list)
    all_messages: list[dict] = field(default_factory=list)
    tier2_window: int = 10  # Messages in tier 2
    recent_window: int = 6  # Messages kept verbatim
    _client: anthropic.Anthropic = field(
        default_factory=anthropic.Anthropic, repr=False
    )

    def _compress(self, messages: list[dict], level: str) -> str:
        """Compress messages at a given detail level."""
        text = "\n".join(
            f"{m['role']}: {m['content'][:300]}" for m in messages
        )

        detail_instruction = {
            "high": "Keep all key facts, decisions, code snippets, and specifics.",
            "low": "Keep only the most important facts and decisions. Max 3 bullet points."
        }

        response = self._client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=400,
            messages=[{
                "role": "user",
                "content": (
                    f"Summarize this conversation. "
                    f"{detail_instruction[level]}\n\n"
                    f"Previous summary context: {self.tier1_summary}\n\n"
                    f"Conversation:\n{text}"
                )
            }]
        )
        return response.content[0].text

    def add_turn(self, user_msg: str, assistant_msg: str) -> None:
        """Add a new turn and manage compression tiers."""
        self.all_messages.append({"role": "user", "content": user_msg})
        self.all_messages.append({"role": "assistant", "content": assistant_msg})

        total = len(self.all_messages)
        total_window = self.tier2_window + self.recent_window

        if total > total_window * 2:
            # Compress old tier2 into tier1
            old_tier2_start = max(0, total - total_window * 2)
            old_tier2_end = total - self.recent_window * 2
            old_tier2 = self.all_messages[old_tier2_start:old_tier2_end]
            if old_tier2:
                self.tier1_summary = self._compress(old_tier2, "low")

        if total > self.recent_window * 2:
            # Create tier2 summary of middle messages
            tier2_start = max(0, total - (self.tier2_window + self.recent_window) * 2)
            tier2_end = total - self.recent_window * 2
            tier2_messages = self.all_messages[tier2_start:tier2_end]
            if tier2_messages:
                self.tier2_summary = self._compress(tier2_messages, "high")

    def get_context(self) -> tuple[str, list[dict]]:
        """Get the current context for an API call."""
        context_parts = []
        if self.tier1_summary:
            context_parts.append(
                f"[Earlier conversation (compressed)]\n{self.tier1_summary}"
            )
        if self.tier2_summary:
            context_parts.append(
                f"[Recent conversation (summarized)]\n{self.tier2_summary}"
            )

        context = "\n\n".join(context_parts)

        # Recent messages (verbatim)
        recent = self.all_messages[-(self.recent_window * 2):]
        while recent and recent[0]["role"] != "user":
            recent = recent[1:]

        return context, recent
```

### 9.3 Selective Context Loading

Instead of sending all history, load only relevant parts based on the current query:

```python
import anthropic
from typing import Optional


def find_relevant_context(
    client: anthropic.Anthropic,
    query: str,
    all_messages: list[dict],
    max_relevant: int = 4
) -> list[dict]:
    """Find the most relevant historical messages for the current query."""
    if len(all_messages) <= max_relevant * 2:
        return all_messages

    # Create a search index of message pairs
    pairs = []
    for i in range(0, len(all_messages) - 1, 2):
        if i + 1 < len(all_messages):
            pairs.append({
                "index": i,
                "user": all_messages[i]["content"][:200],
                "assistant": all_messages[i + 1]["content"][:200]
            })

    # Ask the model to rank relevance
    pairs_text = "\n".join(
        f"[{p['index']}] User: {p['user'][:100]}"
        for p in pairs
    )

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=256,
        messages=[{
            "role": "user",
            "content": (
                f"Given this new query: \"{query}\"\n\n"
                f"Which of these earlier conversation turns are most relevant? "
                f"Return the indices as a JSON array, most relevant first. "
                f"Max {max_relevant} items.\n\n{pairs_text}"
            )
        },
        {"role": "assistant", "content": "["}
        ]
    )

    import json
    try:
        indices = json.loads("[" + response.content[0].text)
        relevant = []
        for idx in indices[:max_relevant]:
            if idx < len(all_messages) - 1:
                relevant.append(all_messages[idx])
                relevant.append(all_messages[idx + 1])
        return relevant
    except (json.JSONDecodeError, IndexError):
        # Fallback to most recent
        return all_messages[-(max_relevant * 2):]
```

### 9.4 Conversation Compaction

When a conversation gets very long, offer to compact it:

```python
import anthropic
import json


def compact_conversation(
    client: anthropic.Anthropic,
    messages: list[dict],
    system: str
) -> list[dict]:
    """Compact a long conversation into a shorter equivalent."""
    full_text = "\n".join(
        f"{m['role'].upper()}: {m['content']}" for m in messages
    )

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2048,
        messages=[{
            "role": "user",
            "content": (
                "Rewrite this conversation as a shorter version that preserves "
                "all essential information: facts established, decisions made, "
                "code written, preferences expressed. Remove small talk, "
                "clarifications that were resolved, and redundant explanations.\n\n"
                "Return the compacted conversation as a JSON array of "
                '{"role": "user"|"assistant", "content": "..."} objects.\n\n'
                f"Conversation:\n{full_text}"
            )
        }]
    )

    try:
        # Extract JSON from response
        import re
        json_match = re.search(r"\[.*\]", response.content[0].text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
    except (json.JSONDecodeError, AttributeError):
        pass

    # Fallback: return last 10 messages
    return messages[-10:]
```

---

## 10. Designing Conversation Flows

### 10.1 Conversation Flow Patterns

Production conversations often follow designed flows rather than freeform chat:

```python
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Callable, Optional
import anthropic


class FlowState(Enum):
    GREETING = auto()
    PROBLEM_IDENTIFICATION = auto()
    INFORMATION_GATHERING = auto()
    SOLUTION_PROPOSAL = auto()
    CONFIRMATION = auto()
    RESOLUTION = auto()
    FEEDBACK = auto()
    END = auto()


@dataclass
class ConversationFlow:
    """State-machine based conversation flow."""
    current_state: FlowState = FlowState.GREETING
    state_data: dict = field(default_factory=dict)
    transitions: dict[FlowState, dict] = field(default_factory=dict)
    messages: list[dict] = field(default_factory=list)
    _client: anthropic.Anthropic = field(
        default_factory=anthropic.Anthropic, repr=False
    )

    def define_transitions(self) -> None:
        """Define valid state transitions and their system prompts."""
        self.transitions = {
            FlowState.GREETING: {
                "system": (
                    "Greet the user warmly. Ask how you can help today. "
                    "Keep it brief (1-2 sentences)."
                ),
                "next_states": [FlowState.PROBLEM_IDENTIFICATION],
                "auto_advance": True  # Always advance after greeting
            },
            FlowState.PROBLEM_IDENTIFICATION: {
                "system": (
                    "Identify the user's core problem. Ask clarifying questions "
                    "if needed. Once you understand the problem, summarize it "
                    "back to the user. End with: 'Is that right?'"
                ),
                "next_states": [
                    FlowState.INFORMATION_GATHERING,
                    FlowState.SOLUTION_PROPOSAL
                ],
                "advance_condition": lambda data: "problem" in data
            },
            FlowState.INFORMATION_GATHERING: {
                "system": (
                    "Gather additional details needed to solve the problem. "
                    "Ask one specific question at a time. Track what you've "
                    "learned."
                ),
                "next_states": [FlowState.SOLUTION_PROPOSAL],
                "advance_condition": lambda data: data.get("info_complete", False)
            },
            FlowState.SOLUTION_PROPOSAL: {
                "system": (
                    "Propose a solution based on the gathered information. "
                    "Be specific and actionable. Ask if the user would like "
                    "to proceed."
                ),
                "next_states": [
                    FlowState.CONFIRMATION,
                    FlowState.INFORMATION_GATHERING  # Back if rejected
                ],
                "advance_condition": lambda data: data.get("solution_accepted")
            },
            FlowState.CONFIRMATION: {
                "system": (
                    "Confirm the solution with the user. Summarize what "
                    "will happen. Ask for final confirmation."
                ),
                "next_states": [FlowState.RESOLUTION],
                "advance_condition": lambda data: data.get("confirmed", False)
            },
            FlowState.RESOLUTION: {
                "system": (
                    "Execute the resolution. Explain what was done. "
                    "Provide any relevant details or next steps."
                ),
                "next_states": [FlowState.FEEDBACK],
                "auto_advance": True
            },
            FlowState.FEEDBACK: {
                "system": (
                    "Ask if there's anything else you can help with. "
                    "Thank the user for their time."
                ),
                "next_states": [
                    FlowState.PROBLEM_IDENTIFICATION,  # New problem
                    FlowState.END
                ],
                "advance_condition": lambda data: data.get("all_done", False)
            }
        }

    def get_current_system(self) -> str:
        """Get the system prompt for the current state."""
        transition = self.transitions.get(self.current_state, {})
        state_system = transition.get("system", "")
        return (
            f"## Current Phase: {self.current_state.name}\n"
            f"{state_system}\n\n"
            f"## Gathered Information\n"
            f"{json.dumps(self.state_data, indent=2)}"
        )

    def advance_state(self, new_state: FlowState) -> None:
        """Transition to a new state."""
        transition = self.transitions.get(self.current_state, {})
        if new_state in transition.get("next_states", []):
            self.current_state = new_state
        else:
            raise ValueError(
                f"Invalid transition: {self.current_state} -> {new_state}"
            )


import json
```

### 10.2 Guided Task Completion

```python
import anthropic
import json


def run_guided_task(
    task_description: str,
    steps: list[dict]
) -> dict:
    """Run a guided multi-step task with the user."""
    client = anthropic.Anthropic()
    messages = []
    results = {}

    for i, step in enumerate(steps):
        step_system = f"""You are guiding the user through step {i+1} of {len(steps)}.

Current step: {step['name']}
Instructions: {step['prompt']}
What to collect: {step.get('collect', 'user confirmation')}

Previous steps completed:
{json.dumps(results, indent=2)}

Guide the user through this step. When the step is complete,
explicitly say "STEP COMPLETE" at the end of your response."""

        # Initial prompt for this step
        if not messages:
            messages.append({
                "role": "user",
                "content": f"I need help with: {task_description}"
            })
        else:
            messages.append({
                "role": "user",
                "content": f"Ready for the next step."
            })

        step_complete = False
        while not step_complete:
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1024,
                system=step_system,
                messages=messages
            )

            reply = response.content[0].text
            messages.append({"role": "assistant", "content": reply})

            if "STEP COMPLETE" in reply:
                step_complete = True
                results[step['name']] = {
                    "status": "complete",
                    "output": reply
                }
            else:
                # Get user input for this step
                user_input = input(f"\n[Step {i+1}] Your response: ").strip()
                if not user_input:
                    break
                messages.append({"role": "user", "content": user_input})

    return results


# Usage
steps = [
    {
        "name": "requirements",
        "prompt": "Help the user define their project requirements",
        "collect": "list of requirements"
    },
    {
        "name": "architecture",
        "prompt": "Based on requirements, propose an architecture",
        "collect": "architecture decision"
    },
    {
        "name": "implementation_plan",
        "prompt": "Create a phased implementation plan",
        "collect": "plan with milestones"
    }
]

# results = run_guided_task("Build a REST API for a blog", steps)
```

### 10.3 Conversation Analytics

Track and analyze conversation patterns:

```python
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class ConversationAnalytics:
    """Track analytics for a conversation."""
    start_time: datetime = field(default_factory=datetime.now)
    turn_timestamps: list[datetime] = field(default_factory=list)
    user_message_lengths: list[int] = field(default_factory=list)
    assistant_message_lengths: list[int] = field(default_factory=list)
    topic_switches: int = 0
    error_recoveries: int = 0
    clarification_requests: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0

    def record_turn(
        self,
        user_msg: str,
        assistant_msg: str,
        input_tokens: int,
        output_tokens: int
    ) -> None:
        """Record metrics for a conversation turn."""
        self.turn_timestamps.append(datetime.now())
        self.user_message_lengths.append(len(user_msg))
        self.assistant_message_lengths.append(len(assistant_msg))
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens

    @property
    def duration_seconds(self) -> float:
        if not self.turn_timestamps:
            return 0
        return (self.turn_timestamps[-1] - self.start_time).total_seconds()

    @property
    def avg_response_time(self) -> float:
        if len(self.turn_timestamps) < 2:
            return 0
        deltas = [
            (self.turn_timestamps[i] - self.turn_timestamps[i-1]).total_seconds()
            for i in range(1, len(self.turn_timestamps))
        ]
        return sum(deltas) / len(deltas)

    @property
    def total_turns(self) -> int:
        return len(self.turn_timestamps)

    def report(self) -> dict:
        """Generate an analytics report."""
        return {
            "total_turns": self.total_turns,
            "duration_seconds": round(self.duration_seconds, 1),
            "avg_user_msg_length": (
                round(sum(self.user_message_lengths) / len(self.user_message_lengths))
                if self.user_message_lengths else 0
            ),
            "avg_assistant_msg_length": (
                round(sum(self.assistant_message_lengths) / len(self.assistant_message_lengths))
                if self.assistant_message_lengths else 0
            ),
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "estimated_cost_usd": round(
                (self.total_input_tokens * 0.003 + self.total_output_tokens * 0.015) / 1000,
                4
            ),
            "topic_switches": self.topic_switches,
            "error_recoveries": self.error_recoveries,
            "clarification_requests": self.clarification_requests
        }
```

---

## Exercises

### Exercise 1: Conversation Manager with Hybrid Memory

Build a complete conversation manager class that implements hybrid memory (sliding window + summarization + entity extraction). It should:

- Keep the last 5 turn pairs verbatim
- Summarize older turns
- Track entities (people, technologies, decisions)
- Inject both summary and entities into the system prompt
- Support conversation export/import (JSON serialization)

<details><summary>Show Answer</summary>

```python
import anthropic
import json
import copy
from dataclasses import dataclass, field, asdict
from typing import Any, Optional
from datetime import datetime


@dataclass
class EntityRecord:
    name: str
    entity_type: str
    attributes: dict[str, Any] = field(default_factory=dict)
    first_turn: int = 0
    last_turn: int = 0


@dataclass
class HybridConversationManager:
    system: str = ""
    model: str = "claude-sonnet-4-20250514"
    max_recent_pairs: int = 5
    messages: list[dict] = field(default_factory=list)
    summary: str = ""
    entities: dict[str, EntityRecord] = field(default_factory=dict)
    turn_count: int = 0
    _client: anthropic.Anthropic = field(
        default_factory=anthropic.Anthropic, repr=False
    )

    def _get_recent_messages(self) -> list[dict]:
        recent = self.messages[-(self.max_recent_pairs * 2):]
        while recent and recent[0]["role"] != "user":
            recent = recent[1:]
        return recent

    def _build_system(self) -> str:
        parts = [self.system]
        if self.entities:
            entity_lines = []
            for e in sorted(
                self.entities.values(), key=lambda x: x.last_turn, reverse=True
            ):
                attrs = ", ".join(f"{k}={v}" for k, v in e.attributes.items())
                entity_lines.append(f"- {e.name} ({e.entity_type}): {attrs}")
            parts.append("## Known Entities\n" + "\n".join(entity_lines))
        if self.summary:
            parts.append(f"## Earlier Conversation Summary\n{self.summary}")
        return "\n\n".join(parts)

    def _update_summary(self) -> None:
        if len(self.messages) <= self.max_recent_pairs * 2:
            return
        old = self.messages[:-(self.max_recent_pairs * 2)]
        text = "\n".join(f"{m['role']}: {m['content'][:200]}" for m in old)
        resp = self._client.messages.create(
            model=self.model,
            max_tokens=400,
            messages=[{
                "role": "user",
                "content": (
                    f"Summarize this conversation concisely. Preserve key facts, "
                    f"decisions, and action items.\n\nPrevious summary: "
                    f"{self.summary or '(none)'}\n\nNew messages:\n{text}"
                )
            }]
        )
        self.summary = resp.content[0].text

    def _update_entities(self, user_msg: str, assistant_msg: str) -> None:
        resp = self._client.messages.create(
            model=self.model,
            max_tokens=512,
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"Extract entities from this exchange. Return JSON:\n"
                        f'{{"entities": [{{"name": "...", "type": "person|tech|decision|project", '
                        f'"attributes": {{"key": "val"}}}}]}}\n\n'
                        f"User: {user_msg}\nAssistant: {assistant_msg[:300]}\n\n"
                        f'Return {{"entities": []}} if none found.'
                    )
                },
                {"role": "assistant", "content": "{"}
            ]
        )
        try:
            data = json.loads("{" + resp.content[0].text)
            for e in data.get("entities", []):
                name = e["name"]
                if name in self.entities:
                    self.entities[name].attributes.update(e.get("attributes", {}))
                    self.entities[name].last_turn = self.turn_count
                else:
                    self.entities[name] = EntityRecord(
                        name=name,
                        entity_type=e.get("type", "unknown"),
                        attributes=e.get("attributes", {}),
                        first_turn=self.turn_count,
                        last_turn=self.turn_count
                    )
        except (json.JSONDecodeError, KeyError):
            pass

    def say(self, user_message: str) -> str:
        self.messages.append({"role": "user", "content": user_message})
        self.turn_count += 1

        response = self._client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=self._build_system(),
            messages=self._get_recent_messages()
        )

        reply = response.content[0].text
        self.messages.append({"role": "assistant", "content": reply})

        # Update memory systems
        self._update_entities(user_message, reply)
        if len(self.messages) > self.max_recent_pairs * 2 + 2:
            self._update_summary()

        return reply

    def export_json(self) -> str:
        data = {
            "system": self.system,
            "messages": self.messages,
            "summary": self.summary,
            "entities": {
                name: {
                    "name": e.name,
                    "entity_type": e.entity_type,
                    "attributes": e.attributes,
                    "first_turn": e.first_turn,
                    "last_turn": e.last_turn
                }
                for name, e in self.entities.items()
            },
            "turn_count": self.turn_count,
            "exported_at": datetime.now().isoformat()
        }
        return json.dumps(data, indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> "HybridConversationManager":
        data = json.loads(json_str)
        mgr = cls(system=data["system"])
        mgr.messages = data["messages"]
        mgr.summary = data["summary"]
        mgr.turn_count = data["turn_count"]
        for name, e_data in data.get("entities", {}).items():
            mgr.entities[name] = EntityRecord(**e_data)
        return mgr

    def status(self) -> dict:
        return {
            "turns": self.turn_count,
            "total_messages": len(self.messages),
            "active_messages": len(self._get_recent_messages()),
            "entities_tracked": len(self.entities),
            "summary_length": len(self.summary),
            "total_chars": sum(len(m["content"]) for m in self.messages)
        }


# Test
conv = HybridConversationManager(
    system="You are a project planning assistant."
)
conv.say("I'm Alice, working on Project Phoenix, a data pipeline using Apache Kafka.")
conv.say("We decided to use Python for the consumer services.")
print(json.dumps(conv.status(), indent=2))
print("\nEntities:", {n: e.attributes for n, e in conv.entities.items()})
```

</details>

### Exercise 2: Topic-Aware Conversation Router

Build a conversation system that detects topic switches and maintains separate context per topic. When the user returns to a previous topic, the relevant context should be restored.

**Requirements:**
- Detect topic switches using an LLM classifier
- Maintain per-topic message history
- When returning to a topic, inject its specific context
- Support at least 3 concurrent topics

<details><summary>Show Answer</summary>

```python
import anthropic
import json
from dataclasses import dataclass, field


@dataclass
class TopicContext:
    topic_name: str
    messages: list[dict] = field(default_factory=list)
    summary: str = ""
    last_active_turn: int = 0


@dataclass
class TopicRouter:
    system: str = ""
    model: str = "claude-sonnet-4-20250514"
    topics: dict[str, TopicContext] = field(default_factory=dict)
    active_topic: str = ""
    global_turn: int = 0
    _client: anthropic.Anthropic = field(
        default_factory=anthropic.Anthropic, repr=False
    )

    def _classify_topic(self, user_msg: str) -> dict:
        existing = list(self.topics.keys()) if self.topics else ["(none)"]
        tools = [{
            "name": "classify",
            "description": "Classify the topic of a message",
            "input_schema": {
                "type": "object",
                "properties": {
                    "topic": {"type": "string", "description": "Topic name (short, descriptive)"},
                    "is_new_topic": {"type": "boolean"},
                    "is_return_to_previous": {"type": "boolean"},
                    "previous_topic_name": {"type": "string"}
                },
                "required": ["topic", "is_new_topic", "is_return_to_previous"]
            }
        }]

        resp = self._client.messages.create(
            model=self.model,
            max_tokens=256,
            tools=tools,
            tool_choice={"type": "tool", "name": "classify"},
            messages=[{
                "role": "user",
                "content": (
                    f"Current topic: {self.active_topic or '(none)'}\n"
                    f"Known topics: {', '.join(existing)}\n"
                    f"User message: {user_msg}\n\n"
                    f"Classify this message's topic."
                )
            }]
        )

        for block in resp.content:
            if block.type == "tool_use":
                return block.input
        return {"topic": self.active_topic or "general", "is_new_topic": False, "is_return_to_previous": False}

    def _get_topic_context(self, topic_name: str) -> str:
        ctx = self.topics.get(topic_name)
        if not ctx or not ctx.messages:
            return ""
        if ctx.summary:
            return f"Previous discussion on '{topic_name}':\n{ctx.summary}"
        # Return last few messages
        recent = ctx.messages[-6:]
        text = "\n".join(f"{m['role']}: {m['content'][:150]}" for m in recent)
        return f"Previous discussion on '{topic_name}':\n{text}"

    def say(self, user_message: str) -> str:
        self.global_turn += 1

        # Classify topic
        classification = self._classify_topic(user_message)
        topic_name = classification["topic"]

        # Handle topic switch
        if topic_name != self.active_topic:
            if classification.get("is_return_to_previous") and topic_name in self.topics:
                print(f"  [Returning to topic: {topic_name}]")
            elif classification.get("is_new_topic"):
                print(f"  [New topic: {topic_name}]")
            else:
                print(f"  [Topic: {topic_name}]")

        # Ensure topic exists
        if topic_name not in self.topics:
            self.topics[topic_name] = TopicContext(topic_name=topic_name)

        self.active_topic = topic_name
        topic_ctx = self.topics[topic_name]
        topic_ctx.last_active_turn = self.global_turn
        topic_ctx.messages.append({"role": "user", "content": user_message})

        # Build system with topic context
        system_parts = [self.system]

        # Add context for the active topic
        if len(topic_ctx.messages) > 2:
            ctx = self._get_topic_context(topic_name)
            if ctx:
                system_parts.append(ctx)

        # Add brief mentions of other topics
        other_topics = [
            t for t in self.topics.values()
            if t.topic_name != topic_name and t.messages
        ]
        if other_topics:
            other_ctx = "Other topics discussed: " + ", ".join(
                t.topic_name for t in other_topics
            )
            system_parts.append(other_ctx)

        # Use the topic's recent messages
        recent = topic_ctx.messages[-10:]
        while recent and recent[0]["role"] != "user":
            recent = recent[1:]

        response = self._client.messages.create(
            model=self.model,
            max_tokens=1024,
            system="\n\n".join(system_parts),
            messages=recent
        )

        reply = response.content[0].text
        topic_ctx.messages.append({"role": "assistant", "content": reply})

        return reply

    def get_topic_summary(self) -> dict:
        return {
            name: {
                "messages": len(ctx.messages),
                "last_active": ctx.last_active_turn
            }
            for name, ctx in self.topics.items()
        }


# Test
router = TopicRouter(system="You are a knowledgeable technical assistant.")
router.say("How do I set up PostgreSQL replication?")
router.say("What about read replicas vs streaming replication?")
router.say("Switching topics - what's the best way to learn Rust?")
router.say("Going back to PostgreSQL - what about failover?")

print("\nTopics:", json.dumps(router.get_topic_summary(), indent=2))
```

</details>

### Exercise 3: Conversation Loop Detector and Breaker

Implement a system that detects when a conversation enters a repetitive loop (the model keeps giving similar responses) and automatically breaks out by reformulating the question or changing the approach.

**Requirements:**
- Detect exact repetition and semantic repetition
- Try at least 2 different strategies to break the loop
- Log loop occurrences for debugging

<details><summary>Show Answer</summary>

```python
import anthropic
import hashlib
import json
from dataclasses import dataclass, field
from difflib import SequenceMatcher


@dataclass
class LoopDetectorConversation:
    system: str = ""
    model: str = "claude-sonnet-4-20250514"
    messages: list[dict] = field(default_factory=list)
    _response_fingerprints: list[str] = field(default_factory=list)
    _loop_log: list[dict] = field(default_factory=list)
    similarity_threshold: float = 0.7
    window_size: int = 3
    _client: anthropic.Anthropic = field(
        default_factory=anthropic.Anthropic, repr=False
    )

    def _fingerprint(self, text: str) -> str:
        normalized = " ".join(text.lower().split())[:300]
        return hashlib.sha256(normalized.encode()).hexdigest()[:16]

    def _similarity(self, a: str, b: str) -> float:
        a_norm = " ".join(a.lower().split())[:500]
        b_norm = " ".join(b.lower().split())[:500]
        return SequenceMatcher(None, a_norm, b_norm).ratio()

    def _detect_loop(self, new_response: str) -> bool:
        if len(self._response_fingerprints) < self.window_size:
            return False

        new_fp = self._fingerprint(new_response)
        recent_fps = self._response_fingerprints[-self.window_size:]

        # Check exact repetition
        if all(fp == new_fp for fp in recent_fps):
            return True

        # Check semantic similarity
        recent_texts = [
            m["content"] for m in self.messages
            if m["role"] == "assistant"
        ][-self.window_size:]

        similarities = [
            self._similarity(new_response, old) for old in recent_texts
        ]
        if all(s > self.similarity_threshold for s in similarities):
            return True

        return False

    def _break_loop(self, original_query: str, attempt: int) -> str:
        strategies = [
            # Strategy 1: Reformulate the question
            {
                "name": "reformulate",
                "prompt": (
                    f"The user's question hasn't been answered satisfactorily "
                    f"despite multiple attempts. Original question: '{original_query}'\n\n"
                    f"Please approach this from a completely different angle. "
                    f"If you were explaining this to someone in a different field, "
                    f"how would you frame it?"
                )
            },
            # Strategy 2: Break into sub-questions
            {
                "name": "decompose",
                "prompt": (
                    f"I notice I may be going in circles. Let me break down the "
                    f"original question into smaller parts.\n\n"
                    f"Original: '{original_query}'\n\n"
                    f"Let me address the most fundamental sub-question first, "
                    f"then build up from there."
                )
            },
            # Strategy 3: Acknowledge and ask for clarification
            {
                "name": "clarify",
                "prompt": (
                    f"I realize I may be misunderstanding what you're asking. "
                    f"Your original question was: '{original_query}'\n\n"
                    f"Could you tell me specifically what aspect you'd like me "
                    f"to focus on? What information would be most useful to you "
                    f"right now?"
                )
            }
        ]

        strategy = strategies[min(attempt, len(strategies) - 1)]

        self._loop_log.append({
            "turn": len(self.messages) // 2,
            "strategy": strategy["name"],
            "original_query": original_query
        })

        return strategy["prompt"]

    def say(self, user_message: str) -> str:
        self.messages.append({"role": "user", "content": user_message})

        response = self._client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=self.system,
            messages=self.messages[-20:]
        )

        reply = response.content[0].text

        if self._detect_loop(reply):
            print("  [Loop detected! Attempting to break...]")

            # Try up to 2 break strategies
            for attempt in range(2):
                breaker = self._break_loop(user_message, attempt)
                self.messages.append({"role": "assistant", "content": reply})
                self.messages.append({"role": "user", "content": breaker})

                response = self._client.messages.create(
                    model=self.model,
                    max_tokens=1024,
                    system=self.system,
                    messages=self.messages[-20:]
                )

                new_reply = response.content[0].text
                if not self._detect_loop(new_reply):
                    reply = new_reply
                    print(f"  [Loop broken with strategy: {self._loop_log[-1]['strategy']}]")
                    break
                reply = new_reply
            else:
                print("  [Could not break loop after 2 attempts]")
                reply = (
                    "I notice I'm having difficulty providing a different perspective "
                    "on this. Could you rephrase your question or tell me what "
                    "specific aspect you'd like me to address differently?"
                )

        self._response_fingerprints.append(self._fingerprint(reply))
        self.messages.append({"role": "assistant", "content": reply})
        return reply

    def get_loop_log(self) -> list[dict]:
        return self._loop_log


# Test
conv = LoopDetectorConversation(
    system="You are a helpful assistant."
)
# Normal usage
print(conv.say("What is recursion in programming?"))
print(conv.say("Can you give me a different explanation?"))
print("\nLoop log:", conv.get_loop_log())
```

</details>

### Exercise 4: Context Window Optimizer

Build a tool that analyzes a conversation's token usage and suggests optimizations. It should: calculate current token usage per component (system prompt, history, response budget), identify messages that can be compressed or removed, and estimate cost savings from different compression strategies.

<details><summary>Show Answer</summary>

```python
import anthropic
from dataclasses import dataclass, field


@dataclass
class TokenUsageReport:
    system_tokens: int = 0
    history_tokens: int = 0
    response_budget: int = 0
    context_limit: int = 200000
    per_message_tokens: list[dict] = field(default_factory=list)

    @property
    def total_used(self) -> int:
        return self.system_tokens + self.history_tokens + self.response_budget

    @property
    def utilization(self) -> float:
        return self.total_used / self.context_limit

    @property
    def remaining(self) -> int:
        return self.context_limit - self.total_used


def estimate_tokens(text: str) -> int:
    """Rough token estimate for English text."""
    return max(1, len(text) // 4)


def analyze_conversation(
    system: str,
    messages: list[dict],
    response_budget: int = 1024,
    context_limit: int = 200000
) -> TokenUsageReport:
    """Analyze token usage in a conversation."""
    report = TokenUsageReport(
        system_tokens=estimate_tokens(system),
        response_budget=response_budget,
        context_limit=context_limit
    )

    total_history = 0
    for i, msg in enumerate(messages):
        content = msg["content"] if isinstance(msg["content"], str) else str(msg["content"])
        tokens = estimate_tokens(content)
        total_history += tokens
        report.per_message_tokens.append({
            "index": i,
            "role": msg["role"],
            "tokens": tokens,
            "preview": content[:60],
            "percentage_of_history": 0  # Filled below
        })

    report.history_tokens = total_history

    # Calculate percentages
    for entry in report.per_message_tokens:
        entry["percentage_of_history"] = round(
            entry["tokens"] / max(total_history, 1) * 100, 1
        )

    return report


def suggest_optimizations(report: TokenUsageReport) -> list[dict]:
    """Suggest optimizations based on the usage report."""
    suggestions = []

    # Check if nearing context limit
    if report.utilization > 0.8:
        suggestions.append({
            "priority": "HIGH",
            "type": "context_pressure",
            "message": (
                f"Context window is {report.utilization:.0%} full. "
                f"Only {report.remaining} tokens remaining."
            ),
            "action": "Apply sliding window or summarization immediately"
        })

    # Find large messages
    large_msgs = [
        m for m in report.per_message_tokens
        if m["tokens"] > report.history_tokens * 0.15
    ]
    if large_msgs:
        for msg in large_msgs:
            suggestions.append({
                "priority": "MEDIUM",
                "type": "large_message",
                "message": (
                    f"Message {msg['index']} ({msg['role']}) uses "
                    f"{msg['percentage_of_history']}% of history "
                    f"({msg['tokens']} tokens): \"{msg['preview']}...\""
                ),
                "action": "Consider summarizing this message",
                "savings": msg["tokens"] - 50  # Estimated compressed size
            })

    # Check system prompt size
    system_ratio = report.system_tokens / report.context_limit
    if system_ratio > 0.05:
        suggestions.append({
            "priority": "LOW",
            "type": "large_system_prompt",
            "message": (
                f"System prompt uses {report.system_tokens} tokens "
                f"({system_ratio:.1%} of context)"
            ),
            "action": "Consider condensing the system prompt"
        })

    # Estimate savings from different strategies
    strategies = []

    # Sliding window (keep last 5 pairs)
    if len(report.per_message_tokens) > 10:
        window_tokens = sum(
            m["tokens"] for m in report.per_message_tokens[-10:]
        )
        savings = report.history_tokens - window_tokens
        strategies.append({
            "strategy": "sliding_window_5",
            "description": "Keep last 5 turn pairs",
            "current_tokens": report.history_tokens,
            "after_tokens": window_tokens,
            "savings_tokens": savings,
            "savings_percent": round(savings / max(report.history_tokens, 1) * 100)
        })

    # Summarization (estimate 10:1 compression)
    if report.history_tokens > 1000:
        summary_tokens = report.history_tokens // 10
        savings = report.history_tokens - summary_tokens - 500  # 500 for recent
        strategies.append({
            "strategy": "summarize_old",
            "description": "Summarize all but last 3 pairs",
            "current_tokens": report.history_tokens,
            "after_tokens": summary_tokens + 500,
            "savings_tokens": max(savings, 0),
            "savings_percent": round(max(savings, 0) / max(report.history_tokens, 1) * 100)
        })

    if strategies:
        suggestions.append({
            "priority": "INFO",
            "type": "compression_strategies",
            "strategies": strategies
        })

    return suggestions


def print_analysis(system: str, messages: list[dict]) -> None:
    """Print a full conversation analysis."""
    report = analyze_conversation(system, messages)
    suggestions = suggest_optimizations(report)

    print("=" * 60)
    print("CONVERSATION TOKEN ANALYSIS")
    print("=" * 60)

    print(f"\nToken Budget:")
    print(f"  System prompt:    {report.system_tokens:>8} tokens")
    print(f"  Message history:  {report.history_tokens:>8} tokens")
    print(f"  Response budget:  {report.response_budget:>8} tokens")
    print(f"  {'Total':>18}: {report.total_used:>8} / {report.context_limit}")
    print(f"  Utilization: {report.utilization:.1%}")

    print(f"\nMessage Breakdown ({len(report.per_message_tokens)} messages):")
    for m in report.per_message_tokens:
        bar = "#" * min(int(m["percentage_of_history"]), 40)
        print(f"  [{m['index']:>2}] {m['role']:>9} {m['tokens']:>6}t "
              f"({m['percentage_of_history']:>5.1f}%) {bar}")

    print(f"\nSuggestions ({len(suggestions)}):")
    for s in suggestions:
        if s["type"] == "compression_strategies":
            print(f"\n  Compression Strategies:")
            for st in s["strategies"]:
                print(f"    - {st['strategy']}: {st['description']}")
                print(f"      {st['current_tokens']} -> {st['after_tokens']} tokens "
                      f"(save {st['savings_percent']}%)")
        else:
            print(f"\n  [{s['priority']}] {s['type']}")
            print(f"    {s['message']}")
            print(f"    Action: {s['action']}")


# Test
system = "You are a helpful assistant. " * 20  # Deliberately large
messages = [
    {"role": "user", "content": f"Question {i}: Tell me something interesting about topic {i}. " * 10}
    if i % 2 == 0 else
    {"role": "assistant", "content": f"Here is a detailed answer about topic {i}. " * 20}
    for i in range(20)
]

print_analysis(system, messages)
```

</details>

### Exercise 5: Conversation Replay and Debugging Tool

Build a tool that can replay a saved conversation turn by turn, showing what the model would say at each step with the current system prompt. This is useful for debugging prompt changes -- you can see how a new system prompt affects responses to the same user inputs.

**Requirements:**
- Accept a saved conversation (list of messages) and a system prompt
- Replay each user message and compare the new response to the original
- Show a diff-like comparison (similar/different)
- Report overall consistency score

<details><summary>Show Answer</summary>

```python
import anthropic
import json
from dataclasses import dataclass, field
from difflib import SequenceMatcher


@dataclass
class ReplayResult:
    turn: int
    user_message: str
    original_response: str
    replayed_response: str
    similarity: float
    key_differences: list[str] = field(default_factory=list)


def replay_conversation(
    original_messages: list[dict],
    system_prompt: str,
    model: str = "claude-sonnet-4-20250514",
    verbose: bool = True
) -> list[ReplayResult]:
    """Replay a conversation with a (potentially different) system prompt."""
    client = anthropic.Anthropic()
    results = []
    replay_messages = []
    turn = 0

    for i in range(0, len(original_messages) - 1, 2):
        if original_messages[i]["role"] != "user":
            continue
        if i + 1 >= len(original_messages):
            break
        if original_messages[i + 1]["role"] != "assistant":
            continue

        turn += 1
        user_msg = original_messages[i]["content"]
        original_response = original_messages[i + 1]["content"]

        # Replay with current system prompt
        replay_messages.append({"role": "user", "content": user_msg})

        response = client.messages.create(
            model=model,
            max_tokens=1024,
            system=system_prompt,
            messages=replay_messages
        )

        replayed_response = response.content[0].text

        # Calculate similarity
        similarity = SequenceMatcher(
            None,
            original_response.lower(),
            replayed_response.lower()
        ).ratio()

        # Find key differences
        differences = []
        orig_words = set(original_response.lower().split())
        replay_words = set(replayed_response.lower().split())
        only_in_original = orig_words - replay_words
        only_in_replay = replay_words - orig_words

        if len(only_in_original) > 5:
            differences.append(
                f"Original had unique terms: {', '.join(list(only_in_original)[:5])}"
            )
        if len(only_in_replay) > 5:
            differences.append(
                f"Replay has unique terms: {', '.join(list(only_in_replay)[:5])}"
            )

        # Length comparison
        len_ratio = len(replayed_response) / max(len(original_response), 1)
        if len_ratio > 1.5:
            differences.append(f"Replay is {len_ratio:.1f}x longer")
        elif len_ratio < 0.67:
            differences.append(f"Replay is {len_ratio:.1f}x shorter")

        result = ReplayResult(
            turn=turn,
            user_message=user_msg,
            original_response=original_response,
            replayed_response=replayed_response,
            similarity=similarity,
            key_differences=differences
        )
        results.append(result)

        # Use replayed response for subsequent context
        replay_messages.append({"role": "assistant", "content": replayed_response})

        if verbose:
            status = "SAME" if similarity > 0.8 else "DIFFERENT" if similarity < 0.4 else "SIMILAR"
            print(f"\n--- Turn {turn} [{status}] (similarity: {similarity:.1%}) ---")
            print(f"  User: {user_msg[:80]}...")
            print(f"  Original: {original_response[:100]}...")
            print(f"  Replayed: {replayed_response[:100]}...")
            for diff in differences:
                print(f"  Delta: {diff}")

    # Overall report
    if results:
        avg_similarity = sum(r.similarity for r in results) / len(results)
        same_count = sum(1 for r in results if r.similarity > 0.8)
        diff_count = sum(1 for r in results if r.similarity < 0.4)

        print(f"\n{'=' * 60}")
        print(f"REPLAY SUMMARY")
        print(f"{'=' * 60}")
        print(f"Turns replayed: {len(results)}")
        print(f"Average similarity: {avg_similarity:.1%}")
        print(f"Consistent responses (>80%): {same_count}")
        print(f"Divergent responses (<40%): {diff_count}")
        print(f"Consistency score: {same_count / len(results):.0%}")

    return results


# Usage example
saved_conversation = [
    {"role": "user", "content": "What's the best way to handle errors in Python?"},
    {"role": "assistant", "content": "The best practice for error handling in Python is to use try/except blocks. Catch specific exceptions rather than bare except clauses. Use finally for cleanup code, and consider creating custom exception classes for your application."},
    {"role": "user", "content": "Show me an example with custom exceptions."},
    {"role": "assistant", "content": "Here's an example:\n\n```python\nclass ValidationError(Exception):\n    def __init__(self, field, message):\n        self.field = field\n        super().__init__(f'{field}: {message}')\n\ntry:\n    raise ValidationError('email', 'Invalid format')\nexcept ValidationError as e:\n    print(f'Validation failed: {e}')\n```"},
]

# Test with original system prompt
original_system = "You are a helpful Python tutor."

# Test with modified system prompt
new_system = "You are a concise Python expert. Use minimal explanations. Code speaks louder than words."

print("REPLAY WITH ORIGINAL PROMPT:")
results_orig = replay_conversation(saved_conversation, original_system)

print("\n\nREPLAY WITH NEW PROMPT:")
results_new = replay_conversation(saved_conversation, new_system)
```

</details>

---

**Previous**: [System Prompt Design](./06_System_Prompt_Design.md) | **Next**: [Multimodal Prompting](./08_Multimodal_Prompting.md)
