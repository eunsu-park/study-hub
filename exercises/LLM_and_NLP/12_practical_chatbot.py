"""
Exercises for Lesson 12: Practical Chatbot
Topic: LLM_and_NLP

Solutions to practice problems from the lesson.
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
import re


# ============================================================
# Shared utilities: lightweight token counting simulation
# (No tiktoken or API calls needed)
# ============================================================

def count_tokens_simple(text: str) -> int:
    """
    Approximate token count using whitespace + punctuation splitting.
    Real tokenizers (tiktoken, SentencePiece) average ~1.3 tokens/word.
    """
    words = text.split()
    return max(1, int(len(words) * 1.3))


# === Exercise 1: History Truncation with Token Limits ===
# Problem: Fix the TokenManager.truncate_history method that may break
# conversation pairs (e.g., keep an assistant message but drop its
# corresponding user message). Add unit tests.

def exercise_1():
    """History truncation with token limits — pair-preserving fix."""
    print("=" * 60)
    print("Exercise 1: History Truncation with Token Limits")
    print("=" * 60)

    class TokenManager:
        """
        Manages token counting and history truncation.
        Uses simple word-based approximation instead of tiktoken.
        """

        def __init__(self, max_tokens: int = 4000):
            self.max_tokens = max_tokens

        def count_tokens(self, text: str) -> int:
            return count_tokens_simple(text)

        def truncate_history(
            self, history: List[Dict], max_history_tokens: int = 2000
        ) -> List[Dict]:
            """
            Remove oldest message PAIRS first to maintain conversational coherence.
            Always removes user+assistant pairs together to avoid orphaned messages.

            The bug in the original: iterating reversed(history) to count tokens
            from newest to oldest, but when hitting the limit it stops without
            ensuring message-pair integrity. An assistant message with no
            corresponding user message confuses the LLM.
            """
            if not history:
                return history

            # Calculate total tokens
            total_tokens = sum(self.count_tokens(msg["content"]) for msg in history)

            if total_tokens <= max_history_tokens:
                return history  # No truncation needed

            # Remove oldest pairs (2 messages at a time)
            truncated = list(history)
            while truncated and total_tokens > max_history_tokens:
                if len(truncated) >= 2:
                    # Remove oldest pair (user + assistant)
                    removed_user = truncated.pop(0)
                    removed_assistant = truncated.pop(0)
                    total_tokens -= (
                        self.count_tokens(removed_user["content"])
                        + self.count_tokens(removed_assistant["content"])
                    )
                else:
                    # Only one message left — remove it
                    removed = truncated.pop(0)
                    total_tokens -= self.count_tokens(removed["content"])

            return truncated

    # Unit tests
    def test_truncate_history():
        tm = TokenManager()

        # Build a conversation with varying message lengths
        history = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you doing today?"},
            {"role": "assistant", "content": "I'm great, thanks for asking!"},
            {"role": "user", "content": "Tell me about Python programming language and its features"},
            {
                "role": "assistant",
                "content": (
                    "Python is a high-level programming language known for its "
                    "readability and versatility. It supports multiple paradigms."
                ),
            },
        ]

        # Test 1: No truncation needed (high limit)
        result = tm.truncate_history(history, max_history_tokens=1000)
        assert len(result) == 6, f"Should keep all messages, got {len(result)}"
        print("  Test 1 PASSED: No truncation when under limit")

        # Test 2: Truncation removes oldest pair(s)
        result = tm.truncate_history(history, max_history_tokens=30)
        assert (
            len(result) % 2 == 0
        ), f"Result should have even number of messages, got {len(result)}"
        assert result[0]["role"] == "user", "First message should be user"
        print(f"  Test 2 PASSED: Truncated to {len(result)} messages (even count)")

        # Test 3: Messages remain as complete pairs
        for i in range(0, len(result), 2):
            assert result[i]["role"] == "user", f"Message {i} should be user"
            if i + 1 < len(result):
                assert (
                    result[i + 1]["role"] == "assistant"
                ), f"Message {i + 1} should be assistant"
        print("  Test 3 PASSED: All messages are in user/assistant pairs")

        # Test 4: Empty history
        result = tm.truncate_history([], max_history_tokens=100)
        assert result == [], "Empty history should return empty"
        print("  Test 4 PASSED: Empty history handled correctly")

        # Test 5: Single message
        result = tm.truncate_history(
            [{"role": "user", "content": "x" * 200}], max_history_tokens=5
        )
        assert len(result) == 0, "Should remove single message if over limit"
        print("  Test 5 PASSED: Single oversized message removed")

        print("\n  All tests passed!")

    test_truncate_history()


# === Exercise 2: Intent-Driven Chatbot Router ===
# Problem: Build a chatbot router that classifies user intent into
# rag_query, chitchat, or action_request using keyword heuristics
# (simulated, no API calls).

def exercise_2():
    """Intent-driven chatbot router with simulated classification."""
    print("\n" + "=" * 60)
    print("Exercise 2: Intent-Driven Chatbot Router")
    print("=" * 60)

    class Intent(Enum):
        RAG_QUERY = "rag_query"
        CHITCHAT = "chitchat"
        ACTION_REQUEST = "action_request"

    @dataclass
    class ChatbotResponse:
        intent: Intent
        response: str
        sources: Optional[List[str]] = None

    class RouterChatbot:
        """
        Intent-based chatbot router.
        Uses keyword heuristics for classification (no API calls).
        In production, this would use an LLM for zero-shot classification.
        """

        # Keyword patterns for each intent
        RAG_KEYWORDS = [
            "what is", "what are", "how does", "explain", "define",
            "tell me about", "describe", "documentation", "policy",
            "return policy", "procedure", "guide", "tutorial",
        ]
        ACTION_KEYWORDS = [
            "cancel", "order", "book", "reserve", "schedule",
            "change", "update my", "delete", "remove", "refund",
            "subscribe", "unsubscribe", "buy", "purchase",
        ]

        def __init__(self):
            self.history: List[Dict] = []

        def classify_intent(self, message: str) -> Intent:
            """Classify user intent using keyword matching."""
            lower = message.lower().strip()

            # Check action keywords first (more specific)
            for keyword in self.ACTION_KEYWORDS:
                if keyword in lower:
                    return Intent.ACTION_REQUEST

            # Check RAG keywords
            for keyword in self.RAG_KEYWORDS:
                if keyword in lower:
                    return Intent.RAG_QUERY

            # Default to chitchat
            return Intent.CHITCHAT

        def handle_rag(self, message: str) -> ChatbotResponse:
            """Handle document-based questions."""
            # Simulated RAG response
            response = (
                f"Based on our documentation, here is what I found about your question: "
                f"'{message[:50]}...' — [simulated RAG response with relevant context]"
            )
            return ChatbotResponse(
                Intent.RAG_QUERY, response, sources=["doc_001.pdf", "faq.html"]
            )

        def handle_chitchat(self, message: str) -> ChatbotResponse:
            """Handle casual conversation."""
            greetings = ["hello", "hi", "hey", "good morning", "good evening"]
            lower = message.lower()

            if any(g in lower for g in greetings):
                response = "Hello! How can I help you today?"
            elif "how are you" in lower:
                response = "I'm doing well, thanks for asking! What can I assist you with?"
            elif "thank" in lower:
                response = "You're welcome! Is there anything else I can help with?"
            else:
                response = "That's interesting! Let me know if you have any questions I can help with."

            return ChatbotResponse(Intent.CHITCHAT, response)

        def handle_action(self, message: str) -> ChatbotResponse:
            """Handle action requests."""
            response = (
                f"I understand you want to: '{message}'. "
                "Let me collect some information to process this request. "
                "Could you provide more details?"
            )
            return ChatbotResponse(Intent.ACTION_REQUEST, response)

        def chat(self, message: str) -> ChatbotResponse:
            """Main entry point: classify and route."""
            intent = self.classify_intent(message)

            if intent == Intent.RAG_QUERY:
                result = self.handle_rag(message)
            elif intent == Intent.CHITCHAT:
                result = self.handle_chitchat(message)
            else:
                result = self.handle_action(message)

            self.history.append({"role": "user", "content": message})
            self.history.append({
                "role": "assistant",
                "content": result.response,
                "intent": intent.value,
            })

            return result

    # Test routing
    bot = RouterChatbot()

    test_messages = [
        ("What are the return policies?", "rag_query"),
        ("Hello! How's your day?", "chitchat"),
        ("I want to cancel my order", "action_request"),
        ("Tell me about your shipping options", "rag_query"),
        ("Thanks for your help!", "chitchat"),
        ("Can I book a meeting for tomorrow?", "action_request"),
    ]

    print("\nRouting Test Results:")
    print("-" * 60)
    all_passed = True
    for msg, expected in test_messages:
        result = bot.chat(msg)
        status = "PASS" if result.intent.value == expected else "FAIL"
        if status == "FAIL":
            all_passed = False
        print(f"  [{status}] [{result.intent.value:<15}] {msg}")
        if result.sources:
            print(f"          Sources: {result.sources}")

    print(f"\n  {'All routing tests passed!' if all_passed else 'Some tests failed.'}")


# === Exercise 3: Stateful Conversation with Missing Slot Recovery ===
# Problem: Extend StatefulChatbot to handle conflicting information
# mid-conversation (e.g., correcting an order number).

def exercise_3():
    """Stateful conversation with slot correction detection."""
    print("\n" + "=" * 60)
    print("Exercise 3: Stateful Conversation with Missing Slot Recovery")
    print("=" * 60)

    class ConversationState(Enum):
        GREETING = "greeting"
        COLLECTING_INFO = "collecting_info"
        CONFIRMING = "confirming"
        COMPLETED = "completed"

    @dataclass
    class ConversationContext:
        state: ConversationState = ConversationState.GREETING
        slots: Dict[str, Any] = field(default_factory=dict)
        history: List[Dict] = field(default_factory=list)
        corrections: List[Dict] = field(default_factory=list)

    class SmartStatefulChatbot:
        """Chatbot that detects slot corrections during conversation."""

        REQUIRED_SLOTS = ["order_id", "issue"]
        SLOT_QUESTIONS = {
            "order_id": "Could you please provide your order number?",
            "issue": "What issue are you experiencing with your order?",
        }

        def __init__(self):
            self.context = ConversationContext()

        def extract_slots(self, message: str) -> Dict[str, Optional[str]]:
            """
            Extract slot values from message using regex patterns.
            Simulates LLM-based extraction without API calls.
            """
            extracted: Dict[str, Optional[str]] = {"order_id": None, "issue": None}

            # Extract order ID patterns (e.g., #12345, order 12345, order number 67890)
            order_patterns = [
                r"#(\d{4,})",
                r"order\s*(?:number|#|num)?\s*(\d{4,})",
                r"(?:it'?s|is)\s*#?(\d{4,})",
            ]
            for pattern in order_patterns:
                match = re.search(pattern, message, re.IGNORECASE)
                if match:
                    extracted["order_id"] = match.group(1)
                    break

            # Extract issue descriptions
            issue_keywords = {
                "damaged": "Item is damaged",
                "broken": "Item is broken",
                "wrong item": "Wrong item received",
                "not delivered": "Order not delivered",
                "missing": "Items missing from order",
                "return": "Want to return item",
                "refund": "Requesting refund",
                "late": "Delivery is late",
            }
            lower = message.lower()
            for keyword, description in issue_keywords.items():
                if keyword in lower:
                    extracted["issue"] = description
                    break

            return extracted

        def detect_correction(self, new_slots: Dict) -> Optional[str]:
            """Check if user is correcting a previously provided slot."""
            for slot, new_value in new_slots.items():
                if new_value and slot in self.context.slots:
                    old_value = self.context.slots[slot]
                    if old_value and old_value != new_value:
                        return slot  # This slot was corrected
            return None

        def process(self, message: str) -> str:
            """Process a user message and return a response."""
            self.context.history.append({"role": "user", "content": message})
            new_slots = self.extract_slots(message)

            # Check for corrections
            corrected_slot = self.detect_correction(new_slots)
            if corrected_slot:
                old_value = self.context.slots[corrected_slot]
                new_value = new_slots[corrected_slot]
                self.context.corrections.append({
                    "slot": corrected_slot,
                    "old": old_value,
                    "new": new_value,
                })
                self.context.slots[corrected_slot] = new_value
                response = (
                    f"Got it! I've updated your {corrected_slot} from "
                    f"'{old_value}' to '{new_value}'. "
                )
            else:
                # Normal slot update
                for slot, value in new_slots.items():
                    if value:
                        self.context.slots[slot] = value
                response = ""

            # Check what's still missing
            missing = [
                s for s in self.REQUIRED_SLOTS if not self.context.slots.get(s)
            ]

            if missing:
                self.context.state = ConversationState.COLLECTING_INFO
                response += self.SLOT_QUESTIONS[missing[0]]
            else:
                self.context.state = ConversationState.CONFIRMING
                response += self._confirm_action()

            self.context.history.append({"role": "assistant", "content": response})
            return response

        def _confirm_action(self) -> str:
            return (
                f"To confirm: Order #{self.context.slots['order_id']}, "
                f"Issue: {self.context.slots['issue']}. Is this correct? (yes/no)"
            )

    # Test conversation with correction
    bot = SmartStatefulChatbot()

    print("\nSimulated Conversation:")
    print("-" * 60)

    messages = [
        "I want to return order 12345",          # Sets order_id, asks for issue
        "The item is damaged",                    # Sets issue, asks to confirm
        "Actually my order is #67890",            # Corrects order_id
    ]

    for msg in messages:
        print(f"\n  User: {msg}")
        response = bot.process(msg)
        print(f"  Bot:  {response}")

    print(f"\n  Final slots: {bot.context.slots}")
    print(f"  Corrections made: {bot.context.corrections}")

    # Verify correction was tracked
    assert len(bot.context.corrections) == 1
    assert bot.context.corrections[0]["old"] == "12345"
    assert bot.context.corrections[0]["new"] == "67890"
    assert bot.context.slots["order_id"] == "67890"
    print("\n  All assertions passed!")


if __name__ == "__main__":
    exercise_1()
    exercise_2()
    exercise_3()
