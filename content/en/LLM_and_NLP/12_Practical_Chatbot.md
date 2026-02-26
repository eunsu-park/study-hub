# 12. Practical Chatbot Project

## Learning Objectives

- Designing conversational AI systems
- Implementing RAG-based chatbots
- Conversation management and memory
- Production deployment considerations

---

## 1. Chatbot Architecture

### Basic Structure

```
┌─────────────────────────────────────────────────────────────┐
│                       Chatbot System                         │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  User Input                                                   │
│      │                                                       │
│      ▼                                                       │
│  [Intent Classification] ──▶ FAQ / RAG / General dialogue branch │
│      │                                                       │
│      ▼                                                       │
│  [Context Retrieval] ◀── Vector DB                          │
│      │                                                       │
│      ▼                                                       │
│  [Prompt Construction] ◀── Conversation History             │
│      │                                                       │
│      ▼                                                       │
│  [LLM Generation]                                            │
│      │                                                       │
│      ▼                                                       │
│  Response Output                                              │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. Basic Chatbot Implementation

### Simple Conversational Chatbot

```python
from openai import OpenAI

class SimpleChatbot:
    def __init__(self, system_prompt=None):
        self.client = OpenAI()
        self.system_prompt = system_prompt or "You are a helpful assistant."
        self.history = []

    def chat(self, user_message):
        # Construct messages
        messages = [{"role": "system", "content": self.system_prompt}]
        messages.extend(self.history)
        messages.append({"role": "user", "content": user_message})

        # API call
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.7
        )

        assistant_message = response.choices[0].message.content

        # Update history
        self.history.append({"role": "user", "content": user_message})
        self.history.append({"role": "assistant", "content": assistant_message})

        return assistant_message

    def clear_history(self):
        self.history = []

# Usage
bot = SimpleChatbot("You are a friendly customer support agent.")
print(bot.chat("Hi, I need help with my order."))
print(bot.chat("My order number is 12345."))
```

---

## 3. RAG Chatbot

### Document-Based Q&A

```python
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

class RAGChatbot:
    def __init__(self, documents, persist_dir="./rag_db"):
        self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        self.embeddings = OpenAIEmbeddings()

        # Split documents
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        chunks = splitter.split_documents(documents)

        # Vector store
        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=persist_dir
        )
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})

        # Conversation history
        self.history = []

        # Setup RAG chain
        self._setup_chain()

    def _setup_chain(self):
        template = """You are a helpful assistant. Answer based on the context.
If you don't know the answer, say so.

Context:
{context}

Conversation History:
{history}

Question: {question}

Answer:"""

        self.prompt = ChatPromptTemplate.from_template(template)

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        def format_history(history):
            if not history:
                return "No previous conversation."
            return "\n".join([f"{m['role']}: {m['content']}" for m in history[-6:]])

        self.chain = (
            {
                "context": self.retriever | format_docs,
                "history": lambda x: format_history(self.history),
                "question": RunnablePassthrough()
            }
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

    def chat(self, question):
        response = self.chain.invoke(question)

        # Update history
        self.history.append({"role": "user", "content": question})
        self.history.append({"role": "assistant", "content": response})

        return response

    def get_sources(self, question):
        """Return retrieved source documents"""
        docs = self.retriever.invoke(question)
        return [{"content": d.page_content, "metadata": d.metadata} for d in docs]
```

---

## 4. Advanced Conversation Management

### Intent Classification

```python
class IntentClassifier:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    def classify(self, message, intents):
        prompt = f"""Classify the user message into one of these intents: {intents}

Message: {message}

Intent (only output the intent name):"""

        response = self.llm.invoke(prompt)
        return response.content.strip()

# Usage
classifier = IntentClassifier()
intent = classifier.classify(
    "I want to return my purchase",
    ["order_status", "return_request", "product_inquiry", "general"]
)
# "return_request"
```

### Slot Extraction

```python
from pydantic import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser

class OrderSlots(BaseModel):
    order_id: str = Field(default=None, description="Order ID")
    product_name: str = Field(default=None, description="Product name")
    issue: str = Field(default=None, description="Customer's issue")

class SlotExtractor:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        self.parser = JsonOutputParser(pydantic_object=OrderSlots)

    def extract(self, message, context=""):
        prompt = f"""Extract information from the message.
{self.parser.get_format_instructions()}

Context: {context}
Message: {message}

JSON:"""

        response = self.llm.invoke(prompt)
        return self.parser.parse(response.content)

# Usage
extractor = SlotExtractor()
slots = extractor.extract("I want to return order #12345, the shirt is too small")
# {'order_id': '12345', 'product_name': 'shirt', 'issue': 'too small'}
```

### Conversation State Management

```python
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Any

class ConversationState(Enum):
    GREETING = "greeting"
    COLLECTING_INFO = "collecting_info"
    PROCESSING = "processing"
    CONFIRMING = "confirming"
    COMPLETED = "completed"

@dataclass
class ConversationContext:
    state: ConversationState = ConversationState.GREETING
    slots: Dict[str, Any] = field(default_factory=dict)
    history: List[Dict] = field(default_factory=list)
    missing_slots: List[str] = field(default_factory=list)

class StatefulChatbot:
    def __init__(self):
        self.context = ConversationContext()
        self.required_slots = ["order_id", "issue"]

    def process(self, message):
        # Extract slots
        new_slots = self.extract_slots(message)
        self.context.slots.update({k: v for k, v in new_slots.items() if v})

        # Check missing slots
        self.context.missing_slots = [
            s for s in self.required_slots
            if s not in self.context.slots or not self.context.slots[s]
        ]

        # State transition
        if self.context.missing_slots:
            self.context.state = ConversationState.COLLECTING_INFO
            return self.ask_for_slot(self.context.missing_slots[0])
        else:
            self.context.state = ConversationState.CONFIRMING
            return self.confirm_action()

    def ask_for_slot(self, slot_name):
        prompts = {
            "order_id": "Could you please provide your order number?",
            "issue": "What issue are you experiencing with your order?"
        }
        return prompts.get(slot_name, f"Please provide {slot_name}.")

    def confirm_action(self):
        return f"Let me confirm: Order {self.context.slots['order_id']}, Issue: {self.context.slots['issue']}. Is this correct?"
```

---

## 5. Streaming Responses

```python
from openai import OpenAI

class StreamingChatbot:
    def __init__(self):
        self.client = OpenAI()
        self.history = []

    def chat_stream(self, message):
        messages = [{"role": "system", "content": "You are helpful."}]
        messages.extend(self.history)
        messages.append({"role": "user", "content": message})

        stream = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            stream=True
        )

        full_response = ""
        for chunk in stream:
            if chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                full_response += content
                yield content

        # Update history
        self.history.append({"role": "user", "content": message})
        self.history.append({"role": "assistant", "content": full_response})

# Usage
bot = StreamingChatbot()
for chunk in bot.chat_stream("Tell me a story"):
    print(chunk, end="", flush=True)
```

---

## 6. FastAPI Web Server

```python
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uvicorn

app = FastAPI()

# Session storage
sessions = {}

class ChatRequest(BaseModel):
    session_id: str
    message: str

class ChatResponse(BaseModel):
    response: str
    sources: list = []

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    # Get/create session
    if request.session_id not in sessions:
        sessions[request.session_id] = RAGChatbot(documents)

    bot = sessions[request.session_id]

    # Generate response
    response = bot.chat(request.message)
    sources = bot.get_sources(request.message)

    return ChatResponse(response=response, sources=sources)

@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    if request.session_id not in sessions:
        sessions[request.session_id] = StreamingChatbot()

    bot = sessions[request.session_id]

    def generate():
        for chunk in bot.chat_stream(request.message):
            yield chunk

    return StreamingResponse(generate(), media_type="text/plain")

@app.delete("/session/{session_id}")
async def clear_session(session_id: str):
    if session_id in sessions:
        del sessions[session_id]
    return {"status": "cleared"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

---

## 7. Gradio UI

```python
import gradio as gr

class ChatbotUI:
    def __init__(self):
        self.bot = RAGChatbot(documents)

    def respond(self, message, history):
        response = self.bot.chat(message)
        return response

    def launch(self):
        demo = gr.ChatInterface(
            fn=self.respond,
            title="Document Q&A Chatbot",
            description="Ask questions about your documents",
            examples=["What is this document about?", "Summarize the main points"],
            theme="soft"
        )
        demo.launch()

# Usage
ui = ChatbotUI()
ui.launch()
```

---

## 8. Production Considerations

### Error Handling

```python
import logging
from tenacity import retry, stop_after_attempt, wait_exponential

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProductionChatbot:
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def chat(self, message):
        try:
            response = self._generate_response(message)
            return response
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "I apologize, but I'm having trouble processing your request. Please try again."
```

### Token/Cost Management

```python
import tiktoken

class TokenManager:
    def __init__(self, model="gpt-3.5-turbo", max_tokens=4000):
        self.encoding = tiktoken.encoding_for_model(model)
        self.max_tokens = max_tokens

    def count_tokens(self, text):
        return len(self.encoding.encode(text))

    def truncate_history(self, history, max_history_tokens=2000):
        """Remove older messages first"""
        total_tokens = 0
        truncated = []

        for msg in reversed(history):
            msg_tokens = self.count_tokens(msg['content'])
            if total_tokens + msg_tokens > max_history_tokens:
                break
            truncated.insert(0, msg)
            total_tokens += msg_tokens

        return truncated
```

### Monitoring

```python
import time
from dataclasses import dataclass
from typing import Optional

@dataclass
class ChatMetrics:
    session_id: str
    message: str
    response: str
    latency_ms: float
    token_count: int
    timestamp: float

class MonitoredChatbot:
    def __init__(self):
        self.metrics = []

    def chat(self, session_id, message):
        start = time.time()

        response = self._generate(message)

        latency = (time.time() - start) * 1000

        # Record metrics
        metric = ChatMetrics(
            session_id=session_id,
            message=message,
            response=response,
            latency_ms=latency,
            token_count=self.token_manager.count_tokens(response),
            timestamp=time.time()
        )
        self.metrics.append(metric)

        return response

    def get_avg_latency(self):
        if not self.metrics:
            return 0
        return sum(m.latency_ms for m in self.metrics) / len(self.metrics)
```

---

## Summary

### Chatbot Design Checklist

```
□ Define purpose (general dialogue / FAQ / document-based)
□ Decide if RAG is needed
□ Conversation history management approach
□ Need for intent classification
□ Error handling and fallbacks
□ Cost management (token limits)
□ Monitoring and logging
```

### Core Patterns

```python
# Basic chatbot
messages = [system_prompt] + history + [user_message]
response = llm.invoke(messages)

# RAG chatbot
docs = retriever.invoke(query)
context = format_docs(docs)
response = llm.invoke(prompt.format(context=context, question=query))

# Streaming
for chunk in llm.stream(messages):
    yield chunk
```

### Next Steps

- Deploy to production (AWS, GCP)
- Set up A/B testing
- Collect user feedback
- Continuous model improvement

## Exercises

### Exercise 1: History Truncation with Token Limits

The `TokenManager.truncate_history` method removes older messages when the history exceeds `max_history_tokens`. However, it has a subtle issue: it may break a conversation pair (e.g., keep an assistant message but drop its corresponding user message). Fix this issue and add a unit test.

<details>
<summary>Show Answer</summary>

```python
import tiktoken

class TokenManager:
    def __init__(self, model="gpt-3.5-turbo", max_tokens=4000):
        self.encoding = tiktoken.encoding_for_model(model)
        self.max_tokens = max_tokens

    def count_tokens(self, text: str) -> int:
        return len(self.encoding.encode(text))

    def truncate_history(self, history: list[dict], max_history_tokens: int = 2000) -> list[dict]:
        """
        Remove oldest message PAIRS first to maintain conversational coherence.
        Always removes user+assistant pairs together to avoid orphaned messages.
        """
        if not history:
            return history

        # Calculate total tokens
        total_tokens = sum(self.count_tokens(msg['content']) for msg in history)

        if total_tokens <= max_history_tokens:
            return history  # No truncation needed

        # Remove oldest pairs (2 messages at a time)
        truncated = list(history)
        while truncated and total_tokens > max_history_tokens:
            # Remove oldest pair (user + assistant)
            if len(truncated) >= 2:
                removed_user = truncated.pop(0)
                removed_assistant = truncated.pop(0)
                total_tokens -= (
                    self.count_tokens(removed_user['content']) +
                    self.count_tokens(removed_assistant['content'])
                )
            else:
                # Only one message left, remove it
                removed = truncated.pop(0)
                total_tokens -= self.count_tokens(removed['content'])

        return truncated


# Unit test
def test_truncate_history():
    tm = TokenManager()

    # Build a conversation with known token counts
    history = [
        {"role": "user", "content": "Hello"},           # ~1 token
        {"role": "assistant", "content": "Hi there!"},  # ~3 tokens
        {"role": "user", "content": "How are you?"},    # ~4 tokens
        {"role": "assistant", "content": "I'm great!"},  # ~3 tokens
        {"role": "user", "content": "Tell me about Python"},  # ~5 tokens
        {"role": "assistant", "content": "Python is a high-level programming language"},  # ~9 tokens
    ]

    # Test 1: No truncation needed
    result = tm.truncate_history(history, max_history_tokens=1000)
    assert len(result) == 6, "Should keep all messages when under limit"

    # Test 2: Truncation removes oldest pair
    result = tm.truncate_history(history, max_history_tokens=25)
    assert len(result) % 2 == 0, "Result should have even number of messages (complete pairs)"
    assert result[0]["role"] == "user", "First message should be user"

    # Test 3: Messages remain as pairs (no orphaned assistant messages)
    for i in range(0, len(result), 2):
        assert result[i]["role"] == "user", f"Message {i} should be user"
        assert result[i+1]["role"] == "assistant", f"Message {i+1} should be assistant"

    print("All tests passed!")

test_truncate_history()
```

**The bug in the original:** The original implementation iterates `reversed(history)` to count tokens from newest to oldest, but when it hits the token limit it stops without ensuring message-pair integrity. An assistant message with no corresponding user message confuses the LLM.
</details>

---

### Exercise 2: Intent-Driven Chatbot Router

Build a simple chatbot router that classifies user intent into one of three categories: `rag_query` (question answerable from documents), `chitchat` (general conversation), or `action_request` (needs to take an action like placing an order). Route each intent to a different handler.

<details>
<summary>Show Answer</summary>

```python
from enum import Enum
from dataclasses import dataclass

class Intent(Enum):
    RAG_QUERY = "rag_query"
    CHITCHAT = "chitchat"
    ACTION_REQUEST = "action_request"

@dataclass
class ChatbotResponse:
    intent: Intent
    response: str
    sources: list = None

class RouterChatbot:
    def __init__(self, rag_chatbot=None):
        from openai import OpenAI
        self.client = OpenAI()
        self.rag_chatbot = rag_chatbot  # Optional RAG component
        self.history = []

    def classify_intent(self, message: str) -> Intent:
        """Classify user intent using zero-shot LLM classification."""
        prompt = f"""Classify this message into exactly one category:
- rag_query: Question that needs to be answered from documents/knowledge base
- chitchat: Casual conversation, greetings, general questions
- action_request: Request to DO something (order, book, cancel, etc.)

Message: "{message}"

Category (reply with just the category name):"""

        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=20
        )
        result = response.choices[0].message.content.strip().lower()

        # Map to enum, default to chitchat if unrecognized
        mapping = {
            "rag_query": Intent.RAG_QUERY,
            "chitchat": Intent.CHITCHAT,
            "action_request": Intent.ACTION_REQUEST,
        }
        return mapping.get(result, Intent.CHITCHAT)

    def handle_rag(self, message: str) -> ChatbotResponse:
        """Handle document-based questions."""
        if self.rag_chatbot:
            response = self.rag_chatbot.chat(message)
            sources = self.rag_chatbot.get_sources(message)
        else:
            response = "I don't have access to documents to answer that question."
            sources = []
        return ChatbotResponse(Intent.RAG_QUERY, response, sources)

    def handle_chitchat(self, message: str) -> ChatbotResponse:
        """Handle casual conversation."""
        messages = [
            {"role": "system", "content": "You are a friendly conversational assistant."},
            {"role": "user", "content": message}
        ]
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.8
        )
        return ChatbotResponse(Intent.CHITCHAT, response.choices[0].message.content)

    def handle_action(self, message: str) -> ChatbotResponse:
        """Handle action requests (placeholder for actual action logic)."""
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

        # Update shared history
        self.history.append({"role": "user", "content": message})
        self.history.append({"role": "assistant", "content": result.response, "intent": intent.value})

        return result

# Test routing
bot = RouterChatbot()

test_messages = [
    "What are the return policies?",  # → rag_query
    "Hello! How's your day?",         # → chitchat
    "I want to cancel my order",      # → action_request
]

for msg in test_messages:
    result = bot.chat(msg)
    print(f"[{result.intent.value}] {msg[:40]}")
    print(f"  → {result.response[:80]}...\n")
```
</details>

---

### Exercise 3: Stateful Conversation with Missing Slot Recovery

Extend the `StatefulChatbot` to handle the case where a user provides conflicting information mid-conversation (e.g., first says order #12345, then says "actually it's #67890"). The bot should update the slot and confirm the correction.

<details>
<summary>Show Answer</summary>

```python
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from openai import OpenAI

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
    corrections: List[Dict] = field(default_factory=list)  # Track corrections

class SmartStatefulChatbot:
    """Chatbot that detects slot corrections during conversation."""

    REQUIRED_SLOTS = ["order_id", "issue"]
    SLOT_QUESTIONS = {
        "order_id": "Could you please provide your order number?",
        "issue": "What issue are you experiencing with your order?"
    }

    def __init__(self):
        self.client = OpenAI()
        self.context = ConversationContext()

    def extract_slots(self, message: str) -> dict:
        """Extract slot values from message."""
        import json
        prompt = f"""Extract order information from the message. Return JSON only.
Fields: order_id (string, e.g. "12345"), issue (string description)
Use null for missing fields.

Message: "{message}"
JSON:"""
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        try:
            return json.loads(response.choices[0].message.content)
        except:
            return {"order_id": None, "issue": None}

    def detect_correction(self, new_slots: dict) -> Optional[str]:
        """Check if user is correcting a previously provided slot."""
        for slot, new_value in new_slots.items():
            if new_value and slot in self.context.slots:
                old_value = self.context.slots[slot]
                if old_value and old_value != new_value:
                    return slot  # This slot was corrected
        return None

    def process(self, message: str) -> str:
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
                "new": new_value
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
        missing = [s for s in self.REQUIRED_SLOTS
                  if not self.context.slots.get(s)]

        if missing:
            self.context.state = ConversationState.COLLECTING_INFO
            response += self.SLOT_QUESTIONS[missing[0]]
        else:
            self.context.state = ConversationState.CONFIRMING
            response += self.confirm_action()

        self.context.history.append({"role": "assistant", "content": response})
        return response

    def confirm_action(self) -> str:
        return (
            f"To confirm: Order #{self.context.slots['order_id']}, "
            f"Issue: {self.context.slots['issue']}. Is this correct? (yes/no)"
        )

# Test correction handling
bot = SmartStatefulChatbot()

print(bot.process("I want to return order 12345"))  # Sets order_id, asks for issue
print(bot.process("The item is damaged"))           # Sets issue, asks to confirm
print(bot.process("Actually my order is 67890"))    # Corrects order_id
print(f"\nCorrections made: {bot.context.corrections}")
# [{'slot': 'order_id', 'old': '12345', 'new': '67890'}]
```
</details>

---

## Course Complete

This completes the LLM & NLP learning course!

### Course Summary

1. **NLP Basics (01-03)**: Tokenization, embeddings, Transformer
2. **Pre-trained Models (04-07)**: BERT, GPT, HuggingFace, fine-tuning
3. **LLM Applications (08-12)**: Prompting, RAG, LangChain, vector DBs, chatbots

### Recommended Next Steps

- Apply to real projects
- Participate in Kaggle NLP competitions
- Read latest LLM papers (Claude, Gemini, Llama)
