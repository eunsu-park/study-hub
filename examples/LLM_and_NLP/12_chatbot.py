"""
12. Practical Chatbot Example

RAG-based conversational AI system
"""

print("=" * 60)
print("Practical Chatbot")
print("=" * 60)


# ============================================
# 1. Simple Conversation Chatbot (Memory)
# ============================================
print("\n[1] Simple Conversation Chatbot")
print("-" * 40)

class SimpleChatbot:
    """Simple chatbot that maintains history"""

    def __init__(self, system_prompt="You are a helpful assistant."):
        self.system_prompt = system_prompt
        self.history = []

    def chat(self, user_message):
        """Process user message (LLM call simulation)"""
        # Add to history
        self.history.append({"role": "user", "content": user_message})

        # In practice, call LLM
        # response = llm.invoke(messages)
        response = f"[Response] This is the answer to: {user_message}"

        self.history.append({"role": "assistant", "content": response})
        return response

    def get_messages(self):
        """Build full message list"""
        messages = [{"role": "system", "content": self.system_prompt}]
        messages.extend(self.history)
        return messages

    def clear_history(self):
        self.history = []

# Test
bot = SimpleChatbot()
print(bot.chat("Hello"))
print(bot.chat("How's the weather today?"))
print(f"History length: {len(bot.history)}")


# ============================================
# 2. RAG Chatbot
# ============================================
print("\n[2] RAG Chatbot")
print("-" * 40)

import numpy as np

class RAGChatbot:
    """Document-based RAG chatbot"""

    def __init__(self, documents):
        self.documents = documents
        self.history = []
        # Simulated embeddings (in practice, use a model)
        self.embeddings = np.random.randn(len(documents), 128)

    def retrieve(self, query, top_k=2):
        """Retrieve relevant documents"""
        query_emb = np.random.randn(128)
        similarities = np.dot(self.embeddings, query_emb) / (
            np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_emb)
        )
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        return [self.documents[i] for i in top_indices]

    def chat(self, question):
        """Generate RAG-based answer"""
        # Retrieve
        relevant_docs = self.retrieve(question)
        context = "\n".join(relevant_docs)

        # Build prompt
        prompt = f"""Context:
{context}

History:
{self._format_history()}

Question: {question}

Answer:"""

        # In practice, call LLM
        response = f"[Context-based response] {relevant_docs[0][:50]}..."

        # Update history
        self.history.append({"role": "user", "content": question})
        self.history.append({"role": "assistant", "content": response})

        return response

    def _format_history(self, max_turns=3):
        recent = self.history[-max_turns*2:]
        return "\n".join([f"{m['role']}: {m['content']}" for m in recent])

# Test
documents = [
    "Python is a programming language created by Guido van Rossum.",
    "Machine learning is a type of artificial intelligence.",
    "Deep learning uses neural networks with many layers."
]

rag_bot = RAGChatbot(documents)
print(rag_bot.chat("What is Python?"))
print(rag_bot.chat("Tell me more about it"))


# ============================================
# 3. Intent Classification
# ============================================
print("\n[3] Intent Classification")
print("-" * 40)

class IntentClassifier:
    """Rule-based intent classification (in practice, use LLM)"""

    def __init__(self):
        self.intents = {
            "greeting": ["hello", "hi", "hey"],
            "goodbye": ["bye", "goodbye"],
            "help": ["help", "how do i"],
            "question": ["what", "why", "how", "when"]
        }

    def classify(self, text):
        text_lower = text.lower()
        for intent, keywords in self.intents.items():
            if any(kw in text_lower for kw in keywords):
                return intent
        return "general"

classifier = IntentClassifier()
test_texts = ["Hello!", "What is AI?", "Goodbye", "Help me please"]
for text in test_texts:
    intent = classifier.classify(text)
    print(f"  [{intent}] {text}")


# ============================================
# 4. Conversation State Management
# ============================================
print("\n[4] Conversation State Management")
print("-" * 40)

from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Any

class State(Enum):
    GREETING = "greeting"
    COLLECTING = "collecting"
    CONFIRMING = "confirming"
    DONE = "done"

@dataclass
class ConversationState:
    state: State = State.GREETING
    slots: Dict[str, Any] = field(default_factory=dict)

class StatefulBot:
    def __init__(self):
        self.context = ConversationState()
        self.required_slots = ["name", "email"]

    def process(self, message):
        if self.context.state == State.GREETING:
            self.context.state = State.COLLECTING
            return "Hello! Please tell me your name."

        elif self.context.state == State.COLLECTING:
            # Slot extraction (simple example)
            if "name" not in self.context.slots:
                self.context.slots["name"] = message
                return "Please provide your email address."
            elif "email" not in self.context.slots:
                self.context.slots["email"] = message
                self.context.state = State.CONFIRMING
                return f"Confirm: {self.context.slots}. Is this correct? (yes/no)"

        elif self.context.state == State.CONFIRMING:
            if "yes" in message.lower():
                self.context.state = State.DONE
                return "Thank you! Processing complete."
            else:
                self.context = ConversationState()
                return "Starting over. Please tell me your name."

        return "How can I help you?"

# Test
stateful_bot = StatefulBot()
print(stateful_bot.process("start"))
print(stateful_bot.process("John Doe"))
print(stateful_bot.process("john@example.com"))
print(stateful_bot.process("yes"))


# ============================================
# 5. OpenAI Chatbot (Code Example)
# ============================================
print("\n[5] OpenAI Chatbot (code)")
print("-" * 40)

openai_bot_code = '''
from openai import OpenAI

class OpenAIChatbot:
    def __init__(self, system_prompt="You are a helpful assistant."):
        self.client = OpenAI()
        self.system_prompt = system_prompt
        self.history = []

    def chat(self, message):
        # Build messages
        messages = [{"role": "system", "content": self.system_prompt}]
        messages.extend(self.history)
        messages.append({"role": "user", "content": message})

        # API call
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.7
        )

        assistant_msg = response.choices[0].message.content

        # Update history
        self.history.append({"role": "user", "content": message})
        self.history.append({"role": "assistant", "content": assistant_msg})

        return assistant_msg

    def chat_stream(self, message):
        """Streaming response"""
        messages = [{"role": "system", "content": self.system_prompt}]
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

        self.history.append({"role": "user", "content": message})
        self.history.append({"role": "assistant", "content": full_response})
'''
print(openai_bot_code)


# ============================================
# 6. FastAPI Server (Code)
# ============================================
print("\n[6] FastAPI Server (code)")
print("-" * 40)

fastapi_code = '''
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()
sessions = {}

class ChatRequest(BaseModel):
    session_id: str
    message: str

@app.post("/chat")
async def chat(request: ChatRequest):
    if request.session_id not in sessions:
        sessions[request.session_id] = OpenAIChatbot()

    bot = sessions[request.session_id]
    response = bot.chat(request.message)

    return {"response": response}

@app.delete("/session/{session_id}")
async def clear_session(session_id: str):
    if session_id in sessions:
        del sessions[session_id]
    return {"status": "cleared"}

# Run: uvicorn main:app --reload
'''
print(fastapi_code)


# ============================================
# 7. Gradio UI (Code)
# ============================================
print("\n[7] Gradio UI (code)")
print("-" * 40)

gradio_code = '''
import gradio as gr

def respond(message, history):
    # Generate chatbot response
    response = bot.chat(message)
    return response

demo = gr.ChatInterface(
    fn=respond,
    title="AI Chatbot",
    description="Ask me anything!",
    examples=["Hello!", "What is AI?"],
    theme="soft"
)

demo.launch()
'''
print(gradio_code)


# ============================================
# Summary
# ============================================
print("\n" + "=" * 60)
print("Chatbot Summary")
print("=" * 60)

summary = """
Chatbot Components:
    1. Conversation history management
    2. Intent classification
    3. Slot extraction
    4. State management
    5. RAG (document-based)
    6. LLM invocation

Key Patterns:
    # Basic conversation
    messages = [system] + history + [user_message]
    response = llm.invoke(messages)

    # RAG
    context = retrieve(query)
    response = llm.invoke(context + query)

    # Streaming
    for chunk in llm.stream(messages):
        yield chunk

Deployment:
    - FastAPI: REST API server
    - Gradio: Quick UI prototype
    - Streamlit: Dashboard style
"""
print(summary)
