# 10. LangChain Basics

> **Version Info**: This lesson is based on LangChain 0.2+ (2024~).
>
> LangChain is a rapidly evolving library. Key changes:
> - **LCEL (LangChain Expression Language)**: Recommended chain composition method
> - **langchain-core, langchain-community**: Package separation
> - **RunnableWithMessageHistory recommended over ConversationChain**
>
> Latest docs: https://python.langchain.com/docs/

## Learning Objectives

- LangChain core concepts
- LLM wrappers and prompts
- Chains and agents
- Memory systems
- LCEL (LangChain Expression Language) deep dive
- LangGraph basics

---

## 1. LangChain Overview

### Installation

```bash
# LangChain 0.2+
pip install langchain langchain-openai langchain-community
```

### Core Components

```
LangChain
├── Models          # LLM wrappers
├── Prompts         # Prompt templates
├── Chains          # Sequential calls
├── Agents          # Tool-using agents
├── Memory          # Conversation history
├── Retrievers      # Document retrieval
└── Callbacks       # Monitoring
```

---

## 2. LLM Wrappers

### ChatOpenAI

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.7,
    max_tokens=500
)

# Simple call
response = llm.invoke("What is the capital of France?")
print(response.content)
```

### Various LLMs

```python
# OpenAI
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4")

# Anthropic
from langchain_anthropic import ChatAnthropic
llm = ChatAnthropic(model="claude-3-opus-20240229")

# HuggingFace
from langchain_huggingface import HuggingFaceEndpoint
llm = HuggingFaceEndpoint(repo_id="mistralai/Mistral-7B-Instruct-v0.1")

# Ollama (local)
from langchain_community.llms import Ollama
llm = Ollama(model="llama2")
```

### Message Types

```python
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="What is 2+2?"),
]

response = llm.invoke(messages)
print(response.content)
```

---

## 3. Prompt Templates

### Basic Template

```python
from langchain_core.prompts import PromptTemplate

template = PromptTemplate(
    input_variables=["topic"],
    template="Write a short poem about {topic}."
)

prompt = template.format(topic="spring")
response = llm.invoke(prompt)
```

### Chat Prompts

```python
from langchain_core.prompts import ChatPromptTemplate

template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that translates {input_language} to {output_language}."),
    ("human", "{text}")
])

messages = template.format_messages(
    input_language="English",
    output_language="Korean",
    text="Hello, how are you?"
)

response = llm.invoke(messages)
```

### Few-shot Prompts

```python
from langchain_core.prompts import FewShotPromptTemplate

examples = [
    {"input": "happy", "output": "sad"},
    {"input": "tall", "output": "short"},
    {"input": "hot", "output": "cold"},
]

example_template = PromptTemplate(
    input_variables=["input", "output"],
    template="Input: {input}\nOutput: {output}"
)

few_shot_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_template,
    prefix="Give the antonym of every input:",
    suffix="Input: {word}\nOutput:",
    input_variables=["word"]
)

prompt = few_shot_prompt.format(word="big")
```

---

## 4. Chains

### LCEL (LangChain Expression Language)

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

# Compose chain
prompt = ChatPromptTemplate.from_template("Tell me a joke about {topic}")
llm = ChatOpenAI(model="gpt-3.5-turbo")
output_parser = StrOutputParser()

# Connect with pipe operator
chain = prompt | llm | output_parser

# Execute
result = chain.invoke({"topic": "programmers"})
print(result)
```

### Sequential Chain

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

# First chain: generate topic
topic_prompt = ChatPromptTemplate.from_template(
    "Generate a random topic for a story."
)

# Second chain: write story
story_prompt = ChatPromptTemplate.from_template(
    "Write a short story about: {topic}"
)

# Connect chains
chain = (
    {"topic": topic_prompt | llm | StrOutputParser()}
    | story_prompt
    | llm
    | StrOutputParser()
)

result = chain.invoke({})
```

### Parallel Chains

```python
from langchain_core.runnables import RunnableParallel

# Parallel execution
parallel_chain = RunnableParallel(
    summary=summary_chain,
    keywords=keyword_chain,
    sentiment=sentiment_chain
)

results = parallel_chain.invoke({"text": "Long article here..."})
# {'summary': '...', 'keywords': '...', 'sentiment': '...'}
```

---

## 5. Output Parsers

### String Parser

```python
from langchain_core.output_parsers import StrOutputParser

parser = StrOutputParser()
chain = prompt | llm | parser  # Convert to string
```

### JSON Parser

```python
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

class Person(BaseModel):
    name: str = Field(description="The person's name")
    age: int = Field(description="The person's age")

parser = JsonOutputParser(pydantic_object=Person)

prompt = ChatPromptTemplate.from_messages([
    ("system", "Extract person info. {format_instructions}"),
    ("human", "{text}")
]).partial(format_instructions=parser.get_format_instructions())

chain = prompt | llm | parser
result = chain.invoke({"text": "John is 25 years old"})
# {'name': 'John', 'age': 25}
```

### Structured Output

```python
from langchain_core.output_parsers import PydanticOutputParser

class MovieReview(BaseModel):
    title: str
    rating: int
    summary: str

parser = PydanticOutputParser(pydantic_object=MovieReview)
```

---

## 6. Agents

### Basic Agent

```python
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub
from langchain_community.tools import DuckDuckGoSearchRun

# Define tools
search = DuckDuckGoSearchRun()
tools = [search]

# Load ReAct prompt
prompt = hub.pull("hwchase17/react")

# Create agent
agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Execute
result = agent_executor.invoke({"input": "What is the weather in Seoul?"})
```

### Custom Tools

```python
from langchain.tools import tool

@tool
def calculate(expression: str) -> str:
    """Calculate a mathematical expression."""
    try:
        return str(eval(expression))
    except:
        return "Error in calculation"

@tool
def get_current_time() -> str:
    """Get the current time."""
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

tools = [calculate, get_current_time]
```

### Tool Class

```python
from langchain.tools import BaseTool
from typing import Optional
from pydantic import Field

class SearchTool(BaseTool):
    name: str = "search"
    description: str = "Search for information on the internet"

    def _run(self, query: str) -> str:
        # Search logic
        return f"Search results for: {query}"

    async def _arun(self, query: str) -> str:
        return self._run(query)
```

---

## 7. Memory

> **Recommended Approach Changed**: In LangChain 0.2+, `ConversationChain`, `ConversationBufferMemory`, etc. are
> deprecated. For new projects, use **RunnableWithMessageHistory** (see below).

### (Legacy) Conversation Buffer Memory

> ⚠️ **Deprecated**: Use `RunnableWithMessageHistory` in "Memory with LCEL" section below

```python
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

memory = ConversationBufferMemory()

conversation = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=True
)

# Conversation
response1 = conversation.predict(input="Hi, I'm John")
response2 = conversation.predict(input="What's my name?")
# "Your name is John"
```

### (Legacy) Summary Memory

```python
from langchain.memory import ConversationSummaryMemory

memory = ConversationSummaryMemory(llm=llm)

# Summarize and store long conversations
```

### (Legacy) Window Memory

```python
from langchain.memory import ConversationBufferWindowMemory

# Keep only recent k conversations
memory = ConversationBufferWindowMemory(k=5)
```

### Memory in LCEL (Recommended)

```python
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

store = {}

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

chain_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history"
)

# Usage
response = chain_with_history.invoke(
    {"input": "What is my name?"},
    config={"configurable": {"session_id": "user123"}}
)
```

---

## 8. RAG with LangChain

### Document Loaders

```python
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    WebBaseLoader
)

# Text file
loader = TextLoader("document.txt")
docs = loader.load()

# PDF
loader = PyPDFLoader("document.pdf")
docs = loader.load()

# Web page
loader = WebBaseLoader("https://example.com")
docs = loader.load()
```

### Text Splitting

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = splitter.split_documents(docs)
```

### Vector Stores

```python
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

# Search
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
docs = retriever.invoke("What is machine learning?")
```

### RAG Chain

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

template = """Answer based on the context:
Context: {context}
Question: {question}
Answer:"""

prompt = ChatPromptTemplate.from_template(template)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

result = rag_chain.invoke("What is machine learning?")
```

---

## 9. Streaming

```python
# Streaming output
for chunk in chain.stream({"topic": "AI"}):
    print(chunk, end="", flush=True)

# Async streaming
async for chunk in chain.astream({"topic": "AI"}):
    print(chunk, end="", flush=True)
```

---

## 10. LCEL (LangChain Expression Language) Deep Dive

LCEL is the recommended way to build chains in LangChain 0.2+. It provides a declarative, composable syntax for building complex LLM applications.

### Pipe Operator for Chain Composition

The pipe operator (`|`) connects components in a left-to-right flow:

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

# Each component is a "Runnable"
prompt = ChatPromptTemplate.from_template("Tell me a joke about {topic}")
llm = ChatOpenAI(model="gpt-3.5-turbo")
output_parser = StrOutputParser()

# Compose with pipe operator
chain = prompt | llm | output_parser

# Execute
result = chain.invoke({"topic": "programmers"})
```

### Core Runnable Components

#### RunnablePassthrough

Passes input through unchanged, useful for routing data:

```python
from langchain_core.runnables import RunnablePassthrough

# Pass through the entire input
chain = RunnablePassthrough() | llm

# Pass through specific field
chain = {"text": RunnablePassthrough()} | prompt | llm
```

#### RunnableParallel

Executes multiple chains in parallel:

```python
from langchain_core.runnables import RunnableParallel

summary_chain = summary_prompt | llm | StrOutputParser()
keyword_chain = keyword_prompt | llm | StrOutputParser()
sentiment_chain = sentiment_prompt | llm | StrOutputParser()

# Run all three chains in parallel
parallel_chain = RunnableParallel(
    summary=summary_chain,
    keywords=keyword_chain,
    sentiment=sentiment_chain
)

results = parallel_chain.invoke({"text": "Long article text here..."})
# {'summary': '...', 'keywords': [...], 'sentiment': 'positive'}
```

#### RunnableLambda

Wraps arbitrary functions as Runnables:

```python
from langchain_core.runnables import RunnableLambda

def extract_text(data):
    """Extract text field from input."""
    return data["text"].upper()

chain = RunnableLambda(extract_text) | llm
result = chain.invoke({"text": "hello world"})
```

### Streaming with LCEL

LCEL supports multiple streaming modes:

```python
# Synchronous streaming
for chunk in chain.stream({"topic": "AI"}):
    print(chunk, end="", flush=True)

# Async streaming
async for chunk in chain.astream({"topic": "AI"}):
    print(chunk, end="", flush=True)

# Stream events (detailed streaming)
async for event in chain.astream_events({"topic": "AI"}, version="v1"):
    kind = event["event"]
    if kind == "on_chat_model_stream":
        print(event["data"]["chunk"].content, end="", flush=True)
```

### Comparison: Old Chain Style vs LCEL

#### Old Style (Deprecated)

```python
from langchain.chains import LLMChain

# Old approach
chain = LLMChain(llm=llm, prompt=prompt)
result = chain.run(topic="AI")
```

#### LCEL Style (Recommended)

```python
# LCEL approach
chain = prompt | llm | StrOutputParser()
result = chain.invoke({"topic": "AI"})
```

**Benefits of LCEL:**
- **Composability**: Easily combine and reuse components
- **Streaming**: Built-in support for streaming outputs
- **Async**: First-class async support
- **Parallelization**: Automatic parallel execution where possible
- **Type safety**: Better IDE support and error messages

### Example: RAG Chain Using LCEL

```python
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# Setup
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(documents, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Prompt template
template = """Answer the question based on the following context:

Context: {context}

Question: {question}

Answer:"""
prompt = ChatPromptTemplate.from_template(template)

# Helper function
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# LCEL-style RAG chain
rag_chain = (
    RunnableParallel(
        context=retriever | format_docs,
        question=RunnablePassthrough()
    )
    | prompt
    | llm
    | StrOutputParser()
)

# Execute
answer = rag_chain.invoke("What is machine learning?")

# Stream the answer
for chunk in rag_chain.stream("What is deep learning?"):
    print(chunk, end="", flush=True)
```

### Advanced: Branching and Routing

```python
from langchain_core.runnables import RunnableBranch

# Route based on input
branch = RunnableBranch(
    (lambda x: "code" in x["topic"], code_chain),
    (lambda x: "math" in x["topic"], math_chain),
    default_chain  # default
)

chain = {"topic": RunnablePassthrough()} | branch | llm
```

---

## 11. LangGraph Basics

**LangGraph** is a library for building stateful, multi-agent applications with LLMs. It extends LangChain with graph-based workflows.

### What is LangGraph?

LangGraph allows you to define applications as graphs where:
- **Nodes** are functions (LLM calls, tool usage, custom logic)
- **Edges** define the flow between nodes
- **State** is maintained throughout the graph execution

**When to use LangGraph (vs Chains):**

| Use Chains (LCEL) | Use LangGraph |
|-------------------|---------------|
| Linear workflows | Cycles, loops |
| Simple branching | Complex routing |
| Stateless | Stateful agents |
| Single agent | Multi-agent systems |

### Installation

```bash
pip install langgraph
```

### StateGraph Concept

LangGraph uses a `StateGraph` that maintains state as it flows through nodes:

```python
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph, END

# Define state schema
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], "The messages in the conversation"]
    next: str

# Create graph
graph = StateGraph(AgentState)
```

### Nodes and Edges

```python
from langchain_core.messages import HumanMessage, AIMessage

def agent_node(state: AgentState):
    """Agent decision node."""
    messages = state["messages"]
    response = llm.invoke(messages)
    return {"messages": messages + [response], "next": "tool"}

def tool_node(state: AgentState):
    """Tool execution node."""
    # Execute tool
    result = "Tool result here"
    return {"messages": state["messages"] + [AIMessage(content=result)], "next": END}

# Add nodes
graph.add_node("agent", agent_node)
graph.add_node("tool", tool_node)

# Add edges
graph.add_edge("agent", "tool")
graph.add_edge("tool", END)

# Set entry point
graph.set_entry_point("agent")

# Compile
app = graph.compile()

# Execute
result = app.invoke({"messages": [HumanMessage(content="Hello")]})
```

### Simple Agent with Tool Use

```python
from langgraph.graph import StateGraph, END
from langchain.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage

# Define tool
@tool
def search(query: str) -> str:
    """Search for information."""
    return f"Search results for: {query}"

tools = [search]
llm_with_tools = llm.bind_tools(tools)

# State
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], "The messages"]

# Agent node
def call_agent(state: AgentState):
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": messages + [response]}

# Tool node
def call_tool(state: AgentState):
    messages = state["messages"]
    last_message = messages[-1]

    # Execute tools
    tool_calls = last_message.tool_calls
    results = []
    for tool_call in tool_calls:
        tool_result = search.invoke(tool_call["args"])
        results.append(ToolMessage(content=tool_result, tool_call_id=tool_call["id"]))

    return {"messages": messages + results}

# Build graph
graph = StateGraph(AgentState)
graph.add_node("agent", call_agent)
graph.add_node("tools", call_tool)

# Conditional routing
def should_continue(state: AgentState):
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return END

graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
graph.add_edge("tools", "agent")
graph.set_entry_point("agent")

# Compile and run
app = graph.compile()
result = app.invoke({"messages": [HumanMessage(content="Search for LangChain news")]})

# Print conversation
for msg in result["messages"]:
    print(f"{msg.__class__.__name__}: {msg.content}")
```

### Conditional Routing

LangGraph supports conditional edges for dynamic routing:

```python
def route_decision(state: AgentState):
    """Decide next node based on state."""
    if state.get("error"):
        return "error_handler"
    elif state.get("needs_review"):
        return "review"
    else:
        return "complete"

graph.add_conditional_edges(
    "process",
    route_decision,
    {
        "error_handler": "error_handler",
        "review": "review",
        "complete": END
    }
)
```

### Visualization

LangGraph can visualize your graph:

```python
from IPython.display import Image, display

# Visualize graph structure
display(Image(app.get_graph().draw_mermaid_png()))
```

### Multi-Agent Example

```python
from langgraph.graph import StateGraph, END

class MultiAgentState(TypedDict):
    messages: Sequence[BaseMessage]
    current_agent: str

def researcher(state: MultiAgentState):
    # Research agent
    return {"messages": [...], "current_agent": "writer"}

def writer(state: MultiAgentState):
    # Writer agent
    return {"messages": [...], "current_agent": "reviewer"}

def reviewer(state: MultiAgentState):
    # Reviewer agent
    return {"messages": [...], "current_agent": END}

# Build multi-agent graph
graph = StateGraph(MultiAgentState)
graph.add_node("researcher", researcher)
graph.add_node("writer", writer)
graph.add_node("reviewer", reviewer)

graph.add_edge("researcher", "writer")
graph.add_edge("writer", "reviewer")
graph.add_edge("reviewer", END)
graph.set_entry_point("researcher")

app = graph.compile()
```

### Key LangGraph Concepts

- **Checkpointing**: Save/restore state at any point
- **Human-in-the-loop**: Pause for human approval before continuing
- **Time travel**: Replay from any checkpoint
- **Persistence**: Save conversation state to database

---

## Summary

### Core Patterns

```python
# Basic LCEL chain
chain = prompt | llm | output_parser

# RAG chain with LCEL
rag_chain = (
    RunnableParallel(context=retriever, question=RunnablePassthrough())
    | prompt | llm | parser
)

# Agent (traditional)
agent = create_react_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools)

# Agent (LangGraph)
graph = StateGraph(AgentState)
graph.add_node("agent", call_agent)
graph.add_conditional_edges("agent", should_continue)
app = graph.compile()
```

### Component Selection Guide

| Situation | Component |
|-----------|-----------|
| Simple call | LLM + Prompt |
| Sequential processing | Chain (LCEL) |
| Parallel execution | RunnableParallel |
| Document-based QA | RAG Chain (LCEL) |
| Simple tool usage | Agent (ReAct) |
| Complex workflows | LangGraph |
| Multi-agent systems | LangGraph |
| Stateful agents | LangGraph |
| Maintain conversation | RunnableWithMessageHistory |

### LCEL vs LangGraph

| Feature | LCEL | LangGraph |
|---------|------|-----------|
| **Use case** | Linear/simple branching | Cycles, complex routing |
| **State** | Stateless | Stateful |
| **Syntax** | Pipe operator (`\|`) | StateGraph |
| **Complexity** | Simple to moderate | Moderate to complex |
| **Best for** | RAG, simple agents | Multi-agent, human-in-loop |

---

## Exercises

### Exercise 1: LCEL Chain Composition

Build a two-step LCEL chain that first identifies the programming language in a code snippet, then explains what the code does — using the detected language in the second prompt. The chain should use `RunnablePassthrough` to thread the original code into the second step.

<details>
<summary>Show Answer</summary>

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# Step 1: Detect the programming language
detect_language_prompt = ChatPromptTemplate.from_template(
    "What programming language is this code written in? "
    "Reply with just the language name, nothing else.\n\nCode:\n{code}"
)

# Step 2: Explain the code using the detected language
explain_prompt = ChatPromptTemplate.from_template(
    "You are an expert {language} developer. "
    "Explain what the following code does in 2-3 sentences:\n\n{code}"
)

# Build individual steps
detect_chain = detect_language_prompt | llm | StrOutputParser()

# Two-step chain: detect language, then pass both language + original code to explain
chain = (
    RunnableParallel(
        language=detect_chain,
        code=RunnablePassthrough()  # Pass original input dict unchanged
    )
    | explain_prompt
    | llm
    | StrOutputParser()
)

# Test
code_snippet = {"code": """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
"""}

result = chain.invoke(code_snippet)
print(result)
# Expected: "This Python function computes the nth Fibonacci number using recursion.
#            It handles the base cases (n=0 returns 0, n=1 returns 1) and recursively
#            sums the two preceding Fibonacci numbers."
```

**Key concepts demonstrated:**
- `RunnableParallel` runs two sub-chains simultaneously: language detection and code passthrough
- `RunnablePassthrough()` passes the input dict unchanged so `code` is still available in step 2
- The output of `RunnableParallel` is a dict `{"language": "Python", "code": "..."}` which matches the `explain_prompt` variables
</details>

---

### Exercise 2: Pydantic Output Parser

Create a LangChain chain that extracts structured product information from unstructured text. The output should be a Pydantic model with fields: `name` (str), `price` (float), `in_stock` (bool), and `features` (list of str).

<details>
<summary>Show Answer</summary>

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from typing import List

class Product(BaseModel):
    name: str = Field(description="Product name")
    price: float = Field(description="Price in USD as a number")
    in_stock: bool = Field(description="Whether the product is in stock")
    features: List[str] = Field(description="List of key product features")

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
parser = JsonOutputParser(pydantic_object=Product)

prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Extract product information from the text and return it as JSON. "
     "{format_instructions}"),
    ("human", "{text}")
]).partial(format_instructions=parser.get_format_instructions())

chain = prompt | llm | parser

# Test with unstructured product description
text = """
The UltraBook Pro 15 is now available for $1,299.99. This laptop is in stock
and ready to ship. Key highlights include a 15-inch 4K display, 16GB RAM,
512GB NVMe SSD, and 18-hour battery life.
"""

result = chain.invoke({"text": text})
print(result)
# Expected:
# {
#   'name': 'UltraBook Pro 15',
#   'price': 1299.99,
#   'in_stock': True,
#   'features': ['15-inch 4K display', '16GB RAM', '512GB NVMe SSD', '18-hour battery life']
# }

# With Pydantic validation
product = Product(**result)
print(f"Product: {product.name}, Price: ${product.price:.2f}")
print(f"In stock: {product.in_stock}")
print(f"Features: {', '.join(product.features)}")
```

**Why this matters:** `JsonOutputParser` with a Pydantic model provides schema validation — if the LLM returns a string for `price` instead of a float, Pydantic raises a validation error. This makes the output reliable for downstream processing.
</details>

---

### Exercise 3: RunnableWithMessageHistory for Multi-turn Chat

Implement a multi-turn customer support chatbot using `RunnableWithMessageHistory`. The bot should remember the conversation context and have a system prompt that sets its role as a product support agent.

<details>
<summary>Show Answer</summary>

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)

# Prompt with conversation history placeholder
prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a helpful customer support agent for TechCorp. "
     "Be concise and friendly. Reference previous messages when relevant."),
    MessagesPlaceholder(variable_name="history"),  # Insert conversation history here
    ("human", "{input}"),
])

chain = prompt | llm | StrOutputParser()

# Session storage (in-memory; use Redis/DB in production)
store = {}

def get_session_history(session_id: str) -> ChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

chatbot = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)

# Simulate a multi-turn conversation
session = {"configurable": {"session_id": "user_42"}}

# Turn 1
r1 = chatbot.invoke({"input": "My order #12345 hasn't arrived yet."}, config=session)
print(f"Bot: {r1}")

# Turn 2 — bot should remember the order number
r2 = chatbot.invoke({"input": "Can you track it?"}, config=session)
print(f"Bot: {r2}")

# Turn 3 — bot should still know the context
r3 = chatbot.invoke({"input": "What if it's lost?"}, config=session)
print(f"Bot: {r3}")

# Inspect stored history
history = get_session_history("user_42")
print(f"\nTotal messages stored: {len(history.messages)}")
# Should be 6 (3 human + 3 AI messages)
```

**Key points:**
- `MessagesPlaceholder` injects all previous messages at that position in the prompt
- `session_id` allows multiple isolated conversations; each user gets their own history
- In production, replace `ChatMessageHistory` with a persistent store (Redis, PostgreSQL) so history survives server restarts
- The history grows with every turn — use `ConversationSummaryMemory` pattern for very long conversations
</details>

---

### Exercise 4: LangChain vs LangGraph Selection

For each application scenario below, decide whether to use LCEL chains or LangGraph. Justify your choice and sketch the high-level architecture.

| Scenario | LCEL or LangGraph? | Why? |
|----------|-------------------|------|
| A. Summarize a PDF document | ? | ? |
| B. An agent that browses the web until it finds an answer | ? | ? |
| C. Translate text to 5 languages in parallel | ? | ? |
| D. A code review pipeline that loops until all tests pass | ? | ? |

<details>
<summary>Show Answer</summary>

| Scenario | Choice | Justification |
|----------|--------|---------------|
| A. Summarize a PDF | **LCEL** | Linear workflow: load → split → summarize. No cycles or complex state needed. `chain = loader | splitter | summarize_prompt | llm | parser` |
| B. Web browsing agent | **LangGraph** | Requires cycles (search → evaluate → search again if needed). State must persist across iterations. Conditional edges decide when to stop. |
| C. Parallel translation | **LCEL** | `RunnableParallel` handles this perfectly: `{"fr": fr_chain, "de": de_chain, "ja": ja_chain, "ko": ko_chain, "es": es_chain}` — all in one call. |
| D. Code review loop | **LangGraph** | Requires a cycle: review → run tests → if tests fail, route back to review. Stateful graph tracks iteration count and test results. |

**LangGraph architecture for scenario B (Web Browsing Agent):**

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, List

class SearchState(TypedDict):
    question: str
    search_results: List[str]
    answer: str
    iterations: int

def search_node(state: SearchState) -> SearchState:
    """Perform web search."""
    results = web_search(state["question"])
    return {**state, "search_results": results, "iterations": state["iterations"] + 1}

def evaluate_node(state: SearchState) -> SearchState:
    """Evaluate if answer is found."""
    answer = llm.invoke(f"Based on {state['search_results']}, answer: {state['question']}")
    return {**state, "answer": answer.content}

def should_continue(state: SearchState) -> str:
    """Decide whether to search again or finish."""
    if "I don't know" in state["answer"] and state["iterations"] < 3:
        return "search"  # Loop back
    return END          # Done

graph = StateGraph(SearchState)
graph.add_node("search", search_node)
graph.add_node("evaluate", evaluate_node)
graph.add_edge("search", "evaluate")
graph.add_conditional_edges("evaluate", should_continue, {"search": "search", END: END})
graph.set_entry_point("search")
app = graph.compile()
```
</details>

---

## Next Steps

Learn about vector databases in [11_Vector_Databases.md](./11_Vector_Databases.md).
