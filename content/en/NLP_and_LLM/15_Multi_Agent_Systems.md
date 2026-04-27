# 15. Multi-Agent Systems

## Learning Objectives

- Understand multi-agent architectures and orchestration patterns
- Implement supervisor, sequential, and parallel agent topologies
- Build inter-agent communication with shared memory and state
- Use frameworks like CrewAI, AutoGen, and LangGraph for multi-agent workflows
- Design and deploy a practical multi-agent RAG system

---

## 1. Multi-Agent Architecture Overview

### Theory: Why Decompose

Two arguments for splitting one agent into many:

**A.1 Specialization via prompting.** Different agents get different system prompts: "You are a careful fact-checker", "You are a creative writer", "You are a code reviewer". Each prompt steers the LLM into a different region of its learned distribution. Each agent excels at *its* role and is freed from compromising for others.

**A.2 Cognitive division of labor.** A complex task often has structurally different sub-tasks (search, draft, critique, revise). Asking one prompt to do all four sequentially in one message dilutes attention. Splitting them into separate calls — each with full context budget for its own task — improves quality.

Counter-arguments (when single-agent wins): the task is simple, communication overhead exceeds the specialization benefit, the same LLM is doing all the "agents" so the diversity is illusory. Multi-agent is not free — it multiplies token cost by the number of agents and adds coordination cost.

### Why Multi-Agent?

> **Single Agent vs Multi-Agent**
>
> - **Single Agent**: One LLM handles all reasoning, tool use, and output generation
> - **Multi-Agent**: Multiple specialized agents collaborate, each with focused capabilities
> - **Key Benefit**: Decomposition of complex tasks into manageable sub-tasks with domain-specific expertise

### Architecture Comparison

| Pattern | Description | Use Case | Complexity |
|---------|-------------|----------|------------|
| Single Agent | One LLM, multiple tools | Simple Q&A, basic RAG | Low |
| Sequential Pipeline | Agents execute in order | Content pipeline, ETL | Medium |
| Supervisor (Hub-and-Spoke) | Orchestrator delegates to workers | Complex research tasks | Medium-High |
| Parallel Fan-Out | Multiple agents run simultaneously | Multi-source search | Medium |
| Hierarchical | Nested supervisors | Enterprise workflows | High |
| Debate/Consensus | Agents argue and converge | Fact-checking, review | High |

### Core Components

```python
from dataclasses import dataclass, field
from typing import Any
from enum import Enum

class AgentRole(Enum):
    RESEARCHER = "researcher"
    WRITER = "writer"
    REVIEWER = "reviewer"
    CODER = "coder"
    SUPERVISOR = "supervisor"

@dataclass
class AgentMessage:
    """Message passed between agents."""
    sender: str
    receiver: str
    content: str
    metadata: dict = field(default_factory=dict)
    message_type: str = "task"  # task, result, feedback, error

@dataclass
class AgentState:
    """Shared state accessible by all agents in a workflow."""
    messages: list[AgentMessage] = field(default_factory=list)
    context: dict[str, Any] = field(default_factory=dict)
    current_step: int = 0
    is_complete: bool = False
    error: str | None = None

    def add_message(self, msg: AgentMessage):
        self.messages.append(msg)

    def get_messages_for(self, agent_name: str) -> list[AgentMessage]:
        return [m for m in self.messages if m.receiver == agent_name]

    def get_latest_result(self) -> str | None:
        results = [m for m in self.messages if m.message_type == "result"]
        return results[-1].content if results else None
```

---

## 2. Orchestration Patterns

### Theory: Topologies

The topology defines the communication graph between agents.

**B.1 Supervisor (manager-workers).** A central supervisor agent reads the task, decides which worker to invoke, collects results, decides next step. Workers don't talk to each other. Simple, debuggable, easy to add new workers.

**B.2 Sequential (pipeline).** Agent A's output is Agent B's input, then C, etc. Pure chain. Useful when sub-tasks are dependent (research → outline → write → edit). Limited because B cannot ask A to clarify.

**B.3 Parallel (broadcast / map).** All agents work on the same input simultaneously, then results are aggregated. Useful for ensembles (3 agents draft, 1 agent picks the best) or independent decomposition (extract entities + extract relations + extract dates from same doc).

**B.4 Hierarchical.** Trees of supervisors managing sub-supervisors managing workers. Used when the task itself decomposes hierarchically.

**B.5 Network (peer-to-peer).** Any agent can talk to any other. Most flexible, most chaotic. Used in debate setups (Liang et al., 2023) where agents iteratively critique each other.

The rule of thumb: **start with supervisor, add complexity only when needed.** Most production multi-agent systems are supervisor-routed.

### Theory: Coordination Protocols

Who decides who acts next:

**D.1 Round-robin.** Fixed turn order. Simple but inefficient — agents speak when they have nothing to add.

**D.2 Supervisor-routed.** A supervisor agent (or LLM) examines state and picks the next agent. Most common in production. Cost: one extra LLM call per step for routing.

**D.3 Voting / consensus.** Each agent proposes; a vote determines the next action. Used when multiple agents have legitimate competing answers (e.g., self-consistency at agent scale).

**D.4 Debate / critique.** Agents take adversarial roles ("argue for", "argue against") and refine through exchange. Empirically improves accuracy on hard reasoning tasks (Du et al., 2023, "Improving Factuality and Reasoning in Language Models through Multiagent Debate"). Cost is proportional to the number of debate rounds.

### Sequential Pipeline

Agents execute in a fixed order, each transforming or enriching the output of the previous agent.

```python
from openai import OpenAI

client = OpenAI()

def create_agent(name: str, system_prompt: str):
    """Factory for creating specialized agents."""
    def agent(input_text: str) -> str:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": input_text},
            ],
            temperature=0.3,
        )
        return response.choices[0].message.content
    agent.__name__ = name
    return agent

# Define specialized agents
researcher = create_agent(
    "researcher",
    "You are a research agent. Given a topic, produce a structured outline "
    "with key facts, statistics, and references. Output as a numbered list."
)

writer = create_agent(
    "writer",
    "You are a writing agent. Given a research outline, write a well-structured "
    "article with an introduction, body sections, and conclusion. "
    "Maintain a professional yet accessible tone."
)

editor = create_agent(
    "editor",
    "You are an editing agent. Review the article for clarity, grammar, "
    "factual consistency, and flow. Return the improved version with "
    "a brief editorial note at the top listing changes made."
)

def sequential_pipeline(topic: str) -> dict[str, str]:
    """Run agents in sequence: Researcher -> Writer -> Editor."""
    results = {}

    print("[1/3] Researching...")
    research = researcher(f"Research the following topic thoroughly: {topic}")
    results["research"] = research

    print("[2/3] Writing...")
    article = writer(f"Write an article based on this research:\n\n{research}")
    results["draft"] = article

    print("[3/3] Editing...")
    final = editor(f"Edit and improve this article:\n\n{article}")
    results["final"] = final

    return results

# Usage
results = sequential_pipeline("The impact of transformer architecture on NLP")
print(results["final"][:500])
```

### Supervisor Pattern

A central orchestrator decides which agent to invoke next based on the current state.

```python
import json

class SupervisorAgent:
    """Orchestrator that delegates tasks to specialized workers."""

    def __init__(self, workers: dict[str, callable]):
        self.workers = workers
        self.client = OpenAI()
        self.worker_descriptions = {
            name: f"Agent '{name}' - available for delegation"
            for name in workers
        }

    def _decide_next_action(self, task: str, history: list[dict]) -> dict:
        """Use LLM to decide which worker to invoke next."""
        worker_list = "\n".join(
            f"- {name}" for name in self.workers
        )
        history_text = "\n".join(
            f"Step {i+1}: [{h['worker']}] {h['result'][:200]}..."
            for i, h in enumerate(history)
        ) if history else "No steps completed yet."

        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": f"""You are a supervisor agent.
Your job is to break down tasks and delegate to available workers.

Available workers:
{worker_list}

Work completed so far:
{history_text}

Respond with JSON: {{"worker": "<name>", "sub_task": "<instruction>", "is_final": false}}
Or if the task is complete: {{"worker": "none", "summary": "<final answer>", "is_final": true}}"""},
                {"role": "user", "content": task},
            ],
            response_format={"type": "json_object"},
            temperature=0.2,
        )
        return json.loads(response.choices[0].message.content)

    def run(self, task: str, max_steps: int = 10) -> str:
        """Execute the supervisor loop."""
        history = []

        for step in range(max_steps):
            decision = self._decide_next_action(task, history)

            if decision.get("is_final"):
                return decision.get("summary", "Task completed.")

            worker_name = decision["worker"]
            sub_task = decision["sub_task"]

            if worker_name not in self.workers:
                history.append({
                    "worker": worker_name,
                    "result": f"Error: Unknown worker '{worker_name}'"
                })
                continue

            print(f"  Step {step+1}: Delegating to [{worker_name}]: {sub_task[:80]}...")
            result = self.workers[worker_name](sub_task)
            history.append({"worker": worker_name, "result": result})

        return f"Supervisor completed {max_steps} steps. Last result: {history[-1]['result']}"

# Usage
supervisor = SupervisorAgent(workers={
    "researcher": researcher,
    "writer": writer,
    "editor": editor,
})

result = supervisor.run(
    "Write a technical blog post about vector databases for LLM applications"
)
```

### Parallel Fan-Out

Multiple agents run concurrently and results are aggregated.

```python
import asyncio
from openai import AsyncOpenAI

async_client = AsyncOpenAI()

async def async_agent(name: str, system_prompt: str, input_text: str) -> dict:
    """Async agent for parallel execution."""
    response = await async_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": input_text},
        ],
        temperature=0.3,
    )
    return {"agent": name, "result": response.choices[0].message.content}

async def parallel_research(query: str) -> list[dict]:
    """Fan-out to multiple research agents, then aggregate."""
    agents = [
        ("technical_analyst", "Analyze from a technical perspective. Focus on architecture and implementation."),
        ("business_analyst", "Analyze from a business perspective. Focus on market impact and ROI."),
        ("risk_analyst", "Analyze risks, limitations, and potential failure modes."),
    ]

    tasks = [
        async_agent(name, prompt, query)
        for name, prompt in agents
    ]

    # Run all agents in parallel
    results = await asyncio.gather(*tasks)
    return results

async def fan_out_then_aggregate(query: str) -> str:
    """Full fan-out/fan-in pattern with aggregation."""
    # Fan-out: parallel research
    research_results = await parallel_research(query)

    # Fan-in: aggregate with a synthesizer agent
    combined = "\n\n".join(
        f"=== {r['agent']} ===\n{r['result']}"
        for r in research_results
    )

    synthesizer_response = await async_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": (
                "Synthesize the following analyses into a single coherent report. "
                "Highlight agreements, disagreements, and key takeaways."
            )},
            {"role": "user", "content": combined},
        ],
        temperature=0.3,
    )

    return synthesizer_response.choices[0].message.content

# Usage
result = asyncio.run(fan_out_then_aggregate("Should we adopt LLMs for customer support?"))
```

---

## 3. Inter-Agent Communication

### Theory: Communication Patterns

How agents exchange information:

**C.1 Shared memory.** A blackboard data structure all agents read/write. Simple but fragile (concurrent writes, ambiguous ownership). Common in research prototypes.

**C.2 Message passing.** Explicit "from A to B" messages with structured payloads. Maps cleanly to function calling at the framework level. Standard in production frameworks.

**C.3 Broadcast.** Supervisor sends to all workers; all workers can read each other's prior messages (via the supervisor's collected state).

**C.4 Tool-call as message.** Treat agent invocations as tool calls — Agent A "calls" Agent B with arguments, gets a return value. Convenient because frameworks already handle tool calls; turns multi-agent into a recursion of single-agent.

### Message Passing Protocol

```python
from collections import defaultdict
from typing import Callable
import threading
import queue

class MessageBus:
    """Central message bus for agent communication."""

    def __init__(self):
        self._queues: dict[str, queue.Queue] = defaultdict(queue.Queue)
        self._subscribers: dict[str, list[Callable]] = defaultdict(list)
        self._lock = threading.Lock()

    def send(self, message: AgentMessage):
        """Send a message to a specific agent."""
        with self._lock:
            self._queues[message.receiver].put(message)
            # Notify subscribers
            for callback in self._subscribers.get(message.receiver, []):
                callback(message)

    def receive(self, agent_name: str, timeout: float = 30.0) -> AgentMessage | None:
        """Blocking receive for an agent."""
        try:
            return self._queues[agent_name].get(timeout=timeout)
        except queue.Empty:
            return None

    def subscribe(self, agent_name: str, callback: Callable):
        """Register a callback for when an agent receives a message."""
        self._subscribers[agent_name].append(callback)

    def broadcast(self, sender: str, content: str, exclude: set[str] | None = None):
        """Send a message to all registered agents."""
        exclude = exclude or set()
        for agent_name in self._queues:
            if agent_name not in exclude and agent_name != sender:
                self.send(AgentMessage(
                    sender=sender,
                    receiver=agent_name,
                    content=content,
                    message_type="broadcast",
                ))
```

### Shared Memory Store

```python
import time
from dataclasses import dataclass

@dataclass
class MemoryEntry:
    key: str
    value: Any
    author: str
    timestamp: float
    ttl: float | None = None  # Time-to-live in seconds

class SharedMemory:
    """Thread-safe shared memory for multi-agent systems."""

    def __init__(self):
        self._store: dict[str, MemoryEntry] = {}
        self._lock = threading.RLock()
        self._history: list[tuple[str, str, str]] = []  # (action, key, agent)

    def write(self, key: str, value: Any, author: str, ttl: float | None = None):
        """Write a value to shared memory."""
        with self._lock:
            self._store[key] = MemoryEntry(
                key=key, value=value, author=author,
                timestamp=time.time(), ttl=ttl,
            )
            self._history.append(("write", key, author))

    def read(self, key: str) -> Any | None:
        """Read a value from shared memory, respecting TTL."""
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            if entry.ttl and (time.time() - entry.timestamp) > entry.ttl:
                del self._store[key]
                return None
            return entry.value

    def read_all(self) -> dict[str, Any]:
        """Read all non-expired entries."""
        with self._lock:
            result = {}
            expired_keys = []
            for key, entry in self._store.items():
                if entry.ttl and (time.time() - entry.timestamp) > entry.ttl:
                    expired_keys.append(key)
                else:
                    result[key] = entry.value
            for k in expired_keys:
                del self._store[k]
            return result

    def get_history(self, agent: str | None = None) -> list[tuple[str, str, str]]:
        """Get action history, optionally filtered by agent."""
        if agent:
            return [(a, k, ag) for a, k, ag in self._history if ag == agent]
        return list(self._history)
```

### Communication Patterns

| Pattern | Description | Pros | Cons |
|---------|-------------|------|------|
| Direct Messaging | Agent-to-agent via message bus | Low latency, simple | Tight coupling |
| Shared Memory | Read/write to central store | Decoupled, persistent | Race conditions |
| Blackboard | Agents post/read from shared board | Flexible, extensible | Ordering issues |
| Pub/Sub | Topic-based message subscription | Scalable, loose coupling | Message loss risk |
| Request/Reply | Synchronous ask-and-answer | Clear flow | Blocking |

---

## 4. CrewAI Framework

### Theory: Framework Comparison

**F.1 CrewAI** (role-based). Agents are defined by Role, Goal, Backstory. Tasks are assigned to agents; the framework orchestrates. Closest to "human team" mental model. Best for well-defined collaborative workflows.

**F.2 AutoGen** (conversational). Agents are defined by their conversational behavior (who they reply to, what they say). Multi-agent conversations are first-class. Best for research-y settings, debate-style protocols.

**F.3 LangGraph** (state-machine). The system is a graph of nodes (agents or functions) connected by conditional edges. State explicitly modeled. Best for complex flows with branching, loops, human-in-the-loop. Most production-grade.

**F.4 Hand-rolled.** A loop with `if/elif` over agent names. Often the right choice for narrow, well-understood production systems — frameworks add observability and tooling but also abstraction debt.

### Overview

CrewAI provides a high-level abstraction for defining agents, tasks, and crews (teams of agents).

```python
from crewai import Agent, Task, Crew, Process
from crewai.tools import tool

# Define custom tools
@tool
def search_papers(query: str) -> str:
    """Search for academic papers on a given topic."""
    # In production, this would call an API like Semantic Scholar
    return f"Found 5 papers related to '{query}': [Paper1: ..., Paper2: ..., ...]"

@tool
def analyze_code(code: str) -> str:
    """Analyze Python code for quality and correctness."""
    return f"Code analysis: {len(code.split(chr(10)))} lines, no critical issues found."

# Define agents with roles and goals
research_agent = Agent(
    role="Senior Research Analyst",
    goal="Discover and synthesize the latest findings in AI research",
    backstory=(
        "You are a seasoned AI researcher with a PhD in machine learning. "
        "You excel at finding relevant papers and extracting key insights."
    ),
    tools=[search_papers],
    verbose=True,
    llm="gpt-4o",
)

writer_agent = Agent(
    role="Technical Writer",
    goal="Transform research findings into clear, engaging technical content",
    backstory=(
        "You are an experienced technical writer who specializes in making "
        "complex AI topics accessible to software engineers."
    ),
    verbose=True,
    llm="gpt-4o",
)

reviewer_agent = Agent(
    role="Technical Reviewer",
    goal="Ensure accuracy, completeness, and clarity of technical content",
    backstory=(
        "You are a meticulous reviewer with deep expertise in AI/ML. "
        "You catch technical errors and suggest improvements."
    ),
    verbose=True,
    llm="gpt-4o",
)

# Define tasks
research_task = Task(
    description=(
        "Research the latest developments in {topic}. "
        "Find at least 3 key papers or developments from the past year. "
        "Provide a structured summary with key findings."
    ),
    expected_output="A structured research summary with 3+ key findings and references.",
    agent=research_agent,
)

writing_task = Task(
    description=(
        "Based on the research findings, write a technical blog post. "
        "Include an introduction, 3-4 main sections, code examples where "
        "relevant, and a conclusion with future directions."
    ),
    expected_output="A complete technical blog post, 1000-1500 words.",
    agent=writer_agent,
    context=[research_task],  # Depends on research output
)

review_task = Task(
    description=(
        "Review the blog post for technical accuracy, clarity, and completeness. "
        "Provide specific feedback and a revised version."
    ),
    expected_output="Reviewed blog post with editorial comments and improvements.",
    agent=reviewer_agent,
    context=[writing_task],
)

# Create and run the crew
crew = Crew(
    agents=[research_agent, writer_agent, reviewer_agent],
    tasks=[research_task, writing_task, review_task],
    process=Process.sequential,
    verbose=True,
)

result = crew.kickoff(inputs={"topic": "multi-agent LLM systems"})
print(result)
```

### CrewAI Hierarchical Process

```python
# Hierarchical process with a manager agent
manager_crew = Crew(
    agents=[research_agent, writer_agent, reviewer_agent],
    tasks=[research_task, writing_task, review_task],
    process=Process.hierarchical,
    manager_llm="gpt-4o",
    verbose=True,
)

# The manager automatically delegates and coordinates
result = manager_crew.kickoff(inputs={"topic": "RAG optimization techniques"})
```

---

## 5. AutoGen Framework

### Conversable Agents

```python
from autogen import ConversableAgent, GroupChat, GroupChatManager

# Configuration for the LLM
llm_config = {
    "config_list": [{"model": "gpt-4o", "api_key": "your-key"}],
    "temperature": 0.3,
}

# Define conversable agents
planner = ConversableAgent(
    name="Planner",
    system_message=(
        "You are a project planner. Break down tasks into clear steps. "
        "Assign work to the coder or analyst as appropriate. "
        "When all tasks are complete, say 'TERMINATE'."
    ),
    llm_config=llm_config,
)

coder = ConversableAgent(
    name="Coder",
    system_message=(
        "You are an expert Python programmer. Write clean, well-documented code. "
        "Always include error handling and type hints. "
        "Test your code mentally before presenting it."
    ),
    llm_config=llm_config,
)

analyst = ConversableAgent(
    name="Analyst",
    system_message=(
        "You are a data analyst. Interpret results, create visualizations, "
        "and provide statistical insights. Focus on actionable conclusions."
    ),
    llm_config=llm_config,
)

critic = ConversableAgent(
    name="Critic",
    system_message=(
        "You are a code reviewer and quality analyst. Review code for bugs, "
        "security issues, and performance. Review analyses for statistical validity. "
        "Be constructive but thorough."
    ),
    llm_config=llm_config,
)

# Group chat with automatic speaker selection
group_chat = GroupChat(
    agents=[planner, coder, analyst, critic],
    messages=[],
    max_round=15,
    speaker_selection_method="auto",  # LLM decides who speaks next
)

manager = GroupChatManager(
    groupchat=group_chat,
    llm_config=llm_config,
)

# Initiate the conversation
planner.initiate_chat(
    manager,
    message=(
        "Analyze a CSV dataset of customer support tickets. "
        "Write Python code to: 1) Load and clean the data, "
        "2) Classify tickets by category using keyword matching, "
        "3) Generate summary statistics and identify top issues."
    ),
)
```

### AutoGen with Code Execution

```python
from autogen import AssistantAgent, UserProxyAgent
from autogen.coding import LocalCommandLineCodeExecutor

# Agent that can write code
assistant = AssistantAgent(
    name="assistant",
    llm_config=llm_config,
    system_message="You are a helpful AI assistant that writes Python code to solve problems.",
)

# Agent with code execution capability
executor = UserProxyAgent(
    name="executor",
    human_input_mode="NEVER",
    code_execution_config={
        "executor": LocalCommandLineCodeExecutor(
            timeout=60,
            work_dir="./coding_workspace",
        ),
    },
    is_termination_msg=lambda msg: "TERMINATE" in msg.get("content", ""),
)

# The assistant writes code, the executor runs it
executor.initiate_chat(
    assistant,
    message="Create a Python script that fetches the top 10 Hacker News stories and displays them.",
)
```

---

## 6. LangGraph Multi-Agent

### Theory: State and Termination

**E.1 State sharing.** Three options: (1) full conversation broadcast (all agents see everything — high context cost), (2) summarized history (a summarizer compresses past turns periodically), (3) supervisor-mediated (workers only see what the supervisor passes to them). Production typically uses option 3 for cost control.

**E.2 Termination conditions.** When does the system stop?
- **Explicit signal** — the supervisor declares "task complete".
- **Step budget** — hard cap on total agent turns.
- **Convergence** — no agent has new information to add.
- **External** — wall-clock timeout, cost ceiling.

A robust multi-agent system needs *all four* in production: explicit primary, hard caps as fallbacks against runaway.

### State-Based Agent Graph

```python
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

class MultiAgentState(TypedDict):
    messages: Annotated[list, add_messages]
    current_agent: str
    research_done: bool
    draft_done: bool
    review_done: bool
    final_output: str

llm = ChatOpenAI(model="gpt-4o", temperature=0.3)

def router_node(state: MultiAgentState) -> MultiAgentState:
    """Supervisor that routes to the next agent."""
    if not state.get("research_done"):
        return {**state, "current_agent": "researcher"}
    elif not state.get("draft_done"):
        return {**state, "current_agent": "writer"}
    elif not state.get("review_done"):
        return {**state, "current_agent": "reviewer"}
    else:
        return {**state, "current_agent": "done"}

def researcher_node(state: MultiAgentState) -> MultiAgentState:
    """Research agent node."""
    messages = [
        SystemMessage(content=(
            "You are a research agent. Analyze the user's request and provide "
            "thorough research findings with key data points."
        )),
        *state["messages"],
    ]
    response = llm.invoke(messages)
    return {
        **state,
        "messages": [response],
        "research_done": True,
    }

def writer_node(state: MultiAgentState) -> MultiAgentState:
    """Writer agent node."""
    messages = [
        SystemMessage(content=(
            "You are a writing agent. Based on the research provided in the "
            "conversation, write a clear, well-structured document."
        )),
        *state["messages"],
    ]
    response = llm.invoke(messages)
    return {
        **state,
        "messages": [response],
        "draft_done": True,
    }

def reviewer_node(state: MultiAgentState) -> MultiAgentState:
    """Reviewer agent node."""
    messages = [
        SystemMessage(content=(
            "You are an editing agent. Review and improve the draft. "
            "Output only the final polished version."
        )),
        *state["messages"],
    ]
    response = llm.invoke(messages)
    return {
        **state,
        "messages": [response],
        "review_done": True,
        "final_output": response.content,
    }

def route_decision(state: MultiAgentState) -> str:
    """Conditional edge: decide the next node."""
    agent = state.get("current_agent", "researcher")
    if agent == "done":
        return END
    return agent

# Build the graph
graph = StateGraph(MultiAgentState)
graph.add_node("router", router_node)
graph.add_node("researcher", researcher_node)
graph.add_node("writer", writer_node)
graph.add_node("reviewer", reviewer_node)

graph.set_entry_point("router")
graph.add_conditional_edges("router", route_decision)
graph.add_edge("researcher", "router")
graph.add_edge("writer", "router")
graph.add_edge("reviewer", "router")

app = graph.compile()

# Run the multi-agent graph
result = app.invoke({
    "messages": [HumanMessage(content="Write a technical overview of vector databases")],
    "current_agent": "",
    "research_done": False,
    "draft_done": False,
    "review_done": False,
    "final_output": "",
})

print(result["final_output"])
```

### LangGraph with Human-in-the-Loop

```python
from langgraph.checkpoint.memory import MemorySaver

# Add checkpointing for human-in-the-loop
memory = MemorySaver()

def human_review_node(state: MultiAgentState) -> MultiAgentState:
    """Node that pauses for human review."""
    # In LangGraph, this node will interrupt execution
    # The human can then approve or request changes
    return state

graph_with_human = StateGraph(MultiAgentState)
graph_with_human.add_node("router", router_node)
graph_with_human.add_node("researcher", researcher_node)
graph_with_human.add_node("writer", writer_node)
graph_with_human.add_node("human_review", human_review_node)
graph_with_human.add_node("reviewer", reviewer_node)

graph_with_human.set_entry_point("router")
graph_with_human.add_conditional_edges("router", route_decision)
graph_with_human.add_edge("researcher", "router")
graph_with_human.add_edge("writer", "human_review")
graph_with_human.add_edge("human_review", "router")
graph_with_human.add_edge("reviewer", "router")

app_with_human = graph_with_human.compile(
    checkpointer=memory,
    interrupt_before=["human_review"],  # Pause before human review
)

# Start execution (will pause at human_review)
config = {"configurable": {"thread_id": "review-1"}}
result = app_with_human.invoke(
    {
        "messages": [HumanMessage(content="Write about LLM caching strategies")],
        "current_agent": "",
        "research_done": False,
        "draft_done": False,
        "review_done": False,
        "final_output": "",
    },
    config=config,
)

# After human reviews, resume execution
result = app_with_human.invoke(None, config=config)
```

---

## 7. Practical Multi-Agent RAG System

### Theory: Failure Modes

**G.1 Runaway loops.** Two agents pass control back and forth without progress. Mitigation: step caps, termination detection.

**G.2 Disagreement deadlock.** Two agents won't agree; the system stalls. Mitigation: supervisor with tiebreaker authority; or default to one agent's output if no consensus.

**G.3 Context explosion.** Each agent's prompt includes the full conversation; total tokens grow as O(n_agents × n_turns). Mitigation: summarization, supervisor-mediated narrowing.

**G.4 Cost amplification.** A 5-agent debate with 5 rounds and a supervisor is 25 + 5 = 30 LLM calls per query. Mitigation: aggressive caching, smaller models for cheaper roles, batch-mode for parallel agents.

**G.5 Spurious specialization.** All agents are the same LLM with different prompts; "diversity" of opinion is illusory. Mitigation: actually use different models for genuinely different perspectives (e.g., GPT-4 + Claude + open-source).

### Architecture

```
User Query
    |
    v
[Query Analyzer Agent] -- classifies intent, extracts entities
    |
    v
[Router Agent] -- decides retrieval strategy
    |
    +---> [Vector Search Agent] -- semantic search over embeddings
    |
    +---> [SQL Agent] -- structured data queries
    |
    +---> [Web Search Agent] -- real-time information
    |
    v
[Synthesizer Agent] -- merges results, resolves conflicts
    |
    v
[QA Agent] -- fact-checks against sources, adds citations
    |
    v
Final Answer
```

### Implementation

```python
import json
from openai import OpenAI
from dataclasses import dataclass

client = OpenAI()

@dataclass
class RetrievalResult:
    source: str
    content: str
    relevance_score: float
    metadata: dict

class MultiAgentRAG:
    """Multi-agent RAG system with specialized retrieval agents."""

    def __init__(self):
        self.model = "gpt-4o"

    def query_analyzer(self, query: str) -> dict:
        """Analyze the user query to determine intent and entities."""
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": (
                    "Analyze the user query. Return JSON with:\n"
                    '- "intent": one of ["factual", "analytical", "comparison", "how-to"]\n'
                    '- "entities": list of key entities\n'
                    '- "time_sensitive": boolean (needs recent data?)\n'
                    '- "needs_structured_data": boolean (needs tables/stats?)\n'
                    '- "complexity": one of ["simple", "moderate", "complex"]'
                )},
                {"role": "user", "content": query},
            ],
            response_format={"type": "json_object"},
            temperature=0.1,
        )
        return json.loads(response.choices[0].message.content)

    def route_query(self, analysis: dict) -> list[str]:
        """Decide which retrieval agents to activate."""
        agents = ["vector_search"]  # Always use vector search

        if analysis.get("time_sensitive"):
            agents.append("web_search")
        if analysis.get("needs_structured_data"):
            agents.append("sql_search")
        if analysis.get("complexity") == "complex":
            agents.append("web_search")  # Additional context for complex queries

        return list(set(agents))

    def vector_search_agent(self, query: str, entities: list[str]) -> list[RetrievalResult]:
        """Semantic search over document embeddings."""
        # In production: embed query, search vector DB (Pinecone, Weaviate, etc.)
        enhanced_query = f"{query} {' '.join(entities)}"
        # Simulated results
        return [
            RetrievalResult(
                source="vector_db",
                content=f"Retrieved document for: {enhanced_query}",
                relevance_score=0.92,
                metadata={"doc_id": "doc_001", "chunk_index": 3},
            )
        ]

    def sql_search_agent(self, query: str, entities: list[str]) -> list[RetrievalResult]:
        """Generate and execute SQL queries for structured data."""
        # Use LLM to generate SQL from natural language
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": (
                    "Generate a SQL query to answer the user's question. "
                    "Available tables: products(id, name, category, price), "
                    "reviews(id, product_id, rating, text, date). "
                    "Return only the SQL query."
                )},
                {"role": "user", "content": query},
            ],
            temperature=0.1,
        )
        sql = response.choices[0].message.content
        # In production: execute SQL against database
        return [
            RetrievalResult(
                source="sql_db",
                content=f"SQL result for: {sql}",
                relevance_score=0.95,
                metadata={"query": sql},
            )
        ]

    def web_search_agent(self, query: str) -> list[RetrievalResult]:
        """Search the web for recent information."""
        # In production: call search API (Tavily, Brave, SerpAPI)
        return [
            RetrievalResult(
                source="web",
                content=f"Web search result for: {query}",
                relevance_score=0.85,
                metadata={"url": "https://example.com"},
            )
        ]

    def synthesizer(self, query: str, results: list[RetrievalResult]) -> str:
        """Synthesize results from multiple sources into a coherent answer."""
        context_parts = []
        for r in results:
            context_parts.append(
                f"[Source: {r.source}, Score: {r.relevance_score:.2f}]\n{r.content}"
            )
        context = "\n\n".join(context_parts)

        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": (
                    "Synthesize the provided context into a clear, accurate answer. "
                    "If sources conflict, note the discrepancy. "
                    "Cite sources inline using [source_type] notation."
                )},
                {"role": "user", "content": f"Question: {query}\n\nContext:\n{context}"},
            ],
            temperature=0.3,
        )
        return response.choices[0].message.content

    def qa_check(self, query: str, answer: str, sources: list[RetrievalResult]) -> str:
        """Quality-check the answer against sources."""
        source_text = "\n".join(r.content for r in sources)

        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": (
                    "You are a fact-checker. Verify the answer against the provided sources. "
                    "If everything checks out, return the answer as-is. "
                    "If there are unsupported claims, flag them with [UNVERIFIED]. "
                    "Add source citations where possible."
                )},
                {"role": "user", "content": (
                    f"Question: {query}\n\nAnswer: {answer}\n\nSources:\n{source_text}"
                )},
            ],
            temperature=0.1,
        )
        return response.choices[0].message.content

    def run(self, query: str) -> str:
        """Execute the full multi-agent RAG pipeline."""
        # Step 1: Analyze query
        analysis = self.query_analyzer(query)
        print(f"Analysis: {json.dumps(analysis, indent=2)}")

        # Step 2: Route to retrieval agents
        active_agents = self.route_query(analysis)
        print(f"Active agents: {active_agents}")

        # Step 3: Parallel retrieval
        all_results = []
        entities = analysis.get("entities", [])

        for agent_name in active_agents:
            if agent_name == "vector_search":
                all_results.extend(self.vector_search_agent(query, entities))
            elif agent_name == "sql_search":
                all_results.extend(self.sql_search_agent(query, entities))
            elif agent_name == "web_search":
                all_results.extend(self.web_search_agent(query))

        # Step 4: Synthesize
        answer = self.synthesizer(query, all_results)

        # Step 5: QA check
        verified_answer = self.qa_check(query, answer, all_results)

        return verified_answer

# Usage
rag = MultiAgentRAG()
answer = rag.run("What are the latest benchmark results for GPT-4o vs Claude 3.5?")
print(answer)
```

### Monitoring and Debugging

```python
import logging
import time
from functools import wraps
from typing import Callable

logger = logging.getLogger("multi_agent")

class AgentTracer:
    """Trace and monitor multi-agent execution."""

    def __init__(self):
        self.traces: list[dict] = []
        self._start_time: float | None = None

    def start_run(self, run_id: str):
        self._start_time = time.time()
        self.traces.append({
            "run_id": run_id,
            "start_time": self._start_time,
            "steps": [],
        })

    def trace_step(self, agent_name: str, action: str):
        """Decorator to trace agent steps."""
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                step_start = time.time()
                try:
                    result = func(*args, **kwargs)
                    duration = time.time() - step_start
                    self.traces[-1]["steps"].append({
                        "agent": agent_name,
                        "action": action,
                        "duration_ms": round(duration * 1000, 2),
                        "status": "success",
                        "output_preview": str(result)[:200],
                    })
                    logger.info(
                        f"[{agent_name}] {action} completed in {duration*1000:.0f}ms"
                    )
                    return result
                except Exception as e:
                    duration = time.time() - step_start
                    self.traces[-1]["steps"].append({
                        "agent": agent_name,
                        "action": action,
                        "duration_ms": round(duration * 1000, 2),
                        "status": "error",
                        "error": str(e),
                    })
                    logger.error(f"[{agent_name}] {action} failed: {e}")
                    raise
            return wrapper
        return decorator

    def get_summary(self) -> dict:
        """Get execution summary for the last run."""
        if not self.traces:
            return {}
        trace = self.traces[-1]
        total_duration = sum(s["duration_ms"] for s in trace["steps"])
        return {
            "run_id": trace["run_id"],
            "total_steps": len(trace["steps"]),
            "total_duration_ms": total_duration,
            "steps": [
                {
                    "agent": s["agent"],
                    "action": s["action"],
                    "duration_ms": s["duration_ms"],
                    "status": s["status"],
                }
                for s in trace["steps"]
            ],
            "errors": [
                s for s in trace["steps"] if s["status"] == "error"
            ],
        }

# Usage
tracer = AgentTracer()
tracer.start_run("rag-query-001")

# Wrap agent calls with tracing
@tracer.trace_step("query_analyzer", "analyze_query")
def traced_analyze(query):
    return {"intent": "factual", "entities": ["GPT-4o"]}

result = traced_analyze("What is GPT-4o?")
print(json.dumps(tracer.get_summary(), indent=2))
```

### Agent Design Best Practices

| Practice | Description |
|----------|-------------|
| Single Responsibility | Each agent should do one thing well |
| Explicit Contracts | Define clear input/output schemas between agents |
| Graceful Degradation | Agents should handle failures without crashing the system |
| Idempotent Operations | Re-running an agent with the same input gives the same output |
| Timeout Boundaries | Set max execution time per agent to prevent runaway costs |
| Observability | Log every agent decision, tool call, and inter-agent message |
| State Checkpointing | Save intermediate state so workflows can resume after failure |
| Cost Budgets | Track token usage per agent and enforce limits |

---

## Next Steps

In [16_Practical_Chatbot.md](./16_Practical_Chatbot.md), we build a complete practical chatbot application combining the agent patterns covered here with real-world deployment considerations.
