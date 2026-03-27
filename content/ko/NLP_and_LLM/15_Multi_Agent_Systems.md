# 15. 멀티 에이전트 시스템

## 학습 목표

- 멀티 에이전트 아키텍처와 오케스트레이션 패턴 이해
- Supervisor, Sequential, Parallel 에이전트 토폴로지 구현
- 공유 메모리와 상태를 활용한 에이전트 간 통신 구축
- CrewAI, AutoGen, LangGraph 프레임워크를 사용한 멀티 에이전트 워크플로우
- 실용적인 멀티 에이전트 RAG 시스템 설계 및 배포

---

## 1. 멀티 에이전트 아키텍처 개요

### 왜 멀티 에이전트인가?

> **단일 에이전트 vs 멀티 에이전트**
>
> - **단일 에이전트**: 하나의 LLM이 모든 추론, 도구 사용, 출력 생성을 처리
> - **멀티 에이전트**: 여러 전문화된 에이전트가 협업하며 각각 집중된 역량을 보유
> - **핵심 이점**: 복잡한 작업을 도메인 전문성을 가진 관리 가능한 하위 작업으로 분해

### 아키텍처 비교

| 패턴 | 설명 | 사용 사례 | 복잡도 |
|------|------|-----------|--------|
| 단일 에이전트 | 하나의 LLM, 다중 도구 | 단순 Q&A, 기본 RAG | 낮음 |
| 순차 파이프라인 | 에이전트가 순서대로 실행 | 콘텐츠 파이프라인, ETL | 중간 |
| Supervisor (Hub-and-Spoke) | 오케스트레이터가 워커에게 위임 | 복잡한 리서치 작업 | 중상 |
| Parallel Fan-Out | 여러 에이전트가 동시 실행 | 다중 소스 검색 | 중간 |
| 계층적 | 중첩된 supervisor | 엔터프라이즈 워크플로우 | 높음 |
| 토론/합의 | 에이전트가 논쟁하고 수렴 | 팩트 체킹, 리뷰 | 높음 |

### 핵심 컴포넌트

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
    """에이전트 간 전달되는 메시지."""
    sender: str
    receiver: str
    content: str
    metadata: dict = field(default_factory=dict)
    message_type: str = "task"  # task, result, feedback, error

@dataclass
class AgentState:
    """워크플로우 내 모든 에이전트가 접근 가능한 공유 상태."""
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

## 2. 오케스트레이션 패턴

### 순차 파이프라인 (Sequential Pipeline)

에이전트가 고정된 순서로 실행되며, 각 에이전트가 이전 에이전트의 출력을 변환하거나 보강한다.

```python
from openai import OpenAI

client = OpenAI()

def create_agent(name: str, system_prompt: str):
    """전문화된 에이전트를 생성하는 팩토리."""
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

# 전문화된 에이전트 정의
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
    """순차적으로 에이전트를 실행: Researcher -> Writer -> Editor."""
    results = {}

    print("[1/3] 리서치 중...")
    research = researcher(f"Research the following topic thoroughly: {topic}")
    results["research"] = research

    print("[2/3] 작성 중...")
    article = writer(f"Write an article based on this research:\n\n{research}")
    results["draft"] = article

    print("[3/3] 편집 중...")
    final = editor(f"Edit and improve this article:\n\n{article}")
    results["final"] = final

    return results

# 사용 예시
results = sequential_pipeline("The impact of transformer architecture on NLP")
print(results["final"][:500])
```

### Supervisor 패턴

중앙 오케스트레이터가 현재 상태에 따라 다음에 호출할 에이전트를 결정한다.

```python
import json

class SupervisorAgent:
    """전문화된 워커에게 작업을 위임하는 오케스트레이터."""

    def __init__(self, workers: dict[str, callable]):
        self.workers = workers
        self.client = OpenAI()
        self.worker_descriptions = {
            name: f"Agent '{name}' - available for delegation"
            for name in workers
        }

    def _decide_next_action(self, task: str, history: list[dict]) -> dict:
        """LLM을 사용하여 다음에 호출할 워커를 결정."""
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
        """Supervisor 루프를 실행."""
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

            print(f"  Step {step+1}: [{worker_name}]에 위임: {sub_task[:80]}...")
            result = self.workers[worker_name](sub_task)
            history.append({"worker": worker_name, "result": result})

        return f"Supervisor가 {max_steps} 단계를 완료. 마지막 결과: {history[-1]['result']}"

# 사용 예시
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

여러 에이전트가 동시에 실행되고 결과가 집계된다.

```python
import asyncio
from openai import AsyncOpenAI

async_client = AsyncOpenAI()

async def async_agent(name: str, system_prompt: str, input_text: str) -> dict:
    """병렬 실행을 위한 비동기 에이전트."""
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
    """여러 리서치 에이전트로 Fan-Out 후 집계."""
    agents = [
        ("technical_analyst", "Analyze from a technical perspective. Focus on architecture and implementation."),
        ("business_analyst", "Analyze from a business perspective. Focus on market impact and ROI."),
        ("risk_analyst", "Analyze risks, limitations, and potential failure modes."),
    ]

    tasks = [
        async_agent(name, prompt, query)
        for name, prompt in agents
    ]

    # 모든 에이전트를 병렬로 실행
    results = await asyncio.gather(*tasks)
    return results

async def fan_out_then_aggregate(query: str) -> str:
    """완전한 fan-out/fan-in 패턴과 집계."""
    # Fan-out: 병렬 리서치
    research_results = await parallel_research(query)

    # Fan-in: synthesizer 에이전트로 집계
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

# 사용 예시
result = asyncio.run(fan_out_then_aggregate("Should we adopt LLMs for customer support?"))
```

---

## 3. 에이전트 간 통신

### 메시지 전달 프로토콜

```python
from collections import defaultdict
from typing import Callable
import threading
import queue

class MessageBus:
    """에이전트 통신을 위한 중앙 메시지 버스."""

    def __init__(self):
        self._queues: dict[str, queue.Queue] = defaultdict(queue.Queue)
        self._subscribers: dict[str, list[Callable]] = defaultdict(list)
        self._lock = threading.Lock()

    def send(self, message: AgentMessage):
        """특정 에이전트에게 메시지를 전송."""
        with self._lock:
            self._queues[message.receiver].put(message)
            # 구독자에게 알림
            for callback in self._subscribers.get(message.receiver, []):
                callback(message)

    def receive(self, agent_name: str, timeout: float = 30.0) -> AgentMessage | None:
        """에이전트를 위한 블로킹 수신."""
        try:
            return self._queues[agent_name].get(timeout=timeout)
        except queue.Empty:
            return None

    def subscribe(self, agent_name: str, callback: Callable):
        """에이전트가 메시지를 수신할 때 호출할 콜백을 등록."""
        self._subscribers[agent_name].append(callback)

    def broadcast(self, sender: str, content: str, exclude: set[str] | None = None):
        """모든 등록된 에이전트에게 메시지를 전송."""
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

### 공유 메모리 저장소

```python
import time
from dataclasses import dataclass

@dataclass
class MemoryEntry:
    key: str
    value: Any
    author: str
    timestamp: float
    ttl: float | None = None  # 초 단위 Time-to-live

class SharedMemory:
    """멀티 에이전트 시스템을 위한 스레드 안전 공유 메모리."""

    def __init__(self):
        self._store: dict[str, MemoryEntry] = {}
        self._lock = threading.RLock()
        self._history: list[tuple[str, str, str]] = []  # (action, key, agent)

    def write(self, key: str, value: Any, author: str, ttl: float | None = None):
        """공유 메모리에 값을 기록."""
        with self._lock:
            self._store[key] = MemoryEntry(
                key=key, value=value, author=author,
                timestamp=time.time(), ttl=ttl,
            )
            self._history.append(("write", key, author))

    def read(self, key: str) -> Any | None:
        """TTL을 준수하며 공유 메모리에서 값을 읽기."""
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            if entry.ttl and (time.time() - entry.timestamp) > entry.ttl:
                del self._store[key]
                return None
            return entry.value

    def read_all(self) -> dict[str, Any]:
        """만료되지 않은 모든 항목을 읽기."""
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
        """액션 히스토리 조회 (선택적으로 에이전트별 필터링)."""
        if agent:
            return [(a, k, ag) for a, k, ag in self._history if ag == agent]
        return list(self._history)
```

### 통신 패턴

| 패턴 | 설명 | 장점 | 단점 |
|------|------|------|------|
| 직접 메시징 | 메시지 버스를 통한 에이전트 간 통신 | 낮은 지연, 단순함 | 강한 결합 |
| 공유 메모리 | 중앙 저장소에 읽기/쓰기 | 느슨한 결합, 영구적 | 경쟁 조건 |
| 블랙보드 | 에이전트가 공유 보드에 게시/읽기 | 유연하고 확장 가능 | 순서 문제 |
| Pub/Sub | 토픽 기반 메시지 구독 | 확장 가능, 느슨한 결합 | 메시지 손실 위험 |
| 요청/응답 | 동기식 질의-응답 | 명확한 흐름 | 블로킹 |

---

## 4. CrewAI 프레임워크

### 개요

CrewAI는 에이전트, 태스크, 크루(에이전트 팀)를 정의하기 위한 고수준 추상화를 제공한다.

```python
from crewai import Agent, Task, Crew, Process
from crewai.tools import tool

# 커스텀 도구 정의
@tool
def search_papers(query: str) -> str:
    """주어진 주제에 대한 학술 논문을 검색."""
    # 프로덕션에서는 Semantic Scholar 같은 API를 호출
    return f"Found 5 papers related to '{query}': [Paper1: ..., Paper2: ..., ...]"

@tool
def analyze_code(code: str) -> str:
    """Python 코드의 품질과 정확성을 분석."""
    return f"Code analysis: {len(code.split(chr(10)))} lines, no critical issues found."

# 역할과 목표가 있는 에이전트 정의
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

# 태스크 정의
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
    context=[research_task],  # 리서치 출력에 의존
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

# 크루 생성 및 실행
crew = Crew(
    agents=[research_agent, writer_agent, reviewer_agent],
    tasks=[research_task, writing_task, review_task],
    process=Process.sequential,
    verbose=True,
)

result = crew.kickoff(inputs={"topic": "multi-agent LLM systems"})
print(result)
```

### CrewAI 계층적 프로세스

```python
# 매니저 에이전트를 사용한 계층적 프로세스
manager_crew = Crew(
    agents=[research_agent, writer_agent, reviewer_agent],
    tasks=[research_task, writing_task, review_task],
    process=Process.hierarchical,
    manager_llm="gpt-4o",
    verbose=True,
)

# 매니저가 자동으로 위임 및 조율
result = manager_crew.kickoff(inputs={"topic": "RAG optimization techniques"})
```

---

## 5. AutoGen 프레임워크

### Conversable Agents

```python
from autogen import ConversableAgent, GroupChat, GroupChatManager

# LLM 설정
llm_config = {
    "config_list": [{"model": "gpt-4o", "api_key": "your-key"}],
    "temperature": 0.3,
}

# 대화 가능한 에이전트 정의
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

# 자동 발화자 선택을 사용한 그룹 채팅
group_chat = GroupChat(
    agents=[planner, coder, analyst, critic],
    messages=[],
    max_round=15,
    speaker_selection_method="auto",  # LLM이 다음 발화자를 결정
)

manager = GroupChatManager(
    groupchat=group_chat,
    llm_config=llm_config,
)

# 대화 시작
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

### AutoGen 코드 실행

```python
from autogen import AssistantAgent, UserProxyAgent
from autogen.coding import LocalCommandLineCodeExecutor

# 코드를 작성할 수 있는 에이전트
assistant = AssistantAgent(
    name="assistant",
    llm_config=llm_config,
    system_message="You are a helpful AI assistant that writes Python code to solve problems.",
)

# 코드 실행 기능이 있는 에이전트
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

# assistant가 코드를 작성하면 executor가 실행
executor.initiate_chat(
    assistant,
    message="Create a Python script that fetches the top 10 Hacker News stories and displays them.",
)
```

---

## 6. LangGraph 멀티 에이전트

### 상태 기반 에이전트 그래프

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
    """다음 에이전트로 라우팅하는 supervisor."""
    if not state.get("research_done"):
        return {**state, "current_agent": "researcher"}
    elif not state.get("draft_done"):
        return {**state, "current_agent": "writer"}
    elif not state.get("review_done"):
        return {**state, "current_agent": "reviewer"}
    else:
        return {**state, "current_agent": "done"}

def researcher_node(state: MultiAgentState) -> MultiAgentState:
    """리서치 에이전트 노드."""
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
    """작성 에이전트 노드."""
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
    """리뷰어 에이전트 노드."""
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
    """조건부 엣지: 다음 노드를 결정."""
    agent = state.get("current_agent", "researcher")
    if agent == "done":
        return END
    return agent

# 그래프 빌드
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

# 멀티 에이전트 그래프 실행
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

### LangGraph Human-in-the-Loop

```python
from langgraph.checkpoint.memory import MemorySaver

# Human-in-the-loop를 위한 체크포인팅 추가
memory = MemorySaver()

def human_review_node(state: MultiAgentState) -> MultiAgentState:
    """사람의 검토를 위해 실행을 일시 중지하는 노드."""
    # LangGraph에서 이 노드는 실행을 중단함
    # 사람이 승인하거나 변경을 요청할 수 있음
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
    interrupt_before=["human_review"],  # human_review 전에 일시 중지
)

# 실행 시작 (human_review에서 일시 중지됨)
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

# 사람이 검토한 후 실행 재개
result = app_with_human.invoke(None, config=config)
```

---

## 7. 실용적 멀티 에이전트 RAG 시스템

### 아키텍처

```
사용자 쿼리
    |
    v
[Query Analyzer Agent] -- 의도 분류, 엔티티 추출
    |
    v
[Router Agent] -- 검색 전략 결정
    |
    +---> [Vector Search Agent] -- 임베딩에 대한 시맨틱 검색
    |
    +---> [SQL Agent] -- 구조화된 데이터 쿼리
    |
    +---> [Web Search Agent] -- 실시간 정보
    |
    v
[Synthesizer Agent] -- 결과 병합, 충돌 해결
    |
    v
[QA Agent] -- 출처 대비 팩트 체킹, 인용 추가
    |
    v
최종 답변
```

### 구현

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
    """전문화된 검색 에이전트가 있는 멀티 에이전트 RAG 시스템."""

    def __init__(self):
        self.model = "gpt-4o"

    def query_analyzer(self, query: str) -> dict:
        """사용자 쿼리를 분석하여 의도와 엔티티를 결정."""
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
        """활성화할 검색 에이전트를 결정."""
        agents = ["vector_search"]  # 항상 벡터 검색 사용

        if analysis.get("time_sensitive"):
            agents.append("web_search")
        if analysis.get("needs_structured_data"):
            agents.append("sql_search")
        if analysis.get("complexity") == "complex":
            agents.append("web_search")  # 복잡한 쿼리에 추가 컨텍스트

        return list(set(agents))

    def vector_search_agent(self, query: str, entities: list[str]) -> list[RetrievalResult]:
        """문서 임베딩에 대한 시맨틱 검색."""
        # 프로덕션: 쿼리 임베딩, 벡터 DB 검색 (Pinecone, Weaviate 등)
        enhanced_query = f"{query} {' '.join(entities)}"
        # 시뮬레이션 결과
        return [
            RetrievalResult(
                source="vector_db",
                content=f"Retrieved document for: {enhanced_query}",
                relevance_score=0.92,
                metadata={"doc_id": "doc_001", "chunk_index": 3},
            )
        ]

    def sql_search_agent(self, query: str, entities: list[str]) -> list[RetrievalResult]:
        """구조화된 데이터를 위한 SQL 쿼리 생성 및 실행."""
        # LLM을 사용하여 자연어에서 SQL 생성
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
        # 프로덕션: 실제 데이터베이스에 대해 SQL 실행
        return [
            RetrievalResult(
                source="sql_db",
                content=f"SQL result for: {sql}",
                relevance_score=0.95,
                metadata={"query": sql},
            )
        ]

    def web_search_agent(self, query: str) -> list[RetrievalResult]:
        """최신 정보를 위한 웹 검색."""
        # 프로덕션: 검색 API 호출 (Tavily, Brave, SerpAPI)
        return [
            RetrievalResult(
                source="web",
                content=f"Web search result for: {query}",
                relevance_score=0.85,
                metadata={"url": "https://example.com"},
            )
        ]

    def synthesizer(self, query: str, results: list[RetrievalResult]) -> str:
        """여러 소스의 결과를 일관된 답변으로 합성."""
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
        """출처 대비 답변 품질 검증."""
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
        """전체 멀티 에이전트 RAG 파이프라인을 실행."""
        # 단계 1: 쿼리 분석
        analysis = self.query_analyzer(query)
        print(f"분석 결과: {json.dumps(analysis, indent=2)}")

        # 단계 2: 검색 에이전트로 라우팅
        active_agents = self.route_query(analysis)
        print(f"활성 에이전트: {active_agents}")

        # 단계 3: 병렬 검색
        all_results = []
        entities = analysis.get("entities", [])

        for agent_name in active_agents:
            if agent_name == "vector_search":
                all_results.extend(self.vector_search_agent(query, entities))
            elif agent_name == "sql_search":
                all_results.extend(self.sql_search_agent(query, entities))
            elif agent_name == "web_search":
                all_results.extend(self.web_search_agent(query))

        # 단계 4: 합성
        answer = self.synthesizer(query, all_results)

        # 단계 5: QA 검증
        verified_answer = self.qa_check(query, answer, all_results)

        return verified_answer

# 사용 예시
rag = MultiAgentRAG()
answer = rag.run("What are the latest benchmark results for GPT-4o vs Claude 3.5?")
print(answer)
```

### 모니터링 및 디버깅

```python
import logging
import time
from functools import wraps
from typing import Callable

logger = logging.getLogger("multi_agent")

class AgentTracer:
    """멀티 에이전트 실행을 추적하고 모니터링."""

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
        """에이전트 단계를 추적하는 데코레이터."""
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
                        f"[{agent_name}] {action} 완료: {duration*1000:.0f}ms"
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
                    logger.error(f"[{agent_name}] {action} 실패: {e}")
                    raise
            return wrapper
        return decorator

    def get_summary(self) -> dict:
        """마지막 실행의 실행 요약을 반환."""
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

# 사용 예시
tracer = AgentTracer()
tracer.start_run("rag-query-001")

# 추적으로 에이전트 호출을 래핑
@tracer.trace_step("query_analyzer", "analyze_query")
def traced_analyze(query):
    return {"intent": "factual", "entities": ["GPT-4o"]}

result = traced_analyze("What is GPT-4o?")
print(json.dumps(tracer.get_summary(), indent=2))
```

### 에이전트 설계 모범 사례

| 사례 | 설명 |
|------|------|
| 단일 책임 원칙 | 각 에이전트는 한 가지를 잘 수행해야 함 |
| 명시적 계약 | 에이전트 간 명확한 입출력 스키마 정의 |
| 우아한 저하 | 에이전트는 시스템을 중단시키지 않고 실패를 처리해야 함 |
| 멱등 연산 | 같은 입력으로 에이전트를 재실행하면 같은 출력을 반환 |
| 타임아웃 경계 | 에이전트별 최대 실행 시간을 설정하여 비용 폭주 방지 |
| 관측 가능성 | 모든 에이전트 결정, 도구 호출, 에이전트 간 메시지를 로깅 |
| 상태 체크포인팅 | 중간 상태를 저장하여 장애 후 워크플로우 재개 가능 |
| 비용 예산 | 에이전트별 토큰 사용량을 추적하고 제한을 적용 |

---

## 다음 단계

[16_Practical_Chatbot.md](./16_Practical_Chatbot.md)에서는 여기서 다룬 에이전트 패턴과 실제 배포 고려사항을 결합하여 완전한 실용적 챗봇 애플리케이션을 구축한다.
