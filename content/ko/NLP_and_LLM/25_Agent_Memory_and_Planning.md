# 25. 에이전트 메모리와 계획 수립 (Agent Memory and Planning)

이전: [프로덕션 LLM 패턴](./24_Production_LLM_Patterns.md) | 다음: [에이전트 평가와 벤치마크](./26_Agent_Evaluation_and_Benchmarks.md)

## 학습 목표

- 에이전트를 위한 메모리 아키텍처(Memory Architecture) 분류 (일화적, 의미적, 절차적)
- 다양한 저장소 백엔드(Storage Backend)를 활용한 단기 및 장기 메모리 시스템 구현
- 실용적 메모리 패턴 구축: 대화 버퍼(Conversation Buffer), 요약(Summary), 엔티티(Entity), 벡터 저장소(Vector Store)
- 작업 분해(Task Decomposition)와 계층적 계획(Hierarchical Planning)을 활용한 계획 프레임워크 설계
- 반복적 개선(Iterative Refinement)과 자기 성찰(Self-Reflection)을 적용하여 에이전트 계획 품질 향상

---

## 목차

구현 참조 전에 [**이론과 원리**](#이론과-원리)를 먼저 읽어 보세요 — 메모리 분류(working / episodic / semantic / procedural), 회상-비용 트레이드오프, 현대 에이전트 아키텍처의 토대가 되는 planner-executor 분해.

1. [에이전트를 위한 메모리 아키텍처](#1-에이전트를-위한-메모리-아키텍처)
2. [단기 메모리 vs 장기 메모리](#2-단기-메모리-vs-장기-메모리)
3. [대화 버퍼 메모리](#3-대화-버퍼-메모리)
4. [요약 메모리](#4-요약-메모리)
5. [엔티티 메모리](#5-엔티티-메모리)
6. [벡터 저장소 메모리](#6-벡터-저장소-메모리)
7. [계획 프레임워크](#7-계획-프레임워크)
8. [계획 후 실행 패턴](#8-계획-후-실행-패턴)
9. [계획을 위한 자기 성찰](#9-계획을-위한-자기-성찰)
10. [메모리 증강 생성](#10-메모리-증강-생성)
11. [연습문제](#연습문제)

---

## 이론과 원리

LLM은 상태가 없습니다 — 컨텍스트 윈도우에 들어가는 것 외에 호출 사이 메모리가 없습니다. *에이전트*는 단일 턴보다 긴 어떤 작업에든 유용하기 위해 메모리가 필요합니다 — 대화 초기에 무엇이 말해졌는지, 연구 작업 중 어떤 사실을 학습했는지, 어떤 계획을 만들었고 어떤 단계가 완료되었는지를 회상하기 위해. 메모리와 계획은 상태 없는 함수 호출자를 사고하는 시스템처럼 보이는 것으로 바꾸는 쌍둥이 추상화입니다. 이 레슨은 메모리 분류, 각 유형을 구현하는 저장소 아키텍처, 긴 작업을 제한된 LLM 호출 시퀀스로 바꾸는 계획 프레임워크(plan-then-execute, ReAct, reflexion)를 다룹니다.

이 섹션은 다음을 다룹니다:

- **(A) 메모리 분류** — episodic, semantic, procedural, working 메모리; 인지과학에서 영감받은 분류.
- **(B) 단기 vs 장기** — 컨텍스트에 들어가는 것 vs 외부화되어야 하는 것.
- **(C) 메모리 구현** — 버퍼, 요약, 엔티티, 벡터 — 각각 사용 사례와 함께.
- **(D) 계획 프레임워크** — 작업 분해, 계층적 계획, 왜 "먼저 계획, 나중에 실행"인가.
- **(E) Plan-and-execute** — 아키텍처, 오류 복구, 동적 재계획.
- **(F) 자기 성찰(Self-reflection)** — Reflexion 형식 향상 루프, 에이전트가 언제 자신을 의심해야 하는가.
- **(G) 메모리 + 계획의 결합** — 새 계획에 정보를 주기 위해 과거 에피소드 사용("기술" 학습의 토대).

### A. 메모리 분류

인지과학은 여러 메모리 시스템을 구별하고, 에이전트 설계가 대략적으로 매핑됩니다:

| 유형 | 인지적 유추 | 에이전트 구현 |
|------|------------|--------------|
| **Working** | 활동 중인 단기 | 현재 LLM 컨텍스트 윈도우 |
| **Episodic** | 특정 과거 사건 | 대화 로그, 작업 실행 추적 |
| **Semantic** | 일반 지식 / 사실 | 학습된 사실의 벡터 데이터베이스 |
| **Procedural** | 무엇을 어떻게 하는지 | 캐시된 성공 계획, 기술, 도구 시퀀스 |

각각 다른 저장소와 검색을 요구합니다. Working memory는 자동(그저 프롬프트에 있는 것). Episodic memory는 대화 히스토리. Semantic memory는 RAG 코퍼스지만 외부가 아닌 에이전트 경험에서 채워집니다. Procedural memory는 가장 야심찬 — 에이전트가 발견한 재사용 가능한 기술을 캐싱.

일반 에이전트는 네 가지 모두에서 이득을 봅니다. 프로덕션 시스템은 종종 처음 둘만(working + episodic) 구현하고 semantic을 RAG에 외주화합니다.

### B. 단기 vs 장기

**단기** — 현재 LLM의 컨텍스트 윈도우에 있는 것. 모델의 컨텍스트 길이로 제한(8K-1M 토큰, 모델에 따라). 접근 저렴(그저 프롬프트에 포함).

**장기** — 컨텍스트 윈도우 밖의 것. 데이터베이스에 저장, 요청 시 검색. 비용 — 쿼리당 검색 호출 1회.

핵심 트레이드오프 — 무엇을 단기에 유지하고 무엇을 장기로 밀지 어떻게 결정하는가? 전략:

- **항상 최근 N 턴 유지** — 단순, 일상 채팅에 작동.
- **요약으로 압축** — 오래된 턴을 단일 요약으로 대체.
- **벡터 저장소** — 모든 것을 인덱싱; 쿼리당 관련 항목 검색.
- **하이브리드** — 최근 턴 그대로 + 중간 연령 요약 + 오래된 것의 벡터 검색.

### C. 메모리 구현

네 흔한 패턴(레슨에서 코드와 함께 다룸):

**C.1 Conversation Buffer Memory.** 모든 턴을 글자 그대로 저장, 모두 다음 프롬프트에 포함. 가장 단순, 대화가 길면 실패.

**C.2 Conversation Summary Memory.** 주기적으로 LLM을 사용해 오래된 턴 요약. 압축하지만 요약이 세부를 잃습니다. 선형 주제 흐름의 채팅에 작동.

**C.3 Entity Memory.** 대화에서 언급된 명명된 엔티티(사람, 장소 등) 추적. 각 엔티티가 에이전트가 학습한 속성을 가진 별도 "카드"를 받습니다. 관계 데이터에 유용.

**C.4 Vector Store Memory.** 모든 턴(또는 모든 사실)을 임베딩하고 벡터 DB에 저장. 쿼리당 가장 관련된 항목 검색. 무한 스케일, 검색 지연 추가.

이들은 구성됩니다. 프로덕션 에이전트는 보통 버퍼(최근 N 턴) + 벡터 저장소(오래된 콘텐츠) + 엔티티(관계 사실)를 동시에 사용.

### D. 계획 프레임워크

단일 LLM 호출보다 긴 작업에 대해 에이전트는 계획해야 합니다 — 어떤 부속 단계가 필요한가, 어떤 순서로, 어떤 의존성으로?

**D.1 암묵적 계획 (ReAct).** 에이전트가 한 번에 한 단계씩 계획, 현재 상태에 기반해 다음 행동 결정. 반응적이지만 근시안적 — 작업을 반복하거나 장거리 구조를 놓칠 수 있음.

**D.2 Plan-then-execute (BabyAGI, Plan-and-Execute).** 먼저 LLM이 전체 계획(부속 작업 목록) 생성. 그 후 별도 실행자가 원래 계획을 가이드로 각 부속 작업 처리. 더 깔끔, 계획이 대략 옳을 때 더 효율적.

**D.3 계층적 계획.** 부속 작업으로 분해, 그 후 각 부속 작업을 다시 분해, 재귀적으로. 진정으로 복잡한 작업(논문 작성, 소프트웨어 구축)에 적합. 가장 유연, 구현이 가장 어려움.

**D.4 Tree-of-thoughts** (Yao 등, 2023). 추론을 트리 탐색으로 — 각 단계에서 여러 후보 다음 생각 생성, 점수 매김, 최선을 확장. 부분 진척 신호가 명확한 문제(수학, 퍼즐)에.

### E. Plan-and-Execute

장기 실행 작업의 지배적 패턴:

```
1. Planner LLM — 목표가 주어졌을 때 부속 작업 목록(계획) 생성.
2. 계획의 각 부속 작업에 대해:
     a. Executor LLM (도구와 함께) — 부속 작업 수행, 결과 반환.
     b. 결과를 working memory에 추가.
3. 모든 부속 작업이 끝난 후, Synthesizer LLM — 결과를 최종 답으로 결합.
```

**재계획.** 부속 작업이 실패하거나 새 정보를 드러낼 때, 선택적으로 새 상태와 함께 planner를 다시 호출. 핵심 트레이드오프 — 얼마나 자주 재계획할지. 항상 재계획은 비싸고; 절대 재계획 안 하면 오류가 전파. 휴리스틱 — 실패 시 재계획, 어떤 부속 작업이 가정과 모순되는 정보를 드러내면 재계획.

**왜 분해가 도움이 되는가.** 단일 LLM 호출은 고정 계산 예산을 가집니다 — 어텐션의 고정 깊이. 다단계 계획은 계산을 많은 호출에 걸쳐 외부화하여 더 많은 총 작업을 허용합니다. Chain-of-Thought(레슨 9)와 같은 원리, 작업 수준에서.

### F. 자기 성찰

작업을 끝낸 에이전트는 자신의 출력을 평가하도록 요청받을 수 있습니다. **Reflexion**(Shinn 등, 2023):

```
1. 에이전트가 작업 시도.
2. 평가자(별도 LLM 또는 점수 루브릭)가 결과 판단.
3. 판단이 부정적이면 에이전트가 성찰: "무엇이 잘못되었나? 다르게 한다면 무엇을 할까?"
4. 성찰이 메모리에 추가.
5. 에이전트가 성찰을 맥락에 두고 작업 재시도.
```

이는 효과적으로 실패를 학습 예시로 변환합니다. 성찰은 이전에 없던 맥락을 추가하여 에이전트가 재시도에서 같은 실수를 피하는 데 도움. 경험적으로 Reflexion 형식 루프가 코딩과 추론 벤치마크(HumanEval, AlfWorld)에서 작업 성공률을 10-30% 향상시킵니다.

비용 — 작업당 더 많은 LLM 호출. 작업이 여러 시도를 허용하고 품질이 지연보다 중요할 때 성찰 사용.

### G. 메모리 + 계획의 결합

가장 야심찬 패턴 — 새 계획에 정보를 주기 위해 과거 에피소드를 사용하는 에이전트.

- 작업 완료 후 (목표, 계획, 결과) 트리플을 episodic memory에 저장.
- 새 작업에 대해 비슷한 과거 에피소드 검색; planner에 few-shot 예시로 사용.
- 시간이 지남에 따라 planner가 비슷한 작업을 분해하는 데 더 능숙해집니다 — 이전에 작동한 것을 보았기 때문.

이것이 메모리와 *학습* 사이의 경계입니다. 에이전트의 행동이 가중치 갱신 없이 시간에 따라 변합니다 — 그저 더 나은 episodic memory를 누적할 뿐. Voyager(Wang 등, 2023)가 Minecraft에 대해 이를 시연 — 캐시된 절차의 "기술 라이브러리"를 만들고 재사용하는 에이전트.

### 이론에서 아래 함수들로

- §1 (메모리 아키텍처) — §A의 네 유형 분류 구현.
- §2 (단기 vs 장기) — §B 저장소 트레이드오프를 틀.
- §3 (버퍼 메모리) — §C.1 구현.
- §4 (요약 메모리) — §C.2 구현.
- §5 (엔티티 메모리) — §C.3 구현.
- §6 (벡터 저장소 메모리) — §C.4 구현(레슨 11 벡터 인덱싱 사용).
- §7 (계획 프레임워크) — §D 개관.
- §8 (plan-and-execute) — §E를 코드로 구현.
- §9 (자기 성찰) — §F Reflexion 루프 구현.
- §10 (메모리 증강 생성) — §G episodic memory + 계획 결합 구현.

---

## 1. 에이전트를 위한 메모리 아키텍처

### 에이전트에 메모리가 필요한 이유

> **메모리 문제 (The Memory Problem)**
>
> LLM은 기본적으로 상태가 없습니다(stateless) — 각 API 호출은 처음부터 시작합니다.
> 에이전트에 메모리가 필요한 이유:
> - 다중 턴 대화(Multi-turn Conversation)에서 맥락 유지
> - 작업 실행 중 학습한 사실 기억
> - 이미 완료한 작업의 반복 방지
> - 시간이 지남에 따라 지속적인 지식 기반(Knowledge Base) 구축

### 에이전트 메모리의 세 가지 유형

| 메모리 유형 | 설명 | 인간 유추 | 에이전트 예시 |
|-------------|------|-----------|---------------|
| **일화적 (Episodic)** | 특정 사건과 경험의 기록 | "2019년에 파리에 갔던 것을 기억해" | 과거 대화 로그, 작업 실행 추적 |
| **의미적 (Semantic)** | 일반적인 사실과 지식 | "파리는 프랑스의 수도야" | 추출된 엔티티, 학습된 사실, 사용자 선호도 |
| **절차적 (Procedural)** | 일을 하는 방법 | "자전거 타는 법을 알아" | 도구 사용 패턴, 성공적인 전략, 워크플로우 |

### 아키텍처 개요

> **에이전트 메모리 아키텍처 (Agent Memory Architecture)**
>
> ```
> Agent Core (LLM)
>     |
>     +---> Working Memory (Context Window)
>     |         |
>     |         +--- System Prompt
>     |         +--- Recent Messages (Short-Term)
>     |         +--- Retrieved Context (from Long-Term)
>     |
>     +---> Long-Term Memory
>               |
>               +--- Episodic Store (conversation logs, traces)
>               +--- Semantic Store (entities, facts, embeddings)
>               +--- Procedural Store (tool patterns, strategies)
> ```

### 메모리 분류 구현 (Memory Taxonomy Implementation)

```python
from dataclasses import dataclass, field
from typing import Any
from enum import Enum
from abc import ABC, abstractmethod
import time


class MemoryType(Enum):
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"


@dataclass
class MemoryEntry:
    """A single memory item."""
    content: str
    memory_type: MemoryType
    timestamp: float = field(default_factory=time.time)
    importance: float = 0.5  # 0.0 to 1.0
    metadata: dict = field(default_factory=dict)
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)

    def touch(self):
        """Update access statistics."""
        self.access_count += 1
        self.last_accessed = time.time()

    @property
    def recency_score(self) -> float:
        """Decay-based recency score."""
        age_hours = (time.time() - self.last_accessed) / 3600
        return 1.0 / (1.0 + 0.1 * age_hours)


class MemoryStore(ABC):
    """Abstract base class for memory stores."""

    @abstractmethod
    def add(self, entry: MemoryEntry) -> str:
        """Store a memory entry. Returns an ID."""
        ...

    @abstractmethod
    def search(self, query: str, top_k: int = 5) -> list[MemoryEntry]:
        """Retrieve relevant memories."""
        ...

    @abstractmethod
    def clear(self) -> None:
        """Remove all entries."""
        ...
```

---

## 2. 단기 메모리 vs 장기 메모리

### 단기 메모리 (Short-Term Memory / Working Memory)

단기 메모리는 LLM의 컨텍스트 윈도우(Context Window) 안에 존재합니다. 빠르지만 용량이 제한됩니다.

```python
from collections import deque


class ShortTermMemory:
    """Fixed-window short-term memory using a ring buffer."""

    def __init__(self, max_messages: int = 20, max_tokens: int = 4096):
        self.messages: deque[dict] = deque(maxlen=max_messages)
        self.max_tokens = max_tokens

    def add(self, role: str, content: str):
        """Add a message to short-term memory."""
        self.messages.append({
            "role": role,
            "content": content,
            "timestamp": time.time(),
        })
        self._enforce_token_limit()

    def _enforce_token_limit(self):
        """Remove oldest messages if token budget exceeded."""
        while self._estimate_tokens() > self.max_tokens and len(self.messages) > 1:
            self.messages.popleft()

    def _estimate_tokens(self) -> int:
        """Rough token estimate: ~4 chars per token."""
        return sum(len(m["content"]) // 4 for m in self.messages)

    def get_messages(self) -> list[dict]:
        """Return messages formatted for the LLM API."""
        return [{"role": m["role"], "content": m["content"]} for m in self.messages]

    @property
    def size(self) -> int:
        return len(self.messages)


# Usage
stm = ShortTermMemory(max_messages=10, max_tokens=2000)
stm.add("user", "What is the capital of France?")
stm.add("assistant", "The capital of France is Paris.")
stm.add("user", "What is its population?")
print(f"Messages in STM: {stm.size}")
print(f"Estimated tokens: {stm._estimate_tokens()}")
```

### 장기 메모리 (Long-Term Memory)

장기 메모리는 컨텍스트 윈도우(Context Window)를 넘어 지속되며 명시적인 검색(Retrieval)이 필요합니다.

```python
import json
import hashlib
from pathlib import Path


class LongTermMemory:
    """Persistent long-term memory with file-backed storage."""

    def __init__(self, storage_path: str = "agent_memory.json"):
        self.storage_path = Path(storage_path)
        self.entries: dict[str, MemoryEntry] = {}
        self._load()

    def _generate_id(self, content: str) -> str:
        return hashlib.sha256(content.encode()).hexdigest()[:12]

    def add(self, entry: MemoryEntry) -> str:
        """Store a memory entry persistently."""
        entry_id = self._generate_id(entry.content + str(entry.timestamp))
        self.entries[entry_id] = entry
        self._save()
        return entry_id

    def search(self, query: str, top_k: int = 5) -> list[MemoryEntry]:
        """Simple keyword search (production systems use embeddings)."""
        query_terms = set(query.lower().split())
        scored = []
        for entry in self.entries.values():
            content_terms = set(entry.content.lower().split())
            overlap = len(query_terms & content_terms)
            if overlap > 0:
                # Combine relevance, recency, and importance
                score = (
                    overlap * 0.4
                    + entry.recency_score * 0.3
                    + entry.importance * 0.3
                )
                scored.append((score, entry))

        scored.sort(key=lambda x: x[0], reverse=True)
        results = [entry for _, entry in scored[:top_k]]
        for entry in results:
            entry.touch()
        return results

    def _save(self):
        """Persist to disk."""
        data = {}
        for eid, entry in self.entries.items():
            data[eid] = {
                "content": entry.content,
                "memory_type": entry.memory_type.value,
                "timestamp": entry.timestamp,
                "importance": entry.importance,
                "metadata": entry.metadata,
                "access_count": entry.access_count,
            }
        self.storage_path.write_text(json.dumps(data, indent=2))

    def _load(self):
        """Load from disk."""
        if self.storage_path.exists():
            data = json.loads(self.storage_path.read_text())
            for eid, d in data.items():
                self.entries[eid] = MemoryEntry(
                    content=d["content"],
                    memory_type=MemoryType(d["memory_type"]),
                    timestamp=d["timestamp"],
                    importance=d["importance"],
                    metadata=d["metadata"],
                    access_count=d["access_count"],
                )

    def clear(self):
        self.entries.clear()
        if self.storage_path.exists():
            self.storage_path.unlink()
```

### 메모리 비교

| 측면 | 단기 메모리 (Short-Term) | 장기 메모리 (Long-Term) |
|------|-----------|-----------|
| 저장소 | 컨텍스트 윈도우 (RAM) | 데이터베이스 / 파일 / 벡터 저장소 |
| 용량 | 4K–200K 토큰 | 무제한 |
| 속도 | 즉각적 (인컨텍스트) | 검색 단계 필요 |
| 지속성 | 세션 종료 시 소실 | 세션 간 유지 |
| 비용 | 요청당 토큰 | 저장소 + 검색 비용 |
| 정확도 | 완벽 (원문 그대로) | 검색 품질에 따라 다름 |

---

## 3. 대화 버퍼 메모리

### 고정 윈도우 버퍼 (Fixed-Window Buffer)

가장 단순한 메모리 패턴: 마지막 N개 메시지를 그대로 유지합니다.

```python
class ConversationBufferMemory:
    """Keep the last N conversation turns as context."""

    def __init__(self, max_turns: int = 10):
        self.max_turns = max_turns
        self.history: list[dict] = []

    def add_user_message(self, content: str):
        self.history.append({"role": "user", "content": content})
        self._trim()

    def add_assistant_message(self, content: str):
        self.history.append({"role": "assistant", "content": content})
        self._trim()

    def _trim(self):
        """Keep only the last max_turns * 2 messages (user + assistant pairs)."""
        max_messages = self.max_turns * 2
        if len(self.history) > max_messages:
            self.history = self.history[-max_messages:]

    def get_context(self) -> list[dict]:
        return self.history.copy()

    def clear(self):
        self.history.clear()
```

### 토큰 예산 기반 슬라이딩 윈도우 (Sliding Window with Token Budget)

```python
class TokenBudgetBuffer:
    """Conversation buffer that respects a token budget."""

    def __init__(self, token_budget: int = 8000):
        self.token_budget = token_budget
        self.history: list[dict] = []

    def add(self, role: str, content: str):
        self.history.append({"role": role, "content": content})
        self._enforce_budget()

    def _estimate_tokens(self, messages: list[dict]) -> int:
        return sum(len(m["content"]) // 4 + 4 for m in messages)

    def _enforce_budget(self):
        """Remove oldest messages until within budget."""
        while (
            self._estimate_tokens(self.history) > self.token_budget
            and len(self.history) > 2
        ):
            # Always keep the first system-level message if present
            if self.history[0]["role"] == "system":
                self.history.pop(1)
            else:
                self.history.pop(0)

    def get_context(self) -> list[dict]:
        return self.history.copy()

    @property
    def token_usage(self) -> int:
        return self._estimate_tokens(self.history)
```

---

## 4. 요약 메모리

### 대화 요약 (Conversation Summary)

모든 메시지를 유지하는 대신, 오래된 맥락을 주기적으로 요약합니다.

```python
import anthropic


class SummaryMemory:
    """Summarize older messages to save token budget."""

    def __init__(self, max_recent: int = 6, model: str = "claude-haiku-4-20250514"):
        self.max_recent = max_recent
        self.model = model
        self.summary: str = ""
        self.recent_messages: list[dict] = []
        self.client = anthropic.Anthropic()

    def add(self, role: str, content: str):
        self.recent_messages.append({"role": role, "content": content})

        # When recent messages exceed the window, compress
        if len(self.recent_messages) > self.max_recent * 2:
            self._compress()

    def _compress(self):
        """Summarize older messages and keep only recent ones."""
        to_summarize = self.recent_messages[:-self.max_recent]
        self.recent_messages = self.recent_messages[-self.max_recent:]

        conversation_text = "\n".join(
            f"{m['role']}: {m['content']}" for m in to_summarize
        )

        prompt = (
            f"Summarize the following conversation, preserving all key facts, "
            f"decisions, and action items. Be concise but complete.\n\n"
            f"Previous summary:\n{self.summary}\n\n"
            f"New messages:\n{conversation_text}"
        )

        response = self.client.messages.create(
            model=self.model,
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}],
        )
        self.summary = response.content[0].text

    def get_context(self) -> list[dict]:
        """Return summary + recent messages for the LLM."""
        messages = []
        if self.summary:
            messages.append({
                "role": "user",
                "content": f"[Conversation summary so far]: {self.summary}",
            })
            messages.append({
                "role": "assistant",
                "content": "Understood. I have the conversation context.",
            })
        messages.extend(self.recent_messages)
        return messages
```

### 점진적 요약 (Progressive Summarization)

```python
class ProgressiveSummary:
    """Multi-level summarization: recent -> summary -> meta-summary."""

    def __init__(self):
        self.meta_summary: str = ""       # High-level summary of all time
        self.session_summary: str = ""    # Summary of current session
        self.recent: list[dict] = []      # Last few messages
        self.turn_count: int = 0

    def add(self, role: str, content: str):
        self.recent.append({"role": role, "content": content})
        self.turn_count += 1

        # Compress recent -> session summary every 10 turns
        if len(self.recent) > 10:
            self._compress_recent()

        # Compress session -> meta summary every 50 turns
        if self.turn_count % 50 == 0 and self.session_summary:
            self._compress_session()

    def _compress_recent(self):
        """Move older recent messages into session summary."""
        older = self.recent[:-4]
        self.recent = self.recent[-4:]
        text = "\n".join(f"{m['role']}: {m['content']}" for m in older)
        if self.session_summary:
            self.session_summary += f"\n\n{text}"
        else:
            self.session_summary = text

    def _compress_session(self):
        """Summarize session into meta-summary (requires LLM call)."""
        if self.meta_summary:
            self.meta_summary += f"\n---\n{self.session_summary}"
        else:
            self.meta_summary = self.session_summary
        self.session_summary = ""

    def get_context_string(self) -> str:
        parts = []
        if self.meta_summary:
            parts.append(f"[Long-term context]: {self.meta_summary}")
        if self.session_summary:
            parts.append(f"[Session context]: {self.session_summary}")
        parts.append("\n".join(
            f"{m['role']}: {m['content']}" for m in self.recent
        ))
        return "\n\n".join(parts)
```

---

## 5. 엔티티 메모리

### 엔티티 추출 및 추적 (Entity Extraction and Tracking)

엔티티 메모리(Entity Memory)는 대화에서 언급된 사람, 장소, 개념에 대한 구조화된 정보를 추출하고 유지합니다.

```python
from dataclasses import dataclass, field


@dataclass
class Entity:
    """A tracked entity with attributes."""
    name: str
    entity_type: str  # "person", "organization", "project", "concept"
    attributes: dict[str, str] = field(default_factory=dict)
    first_mentioned: float = field(default_factory=time.time)
    last_updated: float = field(default_factory=time.time)
    mention_count: int = 1

    def update(self, new_attributes: dict[str, str]):
        """Merge new attributes, overwriting conflicts."""
        self.attributes.update(new_attributes)
        self.last_updated = time.time()
        self.mention_count += 1


class EntityMemory:
    """Track entities mentioned across conversations."""

    def __init__(self):
        self.entities: dict[str, Entity] = {}
        self.client = anthropic.Anthropic()

    def extract_entities(self, text: str) -> list[dict]:
        """Use an LLM to extract entities from text."""
        response = self.client.messages.create(
            model="claude-haiku-4-20250514",
            max_tokens=500,
            messages=[{
                "role": "user",
                "content": (
                    "Extract all named entities from this text. "
                    "Return a JSON array of objects with keys: "
                    "'name', 'type' (person/org/project/concept), "
                    "'attributes' (dict of key-value facts).\n\n"
                    f"Text: {text}\n\n"
                    "JSON:"
                ),
            }],
        )
        import json
        try:
            return json.loads(response.content[0].text)
        except (json.JSONDecodeError, IndexError):
            return []

    def update_from_text(self, text: str):
        """Extract and merge entities from new text."""
        extracted = self.extract_entities(text)
        for entity_data in extracted:
            name = entity_data.get("name", "").lower()
            if not name:
                continue

            if name in self.entities:
                self.entities[name].update(
                    entity_data.get("attributes", {})
                )
            else:
                self.entities[name] = Entity(
                    name=entity_data["name"],
                    entity_type=entity_data.get("type", "concept"),
                    attributes=entity_data.get("attributes", {}),
                )

    def get_relevant_entities(self, query: str, top_k: int = 5) -> list[Entity]:
        """Find entities relevant to a query."""
        query_terms = set(query.lower().split())
        scored = []
        for entity in self.entities.values():
            name_terms = set(entity.name.lower().split())
            attr_text = " ".join(entity.attributes.values()).lower()
            attr_terms = set(attr_text.split())

            overlap = len(query_terms & (name_terms | attr_terms))
            if overlap > 0 or any(term in entity.name.lower() for term in query_terms):
                score = overlap + entity.mention_count * 0.1
                scored.append((score, entity))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [e for _, e in scored[:top_k]]

    def format_entity_context(self, entities: list[Entity]) -> str:
        """Format entities for injection into the prompt."""
        if not entities:
            return ""
        lines = ["[Known entities]:"]
        for e in entities:
            attrs = ", ".join(f"{k}: {v}" for k, v in e.attributes.items())
            lines.append(f"- {e.name} ({e.entity_type}): {attrs}")
        return "\n".join(lines)
```

---

## 6. 벡터 저장소 메모리

### 임베딩 기반 메모리 (Embedding-Based Memory)

벡터 저장소 메모리(Vector Store Memory)는 임베딩(Embedding)을 사용하여 의미적 유사성(Semantic Similarity)으로 메모리를 저장하고 검색하며, 가장 강력한 검색 메커니즘입니다.

```python
import numpy as np
from dataclasses import dataclass, field


@dataclass
class VectorMemoryItem:
    """A memory item with its embedding."""
    content: str
    embedding: np.ndarray
    metadata: dict = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    importance: float = 0.5


class VectorStoreMemory:
    """Memory backed by a vector store for semantic retrieval."""

    def __init__(self, embedding_dim: int = 1536):
        self.items: list[VectorMemoryItem] = []
        self.embedding_dim = embedding_dim

    def _get_embedding(self, text: str) -> np.ndarray:
        """Get embedding from OpenAI API."""
        from openai import OpenAI
        client = OpenAI()
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=text,
        )
        return np.array(response.data[0].embedding)

    def add(self, content: str, importance: float = 0.5,
            metadata: dict | None = None):
        """Store a memory with its embedding."""
        embedding = self._get_embedding(content)
        self.items.append(VectorMemoryItem(
            content=content,
            embedding=embedding,
            metadata=metadata or {},
            importance=importance,
        ))

    def search(self, query: str, top_k: int = 5,
               recency_weight: float = 0.2,
               importance_weight: float = 0.1) -> list[dict]:
        """Retrieve memories by semantic similarity + recency + importance."""
        if not self.items:
            return []

        query_embedding = self._get_embedding(query)

        # Compute cosine similarity
        scores = []
        now = time.time()
        for item in self.items:
            similarity = np.dot(query_embedding, item.embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(item.embedding)
            )

            # Recency decay: exponential decay over hours
            age_hours = (now - item.timestamp) / 3600
            recency = np.exp(-0.1 * age_hours)

            # Combined score
            score = (
                similarity * (1.0 - recency_weight - importance_weight)
                + recency * recency_weight
                + item.importance * importance_weight
            )
            scores.append((score, item))

        scores.sort(key=lambda x: x[0], reverse=True)
        return [
            {
                "content": item.content,
                "score": float(score),
                "metadata": item.metadata,
            }
            for score, item in scores[:top_k]
        ]

    def format_context(self, results: list[dict]) -> str:
        """Format retrieved memories as context for the LLM."""
        if not results:
            return ""
        lines = ["[Retrieved memories]:"]
        for r in results:
            lines.append(f"- (relevance: {r['score']:.2f}) {r['content']}")
        return "\n".join(lines)
```

### 하이브리드 메모리 시스템 (Hybrid Memory System)

```python
class HybridMemory:
    """Combine buffer, summary, entity, and vector memories."""

    def __init__(self):
        self.buffer = ConversationBufferMemory(max_turns=5)
        self.entity = EntityMemory()
        self.vector = VectorStoreMemory()
        self.turn_count = 0

    def add_interaction(self, user_msg: str, assistant_msg: str):
        """Process a full interaction across all memory systems."""
        # Buffer: always store recent messages
        self.buffer.add_user_message(user_msg)
        self.buffer.add_assistant_message(assistant_msg)

        # Entity: extract entities from both messages
        self.entity.update_from_text(user_msg)
        self.entity.update_from_text(assistant_msg)

        # Vector: store the interaction for semantic retrieval
        combined = f"User: {user_msg}\nAssistant: {assistant_msg}"
        self.vector.add(combined, importance=0.5)
        self.turn_count += 1

    def build_context(self, current_query: str) -> list[dict]:
        """Build a rich context combining all memory sources."""
        context_parts = []

        # 1. Entity context
        relevant_entities = self.entity.get_relevant_entities(current_query)
        entity_ctx = self.entity.format_entity_context(relevant_entities)
        if entity_ctx:
            context_parts.append(entity_ctx)

        # 2. Semantic retrieval from vector store
        similar = self.vector.search(current_query, top_k=3)
        vector_ctx = self.vector.format_context(similar)
        if vector_ctx:
            context_parts.append(vector_ctx)

        # 3. Recent conversation buffer
        messages = self.buffer.get_context()

        # Prepend memory context as a system-like message
        if context_parts:
            memory_text = "\n\n".join(context_parts)
            return [
                {"role": "user", "content": f"[Memory context]:\n{memory_text}"},
                {"role": "assistant", "content": "I have the context. How can I help?"},
            ] + messages

        return messages
```

---

## 7. 계획 프레임워크

### 작업 분해 (Task Decomposition)

복잡한 작업은 더 작고 관리 가능한 하위 작업(Sub-task)으로 분해해야 합니다.

```python
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class TaskStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"


@dataclass
class Task:
    """A single task in a plan."""
    task_id: str
    description: str
    status: TaskStatus = TaskStatus.PENDING
    dependencies: list[str] = field(default_factory=list)
    result: Any = None
    subtasks: list["Task"] = field(default_factory=list)

    @property
    def is_ready(self) -> bool:
        """A task is ready when all dependencies are completed."""
        return self.status == TaskStatus.PENDING and not self.dependencies

    def complete(self, result: Any):
        self.status = TaskStatus.COMPLETED
        self.result = result

    def fail(self, reason: str):
        self.status = TaskStatus.FAILED
        self.result = reason


class TaskDecomposer:
    """Decompose a high-level goal into a task graph."""

    def __init__(self):
        self.client = anthropic.Anthropic()

    def decompose(self, goal: str, max_tasks: int = 8) -> list[Task]:
        """Use an LLM to decompose a goal into tasks."""
        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            messages=[{
                "role": "user",
                "content": (
                    f"Decompose this goal into {max_tasks} or fewer concrete tasks. "
                    f"Return a JSON array where each element has: "
                    f"'id' (string), 'description' (string), "
                    f"'dependencies' (list of task IDs that must complete first).\n\n"
                    f"Goal: {goal}\n\nJSON:"
                ),
            }],
        )

        import json
        try:
            tasks_data = json.loads(response.content[0].text)
        except json.JSONDecodeError:
            return [Task("t1", goal)]

        return [
            Task(
                task_id=t["id"],
                description=t["description"],
                dependencies=t.get("dependencies", []),
            )
            for t in tasks_data
        ]
```

### 계층적 계획 (Hierarchical Planning)

```python
class HierarchicalPlanner:
    """Two-level planner: high-level plan -> detailed sub-plans."""

    def __init__(self):
        self.client = anthropic.Anthropic()
        self.decomposer = TaskDecomposer()

    def plan(self, goal: str) -> dict:
        """Create a hierarchical plan."""
        # Level 1: High-level task decomposition
        high_level_tasks = self.decomposer.decompose(goal, max_tasks=5)

        plan = {
            "goal": goal,
            "phases": [],
        }

        # Level 2: Decompose each high-level task further
        for task in high_level_tasks:
            subtasks = self.decomposer.decompose(
                f"Sub-goal: {task.description}", max_tasks=4
            )
            task.subtasks = subtasks
            plan["phases"].append({
                "phase": task.task_id,
                "description": task.description,
                "dependencies": task.dependencies,
                "steps": [
                    {"id": st.task_id, "description": st.description}
                    for st in subtasks
                ],
            })

        return plan

    def get_next_actions(self, plan: dict,
                         completed: set[str]) -> list[str]:
        """Return task IDs that are ready to execute."""
        ready = []
        for phase in plan["phases"]:
            if phase["phase"] in completed:
                continue
            deps_met = all(d in completed for d in phase["dependencies"])
            if deps_met:
                for step in phase["steps"]:
                    if step["id"] not in completed:
                        ready.append(step["id"])
                        break  # One step at a time per phase
        return ready
```

---

## 8. 계획 후 실행 패턴

### 개요

> **계획 후 실행 패턴 (Plan-and-Execute Pattern)**
>
> ```
> User Goal
>     |
>     v
> [Planner LLM] --> Plan (ordered steps)
>     |
>     v
> [Executor LLM] --> Execute step 1 --> Result 1
>     |                                      |
>     v                                      v
> [Re-planner] <--- Update plan based on results
>     |
>     v
> [Executor LLM] --> Execute step 2 --> Result 2
>     |
>     v
> ... (continue until goal achieved)
> ```

### 구현

```python
import json


class PlanAndExecuteAgent:
    """Agent that plans before acting, then iteratively refines."""

    def __init__(self, tools: dict[str, callable]):
        self.client = anthropic.Anthropic()
        self.tools = tools
        self.plan: list[dict] = []
        self.results: list[dict] = []

    def create_plan(self, goal: str) -> list[dict]:
        """Generate an initial plan from the goal."""
        tool_descriptions = "\n".join(
            f"- {name}: {func.__doc__ or 'No description'}"
            for name, func in self.tools.items()
        )

        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            messages=[{
                "role": "user",
                "content": (
                    f"Create a step-by-step plan to achieve this goal. "
                    f"Each step should specify a tool to use and the input.\n\n"
                    f"Available tools:\n{tool_descriptions}\n\n"
                    f"Goal: {goal}\n\n"
                    f"Return a JSON array of steps with keys: "
                    f"'step_number', 'description', 'tool', 'input'."
                ),
            }],
        )

        try:
            self.plan = json.loads(response.content[0].text)
        except json.JSONDecodeError:
            self.plan = [{"step_number": 1, "description": goal,
                          "tool": "none", "input": goal}]
        return self.plan

    def execute_step(self, step: dict) -> dict:
        """Execute a single step of the plan."""
        tool_name = step.get("tool", "")
        tool_input = step.get("input", "")

        if tool_name not in self.tools:
            return {
                "step": step["step_number"],
                "status": "error",
                "result": f"Unknown tool: {tool_name}",
            }

        try:
            result = self.tools[tool_name](tool_input)
            return {
                "step": step["step_number"],
                "status": "success",
                "result": str(result),
            }
        except Exception as e:
            return {
                "step": step["step_number"],
                "status": "error",
                "result": str(e),
            }

    def replan(self, goal: str, remaining_steps: list[dict],
               completed_results: list[dict]) -> list[dict]:
        """Re-plan based on execution results so far."""
        context = json.dumps(completed_results, indent=2)
        remaining = json.dumps(remaining_steps, indent=2)

        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            messages=[{
                "role": "user",
                "content": (
                    f"The original goal: {goal}\n\n"
                    f"Steps completed so far:\n{context}\n\n"
                    f"Remaining planned steps:\n{remaining}\n\n"
                    f"Based on the results so far, should the remaining plan "
                    f"be adjusted? Return the updated remaining steps as JSON. "
                    f"Keep steps that are still relevant, modify or remove "
                    f"those that are no longer needed."
                ),
            }],
        )

        try:
            return json.loads(response.content[0].text)
        except json.JSONDecodeError:
            return remaining_steps

    def run(self, goal: str, max_steps: int = 10,
            replan_interval: int = 3) -> dict:
        """Execute the full plan-and-execute loop."""
        self.create_plan(goal)
        self.results = []

        step_index = 0
        while step_index < len(self.plan) and step_index < max_steps:
            step = self.plan[step_index]
            result = self.execute_step(step)
            self.results.append(result)

            # Re-plan periodically or on failure
            if (
                result["status"] == "error"
                or (step_index + 1) % replan_interval == 0
            ):
                remaining = self.plan[step_index + 1:]
                if remaining:
                    self.plan = (
                        self.plan[:step_index + 1]
                        + self.replan(goal, remaining, self.results)
                    )

            step_index += 1

        return {
            "goal": goal,
            "total_steps": len(self.results),
            "results": self.results,
            "success": all(r["status"] == "success" for r in self.results),
        }
```

---

## 9. 계획을 위한 자기 성찰

### 성찰 기반 개선 (Reflection-Based Refinement)

자기 성찰(Self-Reflection)을 통해 에이전트는 자신의 계획과 출력을 평가한 후 반복적으로 개선할 수 있습니다.

```python
class ReflectiveAgent:
    """Agent that reflects on its outputs and iteratively improves."""

    def __init__(self):
        self.client = anthropic.Anthropic()
        self.reflection_history: list[dict] = []

    def generate(self, task: str) -> str:
        """Generate an initial response."""
        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2048,
            messages=[{"role": "user", "content": task}],
        )
        return response.content[0].text

    def reflect(self, task: str, output: str) -> dict:
        """Critically evaluate the output."""
        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            messages=[{
                "role": "user",
                "content": (
                    f"You are a critical reviewer. Evaluate this output "
                    f"for the given task.\n\n"
                    f"Task: {task}\n\n"
                    f"Output:\n{output}\n\n"
                    f"Provide a JSON response with:\n"
                    f"- 'score': 1-10 quality rating\n"
                    f"- 'strengths': list of what is good\n"
                    f"- 'weaknesses': list of problems\n"
                    f"- 'suggestions': specific improvements to make\n"
                    f"- 'is_satisfactory': boolean (true if score >= 7)"
                ),
            }],
        )

        import json
        try:
            reflection = json.loads(response.content[0].text)
        except json.JSONDecodeError:
            reflection = {
                "score": 5,
                "strengths": [],
                "weaknesses": ["Could not parse reflection"],
                "suggestions": ["Try again"],
                "is_satisfactory": False,
            }

        self.reflection_history.append(reflection)
        return reflection

    def refine(self, task: str, output: str, reflection: dict) -> str:
        """Improve the output based on reflection feedback."""
        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2048,
            messages=[{
                "role": "user",
                "content": (
                    f"Improve this output based on the feedback.\n\n"
                    f"Original task: {task}\n\n"
                    f"Current output:\n{output}\n\n"
                    f"Weaknesses: {reflection['weaknesses']}\n"
                    f"Suggestions: {reflection['suggestions']}\n\n"
                    f"Provide an improved version:"
                ),
            }],
        )
        return response.content[0].text

    def run(self, task: str, max_iterations: int = 3) -> dict:
        """Generate, reflect, and refine iteratively."""
        output = self.generate(task)
        iterations = []

        for i in range(max_iterations):
            reflection = self.reflect(task, output)
            iterations.append({
                "iteration": i + 1,
                "score": reflection["score"],
                "weaknesses": reflection["weaknesses"],
            })

            if reflection.get("is_satisfactory", False):
                break

            output = self.refine(task, output, reflection)

        return {
            "final_output": output,
            "iterations": iterations,
            "total_reflections": len(iterations),
            "final_score": iterations[-1]["score"] if iterations else 0,
        }
```

### 계획 비평가 (Plan Critic)

```python
class PlanCritic:
    """Evaluate and improve agent plans before execution."""

    CRITIQUE_DIMENSIONS = [
        "completeness",    # Does the plan cover all aspects of the goal?
        "feasibility",     # Can each step be executed with available tools?
        "efficiency",      # Is the plan doing unnecessary work?
        "correctness",     # Is the logical ordering of steps correct?
        "safety",          # Could any step cause harm or data loss?
    ]

    def __init__(self):
        self.client = anthropic.Anthropic()

    def critique(self, goal: str, plan: list[dict],
                 available_tools: list[str]) -> dict:
        """Critique a plan across multiple dimensions."""
        dimensions = "\n".join(
            f"- {d}" for d in self.CRITIQUE_DIMENSIONS
        )
        plan_text = json.dumps(plan, indent=2)
        tools_text = ", ".join(available_tools)

        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            messages=[{
                "role": "user",
                "content": (
                    f"Critique this plan for achieving the goal.\n\n"
                    f"Goal: {goal}\n"
                    f"Available tools: {tools_text}\n"
                    f"Plan:\n{plan_text}\n\n"
                    f"Evaluate on these dimensions:\n{dimensions}\n\n"
                    f"Return JSON with a score (1-10) and feedback for each "
                    f"dimension, plus an overall 'approved' boolean."
                ),
            }],
        )

        try:
            return json.loads(response.content[0].text)
        except json.JSONDecodeError:
            return {"approved": False, "feedback": "Could not parse critique"}

    def improve_plan(self, goal: str, plan: list[dict],
                     critique: dict) -> list[dict]:
        """Revise the plan based on critique feedback."""
        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            messages=[{
                "role": "user",
                "content": (
                    f"Revise this plan based on the critique.\n\n"
                    f"Goal: {goal}\n"
                    f"Current plan:\n{json.dumps(plan, indent=2)}\n\n"
                    f"Critique:\n{json.dumps(critique, indent=2)}\n\n"
                    f"Return the improved plan as a JSON array."
                ),
            }],
        )

        try:
            return json.loads(response.content[0].text)
        except json.JSONDecodeError:
            return plan
```

---

## 10. 메모리 증강 생성

### 종합 구현 (Putting It All Together)

메모리 증강 생성(Memory-Augmented Generation)은 여러 메모리 저장소로부터의 검색을 생성 단계와 결합합니다.

```python
class MemoryAugmentedAgent:
    """Agent that uses multiple memory sources to enhance generation."""

    def __init__(self, system_prompt: str = "You are a helpful assistant."):
        self.client = anthropic.Anthropic()
        self.system_prompt = system_prompt
        self.hybrid_memory = HybridMemory()
        self.planner = PlanAndExecuteAgent(tools={})

    def respond(self, user_message: str) -> str:
        """Generate a response augmented with memory context."""
        # Step 1: Build context from memory
        context_messages = self.hybrid_memory.build_context(user_message)

        # Step 2: Prepare the full message list
        messages = context_messages + [
            {"role": "user", "content": user_message},
        ]

        # Step 3: Generate response
        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2048,
            system=self.system_prompt,
            messages=messages,
        )

        assistant_msg = response.content[0].text

        # Step 4: Update memory with new interaction
        self.hybrid_memory.add_interaction(user_message, assistant_msg)

        return assistant_msg

    def respond_with_planning(self, user_message: str) -> dict:
        """For complex tasks, plan first, then execute with memory."""
        # Check if this is a complex task requiring planning
        complexity = self._assess_complexity(user_message)

        if complexity == "simple":
            return {"response": self.respond(user_message), "planned": False}

        # Plan the task
        plan = self.planner.create_plan(user_message)
        results = self.planner.run(user_message)

        # Generate final response incorporating plan results
        context = json.dumps(results["results"], indent=2)
        final_response = self.respond(
            f"Based on these intermediate results, answer the original "
            f"question: {user_message}\n\nResults:\n{context}"
        )

        return {
            "response": final_response,
            "planned": True,
            "plan_steps": len(plan),
            "plan_results": results,
        }

    def _assess_complexity(self, message: str) -> str:
        """Quick heuristic to decide if planning is needed."""
        # Simple heuristic: long messages or keywords suggest complexity
        complex_keywords = [
            "analyze", "compare", "research", "build",
            "create", "design", "investigate", "step by step",
        ]
        if len(message.split()) > 30:
            return "complex"
        if any(kw in message.lower() for kw in complex_keywords):
            return "complex"
        return "simple"
```

### 메모리 생명주기 관리 (Memory Lifecycle Management)

```python
class MemoryManager:
    """Manage memory lifecycle: creation, consolidation, and forgetting."""

    def __init__(self, ltm: LongTermMemory):
        self.ltm = ltm
        self.consolidation_threshold = 100  # Consolidate every N entries

    def should_remember(self, content: str, importance: float) -> bool:
        """Decide whether content is worth storing long-term."""
        # Skip very short or low-importance content
        if len(content.split()) < 5 and importance < 0.3:
            return False
        # Skip duplicate content
        existing = self.ltm.search(content, top_k=1)
        if existing and existing[0].content == content:
            return False
        return True

    def forget(self, max_age_days: int = 30,
               min_importance: float = 0.3):
        """Remove old, unimportant memories."""
        cutoff = time.time() - max_age_days * 86400
        to_remove = []
        for eid, entry in self.ltm.entries.items():
            if (
                entry.timestamp < cutoff
                and entry.importance < min_importance
                and entry.access_count < 3
            ):
                to_remove.append(eid)

        for eid in to_remove:
            del self.ltm.entries[eid]

        if to_remove:
            self.ltm._save()
        return len(to_remove)

    def consolidate(self):
        """Merge similar memories to reduce redundancy."""
        entries = list(self.ltm.entries.values())
        if len(entries) < self.consolidation_threshold:
            return 0

        # Group by memory type and find near-duplicates
        merged_count = 0
        seen_content = set()
        to_remove = []

        for i, entry in enumerate(entries):
            key = entry.content[:100].lower()
            if key in seen_content:
                to_remove.append(i)
                merged_count += 1
            else:
                seen_content.add(key)

        # Remove duplicates (by index in reverse order)
        entry_ids = list(self.ltm.entries.keys())
        for idx in sorted(to_remove, reverse=True):
            if idx < len(entry_ids):
                del self.ltm.entries[entry_ids[idx]]

        if merged_count > 0:
            self.ltm._save()
        return merged_count
```

---

## 연습문제

### 연습문제 1: 계층화된 메모리 저장소 (Tiered Memory Store)

세 개의 계층을 가진 메모리 시스템을 설계하세요: **핫(hot)** (마지막 5개 메시지, 항상 컨텍스트에 포함), **웜(warm)** (마지막 50개 메시지, 키워드 검색 가능), **콜드(cold)** (전체 기록, 디스크에 저장). `add()`, `search()`, `get_context()` 메서드를 가진 `TieredMemory` 클래스를 구현하세요. `search()` 메서드는 핫을 먼저 확인하고, 그 다음 웜, 마지막으로 콜드를 확인하며 일치하는 항목을 찾으면 즉시 반환해야 합니다.

<details>
<summary>정답 보기</summary>

```python
import json
from pathlib import Path
from collections import deque


class TieredMemory:
    """Three-tier memory: hot (in-context), warm (searchable), cold (on-disk)."""

    def __init__(self, hot_size: int = 5, warm_size: int = 50,
                 cold_path: str = "cold_memory.jsonl"):
        self.hot: deque[dict] = deque(maxlen=hot_size * 2)
        self.warm: deque[dict] = deque(maxlen=warm_size * 2)
        self.cold_path = Path(cold_path)

    def add(self, role: str, content: str):
        """Add a message, cascading overflow from hot -> warm -> cold."""
        msg = {"role": role, "content": content, "timestamp": time.time()}

        # If hot is full, overflow the oldest entry to warm
        if len(self.hot) == self.hot.maxlen:
            overflow = self.hot.popleft()
            # If warm is full, overflow to cold
            if len(self.warm) == self.warm.maxlen:
                cold_overflow = self.warm.popleft()
                self._write_cold(cold_overflow)
            self.warm.append(overflow)

        self.hot.append(msg)

    def _write_cold(self, msg: dict):
        """Append a message to the cold store on disk."""
        with open(self.cold_path, "a") as f:
            f.write(json.dumps(msg) + "\n")

    def _search_tier(self, tier: list[dict] | deque[dict],
                     query: str) -> list[dict]:
        """Keyword search within a tier."""
        query_terms = set(query.lower().split())
        results = []
        for msg in tier:
            content_terms = set(msg["content"].lower().split())
            overlap = len(query_terms & content_terms)
            if overlap > 0:
                results.append({"message": msg, "relevance": overlap})
        results.sort(key=lambda x: x["relevance"], reverse=True)
        return results

    def _search_cold(self, query: str) -> list[dict]:
        """Search the cold store on disk."""
        if not self.cold_path.exists():
            return []
        results = []
        query_terms = set(query.lower().split())
        with open(self.cold_path) as f:
            for line in f:
                msg = json.loads(line.strip())
                content_terms = set(msg["content"].lower().split())
                overlap = len(query_terms & content_terms)
                if overlap > 0:
                    results.append({"message": msg, "relevance": overlap})
        results.sort(key=lambda x: x["relevance"], reverse=True)
        return results

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        """Search across tiers: hot first, then warm, then cold."""
        # Try hot first
        results = self._search_tier(self.hot, query)
        if len(results) >= top_k:
            return results[:top_k]

        # Then warm
        warm_results = self._search_tier(self.warm, query)
        results.extend(warm_results)
        if len(results) >= top_k:
            return results[:top_k]

        # Finally cold
        cold_results = self._search_cold(query)
        results.extend(cold_results)
        return results[:top_k]

    def get_context(self) -> list[dict]:
        """Return hot tier messages for direct LLM context injection."""
        return [{"role": m["role"], "content": m["content"]} for m in self.hot]


# Test
memory = TieredMemory(hot_size=3, warm_size=5, cold_path="/tmp/test_cold.jsonl")

for i in range(20):
    memory.add("user", f"Message {i} about topic {'AI' if i % 2 == 0 else 'databases'}")
    memory.add("assistant", f"Response {i}")

print(f"Hot tier size: {len(memory.hot)}")
print(f"Warm tier size: {len(memory.warm)}")

results = memory.search("AI topic")
print(f"Search results for 'AI topic': {len(results)} found")
for r in results:
    print(f"  [{r['relevance']}] {r['message']['content'][:60]}")

context = memory.get_context()
print(f"\nContext messages: {len(context)}")
```

**핵심 설계 포인트:**
- 핫 계층(Hot Tier)은 항상 컨텍스트 윈도우에서 사용 가능 — 검색 비용 없음
- 웜 계층(Warm Tier)은 키워드 검색이 필요하지만 메모리 내에 있음 — 빠름
- 콜드 계층(Cold Tier)은 디스크에 있으며 최후의 수단으로만 접근 — 무한한 기록 처리
- 오버플로우(Overflow)가 자동으로 연쇄: 핫 -> 웜 -> 콜드
</details>

---

### 연습문제 2: 의존성 해결을 포함한 계획 수립 (Planning with Dependency Resolution)

의존성이 있는 작업 목록을 받아 유효한 실행 순서를 반환하는 `DependencyPlanner`를 구현하세요. 순환 의존성(Circular Dependency)을 감지하고 가능한 경우 병렬 실행(Parallel Execution)을 지원해야 합니다.

<details>
<summary>정답 보기</summary>

```python
from collections import defaultdict, deque


class CircularDependencyError(Exception):
    pass


class DependencyPlanner:
    """Plan execution order for tasks with dependencies."""

    def __init__(self):
        self.tasks: dict[str, dict] = {}
        self.graph: dict[str, list[str]] = defaultdict(list)
        self.in_degree: dict[str, int] = defaultdict(int)

    def add_task(self, task_id: str, description: str,
                 dependencies: list[str] | None = None):
        """Register a task with its dependencies."""
        self.tasks[task_id] = {
            "id": task_id,
            "description": description,
            "dependencies": dependencies or [],
        }
        if task_id not in self.in_degree:
            self.in_degree[task_id] = 0

        for dep in (dependencies or []):
            self.graph[dep].append(task_id)
            self.in_degree[task_id] += 1

    def topological_sort(self) -> list[list[str]]:
        """Return tasks grouped by execution wave (parallelizable)."""
        in_deg = dict(self.in_degree)
        waves = []

        while True:
            # Find all tasks with no remaining dependencies
            ready = [t for t in in_deg if in_deg[t] == 0]
            if not ready:
                break

            waves.append(sorted(ready))

            # Remove these tasks and update degrees
            for task_id in ready:
                for dependent in self.graph[task_id]:
                    in_deg[dependent] -= 1
                del in_deg[task_id]

        # Check for circular dependencies
        if in_deg:
            remaining = list(in_deg.keys())
            raise CircularDependencyError(
                f"Circular dependency detected among: {remaining}"
            )

        return waves

    def get_execution_plan(self) -> dict:
        """Generate a complete execution plan with parallel waves."""
        waves = self.topological_sort()

        plan = {
            "total_tasks": len(self.tasks),
            "total_waves": len(waves),
            "waves": [],
        }

        for i, wave in enumerate(waves):
            plan["waves"].append({
                "wave": i + 1,
                "parallel_tasks": [
                    {
                        "id": tid,
                        "description": self.tasks[tid]["description"],
                        "dependencies": self.tasks[tid]["dependencies"],
                    }
                    for tid in wave
                ],
            })

        return plan

    def critical_path(self) -> list[str]:
        """Find the longest dependency chain (critical path)."""
        waves = self.topological_sort()
        # Build a mapping of task_id to its wave index
        task_wave = {}
        for i, wave in enumerate(waves):
            for tid in wave:
                task_wave[tid] = i

        # Find the task in the last wave with the longest chain
        longest = []
        for task_id in waves[-1] if waves else []:
            chain = self._trace_chain(task_id)
            if len(chain) > len(longest):
                longest = chain
        return longest

    def _trace_chain(self, task_id: str) -> list[str]:
        """Trace the longest dependency chain ending at task_id."""
        deps = self.tasks[task_id]["dependencies"]
        if not deps:
            return [task_id]
        longest_prefix = []
        for dep in deps:
            chain = self._trace_chain(dep)
            if len(chain) > len(longest_prefix):
                longest_prefix = chain
        return longest_prefix + [task_id]


# Test
planner = DependencyPlanner()
planner.add_task("gather_reqs", "Gather requirements")
planner.add_task("design_api", "Design API schema", ["gather_reqs"])
planner.add_task("design_db", "Design database schema", ["gather_reqs"])
planner.add_task("impl_api", "Implement API endpoints", ["design_api"])
planner.add_task("impl_db", "Implement database migrations", ["design_db"])
planner.add_task("integration", "Integration testing", ["impl_api", "impl_db"])
planner.add_task("deploy", "Deploy to staging", ["integration"])

plan = planner.get_execution_plan()
for wave in plan["waves"]:
    tasks = [t["id"] for t in wave["parallel_tasks"]]
    print(f"Wave {wave['wave']}: {tasks}")

print(f"\nCritical path: {planner.critical_path()}")
print(f"Minimum waves needed: {plan['total_waves']}")

# Test circular dependency detection
try:
    bad_planner = DependencyPlanner()
    bad_planner.add_task("a", "Task A", ["c"])
    bad_planner.add_task("b", "Task B", ["a"])
    bad_planner.add_task("c", "Task C", ["b"])
    bad_planner.topological_sort()
except CircularDependencyError as e:
    print(f"\nDetected: {e}")
```

**핵심 포인트:**
- 칸 알고리즘(Kahn's Algorithm)을 사용한 위상 정렬(Topological Sort)이 작업을 병렬 실행 가능한 웨이브(Wave)로 그룹화
- 웨이브 1 작업은 의존성이 없으며 모두 동시에 실행 가능
- 순환 의존성 감지(Circular Dependency Detection)가 실행 전에 불가능한 계획을 포착
- 임계 경로 분석(Critical Path Analysis)이 병목 체인을 식별
</details>

---

### 연습문제 3: 중요도 점수를 활용한 요약 메모리 (Summary Memory with Importance Scoring)

중요도 점수(Importance Scoring)를 사용하여 무엇을 그대로 유지하고, 무엇을 요약하고, 무엇을 폐기할지 결정하는 `SmartSummaryMemory`를 구현하세요. 중요도가 0.8 이상인 메시지는 항상 원문 그대로 유지합니다. 중요도가 0.4~0.8인 메시지는 요약합니다. 0.4 미만의 메시지는 폐기합니다.

<details>
<summary>정답 보기</summary>

```python
class SmartSummaryMemory:
    """Memory that triages messages by importance before summarizing."""

    def __init__(self, max_verbatim: int = 10, max_summary_tokens: int = 500):
        self.max_verbatim = max_verbatim
        self.max_summary_tokens = max_summary_tokens
        self.verbatim: list[dict] = []
        self.summary_buffer: list[dict] = []
        self.summary: str = ""
        self.discarded_count: int = 0

    def score_importance(self, role: str, content: str) -> float:
        """Heuristic importance scoring."""
        score = 0.5  # Base score

        # Longer messages tend to be more important
        word_count = len(content.split())
        if word_count > 100:
            score += 0.2
        elif word_count > 50:
            score += 0.1

        # Key phrases that suggest importance
        high_importance = [
            "decision", "agreed", "action item", "deadline",
            "important", "must", "critical", "error", "bug",
            "conclusion", "summary", "result",
        ]
        medium_importance = [
            "question", "suggest", "option", "consider",
            "update", "progress", "status",
        ]

        content_lower = content.lower()
        if any(kw in content_lower for kw in high_importance):
            score += 0.3
        elif any(kw in content_lower for kw in medium_importance):
            score += 0.15

        # User messages slightly more important than assistant messages
        if role == "user":
            score += 0.05

        return min(1.0, score)

    def add(self, role: str, content: str):
        """Add a message, triaging by importance."""
        importance = self.score_importance(role, content)
        msg = {"role": role, "content": content, "importance": importance}

        if importance >= 0.8:
            self.verbatim.append(msg)
            # Overflow oldest verbatim to summary buffer
            if len(self.verbatim) > self.max_verbatim:
                overflow = self.verbatim.pop(0)
                self.summary_buffer.append(overflow)
        elif importance >= 0.4:
            self.summary_buffer.append(msg)
        else:
            self.discarded_count += 1

        # Compress summary buffer when it gets large
        if len(self.summary_buffer) > 10:
            self._compress_summary_buffer()

    def _compress_summary_buffer(self):
        """Compress the summary buffer into a running summary."""
        buffer_text = "\n".join(
            f"{m['role']}: {m['content'][:200]}" for m in self.summary_buffer
        )
        # In production, this would use an LLM call
        # Here we simulate with truncation
        if self.summary:
            self.summary += f"\n\nAdditional context: {buffer_text[:500]}"
        else:
            self.summary = f"Conversation summary: {buffer_text[:500]}"
        self.summary_buffer.clear()

    def get_context(self) -> list[dict]:
        """Build context: summary + verbatim messages."""
        messages = []
        if self.summary:
            messages.append({
                "role": "user",
                "content": f"[Previous conversation summary]: {self.summary}",
            })
            messages.append({
                "role": "assistant",
                "content": "I have the context from our previous discussion.",
            })
        for msg in self.verbatim:
            messages.append({"role": msg["role"], "content": msg["content"]})
        return messages

    def stats(self) -> dict:
        return {
            "verbatim_count": len(self.verbatim),
            "summary_buffer_count": len(self.summary_buffer),
            "has_summary": bool(self.summary),
            "discarded_count": self.discarded_count,
        }


# Test
memory = SmartSummaryMemory(max_verbatim=3)

messages = [
    ("user", "Hi there"),                                          # Low importance
    ("assistant", "Hello! How can I help?"),                       # Low importance
    ("user", "What's the weather?"),                               # Low importance
    ("user", "IMPORTANT: The deadline for the project is Friday"), # High importance
    ("assistant", "Noted. The critical deadline is Friday."),      # High importance
    ("user", "Can you suggest some options for the API design?"),  # Medium importance
    ("user", "We agreed to use REST over GraphQL for this project. This is the final decision."),
    ("assistant", "The conclusion is REST. Action item: update the design doc."),
    ("user", "Just checking in on progress"),                      # Medium importance
    ("user", "Error: the deployment pipeline is broken, must fix immediately"),
]

for role, content in messages:
    memory.add(role, content)

print("Stats:", memory.stats())
print("\nContext messages:")
for msg in memory.get_context():
    print(f"  [{msg['role']}] {msg['content'][:80]}...")
```

**핵심 설계 원칙:**
- 모든 메시지가 동일한 처리를 받을 필요는 없음 — 조기에 분류(Triage)
- 고중요도 메시지(결정, 오류, 마감일)는 절대 압축하지 않음
- 중간 중요도 메시지는 원문 유지 대신 요약
- 낮은 중요도의 잡담은 완전히 폐기하여 토큰 예산 절약
- 중요도 점수 산정기(Importance Scorer)는 프로덕션에서 LLM 기반 점수 산정기로 대체 가능
</details>

---

### 연습문제 4: 반복적 계획 개선 (Iterative Plan Refinement)

계획 생성, 단계 실행, 실패 감지, 나머지 계획 재생성을 지원하는 `RefinablePlan` 클래스를 구현하세요. 에이전트는 단계가 실패할 때 대안적 접근 방식으로 대체할 수 있어야 합니다.

<details>
<summary>정답 보기</summary>

```python
from dataclasses import dataclass, field
from typing import Callable, Any
from enum import Enum
import time


class StepStatus(Enum):
    PENDING = "pending"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"
    REPLACED = "replaced"


@dataclass
class PlanStep:
    step_id: int
    description: str
    action: Callable[[str], str] | None = None
    status: StepStatus = StepStatus.PENDING
    result: str = ""
    alternatives: list[Callable[[str], str]] = field(default_factory=list)
    max_retries: int = 2
    attempts: int = 0


class RefinablePlan:
    """A plan that adapts when steps fail."""

    def __init__(self):
        self.steps: list[PlanStep] = []
        self.execution_log: list[dict] = []

    def add_step(self, description: str, action: Callable[[str], str],
                 alternatives: list[Callable[[str], str]] | None = None,
                 max_retries: int = 2):
        """Add a step with optional alternative actions."""
        step = PlanStep(
            step_id=len(self.steps),
            description=description,
            action=action,
            alternatives=alternatives or [],
            max_retries=max_retries,
        )
        self.steps.append(step)

    def execute_step(self, step: PlanStep, context: str) -> bool:
        """Try to execute a step, falling back to alternatives on failure."""
        actions = [step.action] + step.alternatives

        for i, action in enumerate(actions):
            for attempt in range(step.max_retries + 1):
                step.attempts += 1
                try:
                    result = action(context)
                    step.status = StepStatus.SUCCESS
                    step.result = result

                    label = "primary" if i == 0 else f"alternative-{i}"
                    self.execution_log.append({
                        "step_id": step.step_id,
                        "description": step.description,
                        "action": label,
                        "attempt": attempt + 1,
                        "status": "success",
                        "result_preview": result[:100],
                    })
                    return True

                except Exception as e:
                    self.execution_log.append({
                        "step_id": step.step_id,
                        "description": step.description,
                        "action": "primary" if i == 0 else f"alternative-{i}",
                        "attempt": attempt + 1,
                        "status": "error",
                        "error": str(e),
                    })

        step.status = StepStatus.FAILED
        step.result = "All actions and alternatives exhausted"
        return False

    def run(self, initial_context: str = "") -> dict:
        """Execute the full plan with adaptive refinement."""
        context = initial_context
        completed = 0
        failed = 0

        for step in self.steps:
            success = self.execute_step(step, context)
            if success:
                context = step.result  # Chain output to next step
                completed += 1
            else:
                failed += 1
                # Try to skip and continue with existing context
                step.status = StepStatus.SKIPPED
                self.execution_log.append({
                    "step_id": step.step_id,
                    "description": step.description,
                    "action": "skip",
                    "status": "skipped",
                })

        return {
            "total_steps": len(self.steps),
            "completed": completed,
            "failed": failed,
            "skipped": sum(1 for s in self.steps if s.status == StepStatus.SKIPPED),
            "total_attempts": sum(s.attempts for s in self.steps),
            "execution_log": self.execution_log,
            "final_context": context,
        }


# Test with simulated failures
call_count = {"api_v1": 0, "api_v2": 0}


def fetch_data_v1(ctx: str) -> str:
    call_count["api_v1"] += 1
    if call_count["api_v1"] <= 2:
        raise ConnectionError("API v1 is down")
    return f"Data fetched via v1 for: {ctx[:30]}"


def fetch_data_v2(ctx: str) -> str:
    return f"Data fetched via v2 (fallback) for: {ctx[:30]}"


def process_data(ctx: str) -> str:
    return f"Processed: {ctx[:40]}"


def format_output(ctx: str) -> str:
    return f"Final report: {ctx[:50]}"


plan = RefinablePlan()
plan.add_step(
    "Fetch data from API",
    action=fetch_data_v1,
    alternatives=[fetch_data_v2],
    max_retries=1,
)
plan.add_step("Process and analyze data", action=process_data)
plan.add_step("Format output report", action=format_output)

result = plan.run("customer analytics Q4")
print(f"Completed: {result['completed']}/{result['total_steps']}")
print(f"Total attempts: {result['total_attempts']}")
print(f"Final output: {result['final_context']}")

print("\nExecution log:")
for entry in result["execution_log"]:
    status = entry.get("status", "unknown")
    desc = entry.get("description", "")
    print(f"  Step {entry['step_id']}: [{status}] {desc}")
```

**핵심 패턴:**
- 각 단계는 대안 액션(Alternative Action, 폴백 전략)을 가질 수 있음
- 재시도 로직(Retry Logic)은 단계별이 아닌 액션별로 적용
- 실패한 단계는 건너뛸 수 있어 파이프라인이 계속 진행
- 실행 로그(Execution Log)가 디버깅을 위한 완전한 관찰 가능성 제공
- 컨텍스트가 한 단계에서 다음 단계로 흐름 (체이닝, Chaining)
</details>

---

### 연습문제 5: 망각 곡선을 적용한 벡터 메모리 (Vector Memory with Forgetting Curve)

에빙하우스 망각 곡선(Ebbinghaus Forgetting Curve)을 사용하여 시간이 지남에 따라 메모리 중요도를 감쇠시키는 벡터 메모리 저장소를 구현하세요. 자주 접근되는 메모리는 더 느리게 감쇠합니다. 임계값(Threshold) 이하로 감쇠된 메모리를 제거하는 `consolidate()` 메서드를 포함하세요.

<details>
<summary>정답 보기</summary>

```python
import math
import numpy as np
from dataclasses import dataclass, field


@dataclass
class DecayingMemory:
    """A memory item with Ebbinghaus forgetting curve decay."""
    content: str
    embedding: np.ndarray
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 0
    base_importance: float = 0.5
    stability: float = 1.0  # Higher = slower forgetting

    def access(self):
        """Accessing a memory strengthens it."""
        self.access_count += 1
        self.last_accessed = time.time()
        # Each access increases stability (memory consolidation)
        self.stability = min(10.0, self.stability + 0.5)

    @property
    def retention(self) -> float:
        """
        Ebbinghaus forgetting curve: R = e^(-t/S)
        R = retention (0 to 1)
        t = time since last access (in hours)
        S = stability factor
        """
        hours_since_access = (time.time() - self.last_accessed) / 3600
        return math.exp(-hours_since_access / self.stability)

    @property
    def effective_importance(self) -> float:
        """Importance weighted by retention."""
        return self.base_importance * self.retention


class ForgettingVectorMemory:
    """Vector memory store with Ebbinghaus forgetting curve."""

    def __init__(self, forget_threshold: float = 0.1):
        self.memories: list[DecayingMemory] = []
        self.forget_threshold = forget_threshold

    def _mock_embedding(self, text: str) -> np.ndarray:
        """Generate a deterministic mock embedding for testing."""
        np.random.seed(hash(text) % 2**32)
        return np.random.randn(128).astype(np.float32)

    def add(self, content: str, importance: float = 0.5):
        """Store a new memory."""
        embedding = self._mock_embedding(content)
        mem = DecayingMemory(
            content=content,
            embedding=embedding,
            base_importance=importance,
        )
        self.memories.append(mem)

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """Search by similarity, weighted by retention and importance."""
        query_emb = self._mock_embedding(query)

        scored = []
        for mem in self.memories:
            # Cosine similarity
            sim = np.dot(query_emb, mem.embedding) / (
                np.linalg.norm(query_emb) * np.linalg.norm(mem.embedding) + 1e-8
            )

            # Weighted score: similarity + retention + importance
            score = (
                float(sim) * 0.5
                + mem.retention * 0.3
                + mem.effective_importance * 0.2
            )
            scored.append((score, mem))

        scored.sort(key=lambda x: x[0], reverse=True)

        results = []
        for score, mem in scored[:top_k]:
            mem.access()  # Strengthen retrieved memories
            results.append({
                "content": mem.content,
                "score": round(score, 4),
                "retention": round(mem.retention, 4),
                "effective_importance": round(mem.effective_importance, 4),
                "access_count": mem.access_count,
                "stability": round(mem.stability, 2),
            })
        return results

    def consolidate(self) -> dict:
        """Remove memories that have decayed below the threshold."""
        before = len(self.memories)
        forgotten = []

        self.memories = [
            m for m in self.memories
            if m.effective_importance >= self.forget_threshold
            or (kept := False) is not None  # Side effect: track forgotten
        ]

        # Simpler approach: filter and track
        surviving = []
        for mem in self.memories:
            if mem.effective_importance >= self.forget_threshold:
                surviving.append(mem)
            else:
                forgotten.append(mem.content[:50])

        # Re-check (the above logic was illustrative; use clean version)
        original = self.memories
        self.memories = [
            m for m in original
            if m.effective_importance >= self.forget_threshold
        ]
        forgotten_items = [
            m.content[:50] for m in original
            if m.effective_importance < self.forget_threshold
        ]

        return {
            "before": before,
            "after": len(self.memories),
            "forgotten": len(forgotten_items),
            "forgotten_previews": forgotten_items[:5],
        }

    def stats(self) -> dict:
        """Memory store statistics."""
        if not self.memories:
            return {"count": 0}

        retentions = [m.retention for m in self.memories]
        importances = [m.effective_importance for m in self.memories]

        return {
            "count": len(self.memories),
            "avg_retention": round(sum(retentions) / len(retentions), 4),
            "min_retention": round(min(retentions), 4),
            "avg_effective_importance": round(
                sum(importances) / len(importances), 4
            ),
            "total_accesses": sum(m.access_count for m in self.memories),
        }


# Test
store = ForgettingVectorMemory(forget_threshold=0.1)

# Add memories with varying importance
store.add("The project deadline is March 15", importance=0.9)
store.add("We discussed using PostgreSQL for the database", importance=0.7)
store.add("The weather was nice today", importance=0.2)
store.add("API rate limit is 1000 requests per minute", importance=0.8)
store.add("Someone mentioned a coffee shop nearby", importance=0.1)

print("Initial stats:", store.stats())

# Search strengthens relevant memories
results = store.search("database project deadline", top_k=3)
print("\nSearch results:")
for r in results:
    print(f"  [{r['score']:.3f}] {r['content'][:50]}... "
          f"(retention={r['retention']}, accesses={r['access_count']})")

print("\nStats after search:", store.stats())

# Consolidation
consolidation = store.consolidate()
print(f"\nConsolidation: {consolidation['before']} -> {consolidation['after']} "
      f"({consolidation['forgotten']} forgotten)")
```

**핵심 원칙:**
- 에빙하우스 망각 곡선(Ebbinghaus Forgetting Curve, `R = e^(-t/S)`)이 자연스러운 메모리 감쇠를 모델링
- 안정성(Stability)은 각 메모리 접근 시 증가 (간격 반복, Spaced Repetition 효과)
- 유효 중요도(Effective Importance) = 기본 중요도 x 보유율이므로 중요한 메모리도 접근되지 않으면 퇴색
- 통합(Consolidation)이 감쇠된 메모리를 영구적으로 제거하여 저장소 효율성 유지
- 메모리를 검색하면 강화됨 (인간의 기억과 동일한 원리)
</details>

---

## 다음 단계

[에이전트 평가와 벤치마크](./26_Agent_Evaluation_and_Benchmarks.md)에서는 확립된 벤치마크(Benchmark)를 사용하여 에이전트 성능을 체계적으로 측정하고 맞춤형 평가 프레임워크를 구축하는 방법을 살펴봅니다.
