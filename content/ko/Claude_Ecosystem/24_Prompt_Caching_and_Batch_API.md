# 프롬프트 캐싱과 Batch API

**이전**: [23. 비전 에이전트](./23_Vision_Agents.md) | **다음**: [25. RAG 패턴](./25_RAG_Patterns.md)

---

대규모로 운영할 때 API 비용과 지연 시간은 핵심 관심사가 됩니다. Anthropic은 이 두 가지를 줄이기 위한 두 가지 강력한 메커니즘을 제공합니다: 반복되는 콘텐츠의 재처리를 피하는 **프롬프트 캐싱(Prompt Caching)**과 비동기 워크로드에 50% 할인을 제공하는 **Message Batches API**입니다. 이 레슨에서는 두 기능을 심도 있게 다루며, 최대 비용 절감을 위해 이들을 결합하는 방법도 설명합니다.

**난이도**: ⭐⭐⭐

**사전 요구 사항**:
- Claude API 기초 ([레슨 15](./15_Claude_API_Fundamentals.md))
- 도구 사용과 함수 호출(Tool Use & Function Calling) ([레슨 16](./16_Tool_Use_and_Function_Calling.md))
- 모델과 가격 정책 ([레슨 19](./19_Models_and_Pricing.md))

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. `cache_control` 중단점으로 프롬프트 캐싱 구현
2. 캐시 가능한 콘텐츠 유형 식별 및 캐시 적중률 최적화
3. 캐싱을 통한 비용 절감 계산 (캐시된 토큰에 대해 최대 90% 절감)
4. 캐시 TTL 관리 및 워밍(Warming) 전략 구현
5. Message Batches API로 배치 요청 생성 및 모니터링
6. 캐싱과 배칭을 결합하여 최대 비용 효율성 달성
7. 문서 처리 및 평가를 위한 실제 파이프라인 구축

---

## 목차

1. [프롬프트 캐싱 기초](#1-프롬프트-캐싱-기초)
2. [캐시 가능한 콘텐츠 유형](#2-캐시-가능한-콘텐츠-유형)
3. [캐시 적중률과 비용 절감](#3-캐시-적중률과-비용-절감)
4. [TTL 관리와 캐시 워밍](#4-ttl-관리와-캐시-워밍)
5. [Message Batches API 개요](#5-message-batches-api-개요)
6. [배치 요청 생성과 모니터링](#6-배치-요청-생성과-모니터링)
7. [Batch API 가격과 제한](#7-batch-api-가격과-제한)
8. [캐싱과 배칭 결합](#8-캐싱과-배칭-결합)
9. [실전 패턴](#9-실전-패턴)
10. [연습 문제](#10-연습-문제)

---

## 1. 프롬프트 캐싱 기초

프롬프트 캐싱을 사용하면 여러 API 호출에서 동일하게 유지될 가능성이 높은 요청 부분을 표시할 수 있습니다. Claude가 캐시 적중(Cache Hit)을 만나면 해당 토큰의 처리를 건너뛰어 다음과 같은 결과를 얻습니다:

- 캐시된 입력 토큰에 대해 **90% 비용 절감**
- **상당히 낮은 지연 시간** (캐시된 토큰은 재처리되지 않음)
- **출력 품질 변화 없음** — 캐시 여부에 관계없이 응답이 동일

### 1.1 작동 방식

```
요청 1: [시스템 프롬프트 (5000 토큰)] + [사용자 메시지 (100 토큰)]
         ├── cache_control: ephemeral ──┘
         └── 캐시 WRITE: 5000 토큰 저장, 쓰기 비용 적용

요청 2: [시스템 프롬프트 (5000 토큰)] + [다른 사용자 메시지 (150 토큰)]
         ├── cache_control: ephemeral ──┘
         └── 캐시 HIT: 캐시에서 5000 토큰 읽기 (90% 저렴)
```

### 1.2 기본 사용법

```python
import anthropic

client = anthropic.Anthropic()

# 시스템 프롬프트가 요청 간에 캐시됩니다
response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    system=[
        {
            "type": "text",
            "text": "You are a legal expert assistant. You have deep knowledge of "
                    "contract law, intellectual property, and corporate governance. "
                    "Always cite relevant statutes and case law when applicable. "
                    "Format responses with clear headings and numbered points."
                    + "\n\n" + large_legal_reference_text,  # e.g., 4000+ tokens
            "cache_control": {"type": "ephemeral"},
        }
    ],
    messages=[
        {"role": "user", "content": "What are the key elements of a valid contract?"}
    ],
)

# 사용량 통계에서 캐시 성능 확인
print(f"Input tokens: {response.usage.input_tokens}")
print(f"Cache creation tokens: {response.usage.cache_creation_input_tokens}")
print(f"Cache read tokens: {response.usage.cache_read_input_tokens}")
```

### 1.3 캐시 제어 중단점(Cache Control Breakpoints)

`"type": "ephemeral"`이 포함된 `cache_control` 필드는 캐시 가능한 접두사의 끝을 표시합니다. 핵심 규칙:

- 요청당 최대 **4개의 캐시 중단점**을 배치할 수 있습니다
- 중단점 이전의 콘텐츠가 캐싱 대상입니다
- 최소 캐시 가능 접두사는 **1,024 토큰** (Claude Sonnet) 또는 **2,048 토큰** (Claude Haiku)
- 중단점은 콘텐츠 블록 경계에 배치해야 합니다

```python
# 다중 중단점 예시
response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    system=[
        {
            "type": "text",
            "text": base_system_prompt,  # 중단점 1: 거의 변경되지 않음
            "cache_control": {"type": "ephemeral"},
        }
    ],
    tools=tools_with_cache,  # 중단점 2: 도구 정의
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": long_document,  # 중단점 3: 문서 컨텍스트
                    "cache_control": {"type": "ephemeral"},
                },
                {
                    "type": "text",
                    "text": "Summarize the key findings.",  # 캐시 안 됨 (매번 달라짐)
                },
            ],
        }
    ],
)
```

---

## 2. 캐시 가능한 콘텐츠 유형

### 2.1 시스템 프롬프트(System Prompts)

가장 일반적인 캐싱 대상입니다. 시스템 프롬프트는 요청 간에 거의 변경되지 않습니다:

```python
SYSTEM_PROMPT = {
    "type": "text",
    "text": (
        "You are a customer support agent for Acme Corp.\n\n"
        "## Product Catalog\n"
        + product_catalog_text    # 3000+ 토큰의 제품 데이터
        + "\n\n## Support Policies\n"
        + support_policies_text   # 2000+ 토큰의 정책
    ),
    "cache_control": {"type": "ephemeral"},
}

# 모든 고객 쿼리가 캐시된 시스템 프롬프트를 재사용합니다
for query in customer_queries:
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        system=[SYSTEM_PROMPT],
        messages=[{"role": "user", "content": query}],
    )
```

### 2.2 도구 정의(Tool Definitions)

대규모 도구 세트는 캐싱으로 큰 이점을 얻습니다:

```python
# 리스트의 마지막 도구를 캐시하면 모든 도구가 캐시됩니다
tools = [
    {"name": "search_products", "description": "...", "input_schema": {...}},
    {"name": "check_inventory", "description": "...", "input_schema": {...}},
    {"name": "process_return", "description": "...", "input_schema": {...}},
    {"name": "create_ticket", "description": "...", "input_schema": {...}},
    {
        "name": "send_email",
        "description": "...",
        "input_schema": {...},
        "cache_control": {"type": "ephemeral"},  # 위의 모든 도구를 캐시
    },
]
```

### 2.3 긴 컨텍스트 문서

동일한 문서에 대해 여러 질문을 할 때:

```python
def multi_question_analysis(document: str, questions: list[str]) -> list[str]:
    """Ask multiple questions about the same document, caching it."""
    answers = []
    for question in questions:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2048,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"Document:\n\n{document}",
                            "cache_control": {"type": "ephemeral"},
                        },
                        {
                            "type": "text",
                            "text": f"Question: {question}",
                        },
                    ],
                }
            ],
        )
        answers.append(response.content[0].text)
    return answers
```

### 2.4 대화 기록(Conversation History)

다중 턴 대화에서 이전 턴을 캐시합니다:

```python
def chat_with_caching(messages: list[dict], new_message: str) -> str:
    """Continue a conversation with cached history."""
    # 마지막 기존 메시지에 캐싱 표시
    cached_messages = []
    for i, msg in enumerate(messages):
        if i == len(messages) - 1:
            # 마지막 기존 메시지에 cache_control 추가
            cached_msg = {
                "role": msg["role"],
                "content": [
                    {
                        "type": "text",
                        "text": msg["content"],
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
            }
            cached_messages.append(cached_msg)
        else:
            cached_messages.append(msg)

    # 새 메시지 추가 (캐시 안 됨)
    cached_messages.append({"role": "user", "content": new_message})

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2048,
        messages=cached_messages,
    )
    return response.content[0].text
```

---

## 3. 캐시 적중률과 비용 절감

### 3.1 가격 모델

| 토큰 유형 | 비용 (Sonnet) | 기본 대비 |
|---|---|---|
| 일반 입력 토큰 | 기본 가격 | 1.0x |
| 캐시 쓰기 토큰 | 1.25x 기본 | 25% 프리미엄 |
| 캐시 읽기 토큰 | 0.1x 기본 | **90% 할인** |

### 3.2 손익분기점 분석

캐싱은 캐시된 콘텐츠를 최소 **두 번** 재사용할 때 이익입니다:

```python
def calculate_cache_savings(
    cached_tokens: int,
    num_requests: int,
    base_input_price_per_mtok: float = 3.0,  # Sonnet 가격 예시
) -> dict:
    """Calculate cost savings from prompt caching."""
    # 캐싱 없이
    no_cache_cost = cached_tokens * num_requests * base_input_price_per_mtok / 1_000_000

    # 캐싱 적용: 1회 쓰기 + (N-1)회 읽기
    write_cost = cached_tokens * (base_input_price_per_mtok * 1.25) / 1_000_000
    read_cost = cached_tokens * (num_requests - 1) * (base_input_price_per_mtok * 0.1) / 1_000_000
    cache_cost = write_cost + read_cost

    savings = no_cache_cost - cache_cost
    savings_pct = (savings / no_cache_cost) * 100

    return {
        "without_caching": round(no_cache_cost, 4),
        "with_caching": round(cache_cost, 4),
        "savings": round(savings, 4),
        "savings_percent": round(savings_pct, 1),
        "break_even_requests": 2,  # 항상 2회 요청에서 손익분기
    }


# 예시: 10,000 캐시 토큰, 50회 요청
result = calculate_cache_savings(10_000, 50)
print(f"Without caching: ${result['without_caching']}")
print(f"With caching:    ${result['with_caching']}")
print(f"Savings:         ${result['savings']} ({result['savings_percent']}%)")
# Without caching: $1.5000
# With caching:    $0.1845
# Savings:         $1.3155 (87.7%)
```

### 3.3 캐시 성능 모니터링

```python
class CacheMonitor:
    """Track cache hit rates and cost savings across requests."""

    def __init__(self):
        self.total_requests = 0
        self.cache_hits = 0
        self.cache_writes = 0
        self.total_cached_tokens_read = 0
        self.total_cached_tokens_written = 0

    def record(self, usage):
        """Record usage stats from an API response."""
        self.total_requests += 1
        cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
        cache_write = getattr(usage, "cache_creation_input_tokens", 0) or 0

        if cache_read > 0:
            self.cache_hits += 1
            self.total_cached_tokens_read += cache_read
        if cache_write > 0:
            self.cache_writes += 1
            self.total_cached_tokens_written += cache_write

    @property
    def hit_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.cache_hits / self.total_requests

    def report(self) -> str:
        return (
            f"Requests: {self.total_requests}\n"
            f"Cache hits: {self.cache_hits} ({self.hit_rate:.1%})\n"
            f"Cache writes: {self.cache_writes}\n"
            f"Tokens read from cache: {self.total_cached_tokens_read:,}\n"
            f"Tokens written to cache: {self.total_cached_tokens_written:,}"
        )


# 사용법
monitor = CacheMonitor()

for query in queries:
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        system=[{"type": "text", "text": system_prompt, "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": query}],
    )
    monitor.record(response.usage)

print(monitor.report())
```

---

## 4. TTL 관리와 캐시 워밍

### 4.1 캐시 수명(Cache Lifetime)

캐시된 콘텐츠는 **5분 TTL**(Time To Live)을 갖습니다. 캐시된 접두사가 사용될 때마다 TTL이 리셋됩니다:

```
t=0:00  요청 1 → 캐시 WRITE (TTL 시작: 5분)
t=2:00  요청 2 → 캐시 HIT  (TTL 리셋: 지금부터 5분)
t=5:00  요청 3 → 캐시 HIT  (TTL 리셋: 지금부터 5분)
t=11:00 (6분간 요청 없음) → 캐시 만료
t=11:00 요청 4 → 캐시 WRITE (새 캐시 항목)
```

### 4.2 캐시 워밍 전략(Cache Warming Strategy)

요청 간에 간격이 있는 워크로드의 경우 사전에 캐시를 워밍합니다:

```python
import time
import threading


class CacheWarmer:
    """Keep a cache warm by sending periodic lightweight requests."""

    def __init__(self, system_prompt: list[dict], interval: int = 240):
        """
        Args:
            system_prompt: cache_control이 포함된 시스템 프롬프트.
            interval: 워밍 요청 간격 (초, 기본값 4분).
        """
        self.client = anthropic.Anthropic()
        self.system_prompt = system_prompt
        self.interval = interval
        self._running = False
        self._thread = None

    def start(self):
        """Start the cache warming background thread."""
        self._running = True
        self._thread = threading.Thread(target=self._warm_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop the cache warming thread."""
        self._running = False
        if self._thread:
            self._thread.join()

    def _warm_loop(self):
        while self._running:
            try:
                # 캐시를 유지하기 위한 최소 요청
                self.client.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=1,  # 출력 비용 최소화
                    system=self.system_prompt,
                    messages=[{"role": "user", "content": "ping"}],
                )
            except Exception:
                pass  # 프로덕션에서는 로깅
            time.sleep(self.interval)


# 사용법
warmer = CacheWarmer(
    system_prompt=[{
        "type": "text",
        "text": large_system_prompt,
        "cache_control": {"type": "ephemeral"},
    }],
    interval=240,  # 4분마다 워밍 (5분 TTL 전에)
)
warmer.start()

# ... 작업 수행 ...

warmer.stop()
```

### 4.3 캐시 친화적 요청 설계

```python
# 나쁜 예: 정적 콘텐츠 앞에 동적 콘텐츠가 오면 캐싱이 깨짐
messages = [
    {"role": "user", "content": f"Today is {date}. Analyze: {document}"}
]

# 좋은 예: 정적 콘텐츠를 먼저 cache_control과 함께, 동적 콘텐츠는 뒤에
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": f"Document:\n{document}",  # 정적, 캐시 가능
                "cache_control": {"type": "ephemeral"},
            },
            {
                "type": "text",
                "text": f"Today is {date}. Please analyze the document above.",  # 동적
            },
        ],
    }
]
```

---

## 5. Message Batches API 개요

Message Batches API를 사용하면 대량의 요청을 **50% 할인**으로 비동기적으로 전송할 수 있습니다. 배치 결과는 24시간 이내에 보장되지만 일반적으로 훨씬 빠르게 완료됩니다.

### 5.1 배치 사용 시기

| 사용 사례 | 실시간 API | Batch API |
|---|---|---|
| 대화형 채팅 | 예 | 아니오 |
| 문서 처리 (100개 이상) | 가능 | **권장** |
| 평가 파이프라인 | 가능 | **권장** |
| 데이터 라벨링/분류 | 가능 | **권장** |
| 대규모 콘텐츠 생성 | 가능 | **권장** |
| 지연 시간에 민감한 애플리케이션 | 예 | 아니오 |

### 5.2 배치 생명주기(Batch Lifecycle)

```
CREATE → in_progress → ended
                     ↗
         canceling →
```

- **created**: 배치 수락됨, 처리 대기 중
- **in_progress**: 요청 처리 중
- **canceling**: 취소 요청됨, 진행 중인 요청 완료 중
- **ended**: 모든 요청 완료 (results_counts 확인)

---

## 6. 배치 요청 생성과 모니터링

### 6.1 배치 생성

```python
import anthropic

client = anthropic.Anthropic()

# 배치 요청 정의
batch_requests = [
    {
        "custom_id": f"doc-{i}",
        "params": {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 1024,
            "messages": [
                {"role": "user", "content": f"Summarize this document:\n\n{doc}"}
            ],
        },
    }
    for i, doc in enumerate(documents)
]

# 배치 생성
batch = client.messages.batches.create(requests=batch_requests)

print(f"Batch ID: {batch.id}")
print(f"Status: {batch.processing_status}")
```

### 6.2 진행 상황 모니터링

```python
import time


def wait_for_batch(batch_id: str, poll_interval: int = 30) -> dict:
    """Poll a batch until completion."""
    while True:
        batch = client.messages.batches.retrieve(batch_id)

        print(
            f"Status: {batch.processing_status} | "
            f"Succeeded: {batch.request_counts.succeeded} | "
            f"Errored: {batch.request_counts.errored} | "
            f"Processing: {batch.request_counts.processing}"
        )

        if batch.processing_status == "ended":
            return batch

        time.sleep(poll_interval)


batch_result = wait_for_batch(batch.id)
```

### 6.3 결과 조회

```python
def get_batch_results(batch_id: str) -> dict[str, str]:
    """Retrieve all results from a completed batch."""
    results = {}

    for result in client.messages.batches.results(batch_id):
        custom_id = result.custom_id

        if result.result.type == "succeeded":
            message = result.result.message
            text = message.content[0].text
            results[custom_id] = text
        elif result.result.type == "errored":
            error = result.result.error
            results[custom_id] = f"ERROR: {error.type} - {error.message}"
        elif result.result.type == "expired":
            results[custom_id] = "EXPIRED: Request did not complete in time"

    return results


results = get_batch_results(batch.id)
for doc_id, summary in sorted(results.items()):
    print(f"\n{'='*60}")
    print(f"Document: {doc_id}")
    print(f"Summary: {summary[:200]}...")
```

### 6.4 배치 취소

```python
# 실행 중인 배치 취소
client.messages.batches.cancel(batch.id)

# 취소는 비동기 — 이미 처리 중인 요청은 완료됩니다
# 최종 상태 확인:
final = wait_for_batch(batch.id)
print(f"Succeeded: {final.request_counts.succeeded}")
print(f"Canceled: {final.request_counts.canceled}")
```

---

## 7. Batch API 가격과 제한

### 7.1 가격

| 기능 | 표준 API | Batch API |
|---|---|---|
| 입력 토큰 가격 | 기본 가격 | **기본의 50%** |
| 출력 토큰 가격 | 기본 가격 | **기본의 50%** |
| 프롬프트 캐싱 | 사용 가능 | 사용 가능 |
| 캐시 쓰기 토큰 | 1.25x 기본 | **0.625x 기본** (50% 할인) |
| 캐시 읽기 토큰 | 0.1x 기본 | **0.05x 기본** (50% 할인) |

### 7.2 제한 사항

- 배치당 최대 **10,000개 요청** (변경될 수 있음, 문서 확인)
- 각 요청은 표준 모델 제한을 따름 (최대 토큰 등)
- 배치는 완료되지 않으면 **24시간** 후 만료
- 레이트 제한(Rate Limit)은 실시간 API와 별도

### 7.3 비용 계산 예시

```python
def estimate_batch_cost(
    num_requests: int,
    avg_input_tokens: int,
    avg_output_tokens: int,
    cached_tokens: int = 0,
    model: str = "claude-sonnet-4-20250514",
) -> dict:
    """Estimate costs for batch vs real-time processing."""
    # Sonnet 가격 (백만 토큰당)
    prices = {
        "claude-sonnet-4-20250514": {"input": 3.0, "output": 15.0},
        "claude-haiku-4-20250514": {"input": 0.80, "output": 4.0},
    }
    p = prices[model]

    uncached_tokens = avg_input_tokens - cached_tokens

    # 실시간 비용
    rt_input_cost = (uncached_tokens * num_requests * p["input"]) / 1_000_000
    rt_cache_write = (cached_tokens * p["input"] * 1.25) / 1_000_000
    rt_cache_read = (cached_tokens * (num_requests - 1) * p["input"] * 0.1) / 1_000_000
    rt_output_cost = (avg_output_tokens * num_requests * p["output"]) / 1_000_000
    rt_total = rt_input_cost + rt_cache_write + rt_cache_read + rt_output_cost

    # 배치 비용 (모든 것의 50%)
    batch_total = rt_total * 0.5

    return {
        "realtime_cost": round(rt_total, 4),
        "batch_cost": round(batch_total, 4),
        "batch_savings": round(rt_total - batch_total, 4),
        "batch_savings_pct": 50.0,
    }
```

---

## 8. 캐싱과 배칭 결합

가장 강력한 비용 최적화는 두 가지를 결합하는 것입니다: 공유 컨텍스트를 캐시하고 배치를 통해 실행합니다.

### 8.1 캐시된 시스템 프롬프트를 사용한 배치

```python
# 모든 배치 요청이 동일한 시스템 프롬프트를 공유 → 캐시
system_prompt = [
    {
        "type": "text",
        "text": large_instructions + "\n\n" + reference_data,
        "cache_control": {"type": "ephemeral"},
    }
]

batch_requests = [
    {
        "custom_id": f"item-{i}",
        "params": {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 1024,
            "system": system_prompt,
            "messages": [
                {"role": "user", "content": f"Process: {item}"}
            ],
        },
    }
    for i, item in enumerate(items)
]

batch = client.messages.batches.create(requests=batch_requests)
```

### 8.2 캐시된 문서를 사용한 배치

```python
def batch_analyze_document(document: str, questions: list[str]) -> dict:
    """Ask many questions about a document using batch + caching."""
    batch_requests = [
        {
            "custom_id": f"q-{i}",
            "params": {
                "model": "claude-sonnet-4-20250514",
                "max_tokens": 2048,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": f"Document:\n\n{document}",
                                "cache_control": {"type": "ephemeral"},
                            },
                            {
                                "type": "text",
                                "text": question,
                            },
                        ],
                    }
                ],
            },
        }
        for i, question in enumerate(questions)
    ]

    batch = client.messages.batches.create(requests=batch_requests)
    final = wait_for_batch(batch.id)
    return get_batch_results(batch.id)
```

### 8.3 비용 비교 표

캐시된 토큰 5,000개와 캐시 안 된 토큰 500개가 있는 1,000개 요청의 경우:

| 전략 | 상대 비용 |
|---|---|
| 캐싱 없음, 실시간 | 100% (기준) |
| 캐싱만 | ~25% |
| 배칭만 | ~50% |
| **캐싱 + 배칭** | **~12.5%** |

---

## 9. 실전 패턴

### 9.1 문서 처리 파이프라인

```python
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor


class DocumentPipeline:
    """Process a large set of documents using batch + caching."""

    def __init__(self, extraction_schema: dict):
        self.client = anthropic.Anthropic()
        self.schema = extraction_schema
        self.system_prompt = [
            {
                "type": "text",
                "text": (
                    "You are a document processing assistant.\n"
                    f"Extract data according to this schema:\n"
                    f"{json.dumps(extraction_schema, indent=2)}\n\n"
                    "Return ONLY valid JSON matching the schema. No explanation."
                ),
                "cache_control": {"type": "ephemeral"},
            }
        ]

    def process_batch(self, documents: dict[str, str]) -> dict[str, dict]:
        """Process documents via batch API with caching."""
        batch_requests = [
            {
                "custom_id": doc_id,
                "params": {
                    "model": "claude-sonnet-4-20250514",
                    "max_tokens": 2048,
                    "system": self.system_prompt,
                    "messages": [
                        {"role": "user", "content": f"Extract from:\n\n{content}"}
                    ],
                },
            }
            for doc_id, content in documents.items()
        ]

        # 10,000개 단위로 배치 분할
        all_results = {}
        for i in range(0, len(batch_requests), 10_000):
            chunk = batch_requests[i : i + 10_000]
            batch = self.client.messages.batches.create(requests=chunk)
            final = wait_for_batch(batch.id)
            raw_results = get_batch_results(batch.id)

            for doc_id, text in raw_results.items():
                try:
                    all_results[doc_id] = json.loads(text)
                except json.JSONDecodeError:
                    all_results[doc_id] = {"error": "Failed to parse", "raw": text}

        return all_results


# 사용법
pipeline = DocumentPipeline(
    extraction_schema={
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "author": {"type": "string"},
            "date": {"type": "string"},
            "key_findings": {"type": "array", "items": {"type": "string"}},
            "sentiment": {"type": "string", "enum": ["positive", "negative", "neutral"]},
        },
    }
)

docs = {f"doc-{i}": text for i, text in enumerate(all_documents)}
results = pipeline.process_batch(docs)
```

### 9.2 LLM 평가 파이프라인

```python
class EvalPipeline:
    """Evaluate LLM outputs using Claude as a judge, via batch API."""

    def __init__(self, rubric: str):
        self.client = anthropic.Anthropic()
        self.system_prompt = [
            {
                "type": "text",
                "text": (
                    "You are an LLM output evaluator.\n\n"
                    f"## Rubric\n{rubric}\n\n"
                    "Score each response on the criteria in the rubric.\n"
                    "Return JSON: {\"scores\": {\"criterion\": score}, \"total\": N, \"reasoning\": \"...\"}"
                ),
                "cache_control": {"type": "ephemeral"},
            }
        ]

    def evaluate(self, test_cases: list[dict]) -> list[dict]:
        """
        Evaluate test cases via batch.

        Each test_case: {"id": str, "prompt": str, "response": str}
        """
        batch_requests = [
            {
                "custom_id": tc["id"],
                "params": {
                    "model": "claude-sonnet-4-20250514",
                    "max_tokens": 1024,
                    "system": self.system_prompt,
                    "messages": [
                        {
                            "role": "user",
                            "content": (
                                f"## Prompt\n{tc['prompt']}\n\n"
                                f"## Response to Evaluate\n{tc['response']}"
                            ),
                        }
                    ],
                },
            }
            for tc in test_cases
        ]

        batch = self.client.messages.batches.create(requests=batch_requests)
        final = wait_for_batch(batch.id)
        raw = get_batch_results(batch.id)

        results = []
        for tc_id, text in raw.items():
            try:
                parsed = json.loads(text)
                parsed["id"] = tc_id
                results.append(parsed)
            except json.JSONDecodeError:
                results.append({"id": tc_id, "error": text})

        return sorted(results, key=lambda r: r.get("total", 0), reverse=True)
```

### 9.3 하이브리드 실시간 + 배치 아키텍처

```python
class HybridProcessor:
    """Use real-time for urgent requests, batch for bulk processing."""

    def __init__(self, system_prompt: list[dict]):
        self.client = anthropic.Anthropic()
        self.system_prompt = system_prompt
        self.pending_batch = []

    def process_urgent(self, content: str) -> str:
        """Process a single item in real-time with caching."""
        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            system=self.system_prompt,
            messages=[{"role": "user", "content": content}],
        )
        return response.content[0].text

    def queue_for_batch(self, item_id: str, content: str):
        """Queue an item for batch processing."""
        self.pending_batch.append({
            "custom_id": item_id,
            "params": {
                "model": "claude-sonnet-4-20250514",
                "max_tokens": 1024,
                "system": self.system_prompt,
                "messages": [{"role": "user", "content": content}],
            },
        })

    def flush_batch(self) -> str:
        """Submit all queued items as a batch. Returns batch ID."""
        if not self.pending_batch:
            return None
        batch = self.client.messages.batches.create(requests=self.pending_batch)
        self.pending_batch = []
        return batch.id
```

---

## 10. 연습 문제

### 연습 문제 1: 캐시 성능 추적기

캐시 성능을 추적하고 보고하는 래퍼를 구축하세요:

```python
"""
Exercise 1 starter code — build a caching-aware API wrapper.
"""
import anthropic


class CachedClient:
    """API wrapper that manages caching and tracks performance."""

    def __init__(self):
        self.client = anthropic.Anthropic()
        self.stats = {
            "total_requests": 0,
            "cache_hits": 0,
            "cache_writes": 0,
            "tokens_saved": 0,
            "estimated_savings_usd": 0.0,
        }

    def create_message(self, system_text: str, user_message: str, **kwargs) -> str:
        """
        Send a message with automatic system prompt caching.

        - Wrap system_text with cache_control
        - Track cache hits/misses
        - Calculate cost savings
        """
        # TODO: Implement caching logic
        # TODO: Record usage stats
        # TODO: Calculate savings
        pass

    def report(self) -> str:
        """Return a formatted report of cache performance."""
        # TODO: Format and return stats
        pass
```

### 연습 문제 2: 배치 문서 분류기

문서를 카테고리로 분류하는 배치 파이프라인을 만드세요:

```python
"""
Exercise 2 starter code — batch document classification.
"""


class BatchClassifier:
    """Classify documents in bulk using the Batch API."""

    def __init__(self, categories: list[str]):
        self.client = anthropic.Anthropic()
        self.categories = categories

    def classify(self, documents: dict[str, str]) -> dict[str, dict]:
        """
        Classify each document into one of the categories.

        Args:
            documents: {doc_id: doc_text} mapping

        Returns:
            {doc_id: {"category": str, "confidence": float, "reasoning": str}}
        """
        # TODO: Build batch requests with cached system prompt
        # TODO: Submit batch
        # TODO: Wait for completion
        # TODO: Parse and return results
        pass
```

### 연습 문제 3: 캐시 워밍 서비스

여러 캐시 항목을 동시에 워밍하는 서비스를 구현하세요:

```python
"""
Exercise 3 starter code — multi-entry cache warming service.
"""
import threading


class CacheWarmingService:
    """Keep multiple cache entries warm simultaneously."""

    def __init__(self):
        self.client = anthropic.Anthropic()
        self.entries = {}  # name -> system_prompt
        self._running = False

    def register(self, name: str, system_prompt: list[dict], interval: int = 240):
        """Register a cache entry to keep warm."""
        # TODO: Store entry configuration
        pass

    def unregister(self, name: str):
        """Remove a cache entry from warming."""
        # TODO: Remove entry
        pass

    def start(self):
        """Start warming all registered entries."""
        # TODO: Start background threads
        pass

    def stop(self):
        """Stop all warming threads."""
        # TODO: Clean shutdown
        pass

    def status(self) -> dict:
        """Return the status of all cache entries."""
        # TODO: Report last warm time, hit rate, etc.
        pass
```

### 연습 문제 4: 비용 최적화 도구

API 사용량을 분석하고 최적화를 추천하는 도구를 구축하세요:

```python
"""
Exercise 4 starter code — API usage cost optimizer.
"""


class CostOptimizer:
    """Analyze API usage patterns and recommend cost optimizations."""

    def __init__(self):
        self.usage_log = []

    def log_request(self, usage: dict, request_metadata: dict):
        """Log a request's usage for analysis."""
        # TODO: Store usage data
        pass

    def analyze(self) -> dict:
        """
        Analyze logged usage and return recommendations.

        Returns:
            {
                "total_cost": float,
                "potential_savings": float,
                "recommendations": [
                    {
                        "type": "enable_caching" | "use_batch" | "switch_model",
                        "description": str,
                        "estimated_savings": float,
                    }
                ]
            }
        """
        # TODO: Identify requests with repeated system prompts (→ caching)
        # TODO: Identify non-urgent bulk requests (→ batching)
        # TODO: Identify simple tasks using expensive models (→ model switch)
        pass
```

---

**이전**: [23. 비전 에이전트](./23_Vision_Agents.md) | **다음**: [25. RAG 패턴](./25_RAG_Patterns.md)
