# 24. 프로덕션 LLM 패턴

## 학습 목표

- 프로덕션 워크로드를 위한 견고한 LLM 애플리케이션 아키텍처 설계
- 비용과 지연 시간을 줄이기 위한 캐싱 전략(시맨틱 캐시와 정확 매칭) 구현
- 신뢰성을 위한 폴백, 재시도, 멀티 모델 라우팅 패턴 구축
- LangSmith와 Phoenix를 사용한 디버깅 및 모니터링 관측 가능성 설정
- 프로덕션 준비 LLM 시스템을 위한 종합 배포 체크리스트 적용

---

## 이론과 원리

프로덕션 LLM 시스템은 프로토타입 코드가 무시할 수 있는 세 관심사를 가집니다 — **비용**(LLM 호출이 규모에서 비쌈), **지연**(사용자 대상 앱이 1초 미만 응답을 요구), **신뢰성**(LLM이 가끔 실패, 모델이 폐기됨, 제공자가 다운). 이 레슨의 패턴들 — 캐싱, 폴백, 라우팅, 관측성, A/B 테스트 — 은 작동하는 프로토타입을 예산에 구멍을 내지 않으면서 24/7 규모로 실행되는 시스템으로 전환하는 운영 계층입니다.

이 섹션은 다음을 다룹니다:

- **(A) 비용 경제학** — 돈이 어디로 가는가, 토큰당 수학, 중요한 자릿수.
- **(B) 캐싱** — exact-match 캐시(공짜 승리), 시맨틱 캐시(트레이드오프), 프롬프트 캐싱(제공자 측).
- **(C) 지연 최적화** — 토큰을 기다리는 계층 — TTFB, 스트리밍, 병렬화.
- **(D) 폴백과 재시도 패턴** — 회로 차단기, 멀티 제공자 페일오버, 지수 백오프.
- **(E) 멀티 모델 라우팅** — 가능할 때 작은 모델, 필요할 때 큰 모델 사용.
- **(F) 관측성** — LLM 앱을 위한 분산 추적, 반드시 로그해야 할 것, LangSmith / Phoenix.
- **(G) A/B 테스트와 점진적 롤아웃** — 변경이 프로덕션에서 실제로 도움이 되는지 측정.
- **(H) 속도 제한과 할당량** — 자신과 사용자를 비용 통제 불능에서 보호.

### A. 비용 경제학

토큰당 가격은 모델과 제공자에 따라 다양합니다. 2025년 후반 기준:
- 프런티어 모델(GPT-4 클래스) — $2-15 / 100만 입력 토큰, $10-60 / 100만 출력 토큰.
- 중간 등급(Claude Sonnet, GPT-4o-mini) — $0.15-3 / 100만 입력 토큰.
- 저렴(GPT-4o-mini, 오픈소스) — $0.05-0.5 / 100만 토큰.

전형적 RAG 쿼리 — ~1500 입력 토큰(시스템 프롬프트 + 검색된 청크 + 사용자 쿼리), ~300 출력 토큰. GPT-4 가격에서 — `(1500 · $5 + 300 · $15) / 10^6 = $0.012 / 쿼리`. 일 100만 쿼리에서 그것은 일 $12K — 연 $4M — *LLM 호출만으로*. 임베딩, 검색, 인프라가 추가.

이것이 LLM 앱의 중심 경제 현실입니다. 이 레슨의 모든 최적화 — 캐싱, 라우팅, 작은 모델 — 이 이 때문에 존재합니다. 캐시 적중률을 두 배로 하면 LLM 청구가 절반.

### B. 캐싱

**B.1 Exact-match 캐시.** 입력(프롬프트 + 파라미터) 해싱, 조회, 적중 시 캐시된 응답 반환. 구현이 사소함. 적중률은 입력 분포에 의존 — FAQ 형식 앱에서 높음(30-50%), 개방형 채팅에서 낮음(<5%).

**B.2 시맨틱 캐시.** 쿼리 임베딩, 비슷한 과거 쿼리 조회(cosine 유사도 > 임계값), 충분히 비슷하면 캐시된 응답 반환. 정확성을 더 높은 적중률과 맞바꿉니다. 위험 — *비슷해 보이지만* 실제로 다른 답을 요구하는 쿼리에 캐시된 응답을 반환.

```
exact_cache[hash(query)] = response  # 쉬움
semantic_cache: embed(query) → 과거 쿼리에서 top-1 → sim > 0.97이면 캐시된 응답 반환
```

임계값(0.95-0.99)이 레버. 더 높음 = 더 안전하지만 더 낮은 적중률. 프로덕션 시스템은 보통 보수적 임계값을 설정하고 표본 추출로 검증.

**B.3 제공자 측 프롬프트 캐싱**(Anthropic, OpenAI). 같은 프롬프트 접두사를 여러 번 호출할 때(예: 변하지 않는 긴 시스템 프롬프트 + 검색된 청크) 제공자가 접두사의 KV-cache를 캐시하고 재사용. 캐시된 부분의 입력 토큰 비용을 90% 감소시키고 TTFB를 상당히 향상시킵니다. 활성화는 무료, 접두사를 캐시 가능으로 표시만 하면 됩니다.

### C. 지연 최적화

LLM 앱의 사용자 인지 지연:

```
total = retrieve_latency + LLM_TTFB + decode_time
```

- **검색 지연** — 벡터 검색에 10-200ms; 웹 검색 API에 50-500ms. 최적화 — 인덱스 튜닝, 더 작은 임베딩 모델, 캐싱.
- **LLM TTFB** — 200ms-2s. 프롬프트 길이(prefill)와 제공자 부하에 의해 지배. 최적화 — 더 짧은 프롬프트, 프롬프트 캐싱, 더 작은 모델, 전용 용량.
- **디코드 시간** — 출력 토큰 수 × 토큰당 속도(~50-200 토큰/초). 최적화 — 더 짧은 응답, 스트리밍.

**스트리밍이 절대 지연보다 UX를 더 바꿉니다.** 0.5s에 시작해서 부드럽게 스트리밍하는 응답이 1.5s 기다린 후 던지는 것보다 더 빠르게 느껴집니다. 사용자 대상 응답은 항상 스트리밍.

### D. 폴백과 재시도 패턴

**D.1 지수 백오프 재시도.** 일시적 오류(속도 제한, 타임아웃, 5xx) — 1s, 2s, 4s, 8s 기다린 후 포기. 표준.

**D.2 회로 차단기.** 제공자가 한동안 실패해 왔다면 시도 중단하고 즉시 폴백으로 라우팅. 연쇄 실패 방지.

**D.3 멀티 제공자 페일오버.** 주 제공자 실패 → 보조 → 3차. 각각이 다른 LLM(GPT → Claude → Gemini → 오픈소스). 비용 — 각 제공자가 인증 키 필요, 스키마 차이를 추상화해야 함.

**D.4 정적 폴백.** 모든 LLM 호출이 실패하면 정해진 응답 반환 — "지금 문제가 있습니다, 나중에 다시 시도하세요." 사용자에게 500 오류보다 낫습니다.

패턴 — **항상 작동하는 응답 경로 유지**, 비록 격하되더라도.

### E. 멀티 모델 라우팅

대부분의 프로덕션 트래픽은 단순합니다. 쉬운 쿼리를 저렴한 모델로 라우팅하고 비싼 모델을 어려운 쿼리에 예약하면 품질 손실 없이 비용을 5-10배 줄입니다.

**E.1 휴리스틱 라우팅.** 규칙 기반 — 짧은 쿼리 → 저렴한 모델, 긴 쿼리 → 비싼 모델. 거칠지만 극단에서 효과적.

**E.2 분류기 기반 라우팅.** 작은 분류기가 난이도를 예측; 그에 따라 라우팅. 라벨된 데이터셋에 학습(예: 과거 A/B 테스트에서). 휴리스틱보다 낫지만 학습/유지 비용 추가.

**E.3 캐스케이딩.** 저렴한 모델 먼저 시도; 신뢰도가 낮거나 출력이 검증 실패하면 비싼 모델로 재시도. 표준 패턴.

**E.4 라우터로서의 모델.** 작은 LLM(또는 저렴한 모델 자체)이 쿼리를 검사하고 어느 다운스트림 모델이 처리할지 결정. 가장 유연, 추가 LLM 호출 추가.

### F. 관측성

LLM 앱은 "로직"이 프롬프트와 가중치에 암묵적이라 전통적 앱보다 디버그하기 더 어렵습니다. 관측성은 다음을 로깅 요구:

- 전체 입력 프롬프트(모든 검색된 청크 등 포함)
- 모델 정체성과 파라미터(temperature, top_p 등)
- 전체 출력
- 지연 분해(retrieve / TTFB / decode)
- 토큰 수, 비용
- 도구 호출과 그 결과(에이전트라면)
- 수집된 사용자 피드백(엄지 위/아래)

**LangSmith**(LangChain), **Phoenix**(Arize), **Langfuse**, **Helicone** — LLM 앱을 위한 호스팅/OSS 관측성 플랫폼. 각각이 LLM 호출을 계측(대부분 LLM 프레임워크와 한 줄 설정)하고 추적 UI, 지연 분해, 비용 대시보드, 오류 분석 제공.

관측성 비용은 작음(LLM 호출당 추적 백엔드에 한 HTTP 호출); 가치는 거대(없이 프로덕션 버그 디버그는 본질적으로 불가능).

### G. A/B 테스트와 점진적 롤아웃

오프라인 평가에서 좋아 보이는 변경이 프로덕션에서 도움이 되지 않거나(또는 적극적으로 해를 끼치는) 경우가 종종 있습니다. 알 수 있는 유일한 방법 — 두 버전을 실제 사용자에게 서빙하고 참여 지표 비교.

표준 접근:
1. 후보를 프로덕션 버전과 함께 배포.
2. 트래픽의 작은 비율(1-10%)을 후보로 라우팅.
3. 참여 지표 추적(응답 수락, 후속 비율, 사용자 평가, 다운스트림 전환).
4. 통계적 가설 검정으로 유의성 결정.
5. 후보가 이기면(또는 지지 않으면) 트래픽 점진적 증가.

LLM 특화 변경에 대해 또한 추적 — 요청당 비용, 지연 p50/p95/p99, 오류율, 거부율.

### H. 속도 제한과 할당량

**사용자별 제한** — 어떤 한 사용자도 용량을 독점하거나 놀라운 청구를 일으키지 않도록 방지. 토큰 기반(예: 일 100K 토큰)이 요청 기반보다 비용과 더 정렬.

**전역 제한** — 통제 불능 시나리오로부터 보호 — 루프하는 버그, 바이럴 사건. 분/시간/일당 총 토큰에 단단한 상한.

**비용 천장** — 비용이 임계값을 초과하면 LLM 호출을 비활성화하는 별도 예산 가드. 마지막 방어선.

### 이론에서 아래 함수들로

- §1 (아키텍처) — §A-§H 관심사를 시스템 수준에서 틀.
- §2 (캐싱) — §B.1 exact와 §B.2 시맨틱 캐시 구현.
- §3 (비용 최적화) — §A와 §E를 적용하여 쿼리당 비용 감소.
- §4 (폴백/재시도) — §D 패턴 구현.
- §5 (A/B 테스트) — 트래픽 분할과 유의성 테스트로 §G 구현.
- §6 (관측성) — §F에 따라 LangSmith와 Phoenix 연결.
- §7 (속도 제한 / 라우팅) — §E 멀티 모델 라우팅을 §H 속도 제한과 결합.
- §8 (배포 체크리스트) — §A-§H를 출시 준비 체크리스트로 종합.

---

## 1. LLM 애플리케이션 아키텍처

### 프로덕션 아키텍처 개요

> **프로덕션 LLM 스택**
>
> ```
> 클라이언트 요청
>     |
>     v
> [Rate Limiter] -> [Input Validator] -> [Cache Layer]
>     |                                        |
>     | (캐시 미스)                        (캐시 히트)
>     v                                        |
> [Router] -> [Model A / Model B / ...]        |
>     |                                        |
>     v                                        |
> [Output Validator] -> [Content Filter]       |
>     |                                        |
>     v                                        v
> [Response Logger] ----------------------> 클라이언트
> ```

### 아키텍처 비교

| 패턴 | 지연 | 비용 | 복잡도 | 신뢰성 |
|------|------|------|--------|--------|
| 직접 API 호출 | 높음 | 높음 | 낮음 | 낮음 |
| + 캐싱 | 중간 | 중간 | 중간 | 중간 |
| + 폴백 모델 | 중간 | 중간 | 중간 | 높음 |
| + 멀티 모델 라우터 | 낮음-중간 | 낮음-중간 | 높음 | 매우 높음 |
| 전체 프로덕션 스택 | 낮음-중간 | 낮음 | 높음 | 매우 높음 |

### 기본 애플리케이션 구조

```python
from dataclasses import dataclass, field
from typing import Any
import time
import uuid
import logging

logger = logging.getLogger(__name__)

@dataclass
class LLMRequest:
    """표준화된 요청 객체."""
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    messages: list[dict] = field(default_factory=list)
    model: str = "gpt-4o"
    temperature: float = 0.3
    max_tokens: int = 2048
    metadata: dict = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

@dataclass
class LLMResponse:
    """표준화된 응답 객체."""
    request_id: str
    content: str
    model: str
    tokens_input: int
    tokens_output: int
    latency_ms: float
    cached: bool = False
    metadata: dict = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.tokens_input + self.tokens_output

    @property
    def estimated_cost(self) -> float:
        """모델 가격 기반 비용 추정 (근사치)."""
        pricing = {
            "gpt-4o": (2.50, 10.00),           # 1M 토큰당 (입력, 출력)
            "gpt-4o-mini": (0.15, 0.60),
            "claude-sonnet-4-20250514": (3.00, 15.00),
            "claude-haiku-4-20250514": (0.25, 1.25),
        }
        input_rate, output_rate = pricing.get(self.model, (5.0, 15.0))
        return (
            self.tokens_input * input_rate / 1_000_000
            + self.tokens_output * output_rate / 1_000_000
        )
```

---

## 2. 캐싱 전략

### 정확 매칭 캐시

```python
import hashlib
import json
import sqlite3
import time
from pathlib import Path

class ExactMatchCache:
    """LLM 응답을 위한 SQLite 기반 정확 매칭 캐시."""

    def __init__(self, db_path: str = "llm_cache.db", ttl_hours: float = 24):
        self.db_path = db_path
        self.ttl_seconds = ttl_hours * 3600
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache (
                    cache_key TEXT PRIMARY KEY,
                    response TEXT NOT NULL,
                    model TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    hit_count INTEGER DEFAULT 0
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_created_at ON cache(created_at)
            """)

    def _make_key(self, messages: list[dict], model: str,
                  temperature: float) -> str:
        """요청 파라미터에서 결정적 캐시 키 생성."""
        payload = json.dumps({
            "messages": messages,
            "model": model,
            "temperature": temperature,
        }, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()

    def get(self, messages: list[dict], model: str,
            temperature: float) -> str | None:
        """캐시된 응답이 있고 만료되지 않았으면 검색."""
        key = self._make_key(messages, model, temperature)
        cutoff = time.time() - self.ttl_seconds

        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT response FROM cache WHERE cache_key = ? AND created_at > ?",
                (key, cutoff),
            ).fetchone()

            if row:
                conn.execute(
                    "UPDATE cache SET hit_count = hit_count + 1 WHERE cache_key = ?",
                    (key,),
                )
                return row[0]
        return None

    def put(self, messages: list[dict], model: str,
            temperature: float, response: str):
        """응답을 캐시에 저장."""
        key = self._make_key(messages, model, temperature)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO cache (cache_key, response, model, created_at) "
                "VALUES (?, ?, ?, ?)",
                (key, response, model, time.time()),
            )

    def evict_expired(self):
        """만료된 항목 제거."""
        cutoff = time.time() - self.ttl_seconds
        with sqlite3.connect(self.db_path) as conn:
            deleted = conn.execute(
                "DELETE FROM cache WHERE created_at < ?", (cutoff,)
            ).rowcount
            logger.info(f"만료된 캐시 항목 {deleted}개 제거")

    def stats(self) -> dict:
        with sqlite3.connect(self.db_path) as conn:
            total = conn.execute("SELECT COUNT(*) FROM cache").fetchone()[0]
            total_hits = conn.execute("SELECT SUM(hit_count) FROM cache").fetchone()[0] or 0
            return {"total_entries": total, "total_hits": total_hits}
```

### 시맨틱 캐시

```python
import numpy as np
from openai import OpenAI

client = OpenAI()

class SemanticCache:
    """의미적으로 유사한 쿼리를 매칭하는 캐시."""

    def __init__(self, similarity_threshold: float = 0.92,
                 max_entries: int = 10000):
        self.threshold = similarity_threshold
        self.max_entries = max_entries
        self.entries: list[dict] = []  # 프로덕션: 벡터 DB 사용
        self.embeddings: list[np.ndarray] = []

    def _get_embedding(self, text: str) -> np.ndarray:
        """텍스트의 임베딩 벡터를 얻음."""
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=text,
        )
        return np.array(response.data[0].embedding)

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    def _query_to_key(self, messages: list[dict]) -> str:
        """메시지에서 의미 있는 쿼리를 추출."""
        # 마지막 사용자 메시지를 캐시 키로 사용
        user_messages = [m["content"] for m in messages if m["role"] == "user"]
        return user_messages[-1] if user_messages else ""

    def get(self, messages: list[dict]) -> str | None:
        """의미적으로 유사한 캐시된 응답을 찾음."""
        if not self.entries:
            return None

        query = self._query_to_key(messages)
        query_embedding = self._get_embedding(query)

        # 가장 유사한 항목 찾기
        best_score = 0.0
        best_idx = -1

        for i, emb in enumerate(self.embeddings):
            score = self._cosine_similarity(query_embedding, emb)
            if score > best_score:
                best_score = score
                best_idx = i

        if best_score >= self.threshold:
            logger.info(f"시맨틱 캐시 히트 (유사도={best_score:.4f})")
            return self.entries[best_idx]["response"]

        return None

    def put(self, messages: list[dict], response: str):
        """쿼리-응답 쌍을 시맨틱 캐시에 저장."""
        query = self._query_to_key(messages)
        embedding = self._get_embedding(query)

        self.entries.append({
            "query": query,
            "response": response,
            "timestamp": time.time(),
        })
        self.embeddings.append(embedding)

        # 한도 초과 시 가장 오래된 항목 제거
        if len(self.entries) > self.max_entries:
            self.entries.pop(0)
            self.embeddings.pop(0)

# 결합 캐시 전략
class TieredCache:
    """2단계 캐시: 정확 매칭 우선, 그다음 시맨틱."""

    def __init__(self):
        self.exact = ExactMatchCache(ttl_hours=48)
        self.semantic = SemanticCache(similarity_threshold=0.93)

    def get(self, messages: list[dict], model: str,
            temperature: float) -> tuple[str | None, str]:
        """정확 매칭 우선 시도, 그다음 시맨틱. (응답, 캐시_유형)을 반환."""
        # 1단계: 정확 매칭
        exact_result = self.exact.get(messages, model, temperature)
        if exact_result:
            return exact_result, "exact"

        # 2단계: 시맨틱 매칭 (저온도 요청에만)
        if temperature <= 0.3:
            semantic_result = self.semantic.get(messages)
            if semantic_result:
                return semantic_result, "semantic"

        return None, "miss"

    def put(self, messages: list[dict], model: str,
            temperature: float, response: str):
        self.exact.put(messages, model, temperature, response)
        if temperature <= 0.3:
            self.semantic.put(messages, response)
```

---

## 3. 비용 및 지연 최적화

### 비용 추적 및 예산

```python
from collections import defaultdict
from datetime import datetime, timedelta
import threading

class CostTracker:
    """LLM 지출 예산 추적 및 적용."""

    PRICING = {
        # (입력_1M_토큰당, 출력_1M_토큰당)
        "gpt-4o": (2.50, 10.00),
        "gpt-4o-mini": (0.15, 0.60),
        "claude-sonnet-4-20250514": (3.00, 15.00),
        "claude-haiku-4-20250514": (0.25, 1.25),
    }

    def __init__(self, daily_budget: float = 50.0, monthly_budget: float = 1000.0):
        self.daily_budget = daily_budget
        self.monthly_budget = monthly_budget
        self._daily_spend: dict[str, float] = defaultdict(float)  # 날짜 -> 지출
        self._monthly_spend: dict[str, float] = defaultdict(float)  # 월 -> 지출
        self._lock = threading.Lock()

    def record(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """토큰 사용을 기록하고 비용 반환."""
        input_rate, output_rate = self.PRICING.get(model, (5.0, 15.0))
        cost = (
            input_tokens * input_rate / 1_000_000
            + output_tokens * output_rate / 1_000_000
        )

        today = datetime.now().strftime("%Y-%m-%d")
        month = datetime.now().strftime("%Y-%m")

        with self._lock:
            self._daily_spend[today] += cost
            self._monthly_spend[month] += cost

        return cost

    def check_budget(self) -> dict:
        """예산 한도 내에 있는지 확인."""
        today = datetime.now().strftime("%Y-%m-%d")
        month = datetime.now().strftime("%Y-%m")

        daily_spend = self._daily_spend.get(today, 0)
        monthly_spend = self._monthly_spend.get(month, 0)

        return {
            "daily_spend": round(daily_spend, 4),
            "daily_budget": self.daily_budget,
            "daily_remaining": round(self.daily_budget - daily_spend, 4),
            "daily_exceeded": daily_spend >= self.daily_budget,
            "monthly_spend": round(monthly_spend, 4),
            "monthly_budget": self.monthly_budget,
            "monthly_remaining": round(self.monthly_budget - monthly_spend, 4),
            "monthly_exceeded": monthly_spend >= self.monthly_budget,
        }

    def can_proceed(self) -> bool:
        """새 요청이 예산 내에 있는지 확인."""
        budget = self.check_budget()
        return not budget["daily_exceeded"] and not budget["monthly_exceeded"]
```

### 프롬프트 최적화

```python
class PromptOptimizer:
    """품질 저하 없이 토큰 수를 줄임."""

    @staticmethod
    def compress_system_prompt(prompt: str) -> str:
        """시스템 프롬프트에서 불필요한 장황함을 제거."""
        import re
        prompt = re.sub(r"\n{3,}", "\n\n", prompt)
        prompt = re.sub(r" {2,}", " ", prompt)
        return prompt.strip()

    @staticmethod
    def truncate_context(messages: list[dict], max_context_tokens: int = 4000,
                         preserve_last_n: int = 4) -> list[dict]:
        """최근 메시지를 보존하면서 대화 히스토리를 잘라냄."""
        if len(messages) <= preserve_last_n:
            return messages

        # 시스템 메시지와 마지막 N개 메시지는 항상 유지
        system = [m for m in messages if m["role"] == "system"]
        non_system = [m for m in messages if m["role"] != "system"]

        preserved = non_system[-preserve_last_n:]

        # 버려진 메시지를 요약
        dropped = non_system[:-preserve_last_n]
        if dropped:
            summary_msg = {
                "role": "user",
                "content": f"[이전 {len(dropped)}개 메시지 요약: "
                           f"대화에서 다양한 주제가 논의되었습니다. "
                           f"아래 최근 컨텍스트부터 계속해 주세요.]",
            }
            return system + [summary_msg] + preserved

        return system + preserved

    @staticmethod
    def select_model_by_complexity(messages: list[dict]) -> str:
        """간단한 작업에는 저렴한 모델로 라우팅."""
        last_user_msg = ""
        for m in reversed(messages):
            if m["role"] == "user":
                last_user_msg = m["content"]
                break

        # 모델 선택을 위한 간단한 휴리스틱
        word_count = len(last_user_msg.split())

        # 짧고 간단한 쿼리 -> 저렴한 모델
        if word_count < 30 and "?" in last_user_msg:
            return "gpt-4o-mini"

        # 복잡한 추론 -> 강력한 모델
        complex_keywords = ["analyze", "compare", "design", "architect",
                          "debug", "optimize", "explain in detail"]
        if any(kw in last_user_msg.lower() for kw in complex_keywords):
            return "gpt-4o"

        return "gpt-4o-mini"  # 기본값: 저렴한 모델
```

---

## 4. 폴백 및 재시도 패턴

### 멀티 프로바이더 폴백

```python
from openai import OpenAI
from anthropic import Anthropic
import time

class LLMRouter:
    """폴백이 있는 여러 프로바이더 간 요청 라우팅."""

    def __init__(self):
        self.openai = OpenAI()
        self.anthropic = Anthropic()
        self.cost_tracker = CostTracker()

    def _call_openai(self, messages: list[dict], model: str,
                     temperature: float, max_tokens: int) -> LLMResponse:
        start = time.time()
        response = self.openai.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        latency = (time.time() - start) * 1000
        usage = response.usage
        self.cost_tracker.record(model, usage.prompt_tokens, usage.completion_tokens)
        return LLMResponse(
            request_id="",
            content=response.choices[0].message.content,
            model=model,
            tokens_input=usage.prompt_tokens,
            tokens_output=usage.completion_tokens,
            latency_ms=latency,
        )

    def _call_anthropic(self, messages: list[dict], model: str,
                        temperature: float, max_tokens: int) -> LLMResponse:
        start = time.time()
        # OpenAI 형식을 Anthropic 형식으로 변환
        system = ""
        anthropic_messages = []
        for m in messages:
            if m["role"] == "system":
                system = m["content"]
            else:
                anthropic_messages.append(m)

        response = self.anthropic.messages.create(
            model=model,
            system=system,
            messages=anthropic_messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        latency = (time.time() - start) * 1000
        self.cost_tracker.record(
            model, response.usage.input_tokens, response.usage.output_tokens
        )
        return LLMResponse(
            request_id="",
            content=response.content[0].text,
            model=model,
            tokens_input=response.usage.input_tokens,
            tokens_output=response.usage.output_tokens,
            latency_ms=latency,
        )

    def call(self, request: LLMRequest) -> LLMResponse:
        """자동 폴백 체인으로 LLM 호출."""
        # 폴백 체인 정의
        fallback_chain = [
            ("openai", request.model, request.temperature),
            ("openai", "gpt-4o-mini", request.temperature),
            ("anthropic", "claude-sonnet-4-20250514", request.temperature),
            ("anthropic", "claude-haiku-4-20250514", request.temperature),
        ]

        last_error = None

        for provider, model, temp in fallback_chain:
            try:
                if provider == "openai":
                    response = self._call_openai(
                        request.messages, model, temp, request.max_tokens
                    )
                else:
                    response = self._call_anthropic(
                        request.messages, model, temp, request.max_tokens
                    )

                response.request_id = request.request_id
                logger.info(
                    f"[{request.request_id}] 성공: {provider}/{model} "
                    f"({response.latency_ms:.0f}ms)"
                )
                return response

            except Exception as e:
                last_error = e
                logger.warning(
                    f"[{request.request_id}] 실패 {provider}/{model}: {e}"
                )
                continue

        raise RuntimeError(
            f"요청 {request.request_id}에 대해 모든 프로바이더가 실패. "
            f"마지막 에러: {last_error}"
        )
```

### 지수 백오프를 포함한 재시도

```python
from tenacity import (
    retry, stop_after_attempt, wait_exponential,
    retry_if_exception_type, before_sleep_log,
)

class RetryableLLMClient:
    """설정 가능한 재시도 동작이 있는 LLM 클라이언트."""

    def __init__(self):
        self.client = OpenAI()

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=30),
        retry=retry_if_exception_type((
            ConnectionError,
            TimeoutError,
        )),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    def call(self, messages: list[dict], model: str = "gpt-4o",
             temperature: float = 0.3) -> str:
        """일시적 장애 시 자동 재시도가 있는 LLM 호출."""
        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            timeout=30.0,
        )
        return response.choices[0].message.content

    def call_with_budget_check(self, request: LLMRequest,
                               cost_tracker: CostTracker) -> LLMResponse | None:
        """예산 내에 있을 때만 진행."""
        if not cost_tracker.can_proceed():
            logger.error("예산 초과. 요청 차단됨.")
            return None
        content = self.call(
            request.messages, request.model, request.temperature
        )
        return LLMResponse(
            request_id=request.request_id,
            content=content,
            model=request.model,
            tokens_input=0,
            tokens_output=0,
            latency_ms=0,
        )
```

---

## 5. A/B 테스트 LLM 응답

### A/B 테스트 프레임워크

```python
import random
import hashlib
from dataclasses import dataclass

@dataclass
class Variant:
    name: str
    model: str
    temperature: float
    system_prompt: str
    weight: float = 0.5  # 트래픽 할당

class ABTestManager:
    """서로 다른 LLM 설정을 A/B 테스트."""

    def __init__(self):
        self.experiments: dict[str, list[Variant]] = {}
        self.results: dict[str, list[dict]] = defaultdict(list)

    def create_experiment(self, name: str, variants: list[Variant]):
        """새로운 A/B 테스트 실험 생성."""
        total_weight = sum(v.weight for v in variants)
        if abs(total_weight - 1.0) > 0.01:
            raise ValueError(f"Variant 가중치의 합이 1.0이어야 하지만, {total_weight}임")
        self.experiments[name] = variants

    def assign_variant(self, experiment: str, user_id: str) -> Variant:
        """결정적으로 사용자를 variant에 할당 (스티키 할당)."""
        variants = self.experiments[experiment]
        # user_id를 해시하여 결정적 할당
        hash_val = int(hashlib.md5(
            f"{experiment}:{user_id}".encode()
        ).hexdigest(), 16)
        threshold = hash_val % 1000 / 1000.0

        cumulative = 0.0
        for variant in variants:
            cumulative += variant.weight
            if threshold < cumulative:
                return variant

        return variants[-1]  # 폴백

    def record_result(self, experiment: str, variant_name: str,
                      metrics: dict):
        """실험 결과 기록."""
        self.results[experiment].append({
            "variant": variant_name,
            "timestamp": time.time(),
            **metrics,
        })

    def get_summary(self, experiment: str) -> dict:
        """실험 결과 요약."""
        results = self.results[experiment]
        by_variant = defaultdict(list)
        for r in results:
            by_variant[r["variant"]].append(r)

        summary = {}
        for variant_name, variant_results in by_variant.items():
            latencies = [r.get("latency_ms", 0) for r in variant_results]
            ratings = [r.get("user_rating", 0) for r in variant_results
                       if "user_rating" in r]
            costs = [r.get("cost", 0) for r in variant_results]

            summary[variant_name] = {
                "total_requests": len(variant_results),
                "avg_latency_ms": sum(latencies) / len(latencies) if latencies else 0,
                "avg_rating": sum(ratings) / len(ratings) if ratings else 0,
                "total_cost": sum(costs),
                "avg_cost": sum(costs) / len(costs) if costs else 0,
            }
        return summary

# 사용 예시
ab = ABTestManager()
ab.create_experiment("prompt-style-v2", [
    Variant("concise", "gpt-4o-mini", 0.2,
            "Be concise and direct. Answer in 2-3 sentences.", weight=0.5),
    Variant("detailed", "gpt-4o", 0.3,
            "Provide thorough, detailed answers with examples.", weight=0.5),
])

# 요청별 라우팅
user_id = "user_12345"
variant = ab.assign_variant("prompt-style-v2", user_id)
print(f"사용자 {user_id}가 variant에 할당됨: {variant.name}")
```

---

## 6. 관측 가능성

### LangSmith 통합

```python
import os
from langsmith import Client
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# LangSmith 설정
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "production-llm-app"
# os.environ["LANGCHAIN_API_KEY"] = "your-key"

langsmith = Client()

# 모든 LangChain 호출이 자동으로 추적됨
llm = ChatOpenAI(model="gpt-4o", temperature=0.3)
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("user", "{question}"),
])

chain = prompt | llm

# 이 호출이 자동으로 LangSmith에 기록됨
result = chain.invoke({"question": "What is retrieval-augmented generation?"})

# 수동 추적 어노테이션
from langsmith import traceable

@traceable(name="custom-rag-pipeline", run_type="chain")
def rag_pipeline(query: str) -> str:
    """LangSmith 추적이 있는 커스텀 RAG 파이프라인."""
    # 각 하위 단계가 자동으로 추적됨
    docs = retrieve_documents(query)
    context = format_context(docs)
    answer = generate_answer(query, context)
    return answer

@traceable(run_type="retriever")
def retrieve_documents(query: str) -> list[str]:
    return ["doc1 content", "doc2 content"]

@traceable(run_type="chain")
def format_context(docs: list[str]) -> str:
    return "\n\n".join(docs)

@traceable(run_type="llm")
def generate_answer(query: str, context: str) -> str:
    response = llm.invoke(f"Context: {context}\n\nQuestion: {query}")
    return response.content
```

### Phoenix (Arize) 통합

```python
import phoenix as px
from phoenix.otel import register
from openinference.instrumentation.openai import OpenAIInstrumentor

# 로컬 관측 가능성을 위해 Phoenix 시작
session = px.launch_app()
print(f"Phoenix UI: {session.url}")

# OpenAI 호출 자동 계측
tracer_provider = register(project_name="llm-production")
OpenAIInstrumentor().instrument(tracer_provider=tracer_provider)

# 모든 OpenAI 호출이 이제 Phoenix 대시보드에 표시됨
client = OpenAI()
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Explain vector databases"}],
)

# Phoenix 제공 기능:
# - 추적 시각화 (스팬, 지연, 토큰)
# - 평가 메트릭
# - 임베딩 시각화
# - 데이터셋 관리
```

### 커스텀 관측 가능성

```python
import json
import time
from collections import defaultdict
from datetime import datetime

class LLMObserver:
    """LLM 애플리케이션을 위한 경량 관측 가능성."""

    def __init__(self):
        self.traces: list[dict] = []
        self.metrics = defaultdict(list)

    def trace(self, func):
        """LLM 호출을 추적하는 데코레이터."""
        from functools import wraps

        @wraps(func)
        def wrapper(*args, **kwargs):
            trace_id = str(uuid.uuid4())
            start = time.time()

            trace_entry = {
                "trace_id": trace_id,
                "function": func.__name__,
                "start_time": datetime.now().isoformat(),
                "args_preview": str(args)[:200],
            }

            try:
                result = func(*args, **kwargs)
                duration = (time.time() - start) * 1000

                trace_entry.update({
                    "status": "success",
                    "duration_ms": round(duration, 2),
                    "result_preview": str(result)[:200],
                })
                self.metrics["latency_ms"].append(duration)
                self.metrics["success"].append(1)
                return result

            except Exception as e:
                duration = (time.time() - start) * 1000
                trace_entry.update({
                    "status": "error",
                    "duration_ms": round(duration, 2),
                    "error": str(e),
                })
                self.metrics["errors"].append(str(e))
                self.metrics["success"].append(0)
                raise

            finally:
                self.traces.append(trace_entry)

        return wrapper

    def dashboard(self) -> dict:
        """관측 가능성 대시보드 데이터 반환."""
        latencies = self.metrics.get("latency_ms", [])
        successes = self.metrics.get("success", [])

        return {
            "total_requests": len(successes),
            "success_rate": sum(successes) / len(successes) if successes else 0,
            "avg_latency_ms": sum(latencies) / len(latencies) if latencies else 0,
            "p50_latency_ms": sorted(latencies)[len(latencies) // 2] if latencies else 0,
            "p99_latency_ms": (
                sorted(latencies)[int(len(latencies) * 0.99)]
                if latencies else 0
            ),
            "total_errors": len(self.metrics.get("errors", [])),
            "recent_errors": self.metrics.get("errors", [])[-5:],
        }

# 사용 예시
observer = LLMObserver()

@observer.trace
def llm_call(query: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": query}],
    )
    return response.choices[0].message.content

# 호출 실행
llm_call("What is Python?")
llm_call("Explain Docker")

# 대시보드 확인
print(json.dumps(observer.dashboard(), indent=2))
```

---

## 7. 속도 제한 및 멀티 모델 라우팅

### 토큰 버킷 속도 제한기

```python
import threading
import time

class TokenBucketRateLimiter:
    """토큰 버킷 알고리즘을 사용한 속도 제한기."""

    def __init__(self, requests_per_minute: int = 60,
                 tokens_per_minute: int = 100_000):
        self.rpm_limit = requests_per_minute
        self.tpm_limit = tokens_per_minute
        self.request_tokens = requests_per_minute
        self.token_tokens = tokens_per_minute
        self.last_refill = time.time()
        self._lock = threading.Lock()

    def _refill(self):
        """경과 시간에 따라 토큰을 보충."""
        now = time.time()
        elapsed = now - self.last_refill
        self.request_tokens = min(
            self.rpm_limit,
            self.request_tokens + elapsed * (self.rpm_limit / 60),
        )
        self.token_tokens = min(
            self.tpm_limit,
            self.token_tokens + elapsed * (self.tpm_limit / 60),
        )
        self.last_refill = now

    def acquire(self, estimated_tokens: int = 1000) -> bool:
        """요청에 대한 용량을 획득 시도."""
        with self._lock:
            self._refill()
            if self.request_tokens >= 1 and self.token_tokens >= estimated_tokens:
                self.request_tokens -= 1
                self.token_tokens -= estimated_tokens
                return True
            return False

    def wait_and_acquire(self, estimated_tokens: int = 1000,
                         timeout: float = 60.0) -> bool:
        """용량이 확보될 때까지 대기."""
        start = time.time()
        while time.time() - start < timeout:
            if self.acquire(estimated_tokens):
                return True
            time.sleep(0.1)
        return False
```

### 지능형 멀티 모델 라우터

```python
class ModelRouter:
    """작업 특성에 따라 최적 모델로 요청을 라우팅."""

    MODEL_PROFILES = {
        "gpt-4o": {
            "provider": "openai",
            "capabilities": ["reasoning", "coding", "creative", "analysis"],
            "speed": "medium",
            "cost": "high",
            "context_window": 128_000,
        },
        "gpt-4o-mini": {
            "provider": "openai",
            "capabilities": ["classification", "extraction", "simple_qa"],
            "speed": "fast",
            "cost": "low",
            "context_window": 128_000,
        },
        "claude-sonnet-4-20250514": {
            "provider": "anthropic",
            "capabilities": ["reasoning", "coding", "analysis", "long_context"],
            "speed": "medium",
            "cost": "high",
            "context_window": 200_000,
        },
        "claude-haiku-4-20250514": {
            "provider": "anthropic",
            "capabilities": ["classification", "extraction", "simple_qa"],
            "speed": "fast",
            "cost": "low",
            "context_window": 200_000,
        },
    }

    def __init__(self):
        self.rate_limiters = {
            "openai": TokenBucketRateLimiter(rpm=500, tpm=800_000),
            "anthropic": TokenBucketRateLimiter(rpm=400, tpm=400_000),
        }

    def classify_task(self, messages: list[dict]) -> dict:
        """라우팅을 위한 작업 분류."""
        last_msg = ""
        for m in reversed(messages):
            if m["role"] == "user":
                last_msg = m["content"]
                break

        total_tokens = sum(len(m.get("content", "").split()) * 1.3
                          for m in messages)

        # 간단한 휴리스틱 분류
        task_type = "simple_qa"
        if any(kw in last_msg.lower() for kw in ["code", "implement", "debug", "function"]):
            task_type = "coding"
        elif any(kw in last_msg.lower() for kw in ["analyze", "compare", "evaluate"]):
            task_type = "analysis"
        elif any(kw in last_msg.lower() for kw in ["write", "create", "draft", "story"]):
            task_type = "creative"
        elif any(kw in last_msg.lower() for kw in ["classify", "categorize", "label"]):
            task_type = "classification"
        elif any(kw in last_msg.lower() for kw in ["extract", "parse", "find"]):
            task_type = "extraction"

        return {
            "task_type": task_type,
            "estimated_tokens": int(total_tokens),
            "needs_long_context": total_tokens > 50_000,
        }

    def select_model(self, messages: list[dict],
                     prefer_fast: bool = False,
                     prefer_cheap: bool = False) -> str:
        """요청에 대한 최적 모델 선택."""
        task = self.classify_task(messages)

        candidates = []
        for model, profile in self.MODEL_PROFILES.items():
            # 기능 매칭 검사
            capability_match = task["task_type"] in profile["capabilities"]

            # 컨텍스트 윈도우 검사
            fits_context = task["estimated_tokens"] < profile["context_window"]

            # 속도 제한 가용성 검사
            provider = profile["provider"]
            available = self.rate_limiters[provider].acquire(
                task["estimated_tokens"]
            ) if provider in self.rate_limiters else True

            if fits_context:
                score = 0
                if capability_match:
                    score += 10
                if prefer_fast and profile["speed"] == "fast":
                    score += 5
                if prefer_cheap and profile["cost"] == "low":
                    score += 5
                if available:
                    score += 3

                candidates.append((model, score, available))

        # 점수 내림차순 정렬
        candidates.sort(key=lambda x: (-x[1], -int(x[2])))

        if candidates:
            return candidates[0][0]

        return "gpt-4o-mini"  # 최종 폴백

# 사용 예시
router = ModelRouter()
model = router.select_model(
    messages=[{"role": "user", "content": "Classify this ticket as bug or feature: 'Login page crashes'"}],
    prefer_cheap=True,
)
print(f"선택된 모델: {model}")  # gpt-4o-mini 또는 claude-haiku 가능성 높음
```

---

## 배포 체크리스트

| 카테고리 | 항목 | 우선순위 |
|----------|------|----------|
| **신뢰성** | 멀티 프로바이더 폴백 설정 | 필수 |
| **신뢰성** | 지수 백오프 재시도 | 필수 |
| **신뢰성** | 요청 타임아웃 설정 (30s) | 필수 |
| **신뢰성** | 프로바이더 장애를 위한 서킷 브레이커 | 높음 |
| **성능** | 정확 매칭 캐시 활성화 | 높음 |
| **성능** | 반복 쿼리를 위한 시맨틱 캐시 | 중간 |
| **성능** | 사용자 대면 API에 스트리밍 응답 | 높음 |
| **성능** | 도구 호출 및 검색에 비동기 I/O | 높음 |
| **비용** | 요청별 비용 추적 | 필수 |
| **비용** | 일별/월별 예산 적용 | 필수 |
| **비용** | 모델 라우팅 (간단한 작업에 저렴한 모델) | 높음 |
| **비용** | 프롬프트 토큰 최적화 | 중간 |
| **보안** | 입력 정화 (프롬프트 인젝션 방어) | 필수 |
| **보안** | 출력 필터링 (PII, 유해 콘텐츠) | 필수 |
| **보안** | 사용자/API 키별 속도 제한 | 필수 |
| **보안** | 비밀 관리 (하드코딩된 API 키 없음) | 필수 |
| **관측 가능성** | 요청/응답 로깅 | 필수 |
| **관측 가능성** | 지연, 토큰, 비용 메트릭 | 높음 |
| **관측 가능성** | 에러 알림 | 높음 |
| **관측 가능성** | 추적 시각화 (LangSmith/Phoenix) | 중간 |
| **테스트** | 도구 구현 단위 테스트 | 높음 |
| **테스트** | 모의 LLM을 사용한 통합 테스트 | 높음 |
| **테스트** | 프롬프트 변경을 위한 A/B 테스트 프레임워크 | 중간 |
| **테스트** | 출시 전 레드 팀 평가 | 높음 |

---

## 다음 단계

이 레슨은 NLP 및 LLM 과정의 프로덕션 배포 섹션을 마무리한다. 추가 학습을 위해 [12_Advanced_RAG.md](./12_Advanced_RAG.md)를 다시 방문하여 이러한 프로덕션 패턴을 RAG 시스템에 적용하거나, [15_Multi_Agent_Systems.md](./15_Multi_Agent_Systems.md)에서 대규모 멀티 에이전트 워크플로우 배포를 탐구할 수 있다.
