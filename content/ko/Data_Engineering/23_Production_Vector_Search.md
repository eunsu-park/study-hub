[← 이전: 22. 벡터 저장소와 인덱싱](22_Vector_Storage_and_Indexing.md) | [다음: 개요 →](00_Overview.md)

# 23. 프로덕션 벡터 검색

## 학습 목표

1. 밀집 벡터 검색과 희소 키워드 매칭을 결합한 하이브리드 검색 시스템을 설계한다
2. 메타데이터 필터링 전략을 구현하고 사전 필터링 vs 사후 필터링 트레이드오프를 이해한다
3. 크로스 인코더 모델을 사용한 리랭킹 파이프라인을 구축하여 정밀도를 향상시킨다
4. 샤딩, 복제, 부하 분산을 통해 벡터 검색을 수평 확장한다
5. 적절한 메트릭과 알림 임계값으로 벡터 검색 시스템을 모니터링한다
6. 차원 축소, 양자화, 계층형 저장소를 통해 비용을 최적화한다
7. 배치 임베딩 업데이트로 벡터 검색을 프로덕션 데이터 파이프라인에 통합한다

---

## 개요

프로덕션에서 벡터 검색을 배포하는 것은 임베딩을 인덱싱하고 쿼리를 실행하는 것 그 이상입니다. 프로덕션 시스템은 하이브리드 검색(의미 검색과 키워드 검색의 결합), 비즈니스 메타데이터에 의한 결과 필터링, 정밀도를 위한 리랭킹, 트래픽 급증 처리를 위한 확장을 수행해야 하며, 이 모든 것을 허용 가능한 비용으로 안정적으로 처리해야 합니다.

이 레슨은 벡터 검색의 운영 측면 — 프로토타입을 프로덕션 서비스로 전환하기 위해 데이터 엔지니어가 알아야 할 패턴과 실천법을 다룹니다. 하이브리드 검색 융합, 필터링 전략, 리랭킹 파이프라인, 확장 아키텍처, 모니터링, 비용 최적화, 그리고 더 넓은 데이터 파이프라인과의 통합을 다룹니다.

> **데이터 엔지니어에게 중요한 이유**: 벡터 검색은 점점 더 프로덕션 데이터 제품에 내장되고 있습니다 — 추천 엔진, 고객 지원, 지식 베이스, 사기 탐지. 데이터 엔지니어는 벡터 저장소에 임베딩을 공급하는 파이프라인을 소유하고, 건강을 모니터링하며, 소스 데이터가 변경될 때 인덱스를 최신 상태로 유지합니다.

---

## 1. 하이브리드 검색

### 1.1 하이브리드 검색이 필요한 이유

```
순수 벡터 검색 vs 순수 키워드 검색:

  쿼리: "오류 코드 E-4021 문제 해결"

  벡터 검색 결과:                    키워드 검색 결과:
  1. 일반 문제 해결 가이드            1. E-4021을 구체적으로 언급하는 문서
  2. 일반적인 오류 패턴 문서          2. E-4021 릴리스 노트
  3. 시스템 진단 개요                 3. E-4021 패치 지침

  벡터 검색이 정확한 오류 코드를 놓쳤습니다!

  쿼리: "빠른 근사 최근접 이웃 알고리즘"

  벡터 검색 결과:                    키워드 검색 결과:
  1. HNSW 알고리즘 심층 분석          1. (결과 없음 — 정확한 매칭 없음)
  2. FAISS 성능 가이드               2.
  3. ANN 벤치마크                    3.

  키워드 검색이 의미적 매칭을 놓쳤습니다!

하이브리드 검색은 양쪽을 결합 → 양쪽의 장점을 모두 활용
```

### 1.2 밀집 + 희소 검색 아키텍처

```
하이브리드 검색 파이프라인:

  쿼리: "E-4021 문제 해결 단계"
       │
       ├──────────────────────┐
       │                      │
       ▼                      ▼
  ┌──────────┐         ┌──────────┐
  │ 임베딩    │         │ 토크나이저│
  │ 모델     │         │ (BM25)   │
  └────┬─────┘         └────┬─────┘
       │                     │
       ▼                     ▼
  ┌──────────┐         ┌──────────┐
  │ 벡터     │         │ 역인덱스  │
  │ 인덱스   │         │ (BM25)   │
  │ (HNSW)   │         │          │
  └────┬─────┘         └────┬─────┘
       │                     │
       │  Top-K 밀집         │  Top-K 희소
       │  결과               │  결과
       ▼                     ▼
  ┌──────────────────────────────┐
  │      융합 알고리즘             │
  │  (RRF, 선형 결합,            │
  │   또는 학습된 융합)            │
  └──────────────┬───────────────┘
                 │
                 ▼
          병합된 top-K 결과
```

### 1.3 역순위 융합 (RRF)

```python
"""
역순위 융합(RRF)은 점수 정규화 없이
서로 다른 검색 시스템의 순위 목록을 결합합니다.
"""

def reciprocal_rank_fusion(
    ranked_lists: list[list[str]],
    k: int = 60,
) -> list[tuple[str, float]]:
    """
    RRF를 사용하여 여러 순위 목록을 융합합니다.

    Args:
        ranked_lists: 순위화된 문서 ID 목록들의 리스트
        k: RRF 상수 (높을수록 상위 순위에 가중치 감소)

    Returns:
        점수 내림차순으로 정렬된 (doc_id, rrf_score) 리스트
    """
    scores: dict[str, float] = {}

    for ranked_list in ranked_lists:
        for rank, doc_id in enumerate(ranked_list, start=1):
            if doc_id not in scores:
                scores[doc_id] = 0.0
            scores[doc_id] += 1.0 / (k + rank)

    sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_results

# 예시: 밀집과 희소 결과 융합
dense_results = ["doc_A", "doc_B", "doc_C", "doc_D", "doc_E"]
sparse_results = ["doc_C", "doc_F", "doc_A", "doc_G", "doc_B"]

fused = reciprocal_rank_fusion([dense_results, sparse_results], k=60)
# doc_A: 1/(60+1) + 1/(60+3) = 0.01639 + 0.01587 = 0.03226 (순위 1)
# doc_C: 1/(60+3) + 1/(60+1) = 0.01587 + 0.01639 = 0.03226 (순위 1 동점)
# doc_B: 1/(60+2) + 1/(60+5) = 0.01613 + 0.01538 = 0.03151 (순위 3)
```

### 1.4 가중 선형 결합

```python
"""
선형 결합은 밀집과 희소 점수가 서로 다른 범위에 있으므로
점수 정규화가 필요합니다.
"""

import numpy as np

def normalize_scores(scores: list[float]) -> list[float]:
    """점수를 [0, 1]로 최소-최대 정규화합니다."""
    arr = np.array(scores)
    if arr.max() == arr.min():
        return [1.0] * len(scores)
    return ((arr - arr.min()) / (arr.max() - arr.min())).tolist()

def linear_combination(
    dense_results: list[tuple[str, float]],
    sparse_results: list[tuple[str, float]],
    alpha: float = 0.7,  # 밀집(의미적)에 대한 가중치
    top_k: int = 10,
) -> list[tuple[str, float]]:
    """
    가중 점수로 밀집과 희소 결과를 결합합니다.
    alpha=1.0 → 순수 밀집, alpha=0.0 → 순수 희소
    """
    # 점수 정규화
    dense_ids = [r[0] for r in dense_results]
    dense_scores = normalize_scores([r[1] for r in dense_results])
    sparse_ids = [r[0] for r in sparse_results]
    sparse_scores = normalize_scores([r[1] for r in sparse_results])

    # 병합
    combined: dict[str, float] = {}
    for doc_id, score in zip(dense_ids, dense_scores):
        combined[doc_id] = alpha * score
    for doc_id, score in zip(sparse_ids, sparse_scores):
        combined[doc_id] = combined.get(doc_id, 0.0) + (1 - alpha) * score

    sorted_results = sorted(combined.items(), key=lambda x: x[1], reverse=True)
    return sorted_results[:top_k]

# alpha 튜닝:
#   alpha=0.7 → 대부분의 사용 사례에 좋은 기본값
#   alpha=0.9 → 쿼리가 주로 의미적/대화형일 때
#   alpha=0.3 → 쿼리에 특정 코드, ID, 정확한 용어가 포함될 때
```

### 1.5 SPLADE: 학습된 희소 표현

```
SPLADE (Sparse Lexical and Expansion) — 트랜스포머 학습 희소 벡터:

  전통적 BM25:           SPLADE:
  "machine learning"  →   "machine learning"  →
  {"machine": 1.2,        {"machine": 0.8,
   "learning": 1.5}        "learning": 1.1,
                            "AI": 0.6,          ← 확장!
                            "neural": 0.3,      ← 확장!
                            "algorithm": 0.2}   ← 확장!

  SPLADE는 쿼리와 문서를 관련 용어로 확장하는 것을 학습하여
  BM25의 효율성과 의미적 인식을 결합합니다.

  파이프라인:
    텍스트 → BERT 인코더 → ReLU + log → 희소 벡터 (30K 차원, ~100 비영)
                                          │
                                          ▼
                                    역인덱스 (BM25와 동일)
```

---

## 2. 메타데이터 필터링

### 2.1 사전 필터링 vs 사후 필터링

```
사전 필터링:
  ① 메타데이터 필터 적용 → 후보 집합
  ② 필터링된 후보에만 벡터 검색 실행

  ✓ 필터에 일치하는 정확히 N개의 결과를 보장
  ✗ 필터가 매우 선택적이면 (<1% 데이터), HNSW 그래프가
    필터된 부분집합 내에 연결이 적을 수 있음
    → 재현율 저하

사후 필터링:
  ① 전체 인덱스에 벡터 검색 실행 → 대규모 후보 집합
  ② 후보에 메타데이터 필터 적용

  ✓ 전체 그래프에서 벡터 검색 수행 (좋은 재현율)
  ✗ 필터가 많이 제거하면 N개 미만의 결과 반환 가능
  ✗ 관련 없는 벡터 검색에 컴퓨트 낭비

하이브리드 (대부분의 현대 DB):
  ① 인덱스 힌트를 사용하여 필터된 영역으로 검색 안내
  ② 점수가 벡터 유사도와 필터 일치를 모두 고려
  ✗ 복잡한 구현

  Qdrant 접근법:
    페이로드 인덱스를 사용하여 필터 조건을 만족할 수 없는
    HNSW 순회 경로를 가지치기

  Milvus 접근법:
    파티션 가지치기 (관련 파티션만 검색)
    + 세그먼트 수준 블룸 필터
```

### 2.2 필터 설계 패턴

```python
"""
프로덕션 벡터 검색을 위한 메타데이터 필터링 패턴.
"""

# 패턴 1: 파티션 기반 필터링 (Milvus)
# 필터가 데이터를 크고 안정적인 그룹으로 나눌 때 최적
collection.search(
    data=[query_vector],
    anns_field="embedding",
    param={"metric_type": "L2", "params": {"nprobe": 64}},
    limit=10,
    partition_names=["electronics", "clothing"],  # 파티션 가지치기
)

# 패턴 2: 불리언 표현식 필터링 (Milvus)
# 스칼라 필드에 대한 복합 조건에 최적
collection.search(
    data=[query_vector],
    anns_field="embedding",
    param={"metric_type": "L2", "params": {"nprobe": 64}},
    limit=10,
    expr=(
        'category in ["electronics", "clothing"] '
        'and price >= 10.0 and price <= 100.0 '
        'and in_stock == true '
        'and brand != "Acme"'
    ),
)

# 패턴 3: 중첩 페이로드 필터링 (Qdrant)
# 계층적 메타데이터에 최적
from qdrant_client.models import Filter, FieldCondition, MatchValue, Range

filter_config = Filter(
    must=[
        FieldCondition(key="category", match=MatchValue(value="electronics")),
        FieldCondition(key="price", range=Range(gte=10.0, lte=100.0)),
        FieldCondition(key="supplier.country", match=MatchValue(value="US")),
    ],
    must_not=[
        FieldCondition(key="status", match=MatchValue(value="discontinued")),
    ],
)

# 패턴 4: 태그 기반 필터링 (Pinecone)
# 멀티 레이블 분류에 최적
results = index.query(
    vector=query_vector,
    top_k=10,
    filter={
        "tags": {"$in": ["sale", "featured"]},
        "rating": {"$gte": 4.0},
    },
)
```

### 2.3 필터링 성능 팁

```
메타데이터 필터 최적화:

1. 필터 필드를 인덱싱하세요
   - Qdrant: payload_index (필터된 필드에 자동 생성)
   - Milvus: 스칼라 인덱스 (명시적 생성 권장)
   - Weaviate: 스키마 속성에 indexFilterable=True

2. 높은 카디널리티 범주형 필터에는 파티션을 사용하세요
   - Milvus: 테넌트, 지역, 카테고리별 파티션 생성
   - 사후 필터링 오버헤드 없이 검색 공간 축소

3. 벡터 전용 인덱스에서 높은 선택도 필터를 피하세요
   - 필터가 데이터의 <1%를 선택하면 사전 필터링이 HNSW 재현율 저하
   - 해결책: 오버 페치 (10배 후보 검색) 후 사후 필터

4. 메타데이터를 벡터 저장소에 비정규화하세요
   - 벡터 DB와 관계형 DB 간의 조인 쿼리 방지
   - 저장소를 쿼리 단순성과 교환
   - 소스 데이터 변경 시 업서트로 메타데이터 업데이트

5. 존재 여부 검사에 블룸 필터를 사용하세요
   - "이 문서가 인덱스에 있는가?"
   - 배치 중복 제거를 위한 포인트 룩업보다 빠름
```

---

## 3. 리랭킹 파이프라인

### 3.1 2단계 검색

```
왜 리랭킹이 필요한가?

  1단계 (검색):
    - 빠르지만 근사적
    - 바이 인코더 사용 (쿼리와 문서를 독립적으로 인코딩)
    - ~5ms에 top-100 후보 검색

  2단계 (리랭킹):
    - 느리지만 정밀
    - 크로스 인코더 사용 (쿼리와 문서를 함께 인코딩)
    - ~50ms에 top-100을 top-10으로 리랭킹

  바이 인코더 (1단계):           크로스 인코더 (2단계):
  ┌─────┐    ┌─────┐             ┌──────────────────┐
  │쿼리 │    │문서  │             │ [CLS] 쿼리 [SEP] │
  └──┬──┘    └──┬──┘             │      문서 [SEP]   │
     │          │                └────────┬─────────┘
     ▼          ▼                         │
  인코더     인코더                     인코더
     │          │                         │
     ▼          ▼                         ▼
  q_vec      d_vec                    점수 (0-1)
     │          │
  cosine(q, d) = 0.82

  크로스 인코더는 ~100배 느리지만 ~10-20% 더 정확
  쿼리-문서 상호작용에 어텐션을 줄 수 있기 때문
```

### 3.2 크로스 인코더 리랭킹 구현

```python
"""
sentence-transformers를 사용한 크로스 인코더 리랭킹.
"""

from sentence_transformers import CrossEncoder
import numpy as np

# 크로스 인코더 모델 로드
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2")

def rerank_results(
    query: str,
    documents: list[dict],
    top_k: int = 10,
) -> list[dict]:
    """
    크로스 인코더를 사용하여 검색된 문서를 리랭킹합니다.

    Args:
        query: 사용자 쿼리 문자열
        documents: {"id": ..., "text": ..., "score": ...} 리스트
        top_k: 리랭킹 후 반환할 결과 수

    Returns:
        업데이트된 점수로 리랭킹된 문서 리스트
    """
    if not documents:
        return []

    # 크로스 인코더를 위한 쌍 준비
    pairs = [(query, doc["text"]) for doc in documents]

    # 모든 쌍에 점수 매기기
    scores = reranker.predict(pairs, show_progress_bar=False)

    # 점수 첨부 및 정렬
    for doc, score in zip(documents, scores):
        doc["rerank_score"] = float(score)
        doc["original_score"] = doc.get("score", 0.0)

    reranked = sorted(documents, key=lambda x: x["rerank_score"], reverse=True)
    return reranked[:top_k]


# 검색 파이프라인에서의 사용
def search_and_rerank(query: str, collection, top_k: int = 10):
    """전체 검색 파이프라인: 검색 → 리랭킹."""
    # 1단계: top-100 후보 검색 (빠름, 근사적)
    candidates = collection.search(
        query_embedding=encode_query(query),
        limit=100,
    )

    # 2단계: 크로스 인코더로 리랭킹 (느림, 정밀)
    reranked = rerank_results(query, candidates, top_k=top_k)

    return reranked
```

### 3.3 Cohere Rerank API (관리형)

```python
"""
자체 모델 인프라 없이 프로덕션 리랭킹을 위해
Cohere의 rerank API를 사용합니다.
"""

import cohere

co = cohere.Client("your-api-key")

def rerank_with_cohere(
    query: str,
    documents: list[str],
    top_k: int = 10,
    model: str = "rerank-english-v3.0",
) -> list[dict]:
    """Cohere의 호스팅 크로스 인코더를 사용한 리랭킹."""
    response = co.rerank(
        query=query,
        documents=documents,
        top_n=top_k,
        model=model,
    )

    results = []
    for result in response.results:
        results.append({
            "index": result.index,
            "text": documents[result.index],
            "relevance_score": result.relevance_score,
        })
    return results

# 비용: 검색 1000회당 ~$1 (각 100개 문서 리랭킹)
# 지연시간: 100개 문서에 ~100-200ms
# 정확도: 영어 텍스트에 최신 기술 수준
```

### 3.4 다단계 파이프라인 아키텍처

```
프로덕션 검색 파이프라인:

  쿼리
    │
    ▼
  ┌────────────────────┐
  │ 쿼리 이해           │   의도 분류, 쿼리 확장
  └─────────┬──────────┘
            │
    ┌───────┴───────┐
    │               │
    ▼               ▼
  밀집             희소
  검색             검색              1단계: 검색 (100 후보)
  (바이 인코더)     (BM25/SPLADE)
    │               │
    └───────┬───────┘
            │
            ▼
  ┌────────────────────┐
  │   융합 (RRF)        │            ~100개 고유 후보로 병합
  └─────────┬──────────┘
            │
            ▼
  ┌────────────────────┐
  │  메타데이터 필터     │            비즈니스 규칙 적용 (ACL, 최신성)
  └─────────┬──────────┘
            │
            ▼
  ┌────────────────────┐
  │  크로스 인코더       │            2단계: 리랭킹 (100 → 20)
  │  리랭킹             │
  └─────────┬──────────┘
            │
            ▼
  ┌────────────────────┐
  │  비즈니스 로직       │            중복 제거, 부스트, 다양성, 개인화
  └─────────┬──────────┘
            │
            ▼
       Top-10 결과

  총 지연시간 예산:
    검색: ~5ms
    융합: ~1ms
    필터: ~2ms
    리랭킹: ~50ms
    비즈니스 로직: ~2ms
    ─────────────────
    합계: ~60ms p50
```

---

## 4. 벡터 검색 확장

### 4.1 샤딩 전략

```
샤딩은 벡터를 여러 노드에 분산합니다:

  전략 1: 해시 기반 샤딩
  ┌──────────┐
  │ 쿼리     │─── hash(query) % N ──→ 샤드 K
  └──────────┘                        (하나의 샤드만 검색)
  ✗ 모든 데이터를 검색할 수 없음 (쿼리당 하나의 샤드만)
  ✗ 파티션된 데이터에만 유용 (예: 테넌트별)

  전략 2: 스캐터-개더 (가장 일반적)
  ┌──────────┐
  │ 쿼리     │──→ 모든 샤드에 병렬로
  └──────────┘     │    │    │
                   ▼    ▼    ▼
               샤드1  샤드2  샤드3  (각각 top-K 반환)
                   │    │    │
                   └────┼────┘
                        ▼
                   모든 샤드의 top-K 병합
                   (전역 top-K 반환)

  ✓ 모든 데이터 검색
  ✗ 지연시간 = 가장 느린 샤드 + 병합 시간
  ✗ 샤드가 많을수록 네트워크 오버헤드 증가

  전략 3: 학습된 라우팅
  ┌──────────┐
  │ 쿼리     │──→ 라우터 모델이 상위 2개 샤드 예측
  └──────────┘     │    │
                   ▼    ▼
               샤드2  샤드5  (관련 샤드만 검색)
  ✓ 팬아웃 감소
  ✗ 라우팅 모델 학습 필요
  ✗ 라우팅이 잘못되면 결과 누락 위험
```

### 4.2 고가용성을 위한 복제

```
복제 패턴:

  단일 복제본 (HA 없음):
  ┌──────────┐
  │ 샤드 1   │  ← 단일 장애점
  └──────────┘

  읽기 복제본 (읽기 HA):
  ┌──────────┐    ┌──────────┐    ┌──────────┐
  │ 샤드 1   │    │ 복제본   │    │ 복제본   │
  │ (프라이머리)│───→│ 1a      │───→│ 1b      │
  │ (쓰기)    │    │ (읽기)   │    │ (읽기)   │
  └──────────┘    └──────────┘    └──────────┘

  로드 밸런서가 복제본 간에 읽기를 분산
  프라이머리가 쓰기 처리 → 복제본에 비동기 복제

  Milvus 복제:
    - 복제 그룹: 각 샤드에 N개의 복제본
    - 일관성: 강력 (동기) 또는 최종적 (비동기)
    - 장애 조치: 복제본을 프라이머리로 자동 승격

  Qdrant 복제:
    - 쓰기 일관성을 위한 Raft 합의
    - 설정 가능한 write_consistency_factor (1 = 빠름, N = 안전)
    - 노드 추가 시 자동 샤드 재균형
```

### 4.3 용량 계획

```python
"""
벡터 검색 배포를 위한 용량 계획 계산기.
"""

def estimate_resources(
    n_vectors: int,
    dim: int,
    index_type: str = "HNSW",
    quantization: str = "none",
    qps_target: int = 1000,
    replication_factor: int = 2,
) -> dict:
    """벡터 검색 배포를 위한 컴퓨트 리소스를 추정합니다."""

    # 벡터 저장소
    bytes_per_element = {
        "none": 4,       # float32
        "float16": 2,
        "int8": 1,
        "pq96": 96 / dim,  # 96 서브양자화기의 PQ
    }
    bpe = bytes_per_element.get(quantization, 4)
    vector_memory_gb = (n_vectors * dim * bpe) / (1024**3)

    # 인덱스 오버헤드
    index_overhead = {
        "HNSW": n_vectors * 32 * 2 * 8 / (1024**3),  # M=32
        "IVF": 0.1 * vector_memory_gb,                 # ~10% 오버헤드
        "Flat": 0,
    }
    index_gb = index_overhead.get(index_type, 0)

    # 메타데이터 오버헤드 (벡터당 ~200 바이트 가정)
    metadata_gb = (n_vectors * 200) / (1024**3)

    # 샤드당 합계
    total_per_shard_gb = vector_memory_gb + index_gb + metadata_gb

    # QPS 추정 (대략: HNSW는 768d에서 코어당 ~3000 QPS)
    qps_per_core = {"HNSW": 3000, "IVF": 5000, "Flat": 50}
    cores_needed = qps_target / qps_per_core.get(index_type, 1000)

    # 샤딩: 단일 노드가 모든 데이터를 담을 수 없으면 분할
    max_memory_per_node_gb = 64  # 일반적인 인스턴스
    n_shards = max(1, int(total_per_shard_gb / (max_memory_per_node_gb * 0.7)) + 1)

    total_nodes = n_shards * replication_factor

    return {
        "vector_memory_gb": round(vector_memory_gb, 2),
        "index_overhead_gb": round(index_gb, 2),
        "metadata_gb": round(metadata_gb, 2),
        "total_per_shard_gb": round(total_per_shard_gb, 2),
        "n_shards": n_shards,
        "replication_factor": replication_factor,
        "total_nodes": total_nodes,
        "cores_per_node": max(4, int(cores_needed / n_shards) + 1),
        "ram_per_node_gb": min(max_memory_per_node_gb,
                               int(total_per_shard_gb / n_shards * 1.3) + 1),
    }

# 예시: 5000만 벡터, 768차원, HNSW, int8 양자화
plan = estimate_resources(
    n_vectors=50_000_000,
    dim=768,
    index_type="HNSW",
    quantization="int8",
    qps_target=5000,
    replication_factor=2,
)
```

---

## 5. 모니터링과 관측성

### 5.1 핵심 메트릭

```
벡터 검색 모니터링 메트릭:

  지연시간 메트릭:
  ┌─────────────────────────────────────────────────────────┐
  │ search_latency_p50_ms     목표: < 10ms                  │
  │ search_latency_p95_ms     목표: < 50ms                  │
  │ search_latency_p99_ms     목표: < 100ms                 │
  │ rerank_latency_p50_ms     목표: < 100ms                 │
  │ embedding_latency_p50_ms  목표: < 20ms                  │
  └─────────────────────────────────────────────────────────┘

  처리량 메트릭:
  ┌─────────────────────────────────────────────────────────┐
  │ search_qps                초당 쿼리 수                    │
  │ upsert_rate               초당 벡터 수집률                 │
  │ batch_embedding_rate      초당 임베딩 생성 수              │
  └─────────────────────────────────────────────────────────┘

  품질 메트릭:
  ┌─────────────────────────────────────────────────────────┐
  │ recall@10                 그라운드 트루스 대비 샘플 쿼리    │
  │ mrr@10                    평균 역순위                     │
  │ empty_result_rate         결과 0인 쿼리 비율              │
  │ filter_selectivity        필터링 후 평균 데이터 비율       │
  └─────────────────────────────────────────────────────────┘

  리소스 메트릭:
  ┌─────────────────────────────────────────────────────────┐
  │ memory_usage_pct          목표: < 80%                    │
  │ disk_usage_pct            목표: < 70%                    │
  │ cpu_usage_pct             목표: < 60% 지속               │
  │ index_size_vectors        인덱스 내 총 벡터 수             │
  │ segment_count             인덱스 세그먼트 수               │
  └─────────────────────────────────────────────────────────┘
```

### 5.2 Prometheus 메트릭 구현

```python
"""
Prometheus 메트릭으로 벡터 검색을 계측합니다.
"""

from prometheus_client import Histogram, Counter, Gauge, Summary
import time

# 메트릭 정의
SEARCH_LATENCY = Histogram(
    "vector_search_latency_seconds",
    "벡터 검색 지연시간",
    ["collection", "index_type"],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
)

SEARCH_RESULTS = Histogram(
    "vector_search_result_count",
    "반환된 결과 수",
    ["collection"],
    buckets=[0, 1, 5, 10, 20, 50, 100],
)

SEARCH_QPS = Counter(
    "vector_search_total",
    "총 벡터 검색 수",
    ["collection", "status"],
)

INDEX_SIZE = Gauge(
    "vector_index_size_total",
    "인덱스 내 총 벡터 수",
    ["collection"],
)

EMBEDDING_LATENCY = Histogram(
    "embedding_generation_latency_seconds",
    "임베딩 생성 지연시간",
    ["model"],
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5],
)

def instrumented_search(collection, query_vector, top_k=10, filters=None):
    """Prometheus 계측이 포함된 검색."""
    start = time.perf_counter()
    try:
        results = collection.search(
            query_embedding=query_vector,
            limit=top_k,
            filter=filters,
        )
        duration = time.perf_counter() - start

        SEARCH_LATENCY.labels(
            collection=collection.name,
            index_type="HNSW",
        ).observe(duration)
        SEARCH_RESULTS.labels(collection=collection.name).observe(len(results))
        SEARCH_QPS.labels(collection=collection.name, status="success").inc()

        return results
    except Exception as e:
        SEARCH_QPS.labels(collection=collection.name, status="error").inc()
        raise
```

### 5.3 재현율 모니터링

```python
"""
그라운드 트루스 쿼리를 사용한 지속적 재현율 모니터링.
주기적 작업으로 실행 (예: Airflow로 매시간).
"""

import numpy as np

def compute_recall_at_k(
    predicted: list[list[str]],
    ground_truth: list[list[str]],
    k: int = 10,
) -> float:
    """쿼리 평균 recall@k를 계산합니다."""
    recalls = []
    for pred, truth in zip(predicted, ground_truth):
        pred_set = set(pred[:k])
        truth_set = set(truth[:k])
        if len(truth_set) == 0:
            continue
        recall = len(pred_set & truth_set) / len(truth_set)
        recalls.append(recall)
    return np.mean(recalls) if recalls else 0.0

def recall_monitoring_job(
    collection,
    ground_truth_queries: list[dict],
    alert_threshold: float = 0.90,
):
    """
    그라운드 트루스 쿼리를 실행하여 재현율을 모니터링하고 하락 시 알림.

    ground_truth_queries: [{"vector": [...], "expected_ids": ["a", "b", ...]}]
    """
    predicted = []
    expected = []

    for gt in ground_truth_queries:
        results = collection.search(query_embedding=gt["vector"], limit=10)
        result_ids = [r["id"] for r in results]
        predicted.append(result_ids)
        expected.append(gt["expected_ids"])

    recall = compute_recall_at_k(predicted, expected, k=10)

    if recall < alert_threshold:
        # PagerDuty/Slack/이메일로 알림 전송
        send_alert(
            severity="warning",
            message=f"벡터 검색 recall@10이 {recall:.3f}으로 하락 "
                    f"(임계값: {alert_threshold})",
        )

    return recall
```

### 5.4 Grafana 대시보드 레이아웃

```
권장 Grafana 대시보드 패널:

  행 1: 개요
  ┌───────────────┬───────────────┬───────────────┬───────────────┐
  │ 검색 QPS      │ p50 지연시간   │ p99 지연시간   │ 오류율         │
  │ (카운터)       │ (게이지)       │ (게이지)       │ (퍼센트)       │
  └───────────────┴───────────────┴───────────────┴───────────────┘

  행 2: 품질
  ┌─────────────────────────────┬─────────────────────────────────┐
  │ 시간별 Recall@10             │ 빈 결과율                        │
  │ (시계열)                     │ (시계열)                         │
  └─────────────────────────────┴─────────────────────────────────┘

  행 3: 리소스
  ┌───────────────┬───────────────┬───────────────┬───────────────┐
  │ 메모리 사용률   │ CPU 사용률     │ 디스크 사용률   │ 인덱스 크기    │
  │ (노드별)       │ (노드별)       │ (노드별)       │ (벡터 수)     │
  └───────────────┴───────────────┴───────────────┴───────────────┘

  행 4: 파이프라인
  ┌─────────────────────────────┬─────────────────────────────────┐
  │ 임베딩 생성률                 │ 업서트 처리량                     │
  │ (시계열)                     │ (시계열)                         │
  └─────────────────────────────┴─────────────────────────────────┘
```

---

## 6. 비용 최적화

### 6.1 차원 축소

```python
"""
메모리 절약과 속도 향상을 위한 임베딩 차원 축소.
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.random_projection import GaussianRandomProjection

# 원본: 1536 차원 (OpenAI text-embedding-3-large)
original_dim = 1536
target_dim = 512
n_vectors = 1_000_000

# 방법 1: PCA (최고 품질, 학습 데이터 필요)
pca = PCA(n_components=target_dim)
training_data = np.random.randn(50_000, original_dim).astype('float32')
pca.fit(training_data)

reduced_vectors = pca.transform(original_vectors)
# 설명된 분산 비율: 512 차원에서 일반적으로 90-95%
# 메모리 절감: 3배 축소

# 방법 2: 마트료시카 임베딩 (모델이 지원하는 경우)
# OpenAI text-embedding-3-*는 잘라내기 지원
# 임베딩의 처음 N개 차원만 사용
# response = openai.embeddings.create(
#     model="text-embedding-3-small",
#     input="text",
#     dimensions=512,  # 1536에서 잘라내기
# )

# 방법 3: 랜덤 프로젝션 (빠름, 학습 불필요)
rp = GaussianRandomProjection(n_components=target_dim)
reduced_vectors = rp.fit_transform(original_vectors)
# PCA보다 낮은 품질이지만 O(1) 학습 시간

# 비용 비교 (1000만 벡터):
#   1536d float32: 57.2 GB RAM → ~$400/월 (클라우드)
#   512d float32:  19.1 GB RAM → ~$133/월 (클라우드)
#   512d int8:     4.8 GB RAM  → ~$33/월 (클라우드)
```

### 6.2 양자화 전략

```
양자화는 벡터당 바이트를 줄입니다:

  방법            바이트/차원   품질 손실         속도 영향
  ────────────────────────────────────────────────────────────
  float32         4            기준선            기준선
  float16         2            무시할 수 있음     ~동일
  스칼라 (int8)    1            1-3% 재현율       ~1.5배 빠름
  PQ (m=96)       96/dim       5-10% 재현율      ~2-3배 빠름
  바이너리         1/8          10-20% 재현율     ~10배 빠름

  권장 진행 순서:
  1. float32로 시작 (정확성 우선)
  2. int8/SQ8로 전환 (쉬운 4배 메모리 축소)
  3. 메모리가 여전히 부족하면 PQ 추가 (튜닝 필요)
  4. 바이너리는 초기 후보 스크리닝에만
```

### 6.3 계층형 저장소

```
계층형 저장소는 대규모 컬렉션의 비용을 절감합니다:

  핫 티어 (RAM):
  ┌────────────────────────────────────┐
  │ 최근/자주 접근되는 벡터              │
  │ HNSW 인덱스가 완전히 메모리에        │
  │ 비용: $7-12/GB/월                  │
  │ 지연시간: 1-5ms                    │
  └────────────────────────────────────┘
       ↕ 승격/강등
  웜 티어 (SSD/mmap):
  ┌────────────────────────────────────┐
  │ 오래된 벡터, 중간 접근 빈도          │
  │ HNSW 그래프는 RAM, 벡터는 디스크     │
  │ 비용: $0.50-1/GB/월               │
  │ 지연시간: 5-20ms                   │
  └────────────────────────────────────┘
       ↕ 아카이브
  콜드 티어 (오브젝트 스토리지):
  ┌────────────────────────────────────┐
  │ 아카이브, 거의 접근하지 않음          │
  │ 인덱스 없음 (필요시 브루트포스)       │
  │ 비용: $0.02/GB/월                  │
  │ 지연시간: 100ms-1s                 │
  └────────────────────────────────────┘

  구현:
  - Qdrant: on_disk=True로 mmap (웜 티어)
  - Milvus: MinIO 콜드 티어와 계층형 저장소
  - Pinecone: 자동 (서버리스 가격이 처리)
```

### 6.4 비용 비교 테이블

```
월간 비용 추정: 1000만 벡터, 768차원, 1000 QPS:

  옵션                        메모리      컴퓨트      스토리지    월간 합계
  ──────────────────────────────────────────────────────────────────────
  자체 호스팅 Qdrant            $280      $400        $50        ~$730
  (3x r6i.2xlarge, SQ8)

  자체 호스팅 Milvus            $400      $600        $100       ~$1,100
  (분산형, 5 노드)

  Pinecone 서버리스             N/A       N/A         N/A        ~$700-1,500
  (읽기/쓰기당 과금)           (읽기/쓰기 패턴에 따라)

  Weaviate Cloud               N/A       N/A         N/A        ~$800-1,200
  (관리형)

  EC2 위 FAISS                 $280      $200        $20        ~$500
  (단일 노드, HA 없음)         (가장 저렴하나 내장 HA 없음)
```

---

## 7. 프로덕션 배포 패턴

### 7.1 블루-그린 인덱스 배포

```
무중단 인덱스 업데이트를 위한 블루-그린 배포:

  ┌──────────────┐
  │ 로드 밸런서   │
  └──────┬───────┘
         │
    ┌────┴────┐
    │         │
    ▼         ▼
  ┌─────┐  ┌─────┐
  │Blue │  │Green│
  │(v1) │  │(v2) │   ← Green에 새 인덱스 구축
  │ ✓   │  │ ... │
  └─────┘  └─────┘

  단계:
  1. Blue가 트래픽 처리 중 (인덱스 v1)
  2. Green에 새 인덱스 (v2) 구축 (오프라인, 영향 없음)
  3. Green에서 검증 쿼리 실행
  4. 로드 밸런서를 Green으로 전환
  5. Blue가 다음 빌드 대상이 됨

  장점:
  - 인덱스 재구축 중 무중단
  - 즉시 롤백 (Blue로 다시 전환)
  - 버전 간 A/B 테스트 가능
```

### 7.2 섀도우 인덱스 패턴

```
안전한 임베딩 모델 마이그레이션을 위한 섀도우 인덱싱:

  쿼리
    │
    ├──────────────────┐
    │                  │ (비동기, 비차단)
    ▼                  ▼
  ┌──────────┐   ┌──────────┐
  │ 프라이머리 │   │ 섀도우   │
  │ 인덱스    │   │ 인덱스    │
  │ (모델 A)  │   │ (모델 B) │
  └────┬─────┘   └────┬─────┘
       │               │
       ▼               ▼
  사용자에게        비교를 위해
  응답             결과 로깅

  비교 데이터 수집 후:
  - 섀도우 재현율 > 프라이머리 재현율 → 섀도우 승격
  - 섀도우 재현율 < 프라이머리 재현율 → 섀도우 폐기
  - 점진적 트래픽 전환: 0% → 10% → 50% → 100%
```

### 7.3 임베딩 버전 관리

```python
"""
프로덕션에서 임베딩 모델 버전을 관리합니다.
임베딩 모델을 업데이트하면 모든 벡터를 재임베딩해야 합니다.
"""

class EmbeddingVersionManager:
    """임베딩 모델 버전을 추적하고 관리합니다."""

    def __init__(self, vector_db_client, metadata_store):
        self.db = vector_db_client
        self.meta = metadata_store

    def start_migration(self, new_model: str, new_dim: int):
        """새 임베딩 모델로 마이그레이션을 시작합니다."""
        # 버전 접미사가 있는 새 컬렉션 생성
        new_collection = f"documents_v{self._next_version()}"
        self.db.create_collection(
            name=new_collection,
            dimension=new_dim,
        )

        self.meta.record_migration(
            status="in_progress",
            source_collection=self._current_collection(),
            target_collection=new_collection,
            new_model=new_model,
        )
        return new_collection

    def migrate_batch(self, batch_ids: list[str], new_embeddings: list):
        """재임베딩된 벡터의 배치를 업서트합니다."""
        target = self.meta.get_active_migration()["target_collection"]
        self.db.upsert(collection=target, ids=batch_ids, vectors=new_embeddings)

    def complete_migration(self):
        """새 컬렉션으로 트래픽을 전환합니다."""
        migration = self.meta.get_active_migration()
        # 별칭을 새 컬렉션으로 업데이트
        self.db.update_alias(
            alias="documents",
            collection=migration["target_collection"],
        )
        self.meta.record_migration(status="completed")

    def rollback_migration(self):
        """이전 컬렉션으로 되돌립니다."""
        migration = self.meta.get_active_migration()
        self.db.update_alias(
            alias="documents",
            collection=migration["source_collection"],
        )
        self.meta.record_migration(status="rolled_back")
```

---

## 8. 데이터 파이프라인과의 통합

### 8.1 배치 임베딩 업데이트 파이프라인

```python
"""
증분 벡터 인덱스 업데이트를 위한 Airflow DAG.
매일 실행하여 새/업데이트된 문서를 임베딩하고 벡터 DB에 업서트합니다.
"""

from airflow.decorators import dag, task
from datetime import datetime, timedelta

@dag(
    schedule="0 4 * * *",  # 매일 오전 4시
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["vector", "embeddings"],
    default_args={"retries": 2, "retry_delay": timedelta(minutes=5)},
)
def incremental_vector_update():

    @task()
    def detect_changes(ds=None):
        """마지막 성공 실행 이후 변경된 문서를 감지합니다."""
        # CDC 테이블 또는 데이터 웨어하우스에서 변경 사항 쿼리
        changes = {
            "new_docs": 1200,
            "updated_docs": 350,
            "deleted_doc_ids": ["doc-old-1", "doc-old-2"],
            "source_path": f"s3://lake/gold/documents/dt={ds}/",
        }
        return changes

    @task()
    def generate_embeddings(changes: dict):
        """새/업데이트된 문서의 임베딩을 생성합니다."""
        # API 또는 로컬 모델로 배치 임베딩
        # 대규모 배치는 OOM 방지를 위해 청크 처리
        batch_size = 256
        total = changes["new_docs"] + changes["updated_docs"]
        n_batches = (total + batch_size - 1) // batch_size

        return {
            "embeddings_path": "s3://embeddings/incremental/2024-06-15/",
            "total_embedded": total,
            "n_batches": n_batches,
            "model": "text-embedding-3-small",
            "model_version": "v2",
        }

    @task()
    def upsert_vectors(embedding_info: dict, changes: dict):
        """새 임베딩을 업서트하고 제거된 문서를 삭제합니다."""
        # S3에서 임베딩 로드
        # 벡터 DB에 배치 업서트
        # 제거된 문서 삭제
        return {
            "upserted": embedding_info["total_embedded"],
            "deleted": len(changes["deleted_doc_ids"]),
            "collection": "documents",
        }

    @task()
    def validate_index(upsert_result: dict):
        """업데이트된 인덱스에 품질 검사를 실행합니다."""
        checks = {
            "total_vectors_after": 1_250_000,
            "recall_at_10": 0.96,
            "p99_latency_ms": 4.2,
            "empty_result_rate": 0.002,
        }
        # 재현율이 임계값 아래로 떨어지면 알림
        if checks["recall_at_10"] < 0.90:
            raise ValueError(
                f"재현율이 {checks['recall_at_10']}으로 하락"
            )
        return checks

    changes = detect_changes()
    embeddings = generate_embeddings(changes)
    result = upsert_vectors(embeddings, changes)
    validate_index(result)

incremental_vector_update()
```

### 8.2 Kafka를 활용한 스트리밍 벡터 업데이트

```python
"""
Kafka 컨슈머를 통한 실시간 벡터 업데이트.
임베딩 최신성이 중요한 사용 사례에 적합 (예: 뉴스, 지원 티켓).
"""

from confluent_kafka import Consumer, KafkaError
import json

def vector_update_consumer(
    vector_db_client,
    embedding_model,
    kafka_config: dict,
    topic: str = "document-changes",
):
    """문서 변경 사항을 소비하고 벡터 인덱스를 실시간으로 업데이트합니다."""
    consumer = Consumer({
        "bootstrap.servers": kafka_config["brokers"],
        "group.id": "vector-updater",
        "auto.offset.reset": "earliest",
        "enable.auto.commit": False,
    })
    consumer.subscribe([topic])

    batch = []
    batch_size = 64
    flush_interval_seconds = 5

    try:
        while True:
            msg = consumer.poll(timeout=1.0)
            if msg is None:
                if batch:
                    flush_batch(vector_db_client, embedding_model, batch)
                    consumer.commit()
                    batch = []
                continue
            if msg.error():
                if msg.error().code() != KafkaError._PARTITION_EOF:
                    raise Exception(msg.error())
                continue

            event = json.loads(msg.value())
            batch.append(event)

            if len(batch) >= batch_size:
                flush_batch(vector_db_client, embedding_model, batch)
                consumer.commit()
                batch = []
    finally:
        consumer.close()


def flush_batch(db_client, model, events: list):
    """문서 변경 이벤트 배치를 처리합니다."""
    upserts = []
    deletes = []

    for event in events:
        if event["op"] in ("insert", "update"):
            embedding = model.encode(event["text"])
            upserts.append({
                "id": event["doc_id"],
                "vector": embedding.tolist(),
                "metadata": event.get("metadata", {}),
            })
        elif event["op"] == "delete":
            deletes.append(event["doc_id"])

    if upserts:
        db_client.upsert(collection="documents", points=upserts)
    if deletes:
        db_client.delete(collection="documents", ids=deletes)
```

### 8.3 CDC에서 벡터 DB까지 패턴

```
Change Data Capture → 벡터 DB 통합:

  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
  │ PostgreSQL│───→│ Debezium │───→│  Kafka   │───→│ 벡터     │
  │ (소스)    │    │ (CDC)    │    │ (버퍼)   │    │ 업데이터  │
  └──────────┘    └──────────┘    └──────────┘    └────┬─────┘
                                                       │
                                                       ▼
                                                  ┌──────────┐
                                                  │ 임베딩    │
                                                  │ 모델     │
                                                  └────┬─────┘
                                                       │
                                                       ▼
                                                  ┌──────────┐
                                                  │ 벡터 DB  │
                                                  │ (Qdrant) │
                                                  └──────────┘

  이 패턴은 벡터 인덱스가 소스 데이터와 동기화되도록 합니다:
  - Debezium이 PostgreSQL WAL에서 모든 INSERT/UPDATE/DELETE를 캡처
  - Kafka가 안정성을 위해 이벤트를 버퍼링
  - 벡터 업데이터가 텍스트 필드를 임베딩하고 벡터 DB에 업서트/삭제
  - 엔드투엔드 지연시간: 일반적으로 1-5초
```

---

## 요약

```
핵심 요점:

1. 하이브리드 검색(밀집 + 희소)은 단독보다 우수
   — 융합에 RRF 또는 가중 선형 결합 사용

2. 메타데이터 필터링 전략이 중요: 대규모 파티션에는
   사전 필터, 선택적 쿼리에는 사후 필터, 또는
   데이터베이스 네이티브 하이브리드 접근법 사용

3. 크로스 인코더 리랭킹은 정밀도를 10-20% 향상시키지만
   ~50ms 지연시간 추가 — top-K 정제에만 사용

4. 스캐터-개더 샤딩 + 읽기 복제본으로 확장
   — 메모리 추정 공식을 사용한 용량 계획

5. 재현율, 지연시간 백분위수, 빈 결과율을 모니터링
   — 재현율 저하는 검색 품질의 조용한 킬러

6. 비용을 점진적으로 최적화: int8 양자화 → 차원
   축소 → 계층형 저장소 → PQ (필요시)

7. 배치 DAG(Airflow) 또는 스트리밍 컨슈머(Kafka CDC)로
   데이터 파이프라인에 벡터 검색 통합
```

---

## 연습 문제

1. **하이브리드 검색 융합**: RRF와 선형 결합 융합을 모두 구현하세요. 한 방법이 다른 방법보다 우수한 10개의 쿼리를 생성하세요.

2. **필터 벤치마크**: 풍부한 메타데이터가 있는 데이터셋을 만드세요. 다른 선택도 수준에서 사전 필터링 vs 사후 필터링의 검색 지연시간과 재현율을 측정하세요.

3. **리랭킹 파이프라인**: 검색에 바이 인코더, 리랭킹에 크로스 인코더를 사용하는 2단계 검색 파이프라인을 구축하세요. MRR 향상을 측정하세요.

4. **용량 계획기**: AWS, GCP, Azure 인스턴스에 대한 비용 추정을 포함하도록 용량 계획 계산기를 확장하세요.

5. **모니터링 대시보드**: 벡터 검색 서비스에 Prometheus 메트릭을 구현하고 Grafana 대시보드 JSON을 생성하세요.

---

## 더 읽을거리

- [역순위 융합 논문 (Cormack et al., 2009)](https://dl.acm.org/doi/10.1145/1571941.1572114)
- [SPLADE: 희소 어휘 및 확장 모델](https://arxiv.org/abs/2107.05720)
- [리랭킹을 위한 크로스 인코더](https://www.sbert.net/examples/applications/cross-encoder/README.html)
- [Qdrant 모니터링 가이드](https://qdrant.tech/documentation/guides/monitoring/)
- [Milvus 용량 계획](https://milvus.io/docs/sizing.md)
- [벡터 데이터베이스 벤치마크 (ANN-Benchmarks)](https://ann-benchmarks.com/)

[← 이전: 22. 벡터 저장소와 인덱싱](22_Vector_Storage_and_Indexing.md) | [다음: 개요 →](00_Overview.md)
