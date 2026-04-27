[← 이전: 21. 데이터 버전 관리와 데이터 계약](21_Data_Versioning_and_Contracts.md) | [다음: 23. 프로덕션 벡터 검색 →](23_Production_Vector_Search.md)

# 22. 벡터 저장소와 인덱싱

## 학습 목표

1. 벡터 저장소 아키텍처를 이해하고 인메모리, 디스크 기반, 분산 방식 간의 트레이드오프를 파악한다
2. FAISS 인덱스 유형(Flat, IVF, HNSW, PQ)과 팩토리 문자열을 통한 복합 인덱스 구성을 숙달한다
3. Milvus의 분산 아키텍처(프록시, 쿼리, 데이터, 인덱스 노드)를 설명할 수 있다
4. Weaviate의 모듈 시스템, 벡터라이저 파이프라인, GraphQL API 설계를 이해한다
5. Pinecone의 서버리스 모델과 Qdrant의 고급 메타데이터 필터링을 활용할 수 있다
6. Chroma를 임베디드 및 클라이언트-서버 모드로 배포하여 프로토타이핑과 프로덕션에 활용한다
7. 벡터 데이터베이스를 성능, 확장성, 비용, 생태계 차원에서 비교 평가한다

---

## 개요

현대 데이터 파이프라인은 고차원 벡터 데이터를 점점 더 많이 다루고 있습니다 — 언어 모델, 이미지 인코더, 추천 시스템, 과학 시뮬레이션의 임베딩이 그 예입니다. 이러한 벡터를 효율적으로 저장하고 검색하는 것은 인덱싱 이론, 분산 시스템, 하드웨어 최적화가 교차하는 데이터 엔지니어링 문제입니다.

이 레슨에서는 저장 계층을 다룹니다: 벡터가 주요 도구들에서 어떻게 영속화되고, 인덱싱되고, 쿼리되는지 살펴봅니다. 기초 라이브러리(FAISS)부터 시작하여, 전용 벡터 데이터베이스(Milvus, Weaviate, Pinecone, Qdrant, Chroma)를 검토하고, 기술 선택을 안내하는 벤치마크 비교로 마무리합니다.

> **데이터 엔지니어링 관점**: 벡터 저장소는 ML만의 관심사가 아닙니다. 데이터 엔지니어는 임베딩 파이프라인을 설계하고, 인덱스 수명주기를 관리하고, 메타데이터의 스키마 진화를 처리하며, 벡터 저장소를 웨어하우스, 레이크, 스트리밍 시스템과 함께 더 넓은 데이터 플랫폼에 통합해야 합니다.

---

## 1. 벡터 저장소 아키텍처

### 이론: 차원의 저주

저차원에서 최근접 이웃 검색은 단순 — kd-tree가 O(log N)을 줍니다. 고차원(D > 30 정도)에서 kd-tree는 퇴화: 거의 모든 점이 쿼리에서 "대략 등거리"가 되고, 트리 가지치기가 실패.

전형적 임베딩(BERT의 D = 768, OpenAI의 D = 1536)의 경우, brute-force 비교는 쿼리당 O(N × D). N = 1억, D = 1536에서 이는 쿼리당 1500억 부동 소수점 곱셈 — GPU에서도 초 단위.

이는 더 빠른 하드웨어로 해결 불가능. 수학이 반격. 해결책은 정확한 답을 포기하는 것입니다.

### 이론: 거리 메트릭

"가장 가까움"이 무엇을 의미하는가?

#### B.1 Euclidean (L2)

`d(a, b) = sqrt(sum((a_i - b_i)^2))`. 기하학적 거리. 벡터 크기에 민감.

사용 시점: 벡터가 절대 좌표에 있을 때(예: 물리 센서 데이터, 원시 이미지 픽셀).

#### B.2 Cosine 유사도

`cos(a, b) = (a · b) / (|a| |b|)`. 크기를 무시하고 벡터 사이의 각도 측정.

사용 시점: 벡터가 의미 공간의 방향을 나타낼 때(텍스트 임베딩, 이미지 임베딩). 의미적으로 유사한 두 텍스트가 길이와 무관하게 가까워야 하므로 ML 임베딩의 지배적 선택.

#### B.3 Dot product (inner product)

`a · b = sum(a_i × b_i)`. 벡터가 unit length로 정규화되면 cosine과 동등.

사용 시점: 벡터가 정규화됨(대부분 현대 임베딩은 그러함). 크기 나눗셈 없으므로 cosine보다 빠름 — 그리고 대부분 ANN 라이브러리가 이를 위해 최적화.

#### B.4 실용적 규칙

현대 텍스트/이미지 임베딩의 경우, 벡터가 보통 프로덕션 시점에 L2 정규화됨. 그러면 dot product = cosine similarity, 둘 다 Euclidean보다 빠르게 계산됨. 이것이 FAISS, Milvus, Pinecone 모두 기본으로 하는 것입니다.

### 1.1 저장 모델 분류

```
벡터 저장 모델:

┌──────────────────────────────────────────────────────────────────────┐
│                        인메모리                                       │
│  ┌──────────────────────────────────────────────────┐               │
│  │ 모든 벡터 + 인덱스가 RAM에 상주                      │               │
│  │ ✓ 최저 지연시간 (~1ms p99)                          │               │
│  │ ✓ 가장 단순한 아키텍처                               │               │
│  │ ✗ 비용: $7-12/GB/월 (클라우드 RAM)                   │               │
│  │ ✗ WAL 없이는 장애 시 데이터 유실                      │               │
│  │ 예시: FAISS (기본), Qdrant (기본)                    │               │
│  └──────────────────────────────────────────────────┘               │
│                                                                      │
│                        디스크 기반                                     │
│  ┌──────────────────────────────────────────────────┐               │
│  │ 인덱스는 RAM, 벡터는 디스크 (mmap)                    │               │
│  │ ✓ 10-50배 저렴한 저장 비용                           │               │
│  │ ✓ RAM보다 큰 데이터셋 처리 가능                       │               │
│  │ ✗ 더 높은 지연시간 (~5-20ms p99)                     │               │
│  │ ✗ OS 페이지 캐시에 성능 의존                          │               │
│  │ 예시: FAISS OnDisk, Qdrant mmap, Weaviate          │               │
│  └──────────────────────────────────────────────────┘               │
│                                                                      │
│                        분산형                                         │
│  ┌──────────────────────────────────────────────────┐               │
│  │ 여러 노드에 분할, HA를 위한 복제                       │               │
│  │ ✓ 수십억 벡터까지 확장                               │               │
│  │ ✓ 장애 허용                                         │               │
│  │ ✗ 네트워크 지연 오버헤드                              │               │
│  │ ✗ 운영 복잡성                                       │               │
│  │ 예시: Milvus, Pinecone, Weaviate 클러스터           │               │
│  └──────────────────────────────────────────────────┘               │
└──────────────────────────────────────────────────────────────────────┘
```

### 1.2 메모리 추정

```
벡터 데이터 메모리 공식:

  원시 벡터 메모리 = N × d × 요소당_바이트

  여기서:
    N = 벡터 수
    d = 임베딩 차원
    요소당_바이트는 형식에 따라:
      float32 = 4 바이트
      float16 = 2 바이트
      int8 (SQ8) = 1 바이트
      PQ (m개 서브양자화기) = m 바이트

  예시: 1000만 벡터, 768차원

  형식          벡터 메모리     인덱스 오버헤드 (HNSW M=16)    합계
  ────────────────────────────────────────────────────────────────────
  float32       28.7 GB          ~2.4 GB                       ~31.1 GB
  float16       14.3 GB          ~2.4 GB                       ~16.7 GB
  int8 (SQ8)    7.2 GB           ~2.4 GB                       ~9.6 GB
  PQ (m=96)     0.9 GB           ~2.4 GB                       ~3.3 GB

  HNSW 그래프 오버헤드 ≈ N × M × 2 × 8 바이트
  M=16 기준: 10M × 16 × 2 × 8 = 2.4 GB (그래프 구조만)
```

### 1.3 메모리 맵 저장소

```python
"""
메모리 맵 벡터 저장소는 RAM보다 큰 데이터셋을 다룰 수 있게 합니다.
OS가 물리 메모리에 어떤 페이지가 상주할지 관리합니다.
"""

import numpy as np
import os

def create_mmap_vectors(path: str, n_vectors: int, dim: int) -> np.ndarray:
    """메모리 맵 벡터 파일을 생성합니다."""
    fp = np.memmap(path, dtype='float32', mode='w+', shape=(n_vectors, dim))
    return fp

def load_mmap_vectors(path: str, n_vectors: int, dim: int) -> np.ndarray:
    """기존 메모리 맵 벡터를 로드합니다 (RAM 복사 없음)."""
    fp = np.memmap(path, dtype='float32', mode='r', shape=(n_vectors, dim))
    return fp

# 대규모 데이터셋 사용 패턴
n_vectors = 10_000_000
dim = 768
path = "/data/vectors.mmap"

# 쓰기 단계 (배치 ETL)
vectors = create_mmap_vectors(path, n_vectors, dim)
batch_size = 100_000
for i in range(0, n_vectors, batch_size):
    end = min(i + batch_size, n_vectors)
    vectors[i:end] = np.random.randn(end - i, dim).astype('float32')
    vectors.flush()  # 디스크에 영속화

# 읽기 단계 (검색 서비스)
vectors = load_mmap_vectors(path, n_vectors, dim)
query = np.random.randn(1, dim).astype('float32')

# OS는 검색 중 접근된 페이지만 로드합니다
# RAM이 8 GB이고 벡터가 28.7 GB이면, 한 번에 ~28%만 메모리에 있습니다
```

### 1.4 WAL (Write-Ahead Log)을 통한 내구성

```
WAL은 장애 시 데이터 유실을 방지합니다:

  쓰기 요청
       │
       ▼
  ┌─────────────┐    ① WAL에 추가 (순차 I/O, 빠름)
  │   WAL 파일   │──────────────────────────────────────────┐
  └──────┬──────┘                                           │
         │ ② 클라이언트에 ACK                                │
         ▼                                                  │
  ┌─────────────┐    ③ 세그먼트로 배치 플러시 (백그라운드)       │
  │  인메모리     │◄─────────────────────────────────────────┘
  │  버퍼        │
  └──────┬──────┘
         │ ④ 버퍼가 가득 차면 → 봉인된 세그먼트 생성
         ▼
  ┌─────────────┐
  │  봉인된      │    ⑤ 봉인된 세그먼트에 인덱스 빌드
  │  세그먼트    │
  └─────────────┘

  장애 시: 세그먼트에 아직 플러시되지 않은 WAL 항목을 재생
  성공적인 플러시 후 WAL은 잘림

  WAL 사용 데이터베이스: Milvus, Qdrant, Weaviate
  WAL 없는 라이브러리: FAISS (사용자가 영속화 처리)
```

---

## 2. FAISS 심층 분석

### 이론: ANN 알고리즘 가족

세 가지 지배적 가족, 세 가지 다른 설계 철학.

#### C.1 IVF (Inverted File Index)

클러스터링 접근.

1. **학습:** k-means로 모든 N 벡터를 K 중심점(centroid)으로 클러스터링(K는 일반적으로 1000-10000).
2. **인덱싱:** 각 벡터를 가장 가까운 centroid에 할당; 버킷팅.
3. **쿼리:** 쿼리에 가장 가까운 M centroid 찾음; 그 버킷만 검색.

속도: O(K + M × N/K). 트레이드오프: M 증가는 recall을 향상시키지만 쿼리를 느리게 함. M=1이 70% recall을 줄 수 있음; M=20이 95%를 줄 수 있음.

강점: 단순, 빠른 빌드, 중간 규모(수백만 ~ ~1억 벡터)에서 잘 작동. FAISS에서 `IndexIVFFlat`으로 표준.

#### C.2 HNSW (Hierarchical Navigable Small World)

그래프 접근.

1. **인덱싱:** 다층 그래프 빌드. 최상위 층은 sparse(긴 거리 엣지 적음); 최하위 층은 dense(짧은 엣지 많음). 각 벡터를 각 층의 가장 가까운 이웃과의 엣지로 삽입.
2. **쿼리:** 최상위 층에서 시작, 쿼리를 향해 greedy-walk(항상 쿼리에 가장 가까운 이웃으로 step). 한 층 내려옴; 반복. 최하위 층에서 K 가장 가까운 이웃이 결과.

속도: 쿼리당 O(log N). Recall: 합리적 매개변수로 일반적으로 95-99%.

강점: 중간 규모 워크로드에 대한 best-in-class recall/속도 트레이드오프. pgvector, Qdrant, Weaviate, 대부분 현대 벡터 DB에서 표준.

약점: 높은 메모리(그래프 엣지); 빌드 느림; 업데이트가 까다로움.

#### C.3 PQ (Product Quantization)

압축 접근. 검색 알고리즘 그 자체가 아니라, IVF나 HNSW가 더 적은 메모리를 사용하도록 만드는 방법.

1. 각 D차원 벡터를 D/M 차원의 M개 sub-vector로 분할.
2. 각 sub-space를 독립적으로 256 centroid로 k-means 클러스터링.
3. 각 벡터를 D × 4 바이트 대신 M 바이트(sub-space당 한 centroid 인덱스)로 저장.

압축: 32x ~ 64x. 거리 계산: 클러스터-to-클러스터 lookup table을 통한 근사.

강점: 단일 머신에서 십억 규모 벡터 검색을 가능하게 함. FAISS에서 `IndexIVFPQ`로 표준.

약점: 손실. Recall이 비압축 대비 약간 떨어짐.

#### C.4 조합

프로덕션 시스템은 이것들을 결합. `IndexIVFPQ`(FAISS): 파티셔닝을 위한 IVF + 압축을 위한 PQ. Pinecone: HNSW + PQ 스타일 압축. 각 도구의 인덱스 선택은 이 설계 공간 위의 튜닝.

### 이론: Recall-속도-메모리 삼각형

모든 ANN 인덱스는 3D 표면 어딘가에 있음:

- **Recall:** 인덱스가 진짜 K-NN의 어떤 분수를 반환하는가? 100% = 정확; 더 낮음 = 근사.
- **쿼리 속도:** 노드당 초당 쿼리.
- **메모리:** 인덱스를 보유하는 데 필요한 RAM.

하나를 증가시키면 보통 다른 것이 비용. `IndexFlat`(brute force)는 100% recall이지만 느림. `IndexHNSWFlat`은 빠르고 ~98% recall이지만 raw 벡터의 2배 메모리 사용. `IndexIVFPQ`는 30배 적은 메모리 사용하지만 recall이 90%로 떨어지고 쿼리가 HNSW보다 느림.

인덱스 선택 = 당신의 워크로드를 위해 이 삼각형 위의 점을 선택. 95% recall이 허용 가능하고 메모리가 제약되면 IVFPQ. Recall이 결정적이고 메모리가 풍부하면 HNSW. 쿼리 지연이 우선이면 높은 `ef` 매개변수의 HNSW.

### 2.1 생태계에서의 FAISS 위치

```
FAISS 위치:

  애플리케이션 계층:    LangChain, LlamaIndex, 커스텀 앱
         │
  벡터 DB 계층:        Milvus, Weaviate, Qdrant, Chroma
         │
  검색 엔진 계층:      FAISS, ScaNN, Annoy, hnswlib   ← FAISS는 여기
         │
  하드웨어 계층:       CPU (AVX2/AVX-512), GPU (CUDA)

FAISS는 라이브러리이지 데이터베이스가 아닙니다:
  ✓ 서버 프로세스 없음, 네트워크 프로토콜 없음
  ✓ C++ 코어 + Python 바인딩
  ✓ 팩토리 문자열로 인덱스 유형 조합 가능
  ✓ 학습과 검색에 GPU 가속
  ✗ 내장 영속화 없음 (사용자가 저장/로드)
  ✗ 메타데이터 필터링 없음 (벡터만)
  ✗ 복제나 샤딩 없음
```

### 2.2 핵심 인덱스 유형

```python
import faiss
import numpy as np

dim = 768
n_vectors = 1_000_000
n_query = 100
k = 10

# 샘플 데이터 생성
xb = np.random.randn(n_vectors, dim).astype('float32')
xq = np.random.randn(n_query, dim).astype('float32')

# ─── IndexFlatL2: 정확한 브루트포스 검색 ───
index_flat = faiss.IndexFlatL2(dim)
index_flat.add(xb)
distances, indices = index_flat.search(xq, k)
# 시간: O(N × d) 쿼리당 → 100만 벡터에 ~200ms
# 용도: 그라운드 트루스, 소규모 데이터셋 (<100K)

# ─── IndexIVFFlat: 역파일 + 조대 양자화기 ───
nlist = 1024  # 보로노이 셀 수
quantizer = faiss.IndexFlatL2(dim)
index_ivf = faiss.IndexIVFFlat(quantizer, dim, nlist)
index_ivf.train(xb)  # 클러스터 중심점 학습
index_ivf.add(xb)
index_ivf.nprobe = 32  # 1024개 셀 중 32개 탐색
distances, indices = index_ivf.search(xq, k)
# 시간: O(nprobe/nlist × N × d) → ~6ms
# 재현율: nprobe=32에서 ~95%

# ─── IndexHNSWFlat: 계층적 탐색 가능 소세계 ───
M = 32  # 노드당 연결 수
index_hnsw = faiss.IndexHNSWFlat(dim, M)
index_hnsw.hnsw.efConstruction = 200  # 빌드 품질
index_hnsw.hnsw.efSearch = 64         # 검색 품질
index_hnsw.add(xb)
distances, indices = index_hnsw.search(xq, k)
# 시간: ~2ms  |  재현율: ~99%
# 트레이드오프: 높은 메모리 사용량 (그래프 구조)
```

### 2.3 곱 양자화 (Product Quantization)

```python
# ─── IndexPQ: 메모리 효율을 위한 벡터 압축 ───
m = 96      # 서브양자화기 수 (dim을 나눌 수 있어야 함)
nbits = 8   # 서브양자화기당 비트 수 (각각 256개 중심점)
index_pq = faiss.IndexPQ(dim, m, nbits)
index_pq.train(xb)
index_pq.add(xb)
distances, indices = index_pq.search(xq, k)
# 메모리: 벡터당 96 바이트 (float32의 3072 바이트 대비)
# 압축비: 32배
# 재현율: 데이터 분포에 따라 ~85-92%

# ─── IndexIVFPQ: IVF + PQ (대규모의 워크호스) ───
nlist = 4096
m = 96
nbits = 8
quantizer = faiss.IndexFlatL2(dim)
index_ivfpq = faiss.IndexIVFPQ(quantizer, dim, nlist, m, nbits)
index_ivfpq.train(xb)
index_ivfpq.add(xb)
index_ivfpq.nprobe = 64
distances, indices = index_ivfpq.search(xq, k)
# 메모리: ~100 바이트/벡터  |  속도: ~1ms  |  재현율: ~90%
```

### 2.4 팩토리 문자열을 통한 복합 인덱스

```python
"""
FAISS 팩토리 문자열은 인덱스 유형을 선언적으로 조합합니다:

  "Flat"           → IndexFlatL2
  "IVF1024,Flat"   → 1024 셀의 IndexIVFFlat
  "IVF4096,PQ96"   → 4096 셀, 96 서브양자화기의 IndexIVFPQ
  "HNSW32"         → M=32의 IndexHNSWFlat
  "IVF1024,HNSW32" → HNSW 양자화기가 있는 IVF (빠른 조대 검색)
  "OPQ96,IVF4096,PQ96" → OPQ 회전 + IVF + PQ (최고 압축)
  "IVF4096,SQ8"    → 스칼라 양자화(int8)가 있는 IVF
"""

# 한 줄로 복합 인덱스 구축
index = faiss.index_factory(dim, "OPQ96,IVF4096,PQ96")
index.train(xb)
index.add(xb)

# GPU 학습 (대규모 데이터셋에 훨씬 빠름)
# res = faiss.StandardGpuResources()
# index_gpu = faiss.index_cpu_to_gpu(res, 0, index)

# 저장과 로드
faiss.write_index(index, "/data/faiss_index.bin")
index_loaded = faiss.read_index("/data/faiss_index.bin")
```

### 2.5 FAISS 인덱스 선택 가이드

```
FAISS 인덱스 선택 결정 트리:

  데이터셋 크기?
  │
  ├── < 10만 벡터
  │   └── IndexFlatL2 (정확, 학습 불필요)
  │
  ├── 10만 - 100만 벡터
  │   ├── 메모리 충분? → IndexHNSWFlat (최고 재현율)
  │   └── 메모리 부족? → IndexIVFFlat (nlist=sqrt(N))
  │
  ├── 100만 - 1억 벡터
  │   ├── 지연시간 우선? → IVF + HNSW 양자화기
  │   ├── 메모리 우선? → IVF + PQ (또는 OPQ + IVF + PQ)
  │   └── 균형? → IVF + SQ8
  │
  └── > 1억 벡터
      ├── 단일 머신? → OPQ + IVF + PQ + 디스크 I/O
      └── 멀티 GPU? → GPU 간 IndexIVFPQ 분할

  학습 데이터 요구량:
    IVF: 30 × nlist ~ 256 × nlist 벡터
    PQ: 1만-10만 대표 벡터
    OPQ: PQ와 동일 (회전 행렬 학습)
```

---

## 3. Milvus 아키텍처

### 이론: 대규모 벡터 저장 아키텍처

벡터 인덱스는 한 컴포넌트일 뿐; 벡터를 시스템 *안* 으로 가져오고 *밖* 으로 가져오는 것은 데이터 엔지니어링 문제.

#### E.1 임베딩 파이프라인

1. 소스 콘텐츠(문서, 이미지, 제품)가 primary 저장소(DB, S3)에 살음.
2. 임베딩 서비스(OpenAI API, GPU의 로컬 모델)가 벡터 계산.
3. 벡터가 메타데이터(소스 ID, 필터링을 위한 스키마 필드)와 함께 벡터 DB에 작성됨.
4. 변경 시(새 콘텐츠, 콘텐츠 업데이트) 임베딩이 재계산되고 벡터 DB가 업데이트됨.

이는 CDC + 변환 + 적재 파이프라인(레슨 18, 12)입니다. 벡터 DB가 sink. 실패 모드: 오래된 임베딩(소스 업데이트되었지만 벡터 새로고침 안 됨), 누락된 임베딩(새 콘텐츠가 아직 인덱스 안 됨), 소스와 벡터 DB 메타데이터 사이의 스키마 drift.

#### E.2 파티셔닝과 샤딩

- **테넌트별 파티셔닝.** 각 고객 / 네임스페이스가 자체 벡터 컬렉션. 검색이 한 파티션에 스코프됨. 다중 테넌트 SaaS에 흔함.
- **키별 샤딩.** 단일 논리적 컬렉션이 노드 간 키로 샤딩됨. 각 쿼리가 모든 샤드로 scatter, 결과 gather. 십억 규모의 표준.
- **Hot-cold 계층화.** 최근 벡터는 빠른 HNSW에; 옛 벡터는 압축된 IVFPQ에; 아카이브는 객체 스토리지에. 쿼리가 각 계층에서 누적.

#### E.3 메타데이터-필터 문제

흔한 쿼리: "유사한 제품 찾되, Category=X와 Price<100인 것만." 순진한 접근(top 1000 검색, 필터)은 필터가 선택적일 때(1000개 결과 중 5개만 필터와 일치) 깨짐. 해결책:

- **Pre-filter:** 메타데이터 필터를 먼저 적용, 그다음 매치하는 벡터만 검색. 메타데이터 인덱스 필요.
- **Overshoot가 있는 post-filter:** top 10000 검색, 필터, top 10 반환. 필터가 선택적일 때 컴퓨트 낭비.
- **필터 인식 인덱스:** 그래프 순회 수준에서 필터를 지원하는 인덱스(Weaviate의 filtered HNSW, Qdrant의 filterable HNSW). 필터가 흔할 때 최고의 성능.

이는 프로덕션 벡터 검색의 가장 깊은 설계 이슈 중 하나입니다.

### 3.1 분산 컴포넌트

```
Milvus 분산 아키텍처:

  ┌─────────────────────────────────────────────────────┐
  │                    클라이언트                          │
  │  (Python SDK, Java SDK, Go SDK, REST, gRPC)         │
  └──────────────────────┬──────────────────────────────┘
                         │
  ┌──────────────────────▼──────────────────────────────┐
  │                  프록시 계층                           │
  │  (부하 분산, 요청 라우팅, 인증)                          │
  │  [Proxy 1] [Proxy 2] [Proxy 3]                      │
  └──────────────────────┬──────────────────────────────┘
                         │
  ┌──────────────────────▼──────────────────────────────┐
  │              코디네이터 계층                            │
  │  ┌──────────┐ ┌──────────┐ ┌──────────┐            │
  │  │Root Coord│ │Query     │ │Data Coord│            │
  │  │(DDL, TSO)│ │Coord     │ │(세그먼트) │            │
  │  └──────────┘ └──────────┘ └──────────┘            │
  │                    ┌──────────┐                      │
  │                    │Index     │                      │
  │                    │Coord     │                      │
  │                    └──────────┘                      │
  └──────────────────────┬──────────────────────────────┘
                         │
  ┌──────────────────────▼──────────────────────────────┐
  │                워커 계층                               │
  │  [Query Node 1] [Query Node 2] (검색 실행)            │
  │  [Data Node 1]  [Data Node 2]  (쓰기/플러시)          │
  │  [Index Node 1] [Index Node 2] (인덱스 빌드)          │
  └──────────────────────┬──────────────────────────────┘
                         │
  ┌──────────────────────▼──────────────────────────────┐
  │              저장 계층                                 │
  │  ┌──────────┐ ┌───────────┐ ┌──────────┐           │
  │  │ etcd     │ │ MinIO/S3  │ │ Pulsar/  │           │
  │  │(메타데이터)│ │(세그먼트)  │ │ Kafka    │           │
  │  └──────────┘ └───────────┘ └──────────┘           │
  └─────────────────────────────────────────────────────┘
```

### 3.2 컬렉션, 스키마, 파티션

```python
from pymilvus import (
    connections, Collection, FieldSchema, CollectionSchema, DataType, utility
)

# Milvus 연결
connections.connect("default", host="localhost", port="19530")

# 타입 지정 필드로 스키마 정의
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=512),
    FieldSchema(name="category", dtype=DataType.VARCHAR, max_length=64),
    FieldSchema(name="published_year", dtype=DataType.INT32),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=768),
]

schema = CollectionSchema(fields, description="문서 임베딩")
collection = Collection("documents", schema)

# 데이터 격리를 위한 파티션 생성
collection.create_partition("technical_docs")
collection.create_partition("legal_docs")

# 벡터 필드에 인덱스 생성
index_params = {
    "metric_type": "L2",
    "index_type": "IVF_SQ8",      # 스칼라 양자화가 있는 IVF
    "params": {"nlist": 2048},
}
collection.create_index("embedding", index_params)

# 검색을 위해 컬렉션을 메모리에 로드
collection.load()

# 데이터 삽입
import numpy as np
data = [
    ["문서 A", "문서 B", "문서 C"],            # title
    ["technical", "legal", "technical"],        # category
    [2024, 2023, 2024],                         # published_year
    np.random.randn(3, 768).tolist(),           # embedding
]
collection.insert(data, partition_name="technical_docs")

# 메타데이터 필터와 함께 검색
search_params = {"metric_type": "L2", "params": {"nprobe": 64}}
results = collection.search(
    data=[np.random.randn(768).tolist()],
    anns_field="embedding",
    param=search_params,
    limit=10,
    expr='category == "technical" and published_year >= 2024',
    output_fields=["title", "category"],
)
```

### 3.3 일관성 수준

```
Milvus 일관성 수준:

  수준           보장                              사용 사례
  ──────────────────────────────────────────────────────────────
  Strong         모든 이전 쓰기를 읽음              금융 데이터, 정확한 카운트
  Bounded        T초 이내의 쓰기를 읽음             분석 (T=5초 허용 가능)
  Session        세션 내 자기 쓰기 읽기              대화형 애플리케이션
  Eventually     순서 보장 없음                     배치 검색, 추천

  검색별 설정:
    results = collection.search(..., consistency_level="Session")

  기본값: Bounded Staleness (대부분의 파이프라인에 좋은 균형)
```

---

## 4. Weaviate

### 4.1 모듈 아키텍처

```
Weaviate 모듈 시스템:

  ┌───────────────────────────────────────────┐
  │            Weaviate 코어                    │
  │  (HNSW 인덱스, 역인덱스, GraphQL)           │
  └─────────────────┬─────────────────────────┘
                    │
  ┌─────────────────▼─────────────────────────┐
  │              모듈 슬롯                      │
  │                                            │
  │  벡터라이저 모듈:                            │
  │  ├── text2vec-openai (OpenAI 임베딩)        │
  │  ├── text2vec-cohere (Cohere 임베딩)        │
  │  ├── text2vec-huggingface (로컬 모델)       │
  │  ├── img2vec-neural (이미지 임베딩)          │
  │  └── multi2vec-clip (멀티모달)              │
  │                                            │
  │  생성 모듈:                                 │
  │  ├── generative-openai (GPT 생성)          │
  │  ├── generative-cohere                     │
  │  └── generative-anthropic                  │
  │                                            │
  │  리랭커 모듈:                               │
  │  ├── reranker-cohere                       │
  │  └── reranker-transformers                 │
  └────────────────────────────────────────────┘
```

### 4.2 스키마와 GraphQL API

```python
import weaviate

client = weaviate.Client("http://localhost:8080")

# 클래스 정의 (컬렉션에 해당)
class_obj = {
    "class": "Article",
    "description": "임베딩이 있는 기술 문서",
    "vectorizer": "text2vec-openai",
    "moduleConfig": {
        "text2vec-openai": {
            "model": "text-embedding-3-small",
            "dimensions": 768,
        }
    },
    "properties": [
        {
            "name": "title",
            "dataType": ["text"],
            "moduleConfig": {
                "text2vec-openai": {"skip": False}  # 벡터화에 포함
            }
        },
        {
            "name": "content",
            "dataType": ["text"],
        },
        {
            "name": "category",
            "dataType": ["text"],
            "indexFilterable": True,   # 필터 검색 활성화
            "indexSearchable": True,   # BM25 검색 활성화
        },
        {
            "name": "publishedYear",
            "dataType": ["int"],
            "indexFilterable": True,
        },
    ],
}
client.schema.create_class(class_obj)

# 객체 추가 (Weaviate가 자동으로 벡터화)
client.data_object.create(
    class_name="Article",
    data_object={
        "title": "벡터 데이터베이스 소개",
        "content": "벡터 데이터베이스는 고차원 임베딩을 저장합니다...",
        "category": "databases",
        "publishedYear": 2024,
    }
)
```

### 4.3 Weaviate GraphQL 쿼리

```graphql
# 근접 텍스트 의미 검색 (Weaviate가 쿼리를 벡터화)
{
  Get {
    Article(
      nearText: { concepts: ["분산 벡터 인덱싱"] }
      where: {
        operator: And
        operands: [
          { path: ["category"], operator: Equal, valueText: "databases" }
          { path: ["publishedYear"], operator: GreaterThan, valueInt: 2023 }
        ]
      }
      limit: 10
    ) {
      title
      content
      category
      _additional {
        distance
        certainty
        id
      }
    }
  }
}

# 하이브리드 검색 (BM25 + 벡터, alpha로 가중치 조절)
{
  Get {
    Article(
      hybrid: { query: "HNSW 인덱스 성능", alpha: 0.7 }
      limit: 10
    ) {
      title
      _additional { score }
    }
  }
}
```

---

## 5. Pinecone과 Qdrant

### 5.1 Pinecone 서버리스

```python
from pinecone import Pinecone, ServerlessSpec

pc = Pinecone(api_key="your-api-key")

# 서버리스 인덱스 생성 (인프라 관리 불필요)
pc.create_index(
    name="articles",
    dimension=768,
    metric="cosine",
    spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1"
    )
)

index = pc.Index("articles")

# 메타데이터와 함께 업서트
vectors = [
    {
        "id": "doc-001",
        "values": [0.1, 0.2, ...],  # 768차원 임베딩
        "metadata": {
            "title": "벡터 인덱싱 가이드",
            "category": "technical",
            "year": 2024,
            "tags": ["vectors", "indexing", "HNSW"],
        }
    },
    # ... 더 많은 벡터
]
index.upsert(vectors=vectors, namespace="technical")

# 메타데이터 필터와 함께 쿼리
results = index.query(
    vector=[0.15, 0.22, ...],
    top_k=10,
    namespace="technical",
    filter={
        "$and": [
            {"category": {"$eq": "technical"}},
            {"year": {"$gte": 2024}},
            {"tags": {"$in": ["vectors"]}},
        ]
    },
    include_metadata=True,
)

# 희소-밀집 하이브리드 검색 (Pinecone 네이티브)
results = index.query(
    vector=[0.15, 0.22, ...],               # 밀집
    sparse_vector={                           # 희소 (BM25 유사)
        "indices": [102, 4501, 9832],
        "values": [0.8, 0.4, 0.6],
    },
    top_k=10,
)
```

### 5.2 Qdrant 고급 필터링

```python
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    Filter, FieldCondition, MatchValue, Range,
    SearchParams, QuantizationSearchParams
)

client = QdrantClient(host="localhost", port=6333)

# 양자화 설정과 함께 컬렉션 생성
client.create_collection(
    collection_name="articles",
    vectors_config=VectorParams(
        size=768,
        distance=Distance.COSINE,
        on_disk=True,  # 대규모 데이터셋을 위한 mmap 저장
    ),
    quantization_config={
        "scalar": {
            "type": "int8",
            "quantile": 0.99,
            "always_ram": True,  # 양자화된 벡터를 RAM에 유지
        }
    },
)

# 풍부한 페이로드와 함께 업서트
client.upsert(
    collection_name="articles",
    points=[
        PointStruct(
            id=1,
            vector=[0.1, 0.2, ...],
            payload={
                "title": "벡터 인덱싱 가이드",
                "category": "technical",
                "year": 2024,
                "author": {"name": "Alice", "org": "DataCo"},
                "tags": ["vectors", "indexing"],
            }
        ),
    ]
)

# must/should/must_not을 활용한 고급 필터링
results = client.search(
    collection_name="articles",
    query_vector=[0.15, 0.22, ...],
    query_filter=Filter(
        must=[
            FieldCondition(key="category", match=MatchValue(value="technical")),
            FieldCondition(key="year", range=Range(gte=2023)),
        ],
        must_not=[
            FieldCondition(key="tags", match=MatchValue(value="deprecated")),
        ],
        should=[  # 최소 하나는 일치해야 (점수 향상)
            FieldCondition(key="author.org", match=MatchValue(value="DataCo")),
        ],
    ),
    search_params=SearchParams(
        hnsw_ef=128,
        quantization=QuantizationSearchParams(
            rescore=True,       # 원본 벡터로 재점수화
            oversampling=2.0,   # 재점수화 전 2배 후보 가져오기
        ),
    ),
    limit=10,
)
```

### 5.3 Pinecone vs Qdrant 비교

```
기능                  Pinecone                    Qdrant
────────────────────────────────────────────────────────────────
호스팅                완전 관리형 (SaaS)           자체 호스팅 또는 클라우드
언어                  독점                        Rust (오픈소스)
확장                  자동 서버리스                수동 샤딩 + 복제
메타데이터 필터        JSON 필터 구문              must/should/must_not
하이브리드 검색        네이티브 희소-밀집           BM25 + 밀집 (내장)
양자화                자동                        스칼라, PQ (설정 가능)
가격                  읽기/쓰기 단위당 과금         무료 (자체 호스팅) 또는 클라우드
콜드 스타트            제로로 스케일               항상 실행
디스크 모드            N/A (관리형)                mmap 지원
멀티 테넌시            네임스페이스                 컬렉션 + 페이로드 필터
최대 차원              20,000                     65,535
```

---

## 6. Chroma

### 6.1 임베디드 모드 (프로토타이핑)

```python
import chromadb

# 임베디드 모드 — 인프로세스 실행, 로컬 디렉토리에 영속화
client = chromadb.PersistentClient(path="/data/chroma_db")

# 커스텀 임베딩 함수로 컬렉션 생성
from chromadb.utils import embedding_functions

openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key="your-key",
    model_name="text-embedding-3-small"
)

collection = client.get_or_create_collection(
    name="articles",
    embedding_function=openai_ef,
    metadata={"hnsw:space": "cosine"},  # 거리 메트릭
)

# 문서 추가 (Chroma가 자동으로 임베딩 생성)
collection.add(
    documents=[
        "벡터 데이터베이스와 유사도 검색 소개",
        "고차원 데이터를 위한 고급 인덱싱 기법",
        "벡터 검색 시스템의 프로덕션 배포 패턴",
    ],
    metadatas=[
        {"category": "intro", "year": 2024},
        {"category": "advanced", "year": 2024},
        {"category": "production", "year": 2025},
    ],
    ids=["doc-001", "doc-002", "doc-003"],
)

# 텍스트로 쿼리 (자동 임베딩)
results = collection.query(
    query_texts=["벡터 검색을 어떻게 확장하는가"],
    n_results=5,
    where={"year": {"$gte": 2024}},
    include=["documents", "metadatas", "distances"],
)

# 임베딩으로 직접 쿼리
results = collection.query(
    query_embeddings=[[0.1, 0.2, ...]],
    n_results=5,
)
```

### 6.2 클라이언트-서버 모드 (프로덕션)

```python
# 서버: 별도 프로세스로 실행
# chroma run --host 0.0.0.0 --port 8000 --path /data/chroma_db

# 클라이언트: HTTP로 연결
client = chromadb.HttpClient(host="chroma-server", port=8000)

# 임베디드 모드와 동일한 API
collection = client.get_or_create_collection("articles")
collection.add(documents=["..."], ids=["doc-004"])
```

```
Chroma 배포 모드:

  임베디드 (개발):
  ┌──────────────────────────┐
  │ 애플리케이션 프로세스       │
  │  ├── 앱 코드              │
  │  └── Chroma 라이브러리     │
  │       └── SQLite + HNSW   │
  │            └── /data/     │
  └──────────────────────────┘

  클라이언트-서버 (프로덕션):
  ┌──────────────┐    HTTP    ┌──────────────────┐
  │ 앱 프로세스    │──────────→│ Chroma 서버       │
  │ (씬 클라이언트)│           │  ├── HNSW 인덱스   │
  └──────────────┘            │  ├── SQLite 메타   │
                              │  └── /data/        │
                              └──────────────────┘

  분산형 (Chroma Cloud / 예정):
  ┌──────────────┐    gRPC    ┌──────────────────┐
  │ 앱 프로세스    │──────────→│ 코디네이터        │
  └──────────────┘            │  ├── 샤드 1        │
                              │  ├── 샤드 2        │
                              │  └── 샤드 3        │
                              └──────────────────┘
```

---

## 7. 저장소 영속화 패턴

### 7.1 데이터 파이프라인에서의 인덱스 수명주기

```
임베딩 파이프라인 통합:

  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
  │ 원시 데이터│───→│ 임베딩   │───→│ 인덱스   │───→│ 서빙    │
  │ (S3/Lake) │    │ 서비스    │    │ 빌더     │    │ 계층    │
  └──────────┘    └──────────┘    └──────────┘    └──────────┘
       │               │               │               │
  배치: 시간별     GPU 클러스터     FAISS 학습      로드 밸런싱된
  스트림: 실시간   또는 API 호출    + 양자화         쿼리 노드
  CDC                              + 직렬화

  도구별 영속화 전략:

  도구          주요 저장소        백업 전략
  ─────────────────────────────────────────────────────
  FAISS       faiss.write_index  S3/GCS 업로드 + 버전 태그
  Milvus      MinIO/S3 세그먼트   내장 스냅샷 API
  Weaviate    디스크 (LSMT)      백업 API → S3
  Qdrant      디스크 (세그먼트)   스냅샷 API → S3
  Pinecone    관리형             관리형 (SLA 보장)
  Chroma      SQLite + 파일     파일 수준 백업
```

### 7.2 스냅샷과 백업

```python
"""
프로덕션 배포를 위한 Qdrant 스냅샷 관리.
"""
from qdrant_client import QdrantClient

client = QdrantClient(host="localhost", port=6333)

# 스냅샷 생성 (일관된 시점 백업)
snapshot = client.create_snapshot(collection_name="articles")
# 반환: SnapshotDescription(name='articles-2024-06-15-12-00-00.snapshot')

# 스냅샷 목록
snapshots = client.list_snapshots(collection_name="articles")

# 오프사이트 백업을 위한 스냅샷 다운로드
client.download_snapshot(
    collection_name="articles",
    snapshot_name=snapshot.name,
    path="/backups/articles-latest.snapshot",
)

# 스냅샷에서 복원 (재해 복구)
# client.recover_snapshot(
#     collection_name="articles",
#     location="/backups/articles-latest.snapshot",
# )
```

```python
"""
데이터 파이프라인을 위한 FAISS 인덱스 버전 관리.
"""
import faiss
import json
from datetime import datetime
from pathlib import Path

def save_versioned_index(
    index: faiss.Index,
    base_path: str,
    metadata: dict,
) -> str:
    """버전 메타데이터와 함께 FAISS 인덱스를 저장합니다."""
    version = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = Path(base_path) / version

    path.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(path / "index.faiss"))

    metadata["version"] = version
    metadata["ntotal"] = index.ntotal
    metadata["d"] = index.d
    metadata["timestamp"] = datetime.now().isoformat()

    with open(path / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # 최신 심볼릭 링크 업데이트
    latest = Path(base_path) / "latest"
    if latest.exists():
        latest.unlink()
    latest.symlink_to(path)

    return version

def load_latest_index(base_path: str) -> tuple:
    """최신 버전의 인덱스를 로드합니다."""
    path = Path(base_path) / "latest"
    index = faiss.read_index(str(path / "index.faiss"))
    with open(path / "metadata.json") as f:
        metadata = json.load(f)
    return index, metadata
```

---

## 8. 벤치마크 비교

### 8.1 성능 비교 테이블

```
벤치마크: 100만 벡터, 768차원, top-10, 단일 노드

데이터베이스    인덱스 유형    QPS (p50)  지연시간 p99  Recall@10  메모리
────────────────────────────────────────────────────────────────────────
FAISS          IVF4096,PQ96  8,500      1.2ms        0.92       1.2 GB
FAISS          HNSW32        3,200      2.8ms        0.99       35 GB
Milvus         IVF_SQ8       2,800      4.1ms        0.95       12 GB
Milvus         HNSW          2,100      5.5ms        0.98       36 GB
Weaviate       HNSW+PQ       1,800      6.2ms        0.94       8 GB
Qdrant         HNSW+SQ8      3,100      3.5ms        0.97       14 GB
Pinecone       관리형         2,500      8.0ms        0.96       N/A
Chroma         HNSW          900        12.0ms       0.97       35 GB

참고:
- FAISS 수치는 네트워크 오버헤드 제외 (라이브러리, 서버 아님)
- Pinecone 지연시간은 네트워크 왕복 포함
- 모든 데이터베이스는 ~95%+ 재현율 목표로 튜닝
- 메모리는 인덱스 + 벡터 + 메타데이터 오버헤드 포함
```

### 8.2 기능 비교 매트릭스

```
기능                  FAISS   Milvus   Weaviate  Qdrant   Pinecone  Chroma
──────────────────────────────────────────────────────────────────────────────
유형                  라이브러리 DB      DB        DB       SaaS      DB
관리형 호스팅          ✗       Zilliz   WCS       Qdrant   ✓ 전용    ✗
오픈소스              ✓       ✓        ✓         ✓        ✗         ✓
메타데이터 필터링      ✗       ✓        ✓         ✓        ✓         ✓
하이브리드 검색        ✗       ✓        ✓         ✓        ✓         ✗
멀티 테넌시           ✗       ✓        ✓         ✓        ✓         ✓
자동 벡터화           ✗       ✗        ✓         ✗        ✓(추론)   ✓
GraphQL API          ✗       ✗        ✓         ✗        ✗         ✗
GPU 지원             ✓       ✓        ✗         ✗        N/A       ✗
최대 테스트 규모       1B+     10B+     100M      100M     1B+       1M
디스크 기반 인덱스     ✓       ✓        ✓         ✓        N/A       ✗
복제                  ✗       ✓        ✓         ✓        ✓         ✗
스냅샷/백업           ✗       ✓        ✓         ✓        ✓         수동
```

### 8.3 선택 가이드

```
언제 어떤 것을 사용할까:

  FAISS:
    - 최대 처리량과 제어가 필요할 때
    - 커스텀 검색 엔진을 구축하거나 다른 시스템에 내장할 때
    - GPU 가속이 필요할 때
    - 메타데이터 필터링이 필요 없을 때

  Milvus:
    - 수십억 규모의 데이터셋
    - 분산형 장애 허용 벡터 검색이 필요할 때
    - 복합 필터링 + 벡터 검색
    - Zilliz Cloud 옵션의 엔터프라이즈 배포

  Weaviate:
    - 자동 벡터화를 원할 때 (별도 임베딩 파이프라인 불필요)
    - GraphQL API가 스택에 맞을 때
    - 생성형 검색 (RAG 내장)이 필요할 때
    - 모듈 생태계가 중요할 때

  Qdrant:
    - 최고의 단일 노드 성능을 원할 때
    - 고급 필터링 (must/should/must_not)이 필요할 때
    - Rust 성능 + Python 간편함
    - 낮은 운영 오버헤드로 자체 호스팅

  Pinecone:
    - 인프라 관리 제로
    - 서버리스 스케일링이 필요할 때 (제로 스케일)
    - 예산이 관리형 가격을 허용할 때
    - 빠른 프로덕션 투입 시간

  Chroma:
    - 프로토타이핑과 개발
    - 소규모 프로덕션 (<100만 벡터)
    - 임베디드 모드 (서버 없음)를 원할 때
    - LangChain/LlamaIndex 통합
```

---

## 9. 데이터 엔지니어링 파이프라인과의 통합

### 9.1 Airflow를 활용한 배치 임베딩 파이프라인

```python
"""
배치 벡터 인덱스 업데이트를 위한 Airflow DAG.
벡터 저장소가 데이터 엔지니어링 파이프라인에 어떻게 맞는지 보여줍니다.
"""

from airflow.decorators import dag, task
from datetime import datetime

@dag(
    schedule="@daily",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["vector", "embeddings"],
)
def vector_index_update():

    @task()
    def extract_new_documents(ds=None):
        """마지막 실행 이후 수정된 문서를 추출합니다."""
        # 새/업데이트된 문서를 위해 데이터 웨어하우스 쿼리
        # 실제: Spark SQL, dbt 모델, 또는 직접 DB 쿼리
        return {
            "count": 5000,
            "source": f"s3://data-lake/gold/documents/dt={ds}/",
        }

    @task()
    def generate_embeddings(doc_info: dict):
        """모델 API 또는 로컬 모델을 사용하여 임베딩을 생성합니다."""
        # 배치 임베딩 생성
        # 실제: OpenAI API 호출, GPU에서 로컬 모델 실행
        return {
            "embeddings_path": "s3://embeddings/batch/2024-06-15/",
            "count": doc_info["count"],
            "model": "text-embedding-3-small",
            "dimension": 768,
        }

    @task()
    def upsert_to_vector_db(embedding_info: dict):
        """벡터 데이터베이스에 임베딩을 업서트합니다."""
        # 실제: Milvus/Qdrant/Pinecone에 배치 업서트
        return {
            "upserted": embedding_info["count"],
            "collection": "documents",
        }

    @task()
    def validate_index(upsert_result: dict):
        """업데이트된 인덱스에 검증 검사를 실행합니다."""
        # 검사: 총 개수, 샘플 쿼리 재현율, 지연시간
        checks = {
            "total_vectors": 1_250_000,
            "sample_recall": 0.96,
            "p99_latency_ms": 4.2,
            "status": "passed",
        }
        return checks

    docs = extract_new_documents()
    embeddings = generate_embeddings(docs)
    result = upsert_to_vector_db(embeddings)
    validate_index(result)

vector_index_update()
```

---

## 요약

```
핵심 요점:

1. 저장소 아키텍처 선택(인메모리/디스크/분산)은
   데이터셋 크기, 지연시간 요구사항, 예산에 따라 결정

2. FAISS는 기초 라이브러리 — 인덱스 유형
   (Flat, IVF, HNSW, PQ)과 조합을 위한 팩토리 문자열을 이해해야 함

3. Milvus는 강력한 일관성 옵션을 가진 수십억 규모
   분산 배포에 탁월

4. Weaviate의 모듈 시스템은 자동 벡터화와
   내장 RAG 기능을 지원

5. Pinecone은 제로 운영 서버리스를 제공;
   Qdrant은 최고의 자체 호스팅 단일 노드 성능을 제공

6. Chroma는 프로토타이핑에 이상적이나 확장에 한계가 있음

7. 벡터 저장소는 데이터 파이프라인과 통합되어야 함 — 배치
   임베딩 업데이트, 인덱스 버전 관리, 모니터링은
   데이터 엔지니어의 책임
```

---

## 연습 문제

1. **FAISS 인덱스 비교**: 10만 랜덤 벡터에 Flat, IVF, HNSW, IVFPQ 인덱스를 구축하세요. 각각의 검색 시간, 재현율, 메모리를 측정하세요.

2. **Milvus 컬렉션 설계**: 카테고리 필터링과 가격 범위 쿼리가 있는 전자상거래 제품 검색 시스템을 위한 Milvus 스키마를 설계하세요.

3. **Qdrant vs Chroma**: Qdrant과 Chroma 모두를 사용하여 같은 검색 애플리케이션을 구현하세요. API 사용성과 성능을 비교하세요.

4. **인덱스 버전 관리 파이프라인**: 버전 메타데이터와 함께 FAISS 인덱스를 저장하고 롤백을 구현하는 Python 스크립트를 작성하세요.

5. **벤치마크 실행기**: 주어진 인덱스 설정의 QPS, 지연시간 백분위수, 재현율을 측정하는 벤치마크 스크립트를 작성하세요.

---

## 더 읽을거리

- [FAISS Wiki — 인덱스 선택 가이드라인](https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index)
- [Milvus 문서](https://milvus.io/docs)
- [Weaviate 문서](https://weaviate.io/developers/weaviate)
- [Qdrant 문서](https://qdrant.tech/documentation/)
- [Pinecone 문서](https://docs.pinecone.io/)
- [Chroma 문서](https://docs.trychroma.com/)

[← 이전: 21. 데이터 버전 관리와 데이터 계약](21_Data_Versioning_and_Contracts.md) | [다음: 23. 프로덕션 벡터 검색 →](23_Production_Vector_Search.md)
