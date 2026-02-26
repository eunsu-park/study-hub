# 11. 벡터 데이터베이스

## 학습 목표

- 벡터 데이터베이스 개념
- Chroma, FAISS, Pinecone 사용
- 인덱싱과 검색 최적화
- 실전 활용 패턴

---

## 1. 벡터 데이터베이스 개요

### 왜 벡터 DB인가?

```
전통 DB:
    SELECT * FROM docs WHERE text LIKE '%machine learning%'
    → 키워드 매칭만 가능

벡터 DB:
    query_vector = embed("What is AI?")
    SELECT * FROM docs ORDER BY similarity(vector, query_vector)
    → 의미적 유사성 검색
```

### 주요 벡터 DB

| 이름 | 타입 | 특징 |
|------|------|------|
| Chroma | 로컬/임베디드 | 간단, 개발용 |
| FAISS | 라이브러리 | 빠름, 대규모 |
| Pinecone | 클라우드 | 관리형, 확장성 |
| Weaviate | 오픈소스 | 하이브리드 검색 |
| Qdrant | 오픈소스 | 필터링 강점 |
| Milvus | 오픈소스 | 대규모, 분산 |

---

## 2. Chroma

### 설치 및 기본 사용

```python
pip install chromadb
```

```python
import chromadb
from chromadb.utils import embedding_functions

# 클라이언트 생성
client = chromadb.Client()  # 메모리
# client = chromadb.PersistentClient(path="./chroma_db")  # 영구 저장

# 임베딩 함수
embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

# 컬렉션 생성
collection = client.create_collection(
    name="my_collection",
    embedding_function=embedding_fn
)
```

### 문서 추가

```python
# 문서 추가
collection.add(
    documents=["Document 1 text", "Document 2 text", "Document 3 text"],
    metadatas=[{"source": "a"}, {"source": "b"}, {"source": "a"}],
    ids=["doc1", "doc2", "doc3"]
)

# 임베딩 직접 제공
collection.add(
    embeddings=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
    documents=["Doc 1", "Doc 2"],
    ids=["id1", "id2"]
)
```

### 검색

```python
# 쿼리 검색
results = collection.query(
    query_texts=["What is machine learning?"],
    n_results=3
)

print(results['documents'])  # 문서 내용
print(results['distances'])  # 거리
print(results['metadatas'])  # 메타데이터

# 메타데이터 필터링
results = collection.query(
    query_texts=["query"],
    n_results=5,
    where={"source": "a"}  # source가 "a"인 것만
)

# 복합 필터
results = collection.query(
    query_texts=["query"],
    where={"$and": [{"source": "a"}, {"year": {"$gt": 2020}}]}
)
```

### LangChain 연동

```python
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()

# 생성
vectorstore = Chroma.from_texts(
    texts=["text1", "text2", "text3"],
    embedding=embeddings,
    persist_directory="./chroma_db"
)

# 검색
docs = vectorstore.similarity_search("query", k=3)

# Retriever로 사용
retriever = vectorstore.as_retriever()
```

---

## 3. FAISS

### 설치 및 기본 사용

```python
pip install faiss-cpu  # CPU 버전
# pip install faiss-gpu  # GPU 버전
```

```python
import faiss
import numpy as np

# 인덱스 생성
dimension = 384
index = faiss.IndexFlatL2(dimension)  # L2 거리

# 벡터 추가
vectors = np.random.random((1000, dimension)).astype('float32')
index.add(vectors)

print(f"Total vectors: {index.ntotal}")
```

### 검색

```python
# 검색
query = np.random.random((1, dimension)).astype('float32')
distances, indices = index.search(query, k=5)

print(f"Indices: {indices}")
print(f"Distances: {distances}")
```

### 인덱스 타입

```python
# Flat (정확, 느림)
index = faiss.IndexFlatL2(dimension)

# IVF (근사, 빠름)
quantizer = faiss.IndexFlatL2(dimension)
index = faiss.IndexIVFFlat(quantizer, dimension, nlist=100)
index.train(vectors)  # 학습 필요
index.add(vectors)
index.nprobe = 10  # 검색할 클러스터 수

# HNSW (매우 빠름)
index = faiss.IndexHNSWFlat(dimension, 32)  # 32 = M parameter
index.add(vectors)

# PQ (메모리 효율)
index = faiss.IndexPQ(dimension, 8, 8)  # M=8, nbits=8
index.train(vectors)
index.add(vectors)
```

### 저장/로드

```python
# 저장
faiss.write_index(index, "index.faiss")

# 로드
index = faiss.read_index("index.faiss")
```

### LangChain 연동

```python
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()

# 생성
vectorstore = FAISS.from_texts(
    texts=["text1", "text2"],
    embedding=embeddings
)

# 저장/로드
vectorstore.save_local("faiss_index")
vectorstore = FAISS.load_local("faiss_index", embeddings)
```

---

## 4. Pinecone

### 설치 및 설정

```python
pip install pinecone-client
```

```python
from pinecone import Pinecone, ServerlessSpec

# 클라이언트 생성
pc = Pinecone(api_key="your-api-key")

# 인덱스 생성
pc.create_index(
    name="my-index",
    dimension=384,
    metric="cosine",
    spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1"
    )
)

# 인덱스 연결
index = pc.Index("my-index")
```

### 문서 추가

```python
# Upsert (추가/업데이트)
index.upsert(
    vectors=[
        {"id": "vec1", "values": [0.1, 0.2, ...], "metadata": {"source": "a"}},
        {"id": "vec2", "values": [0.3, 0.4, ...], "metadata": {"source": "b"}},
    ]
)

# 배치 upsert
from itertools import islice

def chunks(iterable, batch_size=100):
    it = iter(iterable)
    chunk = list(islice(it, batch_size))
    while chunk:
        yield chunk
        chunk = list(islice(it, batch_size))

for batch in chunks(vectors, batch_size=100):
    index.upsert(vectors=batch)
```

### 검색

```python
# 쿼리
results = index.query(
    vector=[0.1, 0.2, ...],
    top_k=5,
    include_metadata=True
)

for match in results['matches']:
    print(f"ID: {match['id']}, Score: {match['score']}")
    print(f"Metadata: {match['metadata']}")

# 메타데이터 필터링
results = index.query(
    vector=[0.1, 0.2, ...],
    top_k=5,
    filter={"source": {"$eq": "a"}}
)
```

### LangChain 연동

```python
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()

vectorstore = PineconeVectorStore.from_texts(
    texts=["text1", "text2"],
    embedding=embeddings,
    index_name="my-index"
)

# 검색
docs = vectorstore.similarity_search("query", k=3)
```

---

## 5. 인덱싱 전략

### 인덱스 타입 비교

| 타입 | 정확도 | 속도 | 메모리 | 사용 시점 |
|------|--------|------|--------|----------|
| Flat | 100% | 느림 | 높음 | 소규모 (<100K) |
| IVF | 95%+ | 빠름 | 중간 | 중규모 |
| HNSW | 98%+ | 매우 빠름 | 높음 | 대규모, 실시간 |
| PQ | 90%+ | 빠름 | 낮음 | 메모리 제한 |

### 하이브리드 인덱스

```python
# IVF + PQ
quantizer = faiss.IndexFlatL2(dimension)
index = faiss.IndexIVFPQ(
    quantizer,
    dimension,
    nlist=100,   # 클러스터 수
    m=8,         # PQ 세그먼트 수
    nbits=8      # PQ 비트 수
)
index.train(vectors)
index.add(vectors)
```

---

## 6. 메타데이터 활용

### 필터링 패턴

```python
# Chroma 필터 문법
results = collection.query(
    query_texts=["query"],
    where={
        "$and": [
            {"category": "tech"},
            {"year": {"$gte": 2023}},
            {"author": {"$in": ["Alice", "Bob"]}}
        ]
    }
)

# 지원 연산자
# $eq, $ne: 같음, 다름
# $gt, $gte, $lt, $lte: 비교
# $in, $nin: 포함, 미포함
# $and, $or: 논리 연산
```

### 메타데이터 업데이트

```python
# Chroma
collection.update(
    ids=["doc1"],
    metadatas=[{"source": "updated"}]
)

# Pinecone
index.update(
    id="vec1",
    set_metadata={"source": "updated"}
)
```

---

## 7. 실전 패턴

### 문서 관리 클래스

```python
class VectorStore:
    def __init__(self, persist_dir="./db"):
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction()
        self.collection = self.client.get_or_create_collection(
            name="documents",
            embedding_function=self.embedding_fn
        )

    def add_documents(self, texts, metadatas=None, ids=None):
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in texts]
        self.collection.add(
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )
        return ids

    def search(self, query, k=5, where=None):
        results = self.collection.query(
            query_texts=[query],
            n_results=k,
            where=where
        )
        return results

    def delete(self, ids):
        self.collection.delete(ids=ids)
```

### 증분 업데이트

```python
import hashlib

def get_doc_id(text):
    return hashlib.md5(text.encode()).hexdigest()

def upsert_documents(texts, collection):
    """중복 방지 업서트"""
    ids = [get_doc_id(t) for t in texts]

    # 기존 문서 확인
    existing = collection.get(ids=ids)
    existing_ids = set(existing['ids'])

    # 새 문서만 추가
    new_texts = []
    new_ids = []
    for text, doc_id in zip(texts, ids):
        if doc_id not in existing_ids:
            new_texts.append(text)
            new_ids.append(doc_id)

    if new_texts:
        collection.add(documents=new_texts, ids=new_ids)

    return len(new_texts)
```

### 배치 처리

```python
def batch_add(collection, texts, batch_size=100):
    """대량 문서 배치 추가"""
    total = len(texts)
    for i in range(0, total, batch_size):
        batch = texts[i:i + batch_size]
        ids = [str(uuid.uuid4()) for _ in batch]
        collection.add(documents=batch, ids=ids)
        print(f"Added {min(i + batch_size, total)}/{total}")
```

---

## 8. 성능 최적화

### 임베딩 캐싱

```python
import pickle
import os

class CachedEmbeddings:
    def __init__(self, model, cache_dir="./embed_cache"):
        self.model = model
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def embed(self, text):
        cache_key = hashlib.md5(text.encode()).hexdigest()
        cache_path = os.path.join(self.cache_dir, f"{cache_key}.pkl")

        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                return pickle.load(f)

        embedding = self.model.encode(text)

        with open(cache_path, 'wb') as f:
            pickle.dump(embedding, f)

        return embedding
```

### 인덱스 최적화

```python
# FAISS 검색 파라미터 튜닝
index.nprobe = 20  # 더 많은 클러스터 검색 (정확도 ↑, 속도 ↓)

# 병렬 검색
faiss.omp_set_num_threads(4)  # 스레드 수 설정
```

---

## 정리

### 선택 가이드

| 상황 | 추천 |
|------|------|
| 개발/프로토타입 | Chroma |
| 대규모 로컬 | FAISS |
| 프로덕션 관리형 | Pinecone |
| 오픈소스 셀프호스트 | Qdrant, Milvus |

### 핵심 코드

```python
# Chroma
collection = client.create_collection("name")
collection.add(documents=texts, ids=ids)
results = collection.query(query_texts=["query"], n_results=5)

# FAISS
index = faiss.IndexFlatL2(dimension)
index.add(vectors)
distances, indices = index.search(query, k=5)

# LangChain
vectorstore = Chroma.from_texts(texts, embeddings)
docs = vectorstore.similarity_search("query", k=3)
```

---

## 연습 문제

### 연습 문제 1: FAISS 인덱스(Index) 유형 선택

벡터 검색 시스템을 구축 중이며 다음과 같은 제약 조건이 있습니다. 각 시나리오에 적합한 FAISS 인덱스 유형을 선택하고 이유를 설명하세요.

| 시나리오 | 데이터 크기 | 제약 조건 | 최적 인덱스 유형? |
|---------|-----------|-----------|----------------|
| A. 의료 기록 검색 | 50,000 벡터 | 100% 정확도 필요 | ? |
| B. 실시간 상품 검색 | 천만 벡터 | 50ms 이하 지연 필요 | ? |
| C. 모바일 앱 임베딩 검색 | 500,000 벡터 | 메모리 500MB 제한 | ? |
| D. 야간 배치 추천 | 200만 벡터 | 정확도 95%+, 훈련 시간 허용 | ? |

<details>
<summary>정답 보기</summary>

| 시나리오 | 최적 인덱스 | 이유 |
|---------|-----------|------|
| A. 의료 기록 (50K, 100% 정확도) | **IndexFlatL2** | 근사 없는 정확 검색. 50K × 384차원 × 4바이트 ≈ 73MB — 충분히 관리 가능. 의료 결정에는 정밀도가 필요. |
| B. 실시간 상품 검색 (1000만, 50ms 이하) | **IndexHNSWFlat** | 밀리초 지연으로 98%+ 정확도. 훈련 불필요. 실시간 서빙에 최적의 재현율/지연 트레이드오프(trade-off). |
| C. 모바일 앱 (50만, 500MB 제한) | **IndexIVFPQ** | PQ가 벡터를 1536바이트에서 약 64바이트로 압축 (24배). 50만 × 64바이트 ≈ 32MB, 예산 내. |
| D. 배치 추천 (200만, 95%+) | **IndexIVFFlat** | 빠른 근사 검색으로 좋은 정확도. 훈련은 일회성 비용. `nprobe`로 정확도/속도 조정 가능. |

```python
import faiss
import numpy as np

dimension = 384
vectors = np.random.random((50000, dimension)).astype('float32')

# 시나리오 A: 정확한 플랫(flat) 인덱스
index_a = faiss.IndexFlatL2(dimension)
index_a.add(vectors)

# 시나리오 B: HNSW (빠른 그래프 탐색을 위한 높은 연결성)
index_b = faiss.IndexHNSWFlat(dimension, 32)  # M=32: 높을수록 재현율↑, 메모리↑
index_b.add(vectors[:50000])  # 훈련 불필요

# 시나리오 C: 메모리 효율을 위한 IVF + PQ
quantizer = faiss.IndexFlatL2(dimension)
index_c = faiss.IndexIVFPQ(quantizer, dimension, nlist=1000, m=8, nbits=8)
# m=8: 8개 서브 양자화기, nbits=8: 각 256개 센트로이드 → 벡터당 8바이트
index_c.train(vectors)
index_c.add(vectors)
print(f"시나리오 C 메모리: ~{50000 * 8 / 1e6:.1f}MB (플랫 대비 {50000 * dimension * 4 / 1e6:.0f}MB)")

# 시나리오 D: 조정 가능한 정확도를 위한 IVF
quantizer_d = faiss.IndexFlatL2(dimension)
index_d = faiss.IndexIVFFlat(quantizer_d, dimension, nlist=2000)
index_d.train(vectors)
index_d.add(vectors)
index_d.nprobe = 50  # 2000개 클러스터 중 50개 검색 (~2.5%); 높일수록 재현율 향상
```
</details>

---

### 연습 문제 2: Chroma 메타데이터 필터링

문서 컬렉션(collection)에 연구 논문이 다음 메타데이터와 함께 저장되어 있습니다: `year`(int), `category`(str: "ml", "nlp", "cv"), `citations`(int). 아래 요구사항에 대한 Chroma 쿼리(query)를 작성하세요.

1. 2022년 이후 "nlp" 카테고리의 논문 검색
2. 인용 수 100회 초과인 "ml" 또는 "cv" 카테고리 논문 검색
3. 최근 3년(2023-2025) 논문 중 "cv" 카테고리가 아닌 논문 검색

<details>
<summary>정답 보기</summary>

```python
import chromadb
from chromadb.utils import embedding_functions

client = chromadb.Client()
embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)
collection = client.create_collection("papers", embedding_function=embedding_fn)

# 샘플 데이터
collection.add(
    documents=["Attention is all you need", "BERT pre-training", "ResNet deep residual"],
    metadatas=[
        {"year": 2017, "category": "nlp", "citations": 50000},
        {"year": 2018, "category": "nlp", "citations": 30000},
        {"year": 2016, "category": "cv", "citations": 80000},
    ],
    ids=["p1", "p2", "p3"]
)

# 쿼리 1: 2022년 이후 NLP 논문
results_1 = collection.query(
    query_texts=["트랜스포머 아키텍처"],
    n_results=5,
    where={
        "$and": [
            {"year": {"$gte": 2022}},
            {"category": {"$eq": "nlp"}}
        ]
    }
)

# 쿼리 2: 인용 수 100회 초과인 ML 또는 CV 논문
results_2 = collection.query(
    query_texts=["신경망"],
    n_results=5,
    where={
        "$and": [
            {"category": {"$in": ["ml", "cv"]}},
            {"citations": {"$gt": 100}}
        ]
    }
)

# 쿼리 3: 2023-2025년, CV 제외
results_3 = collection.query(
    query_texts=["딥러닝"],
    n_results=10,
    where={
        "$and": [
            {"year": {"$gte": 2023}},
            {"year": {"$lte": 2025}},
            {"category": {"$ne": "cv"}}
        ]
    }
)

# Chroma 필터 연산자 참고:
# $eq, $ne: 같음, 다름
# $gt, $gte, $lt, $lte: 숫자 비교
# $in, $nin: 목록 포함 여부
# $and, $or: 논리 조합
```

**흔한 실수:** Chroma의 `$and`/`$or`는 리스트(list)를 받으며, 모든 조건이 동일한 중첩 레벨에 있어야 합니다. 같은 `where` 절에 리스트 레벨과 딕셔너리(dict) 레벨 연산자를 혼용할 수 없습니다.
</details>

---

### 연습 문제 3: 콘텐츠 해싱으로 중복 제거

`upsert_documents` 함수를 확장하여 문서 업데이트도 처리하세요: 동일한 ID를 가진 문서가 이미 존재하지만 내용이 다른 경우 업데이트하고, 동일한 경우 건너뛰도록 해야 합니다.

```python
import hashlib

def get_doc_id(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()

# 현재 구현 (추가 전용 중복 제거)
def upsert_documents(texts, collection):
    ids = [get_doc_id(t) for t in texts]
    existing = collection.get(ids=ids)
    existing_ids = set(existing['ids'])

    new_texts = [t for t, id_ in zip(texts, ids) if id_ not in existing_ids]
    new_ids = [id_ for id_, t in zip(ids, texts) if id_ not in existing_ids]

    if new_texts:
        collection.add(documents=new_texts, ids=new_ids)
    return len(new_texts)
```

<details>
<summary>정답 보기</summary>

핵심 통찰: 콘텐츠 기반 ID(MD5 해시)는 동일한 문서에 항상 같은 ID를 생성합니다. 따라서 변경된 문서의 "업데이트"는 이전 내용 ID `old_hash`와 새 내용 ID `new_hash`가 다른 항목임을 의미합니다.

```python
import hashlib
from typing import Optional

def get_content_hash(text: str) -> str:
    """콘텐츠에서 안정적인 ID를 생성합니다."""
    return hashlib.md5(text.encode()).hexdigest()

def smart_upsert(
    texts: list[str],
    doc_keys: list[str],  # 논리적 ID (예: "doc_001", "doc_002")
    collection,
    metadatas: Optional[list[dict]] = None
) -> dict:
    """
    변경 감지 기능이 있는 스마트 업서트(upsert).

    전략: 메타데이터에 논리적 키와 콘텐츠 해시를 모두 저장.
    재인덱싱 시 콘텐츠 해시가 변경되었는지 확인.

    반환: {"added": N, "updated": N, "skipped": N}
    """
    stats = {"added": 0, "updated": 0, "skipped": 0}

    for i, (text, key) in enumerate(zip(texts, doc_keys)):
        new_hash = get_content_hash(text)
        meta = metadatas[i] if metadatas else {}
        meta["doc_key"] = key
        meta["content_hash"] = new_hash

        # 이 논리적 키가 이미 존재하는지 확인 (메타데이터로 조회)
        existing = collection.get(where={"doc_key": {"$eq": key}})

        if not existing["ids"]:
            # 새 문서
            collection.add(
                documents=[text],
                ids=[new_hash],
                metadatas=[meta]
            )
            stats["added"] += 1

        elif existing["metadatas"][0]["content_hash"] == new_hash:
            # 동일한 내용 — 건너뜀
            stats["skipped"] += 1

        else:
            # 내용 변경 — 이전 것 삭제, 새 것 추가
            collection.delete(ids=existing["ids"])
            collection.add(
                documents=[text],
                ids=[new_hash],
                metadatas=[meta]
            )
            stats["updated"] += 1

    return stats

# 테스트
import chromadb
client = chromadb.Client()
coll = client.create_collection("docs")

result = smart_upsert(
    texts=["문서 A 버전 1", "문서 B 버전 1"],
    doc_keys=["doc_A", "doc_B"],
    collection=coll
)
print(result)  # {"added": 2, "updated": 0, "skipped": 0}

result = smart_upsert(
    texts=["문서 A 버전 2", "문서 B 버전 1"],  # A 변경, B 동일
    doc_keys=["doc_A", "doc_B"],
    collection=coll
)
print(result)  # {"added": 0, "updated": 1, "skipped": 1}
```

**콘텐츠 해시 ID가 중요한 이유:** 순차적 ID를 사용하면 매번 업서트마다 문서 텍스트를 비교해야 합니다. 콘텐츠 해시를 사용하면 변경되지 않은 문서는 항상 같은 ID를 생성하므로 텍스트 비교가 필요 없습니다.
</details>

---

## 다음 단계

[실전 챗봇 프로젝트](./12_Practical_Chatbot.md)에서 대화형 AI 시스템을 구축합니다.
