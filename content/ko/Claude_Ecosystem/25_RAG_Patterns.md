# RAG 패턴

**이전**: [24. 프롬프트 캐싱과 Batch API](./24_Prompt_Caching_and_Batch_API.md)

---

검색 증강 생성(Retrieval-Augmented Generation, RAG)은 쿼리 시점에 관련 정보를 동적으로 검색하여 Claude의 기능을 학습 데이터 너머로 확장합니다. 이 레슨에서는 기본 문서 검색부터 고급 다단계 아키텍처까지 RAG 패턴의 전체 스펙트럼을 다루며, Claude의 고유한 강점인 200K 컨텍스트 윈도우, 컨텍스트 검색(Contextual Retrieval), MCP 서버와의 원활한 통합에 초점을 맞춘 실용적 구현을 제공합니다.

**난이도**: ⭐⭐⭐⭐

**사전 요구 사항**:
- Claude API 기초 ([레슨 15](./15_Claude_API_Fundamentals.md))
- 도구 사용과 함수 호출(Tool Use & Function Calling) ([레슨 16](./16_Tool_Use_and_Function_Calling.md))
- Model Context Protocol 기본 ([레슨 12](./12_Model_Context_Protocol.md))
- 프롬프트 캐싱 ([레슨 24](./24_Prompt_Caching_and_Batch_API.md))

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. Claude를 사용한 엔드투엔드 RAG 파이프라인 설계 및 구현
2. 다양한 콘텐츠 유형에 적합한 문서 청킹(Chunking) 전략 선택
3. 임베딩(Embedding)과 BM25를 사용한 하이브리드 검색 구현
4. 긴 컨텍스트(Long-context)와 RAG 접근법 간의 정보에 기반한 결정
5. 검색 정확도를 높이는 컨텍스트 검색(Contextual Retrieval) 파이프라인 구축
6. 검증 가능한 출력을 위한 인용(Citation) 및 근거 확보(Grounding) 패턴 구현
7. 복잡한 쿼리를 위한 다단계 RAG 아키텍처 설계
8. 적절한 메트릭을 사용한 RAG 시스템 평가
9. MCP 통합을 통한 프로덕션 RAG 시스템 구축

---

## 목차

1. [RAG 기초](#1-rag-기초)
2. [문서 청킹 전략](#2-문서-청킹-전략)
3. [임베딩 모델과 벡터 데이터베이스](#3-임베딩-모델과-벡터-데이터베이스)
4. [긴 컨텍스트 vs RAG 트레이드오프](#4-긴-컨텍스트-vs-rag-트레이드오프)
5. [컨텍스트 검색](#5-컨텍스트-검색)
6. [인용과 근거 확보 패턴](#6-인용과-근거-확보-패턴)
7. [다단계 RAG](#7-다단계-rag)
8. [RAG 평가 메트릭](#8-rag-평가-메트릭)
9. [MCP를 활용한 프로덕션 RAG](#9-mcp를-활용한-프로덕션-rag)
10. [연습 문제](#10-연습-문제)

---

## 1. RAG 기초

### 1.1 RAG란?

RAG는 동적으로 검색된 컨텍스트로 LLM의 생성을 증강하는 패턴입니다. 학습 데이터에만 의존하는 대신, 모델이 쿼리 시점에 관련 문서를 받습니다:

```
사용자 쿼리 → 검색기(Retriever) → 관련 문서 → Claude + 문서 → 답변
```

### 1.2 왜 RAG인가?

| 문제 | RAG가 어떻게 도움이 되는가 |
|---|---|
| 지식 컷오프(Knowledge Cutoff) | 최신 정보를 검색 |
| 환각(Hallucination) | 소스 문서에 답변을 근거함 |
| 도메인 특수성 | 파인튜닝 없이 도메인 지식 주입 |
| 데이터 프라이버시 | 민감한 데이터를 자체 인프라에 유지 |
| 확장성 | 어떤 크기의 지식 베이스에서도 작동 |

### 1.3 기본 RAG 파이프라인

```python
import anthropic
import numpy as np
from dataclasses import dataclass


@dataclass
class Document:
    id: str
    content: str
    metadata: dict
    embedding: list[float] | None = None


class SimpleRAG:
    """A minimal RAG pipeline for demonstration."""

    def __init__(self, embedding_fn):
        self.client = anthropic.Anthropic()
        self.embedding_fn = embedding_fn
        self.documents: list[Document] = []

    def index(self, documents: list[Document]):
        """Index documents by computing embeddings."""
        for doc in documents:
            doc.embedding = self.embedding_fn(doc.content)
            self.documents.append(doc)

    def retrieve(self, query: str, top_k: int = 5) -> list[Document]:
        """Retrieve the most relevant documents for a query."""
        query_embedding = self.embedding_fn(query)

        scored = []
        for doc in self.documents:
            score = np.dot(query_embedding, doc.embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(doc.embedding)
            )
            scored.append((score, doc))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scored[:top_k]]

    def generate(self, query: str, top_k: int = 5) -> str:
        """Retrieve relevant docs and generate an answer."""
        docs = self.retrieve(query, top_k)

        context = "\n\n---\n\n".join(
            f"[Source: {doc.id}]\n{doc.content}" for doc in docs
        )

        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2048,
            system=(
                "Answer the user's question based on the provided context. "
                "If the context doesn't contain enough information, say so. "
                "Always cite your sources using [Source: id] format."
            ),
            messages=[
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion: {query}",
                }
            ],
        )
        return response.content[0].text
```

---

## 2. 문서 청킹 전략

문서를 청크로 분할하는 방법은 검색 품질에 큰 영향을 미칩니다. 모든 상황에 맞는 단일 전략은 없습니다.

### 2.1 고정 크기 청킹(Fixed-Size Chunking)

가장 간단한 접근법: 문자 또는 토큰 수로 분할하고 오버랩을 줍니다.

```python
def fixed_size_chunks(
    text: str,
    chunk_size: int = 500,
    overlap: int = 50,
) -> list[str]:
    """Split text into fixed-size chunks with overlap."""
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start = end - overlap

    return chunks
```

**장점**: 단순하고 예측 가능한 청크 크기.
**단점**: 문장, 단락, 논리적 섹션을 중간에 잘라낼 수 있습니다.

### 2.2 의미적 청킹(Semantic Chunking)

자연스러운 경계에서 분할: 단락, 섹션, 또는 문장.

```python
import re


def semantic_chunks(
    text: str,
    max_chunk_size: int = 1000,
    min_chunk_size: int = 100,
) -> list[str]:
    """Split text on semantic boundaries (paragraphs, then sections)."""
    # 섹션별 분할 시도 (Markdown의 ## 제목)
    sections = re.split(r"\n(?=##\s)", text)

    chunks = []
    for section in sections:
        if len(section.split()) <= max_chunk_size:
            if len(section.split()) >= min_chunk_size:
                chunks.append(section.strip())
            elif chunks:
                # 작은 섹션을 이전 청크와 병합
                chunks[-1] += "\n\n" + section.strip()
            else:
                chunks.append(section.strip())
        else:
            # 섹션이 너무 큰 경우, 단락별로 분할
            paragraphs = section.split("\n\n")
            current_chunk = ""
            for para in paragraphs:
                if len((current_chunk + para).split()) <= max_chunk_size:
                    current_chunk += ("\n\n" if current_chunk else "") + para
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = para
            if current_chunk:
                chunks.append(current_chunk.strip())

    return chunks
```

**장점**: 논리적 구조를 보존하여 검색에 유리합니다.
**단점**: 가변적인 청크 크기, 더 복잡한 구현.

### 2.3 재귀 문자 분할(Recursive Character Splitting)

계층적 구분자로 분할: 먼저 섹션, 그 다음 단락, 문장, 단어 순서로.

```python
def recursive_split(
    text: str,
    max_size: int = 500,
    separators: list[str] | None = None,
) -> list[str]:
    """Recursively split text using a hierarchy of separators."""
    if separators is None:
        separators = ["\n\n## ", "\n\n", "\n", ". ", " "]

    if len(text.split()) <= max_size:
        return [text]

    # 각 구분자 수준 시도
    for sep in separators:
        parts = text.split(sep)
        if len(parts) > 1:
            chunks = []
            current = ""
            for part in parts:
                candidate = current + sep + part if current else part
                if len(candidate.split()) <= max_size:
                    current = candidate
                else:
                    if current:
                        chunks.append(current.strip())
                    current = part
            if current:
                chunks.append(current.strip())

            # 여전히 너무 큰 청크를 재귀적으로 분할
            result = []
            for chunk in chunks:
                if len(chunk.split()) > max_size:
                    result.extend(recursive_split(chunk, max_size, separators[1:]))
                else:
                    result.append(chunk)
            return result

    # 대체: 단어 단위로 분할
    return fixed_size_chunks(text, max_size, overlap=50)
```

### 2.4 전략 선택

| 콘텐츠 유형 | 권장 전략 |
|---|---|
| 구조화된 문서 (Markdown, HTML) | 의미적 (제목 기준) |
| 법률/과학 논문 | 의미적 (섹션/단락 기준) |
| 비구조화 텍스트 (로그, 트랜스크립트) | 오버랩 있는 고정 크기 |
| 코드 파일 | 의미적 (함수/클래스 기준) |
| 혼합 콘텐츠 | 재귀 문자 분할 |

---

## 3. 임베딩 모델과 벡터 데이터베이스

### 3.1 Voyage AI를 사용한 임베딩

Anthropic은 Claude와 함께 Voyage AI 임베딩 사용을 권장합니다:

```python
import voyageai


voyage_client = voyageai.Client()  # VOYAGE_API_KEY 환경 변수 사용


def get_embeddings(texts: list[str], input_type: str = "document") -> list[list[float]]:
    """Get embeddings from Voyage AI."""
    result = voyage_client.embed(
        texts,
        model="voyage-3",
        input_type=input_type,  # 인덱싱에는 "document", 검색에는 "query"
    )
    return result.embeddings


def embed_query(query: str) -> list[float]:
    """Embed a search query."""
    return get_embeddings([query], input_type="query")[0]


def embed_documents(docs: list[str]) -> list[list[float]]:
    """Embed documents for indexing."""
    # 128개씩 배치 처리
    all_embeddings = []
    for i in range(0, len(docs), 128):
        batch = docs[i : i + 128]
        embeddings = get_embeddings(batch, input_type="document")
        all_embeddings.extend(embeddings)
    return all_embeddings
```

### 3.2 벡터 데이터베이스 통합

ChromaDB(경량, 임베디드)를 사용한 예시:

```python
import chromadb


def create_rag_collection(
    collection_name: str,
    documents: list[dict],
) -> chromadb.Collection:
    """Create a ChromaDB collection with embedded documents."""
    client = chromadb.PersistentClient(path="./chroma_db")

    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    # 문서 배치 추가
    ids = [doc["id"] for doc in documents]
    texts = [doc["content"] for doc in documents]
    metadatas = [doc.get("metadata", {}) for doc in documents]
    embeddings = embed_documents(texts)

    collection.add(
        ids=ids,
        documents=texts,
        metadatas=metadatas,
        embeddings=embeddings,
    )

    return collection


def search_collection(
    collection: chromadb.Collection,
    query: str,
    top_k: int = 5,
    where: dict | None = None,
) -> list[dict]:
    """Search a collection with optional metadata filtering."""
    query_embedding = embed_query(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where=where,
    )

    return [
        {
            "id": results["ids"][0][i],
            "content": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "distance": results["distances"][0][i],
        }
        for i in range(len(results["ids"][0]))
    ]
```

### 3.3 하이브리드 검색: 임베딩 + BM25

밀집(Dense, 임베딩)과 희소(Sparse, BM25) 검색을 결합하면 정확도가 크게 향상됩니다:

```python
import math
from collections import Counter


class BM25:
    """Simple BM25 implementation for sparse retrieval."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_freqs = {}
        self.doc_lengths = []
        self.avg_dl = 0
        self.corpus_size = 0
        self.tokenized_docs = []

    def fit(self, documents: list[str]):
        """Index documents for BM25 scoring."""
        self.tokenized_docs = [doc.lower().split() for doc in documents]
        self.corpus_size = len(documents)
        self.doc_lengths = [len(doc) for doc in self.tokenized_docs]
        self.avg_dl = sum(self.doc_lengths) / self.corpus_size

        # 문서 빈도 계산
        for doc in self.tokenized_docs:
            unique_terms = set(doc)
            for term in unique_terms:
                self.doc_freqs[term] = self.doc_freqs.get(term, 0) + 1

    def score(self, query: str) -> list[float]:
        """Score all documents against a query."""
        query_terms = query.lower().split()
        scores = [0.0] * self.corpus_size

        for term in query_terms:
            if term not in self.doc_freqs:
                continue

            df = self.doc_freqs[term]
            idf = math.log((self.corpus_size - df + 0.5) / (df + 0.5) + 1)

            for i, doc in enumerate(self.tokenized_docs):
                tf = doc.count(term)
                dl = self.doc_lengths[i]
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * dl / self.avg_dl)
                scores[i] += idf * numerator / denominator

        return scores


class HybridRetriever:
    """Combine embedding similarity with BM25 for hybrid retrieval."""

    def __init__(self, alpha: float = 0.7):
        """
        Args:
            alpha: 임베딩 유사도 가중치 (1-alpha는 BM25에).
        """
        self.alpha = alpha
        self.bm25 = BM25()
        self.documents = []
        self.collection = None

    def index(self, documents: list[dict], collection_name: str = "hybrid"):
        """Index documents for both dense and sparse retrieval."""
        self.documents = documents
        texts = [doc["content"] for doc in documents]

        # 희소 인덱스
        self.bm25.fit(texts)

        # 밀집 인덱스
        self.collection = create_rag_collection(collection_name, documents)

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """Hybrid search combining embedding and BM25 scores."""
        # 밀집 점수 (거리를 유사도로 변환)
        dense_results = search_collection(self.collection, query, top_k=len(self.documents))
        dense_scores = {r["id"]: 1 - r["distance"] for r in dense_results}

        # 희소 점수
        bm25_scores = self.bm25.score(query)
        sparse_scores = {
            self.documents[i]["id"]: score
            for i, score in enumerate(bm25_scores)
        }

        # 점수를 [0, 1]로 정규화
        def normalize(scores: dict) -> dict:
            values = list(scores.values())
            if not values:
                return scores
            min_v, max_v = min(values), max(values)
            if max_v == min_v:
                return {k: 0.5 for k in scores}
            return {k: (v - min_v) / (max_v - min_v) for k, v in scores.items()}

        dense_norm = normalize(dense_scores)
        sparse_norm = normalize(sparse_scores)

        # 점수 결합
        combined = {}
        all_ids = set(dense_norm.keys()) | set(sparse_norm.keys())
        for doc_id in all_ids:
            d_score = dense_norm.get(doc_id, 0)
            s_score = sparse_norm.get(doc_id, 0)
            combined[doc_id] = self.alpha * d_score + (1 - self.alpha) * s_score

        # 정렬 후 top_k 반환
        sorted_ids = sorted(combined, key=combined.get, reverse=True)[:top_k]
        id_to_doc = {doc["id"]: doc for doc in self.documents}

        return [
            {**id_to_doc[doc_id], "score": combined[doc_id]}
            for doc_id in sorted_ids
            if doc_id in id_to_doc
        ]
```

---

## 4. 긴 컨텍스트 vs RAG 트레이드오프

Claude의 200K 토큰 컨텍스트 윈도우는 중요한 설계 결정을 만듭니다: RAG를 사용할지 모든 것을 컨텍스트에 넣을지.

### 4.1 결정 프레임워크

| 요소 | 긴 컨텍스트(Long Context) | RAG |
|---|---|---|
| **지식 베이스 크기** | < 200K 토큰 | 모든 크기 |
| **업데이트 빈도** | 거의 변경되지 않음 | 자주 업데이트됨 |
| **쿼리 유형** | 전체적 이해 필요 | 특정 사실 필요 |
| **지연 시간 요구사항** | 높은 지연 허용 | 빠른 검색 필요 |
| **비용 민감도** | 적은 양 | 대량 |
| **정확도 요구** | 모든 정보가 모델에 보임 | 검색 품질에 의존 |

### 4.2 긴 컨텍스트 사용 시기

```python
def stuff_context_approach(documents: list[str], query: str) -> str:
    """For smaller knowledge bases, just put everything in context."""
    full_context = "\n\n---\n\n".join(documents)

    # 컨텍스트 윈도우에 맞는지 확인
    # 대략적 추정: 1 토큰 ≈ 4 문자
    estimated_tokens = len(full_context) / 4
    if estimated_tokens > 180_000:  # 쿼리 + 응답을 위한 여유 공간
        raise ValueError(f"Context too large: ~{estimated_tokens:.0f} tokens")

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4096,
        system=[
            {
                "type": "text",
                "text": f"Reference documents:\n\n{full_context}",
                "cache_control": {"type": "ephemeral"},  # 문서를 캐시!
            }
        ],
        messages=[{"role": "user", "content": query}],
    )
    return response.content[0].text
```

### 4.3 RAG 사용 시기

- 지식 베이스가 200K 토큰을 초과할 때
- 문서가 자주 업데이트될 때 (재캐싱 방지)
- 쿼리당 소수의 문서만 관련될 때
- 비용이 관심사일 때 (쿼리당 200K 토큰 처리는 비쌈)
- 수백만 개의 문서로 확장해야 할 때

### 4.4 하이브리드: RAG + 긴 컨텍스트

```python
def hybrid_approach(query: str, top_k: int = 20) -> str:
    """Retrieve more documents than typical RAG and use long context."""
    # 넉넉한 후보 세트 검색
    candidates = retriever.search(query, top_k=top_k)

    # 모두 컨텍스트에 넣기 (20개 문서는 보통 200K 토큰 미만)
    context = "\n\n---\n\n".join(
        f"[Document {doc['id']}]\n{doc['content']}" for doc in candidates
    )

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4096,
        system=(
            "You have been given relevant documents to answer the user's question. "
            "Read ALL documents carefully before answering. "
            "Cite specific documents using [Document ID] format."
        ),
        messages=[
            {
                "role": "user",
                "content": f"Documents:\n\n{context}\n\nQuestion: {query}",
            }
        ],
    )
    return response.content[0].text
```

---

## 5. 컨텍스트 검색

Anthropic의 컨텍스트 검색(Contextual Retrieval) 기법은 각 청크를 임베딩하기 전에 컨텍스트를 앞에 추가하여 청크 검색 정확도를 향상시킵니다.

### 5.1 단순 청킹의 문제점

문서를 청킹하면 개별 청크가 종종 맥락을 잃습니다:

```
원본: "2024년 3분기에 매출이 전년 대비 15% 성장하여 23억 달러..."
청크: "매출이 전년 대비 15% 성장하여 23억 달러"
문제: 어떤 회사? 어떤 분기? 어떤 연도?
```

### 5.2 컨텍스트 설명 추가

Claude를 사용하여 각 청크에 대한 짧은 컨텍스트 접두사를 생성합니다:

```python
def add_chunk_context(
    chunk: str,
    full_document: str,
    doc_title: str,
) -> str:
    """Use Claude to generate a contextual prefix for a chunk."""
    response = client.messages.create(
        model="claude-haiku-4-20250514",  # 비용 효율을 위해 Haiku 사용
        max_tokens=200,
        system=[
            {
                "type": "text",
                "text": (
                    "You will be given a document and a chunk from that document. "
                    "Generate a SHORT (1-2 sentence) context that situates the chunk "
                    "within the full document. Include key identifiers like company name, "
                    "date, section title. Return ONLY the context, nothing else."
                ),
            }
        ],
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"Document title: {doc_title}\n\nFull document:\n{full_document}",
                        "cache_control": {"type": "ephemeral"},  # 전체 문서를 캐시
                    },
                    {
                        "type": "text",
                        "text": f"Chunk:\n{chunk}",
                    },
                ],
            }
        ],
    )
    context = response.content[0].text.strip()
    return f"{context}\n\n{chunk}"


def build_contextual_index(
    document: str,
    doc_title: str,
    chunk_fn,
) -> list[dict]:
    """Build a contextual retrieval index for a document."""
    chunks = chunk_fn(document)

    contextualized = []
    for i, chunk in enumerate(chunks):
        contextualized_chunk = add_chunk_context(chunk, document, doc_title)
        contextualized.append({
            "id": f"{doc_title}-chunk-{i}",
            "content": contextualized_chunk,
            "metadata": {
                "source": doc_title,
                "chunk_index": i,
                "original_content": chunk,  # 표시용 원본 유지
            },
        })

    return contextualized
```

### 5.3 하이브리드 검색을 활용한 컨텍스트 검색

컨텍스트 임베딩과 BM25를 결합하면 최상의 결과를 얻을 수 있습니다:

```python
class ContextualRAG:
    """RAG pipeline with contextual retrieval and hybrid search."""

    def __init__(self, alpha: float = 0.7):
        self.client = anthropic.Anthropic()
        self.retriever = HybridRetriever(alpha=alpha)

    def ingest(self, documents: list[dict]):
        """Ingest documents with contextual chunk processing."""
        all_chunks = []
        for doc in documents:
            chunks = build_contextual_index(
                doc["content"],
                doc["title"],
                chunk_fn=lambda t: semantic_chunks(t, max_chunk_size=500),
            )
            all_chunks.extend(chunks)

        self.retriever.index(all_chunks, collection_name="contextual_rag")

    def query(self, question: str, top_k: int = 5) -> str:
        """Query with contextual retrieval."""
        results = self.retriever.search(question, top_k=top_k)

        # 표시에는 원본 콘텐츠를 사용하지만, 검색에는 컨텍스트화된 콘텐츠가 사용됨
        context = "\n\n---\n\n".join(
            f"[Source: {r['metadata']['source']}]\n{r['metadata'].get('original_content', r['content'])}"
            for r in results
        )

        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2048,
            system=(
                "Answer based on the provided sources. "
                "Cite sources using [Source: name] format. "
                "If information is insufficient, say so explicitly."
            ),
            messages=[
                {
                    "role": "user",
                    "content": f"Sources:\n{context}\n\nQuestion: {question}",
                }
            ],
        )
        return response.content[0].text
```

---

## 6. 인용과 근거 확보 패턴

신뢰할 수 있는 인용(Citation)은 신뢰성 있는 RAG 시스템에 필수적입니다.

### 6.1 인라인 인용 패턴(Inline Citation Pattern)

```python
CITATION_SYSTEM_PROMPT = """Answer the user's question based on the provided sources.

CITATION RULES:
1. Every factual claim MUST include a citation in [Source N] format
2. If multiple sources support a claim, cite all: [Source 1][Source 3]
3. Direct quotes must use quotation marks with citation: "exact text" [Source 2]
4. If no source supports a claim, explicitly state it is your general knowledge
5. End your answer with a "Sources Used" section listing all cited sources

Example:
The company reported $2.3B in revenue [Source 1], representing a 15% increase
year-over-year [Source 1][Source 3]. The CEO noted this was "driven primarily by
cloud services" [Source 2].

Sources Used:
- [Source 1]: Q3 2024 Earnings Report
- [Source 2]: CEO Earnings Call Transcript
- [Source 3]: Annual Report 2024"""


def generate_with_citations(query: str, sources: list[dict]) -> str:
    """Generate an answer with inline citations."""
    context = "\n\n".join(
        f"[Source {i+1}]: {src['title']}\n{src['content']}"
        for i, src in enumerate(sources)
    )

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4096,
        system=CITATION_SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": f"{context}\n\nQuestion: {query}",
            }
        ],
    )
    return response.content[0].text
```

### 6.2 도구 사용을 통한 구조화된 인용(Structured Citation)

프로그래밍 방식의 인용 추출을 위해 도구 사용을 활용합니다:

```python
citation_tool = {
    "name": "submit_answer",
    "description": "Submit an answer with structured citations.",
    "input_schema": {
        "type": "object",
        "properties": {
            "answer": {
                "type": "string",
                "description": "The answer text with [N] citation markers.",
            },
            "citations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "marker": {"type": "string", "description": "Citation marker, e.g., [1]"},
                        "source_id": {"type": "string"},
                        "quote": {"type": "string", "description": "Exact quoted text from source"},
                        "relevance": {"type": "string", "enum": ["direct", "supporting", "background"]},
                    },
                    "required": ["marker", "source_id", "quote"],
                },
            },
            "confidence": {
                "type": "string",
                "enum": ["high", "medium", "low"],
                "description": "Confidence that sources fully support the answer.",
            },
            "unsupported_claims": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Any claims in the answer not supported by sources.",
            },
        },
        "required": ["answer", "citations", "confidence"],
    },
}
```

### 6.3 근거 확보 검증(Grounding Verification)

```python
def verify_grounding(answer: str, sources: list[str]) -> dict:
    """Verify that an answer is grounded in the provided sources."""
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2048,
        system=(
            "You are a fact-checking assistant. Verify each claim in the answer "
            "against the provided sources. For each claim, determine if it is:\n"
            "- SUPPORTED: Directly stated or clearly implied by a source\n"
            "- PARTIALLY_SUPPORTED: Related info exists but claim goes beyond sources\n"
            "- UNSUPPORTED: No source evidence for this claim\n"
            "- CONTRADICTED: Sources say the opposite\n\n"
            "Return JSON: {\"claims\": [{\"text\": str, \"status\": str, \"source\": str|null}]}"
        ),
        messages=[
            {
                "role": "user",
                "content": (
                    f"Sources:\n" + "\n---\n".join(sources) +
                    f"\n\nAnswer to verify:\n{answer}"
                ),
            }
        ],
    )
    return json.loads(response.content[0].text)
```

---

## 7. 다단계 RAG

복잡한 쿼리는 종종 단일 검색 단계로 답변할 수 없습니다. 다단계 RAG는 쿼리를 분해하고 반복적으로 정보를 검색합니다.

### 7.1 쿼리 분해(Query Decomposition)

```python
def decompose_query(query: str) -> list[str]:
    """Break a complex query into simpler sub-queries."""
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        system=(
            "Break the user's complex question into 2-5 simpler sub-questions "
            "that, when answered together, fully address the original question. "
            "Return a JSON array of strings. No explanation, just the array."
        ),
        messages=[{"role": "user", "content": query}],
    )
    return json.loads(response.content[0].text)


def multi_step_rag(query: str, retriever, top_k: int = 5) -> str:
    """Answer a complex query by decomposing and retrieving iteratively."""
    sub_queries = decompose_query(query)
    all_context = []

    for sub_query in sub_queries:
        results = retriever.search(sub_query, top_k=top_k)
        for r in results:
            if r["content"] not in [c["content"] for c in all_context]:
                all_context.append(r)

    # 검색된 모든 컨텍스트에서 최종 답변 합성
    context_text = "\n\n---\n\n".join(
        f"[{doc['id']}]\n{doc['content']}" for doc in all_context
    )

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4096,
        system=(
            "Synthesize a comprehensive answer from the provided sources. "
            "The user asked a complex question that was broken into sub-questions. "
            "Address all aspects of the original question."
        ),
        messages=[
            {
                "role": "user",
                "content": (
                    f"Original question: {query}\n\n"
                    f"Sub-questions investigated: {json.dumps(sub_queries)}\n\n"
                    f"Retrieved sources:\n{context_text}"
                ),
            }
        ],
    )
    return response.content[0].text
```

### 7.2 가상 문서 임베딩(HyDE, Hypothetical Document Embeddings)

먼저 가상의 답변을 생성한 후, 이를 사용하여 더 나은 문서를 검색합니다:

```python
def hyde_retrieval(query: str, retriever, top_k: int = 5) -> list[dict]:
    """Use HyDE to improve retrieval quality."""
    # 1단계: 가상의 답변 생성
    response = client.messages.create(
        model="claude-haiku-4-20250514",
        max_tokens=512,
        system=(
            "Write a short, factual answer to the question as if you had access "
            "to a comprehensive knowledge base. This will be used for search, "
            "so include specific terms and details."
        ),
        messages=[{"role": "user", "content": query}],
    )
    hypothetical_answer = response.content[0].text

    # 2단계: 가상 답변을 쿼리로 사용하여 검색
    results = retriever.search(hypothetical_answer, top_k=top_k)

    return results
```

### 7.3 도구 사용을 활용한 에이전틱 RAG(Agentic RAG)

Claude가 언제, 무엇을 검색할지 자율적으로 결정하게 합니다:

```python
def agentic_rag(query: str, retriever) -> str:
    """Let Claude autonomously decide when to search and what to look for."""
    search_tool = {
        "name": "search_knowledge_base",
        "description": "Search the knowledge base for relevant documents.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "top_k": {"type": "integer", "description": "Number of results (1-10)", "default": 5},
            },
            "required": ["query"],
        },
    }

    messages = [{"role": "user", "content": query}]

    while True:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            system=(
                "You are a research assistant with access to a knowledge base. "
                "Use the search tool to find relevant information before answering. "
                "You may search multiple times with different queries. "
                "When you have enough information, provide a comprehensive answer with citations."
            ),
            tools=[search_tool],
            messages=messages,
        )

        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "end_turn":
            return next(b.text for b in response.content if b.type == "text")

        # 검색 도구 호출 실행
        tool_results = []
        for block in response.content:
            if block.type == "tool_use" and block.name == "search_knowledge_base":
                results = retriever.search(
                    block.input["query"],
                    top_k=block.input.get("top_k", 5),
                )
                formatted = "\n\n".join(
                    f"[{r['id']}] (score: {r.get('score', 'N/A')})\n{r['content']}"
                    for r in results
                )
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": formatted,
                })

        messages.append({"role": "user", "content": tool_results})
```

---

## 8. RAG 평가 메트릭

### 8.1 핵심 메트릭

| 메트릭 | 측정 대상 | 범위 |
|---|---|---|
| **컨텍스트 정밀도(Context Precision)** | 검색된 문서가 관련성 있는가? | 0-1 |
| **컨텍스트 재현율(Context Recall)** | 필요한 모든 문서가 검색되었는가? | 0-1 |
| **충실도(Faithfulness)** | 답변이 컨텍스트에 의해 뒷받침되는가? | 0-1 |
| **답변 관련성(Answer Relevance)** | 답변이 쿼리를 다루는가? | 0-1 |

### 8.2 Claude를 심사위원으로 활용한 평가

```python
class RAGEvaluator:
    """Evaluate RAG pipeline quality using Claude as a judge."""

    def __init__(self):
        self.client = anthropic.Anthropic()

    def evaluate_faithfulness(
        self,
        answer: str,
        context: list[str],
    ) -> dict:
        """Evaluate whether the answer is faithful to the provided context."""
        context_text = "\n\n---\n\n".join(context)

        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            system=(
                "Evaluate the faithfulness of the answer to the provided context.\n\n"
                "1. Extract each factual claim from the answer\n"
                "2. Check if each claim is supported by the context\n"
                "3. Calculate faithfulness = supported_claims / total_claims\n\n"
                "Return JSON:\n"
                '{"claims": [{"text": str, "supported": bool}], '
                '"faithfulness_score": float, "reasoning": str}'
            ),
            messages=[
                {
                    "role": "user",
                    "content": f"Context:\n{context_text}\n\nAnswer:\n{answer}",
                }
            ],
        )
        return json.loads(response.content[0].text)

    def evaluate_relevance(
        self,
        query: str,
        answer: str,
    ) -> dict:
        """Evaluate whether the answer is relevant to the query."""
        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            system=(
                "Evaluate answer relevance to the query.\n\n"
                "Consider:\n"
                "1. Does the answer address the question directly?\n"
                "2. Is the answer complete (covers all aspects)?\n"
                "3. Is the answer concise (no unnecessary information)?\n\n"
                "Return JSON:\n"
                '{"relevance_score": float (0-1), "completeness": float (0-1), '
                '"conciseness": float (0-1), "reasoning": str}'
            ),
            messages=[
                {
                    "role": "user",
                    "content": f"Query: {query}\n\nAnswer: {answer}",
                }
            ],
        )
        return json.loads(response.content[0].text)

    def evaluate_context_precision(
        self,
        query: str,
        contexts: list[str],
    ) -> dict:
        """Evaluate whether retrieved contexts are relevant to the query."""
        context_list = "\n\n".join(
            f"[Context {i+1}]:\n{ctx}" for i, ctx in enumerate(contexts)
        )

        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            system=(
                "For each retrieved context, determine if it is relevant to answering "
                "the query. Return JSON:\n"
                '{"contexts": [{"id": int, "relevant": bool}], '
                '"precision": float (relevant / total)}'
            ),
            messages=[
                {
                    "role": "user",
                    "content": f"Query: {query}\n\nRetrieved contexts:\n{context_list}",
                }
            ],
        )
        return json.loads(response.content[0].text)

    def full_evaluation(
        self,
        query: str,
        answer: str,
        contexts: list[str],
    ) -> dict:
        """Run all evaluation metrics."""
        faithfulness = self.evaluate_faithfulness(answer, contexts)
        relevance = self.evaluate_relevance(query, answer)
        precision = self.evaluate_context_precision(query, contexts)

        return {
            "faithfulness": faithfulness["faithfulness_score"],
            "answer_relevance": relevance["relevance_score"],
            "context_precision": precision["precision"],
            "details": {
                "faithfulness": faithfulness,
                "relevance": relevance,
                "precision": precision,
            },
        }
```

---

## 9. MCP를 활용한 프로덕션 RAG

MCP 서버는 RAG 컴포넌트에 대한 깔끔한 추상화를 제공하여 아키텍처를 모듈화하고 재사용 가능하게 만듭니다.

### 9.1 MCP 서버로서의 RAG

```python
"""MCP server that exposes a RAG pipeline as tools."""
from mcp.server import Server
from mcp.types import Tool, TextContent

app = Server("rag-server")

# RAG 컴포넌트 초기화 (프로덕션에서는 영구 스토리지에서 로드)
retriever = None  # 시작 시 초기화


@app.list_tools()
async def list_tools():
    return [
        Tool(
            name="search",
            description="Search the knowledge base for relevant documents.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "top_k": {"type": "integer", "default": 5},
                    "filter_source": {"type": "string", "description": "Optional: filter by source name"},
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="get_document",
            description="Retrieve a full document by ID.",
            inputSchema={
                "type": "object",
                "properties": {
                    "document_id": {"type": "string"},
                },
                "required": ["document_id"],
            },
        ),
        Tool(
            name="ingest",
            description="Add a new document to the knowledge base.",
            inputSchema={
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "content": {"type": "string"},
                    "metadata": {"type": "object"},
                },
                "required": ["title", "content"],
            },
        ),
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict):
    if name == "search":
        results = retriever.search(
            arguments["query"],
            top_k=arguments.get("top_k", 5),
        )
        formatted = "\n\n---\n\n".join(
            f"**[{r['id']}]** (relevance: {r.get('score', 'N/A'):.3f})\n{r['content']}"
            for r in results
        )
        return [TextContent(type="text", text=formatted)]

    elif name == "get_document":
        doc = retriever.get_by_id(arguments["document_id"])
        if doc:
            return [TextContent(type="text", text=doc["content"])]
        return [TextContent(type="text", text="Document not found.")]

    elif name == "ingest":
        doc = {
            "id": arguments["title"].lower().replace(" ", "-"),
            "content": arguments["content"],
            "metadata": arguments.get("metadata", {}),
        }
        retriever.add_document(doc)
        return [TextContent(type="text", text=f"Ingested: {doc['id']}")]
```

### 9.2 다중 소스 RAG 아키텍처(Multi-Source RAG Architecture)

```python
"""Claude Code configuration for multi-source RAG."""

# .claude/settings.json
MCP_CONFIG = {
    "mcpServers": {
        "docs-rag": {
            "command": "python",
            "args": ["-m", "rag_server", "--source", "documentation"],
        },
        "tickets-rag": {
            "command": "python",
            "args": ["-m", "rag_server", "--source", "jira"],
        },
        "code-search": {
            "command": "python",
            "args": ["-m", "rag_server", "--source", "codebase"],
        },
    }
}

# Claude가 이제 모든 소스를 검색할 수 있습니다:
# "docs-rag에서 API 참조를 검색하고, tickets-rag에서 관련 버그를,
#  code-search에서 구현 예제를 검색하세요"
```

### 9.3 프로덕션 체크리스트

- [ ] **청킹(Chunking)**: 콘텐츠 유형에 따라 전략 선택; 청크 크기 테스트
- [ ] **임베딩(Embedding)**: Voyage AI 또는 동등한 것 사용; 효율적 배치 임베딩
- [ ] **검색(Retrieval)**: 하이브리드 검색 구현 (임베딩 + BM25)
- [ ] **컨텍스트 검색**: 청크에 컨텍스트 접두사 추가
- [ ] **캐싱(Caching)**: 시스템 프롬프트와 반복 컨텍스트 캐시
- [ ] **평가(Evaluation)**: 자동 충실도 및 관련성 검사 설정
- [ ] **모니터링(Monitoring)**: 검색 적중률, 답변 품질, 지연 시간 추적
- [ ] **업데이트(Updates)**: 새/업데이트된 문서를 위한 점진적 인덱싱 구현
- [ ] **대체(Fallback)**: 관련 문서를 찾지 못한 경우 처리
- [ ] **레이트 제한(Rate Limiting)**: 배치 수집 시 API 제한 준수

---

## 10. 연습 문제

### 연습 문제 1: 기본 RAG 파이프라인

Markdown 문서 파일 세트를 위한 완전한 RAG 파이프라인을 구축하세요:

```python
"""
Exercise 1 starter code — build a Markdown documentation RAG pipeline.
"""
from pathlib import Path


class MarkdownRAG:
    """RAG pipeline for Markdown documentation."""

    def __init__(self, docs_dir: str):
        self.docs_dir = Path(docs_dir)
        self.client = anthropic.Anthropic()
        # TODO: Initialize embedding function and retriever

    def ingest(self):
        """Load and index all Markdown files from docs_dir."""
        # TODO: Read all .md files
        # TODO: Split into chunks using semantic chunking
        # TODO: Embed and index chunks
        pass

    def query(self, question: str, top_k: int = 5) -> str:
        """Answer a question using the indexed documentation."""
        # TODO: Retrieve relevant chunks
        # TODO: Generate answer with citations
        pass


# 테스트
rag = MarkdownRAG("./docs")
rag.ingest()
answer = rag.query("How do I configure authentication?")
print(answer)
```

### 연습 문제 2: 컨텍스트 검색 구현

각 청크에 컨텍스트를 추가하는 컨텍스트 검색을 구현하세요:

```python
"""
Exercise 2 starter code — implement contextual retrieval.
"""


class ContextualRetriever:
    """Retriever that uses contextual chunk descriptions."""

    def __init__(self):
        self.client = anthropic.Anthropic()
        # TODO: Initialize storage

    def ingest_document(self, title: str, content: str):
        """
        Ingest a document with contextual chunk processing.

        Steps:
        1. Chunk the document
        2. For each chunk, generate a contextual description using Claude
        3. Embed the contextualized chunks
        4. Store in vector database
        """
        # TODO: Implement
        pass

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """Search with hybrid retrieval (embedding + BM25)."""
        # TODO: Implement hybrid search
        pass
```

### 연습 문제 3: RAG 평가 스위트

RAG 파이프라인의 품질을 측정하는 평가 스위트를 구축하세요:

```python
"""
Exercise 3 starter code — RAG evaluation suite.
"""


class RAGTestSuite:
    """Evaluate a RAG pipeline on a test dataset."""

    def __init__(self, rag_pipeline):
        self.rag = rag_pipeline
        self.evaluator = RAGEvaluator()

    def load_test_cases(self, path: str) -> list[dict]:
        """
        Load test cases from a JSON file.

        Each test case: {
            "query": str,
            "expected_answer": str,  # Reference answer
            "required_sources": [str],  # Doc IDs that should be retrieved
        }
        """
        # TODO: Load and validate test cases
        pass

    def run(self, test_cases: list[dict]) -> dict:
        """
        Run evaluation on all test cases.

        Returns aggregate metrics:
        - Average faithfulness
        - Average relevance
        - Average context precision
        - Average context recall
        """
        # TODO: For each test case:
        #   1. Run RAG query
        #   2. Evaluate with all metrics
        #   3. Aggregate results
        pass

    def report(self, results: dict) -> str:
        """Generate a formatted evaluation report."""
        # TODO: Format results as a readable report
        pass
```

### 연습 문제 4: 다단계 RAG 에이전트

복잡한 쿼리를 분해하고 반복적으로 검색하는 RAG 에이전트를 구축하세요:

```python
"""
Exercise 4 starter code — multi-step RAG agent.
"""


class ResearchAgent:
    """Agent that performs multi-step research using RAG."""

    def __init__(self, retriever):
        self.client = anthropic.Anthropic()
        self.retriever = retriever

    def research(self, question: str, max_steps: int = 5) -> dict:
        """
        Perform multi-step research to answer a complex question.

        Returns:
            {
                "answer": str,
                "steps": [
                    {"query": str, "findings": str}
                ],
                "sources": [str],
                "confidence": float,
            }
        """
        # TODO: Decompose the question into sub-queries
        # TODO: For each sub-query, retrieve and analyze
        # TODO: Synthesize findings into a comprehensive answer
        # TODO: Verify answer is grounded in sources
        pass
```

---

**이전**: [24. 프롬프트 캐싱과 Batch API](./24_Prompt_Caching_and_Batch_API.md)
