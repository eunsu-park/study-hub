# 10. RAG 기초

## 학습 목표

- RAG (Retrieval-Augmented Generation) 이해
- 문서 임베딩과 검색
- 청킹 전략
- RAG 파이프라인 구현

---

## 이론과 원리

RAG(Retrieval-Augmented Generation)는 LLM의 두 가지 근본적 한계를 다룹니다 — **지식 컷오프**(모델은 사전학습 코퍼스에 있던 것만 안다)와 **환각(hallucination)**(모델은 불확실할 때 자유롭게 내용을 발명한다). 레시피는 단순합니다 — 추론 시 외부 코퍼스에서 관련 문서를 *검색*하고, 프롬프트 앞에 *붙여*, LLM이 검색된 증거에 기반하여 답을 *생성*하게 합니다. 공학적 도전은 이 세 단계 — 검색, 부착, 생성 — 각각을 신뢰성 있고, 빠르고, 정확하게 만드는 것입니다.

이 섹션은 다음을 다룹니다:

- **(A) 왜 검색하는가** — 파라메트릭(parametric) vs 비파라메트릭(non-parametric) 메모리 트레이드오프, 그리고 RAG가 파인튜닝 대비 무엇을 사주는가.
- **(B) 임베딩과 유사도** — 텍스트가 어떻게 벡터가 되며, 왜 cosine similarity가 자연스러운 척도인가.
- **(C) 청킹(chunking) 전략** — 검색 단위 질문, 슬라이딩 윈도우, 의미적 청킹, 재귀 분할.
- **(D) Dense vs sparse 검색** — 임베딩 기반 vs BM25, 각 사용 시점, 하이브리드 탐색.
- **(E) RAG 파이프라인** — 인용을 포함한 엔드 투 엔드 흐름, 컨텍스트 길이 예산의 역할.
- **(F) 평가** — 검색 지표(recall@k, MRR), 생성 지표(faithfulness, answer relevance), RAGAS 프레임워크.

### A. 왜 검색하는가: 파라메트릭 vs 비파라메트릭 메모리

LLM은 가중치에 지식을 저장합니다 — *파라메트릭 메모리*. RAG는 추론 시 관련 조각을 가져오는 별도의 텍스트 저장소를 추가합니다 — *비파라메트릭 메모리*. 둘은 보완적 속성을 가집니다:

| 속성 | 파라메트릭 (LLM 가중치) | 비파라메트릭 (RAG 코퍼스) |
|------|------------------------|---------------------------|
| 갱신 비용 | 재학습 또는 파인튜닝 (비쌈) | 문서 재인덱싱 (저렴) |
| 용량 | 파라미터 수에 의해 제한 | 저장소에 의해 제한 |
| 인용 가능성 | 출처 표시 없음 | 정확한 출처 가용 |
| 지연 | 단일 forward pass | 검색 + forward pass |
| 프라이버시 | 지식이 구워져 들어감 | 사용자/테넌트별 범위 가능 |

RAG는 다음 사용 사례에서 파인튜닝을 지배합니다 — 지식이 자주 변경(뉴스, 문서), 인용이 필요(법률, 의료), 사용자별 지식 격리 필요(멀티 테넌트 SaaS), 코퍼스가 암기하기에 너무 큼(엔터프라이즈 문서 저장소).

파인튜닝은 다음에서 여전히 이깁니다 — 기술/행동(형식, 어조, 거부 패턴), 매우 안정적인 지식, 검색 오버헤드를 받아들일 수 없는 매우 빡빡한 지연 예산.

### B. 임베딩과 유사도

**B.1 문장 임베딩.** 모델 `f : text → ℝ^d`(보통 d ∈ [384, 1536])로 의미적으로 비슷한 텍스트가 가까운 벡터로 매핑됩니다. 현대 임베딩 모델(Sentence-BERT, OpenAI `text-embedding-3`, BGE, Cohere Embed)은 **대조 손실(contrastive loss)**로 학습됩니다 — 양성 쌍(예: 질문과 그 정답 단락)은 끌어당기고, 음성은 밀어냅니다:

```
L = − log [ exp(sim(q, p+) / τ) / Σ_{p ∈ {p+, p-_1, ..., p-_k}} exp(sim(q, p) / τ) ]
```

학습 후 `sim(q, p)`(보통 cosine)가 모든 `p-`보다 `p+`를 위로 순위 매깁니다. 임베딩 공간이 이 기하를 상속 — 관련 청크가 그 쿼리 근처에 군집합니다.

**B.2 Cosine vs dot product.** Cosine similarity `cos(a, b) = (a · b) / (||a|| · ||b||)`는 벡터 크기에 불변. Dot product `a · b`는 그렇지 않음 — 긴 벡터가 방향과 무관하게 이깁니다. cosine 손실로 학습된 모델(대부분)에는 cosine이 정확한 척도. dot product로 학습된 모델(예: 일부 검색 모델)에는 dot product가 정확. 척도를 학습과 일치시키지 않으면 조용히 품질이 떨어집니다.

**B.3 왜 이 기하가 의미를 부호화하는가.** 임베딩 학습은 방향이 의미 축(주제, 어조, 의도)에 해당하는 저차원 매니폴드를 만듭니다. 대조 목적함수는 모델이 관련을 무관에서 실제로 구별하는 특징을 발견하도록 강제합니다. 충분한 학습 쌍으로 기하가 일반화 — 미지의 쿼리가 미지의-그러나-관련된 단락 근처에 떨어집니다.

### C. 청킹 전략

검색 단위 질문 — 문장을 임베딩하고 검색할까요? 단락? 전체 문서? 트레이드오프:

- **작은 청크**: 정밀한 검색(청크가 대부분 관련), 그러나 답에 필요할 수 있는 주변 맥락을 잃습니다.
- **큰 청크**: 검색당 더 많은 맥락, 그러나 같은 청크가 무관한 쿼리에 검색될 수 있음(recall-precision 희석).

**C.1 고정 크기 + 겹침.** N 토큰 청크(전형적 N = 256-1024)로 분할하고 M 토큰 겹침(M ≈ 0.1·N). 겹침이 청크 경계를 가로지르는 답을 잃지 않게 합니다.

**C.2 문장- 또는 단락-정렬.** 자연스러운 문서 구조를 사용. 생각 중간에 자르지 않지만 가변 청크 크기를 만듭니다.

**C.3 재귀적 문자 텍스트 분할.** 단락 경계 먼저 분할 시도, 다음 문장, 단어, 문자 — 청크가 목표 크기에 맞을 때까지 재귀. LangChain이 사용.

**C.4 의미적 청킹.** 문장에 대한 임베딩 계산; 임베딩이 가까운(같은 주제) 연속 문장 병합. 임베딩 유사도가 급격히 떨어지는 곳에서 분할(주제 변경). 더 비싸고 종종 더 높은 품질 청크.

**C.5 정보 이론적 관점.** 각 청크는 맥락 단위. `k`개 청크 검색은 모델의 컨텍스트 윈도우로 제한된 대략 `k · chunk_size` 토큰의 맥락 예산을 줍니다. 작은 청크 → 더 높은 `k`가 가능 → 다양한 쿼리에 더 나은 recall; 큰 청크 → 더 낮은 `k` → 더 나은 단일 청크 완전성. 최적점은 쿼리 유형(특정 사실 vs 광범위 요약)에 의존하며 보통 경험적입니다.

### D. Dense vs Sparse 검색

**D.1 Sparse (BM25, TF-IDF).** 쿼리와 문서를 단어 가방으로 — 빈도 가중 점수로 어휘 중첩에 의해 순위 매김. BM25 점수:

```
score(q, d) = Σ_{t ∈ q}  IDF(t) · [ (tf(t,d) · (k₁+1)) / (tf(t,d) + k₁ · (1 − b + b · |d|/avgdl)) ]
```

파라미터 `k₁ ≈ 1.5`, `b ≈ 0.75`. 강점 — 정확한 용어 매칭(고유명사, ID, 정확한 구문에 좋음), 임베딩 모델 불필요, 역색인에서 매우 빠름. 약점 — 어휘 불일치(동의어 비가시), 의미적 일반화 없음.

**D.2 Dense (임베딩 기반).** 쿼리와 문서를 벡터로 인코딩, 최근접 이웃 검색. 강점 — 의미적 매칭(동의어, 패러프레이즈), 다국어 인코더로 교차 언어 능력. 약점 — 정확한 어휘 매칭 놓침, 임베딩 모델 + 벡터 인덱스 필요.

**D.3 하이브리드 탐색.** 둘 결합 — BM25에서 top-N과 dense에서 top-N 검색, 합집합 후 재순위 또는 **Reciprocal Rank Fusion (RRF)** 사용:

```
RRF_score(d) = Σ_methods  1 / (k + rank_method(d))   전형적 k = 60
```

하이브리드는 거의 항상 단독보다 우수 — 둘은 분리된 경우에 실패합니다. 프로덕션 RAG 시스템은 기본적으로 하이브리드.

### E. RAG 파이프라인

```
[사용자 쿼리]
    ↓
[쿼리 임베딩]
    ↓
[벡터 저장소 + BM25에서 top-k 검색]
    ↓
[(선택) 크로스 인코더로 재순위]
    ↓
[프롬프트 구축: 지시 + 검색된 청크(출처 포함) + 쿼리]
    ↓
[LLM이 인라인 인용과 함께 답 생성]
    ↓
[(선택) 인용 검증, 환각 시 재시도]
```

**컨텍스트 예산.** 총 프롬프트 토큰 ≤ 컨텍스트 윈도우. 오버헤드(시스템 프롬프트, 지시, 쿼리, 예상 응답)를 빼면 — 검색된 청크의 예산은 32K 컨텍스트 모델에서 일반적으로 4-16K 토큰. 512 토큰 청크로 `k = 8-32` 청크.

**인용 규율.** 견고한 RAG 프롬프트는 청크별 식별자(`[Source 3]`)를 포함하고 모델에게 모든 주장에 대해 출처를 인용하라고 지시합니다. 다운스트림 파이프라인은 인용된 출처가 검색된 집합에 실제로 등장하는지 검증하고, 존재하지 않는 출처를 인용하는 답변은 거부할 수 있습니다.

### F. 평가

**F.1 검색 지표.** 라벨된 (쿼리, 관련 청크 ID) 집합이 주어졌을 때:

- **Recall@k**: 상위 k 검색 집합 내 관련 청크의 비율. RAG 검색의 지배적 지표 — 관련 청크가 검색되지 않으면 어떤 LLM도 정확히 답할 수 없습니다.
- **MRR (Mean Reciprocal Rank)**: 첫 번째 관련 청크의 1/순위, 평균. 관련 청크가 얼마나 위에 나타나는지를 포착.
- **NDCG**: 등급화된 관련성 처리.

**F.2 생성 지표** (RAGAS 프레임워크):

- **Faithfulness**: 답의 모든 주장이 검색된 청크에서 따라오는가? LLM 판사가 주장을 추출하고 각각을 맥락에 대해 검증하여 점수.
- **Answer relevance**: 답이 쿼리를 다루는가? LLM 판사가 답에 대해 그럴듯한 질문을 생성하고 원래 쿼리와 임베딩 유사도 측정.
- **Context precision/recall**: 검색된 청크가 (오직) 관련 정보를 담는가?

이 모두는 다른 LLM을 판사로 계산할 수 있습니다 — 표준 패턴은 GPT-4를 평가자로, 테스트 중인 모든 모델의 출력에 적용.

### 이론에서 아래 함수들로

- §1 (RAG 개요) — §A 파라메트릭 vs 비파라메트릭 트레이드오프를 틀.
- §2 (전처리) — §C 청킹 전략(고정, 재귀, 의미적) 구현.
- §3 (임베딩 생성) — §B 임베딩 모델(Sentence-BERT, OpenAI 임베딩) 호출.
- §4 (벡터 검색) — 기본 dense 검색; 하이브리드(§D)는 레슨 11에서.
- §5 (단순 RAG) — 작은 코퍼스에서 §E 파이프라인 엔드 투 엔드.
- §6 (고급 기법) — 쿼리 재작성과 재순위 매기기 안내(전체 레슨 12).
- §7 (청킹 비교) — §C 전략의 경험적 비교.
- §8 (평가) — §F RAGAS 지표를 단순 RAG에 구현.

---

## 1. RAG 개요

### 왜 RAG인가?

```
LLM의 한계:
- 학습 데이터 이후 정보 모름 (지식 컷오프)
- 환각 (잘못된 정보 생성)
- 특정 도메인 지식 부족

RAG 해결책:
- 외부 지식 검색 후 답변 생성
- 최신 정보 반영 가능
- 출처 제공으로 신뢰성 향상
```

### RAG 아키텍처

> **RAG Pipeline**
>
> 질문 --> 임베딩 --> 벡터 검색 --> 관련 문서
>
> 질문 + 문서 --> LLM --> 답변

---

## 2. 문서 전처리

### 청킹 (Chunking)

```python
def chunk_text(text, chunk_size=500, overlap=50):
    """텍스트를 오버랩이 있는 청크로 분할"""
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap

    return chunks

# 사용
text = "Very long document text here..."
chunks = chunk_text(text, chunk_size=500, overlap=100)
```

### 문장 기반 청킹

```python
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize

def chunk_by_sentences(text, max_sentences=5, overlap_sentences=1):
    sentences = sent_tokenize(text)
    chunks = []

    for i in range(0, len(sentences), max_sentences - overlap_sentences):
        chunk = ' '.join(sentences[i:i + max_sentences])
        chunks.append(chunk)

    return chunks
```

### 시맨틱 청킹

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
)

chunks = splitter.split_text(text)
```

---

## 3. 임베딩 생성

### Sentence Transformers

```python
from sentence_transformers import SentenceTransformer

# 모델 로드
model = SentenceTransformer('all-MiniLM-L6-v2')

# 임베딩 생성
texts = ["Hello world", "How are you?"]
embeddings = model.encode(texts)

print(embeddings.shape)  # (2, 384)
```

### HuggingFace 임베딩

```python
from transformers import AutoTokenizer, AutoModel
import torch

tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

def get_embeddings(texts):
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')

    with torch.no_grad():
        outputs = model(**inputs)

    # Mean pooling
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings.numpy()
```

### OpenAI 임베딩

```python
from openai import OpenAI

client = OpenAI()

def get_openai_embeddings(texts, model="text-embedding-3-small"):
    response = client.embeddings.create(input=texts, model=model)
    return [r.embedding for r in response.data]
```

---

## 4. 벡터 검색

### 코사인 유사도

```python
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def search(query_embedding, document_embeddings, top_k=5):
    similarities = cosine_similarity([query_embedding], document_embeddings)[0]
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    return top_indices, similarities[top_indices]

# 사용
query_emb = model.encode(["What is machine learning?"])[0]
doc_embs = model.encode(documents)

indices, scores = search(query_emb, doc_embs, top_k=3)
```

### FAISS 사용

```python
import faiss
import numpy as np

# 인덱스 생성
dimension = 384  # 임베딩 차원
index = faiss.IndexFlatIP(dimension)  # Inner Product (코사인 유사도용 정규화 필요)

# 정규화 후 추가
embeddings = np.array(embeddings).astype('float32')
faiss.normalize_L2(embeddings)
index.add(embeddings)

# 검색
query_emb = model.encode(["query"])[0].astype('float32').reshape(1, -1)
faiss.normalize_L2(query_emb)

distances, indices = index.search(query_emb, k=5)
```

---

## 5. 간단한 RAG 구현

```python
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import numpy as np

class SimpleRAG:
    def __init__(self, embedding_model='all-MiniLM-L6-v2'):
        self.embed_model = SentenceTransformer(embedding_model)
        self.client = OpenAI()
        self.documents = []
        self.embeddings = None

    def add_documents(self, documents):
        """문서 추가 및 임베딩"""
        self.documents.extend(documents)
        self.embeddings = self.embed_model.encode(self.documents)

    def search(self, query, top_k=3):
        """관련 문서 검색"""
        query_emb = self.embed_model.encode([query])[0]

        # 코사인 유사도
        similarities = np.dot(self.embeddings, query_emb) / (
            np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_emb)
        )

        top_indices = np.argsort(similarities)[-top_k:][::-1]
        return [self.documents[i] for i in top_indices]

    def generate(self, query, top_k=3):
        """RAG 답변 생성"""
        # 검색
        relevant_docs = self.search(query, top_k)
        context = "\n\n".join(relevant_docs)

        # 프롬프트 구성
        prompt = f"""Answer the question based on the context below.

Context:
{context}

Question: {query}

Answer:"""

        # LLM 호출
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )

        return response.choices[0].message.content

# 사용
rag = SimpleRAG()
rag.add_documents([
    "Python is a programming language.",
    "Machine learning is a subset of AI.",
    "RAG combines retrieval with generation."
])

answer = rag.generate("What is RAG?")
print(answer)
```

---

## 6. 고급 RAG 기법

### Hybrid Search

```python
from rank_bm25 import BM25Okapi

class HybridRAG:
    def __init__(self):
        self.documents = []
        self.bm25 = None
        self.embeddings = None

    def add_documents(self, documents):
        self.documents = documents

        # BM25 (키워드 검색)
        tokenized = [doc.lower().split() for doc in documents]
        self.bm25 = BM25Okapi(tokenized)

        # 임베딩 (시맨틱 검색)
        self.embeddings = model.encode(documents)

    def hybrid_search(self, query, top_k=5, alpha=0.5):
        # BM25 점수
        bm25_scores = self.bm25.get_scores(query.lower().split())
        bm25_scores = bm25_scores / bm25_scores.max()  # 정규화

        # 임베딩 점수
        query_emb = model.encode([query])[0]
        embed_scores = cosine_similarity([query_emb], self.embeddings)[0]

        # 결합
        combined = alpha * embed_scores + (1 - alpha) * bm25_scores

        top_indices = np.argsort(combined)[-top_k:][::-1]
        return [self.documents[i] for i in top_indices]
```

### Query Expansion

```python
def expand_query(query, llm_client):
    """쿼리 확장으로 검색 성능 향상"""
    prompt = f"""Generate 3 alternative versions of this search query:
    Original: {query}

    Alternatives:
    1."""

    response = llm_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )

    expanded = [query] + parse_alternatives(response.choices[0].message.content)
    return expanded
```

### Reranking

```python
from sentence_transformers import CrossEncoder

class RAGWithReranker:
    def __init__(self):
        self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

    def search_and_rerank(self, query, candidates, top_k=3):
        # 1단계: 초기 검색 (후보 많이)
        initial_results = self.search(query, top_k=20)

        # 2단계: 리랭킹
        pairs = [[query, doc] for doc in initial_results]
        scores = self.reranker.predict(pairs)

        # 상위 k개 선택
        top_indices = np.argsort(scores)[-top_k:][::-1]
        return [initial_results[i] for i in top_indices]
```

### Multi-Query RAG

```python
def multi_query_rag(question, rag, num_queries=3):
    """여러 관점의 쿼리로 검색"""
    # 다양한 쿼리 생성
    prompt = f"""Generate {num_queries} different search queries for:
    Question: {question}

    Queries:"""

    queries = generate_queries(prompt)

    # 각 쿼리로 검색
    all_docs = set()
    for q in queries:
        docs = rag.search(q, top_k=3)
        all_docs.update(docs)

    return list(all_docs)
```

---

## 7. 청킹 전략 비교

| 전략 | 장점 | 단점 | 사용 시점 |
|------|------|------|----------|
| 고정 크기 | 구현 간단 | 문맥 단절 | 일반적인 텍스트 |
| 문장 기반 | 의미 단위 | 길이 불균일 | 구조화된 텍스트 |
| 시맨틱 | 의미 보존 | 계산 비용 | 고품질 필요 |
| 계층적 | 다단계 검색 | 복잡함 | 긴 문서 |

---

## 8. 평가 메트릭

### 검색 평가

```python
def calculate_recall_at_k(retrieved, relevant, k):
    """Recall@K 계산"""
    retrieved_k = set(retrieved[:k])
    relevant_set = set(relevant)
    return len(retrieved_k & relevant_set) / len(relevant_set)

def calculate_mrr(retrieved, relevant):
    """Mean Reciprocal Rank"""
    for i, doc in enumerate(retrieved):
        if doc in relevant:
            return 1 / (i + 1)
    return 0
```

### 생성 평가

```python
# RAGAS 라이브러리 사용
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision

results = evaluate(
    dataset,
    metrics=[faithfulness, answer_relevancy, context_precision]
)
```

---

## 정리

### RAG 체크리스트

```
□ 적절한 청킹 크기 선택
□ 임베딩 모델 선택 (도메인 고려)
□ 검색 top-k 튜닝
□ 프롬프트 최적화
□ 평가 메트릭 설정
```

### 핵심 코드

```python
# 임베딩
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(documents)

# 검색
query_emb = model.encode([query])[0]
similarities = cosine_similarity([query_emb], embeddings)

# 생성
context = "\n".join(relevant_docs)
prompt = f"Context: {context}\nQuestion: {query}\nAnswer:"
```

---

## 연습 문제

### 연습 문제 1: 청킹(Chunking) 전략 분석

아래 텍스트에 고정 크기 청킹(chunk_size=100, overlap=20)과 문장 기반 청킹을 각각 적용했을 때 출력을 비교하세요. 어느 전략이 더 나은 의미 단위를 생성하는지, 그 이유를 설명하세요.

```python
text = (
    "Machine learning is a subset of artificial intelligence. "
    "It enables systems to learn from data without being explicitly programmed. "
    "Deep learning uses neural networks with many layers. "
    "These networks can learn hierarchical representations of data. "
    "Natural language processing applies these techniques to text. "
    "Modern LLMs like GPT and BERT use transformer architectures. "
    "Transformers rely on self-attention mechanisms. "
    "They have revolutionized NLP tasks."
)
```

<details>
<summary>정답 보기</summary>

```python
# 고정 크기 청킹(chunk_size=100, overlap=20) 출력
def chunk_text(text, chunk_size=100, overlap=20):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks

fixed_chunks = chunk_text(text, 100, 20)
# 청크 0: "Machine learning is a subset of artificial intelligence. It enables systems to learn from data wi"
# 청크 1: "data without being explicitly programmed. Deep learning uses neural networks with many layers. These"
# 문제: 문장이 중간에 잘림!

# 문장 기반 청킹(max_sentences=2, overlap=1) 출력
import nltk
from nltk.tokenize import sent_tokenize
nltk.download('punkt', quiet=True)

def chunk_by_sentences(text, max_sentences=2, overlap_sentences=1):
    sentences = sent_tokenize(text)
    chunks = []
    step = max_sentences - overlap_sentences
    for i in range(0, len(sentences), step):
        chunk = ' '.join(sentences[i:i + max_sentences])
        chunks.append(chunk)
    return chunks

sent_chunks = chunk_by_sentences(text, max_sentences=2, overlap_sentences=1)
# 청크 0: "Machine learning is a subset of artificial intelligence. It enables systems to learn from data without being explicitly programmed."
# 청크 1: "It enables systems to learn from data without being explicitly programmed. Deep learning uses neural networks with many layers."
# 각 청크가 완전한 의미 단위를 형성함!
```

**문장 기반이 더 나은 이유:**
- 고정 크기 청킹은 문장을 중간에 잘라 의미 일관성을 파괴합니다
- "딥 러닝"에 대한 검색 쿼리(query)가 문장 중간에 시작하는 청크에 매칭될 수 있어 LLM이 컨텍스트(context)를 활용하기 어렵습니다
- 문장 기반 청킹은 완전한 사상(thought)을 보존하여 검색 정확도를 높입니다
- 오버랩(overlap) 1문장은 경계에서 컨텍스트 손실을 방지합니다

**고정 크기가 적합한 경우:** 명확한 문장 경계가 없는 매우 긴 문서(예: 로그, 트랜스크립트), 또는 토큰(token) 예산이 엄격하고 문장이 매우 긴 경우.
</details>

---

### 연습 문제 2: 평균 풀링(Mean Pooling) vs CLS 토큰

이 레슨의 `get_embeddings` 함수는 마지막 은닉 상태(hidden state)에 대해 평균 풀링을 사용합니다. 대안으로 `[CLS]` 토큰 표현만 사용하는 방법이 있습니다. 두 가지 방법을 모두 구현하고, 각각 언제 선호되는지 설명하세요.

```python
from transformers import AutoTokenizer, AutoModel
import torch

tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

texts = ["The cat sat on the mat.", "Artificial intelligence is transforming industries."]
inputs = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')

with torch.no_grad():
    outputs = model(**inputs)
    # outputs.last_hidden_state 형태: (batch, seq_len, hidden_dim)
```

<details>
<summary>정답 보기</summary>

```python
# 방법 1: CLS 토큰(첫 번째 토큰)
cls_embeddings = outputs.last_hidden_state[:, 0, :]
# 형태: (batch_size, hidden_dim) = (2, 384)

# 방법 2: 평균 풀링(패딩이 아닌 토큰에 대한 평균)
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state
    # 마스크(mask)를 임베딩 차원에 맞게 확장
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(
        token_embeddings.size()
    ).float()
    # 패딩이 아닌 토큰의 임베딩 합산 후 개수로 나눔
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
    sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
    return sum_embeddings / sum_mask

mean_embeddings = mean_pooling(outputs, inputs['attention_mask'])
# 형태: (batch_size, hidden_dim) = (2, 384)

# 방법 3: 최대 풀링(가장 강한 특징 포착)
max_embeddings = outputs.last_hidden_state.max(dim=1).values
```

**각 방법 사용 시기:**
| 방법 | 최적 사용 사례 | 비고 |
|------|--------------|------|
| CLS 토큰 | CLS로 파인튜닝된 모델 (예: BERT 분류) | CLS가 문장 의미를 인코딩하도록 학습되어야 함 |
| 평균 풀링 | 일반 문장 유사도, SentenceTransformers 모델 | 더 강건함, 모든 토큰을 균등하게 반영 |
| 최대 풀링 | 두드러진 특징 포착 | 핵심 구절이 전체 의미보다 중요할 때 유용 |

**중요:** 평균 풀링에서 어텐션 마스크(attention mask)는 필수입니다 — 없으면 패딩 토큰이 평균에 기여하여 짧은 문장의 임베딩 품질이 저하됩니다.
</details>

---

### 연습 문제 3: 하이브리드 검색(Hybrid Search) 알파 튜닝

`HybridRAG.hybrid_search` 메서드의 `alpha` 파라미터(parameter)는 시맨틱(semantic) 검색과 BM25 키워드 검색 간의 균형을 조절합니다. 다음 쿼리(query) 유형에 대해 권장 `alpha` 값(0=순수 BM25, 1=순수 시맨틱)과 그 이유를 설명하세요.

| 쿼리 | 권장 Alpha | 이유 |
|------|-----------|------|
| "Python 문법 오류 수정" | ? | ? |
| "의식이란 무엇인가?" | ? | ? |
| "RFC 7231 상태 코드" | ? | ? |
| "실패를 어떻게 극복하나요?" | ? | ? |

<details>
<summary>정답 보기</summary>

| 쿼리 | 권장 Alpha | 이유 |
|------|-----------|------|
| "Python 문법 오류 수정" | 0.3 | 기술적 쿼리는 키워드 매칭이 중요 ("문법 오류" 정확 매칭); 시맨틱 검색은 일반 Python 튜토리얼을 반환할 수 있음 |
| "의식이란 무엇인가?" | 0.9 | 철학적/개념적 쿼리는 시맨틱 이해가 필요; "의식"이라는 단어만으로는 "자아 인식", "감각질(qualia)" 등 관련 개념을 놓침 |
| "RFC 7231 상태 코드" | 0.1 | 정확한 식별자("RFC 7231")가 매칭되어야 함; 시맨틱 검색은 모든 HTTP 문서를 반환할 수 있음 |
| "실패를 어떻게 극복하나요?" | 0.8 | 감성적/뉘앙스 있는 쿼리; 시맨틱 검색이 정확한 단어가 없어도 회복력/성장 마인드셋 관련 내용을 찾음 |

```python
# 실용적인 자동 추정 휴리스틱:
def estimate_alpha(query: str) -> float:
    """쿼리 특성에 따라 alpha를 추정합니다."""
    tokens = query.lower().split()

    # 높은 키워드 특이성 신호
    has_numbers = any(t.isdigit() or any(c.isdigit() for c in t) for t in tokens)
    has_technical = any(t in ['오류', 'error', 'rfc', 'api', '수정', 'fix'] for t in tokens)
    is_short = len(tokens) <= 3

    # 높은 시맨틱 신호
    is_question = query.lower().startswith(('무엇', '어떻게', '왜', '설명', 'what', 'how', 'why'))
    is_long = len(tokens) >= 7

    score = 0.5  # 기본값: 균형
    if has_numbers or has_technical: score -= 0.3
    if is_short: score -= 0.1
    if is_question: score += 0.2
    if is_long: score += 0.1

    return max(0.0, min(1.0, score))
```

실제 운영 환경에서는 레이블이 있는 평가 셋(evaluation set)에서 그리드 서치(grid search)를 통해 alpha를 튜닝합니다.
</details>

---

### 연습 문제 4: Recall@K로 RAG 평가

`calculate_recall_at_k` 함수를 구현하고 소규모 테스트 세트에서 두 가지 검색 구성을 평가하세요. 그리고 RAG에서 좋은 Recall@K 값이 무엇인지 설명하세요.

```python
# 테스트 데이터
queries = [
    "What is machine learning?",
    "How does BERT work?",
    "What is a transformer?",
]

# 정답: 각 쿼리에 대해 관련성 있는 문서 인덱스
ground_truth = {
    "What is machine learning?": [0, 2],
    "How does BERT work?": [1, 3],
    "What is a transformer?": [1, 3, 4],
}

# 각 쿼리에 대한 검색 결과(인덱스) — 비교할 두 시스템
system_a = {
    "What is machine learning?": [0, 5, 2, 7, 1],
    "How does BERT work?": [3, 6, 1, 8, 2],
    "What is a transformer?": [4, 1, 6, 3, 9],
}

system_b = {
    "What is machine learning?": [5, 7, 6, 8, 0],
    "How does BERT work?": [6, 8, 2, 9, 1],
    "What is a transformer?": [6, 9, 7, 8, 4],
}
```

<details>
<summary>정답 보기</summary>

```python
def calculate_recall_at_k(retrieved: list, relevant: list, k: int) -> float:
    """
    Recall@K = |retrieved[:k] ∩ relevant| / |relevant|
    측정 대상: 관련 문서 중 상위 K 결과에서 찾은 비율
    """
    retrieved_k = set(retrieved[:k])
    relevant_set = set(relevant)
    if not relevant_set:
        return 0.0
    return len(retrieved_k & relevant_set) / len(relevant_set)


def evaluate_system(system: dict, ground_truth: dict, k: int) -> float:
    """모든 쿼리에 대한 평균 Recall@K."""
    recalls = []
    for query, retrieved in system.items():
        relevant = ground_truth[query]
        recalls.append(calculate_recall_at_k(retrieved, relevant, k))
    return sum(recalls) / len(recalls)


# K=3과 K=5에서 평가
for k in [3, 5]:
    r_a = evaluate_system(system_a, ground_truth, k)
    r_b = evaluate_system(system_b, ground_truth, k)
    print(f"Recall@{k}: 시스템 A = {r_a:.3f}, 시스템 B = {r_b:.3f}")

# 시스템 A:
# Recall@3: 0.833  (A가 관련 문서를 일찍 찾음)
# Recall@5: 1.000  (A가 K=5까지 모든 관련 문서를 찾음)

# 시스템 B:
# Recall@3: 0.333  (B가 상위 3개에서 많은 관련 문서를 놓침)
# Recall@5: 0.667  (B가 K=5에서도 일부를 놓침)
```

**RAG에서 좋은 Recall@K란?**
- **Recall@K 목표:** 최대한 높게 — 관련 문서가 검색되지 않으면 LLM이 답변에 포함시킬 수 없음
- **일반적인 목표치:** 운영 시스템 기준 Recall@3 ≥ 0.7, Recall@5 ≥ 0.85
- **트레이드오프(trade-off):** K가 높을수록 재현율은 높아지지만 컨텍스트 길이가 증가 (비용 + LLM 어텐션(attention) 분산)
- **일반적인 관행:** 검색 시 K=5-10을 사용하고, 리랭킹(reranking) 후 상위 3개만 LLM 프롬프트에 넣음
- Recall은 커버리지를 측정; 순위가 중요할 때는 MRR이나 NDCG를 사용
</details>

---

## 다음 단계

[LangChain 기초](./13_LangChain_Basics.md)에서 LangChain 프레임워크를 학습합니다.
