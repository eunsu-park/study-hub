# 09. RAG 기초

## 학습 목표

- RAG (Retrieval-Augmented Generation) 이해
- 문서 임베딩과 검색
- 청킹 전략
- RAG 파이프라인 구현

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

```
┌─────────────────────────────────────────────────────┐
│                     RAG Pipeline                     │
├─────────────────────────────────────────────────────┤
│                                                      │
│   질문 ──▶ 임베딩 ──▶ 벡터 검색 ──▶ 관련 문서      │
│                           │                          │
│                           ▼                          │
│               질문 + 문서 ──▶ LLM ──▶ 답변          │
│                                                      │
└─────────────────────────────────────────────────────┘
```

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

[LangChain 기초](./10_LangChain_Basics.md)에서 LangChain 프레임워크를 학습합니다.
