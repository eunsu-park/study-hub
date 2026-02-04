# 23. Advanced RAG

## 개요

기본 RAG를 넘어 더 정교한 검색과 생성 전략을 다룹니다. Agentic RAG, Multi-hop Reasoning, HyDE, RAPTOR 등 최신 기법을 학습합니다.

---

## 1. RAG 한계와 고급 기법

### 1.1 기본 RAG의 한계

```
기본 RAG 문제점:
┌─────────────────────────────────────────────────────────┐
│  1. 단일 검색 한계                                       │
│     - 복잡한 질문에 한 번의 검색으로 부족                 │
│     - 다단계 추론 필요                                   │
│                                                         │
│  2. 검색-질문 불일치                                     │
│     - 질문과 문서 스타일 차이                            │
│     - Embedding 유사도의 한계                            │
│                                                         │
│  3. 컨텍스트 길이 제한                                   │
│     - 관련 문서가 많을 때 처리 어려움                    │
│     - 중요 정보 누락 가능                                │
│                                                         │
│  4. 최신성/정확성                                        │
│     - 오래된 정보                                        │
│     - 신뢰도 검증 어려움                                 │
└─────────────────────────────────────────────────────────┘
```

### 1.2 고급 RAG 기법 분류

```
고급 RAG 기법:
┌─────────────────────────────────────────────────────────┐
│  Pre-Retrieval                                          │
│  ├── Query Transformation (HyDE, Query Expansion)       │
│  └── Query Routing                                      │
│                                                         │
│  Retrieval                                              │
│  ├── Hybrid Search (Dense + Sparse)                     │
│  ├── Multi-step Retrieval                               │
│  └── Hierarchical Retrieval (RAPTOR)                    │
│                                                         │
│  Post-Retrieval                                         │
│  ├── Reranking                                          │
│  ├── Context Compression                                │
│  └── Self-Reflection                                    │
│                                                         │
│  Generation                                             │
│  ├── Chain-of-Thought RAG                               │
│  └── Agentic RAG                                        │
└─────────────────────────────────────────────────────────┘
```

---

## 2. Query Transformation

### 2.1 HyDE (Hypothetical Document Embeddings)

```
HyDE 아이디어:
┌─────────────────────────────────────────────────────────┐
│  Query: "What is the capital of France?"                │
│                                                         │
│  기존: query embedding으로 직접 검색                    │
│        (질문 ↔ 문서 스타일 차이)                        │
│                                                         │
│  HyDE: LLM으로 가상 문서 생성 후 검색                   │
│        Query → "Paris is the capital of France..."      │
│        → 이 가상 문서의 embedding으로 검색              │
│        (문서 ↔ 문서 스타일 일치)                        │
└─────────────────────────────────────────────────────────┘
```

```python
from langchain.chains import HypotheticalDocumentEmbedder
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings

class HyDERetriever:
    """HyDE 검색기"""

    def __init__(self, llm, embeddings, vectorstore):
        self.llm = llm
        self.embeddings = embeddings
        self.vectorstore = vectorstore

    def generate_hypothetical_document(self, query: str) -> str:
        """가상 문서 생성"""
        prompt = f"""Write a short passage that would answer the following question.
The passage should be factual and informative.

Question: {query}

Passage:"""

        response = self.llm.invoke(prompt)
        return response

    def retrieve(self, query: str, k: int = 5) -> list:
        """HyDE 검색"""
        # 1. 가상 문서 생성
        hypothetical_doc = self.generate_hypothetical_document(query)

        # 2. 가상 문서 임베딩
        doc_embedding = self.embeddings.embed_query(hypothetical_doc)

        # 3. 유사 문서 검색
        results = self.vectorstore.similarity_search_by_vector(
            doc_embedding, k=k
        )

        return results


# LangChain 내장 HyDE
def setup_hyde_chain():
    base_embeddings = OpenAIEmbeddings()
    llm = OpenAI(temperature=0)

    embeddings = HypotheticalDocumentEmbedder.from_llm(
        llm, base_embeddings, "web_search"
    )

    return embeddings
```

### 2.2 Query Expansion

```python
class QueryExpander:
    """쿼리 확장"""

    def __init__(self, llm):
        self.llm = llm

    def expand_query(self, query: str, num_variations: int = 3) -> list:
        """쿼리를 여러 변형으로 확장"""
        prompt = f"""Generate {num_variations} different versions of the following question.
Each version should ask the same thing but use different words or perspectives.

Original question: {query}

Variations:
1."""

        response = self.llm.invoke(prompt)

        # 파싱
        variations = [query]  # 원본 포함
        for line in response.split("\n"):
            line = line.strip()
            if line and line[0].isdigit():
                # "1. question" 형식
                variation = line.split(".", 1)[-1].strip()
                variations.append(variation)

        return variations[:num_variations + 1]

    def retrieve_with_expansion(
        self,
        query: str,
        retriever,
        k: int = 5
    ) -> list:
        """확장된 쿼리로 검색"""
        variations = self.expand_query(query)

        all_docs = []
        seen = set()

        for variation in variations:
            docs = retriever.get_relevant_documents(variation)
            for doc in docs:
                doc_id = hash(doc.page_content)
                if doc_id not in seen:
                    seen.add(doc_id)
                    all_docs.append(doc)

        # 상위 k개 반환 (RRF 또는 기타 방법으로 정렬)
        return all_docs[:k]
```

---

## 3. Agentic RAG

### 3.1 개념

```
Agentic RAG:
┌─────────────────────────────────────────────────────────┐
│  LLM Agent가 검색 도구를 동적으로 사용                   │
│                                                         │
│  Agent Loop:                                            │
│  1. 질문 분석                                           │
│  2. 필요한 정보 결정                                     │
│  3. 검색 도구 호출 (선택적, 반복 가능)                   │
│  4. 결과 평가                                           │
│  5. 추가 검색 필요? → 반복                              │
│  6. 최종 답변 생성                                       │
│                                                         │
│  vs 기본 RAG:                                           │
│  Query → Retrieve → Generate (고정된 파이프라인)        │
└─────────────────────────────────────────────────────────┘
```

### 3.2 구현

```python
from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate

class AgenticRAG:
    """Agentic RAG 시스템"""

    def __init__(self, llm, vectorstore, web_search=None):
        self.llm = llm
        self.vectorstore = vectorstore
        self.web_search = web_search

        self.tools = self._setup_tools()
        self.agent = self._create_agent()

    def _setup_tools(self) -> list:
        """도구 설정"""
        tools = [
            Tool(
                name="search_knowledge_base",
                func=self._search_kb,
                description="Search the internal knowledge base for relevant information. Use this for company-specific or domain-specific questions."
            ),
            Tool(
                name="search_web",
                func=self._search_web,
                description="Search the web for current information. Use this for recent events or general knowledge."
            ),
            Tool(
                name="lookup_specific",
                func=self._lookup_specific,
                description="Look up specific facts or definitions. Use this when you need precise information."
            )
        ]
        return tools

    def _search_kb(self, query: str) -> str:
        """지식 베이스 검색"""
        docs = self.vectorstore.similarity_search(query, k=3)
        return "\n\n".join([doc.page_content for doc in docs])

    def _search_web(self, query: str) -> str:
        """웹 검색 (외부 API 필요)"""
        if self.web_search:
            return self.web_search.run(query)
        return "Web search not available."

    def _lookup_specific(self, query: str) -> str:
        """특정 정보 조회"""
        docs = self.vectorstore.similarity_search(query, k=1)
        if docs:
            return docs[0].page_content
        return "No specific information found."

    def _create_agent(self):
        """ReAct Agent 생성"""
        prompt = PromptTemplate.from_template("""Answer the following question using the available tools.
Think step by step about what information you need.

Question: {input}

You have access to these tools:
{tools}

Use the following format:
Thought: What do I need to find out?
Action: tool_name
Action Input: the input to the tool
Observation: the result of the tool
... (repeat as needed)
Thought: I now have enough information
Final Answer: the final answer

Begin!

{agent_scratchpad}""")

        agent = create_react_agent(self.llm, self.tools, prompt)
        return AgentExecutor(agent=agent, tools=self.tools, verbose=True)

    def query(self, question: str) -> str:
        """질문 처리"""
        result = self.agent.invoke({"input": question})
        return result["output"]


# 사용 예시
def agentic_rag_example():
    from langchain.llms import OpenAI
    from langchain.vectorstores import Chroma

    llm = OpenAI(temperature=0)
    vectorstore = Chroma(...)  # 설정 필요

    rag = AgenticRAG(llm, vectorstore)

    # 복잡한 질문
    answer = rag.query(
        "Compare our company's revenue growth in 2023 with the industry average"
    )
    print(answer)
```

---

## 4. Multi-hop Reasoning

### 4.1 개념

```
Multi-hop Reasoning:
┌─────────────────────────────────────────────────────────┐
│  질문: "바이든의 출생지의 인구는?"                       │
│                                                         │
│  Hop 1: "바이든의 출생지는?" → "스크랜턴, PA"           │
│  Hop 2: "스크랜턴의 인구는?" → "76,328명"               │
│                                                         │
│  최종 답변: "76,328명"                                   │
│                                                         │
│  단일 검색으로는 직접 답을 찾기 어려움                   │
└─────────────────────────────────────────────────────────┘
```

### 4.2 구현

```python
class MultiHopRAG:
    """Multi-hop Reasoning RAG"""

    def __init__(self, llm, retriever, max_hops: int = 3):
        self.llm = llm
        self.retriever = retriever
        self.max_hops = max_hops

    def decompose_question(self, question: str) -> list:
        """질문을 하위 질문으로 분해"""
        prompt = f"""Break down the following complex question into simpler sub-questions.
Each sub-question should be answerable independently.

Question: {question}

Sub-questions (one per line):"""

        response = self.llm.invoke(prompt)
        sub_questions = [q.strip() for q in response.split("\n") if q.strip()]
        return sub_questions

    def answer_with_hops(self, question: str) -> dict:
        """다단계 추론으로 답변"""
        reasoning_chain = []
        context = ""

        for hop in range(self.max_hops):
            # 현재 컨텍스트로 다음 질문 결정
            if hop == 0:
                current_query = question
            else:
                current_query = self._generate_follow_up(
                    question, context, reasoning_chain
                )

            if current_query is None:
                break

            # 검색
            docs = self.retriever.get_relevant_documents(current_query)
            new_context = "\n".join([doc.page_content for doc in docs])

            # 중간 답변 생성
            intermediate_answer = self._generate_intermediate_answer(
                current_query, new_context
            )

            reasoning_chain.append({
                "hop": hop + 1,
                "query": current_query,
                "answer": intermediate_answer
            })

            context += f"\n{intermediate_answer}"

            # 충분한 정보가 있는지 확인
            if self._has_enough_info(question, context):
                break

        # 최종 답변
        final_answer = self._generate_final_answer(question, reasoning_chain)

        return {
            "question": question,
            "reasoning_chain": reasoning_chain,
            "final_answer": final_answer
        }

    def _generate_follow_up(self, original_q, context, chain) -> str:
        """후속 질문 생성"""
        chain_text = "\n".join([
            f"Q: {step['query']}\nA: {step['answer']}"
            for step in chain
        ])

        prompt = f"""Based on the original question and what we've learned so far,
what additional information do we need?

Original question: {original_q}

What we've found:
{chain_text}

If we have enough information to answer, respond with "DONE".
Otherwise, provide the next question to search for:"""

        response = self.llm.invoke(prompt)

        if "DONE" in response.upper():
            return None
        return response.strip()

    def _generate_intermediate_answer(self, query, context) -> str:
        """중간 답변 생성"""
        prompt = f"""Based on the following context, answer the question briefly.

Context: {context}

Question: {query}

Answer:"""

        return self.llm.invoke(prompt)

    def _has_enough_info(self, question, context) -> bool:
        """충분한 정보가 있는지 확인"""
        prompt = f"""Can you answer the following question based on this information?

Question: {question}
Information: {context}

Answer YES or NO:"""

        response = self.llm.invoke(prompt)
        return "YES" in response.upper()

    def _generate_final_answer(self, question, chain) -> str:
        """최종 답변 생성"""
        chain_text = "\n".join([
            f"Step {step['hop']}: {step['query']} → {step['answer']}"
            for step in chain
        ])

        prompt = f"""Based on the reasoning chain below, provide a final answer.

Question: {question}

Reasoning:
{chain_text}

Final Answer:"""

        return self.llm.invoke(prompt)
```

---

## 5. RAPTOR (Recursive Abstractive Processing for Tree-Organized Retrieval)

### 5.1 개념

```
RAPTOR 구조:
┌─────────────────────────────────────────────────────────┐
│  Level 3 (최고 수준 요약)                               │
│  ┌──────────────────────────────────┐                   │
│  │     Abstract Summary              │                  │
│  └──────────────────────────────────┘                   │
│              ↑                                          │
│  Level 2 (클러스터 요약)                                │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐          │
│  │ Summary1 │    │ Summary2 │    │ Summary3 │          │
│  └──────────┘    └──────────┘    └──────────┘          │
│      ↑   ↑          ↑   ↑          ↑   ↑              │
│  Level 1 (청크 클러스터링)                              │
│  [C1][C2][C3]    [C4][C5][C6]    [C7][C8][C9]          │
│      ↑   ↑   ↑      ↑   ↑   ↑      ↑   ↑   ↑          │
│  Level 0 (원본 청크)                                    │
│  [Chunk1][Chunk2]...[ChunkN]                           │
└─────────────────────────────────────────────────────────┘

검색: 여러 레벨에서 동시에 검색하여 다양한 추상화 수준의 정보 획득
```

### 5.2 구현

```python
from sklearn.cluster import KMeans
import numpy as np

class RAPTOR:
    """RAPTOR 계층적 검색"""

    def __init__(self, llm, embeddings, num_levels: int = 3):
        self.llm = llm
        self.embeddings = embeddings
        self.num_levels = num_levels
        self.tree = {}

    def build_tree(self, documents: list, cluster_size: int = 5):
        """RAPTOR 트리 구축"""
        # Level 0: 원본 청크
        self.tree[0] = documents
        current_docs = documents

        for level in range(1, self.num_levels):
            # 임베딩 계산
            texts = [doc.page_content for doc in current_docs]
            embeddings = self.embeddings.embed_documents(texts)

            # 클러스터링
            n_clusters = max(len(current_docs) // cluster_size, 1)
            kmeans = KMeans(n_clusters=n_clusters)
            clusters = kmeans.fit_predict(embeddings)

            # 클러스터별 요약
            summaries = []
            for cluster_id in range(n_clusters):
                cluster_docs = [
                    doc for doc, c in zip(current_docs, clusters)
                    if c == cluster_id
                ]
                summary = self._summarize_cluster(cluster_docs)
                summaries.append(summary)

            self.tree[level] = summaries
            current_docs = summaries

    def _summarize_cluster(self, docs: list) -> str:
        """클러스터 요약"""
        combined_text = "\n\n".join([doc.page_content for doc in docs])

        prompt = f"""Summarize the following texts into a concise summary that captures the key information.

Texts:
{combined_text}

Summary:"""

        summary = self.llm.invoke(prompt)

        # Document 객체로 래핑
        from langchain.schema import Document
        return Document(page_content=summary)

    def retrieve(self, query: str, k_per_level: int = 2) -> list:
        """계층적 검색"""
        all_results = []

        for level, docs in self.tree.items():
            # 각 레벨에서 검색
            texts = [doc.page_content for doc in docs]
            query_embedding = self.embeddings.embed_query(query)
            doc_embeddings = self.embeddings.embed_documents(texts)

            # 코사인 유사도
            similarities = np.dot(doc_embeddings, query_embedding)
            top_indices = np.argsort(similarities)[-k_per_level:]

            for idx in top_indices:
                all_results.append({
                    "level": level,
                    "document": docs[idx],
                    "score": similarities[idx]
                })

        # 점수로 정렬
        all_results.sort(key=lambda x: x["score"], reverse=True)
        return all_results
```

---

## 6. ColBERT (Contextualized Late Interaction)

### 6.1 개념

```
ColBERT vs Dense Retrieval:
┌─────────────────────────────────────────────────────────┐
│  Dense Retrieval (bi-encoder):                          │
│  Query → [CLS] embedding                                │
│  Doc   → [CLS] embedding                                │
│  Score = dot(query_emb, doc_emb)                        │
│  문제: 단일 벡터로 복잡한 의미 표현 어려움               │
│                                                         │
│  ColBERT (late interaction):                            │
│  Query → [q1, q2, ..., qn] (토큰별 임베딩)              │
│  Doc   → [d1, d2, ..., dm] (토큰별 임베딩)              │
│  Score = Σᵢ maxⱼ sim(qᵢ, dⱼ)                           │
│  장점: 토큰 수준 매칭으로 더 정밀한 검색                 │
└─────────────────────────────────────────────────────────┘
```

### 6.2 사용

```python
from colbert import Indexer, Searcher
from colbert.infra import Run, RunConfig, ColBERTConfig

class ColBERTRetriever:
    """ColBERT 검색기"""

    def __init__(self, index_name: str = "my_index"):
        self.index_name = index_name
        self.config = ColBERTConfig(
            nbits=2,
            doc_maxlen=300,
            query_maxlen=32
        )

    def build_index(self, documents: list, collection_path: str):
        """인덱스 구축"""
        # 문서를 파일로 저장
        with open(collection_path, 'w') as f:
            for doc in documents:
                f.write(doc + "\n")

        with Run().context(RunConfig(nranks=1)):
            indexer = Indexer(
                checkpoint="colbert-ir/colbertv2.0",
                config=self.config
            )
            indexer.index(
                name=self.index_name,
                collection=collection_path
            )

    def search(self, query: str, k: int = 10) -> list:
        """검색"""
        with Run().context(RunConfig(nranks=1)):
            searcher = Searcher(index=self.index_name)
            results = searcher.search(query, k=k)

        return results


# RAGatouille (더 쉬운 ColBERT 래퍼)
def colbert_with_ragatouille():
    from ragatouille import RAGPretrainedModel

    rag = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")

    # 인덱싱
    rag.index(
        collection=[
            "Document 1 content...",
            "Document 2 content..."
        ],
        index_name="my_index"
    )

    # 검색
    results = rag.search("my query", k=5)
    return results
```

---

## 7. Self-RAG (Self-Reflective RAG)

```python
class SelfRAG:
    """Self-RAG: 자기 성찰 RAG"""

    def __init__(self, llm, retriever):
        self.llm = llm
        self.retriever = retriever

    def query(self, question: str) -> dict:
        """Self-RAG 질의"""
        # 1. 검색 필요성 판단
        needs_retrieval = self._assess_retrieval_need(question)

        if not needs_retrieval:
            # 검색 없이 직접 답변
            answer = self._generate_without_retrieval(question)
            return {"answer": answer, "retrieval_used": False}

        # 2. 검색
        docs = self.retriever.get_relevant_documents(question)

        # 3. 관련성 평가 (각 문서별)
        relevant_docs = []
        for doc in docs:
            if self._is_relevant(question, doc):
                relevant_docs.append(doc)

        # 4. 답변 생성
        answer = self._generate_with_context(question, relevant_docs)

        # 5. 답변 품질 평가
        is_supported = self._check_support(answer, relevant_docs)
        is_useful = self._check_usefulness(question, answer)

        # 6. 필요시 재시도
        if not is_supported or not is_useful:
            answer = self._refine_answer(question, relevant_docs, answer)

        return {
            "answer": answer,
            "retrieval_used": True,
            "relevant_docs": relevant_docs,
            "is_supported": is_supported,
            "is_useful": is_useful
        }

    def _assess_retrieval_need(self, question: str) -> bool:
        """검색 필요성 평가"""
        prompt = f"""Determine if external knowledge is needed to answer this question.

Question: {question}

Answer YES if retrieval is needed, NO if you can answer from general knowledge:"""

        response = self.llm.invoke(prompt)
        return "YES" in response.upper()

    def _is_relevant(self, question: str, doc) -> bool:
        """문서 관련성 평가"""
        prompt = f"""Is this document relevant to the question?

Question: {question}
Document: {doc.page_content[:500]}

Answer RELEVANT or IRRELEVANT:"""

        response = self.llm.invoke(prompt)
        return "RELEVANT" in response.upper()

    def _check_support(self, answer: str, docs: list) -> bool:
        """답변이 문서에 의해 뒷받침되는지 확인"""
        context = "\n".join([doc.page_content for doc in docs])

        prompt = f"""Is this answer supported by the given context?

Context: {context}
Answer: {answer}

Respond SUPPORTED or NOT_SUPPORTED:"""

        response = self.llm.invoke(prompt)
        return "SUPPORTED" in response.upper()

    def _check_usefulness(self, question: str, answer: str) -> bool:
        """답변 유용성 확인"""
        prompt = f"""Does this answer actually address the question?

Question: {question}
Answer: {answer}

Respond USEFUL or NOT_USEFUL:"""

        response = self.llm.invoke(prompt)
        return "USEFUL" in response.upper()
```

---

## 핵심 정리

### Advanced RAG 기법
```
1. HyDE: 가상 문서로 검색 품질 향상
2. Query Expansion: 다양한 쿼리로 검색
3. Agentic RAG: LLM Agent의 동적 검색
4. Multi-hop: 다단계 추론
5. RAPTOR: 계층적 요약 트리
6. ColBERT: 토큰 수준 late interaction
7. Self-RAG: 자기 성찰 및 검증
```

### 선택 가이드
```
단순 QA → 기본 RAG
복잡한 질문 → Multi-hop + Agentic
긴 문서 → RAPTOR
정밀 검색 → ColBERT
품질 중요 → Self-RAG
```

---

## 참고 자료

1. Gao et al. (2022). "Precise Zero-Shot Dense Retrieval without Relevance Labels" (HyDE)
2. Sarthi et al. (2024). "RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval"
3. Khattab et al. (2020). "ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction"
4. Asai et al. (2023). "Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection"
