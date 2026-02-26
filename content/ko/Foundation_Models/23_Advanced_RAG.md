# 23. Advanced RAG

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 기본 RAG의 한계(단일 검색, 검색-질문 불일치, 컨텍스트 길이 제한)를 파악하고 고급 기법들이 이를 어떻게 해결하는지 설명할 수 있다
2. 검색 관련성을 높이기 위해 HyDE(Hypothetical Document Embeddings)와 쿼리 확장(query expansion) 등 쿼리 변환(query transformation) 기법을 구현할 수 있다
3. 복잡한 질문을 처리하기 위해 RAPTOR와 멀티홉 추론(multi-hop reasoning)을 활용한 다단계 및 계층적 검색(hierarchical retrieval) 파이프라인을 구축할 수 있다
4. 재순위 매기기(reranking), 컨텍스트 압축(context compression), 자기 반성(self-reflection) 등 검색 후 처리(post-retrieval) 기법을 적용하여 최종 답변 품질을 향상시킬 수 있다
5. 생성 과정에서 언제, 무엇을 검색할지 LLM이 동적으로 결정하는 에이전틱 RAG(Agentic RAG) 시스템을 설계할 수 있다

---

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

---

## 연습 문제

### 연습 문제 1: HyDE vs. 직접 쿼리 — 언제 도움이 되는가?
HyDE는 직접 쿼리로 검색하는 대신 검색 전에 가상의 문서(hypothetical document)를 생성합니다. 두 가지 구체적인 예시를 사용하여 HyDE가 도움이 되는 경우와 그렇지 않은 경우를 분석하세요.

다음 검색 태스크를 고려하세요:
- **태스크 A**: "아스피린의 부작용은 무엇인가요?" (사실적, 잘 문서화된)
- **태스크 B**: "계약 해지 권리에 관한 공식적인 법률 어조로 작성된 문서 찾기"

<details>
<summary>정답 보기</summary>

**HyDE가 도움이 되는 이유**:

핵심 통찰은 **질문 임베딩과 문서 임베딩이 임베딩 공간의 다른 영역을 차지한다**는 것입니다. 질문은 짧고, 의문형이며, 희소합니다("아스피린의 부작용은?"). 문서는 길고, 서술형이며, 밀집되어 있습니다("아스피린은 위장관 출혈, 이명을 일으킬 수 있습니다..."). 직접 쿼리-문서 유사도 검색은 임베딩 모델이 이를 다른 언어 스타일로 보기 때문에 관련 문서를 놓치는 경우가 많습니다.

HyDE는 먼저 가상의 답변 문서를 생성한 다음 *그것*을 검색 쿼리로 임베딩함으로써 이 격차를 해소합니다. 이제 문서-문서 유사도 검색을 수행하게 되는데, 임베딩 모델은 이를 더 잘 처리합니다.

**태스크 A: "아스피린의 부작용은 무엇인가요?"**
```
직접 쿼리 임베딩:
"아스피린의 부작용은 무엇인가요?" → [0.12, -0.34, ...]
타겟 문서:
"아스피린(아세틸살리실산)은 위장관 자극을..." → [0.09, -0.31, ...]
유사도: 보통 (다른 스타일)

HyDE 가상 문서:
"아스피린의 일반적인 부작용으로는 위장관 자극,
고용량 시 이명이 포함됩니다..." → [0.08, -0.32, ...]
타겟 문서:
"아스피린(아세틸살리실산)은 위장관 자극을..." → [0.09, -0.31, ...]
유사도: 높음 (같은 서술형 스타일)
```
**HyDE가 도움됨** — 가상의 답변이 타겟 문서의 스타일과 내용을 밀접하게 반영하기 때문에 사실적 질문에서 효과적입니다.

**태스크 B: "계약 해지 권리에 관한 공식적인 법률 어조로 작성된 문서 찾기"**
- 이 쿼리는 이미 사실적 질문이 아닌 *문서 스타일*을 설명하고 있습니다.
- HyDE는 *가상의 법률 문서*를 생성하지만, 이는 위험을 수반합니다: LLM이 실제 코퍼스에 없는 특정 법률 조항을 환각할 수 있습니다.
- 스타일 매칭 검색의 경우, 환각 위험 없이 하이브리드 검색(dense + BM25)이 더 잘 작동할 수 있습니다.

**HyDE가 도움이 되지 않거나 해로운 경우**:
1. **매우 구체적인 사실 조회**: LLM이 틀린 세부 사항을 확신하며 환각할 수 있는 경우 (예: "X사의 2023년 4분기 매출을 찾아라" — 가상 문서가 그럴듯하지만 틀린 숫자를 생성할 수 있습니다).
2. **잘 매칭된 코퍼스에서 매우 짧은 사실 쿼리**: 직접 임베딩 유사도가 이미 잘 작동하는 경우.
3. **시간에 민감한 쿼리**: 가상 문서가 오래된 훈련 지식을 반영할 수 있는 경우.

</details>

### 연습 문제 2: 멀티홉 검색(Multi-hop Retrieval) 설계
법률 연구 어시스턴트가 다음 질문에 답해야 합니다: "2019년 Smith v. Johnson 사건의 판사가 2020년 지식재산권 분쟁 관련 사건에도 관여했나요?"

이 질문은 최소 2단계의 검색이 필요합니다. 각 단계에서 무엇을 검색하고 어떤 정보를 다음 단계로 전달하는지 명시하여 멀티홉 검색 파이프라인을 설계하세요.

<details>
<summary>정답 보기</summary>

**단일 검색이 실패하는 이유**:
"2019년 Smith v. Johnson 사건의 판사가 2020년 IP 사건에 관여" 같은 단일 쿼리는 이 모든 정보를 포함하는 문서가 없기 때문에 실패할 것입니다. 판사 이름을 미리 알 수 없으므로 쿼리가 충분히 구체적이지 못합니다.

**멀티홉 파이프라인**:

```
1단계: 사건 세부 정보 검색
  쿼리: "Smith v. Johnson 2019 case"
  검색됨: 판사명 = "판사 Robert Thompson"이 포함된 사건 기록
  추출 정보: judge_name = "Robert Thompson"

2단계: 추출된 정보를 이용한 두 번째 검색
  쿼리: "Judge Robert Thompson intellectual property 2020"
  검색됨: 판사 Thompson이 언급된 2020년 IP 사건들
  답변: "예, Thompson 판사는 2020년 DataTech v. InnoSoft를
         비롯한 두 건의 IP 분쟁을 담당했습니다."
```

**구현 패턴**:
```python
class MultiHopRetriever:
    def retrieve(self, question, retriever, llm):
        # Step 1: Initial retrieval
        docs_1 = retriever.get_relevant_documents(question)

        # Extract key entities from step 1 results
        extract_prompt = f"""
        From these documents: {docs_1}
        Extract the specific entity needed for: {question}
        (e.g., person name, case number, date)
        """
        entity = llm.invoke(extract_prompt)

        # Step 2: Targeted retrieval using extracted entity
        follow_up_query = f"{entity} intellectual property disputes 2020"
        docs_2 = retriever.get_relevant_documents(follow_up_query)

        # Combine and generate answer from all retrieved docs
        return docs_1 + docs_2
```

**핵심 설계 원칙**: 각 홉은 모호한 요약이 아닌 *구체적이고 근거 있는 엔티티*(이름, ID, 날짜)를 추출해야 합니다 — 근거 있는 엔티티는 2단계 쿼리를 정확하게 만들고 환각 위험을 줄입니다.

</details>

### 연습 문제 3: 리랭킹(Reranking) vs. 검색 품질
RAG 시스템이 벡터 유사도로 상위 10개 문서를 검색한 다음, 크로스 인코더(cross-encoder) 리랭커를 사용하여 컨텍스트로 쓸 최종 상위 3개를 선택합니다. 이 두 단계 방식이 어느 한 단계만 사용하는 것(직접 상위 3개 검색 또는 리랭킹 없음)보다 뛰어난 이유를 설명하세요.

<details>
<summary>정답 보기</summary>

**검색만 사용 시 (상위 3개 직접 검색) 실패하는 이유**:

벡터 임베딩 모델(바이 인코더, bi-encoder)은 쿼리와 문서를 **독립적으로** 인코딩하고 임베딩을 비교합니다. 이는 효율적이지만 거친 방법입니다:
- 바이 인코더는 전체 문서를 고정 크기 벡터로 압축하므로 세밀한 토큰 상호작용이 손실됩니다.
- 재현율(recall) 최적화 — 의미적으로 관련된 문서 찾기 — 에는 적합하지만, 정밀도(precision) — *가장 관련성 높은* 문서 찾기 — 에는 약합니다.
- "Python 성능 최적화"에 대한 쿼리는 "Python"과 "최적화"에 관한 문서를 별도로 검색할 수 있어 두 문서 모두 높은 유사도 점수를 받지만 어느 것도 특정 질문에 답하지 못합니다.

임베딩 압축 아티팩트로 인해 최고의 답변이 4위나 5위를 받을 수 있어 상위 3개만 검색하면 놓칠 수 있습니다.

**리랭킹만 사용 시 (초기 검색 없이 모든 문서 리랭킹) 불가능한 이유**:

크로스 인코더는 쿼리 + 문서를 함께 처리하여 두 텍스트 전반에 걸친 토큰 수준 어텐션을 가능하게 합니다. 이는 매우 정확하지만 극도로 느립니다 (전체 코퍼스에 대해 O(n)번의 순전파).

쿼리당 100만 문서 코퍼스에 크로스 인코더를 실행하면 수 시간이 걸립니다. 실시간 검색에는 계산적으로 불가능합니다.

**두 단계가 가장 잘 작동하는 이유**:

```
1단계: 바이 인코더 (빠른 재현율)
  - 100만 문서에서 상위 100개를 ~20ms에 검색
  - 높은 재현율: 정답이 상위 100개에 거의 확실히 포함
  - 낮은 정밀도: 일부 비관련 문서 포함

2단계: 크로스 인코더 (정확한 리랭킹)
  - 상위 100개를 리랭킹 → ~200ms에 상위 3개 선택
  - 높은 정밀도: 토큰 수준 어텐션으로 진정한 관련성 식별
  - 실현 가능: 100만 개가 아닌 100개만 점수 매기면 됨

결과: 빠름 (바이 인코더 속도) + 정확함 (크로스 인코더 품질)
총 지연 시간: ~220ms vs 크로스 인코더 단독 사용 시 수 시간
```

**실용적 수치**: 연구에 따르면 두 단계 검색은 바이 인코더 단독 대비 MRR@3(평균 역순위)를 10-20% 향상시키며, 리랭킹 단계에서 약 10배 더 많은 계산이 필요합니다 — 품질이 중요한 애플리케이션에서는 가치 있는 트레이드오프입니다.

</details>

### 연습 문제 4: Self-RAG 리플렉션 토큰(Reflection Token) 설계
Self-RAG는 특별한 리플렉션 토큰을 사용하여 생성을 가이드합니다. 검색 여부와 검색된 정보를 인용할지를 동적으로 결정해야 하는 의료 정보 챗봇을 위한 리플렉션 토큰 로직을 설계하세요.

각 토큰이 무엇을 평가해야 하는지 명시하고, 각 리플렉션 결정이 달라지는 쿼리 예시를 제시하세요.

<details>
<summary>정답 보기</summary>

**의료 챗봇을 위한 리플렉션 토큰 설계**:

**토큰 1: [Retrieve] — 시스템이 외부 정보를 검색해야 하는가?**

| 결정 | 시기 | 예시 |
|----------|------|---------|
| YES | 의학적 사실, 약물 정보, 진단 기준 | "아세트아미노펜의 최대 안전 용량은?" |
| NO | 일반 건강 조언, 공감, 절차적 정보 | "가족에게 진단 결과를 어떻게 말해야 할까요?" |
| NO | 이미 컨텍스트에서 답변됨 | 방금 논의된 내용에 대한 후속 질문 |

**토큰 2: [IsREL] — 검색된 문서가 쿼리와 관련이 있는가?**

```python
# 검색 후, 컨텍스트에 포함하기 전에 평가
if retrieved_doc about "aspirin side effects" for query "acetaminophen dosage":
    [IsREL] = NOT_RELEVANT  # 다른 약물, 이 문서는 건너뜀
    → 더 세밀한 쿼리로 새로운 검색 트리거
```

**토큰 3: [IsSUP] — 생성된 응답이 검색된 문서를 충실히 반영하는가?**

```python
# 응답 생성 후, 소스와 대조하여 검증
retrieved: "건강한 성인의 최대 권장 용량은 하루 4g입니다"
generated: "안전 한도는 하루 3g입니다"

[IsSUP] = PARTIALLY_SUPPORTED
→ 수정 표시 또는 주의 사항 추가: "소스에는 건강한 성인에게 하루 4g이라고
  나와 있으며; 개인 편차가 있을 수 있습니다"
```

**토큰 4: [IsUSE] — 전반적인 응답이 사용자에게 유용한가?**

관련성 + 근거 + 실제 질문 해결 + 해롭지 않음을 종합적으로 평가합니다.

**각 결정이 달라지는 예시**:
- 쿼리: "혈압약과 함께 이부프로펜을 복용할 수 있나요?"
  - [Retrieve] = YES (약물 상호작용 사실 필요)
  - [IsREL] = YES (NSAIDs + 항고혈압제에 대한 약물 상호작용 데이터베이스 항목 검색됨)
  - [IsSUP] = YES (응답이 "NSAIDs가 항고혈압 효과를 감소시킬 수 있다"를 정확히 반영)
  - [IsUSE] = PARTIALLY (유용하지만 안전을 위해 "약사와 상담하세요" 주의 사항 추가 필요)

이 4가지 토큰 체인은 의료 챗봇이 약물 사실을 환각하지 않고, 관련 없는 제약 문서를 포함하지 않으며, 보수적으로 안전을 유지하도록 보장합니다.

</details>
4. Asai et al. (2023). "Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection"
