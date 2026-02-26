# 23. Advanced RAG

## Learning Objectives

After completing this lesson, you will be able to:

1. Identify the limitations of basic RAG (single retrieval, retrieval-question mismatch, context length constraints) and explain how advanced techniques address them
2. Implement query transformation techniques such as HyDE (Hypothetical Document Embeddings) and query expansion to improve retrieval relevance
3. Build multi-step and hierarchical retrieval pipelines using RAPTOR and multi-hop reasoning for complex questions
4. Apply post-retrieval techniques including reranking, context compression, and self-reflection to improve final answer quality
5. Design an Agentic RAG system that uses an LLM to dynamically decide when and what to retrieve during generation

---

## Overview

This lesson covers more sophisticated retrieval and generation strategies beyond basic RAG. We explore Agentic RAG, Multi-hop Reasoning, HyDE, RAPTOR, and other cutting-edge techniques.

---

## 1. RAG Limitations and Advanced Techniques

### 1.1 Limitations of Basic RAG

```
Basic RAG Problems:
┌─────────────────────────────────────────────────────────┐
│  1. Single Retrieval Limitation                         │
│     - Single search insufficient for complex questions  │
│     - Multi-step reasoning required                     │
│                                                         │
│  2. Retrieval-Question Mismatch                         │
│     - Style difference between questions and documents  │
│     - Limitations of embedding similarity               │
│                                                         │
│  3. Context Length Constraint                           │
│     - Difficult to handle many relevant documents       │
│     - Possible omission of important information        │
│                                                         │
│  4. Freshness/Accuracy                                  │
│     - Outdated information                              │
│     - Difficult to verify reliability                   │
└─────────────────────────────────────────────────────────┘
```

### 1.2 Advanced RAG Technique Classification

```
Advanced RAG Techniques:
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
HyDE Idea:
┌─────────────────────────────────────────────────────────┐
│  Query: "What is the capital of France?"                │
│                                                         │
│  Traditional: Search directly with query embedding      │
│        (question ↔ document style difference)           │
│                                                         │
│  HyDE: Generate hypothetical document with LLM then     │
│        search                                           │
│        Query → "Paris is the capital of France..."      │
│        → Search with this hypothetical document's       │
│          embedding                                      │
│        (document ↔ document style match)                │
└─────────────────────────────────────────────────────────┘
```

```python
from langchain.chains import HypotheticalDocumentEmbedder
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings

class HyDERetriever:
    """HyDE Retriever"""

    def __init__(self, llm, embeddings, vectorstore):
        self.llm = llm
        self.embeddings = embeddings
        self.vectorstore = vectorstore

    def generate_hypothetical_document(self, query: str) -> str:
        """Generate hypothetical document"""
        prompt = f"""Write a short passage that would answer the following question.
The passage should be factual and informative.

Question: {query}

Passage:"""

        response = self.llm.invoke(prompt)
        return response

    def retrieve(self, query: str, k: int = 5) -> list:
        """HyDE retrieval"""
        # 1. Generate hypothetical document
        hypothetical_doc = self.generate_hypothetical_document(query)

        # 2. Embed hypothetical document
        doc_embedding = self.embeddings.embed_query(hypothetical_doc)

        # 3. Search for similar documents
        results = self.vectorstore.similarity_search_by_vector(
            doc_embedding, k=k
        )

        return results


# LangChain built-in HyDE
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
    """Query expansion"""

    def __init__(self, llm):
        self.llm = llm

    def expand_query(self, query: str, num_variations: int = 3) -> list:
        """Expand query into multiple variations"""
        prompt = f"""Generate {num_variations} different versions of the following question.
Each version should ask the same thing but use different words or perspectives.

Original question: {query}

Variations:
1."""

        response = self.llm.invoke(prompt)

        # Parse
        variations = [query]  # Include original
        for line in response.split("\n"):
            line = line.strip()
            if line and line[0].isdigit():
                # "1. question" format
                variation = line.split(".", 1)[-1].strip()
                variations.append(variation)

        return variations[:num_variations + 1]

    def retrieve_with_expansion(
        self,
        query: str,
        retriever,
        k: int = 5
    ) -> list:
        """Retrieve with expanded queries"""
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

        # Return top k (sorted by RRF or other method)
        return all_docs[:k]
```

---

## 3. Agentic RAG

### 3.1 Concept

```
Agentic RAG:
┌─────────────────────────────────────────────────────────┐
│  LLM Agent dynamically uses retrieval tools             │
│                                                         │
│  Agent Loop:                                            │
│  1. Analyze question                                    │
│  2. Determine needed information                        │
│  3. Call retrieval tools (optional, repeatable)         │
│  4. Evaluate results                                    │
│  5. Need more search? → Repeat                          │
│  6. Generate final answer                               │
│                                                         │
│  vs Basic RAG:                                          │
│  Query → Retrieve → Generate (fixed pipeline)           │
└─────────────────────────────────────────────────────────┘
```

### 3.2 Implementation

```python
from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate

class AgenticRAG:
    """Agentic RAG System"""

    def __init__(self, llm, vectorstore, web_search=None):
        self.llm = llm
        self.vectorstore = vectorstore
        self.web_search = web_search

        self.tools = self._setup_tools()
        self.agent = self._create_agent()

    def _setup_tools(self) -> list:
        """Setup tools"""
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
        """Search knowledge base"""
        docs = self.vectorstore.similarity_search(query, k=3)
        return "\n\n".join([doc.page_content for doc in docs])

    def _search_web(self, query: str) -> str:
        """Web search (requires external API)"""
        if self.web_search:
            return self.web_search.run(query)
        return "Web search not available."

    def _lookup_specific(self, query: str) -> str:
        """Specific information lookup"""
        docs = self.vectorstore.similarity_search(query, k=1)
        if docs:
            return docs[0].page_content
        return "No specific information found."

    def _create_agent(self):
        """Create ReAct Agent"""
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
        """Process question"""
        result = self.agent.invoke({"input": question})
        return result["output"]


# Usage example
def agentic_rag_example():
    from langchain.llms import OpenAI
    from langchain.vectorstores import Chroma

    llm = OpenAI(temperature=0)
    vectorstore = Chroma(...)  # Setup required

    rag = AgenticRAG(llm, vectorstore)

    # Complex question
    answer = rag.query(
        "Compare our company's revenue growth in 2023 with the industry average"
    )
    print(answer)
```

---

## 4. Multi-hop Reasoning

### 4.1 Concept

```
Multi-hop Reasoning:
┌─────────────────────────────────────────────────────────┐
│  Question: "What is the population of Biden's birthplace?" │
│                                                         │
│  Hop 1: "What is Biden's birthplace?" → "Scranton, PA" │
│  Hop 2: "What is Scranton's population?" → "76,328"    │
│                                                         │
│  Final Answer: "76,328"                                 │
│                                                         │
│  Single search cannot directly find the answer          │
└─────────────────────────────────────────────────────────┘
```

### 4.2 Implementation

```python
class MultiHopRAG:
    """Multi-hop Reasoning RAG"""

    def __init__(self, llm, retriever, max_hops: int = 3):
        self.llm = llm
        self.retriever = retriever
        self.max_hops = max_hops

    def decompose_question(self, question: str) -> list:
        """Decompose question into sub-questions"""
        prompt = f"""Break down the following complex question into simpler sub-questions.
Each sub-question should be answerable independently.

Question: {question}

Sub-questions (one per line):"""

        response = self.llm.invoke(prompt)
        sub_questions = [q.strip() for q in response.split("\n") if q.strip()]
        return sub_questions

    def answer_with_hops(self, question: str) -> dict:
        """Answer with multi-step reasoning"""
        reasoning_chain = []
        context = ""

        for hop in range(self.max_hops):
            # Determine next query based on current context
            if hop == 0:
                current_query = question
            else:
                current_query = self._generate_follow_up(
                    question, context, reasoning_chain
                )

            if current_query is None:
                break

            # Retrieve
            docs = self.retriever.get_relevant_documents(current_query)
            new_context = "\n".join([doc.page_content for doc in docs])

            # Generate intermediate answer
            intermediate_answer = self._generate_intermediate_answer(
                current_query, new_context
            )

            reasoning_chain.append({
                "hop": hop + 1,
                "query": current_query,
                "answer": intermediate_answer
            })

            context += f"\n{intermediate_answer}"

            # Check if enough information
            if self._has_enough_info(question, context):
                break

        # Final answer
        final_answer = self._generate_final_answer(question, reasoning_chain)

        return {
            "question": question,
            "reasoning_chain": reasoning_chain,
            "final_answer": final_answer
        }

    def _generate_follow_up(self, original_q, context, chain) -> str:
        """Generate follow-up question"""
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
        """Generate intermediate answer"""
        prompt = f"""Based on the following context, answer the question briefly.

Context: {context}

Question: {query}

Answer:"""

        return self.llm.invoke(prompt)

    def _has_enough_info(self, question, context) -> bool:
        """Check if there's enough information"""
        prompt = f"""Can you answer the following question based on this information?

Question: {question}
Information: {context}

Answer YES or NO:"""

        response = self.llm.invoke(prompt)
        return "YES" in response.upper()

    def _generate_final_answer(self, question, chain) -> str:
        """Generate final answer"""
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

### 5.1 Concept

```
RAPTOR Structure:
┌─────────────────────────────────────────────────────────┐
│  Level 3 (Highest-level summary)                        │
│  ┌──────────────────────────────────┐                   │
│  │     Abstract Summary              │                  │
│  └──────────────────────────────────┘                   │
│              ↑                                          │
│  Level 2 (Cluster summaries)                            │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐          │
│  │ Summary1 │    │ Summary2 │    │ Summary3 │          │
│  └──────────┘    └──────────┘    └──────────┘          │
│      ↑   ↑          ↑   ↑          ↑   ↑              │
│  Level 1 (Chunk clustering)                             │
│  [C1][C2][C3]    [C4][C5][C6]    [C7][C8][C9]          │
│      ↑   ↑   ↑      ↑   ↑   ↑      ↑   ↑   ↑          │
│  Level 0 (Original chunks)                              │
│  [Chunk1][Chunk2]...[ChunkN]                           │
└─────────────────────────────────────────────────────────┘

Retrieval: Search across multiple levels simultaneously to get information at various abstraction levels
```

### 5.2 Implementation

```python
from sklearn.cluster import KMeans
import numpy as np

class RAPTOR:
    """RAPTOR hierarchical retrieval"""

    def __init__(self, llm, embeddings, num_levels: int = 3):
        self.llm = llm
        self.embeddings = embeddings
        self.num_levels = num_levels
        self.tree = {}

    def build_tree(self, documents: list, cluster_size: int = 5):
        """Build RAPTOR tree"""
        # Level 0: Original chunks
        self.tree[0] = documents
        current_docs = documents

        for level in range(1, self.num_levels):
            # Compute embeddings
            texts = [doc.page_content for doc in current_docs]
            embeddings = self.embeddings.embed_documents(texts)

            # Clustering
            n_clusters = max(len(current_docs) // cluster_size, 1)
            kmeans = KMeans(n_clusters=n_clusters)
            clusters = kmeans.fit_predict(embeddings)

            # Summarize each cluster
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
        """Summarize cluster"""
        combined_text = "\n\n".join([doc.page_content for doc in docs])

        prompt = f"""Summarize the following texts into a concise summary that captures the key information.

Texts:
{combined_text}

Summary:"""

        summary = self.llm.invoke(prompt)

        # Wrap as Document object
        from langchain.schema import Document
        return Document(page_content=summary)

    def retrieve(self, query: str, k_per_level: int = 2) -> list:
        """Hierarchical retrieval"""
        all_results = []

        for level, docs in self.tree.items():
            # Search at each level
            texts = [doc.page_content for doc in docs]
            query_embedding = self.embeddings.embed_query(query)
            doc_embeddings = self.embeddings.embed_documents(texts)

            # Cosine similarity
            similarities = np.dot(doc_embeddings, query_embedding)
            top_indices = np.argsort(similarities)[-k_per_level:]

            for idx in top_indices:
                all_results.append({
                    "level": level,
                    "document": docs[idx],
                    "score": similarities[idx]
                })

        # Sort by score
        all_results.sort(key=lambda x: x["score"], reverse=True)
        return all_results
```

---

## 6. ColBERT (Contextualized Late Interaction)

### 6.1 Concept

```
ColBERT vs Dense Retrieval:
┌─────────────────────────────────────────────────────────┐
│  Dense Retrieval (bi-encoder):                          │
│  Query → [CLS] embedding                                │
│  Doc   → [CLS] embedding                                │
│  Score = dot(query_emb, doc_emb)                        │
│  Problem: Hard to represent complex meaning in single   │
│           vector                                        │
│                                                         │
│  ColBERT (late interaction):                            │
│  Query → [q1, q2, ..., qn] (per-token embeddings)       │
│  Doc   → [d1, d2, ..., dm] (per-token embeddings)       │
│  Score = Σᵢ maxⱼ sim(qᵢ, dⱼ)                           │
│  Advantage: More precise search through token-level     │
│             matching                                    │
└─────────────────────────────────────────────────────────┘
```

### 6.2 Usage

```python
from colbert import Indexer, Searcher
from colbert.infra import Run, RunConfig, ColBERTConfig

class ColBERTRetriever:
    """ColBERT retriever"""

    def __init__(self, index_name: str = "my_index"):
        self.index_name = index_name
        self.config = ColBERTConfig(
            nbits=2,
            doc_maxlen=300,
            query_maxlen=32
        )

    def build_index(self, documents: list, collection_path: str):
        """Build index"""
        # Save documents to file
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
        """Search"""
        with Run().context(RunConfig(nranks=1)):
            searcher = Searcher(index=self.index_name)
            results = searcher.search(query, k=k)

        return results


# RAGatouille (easier ColBERT wrapper)
def colbert_with_ragatouille():
    from ragatouille import RAGPretrainedModel

    rag = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")

    # Indexing
    rag.index(
        collection=[
            "Document 1 content...",
            "Document 2 content..."
        ],
        index_name="my_index"
    )

    # Search
    results = rag.search("my query", k=5)
    return results
```

---

## 7. Self-RAG (Self-Reflective RAG)

```python
class SelfRAG:
    """Self-RAG: Self-reflective RAG"""

    def __init__(self, llm, retriever):
        self.llm = llm
        self.retriever = retriever

    def query(self, question: str) -> dict:
        """Self-RAG query"""
        # 1. Assess retrieval need
        needs_retrieval = self._assess_retrieval_need(question)

        if not needs_retrieval:
            # Answer directly without retrieval
            answer = self._generate_without_retrieval(question)
            return {"answer": answer, "retrieval_used": False}

        # 2. Retrieve
        docs = self.retriever.get_relevant_documents(question)

        # 3. Evaluate relevance (per document)
        relevant_docs = []
        for doc in docs:
            if self._is_relevant(question, doc):
                relevant_docs.append(doc)

        # 4. Generate answer
        answer = self._generate_with_context(question, relevant_docs)

        # 5. Evaluate answer quality
        is_supported = self._check_support(answer, relevant_docs)
        is_useful = self._check_usefulness(question, answer)

        # 6. Retry if needed
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
        """Assess retrieval need"""
        prompt = f"""Determine if external knowledge is needed to answer this question.

Question: {question}

Answer YES if retrieval is needed, NO if you can answer from general knowledge:"""

        response = self.llm.invoke(prompt)
        return "YES" in response.upper()

    def _is_relevant(self, question: str, doc) -> bool:
        """Evaluate document relevance"""
        prompt = f"""Is this document relevant to the question?

Question: {question}
Document: {doc.page_content[:500]}

Answer RELEVANT or IRRELEVANT:"""

        response = self.llm.invoke(prompt)
        return "RELEVANT" in response.upper()

    def _check_support(self, answer: str, docs: list) -> bool:
        """Check if answer is supported by documents"""
        context = "\n".join([doc.page_content for doc in docs])

        prompt = f"""Is this answer supported by the given context?

Context: {context}
Answer: {answer}

Respond SUPPORTED or NOT_SUPPORTED:"""

        response = self.llm.invoke(prompt)
        return "SUPPORTED" in response.upper()

    def _check_usefulness(self, question: str, answer: str) -> bool:
        """Check answer usefulness"""
        prompt = f"""Does this answer actually address the question?

Question: {question}
Answer: {answer}

Respond USEFUL or NOT_USEFUL:"""

        response = self.llm.invoke(prompt)
        return "USEFUL" in response.upper()
```

---

## Key Summary

### Advanced RAG Techniques
```
1. HyDE: Improve retrieval quality with hypothetical documents
2. Query Expansion: Search with diverse queries
3. Agentic RAG: Dynamic retrieval by LLM Agent
4. Multi-hop: Multi-step reasoning
5. RAPTOR: Hierarchical summary tree
6. ColBERT: Token-level late interaction
7. Self-RAG: Self-reflection and verification
```

### Selection Guide
```
Simple QA → Basic RAG
Complex questions → Multi-hop + Agentic
Long documents → RAPTOR
Precise search → ColBERT
Quality critical → Self-RAG
```

---

## References

1. Gao et al. (2022). "Precise Zero-Shot Dense Retrieval without Relevance Labels" (HyDE)
2. Sarthi et al. (2024). "RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval"
3. Khattab et al. (2020). "ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction"

---

## Exercises

### Exercise 1: HyDE vs. Direct Query — When Does It Help?
HyDE generates a hypothetical document before searching, rather than searching directly with the query. Analyze when HyDE helps and when it does not, using two concrete examples.

Given these retrieval tasks:
- **Task A**: "What are the side effects of aspirin?" (factual, well-documented)
- **Task B**: "Find a document written in a formal legal tone about contract termination rights"

<details>
<summary>Show Answer</summary>

**Why HyDE helps**:

The key insight is that **question embeddings and document embeddings occupy different regions of the embedding space**. Questions are short, interrogative, and sparse ("What are side effects of aspirin?"). Documents are long, declarative, and dense ("Aspirin may cause gastrointestinal bleeding, tinnitus..."). Direct query-to-document similarity often misses relevant documents because the embedding model sees them as different linguistic styles.

HyDE bridges this gap by generating a hypothetical answer document first, then embedding *that* as the search query. Now you're doing document-to-document similarity, which embedding models handle better.

**Task A: "What are the side effects of aspirin?"**
```
Direct query embedding:
"What are the side effects of aspirin?" → [0.12, -0.34, ...]
Target document:
"Aspirin (acetylsalicylic acid) can cause..." → [0.09, -0.31, ...]
Similarity: moderate (different style)

HyDE hypothetical document:
"The common side effects of aspirin include gastrointestinal
irritation, bleeding, tinnitus at high doses..." → [0.08, -0.32, ...]
Target document:
"Aspirin (acetylsalicylic acid) can cause..." → [0.09, -0.31, ...]
Similarity: high (same declarative style)
```
**HyDE helps here** — factual questions benefit because the hypothetical answer closely mirrors the style and content of target documents.

**Task B: "Find a document written in a formal legal tone about contract termination"**
- The query is already describing a *document style*, not a factual question.
- HyDE generates a *hypothetical legal document* — but this introduces risk: the LLM may hallucinate specific legal clauses that don't appear in the actual corpus.
- For style-matching retrieval, hybrid search (dense + BM25) may work better without the hallucination risk.

**When HyDE does NOT help (or hurts)**:
1. **Highly specific factual lookups** where the LLM may confidently hallucinate wrong details (e.g., "Find the Q4 2023 revenue of Company X" — hypothetical document might generate plausible but wrong numbers).
2. **Very short factual queries** in well-matched corpora where direct embedding similarity already works well.
3. **Time-sensitive queries** where the hypothetical document may reflect outdated training knowledge.

</details>

### Exercise 2: Multi-hop Retrieval Design
A legal research assistant needs to answer: "Was the judge in the Smith v. Johnson 2019 case also involved in cases related to intellectual property disputes in 2020?"

This question requires at least 2 retrieval steps. Design the multi-hop retrieval pipeline, specifying what is retrieved at each step and what information is passed forward.

<details>
<summary>Show Answer</summary>

**Why single retrieval fails**:
A single query "judge from Smith v. Johnson 2019 involved in IP cases 2020" will likely fail because no document contains all this information. The judge's name isn't known upfront, so the query can't be specific enough.

**Multi-hop pipeline**:

```
Step 1: Retrieve case details
  Query: "Smith v. Johnson 2019 case"
  Retrieved: Case record with judge name = "Judge Robert Thompson"
  Extracted information: judge_name = "Robert Thompson"

Step 2: Use extracted info for second retrieval
  Query: "Judge Robert Thompson intellectual property 2020"
  Retrieved: IP cases from 2020 listing Judge Thompson
  Answer: "Yes, Judge Thompson presided over DataTech v. InnoSoft (2020)
           and two other IP disputes in 2020."
```

**Implementation pattern**:
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

**Key design principle**: Each hop should extract a *specific, grounded entity* (name, ID, date) rather than a vague summary — grounded entities make step 2 queries precise and reduce hallucination risk.

</details>

### Exercise 3: Reranking vs. Retrieval Quality
A RAG system retrieves the top-10 documents by vector similarity and then uses a cross-encoder reranker to select the final top-3 for context. Explain why this two-stage approach outperforms either stage alone (retrieve top-3 directly, or no reranking).

<details>
<summary>Show Answer</summary>

**Why retrieval alone (top-3 directly) can fail**:

Vector embedding models (bi-encoders) encode query and document **independently** and compare their embeddings. This is efficient but coarse:
- Bi-encoders compress entire documents into fixed-size vectors, losing fine-grained token interactions.
- They are optimized for recall (finding semantically related documents) not precision (finding the *most relevant* document).
- A query about "Python performance optimization" may retrieve documents about "Python" and "optimization" separately — both get high similarity scores but neither answers the specific question.

If you only retrieve top-3 with bi-encoder, you risk missing the best answer that scored 4th or 5th due to embedding compression artifacts.

**Why reranking alone (no initial retrieval, rerank everything) is impossible**:

Cross-encoders jointly process query + document together, enabling token-level attention across both. This is highly accurate but extremely slow (O(n) forward passes over entire corpus).

Running a cross-encoder on a 1M-document corpus per query would take hours. It's computationally infeasible for real-time retrieval.

**Why two-stage works best**:

```
Stage 1: Bi-encoder (fast recall)
  - Retrieve top-100 from 1M documents in ~20ms
  - High recall: correct answer is almost certainly in top-100
  - Lower precision: some irrelevant documents included

Stage 2: Cross-encoder (accurate reranking)
  - Rerank top-100 → select top-3 in ~200ms
  - High precision: token-level attention identifies true relevance
  - Feasible: only 100 documents to score (not 1M)

Result: Fast (bi-encoder speed) + Accurate (cross-encoder quality)
Total latency: ~220ms vs hours for cross-encoder alone
```

**Practical numbers**: Studies show two-stage retrieval improves MRR@3 (Mean Reciprocal Rank) by 10-20% over bi-encoder alone, at the cost of ~10x more compute for the reranking stage — a worthwhile trade-off for quality-critical applications.

</details>

### Exercise 4: Self-RAG Reflection Token Design
Self-RAG uses special reflection tokens to guide generation. Design the reflection token logic for a medical information chatbot that must decide whether to retrieve and whether to cite retrieved information.

Specify what each token should evaluate and give an example of a query where each reflection decision differs.

<details>
<summary>Show Answer</summary>

**Reflection token design for medical chatbot**:

**Token 1: [Retrieve] — Should the system retrieve external information?**

| Decision | When | Example |
|----------|------|---------|
| YES | Medical facts, drug info, diagnostic criteria | "What is the maximum safe dose of acetaminophen?" |
| NO | General health advice, empathy, procedural info | "How should I tell my family about my diagnosis?" |
| NO | Already answered from context | Follow-up on something just discussed |

**Token 2: [IsREL] — Is the retrieved document relevant to the query?**

```python
# After retrieval, evaluate before including in context
if retrieved_doc about "aspirin side effects" for query "acetaminophen dosage":
    [IsREL] = NOT_RELEVANT  # Different drug, skip this document
    → Trigger new retrieval with refined query
```

**Token 3: [IsSUP] — Does the generated response faithfully reflect the retrieved documents?**

```python
# After generating response, verify against sources
retrieved: "The maximum recommended dose is 4g/day for healthy adults"
generated: "The safe limit is 3g per day"

[IsSUP] = PARTIALLY_SUPPORTED
→ Flag for revision or add caveat: "Sources indicate 4g/day for healthy adults;
  individual variation may apply"
```

**Token 4: [IsUSE] — Is the overall response useful to the user?**

Evaluates holistically: relevant + supported + addresses the actual question + not harmful.

**Example where each decision differs**:
- Query: "Can I take ibuprofen with my blood pressure medication?"
  - [Retrieve] = YES (drug interaction facts needed)
  - [IsREL] = YES (retrieved drug interaction database entry for NSAIDs + antihypertensives)
  - [IsSUP] = YES (response accurately reflects "NSAIDs can reduce antihypertensive effectiveness")
  - [IsUSE] = PARTIALLY (useful but should add "consult your pharmacist" caveat for safety)

This four-token chain ensures the medical chatbot doesn't hallucinate drug facts, doesn't include irrelevant pharmaceutical documents, and remains conservatively safe.

</details>
4. Asai et al. (2023). "Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection"
