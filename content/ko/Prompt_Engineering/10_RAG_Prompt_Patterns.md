# 10. RAG 프롬프트 패턴(RAG Prompt Patterns)

**이전**: [코드 생성 프롬프팅](./09_Code_Generation_Prompting.md) | **다음**: [프롬프트 최적화](./11_Prompt_Optimization.md)

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 검색된 컨텍스트에 LLM 응답을 기반시키는 검색 증강 생성(RAG) 프롬프트를 설계하기
2. 주장을 원본 문서까지 추적하는 인용(Citation) 및 출처 표시(Attribution) 패턴을 구현하기
3. 제공된 컨텍스트를 넘어선 환각(Hallucination)을 방지하는 충실성(Faithfulness) 기법을 적용하기
4. 프롬프트에서 충돌하는 소스, 누락된 정보, 다중 문서 합성(Multi-Document Synthesis)을 처리하기
5. 다양한 사용 사례에 대해 RAG와 긴 컨텍스트(Long-Context) 접근 방식 간의 트레이드오프를 평가하기

---

검색 증강 생성(Retrieval-Augmented Generation, RAG)은 지식 기반 LLM 애플리케이션을 구축하기 위한 지배적인 패턴입니다. 모델의 매개변수적 지식(학습 중 배운 것)에 의존하는 대신, RAG는 쿼리 시점에 관련 문서를 검색하고 프롬프트에 주입합니다. 이를 통해 모델은 비공개 데이터, 최근 이벤트, 또는 학습 데이터에 없는 도메인 특화 지식에 대한 질문에 답할 수 있습니다.

그러나 RAG는 프롬프트만큼만 좋습니다. 잘못 설계된 RAG 프롬프트는 검색된 컨텍스트를 무시하거나, 문서에 없는 답변을 환각하거나, 출처를 인용하지 못하거나, 검색된 청크가 관련이 없거나 모순될 때 무너질 수 있습니다. 이 레슨에서는 RAG 시스템을 신뢰할 수 있게 만드는 프롬프트 패턴을 다룹니다.

## 목차

1. [검색 증강 생성 개요](#1-검색-증강-생성-개요retrieval-augmented-generation-overview)
2. [컨텍스트 주입 패턴](#2-컨텍스트-주입-패턴context-injection-patterns)
3. [인용 및 출처 표시 프롬프트](#3-인용-및-출처-표시-프롬프트citation-and-attribution-prompts)
4. [충실성과 그라운딩](#4-충실성과-그라운딩faithfulness-and-grounding)
5. [충돌하는 소스 처리](#5-충돌하는-소스-처리handling-conflicting-sources)
6. [컨텍스트 기반 전용 답변 패턴](#6-컨텍스트-기반-전용-답변-패턴answer-only-from-context-patterns)
7. [다중 문서 합성](#7-다중-문서-합성multi-document-synthesis)
8. [프롬프트 내 청크 관련성 점수 매기기](#8-프롬프트-내-청크-관련성-점수-매기기chunk-relevance-scoring-in-prompts)
9. [답변 불가 시나리오 처리](#9-답변-불가-시나리오-처리handling-no-answer-scenarios)
10. [RAG vs 긴 컨텍스트 트레이드오프](#10-rag-vs-긴-컨텍스트-트레이드오프rag-vs-long-context-trade-offs)

---

## 1. 검색 증강 생성 개요(Retrieval-Augmented Generation Overview)

### 1.1 RAG 파이프라인

RAG 시스템은 세 단계로 구성됩니다:

```
User Query
    │
    ▼
┌──────────┐     ┌────────────┐     ┌──────────────┐
│ Retrieve  │────▶│  Augment   │────▶│   Generate   │
│ (Search)  │     │  (Prompt)  │     │  (LLM Call)  │
└──────────┘     └────────────┘     └──────────────┘
    │                  │                    │
    ▼                  ▼                    ▼
 Top-k docs     Prompt with context    Final answer
```

- **검색(Retrieve)**: 벡터 검색, 키워드 검색, 또는 하이브리드 접근 방식을 사용하여 관련 문서 찾기
- **증강(Augment)**: 검색된 문서를 프롬프트 템플릿에 삽입
- **생성(Generate)**: 증강된 프롬프트를 LLM에 보내어 응답 생성

### 1.2 RAG에서 프롬프팅이 중요한 이유

검색 단계에서 올바른 문서를 가져오지만, *프롬프트*가 모델이 그것을 어떻게 사용하는지를 결정합니다:

| 프롬프트 품질 | 동작 |
|-------------|------|
| 컨텍스트 프레이밍 없음 | 모델이 컨텍스트를 무시하고 매개변수적 메모리로 답변 |
| 컨텍스트 있지만 그라운딩 지시 없음 | 모델이 컨텍스트와 자체 지식을 혼합 |
| 그라운딩되었지만 인용 지시 없음 | 모델이 정확하게 답변하지만 추적성을 제공하지 않음 |
| 완전히 지정된 RAG 프롬프트 | 모델이 컨텍스트에서 답변하고, 소스를 인용하고, 부족한 부분을 인정 |

### 1.3 기본 RAG 프롬프트 구조

```python
import anthropic

client = anthropic.Anthropic()

def basic_rag(query: str, retrieved_chunks: list[dict]) -> str:
    """Basic RAG prompt pattern."""
    # Format retrieved context
    context_block = ""
    for i, chunk in enumerate(retrieved_chunks):
        context_block += f"\n[Document {i+1}] (Source: {chunk['source']})\n"
        context_block += f"{chunk['text']}\n"

    prompt = f"""Answer the user's question based on the provided context.

CONTEXT:
{context_block}

USER QUESTION: {query}

INSTRUCTIONS:
- Base your answer ONLY on the provided context
- If the context does not contain enough information to answer, say so explicitly
- Cite the document number(s) that support each claim"""

    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}]
    )
    return message.content[0].text
```

---

## 2. 컨텍스트 주입 패턴(Context Injection Patterns)

컨텍스트를 프롬프트에 주입하는 방법은 모델이 그것을 처리하는 방식에 큰 영향을 미칩니다. 서로 다른 패턴이 서로 다른 사용 사례에 적합합니다.

### 2.1 플랫 컨텍스트 주입(Flat Context Injection)

가장 간단한 패턴: 모든 청크를 하나의 블록으로 연결합니다.

```python
def flat_context_prompt(query: str, chunks: list[str]) -> str:
    """All chunks as a single context block."""
    context = "\n\n---\n\n".join(chunks)
    return f"""Use the following context to answer the question.

Context:
{context}

Question: {query}

Answer:"""
```

**사용 시점**: 소수의 청크(1-3개), 모두 높은 관련성, 소스 추적이 필요 없는 경우.

**한계**: 모델이 청크를 구분할 수 없어 인용이 불가능합니다.

### 2.2 태그된 컨텍스트 주입(Tagged Context Injection)

추적성을 위해 각 청크에 메타데이터를 라벨링합니다:

```python
def tagged_context_prompt(query: str, chunks: list[dict]) -> str:
    """Each chunk tagged with source metadata."""
    context_parts = []
    for i, chunk in enumerate(chunks):
        tag = (
            f"<document id=\"{i+1}\" "
            f"source=\"{chunk['source']}\" "
            f"date=\"{chunk.get('date', 'unknown')}\" "
            f"relevance_score=\"{chunk.get('score', 'N/A')}\">\n"
            f"{chunk['text']}\n"
            f"</document>"
        )
        context_parts.append(tag)

    context_block = "\n\n".join(context_parts)

    return f"""You are a research assistant. Answer the question using ONLY
the provided documents. Cite documents by their ID number.

<documents>
{context_block}
</documents>

<question>{query}</question>

Provide your answer with inline citations like [1], [2], etc."""
```

**사용 시점**: 대부분의 RAG 애플리케이션. XML 스타일 태그는 Claude가 특정 문서를 파싱하고 참조하는 데 도움이 됩니다.

### 2.3 계층적 컨텍스트 주입(Hierarchical Context Injection)

긴 문서의 경우, 먼저 요약을 주입한 다음 관련 세부사항을 주입합니다:

```python
def hierarchical_context_prompt(
    query: str,
    doc_summaries: list[dict],
    relevant_sections: list[dict]
) -> str:
    """Two-level context: summaries + relevant sections."""
    summary_block = ""
    for doc in doc_summaries:
        summary_block += f"\nDocument '{doc['title']}': {doc['summary']}\n"

    detail_block = ""
    for section in relevant_sections:
        detail_block += (
            f"\n[From: {section['doc_title']}, "
            f"Section: {section['section']}]\n"
            f"{section['text']}\n"
        )

    return f"""Answer the question using the provided documents.

DOCUMENT SUMMARIES (for overall context):
{summary_block}

RELEVANT SECTIONS (for detailed answers):
{detail_block}

QUESTION: {query}

Use the document summaries to understand the broader context, but base your
specific claims on the relevant sections. Cite sections, not summaries."""
```

### 2.4 역할 분리 컨텍스트(Role-Separated Context)

시스템 프롬프트를 지시사항에, 사용자 메시지를 컨텍스트 + 쿼리에 사용합니다:

```python
import anthropic

client = anthropic.Anthropic()

def role_separated_rag(query: str, chunks: list[dict]) -> str:
    """System prompt for instructions, user message for context + query."""
    system = """You are a precise research assistant. Your task is to answer
questions using ONLY the provided source documents.

RULES:
1. Every factual claim must cite its source using [Doc N] notation
2. If sources conflict, present both views and note the disagreement
3. If the sources do not contain the answer, say: "The provided sources
   do not contain sufficient information to answer this question."
4. Never use information from your training data -- only the provided sources
5. Keep answers concise but complete"""

    context_block = ""
    for i, chunk in enumerate(chunks):
        context_block += f"\n[Doc {i+1}] ({chunk['source']}):\n{chunk['text']}\n"

    user_message = f"""SOURCE DOCUMENTS:
{context_block}

QUESTION: {query}"""

    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        system=system,
        messages=[{"role": "user", "content": user_message}]
    )
    return message.content[0].text
```

이 패턴이 가장 효과적인 경우가 많은 이유:
- 시스템 프롬프트가 여러 쿼리에 걸쳐 일관된 동작을 설정합니다
- 사용자 메시지에는 가변적인 부분(컨텍스트 + 질문)만 포함됩니다
- 동작 지시사항을 변경하지 않고 컨텍스트를 쉽게 교체할 수 있습니다

---

## 3. 인용 및 출처 표시 프롬프트(Citation and Attribution Prompts)

인용(Citation)은 신뢰와 검증을 위해 중요합니다. 인용 없이는 사용자가 RAG 기반 답변과 환각을 구분할 수 없습니다.

### 3.1 인라인 인용 패턴(Inline Citation Pattern)

```python
inline_citation_system = """Answer questions using the provided sources.

CITATION FORMAT:
- Use inline citations: [1], [2], etc.
- Place the citation immediately after the claim it supports
- A single sentence can have multiple citations if it synthesizes multiple sources
- At the end of your answer, list all cited sources with their titles

EXAMPLE:
The Eiffel Tower is 330 meters tall [1] and was completed in 1889 [2].
It was originally intended to be dismantled after 20 years [1].

Sources cited:
[1] "Eiffel Tower Facts" - paris-tourism.fr
[2] "History of French Architecture" - architecture-digest.com"""
```

### 3.2 인용문 기반 인용(Quote-Based Citation)

검증 가능한 출처 표시가 필요한 경우, 직접 인용문을 요구합니다:

```python
quote_citation_prompt = """Answer the question using ONLY the provided documents.

CITATION RULES:
- For each claim, provide a direct quote from the source document
- Format: "direct quote" (Document N, page/section)
- If you need to paraphrase, first provide the original quote, then your
  interpretation
- Do NOT make claims that cannot be supported by a direct quote

EXAMPLE FORMAT:
According to the policy document, employees are entitled to "a minimum of
15 business days of paid annual leave" (Document 2, Section 4.1). This
means full-time employees get three weeks of vacation per year.

DOCUMENTS:
{context}

QUESTION: {query}"""
```

### 3.3 구조화된 인용 출력(Structured Citation Output)

프로그래밍 방식으로 사용하기 위해, 구조화된 인용 데이터를 요청합니다:

```python
import anthropic
import json

client = anthropic.Anthropic()

def rag_with_structured_citations(query: str, chunks: list[dict]) -> dict:
    """RAG with machine-readable citation output."""
    context_block = ""
    for i, chunk in enumerate(chunks):
        context_block += f"\n<source id=\"{i+1}\" title=\"{chunk['title']}\">\n{chunk['text']}\n</source>\n"

    prompt = f"""Answer the question using the provided sources.

{context_block}

Question: {query}

Respond in this exact JSON format:
{{
    "answer": "Your complete answer text with [N] citation markers",
    "citations": [
        {{
            "marker": "[1]",
            "source_id": 1,
            "source_title": "Title of source",
            "quoted_text": "The exact text from the source that supports this claim",
            "claim": "The specific claim this citation supports"
        }}
    ],
    "confidence": "high|medium|low",
    "unanswered_aspects": ["Any parts of the question not addressed by sources"]
}}"""

    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2048,
        messages=[{"role": "user", "content": prompt}]
    )

    return json.loads(message.content[0].text)
```

### 3.4 인용 검증(Citation Verification)

인용된 답변을 받은 후, 후속 작업에서 인용을 검증할 수 있습니다:

```python
def verify_citations(answer: str, citations: list[dict], sources: list[dict]) -> str:
    """Prompt to verify that citations actually support their claims."""
    verification_prompt = f"""Verify each citation in the following answer.

ANSWER:
{answer}

CITATIONS AND THEIR SOURCE TEXT:
"""
    for c in citations:
        source_text = sources[c["source_id"] - 1]["text"]
        verification_prompt += f"""
Citation {c['marker']}:
- Claim: {c['claim']}
- Quoted text: "{c['quoted_text']}"
- Full source document: {source_text}
"""

    verification_prompt += """
For each citation, assess:
1. Is the quoted text actually present in the source document? (EXACT/PARAPHRASE/NOT_FOUND)
2. Does the source text actually support the claim? (SUPPORTS/PARTIALLY/CONTRADICTS/UNRELATED)
3. Is the claim a faithful representation of the source? (FAITHFUL/EXAGGERATED/MISLEADING)

Format as a table: | Citation | Text Present | Supports Claim | Faithful |"""

    return verification_prompt
```

---

## 4. 충실성과 그라운딩(Faithfulness and Grounding)

충실성(Faithfulness)은 모델의 답변이 학습 데이터의 정보를 추가하지 않고 제공된 컨텍스트에 의해 완전히 뒷받침되는 것을 의미합니다.

### 4.1 그라운딩 지시사항(Grounding Instructions)

```python
grounding_system = """You are a document-grounded assistant.

GROUNDING RULES:
1. ANSWER ONLY from the provided documents. Do not use any knowledge from
   your training data, even if you believe it to be correct.
2. If the documents contain the answer: provide it with citations.
3. If the documents partially answer the question: answer what you can and
   explicitly state what is not covered.
4. If the documents do not contain the answer: say "I cannot answer this
   based on the provided documents."
5. NEVER start with "Based on my knowledge..." or "I know that..."
6. NEVER add context, examples, or elaboration beyond what the documents state.

SELF-CHECK before responding:
- Can I point to a specific passage that supports each sentence of my answer?
- Am I adding any information not present in the documents?
- Would my answer change if I had no training data and only the documents?"""
```

### 4.2 충실성 검증 체인(Faithfulness Verification Chain)

2단계 프로세스를 사용합니다: 생성 후 검증:

```python
import anthropic

client = anthropic.Anthropic()

def faithful_rag(query: str, context: str) -> str:
    """Two-step RAG: generate answer, then verify faithfulness."""

    # Step 1: Generate answer
    generate_prompt = f"""Answer the question using only the provided context.

Context:
{context}

Question: {query}

Provide a detailed answer with [N] citations."""

    answer_msg = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        messages=[{"role": "user", "content": generate_prompt}]
    )
    initial_answer = answer_msg.content[0].text

    # Step 2: Verify faithfulness
    verify_prompt = f"""Review this answer for faithfulness to the source context.

SOURCE CONTEXT:
{context}

QUESTION: {query}

GENERATED ANSWER:
{initial_answer}

For each sentence in the answer:
1. Is it supported by the context? (YES / PARTIAL / NO)
2. If NO or PARTIAL, what should be corrected or removed?

Then provide a REVISED ANSWER that removes or corrects any unfaithful content.
Mark any sentences you removed or changed with [CORRECTED] or [REMOVED]."""

    verify_msg = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1500,
        messages=[{"role": "user", "content": verify_prompt}]
    )

    return verify_msg.content[0].text
```

### 4.3 주장 분해를 통한 검증(Claim Decomposition for Verification)

복잡한 답변을 개별 검증을 위한 원자적 주장으로 분해합니다:

```python
decomposition_prompt = """Decompose the following answer into atomic claims,
then verify each claim against the source documents.

SOURCE DOCUMENTS:
{context}

ANSWER TO VERIFY:
{answer}

STEPS:
1. List each atomic claim (one fact per line)
2. For each claim, find the supporting evidence in the documents
3. Mark each claim as:
   - SUPPORTED: Direct evidence in documents
   - INFERRED: Reasonable inference from documents (explain the inference)
   - UNSUPPORTED: No evidence in documents
   - CONTRADICTED: Documents say the opposite

Format:
| # | Claim | Verdict | Evidence |
|---|-------|---------|----------|
| 1 | ...   | ...     | ...      |

Only claims marked SUPPORTED should remain in the final answer."""
```

---

## 5. 충돌하는 소스 처리(Handling Conflicting Sources)

실제 문서는 종종 서로 일치하지 않습니다. RAG 프롬프트에는 모순을 관리하기 위한 전략이 필요합니다.

### 5.1 충돌 감지 패턴(Conflict Detection Pattern)

```python
conflict_aware_prompt = """Answer the question using the provided documents.

IMPORTANT: If documents provide CONFLICTING information:
1. Present both viewpoints clearly
2. Identify the source of each viewpoint
3. Note the nature of the conflict (factual disagreement, different time periods,
   different scopes, etc.)
4. If one source appears more authoritative or recent, note that -- but present
   both views

DO NOT silently pick one source over another. The user must be aware of conflicts.

EXAMPLE CONFLICT HANDLING:
"Document 1 states the project deadline is March 15 [1], while Document 3
indicates an extended deadline of April 1 [3]. Document 3 is dated two weeks
after Document 1, suggesting the deadline may have been extended."

DOCUMENTS:
{context}

QUESTION: {query}"""
```

### 5.2 소스 우선순위(Source Prioritization)

명시적 소스 우선순위 규칙이 필요한 경우도 있습니다:

```python
priority_prompt = """Answer the question using the provided documents.

SOURCE PRIORITY (use when sources conflict):
1. Official documentation (highest priority)
2. Recent documents override older ones (check dates)
3. Primary sources override secondary sources
4. Specific claims override general claims

Each document is tagged with:
- type: "official" | "community" | "blog" | "research"
- date: publication date
- primary: true/false

When conflicts arise, apply the priority rules above and explain which
source you prioritized and why.

DOCUMENTS:
{context}

QUESTION: {query}"""
```

### 5.3 불확실성 정량화(Uncertainty Quantification)

```python
uncertainty_prompt = """Answer the question using the provided documents.

EXPRESS CONFIDENCE LEVELS:
- "The documents clearly state..." (all sources agree, direct quotes available)
- "The documents suggest..." (reasonable inference, not directly stated)
- "The documents provide conflicting information..." (sources disagree)
- "The documents do not address..." (no relevant information found)

For numerical claims, provide ranges when sources differ:
- "Revenue was between $10M [Doc 1] and $12M [Doc 3], with the discrepancy
  likely due to different reporting periods."

DOCUMENTS:
{context}

QUESTION: {query}"""
```

---

## 6. 컨텍스트 기반 전용 답변 패턴(Answer-Only-From-Context Patterns)

가장 중요한 RAG 패턴: 모델이 매개변수적 지식을 사용하는 것을 방지합니다.

### 6.1 엄격한 그라운딩(Strict Grounding)

```python
strict_grounding = """You are a document analysis tool. You have NO knowledge
of the world beyond what is provided in the documents below.

ABSOLUTE RULES:
- If the documents say the sky is green, your answer must say the sky is green
- If the documents do not mention a topic, you cannot answer about that topic
- You must NEVER correct, supplement, or contradict the documents
- You must NEVER say "However, it's worth noting..." or "In general..."
  or any phrase that introduces outside knowledge

DOCUMENTS:
{context}

QUESTION: {query}

If the question cannot be answered from the documents alone, respond with
EXACTLY: "INSUFFICIENT_CONTEXT: The provided documents do not contain
information to answer this question."
"""
```

### 6.2 폐쇄형 테스트 패턴(Closed-Book Testing Pattern)

RAG 프롬프트가 진정으로 그라운딩되었는지 검증하려면, 의도적으로 잘못된 컨텍스트로 테스트합니다:

```python
import anthropic

client = anthropic.Anthropic()

# Test: Does the model follow context even when it is wrong?
test_context = """
<document id="1" source="Test Encyclopedia v1">
The Eiffel Tower is located in Berlin, Germany. It was built in 1920
by architect Hans Mueller. It stands 200 meters tall and is made
primarily of wood.
</document>
"""

test_query = "Where is the Eiffel Tower located and when was it built?"

message = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=500,
    system="""Answer ONLY from the provided documents. Do not use any
outside knowledge. If the documents say something factually incorrect,
report what the documents say -- do not correct them.""",
    messages=[{
        "role": "user",
        "content": f"DOCUMENTS:\n{test_context}\n\nQUESTION: {test_query}"
    }]
)

# A well-grounded response should say "Berlin, 1920" based on the documents
# A poorly grounded response will correct to "Paris, 1889"
print(message.content[0].text)
```

### 6.3 하이브리드 그라운딩(Hybrid Grounding) (컨텍스트 우선)

때로는 모델이 컨텍스트를 선호하되 자체 지식으로 보완하길 원합니다:

```python
hybrid_grounding = """Answer the question using the provided documents as your
PRIMARY source. You may supplement with your general knowledge ONLY when:

1. The documents provide a partial answer and you are filling in widely-known
   background context
2. You clearly mark supplemented information with [General Knowledge]
3. The supplemented information does not contradict the documents

PRIORITY: Document claims > General knowledge

DOCUMENTS:
{context}

QUESTION: {query}

In your answer, use:
- [Doc N] for document-sourced claims
- [General Knowledge] for any supplemented information"""
```

---

## 7. 다중 문서 합성(Multi-Document Synthesis)

여러 문서에 걸친 합성은 단일 문서 질의응답보다 어렵습니다. 모델은 출처 표시를 유지하면서 다양한 소스의 정보를 통합해야 합니다.

### 7.1 합성 프롬프트 패턴(Synthesis Prompt Pattern)

```python
import anthropic

client = anthropic.Anthropic()

def multi_doc_synthesis(query: str, documents: list[dict]) -> str:
    """Synthesize an answer across multiple documents."""
    doc_block = ""
    for i, doc in enumerate(documents):
        doc_block += f"""
<document id="{i+1}" title="{doc['title']}" date="{doc.get('date', 'N/A')}">
{doc['text']}
</document>
"""

    prompt = f"""Synthesize information from multiple documents to answer the question.

DOCUMENTS:
{doc_block}

QUESTION: {query}

SYNTHESIS INSTRUCTIONS:
1. Read ALL documents before answering
2. Identify information relevant to the question from each document
3. Combine complementary information (different aspects of the same topic)
4. Resolve contradictions using the most recent or authoritative source
5. Cite each document that contributed to your synthesis

FORMAT YOUR RESPONSE AS:
## Answer
[Your synthesized answer with inline citations]

## Sources Used
[List each document and what it contributed to the answer]

## Information Gaps
[What the question asks that the documents do not address]"""

    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2048,
        messages=[{"role": "user", "content": prompt}]
    )
    return message.content[0].text
```

### 7.2 비교 합성(Comparative Synthesis)

문서들이 같은 주제에 대해 다른 관점을 제시하는 경우:

```python
comparative_prompt = """Analyze the following documents that discuss the same
topic from different perspectives.

DOCUMENTS:
{context}

QUESTION: {query}

Structure your response as:

CONSENSUS: What do all documents agree on?
DISAGREEMENTS: Where do documents differ? Present each position with its source.
UNIQUE INSIGHTS: What does each document contribute that others do not?
SYNTHESIS: Your integrated answer that acknowledges all perspectives.

For each point, cite the specific document(s)."""
```

### 7.3 시간적 합성(Temporal Synthesis)

문서가 서로 다른 시간대에 걸쳐 있는 경우:

```python
temporal_prompt = """The following documents describe the same topic at
different points in time. Synthesize a chronological understanding.

DOCUMENTS (arranged by date):
{context}

QUESTION: {query}

Provide:
1. TIMELINE: Key events/changes in chronological order with citations
2. CURRENT STATE: What the most recent document says (prefer this for
   "what is" questions)
3. EVOLUTION: How the answer to this question has changed over time
4. CAVEATS: Note if recent information may be outdated"""
```

---

## 8. 프롬프트 내 청크 관련성 점수 매기기(Chunk Relevance Scoring in Prompts)

검색된 모든 청크가 동등하게 관련이 있지는 않습니다. LLM 자체를 사용하여 답변 생성 전(또는 도중)에 청크를 필터링하고 순위를 매길 수 있습니다.

### 8.1 관련성 점수를 통한 사전 필터링(Pre-Filtering with Relevance Scoring)

```python
import anthropic
import json

client = anthropic.Anthropic()

def score_and_filter_chunks(
    query: str,
    chunks: list[dict],
    threshold: float = 0.5
) -> list[dict]:
    """Use the LLM to score chunk relevance before generating an answer."""
    chunks_text = ""
    for i, chunk in enumerate(chunks):
        chunks_text += f"\nChunk {i+1}: {chunk['text'][:500]}\n"

    scoring_prompt = f"""Rate the relevance of each text chunk to the given question.

Question: {query}

Chunks:
{chunks_text}

For each chunk, provide a relevance score from 0.0 to 1.0:
- 1.0: Directly answers the question
- 0.7-0.9: Contains highly relevant information
- 0.4-0.6: Partially relevant, provides some context
- 0.1-0.3: Tangentially related
- 0.0: Completely irrelevant

Respond as JSON:
[
    {{"chunk_id": 1, "score": 0.9, "reason": "brief reason"}},
    ...
]"""

    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        messages=[{"role": "user", "content": scoring_prompt}]
    )

    scores = json.loads(message.content[0].text)
    relevant_ids = {s["chunk_id"] for s in scores if s["score"] >= threshold}
    return [c for i, c in enumerate(chunks) if (i + 1) in relevant_ids]
```

### 8.2 인라인 관련성 평가(Inline Relevance Assessment)

사전 필터링 대신, 모델에게 인라인으로 관련성을 평가하도록 요청합니다:

```python
inline_relevance = """Answer the question using the provided chunks.

BEFORE answering, assess each chunk's relevance:

CHUNKS:
{context}

QUESTION: {query}

STEP 1 - RELEVANCE ASSESSMENT:
For each chunk, write one line: "Chunk N: [RELEVANT/PARTIAL/IRRELEVANT] - reason"

STEP 2 - ANSWER:
Using ONLY the chunks marked RELEVANT or PARTIAL, provide your answer.
Ignore IRRELEVANT chunks entirely -- do not let them influence your answer.

STEP 3 - CITATION:
Cite only the relevant chunks that supported your answer."""
```

### 8.3 검색 품질 피드백(Retrieval Quality Feedback)

시스템 개선을 위해 LLM을 사용하여 검색 품질에 대한 피드백을 제공합니다:

```python
retrieval_feedback_prompt = """Evaluate the quality of these retrieved chunks
for answering the given question.

QUESTION: {query}

RETRIEVED CHUNKS:
{context}

ASSESSMENT (respond as JSON):
{{
    "overall_quality": "sufficient|partial|insufficient",
    "coverage": "What aspects of the question are covered?",
    "gaps": "What aspects of the question are NOT covered by any chunk?",
    "noise": "Which chunks are irrelevant and should not have been retrieved?",
    "suggested_queries": ["Alternative search queries that might retrieve better results"],
    "can_answer": true/false,
    "confidence_if_answered": "high|medium|low"
}}"""
```

---

## 9. 답변 불가 시나리오 처리(Handling No-Answer Scenarios)

신뢰할 수 있는 RAG 시스템은 검색된 컨텍스트에 답변이 포함되지 않은 경우를 우아하게 처리해야 합니다.

### 9.1 명시적 답변 불가 지시(Explicit No-Answer Instructions)

```python
no_answer_prompt = """Answer the question using ONLY the provided documents.

CRITICAL: If the documents do not contain the answer:
- Do NOT guess or use your training knowledge
- Do NOT provide a partial answer and fill in gaps with assumptions
- DO respond with the structured no-answer format below

NO-ANSWER FORMAT:
{{
    "status": "no_answer",
    "reason": "Why the documents cannot answer this question",
    "closest_info": "What related information the documents DO contain",
    "suggested_reformulation": "A modified question the documents COULD answer"
}}

PARTIAL-ANSWER FORMAT (when documents partially address the question):
{{
    "status": "partial_answer",
    "answered": "What the documents DO answer",
    "unanswered": "What the documents do NOT address",
    "answer": "The partial answer with citations"
}}

DOCUMENTS:
{context}

QUESTION: {query}"""
```

### 9.2 단계적 보류(Tiered Abstention)

```python
tiered_abstention = """Answer the question based on the provided context.

Use these response levels:

LEVEL 1 - CONFIDENT ANSWER:
The context clearly and directly answers the question. Provide the answer
with citations.

LEVEL 2 - INFERRED ANSWER:
The context does not directly answer the question, but a reasonable
inference can be made. Provide the answer, mark inferences with
[Inferred from Doc N], and state your confidence level.

LEVEL 3 - SPECULATIVE:
The context is tangentially related. State what the context DOES say
about related topics, but explicitly note: "The context does not directly
address this question. The following related information may be useful..."

LEVEL 4 - NO ANSWER:
The context is irrelevant to the question. State: "The provided context
does not contain information relevant to this question."

Start your response by declaring your level: "Response Level: N"

CONTEXT:
{context}

QUESTION: {query}"""
```

### 9.3 대체 전략(Fallback Strategies)

```python
import anthropic

client = anthropic.Anthropic()

def rag_with_fallback(query: str, chunks: list[dict]) -> dict:
    """RAG with structured fallback for no-answer scenarios."""
    context = "\n".join(
        f"[Doc {i+1}] {c['text']}" for i, c in enumerate(chunks)
    )

    # First attempt: strict RAG
    strict_prompt = f"""Answer ONLY from the documents. If you cannot answer,
respond with exactly: NO_ANSWER

Documents:
{context}

Question: {query}"""

    msg = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        messages=[{"role": "user", "content": strict_prompt}]
    )

    answer = msg.content[0].text.strip()

    if answer == "NO_ANSWER":
        # Fallback: Try broader interpretation
        fallback_prompt = f"""The following documents do not directly answer the
question, but they may contain related information.

Documents:
{context}

Question: {query}

Provide:
1. What the documents DO say about related topics (with citations)
2. Why the specific question cannot be answered from these documents
3. What additional information would be needed to answer the question"""

        fallback_msg = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            messages=[{"role": "user", "content": fallback_prompt}]
        )

        return {
            "status": "no_direct_answer",
            "related_information": fallback_msg.content[0].text,
            "source": "fallback"
        }

    return {
        "status": "answered",
        "answer": answer,
        "source": "rag"
    }
```

---

## 10. RAG vs 긴 컨텍스트 트레이드오프(RAG vs Long-Context Trade-offs)

현대 모델은 매우 긴 컨텍스트 윈도우(100K-1M 토큰)를 지원합니다. 이는 질문을 제기합니다: 왜 검색을 해야 할까? 모든 문서를 컨텍스트에 넣지 않는 이유는?

### 10.1 비교 매트릭스

| 요소 | RAG (검색) | 긴 컨텍스트(Long-Context) (전체 주입) |
|------|-----------|--------------------------------------|
| **비용** | 낮음 (관련 청크만) | 높음 (쿼리마다 전체 코퍼스) |
| **지연 시간** | 낮음 (작은 프롬프트) | 높음 (긴 프롬프트) |
| **바늘 찾기(Needle-in-Haystack) 정확도** | 검색 품질에 의존 | 매우 긴 컨텍스트에서 어려울 수 있음 |
| **합성 정확도** | 검색되지 않은 관련 정보를 놓칠 수 있음 | 전체 그림을 가짐 |
| **최신성** | 인덱스 업데이트 용이 | 전체 코퍼스를 다시 제출해야 함 |
| **코퍼스 크기** | 수백만 문서까지 확장 가능 | 컨텍스트 윈도우에 의해 제한 |
| **복잡성** | 검색 인프라 필요 | 더 간단한 파이프라인 |
| **충실성** | 높음 (적은 컨텍스트 = 적은 산만함) | 너무 많은 컨텍스트에서 환각 가능 |

### 10.2 의사결정 프레임워크

```python
decision_prompt = """Given a use case, recommend whether to use RAG (retrieval)
or long-context (full document injection) approach.

USE CASE:
- Corpus size: {corpus_size}
- Query type: {query_type}  (factoid / analytical / comparative / creative)
- Latency requirement: {latency}
- Accuracy requirement: {accuracy}
- Update frequency: {update_freq}
- Budget constraint: {budget}

Analyze and recommend:
1. Primary recommendation (RAG vs Long-Context) with justification
2. Hybrid approach if applicable
3. Key risks of the recommended approach
4. Migration path (if starting with one and scaling to the other)"""
```

### 10.3 하이브리드 RAG + 긴 컨텍스트

최선의 접근 방식은 종종 하이브리드입니다: 후보를 검색한 다음 긴 컨텍스트에 넣습니다:

```python
import anthropic

client = anthropic.Anthropic()

def hybrid_rag(
    query: str,
    retrieved_chunks: list[dict],
    full_documents: list[dict],
    max_context_tokens: int = 50000
) -> str:
    """Hybrid approach: retrieve for precision, then include full context."""

    # Start with the most relevant chunks
    priority_context = ""
    for chunk in retrieved_chunks[:5]:  # top 5 most relevant
        priority_context += f"\n[HIGH RELEVANCE - {chunk['source']}]\n{chunk['text']}\n"

    # Then add full documents if space allows
    additional_context = ""
    for doc in full_documents:
        additional_context += f"\n[FULL DOCUMENT - {doc['title']}]\n{doc['text']}\n"

    prompt = f"""Answer the question using the provided materials.

The materials are organized by relevance:
1. HIGH RELEVANCE sections were specifically retrieved for your question
2. FULL DOCUMENT sections provide additional context

Prioritize HIGH RELEVANCE sections for direct answers, but use FULL DOCUMENT
sections for additional context and verification.

MATERIALS:
{priority_context}

{additional_context}

QUESTION: {query}

Provide a comprehensive answer with citations. Prefer citing HIGH RELEVANCE
sections when they contain the answer."""

    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2048,
        messages=[{"role": "user", "content": prompt}]
    )
    return message.content[0].text
```

### 10.4 선택 가이드

**RAG를 선택해야 할 때:**
- 코퍼스가 컨텍스트 윈도우를 초과 (수백만 문서)
- 낮은 지연 시간 필요 (빠른 쿼리)
- 비용 민감성 (토큰당 지불)
- 빈번한 코퍼스 업데이트 (매일 새 문서)
- 많은 사용자로 확장 필요

**긴 컨텍스트를 선택해야 할 때:**
- 코퍼스가 컨텍스트 윈도우에 맞음 (< 100K 토큰)
- 질문이 모든 문서에 대한 전체적 이해 필요
- 최대 정확도 필요 (검색 오류 없음)
- 간단한 파이프라인 선호
- 일회성 분석 작업

**하이브리드를 선택해야 할 때:**
- 중간 크기 코퍼스 (컨텍스트에 맞지만 비용이 많이 듦)
- 사실 기반과 합성 질문의 혼합
- 정밀도(검색)와 재현율(전체 컨텍스트) 모두 필요
- 다양한 쿼리 유형을 처리해야 하는 프로덕션 시스템 구축

---

## 연습문제

### 연습문제 1: 컨텍스트 주입 설계

회사의 내부 지식 베이스를 위한 RAG 시스템을 구축하고 있습니다. 지식 베이스에는 다음이 포함됩니다:
- HR 정책 (20개 문서, 각 약 50페이지)
- 엔지니어링 문서 (500개 문서, 다양한 길이)
- 회의록 (10,000개 문서, 1-3페이지)

이러한 서로 다른 문서 유형을 처리하는 컨텍스트 주입 프롬프트를 설계하세요. 프롬프트는 문서에 유형, 날짜, 작성자를 태깅해야 합니다. 프롬프트 템플릿을 보여주고 설계 선택을 설명하세요.

<details><summary>정답 보기</summary>

```python
import anthropic

client = anthropic.Anthropic()

def corporate_kb_rag(query: str, retrieved_chunks: list[dict]) -> str:
    """RAG prompt designed for a heterogeneous corporate knowledge base."""

    # Format chunks with rich metadata
    context_parts = []
    for i, chunk in enumerate(retrieved_chunks):
        doc_type = chunk.get("type", "unknown")  # "hr_policy", "engineering", "meeting"
        tag = (
            f'<document id="{i+1}" '
            f'type="{doc_type}" '
            f'title="{chunk["title"]}" '
            f'date="{chunk.get("date", "unknown")}" '
            f'author="{chunk.get("author", "unknown")}" '
            f'section="{chunk.get("section", "")}">\n'
            f'{chunk["text"]}\n'
            f'</document>'
        )
        context_parts.append(tag)

    context_block = "\n\n".join(context_parts)

    system = """You are an internal knowledge assistant for the company.

SOURCE PRIORITY RULES:
1. HR Policies: These are authoritative. Quote them exactly. Always cite
   the specific policy section number.
2. Engineering Docs: These are technical references. Cite version/date
   as engineering docs may be outdated.
3. Meeting Notes: These provide context but are NOT authoritative for
   policy or technical decisions. Use them for "who decided what" or
   "what was discussed" questions only.

CONFLICT RESOLUTION:
- HR Policy > Engineering Doc > Meeting Notes
- More recent documents > older documents (check dates)
- If an engineering doc contradicts an HR policy, cite both and note
  that the HR policy is authoritative

ANSWERING RULES:
- Always cite document ID and type
- For HR questions: provide the exact policy text
- For technical questions: note the doc date in case info is outdated
- For "who/when/why" questions: meeting notes are acceptable sources"""

    user_message = f"""RETRIEVED DOCUMENTS:
{context_block}

QUESTION: {query}"""

    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1500,
        system=system,
        messages=[{"role": "user", "content": user_message}]
    )
    return message.content[0].text
```

설계 선택:
1. **풍부한 메타데이터가 포함된 XML 스타일 태그**: 모델이 특정 문서와 그 속성을 참조하기 쉽게 만듭니다.
2. **규칙을 위한 시스템 프롬프트**: 동작 규칙을 가변 콘텐츠와 분리합니다.
3. **소스 우선순위 계층**: HR 정책은 권위 있고, 회의록은 맥락적입니다. 이것은 모델이 비공식 회의 토론을 회사 정책으로 인용하는 것을 방지합니다.
4. **날짜 인식**: 엔지니어링 문서는 오래될 수 있습니다; 프롬프트가 날짜 확인을 권장합니다.
5. **유형별 답변 규칙**: 서로 다른 문서 유형은 다른 응답 스타일을 보장합니다.

</details>

### 연습문제 2: 충실성 테스트

RAG 시스템을 위한 충실성 테스트를 설계하세요. 검색된 컨텍스트에 의도적으로 잘못된 정보가 포함된 세 가지 테스트 케이스를 만드세요. 각 테스트 케이스에 대해 다음을 보여주세요:
1. 쿼리
2. (잘못된) 컨텍스트
3. 충실한 응답은 어떤 모습인지
4. 불충실한 응답은 어떤 모습인지

<details><summary>정답 보기</summary>

```python
import anthropic

client = anthropic.Anthropic()

# Faithfulness test suite
test_cases = [
    {
        "name": "Factual Override",
        "query": "What is the boiling point of water?",
        "context": """<document id="1" source="Internal Chemistry Guide v2">
Water (H2O) boils at 90 degrees Celsius at standard atmospheric pressure.
This is a fundamental property used in our laboratory calibration procedures.
</document>""",
        "faithful_response": "According to the Internal Chemistry Guide [1], "
                            "water boils at 90 degrees Celsius at standard "
                            "atmospheric pressure.",
        "unfaithful_response": "Water boils at 100 degrees Celsius at standard "
                              "atmospheric pressure. Note: the provided document "
                              "contains an error."
    },
    {
        "name": "Missing Information",
        "query": "What are the three laws of thermodynamics?",
        "context": """<document id="1" source="Physics Summary">
The first law of thermodynamics states that energy cannot be created or
destroyed, only transformed. The second law states that entropy in an
isolated system always increases.
</document>""",
        "faithful_response": "Based on the provided document [1], two laws of "
                            "thermodynamics are described: the first law (energy "
                            "conservation) and the second law (entropy increase). "
                            "The document does not mention the third law.",
        "unfaithful_response": "The three laws of thermodynamics are: 1) Energy "
                              "conservation [1], 2) Entropy increase [1], and 3) "
                              "As temperature approaches absolute zero, entropy "
                              "approaches a minimum value."
    },
    {
        "name": "Numerical Contradiction",
        "query": "What is the company's annual revenue?",
        "context": """<document id="1" source="Q3 2024 Earnings Report">
Total annual revenue for fiscal year 2024 was $3.2 billion, representing
a 15% decrease from the previous year.
</document>
<document id="2" source="Investor Presentation Jan 2025">
Our annual revenue reached $4.1 billion in FY2024, a record high for
the company.
</document>""",
        "faithful_response": "The sources provide conflicting revenue figures for "
                            "FY2024. The Q3 2024 Earnings Report states revenue was "
                            "$3.2 billion (a 15% decrease) [1], while the January 2025 "
                            "Investor Presentation claims $4.1 billion (a record high) [2]. "
                            "These figures are contradictory.",
        "unfaithful_response": "The company's annual revenue for FY2024 was $4.1 "
                              "billion, a record high."
    }
]

# Run the test
grounding_system = """Answer ONLY from the provided documents. Do not use any
outside knowledge. If documents contain incorrect information, report what the
documents say. If documents conflict, report all positions."""

for test in test_cases:
    print(f"\n--- Test: {test['name']} ---")
    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=500,
        system=grounding_system,
        messages=[{
            "role": "user",
            "content": f"DOCUMENTS:\n{test['context']}\n\nQUESTION: {test['query']}"
        }]
    )
    response = message.content[0].text
    print(f"Response: {response}")
    print(f"Expected faithful pattern: {test['faithful_response'][:100]}...")
```

충실한 RAG 시스템은 다음을 해야 합니다:
1. 잘못된 사실을 있는 그대로 보고 (테스트 1)
2. 부족한 부분을 채우기보다 누락된 정보를 인정 (테스트 2)
3. 하나를 조용히 선택하기보다 모순을 표면화 (테스트 3)

</details>

### 연습문제 3: 다중 문서 합성

같은 제품에 대한 네 개의 문서(제품 사양서, 사용자 매뉴얼, 고객 리뷰, 지원 티켓)에서 정보를 합성하는 RAG 프롬프트를 작성하세요. 프롬프트는 각 소스에 적절한 가중치를 부여하는 포괄적인 답변을 생성해야 합니다. 구체적인 예시로 시연하세요.

<details><summary>정답 보기</summary>

```python
import anthropic

client = anthropic.Anthropic()

documents = [
    {
        "id": 1,
        "type": "spec_sheet",
        "title": "SmartWidget Pro - Technical Specifications",
        "text": """SmartWidget Pro Model SW-3000
Battery: 4000mAh lithium-ion, rated for 12 hours continuous use
Processor: ARM Cortex-M7 at 480MHz
Connectivity: Wi-Fi 6, Bluetooth 5.2, USB-C
Operating Temperature: -10C to 45C
Weight: 285g
Water Resistance: IP67 rated
Warranty: 2-year limited manufacturer warranty"""
    },
    {
        "id": 2,
        "type": "user_manual",
        "title": "SmartWidget Pro - User Guide v2.1",
        "text": """Battery Life: Under typical usage (screen brightness 50%,
Wi-Fi connected), expect 8-10 hours of battery life. Charging takes
approximately 2 hours via the included USB-C cable. To maximize battery
longevity, avoid charging above 80% for daily use. The device will display
a low battery warning at 15% remaining.

Note: Battery life may be significantly reduced in temperatures below 0C."""
    },
    {
        "id": 3,
        "type": "customer_review",
        "title": "Amazon Review by VerifiedBuyer_2024",
        "text": """I have been using the SmartWidget Pro for 3 months daily. Battery
life is honestly closer to 6-7 hours with heavy use (Bluetooth + Wi-Fi +
high brightness). The IP67 water resistance claim held up -- I accidentally
dropped it in a sink and it was fine. The device feels premium at 285g,
not too heavy. My main complaint: the USB-C port feels flimsy after
repeated plugging."""
    },
    {
        "id": 4,
        "type": "support_ticket",
        "title": "Support Ticket #41892 - Battery Drain Issue",
        "text": """Customer reported battery draining in 4 hours after firmware
update v3.2.1. Engineering confirmed bug in background process management.
Fix deployed in v3.2.2 patch. Post-patch testing showed battery life
restored to 9-hour average. Customers on v3.2.1 should update immediately.
Issue affected approximately 12% of devices."""
    }
]

# Build the prompt
doc_block = ""
for doc in documents:
    doc_block += (
        f'\n<document id="{doc["id"]}" type="{doc["type"]}" '
        f'title="{doc["title"]}">\n{doc["text"]}\n</document>\n'
    )

prompt = f"""Synthesize information from multiple sources to answer the question.

SOURCE WEIGHT GUIDE:
- spec_sheet: Authoritative for rated/designed specifications
- user_manual: Authoritative for recommended usage and expected real-world behavior
- customer_review: Valuable for real-world experience (but subjective, single user)
- support_ticket: Valuable for known issues and fixes (may not apply to all users)

SYNTHESIS RULES:
1. Start with the authoritative specification
2. Layer on real-world expectations from the user manual
3. Add real-world experience from reviews (note sample size limitations)
4. Flag any known issues from support tickets
5. When sources disagree on a measurable quantity, present the range and
   explain why they differ

DOCUMENTS:
{doc_block}

QUESTION: How long does the SmartWidget Pro battery last?

Format your response as:
## Official Specification
[From spec sheet]
## Expected Real-World Performance
[From user manual]
## Actual User Experience
[From reviews]
## Known Issues
[From support tickets]
## Summary
[Integrated answer with all sources considered]"""

message = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1500,
    messages=[{"role": "user", "content": prompt}]
)
print(message.content[0].text)
```

이 프롬프트가 작동하는 이유:
1. **소스 유형화**: 각 문서 유형에 명시적인 신뢰 수준과 사용 사례가 있습니다.
2. **구조화된 합성**: 응답 형식이 소스 계층을 반영하여 각 정보의 출처를 명확하게 합니다.
3. **범위 제시**: 하나의 숫자를 선택하는 대신, 프롬프트가 소스 간 범위를 요청하여 더 정직합니다.
4. **알려진 문제 표시**: 지원 티켓은 사양서와 매뉴얼에서는 절대 언급하지 않을 문제를 드러냅니다.

</details>

### 연습문제 4: 답변 불가 처리

세 가지 구별되는 동작을 가진 RAG 프롬프트를 설계하세요: (1) 컨텍스트가 충분할 때 확신 있는 답변, (2) 컨텍스트가 불완전할 때 부분 답변, (3) 컨텍스트가 관련이 없을 때 우아한 보류. 같은 컨텍스트에 대해 세 가지 쿼리로 프롬프트를 테스트하세요: 완전히 답변 가능한 것, 부분적인 것, 전혀 답변할 수 없는 것.

<details><summary>정답 보기</summary>

```python
import anthropic

client = anthropic.Anthropic()

context = """
<document id="1" source="Employee Handbook 2024">
Vacation Policy: Full-time employees receive 15 days of paid vacation per year
for the first 3 years of employment, increasing to 20 days after 3 years.
Part-time employees receive a prorated amount based on their scheduled hours.
Vacation days do not roll over to the next calendar year. Employees must submit
vacation requests at least 2 weeks in advance through the HR portal.
</document>

<document id="2" source="Benefits Summary">
Health Insurance: The company offers three health plan tiers: Basic, Standard,
and Premium. All full-time employees are eligible after 30 days of employment.
The company covers 80% of the premium for Basic, 70% for Standard, and 60%
for Premium plans. Dental and vision are included in Standard and Premium tiers.
</document>
"""

system = """You are an HR knowledge assistant. Answer questions using ONLY the
provided documents.

RESPONSE PROTOCOL:

If the documents FULLY answer the question:
- Start with: "CONFIDENCE: HIGH"
- Provide the complete answer with [Doc N] citations
- End with source references

If the documents PARTIALLY answer the question:
- Start with: "CONFIDENCE: PARTIAL"
- Answer what you can with citations
- Explicitly list what aspects of the question remain unanswered
- Suggest where the user might find the missing information

If the documents DO NOT address the question:
- Start with: "CONFIDENCE: NONE"
- State that the provided documents do not contain relevant information
- List what topics the documents DO cover (so the user knows what to ask about)
- Suggest who or where the user should direct their question"""

# Test 1: Fully answerable
q1 = "How many vacation days do I get after 5 years at the company?"

# Test 2: Partially answerable
q2 = "What health insurance options are available and how much will I pay monthly?"

# Test 3: Not answerable
q3 = "What is the company's parental leave policy?"

for i, query in enumerate([q1, q2, q3], 1):
    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=500,
        system=system,
        messages=[{
            "role": "user",
            "content": f"DOCUMENTS:\n{context}\n\nQUESTION: {query}"
        }]
    )
    print(f"\n--- Query {i}: {query} ---")
    print(message.content[0].text)
```

예상되는 동작:
1. **Q1 (HIGH)**: "5년 후(3년 이상)에는 연간 20일의 유급 휴가를 받습니다 [Doc 1]."
2. **Q2 (PARTIAL)**: "Basic, Standard, Premium 세 가지 등급이 있습니다. 회사가 각각 80%/70%/60%를 부담합니다 [Doc 2]. 하지만 문서에는 실제 월별 보험료 금액이 명시되어 있지 않습니다. 구체적인 금액은 HR 부서 또는 복리후생 포털에 문의하세요."
3. **Q3 (NONE)**: "제공된 문서에는 육아 휴가에 대한 정보가 포함되어 있지 않습니다. 사용 가능한 문서는 휴가 정책과 건강보험 혜택을 다룹니다. 육아 휴가 정보는 HR 부서에 직접 문의하세요."

</details>

### 연습문제 5: RAG vs 긴 컨텍스트 결정

회사의 코드베이스에 대한 질문에 답변하는 시스템을 설계하고 있습니다. 코드베이스는 5,000개 파일, 총 200만 줄의 코드로 구성되어 있습니다. 아키텍트가 RAG, 긴 컨텍스트, 하이브리드 접근 방식 중 결정하는 데 도움이 되는 프롬프트를 작성하세요. 구체적인 기준을 포함하고 근거와 함께 추천을 제공하세요.

<details><summary>정답 보기</summary>

```python
import anthropic

client = anthropic.Anthropic()

analysis_prompt = """You are a system architect evaluating approaches for a
code Q&A system. Analyze the following use case and recommend an approach.

USE CASE PROFILE:
- Corpus: 5,000 source files, ~2 million lines of code (Python/TypeScript)
- Estimated tokens: ~8 million tokens (entire codebase)
- Query types:
  a) "How does the authentication middleware work?" (requires reading 2-5 files)
  b) "What is the data flow from API to database for user creation?" (requires
     reading 5-15 files across multiple directories)
  c) "Find all places where we handle rate limiting" (search across all files)
  d) "Why does this function exist? What is its history?" (needs git context)
- Query volume: ~200 queries/day from a team of 30 developers
- Latency requirement: < 10 seconds for most queries
- Budget: $500/month for LLM API costs
- Freshness: Code changes multiple times daily; answers must reflect current code

APPROACH OPTIONS:
1. Pure RAG: Embed all code, retrieve top-k chunks per query
2. Pure Long-Context: Send relevant files (up to 200K context window)
3. Hybrid: Retrieve candidates, then stuff into long context
4. Agentic: LLM uses tools to search/read files iteratively

For each approach, analyze:
- Will it work for each query type (a-d)?
- Cost per query estimate
- Latency estimate
- Accuracy tradeoffs
- Infrastructure complexity

Then provide your recommendation with specific implementation guidance."""

message = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=3000,
    messages=[{"role": "user", "content": analysis_prompt}]
)
print(message.content[0].text)
```

예상되는 추천 요약:

**추천: 에이전틱 폴백이 있는 하이브리드 접근 방식**

근거:
- **순수 RAG는 실패합니다** -- 쿼리 유형 (b)와 (d)에서 파일 간 데이터 흐름과 이력 컨텍스트는 격리된 청크로 캡처할 수 없습니다. 코드는 높은 파일 간 의존성을 가집니다.
- **순수 긴 컨텍스트는 실패합니다** -- 코퍼스 크기에서 8M 토큰은 어떤 컨텍스트 윈도우도 훨씬 초과합니다. 관련 파일을 선택하는 것조차 검색이 필요합니다.
- **하이브리드가 최선입니다**: 코드 인식 검색(원시 텍스트 청킹이 아닌 AST 기반 청킹)을 사용하여 후보 파일을 찾은 다음 긴 컨텍스트(100K-200K 토큰)에 넣습니다. 이는 쿼리 유형 (a), (b), (c)를 처리합니다.
- **에이전틱 폴백** -- 쿼리 유형 (d)에서 하이브리드 접근 방식이 낮은 확신도를 반환하면 LLM이 도구(파일 검색, git log, grep)를 사용하여 코드베이스를 반복적으로 탐색합니다.

비용 추정: 하이브리드 쿼리당 약 $0.15-0.50, 에이전틱 쿼리당 약 $1-2. 하루 200개 쿼리에서 월 약 $300-400 -- 예산 내.

</details>

---

**이전**: [코드 생성 프롬프팅](./09_Code_Generation_Prompting.md) | **다음**: [프롬프트 최적화](./11_Prompt_Optimization.md)
