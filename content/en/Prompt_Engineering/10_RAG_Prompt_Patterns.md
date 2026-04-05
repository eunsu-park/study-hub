# 10. RAG Prompt Patterns

**Previous**: [Code Generation Prompting](./09_Code_Generation_Prompting.md) | **Next**: [Prompt Optimization](./11_Prompt_Optimization.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Design retrieval-augmented generation prompts that ground LLM responses in retrieved context
2. Implement citation and attribution patterns that trace claims back to source documents
3. Apply faithfulness techniques to prevent hallucination beyond the provided context
4. Handle conflicting sources, missing information, and multi-document synthesis in prompts
5. Evaluate trade-offs between RAG and long-context approaches for different use cases

---

Retrieval-Augmented Generation (RAG) is the dominant pattern for building knowledge-grounded LLM applications. Instead of relying on the model's parametric knowledge (what it learned during training), RAG retrieves relevant documents at query time and injects them into the prompt. This allows the model to answer questions about private data, recent events, or domain-specific knowledge that was not in its training set.

However, RAG is only as good as its prompts. A poorly designed RAG prompt can ignore retrieved context, hallucinate answers not present in the documents, fail to cite sources, or crumble when the retrieved chunks are irrelevant or contradictory. This lesson covers the prompt patterns that make RAG systems reliable.

## Table of Contents

1. [Retrieval-Augmented Generation Overview](#1-retrieval-augmented-generation-overview)
2. [Context Injection Patterns](#2-context-injection-patterns)
3. [Citation and Attribution Prompts](#3-citation-and-attribution-prompts)
4. [Faithfulness and Grounding](#4-faithfulness-and-grounding)
5. [Handling Conflicting Sources](#5-handling-conflicting-sources)
6. [Answer-Only-From-Context Patterns](#6-answer-only-from-context-patterns)
7. [Multi-Document Synthesis](#7-multi-document-synthesis)
8. [Chunk Relevance Scoring in Prompts](#8-chunk-relevance-scoring-in-prompts)
9. [Handling No-Answer Scenarios](#9-handling-no-answer-scenarios)
10. [RAG vs Long-Context Trade-offs](#10-rag-vs-long-context-trade-offs)

---

## 1. Retrieval-Augmented Generation Overview

### 1.1 The RAG Pipeline

A RAG system has three stages:

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

- **Retrieve**: Find relevant documents using vector search, keyword search, or hybrid approaches
- **Augment**: Insert the retrieved documents into a prompt template
- **Generate**: Send the augmented prompt to the LLM for response generation

### 1.2 Why Prompting Matters in RAG

The retrieval step gets the right documents, but the *prompt* determines how the model uses them:

| Prompt Quality | Behavior |
|---------------|----------|
| No context framing | Model ignores context, answers from parametric memory |
| Context present but no grounding instruction | Model mixes context with its own knowledge |
| Grounded but no citation instruction | Model answers correctly but provides no traceability |
| Fully specified RAG prompt | Model answers from context, cites sources, admits gaps |

### 1.3 Basic RAG Prompt Structure

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

## 2. Context Injection Patterns

How you inject context into the prompt significantly affects how the model processes it. Different patterns suit different use cases.

### 2.1 Flat Context Injection

The simplest pattern: concatenate all chunks into one block.

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

**When to use**: Few chunks (1-3), all highly relevant, no need for source tracking.

**Limitation**: The model cannot distinguish between chunks, making citation impossible.

### 2.2 Tagged Context Injection

Label each chunk with metadata for traceability:

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

**When to use**: Most RAG applications. The XML-style tags help Claude parse and reference specific documents.

### 2.3 Hierarchical Context Injection

For long documents, inject summaries first, then relevant details:

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

### 2.4 Role-Separated Context

Use the system prompt for instructions and the user message for context + query:

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

This pattern is often the most effective because:
- The system prompt establishes consistent behavior across many queries
- The user message contains only the variable parts (context + question)
- It is easy to swap contexts without changing the behavioral instructions

---

## 3. Citation and Attribution Prompts

Citation is critical for trust and verification. Without it, users cannot distinguish RAG-grounded answers from hallucination.

### 3.1 Inline Citation Pattern

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

### 3.2 Quote-Based Citation

When you need verifiable attribution, require direct quotes:

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

### 3.3 Structured Citation Output

For programmatic consumption, request structured citation data:

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

### 3.4 Citation Verification

After getting a cited answer, you can verify citations in a follow-up:

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

## 4. Faithfulness and Grounding

Faithfulness means the model's answer is fully supported by the provided context without adding information from its training data.

### 4.1 Grounding Instructions

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

### 4.2 Faithfulness Verification Chain

Use a two-step process: generate, then verify:

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

### 4.3 Claim Decomposition for Verification

Break complex answers into atomic claims for individual verification:

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

## 5. Handling Conflicting Sources

Real-world documents often disagree. Your RAG prompts need strategies for managing contradictions.

### 5.1 Conflict Detection Pattern

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

### 5.2 Source Prioritization

Sometimes you want explicit source priority rules:

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

### 5.3 Uncertainty Quantification

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

## 6. Answer-Only-From-Context Patterns

The most critical RAG pattern: preventing the model from using its parametric knowledge.

### 6.1 Strict Grounding

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

### 6.2 Closed-Book Testing Pattern

To verify your RAG prompt is truly grounded, test with deliberately wrong context:

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

### 6.3 Hybrid Grounding (Context-Preferred)

Sometimes you want the model to prefer context but fall back to its knowledge:

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

## 7. Multi-Document Synthesis

Synthesis across multiple documents is harder than single-document QA. The model must integrate information from different sources while maintaining attribution.

### 7.1 Synthesis Prompt Pattern

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

### 7.2 Comparative Synthesis

When documents present different perspectives on the same topic:

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

### 7.3 Temporal Synthesis

When documents span different time periods:

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

## 8. Chunk Relevance Scoring in Prompts

Not all retrieved chunks are equally relevant. You can use the LLM itself to filter and rank chunks before (or during) answer generation.

### 8.1 Pre-Filtering with Relevance Scoring

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

### 8.2 Inline Relevance Assessment

Instead of pre-filtering, ask the model to assess relevance inline:

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

### 8.3 Retrieval Quality Feedback

Use the LLM to provide feedback on retrieval quality for system improvement:

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

## 9. Handling No-Answer Scenarios

A reliable RAG system must gracefully handle cases where the retrieved context does not contain the answer.

### 9.1 Explicit No-Answer Instructions

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

### 9.2 Tiered Abstention

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

### 9.3 Fallback Strategies

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

## 10. RAG vs Long-Context Trade-offs

Modern models support very long context windows (100K-1M tokens). This raises the question: why retrieve at all? Why not just put all your documents in the context?

### 10.1 Comparison Matrix

| Factor | RAG (Retrieval) | Long-Context (Stuffing) |
|--------|----------------|------------------------|
| **Cost** | Lower (only relevant chunks) | Higher (entire corpus per query) |
| **Latency** | Lower (smaller prompt) | Higher (longer prompt) |
| **Accuracy on needle-in-haystack** | Depends on retrieval quality | Can struggle with very long contexts |
| **Accuracy on synthesis** | May miss relevant info not retrieved | Has full picture |
| **Freshness** | Easy to update index | Must re-submit full corpus |
| **Corpus size** | Scales to millions of docs | Limited by context window |
| **Complexity** | Requires retrieval infrastructure | Simpler pipeline |
| **Faithfulness** | Higher (less context = less distraction) | May hallucinate with too much context |

### 10.2 Decision Framework

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

### 10.3 Hybrid RAG + Long-Context

The best approach is often a hybrid: retrieve candidates, then stuff them into a long context:

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

### 10.4 When to Choose Which

**Choose RAG when:**
- Corpus exceeds context window (millions of documents)
- Low latency required (fast queries)
- Cost sensitivity (pay per token)
- Frequent corpus updates (new documents daily)
- Need to scale to many users

**Choose Long-Context when:**
- Corpus fits in context window (< 100K tokens)
- Questions require holistic understanding of all documents
- Maximum accuracy needed (no retrieval errors)
- Simple pipeline preferred
- One-off analysis tasks

**Choose Hybrid when:**
- Medium corpus size (fits in context but is expensive)
- Mix of factoid and synthesis questions
- Need both precision (retrieval) and recall (full context)
- Building a production system that must handle diverse query types

---

## Exercises

### Exercise 1: Context Injection Design

You are building a RAG system for a company's internal knowledge base. The knowledge base contains:
- HR policies (20 documents, ~50 pages each)
- Engineering docs (500 documents, varying lengths)
- Meeting notes (10,000 documents, 1-3 pages each)

Design a context injection prompt that handles these different document types. Your prompt should tag documents with their type, date, and author. Show the prompt template and explain your design choices.

<details><summary>Show Answer</summary>

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

Design choices:
1. **XML-style tags** with rich metadata: Makes it easy for the model to reference specific documents and their properties.
2. **System prompt for rules**: Keeps behavioral rules separate from variable content.
3. **Source priority hierarchy**: HR policies are authoritative, meeting notes are contextual. This prevents the model from citing a casual meeting discussion as company policy.
4. **Date-awareness**: Engineering docs can become outdated; the prompt encourages checking dates.
5. **Type-specific answering rules**: Different document types warrant different response styles.

</details>

### Exercise 2: Faithfulness Testing

Design a faithfulness test for a RAG system. Create three test cases where the retrieved context contains deliberately incorrect information. For each test case, show:
1. The query
2. The (incorrect) context
3. What a faithful response looks like
4. What an unfaithful response looks like

<details><summary>Show Answer</summary>

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

A faithful RAG system should:
1. Report incorrect facts as-is (Test 1)
2. Acknowledge missing information rather than filling gaps (Test 2)
3. Surface contradictions rather than silently picking one (Test 3)

</details>

### Exercise 3: Multi-Document Synthesis

Write a RAG prompt that synthesizes information from four documents about the same product (a product spec sheet, a user manual, a customer review, and a support ticket). The prompt should produce a comprehensive answer that weighs each source appropriately. Demonstrate with a concrete example.

<details><summary>Show Answer</summary>

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

This prompt works because:
1. **Source typing**: Each document type has an explicit trust level and use case.
2. **Structured synthesis**: The response format mirrors the source hierarchy, making it clear where each piece of information comes from.
3. **Range presentation**: Rather than picking one number, the prompt asks for a range across sources, which is more honest.
4. **Known issues flagged**: Support tickets reveal problems that specs and manuals would never mention.

</details>

### Exercise 4: No-Answer Handling

Design a RAG prompt that has three distinct behaviors: (1) confident answer when context is sufficient, (2) partial answer when context is incomplete, and (3) graceful abstention when context is irrelevant. Test your prompt with three queries against the same context: one that can be fully answered, one partially, and one not at all.

<details><summary>Show Answer</summary>

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

Expected behaviors:
1. **Q1 (HIGH)**: "After 5 years (which is more than 3), you receive 20 days of paid vacation per year [Doc 1]."
2. **Q2 (PARTIAL)**: "Three tiers are available: Basic, Standard, Premium. The company covers 80%/70%/60% respectively [Doc 2]. However, the documents do not specify the actual monthly premium amounts. Contact HR or check the Benefits Portal for specific dollar amounts."
3. **Q3 (NONE)**: "The provided documents do not contain information about parental leave. The available documents cover: vacation policy and health insurance benefits. For parental leave information, contact the HR department directly."

</details>

### Exercise 5: RAG vs Long-Context Decision

You are designing a system to answer questions about a company's codebase. The codebase has 5,000 files totaling 2 million lines of code. Write a prompt that helps an architect decide between RAG, long-context, and hybrid approaches. Include specific criteria and provide the recommendation with your reasoning.

<details><summary>Show Answer</summary>

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

Expected recommendation summary:

**Recommended: Hybrid approach with agentic fallback**

Reasoning:
- **Pure RAG fails** for query types (b) and (d) -- cross-file data flows and historical context cannot be captured in isolated chunks. Code has high inter-file dependency.
- **Pure Long-Context fails** on corpus size -- 8M tokens far exceeds any context window. Even selecting relevant files requires search.
- **Hybrid is best**: Use code-aware retrieval (AST-based chunking, not raw text chunking) to find candidate files, then stuff them into a long context (100K-200K tokens). This handles query types (a), (b), and (c).
- **Agentic fallback** for query type (d): When the hybrid approach returns low confidence, let the LLM use tools (file search, git log, grep) to iteratively explore the codebase.

Cost estimate: ~$0.15-0.50 per hybrid query, ~$1-2 per agentic query. At 200 queries/day, roughly $300-400/month -- within budget.

</details>

---

**Previous**: [Code Generation Prompting](./09_Code_Generation_Prompting.md) | **Next**: [Prompt Optimization](./11_Prompt_Optimization.md)
