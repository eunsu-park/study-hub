# 10_rag_prompt_patterns.py — RAG prompt patterns: context injection, citations, faithfulness
#
# Run: python 10_rag_prompt_patterns.py

import anthropic
import json

# ---------------------------------------------------------------------------
# 1. Simulated knowledge base (in production, these come from a vector DB)
# ---------------------------------------------------------------------------
KNOWLEDGE_BASE = [
    {
        "id": "doc-1",
        "title": "Company Leave Policy",
        "content": (
            "Employees are entitled to 20 days of paid annual leave per year. "
            "Unused leave may be carried over up to 5 days into the next calendar year. "
            "Leave requests must be submitted at least 2 weeks in advance."
        ),
    },
    {
        "id": "doc-2",
        "title": "Remote Work Guidelines",
        "content": (
            "Employees may work remotely up to 3 days per week with manager approval. "
            "A stable internet connection and a dedicated workspace are required. "
            "Core collaboration hours are 10:00-15:00 local time."
        ),
    },
    {
        "id": "doc-3",
        "title": "Expense Reimbursement",
        "content": (
            "Business expenses over $25 require a receipt. "
            "Reimbursement requests must be filed within 30 days. "
            "Pre-approval is required for any single expense exceeding $500."
        ),
    },
]


def format_context(documents: list[dict]) -> str:
    """Format retrieved documents into a numbered context block."""
    parts = []
    for i, doc in enumerate(documents, 1):
        parts.append(
            f"[Source {i} — {doc['title']} (id: {doc['id']})]\n{doc['content']}"
        )
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# 2. RAG prompt with citation requirements
# ---------------------------------------------------------------------------
RAG_SYSTEM_PROMPT = """\
You are a helpful company assistant. Answer the user's question using ONLY \
the provided context documents.

Rules:
- Cite every factual claim with [Source N].
- If the context does not contain enough information, say "I don't have \
enough information to answer that."
- Never fabricate facts beyond what the sources state."""

RAG_USER_TEMPLATE = """\
<context>
{context}
</context>

Question: {question}"""


def ask_with_rag(client: anthropic.Anthropic, question: str) -> str:
    """Send a RAG-augmented query and return the answer."""
    context_block = format_context(KNOWLEDGE_BASE)
    user_message = RAG_USER_TEMPLATE.format(
        context=context_block, question=question
    )
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=512,
        system=RAG_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )
    return response.content[0].text


# ---------------------------------------------------------------------------
# 3. Faithfulness checker — uses a second LLM call
# ---------------------------------------------------------------------------
FAITHFULNESS_PROMPT = """\
You are a fact-checking auditor. Given a CONTEXT and an ANSWER, determine \
whether every claim in the ANSWER is supported by the CONTEXT.

Respond with a JSON object:
{{"faithful": true/false, "unsupported_claims": ["..."]}}

<context>
{context}
</context>

<answer>
{answer}
</answer>"""


def check_faithfulness(
    client: anthropic.Anthropic, answer: str
) -> dict:
    """Verify that the answer is grounded in the provided context."""
    context_block = format_context(KNOWLEDGE_BASE)
    prompt = FAITHFULNESS_PROMPT.format(context=context_block, answer=answer)
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=256,
        messages=[{"role": "user", "content": prompt}],
    )
    text = response.content[0].text
    # Extract JSON from the response
    try:
        start = text.index("{")
        end = text.rindex("}") + 1
        return json.loads(text[start:end])
    except (ValueError, json.JSONDecodeError):
        return {"faithful": None, "raw": text}


# ---------------------------------------------------------------------------
# 4. Demo runner
# ---------------------------------------------------------------------------
DEMO_QUESTIONS = [
    "How many days of remote work are allowed per week?",
    "What is the deadline for filing expense reimbursements?",
    "Can I carry over unused vacation days?",
    "What is the company's stock option plan?",  # not in context
]


def main() -> None:
    client = anthropic.Anthropic()  # uses ANTHROPIC_API_KEY env var

    for question in DEMO_QUESTIONS:
        print(f"\n{'=' * 60}")
        print(f"Q: {question}")
        print("-" * 60)

        try:
            answer = ask_with_rag(client, question)
            print(f"A: {answer}")

            # Run faithfulness check on the answer
            result = check_faithfulness(client, answer)
            faithful = result.get("faithful")
            tag = "PASS" if faithful else "FAIL" if faithful is False else "?"
            print(f"\nFaithfulness [{tag}]: {json.dumps(result, indent=2)}")

        except anthropic.APIError as exc:
            print(f"[API Error] {exc}")


if __name__ == "__main__":
    main()
