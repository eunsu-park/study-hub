# Exercise: Lesson 10 — RAG Prompt Patterns
# Complete the TODO items below.
#
# Run: python 10_rag_prompt_patterns.py

from __future__ import annotations

import anthropic


# === Exercise 1: Basic RAG Prompt with Citations ===
# Build a prompt that instructs the model to answer ONLY from the provided
# context chunks and cite each claim with [Source N].

CONTEXT_CHUNKS = [
    {"id": 1, "text": "Python was created by Guido van Rossum and first released in 1991."},
    {"id": 2, "text": "Python 3.0 was released on December 3, 2008, and was a major revision."},
    {"id": 3, "text": "The Python Package Index (PyPI) hosts over 400,000 packages."},
]

def build_rag_prompt(query: str, chunks: list[dict]) -> str:
    """Return a prompt that grounds the answer in the given chunks.

    Hint: Format each chunk as 'Source {id}: {text}' and instruct the model
    to cite sources using [Source N] notation.
    """
    # TODO: Build the context section from chunks
    # TODO: Write instructions requiring citation and no external knowledge
    # TODO: Append the user query
    pass


def exercise_1():
    """Verify the RAG prompt contains required structural elements."""
    prompt = build_rag_prompt("When was Python created?", CONTEXT_CHUNKS)
    assert prompt is not None, "build_rag_prompt must return a string"
    assert "Source 1" in prompt, "Prompt must include source references"
    assert "When was Python created?" in prompt, "Prompt must include the query"
    print("  RAG prompt built successfully")
    print(f"  Prompt length: {len(prompt)} chars")


# === Exercise 2: No-Answer Handling ===
# When context does not contain the answer, the model should refuse
# gracefully instead of hallucinating.

def build_rag_prompt_with_refusal(query: str, chunks: list[dict]) -> str:
    """Build a RAG prompt that explicitly handles unanswerable questions.

    Hint: Add an instruction like 'If the provided context does not contain
    enough information, respond with: "I cannot answer this based on the
    available sources."'
    """
    # TODO: Reuse or extend build_rag_prompt
    # TODO: Add a clear refusal instruction with a specific refusal phrase
    pass


def exercise_2():
    """Test that the refusal-aware prompt contains the right instructions."""
    prompt = build_rag_prompt_with_refusal("What is quantum computing?", CONTEXT_CHUNKS)
    assert prompt is not None, "Must return a string"
    assert "cannot answer" in prompt.lower() or "not contain" in prompt.lower(), (
        "Prompt must include a refusal instruction"
    )
    print("  Refusal-aware RAG prompt built successfully")


# === Exercise 3: Multi-Document RAG with Relevance Filter ===
# Pre-filter chunks by a simple keyword relevance score before
# injecting them into the prompt.

def relevance_score(query: str, chunk_text: str) -> float:
    """Return a 0-1 relevance score based on keyword overlap.

    Hint: Tokenize both strings to lowercase word sets, compute
    |intersection| / |query_tokens|.
    """
    # TODO: Tokenize query and chunk_text into lowercase word sets
    # TODO: Compute and return the overlap ratio
    pass


def build_filtered_rag_prompt(
    query: str, chunks: list[dict], threshold: float = 0.2,
) -> str:
    """Build a RAG prompt using only chunks above the relevance threshold.

    Hint: Use relevance_score() to filter, then pass the filtered list
    to build_rag_prompt().
    """
    # TODO: Score each chunk and keep those >= threshold
    # TODO: Build prompt from filtered chunks (fall back to all if none pass)
    pass


def exercise_3():
    """Verify relevance filtering works."""
    score = relevance_score("Python release date", CONTEXT_CHUNKS[0]["text"])
    assert isinstance(score, float), "Score must be a float"
    assert 0.0 <= score <= 1.0, "Score must be between 0 and 1"

    prompt = build_filtered_rag_prompt("Python release date", CONTEXT_CHUNKS)
    assert prompt is not None, "Must return a prompt string"
    print(f"  Relevance score for chunk 1: {score:.2f}")
    print("  Filtered RAG prompt built successfully")


# === Exercise 4: RAG with Structured JSON Output ===
# Ask the model to return a JSON object with 'answer' and 'sources' keys.

def build_json_rag_prompt(query: str, chunks: list[dict]) -> str:
    """Build a RAG prompt that requests structured JSON output.

    Expected output format:
    {"answer": "...", "sources": [1, 3], "confidence": "high|medium|low"}

    Hint: Include the exact JSON schema in the prompt and tell the model
    to return ONLY valid JSON with no extra text.
    """
    # TODO: Build context section
    # TODO: Specify the JSON output schema in the instructions
    # TODO: Include the query
    pass


def exercise_4():
    """Verify the JSON RAG prompt requests structured output."""
    prompt = build_json_rag_prompt("When was Python 3 released?", CONTEXT_CHUNKS)
    assert prompt is not None, "Must return a string"
    assert "json" in prompt.lower() or "JSON" in prompt, "Must mention JSON format"
    assert "answer" in prompt and "sources" in prompt, "Must specify output keys"
    print("  JSON RAG prompt built successfully")


if __name__ == "__main__":
    print("=== Exercise 1: Basic RAG Prompt ===")
    exercise_1()

    print("=== Exercise 2: No-Answer Handling ===")
    exercise_2()

    print("=== Exercise 3: Relevance Filtering ===")
    exercise_3()

    print("=== Exercise 4: Structured JSON RAG ===")
    exercise_4()

    print("\nAll exercises completed!")
