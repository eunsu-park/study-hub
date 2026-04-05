# Exercise: Lesson 14 — Domain-Specific Prompting
# Complete the TODO items below.
#
# Run: python 14_domain_specific_prompting.py

from __future__ import annotations

import json
import re


# === Exercise 1: Structured Data Extraction ===
# Build a prompt that extracts structured fields from unstructured text.

SAMPLE_INVOICE = """
Invoice #INV-2024-0892
Date: March 15, 2024
Bill To: Acme Corp, 123 Main St, Springfield, IL 62701

Items:
  - Widget A x10 @ $25.00 each = $250.00
  - Widget B x5 @ $40.00 each = $200.00
  - Shipping = $15.00

Subtotal: $465.00
Tax (8%): $37.20
Total: $502.20

Payment Terms: Net 30
"""


def build_extraction_prompt(document: str, fields: list[str]) -> str:
    """Build a prompt to extract specified fields from a document.

    The prompt should instruct the model to return a JSON object with
    the requested fields. Missing fields should be null.

    Args:
        document: The raw text to extract from.
        fields: List of field names to extract (e.g., ["invoice_number",
                "date", "total", "line_items"]).

    Hint: Include the document in a clearly delimited section (e.g.,
    <document>...</document>). Specify the exact field names and
    expected types in the instruction.
    """
    # TODO: Build instruction section specifying fields and types
    # TODO: Include the document in a delimited block
    # TODO: Request JSON-only output
    pass


def exercise_1():
    """Verify the extraction prompt is well-formed."""
    fields = ["invoice_number", "date", "total", "line_items", "payment_terms"]
    prompt = build_extraction_prompt(SAMPLE_INVOICE, fields)
    assert prompt is not None, "Must return a string"
    assert "INV-2024-0892" in prompt, "Must include the document text"
    assert "invoice_number" in prompt, "Must reference requested fields"
    assert "json" in prompt.lower() or "JSON" in prompt, "Must request JSON output"
    print(f"  Extraction prompt: {len(prompt)} chars")
    print(f"  Fields requested: {fields}")


# === Exercise 2: Domain-Specific Summarization ===
# Build summarization prompts tailored to different professional domains.

DOMAIN_CONFIGS = {
    "legal": {
        "focus": ["key obligations", "deadlines", "parties involved", "penalties"],
        "tone": "formal and precise",
        "length": "2-3 paragraphs",
    },
    "medical": {
        "focus": ["diagnosis", "treatment plan", "medications", "follow-up"],
        "tone": "clear and accessible to patients",
        "length": "bullet points",
    },
    "technical": {
        "focus": ["problem statement", "root cause", "solution", "impact"],
        "tone": "concise and technical",
        "length": "1 paragraph with bullet details",
    },
}


def build_domain_summary_prompt(text: str, domain: str) -> str:
    """Build a summarization prompt using domain-specific configuration.

    Args:
        text: The text to summarize.
        domain: One of "legal", "medical", "technical".

    Hint: Look up the config from DOMAIN_CONFIGS. Use the focus areas,
    tone, and length to construct precise summarization instructions.
    Raise ValueError for unknown domains.
    """
    # TODO: Look up domain config (raise ValueError if not found)
    # TODO: Build instructions emphasizing domain focus areas
    # TODO: Specify tone and length requirements
    # TODO: Include the text to summarize
    pass


def exercise_2():
    """Verify domain-specific prompts reflect their configuration."""
    sample = "The patient presented with persistent headaches for 2 weeks..."

    for domain in ["legal", "medical", "technical"]:
        prompt = build_domain_summary_prompt(sample, domain)
        assert prompt is not None, f"Must return a string for {domain}"
        config = DOMAIN_CONFIGS[domain]
        lower = prompt.lower()
        assert config["focus"][0].lower() in lower, (
            f"Must include focus area for {domain}"
        )
        print(f"  [{domain}] prompt length: {len(prompt)} chars")

    try:
        build_domain_summary_prompt(sample, "unknown")
        assert False, "Should raise ValueError for unknown domain"
    except ValueError:
        print("  [unknown] correctly raised ValueError")


# === Exercise 3: Classification Prompt with Taxonomy ===
# Build a prompt that classifies text into a predefined taxonomy.

SUPPORT_TAXONOMY = {
    "billing": ["refund", "charge", "invoice", "payment"],
    "technical": ["bug", "error", "crash", "not working"],
    "account": ["password", "login", "access", "profile"],
    "feature_request": ["wish", "would be nice", "suggestion", "please add"],
}


def build_classification_prompt(
    text: str, taxonomy: dict[str, list[str]]
) -> str:
    """Build a prompt to classify text into one of the taxonomy categories.

    The prompt should:
      1. List all categories with their example keywords
      2. Instruct the model to return JSON: {"category": "...", "confidence": 0.0-1.0}
      3. Handle ambiguous cases by choosing the best match

    Hint: Format each category as 'Category: keyword1, keyword2, ...'
    """
    # TODO: Format the taxonomy into the prompt
    # TODO: Include the text to classify
    # TODO: Specify the JSON output format
    pass


def rule_based_classify(text: str, taxonomy: dict[str, list[str]]) -> str:
    """Classify text using simple keyword matching as a baseline.

    Hint: For each category, count how many of its keywords appear in
    the lowercased text. Return the category with the highest count.
    Return 'unknown' if no keywords match.
    """
    # TODO: Count keyword matches per category
    # TODO: Return the category with the most matches, or 'unknown'
    pass


def exercise_3():
    """Verify classification logic works."""
    test_cases = [
        ("I can't login to my account", "account"),
        ("I was charged twice for my order", "billing"),
        ("The app crashes when I open settings", "technical"),
        ("It would be nice to have dark mode", "feature_request"),
    ]
    for text, expected in test_cases:
        result = rule_based_classify(text, SUPPORT_TAXONOMY)
        assert result == expected, f"Expected {expected}, got {result}"
        print(f"  '{text[:40]}...' -> {result}")

    prompt = build_classification_prompt(test_cases[0][0], SUPPORT_TAXONOMY)
    assert prompt is not None
    print(f"  Classification prompt: {len(prompt)} chars")


# === Exercise 4: Multi-Step Data Pipeline Prompt ===
# Chain extraction, classification, and summarization into one prompt.

def build_pipeline_prompt(document: str) -> str:
    """Build a prompt that performs a 3-step analysis pipeline.

    Steps:
      1. Extract: Pull out key entities (people, dates, amounts)
      2. Classify: Categorize the document type (invoice, contract, report)
      3. Summarize: Provide a 2-sentence executive summary

    Output format (JSON):
    {
        "entities": {"people": [...], "dates": [...], "amounts": [...]},
        "document_type": "...",
        "summary": "..."
    }

    Hint: Number the steps clearly and specify the exact JSON structure.
    """
    # TODO: Write the 3-step pipeline instructions
    # TODO: Specify the JSON output schema
    # TODO: Include the document
    pass


def exercise_4():
    """Verify the pipeline prompt includes all three steps."""
    prompt = build_pipeline_prompt(SAMPLE_INVOICE)
    assert prompt is not None, "Must return a string"
    lower = prompt.lower()
    assert "extract" in lower, "Must include extraction step"
    assert "classify" in lower or "categorize" in lower, "Must include classification"
    assert "summar" in lower, "Must include summarization step"
    assert "entities" in prompt, "Must specify entities in output schema"
    print(f"  Pipeline prompt: {len(prompt)} chars")
    print("  All three pipeline steps present")


if __name__ == "__main__":
    print("=== Exercise 1: Data Extraction ===")
    exercise_1()

    print("=== Exercise 2: Domain Summarization ===")
    exercise_2()

    print("=== Exercise 3: Classification ===")
    exercise_3()

    print("=== Exercise 4: Multi-Step Pipeline ===")
    exercise_4()

    print("\nAll exercises completed!")
