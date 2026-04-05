"""
24_caching_batch — Prompt Caching and Batch API Example
Demonstrates cache_control breakpoints and Message Batches.

Requirements: pip install anthropic
Set ANTHROPIC_API_KEY environment variable.

Note: This is a reference example showing the API patterns.
"""

import anthropic


def cached_conversation_example():
    """
    Prompt caching: mark large, reusable content with cache_control
    so subsequent requests reuse the cached prefix.

    Cost savings: cached input tokens are 90% cheaper.
    Cache TTL: 5 minutes (extended on each hit).
    """
    client = anthropic.Anthropic()

    # Large system prompt that benefits from caching
    system_prompt = """You are a senior Python developer reviewing code.
    Follow PEP 8 conventions. Focus on:
    - Type safety and proper type hints
    - Error handling patterns
    - Performance implications
    - Security considerations
    ... (imagine 2000+ tokens of detailed instructions) ..."""

    # First request — creates the cache
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        system=[
            {
                "type": "text",
                "text": system_prompt,
                "cache_control": {"type": "ephemeral"},  # Mark for caching
            }
        ],
        messages=[{"role": "user", "content": "Review this function:\ndef add(a, b): return a + b"}],
    )

    # Check cache usage in response
    print(f"Input tokens: {response.usage.input_tokens}")
    if hasattr(response.usage, "cache_creation_input_tokens"):
        print(f"Cache creation tokens: {response.usage.cache_creation_input_tokens}")
    if hasattr(response.usage, "cache_read_input_tokens"):
        print(f"Cache read tokens: {response.usage.cache_read_input_tokens}")

    return response


def batch_processing_example():
    """
    Message Batches API: submit up to 100,000 requests at once.
    50% cost discount. Results within 24 hours (usually much faster).

    Use cases:
    - Evaluating model outputs on a test set
    - Processing a large document corpus
    - Generating training data
    """
    client = anthropic.Anthropic()

    # Prepare batch requests
    requests = [
        {
            "custom_id": f"review-{i}",
            "params": {
                "model": "claude-sonnet-4-20250514",
                "max_tokens": 512,
                "messages": [
                    {"role": "user", "content": f"Review this code snippet #{i}: def func_{i}(): pass"}
                ],
            },
        }
        for i in range(5)
    ]

    # Submit batch
    batch = client.messages.batches.create(requests=requests)
    print(f"Batch ID: {batch.id}")
    print(f"Status: {batch.processing_status}")

    # Poll for completion
    # result = client.messages.batches.retrieve(batch.id)
    # When done, iterate results:
    # for result in client.messages.batches.results(batch.id):
    #     print(f"{result.custom_id}: {result.result.message.content[0].text[:50]}")

    return batch


def combined_caching_and_batch():
    """
    Combine caching + batching for maximum savings:
    - Batch discount: 50% off
    - Cache discount: 90% off cached input tokens
    - Combined: up to 95% savings on cached input
    """
    system = [
        {
            "type": "text",
            "text": "You are a code reviewer. Provide brief, actionable feedback.",
            "cache_control": {"type": "ephemeral"},
        }
    ]

    # All batch items share the same cached system prompt
    requests = [
        {
            "custom_id": f"file-{name}",
            "params": {
                "model": "claude-sonnet-4-20250514",
                "max_tokens": 256,
                "system": system,  # Same cached system for all
                "messages": [
                    {"role": "user", "content": f"Review: # {name}\ndef main(): pass"}
                ],
            },
        }
        for name in ["auth.py", "api.py", "models.py", "utils.py", "config.py"]
    ]

    print(f"Prepared {len(requests)} batch requests with shared cached system prompt")
    return requests


# === Reference Output ===

if __name__ == "__main__":
    print("Prompt Caching and Batch API Examples")
    print("=" * 45)
    print()
    print("1. Prompt Caching")
    print("   - Mark reusable content with cache_control: {type: 'ephemeral'}")
    print("   - System prompts, long documents, few-shot examples")
    print("   - 90% cost reduction on cached input tokens")
    print("   - 5-minute TTL, extended on each cache hit")
    print()
    print("2. Message Batches API")
    print("   - Submit up to 100K requests per batch")
    print("   - 50% cost discount")
    print("   - Results within 24 hours")
    print("   - Ideal for evaluation, data processing, bulk generation")
    print()
    print("3. Combined (Caching + Batching)")
    print("   - Share cached system prompt across all batch items")
    print("   - Up to 95% savings on cached input tokens")
