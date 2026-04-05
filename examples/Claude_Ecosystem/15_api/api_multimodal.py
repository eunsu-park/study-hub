"""
Claude API: Multimodal Input Examples

Demonstrates sending images and documents to Claude
using the Messages API.

Requirements:
    pip install anthropic
    export ANTHROPIC_API_KEY=sk-ant-...
"""

from __future__ import annotations

import base64
from pathlib import Path


# --- Example 1: Image from URL ---

def image_from_url():
    """Send an image via URL for analysis."""
    import anthropic

    client = anthropic.Anthropic()

    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "url",
                        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a7/Camponotus_flavomarginatus_ant.jpg/320px-Camponotus_flavomarginatus_ant.jpg",
                    },
                },
                {
                    "type": "text",
                    "text": "What insect is shown in this image?",
                },
            ],
        }],
    )
    print(f"Response: {message.content[0].text}")


# --- Example 2: Image from Base64 ---

def image_from_base64(image_path: str):
    """Send a local image encoded as base64."""
    import anthropic

    client = anthropic.Anthropic()

    image_data = Path(image_path).read_bytes()
    base64_image = base64.standard_b64encode(image_data).decode("utf-8")

    # Detect media type from extension
    ext = Path(image_path).suffix.lower()
    media_types = {".png": "image/png", ".jpg": "image/jpeg",
                   ".jpeg": "image/jpeg", ".gif": "image/gif",
                   ".webp": "image/webp"}
    media_type = media_types.get(ext, "image/png")

    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": base64_image,
                    },
                },
                {
                    "type": "text",
                    "text": "Describe this image in detail.",
                },
            ],
        }],
    )
    print(f"Response: {message.content[0].text}")


# --- Example 3: PDF Document ---

def pdf_document(pdf_path: str):
    """Send a PDF document for analysis."""
    import anthropic

    client = anthropic.Anthropic()

    pdf_data = Path(pdf_path).read_bytes()
    base64_pdf = base64.standard_b64encode(pdf_data).decode("utf-8")

    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=2048,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "document",
                    "source": {
                        "type": "base64",
                        "media_type": "application/pdf",
                        "data": base64_pdf,
                    },
                },
                {
                    "type": "text",
                    "text": "Summarize this document in 3 bullet points.",
                },
            ],
        }],
    )
    print(f"Response: {message.content[0].text}")


# --- Main ---

if __name__ == "__main__":
    print("Claude API Multimodal Examples")
    print("=" * 40)
    print("\nNote: These examples require ANTHROPIC_API_KEY.")
    print("Set the environment variable before running.\n")

    # Uncomment to run (requires API key):
    # image_from_url()
    # image_from_base64("photo.png")
    # pdf_document("report.pdf")

    print("Examples defined. Uncomment the calls in __main__ to run.")
