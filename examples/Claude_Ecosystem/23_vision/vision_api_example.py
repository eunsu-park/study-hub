"""
23_vision — Vision Agents Example
Demonstrates sending images to Claude and extracting structured data.

Requirements: pip install anthropic
Set ANTHROPIC_API_KEY environment variable.

Note: This is a reference example showing the API patterns.
Run only if you have an API key configured.
"""

import anthropic
import base64
import json
from pathlib import Path


def image_to_base64(image_path: str) -> str:
    """Read an image file and return base64-encoded string."""
    data = Path(image_path).read_bytes()
    return base64.standard_b64encode(data).decode("utf-8")


def analyze_image_from_base64(client: anthropic.Anthropic, image_b64: str, media_type: str, prompt: str) -> str:
    """Send a base64-encoded image to Claude for analysis."""
    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": image_b64,
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ],
    )
    return message.content[0].text


def compare_images(client: anthropic.Anthropic, images: list[dict], prompt: str) -> str:
    """Send multiple images for comparison."""
    content = []
    for img in images:
        content.append({
            "type": "image",
            "source": {"type": "base64", "media_type": img["media_type"], "data": img["data"]},
        })
    content.append({"type": "text", "text": prompt})

    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        messages=[{"role": "user", "content": content}],
    )
    return message.content[0].text


def extract_structured_data(client: anthropic.Anthropic, image_b64: str, media_type: str) -> dict:
    """Extract structured data from a document image (receipt, form, etc.)."""
    prompt = """Extract all text and data from this image.
Return a JSON object with the following structure:
{
    "type": "receipt|form|document|screenshot|other",
    "extracted_text": "full text content",
    "key_fields": {"field_name": "value", ...},
    "confidence": "high|medium|low"
}
Return ONLY valid JSON, no markdown."""

    text = analyze_image_from_base64(client, image_b64, media_type, prompt)
    return json.loads(text)


# === Usage Examples (reference only — requires API key) ===

if __name__ == "__main__":
    print("Vision Agent Examples")
    print("=" * 40)
    print()
    print("Example 1: Analyze a single image")
    print('  result = analyze_image_from_base64(client, img_b64, "image/png", "Describe this image")')
    print()
    print("Example 2: Compare two images")
    print('  result = compare_images(client, [img1, img2], "What are the differences?")')
    print()
    print("Example 3: Extract structured data")
    print('  data = extract_structured_data(client, receipt_b64, "image/jpeg")')
    print()
    print("Supported formats: JPEG, PNG, GIF, WebP")
    print("Max image size: ~5MB per image")
    print("Multiple images: up to 20 per request")
