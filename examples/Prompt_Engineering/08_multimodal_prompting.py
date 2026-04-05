# 08_multimodal_prompting.py — Vision+text prompting and image analysis
#
# Run: python 08_multimodal_prompting.py

"""
Demonstrates:
  1. Image analysis from a URL       — describe and extract info from images
  2. Image + text combined prompting  — use vision with textual instructions
  3. Structured extraction from images — parse visual data into JSON
  4. Comparing multiple images         — side-by-side analysis

NOTE: Requires the Anthropic API with vision support.
      Uses publicly available sample images for demonstration.
"""

import base64
import json
import os
from pathlib import Path

import anthropic

client: anthropic.Anthropic
MODEL = "claude-sonnet-4-20250514"


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def call_claude_vision(
    text_prompt: str,
    image_sources: list[dict],
    system: str = "",
) -> str:
    """Send a vision request with one or more images + text."""
    content = []
    for img in image_sources:
        content.append({
            "type": "image",
            "source": img,
        })
    content.append({"type": "text", "text": text_prompt})

    message = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        temperature=0.0,
        system=system,
        messages=[{"role": "user", "content": content}],
    )
    return message.content[0].text.strip()


def url_image(url: str) -> dict:
    """Create an image source dict from a URL."""
    return {"type": "url", "url": url}


def base64_image(file_path: str) -> dict:
    """Create an image source dict from a local file (base64-encoded)."""
    path = Path(file_path)
    suffix = path.suffix.lower().lstrip(".")
    media_map = {"jpg": "image/jpeg", "jpeg": "image/jpeg",
                 "png": "image/png", "gif": "image/gif", "webp": "image/webp"}
    media_type = media_map.get(suffix, "image/png")
    data = base64.standard_b64encode(path.read_bytes()).decode()
    return {"type": "base64", "media_type": media_type, "data": data}


# ---------------------------------------------------------------------------
# 1. Basic Image Description
# ---------------------------------------------------------------------------

# Public-domain sample images for demonstration
SAMPLE_CHART_URL = (
    "https://upload.wikimedia.org/wikipedia/commons/thumb/"
    "1/1e/Matplotlib_example_pie_chart.png/"
    "320px-Matplotlib_example_pie_chart.png"
)


def demo_basic_description():
    """Ask Claude to describe an image from a URL."""

    prompt = (
        "Describe this image in detail. What type of chart is it? "
        "What data does it represent? List the categories and their "
        "approximate percentages."
    )

    print("=" * 60)
    print("SECTION 1 — Basic Image Description")
    print("=" * 60)
    result = call_claude_vision(prompt, [url_image(SAMPLE_CHART_URL)])
    print(f"\n{result}")


# ---------------------------------------------------------------------------
# 2. Image + Text Combined Prompt
# ---------------------------------------------------------------------------

def demo_image_plus_text():
    """Combine an image with detailed textual instructions."""

    system = "You are a data analyst. Be precise with numbers."
    prompt = (
        "Analyze this chart and provide:\n"
        "1. The chart type and title (if visible)\n"
        "2. The dominant category\n"
        "3. A one-sentence business insight\n"
        "4. Any data quality issues you notice (missing labels, etc.)"
    )

    print("\n" + "=" * 60)
    print("SECTION 2 — Image + Text Combined Prompt")
    print("=" * 60)
    result = call_claude_vision(
        prompt, [url_image(SAMPLE_CHART_URL)], system=system
    )
    print(f"\n{result}")


# ---------------------------------------------------------------------------
# 3. Structured Extraction from Image
# ---------------------------------------------------------------------------

def demo_structured_extraction():
    """Extract structured JSON data from an image."""

    prompt = (
        "Extract data from this pie chart into JSON format:\n"
        "{\n"
        '  "chart_type": "...",\n'
        '  "title": "...",\n'
        '  "segments": [\n'
        '    {"label": "...", "percentage": ..., "color": "..."}\n'
        "  ]\n"
        "}\n\n"
        "Return ONLY valid JSON. Estimate percentages if exact values "
        "are not labeled."
    )

    print("\n" + "=" * 60)
    print("SECTION 3 — Structured Extraction from Image")
    print("=" * 60)
    result = call_claude_vision(prompt, [url_image(SAMPLE_CHART_URL)])
    print(f"\n{result}")

    # Attempt to parse the JSON
    try:
        import re
        match = re.search(r"\{.*\}", result, re.DOTALL)
        if match:
            data = json.loads(match.group())
            print(f"\nParsed {len(data.get('segments', []))} segments successfully.")
    except json.JSONDecodeError:
        print("\nNote: Could not parse response as JSON.")


# ---------------------------------------------------------------------------
# 4. Multi-Image Comparison (simulated with same image)
# ---------------------------------------------------------------------------

SAMPLE_MAP_URL = (
    "https://upload.wikimedia.org/wikipedia/commons/thumb/"
    "a/a5/Tsunami_travel_time_Tohoku_2011.jpg/"
    "320px-Tsunami_travel_time_Tohoku_2011.jpg"
)


def demo_multi_image():
    """Send multiple images and ask for comparison analysis."""

    prompt = (
        "I'm showing you two different data visualizations.\n\n"
        "Image 1 is a pie chart. Image 2 is a map visualization.\n\n"
        "Compare these two visualization types:\n"
        "1. What type of data is each best suited for?\n"
        "2. What are the strengths and weaknesses of each?\n"
        "3. When would you choose one over the other?"
    )

    print("\n" + "=" * 60)
    print("SECTION 4 — Multi-Image Comparison")
    print("=" * 60)
    result = call_claude_vision(
        prompt,
        [url_image(SAMPLE_CHART_URL), url_image(SAMPLE_MAP_URL)],
    )
    print(f"\n{result}")


# ---------------------------------------------------------------------------
# 5. Local Image (base64) — Template
# ---------------------------------------------------------------------------

def demo_local_image_template():
    """Show how to load a local image file (template, not executed)."""

    print("\n" + "=" * 60)
    print("SECTION 5 — Local Image Template (not executed)")
    print("=" * 60)

    code = '''\
    # To analyze a local image file:
    img_src = base64_image("path/to/screenshot.png")
    result = call_claude_vision(
        "Describe what you see in this screenshot.",
        [img_src],
    )
    print(result)
    '''
    print(code)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: Set the ANTHROPIC_API_KEY environment variable first.")
        raise SystemExit(1)

    client = anthropic.Anthropic()

    try:
        demo_basic_description()
        demo_image_plus_text()
        demo_structured_extraction()
        demo_multi_image()
        demo_local_image_template()
    except anthropic.APIError as exc:
        print(f"\nAPI error: {exc}")
        print("Ensure your API key supports vision and has sufficient quota.")
