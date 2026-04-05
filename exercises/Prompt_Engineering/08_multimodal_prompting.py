# Exercise: Lesson 08 — Multimodal Prompting
# Complete the TODO items below.
#
# Run: python 08_multimodal_prompting.py

import anthropic
import base64
import json
from pathlib import Path

client = anthropic.Anthropic()  # expects ANTHROPIC_API_KEY env var

MODEL = "claude-sonnet-4-20250514"


# === Exercise 1: Encode an Image for the API ===
# Prepare an image file for the Anthropic vision API.
# Hint: The API accepts base64-encoded images with a media type.

SUPPORTED_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
}

def encode_image(image_path: str) -> dict:
    """Read an image file and return the content block for the API.
    Return: {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": "<mime type>",
            "data": "<base64 string>"
        }
    }
    """
    # TODO: Read the file in binary mode
    # TODO: Determine media_type from file extension using SUPPORTED_TYPES
    # TODO: Base64-encode the binary data
    # TODO: Return the structured dict
    # Hint: Use base64.standard_b64encode(data).decode("utf-8")
    pass


def exercise_1():
    # Create a tiny test PNG (1x1 red pixel) for testing
    test_path = "/tmp/test_pixel.png"
    # Minimal valid PNG: 1x1 red pixel
    import struct, zlib
    def make_png():
        sig = b"\x89PNG\r\n\x1a\n"
        ihdr_data = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
        ihdr_crc = zlib.crc32(b"IHDR" + ihdr_data) & 0xFFFFFFFF
        ihdr = struct.pack(">I", 13) + b"IHDR" + ihdr_data + struct.pack(">I", ihdr_crc)
        raw = b"\x00\xff\x00\x00"  # filter byte + RGB
        idat_data = zlib.compress(raw)
        idat_crc = zlib.crc32(b"IDAT" + idat_data) & 0xFFFFFFFF
        idat = struct.pack(">I", len(idat_data)) + b"IDAT" + idat_data + struct.pack(">I", idat_crc)
        iend_crc = zlib.crc32(b"IEND") & 0xFFFFFFFF
        iend = struct.pack(">I", 0) + b"IEND" + struct.pack(">I", iend_crc)
        return sig + ihdr + idat + iend
    Path(test_path).write_bytes(make_png())

    result = encode_image(test_path)
    assert result["type"] == "image"
    assert result["source"]["media_type"] == "image/png"
    assert len(result["source"]["data"]) > 0
    print(f"[Ex1] Encoded image: media_type={result['source']['media_type']}, "
          f"data_length={len(result['source']['data'])} chars")


# === Exercise 2: Single Image Analysis ===
# Send an image with a text prompt for visual analysis.
# Hint: The messages content is a list containing image and text blocks.

def analyze_image(image_path: str, question: str) -> str:
    """Send an image to Claude with a question and return the response."""
    # TODO: Use encode_image() to get the image content block
    # TODO: Build the messages list with both image and text content blocks:
    #   [{"type": "image", "source": {...}}, {"type": "text", "text": question}]
    # TODO: Call the API and return the response text
    pass


def exercise_2():
    # Use the test PNG from exercise 1
    test_path = "/tmp/test_pixel.png"
    if not Path(test_path).exists():
        print("[Ex2] SKIP: test image not found (run exercise 1 first)")
        return
    result = analyze_image(test_path, "Describe this image in one sentence.")
    assert isinstance(result, str) and len(result) > 5
    print(f"[Ex2] Analysis: {result[:100]}")


# === Exercise 3: Structured Vision Extraction ===
# Extract structured data from an image (e.g., a receipt, chart, or diagram).
# Hint: Combine vision with JSON output instructions.

def extract_from_image(image_path: str, schema_description: str) -> dict:
    """Analyze an image and extract structured data matching the schema.
    Args:
        image_path: path to the image file
        schema_description: description of the JSON fields to extract
    Returns: parsed dict of extracted data
    """
    # TODO: Build a prompt that asks Claude to analyze the image and
    #       return ONLY valid JSON matching the schema description
    # TODO: Use encode_image() for the image block
    # TODO: Parse the JSON response
    # Hint: Handle potential markdown code fences in the response
    pass


def exercise_3():
    test_path = "/tmp/test_pixel.png"
    if not Path(test_path).exists():
        print("[Ex3] SKIP: test image not found")
        return
    result = extract_from_image(
        test_path,
        "width_pixels (int), height_pixels (int), dominant_color (str), "
        "description (str)",
    )
    if result:
        print(f"[Ex3] Extracted: {json.dumps(result, indent=2)}")
    else:
        print("[Ex3] Extraction returned None")


# === Exercise 4: Image Comparison Prompt ===
# Send multiple images and ask Claude to compare them.
# Hint: Include multiple image blocks in the content list.

def compare_images(image_paths: list[str], comparison_criteria: str) -> str:
    """Send multiple images for comparison analysis.
    Args:
        image_paths: list of image file paths
        comparison_criteria: what aspects to compare
    Returns: the comparison analysis text
    """
    # TODO: Build content list with alternating image blocks and labels
    #   e.g., [image1_block, {"type": "text", "text": "Image 1"},
    #          image2_block, {"type": "text", "text": "Image 2"}, ...]
    # TODO: Append a final text block with the comparison question
    # TODO: Call the API and return the response
    pass


def exercise_4():
    test_path = "/tmp/test_pixel.png"
    if not Path(test_path).exists():
        print("[Ex4] SKIP: test image not found")
        return
    # Compare the same image with itself (for testing purposes)
    result = compare_images(
        [test_path, test_path],
        "Compare the colors, sizes, and content of these two images.",
    )
    assert isinstance(result, str) and len(result) > 10
    print(f"[Ex4] Comparison: {result[:120]}")


# === Exercise 5: Vision Prompt Builder ===
# Build a reusable prompt template for common vision tasks.
# This is a pure Python exercise (no API call needed).

VISION_TASKS = {
    "describe": "Describe this image in detail.",
    "ocr": "Extract all visible text from this image verbatim.",
    "classify": "Classify this image into one of these categories: {categories}.",
    "count": "Count the number of {object} visible in this image.",
    "caption": "Write a concise caption for this image (max 15 words).",
}

def build_vision_prompt(task: str, **kwargs) -> str:
    """Build a vision analysis prompt from a task template.
    Args:
        task: one of the VISION_TASKS keys
        **kwargs: template variables (e.g., categories, object)
    Returns: the formatted prompt string
    """
    # TODO: Look up the task template in VISION_TASKS
    # TODO: Format it with any provided kwargs
    # TODO: Raise ValueError for unknown tasks
    pass


def exercise_5():
    p1 = build_vision_prompt("describe")
    p2 = build_vision_prompt("classify", categories="cat, dog, bird, other")
    p3 = build_vision_prompt("count", object="people")
    p4 = build_vision_prompt("caption")
    assert "detail" in p1
    assert "cat" in p2
    assert "people" in p3
    assert "15 words" in p4
    print(f"[Ex5] describe:  {p1}")
    print(f"[Ex5] classify:  {p2}")
    print(f"[Ex5] count:     {p3}")
    print(f"[Ex5] caption:   {p4}")
    # Test unknown task
    try:
        build_vision_prompt("unknown_task")
        assert False, "Should have raised ValueError"
    except ValueError:
        print("[Ex5] Unknown task raises ValueError -- PASS")


if __name__ == "__main__":
    print("=== Exercise 1: Encode Image ===")
    exercise_1()

    print("\n=== Exercise 2: Single Image Analysis ===")
    exercise_2()

    print("\n=== Exercise 3: Structured Vision Extraction ===")
    exercise_3()

    print("\n=== Exercise 4: Image Comparison ===")
    exercise_4()

    print("\n=== Exercise 5: Vision Prompt Builder (no API) ===")
    exercise_5()

    print("\nAll exercises completed!")
