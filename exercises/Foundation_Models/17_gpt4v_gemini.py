"""
Exercises for Lesson 17: GPT-4V, Gemini, and Claude
Topic: Foundation_Models

Solutions to practice problems from the lesson.
"""


# === Exercise 1: Model Selection for Production Use Cases ===
# Problem: Select the most appropriate model for 5 production scenarios.

def exercise_1():
    """Solution: Model selection for production"""
    cases = [
        {
            "use_case": "A) Transcribe 10K handwritten medical records/day (cost-sensitive)",
            "model": "gpt-4o-mini or claude-3-haiku",
            "justification": (
                "High volume + cost sensitivity -> cheapest capable model. "
                "At $0.15/1M tokens, 10K pages at ~1K tokens/page = $1.50/day."
            ),
        },
        {
            "use_case": "B) Analyze 2-hour security footage for anomalies",
            "model": "gemini-1.5-pro",
            "justification": (
                "ONLY option with native video support and 2M token context. "
                "2 hours x 263 tokens/sec ~= 1.9M tokens fits in context."
            ),
        },
        {
            "use_case": "C) Real-time voice assistant for customer support",
            "model": "gpt-4o-audio-preview",
            "justification": (
                "ONLY option with native real-time audio I/O and 320ms "
                "average response time. Handles voice without separate STT."
            ),
        },
        {
            "use_case": "D) Legal contract review with complex reasoning",
            "model": "claude-sonnet-4 or claude-3-opus",
            "justification": (
                "Claude ranks highest in reasoning. Constitutional AI "
                "training makes it better calibrated for high-stakes decisions. "
                "200K context handles long contracts."
            ),
        },
        {
            "use_case": "E) Generate product descriptions (100K images/day)",
            "model": "gemini-1.5-flash or gpt-4o-mini",
            "justification": (
                "Highest volume -> lowest cost model. "
                "Gemini Flash at $0.075/1M tokens is cheapest. "
                "Simple descriptive task doesn't need maximum capability."
            ),
        },
    ]

    for c in cases:
        print(f"  {c['use_case']}")
        print(f"    Selected: {c['model']}")
        print(f"    Justification: {c['justification']}")
        print()


# === Exercise 2: GPT-4V Image Token Cost Calculation ===
# Problem: Calculate API cost for 500 product images with detail="high".

def exercise_2():
    """Solution: GPT-4V image token cost calculation"""
    num_images = 500
    prompt_tokens_per_image = 200
    response_tokens_per_image = 800
    gpt4o_input_price = 5.00 / 1_000_000   # per token
    gpt4o_output_price = 15.00 / 1_000_000  # per token

    # Image token calculation for high detail, 1024x1024
    base_tokens = 765
    tiles = 4  # 1024/512 = 2 x 2 = 4 tiles
    tokens_per_tile = 170
    image_tokens_per_image = base_tokens + (tiles * tokens_per_tile)

    total_image_tokens = num_images * image_tokens_per_image
    total_prompt_tokens = num_images * prompt_tokens_per_image
    total_output_tokens = num_images * response_tokens_per_image
    total_input_tokens = total_image_tokens + total_prompt_tokens

    input_cost = total_input_tokens * gpt4o_input_price
    output_cost = total_output_tokens * gpt4o_output_price
    total_cost = input_cost + output_cost
    cost_per_image = total_cost / num_images

    print(f"  Image token calc (high detail, 1024x1024):")
    print(f"    Base: {base_tokens} + {tiles} tiles x {tokens_per_tile} = {image_tokens_per_image} tokens/image")
    print()
    print(f"  Totals for {num_images} images:")
    print(f"    Image tokens: {total_image_tokens:,}")
    print(f"    Text prompt tokens: {total_prompt_tokens:,}")
    print(f"    Total input tokens: {total_input_tokens:,}")
    print(f"    Output tokens: {total_output_tokens:,}")
    print()
    print(f"  Cost breakdown:")
    print(f"    Input cost: {total_input_tokens:,} * $5/1M = ${input_cost:.2f}")
    print(f"    Output cost: {total_output_tokens:,} * $15/1M = ${output_cost:.2f}")
    print(f"    Total: ${total_cost:.2f} for {num_images} images")
    print(f"    Per image: ${cost_per_image:.3f}")
    print()

    # Low detail comparison
    low_detail_tokens = 85
    low_input = num_images * low_detail_tokens + total_prompt_tokens
    low_input_cost = low_input * gpt4o_input_price
    low_total = low_input_cost + output_cost

    print(f"  Comparison with detail='low':")
    print(f"    Low detail: {low_detail_tokens} tokens/image")
    print(f"    Total: ${low_total:.2f} ({(1-low_total/total_cost)*100:.0f}% cheaper)")
    print(f"    Key insight: Output tokens dominate cost "
          f"(${output_cost:.2f} of ${total_cost:.2f} = {output_cost/total_cost*100:.0f}%)")


# === Exercise 3: Prompt Engineering for Structured Output ===
# Problem: Design a Claude prompt for structured product data extraction.

def exercise_3():
    """Solution: Structured output prompt design"""
    prompt = """Analyze this product image and extract structured data.

Return ONLY a valid JSON object with exactly this schema:
{
  "product_name": string or null,
  "product_name_confidence": number (0.0-1.0),
  "brand": string or null,
  "brand_confidence": number (0.0-1.0),
  "price": string or null,
  "price_confidence": number (0.0-1.0),
  "color": string or null,
  "color_confidence": number (0.0-1.0),
  "dimensions": string or null,
  "dimensions_confidence": number (0.0-1.0),
  "visible_defects": array of strings (empty array if none),
  "defects_confidence": number (0.0-1.0)
}

Rules:
1. Use null for fields you cannot determine -- NEVER fabricate or guess
2. Confidence scores: 1.0=clear, 0.7=partial, 0.4=inferred, null field=0.0
3. Return ONLY the JSON object, no other text
4. Dimensions should include units if visible (e.g., "30cm x 20cm x 10cm")"""

    print("  Prompt for structured product data extraction:")
    for line in prompt.split("\n"):
        print(f"    {line}")
    print()

    print("  Key design decisions:")
    print("    - Explicit schema prevents model from inventing field names")
    print("    - null for missing data prevents hallucination")
    print("    - Per-field confidence enables flagging for human review")
    print("    - Regex fallback handles model adding preamble text")

    # Example output
    import json
    example_output = {
        "product_name": "Wireless Bluetooth Headphones",
        "product_name_confidence": 0.9,
        "brand": "Sony",
        "brand_confidence": 1.0,
        "price": "$149.99",
        "price_confidence": 0.8,
        "color": "Black",
        "color_confidence": 1.0,
        "dimensions": None,
        "dimensions_confidence": 0.0,
        "visible_defects": [],
        "defects_confidence": 0.7,
    }
    print()
    print("  Example output:")
    print(f"    {json.dumps(example_output, indent=6)}")


# === Exercise 4: Gemini Long-Context Video Analysis ===
# Problem: Design system for 8-hour surveillance footage analysis.

def exercise_4():
    """Solution: Gemini long-context video analysis design"""
    print("  Challenge: 8 hrs x 3600s x ~263 tokens/s = ~7.57M tokens")
    print("  Exceeds Gemini 1.5 Pro's 2M context by 3.8x")
    print()

    print("  Solution: Sliding window with 90-min segments + 10-min overlap")
    print()
    print("  Architecture:")
    print("    1. Split video into 90-min segments (fits in 2M context)")
    print("    2. 10-min overlaps prevent missing boundary incidents")
    print("    3. Each segment analyzed for:")
    print("       - Shoplifting incidents (timestamp, location, confidence)")
    print("       - Customer flow (count per 15-min intervals)")
    print("       - Anomalies")
    print("    4. Results merged + deduplicated across segments")
    print("    5. Final text-only synthesis generates end-of-day report")
    print()

    # Cost estimation
    total_tokens = 8 * 3600 * 263
    input_cost = total_tokens * (1.25 / 1_000_000)
    output_tokens = 20_000
    output_cost = output_tokens * (5.00 / 1_000_000)
    total_daily = input_cost + output_cost

    print("  Cost estimate:")
    print(f"    Input: {total_tokens/1e6:.2f}M tokens x $1.25/1M = ${input_cost:.2f}/day")
    print(f"    Output: ~{output_tokens/1000:.0f}K tokens x $5/1M = ${output_cost:.2f}/day")
    print(f"    Total: ~${total_daily:.2f}/day")
    print()

    print("  Reliability improvements:")
    print("    - Low temperature (0.1) for factual analysis")
    print("    - Only report HIGH/MEDIUM confidence incidents")
    print("    - 10-min overlaps catch boundary incidents")
    print("    - Final synthesis is text-only (no re-upload)")


if __name__ == "__main__":
    print("=== Exercise 1: Model Selection ===")
    exercise_1()
    print("\n=== Exercise 2: Image Token Cost Calculation ===")
    exercise_2()
    print("\n=== Exercise 3: Structured Output Prompt ===")
    exercise_3()
    print("\n=== Exercise 4: Gemini Video Analysis ===")
    exercise_4()
    print("\nAll exercises completed!")
