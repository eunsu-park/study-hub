"""
Claude API: Parallel Tool Execution Example

Demonstrates how Claude can call multiple tools in parallel
and how to handle multiple simultaneous tool results.

Requirements:
    pip install anthropic
"""

import anthropic
import json
import time
from concurrent.futures import ThreadPoolExecutor

client = anthropic.Anthropic()


# --- Tool Definitions ---

tools = [
    {
        "name": "fetch_weather",
        "description": "Fetch current weather for a city.",
        "input_schema": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"},
            },
            "required": ["city"],
        },
    },
    {
        "name": "fetch_exchange_rate",
        "description": "Fetch exchange rate between two currencies.",
        "input_schema": {
            "type": "object",
            "properties": {
                "from_currency": {"type": "string", "description": "Source currency code"},
                "to_currency": {"type": "string", "description": "Target currency code"},
            },
            "required": ["from_currency", "to_currency"],
        },
    },
    {
        "name": "fetch_news",
        "description": "Fetch latest news headlines for a topic.",
        "input_schema": {
            "type": "object",
            "properties": {
                "topic": {"type": "string", "description": "News topic"},
                "count": {
                    "type": "integer",
                    "description": "Number of headlines",
                    "default": 3,
                },
            },
            "required": ["topic"],
        },
    },
]

# --- Simulated Tool Implementations ---

MOCK_WEATHER = {
    "tokyo": {"temp": 22, "condition": "Partly Cloudy", "humidity": 55},
    "london": {"temp": 14, "condition": "Rainy", "humidity": 85},
    "new york": {"temp": 18, "condition": "Sunny", "humidity": 40},
    "seoul": {"temp": 20, "condition": "Clear", "humidity": 50},
    "paris": {"temp": 16, "condition": "Overcast", "humidity": 70},
}

MOCK_RATES = {
    ("USD", "JPY"): 149.50,
    ("USD", "EUR"): 0.92,
    ("USD", "GBP"): 0.79,
    ("EUR", "JPY"): 162.50,
    ("GBP", "USD"): 1.27,
    ("KRW", "USD"): 0.00075,
}

MOCK_NEWS = {
    "technology": [
        "AI Breakthrough: New Model Achieves Human-Level Reasoning",
        "Quantum Computing Milestone: 1000-Qubit Processor Announced",
        "Open Source LLM Surpasses Commercial Models in Benchmarks",
        "New Programming Language 'Oxide' Gains Rapid Adoption",
    ],
    "finance": [
        "Fed Signals Rate Cut in Coming Months",
        "Tech Stocks Rally to New All-Time Highs",
        "Cryptocurrency Market Cap Exceeds $3 Trillion",
        "Global GDP Growth Forecast Revised Upward",
    ],
}


def fetch_weather(city: str) -> str:
    """Simulate weather API call with latency."""
    time.sleep(0.5)  # Simulate API latency
    data = MOCK_WEATHER.get(city.lower(), {"temp": 20, "condition": "Unknown"})
    return json.dumps({"city": city, **data})


def fetch_exchange_rate(from_currency: str, to_currency: str) -> str:
    """Simulate exchange rate API call with latency."""
    time.sleep(0.3)  # Simulate API latency
    key = (from_currency.upper(), to_currency.upper())
    rate = MOCK_RATES.get(key)
    if rate is None:
        return json.dumps({"error": f"Rate not available: {key[0]}/{key[1]}"})
    return json.dumps({
        "from": from_currency.upper(),
        "to": to_currency.upper(),
        "rate": rate,
    })


def fetch_news(topic: str, count: int = 3) -> str:
    """Simulate news API call with latency."""
    time.sleep(0.4)  # Simulate API latency
    headlines = MOCK_NEWS.get(topic.lower(), ["No news available"])[:count]
    return json.dumps({"topic": topic, "headlines": headlines})


TOOL_MAP = {
    "fetch_weather": lambda args: fetch_weather(args["city"]),
    "fetch_exchange_rate": lambda args: fetch_exchange_rate(
        args["from_currency"], args["to_currency"]
    ),
    "fetch_news": lambda args: fetch_news(args["topic"], args.get("count", 3)),
}


# --- Parallel Execution ---

def execute_tools_parallel(tool_calls: list[dict]) -> list[dict]:
    """Execute multiple tool calls in parallel using ThreadPoolExecutor."""
    results = []

    with ThreadPoolExecutor(max_workers=len(tool_calls)) as executor:
        futures = {}
        for call in tool_calls:
            func = TOOL_MAP.get(call["name"])
            if func:
                future = executor.submit(func, call["input"])
                futures[future] = call["id"]
            else:
                results.append({
                    "type": "tool_result",
                    "tool_use_id": call["id"],
                    "content": json.dumps({"error": f"Unknown tool: {call['name']}"}),
                })

        for future in futures:
            tool_use_id = futures[future]
            result = future.result()
            results.append({
                "type": "tool_result",
                "tool_use_id": tool_use_id,
                "content": result,
            })

    return results


# --- Agentic Loop with Parallel Tools ---

def run_parallel_conversation(user_message: str) -> str:
    """Run a conversation that handles parallel tool calls."""
    messages = [{"role": "user", "content": user_message}]

    print(f"User: {user_message}\n")

    while True:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2048,
            tools=tools,
            messages=messages,
        )

        if response.stop_reason == "tool_use":
            # Collect all tool calls from this response
            tool_calls = []
            for block in response.content:
                if block.type == "tool_use":
                    tool_calls.append({
                        "id": block.id,
                        "name": block.name,
                        "input": block.input,
                    })
                    print(f"  Tool call: {block.name}({json.dumps(block.input)})")

            print(f"\n  Executing {len(tool_calls)} tools in parallel...")
            start = time.time()

            # Execute all tools in parallel
            tool_results = execute_tools_parallel(tool_calls)

            elapsed = time.time() - start
            print(f"  All tools completed in {elapsed:.2f}s\n")

            for result in tool_results:
                print(f"  Result [{result['tool_use_id'][:8]}]: {result['content']}")
            print()

            # Add to conversation
            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": tool_results})

        else:
            final_text = ""
            for block in response.content:
                if hasattr(block, "text"):
                    final_text += block.text
            print(f"Claude: {final_text}")
            return final_text


# --- Main ---

if __name__ == "__main__":
    print("=" * 70)
    print("Parallel Tool Execution Demo")
    print("=" * 70)
    print()

    # This prompt should trigger multiple parallel tool calls
    run_parallel_conversation(
        "I'm planning a trip. Can you check: "
        "1) Weather in Tokyo and London, "
        "2) USD to JPY and USD to GBP exchange rates, "
        "3) Latest technology news headlines?"
    )
