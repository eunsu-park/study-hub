"""
Claude API: Basic Tool Use Example

Demonstrates how to define tools and handle tool calls
with the Claude Messages API.

Requirements:
    pip install anthropic
"""

import anthropic
import json

client = anthropic.Anthropic()  # Uses ANTHROPIC_API_KEY env var


# --- Define Tools ---

tools = [
    {
        "name": "get_stock_price",
        "description": "Get the current stock price for a given ticker symbol.",
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Stock ticker symbol (e.g., 'AAPL', 'GOOGL')",
                },
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "calculate_portfolio_value",
        "description": "Calculate total portfolio value given stock holdings.",
        "input_schema": {
            "type": "object",
            "properties": {
                "holdings": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "ticker": {"type": "string"},
                            "shares": {"type": "number"},
                        },
                        "required": ["ticker", "shares"],
                    },
                    "description": "List of stock holdings",
                },
            },
            "required": ["holdings"],
        },
    },
]


# --- Tool Implementation ---

MOCK_PRICES = {
    "AAPL": 198.50,
    "GOOGL": 175.20,
    "MSFT": 420.80,
    "AMZN": 185.60,
    "TSLA": 245.30,
}


def get_stock_price(ticker: str) -> dict:
    """Simulate fetching a stock price."""
    price = MOCK_PRICES.get(ticker.upper())
    if price is None:
        return {"error": f"Unknown ticker: {ticker}"}
    return {"ticker": ticker.upper(), "price": price, "currency": "USD"}


def calculate_portfolio_value(holdings: list[dict]) -> dict:
    """Calculate total portfolio value."""
    total = 0.0
    details = []

    for holding in holdings:
        ticker = holding["ticker"].upper()
        shares = holding["shares"]
        price = MOCK_PRICES.get(ticker)

        if price is None:
            details.append({
                "ticker": ticker,
                "shares": shares,
                "error": "Unknown ticker",
            })
            continue

        value = price * shares
        total += value
        details.append({
            "ticker": ticker,
            "shares": shares,
            "price": price,
            "value": value,
        })

    return {"total_value": total, "currency": "USD", "holdings": details}


def process_tool_call(tool_name: str, tool_input: dict) -> str:
    """Route tool calls to their implementations."""
    if tool_name == "get_stock_price":
        result = get_stock_price(tool_input["ticker"])
    elif tool_name == "calculate_portfolio_value":
        result = calculate_portfolio_value(tool_input["holdings"])
    else:
        result = {"error": f"Unknown tool: {tool_name}"}
    return json.dumps(result)


# --- Agentic Loop ---

def run_conversation(user_message: str) -> str:
    """Run a conversation with tool use, handling the full agentic loop."""
    messages = [{"role": "user", "content": user_message}]

    print(f"User: {user_message}\n")

    while True:
        # Call Claude
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            tools=tools,
            messages=messages,
        )

        # Check if Claude wants to use tools
        if response.stop_reason == "tool_use":
            # Process all tool calls in the response
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    print(f"  Tool call: {block.name}({json.dumps(block.input)})")
                    result = process_tool_call(block.name, block.input)
                    print(f"  Result: {result}\n")
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                    })

            # Add assistant response and tool results to messages
            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": tool_results})

        else:
            # Claude is done — extract final text response
            final_text = ""
            for block in response.content:
                if hasattr(block, "text"):
                    final_text += block.text
            print(f"Claude: {final_text}")
            return final_text


# --- Main ---

if __name__ == "__main__":
    # Example 1: Simple tool call
    print("=" * 60)
    print("Example 1: Single Tool Call")
    print("=" * 60)
    run_conversation("What's the current price of Apple stock?")

    print("\n")

    # Example 2: Multiple tool calls
    print("=" * 60)
    print("Example 2: Multi-Tool Interaction")
    print("=" * 60)
    run_conversation(
        "I have 100 shares of AAPL, 50 shares of GOOGL, and 30 shares of MSFT. "
        "What's my portfolio worth?"
    )
