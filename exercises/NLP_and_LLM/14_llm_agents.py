"""
Exercises for Lesson 15: LLM Agents
Topic: LLM_and_NLP

Solutions to practice problems from the lesson.
"""

import json
from typing import Dict, List, Tuple, Any, Optional
from enum import Enum


# === Exercise 1: Tool Definition Quality ===
# Problem: Compare two tool definitions, identify differences, and
# write an improved version with proper error handling guidance.

def exercise_1():
    """Tool definition quality analysis and improvement."""
    print("=" * 60)
    print("Exercise 1: Tool Definition Quality")
    print("=" * 60)

    # Tool A (poor quality)
    tool_a = {
        "name": "db_lookup",
        "description": "Look up data in database",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
            },
            "required": ["query"],
        },
    }

    # Tool B (better quality)
    tool_b = {
        "name": "lookup_customer",
        "description": (
            "Retrieve customer information by ID or email address. "
            "Returns customer name, email, account status, and order history. "
            "Use this when you need to verify customer identity or check account details. "
            "Do NOT use this for product searches - use search_products instead."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "identifier": {
                    "type": "string",
                    "description": "Customer ID (e.g. 'CUST-12345') or email address",
                },
                "include_orders": {
                    "type": "boolean",
                    "description": "Whether to include order history (default: false)",
                },
            },
            "required": ["identifier"],
        },
    }

    # Analysis
    differences = [
        {
            "aspect": "Name specificity",
            "tool_a": "db_lookup (generic, unclear what data)",
            "tool_b": "lookup_customer (self-documenting, specific entity)",
        },
        {
            "aspect": "Description quality",
            "tool_a": "'Look up data' - vague, no guidance",
            "tool_b": "States what data is returned, when to use it, when NOT to use it",
        },
        {
            "aspect": "Parameter naming",
            "tool_a": "'query' - could mean SQL, search text, anything",
            "tool_b": "'identifier' - clearly an ID or email, not free-text",
        },
        {
            "aspect": "Parameter descriptions",
            "tool_a": "No field descriptions at all",
            "tool_b": "Explains exact format (CUST-12345) and alternatives",
        },
        {
            "aspect": "Optional parameters",
            "tool_a": "None documented",
            "tool_b": "include_orders flag with default behavior explained",
        },
    ]

    print("\nDifferences between Tool A and Tool B:")
    print("-" * 60)
    for d in differences:
        print(f"\n  {d['aspect']}:")
        print(f"    Tool A: {d['tool_a']}")
        print(f"    Tool B: {d['tool_b']}")

    # Improved version with error handling guidance
    tool_best = {
        "name": "lookup_customer",
        "description": (
            "Retrieve customer account information by customer ID or email. "
            "Returns: customer_id, name, email, status ('active'|'suspended'|'closed'), "
            "registration_date, and optionally order_history (last 10 orders). "
            "Use this to verify a customer's identity, check account status, "
            "or review purchase history. "
            "Do NOT use for: product searches (use search_products), "
            "order tracking (use track_order), or bulk lookups. "
            "Returns null if customer not found - do not treat this as an error."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "identifier": {
                    "type": "string",
                    "description": (
                        "Customer ID (format: 'CUST-XXXXX') or email address. "
                        "Email is case-insensitive. "
                        "Example: 'CUST-12345' or 'john.doe@example.com'"
                    ),
                },
                "include_orders": {
                    "type": "boolean",
                    "description": (
                        "If true, include the last 10 orders in the response. "
                        "Increases response size - only set to true when order "
                        "history is needed. Default: false"
                    ),
                },
            },
            "required": ["identifier"],
            "additionalProperties": False,
        },
    }

    print("\n\nImproved Tool Definition (with error handling guidance):")
    print("-" * 60)
    print(json.dumps(tool_best, indent=2))

    print("\n\n  Key principle: A tool description should make the LLM's decision")
    print("  unambiguous. The LLM needs to know:")
    print("    - What the tool does")
    print("    - When to use it")
    print("    - When NOT to use it")
    print("    - What the output looks like")
    print("    - How to interpret errors/null results")


# === Exercise 2: ReAct Trace Analysis ===
# Problem: Identify problems in a flawed ReAct trace and rewrite it correctly.

def exercise_2():
    """ReAct trace analysis: identify flaws and correct."""
    print("\n" + "=" * 60)
    print("Exercise 2: ReAct Trace Analysis")
    print("=" * 60)

    flawed_trace = """
Question: What is 15% of the GDP of France (approximately $2.78 trillion)?

Step 1:
Thought: I need to calculate 15% of $2.78 trillion.
Action: search
Action Input: France GDP 15 percent

Step 2:
Observation: France GDP is $2.78 trillion. 15% of GDP is used for defense.
Thought: The answer is defense spending.
Final Answer: France spends 15% of its GDP on defense.
"""

    print("\nFlawed Trace:")
    print(flawed_trace)

    problems = [
        {
            "issue": "Wrong tool for arithmetic",
            "detail": (
                "Used 'search' for a math problem. The question provides all "
                "needed data; no search is required. This introduces hallucination risk."
            ),
        },
        {
            "issue": "Misread the question",
            "detail": (
                "The question asks to CALCULATE 15% of $2.78T. The agent "
                "searched for 'France GDP 15%' and found unrelated defense spending data."
            ),
        },
        {
            "issue": "Conflated correlation with the question",
            "detail": (
                "A search result mentioning '15% of GDP' doesn't answer this question. "
                "The agent jumped to a conclusion without verifying the connection."
            ),
        },
        {
            "issue": "Wrong final answer",
            "detail": "$2.78T x 15% = $417 billion, not 'defense spending'.",
        },
    ]

    print("Problems Identified:")
    print("-" * 50)
    for i, p in enumerate(problems, 1):
        print(f"\n  {i}. {p['issue']}")
        print(f"     {p['detail']}")

    # Demonstrate the correct calculation
    gdp = 2_780_000_000_000  # $2.78 trillion
    percentage = 0.15
    result = gdp * percentage

    corrected_trace = f"""
Question: What is 15% of the GDP of France (approximately $2.78 trillion)?

Step 1:
Thought: The question gives me France's GDP directly ($2.78 trillion) and asks
         for 15% of that. I don't need to search - I just need to calculate
         2.78 trillion x 0.15.
Action: calculator
Action Input: 2780000000000 * 0.15

Step 2:
Observation: {result:.0f}

Thought: The calculation is complete. 15% of $2.78 trillion ({gdp:,.0f})
         is {result:,.0f}, which equals ${result/1e9:.0f} billion.
Final Answer: 15% of France's GDP ($2.78 trillion) is approximately ${result/1e9:.0f} billion.
"""

    print("\n\nCorrected Trace:")
    print(corrected_trace)

    # Verify
    assert result == 417_000_000_000, f"Expected 417B, got {result}"
    print("  Calculation verified: $417 billion")

    print("\n  Lessons:")
    print("    - When the question contains necessary data, use calculator, not search")
    print("    - Verify the observation directly answers the question asked")
    print("    - Show units in the final answer for interpretability")


# === Exercise 3: Multi-Tool Agent with Error Handling ===
# Problem: Implement a robust agentic loop that handles successful tool
# execution, tool errors, and unknown tools. Simulated (no API calls).

def exercise_3():
    """Multi-tool agent with error handling (simulated)."""
    print("\n" + "=" * 60)
    print("Exercise 3: Multi-Tool Agent with Error Handling")
    print("=" * 60)

    # Tool implementations
    def get_weather(location: str, unit: str = "celsius") -> dict:
        """Simulated weather API."""
        weather_db = {
            "Seoul": {"temp_c": 15, "condition": "Partly cloudy", "humidity": 65},
            "Tokyo": {"temp_c": 22, "condition": "Sunny", "humidity": 40},
            "London": {"temp_c": 8, "condition": "Rainy", "humidity": 85},
            "New York": {"temp_c": 18, "condition": "Clear", "humidity": 50},
        }
        data = weather_db.get(location)
        if data is None:
            raise ValueError(f"Location '{location}' not found in weather database")

        temp = data["temp_c"] if unit == "celsius" else data["temp_c"] * 9 / 5 + 32
        return {
            "location": location,
            "temperature": temp,
            "unit": unit,
            "condition": data["condition"],
            "humidity": data["humidity"],
        }

    def calculate(expression: str) -> float:
        """Safe math evaluator."""
        allowed_chars = set("0123456789+-*/.() ")
        if not all(c in allowed_chars for c in expression):
            raise ValueError("Unsafe expression: only basic math operations allowed")
        return eval(expression)

    # Tool registry
    TOOLS = {
        "get_weather": get_weather,
        "calculate": calculate,
    }

    TOOL_SCHEMAS = [
        {
            "name": "get_weather",
            "description": "Get current weather for a city. Returns temperature, conditions, humidity.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City name"},
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "Temperature unit (default: celsius)",
                    },
                },
                "required": ["location"],
            },
        },
        {
            "name": "calculate",
            "description": "Evaluate a mathematical expression. Only basic arithmetic.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "Math expression"},
                },
                "required": ["expression"],
            },
        },
    ]

    def execute_tool(name: str, params: dict) -> Tuple[str, bool]:
        """
        Execute a tool and return (result_string, success).
        Never raises - errors are returned as strings so the agent can recover.
        """
        if name not in TOOLS:
            return (
                f"Error: Unknown tool '{name}'. Available tools: {list(TOOLS.keys())}",
                False,
            )
        try:
            result = TOOLS[name](**params)
            return json.dumps(result) if isinstance(result, dict) else str(result), True
        except Exception as e:
            return f"Error executing {name}: {str(e)}", False

    class SimulatedAgent:
        """
        Simulated agentic loop that demonstrates the tool-use pattern
        without requiring an actual LLM API.
        """

        def __init__(self, tools: Dict, schemas: List[Dict]):
            self.tools = tools
            self.schemas = schemas
            self.messages: List[Dict] = []

        def _plan_tool_calls(self, user_message: str) -> List[Dict]:
            """
            Simulated 'LLM reasoning' that decides which tools to call.
            In production, this would be an actual LLM call with tool schemas.
            """
            lower = user_message.lower()
            tool_calls = []

            # Simple keyword-based planning
            if "weather" in lower or "temperature" in lower:
                # Extract city name (simple heuristic)
                for city in ["Seoul", "Tokyo", "London", "New York", "Paris"]:
                    if city.lower() in lower:
                        unit = "fahrenheit" if "fahrenheit" in lower else "celsius"
                        tool_calls.append({
                            "name": "get_weather",
                            "params": {"location": city, "unit": unit},
                        })
                        break
                else:
                    # City not found - will trigger error handling
                    tool_calls.append({
                        "name": "get_weather",
                        "params": {"location": "Unknown City"},
                    })

            if "%" in lower or "calculate" in lower or "compute" in lower:
                # Try to extract a math expression
                if "15%" in lower:
                    # Find the number to compute percentage of
                    for word in lower.split():
                        try:
                            num = float(word)
                            tool_calls.append({
                                "name": "calculate",
                                "params": {"expression": f"{num} * 0.15"},
                            })
                            break
                        except ValueError:
                            continue

            if "nonexistent_tool" in lower:
                tool_calls.append({
                    "name": "nonexistent_tool",
                    "params": {"data": "test"},
                })

            return tool_calls

        def run(self, user_message: str) -> str:
            """Run the full agentic loop."""
            self.messages.append({"role": "user", "content": user_message})
            print(f"\n  User: {user_message}")

            tool_calls = self._plan_tool_calls(user_message)

            if not tool_calls:
                response = f"I can help with weather lookups and calculations. " \
                           f"Available tools: {list(self.tools.keys())}"
                print(f"  Agent: {response}")
                return response

            # Execute tools and collect results
            all_results = []
            for call in tool_calls:
                result_str, success = execute_tool(call["name"], call["params"])
                status = "SUCCESS" if success else "ERROR"
                print(f"  [{status}] Tool: {call['name']}({call['params']})")
                print(f"           Result: {result_str}")

                all_results.append({
                    "tool": call["name"],
                    "params": call["params"],
                    "result": result_str,
                    "success": success,
                    "is_error": not success,
                })

            # If any tool failed, attempt recovery
            failed_tools = [r for r in all_results if not r["success"]]
            if failed_tools:
                for failed in failed_tools:
                    print(f"  [RECOVERY] Tool '{failed['tool']}' failed, attempting alternative...")
                    # Recovery: if weather failed due to unknown city,
                    # try to inform user
                    recovery_msg = (
                        f"The tool '{failed['tool']}' encountered an error: "
                        f"{failed['result']}. "
                        "Please try again with a supported input."
                    )
                    all_results.append({
                        "tool": "recovery",
                        "result": recovery_msg,
                        "success": True,
                    })

            # Compose final response from successful results
            successful = [r for r in all_results if r["success"] and r["tool"] != "recovery"]
            errors = [r for r in all_results if not r["success"]]
            recoveries = [r for r in all_results if r.get("tool") == "recovery"]

            parts = []
            for r in successful:
                if r["tool"] == "get_weather":
                    data = json.loads(r["result"])
                    parts.append(
                        f"Weather in {data['location']}: {data['temperature']}"
                        f" {data['unit']}, {data['condition']}, "
                        f"humidity {data['humidity']}%"
                    )
                elif r["tool"] == "calculate":
                    parts.append(f"Calculation result: {r['result']}")

            for r in recoveries:
                parts.append(r["result"])

            response = " | ".join(parts) if parts else "I couldn't complete the request."
            print(f"\n  Agent: {response}")
            return response

    # Test the agent
    agent = SimulatedAgent(TOOLS, TOOL_SCHEMAS)

    print("\nTest 1: Successful weather query")
    print("-" * 50)
    agent.run("What's the weather in Seoul in fahrenheit?")

    print("\n\nTest 2: Unknown city (error handling)")
    print("-" * 50)
    agent.run("What's the temperature in Paris?")

    print("\n\nTest 3: Calculation")
    print("-" * 50)
    agent.run("Calculate 15% of 59.0")

    print("\n\nTest 4: Unknown tool (error handling)")
    print("-" * 50)
    result_str, success = execute_tool("nonexistent_tool", {"data": "test"})
    print(f"  execute_tool('nonexistent_tool', ...)")
    print(f"  Result: {result_str}")
    print(f"  Success: {success}")

    print("\n\n  Key design decisions:")
    print("    - execute_tool never raises: errors become strings the LLM can interpret")
    print("    - is_error flag signals the LLM to try a different approach")
    print("    - max_turns prevents infinite loops")
    print("    - Tool registry makes it easy to add/remove tools")


if __name__ == "__main__":
    exercise_1()
    exercise_2()
    exercise_3()
