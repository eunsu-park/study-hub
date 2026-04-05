# 16_agent_prompting.py — Tool-use prompts, ReAct pattern, planning prompts
#
# Run: python 16_agent_prompting.py

import anthropic
import json
import math

# ---------------------------------------------------------------------------
# 1. Tool definitions for the Anthropic API
# ---------------------------------------------------------------------------
TOOLS = [
    {
        "name": "calculator",
        "description": "Evaluate a math expression (supports +, -, *, /, **, sqrt, sin, cos, log).",
        "input_schema": {
            "type": "object",
            "properties": {"expression": {"type": "string", "description": "e.g. '2**10' or 'sqrt(144)'"}},
            "required": ["expression"],
        },
    },
    {
        "name": "lookup_constant",
        "description": "Look up a scientific constant (pi, e, c, g, avogadro).",
        "input_schema": {
            "type": "object",
            "properties": {"name": {"type": "string", "description": "Constant name"}},
            "required": ["name"],
        },
    },
    {
        "name": "unit_convert",
        "description": "Convert between units (km/miles, kg/lbs, m/ft, celsius/fahrenheit).",
        "input_schema": {
            "type": "object",
            "properties": {
                "value": {"type": "number"}, "from_unit": {"type": "string"}, "to_unit": {"type": "string"},
            },
            "required": ["value", "from_unit", "to_unit"],
        },
    },
]

# ---------------------------------------------------------------------------
# 2. Tool implementations
# ---------------------------------------------------------------------------
CONSTANTS = {"pi": math.pi, "e": math.e, "c": 299_792_458, "g": 9.80665, "avogadro": 6.02214076e23}
CONVERSIONS = {
    ("km", "miles"): 0.621371, ("miles", "km"): 1.60934,
    ("kg", "lbs"): 2.20462, ("lbs", "kg"): 0.453592,
    ("m", "ft"): 3.28084, ("ft", "m"): 0.3048,
    ("celsius", "fahrenheit"): lambda v: v * 9 / 5 + 32,
    ("fahrenheit", "celsius"): lambda v: (v - 32) * 5 / 9,
}
SAFE_MATH = {"sqrt": math.sqrt, "sin": math.sin, "cos": math.cos,
             "tan": math.tan, "log": math.log, "abs": abs, "pi": math.pi, "e": math.e}


def execute_tool(name: str, inp: dict) -> str:
    """Execute a tool and return JSON result."""
    if name == "calculator":
        try:
            result = eval(inp["expression"], {"__builtins__": {}}, SAFE_MATH)
            return json.dumps({"result": result})
        except Exception as exc:
            return json.dumps({"error": str(exc)})
    elif name == "lookup_constant":
        val = CONSTANTS.get(inp["name"].lower())
        return json.dumps({"value": val} if val else {"error": f"Unknown: {inp['name']}"})
    elif name == "unit_convert":
        key = (inp["from_unit"].lower(), inp["to_unit"].lower())
        factor = CONVERSIONS.get(key)
        if factor is None:
            return json.dumps({"error": f"No conversion for {key}"})
        result = factor(inp["value"]) if callable(factor) else inp["value"] * factor
        return json.dumps({"result": round(result, 6), "unit": inp["to_unit"]})
    return json.dumps({"error": f"Unknown tool: {name}"})


# ---------------------------------------------------------------------------
# 3. Agent system prompt (ReAct-style)
# ---------------------------------------------------------------------------
AGENT_SYSTEM = """\
You are a scientific assistant with access to tools. For each question:
1. THINK about what information and tools you need.
2. USE tools to gather data.
3. SYNTHESIZE results into a clear answer.
Use the calculator tool for any computation."""


# ---------------------------------------------------------------------------
# 4. Agentic loop — multi-turn tool use
# ---------------------------------------------------------------------------
def agent_loop(client: anthropic.Anthropic, question: str, max_turns: int = 6) -> str:
    """Run the agent loop: send message, handle tool calls, repeat."""
    messages = [{"role": "user", "content": question}]

    for turn in range(max_turns):
        print(f"  [Turn {turn + 1}]")
        response = client.messages.create(
            model="claude-sonnet-4-20250514", max_tokens=1024,
            system=AGENT_SYSTEM, tools=TOOLS, messages=messages,
        )
        assistant_content = response.content
        messages.append({"role": "assistant", "content": assistant_content})

        tool_blocks = [b for b in assistant_content if b.type == "tool_use"]
        if not tool_blocks:
            return "\n".join(b.text for b in assistant_content if hasattr(b, "text"))

        tool_results = []
        for tb in tool_blocks:
            print(f"    Tool: {tb.name}({json.dumps(tb.input)})")
            result = execute_tool(tb.name, tb.input)
            print(f"    Result: {result}")
            tool_results.append({"type": "tool_result", "tool_use_id": tb.id, "content": result})
        messages.append({"role": "user", "content": tool_results})

    return "[Agent reached max turns without final answer]"


# ---------------------------------------------------------------------------
# 5. Planning prompt — decompose a goal into steps
# ---------------------------------------------------------------------------
PLANNING_PROMPT = """\
You are a planning assistant. Given a goal, output a numbered step-by-step plan. \
Each step should be one concrete action. Do NOT execute, just plan.

Goal: {goal}"""


def generate_plan(client: anthropic.Anthropic, goal: str) -> str:
    resp = client.messages.create(
        model="claude-sonnet-4-20250514", max_tokens=512,
        messages=[{"role": "user", "content": PLANNING_PROMPT.format(goal=goal)}],
    )
    return resp.content[0].text


# ---------------------------------------------------------------------------
# 6. Main
# ---------------------------------------------------------------------------
def main() -> None:
    client = anthropic.Anthropic()

    # Planning demo
    print("=" * 60)
    print("PLANNING PROMPT")
    print("=" * 60)
    try:
        plan = generate_plan(client, "Build a REST API for a todo app with auth and deploy it")
        print(plan)
    except anthropic.APIError as exc:
        print(f"[API Error] {exc}")

    # Agent tool-use demo
    print("\n" + "=" * 60)
    print("AGENT TOOL-USE (ReAct pattern)")
    print("=" * 60)
    for q in ["What is the speed of light in miles per second?",
              "How many times does light circle Earth in 1 second? (circumference ~ 40075 km)"]:
        print(f"\nQ: {q}\n" + "-" * 40)
        try:
            answer = agent_loop(client, q)
            print(f"\nFinal answer:\n{answer[:300]}")
        except anthropic.APIError as exc:
            print(f"[API Error] {exc}")


if __name__ == "__main__":
    main()
