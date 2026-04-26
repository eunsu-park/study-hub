# 14. LLM Agents

## Learning Objectives

- Understand agent concepts and architecture
- Implement ReAct pattern
- Tool use techniques
- LangChain Agent utilization
- Autonomous agent systems (AutoGPT, etc.)

---

## Theory & Principles

An LLM agent is a system in which the LLM itself **decides what to do next** — what tool to call, what argument to pass, when to stop — instead of following a fixed pipeline. The defining property is **dynamic control flow**: the program's control flow is determined by the LLM's outputs at runtime, not by static code. This unlocks tasks that no fixed pipeline can solve (open-ended research, multi-step planning, error recovery), but introduces failure modes that fixed pipelines do not have (loops, hallucinated tools, runaway costs).

This section covers:

- **(A) The agent loop** — what minimally constitutes an agent, the perception–reasoning–action cycle.
- **(B) ReAct (Reasoning + Acting)** — the dominant prompting paradigm, why interleaving thoughts and actions works.
- **(C) Tool use protocol** — how the LLM emits structured tool calls, how the framework parses and dispatches them.
- **(D) Function calling** — the modern tool-calling API (OpenAI, Anthropic), why JSON-schema-based dispatch beats text parsing.
- **(E) Autonomy spectrum** — from copilots (human approves every action) to fully autonomous agents (AutoGPT-style); the trade-offs.
- **(F) Failure modes** — loops, hallucinated tools, observation truncation, cost runaway, and standard mitigations.

### A. The Agent Loop

At its core, every agent runs a loop:

```
state ← initial_context (user query, tools available, history)
while not done:
    action ← LLM(state)             # decide what to do
    observation ← execute(action)    # carry it out
    state ← state + (action, observation)
done = (action.type == "final_answer") or (steps > max_steps)
```

Three components:

1. **The LLM** (the brain): outputs a structured action choice given the current state.
2. **A set of tools** (effectors): functions the agent can call (search, calculator, code execution, file read/write, API call).
3. **Memory of the trajectory** (state): what has happened so far, accumulated so the LLM can reason over the full history.

This loop is the same whether the framework is LangChain, AutoGen, CrewAI, or hand-rolled. The differences are in how each component is structured and how the loop is observed and controlled.

### B. ReAct: Reasoning + Acting

ReAct (Yao et al., 2022) is the dominant agent prompting pattern. The LLM is prompted to alternate between *thoughts* (free-form reasoning) and *actions* (tool calls):

```
Thought: I need to find the population of France in 2024.
Action: search("France population 2024")
Observation: 68.4 million as of 2024.
Thought: Now I need to compare to Germany.
Action: search("Germany population 2024")
Observation: 84.5 million.
Thought: France has fewer people than Germany.
Action: final_answer("France: 68.4M, Germany: 84.5M; Germany has more.")
```

**Why it works.** The Thought tokens externalize reasoning, just like Chain-of-Thought (lesson 9). The Action tokens commit to a verifiable next step. The Observation tokens ground the LLM in real data, preventing hallucination. The interleaving means each new tool call can incorporate everything learned so far.

ReAct is essentially Chain-of-Thought (lesson 9) extended with **side effects**: the model can not only think but also act on the world and observe results.

### C. Tool Use Protocol

The LLM's output must be structured so the framework can parse "the model wants to call search with arguments X." Two approaches:

**C.1 Text-parsing (older).** The agent prompt instructs the LLM to format actions like `Action: tool_name[argument]`. A regex extracts tool name and argument. Brittle: any deviation breaks parsing.

**C.2 Function calling / tool calling (modern).** The LLM is fine-tuned to emit a structured JSON object specifying tool calls. The API surfaces this as a typed `tool_calls` field. Robust: the structure is enforced at the model level (logit constraints in the API).

Lesson 23 covers tool calling APIs in depth. For agents, the takeaway is: **always use the tool-calling API when the model supports it; reserve text parsing for older or local models that don't.**

### D. Function Calling: JSON-Schema-Based Dispatch

Modern function calling takes a list of tool descriptions in JSON Schema:

```json
{
  "name": "search",
  "description": "Search the web for a query",
  "parameters": {
    "type": "object",
    "properties": {"query": {"type": "string"}},
    "required": ["query"]
  }
}
```

The LLM's response can include:

```json
{
  "tool_calls": [
    {"id": "call_1", "name": "search", "arguments": {"query": "France population 2024"}}
  ]
}
```

The framework dispatches to `tools["search"](**arguments)`, captures the return value, and feeds it back as a tool message. The schema constrains hallucinations (you can't call a non-existent tool, can't pass unknown arguments). Multiple tool calls in one response enable parallel execution.

### E. The Autonomy Spectrum

Agents differ in how much autonomy they have:

| Level | Pattern | Example |
|-------|---------|---------|
| 0 | LLM with no tools | ChatGPT in chat mode |
| 1 | LLM with retrieval | Single-shot RAG (lesson 10) |
| 2 | Tool-calling, single action | "answer this query, you may use these tools" |
| 3 | ReAct loop, bounded steps | "research and answer, max 10 steps" |
| 4 | Self-planning | LLM produces a plan first, then executes |
| 5 | Fully autonomous, indefinite loop | AutoGPT, BabyAGI — pursue a goal until done |

Higher autonomy = more capability but more risk. At level 5, costs and failure modes scale fast. Production agents typically sit at levels 2-3 with hard step caps and cost ceilings.

### F. Failure Modes and Mitigations

**F.1 Loops.** The agent calls the same tool with the same arguments repeatedly. Mitigation: detect repeats (cache + comparison), inject "you have already tried this" feedback.

**F.2 Hallucinated tools.** The LLM emits a tool name not in the available list. Mitigation: function calling (§D) makes this nearly impossible. For text-parsing agents, validate against the tool registry and re-prompt.

**F.3 Observation truncation.** A tool returns 50 KB of text; appended to context, this rapidly exhausts the context window. Mitigation: truncate observations, summarize, or store in a separate "scratchpad" referenced by ID.

**F.4 Cost runaway.** Each step calls the LLM (expensive) and possibly tools (also expensive). Mitigation: hard caps on (a) max steps per agent run, (b) max total tokens, (c) max wall-clock time, (d) per-tool cost budgets.

**F.5 Tool errors.** A tool throws an exception or returns an error. Mitigation: catch, format as observation ("Error: <message>"), let the LLM react. Often the agent recovers by trying a different approach.

**F.6 Unsafe actions.** A tool with side effects (file delete, payment, email send) is called with wrong arguments. Mitigation: human-in-the-loop confirmation for destructive actions; sandbox; per-tool allow-lists.

### From Theory to the Functions Below

- §1 (overview) — frames the §A agent loop and §E autonomy spectrum.
- §2 (ReAct) — implements §B with text-parsing prompts (educational).
- §3 (Tool use) — covers §C, including the older text-parsing approach.
- §3.5 (Tool use deep dive) — multi-provider patterns for §D function calling.
- §4 (LangChain Agent) — wraps §A-§D with `create_tool_calling_agent` and `AgentExecutor`.
- §5 (autonomous agents) — implements §E level-5 (AutoGPT-style) with the §F failure-mode mitigations.
- §6 (multi-agent) — pointer to lesson 15.
- §7 (agent evaluation) — pointer to lesson 26.

---

## 1. LLM Agent Overview

### What is an Agent?

> **LLM Agent**
>
> 1. **LLM (Brain)** -- decision-making
> 2. **Planning** -- plan formulation
> 3. **Tools** (Search, Calculator, Code exec) + **Memory** (Chat history, Knowledge base)

### Agent vs Chatbot

| Aspect | Chatbot | Agent |
|--------|---------|-------|
| Response Method | Single response | Multi-step reasoning |
| Tool Use | Limited | Diverse tools |
| Autonomy | Low | High |
| Planning | None | Yes |
| Example | Customer support bot | AutoGPT, Copilot |

---

## 2. ReAct (Reasoning + Acting)

### ReAct Pattern

```
Thought: Analyze problem and decide next action
Action: Select tool and determine input
Observation: Tool execution result
... (repeat)
Final Answer: Final response
```

### ReAct Implementation

```python
from openai import OpenAI

client = OpenAI()

# Tool definitions
tools = {
    "calculator": lambda expr: eval(expr),
    "search": lambda query: f"Search results: Information about {query}...",
    "get_weather": lambda city: f"Weather in {city}: Sunny, 25°C",
}

def react_agent(question, max_steps=5):
    """ReAct agent"""

    system_prompt = """You are an agent that solves problems step by step.

Available tools:
- calculator: Perform math calculations (e.g., "2 + 3 * 4")
- search: Search for information (e.g., "Python creator")
- get_weather: Check weather (e.g., "Seoul")

Follow this format:

Thought: [Analyze current situation and plan next action]
Action: [tool name]
Action Input: [tool input]

When you receive tool results:
Observation: [result]

When final answer is ready:
Final Answer: [answer]
"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question}
    ]

    for step in range(max_steps):
        response = client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            temperature=0
        )

        assistant_message = response.choices[0].message.content
        messages.append({"role": "assistant", "content": assistant_message})

        print(f"=== Step {step + 1} ===")
        print(assistant_message)

        # Check for Final Answer
        if "Final Answer:" in assistant_message:
            final_answer = assistant_message.split("Final Answer:")[-1].strip()
            return final_answer

        # Parse Action
        if "Action:" in assistant_message and "Action Input:" in assistant_message:
            action_line = assistant_message.split("Action:")[-1].split("\n")[0].strip()
            input_line = assistant_message.split("Action Input:")[-1].split("\n")[0].strip()

            # Execute tool
            if action_line in tools:
                try:
                    observation = tools[action_line](input_line)
                except Exception as e:
                    observation = f"Error: {str(e)}"

                observation_message = f"Observation: {observation}"
                messages.append({"role": "user", "content": observation_message})
                print(observation_message)
            else:
                messages.append({"role": "user", "content": f"Error: Unknown tool '{action_line}'"})

    return "Maximum steps reached, unable to answer"

# Usage
answer = react_agent("Check the weather in Seoul and convert the temperature from Celsius to Fahrenheit.")
print(f"\nFinal answer: {answer}")
```

---

## 3. Tool Use

### Function Calling (OpenAI)

```python
from openai import OpenAI
import json

client = OpenAI()

# Tool definitions
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather information for a specific city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "City name (e.g., Seoul, Tokyo)"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "Temperature unit"
                    }
                },
                "required": ["city"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Search for information on the web.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query"
                    }
                },
                "required": ["query"]
            }
        }
    }
]

# Tool implementations
def get_weather(city, unit="celsius"):
    # In practice, call API
    weather_data = {
        "Seoul": {"temp": 25, "condition": "Sunny"},
        "Tokyo": {"temp": 28, "condition": "Cloudy"},
    }
    data = weather_data.get(city, {"temp": 20, "condition": "Unknown"})
    if unit == "fahrenheit":
        data["temp"] = data["temp"] * 9/5 + 32
    return json.dumps(data)

def search_web(query):
    return json.dumps({"results": f"Search results for '{query}'..."})

tool_implementations = {
    "get_weather": get_weather,
    "search_web": search_web,
}

def agent_with_tools(user_message):
    """Function Calling agent"""
    messages = [{"role": "user", "content": user_message}]

    response = client.chat.completions.create(
        model="gpt-4",
        messages=messages,
        tools=tools,
        tool_choice="auto"  # Automatically select tool
    )

    assistant_message = response.choices[0].message

    # Check if tool call is needed
    if assistant_message.tool_calls:
        messages.append(assistant_message)

        # Process each tool call
        for tool_call in assistant_message.tool_calls:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)

            # Execute tool
            function_response = tool_implementations[function_name](**function_args)

            # Add result
            messages.append({
                "tool_call_id": tool_call.id,
                "role": "tool",
                "name": function_name,
                "content": function_response
            })

        # Final response
        final_response = client.chat.completions.create(
            model="gpt-4",
            messages=messages
        )
        return final_response.choices[0].message.content

    return assistant_message.content

# Usage
result = agent_with_tools("Compare the weather in Seoul and Tokyo.")
print(result)
```

### Code Execution Tool

```python
import subprocess
import tempfile
import os

def execute_python(code):
    """Safely execute Python code"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(code)
        temp_path = f.name

    try:
        result = subprocess.run(
            ['python', temp_path],
            capture_output=True,
            text=True,
            timeout=10  # Set timeout
        )
        output = result.stdout if result.returncode == 0 else result.stderr
        return {"success": result.returncode == 0, "output": output}
    except subprocess.TimeoutExpired:
        return {"success": False, "output": "Timeout"}
    finally:
        os.unlink(temp_path)

# Code execution tool definition
code_tool = {
    "type": "function",
    "function": {
        "name": "execute_python",
        "description": "Execute Python code.",
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python code to execute"
                }
            },
            "required": ["code"]
        }
    }
}
```

## 3.5 Tool Use Deep Dive: Multi-Provider Patterns

Modern LLMs have moved beyond simple text generation to become reasoning engines that can take actions in the real world through tool use (also called function calling). This section provides a comprehensive comparison of how different providers implement this capability, along with advanced patterns for building robust tool-using agents.

### The Agentic Loop

Every tool-using LLM follows the same fundamental loop, regardless of provider:

> **Agentic Tool Use Loop**
>
> 1. **User Message**
> 2. **LLM (Think)** --tool_use--> **Tool Executor** --tool_result--> **LLM**
> 3. Repeat until LLM produces final text response
> 4. **Final Response**

### Claude Tool Use API

Claude uses a structured content-block approach where tool use and tool results are explicit message content types:

```python
import anthropic

client = anthropic.Anthropic()

# Tool definitions use JSON Schema for parameters
tools = [
    {
        "name": "get_weather",
        "description": "Get current weather for a location. Returns temperature, conditions, and humidity.",
        "input_schema": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City and state/country, e.g. 'San Francisco, CA'"
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "Temperature unit (default: celsius)"
                }
            },
            "required": ["location"]
        }
    },
    {
        "name": "calculate",
        "description": "Evaluate a mathematical expression.",
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Math expression to evaluate, e.g. '(4 + 5) * 3'"
                }
            },
            "required": ["expression"]
        }
    }
]

# Tool implementations
def execute_tool(name, input_data):
    if name == "get_weather":
        # In production, call a real weather API
        return f'{{"temp": 22, "condition": "Sunny", "humidity": 45}}'
    elif name == "calculate":
        try:
            return str(eval(input_data["expression"]))
        except Exception as e:
            return f"Error: {e}"
    return "Unknown tool"

# Agentic loop: keep calling until no more tool use
messages = [{"role": "user", "content": "What's the weather in Tokyo? Also, what's 15% of 340?"}]

while True:
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        tools=tools,
        messages=messages
    )

    # Collect assistant response
    messages.append({"role": "assistant", "content": response.content})

    # Check if we need to execute tools
    if response.stop_reason == "tool_use":
        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                result = execute_tool(block.name, block.input)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result
                })

        messages.append({"role": "user", "content": tool_results})
    else:
        # Final text response -- exit loop
        for block in response.content:
            if hasattr(block, "text"):
                print(block.text)
        break
```

### Provider Comparison: Tool Use APIs

| Feature | OpenAI | Claude (Anthropic) | Gemini (Google) |
|---------|--------|-------------------|-----------------|
| **Parameter name** | `tools` | `tools` | `tools` |
| **Schema format** | JSON Schema in `parameters` | JSON Schema in `input_schema` | OpenAPI-style schema |
| **Tool call signal** | `finish_reason: "tool_calls"` | `stop_reason: "tool_use"` | `finish_reason: "TOOL_CALL"` |
| **Call format** | `tool_calls[].function` | Content block `type: "tool_use"` | `function_call` in candidate |
| **Result format** | `role: "tool"` message | `tool_result` content block | `function_response` part |
| **Multi-tool** | Parallel calls in one response | Parallel calls in one response | Sequential (one at a time) |
| **Streaming** | Supported | Supported | Supported |
| **Tool choice** | `tool_choice: "auto"/"required"/"none"` | `tool_choice: {"type": "auto"/"any"/"tool"}` | `tool_config` |

### Tool Definition Best Practices

```python
# GOOD: Detailed description with examples and edge cases
good_tool = {
    "name": "search_database",
    "description": (
        "Search the product database by name, category, or price range. "
        "Returns up to 10 matching products sorted by relevance. "
        "Use this when the user asks about product availability, pricing, "
        "or specifications. Do NOT use for order status queries."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query string (e.g. 'red running shoes size 10')"
            },
            "max_price": {
                "type": "number",
                "description": "Maximum price filter in USD. Omit for no limit."
            },
            "category": {
                "type": "string",
                "enum": ["electronics", "clothing", "home", "sports"],
                "description": "Product category to filter by"
            }
        },
        "required": ["query"]
    }
}

# BAD: Vague description, no guidance on when to use
bad_tool = {
    "name": "search",
    "description": "Search for things",
    "input_schema": {
        "type": "object",
        "properties": {
            "q": {"type": "string"}
        },
        "required": ["q"]
    }
}
```

### ReAct Pattern (Reasoning + Acting)

The ReAct pattern makes the LLM's reasoning process explicit by alternating between thought and action steps:

```
User: "Find the population of France and calculate its GDP per capita
       given a GDP of $2.78 trillion."

Step 1:
  Thought: I need to find France's population first. Let me search for it.
  Action: search("France population 2024")
  Observation: France population is approximately 68.17 million (2024)

Step 2:
  Thought: Now I have the population (68.17 million) and GDP ($2.78 trillion).
           I need to calculate GDP per capita = GDP / population.
  Action: calculate("2780000000000 / 68170000")
  Observation: 40779.17

Step 3:
  Thought: I have all the information needed.
  Final Answer: France's population is approximately 68.17 million.
                With a GDP of $2.78 trillion, the GDP per capita is
                approximately $40,779.
```

### Tool Use Safety

```python
# 1. Parameter validation before execution
def safe_execute(tool_name, params, allowed_tools):
    """Execute tool with safety checks."""
    # Allowlist check
    if tool_name not in allowed_tools:
        return {"error": f"Tool '{tool_name}' is not permitted"}

    # Schema validation
    from jsonschema import validate, ValidationError
    try:
        validate(instance=params, schema=allowed_tools[tool_name]["input_schema"])
    except ValidationError as e:
        return {"error": f"Invalid parameters: {e.message}"}

    # Rate limiting
    if rate_limiter.is_exceeded(tool_name):
        return {"error": "Rate limit exceeded for this tool"}

    # Execute in sandbox (timeout, resource limits)
    try:
        result = execute_with_timeout(tool_name, params, timeout=30)
        return {"result": result}
    except TimeoutError:
        return {"error": "Tool execution timed out"}

# 2. Never pass unsanitized LLM output to system commands
# BAD: os.system(llm_generated_command)
# GOOD: Use parameterized APIs with validated inputs

# 3. Log all tool calls for audit
import logging
logger = logging.getLogger("tool_use")

def audited_execute(tool_name, params, user_id):
    logger.info(f"Tool call: {tool_name}, params: {params}, user: {user_id}")
    result = safe_execute(tool_name, params)
    logger.info(f"Tool result: {tool_name}, success: {'error' not in result}")
    return result
```

---

## 4. LangChain Agent

### Basic Agent

```python
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import Tool, tool
from langchain_community.tools import DuckDuckGoSearchRun

# LLM
llm = ChatOpenAI(model="gpt-4", temperature=0)

# Tool definitions
search = DuckDuckGoSearchRun()

@tool
def calculator(expression: str) -> str:
    """Perform mathematical calculations. Input: math expression (e.g., '2 + 3 * 4')"""
    try:
        return str(eval(expression))
    except:
        return "Calculation error"

@tool
def get_current_time() -> str:
    """Return current time."""
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

tools = [
    Tool(name="Search", func=search.run, description="Web search"),
    calculator,
    get_current_time,
]

# Prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant. Use tools to answer questions."),
    MessagesPlaceholder(variable_name="chat_history", optional=True),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# Create agent
agent = create_openai_tools_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Execute
result = agent_executor.invoke({"input": "Tell me the current time and today's major news."})
print(result["output"])
```

### ReAct Agent (LangChain)

```python
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4", temperature=0)

# ReAct prompt
react_prompt = PromptTemplate.from_template("""Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}""")

# Create agent
react_agent = create_react_agent(llm, tools, react_prompt)
agent_executor = AgentExecutor(agent=react_agent, tools=tools, verbose=True, handle_parsing_errors=True)

# Execute
result = agent_executor.invoke({"input": "Search for 2024 US presidential election results and summarize."})
```

### Agent with Memory

```python
from langchain.memory import ConversationBufferMemory
from langchain.agents import AgentExecutor, create_openai_tools_agent

# Memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Prompt (with memory)
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# Agent
agent = create_openai_tools_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, memory=memory, verbose=True)

# Conversation
agent_executor.invoke({"input": "My name is John."})
agent_executor.invoke({"input": "What did I say my name was?"})
```

---

## 5. Autonomous Agent Systems

### Plan-and-Execute

```python
from langchain.experimental.plan_and_execute import (
    PlanAndExecute,
    load_agent_executor,
    load_chat_planner
)
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4", temperature=0)

# Create Planner and Executor
planner = load_chat_planner(llm)
executor = load_agent_executor(llm, tools, verbose=True)

# Plan-and-Execute agent
agent = PlanAndExecute(planner=planner, executor=executor, verbose=True)

# Execute complex task
result = agent.run("Research the history of Python and create a markdown document summarizing key features by major version.")
```

### AutoGPT Style Agent

```python
class AutoGPTAgent:
    """Autonomous agent"""

    def __init__(self, llm, tools, goals):
        self.llm = llm
        self.tools = {t.name: t for t in tools}
        self.goals = goals
        self.memory = []
        self.completed_tasks = []

    def plan(self):
        """Plan to achieve goals"""
        prompt = f"""You are an autonomous AI agent.

Goals: {self.goals}

Completed tasks:
{self.completed_tasks}

Previous task results:
{self.memory[-5:] if self.memory else "None"}

Available tools:
{list(self.tools.keys())}

Output the next task in JSON format:
{{"task": "task description", "tool": "tool to use", "input": "tool input"}}

If all goals are achieved:
{{"task": "COMPLETE", "summary": "result summary"}}
"""
        response = self.llm.invoke(prompt)
        return json.loads(response.content)

    def execute(self, task):
        """Execute task"""
        if task["task"] == "COMPLETE":
            return {"status": "complete", "summary": task["summary"]}

        tool = self.tools.get(task["tool"])
        if tool:
            result = tool.run(task["input"])
            return {"status": "success", "result": result}
        return {"status": "error", "message": f"Unknown tool: {task['tool']}"}

    def run(self, max_iterations=10):
        """Run agent"""
        for i in range(max_iterations):
            print(f"\n=== Iteration {i+1} ===")

            # Plan
            task = self.plan()
            print(f"Task: {task}")

            # Check completion
            if task.get("task") == "COMPLETE":
                print(f"Goals achieved: {task['summary']}")
                return task["summary"]

            # Execute
            result = self.execute(task)
            print(f"Result: {result}")

            # Update memory
            self.memory.append({"task": task, "result": result})
            if result["status"] == "success":
                self.completed_tasks.append(task["task"])

        return "Max iterations reached"

# Usage
agent = AutoGPTAgent(
    llm=ChatOpenAI(model="gpt-4"),
    tools=tools,
    goals=["Research Seoul population", "Analyze demographics", "Create report"]
)
result = agent.run()
```

---

## 6. Multi-Agent Systems

### Agent Collaboration

```python
class ResearcherAgent:
    """Research agent"""
    def __init__(self, llm):
        self.llm = llm

    def research(self, topic):
        prompt = f"Research '{topic}' and summarize key information."
        return self.llm.invoke(prompt).content

class WriterAgent:
    """Writing agent"""
    def __init__(self, llm):
        self.llm = llm

    def write(self, research_results, style="formal"):
        prompt = f"Write a {style} style document based on the following information:\n{research_results}"
        return self.llm.invoke(prompt).content

class ReviewerAgent:
    """Review agent"""
    def __init__(self, llm):
        self.llm = llm

    def review(self, document):
        prompt = f"Review the following document and suggest improvements:\n{document}"
        return self.llm.invoke(prompt).content

class MultiAgentSystem:
    """Multi-agent system"""

    def __init__(self, llm):
        self.researcher = ResearcherAgent(llm)
        self.writer = WriterAgent(llm)
        self.reviewer = ReviewerAgent(llm)

    def create_document(self, topic, max_revisions=2):
        # 1. Research
        print("=== Research Phase ===")
        research = self.researcher.research(topic)
        print(research[:200] + "...")

        # 2. Write
        print("\n=== Writing Phase ===")
        document = self.writer.write(research)
        print(document[:200] + "...")

        # 3. Review and revise
        for i in range(max_revisions):
            print(f"\n=== Review {i+1} ===")
            review = self.reviewer.review(document)
            print(review[:200] + "...")

            # Revise
            if "no revisions needed" in review:
                break
            document = self.writer.write(f"Original:\n{document}\n\nReview:\n{review}", style="revised")

        return document

# Usage
llm = ChatOpenAI(model="gpt-4")
system = MultiAgentSystem(llm)
final_doc = system.create_document("The Future of Artificial Intelligence")
```

---

## 7. Agent Evaluation

### Tool Selection Accuracy

```python
def evaluate_tool_selection(agent, test_cases):
    """Evaluate tool selection accuracy"""
    correct = 0
    total = len(test_cases)

    for case in test_cases:
        query = case["query"]
        expected_tool = case["expected_tool"]

        # Run agent (tool selection only)
        result = agent.plan(query)
        selected_tool = result.get("tool")

        if selected_tool == expected_tool:
            correct += 1
            print(f"[CORRECT] Query: {query}, Tool: {selected_tool}")
        else:
            print(f"[WRONG] Query: {query}, Expected: {expected_tool}, Got: {selected_tool}")

    accuracy = correct / total
    print(f"\nTool Selection Accuracy: {accuracy:.2%}")
    return accuracy

# Test cases
test_cases = [
    {"query": "Calculate 2 + 3 * 4", "expected_tool": "calculator"},
    {"query": "What's the weather in Seoul today?", "expected_tool": "get_weather"},
    {"query": "Who is the creator of Python?", "expected_tool": "search"},
]

# Evaluate
evaluate_tool_selection(agent, test_cases)
```

### Task Completion Rate

```python
def evaluate_task_completion(agent, tasks):
    """Evaluate task completion rate"""
    results = []

    for task in tasks:
        try:
            result = agent.run(task["input"])
            success = task["validator"](result)
            results.append({
                "task": task["description"],
                "success": success,
                "result": result
            })
        except Exception as e:
            results.append({
                "task": task["description"],
                "success": False,
                "error": str(e)
            })

    completion_rate = sum(r["success"] for r in results) / len(results)
    print(f"Task Completion Rate: {completion_rate:.2%}")
    return results

# Task definitions
tasks = [
    {
        "description": "Weather check and clothing recommendation",
        "input": "Check Seoul weather and recommend what to wear today",
        "validator": lambda r: "Seoul" in r and ("clothing" in r or "wear" in r)
    },
    {
        "description": "Math calculation",
        "input": "What is 123 * 456?",
        "validator": lambda r: "56088" in r
    },
]
```

---

## Summary

### Agent Architecture Comparison

| Architecture | Features | When to Use |
|--------------|----------|-------------|
| ReAct | Reasoning-action iteration | Step-by-step problem solving |
| Function Calling | Structured tool calls | API integration |
| Plan-and-Execute | Plan then execute | Complex tasks |
| AutoGPT | Autonomous goal achievement | Long-term tasks |
| Multi-Agent | Role-based collaboration | Specialized expertise needed |

### Core Code

```python
# ReAct pattern
Thought: Analyze problem
Action: Select tool
Observation: Check result
Final Answer: Final response

# Function Calling (OpenAI)
response = client.chat.completions.create(
    model="gpt-4",
    messages=messages,
    tools=tools,
    tool_choice="auto"
)

# LangChain Agent
agent = create_openai_tools_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools)
result = executor.invoke({"input": query})
```

### Agent Design Checklist

```
□ Clear tool definitions (name, description, parameters)
□ Error handling (tool failures, parsing errors)
□ Memory management (chat history, context)
□ Loop prevention (maximum iterations)
□ Safety measures (restrict dangerous operations)
□ Logging and monitoring
```

---

## Exercises

### Exercise 1: Tool Definition Quality

Below are two tool definitions for a database lookup function. Identify at least four differences, explain why the second definition is better, and write an improved version that also adds proper error handling guidance.

```python
# Tool A (poor quality)
tool_a = {
    "name": "db_lookup",
    "description": "Look up data in database",
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {"type": "string"}
        },
        "required": ["query"]
    }
}

# Tool B (better quality)
tool_b = {
    "name": "lookup_customer",
    "description": (
        "Retrieve customer information by ID or email address. "
        "Returns customer name, email, account status, and order history. "
        "Use this when you need to verify customer identity or check account details. "
        "Do NOT use this for product searches — use search_products instead."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "identifier": {
                "type": "string",
                "description": "Customer ID (e.g. 'CUST-12345') or email address"
            },
            "include_orders": {
                "type": "boolean",
                "description": "Whether to include order history (default: false)"
            }
        },
        "required": ["identifier"]
    }
}
```

<details>
<summary>Show Answer</summary>

**Differences and why Tool B is better:**

1. **Name specificity**: `db_lookup` vs `lookup_customer` — B is self-documenting; the LLM knows exactly what data this tool returns without reading the description
2. **Description quality**: A says "Look up data" (vague). B says what data is returned, when to use it, and what NOT to use it for — this prevents the LLM from using the wrong tool
3. **Parameter naming**: `query` is generic. `identifier` tells the LLM the input should be an ID or email, not a free-text search query
4. **Parameter description**: A has no field descriptions. B explains the exact format (`CUST-12345`) and alternatives (email)
5. **Optional parameters**: B documents an optional `include_orders` flag with its default behavior

**Improved version with error handling guidance:**
```python
tool_best = {
    "name": "lookup_customer",
    "description": (
        "Retrieve customer account information by customer ID or email. "
        "Returns: customer_id, name, email, status ('active'|'suspended'|'closed'), "
        "registration_date, and optionally order_history (last 10 orders). "
        "Use this to verify a customer's identity, check account status, or review purchase history. "
        "Do NOT use for: product searches (use search_products), "
        "order tracking (use track_order), or bulk lookups. "
        "Returns null if customer not found — do not treat this as an error."
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
                )
            },
            "include_orders": {
                "type": "boolean",
                "description": (
                    "If true, include the last 10 orders in the response. "
                    "Increases response size — only set to true when order history is needed. "
                    "Default: false"
                )
            }
        },
        "required": ["identifier"],
        "additionalProperties": False
    }
}
```

**Key principle:** A tool description should make the LLM's decision unambiguous. The LLM needs to know: what the tool does, when to use it, when NOT to use it, what the output looks like, and how to interpret errors/null results.
</details>

---

### Exercise 2: ReAct Trace Analysis

The following is a flawed ReAct agent trace. Identify all the problems and rewrite the trace correctly.

```
Question: What is 15% of the GDP of France (approximately $2.78 trillion)?

Step 1:
Thought: I need to calculate 15% of $2.78 trillion.
Action: search
Action Input: France GDP 15 percent

Step 2:
Observation: France GDP is $2.78 trillion. 15% of GDP is used for defense.
Thought: The answer is defense spending.
Final Answer: France spends 15% of its GDP on defense.
```

<details>
<summary>Show Answer</summary>

**Problems identified:**

1. **Wrong tool for arithmetic**: The agent used `search` to find a pre-existing answer to a math problem. This is fragile — the search result might not contain the exact calculation needed, and it introduces hallucination risk.
2. **Misread the question**: The question asks to *calculate* 15% of $2.78 trillion. The agent instead searched for "15% of GDP" and found a related but different fact (defense spending).
3. **Conflated correlation with the question**: Just because a search result mentions "15% of GDP" doesn't mean it answers the question. The agent jumped to a conclusion without verifying the connection.
4. **Wrong final answer**: $2.78 trillion × 15% = $417 billion, not "defense spending."

**Corrected trace:**
```
Question: What is 15% of the GDP of France (approximately $2.78 trillion)?

Step 1:
Thought: The question gives me France's GDP directly ($2.78 trillion) and asks
         for 15% of that. I don't need to search — I just need to calculate
         2.78 trillion × 0.15.
Action: calculator
Action Input: 2780000000000 * 0.15

Step 2:
Observation: 417000000000

Thought: The calculation is complete. 15% of $2.78 trillion (2,780,000,000,000)
         is 417,000,000,000, which equals $417 billion.
Final Answer: 15% of France's GDP ($2.78 trillion) is approximately $417 billion.
```

**Lessons:**
- When the question itself contains the necessary data, use a calculation tool — not search
- Verify that the observation directly answers the question asked, not a related question
- Show the units in the final answer (billions, trillions) to make it interpretable
</details>

---

### Exercise 3: Multi-Tool Agent with Error Handling

Implement a robust agentic loop using the Claude tool use API that handles three cases: successful tool execution, tool errors, and unknown tools. The agent should recover gracefully from errors and continue trying to answer the question.

<details>
<summary>Show Answer</summary>

```python
import anthropic
import json
from typing import Any

client = anthropic.Anthropic()

# Tool implementations
def get_weather(location: str, unit: str = "celsius") -> dict:
    """Simulated weather API."""
    weather_db = {
        "Seoul": {"temp_c": 15, "condition": "Partly cloudy", "humidity": 65},
        "Tokyo": {"temp_c": 22, "condition": "Sunny", "humidity": 40},
    }
    data = weather_db.get(location)
    if data is None:
        raise ValueError(f"Location '{location}' not found in weather database")

    temp = data["temp_c"] if unit == "celsius" else data["temp_c"] * 9/5 + 32
    return {"location": location, "temperature": temp, "unit": unit,
            "condition": data["condition"], "humidity": data["humidity"]}

def calculate(expression: str) -> float:
    """Safe math evaluator."""
    # Restrict to safe math operations
    allowed_chars = set("0123456789+-*/.() ")
    if not all(c in allowed_chars for c in expression):
        raise ValueError(f"Unsafe expression: only basic math operations allowed")
    return eval(expression)

# Tool registry
TOOLS = {
    "get_weather": get_weather,
    "calculate": calculate,
}

TOOL_SCHEMAS = [
    {
        "name": "get_weather",
        "description": "Get current weather for a city. Returns temperature, conditions, and humidity.",
        "input_schema": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "City name"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
            },
            "required": ["location"]
        }
    },
    {
        "name": "calculate",
        "description": "Evaluate a mathematical expression. Only basic arithmetic (+, -, *, /, parentheses).",
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "Math expression"}
            },
            "required": ["expression"]
        }
    }
]

def execute_tool(name: str, params: dict) -> tuple[str, bool]:
    """
    Execute a tool and return (result_string, success).
    Never raises — errors are returned as strings so the LLM can recover.
    """
    if name not in TOOLS:
        return f"Error: Unknown tool '{name}'. Available tools: {list(TOOLS.keys())}", False

    try:
        result = TOOLS[name](**params)
        return json.dumps(result) if isinstance(result, dict) else str(result), True
    except Exception as e:
        return f"Error executing {name}: {str(e)}", False

def run_agent(user_message: str, max_turns: int = 10) -> str:
    """
    Robust agentic loop with error recovery.
    Returns the final text response from the model.
    """
    messages = [{"role": "user", "content": user_message}]

    for turn in range(max_turns):
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            tools=TOOL_SCHEMAS,
            messages=messages
        )

        # Add assistant response to history
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "tool_use":
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    result_str, success = execute_tool(block.name, block.input)

                    if not success:
                        print(f"[Tool Error] {block.name}: {result_str}")

                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result_str,
                        # Signal error to model so it can try a different approach
                        **({"is_error": True} if not success else {})
                    })

            messages.append({"role": "user", "content": tool_results})

        elif response.stop_reason == "end_turn":
            # Extract final text
            for block in response.content:
                if hasattr(block, "text"):
                    return block.text
            return "No response generated"

        else:
            return f"Unexpected stop reason: {response.stop_reason}"

    return f"Agent exceeded {max_turns} turns without completing"

# Test
result = run_agent("What's the temperature in Seoul in Fahrenheit? And what's 15% of that temperature?")
print(result)
```

**Key design decisions:**
- `execute_tool` never raises — it returns a string error that the LLM can interpret and adapt to
- `is_error: True` in the tool result signals to Claude that the tool failed, triggering it to try a different approach
- `max_turns` prevents infinite loops
- Tool registry pattern makes it easy to add/remove tools without changing the agentic loop
</details>

---

## Next Steps

In [15_Multi_Agent_Systems.md](./15_Multi_Agent_Systems.md), we'll learn about multi-agent architectures and orchestration patterns.
