"""
15. LLM Agents Example

ReAct pattern, Tool Use, LangChain Agent practice
"""

import json
import re
from typing import Dict, List, Callable, Any

print("=" * 60)
print("LLM Agents")
print("=" * 60)


# ============================================
# 1. Tool Definition
# ============================================
print("\n[1] Tool Definition")
print("-" * 40)

class Tool:
    """Tool base class"""

    def __init__(self, name: str, description: str, func: Callable):
        self.name = name
        self.description = description
        self.func = func

    def run(self, input_str: str) -> str:
        """Execute the tool"""
        try:
            return str(self.func(input_str))
        except Exception as e:
            return f"Error: {str(e)}"


# Tool implementations
def calculator(expression: str) -> float:
    """Safe math calculation"""
    # Simple safety check
    allowed_chars = set("0123456789+-*/.() ")
    if not all(c in allowed_chars for c in expression):
        raise ValueError("Invalid characters in expression")
    return eval(expression)

def search(query: str) -> str:
    """Search simulation"""
    # In practice, call API
    results = {
        "Python creator": "Python was developed by Guido van Rossum in 1991.",
        "artificial intelligence": "Artificial intelligence (AI) is a computer system that mimics human intelligence.",
        "Seoul population": "Seoul's population is approximately 9.5 million (as of 2024).",
    }
    for key, value in results.items():
        if key in query.lower():
            return value
    return f"No search results found for '{query}'."

def get_weather(city: str) -> str:
    """Weather lookup simulation"""
    weather_data = {
        "Seoul": {"temp": 25, "condition": "Clear"},
        "Busan": {"temp": 28, "condition": "Cloudy"},
        "Jeju": {"temp": 27, "condition": "Partly cloudy"},
    }
    if city in weather_data:
        data = weather_data[city]
        return f"{city} weather: {data['temp']}C, {data['condition']}"
    return f"Weather information for {city} not found."


# Register tools
tools = [
    Tool("calculator", "Math calculation. Input: math expression (e.g., '2 + 3 * 4')", calculator),
    Tool("search", "Information search. Input: search query", search),
    Tool("get_weather", "Weather lookup. Input: city name", get_weather),
]

print("Available tools:")
for tool in tools:
    print(f"  - {tool.name}: {tool.description}")


# ============================================
# 2. ReAct Pattern Simulation
# ============================================
print("\n[2] ReAct Pattern Simulation")
print("-" * 40)

class ReActAgent:
    """ReAct (Reasoning + Acting) agent"""

    def __init__(self, tools: List[Tool]):
        self.tools = {t.name: t for t in tools}
        self.history = []

    def think(self, question: str, observations: List[str]) -> Dict[str, str]:
        """
        Thinking step (in practice, call LLM)
        Here simulated with rule-based approach
        """
        question_lower = question.lower()

        # Rule-based decision making (in practice, LLM performs this)
        if "weather" in question_lower:
            # Extract city
            cities = ["Seoul", "Busan", "Jeju"]
            for city in cities:
                if city.lower() in question_lower:
                    return {
                        "thought": f"I need to check the weather in {city}.",
                        "action": "get_weather",
                        "action_input": city
                    }

        if "calculat" in question_lower or any(op in question for op in ["+", "-", "*", "/"]):
            # Extract expression
            numbers = re.findall(r'[\d\+\-\*\/\.\(\)\s]+', question)
            if numbers:
                expr = numbers[0].strip()
                return {
                    "thought": f"I need to calculate '{expr}'.",
                    "action": "calculator",
                    "action_input": expr
                }

        if any(keyword in question_lower for keyword in ["who", "what", "where", "search"]):
            return {
                "thought": f"I need to search for '{question}'.",
                "action": "search",
                "action_input": question
            }

        # If enough information is gathered, provide final answer
        if observations:
            return {
                "thought": "Sufficient information has been collected.",
                "final_answer": " ".join(observations)
            }

        return {
            "thought": "I couldn't understand the question.",
            "final_answer": "Sorry, I cannot process this question."
        }

    def act(self, action: str, action_input: str) -> str:
        """Action step"""
        if action in self.tools:
            return self.tools[action].run(action_input)
        return f"Error: Unknown tool '{action}'"

    def run(self, question: str, max_steps: int = 5) -> str:
        """Run the agent"""
        observations = []

        print(f"\nQuestion: {question}")
        print("-" * 30)

        for step in range(max_steps):
            # Think
            result = self.think(question, observations)

            print(f"\n[Step {step + 1}]")
            print(f"Thought: {result.get('thought', '')}")

            # Check for final answer
            if "final_answer" in result:
                print(f"Final Answer: {result['final_answer']}")
                return result["final_answer"]

            # Act
            action = result.get("action")
            action_input = result.get("action_input")

            if action:
                print(f"Action: {action}")
                print(f"Action Input: {action_input}")

                observation = self.act(action, action_input)
                observations.append(observation)
                print(f"Observation: {observation}")

        return "Maximum steps reached"


# Test
agent = ReActAgent(tools)

questions = [
    "What's the weather in Seoul?",
    "Calculate 15 + 27 * 3",
    "Who created Python?",
]

for q in questions:
    result = agent.run(q)


# ============================================
# 3. Function Calling Format
# ============================================
print("\n" + "=" * 60)
print("[3] Function Calling Format")
print("-" * 40)

# OpenAI Function Calling format
function_definitions = [
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
            "description": "Search the web for information.",
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

print("Function Calling definitions:")
print(json.dumps(function_definitions, indent=2, ensure_ascii=False))


# ============================================
# 4. Multi-Agent Simulation
# ============================================
print("\n" + "=" * 60)
print("[4] Multi-Agent Simulation")
print("-" * 40)

class ResearcherAgent:
    """Research agent"""

    def __init__(self):
        self.name = "Researcher"

    def research(self, topic: str) -> str:
        """Research a topic (simulation)"""
        research_db = {
            "artificial intelligence": "AI started at the 1956 Dartmouth Conference and has "
                       "evolved into machine learning, deep learning, NLP, and more.",
            "Python": "Python is a programming language developed by Guido van Rossum in 1991, "
                     "characterized by its concise syntax and rich libraries.",
        }
        for key, value in research_db.items():
            if key in topic.lower():
                return value
        return f"No research results found for {topic}."


class WriterAgent:
    """Writing agent"""

    def __init__(self):
        self.name = "Writer"

    def write(self, research_results: str, style: str = "formal") -> str:
        """Write document (simulation)"""
        if style == "formal":
            return f"## Research Report\n\n{research_results}\n\nThis content was written based on research findings."
        else:
            return f"# Summary\n\n{research_results}"


class ReviewerAgent:
    """Review agent"""

    def __init__(self):
        self.name = "Reviewer"

    def review(self, document: str) -> str:
        """Review document (simulation)"""
        issues = []
        if len(document) < 100:
            issues.append("Content is too short.")
        if "references" not in document.lower():
            issues.append("No references provided.")

        if issues:
            return "Review results:\n" + "\n".join(f"- {issue}" for issue in issues)
        return "Review results: No revisions needed"


class MultiAgentSystem:
    """Multi-agent system"""

    def __init__(self):
        self.researcher = ResearcherAgent()
        self.writer = WriterAgent()
        self.reviewer = ReviewerAgent()

    def create_document(self, topic: str) -> str:
        """Document creation pipeline"""
        print(f"\nTopic: {topic}")

        # 1. Research
        print(f"\n[{self.researcher.name}] Researching...")
        research = self.researcher.research(topic)
        print(f"Research results: {research[:50]}...")

        # 2. Write
        print(f"\n[{self.writer.name}] Writing...")
        document = self.writer.write(research)
        print(f"Writing results: {document[:50]}...")

        # 3. Review
        print(f"\n[{self.reviewer.name}] Reviewing...")
        review = self.reviewer.review(document)
        print(f"Review results: {review}")

        return document


# Test
system = MultiAgentSystem()
doc = system.create_document("History of artificial intelligence")
print(f"\nFinal document:\n{doc}")


# ============================================
# 5. LangChain Agent Code Example
# ============================================
print("\n" + "=" * 60)
print("[5] LangChain Agent Code Example")
print("-" * 40)

langchain_code = '''
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import tool

# LLM
llm = ChatOpenAI(model="gpt-4", temperature=0)

# Tool definitions
@tool
def calculator(expression: str) -> str:
    """Math calculation. Input: math expression (e.g., '2 + 3 * 4')"""
    return str(eval(expression))

@tool
def get_current_time() -> str:
    """Return the current time"""
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

tools = [calculator, get_current_time]

# Prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant."),
    MessagesPlaceholder(variable_name="chat_history", optional=True),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# Create agent
agent = create_openai_tools_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Execute
result = agent_executor.invoke({"input": "Tell me the current time and the result of 15 + 27"})
print(result["output"])
'''
print(langchain_code)


# ============================================
# 6. Autonomous Agent (AutoGPT style)
# ============================================
print("\n" + "=" * 60)
print("[6] Autonomous Agent (AutoGPT style)")
print("-" * 40)

class AutoGPTLikeAgent:
    """AutoGPT-style autonomous agent"""

    def __init__(self, tools: List[Tool], goals: List[str]):
        self.tools = {t.name: t for t in tools}
        self.goals = goals
        self.memory = []
        self.completed_tasks = []

    def plan(self) -> Dict[str, Any]:
        """Plan next task (rule-based simulation)"""
        # Check remaining goals
        remaining_goals = [g for g in self.goals if g not in self.completed_tasks]

        if not remaining_goals:
            return {"task": "COMPLETE", "summary": "All goals achieved!"}

        current_goal = remaining_goals[0]

        # Simple rule-based planning
        if "weather" in current_goal.lower():
            return {
                "task": current_goal,
                "tool": "get_weather",
                "input": "Seoul"
            }
        elif "calculat" in current_goal.lower() or "number" in current_goal.lower():
            return {
                "task": current_goal,
                "tool": "calculator",
                "input": "100 + 200"
            }
        else:
            return {
                "task": current_goal,
                "tool": "search",
                "input": current_goal
            }

    def execute(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """Execute plan"""
        if plan.get("task") == "COMPLETE":
            return {"status": "complete", "summary": plan["summary"]}

        tool_name = plan.get("tool")
        tool_input = plan.get("input")

        if tool_name in self.tools:
            result = self.tools[tool_name].run(tool_input)
            return {"status": "success", "result": result}

        return {"status": "error", "message": f"Unknown tool: {tool_name}"}

    def run(self, max_iterations: int = 5) -> str:
        """Run the agent"""
        print(f"\nGoals: {self.goals}")
        print("-" * 30)

        for i in range(max_iterations):
            print(f"\n=== Iteration {i + 1} ===")

            # Plan
            plan = self.plan()
            print(f"Plan: {plan}")

            if plan.get("task") == "COMPLETE":
                print(f"Complete: {plan['summary']}")
                return plan["summary"]

            # Execute
            result = self.execute(plan)
            print(f"Result: {result}")

            # Update memory and completed list
            self.memory.append({"plan": plan, "result": result})
            if result["status"] == "success":
                self.completed_tasks.append(plan["task"])

        return "Maximum iterations reached"


# Test
auto_agent = AutoGPTLikeAgent(
    tools=tools,
    goals=["Check Seoul weather", "Simple calculation"]
)
auto_agent.run()


# ============================================
# Summary
# ============================================
print("\n" + "=" * 60)
print("LLM Agent Summary")
print("=" * 60)

summary = """
LLM Agent Key Concepts:

1. ReAct Pattern:
   Thought -> Action -> Observation -> ... -> Final Answer

2. Tool Definition:
   - Name: Unique identifier
   - Description: For LLM to determine when to use
   - Function: Actual execution logic

3. Function Calling (OpenAI):
   - Define functions via tools parameter
   - tool_choice="auto" for automatic selection
   - Pass results with role="tool"

4. Multi-Agent:
   - Role division (research, writing, review)
   - Pipeline connection
   - Collaboration protocol

5. Autonomous Agent:
   - Goal-based planning
   - Memory persistence
   - Iterative execution

Agent Design Checklist:
- Clear tool descriptions
- Error handling
- Infinite loop prevention (max_steps)
- Safety measures (limit dangerous operations)
- Logging and debugging
"""
print(summary)
