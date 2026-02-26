# 15. LLM 에이전트 (LLM Agents)

## 학습 목표

- 에이전트 개념과 아키텍처 이해
- ReAct 패턴 구현
- 도구 사용 (Tool Use) 기법
- LangChain Agent 활용
- 자율 에이전트 시스템 (AutoGPT 등)

---

## 1. LLM 에이전트 개요

### 에이전트란?

```
┌─────────────────────────────────────────────────────────────┐
│                      LLM 에이전트                            │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─────────────┐                                            │
│  │    LLM      │  ◀── 두뇌 (의사결정)                       │
│  │  (Brain)    │                                            │
│  └──────┬──────┘                                            │
│         │                                                    │
│         ▼                                                    │
│  ┌─────────────┐                                            │
│  │   Planning  │  ◀── 계획 수립                             │
│  └──────┬──────┘                                            │
│         │                                                    │
│         ▼                                                    │
│  ┌─────────────┐    ┌─────────────┐                         │
│  │    Tools    │    │   Memory    │  ◀── 도구 + 기억        │
│  │ (검색, 계산, │    │ (대화 이력, │                         │
│  │  코드실행)   │    │  지식 베이스)│                         │
│  └─────────────┘    └─────────────┘                         │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 에이전트 vs 챗봇

| 항목 | 챗봇 | 에이전트 |
|------|------|----------|
| 응답 방식 | 단일 응답 | 다단계 추론 |
| 도구 사용 | 제한적 | 다양한 도구 |
| 자율성 | 낮음 | 높음 |
| 계획 수립 | 없음 | 있음 |
| 예시 | 고객 지원 봇 | AutoGPT, Copilot |

---

## 2. ReAct (Reasoning + Acting)

### ReAct 패턴

```
Thought: 문제를 분석하고 다음 행동 결정
Action: 도구 선택 및 입력 결정
Observation: 도구 실행 결과
... (반복)
Final Answer: 최종 답변
```

### ReAct 구현

```python
from openai import OpenAI

client = OpenAI()

# 도구 정의
tools = {
    "calculator": lambda expr: eval(expr),
    "search": lambda query: f"검색 결과: {query}에 대한 정보...",
    "get_weather": lambda city: f"{city}의 날씨: 맑음, 25도",
}

def react_agent(question, max_steps=5):
    """ReAct 에이전트"""

    system_prompt = """당신은 문제를 단계별로 해결하는 에이전트입니다.

사용 가능한 도구:
- calculator: 수학 계산 (예: "2 + 3 * 4")
- search: 정보 검색 (예: "파이썬 창시자")
- get_weather: 날씨 조회 (예: "서울")

다음 형식을 따르세요:

Thought: [현재 상황 분석 및 다음 행동 계획]
Action: [도구 이름]
Action Input: [도구 입력]

도구 결과를 받으면:
Observation: [결과]

최종 답변이 준비되면:
Final Answer: [답변]
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

        # Final Answer 체크
        if "Final Answer:" in assistant_message:
            final_answer = assistant_message.split("Final Answer:")[-1].strip()
            return final_answer

        # Action 파싱
        if "Action:" in assistant_message and "Action Input:" in assistant_message:
            action_line = assistant_message.split("Action:")[-1].split("\n")[0].strip()
            input_line = assistant_message.split("Action Input:")[-1].split("\n")[0].strip()

            # 도구 실행
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

    return "최대 단계 도달, 답변 실패"

# 사용
answer = react_agent("서울의 날씨를 확인하고, 기온을 섭씨에서 화씨로 변환해주세요.")
print(f"\n최종 답변: {answer}")
```

---

## 3. 도구 사용 (Tool Use)

### Function Calling (OpenAI)

```python
from openai import OpenAI
import json

client = OpenAI()

# 도구 정의
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "특정 도시의 현재 날씨 정보를 가져옵니다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "도시 이름 (예: Seoul, Tokyo)"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "온도 단위"
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
            "description": "웹에서 정보를 검색합니다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "검색어"
                    }
                },
                "required": ["query"]
            }
        }
    }
]

# 도구 구현
def get_weather(city, unit="celsius"):
    # 실제로는 API 호출
    weather_data = {
        "Seoul": {"temp": 25, "condition": "Sunny"},
        "Tokyo": {"temp": 28, "condition": "Cloudy"},
    }
    data = weather_data.get(city, {"temp": 20, "condition": "Unknown"})
    if unit == "fahrenheit":
        data["temp"] = data["temp"] * 9/5 + 32
    return json.dumps(data)

def search_web(query):
    return json.dumps({"results": f"'{query}'에 대한 검색 결과..."})

tool_implementations = {
    "get_weather": get_weather,
    "search_web": search_web,
}

def agent_with_tools(user_message):
    """Function Calling 에이전트"""
    messages = [{"role": "user", "content": user_message}]

    response = client.chat.completions.create(
        model="gpt-4",
        messages=messages,
        tools=tools,
        tool_choice="auto"  # 자동으로 도구 선택
    )

    assistant_message = response.choices[0].message

    # 도구 호출 필요 여부 확인
    if assistant_message.tool_calls:
        messages.append(assistant_message)

        # 각 도구 호출 처리
        for tool_call in assistant_message.tool_calls:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)

            # 도구 실행
            function_response = tool_implementations[function_name](**function_args)

            # 결과 추가
            messages.append({
                "tool_call_id": tool_call.id,
                "role": "tool",
                "name": function_name,
                "content": function_response
            })

        # 최종 응답
        final_response = client.chat.completions.create(
            model="gpt-4",
            messages=messages
        )
        return final_response.choices[0].message.content

    return assistant_message.content

# 사용
result = agent_with_tools("서울과 도쿄의 날씨를 비교해주세요.")
print(result)
```

### 코드 실행 도구

```python
import subprocess
import tempfile
import os

def execute_python(code):
    """Python 코드 안전하게 실행"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(code)
        temp_path = f.name

    try:
        result = subprocess.run(
            ['python', temp_path],
            capture_output=True,
            text=True,
            timeout=10  # 타임아웃 설정
        )
        output = result.stdout if result.returncode == 0 else result.stderr
        return {"success": result.returncode == 0, "output": output}
    except subprocess.TimeoutExpired:
        return {"success": False, "output": "Timeout"}
    finally:
        os.unlink(temp_path)

# 코드 실행 도구 정의
code_tool = {
    "type": "function",
    "function": {
        "name": "execute_python",
        "description": "Python 코드를 실행합니다.",
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "실행할 Python 코드"
                }
            },
            "required": ["code"]
        }
    }
}
```

## 3.5 도구 사용 심화: 멀티 프로바이더 패턴

현대 LLM은 단순한 텍스트 생성을 넘어 도구 사용(tool use, 함수 호출이라고도 함)을 통해 현실 세계에서 행동을 취할 수 있는 추론 엔진(reasoning engine)으로 진화했습니다. 이 섹션에서는 다양한 프로바이더(provider)의 구현 방식을 종합적으로 비교하고, 견고한 도구 사용 에이전트를 구축하기 위한 고급 패턴을 다룹니다.

### 에이전트 루프(Agentic Loop)

프로바이더에 관계없이 모든 도구 사용 LLM은 동일한 기본 루프를 따릅니다:

```
┌─────────────────────────────────────────────────────────────┐
│                    에이전트 도구 사용 루프                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────┐                                              │
│  │  사용자  │                                              │
│  │  메시지  │                                              │
│  └────┬─────┘                                              │
│       │                                                    │
│       ▼                                                    │
│  ┌──────────┐    tool_use     ┌──────────┐                │
│  │   LLM    │───────────────>│   도구   │                │
│  │  (사고)  │                 │  실행기  │                │
│  │          │<───────────────│          │                │
│  └──────────┘   tool_result   └──────────┘                │
│       │                                                    │
│       │  (LLM이 최종 텍스트 응답을 생성할 때까지 반복)     │
│       │                                                    │
│       ▼                                                    │
│  ┌──────────┐                                              │
│  │  최종    │                                              │
│  │  응답    │                                              │
│  └──────────┘                                              │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Claude 도구 사용(Tool Use) API

Claude는 도구 사용과 도구 결과가 명시적인 메시지 콘텐츠 타입인 구조화된 콘텐츠 블록(content block) 방식을 사용합니다:

```python
import anthropic

client = anthropic.Anthropic()

# 도구 정의는 파라미터에 JSON Schema를 사용
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

# 도구 구현
def execute_tool(name, input_data):
    if name == "get_weather":
        # 프로덕션에서는 실제 날씨 API 호출
        return f'{{"temp": 22, "condition": "Sunny", "humidity": 45}}'
    elif name == "calculate":
        try:
            return str(eval(input_data["expression"]))
        except Exception as e:
            return f"Error: {e}"
    return "Unknown tool"

# 에이전트 루프: 더 이상 도구 사용이 없을 때까지 반복 호출
messages = [{"role": "user", "content": "What's the weather in Tokyo? Also, what's 15% of 340?"}]

while True:
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        tools=tools,
        messages=messages
    )

    # 어시스턴트 응답 수집
    messages.append({"role": "assistant", "content": response.content})

    # 도구 실행이 필요한지 확인
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
        # 최종 텍스트 응답 -- 루프 종료
        for block in response.content:
            if hasattr(block, "text"):
                print(block.text)
        break
```

### 프로바이더 비교: 도구 사용(Tool Use) API

| 특성 | OpenAI | Claude (Anthropic) | Gemini (Google) |
|------|--------|-------------------|-----------------|
| **파라미터 이름** | `tools` | `tools` | `tools` |
| **스키마 형식** | `parameters`에 JSON Schema | `input_schema`에 JSON Schema | OpenAPI 스타일 스키마 |
| **도구 호출 신호** | `finish_reason: "tool_calls"` | `stop_reason: "tool_use"` | `finish_reason: "TOOL_CALL"` |
| **호출 형식** | `tool_calls[].function` | 콘텐츠 블록 `type: "tool_use"` | candidate의 `function_call` |
| **결과 형식** | `role: "tool"` 메시지 | `tool_result` 콘텐츠 블록 | `function_response` 파트 |
| **다중 도구** | 한 응답에서 병렬 호출 | 한 응답에서 병렬 호출 | 순차적 (한 번에 하나) |
| **스트리밍** | 지원 | 지원 | 지원 |
| **도구 선택** | `tool_choice: "auto"/"required"/"none"` | `tool_choice: {"type": "auto"/"any"/"tool"}` | `tool_config` |

### 도구 정의 모범 사례

```python
# 좋음: 예시와 엣지 케이스를 포함한 상세한 설명
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

# 나쁨: 모호한 설명, 사용 시점 안내 없음
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

### ReAct 패턴 (Reasoning + Acting)

ReAct 패턴은 사고(thought)와 행동(action) 단계를 번갈아 수행하여 LLM의 추론 과정을 명시적으로 만듭니다:

```
User: "프랑스의 인구를 찾고, GDP가 2.78조 달러일 때
       1인당 GDP를 계산해주세요."

Step 1:
  Thought: 먼저 프랑스 인구를 찾아야 합니다. 검색하겠습니다.
  Action: search("France population 2024")
  Observation: 프랑스 인구는 약 6,817만 명 (2024년)

Step 2:
  Thought: 인구(6,817만)와 GDP(2.78조 달러)를 확보했습니다.
           1인당 GDP = GDP / 인구를 계산해야 합니다.
  Action: calculate("2780000000000 / 68170000")
  Observation: 40779.17

Step 3:
  Thought: 필요한 모든 정보를 확보했습니다.
  Final Answer: 프랑스 인구는 약 6,817만 명입니다.
                GDP 2.78조 달러 기준, 1인당 GDP는
                약 $40,779입니다.
```

### 도구 사용 안전성(Tool Use Safety)

```python
# 1. 실행 전 파라미터 검증
def safe_execute(tool_name, params, allowed_tools):
    """안전 검사와 함께 도구 실행."""
    # 허용 목록 검사
    if tool_name not in allowed_tools:
        return {"error": f"Tool '{tool_name}' is not permitted"}

    # 스키마 검증
    from jsonschema import validate, ValidationError
    try:
        validate(instance=params, schema=allowed_tools[tool_name]["input_schema"])
    except ValidationError as e:
        return {"error": f"Invalid parameters: {e.message}"}

    # 속도 제한
    if rate_limiter.is_exceeded(tool_name):
        return {"error": "Rate limit exceeded for this tool"}

    # 샌드박스에서 실행 (타임아웃, 리소스 제한)
    try:
        result = execute_with_timeout(tool_name, params, timeout=30)
        return {"result": result}
    except TimeoutError:
        return {"error": "Tool execution timed out"}

# 2. 비정제 LLM 출력을 시스템 명령에 절대 전달하지 않음
# 나쁨: os.system(llm_generated_command)
# 좋음: 검증된 입력으로 파라미터화된 API 사용

# 3. 감사를 위해 모든 도구 호출 로깅
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

### 기본 에이전트

```python
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import Tool, tool
from langchain_community.tools import DuckDuckGoSearchRun

# LLM
llm = ChatOpenAI(model="gpt-4", temperature=0)

# 도구 정의
search = DuckDuckGoSearchRun()

@tool
def calculator(expression: str) -> str:
    """수학 계산을 수행합니다. 입력: 수학 표현식 (예: '2 + 3 * 4')"""
    try:
        return str(eval(expression))
    except:
        return "계산 오류"

@tool
def get_current_time() -> str:
    """현재 시간을 반환합니다."""
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

tools = [
    Tool(name="Search", func=search.run, description="웹 검색"),
    calculator,
    get_current_time,
]

# 프롬프트
prompt = ChatPromptTemplate.from_messages([
    ("system", "당신은 도움이 되는 AI 어시스턴트입니다. 도구를 사용하여 질문에 답하세요."),
    MessagesPlaceholder(variable_name="chat_history", optional=True),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# 에이전트 생성
agent = create_openai_tools_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# 실행
result = agent_executor.invoke({"input": "현재 시간과 오늘의 주요 뉴스를 알려주세요."})
print(result["output"])
```

### ReAct Agent (LangChain)

```python
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4", temperature=0)

# ReAct 프롬프트
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

# 에이전트 생성
react_agent = create_react_agent(llm, tools, react_prompt)
agent_executor = AgentExecutor(agent=react_agent, tools=tools, verbose=True, handle_parsing_errors=True)

# 실행
result = agent_executor.invoke({"input": "2024년 미국 대통령 선거 결과를 검색하고 요약해주세요."})
```

### 메모리가 있는 에이전트

```python
from langchain.memory import ConversationBufferMemory
from langchain.agents import AgentExecutor, create_openai_tools_agent

# 메모리
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# 프롬프트 (메모리 포함)
prompt = ChatPromptTemplate.from_messages([
    ("system", "당신은 도움이 되는 AI 어시스턴트입니다."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# 에이전트
agent = create_openai_tools_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, memory=memory, verbose=True)

# 대화
agent_executor.invoke({"input": "내 이름은 김철수야."})
agent_executor.invoke({"input": "내 이름이 뭐라고 했지?"})
```

---

## 5. 자율 에이전트 시스템

### Plan-and-Execute

```python
from langchain.experimental.plan_and_execute import (
    PlanAndExecute,
    load_agent_executor,
    load_chat_planner
)
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4", temperature=0)

# Planner와 Executor 생성
planner = load_chat_planner(llm)
executor = load_agent_executor(llm, tools, verbose=True)

# Plan-and-Execute 에이전트
agent = PlanAndExecute(planner=planner, executor=executor, verbose=True)

# 복잡한 작업 실행
result = agent.run("파이썬의 역사에 대해 조사하고, 주요 버전별 특징을 요약한 마크다운 문서를 작성해주세요.")
```

### AutoGPT 스타일 에이전트

```python
class AutoGPTAgent:
    """자율 에이전트"""

    def __init__(self, llm, tools, goals):
        self.llm = llm
        self.tools = {t.name: t for t in tools}
        self.goals = goals
        self.memory = []
        self.completed_tasks = []

    def plan(self):
        """목표 달성을 위한 계획 수립"""
        prompt = f"""당신은 자율 AI 에이전트입니다.

목표: {self.goals}

완료된 작업:
{self.completed_tasks}

이전 작업 결과:
{self.memory[-5:] if self.memory else "없음"}

사용 가능한 도구:
{list(self.tools.keys())}

다음 작업을 JSON 형식으로 출력하세요:
{{"task": "작업 설명", "tool": "사용할 도구", "input": "도구 입력"}}

모든 목표가 달성되었다면:
{{"task": "COMPLETE", "summary": "결과 요약"}}
"""
        response = self.llm.invoke(prompt)
        return json.loads(response.content)

    def execute(self, task):
        """작업 실행"""
        if task["task"] == "COMPLETE":
            return {"status": "complete", "summary": task["summary"]}

        tool = self.tools.get(task["tool"])
        if tool:
            result = tool.run(task["input"])
            return {"status": "success", "result": result}
        return {"status": "error", "message": f"Unknown tool: {task['tool']}"}

    def run(self, max_iterations=10):
        """에이전트 실행"""
        for i in range(max_iterations):
            print(f"\n=== Iteration {i+1} ===")

            # 계획
            task = self.plan()
            print(f"Task: {task}")

            # 완료 확인
            if task.get("task") == "COMPLETE":
                print(f"Goals achieved: {task['summary']}")
                return task["summary"]

            # 실행
            result = self.execute(task)
            print(f"Result: {result}")

            # 메모리 업데이트
            self.memory.append({"task": task, "result": result})
            if result["status"] == "success":
                self.completed_tasks.append(task["task"])

        return "Max iterations reached"

# 사용
agent = AutoGPTAgent(
    llm=ChatOpenAI(model="gpt-4"),
    tools=tools,
    goals=["서울의 인구 조사", "인구 통계 분석", "보고서 작성"]
)
result = agent.run()
```

---

## 6. 멀티 에이전트 시스템

### 에이전트 간 협업

```python
class ResearcherAgent:
    """연구 에이전트"""
    def __init__(self, llm):
        self.llm = llm

    def research(self, topic):
        prompt = f"'{topic}'에 대해 조사하고 핵심 정보를 정리해주세요."
        return self.llm.invoke(prompt).content

class WriterAgent:
    """작문 에이전트"""
    def __init__(self, llm):
        self.llm = llm

    def write(self, research_results, style="formal"):
        prompt = f"다음 정보를 바탕으로 {style} 스타일의 문서를 작성해주세요:\n{research_results}"
        return self.llm.invoke(prompt).content

class ReviewerAgent:
    """검토 에이전트"""
    def __init__(self, llm):
        self.llm = llm

    def review(self, document):
        prompt = f"다음 문서를 검토하고 개선점을 제안해주세요:\n{document}"
        return self.llm.invoke(prompt).content

class MultiAgentSystem:
    """멀티 에이전트 시스템"""

    def __init__(self, llm):
        self.researcher = ResearcherAgent(llm)
        self.writer = WriterAgent(llm)
        self.reviewer = ReviewerAgent(llm)

    def create_document(self, topic, max_revisions=2):
        # 1. 연구
        print("=== 연구 단계 ===")
        research = self.researcher.research(topic)
        print(research[:200] + "...")

        # 2. 작성
        print("\n=== 작성 단계 ===")
        document = self.writer.write(research)
        print(document[:200] + "...")

        # 3. 검토 및 수정
        for i in range(max_revisions):
            print(f"\n=== 검토 {i+1} ===")
            review = self.reviewer.review(document)
            print(review[:200] + "...")

            # 수정
            if "수정 필요 없음" in review:
                break
            document = self.writer.write(f"원본:\n{document}\n\n검토:\n{review}", style="revised")

        return document

# 사용
llm = ChatOpenAI(model="gpt-4")
system = MultiAgentSystem(llm)
final_doc = system.create_document("인공지능의 미래")
```

---

## 7. 에이전트 평가

### 도구 선택 정확도

```python
def evaluate_tool_selection(agent, test_cases):
    """도구 선택 정확도 평가"""
    correct = 0
    total = len(test_cases)

    for case in test_cases:
        query = case["query"]
        expected_tool = case["expected_tool"]

        # 에이전트 실행 (도구 선택만)
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

# 테스트 케이스
test_cases = [
    {"query": "2 + 3 * 4를 계산해줘", "expected_tool": "calculator"},
    {"query": "오늘 서울 날씨 어때?", "expected_tool": "get_weather"},
    {"query": "파이썬 창시자가 누구야?", "expected_tool": "search"},
]

# 평가
evaluate_tool_selection(agent, test_cases)
```

### 작업 완료율

```python
def evaluate_task_completion(agent, tasks):
    """작업 완료율 평가"""
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

# 작업 정의
tasks = [
    {
        "description": "날씨 조회 및 옷차림 추천",
        "input": "서울 날씨를 확인하고 오늘 옷차림을 추천해줘",
        "validator": lambda r: "서울" in r and ("옷" in r or "의류" in r)
    },
    {
        "description": "수학 계산",
        "input": "123 * 456의 결과는?",
        "validator": lambda r: "56088" in r
    },
]
```

---

## 정리

### 에이전트 아키텍처 비교

| 아키텍처 | 특징 | 사용 시점 |
|----------|------|----------|
| ReAct | 추론-행동 반복 | 단계별 문제 해결 |
| Function Calling | 구조화된 도구 호출 | API 연동 |
| Plan-and-Execute | 계획 후 실행 | 복잡한 작업 |
| AutoGPT | 자율 목표 달성 | 장기 작업 |
| Multi-Agent | 역할 분담 협업 | 전문성 필요 |

### 핵심 코드

```python
# ReAct 패턴
Thought: 문제 분석
Action: 도구 선택
Observation: 결과 확인
Final Answer: 최종 답변

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

### 에이전트 설계 체크리스트

```
□ 명확한 도구 정의 (이름, 설명, 파라미터)
□ 에러 처리 (도구 실패, 파싱 오류)
□ 메모리 관리 (대화 이력, 컨텍스트)
□ 루프 방지 (최대 반복 횟수)
□ 안전 장치 (위험한 작업 제한)
□ 로깅 및 모니터링
```

---

## 연습 문제

### 연습 문제 1: 도구 정의 품질

아래는 데이터베이스 조회 함수에 대한 두 가지 도구 정의입니다. 최소 네 가지 차이점을 파악하고, 두 번째 정의가 더 나은 이유를 설명하고, 오류 처리 안내도 추가한 개선된 버전을 작성하세요.

```python
# 도구 A (품질 낮음)
tool_a = {
    "name": "db_lookup",
    "description": "데이터베이스에서 데이터 조회",
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {"type": "string"}
        },
        "required": ["query"]
    }
}

# 도구 B (더 나은 품질)
tool_b = {
    "name": "lookup_customer",
    "description": (
        "ID 또는 이메일 주소로 고객 정보를 조회합니다. "
        "고객 이름, 이메일, 계정 상태, 주문 이력을 반환합니다. "
        "고객 신원 확인이나 계정 정보 확인 시 사용하세요. "
        "상품 검색에는 사용하지 마세요 — search_products를 대신 사용하세요."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "identifier": {
                "type": "string",
                "description": "고객 ID (예: 'CUST-12345') 또는 이메일 주소"
            },
            "include_orders": {
                "type": "boolean",
                "description": "주문 이력 포함 여부 (기본값: false)"
            }
        },
        "required": ["identifier"]
    }
}
```

<details>
<summary>정답 보기</summary>

**차이점과 도구 B가 더 나은 이유:**

1. **이름 구체성**: `db_lookup` vs `lookup_customer` — B는 자기 설명적. LLM이 설명 없이도 어떤 데이터를 반환하는지 알 수 있음
2. **설명 품질**: A는 "데이터 조회"(모호). B는 반환되는 데이터, 사용 시기, 사용하지 말아야 할 경우를 설명 — 잘못된 도구 사용 방지
3. **파라미터(parameter) 이름**: `query`는 일반적. `identifier`는 LLM에게 입력이 ID나 이메일이어야 한다고 알려줌
4. **파라미터 설명**: A는 필드 설명 없음. B는 정확한 형식(`CUST-12345`)과 대안(이메일) 설명
5. **선택적 파라미터(optional parameter)**: B는 선택적 `include_orders` 플래그와 기본 동작 문서화

**오류 처리 안내가 추가된 개선 버전:**
```python
tool_best = {
    "name": "lookup_customer",
    "description": (
        "고객 ID 또는 이메일로 고객 계정 정보를 조회합니다. "
        "반환 항목: customer_id, 이름, 이메일, 상태('active'|'suspended'|'closed'), "
        "가입일, 선택적으로 주문 이력(최근 10건). "
        "고객 신원 확인, 계정 상태 확인, 구매 이력 조회에 사용하세요. "
        "사용하지 말 것: 상품 검색(search_products 사용), "
        "주문 추적(track_order 사용), 대량 조회. "
        "고객을 찾을 수 없으면 null 반환 — 오류로 처리하지 마세요."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "identifier": {
                "type": "string",
                "description": (
                    "고객 ID (형식: 'CUST-XXXXX') 또는 이메일 주소. "
                    "이메일은 대소문자 구분 없음. "
                    "예시: 'CUST-12345' 또는 'john.doe@example.com'"
                )
            },
            "include_orders": {
                "type": "boolean",
                "description": (
                    "true이면 최근 10건의 주문 포함. "
                    "응답 크기가 증가 — 주문 이력이 필요한 경우만 true로 설정. "
                    "기본값: false"
                )
            }
        },
        "required": ["identifier"],
        "additionalProperties": False
    }
}
```

**핵심 원칙:** 도구 설명은 LLM의 결정을 명확하게 만들어야 합니다. LLM에게 필요한 것: 도구의 기능, 사용 시기, 사용하지 말아야 할 경우, 출력 형식, 오류/null 결과 해석 방법.
</details>

---

### 연습 문제 2: ReAct 추적(Trace) 분석

아래는 결함이 있는 ReAct 에이전트 추적입니다. 모든 문제점을 파악하고 올바른 추적을 다시 작성하세요.

```
질문: 프랑스 GDP(약 2조 7,800억 달러)의 15%는 얼마인가?

단계 1:
Thought: 2조 7,800억 달러의 15%를 계산해야 한다.
Action: search
Action Input: 프랑스 GDP 15퍼센트

단계 2:
Observation: 프랑스 GDP는 2조 7,800억 달러. GDP의 15%는 국방에 쓰인다.
Thought: 답은 국방비다.
Final Answer: 프랑스는 GDP의 15%를 국방에 지출한다.
```

<details>
<summary>정답 보기</summary>

**파악된 문제점:**

1. **산술에 잘못된 도구 사용**: 에이전트가 미리 존재하는 수학 문제 답을 찾기 위해 `search`를 사용했습니다. 이는 불안정하며 환각(hallucination) 위험을 도입합니다.
2. **질문 오해**: 질문은 2조 7,800억 달러의 15%를 *계산*하도록 요청합니다. 에이전트는 "GDP의 15%"를 검색하여 다른 사실(국방비)을 찾았습니다.
3. **상관관계와 질문 혼동**: 검색 결과에 "GDP의 15%"가 언급된다고 해서 질문에 대한 답은 아닙니다.
4. **잘못된 최종 답**: 2조 7,800억 × 15% = 4,170억 달러이지, "국방비"가 아닙니다.

**수정된 추적:**
```
질문: 프랑스 GDP(약 2조 7,800억 달러)의 15%는 얼마인가?

단계 1:
Thought: 질문에 프랑스 GDP(2조 7,800억 달러)가 직접 제공되어 있고
         15%를 계산하도록 요청받았습니다. 검색이 필요 없습니다 —
         2조 7,800억 × 0.15를 계산하면 됩니다.
Action: calculator
Action Input: 2780000000000 * 0.15

단계 2:
Observation: 417000000000

Thought: 계산이 완료되었습니다. 2조 7,800억(2,780,000,000,000)의 15%는
         417,000,000,000으로, 4,170억 달러입니다.
Final Answer: 프랑스 GDP 2조 7,800억 달러의 15%는 약 4,170억 달러입니다.
```

**교훈:**
- 질문 자체에 필요한 데이터가 있으면 검색 대신 계산 도구를 사용하세요
- 관찰(observation)이 질문에 직접 답하는지 확인하세요
- 최종 답에 단위(억, 조)를 포함하여 해석 가능하게 만드세요
</details>

---

### 연습 문제 3: 오류 처리가 있는 다중 도구 에이전트

Claude 도구 사용 API를 활용하여 세 가지 경우를 처리하는 강건한 에이전트 루프를 구현하세요: 성공적인 도구 실행, 도구 오류, 알 수 없는 도구. 에이전트는 오류에서 우아하게 복구하고 질문에 계속 답하려고 해야 합니다.

<details>
<summary>정답 보기</summary>

```python
import anthropic
import json
from typing import Any

client = anthropic.Anthropic()

# 도구 구현
def get_weather(location: str, unit: str = "celsius") -> dict:
    """시뮬레이션된 날씨 API."""
    weather_db = {
        "서울": {"temp_c": 15, "condition": "부분 흐림", "humidity": 65},
        "도쿄": {"temp_c": 22, "condition": "맑음", "humidity": 40},
    }
    data = weather_db.get(location)
    if data is None:
        raise ValueError(f"'{location}' 위치를 날씨 데이터베이스에서 찾을 수 없습니다")

    temp = data["temp_c"] if unit == "celsius" else data["temp_c"] * 9/5 + 32
    return {"location": location, "temperature": temp, "unit": unit,
            "condition": data["condition"], "humidity": data["humidity"]}

def calculate(expression: str) -> float:
    """안전한 수학 계산기."""
    # 안전한 수학 연산만 허용
    allowed_chars = set("0123456789+-*/.() ")
    if not all(c in allowed_chars for c in expression):
        raise ValueError(f"안전하지 않은 표현식: 기본 수학 연산만 허용됩니다")
    return eval(expression)

# 도구 레지스트리(registry)
TOOLS = {
    "get_weather": get_weather,
    "calculate": calculate,
}

TOOL_SCHEMAS = [
    {
        "name": "get_weather",
        "description": "도시의 현재 날씨를 가져옵니다. 온도, 날씨 상태, 습도를 반환합니다.",
        "input_schema": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "도시 이름"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
            },
            "required": ["location"]
        }
    },
    {
        "name": "calculate",
        "description": "수학 표현식을 계산합니다. 기본 산술(+, -, *, /, 괄호)만 지원합니다.",
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "수학 표현식"}
            },
            "required": ["expression"]
        }
    }
]

def execute_tool(name: str, params: dict) -> tuple[str, bool]:
    """
    도구를 실행하고 (결과_문자열, 성공 여부)를 반환합니다.
    절대 예외를 발생시키지 않음 — 오류는 LLM이 적응할 수 있도록 문자열로 반환됩니다.
    """
    if name not in TOOLS:
        return f"오류: 알 수 없는 도구 '{name}'. 사용 가능한 도구: {list(TOOLS.keys())}", False

    try:
        result = TOOLS[name](**params)
        return json.dumps(result, ensure_ascii=False) if isinstance(result, dict) else str(result), True
    except Exception as e:
        return f"{name} 실행 오류: {str(e)}", False

def run_agent(user_message: str, max_turns: int = 10) -> str:
    """
    오류 복구 기능이 있는 강건한 에이전트 루프.
    모델의 최종 텍스트 응답을 반환합니다.
    """
    messages = [{"role": "user", "content": user_message}]

    for turn in range(max_turns):
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            tools=TOOL_SCHEMAS,
            messages=messages
        )

        # 어시스턴트 응답을 이력에 추가
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "tool_use":
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    result_str, success = execute_tool(block.name, block.input)

                    if not success:
                        print(f"[도구 오류] {block.name}: {result_str}")

                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result_str,
                        # 오류를 모델에 신호하여 다른 방법을 시도하도록 함
                        **({"is_error": True} if not success else {})
                    })

            messages.append({"role": "user", "content": tool_results})

        elif response.stop_reason == "end_turn":
            # 최종 텍스트 추출
            for block in response.content:
                if hasattr(block, "text"):
                    return block.text
            return "응답이 생성되지 않았습니다"

        else:
            return f"예상치 못한 종료 이유: {response.stop_reason}"

    return f"에이전트가 {max_turns}번의 턴을 초과했습니다"

# 테스트
result = run_agent("서울의 온도는 화씨로 얼마인가요? 그 온도의 15%는 얼마인가요?")
print(result)
```

**주요 설계 결정:**
- `execute_tool`은 절대 예외를 발생시키지 않음 — LLM이 해석하고 적응할 수 있는 문자열 오류를 반환
- 도구 결과의 `is_error: True`가 Claude에 도구 실패를 알려 다른 방법 시도 유도
- `max_turns`로 무한 루프 방지
- 도구 레지스트리(registry) 패턴으로 에이전트 루프를 변경하지 않고도 도구를 쉽게 추가/제거 가능
</details>

---

## 다음 단계

[LLM 평가 지표 (Evaluation Metrics)](./16_Evaluation_Metrics.md)에서 LLM 평가 지표와 벤치마크를 학습합니다.
