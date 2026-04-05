# 비전 에이전트

**이전**: [22. 트러블슈팅과 디버깅](./22_Troubleshooting.md) | **다음**: [24. 프롬프트 캐싱과 Batch API](./24_Prompt_Caching_and_Batch_API.md)

---

Claude의 비전(Vision) 기능은 텍스트 전용 어시스턴트를 시각 정보를 보고, 해석하고, 행동할 수 있는 멀티모달 에이전트로 변화시킵니다. 이 레슨에서는 단일 이미지 전송부터 이미지 이해를 도구 사용, 컴퓨터 제어, MCP 통합과 결합하는 프로덕션급 비전 기반 에이전트 구축까지 모든 것을 다룹니다.

**난이도**: ⭐⭐⭐

**사전 요구 사항**:
- Claude API 기초 ([레슨 15](./15_Claude_API_Fundamentals.md))
- 도구 사용과 함수 호출(Tool Use & Function Calling) ([레슨 16](./16_Tool_Use_and_Function_Calling.md))
- 커스텀 에이전트 구축 ([레슨 18](./18_Building_Custom_Agents.md))
- Model Context Protocol 기본 ([레슨 12](./12_Model_Context_Protocol.md))

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. base64 인코딩과 URL을 통해 Claude에 이미지 전송
2. 단일 요청에서 여러 이미지를 분석하고 비교
3. 문서, 스크린샷, 다이어그램에서 구조화된 데이터 추출
4. 시각과 행동을 결합하는 비전 기반 에이전트 구축
5. UI 자동화를 위한 컴퓨터 사용(Computer Use) 구현
6. 비전 기능과 MCP 서버 통합
7. 비전 중심 워크로드의 비용 최적화

---

## 목차

1. [Claude의 비전 기능 개요](#1-claude의-비전-기능-개요)
2. [Messages API를 통한 이미지 전송](#2-messages-api를-통한-이미지-전송)
3. [다중 이미지 분석과 비교](#3-다중-이미지-분석과-비교)
4. [문서 이해](#4-문서-이해)
5. [비전 기반 에이전트 구축](#5-비전-기반-에이전트-구축)
6. [이미지 기반 도구 사용 패턴](#6-이미지-기반-도구-사용-패턴)
7. [컴퓨터 사용과 UI 자동화](#7-컴퓨터-사용과-ui-자동화)
8. [비전 + MCP 서버 통합](#8-비전--mcp-서버-통합)
9. [비전 워크로드의 비용 최적화](#9-비전-워크로드의-비용-최적화)
10. [연습 문제](#10-연습-문제)

---

## 1. Claude의 비전 기능 개요

Claude는 멀티모달 입력의 일부로 이미지를 기본적으로 처리할 수 있습니다. OCR 후 텍스트 변환 파이프라인과 달리, Claude는 레이아웃, 공간적 관계, 차트, 손글씨 등을 이해하면서 시각적 콘텐츠를 직접 인식합니다.

### 1.1 지원 형식과 제한 사항

| 속성 | 세부 사항 |
|---|---|
| **지원 형식** | JPEG, PNG, GIF, WebP |
| **최대 이미지 크기** | 이미지당 5 MB |
| **최대 해상도** | 가장 긴 변 기준 ~1568 px (자동 리사이즈) |
| **요청당 이미지 수** | 최대 20개 |
| **토큰 비용** | 해상도에 따라 다름 (9장 참조) |

### 1.2 Claude가 볼 수 있는 것

Claude가 잘하는 분야:
- **텍스트 추출**: 인쇄된 텍스트, 손글씨, 스크린샷의 코드
- **차트 해석**: 막대 그래프, 선 그래프, 원형 차트, 산점도
- **다이어그램 이해**: 아키텍처 다이어그램, 플로차트, UML
- **사진 분석**: 객체 인식, 장면 설명, 공간 추론
- **UI 이해**: 버튼 레이블, 폼 필드, 네비게이션 요소
- **비교**: 나란히 놓은 이미지 비교 및 차이점 감지

### 1.3 알려진 한계

- **공간 정밀도**: Claude는 정확한 픽셀 좌표에 어려움을 겪을 수 있습니다
- **작은 텍스트**: 매우 작거나 낮은 대비의 텍스트를 놓칠 수 있습니다
- **계수(Counting)**: 많은 수의 객체를 세는 것은 신뢰하기 어렵습니다
- **회전**: 심하게 회전되거나 뒤집힌 텍스트는 읽기 어렵습니다
- **의료/전문 이미지**: 임상 진단용으로 설계되지 않았습니다

---

## 2. Messages API를 통한 이미지 전송

### 2.1 Base64 인코딩

로컬 이미지에 가장 일반적인 접근 방식:

```python
import anthropic
import base64
from pathlib import Path


def encode_image(image_path: str) -> tuple[str, str]:
    """Encode an image file to base64 and detect its media type."""
    path = Path(image_path)
    suffix = path.suffix.lower()
    media_types = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }
    media_type = media_types.get(suffix, "image/jpeg")

    with open(path, "rb") as f:
        data = base64.standard_b64encode(f.read()).decode("utf-8")

    return data, media_type


client = anthropic.Anthropic()

image_data, media_type = encode_image("screenshot.png")

message = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": image_data,
                    },
                },
                {
                    "type": "text",
                    "text": "Describe what you see in this image.",
                },
            ],
        }
    ],
)

print(message.content[0].text)
```

### 2.2 URL 기반 이미지

공개적으로 접근 가능한 이미지의 경우 URL을 직접 전달할 수 있습니다:

```python
message = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "url",
                        "url": "https://example.com/chart.png",
                    },
                },
                {
                    "type": "text",
                    "text": "Extract the data from this bar chart as JSON.",
                },
            ],
        }
    ],
)
```

### 2.3 이미지 배치 모범 사례

- 이미지를 참조하는 텍스트 프롬프트 **앞에** 이미지를 배치하세요
- 여러 이미지의 경우 논리적으로 정렬하세요 (왼쪽에서 오른쪽, 시간 순서)
- 위치로 이미지를 참조하세요: "첫 번째 이미지", "왼쪽의 차트"
- 텍스트 프롬프트는 구체적으로: "이 스크린샷에서 모든 이메일 주소를 추출하세요"가 "이 이미지에 뭐가 있나요?"보다 좋습니다

---

## 3. 다중 이미지 분석과 비교

Claude는 단일 요청에서 최대 20개의 이미지를 처리할 수 있어, 강력한 비교 및 집계 워크플로우를 가능하게 합니다.

### 3.1 두 이미지 비교

```python
def compare_images(image_path_1: str, image_path_2: str, prompt: str) -> str:
    """Compare two images and return Claude's analysis."""
    data_1, mt_1 = encode_image(image_path_1)
    data_2, mt_2 = encode_image(image_path_2)

    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2048,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {"type": "base64", "media_type": mt_1, "data": data_1},
                    },
                    {
                        "type": "image",
                        "source": {"type": "base64", "media_type": mt_2, "data": data_2},
                    },
                    {
                        "type": "text",
                        "text": prompt,
                    },
                ],
            }
        ],
    )
    return message.content[0].text


# 예시: 전후 UI 디자인 비교
result = compare_images(
    "design_v1.png",
    "design_v2.png",
    "Compare these two UI designs. List all visual differences "
    "including layout changes, color modifications, and element additions or removals.",
)
```

### 3.2 배치 이미지 처리

```python
import glob


def analyze_image_batch(
    image_paths: list[str],
    system_prompt: str,
    user_prompt: str,
) -> str:
    """Process a batch of images with a shared analysis prompt."""
    content = []
    for path in image_paths:
        data, media_type = encode_image(path)
        content.append({
            "type": "image",
            "source": {"type": "base64", "media_type": media_type, "data": data},
        })

    content.append({"type": "text", "text": user_prompt})

    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4096,
        system=system_prompt,
        messages=[{"role": "user", "content": content}],
    )
    return message.content[0].text


# 예시: 제품 스크린샷 세트 분석
screenshots = sorted(glob.glob("screenshots/*.png"))[:20]  # 최대 20개
result = analyze_image_batch(
    screenshots,
    system_prompt="You are a UX auditor. Evaluate each screenshot for accessibility issues.",
    user_prompt="Review each screenshot and provide a numbered list of accessibility issues found.",
)
```

---

## 4. 문서 이해

Claude는 문서, 이미지로 렌더링된 PDF, 기술 다이어그램에서 구조화된 정보를 추출하는 데 뛰어납니다.

### 4.1 PDF 페이지 분석

```python
import fitz  # PyMuPDF


def pdf_pages_to_images(pdf_path: str, dpi: int = 150) -> list[tuple[str, str]]:
    """Convert PDF pages to base64-encoded PNG images."""
    doc = fitz.open(pdf_path)
    images = []
    for page in doc:
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        pix = page.get_pixmap(matrix=mat)
        png_bytes = pix.tobytes("png")
        b64 = base64.standard_b64encode(png_bytes).decode("utf-8")
        images.append((b64, "image/png"))
    doc.close()
    return images


def extract_from_pdf(pdf_path: str, prompt: str) -> str:
    """Extract structured data from a PDF using vision."""
    pages = pdf_pages_to_images(pdf_path)

    # 20페이지 단위로 배치 처리
    all_results = []
    for i in range(0, len(pages), 20):
        batch = pages[i : i + 20]
        content = []
        for b64_data, media_type in batch:
            content.append({
                "type": "image",
                "source": {"type": "base64", "media_type": media_type, "data": b64_data},
            })
        content.append({
            "type": "text",
            "text": f"Pages {i+1}-{i+len(batch)}: {prompt}",
        })

        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            messages=[{"role": "user", "content": content}],
        )
        all_results.append(message.content[0].text)

    return "\n\n---\n\n".join(all_results)


# 예시: 청구서 PDF에서 테이블 추출
result = extract_from_pdf(
    "invoice.pdf",
    "Extract all line items as a JSON array with fields: description, quantity, unit_price, total.",
)
```

### 4.2 아키텍처 다이어그램 해석

```python
def interpret_diagram(image_path: str) -> str:
    """Convert an architecture diagram to a structured description."""
    data, media_type = encode_image(image_path)

    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4096,
        system=(
            "You are a software architect. When given a diagram, produce:\n"
            "1. A list of all components/services shown\n"
            "2. All connections between components with protocols/arrows\n"
            "3. A Mermaid diagram that reproduces the architecture\n"
            "4. Any potential issues or improvements you notice"
        ),
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {"type": "base64", "media_type": media_type, "data": data},
                    },
                    {
                        "type": "text",
                        "text": "Analyze this architecture diagram.",
                    },
                ],
            }
        ],
    )
    return message.content[0].text
```

### 4.3 스크린샷 데이터 추출

```python
import json


def extract_table_from_screenshot(image_path: str) -> list[dict]:
    """Extract tabular data from a screenshot into structured JSON."""
    data, media_type = encode_image(image_path)

    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4096,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {"type": "base64", "media_type": media_type, "data": data},
                    },
                    {
                        "type": "text",
                        "text": (
                            "Extract all data from the table in this screenshot. "
                            "Return ONLY a JSON array of objects where each object "
                            "represents a row and keys are column headers. "
                            "No markdown fences, just raw JSON."
                        ),
                    },
                ],
            }
        ],
    )

    return json.loads(message.content[0].text)
```

---

## 5. 비전 기반 에이전트 구축

비전 기반 에이전트는 Claude의 시각과 도구 사용을 결합하여, 환경을 관찰하고 결정을 내리고 행동을 취할 수 있게 합니다.

### 5.1 관찰-사고-행동 루프(See-Think-Act Loop)

```python
import anthropic
import json


class VisionAgent:
    """An agent that can see images and use tools to act on what it sees."""

    def __init__(self, tools: list[dict], system_prompt: str):
        self.client = anthropic.Anthropic()
        self.tools = tools
        self.system_prompt = system_prompt
        self.messages = []

    def observe(self, image_path: str, instruction: str) -> str:
        """Send an image observation and instruction to the agent."""
        data, media_type = encode_image(image_path)

        self.messages.append({
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {"type": "base64", "media_type": media_type, "data": data},
                },
                {"type": "text", "text": instruction},
            ],
        })

        return self._run_agent_loop()

    def _run_agent_loop(self) -> str:
        """Execute the agent loop until a final text response."""
        while True:
            response = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=4096,
                system=self.system_prompt,
                tools=self.tools,
                messages=self.messages,
            )

            # 모든 콘텐츠 블록 수집
            self.messages.append({"role": "assistant", "content": response.content})

            # stop_reason이 "end_turn"이면 텍스트 반환
            if response.stop_reason == "end_turn":
                text_blocks = [b.text for b in response.content if b.type == "text"]
                return "\n".join(text_blocks)

            # 도구 호출 처리
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    result = self._execute_tool(block.name, block.input)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps(result),
                    })

            self.messages.append({"role": "user", "content": tool_results})

    def _execute_tool(self, name: str, input_data: dict) -> dict:
        """Execute a tool by name. Override in subclasses."""
        raise NotImplementedError(f"Tool '{name}' not implemented")
```

### 5.2 예시: 경비 보고서 에이전트

```python
class ExpenseAgent(VisionAgent):
    """Agent that processes receipt photos and creates expense reports."""

    def __init__(self):
        tools = [
            {
                "name": "log_expense",
                "description": "Log an expense entry extracted from a receipt.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "vendor": {"type": "string", "description": "Business name"},
                        "date": {"type": "string", "description": "Transaction date (YYYY-MM-DD)"},
                        "total": {"type": "number", "description": "Total amount"},
                        "currency": {"type": "string", "description": "Currency code"},
                        "category": {
                            "type": "string",
                            "enum": ["meals", "transport", "lodging", "supplies", "other"],
                        },
                        "items": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "description": {"type": "string"},
                                    "amount": {"type": "number"},
                                },
                            },
                        },
                    },
                    "required": ["vendor", "date", "total", "currency", "category"],
                },
            },
        ]
        super().__init__(
            tools=tools,
            system_prompt=(
                "You are an expense report assistant. When shown a receipt image, "
                "extract all relevant information and use the log_expense tool. "
                "Be precise with amounts and dates."
            ),
        )
        self.expenses = []

    def _execute_tool(self, name: str, input_data: dict) -> dict:
        if name == "log_expense":
            self.expenses.append(input_data)
            return {"status": "logged", "entry_id": len(self.expenses)}
        return {"error": f"Unknown tool: {name}"}


# 사용법
agent = ExpenseAgent()
agent.observe("receipt_lunch.jpg", "Process this receipt.")
agent.observe("receipt_taxi.jpg", "Process this receipt too.")

for expense in agent.expenses:
    print(f"{expense['date']} | {expense['vendor']} | {expense['total']} {expense['currency']}")
```

---

## 6. 이미지 기반 도구 사용 패턴

### 6.1 시각적 의사결정(Visual Decision Making)

Claude는 이미지를 사용하여 어떤 도구를 호출할지 결정할 수 있습니다:

```python
tools = [
    {
        "name": "create_jira_bug",
        "description": "Create a bug report in Jira.",
        "input_schema": {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "description": {"type": "string"},
                "severity": {"type": "string", "enum": ["critical", "major", "minor"]},
                "screenshot_url": {"type": "string"},
            },
            "required": ["title", "description", "severity"],
        },
    },
    {
        "name": "approve_design",
        "description": "Approve a design as matching the specification.",
        "input_schema": {
            "type": "object",
            "properties": {
                "design_id": {"type": "string"},
                "notes": {"type": "string"},
            },
            "required": ["design_id"],
        },
    },
]

# Claude가 스크린샷을 보고 결정: 버그인가, 아니면 디자인이 맞는가?
message = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=2048,
    tools=tools,
    system=(
        "You are a QA engineer. Compare the screenshot against the design spec. "
        "If there are visual bugs, file them with create_jira_bug. "
        "If the implementation matches the spec, use approve_design."
    ),
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {"type": "base64", "media_type": "image/png", "data": spec_b64},
                },
                {
                    "type": "image",
                    "source": {"type": "base64", "media_type": "image/png", "data": impl_b64},
                },
                {
                    "type": "text",
                    "text": "First image is the design spec. Second is the current implementation. Compare them.",
                },
            ],
        }
    ],
)
```

### 6.2 도구 사용을 통한 구조화된 추출(Structured Extraction)

텍스트로 JSON을 요청하는 대신, 도구를 사용하여 보장된 구조화 출력을 얻으세요:

```python
extraction_tool = {
    "name": "save_chart_data",
    "description": "Save extracted chart data.",
    "input_schema": {
        "type": "object",
        "properties": {
            "chart_type": {
                "type": "string",
                "enum": ["bar", "line", "pie", "scatter", "other"],
            },
            "title": {"type": "string"},
            "x_axis_label": {"type": "string"},
            "y_axis_label": {"type": "string"},
            "data_points": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "label": {"type": "string"},
                        "value": {"type": "number"},
                    },
                    "required": ["label", "value"],
                },
            },
        },
        "required": ["chart_type", "title", "data_points"],
    },
}

# Claude가 데이터를 추출하고 구조화된 출력으로 save_chart_data를 호출합니다
```

---

## 7. 컴퓨터 사용과 UI 자동화

Claude의 컴퓨터 사용(Computer Use) 기능을 통해 스크린샷을 찍고, 해석하고, 마우스/키보드 동작을 실행하여 그래픽 인터페이스와 상호작용할 수 있습니다.

### 7.1 컴퓨터 사용 개요

컴퓨터 사용을 통해 Claude는 다음을 할 수 있습니다:
- 스크린샷을 통해 컴퓨터 화면 확인
- 마우스 이동 및 요소 클릭
- 텍스트 입력 및 키보드 단축키 사용
- 스크롤, 드래그 등 기타 UI 동작 수행

> **참고**: 컴퓨터 사용은 현재 베타 상태입니다. `computer-use-2025-01-24` 베타 헤더가 필요하며 특정 도구 타입을 사용합니다.

### 7.2 컴퓨터 사용 설정

```python
import anthropic

client = anthropic.Anthropic()

# 컴퓨터 사용에는 특정 도구 정의가 필요합니다
computer_tool = {
    "type": "computer_20250124",
    "name": "computer",
    "display_width_px": 1920,
    "display_height_px": 1080,
    "display_number": 1,
}

text_editor_tool = {
    "type": "text_editor_20250124",
    "name": "str_replace_editor",
}

bash_tool = {
    "type": "bash_20250124",
    "name": "bash",
}

message = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=4096,
    betas=["computer-use-2025-01-24"],
    tools=[computer_tool, text_editor_tool, bash_tool],
    messages=[
        {
            "role": "user",
            "content": "Open the browser and navigate to https://example.com",
        }
    ],
)
```

### 7.3 스크린샷-동작 루프(Screenshot-Action Loop)

```python
import subprocess


def take_screenshot() -> str:
    """Take a screenshot and return base64-encoded PNG."""
    # macOS 예시
    subprocess.run(["screencapture", "-x", "/tmp/screenshot.png"], check=True)
    with open("/tmp/screenshot.png", "rb") as f:
        return base64.standard_b64encode(f.read()).decode("utf-8")


def execute_computer_action(action: dict):
    """Execute a computer use action (simplified example)."""
    action_type = action.get("action")

    if action_type == "screenshot":
        return take_screenshot()
    elif action_type == "click":
        x, y = action["coordinate"]
        subprocess.run(["cliclick", f"c:{x},{y}"], check=True)
    elif action_type == "type":
        text = action["text"]
        subprocess.run(["cliclick", f"t:{text}"], check=True)
    elif action_type == "key":
        key = action["key"]
        subprocess.run(["cliclick", f"kp:{key}"], check=True)

    return None


def computer_use_loop(instruction: str, max_steps: int = 20):
    """Run a computer use agent loop."""
    messages = [{"role": "user", "content": instruction}]

    for step in range(max_steps):
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            betas=["computer-use-2025-01-24"],
            tools=[computer_tool, text_editor_tool, bash_tool],
            messages=messages,
        )

        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "end_turn":
            return response.content

        # 도구 호출 처리
        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                if block.name == "computer":
                    result = execute_computer_action(block.input)
                    if result:  # 스크린샷 반환됨
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": [
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": "image/png",
                                        "data": result,
                                    },
                                }
                            ],
                        })
                    else:
                        # 동작 후 다음 관찰을 위해 스크린샷 촬영
                        screenshot = take_screenshot()
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": [
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": "image/png",
                                        "data": screenshot,
                                    },
                                }
                            ],
                        })

        messages.append({"role": "user", "content": tool_results})

    return "Max steps reached"
```

### 7.4 안전 고려사항

- **항상 샌드박스 환경에서 실행하세요** (VM, 컨테이너)
- **프로덕션 시스템에 컴퓨터 사용 접근 권한을 절대 부여하지 마세요**
- **동작 허용 목록(Action Allowlist)을 구현하세요**
- **파괴적 동작에 대해 사람이 확인하는 단계(Human-in-the-loop)를 추가하세요**
- **최대 단계 제한을 설정하여** 무한 자동화를 방지하세요

---

## 8. 비전 + MCP 서버 통합

MCP 서버는 Claude에 이미지를 제공하거나, Claude의 비전을 사용하여 외부 소스의 이미지를 처리할 수 있습니다.

### 8.1 이미지 제공 MCP 서버

```python
"""MCP server that provides images from a monitoring dashboard."""
from mcp.server import Server
from mcp.types import Resource, TextContent, ImageContent
import base64
import httpx

app = Server("dashboard-monitor")


@app.list_resources()
async def list_resources():
    return [
        Resource(
            uri="dashboard://grafana/panel/cpu",
            name="CPU Usage Panel",
            mimeType="image/png",
        ),
        Resource(
            uri="dashboard://grafana/panel/memory",
            name="Memory Usage Panel",
            mimeType="image/png",
        ),
    ]


@app.read_resource()
async def read_resource(uri: str):
    # Grafana 렌더 API를 통해 대시보드 패널 스크린샷 가져오기
    panel_id = uri.split("/")[-1]
    grafana_url = f"http://grafana:3000/render/d-solo/abc123?panelId={panel_id}&width=800&height=400"

    async with httpx.AsyncClient() as http_client:
        response = await http_client.get(
            grafana_url,
            headers={"Authorization": "Bearer <token>"},
        )
        image_b64 = base64.standard_b64encode(response.content).decode("utf-8")

    return [
        ImageContent(
            type="image",
            data=image_b64,
            mimeType="image/png",
        )
    ]
```

### 8.2 비전 강화 도구 서버(Vision-Enhanced Tool Server)

```python
"""MCP server that uses vision to process screenshots for test automation."""
from mcp.server import Server
from mcp.types import Tool

app = Server("visual-qa")


@app.list_tools()
async def list_tools():
    return [
        Tool(
            name="verify_ui_element",
            description="Verify a UI element appears correctly in a screenshot.",
            inputSchema={
                "type": "object",
                "properties": {
                    "screenshot_base64": {"type": "string"},
                    "element_description": {"type": "string"},
                    "expected_state": {
                        "type": "string",
                        "enum": ["visible", "hidden", "enabled", "disabled"],
                    },
                },
                "required": ["screenshot_base64", "element_description", "expected_state"],
            },
        ),
    ]
```

---

## 9. 비전 워크로드의 비용 최적화

비전 요청은 이미지 크기에 따라 토큰을 소비합니다. 비용 모델을 이해하면 효율적인 파이프라인을 구축하는 데 도움이 됩니다.

### 9.1 토큰 계산

Claude는 최대 해상도(가장 긴 변 기준 ~1568 px)에 맞게 이미지를 리사이즈한 다음, 타일 수를 기반으로 토큰 비용을 계산합니다:

| 이미지 크기 | 대략적인 토큰 수 |
|---|---|
| 200x200 px | ~170 토큰 |
| 800x600 px | ~800 토큰 |
| 1568x1568 px | ~1,600 토큰 |
| 4000x3000 px (리사이즈됨) | ~1,600 토큰 (리사이즈 후 동일) |

### 9.2 비용 절감 전략

```python
from PIL import Image
import io


def optimize_image_for_claude(
    image_path: str,
    max_dimension: int = 1024,
    quality: int = 80,
) -> tuple[str, str]:
    """Resize and compress an image to minimize token usage."""
    img = Image.open(image_path)

    # 필요 이상으로 큰 경우 리사이즈
    if max(img.size) > max_dimension:
        ratio = max_dimension / max(img.size)
        new_size = (int(img.width * ratio), int(img.height * ratio))
        img = img.resize(new_size, Image.LANCZOS)

    # 더 나은 압축을 위해 JPEG로 변환 (투명도가 필요하지 않은 경우)
    buffer = io.BytesIO()
    if img.mode == "RGBA":
        img.save(buffer, format="PNG", optimize=True)
        media_type = "image/png"
    else:
        img = img.convert("RGB")
        img.save(buffer, format="JPEG", quality=quality)
        media_type = "image/jpeg"

    data = base64.standard_b64encode(buffer.getvalue()).decode("utf-8")
    return data, media_type


def crop_region_of_interest(
    image_path: str,
    bbox: tuple[int, int, int, int],
) -> tuple[str, str]:
    """Crop to only the relevant region before sending to Claude."""
    img = Image.open(image_path)
    cropped = img.crop(bbox)  # (left, top, right, bottom)

    buffer = io.BytesIO()
    cropped.save(buffer, format="PNG")
    data = base64.standard_b64encode(buffer.getvalue()).decode("utf-8")
    return data, "image/png"
```

### 9.3 아키텍처 수준의 최적화

1. **저렴한 모델로 사전 필터링**: 경량 분류기를 사용하여 어떤 이미지가 Claude 분석이 필요한지 결정
2. **결과 캐싱**: 추출된 데이터를 저장하여 동일한 이미지 재처리 방지
3. **전략적 배칭**: 별도 호출 대신 관련 이미지를 하나의 요청으로 전송
4. **적절한 모델 사용**: 간단한 분류에는 Claude Haiku, 상세 분석에는 Sonnet
5. **해상도 낮추기**: 텍스트 추출, 차트 읽기 등 많은 작업이 800px에서도 잘 작동

```python
# 작업 복잡도에 따른 모델 선택
def select_model_for_task(task: str) -> str:
    """Choose the most cost-effective model for a vision task."""
    simple_tasks = {"classify", "detect_presence", "read_text"}
    moderate_tasks = {"extract_data", "compare", "describe"}
    complex_tasks = {"analyze_diagram", "audit_ui", "generate_code_from_design"}

    if task in simple_tasks:
        return "claude-haiku-4-20250514"
    elif task in moderate_tasks:
        return "claude-sonnet-4-20250514"
    else:
        return "claude-sonnet-4-20250514"  # 복잡한 작업에는 최상의 모델 사용
```

---

## 10. 연습 문제

### 연습 문제 1: 영수증 스캐너

다음 기능을 갖춘 영수증 스캐닝 도구를 구축하세요:
1. 영수증 이미지(사진 또는 스캔)를 받아들임
2. 판매처 이름, 날짜, 항목, 소계, 세금, 합계를 추출
3. 구조화된 JSON 반환
4. 여러 영수증 형식 처리

```python
"""
Exercise 1 starter code — complete the extract_receipt function.
"""
import anthropic
import base64
import json
from pathlib import Path


client = anthropic.Anthropic()


def extract_receipt(image_path: str) -> dict:
    """
    Extract structured data from a receipt image.

    Returns:
        {
            "vendor": str,
            "date": str,          # YYYY-MM-DD
            "items": [{"name": str, "qty": int, "price": float}],
            "subtotal": float,
            "tax": float,
            "total": float,
            "payment_method": str  # "cash", "card", or "unknown"
        }
    """
    # TODO: Encode the image
    # TODO: Send to Claude with a structured extraction prompt
    # TODO: Parse the response as JSON
    # TODO: Validate required fields
    pass


# Test with a sample receipt
if __name__ == "__main__":
    result = extract_receipt("sample_receipt.jpg")
    print(json.dumps(result, indent=2))
```

### 연습 문제 2: 시각적 차이 리포터(Visual Diff Reporter)

두 스크린샷을 비교하고 구조화된 차이 보고서를 생성하는 도구를 만드세요:

```python
"""
Exercise 2 starter code — complete the visual_diff_report function.
"""


def visual_diff_report(before_path: str, after_path: str) -> dict:
    """
    Compare two screenshots and generate a diff report.

    Returns:
        {
            "summary": str,
            "changes": [
                {
                    "type": "added" | "removed" | "modified",
                    "element": str,
                    "description": str,
                    "severity": "breaking" | "cosmetic" | "enhancement"
                }
            ],
            "overall_assessment": "pass" | "review_needed" | "fail"
        }
    """
    # TODO: Encode both images
    # TODO: Use tool_use for structured output
    # TODO: Handle the tool_use response
    pass
```

### 연습 문제 3: 대시보드 모니터 에이전트

주기적으로 스크린샷을 찍고 이상 징후에 대해 경고하는 웹 대시보드 모니터링 에이전트를 구축하세요:

```python
"""
Exercise 3 starter code — complete the DashboardMonitor class.
"""
import time


class DashboardMonitor:
    """Agent that monitors a dashboard and alerts on anomalies."""

    def __init__(self, alert_callback):
        self.client = anthropic.Anthropic()
        self.alert_callback = alert_callback
        self.baseline = None

    def set_baseline(self, screenshot_path: str):
        """Set the baseline 'normal' state of the dashboard."""
        # TODO: Store baseline image data
        # TODO: Extract baseline metrics with Claude
        pass

    def check(self, screenshot_path: str) -> dict:
        """Compare current state against baseline."""
        # TODO: Send both baseline and current to Claude
        # TODO: Ask Claude to identify anomalies
        # TODO: Call alert_callback if anomalies are found
        # TODO: Return structured check result
        pass

    def monitor_loop(self, screenshot_fn, interval: int = 60):
        """Continuously monitor the dashboard."""
        # TODO: Implement the monitoring loop
        # Take screenshot -> check -> alert -> wait
        pass
```

### 연습 문제 4: 다중 페이지 문서 처리기

다중 페이지 PDF를 처리하고 각 페이지에서 구조화된 데이터를 추출하는 파이프라인을 만드세요:

```python
"""
Exercise 4 starter code — implement a multi-page PDF processor.
"""


class DocumentProcessor:
    """Process multi-page documents with vision-based extraction."""

    def __init__(self):
        self.client = anthropic.Anthropic()

    def process_document(self, pdf_path: str, schema: dict) -> list[dict]:
        """
        Process each page of a PDF and extract data matching the schema.

        Args:
            pdf_path: Path to PDF file
            schema: JSON schema describing expected fields per page

        Returns:
            List of extracted data dicts, one per page
        """
        # TODO: Convert PDF pages to images
        # TODO: Process pages in batches of 20
        # TODO: Use tool_use with the provided schema
        # TODO: Aggregate results
        pass

    def summarize(self, results: list[dict]) -> str:
        """Generate a summary of the entire document."""
        # TODO: Send all extracted data to Claude for summarization
        pass
```

---

**이전**: [22. 트러블슈팅과 디버깅](./22_Troubleshooting.md) | **다음**: [24. 프롬프트 캐싱과 Batch API](./24_Prompt_Caching_and_Batch_API.md)
