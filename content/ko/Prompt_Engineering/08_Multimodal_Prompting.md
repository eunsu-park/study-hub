# 08. 멀티모달 프롬프팅(Multimodal Prompting)

**이전**: [멀티턴 대화](./07_Multi_Turn_Conversation.md) | **다음**: [코드 생성 프롬프팅](./09_Code_Generation_Prompting.md)

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 이미지 분석, 문서 이해, 시각적 추론을 위한 효과적인 비전+텍스트(Vision+Text) 프롬프트 구성
2. 차트, 그래프, 스크린샷에서 정보를 추출하기 위한 구조화된 프롬프팅 기법 적용
3. 이미지 간 비교, 시퀀싱, 종합을 수행하는 다중 이미지 추론 워크플로 설계
4. 이미지 예제를 사용한 멀티모달 퓨샷(Few-Shot) 프롬프팅 패턴 구현
5. 비전-언어 모델(Vision-Language Model)의 한계와 실패 모드를 식별하고 이를 우회하여 설계

---

텍스트와 이미지를 모두 처리할 수 있는 언어 모델은 연구적 호기심에서 프로덕션 도구로 발전했습니다. 스크린샷과 함께 자연어 질문을 보낼 수 있으면, 전체 범주의 문제가 해결 가능해집니다: 차트에서 데이터 추출, UI 레이아웃 이해, 필기 노트 읽기, 의료 이미지 분석, 시각적 디자인 비교. 그러나 멀티모달 프롬프팅(Multimodal Prompting)은 텍스트 전용 프롬프팅과는 다른 기법이 필요합니다 -- 시각적 작업을 구성하는 방식, 이미지와 텍스트의 순서, 비전 모델이 할 수 없는 것에 대한 인식 모두가 출력 품질에 크게 영향을 미칩니다.

이 레슨은 기본 이미지 분석부터 복잡한 다중 이미지 추론 워크플로까지, 효과적인 멀티모달 프롬프팅을 위한 실용적 기법을 다룹니다.

## 목차

1. [비전+텍스트 프롬프팅 기초](#1-visiontext-prompting-fundamentals)
2. [이미지 분석 프롬프트](#2-image-analysis-prompts)
3. [문서 이해](#3-document-understanding)
4. [차트 및 그래프 해석](#4-chart-and-graph-interpretation)
5. [다중 이미지 추론](#5-multi-image-reasoning)
6. [공간 추론 프롬프트](#6-spatial-reasoning-prompts)
7. [OCR 및 텍스트 추출](#7-ocr-and-text-extraction)
8. [시각적 질의 응답](#8-visual-question-answering)
9. [멀티모달 퓨샷](#9-multimodal-few-shot)
10. [한계 및 실패 모드](#10-limitations-and-failure-modes)

---

## 1. 비전+텍스트 프롬프팅 기초(Vision+Text Prompting Fundamentals)

### 1.1 비전 모델이 이미지를 처리하는 방법

Vision-language models (VLMs) like Claude, GPT-4o, and Gemini process images by converting them into a sequence of visual tokens that are then processed alongside text tokens. Key facts:

- Images are resized and divided into patches (typically 14x14 or 16x16 pixel patches)
- Each patch becomes one or more tokens in the model's context
- Higher resolution images produce more tokens (and cost more)
- The model does not have pixel-level precision -- it understands images at a semantic level

### 1.2 Anthropic API를 통한 이미지 전송

Claude accepts images as part of the message content using base64 encoding or URLs:

```python
import anthropic
import base64
from pathlib import Path


def encode_image(image_path: str) -> tuple[str, str]:
    """Encode an image file to base64 with its media type."""
    path = Path(image_path)
    suffix = path.suffix.lower()
    media_types = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp"
    }
    media_type = media_types.get(suffix, "image/png")

    with open(path, "rb") as f:
        image_data = base64.standard_b64encode(f.read()).decode("utf-8")

    return image_data, media_type


def analyze_image(image_path: str, prompt: str) -> str:
    """Send an image to Claude with a text prompt."""
    client = anthropic.Anthropic()
    image_data, media_type = encode_image(image_path)

    response = client.messages.create(
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
                            "data": image_data
                        }
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ]
    )

    return response.content[0].text


# Usage
result = analyze_image(
    "screenshot.png",
    "Describe what you see in this image. Focus on the main elements and their layout."
)
print(result)
```

### 1.3 OpenAI API를 통한 이미지 전송

```python
from openai import OpenAI
import base64
from pathlib import Path


def analyze_image_openai(image_path: str, prompt: str) -> str:
    """Send an image to GPT-4o with a text prompt."""
    client = OpenAI()

    with open(image_path, "rb") as f:
        image_data = base64.standard_b64encode(f.read()).decode("utf-8")

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_data}",
                            "detail": "high"  # "low", "high", or "auto"
                        }
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ],
        max_tokens=1024
    )

    return response.choices[0].message.content
```

### 1.4 이미지 토큰 비용

Image tokens add to your input cost. Understanding the token count helps with budgeting:

| Resolution | Approximate Tokens (Claude) | Approximate Tokens (GPT-4o) |
|-----------|---------------------------|---------------------------|
| Low (≤512px) | ~1,000 | ~85 (low detail) |
| Medium (~1024px) | ~2,000-4,000 | ~765 (high detail, 1 tile) |
| High (≥2048px) | ~4,000-8,000 | ~1,530+ (high detail, multiple tiles) |

**Best practice**: Resize images to the minimum resolution needed for the task. A chart analysis needs less resolution than reading tiny text.

```python
from PIL import Image
import io
import base64


def prepare_image(
    image_path: str,
    max_dimension: int = 1024,
    quality: int = 85
) -> tuple[str, str]:
    """Resize and compress an image for optimal API usage."""
    img = Image.open(image_path)

    # Resize if larger than max_dimension
    if max(img.size) > max_dimension:
        ratio = max_dimension / max(img.size)
        new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
        img = img.resize(new_size, Image.LANCZOS)

    # Convert to JPEG for compression (unless PNG is needed for transparency)
    buffer = io.BytesIO()
    if img.mode == "RGBA":
        img.save(buffer, format="PNG", optimize=True)
        media_type = "image/png"
    else:
        img = img.convert("RGB")
        img.save(buffer, format="JPEG", quality=quality)
        media_type = "image/jpeg"

    image_data = base64.standard_b64encode(buffer.getvalue()).decode("utf-8")

    return image_data, media_type
```

### 1.5 프롬프트 배치: 이미지 전 대 이미지 후

The placement of text relative to the image matters:

```python
import anthropic

client = anthropic.Anthropic()

# Pattern 1: Instructions BEFORE the image (recommended for analysis tasks)
# The model reads the instructions first, then views the image with context
response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    messages=[{
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "Look at this UI mockup. Identify any accessibility issues (color contrast, text size, missing alt text placeholders, keyboard navigation concerns)."
            },
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": image_data
                }
            }
        ]
    }]
)

# Pattern 2: Image BEFORE instructions (better for open-ended description)
# The model processes the image first, then focuses per instructions
response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    messages=[{
        "role": "user",
        "content": [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": image_data
                }
            },
            {
                "type": "text",
                "text": "What does this image show?"
            }
        ]
    }]
)
```

**Rule of thumb**: Put detailed instructions before the image when you want focused analysis. Put the image first when you want open-ended description.

---

## 2. 이미지 분석 프롬프트(Image Analysis Prompts)

### 2.1 구조화된 이미지 분석

Get consistent, structured output from image analysis:

```python
import anthropic
import json


def structured_image_analysis(
    image_data: str,
    media_type: str,
    analysis_type: str = "general"
) -> dict:
    """Perform structured analysis of an image."""
    client = anthropic.Anthropic()

    analysis_prompts = {
        "general": (
            "Analyze this image and return a JSON object with:\n"
            "- description: 2-3 sentence overview\n"
            "- objects: list of main objects/elements detected\n"
            "- colors: dominant colors\n"
            "- text_visible: any text visible in the image\n"
            "- mood: overall mood/tone\n"
            "- technical: {resolution_quality, lighting, composition}"
        ),
        "ui_review": (
            "Analyze this UI screenshot and return a JSON object with:\n"
            "- page_type: type of page (landing, dashboard, form, etc.)\n"
            "- layout: description of the layout structure\n"
            "- components: list of UI components identified\n"
            "- navigation: navigation elements found\n"
            "- text_content: key text content visible\n"
            "- issues: list of potential UX issues\n"
            "- accessibility: accessibility concerns"
        ),
        "product": (
            "Analyze this product image and return a JSON object with:\n"
            "- product_type: what kind of product\n"
            "- brand: brand if visible\n"
            "- color: product color(s)\n"
            "- condition: new/used/damaged\n"
            "- notable_features: list of visible features\n"
            "- estimated_category: product category for e-commerce"
        )
    }

    prompt = analysis_prompts.get(analysis_type, analysis_prompts["general"])

    tools = [{
        "name": "image_analysis",
        "description": "Return structured image analysis results",
        "input_schema": {
            "type": "object",
            "properties": {
                "description": {"type": "string"},
                "objects": {"type": "array", "items": {"type": "string"}},
                "colors": {"type": "array", "items": {"type": "string"}},
                "text_visible": {"type": "array", "items": {"type": "string"}},
                "mood": {"type": "string"},
                "technical": {
                    "type": "object",
                    "properties": {
                        "resolution_quality": {"type": "string"},
                        "lighting": {"type": "string"},
                        "composition": {"type": "string"}
                    }
                },
                "additional": {"type": "object"}
            }
        }
    }]

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        tools=tools,
        tool_choice={"type": "tool", "name": "image_analysis"},
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": image_data
                    }
                }
            ]
        }]
    )

    for block in response.content:
        if block.type == "tool_use":
            return block.input

    return {}
```

### 2.2 비교 이미지 분석

Compare two images systematically:

```python
import anthropic


def compare_images(
    image1_data: str,
    image1_type: str,
    image2_data: str,
    image2_type: str,
    comparison_focus: str = "general"
) -> str:
    """Compare two images with a specific focus."""
    client = anthropic.Anthropic()

    focus_prompts = {
        "general": "Compare these two images. What are the similarities and differences?",
        "design": (
            "Compare these two UI designs. Analyze:\n"
            "1. Layout differences\n"
            "2. Color scheme changes\n"
            "3. Typography differences\n"
            "4. Component additions/removals\n"
            "5. Which design is more effective and why?"
        ),
        "before_after": (
            "These are before (Image 1) and after (Image 2) images. "
            "Identify all changes between them. List each change as:\n"
            "- What changed\n"
            "- Location in the image\n"
            "- Whether the change is an improvement"
        ),
        "quality": (
            "Compare the quality of these two images. Evaluate:\n"
            "- Resolution and clarity\n"
            "- Color accuracy\n"
            "- Noise levels\n"
            "- Compression artifacts\n"
            "- Overall visual quality score (1-10 each)"
        )
    }

    prompt = focus_prompts.get(comparison_focus, focus_prompts["general"])

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2048,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": f"Image 1:"},
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": image1_type,
                        "data": image1_data
                    }
                },
                {"type": "text", "text": f"Image 2:"},
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": image2_type,
                        "data": image2_data
                    }
                },
                {"type": "text", "text": prompt}
            ]
        }]
    )

    return response.content[0].text
```

### 2.3 영역별 분석

Guide the model to focus on specific regions of an image:

```python
import anthropic


def analyze_region(
    image_data: str,
    media_type: str,
    region_description: str,
    question: str
) -> str:
    """Analyze a specific region of an image."""
    client = anthropic.Anthropic()

    prompt = f"""Focus on the following region of this image:
Region: {region_description}

{question}

Important: Ignore other parts of the image. Only analyze the specified region."""

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": image_data
                    }
                }
            ]
        }]
    )

    return response.content[0].text


# Usage
result = analyze_region(
    image_data,
    "image/png",
    region_description="The navigation bar at the top of the screen",
    question="List all menu items visible and identify any that appear to be dropdown menus."
)
```

---

## 3. 문서 이해(Document Understanding)

### 3.1 PDF 및 스크린샷 분석

Vision models can read documents from screenshots or rendered PDFs:

```python
import anthropic
import json


def extract_document_data(
    image_data: str,
    media_type: str,
    document_type: str = "general"
) -> dict:
    """Extract structured data from a document image."""
    client = anthropic.Anthropic()

    type_prompts = {
        "invoice": {
            "prompt": (
                "Extract all information from this invoice image. "
                "Return a JSON object with: vendor_name, invoice_number, "
                "date, due_date, line_items (array of {description, quantity, "
                "unit_price, total}), subtotal, tax, total_amount, currency, "
                "payment_terms, and any notes."
            ),
            "schema": {
                "type": "object",
                "properties": {
                    "vendor_name": {"type": "string"},
                    "invoice_number": {"type": "string"},
                    "date": {"type": "string"},
                    "due_date": {"type": "string"},
                    "line_items": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "description": {"type": "string"},
                                "quantity": {"type": "number"},
                                "unit_price": {"type": "number"},
                                "total": {"type": "number"}
                            }
                        }
                    },
                    "subtotal": {"type": "number"},
                    "tax": {"type": "number"},
                    "total_amount": {"type": "number"},
                    "currency": {"type": "string"}
                }
            }
        },
        "receipt": {
            "prompt": (
                "Extract data from this receipt. Return JSON with: "
                "store_name, date, time, items (array of {name, price}), "
                "subtotal, tax, total, payment_method."
            ),
            "schema": {
                "type": "object",
                "properties": {
                    "store_name": {"type": "string"},
                    "date": {"type": "string"},
                    "items": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "price": {"type": "number"}
                            }
                        }
                    },
                    "total": {"type": "number"},
                    "payment_method": {"type": "string"}
                }
            }
        },
        "business_card": {
            "prompt": (
                "Extract all information from this business card. "
                "Return JSON with: name, title, company, email, phone, "
                "address, website, social_media."
            ),
            "schema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "title": {"type": "string"},
                    "company": {"type": "string"},
                    "email": {"type": "string"},
                    "phone": {"type": "string"},
                    "website": {"type": "string"}
                }
            }
        },
        "general": {
            "prompt": (
                "Extract the key information from this document. "
                "Return JSON with: document_type, title, date (if present), "
                "key_fields (object of field names to values), "
                "body_text (main content summary), tables (if any)."
            ),
            "schema": {
                "type": "object",
                "properties": {
                    "document_type": {"type": "string"},
                    "title": {"type": "string"},
                    "key_fields": {"type": "object"},
                    "body_text": {"type": "string"}
                }
            }
        }
    }

    config = type_prompts.get(document_type, type_prompts["general"])

    tools = [{
        "name": "extract_document",
        "description": "Extract structured data from a document image",
        "input_schema": config["schema"]
    }]

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2048,
        tools=tools,
        tool_choice={"type": "tool", "name": "extract_document"},
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": config["prompt"]},
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": image_data
                    }
                }
            ]
        }]
    )

    for block in response.content:
        if block.type == "tool_use":
            return block.input

    return {}
```

### 3.2 다중 페이지 문서 처리

For documents with multiple pages, process each page as a separate image:

```python
import anthropic
import json
from typing import Optional


def process_multi_page_document(
    page_images: list[tuple[str, str]],  # List of (image_data, media_type)
    extraction_prompt: str,
    combine_strategy: str = "merge"
) -> dict:
    """Process a multi-page document by analyzing each page and combining results."""
    client = anthropic.Anthropic()
    page_results = []

    for i, (image_data, media_type) in enumerate(page_images):
        prompt = (
            f"This is page {i + 1} of {len(page_images)} of a document.\n\n"
            f"{extraction_prompt}\n\n"
            f"Extract information from THIS page only. Return JSON."
        )

        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2048,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": image_data
                        }
                    }
                ]
            },
            {"role": "assistant", "content": "{"}
            ]
        )

        try:
            page_data = json.loads("{" + response.content[0].text)
            page_results.append({"page": i + 1, "data": page_data})
        except json.JSONDecodeError:
            page_results.append({"page": i + 1, "data": {"raw_text": response.content[0].text}})

    # Combine results based on strategy
    if combine_strategy == "merge":
        return _merge_page_results(client, page_results)
    elif combine_strategy == "concatenate":
        return {"pages": page_results}
    else:
        return {"pages": page_results}


def _merge_page_results(
    client: anthropic.Anthropic,
    page_results: list[dict]
) -> dict:
    """Use LLM to intelligently merge multi-page extraction results."""
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2048,
        messages=[{
            "role": "user",
            "content": (
                "Merge these per-page extraction results into a single "
                "coherent document record. Combine split tables, merge "
                "partial information, and resolve any contradictions.\n\n"
                f"Page results:\n{json.dumps(page_results, indent=2)}\n\n"
                "Return a single merged JSON object."
            )
        },
        {"role": "assistant", "content": "{"}
        ]
    )

    try:
        return json.loads("{" + response.content[0].text)
    except json.JSONDecodeError:
        return {"pages": page_results, "merge_failed": True}
```

### 3.3 문서에서 테이블 추출

Tables are particularly challenging for vision models. Use explicit prompting:

```python
import anthropic
import json


def extract_table(
    image_data: str,
    media_type: str,
    table_description: str = ""
) -> list[list[str]]:
    """Extract a table from a document image into a 2D array."""
    client = anthropic.Anthropic()

    prompt = f"""Extract the table from this image into a structured format.

{f"Table context: {table_description}" if table_description else ""}

Return a JSON object with:
- "headers": array of column header strings
- "rows": array of arrays (each inner array is one row of cell values)
- "notes": any footnotes or notes associated with the table

Rules:
- Preserve the exact text in each cell
- Use null for empty cells
- If a cell spans multiple columns, repeat its value in each column
- Numbers should be strings to preserve formatting (e.g., "$1,234.56")
- Include ALL rows, don't truncate"""

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4096,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": image_data
                    }
                }
            ]
        },
        {"role": "assistant", "content": "{"}
        ]
    )

    try:
        result = json.loads("{" + response.content[0].text)
        return result
    except json.JSONDecodeError:
        return {"error": "Could not parse table data"}
```

---

## 4. 차트 및 그래프 해석(Chart and Graph Interpretation)

### 4.1 차트 읽기 프롬프트

Charts require specific prompting strategies because models need to interpret visual encodings (position, color, size) into data:

```python
import anthropic
import json


def analyze_chart(
    image_data: str,
    media_type: str,
    chart_type: str = "auto"
) -> dict:
    """Analyze a chart or graph image."""
    client = anthropic.Anthropic()

    type_specific = {
        "bar": (
            "For this bar chart, identify:\n"
            "- X-axis label and categories\n"
            "- Y-axis label and scale\n"
            "- Value for each bar (estimate from the axis)\n"
            "- Any trend or notable comparisons"
        ),
        "line": (
            "For this line chart, identify:\n"
            "- X-axis label and range\n"
            "- Y-axis label and range\n"
            "- Number of lines/series and their labels\n"
            "- Key data points (start, end, peaks, valleys)\n"
            "- Overall trend (increasing, decreasing, cyclical)"
        ),
        "pie": (
            "For this pie chart, identify:\n"
            "- Title of the chart\n"
            "- Each segment's label and percentage\n"
            "- The largest and smallest segments\n"
            "- Any segments that are highlighted or separated"
        ),
        "scatter": (
            "For this scatter plot, identify:\n"
            "- X-axis and Y-axis labels\n"
            "- Approximate range of data\n"
            "- Any visible correlation (positive, negative, none)\n"
            "- Outliers or clusters\n"
            "- Any trend lines shown"
        ),
        "auto": (
            "First, identify the type of chart (bar, line, pie, scatter, "
            "heatmap, etc.). Then extract all data and insights."
        )
    }

    prompt = f"""Analyze this chart/graph image carefully.

{type_specific.get(chart_type, type_specific["auto"])}

Return a JSON object with:
- "chart_type": type of chart
- "title": chart title (if visible)
- "axes": {{
    "x": {{"label": "...", "range": "..."}},
    "y": {{"label": "...", "range": "..."}}
  }}
- "data_series": [
    {{
      "name": "series name",
      "data_points": [{{"label": "...", "value": number}}]
    }}
  ]
- "insights": list of key observations
- "data_quality": "exact" | "estimated" (whether values are read precisely or estimated)"""

    tools = [{
        "name": "chart_analysis",
        "description": "Return chart analysis results",
        "input_schema": {
            "type": "object",
            "properties": {
                "chart_type": {"type": "string"},
                "title": {"type": "string"},
                "axes": {"type": "object"},
                "data_series": {"type": "array"},
                "insights": {"type": "array", "items": {"type": "string"}},
                "data_quality": {"type": "string", "enum": ["exact", "estimated"]}
            },
            "required": ["chart_type", "data_series", "insights", "data_quality"]
        }
    }]

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2048,
        tools=tools,
        tool_choice={"type": "tool", "name": "chart_analysis"},
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": image_data
                    }
                }
            ]
        }]
    )

    for block in response.content:
        if block.type == "tool_use":
            return block.input

    return {}
```

### 4.2 차트 데이터 추출 파이프라인

For high-accuracy data extraction, use a multi-pass approach:

```python
import anthropic
import json


def extract_chart_data_multipass(
    image_data: str,
    media_type: str
) -> dict:
    """Extract chart data with multi-pass verification."""
    client = anthropic.Anthropic()

    image_content = {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": media_type,
            "data": image_data
        }
    }

    # Pass 1: Identify chart structure
    structure_response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=512,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "Look at this chart. Tell me ONLY:\n"
                        "1. Chart type\n"
                        "2. Number of data series\n"
                        "3. Number of data points per series\n"
                        "4. Axis labels and ranges\n"
                        "Be precise and concise."
                    )
                },
                image_content
            ]
        }]
    )
    structure = structure_response.content[0].text

    # Pass 2: Extract data values
    data_response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2048,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        f"Chart structure: {structure}\n\n"
                        "Now carefully read each data point from the chart. "
                        "For each data point, estimate the value from the axis "
                        "markings. Return as JSON:\n"
                        '{"data": [{"label": "...", "value": number, '
                        '"confidence": "high"|"medium"|"low"}]}'
                    )
                },
                image_content
            ]
        },
        {"role": "assistant", "content": "{"}
        ]
    )

    try:
        data = json.loads("{" + data_response.content[0].text)
    except json.JSONDecodeError:
        data = {"error": "Parse failed"}

    # Pass 3: Verify with sanity checks
    verify_response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=512,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        f"I extracted this data from the chart:\n"
                        f"{json.dumps(data, indent=2)}\n\n"
                        "Look at the chart again and verify:\n"
                        "1. Are the values approximately correct?\n"
                        "2. Are any data points missing?\n"
                        "3. Do the values match the visual proportions?\n"
                        "Return corrections as JSON if needed, or "
                        '{"verified": true} if correct.'
                    )
                },
                image_content
            ]
        },
        {"role": "assistant", "content": "{"}
        ]
    )

    try:
        verification = json.loads("{" + verify_response.content[0].text)
    except json.JSONDecodeError:
        verification = {"verified": False, "error": "Parse failed"}

    return {
        "structure": structure,
        "data": data,
        "verification": verification
    }
```

### 4.3 대시보드 스크린샷 분석

Analyze complex dashboards with multiple charts:

```python
import anthropic
import json


def analyze_dashboard(
    image_data: str,
    media_type: str,
    context: str = ""
) -> dict:
    """Analyze a dashboard screenshot with multiple data visualizations."""
    client = anthropic.Anthropic()

    prompt = f"""Analyze this dashboard screenshot.

{f"Context: {context}" if context else ""}

For the dashboard as a whole:
1. What is the overall purpose/topic of this dashboard?
2. What time period does it cover?

For EACH chart/widget visible:
1. Type (bar, line, pie, KPI card, table, etc.)
2. Title or label
3. Key metric or data shown
4. Notable values or trends
5. Position on the dashboard (top-left, center, etc.)

Finally, provide:
- 3 key insights from the dashboard
- Any data anomalies or concerns
- Suggested follow-up questions a user might ask

Return the analysis as structured JSON."""

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4096,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": image_data
                    }
                }
            ]
        },
        {"role": "assistant", "content": "{"}
        ]
    )

    try:
        return json.loads("{" + response.content[0].text)
    except json.JSONDecodeError:
        return {"raw_analysis": response.content[0].text}
```

---

## 5. 다중 이미지 추론(Multi-Image Reasoning)

### 5.1 여러 이미지 전송

Both Claude and GPT-4o support multiple images in a single request:

```python
import anthropic


def multi_image_analysis(
    images: list[tuple[str, str]],  # List of (image_data, media_type)
    prompt: str,
    image_labels: list[str] = None
) -> str:
    """Analyze multiple images together."""
    client = anthropic.Anthropic()

    content = []
    for i, (data, mtype) in enumerate(images):
        label = image_labels[i] if image_labels and i < len(image_labels) else f"Image {i + 1}"
        content.append({"type": "text", "text": f"**{label}:**"})
        content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": mtype,
                "data": data
            }
        })

    content.append({"type": "text", "text": prompt})

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2048,
        messages=[{"role": "user", "content": content}]
    )

    return response.content[0].text
```

### 5.2 순차적 이미지 추론

Analyze a sequence of images (e.g., steps in a process, frames from a video):

```python
import anthropic
import json


def analyze_sequence(
    images: list[tuple[str, str]],
    sequence_type: str = "steps"
) -> dict:
    """Analyze a sequence of images for temporal or procedural patterns."""
    client = anthropic.Anthropic()

    type_prompts = {
        "steps": (
            "These images show sequential steps in a process. "
            "For each image, describe what step it represents. "
            "Then identify:\n"
            "- The overall process being shown\n"
            "- Any missing steps between images\n"
            "- Whether the sequence is complete"
        ),
        "evolution": (
            "These images show something changing over time. "
            "For each image, note the timestamp or position in the sequence. "
            "Describe what changed between consecutive images. "
            "Identify the overall trend."
        ),
        "comparison": (
            "These images show different versions or options. "
            "Compare them systematically on key dimensions. "
            "Identify which is best for different use cases."
        )
    }

    content = []
    for i, (data, mtype) in enumerate(images):
        content.append({"type": "text", "text": f"Image {i + 1} of {len(images)}:"})
        content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": mtype,
                "data": data
            }
        })

    content.append({
        "type": "text",
        "text": (
            f"{type_prompts.get(sequence_type, type_prompts['steps'])}\n\n"
            "Return your analysis as JSON with:\n"
            '- "sequence_description": overall description\n'
            '- "per_image": [{image_number, description, key_elements}]\n'
            '- "transitions": [{from_image, to_image, changes}]\n'
            '- "insights": [list of observations]'
        )
    })

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4096,
        messages=[{"role": "user", "content": content}]
    )

    text = response.content[0].text
    try:
        import re
        json_match = re.search(r"\{[\s\S]*\}", text)
        if json_match:
            return json.loads(json_match.group())
    except json.JSONDecodeError:
        pass

    return {"raw_analysis": text}
```

### 5.3 시각적 차이 비교(Visual Diff)

Detect differences between two versions of an image:

```python
import anthropic
import json


def visual_diff(
    before_data: str,
    before_type: str,
    after_data: str,
    after_type: str,
    diff_type: str = "ui"
) -> list[dict]:
    """Find visual differences between two images."""
    client = anthropic.Anthropic()

    diff_prompts = {
        "ui": (
            "Compare these two UI screenshots (before and after). "
            "List EVERY visual change you can detect, no matter how small:\n"
            "- Added elements\n"
            "- Removed elements\n"
            "- Modified elements (color, size, position, text)\n"
            "- Layout changes"
        ),
        "design": (
            "Compare these two design mockups. "
            "List all design changes:\n"
            "- Color changes\n"
            "- Typography changes\n"
            "- Spacing/layout changes\n"
            "- New or removed visual elements"
        ),
        "document": (
            "Compare these two document versions. "
            "List all content changes:\n"
            "- Added text\n"
            "- Removed text\n"
            "- Modified text\n"
            "- Formatting changes"
        )
    }

    tools = [{
        "name": "report_diffs",
        "description": "Report visual differences between images",
        "input_schema": {
            "type": "object",
            "properties": {
                "total_changes": {"type": "integer"},
                "changes": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "change_type": {
                                "type": "string",
                                "enum": ["added", "removed", "modified", "moved"]
                            },
                            "element": {"type": "string"},
                            "location": {"type": "string"},
                            "details": {"type": "string"},
                            "significance": {
                                "type": "string",
                                "enum": ["major", "minor", "cosmetic"]
                            }
                        },
                        "required": ["change_type", "element", "details"]
                    }
                },
                "summary": {"type": "string"}
            },
            "required": ["total_changes", "changes", "summary"]
        }
    }]

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4096,
        tools=tools,
        tool_choice={"type": "tool", "name": "report_diffs"},
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": "BEFORE:"},
                {
                    "type": "image",
                    "source": {"type": "base64", "media_type": before_type, "data": before_data}
                },
                {"type": "text", "text": "AFTER:"},
                {
                    "type": "image",
                    "source": {"type": "base64", "media_type": after_type, "data": after_data}
                },
                {"type": "text", "text": diff_prompts.get(diff_type, diff_prompts["ui"])}
            ]
        }]
    )

    for block in response.content:
        if block.type == "tool_use":
            return block.input

    return {}
```

---

## 6. 공간 추론 프롬프트(Spatial Reasoning Prompts)

### 6.1 공간 관계 분석

Vision models can reason about spatial relationships, but they need explicit prompting:

```python
import anthropic


def analyze_spatial_layout(
    image_data: str,
    media_type: str,
    context: str = "general"
) -> str:
    """Analyze spatial relationships in an image."""
    client = anthropic.Anthropic()

    prompts = {
        "general": (
            "Describe the spatial layout of this image:\n"
            "1. What objects/elements are present?\n"
            "2. Where is each object relative to others? (above, below, left of, right of, overlapping)\n"
            "3. What is in the foreground vs background?\n"
            "4. Estimate approximate distances or proportions between elements\n"
            "5. Is there a visual hierarchy? What draws attention first?"
        ),
        "architecture": (
            "Analyze the architectural layout shown in this image:\n"
            "1. Identify rooms or spaces\n"
            "2. Describe connectivity (doors, passages)\n"
            "3. Note dimensions if visible\n"
            "4. Describe the flow of the space\n"
            "5. Identify any structural elements (walls, columns, stairs)"
        ),
        "ui_layout": (
            "Analyze the UI layout of this screen:\n"
            "1. Grid structure (how many columns/rows)\n"
            "2. Component positioning and alignment\n"
            "3. Whitespace usage\n"
            "4. Visual hierarchy (what draws the eye first, second, third)\n"
            "5. Responsive layout implications (what would break on mobile?)"
        )
    }

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2048,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": prompts.get(context, prompts["general"])},
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": image_data
                    }
                }
            ]
        }]
    )

    return response.content[0].text
```

### 6.2 공간 카운팅 및 측정

Models can count objects but need prompting to be careful:

```python
import anthropic
import json


def count_objects(
    image_data: str,
    media_type: str,
    object_description: str
) -> dict:
    """Count specific objects in an image with verification."""
    client = anthropic.Anthropic()

    # Two-pass counting for accuracy
    prompt = f"""Count the number of {object_description} in this image.

IMPORTANT: Count carefully using this method:
1. Scan the image systematically (left to right, top to bottom)
2. As you find each {object_description}, number it mentally
3. Double-check by scanning again in a different direction
4. If objects partially overlap, count each one separately

Return JSON:
{{
  "count": <number>,
  "confidence": "high" | "medium" | "low",
  "locations": ["brief description of where each one is"],
  "notes": "any caveats about the count"
}}"""

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": image_data
                    }
                }
            ]
        },
        {"role": "assistant", "content": "{"}
        ]
    )

    try:
        return json.loads("{" + response.content[0].text)
    except json.JSONDecodeError:
        return {"error": "Could not parse count"}
```

---

## 7. OCR 및 텍스트 추출(OCR and Text Extraction)

### 7.1 이미지에서 텍스트 추출

Vision models can perform OCR, though dedicated OCR tools may be more accurate for simple text:

```python
import anthropic
import json


def extract_text(
    image_data: str,
    media_type: str,
    extraction_mode: str = "all"
) -> dict:
    """Extract text from an image."""
    client = anthropic.Anthropic()

    mode_prompts = {
        "all": (
            "Extract ALL visible text from this image. "
            "Preserve the original layout as much as possible. "
            "Use line breaks to show text that appears on separate lines. "
            "Note the approximate position of each text block."
        ),
        "structured": (
            "Extract text from this image into structured categories:\n"
            "- headings: text that appears as headings/titles\n"
            "- body: main body text\n"
            "- labels: labels for UI elements, buttons, etc.\n"
            "- metadata: dates, numbers, IDs, etc.\n"
            "- other: anything else"
        ),
        "handwritten": (
            "This image contains handwritten text. "
            "Carefully read and transcribe all handwritten content. "
            "If any words are unclear, provide your best guess in [brackets]. "
            "Preserve line breaks as they appear. "
            "Note the overall legibility (good/fair/poor)."
        ),
        "code": (
            "This image contains source code. "
            "Extract the code exactly as written, preserving:\n"
            "- Indentation\n"
            "- Line breaks\n"
            "- Comments\n"
            "- String literals\n"
            "Identify the programming language if possible."
        )
    }

    prompt = mode_prompts.get(extraction_mode, mode_prompts["all"])

    tools = [{
        "name": "extracted_text",
        "description": "Return extracted text from the image",
        "input_schema": {
            "type": "object",
            "properties": {
                "text_blocks": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "content": {"type": "string"},
                            "category": {"type": "string"},
                            "position": {"type": "string"},
                            "confidence": {
                                "type": "string",
                                "enum": ["high", "medium", "low"]
                            }
                        },
                        "required": ["content", "confidence"]
                    }
                },
                "full_text": {"type": "string"},
                "language": {"type": "string"},
                "overall_legibility": {"type": "string"}
            },
            "required": ["text_blocks", "full_text"]
        }
    }]

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4096,
        tools=tools,
        tool_choice={"type": "tool", "name": "extracted_text"},
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": image_data
                    }
                }
            ]
        }]
    )

    for block in response.content:
        if block.type == "tool_use":
            return block.input

    return {}
```

### 7.2 다국어 OCR

Vision models handle multilingual text well:

```python
import anthropic
import json


def extract_multilingual_text(
    image_data: str,
    media_type: str,
    expected_languages: list[str] = None
) -> dict:
    """Extract text from an image that may contain multiple languages."""
    client = anthropic.Anthropic()

    lang_hint = ""
    if expected_languages:
        lang_hint = f"Expected languages: {', '.join(expected_languages)}. "

    prompt = f"""Extract all text from this image.

{lang_hint}

For each text block:
1. Identify the language
2. Provide the original text exactly as written
3. If the text is not in English, provide an English translation
4. Note the position in the image

Return JSON with:
- "text_blocks": array of {{
    "original": "text as written",
    "language": "detected language",
    "translation": "English translation (null if already English)",
    "position": "where in the image"
  }}
- "languages_detected": list of all languages found"""

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4096,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": image_data
                    }
                }
            ]
        },
        {"role": "assistant", "content": "{"}
        ]
    )

    try:
        return json.loads("{" + response.content[0].text)
    except json.JSONDecodeError:
        return {"raw": response.content[0].text}
```

---

## 8. 시각적 질의 응답(Visual Question Answering)

### 8.1 직접 VQA

Visual question answering is the simplest form of multimodal prompting -- ask a question about an image:

```python
import anthropic


def visual_qa(
    image_data: str,
    media_type: str,
    question: str,
    answer_format: str = "natural"
) -> str:
    """Answer a question about an image."""
    client = anthropic.Anthropic()

    format_instructions = {
        "natural": "",
        "yes_no": "Answer with ONLY 'Yes' or 'No', then a brief explanation.",
        "multiple_choice": "Select the best answer from the options provided.",
        "numeric": "Provide a numeric answer. If estimating, include a range.",
        "list": "Provide your answer as a bullet-point list."
    }

    prompt = f"""{question}

{format_instructions.get(answer_format, "")}"""

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": image_data
                    }
                },
                {"type": "text", "text": prompt}
            ]
        }]
    )

    return response.content[0].text


# Usage examples
# result = visual_qa(img, "image/png", "How many people are in this photo?", "numeric")
# result = visual_qa(img, "image/png", "Is this a indoor or outdoor scene?", "yes_no")
# result = visual_qa(img, "image/png", "What emotions are visible on the faces?", "list")
```

### 8.2 다단계 시각적 추론

For complex questions, guide the model through a reasoning process:

```python
import anthropic


def visual_reasoning(
    image_data: str,
    media_type: str,
    question: str
) -> str:
    """Perform multi-step visual reasoning about an image."""
    client = anthropic.Anthropic()

    prompt = f"""I need you to answer this question about the image, but first
work through the reasoning step by step.

Question: {question}

Follow this process:
1. OBSERVE: List the relevant visual elements you can see
2. IDENTIFY: Note specific details that relate to the question
3. REASON: Connect the observations to form an answer
4. VERIFY: Check if your reasoning is consistent with what you see
5. ANSWER: State your final answer clearly

Show your work for each step."""

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2048,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": image_data
                    }
                },
                {"type": "text", "text": prompt}
            ]
        }]
    )

    return response.content[0].text
```

### 8.3 대화형 VQA

Ask follow-up questions about an image across multiple turns:

```python
import anthropic
from dataclasses import dataclass, field


@dataclass
class VisualConversation:
    """Multi-turn conversation about an image."""
    image_data: str
    media_type: str
    model: str = "claude-sonnet-4-20250514"
    messages: list[dict] = field(default_factory=list)
    _client: anthropic.Anthropic = field(
        default_factory=anthropic.Anthropic, repr=False
    )

    def _build_first_message(self, question: str) -> dict:
        """Build the first message with the image."""
        return {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": self.media_type,
                        "data": self.image_data
                    }
                },
                {"type": "text", "text": question}
            ]
        }

    def ask(self, question: str) -> str:
        """Ask a question about the image."""
        if not self.messages:
            # First question includes the image
            self.messages.append(self._build_first_message(question))
        else:
            # Follow-up questions reference the same image
            self.messages.append({
                "role": "user",
                "content": question
            })

        response = self._client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=(
                "You are analyzing an image provided in the first message. "
                "All follow-up questions refer to the same image. "
                "Be specific and reference visual details."
            ),
            messages=self.messages
        )

        reply = response.content[0].text
        self.messages.append({"role": "assistant", "content": reply})
        return reply


# Usage
# conv = VisualConversation(image_data=img_data, media_type="image/png")
# print(conv.ask("What type of chart is shown in this image?"))
# print(conv.ask("What is the highest value?"))
# print(conv.ask("What trend does the data show?"))
```

---

## 9. 멀티모달 퓨샷(Multimodal Few-Shot)

### 9.1 이미지 예제를 사용한 퓨샷(Few-Shot)

Just as text few-shot provides examples for the model to follow, multimodal few-shot provides image-response pairs:

```python
import anthropic


def multimodal_few_shot(
    examples: list[dict],  # Each: {"image_data": str, "media_type": str, "response": str}
    query_image_data: str,
    query_media_type: str,
    task_description: str
) -> str:
    """Perform a task using multimodal few-shot examples."""
    client = anthropic.Anthropic()

    messages = []

    # Build example turns
    for i, example in enumerate(examples):
        # User turn with example image
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": f"Example {i + 1}:"},
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": example["media_type"],
                        "data": example["image_data"]
                    }
                },
                {"type": "text", "text": task_description}
            ]
        })

        # Assistant turn with example response
        messages.append({
            "role": "assistant",
            "content": example["response"]
        })

    # Add the actual query
    messages.append({
        "role": "user",
        "content": [
            {"type": "text", "text": "Now analyze this image:"},
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": query_media_type,
                    "data": query_image_data
                }
            },
            {"type": "text", "text": task_description}
        ]
    })

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        messages=messages
    )

    return response.content[0].text


# Usage example (with placeholder data)
# examples = [
#     {
#         "image_data": happy_face_b64,
#         "media_type": "image/jpeg",
#         "response": '{"emotion": "happy", "confidence": 0.95}'
#     },
#     {
#         "image_data": sad_face_b64,
#         "media_type": "image/jpeg",
#         "response": '{"emotion": "sad", "confidence": 0.88}'
#     }
# ]
# result = multimodal_few_shot(
#     examples, query_image_b64, "image/jpeg",
#     "Classify the dominant emotion in this facial expression. Return JSON."
# )
```

### 9.2 템플릿 기반 멀티모달 프롬프팅

Create reusable templates for common multimodal tasks:

```python
from dataclasses import dataclass
from typing import Optional
import anthropic
import json


@dataclass
class MultimodalTemplate:
    """Reusable template for multimodal prompts."""
    name: str
    system_prompt: str
    user_prompt_template: str  # {image} placeholder for image position
    output_format: str = "text"  # "text", "json", or "tool"
    tool_schema: Optional[dict] = None

    def render(
        self,
        image_data: str,
        media_type: str,
        **kwargs
    ) -> list[dict]:
        """Render the template into API message format."""
        prompt_text = self.user_prompt_template.format(**kwargs)

        # Split prompt at {image} placeholder to determine image position
        if "{image}" in prompt_text:
            parts = prompt_text.split("{image}")
            content = []
            if parts[0].strip():
                content.append({"type": "text", "text": parts[0].strip()})
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": image_data
                }
            })
            if len(parts) > 1 and parts[1].strip():
                content.append({"type": "text", "text": parts[1].strip()})
        else:
            content = [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": image_data
                    }
                },
                {"type": "text", "text": prompt_text}
            ]

        return [{"role": "user", "content": content}]


# Define reusable templates
TEMPLATES = {
    "product_catalog": MultimodalTemplate(
        name="product_catalog",
        system_prompt=(
            "You extract product information for e-commerce catalogs. "
            "Be precise with colors, dimensions, and materials."
        ),
        user_prompt_template=(
            "Extract product information from this image for a "
            "{marketplace} listing.\n\n{{image}}\n\n"
            "Category hint: {category}"
        ),
        output_format="json"
    ),
    "accessibility_audit": MultimodalTemplate(
        name="accessibility_audit",
        system_prompt=(
            "You are a WCAG 2.1 AA compliance checker. "
            "Be thorough and cite specific WCAG criteria."
        ),
        user_prompt_template=(
            "Audit this UI for WCAG 2.1 AA accessibility compliance.\n\n"
            "{{image}}\n\n"
            "Focus areas: {focus_areas}"
        ),
        output_format="json"
    ),
    "diagram_to_code": MultimodalTemplate(
        name="diagram_to_code",
        system_prompt=(
            "You convert visual diagrams into code. "
            "Produce clean, well-commented code."
        ),
        user_prompt_template=(
            "Convert this {diagram_type} diagram into {language} code.\n\n"
            "{{image}}\n\n"
            "Requirements: {requirements}"
        ),
        output_format="text"
    )
}


def execute_template(
    template_name: str,
    image_data: str,
    media_type: str,
    **kwargs
) -> str:
    """Execute a multimodal template."""
    template = TEMPLATES[template_name]
    client = anthropic.Anthropic()

    messages = template.render(image_data, media_type, **kwargs)

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2048,
        system=template.system_prompt,
        messages=messages
    )

    return response.content[0].text


# Usage
# result = execute_template(
#     "product_catalog",
#     image_data=product_img_b64,
#     media_type="image/jpeg",
#     marketplace="Amazon",
#     category="Electronics"
# )
```

---

## 10. 한계 및 실패 모드(Limitations and Failure Modes)

### 10.1 비전 모델의 알려진 한계

Understanding what vision models cannot do reliably is as important as knowing what they can do:

| Limitation | Description | Workaround |
|-----------|-------------|------------|
| **Small text** | Text smaller than ~10px is unreliable | Crop and zoom the relevant area |
| **Precise counting** | Counting >20 similar objects is error-prone | Use grid overlay or counting aids |
| **Exact measurements** | Cannot give precise pixel values | Provide reference dimensions |
| **Color precision** | Cannot distinguish very similar colors | Provide color labels or reference |
| **Rotated/skewed text** | Accuracy drops for non-horizontal text | Pre-process to deskew |
| **Low contrast** | Struggles with light text on light backgrounds | Enhance contrast before sending |
| **Dense information** | Misses details in very busy images | Crop to relevant sections |
| **Temporal reasoning** | Cannot determine when a photo was taken from content alone | Provide context |

### 10.2 비전에서의 환각(Hallucination)

Vision models can hallucinate -- confidently describe things that are not in the image:

```python
import anthropic


def verified_analysis(
    image_data: str,
    media_type: str,
    question: str
) -> dict:
    """Analyze an image with built-in hallucination checks."""
    client = anthropic.Anthropic()

    # Step 1: Answer the question
    response1 = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": image_data
                    }
                },
                {"type": "text", "text": question}
            ]
        }]
    )
    initial_answer = response1.content[0].text

    # Step 2: Verification pass
    response2 = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": image_data
                    }
                },
                {
                    "type": "text",
                    "text": (
                        f"I previously analyzed this image and said:\n\n"
                        f"{initial_answer}\n\n"
                        f"Now look at the image again and verify EACH claim "
                        f"in my analysis. For each claim, state:\n"
                        f"- CONFIRMED: clearly visible in the image\n"
                        f"- UNCERTAIN: might be correct but hard to verify\n"
                        f"- INCORRECT: not supported by what's in the image\n\n"
                        f"Then provide a corrected analysis."
                    )
                }
            ]
        }]
    )

    return {
        "initial_answer": initial_answer,
        "verification": response2.content[0].text
    }
```

### 10.3 해상도 및 품질 문제

```python
import anthropic


def adaptive_analysis(
    image_data: str,
    media_type: str,
    task: str
) -> str:
    """Adapt analysis approach based on image quality assessment."""
    client = anthropic.Anthropic()

    # First: assess image quality
    quality_response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=256,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": image_data
                    }
                },
                {
                    "type": "text",
                    "text": (
                        "Rate this image's quality for analysis on a scale of 1-5:\n"
                        "- Text readability (1-5)\n"
                        "- Overall clarity (1-5)\n"
                        "- Detail visibility (1-5)\n"
                        "Respond with ONLY three numbers separated by commas."
                    )
                }
            ]
        }]
    )

    quality = quality_response.content[0].text.strip()

    # Parse quality scores
    try:
        scores = [int(s.strip()) for s in quality.split(",")[:3]]
        text_quality, clarity, detail = scores
    except (ValueError, IndexError):
        text_quality = clarity = detail = 3  # Default to medium

    # Adapt prompt based on quality
    caveats = []
    if text_quality <= 2:
        caveats.append(
            "Note: Text in this image appears low quality. "
            "Mark any text you're unsure about with [?]. "
            "Do not guess at illegible text."
        )
    if clarity <= 2:
        caveats.append(
            "Note: This image is blurry or low resolution. "
            "Only describe elements you can clearly identify. "
            "State your confidence level for uncertain observations."
        )
    if detail <= 2:
        caveats.append(
            "Note: Fine details are not clearly visible. "
            "Focus on large, clearly visible elements only."
        )

    caveat_text = "\n".join(caveats) if caveats else ""

    # Perform the actual task with quality-aware prompting
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2048,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": image_data
                    }
                },
                {
                    "type": "text",
                    "text": f"{caveat_text}\n\n{task}"
                }
            ]
        }]
    )

    return response.content[0].text
```

### 10.4 실패 모드 카탈로그

Awareness of failure modes helps you design better prompts:

**Failure Mode 1: Reading text from decorative fonts**
- Vision models struggle with highly stylized, cursive, or artistic text
- Workaround: Describe the text style and ask the model to attempt multiple interpretations

**Failure Mode 2: Understanding infographics with novel visual encodings**
- Standard chart types (bar, line, pie) work well; custom visualizations are harder
- Workaround: Describe the visual encoding system in your prompt

**Failure Mode 3: Spatial precision ("click at coordinate X,Y")**
- Models can identify regions but not precise pixel coordinates
- Workaround: Use relative descriptions ("top-left quadrant") instead of coordinates

**Failure Mode 4: Reasoning about 3D from 2D images**
- Depth estimation, occlusion reasoning, and 3D structure are weak
- Workaround: Provide multiple angles or explicitly ask about depth uncertainty

**Failure Mode 5: Distinguishing similar objects in dense scenes**
- In crowds, cluttered shelves, or dense text, the model may merge or miss items
- Workaround: Crop to smaller regions and process individually

---

## 연습문제

### 연습문제 1: 문서 처리 파이프라인

Build a pipeline that takes a document screenshot (invoice, receipt, or form), automatically detects the document type, extracts structured data, and validates the extraction.

**Requirements:**
- Auto-detect document type from the image
- Use type-specific extraction schemas
- Validate extracted data (e.g., line item totals sum to subtotal)
- Return confidence scores for each extracted field

<details><summary>정답 보기</summary>

```python
import anthropic
import json
from pydantic import BaseModel, Field
from typing import Optional


class LineItem(BaseModel):
    description: str
    quantity: Optional[float] = None
    unit_price: Optional[float] = None
    total: float


class DocumentExtraction(BaseModel):
    document_type: str
    confidence: float = Field(ge=0, le=1)
    vendor_or_source: Optional[str] = None
    date: Optional[str] = None
    reference_number: Optional[str] = None
    line_items: list[LineItem] = Field(default_factory=list)
    subtotal: Optional[float] = None
    tax: Optional[float] = None
    total: Optional[float] = None
    currency: str = "USD"
    field_confidences: dict[str, float] = Field(default_factory=dict)
    validation_warnings: list[str] = Field(default_factory=list)


def process_document(image_data: str, media_type: str) -> DocumentExtraction:
    """Full document processing pipeline."""
    client = anthropic.Anthropic()

    # Step 1: Detect document type
    detect_response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=128,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": image_data
                    }
                },
                {
                    "type": "text",
                    "text": (
                        "What type of document is this? "
                        "Reply with ONE word: invoice, receipt, form, "
                        "letter, contract, or other."
                    )
                }
            ]
        }]
    )
    doc_type = detect_response.content[0].text.strip().lower()

    # Step 2: Extract data using tool calling
    tools = [{
        "name": "extract_document_data",
        "description": "Extract structured data from a document",
        "input_schema": {
            "type": "object",
            "properties": {
                "document_type": {"type": "string"},
                "vendor_or_source": {"type": "string"},
                "date": {"type": "string"},
                "reference_number": {"type": "string"},
                "line_items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "description": {"type": "string"},
                            "quantity": {"type": "number"},
                            "unit_price": {"type": "number"},
                            "total": {"type": "number"}
                        },
                        "required": ["description", "total"]
                    }
                },
                "subtotal": {"type": "number"},
                "tax": {"type": "number"},
                "total": {"type": "number"},
                "currency": {"type": "string"},
                "field_confidences": {
                    "type": "object",
                    "description": "Confidence (0-1) for each extracted field"
                }
            },
            "required": ["document_type", "field_confidences"]
        }
    }]

    extract_response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4096,
        tools=tools,
        tool_choice={"type": "tool", "name": "extract_document_data"},
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        f"This is a {doc_type}. Extract ALL data from it. "
                        f"For each field, rate your confidence from 0.0 to 1.0 "
                        f"in the field_confidences object. "
                        f"Include all line items with exact amounts."
                    )
                },
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": image_data
                    }
                }
            ]
        }]
    )

    raw_data = {}
    for block in extract_response.content:
        if block.type == "tool_use":
            raw_data = block.input

    # Step 3: Validate
    warnings = []

    # Check line item math
    if raw_data.get("line_items") and raw_data.get("subtotal"):
        computed_subtotal = sum(
            item.get("total", 0) for item in raw_data["line_items"]
        )
        if abs(computed_subtotal - raw_data["subtotal"]) > 0.02:
            warnings.append(
                f"Line items sum ({computed_subtotal:.2f}) doesn't match "
                f"subtotal ({raw_data['subtotal']:.2f})"
            )

    # Check subtotal + tax = total
    if raw_data.get("subtotal") and raw_data.get("tax") and raw_data.get("total"):
        computed_total = raw_data["subtotal"] + raw_data["tax"]
        if abs(computed_total - raw_data["total"]) > 0.02:
            warnings.append(
                f"Subtotal + tax ({computed_total:.2f}) doesn't match "
                f"total ({raw_data['total']:.2f})"
            )

    # Check for low confidence fields
    for field_name, conf in raw_data.get("field_confidences", {}).items():
        if conf < 0.5:
            warnings.append(f"Low confidence for '{field_name}': {conf}")

    raw_data["validation_warnings"] = warnings
    raw_data["confidence"] = sum(
        raw_data.get("field_confidences", {}).values()
    ) / max(len(raw_data.get("field_confidences", {"_": 1})), 1)

    return DocumentExtraction.model_validate(raw_data)


# Usage
# result = process_document(invoice_image_b64, "image/png")
# print(result.model_dump_json(indent=2))
```

</details>

### 연습문제 2: 검증이 포함된 차트 데이터 추출

Build a system that extracts data from a chart image with a multi-pass verification approach. The system should: (1) identify chart type and structure, (2) extract data values, (3) verify the extracted data against the visual, and (4) provide a confidence-rated dataset.

<details><summary>정답 보기</summary>

```python
import anthropic
import json
from dataclasses import dataclass, field


@dataclass
class DataPoint:
    label: str
    value: float
    confidence: str  # high, medium, low
    verified: bool = False


@dataclass
class ChartData:
    chart_type: str
    title: str
    x_axis: str
    y_axis: str
    data_points: list[DataPoint] = field(default_factory=list)
    overall_confidence: float = 0.0
    verification_notes: list[str] = field(default_factory=list)


def extract_chart_data_verified(
    image_data: str,
    media_type: str
) -> ChartData:
    """Multi-pass chart data extraction with verification."""
    client = anthropic.Anthropic()

    image_block = {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": media_type,
            "data": image_data
        }
    }

    # Pass 1: Structure identification
    struct_resp = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=512,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": (
                    "Identify the structure of this chart. Report ONLY:\n"
                    "1. Chart type\n"
                    "2. Title (if any)\n"
                    "3. X-axis label and categories/range\n"
                    "4. Y-axis label and range (min, max, step)\n"
                    "5. Number of data series\n"
                    "6. Legend entries (if any)\n"
                    "Be precise about axis ranges and steps."
                )},
                image_block
            ]
        }]
    )
    structure = struct_resp.content[0].text

    # Pass 2: Data extraction
    data_resp = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2048,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": (
                    f"Chart structure:\n{structure}\n\n"
                    "Now read each data point from the chart carefully. "
                    "For each data point:\n"
                    "1. Read the label/category from the x-axis\n"
                    "2. Estimate the value from the y-axis gridlines\n"
                    "3. Rate your confidence (high/medium/low)\n\n"
                    "Return JSON:\n"
                    '{"chart_type": "...", "title": "...", '
                    '"x_axis": "...", "y_axis": "...", '
                    '"data_points": [{"label": "...", "value": number, '
                    '"confidence": "high|medium|low"}]}'
                )},
                image_block
            ]
        },
        {"role": "assistant", "content": "{"}
        ]
    )

    try:
        data = json.loads("{" + data_resp.content[0].text)
    except json.JSONDecodeError:
        return ChartData(chart_type="unknown", title="", x_axis="", y_axis="")

    # Pass 3: Verification
    verify_resp = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": (
                    f"I extracted this data from the chart:\n"
                    f"{json.dumps(data['data_points'], indent=2)}\n\n"
                    "Look at the chart again and verify each value. "
                    "For each data point, check:\n"
                    "1. Does the bar/line/point visually match the value I read?\n"
                    "2. Is the value consistent with the y-axis scale?\n"
                    "3. Are relative comparisons correct (e.g., if A > B in the "
                    "   chart, is A > B in the data)?\n\n"
                    "Return JSON:\n"
                    '{"verified_points": [{"label": "...", "original_value": number, '
                    '"corrected_value": number, "status": "correct|adjusted|uncertain"}], '
                    '"notes": ["any observations"]}'
                )},
                image_block
            ]
        },
        {"role": "assistant", "content": "{"}
        ]
    )

    try:
        verification = json.loads("{" + verify_resp.content[0].text)
    except json.JSONDecodeError:
        verification = {"verified_points": [], "notes": ["Verification parse failed"]}

    # Build corrected result
    corrected_points = {}
    for vp in verification.get("verified_points", []):
        corrected_points[vp["label"]] = {
            "value": vp.get("corrected_value", vp.get("original_value")),
            "status": vp.get("status", "uncertain")
        }

    data_points = []
    for dp in data.get("data_points", []):
        correction = corrected_points.get(dp["label"])
        if correction:
            value = correction["value"]
            verified = correction["status"] == "correct"
            confidence = "high" if verified else dp.get("confidence", "medium")
        else:
            value = dp["value"]
            verified = False
            confidence = dp.get("confidence", "medium")

        data_points.append(DataPoint(
            label=dp["label"],
            value=value,
            confidence=confidence,
            verified=verified
        ))

    # Calculate overall confidence
    confidence_scores = {"high": 1.0, "medium": 0.6, "low": 0.3}
    if data_points:
        overall = sum(
            confidence_scores.get(dp.confidence, 0.5) for dp in data_points
        ) / len(data_points)
    else:
        overall = 0.0

    return ChartData(
        chart_type=data.get("chart_type", "unknown"),
        title=data.get("title", ""),
        x_axis=data.get("x_axis", ""),
        y_axis=data.get("y_axis", ""),
        data_points=data_points,
        overall_confidence=round(overall, 2),
        verification_notes=verification.get("notes", [])
    )


# Usage
# result = extract_chart_data_verified(chart_b64, "image/png")
# for dp in result.data_points:
#     v = "Y" if dp.verified else "N"
#     print(f"  {dp.label}: {dp.value} ({dp.confidence}, verified={v})")
# print(f"Overall confidence: {result.overall_confidence}")
```

</details>

### 연습문제 3: UI 접근성 감사기

Build a tool that takes a UI screenshot and produces a WCAG 2.1 AA compliance report. It should identify issues with color contrast, text size, touch target size, missing labels, and keyboard navigation indicators.

<details><summary>정답 보기</summary>

```python
import anthropic
import json
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class AccessibilityIssue:
    wcag_criterion: str  # e.g., "1.4.3 Contrast"
    severity: str  # critical, major, minor
    element: str
    description: str
    recommendation: str
    location: str  # Where in the image


@dataclass
class AccessibilityReport:
    page_type: str
    overall_score: str  # pass, partial, fail
    issues: list[AccessibilityIssue] = field(default_factory=list)
    positive_aspects: list[str] = field(default_factory=list)
    summary: str = ""


def audit_accessibility(
    image_data: str,
    media_type: str,
    context: str = ""
) -> AccessibilityReport:
    """Audit a UI screenshot for WCAG 2.1 AA compliance."""
    client = anthropic.Anthropic()

    tools = [{
        "name": "accessibility_report",
        "description": "Generate a WCAG 2.1 AA accessibility audit report",
        "input_schema": {
            "type": "object",
            "properties": {
                "page_type": {"type": "string"},
                "overall_score": {
                    "type": "string",
                    "enum": ["pass", "partial", "fail"]
                },
                "issues": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "wcag_criterion": {"type": "string"},
                            "severity": {
                                "type": "string",
                                "enum": ["critical", "major", "minor"]
                            },
                            "element": {"type": "string"},
                            "description": {"type": "string"},
                            "recommendation": {"type": "string"},
                            "location": {"type": "string"}
                        },
                        "required": [
                            "wcag_criterion", "severity", "element",
                            "description", "recommendation"
                        ]
                    }
                },
                "positive_aspects": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "summary": {"type": "string"}
            },
            "required": ["page_type", "overall_score", "issues", "summary"]
        }
    }]

    prompt = f"""Perform a thorough WCAG 2.1 AA accessibility audit on this UI screenshot.

{f"Context: {context}" if context else ""}

Check for these categories of issues:

1. COLOR CONTRAST (WCAG 1.4.3, 1.4.11):
   - Text contrast ratio (minimum 4.5:1 for normal text, 3:1 for large text)
   - Non-text contrast for UI components and graphics (minimum 3:1)
   - Any text on images or gradients that may fail contrast

2. TEXT AND READABILITY (WCAG 1.4.4, 1.4.12):
   - Text size (minimum 16px recommended for body text)
   - Line spacing and paragraph spacing
   - Text that may be cut off or overflow

3. TOUCH TARGETS (WCAG 2.5.8):
   - Interactive elements should be at least 44x44 CSS pixels
   - Adequate spacing between clickable elements

4. VISUAL INDICATORS (WCAG 1.4.1, 2.4.7):
   - Information conveyed by color alone (needs secondary indicator)
   - Focus indicators for keyboard navigation
   - Link styling (distinguishable from regular text)

5. FORM ELEMENTS (WCAG 1.3.1, 3.3.2):
   - Visible labels for all form fields
   - Error state indicators
   - Required field indicators
   - Placeholder text not used as the only label

6. STRUCTURAL (WCAG 1.3.1, 2.4.6):
   - Heading hierarchy (visual heading levels)
   - Logical reading order
   - Grouping of related content

For each issue found:
- Cite the specific WCAG criterion
- Rate severity (critical: blocks users, major: significant barrier, minor: inconvenience)
- Describe the element and location
- Provide a specific recommendation

Also note positive accessibility aspects you observe."""

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4096,
        tools=tools,
        tool_choice={"type": "tool", "name": "accessibility_report"},
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": image_data
                    }
                }
            ]
        }]
    )

    for block in response.content:
        if block.type == "tool_use":
            data = block.input
            issues = [AccessibilityIssue(**i) for i in data.get("issues", [])]
            return AccessibilityReport(
                page_type=data.get("page_type", "unknown"),
                overall_score=data.get("overall_score", "unknown"),
                issues=issues,
                positive_aspects=data.get("positive_aspects", []),
                summary=data.get("summary", "")
            )

    return AccessibilityReport(
        page_type="unknown",
        overall_score="error",
        summary="Audit failed to produce results"
    )


# Usage
# report = audit_accessibility(ui_screenshot_b64, "image/png", "Login page")
# print(f"Score: {report.overall_score}")
# print(f"Issues found: {len(report.issues)}")
# for issue in report.issues:
#     print(f"  [{issue.severity}] {issue.wcag_criterion}: {issue.description}")
```

</details>

### 연습문제 4: 다중 이미지 시각적 이야기 이해

Build a system that takes a series of images (e.g., comic panels, storyboard frames, photo sequence) and produces a narrative understanding: what is happening, the sequence of events, character relationships, and emotional arc.

<details><summary>정답 보기</summary>

```python
import anthropic
import json
from dataclasses import dataclass, field


@dataclass
class FrameAnalysis:
    frame_number: int
    description: str
    characters: list[str]
    actions: list[str]
    emotions: list[str]
    setting: str
    text_content: list[str]


@dataclass
class NarrativeAnalysis:
    title: str
    genre: str
    frames: list[FrameAnalysis] = field(default_factory=list)
    story_summary: str = ""
    character_arcs: dict[str, str] = field(default_factory=dict)
    emotional_arc: list[str] = field(default_factory=list)
    themes: list[str] = field(default_factory=list)


def analyze_visual_narrative(
    images: list[tuple[str, str]],  # (image_data, media_type)
    narrative_type: str = "sequential"  # sequential, parallel, flashback
) -> NarrativeAnalysis:
    """Analyze a series of images as a visual narrative."""
    client = anthropic.Anthropic()

    # Phase 1: Analyze each frame individually
    frame_analyses = []
    for i, (data, mtype) in enumerate(images):
        tools = [{
            "name": "analyze_frame",
            "description": "Analyze a single frame of a visual narrative",
            "input_schema": {
                "type": "object",
                "properties": {
                    "description": {"type": "string"},
                    "characters": {
                        "type": "array", "items": {"type": "string"}
                    },
                    "actions": {
                        "type": "array", "items": {"type": "string"}
                    },
                    "emotions": {
                        "type": "array", "items": {"type": "string"}
                    },
                    "setting": {"type": "string"},
                    "text_content": {
                        "type": "array", "items": {"type": "string"}
                    }
                },
                "required": ["description", "characters", "actions", "emotions", "setting"]
            }
        }]

        resp = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=512,
            tools=tools,
            tool_choice={"type": "tool", "name": "analyze_frame"},
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            f"This is frame {i+1} of {len(images)} in a visual sequence. "
                            f"Analyze what's happening in this frame."
                        )
                    },
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": mtype,
                            "data": data
                        }
                    }
                ]
            }]
        )

        for block in resp.content:
            if block.type == "tool_use":
                fa = FrameAnalysis(
                    frame_number=i + 1,
                    **block.input,
                    text_content=block.input.get("text_content", [])
                )
                frame_analyses.append(fa)

    # Phase 2: Synthesize the narrative (send all images together)
    content = [
        {"type": "text", "text": "Analyze these images as a visual narrative sequence:"}
    ]
    for i, (data, mtype) in enumerate(images):
        content.append({"type": "text", "text": f"Frame {i+1}:"})
        content.append({
            "type": "image",
            "source": {"type": "base64", "media_type": mtype, "data": data}
        })

    per_frame_summary = json.dumps(
        [{"frame": f.frame_number, "desc": f.description, "chars": f.characters}
         for f in frame_analyses],
        indent=2
    )

    content.append({
        "type": "text",
        "text": (
            f"\nPer-frame analysis:\n{per_frame_summary}\n\n"
            f"Narrative type: {narrative_type}\n\n"
            "Now synthesize the overall narrative:\n"
            "1. What story is being told across all frames?\n"
            "2. How do the characters develop?\n"
            "3. What is the emotional arc (how does the mood change)?\n"
            "4. What are the main themes?\n\n"
            "Return JSON with: title, genre, story_summary, "
            "character_arcs (name -> arc description), "
            "emotional_arc (list of mood per frame), themes (list)"
        )
    })

    narrative_resp = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2048,
        messages=[{"role": "user", "content": content}]
    )

    text = narrative_resp.content[0].text
    try:
        import re
        json_match = re.search(r"\{[\s\S]*\}", text)
        if json_match:
            narrative_data = json.loads(json_match.group())
        else:
            narrative_data = {}
    except json.JSONDecodeError:
        narrative_data = {}

    return NarrativeAnalysis(
        title=narrative_data.get("title", "Untitled Sequence"),
        genre=narrative_data.get("genre", "unknown"),
        frames=frame_analyses,
        story_summary=narrative_data.get("story_summary", ""),
        character_arcs=narrative_data.get("character_arcs", {}),
        emotional_arc=narrative_data.get("emotional_arc", []),
        themes=narrative_data.get("themes", [])
    )


# Usage
# images = [
#     (frame1_b64, "image/png"),
#     (frame2_b64, "image/png"),
#     (frame3_b64, "image/png"),
# ]
# narrative = analyze_visual_narrative(images)
# print(f"Title: {narrative.title}")
# print(f"Summary: {narrative.story_summary}")
# for name, arc in narrative.character_arcs.items():
#     print(f"  {name}: {arc}")
```

</details>

### 연습문제 5: 멀티모달 프롬프트 템플릿 라이브러리

Build a reusable library of multimodal prompt templates for common tasks. Each template should include: the task description, system prompt, user prompt with image placement, output format specification, and validation logic. Implement at least 4 templates and a runner that executes them.

<details><summary>정답 보기</summary>

```python
import anthropic
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Optional
from enum import Enum


class OutputFormat(Enum):
    TEXT = "text"
    JSON = "json"
    STRUCTURED = "structured"


@dataclass
class PromptTemplate:
    name: str
    description: str
    system_prompt: str
    user_prompt: str
    output_format: OutputFormat
    tool_schema: Optional[dict] = None
    tool_name: Optional[str] = None
    validators: list[Callable] = field(default_factory=list)
    required_params: list[str] = field(default_factory=list)


def validate_has_keys(result: dict, keys: list[str]) -> list[str]:
    """Validate that result has required keys."""
    missing = [k for k in keys if k not in result]
    return [f"Missing key: {k}" for k in missing]


def validate_non_empty_lists(result: dict, list_keys: list[str]) -> list[str]:
    """Validate that specified list fields are non-empty."""
    errors = []
    for k in list_keys:
        if k in result and isinstance(result[k], list) and len(result[k]) == 0:
            errors.append(f"Empty list: {k}")
    return errors


# Define the template library
TEMPLATE_LIBRARY = {
    "product_listing": PromptTemplate(
        name="product_listing",
        description="Extract product information for an e-commerce listing",
        system_prompt=(
            "You are a product data specialist. Extract accurate, "
            "detailed product information from images for e-commerce listings."
        ),
        user_prompt=(
            "Extract product details from this image for a {marketplace} listing.\n"
            "Category: {category}\n\n"
            "Be specific about colors (use standard color names), "
            "materials (if identifiable), and dimensions (if visible)."
        ),
        output_format=OutputFormat.STRUCTURED,
        tool_name="extract_product",
        tool_schema={
            "type": "object",
            "properties": {
                "product_name": {"type": "string"},
                "category": {"type": "string"},
                "brand": {"type": "string"},
                "colors": {"type": "array", "items": {"type": "string"}},
                "materials": {"type": "array", "items": {"type": "string"}},
                "features": {"type": "array", "items": {"type": "string"}},
                "condition": {"type": "string", "enum": ["new", "used", "refurbished"]},
                "description": {"type": "string"}
            },
            "required": ["product_name", "category", "colors", "description"]
        },
        validators=[
            lambda r: validate_has_keys(r, ["product_name", "colors"]),
            lambda r: validate_non_empty_lists(r, ["colors"])
        ],
        required_params=["marketplace", "category"]
    ),

    "food_nutrition": PromptTemplate(
        name="food_nutrition",
        description="Estimate nutritional information from a food photo",
        system_prompt=(
            "You are a nutritionist AI. Estimate nutritional content from "
            "food images. Be honest about uncertainty -- provide ranges "
            "when exact values are impossible to determine."
        ),
        user_prompt=(
            "Analyze this food image and estimate its nutritional content.\n"
            "Serving context: {serving_context}\n\n"
            "Identify each food item visible and estimate macronutrients."
        ),
        output_format=OutputFormat.STRUCTURED,
        tool_name="nutrition_estimate",
        tool_schema={
            "type": "object",
            "properties": {
                "food_items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "estimated_weight_g": {"type": "number"},
                            "calories": {"type": "number"},
                            "protein_g": {"type": "number"},
                            "carbs_g": {"type": "number"},
                            "fat_g": {"type": "number"}
                        }
                    }
                },
                "total_calories": {"type": "number"},
                "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
                "notes": {"type": "string"}
            },
            "required": ["food_items", "total_calories", "confidence"]
        },
        validators=[
            lambda r: validate_non_empty_lists(r, ["food_items"])
        ],
        required_params=["serving_context"]
    ),

    "ui_feedback": PromptTemplate(
        name="ui_feedback",
        description="Get design feedback on a UI screenshot",
        system_prompt=(
            "You are a senior UX designer with 15 years of experience. "
            "Provide constructive, actionable feedback."
        ),
        user_prompt=(
            "Review this UI design for a {app_type} application.\n"
            "Target audience: {audience}\n\n"
            "Provide feedback on visual design, usability, and information architecture."
        ),
        output_format=OutputFormat.STRUCTURED,
        tool_name="ui_feedback",
        tool_schema={
            "type": "object",
            "properties": {
                "overall_rating": {"type": "integer", "minimum": 1, "maximum": 10},
                "strengths": {"type": "array", "items": {"type": "string"}},
                "improvements": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "area": {"type": "string"},
                            "issue": {"type": "string"},
                            "suggestion": {"type": "string"},
                            "priority": {"type": "string", "enum": ["high", "medium", "low"]}
                        }
                    }
                },
                "design_principles_violated": {"type": "array", "items": {"type": "string"}},
                "summary": {"type": "string"}
            },
            "required": ["overall_rating", "strengths", "improvements", "summary"]
        },
        validators=[
            lambda r: validate_non_empty_lists(r, ["strengths", "improvements"])
        ],
        required_params=["app_type", "audience"]
    ),

    "scene_description": PromptTemplate(
        name="scene_description",
        description="Generate a detailed scene description for alt text or documentation",
        system_prompt=(
            "You write detailed, objective scene descriptions. "
            "Your descriptions should be useful for visually impaired users "
            "and for image search indexing."
        ),
        user_prompt=(
            "Write a comprehensive description of this image.\n"
            "Purpose: {purpose}\n"
            "Detail level: {detail_level}\n\n"
            "Include spatial relationships, colors, and any text visible."
        ),
        output_format=OutputFormat.TEXT,
        validators=[],
        required_params=["purpose", "detail_level"]
    )
}


@dataclass
class TemplateResult:
    template_name: str
    success: bool
    data: Any
    validation_errors: list[str] = field(default_factory=list)
    raw_response: str = ""


def run_template(
    template_name: str,
    image_data: str,
    media_type: str,
    **params
) -> TemplateResult:
    """Execute a template from the library."""
    if template_name not in TEMPLATE_LIBRARY:
        return TemplateResult(
            template_name=template_name,
            success=False,
            data=None,
            validation_errors=[f"Template '{template_name}' not found"]
        )

    template = TEMPLATE_LIBRARY[template_name]

    # Check required params
    missing = [p for p in template.required_params if p not in params]
    if missing:
        return TemplateResult(
            template_name=template_name,
            success=False,
            data=None,
            validation_errors=[f"Missing parameter: {p}" for p in missing]
        )

    client = anthropic.Anthropic()

    # Format user prompt
    user_text = template.user_prompt.format(**params)

    # Build message content
    content = [
        {"type": "text", "text": user_text},
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": media_type,
                "data": image_data
            }
        }
    ]

    # Prepare API call
    api_kwargs = {
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 2048,
        "system": template.system_prompt,
        "messages": [{"role": "user", "content": content}]
    }

    if template.output_format == OutputFormat.STRUCTURED and template.tool_schema:
        api_kwargs["tools"] = [{
            "name": template.tool_name,
            "description": template.description,
            "input_schema": template.tool_schema
        }]
        api_kwargs["tool_choice"] = {"type": "tool", "name": template.tool_name}

    response = client.messages.create(**api_kwargs)

    # Extract result
    if template.output_format == OutputFormat.STRUCTURED:
        data = None
        for block in response.content:
            if block.type == "tool_use":
                data = block.input
        if data is None:
            return TemplateResult(
                template_name=template_name,
                success=False,
                data=None,
                validation_errors=["No tool use in response"]
            )
    else:
        data = response.content[0].text

    # Run validators
    errors = []
    if isinstance(data, dict):
        for validator in template.validators:
            errors.extend(validator(data))

    return TemplateResult(
        template_name=template_name,
        success=len(errors) == 0,
        data=data,
        validation_errors=errors,
        raw_response=str(response.content)
    )


def list_templates() -> list[dict]:
    """List all available templates."""
    return [
        {
            "name": t.name,
            "description": t.description,
            "required_params": t.required_params,
            "output_format": t.output_format.value
        }
        for t in TEMPLATE_LIBRARY.values()
    ]


# Usage
# print("Available templates:")
# for t in list_templates():
#     print(f"  {t['name']}: {t['description']}")
#     print(f"    Params: {t['required_params']}")

# result = run_template(
#     "product_listing",
#     image_data=product_img_b64,
#     media_type="image/jpeg",
#     marketplace="Amazon",
#     category="Electronics"
# )
# if result.success:
#     print(json.dumps(result.data, indent=2))
# else:
#     print("Errors:", result.validation_errors)
```

</details>

---

**이전**: [멀티턴 대화](./07_Multi_Turn_Conversation.md) | **다음**: [코드 생성 프롬프팅](./09_Code_Generation_Prompting.md)
