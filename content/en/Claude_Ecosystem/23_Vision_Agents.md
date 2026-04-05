# Vision Agents

**Previous**: [22. Troubleshooting and Debugging](./22_Troubleshooting.md) | **Next**: [24. Prompt Caching and Batch API](./24_Prompt_Caching_and_Batch_API.md)

---

Claude's vision capabilities transform it from a text-only assistant into a multimodal agent that can see, interpret, and act on visual information. This lesson covers everything from sending a single image to building production-grade vision-powered agents that combine image understanding with tool use, computer control, and MCP integration.

**Difficulty**: ⭐⭐⭐

**Prerequisites**:
- Claude API fundamentals ([Lesson 15](./15_Claude_API_Fundamentals.md))
- Tool use and function calling ([Lesson 16](./16_Tool_Use_and_Function_Calling.md))
- Building custom agents ([Lesson 18](./18_Building_Custom_Agents.md))
- Model Context Protocol basics ([Lesson 12](./12_Model_Context_Protocol.md))

## Learning Objectives

After completing this lesson, you will be able to:

1. Send images to Claude via base64 encoding and URLs
2. Analyze and compare multiple images in a single request
3. Extract structured data from documents, screenshots, and diagrams
4. Build vision-powered agents that combine sight with action
5. Implement computer use for UI automation
6. Integrate vision capabilities with MCP servers
7. Optimize costs for vision-heavy workloads

---

## Table of Contents

1. [Claude's Vision Capabilities Overview](#1-claudes-vision-capabilities-overview)
2. [Sending Images via the Messages API](#2-sending-images-via-the-messages-api)
3. [Multi-Image Analysis and Comparison](#3-multi-image-analysis-and-comparison)
4. [Document Understanding](#4-document-understanding)
5. [Building Vision-Powered Agents](#5-building-vision-powered-agents)
6. [Image-Based Tool Use Patterns](#6-image-based-tool-use-patterns)
7. [Computer Use and UI Automation](#7-computer-use-and-ui-automation)
8. [Vision + MCP Server Integration](#8-vision--mcp-server-integration)
9. [Cost Optimization for Vision Workloads](#9-cost-optimization-for-vision-workloads)
10. [Exercises](#10-exercises)

---

## 1. Claude's Vision Capabilities Overview

Claude can process images natively as part of its multimodal input. Unlike OCR-then-text pipelines, Claude directly perceives the visual content, understanding layout, spatial relationships, charts, handwriting, and more.

### 1.1 Supported Formats and Limits

| Property | Details |
|---|---|
| **Supported formats** | JPEG, PNG, GIF, WebP |
| **Maximum image size** | 5 MB per image |
| **Maximum dimensions** | ~1568 px on the longest side (auto-resized) |
| **Images per request** | Up to 20 images |
| **Token cost** | Varies by resolution (see Section 9) |

### 1.2 What Claude Can See

Claude excels at:
- **Text extraction**: Printed text, handwriting, code in screenshots
- **Chart interpretation**: Bar charts, line graphs, pie charts, scatter plots
- **Diagram understanding**: Architecture diagrams, flowcharts, UML
- **Photo analysis**: Object recognition, scene description, spatial reasoning
- **UI understanding**: Button labels, form fields, navigation elements
- **Comparison**: Side-by-side image comparison and difference detection

### 1.3 Known Limitations

- **Spatial precision**: Claude may struggle with exact pixel coordinates
- **Small text**: Very small or low-contrast text may be missed
- **Counting**: Counting large numbers of objects is unreliable
- **Rotation**: Heavily rotated or upside-down text is harder to read
- **Medical/specialized imagery**: Not designed for clinical diagnosis

---

## 2. Sending Images via the Messages API

### 2.1 Base64 Encoding

The most common approach for local images:

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

### 2.2 URL-Based Images

For publicly accessible images, you can pass a URL directly:

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

### 2.3 Image Placement Best Practices

- Place images **before** the text prompt that references them
- For multiple images, order them logically (left-to-right, chronological)
- Reference images by position: "the first image", "the chart on the left"
- Keep text prompts specific: "Extract all email addresses from this screenshot" is better than "What's in this image?"

---

## 3. Multi-Image Analysis and Comparison

Claude can process up to 20 images in a single request, enabling powerful comparison and aggregation workflows.

### 3.1 Comparing Two Images

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


# Example: Compare before/after UI designs
result = compare_images(
    "design_v1.png",
    "design_v2.png",
    "Compare these two UI designs. List all visual differences "
    "including layout changes, color modifications, and element additions or removals.",
)
```

### 3.2 Batch Image Processing

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


# Example: Analyze a set of product screenshots
screenshots = sorted(glob.glob("screenshots/*.png"))[:20]  # Max 20 images
result = analyze_image_batch(
    screenshots,
    system_prompt="You are a UX auditor. Evaluate each screenshot for accessibility issues.",
    user_prompt="Review each screenshot and provide a numbered list of accessibility issues found.",
)
```

---

## 4. Document Understanding

Claude excels at extracting structured information from documents, PDFs rendered as images, and technical diagrams.

### 4.1 PDF Page Analysis

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

    # Process in batches of 20 pages
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


# Example: Extract tables from an invoice PDF
result = extract_from_pdf(
    "invoice.pdf",
    "Extract all line items as a JSON array with fields: description, quantity, unit_price, total.",
)
```

### 4.2 Architecture Diagram Interpretation

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

### 4.3 Screenshot Data Extraction

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

## 5. Building Vision-Powered Agents

A vision-powered agent combines Claude's sight with tool use, enabling it to observe its environment, make decisions, and take actions.

### 5.1 The See-Think-Act Loop

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

            # Collect all content blocks
            self.messages.append({"role": "assistant", "content": response.content})

            # If stop_reason is "end_turn", return the text
            if response.stop_reason == "end_turn":
                text_blocks = [b.text for b in response.content if b.type == "text"]
                return "\n".join(text_blocks)

            # Process tool calls
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

### 5.2 Example: Expense Report Agent

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


# Usage
agent = ExpenseAgent()
agent.observe("receipt_lunch.jpg", "Process this receipt.")
agent.observe("receipt_taxi.jpg", "Process this receipt too.")

for expense in agent.expenses:
    print(f"{expense['date']} | {expense['vendor']} | {expense['total']} {expense['currency']}")
```

---

## 6. Image-Based Tool Use Patterns

### 6.1 Visual Decision Making

Claude can use images to decide which tools to call:

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

# Claude sees a screenshot and decides: is this a bug, or does the design look correct?
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

### 6.2 Structured Extraction with Tool Use

Instead of asking for JSON in text, use tools for guaranteed structured output:

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

# Claude will extract data and call save_chart_data with structured output
```

---

## 7. Computer Use and UI Automation

Claude's computer use capability allows it to interact with graphical interfaces by taking screenshots, interpreting them, and executing mouse/keyboard actions.

### 7.1 Computer Use Overview

Computer use enables Claude to:
- View a computer screen via screenshots
- Move the mouse and click on elements
- Type text and use keyboard shortcuts
- Scroll, drag, and perform other UI actions

> **Note**: Computer use is currently in beta. It requires the `computer-use-2025-01-24` beta header and uses specific tool types.

### 7.2 Setting Up Computer Use

```python
import anthropic

client = anthropic.Anthropic()

# Computer use requires specific tool definitions
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

### 7.3 The Screenshot-Action Loop

```python
import subprocess


def take_screenshot() -> str:
    """Take a screenshot and return base64-encoded PNG."""
    # macOS example
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

        # Process tool calls
        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                if block.name == "computer":
                    result = execute_computer_action(block.input)
                    if result:  # Screenshot returned
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
                        # After action, take screenshot for next observation
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

### 7.4 Safety Considerations

- **Always run in sandboxed environments** (VMs, containers)
- **Never give computer use access to production systems**
- **Implement action allowlists** for what the agent can do
- **Add human-in-the-loop confirmation** for destructive actions
- **Set maximum step limits** to prevent runaway automation

---

## 8. Vision + MCP Server Integration

MCP servers can provide images to Claude or use Claude's vision to process images from external sources.

### 8.1 Image-Serving MCP Server

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
    # Fetch dashboard panel screenshot via Grafana render API
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

### 8.2 Vision-Enhanced Tool Server

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

## 9. Cost Optimization for Vision Workloads

Vision requests consume tokens based on image size. Understanding the cost model helps you build efficient pipelines.

### 9.1 Token Calculation

Claude resizes images to fit within a maximum dimension (~1568 px on the longest side) and then calculates token cost based on the number of tiles:

| Image Size | Approximate Tokens |
|---|---|
| 200x200 px | ~170 tokens |
| 800x600 px | ~800 tokens |
| 1568x1568 px | ~1,600 tokens |
| 4000x3000 px (resized) | ~1,600 tokens (same after resize) |

### 9.2 Cost Reduction Strategies

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

    # Resize if larger than needed
    if max(img.size) > max_dimension:
        ratio = max_dimension / max(img.size)
        new_size = (int(img.width * ratio), int(img.height * ratio))
        img = img.resize(new_size, Image.LANCZOS)

    # Convert to JPEG for better compression (unless transparency needed)
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

### 9.3 Architecture-Level Optimizations

1. **Pre-filter with cheaper models**: Use a lightweight classifier to decide which images need Claude's analysis
2. **Cache results**: Store extracted data to avoid re-processing identical images
3. **Batch strategically**: Send multiple related images in one request rather than separate calls
4. **Use appropriate models**: Claude Haiku for simple classification, Sonnet for detailed analysis
5. **Reduce resolution**: Many tasks (text extraction, chart reading) work fine at 800px

```python
# Model selection based on task complexity
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
        return "claude-sonnet-4-20250514"  # Use best available for complex tasks
```

---

## 10. Exercises

### Exercise 1: Receipt Scanner

Build a receipt scanning tool that:
1. Accepts a receipt image (photo or scan)
2. Extracts vendor name, date, line items, subtotal, tax, and total
3. Returns structured JSON
4. Handles multiple receipt formats

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

### Exercise 2: Visual Diff Reporter

Create a tool that compares two screenshots and generates a structured diff report:

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

### Exercise 3: Dashboard Monitor Agent

Build an agent that monitors a web dashboard by periodically taking screenshots and alerting on anomalies:

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

### Exercise 4: Multi-Page Document Processor

Create a pipeline that processes a multi-page PDF and extracts structured data from each page:

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

**Previous**: [22. Troubleshooting and Debugging](./22_Troubleshooting.md) | **Next**: [24. Prompt Caching and Batch API](./24_Prompt_Caching_and_Batch_API.md)
