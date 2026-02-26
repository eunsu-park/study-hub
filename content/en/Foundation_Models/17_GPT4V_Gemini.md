# 17. GPT-4V, GPT-4o, Gemini & Claude 3

## Learning Objectives

After completing this lesson, you will be able to:

1. Describe the key capabilities and limitations of GPT-4V, GPT-4o, Gemini 1.5 Pro, and Claude 3 as commercial multimodal AI systems
2. Implement API calls to each of these systems to perform image understanding, OCR, and visual reasoning tasks
3. Compare the multimodal architectures and context window sizes across GPT-4o, Gemini 1.5 Pro, and Claude 3 model families
4. Design practical multimodal applications such as document analysis pipelines and visual question answering systems
5. Evaluate the cost, latency, and capability trade-offs when selecting a commercial multimodal API for a production use case

---

## Overview

GPT-4V(ision), GPT-4o, Gemini, and Claude 3 are currently the most powerful commercial multimodal AI systems. This lesson covers their features, API usage, and practical applications.

> **2024 Updates**:
> - **GPT-4o** (May 2024): "omni" version of GPT-4, native multimodal
> - **Gemini 1.5 Pro**: 2M token context, native video/audio
> - **Claude 3 Family** (March 2024): Haiku, Sonnet, Opus lineup
> - **Claude 3.5 Sonnet** (June 2024): Enhanced vision capabilities

---

## 1. GPT-4V (GPT-4 with Vision)

### 1.1 Feature Overview

```
┌──────────────────────────────────────────────────────────────────┐
│                    GPT-4V Key Features                           │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  🖼️ Image Understanding                                          │
│  - Detailed description and analysis                            │
│  - Multi-image comparison                                       │
│  - Chart/graph interpretation                                   │
│                                                                  │
│  📝 Text Recognition (OCR)                                       │
│  - Handwriting recognition                                      │
│  - Multilingual text                                            │
│  - Document structure understanding                             │
│                                                                  │
│  🔍 Detailed Analysis                                            │
│  - Object identification and counting                           │
│  - Spatial relationship understanding                           │
│  - Attribute reasoning                                          │
│                                                                  │
│  💡 Reasoning and Creation                                       │
│  - Image-based reasoning                                        │
│  - Code generation (UI screenshot → code)                       │
│  - Creative writing                                             │
│                                                                  │
│  ⚠️ Limitations                                                  │
│  - No medical diagnosis                                         │
│  - No face recognition/identity verification                    │
│  - No real-time video (images only)                             │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

### 1.2 API Usage

```python
from openai import OpenAI
import base64
from pathlib import Path

client = OpenAI()

def encode_image(image_path: str) -> str:
    """Encode image to base64"""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def gpt4v_basic(image_path: str, prompt: str) -> str:
    """Basic image analysis"""

    response = client.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{encode_image(image_path)}"
                        }
                    }
                ]
            }
        ],
        max_tokens=1024
    )

    return response.choices[0].message.content


def gpt4v_multi_image(image_paths: list, prompt: str) -> str:
    """Multi-image analysis"""

    content = [{"type": "text", "text": prompt}]

    for path in image_paths:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{encode_image(path)}"}
        })

    response = client.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=[{"role": "user", "content": content}],
        max_tokens=2048
    )

    return response.choices[0].message.content


def gpt4v_with_detail(image_path: str, prompt: str, detail: str = "high") -> str:
    """
    Specify detail level

    detail:
    - "low": fast and cheap, low-resolution analysis
    - "high": detailed analysis, more tokens used
    - "auto": automatic selection
    """

    response = client.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{encode_image(image_path)}",
                            "detail": detail
                        }
                    }
                ]
            }
        ],
        max_tokens=1024
    )

    return response.choices[0].message.content


def gpt4v_url_image(image_url: str, prompt: str) -> str:
    """Analyze image from URL"""

    response = client.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_url}}
                ]
            }
        ],
        max_tokens=1024
    )

    return response.choices[0].message.content
```

### 1.3 Practical Applications

```python
class GPT4VApplications:
    """GPT-4V practical applications"""

    def __init__(self):
        self.client = OpenAI()

    def analyze_ui_screenshot(self, screenshot_path: str) -> dict:
        """UI screenshot analysis and code generation"""

        prompt = """Analyze this UI screenshot and:
        1. List all UI components visible
        2. Describe the layout structure
        3. Generate HTML/CSS code to recreate this UI

        Format your response as JSON with keys:
        - components: list of UI elements
        - layout: description of layout
        - html_code: HTML implementation
        - css_code: CSS styles
        """

        response = self._call_api(screenshot_path, prompt)

        # Parse JSON
        import json
        try:
            return json.loads(response)
        except:
            return {"raw_response": response}

    def extract_data_from_chart(self, chart_path: str) -> dict:
        """Extract data from charts"""

        prompt = """Analyze this chart and extract:
        1. Chart type (bar, line, pie, etc.)
        2. Title and axis labels
        3. All data points with their values
        4. Key insights or trends

        Return as structured JSON.
        """

        return self._call_api(chart_path, prompt)

    def compare_images(self, image_paths: list) -> str:
        """Image comparison analysis"""

        prompt = """Compare these images and describe:
        1. Similarities
        2. Differences
        3. Which image is better quality and why
        4. Any notable features in each
        """

        return gpt4v_multi_image(image_paths, prompt)

    def ocr_with_structure(self, document_path: str) -> dict:
        """Structured OCR"""

        prompt = """Extract all text from this document and preserve:
        1. Headings and hierarchy
        2. Tables (as markdown)
        3. Lists (numbered and bulleted)
        4. Key-value pairs

        Return as structured markdown.
        """

        return self._call_api(document_path, prompt)

    def generate_alt_text(self, image_path: str) -> str:
        """Generate alt text for web accessibility"""

        prompt = """Generate an appropriate alt text for this image.
        The alt text should be:
        1. Concise (under 125 characters)
        2. Descriptive of the main content
        3. Useful for screen reader users

        Just return the alt text, nothing else.
        """

        return self._call_api(image_path, prompt)

    def _call_api(self, image_path: str, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{encode_image(image_path)}",
                                "detail": "high"
                            }
                        }
                    ]
                }
            ],
            max_tokens=2048
        )
        return response.choices[0].message.content
```

---

## 2. GPT-4o (Omni)

### 2.1 GPT-4o Overview

```
┌──────────────────────────────────────────────────────────────────┐
│                    GPT-4o vs GPT-4V Comparison                   │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  GPT-4V (Previous):                                              │
│  - Text + image input                                            │
│  - Separate vision encoder                                       │
│  - Relatively slower response                                    │
│                                                                  │
│  GPT-4o (May 2024):                                              │
│  - Text + image + audio native                                   │
│  - Single model handles all modalities                           │
│  - 2x faster response, 50% cheaper                               │
│  - Real-time voice conversation                                  │
│                                                                  │
│  Key Improvements:                                                │
│  ✅ Speed: Average 320ms response (2x faster than GPT-4V)        │
│  ✅ Cost: $5/1M input, $15/1M output                             │
│  ✅ Vision: Improved OCR, chart interpretation                   │
│  ✅ Audio: Real-time voice input/output                          │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

### 2.2 GPT-4o API Usage

```python
from openai import OpenAI
import base64

client = OpenAI()

def gpt4o_vision(image_path: str, prompt: str) -> str:
    """GPT-4o image analysis"""

    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode()

    response = client.chat.completions.create(
        model="gpt-4o",  # Use GPT-4o
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_data}",
                            "detail": "high"
                        }
                    }
                ]
            }
        ],
        max_tokens=1024
    )

    return response.choices[0].message.content


def gpt4o_audio(audio_path: str, prompt: str) -> str:
    """GPT-4o audio analysis (Realtime API)"""

    # Read audio file
    with open(audio_path, "rb") as f:
        audio_data = base64.b64encode(f.read()).decode()

    response = client.chat.completions.create(
        model="gpt-4o-audio-preview",
        modalities=["text"],
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": audio_data,
                            "format": "wav"
                        }
                    }
                ]
            }
        ]
    )

    return response.choices[0].message.content


# GPT-4o-mini: Low-cost version
def gpt4o_mini_vision(image_path: str, prompt: str) -> str:
    """GPT-4o-mini: Fast and cheap vision model"""

    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode()

    response = client.chat.completions.create(
        model="gpt-4o-mini",  # Low-cost version
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}
                    }
                ]
            }
        ],
        max_tokens=512
    )

    return response.choices[0].message.content
```

---

## 3. Google Gemini

### 3.1 Gemini Model Lineup

```
┌──────────────────────────────────────────────────────────────────┐
│                    Gemini Model Comparison                       │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Gemini 1.5 Flash:                                              │
│  - Fast response, low cost                                      │
│  - 1M token context                                             │
│  - Suitable for real-time applications                          │
│                                                                  │
│  Gemini 1.5 Pro:                                                │
│  - Best performance                                             │
│  - 2M token context                                             │
│  - Complex reasoning, code generation                           │
│                                                                  │
│  Gemini 1.0 Ultra:                                              │
│  - Largest model                                                │
│  - Complex multimodal tasks                                     │
│                                                                  │
│  Special Features:                                               │
│  - Native multimodal (text, image, audio, video)               │
│  - Ultra-long context (1 hour video analysis)                  │
│  - Built-in code execution                                      │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

### 3.2 Gemini API Usage

```python
import google.generativeai as genai
from PIL import Image

# Configure API key
genai.configure(api_key="YOUR_API_KEY")

def gemini_basic(image_path: str, prompt: str) -> str:
    """Basic image analysis"""

    model = genai.GenerativeModel('gemini-1.5-pro')

    image = Image.open(image_path)

    response = model.generate_content([prompt, image])

    return response.text


def gemini_multi_image(image_paths: list, prompt: str) -> str:
    """Multi-image analysis"""

    model = genai.GenerativeModel('gemini-1.5-pro')

    content = [prompt]
    for path in image_paths:
        content.append(Image.open(path))

    response = model.generate_content(content)

    return response.text


def gemini_video_analysis(video_path: str, prompt: str) -> str:
    """Video analysis (Gemini specialized feature)"""

    model = genai.GenerativeModel('gemini-1.5-pro')

    # Upload video
    video_file = genai.upload_file(video_path)

    # Wait for processing
    import time
    while video_file.state.name == "PROCESSING":
        time.sleep(10)
        video_file = genai.get_file(video_file.name)

    if video_file.state.name == "FAILED":
        raise ValueError("Video processing failed")

    response = model.generate_content([prompt, video_file])

    return response.text


def gemini_long_context(documents: list, query: str) -> str:
    """Long document analysis (1M+ tokens)"""

    model = genai.GenerativeModel('gemini-1.5-pro')

    # Combine all documents
    content = [query]
    for doc in documents:
        if doc.endswith('.pdf'):
            content.append(genai.upload_file(doc))
        elif doc.endswith(('.jpg', '.png')):
            content.append(Image.open(doc))
        else:
            with open(doc, 'r') as f:
                content.append(f.read())

    response = model.generate_content(content)

    return response.text


def gemini_with_code_execution(prompt: str) -> dict:
    """Code execution feature"""

    model = genai.GenerativeModel(
        'gemini-1.5-pro',
        tools='code_execution'
    )

    response = model.generate_content(prompt)

    # Extract executed code and results
    result = {
        'text': response.text,
        'code_execution': []
    }

    for part in response.parts:
        if hasattr(part, 'code_execution_result'):
            result['code_execution'].append({
                'code': part.text,
                'output': part.code_execution_result.output
            })

    return result
```

### 3.3 Gemini Specialized Applications

```python
class GeminiApplications:
    """Gemini specialized applications"""

    def __init__(self):
        self.model = genai.GenerativeModel('gemini-1.5-pro')

    def analyze_long_video(
        self,
        video_path: str,
        questions: list
    ) -> dict:
        """Long video analysis (1 hour+)"""

        video_file = self._upload_and_wait(video_path)

        results = {}

        for question in questions:
            prompt = f"""Analyze this video and answer: {question}

            Provide timestamps when relevant.
            """

            response = self.model.generate_content([prompt, video_file])
            results[question] = response.text

        return results

    def multimodal_reasoning(
        self,
        images: list,
        audio_path: str = None,
        text: str = None
    ) -> str:
        """Multimodal reasoning"""

        content = []

        if text:
            content.append(text)

        for img_path in images:
            content.append(Image.open(img_path))

        if audio_path:
            audio_file = self._upload_and_wait(audio_path)
            content.append(audio_file)

        response = self.model.generate_content(content)

        return response.text

    def research_assistant(
        self,
        pdf_paths: list,
        research_question: str
    ) -> dict:
        """Research assistant (long document analysis)"""

        # Upload PDFs
        files = [self._upload_and_wait(path) for path in pdf_paths]

        prompt = f"""You are a research assistant. Analyze these academic papers
        and answer the following research question:

        {research_question}

        Structure your response as:
        1. Summary of relevant findings from each paper
        2. Synthesis of the findings
        3. Gaps or contradictions
        4. Suggested future directions
        """

        content = [prompt] + files

        response = self.model.generate_content(content)

        return {
            'answer': response.text,
            'sources': pdf_paths
        }

    def _upload_and_wait(self, file_path: str):
        """Upload file and wait for processing"""
        import time

        file = genai.upload_file(file_path)

        while file.state.name == "PROCESSING":
            time.sleep(5)
            file = genai.get_file(file.name)

        return file
```

---

## 4. Anthropic Claude 3

### 4.1 Claude 3 Model Lineup

```
┌──────────────────────────────────────────────────────────────────┐
│                    Claude 3 Family (March 2024)                  │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Claude 3 Haiku:                                                 │
│  - Fastest and cheapest                                          │
│  - Real-time applications, high-volume processing                │
│  - Vision support                                                │
│                                                                  │
│  Claude 3 Sonnet:                                                │
│  - Balance of speed and performance                              │
│  - Suitable for most business use cases                          │
│  - Vision support                                                │
│                                                                  │
│  Claude 3 Opus:                                                  │
│  - Highest performance                                           │
│  - Complex reasoning, analysis tasks                             │
│  - Vision support                                                │
│                                                                  │
│  Claude 3.5 Sonnet (June 2024):                                  │
│  - Opus-level performance at Sonnet pricing                      │
│  - Enhanced vision, coding capabilities                          │
│  - 200K token context                                            │
│                                                                  │
│  Features:                                                        │
│  ✅ 200K context window (all models)                              │
│  ✅ Multimodal: Image understanding                               │
│  ✅ Safety: Constitutional AI applied                             │
│  ✅ Tool use: Function Calling support                            │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

### 4.2 Claude API Usage

```python
import anthropic
import base64

client = anthropic.Anthropic()


def claude_vision(image_path: str, prompt: str, model: str = "claude-sonnet-4-20250514") -> str:
    """Claude vision analysis"""

    # Encode image
    with open(image_path, "rb") as f:
        image_data = base64.standard_b64encode(f.read()).decode("utf-8")

    # Determine media type
    if image_path.endswith(".png"):
        media_type = "image/png"
    elif image_path.endswith(".gif"):
        media_type = "image/gif"
    elif image_path.endswith(".webp"):
        media_type = "image/webp"
    else:
        media_type = "image/jpeg"

    message = client.messages.create(
        model=model,
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
                        "text": prompt
                    }
                ],
            }
        ],
    )

    return message.content[0].text


def claude_multi_image(image_paths: list, prompt: str) -> str:
    """Claude multi-image analysis"""

    content = []

    for path in image_paths:
        with open(path, "rb") as f:
            image_data = base64.standard_b64encode(f.read()).decode("utf-8")

        media_type = "image/png" if path.endswith(".png") else "image/jpeg"

        content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": media_type,
                "data": image_data,
            }
        })

    content.append({"type": "text", "text": prompt})

    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2048,
        messages=[{"role": "user", "content": content}],
    )

    return message.content[0].text


def claude_with_tools(prompt: str, image_path: str = None) -> dict:
    """Claude Tool Use (Function Calling)"""

    tools = [
        {
            "name": "get_weather",
            "description": "Get current weather for a location",
            "input_schema": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City name"
                    }
                },
                "required": ["location"]
            }
        }
    ]

    content = [{"type": "text", "text": prompt}]

    if image_path:
        with open(image_path, "rb") as f:
            image_data = base64.standard_b64encode(f.read()).decode("utf-8")
        content.insert(0, {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": image_data,
            }
        })

    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        tools=tools,
        messages=[{"role": "user", "content": content}],
    )

    return {
        "content": message.content,
        "stop_reason": message.stop_reason
    }
```

### 4.3 Claude Specialized Features

```python
class ClaudeApplications:
    """Claude specialized applications"""

    def __init__(self):
        self.client = anthropic.Anthropic()

    def long_document_analysis(self, document_text: str, query: str) -> str:
        """Long document analysis (200K tokens)"""

        message = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            messages=[
                {
                    "role": "user",
                    "content": f"""Analyze the following document and answer the question.

Document:
{document_text}

Question: {query}
"""
                }
            ],
        )

        return message.content[0].text

    def code_review(self, code: str, language: str = "python") -> str:
        """Code review"""

        message = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2048,
            messages=[
                {
                    "role": "user",
                    "content": f"""Please review the following {language} code.

```{language}
{code}
```

Include:
1. Potential bugs
2. Performance improvements
3. Code style suggestions
4. Security issues
"""
                }
            ],
        )

        return message.content[0].text

    def structured_output(self, image_path: str, schema: dict) -> dict:
        """Generate structured output"""
        import json

        with open(image_path, "rb") as f:
            image_data = base64.standard_b64encode(f.read()).decode("utf-8")

        message = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2048,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": image_data,
                            }
                        },
                        {
                            "type": "text",
                            "text": f"""Analyze this image and return results matching the following JSON schema:

{json.dumps(schema, indent=2)}

Return only JSON."""
                        }
                    ]
                }
            ],
        )

        return json.loads(message.content[0].text)
```

---

## 5. Comparison and Selection Guide

### 5.1 Multimodal Model Comparison

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    2024 Multimodal Model Comparison                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Feature            GPT-4o      Gemini 1.5 Pro   Claude 3.5 Sonnet          │
│  ─────────────────────────────────────────────────────────────────────────  │
│  Image Understanding ★★★★★     ★★★★★         ★★★★★                  │
│  Video Analysis      ✗           ★★★★★ (native) ✗                        │
│  Audio Analysis      ★★★★☆     ★★★★☆         ✗                        │
│  Context             128K        2M               200K                       │
│  Code Execution      ✗           ★★★★☆ (built-in) ✗                       │
│  Speed               ★★★★★     ★★★★☆ (Flash)  ★★★★☆                  │
│  Price               Medium      Low              Medium                     │
│  Coding Ability      ★★★★☆     ★★★★☆         ★★★★★                  │
│  Reasoning Ability   ★★★★★     ★★★★☆         ★★★★★                  │
│                                                                             │
│  Recommended Use Cases:                                                      │
│  - GPT-4o: Real-time multimodal, voice chat, fast response needed           │
│  - Gemini: Video analysis, ultra-long docs, multimodal complex tasks        │
│  - Claude: Complex reasoning, code review, long doc analysis, safety-critical│
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 5.2 Use Case Selection

```python
def select_model(use_case: str) -> str:
    """Select model by use case (2024 update)"""

    recommendations = {
        # GPT-4o is better for
        "ui_to_code": "gpt-4o",
        "realtime_chat": "gpt-4o",
        "voice_assistant": "gpt-4o-audio-preview",
        "quick_vision": "gpt-4o",

        # Gemini is better for
        "video_analysis": "gemini-1.5-pro",
        "very_long_document": "gemini-1.5-pro",  # 2M context
        "audio_transcription": "gemini-1.5-pro",
        "multimodal_app": "gemini-1.5-pro",

        # Claude is better for
        "complex_reasoning": "claude-sonnet-4-20250514",
        "code_review": "claude-sonnet-4-20250514",
        "long_document": "claude-sonnet-4-20250514",  # 200K context
        "safety_critical": "claude-sonnet-4-20250514",

        # Cost optimization
        "high_volume": "gemini-1.5-flash",
        "quick_caption": "gpt-4o-mini",
        "simple_classification": "claude-3-haiku-20240307",
    }

    return recommendations.get(use_case, "gpt-4o")
```

---

## 6. Cost Optimization

### 6.1 Cost Calculation

```python
class CostEstimator:
    """API cost estimation"""

    # 2024 pricing (USD per 1M tokens)
    PRICING = {
        "gpt-4-vision-preview": {
            "input": 10.0,   # per 1M tokens
            "output": 30.0,  # per 1M tokens
            "image_low": 85,   # tokens
            "image_high": 765, # tokens (base) + tiles
        },
        "gpt-4o": {
            "input": 5.0,    # per 1M tokens
            "output": 15.0,  # per 1M tokens
            "image_low": 85,
            "image_high": 765,
        },
        "gpt-4o-mini": {
            "input": 0.15,   # per 1M tokens
            "output": 0.60,  # per 1M tokens
            "image_low": 85,
            "image_high": 765,
        },
        "gemini-1.5-pro": {
            "input": 1.25,   # per 1M tokens
            "output": 5.0,
            "image": 258,  # tokens per image
            "video": 263,  # tokens per second
            "audio": 32,   # tokens per second
        },
        "gemini-1.5-flash": {
            "input": 0.075,
            "output": 0.30,
        },
        "claude-3-opus": {
            "input": 15.0,   # per 1M tokens
            "output": 75.0,
        },
        "claude-sonnet-4-20250514": {
            "input": 3.0,    # per 1M tokens
            "output": 15.0,
        },
        "claude-3-haiku": {
            "input": 0.25,   # per 1M tokens
            "output": 1.25,
        },
    }

    def estimate_gpt4v_cost(
        self,
        num_images: int,
        avg_prompt_tokens: int,
        avg_response_tokens: int,
        detail: str = "high"
    ) -> float:
        """Estimate GPT-4V cost"""

        pricing = self.PRICING["gpt-4-vision-preview"]

        # Image tokens
        if detail == "low":
            image_tokens = num_images * pricing["image_low"]
        else:
            image_tokens = num_images * pricing["image_high"]

        total_input = avg_prompt_tokens + image_tokens
        total_output = avg_response_tokens

        cost = (total_input / 1000 * pricing["input"] +
                total_output / 1000 * pricing["output"])

        return cost

    def estimate_gemini_cost(
        self,
        num_images: int = 0,
        video_seconds: int = 0,
        audio_seconds: int = 0,
        text_chars: int = 0,
        output_chars: int = 0,
        model: str = "gemini-1.5-pro"
    ) -> float:
        """Estimate Gemini cost"""

        pricing = self.PRICING[model]

        input_cost = text_chars / 1000 * pricing["input"]
        output_cost = output_chars / 1000 * pricing["output"]

        if model == "gemini-1.5-pro":
            # Multimedia cost
            image_tokens = num_images * pricing["image"]
            video_tokens = video_seconds * pricing["video"]
            audio_tokens = audio_seconds * pricing["audio"]

            media_chars = (image_tokens + video_tokens + audio_tokens) * 4  # Token → char approximation
            input_cost += media_chars / 1000 * pricing["input"]

        return input_cost + output_cost


# Usage example
estimator = CostEstimator()

# Compare cost for 100 image analysis
gpt4v_cost = estimator.estimate_gpt4v_cost(
    num_images=100,
    avg_prompt_tokens=100,
    avg_response_tokens=500,
    detail="high"
)

gemini_cost = estimator.estimate_gemini_cost(
    num_images=100,
    text_chars=500,
    output_chars=2000,
    model="gemini-1.5-pro"
)

print(f"GPT-4V cost: ${gpt4v_cost:.2f}")
print(f"Gemini Pro cost: ${gemini_cost:.2f}")
```

---

## References

### Official Documentation
- [OpenAI GPT-4o Documentation](https://platform.openai.com/docs/guides/vision)
- [Google Gemini API](https://ai.google.dev/docs)
- [Anthropic Claude Documentation](https://docs.anthropic.com/)

### Benchmarks
- [MMMU Benchmark](https://mmmu-benchmark.github.io/)
- [VQA Challenge](https://visualqa.org/)
- [LMSYS Chatbot Arena](https://chat.lmsys.org/)

### Related Lessons
- [16_Vision_Language_Advanced.md](16_Vision_Language_Advanced.md)
- [24_API_Evaluation.md](24_API_Evaluation.md)

---

## Exercises

### Exercise 1: Model Selection for Production Use Cases
Using the comparison table from the lesson, select the most appropriate model for each of the following production use cases. Justify your choice by referencing specific capabilities or cost considerations.

| Use Case | Selected Model | Key Justification |
|----------|---------------|-------------------|
| A) Transcribe handwritten medical records (10K pages/day, cost-sensitive) | ??? | ??? |
| B) Analyze 2-hour security camera footage to detect anomalies | ??? | ??? |
| C) Real-time voice assistant for customer support | ??? | ??? |
| D) Legal contract review with complex multi-step reasoning | ??? | ??? |
| E) Generate product descriptions for an e-commerce catalog (100K images/day) | ??? | ??? |

<details>
<summary>Show Answer</summary>

| Use Case | Selected Model | Key Justification |
|----------|---------------|-------------------|
| A) Transcribe handwritten medical records | `gpt-4o-mini` or `claude-3-haiku` | High volume + cost sensitivity → cheapest capable model. Both have strong OCR. At $0.15/1M tokens (GPT-4o-mini), 10K pages at ~1K tokens/page = $1.50/day. Medical context requires high accuracy, so verify with sampling before deployment. |
| B) 2-hour security footage analysis | `gemini-1.5-pro` | **Only option** with native video support and sufficient context (2M tokens). 2 hours × 263 tokens/second ≈ 1.9M tokens — fits within Gemini 1.5 Pro's 2M context. Neither GPT-4o nor Claude supports video input natively. |
| C) Real-time voice assistant | `gpt-4o-audio-preview` | **Only option** with native real-time audio input/output and 320ms average response time. The "omni" model handles voice natively without a separate speech-to-text step. |
| D) Legal contract review + complex reasoning | `claude-sonnet-4-20250514` or `claude-3-opus` | Claude ranks highest in reasoning and coding; Constitutional AI training makes it better calibrated for high-stakes decisions. 200K context handles long contracts. Safety-critical → Claude's careful, nuanced responses reduce hallucination risk. |
| E) E-commerce descriptions (100K images/day) | `gemini-1.5-flash` or `gpt-4o-mini` | Highest volume → lowest cost model. Gemini 1.5 Flash ($0.075/1M input tokens) is the cheapest. Simple descriptive task doesn't require maximum capability — test quality at small scale first. |

</details>

### Exercise 2: GPT-4V Image Token Cost Calculation
GPT-4V charges differently based on the `detail` parameter. Calculate the total API cost for the following batch job:

- Task: Process 500 product images with `detail="high"` to extract structured data
- Average prompt length: 200 tokens per image
- Average response length: 800 tokens per image
- GPT-4o pricing: $5.00/1M input tokens, $15.00/1M output tokens
- High-detail image: 765 base tokens + 170 tokens per tile (each image is 1024×1024 → generates 4 tiles)

```python
# Calculate:
# 1. Total image tokens for 500 images
# 2. Total prompt tokens (text only)
# 3. Total output tokens
# 4. Total cost in USD
```

<details>
<summary>Show Answer</summary>

```python
# Setup
num_images = 500
prompt_tokens_per_image = 200  # text prompt tokens
response_tokens_per_image = 800
gpt4o_input_price = 5.00 / 1_000_000   # per token
gpt4o_output_price = 15.00 / 1_000_000  # per token

# Image token calculation for high detail
# 1024×1024 at high detail:
#   Base tokens: 765
#   Tiles: 1024/512 = 2 × 2 = 4 tiles, each 512×512
#   Tile tokens: 4 tiles × 170 tokens/tile = 680
#   Total per image: 765 + 680 = 1445 tokens
image_tokens_per_image = 765 + (4 * 170)  # = 1445
total_image_tokens = 500 * 1445  # = 722,500 tokens

# Text tokens
total_prompt_tokens = 500 * 200  # = 100,000 tokens
total_output_tokens = 500 * 800  # = 400,000 tokens

# Total input tokens = image + text prompt
total_input_tokens = total_image_tokens + total_prompt_tokens
                   = 722,500 + 100,000 = 822,500 tokens

# Cost calculation
input_cost = 822,500 * (5.00 / 1_000_000) = $4.11
output_cost = 400,000 * (15.00 / 1_000_000) = $6.00

total_cost = $4.11 + $6.00 = $10.11 for 500 images

# Per-image cost breakdown:
cost_per_image = $10.11 / 500 = $0.020 per image

# Comparison: using detail="low" instead
# Low detail: 85 tokens per image
low_detail_image_tokens = 500 * 85 = 42,500 tokens
low_detail_input_cost = (42,500 + 100,000) * (5.00 / 1_000_000) = $0.71
low_detail_output_cost = $6.00  # same output
low_detail_total = $6.71  # 34% cheaper, but lower quality
```

Key insight: For this batch job, output tokens dominate the cost ($6.00 of $10.11 = 59%). Reducing response length is more impactful than switching from high to low detail for cost optimization.

</details>

### Exercise 3: Prompt Engineering for Structured Output
Design a robust API prompt for Claude that extracts structured data from a product image. The output must be valid JSON matching a specific schema, and the prompt must handle edge cases gracefully.

Requirements:
- Extract: product name, brand, price (if visible), color, dimensions (if visible), any visible defects
- Return valid JSON (parseable with `json.loads()`)
- If information is not visible, use `null` rather than fabricating data
- Confidence score (0-1) for each extracted field

```python
import anthropic
import json

def extract_product_data(image_path: str) -> dict:
    client = anthropic.Anthropic()

    # Design the prompt here
    prompt = """???"""

    # Your implementation
    pass
```

<details>
<summary>Show Answer</summary>

```python
import anthropic
import json
import base64
import re

def extract_product_data(image_path: str) -> dict:
    client = anthropic.Anthropic()

    with open(image_path, "rb") as f:
        image_data = base64.standard_b64encode(f.read()).decode("utf-8")

    media_type = "image/jpeg"
    if image_path.endswith(".png"):
        media_type = "image/png"

    prompt = """Analyze this product image and extract structured data.

Return ONLY a valid JSON object with exactly this schema:
{
  "product_name": string or null,
  "product_name_confidence": number (0.0-1.0),
  "brand": string or null,
  "brand_confidence": number (0.0-1.0),
  "price": string or null,
  "price_confidence": number (0.0-1.0),
  "color": string or null,
  "color_confidence": number (0.0-1.0),
  "dimensions": string or null,
  "dimensions_confidence": number (0.0-1.0),
  "visible_defects": array of strings (empty array if none),
  "defects_confidence": number (0.0-1.0)
}

Rules:
1. Use null for fields you cannot determine from the image — NEVER fabricate or guess
2. Confidence scores reflect how clearly visible/readable each field is:
   - 1.0: Clearly visible and unambiguous
   - 0.7: Visible but partially obscured or requires inference
   - 0.4: Inferred from context, not directly visible
   - null field → confidence score of 0.0
3. Return ONLY the JSON object, no other text
4. Dimensions should include units if visible (e.g., "30cm × 20cm × 10cm")"""

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

    response_text = message.content[0].text

    # Extract JSON even if model adds extra text despite instructions
    json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
    if json_match:
        return json.loads(json_match.group())

    return json.loads(response_text)  # Try direct parse as fallback
```

Key design decisions:
- Explicit schema in the prompt prevents the model from inventing field names.
- `null` for missing data prevents hallucination of unverifiable information.
- Confidence scores enable downstream logic to decide when to flag for human review (e.g., confidence < 0.6).
- Regex fallback handles cases where the model adds preamble text despite instructions.
- Separate confidence score per field is more useful than a single overall confidence.

</details>

### Exercise 4: Gemini Long-Context Video Analysis Design
Design a production system that uses Gemini 1.5 Pro to analyze 8-hour surveillance footage from a retail store. The system must:
1. Detect shoplifting incidents
2. Track customer flow patterns
3. Identify peak hours
4. Generate an end-of-day summary report

Address the 2M token context limit, cost management, and output reliability in your design.

<details>
<summary>Show Answer</summary>

**Architecture Design**:

**Challenge**: 8 hours × 3600 seconds × ~263 tokens/second ≈ 7.57M tokens — exceeds Gemini 1.5 Pro's 2M context by 3.8x.

**Solution: Sliding window with key-frame sampling**:

```python
import google.generativeai as genai
from datetime import datetime, timedelta

class SurveillanceAnalyzer:

    def __init__(self):
        self.model = genai.GenerativeModel('gemini-1.5-pro')
        self.segment_duration = 90 * 60  # 90-minute segments (fits in 2M context)

    def analyze_day(self, video_path: str) -> dict:
        """Process 8-hour footage in segments"""
        results = {
            'shoplifting_incidents': [],
            'customer_flow': [],
            'peak_hours': [],
            'summary': ''
        }

        # Split into 90-minute segments with 10-minute overlaps
        # (overlap prevents missing incidents at segment boundaries)
        segments = self._split_video(video_path, segment_minutes=90, overlap_minutes=10)

        for i, (segment_path, start_time) in enumerate(segments):
            segment_result = self._analyze_segment(segment_path, start_time, i)
            self._merge_results(results, segment_result)

        # Final synthesis prompt with aggregated data
        results['summary'] = self._generate_summary(results)

        return results

    def _analyze_segment(self, video_path: str, start_time: datetime, segment_idx: int) -> dict:
        """Analyze a single 90-minute segment"""

        video_file = self._upload_and_wait(video_path)

        prompt = f"""Analyze this retail surveillance footage segment
        (Segment {segment_idx+1}, starting at {start_time.strftime('%H:%M')}).

        Identify:
        1. SHOPLIFTING: Any suspicious behavior (concealing items, bypassing checkout).
           For each incident: timestamp, location in frame, description, confidence (HIGH/MEDIUM/LOW)

        2. CUSTOMER_FLOW: Approximate customer count in 15-minute intervals.
           Format: [{{time: "HH:MM", count: N}}]

        3. ANOMALIES: Any other noteworthy events.

        Return as JSON with keys: shoplifting_incidents, customer_flow, anomalies.
        Only report HIGH or MEDIUM confidence incidents for shoplifting.
        """

        response = self.model.generate_content(
            [prompt, video_file],
            generation_config={"temperature": 0.1}  # Low temp for factual analysis
        )

        return self._parse_response(response.text, start_time)

    def _generate_summary(self, results: dict) -> str:
        """Generate final report from aggregated results"""

        summary_prompt = f"""Based on today's retail surveillance analysis:

        - Detected incidents: {len(results['shoplifting_incidents'])}
        - Total customers tracked: {sum(h['count'] for h in results['customer_flow'])}
        - Peak period: {self._find_peak(results['customer_flow'])}

        Write a concise end-of-day security and operations report for store management.
        Include recommendations for staffing adjustments and security focus areas."""

        # Text-only final synthesis (no video upload needed)
        response = self.model.generate_content(summary_prompt)
        return response.text
```

**Cost management**:
- 8 hours at 263 tokens/second = 7.57M input tokens
- At $1.25/1M tokens = ~$9.46 per day for video processing
- Plus text output: ~$5.00/1M × estimated 20K output tokens = ~$0.10
- Total ≈ $9.56/day — may justify reducing frame rate (1 fps instead of original) to cut costs by 50-75%

**Reliability improvements**:
- Low temperature (0.1) for factual analysis reduces hallucinated incidents.
- Only report HIGH/MEDIUM confidence incidents to reduce false positives.
- 10-minute overlapping segments ensure incidents at boundaries are not missed.
- Final synthesis step (text-only) avoids re-uploading video for report generation.

</details>
