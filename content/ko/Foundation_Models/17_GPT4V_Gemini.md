# 17. GPT-4V, GPT-4o, Gemini & Claude 3

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. GPT-4V, GPT-4o, Gemini 1.5 Pro, Claude 3의 핵심 기능과 한계를 상용 멀티모달 AI 시스템으로서 설명할 수 있다
2. 각 시스템의 API를 호출하여 이미지 이해, OCR, 시각적 추론(visual reasoning) 태스크를 수행할 수 있다
3. GPT-4o, Gemini 1.5 Pro, Claude 3 모델 패밀리 간의 멀티모달 아키텍처와 컨텍스트 윈도우(context window) 크기를 비교할 수 있다
4. 문서 분석 파이프라인과 시각적 질의응답(Visual Question Answering) 시스템 등 실용적인 멀티모달 애플리케이션을 설계할 수 있다
5. 프로덕션 사용 사례에서 상용 멀티모달 API를 선택할 때 비용, 지연 시간(latency), 기능 간 트레이드오프를 평가할 수 있다

---

## 개요

GPT-4V(ision), GPT-4o, Gemini, Claude 3는 현재 가장 강력한 상용 멀티모달 AI입니다. 이 레슨에서는 이들의 기능, API 사용법, 그리고 실전 응용 사례를 다룹니다.

> **2024년 업데이트**:
> - **GPT-4o** (2024.05): GPT-4의 "omni" 버전, 네이티브 멀티모달
> - **Gemini 1.5 Pro**: 2M 토큰 컨텍스트, 비디오/오디오 네이티브
> - **Claude 3 Family** (2024.03): Haiku, Sonnet, Opus 라인업
> - **Claude 3.5 Sonnet** (2024.06): 비전 기능 강화

---

## 1. GPT-4V (GPT-4 with Vision)

### 1.1 기능 개요

```
┌──────────────────────────────────────────────────────────────────┐
│                    GPT-4V 주요 기능                              │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  🖼️ 이미지 이해                                                  │
│  - 상세 설명 및 분석                                            │
│  - 다중 이미지 비교                                             │
│  - 차트/그래프 해석                                             │
│                                                                  │
│  📝 텍스트 인식 (OCR)                                            │
│  - 손글씨 인식                                                   │
│  - 다국어 텍스트                                                │
│  - 문서 구조 이해                                               │
│                                                                  │
│  🔍 세부 분석                                                    │
│  - 객체 식별 및 카운팅                                          │
│  - 공간 관계 이해                                               │
│  - 속성 추론                                                     │
│                                                                  │
│  💡 추론 및 창작                                                  │
│  - 이미지 기반 추론                                             │
│  - 코드 생성 (UI 스크린샷 → 코드)                               │
│  - 창의적 글쓰기                                                │
│                                                                  │
│  ⚠️ 제한 사항                                                    │
│  - 의료 진단 불가                                               │
│  - 얼굴 인식/신원 확인 불가                                     │
│  - 실시간 비디오 미지원 (이미지만)                              │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

### 1.2 API 사용법

```python
from openai import OpenAI
import base64
from pathlib import Path

client = OpenAI()

def encode_image(image_path: str) -> str:
    """이미지를 base64로 인코딩"""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def gpt4v_basic(image_path: str, prompt: str) -> str:
    """기본 이미지 분석"""

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
    """다중 이미지 분석"""

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
    상세 수준 지정

    detail:
    - "low": 빠르고 저렴, 저해상도 분석
    - "high": 상세 분석, 더 많은 토큰 사용
    - "auto": 자동 선택
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
    """URL 이미지 분석"""

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

### 1.3 실전 응용

```python
class GPT4VApplications:
    """GPT-4V 실전 응용"""

    def __init__(self):
        self.client = OpenAI()

    def analyze_ui_screenshot(self, screenshot_path: str) -> dict:
        """UI 스크린샷 분석 및 코드 생성"""

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

        # JSON 파싱
        import json
        try:
            return json.loads(response)
        except:
            return {"raw_response": response}

    def extract_data_from_chart(self, chart_path: str) -> dict:
        """차트에서 데이터 추출"""

        prompt = """Analyze this chart and extract:
        1. Chart type (bar, line, pie, etc.)
        2. Title and axis labels
        3. All data points with their values
        4. Key insights or trends

        Return as structured JSON.
        """

        return self._call_api(chart_path, prompt)

    def compare_images(self, image_paths: list) -> str:
        """이미지 비교 분석"""

        prompt = """Compare these images and describe:
        1. Similarities
        2. Differences
        3. Which image is better quality and why
        4. Any notable features in each
        """

        return gpt4v_multi_image(image_paths, prompt)

    def ocr_with_structure(self, document_path: str) -> dict:
        """구조화된 OCR"""

        prompt = """Extract all text from this document and preserve:
        1. Headings and hierarchy
        2. Tables (as markdown)
        3. Lists (numbered and bulleted)
        4. Key-value pairs

        Return as structured markdown.
        """

        return self._call_api(document_path, prompt)

    def generate_alt_text(self, image_path: str) -> str:
        """웹 접근성을 위한 대체 텍스트 생성"""

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

### 2.1 GPT-4o 개요

```
┌──────────────────────────────────────────────────────────────────┐
│                    GPT-4o vs GPT-4V 비교                         │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  GPT-4V (기존):                                                  │
│  - 텍스트 + 이미지 입력                                          │
│  - 별도의 비전 인코더                                            │
│  - 비교적 느린 응답                                              │
│                                                                  │
│  GPT-4o (2024.05):                                               │
│  - 텍스트 + 이미지 + 오디오 네이티브                             │
│  - 단일 모델에서 모든 모달리티 처리                              │
│  - 2배 빠른 응답, 50% 저렴한 가격                                │
│  - 실시간 음성 대화 가능                                         │
│                                                                  │
│  주요 개선점:                                                    │
│  ✅ 속도: 평균 320ms 응답 (GPT-4V 대비 2배)                      │
│  ✅ 비용: 입력 $5/1M, 출력 $15/1M                                │
│  ✅ 비전: 향상된 OCR, 차트 해석                                  │
│  ✅ 오디오: 실시간 음성 입출력                                   │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

### 2.2 GPT-4o API 사용법

```python
from openai import OpenAI
import base64

client = OpenAI()

def gpt4o_vision(image_path: str, prompt: str) -> str:
    """GPT-4o 이미지 분석"""

    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode()

    response = client.chat.completions.create(
        model="gpt-4o",  # GPT-4o 사용
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
    """GPT-4o 오디오 분석 (Realtime API)"""

    # 오디오 파일 읽기
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


# GPT-4o-mini: 저비용 버전
def gpt4o_mini_vision(image_path: str, prompt: str) -> str:
    """GPT-4o-mini: 빠르고 저렴한 비전 모델"""

    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode()

    response = client.chat.completions.create(
        model="gpt-4o-mini",  # 저비용 버전
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

### 2.1 Gemini 모델 라인업

```
┌──────────────────────────────────────────────────────────────────┐
│                    Gemini 모델 비교                              │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Gemini 1.5 Flash:                                              │
│  - 빠른 응답, 저비용                                            │
│  - 1M 토큰 컨텍스트                                             │
│  - 실시간 응용에 적합                                           │
│                                                                  │
│  Gemini 1.5 Pro:                                                │
│  - 최고 성능                                                    │
│  - 2M 토큰 컨텍스트                                             │
│  - 복잡한 추론, 코드 생성                                       │
│                                                                  │
│  Gemini 1.0 Ultra:                                              │
│  - 가장 큰 모델                                                 │
│  - 복잡한 멀티모달 태스크                                       │
│                                                                  │
│  특별 기능:                                                      │
│  - 네이티브 멀티모달 (텍스트, 이미지, 오디오, 비디오)           │
│  - 초장문 컨텍스트 (1시간 비디오 분석 가능)                     │
│  - Code execution 내장                                          │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

### 2.2 Gemini API 사용법

```python
import google.generativeai as genai
from PIL import Image

# API 키 설정
genai.configure(api_key="YOUR_API_KEY")

def gemini_basic(image_path: str, prompt: str) -> str:
    """기본 이미지 분석"""

    model = genai.GenerativeModel('gemini-1.5-pro')

    image = Image.open(image_path)

    response = model.generate_content([prompt, image])

    return response.text


def gemini_multi_image(image_paths: list, prompt: str) -> str:
    """다중 이미지 분석"""

    model = genai.GenerativeModel('gemini-1.5-pro')

    content = [prompt]
    for path in image_paths:
        content.append(Image.open(path))

    response = model.generate_content(content)

    return response.text


def gemini_video_analysis(video_path: str, prompt: str) -> str:
    """비디오 분석 (Gemini 특화 기능)"""

    model = genai.GenerativeModel('gemini-1.5-pro')

    # 비디오 업로드
    video_file = genai.upload_file(video_path)

    # 처리 완료 대기
    import time
    while video_file.state.name == "PROCESSING":
        time.sleep(10)
        video_file = genai.get_file(video_file.name)

    if video_file.state.name == "FAILED":
        raise ValueError("Video processing failed")

    response = model.generate_content([prompt, video_file])

    return response.text


def gemini_long_context(documents: list, query: str) -> str:
    """긴 문서 분석 (1M+ 토큰)"""

    model = genai.GenerativeModel('gemini-1.5-pro')

    # 모든 문서 결합
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
    """코드 실행 기능"""

    model = genai.GenerativeModel(
        'gemini-1.5-pro',
        tools='code_execution'
    )

    response = model.generate_content(prompt)

    # 실행된 코드와 결과 추출
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

### 2.3 Gemini 특화 응용

```python
class GeminiApplications:
    """Gemini 특화 응용"""

    def __init__(self):
        self.model = genai.GenerativeModel('gemini-1.5-pro')

    def analyze_long_video(
        self,
        video_path: str,
        questions: list
    ) -> dict:
        """긴 비디오 분석 (1시간+)"""

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
        """멀티모달 추론"""

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
        """연구 보조 (긴 문서 분석)"""

        # PDF 업로드
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
        """파일 업로드 및 처리 대기"""
        import time

        file = genai.upload_file(file_path)

        while file.state.name == "PROCESSING":
            time.sleep(5)
            file = genai.get_file(file.name)

        return file
```

---

## 4. Anthropic Claude 3

### 4.1 Claude 3 모델 라인업

```
┌──────────────────────────────────────────────────────────────────┐
│                    Claude 3 Family (2024.03)                     │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Claude 3 Haiku:                                                 │
│  - 가장 빠르고 저렴                                              │
│  - 실시간 응용, 대량 처리                                        │
│  - 비전 지원                                                     │
│                                                                  │
│  Claude 3 Sonnet:                                                │
│  - 속도와 성능의 균형                                            │
│  - 대부분의 비즈니스 용도에 적합                                 │
│  - 비전 지원                                                     │
│                                                                  │
│  Claude 3 Opus:                                                  │
│  - 최고 성능                                                     │
│  - 복잡한 추론, 분석 태스크                                      │
│  - 비전 지원                                                     │
│                                                                  │
│  Claude 3.5 Sonnet (2024.06):                                    │
│  - Opus 수준 성능, Sonnet 가격                                   │
│  - 향상된 비전, 코딩 능력                                        │
│  - 200K 토큰 컨텍스트                                            │
│                                                                  │
│  특징:                                                            │
│  ✅ 200K 컨텍스트 윈도우 (전 모델)                                │
│  ✅ 멀티모달: 이미지 이해                                         │
│  ✅ 안전성: Constitutional AI 적용                                │
│  ✅ 도구 사용: Function Calling 지원                              │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

### 4.2 Claude API 사용법

```python
import anthropic
import base64

client = anthropic.Anthropic()


def claude_vision(image_path: str, prompt: str, model: str = "claude-sonnet-4-6") -> str:
    """Claude 비전 분석"""

    # 이미지 인코딩
    with open(image_path, "rb") as f:
        image_data = base64.standard_b64encode(f.read()).decode("utf-8")

    # 미디어 타입 결정
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
    """Claude 다중 이미지 분석"""

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
        model="claude-sonnet-4-6",
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
        model="claude-sonnet-4-6",
        max_tokens=1024,
        tools=tools,
        messages=[{"role": "user", "content": content}],
    )

    return {
        "content": message.content,
        "stop_reason": message.stop_reason
    }
```

### 4.3 Claude 특화 기능

```python
class ClaudeApplications:
    """Claude 특화 응용"""

    def __init__(self):
        self.client = anthropic.Anthropic()

    def long_document_analysis(self, document_text: str, query: str) -> str:
        """긴 문서 분석 (200K 토큰)"""

        message = self.client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=4096,
            messages=[
                {
                    "role": "user",
                    "content": f"""다음 문서를 분석하고 질문에 답하세요.

문서:
{document_text}

질문: {query}
"""
                }
            ],
        )

        return message.content[0].text

    def code_review(self, code: str, language: str = "python") -> str:
        """코드 리뷰"""

        message = self.client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=2048,
            messages=[
                {
                    "role": "user",
                    "content": f"""다음 {language} 코드를 리뷰해주세요.

```{language}
{code}
```

다음을 포함해주세요:
1. 잠재적 버그
2. 성능 개선 사항
3. 코드 스타일 제안
4. 보안 문제
"""
                }
            ],
        )

        return message.content[0].text

    def structured_output(self, image_path: str, schema: dict) -> dict:
        """구조화된 출력 생성"""
        import json

        with open(image_path, "rb") as f:
            image_data = base64.standard_b64encode(f.read()).decode("utf-8")

        message = self.client.messages.create(
            model="claude-sonnet-4-6",
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
                            "text": f"""이 이미지를 분석하고 다음 JSON 스키마에 맞춰 결과를 반환하세요:

{json.dumps(schema, indent=2, ensure_ascii=False)}

JSON만 반환하세요."""
                        }
                    ]
                }
            ],
        )

        return json.loads(message.content[0].text)
```

---

## 5. 비교 및 선택 가이드

### 5.1 멀티모달 모델 비교

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    2024 멀티모달 모델 비교                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  기능            GPT-4o      Gemini 1.5 Pro   Claude 3.5 Sonnet            │
│  ─────────────────────────────────────────────────────────────────────────  │
│  이미지 이해     ★★★★★     ★★★★★         ★★★★★                    │
│  비디오 분석     ✗           ★★★★★ (네이티브) ✗                          │
│  오디오 분석     ★★★★☆     ★★★★☆         ✗                          │
│  컨텍스트        128K        2M               200K                         │
│  코드 실행       ✗           ★★★★☆ (내장)  ✗                          │
│  속도            ★★★★★     ★★★★☆ (Flash) ★★★★☆                    │
│  가격            중간        낮음             중간                         │
│  코딩 능력       ★★★★☆     ★★★★☆         ★★★★★                    │
│  추론 능력       ★★★★★     ★★★★☆         ★★★★★                    │
│                                                                             │
│  추천 사용 사례:                                                            │
│  - GPT-4o: 실시간 멀티모달, 음성 대화, 빠른 응답 필요 시                    │
│  - Gemini: 비디오 분석, 초장문 문서, 멀티모달 복합 태스크                   │
│  - Claude: 복잡한 추론, 코드 리뷰, 긴 문서 분석, 안전성 중요 시             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 5.2 사용 사례별 선택

```python
def select_model(use_case: str) -> str:
    """사용 사례별 모델 선택 (2024 업데이트)"""

    recommendations = {
        # GPT-4o가 좋은 경우
        "ui_to_code": "gpt-4o",
        "realtime_chat": "gpt-4o",
        "voice_assistant": "gpt-4o-audio-preview",
        "quick_vision": "gpt-4o",

        # Gemini가 좋은 경우
        "video_analysis": "gemini-1.5-pro",
        "very_long_document": "gemini-1.5-pro",  # 2M 컨텍스트
        "audio_transcription": "gemini-1.5-pro",
        "multimodal_app": "gemini-1.5-pro",

        # Claude가 좋은 경우
        "complex_reasoning": "claude-sonnet-4-6",
        "code_review": "claude-sonnet-4-6",
        "long_document": "claude-sonnet-4-6",  # 200K 컨텍스트
        "safety_critical": "claude-sonnet-4-6",

        # 비용 최적화
        "high_volume": "gemini-1.5-flash",
        "quick_caption": "gpt-4o-mini",
        "simple_classification": "claude-3-haiku-20240307",
    }

    return recommendations.get(use_case, "gpt-4o")
```

---

## 6. 비용 최적화

### 6.1 비용 계산

```python
class CostEstimator:
    """API 비용 추정"""

    # 2024년 기준 가격 (USD per 1M tokens)
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
        "claude-sonnet-4-6": {
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
        """GPT-4V 비용 추정"""

        pricing = self.PRICING["gpt-4-vision-preview"]

        # 이미지 토큰
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
        """Gemini 비용 추정"""

        pricing = self.PRICING[model]

        input_cost = text_chars / 1000 * pricing["input"]
        output_cost = output_chars / 1000 * pricing["output"]

        if model == "gemini-1.5-pro":
            # 멀티미디어 비용
            image_tokens = num_images * pricing["image"]
            video_tokens = video_seconds * pricing["video"]
            audio_tokens = audio_seconds * pricing["audio"]

            media_chars = (image_tokens + video_tokens + audio_tokens) * 4  # 토큰 → 문자 근사
            input_cost += media_chars / 1000 * pricing["input"]

        return input_cost + output_cost


# 사용 예시
estimator = CostEstimator()

# 100개 이미지 분석 비용 비교
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

## 참고 자료

### 공식 문서
- [OpenAI GPT-4o Documentation](https://platform.openai.com/docs/guides/vision)
- [Google Gemini API](https://ai.google.dev/docs)
- [Anthropic Claude Documentation](https://docs.anthropic.com/)

### 벤치마크
- [MMMU Benchmark](https://mmmu-benchmark.github.io/)
- [VQA Challenge](https://visualqa.org/)
- [LMSYS Chatbot Arena](https://chat.lmsys.org/)

### 관련 레슨
- [16_Vision_Language_Advanced.md](16_Vision_Language_Advanced.md)
- [24_API_Evaluation.md](24_API_Evaluation.md)

---

## 연습 문제

### 연습 문제 1: 프로덕션 사용 사례별 모델 선택
레슨의 비교 표를 사용하여 다음 프로덕션 사용 사례에 가장 적합한 모델을 선택하세요. 특정 기능 또는 비용 고려사항을 참조하여 선택을 정당화하세요.

| 사용 사례 | 선택된 모델 | 핵심 정당화 |
|----------|------------|------------|
| A) 손으로 쓴 의료 기록 전사 (하루 1만 페이지, 비용 민감) | ??? | ??? |
| B) 2시간 보안 카메라 영상에서 이상 탐지 | ??? | ??? |
| C) 고객 지원을 위한 실시간 음성 도우미 | ??? | ??? |
| D) 복잡한 다단계 추론을 포함한 법적 계약 검토 | ??? | ??? |
| E) 전자상거래 카탈로그 제품 설명 생성 (하루 10만 개 이미지) | ??? | ??? |

<details>
<summary>정답 보기</summary>

| 사용 사례 | 선택된 모델 | 핵심 정당화 |
|----------|------------|------------|
| A) 손으로 쓴 의료 기록 전사 | `gpt-4o-mini` 또는 `claude-3-haiku` | 높은 볼륨 + 비용 민감 → 가장 저렴하고 유능한 모델. 두 모델 모두 강력한 OCR 성능. GPT-4o-mini는 $0.15/1M 토큰으로, 페이지당 약 1K 토큰의 1만 페이지 = 하루 $1.50. 의료 맥락은 높은 정확도 필요, 배포 전 샘플링으로 검증 필요. |
| B) 2시간 보안 영상 분석 | `gemini-1.5-pro` | 네이티브 비디오 지원과 충분한 컨텍스트(2M 토큰)를 가진 **유일한 옵션**. 2시간 × 263 토큰/초 ≈ 190만 토큰 — Gemini 1.5 Pro의 2M 컨텍스트 내에 맞음. GPT-4o와 Claude 모두 비디오 입력을 네이티브로 지원하지 않음. |
| C) 실시간 음성 도우미 | `gpt-4o-audio-preview` | 네이티브 실시간 오디오 입출력과 평균 320ms 응답 시간을 가진 **유일한 옵션**. "omni" 모델은 별도의 음성-텍스트 변환 단계 없이 음성을 네이티브로 처리. |
| D) 법적 계약 검토 + 복잡한 추론 | `claude-sonnet-4-6` 또는 `claude-3-opus` | Claude가 추론과 코딩에서 최고 순위; Constitutional AI 훈련이 고위험 결정에 더 잘 보정됨. 200K 컨텍스트로 긴 계약서 처리 가능. 안전 중요 → Claude의 신중하고 미묘한 응답이 환각(hallucination) 위험 감소. |
| E) 전자상거래 설명 (하루 10만 이미지) | `gemini-1.5-flash` 또는 `gpt-4o-mini` | 가장 높은 볼륨 → 가장 저렴한 모델. Gemini 1.5 Flash($0.075/1M 입력 토큰)가 가장 저렴. 단순 설명 태스크는 최대 성능이 필요 없음 — 소규모로 먼저 품질 테스트. |

</details>

### 연습 문제 2: GPT-4V 이미지 토큰 비용 계산
GPT-4V는 `detail` 파라미터에 따라 다르게 청구됩니다. 다음 배치 작업의 총 API 비용을 계산하세요:

- 태스크: `detail="high"`로 500개 제품 이미지를 처리하여 구조화된 데이터 추출
- 이미지당 평균 프롬프트 길이: 200 토큰
- 이미지당 평균 응답 길이: 800 토큰
- GPT-4o 가격: 입력 $5.00/1M 토큰, 출력 $15.00/1M 토큰
- 고해상도 이미지: 기본 765 토큰 + 타일당 170 토큰 (각 이미지는 1024×1024 → 4개 타일 생성)

```python
# 계산:
# 1. 500개 이미지의 총 이미지 토큰
# 2. 총 프롬프트 토큰 (텍스트만)
# 3. 총 출력 토큰
# 4. USD 총 비용
```

<details>
<summary>정답 보기</summary>

```python
# 설정
num_images = 500
prompt_tokens_per_image = 200  # 텍스트 프롬프트 토큰
response_tokens_per_image = 800
gpt4o_input_price = 5.00 / 1_000_000   # 토큰당
gpt4o_output_price = 15.00 / 1_000_000  # 토큰당

# 고해상도 이미지 토큰 계산
# 고해상도 1024×1024:
#   기본 토큰: 765
#   타일: 1024/512 = 2 × 2 = 4타일, 각 512×512
#   타일 토큰: 4타일 × 170 토큰/타일 = 680
#   이미지당 총계: 765 + 680 = 1445 토큰
image_tokens_per_image = 765 + (4 * 170)  # = 1445
total_image_tokens = 500 * 1445  # = 722,500 토큰

# 텍스트 토큰
total_prompt_tokens = 500 * 200  # = 100,000 토큰
total_output_tokens = 500 * 800  # = 400,000 토큰

# 총 입력 토큰 = 이미지 + 텍스트 프롬프트
total_input_tokens = total_image_tokens + total_prompt_tokens
                   = 722,500 + 100,000 = 822,500 토큰

# 비용 계산
input_cost = 822,500 * (5.00 / 1_000_000) = $4.11
output_cost = 400,000 * (15.00 / 1_000_000) = $6.00

total_cost = $4.11 + $6.00 = 500개 이미지에 $10.11

# 이미지당 비용 분석:
cost_per_image = $10.11 / 500 = $0.020 per image

# 비교: detail="low" 사용 시
# 저해상도: 이미지당 85 토큰
low_detail_image_tokens = 500 * 85 = 42,500 토큰
low_detail_input_cost = (42,500 + 100,000) * (5.00 / 1_000_000) = $0.71
low_detail_output_cost = $6.00  # 동일 출력
low_detail_total = $6.71  # 34% 저렴하지만 품질 낮음
```

핵심 통찰: 이 배치 작업에서 출력 토큰이 비용을 지배합니다($10.11 중 $6.00 = 59%). 고해상도에서 저해상도로 전환하는 것보다 응답 길이를 줄이는 것이 비용 최적화에 더 효과적입니다.

</details>

### 연습 문제 3: 구조화된 출력을 위한 프롬프트 엔지니어링
제품 이미지에서 구조화된 데이터를 추출하는 강건한 Claude API 프롬프트를 설계하세요. 출력은 특정 스키마와 일치하는 유효한 JSON이어야 하며, 프롬프트는 엣지 케이스(edge case)를 우아하게 처리해야 합니다.

요구사항:
- 추출: 제품명, 브랜드, 가격(보이는 경우), 색상, 치수(보이는 경우), 눈에 보이는 결함
- 유효한 JSON 반환 (`json.loads()`로 파싱 가능)
- 정보가 보이지 않으면 데이터를 날조하지 말고 `null` 사용
- 각 추출된 필드에 대한 신뢰도 점수 (0-1)

```python
import anthropic
import json

def extract_product_data(image_path: str) -> dict:
    client = anthropic.Anthropic()

    # 여기에 프롬프트 설계
    prompt = """???"""

    # 구현
    pass
```

<details>
<summary>정답 보기</summary>

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

    prompt = """이 제품 이미지를 분석하고 구조화된 데이터를 추출하세요.

정확히 이 스키마와 일치하는 유효한 JSON 객체만 반환하세요:
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
  "visible_defects": array of strings (없으면 빈 배열),
  "defects_confidence": number (0.0-1.0)
}

규칙:
1. 이미지에서 확인할 수 없는 필드에는 null 사용 — 절대 날조하거나 추측하지 마세요
2. 신뢰도 점수는 각 필드가 얼마나 명확하게 보이는지/읽히는지를 반영합니다:
   - 1.0: 명확히 보이며 모호하지 않음
   - 0.7: 보이지만 부분적으로 가려지거나 추론 필요
   - 0.4: 맥락에서 추론, 직접 보이지 않음
   - null 필드 → 신뢰도 점수 0.0
3. JSON 객체만 반환하고 다른 텍스트 없음
4. 치수는 보이는 경우 단위 포함 (예: "30cm × 20cm × 10cm")"""

    message = client.messages.create(
        model="claude-sonnet-4-6",
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

    # 모델이 지시에도 불구하고 추가 텍스트를 추가하는 경우에도 JSON 추출
    json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
    if json_match:
        return json.loads(json_match.group())

    return json.loads(response_text)  # 직접 파싱을 폴백으로 시도
```

핵심 설계 결정:
- 프롬프트의 명시적 스키마는 모델이 필드 이름을 임의로 만드는 것을 방지합니다.
- 누락 데이터에 `null` 사용은 검증되지 않은 정보의 환각을 방지합니다.
- 신뢰도 점수는 다운스트림 로직이 언제 사람 검토를 위해 플래그를 세울지 결정할 수 있게 합니다(예: 신뢰도 < 0.6).
- 정규식(regex) 폴백은 모델이 지시에도 불구하고 서문 텍스트를 추가하는 경우를 처리합니다.
- 필드별 개별 신뢰도 점수가 단일 전체 신뢰도보다 더 유용합니다.

</details>

### 연습 문제 4: Gemini 긴 컨텍스트 비디오 분석 설계
Gemini 1.5 Pro를 사용하여 소매점의 8시간 감시 영상을 분석하는 프로덕션 시스템을 설계하세요. 시스템은 다음을 수행해야 합니다:
1. 절도 사건 탐지
2. 고객 흐름 패턴 추적
3. 피크 시간 식별
4. 일일 요약 보고서 생성

설계에서 2M 토큰 컨텍스트 제한, 비용 관리, 출력 신뢰성을 다루세요.

<details>
<summary>정답 보기</summary>

**아키텍처 설계**:

**문제**: 8시간 × 3600초 × ~263 토큰/초 ≈ 757만 토큰 — Gemini 1.5 Pro의 2M 컨텍스트를 3.8배 초과.

**해결책: 핵심 프레임 샘플링을 사용한 슬라이딩 윈도우**:

```python
import google.generativeai as genai
from datetime import datetime, timedelta

class SurveillanceAnalyzer:

    def __init__(self):
        self.model = genai.GenerativeModel('gemini-1.5-pro')
        self.segment_duration = 90 * 60  # 90분 세그먼트 (2M 컨텍스트 내)

    def analyze_day(self, video_path: str) -> dict:
        """8시간 영상을 세그먼트로 처리"""
        results = {
            'shoplifting_incidents': [],
            'customer_flow': [],
            'peak_hours': [],
            'summary': ''
        }

        # 10분 오버랩으로 90분 세그먼트로 분할
        # (오버랩은 세그먼트 경계에서 사건을 놓치지 않도록 방지)
        segments = self._split_video(video_path, segment_minutes=90, overlap_minutes=10)

        for i, (segment_path, start_time) in enumerate(segments):
            segment_result = self._analyze_segment(segment_path, start_time, i)
            self._merge_results(results, segment_result)

        # 집계된 데이터로 최종 합성 프롬프트
        results['summary'] = self._generate_summary(results)

        return results

    def _analyze_segment(self, video_path: str, start_time: datetime, segment_idx: int) -> dict:
        """단일 90분 세그먼트 분석"""

        video_file = self._upload_and_wait(video_path)

        prompt = f"""이 소매점 감시 영상 세그먼트를 분석하세요
        (세그먼트 {segment_idx+1}, {start_time.strftime('%H:%M')} 시작).

        다음을 식별하세요:
        1. 절도(SHOPLIFTING): 의심스러운 행동 (물건 숨기기, 계산대 우회).
           각 사건에 대해: 타임스탬프, 프레임 내 위치, 설명, 신뢰도 (HIGH/MEDIUM/LOW)

        2. 고객_흐름(CUSTOMER_FLOW): 15분 간격의 대략적인 고객 수.
           형식: [{{time: "HH:MM", count: N}}]

        3. 이상징후(ANOMALIES): 기타 주목할 만한 이벤트.

        JSON으로 반환하며 키는: shoplifting_incidents, customer_flow, anomalies.
        절도의 경우 HIGH 또는 MEDIUM 신뢰도 사건만 보고하세요.
        """

        response = self.model.generate_content(
            [prompt, video_file],
            generation_config={"temperature": 0.1}  # 사실적 분석을 위한 낮은 온도
        )

        return self._parse_response(response.text, start_time)

    def _generate_summary(self, results: dict) -> str:
        """집계된 결과에서 최종 보고서 생성"""

        summary_prompt = f"""오늘의 소매점 감시 분석을 기반으로:

        - 탐지된 사건: {len(results['shoplifting_incidents'])}
        - 추적된 총 고객 수: {sum(h['count'] for h in results['customer_flow'])}
        - 피크 시간대: {self._find_peak(results['customer_flow'])}

        매장 관리자를 위한 간결한 일일 보안 및 운영 보고서를 작성하세요.
        직원 배치 조정 및 보안 집중 영역에 대한 권장 사항을 포함하세요."""

        # 텍스트 전용 최종 합성 (비디오 재업로드 불필요)
        response = self.model.generate_content(summary_prompt)
        return response.text
```

**비용 관리**:
- 8시간 × 263 토큰/초 = 757만 입력 토큰
- $1.25/1M 토큰으로 = 비디오 처리에 하루 약 $9.46
- 텍스트 출력 추가: ~$5.00/1M × 약 2만 출력 토큰 = 약 $0.10
- 총계 ≈ $9.56/일 — 프레임 레이트 감소(원본 대신 1fps)로 50-75% 비용 절감 가능

**신뢰성 개선**:
- 사실적 분석을 위한 낮은 온도(0.1)로 날조된 사건 감소.
- HIGH/MEDIUM 신뢰도 사건만 보고하여 거짓 양성(false positive) 감소.
- 10분 오버랩 세그먼트로 경계에서 사건을 놓치지 않도록 보장.
- 최종 합성 단계(텍스트 전용)로 보고서 생성을 위한 비디오 재업로드 방지.

</details>
