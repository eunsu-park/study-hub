# 17. GPT-4V & Gemini

## 개요

GPT-4V(ision)와 Gemini는 현재 가장 강력한 상용 멀티모달 AI입니다. 이 레슨에서는 이들의 기능, API 사용법, 그리고 실전 응용 사례를 다룹니다.

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

## 2. Google Gemini

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

## 3. 비교 및 선택 가이드

### 3.1 GPT-4V vs Gemini 비교

```
┌────────────────────────────────────────────────────────────────┐
│              GPT-4V vs Gemini 비교                             │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  기능              GPT-4V              Gemini 1.5 Pro          │
│  ───────────────────────────────────────────────────────────  │
│  이미지 이해       ★★★★★            ★★★★★                │
│  비디오 분석       ✗ (프레임만)       ★★★★★ (네이티브)      │
│  오디오 분석       ✗                  ★★★★☆                │
│  컨텍스트 길이     128K               2M                       │
│  코드 실행         ✗                  ★★★★☆ (내장)         │
│  가격              높음               중간                     │
│  속도              중간               Flash는 빠름             │
│  일관성            높음               높음                     │
│                                                                │
│  추천 사용 사례:                                               │
│  - GPT-4V: 복잡한 이미지 추론, UI 코드 생성, 정밀 OCR         │
│  - Gemini: 비디오 분석, 초장문 문서, 멀티모달 복합 태스크     │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### 3.2 사용 사례별 선택

```python
def select_model(use_case: str) -> str:
    """사용 사례별 모델 선택"""

    recommendations = {
        # GPT-4V가 좋은 경우
        "ui_to_code": "gpt-4-vision-preview",
        "precise_ocr": "gpt-4-vision-preview",
        "image_reasoning": "gpt-4-vision-preview",
        "chart_analysis": "gpt-4-vision-preview",

        # Gemini가 좋은 경우
        "video_analysis": "gemini-1.5-pro",
        "long_document": "gemini-1.5-pro",
        "audio_transcription": "gemini-1.5-pro",
        "multimodal_app": "gemini-1.5-pro",

        # 비용 최적화
        "high_volume": "gemini-1.5-flash",
        "quick_caption": "gemini-1.5-flash",
    }

    return recommendations.get(use_case, "gpt-4-vision-preview")
```

---

## 4. 비용 최적화

### 4.1 비용 계산

```python
class CostEstimator:
    """API 비용 추정"""

    # 2024년 기준 가격 (USD)
    PRICING = {
        "gpt-4-vision-preview": {
            "input": 0.01,   # per 1K tokens
            "output": 0.03,  # per 1K tokens
            "image_low": 85,   # tokens
            "image_high": 765, # tokens (base) + tiles
        },
        "gemini-1.5-pro": {
            "input": 0.00125,  # per 1K chars
            "output": 0.00375,
            "image": 258,  # tokens per image
            "video": 263,  # tokens per second
            "audio": 32,   # tokens per second
        },
        "gemini-1.5-flash": {
            "input": 0.000125,
            "output": 0.000375,
        }
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
- [OpenAI GPT-4V Documentation](https://platform.openai.com/docs/guides/vision)
- [Google Gemini API](https://ai.google.dev/docs)

### 벤치마크
- [MMMU Benchmark](https://mmmu-benchmark.github.io/)
- [VQA Challenge](https://visualqa.org/)

### 관련 레슨
- [16_Vision_Language_Advanced.md](16_Vision_Language_Advanced.md)
- [24_API_Evaluation.md](24_API_Evaluation.md)
