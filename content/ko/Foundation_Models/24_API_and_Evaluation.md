# 24. API & 평가

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 상용 LLM API 제공자(OpenAI, Anthropic, Google)를 가격, 컨텍스트 길이, 기능 측면에서 비교할 수 있다
2. 속도 제한(rate limiting), 재시도(retry), 토큰 카운팅(token counting), 비용 추적(cost tracking)을 처리하는 견고한 API 클라이언트 래퍼를 구현할 수 있다
3. 프로덕션 애플리케이션에서 API 비용을 최적화하기 위해 프롬프트 엔지니어링(prompt engineering) 기법과 캐싱(caching) 전략을 적용할 수 있다
4. 표준 벤치마크(MMLU, HellaSwag, HumanEval)를 사용하여 LLM 성능을 평가하고 각 벤치마크가 측정하는 것을 설명할 수 있다
5. 자동화된 벤치마크, LLM-as-a-judge, 인간 평가를 결합한 포괄적인 LLM 평가 파이프라인을 설계할 수 있다

---

## 개요

상용 LLM API 사용법과 비용 최적화, 그리고 LLM 성능 평가를 위한 벤치마크와 방법론을 다룹니다.

---

## 1. 상용 LLM API

### 1.1 주요 제공자 비교

```
API 제공자 비교 (2024):
┌────────────────────────────────────────────────────────────────┐
│  Provider    │ Model          │ Input/1M  │ Output/1M │ Context │
├──────────────┼────────────────┼───────────┼───────────┼─────────┤
│  OpenAI      │ GPT-4 Turbo    │ $10       │ $30       │ 128K    │
│              │ GPT-4o         │ $5        │ $15       │ 128K    │
│              │ GPT-3.5 Turbo  │ $0.50     │ $1.50     │ 16K     │
├──────────────┼────────────────┼───────────┼───────────┼─────────┤
│  Anthropic   │ Claude 3 Opus  │ $15       │ $75       │ 200K    │
│              │ Claude 3 Sonnet│ $3        │ $15       │ 200K    │
│              │ Claude 3 Haiku │ $0.25     │ $1.25     │ 200K    │
├──────────────┼────────────────┼───────────┼───────────┼─────────┤
│  Google      │ Gemini 1.5 Pro │ $3.50     │ $10.50    │ 1M      │
│              │ Gemini 1.5 Flash│ $0.35    │ $1.05     │ 1M      │
└────────────────────────────────────────────────────────────────┘
```

### 1.2 OpenAI API

```python
from openai import OpenAI
import tiktoken

class OpenAIClient:
    """OpenAI API 클라이언트"""

    def __init__(self, api_key: str = None):
        self.client = OpenAI(api_key=api_key)
        self.token_encoder = tiktoken.get_encoding("cl100k_base")

    def chat(
        self,
        messages: list,
        model: str = "gpt-4o",
        temperature: float = 0.7,
        max_tokens: int = 1000,
        **kwargs
    ) -> dict:
        """채팅 완성"""
        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )

        return {
            "content": response.choices[0].message.content,
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            },
            "model": response.model,
            "finish_reason": response.choices[0].finish_reason
        }

    def stream_chat(self, messages: list, model: str = "gpt-4o", **kwargs):
        """스트리밍 채팅"""
        stream = self.client.chat.completions.create(
            model=model,
            messages=messages,
            stream=True,
            **kwargs
        )

        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    def count_tokens(self, text: str) -> int:
        """토큰 수 계산"""
        return len(self.token_encoder.encode(text))

    def estimate_cost(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        model: str = "gpt-4o"
    ) -> float:
        """비용 추정"""
        pricing = {
            "gpt-4o": {"input": 5.0, "output": 15.0},
            "gpt-4-turbo": {"input": 10.0, "output": 30.0},
            "gpt-3.5-turbo": {"input": 0.5, "output": 1.5}
        }

        if model not in pricing:
            return 0.0

        cost = (
            prompt_tokens * pricing[model]["input"] / 1_000_000 +
            completion_tokens * pricing[model]["output"] / 1_000_000
        )

        return cost


# Function calling
def function_calling_example():
    client = OpenAI()

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather for a location",
                "parameters": {
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
        }
    ]

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "What's the weather in Seoul?"}],
        tools=tools,
        tool_choice="auto"
    )

    # Tool call 처리
    if response.choices[0].message.tool_calls:
        tool_call = response.choices[0].message.tool_calls[0]
        print(f"Function: {tool_call.function.name}")
        print(f"Arguments: {tool_call.function.arguments}")
```

### 1.3 Anthropic API

```python
from anthropic import Anthropic

class AnthropicClient:
    """Anthropic Claude API 클라이언트"""

    def __init__(self, api_key: str = None):
        self.client = Anthropic(api_key=api_key)

    def chat(
        self,
        messages: list,
        model: str = "claude-3-sonnet-20240229",
        max_tokens: int = 1000,
        system: str = None,
        **kwargs
    ) -> dict:
        """채팅"""
        response = self.client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system or "You are a helpful assistant.",
            messages=messages,
            **kwargs
        )

        return {
            "content": response.content[0].text,
            "usage": {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens
            },
            "model": response.model,
            "stop_reason": response.stop_reason
        }

    def stream_chat(self, messages: list, **kwargs):
        """스트리밍"""
        with self.client.messages.stream(
            messages=messages,
            **kwargs
        ) as stream:
            for text in stream.text_stream:
                yield text

    def vision(
        self,
        image_url: str,
        prompt: str,
        model: str = "claude-3-sonnet-20240229"
    ) -> str:
        """비전 API"""
        import base64
        import httpx

        # 이미지 로드
        if image_url.startswith("http"):
            image_data = base64.standard_b64encode(
                httpx.get(image_url).content
            ).decode("utf-8")
        else:
            with open(image_url, "rb") as f:
                image_data = base64.standard_b64encode(f.read()).decode("utf-8")

        response = self.client.messages.create(
            model=model,
            max_tokens=1024,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": image_data
                        }
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }]
        )

        return response.content[0].text
```

### 1.4 Google Gemini API

```python
import google.generativeai as genai

class GeminiClient:
    """Google Gemini API 클라이언트"""

    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.5-pro')

    def chat(
        self,
        messages: list,
        temperature: float = 0.7,
        max_tokens: int = 1000
    ) -> dict:
        """채팅"""
        # OpenAI 형식을 Gemini 형식으로 변환
        history = []
        for msg in messages[:-1]:
            role = "user" if msg["role"] == "user" else "model"
            history.append({"role": role, "parts": [msg["content"]]})

        chat = self.model.start_chat(history=history)
        response = chat.send_message(
            messages[-1]["content"],
            generation_config=genai.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_tokens
            )
        )

        return {
            "content": response.text,
            "usage": {
                "prompt_tokens": response.usage_metadata.prompt_token_count,
                "completion_tokens": response.usage_metadata.candidates_token_count
            }
        }

    def multimodal(
        self,
        image_path: str,
        prompt: str
    ) -> str:
        """멀티모달 입력"""
        import PIL.Image

        img = PIL.Image.open(image_path)
        response = self.model.generate_content([prompt, img])

        return response.text
```

---

## 2. 비용 최적화

### 2.1 비용 모니터링

```python
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List
import json

@dataclass
class UsageRecord:
    """API 사용 기록"""
    timestamp: datetime
    model: str
    prompt_tokens: int
    completion_tokens: int
    cost: float
    request_type: str = "chat"

class CostTracker:
    """비용 추적기"""

    def __init__(self):
        self.records: List[UsageRecord] = []
        self.pricing = {
            "gpt-4o": {"input": 5.0, "output": 15.0},
            "gpt-4-turbo": {"input": 10.0, "output": 30.0},
            "gpt-3.5-turbo": {"input": 0.5, "output": 1.5},
            "claude-3-opus": {"input": 15.0, "output": 75.0},
            "claude-3-sonnet": {"input": 3.0, "output": 15.0},
            "claude-3-haiku": {"input": 0.25, "output": 1.25}
        }

    def log_request(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        request_type: str = "chat"
    ):
        """요청 로깅"""
        cost = self._calculate_cost(model, prompt_tokens, completion_tokens)

        record = UsageRecord(
            timestamp=datetime.now(),
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost=cost,
            request_type=request_type
        )

        self.records.append(record)
        return cost

    def _calculate_cost(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int
    ) -> float:
        """비용 계산"""
        if model not in self.pricing:
            return 0.0

        pricing = self.pricing[model]
        return (
            prompt_tokens * pricing["input"] / 1_000_000 +
            completion_tokens * pricing["output"] / 1_000_000
        )

    def get_summary(self, period: str = "day") -> Dict:
        """사용량 요약"""
        from collections import defaultdict

        summary = defaultdict(lambda: {"tokens": 0, "cost": 0, "requests": 0})

        for record in self.records:
            model = record.model
            summary[model]["tokens"] += record.prompt_tokens + record.completion_tokens
            summary[model]["cost"] += record.cost
            summary[model]["requests"] += 1

        return dict(summary)

    def set_budget_alert(self, daily_limit: float):
        """일일 예산 알림 설정"""
        today_cost = sum(
            r.cost for r in self.records
            if r.timestamp.date() == datetime.now().date()
        )

        if today_cost > daily_limit:
            return f"⚠️ Daily budget exceeded: ${today_cost:.2f} / ${daily_limit:.2f}"

        return None
```

### 2.2 최적화 전략

```python
class CostOptimizer:
    """비용 최적화 전략"""

    def __init__(self):
        self.cache = {}

    def semantic_cache(self, query: str, threshold: float = 0.95):
        """시맨틱 캐싱"""
        # 유사한 이전 쿼리 찾기
        from sentence_transformers import SentenceTransformer
        import numpy as np

        if not hasattr(self, 'encoder'):
            self.encoder = SentenceTransformer('all-MiniLM-L6-v2')

        query_emb = self.encoder.encode(query)

        for cached_query, (cached_emb, response) in self.cache.items():
            similarity = np.dot(query_emb, cached_emb) / (
                np.linalg.norm(query_emb) * np.linalg.norm(cached_emb)
            )
            if similarity > threshold:
                return response

        return None

    def add_to_cache(self, query: str, response: str):
        """캐시에 추가"""
        if hasattr(self, 'encoder'):
            emb = self.encoder.encode(query)
            self.cache[query] = (emb, response)

    def select_model(
        self,
        task_complexity: str,
        latency_requirement: str = "normal"
    ) -> str:
        """태스크에 맞는 모델 선택"""
        model_map = {
            # (complexity, latency) -> model
            ("simple", "fast"): "gpt-3.5-turbo",
            ("simple", "normal"): "gpt-3.5-turbo",
            ("medium", "fast"): "claude-3-haiku",
            ("medium", "normal"): "gpt-4o",
            ("complex", "fast"): "gpt-4o",
            ("complex", "normal"): "claude-3-opus",
        }

        return model_map.get(
            (task_complexity, latency_requirement),
            "gpt-4o"
        )

    def prompt_compression(self, text: str, target_ratio: float = 0.5) -> str:
        """프롬프트 압축"""
        # LLMLingua 등 사용 가능
        # 여기서는 간단한 요약 방식
        words = text.split()
        target_len = int(len(words) * target_ratio)

        # 중요 문장 선택 (실제로는 더 정교한 방법 필요)
        sentences = text.split('.')
        if len(sentences) <= 2:
            return text

        # 첫 문장과 마지막 문장 유지
        compressed = sentences[0] + '.' + sentences[-1]
        return compressed
```

---

## 3. LLM 평가

### 3.1 벤치마크

```
주요 벤치마크:
┌────────────────────────────────────────────────────────────────┐
│  General                                                        │
│  - MMLU: 57 subjects, multiple choice                          │
│  - HellaSwag: Commonsense reasoning                            │
│  - WinoGrande: Coreference resolution                          │
│                                                                │
│  Reasoning                                                      │
│  - GSM8K: Grade school math                                    │
│  - MATH: Competition math                                       │
│  - ARC: Science questions                                       │
│                                                                │
│  Coding                                                         │
│  - HumanEval: Python code generation                           │
│  - MBPP: Python problems                                       │
│  - CodeContests: Competitive programming                       │
│                                                                │
│  Chat/Instruction                                               │
│  - MT-Bench: Multi-turn conversation                           │
│  - AlpacaEval: Instruction following                           │
│  - Chatbot Arena: Human preference                             │
└────────────────────────────────────────────────────────────────┘
```

### 3.2 자동 평가

```python
import re
from typing import List, Dict

class LLMEvaluator:
    """LLM 자동 평가"""

    def __init__(self, model_client):
        self.client = model_client

    def evaluate_factuality(
        self,
        question: str,
        answer: str,
        reference: str
    ) -> Dict:
        """사실성 평가"""
        prompt = f"""Evaluate if the answer is factually consistent with the reference.

Question: {question}
Answer: {answer}
Reference: {reference}

Score from 1-5 where:
1 = Completely incorrect
3 = Partially correct
5 = Completely correct

Provide your score and brief explanation.
Format: Score: X
Explanation: ..."""

        response = self.client.chat([{"role": "user", "content": prompt}])
        text = response["content"]

        # 점수 추출
        score_match = re.search(r'Score:\s*(\d)', text)
        score = int(score_match.group(1)) if score_match else 3

        return {
            "score": score,
            "explanation": text
        }

    def evaluate_helpfulness(
        self,
        instruction: str,
        response: str
    ) -> Dict:
        """유용성 평가"""
        prompt = f"""Evaluate how helpful and complete the response is.

Instruction: {instruction}
Response: {response}

Rate on these criteria (1-5 each):
1. Relevance: Does it address the instruction?
2. Completeness: Does it fully answer?
3. Clarity: Is it well-written and clear?
4. Accuracy: Is the information correct?

Format:
Relevance: X
Completeness: X
Clarity: X
Accuracy: X
Overall: X"""

        response = self.client.chat([{"role": "user", "content": prompt}])
        text = response["content"]

        # 점수 파싱
        scores = {}
        for criterion in ["Relevance", "Completeness", "Clarity", "Accuracy", "Overall"]:
            match = re.search(rf'{criterion}:\s*(\d)', text)
            scores[criterion.lower()] = int(match.group(1)) if match else 3

        return scores

    def pairwise_comparison(
        self,
        instruction: str,
        response_a: str,
        response_b: str
    ) -> str:
        """쌍대 비교"""
        prompt = f"""Compare these two responses to the instruction.

Instruction: {instruction}

Response A:
{response_a}

Response B:
{response_b}

Which response is better? Consider helpfulness, accuracy, and clarity.
Answer with:
- "A" if Response A is better
- "B" if Response B is better
- "TIE" if they are equally good

Your choice:"""

        response = self.client.chat([{"role": "user", "content": prompt}])
        text = response["content"].strip().upper()

        if "A" in text and "B" not in text:
            return "A"
        elif "B" in text and "A" not in text:
            return "B"
        else:
            return "TIE"


# MT-Bench 스타일 평가
class MTBenchEvaluator:
    """MT-Bench 스타일 다중 턴 평가"""

    def __init__(self, judge_model):
        self.judge = judge_model

    def evaluate_conversation(
        self,
        conversation: List[Dict]
    ) -> Dict:
        """대화 평가"""
        # 각 턴별 평가
        turn_scores = []

        for i, turn in enumerate(conversation):
            if turn["role"] == "assistant":
                context = conversation[:i+1]
                score = self._evaluate_turn(context)
                turn_scores.append(score)

        return {
            "turn_scores": turn_scores,
            "average": sum(turn_scores) / len(turn_scores) if turn_scores else 0
        }

    def _evaluate_turn(self, context: List[Dict]) -> float:
        """개별 턴 평가"""
        # 평가 프롬프트 구성
        context_str = "\n".join([
            f"{msg['role']}: {msg['content']}"
            for msg in context
        ])

        prompt = f"""Rate the assistant's last response on a scale of 1-10.

Conversation:
{context_str}

Consider:
- Helpfulness
- Relevance
- Accuracy
- Depth

Score (1-10):"""

        response = self.judge.chat([{"role": "user", "content": prompt}])
        score_match = re.search(r'\d+', response["content"])

        return float(score_match.group()) if score_match else 5.0
```

### 3.3 인간 평가

```python
from dataclasses import dataclass
from typing import Optional
import random

@dataclass
class EvaluationItem:
    """평가 항목"""
    id: str
    instruction: str
    response_a: str
    response_b: str
    model_a: str
    model_b: str
    winner: Optional[str] = None
    annotator: Optional[str] = None

class HumanEvaluation:
    """인간 평가 관리"""

    def __init__(self):
        self.items: List[EvaluationItem] = []
        self.results: Dict[str, int] = {}

    def add_comparison(
        self,
        instruction: str,
        responses: Dict[str, str]  # {model_name: response}
    ):
        """비교 항목 추가"""
        models = list(responses.keys())
        if len(models) != 2:
            raise ValueError("Exactly 2 models required")

        # 순서 랜덤화 (bias 방지)
        if random.random() > 0.5:
            models = models[::-1]

        item = EvaluationItem(
            id=str(len(self.items)),
            instruction=instruction,
            response_a=responses[models[0]],
            response_b=responses[models[1]],
            model_a=models[0],
            model_b=models[1]
        )

        self.items.append(item)

    def record_judgment(
        self,
        item_id: str,
        winner: str,  # "A", "B", or "TIE"
        annotator: str
    ):
        """평가 결과 기록"""
        for item in self.items:
            if item.id == item_id:
                item.winner = winner
                item.annotator = annotator

                # 승자 모델 기록
                if winner == "A":
                    winning_model = item.model_a
                elif winner == "B":
                    winning_model = item.model_b
                else:
                    winning_model = "TIE"

                self.results[winning_model] = self.results.get(winning_model, 0) + 1
                break

    def get_elo_ratings(self) -> Dict[str, float]:
        """Elo 레이팅 계산"""
        # 초기 레이팅
        ratings = {}
        for item in self.items:
            ratings[item.model_a] = 1500
            ratings[item.model_b] = 1500

        K = 32  # K-factor

        for item in self.items:
            if item.winner is None:
                continue

            ra = ratings[item.model_a]
            rb = ratings[item.model_b]

            # Expected scores
            ea = 1 / (1 + 10 ** ((rb - ra) / 400))
            eb = 1 / (1 + 10 ** ((ra - rb) / 400))

            # Actual scores
            if item.winner == "A":
                sa, sb = 1, 0
            elif item.winner == "B":
                sa, sb = 0, 1
            else:
                sa, sb = 0.5, 0.5

            # Update ratings
            ratings[item.model_a] += K * (sa - ea)
            ratings[item.model_b] += K * (sb - eb)

        return ratings
```

---

## 핵심 정리

### API 사용 체크리스트
```
□ API 키 환경 변수로 관리
□ 토큰 수 사전 계산
□ 비용 모니터링 설정
□ Rate limit 처리
□ 에러 핸들링 및 재시도
□ 캐싱 전략 구현
```

### 평가 방법 선택
```
- 객관식 문제 → 정확도
- 생성 태스크 → LLM-as-Judge
- 채팅/대화 → MT-Bench/Chatbot Arena
- 코딩 → pass@k, HumanEval
- 프로덕션 → A/B 테스트
```

---

## 참고 자료

1. [OpenAI API Documentation](https://platform.openai.com/docs)
2. [Anthropic Claude Documentation](https://docs.anthropic.com/)
3. Zheng et al. (2023). "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena"

---

## 연습 문제

### 연습 문제 1: API 비용 계산 및 모델 선택
팀에서 문서 요약 서비스를 구축하고 있습니다. 각 요청은 2,000 토큰 문서(입력)를 포함하고 300 토큰 요약(출력)을 생성합니다. 하루에 10,000건의 요청을 처리해야 합니다.

A) GPT-4o ($5/1M 입력, $15/1M 출력) vs. Claude 3 Haiku ($0.25/1M 입력, $1.25/1M 출력)에 대한 일일 비용을 계산하세요.
B) GPT-4o에서 Haiku로 전환할 때 하루 $100 이상의 비용 절감이 발생하는 최소 일일 요청 수는?
C) 요청의 20%가 "복잡한" 경우(추론, 고품질 필요), 라우팅 전략을 제안하세요.

<details>
<summary>정답 보기</summary>

**A) 10,000건 요청 시 일일 비용**:

```python
requests_per_day = 10_000
input_tokens_per_req = 2_000
output_tokens_per_req = 300

# GPT-4o: $5/1M input, $15/1M output
gpt4o_input_cost = (requests_per_day * input_tokens_per_req / 1_000_000) * 5
                 = (10_000 * 2_000 / 1_000_000) * 5 = 20M tokens * $5/M = $100/day

gpt4o_output_cost = (requests_per_day * output_tokens_per_req / 1_000_000) * 15
                  = (10_000 * 300 / 1_000_000) * 15 = 3M tokens * $15/M = $45/day

gpt4o_total = $100 + $45 = $145/day

# Claude 3 Haiku: $0.25/1M input, $1.25/1M output
haiku_input_cost = 20M tokens * $0.25/M = $5/day
haiku_output_cost = 3M tokens * $1.25/M = $3.75/day
haiku_total = $5 + $3.75 = $8.75/day

# Haiku 사용 시 일일 절감: $145 - $8.75 = $136.25/day
```

**B) 하루 $100 이상 절감을 위한 요청 수 임계값**:

```python
# 요청당 비용:
gpt4o_per_req = (2000/1M * $5) + (300/1M * $15) = $0.01 + $0.0045 = $0.0145
haiku_per_req = (2000/1M * $0.25) + (300/1M * $1.25) = $0.0005 + $0.000375 = $0.000875

savings_per_req = $0.0145 - $0.000875 = $0.013625

# 하루 $100 이상 절감 조건:
min_requests = $100 / $0.013625 ≈ 7,339 requests/day
```
**정답: 하루 7,339건 이상부터 일일 $100 이상 절감.** 10K 건/일에서는 $136/일 = 월 ~$4,100 절감.

**C) 복잡도 혼합 시 라우팅 전략**:

```python
def route_request(document: str, task_complexity: str) -> str:
    """Route to appropriate model based on complexity"""
    if task_complexity == "complex":  # 20% of requests
        # Legal contracts, technical papers, multi-document synthesis
        return "claude-3-sonnet"  # Balance of quality and cost
    else:  # 80% of requests
        # News articles, product descriptions, standard reports
        return "claude-3-haiku"

# 라우팅 적용 시 비용 (10K 건/일):
# 복잡한 2,000건 → Sonnet ($3/1M 입력, $15/1M 출력)
#   2,000 * (2000 * $3/M + 300 * $15/M) = $12 + $9 = $21/일
# 일반 8,000건 → Haiku
#   8,000 * (2000 * $0.25/M + 300 * $1.25/M) = $4 + $3 = $7/일
# 라우팅 총 비용: $28/일 vs. 전체 GPT-4o $145/일 → 80% 비용 절감
```

</details>

### 연습 문제 2: LLM-as-a-Judge 편향 분석
LLM-as-a-Judge는 강력한 LLM을 사용하여 모델 출력을 평가합니다. LLM 평가자의 평가를 왜곡할 수 있는 세 가지 특정 편향을 식별하고, 각각에 대한 완화 전략을 제안하세요.

<details>
<summary>정답 보기</summary>

**편향 1: 위치 편향(Position Bias) (순서 편향)**

LLM 평가자는 품질에 관계없이 프롬프트에서 먼저(또는 마지막에) 나타나는 응답을 선호하는 경향이 있습니다.

- **예시**: 응답 A vs. 응답 B를 비교할 때, 같은 평가자가 한 순서에서는 A=7, B=5로 점수를 매기지만, 순서를 바꾸면 A=5, B=7로 바꿔 매깁니다.
- **완화 방법**: 항상 두 가지 순서(A vs. B 그리고 B vs. A)를 평가하고 결과를 평균냅니다. 쌍비교에서 평가자가 순서 간 일관성이 없으면 동점으로 처리합니다. 이를 **위치 편향 제거(positional debiasing)**라고 하며 Chatbot Arena에서 사용됩니다.

**편향 2: 장황함 편향(Verbosity Bias) (길이 선호)**

LLM 평가자는 간결한 답변이 더 적절한 경우에도 더 길고 상세한 응답을 고품질로 평가하는 경향이 있습니다.

- **예시**: 500단어 답변이 질문에 더 정확하고 직접적으로 답하는 50단어 답변보다 높은 점수를 받습니다.
- **완화 방법**: 불필요한 길이에 불이익을 주는 명시적 채점 기준 추가: "고품질 답변은 질문에 정확하고 간결하게 답하는 것입니다. 긴 답변이 자동으로 더 좋은 것은 아닙니다." 또한 보정 기준점으로 참조 답변 길이를 제공합니다.

**편향 3: 자기강화 편향(Self-enhancement Bias) (평가자 = 평가 모델과 같은 계열인 경우)**

GPT-4를 사용하여 GPT-4 vs. Claude-3을 판단하면 자체 모델의 스타일과 내용을 체계적으로 선호할 수 있습니다.

- **예시**: GPT-4 평가자가 동등하게 좋은 Claude 응답보다 GPT-4o 답변에 평균 +0.3점 높은 점수를 매깁니다.
- **완화 방법**: 다양한 모델 계열의 여러 평가자를 사용하고 점수를 앙상블합니다. 중요한 평가의 경우, 평가되는 모델과 다른 제공자의 평가자를 최소 하나 포함합니다. 평가자-모델 일치도를 보고합니다 (다른 계열의 평가자들이 동의하면 결과가 더 신뢰할 수 있습니다).

</details>

### 연습 문제 3: 특정 사용 사례를 위한 벤치마크 선택
세 가지 다른 제품 배포에 LLM을 평가하고 있습니다. 각각에 대해 {MMLU, HumanEval, MT-Bench, GSM8K, HellaSwag, MATH} 목록에서 가장 관련성 높은 벤치마크를 선택하고, 해당 벤치마크에서 높은 점수가 보장하는 것과 보장하지 않는 것을 설명하세요.

| 배포 | 주요 용도 | 최적 벤치마크 | 보장 사항 | 보장하지 않는 사항 |
|------------|------------|----------------|------------|-------------------|
| A) Python 개발자용 AI 코딩 어시스턴트 | 코드 완성, 디버깅 | ??? | ??? | ??? |
| B) 다중 턴 지원을 위한 고객 서비스 챗봇 | 대화, 정책 준수 | ??? | ??? | ??? |
| C) 투자 보고서를 위한 금융 분석 어시스턴트 | 정량적 추론 | ??? | ??? | ??? |

<details>
<summary>정답 보기</summary>

| 배포 | 최적 벤치마크 | 보장 사항 | 보장하지 않는 사항 |
|------------|----------------|------------|-------------------|
| A) AI 코딩 어시스턴트 | **HumanEval** (Python 코드 생성, pass@k) | 일반적인 알고리즘 태스크에 대해 구문적으로 올바르고 기능적으로 완전한 Python 함수 생성 능력 | 도메인 특화 코드(금융, 의료 API) 성능, 기존 코드 디버깅, 코드 리뷰 품질, 긴(100줄 이상) 함수 성능 |
| B) 고객 서비스 챗봇 | **MT-Bench** (다중 턴 대화 품질, 1-10 점수) | 다중 턴 대화 일관성 품질, 대화 전반에 걸쳐 컨텍스트 유지 능력, 후속 질문 처리 | 특정 회사 정책이나 금지된 주제 준수, 실제 고객 만족도, 모호한 실제 요청 처리, 도메인 특화 어휘 성능 |
| C) 금융 분석 어시스턴트 | **GSM8K + MATH** (다양한 난이도의 수학적 추론) | 정확한 산술, 대수적 조작, 구조화된 수학적 추론, 정량적 내용이 있는 문장형 문제 풀기 | 도메인 특화 금융 지식(CAPM, Black-Scholes), 규제 준수 인식, 실제 금융 데이터 형식 처리 능력, 보고서의 데이터 불일치 탐지 |

**핵심 통찰**: 모든 벤치마크는 실제 능력의 대리 지표를 평가합니다. HumanEval에서 90%를 받은 모델도 회사 특화 코드베이스에서는 실패할 수 있습니다. 항상 실제 배포를 대표하는 도메인 특화 평가 세트로 벤치마크를 보완하세요.

</details>

### 연습 문제 4: 평가 파이프라인 설계
고객 대면 법률 문서 Q&A 시스템을 위한 프로덕션 평가 파이프라인을 설계하세요. 이 시스템은 관련 조항을 검색하고 답변을 생성합니다. 평가 차원, 지표, 합격 임계값을 명시하세요.

<details>
<summary>정답 보기</summary>

**프로덕션 평가 파이프라인**:

```
평가 파이프라인:
┌──────────────────────────────────────────────────┐
│  입력: (질문, 검색된_조항, 답변)                   │
│                                                  │
│  차원 1: 사실 정확도                               │
│  차원 2: 법률 인용 정확도                          │
│  차원 3: 완전성                                   │
│  차원 4: 안전성 (유해한 조언 없음)                  │
│  차원 5: 유창성                                   │
└──────────────────────────────────────────────────┘
```

**평가 차원 및 지표**:

| 차원 | 방법 | 지표 | 합격 임계값 |
|-----------|--------|--------|----------------|
| 사실 정확도 | 참조 답변이 있는 LLM-as-Judge | 1-5 점수 | 평균 ≥ 4.0 |
| 인용 정확도 | 규칙 기반: 인용된 조항 번호가 검색된 문서에 있는지 확인 | 정밀도/재현율 | 인용 정밀도 ≥ 90% |
| 완전성 | LLM-as-Judge: "답변이 질문의 모든 부분을 다루는가?" | 이진 + 점수 | ≥ 80%가 "완전"으로 평가 |
| 안전성 | 키워드 필터 + "확정적인 법률 조언"에 대한 LLM 분류기 | 이진 플래그 | 면책 조항 없는 확정적 조언 발생률 0% |
| 유창성 | 자동화: 퍼플렉시티 + 문법 검사 | 1-5 점수 | ≥ 4.0 |

**오프라인 vs. 온라인 평가**:

```python
# 오프라인 (개발 중):
# - 인간이 큐레이션한 200개 법률 Q&A 쌍 골든 테스트 세트
# - 모든 모델 업데이트 후 실행
# - 게이트: 5개 차원 모두 임계값을 통과해야 함

# 온라인 (프로덕션):
# - 실제 쿼리의 1% 샘플링하여 인간 리뷰
# - 추적:
#   - 사용자 만족도 (좋아요/싫어요)
#   - 에스컬레이션 비율 (답변이 불명확해서 사용자가 후속 질문하는 횟수)
#   - 거절 비율 (모델이 지나치게 많은 면책 조항 추가 → 사용자가 답변을 얻지 못함)
# - 정확도가 기준선에서 5% 이상 하락하면 알림
```

**법률 도메인의 중요 안전 규칙**: 특정 법률 조언으로 해석될 수 있는 모든 답변에는 다음 면책 조항이 포함되어야 합니다: "이것은 법률 조언이 아닙니다. 귀하의 특정 상황에 대해서는 자격을 갖춘 변호사와 상담하십시오." 평가 파이프라인은 이 주의 사항 없이 확정적인 법적 결론을 내리는 모든 답변을 표시하고 거부해야 합니다.

</details>
4. Chen et al. (2021). "Evaluating Large Language Models Trained on Code" (HumanEval)
