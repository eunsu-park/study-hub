# 08. 프롬프트 엔지니어링

## 학습 목표

- 효과적인 프롬프트 작성
- Zero-shot, Few-shot 기법
- Chain-of-Thought (CoT)
- 고급 프롬프팅 기법

---

## 1. 프롬프트 기초

### 프롬프트 구성 요소

```
┌─────────────────────────────────────────┐
│ [시스템 지시]                            │
│ 당신은 도움이 되는 AI 어시스턴트입니다.    │
├─────────────────────────────────────────┤
│ [컨텍스트]                               │
│ 다음 텍스트를 참고하세요: ...             │
├─────────────────────────────────────────┤
│ [태스크 지시]                            │
│ 위 텍스트를 요약해주세요.                 │
├─────────────────────────────────────────┤
│ [출력 형식]                              │
│ JSON 형식으로 응답해주세요.               │
└─────────────────────────────────────────┘
```

### 기본 원칙

```
1. 명확성: 모호하지 않게 작성
2. 구체성: 원하는 것을 정확히 명시
3. 예시: 가능하면 예시 제공
4. 제약: 출력 형식, 길이 등 제약 명시
```

---

## 2. Zero-shot vs Few-shot

### Zero-shot

```
예시 없이 태스크만 설명

프롬프트:
"""
다음 리뷰의 감성을 분석해주세요.
리뷰: "이 영화는 정말 지루했어요."
감성:
"""

응답: 부정적
```

### Few-shot

```
몇 개의 예시 제공

프롬프트:
"""
다음 리뷰의 감성을 분석해주세요.

리뷰: "정말 재미있는 영화였어요!"
감성: 긍정

리뷰: "최악의 영화, 시간 낭비"
감성: 부정

리뷰: "그냥 그랬어요"
감성: 중립

리뷰: "이 영화는 정말 지루했어요."
감성:
"""

응답: 부정
```

### Few-shot 팁

```python
# 예시 선택 기준
1. 다양성: 모든 클래스의 예시 포함
2. 대표성: 전형적인 예시 사용
3. 유사성: 실제 입력과 유사한 예시
4. 최신성: 관련성 높은 예시

# 예시 개수
- 일반적으로 3-5개
- 복잡한 태스크: 5-10개
- 토큰 제한 고려
```

---

## 3. Chain-of-Thought (CoT)

### 기본 CoT

```
단계별 추론 유도

프롬프트:
"""
Q: Roger has 5 tennis balls. He buys 2 more cans of 3 balls each.
   How many balls does he have now?

A: Let's think step by step.
1. Roger started with 5 balls.
2. He bought 2 cans × 3 balls = 6 balls.
3. Total: 5 + 6 = 11 balls.
The answer is 11.
"""
```

### Zero-shot CoT

```
간단하게 추론 유도

프롬프트:
"""
Q: 5 + 7 × 3 = ?

Let's think step by step.
"""

응답:
1. First, we need to follow order of operations (PEMDAS).
2. Multiplication comes before addition.
3. 7 × 3 = 21
4. 5 + 21 = 26
The answer is 26.
```

### Self-Consistency

```python
# 여러 추론 경로 생성 후 다수결

responses = []
for _ in range(5):
    response = model.generate(prompt, temperature=0.7)
    responses.append(extract_answer(response))

# 가장 많이 나온 답 선택
final_answer = max(set(responses), key=responses.count)
```

---

## 4. 역할 부여 (Role Playing)

### 전문가 역할

```
시스템 프롬프트:
"""
당신은 10년 경력의 파이썬 개발자입니다.
코드 리뷰를 할 때 다음을 확인합니다:
- 코드 가독성
- 버그 가능성
- 성능 최적화
- 보안 취약점
"""

사용자:
"""
다음 코드를 리뷰해주세요:
def get_user(id):
    return db.execute(f"SELECT * FROM users WHERE id = {id}")
"""
```

### 페르소나

```
"""
당신은 친절하고 인내심 있는 초등학교 선생님입니다.
복잡한 개념을 쉬운 비유로 설명합니다.
항상 격려하는 어조를 사용합니다.

질문: 중력이 뭐예요?
"""
```

---

## 5. 출력 형식 지정

### JSON 출력

```
프롬프트:
"""
다음 텍스트에서 인물과 장소를 추출해주세요.

텍스트: "철수는 서울에서 영희를 만났다."

JSON 형식으로 응답:
{
  "persons": [...],
  "locations": [...]
}
"""
```

### 구조화된 출력

```
프롬프트:
"""
다음 기사를 분석해주세요.

## 요약
(2-3문장)

## 핵심 포인트
- 포인트 1
- 포인트 2

## 감성
(긍정/부정/중립)
"""
```

### XML 태그

```
프롬프트:
"""
다음 텍스트를 번역하고 설명해주세요.

<text>Hello, how are you?</text>

<translation>번역 결과</translation>
<explanation>번역 설명</explanation>
"""
```

---

## 6. 고급 기법

### Self-Ask

```
모델이 스스로 질문하고 답변

"""
질문: 바이든 대통령의 고향은 어디인가요?

후속 질문 필요: 네
후속 질문: 바이든 대통령은 누구인가요?
중간 답변: 조 바이든은 미국의 46대 대통령입니다.

후속 질문 필요: 네
후속 질문: 조 바이든은 어디서 태어났나요?
중간 답변: 펜실베이니아 주 스크랜턴에서 태어났습니다.

후속 질문 필요: 아니오
최종 답변: 바이든 대통령의 고향은 펜실베이니아 주 스크랜턴입니다.
"""
```

### ReAct (Reason + Act)

```
추론과 행동을 번갈아 수행

"""
질문: 2023년 노벨 물리학상 수상자는 누구인가요?

Thought: 2023년 노벨 물리학상 수상자를 찾아야 합니다.
Action: Search[2023 노벨 물리학상]
Observation: 피에르 아고스티니, 페렌츠 크라우스, 앤 륄리에가 수상했습니다.

Thought: 검색 결과를 확인했습니다.
Action: Finish[피에르 아고스티니, 페렌츠 크라우스, 앤 륄리에]
"""
```

### Tree of Thoughts

```python
# 여러 사고 경로를 트리로 탐색

def tree_of_thoughts(problem, depth=3, branches=3):
    thoughts = []

    for _ in range(branches):
        # 첫 번째 생각 생성
        thought = generate_thought(problem)
        score = evaluate_thought(thought)
        thoughts.append((thought, score))

    # 상위 생각 선택
    best_thoughts = sorted(thoughts, key=lambda x: x[1], reverse=True)[:2]

    # 재귀적으로 확장
    for thought, _ in best_thoughts:
        if depth > 0:
            extended = tree_of_thoughts(thought, depth-1, branches)
            thoughts.extend(extended)

    return thoughts
```

---

## 7. 프롬프트 최적화

### 반복적 개선

```python
# 1. 기본 프롬프트로 시작
prompt_v1 = "Summarize this text: {text}"

# 2. 결과 분석 후 개선
prompt_v2 = """
Summarize the following text in 2-3 sentences.
Focus on the main points.
Text: {text}
Summary:
"""

# 3. 예시 추가
prompt_v3 = """
Summarize the following text in 2-3 sentences.

Example:
Text: [긴 기사]
Summary: [간단한 요약]

Text: {text}
Summary:
"""
```

### A/B 테스트

```python
import random

def ab_test_prompts(test_cases, prompt_a, prompt_b):
    results = {'A': 0, 'B': 0}

    for case in test_cases:
        response_a = model.generate(prompt_a.format(**case))
        response_b = model.generate(prompt_b.format(**case))

        # 평가 (자동 또는 수동)
        score_a = evaluate(response_a, case['expected'])
        score_b = evaluate(response_b, case['expected'])

        if score_a > score_b:
            results['A'] += 1
        else:
            results['B'] += 1

    return results
```

---

## 8. 프롬프트 템플릿

### 분류

```python
CLASSIFICATION_PROMPT = """
Classify the following text into one of these categories: {categories}

Text: {text}

Category:"""
```

### 요약

```python
SUMMARIZATION_PROMPT = """
Summarize the following text in {num_sentences} sentences.
Focus on the key points and main arguments.

Text:
{text}

Summary:"""
```

### 질의응답

```python
QA_PROMPT = """
Answer the question based on the context below.
If the answer cannot be found, say "I don't know."

Context: {context}

Question: {question}

Answer:"""
```

### 코드 생성

```python
CODE_GENERATION_PROMPT = """
Write a {language} function that {task_description}.

Requirements:
{requirements}

Function:
```{language}
"""
```

---

## 9. Python에서 프롬프트 관리

### 템플릿 클래스

```python
class PromptTemplate:
    def __init__(self, template: str):
        self.template = template

    def format(self, **kwargs) -> str:
        return self.template.format(**kwargs)

    @classmethod
    def from_file(cls, path: str):
        with open(path, 'r') as f:
            return cls(f.read())

# 사용
template = PromptTemplate("""
You are a {role}.
Task: {task}
Input: {input}
Output:
""")

prompt = template.format(
    role="helpful assistant",
    task="translate to Korean",
    input="Hello, world!"
)
```

### LangChain 프롬프트

```python
from langchain.prompts import PromptTemplate, FewShotPromptTemplate

# 기본 템플릿
prompt = PromptTemplate(
    input_variables=["text"],
    template="Summarize: {text}"
)

# Few-shot 템플릿
examples = [
    {"input": "긴 텍스트 1", "output": "요약 1"},
    {"input": "긴 텍스트 2", "output": "요약 2"},
]

few_shot_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=PromptTemplate(
        input_variables=["input", "output"],
        template="Input: {input}\nOutput: {output}"
    ),
    prefix="Summarize the following texts:",
    suffix="Input: {text}\nOutput:",
    input_variables=["text"]
)
```

---

## 정리

### 프롬프트 체크리스트

```
□ 명확한 지시 제공
□ 필요시 예시 포함 (Few-shot)
□ 출력 형식 지정
□ 역할/페르소나 설정
□ 단계별 추론 유도 (필요시)
□ 제약 조건 명시
```

### 기법 선택 가이드

| 상황 | 추천 기법 |
|------|----------|
| 간단한 태스크 | Zero-shot |
| 특정 형식 필요 | Few-shot + 형식 지정 |
| 추론 필요 | Chain-of-Thought |
| 복잡한 추론 | Tree of Thoughts |
| 도구 사용 필요 | ReAct |

---

## 연습 문제

### 연습 문제 1: Zero-shot vs Few-shot 비교

제품 리뷰를 긍정(Positive), 부정(Negative), 중립(Neutral) 세 가지로 분류하는 태스크가 있습니다. Zero-shot 프롬프트와 Few-shot 프롬프트를 각각 작성하고, 각각 어떤 상황에서 선호하는지 설명하세요.

<details>
<summary>정답 보기</summary>

**Zero-shot 프롬프트(prompt):**
```
아래 제품 리뷰의 감성을 긍정(Positive), 부정(Negative), 중립(Neutral) 중 하나로 분류하세요.

리뷰: "{review}"

감성:
```

**Few-shot 프롬프트(prompt):**
```
아래 제품 리뷰들의 감성을 분류하세요.

리뷰: "정말 마음에 들어요! 올해 최고의 구매입니다."
감성: Positive

리뷰: "품질이 최악이에요. 이틀 만에 망가졌어요."
감성: Negative

리뷰: "설명대로 작동합니다. 특별한 건 없어요."
감성: Neutral

리뷰: "{review}"
감성:
```

**각 방식을 선호하는 경우:**
- **Zero-shot**: 태스크가 간단하고 모델의 사전 학습이 충분할 때. 더 빠르게 작성 가능하고 토큰(token)을 적게 소모합니다.
- **Few-shot**: 특정 출력 형식이 필요하거나, 태스크가 모호하거나, 일관된 스타일이 필요할 때. 도메인 특화 태스크나 희귀한 레이블(label) 집합에 특히 유용합니다.

기본 원칙: Zero-shot으로 시작하고, 결과가 일관적이지 않으면 Few-shot으로 전환하세요.
</details>

---

### 연습 문제 2: Chain-of-Thought 프롬프트 설계

사용자가 휴가를 감당할 수 있는지 계산하고 싶습니다. 세후 월 수입은 420만 원, 고정 월 지출은 280만 원, 휴가 비용은 150만 원입니다. 여행은 3개월 후입니다. 모델이 올바르게 추론하도록 유도하는 Zero-shot CoT 프롬프트를 작성하세요.

<details>
<summary>정답 보기</summary>

```
한 사용자가 휴가를 위한 저축이 가능한지 알고 싶습니다.
- 월 수입 (세후): 420만 원
- 고정 월 지출: 280만 원
- 휴가 비용: 150만 원
- 휴가까지 남은 기간: 3개월

감당할 수 있을까요? 단계적으로 생각해봅시다.
```

**기대하는 모델 추론:**
```
1. 월 저축 가능액 = 수입 - 지출 = 420만 - 280만 = 140만 원
2. 3개월 총 저축 가능액 = 140만 × 3 = 420만 원
3. 휴가 비용 = 150만 원
4. 420만 > 150만 이므로, 충분히 감당 가능합니다.
   남는 금액: 420만 - 150만 = 270만 원

답: 네, 휴가 비용을 감당할 수 있습니다.
```

**CoT가 도움이 되는 이유:** 단계별 추론 없이는 모델이 산수 오류를 범할 수 있습니다. "단계적으로 생각해봅시다"라는 문구가 구조화된 추론을 유도하여 다단계 수학 문제에서 오류를 줄입니다.
</details>

---

### 연습 문제 3: 구조화된 출력 추출

채용 공고에서 구조화된 정보를 추출하는 프롬프트(prompt)를 작성하세요. 출력은 `title`, `company`, `location`, `salary_range`, `required_skills`(목록), `experience_years`(숫자) 필드를 갖는 유효한 JSON이어야 합니다. 공고에 언급되지 않은 필드는 적절히 처리하세요.

<details>
<summary>정답 보기</summary>

```python
EXTRACTION_PROMPT = """
아래 채용 공고에서 정보를 추출하여 JSON으로 반환하세요.
언급되지 않은 필드는 null을 사용하세요.

필수 JSON 구조:
{{
  "title": "직무 타이틀",
  "company": "회사명",
  "location": "도시/원격",
  "salary_range": "예: 5000만-7000만 원 또는 null",
  "required_skills": ["스킬1", "스킬2"],
  "experience_years": 3
}}

채용 공고:
{posting}

JSON:
"""

# 사용 예시
posting = """
DataCorp 시니어 ML 엔지니어
서울, 대한민국 (하이브리드)
Python과 PyTorch 5년 이상 경력자를 찾습니다.
분산 학습 및 MLflow 지식은 우대 사항입니다.
"""

# 기대 출력:
# {
#   "title": "시니어 ML 엔지니어",
#   "company": "DataCorp",
#   "location": "서울, 대한민국 (하이브리드)",
#   "salary_range": null,
#   "required_skills": ["Python", "PyTorch"],
#   "experience_years": 5
# }
```

**주요 설계 결정:**
- Python `.format()`에서 리터럴 중괄호는 `{{` `}}`로 이스케이프(escape)합니다
- 정확한 스키마(schema)를 제공하면 모델이 임의 필드를 추가하는 것을 방지합니다
- `null` 지시로 누락된 필드에 데이터를 만들어내는 것을 방지합니다
- 마지막에 "JSON:"을 붙이면 모델이 JSON을 직접 출력하도록 유도합니다
</details>

---

### 연습 문제 4: Self-Consistency 구현

여러 추론 경로를 생성하고 가장 많이 나온 답을 반환하는 자기 일관성(Self-Consistency) 프롬프팅 기법을 구현하세요. 이 함수는 모든 예/아니오 또는 단답형 질문에 적용 가능해야 합니다.

<details>
<summary>정답 보기</summary>

```python
from collections import Counter

def self_consistency(
    prompt: str,
    model,
    n_samples: int = 5,
    temperature: float = 0.7
) -> tuple[str, dict]:
    """
    여러 추론 경로를 생성하고 다수결 답을 반환합니다.

    Args:
        prompt: 질문 프롬프트 (CoT 지시 포함 권장)
        model: model.generate(prompt, temperature) -> str를 제공하는 객체
        n_samples: 독립 샘플 생성 횟수
        temperature: 샘플링 온도 (다양성을 위해 > 0)

    Returns:
        (최종_답, 투표_집계)
    """
    answers = []

    for _ in range(n_samples):
        response = model.generate(prompt, temperature=temperature)
        answer = extract_final_answer(response)
        answers.append(answer)

    vote_counts = Counter(answers)
    final_answer = vote_counts.most_common(1)[0][0]

    return final_answer, dict(vote_counts)


def extract_final_answer(response: str) -> str:
    """CoT 응답에서 최종 답을 추출합니다."""
    import re
    match = re.search(r"(?:answer is|따라서|정답은)[:\s]+(.+?)(?:\.|$)",
                      response, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    # 마지막 비어있지 않은 줄로 대체
    lines = [l.strip() for l in response.strip().split('\n') if l.strip()]
    return lines[-1] if lines else response.strip()


# 사용 예시
cot_prompt = """
Q: 기차가 1.5시간에 120km를 달리고, 이후 1시간에 80km를 달렸습니다.
   전체 여정의 평균 속도는 얼마인가요?

단계적으로 생각해봅시다.
"""

# Self-Consistency 적용 (temperature > 0으로 다양한 추론 경로 생성)
# 올바른 모든 경로가 수렴: (120+80) / (1.5+1) = 200/2.5 = 80 km/h
```

**작동 원리:** Temperature > 0이면 모델이 다양한 추론 경로를 탐색합니다. 잘못된 경로는 서로 다른 답으로 분산되고, 올바른 추론 경로는 동일한 답으로 수렴합니다. 다수결 투표(majority voting)가 개별 오류를 걸러냅니다.
</details>

---

### 연습 문제 5: 프롬프트 템플릿 개선

아래 코드 리뷰 프롬프트 템플릿에는 여러 약점이 있습니다. 최소 네 가지 문제점을 파악하고 개선된 버전을 작성하세요.

```python
# 원본 (취약한) 프롬프트
REVIEW_PROMPT = "이 코드를 리뷰해줘: {code}"
```

<details>
<summary>정답 보기</summary>

**원본 프롬프트의 문제점:**
1. **역할/전문성 컨텍스트 없음** — 어떤 종류의 리뷰어(reviewer)인지 불명확
2. **출력 구조 없음** — 응답 형식을 예측할 수 없음 (글머리 목록? 단락? 점수?)
3. **리뷰 기준 없음** — 보안? 성능? 스타일? 정확성? 무엇을 중점적으로 봐야 하는가?
4. **언어 명시 없음** — 언어마다 모범 사례(best practice)가 다름
5. **심각도 지침 없음** — 모든 이슈를 동등하게 취급; 치명적 버그(bug)와 사소한 개선 사항이 구분되지 않음

**개선된 버전:**
```python
CODE_REVIEW_PROMPT = """
당신은 철저한 코드 리뷰를 수행하는 시니어 소프트웨어 엔지니어입니다.

아래 {language} 코드를 심각도별로 구분하여 리뷰해주세요.

리뷰할 코드:
```{language}
{code}
```

정확히 다음 형식으로 리뷰를 제공하세요:

## 치명적 이슈 (머지 전 반드시 수정)
- [이슈]: [설명 및 수정 제안]

## 경고 (수정 권장)
- [이슈]: [설명 및 수정 제안]

## 제안 (있으면 좋음)
- [이슈]: [설명 및 수정 제안]

## 요약
[코드가 잘 된 점을 포함한 2-3문장 전체 평가]
"""

# 사용 예시
review = CODE_REVIEW_PROMPT.format(
    language="python",
    code="""
def get_user(id):
    return db.execute(f"SELECT * FROM users WHERE id = {id}")
"""
)
# 기대 결과: SQL 인젝션(injection) 취약점이 치명적 이슈로 표시됨
```

개선된 프롬프트는 일관된 구조를 강제하고, 모델의 전문성에 집중시키며, 치명적 이슈와 사소한 제안을 구분하여 개발자가 실행 가능한 출력을 얻도록 합니다.
</details>

---

## 다음 단계

[RAG 기초](./09_RAG_Basics.md)에서 검색 증강 생성(RAG) 시스템을 학습합니다.
