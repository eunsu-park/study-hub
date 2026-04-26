# 09. 프롬프트 엔지니어링

## 학습 목표

- 효과적인 프롬프트 작성
- Zero-shot, Few-shot 기법
- Chain-of-Thought (CoT)
- 고급 프롬프팅 기법

---

## 이론과 원리

프롬프트는 모델에게 *그저 텍스트*입니다 — 특별한 지위도, 특권 채널도 없습니다. 그렇다면 왜 작은 문구 변경이 어려운 작업에서 정확도를 20+ 점 움직일까요? LLM의 행동이 **조건부 확률** `p(answer | prompt)`이고, 조건 맥락의 작은 시프트가 모델을 학습된 분포의 완전히 다른 영역으로 이동시킬 수 있기 때문입니다. 프롬프트 엔지니어링은 어떤 조건 패턴이 모델을 정확하고, 형식이 맞고, 안전한 완성으로 신뢰성 있게 조종하는지에 대한 경험적 연구입니다.

이 섹션은 다음을 다룹니다:

- **(A) 조건 분포 관점** — 왜 프롬프트 엔지니어링이 작동하며, 프롬프트를 쓸 때 우리가 실제로 무엇을 하는가.
- **(B) Zero-shot vs few-shot** — 인컨텍스트 학습 메커니즘과 각각 적합한 시점.
- **(C) Chain-of-Thought (CoT)** — "let's think step by step"이 작동하는 이유와 모델의 계산 그래프에 대한 효과.
- **(D) Self-consistency, ToT, ReAct** — 프롬프트 위에 표본 추출과 탐색을 감싼 기법들.
- **(E) 프롬프트 구조와 분해** — 역할(role), 지시, 맥락, 예시, 형식 명세 — 견고한 프롬프트의 "해부".
- **(F) 실패 모드** — 지시 표류, 프롬프트 인젝션 취약성, 형식 부서짐.

### A. 조건 분포 관점

LLM은 `p(next_token | full_context)`를 모델링합니다. 프롬프트는 그 후 모든 예측을 조건짓는 맥락을 설정합니다. 사람에게 의미적으로 비슷해 보이는 두 프롬프트가 매우 다른 분포에 해당할 수 있습니다:

- "Solve this math problem: 23 × 47 =" — 모델은 학습 데이터에서 "Solve this math problem:" 다음에 오는 것 — 교과서식 설명, 종종 정확 — 에 조건부.
- "23 × 47 =" — 모델은 원시 산술 다음에 오는 것에 조건부; 종종 잡음 많은 맥락(오타 있는 포럼 글)에서 나타나며 정확도가 떨어집니다.

모델은 정답을 맞히려는 "의도"가 없습니다. 텍스트가 어떻게 이어지는지에 대한 학습된 분포가 있을 뿐입니다. 프롬프트 엔지니어링은 — 높은 확률 완성이 또한 원하는 완성이 되도록 — **조건 짓기를 조종**하는 실천입니다.

이 관점이 많은 경험적 놀라움을 설명합니다:
- "You are an expert mathematician"이 정확도를 올립니다 — 사전학습의 전문가 작성 콘텐츠가 더 정확하기 때문.
- Few-shot 예시는 모델이 기계적으로 이어가는 패턴을 설정합니다.
- "Let's think step by step"은 사전학습 데이터가 정답과 연관 짓는 추론 사슬 텍스트를 유도합니다.

### B. Zero-Shot vs Few-Shot

**B.1 Zero-shot.** 작업 설명과 입력만:

```
Translate to French: "Hello, how are you?"
Answer:
```

사전학습에 잘 표현되어 있고 표준 형식이 있는 작업에 작동. 새로운 작업이나 비표준 형식에는 실패.

**B.2 Few-shot.** `k`개의 예시 입력-출력 쌍을 앞에 붙입니다:

```
English: "Hello" → French: "Bonjour"
English: "Thank you" → French: "Merci"
English: "Goodbye" → French: "
```

모델이 형식을 패턴 매칭하고 이어갑니다. 예시는 *작업*(무엇을 할지)과 *형식*(어떻게 출력할지)을 모두 가르칩니다. 표준 `k = 3-8`.

**B.3 Induction-head 메커니즘 (메커니즘 해석).** Olsson 등(2022)은 학습된 Transformer에서 "접두사에서 `[A]`의 이전 출현을 찾고, 그 후에 무엇이 왔는지 보고, 지금 같은 것을 예측"하는 특정 어텐션 헤드를 식별했습니다. Few-shot 프롬프트는 이 헤드들을 위한 순수 연료입니다 — 각 예시가 induction head가 복사할 수 있는 `[입력 패턴] → [출력 패턴]` 연관을 만듭니다.

**B.4 언제 선택할지.** Zero-shot: 작업이 일반적, 모델이 큼(10B+), 토큰이 비쌈. Few-shot: 작업이 비표준 형식, 예시가 짧음, 정답 예시 보유.

### C. Chain-of-Thought (CoT)

Wei 등(2022)은 LM에게 최종 답변 전에 중간 추론 단계를 생성하도록 프롬프팅하면 다단계 문제의 정확도가 극적으로 향상됨을 보였습니다:

```
Q: Roger has 5 tennis balls. He buys 2 more cans of tennis balls. Each can has 3 balls. How many balls does he have?
A: Roger started with 5 balls. 2 cans of 3 balls each is 6 balls. 5 + 6 = 11. The answer is 11.
```

**작동 원리.** Transformer는 고정 깊이입니다 — `L` 레이어 각각이 어텐션 + FFN 한 라운드를 수행. `L` 레이어로 모델은 토큰당 *내부적으로* 최대 `L`개의 순차 연산을 계산할 수 있습니다. 다단계 수학 문제는 5+ 순차 연산이 필요할 수 있어 단일 토큰에 대한 모델의 내부 깊이 예산을 초과합니다.

CoT는 계산을 여러 토큰에 걸쳐 외부화합니다. 각 중간 단계는 새 토큰이고, 현재 상태에서 생성됩니다. 전체 추론은 `T`개의 생성 토큰에 걸쳐 펼쳐지며 유효 `L · T` 순차 연산. 모델은 *토큰을 소비*하여 중간 상태에 "소리 내어 생각"합니다.

이것이 또한 CoT가 규모(초기 실험에서 ~60B+)에서만 발현하는 이유입니다 — 작은 모델은 프롬프팅받아도 일관된 중간 단계를 신뢰성 있게 생성할 수 없습니다.

**Zero-shot CoT** (Kojima 등, 2022): 프롬프트에 "Let's think step by step."를 추가만 하면 됩니다. 예시 없이도 대부분의 큰 LLM에서 작동.

### D. Self-Consistency, Tree-of-Thoughts, ReAct

CoT나 LLM 호출을 추가 구조로 감싸는 세 가지 프롬프팅 기법군:

**D.1 Self-consistency** (Wang 등, 2022). 온도 `T > 0`에서 `k`개의 독립 CoT 궤적 표본 추출. 최종 답에 대한 **다수결**. 직관 — 많은 다른 추론 경로가 같은 정답에 도달할 수 있지만, 각 잘못된 경로는 자기 방식으로 잘못됩니다. 투표가 추론 잡음을 한계화합니다. 단일 궤적 CoT 대비 5-10 점 증가가 흔합니다.

**D.2 Tree-of-Thoughts (ToT)** (Yao 등, 2023). 추론을 트리 탐색으로 — 각 단계에서 `b`개의 후보 다음 생각 생성, LLM으로 각각 점수("올바른 길인가?"), 상위 `k`만 확장. 가능한 추론 단계의 트리에 대한 표준 탐색 알고리즘(BFS, DFS, beam). 부분 진척 신호가 명확한 문제에 유용.

**D.3 ReAct** (Yao 등, 2022). 추론 단계("Thought:"), 도구 행동("Action:"), 관측("Observation:")을 교차. 각 도구 결과가 다음 생각의 새 맥락이 됩니다. 현대 에이전트 프레임워크의 토대(레슨 14-15에서 다룸).

### E. 프롬프트 구조: 견고한 프롬프트의 해부

프로덕션 프롬프트는 예측 가능한 구성 요소를 가집니다:

```
[Role/system]      "You are a careful financial analyst."
[Task/instruction] "Extract company revenue and growth rate from the report."
[Context]          "Here is the report: <DOC>...</DOC>"
[Examples]         "Example 1: ..."
[Format spec]      "Output JSON with keys: revenue, growth_rate."
[Constraints]      "If a value is missing, use null. Do not invent numbers."
[Input]            "Now process: <DOC>actual document</DOC>"
```

각 부분이 알려진 실패 모드를 다룹니다:
- 역할(role)이 도메인에 적합한 분포로 조건짓기를 사전 설정.
- 작업(task)이 고수준 목표를 줍니다.
- 맥락(context)이 명시적 구분자로 관련 데이터를 격리(모델이 데이터를 지시와 혼동하는 것 방지).
- 예시가 인컨텍스트 학습 패턴을 설정.
- 형식 명세가 출력 모양을 제약.
- 제약이 그렇지 않으면 환각을 일으킬 가장자리 경우를 처리.

순서가 중요합니다 — "지시가 맥락 뒤에 오는" 패턴은 많은 모델에 작동하지만 부서지기 쉽습니다. 많은 제공자(OpenAI, Anthropic)는 입력 데이터 *전*에 지시를 권장합니다.

### F. 실패 모드

**F.1 지시 표류 / 망각.** 긴 맥락에서 모델이 초기 지시를 "놓칩니다". 완화 — 핵심 지시를 끝에 반복("Remember: respond in JSON only.").

**F.2 프롬프트 인젝션.** 악의적 사용자 입력에 "Ignore previous instructions and ..." 같은 텍스트가 포함됩니다. 모델은 특권-vs-사용자 채널이 없어 따를 수 있습니다. 완화 — 입력 정제, 출력 검증, 아키텍처 분리(레슨 21에서 다룸).

**F.3 형식 부서짐.** "Output JSON"은 95%의 시간 JSON을 만들지만 5%에서 실패합니다. 신뢰성 있는 구조화 출력에는 grammar-constrained 디코딩이나 검증+재시도 사용(레슨 22에서 다룸).

**F.4 적대적 민감성.** 작은 문구 변경("Solve" vs "Compute")이 정확도를 몇 점 움직일 수 있습니다. 완화 — 프롬프트 앙상블, 자동 프롬프트 탐색(APE, OPRO).

**F.5 아첨(sycophancy).** RLHF로 학습된 모델은 종종 틀려도 사용자에게 동의합니다. 완화 — 명시적 "be honest, disagree with me if I'm wrong" 지시; 또는 RLHF 튜닝이 덜 된 베이스 모델 사용.

### 이론에서 아래 함수들로

- §1 (기초) — §A 조건 분포 관점의 구체적 예시(재표현, 역할 주입).
- §2 (zero-shot vs few-shot) — ICL induction-head 직관과 함께 §B 구현.
- §3 (CoT) — §C 코딩, zero-shot "let's think step by step"과 few-shot CoT 예시 포함.
- §4 (역할 놀이) — §E 역할/시스템 구성 요소를 주요 레버로.
- §5 (출력 형식) — §E 형식 명세 구성 요소, §F.3의 형식 부서짐과 함께.
- §6 (고급 기법) — §D self-consistency 구현과 ToT/ReAct의 시작.

---

## 1. 프롬프트 기초

### 프롬프트 구성 요소

> **[시스템 지시]**
> 당신은 도움이 되는 AI 어시스턴트입니다.
>
> [컨텍스트]
> - **다음 텍스트를 참고하세요**: ...
>
> [태스크 지시]
> 위 텍스트를 요약해주세요.
>
> [출력 형식]
> JSON 형식으로 응답해주세요.


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

[RAG 기초](./10_RAG_Fundamentals.md)에서 검색 증강 생성(RAG) 시스템을 학습합니다.
