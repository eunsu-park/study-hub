# Emergent Abilities (창발적 능력)

## 학습 목표
- Emergent Abilities의 정의와 특징 이해
- 규모에 따른 능력 출현 패턴 파악
- Chain-of-Thought 등 주요 창발적 능력 학습
- Capability Elicitation(능력 유도) 기법 습득

---

## 1. Emergent Abilities란?

### 1.1 정의

**Emergent Abilities**(창발적 능력)은 작은 모델에서는 없다가 특정 규모 이상에서 **갑자기** 나타나는 능력을 의미합니다.

```
┌─────────────────────────────────────────────────────────────────┐
│                    창발(Emergence)의 특징                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Performance                                                    │
│       │                                                         │
│   100%├─────────────────────────────────────●───── 대형 모델     │
│       │                                   ╱                     │
│       │                                 ╱                       │
│    50%├─ · · · · · · · · · · · · · · ╱· · · · · · · · · · · ·  │
│       │                            ╱                            │
│       │                          ╱  ← Phase Transition          │
│       │                        ╱     (상전이)                    │
│       │──────────────────────●─────────────────── 소형 모델      │
│     0%├───────┬───────┬───────┬───────┬───────┬──────▶          │
│       │     10^21  10^22  10^23  10^24  10^25    Training FLOPs │
│                                                                 │
│  핵심 특징:                                                       │
│  • Random guessing → 갑작스러운 성능 향상                         │
│  • 중간 단계가 거의 없음                                          │
│  • 예측이 어려움 (smooth하지 않음)                                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 Emergence vs Gradual Improvement

```python
"""
두 가지 성능 향상 패턴:

1. Gradual (점진적) - Scaling Law 따름
   - Loss가 power law로 천천히 감소
   - 예측 가능
   - 예: Perplexity, 일반적인 생성 품질

2. Emergent (창발적) - 갑작스러운 전환
   - 특정 규모까지 random, 이후 급격히 향상
   - 예측 어려움
   - 예: 다자리 연산, Chain-of-Thought, 코드 생성

왜 Emergence가 발생하는가?
- 가설 1: 충분한 capacity가 필요한 태스크
- 가설 2: 여러 서브스킬의 조합 필요
- 가설 3: Metric의 문제 (accuracy는 threshold 기반)
"""
```

---

## 2. 주요 Emergent Abilities

### 2.1 능력 카탈로그

| 능력 | 설명 | 출현 규모 (대략) |
|------|------|-----------------|
| **Arithmetic** | 다자리 덧셈/뺄셈 | ~10^22 FLOPs |
| **Word Unscrambling** | 섞인 문자 복원 | ~10^22 FLOPs |
| **Chain-of-Thought** | 단계적 추론 | ~10^23 FLOPs |
| **Multi-step Math** | 복잡한 수학 문제 | ~10^23 FLOPs |
| **Code Generation** | 복잡한 코드 작성 | ~10^23 FLOPs |
| **Translation** (저자원) | 학습 데이터 적은 언어 번역 | ~10^23 FLOPs |
| **Analogical Reasoning** | 유추 추론 | ~10^24 FLOPs |
| **Theory of Mind** | 타인의 믿음/의도 추론 | ~10^24 FLOPs |

### 2.2 BIG-bench 태스크 분석

```
┌─────────────────────────────────────────────────────────────────┐
│              BIG-bench에서 관찰된 Emergent Tasks                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  □ = Linear improvement (Scaling Law)                           │
│  ● = Emergent (Phase Transition)                                │
│                                                                 │
│  10^21 ─┬─ □ 기본 문법                                           │
│         │   □ 간단한 QA                                          │
│  10^22 ─┼─ □ 요약                                                │
│         │   ● 3자리 덧셈                                         │
│         │   ● Word unscrambling                                 │
│  10^23 ─┼─ □ 번역 (일반)                                         │
│         │   ● Chain-of-Thought                                  │
│         │   ● 다단계 수학                                        │
│         │   ● 코드 생성                                          │
│  10^24 ─┼─ □ 창의적 글쓰기                                       │
│         │   ● 유추 추론                                          │
│         │   ● Theory of Mind                                    │
│  10^25 ─┴─ ● 복잡한 논리 추론                                     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. Chain-of-Thought (CoT)

### 3.1 CoT의 발견

Chain-of-Thought는 2022년 Google의 Wei et al. 논문에서 체계적으로 연구되었습니다.

```
┌─────────────────────────────────────────────────────────────────┐
│              Chain-of-Thought Prompting 비교                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Standard Prompting:                                            │
│  ────────────────────────────────────────────────               │
│  Q: Roger has 5 tennis balls. He buys 2 cans of balls.          │
│     Each can has 3 balls. How many balls does he have?          │
│  A: 11                                                          │
│                                                                 │
│  → 작은 모델: 자주 틀림 (예: "8", "6")                            │
│  → 큰 모델도 복잡한 문제에서 실패                                  │
│                                                                 │
│  Chain-of-Thought Prompting:                                    │
│  ────────────────────────────────────────────────               │
│  Q: Roger has 5 tennis balls. He buys 2 cans of balls.          │
│     Each can has 3 balls. How many balls does he have?          │
│  A: Roger started with 5 balls.                                 │
│     He bought 2 cans × 3 balls = 6 balls.                       │
│     Total: 5 + 6 = 11 balls.                                    │
│     The answer is 11.                                           │
│                                                                 │
│  → 중간 추론 단계를 명시적으로 생성                                │
│  → 복잡한 문제에서 정확도 크게 향상                                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 CoT 구현

```python
def standard_prompt(question):
    """Standard prompting - 답만 요청"""
    return f"""
Answer the following question:

Q: {question}
A:"""

def cot_prompt(question):
    """Chain-of-Thought prompting - 추론 과정 유도"""
    return f"""
Answer the following question step by step.
Show your reasoning before giving the final answer.

Q: {question}
A: Let's think step by step."""

def few_shot_cot_prompt(question, examples):
    """Few-shot CoT - 예시와 함께"""
    prompt = "Solve the following problems step by step:\n\n"

    for ex in examples:
        prompt += f"Q: {ex['question']}\n"
        prompt += f"A: {ex['reasoning']}\n"
        prompt += f"   The answer is {ex['answer']}.\n\n"

    prompt += f"Q: {question}\n"
    prompt += "A: Let's think step by step."

    return prompt

# 예시 사용
examples = [
    {
        "question": "There are 15 trees in the grove. Grove workers plant trees today. After they are done, there will be 21 trees. How many trees did they plant?",
        "reasoning": "We start with 15 trees. Later we have 21 trees. The difference is 21 - 15 = 6.",
        "answer": "6"
    },
    {
        "question": "If there are 3 cars in the parking lot and 2 more arrive, how many cars are there?",
        "reasoning": "There are 3 cars initially. 2 more arrive. 3 + 2 = 5.",
        "answer": "5"
    }
]

question = "Janet's ducks lay 16 eggs per day. She eats 3 for breakfast and bakes muffins with 4. She sells the rest at $2 each. How much does she make daily?"

prompt = few_shot_cot_prompt(question, examples)
# GPT-4 Response:
# "Janet's ducks lay 16 eggs per day.
#  She uses 3 + 4 = 7 eggs.
#  She sells 16 - 7 = 9 eggs.
#  At $2 each: 9 × $2 = $18.
#  The answer is $18."
```

### 3.3 CoT가 효과적인 이유

```
┌─────────────────────────────────────────────────────────────────┐
│                 Chain-of-Thought 작동 원리 가설                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  가설 1: 작업 메모리 확장                                         │
│  ─────────────────────────────────                              │
│  • 중간 결과를 텍스트로 저장                                      │
│  • Transformer의 제한된 context를 우회                           │
│  • "외부 메모리"처럼 활용                                         │
│                                                                 │
│  가설 2: 문제 분해                                                │
│  ─────────────────────────────────                              │
│  • 복잡한 문제를 작은 단계로 분해                                  │
│  • 각 단계는 모델이 이미 할 수 있는 것                             │
│  • 단계들의 조합으로 복잡한 문제 해결                              │
│                                                                 │
│  가설 3: 분포 이동                                                │
│  ─────────────────────────────────                              │
│  • 학습 데이터에 추론 과정이 포함                                  │
│  • "step by step"이 해당 분포를 활성화                            │
│  • 학습된 패턴을 재사용                                           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 4. CoT의 변형들

### 4.1 Zero-shot CoT

```python
def zero_shot_cot(question):
    """
    Zero-shot CoT: "Let's think step by step"만 추가

    Kojima et al. (2022) 발견:
    - 예시 없이도 효과적
    - 다양한 추론 태스크에 적용 가능
    """
    return f"""
Q: {question}
A: Let's think step by step."""

# 간단하지만 효과적!
question = "A juggler can juggle 16 balls. Half are golf balls. Half of the golf balls are blue. How many blue golf balls?"
# Response: "16 balls total. Half are golf balls: 16/2 = 8. Half of golf balls are blue: 8/2 = 4. The answer is 4."
```

### 4.2 Self-Consistency

```python
def self_consistency(question, model, n_samples=5, temperature=0.7):
    """
    Self-Consistency: 여러 추론 경로를 생성하고 다수결

    Wang et al. (2022):
    - 같은 문제에 여러 CoT 생성
    - 최종 답변에 대해 투표
    - CoT만 사용할 때보다 정확도 향상
    """
    prompt = cot_prompt(question)
    answers = []

    for _ in range(n_samples):
        response = model.generate(prompt, temperature=temperature)
        # 최종 답변 추출 (예: "The answer is X" 패턴)
        answer = extract_answer(response)
        answers.append(answer)

    # 다수결
    from collections import Counter
    most_common = Counter(answers).most_common(1)[0][0]
    return most_common

# 예시 결과:
# Sample 1: "... The answer is 4."
# Sample 2: "... The answer is 4."
# Sample 3: "... The answer is 8."  (오류)
# Sample 4: "... The answer is 4."
# Sample 5: "... The answer is 4."
# Final: 4 (4/5 투표)
```

### 4.3 Tree of Thoughts (ToT)

```
┌─────────────────────────────────────────────────────────────────┐
│                    Tree of Thoughts                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  CoT: 단일 선형 경로                                              │
│       Start → Step1 → Step2 → Step3 → Answer                    │
│                                                                 │
│  ToT: 트리 형태의 탐색                                            │
│                                                                 │
│                        Start                                    │
│                       /  |  \                                   │
│                    A1   A2   A3                                 │
│                   / \    |    \                                 │
│                 B1  B2  B3    B4                                │
│                 |    ✗   |     |                                │
│                C1       C2    C3                                │
│                 |        |     ✗                                │
│              Answer    Answer                                   │
│                                                                 │
│  특징:                                                           │
│  • 여러 경로 동시 탐색                                            │
│  • 각 단계에서 평가 후 가지치기                                    │
│  • BFS/DFS 탐색 전략                                             │
│  • 복잡한 계획/퍼즐 문제에 효과적                                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

```python
def tree_of_thoughts(problem, model, breadth=3, depth=3):
    """
    Tree of Thoughts 구현 개요

    Yao et al. (2023):
    - 여러 "생각" 후보 생성
    - 각 생각을 평가
    - 유망한 경로만 확장
    """
    def generate_thoughts(state, n=breadth):
        """현재 상태에서 가능한 다음 생각들 생성"""
        prompt = f"Given: {state}\nGenerate {n} possible next steps:"
        return model.generate(prompt).split('\n')[:n]

    def evaluate_thought(state, thought):
        """생각의 유망성 평가 (0-1)"""
        prompt = f"State: {state}\nThought: {thought}\nRate this step (0-10):"
        score = model.generate(prompt)
        return float(score) / 10

    def solve(state, current_depth=0):
        if current_depth >= depth:
            return state

        thoughts = generate_thoughts(state)
        scored = [(t, evaluate_thought(state, t)) for t in thoughts]
        best_thought = max(scored, key=lambda x: x[1])[0]

        new_state = state + " → " + best_thought
        return solve(new_state, current_depth + 1)

    return solve(problem)
```

---

## 5. 능력 유도 (Capability Elicitation)

### 5.1 왜 유도가 필요한가?

```
┌─────────────────────────────────────────────────────────────────┐
│                    능력 유도의 필요성                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  문제: 모델이 능력을 "갖고 있지만" 발휘하지 않음                    │
│                                                                 │
│  예시:                                                           │
│  ──────                                                         │
│  Q: What's 37 × 23?                                             │
│  A: 851 (정답)                                                   │
│                                                                 │
│  Q: Calculate 37 times 23 without showing work.                 │
│  A: 852 (오답)                                                   │
│                                                                 │
│  같은 모델, 같은 문제인데 프롬프트에 따라 다른 결과!                │
│                                                                 │
│  해결: 적절한 프롬프팅으로 잠재 능력 유도                           │
│  • CoT: "Let's think step by step"                              │
│  • Role: "You are an expert mathematician"                      │
│  • Format: "Show your calculation"                              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 5.2 유도 기법들

```python
"""
주요 Capability Elicitation 기법:

1. Role Assignment (역할 부여)
   "You are a world-class programmer..."
   "Act as a senior software engineer..."

2. Step-by-step Instructions
   "First, understand the problem..."
   "Then, break it down..."

3. Format Specification
   "Answer in JSON format"
   "Provide your reasoning, then the answer"

4. Confidence Calibration
   "If unsure, say 'I don't know'"
   "Rate your confidence (1-10)"

5. Self-Verification
   "Check your answer"
   "Verify each step"
"""

def enhanced_prompt(question, technique="all"):
    """다양한 유도 기법을 결합한 프롬프트"""

    prompt = """You are an expert problem solver. Follow these steps carefully:

1. First, understand what the question is asking
2. Identify the key information and constraints
3. Think through the solution step by step
4. Double-check your reasoning
5. Provide your final answer clearly

Question: {question}

Solution:
Let me work through this systematically.
"""
    return prompt.format(question=question)

# 사용 예시
question = "A train travels 60 km in the first hour and 80 km in the second hour. What is its average speed?"
prompt = enhanced_prompt(question)
```

### 5.3 Persona/Role의 효과

```python
# 실험: 같은 문제, 다른 역할

prompts = {
    "basic": "Solve: {problem}",

    "expert": """You are a mathematics professor with 30 years of experience.
Solve the following problem with the precision and rigor expected in academia.

Problem: {problem}""",

    "teacher": """You are a patient high school math teacher.
Explain your solution clearly so a student can follow along.

Problem: {problem}""",

    "programmer": """You are a software engineer.
Approach this problem systematically, as if writing an algorithm.

Problem: {problem}"""
}

# 연구 결과:
# - Expert persona: 복잡한 문제에서 정확도 향상
# - Teacher persona: 설명 품질 향상
# - Programmer persona: 구조화된 접근

# 주의: 효과는 모델 크기에 따라 다름
# 작은 모델: persona 효과 미미
# 큰 모델: 의미있는 차이 발생
```

---

## 6. Emergence 논쟁

### 6.1 "Emergence is a Mirage?" (2023)

```
┌─────────────────────────────────────────────────────────────────┐
│            Emergence에 대한 반론 (Schaeffer et al. 2023)          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  주장: Emergence는 metric의 artifact일 수 있음                   │
│                                                                 │
│  논거:                                                           │
│  ────────                                                       │
│  1. Accuracy는 "all-or-nothing" metric                          │
│     • 부분 정답 = 0점                                            │
│     • 실제로는 점진적 향상이 있었을 수 있음                        │
│                                                                 │
│  2. Continuous metric으로 측정 시:                               │
│     • Brier score, log-likelihood 등                            │
│     • "갑작스러운 전환" 사라짐                                    │
│     • 대신 smooth한 향상 관찰                                    │
│                                                                 │
│  3. 예시: 다자리 덧셈                                             │
│     • Accuracy: 0% → 0% → 100% (emerge!)                        │
│     • Token-level acc: 40% → 60% → 100% (smooth)               │
│                                                                 │
│  결론 (논쟁적):                                                   │
│  • "True emergence"는 metric 선택의 문제일 수 있음                │
│  • 그러나 실용적으로 중요한 것은 task performance                 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 6.2 현재 컨센서스

```python
"""
Emergence 논쟁의 현재 상태 (2024):

찬성 측:
- 일부 능력은 확실히 갑자기 나타남 (실용적 관점)
- In-context learning 자체가 emergent
- 복잡한 추론 능력은 threshold 존재

반대 측:
- Metric 선택이 "emergence" 환상 생성
- 충분히 세밀한 metric으로는 smooth
- "Predictable" scaling으로 설명 가능

실용적 합의:
- "Emergence"든 아니든, 특정 규모 이상에서 유용한 능력 발현
- 능력 예측은 여전히 어려움
- Capability elicitation이 중요
"""
```

---

## 7. 실습: Emergence 관찰하기

### 7.1 모델 크기별 능력 비교

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def compare_model_capabilities(question, model_names):
    """
    여러 크기의 모델에서 같은 문제를 테스트
    """
    results = {}

    for model_name in model_names:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )

        inputs = tokenizer(question, return_tensors="pt").to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=200, temperature=0.1)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        results[model_name] = response

    return results

# 테스트할 모델들 (크기순)
models = [
    "microsoft/phi-2",           # 2.7B
    "meta-llama/Llama-2-7b-hf",  # 7B
    "meta-llama/Llama-2-13b-hf", # 13B
    "meta-llama/Llama-2-70b-hf", # 70B
]

# Emergence 테스트 질문들
test_questions = {
    "arithmetic": "What is 347 × 29? Show your work.",
    "reasoning": """
        If John is taller than Mary, and Mary is taller than Tom,
        is John taller than Tom? Explain your reasoning.
    """,
    "code": """
        Write a Python function to find the nth Fibonacci number
        using dynamic programming.
    """,
}

# 결과 비교 (실행 시 상당한 메모리 필요)
# for q_name, question in test_questions.items():
#     print(f"\n=== {q_name} ===")
#     results = compare_model_capabilities(question, models)
#     for model, response in results.items():
#         print(f"\n{model}:\n{response}")
```

### 7.2 CoT 효과 측정

```python
def measure_cot_effect(questions, model, tokenizer):
    """
    Standard vs CoT 프롬프팅 효과 비교
    """
    results = {"standard": [], "cot": []}

    for q in questions:
        # Standard prompting
        standard = f"Q: {q['question']}\nA:"
        std_output = generate(model, tokenizer, standard)
        std_correct = check_answer(std_output, q['answer'])
        results["standard"].append(std_correct)

        # CoT prompting
        cot = f"Q: {q['question']}\nA: Let's think step by step."
        cot_output = generate(model, tokenizer, cot)
        cot_correct = check_answer(cot_output, q['answer'])
        results["cot"].append(cot_correct)

    # 정확도 계산
    std_acc = sum(results["standard"]) / len(questions)
    cot_acc = sum(results["cot"]) / len(questions)

    print(f"Standard Prompting Accuracy: {std_acc:.1%}")
    print(f"Chain-of-Thought Accuracy: {cot_acc:.1%}")
    print(f"Improvement: {cot_acc - std_acc:.1%}")

    return results

# 테스트 데이터셋 (GSM8K 스타일)
test_questions = [
    {"question": "Janet has 10 apples. She gives 3 to her friend. How many does she have?", "answer": "7"},
    {"question": "A store has 24 shirts. If 6 are sold each day, how many days until they're gone?", "answer": "4"},
    # ... 더 많은 문제
]
```

---

## 정리

### 핵심 개념
- **Emergent Abilities**: 규모에 따라 갑자기 나타나는 능력
- **Chain-of-Thought**: 단계적 추론으로 복잡한 문제 해결
- **Self-Consistency**: 다수결로 CoT 정확도 향상
- **Capability Elicitation**: 프롬프팅으로 잠재 능력 유도

### 실무 적용
1. 복잡한 추론 → CoT 사용
2. 높은 정확도 필요 → Self-consistency
3. 창의적 탐색 → Tree of Thoughts
4. 능력 극대화 → Role/Persona 설정

### 다음 단계
- [08_LLaMA_Family.md](08_LLaMA_Family.md): 최신 LLM 아키텍처
- [19_PEFT_Unified.md](19_PEFT_Unified.md): 효율적 적응 기법

---

## 참고 자료

### 핵심 논문
- Wei et al. (2022). "Emergent Abilities of Large Language Models"
- Wei et al. (2022). "Chain-of-Thought Prompting Elicits Reasoning in LLMs"
- Wang et al. (2022). "Self-Consistency Improves Chain of Thought Reasoning"
- Yao et al. (2023). "Tree of Thoughts: Deliberate Problem Solving with LLMs"
- Schaeffer et al. (2023). "Are Emergent Abilities of LLMs a Mirage?"

### 추가 자료
- [BIG-bench](https://github.com/google/BIG-bench)
- [Chain-of-Thought Hub](https://github.com/FranxYao/chain-of-thought-hub)

---

## 연습 문제

### 연습 문제 1: 지표 선택과 창발 착시(Emergence Mirage)

Schaeffer et al. (2023)은 창발적 능력이 지표 선택의 부산물일 수 있다고 주장합니다. 증가하는 컴퓨트 규모에 따라 다자리(multi-digit) 덧셈으로 평가받는 모델을 생각해보세요.

1. **정확도(Accuracy)** (완전 일치)를 지표로 사용하면 왜 "급격한 전환(phase transition)"처럼 보이는지 설명하세요.
2. **토큰 수준 정확도(Token-level accuracy)** (자릿수별 정확도)를 사용하면 왜 부드러운 곡선이 나타나는지 설명하세요.
3. 어떤 지표를 선호해야 하며, 각 지표는 어떤 맥락에서 적절할 수 있나요?

<details>
<summary>정답 보기</summary>

**파트 1: 정확도가 급격한 전환을 만드는 이유**

정확도(완전 일치)는 임계값 지표입니다: 전체 답이 맞으면(1) 또는 틀리면(0)입니다. 모델이 5자리 중 2자리를 맞추다가 4자리를 맞추게 되더라도 정확도 점수는 여전히 0입니다. 모든 자릿수를 맞추는 임계값을 넘어야만 정확도가 1로 급등합니다. 이렇게 점진적 향상을 가리기 때문에 갑작스러운 "창발"처럼 보입니다.

**파트 2: 토큰 수준 정확도가 부드러운 이유**

토큰 수준 정확도는 부분적 진전을 반영합니다: 5자리 중 3자리를 정확히 예측하면 0.6점을 받습니다. 모델이 규모가 커지면서 점점 더 많은 자릿수를 맞추면 이 지표는 완만하게 상승하여 내재된 점진적 향상을 드러냅니다.

**파트 3: 어떤 지표를 선호할 것인가**

- **정확도**는 태스크가 완전한 정확성을 요구할 때 적절합니다(예: 한 자릿수라도 틀린 계산기는 쓸모없음). 능력이 실용적으로 사용 가능한지를 올바르게 반영합니다.
- **토큰 수준 정확도**는 모델의 내부 학습 궤적을 이해하고 완전한 정확도가 언제 나타날지 예측하는 데 더 적합합니다.
- **창발 메커니즘 연구**를 위해서는 연속 지표가 선호됩니다. **배포 결정**을 위해서는 태스크 수준 정확도가 더 의미 있습니다.

어떤 지표도 "틀리지" 않습니다 — 서로 다른 질문에 답할 뿐입니다.

</details>

---

### 연습 문제 2: Self-Consistency 구현

아래 `self_consistency` 함수를 완성하세요. 이 함수는 수학 문제에 대해 Chain-of-Thought 프롬프팅을 사용하여 `n_samples`개의 응답을 생성하고, 각 응답에서 숫자 최종 답을 추출한 후, 가장 많이 나온 답을 반환해야 합니다.

```python
import re
from collections import Counter

def extract_numeric_answer(response: str) -> str:
    """응답에서 마지막으로 언급된 숫자를 추출합니다."""
    numbers = re.findall(r'\b\d+(?:\.\d+)?\b', response)
    return numbers[-1] if numbers else "unknown"

def self_consistency(question: str, model, n_samples: int = 5, temperature: float = 0.7) -> str:
    """
    TODO: Self-consistency 디코딩을 구현하세요.
    1. 질문에 대한 CoT 프롬프트 구성
    2. 모델에서 n_samples개의 응답 생성
    3. 각 응답에서 숫자 답 추출
    4. 다수결 투표 답 반환
    """
    # 여기에 구현하세요
    pass
```

<details>
<summary>정답 보기</summary>

```python
import re
from collections import Counter

def extract_numeric_answer(response: str) -> str:
    """응답에서 마지막으로 언급된 숫자를 추출합니다."""
    numbers = re.findall(r'\b\d+(?:\.\d+)?\b', response)
    return numbers[-1] if numbers else "unknown"

def self_consistency(question: str, model, n_samples: int = 5, temperature: float = 0.7) -> str:
    """
    Self-consistency 디코딩:
    여러 CoT 경로를 생성하고 다수결 투표를 수행합니다.
    """
    # 1단계: CoT 프롬프트 구성
    cot_prompt = f"Q: {question}\nA: Let's think step by step."

    # 2단계: 여러 응답 생성
    answers = []
    for _ in range(n_samples):
        response = model.generate(cot_prompt, temperature=temperature)
        answer = extract_numeric_answer(response)
        answers.append(answer)

    # 3단계: 다수결 투표
    counter = Counter(answers)
    majority_answer = counter.most_common(1)[0][0]

    return majority_answer

# 왜 작동하는가:
# - 잘못된 추론 경로는 다양한 오답을 생성하는 경향이 있음
# - 올바른 추론 경로는 같은 정답으로 수렴하는 경향이 있음
# - 다수결 투표로 개별적인 오류를 걸러냄
```

</details>

---

### 연습 문제 3: 능력 이끌어내기(Capability Elicitation) 실험 설계

**역할 부여(Role assignment, 페르소나 프롬프팅)**가 논리 퍼즐에서 모델의 정확도를 향상시키는지 테스트하는 실험을 설계하세요. 다음을 명시하세요:

1. 독립 변수(조작하는 것)
2. 종속 변수(측정하는 것)
3. 통제 조건과 최소 두 가지 처치 조건
4. 통제해야 할 잠재적 교란 변수(confound)

<details>
<summary>정답 보기</summary>

**실험 설계:**

**1. 독립 변수:** 시스템 프롬프트에서 모델에 부여하는 역할/페르소나.

**2. 종속 변수:** 표준화된 논리 퍼즐 세트(예: 공개된 논리 벤치마크의 50문제)에 대한 정확도(% 정확).

**3. 조건:**
- **통제(Control):** 페르소나 없음 — 단순 프롬프트: `"Solve the following logic puzzle: {puzzle}"`
- **처치 A (전문가):** `"You are a world-class logician with 30 years of experience. Solve the following logic puzzle: {puzzle}"`
- **처치 B (교사):** `"You are a patient teacher explaining logical reasoning. Solve the following logic puzzle step by step: {puzzle}"`

**4. 잠재적 교란 변수:**
- **온도(Temperature)와 샘플링:** 다른 프롬프트가 샘플링 무작위성과 다르게 상호작용할 수 있습니다. `temperature=0` (탐욕적 디코딩)으로 통제하거나, 동일한 온도로 여러 시드를 사용하여 평균을 냅니다.
- **문제 순서 효과:** 위치 편향을 방지하기 위해 조건 간 퍼즐 순서를 무작위화합니다.
- **CoT vs CoT 없음:** "교사" 프롬프트는 암묵적으로 단계별 추론을 장려합니다. 페르소나 효과와 CoT 효과를 분리하려면 모든 조건에서 "step by step" 표현을 동일하게 포함하거나 제외해야 합니다.

</details>

---

### 연습 문제 4: Tree of Thoughts vs Chain-of-Thought

CoT (Chain-of-Thought)와 ToT (Tree of Thoughts)를 다음 차원에서 비교하세요. 각 셀에 트레이드오프를 간단히 설명하세요.

| 차원 | CoT | ToT |
|------|-----|-----|
| 탐색 전략 | ? | ? |
| 계산 비용 | ? | ? |
| 적합한 문제 유형 | ? | ? |
| 실패 유형 | ? | ? |

<details>
<summary>정답 보기</summary>

| 차원 | CoT | ToT |
|------|-----|-----|
| **탐색 전략** | 단일 선형 경로 — 탐욕적 좌→우 생성. 빠르지만 잠재적으로 잘못된 방향에 일찍 확정됨. | 트리 구조 BFS/DFS — 여러 분기를 탐색하고 역추적. 더 철저하지만 많은 부분 상태를 생성하고 평가해야 함. |
| **계산 비용** | 낮음 — 단일 응답 길이에 비례. 일반적으로 1× 비용. | 높음 — 분기 계수 × 깊이 × 평가 비용에 비례. CoT보다 10-100× 더 비쌀 수 있음. |
| **적합한 문제 유형** | 각 단계가 자연스럽게 이전 단계에서 따르는 순차적 추론 태스크 (산술, 간단한 QA). | 초기 결정이 이후 옵션을 강하게 제약하고 역추적이 가치 있는 조합적 또는 계획 문제 (퍼즐, 다단계 계획, 코드 디버깅). |
| **실패 유형** | **연쇄 오류(Cascading error)** — 초기의 한 가지 잘못된 단계가 전체 체인에 전파되어 확신에 찬 오답을 생성. | **평가 병목(Evaluation bottleneck)** — 품질이 노드 평가 함수의 정확도에 달려 있음. 빈약한 평가자는 올바른 분기를 잘못 가지치기하거나 잘못된 분기에 과투자함. |

</details>
