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
