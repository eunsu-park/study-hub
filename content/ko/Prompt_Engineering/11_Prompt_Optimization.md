# 11. 프롬프트 최적화(Prompt Optimization)

**이전**: [RAG 프롬프트 패턴](./10_RAG_Prompt_Patterns.md) | **다음**: [평가와 지표](./12_Evaluation_and_Metrics.md)

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 수동 프롬프트 엔지니어링의 근본적인 확장성 한계와 자동화된 최적화가 필요한 시점을 설명하기
2. DSPy 프레임워크를 적용하여 시그니처를 정의하고, 모듈을 구축하고, 체계적인 프롬프트 튜닝을 위한 옵티마이저를 실행하기
3. LLM 기반 프롬프트 최적화의 대표적 접근법으로 OPRO와 APE를 설명하기
4. 작업 성능을 유지하면서 토큰 사용량을 줄이는 프롬프트 압축(Prompt Compression) 기법을 구현하기
5. 비용-품질 트레이드오프를 평가하고 프롬프트 최적화 vs 모델 전환 시점을 결정하기

---

수동 프롬프트 엔지니어링 -- 프롬프트를 손으로 반복적으로 편집하고, 몇 가지 예시에서 테스트하고, 직관에 기반하여 조정하는 과정 -- 은 간단한 작업에 놀라울 정도로 잘 작동합니다. 하지만 한계에 부딪힙니다. 서로 다른 사용자 집단을 대상으로 하는 수십 개의 프롬프트가 있을 때, 작은 문구 변경이 예상치 못한 회귀를 일으킬 때, 품질, 비용, 지연 시간 등 여러 지표를 동시에 최적화해야 할 때, 수동 반복은 병목이 됩니다. 50개의 프롬프트 변형을 손으로 A/B 테스트할 수 없습니다. 가능한 지시사항의 공간을 체계적으로 탐색할 수 없습니다. "개선된" 프롬프트가 테스트하는 것을 잊은 엣지 케이스를 깨뜨리지 않는다고 보장할 수 없습니다.

이 레슨에서는 *자동 프롬프트 최적화(Automatic Prompt Optimization)*의 신흥 분야를 다룹니다: 프롬프트 설계를 최적화 문제로 취급하고 알고리즘적 검색을 적용하여 더 나은 프롬프트를 찾는 도구와 기법입니다. 개념적 기초부터 시작하여, 주요 프레임워크(DSPy, OPRO, APE)를 탐색하고, 비용-품질 트레이드오프와 프롬프트 압축 같은 실용적 관심사를 다룹니다.

## 목차

1. [수동 프롬프트 엔지니어링의 한계](#1-수동-프롬프트-엔지니어링의-한계)
2. [DSPy 프레임워크](#2-dspy-프레임워크)
3. [OPRO: 프롬프팅에 의한 최적화](#3-opro-프롬프팅에-의한-최적화optimization-by-prompting)
4. [APE: 자동 프롬프트 엔지니어](#4-ape-자동-프롬프트-엔지니어automatic-prompt-engineer)
5. [자동 프롬프트 생성](#5-자동-프롬프트-생성)
6. [프롬프트를 위한 기울기 없는 최적화](#6-프롬프트를-위한-기울기-없는-최적화gradient-free-optimization-for-prompts)
7. [베이지안 프롬프트 최적화](#7-베이지안-프롬프트-최적화bayesian-prompt-optimization)
8. [프롬프트 압축](#8-프롬프트-압축prompt-compression)
9. [비용-품질 트레이드오프](#9-비용-품질-트레이드오프cost-quality-trade-offs)
10. [최적화 vs 모델 전환 시점](#10-최적화-vs-모델-전환-시점)

---

## 1. 수동 프롬프트 엔지니어링의 한계

### 1.1 확장성 문제

수동 프롬프트 엔지니어링은 다음 루프를 통해 작동합니다:

```
Write prompt → Test on examples → Read outputs → Edit prompt → Repeat
```

이것은 여러 방식으로 무너집니다:

| 문제 | 설명 |
|------|------|
| **평가 편향(Evaluation Bias)** | 사람은 5-10개 예시로 테스트; 실제 워크로드에는 수천 개의 엣지 케이스가 있음 |
| **지역 최적값(Local Optima)** | 작은 편집은 프롬프트 공간의 아주 작은 이웃만 탐색 |
| **상호작용 효과(Interaction Effects)** | 하나의 지시를 변경하면 다른 것을 깨뜨릴 수 있음; 수동으로 추적하기 어려움 |
| **다중 목표(Multi-objective)** | 정확도 vs 비용 vs 지연 시간을 동시에 최적화하는 것은 직관적이지 않음 |
| **재현성(Reproducibility)** | "문구를 조정했더니 나아졌다"는 방법론이 아님 |
| **버전 폭발(Version Explosion)** | 수십 개 프롬프트 x 여러 모델 x 다양한 사용 사례 = 관리 불가 |

### 1.2 프롬프트 공간은 광대하다

50개 단어로 된 간단한 지시 프롬프트를 생각해 보세요. 합리적인 영어 의역으로만 제한하더라도, 동일한 지시를 표현하는 수천 가지 의미적으로 동등한 방법이 있습니다. 각 변형은 다른 모델 동작을 생성할 수 있습니다. 수동 탐색은 이 공간의 아주 작은 부분만 커버합니다.

```python
# Example: These prompts are semantically similar but produce different results
prompts = [
    "Classify the sentiment of this review as positive or negative.",
    "Determine whether this review expresses a positive or negative sentiment.",
    "Is the sentiment of the following review positive or negative? Answer with one word.",
    "Read the review below. Output POSITIVE or NEGATIVE based on the overall sentiment.",
    "You are a sentiment classifier. Given a product review, output the sentiment label.",
]
# A human might try 2-3 of these. An optimizer tests all of them (and more).
```

### 1.3 자동화된 최적화를 고려할 시점

자동화된 프롬프트 최적화가 투자할 가치가 있는 경우:

1. **대량 프로덕션 작업**: 2% 정확도 향상이 중요한 일일 수천 건의 호출
2. **비용 민감 배포**: 프롬프트 길이를 30% 줄이면 상당한 비용 절감
3. **다중 프롬프트 시스템**: 프롬프트가 상호작용하는 경우(예: 에이전트 파이프라인) 하나의 프롬프트를 수동 조정하면 다른 것에 영향
4. **모델 마이그레이션**: 모델을 전환하고 모든 프롬프트를 재조정해야 할 때
5. **측정 가능한 목표**: 명확한 평가 지표(정확도, F1, 정확 일치 등)가 있을 때

자동화된 최적화가 가치가 없는 경우:
- 평가 데이터셋이 없는 경우
- 작업이 창의적/주관적인 경우 (명확한 "정답"이 없음)
- 프로토타이핑 중이고 프롬프트가 근본적으로 변경될 경우
- 프롬프트가 드물게 사용되는 경우 (하루 100건 미만)

---

## 2. DSPy 프레임워크

DSPy(Declarative Self-improving Language Programs)는 프로그래밍 방식의 프롬프트 최적화를 위한 가장 성숙한 프레임워크입니다. 프롬프트를 수동으로 작성하는 대신, LLM이 *무엇을* 해야 하는지(시그니처를 통해) 선언하고 DSPy의 옵티마이저가 *어떻게* 프롬프트를 작성할지 결정하게 합니다.

### 2.1 핵심 개념

```
┌─────────────┐    ┌──────────────┐    ┌──────────────┐
│  Signature   │───▶│   Module      │───▶│  Optimizer    │
│  (What)      │    │  (How)        │    │  (Search)     │
└─────────────┘    └──────────────┘    └──────────────┘
   Input/Output       Prompt strategy     Find best prompt
   declaration         (CoT, ReAct,        for your data
                       etc.)
```

- **시그니처(Signature)**: 입력과 출력 필드를 선언 (예: `"question -> answer"`)
- **모듈(Module)**: 프롬프팅 전략으로 시그니처를 래핑 (예: `dspy.ChainOfThought`)
- **옵티마이저(Optimizer, Teleprompter)**: 학습 데이터에서 평가하여 최적의 프롬프트/예시를 검색

### 2.2 기본 DSPy 프로그램

```python
import dspy

# Configure the language model
lm = dspy.LM("anthropic/claude-sonnet-4-20250514", api_key="your-key")
dspy.configure(lm=lm)

# Define a signature: input -> output
class SentimentClassification(dspy.Signature):
    """Classify the sentiment of a product review."""
    review: str = dspy.InputField(desc="Product review text")
    sentiment: str = dspy.OutputField(desc="Either 'positive' or 'negative'")

# Create a simple module (zero-shot)
classify = dspy.Predict(SentimentClassification)

# Use it
result = classify(review="This laptop is amazing! Best purchase I've made.")
print(result.sentiment)  # "positive"
```

### 2.3 사고의 연쇄 모듈(Chain-of-Thought Module)

```python
import dspy

# Chain-of-Thought adds reasoning before the answer
class FactCheck(dspy.Signature):
    """Determine if a claim is supported by the provided evidence."""
    evidence: str = dspy.InputField(desc="Source text with factual information")
    claim: str = dspy.InputField(desc="Claim to verify against the evidence")
    verdict: str = dspy.OutputField(desc="SUPPORTED, REFUTED, or NOT_ENOUGH_INFO")

# Wrap with ChainOfThought -- DSPy adds "reasoning" automatically
fact_checker = dspy.ChainOfThought(FactCheck)

result = fact_checker(
    evidence="The Eiffel Tower was completed in 1889 and stands 330 meters tall.",
    claim="The Eiffel Tower is taller than 300 meters."
)
print(result.reasoning)  # Shows the model's reasoning process
print(result.verdict)    # "SUPPORTED"
```

### 2.4 다단계 프로그램(Multi-Step Programs)

```python
import dspy

class QuestionToQuery(dspy.Signature):
    """Convert a natural language question to a search query."""
    question: str = dspy.InputField()
    search_query: str = dspy.OutputField()

class AnswerFromContext(dspy.Signature):
    """Answer a question based on retrieved context."""
    context: str = dspy.InputField(desc="Retrieved documents")
    question: str = dspy.InputField()
    answer: str = dspy.OutputField()

class RAGPipeline(dspy.Module):
    def __init__(self):
        self.query_gen = dspy.Predict(QuestionToQuery)
        self.answer_gen = dspy.ChainOfThought(AnswerFromContext)

    def forward(self, question: str) -> str:
        # Step 1: Generate search query
        query_result = self.query_gen(question=question)

        # Step 2: Retrieve documents (your retrieval function)
        context = retrieve_documents(query_result.search_query)

        # Step 3: Generate answer from context
        answer_result = self.answer_gen(context=context, question=question)
        return answer_result.answer

def retrieve_documents(query: str) -> str:
    """Placeholder: your actual retrieval logic here."""
    # In production: vector search, BM25, hybrid, etc.
    return f"Retrieved context for: {query}"

rag = RAGPipeline()
answer = rag("What year was the Python programming language created?")
```

### 2.5 DSPy로 최적화하기

DSPy의 진정한 힘은 옵티마이저에 있습니다. 학습 예시가 주어지면, 옵티마이저가 최적의 프롬프트 구성을 검색합니다:

```python
import dspy
from dspy.evaluate import Evaluate

# Define your training data
trainset = [
    dspy.Example(
        review="Terrible product, broke after one day",
        sentiment="negative"
    ).with_inputs("review"),
    dspy.Example(
        review="Absolutely love it! Works perfectly",
        sentiment="positive"
    ).with_inputs("review"),
    # ... more examples (aim for 50-200)
]

# Define your metric
def accuracy_metric(example, prediction, trace=None):
    return example.sentiment.lower() == prediction.sentiment.lower()

# Choose an optimizer
optimizer = dspy.BootstrapFewShot(
    metric=accuracy_metric,
    max_bootstrapped_demos=4,  # Max few-shot examples to include
    max_labeled_demos=4,       # Max labeled examples to use
)

# Optimize the module
classify = dspy.Predict(SentimentClassification)
optimized_classify = optimizer.compile(classify, trainset=trainset)

# The optimized module now includes automatically selected few-shot examples
# and potentially rewritten instructions
result = optimized_classify(review="Not worth the money, very disappointing")
print(result.sentiment)

# Evaluate on a test set
evaluate = Evaluate(devset=testset, metric=accuracy_metric, num_threads=4)
score = evaluate(optimized_classify)
print(f"Accuracy: {score}%")
```

### 2.6 고급 옵티마이저

```python
import dspy

# MIPROv2: Optimizes both instructions AND few-shot examples
optimizer = dspy.MIPROv2(
    metric=accuracy_metric,
    num_candidates=10,      # Number of instruction candidates to generate
    init_temperature=1.0,   # Higher = more diverse candidates
)
optimized = optimizer.compile(classify, trainset=trainset)

# BootstrapFewShotWithRandomSearch: Adds random search over configurations
optimizer = dspy.BootstrapFewShotWithRandomSearch(
    metric=accuracy_metric,
    max_bootstrapped_demos=4,
    num_candidate_programs=16,  # Number of random configurations to try
)
optimized = optimizer.compile(classify, trainset=trainset)
```

### 2.7 최적화된 프로그램 저장 및 로드

```python
import dspy

# Save the optimized program
optimized_classify.save("optimized_sentiment.json")

# Load it later
loaded_classify = dspy.Predict(SentimentClassification)
loaded_classify.load("optimized_sentiment.json")

# Use in production
result = loaded_classify(review="Great product!")
```

---

## 3. OPRO: 프롬프팅에 의한 최적화(Optimization by PROmpting)

OPRO(Yang et al., 2023)는 LLM 자체를 옵티마이저로 사용합니다. 외부 검색 알고리즘 대신, OPRO는 이전 프롬프트의 성능을 기반으로 LLM에게 더 나은 프롬프트를 생성하도록 요청합니다.

### 3.1 OPRO 개념

```
┌────────────────────────────────────┐
│         OPRO Optimization Loop     │
│                                     │
│  1. Start with initial prompt(s)    │
│  2. Evaluate on training examples   │
│  3. Show LLM the prompt-score pairs │
│  4. Ask LLM to generate a better    │
│     prompt                          │
│  5. Evaluate the new prompt         │
│  6. Repeat from step 3             │
└────────────────────────────────────┘
```

### 3.2 OPRO 구현

```python
import anthropic
import json
from dataclasses import dataclass

client = anthropic.Anthropic()

@dataclass
class PromptScore:
    prompt: str
    score: float

def evaluate_prompt(prompt: str, test_cases: list[dict]) -> float:
    """Evaluate a prompt on test cases and return accuracy."""
    correct = 0
    for case in test_cases:
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=100,
            messages=[{
                "role": "user",
                "content": prompt.format(input=case["input"])
            }]
        )
        response = message.content[0].text.strip().lower()
        if response == case["expected"].lower():
            correct += 1
    return correct / len(test_cases)

def opro_optimize(
    initial_prompts: list[str],
    test_cases: list[dict],
    num_iterations: int = 10,
    candidates_per_iteration: int = 5
) -> PromptScore:
    """OPRO-style prompt optimization using LLM as optimizer."""
    # Evaluate initial prompts
    history: list[PromptScore] = []
    for prompt in initial_prompts:
        score = evaluate_prompt(prompt, test_cases)
        history.append(PromptScore(prompt=prompt, score=score))
        print(f"Initial prompt score: {score:.2f}")

    best = max(history, key=lambda x: x.score)

    for iteration in range(num_iterations):
        # Build the meta-prompt showing history
        history_text = "\n".join(
            f"Prompt: \"{ps.prompt}\"\nAccuracy: {ps.score:.2f}\n"
            for ps in sorted(history, key=lambda x: x.score)[-10:]  # Show top 10
        )

        meta_prompt = f"""You are an expert prompt engineer. Your task is to
generate a better prompt for a text classification task.

Here are previous prompts and their accuracy scores (higher is better):

{history_text}

The task is to classify text sentiment as "positive" or "negative".
The prompt should contain {{input}} as a placeholder for the text to classify.

Generate {candidates_per_iteration} new prompt variants that might score higher.
Learn from the patterns in high-scoring prompts.
Return each prompt on a separate line, prefixed with "PROMPT: "
"""

        meta_response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1000,
            messages=[{"role": "user", "content": meta_prompt}]
        )

        # Parse and evaluate new candidates
        new_prompts = [
            line.replace("PROMPT: ", "").strip()
            for line in meta_response.content[0].text.split("\n")
            if line.strip().startswith("PROMPT: ")
        ]

        for prompt in new_prompts:
            if not prompt or "{input}" not in prompt:
                continue
            score = evaluate_prompt(prompt, test_cases)
            history.append(PromptScore(prompt=prompt, score=score))
            if score > best.score:
                best = PromptScore(prompt=prompt, score=score)
                print(f"Iteration {iteration}: New best! Score: {score:.2f}")

    return best

# Usage
test_cases = [
    {"input": "Absolutely wonderful product!", "expected": "positive"},
    {"input": "Terrible waste of money", "expected": "negative"},
    {"input": "Love it, works great", "expected": "positive"},
    {"input": "Broke after one week, very disappointed", "expected": "negative"},
    {"input": "Decent value for the price", "expected": "positive"},
    {"input": "Would not recommend to anyone", "expected": "negative"},
    # ... more cases for reliable evaluation
]

initial_prompts = [
    "Is this review positive or negative? {input}",
    "Classify the sentiment: {input}\nAnswer: positive or negative",
]

best = opro_optimize(initial_prompts, test_cases, num_iterations=5)
print(f"\nBest prompt (score {best.score:.2f}): {best.prompt}")
```

### 3.3 OPRO 인사이트

OPRO 논문의 주요 발견:

1. **LLM은 프롬프트를 최적화할 수 있다**: 프롬프트-점수 쌍이 주어지면 LLM이 개선된 프롬프트를 생성
2. **지시사항 위치가 중요하다**: OPRO는 예시 앞이 아닌 뒤에 지시사항을 배치하면 종종 성능이 향상됨을 발견
3. **최적화 궤적**: 초기 반복에서 성능이 빠르게 향상되다가 안정화
4. **온도가 중요하다**: 메타 프롬프트에서 높은 온도는 더 다양한 후보를 생성

---

## 4. APE: 자동 프롬프트 엔지니어(Automatic Prompt Engineer)

APE(Zhou et al., 2022)는 프롬프트를 자동으로 생성하고 선택합니다. OPRO의 반복적 개선과 달리, APE는 처음부터 많은 후보를 생성하고 최적의 것을 선택합니다.

### 4.1 APE 개념

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  Generate     │───▶│  Evaluate     │───▶│   Select     │
│  Candidates   │    │  All          │    │   Best       │
│  (from I/O    │    │  Candidates   │    │              │
│   examples)   │    │              │    │              │
└──────────────┘    └──────────────┘    └──────────────┘
       │                                        │
       ▼                                        ▼
  "Given these                             Best prompt
   input-output                            for production
   pairs, what
   instruction
   could produce
   this output?"
```

### 4.2 APE 구현

```python
import anthropic
from dataclasses import dataclass

client = anthropic.Anthropic()

def ape_generate_instructions(
    input_output_pairs: list[dict],
    num_candidates: int = 20
) -> list[str]:
    """Generate instruction candidates from input-output examples (APE step 1)."""
    pairs_text = "\n".join(
        f"Input: {pair['input']}\nOutput: {pair['output']}"
        for pair in input_output_pairs[:10]  # Use a subset for generation
    )

    prompt = f"""Given the following input-output pairs, generate {num_candidates}
different instructions that would produce the correct output for each input.

Input-Output pairs:
{pairs_text}

Generate diverse instructions. Some should be short and direct, others detailed.
Some should include format specifications, others should be open-ended.

Return each instruction on its own line, prefixed with "INSTRUCTION: "
"""

    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2000,
        messages=[{"role": "user", "content": prompt}]
    )

    instructions = [
        line.replace("INSTRUCTION: ", "").strip()
        for line in message.content[0].text.split("\n")
        if line.strip().startswith("INSTRUCTION: ")
    ]
    return instructions

def ape_evaluate_and_select(
    instructions: list[str],
    eval_set: list[dict],
    top_k: int = 3
) -> list[dict]:
    """Evaluate all instruction candidates and select the best (APE step 2)."""
    results = []
    for instruction in instructions:
        correct = 0
        for case in eval_set:
            full_prompt = f"{instruction}\n\nInput: {case['input']}"
            message = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=200,
                messages=[{"role": "user", "content": full_prompt}]
            )
            response = message.content[0].text.strip()
            if case["output"].lower() in response.lower():
                correct += 1
        score = correct / len(eval_set)
        results.append({"instruction": instruction, "score": score})
        print(f"Score {score:.2f}: {instruction[:60]}...")

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]

# Usage
examples = [
    {"input": "The quick brown fox", "output": "5"},
    {"input": "Hello world", "output": "2"},
    {"input": "One two three four five six", "output": "6"},
    {"input": "Python is great", "output": "3"},
]

# Step 1: Generate instruction candidates
candidates = ape_generate_instructions(examples, num_candidates=15)
print(f"Generated {len(candidates)} candidates")

# Step 2: Evaluate and select
best = ape_evaluate_and_select(candidates, examples, top_k=3)
for result in best:
    print(f"\nScore: {result['score']:.2f}")
    print(f"Instruction: {result['instruction']}")
```

### 4.3 반복적 개선이 포함된 APE

APE의 생성과 반복적 개선을 결합합니다:

```python
def ape_with_refinement(
    examples: list[dict],
    eval_set: list[dict],
    num_iterations: int = 3
) -> dict:
    """APE with iterative refinement of top candidates."""
    # Initial generation
    candidates = ape_generate_instructions(examples, num_candidates=20)
    best_results = ape_evaluate_and_select(candidates, eval_set, top_k=5)

    for iteration in range(num_iterations):
        # Refine top candidates
        refinement_prompt = f"""Here are the best-performing instructions so far:

{chr(10).join(f'Score {r["score"]:.2f}: {r["instruction"]}' for r in best_results)}

Generate 10 new instructions that combine the strengths of the
high-scoring instructions. Try to improve on them while keeping
what makes them effective.

Return each instruction on its own line, prefixed with "INSTRUCTION: "
"""

        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1000,
            messages=[{"role": "user", "content": refinement_prompt}]
        )

        new_candidates = [
            line.replace("INSTRUCTION: ", "").strip()
            for line in message.content[0].text.split("\n")
            if line.strip().startswith("INSTRUCTION: ")
        ]

        new_results = ape_evaluate_and_select(new_candidates, eval_set, top_k=5)
        all_results = best_results + new_results
        all_results.sort(key=lambda x: x["score"], reverse=True)
        best_results = all_results[:5]

        print(f"\nIteration {iteration + 1} best score: {best_results[0]['score']:.2f}")

    return best_results[0]
```

---

## 5. 자동 프롬프트 생성

최적화를 넘어, LLM은 작업 설명이나 예시로부터 처음부터 프롬프트를 생성할 수 있습니다.

### 5.1 작업 설명에서 프롬프트로

```python
import anthropic

client = anthropic.Anthropic()

def generate_prompt_from_task(task_description: str, model_name: str) -> str:
    """Generate a complete prompt from a task description."""
    meta_prompt = f"""You are a prompt engineering expert. Create an effective
prompt for the following task.

TASK: {task_description}

TARGET MODEL: {model_name}

Generate a complete, ready-to-use prompt that includes:
1. Clear role/persona (if beneficial)
2. Detailed task instructions
3. Input/output format specification
4. Edge case handling instructions
5. Example(s) if few-shot would help

The prompt should contain {{input}} as a placeholder for the actual input.

Output ONLY the prompt text, nothing else."""

    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1500,
        messages=[{"role": "user", "content": meta_prompt}]
    )
    return message.content[0].text

# Usage
prompt = generate_prompt_from_task(
    task_description="Extract all email addresses from unstructured text and "
                     "return them as a JSON array. Handle edge cases like "
                     "obfuscated emails (user [at] domain [dot] com).",
    model_name="Claude Sonnet"
)
print(prompt)
```

### 5.2 예시에서 프롬프트 생성

```python
def generate_prompt_from_examples(
    examples: list[dict],
    task_hint: str = ""
) -> str:
    """Infer the task from examples and generate a prompt."""
    examples_text = "\n\n".join(
        f"Input: {e['input']}\nExpected Output: {e['output']}"
        for e in examples
    )

    meta_prompt = f"""Analyze these input-output examples and generate a prompt
that would produce the correct output for any similar input.

EXAMPLES:
{examples_text}

{f"HINT about the task: {task_hint}" if task_hint else ""}

Steps:
1. Identify the pattern in the input-output mapping
2. Describe the task in clear, unambiguous language
3. Generate a prompt that captures this task completely
4. Include edge case handling based on patterns in the examples

The prompt should contain {{input}} as a placeholder.
Output ONLY the prompt text."""

    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1000,
        messages=[{"role": "user", "content": meta_prompt}]
    )
    return message.content[0].text
```

---

## 6. 프롬프트를 위한 기울기 없는 최적화(Gradient-Free Optimization for Prompts)

프롬프트는 이산적 텍스트(연속 벡터가 아님)이므로 기울기 기반 최적화를 사용할 수 없습니다. 대신 기울기 없는 최적화 방법을 적용합니다.

### 6.1 랜덤 검색(Random Search)

가장 간단한 접근: 랜덤 프롬프트 변형을 생성하고 최적의 것을 유지합니다.

```python
import anthropic
import random

client = anthropic.Anthropic()

def random_search_optimization(
    base_prompt: str,
    eval_fn: callable,
    num_trials: int = 50
) -> dict:
    """Random search over prompt variants."""
    # Generate variants by asking the LLM to paraphrase
    variants = [base_prompt]
    for _ in range(num_trials):
        msg = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=500,
            messages=[{
                "role": "user",
                "content": f"""Paraphrase this prompt in a different way while
preserving the exact same meaning and intent. Change the wording, structure,
or organization but keep all instructions intact.

Original prompt: {base_prompt}

Paraphrased prompt:"""
            }]
        )
        variants.append(msg.content[0].text.strip())

    # Evaluate all variants
    results = []
    for variant in variants:
        score = eval_fn(variant)
        results.append({"prompt": variant, "score": score})

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[0]
```

### 6.2 진화적 검색(Evolutionary Search)

프롬프트를 진화 알고리즘의 개체로 취급합니다:

```python
import anthropic
import random

client = anthropic.Anthropic()

def evolutionary_prompt_search(
    initial_prompts: list[str],
    eval_fn: callable,
    num_generations: int = 10,
    population_size: int = 20,
    mutation_rate: float = 0.3
) -> dict:
    """Evolutionary optimization of prompts."""
    # Initialize population
    population = [
        {"prompt": p, "score": eval_fn(p)}
        for p in initial_prompts
    ]

    for gen in range(num_generations):
        # Selection: Keep top 50%
        population.sort(key=lambda x: x["score"], reverse=True)
        survivors = population[:population_size // 2]

        # Crossover: Combine elements from two parents
        children = []
        while len(children) < population_size // 4:
            parent1, parent2 = random.sample(survivors, 2)
            crossover_prompt = f"""Combine the best elements of these two prompts
into a single new prompt:

Prompt A (score {parent1['score']:.2f}): {parent1['prompt']}

Prompt B (score {parent2['score']:.2f}): {parent2['prompt']}

Create a new prompt that takes the most effective elements from both.
Output ONLY the combined prompt."""

            msg = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=500,
                messages=[{"role": "user", "content": crossover_prompt}]
            )
            child_prompt = msg.content[0].text.strip()
            score = eval_fn(child_prompt)
            children.append({"prompt": child_prompt, "score": score})

        # Mutation: Randomly modify some prompts
        mutants = []
        for survivor in survivors:
            if random.random() < mutation_rate:
                mutation_prompt = f"""Make a small but meaningful change to this prompt.
Change one instruction, add a clarification, or reorganize a section.

Original: {survivor['prompt']}

Modified prompt:"""

                msg = client.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=500,
                    messages=[{"role": "user", "content": mutation_prompt}]
                )
                mutant_prompt = msg.content[0].text.strip()
                score = eval_fn(mutant_prompt)
                mutants.append({"prompt": mutant_prompt, "score": score})

        # New generation
        population = survivors + children + mutants
        population.sort(key=lambda x: x["score"], reverse=True)
        population = population[:population_size]

        print(f"Generation {gen}: Best score = {population[0]['score']:.2f}")

    return population[0]
```

### 6.3 최적화 방법 비교

| 방법 | 장점 | 단점 | 최적 대상 |
|------|------|------|----------|
| 랜덤 검색(Random Search) | 간단, 병렬화 가능 | 결과에서 학습 없음 | 빠른 베이스라인 |
| OPRO | LLM 지능 활용 | 비용이 많이 듦 (많은 LLM 호출) | 지시사항 최적화 |
| APE | 초기 생성에 우수 | 개선에 덜 효과적 | 콜드 스타트 시나리오 |
| 진화적(Evolutionary) | 체계적 탐색 | 느린 수렴 | 복잡한 프롬프트 공간 |
| DSPy 옵티마이저 | 통합 프레임워크 | 학습 곡선 | 프로덕션 시스템 |
| 베이지안(Bayesian) | 샘플 효율적 | 복잡한 구현 | 비용이 많이 드는 평가 |

---

## 7. 베이지안 프롬프트 최적화(Bayesian Prompt Optimization)

베이지안 최적화(Bayesian Optimization)는 각 평가가 비용이 많이 들고(여러 LLM 호출이 필요) 평가 횟수를 최소화하려는 프롬프트 최적화에 특히 적합합니다.

### 7.1 개념

```
┌──────────────┐    ┌──────────────────┐    ┌──────────────┐
│ Surrogate     │───▶│  Acquisition      │───▶│ Evaluate     │
│ Model         │    │  Function         │    │ Candidate    │
│ (predict      │    │  (select next     │    │ (actual LLM  │
│  performance) │    │   candidate)      │    │  evaluation) │
└──────────────┘    └──────────────────┘    └──────────────┘
       ▲                                          │
       └──────────────────────────────────────────┘
                     Update model with result
```

### 7.2 간소화된 베이지안 프롬프트 최적화

```python
import anthropic
import random
import math

client = anthropic.Anthropic()

class BayesianPromptOptimizer:
    """Simplified Bayesian prompt optimization using Thompson sampling."""

    def __init__(self, eval_fn: callable):
        self.eval_fn = eval_fn
        self.history: list[dict] = []  # {"prompt": ..., "score": ...}

    def generate_candidates(self, n: int = 5) -> list[str]:
        """Generate new prompt candidates informed by history."""
        if not self.history:
            # Cold start: generate diverse candidates
            msg = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1500,
                messages=[{
                    "role": "user",
                    "content": f"Generate {n} diverse prompts for sentiment "
                              "classification. Each should contain {{input}} "
                              "as placeholder. Make them very different in "
                              "style and approach."
                }]
            )
            return self._parse_prompts(msg.content[0].text)

        # Informed generation: focus on high-scoring regions
        sorted_history = sorted(self.history, key=lambda x: x["score"], reverse=True)
        top_prompts = sorted_history[:3]
        bottom_prompts = sorted_history[-3:]

        msg = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1500,
            messages=[{
                "role": "user",
                "content": f"""Generate {n} new prompt candidates for sentiment classification.

HIGH-PERFORMING prompts (learn from these):
{chr(10).join(f'Score {p["score"]:.2f}: {p["prompt"][:200]}' for p in top_prompts)}

LOW-PERFORMING prompts (avoid these patterns):
{chr(10).join(f'Score {p["score"]:.2f}: {p["prompt"][:200]}' for p in bottom_prompts)}

Generate prompts that are similar to the high-performing ones but explore
new variations. Each must contain {{input}} as placeholder.
"""
            }]
        )
        return self._parse_prompts(msg.content[0].text)

    def optimize(self, num_iterations: int = 10, candidates_per_round: int = 5) -> dict:
        """Run Bayesian-inspired prompt optimization."""
        for iteration in range(num_iterations):
            candidates = self.generate_candidates(candidates_per_round)

            for candidate in candidates:
                score = self.eval_fn(candidate)
                self.history.append({"prompt": candidate, "score": score})

            best = max(self.history, key=lambda x: x["score"])
            print(f"Iteration {iteration}: Best score = {best['score']:.2f} "
                  f"(total evaluations: {len(self.history)})")

        return max(self.history, key=lambda x: x["score"])

    def _parse_prompts(self, text: str) -> list[str]:
        """Parse numbered prompts from LLM output."""
        lines = text.strip().split("\n")
        prompts = []
        current = []
        for line in lines:
            if line.strip() and (line.strip()[0].isdigit() and "." in line[:5]):
                if current:
                    prompts.append(" ".join(current))
                current = [line.split(".", 1)[-1].strip()]
            elif current:
                current.append(line.strip())
        if current:
            prompts.append(" ".join(current))
        return [p for p in prompts if "{input}" in p]
```

---

## 8. 프롬프트 압축(Prompt Compression)

프롬프트 압축(Prompt Compression)은 작업 성능을 유지하면서 토큰 수를 줄입니다. 이는 비용을 절감하고 지연 시간을 개선할 수 있습니다.

### 8.1 프롬프트를 압축하는 이유

| 토큰 수 | 비용 영향 ($3/1M 토큰 기준) | 지연 시간 영향 |
|---------|---------------------------|---------------|
| 1,000 토큰 | $0.003/호출 | ~1초 |
| 5,000 토큰 | $0.015/호출 | ~3초 |
| 10,000 토큰 | $0.030/호출 | ~5초 |
| 50,000 토큰 | $0.150/호출 | ~15초 |

하루 10,000건의 호출에서 5,000 토큰에서 2,000 토큰으로 줄이면 하루 $90($32,850/년)을 절약합니다.

### 8.2 수동 압축 기법

```python
# BEFORE: Verbose prompt (287 tokens)
verbose_prompt = """
You are an expert data analyst with over 20 years of experience in the field
of business intelligence and data analytics. Your specialty is in analyzing
customer feedback data from various sources including surveys, reviews, and
support tickets. You have deep expertise in sentiment analysis, theme
extraction, and trend identification.

Given the following customer review, I would like you to please analyze it
carefully and thoughtfully. Consider all aspects of the review including the
tone, specific complaints or praises, and any suggestions the customer might
have. After your thorough analysis, please provide your assessment of the
overall sentiment of the review.

Your response should be formatted as follows:
- Sentiment: positive, negative, or neutral
- Confidence: high, medium, or low
- Key themes: a list of main topics mentioned

Please be as accurate as possible in your analysis. Here is the review:

{review}
"""

# AFTER: Compressed prompt (89 tokens) -- same performance
compressed_prompt = """Analyze this customer review.

Output format:
- Sentiment: positive/negative/neutral
- Confidence: high/medium/low
- Key themes: [list]

Review: {review}"""
```

### 8.3 LLM 기반 압축 (LLMLingua 접근법)

LLMLingua와 유사한 도구는 작은 모델을 사용하여 작업 성능에 가장 적게 기여하는 토큰을 식별하고 제거합니다:

```python
import anthropic

client = anthropic.Anthropic()

def compress_prompt(
    original_prompt: str,
    target_ratio: float = 0.5,
    task_description: str = ""
) -> str:
    """Use an LLM to compress a prompt while preserving its effectiveness."""
    compression_prompt = f"""Compress the following prompt to approximately
{int(target_ratio * 100)}% of its current length while preserving ALL
task-critical information.

COMPRESSION RULES:
1. Remove filler words and redundant phrases
2. Keep ALL technical instructions and constraints
3. Keep ALL format specifications
4. Remove personality/role descriptions unless they affect output quality
5. Combine redundant sentences
6. Use abbreviations only if unambiguous
7. Keep examples if they serve as few-shot demonstrations
8. Remove motivational language ("please", "carefully", "thoroughly")

{f"Task context: {task_description}" if task_description else ""}

ORIGINAL PROMPT:
---
{original_prompt}
---

COMPRESSED PROMPT (preserve {{placeholders}}):"""

    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1000,
        messages=[{"role": "user", "content": compression_prompt}]
    )
    return message.content[0].text.strip()

# Usage
original = """You are a helpful assistant that specializes in extracting
structured information from unstructured text. Given a block of text that
describes a person, please carefully extract the following information and
return it as a JSON object:

- name: The person's full name (first and last)
- age: Their age as an integer (if mentioned)
- occupation: Their job title or profession (if mentioned)
- location: Where they live or work (if mentioned)

If any field is not mentioned in the text, set its value to null.
Please ensure the JSON is properly formatted.

Text: {text}"""

compressed = compress_prompt(original, target_ratio=0.5)
print(compressed)
# Expected output (roughly):
# "Extract from text as JSON: {name, age, occupation, location}. Null if not mentioned.
#  Text: {text}"
```

### 8.4 압축 품질 측정

```python
import anthropic

client = anthropic.Anthropic()

def evaluate_compression(
    original_prompt: str,
    compressed_prompt: str,
    test_cases: list[dict]
) -> dict:
    """Compare performance of original vs compressed prompt."""
    original_scores = []
    compressed_scores = []

    for case in test_cases:
        # Evaluate original
        orig_msg = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=500,
            messages=[{
                "role": "user",
                "content": original_prompt.format(**case["inputs"])
            }]
        )
        orig_correct = case["expected"] in orig_msg.content[0].text
        original_scores.append(1 if orig_correct else 0)

        # Evaluate compressed
        comp_msg = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=500,
            messages=[{
                "role": "user",
                "content": compressed_prompt.format(**case["inputs"])
            }]
        )
        comp_correct = case["expected"] in comp_msg.content[0].text
        compressed_scores.append(1 if comp_correct else 0)

    orig_accuracy = sum(original_scores) / len(original_scores)
    comp_accuracy = sum(compressed_scores) / len(compressed_scores)

    # Estimate token savings
    orig_tokens = len(original_prompt.split()) * 1.3  # rough estimate
    comp_tokens = len(compressed_prompt.split()) * 1.3

    return {
        "original_accuracy": orig_accuracy,
        "compressed_accuracy": comp_accuracy,
        "accuracy_drop": orig_accuracy - comp_accuracy,
        "compression_ratio": comp_tokens / orig_tokens,
        "token_savings": 1 - (comp_tokens / orig_tokens),
        "acceptable": (orig_accuracy - comp_accuracy) < 0.02  # <2% drop
    }
```

---

## 9. 비용-품질 트레이드오프(Cost-Quality Trade-offs)

모든 프롬프트 최적화 결정에는 비용, 품질, 지연 시간 사이의 트레이드오프가 수반됩니다.

### 9.1 비용-품질 프론티어(Cost-Quality Frontier)

```
Quality
  ▲
  │     ●  Long detailed prompt + Claude Opus
  │    ● ●  Optimized prompt + Claude Sonnet
  │   ●     DSPy-optimized + Claude Sonnet
  │  ●      Compressed prompt + Claude Sonnet
  │ ●       Short prompt + Claude Haiku
  │●        Minimal prompt + Claude Haiku
  └──────────────────────────────────▶ Cost per call
```

### 9.2 비용 최적화 전략

```python
import anthropic

client = anthropic.Anthropic()

class CostAwarePromptSelector:
    """Select the cheapest prompt-model combination that meets quality threshold."""

    def __init__(self, quality_threshold: float = 0.95):
        self.quality_threshold = quality_threshold
        self.configurations = []

    def add_configuration(
        self,
        name: str,
        model: str,
        prompt: str,
        cost_per_1k_tokens: float
    ):
        self.configurations.append({
            "name": name,
            "model": model,
            "prompt": prompt,
            "cost_per_1k_tokens": cost_per_1k_tokens
        })

    def evaluate_all(self, test_cases: list[dict]) -> list[dict]:
        """Evaluate all configurations and rank by cost-effectiveness."""
        results = []
        for config in self.configurations:
            correct = 0
            total_tokens = 0
            for case in test_cases:
                msg = client.messages.create(
                    model=config["model"],
                    max_tokens=200,
                    messages=[{
                        "role": "user",
                        "content": config["prompt"].format(**case["inputs"])
                    }]
                )
                if case["expected"] in msg.content[0].text:
                    correct += 1
                total_tokens += msg.usage.input_tokens + msg.usage.output_tokens

            accuracy = correct / len(test_cases)
            avg_tokens = total_tokens / len(test_cases)
            cost_per_call = avg_tokens * config["cost_per_1k_tokens"] / 1000

            results.append({
                "name": config["name"],
                "accuracy": accuracy,
                "avg_tokens": avg_tokens,
                "cost_per_call": cost_per_call,
                "meets_threshold": accuracy >= self.quality_threshold
            })

        # Sort by cost (ascending) among configurations that meet threshold
        qualifying = [r for r in results if r["meets_threshold"]]
        qualifying.sort(key=lambda x: x["cost_per_call"])

        return qualifying

# Usage
selector = CostAwarePromptSelector(quality_threshold=0.90)

selector.add_configuration(
    name="Full prompt + Opus",
    model="claude-opus-4-20250514",
    prompt="Detailed prompt here... {input}",
    cost_per_1k_tokens=0.015
)
selector.add_configuration(
    name="Optimized prompt + Sonnet",
    model="claude-sonnet-4-20250514",
    prompt="Concise prompt... {input}",
    cost_per_1k_tokens=0.003
)
selector.add_configuration(
    name="Minimal prompt + Haiku",
    model="claude-haiku-4-20250514",
    prompt="Classify: {input}",
    cost_per_1k_tokens=0.00025
)
```

### 9.3 캐스케이딩 전략(Cascading Strategy)

저렴한 모델을 먼저 사용하고, 필요한 경우에만 비싼 모델로 에스컬레이션합니다:

```python
import anthropic

client = anthropic.Anthropic()

def cascading_prompt(query: str, prompt_template: str) -> dict:
    """Try cheap model first, escalate if confidence is low."""
    models = [
        {"name": "claude-haiku-4-20250514", "cost": "low"},
        {"name": "claude-sonnet-4-20250514", "cost": "medium"},
        {"name": "claude-opus-4-20250514", "cost": "high"},
    ]

    for model_config in models:
        message = client.messages.create(
            model=model_config["name"],
            max_tokens=500,
            messages=[{
                "role": "user",
                "content": prompt_template.format(query=query) +
                          "\n\nEnd your response with CONFIDENCE: HIGH/MEDIUM/LOW"
            }]
        )

        response = message.content[0].text
        confidence = "LOW"
        if "CONFIDENCE: HIGH" in response:
            confidence = "HIGH"
        elif "CONFIDENCE: MEDIUM" in response:
            confidence = "MEDIUM"

        if confidence == "HIGH":
            return {
                "answer": response,
                "model_used": model_config["name"],
                "cost_tier": model_config["cost"]
            }

    # Final model's answer regardless of confidence
    return {
        "answer": response,
        "model_used": models[-1]["name"],
        "cost_tier": "high"
    }
```

---

## 10. 최적화 vs 모델 전환 시점

### 10.1 의사결정 프레임워크(Decision Framework)

```
                    Is the task well-defined with clear metrics?
                              │
                    ┌─────────┴──────────┐
                    │ NO                  │ YES
                    ▼                     ▼
           Improve task            Is the current model
           definition first        getting > 70% accuracy?
                                        │
                              ┌─────────┴──────────┐
                              │ NO                  │ YES
                              ▼                     ▼
                     Consider switching      Optimize the prompt
                     to a more capable       (DSPy, OPRO, manual)
                     model first
                              │                     │
                              ▼                     ▼
                     Still < 70% after       Reached > 95%?
                     model upgrade?                │
                              │              ┌─────┴──────┐
                              ▼              │ YES        │ NO
                     Re-examine the task:    Done!   Try model upgrade
                     - Is it too hard for            + prompt optimization
                       current LLMs?                 together
                     - Do you need fine-tuning?
                     - Is the evaluation correct?
```

### 10.2 최적화 vs 모델 전환 비교

| 접근법 | 노력 | 비용 | 일반적인 개선 |
|--------|------|------|-------------|
| 수동 프롬프트 편집 | 낮음 | 무료 | 5-15% |
| 퓨샷 예시 선택(Few-shot Example Selection) | 낮음 | 약간의 토큰 증가 | 5-20% |
| DSPy BootstrapFewShot | 중간 | 최적화 LLM 호출 | 10-25% |
| DSPy MIPROv2 | 중간-높음 | 많은 최적화 호출 | 15-30% |
| OPRO | 중간 | 많은 최적화 호출 | 10-20% |
| 모델 업그레이드 (Haiku → Sonnet) | 낮음 | 호출당 비용 증가 | 10-30% |
| 모델 업그레이드 (Sonnet → Opus) | 낮음 | 호출당 비용 대폭 증가 | 5-20% |
| 파인튜닝(Fine-tuning) | 높음 | 학습 컴퓨트 + 데이터 | 20-40% |

### 10.3 실용적 의사결정 규칙

```python
# Pseudocode for the optimization decision
def decide_optimization_strategy(
    current_accuracy: float,
    target_accuracy: float,
    current_model: str,
    budget_per_call: float,
    call_volume_per_day: int,
    has_training_data: bool
) -> str:
    gap = target_accuracy - current_accuracy

    if gap <= 0:
        return "Already meeting target. Consider cost optimization."

    if gap <= 0.05:  # Small gap (< 5%)
        return "Try manual prompt optimization or few-shot tuning."

    if gap <= 0.15:  # Medium gap (5-15%)
        if has_training_data:
            return "Use DSPy or OPRO with your training data."
        else:
            return "Build an evaluation dataset first, then use DSPy."

    if gap <= 0.30:  # Large gap (15-30%)
        if current_model != "claude-opus-4-20250514":
            return "Upgrade model AND optimize prompts."
        else:
            return "Consider fine-tuning or re-examining the task definition."

    # Very large gap (> 30%)
    return ("Task may be too hard for prompting alone. Consider: "
            "1) Fine-tuning, 2) Breaking into subtasks, "
            "3) Adding retrieval (RAG), 4) Human-in-the-loop")
```

### 10.4 최적화 워크플로우(The Optimization Workflow)

1. **베이스라인(Baseline)**: 기본 프롬프트로 현재 성능 측정
2. **빠른 성과(Quick wins)**: 수동 개선 시도 (형식, 예시, 제약 조건)
3. **체계적 검색(Systematic search)**: 빠른 성과가 정체되면 DSPy 또는 OPRO 적용
4. **모델 탐색(Model exploration)**: 더 강력하거나 약한 모델로 테스트하여 한계 파악
5. **비용 최적화(Cost optimization)**: 품질 목표가 달성되면 프롬프트를 압축하고 더 저렴한 모델 시도
6. **모니터링(Monitor)**: 회귀를 감지하기 위한 지속적 평가 설정

```python
import anthropic
import json
from datetime import datetime

client = anthropic.Anthropic()

def optimization_experiment_log(
    experiment_name: str,
    prompt: str,
    model: str,
    eval_results: dict,
    notes: str = ""
) -> dict:
    """Log an optimization experiment for tracking."""
    entry = {
        "timestamp": datetime.now().isoformat(),
        "experiment": experiment_name,
        "model": model,
        "prompt_length_tokens": len(prompt.split()) * 1.3,
        "accuracy": eval_results.get("accuracy"),
        "f1": eval_results.get("f1"),
        "cost_per_call": eval_results.get("cost_per_call"),
        "latency_ms": eval_results.get("latency_ms"),
        "notes": notes,
        "prompt_hash": hash(prompt),
    }

    # Append to log file
    with open("optimization_log.jsonl", "a") as f:
        f.write(json.dumps(entry) + "\n")

    return entry
```

---

## 연습문제

### 연습문제 1: DSPy 시그니처 설계

고객 지원 티켓 분류기를 위한 DSPy 시그니처와 모듈을 설계하세요. 시스템은 다음을 수행해야 합니다:
1. 티켓을 카테고리로 분류 (billing, technical, account, general)
2. 우선순위 지정 (low, medium, high, urgent)
3. 티켓에 사람의 에스컬레이션이 필요한지 판단

DSPy 프로그램(시그니처 + 모듈)을 작성하고 어떻게 최적화할 것인지 설명하세요.

<details><summary>정답 보기</summary>

```python
import dspy

# Configure the language model
lm = dspy.LM("anthropic/claude-sonnet-4-20250514")
dspy.configure(lm=lm)

# Signature for ticket classification
class ClassifyTicket(dspy.Signature):
    """Classify a customer support ticket by category and priority."""
    ticket_text: str = dspy.InputField(desc="The customer's support ticket text")
    customer_tier: str = dspy.InputField(
        desc="Customer tier: free, pro, enterprise"
    )
    category: str = dspy.OutputField(
        desc="One of: billing, technical, account, general"
    )
    priority: str = dspy.OutputField(
        desc="One of: low, medium, high, urgent"
    )

# Signature for escalation decision
class DecideEscalation(dspy.Signature):
    """Determine if a classified ticket needs human escalation."""
    ticket_text: str = dspy.InputField(desc="The customer's support ticket text")
    category: str = dspy.InputField(desc="Ticket category")
    priority: str = dspy.InputField(desc="Ticket priority")
    customer_tier: str = dspy.InputField(desc="Customer tier")
    needs_escalation: bool = dspy.OutputField(
        desc="True if ticket needs human agent, False for auto-response"
    )
    escalation_reason: str = dspy.OutputField(
        desc="Why escalation is or is not needed"
    )

# Multi-step module
class TicketTriageSystem(dspy.Module):
    def __init__(self):
        self.classifier = dspy.ChainOfThought(ClassifyTicket)
        self.escalation = dspy.Predict(DecideEscalation)

    def forward(self, ticket_text: str, customer_tier: str = "free"):
        # Step 1: Classify
        classification = self.classifier(
            ticket_text=ticket_text,
            customer_tier=customer_tier
        )

        # Step 2: Decide escalation
        escalation = self.escalation(
            ticket_text=ticket_text,
            category=classification.category,
            priority=classification.priority,
            customer_tier=customer_tier
        )

        return dspy.Prediction(
            category=classification.category,
            priority=classification.priority,
            needs_escalation=escalation.needs_escalation,
            escalation_reason=escalation.escalation_reason
        )

# Optimization setup
def triage_metric(example, prediction, trace=None):
    """Multi-criteria metric for ticket triage."""
    category_correct = example.category == prediction.category
    priority_correct = example.priority == prediction.priority
    escalation_correct = example.needs_escalation == prediction.needs_escalation

    # Weight: category most important, then priority, then escalation
    score = (
        0.4 * category_correct +
        0.3 * priority_correct +
        0.3 * escalation_correct
    )
    return score

# Training data
trainset = [
    dspy.Example(
        ticket_text="I was charged twice for my subscription this month",
        customer_tier="pro",
        category="billing",
        priority="high",
        needs_escalation=True
    ).with_inputs("ticket_text", "customer_tier"),
    # ... 50+ more examples
]

# Optimize
optimizer = dspy.BootstrapFewShotWithRandomSearch(
    metric=triage_metric,
    max_bootstrapped_demos=3,
    num_candidate_programs=10
)

triage_system = TicketTriageSystem()
optimized_system = optimizer.compile(triage_system, trainset=trainset)

# Save optimized system
optimized_system.save("optimized_triage.json")
```

핵심 설계 결정:
1. **두 개의 시그니처**: 분류와 에스컬레이션의 분리로 독립적 최적화 가능.
2. **분류에 ChainOfThought**: 모호한 티켓에서 추론이 도움됨.
3. **에스컬레이션에 Predict**: 더 간단한 결정; CoT 오버헤드가 정당화되지 않음.
4. **다중 기준 지표**: 가중 점수가 비즈니스 우선순위를 반영 (카테고리 정확도 > 우선순위 정확도 > 에스컬레이션 정확도).
5. **BootstrapFewShotWithRandomSearch**: 최적화 능력과 컴퓨트 비용의 좋은 균형.

</details>

### 연습문제 2: OPRO 구현

요약 프롬프트를 최적화하기 위한 간소화된 OPRO 루프를 구현하세요. 프롬프트는 뉴스 기사를 2-3문장으로 요약해야 합니다. 평가 지표를 정의하고 최소 3회 반복의 최적화를 실행하세요.

<details><summary>정답 보기</summary>

```python
import anthropic
from dataclasses import dataclass

client = anthropic.Anthropic()

# Evaluation dataset
eval_articles = [
    {
        "article": "Apple today announced its Q4 2024 earnings, reporting revenue "
                   "of $94.9 billion, up 6% year over year. iPhone revenue came in at "
                   "$46.2 billion, while Services revenue hit a new all-time high of "
                   "$25.0 billion. CEO Tim Cook cited strong demand for iPhone 16 Pro "
                   "models and growing subscription services as key drivers.",
        "key_facts": ["$94.9 billion revenue", "6% growth", "iPhone 16 Pro",
                      "Services $25 billion", "all-time high"]
    },
    {
        "article": "Researchers at MIT have developed a new type of solar cell that "
                   "achieves 29.1% efficiency, breaking the previous record of 27.6%. "
                   "The perovskite-silicon tandem cell uses a novel interface layer that "
                   "reduces energy loss. The team expects commercial production within "
                   "3-5 years, which could significantly reduce solar energy costs.",
        "key_facts": ["29.1% efficiency", "perovskite-silicon", "MIT",
                      "previous record 27.6%", "3-5 years commercial"]
    },
    {
        "article": "The European Union has reached a preliminary agreement on the AI Act, "
                   "the world's first comprehensive AI regulation. The law bans AI systems "
                   "used for social scoring and real-time biometric surveillance in public "
                   "spaces, with exceptions for law enforcement. Companies have 24 months "
                   "to comply after the law takes effect.",
        "key_facts": ["EU AI Act", "first comprehensive AI regulation",
                      "bans social scoring", "biometric surveillance ban",
                      "24 months compliance"]
    }
]

def evaluate_summary_prompt(prompt_template: str) -> float:
    """Evaluate a summarization prompt on fact coverage and brevity."""
    total_score = 0
    for case in eval_articles:
        full_prompt = prompt_template.replace("{article}", case["article"])
        msg = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=200,
            messages=[{"role": "user", "content": full_prompt}]
        )
        summary = msg.content[0].text.strip()

        # Score 1: Fact coverage (0-1)
        facts_found = sum(
            1 for fact in case["key_facts"] if fact.lower() in summary.lower()
        )
        coverage = facts_found / len(case["key_facts"])

        # Score 2: Brevity (penalize if > 3 sentences)
        sentences = summary.count(".") + summary.count("!") + summary.count("?")
        brevity = 1.0 if sentences <= 3 else max(0, 1.0 - (sentences - 3) * 0.2)

        total_score += 0.7 * coverage + 0.3 * brevity

    return total_score / len(eval_articles)

@dataclass
class PromptResult:
    prompt: str
    score: float

def opro_summarization(num_iterations: int = 3) -> PromptResult:
    """OPRO optimization for summarization prompts."""
    # Initial prompts
    history = []
    initial_prompts = [
        "Summarize this article in 2-3 sentences:\n\n{article}",
        "Write a brief summary of the following news article. Include the most important facts and figures. Keep it to 2-3 sentences.\n\n{article}",
        "Read this article and provide a concise summary that captures the key facts, numbers, and implications. Maximum 3 sentences.\n\nArticle: {article}\n\nSummary:",
    ]

    for p in initial_prompts:
        score = evaluate_summary_prompt(p)
        history.append(PromptResult(prompt=p, score=score))
        print(f"Initial score {score:.3f}: {p[:60]}...")

    for iteration in range(num_iterations):
        # Build meta-prompt with history
        history_text = "\n\n".join(
            f"PROMPT (score={h.score:.3f}):\n{h.prompt}"
            for h in sorted(history, key=lambda x: x.score)[-5:]
        )

        meta_prompt = f"""You are optimizing a summarization prompt. Here are
previous attempts and their scores (higher is better, max 1.0).

Scoring criteria:
- 70% weight: Coverage of key facts and numbers from the article
- 30% weight: Brevity (2-3 sentences ideal, penalized for more)

PREVIOUS ATTEMPTS:
{history_text}

Generate 5 new prompt variants that might score higher. Each must contain
{{article}} as a placeholder.

Learn from patterns:
- What do high-scoring prompts have in common?
- What do low-scoring prompts lack?

Return each prompt on a separate line, prefixed with "PROMPT: "
"""

        msg = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1500,
            messages=[{"role": "user", "content": meta_prompt}]
        )

        new_prompts = [
            line.replace("PROMPT: ", "").strip()
            for line in msg.content[0].text.split("\n")
            if line.strip().startswith("PROMPT: ")
        ]

        for p in new_prompts:
            if "{article}" not in p:
                continue
            score = evaluate_summary_prompt(p)
            history.append(PromptResult(prompt=p, score=score))
            print(f"  Iteration {iteration} score {score:.3f}: {p[:60]}...")

    best = max(history, key=lambda x: x.score)
    print(f"\nBest prompt (score {best.score:.3f}):\n{best.prompt}")
    return best

result = opro_summarization(num_iterations=3)
```

평가 지표는 사실 커버리지(핵심 숫자와 이름이 보존되는가?)와 간결성(요약이 간결한가?) 두 가지 목표의 균형을 맞춥니다. 70/30 가중치는 요약에서 정확도가 길이 제어보다 더 중요하다는 것을 반영합니다.

</details>

### 연습문제 3: 프롬프트 압축

다음의 장황한 프롬프트를 원래 토큰 수의 50% 미만으로 압축하면서 성능을 유지하세요. 압축 전략을 설명하고 압축된 버전이 품질을 유지하는지 어떻게 검증할 것인지 설명하세요.

```
You are an expert financial analyst with deep knowledge of stock market
analysis, corporate earnings reports, and economic indicators. You have
been working in the financial industry for over 15 years and have a
track record of accurate analysis.

I am going to provide you with a quarterly earnings report summary for
a publicly traded company. Your task is to carefully analyze the report
and provide a comprehensive assessment that includes the following elements:

1. Revenue Analysis: Compare the reported revenue against analyst
   expectations and the same quarter last year. Note if it was a beat
   or miss and by what percentage.

2. Profitability Assessment: Analyze gross margin, operating margin,
   and net margin trends. Flag any significant changes.

3. Forward Guidance: Summarize management's guidance for the next
   quarter and full year. Note if guidance was raised, maintained,
   or lowered.

4. Key Risks: Identify the top 3 risks mentioned in the report or
   implied by the financial data.

5. Overall Rating: Provide a rating of BULLISH, NEUTRAL, or BEARISH
   with a brief justification.

Please be thorough but concise in your analysis. Use specific numbers
from the report to support your points. Format your response with clear
headers for each section.

Earnings Report:
{report}
```

<details><summary>정답 보기</summary>

**압축 전략:**
1. 역할/페르소나 설명 제거 (유능한 모델에서는 출력을 개선하지 않음)
2. "메타-지시사항" 제거 ("be thorough", "carefully analyze" 등)
3. 섹션 설명을 핵심 요구사항만으로 축약
4. 구조적 요구사항 유지 (출력 형식에 영향)

**압축된 프롬프트 (원본의 약 45%):**

```
Analyze this earnings report:

1. Revenue: vs expectations and YoY. Beat/miss by what %?
2. Profitability: Gross/operating/net margin trends. Flag significant changes.
3. Guidance: Next quarter + full year. Raised/maintained/lowered?
4. Risks: Top 3 risks from report or data.
5. Rating: BULLISH/NEUTRAL/BEARISH with justification.

Use specific numbers. Format with section headers.

{report}
```

**검증 접근법:**

```python
import anthropic

client = anthropic.Anthropic()

def verify_compression(original: str, compressed: str, test_reports: list[str]):
    """Compare original and compressed prompt outputs."""
    results = []
    for report in test_reports:
        # Generate with original
        orig_msg = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1500,
            messages=[{
                "role": "user",
                "content": original.replace("{report}", report)
            }]
        )
        orig_response = orig_msg.content[0].text

        # Generate with compressed
        comp_msg = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1500,
            messages=[{
                "role": "user",
                "content": compressed.replace("{report}", report)
            }]
        )
        comp_response = comp_msg.content[0].text

        # Compare using LLM judge
        judge_msg = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=500,
            messages=[{
                "role": "user",
                "content": f"""Compare these two analyses of the same earnings report.
Rate each on: (1) Completeness (all 5 sections present), (2) Accuracy (numbers cited),
(3) Actionability (clear recommendation).

Analysis A:
{orig_response}

Analysis B:
{comp_response}

Score each 1-5 on each criterion. Format as:
A: completeness=N, accuracy=N, actionability=N, total=N
B: completeness=N, accuracy=N, actionability=N, total=N"""
            }]
        )

        results.append({
            "report_preview": report[:100],
            "original_tokens": orig_msg.usage.input_tokens,
            "compressed_tokens": comp_msg.usage.input_tokens,
            "judge": judge_msg.content[0].text
        })

    return results
```

제거된 것과 이유:
- **역할 설명** ("expert financial analyst..."): Claude는 페르소나 프라이밍 없이도 재무 분석을 잘 수행합니다; 구체적인 지시사항이 역할보다 더 중요합니다.
- **보충 문구** ("carefully analyze", "comprehensive assessment", "please be thorough but concise"): 이것들은 지시 따르기(Instruction-Following) 모델의 동작을 변경하지 않습니다.
- **중복 설명**: "Compare the reported revenue against analyst expectations and the same quarter last year"가 "vs expectations and YoY"로 -- 동일한 정보를 15단어 대신 5단어로.

</details>

### 연습문제 4: 비용-품질 분석

하루 50,000회 실행되는 텍스트 분류 작업이 있습니다. 현재 500 토큰 프롬프트를 사용하는 Claude Opus로 97% 정확도를 달성합니다. 최소 95% 정확도를 유지하면서 가장 저렴한 모델-프롬프트 조합을 찾는 실험을 설계하세요. 실험 코드를 작성하세요.

<details><summary>정답 보기</summary>

```python
import anthropic
import json
import time
from dataclasses import dataclass
from typing import Optional

client = anthropic.Anthropic()

@dataclass
class ModelConfig:
    name: str
    model_id: str
    input_cost_per_mtok: float   # $ per million input tokens
    output_cost_per_mtok: float  # $ per million output tokens

MODELS = [
    ModelConfig("Opus", "claude-opus-4-20250514", 15.0, 75.0),
    ModelConfig("Sonnet", "claude-sonnet-4-20250514", 3.0, 15.0),
    ModelConfig("Haiku", "claude-haiku-4-20250514", 0.25, 1.25),
]

@dataclass
class PromptVariant:
    name: str
    template: str
    estimated_input_tokens: int

PROMPTS = [
    PromptVariant(
        "Original (500 tok)",
        """You are an expert content moderator. Classify the following
user-generated content into one of these categories: safe, spam,
harassment, misinformation, adult_content.

Consider the following guidelines:
- safe: Regular content that follows community standards
- spam: Promotional, repetitive, or off-topic commercial content
- harassment: Personal attacks, threats, or bullying behavior
- misinformation: Verifiably false claims about health, science, or politics
- adult_content: Explicit sexual content or graphic violence

Analyze the content carefully. Consider context, tone, and intent.
Output ONLY the category label, nothing else.

Content: {text}""",
        500
    ),
    PromptVariant(
        "Medium (200 tok)",
        """Classify this content as: safe, spam, harassment, misinformation,
or adult_content.

Definitions:
- spam: commercial/repetitive
- harassment: attacks/threats
- misinformation: false factual claims
- adult_content: explicit/graphic

Output ONLY the label.

Content: {text}""",
        200
    ),
    PromptVariant(
        "Minimal (50 tok)",
        """Classify as safe/spam/harassment/misinformation/adult_content.
Output one word only.

{text}""",
        50
    ),
]

def run_experiment(
    eval_set: list[dict],
    sample_size: int = 200
) -> list[dict]:
    """Test all model-prompt combinations."""
    results = []
    sample = eval_set[:sample_size]

    for model in MODELS:
        for prompt_var in PROMPTS:
            correct = 0
            total_input_tokens = 0
            total_output_tokens = 0
            latencies = []

            for case in sample:
                full_prompt = prompt_var.template.format(text=case["text"])
                start = time.time()
                msg = client.messages.create(
                    model=model.model_id,
                    max_tokens=20,
                    messages=[{"role": "user", "content": full_prompt}]
                )
                latency = time.time() - start
                latencies.append(latency)

                response = msg.content[0].text.strip().lower()
                if response == case["label"].lower():
                    correct += 1
                total_input_tokens += msg.usage.input_tokens
                total_output_tokens += msg.usage.output_tokens

            accuracy = correct / len(sample)
            avg_input = total_input_tokens / len(sample)
            avg_output = total_output_tokens / len(sample)
            cost_per_call = (
                avg_input * model.input_cost_per_mtok / 1_000_000 +
                avg_output * model.output_cost_per_mtok / 1_000_000
            )
            daily_cost = cost_per_call * 50_000
            avg_latency = sum(latencies) / len(latencies)

            result = {
                "model": model.name,
                "prompt": prompt_var.name,
                "accuracy": accuracy,
                "cost_per_call": cost_per_call,
                "daily_cost": daily_cost,
                "monthly_cost": daily_cost * 30,
                "avg_latency_ms": avg_latency * 1000,
                "meets_threshold": accuracy >= 0.95
            }
            results.append(result)
            print(f"{model.name} + {prompt_var.name}: "
                  f"acc={accuracy:.3f}, "
                  f"${daily_cost:.2f}/day, "
                  f"{avg_latency*1000:.0f}ms")

    # Sort qualifying results by cost
    qualifying = [r for r in results if r["meets_threshold"]]
    qualifying.sort(key=lambda x: x["daily_cost"])

    print("\n=== QUALIFYING CONFIGURATIONS (>= 95% accuracy) ===")
    for r in qualifying:
        print(f"{r['model']} + {r['prompt']}: "
              f"acc={r['accuracy']:.3f}, "
              f"${r['daily_cost']:.2f}/day, "
              f"${r['monthly_cost']:.2f}/month")

    if qualifying:
        winner = qualifying[0]
        baseline = next(r for r in results
                       if r["model"] == "Opus" and "500" in r["prompt"])
        savings = baseline["monthly_cost"] - winner["monthly_cost"]
        print(f"\nRECOMMENDATION: {winner['model']} + {winner['prompt']}")
        print(f"Monthly savings: ${savings:.2f}")

    return results

# Generate synthetic eval set for demonstration
eval_set = [
    {"text": "Buy now! Limited time offer! Click here!", "label": "spam"},
    {"text": "I really enjoyed the new park downtown", "label": "safe"},
    {"text": "You're an idiot and nobody likes you", "label": "harassment"},
    # ... 200+ labeled examples for reliable evaluation
]

results = run_experiment(eval_set, sample_size=len(eval_set))
```

실험은 9가지 조합 (3개 모델 x 3개 프롬프트)을 테스트하고 95% 정확도 임계값을 충족하는 가장 저렴한 것을 선택합니다. 하루 50,000건 호출에서 Opus+500tok과 Haiku+50tok 사이의 비용 차이는 하루 수백 달러에 달할 수 있습니다.

</details>

### 연습문제 5: 최적화 파이프라인

다음을 수행하는 완전한 프롬프트 최적화 파이프라인을 설계하세요: (1) 베이스라인 프롬프트에서 시작, (2) DSPy 최적화 적용, (3) 결과 압축, (4) 압축이 품질을 저하시키지 않았는지 검증, (5) 문서와 함께 프로덕션 준비 프롬프트를 출력. 파이프라인 코드를 작성하세요.

<details><summary>정답 보기</summary>

```python
import anthropic
import dspy
import json
from datetime import datetime
from dataclasses import dataclass, asdict

client = anthropic.Anthropic()

@dataclass
class PipelineResult:
    stage: str
    prompt_or_config: str
    accuracy: float
    token_count: int
    cost_per_call: float
    timestamp: str

class PromptOptimizationPipeline:
    """End-to-end prompt optimization pipeline."""

    def __init__(
        self,
        task_name: str,
        eval_set: list[dict],
        accuracy_threshold: float = 0.95,
        max_accuracy_drop_from_compression: float = 0.02
    ):
        self.task_name = task_name
        self.eval_set = eval_set
        self.accuracy_threshold = accuracy_threshold
        self.max_compression_drop = max_accuracy_drop_from_compression
        self.log: list[PipelineResult] = []

    def evaluate_prompt(self, prompt_template: str) -> dict:
        """Evaluate a prompt on the full eval set."""
        correct = 0
        total_tokens = 0
        for case in self.eval_set:
            msg = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=100,
                messages=[{
                    "role": "user",
                    "content": prompt_template.format(**case["inputs"])
                }]
            )
            response = msg.content[0].text.strip()
            if case["expected"].lower() in response.lower():
                correct += 1
            total_tokens += msg.usage.input_tokens

        accuracy = correct / len(self.eval_set)
        avg_tokens = total_tokens / len(self.eval_set)
        cost = avg_tokens * 3.0 / 1_000_000  # Sonnet input pricing

        return {"accuracy": accuracy, "avg_tokens": avg_tokens, "cost_per_call": cost}

    def stage_1_baseline(self, baseline_prompt: str) -> PipelineResult:
        """Stage 1: Evaluate baseline prompt."""
        print("\n=== Stage 1: Baseline Evaluation ===")
        metrics = self.evaluate_prompt(baseline_prompt)
        result = PipelineResult(
            stage="baseline",
            prompt_or_config=baseline_prompt,
            accuracy=metrics["accuracy"],
            token_count=int(metrics["avg_tokens"]),
            cost_per_call=metrics["cost_per_call"],
            timestamp=datetime.now().isoformat()
        )
        self.log.append(result)
        print(f"Baseline accuracy: {metrics['accuracy']:.3f}, "
              f"tokens: {metrics['avg_tokens']:.0f}")
        return result

    def stage_2_dspy_optimize(self, baseline_prompt: str) -> PipelineResult:
        """Stage 2: DSPy optimization."""
        print("\n=== Stage 2: DSPy Optimization ===")

        # eval_set을 DSPy 형식으로 변환
        # 데모 목적으로 개념적 접근 방식을 보여줌
        # 실제로는 적절한 DSPy 시그니처를 정의해야 함

        # OPRO 스타일 접근 방식으로 최적화된 프롬프트 생성
        best_prompt = baseline_prompt
        best_score = 0

        for iteration in range(5):
            # LLM에게 프롬프트 개선 요청
            improve_msg = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=500,
                messages=[{
                    "role": "user",
                    "content": f"""Improve this prompt for better accuracy.
Keep the {{input}} placeholder. Return ONLY the improved prompt.

Current prompt (accuracy {best_score:.3f}):
{best_prompt}

Focus on: clearer instructions, better format specification,
disambiguation of edge cases."""
                }]
            )
            candidate = improve_msg.content[0].text.strip()
            if "{input}" not in candidate and "inputs" not in candidate:
                continue

            metrics = self.evaluate_prompt(candidate)
            if metrics["accuracy"] > best_score:
                best_score = metrics["accuracy"]
                best_prompt = candidate
                print(f"  Iteration {iteration}: New best {best_score:.3f}")

        metrics = self.evaluate_prompt(best_prompt)
        result = PipelineResult(
            stage="optimized",
            prompt_or_config=best_prompt,
            accuracy=metrics["accuracy"],
            token_count=int(metrics["avg_tokens"]),
            cost_per_call=metrics["cost_per_call"],
            timestamp=datetime.now().isoformat()
        )
        self.log.append(result)
        return result

    def stage_3_compress(self, optimized_prompt: str) -> PipelineResult:
        """Stage 3: Prompt compression."""
        print("\n=== Stage 3: Compression ===")

        compress_msg = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=500,
            messages=[{
                "role": "user",
                "content": f"""Compress this prompt to ~50% of its length.
Keep ALL instructions and format specs. Remove filler and redundancy.
Preserve all placeholders (like {{input}}).

Original:
{optimized_prompt}

Compressed (output ONLY the compressed prompt):"""
            }]
        )
        compressed = compress_msg.content[0].text.strip()
        metrics = self.evaluate_prompt(compressed)
        result = PipelineResult(
            stage="compressed",
            prompt_or_config=compressed,
            accuracy=metrics["accuracy"],
            token_count=int(metrics["avg_tokens"]),
            cost_per_call=metrics["cost_per_call"],
            timestamp=datetime.now().isoformat()
        )
        self.log.append(result)
        print(f"Compressed accuracy: {metrics['accuracy']:.3f}, "
              f"tokens: {metrics['avg_tokens']:.0f}")
        return result

    def stage_4_validate(self) -> bool:
        """Stage 4: Validate compression did not degrade quality."""
        print("\n=== Stage 4: Validation ===")
        optimized = next(r for r in self.log if r.stage == "optimized")
        compressed = next(r for r in self.log if r.stage == "compressed")

        accuracy_drop = optimized.accuracy - compressed.accuracy
        acceptable = accuracy_drop <= self.max_compression_drop

        print(f"Accuracy drop from compression: {accuracy_drop:.3f}")
        print(f"Threshold: {self.max_compression_drop}")
        print(f"Acceptable: {acceptable}")

        if not acceptable:
            print("WARNING: Compression degraded quality beyond threshold!")
            print("Falling back to uncompressed optimized prompt.")

        return acceptable

    def stage_5_produce_artifact(self) -> dict:
        """Stage 5: Generate production artifact with documentation."""
        print("\n=== Stage 5: Production Artifact ===")

        compression_ok = self.stage_4_validate()

        if compression_ok:
            final = next(r for r in self.log if r.stage == "compressed")
        else:
            final = next(r for r in self.log if r.stage == "optimized")

        baseline = next(r for r in self.log if r.stage == "baseline")

        artifact = {
            "task": self.task_name,
            "production_prompt": final.prompt_or_config,
            "model": "claude-sonnet-4-20250514",
            "metrics": {
                "accuracy": final.accuracy,
                "avg_input_tokens": final.token_count,
                "cost_per_call": final.cost_per_call,
            },
            "improvements_over_baseline": {
                "accuracy_change": final.accuracy - baseline.accuracy,
                "token_reduction": 1 - (final.token_count / baseline.token_count),
                "cost_reduction": 1 - (final.cost_per_call / baseline.cost_per_call),
            },
            "optimization_log": [asdict(r) for r in self.log],
            "created_at": datetime.now().isoformat(),
            "eval_set_size": len(self.eval_set),
        }

        # Save artifact
        filename = f"prompt_artifact_{self.task_name}.json"
        with open(filename, "w") as f:
            json.dump(artifact, f, indent=2)

        print(f"\nProduction artifact saved to {filename}")
        print(f"Final accuracy: {final.accuracy:.3f}")
        print(f"Cost per call: ${final.cost_per_call:.6f}")
        print(f"Improvement: {artifact['improvements_over_baseline']}")

        return artifact

    def run(self, baseline_prompt: str) -> dict:
        """Run the full pipeline."""
        self.stage_1_baseline(baseline_prompt)
        optimized = self.stage_2_dspy_optimize(baseline_prompt)
        self.stage_3_compress(optimized.prompt_or_config)
        return self.stage_5_produce_artifact()

# Usage
pipeline = PromptOptimizationPipeline(
    task_name="sentiment_classification",
    eval_set=[
        {"inputs": {"input": "Great product!"}, "expected": "positive"},
        {"inputs": {"input": "Terrible, broke immediately"}, "expected": "negative"},
        # ... 100+ examples
    ],
    accuracy_threshold=0.95,
    max_accuracy_drop_from_compression=0.02
)

artifact = pipeline.run(
    baseline_prompt="Classify the sentiment of this text as positive or negative: {input}"
)
```

파이프라인은 단계 간 품질 게이트가 있는 명확한 5단계 프로세스를 따릅니다. 각 단계는 결과를 기록하여 재현성과 비교를 가능하게 합니다. 압축 검증 단계(Stage 4)는 안전망으로 작동하여 압축이 품질을 해치면 압축되지 않은 프롬프트로 폴백합니다. 최종 아티팩트는 프로덕션 배포 준비가 된 문서화된 JSON 파일입니다.

</details>

---

**이전**: [RAG 프롬프트 패턴](./10_RAG_Prompt_Patterns.md) | **다음**: [평가와 지표](./12_Evaluation_and_Metrics.md)
