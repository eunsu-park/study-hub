# 20. Instruction Tuning

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. Instruction Tuning이 사전 학습된 LLM을 텍스트 완성 모델에서 지시 수행(instruction-following) 어시스턴트로 변환하는 방법을 설명할 수 있다
2. 표준화된 프롬프트 템플릿(Alpaca, ShareGPT)을 사용하여 인스트럭션 튜닝 데이터셋을 구성하고 고품질 예시를 큐레이션할 수 있다
3. Hugging Face Trainer API를 사용하여 인스트럭션 데이터로 지도 파인튜닝(Supervised Fine-Tuning, SFT)을 구현할 수 있다
4. 인스트럭션 튜닝에서 데이터셋 다양성과 품질의 역할을 분석하고 FLAN, Alpaca, Dolly 데이터셋을 비교할 수 있다
5. 자동화된 벤치마크를 사용하여 인스트럭션 튜닝된 모델을 평가하고 제로샷(zero-shot) 일반화의 한계를 설명할 수 있다

---

## 개요

Instruction Tuning은 pre-trained LLM을 자연어 지시사항을 따르도록 fine-tuning하는 방법입니다. 이를 통해 모델이 다양한 태스크를 zero-shot으로 수행할 수 있게 됩니다.

---

## 1. Instruction Tuning 개요

### 1.1 개념

```
Before Instruction Tuning:
User: "Translate to French: Hello"
Model: "Translate to French: Hello. How are you? I am..."
(completion 모드로 동작)

After Instruction Tuning:
User: "Translate to French: Hello"
Model: "Bonjour"
(instruction following)

핵심 변화:
- 문장 완성 → 지시 수행
- Emergent abilities 향상
- Zero-shot 일반화
```

### 1.2 학습 데이터 형식

```python
# Instruction tuning 데이터 예시
instruction_data = [
    {
        "instruction": "Summarize the following article.",
        "input": "The stock market experienced significant volatility...",
        "output": "Stock markets showed high volatility due to..."
    },
    {
        "instruction": "Translate the following text to Korean.",
        "input": "Hello, how are you?",
        "output": "안녕하세요, 어떻게 지내세요?"
    },
    {
        "instruction": "Write a poem about autumn.",
        "input": "",
        "output": "Leaves of gold and crimson fall..."
    }
]

# Prompt template
def format_instruction(example):
    if example["input"]:
        return f"""### Instruction:
{example["instruction"]}

### Input:
{example["input"]}

### Response:
{example["output"]}"""
    else:
        return f"""### Instruction:
{example["instruction"]}

### Response:
{example["output"]}"""
```

---

## 2. FLAN (Finetuned Language Net)

### 2.1 FLAN-T5

```
FLAN 학습 데이터:
┌─────────────────────────────────────────────────────────┐
│  1,836 tasks from 473 datasets                          │
│                                                          │
│  Categories:                                             │
│  - NLU (sentiment, NLI, QA)                             │
│  - NLG (summarization, translation)                     │
│  - Reasoning (math, logic)                              │
│  - Dialog                                               │
│                                                          │
│  Data mixing:                                            │
│  - Task proportional mixing                              │
│  - Examples proportional mixing                          │
│  - Temperature-based sampling (T=3)                      │
└─────────────────────────────────────────────────────────┘
```

```python
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

def use_flan_t5():
    """FLAN-T5 사용"""
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-xl")
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-xl")

    # Zero-shot instruction
    prompts = [
        "Translate to German: The weather is nice today.",
        "What is the sentiment of: I love this product!",
        "Answer the question: What is the capital of France?",
        "Summarize: The quick brown fox jumps over the lazy dog. The dog was sleeping."
    ]

    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt")
        outputs = model.generate(**inputs, max_new_tokens=100)
        print(f"Q: {prompt}")
        print(f"A: {tokenizer.decode(outputs[0], skip_special_tokens=True)}\n")
```

### 2.2 Chain-of-Thought FLAN

```python
# CoT 데이터 포함
cot_example = {
    "instruction": "Solve the math problem step by step.",
    "input": "If John has 5 apples and gives 2 to Mary, how many does he have?",
    "output": """Let me solve this step by step:
1. John starts with 5 apples
2. John gives 2 apples to Mary
3. Remaining apples = 5 - 2 = 3

Therefore, John has 3 apples."""
}

# 학습 시 CoT 데이터 비율 조절
# 일반적으로 9:1 (non-CoT : CoT)
```

---

## 3. Self-Instruct

### 3.1 개념

```
Self-Instruct 파이프라인:
┌────────────────────────────────────────────────────────┐
│  1. Seed Tasks (175개 수동 작성)                        │
│         ↓                                              │
│  2. Task Generation (LLM이 새 instruction 생성)        │
│         ↓                                              │
│  3. Instance Generation (input/output 쌍 생성)        │
│         ↓                                              │
│  4. Filtering (품질 필터링)                            │
│         ↓                                              │
│  5. Fine-tuning                                        │
└────────────────────────────────────────────────────────┘

장점:
- 인간 라벨링 최소화
- 다양한 태스크 자동 생성
- 비용 효율적
```

```python
import openai
from typing import List, Dict
import json
import random

class SelfInstructGenerator:
    """Self-Instruct 데이터 생성기"""

    def __init__(self, seed_tasks: List[Dict], model: str = "gpt-4"):
        self.seed_tasks = seed_tasks
        self.generated_tasks = []
        self.model = model

    def generate_instruction(self, num_examples: int = 3) -> str:
        """새로운 instruction 생성"""
        # 시드에서 샘플
        examples = random.sample(self.seed_tasks + self.generated_tasks,
                                min(num_examples, len(self.seed_tasks)))

        examples_text = "\n".join([
            f"Task {i+1}: {ex['instruction']}"
            for i, ex in enumerate(examples)
        ])

        prompt = f"""Here are some example tasks:

{examples_text}

Generate a new and different task instruction. Be creative and diverse.
The task should be clear and specific.

New task instruction:"""

        response = openai.ChatCompletion.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.8,
            max_tokens=100
        )

        return response.choices[0].message.content.strip()

    def generate_instance(self, instruction: str) -> Dict:
        """instruction에 대한 input/output 생성"""
        prompt = f"""Given the following instruction, generate an appropriate input and output pair.

Instruction: {instruction}

Generate:
1. An input (can be empty if not needed)
2. The expected output

Format:
Input: [your input or "N/A"]
Output: [expected output]"""

        response = openai.ChatCompletion.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=500
        )

        # 파싱
        text = response.choices[0].message.content
        input_text = self._extract_field(text, "Input:")
        output_text = self._extract_field(text, "Output:")

        return {
            "instruction": instruction,
            "input": input_text if input_text != "N/A" else "",
            "output": output_text
        }

    def _extract_field(self, text: str, field: str) -> str:
        lines = text.split("\n")
        for i, line in enumerate(lines):
            if field in line:
                # 같은 줄 또는 다음 줄
                content = line.replace(field, "").strip()
                if content:
                    return content
                elif i + 1 < len(lines):
                    return lines[i + 1].strip()
        return ""

    def filter_instance(self, instance: Dict) -> bool:
        """품질 필터링"""
        # 길이 체크
        if len(instance["instruction"]) < 10:
            return False
        if len(instance["output"]) < 5:
            return False

        # 중복 체크
        for existing in self.generated_tasks:
            if self._similarity(instance["instruction"],
                              existing["instruction"]) > 0.7:
                return False

        return True

    def _similarity(self, a: str, b: str) -> float:
        """간단한 유사도 (실제로는 embedding 사용)"""
        words_a = set(a.lower().split())
        words_b = set(b.lower().split())
        intersection = words_a & words_b
        union = words_a | words_b
        return len(intersection) / len(union) if union else 0

    def generate_dataset(self, num_instances: int = 1000) -> List[Dict]:
        """데이터셋 생성"""
        while len(self.generated_tasks) < num_instances:
            # 새 instruction 생성
            instruction = self.generate_instruction()

            # Instance 생성
            instance = self.generate_instance(instruction)

            # 필터링
            if self.filter_instance(instance):
                self.generated_tasks.append(instance)
                print(f"Generated {len(self.generated_tasks)}/{num_instances}")

        return self.generated_tasks
```

---

## 4. Evol-Instruct (WizardLM)

### 4.1 개념

```
Evol-Instruct: instruction의 복잡도를 점진적으로 증가

Evolution Strategies:
┌────────────────────────────────────────────────────────┐
│  In-Depth Evolution:                                   │
│  - Add constraints (제약 추가)                         │
│  - Deepen (더 깊게)                                    │
│  - Concretize (구체화)                                │
│  - Increase reasoning (추론 강화)                      │
│  - Complicate input (입력 복잡화)                      │
│                                                        │
│  In-Breadth Evolution:                                 │
│  - Mutation (변형)                                     │
│  - Topic extension (주제 확장)                         │
│  - Method variation (방법 변경)                        │
└────────────────────────────────────────────────────────┘
```

```python
class EvolInstructGenerator:
    """Evol-Instruct 데이터 생성"""

    EVOLUTION_PROMPTS = {
        "add_constraints": """I want you to make the instruction more complex.
You should add one or more constraints/requirements to the instruction.

Original instruction: {instruction}

Evolved instruction with added constraints:""",

        "deepen": """I want you to make the instruction more complex.
If the original instruction can be solved in a few steps, please rewrite it
to require more steps to solve.

Original instruction: {instruction}

More complex instruction requiring deeper reasoning:""",

        "concretize": """I want you to make the instruction more concrete and specific.
Replace general concepts with specific examples.

Original instruction: {instruction}

More specific instruction:""",

        "reasoning": """I want you to make the instruction require multi-step reasoning.
The answer should require combining multiple pieces of information.

Original instruction: {instruction}

Instruction requiring multi-step reasoning:"""
    }

    def __init__(self, model: str = "gpt-4"):
        self.model = model

    def evolve_instruction(
        self,
        instruction: str,
        strategy: str = "deepen"
    ) -> str:
        """Instruction 진화"""
        prompt = self.EVOLUTION_PROMPTS[strategy].format(instruction=instruction)

        response = openai.ChatCompletion.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=200
        )

        return response.choices[0].message.content.strip()

    def multi_round_evolution(
        self,
        instruction: str,
        rounds: int = 3
    ) -> List[str]:
        """다중 라운드 진화"""
        evolved = [instruction]
        current = instruction

        strategies = ["add_constraints", "deepen", "reasoning", "concretize"]

        for i in range(rounds):
            strategy = strategies[i % len(strategies)]
            current = self.evolve_instruction(current, strategy)
            evolved.append(current)

        return evolved


# 예시
def evol_instruct_example():
    """Evol-Instruct 예시"""
    generator = EvolInstructGenerator()

    # 원본 instruction
    original = "Write a function to sort a list."

    # 진화
    evolved = generator.multi_round_evolution(original, rounds=3)

    print("Evolution chain:")
    for i, inst in enumerate(evolved):
        print(f"\nRound {i}: {inst}")

    # 예상 결과:
    # Round 0: Write a function to sort a list.
    # Round 1: Write a function to sort a list of integers in ascending order,
    #          handling negative numbers and duplicates.
    # Round 2: Write a function to sort a list of integers using merge sort,
    #          with O(n log n) time complexity, handling edge cases like
    #          empty lists and lists with one element.
    # Round 3: Implement a stable merge sort algorithm that sorts a list of
    #          objects by a given key, maintains relative order of equal
    #          elements, handles None values, and returns both the sorted
    #          list and the number of comparisons made.
```

---

## 5. Alpaca/Vicuna 스타일 학습

### 5.1 Stanford Alpaca

```python
# Alpaca 데이터 형식
alpaca_format = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:
{output}"""

# 학습 코드
from transformers import (
    LlamaForCausalLM, LlamaTokenizer,
    TrainingArguments, Trainer, DataCollatorForSeq2Seq
)
from datasets import load_dataset

def train_alpaca_style():
    """Alpaca 스타일 학습"""

    # 모델 로드
    model = LlamaForCausalLM.from_pretrained(
        "meta-llama/Llama-2-7b",
        torch_dtype=torch.float16
    )
    tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b")
    tokenizer.pad_token = tokenizer.eos_token

    # 데이터셋 로드
    dataset = load_dataset("tatsu-lab/alpaca")

    def format_example(example):
        if example["input"]:
            text = f"""Below is an instruction that describes a task, paired with an input that provides further context.

### Instruction:
{example["instruction"]}

### Input:
{example["input"]}

### Response:
{example["output"]}{tokenizer.eos_token}"""
        else:
            text = f"""Below is an instruction that describes a task.

### Instruction:
{example["instruction"]}

### Response:
{example["output"]}{tokenizer.eos_token}"""

        return tokenizer(text, truncation=True, max_length=512)

    tokenized_dataset = dataset.map(format_example)

    # 학습 설정
    training_args = TrainingArguments(
        output_dir="./alpaca-output",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=8,
        learning_rate=2e-5,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        fp16=True,
        logging_steps=10,
        save_strategy="epoch"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True)
    )

    trainer.train()
```

### 5.2 ShareGPT/Vicuna 형식

```python
# ShareGPT 대화 형식
sharegpt_format = {
    "conversations": [
        {"from": "human", "value": "What is machine learning?"},
        {"from": "gpt", "value": "Machine learning is a subset of AI..."},
        {"from": "human", "value": "Can you give an example?"},
        {"from": "gpt", "value": "Sure! A common example is spam detection..."}
    ]
}

# Vicuna 대화 템플릿
def format_vicuna_conversation(conversations):
    """Vicuna 형식으로 변환"""
    formatted = ""

    for turn in conversations:
        if turn["from"] == "human":
            formatted += f"USER: {turn['value']}\n"
        else:
            formatted += f"ASSISTANT: {turn['value']}</s>\n"

    return formatted

# Chat template (HuggingFace 방식)
def apply_chat_template(tokenizer, messages):
    """Chat template 적용"""
    # tokenizer에 chat_template이 설정되어 있는 경우
    return tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True
    )
```

---

## 6. 학습 전략

### 6.1 데이터 품질 vs 양

```python
class DataQualityChecker:
    """데이터 품질 검사"""

    def check_quality(self, example: Dict) -> Dict:
        """품질 점수 계산"""
        scores = {}

        # 1. 길이 적절성
        inst_len = len(example["instruction"].split())
        out_len = len(example["output"].split())
        scores["length"] = min(inst_len / 20, 1.0) * min(out_len / 50, 1.0)

        # 2. 형식 일관성
        scores["format"] = 1.0 if self._check_format(example) else 0.5

        # 3. 응답 관련성 (간단한 휴리스틱)
        keywords = set(example["instruction"].lower().split())
        response_words = set(example["output"].lower().split())
        overlap = len(keywords & response_words) / len(keywords) if keywords else 0
        scores["relevance"] = min(overlap * 2, 1.0)

        # 4. 유해성 (간단한 필터)
        scores["safety"] = 0.0 if self._contains_harmful(example["output"]) else 1.0

        # 종합 점수
        scores["total"] = sum(scores.values()) / len(scores)

        return scores

    def _check_format(self, example: Dict) -> bool:
        """형식 검사"""
        return (
            len(example["instruction"]) > 0 and
            len(example["output"]) > 0 and
            not example["output"].startswith("I cannot") and
            not example["output"].startswith("As an AI")
        )

    def _contains_harmful(self, text: str) -> bool:
        """유해 콘텐츠 검사 (간단한 버전)"""
        harmful_patterns = ["hack", "illegal", "weapon", "drug"]
        return any(p in text.lower() for p in harmful_patterns)
```

### 6.2 데이터 믹싱

```python
def create_instruction_mix(
    datasets: Dict[str, List[Dict]],
    weights: Dict[str, float],
    total_size: int
) -> List[Dict]:
    """태스크별 데이터 믹싱"""
    mixed = []

    for task, data in datasets.items():
        weight = weights.get(task, 1.0)
        num_samples = int(total_size * weight / sum(weights.values()))
        sampled = random.sample(data, min(num_samples, len(data)))
        mixed.extend(sampled)

    random.shuffle(mixed)
    return mixed[:total_size]

# 예시 믹스
datasets = {
    "qa": qa_data,
    "summarization": summary_data,
    "translation": translation_data,
    "coding": coding_data,
    "reasoning": reasoning_data
}

weights = {
    "qa": 1.0,
    "summarization": 1.0,
    "translation": 0.5,
    "coding": 2.0,  # 코딩에 더 가중치
    "reasoning": 1.5
}

mixed_dataset = create_instruction_mix(datasets, weights, total_size=50000)
```

---

## 핵심 정리

### Instruction Tuning 핵심
```
1. FLAN: 다양한 태스크 믹싱, CoT 포함
2. Self-Instruct: LLM으로 데이터 자동 생성
3. Evol-Instruct: 점진적 복잡도 증가
4. Alpaca: 간단한 instruction format
5. Vicuna/ShareGPT: 대화 형식 학습
```

### 데이터 품질 체크리스트
```
□ Instruction이 명확한가?
□ Output이 instruction을 따르는가?
□ 형식이 일관적인가?
□ 유해 콘텐츠가 없는가?
□ 다양성이 충분한가?
□ 난이도 분포가 적절한가?
```

---

## 참고 자료

1. Wei et al. (2021). "Finetuned Language Models Are Zero-Shot Learners" (FLAN)
2. Wang et al. (2022). "Self-Instruct: Aligning Language Models with Self-Generated Instructions"
3. Xu et al. (2023). "WizardLM: Empowering Large Language Models to Follow Complex Instructions"
4. Taori et al. (2023). "Stanford Alpaca"

---

## 연습 문제

### 연습 문제 1: 지시 형식 설계
"파이썬 리스트가 정렬되어 있는지 확인하는 방법"에 대한 동일한 질문/답변 쌍을 세 가지 다른 형식(Alpaca, ShareGPT, JSON 구조화 출력)으로 포맷하세요. 각 형식이 가장 적합한 사용 사례를 식별하세요.

<details>
<summary>정답 보기</summary>

**Alpaca 형식** (단일 턴 지시-응답):
```json
{
  "instruction": "파이썬 리스트가 정렬되어 있는지 확인하는 방법을 알려주세요.",
  "input": "",
  "output": "파이썬 리스트가 정렬되어 있는지 확인하는 몇 가지 방법이 있습니다:\n\n1. **한 줄 표현식** (O(n), 조기 종료):\n```python\nis_sorted = all(lst[i] <= lst[i+1] for i in range(len(lst)-1))\n```\n\n2. **정렬과 비교** (O(n log n)):\n```python\nis_sorted = lst == sorted(lst)\n```\n\n첫 번째 방법이 더 효율적입니다. 정렬되지 않은 쌍을 찾자마자 멈추지만, 두 번째 방법은 이미 정렬되지 않은 경우에도 항상 전체 리스트를 정렬합니다."
}
```
적합한 경우: 단일 턴 질의응답(QA) 쌍, 지시 따르기 태스크.

**ShareGPT 형식** (다중 턴 대화):
```json
{
  "conversations": [
    {
      "from": "human",
      "value": "Hey, I'm trying to check if my Python list is sorted. Is there a quick way to do this?"
    },
    {
      "from": "gpt",
      "value": "Yes! There are a few ways depending on your needs:\n\n1. **One-liner** (O(n), early termination):\n```python\nis_sorted = all(lst[i] <= lst[i+1] for i in range(len(lst)-1))\n```\n\n2. **Simple comparison** (O(n log n)):\n```python\nis_sorted = lst == sorted(lst)\n```\n\nWhich would you prefer?"
    },
    {
      "from": "human",
      "value": "What's the difference between them?"
    },
    {
      "from": "gpt",
      "value": "The key difference is efficiency: the first stops as soon as it finds an out-of-order pair (best case O(1)), while the second always sorts the entire list even if it's already unsorted."
    }
  ]
}
```
적합한 경우: 대화형 모델 훈련, 후속 질문 처리.

**JSON 구조화 출력 형식**:
```json
{
  "instruction": "How do I check if a Python list is sorted? Provide a structured response.",
  "input": "list = [1, 2, 3, 4, 5]",
  "output": "{\"solution\": \"all(lst[i] <= lst[i+1] for i in range(len(lst)-1))\", \"complexity\": \"O(n)\", \"alternative\": \"lst == sorted(lst)\", \"explanation\": \"Iterates through pairs, returns False on first out-of-order element\"}"
}
```
적합한 경우: 구조화된 데이터(JSON, XML)를 출력해야 하는 모델 훈련, 도구 사용 시나리오.

</details>

### 연습 문제 2: 데이터셋 다양성 대 품질 트레이드오프
FLAN은 1,836개의 다양한 태스크를 사용하고, Alpaca는 단일 셀프-인스트럭트(self-instruct) 파이프라인에서 생성된 52,000개의 예제를 사용합니다. 각 접근법의 강점을 설명하고, 특정 배포 환경에서 어떤 약점이 더 중요한지 드러낼 수 있는 평가 방법을 제안하세요.

<details>
<summary>정답 보기</summary>

**FLAN의 강점**:
- 극도의 태스크 다양성: 1,836개 벤치마크에 걸쳐 분류, 요약, 번역, 상식 추론 등을 다룹니다.
- 학술 NLP 벤치마크의 잘 선별된 인간 검증 태스크.
- 태스크 형식 변형을 통해 제로샷(zero-shot) 일반화를 명시적으로 테스트합니다.
- **약점**: 태스크 공식화가 학술/벤치마크 스타일입니다. 실제 사용자 쿼리는 대화체이고 비공식적이며 벤치마크 템플릿과 맞지 않습니다. FLAN으로 훈련된 모델은 "Classify the sentiment: [text]"에는 답할 수 있지만 "이 리뷰가 긍정적인가요 부정적인가요?"에는 어려움을 겪을 수 있습니다.

**Alpaca의 강점**:
- 실제 사용자 쿼리와 유사한 자연스러운 지시 스타일의 52,000개 예제.
- 175개의 시드(seed) 예제에서 셀프-인스트럭트를 통해 생성된 다양한 지시 유형.
- 실제 사용에 맞는 대화체 실용적 형식.
- **약점**: GPT-3/GPT-4가 자동 생성 — 편향이 상속되고, 일관성이 부족하며, 품질이 다양합니다. 단일 모델 "교사"는 관점 다양성이 제한됩니다.

**어떤 약점이 더 중요한지 드러내는 평가**:
*고객 지원 챗봇* 배포 시:
1. **테스트셋 1 (FLAN 약점)**: 비공식 언어의 100개 대화체 고객 쿼리("내 주문이 늦어요, 어떻게 해요?" vs. "문제: [배송 지연]. 카테고리: [배송 문제]") — 학술적 형식이 자연스러운 언어로 일반화되는지 측정.
2. **테스트셋 2 (Alpaca 약점)**: 잘못된 전제가 있는 100개 엣지 케이스("디지털 아이템을 환불받을 수 있나요? 예 또는 아니오로만") — 제약 조건 하에서 지시 따르기 정확도 측정.
3. **테스트셋 3 (두 가지 모두)**: 대화 중간에 태스크 전환이 있는 다중 턴 대화 — 단일 턴 데이터(Alpaca) 또는 프롬프트당 단일 태스크(FLAN) 훈련이 실제 대화로 일반화되는지 테스트.

더 큰 격차를 드러내는 테스트가 해당 배포 환경에 더 적합한 데이터셋 접근법을 결정합니다.

</details>

### 연습 문제 3: 손실 마스킹 구현
지시 튜닝에서 훈련 손실은 지시 토큰이 아닌 **응답** 토큰에서만 계산되어야 합니다. 마스킹 로직을 구현하세요.

```python
def create_labels_with_masking(
    input_ids: list,
    instruction_length: int
) -> list:
    """
    Create labels for instruction tuning where
    instruction tokens are masked (set to -100)
    and only response tokens contribute to the loss.

    Args:
        input_ids: Full tokenized sequence [instruction + response]
        instruction_length: Number of instruction tokens to mask

    Returns:
        labels: Same length as input_ids, with -100 for masked positions
    """
    IGNORE_INDEX = -100
    # Implement here
    pass

# Example:
# input_ids = [101, 234, 567, 789, 012, 345, 678]  (7 tokens)
# instruction_length = 3  (first 3 are instruction tokens)
# Expected labels = [-100, -100, -100, 789, 012, 345, 678]
# Why: loss is only computed on tokens 4-7 (the response)
```

<details>
<summary>정답 보기</summary>

```python
def create_labels_with_masking(
    input_ids: list,
    instruction_length: int
) -> list:
    """
    Create labels for instruction tuning where
    instruction tokens are masked (set to -100)
    """
    IGNORE_INDEX = -100
    labels = list(input_ids)  # Copy to avoid modifying original

    # Mask all instruction tokens
    for i in range(instruction_length):
        labels[i] = IGNORE_INDEX

    return labels


# Example verification:
input_ids = [101, 234, 567, 789, 012, 345, 678]
labels = create_labels_with_masking(input_ids, instruction_length=3)
# labels = [-100, -100, -100, 789, 12, 345, 678]  ✓


# Extended version for chat templates (mask system + instruction turns)
def create_labels_for_conversation(
    tokenizer,
    conversation: list,
    model_response_roles: list = ["assistant", "gpt"]
) -> dict:
    """
    Create labels for multi-turn conversations.
    Only assistant turns contribute to loss.

    conversation: [{"role": "user", "value": "..."}, {"role": "assistant", "value": "..."}]
    """
    IGNORE_INDEX = -100

    # Tokenize full conversation
    full_text = tokenizer.apply_chat_template(conversation, tokenize=False)
    input_ids = tokenizer(full_text, return_tensors="pt").input_ids[0]
    labels = input_ids.clone()

    # Find and mask non-assistant tokens
    # Strategy: tokenize each turn separately, find positions in full sequence
    offset = 0
    for turn in conversation:
        turn_tokens = tokenizer(turn["value"], return_tensors="pt").input_ids[0]
        turn_len = len(turn_tokens)

        if turn["role"] not in model_response_roles:
            # Mask this turn (instruction/system/user)
            labels[offset:offset + turn_len] = IGNORE_INDEX

        offset += turn_len

    return {"input_ids": input_ids, "labels": labels}
```

**지시 토큰을 마스킹하는 이유**:
- 마스킹 없이: 모델은 "Translate: Hello"와 "Bonjour" 모두를 예측하도록 훈련됩니다. 지시 토큰이 손실에 동등하게 기여하므로 지시를 따르기보다 그대로 반복하는 것을 학습할 수 있습니다.
- 마스킹 있이: 모델은 지시 맥락이 주어졌을 때 올바른 응답만 예측하는 법을 학습합니다. 지시는 생성할 출력이 아닌 주의를 기울일 입력임을 학습합니다.

</details>

### 연습 문제 4: Evol-Instruct 품질 분석
WizardLM의 Evol-Instruct(진화적 지시 생성)는 지시 복잡도를 점진적으로 높입니다. "복잡도"가 항상 바람직한지 평가하고, 모델 품질을 저하시킬 수 있는 사례를 식별하세요.

시드 지시: "리스트를 정렬하는 함수를 작성하세요"

3단계 진화를 추적하고 추가된 복잡도가 모델 훈련에 해를 끼칠 수 있는 단계 하나를 식별하세요.

<details>
<summary>정답 보기</summary>

**진화 추적**:

**레벨 0 (시드)**: "리스트를 정렬하는 함수를 작성하세요"
- 간단하고, 명확하며, 기본적인 파이썬 태스크.

**레벨 1 (제약 추가)**: "특정 키로 딕셔너리 리스트를 정렬하는 함수를 작성하되, 누락된 키를 우아하게 처리하세요"
- 좋은 복잡도 증가: 현실적인 실제 시나리오 추가, 오류 처리 테스트.
- 훈련 가치: 높음 — 우아한 저하(graceful degradation)를 가르칩니다.

**레벨 2 (다중 요구사항)**: "지정된 정렬 순서(키별 오름차순/내림차순)로 여러 키로 딕셔너리 리스트를 정렬하고, 누락된 키는 None으로 처리하여 끝에 배치하며, O(n log n) 시간 복잡도 보장 및 단위 테스트 포함"
- 적당한 복잡도 증가: 현실적이지만 더 긴 출력 필요.
- 훈련 가치: 중간 — 훈련 컨텍스트 윈도우에 비해 너무 길 수 있음.

**레벨 3 (잠재적으로 유해)**: "다음을 수행하는 범용 정렬 함수 구현: (1) 비교 가능한 모든 이터러블 객체 허용, (2) 비용이 많이 드는 비교에 대한 메모이제이션이 있는 커스텀 비교 함수 지원, (3) 입력 크기 휴리스틱에 따라 전환되는 QuickSort와 TimSort 내부 구현, (4) 독스트링에 상세한 복잡도 분석 제공, (5) 스레딩 락으로 동시 접근 처리, (6) 정렬 안정성 보장 증명, (7) 엣지 케이스를 다루는 10개 단위 테스트 포함"
- **문제점**: 이것은 하나의 프롬프트에 비합리적으로 복잡한 응답입니다.
- **훈련에 해로운 이유**:
  1. 예상 "출력"이 수백 줄의 코드 — 모델은 단일 턴 프롬프트에서 극히 긴 출력을 생성하는 법을 학습해야 하는데, 이는 실제 사용자가 상호작용하는 방식이 아닙니다.
  2. 지시에 모순된 요구사항 포함 (QuickSort는 안정적이지 않음; "안정 정렬 + QuickSort" 요구사항은 모순되거나 복잡한 해결책 필요).
  3. 올바른 답은 아키텍처 지식이 필요 (정렬에 스레딩 락은 일반적으로 안티패턴); 이것으로 훈련하면 불필요한 복잡도를 추가하는 것을 가르칩니다.
  4. 지나치게 복잡한 예제로 훈련된 모델은 프로덕션에서 요청하지 않은 복잡도를 추가할 수 있습니다("과잉 엔지니어링" 동작).

**결론**: Evol-Instruct는 1-2 레벨에서는 유용하지만, 자동 생성된 3+ 레벨 예제는 훈련 전 전문가 검증이 필요한 경우가 많습니다.

</details>
5. Zheng et al. (2023). "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena"
