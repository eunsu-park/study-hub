# 20. Instruction Tuning

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
5. Zheng et al. (2023). "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena"
