# 20. Instruction Tuning

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain how Instruction Tuning transforms a pretrained LLM from a text completion model into an instruction-following assistant
2. Construct instruction tuning datasets using standardized prompt templates (Alpaca, ShareGPT) and curate high-quality examples
3. Implement supervised fine-tuning (SFT) with instruction data using the Hugging Face Trainer API
4. Analyze the role of dataset diversity and quality in instruction tuning, comparing FLAN, Alpaca, and Dolly datasets
5. Evaluate instruction-tuned models using automated benchmarks and describe the limitations of zero-shot generalization

---

## Overview

Instruction Tuning is a method of fine-tuning pre-trained LLMs to follow natural language instructions. This enables models to perform various tasks in a zero-shot manner.

---

## 1. Instruction Tuning Overview

### 1.1 Concept

```
Before Instruction Tuning:
User: "Translate to French: Hello"
Model: "Translate to French: Hello. How are you? I am..."
(operates in completion mode)

After Instruction Tuning:
User: "Translate to French: Hello"
Model: "Bonjour"
(instruction following)

Key Changes:
- Sentence completion → Instruction execution
- Improved emergent abilities
- Zero-shot generalization
```

### 1.2 Training Data Format

```python
# Instruction tuning data example
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
FLAN Training Data:
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
    """Using FLAN-T5"""
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-xl")
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-xl")

    # Zero-shot instructions
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
# Including CoT data
cot_example = {
    "instruction": "Solve the math problem step by step.",
    "input": "If John has 5 apples and gives 2 to Mary, how many does he have?",
    "output": """Let me solve this step by step:
1. John starts with 5 apples
2. John gives 2 apples to Mary
3. Remaining apples = 5 - 2 = 3

Therefore, John has 3 apples."""
}

# Adjust CoT data ratio during training
# Typically 9:1 (non-CoT : CoT)
```

---

## 3. Self-Instruct

### 3.1 Concept

```
Self-Instruct Pipeline:
┌────────────────────────────────────────────────────────┐
│  1. Seed Tasks (175 manually written)                   │
│         ↓                                              │
│  2. Task Generation (LLM generates new instructions)   │
│         ↓                                              │
│  3. Instance Generation (generate input/output pairs)  │
│         ↓                                              │
│  4. Filtering (quality filtering)                      │
│         ↓                                              │
│  5. Fine-tuning                                        │
└────────────────────────────────────────────────────────┘

Advantages:
- Minimal human labeling
- Automatic diverse task generation
- Cost-effective
```

```python
import openai
from typing import List, Dict
import json
import random

class SelfInstructGenerator:
    """Self-Instruct data generator"""

    def __init__(self, seed_tasks: List[Dict], model: str = "gpt-4"):
        self.seed_tasks = seed_tasks
        self.generated_tasks = []
        self.model = model

    def generate_instruction(self, num_examples: int = 3) -> str:
        """Generate new instruction"""
        # Sample from seeds
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
        """Generate input/output for instruction"""
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

        # Parse
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
                # Same line or next line
                content = line.replace(field, "").strip()
                if content:
                    return content
                elif i + 1 < len(lines):
                    return lines[i + 1].strip()
        return ""

    def filter_instance(self, instance: Dict) -> bool:
        """Quality filtering"""
        # Length check
        if len(instance["instruction"]) < 10:
            return False
        if len(instance["output"]) < 5:
            return False

        # Duplicate check
        for existing in self.generated_tasks:
            if self._similarity(instance["instruction"],
                              existing["instruction"]) > 0.7:
                return False

        return True

    def _similarity(self, a: str, b: str) -> float:
        """Simple similarity (actually use embeddings)"""
        words_a = set(a.lower().split())
        words_b = set(b.lower().split())
        intersection = words_a & words_b
        union = words_a | words_b
        return len(intersection) / len(union) if union else 0

    def generate_dataset(self, num_instances: int = 1000) -> List[Dict]:
        """Generate dataset"""
        while len(self.generated_tasks) < num_instances:
            # Generate new instruction
            instruction = self.generate_instruction()

            # Generate instance
            instance = self.generate_instance(instruction)

            # Filter
            if self.filter_instance(instance):
                self.generated_tasks.append(instance)
                print(f"Generated {len(self.generated_tasks)}/{num_instances}")

        return self.generated_tasks
```

---

## 4. Evol-Instruct (WizardLM)

### 4.1 Concept

```
Evol-Instruct: Progressively increase instruction complexity

Evolution Strategies:
┌────────────────────────────────────────────────────────┐
│  In-Depth Evolution:                                   │
│  - Add constraints                                     │
│  - Deepen                                              │
│  - Concretize                                          │
│  - Increase reasoning                                  │
│  - Complicate input                                    │
│                                                        │
│  In-Breadth Evolution:                                 │
│  - Mutation                                            │
│  - Topic extension                                     │
│  - Method variation                                    │
└────────────────────────────────────────────────────────┘
```

```python
class EvolInstructGenerator:
    """Evol-Instruct data generation"""

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
        """Evolve instruction"""
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
        """Multi-round evolution"""
        evolved = [instruction]
        current = instruction

        strategies = ["add_constraints", "deepen", "reasoning", "concretize"]

        for i in range(rounds):
            strategy = strategies[i % len(strategies)]
            current = self.evolve_instruction(current, strategy)
            evolved.append(current)

        return evolved


# Example
def evol_instruct_example():
    """Evol-Instruct example"""
    generator = EvolInstructGenerator()

    # Original instruction
    original = "Write a function to sort a list."

    # Evolve
    evolved = generator.multi_round_evolution(original, rounds=3)

    print("Evolution chain:")
    for i, inst in enumerate(evolved):
        print(f"\nRound {i}: {inst}")

    # Expected result:
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

## 5. Alpaca/Vicuna Style Training

### 5.1 Stanford Alpaca

```python
# Alpaca data format
alpaca_format = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:
{output}"""

# Training code
from transformers import (
    LlamaForCausalLM, LlamaTokenizer,
    TrainingArguments, Trainer, DataCollatorForSeq2Seq
)
from datasets import load_dataset

def train_alpaca_style():
    """Alpaca-style training"""

    # Load model
    model = LlamaForCausalLM.from_pretrained(
        "meta-llama/Llama-2-7b",
        torch_dtype=torch.float16
    )
    tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b")
    tokenizer.pad_token = tokenizer.eos_token

    # Load dataset
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

    # Training configuration
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

### 5.2 ShareGPT/Vicuna Format

```python
# ShareGPT conversation format
sharegpt_format = {
    "conversations": [
        {"from": "human", "value": "What is machine learning?"},
        {"from": "gpt", "value": "Machine learning is a subset of AI..."},
        {"from": "human", "value": "Can you give an example?"},
        {"from": "gpt", "value": "Sure! A common example is spam detection..."}
    ]
}

# Vicuna conversation template
def format_vicuna_conversation(conversations):
    """Convert to Vicuna format"""
    formatted = ""

    for turn in conversations:
        if turn["from"] == "human":
            formatted += f"USER: {turn['value']}\n"
        else:
            formatted += f"ASSISTANT: {turn['value']}</s>\n"

    return formatted

# Chat template (HuggingFace method)
def apply_chat_template(tokenizer, messages):
    """Apply chat template"""
    # When chat_template is set on tokenizer
    return tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True
    )
```

---

## 6. Training Strategies

### 6.1 Data Quality vs Quantity

```python
class DataQualityChecker:
    """Data quality inspection"""

    def check_quality(self, example: Dict) -> Dict:
        """Calculate quality score"""
        scores = {}

        # 1. Length appropriateness
        inst_len = len(example["instruction"].split())
        out_len = len(example["output"].split())
        scores["length"] = min(inst_len / 20, 1.0) * min(out_len / 50, 1.0)

        # 2. Format consistency
        scores["format"] = 1.0 if self._check_format(example) else 0.5

        # 3. Response relevance (simple heuristic)
        keywords = set(example["instruction"].lower().split())
        response_words = set(example["output"].lower().split())
        overlap = len(keywords & response_words) / len(keywords) if keywords else 0
        scores["relevance"] = min(overlap * 2, 1.0)

        # 4. Harmfulness (simple filter)
        scores["safety"] = 0.0 if self._contains_harmful(example["output"]) else 1.0

        # Overall score
        scores["total"] = sum(scores.values()) / len(scores)

        return scores

    def _check_format(self, example: Dict) -> bool:
        """Format check"""
        return (
            len(example["instruction"]) > 0 and
            len(example["output"]) > 0 and
            not example["output"].startswith("I cannot") and
            not example["output"].startswith("As an AI")
        )

    def _contains_harmful(self, text: str) -> bool:
        """Harmful content check (simple version)"""
        harmful_patterns = ["hack", "illegal", "weapon", "drug"]
        return any(p in text.lower() for p in harmful_patterns)
```

### 6.2 Data Mixing

```python
def create_instruction_mix(
    datasets: Dict[str, List[Dict]],
    weights: Dict[str, float],
    total_size: int
) -> List[Dict]:
    """Task-specific data mixing"""
    mixed = []

    for task, data in datasets.items():
        weight = weights.get(task, 1.0)
        num_samples = int(total_size * weight / sum(weights.values()))
        sampled = random.sample(data, min(num_samples, len(data)))
        mixed.extend(sampled)

    random.shuffle(mixed)
    return mixed[:total_size]

# Example mix
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
    "coding": 2.0,  # More weight on coding
    "reasoning": 1.5
}

mixed_dataset = create_instruction_mix(datasets, weights, total_size=50000)
```

---

## Key Summary

### Instruction Tuning Core
```
1. FLAN: Diverse task mixing, includes CoT
2. Self-Instruct: Automatic data generation with LLM
3. Evol-Instruct: Progressive complexity increase
4. Alpaca: Simple instruction format
5. Vicuna/ShareGPT: Conversation format training
```

### Data Quality Checklist
```
□ Is the instruction clear?
□ Does the output follow the instruction?
□ Is the format consistent?
□ No harmful content?
□ Sufficient diversity?
□ Appropriate difficulty distribution?
```

---

## References

1. Wei et al. (2021). "Finetuned Language Models Are Zero-Shot Learners" (FLAN)
2. Wang et al. (2022). "Self-Instruct: Aligning Language Models with Self-Generated Instructions"
3. Xu et al. (2023). "WizardLM: Empowering Large Language Models to Follow Complex Instructions"
4. Taori et al. (2023). "Stanford Alpaca"
5. Zheng et al. (2023). "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena"

---

## Exercises

### Exercise 1: Instruction Format Design
Convert the following raw question-answer pair into three different instruction tuning formats: Alpaca, ShareGPT (multi-turn), and a JSON-structured output format. Explain when each format is most appropriate.

**Raw content**: A user asks how to check if a Python list is sorted, and the answer should provide a clean one-liner solution.

<details>
<summary>Show Answer</summary>

**Alpaca format** (single-turn, explicit instruction):
```json
{
  "instruction": "Write a Python one-liner to check if a list is sorted in ascending order.",
  "input": "",
  "output": "You can check if a list is sorted using: `all(lst[i] <= lst[i+1] for i in range(len(lst)-1))` or more concisely with `lst == sorted(lst)`. The first approach is O(n) and stops early if unsorted; the second creates a new sorted copy (also O(n log n))."
}
```
Best for: Single-turn question-answer pairs, instruction-following tasks.

**ShareGPT format** (multi-turn conversation):
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
Best for: Training conversational models, handling follow-up questions.

**JSON-structured output format**:
```json
{
  "instruction": "How do I check if a Python list is sorted? Provide a structured response.",
  "input": "list = [1, 2, 3, 4, 5]",
  "output": "{\"solution\": \"all(lst[i] <= lst[i+1] for i in range(len(lst)-1))\", \"complexity\": \"O(n)\", \"alternative\": \"lst == sorted(lst)\", \"explanation\": \"Iterates through pairs, returns False on first out-of-order element\"}"
}
```
Best for: Training models that need to output structured data (JSON, XML), tool use scenarios.

</details>

### Exercise 2: Dataset Diversity vs. Quality Trade-off
FLAN uses 1,836 diverse tasks, while Alpaca uses 52K examples from a single self-instruct pipeline. Describe the strength of each approach and propose an evaluation that would reveal which weakness matters more for a specific deployment.

<details>
<summary>Show Answer</summary>

**FLAN strengths**:
- Extreme task diversity: covers classification, summarization, translation, commonsense reasoning across 1,836 benchmarks.
- Well-curated, human-validated tasks from academic NLP benchmarks.
- Explicitly tests zero-shot generalization through task formatting variations.
- **Weakness**: Task formulations are academic/benchmark-style. Real user queries are conversational, informal, and don't match benchmark templates. Models trained on FLAN can answer "Classify the sentiment: [text]" but struggle with "Is this review positive or negative?"

**Alpaca strengths**:
- 52K examples in natural instruction style, resembling real user queries.
- Diverse instruction types generated via Self-Instruct from just 175 seed examples.
- Conversational, practical format matching real-world usage.
- **Weakness**: Self-generated by GPT-3/GPT-4 — inherited biases, inconsistencies, and quality varies. Single model "teacher" means limited perspective diversity.

**Evaluation to reveal which weakness matters more**:
For a *customer support chatbot* deployment:
1. **Test set 1 (FLAN weakness)**: 100 conversational customer queries in informal language ("my order is late, what do I do?" vs. "Problem: [order late]. Category: [shipping issue]") — measures whether academic formatting generalizes to natural speech.
2. **Test set 2 (Alpaca weakness)**: 100 edge cases with misleading premises ("Can I get a refund on a digital item? Yes or No only") — measures instruction-following precision under constraints.
3. **Test set 3 (both)**: Multi-turn conversations with task switches mid-conversation — tests if training on single-turn data (Alpaca) vs. single-task-per-prompt (FLAN) generalizes to real conversations.

The test that reveals the bigger gap determines which dataset approach is more appropriate for the deployment.

</details>

### Exercise 3: Loss Masking Implementation
In instruction tuning, the training loss should only be computed on the **response** tokens, not the instruction tokens. Implement the masking logic.

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
<summary>Show Answer</summary>

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

**Why masking instruction tokens matters**:
- Without masking: the model is trained to predict both "Translate: Hello" and "Bonjour". It may learn to repeat instructions verbatim rather than follow them, since instruction tokens contribute equally to the loss.
- With masking: the model only learns to predict the correct response given the instruction context. It learns that instructions are inputs to attend to, not outputs to generate.

</details>

### Exercise 4: Evol-Instruct Quality Analysis
WizardLM's Evol-Instruct progressively increases instruction complexity. Evaluate whether "complexity" is always desirable and identify cases where it could degrade model quality.

Given a seed instruction: "Write a function to sort a list"

Trace through 3 levels of evolution and identify one level where the added complexity might harm model training.

<details>
<summary>Show Answer</summary>

**Evolution trace**:

**Level 0 (Seed)**: "Write a function to sort a list"
- Simple, clear, fundamental Python task.

**Level 1 (Add constraints)**: "Write a function to sort a list of dictionaries by a specific key, handling missing keys gracefully"
- Good complexity increase: adds realistic real-world scenario, tests error handling.
- Training value: high — teaches graceful degradation.

**Level 2 (Multiple requirements)**: "Write a function to sort a list of dictionaries by multiple keys in specified sort orders (ascending/descending per key), handling missing keys by treating them as None and placing them at the end, with O(n log n) time complexity guarantee and a unit test"
- Moderate complexity increase: realistic but requires more output length.
- Training value: medium — may be too long for the training context window.

**Level 3 (Potentially harmful)**: "Implement a generic sort function that: (1) accepts any iterable of comparable objects, (2) supports custom comparison functions with memoization for expensive comparisons, (3) implements both QuickSort and TimSort internally, switching based on input size heuristic, (4) provides detailed complexity analysis in docstring, (5) handles concurrent access via threading locks, (6) provides sort stability guarantee proof, and (7) includes 10 unit tests covering edge cases"
- **Problem**: This is an unreasonably complex response to one prompt.
- **Why it harms training**:
  1. The expected "output" is hundreds of lines of code — the model must learn to generate extremely long outputs from a single-turn prompt, which isn't how real users interact.
  2. The instruction contains contradictory requirements (QuickSort is not stable; the requirement for "stable sort + QuickSort" is contradictory or requires complex workarounds).
  3. A correct answer would require architectural knowledge (threading locks for sorting is generally anti-pattern); training on this teaches the model to add unnecessary complexity.
  4. Models trained on over-complicated examples may add unasked-for complexity in production ("over-engineering" behavior).

**Conclusion**: Evol-Instruct is valuable for levels 1-2, but auto-generated level 3+ examples often become incoherent or require expert validation before training.

</details>
