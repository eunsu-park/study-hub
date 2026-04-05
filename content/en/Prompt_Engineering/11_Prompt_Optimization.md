# 11. Prompt Optimization

**Previous**: [RAG Prompt Patterns](./10_RAG_Prompt_Patterns.md) | **Next**: [Evaluation and Metrics](./12_Evaluation_and_Metrics.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain why manual prompt engineering has fundamental scalability limits and when automated optimization becomes necessary
2. Apply the DSPy framework to define signatures, build modules, and run optimizers for systematic prompt tuning
3. Describe OPRO and APE as representative approaches to LLM-driven prompt optimization
4. Implement prompt compression techniques to reduce token usage while preserving task performance
5. Evaluate cost-quality trade-offs and decide when to optimize prompts vs switch models

---

Manual prompt engineering -- the process of iteratively editing prompts by hand, testing them on a few examples, and tweaking based on intuition -- works remarkably well for simple tasks. But it hits a ceiling. When you have dozens of prompts serving different user populations, when small wording changes produce unexpected regressions, when you need to optimize across multiple metrics simultaneously (quality, cost, latency), manual iteration becomes a bottleneck. You cannot A/B test 50 prompt variants by hand. You cannot systematically explore the space of possible instructions. You cannot guarantee that your "improved" prompt does not break edge cases you forgot to test.

This lesson covers the emerging field of *automatic prompt optimization*: tools and techniques that treat prompt design as an optimization problem and apply algorithmic search to find better prompts. We start with the conceptual foundations, explore major frameworks (DSPy, OPRO, APE), and address practical concerns like cost-quality trade-offs and prompt compression.

## Table of Contents

1. [Why Manual Prompt Engineering Has Limits](#1-why-manual-prompt-engineering-has-limits)
2. [DSPy Framework](#2-dspy-framework)
3. [OPRO: Optimization by PROmpting](#3-opro-optimization-by-prompting)
4. [APE: Automatic Prompt Engineer](#4-ape-automatic-prompt-engineer)
5. [Automatic Prompt Generation](#5-automatic-prompt-generation)
6. [Gradient-Free Optimization for Prompts](#6-gradient-free-optimization-for-prompts)
7. [Bayesian Prompt Optimization](#7-bayesian-prompt-optimization)
8. [Prompt Compression](#8-prompt-compression)
9. [Cost-Quality Trade-offs](#9-cost-quality-trade-offs)
10. [When to Optimize vs When to Switch Models](#10-when-to-optimize-vs-when-to-switch-models)

---

## 1. Why Manual Prompt Engineering Has Limits

### 1.1 The Scalability Problem

Manual prompt engineering works through this loop:

```
Write prompt → Test on examples → Read outputs → Edit prompt → Repeat
```

This breaks down in several ways:

| Problem | Description |
|---------|-------------|
| **Evaluation bias** | Humans test on 5-10 examples; real workloads have thousands of edge cases |
| **Local optima** | Small edits explore a tiny neighborhood of prompt space |
| **Interaction effects** | Changing one instruction can break another; hard to track manually |
| **Multi-objective** | Optimizing for accuracy vs cost vs latency simultaneously is unintuitive |
| **Reproducibility** | "I tweaked the wording and it got better" is not a methodology |
| **Version explosion** | Dozens of prompts x multiple models x different use cases = unmanageable |

### 1.2 The Prompt Space is Vast

Consider a simple instruction prompt with 50 words. Even restricting to reasonable English paraphrases, there are thousands of semantically equivalent ways to express the same instruction. Each variant may produce different model behavior. Manual exploration covers a tiny fraction of this space.

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

### 1.3 When to Consider Automated Optimization

Automated prompt optimization is worth the investment when:

1. **High-volume production tasks**: Thousands of daily calls where a 2% accuracy improvement matters
2. **Cost-sensitive deployments**: Where reducing prompt length by 30% saves significant money
3. **Multi-prompt systems**: Where prompts interact (e.g., agent pipelines) and manual tuning of one prompt affects others
4. **Model migration**: When switching models and all prompts need re-tuning
5. **Measurable objectives**: When you have clear evaluation metrics (accuracy, F1, exact match, etc.)

Automated optimization is NOT worth it when:
- You have no evaluation dataset
- The task is creative/subjective (no clear "correct" answer)
- You are prototyping and the prompt will change fundamentally
- The prompt is used rarely (< 100 calls/day)

---

## 2. DSPy Framework

DSPy (Declarative Self-improving Language Programs) is the most mature framework for programmatic prompt optimization. Instead of writing prompts by hand, you declare *what* the LLM should do (via signatures) and let DSPy's optimizers figure out *how* to prompt it.

### 2.1 Core Concepts

```
┌─────────────┐    ┌──────────────┐    ┌──────────────┐
│  Signature   │───▶│   Module      │───▶│  Optimizer    │
│  (What)      │    │  (How)        │    │  (Search)     │
└─────────────┘    └──────────────┘    └──────────────┘
   Input/Output       Prompt strategy     Find best prompt
   declaration         (CoT, ReAct,        for your data
                       etc.)
```

- **Signature**: Declares input and output fields (e.g., `"question -> answer"`)
- **Module**: Wraps a signature with a prompting strategy (e.g., `dspy.ChainOfThought`)
- **Optimizer (Teleprompter)**: Searches for the best prompt/examples by evaluating on training data

### 2.2 Basic DSPy Program

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

### 2.3 Chain-of-Thought Module

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

### 2.4 Multi-Step Programs

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

### 2.5 Optimizing with DSPy

The real power of DSPy is its optimizers. Given training examples, an optimizer searches for the best prompt configuration:

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

### 2.6 Advanced Optimizers

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

### 2.7 Saving and Loading Optimized Programs

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

## 3. OPRO: Optimization by PROmpting

OPRO (Yang et al., 2023) uses the LLM itself as the optimizer. Instead of external search algorithms, OPRO asks the LLM to generate better prompts based on the performance of previous prompts.

### 3.1 OPRO Concept

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

### 3.2 OPRO Implementation

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

### 3.3 OPRO Insights

Key findings from the OPRO paper:

1. **LLMs can optimize prompts**: Given prompt-score pairs, LLMs generate improved prompts
2. **Instruction position matters**: OPRO found that placing instructions after examples (not before) often improved performance
3. **Optimization trajectory**: Performance improves rapidly in early iterations, then plateaus
4. **Temperature matters**: Higher temperature in the meta-prompt generates more diverse candidates

---

## 4. APE: Automatic Prompt Engineer

APE (Zhou et al., 2022) generates and selects prompts automatically. Unlike OPRO's iterative refinement, APE generates many candidates upfront and selects the best ones.

### 4.1 APE Concept

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

### 4.2 APE Implementation

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

### 4.3 APE with Iterative Refinement

Combine APE's generation with iterative improvement:

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

## 5. Automatic Prompt Generation

Beyond optimization, LLMs can generate prompts from scratch given task descriptions or examples.

### 5.1 Task Description to Prompt

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

### 5.2 Prompt Generation from Examples

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

## 6. Gradient-Free Optimization for Prompts

Since prompts are discrete text (not continuous vectors), we cannot use gradient-based optimization. Instead, we apply gradient-free optimization methods.

### 6.1 Random Search

The simplest approach: generate random prompt variants and keep the best one.

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

### 6.2 Evolutionary Search

Treat prompts as individuals in an evolutionary algorithm:

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

### 6.3 Comparison of Optimization Methods

| Method | Pros | Cons | Best For |
|--------|------|------|----------|
| Random Search | Simple, parallelizable | No learning from results | Quick baselines |
| OPRO | Leverages LLM intelligence | Expensive (many LLM calls) | Instruction optimization |
| APE | Good at initial generation | Less effective at refinement | Cold-start scenarios |
| Evolutionary | Systematic exploration | Slow convergence | Complex prompt spaces |
| DSPy Optimizers | Integrated framework | Learning curve | Production systems |
| Bayesian | Sample-efficient | Complex implementation | Expensive evaluation |

---

## 7. Bayesian Prompt Optimization

Bayesian optimization is particularly well-suited for prompt optimization because evaluations are expensive (each requires multiple LLM calls) and we want to minimize the number of evaluations.

### 7.1 Concept

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

### 7.2 Simplified Bayesian Prompt Optimization

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

## 8. Prompt Compression

Prompt compression reduces token count while preserving task performance. This saves cost and can improve latency.

### 8.1 Why Compress Prompts?

| Token Count | Cost Impact (at $3/1M tokens) | Latency Impact |
|-------------|-------------------------------|----------------|
| 1,000 tokens | $0.003/call | ~1s |
| 5,000 tokens | $0.015/call | ~3s |
| 10,000 tokens | $0.030/call | ~5s |
| 50,000 tokens | $0.150/call | ~15s |

At 10,000 calls/day, reducing from 5,000 to 2,000 tokens saves $90/day ($32,850/year).

### 8.2 Manual Compression Techniques

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

### 8.3 LLM-Based Compression (LLMLingua Approach)

LLMLingua and similar tools use a small model to identify and remove tokens that contribute least to task performance:

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

### 8.4 Measuring Compression Quality

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

## 9. Cost-Quality Trade-offs

Every prompt optimization decision involves trade-offs between cost, quality, and latency.

### 9.1 The Cost-Quality Frontier

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

### 9.2 Cost Optimization Strategies

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

### 9.3 Cascading Strategy

Use cheap models first, escalate to expensive models only when needed:

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

## 10. When to Optimize vs When to Switch Models

### 10.1 Decision Framework

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

### 10.2 Optimization vs Model Switching Comparison

| Approach | Effort | Cost | Typical Improvement |
|----------|--------|------|-------------------|
| Manual prompt editing | Low | Free | 5-15% |
| Few-shot example selection | Low | Slight token increase | 5-20% |
| DSPy BootstrapFewShot | Medium | Optimization LLM calls | 10-25% |
| DSPy MIPROv2 | Medium-High | Many optimization calls | 15-30% |
| OPRO | Medium | Many optimization calls | 10-20% |
| Model upgrade (Haiku to Sonnet) | Low | Higher per-call cost | 10-30% |
| Model upgrade (Sonnet to Opus) | Low | Much higher per-call cost | 5-20% |
| Fine-tuning | High | Training compute + data | 20-40% |

### 10.3 Practical Decision Rules

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

### 10.4 The Optimization Workflow

1. **Baseline**: Measure current performance with basic prompt
2. **Quick wins**: Try manual improvements (format, examples, constraints)
3. **Systematic search**: Apply DSPy or OPRO if quick wins plateau
4. **Model exploration**: Test with stronger/weaker models to understand limits
5. **Cost optimization**: Once quality target is met, compress prompts and try cheaper models
6. **Monitor**: Set up continuous evaluation to catch regressions

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

## Exercises

### Exercise 1: DSPy Signature Design

Design DSPy signatures and a module for a customer support ticket classifier. The system should:
1. Classify tickets into categories (billing, technical, account, general)
2. Assign priority (low, medium, high, urgent)
3. Determine if the ticket needs human escalation

Write the DSPy program (signatures + module) and describe how you would optimize it.

<details><summary>Show Answer</summary>

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

Key design decisions:
1. **Two signatures**: Separation of classification and escalation allows independent optimization.
2. **ChainOfThought for classification**: Reasoning helps with ambiguous tickets.
3. **Predict for escalation**: Simpler decision; CoT overhead not justified.
4. **Multi-criteria metric**: Weighted scoring reflects business priority (category accuracy > priority accuracy > escalation accuracy).
5. **BootstrapFewShotWithRandomSearch**: Good balance of optimization power and compute cost.

</details>

### Exercise 2: OPRO Implementation

Implement a simplified OPRO loop for optimizing a summarization prompt. The prompt should summarize news articles into 2-3 sentences. Define your evaluation metric and run at least 3 iterations of optimization.

<details><summary>Show Answer</summary>

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

The evaluation metric balances two objectives: fact coverage (are key numbers and names preserved?) and brevity (is the summary concise?). The 70/30 weighting reflects that accuracy matters more than length control for summarization.

</details>

### Exercise 3: Prompt Compression

Take the following verbose prompt and compress it to less than 50% of its original token count while maintaining performance. Describe your compression strategy and how you would verify that the compressed version maintains quality.

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

<details><summary>Show Answer</summary>

**Compression strategy:**
1. Remove role/persona description (does not improve output for capable models)
2. Remove "meta-instructions" ("be thorough", "carefully analyze", etc.)
3. Condense section descriptions to key requirements only
4. Keep the structural requirements (they affect output format)

**Compressed prompt (approximately 45% of original):**

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

**Verification approach:**

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

What was removed and why:
- **Role description** ("expert financial analyst..."): Claude performs well on financial analysis without persona priming; the specific instructions matter more than the role.
- **Filler phrases** ("carefully analyze", "comprehensive assessment", "please be thorough but concise"): These do not change behavior for instruction-following models.
- **Redundant explanations**: "Compare the reported revenue against analyst expectations and the same quarter last year" becomes "vs expectations and YoY" -- same information in 5 words vs 15.

</details>

### Exercise 4: Cost-Quality Analysis

You have a text classification task running 50,000 times per day. Currently using Claude Opus with a 500-token prompt achieving 97% accuracy. Design an experiment to find the cheapest model-prompt combination that maintains at least 95% accuracy. Write the experiment code.

<details><summary>Show Answer</summary>

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

The experiment tests 9 combinations (3 models x 3 prompts) and selects the cheapest one that meets the 95% accuracy threshold. At 50,000 daily calls, the cost difference between Opus+500tok and Haiku+50tok could be hundreds of dollars per day.

</details>

### Exercise 5: Optimization Pipeline

Design a complete prompt optimization pipeline that: (1) starts with a baseline prompt, (2) applies DSPy optimization, (3) compresses the result, (4) validates that compression did not degrade quality, and (5) outputs a production-ready prompt with documentation. Write the pipeline code.

<details><summary>Show Answer</summary>

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

        # Convert eval_set to DSPy format
        # For demonstration, we show the conceptual approach
        # In practice, you'd define proper DSPy signatures

        # Generate optimized prompt using OPRO-style approach
        best_prompt = baseline_prompt
        best_score = 0

        for iteration in range(5):
            # Ask LLM to improve the prompt
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

The pipeline follows a clear five-stage process with quality gates between stages. Each stage logs its results, enabling reproducibility and comparison. The compression validation stage (Stage 4) acts as a safety net, falling back to the uncompressed prompt if compression hurts quality. The final artifact is a documented JSON file ready for production deployment.

</details>

---

**Previous**: [RAG Prompt Patterns](./10_RAG_Prompt_Patterns.md) | **Next**: [Evaluation and Metrics](./12_Evaluation_and_Metrics.md)
