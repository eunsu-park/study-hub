# Emergent Abilities

## Learning Objectives
- Understand the definition and characteristics of Emergent Abilities
- Identify patterns of capability emergence with scale
- Learn major emergent abilities like Chain-of-Thought
- Master Capability Elicitation techniques

---

## 1. What are Emergent Abilities?

### 1.1 Definition

**Emergent Abilities** refer to capabilities that are absent in smaller models but **suddenly** appear beyond a certain scale.

```
┌─────────────────────────────────────────────────────────────────┐
│                    Characteristics of Emergence                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Performance                                                    │
│       │                                                         │
│   100%├─────────────────────────────────────●───── Large models│
│       │                                   ╱                     │
│       │                                 ╱                       │
│    50%├─ · · · · · · · · · · · · · · ╱· · · · · · · · · · · ·  │
│       │                            ╱                            │
│       │                          ╱  ← Phase Transition          │
│       │                        ╱                                │
│       │──────────────────────●─────────────────── Small models  │
│     0%├───────┬───────┬───────┬───────┬───────┬──────▶          │
│       │     10^21  10^22  10^23  10^24  10^25    Training FLOPs │
│                                                                 │
│  Key characteristics:                                           │
│  • Random guessing → Sudden performance improvement             │
│  • Almost no intermediate stages                                │
│  • Difficult to predict (not smooth)                            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 Emergence vs Gradual Improvement

```python
"""
Two patterns of performance improvement:

1. Gradual - Follows Scaling Law
   - Loss decreases slowly via power law
   - Predictable
   - Examples: Perplexity, general generation quality

2. Emergent - Sudden transition
   - Random until certain scale, then sharp improvement
   - Difficult to predict
   - Examples: Multi-digit arithmetic, Chain-of-Thought, code generation

Why does Emergence occur?
- Hypothesis 1: Tasks requiring sufficient capacity
- Hypothesis 2: Combination of multiple sub-skills needed
- Hypothesis 3: Metric issue (accuracy is threshold-based)
"""
```

---

## 2. Major Emergent Abilities

### 2.1 Capability Catalog

| Capability | Description | Emergence Scale (approx.) |
|------|------|-----------------|
| **Arithmetic** | Multi-digit addition/subtraction | ~10^22 FLOPs |
| **Word Unscrambling** | Restore scrambled letters | ~10^22 FLOPs |
| **Chain-of-Thought** | Step-by-step reasoning | ~10^23 FLOPs |
| **Multi-step Math** | Complex math problems | ~10^23 FLOPs |
| **Code Generation** | Complex code writing | ~10^23 FLOPs |
| **Translation** (low-resource) | Translating languages with limited training data | ~10^23 FLOPs |
| **Analogical Reasoning** | Analogy reasoning | ~10^24 FLOPs |
| **Theory of Mind** | Inferring others' beliefs/intentions | ~10^24 FLOPs |

### 2.2 BIG-bench Task Analysis

```
┌─────────────────────────────────────────────────────────────────┐
│              Emergent Tasks Observed in BIG-bench                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  □ = Linear improvement (Scaling Law)                           │
│  ● = Emergent (Phase Transition)                                │
│                                                                 │
│  10^21 ─┬─ □ Basic grammar                                      │
│         │   □ Simple QA                                          │
│  10^22 ─┼─ □ Summarization                                      │
│         │   ● 3-digit addition                                   │
│         │   ● Word unscrambling                                  │
│  10^23 ─┼─ □ Translation (general)                              │
│         │   ● Chain-of-Thought                                   │
│         │   ● Multi-step math                                    │
│         │   ● Code generation                                    │
│  10^24 ─┼─ □ Creative writing                                   │
│         │   ● Analogical reasoning                               │
│         │   ● Theory of Mind                                     │
│  10^25 ─┴─ ● Complex logical reasoning                          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. Chain-of-Thought (CoT)

### 3.1 Discovery of CoT

Chain-of-Thought was systematically studied in Wei et al.'s 2022 Google paper.

```
┌─────────────────────────────────────────────────────────────────┐
│              Chain-of-Thought Prompting Comparison               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Standard Prompting:                                            │
│  ────────────────────────────────────────────────               │
│  Q: Roger has 5 tennis balls. He buys 2 cans of balls.          │
│     Each can has 3 balls. How many balls does he have?          │
│  A: 11                                                          │
│                                                                 │
│  → Small models: Often wrong (e.g., "8", "6")                   │
│  → Even large models fail on complex problems                   │
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
│  → Explicitly generate intermediate reasoning steps             │
│  → Significantly improved accuracy on complex problems          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 CoT Implementation

```python
def standard_prompt(question):
    """Standard prompting - request answer only"""
    return f"""
Answer the following question:

Q: {question}
A:"""

def cot_prompt(question):
    """Chain-of-Thought prompting - elicit reasoning process"""
    return f"""
Answer the following question step by step.
Show your reasoning before giving the final answer.

Q: {question}
A: Let's think step by step."""

def few_shot_cot_prompt(question, examples):
    """Few-shot CoT - with examples"""
    prompt = "Solve the following problems step by step:\n\n"

    for ex in examples:
        prompt += f"Q: {ex['question']}\n"
        prompt += f"A: {ex['reasoning']}\n"
        prompt += f"   The answer is {ex['answer']}.\n\n"

    prompt += f"Q: {question}\n"
    prompt += "A: Let's think step by step."

    return prompt

# Example usage
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

### 3.3 Why CoT is Effective

```
┌─────────────────────────────────────────────────────────────────┐
│                 Hypotheses for Chain-of-Thought Mechanism        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Hypothesis 1: Working Memory Extension                         │
│  ─────────────────────────────────                              │
│  • Store intermediate results as text                           │
│  • Bypass Transformer's limited context                         │
│  • Use as "external memory"                                     │
│                                                                 │
│  Hypothesis 2: Problem Decomposition                            │
│  ─────────────────────────────────                              │
│  • Break complex problems into small steps                      │
│  • Each step is something the model can already do              │
│  • Solve complex problems through combination of steps          │
│                                                                 │
│  Hypothesis 3: Distribution Shift                               │
│  ─────────────────────────────────                              │
│  • Training data contains reasoning processes                   │
│  • "step by step" activates that distribution                   │
│  • Reuse learned patterns                                       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 4. Variants of CoT

### 4.1 Zero-shot CoT

```python
def zero_shot_cot(question):
    """
    Zero-shot CoT: Just add "Let's think step by step"

    Kojima et al. (2022) discovery:
    - Effective without examples
    - Applicable to various reasoning tasks
    """
    return f"""
Q: {question}
A: Let's think step by step."""

# Simple but effective!
question = "A juggler can juggle 16 balls. Half are golf balls. Half of the golf balls are blue. How many blue golf balls?"
# Response: "16 balls total. Half are golf balls: 16/2 = 8. Half of golf balls are blue: 8/2 = 4. The answer is 4."
```

### 4.2 Self-Consistency

```python
def self_consistency(question, model, n_samples=5, temperature=0.7):
    """
    Self-Consistency: Generate multiple reasoning paths and vote

    Wang et al. (2022):
    - Generate multiple CoTs for same problem
    - Vote on final answer
    - Improved accuracy over CoT alone
    """
    prompt = cot_prompt(question)
    answers = []

    for _ in range(n_samples):
        response = model.generate(prompt, temperature=temperature)
        # Extract final answer (e.g., "The answer is X" pattern)
        answer = extract_answer(response)
        answers.append(answer)

    # Majority vote
    from collections import Counter
    most_common = Counter(answers).most_common(1)[0][0]
    return most_common

# Example result:
# Sample 1: "... The answer is 4."
# Sample 2: "... The answer is 4."
# Sample 3: "... The answer is 8."  (error)
# Sample 4: "... The answer is 4."
# Sample 5: "... The answer is 4."
# Final: 4 (4/5 votes)
```

### 4.3 Tree of Thoughts (ToT)

```
┌─────────────────────────────────────────────────────────────────┐
│                    Tree of Thoughts                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  CoT: Single linear path                                        │
│       Start → Step1 → Step2 → Step3 → Answer                    │
│                                                                 │
│  ToT: Tree-shaped exploration                                   │
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
│  Features:                                                      │
│  • Explore multiple paths simultaneously                        │
│  • Evaluate and prune at each step                              │
│  • BFS/DFS search strategies                                    │
│  • Effective for complex planning/puzzle problems               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

```python
def tree_of_thoughts(problem, model, breadth=3, depth=3):
    """
    Tree of Thoughts implementation overview

    Yao et al. (2023):
    - Generate multiple "thought" candidates
    - Evaluate each thought
    - Expand only promising paths
    """
    def generate_thoughts(state, n=breadth):
        """Generate possible next thoughts from current state"""
        prompt = f"Given: {state}\nGenerate {n} possible next steps:"
        return model.generate(prompt).split('\n')[:n]

    def evaluate_thought(state, thought):
        """Evaluate promise of thought (0-1)"""
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

## 5. Capability Elicitation

### 5.1 Why Elicitation is Needed

```
┌─────────────────────────────────────────────────────────────────┐
│                    Need for Capability Elicitation               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Problem: Model "has" capability but doesn't demonstrate it     │
│                                                                 │
│  Example:                                                       │
│  ──────                                                         │
│  Q: What's 37 × 23?                                             │
│  A: 851 (correct)                                               │
│                                                                 │
│  Q: Calculate 37 times 23 without showing work.                 │
│  A: 852 (incorrect)                                             │
│                                                                 │
│  Same model, same problem, different results based on prompt!   │
│                                                                 │
│  Solution: Elicit latent capabilities with appropriate prompting│
│  • CoT: "Let's think step by step"                              │
│  • Role: "You are an expert mathematician"                      │
│  • Format: "Show your calculation"                              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 5.2 Elicitation Techniques

```python
"""
Main Capability Elicitation techniques:

1. Role Assignment
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
    """Prompt combining various elicitation techniques"""

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

# Usage example
question = "A train travels 60 km in the first hour and 80 km in the second hour. What is its average speed?"
prompt = enhanced_prompt(question)
```

### 5.3 Effect of Persona/Role

```python
# Experiment: Same problem, different roles

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

# Research findings:
# - Expert persona: Improved accuracy on complex problems
# - Teacher persona: Better explanation quality
# - Programmer persona: Structured approach

# Note: Effect varies by model size
# Small models: Minimal persona effect
# Large models: Significant differences occur
```

---

## 6. The Emergence Debate

### 6.1 "Emergence is a Mirage?" (2023)

```
┌─────────────────────────────────────────────────────────────────┐
│            Counter-argument on Emergence (Schaeffer et al. 2023) │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Claim: Emergence may be an artifact of metrics                 │
│                                                                 │
│  Arguments:                                                     │
│  ────────                                                       │
│  1. Accuracy is an "all-or-nothing" metric                      │
│     • Partial answer = 0 score                                  │
│     • There may have been gradual improvement in reality        │
│                                                                 │
│  2. When measured with continuous metrics:                      │
│     • Brier score, log-likelihood, etc.                         │
│     • "Sudden transition" disappears                            │
│     • Instead, smooth improvement observed                      │
│                                                                 │
│  3. Example: Multi-digit addition                               │
│     • Accuracy: 0% → 0% → 100% (emerge!)                        │
│     • Token-level acc: 40% → 60% → 100% (smooth)               │
│                                                                 │
│  Conclusion (controversial):                                    │
│  • "True emergence" may be a matter of metric choice            │
│  • However, what matters practically is task performance        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 6.2 Current Consensus

```python
"""
Current state of Emergence debate (2024):

Pro side:
- Some abilities clearly appear suddenly (practical perspective)
- In-context learning itself is emergent
- Complex reasoning abilities have thresholds

Con side:
- Metric choice creates "emergence" illusion
- With sufficiently fine-grained metrics, it's smooth
- Can be explained by "predictable" scaling

Practical consensus:
- Whether "emergence" or not, useful abilities manifest beyond certain scales
- Capability prediction remains difficult
- Capability elicitation is important
"""
```

---

## 7. Practice: Observing Emergence

### 7.1 Comparing Capabilities by Model Size

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def compare_model_capabilities(question, model_names):
    """
    Test same problem across multiple model sizes
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

# Models to test (by size)
models = [
    "microsoft/phi-2",           # 2.7B
    "meta-llama/Llama-2-7b-hf",  # 7B
    "meta-llama/Llama-2-13b-hf", # 13B
    "meta-llama/Llama-2-70b-hf", # 70B
]

# Emergence test questions
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

# Compare results (requires significant memory to run)
# for q_name, question in test_questions.items():
#     print(f"\n=== {q_name} ===")
#     results = compare_model_capabilities(question, models)
#     for model, response in results.items():
#         print(f"\n{model}:\n{response}")
```

### 7.2 Measuring CoT Effect

```python
def measure_cot_effect(questions, model, tokenizer):
    """
    Compare Standard vs CoT prompting effects
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

    # Calculate accuracy
    std_acc = sum(results["standard"]) / len(questions)
    cot_acc = sum(results["cot"]) / len(questions)

    print(f"Standard Prompting Accuracy: {std_acc:.1%}")
    print(f"Chain-of-Thought Accuracy: {cot_acc:.1%}")
    print(f"Improvement: {cot_acc - std_acc:.1%}")

    return results

# Test dataset (GSM8K style)
test_questions = [
    {"question": "Janet has 10 apples. She gives 3 to her friend. How many does she have?", "answer": "7"},
    {"question": "A store has 24 shirts. If 6 are sold each day, how many days until they're gone?", "answer": "4"},
    # ... more problems
]
```

---

## Summary

### Key Concepts
- **Emergent Abilities**: Capabilities that suddenly appear at scale
- **Chain-of-Thought**: Solve complex problems through step-by-step reasoning
- **Self-Consistency**: Improve CoT accuracy through majority voting
- **Capability Elicitation**: Elicit latent capabilities through prompting

### Practical Applications
1. Complex reasoning → Use CoT
2. High accuracy needed → Self-consistency
3. Creative exploration → Tree of Thoughts
4. Maximize capabilities → Set Role/Persona

### Next Steps
- [08_LLaMA_Family.md](08_LLaMA_Family.md): State-of-the-art LLM architectures
- [19_PEFT_Unified.md](19_PEFT_Unified.md): Efficient adaptation techniques

---

## References

### Key Papers
- Wei et al. (2022). "Emergent Abilities of Large Language Models"
- Wei et al. (2022). "Chain-of-Thought Prompting Elicits Reasoning in LLMs"
- Wang et al. (2022). "Self-Consistency Improves Chain of Thought Reasoning"
- Yao et al. (2023). "Tree of Thoughts: Deliberate Problem Solving with LLMs"
- Schaeffer et al. (2023). "Are Emergent Abilities of LLMs a Mirage?"

### Additional Resources
- [BIG-bench](https://github.com/google/BIG-bench)
- [Chain-of-Thought Hub](https://github.com/FranxYao/chain-of-thought-hub)

---

## Exercises

### Exercise 1: Metric Choice and the Emergence Mirage

Schaeffer et al. (2023) argue that emergent abilities may be an artifact of metric choice. Consider a model being evaluated on multi-digit addition across increasing compute scales.

1. Explain why using **accuracy** (exact match) as the metric would produce a "phase transition" appearance.
2. Explain why using **token-level accuracy** (per-digit correctness) would produce a smooth curve instead.
3. Which metric should researchers prefer, and why might each metric be appropriate in different contexts?

<details>
<summary>Show Answer</summary>

**Part 1: Accuracy creates phase transitions**

Accuracy (exact match) is a threshold metric: the full answer is either correct (1) or wrong (0). Even if the model improves from getting 2 out of 5 digits right to 4 out of 5, the accuracy score stays 0. Only when the model crosses the threshold of getting all digits correct does accuracy jump to 1. This masking of gradual improvement makes it look like a sudden "emergence."

**Part 2: Token-level accuracy is smooth**

Token-level accuracy credits partial progress: correctly predicting 3 of 5 digits scores 0.6. As the model scales and gets progressively more digits right, this metric rises smoothly, revealing the underlying gradual improvement.

**Part 3: Which to prefer**

- **Accuracy** is appropriate when the end-task requires complete correctness (e.g., a calculator that gives a wrong digit is useless). It correctly reflects whether a capability is practically usable.
- **Token-level accuracy** is better for understanding the model's internal learning trajectory and predicting when full accuracy will emerge.
- For **research on emergence mechanisms**, continuous metrics are preferred. For **deployment decisions**, task-level accuracy is more meaningful.

Neither metric is "wrong" — they answer different questions.

</details>

---

### Exercise 2: Implementing Self-Consistency

Complete the `self_consistency` function below. The function should generate `n_samples` responses to a math question using Chain-of-Thought prompting, extract the numeric final answer from each response, and return the most common answer.

```python
import re
from collections import Counter

def extract_numeric_answer(response: str) -> str:
    """Extract the last number mentioned in the response."""
    numbers = re.findall(r'\b\d+(?:\.\d+)?\b', response)
    return numbers[-1] if numbers else "unknown"

def self_consistency(question: str, model, n_samples: int = 5, temperature: float = 0.7) -> str:
    """
    TODO: Implement self-consistency decoding.
    1. Build a CoT prompt for the question
    2. Generate n_samples responses from the model
    3. Extract the numeric answer from each response
    4. Return the majority vote answer
    """
    # Your implementation here
    pass
```

<details>
<summary>Show Answer</summary>

```python
import re
from collections import Counter

def extract_numeric_answer(response: str) -> str:
    """Extract the last number mentioned in the response."""
    numbers = re.findall(r'\b\d+(?:\.\d+)?\b', response)
    return numbers[-1] if numbers else "unknown"

def self_consistency(question: str, model, n_samples: int = 5, temperature: float = 0.7) -> str:
    """
    Self-consistency decoding:
    generate multiple CoT paths and take majority vote.
    """
    # Step 1: Build CoT prompt
    cot_prompt = f"Q: {question}\nA: Let's think step by step."

    # Step 2: Generate multiple responses
    answers = []
    for _ in range(n_samples):
        response = model.generate(cot_prompt, temperature=temperature)
        answer = extract_numeric_answer(response)
        answers.append(answer)

    # Step 3: Majority vote
    counter = Counter(answers)
    majority_answer = counter.most_common(1)[0][0]

    return majority_answer

# Why this works:
# - Incorrect reasoning paths tend to produce diverse wrong answers
# - Correct reasoning paths tend to converge on the same right answer
# - Majority vote filters out idiosyncratic errors
```

</details>

---

### Exercise 3: Capability Elicitation Experiment Design

Design an experiment to test whether **role assignment** (persona prompting) improves a model's accuracy on logic puzzles. Specify:

1. The independent variable (what you manipulate)
2. The dependent variable (what you measure)
3. A control condition and at least two treatment conditions
4. A potential confound you need to control for

<details>
<summary>Show Answer</summary>

**Experimental Design:**

**1. Independent variable:** The role/persona assigned to the model in the system prompt.

**2. Dependent variable:** Accuracy (% correct) on a standardized set of logic puzzles (e.g., 50 problems from a published logic benchmark).

**3. Conditions:**
- **Control:** No persona — plain prompt: `"Solve the following logic puzzle: {puzzle}"`
- **Treatment A (Expert):** `"You are a world-class logician with 30 years of experience. Solve the following logic puzzle: {puzzle}"`
- **Treatment B (Teacher):** `"You are a patient teacher explaining logical reasoning. Solve the following logic puzzle step by step: {puzzle}"`

**4. Potential confound:**
- **Temperature and sampling:** Different prompts might interact with sampling randomness differently. Control by using `temperature=0` (greedy decoding) for all conditions, or use the same temperature with multiple seeds and average.
- **Problem ordering effect:** Randomize puzzle order across conditions to prevent position bias.
- **CoT vs no-CoT:** The "teacher" prompt implicitly encourages step-by-step reasoning. To isolate persona effects from CoT effects, ensure all conditions either include or exclude "step by step" phrasing uniformly.

</details>

---

### Exercise 4: Tree of Thoughts vs Chain-of-Thought

Compare CoT (Chain-of-Thought) and ToT (Tree of Thoughts) on the following dimensions. For each cell, briefly explain the trade-off.

| Dimension | CoT | ToT |
|-----------|-----|-----|
| Search strategy | ? | ? |
| Computational cost | ? | ? |
| Suitable problem types | ? | ? |
| Failure mode | ? | ? |

<details>
<summary>Show Answer</summary>

| Dimension | CoT | ToT |
|-----------|-----|-----|
| **Search strategy** | Single linear path — greedy left-to-right generation. Fast but commits early to potentially wrong directions. | Tree-structured BFS/DFS — explores multiple branches and backtracks. More thorough but requires generating and evaluating many partial states. |
| **Computational cost** | Low — proportional to a single response length. Typically 1× cost. | High — proportional to branching factor × depth × evaluation cost. Can be 10-100× more expensive than CoT. |
| **Suitable problem types** | Sequential reasoning tasks where each step naturally follows from the last (arithmetic, straightforward QA). | Combinatorial or planning problems where early decisions strongly constrain later options and backtracking is valuable (puzzles, multi-step planning, code debugging). |
| **Failure mode** | **Cascading error** — one early wrong step propagates through the entire chain, producing a confident wrong answer. | **Evaluation bottleneck** — quality depends on the accuracy of the node evaluation function. A poor evaluator will mis-prune correct branches or over-invest in wrong ones. |

</details>
