# 09. Prompt Engineering

## Learning Objectives

- Effective prompt writing
- Zero-shot, Few-shot techniques
- Chain-of-Thought (CoT)
- Advanced prompting techniques

---

## Theory & Principles

A prompt is, to the model, *just text* — it has no special status, no privileged channel. Why then do small wording changes shift accuracy by 20+ points on hard tasks? Because an LLM's behavior is **conditional probability** `p(answer | prompt)`, and tiny shifts in the conditioning context can move the model into entirely different regions of its learned distribution. Prompt engineering is the empirical study of which conditioning patterns reliably steer the model toward correct, formatted, and safe completions.

This section covers:

- **(A) The conditioning-distribution view** — why prompt engineering works at all, and what we are actually doing when we write a prompt.
- **(B) Zero-shot vs few-shot** — the in-context-learning mechanism and when each is appropriate.
- **(C) Chain-of-Thought (CoT)** — why "let's think step by step" works and what it does to the model's computational graph.
- **(D) Self-consistency, ToT, ReAct** — sampling and search wrapped around prompts.
- **(E) Prompt structure and decomposition** — role, instruction, context, examples, format specification, the "anatomy" of a robust prompt.
- **(F) Failure modes** — instruction drift, prompt injection vulnerability, format brittleness.

### A. The Conditioning-Distribution View

An LLM models `p(next_token | full_context)`. The prompt sets the context that conditions every prediction that follows. Two prompts that look semantically similar to a human can correspond to wildly different distributions:

- "Solve this math problem: 23 × 47 =" — the model is conditioned on what follows "Solve this math problem:" in its training data: textbook explanations, often correct.
- "23 × 47 =" — the model is conditioned on what follows raw arithmetic; this often appears in noisy contexts (forum posts with typos), and accuracy drops.

The model has no "intention" to be correct. It has a learned distribution over how text continues. Prompt engineering is the practice of **steering the conditioning** so that the high-probability completions are also the desired completions.

This view explains many empirical surprises:
- "You are an expert mathematician" boosts accuracy because expert-written content in pre-training is more correct.
- Few-shot examples set a pattern that the model continues mechanically.
- "Let's think step by step" elicits chain-of-reasoning text that pre-training data associates with correct answers.

### B. Zero-Shot vs Few-Shot

**B.1 Zero-shot.** Just the task description and the input:

```
Translate to French: "Hello, how are you?"
Answer:
```

Works for tasks that are well-represented in pre-training and have a canonical format. Fails for novel tasks or unusual formats.

**B.2 Few-shot.** Pre-pend `k` example input-output pairs:

```
English: "Hello" → French: "Bonjour"
English: "Thank you" → French: "Merci"
English: "Goodbye" → French: "
```

The model pattern-matches the format and continues. Examples teach both the *task* (what to do) and the *format* (how to output). Standard `k = 3-8`.

**B.3 The induction-head mechanism (mechanistic interpretation).** Olsson et al. (2022) identified specific attention heads in trained Transformers that perform "find a previous occurrence of `[A]` in the prefix, look at what came after it, and predict the same thing now." Few-shot prompts are pure fuel for these heads: each example creates a `[input pattern] → [output pattern]` association that the induction head can copy.

**B.4 When to choose.** Zero-shot when: task is generic, model is large (10B+), tokens are expensive. Few-shot when: task has unusual format, examples are short, you have ground-truth examples.

### C. Chain-of-Thought (CoT)

Wei et al. (2022) showed that prompting an LM to produce intermediate reasoning steps before its final answer dramatically improves accuracy on multi-step problems:

```
Q: Roger has 5 tennis balls. He buys 2 more cans of tennis balls. Each can has 3 balls. How many balls does he have?
A: Roger started with 5 balls. 2 cans of 3 balls each is 6 balls. 5 + 6 = 11. The answer is 11.
```

**Why it works.** The Transformer has fixed depth — `L` layers, each performing one round of attention + FFN. With `L` layers, the model can compute at most `L` sequential operations *internally* per token. A multi-step math problem may need 5+ sequential operations, exceeding the model's internal depth budget for a single token.

CoT externalizes the computation across multiple tokens. Each intermediate step is a new token, generated from the current state. The full reasoning unfolds across `T` generated tokens with effectively `L · T` sequential operations. The model "thinks out loud" by *spending tokens* on intermediate state.

This is also why CoT only emerges at scale (~60B+ in early experiments): smaller models cannot reliably generate coherent intermediate steps even when prompted to.

**Zero-shot CoT** (Kojima et al., 2022): just append "Let's think step by step." to the prompt. Works without any examples, on most large LLMs.

### D. Self-Consistency, Tree-of-Thoughts, ReAct

Three families of prompting techniques that wrap CoT or LLM calls in additional structure:

**D.1 Self-consistency** (Wang et al., 2022). Sample `k` independent CoT trajectories at temperature `T > 0`. Take the **majority vote** over final answers. The intuition: many different reasoning paths can reach the same correct answer, but each wrong path is wrong in its own way. Voting marginalizes out reasoning noise. Often gains 5-10 points over single-trajectory CoT.

**D.2 Tree-of-Thoughts (ToT)** (Yao et al., 2023). Treat reasoning as tree search: at each step, generate `b` candidate next-thoughts, score each with the LLM ("is this on track?"), expand only the top `k`. Standard search algorithms (BFS, DFS, beam) over the tree of possible reasoning steps. Useful when the problem has clear partial-progress signals.

**D.3 ReAct** (Yao et al., 2022). Interleave reasoning steps ("Thought:") with tool actions ("Action:") and observations ("Observation:"). Each tool result becomes new context for the next thought. Foundation of modern agent frameworks (covered in lessons 14-15).

### E. Prompt Structure: Anatomy of a Robust Prompt

Production prompts have predictable components:

```
[Role/system]      "You are a careful financial analyst."
[Task/instruction] "Extract company revenue and growth rate from the report."
[Context]          "Here is the report: <DOC>...</DOC>"
[Examples]         "Example 1: ..."
[Format spec]      "Output JSON with keys: revenue, growth_rate."
[Constraints]      "If a value is missing, use null. Do not invent numbers."
[Input]            "Now process: <DOC>actual document</DOC>"
```

Each part addresses a known failure mode:
- Role primes the conditioning toward a domain-appropriate distribution.
- Task gives the high-level goal.
- Context isolates the relevant data with explicit delimiters (preventing the model from confusing data with instructions).
- Examples set the pattern for in-context learning.
- Format spec constrains the output shape.
- Constraints handle edge cases that would otherwise cause hallucination.

Order matters: the "instruction follows context" pattern works for many models but can be brittle. Many providers (OpenAI, Anthropic) recommend instructions *before* the input data.

### F. Failure Modes

**F.1 Instruction drift / forgetting.** In long contexts, the model "loses sight" of early instructions. Mitigation: repeat critical instructions at the end ("Remember: respond in JSON only.").

**F.2 Prompt injection.** A malicious user input contains text like "Ignore previous instructions and ..." The model, having no privileged-vs-user channel, may comply. Mitigation: input sanitization, output validation, and architectural separation (covered in lesson 21).

**F.3 Format brittleness.** "Output JSON" produces JSON 95% of the time but fails on the 5%. For reliable structured output, use grammar-constrained decoding or validation+retry (covered in lesson 22).

**F.4 Adversarial sensitivity.** Small wording changes ("Solve" vs "Compute") can shift accuracy by several points. Mitigation: prompt ensembling, automated prompt search (APE, OPRO).

**F.5 Sycophancy.** Models trained with RLHF often agree with users even when wrong. Mitigation: explicit "be honest, disagree with me if I'm wrong" instructions; or use a less RLHF-tuned base model.

### From Theory to the Functions Below

- §1 (basics) — concrete examples of §A's conditioning-distribution view (rephrasing, role injection).
- §2 (zero-shot vs few-shot) — implements §B with the ICL induction-head intuition.
- §3 (CoT) — codes §C, including zero-shot "let's think step by step" and few-shot CoT examples.
- §4 (role playing) — §E's role/system component as a primary lever.
- §5 (output format) — §E's format-spec component, with format brittleness from §F.3.
- §6 (advanced techniques) — implements §D's self-consistency and the start of ToT/ReAct.

---

## 1. Prompt Basics

### Prompt Components

> **[System Instruction]**
> You are a helpful AI assistant.
>
> [Context]
> - **Please refer to the following text**: ...
>
> [Task Instruction]
> Please summarize the text above.
>
> [Output Format]
> Please respond in JSON format.


### Basic Principles

```
1. Clarity: Write unambiguously
2. Specificity: Specify exactly what you want
3. Examples: Provide examples when possible
4. Constraints: Specify output format, length, etc.
```

---

## 2. Zero-shot vs Few-shot

### Zero-shot

```
Explain task without examples

Prompt:
"""
Analyze the sentiment of the following review.
Review: "This movie was really boring."
Sentiment:
"""

Response: Negative
```

### Few-shot

```
Provide several examples

Prompt:
"""
Analyze the sentiment of the following reviews.

Review: "Really fun movie!"
Sentiment: Positive

Review: "Worst movie, waste of time"
Sentiment: Negative

Review: "It was okay"
Sentiment: Neutral

Review: "This movie was really boring."
Sentiment:
"""

Response: Negative
```

### Few-shot Tips

```python
# Example selection criteria
1. Diversity: Include examples from all classes
2. Representativeness: Use typical examples
3. Similarity: Examples similar to actual input
4. Relevance: Highly relevant examples

# Number of examples
- Generally 3-5
- Complex tasks: 5-10
- Consider token limits
```

---

## 3. Chain-of-Thought (CoT)

### Basic CoT

```
Guide step-by-step reasoning

Prompt:
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
Simply guide reasoning

Prompt:
"""
Q: 5 + 7 × 3 = ?

Let's think step by step.
"""

Response:
1. First, we need to follow order of operations (PEMDAS).
2. Multiplication comes before addition.
3. 7 × 3 = 21
4. 5 + 21 = 26
The answer is 26.
```

### Self-Consistency

```python
# Generate multiple reasoning paths and take majority vote

responses = []
for _ in range(5):
    response = model.generate(prompt, temperature=0.7)
    responses.append(extract_answer(response))

# Select most common answer
final_answer = max(set(responses), key=responses.count)
```

---

## 4. Role Playing

### Expert Role

```
System prompt:
"""
You are a Python developer with 10 years of experience.
When reviewing code, you check for:
- Code readability
- Potential bugs
- Performance optimization
- Security vulnerabilities
"""

User:
"""
Please review the following code:
def get_user(id):
    return db.execute(f"SELECT * FROM users WHERE id = {id}")
"""
```

### Persona

```
"""
You are a kind and patient elementary school teacher.
You explain complex concepts using easy analogies.
You always use an encouraging tone.

Question: What is gravity?
"""
```

---

## 5. Specifying Output Format

### JSON Output

```
Prompt:
"""
Extract persons and locations from the following text.

Text: "Cheolsu met Younghee in Seoul."

Respond in JSON format:
{
  "persons": [...],
  "locations": [...]
}
"""
```

### Structured Output

```
Prompt:
"""
Analyze the following article.

## Summary
(2-3 sentences)

## Key Points
- Point 1
- Point 2

## Sentiment
(Positive/Negative/Neutral)
"""
```

### XML Tags

```
Prompt:
"""
Translate and explain the following text.

<text>Hello, how are you?</text>

<translation>Translation result</translation>
<explanation>Translation explanation</explanation>
"""
```

---

## 6. Advanced Techniques

### Self-Ask

```
Model asks and answers its own questions

"""
Question: Where is President Biden's hometown?

Follow-up needed: Yes
Follow-up question: Who is President Biden?
Intermediate answer: Joe Biden is the 46th president of the United States.

Follow-up needed: Yes
Follow-up question: Where was Joe Biden born?
Intermediate answer: He was born in Scranton, Pennsylvania.

Follow-up needed: No
Final answer: President Biden's hometown is Scranton, Pennsylvania.
"""
```

### ReAct (Reason + Act)

```
Alternate between reasoning and actions

"""
Question: Who won the 2023 Nobel Prize in Physics?

Thought: I need to find who won the 2023 Nobel Prize in Physics.
Action: Search[2023 Nobel Prize in Physics]
Observation: Pierre Agostini, Ferenc Krausz, and Anne L'Huillier won.

Thought: I have confirmed the search results.
Action: Finish[Pierre Agostini, Ferenc Krausz, Anne L'Huillier]
"""
```

### Tree of Thoughts

```python
# Explore multiple thought paths as a tree

def tree_of_thoughts(problem, depth=3, branches=3):
    thoughts = []

    for _ in range(branches):
        # Generate first thought
        thought = generate_thought(problem)
        score = evaluate_thought(thought)
        thoughts.append((thought, score))

    # Select top thoughts
    best_thoughts = sorted(thoughts, key=lambda x: x[1], reverse=True)[:2]

    # Recursively expand
    for thought, _ in best_thoughts:
        if depth > 0:
            extended = tree_of_thoughts(thought, depth-1, branches)
            thoughts.extend(extended)

    return thoughts
```

---

## 7. Prompt Optimization

### Iterative Improvement

```python
# 1. Start with basic prompt
prompt_v1 = "Summarize this text: {text}"

# 2. Improve after analyzing results
prompt_v2 = """
Summarize the following text in 2-3 sentences.
Focus on the main points.
Text: {text}
Summary:
"""

# 3. Add examples
prompt_v3 = """
Summarize the following text in 2-3 sentences.

Example:
Text: [Long article]
Summary: [Brief summary]

Text: {text}
Summary:
"""
```

### A/B Testing

```python
import random

def ab_test_prompts(test_cases, prompt_a, prompt_b):
    results = {'A': 0, 'B': 0}

    for case in test_cases:
        response_a = model.generate(prompt_a.format(**case))
        response_b = model.generate(prompt_b.format(**case))

        # Evaluation (automatic or manual)
        score_a = evaluate(response_a, case['expected'])
        score_b = evaluate(response_b, case['expected'])

        if score_a > score_b:
            results['A'] += 1
        else:
            results['B'] += 1

    return results
```

---

## 8. Prompt Templates

### Classification

```python
CLASSIFICATION_PROMPT = """
Classify the following text into one of these categories: {categories}

Text: {text}

Category:"""
```

### Summarization

```python
SUMMARIZATION_PROMPT = """
Summarize the following text in {num_sentences} sentences.
Focus on the key points and main arguments.

Text:
{text}

Summary:"""
```

### Question Answering

```python
QA_PROMPT = """
Answer the question based on the context below.
If the answer cannot be found, say "I don't know."

Context: {context}

Question: {question}

Answer:"""
```

### Code Generation

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

## 9. Managing Prompts in Python

### Template Class

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

# Usage
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

### LangChain Prompts

```python
from langchain.prompts import PromptTemplate, FewShotPromptTemplate

# Basic template
prompt = PromptTemplate(
    input_variables=["text"],
    template="Summarize: {text}"
)

# Few-shot template
examples = [
    {"input": "Long text 1", "output": "Summary 1"},
    {"input": "Long text 2", "output": "Summary 2"},
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

## Summary

### Prompt Checklist

```
□ Provide clear instructions
□ Include examples if needed (Few-shot)
□ Specify output format
□ Set role/persona
□ Guide step-by-step reasoning (if needed)
□ Specify constraints
```

### Technique Selection Guide

| Situation | Recommended Technique |
|-----------|----------------------|
| Simple task | Zero-shot |
| Specific format needed | Few-shot + format specification |
| Reasoning required | Chain-of-Thought |
| Complex reasoning | Tree of Thoughts |
| Tool usage needed | ReAct |

---

## Exercises

### Exercise 1: Zero-shot vs Few-shot Comparison

You need to classify product reviews into three categories: Positive, Negative, or Neutral. Write both a zero-shot and a few-shot prompt for this task, then explain when you would prefer each approach.

<details>
<summary>Show Answer</summary>

**Zero-shot prompt:**
```
Classify the sentiment of the following product review as Positive, Negative, or Neutral.

Review: "{review}"

Sentiment:
```

**Few-shot prompt:**
```
Classify the sentiment of the following product reviews.

Review: "Absolutely love this! Best purchase I've made all year."
Sentiment: Positive

Review: "Terrible quality. Broke after two days of use."
Sentiment: Negative

Review: "It does what it says. Nothing special."
Sentiment: Neutral

Review: "{review}"
Sentiment:
```

**When to prefer each:**
- **Zero-shot**: Use when the task is straightforward and the model's training already covers it well. Faster to write and uses fewer tokens.
- **Few-shot**: Use when you need a specific output format, the task is ambiguous, or you want consistent style. Especially valuable for rare label sets or domain-specific tasks where the model may not have strong priors.

A good rule of thumb: start with zero-shot, switch to few-shot if results are inconsistent.
</details>

---

### Exercise 2: Chain-of-Thought Prompt Design

A user wants to calculate whether they can afford a vacation. They earn $4,200/month after taxes, their fixed monthly expenses are $2,800, and the vacation costs $1,500. The trip is 3 months away. Write a zero-shot CoT prompt that guides the model to reason through this correctly.

<details>
<summary>Show Answer</summary>

```
A user wants to know if they can save enough for a vacation.
- Monthly income (after tax): $4,200
- Fixed monthly expenses: $2,800
- Vacation cost: $1,500
- Months until vacation: 3

Can they afford it? Let's think step by step.
```

**Expected model reasoning:**
```
1. Monthly savings = Income - Expenses = $4,200 - $2,800 = $1,400
2. Total savings in 3 months = $1,400 × 3 = $4,200
3. Vacation cost = $1,500
4. $4,200 > $1,500, so yes, they can afford it.
   They'll have $4,200 - $1,500 = $2,700 left over.

Answer: Yes, they can afford the vacation.
```

**Why CoT helps here:** Without step-by-step reasoning, the model might jump to an answer and make arithmetic errors. The phrase "Let's think step by step" elicits structured reasoning that reduces errors on multi-step math problems.
</details>

---

### Exercise 3: Structured Output Extraction

Write a prompt that extracts structured information from a job posting. The output should be valid JSON with fields: `title`, `company`, `location`, `salary_range`, `required_skills` (list), and `experience_years` (number). Handle the case where a field is not mentioned in the posting.

<details>
<summary>Show Answer</summary>

```python
EXTRACTION_PROMPT = """
Extract job information from the posting below and return it as JSON.
Use null for any fields not mentioned.

Required JSON structure:
{{
  "title": "job title",
  "company": "company name",
  "location": "city/remote",
  "salary_range": "e.g. $80,000-$100,000 or null",
  "required_skills": ["skill1", "skill2"],
  "experience_years": 3
}}

Job posting:
{posting}

JSON:
"""

# Example usage
posting = """
Senior ML Engineer at DataCorp
San Francisco, CA (hybrid)
We're looking for someone with 5+ years of experience in Python and PyTorch.
Knowledge of distributed training and MLflow is a plus.
"""

# Expected output:
# {
#   "title": "Senior ML Engineer",
#   "company": "DataCorp",
#   "location": "San Francisco, CA (hybrid)",
#   "salary_range": null,
#   "required_skills": ["Python", "PyTorch"],
#   "experience_years": 5
# }
```

**Key design decisions:**
- Use double braces `{{` `}}` to escape literal braces in Python f-strings/`.format()` templates
- Providing the exact schema reduces hallucinated fields
- `null` instruction prevents the model from inventing data for missing fields
- Adding "JSON:" at the end primes the model to output JSON directly
</details>

---

### Exercise 4: Self-Consistency Implementation

Implement the self-consistency prompting technique that generates multiple reasoning paths and returns the most common answer. The function should work for any yes/no or short-answer question.

<details>
<summary>Show Answer</summary>

```python
from collections import Counter

def self_consistency(
    prompt: str,
    model,
    n_samples: int = 5,
    temperature: float = 0.7
) -> tuple[str, dict]:
    """
    Generate multiple reasoning paths and return the majority answer.

    Args:
        prompt: The question prompt (should include CoT instruction)
        model: A callable model.generate(prompt, temperature) -> str
        n_samples: Number of independent samples to generate
        temperature: Sampling temperature (>0 for diversity)

    Returns:
        (final_answer, vote_counts)
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
    """Extract the answer from a CoT response."""
    # Look for "The answer is X" pattern
    import re
    match = re.search(r"(?:answer is|therefore)[:\s]+(.+?)(?:\.|$)",
                      response, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    # Fall back to last non-empty line
    lines = [l.strip() for l in response.strip().split('\n') if l.strip()]
    return lines[-1] if lines else response.strip()


# Usage example
cot_prompt = """
Q: If a train travels 120 km in 1.5 hours, and then 80 km in 1 hour,
   what is its average speed for the entire journey?

Let's think step by step.
"""

# With self-consistency (temperature > 0 generates diverse paths)
# All correct paths converge: (120+80) / (1.5+1) = 200/2.5 = 80 km/h
```

**Why this works:** Temperature > 0 causes the model to explore different reasoning paths. Incorrect paths tend to disagree with each other, while correct reasoning paths consistently converge on the same answer. Majority voting filters out individual errors.
</details>

---

### Exercise 5: Prompt Template Evaluation

The following prompt template for a code review task has several weaknesses. Identify at least four problems and write an improved version.

```python
# Original (weak) prompt
REVIEW_PROMPT = "Review this code: {code}"
```

<details>
<summary>Show Answer</summary>

**Problems with the original prompt:**
1. **No role/expertise context** — the model doesn't know what kind of reviewer to be
2. **No output structure** — response format is unpredictable (bullet list? paragraph? score?)
3. **No review criteria** — security? performance? style? correctness? all of them?
4. **No language specification** — different languages have different best practices
5. **No severity guidance** — all issues treated equally; critical bugs not distinguished from nits

**Improved version:**
```python
CODE_REVIEW_PROMPT = """
You are a senior software engineer conducting a thorough code review.

Review the following {language} code and provide feedback organized by severity.

Code to review:
```{language}
{code}
```

Provide your review in this exact format:

## Critical Issues (must fix before merging)
- [issue]: [explanation and suggested fix]

## Warnings (should fix)
- [issue]: [explanation and suggested fix]

## Suggestions (nice to have)
- [issue]: [explanation and suggested fix]

## Summary
[2-3 sentence overall assessment including what the code does well]
"""

# Usage
review = CODE_REVIEW_PROMPT.format(
    language="python",
    code="""
def get_user(id):
    return db.execute(f"SELECT * FROM users WHERE id = {id}")
"""
)
# Expected: Critical issue flagged for SQL injection vulnerability
```

The improved prompt enforces consistent structure, focuses the model's expertise, and separates critical issues from minor suggestions — making the output actionable for developers.
</details>

---

## Next Steps

Learn about Retrieval-Augmented Generation (RAG) systems in [10_RAG_Fundamentals.md](./10_RAG_Fundamentals.md).
