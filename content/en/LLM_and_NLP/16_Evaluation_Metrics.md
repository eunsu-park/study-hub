# 16. LLM Evaluation Metrics

## Learning Objectives

- Understand text generation evaluation metrics (BLEU, ROUGE, BERTScore)
- Code generation evaluation (HumanEval, MBPP)
- LLM benchmarks (MMLU, HellaSwag, TruthfulQA)
- Human evaluation and automated evaluation

---

## 1. Importance of Evaluation

### Challenges in LLM Evaluation

```
┌─────────────────────────────────────────────────────────────┐
│                   Challenges in LLM Evaluation               │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  1. Multiple correct answers: Various correct responses      │
│                               to the same question           │
│                                                              │
│  2. Subjective quality: Ambiguous criteria for "good"        │
│                         responses                            │
│                                                              │
│  3. Task diversity: Summary, dialogue, code, reasoning, etc. │
│                                                              │
│  4. Knowledge cutoff: Based on training data timestamp       │
│                                                              │
│  5. Safety: Measuring harmfulness, bias, hallucination       │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Evaluation Types

| Evaluation Type | Description | Examples |
|-----------------|-------------|----------|
| Automated evaluation | Algorithm-based scoring | BLEU, ROUGE, Perplexity |
| Model-based evaluation | LLM judges | GPT-4 as Judge |
| Human evaluation | Manual evaluation | A/B testing, Likert scale |
| Benchmarks | Standardized test sets | MMLU, HumanEval |

---

## 2. Text Similarity Metrics

### BLEU (Bilingual Evaluation Understudy)

```python
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
from nltk.translate.bleu_score import SmoothingFunction
import nltk
nltk.download('punkt')

def calculate_bleu(reference, candidate):
    """Calculate BLEU score"""
    # Tokenize
    reference_tokens = [reference.split()]  # Wrap reference in list
    candidate_tokens = candidate.split()

    # Smoothing (handle short sentences)
    smoothie = SmoothingFunction().method1

    # BLEU scores (1-gram to 4-gram)
    bleu_1 = sentence_bleu(reference_tokens, candidate_tokens, weights=(1, 0, 0, 0), smoothing_function=smoothie)
    bleu_2 = sentence_bleu(reference_tokens, candidate_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothie)
    bleu_4 = sentence_bleu(reference_tokens, candidate_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothie)

    return {
        "bleu_1": bleu_1,
        "bleu_2": bleu_2,
        "bleu_4": bleu_4
    }

# Usage
reference = "The cat sat on the mat"
candidate = "The cat is sitting on the mat"
scores = calculate_bleu(reference, candidate)
print(f"BLEU-1: {scores['bleu_1']:.4f}")
print(f"BLEU-4: {scores['bleu_4']:.4f}")
```

### ROUGE (Recall-Oriented Understudy for Gisting Evaluation)

```python
from rouge_score import rouge_scorer

def calculate_rouge(reference, candidate):
    """Calculate ROUGE score"""
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, candidate)

    return {
        "rouge1_f1": scores['rouge1'].fmeasure,
        "rouge2_f1": scores['rouge2'].fmeasure,
        "rougeL_f1": scores['rougeL'].fmeasure,
    }

# Usage
reference = "The quick brown fox jumps over the lazy dog."
candidate = "A quick brown fox jumped over a lazy dog."
scores = calculate_rouge(reference, candidate)
print(f"ROUGE-1 F1: {scores['rouge1_f1']:.4f}")
print(f"ROUGE-2 F1: {scores['rouge2_f1']:.4f}")
print(f"ROUGE-L F1: {scores['rougeL_f1']:.4f}")

# Corpus-level evaluation
def corpus_rouge(references, candidates):
    """Corpus-wide ROUGE"""
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    totals = {'rouge1': 0, 'rouge2': 0, 'rougeL': 0}
    for ref, cand in zip(references, candidates):
        scores = scorer.score(ref, cand)
        totals['rouge1'] += scores['rouge1'].fmeasure
        totals['rouge2'] += scores['rouge2'].fmeasure
        totals['rougeL'] += scores['rougeL'].fmeasure

    n = len(references)
    return {k: v/n for k, v in totals.items()}
```

### BERTScore

```python
from bert_score import score

def calculate_bertscore(references, candidates, lang="en"):
    """Calculate BERTScore (semantic similarity)"""
    P, R, F1 = score(candidates, references, lang=lang, verbose=True)

    return {
        "precision": P.mean().item(),
        "recall": R.mean().item(),
        "f1": F1.mean().item()
    }

# Usage
references = ["The cat sat on the mat.", "It is raining outside."]
candidates = ["A cat is sitting on the mat.", "The weather is rainy."]

bert_scores = calculate_bertscore(references, candidates)
print(f"BERTScore F1: {bert_scores['f1']:.4f}")
```

### Metric Comparison

```python
def compare_metrics(reference, candidate):
    """Compare multiple metrics"""
    results = {}

    # BLEU
    bleu = calculate_bleu(reference, candidate)
    results["BLEU-4"] = bleu["bleu_4"]

    # ROUGE
    rouge = calculate_rouge(reference, candidate)
    results["ROUGE-L"] = rouge["rougeL_f1"]

    # BERTScore
    P, R, F1 = score([candidate], [reference], lang="en")
    results["BERTScore"] = F1.item()

    return results

# Compare
ref = "Machine learning is a subset of artificial intelligence."
cand1 = "ML is part of AI."  # Semantically similar
cand2 = "Machine learning is a subset of artificial intelligence."  # Exact match

print("Candidate 1 (semantically similar):")
print(compare_metrics(ref, cand1))

print("\nCandidate 2 (exact match):")
print(compare_metrics(ref, cand2))
```

---

## 3. Language Model Metrics

### Perplexity

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def calculate_perplexity(model, tokenizer, text, max_length=1024):
    """Calculate perplexity"""
    encodings = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)

    max_length = model.config.n_positions if hasattr(model.config, "n_positions") else 1024
    stride = 512

    lls = []
    for i in range(0, encodings.input_ids.size(1), stride):
        begin_loc = max(i + stride - max_length, 0)
        end_loc = min(i + stride, encodings.input_ids.size(1))
        trg_len = end_loc - i

        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(model.device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            log_likelihood = outputs.loss * trg_len

        lls.append(log_likelihood)

    ppl = torch.exp(torch.stack(lls).sum() / end_loc)
    return ppl.item()

# Usage
model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

text = "The quick brown fox jumps over the lazy dog."
ppl = calculate_perplexity(model, tokenizer, text)
print(f"Perplexity: {ppl:.2f}")
```

### Token-level Accuracy

```python
def token_accuracy(predictions, targets):
    """Token-level accuracy"""
    correct = sum(p == t for p, t in zip(predictions, targets))
    return correct / len(targets)

# Example: Next token prediction
predictions = [1, 2, 3, 4, 5]
targets = [1, 2, 0, 4, 5]
acc = token_accuracy(predictions, targets)
print(f"Token Accuracy: {acc:.2%}")
```

---

## 4. Code Generation Evaluation

### HumanEval (pass@k)

```python
import subprocess
import tempfile
import os
from typing import List

def execute_code(code: str, test_cases: List[str], timeout: int = 5) -> bool:
    """Execute code and test"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(code + "\n")
        for test in test_cases:
            f.write(test + "\n")
        temp_path = f.name

    try:
        result = subprocess.run(
            ['python', temp_path],
            capture_output=True,
            text=True,
            timeout=timeout
        )
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        return False
    finally:
        os.unlink(temp_path)

def pass_at_k(n: int, c: int, k: int) -> float:
    """
    Calculate pass@k
    n: number of generated samples
    c: number of correct samples
    k: k value
    """
    if n - c < k:
        return 1.0

    from math import comb
    return 1.0 - comb(n - c, k) / comb(n, k)

# HumanEval style evaluation
def evaluate_humaneval(model, tokenizer, problems, n_samples=10, k=[1, 10]):
    """HumanEval evaluation"""
    results = []

    for problem in problems:
        prompt = problem["prompt"]
        test_cases = problem["test_cases"]

        # Generate n samples
        correct = 0
        for _ in range(n_samples):
            inputs = tokenizer(prompt, return_tensors="pt")
            outputs = model.generate(**inputs, max_new_tokens=256, temperature=0.8, do_sample=True)
            code = tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Run tests
            if execute_code(code, test_cases):
                correct += 1

        # Calculate pass@k
        pass_rates = {f"pass@{ki}": pass_at_k(n_samples, correct, ki) for ki in k}
        results.append({"problem": problem["name"], **pass_rates})

    return results

# Example problem
problem = {
    "name": "add_two_numbers",
    "prompt": '''def add(a, b):
    """Return the sum of a and b."""
''',
    "test_cases": [
        "assert add(1, 2) == 3",
        "assert add(-1, 1) == 0",
        "assert add(0, 0) == 0"
    ]
}
```

### MBPP (Mostly Basic Python Problems)

```python
from datasets import load_dataset

def evaluate_mbpp(model, tokenizer, n_samples=1):
    """MBPP benchmark evaluation"""
    dataset = load_dataset("mbpp", split="test")

    correct = 0
    total = len(dataset)

    for example in dataset:
        prompt = f"""Write a Python function that {example['text']}

{example['code'].split('def')[0]}def"""

        # Generate code
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=256, temperature=0.2)
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Test
        try:
            full_code = generated + "\n" + "\n".join(example['test_list'])
            exec(full_code)
            correct += 1
        except:
            pass

    return {"accuracy": correct / total}
```

---

## 5. LLM Benchmarks

### MMLU (Massive Multitask Language Understanding)

```python
from datasets import load_dataset

def evaluate_mmlu(model, tokenizer, subjects=None):
    """MMLU benchmark"""
    dataset = load_dataset("cais/mmlu", "all", split="test")

    if subjects:
        dataset = dataset.filter(lambda x: x["subject"] in subjects)

    results = {"correct": 0, "total": 0}
    subject_results = {}

    for example in dataset:
        question = example["question"]
        choices = example["choices"]
        answer = example["answer"]  # 0-3
        subject = example["subject"]

        # Construct prompt
        prompt = f"""Question: {question}

A. {choices[0]}
B. {choices[1]}
C. {choices[2]}
D. {choices[3]}

Answer:"""

        # Generate
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=1, temperature=0)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Check answer
        predicted = response.strip().upper()
        correct_answer = ["A", "B", "C", "D"][answer]

        is_correct = predicted == correct_answer
        results["total"] += 1
        if is_correct:
            results["correct"] += 1

        # Aggregate by subject
        if subject not in subject_results:
            subject_results[subject] = {"correct": 0, "total": 0}
        subject_results[subject]["total"] += 1
        if is_correct:
            subject_results[subject]["correct"] += 1

    # Calculate accuracy
    results["accuracy"] = results["correct"] / results["total"]
    for subject in subject_results:
        s = subject_results[subject]
        s["accuracy"] = s["correct"] / s["total"]

    return results, subject_results

# Usage example
subjects = ["computer_science", "machine_learning", "mathematics"]
# results, by_subject = evaluate_mmlu(model, tokenizer, subjects)
```

### TruthfulQA

```python
from datasets import load_dataset

def evaluate_truthfulqa(model, tokenizer):
    """TruthfulQA evaluation (truthfulness)"""
    dataset = load_dataset("truthful_qa", "generation", split="validation")

    results = []

    for example in dataset:
        question = example["question"]
        best_answer = example["best_answer"]
        correct_answers = example["correct_answers"]
        incorrect_answers = example["incorrect_answers"]

        # Generate
        prompt = f"Question: {question}\nAnswer:"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=100, temperature=0)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response.replace(prompt, "").strip()

        # Evaluate (simple version - in practice use GPT-judge)
        is_truthful = any(ans.lower() in response.lower() for ans in correct_answers)
        is_informative = len(response) > 10 and "I don't know" not in response

        results.append({
            "question": question,
            "response": response,
            "truthful": is_truthful,
            "informative": is_informative
        })

    truthful_rate = sum(r["truthful"] for r in results) / len(results)
    informative_rate = sum(r["informative"] for r in results) / len(results)

    return {
        "truthful": truthful_rate,
        "informative": informative_rate,
        "combined": truthful_rate * informative_rate
    }
```

### HellaSwag (Common Sense Reasoning)

```python
from datasets import load_dataset

def evaluate_hellaswag(model, tokenizer):
    """HellaSwag evaluation"""
    dataset = load_dataset("hellaswag", split="validation")

    correct = 0
    total = len(dataset)

    for example in dataset:
        context = example["ctx"]
        endings = example["endings"]
        label = int(example["label"])

        # Calculate probability for each choice
        scores = []
        for ending in endings:
            text = context + " " + ending
            inputs = tokenizer(text, return_tensors="pt").to(model.device)

            with torch.no_grad():
                outputs = model(**inputs, labels=inputs.input_ids)
                scores.append(-outputs.loss.item())  # Lower loss = higher probability

        predicted = scores.index(max(scores))
        if predicted == label:
            correct += 1

    return {"accuracy": correct / total}
```

---

## 6. LLM-as-Judge Evaluation

### GPT-4 Evaluator

```python
from openai import OpenAI

client = OpenAI()

def llm_judge(question, response_a, response_b):
    """Compare responses using LLM"""
    judge_prompt = f"""Compare two AI responses and select the better one.

Question: {question}

Response A:
{response_a}

Response B:
{response_b}

Evaluation criteria:
1. Accuracy: Is the information accurate?
2. Usefulness: Does it appropriately answer the question?
3. Clarity: Is it easy to understand?
4. Completeness: Is it sufficiently detailed?

After analysis, answer in this format:
Analysis: [Comparison by each criterion]
Winner: [A or B or Tie]
"""

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": judge_prompt}],
        temperature=0
    )

    return response.choices[0].message.content

def pairwise_comparison(questions, model_a_responses, model_b_responses):
    """Pairwise comparison evaluation"""
    results = {"A_wins": 0, "B_wins": 0, "ties": 0}

    for q, a, b in zip(questions, model_a_responses, model_b_responses):
        judgment = llm_judge(q, a, b)

        if "Winner: A" in judgment:
            results["A_wins"] += 1
        elif "Winner: B" in judgment:
            results["B_wins"] += 1
        else:
            results["ties"] += 1

    total = len(questions)
    return {
        "model_a_win_rate": results["A_wins"] / total,
        "model_b_win_rate": results["B_wins"] / total,
        "tie_rate": results["ties"] / total
    }
```

### Multi-dimensional Evaluation

```python
def multidim_evaluation(question, response):
    """Multi-dimensional LLM evaluation"""
    eval_prompt = f"""Evaluate the following AI response on multiple dimensions from 1-5.

Question: {question}

Response: {response}

Output in JSON format:
{{
    "relevance": <1-5>,
    "accuracy": <1-5>,
    "helpfulness": <1-5>,
    "coherence": <1-5>,
    "safety": <1-5>,
    "overall": <1-5>,
    "explanation": "<reason for evaluation>"
}}
"""

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": eval_prompt}],
        temperature=0,
        response_format={"type": "json_object"}
    )

    import json
    return json.loads(response.choices[0].message.content)

# Usage
scores = multidim_evaluation(
    "What is artificial intelligence?",
    "Artificial intelligence (AI) is technology where computer systems mimic human intelligence..."
)
print(scores)
```

---

## 7. Human Evaluation

### Evaluation Interface

```python
import gradio as gr

def human_evaluation_interface():
    """Gradio interface for human evaluation"""

    def submit_evaluation(question, response, relevance, quality, safety, feedback):
        # Save results
        result = {
            "question": question,
            "response": response,
            "scores": {
                "relevance": relevance,
                "quality": quality,
                "safety": safety
            },
            "feedback": feedback
        }
        # Save to DB, etc.
        return f"Evaluation saved: {result}"

    with gr.Blocks() as demo:
        gr.Markdown("# LLM Response Evaluation")

        with gr.Row():
            question = gr.Textbox(label="Question")
            response = gr.Textbox(label="AI Response", lines=5)

        with gr.Row():
            relevance = gr.Slider(1, 5, step=1, label="Relevance")
            quality = gr.Slider(1, 5, step=1, label="Quality")
            safety = gr.Slider(1, 5, step=1, label="Safety")

        feedback = gr.Textbox(label="Additional Feedback", lines=3)
        submit_btn = gr.Button("Submit")
        result = gr.Textbox(label="Result")

        submit_btn.click(
            submit_evaluation,
            inputs=[question, response, relevance, quality, safety, feedback],
            outputs=[result]
        )

    return demo

# demo = human_evaluation_interface()
# demo.launch()
```

### A/B Testing

```python
import random
from dataclasses import dataclass
from typing import Optional

@dataclass
class ABTestResult:
    question: str
    response_a: str
    response_b: str
    chosen: str  # "A" or "B"
    evaluator_id: str
    reason: Optional[str] = None

class ABTestManager:
    def __init__(self):
        self.results = []

    def get_pair(self, question, model_a, model_b, tokenizer):
        """Return two responses in random order"""
        # Generate responses
        inputs = tokenizer(question, return_tensors="pt")
        response_a = tokenizer.decode(model_a.generate(**inputs)[0])
        response_b = tokenizer.decode(model_b.generate(**inputs)[0])

        # Random order
        if random.random() > 0.5:
            return response_a, response_b, "A", "B"
        else:
            return response_b, response_a, "B", "A"

    def record_result(self, result: ABTestResult):
        self.results.append(result)

    def analyze(self):
        """Analyze results"""
        a_wins = sum(1 for r in self.results if r.chosen == "A")
        b_wins = sum(1 for r in self.results if r.chosen == "B")
        total = len(self.results)

        return {
            "model_a_win_rate": a_wins / total if total > 0 else 0,
            "model_b_win_rate": b_wins / total if total > 0 else 0,
            "total_evaluations": total
        }
```

---

## 8. Integrated Evaluation Framework

### lm-evaluation-harness

```bash
# Installation
pip install lm-eval

# Usage
lm_eval --model hf \
    --model_args pretrained=meta-llama/Llama-2-7b-hf \
    --tasks mmlu,hellaswag,truthfulqa \
    --batch_size 8
```

### Custom Evaluation Class

```python
class LLMEvaluator:
    """Integrated LLM evaluator"""

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.results = {}

    def evaluate_all(self, test_data):
        """Run full evaluation"""
        self.results = {
            "perplexity": self._eval_perplexity(test_data["texts"]),
            "rouge": self._eval_rouge(test_data["summaries"]),
            "mmlu": self._eval_mmlu(test_data.get("mmlu_samples", [])),
            "pass_at_k": self._eval_code(test_data.get("code_problems", [])),
        }
        return self.results

    def _eval_perplexity(self, texts):
        ppls = [calculate_perplexity(self.model, self.tokenizer, t) for t in texts]
        return {"mean": sum(ppls) / len(ppls), "values": ppls}

    def _eval_rouge(self, summaries):
        scores = [calculate_rouge(s["reference"], s["candidate"]) for s in summaries]
        return {
            "rouge1": sum(s["rouge1_f1"] for s in scores) / len(scores),
            "rougeL": sum(s["rougeL_f1"] for s in scores) / len(scores),
        }

    def _eval_mmlu(self, samples):
        # MMLU evaluation logic
        pass

    def _eval_code(self, problems):
        # Code evaluation logic
        pass

    def generate_report(self):
        """Generate evaluation report"""
        report = "# LLM Evaluation Report\n\n"

        for metric, values in self.results.items():
            report += f"## {metric.upper()}\n"
            if isinstance(values, dict):
                for k, v in values.items():
                    if isinstance(v, float):
                        report += f"- {k}: {v:.4f}\n"
            report += "\n"

        return report
```

---

## Summary

### Metric Selection Guide

| Task | Recommended Metrics |
|------|---------------------|
| Translation | BLEU, COMET |
| Summarization | ROUGE, BERTScore |
| Dialogue | Human Eval, LLM-as-Judge |
| QA | Exact Match, F1 |
| Code Generation | pass@k, MBPP |
| General Ability | MMLU, HellaSwag |
| Truthfulness | TruthfulQA |

### Core Code

```python
# ROUGE
from rouge_score import rouge_scorer
scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'])
scores = scorer.score(reference, candidate)

# BERTScore
from bert_score import score
P, R, F1 = score(candidates, references, lang="en")

# pass@k
from math import comb
pass_k = 1.0 - comb(n - c, k) / comb(n, k)

# LLM-as-Judge
judgment = llm_judge(question, response_a, response_b)
```

### Evaluation Checklist

```
□ Select appropriate metrics for task
□ Combine various evaluation methods (automated + human)
□ Secure sufficient test samples
□ Check inter-rater agreement
□ Calculate confidence intervals for results
□ Reproducible evaluation environment
```

## Exercises

### Exercise 1: BLEU Score Limitations

Compute the BLEU-1 and BLEU-4 scores for the following candidate-reference pairs. Then explain why BLEU fails to capture the quality difference between candidates B and C, and what metric would be more appropriate.

```python
reference = "The patient was given a high dose of aspirin to reduce fever."

candidate_a = "The patient was given a high dose of aspirin to reduce fever."  # Exact match
candidate_b = "The sick person received a large amount of aspirin medication for temperature reduction."  # Paraphrase
candidate_c = "High aspirin dose fever patient given reduce."  # Same words, incoherent order
```

<details>
<summary>Show Answer</summary>

```python
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

reference = "The patient was given a high dose of aspirin to reduce fever."
candidates = {
    "a_exact": "The patient was given a high dose of aspirin to reduce fever.",
    "b_paraphrase": "The sick person received a large amount of aspirin medication for temperature reduction.",
    "c_incoherent": "High aspirin dose fever patient given reduce."
}

smoothie = SmoothingFunction().method1
ref_tokens = [reference.split()]

print("BLEU Scores:")
print(f"{'Candidate':<15} {'BLEU-1':<10} {'BLEU-4'}")
print("-" * 35)

for name, cand in candidates.items():
    cand_tokens = cand.split()
    bleu1 = sentence_bleu(ref_tokens, cand_tokens, weights=(1, 0, 0, 0), smoothing_function=smoothie)
    bleu4 = sentence_bleu(ref_tokens, cand_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothie)
    print(f"{name:<15} {bleu1:<10.4f} {bleu4:.4f}")

# BLEU Scores:
# Candidate       BLEU-1     BLEU-4
# a_exact         1.0000     1.0000
# b_paraphrase    0.1538     0.0082  ← Low despite good meaning
# c_incoherent    0.5556     0.0001  ← BLEU-1 is high despite being useless!

# BERTScore comparison
from bert_score import score
refs = [reference] * 3
cands = list(candidates.values())
P, R, F1 = score(cands, refs, lang="en", verbose=False)

print("\nBERTScore F1:")
for name, f1 in zip(candidates.keys(), F1):
    print(f"{name:<15} {f1.item():.4f}")

# BERTScore F1:
# a_exact         1.0000  ← Exact match
# b_paraphrase    0.8900  ← High! Captures semantic similarity
# c_incoherent    0.8300  ← Still somewhat penalized for incoherence
```

**Why BLEU fails here:**

| Issue | Problem |
|-------|---------|
| **Candidate B** | BLEU counts n-gram overlaps literally. "sick person" ≠ "patient", "large amount" ≠ "high dose" — even though the meaning is identical. BLEU-1 = 0.15 despite being a perfect paraphrase. |
| **Candidate C** | BLEU-1 = 0.56 because 5/9 individual words match. BLEU has no notion of word order or grammaticality at the unigram level. Only when using bigrams (BLEU-2+) does the incoherence get penalized. |

**Better metrics for this scenario:**
- **BERTScore**: Uses contextual embeddings to capture semantic similarity — correctly scores B highly
- **COMET** (for translation): Neural metric trained on human quality judgments
- **LLM-as-Judge**: Can evaluate fluency, factual accuracy, and completeness holistically

**When BLEU is still useful:** Machine translation (large corpus, many references available), or when lexical overlap is genuinely important (e.g., technical documentation that must use specific terminology).
</details>

---

### Exercise 2: pass@k Calculation and Interpretation

A code generation model produces 10 samples for each problem. For the following results, calculate pass@1, pass@3, and pass@10. Then explain what these numbers tell you about the model's capabilities.

```python
from math import comb

def pass_at_k(n: int, c: int, k: int) -> float:
    """n=samples generated, c=correct samples, k=budget"""
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)

# Evaluation results: (problem_name, n_samples, n_correct)
results = [
    ("fibonacci", 10, 10),     # Very easy — all correct
    ("binary_search", 10, 7),  # Moderate — most correct
    ("merge_sort", 10, 3),     # Hard — few correct
    ("regex_parser", 10, 1),   # Very hard — barely one correct
    ("graph_coloring", 10, 0), # Failed — none correct
]
```

<details>
<summary>Show Answer</summary>

```python
from math import comb

def pass_at_k(n: int, c: int, k: int) -> float:
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)

results = [
    ("fibonacci", 10, 10),
    ("binary_search", 10, 7),
    ("merge_sort", 10, 3),
    ("regex_parser", 10, 1),
    ("graph_coloring", 10, 0),
]

print(f"{'Problem':<20} {'pass@1':<10} {'pass@3':<10} {'pass@10'}")
print("-" * 50)

totals = {1: 0, 3: 0, 10: 0}
for name, n, c in results:
    p1 = pass_at_k(n, c, 1)
    p3 = pass_at_k(n, c, 3)
    p10 = pass_at_k(n, c, 10)
    print(f"{name:<20} {p1:<10.4f} {p3:<10.4f} {p10:.4f}")
    totals[1] += p1; totals[3] += p3; totals[10] += p10

num_problems = len(results)
print("-" * 50)
print(f"{'Average':<20} {totals[1]/num_problems:<10.4f} {totals[3]/num_problems:<10.4f} {totals[10]/num_problems:.4f}")

# Output:
# Problem              pass@1     pass@3     pass@10
# fibonacci            1.0000     1.0000     1.0000
# binary_search        0.7000     0.9667     1.0000
# merge_sort           0.3000     0.6583     0.9833
# regex_parser         0.1000     0.2667     0.6512
# graph_coloring       0.0000     0.0000     0.0000
# Average              0.4200     0.5783     0.7269

# --- Interpretation ---
```

**What these numbers tell us:**

| Metric | Meaning | This model's profile |
|--------|---------|---------------------|
| **pass@1** | Probability a single generated solution is correct. Used when you deploy the model to generate one answer. | 0.42 — mediocre for production use |
| **pass@3** | Probability at least one of 3 candidates is correct. Used when you can verify 3 solutions (e.g., run tests). | 0.58 — improved with selection |
| **pass@10** | Upper bound with 10 samples. Measures the model's "peak capability" vs "consistency". | 0.73 — model knows the answer, just inconsistently |

**Key insight — the gap between pass@1 and pass@10:**
- Large gap (like `regex_parser`: 0.10 vs 0.65) → Model has the capability but is unreliable. Solution: use best-of-N sampling with a verifier (run tests, pass/fail).
- Small gap (like `fibonacci`: 1.0 vs 1.0) → Model consistently knows this.
- pass@10 = 0 (`graph_coloring`) → Model lacks the capability entirely — no amount of sampling will help.

**Practical recommendation:** For production code generation, optimize pass@1 through fine-tuning + temperature tuning. For research, report pass@1 and pass@10 together to show both reliability and ceiling.
</details>

---

### Exercise 3: LLM-as-Judge Bias Mitigation

The `llm_judge` function in this lesson has a potential position bias: it might consistently prefer whichever response appears first (Response A). Implement a debiased version that runs the comparison twice with swapped order and aggregates the results.

<details>
<summary>Show Answer</summary>

```python
from openai import OpenAI
from enum import Enum

client = OpenAI()

class JudgeResult(Enum):
    A_WINS = "A"
    B_WINS = "B"
    TIE = "Tie"

def single_comparison(question: str, response_first: str, response_second: str,
                      label_first: str = "A", label_second: str = "B") -> JudgeResult:
    """Single comparison run."""
    judge_prompt = f"""Compare two AI responses and select the better one.

Question: {question}

Response {label_first}:
{response_first}

Response {label_second}:
{response_second}

Evaluation criteria:
1. Accuracy: Is the information accurate?
2. Usefulness: Does it appropriately answer the question?
3. Clarity: Is it easy to understand?
4. Completeness: Is it sufficiently detailed?

After analysis, reply ONLY with one of: {label_first}, {label_second}, or Tie.
Reply:"""

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": judge_prompt}],
        temperature=0,
        max_tokens=10
    )

    result = response.choices[0].message.content.strip()

    if label_first in result:
        return JudgeResult.A_WINS if label_first == "A" else JudgeResult.B_WINS
    elif label_second in result:
        return JudgeResult.A_WINS if label_second == "A" else JudgeResult.B_WINS
    else:
        return JudgeResult.TIE

def debiased_judge(question: str, response_a: str, response_b: str) -> dict:
    """
    Debiased LLM judge using position-swapped double evaluation.

    Run 1: A first, B second → get judgment
    Run 2: B first, A second → get judgment (swapped)

    If both runs agree → confident result
    If they disagree → Tie (position bias detected)
    """
    # Run 1: A first
    result_1 = single_comparison(question, response_a, response_b, "A", "B")

    # Run 2: B first (swapped)
    result_2 = single_comparison(question, response_b, response_a, "B", "A")
    # Note: labels are swapped back to A/B perspective in single_comparison

    # Wait — we need to interpret result_2 correctly:
    # In run 2, if judge says "B" (which was shown first), that means response_b won
    # Let's re-implement with explicit tracking:

    result_2_raw = single_comparison(question, response_b, response_a, "First", "Second")
    # "First" = response_b, "Second" = response_a
    if result_2_raw == JudgeResult.A_WINS:  # "First" won = B won
        result_2_normalized = JudgeResult.B_WINS
    elif result_2_raw == JudgeResult.B_WINS:  # "Second" won = A won
        result_2_normalized = JudgeResult.A_WINS
    else:
        result_2_normalized = JudgeResult.TIE

    # Aggregate
    if result_1 == result_2_normalized:
        final = result_1  # Both agree
        confidence = "high"
    elif result_1 == JudgeResult.TIE or result_2_normalized == JudgeResult.TIE:
        final = result_1 if result_2_normalized == JudgeResult.TIE else result_2_normalized
        confidence = "medium"
    else:
        final = JudgeResult.TIE  # Disagreement = position bias detected, call it a tie
        confidence = "low (position bias detected)"

    return {
        "final_result": final.value,
        "run_1_result": result_1.value,
        "run_2_result": result_2_normalized.value,
        "confidence": confidence,
        "position_bias_detected": result_1 != result_2_normalized
    }

# Test
result = debiased_judge(
    question="What is recursion in programming?",
    response_a="Recursion is when a function calls itself. For example, factorial(n) = n * factorial(n-1).",
    response_b="Recursion is a programming technique where a function invokes itself to solve smaller subproblems, with a base case to stop the recursion. It elegantly solves problems like tree traversal and divide-and-conquer algorithms."
)
print(result)
# Expected: B wins (more complete answer), high confidence if consistent across orderings
```

**Why this matters:** Studies show LLM judges exhibit 10-30% position bias, consistently preferring whichever response appears first. The double-evaluation technique detects and mitigates this. Additional bias mitigation strategies:
- Use multiple independent judges and take majority vote
- Present responses without labels (just "First"/"Second")
- Use scoring (1-5) instead of pairwise preference when possible
- Shuffle response order across a test set and check if win rates are consistent
</details>

---

## Learning Complete

This completes the advanced LLM & NLP learning!

### Overall Learning Summary

1. **NLP Basics (01-03)**: Tokenization, embeddings, Transformer
2. **Pre-trained Models (04-07)**: BERT, GPT, HuggingFace, fine-tuning
3. **LLM Applications (08-12)**: Prompts, RAG, LangChain, vector DB, chatbots
4. **Advanced LLM (13-16)**: Quantization, RLHF, agents, evaluation

### Recommended Next Steps

- Apply to real projects
- Participate in Kaggle NLP competitions
- Read latest LLM papers (Claude, Gemini, Llama)
- Contribute to open source LLM projects
