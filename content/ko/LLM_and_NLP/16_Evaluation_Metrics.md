# 16. LLM 평가 지표 (Evaluation Metrics)

## 학습 목표

- 텍스트 생성 평가 지표 이해 (BLEU, ROUGE, BERTScore)
- 코드 생성 평가 (HumanEval, MBPP)
- LLM 벤치마크 (MMLU, HellaSwag, TruthfulQA)
- 인간 평가와 자동 평가

---

## 1. 평가의 중요성

### LLM 평가의 어려움

```
┌─────────────────────────────────────────────────────────────┐
│                   LLM 평가의 어려움                          │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  1. 정답이 여러 개: 같은 질문에 다양한 정답 가능              │
│                                                              │
│  2. 주관적 품질: "좋은" 응답의 기준이 모호                   │
│                                                              │
│  3. 태스크 다양성: 요약, 대화, 코드, 추론 등 다양             │
│                                                              │
│  4. 지식 시점: 학습 데이터 기준 시점                         │
│                                                              │
│  5. 안전성: 유해성, 편향, 환각 측정                          │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 평가 유형

| 평가 유형 | 설명 | 예시 |
|----------|------|------|
| 자동 평가 | 알고리즘 기반 점수 | BLEU, ROUGE, Perplexity |
| 모델 기반 평가 | LLM이 평가 | GPT-4 as Judge |
| 인간 평가 | 사람이 직접 평가 | A/B 테스트, 리커트 척도 |
| 벤치마크 | 표준화된 테스트셋 | MMLU, HumanEval |

---

## 2. 텍스트 유사도 지표

### BLEU (Bilingual Evaluation Understudy)

```python
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
from nltk.translate.bleu_score import SmoothingFunction
import nltk
nltk.download('punkt')

def calculate_bleu(reference, candidate):
    """BLEU 점수 계산"""
    # 토큰화
    reference_tokens = [reference.split()]  # 참조문은 리스트로 감싸기
    candidate_tokens = candidate.split()

    # Smoothing (짧은 문장 처리)
    smoothie = SmoothingFunction().method1

    # BLEU 점수 (1-gram부터 4-gram까지)
    bleu_1 = sentence_bleu(reference_tokens, candidate_tokens, weights=(1, 0, 0, 0), smoothing_function=smoothie)
    bleu_2 = sentence_bleu(reference_tokens, candidate_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothie)
    bleu_4 = sentence_bleu(reference_tokens, candidate_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothie)

    return {
        "bleu_1": bleu_1,
        "bleu_2": bleu_2,
        "bleu_4": bleu_4
    }

# 사용
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
    """ROUGE 점수 계산"""
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, candidate)

    return {
        "rouge1_f1": scores['rouge1'].fmeasure,
        "rouge2_f1": scores['rouge2'].fmeasure,
        "rougeL_f1": scores['rougeL'].fmeasure,
    }

# 사용
reference = "The quick brown fox jumps over the lazy dog."
candidate = "A quick brown fox jumped over a lazy dog."
scores = calculate_rouge(reference, candidate)
print(f"ROUGE-1 F1: {scores['rouge1_f1']:.4f}")
print(f"ROUGE-2 F1: {scores['rouge2_f1']:.4f}")
print(f"ROUGE-L F1: {scores['rougeL_f1']:.4f}")

# 코퍼스 레벨 평가
def corpus_rouge(references, candidates):
    """코퍼스 전체 ROUGE"""
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
    """BERTScore 계산 (의미적 유사도)"""
    P, R, F1 = score(candidates, references, lang=lang, verbose=True)

    return {
        "precision": P.mean().item(),
        "recall": R.mean().item(),
        "f1": F1.mean().item()
    }

# 사용
references = ["The cat sat on the mat.", "It is raining outside."]
candidates = ["A cat is sitting on the mat.", "The weather is rainy."]

bert_scores = calculate_bertscore(references, candidates)
print(f"BERTScore F1: {bert_scores['f1']:.4f}")
```

### 지표 비교

```python
def compare_metrics(reference, candidate):
    """여러 지표 비교"""
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

# 비교
ref = "Machine learning is a subset of artificial intelligence."
cand1 = "ML is part of AI."  # 의미적으로 유사
cand2 = "Machine learning is a subset of artificial intelligence."  # 완전 동일

print("후보 1 (의미적 유사):")
print(compare_metrics(ref, cand1))

print("\n후보 2 (완전 동일):")
print(compare_metrics(ref, cand2))
```

---

## 3. 언어 모델 지표

### Perplexity (당혹도)

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def calculate_perplexity(model, tokenizer, text, max_length=1024):
    """퍼플렉시티 계산"""
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

# 사용
model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

text = "The quick brown fox jumps over the lazy dog."
ppl = calculate_perplexity(model, tokenizer, text)
print(f"Perplexity: {ppl:.2f}")
```

### Token-level Accuracy

```python
def token_accuracy(predictions, targets):
    """토큰 레벨 정확도"""
    correct = sum(p == t for p, t in zip(predictions, targets))
    return correct / len(targets)

# 예시: 다음 토큰 예측
predictions = [1, 2, 3, 4, 5]
targets = [1, 2, 0, 4, 5]
acc = token_accuracy(predictions, targets)
print(f"Token Accuracy: {acc:.2%}")
```

---

## 4. 코드 생성 평가

### HumanEval (pass@k)

```python
import subprocess
import tempfile
import os
from typing import List

def execute_code(code: str, test_cases: List[str], timeout: int = 5) -> bool:
    """코드 실행 및 테스트"""
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
    pass@k 계산
    n: 생성된 샘플 수
    c: 정답 샘플 수
    k: k값
    """
    if n - c < k:
        return 1.0

    from math import comb
    return 1.0 - comb(n - c, k) / comb(n, k)

# HumanEval 스타일 평가
def evaluate_humaneval(model, tokenizer, problems, n_samples=10, k=[1, 10]):
    """HumanEval 평가"""
    results = []

    for problem in problems:
        prompt = problem["prompt"]
        test_cases = problem["test_cases"]

        # n개 샘플 생성
        correct = 0
        for _ in range(n_samples):
            inputs = tokenizer(prompt, return_tensors="pt")
            outputs = model.generate(**inputs, max_new_tokens=256, temperature=0.8, do_sample=True)
            code = tokenizer.decode(outputs[0], skip_special_tokens=True)

            # 테스트 실행
            if execute_code(code, test_cases):
                correct += 1

        # pass@k 계산
        pass_rates = {f"pass@{ki}": pass_at_k(n_samples, correct, ki) for ki in k}
        results.append({"problem": problem["name"], **pass_rates})

    return results

# 예제 문제
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
    """MBPP 벤치마크 평가"""
    dataset = load_dataset("mbpp", split="test")

    correct = 0
    total = len(dataset)

    for example in dataset:
        prompt = f"""Write a Python function that {example['text']}

{example['code'].split('def')[0]}def"""

        # 코드 생성
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=256, temperature=0.2)
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # 테스트
        try:
            full_code = generated + "\n" + "\n".join(example['test_list'])
            exec(full_code)
            correct += 1
        except:
            pass

    return {"accuracy": correct / total}
```

---

## 5. LLM 벤치마크

### MMLU (Massive Multitask Language Understanding)

```python
from datasets import load_dataset

def evaluate_mmlu(model, tokenizer, subjects=None):
    """MMLU 벤치마크"""
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

        # 프롬프트 구성
        prompt = f"""Question: {question}

A. {choices[0]}
B. {choices[1]}
C. {choices[2]}
D. {choices[3]}

Answer:"""

        # 생성
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=1, temperature=0)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # 정답 확인
        predicted = response.strip().upper()
        correct_answer = ["A", "B", "C", "D"][answer]

        is_correct = predicted == correct_answer
        results["total"] += 1
        if is_correct:
            results["correct"] += 1

        # 과목별 집계
        if subject not in subject_results:
            subject_results[subject] = {"correct": 0, "total": 0}
        subject_results[subject]["total"] += 1
        if is_correct:
            subject_results[subject]["correct"] += 1

    # 정확도 계산
    results["accuracy"] = results["correct"] / results["total"]
    for subject in subject_results:
        s = subject_results[subject]
        s["accuracy"] = s["correct"] / s["total"]

    return results, subject_results

# 사용 예시
subjects = ["computer_science", "machine_learning", "mathematics"]
# results, by_subject = evaluate_mmlu(model, tokenizer, subjects)
```

### TruthfulQA

```python
from datasets import load_dataset

def evaluate_truthfulqa(model, tokenizer):
    """TruthfulQA 평가 (진실성)"""
    dataset = load_dataset("truthful_qa", "generation", split="validation")

    results = []

    for example in dataset:
        question = example["question"]
        best_answer = example["best_answer"]
        correct_answers = example["correct_answers"]
        incorrect_answers = example["incorrect_answers"]

        # 생성
        prompt = f"Question: {question}\nAnswer:"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=100, temperature=0)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response.replace(prompt, "").strip()

        # 평가 (간단한 버전 - 실제로는 GPT-judge 사용)
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

### HellaSwag (상식 추론)

```python
from datasets import load_dataset

def evaluate_hellaswag(model, tokenizer):
    """HellaSwag 평가"""
    dataset = load_dataset("hellaswag", split="validation")

    correct = 0
    total = len(dataset)

    for example in dataset:
        context = example["ctx"]
        endings = example["endings"]
        label = int(example["label"])

        # 각 선택지에 대한 확률 계산
        scores = []
        for ending in endings:
            text = context + " " + ending
            inputs = tokenizer(text, return_tensors="pt").to(model.device)

            with torch.no_grad():
                outputs = model(**inputs, labels=inputs.input_ids)
                scores.append(-outputs.loss.item())  # 낮은 loss = 높은 확률

        predicted = scores.index(max(scores))
        if predicted == label:
            correct += 1

    return {"accuracy": correct / total}
```

---

## 6. LLM-as-Judge 평가

### GPT-4 평가자

```python
from openai import OpenAI

client = OpenAI()

def llm_judge(question, response_a, response_b):
    """LLM을 사용한 응답 비교"""
    judge_prompt = f"""두 AI 응답을 비교하여 더 나은 것을 선택하세요.

질문: {question}

응답 A:
{response_a}

응답 B:
{response_b}

평가 기준:
1. 정확성: 정보가 정확한가?
2. 유용성: 질문에 적절히 답변했는가?
3. 명확성: 이해하기 쉬운가?
4. 완전성: 충분히 상세한가?

분석 후 다음 형식으로 답하세요:
분석: [각 기준별 비교]
승자: [A 또는 B 또는 동점]
"""

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": judge_prompt}],
        temperature=0
    )

    return response.choices[0].message.content

def pairwise_comparison(questions, model_a_responses, model_b_responses):
    """쌍대 비교 평가"""
    results = {"A_wins": 0, "B_wins": 0, "ties": 0}

    for q, a, b in zip(questions, model_a_responses, model_b_responses):
        judgment = llm_judge(q, a, b)

        if "승자: A" in judgment:
            results["A_wins"] += 1
        elif "승자: B" in judgment:
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

### 다차원 평가

```python
def multidim_evaluation(question, response):
    """다차원 LLM 평가"""
    eval_prompt = f"""다음 AI 응답을 여러 차원에서 1-5점으로 평가하세요.

질문: {question}

응답: {response}

다음 형식으로 JSON 출력:
{{
    "relevance": <1-5>,
    "accuracy": <1-5>,
    "helpfulness": <1-5>,
    "coherence": <1-5>,
    "safety": <1-5>,
    "overall": <1-5>,
    "explanation": "<평가 이유>"
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

# 사용
scores = multidim_evaluation(
    "인공지능이란 무엇인가요?",
    "인공지능(AI)은 컴퓨터 시스템이 인간의 지능을 모방하는 기술입니다..."
)
print(scores)
```

---

## 7. 인간 평가

### 평가 인터페이스

```python
import gradio as gr

def human_evaluation_interface():
    """인간 평가용 Gradio 인터페이스"""

    def submit_evaluation(question, response, relevance, quality, safety, feedback):
        # 결과 저장
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
        # DB에 저장 등
        return f"평가 저장됨: {result}"

    with gr.Blocks() as demo:
        gr.Markdown("# LLM 응답 평가")

        with gr.Row():
            question = gr.Textbox(label="질문")
            response = gr.Textbox(label="AI 응답", lines=5)

        with gr.Row():
            relevance = gr.Slider(1, 5, step=1, label="관련성")
            quality = gr.Slider(1, 5, step=1, label="품질")
            safety = gr.Slider(1, 5, step=1, label="안전성")

        feedback = gr.Textbox(label="추가 피드백", lines=3)
        submit_btn = gr.Button("제출")
        result = gr.Textbox(label="결과")

        submit_btn.click(
            submit_evaluation,
            inputs=[question, response, relevance, quality, safety, feedback],
            outputs=[result]
        )

    return demo

# demo = human_evaluation_interface()
# demo.launch()
```

### A/B 테스트

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
        """무작위 순서로 두 응답 반환"""
        # 응답 생성
        inputs = tokenizer(question, return_tensors="pt")
        response_a = tokenizer.decode(model_a.generate(**inputs)[0])
        response_b = tokenizer.decode(model_b.generate(**inputs)[0])

        # 무작위 순서
        if random.random() > 0.5:
            return response_a, response_b, "A", "B"
        else:
            return response_b, response_a, "B", "A"

    def record_result(self, result: ABTestResult):
        self.results.append(result)

    def analyze(self):
        """결과 분석"""
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

## 8. 통합 평가 프레임워크

### lm-evaluation-harness

```bash
# 설치
pip install lm-eval

# 사용
lm_eval --model hf \
    --model_args pretrained=meta-llama/Llama-2-7b-hf \
    --tasks mmlu,hellaswag,truthfulqa \
    --batch_size 8
```

### 커스텀 평가 클래스

```python
class LLMEvaluator:
    """통합 LLM 평가기"""

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.results = {}

    def evaluate_all(self, test_data):
        """전체 평가 실행"""
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
        # MMLU 평가 로직
        pass

    def _eval_code(self, problems):
        # 코드 평가 로직
        pass

    def generate_report(self):
        """평가 보고서 생성"""
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

## 정리

### 평가 지표 선택 가이드

| 태스크 | 추천 지표 |
|--------|----------|
| 번역 | BLEU, COMET |
| 요약 | ROUGE, BERTScore |
| 대화 | Human Eval, LLM-as-Judge |
| QA | Exact Match, F1 |
| 코드 생성 | pass@k, MBPP |
| 일반 능력 | MMLU, HellaSwag |
| 진실성 | TruthfulQA |

### 핵심 코드

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

### 평가 체크리스트

```
□ 태스크에 맞는 지표 선택
□ 다양한 평가 방법 조합 (자동 + 인간)
□ 충분한 테스트 샘플 확보
□ 평가자 간 일치도 확인
□ 결과 신뢰구간 계산
□ 재현 가능한 평가 환경
```

---

## 연습 문제

### 연습 문제 1: BLEU 점수 한계

아래 참조 번역과 세 후보 문장에 대해 BLEU-1 및 BLEU-4 점수를 직접 계산하고, 각 결과가 번역 품질을 제대로 반영하는지 분석하세요.

```python
reference = "The cat sat on the mat"

candidates = {
    "exact":       "The cat sat on the mat",         # 완전 일치
    "paraphrase":  "A feline rested upon the rug",   # 의미는 같지만 단어가 다름
    "incoherent":  "The the the the the the the",    # 문법적으로 틀린 반복
}
```

각 후보에 대해 다음을 수행하세요:
1. `nltk.translate.bleu_score`로 BLEU-1 및 BLEU-4 점수 계산
2. 결과를 표로 정리 (후보 | BLEU-1 | BLEU-4 | 품질)
3. BLEU가 왜 의미론적 동의어(paraphrase)와 무의미 반복(incoherent)을 올바르게 구분하지 못하는지 설명
4. BERTScore가 이 두 경우를 더 잘 처리하는 이유를 2-3 문장으로 설명

<details>
<summary>정답 보기</summary>

```python
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import nltk

reference = "The cat sat on the mat".split()
candidates = {
    "exact":       "The cat sat on the mat".split(),
    "paraphrase":  "A feline rested upon the rug".split(),
    "incoherent":  "The the the the the the the".split(),
}

smoothie = SmoothingFunction().method1
print(f"{'후보':<12} {'BLEU-1':>8} {'BLEU-4':>8}")
print("-" * 30)
for name, cand in candidates.items():
    b1 = sentence_bleu([reference], cand, weights=(1,0,0,0), smoothing_function=smoothie)
    b4 = sentence_bleu([reference], cand, weights=(.25,.25,.25,.25), smoothing_function=smoothie)
    print(f"{name:<12} {b1:>8.3f} {b4:>8.3f}")

# 예상 결과:
# 후보         BLEU-1   BLEU-4
# ------------------------------
# exact         1.000    1.000
# paraphrase    0.000    0.000   ← 의미가 같아도 0
# incoherent    0.857    0.001   ← 높은 BLEU-1, 낮은 BLEU-4
```

**핵심 인사이트:**
- **paraphrase**: 모든 단어가 참조와 다르므로 BLEU-1=0. 그러나 실제로는 의미 있는 번역임
- **incoherent**: "the"가 참조에 존재하므로 BLEU-1이 높게 나옴 (브리바이어티 패널티로 낮아지지만)
- **BERTScore 우위**: BERT 임베딩은 "feline"과 "cat"의 코사인 유사도를 계산하므로 의미적 유사성을 포착함. "the the the"는 문맥 임베딩이 분산되어 낮은 점수를 받음

</details>

---

### 연습 문제 2: pass@k 계산 및 해석

5개의 코딩 문제에 대해 각각 n=10번씩 LLM 응답을 생성하고 테스트를 통과한 횟수를 기록했습니다. 아래 데이터로 pass@1, pass@3, pass@10을 계산하세요.

```python
results = {
    "problem_1": {"n": 10, "c": 10},  # 10번 중 10번 통과
    "problem_2": {"n": 10, "c": 7},   # 10번 중 7번 통과
    "problem_3": {"n": 10, "c": 3},   # 10번 중 3번 통과
    "problem_4": {"n": 10, "c": 1},   # 10번 중 1번 통과
    "problem_5": {"n": 10, "c": 0},   # 10번 중 0번 통과
}
```

1. pass@k 공식 구현: `pass_at_k(n, c, k) = 1 - C(n-c, k) / C(n, k)`
2. 각 문제에 대해 k=1, 3, 10 값을 계산하여 표로 정리
3. 전체 평균 pass@1 및 pass@10 계산
4. 다음 해석 질문에 답하세요:
   - pass@1과 pass@10의 차이가 실제 배포 시 무엇을 의미하는가?
   - problem_4 (c=1)에서 pass@10 = 1.0인 이유는?

<details>
<summary>정답 보기</summary>

```python
from math import comb

def pass_at_k(n: int, c: int, k: int) -> float:
    """
    n: 총 샘플 수
    c: 정답 샘플 수
    k: 허용 시도 횟수
    """
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)

results = {
    "problem_1": (10, 10),
    "problem_2": (10, 7),
    "problem_3": (10, 3),
    "problem_4": (10, 1),
    "problem_5": (10, 0),
}

print(f"{'문제':<12} {'pass@1':>8} {'pass@3':>8} {'pass@10':>9}")
print("-" * 40)
for name, (n, c) in results.items():
    p1  = pass_at_k(n, c, 1)
    p3  = pass_at_k(n, c, 3)
    p10 = pass_at_k(n, c, 10)
    print(f"{name:<12} {p1:>8.3f} {p3:>8.3f} {p10:>9.3f}")

# 문제         pass@1   pass@3   pass@10
# ----------------------------------------
# problem_1     1.000    1.000     1.000
# problem_2     0.700    0.933     1.000
# problem_3     0.300    0.633     1.000
# problem_4     0.100    0.267     1.000
# problem_5     0.000    0.000     0.000
```

**해석:**

| 지표 | 의미 | 활용 |
|------|------|------|
| pass@1 | 한 번 생성했을 때 정답일 확률 | 배포 신뢰도 (실제 사용자 경험) |
| pass@10 | 10번 시도 중 한 번이라도 정답일 확률 | 모델 능력 상한선 |

- **pass@10 = 1.0 for problem_4**: c=1이므로, 10번 모두 시도하면 그 정답이 반드시 포함됨 (k=n이면 c≥1일 때 항상 1.0)
- **배포 의미**: pass@1이 낮고 pass@10이 높은 문제는 모델이 "풀 수 있지만 자주 실패"하는 케이스 → 재시도(retry) 전략이 효과적

</details>

---

### 연습 문제 3: LLM-as-Judge 편향 완화

LLM 심판(judge)은 응답 순서에 따라 편향된 판단을 내리는 경향이 있습니다(위치 편향, Position Bias). 두 번의 평가(순서 바꾸기)를 통해 편향을 감지하는 공정한 심판 함수를 구현하세요.

```python
from enum import Enum

class JudgeResult(Enum):
    A_WINS = "A"
    B_WINS = "B"
    TIE    = "Tie"

def debiased_judge(question: str, response_a: str, response_b: str) -> dict:
    """
    두 번 평가하여 위치 편향을 감지합니다.
    - 1차: A를 'First', B를 'Second'로 제시
    - 2차: B를 'First', A를 'Second'로 제시
    - 두 결과가 일치하면 high confidence
    - 불일치하면 Tie (위치 편향 감지)

    반환: {
        "winner": JudgeResult,
        "confidence": str,
        "eval_1": str,
        "eval_2_normalized": str,
    }
    """
    # TODO: 구현하세요
    pass
```

요구사항:
1. `single_comparison(question, first, second, label_first, label_second) -> JudgeResult` 헬퍼 함수 구현 (LLM 호출 모의)
2. `debiased_judge` 완성: 두 번 평가 후 결과 정규화 및 비교
3. 세 가지 시나리오 테스트: 일관된 A 승리, 일관된 B 승리, 편향 감지(불일치)

<details>
<summary>정답 보기</summary>

```python
from enum import Enum
import anthropic

class JudgeResult(Enum):
    A_WINS = "A"
    B_WINS = "B"
    TIE    = "Tie"

client = anthropic.Anthropic()

def single_comparison(
    question: str,
    first: str, second: str,
    label_first: str, label_second: str
) -> JudgeResult:
    prompt = f"""질문: {question}

응답 {label_first}: {first}

응답 {label_second}: {second}

어느 응답이 더 도움이 됩니까? "{label_first}", "{label_second}", 또는 "동점" 중 하나만 답하세요."""

    result = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=10,
        messages=[{"role": "user", "content": prompt}]
    )
    verdict = result.content[0].text.strip()

    if label_first in verdict:
        return JudgeResult.A_WINS   # first = A
    elif label_second in verdict:
        return JudgeResult.B_WINS   # second = B
    else:
        return JudgeResult.TIE

def debiased_judge(question: str, response_a: str, response_b: str) -> dict:
    # 1차 평가: A=First, B=Second
    eval_1 = single_comparison(question, response_a, response_b, "First", "Second")
    # A가 이기면 A_WINS, B가 이기면 B_WINS

    # 2차 평가: B=First, A=Second (순서 반전)
    eval_2_raw = single_comparison(question, response_b, response_a, "First", "Second")
    # eval_2_raw에서 "First 승" = B가 더 나음 = B_WINS
    # eval_2_raw에서 "Second 승" = A가 더 나음 = A_WINS
    if eval_2_raw == JudgeResult.A_WINS:   # "First"(=B) 승
        eval_2_normalized = JudgeResult.B_WINS
    elif eval_2_raw == JudgeResult.B_WINS: # "Second"(=A) 승
        eval_2_normalized = JudgeResult.A_WINS
    else:
        eval_2_normalized = JudgeResult.TIE

    # 결과 비교
    if eval_1 == eval_2_normalized:
        final = eval_1
        confidence = "high"
    else:
        final = JudgeResult.TIE
        confidence = "low (위치 편향 감지됨)"

    return {
        "winner": final,
        "confidence": confidence,
        "eval_1": eval_1.value,
        "eval_2_normalized": eval_2_normalized.value,
    }

# 테스트
q = "Python에서 리스트를 역순으로 만드는 방법은?"
good = "list.reverse()를 사용하거나 슬라이싱 lst[::-1]을 사용합니다."
poor = "반복문으로 새 리스트를 만드세요."

result = debiased_judge(q, good, poor)
print(f"승자: {result['winner'].value}")
print(f"신뢰도: {result['confidence']}")
print(f"1차 평가: {result['eval_1']}, 2차 평가(정규화): {result['eval_2_normalized']}")
```

**핵심 개념:**
- **위치 편향(Position Bias)**: LLM은 첫 번째로 제시된 응답을 선호하는 경향이 있음
- **이중 평가**: 순서를 바꿔 두 번 평가하면 편향 여부를 감지할 수 있음
- **신뢰도**: 두 평가가 일치하면 high, 불일치하면 위치 편향으로 간주하고 Tie 처리

</details>

---

## 학습 완료

이것으로 LLM & NLP 심화 학습을 완료했습니다!

### 전체 학습 요약

1. **NLP 기초 (01-03)**: 토큰화, 임베딩, Transformer
2. **사전학습 모델 (04-07)**: BERT, GPT, HuggingFace, 파인튜닝
3. **LLM 활용 (08-12)**: 프롬프트, RAG, LangChain, 벡터 DB, 챗봇
4. **LLM 심화 (13-16)**: 양자화, RLHF, 에이전트, 평가

### 다음 단계 추천

- 실제 프로젝트에 적용
- Kaggle NLP 대회 참가
- 최신 LLM 논문 읽기 (Claude, Gemini, Llama)
- 오픈소스 LLM 기여
