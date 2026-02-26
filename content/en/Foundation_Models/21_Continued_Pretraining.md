# 21. Continued Pre-training

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain why continued pre-training is necessary for domain adaptation and how it differs from instruction tuning and standard fine-tuning
2. Design a continued pre-training pipeline that sequences domain pre-training followed by instruction tuning for a target domain
3. Identify and mitigate catastrophic forgetting by applying data mixing strategies that blend domain and general-purpose corpora
4. Implement continued pre-training on a base LLM using large-scale domain data with appropriate learning rate scheduling
5. Evaluate the impact of continued pre-training by comparing domain-specific benchmarks before and after adaptation

---

## Overview

Continued Pre-training is a method of further training existing pre-trained models to adapt them to specific domains or tasks. Unlike typical fine-tuning, it performs language modeling on large amounts of domain data.

---

## 1. Continued Pre-training Overview

### 1.1 Why Is It Needed?

```
Scenario:
┌─────────────────────────────────────────────────────────┐
│  Base Model (LLaMA-7B)                                  │
│  - Training: General web text                           │
│  - Strength: General language understanding             │
│  - Weakness: Lacking domain-specific knowledge          │
│                                                         │
│  Target Domain: Medical                                 │
│  - Specialized terminology (drug names, diseases)       │
│  - Domain-specific reasoning                            │
│  - Special document formats                             │
└─────────────────────────────────────────────────────────┘

Solution:
1. Instruction Tuning alone is insufficient for knowledge injection
2. Learn domain knowledge through Continued Pre-training
3. Then apply Instruction Tuning for task adaptation
```

### 1.2 Training Pipeline

```
General Pipeline:
┌─────────────────────────────────────────────────────────┐
│                                                         │
│  Pre-trained Model                                      │
│         ↓                                              │
│  [Continued Pre-training]                              │
│  - Domain data (10B+ tokens)                            │
│  - Causal LM objective                                  │
│  - Lower learning rate                                  │
│         ↓                                              │
│  Domain-Adapted Model                                   │
│         ↓                                              │
│  [Instruction Tuning]                                  │
│  - Domain-specific instructions                        │
│         ↓                                              │
│  Final Domain Model                                     │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## 2. Catastrophic Forgetting

### 2.1 Problem Definition

```
Catastrophic Forgetting:
The phenomenon of forgetting existing knowledge while learning new knowledge

Example:
┌────────────────────────────────────────┐
│  Before CPT:                           │
│  Q: "What is the capital of France?"   │
│  A: "Paris"  ✓                         │
│                                        │
│  After CPT (medical domain):           │
│  Q: "What is the capital of France?"   │
│  A: "The patient presented with..."  ✗ │
└────────────────────────────────────────┘
```

### 2.2 Mitigation Strategies

```python
import torch
import torch.nn as nn
from typing import Dict, List, Optional

class ContinuedPretrainingWithRegularization:
    """Training with Catastrophic Forgetting mitigation"""

    def __init__(
        self,
        model: nn.Module,
        reference_model: nn.Module,  # Frozen original
        reg_weight: float = 0.1
    ):
        self.model = model
        self.reference_model = reference_model
        self.reg_weight = reg_weight

        # Reference model freeze
        for param in self.reference_model.parameters():
            param.requires_grad = False

    def compute_loss(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        regularization: str = "kl"
    ) -> torch.Tensor:
        """
        Regularization methods:
        - "kl": KL divergence with reference model
        - "ewc": Elastic Weight Consolidation
        - "replay": Experience replay (separate implementation)
        """
        # Main loss
        outputs = self.model(input_ids, labels=labels)
        lm_loss = outputs.loss

        # Regularization
        if regularization == "kl":
            reg_loss = self._kl_regularization(input_ids)
        elif regularization == "ewc":
            reg_loss = self._ewc_regularization()
        else:
            reg_loss = 0.0

        return lm_loss + self.reg_weight * reg_loss

    def _kl_regularization(self, input_ids: torch.Tensor) -> torch.Tensor:
        """KL divergence-based regularization"""
        with torch.no_grad():
            ref_logits = self.reference_model(input_ids).logits

        current_logits = self.model(input_ids).logits

        # KL(current || reference)
        kl_loss = nn.functional.kl_div(
            nn.functional.log_softmax(current_logits, dim=-1),
            nn.functional.softmax(ref_logits, dim=-1),
            reduction="batchmean"
        )

        return kl_loss

    def _ewc_regularization(self) -> torch.Tensor:
        """
        Elastic Weight Consolidation

        L_ewc = Σᵢ Fᵢ(θᵢ - θᵢ*)²

        Fᵢ: Fisher information (importance)
        θᵢ*: original parameters
        """
        if not hasattr(self, 'fisher_info'):
            # Fisher information needs to be pre-computed
            return torch.tensor(0.0)

        ewc_loss = 0.0
        for name, param in self.model.named_parameters():
            if name in self.fisher_info:
                ewc_loss += (
                    self.fisher_info[name] *
                    (param - self.original_params[name]).pow(2)
                ).sum()

        return ewc_loss

    def compute_fisher_information(
        self,
        dataloader,
        num_samples: int = 1000
    ):
        """Compute Fisher Information"""
        self.fisher_info = {}
        self.original_params = {}

        # Save original parameters
        for name, param in self.model.named_parameters():
            self.original_params[name] = param.clone().detach()
            self.fisher_info[name] = torch.zeros_like(param)

        self.model.eval()
        for i, batch in enumerate(dataloader):
            if i >= num_samples:
                break

            input_ids = batch["input_ids"]
            outputs = self.model(input_ids)
            log_probs = nn.functional.log_softmax(outputs.logits, dim=-1)

            # Sample from output distribution
            sampled = torch.multinomial(
                log_probs.view(-1, log_probs.size(-1)).exp(), 1
            )

            # Compute gradients
            loss = -log_probs.view(-1, log_probs.size(-1)).gather(1, sampled).mean()
            loss.backward()

            # Accumulate squared gradients
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    self.fisher_info[name] += param.grad.pow(2)

            self.model.zero_grad()

        # Normalize
        for name in self.fisher_info:
            self.fisher_info[name] /= num_samples
```

### 2.3 Experience Replay

```python
class ExperienceReplayTrainer:
    """Prevent forgetting with Experience Replay"""

    def __init__(
        self,
        model: nn.Module,
        domain_dataloader,
        general_dataloader,  # General data
        replay_ratio: float = 0.1
    ):
        self.model = model
        self.domain_dataloader = domain_dataloader
        self.general_dataloader = general_dataloader
        self.replay_ratio = replay_ratio

    def train_step(self, optimizer) -> Dict[str, float]:
        """Mixed training with domain + general data"""
        # Domain data
        domain_batch = next(iter(self.domain_dataloader))
        domain_loss = self._compute_lm_loss(domain_batch)

        # Replay (general data)
        if torch.rand(1).item() < self.replay_ratio:
            general_batch = next(iter(self.general_dataloader))
            replay_loss = self._compute_lm_loss(general_batch)
            total_loss = domain_loss + replay_loss
        else:
            replay_loss = torch.tensor(0.0)
            total_loss = domain_loss

        # Backward
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        return {
            "domain_loss": domain_loss.item(),
            "replay_loss": replay_loss.item() if isinstance(replay_loss, torch.Tensor) else 0.0
        }

    def _compute_lm_loss(self, batch) -> torch.Tensor:
        outputs = self.model(
            input_ids=batch["input_ids"],
            labels=batch["labels"]
        )
        return outputs.loss
```

---

## 3. Data Preparation

### 3.1 Domain Data Collection

```python
class DomainDataPipeline:
    """Domain data preprocessing pipeline"""

    def __init__(self, domain: str):
        self.domain = domain
        self.quality_filters = []

    def add_filter(self, filter_fn):
        self.quality_filters.append(filter_fn)

    def process_document(self, doc: str) -> Optional[str]:
        """Document preprocessing"""
        # Basic cleaning
        doc = self._clean_text(doc)

        # Quality filtering
        for filter_fn in self.quality_filters:
            if not filter_fn(doc):
                return None

        return doc

    def _clean_text(self, text: str) -> str:
        """Text cleaning"""
        import re

        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)

        # Normalize special characters
        text = re.sub(r'\s+', ' ', text)

        # Domain-specific cleaning
        if self.domain == "medical":
            # Anonymize patient information
            text = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', '[ID]', text)  # SSN pattern

        return text.strip()


# Quality filter examples
def length_filter(min_len: int = 100, max_len: int = 100000):
    def filter_fn(doc):
        return min_len <= len(doc) <= max_len
    return filter_fn

def language_filter(target_lang: str = "en"):
    def filter_fn(doc):
        from langdetect import detect
        try:
            return detect(doc) == target_lang
        except:
            return False
    return filter_fn

def perplexity_filter(model, tokenizer, max_ppl: float = 100):
    """Filter low-quality (high perplexity) documents"""
    def filter_fn(doc):
        inputs = tokenizer(doc, return_tensors="pt", truncation=True)
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
        ppl = torch.exp(outputs.loss).item()
        return ppl < max_ppl
    return filter_fn
```

### 3.2 Data Mixing Strategy

```python
class CurriculumDataMixer:
    """Curriculum learning-based data mixing"""

    def __init__(
        self,
        domain_data: List[str],
        general_data: List[str],
        total_steps: int
    ):
        self.domain_data = domain_data
        self.general_data = general_data
        self.total_steps = total_steps

    def get_mix_ratio(self, current_step: int) -> float:
        """
        Progressively increase domain data ratio

        Step 0: 50% domain, 50% general
        Step T: 90% domain, 10% general
        """
        progress = current_step / self.total_steps
        domain_ratio = 0.5 + 0.4 * progress  # 0.5 → 0.9
        return domain_ratio

    def sample_batch(self, batch_size: int, current_step: int) -> List[str]:
        """Sample batch appropriate for current step"""
        domain_ratio = self.get_mix_ratio(current_step)
        num_domain = int(batch_size * domain_ratio)
        num_general = batch_size - num_domain

        batch = (
            random.sample(self.domain_data, min(num_domain, len(self.domain_data))) +
            random.sample(self.general_data, min(num_general, len(self.general_data)))
        )

        random.shuffle(batch)
        return batch
```

---

## 4. Training Configuration

### 4.1 Learning Rate Strategy

```python
from transformers import get_scheduler

def get_cpt_lr_scheduler(
    optimizer,
    num_training_steps: int,
    warmup_ratio: float = 0.03,
    min_lr_ratio: float = 0.1
):
    """
    LR scheduler for Continued Pre-training

    - Low initial LR (prevent base model damage)
    - Long warmup
    - Cosine decay
    """
    num_warmup_steps = int(num_training_steps * warmup_ratio)

    scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )

    return scheduler


# Recommended hyperparameters
CPT_CONFIG = {
    "learning_rate": 1e-5,  # Lower LR than base model
    "warmup_ratio": 0.03,
    "weight_decay": 0.01,
    "max_grad_norm": 1.0,
    "batch_size": 256,  # Large batch for stability
    "gradient_accumulation_steps": 16,
    "num_epochs": 1,  # Usually 1 epoch is sufficient
}
```

### 4.2 Checkpointing Strategy

```python
class CPTCheckpointer:
    """Continued Pre-training checkpointer"""

    def __init__(
        self,
        model,
        save_dir: str,
        eval_dataloader,
        save_steps: int = 1000,
        keep_last_n: int = 3
    ):
        self.model = model
        self.save_dir = save_dir
        self.eval_dataloader = eval_dataloader
        self.save_steps = save_steps
        self.keep_last_n = keep_last_n
        self.saved_checkpoints = []
        self.best_ppl = float('inf')

    def maybe_save(self, step: int, loss: float):
        """Conditional save"""
        if step % self.save_steps == 0:
            # Evaluate
            ppl = self._evaluate()

            # Save
            ckpt_path = f"{self.save_dir}/checkpoint-{step}"
            self.model.save_pretrained(ckpt_path)
            self.saved_checkpoints.append((step, ppl, ckpt_path))

            # Update best
            if ppl < self.best_ppl:
                self.best_ppl = ppl
                best_path = f"{self.save_dir}/best"
                self.model.save_pretrained(best_path)
                print(f"New best: ppl={ppl:.2f}")

            # Delete old checkpoints
            self._cleanup_old_checkpoints()

    def _evaluate(self) -> float:
        """Perplexity evaluation"""
        self.model.eval()
        total_loss = 0
        total_tokens = 0

        with torch.no_grad():
            for batch in self.eval_dataloader:
                outputs = self.model(**batch)
                total_loss += outputs.loss.item() * batch["input_ids"].numel()
                total_tokens += batch["input_ids"].numel()

        self.model.train()
        ppl = math.exp(total_loss / total_tokens)
        return ppl

    def _cleanup_old_checkpoints(self):
        """Delete old checkpoints"""
        if len(self.saved_checkpoints) > self.keep_last_n:
            # Sort by PPL
            sorted_ckpts = sorted(self.saved_checkpoints, key=lambda x: x[1])
            to_keep = sorted_ckpts[:self.keep_last_n]
            to_remove = set(self.saved_checkpoints) - set(to_keep)

            for _, _, path in to_remove:
                if os.path.exists(path):
                    shutil.rmtree(path)

            self.saved_checkpoints = list(to_keep)
```

---

## 5. Domain-Specific Examples

### 5.1 Medical Domain

```python
class MedicalCPT:
    """Medical domain Continued Pre-training"""

    def __init__(self, base_model_name: str):
        self.model = AutoModelForCausalLM.from_pretrained(base_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    def prepare_medical_data(self, sources: List[str]) -> List[str]:
        """Prepare medical data"""
        processed = []

        for source in sources:
            if source == "pubmed":
                # PubMed abstracts
                data = self._load_pubmed()
            elif source == "clinical_notes":
                # Clinical notes (anonymized)
                data = self._load_clinical_notes()
            elif source == "medical_textbooks":
                # Medical textbooks
                data = self._load_textbooks()

            processed.extend(data)

        return processed

    def _load_pubmed(self) -> List[str]:
        """Load PubMed data"""
        from datasets import load_dataset

        dataset = load_dataset("pubmed", split="train")
        return [ex["abstract"] for ex in dataset if len(ex["abstract"]) > 100]

    def train(self, data: List[str], output_dir: str):
        """Run training"""
        # Tokenize
        def tokenize(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                max_length=2048
            )

        dataset = Dataset.from_dict({"text": data})
        tokenized = dataset.map(tokenize, batched=True)

        # Training
        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=16,
            learning_rate=5e-6,  # Low LR
            num_train_epochs=1,
            warmup_ratio=0.05,
            lr_scheduler_type="cosine",
            logging_steps=100,
            save_steps=500,
            fp16=True
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized,
            data_collator=DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer, mlm=False
            )
        )

        trainer.train()
```

### 5.2 Code Domain

```python
class CodeCPT:
    """Code domain Continued Pre-training"""

    def prepare_code_data(self) -> List[str]:
        """Prepare code data"""
        from datasets import load_dataset

        # The Stack
        dataset = load_dataset(
            "bigcode/the-stack",
            data_dir="data/python",
            split="train",
            streaming=True
        )

        processed = []
        for example in dataset:
            code = example["content"]

            # Quality filtering
            if self._is_quality_code(code):
                processed.append(code)

            if len(processed) >= 1000000:  # 1M samples
                break

        return processed

    def _is_quality_code(self, code: str) -> bool:
        """Code quality check"""
        # Length
        if len(code) < 50 or len(code) > 100000:
            return False

        # Comment ratio
        lines = code.split("\n")
        comment_lines = sum(1 for l in lines if l.strip().startswith("#"))
        if len(lines) > 0 and comment_lines / len(lines) > 0.5:
            return False

        # Syntax check
        try:
            import ast
            ast.parse(code)
            return True
        except SyntaxError:
            return False
```

---

## Key Summary

### Continued Pre-training Core
```
1. Purpose: Domain knowledge injection
2. Data: Large amounts of domain text (10B+ tokens)
3. Method: Causal LM objective
4. Caution: Catastrophic forgetting
```

### Forgetting Mitigation Strategies
```
1. KL Regularization: Minimize KL with reference model
2. EWC: Preserve important parameters
3. Experience Replay: Mix general data
4. Curriculum: Progressive domain ratio increase
```

### Training Recommendations
```
- Learning Rate: 1/10 ~ 1/5 of base
- Warmup: 3-5%
- Batch Size: Large batch (256+)
- Epochs: 1 epoch
- Checkpointing: Save frequently, monitor perplexity
```

---

## References

1. Gururangan et al. (2020). "Don't Stop Pretraining: Adapt Language Models to Domains and Tasks"
2. Ke et al. (2023). "Continual Pre-training of Language Models"
3. Ibrahim et al. (2024). "Simple and Scalable Strategies to Continually Pre-train Large Language Models"
4. Xie et al. (2023). "Efficient Continual Pre-training for Building Domain Specific Large Language Models"

---

## Exercises

### Exercise 1: Continued Pre-training vs. Instruction Tuning
A biomedical AI company wants to build a model that answers clinician questions about drug interactions. They have two resources: (A) 50GB of PubMed abstracts and clinical trial reports, and (B) 5,000 curated question-answer pairs from expert pharmacologists.

Explain why the company should do both continued pre-training AND instruction tuning rather than either alone. Describe the ideal training pipeline order.

<details>
<summary>Show Answer</summary>

**Why instruction tuning alone is insufficient**:
- The base model (e.g., LLaMA-7B) knows very little about drug interactions, rare side effects, drug-drug interaction mechanisms, or pharmacokinetic terminology.
- 5,000 QA pairs can teach the model to answer in the right format, but they cannot inject the vast factual knowledge needed to answer novel questions correctly.
- The model would learn to pattern-match the QA format ("Q: What are the side effects of X? A: The side effects of X are...") while hallucinating the actual medical content.
- Example failure: "What is the CYP3A4 interaction between ketoconazole and midazolam?" — the base model lacks this pharmacological knowledge; 5K QA pairs are insufficient to teach all possible drug pair interactions.

**Why continued pre-training alone is insufficient**:
- After pre-training on PubMed abstracts, the model understands biomedical language but still behaves as a text completion engine.
- Asked "What are the contraindications of warfarin?", it might continue: "...warfarin has several documented contraindications, including... (continues with a different paper excerpt)" rather than giving a structured clinical answer.
- It cannot follow instructions, answer questions concisely, or format responses appropriately for clinical use.

**Ideal pipeline**:
```
Base LLM (LLaMA-7B)
    ↓
[Phase 1] Continued Pre-training
    - Data: 50GB PubMed + clinical trials
    - Objective: Causal LM on domain text
    - LR: 5e-6 (low, 1/5 of base LR)
    - Goal: Inject pharmacological knowledge
    ↓
Domain-adapted LLM (knows pharmacology)
    ↓
[Phase 2] Supervised Fine-tuning (SFT)
    - Data: 5,000 clinical QA pairs
    - Objective: Instruction following
    - LR: 1e-5
    - Goal: Format responses for clinical use
    ↓
Medical Instruction-Following LLM
    ↓
[Optional Phase 3] RLHF/DPO
    - Data: Clinical expert preferences
    - Goal: Align with clinical best practices
```

The sequence matters: you can't instruction-tune first (the model has no domain knowledge to apply to instructions) and you can't skip instruction tuning (domain pre-training doesn't teach the model to be helpful).

</details>

### Exercise 2: Catastrophic Forgetting Data Mixing
During continued pre-training on domain data, catastrophic forgetting can degrade the model's general capabilities. A team trains LLaMA-7B on 50GB of medical data and finds that afterward the model:
- Improved: Medical QA accuracy +25%
- Degraded: General reasoning (BBH benchmark) -18%, Coding (HumanEval) -31%

Propose a data mixing strategy to recover the coding and reasoning capabilities while retaining the medical knowledge gain.

<details>
<summary>Show Answer</summary>

**Analysis of degradation**:
- **Coding -31%**: Severe degradation suggests Python/code syntax was heavily displaced by medical terminology. The model "forgot" programming patterns.
- **Reasoning -18%**: Moderate degradation — general reasoning partially displaced but not catastrophic.
- **Root cause**: The 50GB medical dataset contained nearly zero code or mathematical reasoning. The model's representations were overwritten by medical text patterns.

**Data Mixing Strategy**:

```python
# Replay-based mixing: blend domain data with general data
mixing_proportions = {
    "medical": 0.55,       # Primary target (reduced from 100%)
    "code": 0.20,          # Recovery: code specifically (matches degradation severity)
    "general_text": 0.15,  # General reasoning recovery
    "math": 0.10,          # Supports reasoning capabilities
}
# Total: 1.0

# Training data composition per batch:
# For a 50GB medical corpus:
# - Medical: 50GB × 0.55 = 27.5GB (still enough for domain adaptation)
# - Code (e.g., from The Stack): ~18GB
# - General (e.g., SlimPajama subset): ~14GB
# - Math (e.g., OpenWebMath): ~9GB
```

**Key principles**:
1. **Proportional recovery**: Code gets 20% (vs. reasoning's 15%) because coding degraded more severely (31% vs. 18%).
2. **Medical still dominant** (55%): Maintains the primary domain adaptation goal.
3. **Diverse mixing**: Using multiple non-medical sources prevents the model from over-specializing in any single replacement domain.
4. **Evaluation during training**: Monitor all three benchmarks (medical, coding, reasoning) at regular intervals; stop when coding/reasoning recover to within 5% of baseline while medical performance peaks.

**Expected outcome**: Medical QA: +20% (slight reduction from +25% but still substantial), Coding: -5% (vs. -31% without mixing), Reasoning: -3% (vs. -18% without mixing).

</details>

### Exercise 3: Learning Rate and Data Order Strategy
A team is continuing pre-training on 10B tokens of financial news data. Explain the rationale for using a much lower learning rate than the original pre-training, and describe how data ordering (curriculum) could improve the final model quality.

<details>
<summary>Show Answer</summary>

**Lower learning rate rationale**:

During original pre-training, the model trained from random initialization on ~1T+ tokens. High learning rates were appropriate because:
- Weights started at random values — large updates were needed to build any useful representations.
- The large dataset provided enough gradient signal to justify aggressive updates.

During continued pre-training:
- The model already has well-calibrated weights representing excellent general language understanding.
- High learning rate → large weight updates → catastrophic forgetting of general capabilities (the existing good representations are overwritten).
- Low learning rate (e.g., 5e-6 vs. original 3e-4) ensures:
  - New domain knowledge is integrated **on top of** existing representations.
  - The model makes small, conservative adjustments rather than overwriting existing circuits.
  - The loss surface for the domain data is explored cautiously around the existing good minimum.

**Warm-up**: Use 3-5% warm-up steps even with low LR — the initial gradient estimates for domain data are noisy and warm-up prevents destabilizing the model early.

**Curriculum learning for financial news**:

```
Recommended ordering:
Phase 1: High-quality, clean text (10%)
   - Well-edited financial reports (10-K, 10-Q filings)
   - Establishes domain vocabulary

Phase 2: Structured domain data (40%)
   - Analyst reports, earnings call transcripts
   - Builds financial reasoning patterns

Phase 3: Diverse domain data (40%)
   - News articles, press releases, market commentary
   - Broad coverage of domain language

Phase 4: Noisy/social data (10%)
   - Reddit/Twitter financial discussions
   - Slang, abbreviations, informal usage
```

**Rationale**: Starting with high-quality, formal text establishes clean representations for financial concepts. Adding noisier data later means the model has a stable foundation to build slang/informal mappings on, rather than confusing them from the start. This is similar to how humans learn formal language before slang.

**Cosine LR decay with restarts**: Use cosine schedule that decays to 1/10 of initial LR, optionally with a brief warm restart at the phase boundaries — this helps the model transition between data distributions without destabilization.

</details>

### Exercise 4: Continued Pre-training Evaluation
After continued pre-training a 7B LLM on legal documents, design a minimal evaluation suite that measures whether the adaptation was successful without regressing general capabilities.

Specify: (A) what metrics to measure, (B) what datasets or tests to use, and (C) a threshold at which you'd consider the training a success.

<details>
<summary>Show Answer</summary>

**Evaluation Suite Design**:

**A) Metrics to measure**:

Domain-specific metrics:
1. **Legal NER F1**: Named entity recognition for legal entities (case names, statutes, parties).
2. **Legal QA accuracy**: Question answering on legal questions (e.g., CUAD dataset — contract clauses).
3. **Legal document perplexity**: Perplexity on held-out legal text (measures language model quality on domain).
4. **Citation format accuracy**: Does the model correctly format legal citations (Bluebook style)?

General capability metrics (regression tests):
5. **BIG-Bench Hard (BBH)**: General reasoning benchmark — 23 challenging reasoning tasks.
6. **MMLU**: 57-subject knowledge benchmark — general knowledge preservation.
7. **HellaSwag**: Commonsense reasoning.
8. **HumanEval** (optional, if coding was in original pretraining): Coding preservation.

**B) Datasets and tests**:

```python
evaluation_suite = {
    # Domain-specific
    "legal_ner": "LexNER or MultiLegalPile NER annotations",
    "legal_qa": "CUAD (Contract Understanding Atticus Dataset) - 510 contracts",
    "legal_perplexity": "Held-out 10% of legal pretraining data",

    # General capabilities (baseline from base model)
    "bbh": "BIG-Bench Hard - 6,511 examples",
    "mmlu": "MMLU - 14,000+ examples, 57 subjects",
    "hellaswag": "HellaSwag - 10,000 validation examples",
}
```

**C) Success thresholds**:

| Metric | Baseline (pre-CPT) | Target (post-CPT) | Pass/Fail |
|--------|-------------------|-------------------|-----------|
| Legal QA accuracy | ~25% (random LLM) | ≥60% | Domain goal |
| Legal document perplexity | ~40 | ≤15 | Language quality |
| BBH accuracy | Measured baseline | ≥ baseline - 5% | Regression tolerance |
| MMLU accuracy | Measured baseline | ≥ baseline - 5% | Regression tolerance |
| HellaSwag | Measured baseline | ≥ baseline - 3% | Regression tolerance |

**Success condition**: ALL of the following:
- Legal QA accuracy ≥ 60%
- BBH regression ≤ 5% (e.g., 45% → 42.75% is acceptable)
- MMLU regression ≤ 5%
- If any general capability drops > 8%, rerun with more replay data (increase general data mixing ratio by 10-15%)

</details>
