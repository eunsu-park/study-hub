# 21. Continued Pre-training

## 개요

Continued Pre-training(지속 사전학습)은 기존 pre-trained 모델을 특정 도메인이나 태스크에 맞게 추가 학습하는 방법입니다. 일반적인 fine-tuning과 달리 대량의 도메인 데이터로 language modeling을 수행합니다.

---

## 1. Continued Pre-training 개요

### 1.1 왜 필요한가?

```
시나리오:
┌─────────────────────────────────────────────────────────┐
│  Base Model (LLaMA-7B)                                  │
│  - 학습: 일반 웹 텍스트                                  │
│  - 강점: 일반적인 언어 이해                              │
│  - 약점: 도메인 특화 지식 부족                           │
│                                                         │
│  목표 도메인: 의료                                       │
│  - 전문 용어 (약물명, 질병명)                            │
│  - 도메인 특화 추론                                      │
│  - 특수 문서 형식                                        │
└─────────────────────────────────────────────────────────┘

해결책:
1. Instruction Tuning만으로는 지식 주입 어려움
2. Continued Pre-training으로 도메인 지식 학습
3. 이후 Instruction Tuning으로 태스크 적응
```

### 1.2 학습 파이프라인

```
일반적인 파이프라인:
┌─────────────────────────────────────────────────────────┐
│                                                         │
│  Pre-trained Model                                      │
│         ↓                                              │
│  [Continued Pre-training]                              │
│  - 도메인 데이터 (10B+ tokens)                          │
│  - Causal LM objective                                  │
│  - Lower learning rate                                  │
│         ↓                                              │
│  Domain-Adapted Model                                   │
│         ↓                                              │
│  [Instruction Tuning]                                  │
│  - 도메인 특화 instructions                            │
│         ↓                                              │
│  Final Domain Model                                     │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## 2. Catastrophic Forgetting

### 2.1 문제 정의

```
Catastrophic Forgetting:
새로운 지식을 학습하면서 기존 지식을 잊어버리는 현상

예시:
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

### 2.2 완화 전략

```python
import torch
import torch.nn as nn
from typing import Dict, List, Optional

class ContinuedPretrainingWithRegularization:
    """Catastrophic Forgetting 완화를 위한 학습"""

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
        - "replay": Experience replay (별도 구현)
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
        """KL divergence 기반 정규화"""
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
            # Fisher information 사전 계산 필요
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
        """Fisher Information 계산"""
        self.fisher_info = {}
        self.original_params = {}

        # Original parameters 저장
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
    """Experience Replay로 forgetting 방지"""

    def __init__(
        self,
        model: nn.Module,
        domain_dataloader,
        general_dataloader,  # 일반 데이터
        replay_ratio: float = 0.1
    ):
        self.model = model
        self.domain_dataloader = domain_dataloader
        self.general_dataloader = general_dataloader
        self.replay_ratio = replay_ratio

    def train_step(self, optimizer) -> Dict[str, float]:
        """도메인 + 일반 데이터 혼합 학습"""
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

## 3. 데이터 준비

### 3.1 도메인 데이터 수집

```python
class DomainDataPipeline:
    """도메인 데이터 전처리 파이프라인"""

    def __init__(self, domain: str):
        self.domain = domain
        self.quality_filters = []

    def add_filter(self, filter_fn):
        self.quality_filters.append(filter_fn)

    def process_document(self, doc: str) -> Optional[str]:
        """문서 전처리"""
        # 기본 정제
        doc = self._clean_text(doc)

        # 품질 필터링
        for filter_fn in self.quality_filters:
            if not filter_fn(doc):
                return None

        return doc

    def _clean_text(self, text: str) -> str:
        """텍스트 정제"""
        import re

        # HTML 태그 제거
        text = re.sub(r'<[^>]+>', '', text)

        # 특수 문자 정규화
        text = re.sub(r'\s+', ' ', text)

        # 도메인 특화 정제
        if self.domain == "medical":
            # 환자 정보 익명화
            text = re.sub(r'\b\d{6}-\d{7}\b', '[ID]', text)  # 주민번호 패턴

        return text.strip()


# 품질 필터 예시
def length_filter(min_len: int = 100, max_len: int = 100000):
    def filter_fn(doc):
        return min_len <= len(doc) <= max_len
    return filter_fn

def language_filter(target_lang: str = "ko"):
    def filter_fn(doc):
        from langdetect import detect
        try:
            return detect(doc) == target_lang
        except:
            return False
    return filter_fn

def perplexity_filter(model, tokenizer, max_ppl: float = 100):
    """품질이 낮은 (perplexity 높은) 문서 필터링"""
    def filter_fn(doc):
        inputs = tokenizer(doc, return_tensors="pt", truncation=True)
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
        ppl = torch.exp(outputs.loss).item()
        return ppl < max_ppl
    return filter_fn
```

### 3.2 데이터 믹싱 전략

```python
class CurriculumDataMixer:
    """커리큘럼 학습 기반 데이터 믹싱"""

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
        점진적으로 도메인 데이터 비율 증가

        Step 0: 50% domain, 50% general
        Step T: 90% domain, 10% general
        """
        progress = current_step / self.total_steps
        domain_ratio = 0.5 + 0.4 * progress  # 0.5 → 0.9
        return domain_ratio

    def sample_batch(self, batch_size: int, current_step: int) -> List[str]:
        """현재 step에 맞는 배치 샘플링"""
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

## 4. 학습 설정

### 4.1 Learning Rate 전략

```python
from transformers import get_scheduler

def get_cpt_lr_scheduler(
    optimizer,
    num_training_steps: int,
    warmup_ratio: float = 0.03,
    min_lr_ratio: float = 0.1
):
    """
    Continued Pre-training용 LR 스케줄러

    - 낮은 초기 LR (base model 손상 방지)
    - 긴 warmup
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


# 권장 하이퍼파라미터
CPT_CONFIG = {
    "learning_rate": 1e-5,  # Base model 대비 낮은 LR
    "warmup_ratio": 0.03,
    "weight_decay": 0.01,
    "max_grad_norm": 1.0,
    "batch_size": 256,  # Large batch for stability
    "gradient_accumulation_steps": 16,
    "num_epochs": 1,  # 보통 1 epoch면 충분
}
```

### 4.2 체크포인팅 전략

```python
class CPTCheckpointer:
    """Continued Pre-training 체크포인터"""

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
        """조건부 저장"""
        if step % self.save_steps == 0:
            # 평가
            ppl = self._evaluate()

            # 저장
            ckpt_path = f"{self.save_dir}/checkpoint-{step}"
            self.model.save_pretrained(ckpt_path)
            self.saved_checkpoints.append((step, ppl, ckpt_path))

            # Best 업데이트
            if ppl < self.best_ppl:
                self.best_ppl = ppl
                best_path = f"{self.save_dir}/best"
                self.model.save_pretrained(best_path)
                print(f"New best: ppl={ppl:.2f}")

            # 오래된 체크포인트 삭제
            self._cleanup_old_checkpoints()

    def _evaluate(self) -> float:
        """Perplexity 평가"""
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
        """오래된 체크포인트 삭제"""
        if len(self.saved_checkpoints) > self.keep_last_n:
            # PPL 기준 정렬
            sorted_ckpts = sorted(self.saved_checkpoints, key=lambda x: x[1])
            to_keep = sorted_ckpts[:self.keep_last_n]
            to_remove = set(self.saved_checkpoints) - set(to_keep)

            for _, _, path in to_remove:
                if os.path.exists(path):
                    shutil.rmtree(path)

            self.saved_checkpoints = list(to_keep)
```

---

## 5. 도메인별 예시

### 5.1 의료 도메인

```python
class MedicalCPT:
    """의료 도메인 Continued Pre-training"""

    def __init__(self, base_model_name: str):
        self.model = AutoModelForCausalLM.from_pretrained(base_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    def prepare_medical_data(self, sources: List[str]) -> List[str]:
        """의료 데이터 준비"""
        processed = []

        for source in sources:
            if source == "pubmed":
                # PubMed abstracts
                data = self._load_pubmed()
            elif source == "clinical_notes":
                # 임상 노트 (익명화)
                data = self._load_clinical_notes()
            elif source == "medical_textbooks":
                # 의학 교과서
                data = self._load_textbooks()

            processed.extend(data)

        return processed

    def _load_pubmed(self) -> List[str]:
        """PubMed 데이터 로드"""
        from datasets import load_dataset

        dataset = load_dataset("pubmed", split="train")
        return [ex["abstract"] for ex in dataset if len(ex["abstract"]) > 100]

    def train(self, data: List[str], output_dir: str):
        """학습 실행"""
        # 토크나이징
        def tokenize(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                max_length=2048
            )

        dataset = Dataset.from_dict({"text": data})
        tokenized = dataset.map(tokenize, batched=True)

        # 학습
        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=16,
            learning_rate=5e-6,  # 낮은 LR
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

### 5.2 코드 도메인

```python
class CodeCPT:
    """코드 도메인 Continued Pre-training"""

    def prepare_code_data(self) -> List[str]:
        """코드 데이터 준비"""
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

            # 품질 필터링
            if self._is_quality_code(code):
                processed.append(code)

            if len(processed) >= 1000000:  # 1M 샘플
                break

        return processed

    def _is_quality_code(self, code: str) -> bool:
        """코드 품질 검사"""
        # 길이
        if len(code) < 50 or len(code) > 100000:
            return False

        # 주석 비율
        lines = code.split("\n")
        comment_lines = sum(1 for l in lines if l.strip().startswith("#"))
        if len(lines) > 0 and comment_lines / len(lines) > 0.5:
            return False

        # 구문 검사
        try:
            import ast
            ast.parse(code)
            return True
        except SyntaxError:
            return False
```

---

## 핵심 정리

### Continued Pre-training 핵심
```
1. 목적: 도메인 지식 주입
2. 데이터: 대량의 도메인 텍스트 (10B+ tokens)
3. 방법: Causal LM objective
4. 주의: Catastrophic forgetting
```

### Forgetting 완화 전략
```
1. KL Regularization: reference 모델과의 KL 최소화
2. EWC: 중요 파라미터 보존
3. Experience Replay: 일반 데이터 혼합
4. Curriculum: 점진적 도메인 비율 증가
```

### 학습 권장 사항
```
- Learning Rate: base의 1/10 ~ 1/5
- Warmup: 3-5%
- Batch Size: 큰 배치 (256+)
- Epochs: 1 epoch
- Checkpointing: 자주 저장, perplexity 모니터링
```

---

## 참고 자료

1. Gururangan et al. (2020). "Don't Stop Pretraining: Adapt Language Models to Domains and Tasks"
2. Ke et al. (2023). "Continual Pre-training of Language Models"
3. Ibrahim et al. (2024). "Simple and Scalable Strategies to Continually Pre-train Large Language Models"
4. Xie et al. (2023). "Efficient Continual Pre-training for Building Domain Specific Large Language Models"
