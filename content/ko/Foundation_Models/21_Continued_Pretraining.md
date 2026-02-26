# 21. Continued Pre-training

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 도메인 적응(domain adaptation)에 지속 사전학습(Continued Pre-training)이 필요한 이유를 설명하고 인스트럭션 튜닝(instruction tuning) 및 일반 파인튜닝과의 차이를 구별할 수 있다
2. 목표 도메인에 맞게 도메인 사전학습 후 인스트럭션 튜닝을 순차적으로 적용하는 지속 사전학습 파이프라인을 설계할 수 있다
3. 도메인 데이터와 범용 코퍼스를 혼합하는 데이터 믹싱(data mixing) 전략을 적용하여 파국적 망각(catastrophic forgetting)을 파악하고 완화할 수 있다
4. 적절한 학습률 스케줄링(learning rate scheduling)과 함께 대규모 도메인 데이터를 사용하여 베이스 LLM에 지속 사전학습을 구현할 수 있다
5. 적응 전후의 도메인 특화 벤치마크를 비교하여 지속 사전학습의 효과를 평가할 수 있다

---

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

---

## 연습 문제

### 연습 문제 1: 지속 사전 훈련 vs. 지시 튜닝
생의학 AI 회사가 약물 상호작용에 대한 임상의 질문에 답하는 모델을 만들고 싶습니다. 두 가지 리소스가 있습니다: (A) 50GB의 PubMed 초록과 임상시험 보고서, (B) 전문 약리학자가 만든 5,000개의 큐레이션된 질의응답 쌍.

회사가 지속 사전 훈련(Continued Pre-training, CPT)과 지시 튜닝(Instruction Tuning)을 둘 다 해야 하는 이유를 설명하세요. 이상적인 훈련 파이프라인 순서를 서술하세요.

<details>
<summary>정답 보기</summary>

**지시 튜닝만으로 부족한 이유**:
- 기반 모델(예: LLaMA-7B)은 약물 상호작용, 희귀한 부작용, 약물-약물 상호작용 메커니즘, 약동학(pharmacokinetic) 용어에 대해 거의 알지 못합니다.
- 5,000개의 QA 쌍은 올바른 형식으로 답하는 것을 가르칠 수 있지만, 새로운 질문에 정확히 답하는 데 필요한 방대한 사실 지식을 주입할 수는 없습니다.
- 모델은 QA 형식("Q: X의 부작용은? A: X의 부작용은...")을 패턴 매칭하는 법을 학습하면서 실제 의학적 내용은 환각할 것입니다.
- 예시 실패 사례: "케토코나졸과 미다졸람 간의 CYP3A4 상호작용은 무엇인가?" — 기반 모델은 이 약리학적 지식이 없으며, 5K QA 쌍으로는 가능한 모든 약물 쌍 상호작용을 가르치기에 불충분합니다.

**지속 사전 훈련만으로 부족한 이유**:
- PubMed 초록으로 사전 훈련 후, 모델은 생의학 언어를 이해하지만 여전히 텍스트 완성 엔진으로 동작합니다.
- "와파린의 금기 사항은 무엇인가?"라고 물으면, "...와파린에는 여러 가지 문서화된 금기 사항이 있으며, 여기에는... (다른 논문 발췌로 계속)"처럼 대답할 수 있어 임상적 사용을 위한 구조화된 답변을 제공하지 못할 수 있습니다.
- 지시를 따르거나, 간결하게 답변하거나, 임상 사용에 적합하게 응답을 형식화할 수 없습니다.

**이상적인 파이프라인**:
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

순서가 중요합니다: 먼저 지시 튜닝을 할 수 없는 이유는 (적용할 도메인 지식이 없어서) 이고, 지시 튜닝을 건너뛸 수 없는 이유는 (도메인 사전 훈련이 모델을 유용하게 만드는 법을 가르치지 않기 때문)입니다.

</details>

### 연습 문제 2: 재앙적 망각(Catastrophic Forgetting) 데이터 혼합
도메인 데이터에 대한 지속 사전 훈련 중 재앙적 망각(catastrophic forgetting)은 모델의 일반 능력을 저하시킬 수 있습니다. 한 팀이 50GB의 의학 데이터로 LLaMA-7B를 훈련한 후 다음을 발견했습니다:
- 개선: 의학 QA 정확도 +25%
- 저하: 일반 추론(BBH 벤치마크) -18%, 코딩(HumanEval) -31%

코딩과 추론 능력을 회복하면서 의학 지식 향상을 유지할 수 있는 데이터 혼합 전략을 제안하세요.

<details>
<summary>정답 보기</summary>

**저하 분석**:
- **코딩 -31%**: 의학 용어가 Python/코드 구문을 크게 대체했음을 나타내는 심각한 저하. 모델이 프로그래밍 패턴을 "잊었습니다".
- **추론 -18%**: 중간 정도의 저하 — 일반 추론은 부분적으로 대체되었지만 재앙적이지는 않음.
- **근본 원인**: 50GB 의학 데이터셋에는 코드나 수학적 추론이 거의 없었습니다. 모델 표현이 의학 텍스트 패턴으로 덮어써졌습니다.

**데이터 혼합 전략**:

```python
# 리플레이(replay) 기반 혼합: 도메인 데이터와 일반 데이터를 혼합
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

**핵심 원칙**:
1. **비례적 회복**: 코딩이 더 심하게 저하되었으므로 코딩에 20%(추론 15%보다 많은)를 할당합니다.
2. **의학이 여전히 지배적** (55%): 주요 도메인 적응 목표를 유지합니다.
3. **다양한 혼합**: 여러 비의학 소스를 사용하면 모델이 단일 대체 도메인에 과도하게 특화되는 것을 방지합니다.
4. **훈련 중 평가**: 정기적으로 세 가지 벤치마크(의학, 코딩, 추론)를 모두 모니터링하고, 코딩/추론이 기준선의 5% 이내로 회복되면서 의학 성능이 최고점에 달할 때 훈련을 중단합니다.

**예상 결과**: 의학 QA: +20%(+25%에서 약간 감소하지만 여전히 상당함), 코딩: -5%(혼합 없이는 -31%), 추론: -3%(혼합 없이는 -18%).

</details>

### 연습 문제 3: 학습률 및 데이터 순서 전략
한 팀이 100억 토큰의 금융 뉴스 데이터로 지속 사전 훈련을 진행하고 있습니다. 원래 사전 훈련보다 훨씬 낮은 학습률을 사용하는 이유를 설명하고, 데이터 순서(커리큘럼)가 최종 모델 품질을 어떻게 향상시킬 수 있는지 서술하세요.

<details>
<summary>정답 보기</summary>

**낮은 학습률 사용 이유**:

원래 사전 훈련 시, 모델은 무작위 초기화에서 시작해 1T+ 토큰으로 훈련했습니다. 높은 학습률이 적합했던 이유:
- 가중치가 무작위 값에서 시작 — 유용한 표현을 구축하려면 큰 업데이트가 필요했습니다.
- 대용량 데이터셋이 적극적인 업데이트를 정당화하기에 충분한 그래디언트 신호를 제공했습니다.

지속 사전 훈련 시:
- 모델은 이미 훌륭한 일반 언어 이해를 나타내는 잘 보정된 가중치를 갖고 있습니다.
- 높은 학습률 → 큰 가중치 업데이트 → 재앙적 망각 (기존의 좋은 표현이 덮어써짐).
- 낮은 학습률(예: 원래 3e-4 대비 5e-6)은 다음을 보장합니다:
  - 새로운 도메인 지식이 기존 표현 **위에** 통합됩니다.
  - 모델이 기존 회로를 덮어쓰기보다 작고 보수적인 조정을 합니다.
  - 도메인 데이터의 손실 표면(loss surface)이 기존의 좋은 최솟값 주변에서 신중하게 탐색됩니다.

**워밍업**: 낮은 LR에서도 3-5% 워밍업 스텝을 사용하세요 — 도메인 데이터에 대한 초기 그래디언트 추정치가 노이지(noisy)하고 워밍업이 초기에 모델을 불안정하게 만드는 것을 방지합니다.

**금융 뉴스에 대한 커리큘럼 학습**:

```
권장 순서:
Phase 1: 고품질, 정제된 텍스트 (10%)
   - 잘 편집된 재무 보고서 (10-K, 10-Q 제출)
   - 도메인 어휘 확립

Phase 2: 구조화된 도메인 데이터 (40%)
   - 애널리스트 보고서, 실적 발표 전화 기록
   - 금융 추론 패턴 구축

Phase 3: 다양한 도메인 데이터 (40%)
   - 뉴스 기사, 보도 자료, 시장 해설
   - 도메인 언어의 광범위한 커버리지

Phase 4: 노이지/소셜 데이터 (10%)
   - Reddit/Twitter 금융 토론
   - 슬랭, 약어, 비공식 사용법
```

**근거**: 고품질, 공식적인 텍스트로 시작하면 금융 개념에 대한 깔끔한 표현이 확립됩니다. 나중에 더 노이지한 데이터를 추가하면 처음부터 슬랭/비공식 매핑을 혼동하기보다 모델이 안정적인 기반 위에 슬랭/비공식 매핑을 구축할 수 있습니다. 이는 인간이 공식 언어를 먼저 배우고 나서 슬랭을 익히는 방식과 유사합니다.

**재시작이 있는 코사인 LR 감쇠**: 최초 LR의 1/10로 감쇠하는 코사인 스케줄을 사용하고, 선택적으로 단계 경계에서 간단한 웜 재시작을 적용합니다 — 이는 불안정화 없이 모델이 데이터 분포 간 전환을 돕습니다.

</details>

### 연습 문제 4: 지속 사전 훈련 평가
법률 문서로 7B LLM을 지속 사전 훈련한 후, 일반 능력을 퇴보시키지 않으면서 적응이 성공적이었는지 측정하는 최소한의 평가 스위트를 설계하세요.

다음을 명시하세요: (A) 측정할 지표, (B) 사용할 데이터셋 또는 테스트, (C) 훈련이 성공적이었다고 판단할 임계값.

<details>
<summary>정답 보기</summary>

**평가 스위트 설계**:

**A) 측정할 지표**:

도메인 특화 지표:
1. **법률 NER F1**: 법률 엔티티(사건명, 법령, 당사자)에 대한 개체명 인식(Named Entity Recognition).
2. **법률 QA 정확도**: 법률 질문에 대한 질의응답 (예: CUAD 데이터셋 — 계약 조항).
3. **법률 문서 퍼플렉시티(perplexity)**: 보류된 법률 텍스트에 대한 퍼플렉시티 (도메인 언어 모델 품질 측정).
4. **인용 형식 정확도**: 모델이 법률 인용 형식(Bluebook 스타일)을 올바르게 포맷하는지.

일반 능력 지표 (회귀 테스트):
5. **BIG-Bench Hard (BBH)**: 일반 추론 벤치마크 — 23개의 도전적인 추론 태스크.
6. **MMLU**: 57개 과목 지식 벤치마크 — 일반 지식 보존.
7. **HellaSwag**: 상식 추론.
8. **HumanEval** (선택 사항, 원래 사전 훈련에 코딩이 포함된 경우): 코딩 보존.

**B) 데이터셋 및 테스트**:

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

**C) 성공 임계값**:

| 지표 | 기준선 (CPT 이전) | 목표 (CPT 이후) | 성공/실패 |
|--------|-------------------|-------------------|-----------|
| 법률 QA 정확도 | ~25% (무작위 LLM) | ≥60% | 도메인 목표 |
| 법률 문서 퍼플렉시티 | ~40 | ≤15 | 언어 품질 |
| BBH 정확도 | 측정된 기준선 | ≥ 기준선 - 5% | 회귀 허용 범위 |
| MMLU 정확도 | 측정된 기준선 | ≥ 기준선 - 5% | 회귀 허용 범위 |
| HellaSwag | 측정된 기준선 | ≥ 기준선 - 3% | 회귀 허용 범위 |

**성공 조건**: 다음을 모두 만족해야 함:
- 법률 QA 정확도 ≥ 60%
- BBH 회귀 ≤ 5% (예: 45% → 42.75%는 허용 가능)
- MMLU 회귀 ≤ 5%
- 일반 능력이 8% 이상 하락하면 더 많은 리플레이 데이터로 재실행 (일반 데이터 혼합 비율을 10-15% 증가)

</details>
4. Xie et al. (2023). "Efficient Continual Pre-training for Building Domain Specific Large Language Models"
