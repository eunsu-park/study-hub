# 04. Pre-training 목적함수

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 세 가지 주요 사전 학습(Pre-training) 패러다임(Causal LM, Masked LM, Prefix LM)을 비교하고, 각각이 모델의 컨텍스트 접근 방식, 학습 신호 밀도, 생성 대 이해 과제 적합성에 미치는 영향을 설명할 수 있습니다.
2. CLM(Causal Language Modeling)과 MLM(Masked Language Modeling)의 수학적 목적함수를 도출하고, 인과 마스크(Causal Mask)와 마스킹 비율(Masking Rate)의 역할을 설명할 수 있습니다.
3. 스팬 손상(Span Corruption, T5 방식) 및 접두사 CLM(CLM with prefix) 등 고급 목적함수를 설명하고, 각 목적함수를 사용하는 대표 모델을 식별할 수 있습니다.
4. 사전 학습 목적함수 선택이 퓨샷 프롬프팅(Few-shot Prompting), 명령어 따르기(Instruction Following), 제로샷 일반화(Zero-shot Generalization)와 같은 발현 능력(Emergent Capabilities)에 미치는 영향을 분석할 수 있습니다.
5. CLM, MLM, 인코더-디코더 목적함수 선택 시 학습 효율성과 표현 품질 간의 트레이드오프를 평가할 수 있습니다.
6. 지속적 사전 학습(Continued Pre-training), 도메인 적응 사전 학습(Domain-Adaptive Pre-training), 다중 과제 목적함수(Multi-task Objectives)가 표준 사전 학습을 전문 도메인으로 확장하는 방식을 식별할 수 있습니다.

---

## 개요

Pre-training 목적함수는 Foundation Model이 대규모 데이터에서 **어떤 패턴을 학습할지** 결정합니다. 목적함수 선택이 모델의 능력과 downstream task 성능에 직접적인 영향을 미칩니다.

---

## 1. Language Modeling 패러다임

### 1.1 세 가지 주요 접근법

```
┌─────────────────────────────────────────────────────────────────┐
│                    Language Modeling 패러다임                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Causal LM (Autoregressive)        Masked LM (Bidirectional)   │
│  ┌───┬───┬───┬───┬───┐            ┌───┬───┬───┬───┬───┐        │
│  │ A │ B │ C │ D │ ? │            │ A │[M]│ C │[M]│ E │        │
│  └───┴───┴───┴───┴───┘            └───┴───┴───┴───┴───┘        │
│       ↓                                 ↓                       │
│  P(x_t | x_<t)                     P(x_mask | x_context)        │
│  "다음 토큰 예측"                   "마스킹된 토큰 복원"           │
│                                                                 │
│  Prefix LM (Encoder-Decoder)                                    │
│  ┌───┬───┬───┐ → ┌───┬───┬───┐                                 │
│  │ A │ B │ C │   │ X │ Y │ Z │                                 │
│  └───┴───┴───┘   └───┴───┴───┘                                 │
│  Bidirectional    Autoregressive                                │
│  "입력 인코딩"      "출력 생성"                                   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 각 패러다임 비교

| 특성 | Causal LM | Masked LM | Prefix LM |
|------|-----------|-----------|-----------|
| 대표 모델 | GPT, LLaMA | BERT, RoBERTa | T5, BART |
| 컨텍스트 | 왼쪽만 참조 | 양방향 참조 | 인코더: 양방향, 디코더: 왼쪽 |
| 학습 신호 | 모든 토큰 | 마스킹된 토큰만 (15%) | Span/시퀀스 |
| 생성 능력 | 자연스러운 생성 | 추가 학습 필요 | 자연스러운 생성 |
| 이해 능력 | Zero-shot으로 가능 | 강력한 표현 학습 | 균형적 |

---

## 2. Causal Language Modeling (CLM)

### 2.1 수학적 정의

```
목적함수:
L_CLM = -Σ log P(x_t | x_1, x_2, ..., x_{t-1})

특징:
- 시퀀스의 모든 토큰을 학습 신호로 사용
- Autoregressive: 왼쪽→오른쪽 순차 생성
- Causal Mask로 미래 토큰 접근 차단
```

### 2.2 PyTorch 구현

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalLMHead(nn.Module):
    """Causal Language Model 출력 레이어"""

    def __init__(self, hidden_dim: int, vocab_size: int):
        super().__init__()
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: (batch, seq_len, hidden_dim)
        Returns:
            logits: (batch, seq_len, vocab_size)
        """
        return self.lm_head(hidden_states)


def causal_lm_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100
) -> torch.Tensor:
    """
    Causal LM Loss 계산

    Args:
        logits: (batch, seq_len, vocab_size)
        labels: (batch, seq_len) - 다음 토큰이 레이블
    """
    # Shift: logits[:-1]이 labels[1:]을 예측
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    # Flatten
    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=ignore_index
    )
    return loss


# Causal Mask 생성
def create_causal_mask(seq_len: int) -> torch.Tensor:
    """
    상삼각 마스크 생성 (미래 토큰 차단)

    Returns:
        mask: (seq_len, seq_len) - True = 마스킹
    """
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
    return mask


# 사용 예시
batch_size, seq_len, hidden_dim, vocab_size = 4, 128, 768, 50257
hidden_states = torch.randn(batch_size, seq_len, hidden_dim)
labels = torch.randint(0, vocab_size, (batch_size, seq_len))

lm_head = CausalLMHead(hidden_dim, vocab_size)
logits = lm_head(hidden_states)
loss = causal_lm_loss(logits, labels)
print(f"CLM Loss: {loss.item():.4f}")
```

### 2.3 GPT 스타일 학습

```python
class GPTPretraining:
    """GPT 스타일 Pre-training"""

    def __init__(self, model, tokenizer, max_length=1024):
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length

    def prepare_data(self, texts: list[str]) -> dict:
        """
        연속된 텍스트를 고정 길이로 분할

        Document 1: "The cat sat on..."
        Document 2: "Machine learning is..."

        → [BOS] The cat sat on... [EOS] [BOS] Machine learning is... [EOS]
        → 고정 길이 청크로 분할 (max_length 단위)
        """
        # 전체 텍스트 연결
        all_tokens = []
        for text in texts:
            tokens = self.tokenizer.encode(text)
            all_tokens.extend(tokens)
            all_tokens.append(self.tokenizer.eos_token_id)

        # 고정 길이로 분할
        chunks = []
        for i in range(0, len(all_tokens) - self.max_length, self.max_length):
            chunk = all_tokens[i:i + self.max_length]
            chunks.append(chunk)

        return {
            'input_ids': torch.tensor(chunks),
            'labels': torch.tensor(chunks)  # 동일 (shift는 loss에서)
        }

    def train_step(self, batch):
        """단일 학습 스텝"""
        input_ids = batch['input_ids']
        labels = batch['labels']

        # Forward
        outputs = self.model(input_ids)
        logits = outputs.logits

        # Loss
        loss = causal_lm_loss(logits, labels)

        return loss
```

---

## 3. Masked Language Modeling (MLM)

### 3.1 BERT 스타일 MLM

```
원본: "The quick brown fox jumps over the lazy dog"

마스킹 전략 (15% 토큰):
- 80%: [MASK] 토큰으로 대체
- 10%: 랜덤 토큰으로 대체
- 10%: 원본 유지

결과: "The [MASK] brown fox jumps over the [MASK] dog"
                ↓                          ↓
목표:        "quick"                    "lazy"
```

### 3.2 구현

```python
import random

class MLMDataCollator:
    """Masked Language Modeling 데이터 전처리"""

    def __init__(
        self,
        tokenizer,
        mlm_probability: float = 0.15,
        mask_token_ratio: float = 0.8,
        random_token_ratio: float = 0.1
    ):
        self.tokenizer = tokenizer
        self.mlm_probability = mlm_probability
        self.mask_token_ratio = mask_token_ratio
        self.random_token_ratio = random_token_ratio

        # 특수 토큰 ID
        self.mask_token_id = tokenizer.mask_token_id
        self.vocab_size = tokenizer.vocab_size
        self.special_tokens = set([
            tokenizer.cls_token_id,
            tokenizer.sep_token_id,
            tokenizer.pad_token_id
        ])

    def __call__(self, batch: list[dict]) -> dict:
        """배치 처리"""
        input_ids = torch.stack([item['input_ids'] for item in batch])

        # 마스킹
        input_ids, labels = self.mask_tokens(input_ids)

        return {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': torch.stack([item['attention_mask'] for item in batch])
        }

    def mask_tokens(
        self,
        input_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        토큰 마스킹 수행

        Returns:
            masked_input_ids: 마스킹된 입력
            labels: 원본 토큰 (마스킹 안 된 위치는 -100)
        """
        labels = input_ids.clone()

        # 마스킹 확률 행렬
        probability_matrix = torch.full(input_ids.shape, self.mlm_probability)

        # 특수 토큰은 마스킹하지 않음
        special_tokens_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        for token_id in self.special_tokens:
            special_tokens_mask |= (input_ids == token_id)

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)

        # 마스킹할 위치 선택
        masked_indices = torch.bernoulli(probability_matrix).bool()

        # 마스킹 안 된 위치는 -100 (loss 무시)
        labels[~masked_indices] = -100

        # 80%: [MASK]로 대체
        indices_replaced = torch.bernoulli(
            torch.full(input_ids.shape, self.mask_token_ratio)
        ).bool() & masked_indices
        input_ids[indices_replaced] = self.mask_token_id

        # 10%: 랜덤 토큰
        indices_random = torch.bernoulli(
            torch.full(input_ids.shape, self.random_token_ratio / (1 - self.mask_token_ratio))
        ).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(self.vocab_size, input_ids.shape, dtype=torch.long)
        input_ids[indices_random] = random_words[indices_random]

        # 나머지 10%: 원본 유지 (암묵적으로 처리됨)

        return input_ids, labels


def mlm_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """MLM Loss (마스킹된 위치만 계산)"""
    return F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        labels.view(-1),
        ignore_index=-100
    )
```

### 3.3 RoBERTa 개선점

```python
class RoBERTaMLM:
    """
    RoBERTa: MLM 개선 버전

    BERT 대비 변경점:
    1. Dynamic Masking: 에폭마다 다른 마스킹
    2. 더 긴 시퀀스 (512 → 더 길게)
    3. 더 큰 배치 (256 → 8K)
    4. NSP 제거
    5. 더 많은 데이터, 더 긴 학습
    """

    def __init__(self, tokenizer):
        self.collator = MLMDataCollator(tokenizer)

    def create_epoch_data(self, texts: list[str], epoch: int):
        """
        Dynamic Masking: 매 에폭마다 새로운 마스킹 패턴
        """
        # 시드를 에폭에 따라 변경
        random.seed(epoch)
        torch.manual_seed(epoch)

        # 데이터 전처리 (새로운 마스킹 적용)
        # ...
```

---

## 4. Span Corruption (T5)

### 4.1 개념

```
원본: "The quick brown fox jumps over the lazy dog"

Span Corruption:
- 연속된 토큰 span을 하나의 sentinel 토큰으로 대체
- 디코더가 원본 span 복원

입력: "The <X> fox <Y> over the lazy dog"
출력: "<X> quick brown <Y> jumps"

특징:
- 평균 span 길이: 3 토큰
- 마스킹 비율: 15%
- Sentinel: <extra_id_0>, <extra_id_1>, ...
```

### 4.2 구현

```python
class SpanCorruptionCollator:
    """T5 스타일 Span Corruption"""

    def __init__(
        self,
        tokenizer,
        noise_density: float = 0.15,
        mean_span_length: float = 3.0
    ):
        self.tokenizer = tokenizer
        self.noise_density = noise_density
        self.mean_span_length = mean_span_length

        # Sentinel 토큰 (<extra_id_0>, <extra_id_1>, ...)
        self.sentinel_start_id = tokenizer.convert_tokens_to_ids("<extra_id_0>")

    def __call__(self, examples: list[dict]) -> dict:
        """배치 처리"""
        batch_inputs = []
        batch_targets = []

        for example in examples:
            input_ids = example['input_ids']
            inputs, targets = self.corrupt_span(input_ids)
            batch_inputs.append(inputs)
            batch_targets.append(targets)

        # 패딩
        inputs_padded = self._pad_sequences(batch_inputs)
        targets_padded = self._pad_sequences(batch_targets)

        return {
            'input_ids': inputs_padded,
            'labels': targets_padded
        }

    def corrupt_span(
        self,
        input_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Span Corruption 적용
        """
        length = len(input_ids)
        num_noise_tokens = int(length * self.noise_density)
        num_spans = max(1, int(num_noise_tokens / self.mean_span_length))

        # Span 시작 위치 샘플링
        span_starts = sorted(random.sample(range(length - 1), num_spans))

        # 각 span의 길이 (지수 분포)
        span_lengths = torch.poisson(
            torch.full((num_spans,), self.mean_span_length - 1)
        ).long() + 1

        # Span 마스크 생성
        noise_mask = torch.zeros(length, dtype=torch.bool)
        for start, span_len in zip(span_starts, span_lengths):
            end = min(start + span_len, length)
            noise_mask[start:end] = True

        # 입력 구성: 노이즈 span을 sentinel로 대체
        input_tokens = []
        target_tokens = []
        sentinel_id = self.sentinel_start_id

        i = 0
        while i < length:
            if noise_mask[i]:
                # Span 시작: sentinel 추가
                input_tokens.append(sentinel_id)
                target_tokens.append(sentinel_id)

                # Span 내용을 target에 추가
                while i < length and noise_mask[i]:
                    target_tokens.append(input_ids[i].item())
                    i += 1

                sentinel_id += 1
            else:
                input_tokens.append(input_ids[i].item())
                i += 1

        return torch.tensor(input_tokens), torch.tensor(target_tokens)

    def _pad_sequences(self, sequences: list[torch.Tensor]) -> torch.Tensor:
        """시퀀스 패딩"""
        max_len = max(len(seq) for seq in sequences)
        padded = torch.full((len(sequences), max_len), self.tokenizer.pad_token_id)
        for i, seq in enumerate(sequences):
            padded[i, :len(seq)] = seq
        return padded
```

---

## 5. UL2: Unified Language Learner

### 5.1 Mixture of Denoisers (MoD)

```
UL2: 여러 목적함수를 혼합하여 학습

┌────────────────────────────────────────────────────────────────┐
│                    Mixture of Denoisers                        │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  R-Denoiser (Regular)      S-Denoiser (Short)                 │
│  - 짧은 span (3-8 토큰)     - 매우 짧은 span (≤3 토큰)          │
│  - 15% 마스킹               - 15% 마스킹                        │
│  - NLU 태스크에 유리         - 세밀한 이해에 유리                 │
│                                                                │
│  X-Denoiser (Extreme)                                          │
│  - 긴 span (12-64 토큰)                                        │
│  - 50% 마스킹                                                  │
│  - 생성 태스크에 유리                                           │
│                                                                │
│  Mode Switching: 입력에 [R], [S], [X] 프리픽스 추가             │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### 5.2 구현

```python
class UL2Collator:
    """UL2 Mixture of Denoisers"""

    DENOISERS = {
        'R': {  # Regular
            'span_length': (3, 8),
            'noise_density': 0.15,
            'prefix': '[R]'
        },
        'S': {  # Short
            'span_length': (1, 3),
            'noise_density': 0.15,
            'prefix': '[S]'
        },
        'X': {  # Extreme
            'span_length': (12, 64),
            'noise_density': 0.5,
            'prefix': '[X]'
        }
    }

    def __init__(self, tokenizer, denoiser_weights: dict = None):
        self.tokenizer = tokenizer
        # 기본 가중치: R=50%, S=25%, X=25%
        self.weights = denoiser_weights or {'R': 0.5, 'S': 0.25, 'X': 0.25}

    def __call__(self, examples: list[dict]) -> dict:
        """배치 처리: 각 예제에 랜덤 denoiser 적용"""
        batch_inputs = []
        batch_targets = []

        for example in examples:
            # Denoiser 선택
            denoiser = random.choices(
                list(self.DENOISERS.keys()),
                weights=list(self.weights.values())
            )[0]

            config = self.DENOISERS[denoiser]

            # Span corruption 적용
            inputs, targets = self.apply_denoiser(
                example['input_ids'],
                config
            )

            batch_inputs.append(inputs)
            batch_targets.append(targets)

        return self._collate(batch_inputs, batch_targets)

    def apply_denoiser(
        self,
        input_ids: torch.Tensor,
        config: dict
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """특정 denoiser 설정으로 corruption 적용"""
        # 프리픽스 추가
        prefix_ids = self.tokenizer.encode(
            config['prefix'],
            add_special_tokens=False
        )

        # Span corruption (config에 따라)
        span_len = random.randint(*config['span_length'])
        # ... corruption 로직

        # 프리픽스 + 입력
        inputs = torch.cat([
            torch.tensor(prefix_ids),
            input_ids  # corrupted
        ])

        return inputs, targets
```

---

## 6. Next Sentence Prediction (NSP) vs Sentence Order Prediction (SOP)

### 6.1 NSP (BERT)

```python
class NSPDataCollator:
    """
    Next Sentence Prediction

    50%: 실제 다음 문장 (IsNext)
    50%: 랜덤 문장 (NotNext)

    문제점: 너무 쉬움 → RoBERTa에서 제거
    """

    def create_nsp_pair(
        self,
        sentence_a: str,
        sentence_b: str,
        all_sentences: list[str]
    ) -> tuple[str, str, int]:
        """NSP 데이터 생성"""
        if random.random() < 0.5:
            # 실제 다음 문장
            return sentence_a, sentence_b, 1  # IsNext
        else:
            # 랜덤 문장
            random_sentence = random.choice(all_sentences)
            return sentence_a, random_sentence, 0  # NotNext
```

### 6.2 SOP (ALBERT)

```python
class SOPDataCollator:
    """
    Sentence Order Prediction (더 어려운 태스크)

    50%: 정상 순서 (A → B)
    50%: 역순 (B → A)

    토픽 예측이 아닌 순서 예측 → 더 유용한 학습 신호
    """

    def create_sop_pair(
        self,
        sentence_a: str,
        sentence_b: str
    ) -> tuple[str, str, int]:
        """SOP 데이터 생성"""
        if random.random() < 0.5:
            return sentence_a, sentence_b, 1  # 정상 순서
        else:
            return sentence_b, sentence_a, 0  # 역순
```

---

## 7. Pre-training 목적함수 선택 가이드

### 7.1 태스크별 권장 목적함수

```
┌──────────────────┬─────────────────────────────────────────┐
│ Downstream Task  │ 권장 Pre-training 목적함수              │
├──────────────────┼─────────────────────────────────────────┤
│ 텍스트 생성      │ Causal LM (GPT 스타일)                  │
│ 텍스트 분류      │ MLM (BERT) 또는 Causal LM + Fine-tuning │
│ 질의응답         │ Span Corruption (T5) 또는 MLM           │
│ 번역/요약        │ Encoder-Decoder (T5, BART)              │
│ 범용 (Few-shot)  │ Causal LM 대규모 (GPT-3 스타일)         │
│ 범용 (다양한)    │ UL2 (Mixture of Denoisers)              │
└──────────────────┴─────────────────────────────────────────┘
```

### 7.2 모델 크기별 전략

| 모델 크기 | 권장 접근법 | 이유 |
|-----------|-------------|------|
| < 1B | MLM + Fine-tuning | 태스크 특화 성능 우수 |
| 1B - 10B | Causal LM | 범용성과 효율의 균형 |
| > 10B | Causal LM | In-context Learning 출현 |

---

## 8. 실습: 목적함수 비교

```python
from transformers import (
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    T5ForConditionalGeneration,
    AutoTokenizer
)

def compare_objectives():
    """세 가지 목적함수 비교"""

    # 1. Causal LM (GPT-2)
    causal_tokenizer = AutoTokenizer.from_pretrained('gpt2')
    causal_model = AutoModelForCausalLM.from_pretrained('gpt2')

    text = "The capital of France is"
    inputs = causal_tokenizer(text, return_tensors='pt')

    # 생성
    outputs = causal_model.generate(
        inputs['input_ids'],
        max_new_tokens=5,
        do_sample=False
    )
    print("Causal LM:", causal_tokenizer.decode(outputs[0]))
    # → "The capital of France is Paris."

    # 2. Masked LM (BERT)
    mlm_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    mlm_model = AutoModelForMaskedLM.from_pretrained('bert-base-uncased')

    text = "The capital of France is [MASK]."
    inputs = mlm_tokenizer(text, return_tensors='pt')

    outputs = mlm_model(**inputs)
    mask_idx = (inputs['input_ids'] == mlm_tokenizer.mask_token_id).nonzero()[0, 1]
    predicted_id = outputs.logits[0, mask_idx].argmax()
    print("Masked LM:", mlm_tokenizer.decode(predicted_id))
    # → "paris"

    # 3. Span Corruption (T5)
    t5_tokenizer = AutoTokenizer.from_pretrained('t5-small')
    t5_model = T5ForConditionalGeneration.from_pretrained('t5-small')

    text = "translate English to French: The house is wonderful."
    inputs = t5_tokenizer(text, return_tensors='pt')

    outputs = t5_model.generate(inputs['input_ids'], max_new_tokens=20)
    print("T5:", t5_tokenizer.decode(outputs[0], skip_special_tokens=True))
    # → "La maison est merveilleuse."

if __name__ == "__main__":
    compare_objectives()
```

---

## 참고 자료

### 논문
- Devlin et al. (2018). "BERT: Pre-training of Deep Bidirectional Transformers"
- Radford et al. (2019). "Language Models are Unsupervised Multitask Learners" (GPT-2)
- Raffel et al. (2019). "Exploring the Limits of Transfer Learning with T5"
- Tay et al. (2022). "UL2: Unifying Language Learning Paradigms"

### 관련 레슨
- [../LLM_and_NLP/03_BERT_GPT_Architecture.md](../LLM_and_NLP/03_BERT_GPT_Architecture.md)
- [../Deep_Learning/12_Transformer_Architecture.md](../Deep_Learning/12_Transformer_Architecture.md)

---

## 연습 문제

### 연습 문제 1: 목적 함수(Objective Function) 비교

아래 각 시나리오에 대해 가장 적합한 사전 학습 목적 함수(CLM, MLM, 스팬 손상(Span Corruption))를 식별하고, 선택 이유를 설명하세요.

1. 유창한 다중 턴 대화를 생성해야 하는 챗봇을 만들고 있습니다.
2. 영화 리뷰의 감성 분류(Sentiment Classification)를 위해 모델을 파인튜닝하고 싶습니다.
3. 문서와 질문을 받아 짧은 답변을 생성하는 질의응답(QA) 시스템을 구축하고 있습니다.
4. 태스크별 파인튜닝 없이 분류와 생성 모두에서 잘 작동하는 단일 모델을 원합니다.

<details>
<summary>정답 보기</summary>

1. **챗봇 (Causal LM / CLM)** — 생성은 이전 컨텍스트를 기반으로 다음 토큰을 자기회귀적으로 예측해야 합니다. CLM은 모델이 이전 컨텍스트에서 유창한 연속물을 생성하도록 학습하는 자연스러운 목적 함수입니다. GPT 스타일 모델이 뛰어납니다.

2. **감성 분류 (MLM)** — 파인튜닝을 통한 이해 중심 태스크에서는 양방향 컨텍스트(MLM)가 CLM보다 더 풍부한 토큰 표현을 생성합니다. 동일한 파라미터 수에서 BERT 스타일 모델이 분류 태스크에서 GPT 모델보다 일반적으로 우수합니다.

3. **질의응답 (스팬 손상(Span Corruption) / 인코더-디코더)** — T5 스타일 스팬 손상은 모델이 컨텍스트에서 누락된 스팬을 재구성하도록 학습시켜, QA 태스크 구조(마스킹된 스팬이 있는 문서에서 답변 복원)를 직접 모방합니다. 인코더-디코더 아키텍처는 유연한 길이 출력도 가능합니다.

4. **분류와 생성 모두를 위한 단일 모델 (UL2 / Mixture of Denoisers)** — UL2는 짧은 스팬 MLM 유사 태스크와 긴 스팬 생성 태스크를 모두 커버하는 여러 디노이징(denoising) 목적 함수(R, S, X 디노이저)로 학습합니다. 모드 프리픽스([R], [S], [X])를 통해 파인튜닝 없이 추론 시 태스크 유형 지정이 가능합니다.

</details>

---

### 연습 문제 2: BERT 마스킹 전략

BERT의 MLM에서 토큰의 15%가 마스킹 대상으로 선택됩니다. 그 중 80%는 `[MASK]`로 교체되고, 10%는 임의 토큰으로 교체되며, 10%는 변경 없이 유지됩니다.

1. 선택된 토큰의 10%를 변경 없이 유지하는 이유는 무엇인가요 (0%가 아닌)?
2. 선택된 토큰의 10%를 임의 토큰으로 교체하는 이유는 무엇인가요 (항상 `[MASK]`를 사용하지 않고)?
3. 15% 대신 50%의 토큰을 마스킹하면 어떤 문제가 발생하나요?

<details>
<summary>정답 보기</summary>

**1. 10%를 변경 없이 유지하는 이유:**
일부 토큰을 변경 없이 유지하지 않으면, 모델은 정확히 그대로 나타나는 토큰에 대한 좋은 표현을 학습하지 못합니다. 파인튜닝 중에는 `[MASK]` 토큰이 나타나지 않습니다 — 모델은 실제 토큰의 표현에 의존해야 합니다. 10% 유지는 `[MASK]` 위치뿐만 아니라 모든 토큰에 대해 유용한 표현을 유지하도록 강제합니다.

**2. 10%를 임의 토큰으로 교체하는 이유:**
임의 토큰 대체 없이는, 모델이 `[MASK]` 토큰은 항상 예측이 필요하다고 "부정행위"를 학습하고 다른 토큰은 그대로 통과시킬 수 있습니다. 임의 대체는 모델이 모든 토큰을 잠재적으로 "잘못된" 것으로 간주하고 각 위치에 대해 문맥적으로 근거한 예측을 생성하도록 강제합니다 — 마스킹된 위치뿐만 아니라 모든 토큰 표현의 품질을 향상시킵니다.

**3. 50% 마스킹 시 문제:**
- 너무 많은 컨텍스트가 파괴됩니다: 모델은 마스킹된 위치를 올바르게 추론하기 위한 충분한 주변 토큰이 없어 태스크가 너무 어렵거나 풀 수 없게 됩니다.
- 많은 예측이 컨텍스트 기반이 아닌 추측이 되므로 학습 신호가 노이즈가 됩니다.
- 모델이 자연 텍스트와 거의 다른 시퀀스를 보기 때문에 결과 표현이 저하될 수 있습니다.
- 15% 비율은 학습 신호 품질과 컨텍스트 가용성의 균형을 맞추기 위해 경험적으로 발견된 값입니다.

</details>

---

### 연습 문제 3: 인과 마스크(Causal Mask) 구현

수업의 `create_causal_mask` 함수는 불리언 상삼각 행렬을 생성합니다. 이 마스크가 스케일드 닷 프로덕트 어텐션(Scaled Dot-Product Attention)에 적용될 때 어떤 일이 발생하는지 추적하세요.

4-토큰 시퀀스에 대한 쿼리-키 닷 프로덕트 행렬:
```
scores = [[0.9, 0.3, 0.7, 0.5],
          [0.2, 0.8, 0.1, 0.4],
          [0.6, 0.5, 0.9, 0.3],
          [0.4, 0.7, 0.8, 0.6]]
```

마스킹된 스코어 행렬과 소프트맥스 후 최종 어텐션 가중치를 (수치가 아닌 개념적으로) 보여주세요.

<details>
<summary>정답 보기</summary>

**1단계: 인과 마스크 (True = 차단)**

```python
mask = torch.triu(torch.ones(4, 4), diagonal=1).bool()
# [[False, True,  True,  True ],
#  [False, False, True,  True ],
#  [False, False, False, True ],
#  [False, False, False, False]]
```

**2단계: 마스크 적용 (마스킹된 위치를 -무한대로 설정)**

```
masked_scores = [[ 0.9, -inf, -inf, -inf],
                 [ 0.2,  0.8, -inf, -inf],
                 [ 0.6,  0.5,  0.9, -inf],
                 [ 0.4,  0.7,  0.8,  0.6]]
```

**3단계: 행 단위 소프트맥스**

소프트맥스 후 `-inf` 위치는 0이 됩니다 (어텐션 가중치가 0):
- 토큰 1 (행 0): 자기 자신(위치 0)에만 어텐션
- 토큰 2 (행 1): 위치 0과 1 사이에 어텐션 분배
- 토큰 3 (행 2): 위치 0, 1, 2 사이에 어텐션 분배
- 토큰 4 (행 3): 네 위치 모두에 걸쳐 어텐션 분배

이를 통해 위치 t는 위치 ≤ t에만 어텐션을 줄 수 있어, 학습 중 미래 토큰의 정보 "누출"을 방지합니다. 모델은 과거 컨텍스트만 사용하여 다음 토큰을 예측하도록 학습됩니다.

</details>

---

### 연습 문제 4: 스팬 손상(Span Corruption) vs MLM 신호 밀도

T5의 스팬 손상은 토큰의 ~15%를 마스킹하지만 각 스팬(평균 3 토큰)을 단일 센티넬(sentinel) 토큰으로 교체하는 반면, BERT는 개별 토큰의 15%를 마스킹합니다.

512 토큰 시퀀스에서 각 목적 함수의 **학습 신호 밀도(training signal density)** (예측되는 출력 토큰의 비율)를 계산하세요. 어느 것이 더 조밀한 감독 신호를 제공하나요?

<details>
<summary>정답 보기</summary>

**BERT MLM:**
- 입력 토큰: 512
- 마스킹된 토큰: 512 × 0.15 = 76.8 ≈ 77개 토큰 예측
- 출력 예측 비율: 77 / 512 = **15%**

**T5 스팬 손상(Span Corruption):**
- 입력 토큰: 512
- 노이즈 토큰: 512 × 0.15 = 76.8 ≈ 77개 토큰 손상
- 스팬 수 (평균 3 토큰): 77 / 3 ≈ 26개 스팬
- 출력 시퀀스: 26개 센티넬 + 77개 원본 토큰 = 103개 토큰
- 디코더는 103개 출력 토큰 모두 예측 필요
- 출력 예측 비율: 103 / 512 = **~20%**

더 중요한 차이는 **디코더의 역할**입니다: T5에서 디코더는 전체 대상 시퀀스를 자기회귀적으로 예측하므로 센티넬 구조 생성도 학습합니다. BERT에서 인코더는 마스킹된 모든 위치를 병렬로 예측합니다(비자기회귀적).

**핵심 인사이트:** 스팬 손상은 모델이 고립된 토큰이 아닌 더 긴 연속 시퀀스(전체 스팬)를 예측해야 하므로, 일관된 다중 토큰 출력이 필요한 생성 태스크를 더 잘 준비시킵니다.

</details>
