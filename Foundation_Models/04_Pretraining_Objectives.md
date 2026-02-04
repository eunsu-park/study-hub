# 04. Pre-training 목적함수

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
