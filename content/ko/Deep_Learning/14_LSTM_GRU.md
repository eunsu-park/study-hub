# 09. LSTM과 GRU

## 학습 목표

- LSTM과 GRU의 구조 이해
- 게이트 메커니즘
- 장기 의존성 학습
- PyTorch 구현

---

## 1. LSTM (Long Short-Term Memory)

LSTM의 핵심 통찰은 기본 RNN의 곱셈적 은닉 상태 업데이트(h = W * h)를 **덧셈적(additive)** 셀 상태 업데이트(C(t) = f * C(t-1) + i * g)로 대체하는 것입니다. 역전파(backpropagation) 시 덧셈 연산은 기울기를 변형 없이 통과시키는데 — 이것이 기울기 소실 문제를 해결하는 핵심 메커니즘입니다.

### 문제: RNN의 기울기 소실

```
h100 ← W × W × ... × W × h1
            ↑
    Gradient converges to 0
```

### 해결: 셀 상태 (Cell State)

```
LSTM = Cell State (long-term memory) + Hidden State (short-term memory)
```

### LSTM 구조

```
       ┌──────────────────────────────────────┐
       │            Cell State (C)              │
       │     ×─────(+)─────────────────────►    │
       │     ↑      ↑                           │
       │    forget  input                       │
       │    gate    gate                        │
       │     ↑      ↑                           │
h(t-1)─┴──►[σ]   [σ][tanh]    [σ]──►×──────►h(t)
           f(t)   i(t) g(t)   o(t)     ↑
                              output gate
```

### 게이트 수식

```python
# --- Why σ (sigmoid) for gates and tanh for candidate? ---
# Gates (f, i, o) use σ because they act as "soft switches": σ outputs
# values in (0, 1), so multiplying by a gate smoothly interpolates between
# "block everything" (0) and "pass everything" (1).
# The candidate g uses tanh because it produces the *new information* to
# be stored — tanh outputs in (-1, 1), centering the values around zero,
# which keeps the cell state well-conditioned and avoids drift.

# Forget Gate: How much to forget from previous memory
f(t) = σ(W_f × [h(t-1), x(t)] + b_f)       # ≈1 → remember, ≈0 → forget

# Input Gate: How much new information to store
i(t) = σ(W_i × [h(t-1), x(t)] + b_i)       # ≈1 → write, ≈0 → ignore

# Cell Candidate: New candidate information
g(t) = tanh(W_g × [h(t-1), x(t)] + b_g)    # value in (-1, 1)

# Cell State Update — this is the key to LSTM's long-range memory:
# The update is *additive* (f×C + i×g), not multiplicative (W×h as in
# vanilla RNN).  Additive updates let gradients flow unchanged through
# the forget gate path, avoiding the vanishing gradient that plagues RNNs
# where gradients must pass through W^T at every time step.
C(t) = f(t) × C(t-1) + i(t) × g(t)

# Output Gate: How much of cell state to output
o(t) = σ(W_o × [h(t-1), x(t)] + b_o)

# Hidden State
h(t) = o(t) × tanh(C(t))
```

---

## 2. GRU (Gated Recurrent Unit)

### LSTM의 단순화 버전

GRU는 셀 상태와 은닉 상태를 하나의 상태 벡터로 통합하고 3개 대신 2개의 게이트를 사용하여 더 적은 파라미터로 LSTM에 필적하는 성능을 달성합니다. **파라미터 비교**: LSTM은 4개의 게이트 행렬 × (입력 + 은닉) 가중치 = 4(n*m + n*n)개입니다. GRU는 3개의 게이트 행렬 = 3(n*m + n*n)개입니다. hidden_size=512, input_size=300의 경우: LSTM은 약 3.4M 개의 파라미터를, GRU는 약 2.5M 개의 파라미터를 가집니다 — 약 25% 적어 학습 속도가 빠르고 메모리 사용량이 낮습니다.

```
GRU = Reset Gate + Update Gate
(Merges cell state and hidden state)
```

### GRU 구조

```
       Update Gate (z)
       ┌────────────────────────────┐
       │                            │
h(t-1)─┴──►[σ]───z(t)──────×──(+)──►h(t)
              │           ↑    ↑
              │      ┌────┘    │
              │      │   ×─────┘
              │      │   ↑
              ├──►[σ]   [tanh]
              │   r(t)    │
              │    │      │
              └────×──────┘
                Reset Gate (r)
```

### 게이트 수식

```python
# Update Gate: Ratio of previous state vs new state
z(t) = σ(W_z × [h(t-1), x(t)] + b_z)

# Reset Gate: How much to forget previous state
r(t) = σ(W_r × [h(t-1), x(t)] + b_r)

# Candidate Hidden
h̃(t) = tanh(W × [r(t) × h(t-1), x(t)] + b)

# Hidden State Update
h(t) = (1 - z(t)) × h(t-1) + z(t) × h̃(t)
```

---

## 3. PyTorch LSTM/GRU

### LSTM

```python
lstm = nn.LSTM(
    input_size=10,
    hidden_size=20,
    num_layers=2,      # Stacked LSTM: first layer extracts low-level temporal
                        # patterns, second layer captures higher-level abstractions
    batch_first=True,
    dropout=0.1,        # Dropout between LSTM layers (not within a single layer) —
                        # regularizes to prevent co-adaptation of stacked layers
    bidirectional=False
)

# Forward pass
# output: Hidden states at all times
# (h_n, c_n): Last (hidden, cell) states
output, (h_n, c_n) = lstm(x)
```

### GRU

```python
gru = nn.GRU(
    input_size=10,
    hidden_size=20,
    num_layers=2,
    batch_first=True
)

# Forward pass (no cell state)
output, h_n = gru(x)
```

---

## 4. LSTM 분류기

```python
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim,
            num_layers=2,           # Stacked: layer 1 captures token-level patterns,
                                    # layer 2 captures phrase/sentence-level patterns
            batch_first=True,
            dropout=0.3,            # Dropout between the two LSTM layers — prevents
                                    # the second layer from over-relying on specific
                                    # activation patterns from the first
            bidirectional=True      # Process sequence both forward and backward —
                                    # each position gets context from past AND future
        )
        # Bidirectional so hidden_dim * 2 (forward + backward hidden states concatenated)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        # x: (batch, seq) - token indices
        embedded = self.embedding(x)

        # LSTM
        output, (h_n, c_n) = self.lstm(embedded)

        # Combine bidirectional last hidden states
        # h_n: (num_layers*2, batch, hidden)
        forward_last = h_n[-2]  # Forward last layer
        backward_last = h_n[-1]  # Backward last layer
        combined = torch.cat([forward_last, backward_last], dim=1)

        return self.fc(combined)
```

---

## 5. 시퀀스 생성 (언어 모델)

```python
class LSTMLanguageModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        embedded = self.embedding(x)
        output, hidden = self.lstm(embedded, hidden)
        logits = self.fc(output)
        return logits, hidden

    def generate(self, start_token, max_len, temperature=1.0):
        self.eval()
        tokens = [start_token]
        # hidden state carries context from all previously generated tokens —
        # this is what makes the generation autoregressive and coherent
        hidden = None

        with torch.no_grad():
            for _ in range(max_len):
                x = torch.tensor([[tokens[-1]]])
                logits, hidden = self(x, hidden)

                # Temperature sampling: dividing logits by temperature before
                # softmax controls diversity. T<1 sharpens the distribution
                # (more deterministic), T>1 flattens it (more random/creative)
                probs = F.softmax(logits[0, -1] / temperature, dim=0)
                next_token = torch.multinomial(probs, 1).item()
                tokens.append(next_token)

        return tokens
```

---

## 6. LSTM vs GRU 비교

| 항목 | LSTM | GRU |
|------|------|-----|
| 게이트 수 | 3개 (f, i, o) | 2개 (r, z) |
| 상태 | 셀 + 은닉 | 은닉만 |
| 파라미터 | 더 많음 | 더 적음 |
| 학습 속도 | 느림 | 빠름 |
| 성능 | 복잡한 패턴 | 비슷하거나 약간 낮음 |

### 선택 가이드

- **LSTM**: 긴 시퀀스, 복잡한 의존성
- **GRU**: 빠른 학습, 제한된 자원

---

## 7. 실전 팁

### 초기화

```python
# Initialize hidden state
def init_hidden(batch_size, hidden_size, num_layers, bidirectional):
    num_directions = 2 if bidirectional else 1
    h = torch.zeros(num_layers * num_directions, batch_size, hidden_size)
    c = torch.zeros(num_layers * num_directions, batch_size, hidden_size)
    return (h.to(device), c.to(device))
```

### Dropout 패턴

```python
class LSTMWithDropout(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, dropout=0.5):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        output, (h_n, _) = self.lstm(x)
        # Apply dropout to last hidden state
        dropped = self.dropout(h_n[-1])
        return self.fc(dropped)
```

---

## 정리

### 핵심 개념

1. **LSTM**: 셀 상태로 장기 기억 유지, 3개 게이트
2. **GRU**: LSTM 단순화, 2개 게이트
3. **게이트**: 정보 흐름 제어 (시그모이드 × 값)

### 핵심 코드

```python
# LSTM
lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
output, (h_n, c_n) = lstm(x)

# GRU
gru = nn.GRU(input_size, hidden_size, batch_first=True)
output, h_n = gru(x)
```

---

## 연습 문제

### 연습 1: LSTM 게이트 값 확인하기

타임 스텝에 걸쳐 LSTM 게이트 활성화가 어떻게 변하는지 관찰하세요.

1. 단일 레이어 `nn.LSTM(input_size=1, hidden_size=4, batch_first=True)`를 생성하세요.
2. 20개의 사인파 값 시퀀스를 입력으로 전달하세요.
3. `lstm.weight_ih_l0`과 `lstm.weight_hh_l0`에서 추출한 가중치 행렬을 사용하여 LSTM 한 스텝을 수동으로 구현하여 내부 게이트 값에 접근하세요.
4. 20 타임 스텝에 걸쳐 망각 게이트(Forget Gate), 입력 게이트(Input Gate), 출력 게이트(Output Gate) 값을 그래프로 그리세요.
5. 관찰하세요: 망각 게이트가 언제 0(망각)에 가까워지고 언제 1(기억)에 가까워지는가?

### 연습 2: 긴 시퀀스에서 LSTM vs GRU 비교

다양한 길이의 시퀀스에서 LSTM과 GRU를 비교하세요.

1. `seq_len` 값을 10, 30, 50, 100으로 설정하여 사인파 예측 태스크를 생성하세요.
2. 동일한 은닉 크기로 `nn.LSTM`과 `nn.GRU` 각각을 40 에포크 학습하세요.
3. 각 조합에 대한 테스트 MSE를 기록하세요.
4. 결과 표를 만들고 두 모델에 대해 MSE vs 시퀀스 길이 그래프를 그리세요.
5. 어떤 모델이 더 긴 시퀀스를 더 잘 처리하는지, 그 이유에 대한 가설을 설명하세요.

### 연습 3: 양방향 LSTM 감성 분류기

수업의 `LSTMClassifier`를 구성하고 실제 데이터셋에서 평가하세요.

1. `torchtext`의 AG News 또는 SST-2 데이터셋을 사용하거나, 300개 샘플 규모의 장난감 데이터셋을 만드세요.
2. `embed_dim=64`, `hidden_dim=128`, `num_layers=2`, `bidirectional=True`로 `LSTMClassifier`를 구성하세요.
3. Adam과 크로스 엔트로피 손실로 15 에포크 학습하세요.
4. 테스트 정확도와 혼동 행렬(Confusion Matrix)을 보고하세요.
5. 임베딩 노름(Embedding Norm)이 가장 높은 토큰 상위 5개를 확인하세요 — 의미 있는 단어들인가요?

### 연습 4: 텍스트 생성에서 온도(Temperature) 샘플링 탐구

LSTM 언어 모델에서 온도가 다양성을 어떻게 제어하는지 탐구하세요.

1. 짧은 텍스트 코퍼스(예: 책의 몇 문단)에서 `LSTMLanguageModel`을 학습하세요.
2. 온도 0.5, 1.0, 1.5로 `generate` 메서드를 사용하세요.
3. 각 온도에서 50개 토큰을 생성하고 결과를 나란히 표시하세요.
4. 질적 차이를 설명하세요: 낮은 온도는 더 예측 가능한 텍스트를, 높은 온도는 더 무작위적인 텍스트를 생성합니다. 온도가 적용된 소프트맥스(Softmax) 공식을 참조하여 수학적으로 설명하세요.

### 연습 5: NumPy로 GRU 직접 구현하기

신경망 라이브러리 없이 NumPy로 GRU 업데이트 방정식 한 스텝을 구현하세요.

1. 소규모 난수 값으로 초기화된 가중치 행렬 `W_z`, `W_r`, `W_h` (각각 `(hidden+input, hidden)` 형태)를 정의하세요.
2. NumPy만 사용하여 업데이트 게이트(Update Gate) `z`, 리셋 게이트(Reset Gate) `r`, 후보(Candidate) `h_tilde`, 새 은닉 상태 `h`를 구현하세요.
3. 난수 입력 시퀀스에서 10 스텝을 순차적으로 실행하세요.
4. `gru.weight_ih_l0` 등으로 동일한 가중치를 수동으로 불러온 `nn.GRU`와 출력을 비교하세요.
5. 모든 은닉 상태 값이 허용 오차 `1e-5` 이내로 일치하는지 검증하세요.

---

## 다음 단계

[Attention과 Transformer](./16_Attention_Transformer.md)에서 Seq2Seq와 Attention을 학습합니다.
