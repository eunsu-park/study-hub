# 08. RNN 기초 (Recurrent Neural Networks)

## 학습 목표

- 순환 신경망의 개념과 구조
- 시퀀스 데이터 처리
- PyTorch nn.RNN 사용법
- 기울기 소실 문제 이해

---

## 1. RNN이란?

피드포워드 네트워크(feedforward network)는 각 입력을 독립적으로 처리하기 때문에 순서나 맥락이라는 개념이 없습니다. "The dog bit the man"과 "The man bit the dog"을 단어 집합(bag of words)으로 처리하면 동일하게 보입니다. RNN은 이전 시간 단계의 맥락을 전달하는 은닉 상태(hidden state)를 유지하여 시퀀스의 순서와 이력을 인식할 수 있습니다.

### 순차 데이터의 특성

```
Time series: [1, 2, 3, 4, 5, ...]  - Previous values affect next values
Text: "I go to school"              - Previous words affect next words
```

### MLP의 한계

- 고정된 입력 크기
- 순서 정보 무시
- 가변 길이 시퀀스 처리 불가

### RNN의 해결

```
h(t) = tanh(W_xh × x(t) + W_hh × h(t-1) + b)

h(t): Current hidden state
x(t): Current input
h(t-1): Previous hidden state
```

**직관**: 이를 [새로운 문맥] = blend([새 입력], [이전 문맥])으로 생각할 수 있습니다. W_xh는 현재 입력에서 특징을 추출하고, W_hh는 이전 문맥에서 관련된 부분을 선택적으로 기억하며, tanh은 결과를 [-1, 1]로 압축하여 값이 폭발하는 것을 방지합니다. 동일한 가중치(W_xh, W_hh)가 모든 시간 단계에서 재사용됩니다 — 네트워크는 고정된 크기의 회로가 아니라 *반복(loop)*하는 프로그램처럼 동작합니다.

---

## 2. RNN 구조

### 시간 펼침 (Unrolling)

```
    x1      x2      x3      x4
    ↓       ↓       ↓       ↓
  ┌───┐   ┌───┐   ┌───┐   ┌───┐
  │ h │──►│ h │──►│ h │──►│ h │──► Output
  └───┘   └───┘   └───┘   └───┘
    h0      h1      h2      h3
```

### 파라미터 공유

**왜 시간 단계 전반에 걸쳐 가중치를 공유하는가?** 각 시간 단계마다 다른 가중치를 사용하면: (1) 사전에 시퀀스 길이를 알아야 하고, (2) 시퀀스 길이에 따라 O(T) 개의 파라미터가 필요하며, (3) 테스트 시 다른 길이의 시퀀스에 일반화할 수 없습니다. 가중치를 공유하면 RNN이 길이에 독립적이 되어 — 코드의 `for` 반복문처럼 — 동일한 전이 함수(transition function)가 매 단계마다 적용됩니다.

- 모든 시간 단계에서 동일한 W_xh, W_hh 사용
- 가변 길이 시퀀스 처리 가능

---

## 3. PyTorch RNN

### 기본 사용법

```python
import torch
import torch.nn as nn

# Create RNN
rnn = nn.RNN(
    input_size=10,    # Input dimension
    hidden_size=20,   # Hidden state dimension — 20 is small for demo;
                      # real tasks typically use 128-512
    num_layers=2,     # Number of RNN layers — stacking adds depth,
                      # letting higher layers learn more abstract patterns
    batch_first=True  # batch_first=True: input shape is (batch, seq_len, features)
                      # instead of (seq_len, batch, features) — matches DataLoader convention
)

# Input shape: (batch_size, seq_len, input_size)
x = torch.randn(32, 15, 10)  # Batch 32, Sequence 15, Features 10

# Forward pass
# output: Hidden states at all times (batch, seq, hidden)
# h_n: Last hidden state (layers, batch, hidden)
output, h_n = rnn(x)

print(f"output: {output.shape}")  # (32, 15, 20)
print(f"h_n: {h_n.shape}")        # (2, 32, 20)
```

### 양방향 RNN

```python
rnn_bi = nn.RNN(
    input_size=10,
    hidden_size=20,
    num_layers=1,
    batch_first=True,
    bidirectional=True  # Bidirectional
)

output, h_n = rnn_bi(x)
print(f"output: {output.shape}")  # (32, 15, 40) - Forward+Backward
print(f"h_n: {h_n.shape}")        # (2, 32, 20) - Last state per direction
```

---

## 4. RNN 분류기 구현

### 시퀀스 분류 모델

```python
class RNNClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, num_layers=1):
        super().__init__()
        self.rnn = nn.RNN(
            input_size, hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x: (batch, seq, features)
        output, h_n = self.rnn(x)

        # Use last time step's hidden state
        # h_n[-1]: Last layer's hidden state
        out = self.fc(h_n[-1])
        return out
```

### Many-to-Many 구조

```python
class RNNSeq2Seq(nn.Module):
    """Sequence → Sequence"""
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        output, _ = self.rnn(x)
        # Apply FC to all time steps
        out = self.fc(output)  # (batch, seq, output_size)
        return out
```

---

## 5. 기울기 소실 문제

### 문제

```
In long sequences:
h100 ← W_hh × W_hh × ... × W_hh × h1
                    ↑
            100 multiplications → Exploding or vanishing gradients
```

### 원인

T 시간 단계 이후, 초기 입력에 대한 기울기는 다음 곱(product)에 비례합니다: grad ~ T 단계에 걸친 product_i(tanh'(.) * W_hh). |tanh'| <= 1이므로 이 곱은 T에 따라 지수적으로 줄어들어 — 네트워크는 초기 시간 단계의 정보를 사실상 "잊어버리게" 됩니다. 반대로 W_hh의 스펙트럴 반경(spectral radius)이 1을 초과하면 곱이 지수적으로 증가하여 기울기 폭발이 발생합니다.

- |W_hh| > 1: 기울기 폭발
- |W_hh| < 1: 기울기 소실

### 해결책

1. **LSTM/GRU 사용** (다음 레슨)
2. **Gradient Clipping**

```python
# Gradient clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

---

## 6. 시계열 예측 예제

### 사인파 예측

```python
import numpy as np

# Generate data
def generate_sin_data(seq_len=50, n_samples=1000):
    X = []
    y = []
    for _ in range(n_samples):
        start = np.random.uniform(0, 2*np.pi)
        seq = np.sin(np.linspace(start, start + 4*np.pi, seq_len + 1))
        X.append(seq[:-1].reshape(-1, 1))
        y.append(seq[-1])
    return np.array(X), np.array(y)

X, y = generate_sin_data()
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)

# Model
class SinPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.RNN(1, 32, batch_first=True)
        self.fc = nn.Linear(32, 1)

    def forward(self, x):
        _, h_n = self.rnn(x)
        return self.fc(h_n[-1]).squeeze()
```

---

## 7. 텍스트 분류 예제

### 문자 수준 RNN

```python
class CharRNN(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.RNN(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x: (batch, seq) - character indices
        embedded = self.embedding(x)  # (batch, seq, embed)
        output, h_n = self.rnn(embedded)
        out = self.fc(h_n[-1])
        return out

# Example
vocab_size = 27  # a-z + space
model = CharRNN(vocab_size, embed_size=32, hidden_size=64, num_classes=5)
```

---

## 8. 주의사항

### 입력 형태

```python
# batch_first=True  → (batch, seq, feature)
# batch_first=False → (seq, batch, feature)  # Default
```

### 가변 길이 시퀀스

```python
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# Padded sequences and actual lengths
padded_seqs = ...  # (batch, max_len, features)
lengths = ...      # Actual length of each sequence

# Pack (ignore padding)
packed = pack_padded_sequence(padded_seqs, lengths,
                               batch_first=True, enforce_sorted=False)
output, h_n = rnn(packed)

# Unpack
output_padded, _ = pad_packed_sequence(output, batch_first=True)
```

---

## 9. RNN 변형 비교

| 모델 | 장점 | 단점 |
|------|------|------|
| Simple RNN | 단순, 빠름 | 긴 시퀀스 학습 어려움 |
| LSTM | 장기 의존성 학습 | 복잡, 느림 |
| GRU | LSTM과 유사, 더 단순 | - |

---

## 정리

### 핵심 개념

1. **순환 구조**: 이전 상태가 다음 계산에 영향
2. **파라미터 공유**: 시간 독립적 가중치
3. **기울기 문제**: 긴 시퀀스에서 학습 어려움

### 핵심 코드

```python
rnn = nn.RNN(input_size, hidden_size, batch_first=True)
output, h_n = rnn(x)  # output: all, h_n: last
```

---

## 연습 문제

### 연습 1: 은닉 상태(Hidden State) 형태 이해하기

RNN 출력의 형태에 대한 이해를 검증하세요.

1. `nn.RNN(input_size=5, hidden_size=16, num_layers=2, batch_first=True)`를 생성하세요.
2. 형태 `(8, 20, 5)` (batch=8, seq_len=20, features=5)의 배치를 전달하세요.
3. `output`과 `h_n`의 형태를 출력하세요.
4. 각 차원이 의미하는 바를 설명하세요. 특히, `h_n.shape[0] == 2` (`num_layers`와 동일)인 이유는?
5. `bidirectional=True`로 반복하고 형태가 어떻게 변하는지 설명하세요.

### 연습 2: 사인파 예측

RNN을 사용하여 사인파 시퀀스의 다음 값을 예측하는 모델을 학습하세요.

1. 수업의 `generate_sin_data` 함수로 `seq_len=30`, 1000개 학습 샘플을 생성하세요.
2. MSE 손실과 Adam 옵티마이저로 `SinPredictor`를 50 에포크 학습하세요.
3. 테스트 예측 5개(모델 출력)를 실제값과 비교하여 그래프로 표시하세요.
4. 최종 테스트 MSE를 보고하고, 단순 RNN이 더 긴 시퀀스에서 어려움을 겪는 이유를 논하세요.

### 연습 3: 양방향(Bidirectional) RNN을 활용한 감성 분류

양방향 RNN 텍스트 분류기를 구성하세요.

1. 간단한 규칙으로 긍정(1) 또는 부정(0)으로 레이블된 200개 문장 장난감 데이터셋을 만드세요.
2. `bidirectional=True`로 `CharRNN`을 구성하세요.
3. FC 레이어 앞에서 순방향과 역방향의 최종 은닉 상태를 연결(Concatenation)하세요.
4. 20 에포크 학습하고 홀드아웃 20% 분할에서 정확도를 보고하세요.
5. 양방향성이 분류에 도움이 되지만 시퀀스 생성에는 적용할 수 없는 이유를 한 문장으로 설명하세요.

### 연습 4: 기울기 클리핑(Gradient Clipping) 효과

기울기 폭발을 관찰하고 클리핑이 이를 방지하는 방법을 확인하세요.

1. 100 타임 스텝 시퀀스에서 5층 스택 RNN을 구성하세요.
2. `torch.nn.init.normal_(std=2.0)`으로 큰 값의 가중치를 초기화하세요.
3. 순전파 및 역전파(클리핑 없이)를 실행하고 기울기 노름을 출력하세요.
4. `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)`을 적용하고 기울기 노름을 다시 출력하세요.
5. 기울기 클리핑 유무로 30 에포크 학습하고 손실 곡선을 비교하세요.

### 연습 5: 다대다(Many-to-Many) 시퀀스 레이블링

시퀀스의 모든 토큰에 레이블을 부여하는 POS 태깅 스타일 모델을 구현하세요.

1. 합성 데이터를 만드세요: 정수 시퀀스에서 각 원소의 레이블은 `element % 3` (3가지 클래스).
2. `RNNSeq2Seq`를 사용하여 각 타임 스텝에서 예측을 출력하세요.
3. 모든 타임 스텝에 걸쳐 `nn.CrossEntropyLoss`를 동시에 적용하세요.
4. 30 에포크 학습하고 토큰별 정확도를 보고하세요.
5. 3개의 테스트 샘플에 대해 예측 레이블 시퀀스와 실제 레이블을 시각화하세요.

---

## 다음 단계

[LSTM과 GRU](./14_LSTM_GRU.md)에서 LSTM과 GRU를 학습합니다.
