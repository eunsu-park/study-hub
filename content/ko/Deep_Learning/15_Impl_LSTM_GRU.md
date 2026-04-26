# 15. LSTM / GRU

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 시간 역전파(Backpropagation Through Time, BPTT) 공식을 사용하여 Vanilla RNN에서의 기울기 소실 문제(Vanishing Gradient Problem)를 설명하고, 이것이 장기 의존성 학습을 어떻게 제한하는지 서술합니다.
2. LSTM 아키텍처(입력 게이트, 망각 게이트, 출력 게이트 및 셀 상태(Cell State))를 설명하고, 셀 상태가 어떻게 장거리 기울기 흐름을 가능하게 하는지 설명합니다.
3. GRU 아키텍처(리셋 게이트, 업데이트 게이트)를 설명하고, 용량과 연산 비용 측면에서 GRU와 LSTM 간의 트레이드오프를 설명합니다.
4. PyTorch의 내장 RNN 모듈을 사용하지 않고 LSTM과 GRU 셀을 처음부터 구현하고, PyTorch의 `nn.LSTM`/`nn.GRU`와 비교 검증합니다.
5. 적층 LSTM/GRU 층을 사용하여 시퀀스-투-시퀀스(Sequence-to-Sequence) 모델을 구축하고 시계열 또는 NLP 작업에 적용합니다.
6. 벤치마크 시퀀스 작업에서 Vanilla RNN, LSTM, GRU의 성능과 학습 역학을 비교하고 결과를 해석합니다.

---

## 이론과 원리

LSTM과 GRU를 처음부터 구현하는 것은 `nn.LSTM`이 내부에서 실제로 무엇을 하는지, 그리고 시퀀스 학습이 작동(또는 실패)하게 하는 BPTT 메커니즘을 이해하는 가장 깔끔한 연습입니다. 이 섹션은 구현을 세 조각에 고정합니다: 정확한 게이트 방정식, PyTorch가 숨기는 펼치기 패턴, 셀 상태 고속도로를 통한 그래디언트 흐름.

이 섹션에서 다루는 내용:

- **A.** 방정식에서 텐서 연산으로: 네 게이트 행렬 곱
- **B.** 펼치기와 PyTorch가 BPTT 그래프를 어떻게 만들어 주는가
- **C.** 은닉 상태 vs 셀 상태 초기화와 detach
- **D.** 스텝별 vs 배치 계산과 CuDNN이 더 빠른 이유

### A. 방정식에서 텐서 연산으로

LSTM의 네 게이트는 네 개의 별도 행렬 곱이 아닌 하나의 결합된 행렬 곱으로 구현할 수 있습니다. 게이트 가중치를 연결:

```
W = [W_i ; W_f ; W_g ; W_o] in R^{4d x (d_in + d_h)}     (적층된 행)
```

그러면 단일 행렬-벡터 곱이 모든 게이트의 사전 활성화를 한 번에 계산합니다:

```
gates = W [x_t ; h_{t-1}] + b           in R^{4d}
i, f, g, o = gates.chunk(4, dim=-1)
i = sigmoid(i);   f = sigmoid(f)
g = tanh(g);      o = sigmoid(o)
c_t = f * c_{t-1} + i * g
h_t = o * tanh(c_t)
```

네 개 대신 한 번의 matmul인 이유는? 현대 GPU에서, 한 번의 큰 행렬 곱이 더 나은 메모리 지역성과 커널 런치 분산 비용 때문에 네 번의 작은 것보다 상당히 빠릅니다. PyTorch의 `nn.LSTM`은 이를 내부적으로 합니다; 커스텀 셀을 작성할 때도 그래야 합니다.

### B. 펼치기와 BPTT 그래프

PyTorch의 `nn.LSTM`은 `(seq_len, batch, d_in)` 텐서를 받아 모든 시간 스텝의 은닉 상태를 반환합니다. 내부적으로 이는 단지 `for` 루프입니다:

```
for t in range(T):
    h_t, c_t = cell(x_t, (h_{t-1}, c_{t-1}))
    outputs.append(h_t)
```

각 반복은 autograd DAG에 노드를 추가하며, 엣지가 `(h_{t-1}, c_{t-1}) -> (h_t, c_t)`로 갑니다. 루프 후 그래프는 `O(T)` 노드를 가지며; `loss.backward()`가 모두를 역순으로 순회 — 이것이 BPTT입니다. 모든 중간 활성화가 역전파를 위해 유지되기 때문에 메모리가 `T`에 선형적으로 자랍니다.

매우 긴 시퀀스의 경우 이는 금지적이 됩니다. 두 가지 해결책:

- **Truncated BPTT (TBPTT)**: 시퀀스를 길이 `K`의 청크로 처리하고, autograd 그래프를 끊되 값은 보존하기 위해 청크 사이에 `h.detach()`를 호출.
- **Gradient checkpointing**: 메모리를 절약하기 위해 역전파 중 선택된 순전파 스텝을 재실행, 메모리를 위해 계산을 거래.

### C. 은닉 상태와 셀 상태 초기화

`(h_0, c_0)`은 어딘가에서 시작해야 합니다. 세 가지 흔한 선택:

1. **0**: 가장 흔함; 시퀀스가 "신선"할 때 합리적.
2. **학습된 초기 상태**: `h_0`과 `c_0`을 파라미터로 등록; 네트워크가 최선의 시작점을 학습.
3. **이월된 상태**: 긴 문서의 언어 모델링에서, 배치 `b+1`의 `(h_0, c_0)`은 배치 `b`의 `(h_T, c_T).detach()`. Detach가 중요합니다: 그것 없이는 BPTT 그래프가 배치를 가로질러 확장되어 무한 메모리를 소비합니다.

Detach 패턴(값 이월, 그래프 폐기)은 RNN 학습에서 가장 오류가 발생하기 쉬운 부분 중 하나입니다. `.detach()` 잊으면 많은 배치 후 조용한 OOM을 일으킵니다; 잘못 사용하면 실제로 원했던 그래디언트 흐름을 자릅니다.

### D. 스텝별 vs 배치, 그리고 CuDNN

시간에 대한 순진한 Python `for` 루프는 스텝당 Python 오버헤드를 가집니다. PyTorch의 `nn.LSTM`(CUDA에서 실행 시)은 CuDNN의 손으로 최적화된 LSTM 커널을 호출하며, 이는 네 게이트 연산을 융합하고 GPU 메모리 대역폭에서 점화식을 실행합니다. 수동 루프 대비 속도 향상은 10-50배가 될 수 있습니다.

커스텀 셀을 구현할 때 보통 Python 루프 비용을 지불합니다 — 이해와 새로운 셀 설계에 유용하지만 프로덕션의 `nn.LSTM`과 경쟁할 수 없습니다. PyTorch 2.0의 `torch.compile`은 Python 루프를 융합 커널로 JIT 컴파일하여 이 격차를 부분적으로 줄이지만, 평이한 LSTM/GRU의 경우 정전적인 답은 "내장 모듈을 사용하라"입니다.

### 이론에서 아래 코드로

| 이론 개념 | 본 레슨의 코드 구성 |
|-----------|---------------------|
| 결합된 게이트 matmul | `gates = self.linear(torch.cat([x_t, h_prev], dim=-1)).chunk(4, dim=-1)` |
| 수동 펼치기 | `for t in range(seq_len): ...` |
| 상태 detach | 배치 사이의 `h.detach(), c.detach()` |
| CuDNN 고속 경로 | 커스텀 셀 vs `nn.LSTM(...)` |

---


## 개요

LSTM(Long Short-Term Memory)과 GRU(Gated Recurrent Unit)는 **vanishing gradient 문제**를 해결한 순환 신경망(RNN) 변형입니다. 게이트 메커니즘을 통해 장기 의존성(long-term dependency)을 효과적으로 학습합니다.

---

## 수학적 배경

### 1. Vanilla RNN의 문제

```
Vanilla RNN:
  h_t = tanh(W_h · h_{t-1} + W_x · x_t + b)

문제: Backpropagation Through Time (BPTT)

∂L/∂h_0 = ∂L/∂h_T · ∂h_T/∂h_{T-1} · ... · ∂h_1/∂h_0
        = ∂L/∂h_T · Π_{t=1}^{T} ∂h_t/∂h_{t-1}

∂h_t/∂h_{t-1} = diag(1 - tanh²(·)) · W_h

결과:
- |eigenvalue(W_h)| < 1 → Vanishing gradient
- |eigenvalue(W_h)| > 1 → Exploding gradient

→ 긴 시퀀스에서 초기 정보 학습 불가
```

### 2. LSTM 수식

```
입력: x_t (현재 입력), h_{t-1} (이전 hidden), c_{t-1} (이전 cell)
출력: h_t (현재 hidden), c_t (현재 cell)

1. Forget Gate (무엇을 버릴까?)
   f_t = σ(W_f · [h_{t-1}, x_t] + b_f)

2. Input Gate (무엇을 저장할까?)
   i_t = σ(W_i · [h_{t-1}, x_t] + b_i)

3. Candidate Cell (새로운 정보)
   c̃_t = tanh(W_c · [h_{t-1}, x_t] + b_c)

4. Cell State 업데이트
   c_t = f_t ⊙ c_{t-1} + i_t ⊙ c̃_t
        ↑ 이전 정보   ↑ 새 정보

5. Output Gate (무엇을 출력할까?)
   o_t = σ(W_o · [h_{t-1}, x_t] + b_o)

6. Hidden State
   h_t = o_t ⊙ tanh(c_t)

σ: sigmoid (0~1)
⊙: element-wise 곱
```

### 3. GRU 수식

```
GRU: LSTM의 간소화 버전 (cell state 없음)

1. Reset Gate (이전 정보 얼마나 무시?)
   r_t = σ(W_r · [h_{t-1}, x_t] + b_r)

2. Update Gate (이전 vs 새 정보 비율)
   z_t = σ(W_z · [h_{t-1}, x_t] + b_z)

3. Candidate Hidden
   h̃_t = tanh(W_h · [r_t ⊙ h_{t-1}, x_t] + b_h)

4. Hidden State
   h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ h̃_t
        ↑ 이전 정보 유지      ↑ 새 정보

LSTM vs GRU:
- GRU: 2개 게이트 (reset, update)
- LSTM: 3개 게이트 (forget, input, output) + cell state
- GRU가 파라미터 25% 적음
- 성능은 task에 따라 비슷
```

### 4. 왜 Gradient가 보존되는가?

```
LSTM Cell State 업데이트:
  c_t = f_t ⊙ c_{t-1} + i_t ⊙ c̃_t

Gradient:
  ∂c_t/∂c_{t-1} = f_t  (forget gate)

f_t ≈ 1이면 gradient가 거의 그대로 전파!

이것이 "highway" 역할:
- Cell state가 변형 없이 흐를 수 있음
- 긴 시퀀스에서도 gradient 유지
- 모델이 f_t를 학습해 어떤 정보를 유지할지 결정
```

---

## 아키텍처

### LSTM 구조 다이어그램

```
                      ┌─────────────────────────────────┐
                      │           Cell State c_t         │
c_{t-1} ─────────────►│   ⊙────────────+────────────►  c_t
                      │   ↑ forget     ↑ input           │
                      │   f_t        i_t ⊙ c̃_t          │
                      │                                  │
                      │   ┌──────────────────────┐       │
                      │   │  σ   σ   tanh   σ   │       │
                      │   │  f   i    c̃    o   │       │
                      │   └──────────────────────┘       │
                      │         ↑                        │
                      │    [h_{t-1}, x_t]                │
h_{t-1} ─────────────►│                                  ├───► h_t
                      │                 ⊙ ◄── tanh(c_t)  │
                      │                 o_t              │
                      └─────────────────────────────────┘
                                    ↑
                                   x_t
```

### GRU 구조 다이어그램

```
                      ┌─────────────────────────────────┐
                      │                                  │
h_{t-1} ─────────────►│ ⊙────────────+────────────────►│ h_t
                      │ (1-z)        z ⊙ h̃             │
                      │              ↑                   │
                      │   ┌──────────────────────┐       │
                      │   │    σ   σ   tanh     │       │
                      │   │    r   z    h̃      │       │
                      │   └──────────────────────┘       │
                      │         ↑                        │
                      │    [h_{t-1}, x_t]                │
                      │    [r⊙h_{t-1}, x_t]              │
                      └─────────────────────────────────┘
                                    ↑
                                   x_t
```

### 파라미터 수

```
LSTM:
  4개 게이트 × (input_size × hidden_size + hidden_size × hidden_size + hidden_size)
  = 4 × (input_size + hidden_size + 1) × hidden_size

예: input=128, hidden=256
  = 4 × (128 + 256 + 1) × 256 = 394,240

GRU:
  3개 게이트
  = 3 × (input_size + hidden_size + 1) × hidden_size

예: input=128, hidden=256
  = 3 × (128 + 256 + 1) × 256 = 295,680  (25% 적음)
```

---

## 파일 구조

```
06_LSTM_GRU/
├── README.md                      # 이 파일
├── numpy/
│   ├── lstm_numpy.py             # NumPy LSTM (forward + backward)
│   └── gru_numpy.py              # NumPy GRU
├── pytorch_lowlevel/
│   └── lstm_gru_lowlevel.py      # F.linear 사용, nn.LSTM 미사용
├── paper/
│   ├── lstm_paper.py             # 원본 1997 논문 구현
│   └── gru_paper.py              # 2014 논문 구현
└── exercises/
    ├── 01_gradient_flow.md       # BPTT gradient 분석
    └── 02_sequence_tasks.md      # 시퀀스 분류/생성
```

---

## 핵심 개념

### 1. 게이트의 역할

```
Forget Gate (f):
- 1에 가까움: 이전 정보 유지
- 0에 가까움: 이전 정보 삭제
- 예: 새 문장 시작 시 이전 문맥 리셋

Input Gate (i):
- 새 정보의 중요도 결정
- Candidate (c̃)와 함께 작동

Output Gate (o):
- Cell state 중 무엇을 hidden으로 노출
- 예: 내부적으로는 기억하지만 출력하지 않음
```

### 2. Peephole Connection (선택적)

```
기본 LSTM: 게이트가 [h_{t-1}, x_t]만 참조
Peephole: 게이트가 c_{t-1}도 참조

f_t = σ(W_f · [h_{t-1}, x_t] + W_{cf} · c_{t-1} + b_f)
i_t = σ(W_i · [h_{t-1}, x_t] + W_{ci} · c_{t-1} + b_i)
o_t = σ(W_o · [h_{t-1}, x_t] + W_{co} · c_t + b_o)

효과: cell state 정보를 게이트 결정에 직접 활용
```

### 3. Bidirectional LSTM

```
시퀀스를 양방향으로 처리:

Forward:  → h_1 → h_2 → h_3 → h_4 →
Backward: ← h_4 ← h_3 ← h_2 ← h_1 ←

출력: [forward_h_t; backward_h_t] (concatenate)

장점:
- 미래 컨텍스트도 활용
- NER, POS tagging에 효과적
- Transformer 등장 전 NLP 표준
```

### 4. Stacked LSTM

```
여러 LSTM 레이어 쌓기:

x_t → LSTM_1 → h_t^1 → LSTM_2 → h_t^2 → ... → output

각 레이어:
- 이전 레이어의 hidden을 입력으로
- 더 추상적인 표현 학습

주의: 깊어질수록 학습 어려움
- Dropout 필수 (특히 레이어 간)
- Residual connection 도움
```

---

## 구현 레벨

### Level 1: NumPy From-Scratch (numpy/)

- 모든 게이트 연산 직접 구현
- BPTT gradient 수동 계산
- Cell state gradient 유도

### Level 2: PyTorch Low-Level (pytorch_lowlevel/)

- F.linear, torch.sigmoid, torch.tanh 사용
- nn.LSTM 미사용
- 파라미터 수동 관리
- Bidirectional, Stacked 구현

### Level 3: Paper Implementation (paper/)

- Hochreiter & Schmidhuber (1997) LSTM
- Cho et al. (2014) GRU
- Peephole connections

---

## 학습 체크리스트

- [ ] Vanilla RNN의 vanishing gradient 문제
- [ ] LSTM 4개 수식 암기
- [ ] GRU 3개 수식 암기
- [ ] Cell state가 gradient를 보존하는 이유
- [ ] 각 게이트의 역할 설명
- [ ] LSTM vs GRU 장단점
- [ ] BPTT 구현
- [ ] Bidirectional, Stacked 구조

---

## 참고 자료

- Hochreiter & Schmidhuber (1997). "Long Short-Term Memory"
- Cho et al. (2014). "Learning Phrase Representations using RNN Encoder-Decoder"
- [colah's blog: Understanding LSTM](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- [d2l.ai: LSTM](https://d2l.ai/chapter_recurrent-modern/lstm.html)
- [../02_MLP/README.md](../02_MLP/README.md)
