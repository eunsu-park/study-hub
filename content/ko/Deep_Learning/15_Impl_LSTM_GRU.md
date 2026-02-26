# 06. LSTM / GRU

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 시간 역전파(Backpropagation Through Time, BPTT) 공식을 사용하여 Vanilla RNN에서의 기울기 소실 문제(Vanishing Gradient Problem)를 설명하고, 이것이 장기 의존성 학습을 어떻게 제한하는지 서술합니다.
2. LSTM 아키텍처(입력 게이트, 망각 게이트, 출력 게이트 및 셀 상태(Cell State))를 설명하고, 셀 상태가 어떻게 장거리 기울기 흐름을 가능하게 하는지 설명합니다.
3. GRU 아키텍처(리셋 게이트, 업데이트 게이트)를 설명하고, 용량과 연산 비용 측면에서 GRU와 LSTM 간의 트레이드오프를 설명합니다.
4. PyTorch의 내장 RNN 모듈을 사용하지 않고 LSTM과 GRU 셀을 처음부터 구현하고, PyTorch의 `nn.LSTM`/`nn.GRU`와 비교 검증합니다.
5. 적층 LSTM/GRU 층을 사용하여 시퀀스-투-시퀀스(Sequence-to-Sequence) 모델을 구축하고 시계열 또는 NLP 작업에 적용합니다.
6. 벤치마크 시퀀스 작업에서 Vanilla RNN, LSTM, GRU의 성능과 학습 역학을 비교하고 결과를 해석합니다.

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
