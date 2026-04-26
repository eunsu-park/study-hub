# 06. Multi-Layer Perceptron (MLP)

[이전: Linear & Logistic Regression](./05_Impl_Linear_Logistic.md) | [다음: CNN 기초](./07_CNN_Basics.md)

---

## 개요

MLP는 딥러닝의 기본 building block입니다. **Backpropagation**을 통해 여러 레이어를 학습하는 방법을 이해하는 것이 핵심입니다.

## 학습 목표

1. **Forward Pass**: 다층 구조에서 순전파 이해
2. **Backward Pass**: Chain Rule을 이용한 역전파
3. **Activation Functions**: ReLU, Sigmoid, Tanh의 특성과 미분
4. **Weight Initialization**: 올바른 초기화의 중요성

---

## 이론과 원리

MLP는 전체 역전파 메커니즘이 실제로 할 일이 있는 가장 간단한 신경망 아키텍처입니다. 한 은닉층을 통과하는 그래디언트를 손으로 추적하고 — 그것이 어떻게 L층으로 일반화되는지 보는 것은 — 본 과정 전체에서 가장 유용한 연습입니다. 이 공식을 유도할 수 있게 되면 앞으로 읽는 모든 PyTorch backward 안에서 그것들을 알아볼 수 있습니다.

이 섹션에서 다루는 내용:

- **A.** L층 MLP의 순전파와 모든 양의 형상
- **B.** 층별로 유도하는 역전파
- **C.** 초기화: Xavier와 He가 호환되지 않는 이유
- **D.** 너비 기반 스케일링이 중요한 이유 (분산 보존)

### A. L층 MLP의 순전파

L층, 배치 크기 N, 층 너비 `d_0 -> d_1 -> ... -> d_L`인 MLP에 대해:

```
z_l = h_{l-1} W_l + b_l                  (사전 활성화)       [N x d_l]
h_l = \sigma_l(z_l)                      (활성화)            [N x d_l]
h_0 = X                                  (입력)              [N x d_0]
\hat{y} = h_L                            (출력)              [N x d_L]
```

표기:
- `W_l \in R^{d_{l-1} x d_l}` (가중치 행렬), `b_l \in R^{d_l}` (배치 전체에 브로드캐스트되는 편향)
- `\sigma_l`은 층 l의 활성화입니다 (보통 은닉층은 ReLU, 분류 마지막 층은 softmax)

총 파라미터 개수는 `sum_l (d_{l-1} d_l + d_l)`이며, 행렬 항이 지배적입니다.

### B. 역전파: 한 번 유도하고 L번 적용

각 층의 상류 그래디언트를 정의:

```
\delta_l = dL / dz_l                     [N x d_l]
```

체인 룰이 깔끔한 점화식을 줍니다. 출력에서 시작:

```
\delta_L = (dL / d\hat{y}) \odot \sigma_L'(z_L)
```

Softmax + cross-entropy의 경우 이것이 극적으로 단순화됩니다: `\delta_L = (\hat{y} - y_onehot) / N`.

그리고 `l = L-1, L-2, ..., 1`에 대해:

```
\delta_l = (\delta_{l+1} W_{l+1}^T) \odot \sigma_l'(z_l)         [N x d_l]
dL/dW_l = h_{l-1}^T \delta_l                                    [d_{l-1} x d_l]
dL/db_l = 배치에 대한 \delta_l의 합                              [d_l]
```

기억할 두 가지 패턴:

1. **전치는 방향을 뒤집습니다.** 순전파는 `W_l`을 곱하고; 역전파는 `W_l^T`를 곱합니다. 순전파와 역전파 야코비안 사이의 "가중치 묶음(weight tying)"은 선형층 본질에 내재합니다.
2. **각 층의 그래디언트는 `\delta_{l+1}`과 지역 캐시된 `(h_{l-1}, z_l)`에만 의존합니다.** 이 성질이 역전파를 선형 시간이고 모듈러하게 만듭니다.

### C. 초기화: Xavier vs He

가중치를 `N(0, \sigma^2)`로 초기화하면, 한 층 후 `z_l`의 분산은 (활성화를 독립으로 취급) `Var(z_l) = d_{l-1} \sigma^2 Var(h_{l-1})`입니다. 층 간 분산을 일정하게 유지하려면(신호가 폭발하지도 소실되지도 않게) `\sigma^2 = 1 / d_{l-1}`을 선택합니다.

하지만 그 계산은 활성화가 선형(또는 기댓값에서 ReLU가 절반만 활성)이라고 가정합니다. 두 주요 활성화 체제에 대해:

- **Xavier / Glorot 초기화** (Glorot & Bengio 2010): `\sigma^2 = 2 / (d_{in} + d_{out})`. 활성화가 0 근처에서 거의 선형인 tanh / sigmoid를 위해 설계됨.
- **He 초기화** (He et al. 2015): `\sigma^2 = 2 / d_{in}`. 인자 2는 ReLU가 평균적으로 활성화의 절반을 죽이는 것을 보상합니다. ReLU/Leaky ReLU/GELU에 사용하세요.

활성화에 잘못된 것을 고르면 "깊은 MLP가 학습되지 않음"의 가장 흔한 이유 중 하나입니다 — 분산이 0으로 감소(역전파에서 그래디언트 소실)하거나 폭발(NaN 손실)합니다.

### D. 깊이에 따른 분산 보존

깊은 네트워크에서 이 초기화 규칙이 그토록 중요한 이유: 분산이 곱셈적으로 누적됩니다. 각 층이 분산을 인자 `c`만큼 변경한다면, L층 후 활성화 분산은 `c^L Var(h_0)`입니다. `c = 1.5`이고 `L = 50`이면 `1.5^50 \approx 6 * 10^8` — 그래디언트 폭발. `c = 0.5`이면 `0.5^50 \approx 10^{-15}` — 그래디언트 소실. 오직 `c \approx 1`만이 안정적 학습을 허용하며, 그것이 정확히 He / Xavier가 기댓값에서 달성하는 바입니다. BatchNorm과 LayerNorm은 신중한 초기화가 아닌 *측정*으로 같은 성질을 강제합니다.

### 이론에서 아래 코드로

| 이론 개념 | 본 레슨의 코드 구성 |
|-----------|---------------------|
| 순전파 `h_l = \sigma(h_{l-1} W_l + b_l)` | 손으로 쓴 순전파 루프 |
| 역전파 점화식 `\delta_l = (\delta_{l+1} W_{l+1}^T) \odot \sigma'(z_l)` | 손으로 쓴 역전파 루프 |
| ReLU용 He 초기화 | `nn.init.kaiming_normal_` 또는 수동 `* sqrt(2/d_in)` |
| 분산 보존 | 손실이 발산하지 않고 수렴한다는 경험적 관찰 |

---


## 수학적 배경

### 1. Forward Pass

```
입력: x ∈ ℝ^d₀

레이어 1: z₁ = W₁x + b₁,  a₁ = σ(z₁)
레이어 2: z₂ = W₂a₁ + b₂,  a₂ = σ(z₂)
...
출력:    ŷ = aₙ

여기서:
- Wᵢ ∈ ℝ^(dᵢ × dᵢ₋₁): 가중치 행렬
- bᵢ ∈ ℝ^dᵢ: 편향
- σ: 활성화 함수
```

### 2. Backward Pass (Backpropagation)

```
손실: L = Loss(y, ŷ)

Chain Rule:
∂L/∂Wᵢ = ∂L/∂aᵢ × ∂aᵢ/∂zᵢ × ∂zᵢ/∂Wᵢ

역전파 순서:
1. ∂L/∂ŷ (출력에서 손실의 미분)
2. ∂L/∂zₙ = ∂L/∂ŷ × σ'(zₙ)
3. ∂L/∂Wₙ = aₙ₋₁ᵀ × ∂L/∂zₙ
4. ∂L/∂aₙ₋₁ = ∂L/∂zₙ × Wₙᵀ
5. 반복...
```

### 3. 활성화 함수

```
ReLU:     σ(z) = max(0, z)
          σ'(z) = 1 if z > 0 else 0

Sigmoid:  σ(z) = 1/(1 + e⁻ᶻ)
          σ'(z) = σ(z)(1 - σ(z))

Tanh:     σ(z) = (eᶻ - e⁻ᶻ)/(eᶻ + e⁻ᶻ)
          σ'(z) = 1 - σ(z)²
```

---

## 파일 구조

```
02_MLP/
├── README.md
├── numpy/
│   ├── mlp_numpy.py          # 완전한 MLP 구현
│   ├── activations_numpy.py   # 활성화 함수들
│   └── test_mlp.py           # 테스트
├── pytorch_lowlevel/
│   └── mlp_lowlevel.py       # nn.Linear 없이 구현
├── paper/
│   └── mlp_paper.py          # Clean nn.Module
└── exercises/
    ├── 01_add_dropout.md
    ├── 02_batch_norm.md
    └── 03_xor_problem.md
```

---

## 핵심 개념

### 1. Vanishing/Exploding Gradients

```
문제: 레이어가 깊어지면 gradient가 사라지거나 폭발
- Sigmoid: σ'(z) ≤ 0.25 → 곱하면 0에 수렴
- 해결: ReLU, 적절한 초기화, BatchNorm, ResNet

예:
10 layers, Sigmoid → gradient ≈ 0.25^10 ≈ 10^-6
```

### 2. Xavier/He 초기화

```python
# Xavier (Glorot): tanh, sigmoid용
W = np.random.randn(in_dim, out_dim) * np.sqrt(1 / in_dim)
# 또는
W = np.random.randn(in_dim, out_dim) * np.sqrt(2 / (in_dim + out_dim))

# He (Kaiming): ReLU용
W = np.random.randn(in_dim, out_dim) * np.sqrt(2 / in_dim)
```

### 3. Universal Approximation Theorem

> 하나의 hidden layer를 가진 feedforward 네트워크는 충분한 뉴런이 있다면 임의의 연속 함수를 근사할 수 있다.

---

## 연습 문제

### 기초
1. XOR 문제 해결 (2-layer MLP)
2. 다양한 활성화 함수 비교
3. 초기화 방법에 따른 학습 곡선 비교

### 중급
1. Dropout 구현
2. Batch Normalization 구현
3. Learning Rate Scheduler 구현

### 고급
1. MNIST 분류 (98% 이상 정확도)
2. Gradient Clipping 구현
3. Weight Decay (L2 정규화) 구현

---

## 참고 자료

- Rumelhart et al. (1986). "Learning representations by back-propagating errors"
- Glorot & Bengio (2010). "Understanding the difficulty of training deep feedforward neural networks"
- He et al. (2015). "Delving Deep into Rectifiers" (He initialization)
