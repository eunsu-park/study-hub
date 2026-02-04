# 02. Multi-Layer Perceptron (MLP)

## 개요

MLP는 딥러닝의 기본 building block입니다. **Backpropagation**을 통해 여러 레이어를 학습하는 방법을 이해하는 것이 핵심입니다.

## 학습 목표

1. **Forward Pass**: 다층 구조에서 순전파 이해
2. **Backward Pass**: Chain Rule을 이용한 역전파
3. **Activation Functions**: ReLU, Sigmoid, Tanh의 특성과 미분
4. **Weight Initialization**: 올바른 초기화의 중요성

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
