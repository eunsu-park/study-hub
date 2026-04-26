# 05. Linear & Logistic Regression

[이전: 훈련 기법](./04_Training_Techniques.md) | [다음: Multi-Layer Perceptron (MLP)](./06_Impl_MLP.md)

---

## 개요

선형 회귀와 로지스틱 회귀는 딥러닝의 가장 기본적인 building block입니다. 신경망의 각 레이어는 본질적으로 선형 변환 + 비선형 활성화의 조합입니다.

## 학습 목표

1. **수학적 이해**
   - Gradient Descent 원리
   - Loss Function (MSE, Cross-Entropy)
   - 행렬 미분

2. **구현 능력**
   - Forward/Backward pass 직접 구현
   - 가중치 초기화
   - 학습 루프 작성

3. **실습**
   - MNIST 이진 분류
   - 과적합/정규화 실험

---

## 이론과 원리

선형 회귀와 로지스틱 회귀는 본 과정에서 가장 간단한 모델이며, 그 단순함이 바로 이들이 신중한 이론적 다룸을 받을 가치가 있는 이유입니다. 둘 다 닫힌 형태(closed-form) 또는 거의 닫힌 형태의 해를 가지고, 둘 다 볼록이며, 손으로 구현하는 방식이 정확히 이후 모든 네트워크의 모든 이후 층으로 확장되는 템플릿입니다. 여기의 수학을 이해하면 이후 모든 것이 친숙하게 느껴집니다.

이 섹션에서 다루는 내용:

- **A.** 선형 회귀: 닫힌 형태 해 vs 경사 하강
- **B.** 베르누이 모델 하의 최대 우도(maximum likelihood)로서 로지스틱 회귀
- **C.** 회귀에 MSE, 분류에 cross-entropy인 이유 (반대가 아닌)
- **D.** 볼록성, 유일성, 그리고 은닉층을 추가하면 무엇이 바뀌는가

### A. 선형 회귀: 닫힌 형태 vs 경사 하강

선형 회귀는 평균 제곱 오차를 최소화합니다:

```
L(w) = (1 / 2N) * ||X w - y||^2
```

`\nabla_w L = 0`을 두면 **정규 방정식(normal equations)**과 닫힌 형태 해가 나옵니다:

```
\nabla_w L = (1 / N) X^T (X w - y) = 0
=>  w* = (X^T X)^{-1} X^T y                  (X^T X가 가역일 때)
```

닫힌 형태가 존재하는데 왜 경사 하강을 사용할까요? 세 가지 이유:

1. **비용.** `(X^T X)^{-1}`은 `d`개 특징에 대해 `O(d^3)`입니다. `d = 10^6`이면 불가능합니다. 경사 하강은 스텝당 `O(N * d)` 비용입니다.
2. **메모리.** `X^T X`를 만들려면 `O(d^2)` 메모리가 필요합니다; 고차원 특징(이미지, 텍스트)에서는 이것이 RAM을 초과합니다.
3. **일반성.** 경사 하강은 비선형 모델에도 변경 없이 확장됩니다. 닫힌 형태는 그렇지 않습니다.

따라서 닫힌 형태는 경사 하강의 *목표*입니다: 스텝이 0으로, 반복이 무한대로 가면 `w_t -> w*`.

### B. 최대 우도로서 로지스틱 회귀

로지스틱 회귀는 라벨 `y \in {0, 1}`이 `x` 조건부 베르누이 분포라고 가정합니다:

```
p(y=1 | x) = \sigma(w^T x + b),       \sigma(z) = 1 / (1 + e^{-z})
```

데이터셋에 대한 음의 로그 우도(NLL)는:

```
NLL(w) = - sum_i [ y_i log p_i + (1 - y_i) log(1 - p_i) ]
```

이것이 정확히 **이진 cross-entropy**입니다. 따라서 cross-entropy 최소화는 베르누이 모델 하의 최대 우도 추정 *바로 그것*입니다. 그래디언트는 놀랍도록 깔끔한 형태를 가집니다:

```
\nabla_w NLL = sum_i (p_i - y_i) x_i
```

`(p_i - y_i)` 항에 주목하세요: 이는 확률 공간에서의 예측 오차입니다. sigmoid의 도함수가 log의 도함수와 깔끔하게 상쇄됩니다 — 이것이 그래디언트에 `\sigma'` 항이 나타나지 않는 이유입니다.

### C. 손실 함수 선택: MSE vs Cross-Entropy

왜 연속 목표값에는 MSE, 클래스 라벨에는 cross-entropy인가요?

**회귀의 MSE**는 가우시안 잡음 가정에서 옵니다: `y = w^T x + \epsilon`, `\epsilon ~ N(0, \sigma^2)`. NLL은 `(1 / 2 \sigma^2) (y - w^T x)^2 + const`이며, 상수까지 MSE입니다.

**분류의 Cross-entropy**는 위의 베르누이/범주형 NLL에서 옵니다. sigmoid 출력으로 분류에 MSE를 사용하면 그래디언트는:

```
\nabla_w MSE = (p - y) * \sigma'(z) * x = (p - y) * p (1 - p) * x
```

추가 인자 `p (1 - p)`는 예측이 *틀렸을* 때조차 (예: `p = 0.99`인데 `y = 0`) `p \to 0`이나 `p \to 1`이면 사라집니다. 가장 강하게 되돌려 밀고 싶을 때 정확히 그래디언트가 평탄해집니다. Cross-entropy는 이를 피하며 — 그런 감쇠 인자가 없습니다 — 따라서 엄격히 선호됩니다.

### D. 볼록성과 은닉층이 바꾸는 것

선형 회귀의 MSE와 로지스틱 회귀의 NLL 모두 `w`에 대해 **볼록**합니다. 볼록성은 임의의 지역 최솟값이 전역 최솟값임을 보장하므로, (적절한 스텝 크기의) 경사 하강은 임의의 시작점에서 최적해로 수렴합니다. 사실상 튜닝할 것이 없습니다.

비선형성을 가진 은닉층을 단 하나만 추가해도 볼록성이 깨집니다. 손실 표면은 많은 지역 최솟값, 안장점, 평지를 가진 비볼록 지형이 됩니다. 이것이 딥러닝에 다음이 필요한 이유입니다:

- 신중한 초기화("좋은" 영역에서 시작하기 위해),
- vanilla GD 대신 적응형 옵티마이저(Adam),
- 정규화(dropout, BN)로 지형을 평탄화,
- 볼록 최적화에 대응 개념이 없는 학습률 워밍업 같은 트릭.

본 레슨 모델의 단순함이 모든 것이 "그냥 작동"한다는 보장을 받는 마지막 순간입니다.

### 이론에서 아래 코드로

| 이론 개념 | 본 레슨의 코드 구성 |
|-----------|---------------------|
| MSE 그래디언트 `(1/N) X^T (X w - y)` | `dL_dw = X.T @ (pred - y) / N` |
| 닫힌 형태 `w* = (X^T X)^{-1} X^T y` | `np.linalg.lstsq`로 선택적 검증 |
| Sigmoid + BCE 깔끔한 그래디언트 | `dL_dw = X.T @ (p - y)` (`\sigma'` 인자 없음) |
| 볼록성 보장 | 두 모델이 임의의 초기화에서 수렴한다는 사실 |

---


## 수학적 배경

### 1. Linear Regression

```
모델:    ŷ = Xw + b
손실:    L = (1/2n) Σ(y - ŷ)²  (MSE)

그래디언트:
∂L/∂w = (1/n) X^T (ŷ - y)
∂L/∂b = (1/n) Σ(ŷ - y)

업데이트:
w ← w - η × ∂L/∂w
b ← b - η × ∂L/∂b
```

### 2. Logistic Regression

```
모델:    z = Xw + b
         ŷ = σ(z) = 1/(1 + e^(-z))

손실:    L = -(1/n) Σ[y·log(ŷ) + (1-y)·log(1-ŷ)]  (BCE)

그래디언트:
∂L/∂w = (1/n) X^T (ŷ - y)  ← 놀랍게도 Linear와 같은 형태!
∂L/∂b = (1/n) Σ(ŷ - y)
```

---

## 파일 구조

```
01_Linear_Logistic/
├── README.md                 # 이 파일
├── theory.md                 # 상세 이론 (수학적 유도)
├── numpy/
│   ├── linear_numpy.py       # Linear Regression (NumPy)
│   ├── logistic_numpy.py     # Logistic Regression (NumPy)
│   └── test_numpy.py         # 단위 테스트
├── pytorch_lowlevel/
│   ├── linear_lowlevel.py    # PyTorch 기본 ops 사용
│   └── logistic_lowlevel.py
├── paper/
│   └── linear_paper.py       # 클린한 nn.Module 구현
└── exercises/
    ├── 01_regularization.md  # L1/L2 정규화 추가
    └── 02_softmax.md         # Softmax 확장
```

---

## 빠른 시작

### NumPy 구현 실행

```bash
cd numpy/
python linear_numpy.py      # 선형 회귀 학습
python logistic_numpy.py    # 로지스틱 회귀 학습
python test_numpy.py        # 테스트 실행
```

### PyTorch 구현 실행

```bash
cd pytorch_lowlevel/
python linear_lowlevel.py
```

---

## 핵심 개념

### 1. Gradient Descent

```python
# 기본 알고리즘
for epoch in range(n_epochs):
    # Forward
    y_pred = model.forward(X)

    # Loss
    loss = compute_loss(y, y_pred)

    # Backward (gradient 계산)
    gradients = compute_gradients(y, y_pred)

    # Update
    model.weights -= learning_rate * gradients
```

### 2. 행렬 미분 (중요!)

```
∂(Xw)/∂w = X^T
∂(w^T X^T)/∂w = X
∂(||Xw - y||²)/∂w = 2 X^T (Xw - y)
```

### 3. Sigmoid와 그 미분

```python
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    s = sigmoid(z)
    return s * (1 - s)  # σ(z)(1 - σ(z))
```

---

## 연습 문제

### 기초
1. Linear Regression에 bias 없이 구현해보기
2. 학습률(lr)을 바꾸며 수렴 속도 관찰
3. Batch vs Stochastic Gradient Descent 비교

### 중급
1. L2 정규화 추가 (Ridge)
2. L1 정규화 추가 (Lasso)
3. Mini-batch GD 구현

### 고급
1. Momentum, Adam 옵티마이저 구현
2. Early Stopping 구현
3. Softmax Regression (다중 클래스) 확장

---

## 참고 자료

- CS229 (Stanford) Lecture Notes
- Deep Learning Book Chapter 5, 6
- [Coursera ML - Andrew Ng](https://www.coursera.org/learn/machine-learning)
