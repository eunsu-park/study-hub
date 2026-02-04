# 01. Linear & Logistic Regression

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
