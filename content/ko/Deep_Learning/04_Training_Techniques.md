# 04. 학습 기법

[이전: 역전파](./03_Backpropagation.md) | [다음: Linear & Logistic Regression](./05_Impl_Linear_Logistic.md)

---

## 학습 목표

- 경사 하강법 변형 이해 (SGD, Momentum, Adam)
- 학습률 스케줄링
- 정규화 기법 (Dropout, Weight Decay, Batch Norm)
- 과적합 방지와 조기 종료

---

## 1. 경사 하강법 (Gradient Descent)

### 기본 원리

```
W(t+1) = W(t) - η × ∇L
```
- η: 학습률 (learning rate)
- ∇L: 손실 함수의 기울기

### 변형들

| 방법 | 수식 | 특징 |
|------|------|------|
| SGD | W -= lr × g | 단순, 느림 |
| Momentum | v = βv + g; W -= lr × v | 관성 추가 |
| AdaGrad | 적응적 학습률 | 희소 데이터에 유리 |
| RMSprop | 지수 이동 평균 | AdaGrad 개선 |
| Adam | Momentum + RMSprop | 가장 보편적 |

---

## 2. Momentum

언덕이 많은 지형을 굴러 내려가는 공을 생각해 보세요. 모멘텀(Momentum) 없이는 공이 국소적인 기울기에만 따라 움직이므로 작은 웅덩이에 갇히거나 좁은 계곡에서 진동할 수 있습니다. 모멘텀이 있으면 공은 과거 기울기로부터 속도를 누적합니다. 이를 통해 작은 극소(Local Minima)를 통과하고, 기울기 부호가 계속 바뀌는 방향에서의 진동을 억제할 수 있습니다.

관성을 추가하여 진동을 줄입니다.

```
v(t) = β × v(t-1) + ∇L
W(t+1) = W(t) - η × v(t)
```

### NumPy 구현

```python
def sgd_momentum(W, grad, v, lr=0.01, beta=0.9):
    # Why exponential moving average?  v accumulates past gradients with
    # exponential decay (beta=0.9 means ~10-step memory).  This smooths out
    # noisy per-sample gradients and builds up speed in consistent directions.
    v = beta * v + grad          # Update velocity
    W = W - lr * v               # Update weights using smoothed direction
    return W, v
```

### PyTorch

```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
```

---

## 3. Adam Optimizer

Momentum과 RMSprop의 장점을 결합합니다.

```
m(t) = β₁ × m(t-1) + (1-β₁) × g      # 1st moment (mean of gradients)
v(t) = β₂ × v(t-1) + (1-β₂) × g²     # 2nd moment (mean of squared gradients)
m̂ = m / (1 - β₁ᵗ)                    # Bias correction
v̂ = v / (1 - β₂ᵗ)
W = W - η × m̂ / (√v̂ + ε)
```

**편향 보정(Bias Correction)이 필요한 이유는?** `m`과 `v` 모두 0으로 초기화됩니다. 기본값 `β₁=0.9`에서 첫 번째 스텝 `t=1`일 때: `m₁ = 0.9 × 0 + 0.1 × g₁ = 0.1 × g₁`. 이는 실제 기울기 평균을 10배 과소 추정합니다. `(1 - β₁^t) = (1 - 0.9^1) = 0.1`로 나누면 이를 보정합니다: `m̂₁ = 0.1 × g₁ / 0.1 = g₁`. 이 보정 없이는 초반 몇 번의 업데이트가 너무 작아집니다. `t`가 커질수록 `β₁^t → 0`이 되므로 편향은 사라집니다.

### NumPy 구현

```python
def adam(W, grad, m, v, t, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
    # Why first moment?  m tracks the exponential moving average of gradients —
    # this is the "momentum" component, smoothing noisy gradient estimates.
    m = beta1 * m + (1 - beta1) * grad
    # Why second moment?  v tracks the EMA of squared gradients — this estimates
    # per-parameter gradient variance, giving each weight its own adaptive lr.
    v = beta2 * v + (1 - beta2) * (grad ** 2)

    # Why bias correction?  m and v are initialized to 0, so early estimates
    # are biased toward zero.  Dividing by (1 - beta^t) exactly cancels this.
    m_hat = m / (1 - beta1 ** t)
    v_hat = v / (1 - beta2 ** t)

    # Why divide by sqrt(v_hat)?  Parameters with large gradient variance get
    # smaller effective lr (cautious updates), while stable gradients get larger
    # lr (confident updates).  eps prevents division by zero.
    W = W - lr * m_hat / (np.sqrt(v_hat) + eps)
    return W, m, v
```

### PyTorch

```python
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
```

---

## 4. 학습률 스케줄링

왜 고정 학습률을 사용하지 않을까요? 높은 학습률은 초기에 빠른 진전을 가능하게 합니다 — 모델이 손실 지형의 나쁜 영역에서 빠르게 벗어납니다 — 하지만 최적점 근처에서 진동하거나 오버슈팅이 발생합니다. 낮은 학습률은 부드럽게 수렴하지만 초기 에폭이 너무 느립니다. 학습률 스케줄링은 이 두 가지 장점을 모두 제공합니다: 초반에는 높게 시작해 속도를 높이고, 시간이 지남에 따라 감소시켜 정밀하게 수렴합니다.

학습 중 학습률을 조절합니다.

### 주요 방법

| 방법 | 특징 |
|------|------|
| Step Decay | N 에폭마다 γ 배로 감소 |
| Exponential | lr = lr₀ × γᵉᵖᵒᶜʰ |
| Cosine Annealing | 코사인 함수로 감소 |
| ReduceLROnPlateau | 검증 손실 정체 시 감소 |
| Warmup | 초기에 점진적 증가 |

### PyTorch 예시

```python
# Step Decay
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

# Cosine Annealing
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

# ReduceLROnPlateau
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', patience=10, factor=0.5
)

# In training loop
for epoch in range(epochs):
    train(...)
    scheduler.step()  # Call at end of epoch
```

---

## 5. Dropout

학습 중 랜덤하게 뉴런을 비활성화합니다.

### 원리

```
Training: y = x × mask / (1 - p)   # mask is Bernoulli(1-p)
Inference: y = x                   # No mask
```

### NumPy 구현

```python
def dropout(x, p=0.5, training=True):
    if not training:
        return x
    mask = (np.random.rand(*x.shape) > p).astype(float)
    return x * mask / (1 - p)
```

### PyTorch

```python
class MLPWithDropout(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_p=0.5):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(p=dropout_p)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)  # Active only during training
        x = self.fc2(x)
        return x

# During inference
model.eval()  # Disable dropout
```

---

## 6. Batch Normalization

각 층의 입력을 정규화합니다.

### 수식

```
μ = mean(x)
σ² = var(x)
x̂ = (x - μ) / √(σ² + ε)
y = γ × x̂ + β   # Learnable parameters
```

### NumPy 구현

```python
def batch_norm(x, gamma, beta, eps=1e-5, training=True,
               running_mean=None, running_var=None, momentum=0.1):
    if training:
        mean = np.mean(x, axis=0)
        var = np.var(x, axis=0)

        # Update running averages
        if running_mean is not None:
            running_mean = momentum * mean + (1 - momentum) * running_mean
            running_var = momentum * var + (1 - momentum) * running_var
    else:
        mean = running_mean
        var = running_var

    x_norm = (x - mean) / np.sqrt(var + eps)
    return gamma * x_norm + beta
```

### PyTorch

```python
class CNNWithBatchNorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3)
        self.bn1 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(64, 10)
        self.bn_fc = nn.BatchNorm1d(10)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = x.flatten(1)
        x = self.bn_fc(self.fc1(x))
        return x
```

---

## 7. Weight Decay (L2 정규화)

왜 큰 가중치에 패널티를 줄까요? 큰 가중치는 작은 입력 변화를 증폭시켜 모델이 학습 노이즈에 과도하게 민감해집니다(과적합). L2 정규화는 손실에 `lambda * ||W||^2`를 더해 모든 가중치를 0 방향으로 비례적으로 축소시킵니다 — 부드럽고 완만한 수축입니다. 반면 L1 정규화는 일부 가중치를 *정확히* 0으로 만듭니다(특징 선택에는 유용하지만 딥러닝에서는 덜 사용됨). L2가 기본 선택인 이유는 더 부드러운 최적화 지형을 만들기 때문입니다.

가중치 크기에 패널티를 부여합니다.

### 수식

```
L_total = L_data + λ × ||W||²
∇L_total = ∇L_data + 2λW
```

### PyTorch

```python
# Method 1: Set in optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

# Method 2: Add directly to loss
l2_lambda = 1e-4
l2_reg = sum(p.pow(2).sum() for p in model.parameters())
loss = criterion(output, target) + l2_lambda * l2_reg
```

---

## 8. 조기 종료 (Early Stopping)

검증 손실이 개선되지 않으면 학습을 중단합니다.

### PyTorch 구현

```python
class EarlyStopping:
    def __init__(self, patience=10, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

# Usage
early_stopping = EarlyStopping(patience=10)
for epoch in range(epochs):
    train_loss = train(model, train_loader)
    val_loss = validate(model, val_loader)

    early_stopping(val_loss)
    if early_stopping.early_stop:
        print(f"Early stopping at epoch {epoch}")
        break
```

---

## 9. 데이터 증강 (Data Augmentation)

훈련 데이터를 변형하여 다양성을 증가시킵니다.

### 이미지 데이터

```python
from torchvision import transforms

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
```

---

## 10. NumPy vs PyTorch 비교

### Optimizer 구현

```python
# NumPy (manual implementation)
m = np.zeros_like(W)
v = np.zeros_like(W)
for t in range(1, epochs + 1):
    grad = compute_gradient(W, X, y)
    m = beta1 * m + (1 - beta1) * grad
    v = beta2 * v + (1 - beta2) * grad**2
    m_hat = m / (1 - beta1**t)
    v_hat = v / (1 - beta2**t)
    W -= lr * m_hat / (np.sqrt(v_hat) + eps)

# PyTorch (automatic)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
for epoch in range(epochs):
    loss = criterion(model(X), y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

---

## 정리

### 핵심 개념

1. **Optimizer**: Adam이 기본 선택, SGD+Momentum도 여전히 유효
2. **학습률**: 적절한 스케줄링으로 수렴 개선
3. **정규화**: Dropout, BatchNorm, Weight Decay 조합
4. **조기 종료**: 과적합 방지의 기본

### 권장 시작 설정

```python
# Basic configuration
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
```

---

## 연습 문제

### 연습 1: Adam vs SGD 수렴 비교

간단한 회귀 태스크에서 Adam과 SGD 옵티마이저(Optimizer)를 비교하세요.

1. 합성 데이터 `y = 2x + 1 + noise`를 200개 샘플로 생성하세요.
2. 단일 선형 층을 `torch.optim.SGD(lr=0.01)`과 `torch.optim.Adam(lr=0.001)`으로 각각 200 에포크 동안 학습하세요.
3. 각 옵티마이저의 학습 손실 곡선을 기록하고 시각화하세요.
4. 각 옵티마이저의 수렴 속도와 손실 곡선의 안정성을 비교하세요.

### 연습 2: Adam 직접 구현하기

NumPy만 사용하여 Adam 업데이트 규칙을 직접 구현하세요.

1. 단일 매개변수 `W`에 대해 `m=0`, `v=0`, `t=0`으로 초기화하세요.
2. `N(0, 1)`에서 추출한 난수 기울기를 사용하여 100번 기울기 갱신을 시뮬레이션하세요.
3. 각 단계에서 편향 보정(Bias Correction)이 적용된 Adam 업데이트를 실행하세요.
4. 최종 `W` 값을 `torch.optim.Adam`과 비교하여 올바른지 확인하세요.

### 연습 3: 학습률 스케줄링(Learning Rate Scheduling) 비교

다양한 스케줄이 학습 동역학에 미치는 영향을 관찰하세요.

1. Adam (lr=0.01)으로 MNIST에서 소형 MLP를 20 에포크 학습하세요.
2. 세 가지 스케줄러를 비교하세요: 스케줄링 없음, `StepLR(step_size=5, gamma=0.5)`, `CosineAnnealingLR(T_max=20)`.
3. 세 경우에 대한 검증 정확도 vs. 에포크 그래프를 그리세요.
4. 처음에 높은 학습률을 사용하다 감소시키는 방식이 왜 일정한 낮은 학습률보다 유리한지 설명하세요.

### 연습 4: 드롭아웃(Dropout) 정규화 효과

드롭아웃이 과적합(Overfitting)을 줄이는 효과를 경험적으로 검증하세요.

1. MNIST 학습 데이터에서 500개 포인트를 샘플링하여 소형 데이터셋을 만드세요.
2. 동일한 3층 MLP 두 개를 50 에포크 학습하세요 — 하나는 `Dropout(0.5)`, 하나는 드롭아웃 없이.
3. 학습 정확도와 검증 정확도를 모두 기록하세요.
4. 각 모델의 정확도 격차(학습 - 검증)를 시각화하고, 관찰된 정규화 효과를 설명하세요.

### 연습 5: 조기 종료(Early Stopping)와 모델 체크포인팅

`EarlyStopping` 클래스에 최적 모델 저장 기능을 추가하세요.

1. `EarlyStopping.__init__`에 `save_path` 매개변수를 추가하세요.
2. 검증 손실이 개선될 때마다 `torch.save(model.state_dict(), save_path)`로 모델을 저장하세요.
3. 학습이 끝난 후(조기 종료 또는 에포크 한계 도달) 최적 체크포인트를 불러와 테스트 셋에서 평가하세요.
4. 최종 체크포인트와 최적 체크포인트를 사용했을 때의 테스트 정확도를 비교하고, 최적 모델 저장이 가장 중요한 상황을 설명하세요.

---

## 다음 단계

[CNN 기초 (Convolutional Neural Networks)](./07_CNN_Basics.md)에서 합성곱 신경망을 학습합니다.
