[이전: 손실 함수](./24_Loss_Functions.md) | [다음: 정규화 레이어](./26_Normalization_Layers.md)

---

# 25. 옵티마이저(Optimizers)

## 학습 목표
- 경사 하강법의 변형들과 딥러닝의 최적화 환경 이해하기
- 고전적 옵티마이저(SGD, Momentum, Nesterov)와 적응적 방법(Adagrad, RMSprop, Adam, AdamW) 마스터하기
- 대규모 학습을 위한 현대적 옵티마이저(LAMB, Adafactor, Lion, 8-bit Adam) 탐구하기
- 학습률 스케줄러 구현하기(코사인 어닐링, OneCycleLR, 워밍업 전략)
- 실용적인 최적화 기법 적용하기(그래디언트 클리핑, 축적, 파라미터 그룹별 설정)
- 다양한 아키텍처와 과제에 적합한 옵티마이저와 하이퍼파라미터 선택하기

**난이도**: ⭐⭐⭐

---

## 1. 경사 하강법 기초

### 1.1 경사 하강법의 변형

**배치 경사 하강법(Batch Gradient Descent)**은 전체 학습 데이터셋을 사용하여 그래디언트를 계산합니다:

$$
\theta_{t+1} = \theta_t - \eta \nabla_\theta \mathcal{L}(\theta_t)
$$

여기서 $\mathcal{L}$은 모든 학습 예제에 대한 평균 손실입니다.

**확률적 경사 하강법(Stochastic Gradient Descent, SGD)**은 하나의 무작위 예제를 사용합니다:

$$
\theta_{t+1} = \theta_t - \eta \nabla_\theta \ell(x_i, y_i; \theta_t)
$$

**미니배치 경사 하강법(Mini-batch Gradient Descent)**은 실용적인 중간 지점입니다:

$$
\theta_{t+1} = \theta_t - \eta \nabla_\theta \frac{1}{|\mathcal{B}|} \sum_{(x,y) \in \mathcal{B}} \ell(x, y; \theta_t)
$$

```python
import torch
import torch.nn as nn
import torch.optim as optim

# Example: Different GD variants
def train_epoch_batch_gd(model, train_loader, criterion, optimizer):
    """Batch GD: accumulate gradients over entire epoch"""
    model.train()
    optimizer.zero_grad()
    total_loss = 0

    for X, y in train_loader:
        output = model(X)
        loss = criterion(output, y)
        loss.backward()  # Accumulate gradients
        total_loss += loss.item()

    optimizer.step()  # Single update after full pass
    return total_loss / len(train_loader)

def train_epoch_sgd(model, train_loader, criterion, optimizer):
    """SGD: update after each batch"""
    model.train()
    total_loss = 0

    for X, y in train_loader:
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()  # Update after each batch
        total_loss += loss.item()

    return total_loss / len(train_loader)

def train_epoch_mini_batch(model, train_loader, criterion, optimizer, accumulation_steps=4):
    """Mini-batch with gradient accumulation"""
    model.train()
    optimizer.zero_grad()
    total_loss = 0

    for i, (X, y) in enumerate(train_loader):
        output = model(X)
        loss = criterion(output, y) / accumulation_steps
        loss.backward()

        if (i + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item() * accumulation_steps

    return total_loss / len(train_loader)
```

### 1.2 최적화 환경

심층 신경망은 다음과 같은 특성을 가진 매우 비볼록(non-convex)한 손실 표면을 만듭니다:

- **국소 최솟값(Local minima)**: 그래디언트는 0이지만 전역 최솟값이 아닌 지점
- **안장점(Saddle points)**: 그래디언트는 0이지만 최솟값이 아닌 지점 (고차원에서 매우 흔함)
- **평탄 지역(Plateaus)**: 그래디언트가 0에 가까운 평평한 영역
- **협곡(Ravines)**: 일부 방향으로는 가파르고 다른 방향으로는 평평한 지역

```
Loss Surface Visualization:

           Global Minimum
                |
    ┌───────────▼───────────┐
    |                        |
    |  ╱╲      ╱╲      ╱╲   |  <- Local Minima
    | ╱  ╲    ╱  ╲    ╱  ╲  |
    |╱    ╲__╱    ╲__╱    ╲ |
    |      ▲       ▲        ▼|  <- Saddle Point
    |  Plateau  Local Min    |
    └────────────────────────┘

Gradient Behavior:
- At local minima: ∇L = 0, all eigenvalues > 0
- At saddle points: ∇L = 0, some eigenvalues < 0
- On plateaus: ∇L ≈ 0, very slow progress
```

**비볼록성에도 불구하고 최적화가 작동하는 이유:**

1. 고차원 공간에는 국소 최솟값보다 안장점이 기하급수적으로 많음
2. 국소 최솟값의 손실 값이 종종 전역 최솟값과 비슷함
3. 현대 옵티마이저(모멘텀 포함)는 안장점을 벗어날 수 있음
4. 과잉 매개변수화(overparameterization)가 많은 좋은 해를 만듦

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def visualize_loss_surface():
    """Visualize a toy non-convex loss surface"""
    x = np.linspace(-3, 3, 100)
    y = np.linspace(-3, 3, 100)
    X, Y = np.meshgrid(x, y)

    # Non-convex function with multiple minima
    Z = np.sin(X**2 + Y**2) / (X**2 + Y**2 + 1) + 0.1 * (X**2 + Y**2)

    fig = plt.figure(figsize=(12, 5))

    # 3D surface
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
    ax1.set_title('Loss Surface (3D)')
    ax1.set_xlabel('θ₁')
    ax1.set_ylabel('θ₂')
    ax1.set_zlabel('Loss')

    # Contour plot
    ax2 = fig.add_subplot(122)
    contour = ax2.contour(X, Y, Z, levels=20, cmap='viridis')
    ax2.clabel(contour, inline=True, fontsize=8)
    ax2.set_title('Loss Surface (Contour)')
    ax2.set_xlabel('θ₁')
    ax2.set_ylabel('θ₂')

    plt.tight_layout()
    plt.savefig('loss_surface.png', dpi=150)
    plt.close()

# Simulate optimization trajectory
def optimize_trajectory(optimizer_fn, steps=100):
    """Track optimizer path on loss surface"""
    trajectory = []
    theta = torch.tensor([2.5, 2.5], requires_grad=True)
    optimizer = optimizer_fn([theta])

    for _ in range(steps):
        optimizer.zero_grad()

        # Loss function (same as above)
        loss = torch.sin(theta[0]**2 + theta[1]**2) / (theta[0]**2 + theta[1]**2 + 1) + \
               0.1 * (theta[0]**2 + theta[1]**2)
        loss.backward()

        trajectory.append(theta.detach().clone().numpy())
        optimizer.step()

    return np.array(trajectory)

# Compare trajectories
sgd_traj = optimize_trajectory(lambda p: optim.SGD(p, lr=0.1))
momentum_traj = optimize_trajectory(lambda p: optim.SGD(p, lr=0.1, momentum=0.9))
adam_traj = optimize_trajectory(lambda p: optim.Adam(p, lr=0.1))
```

---

## 2. 고전적 옵티마이저

### 2.1 확률적 경사 하강법(SGD)

기본 SGD 업데이트 규칙:

$$
\theta_{t+1} = \theta_t - \eta \nabla_\theta \mathcal{L}(\theta_t)
$$

**순수 SGD의 문제점:**
- 협곡에서 진동 (일부 방향으로 가파름)
- 평탄한 지역에서 느린 수렴
- 안장점을 벗어나기 어려움
- 모든 파라미터에 동일한 학습률 사용

### 2.2 모멘텀을 가진 SGD

모멘텀(Momentum)은 일관된 방향으로 가속하고 진동을 감쇠합니다:

$$
\begin{align}
v_t &= \beta v_{t-1} + \nabla_\theta \mathcal{L}(\theta_t) \\
\theta_{t+1} &= \theta_t - \eta v_t
\end{align}
$$

여기서 $\beta \in [0, 1)$는 모멘텀 계수입니다 (일반적으로 0.9).

**물리적 비유**: 언덕을 굴러내려가는 공이 속도를 축적합니다.

```
Momentum Effect:

Without Momentum:        With Momentum:
     ↓                       ↓
   ↙ ↘                    ↙  ↘
  ↙   ↘                 ↙     ↘
 ↙  ↓  ↘              ↙    ↓   ↘
↙   ↓   ↘           ↙      ↓    ↘  <- Smoother, faster
   zigzag            straighter path
```

**수동 구현:**

```python
class SGDMomentum:
    """Manual implementation of SGD with momentum"""
    def __init__(self, params, lr=0.01, momentum=0.9, weight_decay=0):
        self.params = list(params)
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.velocities = [torch.zeros_like(p) for p in self.params]

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()

    def step(self):
        with torch.no_grad():
            for i, p in enumerate(self.params):
                if p.grad is None:
                    continue

                # Gradient with weight decay
                grad = p.grad
                if self.weight_decay != 0:
                    grad = grad.add(p, alpha=self.weight_decay)

                # Momentum update
                self.velocities[i] = self.momentum * self.velocities[i] + grad
                p.add_(self.velocities[i], alpha=-self.lr)

# Test implementation
model = nn.Linear(10, 1)
criterion = nn.MSELoss()

# Custom optimizer
optimizer = SGDMomentum(model.parameters(), lr=0.01, momentum=0.9)

# PyTorch equivalent
optimizer_torch = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# Both should produce identical results
X = torch.randn(32, 10)
y = torch.randn(32, 1)

# Custom
optimizer.zero_grad()
loss = criterion(model(X), y)
loss.backward()
optimizer.step()

# Check velocity is being maintained
print(f"Velocity norm: {torch.norm(optimizer.velocities[0]):.6f}")
```

### 2.3 네스테로프 가속 경사법(NAG)

네스테로프 모멘텀(Nesterov momentum)은 그래디언트를 계산하기 전에 "미리 내다봅니다":

$$
\begin{align}
v_t &= \beta v_{t-1} + \nabla_\theta \mathcal{L}(\theta_t - \beta v_{t-1}) \\
\theta_{t+1} &= \theta_t - \eta v_t
\end{align}
$$

**직관**: 현재 위치가 아닌 "미리 보는" 위치에서 그래디언트를 계산합니다.

```python
# PyTorch SGD with Nesterov momentum
optimizer_sgd = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
optimizer_nesterov = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, nesterov=True)

# Comparison on a simple problem
def compare_sgd_variants():
    """Compare vanilla SGD, momentum, and Nesterov"""
    torch.manual_seed(42)

    # Simple 2D problem
    target = torch.tensor([1.0, 1.0])

    results = {}
    for name, opt_fn in [
        ('SGD', lambda p: optim.SGD(p, lr=0.1)),
        ('Momentum', lambda p: optim.SGD(p, lr=0.1, momentum=0.9)),
        ('Nesterov', lambda p: optim.SGD(p, lr=0.1, momentum=0.9, nesterov=True))
    ]:
        theta = torch.tensor([0.0, 0.0], requires_grad=True)
        optimizer = opt_fn([theta])
        losses = []

        for _ in range(50):
            optimizer.zero_grad()
            loss = torch.sum((theta - target)**2)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        results[name] = losses

    # Plot convergence
    plt.figure(figsize=(10, 6))
    for name, losses in results.items():
        plt.plot(losses, label=name, linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('SGD Variants Convergence Comparison')
    plt.legend()
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.savefig('sgd_comparison.png', dpi=150)
    plt.close()

compare_sgd_variants()
```

---

## 3. 적응적 학습률 방법

### 3.1 Adagrad

Adagrad는 과거 그래디언트에 기반하여 파라미터별로 학습률을 조정합니다:

$$
\begin{align}
g_t &= \nabla_\theta \mathcal{L}(\theta_t) \\
G_t &= G_{t-1} + g_t \odot g_t \\
\theta_{t+1} &= \theta_t - \frac{\eta}{\sqrt{G_t + \epsilon}} \odot g_t
\end{align}
$$

여기서 $G_t$는 제곱된 그래디언트를 요소별로 축적합니다.

**장점:**
- 자동 학습률 어닐링
- 드문 특성에 대해 더 큰 업데이트
- 희소 데이터에 잘 작동

**단점:**
- 학습률이 단조 감소함
- 너무 일찍 학습을 멈출 수 있음

```python
# PyTorch Adagrad
optimizer = optim.Adagrad(model.parameters(), lr=0.01, eps=1e-8)

# Adagrad is useful for NLP with sparse features
import torch.nn.functional as F

class WordEmbeddingModel(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.fc = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        embedded = self.embedding(x)
        return self.fc(embedded.mean(dim=1))

model = WordEmbeddingModel(vocab_size=10000, embed_dim=128)
optimizer = optim.Adagrad(model.parameters(), lr=0.01)  # Good for sparse embeddings
```

### 3.2 RMSprop

RMSprop은 지수 이동 평균을 사용하여 Adagrad의 공격적인 학습률 감소를 수정합니다:

$$
\begin{align}
g_t &= \nabla_\theta \mathcal{L}(\theta_t) \\
E[g^2]_t &= \beta E[g^2]_{t-1} + (1-\beta) g_t \odot g_t \\
\theta_{t+1} &= \theta_t - \frac{\eta}{\sqrt{E[g^2]_t + \epsilon}} \odot g_t
\end{align}
$$

여기서 $\beta \in [0, 1)$ (일반적으로 0.9).

```python
# PyTorch RMSprop
optimizer = optim.RMSprop(model.parameters(), lr=0.001, alpha=0.9, eps=1e-8)

# RMSprop is popular for RNNs
class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out[:, -1, :])

rnn_model = SimpleRNN(input_size=10, hidden_size=64, output_size=2)
optimizer = optim.RMSprop(rnn_model.parameters(), lr=0.001)  # Good for RNNs
```

### 3.3 Adam (Adaptive Moment Estimation)

Adam은 모멘텀과 RMSprop을 결합하여 1차 및 2차 모멘트 추정치를 모두 유지합니다:

$$
\begin{align}
g_t &= \nabla_\theta \mathcal{L}(\theta_t) \\
m_t &= \beta_1 m_{t-1} + (1-\beta_1) g_t \quad \text{(1차 모멘트)} \\
v_t &= \beta_2 v_{t-1} + (1-\beta_2) g_t \odot g_t \quad \text{(2차 모멘트)} \\
\hat{m}_t &= \frac{m_t}{1-\beta_1^t} \quad \text{(편향 보정)} \\
\hat{v}_t &= \frac{v_t}{1-\beta_2^t} \quad \text{(편향 보정)} \\
\theta_{t+1} &= \theta_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t
\end{align}
$$

**기본 하이퍼파라미터** (실제로 잘 작동함):
- $\beta_1 = 0.9$ (모멘텀)
- $\beta_2 = 0.999$ (RMSprop 감쇠)
- $\epsilon = 10^{-8}$

**수동 구현:**

```python
class AdamOptimizer:
    """Manual implementation of Adam optimizer"""
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        self.params = list(params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay

        # Initialize moment estimates
        self.m = [torch.zeros_like(p) for p in self.params]
        self.v = [torch.zeros_like(p) for p in self.params]
        self.t = 0

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()

    def step(self):
        self.t += 1

        with torch.no_grad():
            for i, p in enumerate(self.params):
                if p.grad is None:
                    continue

                grad = p.grad

                # Weight decay (L2 regularization)
                if self.weight_decay != 0:
                    grad = grad.add(p, alpha=self.weight_decay)

                # Update biased first moment estimate
                self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad

                # Update biased second raw moment estimate
                self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * grad.pow(2)

                # Compute bias-corrected first moment estimate
                m_hat = self.m[i] / (1 - self.beta1 ** self.t)

                # Compute bias-corrected second raw moment estimate
                v_hat = self.v[i] / (1 - self.beta2 ** self.t)

                # Update parameters
                p.add_(m_hat / (v_hat.sqrt() + self.eps), alpha=-self.lr)

# Verify against PyTorch implementation
def test_adam_implementation():
    torch.manual_seed(42)

    # Create two identical models
    model1 = nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 1))
    model2 = nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 1))
    model2.load_state_dict(model1.state_dict())

    # Use both optimizers
    opt_custom = AdamOptimizer(model1.parameters(), lr=0.001)
    opt_torch = optim.Adam(model2.parameters(), lr=0.001)

    criterion = nn.MSELoss()

    # Run a few steps
    for _ in range(10):
        X = torch.randn(32, 10)
        y = torch.randn(32, 1)

        # Custom Adam
        opt_custom.zero_grad()
        loss1 = criterion(model1(X), y)
        loss1.backward()
        opt_custom.step()

        # PyTorch Adam
        opt_torch.zero_grad()
        loss2 = criterion(model2(X), y)
        loss2.backward()
        opt_torch.step()

    # Check parameters are close (may have small numerical differences)
    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        print(f"Param diff: {torch.max(torch.abs(p1 - p2)).item():.2e}")

test_adam_implementation()

# PyTorch Adam
optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-8)
```

### 3.4 AdamW (분리된 가중치 감쇠를 가진 Adam)

**핵심 통찰**: L2 정규화와 가중치 감쇠는 SGD에서는 동등하지만 Adam에서는 그렇지 않습니다!

**L2 정규화** (전통적):
$$
\mathcal{L}_{\text{total}} = \mathcal{L} + \frac{\lambda}{2} ||\theta||^2
$$

**가중치 감쇠** (분리):
$$
\theta_{t+1} = (1 - \eta \lambda) \theta_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t
$$

AdamW는 그래디언트를 통해서가 아니라 파라미터에 직접 가중치 감쇠를 적용합니다:

```python
# Adam with L2 regularization (traditional)
optimizer_adam = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)

# AdamW with decoupled weight decay (better!)
optimizer_adamw = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

# Visualize the difference
def compare_adam_adamw():
    """Show difference between Adam and AdamW"""
    torch.manual_seed(42)

    # Simple overparameterized model
    model_adam = nn.Sequential(nn.Linear(10, 100), nn.ReLU(), nn.Linear(100, 1))
    model_adamw = nn.Sequential(nn.Linear(10, 100), nn.ReLU(), nn.Linear(100, 1))
    model_adamw.load_state_dict(model_adam.state_dict())

    opt_adam = optim.Adam(model_adam.parameters(), lr=0.001, weight_decay=0.01)
    opt_adamw = optim.AdamW(model_adamw.parameters(), lr=0.001, weight_decay=0.01)

    criterion = nn.MSELoss()

    # Track weight norms
    adam_norms = []
    adamw_norms = []

    for _ in range(100):
        X = torch.randn(32, 10)
        y = torch.randn(32, 1)

        # Adam
        opt_adam.zero_grad()
        loss = criterion(model_adam(X), y)
        loss.backward()
        opt_adam.step()

        # AdamW
        opt_adamw.zero_grad()
        loss = criterion(model_adamw(X), y)
        loss.backward()
        opt_adamw.step()

        # Record weight norms
        adam_norms.append(torch.norm(model_adam[0].weight).item())
        adamw_norms.append(torch.norm(model_adamw[0].weight).item())

    plt.figure(figsize=(10, 6))
    plt.plot(adam_norms, label='Adam (L2 reg)', linewidth=2)
    plt.plot(adamw_norms, label='AdamW (decoupled)', linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('Weight Norm')
    plt.title('Adam vs AdamW: Weight Regularization')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('adam_adamw_comparison.png', dpi=150)
    plt.close()

compare_adam_adamw()
```

**AdamW를 사용할 때:**
- 가중치 감쇠를 사용할 때는 거의 항상 Adam보다 AdamW를 선호
- 트랜스포머 및 대규모 모델에 특히 중요
- BERT, GPT 및 대부분의 현대 NLP 모델의 기본 옵티마이저

---

## 4. 현대적 옵티마이저

### 4.1 LAMB (Layer-wise Adaptive Moments optimizer for Batch training)

LAMB는 레이어별 적응을 사용하여 대규모 배치 학습을 가능하게 합니다:

$$
r_t = \frac{||\theta_t||}{||\hat{m}_t / (\sqrt{\hat{v}_t} + \epsilon)||} \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
$$

$$
\theta_{t+1} = \theta_t - \eta r_t
$$

**사용 사례**: 배치 크기 32K로 BERT 학습 (Adam은 256).

```python
# LAMB is not in PyTorch by default, but available in apex or standalone
# pip install pytorch-lamb

try:
    from pytorch_lamb import Lamb

    # LAMB optimizer for large-batch training
    optimizer = Lamb(
        model.parameters(),
        lr=0.001,
        betas=(0.9, 0.999),
        eps=1e-6,
        weight_decay=0.01,
        bias_correction=True
    )

    # Typical use: scale learning rate with batch size
    # LR = base_lr * sqrt(batch_size / base_batch_size)
    base_lr = 0.001
    base_batch_size = 256
    large_batch_size = 8192

    scaled_lr = base_lr * (large_batch_size / base_batch_size) ** 0.5
    optimizer = Lamb(model.parameters(), lr=scaled_lr, weight_decay=0.01)

except ImportError:
    print("LAMB not installed. Install with: pip install pytorch-lamb")

# LARS (Layer-wise Adaptive Rate Scaling) - similar idea for CNNs
# Used for ImageNet training with large batches
```

### 4.2 Adafactor

Adafactor는 전체 2차 모멘트를 저장하지 않음으로써 메모리를 줄입니다:

**핵심 아이디어**: 2차 모멘트 행렬을 인수분해하여 메모리를 O(nm)에서 O(n+m)로 줄입니다.

```python
# PyTorch doesn't have built-in Adafactor, but transformers library does
try:
    from transformers import Adafactor

    # Adafactor: memory-efficient for large models
    optimizer = Adafactor(
        model.parameters(),
        lr=1e-3,
        eps=(1e-30, 1e-3),
        clip_threshold=1.0,
        decay_rate=-0.8,
        beta1=None,
        weight_decay=0.0,
        relative_step=False,
        scale_parameter=False,
        warmup_init=False
    )

    # With relative step (learning rate is automatically computed)
    optimizer = Adafactor(
        model.parameters(),
        scale_parameter=True,
        relative_step=True,
        warmup_init=True,
        lr=None  # LR is computed automatically
    )

except ImportError:
    print("Adafactor not available. Install with: pip install transformers")
```

**Adafactor를 사용할 때:**
- 매우 큰 모델 학습 (수십억 개의 파라미터)
- 제한된 GPU 메모리
- T5, UL2 모델은 Adafactor로 학습됨

### 4.3 Lion (Evolved Sign Momentum)

Lion은 그래디언트의 부호만 사용하여 메모리와 계산을 줄입니다:

$$
\begin{align}
c_t &= \beta_1 m_{t-1} + (1-\beta_1) g_t \\
\theta_{t+1} &= \theta_t - \eta \cdot \text{sign}(c_t) \\
m_t &= \beta_2 m_{t-1} + (1-\beta_2) g_t
\end{align}
$$

**장점:**
- 2배 메모리 효율적 (모멘텀만 저장, 2차 모멘트 없음)
- 더 빠른 계산 (제곱근 연산 없음)
- Adam과 경쟁하거나 더 나은 성능

```python
# Lion optimizer (pip install lion-pytorch)
try:
    from lion_pytorch import Lion

    optimizer = Lion(
        model.parameters(),
        lr=1e-4,  # Use ~3-10x smaller LR than Adam
        betas=(0.9, 0.99),
        weight_decay=0.01
    )

    # Typical usage: train vision transformer
    # Note: Lion needs smaller learning rate than Adam

except ImportError:
    # Manual implementation (simplified)
    class Lion(torch.optim.Optimizer):
        def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
            defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
            super().__init__(params, defaults)

        def step(self):
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is None:
                        continue

                    grad = p.grad
                    state = self.state[p]

                    # Initialize momentum
                    if len(state) == 0:
                        state['exp_avg'] = torch.zeros_like(p)

                    exp_avg = state['exp_avg']
                    beta1, beta2 = group['betas']

                    # Weight decay
                    if group['weight_decay'] != 0:
                        p.mul_(1 - group['lr'] * group['weight_decay'])

                    # Update (sign of interpolated gradient)
                    update = exp_avg.mul(beta1).add(grad, alpha=1 - beta1).sign_()
                    p.add_(update, alpha=-group['lr'])

                    # Update momentum
                    exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)

    print("Using simplified Lion implementation")
```

### 4.4 8-bit Adam (bitsandbytes)

8-bit Adam은 옵티마이저 상태를 양자화하여 메모리를 줄입니다:

```python
# 8-bit Adam from bitsandbytes
# pip install bitsandbytes

try:
    import bitsandbytes as bnb

    # 8-bit Adam: same performance, 75% less memory
    optimizer = bnb.optim.Adam8bit(
        model.parameters(),
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01
    )

    # Also available: AdamW8bit, Lion8bit, etc.
    optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=1e-3)

    print("Using 8-bit Adam - significant memory savings!")

except ImportError:
    print("bitsandbytes not installed. Install with: pip install bitsandbytes")
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
```

**8-bit 옵티마이저를 사용할 때:**
- 제한된 GPU 메모리로 대규모 모델 학습
- LLM 파인튜닝 (예: 소비자 GPU에서 LLaMA-7B)
- 최소한의 성능 영향 (~0.1% 차이)

### 4.5 Sophia (Second-order Clipped Stochastic Optimization)

Sophia는 더 나은 곡률 적응을 위해 헤시안 대각 정보를 사용합니다:

$$
\theta_{t+1} = \theta_t - \eta \cdot \frac{m_t}{\max\{h_t, \epsilon\}}
$$

여기서 $h_t$는 헤시안 대각선의 추정치입니다.

**사용 사례**: Adam보다 2배 빠르게 언어 모델 학습.

```python
# Sophia is experimental and requires custom implementation
# Available at: https://github.com/Liuhong99/Sophia

# Typical usage for LLM training:
# optimizer = SophiaG(model.parameters(), lr=2e-4, rho=0.04, weight_decay=0.1)
```

### 4.6 옵티마이저 비교 표

| 옵티마이저 | 메모리 | 속도 | 사용 사례 | 일반적 LR |
|-----------|--------|-------|----------|------------|
| SGD+Momentum | 낮음 | 빠름 | CNN, ResNet | 0.1 - 0.01 |
| Adam | 높음 | 중간 | 일반, 트랜스포머 | 1e-3 - 1e-4 |
| AdamW | 높음 | 중간 | 트랜스포머, 파인튜닝 | 1e-3 - 1e-4 |
| LAMB | 높음 | 중간 | 대규모 배치 학습 | 1e-2 - 1e-3 |
| Adafactor | 중간 | 느림 | 매우 큰 모델 | 1e-3 - 1e-2 |
| Lion | 중간 | 빠름 | Vision Transformer | 1e-4 - 1e-5 |
| 8-bit Adam | 낮음 | 중간 | 제한된 GPU 메모리 | 1e-3 - 1e-4 |
| Sophia | 높음 | 중간 | LLM 사전학습 | 2e-4 - 5e-4 |

---

## 5. 학습률 스케줄러

### 5.1 스텝 기반 스케줄러

**StepLR**: step_size 에폭마다 LR을 gamma로 감소:

```python
# StepLR: multiply LR by 0.1 every 30 epochs
optimizer = optim.SGD(model.parameters(), lr=0.1)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

# Epoch 0-29: lr=0.1, Epoch 30-59: lr=0.01, Epoch 60+: lr=0.001
for epoch in range(100):
    train_one_epoch(model, optimizer)
    scheduler.step()  # Update LR at end of epoch

# MultiStepLR: decay at specific milestones
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 80], gamma=0.1)
# Epoch 0-29: lr=0.1, Epoch 30-79: lr=0.01, Epoch 80+: lr=0.001
```

### 5.2 지수 감소

**ExponentialLR**: 매 에폭마다 LR을 gamma로 곱함:

$$
\eta_t = \eta_0 \cdot \gamma^t
$$

```python
# Exponential decay
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

# Each epoch: lr *= 0.95
for epoch in range(100):
    train_one_epoch(model, optimizer)
    scheduler.step()
```

### 5.3 코사인 어닐링

**CosineAnnealingLR**: 코사인 곡선을 따르는 부드러운 감소:

$$
\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})(1 + \cos(\frac{t}{T}\pi))
$$

```python
# Cosine annealing
optimizer = optim.AdamW(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=100,  # Total number of epochs
    eta_min=1e-6  # Minimum learning rate
)

# LR smoothly decays from 0.001 to 1e-6 over 100 epochs

# Cosine annealing with warm restarts
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer,
    T_0=10,  # First restart after 10 epochs
    T_mult=2,  # Double period after each restart
    eta_min=1e-6
)

# LR schedule: 10 epochs, restart, 20 epochs, restart, 40 epochs, ...
```

**코사인 어닐링 시각화:**

```python
def visualize_cosine_schedule():
    """Visualize cosine annealing schedule"""
    import numpy as np

    epochs = 100
    lr_max = 0.001
    lr_min = 1e-6

    # Standard cosine
    lrs_cosine = [
        lr_min + 0.5 * (lr_max - lr_min) * (1 + np.cos(np.pi * t / epochs))
        for t in range(epochs)
    ]

    # Cosine with warm restarts (T_0=10, T_mult=2)
    lrs_restart = []
    t = 0
    T_cur = 10
    for epoch in range(epochs):
        lr = lr_min + 0.5 * (lr_max - lr_min) * (1 + np.cos(np.pi * t / T_cur))
        lrs_restart.append(lr)
        t += 1
        if t >= T_cur:  # Restart
            t = 0
            T_cur *= 2

    plt.figure(figsize=(12, 6))
    plt.plot(lrs_cosine, label='Cosine Annealing', linewidth=2)
    plt.plot(lrs_restart, label='Cosine with Warm Restarts', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Cosine Annealing Schedules')
    plt.legend()
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.savefig('cosine_schedule.png', dpi=150)
    plt.close()

visualize_cosine_schedule()
```

### 5.4 OneCycleLR (Super-convergence)

OneCycleLR은 워밍업, 피크, 감소를 포함한 단일 사이클을 사용합니다:

```
LR Schedule:
   max_lr ────────────╱╲
                    ╱    ╲
                  ╱        ╲
   div_factor  ╱            ╲────── final_div
              ↑               ↑
           warmup          annealing
```

```python
# OneCycleLR: state-of-the-art for many tasks
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=0.1,  # Peak learning rate
    total_steps=None,  # Will be computed from epochs and steps_per_epoch
    epochs=90,
    steps_per_epoch=len(train_loader),
    pct_start=0.3,  # Warmup: 30% of training
    anneal_strategy='cos',  # 'cos' or 'linear'
    div_factor=25.0,  # Initial LR = max_lr / div_factor
    final_div_factor=10000.0  # Final LR = max_lr / final_div_factor
)

# Must call scheduler.step() after EVERY batch
for epoch in range(90):
    for batch in train_loader:
        optimizer.zero_grad()
        loss = model(batch)
        loss.backward()
        optimizer.step()
        scheduler.step()  # Update after each batch!

# Visualize OneCycleLR
def visualize_onecycle():
    """Visualize OneCycleLR schedule"""
    model = nn.Linear(10, 1)
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.1,
        total_steps=1000,
        pct_start=0.3,
        anneal_strategy='cos'
    )

    lrs = []
    for _ in range(1000):
        lrs.append(optimizer.param_groups[0]['lr'])
        optimizer.step()
        scheduler.step()

    plt.figure(figsize=(12, 6))
    plt.plot(lrs, linewidth=2)
    plt.xlabel('Step')
    plt.ylabel('Learning Rate')
    plt.title('OneCycleLR Schedule')
    plt.axvline(300, color='r', linestyle='--', alpha=0.5, label='End of warmup')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('onecycle_schedule.png', dpi=150)
    plt.close()

visualize_onecycle()
```

### 5.5 선형 워밍업 + 코사인 감소 (트랜스포머 표준)

대부분의 트랜스포머 모델은 워밍업 후 코사인 감소를 사용합니다:

```python
import math

class WarmupCosineSchedule(torch.optim.lr_scheduler.LambdaLR):
    """Linear warmup + cosine decay (BERT/GPT standard)"""
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr_ratio=0.0):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr_ratio = min_lr_ratio
        super().__init__(optimizer, self.lr_lambda)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            # Linear warmup
            return float(step) / float(max(1, self.warmup_steps))

        # Cosine decay
        progress = float(step - self.warmup_steps) / float(max(1, self.total_steps - self.warmup_steps))
        return self.min_lr_ratio + (1 - self.min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))

# Usage for transformer training
optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)
total_steps = len(train_loader) * num_epochs
warmup_steps = int(0.1 * total_steps)  # 10% warmup

scheduler = WarmupCosineSchedule(
    optimizer,
    warmup_steps=warmup_steps,
    total_steps=total_steps,
    min_lr_ratio=0.1  # Decay to 10% of peak LR
)

# Training loop
for epoch in range(num_epochs):
    for batch in train_loader:
        optimizer.zero_grad()
        loss = model(batch)
        loss.backward()
        optimizer.step()
        scheduler.step()  # Update after each batch

# Visualize
def visualize_warmup_cosine():
    """Visualize warmup + cosine schedule"""
    model = nn.Linear(10, 1)
    optimizer = optim.AdamW(model.parameters(), lr=5e-4)

    total_steps = 10000
    warmup_steps = 1000
    scheduler = WarmupCosineSchedule(optimizer, warmup_steps, total_steps, min_lr_ratio=0.1)

    lrs = []
    for _ in range(total_steps):
        lrs.append(optimizer.param_groups[0]['lr'])
        optimizer.step()
        scheduler.step()

    plt.figure(figsize=(12, 6))
    plt.plot(lrs, linewidth=2)
    plt.xlabel('Step')
    plt.ylabel('Learning Rate')
    plt.title('Linear Warmup + Cosine Decay (Transformer Standard)')
    plt.axvline(warmup_steps, color='r', linestyle='--', alpha=0.5, label='End of warmup')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('warmup_cosine_schedule.png', dpi=150)
    plt.close()

visualize_warmup_cosine()
```

### 5.6 ReduceLROnPlateau

검증 메트릭이 정체될 때 LR 감소:

```python
# ReduceLROnPlateau: data-driven LR reduction
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',  # 'min' for loss, 'max' for accuracy
    factor=0.5,  # Multiply LR by 0.5
    patience=10,  # Wait 10 epochs before reducing
    verbose=True,
    min_lr=1e-6
)

for epoch in range(100):
    train_loss = train_one_epoch(model, optimizer)
    val_loss = validate(model)

    # Step based on validation loss
    scheduler.step(val_loss)

    print(f"Epoch {epoch}: LR = {optimizer.param_groups[0]['lr']:.6f}")
```

### 5.7 커스텀 스케줄러

LambdaLR을 사용한 커스텀 스케줄 구현:

```python
# Custom schedule: polynomial decay with warmup
def polynomial_decay_with_warmup(step, warmup_steps, total_steps, power=1.0):
    if step < warmup_steps:
        return float(step) / float(max(1, warmup_steps))
    else:
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return (1 - progress) ** power

optimizer = optim.AdamW(model.parameters(), lr=5e-4)
scheduler = optim.lr_scheduler.LambdaLR(
    optimizer,
    lr_lambda=lambda step: polynomial_decay_with_warmup(step, 1000, 10000, power=2.0)
)

# Or chain multiple schedulers
scheduler1 = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=1000)
scheduler2 = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
scheduler = optim.lr_scheduler.ChainedScheduler([scheduler1, scheduler2])
```

### 5.8 스케줄러 시각화 모음

```python
def compare_all_schedulers():
    """Compare all scheduler types"""
    model = nn.Linear(10, 1)
    total_steps = 1000

    schedulers = {
        'StepLR': lambda opt: optim.lr_scheduler.StepLR(opt, step_size=300, gamma=0.1),
        'ExponentialLR': lambda opt: optim.lr_scheduler.ExponentialLR(opt, gamma=0.995),
        'CosineAnnealing': lambda opt: optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps),
        'OneCycleLR': lambda opt: optim.lr_scheduler.OneCycleLR(opt, max_lr=0.1, total_steps=total_steps),
        'Warmup+Cosine': lambda opt: WarmupCosineSchedule(opt, warmup_steps=100, total_steps=total_steps),
    }

    plt.figure(figsize=(14, 8))

    for name, sched_fn in schedulers.items():
        optimizer = optim.SGD(model.parameters(), lr=0.1)
        scheduler = sched_fn(optimizer)

        lrs = []
        for _ in range(total_steps):
            lrs.append(optimizer.param_groups[0]['lr'])
            optimizer.step()
            scheduler.step()

        plt.plot(lrs, label=name, linewidth=2)

    plt.xlabel('Step')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Scheduler Comparison')
    plt.legend()
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.savefig('scheduler_comparison.png', dpi=150)
    plt.close()

compare_all_schedulers()
```

---

## 6. 실용적 기법

### 6.1 학습률 탐색 (LR Range Test)

학습률을 점진적으로 증가시켜 최적 학습률 찾기:

```python
class LRFinder:
    """Learning rate range test (Leslie Smith)"""
    def __init__(self, model, optimizer, criterion, device='cuda'):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device

    def range_test(self, train_loader, start_lr=1e-7, end_lr=10, num_iter=100, smooth_f=0.05):
        """Run LR range test"""
        # Save initial state
        model_state = self.model.state_dict()
        optim_state = self.optimizer.state_dict()

        # Update LR
        lrs = []
        losses = []
        best_loss = float('inf')

        lr_mult = (end_lr / start_lr) ** (1 / num_iter)
        lr = start_lr

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        self.model.train()
        iterator = iter(train_loader)

        for i in range(num_iter):
            try:
                X, y = next(iterator)
            except StopIteration:
                iterator = iter(train_loader)
                X, y = next(iterator)

            X, y = X.to(self.device), y.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            output = self.model(X)
            loss = self.criterion(output, y)

            # Smooth loss
            if i == 0:
                avg_loss = loss.item()
            else:
                avg_loss = smooth_f * loss.item() + (1 - smooth_f) * avg_loss

            # Stop if loss is exploding
            if avg_loss > 4 * best_loss or torch.isnan(loss):
                break

            if avg_loss < best_loss:
                best_loss = avg_loss

            # Record
            lrs.append(lr)
            losses.append(avg_loss)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Update LR
            lr *= lr_mult
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

        # Restore initial state
        self.model.load_state_dict(model_state)
        self.optimizer.load_state_dict(optim_state)

        return lrs, losses

    def plot(self, lrs, losses, skip_start=10, skip_end=5):
        """Plot LR finder results"""
        if skip_end > 0:
            lrs = lrs[skip_start:-skip_end]
            losses = losses[skip_start:-skip_end]
        else:
            lrs = lrs[skip_start:]
            losses = losses[skip_start:]

        plt.figure(figsize=(10, 6))
        plt.plot(lrs, losses, linewidth=2)
        plt.xscale('log')
        plt.xlabel('Learning Rate')
        plt.ylabel('Loss')
        plt.title('LR Range Test')
        plt.grid(True, alpha=0.3)

        # Find steepest descent
        min_grad_idx = np.gradient(np.array(losses)).argmin()
        suggested_lr = lrs[min_grad_idx]
        plt.axvline(suggested_lr, color='r', linestyle='--',
                    label=f'Suggested LR: {suggested_lr:.2e}')
        plt.legend()
        plt.savefig('lr_finder.png', dpi=150)
        plt.close()

        return suggested_lr

# Usage
model = nn.Sequential(nn.Linear(10, 50), nn.ReLU(), nn.Linear(50, 1))
optimizer = optim.SGD(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

lr_finder = LRFinder(model, optimizer, criterion)
lrs, losses = lr_finder.range_test(train_loader, start_lr=1e-7, end_lr=1)
suggested_lr = lr_finder.plot(lrs, losses)

print(f"Suggested learning rate: {suggested_lr:.2e}")
```

### 6.2 그래디언트 클리핑

클리핑으로 그래디언트 폭발 방지:

```python
# Gradient clipping by norm (most common)
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(num_epochs):
    for X, y in train_loader:
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output, y)
        loss.backward()

        # Clip gradients to max norm of 1.0
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

# Gradient clipping by value
torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=0.5)

# Monitor gradient norms
def get_grad_norm(model):
    """Compute total gradient norm"""
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5

# Training with gradient monitoring
grad_norms = []
for epoch in range(num_epochs):
    for X, y in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(X), y)
        loss.backward()

        # Monitor before clipping
        grad_norm_before = get_grad_norm(model)

        # Clip
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        grad_norm_after = get_grad_norm(model)
        grad_norms.append((grad_norm_before, grad_norm_after))

        optimizer.step()

# Plot gradient norms
before, after = zip(*grad_norms)
plt.figure(figsize=(12, 6))
plt.plot(before, label='Before clipping', alpha=0.7)
plt.plot(after, label='After clipping', alpha=0.7)
plt.xlabel('Step')
plt.ylabel('Gradient Norm')
plt.title('Gradient Clipping Effect')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('gradient_clipping.png', dpi=150)
plt.close()
```

### 6.3 그래디언트 축적

제한된 메모리로 더 큰 배치 크기 시뮬레이션:

```python
# Gradient accumulation
accumulation_steps = 4  # Effective batch size = batch_size * accumulation_steps

optimizer.zero_grad()

for i, (X, y) in enumerate(train_loader):
    # Forward pass
    output = model(X)
    loss = criterion(output, y)

    # Normalize loss to account for accumulation
    loss = loss / accumulation_steps
    loss.backward()

    # Update only after accumulating gradients
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()

# Complete implementation with proper handling
def train_with_accumulation(model, train_loader, optimizer, criterion,
                            accumulation_steps=4, clip_norm=1.0):
    model.train()
    total_loss = 0
    optimizer.zero_grad()

    for i, (X, y) in enumerate(train_loader):
        X, y = X.to(device), y.to(device)

        # Forward pass
        output = model(X)
        loss = criterion(output, y) / accumulation_steps
        loss.backward()

        # Update parameters
        if (i + 1) % accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item() * accumulation_steps

    # Handle remaining gradients
    if (i + 1) % accumulation_steps != 0:
        optimizer.step()
        optimizer.zero_grad()

    return total_loss / len(train_loader)
```

### 6.4 혼합 정밀도 학습

더 빠른 학습을 위해 옵티마이저와 결합:

```python
from torch.cuda.amp import GradScaler, autocast

# Mixed precision training setup
model = model.to('cuda')
optimizer = optim.AdamW(model.parameters(), lr=1e-3)
scaler = GradScaler()

for epoch in range(num_epochs):
    for X, y in train_loader:
        X, y = X.to('cuda'), y.to('cuda')

        optimizer.zero_grad()

        # Forward pass in fp16
        with autocast():
            output = model(X)
            loss = criterion(output, y)

        # Backward pass with scaling
        scaler.scale(loss).backward()

        # Gradient clipping (unscale first!)
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Update with scaled gradients
        scaler.step(optimizer)
        scaler.update()

# Mixed precision with gradient accumulation
scaler = GradScaler()
accumulation_steps = 4

for i, (X, y) in enumerate(train_loader):
    with autocast():
        output = model(X)
        loss = criterion(output, y) / accumulation_steps

    scaler.scale(loss).backward()

    if (i + 1) % accumulation_steps == 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
```

### 6.5 파라미터 그룹별 학습률

모델의 다른 부분에 다른 LR:

```python
# Example: Fine-tuning with different LRs for backbone and head
class TransferModel(nn.Module):
    def __init__(self, backbone, num_classes):
        super().__init__()
        self.backbone = backbone  # Pretrained
        self.head = nn.Linear(512, num_classes)  # New

    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)

# Different learning rates for different parts
model = TransferModel(pretrained_backbone, num_classes=10)

optimizer = optim.AdamW([
    {'params': model.backbone.parameters(), 'lr': 1e-5},  # Low LR for backbone
    {'params': model.head.parameters(), 'lr': 1e-3}  # High LR for head
], weight_decay=0.01)

# Can also use different weight decay
optimizer = optim.AdamW([
    {'params': model.backbone.parameters(), 'lr': 1e-5, 'weight_decay': 0.01},
    {'params': model.head.parameters(), 'lr': 1e-3, 'weight_decay': 0.0}
])

# Advanced: layer-wise learning rate decay (LLRD)
def get_layer_wise_lr_groups(model, base_lr=1e-3, decay_rate=0.9):
    """Apply lower LR to earlier layers"""
    num_layers = len(list(model.named_parameters()))

    param_groups = []
    for i, (name, param) in enumerate(model.named_parameters()):
        # Earlier layers get lower LR
        layer_lr = base_lr * (decay_rate ** (num_layers - i - 1))
        param_groups.append({'params': param, 'lr': layer_lr})

    return param_groups

# Usage for transformer fine-tuning
param_groups = get_layer_wise_lr_groups(model, base_lr=1e-4, decay_rate=0.95)
optimizer = optim.AdamW(param_groups, weight_decay=0.01)

# Discriminative fine-tuning (common in NLP)
def get_discriminative_lr_groups(model, base_lr=1e-3, lr_mult=2.6):
    """Higher LR for later layers"""
    layers = [model.layer1, model.layer2, model.layer3, model.head]

    param_groups = []
    for i, layer in enumerate(layers):
        lr = base_lr * (lr_mult ** i)
        param_groups.append({'params': layer.parameters(), 'lr': lr})

    return param_groups
```

---

## 7. 올바른 옵티마이저 선택하기

### 7.1 결정 가이드

```
Optimization Decision Tree:

Are you training from scratch?
│
├─ Yes: What architecture?
│  ├─ CNN (ResNet, EfficientNet)
│  │  └─ SGD + Momentum (0.9) + Cosine Annealing
│  │     LR: 0.1, batch 256
│  │
│  ├─ Transformer (BERT, GPT, ViT)
│  │  └─ AdamW + Linear Warmup (10%) + Cosine Decay
│  │     LR: 5e-4, batch 256-512, weight_decay 0.01
│  │
│  ├─ GAN
│  │  └─ Adam (β₁=0.5, β₂=0.999) or RMSprop
│  │     LR: 2e-4, no momentum, no warmup
│  │
│  └─ RNN/LSTM
│     └─ AdamW or RMSprop + Gradient Clipping (1.0)
│        LR: 1e-3
│
└─ No: Fine-tuning pretrained model?
   └─ AdamW + Low LR + Per-parameter groups
      Backbone LR: 1e-5, Head LR: 1e-3
      Linear warmup (5-10%) + Cosine decay
```

### 7.2 SGD vs Adam 논쟁

**SGD + Momentum을 사용할 때:**

✅ 처음부터 CNN 학습 (ResNet, VGG)
✅ 일반화가 중요할 때
✅ 광범위한 하이퍼파라미터 검색에 시간을 들일 수 있을 때
✅ 큰 배치 크기를 사용할 수 있을 때

**장점:**
- 종종 더 나은 최종 정확도 (ImageNet에서 0.5-1%)
- 새로운 데이터에 더 나은 일반화
- 큰 배치 크기에서 더 안정적

**단점:**
- 신중한 LR 튜닝 필요
- 더 긴 학습 필요 (90-200 에폭)
- 초기화에 민감
- 큰 배치에는 워밍업 필요

**Adam/AdamW를 사용할 때:**

✅ 트랜스포머 학습
✅ 빠른 프로토타이핑
✅ 사전학습된 모델 파인튜닝
✅ 제한된 계산/시간으로 작업할 때
✅ 작은 배치 크기

**장점:**
- 하이퍼파라미터에 강건 (기본값으로 작동)
- 더 빠른 수렴 (적은 에폭)
- LR 선택에 덜 민감
- 적응형 문제에 좋음 (NLP, RL)

**단점:**
- 비전에서 약간 나쁜 일반화 가능
- 더 높은 메모리 사용
- 더 쉽게 과적합 가능

```python
# Example: Training ResNet on CIFAR-10

# SGD approach (better accuracy, more tuning)
def train_resnet_sgd():
    model = torchvision.models.resnet18(num_classes=10)

    optimizer = optim.SGD(
        model.parameters(),
        lr=0.1,
        momentum=0.9,
        weight_decay=5e-4,
        nesterov=True
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=200,
        eta_min=1e-4
    )

    # Train for 200 epochs
    # Expected accuracy: ~95%

# AdamW approach (faster convergence, easier tuning)
def train_resnet_adamw():
    model = torchvision.models.resnet18(num_classes=10)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=0.01
    )

    # Simple schedule
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

    # Train for 100 epochs
    # Expected accuracy: ~94% (slightly lower but faster)
```

### 7.3 일반적인 레시피

**레시피 1: ImageNet 학습 (ResNet-50)**

```python
# Standard recipe for ImageNet
model = torchvision.models.resnet50(pretrained=False)
optimizer = optim.SGD(
    model.parameters(),
    lr=0.1,
    momentum=0.9,
    weight_decay=1e-4
)

# Warmup for 5 epochs, then cosine decay for 90 epochs
warmup_scheduler = optim.lr_scheduler.LinearLR(
    optimizer,
    start_factor=0.1,
    total_iters=5 * len(train_loader)
)
cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=90 * len(train_loader),
    eta_min=1e-5
)
scheduler = optim.lr_scheduler.SequentialLR(
    optimizer,
    schedulers=[warmup_scheduler, cosine_scheduler],
    milestones=[5 * len(train_loader)]
)

# Batch size 256, 90 epochs total
# Expected Top-1: 76.2%, Top-5: 93.0%
```

**레시피 2: BERT 사전학습**

```python
# BERT-base configuration
optimizer = optim.AdamW(
    model.parameters(),
    lr=1e-4,
    betas=(0.9, 0.999),
    eps=1e-6,
    weight_decay=0.01
)

# 10% linear warmup + 90% linear decay
total_steps = len(train_loader) * num_epochs
warmup_steps = int(0.1 * total_steps)

scheduler = optim.lr_scheduler.LambdaLR(
    optimizer,
    lr_lambda=lambda step: min(
        step / warmup_steps,
        (total_steps - step) / (total_steps - warmup_steps)
    )
)

# Batch size 256, gradient accumulation 4 (effective 1024)
# Gradient clipping: 1.0
```

**레시피 3: 사전학습된 모델 파인튜닝**

```python
# Fine-tuning BERT for classification
model = AutoModel.from_pretrained('bert-base-uncased')
classifier = nn.Linear(768, num_classes)

# Different LR for pretrained vs new layers
optimizer = optim.AdamW([
    {'params': model.parameters(), 'lr': 2e-5},  # Very low for pretrained
    {'params': classifier.parameters(), 'lr': 2e-4}  # Higher for new layer
], weight_decay=0.01)

# Short warmup + cosine decay
total_steps = len(train_loader) * 5  # 5 epochs
warmup_steps = int(0.1 * total_steps)

scheduler = WarmupCosineSchedule(optimizer, warmup_steps, total_steps)

# Small batch size (16-32), few epochs (3-5)
# Gradient clipping: 1.0
```

**레시피 4: GAN 학습**

```python
# DCGAN recipe
generator = Generator()
discriminator = Discriminator()

# Adam with β₁=0.5 (less momentum than default)
opt_G = optim.Adam(generator.parameters(), lr=2e-4, betas=(0.5, 0.999))
opt_D = optim.Adam(discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))

# No scheduler, no weight decay
# Alternate training: 1 D step, 1 G step
# No gradient clipping usually
```

**레시피 5: Vision Transformer (ViT)**

```python
# ViT-B/16 on ImageNet
model = torchvision.models.vit_b_16(pretrained=False)

optimizer = optim.AdamW(
    model.parameters(),
    lr=3e-3,  # Higher than BERT!
    weight_decay=0.3,  # Stronger regularization
    betas=(0.9, 0.999)
)

# Linear warmup + cosine decay
total_steps = len(train_loader) * 300  # 300 epochs
warmup_steps = int(0.05 * total_steps)  # 5% warmup

scheduler = WarmupCosineSchedule(optimizer, warmup_steps, total_steps, min_lr_ratio=0.01)

# Large batch (1024-4096), strong augmentation
# Gradient clipping: 1.0
```

### 7.4 최적화 문제 디버깅

**문제 1: 손실이 감소하지 않음**

```python
# Checklist:
# 1. Check learning rate (try LR finder)
# 2. Verify gradients are flowing
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: grad_norm={param.grad.norm():.6f}")
    else:
        print(f"{name}: NO GRADIENT!")  # Problem!

# 3. Check for NaN/Inf
if torch.isnan(loss) or torch.isinf(loss):
    print("Loss is NaN/Inf! Reduce learning rate or check data.")

# 4. Verify data is changing
print(f"Batch variance: {X.var().item():.6f}")  # Should be non-zero

# 5. Try simpler optimizer (SGD) to isolate issue
optimizer = optim.SGD(model.parameters(), lr=0.001)
```

**문제 2: 손실 폭발**

```python
# Solutions:
# 1. Gradient clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# 2. Lower learning rate
optimizer = optim.Adam(model.parameters(), lr=1e-4)  # Try 10x lower

# 3. Check for numerical instability
# Use float32, avoid log(0), div by 0
loss = torch.log(output + 1e-8)  # Add epsilon

# 4. Use mixed precision
from torch.cuda.amp import autocast
with autocast():
    output = model(X)
```

**문제 3: 학습과 검증 성능 차이**

```python
# Solutions:
# 1. Increase regularization
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.1)  # Higher

# 2. Add dropout
model = nn.Sequential(
    nn.Linear(100, 50),
    nn.ReLU(),
    nn.Dropout(0.5),  # Add dropout
    nn.Linear(50, 10)
)

# 3. Reduce model capacity
# Use smaller model or fewer layers

# 4. More data augmentation
# Stronger augmentation on training data
```

**문제 4: 느린 수렴**

```python
# Solutions:
# 1. Increase learning rate (use LR finder)
# 2. Switch from SGD to Adam
optimizer = optim.AdamW(model.parameters(), lr=1e-3)

# 3. Add learning rate warmup
scheduler = WarmupCosineSchedule(optimizer, warmup_steps=1000, total_steps=10000)

# 4. Check batch size (try larger)
# 5. Verify batch normalization is working
# 6. Check weight initialization
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight)
        m.bias.data.fill_(0.01)

model.apply(init_weights)
```

---

## 연습 문제

### 연습 문제 1: 옵티마이저 구현 및 비교
RMSprop을 처음부터 구현하고 2D 토이 문제에서 SGD 및 Adam과 궤적을 비교하세요. 비볼록 표면(예: Rosenbrock 함수)에서 최적화 경로를 시각화하세요. 수렴 속도(손실 < 0.01에 도달하는 반복 횟수)를 측정하세요.

**보너스**: RMSprop 구현에 네스테로프 모멘텀을 추가하세요.

### 연습 문제 2: 스케줄러 절제 연구
다섯 가지 다른 스케줄러로 CIFAR-10에서 ResNet-18을 학습하세요: (1) 스케줄러 없음, (2) StepLR, (3) CosineAnnealingLR, (4) OneCycleLR, (5) Warmup + Cosine. 동일한 옵티마이저(SGD, momentum=0.9)와 초기 LR을 사용하세요. 학습 곡선을 그리고 최종 테스트 정확도를 보고하세요. 어떤 스케줄러가 최고 정확도를 달성했나요? 어떤 것이 가장 빨리 수렴했나요?

**보너스**: LR 탐색기를 사용하여 OneCycleLR의 최적 초기 LR을 자동으로 결정하세요.

### 연습 문제 3: 대규모 배치 학습 시뮬레이션
그래디언트 축적을 사용하여 대규모 배치 학습을 시뮬레이션하세요. 실제 배치 크기 64와 accumulation_steps=16을 사용하여 유효 배치 크기 1024로 모델을 학습하세요. 학습 시간, 메모리 사용량 및 최종 정확도를 batch_size=1024를 직접 사용한 것과 비교하세요(GPU 메모리가 허용하는 경우). LARS 스타일의 레이어별 LR 스케일링을 구현하고 대규모 배치에 도움이 되는지 보여주세요.

**보너스**: 혼합 정밀도 학습을 추가하고 속도 향상을 측정하세요.

---

## 참고 자료

1. **Ruder, S.** (2016). *An overview of gradient descent optimization algorithms*. arXiv:1609.04747
   - SGD, momentum, Adagrad, RMSprop, Adam의 포괄적인 서베이

2. **Kingma, D. P., & Ba, J.** (2015). *Adam: A Method for Stochastic Optimization*. ICLR 2015
   - 원본 Adam 논문

3. **Loshchilov, I., & Hutter, F.** (2019). *Decoupled Weight Decay Regularization*. ICLR 2019
   - AdamW: Adam에서 가중치 감쇠 수정

4. **You, Y., et al.** (2020). *Large Batch Optimization for Deep Learning: Training BERT in 76 minutes*. ICLR 2020
   - LAMB 옵티마이저

5. **Smith, L. N.** (2018). *A disciplined approach to neural network hyper-parameters*. arXiv:1803.09820
   - 순환 학습률, LR range test, super-convergence

6. **Chen, X., et al.** (2023). *Symbolic Discovery of Optimization Algorithms*. arXiv:2302.06675
   - Lion 옵티마이저 (Google)

7. **Dettmers, T., et al.** (2022). *8-bit Optimizers via Block-wise Quantization*. ICLR 2022
   - 메모리 효율적인 학습을 위한 8-bit Adam

8. **PyTorch Documentation**: https://pytorch.org/docs/stable/optim.html
   - 공식 옵티마이저 및 스케줄러 문서

9. **Goodfellow, I., et al.** (2016). *Deep Learning*. MIT Press
   - Chapter 8: Optimization for Training Deep Models

10. **Zhang, M., et al.** (2020). *Lookahead Optimizer: k steps forward, 1 step back*. NeurIPS 2019
    - 모든 옵티마이저를 위한 Lookahead 래퍼
