# 02. 타이니 GAN (Tiny GAN)

**난이도: ⭐⭐⭐ (고급)**

## 학습 목표

- 적대적 생성 네트워크(Generative Adversarial Network, GAN)의 핵심 원리 이해
- 미니맥스 게임(Minimax Game)과 내쉬 균형(Nash Equilibrium)의 관계 파악
- 생성자(Generator)와 판별자(Discriminator)의 교대 학습 구현
- 모드 붕괴(Mode Collapse), 학습 불안정 등 일반적 실패 모드 진단
- 200-400줄 이내의 단일 파일 GAN 구현

**관련 토픽**: Deep_Learning, Probability_and_Statistics

---

## 1. 이론적 배경

### 1.1 적대적 학습 (Adversarial Training)

GAN은 두 신경망이 **서로 경쟁하며 학습**하는 프레임워크입니다:

```
                 노이즈 z ~ p(z)
                      │
                      ▼
               ┌─────────────┐
               │   생성자 G  │   ← 가짜 데이터 생성
               └──────┬──────┘
                      │ G(z)
                      ▼
              ┌───────────────┐
  실제 x ──→ │   판별자 D    │ ──→ 진짜/가짜 확률
              └───────────────┘
```

- **생성자(Generator, G)**: 랜덤 노이즈 `z`를 입력받아 실제 데이터와 유사한 샘플을 생성
- **판별자(Discriminator, D)**: 입력 데이터가 실제인지 생성된 것인지를 분류

### 1.2 미니맥스 게임 (Minimax Game)

GAN의 학습은 다음 미니맥스 목적 함수로 정의됩니다:

$$
\min_G \max_D \; V(D, G) = \mathbb{E}_{x \sim p_{data}}[\log D(x)] + \mathbb{E}_{z \sim p_z}[\log(1 - D(G(z)))]
$$

직관적 해석:
- **판별자 D**는 $V$를 **최대화**: 실제 데이터에 높은 확률, 가짜 데이터에 낮은 확률 할당
- **생성자 G**는 $V$를 **최소화**: 판별자를 속여 가짜 데이터에 높은 확률을 할당하도록 유도

### 1.3 내쉬 균형 (Nash Equilibrium)

이론적으로 GAN은 다음 조건에서 **내쉬 균형**에 도달합니다:
- $G$가 실제 데이터 분포 $p_{data}$를 완벽히 모방
- $D(x) = 0.5$ (모든 입력에 대해 진짜/가짜 구분 불가)

```python
# 이상적인 균형 상태
# D(x_real) → 0.5   (실제 데이터도 확신 못함)
# D(G(z))  → 0.5   (가짜 데이터도 확신 못함)
# p_G      ≡ p_data (생성 분포 = 실제 분포)
```

실제로는 이 균형에 도달하기 어렵고, 다양한 학습 불안정 문제가 발생합니다.

---

## 2. 구현 워크스루

### 2.1 데이터 생성

간단한 2D 분포를 학습 대상으로 사용합니다.

```python
import numpy as np

def sample_real_data(n):
    """Generate samples from a 2D Gaussian mixture."""
    k = np.random.choice(4, size=n)
    centers = np.array([[2, 2], [-2, 2], [-2, -2], [2, -2]], dtype=np.float32)
    samples = centers[k] + np.random.randn(n, 2).astype(np.float32) * 0.3
    return samples

def sample_noise(n, dim=16):
    """Sample latent noise vectors."""
    return np.random.randn(n, dim).astype(np.float32)
```

### 2.2 네트워크 정의

생성자와 판별자를 간단한 MLP로 구현합니다.

```python
def init_weights(shape):
    """Xavier initialization."""
    fan_in, fan_out = shape
    scale = np.sqrt(2.0 / (fan_in + fan_out))
    return np.random.randn(*shape).astype(np.float32) * scale

class Generator:
    def __init__(self, z_dim=16, hidden=64, out_dim=2):
        self.W1 = init_weights((z_dim, hidden))
        self.b1 = np.zeros(hidden, dtype=np.float32)
        self.W2 = init_weights((hidden, hidden))
        self.b2 = np.zeros(hidden, dtype=np.float32)
        self.W3 = init_weights((hidden, out_dim))
        self.b3 = np.zeros(out_dim, dtype=np.float32)

    def forward(self, z):
        h1 = np.maximum(0, z @ self.W1 + self.b1)       # ReLU
        h2 = np.maximum(0, h1 @ self.W2 + self.b2)      # ReLU
        return h2 @ self.W3 + self.b3                    # Linear

    def parameters(self):
        return [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3]

class Discriminator:
    def __init__(self, in_dim=2, hidden=64):
        self.W1 = init_weights((in_dim, hidden))
        self.b1 = np.zeros(hidden, dtype=np.float32)
        self.W2 = init_weights((hidden, hidden))
        self.b2 = np.zeros(hidden, dtype=np.float32)
        self.W3 = init_weights((hidden, 1))
        self.b3 = np.zeros(1, dtype=np.float32)

    def forward(self, x):
        h1 = np.maximum(0, x @ self.W1 + self.b1)       # ReLU
        h2 = np.maximum(0, h1 @ self.W2 + self.b2)      # ReLU
        logit = h2 @ self.W3 + self.b3
        return 1.0 / (1.0 + np.exp(-logit))             # Sigmoid

    def parameters(self):
        return [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3]
```

### 2.3 학습 루프

판별자와 생성자를 교대로 학습시킵니다.

```python
def train_gan(epochs=5000, batch_size=128, lr=1e-3):
    G = Generator()
    D = Discriminator()

    for epoch in range(epochs):
        # --- Train Discriminator ---
        real = sample_real_data(batch_size)
        z = sample_noise(batch_size)
        fake = G.forward(z)

        d_real = D.forward(real)
        d_fake = D.forward(fake)

        # D loss: -[log(D(x)) + log(1 - D(G(z)))]
        d_loss = -np.mean(np.log(d_real + 1e-8) + np.log(1 - d_fake + 1e-8))
        # ... compute gradients and update D parameters ...

        # --- Train Generator ---
        z = sample_noise(batch_size)
        fake = G.forward(z)
        d_fake = D.forward(fake)

        # G loss: -log(D(G(z)))  (non-saturating variant)
        g_loss = -np.mean(np.log(d_fake + 1e-8))
        # ... compute gradients and update G parameters ...

        if epoch % 500 == 0:
            print(f"Epoch {epoch}: D_loss={d_loss:.4f}, G_loss={g_loss:.4f}")
```

### 2.4 비포화 생성자 손실 (Non-saturating Loss)

원래 미니맥스 목적에서 생성자 손실은 $\log(1 - D(G(z)))$이지만, 학습 초기에 그래디언트가 포화됩니다. 실전에서는 **비포화(non-saturating)** 변형을 사용합니다:

```python
# 원래 (포화됨, 그래디언트 약함)
g_loss_original = np.mean(np.log(1 - d_fake + 1e-8))

# 비포화 변형 (강한 그래디언트)
g_loss_nonsaturating = -np.mean(np.log(d_fake + 1e-8))
```

---

## 3. 일반적 실패 모드

### 3.1 모드 붕괴 (Mode Collapse)

생성자가 데이터 분포의 일부 모드(mode)만 생성하고 다른 모드를 무시하는 현상입니다.

```
실제 분포: 4개의 가우시안 혼합
   ● ●           ● ●
   ● ●           ● ●

모드 붕괴 시:
   ● ●
   ● ●           (한 곳에만 집중)
```

**완화 방법**:
- 미니배치 판별(Minibatch discrimination)
- 학습률 및 아키텍처 조정
- 스펙트럴 정규화(Spectral normalization)

### 3.2 학습 불안정 (Training Instability)

판별자가 너무 강해지거나 너무 약해지면 학습이 발산합니다.

| 상황 | 증상 | 해결 |
|------|------|------|
| D가 너무 강함 | G 그래디언트 소실 | D 학습 횟수 줄이기 |
| D가 너무 약함 | G에 유용한 신호 없음 | D 용량 늘리기 |
| 학습률 과다 | 손실 발산 | 학습률 감소 |
| 학습률 과소 | 수렴 느림 | 학습률 증가 |

### 3.3 그래디언트 소실/폭발

```python
# 안정적 학습을 위한 실전 팁
# 1. 로그 확률에 작은 epsilon 추가
np.log(d_real + 1e-8)   # 수치 안정성

# 2. 그래디언트 클리핑
grad = np.clip(grad, -1.0, 1.0)

# 3. 학습률 스케줄링
lr = lr_init * (1.0 / (1.0 + decay * epoch))
```

---

## 4. 연습문제

### 연습문제 1: 1차원 GAN

단일 가우시안 분포 $\mathcal{N}(5, 1)$을 학습하는 1D GAN을 구현하세요. 생성된 샘플의 평균과 분산이 목표에 수렴하는지 확인하세요.

### 연습문제 2: WGAN 손실

Wasserstein 손실을 구현하세요: $L_D = \mathbb{E}[D(G(z))] - \mathbb{E}[D(x)]$, 가중치 클리핑 포함.

### 연습문제 3: 학습 곡선 시각화

`d_loss`, `g_loss`, 그리고 생성 샘플의 분포를 에포크별로 시각화하세요.

### 연습문제 4: 판별자-생성자 학습 비율

판별자와 생성자의 학습 비율(예: D를 5번 학습 후 G를 1번 학습)을 변경하며 효과를 비교하세요.

---

## 5. 참고 자료

- Goodfellow, I. J., et al. (2014). "Generative Adversarial Nets." *NeurIPS*.
- Arjovsky, M., Chintala, S., & Bottou, L. (2017). "Wasserstein GAN." *ICML*.
- Salimans, T., et al. (2016). "Improved Techniques for Training GANs." *NeurIPS*.
- Goodfellow, I. (2016). "NIPS 2016 Tutorial: Generative Adversarial Networks." https://arxiv.org/abs/1701.00160

---

**이전 레슨**: [01_Micro_Autograd.md](01_Micro_Autograd.md) — 마이크로 오토그래드
**다음 레슨**: [03_Nano_RL.md](03_Nano_RL.md) — 나노 RL: 정책 그래디언트 구현
