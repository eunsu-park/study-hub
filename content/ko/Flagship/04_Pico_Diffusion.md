# 04. 피코 디퓨전 (Pico Diffusion)

**난이도: ⭐⭐⭐⭐ (전문)**

## 학습 목표

- 전방 확산 과정(Forward Diffusion Process)의 수학적 정의 이해
- 노이즈 스케줄(Noise Schedule)의 역할과 설계 파악
- 디노이징 점수 매칭(Denoising Score Matching)의 원리 학습
- ELBO(Evidence Lower Bound)와 확산 모델의 연결 이해
- 단일 파일로 DDPM(Denoising Diffusion Probabilistic Model) 구현

**관련 토픽**: Deep_Learning, Probability_and_Statistics

---

## 1. 이론적 배경

### 1.1 전방 확산 과정 (Forward Diffusion)

전방 확산은 데이터에 점진적으로 가우시안 노이즈(Gaussian Noise)를 추가하여 순수 노이즈로 변환하는 과정입니다.

$$
q(x_t \mid x_{t-1}) = \mathcal{N}(x_t; \sqrt{1 - \beta_t} \, x_{t-1}, \, \beta_t \mathbf{I})
$$

```
전방 확산 과정:

  x_0          x_1          x_2              x_T
 (데이터)  → (약간 노이즈) → (더 노이즈) → ... → (순수 노이즈)
   ■■          ■░          ░░               ░░░
   ■■          ░■          ░░               ░░░

  t=0         t=1          t=2              t=T
           β₁ 추가       β₂ 추가          β_T 추가
```

**닫힌 형태(Closed Form)**: 임의의 시점 $t$에서 $x_t$를 직접 샘플링할 수 있습니다.

$$
q(x_t \mid x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t} \, x_0, \, (1 - \bar{\alpha}_t) \mathbf{I})
$$

여기서 $\alpha_t = 1 - \beta_t$, $\bar{\alpha}_t = \prod_{s=1}^{t} \alpha_s$입니다.

```python
def forward_diffusion_sample(x_0, t, noise_schedule):
    """Sample x_t from q(x_t | x_0) in closed form."""
    alpha_bar = noise_schedule['alpha_bar'][t]
    noise = np.random.randn(*x_0.shape).astype(np.float32)
    x_t = np.sqrt(alpha_bar) * x_0 + np.sqrt(1 - alpha_bar) * noise
    return x_t, noise
```

### 1.2 노이즈 스케줄 (Noise Schedule)

$\beta_t$ 시퀀스는 노이즈가 추가되는 속도를 제어합니다.

```python
def linear_schedule(T=1000, beta_start=1e-4, beta_end=0.02):
    """Linear noise schedule (Ho et al. 2020)."""
    betas = np.linspace(beta_start, beta_end, T, dtype=np.float32)
    alphas = 1.0 - betas
    alpha_bar = np.cumprod(alphas)
    return {
        'betas': betas,
        'alphas': alphas,
        'alpha_bar': alpha_bar,
        'sqrt_alpha_bar': np.sqrt(alpha_bar),
        'sqrt_one_minus_alpha_bar': np.sqrt(1.0 - alpha_bar),
    }

def cosine_schedule(T=1000, s=0.008):
    """Cosine noise schedule (Nichol & Dhariwal 2021)."""
    steps = np.arange(T + 1, dtype=np.float64)
    f = np.cos((steps / T + s) / (1 + s) * np.pi / 2) ** 2
    alpha_bar = (f / f[0]).astype(np.float32)
    betas = np.clip(1 - alpha_bar[1:] / alpha_bar[:-1], 0.0, 0.999)
    alphas = 1.0 - betas
    return {'betas': betas, 'alphas': alphas, 'alpha_bar': alpha_bar[1:]}
```

| 스케줄 | 특징 | 장점 |
|--------|------|------|
| 선형(Linear) | $\beta_t$가 선형 증가 | 구현 간단, 원논문 기본값 |
| 코사인(Cosine) | $\bar{\alpha}_t$가 코사인 감소 | 초기에 노이즈 천천히 추가, 품질 향상 |

### 1.3 디노이징 (Denoising)

역방향 과정(Reverse Process)은 노이즈를 제거하여 데이터를 복원합니다:

$$
p_\theta(x_{t-1} \mid x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \, \sigma_t^2 \mathbf{I})
$$

신경망 $\epsilon_\theta(x_t, t)$는 $x_t$에 추가된 노이즈를 예측하며, 이를 사용하여 평균 $\mu_\theta$를 계산합니다:

$$
\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}} \epsilon_\theta(x_t, t) \right)
$$

### 1.4 ELBO 연결

확산 모델의 학습 목표는 변분 하한(Evidence Lower Bound, ELBO)에서 유도됩니다. 간소화된 학습 목표:

$$
L_{\text{simple}} = \mathbb{E}_{t, x_0, \epsilon} \left[ \| \epsilon - \epsilon_\theta(x_t, t) \|^2 \right]
$$

이는 직관적으로 "신경망이 추가된 노이즈를 정확히 예측하도록 학습"하는 것입니다.

---

## 2. 구현 워크스루

### 2.1 간단한 노이즈 예측 네트워크

2D 데이터에 대한 간소화된 구현입니다.

```python
import numpy as np

class TimeEmbedding:
    """Sinusoidal time embedding."""

    def __init__(self, dim=32):
        self.dim = dim

    def __call__(self, t, T=1000):
        half = self.dim // 2
        freqs = np.exp(-np.log(T) * np.arange(half, dtype=np.float32) / half)
        args = t * freqs
        return np.concatenate([np.sin(args), np.cos(args)])

class NoisePredictor:
    """Simple MLP that predicts noise ε given (x_t, t)."""

    def __init__(self, data_dim=2, time_dim=32, hidden=128):
        in_dim = data_dim + time_dim
        self.time_embed = TimeEmbedding(time_dim)

        # Layer 1
        self.W1 = np.random.randn(in_dim, hidden).astype(np.float32) * 0.02
        self.b1 = np.zeros(hidden, dtype=np.float32)
        # Layer 2
        self.W2 = np.random.randn(hidden, hidden).astype(np.float32) * 0.02
        self.b2 = np.zeros(hidden, dtype=np.float32)
        # Layer 3
        self.W3 = np.random.randn(hidden, data_dim).astype(np.float32) * 0.02
        self.b3 = np.zeros(data_dim, dtype=np.float32)

    def forward(self, x_t, t):
        t_emb = self.time_embed(t)
        inp = np.concatenate([x_t, t_emb])
        h1 = np.maximum(0, inp @ self.W1 + self.b1)     # ReLU
        h2 = np.maximum(0, h1 @ self.W2 + self.b2)      # ReLU
        return h2 @ self.W3 + self.b3                    # Linear (predict noise)

    def parameters(self):
        return [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3]
```

### 2.2 학습 루프

```python
def train_diffusion(model, data, schedule, epochs=5000, lr=1e-3, batch_size=64):
    """Train denoising diffusion model."""
    T = len(schedule['betas'])

    for epoch in range(epochs):
        # Sample batch
        idx = np.random.choice(len(data), batch_size)
        x_0 = data[idx]

        total_loss = 0.0
        for i in range(batch_size):
            # Random timestep
            t = np.random.randint(0, T)

            # Forward diffusion: q(x_t | x_0)
            x_t, noise_true = forward_diffusion_sample(x_0[i], t, schedule)

            # Predict noise
            noise_pred = model.forward(x_t, t)

            # MSE loss: ||ε - ε_θ(x_t, t)||²
            loss = np.mean((noise_true - noise_pred) ** 2)
            total_loss += loss

            # ... compute gradients and update parameters ...

        if epoch % 500 == 0:
            avg_loss = total_loss / batch_size
            print(f"Epoch {epoch}: loss={avg_loss:.6f}")
```

### 2.3 샘플링 (역방향 과정)

학습된 모델로 새로운 데이터를 생성합니다.

```python
def sample(model, schedule, n_samples=64, data_dim=2):
    """Generate samples via reverse diffusion."""
    T = len(schedule['betas'])
    x = np.random.randn(n_samples, data_dim).astype(np.float32)  # x_T ~ N(0, I)

    for t in reversed(range(T)):
        alpha = schedule['alphas'][t]
        alpha_bar = schedule['alpha_bar'][t]
        beta = schedule['betas'][t]

        # Predict noise for each sample
        noise_pred = np.array([model.forward(x[i], t) for i in range(n_samples)])

        # Compute mean: μ_θ(x_t, t)
        mean = (1.0 / np.sqrt(alpha)) * (
            x - (beta / np.sqrt(1 - alpha_bar)) * noise_pred
        )

        # Add noise (except at t=0)
        if t > 0:
            noise = np.random.randn(n_samples, data_dim).astype(np.float32)
            x = mean + np.sqrt(beta) * noise
        else:
            x = mean

    return x
```

---

## 3. 핵심 분석

### 3.1 왜 노이즈를 예측하는가?

$x_0$를 직접 예측하는 대신 노이즈 $\epsilon$을 예측하는 이유:

1. **학습 안정성**: 노이즈 예측이 수치적으로 더 안정적
2. **손실 함수 단순화**: ELBO를 간소화한 $L_{\text{simple}}$과 직접 대응
3. **모든 시점에서 균등한 기여**: 노이즈 스케일이 정규화됨

### 3.2 시간 임베딩의 역할

```python
# 시간 정보 없이: 모델이 현재 노이즈 레벨을 모름
noise_pred = model(x_t)          # ❌ 어떤 t인지 알 수 없음

# 시간 임베딩 포함: 모델이 적절한 디노이징 수행
noise_pred = model(x_t, t_emb)   # ✅ 노이즈 레벨에 맞는 예측
```

사인파(sinusoidal) 임베딩은 트랜스포머(Transformer)의 위치 인코딩과 동일한 원리입니다.

---

## 4. 연습문제

### 연습문제 1: 스케줄 비교

선형 스케줄과 코사인 스케줄로 각각 학습한 후, 생성 품질을 비교하세요.

### 연습문제 2: 스텝 수 변경

$T$를 100, 500, 1000으로 변경하며 생성 품질과 속도의 트레이드오프를 관찰하세요.

### 연습문제 3: DDIM 샘플링

결정론적(deterministic) 샘플링인 DDIM(Denoising Diffusion Implicit Models)을 구현하여 더 적은 스텝으로 샘플링하세요.

### 연습문제 4: 조건부 생성

클래스 레이블을 시간 임베딩과 함께 모델에 입력하여 조건부 생성(Conditional Generation)을 구현하세요.

### 연습문제 5: 시각화

전방 확산의 각 스텝에서 $x_t$의 변화를 시각화하고, 역방향 샘플링 과정도 시각화하세요.

---

## 5. 참고 자료

- Ho, J., Jain, A., & Abbeel, P. (2020). "Denoising Diffusion Probabilistic Models." *NeurIPS*. https://arxiv.org/abs/2006.11239
- Nichol, A. & Dhariwal, P. (2021). "Improved Denoising Diffusion Probabilistic Models." *ICML*.
- Song, J., Meng, C., & Ermon, S. (2020). "Denoising Diffusion Implicit Models." *ICLR*.
- Luo, C. (2022). "Understanding Diffusion Models: A Unified Perspective." https://arxiv.org/abs/2208.11970

---

**이전 레슨**: [03_Nano_RL.md](03_Nano_RL.md) — 나노 RL
**다음 레슨**: [05_Micro_VAE.md](05_Micro_VAE.md) — 마이크로 VAE: 변분 오토인코더 구현
