# 05. 마이크로 VAE (Micro VAE)

**난이도: ⭐⭐⭐ (고급)**

## 학습 목표

- 잠재 변수 모델(Latent Variable Model)의 동기와 수학적 구조 이해
- 증거 하한(Evidence Lower Bound, ELBO)의 유도 과정 파악
- 재매개변수화 트릭(Reparameterization Trick)이 왜 필요한지 이해
- KL 발산(Kullback-Leibler Divergence)의 역할과 닫힌 형태 계산
- 단일 파일로 VAE(Variational Autoencoder)를 구현하고 학습

**관련 토픽**: Deep_Learning, Probability_and_Statistics

---

## 1. 이론적 배경

### 1.1 잠재 변수 모델 (Latent Variable Model)

잠재 변수 모델은 관측 데이터 $x$ 뒤에 숨겨진(관측되지 않은) 잠재 변수 $z$가 존재한다고 가정합니다.

```
잠재 변수 모델의 구조:

  z ~ p(z)           ← 잠재 변수 (사전 분포)
      │
      ▼
  x ~ p_θ(x|z)      ← 관측 데이터 (조건부 분포, 디코더)
```

**핵심 아이디어**: 복잡한 데이터 분포 $p(x)$를 간단한 잠재 공간(latent space)의 분포 $p(z)$와 변환 $p_\theta(x \mid z)$로 분해합니다.

**한계**: 주변 우도(marginal likelihood) $p_\theta(x) = \int p_\theta(x \mid z) p(z) dz$는 일반적으로 다루기 어렵습니다(intractable).

### 1.2 ELBO (Evidence Lower Bound)

직접 $p_\theta(x)$를 최대화할 수 없으므로, 변분 추론(Variational Inference)을 사용합니다. 인코더 $q_\phi(z \mid x)$를 도입하여 ELBO를 유도합니다:

$$
\log p_\theta(x) \geq \underbrace{\mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)]}_{\text{복원 항 (Reconstruction)}} - \underbrace{D_{KL}(q_\phi(z|x) \| p(z))}_{\text{정규화 항 (Regularization)}}
$$

**ELBO = 복원 항 - KL 항**:

| 항 | 역할 | 직관 |
|---|------|------|
| 복원 항 | 디코더가 데이터를 잘 복원 | 인코딩 정보의 충실한 복원 |
| KL 항 | 잠재 분포가 사전 분포와 유사 | 잠재 공간의 규칙성 유지 |

```
         x (입력)
         │
         ▼
    ┌──────────┐
    │ 인코더   │ q_φ(z|x)    ← 잠재 분포 추론
    │ (Encoder)│
    └────┬─────┘
         │ μ, σ²
         ▼
    ┌──────────┐
    │ 샘플링   │ z = μ + σ·ε  ← 재매개변수화 트릭
    │ (Sample) │  (ε ~ N(0,1))
    └────┬─────┘
         │ z
         ▼
    ┌──────────┐
    │ 디코더   │ p_θ(x|z)    ← 데이터 복원
    │ (Decoder)│
    └────┬─────┘
         │
         ▼
       x̂ (복원)
```

### 1.3 재매개변수화 트릭 (Reparameterization Trick)

$z$를 $q_\phi(z \mid x)$에서 직접 샘플링하면 역전파가 불가능합니다. 재매개변수화 트릭은 확률적 노드를 결정론적 함수로 변환합니다:

$$
z = \mu_\phi(x) + \sigma_\phi(x) \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0, \mathbf{I})
$$

```python
# 문제: 샘플링은 미분 불가
z = np.random.normal(mu, sigma)  # ❌ ∂z/∂mu, ∂z/∂sigma 계산 불가

# 해결: 재매개변수화 트릭
eps = np.random.randn(*mu.shape)  # 외부 노이즈
z = mu + sigma * eps              # ✅ ∂z/∂mu = 1, ∂z/∂sigma = eps
```

### 1.4 KL 발산 (KL Divergence)

$q_\phi(z \mid x) = \mathcal{N}(\mu, \sigma^2 \mathbf{I})$이고 $p(z) = \mathcal{N}(0, \mathbf{I})$일 때, KL 발산은 닫힌 형태로 계산됩니다:

$$
D_{KL} = -\frac{1}{2} \sum_{j=1}^{d} \left( 1 + \log \sigma_j^2 - \mu_j^2 - \sigma_j^2 \right)
$$

```python
def kl_divergence(mu, log_var):
    """Closed-form KL divergence: q(z|x) || N(0, I)."""
    return -0.5 * np.sum(1 + log_var - mu**2 - np.exp(log_var))
```

---

## 2. 구현 워크스루

### 2.1 인코더와 디코더

```python
import numpy as np

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

def relu(x):
    return np.maximum(0, x)

class Encoder:
    """Maps input x to latent distribution parameters (mu, log_var)."""

    def __init__(self, in_dim, hidden_dim, latent_dim):
        scale = lambda fan_in, fan_out: np.sqrt(2.0 / (fan_in + fan_out))
        self.W1 = np.random.randn(in_dim, hidden_dim).astype(np.float32) * scale(in_dim, hidden_dim)
        self.b1 = np.zeros(hidden_dim, dtype=np.float32)
        self.W_mu = np.random.randn(hidden_dim, latent_dim).astype(np.float32) * scale(hidden_dim, latent_dim)
        self.b_mu = np.zeros(latent_dim, dtype=np.float32)
        self.W_lv = np.random.randn(hidden_dim, latent_dim).astype(np.float32) * scale(hidden_dim, latent_dim)
        self.b_lv = np.zeros(latent_dim, dtype=np.float32)

    def forward(self, x):
        h = relu(x @ self.W1 + self.b1)
        mu = h @ self.W_mu + self.b_mu
        log_var = h @ self.W_lv + self.b_lv
        return mu, log_var

    def parameters(self):
        return [self.W1, self.b1, self.W_mu, self.b_mu, self.W_lv, self.b_lv]

class Decoder:
    """Maps latent vector z back to data space."""

    def __init__(self, latent_dim, hidden_dim, out_dim):
        scale = lambda fan_in, fan_out: np.sqrt(2.0 / (fan_in + fan_out))
        self.W1 = np.random.randn(latent_dim, hidden_dim).astype(np.float32) * scale(latent_dim, hidden_dim)
        self.b1 = np.zeros(hidden_dim, dtype=np.float32)
        self.W2 = np.random.randn(hidden_dim, out_dim).astype(np.float32) * scale(hidden_dim, out_dim)
        self.b2 = np.zeros(out_dim, dtype=np.float32)

    def forward(self, z):
        h = relu(z @ self.W1 + self.b1)
        return sigmoid(h @ self.W2 + self.b2)  # [0, 1] output

    def parameters(self):
        return [self.W1, self.b1, self.W2, self.b2]
```

### 2.2 VAE 클래스

```python
class VAE:
    """Variational Autoencoder."""

    def __init__(self, in_dim, hidden_dim=128, latent_dim=2):
        self.encoder = Encoder(in_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim, in_dim)
        self.latent_dim = latent_dim

    def reparameterize(self, mu, log_var):
        """Reparameterization trick: z = mu + sigma * epsilon."""
        std = np.exp(0.5 * log_var)
        eps = np.random.randn(*mu.shape).astype(np.float32)
        return mu + std * eps

    def forward(self, x):
        mu, log_var = self.encoder.forward(x)
        z = self.reparameterize(mu, log_var)
        x_recon = self.decoder.forward(z)
        return x_recon, mu, log_var

    def loss(self, x, x_recon, mu, log_var):
        """ELBO loss = Reconstruction + KL divergence."""
        # Binary cross-entropy reconstruction loss
        recon_loss = -np.sum(
            x * np.log(x_recon + 1e-8) + (1 - x) * np.log(1 - x_recon + 1e-8)
        )
        # KL divergence (closed form)
        kl_loss = -0.5 * np.sum(1 + log_var - mu**2 - np.exp(log_var))
        return recon_loss + kl_loss

    def parameters(self):
        return self.encoder.parameters() + self.decoder.parameters()
```

### 2.3 학습 루프

```python
def train_vae(model, data, epochs=5000, lr=1e-3, batch_size=64):
    """Train VAE with simple SGD."""

    for epoch in range(epochs):
        idx = np.random.choice(len(data), batch_size)
        x_batch = data[idx]

        total_loss = 0.0
        for i in range(batch_size):
            x = x_batch[i]
            x_recon, mu, log_var = model.forward(x)
            loss = model.loss(x, x_recon, mu, log_var)
            total_loss += loss

            # ... compute gradients and update parameters ...

        if epoch % 500 == 0:
            avg_loss = total_loss / batch_size
            print(f"Epoch {epoch}: loss={avg_loss:.4f}")
```

### 2.4 생성과 보간

학습된 VAE로 새로운 데이터를 생성하고, 잠재 공간에서 보간(interpolation)합니다.

```python
def generate(model, n_samples=16):
    """Generate new samples from prior p(z) = N(0, I)."""
    z = np.random.randn(n_samples, model.latent_dim).astype(np.float32)
    samples = np.array([model.decoder.forward(z[i]) for i in range(n_samples)])
    return samples

def interpolate(model, x1, x2, steps=10):
    """Interpolate between two data points in latent space."""
    mu1, _ = model.encoder.forward(x1)
    mu2, _ = model.encoder.forward(x2)

    results = []
    for alpha in np.linspace(0, 1, steps):
        z = (1 - alpha) * mu1 + alpha * mu2
        x_interp = model.decoder.forward(z)
        results.append(x_interp)
    return np.array(results)
```

---

## 3. 핵심 분석

### 3.1 KL 붕괴 (KL Collapse / Posterior Collapse)

학습 초기에 디코더가 $z$를 무시하고 평균적인 출력만 생성하는 현상입니다. KL 항이 0에 가까워지고, 잠재 변수가 무의미해집니다.

```python
# KL 붕괴 진단
# KL 값이 매우 작으면 (< 0.1) → 잠재 변수 미활용
# KL 값이 매우 크면 (> 100) → 사전 분포와 너무 동떨어짐

# 완화 방법: KL 어닐링 (KL Annealing)
beta = min(1.0, epoch / warmup_epochs)  # 점진적 증가
loss = recon_loss + beta * kl_loss
```

### 3.2 β-VAE

KL 항에 가중치 $\beta$를 부여하여 잠재 공간의 분리(disentanglement)를 제어합니다:

$$
L_{\beta\text{-VAE}} = \text{Recon} + \beta \cdot D_{KL}
$$

| $\beta$ 값 | 효과 |
|------------|------|
| $\beta < 1$ | 복원 품질 우선, 잠재 공간 불규칙 |
| $\beta = 1$ | 표준 VAE |
| $\beta > 1$ | 잠재 공간 분리 강화, 복원 품질 하락 가능 |

### 3.3 VAE vs AE vs GAN

| 특성 | 오토인코더(AE) | VAE | GAN |
|------|---------------|-----|-----|
| 잠재 공간 | 비정규 | 정규(가우시안) | 비정규 |
| 생성 가능 | 어려움 | 용이 | 용이 |
| 학습 안정성 | 높음 | 높음 | 낮음 |
| 샘플 품질 | — | 약간 흐림 | 선명 |
| 확률적 해석 | 없음 | 있음 | 암묵적 |

---

## 4. 연습문제

### 연습문제 1: 잠재 공간 시각화

2D 잠재 공간($d=2$)을 사용하여 학습한 후, 인코더의 $\mu$ 값을 2D 평면에 시각화하세요. 클래스별 클러스터가 형성되는지 확인하세요.

### 연습문제 2: 잠재 차원 실험

잠재 차원 $d$를 2, 8, 32, 128로 변경하며 복원 품질과 KL 발산의 변화를 관찰하세요.

### 연습문제 3: KL 어닐링

KL 어닐링(warmup 0 → $\beta=1$)을 구현하고, 어닐링 유무에 따른 학습 곡선을 비교하세요.

### 연습문제 4: 조건부 VAE (CVAE)

클래스 레이블을 인코더와 디코더에 입력으로 추가하여 조건부 VAE를 구현하세요.

### 연습문제 5: 잠재 공간 산술

"웃는 얼굴" - "무표정" + "안경" = "웃는 안경" 같은 잠재 공간 산술이 가능한지 실험하세요.

---

## 5. 참고 자료

- Kingma, D. P. & Welling, M. (2013). "Auto-Encoding Variational Bayes." *ICLR*. https://arxiv.org/abs/1312.6114
- Doersch, C. (2016). "Tutorial on Variational Autoencoders." https://arxiv.org/abs/1606.05908
- Higgins, I., et al. (2017). "β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework." *ICLR*.
- Rezende, D. J. & Mohamed, S. (2015). "Variational Inference with Normalizing Flows." *ICML*.

---

**이전 레슨**: [04_Pico_Diffusion.md](04_Pico_Diffusion.md) — 피코 디퓨전
**다음 토픽으로**: [00_Overview.md](00_Overview.md) — Flagship 개요로 돌아가기
