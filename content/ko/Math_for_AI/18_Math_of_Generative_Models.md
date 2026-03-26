# 18. 생성 모델의 수학

## 학습 목표

- 생성 모델의 목표와 명시적/암묵적 밀도 모델의 차이를 이해할 수 있다
- VAE의 ELBO를 완전히 유도하고 재구성 항과 KL 항의 의미를 설명할 수 있다
- GAN의 미니맥스 게임 이론과 JS 발산의 관계를 이해할 수 있다
- Wasserstein 거리와 최적 수송 이론의 기초를 이해할 수 있다
- 확산 모델의 정방향/역방향 과정과 스코어 매칭의 수학을 이해할 수 있다
- Flow Matching과 연속 정규화 흐름(CNF)의 최신 발전을 이해할 수 있다

---

## 1. 생성 모델의 목표

### 1.1 데이터 분포 학습

**목표**: 관측된 데이터 $\{\mathbf{x}^{(1)}, \ldots, \mathbf{x}^{(n)}\} \sim p_{\text{data}}(\mathbf{x})$로부터 진짜 데이터 분포 $p_{\text{data}}$를 학습

**응용**:
- **샘플링**: 새로운 데이터 생성
- **밀도 추정**: $p(\mathbf{x})$ 계산 (이상 탐지)
- **조건부 생성**: $p(\mathbf{y}|\mathbf{x})$ (이미지-텍스트, 스타일 변환)

### 1.2 명시적 vs 암묵적 밀도

**명시적 밀도 모델**:
- 확률 밀도 $p_\theta(\mathbf{x})$를 직접 정의
- 우도 $\log p_\theta(\mathbf{x})$를 최대화
- 예: VAE, 자기회귀 모델, 정규화 흐름

**암묵적 밀도 모델**:
- 밀도를 명시하지 않고 샘플링 과정만 정의
- $\mathbf{x} = G_\theta(\mathbf{z})$, $\mathbf{z} \sim p(\mathbf{z})$
- 예: GAN

### 1.3 학습 패러다임

| 방법 | 목적 함수 | 샘플링 | 밀도 평가 |
|------|----------|--------|----------|
| **VAE** | ELBO 최대화 | 빠름 | 근사 |
| **GAN** | 적대적 게임 | 빠름 | 불가 |
| **정규화 흐름** | 정확한 우도 | 빠름 | 정확 |
| **확산 모델** | 노이즈 예측 | 느림 (반복) | 암묵적 |

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# 간단한 1D 데이터 분포 (혼합 가우시안)
def generate_data(n_samples=1000):
    """진짜 데이터 분포: 두 개의 가우시안 혼합"""
    modes = np.random.choice([0, 1], size=n_samples, p=[0.3, 0.7])
    data = np.where(modes == 0,
                    np.random.randn(n_samples) * 0.5 - 2,
                    np.random.randn(n_samples) * 0.7 + 2)
    return data

data = generate_data(n_samples=5000)

plt.figure(figsize=(10, 4))
plt.hist(data, bins=50, density=True, alpha=0.7, label='True data distribution')
plt.xlabel('x')
plt.ylabel('Density')
plt.title('Example data distribution (mixture of Gaussians)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('data_distribution.png', dpi=150, bbox_inches='tight')
print("데이터 분포 시각화 저장 완료")
```

## 2. VAE의 수학: ELBO 완전 유도

### 2.1 잠재 변수 모델

**생성 과정**:
1. 잠재 변수 샘플링: $\mathbf{z} \sim p(\mathbf{z})$ (보통 $\mathcal{N}(\mathbf{0}, I)$)
2. 데이터 생성: $\mathbf{x} \sim p_\theta(\mathbf{x}|\mathbf{z})$

**목표**: 주변 우도(marginal likelihood) 최대화

$$\log p_\theta(\mathbf{x}) = \log \int p_\theta(\mathbf{x}|\mathbf{z}) p(\mathbf{z}) d\mathbf{z}$$

**문제**: 적분이 다루기 어려움 (intractable)

### 2.2 ELBO 유도 (방법 1: 옌센 부등식)

로그의 오목성과 옌센 부등식 이용:

$$\log p_\theta(\mathbf{x}) = \log \mathbb{E}_{q_\phi(\mathbf{z}|\mathbf{x})} \left[ \frac{p_\theta(\mathbf{x}, \mathbf{z})}{q_\phi(\mathbf{z}|\mathbf{x})} \right]$$

$$\geq \mathbb{E}_{q_\phi(\mathbf{z}|\mathbf{x})} \left[ \log \frac{p_\theta(\mathbf{x}, \mathbf{z})}{q_\phi(\mathbf{z}|\mathbf{x})} \right]$$

$$= \mathbb{E}_{q_\phi(\mathbf{z}|\mathbf{x})} \left[ \log p_\theta(\mathbf{x}|\mathbf{z}) + \log p(\mathbf{z}) - \log q_\phi(\mathbf{z}|\mathbf{x}) \right]$$

$$= \mathbb{E}_{q_\phi(\mathbf{z}|\mathbf{x})} \left[ \log p_\theta(\mathbf{x}|\mathbf{z}) \right] - D_{\text{KL}}(q_\phi(\mathbf{z}|\mathbf{x}) \| p(\mathbf{z}))$$

이것이 **ELBO** (Evidence Lower BOund):

$$\mathcal{L}(\theta, \phi; \mathbf{x}) = \mathbb{E}_{q_\phi(\mathbf{z}|\mathbf{x})} \left[ \log p_\theta(\mathbf{x}|\mathbf{z}) \right] - D_{\text{KL}}(q_\phi(\mathbf{z}|\mathbf{x}) \| p(\mathbf{z}))$$

### 2.3 ELBO 유도 (방법 2: KL 분해)

후방 분포 $p_\theta(\mathbf{z}|\mathbf{x})$와 근사 $q_\phi(\mathbf{z}|\mathbf{x})$의 KL 발산:

$$D_{\text{KL}}(q_\phi(\mathbf{z}|\mathbf{x}) \| p_\theta(\mathbf{z}|\mathbf{x})) = \mathbb{E}_{q_\phi} \left[ \log \frac{q_\phi(\mathbf{z}|\mathbf{x})}{p_\theta(\mathbf{z}|\mathbf{x})} \right]$$

베이즈 정리: $p_\theta(\mathbf{z}|\mathbf{x}) = \frac{p_\theta(\mathbf{x}|\mathbf{z})p(\mathbf{z})}{p_\theta(\mathbf{x})}$

$$D_{\text{KL}}(q_\phi \| p_\theta) = \mathbb{E}_{q_\phi} \left[ \log q_\phi(\mathbf{z}|\mathbf{x}) - \log p_\theta(\mathbf{x}|\mathbf{z}) - \log p(\mathbf{z}) + \log p_\theta(\mathbf{x}) \right]$$

정리하면:

$$\log p_\theta(\mathbf{x}) = D_{\text{KL}}(q_\phi \| p_\theta) + \mathcal{L}(\theta, \phi; \mathbf{x})$$

$D_{\text{KL}} \geq 0$이므로 $\mathcal{L}$은 $\log p_\theta(\mathbf{x})$의 하한입니다.

### 2.4 ELBO의 두 항 해석

$$\mathcal{L} = \underbrace{\mathbb{E}_{q_\phi(\mathbf{z}|\mathbf{x})} \left[ \log p_\theta(\mathbf{x}|\mathbf{z}) \right]}_{\text{재구성 항}} - \underbrace{D_{\text{KL}}(q_\phi(\mathbf{z}|\mathbf{x}) \| p(\mathbf{z}))}_{\text{정규화 항}}$$

**재구성 항**:
- 잠재 표현 $\mathbf{z}$에서 $\mathbf{x}$를 얼마나 잘 복원하는가
- 디코더 $p_\theta(\mathbf{x}|\mathbf{z})$의 품질

**정규화 항**:
- 인코더 분포를 사전 분포에 가깝게 유지
- 잠재 공간의 구조화
- 과적합 방지

### 2.5 재매개변수화 트릭 (Reparameterization Trick)

**문제**: $\nabla_\phi \mathbb{E}_{q_\phi(\mathbf{z}|\mathbf{x})} [f(\mathbf{z})]$를 어떻게 계산?

**해결**: $\mathbf{z}$를 결정론적 함수 + 노이즈로 표현

가우시안의 경우: $q_\phi(\mathbf{z}|\mathbf{x}) = \mathcal{N}(\boldsymbol{\mu}_\phi(\mathbf{x}), \boldsymbol{\sigma}_\phi^2(\mathbf{x}))$

$$\mathbf{z} = \boldsymbol{\mu}_\phi(\mathbf{x}) + \boldsymbol{\sigma}_\phi(\mathbf{x}) \odot \boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, I)$$

이제 $\mathbf{z}$가 $\phi$에 대해 미분 가능합니다.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, input_dim=784, latent_dim=20):
        super().__init__()

        # 인코더
        self.fc1 = nn.Linear(input_dim, 400)
        self.fc_mu = nn.Linear(400, latent_dim)
        self.fc_logvar = nn.Linear(400, latent_dim)

        # 디코더
        self.fc3 = nn.Linear(latent_dim, 400)
        self.fc4 = nn.Linear(400, input_dim)

    def encode(self, x):
        """인코더: x -> (mu, logvar)"""
        h = F.relu(self.fc1(x))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """재매개변수화: z = mu + sigma * epsilon"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + std * eps
        return z

    def decode(self, z):
        """디코더: z -> x_recon"""
        h = F.relu(self.fc3(z))
        x_recon = torch.sigmoid(self.fc4(h))
        return x_recon

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar

def vae_loss(x, x_recon, mu, logvar):
    """
    VAE 손실 함수

    Parameters:
    -----------
    x : Tensor
        원본 입력
    x_recon : Tensor
        재구성된 출력
    mu, logvar : Tensor
        잠재 분포의 평균과 로그 분산

    Returns:
    --------
    loss : Tensor
        총 손실 (재구성 + KL)
    """
    # 재구성 손실 (Bernoulli 분포 가정)
    recon_loss = F.binary_cross_entropy(x_recon, x, reduction='sum')

    # KL 발산 (가우시안 간의 닫힌 형태)
    # KL(N(mu, sigma^2) || N(0, 1)) = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return recon_loss + kl_loss, recon_loss, kl_loss

# 예제: 간단한 2D 잠재 공간
vae_model = VAE(input_dim=2, latent_dim=2)
x_sample = torch.randn(10, 2)

x_recon, mu, logvar = vae_model(x_sample)
total_loss, recon, kl = vae_loss(x_sample, x_recon, mu, logvar)

print(f"총 손실: {total_loss.item():.4f}")
print(f"재구성 손실: {recon.item():.4f}")
print(f"KL 손실: {kl.item():.4f}")
```

### 2.6 가우시안 간 KL 발산의 닫힌 형태

$q = \mathcal{N}(\boldsymbol{\mu}, \text{diag}(\boldsymbol{\sigma}^2))$, $p = \mathcal{N}(\mathbf{0}, I)$일 때:

$$D_{\text{KL}}(q \| p) = \frac{1}{2} \sum_{i=1}^{d} \left( \sigma_i^2 + \mu_i^2 - 1 - \log \sigma_i^2 \right)$$

**유도**: KL 발산의 정의와 가우시안의 엔트로피 공식 사용

## 3. GAN의 수학

### 3.1 미니맥스 게임

**생성자** (Generator) $G$: $\mathbf{z} \sim p(\mathbf{z}) \mapsto G(\mathbf{z}) \approx p_{\text{data}}$

**판별자** (Discriminator) $D$: $\mathbf{x} \mapsto D(\mathbf{x}) \in [0, 1]$ (진짜 확률)

**목적 함수**:

$$\min_G \max_D \mathbb{E}_{\mathbf{x} \sim p_{\text{data}}} [\log D(\mathbf{x})] + \mathbb{E}_{\mathbf{z} \sim p(\mathbf{z})} [\log(1 - D(G(\mathbf{z})))]$$

**해석**:
- $D$는 진짜 데이터에 높은 확률, 가짜에 낮은 확률을 부여하려 함
- $G$는 $D$를 속이려 함 (가짜가 진짜처럼 보이게)

### 3.2 최적 판별자

$G$가 고정되었을 때, 최적 판별자는:

$$D^*(x) = \frac{p_{\text{data}}(\mathbf{x})}{p_{\text{data}}(\mathbf{x}) + p_g(\mathbf{x})}$$

여기서 $p_g$는 생성자가 유도하는 분포입니다.

**증명**: 목적 함수를 $D(\mathbf{x})$에 대해 미분하고 0으로 설정

$$\frac{\partial}{\partial D(\mathbf{x})} \left[ p_{\text{data}}(\mathbf{x}) \log D(\mathbf{x}) + p_g(\mathbf{x}) \log(1 - D(\mathbf{x})) \right] = 0$$

$$\frac{p_{\text{data}}(\mathbf{x})}{D(\mathbf{x})} - \frac{p_g(\mathbf{x})}{1 - D(\mathbf{x})} = 0$$

### 3.3 JS 발산과의 관계

최적 판별자 $D^*$를 대입하면, 생성자의 목적 함수는:

$$C(G) = -\log 4 + 2 \cdot \text{JS}(p_{\text{data}} \| p_g)$$

여기서 **Jensen-Shannon 발산**:

$$\text{JS}(p \| q) = \frac{1}{2} D_{\text{KL}}\left(p \middle\| \frac{p+q}{2}\right) + \frac{1}{2} D_{\text{KL}}\left(q \middle\| \frac{p+q}{2}\right)$$

**결론**: GAN을 학습하는 것은 JS 발산을 최소화하는 것과 같습니다.

### 3.4 학습 불안정성의 수학적 원인

**그래디언트 소실**:
- $p_g$와 $p_{\text{data}}$가 겹치지 않으면 $D$가 완벽해짐
- $\nabla_\theta \log(1 - D(G(\mathbf{z}))) \approx 0$

**모드 붕괴** (Mode Collapse):
- 생성자가 일부 모드만 생성 ($p_g$가 $p_{\text{data}}$의 일부만 커버)
- JS 발산은 여전히 작을 수 있음

```python
# 간단한 GAN 예제 (1D)
class Generator(nn.Module):
    def __init__(self, latent_dim=10, output_dim=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 50),
            nn.ReLU(),
            nn.Linear(50, output_dim)
        )

    def forward(self, z):
        return self.net(z)

class Discriminator(nn.Module):
    def __init__(self, input_dim=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 50),
            nn.LeakyReLU(0.2),
            nn.Linear(50, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

# 모델 초기화
latent_dim = 10
G = Generator(latent_dim=latent_dim, output_dim=1)
D = Discriminator(input_dim=1)

# 손실 함수
criterion = nn.BCELoss()

# 옵티마이저
optimizer_G = torch.optim.Adam(G.parameters(), lr=2e-4)
optimizer_D = torch.optim.Adam(D.parameters(), lr=2e-4)

# 간단한 학습 루프
def train_gan_step(real_data, G, D, optimizer_G, optimizer_D, latent_dim):
    """GAN 1 스텝 학습"""
    batch_size = real_data.size(0)

    # 레이블
    real_labels = torch.ones(batch_size, 1)
    fake_labels = torch.zeros(batch_size, 1)

    # 판별자 학습
    optimizer_D.zero_grad()

    # 진짜 데이터
    D_real = D(real_data)
    loss_D_real = criterion(D_real, real_labels)

    # 가짜 데이터
    z = torch.randn(batch_size, latent_dim)
    fake_data = G(z)
    D_fake = D(fake_data.detach())
    loss_D_fake = criterion(D_fake, fake_labels)

    loss_D = loss_D_real + loss_D_fake
    loss_D.backward()
    optimizer_D.step()

    # 생성자 학습
    optimizer_G.zero_grad()

    z = torch.randn(batch_size, latent_dim)
    fake_data = G(z)
    D_fake = D(fake_data)
    loss_G = criterion(D_fake, real_labels)  # 진짜로 분류되길 원함

    loss_G.backward()
    optimizer_G.step()

    return loss_D.item(), loss_G.item()

# 예제 데이터
real_samples = torch.FloatTensor(generate_data(100)).unsqueeze(1)
loss_d, loss_g = train_gan_step(real_samples, G, D, optimizer_G, optimizer_D, latent_dim)

print(f"판별자 손실: {loss_d:.4f}")
print(f"생성자 손실: {loss_g:.4f}")
```

## 4. Wasserstein 거리와 WGAN

### 4.1 최적 수송 (Optimal Transport)

두 확률 분포 $P, Q$ 간의 **Wasserstein-1 거리** (Earth Mover's Distance):

$$W_1(P, Q) = \inf_{\gamma \in \Pi(P, Q)} \mathbb{E}_{(\mathbf{x}, \mathbf{y}) \sim \gamma} \left[ \|\mathbf{x} - \mathbf{y}\| \right]$$

여기서 $\Pi(P, Q)$는 $P$와 $Q$를 주변 분포로 하는 모든 결합 분포입니다.

**직관**: $P$를 $Q$로 "운반"하는 최소 비용

### 4.2 칸토로비치-루빈슈타인 쌍대성

**쌍대 문제**:

$$W_1(P, Q) = \sup_{f: \|f\|_L \leq 1} \mathbb{E}_{\mathbf{x} \sim P}[f(\mathbf{x})] - \mathbb{E}_{\mathbf{y} \sim Q}[f(\mathbf{y})]$$

여기서 $\|f\|_L \leq 1$는 **1-립시츠 함수** 제약:

$$|f(\mathbf{x}_1) - f(\mathbf{x}_2)| \leq \|\mathbf{x}_1 - \mathbf{x}_2\|$$

### 4.3 WGAN의 목적 함수

**판별자를 critic $f_w$로 대체** (출력이 확률이 아님):

$$\min_G \max_{w: f_w \text{ is 1-Lipschitz}} \mathbb{E}_{\mathbf{x} \sim p_{\text{data}}}[f_w(\mathbf{x})] - \mathbb{E}_{\mathbf{z} \sim p(\mathbf{z})}[f_w(G(\mathbf{z}))]$$

**장점**:
- 학습 안정성 향상
- 의미 있는 손실 값 (거리 근사)
- 모드 붕괴 완화

### 4.4 립시츠 제약: 그래디언트 패널티

**원래 WGAN**: 가중치 클리핑 (weight clipping) → 제한적

**WGAN-GP**: 그래디언트 패널티

$$\mathcal{L} = \mathbb{E}_{\tilde{\mathbf{x}}} [f(\tilde{\mathbf{x}})] - \mathbb{E}_{\mathbf{x}} [f(\mathbf{x})] + \lambda \mathbb{E}_{\hat{\mathbf{x}}} \left[ (\|\nabla_{\hat{\mathbf{x}}} f(\hat{\mathbf{x}})\| - 1)^2 \right]$$

여기서 $\hat{\mathbf{x}} = \epsilon \mathbf{x} + (1 - \epsilon) \tilde{\mathbf{x}}$는 진짜와 가짜 사이의 보간입니다.

```python
def gradient_penalty(D, real_data, fake_data, device='cpu'):
    """
    그래디언트 패널티 계산 (WGAN-GP)

    Parameters:
    -----------
    D : nn.Module
        판별자 (critic)
    real_data : Tensor
        진짜 데이터
    fake_data : Tensor
        가짜 데이터
    device : str
        디바이스

    Returns:
    --------
    gp : Tensor
        그래디언트 패널티
    """
    batch_size = real_data.size(0)

    # 랜덤 보간
    epsilon = torch.rand(batch_size, 1).to(device)
    interpolates = epsilon * real_data + (1 - epsilon) * fake_data
    interpolates.requires_grad_(True)

    # 판별자 출력
    D_interpolates = D(interpolates)

    # 그래디언트 계산
    gradients = torch.autograd.grad(
        outputs=D_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(D_interpolates),
        create_graph=True,
        retain_graph=True
    )[0]

    # 그래디언트 norm
    gradients = gradients.view(batch_size, -1)
    gradient_norm = gradients.norm(2, dim=1)

    # 패널티
    gp = ((gradient_norm - 1) ** 2).mean()

    return gp

# 예제
real = torch.randn(32, 1, requires_grad=False)
fake = torch.randn(32, 1, requires_grad=False)

D_critic = Discriminator(input_dim=1)
gp = gradient_penalty(D_critic, real, fake)

print(f"그래디언트 패널티: {gp.item():.4f}")
```

## 5. 확산 모델의 수학 (Diffusion Models)

### 5.1 정방향 과정 (Forward Process)

데이터 $\mathbf{x}_0 \sim q(\mathbf{x}_0)$에 점진적으로 가우시안 노이즈 추가:

$$q(\mathbf{x}_t | \mathbf{x}_{t-1}) = \mathcal{N}(\mathbf{x}_t; \sqrt{1 - \beta_t} \mathbf{x}_{t-1}, \beta_t I)$$

**노이즈 스케줄**: $\beta_1, \ldots, \beta_T$ (보통 $10^{-4} \to 0.02$)

**닫힌 형태**: $\alpha_t = 1 - \beta_t$, $\bar{\alpha}_t = \prod_{s=1}^t \alpha_s$

$$q(\mathbf{x}_t | \mathbf{x}_0) = \mathcal{N}(\mathbf{x}_t; \sqrt{\bar{\alpha}_t} \mathbf{x}_0, (1 - \bar{\alpha}_t) I)$$

즉, 한 번에 $\mathbf{x}_t = \sqrt{\bar{\alpha}_t} \mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t} \boldsymbol{\epsilon}$, $\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, I)$

### 5.2 역방향 과정 (Reverse Process)

**목표**: $\mathbf{x}_T \sim \mathcal{N}(\mathbf{0}, I)$에서 $\mathbf{x}_0$로 복원

$$p_\theta(\mathbf{x}_{t-1} | \mathbf{x}_t) = \mathcal{N}(\mathbf{x}_{t-1}; \boldsymbol{\mu}_\theta(\mathbf{x}_t, t), \Sigma_\theta(\mathbf{x}_t, t))$$

**이론**: $\beta_t$가 충분히 작으면, 역방향도 가우시안입니다.

### 5.3 DDPM 손실 유도

변분 하한(ELBO)을 유도하면 (VAE와 유사):

$$\mathcal{L} = \mathbb{E}_q \left[ D_{\text{KL}}(q(\mathbf{x}_T|\mathbf{x}_0) \| p(\mathbf{x}_T)) + \sum_{t=2}^T D_{\text{KL}}(q(\mathbf{x}_{t-1}|\mathbf{x}_t, \mathbf{x}_0) \| p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t)) - \log p_\theta(\mathbf{x}_0|\mathbf{x}_1) \right]$$

**핵심**: 후방 분포 $q(\mathbf{x}_{t-1}|\mathbf{x}_t, \mathbf{x}_0)$의 닫힌 형태 존재

$$q(\mathbf{x}_{t-1}|\mathbf{x}_t, \mathbf{x}_0) = \mathcal{N}(\mathbf{x}_{t-1}; \tilde{\boldsymbol{\mu}}_t(\mathbf{x}_t, \mathbf{x}_0), \tilde{\beta}_t I)$$

여기서:

$$\tilde{\boldsymbol{\mu}}_t(\mathbf{x}_t, \mathbf{x}_0) = \frac{\sqrt{\bar{\alpha}_{t-1}} \beta_t}{1 - \bar{\alpha}_t} \mathbf{x}_0 + \frac{\sqrt{\alpha_t}(1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t} \mathbf{x}_t$$

**노이즈 예측으로 재매개변수화**:

$\mathbf{x}_0 = \frac{1}{\sqrt{\bar{\alpha}_t}}(\mathbf{x}_t - \sqrt{1 - \bar{\alpha}_t} \boldsymbol{\epsilon})$를 대입하면,

**간단한 손실**:

$$\mathcal{L}_{\text{simple}} = \mathbb{E}_{t, \mathbf{x}_0, \boldsymbol{\epsilon}} \left[ \|\boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)\|^2 \right]$$

즉, **노이즈 예측** 문제로 귀결됩니다!

```python
def linear_beta_schedule(timesteps, beta_start=1e-4, beta_end=0.02):
    """선형 노이즈 스케줄"""
    return np.linspace(beta_start, beta_end, timesteps)

def get_diffusion_parameters(timesteps=1000):
    """확산 모델 파라미터 계산"""
    betas = linear_beta_schedule(timesteps)
    alphas = 1.0 - betas
    alphas_cumprod = np.cumprod(alphas)
    alphas_cumprod_prev = np.concatenate([[1.0], alphas_cumprod[:-1]])

    return {
        'betas': betas,
        'alphas': alphas,
        'alphas_cumprod': alphas_cumprod,
        'alphas_cumprod_prev': alphas_cumprod_prev,
        'sqrt_alphas_cumprod': np.sqrt(alphas_cumprod),
        'sqrt_one_minus_alphas_cumprod': np.sqrt(1.0 - alphas_cumprod)
    }

params = get_diffusion_parameters(timesteps=1000)

# 노이즈 추가 예제
def q_sample(x_0, t, params, noise=None):
    """
    정방향 과정: x_0에서 x_t 샘플링

    Parameters:
    -----------
    x_0 : Tensor
        원본 데이터
    t : int
        타임스텝
    params : dict
        확산 파라미터
    noise : Tensor, optional
        노이즈 (None이면 샘플링)

    Returns:
    --------
    x_t : Tensor
        노이즈가 추가된 데이터
    """
    if noise is None:
        noise = torch.randn_like(x_0)

    sqrt_alpha_cumprod_t = params['sqrt_alphas_cumprod'][t]
    sqrt_one_minus_alpha_cumprod_t = params['sqrt_one_minus_alphas_cumprod'][t]

    x_t = sqrt_alpha_cumprod_t * x_0 + sqrt_one_minus_alpha_cumprod_t * noise

    return x_t

# 예제: 1D 데이터에 노이즈 추가
x_0 = torch.FloatTensor([[1.0], [2.0], [3.0]])
timesteps_to_visualize = [0, 250, 500, 750, 999]

fig, axes = plt.subplots(1, len(timesteps_to_visualize), figsize=(15, 3))

for idx, t in enumerate(timesteps_to_visualize):
    x_t = q_sample(x_0, t, params)
    axes[idx].hist(x_t.numpy(), bins=20, alpha=0.7)
    axes[idx].set_title(f't={t}')
    axes[idx].set_xlabel('Value')
    axes[idx].set_xlim(-4, 4)

plt.tight_layout()
plt.savefig('diffusion_forward_process.png', dpi=150, bbox_inches='tight')
print("확산 정방향 과정 시각화 저장 완료")
```

### 5.4 스코어 매칭과의 관계

**스코어 함수**: $\nabla_{\mathbf{x}} \log p(\mathbf{x})$

**스코어 매칭**: 스코어 함수를 신경망으로 근사

$$\mathcal{L}_{\text{score}} = \mathbb{E}_{p(\mathbf{x})} \left[ \left\| \nabla_{\mathbf{x}} \log p(\mathbf{x}) - s_\theta(\mathbf{x}) \right\|^2 \right]$$

**연결**: 노이즈 예측 $\boldsymbol{\epsilon}_\theta$와 스코어 추정의 관계

$$\nabla_{\mathbf{x}_t} \log q(\mathbf{x}_t|\mathbf{x}_0) = -\frac{\boldsymbol{\epsilon}}{\sqrt{1 - \bar{\alpha}_t}}$$

따라서 $\boldsymbol{\epsilon}_\theta$를 학습하는 것은 스코어 함수를 학습하는 것과 같습니다.

### 5.5 랑주뱅 역학 (Langevin Dynamics)

**샘플링 방법**: 스코어 함수를 이용한 마르코프 체인 몬테카를로

$$\mathbf{x}_{t+1} = \mathbf{x}_t + \epsilon \nabla_{\mathbf{x}} \log p(\mathbf{x}_t) + \sqrt{2\epsilon} \mathbf{z}_t$$

여기서 $\mathbf{z}_t \sim \mathcal{N}(\mathbf{0}, I)$

확산 모델의 역방향 과정은 **이산화된 랑주뱅 역학**으로 볼 수 있습니다.

## 6. 최신 발전: Flow Matching과 CNF

### 6.1 연속 정규화 흐름 (Continuous Normalizing Flow)

**아이디어**: ODE로 데이터 분포 변환

$$\frac{d\mathbf{x}(t)}{dt} = f_\theta(\mathbf{x}(t), t)$$

**시간 $t=0$**: $\mathbf{x}(0) \sim p_0$ (노이즈)
**시간 $t=1$**: $\mathbf{x}(1) \sim p_1$ (데이터)

**Neural ODE**: 신경망으로 $f_\theta$ 매개변수화

### 6.2 Flow Matching

**목표**: 벡터 필드 $u_t(\mathbf{x})$를 직접 학습

**조건부 흐름 매칭** (Conditional Flow Matching):

$$\mathcal{L}_{\text{CFM}} = \mathbb{E}_{t, \mathbf{x}_0, \mathbf{x}_1} \left[ \left\| u_t(\mathbf{x}_t) - \frac{d\mathbf{x}_t}{dt} \right\|^2 \right]$$

여기서 $\mathbf{x}_t = (1-t)\mathbf{x}_0 + t\mathbf{x}_1$는 선형 보간입니다.

**장점**:
- 시뮬레이션 없는 학습 (simulation-free training)
- 확산 모델보다 빠른 샘플링
- 안정적인 학습

### 6.3 일관성 모델 (Consistency Models)

**아이디어**: ODE 궤적 위의 모든 점을 같은 점으로 매핑

$$f_\theta(\mathbf{x}_t, t) = f_\theta(\mathbf{x}_{t'}, t') \quad \forall t, t'$$

**일관성 손실**:

$$\mathcal{L} = \mathbb{E} \left[ d(f_\theta(\mathbf{x}_{t_{n+1}}, t_{n+1}), f_{\theta^-}(\mathbf{x}_{t_n}, t_n)) \right]$$

여기서 $\theta^-$는 EMA(exponential moving average) 파라미터입니다.

**장점**: 1-스텝 생성 가능 (확산 모델은 수백 스텝 필요)

```python
# 간단한 Flow Matching 예제
class FlowMatchingModel(nn.Module):
    def __init__(self, dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim + 1, 64),  # x + t
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, dim)
        )

    def forward(self, x, t):
        """벡터 필드 예측"""
        t_expanded = t.view(-1, 1).expand(-1, x.size(1))
        xt = torch.cat([x, t_expanded[:, :1]], dim=1)
        return self.net(xt)

def flow_matching_loss(model, x0, x1):
    """
    Flow Matching 손실

    Parameters:
    -----------
    model : nn.Module
        벡터 필드 모델
    x0 : Tensor
        시작 분포 샘플 (노이즈)
    x1 : Tensor
        목표 분포 샘플 (데이터)

    Returns:
    --------
    loss : Tensor
        Flow Matching 손실
    """
    batch_size = x0.size(0)

    # 랜덤 시간
    t = torch.rand(batch_size, 1)

    # 선형 보간
    x_t = (1 - t) * x0 + t * x1

    # 진짜 벡터 필드 (선형 보간의 도함수)
    true_velocity = x1 - x0

    # 예측 벡터 필드
    pred_velocity = model(x_t, t.squeeze())

    # 손실
    loss = F.mse_loss(pred_velocity, true_velocity)

    return loss

# 예제
fm_model = FlowMatchingModel(dim=2)
x0_samples = torch.randn(32, 2)
x1_samples = torch.randn(32, 2) + torch.tensor([2.0, 0.0])

loss = flow_matching_loss(fm_model, x0_samples, x1_samples)
print(f"Flow Matching 손실: {loss.item():.4f}")
```

## 연습 문제

### 문제 1: ELBO 유도의 두 방법
1. 옌센 부등식을 이용한 ELBO 유도를 단계별로 작성하시오.
2. KL 분해를 이용한 유도를 단계별로 작성하시오.
3. 두 방법이 동일한 결과를 주는 이유를 설명하시오.

### 문제 2: VAE 구현과 실험
2D 가우시안 혼합 데이터에 대해:
1. 2D 잠재 공간을 가진 VAE를 구현하시오.
2. 재구성 손실과 KL 손실의 학습 곡선을 플로팅하시오.
3. 잠재 공간을 시각화하고, 그리드 샘플링으로 디코더의 출력 시각화하시오.
4. $\beta$-VAE를 구현하고 (KL 항에 가중치 $\beta$), $\beta$의 영향 분석하시오.

### 문제 3: GAN의 최적 판별자 증명
$G$가 고정되었을 때, 판별자의 목적 함수:

$$\max_D \mathbb{E}_{\mathbf{x} \sim p_{\text{data}}} [\log D(\mathbf{x})] + \mathbb{E}_{\mathbf{x} \sim p_g} [\log(1 - D(\mathbf{x}))]$$

가 $D^*(\mathbf{x}) = \frac{p_{\text{data}}(\mathbf{x})}{p_{\text{data}}(\mathbf{x}) + p_g(\mathbf{x})}$에서 최댓값을 가짐을 증명하시오.

### 문제 4: 확산 모델의 정방향 과정
1. $q(\mathbf{x}_t|\mathbf{x}_0) = \mathcal{N}(\sqrt{\bar{\alpha}_t}\mathbf{x}_0, (1-\bar{\alpha}_t)I)$를 유도하시오 (재귀적으로).
2. 1D 데이터에 대해 다양한 $t$에서 $\mathbf{x}_t$의 분포를 시각화하시오.
3. $T \to \infty$일 때 $\mathbf{x}_T \approx \mathcal{N}(\mathbf{0}, I)$임을 보이시오.

### 문제 5: Flow Matching vs 확산 모델
1. 간단한 Flow Matching 모델과 DDPM을 구현하시오.
2. 동일한 2D 데이터에 대해 두 모델을 학습시키시오.
3. 샘플링 속도 (스텝 수) 비교하시오.
4. 생성 품질 (FID, Inception Score 등) 비교하시오.

## 참고 자료

### 논문
- **VAE**: Kingma, D. P., & Welling, M. (2014). "Auto-Encoding Variational Bayes." *ICLR*.
- **GAN**: Goodfellow, I., et al. (2014). "Generative Adversarial Nets." *NeurIPS*.
- **WGAN**: Arjovsky, M., et al. (2017). "Wasserstein Generative Adversarial Networks." *ICML*.
- **WGAN-GP**: Gulrajani, I., et al. (2017). "Improved Training of Wasserstein GANs." *NeurIPS*.
- **DDPM**: Ho, J., et al. (2020). "Denoising Diffusion Probabilistic Models." *NeurIPS*.
- **Score-Based Models**: Song, Y., & Ermon, S. (2019). "Generative Modeling by Estimating Gradients of the Data Distribution." *NeurIPS*.
- **Flow Matching**: Lipman, Y., et al. (2023). "Flow Matching for Generative Modeling." *ICLR*.
- **Consistency Models**: Song, Y., et al. (2023). "Consistency Models." *ICML*.

### 온라인 자료
- [Lil'Log: From Autoencoder to Beta-VAE (Lilian Weng)](https://lilianweng.github.io/posts/2018-08-12-vae/)
- [Lil'Log: What are Diffusion Models? (Lilian Weng)](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/)
- [Understanding Diffusion Models (Calvin Luo)](https://arxiv.org/abs/2208.11970)
- [The Annotated Diffusion Model](https://huggingface.co/blog/annotated-diffusion)

### 라이브러리
- `torch`: PyTorch 구현
- `diffusers` (Hugging Face): 확산 모델 라이브러리
- `stable-diffusion`: Stable Diffusion 구현
