[이전: 생성 모델 - VAE](./30_Generative_Models_VAE.md) | [다음: 확산 모델](./32_Diffusion_Models.md)

---

# 31. Variational Autoencoder (VAE)

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. VAE의 생성 모델 목표를 설명하고, 주변 우도(Marginal Likelihood) p(x)의 직접적인 최대화가 왜 다루기 어려운지(Intractable) 서술합니다.
2. 변분 추론(Variational Inference) 프레임워크로부터 증거 하한(Evidence Lower BOund, ELBO)을 유도하고, 그 두 구성 요소(재구성 손실과 KL 발산 정규화)를 해석합니다.
3. 재파라미터화 트릭(Reparameterization Trick)을 설명하고, 확률적 샘플링 단계를 통한 역전파 기울기에 왜 필요한지 설명합니다.
4. 인코더(사후 확률 네트워크), 디코더(우도 네트워크), ELBO 손실 계산을 포함하여 PyTorch에서 VAE를 처음부터 구현합니다.
5. 잠재 공간 보간(Latent Space Interpolation)과 산술 연산을 수행하여 VAE가 매끄럽고 구조화된 잠재 다양체(Latent Manifold)를 학습함을 시연합니다.
6. 학습 안정성, 출력 선명도, 잠재 공간 해석 가능성 측면에서 VAE와 GAN을 비교하고, 각각이 탁월한 시나리오를 식별합니다.

---

## 이론과 원리

이 구현 레슨은 이전 레슨의 VAE 수학을 구체적 텐서 연산에 고정합니다. 세 가지 디테일이 일관되게 첫 구현자를 잡습니다: `log_var` 매개변수화(그리고 왜 분산이 아닌 로그 분산을 예측하는지), KL 닫힌 형태 항과 그 부호, 그리고 둘 다 하나의 스칼라 손실로 합쳐질 때 재구성 vs KL의 상대 가중.

이 섹션에서 다루는 내용:

- **A.** 왜 `\sigma` 대신 `log \sigma^2` 예측
- **B.** KL 닫힌 형태, 항별로
- **C.** 재구성 손실 선택: BCE vs MSE
- **D.** 잠재 공간 산술과 그것이 작동하는 이유

### A. `var` 아니고 `log_var`인 이유

Encoder는 양의 분산을 출력해야 합니다. 이를 보장하는 두 방법:

1. **`\sigma`를 직접 출력하고 양수성을 위해 `softplus`나 `exp` 적용.**
2. **`log \sigma^2`를 출력하고 샘플링 시 지수화.**

옵션 2가 보편적으로 선호되는데:

- `log \sigma^2`은 제약이 없음 — 네트워크가 어떤 활성화 함수 없이도 음수(작은 분산용)나 큰 양수(높은 분산용)를 포함한 임의의 실수 값을 만들 수 있음.
- 필요할 때 `exp(0.5 * log_var)`이 `\sigma`를 깔끔히 줌.
- KL 공식이 `log \sigma^2`을 직접 가지므로 추가 로그 계산 없음.

수치 안정성도 `log_var`을 선호: 작은 분산(`\sigma \approx 10^{-3}`)은 언더플로우 위험이 있는 직접 float보다 `log_var \approx -7`로 표현하기 더 쉽습니다.

### B. KL 닫힌 형태, 항별로

`d`-차원 가우시안 `q = N(\mu, diag(\sigma^2))`와 사전 `p = N(0, I)`의 경우:

```
KL(q || p) = 0.5 * sum_{j=1}^{d} [\mu_j^2 + \sigma_j^2 - log \sigma_j^2 - 1]
```

각 항이 의미를 가짐:

- `\mu_j^2`: 0(사전의 평균)에서 먼 사후 평균을 벌함.
- `\sigma_j^2`: 1(사전의 분산)보다 큰 사후 분산을 벌함.
- `-log \sigma_j^2`: 1보다 *작은* 분산을 벌함(`\sigma \to 0`에 따라 로그가 `+inf`로).
- `-1`: `q = p`일 때 정확히 값이 0이 되게 하는 상수.

항 `\sigma^2 - log \sigma^2 - 1`은 `\sigma > 0`에 대해 비음수이며 `\sigma = 1`에서 최솟값 0. 분산을 1 쪽으로 밂.

코드: `kl = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp()).sum(dim=-1).mean()`. 부호 관례: 총 손실 = `-ELBO`를 위해 KL이 재구성 손실에 *더해짐*.

### C. 재구성 손실: BCE vs MSE

두 흔한 선택:

- **픽셀당 BCE** (픽셀 강도를 베르누이 확률로 취급, 공간 차원에 걸쳐 합): 원본 VAE 논문이 이진화된 MNIST에 사용. 날카로운 경계; `[0, 1]` 강도 이미지에 작동.
- **픽셀당 MSE** (출력을 진실 픽셀의 가우시안 잡음 관찰로 취급): 컬러 이미지의 표준. 약간 더 흐림; 가우시안 관찰 잡음 하의 최대 우도에 대응.

결정적으로: ELBO의 재구성 항은 `-log p(x | z)`. BCE의 경우 이는 `BCELoss(x_hat, x)`; 고정 분산 `\sigma^2 = 0.5`의 가우시안 관찰의 경우 `0.5 * MSE`. 인자가 KL과 균형 맞출 때 중요.

### D. 잠재 공간 산술

VAE 잠재 공간은 의미 있는 산술 — `z(king) - z(man) + z(woman) ≈ z(queen)` 스타일 연산 — 과 부드러운 보간을 지원. 왜 이것이 작동할까요?

두 구조적 성질:

1. **연속성**: `z`의 작은 변화가 `decoder(z)`의 작은 변화를 만듦. KL 항이 encoder가 다른 예제를 격렬히 다른 `z` 값에 두지 못하게 함으로써 이를 강제.
2. **위상적 덮음**: 사전 `N(0, I)`이 잠재 공간 전반에 확률 질량을 밀집하게 두고, encoder는 이를 존중하는 사후를 만들도록 학습. 두 점 사이의 선형 보간이 사전의 지지 내에 머묾.

GAN 잠재 공간도 산술을 지원하지만 덜 신뢰 가능 — 연속성을 강제하는 KL 항이 없으므로, 보간된 잠재가 생성기가 합리적 출력을 만들도록 학습되지 않은 "공백" 영역을 통과할 수 있음.

### 이론에서 아래 코드로

| 이론 개념 | 본 레슨의 코드 구성 |
|-----------|---------------------|
| Encoder 출력 | `mu, log_var = self.encoder(x).split(latent_dim, dim=-1)` |
| 재매개변수화 | `z = mu + torch.exp(0.5 * log_var) * torch.randn_like(mu)` |
| KL 닫힌 형태 | `-0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())` |
| 재구성 | `F.binary_cross_entropy(x_hat, x, reduction='sum')` |
| 잠재 산술 | `z_new = z_a - z_b + z_c; out = decoder(z_new)` |

---


## 개요

Variational Autoencoder (VAE)는 생성 모델의 기초가 되는 아키텍처로, 데이터의 잠재 표현(latent representation)을 학습하고 새로운 샘플을 생성할 수 있습니다. "Auto-Encoding Variational Bayes" (Kingma & Welling, 2013)

---

## 수학적 배경

### 1. 생성 모델 목표

```
목표: p(x) 모델링
- x: 관측 데이터 (이미지 등)
- z: 잠재 변수 (latent variable)

생성 과정:
z ~ p(z)         # Prior (보통 N(0, I))
x ~ p(x|z)       # Decoder/Generator

문제: p(x) = ∫ p(x|z)p(z)dz 는 계산 불가능 (intractable)
```

### 2. Variational Inference

```
사후 분포 p(z|x)도 계산 불가능
→ 근사 분포 q(z|x)를 학습 (Encoder)

ELBO (Evidence Lower BOund):
log p(x) ≥ E_q[log p(x|z)] - KL(q(z|x) || p(z))
         ────────────────   ─────────────────────
         Reconstruction     Regularization
         Loss               (Prior matching)

최대화할 목표:
L(θ, φ; x) = E_q_φ(z|x)[log p_θ(x|z)] - KL(q_φ(z|x) || p(z))
```

### 3. Reparameterization Trick

```
문제: z ~ q(z|x) = N(μ, σ²) 에서 샘플링은 미분 불가

해결: Reparameterization
ε ~ N(0, I)
z = μ + σ ⊙ ε

이제 그래디언트가 μ, σ를 통해 역전파 가능!

┌─────────────────────────────────────────┐
│  Encoder                                │
│  x → [μ, log σ²]                        │
│                                         │
│  Reparameterization                     │
│  ε ~ N(0, I)                           │
│  z = μ + σ ⊙ ε                         │
│                                         │
│  Decoder                                │
│  z → x̂                                  │
└─────────────────────────────────────────┘
```

### 4. 손실 함수

```
L = L_recon + β * L_KL

Reconstruction Loss (이미지):
- Binary: BCE(x, x̂) = -Σ[x·log(x̂) + (1-x)·log(1-x̂)]
- Continuous: MSE(x, x̂) = ||x - x̂||²

KL Divergence (Gaussian prior):
KL(N(μ, σ²) || N(0, 1)) = -½ Σ(1 + log σ² - μ² - σ²)

β-VAE:
β > 1: 더 강한 disentanglement
β < 1: 더 나은 reconstruction
```

---

## VAE 아키텍처

### 표준 VAE (MNIST)

```
Encoder:
Input (28×28×1)
    ↓
Conv2d(1→32, k=3, s=2, p=1)  → (14×14×32)
    ↓ ReLU
Conv2d(32→64, k=3, s=2, p=1) → (7×7×64)
    ↓ ReLU
Flatten → (7×7×64 = 3136)
    ↓
Linear(3136→256)
    ↓ ReLU
┌────────────────┬────────────────┐
│ Linear(256→z)  │ Linear(256→z)  │
│     μ          │    log σ²      │
└────────────────┴────────────────┘

Reparameterization:
z = μ + σ ⊙ ε,  ε ~ N(0, I)

Decoder:
z (latent_dim)
    ↓
Linear(z→256)
    ↓ ReLU
Linear(256→3136)
    ↓ ReLU
Reshape → (7×7×64)
    ↓
ConvT2d(64→32, k=3, s=2, p=1, op=1) → (14×14×32)
    ↓ ReLU
ConvT2d(32→1, k=3, s=2, p=1, op=1)  → (28×28×1)
    ↓ Sigmoid
Output (28×28×1)
```

---

## 파일 구조

```
11_VAE/
├── README.md
├── numpy/
│   └── vae_numpy.py          # NumPy VAE (forward만)
├── pytorch_lowlevel/
│   └── vae_lowlevel.py       # PyTorch Low-Level VAE
├── paper/
│   └── vae_paper.py          # 논문 재현
└── exercises/
    ├── 01_latent_space.md    # 잠재 공간 시각화
    └── 02_interpolation.md   # 잠재 공간 보간
```

---

## 핵심 개념

### 1. Latent Space

```
좋은 잠재 공간의 특성:
1. Continuity: 가까운 점들은 비슷한 출력
2. Completeness: 모든 점이 의미있는 출력 생성
3. (Disentanglement): 각 차원이 독립적 특성 제어

VAE vs AE:
- AE: 점 임베딩 → 불연속적, 빈 공간 있음
- VAE: 분포 임베딩 → 연속적, 샘플링 가능
```

### 2. VAE Variants

```
β-VAE (β > 1):
- 더 강한 KL regularization
- Better disentanglement
- Worse reconstruction

Conditional VAE (CVAE):
- 조건 c 추가: q(z|x, c), p(x|z, c)
- 조건부 생성 가능

VQ-VAE:
- 연속 잠재 공간 대신 이산 코드북
- DALL-E, AudioLM 등에 사용
```

### 3. 학습 안정성

```
KL Annealing:
- 초기: β=0 (reconstruction에 집중)
- 점진적으로 β→1 (정규화 추가)

Free Bits:
- KL 최소값 보장 (posterior collapse 방지)
- L_KL = max(KL, λ)
```

---

## 구현 레벨

### Level 2: PyTorch Low-Level (pytorch_lowlevel/)
- F.conv2d, F.linear 직접 사용
- reparameterization trick 구현
- ELBO 손실 함수 구현

### Level 3: Paper Implementation (paper/)
- β-VAE 구현
- CVAE (Conditional) 구현
- 잠재 공간 시각화

---

## 학습 체크리스트

- [ ] ELBO 유도 과정 이해
- [ ] Reparameterization trick 이해
- [ ] KL divergence 계산
- [ ] β의 역할 이해
- [ ] 잠재 공간 시각화
- [ ] Conditional VAE 구현

---

## 참고 자료

- Kingma & Welling (2013). "Auto-Encoding Variational Bayes"
- Higgins et al. (2017). "β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework"
- [생성 모델 - VAE (Variational Autoencoder)](./30_Generative_Models_VAE.md)
