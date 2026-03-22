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
