# 11. Variational Autoencoder (VAE)

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
- [../Deep_Learning/16_VAE.md](../Deep_Learning/16_VAE.md)
