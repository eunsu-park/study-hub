[이전: 생성 모델 - GAN](./28_Generative_Models_GAN.md) | [다음: 생성 모델 - VAE](./30_Generative_Models_VAE.md)

---

# 29. 생성적 적대 신경망(Generative Adversarial Networks, GAN)

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. GAN 미니맥스(Minimax) 목적함수를 공식화하고, 적대적 학습 게임에서 생성자(Generator)와 판별자(Discriminator)의 역할을 설명합니다.
2. 고정된 생성자에 대한 최적 판별자 D*(x)를 유도하고, 이를 사용하여 GAN 목적함수가 젠슨-섀넌 발산(Jensen-Shannon Divergence)을 최소화함을 보입니다.
3. GAN 학습의 일반적인 불안정성(모드 붕괴(Mode Collapse), 기울기 소실, 비수렴)을 식별하고 설명하며, 이를 완화하는 기법을 서술합니다.
4. PyTorch에서 생성자에는 전치 합성곱(Transposed Convolution), 판별자에는 스트라이드 합성곱(Strided Convolution)을 사용하여 DCGAN(Deep Convolutional GAN)을 처음부터 구현합니다.
5. 이미지 데이터셋에서 GAN을 훈련하고, 시각적 검사와 프레쳇 인셉션 거리(Fréchet Inception Distance, FID) 지표를 사용하여 생성 샘플 품질을 평가합니다.
6. GAN 변형(DCGAN, WGAN, 조건부 GAN)을 비교하고, 각각이 학습 안정성 향상을 위해 도입한 아키텍처 또는 손실 수정 사항을 설명합니다.

---

## 개요

생성적 적대 신경망(Generative Adversarial Networks, GAN)은 생성자(Generator)와 판별자(Discriminator) 간의 적대적 게임을 통해 현실적인 데이터를 생성하는 방법을 학습합니다. "Generative Adversarial Networks" (Goodfellow et al., 2014)

---

## 수학적 배경

### 1. 미니맥스 게임(Minimax Game)

```
목표: 생성자 G가 판별자 D를 속임

미니맥스 목적 함수:
min_G max_D V(D, G) = Eₓ~pdata[log D(x)] + Ez~pz[log(1 - D(G(z)))]
                      ──────────────────   ──────────────────────
                      실제 데이터           가짜 데이터
                      (D(x)→1 최대화)      (D(G(z))→0 최대화)

D의 목표: V 최대화 (실제와 가짜 구별)
G의 목표: V 최소화 (D를 속여 가짜를 실제로 인식하게 함)

최적 판별자:
D*(x) = pdata(x) / (pdata(x) + pg(x))

내쉬 균형에서: pg = pdata, D*(x) = 1/2
```

### 2. 학습 역학(Training Dynamics)

```
교대 최적화:

단계 1: D 업데이트 (G 고정)
  log D(x) + log(1 - D(G(z))) 최대화
  → D가 실제 vs 가짜 분류 학습

단계 2: G 업데이트 (D 고정)
  log(1 - D(G(z))) 최소화
  또는 log D(G(z)) 최대화  ← 비포화 변형(더 나은 그래디언트)

왜 log D(G(z))를 최대화?
- 초기 학습: G가 나쁨 → D(G(z)) ≈ 0
- log(1 - D(G(z))) ≈ 0 → 소실 그래디언트
- log D(G(z))는 더 강한 그래디언트 제공

┌─────────────────────────────────────────┐
│  GAN 학습 루프:                         │
│                                         │
│  for epoch in epochs:                   │
│    for real_batch in dataloader:        │
│      # 1. 판별자 업데이트               │
│      z ~ N(0, I)                        │
│      fake = G(z)                        │
│      loss_D = -log D(real) - log(1-D(fake))
│      D.step()                           │
│                                         │
│      # 2. 생성자 업데이트               │
│      z ~ N(0, I)                        │
│      fake = G(z)                        │
│      loss_G = -log D(fake)              │
│      G.step()                           │
└─────────────────────────────────────────┘
```

### 3. 손실 함수(Loss Functions)

```
원본 GAN (Minimax):
L_D = -[log D(x) + log(1 - D(G(z)))]
L_G = -log D(G(z))  (비포화)

WGAN (Wasserstein GAN):
L_D = -[D(x) - D(G(z))]  (시그모이드 없음, 판별자 대신 비평가)
L_G = -D(G(z))
+ 가중치 클리핑 또는 그래디언트 페널티

LSGAN (Least Squares GAN):
L_D = (D(x) - 1)² + D(G(z))²
L_G = (D(G(z)) - 1)²

힌지 손실(Hinge Loss, Spectral Norm GAN):
L_D = -min(0, -1 + D(x)) - min(0, -1 - D(G(z)))
L_G = -D(G(z))
```

---

## DCGAN 아키텍처

심층 합성곱 GAN(Deep Convolutional GAN, Radford et al., 2015) - 안정적인 학습 가이드라인

### 생성자(Generator, 64×64 RGB 이미지)

```
잠재 코드 z (100차원)
    ↓
Linear(100→4×4×1024) + BatchNorm + ReLU
    ↓
Reshape → (4×4×1024)
    ↓
ConvTranspose2d(1024→512, k=4, s=2, p=1) → (8×8×512)
    ↓ BatchNorm + ReLU
ConvTranspose2d(512→256, k=4, s=2, p=1)  → (16×16×256)
    ↓ BatchNorm + ReLU
ConvTranspose2d(256→128, k=4, s=2, p=1)  → (32×32×128)
    ↓ BatchNorm + ReLU
ConvTranspose2d(128→3, k=4, s=2, p=1)    → (64×64×3)
    ↓ Tanh
출력 (64×64×3, 범위 [-1, 1])

주요 설계 선택:
- 완전 연결층 없음 (첫 번째 프로젝션 제외)
- 업샘플링에 전치 합성곱 사용
- 출력층을 제외한 모든 층에 BatchNorm
- G에서 ReLU 활성화
- Tanh 출력 (이미지를 [-1, 1]로 정규화)
```

### 판별자(Discriminator, 64×64 RGB 이미지)

```
입력 (64×64×3)
    ↓
Conv2d(3→128, k=4, s=2, p=1)  → (32×32×128)
    ↓ LeakyReLU(0.2)
Conv2d(128→256, k=4, s=2, p=1) → (16×16×256)
    ↓ BatchNorm + LeakyReLU(0.2)
Conv2d(256→512, k=4, s=2, p=1) → (8×8×512)
    ↓ BatchNorm + LeakyReLU(0.2)
Conv2d(512→1024, k=4, s=2, p=1) → (4×4×1024)
    ↓ BatchNorm + LeakyReLU(0.2)
Conv2d(1024→1, k=4, s=1, p=0) → (1×1×1)
    ↓ Sigmoid (또는 WGAN의 경우 제거)
출력 (스칼라 확률)

주요 설계 선택:
- 완전 연결층 없음 (마지막 합성곱에 암묵적으로 포함)
- 다운샘플링에 스트라이드 합성곱 사용 (풀링 없음)
- 입력/출력층을 제외한 모든 층에 BatchNorm
- LeakyReLU 활성화 (α=0.2)
- 이진 분류를 위한 시그모이드 출력
```

### DCGAN 가이드라인

```
1. 풀링을 스트라이드 합성곱(D) / 전치 합성곱(G)으로 대체
2. G와 D 모두에서 BatchNorm 사용
3. 완전 연결 은닉층 제거
4. G에서 ReLU 사용 (출력 제외: Tanh)
5. D에서 LeakyReLU 사용 (α=0.2)
```

---

## 학습 기법

### 1. 레이블 스무딩(Label Smoothing)

```
문제: D가 너무 확신함 (D(real)→1, D(fake)→0)
→ G의 소실 그래디언트

해결: 레이블 스무딩
실제 레이블: 1.0 → 0.9 (단방향 레이블 스무딩)
가짜 레이블: 0.0 (그대로 유지)

loss_D_real = BCE(D(real), 0.9)  # 1.0 대신
loss_D_fake = BCE(D(fake), 0.0)
```

### 2. 특징 매칭(Feature Matching)

```
문제: G가 D를 속이기 위해 최적화하지, 현실적인 샘플 생성에 최적화하지 않음

해결: 중간 특징 매칭
loss_G = ||E[f(x)] - E[f(G(z))]||²

여기서 f(·)는 D의 중간층

학습을 안정화하고, 모드 붕괴(Mode Collapse) 감소
```

### 3. 미니배치 판별(Minibatch Discrimination)

```
문제: G가 제한된 다양성 생성 (모드 붕괴)

해결: D가 전체 배치를 보도록 함
1. 각 샘플에서 특징 추출
2. 배치 내 유사도 계산
3. 배치 통계를 각 샘플에 추가

D가 G가 동일한 샘플을 생성하는지 감지 가능
```

### 4. 스펙트럴 정규화(Spectral Normalization)

```
문제: 판별자 그래디언트 폭발

해결: 가중치 행렬 정규화
W_SN = W / σ(W)

여기서 σ(W)는 최대 특이값

학습 안정화 (Miyato et al., 2018)
BigGAN, StyleGAN에서 사용
```

### 5. 점진적 성장(Progressive Growing)

```
낮은 해상도(4×4)에서 학습 시작
점진적으로 층을 추가하여 고해상도(1024×1024) 도달

4×4 → 8×8 → 16×16 → ... → 1024×1024

해상도 간 부드러운 전환
ProGAN, StyleGAN에서 사용
```

---

## 모드 붕괴(Mode Collapse)

### 1. 모드 붕괴란?

```
문제: G가 제한된 다양성 생성

데이터 분포의 모드:
                 모드 1  모드 2  모드 3
실제 데이터:        ●●●     ●●●     ●●●
건강한 GAN:        ○○○     ○○○     ○○○
모드 붕괴:         ○○○
완전 붕괴:                  ○○○○○○○

생성자가 데이터 분포의 일부를 무시
```

### 2. 모드 붕괴 감지

```
증상:
1. 생성된 샘플이 유사해 보임
2. 다른 z에도 불구하고 낮은 다양성
3. 학습 손실이 진동
4. 낮은 FID에도 불구하고 높은 NLL (음의 로그 우도)

메트릭:
- 인셉션 스코어(Inception Score, IS): 다양성 + 품질 측정
- 프레셰 인셉션 거리(Fréchet Inception Distance, FID): 분포 거리
- 정밀도/재현율(Precision/Recall): 모드 커버리지
```

### 3. 완화 전략

```
1. 언롤드 GAN(Unrolled GAN):
   - G가 미래 D에 대해 최적화 (k 단계 앞)
   - G가 현재 D를 악용하는 것 방지

2. 미니배치 판별:
   - D가 다양성 부족 감지

3. WGAN / WGAN-GP:
   - 더 부드러운 그래디언트 흐름
   - 더 나은 학습 안정성

4. 다중 판별자:
   - 각 D가 다른 모드 캡처

5. 정규화:
   - D 입력에 노이즈 추가
   - D에서 드롭아웃
```

---

## 파일 구조

```
14_GAN/
├── README.md
├── pytorch_lowlevel/
│   ├── dcgan_mnist.py        # MNIST용 DCGAN (28×28)
│   └── dcgan_cifar.py        # CIFAR-10용 DCGAN (32×32)
├── paper/
│   ├── dcgan_paper.py        # 전체 DCGAN (64×64)
│   ├── wgan_gp.py            # 그래디언트 페널티가 있는 WGAN
│   ├── stylegan_simple.py    # 단순화된 StyleGAN
│   └── conditional_gan.py    # 조건부 GAN (cGAN)
└── exercises/
    ├── 01_mode_collapse.md   # 모드 붕괴 진단
    └── 02_spectral_norm.md   # 스펙트럴 정규화 구현
```

---

## 핵심 개념

### 1. GAN 변형

```
조건부 GAN (Conditional GAN, cGAN):
- 클래스 레이블 c 추가: G(z, c), D(x, c)
- 제어된 생성 (예: 숫자 7 생성)

WGAN (Wasserstein GAN):
- JS 발산을 와서스테인 거리로 대체
- D에서 시그모이드 없음 (비평가가 됨)
- 가중치 클리핑 또는 그래디언트 페널티
- 더 안정적인 학습

StyleGAN:
- 점진적 아키텍처
- 스타일 모듈레이션 (AdaIN)
- 분리된 잠재 공간 W
- 최첨단 이미지 품질

CycleGAN:
- 쌍을 이루지 않은 이미지 간 변환
- 사이클 일관성 손실: G(F(x)) ≈ x
- 사용 사례: 말↔얼룩말, 여름↔겨울
```

### 2. 평가 메트릭

```
인셉션 스코어(Inception Score, IS):
IS = exp(E[KL(p(y|x) || p(y))])

높을수록 좋음 (품질 + 다양성)
범위: 1부터 C (클래스 수)

프레셰 인셉션 거리(Fréchet Inception Distance, FID):
FID = ||μ_real - μ_fake||² + Tr(Σ_real + Σ_fake - 2√(Σ_real·Σ_fake))

낮을수록 좋음 (실제 분포에 가까움)
GAN의 골드 스탠다드

정밀도/재현율(Precision/Recall):
정밀도 = 품질 (가짜 샘플이 현실적임)
재현율 = 커버리지 (모든 모드가 캡처됨)
```

### 3. 학습 팁

```
1. 학습률:
   - D: 2e-4 (Adam, β₁=0.5, β₂=0.999)
   - G: 2e-4 (동일한 옵티마이저 설정)

2. 업데이트 빈도:
   - 1-5 D 업데이트당 1 G 업데이트
   - D가 약간 앞서야 함

3. 초기화:
   - Xavier/He 초기화
   - BatchNorm 매개변수: γ=1, β=0

4. 데이터:
   - 이미지를 [-1, 1]로 정규화 (Tanh 출력)
   - 무작위 뒤집기 증강

5. 잠재 코드:
   - z ~ N(0, I), 차원 = 100-512
   - 균일 분포 U(-1, 1) 사용 가능

6. 모니터링:
   - D(real), D(fake) 로깅 (0.5 주변에 머물러야 함)
   - 매 에포크마다 고정된 z 샘플 생성
   - 모드 붕괴 주시 (반복적인 샘플)
```

---

## 구현 레벨

### 레벨 2: PyTorch 로우레벨 (pytorch_lowlevel/)
- 처음부터 DCGAN 아키텍처 구축
- 교대 학습 루프 구현
- MNIST 및 CIFAR-10에서 학습
- 생성된 샘플 시각화

### 레벨 3: 논문 구현 (paper/)
- 학습 트릭이 있는 전체 DCGAN
- 그래디언트 페널티가 있는 WGAN
- 조건부 GAN (클래스 조건부)
- 스타일 모듈레이션이 있는 단순화된 StyleGAN
- FID/IS 평가

---

## 학습 루프

```python
# 의사코드
for epoch in epochs:
    for real_images, _ in dataloader:
        batch_size = real_images.size(0)

        # ========== 판별자 학습 ==========
        # 실제 이미지
        real_labels = torch.ones(batch_size, 1) * 0.9  # 레이블 스무딩
        output_real = D(real_images)
        loss_D_real = BCE(output_real, real_labels)

        # 가짜 이미지
        z = torch.randn(batch_size, latent_dim)
        fake_images = G(z)
        fake_labels = torch.zeros(batch_size, 1)
        output_fake = D(fake_images.detach())  # G 그래디언트 방지를 위해 detach
        loss_D_fake = BCE(output_fake, fake_labels)

        # 전체 D 손실
        loss_D = loss_D_real + loss_D_fake
        optimizer_D.zero_grad()
        loss_D.backward()
        optimizer_D.step()

        # ========== 생성자 학습 ==========
        z = torch.randn(batch_size, latent_dim)
        fake_images = G(z)
        output = D(fake_images)
        real_labels = torch.ones(batch_size, 1)
        loss_G = BCE(output, real_labels)  # G는 D(fake)→1을 원함

        optimizer_G.zero_grad()
        loss_G.backward()
        optimizer_G.step()
```

---

## 샘플링

```python
# 새 이미지 생성
z = torch.randn(64, latent_dim)  # 64개 배치
with torch.no_grad():
    fake_images = G(z)

# [-1, 1]에서 [0, 1]로 역정규화
fake_images = (fake_images + 1) / 2

# 조건부 GAN의 경우
labels = torch.randint(0, 10, (64,))  # 10개 클래스에서 64개 샘플 생성
fake_images = G(z, labels)
```

---

## 학습 체크리스트

- [ ] 미니맥스 게임 공식 이해
- [ ] DCGAN 아키텍처 가이드라인 구현
- [ ] 교대 학습 루프 마스터
- [ ] 모드 붕괴 증상 인식
- [ ] 레이블 스무딩 및 스펙트럴 정규화 구현
- [ ] WGAN 및 그래디언트 페널티 이해
- [ ] FID 및 인셉션 스코어 계산
- [ ] 조건부 GAN 구현

---

## 참고 문헌

- Goodfellow et al. (2014). "Generative Adversarial Networks"
- Radford et al. (2015). "Unsupervised Representation Learning with Deep Convolutional GANs"
- Arjovsky et al. (2017). "Wasserstein GAN"
- Gulrajani et al. (2017). "Improved Training of Wasserstein GANs"
- Miyato et al. (2018). "Spectral Normalization for GANs"
- Karras et al. (2019). "A Style-Based Generator Architecture for GANs"
- [생성 모델 - GAN (Generative Adversarial Networks)](./28_Generative_Models_GAN.md)
