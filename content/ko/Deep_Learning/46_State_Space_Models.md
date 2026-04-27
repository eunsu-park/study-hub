[이전: 확산 모델 심화](./45_Diffusion_Models_Advanced.md) | [다음: 전문가 혼합 모델](./47_Mixture_of_Experts.md)

---

# 46. 상태 공간 모델(State Space Models)

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 대안적 아키텍처를 필요로 하는 트랜스포머 어텐션(Transformer attention)의 한계를 설명할 수 있다
2. 연속 및 이산 상태 공간 모델(SSM) 공식을 유도할 수 있다
3. S4와 구조화된 행렬을 통한 효율적 계산을 설명할 수 있다
4. Mamba의 선택적 상태 공간(Selective State Space) 메커니즘과 하드웨어 인식 알고리즘을 설명할 수 있다
5. 속도, 품질, 스케일링 측면에서 SSM과 트랜스포머를 비교할 수 있다
6. 어텐션과 SSM을 결합한 하이브리드 아키텍처를 이해할 수 있다
7. PyTorch로 기본 SSM 레이어를 구현할 수 있다

---

## 목차

1. [트랜스포머의 한계](#1-트랜스포머의-한계)
2. [상태 공간 모델: 수학적 기초](#2-상태-공간-모델-수학적-기초)
3. [S4: 시퀀스를 위한 구조화된 상태 공간](#3-s4-시퀀스를-위한-구조화된-상태-공간)
4. [Mamba: 선택적 상태 공간](#4-mamba-선택적-상태-공간)
5. [Mamba-2와 개선 사항](#5-mamba-2와-개선-사항)
6. [SSM 대 트랜스포머 비교](#6-ssm-대-트랜스포머-비교)
7. [하이브리드 아키텍처](#7-하이브리드-아키텍처)
8. [구현 세부 사항과 훈련](#8-구현-세부-사항과-훈련)
9. [연습문제](#9-연습문제)

## 1. 트랜스포머의 한계

### 이론: Transformer가 벽에 부딪히는 이유

표준 attention 층은 `softmax(QK^T / sqrt(d)) V`를 계산. `QK^T` 행렬은 시퀀스 길이 `N`에 대해 `N x N`이며, `O(N^2)` 시간과 메모리를 요구. `N = 100k`(긴 문서)의 경우, fp32에서 10^10 연산과 40 GB 메모리 — 완전히 비실용적.

많은 부분적 수정이 존재(희소 attention, 선형 attention 근사, FlashAttention의 메모리 트릭), 하지만 그것들이 품질 또는 근본적 복잡도를 거래. SSM은 다른 길을 취함: attention을 완전히 떨어뜨리고 구성상 O(N)인 다른 시퀀스 모델링 원시를 사용.


### 1.1 이차 어텐션 문제(Quadratic Attention Problem)

자기 어텐션(Self-attention)은 모든 토큰 간의 쌍별 상호작용을 계산합니다:

```
자기 어텐션 복잡도:

입력: N개 토큰의 시퀀스, 각각 차원 D

Q, K, V = X @ W_Q, X @ W_K, X @ W_V      O(N * D²)
Attn = softmax(Q @ K^T / √d) @ V          O(N² * D)

긴 시퀀스의 경우:
  N = 1K    → 1M 연산         ✓ 빠름
  N = 8K    → 64M 연산        ✓ 관리 가능
  N = 32K   → 1B 연산         △ 느림
  N = 128K  → 16B 연산        ✗ 매우 비쌈
  N = 1M    → 1T 연산         ✗ 비현실적

어텐션 행렬의 메모리:
  N = 8K, float16:  8K × 8K × 2 바이트 = 헤드당, 레이어당 128 MB
  N = 128K:         128K × 128K × 2 바이트 = 헤드당, 레이어당 32 GB
```

### 1.2 어텐션 근사와 한계

```
접근법             복잡도      품질       채택
──────────────────────────────────────────────────
전체 어텐션         O(N²)      최고       보편적
FlashAttention     O(N²)*     최고       보편적 (*IO 최적화)
선형 어텐션         O(N)       저하됨     제한적
희소 어텐션         O(N√N)     좋음       일부 모델
슬라이딩 윈도우     O(N*W)     좋음       Mistral 등
SSM               O(N)       좋음       Mamba 등
```

### 1.3 추론 병목(Inference Bottleneck)

```
자기회귀 트랜스포머 추론:

단계 1: 프롬프트 처리 (프리필)     → O(N²) 하지만 병렬화 가능
단계 2: 토큰 하나씩 생성          → 각 새 토큰이 이전 모든 토큰에 어텐드

토큰 1:    [1]에 어텐드                           → 1 어텐션 연산
토큰 2:    [1, 2]에 어텐드                        → 2 어텐션 연산
토큰 3:    [1, 2, 3]에 어텐드                     → 3 어텐션 연산
...
토큰 N:    [1, 2, ..., N]에 어텐드                → N 어텐션 연산

생성 총합: 1 + 2 + ... + N = O(N²) 총
추가: KV 캐시가 선형으로 증가: O(N * D * L) 메모리

SSM 추론:
각 토큰: 고정 크기 상태 업데이트 → 토큰당 O(1), 총 O(N)
KV 캐시 불필요 → 상수 메모리
```

---

## 2. 상태 공간 모델: 수학적 기초

### 이론: 연속 SSM과 이산화

연속 시간 SSM은 선형 ODE:

```
\dot{h}(t) = A h(t) + B u(t)
y(t) = C h(t)
```

`u(t)`가 입력, `h(t)`가 은닉 상태, `y(t)`가 출력. `A, B, C`가 학습 가능한 행렬. 이는 정확히 제어 이론의 표준 선형 시불변(LTI) 시스템.

이산 시퀀스에 사용하기 위해, 스텝 크기 `\Delta`로 이산화:

```
h_t = \bar A h_{t-1} + \bar B u_t
y_t = C h_t
\bar A = exp(\Delta A),  \bar B = (\Delta A)^{-1} (exp(\Delta A) - I) \cdot \Delta B
```

이제 선형 점화식 — 비선형성 없는 RNN과 정확히 같음. 두 사실이 이를 딥러닝에 유용하게 만듦:

1. **선형 점화식은 *parallel scan* 알고리즘을 통해 병렬화 가능** — O(N) 작업이지만 O(log N) 깊이, 표준 RNN과 달리 GPU 병렬성 활용.
2. **점화식이 커널 `K = (CB, CAB, CA^2 B, ...)`를 가진 긴 합성곱으로 표현될 수 있음**, FFT를 통해 O(N log N)에서 계산 가능.

따라서 SSM은 RNN 같은 순차 모델링과 CNN 같은 병렬성을 결합. 트릭은 attention과 경쟁할 만큼 *충분히 표현력 있게* 만드는 것.


### 2.1 연속 상태 공간 모델

SSM은 제어 이론에서 유래합니다. 연속 시간 선형 SSM:

```
상태 방정식:      h'(t) = A h(t) + B x(t)
출력 방정식:      y(t)  = C h(t) + D x(t)

여기서:
  x(t) ∈ R^1        입력 신호 (채널당 스칼라)
  h(t) ∈ R^N        은닉 상태 (N차원)
  y(t) ∈ R^1        출력 신호
  A ∈ R^{N×N}       상태 행렬 (동역학)
  B ∈ R^{N×1}       입력 행렬
  C ∈ R^{1×N}       출력 행렬
  D ∈ R^{1×1}       스킵 연결 (보통 0으로 설정)
```

### 2.2 이산화(Discretization)

SSM을 시퀀스에 적용하기 위해 스텝 크기 Δ로 이산화합니다:

```
영차 유지(Zero-Order Hold, ZOH) 이산화:

Ā = exp(ΔA)                    ← 행렬 지수
B̄ = (ΔA)^{-1} (exp(ΔA) - I) ΔB
  ≈ ΔB                         ← 1차 근사

이산 반복:
  h_k = Ā h_{k-1} + B̄ x_k
  y_k = C h_k

이것은 선형 반복식 — 두 가지 모드로 계산 가능:
  1. 반복 모드: 단계당 O(N) — 추론에 적합
  2. 합성곱 모드: 전체 시퀀스에 O(L log L) — 훈련에 적합
```

### 2.3 합성곱 관점(Convolution View)

이산 SSM은 합성곱으로 전개될 수 있습니다:

```
h_0 = B̄ x_0
h_1 = Ā B̄ x_0 + B̄ x_1
h_2 = Ā² B̄ x_0 + Ā B̄ x_1 + B̄ x_2
...

y_k = C h_k = C Ā^k B̄ x_0 + C Ā^{k-1} B̄ x_1 + ... + C B̄ x_k

커널을 사용한 합성곱:
  K̄ = (C B̄, C Ā B̄, C Ā² B̄, ..., C Ā^{L-1} B̄)

y = K̄ * x    (합성곱, FFT를 통해 O(L log L)로 계산)
```

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicSSM(nn.Module):
    """기본 상태 공간 모델 레이어."""

    def __init__(self, d_model, state_dim=64):
        super().__init__()
        self.d_model = d_model
        self.state_dim = state_dim

        # SSM 파라미터 (채널별)
        self.A = nn.Parameter(torch.randn(d_model, state_dim))
        self.B = nn.Parameter(torch.randn(d_model, state_dim))
        self.C = nn.Parameter(torch.randn(d_model, state_dim))
        self.D = nn.Parameter(torch.ones(d_model))

        # 이산화 스텝 크기 (학습 가능, 채널별)
        self.log_delta = nn.Parameter(torch.randn(d_model) - 4.0)

    def discretize(self):
        """ZOH를 사용하여 이산 A_bar, B_bar 계산."""
        delta = torch.exp(self.log_delta)  # (D,)
        # 간략화: 효율성을 위한 대각 A
        A_bar = torch.exp(delta.unsqueeze(-1) * self.A)  # (D, N)
        B_bar = delta.unsqueeze(-1) * self.B               # (D, N)
        return A_bar, B_bar

    def forward_recurrent(self, x):
        """
        반복 모드: 단계당 O(L), 총 O(L*N).
        x: (B, L, D)
        """
        B_size, L, D = x.shape
        A_bar, B_bar = self.discretize()

        h = torch.zeros(B_size, D, self.state_dim, device=x.device)
        outputs = []

        for t in range(L):
            x_t = x[:, t, :]  # (B, D)
            h = A_bar.unsqueeze(0) * h + B_bar.unsqueeze(0) * x_t.unsqueeze(-1)
            y_t = (self.C.unsqueeze(0) * h).sum(dim=-1)  # (B, D)
            y_t = y_t + self.D * x_t
            outputs.append(y_t)

        return torch.stack(outputs, dim=1)  # (B, L, D)

    def compute_kernel(self, L):
        """길이 L의 합성곱 커널 계산."""
        A_bar, B_bar = self.discretize()

        # K[i] = C * A_bar^i * B_bar (각 채널에 대해)
        kernel = []
        power = torch.ones_like(A_bar)  # A_bar^0 = I (대각)
        for i in range(L):
            k_i = (self.C * power * B_bar).sum(dim=-1)  # (D,)
            kernel.append(k_i)
            power = power * A_bar
        return torch.stack(kernel, dim=-1)  # (D, L)

    def forward_conv(self, x):
        """
        합성곱 모드: FFT를 통해 O(L log L).
        x: (B, L, D)
        """
        B_size, L, D = x.shape
        K = self.compute_kernel(L)  # (D, L)

        # FFT 합성곱
        x_perm = x.transpose(1, 2)  # (B, D, L)
        K_f = torch.fft.rfft(K, n=2*L)
        x_f = torch.fft.rfft(x_perm, n=2*L)
        y = torch.fft.irfft(K_f * x_f, n=2*L)[..., :L]  # (B, D, L)

        y = y.transpose(1, 2)  # (B, L, D)
        y = y + self.D * x  # 스킵 연결
        return y

    def forward(self, x, mode="conv"):
        if mode == "conv":
            return self.forward_conv(x)
        else:
            return self.forward_recurrent(x)
```

---

## 3. S4: 시퀀스를 위한 구조화된 상태 공간

### 이론: S4: 구조화 행렬

전체 `A in R^{d x d}`를 가진 순진한 SSM은 너무 비쌈. S4 (Gu, Goel, Re 2021)는 `A`를 구조화된 형태로 제한: HiPPO-LegS, 다항식으로 연속 함수를 근사하는 데서 유도. 이 특정 구조가 두 결정적 성질을 가짐:

1. **장거리 기억**: 상태가 제어 가능한 망각률로 임의로 먼 과거의 정보를 포착.
2. **효율적 계산**: Cauchy 커널을 통해 합성곱 커널을 O(N log N)에서 계산 가능.

S4는 장거리 벤치마크(Path-X, Long Range Arena)에서 Transformer 성능을 일치시킨 첫 SSM 변형이었으며, 장거리 컨텍스트 모델링에 O(N^2) attention이 필수가 아님을 증명.


### 3.1 장거리 의존성의 과제

위의 나이브 SSM에는 치명적인 문제가 있습니다: 행렬 A_bar^k가 k가 커짐에 따라 감쇠하거나 폭발하여 장거리 의존성 포착이 어렵습니다.

```
문제:
  A_bar의 고유값이 |λ| < 1이면: 신호가 지수적으로 감쇠
  A_bar의 고유값이 |λ| > 1이면: 신호가 폭발
  A_bar의 고유값이 |λ| = 1이면: 기울기 소실

이것은 정확히 기울기 소실/폭발 문제!
```

### 3.2 HiPPO 초기화

S4(Gu et al., 2022)는 **HiPPO**(High-order Polynomial Projection Operator, 고차 다항식 투영 연산자) 초기화로 이 문제를 해결했습니다:

```
HiPPO-LegS 행렬:

A_{nk} = -(2n+1)^{1/2} (2k+1)^{1/2}  n > k인 경우
A_{nk} = -(n+1)                        n = k인 경우
A_{nk} = 0                              n < k인 경우

이 행렬은 상태 h(t)가 입력 신호의 이력을 다항식 기저로
최적으로 압축하는 특별한 속성을 가집니다.
```

```python
def hippo_legs_matrix(N):
    """크기 N×N의 HiPPO-LegS 행렬 생성."""
    P = torch.sqrt(1 + 2 * torch.arange(N, dtype=torch.float32))
    A = torch.zeros(N, N)
    for n in range(N):
        for k in range(n + 1):
            if n > k:
                A[n, k] = -P[n] * P[k]
            elif n == k:
                A[n, k] = -(n + 1)
    return A
```

### 3.3 S4 아키텍처

```
S4 블록:

입력 (B, L, D)
    │
    ▼
선형 투영 → (B, L, H)     H개의 독립적인 SSM 채널
    │
    ▼
┌──────────────────┐
│  H개 병렬 SSM    │    각 SSM: state_dim = 64
│  (대각 또는      │    HiPPO 초기화된 A 사용
│   DPLR 형태)     │    훈련 시 합성곱 모드
└──────────────────┘
    │
    ▼
활성화 (GELU)
    │
    ▼
선형 투영 → (B, L, D)
    │
    ▼
잔차 + LayerNorm → 출력 (B, L, D)
```

### 3.4 DPLR 매개변수화

S4는 효율적인 커널 계산을 위해 A를 대각 + 저랭크(Diagonal Plus Low-Rank, DPLR)로 표현합니다:

```
A = Λ - P P*    (대각 + 랭크-1 보정)

코시 커널(Cauchy kernel)을 통한 커널 계산:
  K̂(ω) = C * (iω - A)^{-1} * B

DPLR A의 경우:
  (iω - Λ + PP*)^{-1}는 우드버리 항등식(Woodbury identity)으로 O(N)에 계산 가능

총 커널 계산: O(N² * L) 대신 O(N * L)
합성곱을 위한 FFT: 총 O(L log L)
```

---

## 4. Mamba: 선택적 상태 공간

### 이론: Mamba: 선택적 State Space

S4의 `(A, B, C)`은 *고정*(입력 무관)이며, 이는 특정 입력에 선택적으로 집중할 수 없음을 의미. **Mamba** (Gu & Dao 2023)는 그것들을 *입력 의존*으로 만듦:

```
B_t = Linear(u_t),  C_t = Linear(u_t),  \Delta_t = softplus(Linear(u_t))
h_t = \bar A_t h_{t-1} + \bar B_t u_t              (\bar A_t가 \Delta_t에 의존)
y_t = C_t h_t
```

이제 SSM은 중요한 토큰을 "확대"(작은 `\Delta`가 상태 유지)하고 중요하지 않은 것을 "건너뜀"(큰 `\Delta`가 감쇠 가속) 가능. 이는 attention의 `softmax(QK^T)`와 유사한 내용 기반 선택 메커니즘이지만, 근본적으로 다른 스케일링.

Mamba는 또한 HBM이 아닌 SRAM에서 parallel scan을 수행하는 **하드웨어 인식 커널**을 도입, 긴 컨텍스트에서 동등 Transformer보다 5배 빠른 학습 달성. 실험적으로, Mamba는 표준 NLP 벤치마크에서 비슷한 크기의 Transformer LM을 일치시키거나 능가.

현재 합의: SSM은 이제 매우 긴 시퀀스에 Transformer의 진짜 대안이며, 하이브리드 아키텍처(Mamba + attention 층)가 점점 인기.


### 4.1 선택성 문제(Selectivity Problem)

S4와 같은 선형 SSM에는 근본적인 한계가 있습니다: **선형 시불변(Linear Time-Invariant, LTI)** 시스템이므로 입력 내용에 관계없이 A, B, C가 고정됩니다:

```
LTI SSM:
  h_k = Ā h_{k-1} + B̄ x_k     ← 모든 토큰에 같은 Ā, B̄
  y_k = C h_k                    ← 모든 토큰에 같은 C

문제: 토큰을 선택적으로 집중하거나 무시할 수 없음
  "The cat sat on the mat" → "cat"과 "mat"이 동등하게 처리됨
  내용 기반 필터링을 구현할 수 없음
```

### 4.2 Mamba의 선택 메커니즘

Gu & Dao (2023)는 SSM 파라미터를 **입력 의존적**으로 만들었습니다:

```
표준 SSM:  B, C, Δ는 고정 파라미터
Mamba SSM: B, C, Δ는 입력 x의 함수

  B_k = Linear_B(x_k)      ← 입력 의존적
  C_k = Linear_C(x_k)      ← 입력 의존적
  Δ_k = softplus(Linear_Δ(x_k))  ← 입력 의존적 스텝 크기

  Ā_k = exp(Δ_k * A)       ← 이제 토큰마다 변함!
  B̄_k = Δ_k * B_k

핵심 통찰: Δ는 "상태를 얼마나 업데이트할지"를 제어
  큰 Δ → 상태를 크게 업데이트 (이 토큰에 주목)
  작은 Δ → 이 토큰 무시 (상태가 거의 변하지 않음)
```

### 4.3 하드웨어 인식 알고리즘(Hardware-Aware Algorithm)

선택성이 있으면 Mamba는 합성곱 모드를 사용할 수 없습니다(파라미터가 단계마다 변하므로). Gu & Dao는 맞춤형 CUDA 커널을 설계했습니다:

```
문제:
  선택적 SSM: h_k = A_k h_{k-1} + B_k x_k   (시변!)
  커널을 사전 계산할 수 없음 → 반복을 사용해야 함
  나이브 반복: O(L * N * D) 많은 HBM 읽기 포함

해결책: 하드웨어 인식 선택적 스캔

  1. (x, Δ, B, C)의 청크를 HBM에서 SRAM으로 로드
  2. 이산화된 (Ā, B̄)를 SRAM에서 계산
  3. SRAM에서 반복 실행 (빠름!)
  4. 출력만 HBM에 다시 저장
  5. 역전파: 중간 상태를 재계산
     (재계산이 HBM I/O보다 빠름)

메모리:  O(B * L * D * N) → 재계산으로 O(B * L * D)
속도:    나이브 구현 대비 ~3-5배 빠름
```

### 4.4 Mamba 블록 아키텍처

```
Mamba 블록 (트랜스포머 블록을 대체):

입력 (B, L, D)
    │
    ├────────────────────────┐
    ▼                        ▼
Linear (D → E)           Linear (D → E)
    │                        │
    ▼                        │
Conv1D (kernel=4)            │
    │                        │
    ▼                        │
SiLU 활성화                   │
    │                        │
    ▼                        │
┌────────────────┐           │
│ 선택적 SSM     │           │
│ (x에서 B,C,Δ) │           │
└────────────────┘           │
    │                        │
    ▼                        ▼
    × ◄──── SiLU(·) ────────┘    (게이팅)
    │
    ▼
Linear (E → D)
    │
    ▼
출력 (B, L, D)

E = expand_factor * D (일반적으로 2배)
어텐션 없음, MLP 없음 — 이 블록만 반복
```

```python
class MambaBlock(nn.Module):
    """간략화된 Mamba 블록 (맞춤형 CUDA 커널 없이)."""

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        d_inner = int(expand * d_model)
        self.d_inner = d_inner

        # 입력 투영
        self.in_proj = nn.Linear(d_model, d_inner * 2, bias=False)

        # Conv1D
        self.conv1d = nn.Conv1d(
            d_inner, d_inner, kernel_size=d_conv,
            padding=d_conv - 1, groups=d_inner
        )

        # SSM 파라미터
        self.A_log = nn.Parameter(torch.log(torch.arange(1, d_state + 1, dtype=torch.float32)
                                             .unsqueeze(0).expand(d_inner, -1)))
        self.D = nn.Parameter(torch.ones(d_inner))

        # 선택 투영 (입력 의존적 B, C, delta)
        self.x_proj = nn.Linear(d_inner, d_state * 2 + 1, bias=False)  # B, C, delta
        self.dt_proj = nn.Linear(1, d_inner, bias=True)

        # 출력 투영
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)

    def selective_scan(self, x, delta, B, C):
        """
        선택적 SSM 실행.
        x: (B, L, D_inner)
        delta: (B, L, D_inner)
        B: (B, L, N)
        C: (B, L, N)
        """
        batch, L, d_inner = x.shape
        N = self.d_state

        A = -torch.exp(self.A_log)  # (D_inner, N)

        # 토큰별 이산화
        delta_A = torch.exp(delta.unsqueeze(-1) * A)  # (B, L, D_inner, N)
        delta_B_x = delta.unsqueeze(-1) * B.unsqueeze(2) * x.unsqueeze(-1)  # (B, L, D_inner, N)

        # 순차 스캔
        h = torch.zeros(batch, d_inner, N, device=x.device)
        outputs = []

        for t in range(L):
            h = delta_A[:, t] * h + delta_B_x[:, t]  # (B, D_inner, N)
            y_t = (h * C[:, t].unsqueeze(1)).sum(dim=-1)  # (B, D_inner)
            outputs.append(y_t)

        y = torch.stack(outputs, dim=1)  # (B, L, D_inner)
        y = y + self.D * x  # 스킵 연결
        return y

    def forward(self, x):
        """
        x: (B, L, D)
        """
        B_size, L, D = x.shape

        # 이중 투영
        xz = self.in_proj(x)  # (B, L, 2*D_inner)
        x_branch, z = xz.chunk(2, dim=-1)

        # Conv1D
        x_branch = x_branch.transpose(1, 2)  # (B, D_inner, L)
        x_branch = self.conv1d(x_branch)[:, :, :L]  # 인과적 패딩
        x_branch = x_branch.transpose(1, 2)  # (B, L, D_inner)
        x_branch = F.silu(x_branch)

        # 입력 의존적 SSM 파라미터
        x_ssm = self.x_proj(x_branch)  # (B, L, 2N+1)
        B_param = x_ssm[:, :, :self.d_state]
        C_param = x_ssm[:, :, self.d_state:2*self.d_state]
        delta = F.softplus(self.dt_proj(x_ssm[:, :, -1:]))  # (B, L, D_inner)

        # 선택적 스캔
        y = self.selective_scan(x_branch, delta, B_param, C_param)

        # 게이팅
        y = y * F.silu(z)

        # 출력 투영
        return self.out_proj(y)
```

### 4.5 Mamba 모델

```python
class MambaModel(nn.Module):
    """전체 Mamba 언어 모델."""

    def __init__(self, vocab_size, d_model=768, n_layers=24,
                 d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)

        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'norm': nn.LayerNorm(d_model),
                'mamba': MambaBlock(d_model, d_state, d_conv, expand),
            })
            for _ in range(n_layers)
        ])

        self.norm_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids):
        x = self.embedding(input_ids)

        for layer in self.layers:
            x = x + layer['mamba'](layer['norm'](x))

        x = self.norm_f(x)
        logits = self.lm_head(x)
        return logits

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=100, temperature=1.0):
        """단계당 상수 메모리의 자기회귀 생성."""
        for _ in range(max_new_tokens):
            logits = self.forward(input_ids)
            next_logits = logits[:, -1, :] / temperature
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)
        return input_ids
```

---

## 5. Mamba-2와 개선 사항

### 5.1 Mamba-2: 상태 공간 이중성(State Space Duality, SSD)

Dao & Gu (2024)는 SSM과 어텐션 사이의 깊은 연결을 보여주었습니다:

```
핵심 통찰: 구조화된 SSM은 반분리 행렬 곱셈(semi-separable matrix
          multiplication)의 한 형태와 동치

SSM 반복:
  h_k = A_k h_{k-1} + B_k x_k
  y_k = C_k h_k

행렬 곱셈으로 표현 가능:
  y = M x

여기서 M은 반분리 행렬:
  M_{ij} = C_i (∏_{k=j+1}^{i} A_k) B_j    i ≥ j인 경우
  M_{ij} = 0                                  i < j인 경우

이것은 인과적 어텐션(causal attention)과 구조적으로 유사!
```

### 5.2 SSD 알고리즘

```
Mamba-2 SSD 알고리즘:

1. 시퀀스를 크기 Q(예: 256)의 블록으로 분할
2. 각 청크 내부: 반분리 구조를 사용한 "미니 어텐션" 계산
3. 청크 간: 반복으로 상태 전파

복잡도: O(L * Q * N)  여기서 Q는 청크 크기
  Q = √L이면: 총 = O(L * √L * N) — 차이차(subquadratic)

이점:
  - 텐서 코어(행렬 곱셈 하드웨어) 활용 가능
  - Mamba-1 대비 GPU에서 2-8배 빠름
  - 명시적 다중 헤드 구조 (어텐션과 유사)
```

### 5.3 Mamba-2 개선 사항

```
Mamba-1 대 Mamba-2:

특성                    Mamba-1              Mamba-2
──────────────────────────────────────────────────────────
핵심 알고리즘            선택적 스캔           SSD (청크 단위)
다중 헤드 지원           아니오 (단일 헤드)    예 (어텐션과 유사)
GPU 활용               맞춤형 커널           텐서 코어
속도 (훈련)             ~1.0배               ~2-8배 빠름
상태 차원 (N)           16                   64-256
병렬성                 시퀀스 수준            블록 수준
```

---

## 6. SSM 대 트랜스포머 비교

### 6.1 계산 복잡도

```
연산              트랜스포머         SSM (Mamba)     비고
────────────────────────────────────────────────────────────
훈련 (단계당)     O(N²D)            O(NND_s)        D_s = 상태 차원
추론 (토큰당)     O(ND + KV)        O(D * D_s)      KV = KV 캐시 접근
메모리 (추론)     O(N * D * L)      O(D * D_s)      L = 레이어 수
프리필            O(N²)             O(N)            초기 프롬프트
1M에서 처리량     매우 낮음          높음            긴 컨텍스트 장점
```

### 6.2 품질 비교

```
과제 유형             트랜스포머      Mamba     설명
──────────────────────────────────────────────────────────────
언어 모델링           ★★★★★          ★★★★     비슷하지만 트랜스포머가 약간 우위
인컨텍스트 학습       ★★★★★          ★★★      어텐션이 검색에서 뛰어남
장거리 의존성         ★★★            ★★★★★    SSM 상태가 이력을 압축
복사/검색            ★★★★★          ★★       SSM이 정확한 회상에서 어려움
오디오/신호           ★★★            ★★★★★    SSM이 신호에 자연스러움
DNA/유전체학          ★★★            ★★★★★    매우 긴 시퀀스에서 SSM 유리
코드 생성            ★★★★★          ★★★★     구조에 어텐션이 도움
```

### 6.3 스케일링 행동

```
모델 크기       트랜스포머 PPL    Mamba PPL     비고
────────────────────────────────────────────────────────
125M             ~30.0              ~29.5        Mamba가 약간 좋음
350M             ~24.0              ~23.5        비슷
1.3B             ~18.5              ~18.2        비슷
2.8B             ~15.8              ~15.5        비슷

동일 파라미터 수에서 Mamba는 표준 언어 벤치마크(Pile, LAMBADA 등)에서
트랜스포머 품질에 필적하면서도 추론은 더 빠릅니다.
```

---

## 7. 하이브리드 아키텍처

### 7.1 하이브리드를 사용하는 이유

SSM과 트랜스포머는 상호 보완적인 강점을 가집니다:

```
결합:
  SSM 강점:  토큰당 O(1) 추론, 장거리, 신호 처리
  어텐션 강점: 정확한 검색, 인컨텍스트 학습, 복사

하이브리드 접근:
  모델 대부분에 SSM 레이어 사용 (저렴, 압축에 좋음)
  검색 과제를 위해 어텐션 레이어를 간간이 배치 (비싸지만 정확)
```

### 7.2 Jamba (AI21)

```
Jamba 아키텍처 (AI21, 2024):

총: 52B 파라미터 (MoE로 12B 활성)

레이어 구성:
  ┌──────────────────────────────────────┐
  │  Mamba 레이어    ← 대부분의 레이어    │
  │  Mamba 레이어                        │
  │  Mamba 레이어                        │
  │  Attention 레이어  ← 4번째마다        │
  │  Mamba 레이어                        │
  │  Mamba 레이어                        │
  │  Mamba 레이어                        │
  │  Attention + MoE 레이어              │
  │  ...                                │
  └──────────────────────────────────────┘

비율: Mamba 대 Attention 레이어 ~7:1
결과: 256K 컨텍스트, 단일 80GB GPU에 적합
```

### 7.3 기타 하이브리드 설계

```
모델              아키텍처                          컨텍스트    출시
──────────────────────────────────────────────────────────────────
Jamba             Mamba + Attention + MoE           256K       2024
Zamba             공유 어텐션 + Mamba                가변       2024
Griffin           게이트 선형 반복 + 어텐션           가변       2024 (Google)
RecurrentGemma    LRRL (선형 반복) + 어텐션           가변       2024 (Google)
StripedHyena      Hyena (합성곱) + Attention         가변       2023 (Together)
```

### 7.4 간단한 하이브리드 구현

```python
class HybridBlock(nn.Module):
    """Mamba 또는 Attention이 될 수 있는 블록."""

    def __init__(self, d_model, block_type="mamba", n_heads=8, d_state=16):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.block_type = block_type

        if block_type == "mamba":
            self.layer = MambaBlock(d_model, d_state=d_state)
        elif block_type == "attention":
            self.layer = nn.MultiheadAttention(
                d_model, n_heads, batch_first=True
            )
        else:
            raise ValueError(f"알 수 없는 블록 유형: {block_type}")

    def forward(self, x, mask=None):
        residual = x
        x = self.norm(x)

        if self.block_type == "mamba":
            x = self.layer(x)
        else:
            x, _ = self.layer(x, x, x, attn_mask=mask)

        return residual + x


class HybridModel(nn.Module):
    """하이브리드 Mamba-Attention 모델 (Jamba 스타일)."""

    def __init__(self, vocab_size, d_model=768, n_layers=24, n_heads=8,
                 d_state=16, attn_every_n=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)

        self.layers = nn.ModuleList()
        for i in range(n_layers):
            if (i + 1) % attn_every_n == 0:
                block_type = "attention"
            else:
                block_type = "mamba"
            self.layers.append(
                HybridBlock(d_model, block_type, n_heads, d_state)
            )

        self.norm_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.norm_f(x)
        return self.lm_head(x)
```

---

## 8. 구현 세부 사항과 훈련

### 8.1 SSM 훈련 팁

```
하이퍼파라미터            권장 값                    비고
──────────────────────────────────────────────────────────────
학습률                  3e-4 ~ 8e-4               트랜스포머와 유사
가중치 감쇠              0.1                       표준
옵티마이저               AdamW                     트랜스포머와 동일
웜업 단계               총 단계의 1-2%             표준 웜업
상태 차원 (N)           16 (Mamba-1), 64+ (Mamba-2) 큰 N = 더 많은 메모리
확장 팩터               2                         내부 차원 = 2 * d_model
합성곱 커널 크기         4                         로컬 컨텍스트 윈도우
초기화                  A: HiPPO 또는 로그 간격     성능에 중요
                       B, C: 작은 랜덤
                       Δ: U(0.001, 0.1)의 역softplus
```

### 8.2 긴 시퀀스 훈련

```python
def train_ssm_long_context(model, dataloader, optimizer, max_length=65536):
    """긴 컨텍스트 SSM 모델의 훈련 루프."""
    model.train()

    for batch in dataloader:
        input_ids = batch['input_ids']  # (B, L), L이 매우 길 수 있음
        labels = batch['labels']

        # SSM은 메모리 문제 없이 긴 시퀀스 처리 가능
        # SSM 레이어에서는 기울기 체크포인팅 불필요
        # (어텐션 메모리가 이차적으로 증가하는 트랜스포머와 달리)
        logits = model(input_ids)
        loss = F.cross_entropy(
            logits[:, :-1].reshape(-1, logits.size(-1)),
            labels[:, 1:].reshape(-1)
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    return loss.item()
```

### 8.3 효율적 추론

```python
class MambaInferenceCache:
    """효율적인 Mamba 자기회귀 추론을 위한 캐시."""

    def __init__(self, n_layers, d_inner, d_state, d_conv, device):
        self.n_layers = n_layers
        # SSM 상태: 시퀀스 길이와 관계없이 상수 크기!
        self.ssm_states = [
            torch.zeros(1, d_inner, d_state, device=device)
            for _ in range(n_layers)
        ]
        # 합성곱 상태: 작은 슬라이딩 윈도우
        self.conv_states = [
            torch.zeros(1, d_inner, d_conv, device=device)
            for _ in range(n_layers)
        ]

    def memory_usage(self):
        """트랜스포머 KV 캐시와 비교."""
        ssm_mem = sum(s.numel() * 2 for s in self.ssm_states)  # float16
        conv_mem = sum(s.numel() * 2 for s in self.conv_states)
        total = ssm_mem + conv_mem
        return total  # 상수! 시퀀스 길이에 따라 증가하지 않음
```

---

## 9. 연습문제

### 연습문제 1: 기본 SSM 레이어

기본 SSM을 구현하고 반복 모드와 합성곱 모드의 동치성을 확인하세요:

```python
"""
연습문제 1: SSM 모드 동치성.

과제:
1. 반복 및 합성곱 순전파를 모두 갖춘 BasicSSM 클래스 구현
2. 랜덤 입력 시퀀스 생성
3. 두 모드가 동일한 출력을 생성하는지 확인 (부동소수점 허용 오차 내)
4. 다양한 시퀀스 길이(100, 1000, 10000, 100000)에서 두 모드를 벤치마크
5. 실행 시간 비교를 플로팅

예상: 합성곱 모드가 긴 시퀀스에서 더 빠름 (O(L log L) vs O(L*N))
"""

def verify_ssm_equivalence():
    d_model = 64
    state_dim = 16
    ssm = BasicSSM(d_model, state_dim)

    for L in [100, 1000, 10000]:
        x = torch.randn(1, L, d_model)
        y_rec = ssm.forward_recurrent(x)
        y_conv = ssm.forward_conv(x)

        diff = (y_rec - y_conv).abs().max().item()
        print(f"L={L}: 최대 차이 = {diff:.2e}")
        assert diff < 1e-4, f"L={L}에서 모드 발산!"

# TODO: 검증 및 벤치마크 실행
```

### 연습문제 2: 선택적 SSM

Mamba 선택 메커니즘을 구현하고 LTI SSM 대비 장점을 보이세요:

```python
"""
연습문제 2: 필터링 과제에서 선택적 vs 비선택적 SSM.

과제: 특별한 "마커" 토큰이 있는 랜덤 토큰 시퀀스가 주어졌을 때,
모델은 마커 뒤에 오는 토큰만 출력해야 합니다.

예시:
  입력:   [a, b, MARKER, c, d, MARKER, e, f]
  타겟:   [0, 0, 0,      c, 0, 0,      e, 0]

이것은 내용 의존적 필터링이 필요 — LTI SSM으로는 불가능!

1. 비선택적 (LTI) SSM 기준선 구현
2. 선택적 SSM (Mamba 스타일) 구현
3. 두 모델을 필터링 과제에서 훈련
4. 선택적 SSM만 학습할 수 있음을 보이세요

예상: LTI SSM ~50% 정확도, 선택적 SSM ~99%+ 정확도
"""

def generate_filtering_data(batch_size, seq_len, vocab_size=32, marker_id=0):
    """선택적 필터링 과제를 위한 데이터 생성."""
    # TODO: 랜덤 마커 위치가 있는 입력 시퀀스 생성
    # TODO: 타겟 생성: 마커 뒤 토큰 복사, 그 외 0
    pass

# TODO: 두 모델 훈련 및 비교
```

### 연습문제 3: 하이브리드 모델

간단한 하이브리드 Mamba-Attention 모델을 구축하고 평가하세요:

```python
"""
연습문제 3: 하이브리드 모델 비교.

과제:
1. 동일한 파라미터 수의 세 모델 생성:
   a. 순수 트랜스포머 (모든 어텐션 레이어)
   b. 순수 Mamba (모든 Mamba 레이어)
   c. 하이브리드 (4번째마다 어텐션, 나머지 Mamba)

2. 세 모델 모두 간단한 언어 모델링 과제(예: WikiText-2)에서 훈련
3. 비교:
   - 훈련 손실 곡선
   - 다양한 시퀀스 길이에서의 추론 속도
   - 추론 중 메모리 사용량

4. 검색 과제에서 테스트: "라커 42의 열쇠는 파란색이다. ...
   [긴 컨텍스트] ... 라커 42의 열쇠는 무슨 색인가?"

예상:
  - 순수 트랜스포머: 최고 검색, 가장 느린 추론
  - 순수 Mamba: 가장 빠른 추론, 최악의 검색
  - 하이브리드: 양쪽의 좋은 균형
"""

# TODO: 모델 구축, 훈련 및 평가
```

### 연습문제 4: 시계열을 위한 SSM

시계열 예측 과제에 SSM을 적용하세요:

```python
"""
연습문제 4: 시계열 예측을 위한 SSM.

과제:
1. 다변량 시계열 데이터셋 생성 또는 로드
   (예: 서로 다른 주파수의 합성 사인파)
2. SSM 기반 예측 모델 구축
3. 트랜스포머 기준선과 비교
4. 증가하는 길이의 시퀀스에서 평가 (256, 1024, 4096, 16384)
5. 정확도와 추론 시간 모두 측정

시작 코드:
"""

class SSMForecaster(nn.Module):
    def __init__(self, input_dim, d_model=128, n_layers=4,
                 state_dim=16, forecast_horizon=96):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.ssm_layers = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(d_model),
                BasicSSM(d_model, state_dim)
            )
            for _ in range(n_layers)
        ])
        self.output_proj = nn.Linear(d_model, input_dim * forecast_horizon)
        self.forecast_horizon = forecast_horizon
        self.input_dim = input_dim

    def forward(self, x):
        # x: (B, L, input_dim)
        h = self.input_proj(x)
        for layer in self.ssm_layers:
            h = h + layer(h)
        # 예측을 위해 마지막 은닉 상태 사용
        out = self.output_proj(h[:, -1, :])
        return out.view(-1, self.forecast_horizon, self.input_dim)

# TODO: 데이터 생성, 훈련 및 트랜스포머와 비교
```

---

**이전**: [확산 모델 심화](./45_Diffusion_Models_Advanced.md) | **다음**: [전문가 혼합 모델](./47_Mixture_of_Experts.md)

---

*레슨 46 끝*
