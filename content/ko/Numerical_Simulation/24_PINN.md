[이전: GPU 가속](./23_GPU_Acceleration.md)

---

# 24. 물리 정보 신경망(Physics-Informed Neural Networks, PINNs)

> **사전 지식**: 신경망에 대한 기본적인 이해([딥러닝 1~3강](../Deep_Learning/01_PyTorch_Basics.md))와 PDE(본 토픽 6~10강)가 필요합니다.

## 학습 목표

이 강의를 완료하면 다음을 수행할 수 있습니다:

1. PINN 프레임워크를 설명한다: 신경망 손실 함수에 물리(PDE)를 내재화하는 방식
2. 1D 열 방정식을 풀기 위한 PINN을 처음부터 구현한다
3. 데이터, PDE 잔차, 경계 조건을 균형 있게 결합하는 복합 손실 함수를 설계한다
4. 순방향 문제(PDE 풀기)와 역문제(매개변수 탐색)에 PINN을 훈련시킨다
5. PINN의 한계를 평가하고 전통적인 수치 방법과 비교한다

---

## 목차

1. [동기: 물리와 ML의 연결](#1-동기-물리와-ml의-연결)
2. [PINN 아키텍처](#2-pinn-아키텍처)
3. [손실 함수 설계](#3-손실-함수-설계)
4. [구현: 1D 열 방정식](#4-구현-1d-열-방정식)
5. [역문제](#5-역문제)
6. [고급 주제](#6-고급-주제)
7. [연습문제](#7-연습문제)

---

## 1. 동기: 물리와 ML의 연결

### 1.1 전통적 PDE 풀이 vs 신경망 방식

```
전통적 수치 방법:                      신경망 방식:
┌──────────────────────────┐           ┌──────────────────────────┐
│ • 도메인 이산화           │           │ • NN이 해 함수를 근사하도록│
│ • 시스템 구성             │           │   훈련                    │
│ • 선형 시스템 풀기         │           │ • 메쉬 불필요             │
│                          │           │ • 미분 가능 (자동미분)     │
│ 장점: 정확, 검증됨         │           │                          │
│ 단점: 메쉬 의존적,         │           │ 장점: 메쉬 불필요, 고차원  │
│       차원의 저주          │           │       처리 가능           │
│                          │           │ 단점: 훈련 비용,          │
│                          │           │       정확도 문제         │
└──────────────────────────┘           └──────────────────────────┘
```

### 1.2 PINN의 아이디어 (Raissi et al., 2019)

PINN은 **데이터**와 **물리** 두 가지를 모두 만족하도록 훈련된 신경망입니다:

```
표준 ML:                               PINN:

  Loss = Σ (NN(x) - y_data)²          Loss = L_data + L_pde + L_bc

  데이터만 맞춥니다.                    L_data: 관측 데이터에 맞춤
  물리적 제약 없음.                     L_pde:  지배 PDE를 만족
  비물리적 결과가 나올 수 있음.          L_bc:   경계 조건을 만족

  PDE는 정칙화(regularizer) 역할을 하여,  희소 데이터가 있어도 물리가
  해가 물리적으로 일관되도록             해를 물리적으로 의미있게
  제약을 가합니다.                       제약합니다.
```

### 1.3 PINN을 사용하는 시점

| 시나리오 | 전통적 방법 | PINN | 권장 |
|---------|-----------|------|------|
| 순방향 PDE (조건 완비) | 빠름, 정확 | 느림, 덜 정확 | 전통적 방법 |
| 희소/잡음 데이터 + 알려진 물리 | 데이터 활용 불가 | 둘 다 결합 | PINN |
| 역문제 (매개변수 식별) | 별도 최적화 필요 | 자연스러운 프레임워크 | PINN |
| 고차원 PDE (> 3D) | 차원의 저주 | 잘 처리 | PINN |
| 복잡한 기하, 메쉬 없음 | 메쉬 생성 어려움 | 메쉬 불필요 | PINN |

---

## 2. PINN 아키텍처

### 2.1 네트워크 구조

```
입력: (x, t)                      출력: u(x, t)
  좌표                               해 값

  ┌───────┐    ┌─────────┐    ┌─────────┐    ┌───────┐
  │ (x,t) │───►│ Hidden  │───►│ Hidden  │───►│ u(x,t)│
  │       │    │ Layer 1 │    │ Layer 2 │    │       │
  │       │    │ (tanh)  │    │ (tanh)  │    │       │
  └───────┘    └─────────┘    └─────────┘    └───────┘
       │                                          │
       │    자동 미분(Automatic Differentiation)   │
       │    ┌──────────────────────────┐          │
       └───►│  ∂u/∂t, ∂u/∂x, ∂²u/∂x² │◄─────────┘
            │  (자동미분으로 계산)       │
            └──────────────────────────┘
                       │
                       ▼
              PDE 잔차: f = ∂u/∂t - α∂²u/∂x²
              목표: 어디서든 f ≈ 0
```

### 2.2 핵심 통찰: 자동 미분

신경망 u(x,t;θ)는 입력 (x,t)에 대해 미분 가능합니다. 즉, 유한 차분 없이도 자동 미분을 통해 ∂u/∂t, ∂u/∂x, ∂²u/∂x²을 정확하게 계산할 수 있습니다.

```python
import numpy as np


class SimpleNN:
    """PINN을 위한 최소한의 완전 연결 신경망.

    아키텍처: input(2) → hidden(n) → hidden(n) → output(1)
    활성화 함수: tanh (부드럽고, 미분이 잘 동작함)
    """

    def __init__(self, hidden_size=32, seed=42):
        rng = np.random.RandomState(seed)
        # Xavier 초기화
        s1 = np.sqrt(2.0 / 2)
        s2 = np.sqrt(2.0 / hidden_size)

        self.W1 = rng.randn(2, hidden_size) * s1
        self.b1 = np.zeros(hidden_size)
        self.W2 = rng.randn(hidden_size, hidden_size) * s2
        self.b2 = np.zeros(hidden_size)
        self.W3 = rng.randn(hidden_size, 1) * s2
        self.b3 = np.zeros(1)

    def forward(self, x_t):
        """순전파: (x, t) → u(x, t)."""
        self.z1 = x_t @ self.W1 + self.b1
        self.h1 = np.tanh(self.z1)
        self.z2 = self.h1 @ self.W2 + self.b2
        self.h2 = np.tanh(self.z2)
        self.u = self.h2 @ self.W3 + self.b3
        return self.u

    def get_params(self):
        return [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3]
```

---

## 3. 손실 함수 설계

### 3.1 복합 손실

PINN 손실은 세 가지 요소로 구성됩니다:

```
L_total = λ_data · L_data + λ_pde · L_pde + λ_bc · L_bc

L_data = (1/N_d) Σ |u_NN(x_i, t_i) - u_observed_i|²
  → 관측 데이터 포인트에 맞춤 (데이터가 있는 경우)

L_pde = (1/N_r) Σ |f(x_j, t_j)|²
  → 배치점(collocation point)에서 PDE 잔차는 0이어야 함
  → f = ∂u/∂t - α·∂²u/∂x² (열 방정식의 경우)

L_bc = (1/N_b) Σ |u_NN(x_k, t_k) - u_bc_k|²
  → 경계 조건 만족

배치점(Collocation points): 도메인 내에서 PDE를 강제하는 무작위 또는
준난수(quasi-random) 포인트 (메쉬 불필요!)
```

### 3.2 배치점(Collocation Points)

```
도메인: x ∈ [0, 1], t ∈ [0, 1]

  t
  1 ┌──────────────────────────────┐
    │  ·  ·    ·   ·  ·   ·  ·   │ ← 내부 배치점 (L_pde)
    │    ·  ·  ·   ·  ·     ·    │
    │  ·    ·    ·   ·    ·  ·   │
    │    ·  ·  ·   ·    ·    ·   │
    │  ·  ·    ·   ·  ·   ·     │
  0 ├──●──●──●──●──●──●──●──●───┤
    │  ↑                         │
    │  초기 조건 (L_bc)           │
    ●                            ●  ← 경계 조건 (L_bc)
    ●                            ●
    ●                            ●
    └──────────────────────────────┘
    0                            1  x

  격자 구조 불필요 — 포인트는 무작위로 배치 가능.
```

### 3.3 손실 가중치

손실 요소 간 균형을 맞추는 것이 중요합니다. 일반적인 전략:

| 전략 | 설명 |
|------|------|
| 고정 가중치 | λ_data=1, λ_pde=1, λ_bc=10 (수동 조정) |
| 적응형 (GradNorm) | 각 손실 요소의 기울기를 정규화 |
| 커리큘럼 | λ_bc를 크게 시작 (경계 조건 먼저 학습), 이후 λ_pde 증가 |
| 자기 적응형 | λ를 훈련 가능한 매개변수로 학습 |

---

## 4. 구현: 1D 열 방정식

### 4.1 문제 설정

```
열 방정식: ∂u/∂t = α · ∂²u/∂x²

도메인: x ∈ [0, 1], t ∈ [0, 1]
초기 조건(IC): u(x, 0) = sin(πx)
경계 조건(BC): u(0, t) = u(1, t) = 0

해석적 해: u(x, t) = sin(πx) · exp(-α·π²·t)
```

### 4.2 PINN 구현

```python
class HeatPINN:
    """순수 NumPy로 유한 차분 미분을 사용하는 1D 열 방정식 PINN.

    실제 자동 미분 없이 수치 미분을 사용합니다 (순수 NumPy 구현이므로).
    실전에서는 PyTorch 또는 JAX의 자동 미분을 사용하세요.

    ∂u/∂t = α · ∂²u/∂x²
    """

    def __init__(self, nn, alpha=0.01, n_colloc=1000, n_bc=100,
                 n_ic=100):
        self.nn = nn
        self.alpha = alpha

        rng = np.random.RandomState(0)

        # 내부 배치점
        self.x_colloc = rng.uniform(0.01, 0.99, (n_colloc, 1))
        self.t_colloc = rng.uniform(0.01, 0.99, (n_colloc, 1))

        # 초기 조건 포인트 (t=0)
        self.x_ic = rng.uniform(0, 1, (n_ic, 1))
        self.t_ic = np.zeros((n_ic, 1))
        self.u_ic = np.sin(np.pi * self.x_ic)

        # 경계 조건 포인트 (x=0 및 x=1)
        self.t_bc = rng.uniform(0, 1, (n_bc, 1))
        self.x_bc_left = np.zeros((n_bc, 1))
        self.x_bc_right = np.ones((n_bc, 1))
        self.u_bc = np.zeros((n_bc, 1))  # 경계에서 u=0

    def compute_pde_residual(self, eps=1e-4):
        """유한 차분 근사를 사용한 PDE 잔차 계산.

        f = ∂u/∂t - α · ∂²u/∂x²

        자동미분을 사용하는 실제 PINN에서는 미분이 정확합니다.
        여기서는 중앙 차분으로 근사합니다.
        """
        x = self.x_colloc
        t = self.t_colloc
        xt = np.hstack([x, t])

        u = self.nn.forward(xt)

        # 유한 차분으로 ∂u/∂t 계산
        xt_tp = np.hstack([x, t + eps])
        xt_tm = np.hstack([x, t - eps])
        du_dt = (self.nn.forward(xt_tp) - self.nn.forward(xt_tm)) / (2*eps)

        # 유한 차분으로 ∂²u/∂x² 계산
        xt_xp = np.hstack([x + eps, t])
        xt_xm = np.hstack([x - eps, t])
        d2u_dx2 = (self.nn.forward(xt_xp) - 2*u
                    + self.nn.forward(xt_xm)) / eps**2

        # PDE 잔차
        residual = du_dt - self.alpha * d2u_dx2
        return residual

    def compute_loss(self):
        """복합 PINN 손실: L_pde + L_ic + L_bc."""
        # PDE 잔차 손실
        residual = self.compute_pde_residual()
        L_pde = np.mean(residual**2)

        # 초기 조건 손실
        xt_ic = np.hstack([self.x_ic, self.t_ic])
        u_pred_ic = self.nn.forward(xt_ic)
        L_ic = np.mean((u_pred_ic - self.u_ic)**2)

        # 경계 조건 손실
        xt_left = np.hstack([self.x_bc_left, self.t_bc])
        xt_right = np.hstack([self.x_bc_right, self.t_bc])
        u_left = self.nn.forward(xt_left)
        u_right = self.nn.forward(xt_right)
        L_bc = np.mean(u_left**2) + np.mean(u_right**2)

        # 가중 합산
        L_total = L_pde + 10.0 * L_ic + 10.0 * L_bc
        return L_total, L_pde, L_ic, L_bc

    def analytical_solution(self, x, t):
        """비교를 위한 해석적 해."""
        return np.sin(np.pi * x) * np.exp(-self.alpha * np.pi**2 * t)
```

### 4.3 랜덤 탐색을 이용한 훈련 (단순화 버전)

```python
def train_pinn_simple(pinn, iterations=5000, lr=0.001):
    """매개변수 섭동을 이용한 PINN 훈련 (단순화).

    실전에서는 자동 미분(PyTorch/JAX)을 사용한
    기울기 기반 최적화(Adam)를 사용합니다.
    여기서는 시연을 위해 랜덤 섭동을 사용합니다.
    """
    params = pinn.nn.get_params()
    best_loss = float('inf')
    best_params = [p.copy() for p in params]

    for it in range(iterations):
        # 매개변수 섭동
        for p in params:
            p += np.random.randn(*p.shape) * lr

        loss, l_pde, l_ic, l_bc = pinn.compute_loss()

        if loss < best_loss:
            best_loss = loss
            best_params = [p.copy() for p in params]
        else:
            # 되돌리기
            for p, bp in zip(params, best_params):
                p[:] = bp

        if (it + 1) % 500 == 0:
            print(f"  Iter {it+1}: loss={best_loss:.6f} "
                  f"(pde={l_pde:.6f}, ic={l_ic:.6f}, bc={l_bc:.6f})")

    return best_loss
```

---

## 5. 역문제

### 5.1 역문제 프레임워크

알려진 매개변수로 PDE를 푸는 대신, 관측 데이터로부터 미지의 매개변수를 탐색합니다:

```
순방향 문제:
  알려진 것: PDE, 경계 조건, 매개변수 (α)
  미지의 것: 해 u(x, t)

역문제:
  알려진 것: PDE 구조, u의 일부 관측값
  미지의 것: 매개변수 α (예: 열 확산율)

역문제를 위한 PINN:
  L = L_data + L_pde(α) + L_bc
  α는 훈련 가능한 매개변수 (네트워크 가중치와 동일)
  최적화기가 u와 α를 동시에 학습합니다!
```

### 5.2 매개변수 탐색 예제

```python
class InversePINN:
    """역문제용 PINN: 열 확산율 α 탐색.

    시간에 따른 온도의 희소 관측값이 주어졌을 때,
    재료의 열 확산율 α를 찾습니다.
    """

    def __init__(self, nn, x_data, t_data, u_data):
        self.nn = nn
        self.x_data = x_data
        self.t_data = t_data
        self.u_data = u_data

        # α는 미지수 — 초기 추정값으로 시작
        self.log_alpha = np.log(0.05)  # log 변환으로 양수 보장

    @property
    def alpha(self):
        return np.exp(self.log_alpha)

    def compute_loss(self):
        """역문제 손실: 데이터 맞춤 + PDE 만족."""
        # 데이터 손실
        xt_data = np.hstack([self.x_data, self.t_data])
        u_pred = self.nn.forward(xt_data)
        L_data = np.mean((u_pred - self.u_data)**2)

        # PDE 잔차 (현재 α 추정값 사용)
        # 순방향 문제와 동일하나 α가 훈련 가능
        # L_pde = ... (현재 self.alpha로 계산)

        return L_data  # 전체 구현에서는 + L_pde
```

### 5.3 역 PINN의 응용

| 응용 분야 | 알려진 것 | 미지의 것 |
|----------|---------|---------|
| 재료 특성 분석 | 온도 측정값 | 열 확산율 |
| 유체 역학 | 속도장 (PIV) | 점성, 압력장 |
| 구조 건전성 모니터링 | 진동 데이터 | 강성 (손상 감지) |
| 역학(Epidemiology) | 확진자 수 | SIR 모델 매개변수 |

---

## 6. 고급 주제

### 6.1 훈련 상의 어려움

| 어려움 | 원인 | 해결책 |
|--------|-----|--------|
| 수렴 실패 | 손실 불균형 | 적응형 손실 가중치 |
| 느린 훈련 | 조건 불량 문제 | 학습률 스케줄링, 푸리에 특징(Fourier features) |
| 급격한 기울기 / 충격파 | tanh 활성화가 너무 부드러움 | sin 활성화 사용 (SIREN) |
| 고차원 PDE | 차원의 저주 | 잔차 연결(Residual connections), 도메인 분해 사용 |

### 6.2 DeepXDE 라이브러리

```python
# DeepXDE는 PINN을 위한 고수준 API를 제공합니다
# (참고용 — 설치 필요)

# import deepxde as dde
#
# def pde(x, y):
#     dy_t = dde.grad.jacobian(y, x, i=0, j=1)
#     dy_xx = dde.grad.hessian(y, x, i=0, j=0)
#     return dy_t - 0.01 * dy_xx
#
# geom = dde.geometry.Interval(0, 1)
# timedomain = dde.geometry.TimeDomain(0, 1)
# geomtime = dde.geometry.GeometryXTime(geom, timedomain)
#
# bc = dde.icbc.DirichletBC(geomtime, lambda x: 0, ...)
# ic = dde.icbc.IC(geomtime, lambda x: np.sin(np.pi * x[:, 0:1]), ...)
#
# data = dde.data.TimePDE(geomtime, pde, [bc, ic], num_domain=2540)
# net = dde.nn.FNN([2] + [32]*3 + [1], "tanh", "Glorot uniform")
# model = dde.Model(data, net)
# model.compile("adam", lr=0.001)
# model.train(epochs=10000)
```

### 6.3 PINN vs 전통적 방법

| 측면 | FEM/FDM | PINN |
|------|---------|------|
| 정확도 | 높음 (오차 범위가 잘 연구됨) | 중간 (훈련에 따라 다름) |
| 속도 (소규모 PDE) | 빠름 | 느림 (훈련 오버헤드) |
| 속도 (고차원 PDE) | 차원의 저주 | 잘 처리 |
| 메쉬 | 필요 | 메쉬 불필요 |
| 역문제 | 별도 최적화 필요 | 프레임워크에 내장 |
| 데이터 동화(Data Assimilation) | 부가적 | 자연스러움 |
| 신뢰성 | 수렴 이론 검증됨 | 수렴 보장 없음 |

---

## 7. 연습문제

### 연습문제 1: 1D 열 방정식용 PINN

열 방정식을 위한 PINN을 구현하세요:
1. 도메인: x ∈ [0,1], t ∈ [0,1], α = 0.01
2. 초기 조건: u(x,0) = sin(πx), 경계 조건: u(0,t) = u(1,t) = 0
3. 내부 배치점 500개, 경계 조건 포인트 50개, 초기 조건 포인트 50개 사용
4. 훈련 후 해석적 해와 비교
5. 도메인 전체에 대한 오차 분포를 그래프로 그리세요

### 연습문제 2: 손실 요소 분석

손실 가중치의 효과를 연구하세요:
1. λ_pde = 1, λ_bc = {0.1, 1, 10, 100}으로 훈련
2. 각 경우에 대해 최종 해와 요소별 손실 곡선을 그리세요
3. 전체 오차를 최소화하는 최적 λ_bc를 찾으세요
4. λ_bc가 너무 낮으면 경계 위반이 발생하고, 너무 높으면 PDE 수렴이 느려지는 이유를 설명하세요

### 연습문제 3: 배치점 연구

배치점(collocation point) 전략을 조사하세요:
1. 균등 무작위 vs 라틴 하이퍼큐브 샘플링(Latin Hypercube Sampling) vs 정규 격자
2. 포인트 수: 100, 500, 1000, 5000
3. 각 전략과 포인트 수에 대한 정확도 비교
4. 1% 상대 오차를 달성하는 최소 포인트 수를 결정하세요

### 연습문제 4: 역문제

관측값으로부터 확산율 α를 탐색하세요:
1. α = 0.02인 해석적 해에서 "관측" 데이터 생성
2. 관측값에 5% 가우시안 잡음 추가
3. 희소한 관측 포인트 20개만 사용
4. 역 PINN을 훈련하여 α를 복원
5. 추정된 α가 실제 값에 얼마나 가까운지 확인하세요

### 연습문제 5: 버거스 방정식 PINN

PINN으로 점성 버거스(Burgers') 방정식을 풀어보세요:
1. ∂u/∂t + u·∂u/∂x = ν·∂²u/∂x², ν = 0.01/π
2. 초기 조건: u(x,0) = -sin(πx), 경계 조건: u(-1,t) = u(1,t) = 0
3. 이 방정식은 급격한 기울기(충격파 유사)를 형성하여 PINN에 더 어렵습니다
4. PINN 해를 기준 유한 차분 해와 비교
5. PINN이 급격한 기울기에 어려움을 겪는 이유를 논의하세요

---

*24강 끝*
