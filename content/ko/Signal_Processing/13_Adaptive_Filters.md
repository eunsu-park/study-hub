# 13. 적응 필터

**이전**: [12. 다중 레이트 신호 처리](./12_Multirate_Signal_Processing.md) | **다음**: [14. 시간-주파수 분석](./14_Time_Frequency_Analysis.md)

---

적응 필터(adaptive filter)는 최적화 알고리즘에 따라 계수가 자동으로 조정되는 필터입니다. 신호 및 잡음 통계에 대한 완전한 사전 지식을 바탕으로 설계된 고정 필터와 달리, 적응 필터는 데이터로부터 지속적으로 파라미터를 갱신함으로써 미지(未知) 또는 시변(time-varying) 환경에서도 동작할 수 있습니다. 이는 노이즈 캔슬링 헤드폰, 전화기의 에코 제거, 모뎀의 채널 등화(channel equalization) 등 수많은 실제 시스템의 핵심 기술입니다.

**난이도**: ⭐⭐⭐⭐

**선수 지식**: FIR/IIR 필터 설계, 선형대수, 기본 최적화 개념

**학습 목표**:
- 위너 필터(Wiener filter)를 최적 MMSE 선형 필터로 유도
- 최급강하법(method of steepest descent)과 수렴 특성 이해
- LMS 알고리즘 유도 및 구현, 수렴 동작 분석
- 향상된 수렴을 위한 정규화 LMS(Normalized LMS, NLMS) 구현
- 행렬 역 보조정리(matrix inversion lemma)를 이용한 RLS 알고리즘 유도 및 구현
- 복잡도, 수렴, 추적 측면에서 LMS와 RLS 비교
- 시스템 식별, 잡음 제거, 에코 제거, 등화에 적응 필터 적용

---

## 목차

1. [적응 필터링이 필요한 이유](#1-적응-필터링이-필요한-이유)
2. [위너 필터: 최적 MMSE 해](#2-위너-필터-최적-mmse-해)
3. [최급강하법](#3-최급강하법)
4. [LMS 알고리즘](#4-lms-알고리즘)
5. [LMS 수렴 분석](#5-lms-수렴-분석)
6. [정규화 LMS (NLMS)](#6-정규화-lms-nlms)
7. [RLS 알고리즘](#7-rls-알고리즘)
8. [비교: LMS vs RLS](#8-비교-lms-vs-rls)
9. [응용: 시스템 식별](#9-응용-시스템-식별)
10. [응용: 잡음 제거](#10-응용-잡음-제거)
11. [응용: 에코 제거](#11-응용-에코-제거)
12. [응용: 채널 등화](#12-응용-채널-등화)
13. [응용: 적응 빔포밍](#13-응용-적응-빔포밍)
14. [Python 구현: 완전한 적응 필터링 툴킷](#14-python-구현-완전한-적응-필터링-툴킷)
15. [연습 문제](#15-연습-문제)
16. [요약](#16-요약)
17. [참고 문헌](#17-참고-문헌)

---

## 1. 적응 필터링이 필요한 이유

### 1.1 고정 필터의 한계

기존 FIR 및 IIR 필터는 설계 시점에 신호 및 잡음 특성에 대한 완전한 지식을 필요로 합니다. 다음과 같은 경우를 고려해 보세요:

- **통계를 모를 때**: 잡음의 스펙트럼 특성을 모르면 최적 필터를 설계할 수 없습니다.
- **통계가 시변일 때**: 무선 채널은 송수신기의 이동에 따라 변합니다. 한 채널 실현(realization)에 맞게 설계된 필터는 잠시 후 준최적(suboptimal)이 됩니다.
- **실시간 동작이 필요할 때**: 일부 환경에서는 오프라인 설계 단계 없이 지속적인 적응이 필요합니다.

### 1.2 적응 필터링 프레임워크

적응 필터는 두 부분으로 구성됩니다:

1. **파라미터화된 필터 구조** (보통 FIR): 입력 $x(n)$으로부터 출력 $y(n)$을 계산합니다.
2. **적응 알고리즘**: 어떤 비용 함수를 최소화하도록 필터 계수 $\mathbf{w}(n)$을 조정합니다.

```
                    ┌──────────────────────┐
     x(n) ────────▶│   Adaptive Filter    │────────▶ y(n)
                    │   w(n)               │
                    └──────────┬───────────┘
                               │
                               │  e(n) = d(n) - y(n)
                               │
     d(n) ─────────────────────┴─────────▶ Error
     (desired signal)                      Computation
                                              │
                                              ▼
                                     Adaptation Algorithm
                                     (update w(n+1))
```

**오차 신호(error signal)**는:

$$e(n) = d(n) - y(n) = d(n) - \mathbf{w}^T(n) \mathbf{x}(n)$$

여기서:
- $d(n)$은 **원하는(기준) 신호(desired/reference signal)**
- $\mathbf{x}(n) = [x(n), x(n-1), \ldots, x(n-M+1)]^T$는 입력 벡터
- $\mathbf{w}(n) = [w_0(n), w_1(n), \ldots, w_{M-1}(n)]^T$는 필터 가중치 벡터
- $M$은 필터 차수

### 1.3 주요 구성

적응 필터는 네 가지 주요 구성으로 사용됩니다:

| 구성 | 입력 $x(n)$ | 원하는 신호 $d(n)$ | 목적 |
|------|------------|------------------|------|
| 시스템 식별 | 미지 시스템의 입력 | 미지 시스템의 출력 | 미지 시스템 모델링 |
| 역 모델링 | 미지 시스템의 출력 | 지연된 입력 | 채널 등화 |
| 잡음 제거 | 상관된 잡음 기준 | 신호 + 잡음 | 신호 추출 |
| 예측 | 신호의 지연 버전 | 현재 신호 | 미래 값 예측 |

---

## 2. 위너 필터: 최적 MMSE 해

### 2.1 비용 함수

**최소 평균 제곱 오차(minimum mean square error, MMSE)** 기준은 기대 제곱 오차를 최소화합니다:

$$J(\mathbf{w}) = E\left[|e(n)|^2\right] = E\left[|d(n) - \mathbf{w}^T \mathbf{x}(n)|^2\right]$$

전개하면:

$$J(\mathbf{w}) = E[d^2(n)] - 2\mathbf{w}^T E[d(n)\mathbf{x}(n)] + \mathbf{w}^T E[\mathbf{x}(n)\mathbf{x}^T(n)] \mathbf{w}$$

다음을 정의합니다:
- **자기상관 행렬(autocorrelation matrix)**: $\mathbf{R} = E[\mathbf{x}(n)\mathbf{x}^T(n)]$ ($M \times M$ 양정치 행렬)
- **상호상관 벡터(cross-correlation vector)**: $\mathbf{p} = E[d(n)\mathbf{x}(n)]$ ($M \times 1$ 벡터)
- $\sigma_d^2 = E[d^2(n)]$

비용 함수는 **2차 볼(quadratic bowl)** 형태가 됩니다:

$$J(\mathbf{w}) = \sigma_d^2 - 2\mathbf{w}^T \mathbf{p} + \mathbf{w}^T \mathbf{R} \mathbf{w}$$

### 2.2 위너-호프 방정식(Wiener-Hopf Equation)

기울기를 구하고 0으로 설정하면:

$$\nabla_{\mathbf{w}} J = -2\mathbf{p} + 2\mathbf{R}\mathbf{w} = \mathbf{0}$$

이로부터 **위너-호프 방정식(Wiener-Hopf equation)**(정규 방정식)이 도출됩니다:

$$\boxed{\mathbf{R}\mathbf{w}_{opt} = \mathbf{p}}$$

최적(위너) 필터는:

$$\mathbf{w}_{opt} = \mathbf{R}^{-1}\mathbf{p}$$

최적 해에서의 **최소 MSE**는:

$$J_{min} = \sigma_d^2 - \mathbf{p}^T \mathbf{R}^{-1} \mathbf{p}$$

### 2.3 성능 곡면

$\mathbf{R}$이 양정치이므로, 비용 함수 $J(\mathbf{w})$는 볼록 2차함수로 그릇 모양의 곡면(타원 포물면)을 형성합니다. 모든 하강 알고리즘은 유일한 전역 최솟값으로 수렴합니다.

고유분해 $\mathbf{R} = \mathbf{Q}\boldsymbol{\Lambda}\mathbf{Q}^T$를 사용하면, 회전 좌표계 $\mathbf{v} = \mathbf{Q}^T(\mathbf{w} - \mathbf{w}_{opt})$에서의 비용 함수는:

$$J(\mathbf{v}) = J_{min} + \sum_{k=0}^{M-1} \lambda_k v_k^2$$

여기서 $\lambda_k$는 $\mathbf{R}$의 고유값입니다. $J$의 등고선은 고유벡터 방향으로 정렬되고 고유값에 의해 크기가 결정되는 타원입니다.

### 2.4 위너 해의 한계

위너 필터는 다음을 필요로 합니다:
1. $\mathbf{R}$과 $\mathbf{p}$에 대한 지식 (2차 통계)
2. 신호의 정상성(stationarity)
3. $\mathbf{R}^{-1}$ 계산 ($O(M^3)$ 연산)

실제로 이 조건들은 정확히 만족되기 어렵기 때문에, 반복적이고 적응적인 접근법이 필요합니다.

---

## 3. 최급강하법

### 3.1 MSE 곡면에서의 경사 하강

위너-호프 방정식을 직접 풀지 않고, 경사 하강(gradient descent)을 통해 반복적으로 $\mathbf{w}_{opt}$에 도달할 수 있습니다:

$$\mathbf{w}(n+1) = \mathbf{w}(n) - \mu \nabla_{\mathbf{w}} J(n)$$

MSE 비용 함수의 진짜 기울기는:

$$\nabla_{\mathbf{w}} J = -2\mathbf{p} + 2\mathbf{R}\mathbf{w}(n)$$

따라서 갱신 규칙은:

$$\boxed{\mathbf{w}(n+1) = \mathbf{w}(n) + 2\mu\left(\mathbf{p} - \mathbf{R}\mathbf{w}(n)\right)}$$

이것이 **최급강하법(steepest descent)** 알고리즘입니다. 여전히 $\mathbf{R}$과 $\mathbf{p}$에 대한 지식이 필요하므로 진정한 적응형은 아닙니다.

### 3.2 수렴 분석

가중치 오차 벡터를 정의합니다: $\boldsymbol{\epsilon}(n) = \mathbf{w}(n) - \mathbf{w}_{opt}$

갱신식에 대입하면:

$$\boldsymbol{\epsilon}(n+1) = (\mathbf{I} - 2\mu\mathbf{R})\boldsymbol{\epsilon}(n)$$

고유분해 $\mathbf{R} = \mathbf{Q}\boldsymbol{\Lambda}\mathbf{Q}^T$를 사용하여, 회전 좌표 $\mathbf{v}(n) = \mathbf{Q}^T \boldsymbol{\epsilon}(n)$에서:

$$v_k(n+1) = (1 - 2\mu\lambda_k) v_k(n)$$

수렴을 위해서는 모든 $k$에 대해 $|1 - 2\mu\lambda_k| < 1$이 필요하며, 이는:

$$\boxed{0 < \mu < \frac{1}{\lambda_{max}}}$$

여기서 $\lambda_{max}$는 $\mathbf{R}$의 최대 고유값입니다.

### 3.3 수렴 속도와 고유값 분산

각 모드 $v_k$의 수렴 속도는 $|1 - 2\mu\lambda_k|$에 의해 결정됩니다. 각 모드에 대한 최적 스텝 크기는 $\mu_k = 1/(2\lambda_k)$이지만, 단일 $\mu$를 사용하므로:

- 가장 빠르게 수렴하는 모드는 $\lambda_{max}$에 해당
- 가장 느리게 수렴하는 모드는 $\lambda_{min}$에 해당

**고유값 분산(eigenvalue spread)**(조건수, condition number):

$$\chi(\mathbf{R}) = \frac{\lambda_{max}}{\lambda_{min}}$$

이것이 전체 수렴 속도를 지배합니다. 큰 고유값 분산은 느린 수렴을 의미합니다. 알고리즘이 성능 곡면의 좁은 계곡을 가로질러 "지그재그"로 움직이기 때문입니다.

### 3.4 학습 곡선

반복 횟수의 함수로서의 MSE가 **학습 곡선(learning curve)**입니다:

$$J(n) = J_{min} + \sum_{k=0}^{M-1} \lambda_k v_k^2(0)(1 - 2\mu\lambda_k)^{2n}$$

각 모드는 시상수(time constant)와 함께 기하급수적으로 감소합니다:

$$\tau_k = \frac{-1}{2\ln|1 - 2\mu\lambda_k|} \approx \frac{1}{4\mu\lambda_k} \quad \text{(소 } \mu \text{에 대해)}$$

가장 느린 모드의 시상수는 $\tau_{max} \approx 1/(4\mu\lambda_{min})$입니다.

---

## 4. LMS 알고리즘

### 4.1 유도

최급강하법 알고리즘은 진짜 기울기 $\nabla J = -2\mathbf{p} + 2\mathbf{R}\mathbf{w}(n)$을 필요로 합니다. Widrow와 Hoff (1960)의 핵심 통찰은 진짜 기울기를 **순간 추정값(instantaneous estimate)**으로 대체하는 것입니다:

$$\hat{\nabla} J(n) = -2e(n)\mathbf{x}(n)$$

이는 기댓값을 순간 샘플로 대체하여 얻어집니다:
- $\mathbf{R}\mathbf{w}(n) \approx \mathbf{x}(n)\mathbf{x}^T(n)\mathbf{w}(n) = \mathbf{x}(n)y(n)$
- $\mathbf{p} \approx d(n)\mathbf{x}(n)$

**LMS 알고리즘**은:

$$\boxed{\mathbf{w}(n+1) = \mathbf{w}(n) + \mu \, e(n) \, \mathbf{x}(n)}$$

여기서 $e(n) = d(n) - \mathbf{w}^T(n)\mathbf{x}(n)$.

### 4.2 알고리즘 요약

```
LMS Algorithm
─────────────────────────────────────────────
Initialize: w(0) = 0 (or small random values)
Parameters: step size μ, filter order M

For each new sample n = 0, 1, 2, ...
  1. Form input vector: x(n) = [x(n), x(n-1), ..., x(n-M+1)]^T
  2. Compute output:    y(n) = w^T(n) x(n)
  3. Compute error:     e(n) = d(n) - y(n)
  4. Update weights:    w(n+1) = w(n) + μ e(n) x(n)
```

**계산 복잡도**: 샘플당 $O(M)$ 곱셈과 덧셈 - 매우 효율적입니다.

### 4.3 LMS의 특성

1. **단순성**: 행렬 역산 없음, 자기상관 추정 없음
2. **낮은 복잡도**: 반복당 $2M$ 곱셈
3. **확률적 기울기**: 기울기 추정값에 잡음이 있지만 불편향(unbiased): $E[\hat{\nabla}J] = \nabla J$
4. **자기 조정**: 신호 통계의 느린 변화를 자동으로 추적

---

## 5. LMS 수렴 분석

### 5.1 평균 수렴

LMS 갱신의 기댓값 계산 (**독립성 가정** 하에 - $\mathbf{x}(n)$이 $\mathbf{w}(n)$에 독립):

$$E[\mathbf{w}(n+1)] = E[\mathbf{w}(n)] + \mu E[e(n)\mathbf{x}(n)]$$

대수 계산 후:

$$E[\boldsymbol{\epsilon}(n+1)] = (\mathbf{I} - 2\mu\mathbf{R}) E[\boldsymbol{\epsilon}(n)]$$

이는 최급강하법과 동일한 점화식이므로, **평균 수렴** 조건은:

$$0 < \mu < \frac{1}{\lambda_{max}}$$

실제로는 다음을 사용합니다:

$$0 < \mu < \frac{1}{\text{tr}(\mathbf{R})} = \frac{1}{M \cdot \sigma_x^2}$$

$\text{tr}(\mathbf{R}) = \sum_k \lambda_k \geq \lambda_{max}$이고, 정상 입력에 대해 $\text{tr}(\mathbf{R}) = M\sigma_x^2$이기 때문입니다.

### 5.2 평균 제곱 수렴

MSE가 수렴하기 위한 조건(평균 제곱 안정성)은 더 엄격합니다:

$$0 < \mu < \frac{2}{\lambda_{max} + \text{tr}(\mathbf{R})}$$

실용적으로 안전한 선택은:

$$\mu < \frac{1}{3 \, \text{tr}(\mathbf{R})} = \frac{1}{3M\sigma_x^2}$$

### 5.3 초과 MSE와 오조정

수렴 후에도 LMS 알고리즘은 확률적 기울기가 가중치 갱신에 잡음을 도입하기 때문에 $J_{min}$에 도달하지 못합니다. **초과 MSE(excess MSE)**는:

$$J_{excess} = J_{steady-state} - J_{min}$$

**오조정(misadjustment)**은:

$$\mathcal{M} = \frac{J_{excess}}{J_{min}} \approx \mu \, \text{tr}(\mathbf{R}) = \mu M \sigma_x^2$$

이는 근본적인 **트레이드오프**를 드러냅니다:
- **큰 $\mu$**: 빠른 수렴이지만 큰 오조정 (잡음이 많은 정상 상태)
- **작은 $\mu$**: 느린 수렴이지만 작은 오조정 (정확한 정상 상태)

### 5.4 스텝 크기 선택 지침

| 기준 | 스텝 크기 |
|------|----------|
| 안정성 (평균) | $\mu < 1/\lambda_{max}$ |
| 안정성 (평균 제곱) | $\mu < 2/(\lambda_{max} + \text{tr}(\mathbf{R}))$ |
| 실용적 규칙 | $\mu \in [0.01, 0.1] / (M \sigma_x^2)$ |
| 오조정 $\leq$ 10% | $\mu \leq 0.1 / (M \sigma_x^2)$ |

### 5.5 수렴 시간

가장 느린 모드의 근사 시상수는:

$$\tau_{mse} \approx \frac{1}{4\mu\lambda_{min}}$$

오조정 제약 $\mathcal{M} = \mu M \sigma_x^2$와 결합하면:

$$\tau_{mse} \approx \frac{M \sigma_x^2}{4\mathcal{M}\lambda_{min}} = \frac{\chi(\mathbf{R})}{4\mathcal{M}} \cdot \frac{M\sigma_x^2}{\lambda_{max}}$$

큰 고유값 분산 $\chi(\mathbf{R})$은 주어진 오조정에서 수렴하는 데 많은 반복이 필요함을 의미합니다.

---

## 6. 정규화 LMS (NLMS)

### 6.1 동기

표준 LMS는 고정 스텝 크기 $\mu$를 가지므로, 실제 적응 속도가 입력 전력 $\|\mathbf{x}(n)\|^2$에 따라 달라집니다. 입력 전력이 변할 때 LMS는 불안정해지거나 너무 느리게 수렴할 수 있습니다.

### 6.2 유도

NLMS 알고리즘은 스텝 크기를 입력 전력으로 정규화하여 얻어집니다:

$$\boxed{\mathbf{w}(n+1) = \mathbf{w}(n) + \frac{\tilde{\mu}}{\|\mathbf{x}(n)\|^2 + \delta} \, e(n) \, \mathbf{x}(n)}$$

여기서:
- $\tilde{\mu} \in (0, 2)$는 정규화된 스텝 크기
- $\delta > 0$는 0으로 나누는 것을 방지하는 작은 정규화 상수

### 6.3 제약 최적화로부터의 유도

NLMS는 다음 제약 최적화 문제를 풀어서 유도할 수 있습니다:

$$\min_{\mathbf{w}(n+1)} \|\mathbf{w}(n+1) - \mathbf{w}(n)\|^2 \quad \text{제약 조건:} \quad \mathbf{w}^T(n+1)\mathbf{x}(n) = d(n)$$

즉, 최신 데이터 포인트를 완벽하게 피팅하는 현재 가중치 벡터에 가장 가까운 가중치 벡터를 찾는 것입니다. 라그랑주 승수(Lagrange multipliers)를 사용하면 $\tilde{\mu} = 1$인 NLMS 갱신이 얻어집니다.

### 6.4 NLMS의 장점

1. **강건한 수렴**: 스텝 크기가 입력 전력에 자동 적응
2. **단순한 튜닝**: 설정할 파라미터가 $\tilde{\mu} \in (0, 2)$ 하나뿐
3. **비정상 입력에 적합**: 변동하는 신호 레벨에서도 잘 작동
4. **최소한의 추가 비용**: 반복당 한 번의 내적만 추가

### 6.5 NLMS 수렴

NLMS의 수렴 조건은 단순합니다:

$$0 < \tilde{\mu} < 2$$

오조정은 근사적으로:

$$\mathcal{M}_{NLMS} \approx \frac{\tilde{\mu}}{2 - \tilde{\mu}} \cdot \frac{1}{M}$$

일반적인 선택은 $\tilde{\mu} \in [0.1, 1.0]$입니다.

---

## 7. RLS 알고리즘

### 7.1 동기

LMS가 기울기를 확률적으로 추정하는(한 번에 한 샘플) 반면, **순환 최소 제곱(Recursive Least Squares, RLS)** 알고리즘은 모든 과거 데이터에 걸친 결정론적 비용 함수를 최소화합니다:

$$J_{RLS}(n) = \sum_{i=0}^{n} \lambda^{n-i} |e(i)|^2$$

여기서 $\lambda \in (0, 1]$은 **망각 인자(forgetting factor)**(보통 $0.95 \leq \lambda \leq 1.0$)입니다. 최근 샘플에 오래된 샘플보다 더 큰 가중치가 부여되어 비정상 환경에서 추적 능력을 제공합니다.

### 7.2 가중 LS에 대한 정규 방정식

비용 함수는 다음에 의해 최소화됩니다:

$$\mathbf{w}(n) = \boldsymbol{\Phi}^{-1}(n) \boldsymbol{\theta}(n)$$

여기서:
- $\boldsymbol{\Phi}(n) = \sum_{i=0}^{n} \lambda^{n-i} \mathbf{x}(i)\mathbf{x}^T(i)$는 가중 샘플 상관 행렬
- $\boldsymbol{\theta}(n) = \sum_{i=0}^{n} \lambda^{n-i} d(i)\mathbf{x}(i)$는 가중 상호상관 벡터

둘 다 순환 갱신을 가집니다:

$$\boldsymbol{\Phi}(n) = \lambda \boldsymbol{\Phi}(n-1) + \mathbf{x}(n)\mathbf{x}^T(n)$$

$$\boldsymbol{\theta}(n) = \lambda \boldsymbol{\theta}(n-1) + d(n)\mathbf{x}(n)$$

### 7.3 행렬 역 보조정리

각 단계에서 $\boldsymbol{\Phi}^{-1}(n)$을 다시 계산하는 것($O(M^3)$)을 피하기 위해 **행렬 역 보조정리(matrix inversion lemma, Woodbury identity)**를 사용합니다:

$$(\mathbf{A} + \mathbf{u}\mathbf{v}^T)^{-1} = \mathbf{A}^{-1} - \frac{\mathbf{A}^{-1}\mathbf{u}\mathbf{v}^T\mathbf{A}^{-1}}{1 + \mathbf{v}^T\mathbf{A}^{-1}\mathbf{u}}$$

$\mathbf{P}(n) = \boldsymbol{\Phi}^{-1}(n)$으로 정의하면:

$$\mathbf{P}(n) = \lambda^{-1}\mathbf{P}(n-1) - \lambda^{-1}\mathbf{k}(n)\mathbf{x}^T(n)\mathbf{P}(n-1)$$

여기서 **이득 벡터(gain vector)**는:

$$\mathbf{k}(n) = \frac{\lambda^{-1}\mathbf{P}(n-1)\mathbf{x}(n)}{1 + \lambda^{-1}\mathbf{x}^T(n)\mathbf{P}(n-1)\mathbf{x}(n)}$$

### 7.4 RLS 알고리즘 요약

```
RLS Algorithm
─────────────────────────────────────────────
Initialize: w(0) = 0, P(0) = δ^{-1} I (δ small, e.g., 0.01)
Parameters: forgetting factor λ (e.g., 0.99), regularization δ

For each new sample n = 1, 2, ...
  1. Compute gain vector:
     k(n) = P(n-1) x(n) / [λ + x^T(n) P(n-1) x(n)]

  2. Compute a priori error:
     e(n) = d(n) - w^T(n-1) x(n)

  3. Update weights:
     w(n) = w(n-1) + k(n) e(n)

  4. Update inverse correlation matrix:
     P(n) = λ^{-1} [P(n-1) - k(n) x^T(n) P(n-1)]
```

**계산 복잡도**: 샘플당 $O(M^2)$ ($\mathbf{P}$ 행렬 갱신 때문).

### 7.5 망각 인자

망각 인자 $\lambda$는 알고리즘의 **유효 메모리(effective memory)**를 결정합니다:

$$N_{eff} = \frac{1}{1 - \lambda}$$

| $\lambda$ | $N_{eff}$ | 동작 |
|-----------|-----------|------|
| 1.0 | $\infty$ | 성장하는 윈도우 (정상 환경) |
| 0.99 | 100 | 느리게 변하는 통계에 적합 |
| 0.95 | 20 | 빠르게 변하는 통계에 적합 |
| 0.9 | 10 | 매우 빠른 추적, 하지만 잡음이 많음 |

### 7.6 RLS의 특성

1. **빠른 수렴**: 약 $2M$ 반복에서 수렴 (고유값 분산에 독립적)
2. **고유값 분산 문제 없음**: $\mathbf{P}$ 행렬이 입력을 백색화(whitens)
3. **높은 복잡도**: LMS의 $O(M)$ 대비 $O(M^2)$
4. **수치적 민감성**: $\mathbf{P}$ 행렬이 양정치성을 잃을 수 있음; 안정화된 버전 존재 (QR-RLS, 격자 RLS)

---

## 8. 비교: LMS vs RLS

| 특성 | LMS | NLMS | RLS |
|------|-----|------|-----|
| 샘플당 복잡도 | $O(M)$ | $O(M)$ | $O(M^2)$ |
| 메모리 | $O(M)$ | $O(M)$ | $O(M^2)$ |
| 수렴 속도 | 느림 ($\chi$에 의존) | 보통 | 빠름 ($\sim 2M$ 반복) |
| 오조정 | 높음 | 보통 | 낮음 |
| 추적 능력 | 보통 | 보통 | 좋음 |
| 수치적 안정성 | 우수 | 우수 | 불안정할 수 있음 |
| 고유값 분산 민감도 | 높음 | 보통 | 없음 |
| 스텝 크기 파라미터 | $\mu$ (설정 까다로움) | $\tilde{\mu} \in (0,2)$ | $\lambda$ (설정 더 쉬움) |

**경험 규칙**: 계산 비용이 최우선이거나 필터가 길 때는 LMS/NLMS를 사용합니다. 빠른 수렴이 필수적이고 필터 차수가 적절할 때는 RLS를 사용합니다.

---

## 9. 응용: 시스템 식별

### 9.1 문제 설정

```
                    ┌────────────────────┐
     x(n) ────────▶│  Unknown System     │────────▶ d(n) = h*x(n) + v(n)
         │          │  h = [h0, h1, ...]  │
         │          └────────────────────┘
         │
         │          ┌────────────────────┐
         └────────▶│  Adaptive Filter   │────────▶ y(n) = w^T x(n)
                    │  w(n)              │
                    └────────────────────┘
                                                    e(n) = d(n) - y(n) → 0
```

적응 필터는 미지 시스템의 임펄스 응답을 학습합니다. 알고리즘이 수렴하면 $\mathbf{w}_{opt} \approx \mathbf{h}$가 됩니다.

### 9.2 사용 시기

- **플랜트 모델링**: 제어 시스템에는 시스템 모델이 필요합니다
- **음향 경로 식별**: 실내 임펄스 응답 파악
- **적응 역 제어**: 순방향 모델을 식별한 후 역산

---

## 10. 응용: 잡음 제거

### 10.1 적응 잡음 제거기 (ANC)

```
     Signal s(n) + Noise n0(n) = d(n)    (primary input)

     Noise reference n1(n)               (reference input, correlated with n0)
              │
              ▼
     ┌────────────────────┐
     │  Adaptive Filter   │ ──▶ ŷ(n) ≈ n0(n)
     │  w(n)              │
     └────────────────────┘
                                         e(n) = d(n) - ŷ(n) ≈ s(n)
```

**핵심 통찰**: 기준 입력 $n_1(n)$은 잡음 $n_0(n)$과 상관되어 있지만 신호 $s(n)$과는 상관되지 않습니다. 적응 필터는 $n_1(n)$을 $n_0(n)$의 추정값으로 변환합니다. 그러면 오차 신호가 깨끗한 신호 $s(n)$의 추정값이 됩니다.

### 10.2 수학적 정당성

MSE는:

$$E[e^2(n)] = E[(s(n) + n_0(n) - \hat{y}(n))^2]$$

$s(n)$이 $n_0(n)$과 $n_1(n)$ 모두와 상관되지 않으므로:

$$E[e^2(n)] = E[s^2(n)] + E[(n_0(n) - \hat{y}(n))^2]$$

$\mathbf{w}$에 대해 $E[e^2(n)]$을 최소화하면 $E[(n_0(n) - \hat{y}(n))^2]$가 최소화되어 $\hat{y}(n) \to n_0(n)$이 되고 $e(n) \to s(n)$이 됩니다.

**신호는 잡음 추정의 부산물로 추출됩니다.**

---

## 11. 응용: 에코 제거

### 11.1 음향 에코 제거 (AEC)

스피커폰 시스템에서 원단(far-end) 음성이 스피커를 통해 재생되고, 방 안에서 반향되어 마이크로폰에 포착됩니다. 적응 필터는 스피커에서 마이크로폰까지의 음향 경로를 모델링합니다.

```
Far-end ──▶ Loudspeaker ──▶ Room ──▶ Microphone ──▶ Near-end + Echo
  x(n)                   h(n)                        d(n) = s(n) + h*x(n)
    │
    │         ┌──────────────────┐
    └────────▶│  Adaptive Filter │──▶ ŷ(n) ≈ h*x(n)
              │  w(n) ≈ h        │
              └──────────────────┘
                                       e(n) = d(n) - ŷ(n) ≈ s(n)
```

**도전 과제**:
- 음향 임펄스 응답은 매우 길 수 있습니다 (8 kHz에서 100-500 ms = 800-4000 탭)
- 이중 통화(double-talk): 두 화자가 동시에 활성화
- 비정상성: 사람이 이동하고 문이 열림

### 11.2 네트워크 에코 제거

전화 네트워크에서 하이브리드(2선-4선 변환)의 임피던스 불일치가 전기적 에코를 생성합니다. 에코 경로는 짧지만 요구 사항이 엄격합니다 (>40 dB 에코 반환 손실 향상).

---

## 12. 응용: 채널 등화

### 12.1 문제

전송된 신호 $a(n)$이 분산 채널 $c(n)$을 통과하여 심볼 간 간섭(inter-symbol interference, ISI)을 생성합니다:

$$x(n) = \sum_k c(k) a(n-k) + v(n)$$

등화기(equalizer)는 채널 왜곡을 되돌리는 적응 필터입니다:

$$\hat{a}(n - \Delta) = \mathbf{w}^T(n) \mathbf{x}(n)$$

여기서 $\Delta$는 $w(n) * c(n) \approx \delta(n - \Delta)$가 되도록 선택된 결정 지연(decision delay)입니다.

### 12.2 훈련 및 결정 주도 모드

- **훈련 모드**: 알려진 시퀀스가 전송됩니다; $d(n) = a(n-\Delta)$
- **결정 주도 모드**: 초기 수렴 후, 슬라이서(slicer) 출력 $\hat{a}(n-\Delta)$를 $d(n)$으로 사용

---

## 13. 응용: 적응 빔포밍

### 13.1 문제

$M$개의 센서로 구성된 배열이 여러 방향에서 신호를 수신합니다. 목표는 원하는 신호 방향으로 빔을 조향하면서 간섭원(interferer)을 널링(nulling)하는 것입니다.

배열에서 수신된 신호는:

$$\mathbf{x}(n) = s(n)\mathbf{a}(\theta_s) + \sum_{k=1}^{K} i_k(n)\mathbf{a}(\theta_k) + \mathbf{v}(n)$$

여기서 $\mathbf{a}(\theta)$는 방향 $\theta$에 대한 **조향 벡터(steering vector)**입니다.

### 13.2 최소 분산 왜곡 없는 응답 (MVDR)

Capon 빔포머는 다음을 풉니다:

$$\min_{\mathbf{w}} \mathbf{w}^H \mathbf{R} \mathbf{w} \quad \text{제약 조건:} \quad \mathbf{w}^H \mathbf{a}(\theta_s) = 1$$

해:

$$\mathbf{w}_{MVDR} = \frac{\mathbf{R}^{-1}\mathbf{a}(\theta_s)}{\mathbf{a}^H(\theta_s)\mathbf{R}^{-1}\mathbf{a}(\theta_s)}$$

적응형 변형은 RLS와 유사한 갱신을 사용하여 $\mathbf{R}$을 순환적으로 추정합니다.

---

## 14. Python 구현: 완전한 적응 필터링 툴킷

### 14.1 LMS, NLMS, RLS 구현

```python
import numpy as np
import matplotlib.pyplot as plt


def lms_filter(x, d, M, mu):
    """
    LMS adaptive filter.

    Parameters
    ----------
    x : ndarray
        Input signal
    d : ndarray
        Desired (reference) signal
    M : int
        Filter order (number of taps)
    mu : float
        Step size

    Returns
    -------
    y : ndarray
        Filter output
    e : ndarray
        Error signal
    w_history : ndarray
        Weight history (N x M)
    """
    N = len(x)
    w = np.zeros(M)
    y = np.zeros(N)
    e = np.zeros(N)
    w_history = np.zeros((N, M))

    for n in range(M, N):
        x_vec = x[n:n-M:-1] if M > 1 else np.array([x[n]])
        # Proper construction of input vector
        x_vec = x[n-M+1:n+1][::-1]

        y[n] = np.dot(w, x_vec)
        e[n] = d[n] - y[n]
        w = w + mu * e[n] * x_vec
        w_history[n] = w

    return y, e, w_history


def nlms_filter(x, d, M, mu_tilde, delta=1e-6):
    """
    Normalized LMS adaptive filter.

    Parameters
    ----------
    x : ndarray
        Input signal
    d : ndarray
        Desired (reference) signal
    M : int
        Filter order
    mu_tilde : float
        Normalized step size (0 < mu_tilde < 2)
    delta : float
        Regularization constant

    Returns
    -------
    y, e, w_history : ndarrays
    """
    N = len(x)
    w = np.zeros(M)
    y = np.zeros(N)
    e = np.zeros(N)
    w_history = np.zeros((N, M))

    for n in range(M, N):
        x_vec = x[n-M+1:n+1][::-1]

        y[n] = np.dot(w, x_vec)
        e[n] = d[n] - y[n]

        norm_sq = np.dot(x_vec, x_vec) + delta
        w = w + (mu_tilde / norm_sq) * e[n] * x_vec
        w_history[n] = w

    return y, e, w_history


def rls_filter(x, d, M, lam=0.99, delta=0.01):
    """
    Recursive Least Squares adaptive filter.

    Parameters
    ----------
    x : ndarray
        Input signal
    d : ndarray
        Desired (reference) signal
    M : int
        Filter order
    lam : float
        Forgetting factor (0 < lambda <= 1)
    delta : float
        Regularization for P initialization

    Returns
    -------
    y, e, w_history : ndarrays
    """
    N = len(x)
    w = np.zeros(M)
    P = (1.0 / delta) * np.eye(M)
    y = np.zeros(N)
    e = np.zeros(N)
    w_history = np.zeros((N, M))

    for n in range(M, N):
        x_vec = x[n-M+1:n+1][::-1]

        # Gain vector
        Px = P @ x_vec
        denom = lam + x_vec @ Px
        k = Px / denom

        # A priori error
        y[n] = np.dot(w, x_vec)
        e[n] = d[n] - y[n]

        # Weight update
        w = w + k * e[n]

        # Inverse correlation matrix update
        P = (1.0 / lam) * (P - np.outer(k, x_vec @ P))

        w_history[n] = w

    return y, e, w_history
```

### 14.2 시스템 식별 예제

```python
# System Identification Demo
np.random.seed(42)

# Unknown system (FIR)
h_true = np.array([0.5, 1.2, -0.8, 0.3, -0.1])
M = len(h_true)

# Generate input signal (white noise)
N = 2000
x = np.random.randn(N)

# System output + measurement noise
d = np.convolve(x, h_true, mode='full')[:N] + 0.01 * np.random.randn(N)

# Run adaptive filters
mu_lms = 0.01
_, e_lms, w_lms = lms_filter(x, d, M, mu_lms)
_, e_nlms, w_nlms = nlms_filter(x, d, M, mu_tilde=0.5)
_, e_rls, w_rls = rls_filter(x, d, M, lam=0.99)

# Plot learning curves
fig, axes = plt.subplots(2, 1, figsize=(12, 8))

# MSE learning curves (smoothed)
window = 50
mse_lms = np.convolve(e_lms**2, np.ones(window)/window, mode='valid')
mse_nlms = np.convolve(e_nlms**2, np.ones(window)/window, mode='valid')
mse_rls = np.convolve(e_rls**2, np.ones(window)/window, mode='valid')

axes[0].semilogy(mse_lms, label='LMS', alpha=0.8)
axes[0].semilogy(mse_nlms, label='NLMS', alpha=0.8)
axes[0].semilogy(mse_rls, label='RLS', alpha=0.8)
axes[0].set_xlabel('Iteration')
axes[0].set_ylabel('MSE')
axes[0].set_title('Learning Curves: System Identification')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Final weight comparison
x_pos = np.arange(M)
width = 0.2
axes[1].bar(x_pos - 1.5*width, h_true, width, label='True', color='black')
axes[1].bar(x_pos - 0.5*width, w_lms[-1], width, label='LMS')
axes[1].bar(x_pos + 0.5*width, w_nlms[-1], width, label='NLMS')
axes[1].bar(x_pos + 1.5*width, w_rls[-1], width, label='RLS')
axes[1].set_xlabel('Tap index')
axes[1].set_ylabel('Weight value')
axes[1].set_title('Identified Impulse Response')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('system_identification.png', dpi=150, bbox_inches='tight')
plt.show()

# Print final weights
print("True system:  ", h_true)
print("LMS weights:  ", np.round(w_lms[-1], 4))
print("NLMS weights: ", np.round(w_nlms[-1], 4))
print("RLS weights:  ", np.round(w_rls[-1], 4))
```

### 14.3 잡음 제거 데모

```python
# Adaptive Noise Cancellation Demo
np.random.seed(42)

N = 5000
t = np.arange(N) / 1000.0  # 1 kHz sampling rate

# Clean signal: sum of sinusoids
s = np.sin(2 * np.pi * 50 * t) + 0.5 * np.sin(2 * np.pi * 120 * t)

# Noise source
noise_source = np.random.randn(N)

# Noise that corrupts the signal (filtered version of noise source)
noise_path = np.array([1.0, -0.5, 0.3, -0.1])
n0 = np.convolve(noise_source, noise_path, mode='full')[:N]

# Primary input: signal + noise
d = s + n0

# Reference input: correlated with noise but not with signal
# (different path from the noise source)
ref_path = np.array([0.8, -0.4, 0.2])
n1 = np.convolve(noise_source, ref_path, mode='full')[:N]

# Apply adaptive noise canceller
M = 8  # Filter order (longer than the noise path to be safe)
mu = 0.01

y_lms, e_lms, _ = lms_filter(n1, d, M, mu)
y_nlms, e_nlms, _ = nlms_filter(n1, d, M, mu_tilde=0.5)
y_rls, e_rls, _ = rls_filter(n1, d, M, lam=0.995)

# Plot results
fig, axes = plt.subplots(4, 1, figsize=(14, 12))

axes[0].plot(t[:500], s[:500], 'g', linewidth=1.5, label='Clean signal')
axes[0].set_title('Original Clean Signal')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(t[:500], d[:500], 'r', alpha=0.7, label='Signal + Noise')
axes[1].set_title('Noisy Signal (Primary Input)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

axes[2].plot(t[:500], e_nlms[:500], 'b', alpha=0.7, label='NLMS output')
axes[2].plot(t[:500], s[:500], 'g--', alpha=0.5, label='Clean (reference)')
axes[2].set_title('Recovered Signal (NLMS Noise Canceller)')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

# SNR improvement over time
window = 200
snr_input = 10 * np.log10(
    np.convolve(s**2, np.ones(window)/window, mode='same') /
    np.convolve(n0**2, np.ones(window)/window, mode='same') + 1e-10
)
residual_nlms = e_nlms - s
snr_output = 10 * np.log10(
    np.convolve(s**2, np.ones(window)/window, mode='same') /
    np.convolve(residual_nlms**2, np.ones(window)/window, mode='same') + 1e-10
)

axes[3].plot(t, snr_input, 'r', alpha=0.7, label='Input SNR')
axes[3].plot(t, snr_output, 'b', alpha=0.7, label='Output SNR (NLMS)')
axes[3].set_xlabel('Time (s)')
axes[3].set_ylabel('SNR (dB)')
axes[3].set_title('SNR Improvement')
axes[3].legend()
axes[3].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('noise_cancellation.png', dpi=150, bbox_inches='tight')
plt.show()

# Compute overall SNR improvement
snr_in = 10 * np.log10(np.mean(s[M:]**2) / np.mean(n0[M:]**2))
snr_out_nlms = 10 * np.log10(
    np.mean(s[1000:]**2) / np.mean((e_nlms[1000:] - s[1000:])**2)
)
print(f"Input SNR:       {snr_in:.1f} dB")
print(f"Output SNR (NLMS): {snr_out_nlms:.1f} dB")
print(f"SNR improvement:   {snr_out_nlms - snr_in:.1f} dB")
```

### 14.4 시변 시스템 추적

```python
# Tracking a time-varying system
np.random.seed(42)

N = 4000
x = np.random.randn(N)

# Time-varying system: coefficients change at n=2000
h1 = np.array([1.0, 0.5, -0.3])
h2 = np.array([0.2, -0.8, 1.0])
M = 3

d = np.zeros(N)
for n in range(M, N):
    x_vec = x[n-M+1:n+1][::-1]
    if n < 2000:
        d[n] = np.dot(h1, x_vec) + 0.01 * np.random.randn()
    else:
        d[n] = np.dot(h2, x_vec) + 0.01 * np.random.randn()

# Compare algorithms
_, e_lms, w_lms = lms_filter(x, d, M, mu=0.05)
_, e_nlms, w_nlms = nlms_filter(x, d, M, mu_tilde=0.8)
_, e_rls, w_rls = rls_filter(x, d, M, lam=0.98)

# Plot weight trajectories
fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

titles = ['LMS', 'NLMS', 'RLS']
w_histories = [w_lms, w_nlms, w_rls]
colors = ['tab:blue', 'tab:orange', 'tab:green']

for ax, title, w_hist in zip(axes, titles, w_histories):
    for i in range(M):
        ax.plot(w_hist[:, i], label=f'w[{i}]', alpha=0.8)
    # Plot true values
    ax.axhline(y=h1[0], color='gray', linestyle=':', alpha=0.3)
    ax.axhline(y=h1[1], color='gray', linestyle=':', alpha=0.3)
    ax.axhline(y=h1[2], color='gray', linestyle=':', alpha=0.3)
    ax.axvline(x=2000, color='red', linestyle='--', alpha=0.5, label='System change')
    ax.set_title(f'{title} Weight Tracking')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

axes[-1].set_xlabel('Iteration')
plt.tight_layout()
plt.savefig('tracking_demo.png', dpi=150, bbox_inches='tight')
plt.show()
```

---

## 15. 연습 문제

### 연습 문제 1: 위너 필터

$x(n)$이 분산 $\sigma_x^2 = 1$인 백색 잡음이고, 원하는 신호가 $d(n) = 0.8x(n) + 0.5x(n-1) - 0.3x(n-2) + v(n)$인 시스템을 고려하세요. 여기서 $v(n)$은 분산 $\sigma_v^2 = 0.1$인 백색 잡음으로 $x(n)$과 독립입니다.

(a) 3탭 위너 필터에 대한 자기상관 행렬 $\mathbf{R}$을 계산하세요.

(b) 상호상관 벡터 $\mathbf{p}$를 계산하세요.

(c) $\mathbf{R}\mathbf{w}_{opt} = \mathbf{p}$를 풀어 최적 위너 필터 $\mathbf{w}_{opt}$를 구하세요.

(d) 최소 MSE $J_{min}$을 계산하세요.

### 연습 문제 2: LMS 수렴

$M = 10$ 탭을 가진 LMS 필터가 자기상관 행렬의 고유값이 $\lambda_{max} = 5.0$, $\lambda_{min} = 0.1$인 입력 신호에 적용됩니다.

(a) 평균 수렴을 위한 최대 스텝 크기는 얼마입니까?

(b) 조건수 $\chi(\mathbf{R})$은 얼마입니까?

(c) $\mu = 0.01$이면, $\text{tr}(\mathbf{R}) = 10$일 때 오조정 $\mathcal{M}$을 계산하세요.

(d) 가장 느린 모드의 수렴 시상수 $\tau_{mse}$를 추정하세요.

(e) LMS를 적용하기 전에 입력을 백색화(whitening)하면 수렴이 어떻게 변할지 질적으로 설명하세요.

### 연습 문제 3: NLMS vs LMS

전력이 500 샘플마다 0.1과 10.0 사이를 교대하는 비정상 입력 신호에 대해 잡음 제거를 위한 LMS와 NLMS를 모두 구현하세요. 필터 차수 $M = 16$을 사용하세요.

(a) 고정 스텝 크기를 가진 LMS가 고전력 구간에서 발산하거나 저전력 구간에서 너무 느리게 수렴함을 보이세요.

(b) NLMS가 전력 변동을 우아하게 처리함을 시연하세요.

(c) 두 알고리즘의 MSE 학습 곡선을 그리세요.

### 연습 문제 4: RLS 구현

임펄스 응답 $h = [1, -0.5, 0.25, -0.125]$를 가진 시스템을 식별하기 위해 망각 인자 $\lambda = 0.99$로 RLS를 구현하세요.

(a) 각 가중치가 실제 값으로 수렴하는 것을 그래프로 나타내세요. LMS 및 NLMS와 비교하세요.

(b) $\lambda$를 0.9에서 1.0까지 변화시키고 정상 상태 MSE 대 수렴 시간 트레이드오프를 그리세요.

(c) $n = 1000$에서 시스템 변화를 도입하세요 ($h$를 $[0.5, 0.3, -0.2, 0.1]$로 변경). LMS, NLMS, RLS의 추적 성능을 비교하세요.

### 연습 문제 5: 에코 제거 시뮬레이션

음향 에코 제거 시나리오를 시뮬레이션하세요:

(a) 다양한 주파수의 정현파 합으로 "원단 음성" 신호를 생성하세요.

(b) 실내 임펄스 응답을 생성하세요 (길이 100의 지수적으로 감소하는 랜덤 시퀀스 사용).

(c) 근단(near-end) 잡음을 추가하세요.

(d) 필터 차수 128의 NLMS를 적용하세요. 시간에 따른 에코 반환 손실 향상(ERLE)을 그리세요:

$$\text{ERLE}(n) = 10 \log_{10} \frac{E[d^2(n)]}{E[e^2(n)]}$$

(e) 이중 통화(근단 음성 추가)가 적응 필터에 미치는 영향을 조사하세요.

### 연습 문제 6: 적응 등화

디지털 통신 채널이 임펄스 응답 $c = [0.5, 1.0, 0.5]$를 가집니다 (ISI 발생).

(a) 랜덤 BPSK 신호($a(n) \in \{-1, +1\}$)를 생성하고 채널을 통과시키세요. SNR = 20 dB에서 잡음을 추가하세요.

(b) $M = 11$ 탭과 결정 지연 $\Delta = 5$로 LMS를 사용하여 적응 등화기를 설계하세요.

(c) 훈련 길이의 함수로 비트 오류율(BER)을 그리세요.

(d) 500개의 훈련 심볼 후 결정 주도 모드로 전환하고 BER이 안정적으로 유지됨을 검증하세요.

(e) 등화 전후의 아이 다이어그램(eye diagram)을 비교하세요.

### 연습 문제 7: 필터 차수의 영향

실제 시스템 $h = [0.5, 1.2, -0.8, 0.3, -0.1]$에 대한 시스템 식별 문제에서:

(a) 필터 차수 $M = 3, 5, 7, 10, 20$으로 LMS를 실행하고 정상 상태 MSE를 비교하세요.

(b) $M < 5$ (과소 모델링)와 $M > 5$ (과대 모델링)일 때 어떤 일이 발생하는지 설명하세요.

(c) 각 $M$에 대해 식별된 임펄스 응답을 그리세요.

---

## 16. 요약

| 개념 | 핵심 공식 / 아이디어 |
|------|-------------------|
| 위너 필터 | $\mathbf{w}_{opt} = \mathbf{R}^{-1}\mathbf{p}$ (최적 MMSE) |
| 최급강하법 | $\mathbf{w}(n+1) = \mathbf{w}(n) + 2\mu(\mathbf{p} - \mathbf{R}\mathbf{w}(n))$ |
| 수렴 조건 | $0 < \mu < 1/\lambda_{max}$ |
| LMS 갱신 | $\mathbf{w}(n+1) = \mathbf{w}(n) + \mu \, e(n) \, \mathbf{x}(n)$ |
| LMS 오조정 | $\mathcal{M} = \mu \, \text{tr}(\mathbf{R})$ |
| NLMS 갱신 | $\mathbf{w}(n+1) = \mathbf{w}(n) + \frac{\tilde{\mu}}{\|\mathbf{x}\|^2+\delta} e(n)\mathbf{x}(n)$ |
| RLS 이득 | $\mathbf{k}(n) = \frac{\mathbf{P}(n-1)\mathbf{x}(n)}{\lambda + \mathbf{x}^T(n)\mathbf{P}(n-1)\mathbf{x}(n)}$ |
| 망각 인자 메모리 | $N_{eff} = 1/(1-\lambda)$ |
| 잡음 제거 | 오차 신호 $e(n) = d(n) - \hat{y}(n) \approx s(n)$ |
| 트레이드오프 | 빠른 수렴 vs 낮은 오조정 |

**핵심 정리**:
1. 위너 필터는 이론적 최적을 제공하지만 알려진 통계가 필요합니다.
2. LMS는 기울기를 순간 추정값으로 근사합니다 - 단순하고 강건하며 $O(M)$입니다.
3. NLMS는 입력 전력으로 정규화합니다 - 변동하는 신호 레벨에서 더 나은 안정성을 제공합니다.
4. RLS는 지수 가중치로 모든 과거 데이터를 사용합니다 - $O(M^2)$ 비용으로 빠른 수렴을 달성합니다.
5. 오조정-수렴 트레이드오프는 모든 적응 알고리즘에 근본적입니다.
6. 적응 필터는 잡음 제거에서 등화까지 수많은 응용을 지원합니다.

---

## 17. 참고 문헌

1. S. Haykin, *Adaptive Filter Theory*, 5th ed., Pearson, 2014.
2. A.H. Sayed, *Adaptive Filters*, Wiley-IEEE Press, 2008.
3. P.S.R. Diniz, *Adaptive Filtering: Algorithms and Practical Implementation*, 4th ed., Springer, 2013.
4. B. Widrow and S.D. Stearns, *Adaptive Signal Processing*, Pearson, 1985.
5. S. Haykin, "Adaptive filter theory," in *Proc. IEEE*, vol. 90, no. 2, pp. 211-259, 2002.
6. B. Farhang-Boroujeny, *Adaptive Filters: Theory and Applications*, 2nd ed., Wiley, 2013.

---

**이전**: [12. 다중 레이트 신호 처리](./12_Multirate_Signal_Processing.md) | **다음**: [14. 시간-주파수 분석](./14_Time_Frequency_Analysis.md)
