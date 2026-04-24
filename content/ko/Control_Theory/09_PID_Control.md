# 레슨 9: PID 제어(PID Control)

## 학습 목표

- PID 제어기의 구조와 동작 원리를 이해한다
- 각 항(P, I, D)이 시스템 성능에 미치는 영향을 기술한다
- 고전 기법(Ziegler-Nichols, Cohen-Coon, IMC)으로 PID 제어기를 튜닝한다
- 실제 문제 — 적분기 윈드업(integrator windup), 미분 킥(derivative kick), 잡음 증폭 — 을 인식한다
- 일반적인 플랜트 유형에 PID 제어를 적용한다
- 합리적인 기본값을 갖는 PID 제어기를 코드로 구현하고 동작을 시뮬레이션한다

## 0. PID가 지배하는 이유 — 짧은 역사

오늘날 플랜트 설계자가 제어기를 고르면 약 90%의 확률로 PID를 선택한다. 이 사실은 70년 넘게 석유화학 플랜트, 공정 라인, 항공 자동조종, 가정용 온도조절기, 쿼드콥터 비행 스택에 걸쳐 유지되어 왔다. 그 이유는 세 가지다:

- **모델 없이도 튜닝할 수 있을 만큼 단순하다.** 기술자 한 명이 스톱워치, 계단 변화, 그리고 차트 레코더만으로 쓸 만한 루프를 얻을 수 있다. 현대 기법(상태공간, $H_\infty$)은 거의 언제나 플랜트 모델을 요구한다. PID는 놀랄 만큼의 무지도 용납한다.
- **기본 원리 기반 플랜트 대부분에는 세 개의 손잡이면 충분하다.** 우세한 1차 또는 2차 동역학을 가진 거의 모든 플랜트는 세 개의 이득으로 안정화되고 기준을 추종할 수 있다. 더 정교한 동역학(데드타임(dead time), 공진)은 이 공식을 늘이지만, "세 손잡이" 직관은 상당히 멀리 간다.
- **현장 엔지니어 모두가 이미 안다.** 운영자가 새벽 3시에 실시간 장애를 진단해야 하는 분야에서, 당직 팀원 누구나 동작을 이해하는 제어기는 그 자체로 안전 기능이다.

한계도 실재한다 — PID는 강한 비선형 플랜트, 데드타임이 큰 플랜트, 결합이 강한 MIMO 시스템에서는 평범하다. 그러나 대부분의 산업에서는 이런 한계가 예외적 사례다.

읽는 동안 구체적 비유를 머릿속에 두자:

- **비례(Proportional)**는 설정값에서 떨어진 만큼 비례해 시스템을 되돌리는 스프링이다.
- **적분(Integral)**은 과거 오차를 쌓아 두고 오차가 작아져도 계속 밀어주는 톱니장치다.
- **미분(Derivative)**은 빠른 변화에 저항하는 쇼크 업소버다.

자동차 서스펜션에는 이 셋이 모두 있다. PID 루프도 마찬가지다. 이 대응은 우연이 아니다 — 둘 다 기본 원리 물리를 따르는 선형 귀환 시스템이다.

## 1. PID 제어기(The PID Controller)

**PID 제어기(Proportional-Integral-Derivative)**는 산업에서 가장 널리 사용되는 제어기다. 오차 $e(t) = r(t) - y(t)$로부터 제어 신호를 계산한다:

$$u(t) = K_p e(t) + K_i \int_0^t e(\tau) \, d\tau + K_d \frac{de(t)}{dt}$$

**전달함수(이상형, ideal form):**

$$G_c(s) = K_p + \frac{K_i}{s} + K_d s = K_p\left(1 + \frac{1}{T_i s} + T_d s\right)$$

여기서 $T_i = K_p/K_i$는 **적분 시간(integral time)**, $T_d = K_d/K_p$는 **미분 시간(derivative time)**이다.

### 1.1 왜 PID인가?

- 세 가지 가장 중요한 제어 작업 처리: 추종(tracking), 외란 제거(disturbance rejection), 잡음 여과(noise filtering)
- 튜닝할 파라미터가 3개뿐
- 다양한 플랜트에 잘 작동
- 산업 제어기의 90% 이상이 PID(또는 PI)

## 2. 각 항의 효과(Effect of Each Term)

### 2.1 비례(Proportional, P) 작용

$$u_P(t) = K_p e(t)$$

- 출력이 현재 오차에 비례
- 이득을 **증가**시키면 → 정상상태 오차 감소, 응답 속도 증가
- **그러나:** 계단 외란에 대해 정상상태 오차를 제거할 수 없음 (형 0 시스템에서)
- $K_p$가 너무 크면 → 불안정(위상 여유 감소)

### 2.2 적분(Integral, I) 작용

$$u_I(t) = K_i \int_0^t e(\tau) \, d\tau$$

- 과거 오차를 누적
- 계단 입력에 대한 **정상상태 오차를 제거**($s = 0$에 극점 추가, 시스템 형 증가)
- **그러나:** 위상 지연 추가 → 시스템을 불안정화할 수 있음
- 급격한 변화에 대한 응답이 느림

### 2.3 미분(Derivative, D) 작용

$$u_D(t) = K_d \frac{de(t)}{dt}$$

- 오차의 변화율에 반응
- 미래 오차를 **예측** → 과도 응답 개선, 제동(damping) 추가
- **그러나:** 고주파 잡음을 증폭
- 정상상태 오차에는 영향 없음

### 2.4 요약 표

| 작용 | 상승 시간에 미치는 영향 | 오버슈트에 미치는 영향 | 정착 시간에 미치는 영향 | 정상상태 오차 |
|--------|-------------------|--------------------|------------------------|--------------------|
| $K_p$ 증가 | 감소 | 증가 | 약간 변화 | 감소 |
| $K_i$ 증가 | 감소 | 증가 | 증가 | 제거 |
| $K_d$ 증가 | 약간 변화 | 감소 | 감소 | 영향 없음 |

**주의사항:** 이 표는 일반적 경향을 보여준다. 실제 효과는 특정 플랜트와 동작점에 따라 다르다.

### 2.5 계산 예제: 1차 플랜트에 P 단독 적용

플랜트 $G_p(s) = 2/(s+1)$ — DC 이득 2, 시상수 1 s를 고려하자. P 제어기로 루프를 닫으면:

$$\frac{Y(s)}{R(s)} = \frac{K_p \cdot 2}{(s+1) + K_p \cdot 2} = \frac{2K_p}{s + (1 + 2K_p)}$$

폐루프는 여전히 1차이며, 시상수는 $1/(1+2K_p)$, DC 이득은 $2K_p/(1+2K_p)$이다.

| $K_p$ | 폐루프 DC 이득 | 폐루프 $\tau$ | 계단 정상상태 오차 |
|-------|--------------------|--------------------|-------------------------|
| $0.5$ | $0.50$ | $0.50\,\text{s}$ | $50\%$ |
| $2$ | $0.80$ | $0.20\,\text{s}$ | $20\%$ |
| $10$ | $0.95$ | $0.048\,\text{s}$ | $5\%$ |
| $100$ | $0.995$ | $0.005\,\text{s}$ | $0.5\%$ |

이 표에서 얻는 두 가지 교훈:

1. $K_p$를 올리면 정상상태 오차가 0에 수렴하지만 결코 도달하지 못한다 — 이것이 형 0 플랜트에서 순수 P가 오프셋(offset)을 남기는 이유다.
2. $K_p$를 올리면 응답도 빨라진다(더 작은 $\tau$). 1차 플랜트에서는 이것이 거의 공짜지만, 그 이상의 차수에서는 큰 $K_p$가 루프를 불안정화한다. 이 절충은 차수 $\geq 2$에서야 드러난다.

## 3. 일반적인 PID 구성(Common PID Configurations)

### 3.1 P 제어기

$$G_c(s) = K_p$$

가장 간단한 제어기. 플랜트가 이미 적분기를 가지고 있거나 약간의 정상상태 오차가 허용될 때 유용하다.

### 3.2 PI 제어기

$$G_c(s) = K_p\left(1 + \frac{1}{T_i s}\right) = K_p \frac{T_i s + 1}{T_i s}$$

산업의 주력 제어기. 관리 가능한 복잡도로 계단 입력에 대한 정상상태 오차를 0으로 만든다.

PI 제어기는 $s = -1/T_i$에 **영점(zero)**을 추가하고 $s = 0$에 **극점(pole)**을 추가한다. 영점은 적분기가 도입한 위상 지연을 부분적으로 보상한다.

### 3.3 PD 제어기

$$G_c(s) = K_p(1 + T_d s)$$

적분기를 추가하지 않고 과도 응답을 개선한다. 정상상태 오차를 플랜트 자체가 처리할 때(형 1 이상) 유용하다.

### 3.4 PID 제어기

$$G_c(s) = K_p\left(1 + \frac{1}{T_i s} + T_d s\right) = K_p \frac{T_d T_i s^2 + T_i s + 1}{T_i s}$$

세 가지 작용을 결합. 분자의 두 영점을 배치하여 과도 성능과 정상상태 성능을 모두 개선할 수 있다.

## 4. PID 튜닝 방법(PID Tuning Methods)

### 4.1 Ziegler-Nichols: 개루프 방법 (공정 반응 곡선)

개루프 플랜트에 계단 입력을 인가하고 응답을 측정한다:

```
y(t)
 ^            _______________
 |           /
 |          / ← tangent at steepest point
 |     ____/
 |    /
 |───┘
 └──────┬──┬─────────────→ t
        L   T
```

- $L$: **지연 시간(delay time)** (계단에서 접선이 시간축과 만나는 지점까지)
- $T$: **시상수(time constant)** (접선이 최종값에 도달하기까지)
- $K_0$: 플랜트 DC 이득

**Ziegler-Nichols 개루프 튜닝 규칙:**

| 제어기 | $K_p$ | $T_i$ | $T_d$ |
|-----------|-------|-------|-------|
| P | $T/(K_0 L)$ | — | — |
| PI | $0.9T/(K_0 L)$ | $L/0.3$ | — |
| PID | $1.2T/(K_0 L)$ | $2L$ | $0.5L$ |

### 4.2 Ziegler-Nichols: 폐루프 방법 (극한 이득)

1. $K_i = 0$, $K_d = 0$ 설정(P 단독 제어)
2. 시스템이 지속 진동을 보일 때까지 $K_p$ 증가
3. **극한 이득(ultimate gain)** $K_u$와 **극한 주기(ultimate period)** $T_u$ 기록

**Ziegler-Nichols 폐루프 튜닝 규칙:**

| 제어기 | $K_p$ | $T_i$ | $T_d$ |
|-----------|-------|-------|-------|
| P | $0.5K_u$ | — | — |
| PI | $0.45K_u$ | $T_u/1.2$ | — |
| PID | $0.6K_u$ | $T_u/2$ | $T_u/8$ |

**참고:** Ziegler-Nichols 튜닝은 일반적으로 약 25% 오버슈트가 있는 공격적 설정을 제공한다. 추가적인 개선이 보통 필요하다.

### 4.3 Cohen-Coon 방법

동일한 개루프 측정값($K_0$, $L$, $T$)을 사용하지만 덜 공격적인 튜닝을 제공. 큰 데드타임을 가진 플랜트에 더 적합.

### 4.4 내부 모델 제어(IMC) 튜닝

플랜트 모델 $G_p(s)$에 기반하여 플랜트를 근사적으로 역변환하도록 제어기를 설계한다:

1차 플러스 데드타임(FOPDT) 모델 $G_p = \frac{K_0 e^{-Ls}}{\tau s + 1}$에 대해:

$$K_p = \frac{\tau}{K_0(\lambda + L)}, \quad T_i = \tau, \quad T_d = 0$$

여기서 $\lambda$는 원하는 폐루프 시상수(단일 튜닝 파라미터).

- 더 큰 $\lambda$ → 느리지만 더 견고함
- 더 작은 $\lambda$ → 빠르지만 덜 견고함

### 4.5 튜닝 실습 워크스루

계단 응답에서 식별된 FOPDT 플랜트: $K_0 = 1.5$, $T = 4\,\text{s}$, $L = 0.8\,\text{s}$.

**Ziegler-Nichols PID(개루프):**

$$K_p = \frac{1.2 \cdot 4}{1.5 \cdot 0.8} = 4.0, \quad T_i = 2 \cdot 0.8 = 1.6\,\text{s}, \quad T_d = 0.5 \cdot 0.8 = 0.4\,\text{s}$$

예상 거동: 약 25% 오버슈트, 공격적 — 출발점이지 완성된 설계는 아니다.

**$\lambda = 1\,\text{s}$로 IMC-PI 튜닝:**

$$K_p = \frac{4}{1.5 \cdot (1 + 0.8)} = 1.48, \quad T_i = 4\,\text{s}, \quad T_d = 0$$

훨씬 보수적: 낮은 이득, 미분 작용 없음. 최소 오버슈트로 부드럽지만 Z-N보다 눈에 띄게 느릴 것으로 예상.

**$\lambda = 0.4\,\text{s}$의 IMC-PI:** $K_p = \frac{4}{1.5 \cdot 1.2} = 2.22$. Z-N과 보수적 IMC의 대략 중간.

어느 것이 "옳은가"? 답은 허용 가능한 견고성 여유 하에서 사양을 만족하는 집합이다. 세 가지 모두를 시뮬레이션하고, 애플리케이션에 맞는 절충을 가진 것을 고르자.

## 5. 실제 PID 문제(Practical PID Issues)

### 5.1 미분 킥(Derivative Kick)

설정값이 급변할 때 $de/dt$가 큰 스파이크(미분 킥)를 생성한다. 해결책: 미분 작용을 출력에만 적용한다(**측정값에 대한 미분(derivative on measurement)**):

$$u_D = -K_d \frac{dy}{dt} \quad \text{(대신)} \quad u_D = K_d \frac{de}{dt}$$

### 5.2 적분기 윈드업(Integrator Windup)

액추에이터가 포화(물리적 한계에 도달)되면 적분 항이 계속 오차를 누적하여, 시스템이 선형 영역으로 돌아올 때 큰 오버슈트를 유발한다.

**안티 윈드업(anti-windup) 해결책:**
- **클램핑(clamping)**: 출력이 포화되면 적분을 중단
- **역계산(back-calculation)**: 원하는 액추에이터 출력과 실제 출력의 차이에 기반하여 적분 항 감소
- **조건부 적분(conditional integration)**: 오차가 작을 때만 적분

### 5.3 미분 잡음 증폭

순수 미분은 고주파 잡음을 증폭한다. **여과된 미분(filtered derivative)**을 사용한다:

$$D(s) = \frac{K_d s}{1 + s/(N\omega_c)}$$

여기서 $N$은 일반적으로 5-20. 이것은 고주파에서 미분 이득을 제한한다.

### 5.4 실제 PID 형태

모든 실용적 수정을 결합:

$$G_c(s) = K_p\left(1 + \frac{1}{T_i s}\right) - K_d \frac{s}{1 + s/N_f} \cdot Y(s)/E(s)$$

적분기에 안티 윈드업 포함.

### 5.5 참조 구현

실제 시스템에 삽입할 만큼 간결한 코드로 표현된 실용적 형태:

```python
class PID:
    """Discrete-time PID with filtered derivative, derivative-on-measurement,
    and back-calculation anti-windup. One call per sample period."""

    def __init__(self, kp, ki, kd, dt,
                 u_min=-float("inf"), u_max=float("inf"),
                 n_filter=10.0, kt=None):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.dt = dt
        self.u_min, self.u_max = u_min, u_max
        self.n = n_filter                       # derivative filter strength
        self.kt = kt if kt is not None else max(ki, 1e-6)   # back-calc gain
        self.integral = 0.0
        self.prev_y = None
        self.prev_d = 0.0

    def step(self, r, y):
        e = r - y

        # Proportional
        p = self.kp * e

        # Derivative on measurement (avoids derivative kick on setpoint changes),
        # with first-order filter so high-frequency noise cannot ring up the term.
        if self.prev_y is None:
            self.prev_y = y
        dy = (y - self.prev_y) / self.dt
        d = self.prev_d + self.n * self.dt * (-self.kd * dy - self.prev_d)
        self.prev_d = d
        self.prev_y = y

        # Integral + back-calculation anti-windup: clamp the output, then use the
        # amount we clamped to pull the integral back toward the linear region.
        u_unclamped = p + self.integral + d
        u = max(self.u_min, min(self.u_max, u_unclamped))
        self.integral += self.dt * (self.ki * e + self.kt * (u - u_unclamped))

        return u
```

각 조각에는 이유가 있다: 측정값에 대한 미분은 설정값 계단에서의 킥을 제거한다. 여과된 미분은 잡음이 무한히 증폭될 수 없게 한다. 역계산은 포화 간극을 이득 $K_t$로 적분기에 되먹여, 액추에이터가 포화에서 벗어날 때 루프가 사용할 수 없는 큰 $I$를 쌓아두지 않게 한다.

## 6. PID 설계 예제(PID Design Example)

**플랜트:** DC 모터 $G_p(s) = \frac{10}{s(s+5)}$ (형 1, 계단에 대한 정상상태 오차 없음)

**요구사항:** 오버슈트 0, 정착 시간 $< 2$ s, 경사 오차 0.

**설계:**
1. PI 제어기 필요 (경사 오차 0을 위해 형 2): $G_c(s) = K_p(1 + 1/(T_i s))$
2. 제동을 위한 미분 추가: PID $G_c(s) = K_p(1 + 1/(T_i s) + T_d s)$

**Ziegler-Nichols (폐루프):** P 단독으로 Routh에서 $K_u$ 찾기: $s^2 + 5s + 10K_p = 0$ → $K_u = \infty$ (형 1은 P 단독으로 항상 안정). 이 방법은 여기에 직접 적용되지 않는다.

**대안(극점 배치):** 원하는 폐루프 극점을 선택한 후 PID 파라미터를 구한다 — 상태공간 레슨에서 다루는 더 체계적인 접근법.

## 7. 파이썬(Python) 시뮬레이션

아래 스니펫을 5.5절의 `PID` 클래스와 함께 붙여넣으면 완전한 튜닝 가능한 시험대가 된다:

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

def simulate(plant_num, plant_den, pid, t_end=10.0, dt=0.01, setpoint=1.0):
    """Closed-loop simulation with a PID controller and a scalar plant."""
    sys = signal.TransferFunction(plant_num, plant_den)
    ss = sys.to_ss()            # continuous state-space for the plant
    x = np.zeros(ss.A.shape[0])

    t_hist, y_hist, u_hist = [], [], []
    n_steps = int(t_end / dt)
    for k in range(n_steps):
        y = float((ss.C @ x)[0])
        u = pid.step(setpoint, y)

        # Simple forward-Euler step on the plant ẋ = A x + B u
        x = x + dt * (ss.A @ x + ss.B.flatten() * u)

        t_hist.append(k * dt)
        y_hist.append(y)
        u_hist.append(u)

    return np.array(t_hist), np.array(y_hist), np.array(u_hist)


# Plant: first-order with DC gain 2 and time constant 1 s
plant_num, plant_den = [2], [1, 1]

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 6))

for label, gains in [
    ("P",  (2.0, 0.0, 0.0)),
    ("PI", (2.0, 3.0, 0.0)),
    ("PID", (2.0, 3.0, 0.3)),
]:
    pid = PID(*gains, dt=0.01, u_min=-5, u_max=5)
    t, y, u = simulate(plant_num, plant_den, pid)
    ax1.plot(t, y, label=label)
    ax2.plot(t, u, label=label)

ax1.axhline(1.0, color="gray", linestyle="--", linewidth=0.8)
ax1.set_ylabel("output y")
ax2.set_ylabel("control u")
ax2.set_xlabel("time [s]")
for ax in (ax1, ax2):
    ax.legend()
    ax.grid(True)
plt.tight_layout()
plt.show()
```

실행 후 실험해 보자: $K_p$를 2에서 0.5로 낮추면 오프셋(P 단독)이 다시 나타난다. $K_i$를 3에서 10으로 올리면 오버슈트가 커진다. $K_d = 0.3$을 더하면 오버슈트가 다시 줄어든다. 이 세 슬라이더가 PID다 — 이 분야의 나머지는 그저 각 슬라이더가 언제 당신을 속이는지 아는 것이다.

## 8. 흔한 함정(Common Pitfalls)

1. **잘못된 지표로 튜닝.** 계단에 대한 5% 오버슈트를 최적화하는 엔지니어가 외란 제거 성능을 뜻하지 않게 악화시킬 수 있다. 항상 설정값-출력 및 외란-출력 응답을 모두 측정하라.
2. **측정 동역학을 포함하지 않음.** PID 설계 목적상, 2초 지연이 있는 열전대는 (센서일지언정) 플랜트의 일부다. 이를 모델에서 빼면 과도하게 공격적인 튜닝이 나온다.
3. **적분 동역학이 있는 플랜트에 적분이 "도움이 될" 것이라 기대.** $1/s$ 형태의 플랜트는 이미 P 단독으로 계단 오차를 제거한다. I를 추가하면 이중 적분기가 되어 경계 안정이 된다. 플랜트가 이미 형 1 이상인지 인식하라.
4. **여과 없는 오차 미분.** 실제 하드웨어에서 이것은 거의 항상 버그다 — ADC와 센서 잡음이 $K_d$를 통해 액추에이터로 쏟아진다. 항상 여과하고, 측정값에 대한 미분을 고려하라.
5. **Ziegler-Nichols를 맹목적으로 적용.** Z-N은 출발점이지 최종 답이 아니다. 25% 오버슈트 설계 선택은 1940년대 공정 제어에는 적절했지만, 현대 애플리케이션은 보통 더 작은 오버슈트를 원하므로 Z-N 이득은 첫 시뮬레이션 후 절반으로 낮추거나 탈조율된다.
6. **샘플 시간 불일치.** 5.5절의 이산 시간 PID는 `step()` 호출이 매 `dt` 초마다 일어난다고 가정한다. 비 RTOS 시스템에서 흔한 20%+ 지터(jitter)는 루프를 소리 없이 탈조율한다. 실시간 스케줄러에서 실행하거나 실제 측정된 샘플 주기를 사용해 이득을 다시 유도하라.
7. **안티 윈드업을 잘못된 신호에 적용.** 역계산은 포화된 제어기 출력과 포화되지 않은 출력의 차이에 기반하여 적분기를 감소시킨다. 피드백 경로 없는 단순 클램핑은 적분기를 포화값에 얼어붙게 둔다 — 설정값이 마침내 도달 가능해졌을 때, 적분기는 여전히 거대한 저장값을 가지고 있다.

## 연습 문제

### 연습 문제 1: PID 효과 분석

플랜트 $G_p(s) = \frac{1}{s+1}$인 단위 귀환 시스템에 대해:

1. P 제어($K_p = 10$)로 폐루프 전달함수, 정상상태 계단 오차, $M_p$를 구하라
2. 적분 작용($T_i = 2$)을 추가하고 반복하라
3. 미분 작용($T_d = 0.1$)을 추가하고 오버슈트에 대한 영향을 분석하라

### 연습 문제 2: Ziegler-Nichols 튜닝

플랜트의 계단 응답: 지연 $L = 0.5$ s, 시상수 $T = 3$ s, DC 이득 $K_0 = 2$.

1. Ziegler-Nichols 개루프 방법으로 PID 파라미터를 계산하라
2. 같은 방법으로 PI 파라미터를 계산하라
3. $\lambda = 1$ s의 IMC 방법으로 PI 파라미터를 계산하고 비교하라

### 연습 문제 3: 안티 윈드업

$\pm 1$에서 포화하는 액추에이터를 가진 플랜트를 $K_p = 5$, $K_i = 10$의 PI 제어기로 제어할 때 크기 2의 계단을 추종하면 왜 윈드업이 일어나는지 설명하라. 역계산 안티 윈드업이 어떻게 도움이 되는지 기술하라.

### 연습 문제 4: 미분 여과

$K_p = 2$, $K_i = 1$, $K_d = 0.5$인 PID와 100 Hz 샘플 속도에서 표준편차 0.01의 백색 잡음을 가진 센서에 대해, 여과되지 않은 $D(s) = K_d s$와 $N = 10$인 여과된 $D(s) = K_d s / (1 + s/N)$에서 미분 항이 제어 신호에 기여하는 RMS 값을 계산하라.

### 연습 문제 5: 시뮬레이션

7절의 파이썬 시험대를 2차 플랜트 $G_p(s) = 4 / (s^2 + 2s + 4)$로 확장하라. Z-N 폐루프로 PID를 튜닝하고(스윕으로 $K_u$를 찾고 진동 주기에서 $T_u$를 계산) 결과 응답을 IMC-PI 설계와 비교하라.

---

*이전: [레슨 8 — Nyquist 안정성 판별법](08_Nyquist_Stability.md) | 다음: [레슨 10 — 리드-래그 보상](10_Lead_Lag_Compensation.md)*
