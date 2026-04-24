# 레슨 4: 시간 영역 분석(Time-Domain Analysis)

## 학습 목표

- 1차 및 2차 시스템의 계단 응답(step response)을 계산하고 해석한다
- 시간 영역 사양(time-domain specification) — 상승 시간, 정착 시간, 오버슈트, 정상상태 오차 — 을 정의하고 계산한다
- 2차 시스템 파라미터 ($\zeta$, $\omega_n$)와 시간 영역 성능의 관계를 이해한다
- 최종값 정리(final value theorem)와 시스템 형(system type)을 이용하여 정상상태 오차를 분석한다
- 오차 상수(error constant) ($K_p$, $K_v$, $K_a$)를 적용하여 추종 정확도를 결정한다
- 파이썬(Python)으로 계단 응답을 시뮬레이션하고 분석 시 흔히 빠지는 함정을 인식한다

## 0. 동기 — 시간 영역이 왜 중요한가

주파수 영역 도구(Bode, Nyquist, 근궤적)는 강력하지만, 이해 관계자가 "엘리베이터가 부드럽게 멈추는가?", "드론이 고도를 유지하는가?" 라고 물을 때 관심사는 **주파수**가 아니라 **시간**이다. 시간 영역 분석은 전달함수(transfer function)를 오실로스코프나 애니메이션으로 볼 수 있는 신호로 바꿔 준다.

시간 영역 사양이 설계를 지배하는 세 가지 예:

- **엘리베이터 승차감**: 10층 도착 시 5%의 오버슈트는 캐빈이 10층을 몇 센티미터 지나쳤다가 다시 내려온다는 의미 — 승객에게 불쾌감을 준다. 사양서에는 "$M_p \leq 2\%$"라고 적혀 있고, 설계자는 이를 제동비(damping ratio) $\zeta \geq 0.78$로 환산한다.
- **카메라 자동 초점**: $\zeta = 0.3$으로 구동되는 렌즈는 선명→흐림→선명으로 보이는 "헌팅(hunting)" 현상을 일으킨다. 사용자는 0.2초 안에 선명해지기를 기대하며, 이것이 정착 시간 $t_s$를 제약하고 $\zeta\omega_n$을 결정한다.
- **항공기 피치 제어**: 제동이 너무 약하면 조종사 유발 진동(pilot-induced oscillation)이 생기고, 너무 강하면 기체가 "물컹한(mushy)" 느낌이 된다. 인증 표준이 $\zeta \in [0.5, 0.9]$를 요구하는 이유는 오직 시간 영역에서만 표현 가능하다.

이 레슨의 나머지는 이 세 가지 질문에 정량적으로 답하기 위한 장치다. 읽는 동안 하나의 구체적인 물리 시스템을 염두에 두면 공식이 덜 추상적으로 느껴진다.

## 1. 표준 시험 신호(Standard Test Signals)

제어 엔지니어는 표준 입력 신호를 사용하여 시스템 성능을 분석한다:

| 신호 | 시간 영역 | 라플라스 변환 |
|--------|------------|-------------------|
| 임펄스 $\delta(t)$ | $\delta(t)$ | $1$ |
| 계단 $u(t)$ | $u(t)$ | $\frac{1}{s}$ |
| 경사 $tu(t)$ | $tu(t)$ | $\frac{1}{s^2}$ |
| 포물선 $\frac{1}{2}t^2 u(t)$ | $\frac{1}{2}t^2 u(t)$ | $\frac{1}{s^3}$ |

**계단 응답(step response)**은 과도 및 정상상태 거동을 모두 드러내기 때문에 가장 널리 사용되는 시험 신호다.

> **각 신호의 의미**: 계단(step) = 설정값의 급격한 변경(운전원이 버튼을 누름). 경사(ramp) = 일정 속도로 움직이는 목표(비행기를 추적하는 카메라). 포물선(parabola) = 일정 가속도 추적(드물지만 일부 레이더 시스템). 임펄스(impulse)는 수학적 탐침에 가깝다 — 그 응답은 시스템의 "지문"으로, 다른 모든 응답은 합성곱(convolution)을 통해 여기서 계산된다.

## 2. 1차 시스템 응답(First-Order System Response)

$$G(s) = \frac{K}{\tau s + 1}$$

### 2.1 계단 응답

단위 계단 입력 $R(s) = 1/s$에 대해:

$$Y(s) = \frac{K}{s(\tau s + 1)}$$

$$y(t) = K(1 - e^{-t/\tau})$$

**주요 특성:**
- 최종값: $K$ (DC 이득)
- $t = \tau$에서: $y(\tau) = K(1 - e^{-1}) = 0.632K$ (최종값의 63.2%)
- $t = 4\tau$에서: $y = 0.982K$ (98.2% — 사실상 정착)
- 오버슈트 없음, 진동 없음
- 상승 시간(10%에서 90%): $t_r = 2.2\tau$

### 2.2 임펄스 응답(Impulse Response)

$$y(t) = \frac{K}{\tau}e^{-t/\tau}$$

임펄스 응답은 동일한 시상수(time constant)로 지수적으로 감쇠한다.

### 2.3 계산 예제: RC 저역통과 필터

저항 $R = 1\,\text{k}\Omega$과 커패시터 $C = 10\,\mu\text{F}$를 직렬로 연결하고 출력을 커패시터 양단에서 측정하면 1차 저역통과 필터(low-pass filter)가 된다:

$$G(s) = \frac{1}{RCs + 1} \quad \Rightarrow \quad \tau = RC = 10\,\text{ms}, \ K = 1$$

입력에 5 V 계단을 가하면 커패시터 전압은 $y(t) = 5(1 - e^{-t/0.01})$을 따른다:

| $t$ | $y(t)$ | 최종값 비율 |
|-----|--------|-----------|
| $0$ | $0\,\text{V}$ | $0\%$ |
| $1\,\tau = 10\,\text{ms}$ | $3.16\,\text{V}$ | $63.2\%$ |
| $3\,\tau = 30\,\text{ms}$ | $4.75\,\text{V}$ | $95.0\%$ |
| $5\,\tau = 50\,\text{ms}$ | $4.97\,\text{V}$ | $99.3\%$ |

온도 계단에 반응하는 온도계, 새로운 수위로 차는 물탱크, 가열되는 모터 권선 모두 같은 수식을 따른다 — 에너지 저장 요소와 소산 요소가 결합된 임의의 시스템이다. 1차 형상을 알아보면 그 기저 구조도 알아본다.

## 3. 2차 시스템 응답(Second-Order System Response)

$$G(s) = \frac{\omega_n^2}{s^2 + 2\zeta\omega_n s + \omega_n^2}$$

계단 응답은 제동비(damping ratio) $\zeta$에 결정적으로 의존한다:

### 3.1 부족 제동(Underdamped) 경우 ($0 < \zeta < 1$)

극점: $s = -\sigma \pm j\omega_d$ (여기서 $\sigma = \zeta\omega_n$, $\omega_d = \omega_n\sqrt{1-\zeta^2}$)

$$y(t) = 1 - \frac{e^{-\sigma t}}{\sqrt{1-\zeta^2}}\sin(\omega_d t + \phi)$$

여기서 $\phi = \cos^{-1}\zeta$.

이것이 가장 일반적이고 흥미로운 경우로 — 응답이 진동하면서 최종값에 수렴한다.

### 3.2 임계 제동(Critically Damped) 경우 ($\zeta = 1$)

극점: $s = -\omega_n$ (중복)

$$y(t) = 1 - (1 + \omega_n t)e^{-\omega_n t}$$

진동 없이 가장 빠른 응답이다.

### 3.3 과제동(Overdamped) 경우 ($\zeta > 1$)

극점: $s = -\zeta\omega_n \pm \omega_n\sqrt{\zeta^2 - 1}$ (두 개의 서로 다른 실수 음수)

$$y(t) = 1 + \frac{1}{2}\left(\frac{e^{s_1 t}}{s_1/\omega_n^2} + \frac{e^{s_2 t}}{s_2/\omega_n^2}\right)$$

임계 제동보다 느리고, 진동 없음.

### 3.4 물리적 직관: 스프링-질량-댐퍼(Spring-Mass-Damper)

스프링 강성(stiffness) $k$와 점성 감쇠 계수(viscous damping coefficient) $b$를 가진 스프링 위의 질량 $m$은 다음 운동 방정식을 따른다:

$$m\ddot{x} + b\dot{x} + kx = F(t)$$

초기 조건이 0인 라플라스 변환을 취하면:

$$G(s) = \frac{X(s)}{F(s)} = \frac{1}{ms^2 + bs + k}$$

표준형과 대응시키면 각 파라미터의 물리적 의미가 드러난다:

$$\omega_n = \sqrt{\frac{k}{m}} \qquad \zeta = \frac{b}{2\sqrt{mk}}$$

이것이 **바로** 모든 제어 교재에 등장하는 정전적(canonical) 예제다 — 추상적 파라미터를 만질 수 있는 것에 고정해 주기 때문이다:

- **고유 진동수(natural frequency) $\omega_n$** = 시스템이 _원하는_ 진동 속도(스프링이 강할수록, 질량이 가벼울수록 빠름). $\omega_n = 2\pi \cdot 1\,\text{Hz}$인 자동차 서스펜션은 푹신하고, $\omega_n = 2\pi \cdot 3\,\text{Hz}$인 경주차 서스펜션은 단단하다.
- **제동비(damping ratio) $\zeta$** = 진동이 얼마나 빨리 사라지는가. $\zeta = 0$은 순수 스프링(영원히 울림), $\zeta = 1$은 닫히는 문의 댐퍼(오버슈트 없음). 자동차는 반응성과 안정성의 균형을 위해 $\zeta \approx 0.25\text{–}0.35$를 목표로 한다.

교재에서 $\zeta = 0.7$을 보면 세단 서스펜션을 떠올리자: 눈에 띄게 부드럽고, 승객이 거의 못 느낄 정도의 작은 오버슈트가 있다.

## 4. 시간 영역 사양(Time-Domain Specifications)

최종값 $y_{\text{final}}$을 갖는 시스템의 계단 응답에 대해:

```
y(t)
 ^
 |        M_p
 |    ┌────*────┐
 |   /  ╲      / ╲
 | /     ╲────     ───── y_final ──────
 |/            (within ±2% or ±5%)
 |
 ├──┤  ├──────────────────┤
 0  t_r       t_s                    t →
```

| 사양 | 기호 | 정의 |
|--------------|--------|------------|
| **상승 시간(Rise time)** | $t_r$ | 최종값의 10%에서 90%까지 도달하는 시간 |
| **최고 시간(Peak time)** | $t_p$ | 첫 번째 최고점에 도달하는 시간 |
| **최대 오버슈트(Maximum overshoot)** | $M_p$ | $\frac{y_{\max} - y_{\text{final}}}{y_{\text{final}}} \times 100\%$ |
| **정착 시간(Settling time)** | $t_s$ | 최종값의 $\pm 2\%$ (또는 $\pm 5\%$) 이내에 머무르기 시작하는 시간 |
| **정상상태 오차(Steady-state error)** | $e_{ss}$ | $\lim_{t\to\infty} [r(t) - y(t)]$ |

### 4.1 2차 부족 제동 시스템의 공식

$$t_r \approx \frac{1.8}{\omega_n} \quad \text{(근사값, } 0.3 < \zeta < 0.8\text{인 경우)}$$

$$t_p = \frac{\pi}{\omega_d} = \frac{\pi}{\omega_n\sqrt{1-\zeta^2}}$$

$$M_p = e^{-\pi\zeta/\sqrt{1-\zeta^2}} \times 100\%$$

$$t_s \approx \frac{4}{\zeta\omega_n} \quad \text{(2% 기준)} \qquad t_s \approx \frac{3}{\zeta\omega_n} \quad \text{(5% 기준)}$$

반복해서 등장하는 작은 참조표는 외워 둘 가치가 있다:

| $\zeta$ | $M_p$ | 전형적 용도 |
|---------|-------|-------------|
| $0.2$ | $52.7\%$ | 의도적으로 쓰이지 않음 — 진단용 |
| $0.4$ | $25.4\%$ | 약간의 오버슈트가 허용되는 공격적 서보 |
| $0.5$ | $16.3\%$ | 흔한 절충점 |
| $0.707$ | $4.3\%$ | "공학적 기본값" — 작은 오버슈트로 최소 $t_s$ |
| $0.8$ | $1.5\%$ | 정밀 위치 결정 |
| $1.0$ | $0\%$ | 임계 제동 — 오버슈트 없음, 비진동 중 가장 빠름 |

### 4.2 설계 함의(Design Implications)

이 공식들은 근본적인 절충 관계를 드러낸다:
- **빠른 응답** (더 큰 $\omega_n$) $\Rightarrow$ $t_r$, $t_p$, $t_s$ 감소, 그러나 더 많은 제어 노력 필요
- **낮은 오버슈트** (더 큰 $\zeta$) $\Rightarrow$ 더 작은 $M_p$, 그러나 느린 응답 (더 큰 $t_r$)
- $t_s$는 $\sigma = \zeta\omega_n$에 의존 — 정착 시간을 줄이려면 $\zeta$와 $\omega_n$ 모두 증가시켜야 함

**전형적인 설계 목표:** $\zeta \approx 0.4\text{–}0.8$ 범위가 속도와 오버슈트의 균형을 맞춘다.

### 4.3 극점 배치 관점(Pole Placement Perspective)

2차 사양은 $s$-평면 위의 영역에 직접 대응된다:

- $t_s \leq T_s$: 극점이 $\text{Re}(s) \leq -4/T_s$를 만족해야 함 (수직선 왼쪽)
- $M_p \leq M$: 극점이 $\zeta \geq \zeta_{\min}$을 만족해야 함 (원점에서의 쐐기 내부)
- $t_p \leq T_p$: 극점이 $\omega_d \geq \pi/T_p$를 만족해야 함 (수평선 위)

실현 가능한 영역은 이 제약 조건들의 **교집합**이다.

### 4.4 계산 예제: 위치 결정 서보(Positioning Servo)

위치 결정 테이블의 설계 사양:

- 정착 시간 $t_s \leq 0.5\,\text{s}$ (2% 기준)
- 오버슈트 $M_p \leq 10\%$

**1단계 — $M_p$를 $\zeta$로 변환**. $M_p = e^{-\pi\zeta/\sqrt{1-\zeta^2}}$을 역산하면

$$\zeta = \frac{-\ln(M_p)}{\sqrt{\pi^2 + \ln^2(M_p)}}$$

$M_p = 0.10$일 때: $\zeta \approx 0.591$. 여유를 주기 위해 $\zeta = 0.7$을 선택한다 ($M_p \approx 4.6\%$ — 사양에 충분히 여유 있음).

**2단계 — $t_s$를 $\omega_n$으로 변환**. $t_s \approx 4/(\zeta\omega_n)$에서:

$$\omega_n \geq \frac{4}{\zeta t_s} = \frac{4}{0.7 \times 0.5} = 11.43\,\text{rad/s}$$

여유를 위해 $\omega_n = 12\,\text{rad/s}$로 올림한다.

**3단계 — 파생 사양**. $\zeta = 0.7$, $\omega_n = 12$일 때:

- $t_p = \pi / (\omega_n\sqrt{1-\zeta^2}) = \pi / (12 \cdot 0.714) \approx 0.367\,\text{s}$
- $t_r \approx 1.8 / \omega_n = 0.15\,\text{s}$
- $t_s \approx 4 / (0.7 \cdot 12) = 0.476\,\text{s}$ ✓ 사양 만족
- $M_p \approx 4.6\%$ ✓ 사양 만족

**4단계 — 극점 배치**. 원하는 폐루프 극점은 $s = -\zeta\omega_n \pm j\omega_n\sqrt{1-\zeta^2} = -8.4 \pm j8.57$에 있다. 이후의 모든 설계 선택(제어기 이득, 보상기 구조)은 우세 극점을 이 위치에 두는 것을 목표로 한다.

## 5. 추가 극점 및 영점의 영향

### 5.1 추가 극점(Additional Poles)

$s = -p_3$에 위치한 세 번째 극점은 더 느린 성분을 추가한다:
- $|p_3| \gg \sigma$인 경우: 무시할 수 있는 영향 (극점이 우세 쌍에 비해 "빠름")
- $|p_3| \approx \sigma$인 경우: 상승 시간과 정착 시간을 크게 증가시킴
- 경험 법칙: $|p_3| > 5\sigma$이면 세 번째 극점을 무시할 수 있음

### 5.2 추가 영점(Additional Zeros)

$s = -z$에 위치한 영점은 오버슈트에 영향을 미친다:
- **우세 극점에 가까운 좌반평면(LHP) 영점**: 오버슈트를 증가시키고 응답을 빠르게 함
- **우세 극점에서 먼 좌반평면(LHP) 영점**: 무시할 수 있는 영향
- **우반평면(RHP) 영점** ($s = +z$): 초기 음의 응답(undershoot)을 유발 (비최소 위상(non-minimum phase) 거동)

> **현장의 비최소 위상**: 승압 컨버터(boost converter), 일부 항공기 피치 동역학, 그리고 거꾸로 선 진자(inverted pendulum) 카트의 위치는 모두 비최소 위상이다. 증상은 "조종간을 앞으로 밀면 기체가 먼저 아주 조금 떨어졌다가 올라간다"이다. 제어기 튜닝으로는 초기 역방향 응답을 제거할 수 없다 — 이를 우회해서 설계해야 한다.

## 6. 정상상태 오차 분석(Steady-State Error Analysis)

### 6.1 최종값 정리(Final Value Theorem)

$Y(s)$의 모든 극점이 ($s = 0$을 제외하고) 좌반평면에 있는 경우:

$$y(\infty) = \lim_{s \to 0} sY(s)$$

개루프 전달함수(open-loop transfer function) $G(s)$를 갖는 단위 귀환(unity-feedback) 시스템에서:

$$e_{ss} = \lim_{s \to 0} sE(s) = \lim_{s \to 0} \frac{sR(s)}{1 + G(s)}$$

### 6.2 시스템 형(System Type)

**시스템 형**은 개루프 전달함수에서 자유 적분기(free integrator) — $s = 0$에서의 극점 — 의 개수다:

$$G(s) = \frac{K \prod(s - z_i)}{s^N \prod(s - p_j)} \quad \Rightarrow \quad \text{형(Type) } N$$

### 6.3 오차 상수와 정상상태 오차

| 입력 | 오차 상수 | 공식 | 형 0 | 형 1 | 형 2 |
|-------|---------------|---------|--------|--------|--------|
| 계단 $1/s$ | $K_p = \lim_{s\to 0} G(s)$ | $\frac{1}{1+K_p}$ | $\frac{1}{1+K_p}$ | $0$ | $0$ |
| 경사 $1/s^2$ | $K_v = \lim_{s\to 0} sG(s)$ | $\frac{1}{K_v}$ | $\infty$ | $\frac{1}{K_v}$ | $0$ |
| 포물선 $1/s^3$ | $K_a = \lim_{s\to 0} s^2 G(s)$ | $\frac{1}{K_a}$ | $\infty$ | $\infty$ | $\frac{1}{K_a}$ |

**핵심 통찰:** 루프 내의 각 적분기는 한 단계 더 복잡한 입력에 대한 정상상태 오차를 제거한다. 그러나 적분기는 안정성에도 영향을 미치므로(위상 지연 추가), 절충 관계가 존재한다.

### 6.4 예제

$G(s) = \frac{100}{s(s+5)}$가 주어진 경우 (형 1 시스템):

- $K_p = \lim_{s\to 0} G(s) = \infty$ → 계단 오차 0
- $K_v = \lim_{s\to 0} sG(s) = 100/5 = 20$ → 경사 오차 = $1/20 = 5\%$
- $K_a = \lim_{s\to 0} s^2 G(s) = 0$ → 무한 포물선 오차

## 7. 파이썬(Python) 시뮬레이션

위 공식들은 우아하지만, 몇 줄의 `scipy.signal` 코드가 그 어떤 유도보다 빠르게 직관을 쌓아 준다. 아래 스니펫은 다섯 개의 대표적 $\zeta$ 값에 대한 계단 응답을 시뮬레이션하고 플롯한다 — `step_response.py`로 저장해 실행해 보자:

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

omega_n = 1.0
t = np.linspace(0, 15, 1500)

for zeta in [0.1, 0.4, 0.707, 1.0, 2.0]:
    num = [omega_n ** 2]
    den = [1, 2 * zeta * omega_n, omega_n ** 2]
    sys = signal.TransferFunction(num, den)
    _, y = signal.step(sys, T=t)
    plt.plot(t, y, label=f"ζ = {zeta}")

plt.axhline(1.0, color="gray", linestyle="--", linewidth=0.8)
plt.xlabel("time [s]")
plt.ylabel("output")
plt.title(f"Second-order step response (ωₙ = {omega_n} rad/s)")
plt.legend()
plt.grid(True)
plt.show()
```

실험해 보자: $\omega_n$을 2로 바꾸면 모든 시간축이 절반으로 줄어든다 — $\omega_n$을 스케일하는 것은 시간을 다시 스케일하는 것과 같다.

시뮬레이션된 응답에서 사양을 수치적으로 계산하려면:

```python
def compute_specs(t, y, y_final=1.0):
    """Return (overshoot%, peak_time, settle_time_2pct, rise_time_10_90)."""
    y_max = y.max()
    overshoot = (y_max - y_final) / y_final * 100.0

    peak_idx = int(np.argmax(y))
    peak_time = t[peak_idx]

    tolerance = 0.02 * y_final
    outside = np.where(np.abs(y - y_final) > tolerance)[0]
    settle_time = t[outside[-1]] if len(outside) else t[0]

    t10 = t[np.argmax(y >= 0.1 * y_final)]
    t90 = t[np.argmax(y >= 0.9 * y_final)]
    rise_time = t90 - t10

    return overshoot, peak_time, settle_time, rise_time
```

계산된 값은 4.1절의 공식과 몇 퍼센트 이내로 일치한다 — 차이는 선형 보간과 해석적 극값의 차이에서 온다.

## 8. 흔한 함정

학생(그리고 숙련된 엔지니어)이 반복해서 빠지는 실수들의 짧은 목록:

1. **3차 시스템에 2차 공식 적용.** $M_p = e^{-\pi\zeta/\sqrt{1-\zeta^2}}$ 공식은 정확히 하나의 복소 공액 극점 쌍이 있고 그 외에는 아무것도 없다고 가정한다. 시스템에 중요한 세 번째 극점이나 LHP 영점이 있으면 — 공식을 대입하지 말고 시뮬레이션하라.
2. **상승 시간 관례의 혼동.** 교재마다 0→100%(과제동), 10→90%, 혹은 $y_{\text{final}}$에 처음 도달하는 시간을 사용한다. 이들 간에 2배 차이가 날 수 있다. 사양서가 어떤 정의를 쓰는지 항상 확인한 후 약속하라.
3. **2% 대 5% 정착 기준 망각.** 2% 정착은 5% 정착보다 약 33% 더 걸린다. 같은 시스템, 다른 숫자 — 기준이 중요하다.
4. **불안정 시스템에 최종값 정리 적용.** 이 정리는 폐루프 모든 극점이 LHP에 있을 것을 요구한다. 극점이 $s = +2$에 있는 시스템에 적용하면 실제로는 출력이 발산하는데도 허위 "정상 상태"를 내놓는다.
5. **외란 오차 무시.** 오차 상수 $K_p, K_v, K_a$는 기준 신호 추종을 기술할 뿐이며 외란 제거에는 아무 말도 하지 않는다. 형 1 시스템은 계단 기준 입력에 대해 정상상태 오차가 0이지만 상수 외란에 대해서는 오차가 있을 수 있다 — 이들은 다른 계산이다.
6. **액추에이터 한계를 고려하지 않은 $\omega_n$ 설계.** $\omega_n = 50\,\text{rad/s}$인 자동차 서스펜션은 종이 위에서는 좋아 보이지만, 유압 램이 그 속도로 움직일 수 없다면 실현 불가다. 필요한 제어 노력을 항상 점검하라.

## 연습 문제

### 연습 문제 1: 2차 시스템 사양

단위 귀환 시스템의 개루프 전달함수가 다음과 같다:

$$G(s) = \frac{50}{s(s+5)}$$

1. 폐루프 전달함수를 구하라
2. $\omega_n$과 $\zeta$를 구하라
3. $M_p$, $t_p$, $t_s$ (2% 기준)를 계산하라
4. 단위 경사 입력에 대한 정상상태 오차를 계산하라

### 연습 문제 2: 우세 극점(Dominant Poles)

시스템의 폐루프 극점이 $s = -2 \pm j3$과 $s = -20$에 위치한다.

1. 세 번째 극점을 무시할 수 있는가? 근거를 제시하라.
2. 우세 2차 근사(dominant second-order approximation)를 사용하여 $M_p$와 $t_s$를 추정하라.

### 연습 문제 3: 시스템 형 설계

플랜트 $G_p(s) = 1/(s+2)$를 갖는 시스템에서 다음 조건을 만족하도록 제어기 $G_c(s) = K(s+a)/s$를 설계하라:
- 계단 입력에 대한 정상상태 오차가 0
- 단위 경사 입력에 대한 정상상태 오차 $\leq 0.02$

필요한 최솟값 $K$는 얼마인가?

### 연습 문제 4: 사양으로부터 역설계

위치 결정 서보가 $M_p \leq 5\%$와 $t_s \leq 0.2\,\text{s}$ (2% 기준)를 만족해야 한다. $(\zeta, \omega_n)$ 평면에서 허용 영역을 구하고, 그 내부의 한 쌍을 선택한 뒤 4.1절의 공식으로 검증하라.

### 연습 문제 5: 시뮬레이션과 비교

7절의 파이썬 스니펫을 사용해 $\omega_n = 4$, $\zeta \in \{0.3, 0.5, 0.7, 0.9\}$에 대한 계단 응답을 시뮬레이션하라. 시뮬레이션 곡선에서 $M_p$, $t_p$, $t_s$를 측정하고 해석적 공식과 비교하라. 5%보다 큰 차이가 있으면 그 이유를 설명하라.

---

*이전: [레슨 3 — 전달함수와 블록선도](03_Transfer_Functions_and_Block_Diagrams.md) | 다음: [레슨 5 — 안정성 분석](05_Stability_Analysis.md)*
