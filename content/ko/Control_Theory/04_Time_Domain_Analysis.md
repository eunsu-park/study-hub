# 레슨 4: 시간 영역 분석(Time-Domain Analysis)

## 학습 목표

- 1차 및 2차 시스템의 계단 응답(step response)을 계산하고 해석한다
- 시간 영역 사양(time-domain specification) — 상승 시간, 정착 시간, 오버슈트, 정상상태 오차 — 을 정의하고 계산한다
- 2차 시스템 파라미터 ($\zeta$, $\omega_n$)와 시간 영역 성능의 관계를 이해한다
- 최종값 정리(final value theorem)와 시스템 형(system type)을 이용하여 정상상태 오차를 분석한다
- 오차 상수(error constant) ($K_p$, $K_v$, $K_a$)를 적용하여 추종 정확도를 결정한다

## 1. 표준 시험 신호(Standard Test Signals)

제어 엔지니어는 표준 입력 신호를 사용하여 시스템 성능을 분석한다:

| 신호 | 시간 영역 | 라플라스 변환 |
|--------|------------|-------------------|
| 임펄스 $\delta(t)$ | $\delta(t)$ | $1$ |
| 계단 $u(t)$ | $u(t)$ | $\frac{1}{s}$ |
| 경사 $tu(t)$ | $tu(t)$ | $\frac{1}{s^2}$ |
| 포물선 $\frac{1}{2}t^2 u(t)$ | $\frac{1}{2}t^2 u(t)$ | $\frac{1}{s^3}$ |

**계단 응답(step response)**은 과도 및 정상상태 거동을 모두 드러내기 때문에 가장 널리 사용되는 시험 신호다.

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
- **우반평면(RHP) 영점** ($s = +z$): 초기 음의 응답(undershooot)을 유발 (비최소 위상(non-minimum phase) 거동)

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

---

*이전: [레슨 3 — 전달함수와 블록선도](03_Transfer_Functions_and_Block_Diagrams.md) | 다음: [레슨 5 — 안정성 분석](05_Stability_Analysis.md)*
