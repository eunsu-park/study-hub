# ODE 고급

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. ODE 시스템에서 강성(Stiffness)을 정의하고 강성 문제에서 명시적 방법이 실패하는 이유를 설명할 수 있습니다.
2. 암시적 방법(Implicit Method)(후진 오일러(Backward Euler), 사다리꼴 공식)을 구현하고 A-안정성(A-stability)을 달성하는 방법을 설명할 수 있습니다.
3. 강성 솔버(Radau, BDF)와 적절한 허용 오차 설정으로 `scipy.integrate.solve_ivp`를 적용할 수 있습니다.
4. 시스템의 고유값 구조를 분석하여 강성을 진단하고 적절한 솔버를 선택할 수 있습니다.
5. 화학 반응 동역학(Chemical Kinetics)과 전기 회로에서 발생하는 실용적인 강성 ODE 문제를 풀 수 있습니다.

---

## 개요

강성(stiff) 문제, 암시적 방법, scipy.integrate의 고급 사용법을 학습합니다. 실제 응용에서 자주 등장하는 어려운 ODE 문제들을 다룹니다.

**이 레슨이 중요한 이유:** 화학 반응, 전기 회로, 생물학적 네트워크 등 많은 실제 시스템은 매우 다른 시간 스케일에서 진행되는 과정을 포함합니다. 화학 반응에서 마이크로초 단위의 빠른 과도 현상과 수 시간에 걸친 느린 평형이 공존합니다. 단순한 명시적 솔버는 전체 시뮬레이션에 마이크로초 크기의 스텝을 사용해야 하므로 비실용적입니다. 암시적 방법(Implicit Method)과 강성 솔버(Stiff Solver)는 이러한 문제를 다루기 위한 필수 도구입니다.

---

## 1. 강성 문제 (Stiff Problems)

### 1.1 강성의 정의

시스템에 매우 다른 시간 스케일이 포함되어 있으면 **강성(Stiff)**이라고 합니다. 수학적으로, 강성은 야코비 행렬의 고유값 크기 비율로 특성화됩니다:

$$\text{강성비(Stiffness ratio)} = \frac{|\lambda_{\max}|}{|\lambda_{\min}|}$$

이 비율이 크면(예: 500 이상) 명시적 방법은 안정성을 위해 가장 빠른 모드가 결정하는 시간 스텝을 사용해야 합니다($\Delta t \lesssim 2/|\lambda_{\max}|$). 하지만 관심 있는 해는 느린 시간 스케일($1/|\lambda_{\min}|$)에서 진화하므로, 이미 감쇠한 성분에 대해 막대한 계산 낭비가 발생합니다.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

"""
강성 문제의 특징:
1. 해의 성분들이 매우 다른 시간 스케일을 가짐
2. 일부 성분은 빠르게 감쇠하고, 다른 성분은 느리게 변화
3. 명시적 방법 사용 시 안정성을 위해 매우 작은 스텝 필요
"""

def stiff_example():
    """강성 시스템 예시"""
    # dy₁/dt = -500*y₁ + 500*y₂
    # dy₂/dt = y₁ - y₂
    #
    # 고유값: λ₁ ≈ -500.002, λ₂ ≈ -0.998
    # 비율: |λ₁/λ₂| ≈ 500 (강성비)

    # 빠른 모드(λ₁ ≈ -500)는 ~0.002 시간 단위 내에 감쇠하지만, 느린 모드
    # (λ₂ ≈ -1)는 ~5 시간 단위가 필요합니다. RK45는 전체 시간 동안 빠른 모드의
    # 안정성 한계를 추적해야 하지만, Radau의 암시적 공식은 무조건 안정하므로
    # 과도 현상이 감쇠한 후에는 큰 스텝을 사용할 수 있습니다.
    def stiff_system(t, y):
        return [-500*y[0] + 500*y[1], y[0] - y[1]]

    y0 = [1.0, 0.0]
    t_span = (0, 5)

    # 명시적 RK45: 안정 영역이 dt를 ~2/500 = 0.004로 제한하여 많은 스텝 필요
    sol_rk45 = solve_ivp(stiff_system, t_span, y0, method='RK45',
                         rtol=1e-6, atol=1e-9)

    # 암시적 Radau: A-안정이므로 dt는 안정성이 아닌 정확도에 의해서만 제한됨
    sol_radau = solve_ivp(stiff_system, t_span, y0, method='Radau',
                          rtol=1e-6, atol=1e-9)

    print(f"RK45 스텝 수: {len(sol_rk45.t)}")
    print(f"Radau 스텝 수: {len(sol_radau.t)}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].semilogy(sol_rk45.t, np.abs(sol_rk45.y[0]), 'b-', label='y₁')
    axes[0].semilogy(sol_rk45.t, np.abs(sol_rk45.y[1]), 'r-', label='y₂')
    axes[0].set_xlabel('t')
    axes[0].set_ylabel('|y|')
    axes[0].set_title(f'RK45 ({len(sol_rk45.t)} 스텝)')
    axes[0].legend()
    axes[0].grid(True)

    axes[1].semilogy(sol_radau.t, np.abs(sol_radau.y[0]), 'b-', label='y₁')
    axes[1].semilogy(sol_radau.t, np.abs(sol_radau.y[1]), 'r-', label='y₂')
    axes[1].set_xlabel('t')
    axes[1].set_ylabel('|y|')
    axes[1].set_title(f'Radau ({len(sol_radau.t)} 스텝)')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()

stiff_example()
```

### 1.2 화학 반응 동역학

Robertson 문제는 강성 솔버의 대표적 벤치마크입니다. 세 반응 속도 상수가 9차수에 걸쳐 분포하여($k_1 = 0.04$, $k_2 = 3 \times 10^7$, $k_3 = 10^4$) 극심한 강성을 만듭니다. 종 B는 거의 즉각적으로 준정상 상태(Quasi-Steady State)에 도달하는 반면, A와 C는 수백만 시간 단위에 걸쳐 진화합니다.

```python
def chemical_kinetics():
    """강성 화학 반응 시스템 (Robertson 문제)"""
    # A → B (k₁ = 0.04)
    # B + B → C + B (k₂ = 3e7)
    # B + C → A + C (k₃ = 1e4)

    k1, k2, k3 = 0.04, 3e7, 1e4

    def robertson(t, y):
        A, B, C = y
        dA = -k1*A + k3*B*C
        dB = k1*A - k2*B*B - k3*B*C
        dC = k2*B*B
        return [dA, dB, dC]

    y0 = [1.0, 0.0, 0.0]
    t_span = (0, 1e7)
    t_eval = np.logspace(-5, 7, 200)

    # BDF는 고주파 과도 현상을 효율적으로 감쇠시키므로 선택됨.
    # 로그 간격 시간 샘플링은 해의 다중 스케일 거동과 일치:
    # 빠른 초기 과도 현상(t < 1) 이후 느린 평형화(t ~ 10^7).
    sol = solve_ivp(robertson, t_span, y0, method='BDF',
                    t_eval=t_eval, rtol=1e-8, atol=1e-10)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 선형 스케일
    axes[0].semilogx(sol.t, sol.y[0], label='A')
    axes[0].semilogx(sol.t, sol.y[2], label='C')
    axes[0].set_xlabel('t')
    axes[0].set_ylabel('농도')
    axes[0].set_title('Robertson 화학 반응 (A, C)')
    axes[0].legend()
    axes[0].grid(True)

    # B는 매우 작은 값이므로 별도 표시
    axes[1].semilogx(sol.t, sol.y[1] * 1e4, label='B × 10⁴')
    axes[1].set_xlabel('t')
    axes[1].set_ylabel('농도 × 10⁴')
    axes[1].set_title('Robertson 화학 반응 (B)')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()

chemical_kinetics()
```

---

## 2. 암시적 방법

**왜 암시적 방법인가?** 명시적 방법은 현재 시각의 $f$를 평가하여 미래를 예측합니다. 암시적 방법은 미래 시각의 $f$를 평가하므로 각 스텝에서 (일반적으로 비선형인) 방정식을 풀어야 합니다. 이 추가 비용으로 무조건적 안정성(Unconditional Stability)을 얻습니다: 스텝 크기에 관계없이 방법이 안정하므로, $\Delta t$를 안정성이 아닌 정확도 기준으로 선택할 수 있습니다.

### 2.1 후진 오일러 (다시 보기)

후진 오일러(Backward Euler)법은 가장 간단한 암시적 기법입니다. 각 스텝에서 다음을 풀어야 합니다:

$$y_{n+1} = y_n + h \cdot f(t_{n+1}, y_{n+1})$$

$y_{n+1}$이 양변에 나타나므로 뉴턴-랩슨(Newton-Raphson) 반복을 사용합니다. 잔차 $F(y_{n+1}) = y_{n+1} - y_n - h \cdot f(t_{n+1}, y_{n+1})$와 야코비안 $J_F = I - h \cdot J_f$를 정의하면, 뉴턴 업데이트 $\delta = -J_F^{-1} F$는 초기 추정이 가까울 때 이차적으로 수렴합니다.

```python
def backward_euler_system(f, jacobian, y0, t_span, n_steps, tol=1e-10):
    """
    시스템에 대한 후진 오일러 + 뉴턴-랩슨

    y_{n+1} = y_n + h * f(t_{n+1}, y_{n+1})
    """
    t = np.linspace(t_span[0], t_span[1], n_steps + 1)
    h = t[1] - t[0]
    n = len(y0)

    y = np.zeros((n_steps + 1, n))
    y[0] = y0

    for i in range(n_steps):
        # 현재 해를 뉴턴 반복의 초기 추정값으로 사용
        y_guess = y[i].copy()

        for _ in range(100):
            # 잔차: y_guess가 암시적 방정식을 만족하면 0이 되어야 함
            F = y_guess - y[i] - h * np.array(f(t[i+1], y_guess))
            # 뉴턴 야코비안: I - h*J_f; J*delta = -F를 푸는 것이 핵심 비용
            J = np.eye(n) - h * np.array(jacobian(t[i+1], y_guess))

            delta = np.linalg.solve(J, -F)
            y_guess = y_guess + delta

            # 이차 수렴: 가까워지면 각 반복이 오차를 제곱함
            if np.linalg.norm(delta) < tol:
                break

        y[i+1] = y_guess

    return t, y

# 테스트: 강성 시스템
def stiff_f(t, y):
    return [-500*y[0] + 500*y[1], y[0] - y[1]]

def stiff_jacobian(t, y):
    return [[-500, 500], [1, -1]]

t, y = backward_euler_system(stiff_f, stiff_jacobian, [1.0, 0.0], (0, 1), 50)

plt.figure(figsize=(10, 5))
plt.plot(t, y[:, 0], 'b-o', label='y₁')
plt.plot(t, y[:, 1], 'r-o', label='y₂')
plt.xlabel('t')
plt.ylabel('y')
plt.title('후진 오일러 (50 스텝)')
plt.legend()
plt.grid(True)
plt.show()
```

### 2.2 Crank-Nicolson 방법

```python
def crank_nicolson(f, jacobian, y0, t_span, n_steps, tol=1e-10):
    """
    Crank-Nicolson (트래피조이달 규칙)

    y_{n+1} = y_n + h/2 * (f(t_n, y_n) + f(t_{n+1}, y_{n+1}))

    2차 정확도, A-안정
    """
    t = np.linspace(t_span[0], t_span[1], n_steps + 1)
    h = t[1] - t[0]
    n = len(y0)

    y = np.zeros((n_steps + 1, n))
    y[0] = y0

    for i in range(n_steps):
        f_n = np.array(f(t[i], y[i]))
        y_guess = y[i] + h * f_n  # 초기 추측 (전진 오일러)

        for _ in range(100):
            f_new = np.array(f(t[i+1], y_guess))
            F = y_guess - y[i] - h/2 * (f_n + f_new)
            J = np.eye(n) - h/2 * np.array(jacobian(t[i+1], y_guess))

            delta = np.linalg.solve(J, -F)
            y_guess = y_guess + delta

            if np.linalg.norm(delta) < tol:
                break

        y[i+1] = y_guess

    return t, y

# 비교
t_be, y_be = backward_euler_system(stiff_f, stiff_jacobian, [1.0, 0.0], (0, 1), 20)
t_cn, y_cn = crank_nicolson(stiff_f, stiff_jacobian, [1.0, 0.0], (0, 1), 20)

# 참조 해
sol_ref = solve_ivp(stiff_f, (0, 1), [1.0, 0.0], method='Radau',
                    t_eval=np.linspace(0, 1, 200))

plt.figure(figsize=(10, 5))
plt.plot(sol_ref.t, sol_ref.y[0], 'k-', linewidth=2, label='참조')
plt.plot(t_be, y_be[:, 0], 'bo-', label='후진 오일러')
plt.plot(t_cn, y_cn[:, 0], 'rs-', label='Crank-Nicolson')
plt.xlabel('t')
plt.ylabel('y₁')
plt.title('암시적 방법 비교')
plt.legend()
plt.grid(True)
plt.show()
```

### 2.3 BDF 방법

```python
"""
BDF (Backward Differentiation Formula) 방법

BDF1 (후진 오일러):
    y_{n+1} - y_n = h * f(t_{n+1}, y_{n+1})

BDF2:
    (3y_{n+1} - 4y_n + y_{n-1}) / 2 = h * f(t_{n+1}, y_{n+1})

BDF3:
    (11y_{n+1} - 18y_n + 9y_{n-1} - 2y_{n-2}) / 6 = h * f(t_{n+1}, y_{n+1})

특징:
- A-안정 (BDF1, BDF2)
- 강성 문제에 효과적
- scipy의 'BDF' 솔버가 가변 차수 BDF 구현
"""

def bdf2_solver(f, jacobian, y0, t_span, n_steps, tol=1e-10):
    """BDF2 방법"""
    t = np.linspace(t_span[0], t_span[1], n_steps + 1)
    h = t[1] - t[0]
    n = len(y0)

    y = np.zeros((n_steps + 1, n))
    y[0] = y0

    # 첫 스텝은 후진 오일러로
    y_guess = y[0].copy()
    for _ in range(100):
        F = y_guess - y[0] - h * np.array(f(t[1], y_guess))
        J = np.eye(n) - h * np.array(jacobian(t[1], y_guess))
        delta = np.linalg.solve(J, -F)
        y_guess = y_guess + delta
        if np.linalg.norm(delta) < tol:
            break
    y[1] = y_guess

    # 이후 BDF2
    for i in range(1, n_steps):
        y_guess = y[i].copy()

        for _ in range(100):
            f_new = np.array(f(t[i+1], y_guess))
            # BDF2: (3y_{n+1} - 4y_n + y_{n-1})/2 = h*f(t_{n+1}, y_{n+1})
            F = 1.5*y_guess - 2*y[i] + 0.5*y[i-1] - h * f_new
            J = 1.5*np.eye(n) - h * np.array(jacobian(t[i+1], y_guess))

            delta = np.linalg.solve(J, -F)
            y_guess = y_guess + delta

            if np.linalg.norm(delta) < tol:
                break

        y[i+1] = y_guess

    return t, y

t_bdf2, y_bdf2 = bdf2_solver(stiff_f, stiff_jacobian, [1.0, 0.0], (0, 1), 20)

plt.figure(figsize=(10, 5))
plt.plot(sol_ref.t, sol_ref.y[0], 'k-', linewidth=2, label='참조')
plt.plot(t_bdf2, y_bdf2[:, 0], 'go-', label='BDF2')
plt.xlabel('t')
plt.ylabel('y₁')
plt.title('BDF2 방법')
plt.legend()
plt.grid(True)
plt.show()
```

---

## 3. scipy.integrate 고급

### 3.1 야코비안 제공

```python
def with_jacobian():
    """야코비안을 제공하면 성능 향상"""

    def system(t, y):
        return [
            -100*y[0] + y[1],
            y[0] - 100*y[1] + y[2],
            y[1] - 100*y[2]
        ]

    def jacobian(t, y):
        return [
            [-100, 1, 0],
            [1, -100, 1],
            [0, 1, -100]
        ]

    y0 = [1.0, 0.0, 0.0]
    t_span = (0, 0.5)

    import time

    # 야코비안 없이
    start = time.time()
    sol1 = solve_ivp(system, t_span, y0, method='Radau')
    time1 = time.time() - start

    # 야코비안 제공
    start = time.time()
    sol2 = solve_ivp(system, t_span, y0, method='Radau', jac=jacobian)
    time2 = time.time() - start

    print(f"야코비안 없이: {time1:.4f}초, {len(sol1.t)} 스텝")
    print(f"야코비안 제공: {time2:.4f}초, {len(sol2.t)} 스텝")

with_jacobian()
```

### 3.2 Dense Output

```python
def dense_output_example():
    """Dense output으로 연속적인 해 얻기"""

    def oscillator(t, y):
        return [y[1], -y[0]]

    sol = solve_ivp(oscillator, (0, 10), [1, 0],
                    method='RK45', dense_output=True)

    # 임의의 시간에서 해 평가
    t_dense = np.linspace(0, 10, 1000)
    y_dense = sol.sol(t_dense)

    plt.figure(figsize=(10, 5))
    plt.plot(t_dense, y_dense[0], 'b-', label='Dense output')
    plt.plot(sol.t, sol.y[0], 'ro', markersize=5, label='솔버 스텝')
    plt.xlabel('t')
    plt.ylabel('y')
    plt.title('Dense Output')
    plt.legend()
    plt.grid(True)
    plt.show()

dense_output_example()
```

### 3.3 질량 행렬 (DAE)

```python
def dae_example():
    """
    미분-대수 방정식 (DAE)

    M * y' = f(t, y)

    여기서 M이 특이 행렬이면 대수 구속조건 포함
    """

    # 예: 단진자 (제약조건: x² + y² = L²)
    # x'' = λx
    # y'' = λy - g
    # x² + y² = L²

    # 인덱스-1 형태로 변환
    g = 9.8
    L = 1.0

    def pendulum(t, y):
        x, y_pos, vx, vy, lam = y

        # 가속도
        ax = lam * x
        ay = lam * y_pos - g

        # 구속조건 (λ 결정)
        # 사실 이건 더 복잡한 처리가 필요...
        # 간단한 예시만 보여줌

        return [vx, vy, ax, ay, 0]

    # scipy에서는 mass_matrix 옵션으로 DAE 풀 수 있음 (Radau)
    # 여기서는 개념만 설명

    print("DAE는 scipy.integrate.solve_ivp의 Radau 솔버와")
    print("mass_matrix 옵션으로 풀 수 있습니다.")

dae_example()
```

---

## 4. 경계값 문제 (BVP)

### 4.1 슈팅 방법

```python
from scipy.optimize import brentq

def shooting_method():
    """
    슈팅 방법으로 BVP 풀기

    y'' = -y, y(0) = 0, y(π) = 0
    정확해: y = sin(x)
    """

    def ode(t, y):
        return [y[1], -y[0]]

    def shoot(initial_slope):
        """주어진 초기 기울기로 IVP 풀기"""
        sol = solve_ivp(ode, (0, np.pi), [0, initial_slope],
                        dense_output=True)
        return sol.sol(np.pi)[0]  # y(π) 반환

    # 올바른 초기 기울기 찾기
    slope = brentq(shoot, 0.5, 1.5)
    print(f"찾은 초기 기울기: {slope:.6f} (정확값: 1.0)")

    # 최종 해
    sol = solve_ivp(ode, (0, np.pi), [0, slope], dense_output=True)
    t = np.linspace(0, np.pi, 100)

    plt.figure(figsize=(10, 5))
    plt.plot(t, sol.sol(t)[0], 'b-', label='수치해')
    plt.plot(t, np.sin(t), 'r--', label='정확해 sin(x)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('슈팅 방법')
    plt.legend()
    plt.grid(True)
    plt.show()

shooting_method()
```

### 4.2 scipy.integrate.solve_bvp

```python
from scipy.integrate import solve_bvp

def bvp_example():
    """
    y'' + y = 0, y(0) = 0, y(π) = 0

    1차 시스템으로 변환:
    y₁' = y₂
    y₂' = -y₁
    """

    def ode(x, y):
        return np.vstack([y[1], -y[0]])

    def bc(ya, yb):
        return np.array([ya[0], yb[0]])  # y(0) = 0, y(π) = 0

    # 초기 추측
    x = np.linspace(0, np.pi, 10)
    y = np.zeros((2, x.size))
    y[0] = np.sin(x)  # 초기 추측

    sol = solve_bvp(ode, bc, x, y)

    x_plot = np.linspace(0, np.pi, 100)
    y_plot = sol.sol(x_plot)

    plt.figure(figsize=(10, 5))
    plt.plot(x_plot, y_plot[0], 'b-', label='수치해')
    plt.plot(x_plot, np.sin(x_plot), 'r--', label='정확해')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('solve_bvp')
    plt.legend()
    plt.grid(True)
    plt.show()

bvp_example()
```

### 4.3 비선형 BVP

```python
def nonlinear_bvp():
    """
    비선형 BVP: y'' = y² - 1
    y(0) = 0, y(1) = 1
    """

    def ode(x, y):
        return np.vstack([y[1], y[0]**2 - 1])

    def bc(ya, yb):
        return np.array([ya[0], yb[0] - 1])

    x = np.linspace(0, 1, 10)
    y = np.zeros((2, x.size))
    y[0] = x  # 선형 초기 추측

    sol = solve_bvp(ode, bc, x, y)

    if sol.success:
        x_plot = np.linspace(0, 1, 100)
        y_plot = sol.sol(x_plot)

        plt.figure(figsize=(10, 5))
        plt.plot(x_plot, y_plot[0], 'b-', linewidth=2)
        plt.scatter([0, 1], [0, 1], color='red', s=100, zorder=5)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title("비선형 BVP: y'' = y² - 1")
        plt.grid(True)
        plt.show()
    else:
        print("수렴 실패")

nonlinear_bvp()
```

---

## 5. 특수 문제

### 5.1 주기 궤도 찾기

```python
def find_periodic_orbit():
    """Van der Pol 진동자의 주기 궤도"""
    mu = 1.0

    def vdp(t, y):
        return [y[1], mu * (1 - y[0]**2) * y[1] - y[0]]

    # 포앙카레 단면
    def poincare(t, y):
        return y[1]  # y' = 0일 때

    poincare.direction = -1  # 위에서 아래로 교차

    # 충분히 긴 시간 적분
    sol = solve_ivp(vdp, (0, 100), [2, 0], events=poincare,
                    dense_output=True, max_step=0.01)

    # 리밋 사이클에 수렴 후의 교차점들
    crossings = sol.y_events[0][-5:, 0]  # 마지막 5개 교차점의 x 값
    print(f"리밋 사이클 진폭: {np.mean(crossings):.6f}")

    # 마지막 주기 시각화
    t_last = sol.t_events[0][-2:]
    period = t_last[1] - t_last[0]
    print(f"주기: {period:.6f}")

    t_plot = np.linspace(t_last[0], t_last[1], 200)
    y_plot = sol.sol(t_plot)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(y_plot[0], y_plot[1])
    axes[0].set_xlabel('x')
    axes[0].set_ylabel("x'")
    axes[0].set_title('리밋 사이클')
    axes[0].grid(True)

    axes[1].plot(t_plot - t_last[0], y_plot[0])
    axes[1].set_xlabel('t (주기 내)')
    axes[1].set_ylabel('x')
    axes[1].set_title(f'한 주기 (T = {period:.4f})')
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()

find_periodic_orbit()
```

### 5.2 분기 분석

```python
def bifurcation_analysis():
    """파라미터에 따른 정상 상태 변화"""
    # dy/dt = r*y - y³
    # 평형점: y* = 0 또는 y* = ±√r (r > 0)

    r_values = np.linspace(-1, 2, 100)

    plt.figure(figsize=(10, 6))

    # 안정 분기
    r_pos = r_values[r_values >= 0]
    plt.plot(r_pos, np.sqrt(r_pos), 'b-', linewidth=2, label='안정')
    plt.plot(r_pos, -np.sqrt(r_pos), 'b-', linewidth=2)

    # 불안정 분기
    plt.plot(r_values[r_values <= 0], np.zeros_like(r_values[r_values <= 0]),
             'r--', linewidth=2, label='불안정')
    plt.plot(r_values[r_values > 0], np.zeros_like(r_values[r_values > 0]),
             'r--', linewidth=2)

    plt.xlabel('r')
    plt.ylabel('y*')
    plt.title("피치포크 분기: y' = ry - y³")
    plt.legend()
    plt.grid(True)
    plt.axhline(y=0, color='k', linewidth=0.5)
    plt.axvline(x=0, color='k', linewidth=0.5)
    plt.show()

bifurcation_analysis()
```

---

## 연습 문제

### 문제 1
다음 강성 시스템을 BDF와 Radau로 풀고 비교하세요:
dy₁/dt = -1000*y₁ + y₂
dy₂/dt = 999*y₁ - 2*y₂
y₁(0) = 1, y₂(0) = 0

```python
def exercise_1():
    def system(t, y):
        return [-1000*y[0] + y[1], 999*y[0] - 2*y[1]]

    y0 = [1.0, 0.0]
    t_span = (0, 1)

    sol_bdf = solve_ivp(system, t_span, y0, method='BDF',
                        t_eval=np.linspace(0, 1, 100))
    sol_radau = solve_ivp(system, t_span, y0, method='Radau',
                          t_eval=np.linspace(0, 1, 100))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].semilogy(sol_bdf.t, np.abs(sol_bdf.y[0]), label='y₁')
    axes[0].semilogy(sol_bdf.t, np.abs(sol_bdf.y[1]), label='y₂')
    axes[0].set_title('BDF')
    axes[0].legend()
    axes[0].grid(True)

    axes[1].semilogy(sol_radau.t, np.abs(sol_radau.y[0]), label='y₁')
    axes[1].semilogy(sol_radau.t, np.abs(sol_radau.y[1]), label='y₂')
    axes[1].set_title('Radau')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()

exercise_1()
```

---

## 요약

| 문제 유형 | 권장 솔버 | 특징 |
|----------|----------|------|
| 일반 비강성 | RK45, DOP853 | 명시적, 빠름 |
| 강성 문제 | Radau, BDF | 암시적, 안정 |
| DAE | Radau + mass_matrix | 대수 구속조건 |
| BVP | solve_bvp, 슈팅 | 경계조건 |

| 암시적 방법 | 차수 | A-안정 |
|------------|------|--------|
| 후진 오일러 | 1 | O |
| Crank-Nicolson | 2 | O |
| BDF1-2 | 1-2 | O |
| BDF3-6 | 3-6 | X |
