# 14. 비압축성 유동 (Incompressible Flow)

## 학습 목표
- 유동함수-와도 정식화 이해
- Lid-Driven Cavity 문제 구현
- 압력-속도 결합 문제 이해
- SIMPLE 알고리즘 기초 학습
- 엇갈린 격자 (Staggered Grid) 개념 파악

---

## 1. 비압축성 Navier-Stokes 방정식

### 1.1 원시변수 정식화 (Primitive Variable Formulation)

```
비압축성 NS 방정식 (원시변수: u, v, p):

연속방정식:
∂u/∂x + ∂v/∂y = 0

운동량 방정식:
∂u/∂t + u∂u/∂x + v∂u/∂y = -1/ρ ∂p/∂x + ν(∂²u/∂x² + ∂²u/∂y²)
∂v/∂t + u∂v/∂x + v∂v/∂y = -1/ρ ∂p/∂y + ν(∂²v/∂x² + ∂²v/∂y²)

문제점:
- 3개 방정식, 3개 미지수 (u, v, p)
- 압력에 대한 독립적 방정식 없음
- 압력-속도 결합 (coupling) 문제
```

### 1.2 압력 Poisson 방정식

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

"""
연속방정식의 발산을 취하면:
∇²p = -ρ∇·(u·∇u)

정상 상태에서:
∇²p = ρ[(∂u/∂x)² + 2(∂u/∂y)(∂v/∂x) + (∂v/∂y)²]

이 방정식으로 압력을 결정할 수 있음
"""

def pressure_poisson_concept():
    """압력 Poisson 방정식 개념 시각화"""

    print("=" * 60)
    print("압력-속도 결합 문제")
    print("=" * 60)

    explanation = """
    비압축성 유동의 핵심 문제:

    1. 압력의 역할:
       - 압력은 연속방정식(∇·u = 0)을 만족시키기 위해 조정됨
       - 압력파가 무한히 빠르게 전파 (비압축성)
       - 압력에 대한 독립적 지배방정식 없음

    2. 해결 방법:
       a) 유동함수-와도 정식화:
          - 연속방정식 자동 만족
          - 2D 유동에만 적용 가능

       b) 압력 Poisson 방정식:
          - 연속방정식에서 압력 방정식 유도
          - Projection/Fractional Step Method

       c) SIMPLE 계열 알고리즘:
          - 압력-속도 반복 보정
          - 산업계 표준
    """
    print(explanation)

pressure_poisson_concept()
```

---

## 2. 유동함수-와도 정식화 (Stream Function-Vorticity)

### 2.1 정의

```
2D 비압축성 유동:

유동함수 ψ (Stream Function):
u = ∂ψ/∂y,  v = -∂ψ/∂x
→ 연속방정식 자동 만족: ∂u/∂x + ∂v/∂y = 0

와도 ω (Vorticity):
ω = ∂v/∂x - ∂u/∂y = -∇²ψ

와도 수송 방정식:
∂ω/∂t + u∂ω/∂x + v∂ω/∂y = ν∇²ω

Poisson 방정식:
∇²ψ = -ω

장점: 압력항 제거, 연속방정식 자동 만족
단점: 2D에만 적용, 경계조건 복잡
```

```python
def stream_function_vorticity_derivation():
    """유동함수-와도 정식화 유도"""

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # (1) 유동함수 개념
    ax1 = axes[0]

    # 유선 (등 ψ 선)
    x = np.linspace(-2, 2, 100)
    y = np.linspace(-2, 2, 100)
    X, Y = np.meshgrid(x, y)

    # 예: 균일 유동 + 이중점 (Doublet)
    U_inf = 1.0
    kappa = 1.0  # 이중점 강도
    r2 = X**2 + Y**2 + 0.01
    psi = U_inf * Y - kappa * Y / r2

    # 속도장
    u = U_inf + kappa * (X**2 - Y**2) / r2**2 * 2 * kappa
    v = -2 * kappa * X * Y / r2**2 * 2 * kappa

    # 간소화된 속도장
    u = np.gradient(psi, y, axis=0)
    v = -np.gradient(psi, x, axis=1)

    levels = np.linspace(-3, 3, 21)
    cs = ax1.contour(X, Y, psi, levels=levels, colors='blue', linewidths=0.5)
    ax1.streamplot(X, Y, u, v, color='red', density=1, linewidth=0.5)

    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title(r'유동함수 ψ (등ψ선 = 유선)')
    ax1.set_aspect('equal')
    ax1.set_xlim(-2, 2)
    ax1.set_ylim(-2, 2)

    # (2) 와도 개념
    ax2 = axes[1]

    # 와류 (Vortex) 예제
    Gamma = 2 * np.pi  # 순환
    r = np.sqrt(X**2 + Y**2) + 0.1
    theta = np.arctan2(Y, X)

    # Rankine 와류 (코어 반경 = 0.5)
    r_core = 0.5
    omega = np.where(r < r_core,
                    Gamma / (np.pi * r_core**2),  # 고체 회전
                    0)  # 포텐셜 유동

    im = ax2.pcolormesh(X, Y, omega, cmap='RdBu_r', shading='auto')
    plt.colorbar(im, ax=ax2, label=r'$\omega$ [1/s]')

    # 속도장 (접선방향)
    u_theta = np.where(r < r_core,
                      Gamma * r / (2 * np.pi * r_core**2),
                      Gamma / (2 * np.pi * r))
    u_vortex = -u_theta * np.sin(theta)
    v_vortex = u_theta * np.cos(theta)

    skip = 5
    ax2.quiver(X[::skip, ::skip], Y[::skip, ::skip],
              u_vortex[::skip, ::skip], v_vortex[::skip, ::skip],
              color='black', alpha=0.7)

    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_title(r'와도 ω (Rankine 와류)')
    ax2.set_aspect('equal')
    ax2.set_xlim(-2, 2)
    ax2.set_ylim(-2, 2)

    plt.tight_layout()
    plt.savefig('stream_vorticity.png', dpi=150, bbox_inches='tight')
    plt.show()

# stream_function_vorticity_derivation()
```

---

## 3. Lid-Driven Cavity 문제

### 3.1 문제 정의

```
Lid-Driven Cavity:
- 정사각형 공동 (cavity)
- 상단 벽면이 일정 속도로 이동
- 나머지 벽면은 정지

경계조건:
- 상단 (y=H): u = U_lid, v = 0
- 하단 (y=0): u = v = 0
- 좌측 (x=0): u = v = 0
- 우측 (x=L): u = v = 0

특성:
- 레이놀즈 수에 따른 유동 패턴 변화
- Re 낮음: 하나의 주 와류
- Re 높음: 코너 와류 발생
- CFD 코드 검증용 표준 문제
```

```python
def lid_driven_cavity_stream_vorticity():
    """
    Lid-Driven Cavity 시뮬레이션
    유동함수-와도 정식화
    """

    # 격자 설정
    N = 41  # 격자점 수 (홀수)
    L = 1.0  # 공동 크기
    h = L / (N - 1)

    x = np.linspace(0, L, N)
    y = np.linspace(0, L, N)
    X, Y = np.meshgrid(x, y)

    # 물성치 및 설정
    Re = 100  # 레이놀즈 수
    U_lid = 1.0
    nu = U_lid * L / Re

    # 시간 설정
    dt = 0.001
    n_steps = 10000

    # 초기화
    psi = np.zeros((N, N))  # 유동함수
    omega = np.zeros((N, N))  # 와도

    # CFL 확인
    CFL = U_lid * dt / h
    print(f"Re = {Re}, N = {N}, h = {h:.4f}")
    print(f"dt = {dt}, CFL = {CFL:.4f}")

    def apply_bc_psi(psi):
        """유동함수 경계조건: ψ = 0 on walls"""
        psi[0, :] = 0   # 하단
        psi[-1, :] = 0  # 상단
        psi[:, 0] = 0   # 좌측
        psi[:, -1] = 0  # 우측
        return psi

    def apply_bc_omega(omega, psi, h, U_lid):
        """
        와도 경계조건 (Thom's formula):
        벽면에서 ω = -2(ψ_neighbor - ψ_wall)/h²
        """
        # 하단 (no-slip)
        omega[0, :] = -2 * psi[1, :] / h**2

        # 상단 (moving lid)
        omega[-1, :] = -2 * psi[-2, :] / h**2 - 2 * U_lid / h

        # 좌측
        omega[:, 0] = -2 * psi[:, 1] / h**2

        # 우측
        omega[:, -1] = -2 * psi[:, -2] / h**2

        return omega

    def solve_poisson(psi, omega, h, n_iter=50, tol=1e-6):
        """
        Poisson 방정식 풀이: ∇²ψ = -ω
        Gauss-Seidel 반복
        """
        for _ in range(n_iter):
            psi_old = psi.copy()

            for i in range(1, N-1):
                for j in range(1, N-1):
                    psi[i, j] = 0.25 * (psi[i+1, j] + psi[i-1, j] +
                                       psi[i, j+1] + psi[i, j-1] +
                                       h**2 * omega[i, j])

            # 경계조건
            psi = apply_bc_psi(psi)

            # 수렴 체크
            if np.max(np.abs(psi - psi_old)) < tol:
                break

        return psi

    def compute_velocity(psi, h):
        """유동함수에서 속도 계산"""
        u = np.zeros_like(psi)
        v = np.zeros_like(psi)

        # 내부점 (중심차분)
        for i in range(1, N-1):
            for j in range(1, N-1):
                u[i, j] = (psi[i+1, j] - psi[i-1, j]) / (2 * h)
                v[i, j] = -(psi[i, j+1] - psi[i, j-1]) / (2 * h)

        return u, v

    def advect_diffuse_omega(omega, u, v, nu, h, dt):
        """와도 수송 방정식 시간 전진"""
        omega_new = omega.copy()

        for i in range(1, N-1):
            for j in range(1, N-1):
                # 대류항 (풍상차분)
                if u[i, j] > 0:
                    domega_dx = (omega[i, j] - omega[i, j-1]) / h
                else:
                    domega_dx = (omega[i, j+1] - omega[i, j]) / h

                if v[i, j] > 0:
                    domega_dy = (omega[i, j] - omega[i-1, j]) / h
                else:
                    domega_dy = (omega[i+1, j] - omega[i, j]) / h

                convection = u[i, j] * domega_dx + v[i, j] * domega_dy

                # 확산항 (중심차분)
                diffusion = nu * ((omega[i+1, j] - 2*omega[i, j] + omega[i-1, j]) / h**2 +
                                 (omega[i, j+1] - 2*omega[i, j] + omega[i, j-1]) / h**2)

                omega_new[i, j] = omega[i, j] + dt * (-convection + diffusion)

        return omega_new

    # 시간 전진
    print("\n시뮬레이션 시작...")

    for n in range(n_steps):
        # 1. Poisson 방정식 풀이
        psi = solve_poisson(psi, omega, h)

        # 2. 속도 계산
        u, v = compute_velocity(psi, h)

        # 3. 와도 경계조건
        omega = apply_bc_omega(omega, psi, h, U_lid)

        # 4. 와도 수송
        omega = advect_diffuse_omega(omega, u, v, nu, h, dt)

        # 진행 상황 출력
        if n % 2000 == 0:
            print(f"Step {n}: max|ω| = {np.max(np.abs(omega)):.4f}, "
                  f"max|ψ| = {np.max(np.abs(psi)):.6f}")

    print("시뮬레이션 완료!")

    # 최종 속도장
    u, v = compute_velocity(psi, h)

    # 결과 시각화
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    # (1) 유선
    ax1 = axes[0, 0]
    levels = np.linspace(psi.min(), psi.max(), 30)
    cs = ax1.contour(X, Y, psi, levels=levels, colors='blue', linewidths=0.5)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title(f'유선 (Streamlines), Re = {Re}')
    ax1.set_aspect('equal')

    # (2) 와도 분포
    ax2 = axes[0, 1]
    vmax = np.max(np.abs(omega)) * 0.8
    im = ax2.pcolormesh(X, Y, omega, cmap='RdBu_r', shading='auto',
                       vmin=-vmax, vmax=vmax)
    plt.colorbar(im, ax=ax2, label=r'$\omega$')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_title('와도 분포')
    ax2.set_aspect('equal')

    # (3) 속도 벡터
    ax3 = axes[1, 0]
    skip = 2
    speed = np.sqrt(u**2 + v**2)
    ax3.quiver(X[::skip, ::skip], Y[::skip, ::skip],
              u[::skip, ::skip], v[::skip, ::skip],
              speed[::skip, ::skip], cmap='jet', scale=20)
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    ax3.set_title('속도 벡터장')
    ax3.set_aspect('equal')

    # (4) 중심선 속도 프로파일
    ax4 = axes[1, 1]

    # 수직 중심선 (x = 0.5)에서 u 분포
    j_center = N // 2
    u_centerline = u[:, j_center]

    # Ghia et al. (1982) 참조값 (Re=100)
    y_ghia = np.array([0.0000, 0.0547, 0.0625, 0.0703, 0.1016, 0.1719,
                      0.2813, 0.4531, 0.5000, 0.6172, 0.7344, 0.8516, 0.9531, 1.0000])
    u_ghia = np.array([0.0000, -0.0372, -0.0419, -0.0477, -0.0643, -0.1015,
                      -0.1566, -0.2109, -0.2058, -0.1364, 0.0033, 0.2315, 0.6872, 1.0000])

    ax4.plot(u_centerline, y, 'b-', linewidth=2, label='Present')
    ax4.plot(u_ghia, y_ghia, 'ro', markersize=6, label='Ghia et al. (1982)')
    ax4.set_xlabel('u')
    ax4.set_ylabel('y')
    ax4.set_title('수직 중심선 속도 프로파일 (x = 0.5)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('lid_driven_cavity.png', dpi=150, bbox_inches='tight')
    plt.show()

    return psi, omega, u, v, X, Y

# psi, omega, u, v, X, Y = lid_driven_cavity_stream_vorticity()
```

### 3.2 레이놀즈 수 효과

```python
def reynolds_effect_cavity():
    """다양한 레이놀즈 수에서 Lid-Driven Cavity 비교"""

    reynolds_numbers = [100, 400, 1000]
    results = []

    N = 51
    L = 1.0
    h = L / (N - 1)

    for Re in reynolds_numbers:
        print(f"\n=== Re = {Re} ===")

        # 설정
        U_lid = 1.0
        nu = U_lid * L / Re
        dt = min(0.001, 0.25 * h**2 / nu)  # 확산 안정성
        n_steps = int(20 / dt)  # 충분한 시간

        # 초기화
        psi = np.zeros((N, N))
        omega = np.zeros((N, N))

        # 간략화된 시뮬레이션 (빠른 실행)
        for n in range(min(n_steps, 5000)):
            # Poisson
            for _ in range(20):
                psi_old = psi.copy()
                psi[1:-1, 1:-1] = 0.25 * (psi[2:, 1:-1] + psi[:-2, 1:-1] +
                                         psi[1:-1, 2:] + psi[1:-1, :-2] +
                                         h**2 * omega[1:-1, 1:-1])
                psi[0, :] = psi[-1, :] = psi[:, 0] = psi[:, -1] = 0

            # 속도
            u = np.zeros((N, N))
            v = np.zeros((N, N))
            u[1:-1, 1:-1] = (psi[2:, 1:-1] - psi[:-2, 1:-1]) / (2*h)
            v[1:-1, 1:-1] = -(psi[1:-1, 2:] - psi[1:-1, :-2]) / (2*h)

            # 와도 경계조건
            omega[0, :] = -2 * psi[1, :] / h**2
            omega[-1, :] = -2 * psi[-2, :] / h**2 - 2 * U_lid / h
            omega[:, 0] = -2 * psi[:, 1] / h**2
            omega[:, -1] = -2 * psi[:, -2] / h**2

            # 와도 전진 (FTCS + 풍상)
            omega_new = omega.copy()
            for i in range(1, N-1):
                for j in range(1, N-1):
                    conv_x = u[i,j] * (omega[i,j] - omega[i,j-1])/h if u[i,j]>0 else u[i,j] * (omega[i,j+1] - omega[i,j])/h
                    conv_y = v[i,j] * (omega[i,j] - omega[i-1,j])/h if v[i,j]>0 else v[i,j] * (omega[i+1,j] - omega[i,j])/h
                    diff = nu * ((omega[i+1,j] - 2*omega[i,j] + omega[i-1,j]) +
                                (omega[i,j+1] - 2*omega[i,j] + omega[i,j-1])) / h**2
                    omega_new[i,j] = omega[i,j] + dt * (-conv_x - conv_y + diff)
            omega = omega_new

        results.append((Re, psi.copy(), omega.copy(), u.copy(), v.copy()))
        print(f"  완료: max|ψ| = {np.max(np.abs(psi)):.6f}")

    # 비교 시각화
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    x = np.linspace(0, L, N)
    y = np.linspace(0, L, N)
    X, Y = np.meshgrid(x, y)

    for idx, (Re, psi, omega, u, v) in enumerate(results):
        # 유선
        ax = axes[0, idx]
        levels = np.linspace(psi.min(), psi.max(), 25)
        ax.contour(X, Y, psi, levels=levels, colors='blue', linewidths=0.5)
        ax.set_title(f'Re = {Re}')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_aspect('equal')

        # 와도
        ax = axes[1, idx]
        vmax = min(5, np.max(np.abs(omega)))
        ax.pcolormesh(X, Y, omega, cmap='RdBu_r', shading='auto',
                     vmin=-vmax, vmax=vmax)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_aspect('equal')

    axes[0, 0].set_ylabel('유선')
    axes[1, 0].set_ylabel('와도')

    plt.suptitle('레이놀즈 수에 따른 Lid-Driven Cavity 유동 패턴', fontsize=14)
    plt.tight_layout()
    plt.savefig('cavity_reynolds_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()

# reynolds_effect_cavity()
```

---

## 4. 엇갈린 격자 (Staggered Grid)

### 4.1 동일 위치 격자의 문제점

```
동일 위치 격자 (Collocated Grid):
- 모든 변수 (u, v, p)가 같은 위치에 저장
- 압력의 중심차분: (p_{i+1} - p_{i-1})/(2Δx)
- 문제: 체커보드 (checkerboard) 불안정성

체커보드 현상:
- 압력 진동이 수치해에 영향 없음
- p(i) = p0 + (-1)^i * ε 형태의 진동
- 중심차분에서 진동이 상쇄됨
```

```python
def checkerboard_problem():
    """체커보드 불안정성 시각화"""

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # (1) 동일 위치 격자
    ax1 = axes[0]
    n = 6
    for i in range(n):
        for j in range(n):
            ax1.plot(i, j, 'ko', markersize=10)
            ax1.text(i, j+0.2, 'u,v,p', fontsize=6, ha='center')

    ax1.set_xlim(-0.5, n-0.5)
    ax1.set_ylim(-0.5, n)
    ax1.set_title('동일 위치 격자 (Collocated)')
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)

    # (2) 체커보드 압력
    ax2 = axes[1]
    n = 8
    x = np.arange(n)
    y = np.arange(n)
    X, Y = np.meshgrid(x, y)
    P = (-1) ** (X + Y)  # 체커보드 패턴

    im = ax2.pcolormesh(X-0.5, Y-0.5, P, cmap='RdBu_r', shading='auto')
    ax2.set_title('체커보드 압력 분포')
    ax2.set_aspect('equal')
    plt.colorbar(im, ax=ax2)

    # 압력 구배 (중심차분 = 0!)
    dpdx = np.zeros_like(P)
    for i in range(1, n-1):
        for j in range(1, n-1):
            dpdx[i, j] = (P[i, j+1] - P[i, j-1]) / 2  # 항상 0!

    ax2.set_xlabel('중심차분 ∂p/∂x = 0 (잘못됨!)')

    # (3) 엇갈린 격자
    ax3 = axes[2]
    n = 5

    # 압력 점 (셀 중심)
    for i in range(n):
        for j in range(n):
            ax3.plot(i+0.5, j+0.5, 'ro', markersize=10, label='p' if i==0 and j==0 else '')

    # u-속도 점 (수직 면)
    for i in range(n+1):
        for j in range(n):
            ax3.plot(i, j+0.5, 'b>', markersize=8, label='u' if i==0 and j==0 else '')

    # v-속도 점 (수평 면)
    for i in range(n):
        for j in range(n+1):
            ax3.plot(i+0.5, j, 'g^', markersize=8, label='v' if i==0 and j==0 else '')

    # 격자선
    for i in range(n+1):
        ax3.axvline(x=i, color='gray', linestyle='-', linewidth=0.5)
        ax3.axhline(y=i, color='gray', linestyle='-', linewidth=0.5)

    ax3.set_xlim(-0.3, n+0.3)
    ax3.set_ylim(-0.3, n+0.3)
    ax3.set_title('엇갈린 격자 (Staggered)')
    ax3.legend(loc='upper right')
    ax3.set_aspect('equal')

    plt.tight_layout()
    plt.savefig('staggered_grid.png', dpi=150, bbox_inches='tight')
    plt.show()

# checkerboard_problem()
```

### 4.2 엇갈린 격자 구조

```
엇갈린 격자 (MAC Grid):

     v(i,j+1)
        │
  ──────┼──────
  │     │     │
  │  p(i,j)   │
u(i,j)─┼──u(i+1,j)
  │     │     │
  │     │     │
  ──────┼──────
     v(i,j)

저장 위치:
- p: 셀 중심 (i, j)
- u: 셀 동쪽 면 (i+1/2, j)
- v: 셀 북쪽 면 (i, j+1/2)

장점:
- 체커보드 방지
- 질량 보존 정확
- 압력-속도 결합 자연스러움
```

---

## 5. SIMPLE 알고리즘

### 5.1 개념

```
SIMPLE (Semi-Implicit Method for Pressure-Linked Equations):

기본 아이디어:
1. 추측 압력장 p* 사용
2. 추측 속도장 u*, v* 계산 (운동량 방정식)
3. 압력 보정 p' 계산 (연속방정식)
4. 압력/속도 갱신: p = p* + p', u = u* + u'
5. 수렴까지 반복

핵심 방정식:
∇²p' = ρ/Δt · ∇·u*

여기서 p' = p - p* (압력 보정)
```

```python
def simple_algorithm_concept():
    """SIMPLE 알고리즘 개념 설명"""

    print("=" * 60)
    print("SIMPLE 알고리즘 흐름도")
    print("=" * 60)

    flowchart = """
    ┌─────────────────────────────────────────┐
    │           초기 추측: p*, u*, v*           │
    └─────────────────────────────────────────┘
                        │
                        ▼
    ┌─────────────────────────────────────────┐
    │  Step 1: 운동량 방정식 풀이               │
    │  → 추측 속도 u*, v* 계산                  │
    │  (압력 p* 사용)                           │
    └─────────────────────────────────────────┘
                        │
                        ▼
    ┌─────────────────────────────────────────┐
    │  Step 2: 압력 보정 방정식 풀이            │
    │  ∇²p' = (ρ/Δt) ∇·u*                      │
    │  (u*의 발산 = 질량 불균형)                │
    └─────────────────────────────────────────┘
                        │
                        ▼
    ┌─────────────────────────────────────────┐
    │  Step 3: 압력/속도 보정                   │
    │  p = p* + αp·p'                          │
    │  u = u* - (Δt/ρ)·∂p'/∂x                  │
    │  v = v* - (Δt/ρ)·∂p'/∂y                  │
    └─────────────────────────────────────────┘
                        │
                        ▼
    ┌─────────────────────────────────────────┐
    │  수렴 확인: |∇·u| < ε ?                   │
    │                                          │
    │  No → p* = p, u* = u, v* = v (반복)      │
    │  Yes → 종료                               │
    └─────────────────────────────────────────┘

    Under-relaxation 계수:
    - αp ~ 0.3 (압력)
    - αu ~ 0.7 (속도)
    """
    print(flowchart)

simple_algorithm_concept()
```

### 5.2 SIMPLE 알고리즘 구현

```python
def simple_lid_driven_cavity():
    """
    SIMPLE 알고리즘으로 Lid-Driven Cavity 풀이
    엇갈린 격자 사용
    """

    # 격자 설정
    Nx = 32  # 셀 수 (x 방향)
    Ny = 32  # 셀 수 (y 방향)
    L = 1.0

    dx = L / Nx
    dy = L / Ny

    # 물성치
    rho = 1.0
    mu = 0.01
    Re = rho * 1.0 * L / mu
    print(f"Reynolds number: Re = {Re}")

    # Under-relaxation
    alpha_u = 0.7
    alpha_p = 0.3

    # 배열 초기화 (엇갈린 격자)
    # u: (Ny, Nx+1) - 수직 면
    # v: (Ny+1, Nx) - 수평 면
    # p: (Ny, Nx) - 셀 중심
    u = np.zeros((Ny, Nx + 1))
    v = np.zeros((Ny + 1, Nx))
    p = np.zeros((Ny, Nx))
    p_prime = np.zeros((Ny, Nx))

    # 경계조건
    U_lid = 1.0

    def apply_boundary_conditions():
        """경계조건 적용"""
        nonlocal u, v

        # 상단 (lid): u = U_lid
        u[-1, :] = 2 * U_lid - u[-2, :]  # 선형 외삽 (벽면에서 U_lid)

        # 하단: u = 0
        u[0, :] = -u[1, :]  # 선형 외삽

        # 좌/우: u = 0 (이미 0)
        u[:, 0] = 0
        u[:, -1] = 0

        # v 경계조건
        v[0, :] = 0    # 하단
        v[-1, :] = 0   # 상단
        v[:, 0] = -v[:, 1]   # 좌측
        v[:, -1] = -v[:, -2]  # 우측

    def solve_momentum(u, v, p, mu, rho, dx, dy, dt):
        """운동량 방정식 풀이 (추측 속도)"""
        u_star = u.copy()
        v_star = v.copy()

        # u-운동량
        for j in range(1, Ny - 1):
            for i in range(1, Nx):
                # 대류항 (풍상)
                u_face = 0.5 * (u[j, i] + u[j, i-1]) if i > 0 else u[j, i]

                if u_face > 0:
                    dudx = (u[j, i] - u[j, i-1]) / dx
                else:
                    dudx = (u[j, i+1] - u[j, i]) / dx if i < Nx else (u[j, i] - u[j, i-1]) / dx

                v_face = 0.25 * (v[j, min(i, Nx-1)] + v[j+1, min(i, Nx-1)] +
                                v[j, max(i-1, 0)] + v[j+1, max(i-1, 0)])

                if v_face > 0:
                    dudy = (u[j, i] - u[j-1, i]) / dy
                else:
                    dudy = (u[j+1, i] - u[j, i]) / dy if j < Ny-1 else (u[j, i] - u[j-1, i]) / dy

                # 확산항
                d2udx2 = (u[j, i+1] - 2*u[j, i] + u[j, i-1]) / dx**2 if 0 < i < Nx else 0
                d2udy2 = (u[j+1, i] - 2*u[j, i] + u[j-1, i]) / dy**2

                # 압력 구배
                dpdx = (p[j, min(i, Nx-1)] - p[j, max(i-1, 0)]) / dx

                # 시간 전진
                conv = u_face * dudx + v_face * dudy
                diff = mu / rho * (d2udx2 + d2udy2)
                u_star[j, i] = u[j, i] + dt * (-conv - dpdx / rho + diff)

        # v-운동량 (유사하게)
        for j in range(1, Ny):
            for i in range(1, Nx - 1):
                u_face = 0.25 * (u[min(j, Ny-1), i] + u[min(j, Ny-1), i+1] +
                                u[max(j-1, 0), i] + u[max(j-1, 0), i+1])

                if u_face > 0:
                    dvdx = (v[j, i] - v[j, i-1]) / dx
                else:
                    dvdx = (v[j, i+1] - v[j, i]) / dx if i < Nx-1 else (v[j, i] - v[j, i-1]) / dx

                v_face = 0.5 * (v[j, i] + v[j-1, i]) if j > 0 else v[j, i]

                if v_face > 0:
                    dvdy = (v[j, i] - v[j-1, i]) / dy
                else:
                    dvdy = (v[j+1, i] - v[j, i]) / dy if j < Ny else (v[j, i] - v[j-1, i]) / dy

                d2vdx2 = (v[j, i+1] - 2*v[j, i] + v[j, i-1]) / dx**2
                d2vdy2 = (v[j+1, i] - 2*v[j, i] + v[j-1, i]) / dy**2 if 0 < j < Ny else 0

                dpdy = (p[min(j, Ny-1), i] - p[max(j-1, 0), i]) / dy

                conv = u_face * dvdx + v_face * dvdy
                diff = mu / rho * (d2vdx2 + d2vdy2)
                v_star[j, i] = v[j, i] + dt * (-conv - dpdy / rho + diff)

        return u_star, v_star

    def solve_pressure_correction(u_star, v_star, dx, dy, dt, rho, n_iter=50):
        """압력 보정 방정식 풀이"""
        p_prime = np.zeros((Ny, Nx))

        for _ in range(n_iter):
            p_old = p_prime.copy()

            for j in range(Ny):
                for i in range(Nx):
                    # 질량 불균형 (발산)
                    div = ((u_star[j, i+1] - u_star[j, i]) / dx +
                          (v_star[j+1, i] - v_star[j, i]) / dy)

                    # 이웃 압력
                    p_E = p_prime[j, i+1] if i < Nx-1 else 0
                    p_W = p_prime[j, i-1] if i > 0 else 0
                    p_N = p_prime[j+1, i] if j < Ny-1 else 0
                    p_S = p_prime[j-1, i] if j > 0 else 0

                    # Poisson 방정식 계수
                    aE = 1/dx**2 if i < Nx-1 else 0
                    aW = 1/dx**2 if i > 0 else 0
                    aN = 1/dy**2 if j < Ny-1 else 0
                    aS = 1/dy**2 if j > 0 else 0
                    aP = aE + aW + aN + aS

                    if aP > 0:
                        p_prime[j, i] = (aE*p_E + aW*p_W + aN*p_N + aS*p_S -
                                        rho/dt * div) / aP

        return p_prime

    def correct_velocity(u_star, v_star, p_prime, dx, dy, dt, rho):
        """속도 보정"""
        u_new = u_star.copy()
        v_new = v_star.copy()

        # u 보정
        for j in range(Ny):
            for i in range(1, Nx):
                dpdx = (p_prime[j, i] - p_prime[j, i-1]) / dx
                u_new[j, i] = u_star[j, i] - dt / rho * dpdx

        # v 보정
        for j in range(1, Ny):
            for i in range(Nx):
                dpdy = (p_prime[j, i] - p_prime[j-1, i]) / dy
                v_new[j, i] = v_star[j, i] - dt / rho * dpdy

        return u_new, v_new

    # 시뮬레이션
    dt = 0.001
    n_outer = 500

    print("\nSIMPLE 알고리즘 시작...")

    for n in range(n_outer):
        apply_boundary_conditions()

        # 1. 운동량 방정식
        u_star, v_star = solve_momentum(u, v, p, mu, rho, dx, dy, dt)

        # 2. 압력 보정
        p_prime = solve_pressure_correction(u_star, v_star, dx, dy, dt, rho)

        # 3. 속도/압력 보정
        u_new, v_new = correct_velocity(u_star, v_star, p_prime, dx, dy, dt, rho)
        p_new = p + alpha_p * p_prime

        # Under-relaxation
        u = alpha_u * u_new + (1 - alpha_u) * u
        v = alpha_u * v_new + (1 - alpha_u) * v
        p = p_new

        # 수렴 체크
        if n % 100 == 0:
            div_max = 0
            for j in range(Ny):
                for i in range(Nx):
                    div = abs((u[j, i+1] - u[j, i]) / dx +
                             (v[j+1, i] - v[j, i]) / dy)
                    div_max = max(div_max, div)
            print(f"Iteration {n}: max|div(u)| = {div_max:.2e}")

    print("완료!")

    # 셀 중심 속도로 변환
    u_center = 0.5 * (u[:, :-1] + u[:, 1:])
    v_center = 0.5 * (v[:-1, :] + v[1:, :])

    # 결과 시각화
    x = np.linspace(dx/2, L - dx/2, Nx)
    y = np.linspace(dy/2, L - dy/2, Ny)
    X, Y = np.meshgrid(x, y)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 속도 벡터
    ax1 = axes[0]
    speed = np.sqrt(u_center**2 + v_center**2)
    ax1.streamplot(X, Y, u_center, v_center, color=speed, cmap='jet',
                  density=2, linewidth=1)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title(f'유선 (SIMPLE), Re = {Re:.0f}')
    ax1.set_aspect('equal')

    # 압력 분포
    ax2 = axes[1]
    im = ax2.pcolormesh(X, Y, p, cmap='coolwarm', shading='auto')
    plt.colorbar(im, ax=ax2, label='p')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_title('압력 분포')
    ax2.set_aspect('equal')

    # 중심선 속도
    ax3 = axes[2]
    j_center = Nx // 2
    ax3.plot(u_center[:, j_center], y, 'b-', linewidth=2)
    ax3.set_xlabel('u')
    ax3.set_ylabel('y')
    ax3.set_title('수직 중심선 u-속도')
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('simple_cavity.png', dpi=150, bbox_inches='tight')
    plt.show()

    return u, v, p

# u, v, p = simple_lid_driven_cavity()
```

---

## 6. SIMPLE 변형 알고리즘

### 6.1 SIMPLER, SIMPLEC, PISO

```
SIMPLE 변형:

1. SIMPLER (SIMPLE Revised):
   - 압력 직접 계산 (추측 불필요)
   - 수렴 빠름
   - 계산량 증가

2. SIMPLEC (SIMPLE Consistent):
   - 이웃 속도 보정항 고려
   - Under-relaxation 덜 필요
   - 수렴 개선

3. PISO (Pressure-Implicit with Splitting of Operators):
   - 비정상 문제에 적합
   - 추가 압력/속도 보정
   - 시간 정확도 향상

선택 가이드:
- 정상 문제: SIMPLE 또는 SIMPLEC
- 비정상 문제: PISO
- 빠른 수렴 필요: SIMPLER
```

```python
def algorithm_comparison():
    """알고리즘 비교 개념도"""

    print("=" * 60)
    print("SIMPLE 계열 알고리즘 비교")
    print("=" * 60)

    comparison = """
    ┌─────────────┬────────────┬────────────┬────────────┐
    │  알고리즘   │   SIMPLE   │  SIMPLEC   │    PISO    │
    ├─────────────┼────────────┼────────────┼────────────┤
    │ 압력 보정   │    1회     │    1회     │   2+회     │
    │ 반복/시간   │   많음     │   적음     │   적음     │
    │ αp         │  0.3~0.5   │  0.7~1.0   │    1.0     │
    │ 적용 분야   │   정상     │   정상     │   비정상   │
    │ 계산 비용   │   낮음     │   중간     │   높음     │
    └─────────────┴────────────┴────────────┴────────────┘

    각 알고리즘의 핵심 차이:

    SIMPLE:
    - 표준 방법, 간단하고 견고함
    - Under-relaxation 필수 (αp ~ 0.3)

    SIMPLEC:
    - 속도 보정 시 이웃 기여 고려
    - αp를 1에 가깝게 사용 가능

    PISO:
    - 비정상 문제를 위한 predictor-corrector
    - 각 시간 단계에서 2회 이상 보정
    - Courant 수 1 이하 필요
    """
    print(comparison)

algorithm_comparison()
```

---

## 7. 연습 문제

### 연습 1: 유동함수-와도
유동함수 ψ = xy에 해당하는 속도장과 와도를 구하고, 이것이 비압축성 조건을 만족하는지 확인하시오.

### 연습 2: Lid-Driven Cavity
Re = 400에서 Lid-Driven Cavity 시뮬레이션을 수행하고, Re = 100 결과와 비교하시오. 코너 와류의 발달을 관찰하시오.

### 연습 3: SIMPLE Under-relaxation
SIMPLE 알고리즘에서 under-relaxation 계수 αp를 0.1, 0.3, 0.5로 변화시키며 수렴 속도를 비교하시오.

### 연습 4: 격자 수렴
Lid-Driven Cavity 문제에서 격자 크기를 16x16, 32x32, 64x64로 변화시키며 수치해의 수렴을 분석하시오.

---

## 8. 참고자료

### 핵심 논문
- Ghia et al. (1982) "High-Re Solutions for Incompressible Flow Using the Navier-Stokes Equations and a Multigrid Method" - Lid-Driven Cavity 벤치마크
- Patankar & Spalding (1972) "A calculation procedure for heat, mass and momentum transfer in three-dimensional parabolic flows" - SIMPLE 알고리즘

### 교재
- Patankar, "Numerical Heat Transfer and Fluid Flow" (SIMPLE 상세)
- Ferziger & Peric, "Computational Methods for Fluid Dynamics"
- Moukalled et al., "The Finite Volume Method in Computational Fluid Dynamics"

### CFD 코드
- OpenFOAM: icoFoam (비압축성 층류)
- SIMPLE 구현 튜토리얼: CFD-Online, PyFR

---

## 요약

```
비압축성 유동 핵심:

1. 지배방정식:
   - 연속: ∇·u = 0
   - 운동량: Du/Dt = -∇p/ρ + ν∇²u

2. 정식화 방법:
   a) 유동함수-와도 (2D):
      - ∇²ψ = -ω
      - ∂ω/∂t + (u·∇)ω = ν∇²ω
   b) 원시변수 + SIMPLE:
      - 압력-속도 결합 처리

3. 엇갈린 격자:
   - 체커보드 방지
   - u, v는 셀 면, p는 셀 중심

4. SIMPLE 알고리즘:
   ① 추측 압력으로 운동량 방정식
   ② 압력 보정 Poisson 방정식
   ③ 속도/압력 갱신
   ④ 수렴까지 반복

5. 수치적 고려사항:
   - Under-relaxation: αp ~ 0.3, αu ~ 0.7
   - 격자 수렴 확인 필수
   - CFL 조건 준수
```

---

다음 레슨에서는 전자기학의 수치해석과 Maxwell 방정식의 이산화를 다룹니다.
