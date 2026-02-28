# 수치 시뮬레이션 Overview

## 개요

이 폴더는 파이썬을 이용한 수치 시뮬레이션 학습 자료를 담고 있습니다. 상미분방정식(ODE)의 기초부터 자기유체역학(MHD)과 플라즈마 시뮬레이션까지 전 범위를 다룹니다.

---

## 학습 로드맵

```
기초 (01-02)
    ↓
상미분방정식 ODE (03-06)
    ↓
편미분방정식 PDE 기초 (07-08)
    ↓
열/파동/정상상태 방정식 (09-12)
    ↓
전산유체역학 CFD (13-14)
    ↓
전자기 시뮬레이션 (15-16)
    ↓
자기유체역학 MHD (17-18)
    ↓
플라즈마 시뮬레이션 (19)
    ↓
몬테카를로 시뮬레이션 (20)
    ↓
스펙트럼 방법 (21)
    ↓
유한 요소 방법 (22)
    ↓
GPU 가속 (23)
    ↓
물리 정보 신경망 PINN (24)
```

---

## 파일 목록

| 파일 | 주제 | 핵심 내용 |
|------|------|----------|
| [01_Numerical_Analysis_Basics.md](./01_Numerical_Analysis_Basics.md) | 수치해석 기초 | 부동소수점, 오차 분석, 수치 미분/적분 |
| [02_Linear_Algebra_Review.md](./02_Linear_Algebra_Review.md) | 선형대수 복습 | 행렬 연산, 고유값, 분해(LU, QR, SVD) |
| [03_ODE_Basics.md](./03_ODE_Basics.md) | 상미분방정식 기초 | ODE 개념, 초기값 문제, 해석적 해 |
| [04_ODE_Numerical_Methods.md](./04_ODE_Numerical_Methods.md) | ODE 수치해법 | Euler, RK2, RK4, 적응형 스텝 |
| [05_ODE_Advanced.md](./05_ODE_Advanced.md) | ODE 고급 | 강성(stiff) 문제, 암시적 방법, scipy.integrate |
| [06_ODE_Systems.md](./06_ODE_Systems.md) | 연립 ODE와 시스템 | Lotka-Volterra, 진자, 혼돈계 (Lorenz) |
| [07_PDE_Overview.md](./07_PDE_Overview.md) | 편미분방정식 개요 | PDE 분류, 경계조건, 초기조건 |
| [08_Finite_Difference_Basics.md](./08_Finite_Difference_Basics.md) | 유한차분법 기초 | 격자, 이산화, 안정성 조건 (CFL) |
| [09_Heat_Equation.md](./09_Heat_Equation.md) | 열방정식 | 1D/2D 열전도, 명시적/암시적 방법 |
| [10_Wave_Equation.md](./10_Wave_Equation.md) | 파동방정식 | 1D/2D 파동, 경계 반사, 흡수 경계 |
| [11_Laplace_Poisson.md](./11_Laplace_Poisson.md) | 라플라스/포아송 | 정상상태, 반복법 (Jacobi, Gauss-Seidel, SOR) |
| [12_Advection_Equation.md](./12_Advection_Equation.md) | 이류방정식 | Upwind, Lax-Wendroff, 수치 분산/확산 |
| [13_CFD_Basics.md](./13_CFD_Basics.md) | CFD 기초 | 유체역학 개념, Navier-Stokes 소개 |
| [14_Incompressible_Flow.md](./14_Incompressible_Flow.md) | 비압축성 유동 | 유선함수-와도, 압력-속도 결합, SIMPLE |
| [15_Electromagnetics_Numerical.md](./15_Electromagnetics_Numerical.md) | 전자기학 수치해석 | Maxwell 방정식, FDTD 기초 |
| [16_FDTD_Implementation.md](./16_FDTD_Implementation.md) | FDTD 구현 | 1D/2D 전자기파 시뮬레이션, 흡수경계(PML) |
| [17_MHD_Basics.md](./17_MHD_Basics.md) | MHD 기초 이론 | 자기유체역학 개념, 이상 MHD 방정식 |
| [18_MHD_Numerical_Methods.md](./18_MHD_Numerical_Methods.md) | MHD 수치해법 | 보존형, Godunov 방법, MHD 리만 문제 |
| [19_Plasma_Simulation.md](./19_Plasma_Simulation.md) | 플라즈마 시뮬레이션 | PIC 방법 기초, 입자-격자 상호작용 |
| [20_Monte_Carlo_Simulation.md](./20_Monte_Carlo_Simulation.md) | 몬테카를로 시뮬레이션 | 난수 생성, MC 적분, Ising 모델, 옵션 가격, 분산 감소 |
| [21_Spectral_Methods.md](./21_Spectral_Methods.md) | 스펙트럼 방법 | 푸리에 스펙트럼, FFT 미분, 체비셰프 콜로케이션, 디에일리어싱 |
| [22_Finite_Element_Method.md](./22_Finite_Element_Method.md) | 유한 요소 방법 | 약형식, 기저 함수, 강성 행렬 조립, 1D/2D FEM |
| [23_GPU_Acceleration.md](./23_GPU_Acceleration.md) | GPU 가속(GPU Acceleration) | CUDA 기초, CuPy, GPU 가속 PDE 솔버, 성능 최적화 |
| [24_Physics_Informed_Neural_Networks.md](./24_Physics_Informed_Neural_Networks.md) | 물리 정보 신경망(PINN) | 신경망 솔버, PDE 기반 손실 함수, 자동 미분, 역문제 |

---

## 필요 라이브러리

```bash
# 기본
pip install numpy scipy matplotlib

# 성능 최적화 (선택)
pip install numba

# 3D 시각화 (선택)
pip install mayavi
```

### 라이브러리 역할

| 라이브러리 | 용도 |
|-----------|------|
| NumPy | 배열 연산, 선형대수 |
| SciPy | ODE 솔버, 희소행렬, 최적화 |
| Matplotlib | 2D 시각화, 애니메이션 |
| Numba | JIT 컴파일, 성능 최적화 |

---

## 권장 학습 순서

### 1단계: 기초 (1-2주)
- 01_Numerical_Analysis_Basics.md
- 02_Linear_Algebra_Review.md

### 2단계: ODE (2-3주)
- 03_ODE_Basics.md
- 04_ODE_Numerical_Methods.md
- 05_ODE_Advanced.md
- 06_ODE_Systems.md

### 3단계: PDE 기초 (2-3주)
- 07_PDE_Overview.md
- 08_Finite_Difference_Basics.md
- 09_Heat_Equation.md
- 10_Wave_Equation.md

### 4단계: 정상상태와 이류 (1-2주)
- 11_Laplace_Poisson.md
- 12_Advection_Equation.md

### 5단계: CFD (2-3주)
- 13_CFD_Basics.md
- 14_Incompressible_Flow.md

### 6단계: 전자기 (2주)
- 15_Electromagnetics_Numerical.md
- 16_FDTD_Implementation.md

### 7단계: MHD와 플라즈마 (3-4주)
- 17_MHD_Basics.md
- 18_MHD_Numerical_Methods.md
- 19_Plasma_Simulation.md

### 8단계: 확률적 시뮬레이션 (2주)
- 20_Monte_Carlo_Simulation.md

### 9단계: 고급 방법 (2-3주)
- 21_Spectral_Methods.md
- 22_Finite_Element_Method.md

### 10단계: 현대 계산 기법 (2-3주)
- 23_GPU_Acceleration.md
- 24_Physics_Informed_Neural_Networks.md

---

## 선수 지식

1. **Python 기초**: NumPy 배열 연산
2. **미적분학**: 미분, 적분, 편미분
3. **선형대수**: 행렬, 고유값, 분해
4. **물리학**: 역학, 전자기학 기초 (CFD/MHD의 경우)

---

## 시뮬레이션 코드 구조 예시

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# 1. 파라미터 설정
nx, ny = 100, 100
dx, dy = 1.0, 1.0
dt = 0.01
n_steps = 1000

# 2. 초기조건
u = np.zeros((nx, ny))

# 3. 시간 적분 루프
for step in range(n_steps):
    # 경계조건 적용
    # 공간 미분 계산
    # 시간 전진
    pass

# 4. 결과 시각화
plt.imshow(u)
plt.colorbar()
plt.show()
```

---

## 참고 자료

### 교재
- Computational Physics - Mark Newman
- Numerical Recipes - Press et al.
- CFD Python (12 Steps to Navier-Stokes) - Lorena Barba

### 온라인
- SciPy 공식 문서: https://docs.scipy.org
- Lorena Barba CFD Python: https://github.com/barbagroup/CFDPython
