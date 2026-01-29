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
```

---

## 파일 목록

| 파일 | 주제 | 핵심 내용 |
|------|------|----------|
| 01 | 수치해석 기초 | 부동소수점, 오차 분석, 수치 미분/적분 |
| 02 | 선형대수 복습 | 행렬 연산, 고유값, 분해(LU, QR, SVD) |
| 03 | 상미분방정식 기초 | ODE 개념, 초기값 문제, 해석적 해 |
| 04 | ODE 수치해법 | Euler, RK2, RK4, 적응형 스텝 |
| 05 | ODE 고급 | 강성(stiff) 문제, 암시적 방법, scipy.integrate |
| 06 | 연립 ODE와 시스템 | Lotka-Volterra, 진자, 혼돈계 (Lorenz) |
| 07 | 편미분방정식 개요 | PDE 분류, 경계조건, 초기조건 |
| 08 | 유한차분법 기초 | 격자, 이산화, 안정성 조건 (CFL) |
| 09 | 열방정식 | 1D/2D 열전도, 명시적/암시적 방법 |
| 10 | 파동방정식 | 1D/2D 파동, 경계 반사, 흡수 경계 |
| 11 | 라플라스/포아송 | 정상상태, 반복법 (Jacobi, Gauss-Seidel, SOR) |
| 12 | 이류방정식 | Upwind, Lax-Wendroff, 수치 분산/확산 |
| 13 | CFD 기초 | 유체역학 개념, Navier-Stokes 소개 |
| 14 | 비압축성 유동 | 유선함수-와도, 압력-속도 결합, SIMPLE |
| 15 | 전자기학 수치해석 | Maxwell 방정식, FDTD 기초 |
| 16 | FDTD 구현 | 1D/2D 전자기파 시뮬레이션, 흡수경계(PML) |
| 17 | MHD 기초 이론 | 자기유체역학 개념, 이상 MHD 방정식 |
| 18 | MHD 수치해법 | 보존형, Godunov 방법, MHD 리만 문제 |
| 19 | 플라즈마 시뮬레이션 | PIC 방법 기초, 입자-격자 상호작용 |
| 20 | 몬테카를로 시뮬레이션 | 난수 생성, MC 적분, Ising 모델, 옵션 가격, 분산 감소 |

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
- 01_수치해석_기초.md
- 02_선형대수_복습.md

### 2단계: ODE (2-3주)
- 03_상미분방정식_기초.md
- 04_ODE_수치해법.md
- 05_ODE_고급.md
- 06_연립_ODE와_시스템.md

### 3단계: PDE 기초 (2-3주)
- 07_편미분방정식_개요.md
- 08_유한차분법_기초.md
- 09_열방정식.md
- 10_파동방정식.md

### 4단계: 정상상태와 이류 (1-2주)
- 11_라플라스_포아송.md
- 12_이류방정식.md

### 5단계: CFD (2-3주)
- 13_CFD_기초.md
- 14_비압축성_유동.md

### 6단계: 전자기 (2주)
- 15_전자기학_수치해석.md
- 16_FDTD_구현.md

### 7단계: MHD와 플라즈마 (3-4주)
- 17_MHD_기초_이론.md
- 18_MHD_수치해법.md
- 19_플라즈마_시뮬레이션.md

### 8단계: 확률적 시뮬레이션 (2주)
- 20_몬테카를로_시뮬레이션.md

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
