# 스펙트럼 방법(Spectral Methods)

## 학습 목표
- 스펙트럼 방법의 수학적 기초 이해하기
- 푸리에 스펙트럼 방법과 FFT 기반 미분 마스터하기
- 체비셰프 콜로케이션(Chebyshev collocation)과 의사-스펙트럼(pseudospectral) 기법 학습하기
- 에일리어싱 오류를 방지하기 위한 디에일리어싱(dealiasing) 전략(3/2 규칙) 적용하기
- PDE를 위한 스펙트럼 솔버 구현하기(버거스 방정식, KdV 방정식)

## 목차
1. [스펙트럼 방법 소개](#1-스펙트럼-방법-소개)
2. [푸리에 스펙트럼 방법](#2-푸리에-스펙트럼-방법)
3. [이산 푸리에 변환과 FFT](#3-이산-푸리에-변환과-fft)
4. [스펙트럼 미분](#4-스펙트럼-미분)
5. [체비셰프 다항식](#5-체비셰프-다항식)
6. [의사-스펙트럼 방법](#6-의사-스펙트럼-방법)
7. [디에일리어싱](#7-디에일리어싱)
8. [응용: 스펙트럼 방법으로 PDE 풀기](#8-응용-스펙트럼-방법으로-pde-풀기)
9. [연습 문제](#9-연습-문제)

---

## 1. 스펙트럼 방법 소개

### 1.1 스펙트럼 방법이란?

**왜 스펙트럼 방법인가?** 유한 차분 방법은 $O(h^p)$로 수렴합니다 -- 해상도를 두 배로 늘려도 오차는 고정된 비율로만 줄어듭니다. 스펙트럼 방법은 매끄러운 문제에 대해 $O(e^{-cN})$로 수렴합니다 -- 해상도를 두 배로 늘리면 오차가 수 자릿수만큼 줄어들 수 있습니다. 이는 해가 매끄러울 때 스펙트럼 방법을 훨씬 더 효율적으로 만들며, 난류 시뮬레이션(DNS, Direct Numerical Simulation), 기상 예측, 양자역학에서 선택되는 방법인 이유입니다.

스펙트럼 방법은 전역 기저 함수(예: 푸리에 급수, 체비셰프 다항식)를 사용하여 미분 방정식의 해를 근사합니다. 국소 근사를 사용하는 유한 차분이나 유한 요소 방법과 달리, 스펙트럼 방법은 매끄러운 문제에 대해 **지수 수렴(exponential convergence)**을 달성합니다.

```
┌─────────────────────────────────────────────────────────────┐
│             비교: 유한 차분 vs 스펙트럼                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  유한 차분(Finite Difference):  스펙트럼 방법:              │
│  ┌─┬─┬─┬─┬─┬─┬─┐                ┌──────────────┐           │
│  │ │ │ │ │ │ │ │                │ ∑ aₖ φₖ(x)   │           │
│  └─┴─┴─┴─┴─┴─┴─┘                └──────────────┘           │
│  국소 스텐실                      전역 기저 함수              │
│  O(h^p) 수렴                      O(e^(-cn)) 수렴          │
│  (p = 차수)                       (지수적!)                 │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**장점:**
- 매끄러운 해에 대한 지수적 정확도
- 미분의 자연스러운 처리 (스펙트럼 공간에서 곱셈)
- FFT를 사용한 효율적 계산 (O(N log N) 연산)

**제한사항:**
- 매끄럽고 주기적(푸리에) 또는 잘 작동하는(체비셰프) 함수 필요
- 복잡한 형상 처리가 어려움
- 비주기 경계는 특별한 처리 필요

### 1.2 스펙트럼 방법의 유형

```python
"""
스펙트럼 방법의 세 가지 주요 유형:

1. 갤러킨 방법(Galerkin Method):
   - 가중 잔차 접근법
   - PDE를 기저 함수에 투영
   - 예: u(x) = Σ aₙ φₙ(x), minimize ∫R(u)φₘ dx

2. 콜로케이션 방법(Collocation Method) (의사-스펙트럼):
   - 콜로케이션 점에서 PDE를 정확히 만족
   - 갤러킨보다 빠르지만 약간 덜 정확
   - 예: R(u(xⱼ)) = 0 at grid points xⱼ

3. 타우 방법(Tau Method):
   - 갤러킨과 유사하지만 경계에서 잔차 허용
   - 비주기 문제에 유용
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, fftfreq

np.random.seed(42)
```

---

## 2. 푸리에 스펙트럼 방법

### 2.1 푸리에 급수 표현

[0, 2π]에서 주기 함수의 경우 다음과 같이 전개합니다:

```
u(x) = Σ ûₖ e^(ikx)
       k=-∞ to ∞

where ûₖ = (1/2π) ∫₀^(2π) u(x) e^(-ikx) dx
```

실제로는 N개 모드로 잘라냅니다:

```python
def fourier_series_example():
    """
    매끄러운 주기 함수의 푸리에 급수 근사 시연.
    """
    N = 64
    x = np.linspace(0, 2*np.pi, N, endpoint=False)

    # 원본 함수: u(x) = sin(x) + 0.5*sin(2x) + 0.1*cos(5x)
    u = np.sin(x) + 0.5*np.sin(2*x) + 0.1*np.cos(5*x)

    # FFT를 사용하여 푸리에 계수 계산
    u_hat = fft(u)

    # 스펙트럼 계수로부터 함수 재구성
    u_reconstructed = np.real(ifft(u_hat))

    # 플롯
    plt.figure(figsize=(10, 4))
    plt.plot(x, u, 'b-', label='Original', linewidth=2)
    plt.plot(x, u_reconstructed, 'r--', label='Reconstructed', linewidth=2)
    plt.xlabel('x')
    plt.ylabel('u(x)')
    plt.title('Fourier Series Representation')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('fourier_series.png', dpi=150)
    plt.close()

    # 스펙트럼 계수 출력 (크기)
    k = fftfreq(N, 1/N)
    plt.figure(figsize=(10, 4))
    plt.stem(k, np.abs(u_hat), basefmt=' ')
    plt.xlabel('Wavenumber k')
    plt.ylabel('|û_k|')
    plt.title('Fourier Spectrum')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('fourier_spectrum.png', dpi=150)
    plt.close()

    print(f"L2 error: {np.linalg.norm(u - u_reconstructed):.2e}")

fourier_series_example()
```

**출력:**
```
L2 error: 1.34e-14
```

재구성은 본질적으로 정확합니다 (기계 정밀도까지).

---

## 3. 이산 푸리에 변환과 FFT

### 3.1 DFT 정의

N개의 이산 점에 대해 DFT는 다음과 같습니다:

```
ûₖ = Σ_{j=0}^{N-1} uⱼ e^(-2πijk/N),  k = 0, 1, ..., N-1

역변환: uⱼ = (1/N) Σ_{k=0}^{N-1} ûₖ e^(2πijk/N)
```

### 3.2 FFT 알고리즘

고속 푸리에 변환(Fast Fourier Transform)은 복잡도를 O(N²)에서 O(N log N)으로 줄입니다:

```
┌────────────────────────────────────────────────────────────┐
│              쿨리-튜키 FFT 알고리즘                          │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  DFT[u₀, u₁, ..., u_{N-1}]                                │
│       = DFT[u₀, u₂, u₄, ...] (짝수 인덱스)                 │
│         + W * DFT[u₁, u₃, u₅, ...] (홀수 인덱스)           │
│                                                            │
│  where W = e^(-2πi/N) (트위들 인수)                        │
│                                                            │
│  기본 경우(N=1)까지 재귀적으로 분할                         │
│  복잡도: O(N log N)                                         │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

```python
from scipy.fft import fft, ifft, fftfreq
import time

def fft_performance_test():
    """
    나이브 DFT vs FFT 성능 비교.
    """
    def naive_dft(u):
        N = len(u)
        u_hat = np.zeros(N, dtype=complex)
        for k in range(N):
            for j in range(N):
                u_hat[k] += u[j] * np.exp(-2j * np.pi * k * j / N)
        return u_hat

    sizes = [32, 64, 128, 256, 512]
    naive_times = []
    fft_times = []

    for N in sizes:
        u = np.random.randn(N)

        # 나이브 DFT
        start = time.time()
        u_hat_naive = naive_dft(u)
        naive_times.append(time.time() - start)

        # FFT
        start = time.time()
        u_hat_fft = fft(u)
        fft_times.append(time.time() - start)

        # 등가성 검증
        error = np.linalg.norm(u_hat_naive - u_hat_fft)
        print(f"N={N:4d}: Naive={naive_times[-1]:.4f}s, FFT={fft_times[-1]:.6f}s, Error={error:.2e}")

    # 플롯
    plt.figure(figsize=(8, 5))
    plt.loglog(sizes, naive_times, 'o-', label='Naive DFT O(N²)', linewidth=2)
    plt.loglog(sizes, fft_times, 's-', label='FFT O(N log N)', linewidth=2)
    plt.xlabel('N')
    plt.ylabel('Time (s)')
    plt.title('DFT vs FFT Performance')
    plt.legend()
    plt.grid(True, which='both', alpha=0.3)
    plt.tight_layout()
    plt.savefig('fft_performance.png', dpi=150)
    plt.close()

fft_performance_test()
```

---

## 4. 스펙트럼 미분

### 4.1 스펙트럼 공간에서의 미분

스펙트럼 방법의 핵심 장점: **미분이 스펙트럼 공간에서 곱셈이 됨**.

푸리에 기저의 경우:
```
u(x) = Σ ûₖ e^(ikx)
du/dx = Σ (ik) ûₖ e^(ikx)
```

코드로:
```python
def spectral_derivative(u, L=2*np.pi):
    """
    스펙트럼 미분을 사용하여 도함수 계산.

    Parameters:
    -----------
    u : array, 균일 격자에서의 함수 값
    L : float, 영역 길이 (기본값 2π)

    Returns:
    --------
    du_dx : array, 도함수
    """
    N = len(u)
    u_hat = fft(u)
    k = fftfreq(N, L/N) * 2 * np.pi  # 파수

    # 스펙트럼 공간에서의 도함수: ik로 곱하기
    du_hat = 1j * k * u_hat

    # 물리 공간으로 역변환
    du_dx = np.real(ifft(du_hat))

    return du_dx

# 예제: u(x) = sin(x) + 0.5*sin(2x) 미분
N = 128
x = np.linspace(0, 2*np.pi, N, endpoint=False)
u = np.sin(x) + 0.5*np.sin(2*x)
du_dx_exact = np.cos(x) + np.cos(2*x)
du_dx_spectral = spectral_derivative(u)

error = np.linalg.norm(du_dx_exact - du_dx_spectral, np.inf)
print(f"Max error in derivative: {error:.2e}")
```

**출력:**
```
Max error in derivative: 1.78e-14
```

### 4.2 고차 도함수

```python
def spectral_derivative_n(u, n, L=2*np.pi):
    """
    스펙트럼 미분을 사용하여 n차 도함수 계산.
    """
    N = len(u)
    u_hat = fft(u)
    k = fftfreq(N, L/N) * 2 * np.pi

    # n차 도함수: (ik)^n으로 곱하기
    du_hat = (1j * k)**n * u_hat

    du_dx = np.real(ifft(du_hat))
    return du_dx

# 2차 도함수 테스트
u = np.sin(x)
d2u_dx2_exact = -np.sin(x)
d2u_dx2_spectral = spectral_derivative_n(u, n=2)

error = np.linalg.norm(d2u_dx2_exact - d2u_dx2_spectral, np.inf)
print(f"Max error in 2nd derivative: {error:.2e}")
```

---

## 5. 체비셰프 다항식

### 5.1 제1종 체비셰프 다항식

[-1, 1]에서 비주기 문제의 경우 체비셰프 다항식이 최적입니다:

```
T₀(x) = 1
T₁(x) = x
T₂(x) = 2x² - 1
T₃(x) = 4x³ - 3x
...
Tₙ(x) = cos(n arccos(x))
```

**재귀 관계:**
```
T_{n+1}(x) = 2x Tₙ(x) - T_{n-1}(x)
```

```python
def chebyshev_polynomials(n, x):
    """
    점 x에서 체비셰프 다항식 T₀, T₁, ..., Tₙ 평가.

    Returns:
    --------
    T : 형상 (n+1, len(x))의 배열
    """
    T = np.zeros((n+1, len(x)))
    T[0] = 1
    if n >= 1:
        T[1] = x
    for k in range(2, n+1):
        T[k] = 2*x*T[k-1] - T[k-2]
    return T

# 처음 5개 체비셰프 다항식 플롯
x = np.linspace(-1, 1, 200)
T = chebyshev_polynomials(4, x)

plt.figure(figsize=(10, 5))
for k in range(5):
    plt.plot(x, T[k], label=f'T_{k}(x)', linewidth=2)
plt.xlabel('x')
plt.ylabel('Tₖ(x)')
plt.title('Chebyshev Polynomials')
plt.legend()
plt.grid(True)
plt.axhline(0, color='k', linewidth=0.5)
plt.axvline(0, color='k', linewidth=0.5)
plt.tight_layout()
plt.savefig('chebyshev_polynomials.png', dpi=150)
plt.close()
```

### 5.2 체비셰프-가우스-로바토 점

최적 콜로케이션 점 (끝점 포함):

```
xⱼ = cos(πj/N),  j = 0, 1, ..., N
```

```python
def chebyshev_points(N):
    """
    체비셰프-가우스-로바토 점 계산.
    """
    j = np.arange(N+1)
    x = np.cos(np.pi * j / N)
    return x

N = 16
x_cheb = chebyshev_points(N)

plt.figure(figsize=(10, 3))
plt.plot(x_cheb, np.zeros_like(x_cheb), 'ro', markersize=8)
plt.xlim(-1.1, 1.1)
plt.ylim(-0.5, 0.5)
plt.xlabel('x')
plt.title(f'Chebyshev-Gauss-Lobatto Points (N={N})')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('chebyshev_points.png', dpi=150)
plt.close()

print("Chebyshev points (clustered near boundaries):")
print(x_cheb)
```

### 5.3 체비셰프 미분 행렬

```python
def chebyshev_diff_matrix(N):
    """
    체비셰프 미분 행렬 D 계산.

    u'(xⱼ) ≈ Σ Dⱼₖ u(xₖ)
    """
    x = chebyshev_points(N)
    D = np.zeros((N+1, N+1))

    c = np.ones(N+1)
    c[0] = 2
    c[N] = 2

    for i in range(N+1):
        for j in range(N+1):
            if i != j:
                D[i, j] = (c[i] / c[j]) * ((-1)**(i+j)) / (x[i] - x[j])

    # 대각 원소
    for i in range(N+1):
        D[i, i] = -np.sum(D[i, :])

    return D, x

# 테스트: u(x) = exp(x) 미분
N = 16
D, x = chebyshev_diff_matrix(N)
u = np.exp(x)
du_dx_exact = np.exp(x)
du_dx_cheb = D @ u

error = np.linalg.norm(du_dx_exact - du_dx_cheb, np.inf)
print(f"Chebyshev differentiation error: {error:.2e}")
```

---

## 6. 의사-스펙트럼 방법

### 6.1 갤러킨 vs 의사-스펙트럼

```
┌────────────────────────────────────────────────────────────┐
│           갤러킨 vs 의사-스펙트럼(콜로케이션)                │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  갤러킨(Galerkin):                                         │
│    ∫ R(u) φₘ(x) dx = 0  모든 기저 함수 φₘ에 대해          │
│    → 적분 필요 (내적)                                      │
│    → 높은 정확도, 더 비싼 계산                              │
│                                                            │
│  의사-스펙트럼(Pseudospectral, Collocation):               │
│    R(u(xⱼ)) = 0  콜로케이션 점 xⱼ에서                     │
│    → 적분 없음, 격자 점에서 평가                           │
│    → 약간 낮은 정확도, 훨씬 빠름                           │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

```python
def pseudospectral_example():
    """
    의사-스펙트럼 방법을 사용하여 u''(x) = -π² sin(πx), u(0)=u(1)=0 풀기.

    정확한 해: u(x) = sin(πx)
    """
    N = 16

    # 체비셰프 점을 [-1,1]에서 [0,1]로 매핑
    x_cheb = (chebyshev_points(N) + 1) / 2

    # 미분 행렬 구축 ([0,1] 영역으로 스케일 필요)
    D, _ = chebyshev_diff_matrix(N)
    D = D * 2  # [0,1] 영역으로 스케일
    D2 = D @ D  # 2차 도함수

    # 경계 조건: u(0) = u(N) = 0
    # 내부 점: 인덱스 1부터 N-1까지
    D2_interior = D2[1:N, 1:N]

    # 우변: f(x) = -π² sin(πx)
    f = -np.pi**2 * np.sin(np.pi * x_cheb[1:N])

    # 선형 시스템 풀기
    u_interior = np.linalg.solve(D2_interior, f)

    # 경계값 추가
    u = np.zeros(N+1)
    u[1:N] = u_interior

    # 정확한 해와 비교
    u_exact = np.sin(np.pi * x_cheb)

    plt.figure(figsize=(10, 5))
    plt.plot(x_cheb, u_exact, 'b-', label='Exact', linewidth=2)
    plt.plot(x_cheb, u, 'ro', label='Pseudospectral', markersize=8)
    plt.xlabel('x')
    plt.ylabel('u(x)')
    plt.title("Pseudospectral Solution: u'' = -π² sin(πx)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('pseudospectral_example.png', dpi=150)
    plt.close()

    error = np.linalg.norm(u_exact - u, np.inf)
    print(f"Pseudospectral error: {error:.2e}")

pseudospectral_example()
```

---

## 7. 디에일리어싱

### 7.1 에일리어싱 문제

PDE의 비선형 항 (예: 버거스 방정식의 u ∂u/∂x)은 고주파 모드를 생성하여 저주파로 **에일리어스(alias)**될 수 있습니다:

```
┌────────────────────────────────────────────────────────────┐
│                    에일리어싱 예제                           │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  u가 최대 파수 k_max = N/2를 가지면,                       │
│  u²는 2*k_max = N까지 모드를 가짐                          │
│                                                            │
│  하지만 DFT는 N/2까지만 해상!                              │
│  → 고주파가 감싸져서 에일리어스됨                           │
│                                                            │
│  해결책: 3/2 규칙 (zero-padding)                           │
│    1. u를 3N/2 모드로 패딩                                 │
│    2. 확장 공간에서 비선형성 계산                          │
│    3. N 모드로 다시 잘라냄                                 │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

### 7.2 3/2 규칙 구현

```python
def dealias_product_3_2_rule(u, v):
    """
    3/2 규칙을 사용하여 디에일리어스된 곱 w = u*v 계산.

    Parameters:
    -----------
    u, v : 길이 N의 배열

    Returns:
    --------
    w : 디에일리어스된 곱 (길이 N)
    """
    N = len(u)
    M = 3 * N // 2  # 확장 격자

    # 푸리에 변환
    u_hat = fft(u)
    v_hat = fft(v)

    # 3N/2로 제로 패딩
    u_hat_padded = np.zeros(M, dtype=complex)
    v_hat_padded = np.zeros(M, dtype=complex)

    # 저주파
    u_hat_padded[:N//2] = u_hat[:N//2]
    u_hat_padded[-N//2:] = u_hat[-N//2:]

    v_hat_padded[:N//2] = v_hat[:N//2]
    v_hat_padded[-N//2:] = v_hat[-N//2:]

    # 확장 격자로 역변환
    u_extended = ifft(u_hat_padded)
    v_extended = ifft(v_hat_padded)

    # 물리 공간에서 곱셈
    w_extended = u_extended * v_extended

    # 다시 변환
    w_hat_extended = fft(w_extended)

    # 원래 해상도로 잘라내기
    w_hat = np.zeros(N, dtype=complex)
    w_hat[:N//2] = w_hat_extended[:N//2]
    w_hat[-N//2:] = w_hat_extended[-N//2:]

    # 스케일 (확장 격자 고려)
    w_hat *= M / N

    w = ifft(w_hat)
    return np.real(w)

# 테스트: 에일리어스된 곱 vs 디에일리어스된 곱 비교
N = 64
x = np.linspace(0, 2*np.pi, N, endpoint=False)
u = np.sin(3*x)
v = np.cos(4*x)

# 에일리어스된 곱 (직접 곱셈)
w_aliased = u * v

# 디에일리어스된 곱 (3/2 규칙)
w_dealiased = dealias_product_3_2_rule(u, v)

# 정확한 곱
w_exact = u * v  # 이 예제의 경우, 정확한 해 = 0.5*(sin(7x) - sin(x))

print(f"Aliased error:   {np.linalg.norm(w_exact - w_aliased):.2e}")
print(f"Dealiased error: {np.linalg.norm(w_exact - w_dealiased):.2e}")
```

---

## 8. 응용: 스펙트럼 방법으로 PDE 풀기

### 8.1 버거스 방정식

점성 버거스 방정식은 기본적인 비선형 PDE입니다:

```
∂u/∂t + u ∂u/∂x = ν ∂²u/∂x²

초기 조건: u(x,0) = sin(x)
주기 경계: u(0,t) = u(2π,t)
```

```python
def burgers_spectral(nu=0.01, T=2.0, N=128, dt=0.001):
    """
    디에일리어싱을 사용한 푸리에 스펙트럼 방법으로 버거스 방정식 풀기.

    ∂u/∂t + u ∂u/∂x = ν ∂²u/∂x²
    """
    # 격자
    L = 2 * np.pi
    x = np.linspace(0, L, N, endpoint=False)
    k = fftfreq(N, L/N) * 2 * np.pi

    # 초기 조건
    u = np.sin(x)

    # 시간 적분 (RK4)
    nt = int(T / dt)
    time = 0.0

    # 선택한 시간에 해 저장
    t_save = [0, 0.5, 1.0, 1.5, 2.0]
    u_save = []

    def rhs(u):
        """스펙트럼 공간에서 버거스 방정식의 우변."""
        u_hat = fft(u)

        # 선형 항: ν ∂²u/∂x²
        linear = -nu * k**2 * u_hat

        # 비선형 항: -u ∂u/∂x (디에일리어싱으로 물리 공간에서 계산)
        ux = np.real(ifft(1j * k * u_hat))
        nonlinear_phys = -u * ux
        nonlinear_phys = dealias_product_3_2_rule(u, ux)
        nonlinear = fft(nonlinear_phys)

        du_dt_hat = linear + nonlinear
        return np.real(ifft(du_dt_hat))

    for n in range(nt):
        # RK4 시간 진전
        k1 = rhs(u)
        k2 = rhs(u + 0.5*dt*k1)
        k3 = rhs(u + 0.5*dt*k2)
        k4 = rhs(u + dt*k3)

        u = u + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
        time += dt

        # 해 저장
        if any(abs(time - t) < dt/2 for t in t_save):
            u_save.append(u.copy())

    # 플롯
    plt.figure(figsize=(10, 6))
    for i, t in enumerate(t_save[:len(u_save)]):
        plt.plot(x, u_save[i], label=f't = {t:.1f}', linewidth=2)
    plt.xlabel('x')
    plt.ylabel('u(x,t)')
    plt.title(f'Burgers Equation (ν={nu})')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('burgers_spectral.png', dpi=150)
    plt.close()

    return x, u_save

x, u_save = burgers_spectral(nu=0.01, T=2.0)
print("Burgers equation solved successfully.")
```

### 8.2 코르테베흐-드 브리스(KdV) 방정식

KdV 방정식은 솔리톤 파를 모델링합니다:

```
∂u/∂t + u ∂u/∂x + ∂³u/∂x³ = 0

초기 조건: u(x,0) = -6κ² sech²(κx)  (단일 솔리톤)
```

```python
def kdv_spectral(kappa=0.5, T=10.0, N=256, dt=0.01):
    """
    푸리에 스펙트럼 방법을 사용하여 KdV 방정식 풀기.

    ∂u/∂t + u ∂u/∂x + ∂³u/∂x³ = 0
    """
    # 격자 (솔리톤 전파를 위한 더 큰 영역)
    L = 40 * np.pi
    x = np.linspace(-L/2, L/2, N, endpoint=False)
    k = fftfreq(N, L/N) * 2 * np.pi

    # 초기 조건: 단일 솔리톤
    u = -6 * kappa**2 / np.cosh(kappa * x)**2

    # 시간 적분
    nt = int(T / dt)

    def rhs(u):
        """KdV 방정식의 우변."""
        u_hat = fft(u)

        # 분산: ∂³u/∂x³
        dispersion = -(1j * k)**3 * u_hat

        # 비선형성: -u ∂u/∂x
        ux = np.real(ifft(1j * k * u_hat))
        nonlinear_phys = -u * ux
        nonlinear = fft(nonlinear_phys)

        du_dt_hat = dispersion + nonlinear
        return np.real(ifft(du_dt_hat))

    # RK4 시간 진전
    u_save = [u.copy()]
    for n in range(nt):
        k1 = rhs(u)
        k2 = rhs(u + 0.5*dt*k1)
        k3 = rhs(u + 0.5*dt*k2)
        k4 = rhs(u + dt*k3)

        u = u + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)

        if n % 100 == 0:
            u_save.append(u.copy())

    # 솔리톤 전파 플롯
    plt.figure(figsize=(12, 6))
    for i, u_snap in enumerate(u_save[::5]):
        plt.plot(x, u_snap, label=f't = {i*5*dt:.1f}', alpha=0.7)
    plt.xlabel('x')
    plt.ylabel('u(x,t)')
    plt.title('KdV Equation: Soliton Propagation')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('kdv_spectral.png', dpi=150)
    plt.close()

    return x, u_save

x, u_save = kdv_spectral(kappa=0.5, T=10.0)
print("KdV equation solved successfully.")
```

---

## 연습 문제

### 연습 1: 지수 수렴(Exponential Convergence)
스펙트럼 방법(Spectral Methods)이 매끄러운 주기 함수에 대해 지수 수렴을 달성함을 보이시오. [0, 2π]에서 N개의 푸리에(Fourier) 모드를 사용하여 u(x) = e^(sin(x))를 근사하시오. N = 4, 8, 16, 32, 64에 대해 L∞ 오차를 계산하고 로그 스케일 플롯에서 지수 감소를 검증하시오.

<details><summary>정답 보기</summary>

```python
def test_exponential_convergence():
    def u_exact(x):
        return np.exp(np.sin(x))

    N_values = [4, 8, 16, 32, 64]
    errors = []

    x_fine = np.linspace(0, 2*np.pi, 1000, endpoint=False)
    u_fine = u_exact(x_fine)

    for N in N_values:
        x = np.linspace(0, 2*np.pi, N, endpoint=False)
        u = u_exact(x)

        # Interpolate using spectral method
        u_hat = fft(u)

        # Evaluate on fine grid
        k = fftfreq(N, 2*np.pi/N) * 2 * np.pi
        u_interp = np.zeros(len(x_fine))
        for j, xj in enumerate(x_fine):
            u_interp[j] = np.real(np.sum(u_hat * np.exp(1j * k * xj)) / N)

        error = np.linalg.norm(u_fine - u_interp, np.inf)
        errors.append(error)
        print(f"N={N:3d}: Error = {error:.2e}")

    plt.figure(figsize=(8, 5))
    plt.semilogy(N_values, errors, 'o-', linewidth=2, markersize=8)
    plt.xlabel('N (number of modes)')
    plt.ylabel('L∞ error')
    plt.title('Exponential Convergence of Spectral Method')
    plt.grid(True, which='both', alpha=0.3)
    plt.tight_layout()
    plt.savefig('exponential_convergence.png', dpi=150)
    plt.close()

test_exponential_convergence()
```

오차는 N=4에서 ~1e-1에서 N=64에서 ~1e-14로 급격히 감소하여 지수적(스펙트럼) 수렴을 확인합니다. 다항식 방법은 O(N^{-p})의 대수적 감소를 보일 것입니다.
</details>

### 연습 2: 체비셰프 보간과 룬지 현상(Runge Phenomenon)
룬지 함수(Runge function) f(x) = 1/(1 + 25x²)를 [-1, 1]에서 (a) N+1개의 균일 간격 점과 (b) 체비셰프-가우스-로바토(Chebyshev-Gauss-Lobatto) 점을 사용하여 N = 10, 20에 대해 보간하시오. 세밀한 격자에서 L∞ 오차를 계산하고, 체비셰프 절점이 룬지 현상을 피하는 이유를 설명하시오.

<details><summary>정답 보기</summary>

```python
def runge_vs_chebyshev(N=16):
    f = lambda x: 1.0 / (1 + 25 * x**2)
    x_fine = np.linspace(-1, 1, 500)
    f_fine = f(x_fine)

    # Uniform nodes
    x_uni = np.linspace(-1, 1, N+1)
    p_uni = np.polyfit(x_uni, f(x_uni), N)
    err_uni = np.max(np.abs(np.polyval(p_uni, x_fine) - f_fine))

    # Chebyshev-Gauss-Lobatto nodes
    j = np.arange(N+1)
    x_cheb = np.cos(np.pi * j / N)
    # Barycentric interpolation at fine points
    # (simple approach: use numpy polynomial fit on Chebyshev nodes)
    p_cheb = np.polyfit(x_cheb, f(x_cheb), N)
    err_cheb = np.max(np.abs(np.polyval(p_cheb, x_fine) - f_fine))

    print(f"N={N}: Uniform L∞ error = {err_uni:.4e}, Chebyshev L∞ error = {err_cheb:.4e}")

runge_vs_chebyshev(N=10)
runge_vs_chebyshev(N=20)
```

균일 절점은 끝점 근처에서 큰 진동(룬지 현상)을 발생시켜 N이 증가할수록 오차가 커집니다. 체비셰프 절점은 ±1 근처에 집중되어 리베스크 상수(Lebesgue constant)를 최소화하고 진동을 억제합니다. 체비셰프 보간은 진함수로 빠르게 수렴합니다.
</details>

### 연습 3: 스펙트럼 열 방정식(Spectral Heat Equation)
주기 경계 조건과 초기 조건 u(x,0) = sin(x)를 사용하여 [0, 2π]에서 1D 열 방정식 ∂u/∂t = ∂²u/∂x²을 풀어보시오. 적분 인수(integrating factor) 방법으로 푸리에 스펙트럼 방법을 사용하여 정확한 시간 적분을 수행하시오. t = 1에서 수치해를 정확한 해 u(x,t) = e^(-t) sin(x)와 비교하시오.

<details><summary>정답 보기</summary>

```python
def spectral_heat_equation(T=1.0, N=64, dt=0.01):
    x = np.linspace(0, 2*np.pi, N, endpoint=False)
    k = fftfreq(N, 1.0/N)  # integer wavenumbers

    # Initial condition
    u = np.sin(x)
    u_hat = fft(u)

    # Exact time integration: u_hat(k, t) = u_hat(k, 0) * exp(-k^2 * t)
    t_final = T
    u_hat_final = u_hat * np.exp(-k**2 * t_final)
    u_spectral = np.real(ifft(u_hat_final))

    u_exact = np.exp(-T) * np.sin(x)
    error = np.linalg.norm(u_spectral - u_exact, np.inf)
    print(f"Spectral heat equation error at t={T}: {error:.2e}")

    plt.figure(figsize=(8, 4))
    plt.plot(x, u_exact, 'b-', label='Exact', linewidth=2)
    plt.plot(x, u_spectral, 'r--', label='Spectral', linewidth=2)
    plt.xlabel('x'); plt.ylabel('u(x,t)')
    plt.title(f'Heat Equation at t={T}')
    plt.legend(); plt.grid(True)
    plt.tight_layout()
    plt.savefig('spectral_heat.png', dpi=150)
    plt.close()

spectral_heat_equation(T=1.0)
```

적분 인수 방법은 스펙트럼 공간에서 선형 부분을 정확히 풀어냅니다. 오차는 기계 정밀도 수준(~1e-15)에 있어 주기 경계를 가진 매끄러운 문제에 대한 스펙트럼 방법의 강력함을 보여줍니다.
</details>

### 연습 4: KdV 2-솔리톤 충돌
KdV 스펙트럼 솔버를 수정하여 두 솔리톤을 초기화하시오. 초기 조건:
```
u(x,0) = -6κ₁² sech²(κ₁(x+10)) - 6κ₂² sech²(κ₂(x-10))
```
κ₁ = 0.5, κ₂ = 0.8. T = 20까지 시뮬레이션하고, 충돌 후 두 솔리톤이 원래의 형태와 속도로 나오는지 검증하시오(탄성 충돌).

<details><summary>정답 보기</summary>

```python
def kdv_two_soliton(T=20.0, N=512, dt=0.005):
    L = 60 * np.pi
    x = np.linspace(-L/2, L/2, N, endpoint=False)
    k = fftfreq(N, L/N) * 2 * np.pi

    kappa1, kappa2 = 0.5, 0.8
    # Two-soliton initial condition (solitons at x=-10 and x=+10)
    u = (-6 * kappa1**2 / np.cosh(kappa1 * (x + 10))**2
         - 6 * kappa2**2 / np.cosh(kappa2 * (x - 10))**2)

    def rhs(u):
        u_hat = fft(u)
        dispersion = -(1j * k)**3 * u_hat
        ux = np.real(ifft(1j * k * u_hat))
        nonlinear = fft(-u * ux)
        return np.real(ifft(dispersion + nonlinear))

    u_initial = u.copy()
    nt = int(T / dt)
    for n in range(nt):
        k1 = rhs(u); k2 = rhs(u + 0.5*dt*k1)
        k3 = rhs(u + 0.5*dt*k2); k4 = rhs(u + dt*k3)
        u = u + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)

    plt.figure(figsize=(10, 4))
    plt.plot(x, u_initial, 'b-', label='t=0', linewidth=2)
    plt.plot(x, u, 'r-', label=f't={T}', linewidth=2)
    plt.xlabel('x'); plt.ylabel('u')
    plt.title('KdV Two-Soliton Collision')
    plt.legend(); plt.grid(True); plt.xlim(-L/4, L/4)
    plt.tight_layout()
    plt.savefig('kdv_two_soliton.png', dpi=150)
    plt.close()
    print(f"Mass conservation: initial={np.sum(u_initial):.6f}, final={np.sum(u):.6f}")

kdv_two_soliton()
```

충돌 후 두 솔리톤은 원래의 진폭과 속도로 다시 나타나며, 위치만 이동합니다. 이 탄성 충돌(에너지 손실 없음)은 방정식의 적분 가능 구조에서 비롯된 KdV 솔리톤의 특성입니다. 질량(및 기타 보존량)은 높은 정확도로 보존되어야 합니다.
</details>

### 연습 5: 에일리어싱 오류(Aliasing Error) 시연
N = 32의 구체적인 예에서 에일리어싱(aliasing)의 효과를 보이시오. u = sin(N/4 · x)와 v = sin(N/4 · x)를 계산하고 디에일리어싱(dealiasing) 없이 u·v를 스펙트럼 공간에서 계산하시오. 결과를 3/2 규칙을 사용한 올바른 디에일리어싱 버전과 비교하시오.

<details><summary>정답 보기</summary>

```python
def aliasing_demonstration(N=32):
    x = np.linspace(0, 2*np.pi, N, endpoint=False)
    k = fftfreq(N, 1.0/N)

    # Two modes that when multiplied produce frequencies beyond N/2
    u = np.sin(int(N/4) * x)   # mode N/4
    v = np.sin(int(N/4) * x)   # same mode

    # Naive (aliased) product
    w_naive = u * v
    w_naive_hat = np.abs(fft(w_naive))

    # Dealiased product using 3/2 rule
    w_dealiased = dealias_product_3_2_rule(u, v)
    w_dealiased_hat = np.abs(fft(w_dealiased))

    # Exact: sin(k0*x)*sin(k0*x) = 0.5*(1 - cos(2*k0*x))
    # Mode 0 and mode 2*k0 should appear
    k0 = int(N/4)
    w_exact_hat = np.zeros(N)
    w_exact_hat[0] = N/2           # DC component
    w_exact_hat[min(2*k0, N-1)] = N/4  # frequency 2*k0

    print(f"Mode 0:    naive={w_naive_hat[0]:.2f}, dealiased={w_dealiased_hat[0]:.2f}, exact={N/2:.2f}")
    print(f"Mode {2*k0}: naive={w_naive_hat[2*k0 % N]:.2f}, dealiased={w_dealiased_hat[2*k0 % N]:.2f}, exact={N/4:.2f}")
    print("Aliasing causes spurious energy at wrong wavenumbers in the naive case.")

aliasing_demonstration(N=32)
```

모드 N/4를 제곱하면 곱에 나이퀴스트 주파수(Nyquist frequency)인 모드 N/2가 포함됩니다. 디에일리어싱 없이는 이것이 모드 0으로 에일리어스되어 저주파 내용을 오염시킵니다. 3/2 규칙은 곱셈 전에 스펙트럼을 제로 패딩하여 N/2 모드를 올바르게 캡처하고 정확한 결과를 반환합니다.
</details>

---

## 이동
- 이전: [20. 몬테카를로 시뮬레이션](20_Monte_Carlo_Simulation.md)
- 다음: [22. 유한 요소 방법](22_Finite_Element_Method.md)
- [개요로 돌아가기](00_Overview.md)
