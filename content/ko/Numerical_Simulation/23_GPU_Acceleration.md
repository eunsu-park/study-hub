[이전: 유한 요소법](./22_Finite_Element_Method.md) | [다음: 물리 정보 신경망](./24_PINN.md)

---

# 23. 수치 시뮬레이션을 위한 GPU 가속

> **사전 지식**: NumPy 배열 연산과 기본적인 PDE 풀이(6~10강)에 대한 이해가 필요합니다.

## 학습 목표

이 강의를 완료하면 다음을 수행할 수 있습니다:

1. 과학적 컴퓨팅을 위한 CUDA 프로그래밍 모델과 GPU 아키텍처를 설명한다
2. CuPy를 NumPy의 GPU 가속 대체재로 사용한다
3. GPU에서 PDE 솔버(유한 차분법, 스펙트럴 방법)를 가속한다
4. CPU 대 GPU 성능을 벤치마킹하고 GPU 가속이 유리한 상황을 파악한다
5. GPU 메모리 관리와 데이터 전송 최적화를 처리한다

---

## 목차

1. [시뮬레이션에 GPU를 쓰는 이유](#1-시뮬레이션에-gpu를-쓰는-이유)
2. [CUDA 프로그래밍 모델](#2-cuda-프로그래밍-모델)
3. [CuPy: GPU 가속 NumPy](#3-cupy-gpu-가속-numpy)
4. [GPU 가속 PDE 솔버](#4-gpu-가속-pde-솔버)
5. [성능 최적화](#5-성능-최적화)
6. [CPU vs GPU 벤치마킹](#6-cpu-vs-gpu-벤치마킹)
7. [연습문제](#7-연습문제)

---

## 1. 시뮬레이션에 GPU를 쓰는 이유

### 1.1 CPU vs GPU 아키텍처

```
CPU (소수의 강력한 코어):            GPU (수천 개의 단순 코어):

  ┌────────────────────┐           ┌────────────────────────────────┐
  │  Core 1 │  Core 2  │           │ ████████████████████████████   │
  │ (fast)  │ (fast)   │           │ ████████████████████████████   │
  ├─────────┼──────────┤           │ ████████████████████████████   │
  │  Core 3 │  Core 4  │           │ ████████████████████████████   │
  │ (fast)  │ (fast)   │           │     Thousands of cores         │
  ├─────────┴──────────┤           │     (individually slower)      │
  │    Large Cache     │           ├────────────────────────────────┤
  │    Branch Pred.    │           │      High-bandwidth memory     │
  │    Out-of-order    │           │      (HBM: 1-3 TB/s)          │
  └────────────────────┘           └────────────────────────────────┘

  적합한 작업: 순차적이고 복잡한      적합한 작업: 대규모 배열에 대한
  분기를 포함하는 로직               병렬적이고 단순한 연산
```

### 1.2 GPU 가속이 유리한 경우

| GPU에 유리한 작업 | GPU에 불리한 작업 |
|-----------------|-----------------|
| 대규모 배열 연산 (원소별) | 작은 배열 (< 10,000 원소) |
| 행렬 곱셈 | 복잡한 분기 로직 |
| 스텐실(Stencil) 연산 (PDE 유한 차분) | 순차적 알고리즘 |
| 대규모 격자에서의 FFT | 비규칙적 메모리 접근 |
| 몬테카를로 (완전 병렬) | I/O 병목 연산 |

경험 법칙: N > 10,000이고 연산이 데이터 병렬(data-parallel)일 때 GPU가 유리합니다.

### 1.3 일반적인 속도 향상

```
연산                        배열 크기       CPU 시간    GPU 시간    속도 향상
─────────────────────────────────────────────────────────────────────────
행렬 곱셈                   1000×1000      50 ms       2 ms        25×
원소별 연산                  10M            120 ms      3 ms        40×
2D FFT                     4096×4096      200 ms      8 ms        25×
유한 차분 스텐실              1000×1000      80 ms       1.5 ms      53×
행렬 곱셈                   100×100        0.1 ms      0.5 ms      0.2× ✗
  (너무 작음 — 전송 오버헤드가 지배적)
```

---

## 2. CUDA 프로그래밍 모델

### 2.1 핵심 개념

```
CUDA 계층 구조:

  Grid ─────────────────────────────────────
  │  Block (0,0)    Block (1,0)    Block (2,0)
  │  ┌──────────┐  ┌──────────┐  ┌──────────┐
  │  │ Thread   │  │ Thread   │  │ Thread   │
  │  │ (0,0)    │  │ (0,0)    │  │ (0,0)    │
  │  │ Thread   │  │ Thread   │  │ Thread   │
  │  │ (1,0)    │  │ (1,0)    │  │ (1,0)    │
  │  │ ...      │  │ ...      │  │ ...      │
  │  └──────────┘  └──────────┘  └──────────┘
  │
  │  Block (0,1)    Block (1,1)    Block (2,1)
  │  ┌──────────┐  ┌──────────┐  ┌──────────┐
  │  │ ...      │  │ ...      │  │ ...      │
  │  └──────────┘  └──────────┘  └──────────┘
  ───────────────────────────────────────────

  Grid: 블록들의 집합 (하나의 커널로 실행)
  Block: 스레드 그룹 (공유 메모리, 동기화)
  Thread: 단일 실행 단위
  Warp: 동시에 실행되는 32개 스레드 (SIMT)
```

### 2.2 메모리 계층 구조

```
GPU 메모리 계층 구조:

  ┌─────────────────────────────────┐
  │        Host (CPU) Memory        │  ← 시스템 RAM (접근 속도 느림)
  └──────────────┬──────────────────┘
                 │ PCIe / NVLink
  ┌──────────────┴──────────────────┐
  │       Global Memory (HBM)       │  ← 주 GPU 메모리 (대용량, ~GB)
  ├─────────────────────────────────┤
  │       Shared Memory (SRAM)      │  ← 블록별 캐시 (~48-164 KB)
  ├─────────────────────────────────┤
  │       Registers                 │  ← 스레드별 (가장 빠름)
  └─────────────────────────────────┘

  핵심 원칙: CPU와 GPU 간의 데이터 전송을 최소화한다.
  데이터가 GPU에 올라오면, 가능한 많은 연산을 GPU에서 유지한다.
```

---

## 3. CuPy: GPU 가속 NumPy

### 3.1 CuPy 기초

CuPy는 GPU에서 동작하는 NumPy 호환 API를 제공합니다:

```python
import numpy as np

# NumPy (CPU)
a_cpu = np.random.randn(10000, 10000)
b_cpu = np.random.randn(10000, 10000)
c_cpu = a_cpu @ b_cpu  # CPU 행렬 곱셈

# CuPy (GPU) — 동일한 API!
import cupy as cp

a_gpu = cp.random.randn(10000, 10000)
b_gpu = cp.random.randn(10000, 10000)
c_gpu = a_gpu @ b_gpu  # GPU 행렬 곱셈 (훨씬 빠름)

# CPU와 GPU 간 데이터 전송
a_gpu = cp.asarray(a_cpu)        # CPU → GPU
c_cpu = cp.asnumpy(c_gpu)        # GPU → CPU
c_cpu = c_gpu.get()              # 위와 동일
```

### 3.2 CuPy가 NumPy를 앞서는 경우

```python
# 예제: 대규모 배열에 대한 원소별 연산
import cupy as cp
import numpy as np
import time

def benchmark(lib, N=10_000_000):
    """NumPy vs CuPy의 일반 연산 비교."""
    x = lib.random.randn(N)
    y = lib.random.randn(N)

    if lib == cp:
        cp.cuda.Stream.null.synchronize()  # GPU 준비 확인

    start = time.perf_counter()

    # 원소별 연산 연쇄 (GPU가 뛰어난 부분)
    z = lib.sin(x) * lib.cos(y)
    z = z + lib.exp(-x**2)
    z = lib.sqrt(lib.abs(z))
    result = z.sum()

    if lib == cp:
        cp.cuda.Stream.null.synchronize()  # GPU 완료 대기

    elapsed = time.perf_counter() - start
    return elapsed

# 일반적인 결과:
# NumPy:  ~120 ms
# CuPy:   ~3 ms (40배 속도 향상)
```

### 3.3 CuPy에서 커스텀 CUDA 커널 작성

```python
import cupy as cp

# 특정 스텐실 연산을 위한 커스텀 CUDA 커널 정의
laplacian_kernel = cp.ElementwiseKernel(
    'raw float64 u, int32 nx, int32 ny, float64 dx2, float64 dy2',
    'float64 lap',
    '''
    int i = _ind.get()[0];
    int ix = i / ny;
    int iy = i % ny;

    if (ix > 0 && ix < nx-1 && iy > 0 && iy < ny-1) {
        lap = (u[i+ny] - 2*u[i] + u[i-ny]) / dx2 +
              (u[i+1] - 2*u[i] + u[i-1]) / dy2;
    } else {
        lap = 0.0;
    }
    ''',
    'laplacian_2d'
)
```

---

## 4. GPU 가속 PDE 솔버

### 4.1 열 방정식 (유한 차분법)

```python
import numpy as np


def heat_equation_cpu(nx, ny, nt, dt, alpha=0.01):
    """NumPy를 사용한 CPU 기반 2D 열 방정식 솔버.

    du/dt = alpha * (d²u/dx² + d²u/dy²)
    양해적 오일러(Explicit Euler) 시간 진행.
    """
    dx = 1.0 / (nx - 1)
    dy = 1.0 / (ny - 1)

    u = np.zeros((nx, ny))
    # 초기 조건: 중앙의 뜨거운 점
    u[nx//4:3*nx//4, ny//4:3*ny//4] = 1.0

    rx = alpha * dt / dx**2
    ry = alpha * dt / dy**2

    for _ in range(nt):
        # 배열 슬라이싱을 통한 라플라시안 (벡터화)
        laplacian = (
            rx * (u[2:, 1:-1] - 2*u[1:-1, 1:-1] + u[:-2, 1:-1]) +
            ry * (u[1:-1, 2:] - 2*u[1:-1, 1:-1] + u[1:-1, :-2])
        )
        u[1:-1, 1:-1] += laplacian

    return u


def heat_equation_gpu(nx, ny, nt, dt, alpha=0.01):
    """CuPy를 사용한 동일한 GPU 솔버 — np를 cp로 교체.

    API가 동일하며, CuPy가 GPU 메모리와 커널을 처리합니다.
    """
    # 실제 구현: import cupy as cp 후 np→cp 교체
    # dx = 1.0 / (nx - 1)
    # dy = 1.0 / (ny - 1)
    # u = cp.zeros((nx, ny))
    # ... (cp 대신 np로 동일한 코드)
    pass  # 전체 구현은 예제 파일 참조
```

### 4.2 GPU에서의 스펙트럴 방법

```python
def spectral_diffusion_gpu(N, nt, dt, nu=0.01):
    """GPU에서의 스펙트럴 방법을 이용한 1D 확산.

    FFT 기반 스펙트럴 방법은 GPU에서 엄청난 이점을 누립니다:
    CuPy의 FFT는 cuFFT (NVIDIA의 최적화된 FFT 라이브러리)를 사용합니다.

    du/dt = nu * d²u/dx²
    푸리에 공간에서: du_hat/dt = -nu * k² * u_hat
    """
    # CuPy를 사용한 실제 구현:
    # x = cp.linspace(0, 2*cp.pi, N, endpoint=False)
    # u = cp.sin(x) + 0.5 * cp.sin(3*x)
    # k = cp.fft.fftfreq(N, d=2*cp.pi/N) * 2 * cp.pi
    #
    # for _ in range(nt):
    #     u_hat = cp.fft.fft(u)
    #     u_hat *= cp.exp(-nu * k**2 * dt)
    #     u = cp.fft.ifft(u_hat).real

    # N > 4096일 때 일반적인 속도 향상: 20-50배
    pass
```

### 4.3 입자 시뮬레이션

```python
def nbody_step_cpu(positions, velocities, masses, dt, G=1.0):
    """N체 중력 시뮬레이션 스텝 (CPU).

    O(N²) 쌍별 힘 계산 — 각 입자의 힘이 독립적이므로
    GPU에 완벽하게 적합합니다.
    """
    N = len(positions)
    forces = np.zeros_like(positions)

    for i in range(N):
        for j in range(N):
            if i != j:
                r = positions[j] - positions[i]
                dist = np.linalg.norm(r) + 1e-10
                forces[i] += G * masses[j] * r / dist**3

    velocities += forces * dt
    positions += velocities * dt
    return positions, velocities
```

GPU 버전은 이중 루프를 벡터화된 쌍별 계산으로 대체합니다:

```python
def nbody_step_gpu_vectorized(positions, velocities, masses, dt, G=1.0):
    """GPU에서 벡터화된 N체 시뮬레이션 (명시적 루프 없음).

    브로드캐스팅 사용: 모든 쌍별 거리를 한 번에 계산합니다.
    메모리: O(N²), 하지만 N < 50,000일 때 CPU 루프보다 훨씬 빠릅니다.
    """
    # dx[i,j] = positions[j] - positions[i] (N×N×3 텐서)
    # dist[i,j] = ||dx[i,j]||
    # forces[i] = sum_j G * m[j] * dx[i,j] / dist[i,j]^3
    #
    # 모두 CuPy 브로드캐스팅으로 계산 — Python 루프 없음
    pass
```

---

## 5. 성능 최적화

### 5.1 데이터 전송 최소화

```python
# 나쁜 예: 매 반복마다 전송
for step in range(1000):
    u_cpu = np.array(...)
    u_gpu = cp.asarray(u_cpu)     # CPU → GPU 전송
    result_gpu = compute(u_gpu)
    result_cpu = result_gpu.get()  # GPU → CPU 전송

# 좋은 예: 한 번 전송 후 GPU에서 모든 연산 수행
u_gpu = cp.asarray(u_cpu)  # 최초 1회 전송
for step in range(1000):
    u_gpu = compute(u_gpu)  # GPU에 유지
result_cpu = u_gpu.get()    # 최초 1회 반환 전송
```

### 5.2 메모리 관리

```python
# GPU 메모리는 제한적 — 사용량 모니터링
# mempool = cp.get_default_memory_pool()
# print(f"GPU memory used: {mempool.used_bytes() / 1e9:.2f} GB")
# print(f"GPU memory total: {mempool.total_bytes() / 1e9:.2f} GB")
#
# 미사용 메모리 해제:
# mempool.free_all_blocks()

# 대규모 시뮬레이션: 메모리 맵 배열 사용
# 또는 GPU 메모리에 맞게 데이터를 청크 단위로 처리
```

### 5.3 커널 퓨전(Kernel Fusion)

```python
# 분리된 연산 = 별도의 커널 실행
# y = cp.sin(x)
# z = cp.exp(y)
# w = z + 1.0

# 퓨전 = 하나의 커널 실행 (더 빠름)
# cp.fuse를 통한 자동 퓨전:

# @cp.fuse()
# def fused_op(x):
#     return cp.exp(cp.sin(x)) + 1.0
#
# w = fused_op(x)  # 단일 커널 실행
```

---

## 6. CPU vs GPU 벤치마킹

### 6.1 공정한 벤치마킹

```python
def benchmark_fair(func_cpu, func_gpu, *args, warmup=5, trials=10):
    """공정한 CPU vs GPU 벤치마크.

    핵심: GPU는 워밍업 시간이 필요합니다 (JIT 컴파일, 메모리 할당).
    항상 처음 몇 번의 실행을 버리고 올바르게 동기화합니다.
    """
    # 워밍업 (버림)
    for _ in range(warmup):
        func_gpu(*args)
        # cp.cuda.Stream.null.synchronize()

    # 시간 측정 실행
    cpu_times = []
    for _ in range(trials):
        start = time.perf_counter()
        func_cpu(*args)
        cpu_times.append(time.perf_counter() - start)

    gpu_times = []
    for _ in range(trials):
        # cp.cuda.Stream.null.synchronize()
        start = time.perf_counter()
        func_gpu(*args)
        # cp.cuda.Stream.null.synchronize()
        gpu_times.append(time.perf_counter() - start)

    return np.median(cpu_times), np.median(gpu_times)
```

### 6.2 교차점(Crossover Point)

```
배열 크기에 따른 성능:

Time │
     │  CPU
     │  ╲
     │   ╲          GPU 오버헤드
     │    ╲        (전송, 실행)
     │     ╲           │
     │      ╲──────────┘
     │       ╲
     │  GPU   ╲─────────── GPU 우세 구간
     │         ╲
     │          ╲
     └──────────────────────────── 배열 크기
              ↑
         교차점
         (일반적으로 ~10,000 원소)
```

---

## 7. 연습문제

### 연습문제 1: CuPy vs NumPy 벤치마크

다음 연산에 대해 CuPy와 NumPy를 비교하세요:
1. 행렬 곱셈 (N = 100, 1000, 5000, 10000)
2. 원소별 연산 연쇄 (sin, cos, exp), 배열 크기 10K ~ 10M
3. 64×64에서 4096×4096 격자에서의 2D FFT
4. 각 연산에서 GPU가 더 빨라지는 교차점을 찾으세요
5. 배열 크기 대비 속도 향상을 그래프로 그리세요

### 연습문제 2: GPU 열 방정식

CPU와 GPU 모두에서 2D 열 방정식을 구현하세요:
1. 격자 크기: 100×100, 500×500, 1000×1000, 2000×2000
2. 1000 타임 스텝 실행
3. CPU와 GPU가 동일한 결과를 산출하는지 검증 (부동소수점 허용 오차 내)
4. 벤치마킹하고 격자 크기 대비 속도 향상을 그래프로 그리세요
5. 각 격자 크기에 대한 GPU 메모리 사용량을 추정하세요

### 연습문제 3: GPU에서의 스펙트럴 솔버

스펙트럴 방법을 사용한 1D 이류 방정식을 구현하세요:
1. du/dt + c * du/dx = 0 (c = 1)
2. N = 256, 1024, 4096, 16384 격자점
3. FFT를 이용한 공간 미분
4. CPU (NumPy FFT) vs GPU (CuPy cuFFT) 성능 비교
5. 해석적 해와 수치 해를 비교 검증

### 연습문제 4: N체 시뮬레이션

N체 중력 시뮬레이션을 GPU로 가속하세요:
1. N = 100, 500, 1000, 5000, 10000 입자
2. CPU: NumPy 벡터화 쌍별 계산
3. GPU: CuPy 브로드캐스팅을 이용한 쌍별 거리 계산
4. 100 타임 스텝 실행 후 벤치마킹
5. GPU가 CPU보다 10배 빨라지는 N 값을 찾으세요

### 연습문제 5: 메모리 최적화 대규모 시뮬레이션

GPU 메모리에 담기 어려운 시뮬레이션을 처리하세요:
1. 10000×10000 격자 열 방정식 생성 (float64 기준 800 MB)
2. GPU 여유 메모리가 1 GB 미만이면 헤일로 교환(halo exchange)을 사용해 4개 타일로 분할
3. 각 타일을 GPU에서 처리하고 CPU에서 경계 교환
4. 전체 GPU 방식과 비교 (메모리가 허용하는 경우)
5. 타일링 오버헤드와 GPU 가속 간의 트레이드오프를 논의하세요

---

*23강 끝*
