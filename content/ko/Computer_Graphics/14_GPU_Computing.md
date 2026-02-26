# 14. GPU 컴퓨팅

[← 이전: 13. 파티클 시스템과 이펙트](13_Particle_Systems_and_Effects.md) | [다음: 15. 실시간 렌더링 기법 →](15_Real_Time_Rendering_Techniques.md)

---

## 학습 목표

1. GPU 아키텍처 이해: SIMT(Single Instruction, Multiple Threads) 실행 모델, 워프(Warp)/웨이브프런트(Wavefront), 점유율(Occupancy)
2. 컴퓨트 셰이더(Compute Shader)와 실행 모델(워크 그룹, 공유 메모리) 설명
3. 일반적인 GPGPU 병렬 패턴 구현: 리덕션(Reduction), 프리픽스 합(Prefix Sum), 병렬 정렬(Parallel Sorting)
4. 이미지 처리 작업에 GPU 컴퓨팅 적용
5. GPU 컴퓨팅 프레임워크 비교: CUDA, OpenCL, 컴퓨트 셰이더, WebGPU
6. GPU 메모리 계층 이해: 전역(Global), 공유(Shared), 로컬(Local, 레지스터), 상수(Constant) 메모리
7. 실용적 고려사항 파악: CPU-GPU 데이터 전송, 동기화(Synchronization), 점유율(Occupancy) 튜닝
8. Python으로 GPU 스타일의 병렬 알고리즘을 구현하고 GPU 대응 코드를 이해

---

## 왜 중요한가

현대 GPU는 단순한 렌더링 기계가 아닙니다 -- 초당 수조 회 연산이 가능한 대규모 병렬 프로세서입니다. 고성능 GPU는 수천 개의 코어가 일제히 동작하며, 적합한 워크로드에서 CPU 대비 10~100배의 처리량을 제공합니다. 이 성능은 실시간 그래픽스뿐 아니라 딥러닝(Deep Learning), 과학 시뮬레이션, 암호화폐 채굴, 데이터 분석을 구동합니다.

GPU가 어떻게 연산하는지 이해하는 것은 그래픽스나 성능이 중요한 컴퓨팅 분야에서 일하는 모든 사람에게 필수입니다. CUDA나 Vulkan 컴퓨트 셰이더를 직접 작성하지 않더라도, GPU가 잘하는 것(그리고 못하는 것)을 아는 것은 알고리즘과 데이터 구조를 설계하는 방식에 근본적인 영향을 미칩니다.

---

## 1. GPU 아키텍처

### 1.1 CPU와 GPU 설계 철학

CPU는 **지연 시간(Latency)** 최적화를 위해 설계되었습니다 -- 큰 캐시, 분기 예측(Branch Prediction), 비순서 실행(Out-of-order Execution)으로 단일 스레드를 최대한 빠르게 실행합니다. GPU는 **처리량(Throughput)** 최적화를 위해 설계되었습니다 -- 수천 개의 스레드를 동시에 실행하며 대규모 병렬성으로 메모리 지연을 숨깁니다.

```
CPU: Few cores, complex control logic
┌─────────────────────────────────┐
│  Core 0  │  Core 1  │  Core 2  │  ...  (4-64 cores)
│  [ALU]   │  [ALU]   │  [ALU]   │
│  [Cache] │  [Cache] │  [Cache] │
│  [Branch Pred]  [OoO Exec]     │
└─────────────────────────────────┘

GPU: Many cores, simple control logic
┌────────────────────────────────────────────────┐
│  SM 0          │  SM 1          │  SM 2    ... │  (30-150 SMs)
│  ┌──┬──┬──┬──┐│  ┌──┬──┬──┬──┐│               │
│  │32│32│32│32││  │32│32│32│32││               │  (32-128 cores/SM)
│  └──┴──┴──┴──┘│  └──┴──┴──┴──┘│               │
│  [Shared Mem]  │  [Shared Mem]  │               │
│  [Scheduler]   │  [Scheduler]   │               │
└────────────────────────────────────────────────┘
```

### 1.2 SIMT: 단일 명령, 다중 스레드

GPU는 **SIMT**(Single Instruction, Multiple Threads, 단일 명령 다중 스레드) 모델로 명령을 실행합니다. 스레드 그룹(NVIDIA에서는 **워프(Warp)**, AMD에서는 **웨이브프런트(Wavefront)**라고 함)이 동일한 명령을 동시에 실행합니다:

- **NVIDIA**: 워프(Warp) = 32개 스레드
- **AMD**: 웨이브프런트(Wavefront) = 32 또는 64개 스레드

워프 내 모든 스레드는 동기적으로(Lockstep) 실행됩니다. 스레드가 다른 분기를 취하면(분기 발산, Divergence), 두 경로 모두 순차적으로 실행되며 활성 경로에 없는 스레드는 마스킹됩니다. 이로 인해 분기(Branching)가 비용이 큽니다.

### 1.3 스레드 계층

GPU 스레드는 계층적으로 구성됩니다:

| 레벨 | NVIDIA 용어 | OpenGL/Vulkan 용어 | 크기 |
|------|------------|-------------------|------|
| 단일 스레드 | Thread | Invocation | 1 |
| 동기 그룹 | Warp (32) | Subgroup | 32-64 |
| 협력 그룹 | Thread Block | Work Group | 64-1024 |
| 전체 디스패치 | Grid | Dispatch | 수백만 |

**워크 그룹(Work Group)** 스레드는 다음을 할 수 있습니다:
- **배리어(Barrier)**를 통한 동기화 (모든 스레드가 배리어에 도달할 때까지 대기)
- **공유 메모리(Shared Memory)**(빠른 온칩 SRAM)를 통한 데이터 공유
- 그룹 내 통신

서로 다른 워크 그룹의 스레드는 단일 디스패치 중에 **직접 통신하거나 동기화할 수 없습니다**.

### 1.4 점유율 (Occupancy)

**점유율(Occupancy)**은 활성 워프 수 대 SM이 지원할 수 있는 최대 워프 수의 비율입니다. 점유율이 높을수록 지연 숨기기(Latency Hiding)가 더 효과적입니다:

$$\text{Occupancy} = \frac{\text{Active warps per SM}}{\text{Maximum warps per SM}}$$

점유율은 다음에 의해 제한됩니다:
- **스레드당 레지스터 수**: 레지스터가 많을수록 SM에 더 적은 스레드 배치
- **워크 그룹당 공유 메모리**: 공유 메모리가 클수록 SM당 더 적은 그룹 배치
- **워크 그룹 크기**: 워프 크기의 배수여야 함

**경험 법칙**: 50% 이상의 점유율 목표. 때로는 더 나은 레지스터 활용으로 낮은 점유율이 더 빠를 수 있습니다(명령 수준 병렬성이 보완).

---

## 2. 컴퓨트 셰이더

### 2.1 컴퓨트 셰이더란?

**컴퓨트 셰이더(Compute Shader)**는 그래픽스 파이프라인 외부에서 실행되는 GPU 프로그램입니다. 임의의 버퍼와 텍스처를 읽고 쓰며 범용 연산을 수행합니다.

OpenGL/Vulkan에서:

```glsl
#version 430
layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(std430, binding = 0) buffer InputBuffer {
    float data_in[];
};

layout(std430, binding = 1) buffer OutputBuffer {
    float data_out[];
};

uniform uint N;

void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= N) return;

    // Each thread processes one element
    data_out[idx] = data_in[idx] * 2.0;
}
```

CPU에서 디스패치:
```cpp
glDispatchCompute(ceil(N / 256.0), 1, 1);
glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
```

### 2.2 워크 그룹 레이아웃

컴퓨트 셰이더는 3D 워크 그룹 크기(로컬 크기)를 정의하며 워크 그룹의 3D 그리드로 디스패치됩니다:

- **1D 디스패치** (배열용 일반적): `local_size_x = 256`, 디스패치 `(ceil(N/256), 1, 1)`
- **2D 디스패치** (이미지): `local_size_x = 16, local_size_y = 16`, 디스패치 `(ceil(W/16), ceil(H/16), 1)`
- **3D 디스패치** (볼륨): 체적 데이터에 사용

**워크 그룹 크기 선택**:
- 완전한 활용을 위해 워프 크기(32 또는 64)의 배수여야 함
- 일반적인 선택: 64, 128, 256 또는 512개 스레드
- 2D의 경우: 16x16 = 256개 스레드가 인기 있는 선택
- 더 큰 그룹은 공유 메모리를 더 많이 사용하지만 스레드 간 통신이 더 많이 가능

### 2.3 공유 메모리 (Shared Memory)

**공유 메모리(Shared Memory)**(HLSL에서 "groupshared", GLSL에서 "shared")는 워크 그룹의 모든 스레드가 접근할 수 있는 빠른 온칩 메모리입니다:

```glsl
shared float cache[256];

void main() {
    uint local_idx = gl_LocalInvocationID.x;
    uint global_idx = gl_GlobalInvocationID.x;

    // Load from global memory into shared memory (fast subsequent access)
    cache[local_idx] = data_in[global_idx];

    // Barrier: ensure all threads have loaded before proceeding
    barrier();

    // Now every thread can read any element in cache[]
    // Example: average with neighbors
    float left  = (local_idx > 0) ? cache[local_idx - 1] : cache[local_idx];
    float right = (local_idx < 255) ? cache[local_idx + 1] : cache[local_idx];
    data_out[global_idx] = (left + cache[local_idx] + right) / 3.0;
}
```

공유 메모리는 SM당 일반적으로 16~96 KB이며, 전역 메모리보다 약 100배 빠른 접근 지연 시간을 가집니다.

---

## 3. GPGPU 병렬 패턴

### 3.1 병렬 리덕션 (Parallel Reduction)

**문제**: $N$개 요소의 합(또는 최대, 최소 등)을 계산합니다.

**순차적**: $O(N)$

**병렬**: $O(N/P + \log P)$, 여기서 $P$는 스레드 수.

**알고리즘**:

```
Step 0: [5, 3, 8, 1, 4, 7, 2, 6]    (8 elements)
Step 1: [8, _, 9, _, 11, _, 8, _]    (add pairs)
Step 2: [17, _, _, _, 19, _, _, _]   (add pairs of sums)
Step 3: [36, _, _, _, _, _, _, _]    (final sum)
```

각 단계에서 활성 스레드가 절반으로 줄어듭니다. $\log_2 N$ 단계 후 결과는 요소 0에 있습니다.

```python
import numpy as np

def gpu_style_reduction(data, op=np.add):
    """
    Simulate a GPU parallel reduction.
    Each 'step' represents one GPU synchronization barrier.
    The actual GPU executes all active threads simultaneously.
    """
    n = len(data)
    # Why copy: GPU would work on a buffer, not modify the input
    buf = data.copy().astype(float)

    stride = 1
    steps = 0
    while stride < n:
        # In a real GPU, all threads with (idx % (2*stride) == 0) execute
        # Why this stride pattern: avoids bank conflicts in shared memory
        for i in range(0, n, 2 * stride):
            if i + stride < n:
                buf[i] = op(buf[i], buf[i + stride])
        stride *= 2
        steps += 1

    print(f"  Reduction: {n} elements in {steps} steps (log2={np.log2(n):.0f})")
    return buf[0]


data = np.array([5, 3, 8, 1, 4, 7, 2, 6, 9, 10, 11, 12, 13, 14, 15, 16])
total = gpu_style_reduction(data)
print(f"  Sum = {total} (expected: {data.sum()})")
```

### 3.2 프리픽스 합 (Prefix Sum, Scan)

**문제**: 배열 $[a_0, a_1, ..., a_{n-1}]$이 주어졌을 때, 누적 합 $[a_0, a_0+a_1, a_0+a_1+a_2, ...]$ (포함 스캔, Inclusive Scan) 또는 $[0, a_0, a_0+a_1, ...]$ (배타 스캔, Exclusive Scan)을 계산합니다.

프리픽스 합은 다음에서 사용되는 핵심 빌딩 블록입니다:
- 스트림 컴팩션(Stream Compaction) (죽은 파티클 제거)
- 기수 정렬(Radix Sort)
- 히스토그램 평탄화(Histogram Equalization)
- 공간 데이터 구조 구축

**블렐로크 스캔(Blelloch Scan)** (작업 효율적인 병렬 프리픽스 합):

**업-스윕(Up-sweep)** (리덕션):
```
Step 0: [3, 1, 7, 0, 4, 1, 6, 3]
Step 1: [3, 4, 7, 7, 4, 5, 6, 9]   (pairs summed)
Step 2: [3, 4, 7, 11, 4, 5, 6, 14]  (quads summed)
Step 3: [3, 4, 7, 11, 4, 5, 6, 25]  (total in last)
```

**다운-스윕(Down-sweep)** (분배):
```
Step 0: [3, 4, 7, 11, 4, 5, 6, 0]  (set last to 0)
Step 1: [3, 4, 7, 0, 4, 5, 6, 11]
Step 2: [3, 0, 7, 4, 4, 11, 6, 16]
Step 3: [0, 3, 4, 11, 11, 15, 16, 22]  (exclusive scan)
```

```python
def blelloch_scan(data):
    """
    Work-efficient parallel exclusive prefix sum (Blelloch 1990).
    Two phases: up-sweep (reduce) and down-sweep (distribute).
    Total work: O(n), span: O(log n).
    """
    n = len(data)
    buf = data.copy().astype(float)

    # Up-sweep: build partial sums (like reduction)
    stride = 1
    while stride < n:
        # Why 2*stride-1 indexing: we accumulate at specific positions
        for i in range(2 * stride - 1, n, 2 * stride):
            buf[i] += buf[i - stride]
        stride *= 2

    # Set last element to 0 (exclusive scan identity)
    buf[n - 1] = 0

    # Down-sweep: distribute partial sums
    stride = n // 2
    while stride >= 1:
        for i in range(2 * stride - 1, n, 2 * stride):
            temp = buf[i - stride]
            buf[i - stride] = buf[i]
            buf[i] += temp
        stride //= 2

    return buf


data = np.array([3, 1, 7, 0, 4, 1, 6, 3])
result = blelloch_scan(data)
expected = np.concatenate([[0], np.cumsum(data)[:-1]])
print(f"  Input:    {data}")
print(f"  Scan:     {result.astype(int)}")
print(f"  Expected: {expected}")
```

### 3.3 병렬 정렬: 바이토닉 정렬 (Bitonic Sort)

**바이토닉 정렬(Bitonic Sort)**은 비교 패턴이 고정되어 있어(데이터 독립적) 효율적인 병렬 구현이 가능하여 GPU 실행에 적합한 비교 기반 정렬 알고리즘입니다.

**바이토닉 수열(Bitonic Sequence)**은 단조 증가 후 단조 감소하는(또는 반대) 수열입니다. 바이토닉 병합(Bitonic Merge)은 $O(\log n)$ 병렬 단계로 바이토닉 수열을 정렬합니다.

**전체 바이토닉 정렬**: $O(\log^2 n)$ 병렬 단계, 각 단계에서 요소 쌍을 비교하고 교환합니다.

```python
def bitonic_sort(data):
    """
    Bitonic sort: GPU-friendly parallel sorting algorithm.
    Comparison pattern is independent of data values,
    making it ideal for SIMT execution (no divergence).
    """
    arr = data.copy()
    n = len(arr)

    # k: size of bitonic subsequences (doubles each outer step)
    k = 2
    while k <= n:
        # j: comparison distance (halves each inner step)
        j = k // 2
        while j >= 1:
            # All comparisons at this (k, j) level can execute in parallel
            for i in range(n):
                partner = i ^ j  # XOR determines comparison partner

                if partner > i:  # Avoid double-processing
                    # Direction: ascending if in first half of k-block, else descending
                    ascending = ((i & k) == 0)

                    if ascending:
                        if arr[i] > arr[partner]:
                            arr[i], arr[partner] = arr[partner], arr[i]
                    else:
                        if arr[i] < arr[partner]:
                            arr[i], arr[partner] = arr[partner], arr[i]

            j //= 2
        k *= 2

    return arr


data = np.array([8, 3, 5, 1, 7, 2, 6, 4])
sorted_data = bitonic_sort(data)
print(f"  Input:  {data}")
print(f"  Sorted: {sorted_data}")
```

### 3.4 스트림 컴팩션 (Stream Compaction)

**문제**: 배열과 조건자(Predicate)가 주어졌을 때, 조건을 만족하는 요소를 추출합니다(예: "죽은" 파티클 제거).

**GPU 접근 방식**:
1. 각 요소에 대해 병렬로 조건자를 평가 → 플래그 배열 생성 [1, 0, 1, 1, 0, ...]
2. 플래그에 프리픽스 합 적용 → 각 생존 요소의 출력 인덱스 계산
3. 스캐터(Scatter): 각 생존 요소가 계산된 출력 인덱스에 기록

총 작업: $O(n)$, 스팬(Span): $O(\log n)$.

---

## 4. GPU에서의 이미지 처리

### 4.1 컨볼루션 (Convolution)

이미지 컨볼루션(Image Convolution)은 각 픽셀에 커널(필터)을 적용합니다:

$$(I * K)(x, y) = \sum_{i=-r}^{r}\sum_{j=-r}^{r} K(i, j) \cdot I(x+i, y+j)$$

이는 자연스럽게 병렬화됩니다: 각 픽셀의 출력은 자신의 이웃에만 의존합니다.

**GPU 구현**: 각 스레드가 하나의 출력 픽셀을 처리합니다. 공유 메모리는 이미지의 로컬 타일을 저장해 중복된 전역 메모리 읽기를 방지합니다.

### 4.2 분리 가능한 필터 (Separable Filters)

많은 유용한 커널(가우시안(Gaussian), 소벨(Sobel), 박스 필터(Box Filter))은 **분리 가능(Separable)**합니다: 2D 커널을 두 번의 1D 패스로 분해할 수 있습니다:

$$K_{2D} = K_{\text{row}} \cdot K_{\text{col}}^T$$

표준 편차 $\sigma$인 가우시안의 경우:

$$G(x) = \frac{1}{\sqrt{2\pi}\sigma} e^{-x^2/(2\sigma^2)}$$

**분리 컨볼루션(Separable Convolution)**은 작업량을 $O(n^2 k^2)$에서 $O(n^2 \cdot 2k)$로 줄입니다, 여기서 $k$는 커널 반경입니다. 5x5 커널의 경우 2.5배, 11x11의 경우 5.5배 빠릅니다.

### 4.3 Python 구현: 이미지 컨볼루션

```python
import numpy as np

def gpu_style_convolution(image, kernel):
    """
    Simulate GPU-style image convolution.
    In a real GPU, each thread computes one output pixel.
    Shared memory would cache the local tile.
    """
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2
    h, w = image.shape

    # Why we pad: border pixels need neighbors that may be outside the image
    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='reflect')
    output = np.zeros_like(image)

    # Each (i, j) represents a GPU thread computing one output pixel
    for i in range(h):
        for j in range(w):
            # In a real GPU compute shader, this inner sum would use
            # shared memory to avoid redundant global memory reads
            val = 0.0
            for ki in range(kh):
                for kj in range(kw):
                    val += padded[i + ki, j + kj] * kernel[ki, kj]
            output[i, j] = val

    return output


def gaussian_kernel(size, sigma):
    """Create a 2D Gaussian kernel."""
    ax = np.arange(size) - size // 2
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return kernel / kernel.sum()


def separable_gaussian(image, sigma, radius=None):
    """
    Separable Gaussian blur: two 1D passes instead of one 2D pass.
    This is how GPUs actually implement Gaussian blur.
    """
    if radius is None:
        radius = int(3 * sigma)
    size = 2 * radius + 1

    # 1D Gaussian kernel
    ax = np.arange(size) - radius
    k1d = np.exp(-ax**2 / (2 * sigma**2))
    k1d /= k1d.sum()

    h, w = image.shape

    # Horizontal pass: each thread processes one pixel in a row
    padded_h = np.pad(image, ((0, 0), (radius, radius)), mode='reflect')
    temp = np.zeros_like(image)
    for i in range(h):
        for j in range(w):
            temp[i, j] = np.dot(padded_h[i, j:j+size], k1d)

    # Vertical pass: each thread processes one pixel in a column
    padded_v = np.pad(temp, ((radius, radius), (0, 0)), mode='reflect')
    output = np.zeros_like(image)
    for i in range(h):
        for j in range(w):
            output[i, j] = np.dot(padded_v[i:i+size, j], k1d)

    return output


# Demo: Compare 2D vs separable Gaussian
np.random.seed(42)
image = np.random.rand(64, 64)

kernel_2d = gaussian_kernel(5, 1.0)
result_2d = gpu_style_convolution(image, kernel_2d)
result_sep = separable_gaussian(image, 1.0, radius=2)

diff = np.abs(result_2d - result_sep).max()
print(f"  2D vs Separable max difference: {diff:.10f}")
print(f"  2D kernel operations per pixel: {5*5} = 25")
print(f"  Separable operations per pixel: {5+5} = 10")
```

---

## 5. GPU 컴퓨팅 프레임워크

### 5.1 CUDA (NVIDIA)

**CUDA**(Compute Unified Device Architecture)는 NVIDIA의 독점 GPU 컴퓨팅 플랫폼(2007년)입니다. GPU 커널 작성을 위한 C/C++ 확장을 제공합니다.

```cuda
// CUDA kernel: vector addition
__global__ void vecAdd(float *A, float *B, float *C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

// Launch: 256 threads per block
vecAdd<<<ceil(N/256.0), 256>>>(d_A, d_B, d_C, N);
```

**장점**: 성숙한 생태계 (cuBLAS, cuDNN, cuFFT), 뛰어난 툴링 (Nsight), ML/HPC에서 주도적.
**단점**: NVIDIA 전용, 독점.

### 5.2 OpenCL

**OpenCL**(Open Computing Language)은 CPU, GPU, FPGA, DSP를 위한 크로스 플랫폼 병렬 컴퓨팅 표준입니다.

```opencl
__kernel void vecAdd(__global float* A, __global float* B,
                     __global float* C, int N) {
    int idx = get_global_id(0);
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}
```

**장점**: 크로스 벤더 (AMD, NVIDIA, Intel, ARM), 크로스 디바이스.
**단점**: API가 더 장황하고, 최적화와 툴링에서 CUDA보다 뒤처짐.

### 5.3 그래픽스 API 컴퓨트 셰이더

**OpenGL 컴퓨트 셰이더(Compute Shaders)**(4.3+), **Vulkan Compute**, **Metal Compute**, **DirectX 컴퓨트 셰이더**는 그래픽스 API 내에서 GPU 컴퓨팅을 제공합니다:

| 기능 | OpenGL CS | Vulkan CS | Metal CS | DX12 CS |
|------|-----------|-----------|----------|---------|
| API 통합 | 좋음 | 탁월 | 탁월 | 탁월 |
| 크로스 플랫폼 | Linux/Win/macOS* | Linux/Win/Android | macOS/iOS | Windows/Xbox |
| 명시적 제어 | 낮음 | 높음 | 높음 | 높음 |
| 비동기 컴퓨트 | 아니오 | 예 | 예 | 예 |

*OpenGL은 macOS에서 deprecated됨

### 5.4 WebGPU

**WebGPU**는 WebGL을 대체하는 차세대 웹 GPU API입니다:

```wgsl
// WebGPU compute shader (WGSL language)
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    if (idx >= arrayLength(&input)) { return; }
    output[idx] = input[idx] * 2.0;
}
```

**장점**: 웹 네이티브, 현대적인 명시적 API 설계, 브라우저에서 실행.
**단점**: 아직 성숙 중 (2025년 기준), 네이티브 API 대비 성능 오버헤드.

### 5.5 비교 요약

| 프레임워크 | 벤더 종속 | 언어 | 성능 | 생태계 |
|-----------|---------|------|------|--------|
| CUDA | NVIDIA 전용 | C/C++/PTX | 최고 (NVIDIA) | 가장 큼 (ML/HPC) |
| OpenCL | 크로스 벤더 | C99/SPIR-V | 좋음 | 중간 |
| Vulkan Compute | 크로스 벤더 | GLSL/SPIR-V | 탁월 | 성장 중 |
| Metal Compute | Apple 전용 | MSL | 탁월 (Apple) | Apple 생태계 |
| WebGPU | 크로스 벤더 | WGSL | 좋음 | 웹/성장 중 |

---

## 6. GPU 메모리 계층

### 6.1 메모리 유형

```
┌──────────────────────────────────────────────┐
│                  Global Memory                │  (VRAM: 8-24 GB, ~500 GB/s)
│  Large capacity, high latency (~400 cycles)   │
├──────────────────────────────────────────────┤
│         L2 Cache (~4-6 MB, shared)           │
├────────────┬────────────┬────────────────────┤
│   SM 0     │   SM 1     │   SM 2    ...      │
│ ┌────────┐ │ ┌────────┐ │                    │
│ │Shared  │ │ │Shared  │ │  (16-96 KB/SM)     │
│ │Memory  │ │ │Memory  │ │  ~20 cycles        │
│ ├────────┤ │ ├────────┤ │                    │
│ │Register│ │ │Register│ │  (256 KB/SM)       │
│ │File    │ │ │File    │ │  1 cycle           │
│ └────────┘ │ └────────┘ │                    │
└────────────┴────────────┴────────────────────┘
```

| 메모리 | 범위 | 크기 | 지연 시간 | 대역폭 |
|--------|------|------|----------|--------|
| 레지스터(Registers) | 스레드당 | ~255 regs/thread | 1 사이클 | 최고 |
| 공유 메모리(Shared memory) | 워크 그룹당 | 16-96 KB/SM | ~20 사이클 | ~10 TB/s |
| L1 캐시 | SM당 | 32-128 KB | ~30 사이클 | 높음 |
| L2 캐시 | 전역 | 4-6 MB | ~200 사이클 | ~4 TB/s |
| 전역(VRAM) | 전역 | 8-24 GB | ~400 사이클 | ~500-900 GB/s |
| 상수(Constant) | 전역 (읽기 전용) | 64 KB | ~4 사이클 (캐시됨) | 브로드캐스트 |

### 6.2 메모리 접근 패턴

**병합 접근(Coalesced Access)**: 연속된 스레드가 연속된 메모리 주소에 접근하면 GPU는 읽기를 단일 와이드 트랜잭션(128 바이트)으로 결합합니다. 이는 성능에 매우 중요합니다:

```
Good (coalesced):
  Thread 0 reads data[0]
  Thread 1 reads data[1]
  Thread 2 reads data[2]
  ...
  → 1 memory transaction

Bad (strided):
  Thread 0 reads data[0]
  Thread 1 reads data[128]
  Thread 2 reads data[256]
  ...
  → 32 separate transactions (32x slower!)
```

**구조체 배열(AoS, Array of Structures)과 배열 구조체(SoA, Structure of Arrays)**:

```
AoS (bad for GPU):
  struct Particle { float x, y, z, vx, vy, vz; };
  Particle particles[N];
  // Thread k reads particles[k].x → non-coalesced

SoA (good for GPU):
  float x[N], y[N], z[N], vx[N], vy[N], vz[N];
  // Thread k reads x[k] → coalesced
```

### 6.3 뱅크 충돌 (Bank Conflicts)

공유 메모리는 **뱅크(Bank)**(일반적으로 32개 뱅크, 각 4바이트)로 나뉩니다. 두 스레드가 동시에 같은 뱅크에 접근하면 접근이 직렬화됩니다(뱅크 충돌).

```
No conflict: Each thread accesses a different bank
  Thread 0 → Bank 0, Thread 1 → Bank 1, ...

Bank conflict (2-way): Two threads hit the same bank
  Thread 0 → Bank 0, Thread 1 → Bank 0  (serialized: 2x slower)

Broadcast: All threads read the SAME address (no conflict)
  Thread 0 → Bank 0[addr X], Thread 1 → Bank 0[addr X]  (broadcast)
```

**완화 방법**: 같은 뱅크를 반복 접근하는 스트라이드(Stride) 패턴을 피하기 위해 공유 메모리 배열을 패딩(Padding)합니다.

---

## 7. 실용적 고려사항

### 7.1 데이터 전송

PCIe를 통한 CPU-GPU 데이터 전송이 병목이 되는 경우가 많습니다:

| 버스 | 대역폭 | 지연 시간 |
|------|--------|---------|
| PCIe 3.0 x16 | ~16 GB/s | ~10 us |
| PCIe 4.0 x16 | ~32 GB/s | ~10 us |
| PCIe 5.0 x16 | ~64 GB/s | ~10 us |
| GPU 메모리 (VRAM) | ~500-900 GB/s | ~400 사이클 |

**원칙**: CPU-GPU 전송을 최소화합니다. 데이터를 최대한 오래 GPU에 유지합니다. GPU에서 연산하고 최종 결과만 전송합니다.

### 7.2 동기화 (Synchronization)

**워크 그룹 내**: 배리어(GLSL의 `barrier()`, CUDA의 `__syncthreads()`) 사용.

**워크 그룹 간**: 메모리 배리어를 사용하고 다른 디스패치를 재실행합니다. 컴퓨트 셰이더는 단일 디스패치 내에서 워크 그룹 간 동기화를 할 수 없습니다.

**CPU-GPU**: 펜스(Fence)를 사용해 GPU 완료를 대기합니다. 타이트한 루프에서 왕복(Round-trip) 동기화를 피합니다(파이프라인 지연 발생).

### 7.3 GPU 사용 시기

GPU가 뛰어난 경우:
- 문제가 **대규모 병렬(Massively Parallel)** (수백만 개 요소에 동일 연산)
- **충분히 큰 데이터** (전송 오버헤드를 분산시키기 위해 >100K 요소)
- 메모리 접근이 **규칙적** (병합, 예측 가능)
- 분기가 **최소** (균일한 제어 흐름)

GPU가 적합하지 않은 경우:
- 데이터 의존성이 있는 순차적 알고리즘
- 작은 문제 (GPU 런치 오버헤드가 지배)
- 불규칙한 메모리 접근 (포인터 체이싱, 트리)
- 심한 분기 (워프 발산)

---

## 8. GPU 패턴의 Python 시뮬레이션

```python
import numpy as np
import time

def simulate_parallel_map(data, func):
    """
    Simulate a GPU parallel map: apply func to each element independently.
    On a real GPU, each thread processes one element simultaneously.
    In Python, we use NumPy vectorization as an analogue.
    """
    return func(data)


def simulate_parallel_reduction(data):
    """Simulate GPU reduction using NumPy (which uses SIMD internally)."""
    return np.sum(data)


def simulate_image_convolution_tiled(image, kernel, tile_size=16):
    """
    Simulate tiled GPU convolution with shared memory.
    Each tile loads its data into 'shared memory' (a local array),
    including halo pixels needed for the kernel.
    """
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2
    h, w = image.shape

    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='reflect')
    output = np.zeros_like(image)

    # Process in tiles (simulating GPU work groups)
    num_tiles_y = (h + tile_size - 1) // tile_size
    num_tiles_x = (w + tile_size - 1) // tile_size

    tiles_processed = 0
    for ty in range(num_tiles_y):
        for tx in range(num_tiles_x):
            y_start = ty * tile_size
            x_start = tx * tile_size
            y_end = min(y_start + tile_size, h)
            x_end = min(x_start + tile_size, w)

            # Simulated "shared memory" load: tile + halo
            # Why halo: kernel needs neighboring pixels outside the tile
            shared_y_start = y_start
            shared_y_end = y_end + 2 * pad_h
            shared_x_start = x_start
            shared_x_end = x_end + 2 * pad_w

            shared_mem = padded[shared_y_start:shared_y_end,
                                shared_x_start:shared_x_end]

            # Each "thread" in the tile computes one output pixel
            for local_y in range(y_end - y_start):
                for local_x in range(x_end - x_start):
                    val = 0.0
                    for ki in range(kh):
                        for kj in range(kw):
                            val += shared_mem[local_y + ki, local_x + kj] * kernel[ki, kj]
                    output[y_start + local_y, x_start + local_x] = val

            tiles_processed += 1

    print(f"  Processed {tiles_processed} tiles ({tile_size}x{tile_size})")
    return output


# --- Performance comparison demo ---

N = 1_000_000
data = np.random.rand(N)

# Map operation: square each element
start = time.perf_counter()
result_loop = np.array([x ** 2 for x in data])  # "CPU serial"
loop_time = time.perf_counter() - start

start = time.perf_counter()
result_vec = simulate_parallel_map(data, lambda x: x ** 2)  # "GPU parallel"
vec_time = time.perf_counter() - start

print(f"  Map {N:,} elements:")
print(f"    Python loop: {loop_time:.4f}s")
print(f"    NumPy (GPU-like): {vec_time:.4f}s")
print(f"    Speedup: {loop_time/vec_time:.1f}x")

# Reduction
start = time.perf_counter()
total = simulate_parallel_reduction(data)
red_time = time.perf_counter() - start
print(f"\n  Reduction of {N:,} elements: sum={total:.2f} in {red_time:.6f}s")

# Tiled image convolution
print(f"\n  Tiled image convolution:")
image = np.random.rand(128, 128)
kernel = gaussian_kernel(5, 1.0)  # Reuse from previous section
result = simulate_image_convolution_tiled(image, kernel, tile_size=16)
print(f"  Output shape: {result.shape}, mean: {result.mean():.4f}")
```

---

## 9. 그래픽스에서의 GPU: 렌더링을 넘어서

그래픽스 엔진의 GPU는 전통적인 렌더링을 넘어 다양한 컴퓨트 역할을 수행합니다:

| 작업 | 기법 |
|------|------|
| 물리 시뮬레이션 | 파티클 시스템, 천(Cloth), 유체 (SPH) |
| 컬링(Culling) | GPU에서 시야 절두체/오클루전 컬링 |
| 애니메이션 | GPU에서 스키닝(Skinning), 블렌드 셰이프(Blend Shapes) |
| 지형(Terrain) | 높이맵(Heightmap) 생성, 테셀레이션(Tessellation) |
| 포스트 프로세싱 | 블룸(Bloom), SSAO, 모션 블러(Motion Blur), 톤 매핑(Tone Mapping) |
| AI/ML | 뉴럴 렌더링(Neural Rendering), DLSS, 디노이징(Denoising) |
| 레이 트레이싱 | BVH 순회, 교차 (RT 코어) |

**비동기 컴퓨트(Async Compute)**: 현대 API(Vulkan, DX12, Metal)는 서로 다른 GPU 큐에서 컴퓨트와 그래픽스 작업을 동시에 실행할 수 있도록 지원하여 활용률을 높입니다.

---

## 요약

| 개념 | 핵심 아이디어 |
|------|-------------|
| SIMT | 워프(32개) 스레드가 동일한 명령 실행; 발산(Divergence)은 비용이 큼 |
| 워크 그룹 | 동기화하고 메모리를 공유할 수 있는 스레드 (64-1024개) |
| 점유율 | 활성 워프 대 최대 비율; 높을수록 지연 숨기기가 더 효과적 |
| 공유 메모리 | 빠른 온칩 SRAM; ~20 사이클; 워크 그룹 내 공유 |
| 병합 접근 | 연속된 스레드가 연속된 주소를 읽을 때 효율적 |
| 리덕션 | $O(\log n)$ 단계의 병렬 합; 기본적인 GPGPU 패턴 |
| 프리픽스 합 | $O(\log n)$ 스팬의 누적 합; 컴팩션, 정렬의 빌딩 블록 |
| 바이토닉 정렬 | 고정된 비교 패턴; $O(\log^2 n)$ 병렬 단계; 발산 없음 |
| 분리 가능한 필터 | 2D 컨볼루션을 두 번의 1D 패스로 분해; $O(k)$만큼 작업 감소 |
| SoA vs AoS | 배열 구조체(SoA)가 GPU에 친화적 (병합); 구조체 배열(AoS)은 그렇지 않음 |

## 연습 문제

1. **리덕션 변형**: GPU 스타일 리덕션 패턴을 사용해 병렬 최대(Parallel Max)와 병렬 최대 인덱스(Parallel Argmax)를 구현하세요. 1024개의 임의 값 배열로 테스트하세요.

2. **프리픽스 합 검증**: 2의 거듭제곱 크기 배열에 대해 블렐로크 스캔을 구현하세요. `np.cumsum`과 결과를 비교하세요. 2의 거듭제곱이 아닌 크기를 처리하도록 확장하세요.

3. **히스토그램**: 그레이스케일 이미지의 픽셀 강도 히스토그램(256 빈)을 계산하는 GPU 알고리즘을 설계하세요. 빈(Bin) 증분을 위한 원자적 연산(Atomic Operations) 처리 방법을 설명하세요.

4. **타일링 분석**: 1920x1080 이미지에 7x7 가우시안 커널과 16x16 스레드 워크 그룹을 사용할 때 계산하세요: (a) 타일 수, (b) 타일당 공유 메모리(헤일로 포함), (c) 비타일 대비 절약된 전역 메모리 읽기 총 횟수.

5. **SoA 변환**: `AoS` 파티클 구조체 `[{x,y,z,vx,vy,vz}, ...]`를 `SoA` 레이아웃 `{x[], y[], z[], vx[], vy[], vz[]}`로 변환하세요. 두 레이아웃에 대해 NumPy 벡터화 업데이트 속도를 측정하세요.

6. **정렬 비교**: 바이토닉 정렬을 구현하고 크기 256, 1024, 4096 배열에 대해 Python의 내장 `sorted()`와 비교하세요. GPU에서 병렬 방식이 더 빠를 시나리오를 논의하세요.

## 더 읽어보기

- Kirk, D. and Hwu, W. *Programming Massively Parallel Processors*, 4th ed. Morgan Kaufmann, 2022. (표준 CUDA/GPU 컴퓨팅 교재)
- Harris, M. "Parallel Prefix Sum (Scan) with CUDA." *GPU Gems 3*, NVIDIA, 2007. (고전적인 GPU 스캔 구현)
- Akenine-Moller, T. et al. *Real-Time Rendering*, 4th ed. CRC Press, 2018. (23장: 그래픽스 프로그래머를 위한 GPU 아키텍처)
- Sellers, G. *Vulkan Programming Guide*. Addison-Wesley, 2016. (Vulkan 컴퓨트 셰이더)
- WebGPU Specification. W3C, 2024. https://www.w3.org/TR/webgpu/ (웹 네이티브 GPU 컴퓨팅)
