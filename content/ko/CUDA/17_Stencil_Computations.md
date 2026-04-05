# 17. Stencil Computations

**이전**: [Parallel Sort](./16_Parallel_Sort.md) | **다음**: [Histogram and Binning](./18_Histogram_and_Binning.md)

---

## 학습 목표

이 레슨을 마치면 다음을 할 수 있습니다:

1. 이웃 값을 읽는 1D, 2D, 3D stencil kernel 작성하기
2. 중복 global memory 로드를 제거하기 위해 halo cell과 함께 shared memory tiling 사용하기
3. Stencil kernel에서 주기적(periodic) 및 Dirichlet 경계 조건 구현하기
4. 명시적 유한 차분 시뮬레이션을 위한 시간 스텝 루프 구축하기
5. Stencil의 산술 강도를 식별하고 roofline 성능 예측하기

---

## 1. Stencil이란?

**Stencil computation**은 이웃 값의 고정 패턴을 사용하여 그리드의 각 점을 업데이트합니다. 각 thread는 입력 그리드의 고정된 이웃 영역에서 읽어 하나의 출력 그리드 점을 계산합니다.

```
1D 3점 stencil (r=1):    out[i] = f(in[i-1], in[i], in[i+1])
2D 5점 stencil (4-이웃):
    out[i][j] = f(in[i-1][j], in[i][j-1], in[i][j], in[i][j+1], in[i+1][j])
2D 9점 stencil (8-이웃):
    위에 더해 대각선 포함
```

Stencil은 다음의 핵심 연산입니다:
- 유한 차분 PDE 솔버 (열, 파동, 확산 방정식)
- 유한 요소 방법
- 이미지 합성곱 (가우시안 블러, Sobel, 라플라시안)
- 격자 볼츠만 시뮬레이션

---

## 2. 나이브 2D Stencil Kernel

가장 간단한 구현 — 출력 점당 하나의 thread, global memory에서 직접 읽기:

```c
// 2D 5점 라플라시안 stencil: out[i][j] = in[i-1][j] + in[i+1][j]
//                                       + in[i][j-1] + in[i][j+1]
//                                       - 4*in[i][j]
__global__ void laplacian_2d_naive(
    const float *in, float *out, int Nx, int Ny)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y;  // 행
    int j = blockIdx.x * blockDim.x + threadIdx.x;  // 열

    if (i > 0 && i < Ny - 1 && j > 0 && j < Nx - 1) {
        out[i * Nx + j] =
            in[(i - 1) * Nx + j] + in[(i + 1) * Nx + j] +
            in[i * Nx + (j - 1)] + in[i * Nx + (j + 1)] -
            4.0f * in[i * Nx + j];
    }
}
```

**문제**: 각 요소가 5번 로드됩니다 (stencil 적용당 한 번씩). 5점 stencil이 있는 32×32 thread block에서 경계 thread들은 내부 이웃도 필요로 하는 데이터를 로드합니다 — 상당한 중복입니다.

---

## 3. Halo Cell을 이용한 Shared Memory Tiling

데이터의 타일을 shared memory에 로드할 때, block 내의 모든 읽기를 만족시키기 위해 **halo cell** (5점 stencil의 경우 반지름 1)의 경계를 포함합니다:

```c
#define TILE_W 32
#define TILE_H 32
#define RADIUS  1   // stencil 반지름

__global__ void laplacian_2d_tiled(
    const float *in, float *out, int Nx, int Ny)
{
    // Shared memory 타일은 halo를 포함: (TILE_H+2*R) x (TILE_W+2*R)
    __shared__ float s[(TILE_H + 2 * RADIUS)][(TILE_W + 2 * RADIUS)];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    // 글로벌 인덱스 (내부 thread)
    int i = blockIdx.y * TILE_H + ty;
    int j = blockIdx.x * TILE_W + tx;

    // Shared memory 인덱스는 halo 오프셋 포함
    int si = ty + RADIUS;
    int sj = tx + RADIUS;

    // 타일의 내부 로드
    s[si][sj] = (i < Ny && j < Nx) ? in[i * Nx + j] : 0.0f;

    // Halo 행 로드 (위, 아래)
    if (ty < RADIUS) {
        int above = (i - RADIUS >= 0) ? (i - RADIUS) : 0;
        int below = (i + TILE_H < Ny) ? (i + TILE_H) : Ny - 1;
        s[ty][sj]               = in[above * Nx + j];          // 위 halo
        s[ty + TILE_H + RADIUS][sj] = in[below * Nx + j];      // 아래 halo
    }
    // Halo 열 로드 (왼쪽, 오른쪽)
    if (tx < RADIUS) {
        int left  = (j - RADIUS >= 0) ? (j - RADIUS) : 0;
        int right = (j + TILE_W < Nx) ? (j + TILE_W) : Nx - 1;
        s[si][tx]               = in[i * Nx + left];           // 왼쪽 halo
        s[si][tx + TILE_W + RADIUS] = in[i * Nx + right];      // 오른쪽 halo
    }
    __syncthreads();

    // Stencil 적용 (경계 thread 건너뜀)
    if (i > 0 && i < Ny - 1 && j > 0 && j < Nx - 1) {
        out[i * Nx + j] =
            s[si - 1][sj] + s[si + 1][sj] +
            s[si][sj - 1] + s[si][sj + 1] -
            4.0f * s[si][sj];
    }
}
```

**메모리 접근 감소**: tiling 없이는 각 요소가 global memory에서 ~5번 읽힙니다. Tiling을 사용하면 내부 요소는 shared memory에 한 번 로드되고 5번 읽힙니다. Global 로드는 점당 5회에서 ~1.06회로 감소 (halo 오버헤드).

---

## 4. Stencil Kernel의 산술 강도

```
5점 2D stencil, float (4바이트):
  산술:  5 읽기 + 1 쓰기 → 4 덧셈 + 1 곱셈 = 5 FLOP
  메모리: 5 읽기 + 1 쓰기 = 6 × 4 = 24 바이트 (나이브, 캐시 없음)
  AI:    5 / 24 ≈ 0.21 FLOP/바이트

Shared memory tiling 사용 시 (효과적):
  각 요소를 global에서 한 번 로드, 5번 사용 →
  메모리: ~1.06 × 4 = 4.24 바이트/출력
  AI:    5 / 4.24 ≈ 1.18 FLOP/바이트

RTX 3090: 메모리 대역폭 = 936 GB/s, FP32 peak = 35.6 TFLOPS
Ridge 포인트: 35600 / 936 ≈ 38 FLOP/바이트 → stencil은 메모리 바운드
최대 처리량 (tiled): 936 GB/s × 1.18 FLOP/바이트 ≈ 1.1 TFLOPS
```

Stencil은 거의 항상 메모리 대역폭 바운드입니다. Tiling은 도움이 되지만 근본적인 AI 한계를 바꾸지는 않습니다.

---

## 5. 열 방정식을 위한 시간 스텝 루프

2D 열 방정식: ∂u/∂t = α (∂²u/∂x² + ∂²u/∂y²)

명시적 유한 차분 이산화 (시간에서 전방 오일러, 공간에서 중앙 차분):

```
u[i,j,t+1] = u[i,j,t] + α·Δt/Δx² · (u[i-1,j,t] + u[i+1,j,t] +
                                       u[i,j-1,t] + u[i,j+1,t] - 4·u[i,j,t])
```

`r = α·Δt/Δx²`로 설정. 안정성 조건 (CFL): `r ≤ 0.25` (2D).

```c
// 열 방정식 stencil kernel
__global__ void heat_step(const float *u_old, float *u_new,
                          float r, int Nx, int Ny) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i > 0 && i < Ny - 1 && j > 0 && j < Nx - 1) {
        float center = u_old[i * Nx + j];
        float laplacian =
            u_old[(i - 1) * Nx + j] + u_old[(i + 1) * Nx + j] +
            u_old[i * Nx + (j - 1)] + u_old[i * Nx + (j + 1)] -
            4.0f * center;
        u_new[i * Nx + j] = center + r * laplacian;
    } else if (i < Ny && j < Nx) {
        // Dirichlet BC: 경계는 고정됨
        u_new[i * Nx + j] = u_old[i * Nx + j];
    }
}

// 호스트: ping-pong 버퍼를 이용한 시간 스텝 루프
void run_heat_simulation(int Nx, int Ny, int steps,
                         float alpha, float dt, float dx) {
    float r = alpha * dt / (dx * dx);
    if (r > 0.25f) {
        fprintf(stderr, "CFL 위반: r=%.4f > 0.25, 시뮬레이션 불안정\n", r);
        return;
    }

    size_t bytes = Nx * Ny * sizeof(float);
    float *d_u0, *d_u1;
    cudaMalloc(&d_u0, bytes);
    cudaMalloc(&d_u1, bytes);

    // 초기화 (예: 중심에 가우시안 열원)
    init_gaussian<<<dim3((Nx+15)/16,(Ny+15)/16), dim3(16,16)>>>(d_u0, Nx, Ny);

    dim3 block(16, 16);
    dim3 grid((Nx + 15) / 16, (Ny + 15) / 16);

    for (int t = 0; t < steps; t++) {
        heat_step<<<grid, block>>>(d_u0, d_u1, r, Nx, Ny);
        float *tmp = d_u0; d_u0 = d_u1; d_u1 = tmp;  // ping-pong 교환
    }

    // d_u0가 최종 상태를 보유
    cudaFree(d_u0); cudaFree(d_u1);
}
```

---

## 6. 주기적 경계 조건

주기적 (wrap-around) 경계의 경우, 클램핑된 인덱스를 모듈식 인덱스로 교체합니다:

```c
__global__ void laplacian_periodic(const float *in, float *out, int Nx, int Ny) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= Ny || j >= Nx) return;

    // 주기적 wrap: (i-1+Ny)%Ny는 음수 모듈로 방지
    int im = (i - 1 + Ny) % Ny;
    int ip = (i + 1) % Ny;
    int jm = (j - 1 + Nx) % Nx;
    int jp = (j + 1) % Nx;

    out[i * Nx + j] =
        in[im * Nx + j] + in[ip * Nx + j] +
        in[i * Nx + jm] + in[i * Nx + jp] -
        4.0f * in[i * Nx + j];
}
```

**성능 참고**: 모듈로 연산 (`%`)은 GPU에서 비쌉니다 (나눗셈). 2의 거듭제곱의 경우 `% N`을 `& (N-1)`로 교체하세요. 임의의 N의 경우, 조건부 덧셈/뺄셈을 사용하세요:

```c
// 나눗셈 없는 더 빠른 주기적 인덱스
__device__ int periodic(int idx, int n) {
    if (idx < 0)  return idx + n;
    if (idx >= n) return idx - n;
    return idx;
}
```

---

## 7. 3D Stencil

3D로 확장하면 3D thread block과 7점 stencil (6개 면 이웃)을 사용합니다:

```c
// 3D 7점 stencil (3D에서의 라플라시안)
__global__ void laplacian_3d(const float *in, float *out,
                              int Nx, int Ny, int Nz) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;

    if (ix < 1 || ix >= Nx-1 || iy < 1 || iy >= Ny-1 || iz < 1 || iz >= Nz-1)
        return;

    int stride_y = Nx;
    int stride_z = Nx * Ny;
    int idx      = iz * stride_z + iy * stride_y + ix;

    out[idx] =
        in[idx - 1]         + in[idx + 1]         +   // x 이웃
        in[idx - stride_y]  + in[idx + stride_y]  +   // y 이웃
        in[idx - stride_z]  + in[idx + stride_z]  -   // z 이웃
        6.0f * in[idx];

    // 실행: dim3 block(8,8,8), grid((Nx+7)/8, (Ny+7)/8, (Nz+7)/8)
    // z의 최대 grid dim: 65535 — Nz < 524280에 충분
}
```

**3D의 Shared memory**: 3D halo 로딩은 복잡하지만 높은 FLOP 수를 위해 중요합니다. (8+2)×(8+2)×(8+2) 타일은 10³ × 4 = 4000바이트를 사용 — 48 KB shared memory 내에 충분히 들어감.

---

## 핵심 요약

- Stencil kernel은 **이전** 시간 스텝(또는 공간의 인접 이웃)의 고정 이웃 영역에서 각 그리드 점을 업데이트합니다
- **나이브** stencil은 global memory에서 각 요소를 5회 이상 재로드합니다; halo cell이 있는 **shared memory tiling**은 이를 요소당 ~1회 로드로 줄입니다
- Stencil은 거의 항상 **메모리 대역폭 바운드** — 산술 강도가 낮습니다 (0.2–1.2 FLOP/바이트)
- **Ping-pong 버퍼** (`u_old`, `u_new`)는 시간 스텝 루프에서 읽기-쓰기 경쟁을 방지합니다
- **Dirichlet BC**: 경계 값 고정; **Neumann BC**: 경계 기울기 고정; **주기적 BC**: wrap-around 인덱스
- CFL 안정성 조건은 최대 시간 스텝을 제한합니다: 2D에서 `Δt ≤ Δx²/(4α)`

---

**다음**: [18. Histogram and Binning](./18_Histogram_and_Binning.md) — global atomic, shared memory privatization, CUB DeviceHistogram을 사용한 병렬 histogram 계산을 구현합니다.
