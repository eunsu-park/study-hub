# 23. PDE 솔버 — 열 방정식 (Heat Equation)

**이전**: [FFT on GPU](./22_FFT_on_GPU.md) | **다음**: [Fluid Dynamics LBM](./24_Fluid_Dynamics_LBM.md)

---

## 학습 목표

이 레슨을 마치면 다음을 할 수 있습니다:

1. 명시적 유한 차분으로 2D 열 방정식 이산화하기
2. 이산 라플라시안 연산자를 위한 CUDA stencil kernel 작성하기
3. CFL 안정 조건 식별 및 안전한 시간 단계 선택하기
4. GPU에서 Dirichlet 및 Neumann 경계 조건 구현하기
5. 잔차의 L2-norm을 사용한 정상 상태 수렴 측정하기

---

## 1. 2D 열 방정식

열 방정식은 2D 영역에서 온도 u(x, y, t)의 확산을 지배합니다:

```
∂u/∂t = α · (∂²u/∂x² + ∂²u/∂y²)

α = 열 확산율 [m²/s]
∇²u = 라플라시안 = ∂²u/∂x² + ∂²u/∂y² (2차 공간 미분)

물리적 해석:
  이웃이 더 뜨거우면 u가 증가 (양의 라플라시안)
  이웃이 더 차가우면 u가 감소 (음의 라플라시안)
  정상 상태: ∇²u = 0  (라플라스 방정식)
```

---

## 2. 명시적 유한 차분 이산화

간격 Δx = Δy = h로 Nx × Ny 격자로 영역을 이산화합니다:

```
공간:  u[i][j] ≈ u(i·h, j·h)        i = 0..Ny-1, j = 0..Nx-1
시간:  u^n[i][j] = 시간 단계 n에서의 u

시간에서의 Forward Euler:
  ∂u/∂t ≈ (u^{n+1}[i][j] - u^n[i][j]) / Δt

공간에서의 중앙 차분:
  ∂²u/∂x² ≈ (u[i][j-1] - 2u[i][j] + u[i][j+1]) / h²
  ∂²u/∂y² ≈ (u[i-1][j] - 2u[i][j] + u[i+1][j]) / h²

결합 업데이트 규칙 (r = α·Δt/h²으로 설정):
  u^{n+1}[i][j] = u^n[i][j] + r · (u[i-1][j] + u[i+1][j]
                                   + u[i][j-1] + u[i][j+1]
                                   - 4·u[i][j])
```

---

## 3. CFL 안정 조건

명시적 방식은 **조건부 안정** — 시간 단계가 충분히 작을 때만 안정합니다:

```
2D 열 방정식의 CFL 조건:
  r = α·Δt/h² ≤ 1/4

따라서: Δt ≤ h² / (4·α)

예: α = 0.1, h = 0.01
  Δt_max = 0.01² / (4 × 0.1) = 0.0025초/단계
  T = 1초 시뮬레이션: 최소 400번의 시간 단계 필요

위반 (r > 0.25): 해가 한계 없이 성장 (수치 폭발)
```

```c
// 시뮬레이션 시작 전 매개변수 검증
bool check_cfl(float alpha, float dt, float h) {
    float r = alpha * dt / (h * h);
    if (r > 0.25f) {
        fprintf(stderr, "CFL 위반! r=%.4f > 0.25. dt를 %.6f로 줄이세요\n",
                r, 0.25f * h * h / alpha);
        return false;
    }
    printf("CFL: r=%.4f (안정, 최대=0.25)\n", r);
    return true;
}
```

---

## 4. 열 방정식 Kernel

```c
// 2D 열 방정식 업데이트: 내부 격자점당 하나의 thread
__global__ void heat_eq_step(
    const float *u_old, float *u_new,
    int Nx, int Ny, float r)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;  // 열
    int i = blockIdx.y * blockDim.y + threadIdx.y;  // 행

    if (i <= 0 || i >= Ny - 1 || j <= 0 || j >= Nx - 1) return;

    int idx  = i * Nx + j;
    float u0 = u_old[idx];

    float laplacian =
        u_old[(i - 1) * Nx + j] +   // 위
        u_old[(i + 1) * Nx + j] +   // 아래
        u_old[i * Nx + (j - 1)] +   // 왼쪽
        u_old[i * Nx + (j + 1)] -   // 오른쪽
        4.0f * u0;

    u_new[idx] = u0 + r * laplacian;
}

// Shared memory 버전 (전체 halo 구현은 레슨 17 참조)
__global__ void heat_eq_step_shared(
    const float *u_old, float *u_new,
    int Nx, int Ny, float r)
{
    __shared__ float s[18][18];  // 16×16 block + 양쪽 1-셀 halo
    int tx = threadIdx.x, ty = threadIdx.y;
    int j  = blockIdx.x * 16 + tx;
    int i  = blockIdx.y * 16 + ty;

    // 중심 로드
    s[ty + 1][tx + 1] = (i < Ny && j < Nx) ? u_old[i * Nx + j] : 0.f;

    // Halo 로드 (단순화: 위/아래만 표시)
    if (ty == 0)
        s[0][tx + 1] = (i > 0 && j < Nx) ? u_old[(i - 1) * Nx + j] : 0.f;
    if (ty == 15)
        s[17][tx + 1] = (i < Ny - 1 && j < Nx) ? u_old[(i + 1) * Nx + j] : 0.f;
    // (왼쪽/오른쪽 halo: tx==0 및 tx==15 케이스 — 간략화를 위해 생략)
    __syncthreads();

    if (i <= 0 || i >= Ny - 1 || j <= 0 || j >= Nx - 1) return;
    float laplacian = s[ty][tx+1] + s[ty+2][tx+1] +
                      s[ty+1][tx] + s[ty+1][tx+2] - 4.f * s[ty+1][tx+1];
    u_new[i * Nx + j] = s[ty+1][tx+1] + r * laplacian;
}
```

---

## 5. 경계 조건

### Dirichlet (고정 값)

경계 값이 고정 상수입니다 (예: 가열된 벽의 온도):

```c
// Dirichlet BC 적용: 경계를 상수 값으로 설정
__global__ void apply_dirichlet(float *u, int Nx, int Ny,
                                 float top, float bottom,
                                 float left, float right) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // 위쪽 및 아래쪽 행
    if (idx < Nx) {
        u[0 * Nx + idx]        = top;     // 행 0
        u[(Ny-1) * Nx + idx]   = bottom;  // 행 Ny-1
    }
    // 왼쪽 및 오른쪽 열
    if (idx < Ny) {
        u[idx * Nx + 0]        = left;    // 열 0
        u[idx * Nx + (Nx - 1)] = right;   // 열 Nx-1
    }
}
```

### Neumann (고정 플럭스 / 단열)

제로 플럭스 (단열) 경계: ∂u/∂n = 0. 고스트 셀 (미러)로 구현:

```c
// 위쪽 행의 Neumann BC: u[-1][j] = u[1][j] (미러)
// 각 업데이트 단계 후 적용
__global__ void apply_neumann_top(float *u, int Nx) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < Nx) u[0 * Nx + j] = u[1 * Nx + j];   // 고스트 = 미러
}

__global__ void apply_neumann_bottom(float *u, int Nx, int Ny) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < Nx) u[(Ny-1) * Nx + j] = u[(Ny-2) * Nx + j];
}
```

---

## 6. 수렴 측정

정상 상태 문제의 경우, 단계당 변화의 L2-norm을 확인합니다:

```c
// L2 잔차: ||u_new - u_old||_2
__global__ void l2_residual(
    const float *u_new, const float *u_old,
    float *partial_sq, int N)
{
    extern __shared__ float sdata[];
    int i   = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    float diff = (i < N) ? (u_new[i] - u_old[i]) : 0.f;
    sdata[tid] = diff * diff;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) sdata[tid] += sdata[tid + stride];
        __syncthreads();
    }
    if (tid == 0) partial_sq[blockIdx.x] = sdata[0];
}

// 전체 수렴 루프
void run_to_convergence(float *d_u0, float *d_u1, int Nx, int Ny,
                        float r, float tol) {
    dim3 block(16, 16);
    dim3 grid((Nx + 15) / 16, (Ny + 15) / 16);
    int N = Nx * Ny;
    const int BLOCK = 256;
    int n_blocks = (N + BLOCK - 1) / BLOCK;

    float *d_partial; cudaMalloc(&d_partial, n_blocks * sizeof(float));

    for (int step = 0; step < 100000; step++) {
        heat_eq_step<<<grid, block>>>(d_u0, d_u1, Nx, Ny, r);

        if (step % 100 == 0) {
            l2_residual<<<n_blocks, BLOCK, BLOCK * sizeof(float)>>>(
                d_u1, d_u0, d_partial, N);
            // d_partial를 reduce (CUB 사용) 후 제곱근
            float res = sqrtf(cub_reduce_sum(d_partial, n_blocks));
            printf("단계 %d: L2 잔차 = %.2e\n", step, res);
            if (res < tol) { printf("수렴!\n"); break; }
        }

        float *tmp = d_u0; d_u0 = d_u1; d_u1 = tmp;  // ping-pong
    }
    cudaFree(d_partial);
}
```

---

## 7. 완전한 시뮬레이션 예시

```c
int main() {
    const int Nx = 512, Ny = 512;
    const float alpha = 0.1f;
    const float h = 1.0f / (Nx - 1);           // 격자 간격
    const float dt = 0.24f * h * h / alpha;    // r = 0.24 < 0.25 (안전)
    const float r  = alpha * dt / (h * h);

    check_cfl(alpha, dt, h);

    size_t bytes = Nx * Ny * sizeof(float);
    float *d_u0, *d_u1;
    cudaMalloc(&d_u0, bytes);
    cudaMalloc(&d_u1, bytes);
    cudaMemset(d_u0, 0, bytes);  // 초기 온도 = 0

    // Dirichlet BC: 위쪽 벽 T=1, 나머지 T=0
    apply_dirichlet<<<(max(Nx,Ny)+255)/256, 256>>>(d_u0, Nx, Ny, 1.f, 0.f, 0.f, 0.f);
    cudaMemcpy(d_u1, d_u0, bytes, cudaMemcpyDeviceToDevice);

    dim3 block(16, 16), grid((Nx+15)/16, (Ny+15)/16);

    const int STEPS = 10000;
    for (int t = 0; t < STEPS; t++) {
        heat_eq_step<<<grid, block>>>(d_u0, d_u1, Nx, Ny, r);
        // Dirichlet BC 재적용 (stencil kernel이 경계를 덮어쓸 수 있음)
        apply_dirichlet<<<(max(Nx,Ny)+255)/256, 256>>>(d_u1, Nx, Ny, 1.f, 0.f, 0.f, 0.f);
        float *tmp = d_u0; d_u0 = d_u1; d_u1 = tmp;
    }

    // 다운로드 및 저장
    std::vector<float> h_u(Nx * Ny);
    cudaMemcpy(h_u.data(), d_u0, bytes, cudaMemcpyDeviceToHost);
    save_pgm("heat.pgm", h_u.data(), Nx, Ny);

    cudaFree(d_u0); cudaFree(d_u1);
}
```

**예상 결과**: 10,000 단계 후 온도 장은 u=1 (위쪽)에서 u=0 (아래쪽)으로 기울기를 보이며, 정상 상태 선형 프로파일 u(y) = 1 - y (단위 정사각형 영역)로 수렴합니다.

---

## 핵심 요약

- 2D 열 방정식 ∂u/∂t = α∇²u는 5-point stencil 업데이트로 이산화됩니다: `u_new = u_old + r * (이웃 - 4*중심)`
- **CFL 조건**: `r = α·Δt/h² ≤ 0.25` — 위반 시 지수적 성장 (수치 불안정)
- **Ping-pong 버퍼**: stencil에서 읽기-쓰기 충돌을 피하기 위해 `u_old`와 `u_new` 사이를 교대
- **Dirichlet BC**: 각 단계 후 경계 값 고정; **Neumann BC**: 내부 행/열을 경계에 복사 (고스트 셀 방식)
- 정상 상태 수렴은 `u_new - u_old`의 L2-norm으로 측정; 허용 한계 아래로 내려가면 종료
- 명시적 방식은 코딩이 간단하지만 CFL에 제약됩니다; 암시적 방식 (Crank-Nicolson)은 더 큰 Δt를 허용하지만 단계당 선형 시스템을 풀어야 합니다

---

**다음**: [24. Fluid Dynamics LBM](./24_Fluid_Dynamics_LBM.md) — D2Q9 격자에서 격자 볼츠만 방법으로 비압축성 유동을 시뮬레이션하고, streaming, collision, bounce-back 경계 조건을 구현합니다.
