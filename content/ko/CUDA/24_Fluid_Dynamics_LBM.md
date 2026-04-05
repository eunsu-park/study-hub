# 24. 유체 역학 — 격자 볼츠만 방법 (Lattice Boltzmann Method)

**이전**: [PDE Solvers Heat Equation](./23_PDE_Solvers_Heat_Equation.md) | **다음**: [Molecular Dynamics](./25_Molecular_Dynamics.md)

---

## 학습 목표

이 레슨을 마치면 다음을 할 수 있습니다:

1. D2Q9 속도 모델과 분포 함수의 물리적 의미 설명하기
2. BGK collision 단계 (평형으로의 완화) 구현하기
3. 분포 함수를 이동시키는 streaming 단계 구현하기
4. 고체 벽에 대한 no-slip bounce-back 경계 조건 적용하기
5. 분포 함수에서 거시적 밀도 및 속도를 복원하고 뚜껑-구동 공동(lid-driven cavity) 벤치마크로 검증하기

---

## 1. 격자 볼츠만 방법 개요

격자 볼츠만 방법(LBM)은 Navier-Stokes 방정식을 직접 풀지 않고, 정규 격자에서 중간규모 **입자 분포 함수** f_i의 진화를 추적하여 유체 역학을 시뮬레이션합니다.

각 격자 노드는 Q개의 분포 값을 가집니다(속도 방향당 하나). 각 시간 단계:
1. **Collision**: 분포를 국소 평형으로 완화
2. **Streaming**: 속도 방향을 따라 분포 이동

LBM은 GPU에 이상적입니다:
- 모든 collision 단계는 **국소적** (이웃 필요 없음)
- Streaming은 **규칙적인 이동 패턴** (구조화된 메모리 접근)
- 알고리즘은 격자 노드당 완전 병렬

---

## 2. D2Q9 속도 모델

D2Q9 모델은 2D 격자에서 9개의 이산 속도를 사용합니다:

```
속도 인덱스 및 방향:
  6  2  5
  3  0  1      e_i = 속도 방향 i
  7  4  8

e_0 = ( 0,  0)    가중치 w_0 = 4/9
e_1 = ( 1,  0)    가중치 w_1 = 1/9
e_2 = ( 0,  1)    가중치 w_2 = 1/9
e_3 = (-1,  0)    가중치 w_3 = 1/9
e_4 = ( 0, -1)    가중치 w_4 = 1/9
e_5 = ( 1,  1)    가중치 w_5 = 1/36
e_6 = (-1,  1)    가중치 w_6 = 1/36
e_7 = (-1, -1)    가중치 w_7 = 1/36
e_8 = ( 1, -1)    가중치 w_8 = 1/36
```

```c
// D2Q9 상수
__constant__ int ex[9] = { 0,  1,  0, -1,  0,  1, -1, -1,  1};
__constant__ int ey[9] = { 0,  0,  1,  0, -1,  1,  1, -1, -1};
__constant__ float w[9] = {4.f/9, 1.f/9, 1.f/9, 1.f/9, 1.f/9,
                            1.f/36, 1.f/36, 1.f/36, 1.f/36};
// bounce-back을 위한 반대 방향
__constant__ int opp[9] = {0, 3, 4, 1, 2, 7, 8, 5, 6};
```

---

## 3. 거시적 변수

밀도 ρ와 운동량 ρu는 분포 함수의 모멘트입니다:

```
ρ(x, t)    = Σ_i f_i(x, t)               (영차 모멘트 = 밀도)
ρ·u(x, t)  = Σ_i e_i · f_i(x, t)        (1차 모멘트 = 운동량)
u = ρu / ρ                                (속도)
```

```c
// f_i에서 거시적 밀도와 속도 복원
__device__ void macro_vars(const float *f, float *rho, float *ux, float *uy) {
    *rho = 0.f;  *ux = 0.f;  *uy = 0.f;
    for (int q = 0; q < 9; q++) {
        *rho += f[q];
        *ux  += ex[q] * f[q];
        *uy  += ey[q] * f[q];
    }
    *ux /= *rho;
    *uy /= *rho;
}
```

---

## 4. BGK 평형 및 Collision

BGK (Bhatnagar-Gross-Krook) collision은 f_i를 속도 1/τ로 국소 Maxwell-Boltzmann 평형 f_i^eq를 향해 완화합니다:

```
f_i^eq = w_i · ρ · [1 + (e_i·u)/c_s² + (e_i·u)²/(2c_s⁴) - u²/(2c_s²)]

여기서 c_s² = 1/3  (LB 단위에서 격자 음속의 제곱)

BGK collision:
f_i*(x,t) = f_i(x,t) - (1/τ) · [f_i(x,t) - f_i^eq(x,t)]

τ = 완화 시간, 동점 점성도와 관련: ν = c_s²(τ - 0.5)
레이놀즈 수: Re = U·L / ν = U·L / [c_s²(τ - 0.5)]
```

```c
// 평형 분포 계산
__device__ float f_eq(int q, float rho, float ux, float uy) {
    float eu  = ex[q] * ux + ey[q] * uy;     // e_i · u
    float u2  = ux * ux + uy * uy;            // |u|²
    // c_s² = 1/3이므로 1/c_s² = 3, 1/(2c_s²) = 3/2, 1/(2c_s⁴) = 9/2
    return w[q] * rho * (1.f + 3.f*eu + 4.5f*eu*eu - 1.5f*u2);
}

// 결합 collision kernel (f 배열에서 in-place)
__global__ void collide(float *f, const bool *solid, int Nx, int Ny, float tau_inv) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= Nx || y >= Ny || solid[y * Nx + x]) return;

    int node = y * Nx + x;
    float fi[9];
    for (int q = 0; q < 9; q++) fi[q] = f[node * 9 + q];

    float rho, ux, uy;
    macro_vars(fi, &rho, &ux, &uy);

    for (int q = 0; q < 9; q++) {
        float feq = f_eq(q, rho, ux, uy);
        f[node * 9 + q] = fi[q] - tau_inv * (fi[q] - feq);
    }
}
```

---

## 5. Streaming 단계

Collision 후, 각 분포 함수 f_i가 방향 e_i로 이웃 노드로 이동합니다:

```
f_i(x + e_i, t+1) ← f_i*(x, t)    (streaming)
```

```c
// Streaming: pull 방식 — 각 노드가 상류 이웃에서 읽음
// (이중 버퍼링 없이 경쟁 조건 방지)
__global__ void stream(const float *f_in, float *f_out,
                       const bool *solid, int Nx, int Ny) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= Nx || y >= Ny) return;

    int node = y * Nx + x;

    for (int q = 0; q < 9; q++) {
        // 소스 노드: 방향 q가 온 곳
        int xs = (x - ex[q] + Nx) % Nx;   // 개방 경계를 위한 주기적 랩
        int ys = (y - ey[q] + Ny) % Ny;
        int src = ys * Nx + xs;

        if (solid[src]) {
            // Bounce-back: 고체 노드에서 방향 역전
            f_out[node * 9 + q] = f_in[node * 9 + opp[q]];
        } else {
            f_out[node * 9 + q] = f_in[src * 9 + q];
        }
    }
}
```

**Pull 방식**은 이웃에 밀어내는 대신 이웃에서 읽습니다 — 경쟁 조건이 없고 GPU에 친화적입니다.

---

## 6. Bounce-Back 경계 조건 (No-Slip)

Bounce-back 규칙은 고체 노드에서 들어오는 분포를 역전시켜 no-slip 조건 (벽에서 속도 = 0)을 발생시킵니다:

```
고체 경계에서: f_i(wall, t+1) = f_{opp(i)}(wall, t)

여기서 opp(i)는 i의 반대 방향:
  opp(1) = 3   (오른쪽 ↔ 왼쪽)
  opp(2) = 4   (위 ↔ 아래)
  opp(5) = 7   (오른쪽-위 ↔ 왼쪽-아래)
  등등
```

이동하는 벽 (예: 뚜껑-구동 공동)에는 운동량 보정 추가:

```c
// 이동하는 뚜껑 (y=Ny-1의 상단 벽이 x 방향으로 u_lid 속도로 이동)
__global__ void moving_lid_bc(float *f, int Nx, int Ny, float u_lid) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= Nx) return;
    int y = Ny - 1;
    int node = y * Nx + x;

    float rho_lid = 0.f;
    // 알려진 분포에서 밀도 추정
    rho_lid = f[node*9+0] + f[node*9+1] + f[node*9+3]
            + 2.f*(f[node*9+2] + f[node*9+5] + f[node*9+6]);
    rho_lid /= (1.f + 1.5f * u_lid);  // Zou-He 속도 BC에서

    // f_4, f_7, f_8에 Zou-He 속도 경계 조건 적용
    f[node*9+4] = f[node*9+2] - (2.f/3.f) * rho_lid * 0.f;       // vy=0
    f[node*9+7] = f[node*9+5] - 0.5f*(f[node*9+1]-f[node*9+3])
                              - (1.f/6.f)*rho_lid*u_lid;
    f[node*9+8] = f[node*9+6] + 0.5f*(f[node*9+1]-f[node*9+3])
                              + (1.f/6.f)*rho_lid*u_lid;
}
```

---

## 7. 메인 LBM 루프 및 레이놀즈 수

```c
void run_lbm(int Nx, int Ny, int steps, float tau, float u_lid) {
    float tau_inv = 1.0f / tau;
    float nu = (1.0f/3.0f) * (tau - 0.5f);  // 동점 점성도
    float Re = u_lid * Ny / nu;
    printf("Re = %.1f, tau = %.3f, nu = %.5f\n", Re, tau, nu);

    // 할당: Nx * Ny 노드 × 9개 분포
    size_t bytes = Nx * Ny * 9 * sizeof(float);
    float *d_f0, *d_f1;
    bool  *d_solid;
    cudaMalloc(&d_f0,   bytes);
    cudaMalloc(&d_f1,   bytes);
    cudaMalloc(&d_solid, Nx * Ny * sizeof(bool));

    // 초기화: f = f_eq(rho=1, ux=0, uy=0) 모든 곳에서
    init_equilibrium<<<dim3((Nx+15)/16,(Ny+15)/16), dim3(16,16)>>>(d_f0, Nx, Ny);

    // 고체 노드 표시 (벽: y=0, y=Ny-1, x=0, x=Nx-1)
    mark_solid_walls<<<(Nx*Ny+255)/256, 256>>>(d_solid, Nx, Ny);

    dim3 block(16, 16), grid((Nx+15)/16, (Ny+15)/16);

    for (int t = 0; t < steps; t++) {
        collide<<<grid, block>>>(d_f0, d_solid, Nx, Ny, tau_inv);
        moving_lid_bc<<<(Nx+255)/256, 256>>>(d_f0, Nx, Ny, u_lid);
        stream<<<grid, block>>>(d_f0, d_f1, d_solid, Nx, Ny);

        // 버퍼 교환
        float *tmp = d_f0; d_f0 = d_f1; d_f1 = tmp;
    }

    // 시각화를 위한 속도 장 추출
    extract_velocity<<<grid, block>>>(d_f0, d_ux, d_uy, d_solid, Nx, Ny);

    cudaFree(d_f0); cudaFree(d_f1); cudaFree(d_solid);
}
```

**벤치마크 목표**: RTX 3090에서 1024×1024 D2Q9 LBM 시뮬레이션은 초당 ~3 × 10⁹ 노드-업데이트 (3 GNUPS)를 달성합니다.

---

## 핵심 요약

- **D2Q9 LBM**은 노드당 9개의 속도 방향을 사용합니다; 각 노드는 9개의 분포 값 f_i를 저장합니다
- **Collision** (BGK): f_i를 속도 1/τ로 평형 f_i^eq를 향해 완화; τ는 점성도와 레이놀즈 수를 제어합니다
- **Streaming** (pull 방식): 각 노드가 상류 이웃에서 읽음 — 자연스럽게 경쟁 조건 없음
- **Bounce-back**은 고체 벽에서 들어오는 분포를 역전시켜 no-slip 조건 생성
- **이동하는 벽** (뚜껑-구동 공동): Zou-He 속도 BC가 벽 속도를 올바르게 지정
- 레이놀즈 수 Re = U·L / ν, ν = c_s²(τ - 0.5); 단순 BGK로 안정적인 층류를 위해 Re < ~1000 유지

---

**다음**: [25. Molecular Dynamics](./25_Molecular_Dynamics.md) — Lennard-Jones 분자 동역학 시뮬레이션을 이웃 목록, velocity Verlet 적분, 주기적 경계 조건으로 구현합니다.
