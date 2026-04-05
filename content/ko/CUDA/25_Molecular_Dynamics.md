# 25. 분자 동역학 (Molecular Dynamics)

**이전**: [Fluid Dynamics LBM](./24_Fluid_Dynamics_LBM.md) | **다음**: [Image Processing GPU](./26_Image_Processing_GPU.md)

---

## 학습 목표

이 레슨을 마치면 다음을 할 수 있습니다:

1. Lennard-Jones 퍼텐셜 평가 및 원자 간 쌍별 힘 계산하기
2. 이웃 목록 (Verlet 목록)을 구현하여 힘 계산을 O(N²)에서 O(N·k)로 줄이기
3. 최소 이미지 규칙으로 주기적 경계 조건 적용하기
4. velocity Verlet로 운동 방정식 적분 및 에너지 보존 검증하기
5. NVT 앙상블 시뮬레이션을 위한 속도 재스케일링 온도 조절기 구현하기

---

## 1. Lennard-Jones 퍼텐셜

Lennard-Jones (LJ) 12-6 퍼텐셜은 비활성 기체 원자(아르곤 등) 간의 반 데르 발스 상호작용을 모델링합니다:

```
U(r) = 4ε · [(σ/r)^12 - (σ/r)^6]

r     = 입자 간 거리
ε     = 퍼텐셜 우물 깊이 (에너지 스케일)
σ     = U=0이 되는 거리 (길이 스케일)

r_min = 2^(1/6) σ ≈ 1.122 σ   (평형 거리, U = -ε)
r > r_min: 인력  (r^-6 지배)
r < r_min: 척력  (r^-12 지배 — 매우 가파름)
```

힘 (퍼텐셜의 음의 기울기):

```
F(r) = -dU/dr = 4ε · [12σ^12/r^13 - 6σ^6/r^7] (r̂ 방향)
     = (48ε/r²) · [(σ/r)^12 - 0.5·(σ/r)^6] · r_vec
```

환산 LJ 단위 (ε = σ = m = 1)에서 임계 cutoff는 일반적으로 r_cut = 2.5σ — 이를 넘어서면 LJ 힘은 최솟값의 1% 미만입니다.

---

## 2. 나이브 쌍별 힘 Kernel

```c
// LJ 힘 계산 — O(N²), 원자당 하나의 thread
__global__ void lj_forces_naive(
    const float4 *pos,    // (x, y, z, type)
    float4       *force,  // (fx, fy, fz, potential)
    int N, float L,       // 박스 길이 (정육면체 주기적 박스)
    float r_cut2)         // r_cut² (초기 필터에서 sqrt 회피)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    float4 pi = pos[i];
    float fx = 0.f, fy = 0.f, fz = 0.f, pe = 0.f;

    for (int j = 0; j < N; j++) {
        if (j == i) continue;
        float4 pj = pos[j];

        // 최소 이미지 규칙 (주기적 BC)
        float dx = pj.x - pi.x;
        float dy = pj.y - pi.y;
        float dz = pj.z - pi.z;
        dx -= L * rintf(dx / L);   // 가장 가까운 이미지로 반올림
        dy -= L * rintf(dy / L);
        dz -= L * rintf(dz / L);

        float r2 = dx*dx + dy*dy + dz*dz;
        if (r2 >= r_cut2) continue;

        // 환산 단위의 LJ (ε=σ=1):  F = 48/r² * (1/r^12 - 0.5/r^6)
        float r2i  = 1.0f / r2;
        float r6i  = r2i * r2i * r2i;
        float fscl = 48.0f * r2i * r6i * (r6i - 0.5f);

        fx += fscl * dx;
        fy += fscl * dy;
        fz += fscl * dz;
        pe += 4.0f * r6i * (r6i - 1.0f);   // 퍼텐셜 (절반만 — 뉴턴 3법칙 쌍)
    }
    pe *= 0.5f;  // 각 쌍이 두 번 집계됨

    force[i] = make_float4(fx, fy, fz, pe);
}
```

N=10,000 원자의 경우 단계당 1억 쌍 평가가 필요합니다 — 비용이 높지만 올바릅니다. 아래의 이웃 목록은 이를 ~100배 줄입니다.

---

## 3. Verlet 이웃 목록

이웃 목록은 원자 i에 대해 r < r_cut + r_skin인 모든 쌍 (i, j)를 저장합니다. 스킨 거리 r_skin (일반적으로 0.3σ)은 목록이 재구성이 필요하기 전까지 여러 시간 단계 동안 유효함을 의미합니다:

```c
// GPU에서 이웃 목록 구성
// d_neighbors[i * MAX_NBRS + k] = j  (원자 i의 k번째 이웃)
// d_num_nbrs[i]                 = 원자 i의 이웃 수
__global__ void build_neighbor_list(
    const float4 *pos,
    int *d_neighbors, int *d_num_nbrs,
    int N, float L, float r_list2,  // (r_cut + r_skin)²
    int MAX_NBRS)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    float4 pi = pos[i];
    int count = 0;

    for (int j = 0; j < N; j++) {
        if (j == i) continue;
        float4 pj = pos[j];
        float dx = pj.x - pi.x; dx -= L * rintf(dx / L);
        float dy = pj.y - pi.y; dy -= L * rintf(dy / L);
        float dz = pj.z - pi.z; dz -= L * rintf(dz / L);
        float r2 = dx*dx + dy*dy + dz*dz;

        if (r2 < r_list2 && count < MAX_NBRS)
            d_neighbors[i * MAX_NBRS + count++] = j;
    }
    d_num_nbrs[i] = count;
}

// 이웃 목록을 사용한 힘 계산 — O(N * avg_neighbors)
__global__ void lj_forces_nblist(
    const float4 *pos,
    const int *d_neighbors, const int *d_num_nbrs,
    float4 *force, int N, float L, float r_cut2, int MAX_NBRS)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    float4 pi = pos[i];
    float fx = 0.f, fy = 0.f, fz = 0.f, pe = 0.f;
    int nnbr = d_num_nbrs[i];

    for (int k = 0; k < nnbr; k++) {
        int j = d_neighbors[i * MAX_NBRS + k];
        float4 pj = pos[j];
        float dx = pj.x - pi.x; dx -= L * rintf(dx / L);
        float dy = pj.y - pi.y; dy -= L * rintf(dy / L);
        float dz = pj.z - pi.z; dz -= L * rintf(dz / L);
        float r2 = dx*dx + dy*dy + dz*dz;
        if (r2 >= r_cut2) continue;

        float r2i  = 1.0f / r2;
        float r6i  = r2i * r2i * r2i;
        float fscl = 48.0f * r2i * r6i * (r6i - 0.5f);
        fx += fscl * dx; fy += fscl * dy; fz += fscl * dz;
        pe += 2.0f * r6i * (r6i - 1.0f);
    }
    force[i] = make_float4(fx, fy, fz, pe);
}
```

**재구성 빈도**: 마지막 재구성 이후 어떤 원자가 r_skin/2 이상 이동했을 때 재구성합니다. 변위 kernel + reduction으로 확인합니다.

---

## 4. Velocity Verlet 적분

```c
// 단계 1: 반-단계 속도 업데이트 + 전체 위치 업데이트
__global__ void verlet_step1(float4 *pos, float4 *vel, const float4 *force,
                              int N, float dt, float L) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    float4 p = pos[i], v = vel[i], f = force[i];
    float dt2 = 0.5f * dt;

    // 반-단계 속도 (LJ 단위에서 질량=1 가정)
    v.x += dt2 * f.x;  v.y += dt2 * f.y;  v.z += dt2 * f.z;
    // 전체-단계 위치
    p.x += dt * v.x;   p.y += dt * v.y;   p.z += dt * v.z;

    // 주기적 박스 랩핑
    p.x -= L * floorf(p.x / L);
    p.y -= L * floorf(p.y / L);
    p.z -= L * floorf(p.z / L);

    pos[i] = p; vel[i] = v;
}

// 단계 2: 새 힘으로 속도 업데이트 완료
__global__ void verlet_step2(float4 *vel, const float4 *force, int N, float dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float4 v = vel[i], f = force[i];
    float dt2 = 0.5f * dt;
    v.x += dt2 * f.x;  v.y += dt2 * f.y;  v.z += dt2 * f.z;
    vel[i] = v;
}
```

---

## 5. 에너지 보존 및 NVT 온도 조절기

**에너지 모니터링** (NVE 앙상블 확인):

```c
// 운동 에너지: KE = 0.5 * Σ m*v²  (LJ 단위에서 m=1)
__global__ void kinetic_energy(const float4 *vel, float *ke_partial, int N) {
    extern __shared__ float sdata[];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float4 v = (i < N) ? vel[i] : make_float4(0,0,0,0);
    sdata[threadIdx.x] = 0.5f * (v.x*v.x + v.y*v.y + v.z*v.z);
    __syncthreads();
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) ke_partial[blockIdx.x] = sdata[0];
}

// NVT: 속도 재스케일링 온도 조절기
// KE = 목표_KE = 1.5 * N * k_B * T (LJ 단위에서 k_B=1)가 되도록 속도 스케일링
__global__ void rescale_velocities(float4 *vel, float scale, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    vel[i].x *= scale;
    vel[i].y *= scale;
    vel[i].z *= scale;
}

void apply_thermostat(float4 *d_vel, int N, float T_target) {
    float ke = compute_kinetic_energy(d_vel, N);  // reduction 사용
    float T_current = 2.0f * ke / (3.0f * N);     // LJ 단위에서 k_B=1
    float scale = sqrtf(T_target / T_current);
    rescale_velocities<<<(N+255)/256, 256>>>(d_vel, scale, N);
}
```

---

## 6. 완전한 MD 루프

```c
void run_md(int N, int steps, float dt, float T, float L) {
    const int BLOCK = 256;
    const float r_cut = 2.5f, r_skin = 0.3f;
    const float r_cut2  = r_cut * r_cut;
    const float r_list2 = (r_cut + r_skin) * (r_cut + r_skin);
    const int   MAX_NBRS = 200;

    float4 *d_pos, *d_vel, *d_force;
    int    *d_nbr, *d_nnbr;
    cudaMalloc(&d_pos,  N * sizeof(float4));
    cudaMalloc(&d_vel,  N * sizeof(float4));
    cudaMalloc(&d_force, N * sizeof(float4));
    cudaMalloc(&d_nbr,  N * MAX_NBRS * sizeof(int));
    cudaMalloc(&d_nnbr, N * sizeof(int));

    // FCC 격자 위치 및 Maxwell-Boltzmann 속도 초기화
    init_fcc_lattice(d_pos, d_vel, N, L, T);

    // 초기 이웃 목록
    build_neighbor_list<<<(N+BLOCK-1)/BLOCK, BLOCK>>>(
        d_pos, d_nbr, d_nnbr, N, L, r_list2, MAX_NBRS);
    lj_forces_nblist<<<(N+BLOCK-1)/BLOCK, BLOCK>>>(
        d_pos, d_nbr, d_nnbr, d_force, N, L, r_cut2, MAX_NBRS);

    for (int t = 0; t < steps; t++) {
        // Verlet 단계 1: v += 0.5*dt*f, x += dt*v
        verlet_step1<<<(N+BLOCK-1)/BLOCK, BLOCK>>>(d_pos, d_vel, d_force, N, dt, L);

        // 필요 시 이웃 목록 재구성 (최대 변위 확인)
        if (need_rebuild(d_pos, d_pos_ref, N, r_skin))
            build_neighbor_list<<<(N+BLOCK-1)/BLOCK, BLOCK>>>(
                d_pos, d_nbr, d_nnbr, N, L, r_list2, MAX_NBRS);

        // 힘 재계산
        lj_forces_nblist<<<(N+BLOCK-1)/BLOCK, BLOCK>>>(
            d_pos, d_nbr, d_nnbr, d_force, N, L, r_cut2, MAX_NBRS);

        // Verlet 단계 2: v += 0.5*dt*f_new
        verlet_step2<<<(N+BLOCK-1)/BLOCK, BLOCK>>>(d_vel, d_force, N, dt);

        // NVT 온도 조절기 (10 단계마다 적용)
        if (t % 10 == 0) apply_thermostat(d_vel, N, T);

        // 진단 출력
        if (t % 100 == 0) {
            float ke = compute_kinetic_energy(d_vel, N);
            float pe = compute_potential_energy(d_force, N);
            printf("단계 %d: KE=%.4f PE=%.4f E_tot=%.4f T=%.4f\n",
                   t, ke, pe, ke+pe, 2.f*ke/(3.f*N));
        }
    }

    cudaFree(d_pos); cudaFree(d_vel); cudaFree(d_force);
    cudaFree(d_nbr); cudaFree(d_nnbr);
}
```

---

## 핵심 요약

- **Lennard-Jones 퍼텐셜**은 단거리 척력 (r^-12)과 반 데르 발스 인력 (r^-6)을 모델링합니다; r_cut = 2.5σ에서 cutoff하면 <1% 에너지 오차로 계산을 줄입니다
- **최소 이미지 규칙**: 주기적 BC의 경우, 각 쌍 거리를 가장 가까운 박스 이미지로 이동합니다 (L × round(Δr/L) 빼기)
- **Verlet 이웃 목록**은 O(N²) 힘 계산을 O(N·k)로 줄입니다 (k ≈ cutoff 구 내의 평균 이웃); 스킨 r_skin을 사용하여 ~20 단계마다 재구성
- **Velocity Verlet**은 MD 적분기의 선택입니다: 시간 가역적, 2차 정확도, 장기 실행에서 에너지 보존
- 에너지 보존 (안정적인 총 에너지)은 NVE 앙상블의 핵심 정확성 검사; 드리프트는 너무 큰 시간 단계 또는 버그를 나타냅니다
- **속도 재스케일링** (NVT 온도 조절기)은 즉각적으로 온도를 설정하지만 엄격한 NVT가 아닙니다; 프로덕션에서는 Nosé-Hoover 온도 조절기가 선호됩니다

---

**다음**: [26. Image Processing GPU](./26_Image_Processing_GPU.md) — GPU kernel과 texture memory를 사용하여 Gaussian blur, Sobel 에지 검출, bilateral 필터링, histogram 균등화를 적용합니다.
