# 20. N-Body 시뮬레이션

**이전**: [Sparse Matrix Ops](./19_Sparse_Matrix_Ops.md) | **다음**: [Monte Carlo Methods](./21_Monte_Carlo_Methods.md)

---

## 학습 목표

이 레슨을 마치면 다음을 할 수 있습니다:

1. GPU에서 O(N²) 직접 중력 합산 구현하기
2. tile 기반 shared memory 최적화로 전역 메모리 로드를 N²에서 N²/TILE_SIZE로 줄이기
3. velocity Verlet 방식으로 입자 위치 적분하기
4. 소프트닝 매개변수와 물리적 필요성 이해하기
5. Barnes-Hut O(N log N) 근사 및 실시간 시각화를 위한 OpenGL interop 설명하기

---

## 1. N-Body 문제

입자 i에 입자 j로부터 가해지는 중력:

```
F_ij = G * m_i * m_j / (r_ij² + ε²)^(3/2) * r_ij_vec

여기서 r_ij_vec = (x_j - x_i, y_j - y_i, z_j - z_i)
      r_ij²    = dot(r_ij_vec, r_ij_vec)
      ε        = 소프트닝 매개변수 (r→0일 때 0으로 나누기 방지)
```

각 입자 i는 N-1개의 다른 입자로부터의 기여를 모두 합산해야 합니다 — 총 O(N²) 작업. N=10,000 입자의 경우 시간 단계당 1억 번의 힘 평가가 필요합니다. GPU는 이에 탁월합니다: 계산이 완전 병렬이며 계산 집약적입니다.

```
N=10,000 입자 × 10,000 상호작용 × 20 FLOP/상호작용 = 2 GFLOP/단계
RTX 3090 FP32 피크 = 35.6 TFLOPS → 잠재적 속도향상 ~10,000× vs 단일 CPU 코어
```

---

## 2. 나이브 N-Body Kernel

```c
// 입자 상태
struct float4;  // x, y, z, 질량

// 모든 입자의 가속도 계산 (입자당 하나의 thread)
__global__ void nbody_naive(
    const float4 *pos,   // [N] (x, y, z, 질량)
    float4       *acc,   // [N] (ax, ay, az, 미사용)
    int N, float G, float eps2)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    float4 pi = pos[i];
    float ax = 0.f, ay = 0.f, az = 0.f;

    for (int j = 0; j < N; j++) {
        float4 pj = pos[j];           // 모든 j에 대해 전역 메모리 로드
        float dx = pj.x - pi.x;
        float dy = pj.y - pi.y;
        float dz = pj.z - pi.z;
        float dist2 = dx*dx + dy*dy + dz*dz + eps2;
        float inv_dist3 = G * pj.w * rsqrtf(dist2) / dist2;  // G*m_j / r^3
        ax += dx * inv_dist3;
        ay += dy * inv_dist3;
        az += dz * inv_dist3;
    }

    acc[i] = make_float4(ax, ay, az, 0.f);
}
```

**병목**: 각 thread i는 모든 j에 대해 전역 메모리에서 `pos[j]`를 로드합니다. thread당 N번 로드 × N개 thread = N² 전역 로드 = N=10,000일 때 4억 번 로드. float4당 4바이트: 6.4GB. 900GB/s 대역폭에서: 단계당 7ms — 대부분 메모리 바운드.

---

## 3. Tile 기반 Shared Memory 최적화

TILE_SIZE개 입자를 shared memory에 로드합니다; block의 모든 thread가 그 TILE_SIZE번 로드를 재사용합니다:

```c
#define TILE 256

__global__ void nbody_tiled(
    const float4 *pos, float4 *acc,
    int N, float G, float eps2)
{
    __shared__ float4 sh_pos[TILE];

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float4 pi = (i < N) ? pos[i] : make_float4(0,0,0,0);

    float ax = 0.f, ay = 0.f, az = 0.f;

    // 입력 입자를 TILE_SIZE 크기의 tile로 처리
    for (int tile = 0; tile < (N + TILE - 1) / TILE; tile++) {
        // 하나의 tile 입자를 협력하여 shared memory에 로드
        int j = tile * TILE + threadIdx.x;
        sh_pos[threadIdx.x] = (j < N) ? pos[j] : make_float4(0,0,0,0);
        __syncthreads();

        // 각 thread i가 sh_pos의 모든 TILE개 입자로부터 힘을 누적
        // (ILP를 위해 8× 언롤)
        #pragma unroll 8
        for (int k = 0; k < TILE; k++) {
            float dx = sh_pos[k].x - pi.x;
            float dy = sh_pos[k].y - pi.y;
            float dz = sh_pos[k].z - pi.z;
            float dist2 = dx*dx + dy*dy + dz*dz + eps2;
            float inv_dist3 = G * sh_pos[k].w * rsqrtf(dist2) / dist2;
            ax += dx * inv_dist3;
            ay += dy * inv_dist3;
            az += dz * inv_dist3;
        }
        __syncthreads();
    }

    if (i < N) acc[i] = make_float4(ax, ay, az, 0.f);
}
```

**메모리 트래픽 감소**: TILE=256일 때, 256개 thread의 block은 256개 입자 한 tile(256 × 16바이트 = 4KB)을 로드하고 256번 재사용합니다. 전역 로드: N/TILE tile × N thread × tile당 1 로드 = N²/TILE — TILE배 감소.

```
TILE=256, N=10,000:
  나이브:  N² = 1억 번 전역 로드
  Tile:   N²/TILE = 39만 번 전역 로드  → 256배 더 적은 전역 메모리 바이트
```

Tile 기반 kernel은 메모리 바운드가 아니라 **계산 바운드**가 됩니다. 내부 루프는 입자 쌍당 ~20 FLOP이며 피크 FP32 처리량에 근접하여 실행됩니다.

---

## 4. Velocity Verlet 적분

Velocity Verlet 적분기는 2차 정확도이며 시간 가역적입니다 — N-body 물리학에 이상적입니다:

```
x(t + Δt) = x(t) + v(t)·Δt + 0.5·a(t)·Δt²
a(t + Δt) = F(x(t + Δt)) / m           (새 위치에서 힘 재계산)
v(t + Δt) = v(t) + 0.5·(a(t) + a(t+Δt))·Δt
```

```c
// Verlet 적분 단계 (입자당 하나의 thread)
__global__ void integrate_verlet(
    float4 *pos,    // (x, y, z, 질량)
    float4 *vel,    // (vx, vy, vz, 0)
    float4 *acc_old, // a(t)
    float4 *acc_new, // a(t+Δt) — 이미 계산됨
    int N, float dt)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    float4 p  = pos[i];
    float4 v  = vel[i];
    float4 a0 = acc_old[i];
    float4 a1 = acc_new[i];

    // 위치 업데이트 (이전 가속도 사용)
    p.x += v.x * dt + 0.5f * a0.x * dt * dt;
    p.y += v.y * dt + 0.5f * a0.y * dt * dt;
    p.z += v.z * dt + 0.5f * a0.z * dt * dt;

    // 속도 업데이트 (이전 및 새 가속도의 평균)
    v.x += 0.5f * (a0.x + a1.x) * dt;
    v.y += 0.5f * (a0.y + a1.y) * dt;
    v.z += 0.5f * (a0.z + a1.z) * dt;

    pos[i] = p;
    vel[i] = v;
}

// 메인 시뮬레이션 루프
void simulate_nbody(int N, int steps, float dt, float G, float eps) {
    float eps2 = eps * eps;
    const int BLOCK = TILE;

    float4 *d_pos, *d_vel, *d_acc0, *d_acc1;
    cudaMalloc(&d_pos,  N * sizeof(float4));
    cudaMalloc(&d_vel,  N * sizeof(float4));
    cudaMalloc(&d_acc0, N * sizeof(float4));
    cudaMalloc(&d_acc1, N * sizeof(float4));

    // 호스트에서 pos와 vel 초기화 및 업로드
    // ...

    // 초기 힘 계산
    nbody_tiled<<<(N + BLOCK - 1) / BLOCK, BLOCK>>>(d_pos, d_acc0, N, G, eps2);

    for (int t = 0; t < steps; t++) {
        // 1. 위치 업데이트 (acc0 = a(t) 사용)
        // 단순화: 현재 vel과 반-단계 acc로 pos 전진
        // (전체 verlet은 두 개의 반-속도 업데이트로 분할)

        // 2. 새 위치에서 힘 재계산
        nbody_tiled<<<(N + BLOCK - 1) / BLOCK, BLOCK>>>(d_pos, d_acc1, N, G, eps2);

        // 3. 속도 업데이트 완료
        integrate_verlet<<<(N + BLOCK - 1) / BLOCK, BLOCK>>>(
            d_pos, d_vel, d_acc0, d_acc1, N, dt);

        // acc 버퍼 교환
        float4 *tmp = d_acc0; d_acc0 = d_acc1; d_acc1 = tmp;
    }

    cudaFree(d_pos); cudaFree(d_vel); cudaFree(d_acc0); cudaFree(d_acc1);
}
```

---

## 5. 소프트닝 매개변수

소프트닝 없이는 두 인접 입자가 r → 0으로 갈수록 발산하는 힘을 생성하여 수치적 폭발을 일으킵니다. 소프트닝 길이 ε는 최소 유효 거리를 설정합니다:

```
소프트닝 없음:  inv_dist3 = 1 / r^3       (r → 0으로 발산)
소프트닝 있음:  inv_dist3 = 1 / (r² + ε²)^(3/2)   (유한, 최대 ≈ 1/ε³)

전형적 선택: ε ≈ 입자 간 평균 간격의 0.01 ~ 0.1
너무 작으면: 근접 조우 후 입자들이 분산 (수치 불안정)
너무 크면: 근거리에서 힘이 과소평가됨 (비물리적 소프트닝)
```

---

## 6. Barnes-Hut O(N log N) 근사

대형 N (>100K)의 경우, 직접 O(N²)이 너무 느려집니다. **Barnes-Hut**은 옥트리(3D) 또는 쿼드트리(2D)를 구성하고 멀리 있는 입자 클러스터를 단일 슈퍼-입자로 근사합니다:

```
기준 (개각도 θ):
  cluster_size / 클러스터까지의 거리 < θ이면:
      전체 클러스터를 질량 중심의 하나의 입자로 처리
  그렇지 않으면:
      하위 노드로 재귀적으로 내려감

θ = 0.5가 전형적: < 1% 힘 오차로 O(N log N) 복잡도 달성
```

GPU Barnes-Hut는 처음부터 구현하기 복잡합니다(GPU에서의 트리 구성은 비자명함). NVIDIA GPU Gems 3 챕터와 CUDA SDK 샘플이 참조 구현을 제공합니다. 프로덕션에서는 더 높은 차수 정확도로 O(N)을 달성하는 **Fast Multipole Method** (FMM)를 고려하세요.

---

## 7. OpenGL Interop (시각화 개념)

매 프레임마다 CPU로 복사하지 않고 N-body 궤적을 시각화하려면:

```c
// CUDA 버퍼를 OpenGL VBO로도 등록
GLuint vbo;
glGenBuffers(1, &vbo);
glBindBuffer(GL_ARRAY_BUFFER, vbo);
glBufferData(GL_ARRAY_BUFFER, N * sizeof(float4), NULL, GL_DYNAMIC_DRAW);

cudaGraphicsResource_t cuda_vbo;
cudaGraphicsGLRegisterBuffer(&cuda_vbo, vbo, cudaGraphicsMapFlagsWriteDiscard);

// 매 프레임:
cudaGraphicsMapResources(1, &cuda_vbo, 0);
float4 *d_pos_gl;
size_t  bytes;
cudaGraphicsResourceGetMappedPointer((void**)&d_pos_gl, &bytes, cuda_vbo);

// kernel이 OpenGL 버퍼에 직접 씀 — CPU 왕복 없음
nbody_tiled<<<grid, TILE>>>(d_pos_gl, d_acc, N, G, eps2);
integrate<<<grid, TILE>>>(d_pos_gl, d_vel, d_acc, N, dt);

cudaGraphicsUnmapResources(1, &cuda_vbo, 0);
// OpenGL이 d_pos_gl에서 직접 렌더링
glDrawArrays(GL_POINTS, 0, N);
```

이를 통해 N이 최대 ~100만 입자일 때 60fps의 실시간 인터랙티브 시각화가 가능합니다.

---

## 핵심 요약

- 직접 N-body 힘 계산은 O(N²) — 완전 병렬 (입자 간 누적을 제외한 통신 불필요)
- **Tile 기반 shared memory**는 전역 메모리 트래픽을 TILE_SIZE 배(예: 256×) 줄여 메모리 바운드 kernel을 계산 바운드로 전환합니다
- **Velocity Verlet**은 표준 적분기: 2차 정확도, 시간 가역적, 에너지 보존 (단순 Euler와 비교)
- **소프트닝** (ε > 0)은 r → 0에서의 힘 발산을 방지하며 유한 입자 크기에 의해 물리적으로 동기 부여됩니다
- **Barnes-Hut** (O(N log N))은 계층적 트리를 통해 원거리 힘을 근사합니다 — N > 100K 입자에 필요합니다
- **CUDA-OpenGL interop**은 시각화를 위한 CPU-GPU 왕복을 제거합니다 — 입자가 GPU 메모리에서 직접 렌더링됩니다

---

**다음**: [21. Monte Carlo Methods](./21_Monte_Carlo_Methods.md) — cuRAND로 GPU에서 난수를 생성하고 π 추정 및 Black-Scholes 옵션 가격 산정을 위한 병렬 Monte Carlo 시뮬레이션을 구현합니다.
