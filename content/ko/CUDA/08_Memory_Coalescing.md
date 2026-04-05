# 08. 메모리 합치기

**이전**: [원자적 연산](./07_Atomic_Operations.md) | **다음**: [점유율과 실행 구성](./09_Occupancy_and_Launch_Config.md)

---

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 128바이트 트랜잭션 세분성 규칙과 그 의미 설명
2. 합치된 vs 비합치된 접근 패턴을 시각적으로 식별
3. 벤치마크 커널로 스트라이드 패널티 측정
4. 합치기를 위해 구조체 배열(AoS)을 배열 구조체(SoA)로 재구성
5. Nsight Compute로 합치기 효율 확인

---

## 1. 128바이트 트랜잭션 규칙

32개 스레드로 구성된 warp가 전역 메모리에 접근할 때, 하드웨어는 요청을 **128바이트 트랜잭션** (캐시 라인 1개)으로 결합합니다. 필요한 트랜잭션 수는 전적으로 접근 패턴에 달려 있습니다:

```
32 스레드 × 4바이트 (float) = 총 128바이트

최선의 경우 (합치됨): 32개 스레드 모두 연속 주소에 접근
  → warp 전체에 트랜잭션 1개  ✓

완전히 합치된 스트라이드-1 접근:
  스레드 0: 주소 0
  스레드 1: 주소 4
  스레드 2: 주소 8
  ...
  스레드 31: 주소 124
  → 모두 128바이트 캐시 라인 안에 들어감 → 트랜잭션 1개 ✓

스트라이드-2 접근:
  스레드 0: 주소 0
  스레드 1: 주소 8
  ...
  스레드 31: 주소 248
  → 2개의 캐시 라인에 걸침 → 트랜잭션 2개 (50% 효율)

스트라이드-32 (최악):
  스레드 0: 주소 0
  스레드 1: 주소 128
  스레드 2: 주소 256
  ...
  → 스레드마다 다른 캐시 라인 → 트랜잭션 32개 (3% 효율)
```

---

## 2. 시각화: 합치된 vs 스트라이드 접근

```
메모리 레이아웃: [ 0 ][ 1 ][ 2 ][ 3 ]...[ 31 ][ 32 ]...[ 63 ]
                  ←──────────── 캐시 라인 0 ────────────→

합치됨: 스레드 k가 원소 k에 접근
  T0→[0], T1→[1], ..., T31→[31]
  ──────────────────────────────────
  캐시 라인 1개 로드 → 트랜잭션 1개 → 100% 효율

스트라이드-2: 스레드 k가 원소 2k에 접근
  T0→[0], T1→[2], ..., T15→[30] | T16→[32], ..., T31→[62]
  ────────────────────────────────────────────────────────
  2개의 캐시 라인 점유 → 트랜잭션 2개, 하지만 각각에서 16바이트만 사용
  50% 효율 (2× 대역폭 낭비)

스트라이드-32: 스레드 k가 원소 32k에 접근
  T0→[0], T1→[32], ..., T31→[992]
  각 원소가 다른 캐시 라인 → 트랜잭션 32개
  3.1% 효율 (32× 대역폭 낭비)
```

---

## 3. 스트라이드 벤치마크

```c
// benchmark_stride.cu
__global__ void stride_read(const float *data, float *result, int stride, long n) {
    long i = (long)(blockIdx.x * blockDim.x + threadIdx.x) * stride;
    if (i < n) {
        result[blockIdx.x * blockDim.x + threadIdx.x] = data[i];
    }
}

// 각 스트라이드 값에 대해 유효 대역폭 측정:
// stride=1 (합치됨):   ~2000 GB/s  (A100 최대)
// stride=2:            ~1000 GB/s  (warp당 트랜잭션 2개)
// stride=4:            ~ 500 GB/s
// stride=8:            ~ 250 GB/s
// stride=32:           ~  62 GB/s  (warp당 트랜잭션 32개, 최대의 ~3%)
```

대역폭은 **스트라이드에 선형적으로 저하**됩니다 — 스트라이드가 2배가 될 때마다 유효 대역폭이 절반으로 줄어듭니다.

---

## 4. 구조체 배열(AoS) vs 배열 구조체(SoA)

이것이 GPU 코드에서 가장 흔한 합치기 설계 결정입니다.

### AoS 레이아웃 (GPU에 나쁨)

```c
struct Particle {
    float x, y, z;     // 위치
    float vx, vy, vz;  // 속도
    float mass;
};

Particle particles[N];  // AoS: [x0,y0,z0,vx0,vy0,vz0,m0, x1,y1,z1,...]

// x에 접근하는 커널:
float px = particles[tid].x;
// 스레드 0: 주소 = 0   (x0)
// 스레드 1: 주소 = 28  (x1, 스트라이드 = sizeof(Particle) = 28바이트)
// 스레드 2: 주소 = 56  (x2)
// → 스트라이드 = float 7개 → 7× 대역폭 낭비
```

### SoA 레이아웃 (GPU에 좋음)

```c
struct ParticlesSoA {
    float *x, *y, *z;
    float *vx, *vy, *vz;
    float *mass;
};

ParticlesSoA p;  // SoA: [x0,x1,x2,...,xN, y0,y1,...,yN, ...]

// x에 접근하는 커널:
float px = p.x[tid];
// 스레드 0: 주소 = 0  (x0)
// 스레드 1: 주소 = 4  (x1)
// 스레드 2: 주소 = 8  (x2)
// → 스트라이드 = float 1개 → 트랜잭션 1개 → 100% 효율 ✓
```

### 완전한 예시: N-체 힘 계산 (AoS → SoA)

```c
// AoS 버전 (기준)
struct Body { float x, y, z, mass; };

__global__ void force_aos(const Body *bodies, float3 *forces, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float3 f = {0.0f, 0.0f, 0.0f};
    float xi = bodies[i].x;  // 스트라이드 = sizeof(Body) = 16바이트
    float yi = bodies[i].y;
    float zi = bodies[i].z;
    for (int j = 0; j < N; j++) {
        float dx = bodies[j].x - xi;  // 각 j에 대해 스트라이드 접근
        float dy = bodies[j].y - yi;
        float dz = bodies[j].z - zi;
        float r  = sqrtf(dx*dx + dy*dy + dz*dz + 1e-6f);
        float f_mag = bodies[j].mass / (r * r * r);
        f.x += dx * f_mag;
        f.y += dy * f_mag;
        f.z += dz * f_mag;
    }
    forces[i] = f;
}

// SoA 버전 (최적화)
__global__ void force_soa(
    const float *x, const float *y, const float *z, const float *mass,
    float *fx, float *fy, float *fz, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float xi = x[i], yi = y[i], zi = z[i];  // 합치된 읽기 ✓
    float fix = 0, fiy = 0, fiz = 0;
    for (int j = 0; j < N; j++) {
        float dx = x[j] - xi;  // j 루프에서 순차 읽기 (L1 캐시됨)
        float dy = y[j] - yi;
        float dz = z[j] - zi;
        float r  = sqrtf(dx*dx + dy*dy + dz*dz + 1e-6f);
        float f_mag = mass[j] / (r * r * r);
        fix += dx * f_mag;
        fiy += dy * f_mag;
        fiz += dz * f_mag;
    }
    fx[i] = fix; fy[i] = fiy; fz[i] = fiz;  // 합치된 쓰기 ✓
}
```

이런 유형의 커널에서 일반적인 속도 향상: **2–4배**.

---

## 5. 행렬 접근: 행 vs 열

2D 행렬에 접근할 때:

```c
// 행 접근 (행 우선에서 합치됨): 스레드 k가 행 0, 열 k 읽기
float val = matrix[0 * N + tid];   // 연속 → 합치됨 ✓

// 열 접근 (행 우선에서 스트라이드): 스레드 k가 행 k, 열 0 읽기
float val = matrix[tid * N + 0];   // 스트라이드 N → 높은 비합치됨 ✗
```

열 접근의 경우, 해결책은 **처리 전에 전치**하거나 L05의 **타일 전치** 기법을 사용하는 것입니다.

---

## 6. 벡터화 로드: `float4`

단일 명령으로 float 4개를 로드합니다 — 명령 효율 향상:

```c
// 스칼라 로드 (float 4개에 메모리 명령 4개)
float a0 = data[4*i + 0];
float a1 = data[4*i + 1];
float a2 = data[4*i + 2];
float a3 = data[4*i + 3];

// 벡터화 로드 (메모리 명령 1개, 128비트)
float4 v = reinterpret_cast<float4*>(data)[i];
float a0 = v.x, a1 = v.y, a2 = v.z, a3 = v.w;
```

요구사항:
- 데이터가 16바이트 정렬되어야 함 (float: `cudaMalloc`이 256바이트 정렬 보장)
- 총 원소 수가 4의 배수여야 함

메모리 병목 커널에서 명령 오버헤드 감소로 일반적으로 **5–15% 속도 향상** 제공.

---

## 7. Nsight Compute로 합치기 프로파일링

```bash
ncu --metrics \
    l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum,\
    l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,\
    l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio \
    ./my_kernel

# 핵심 메트릭:
# sectors_per_request = 1.0 → 완벽한 합치기 (warp당 캐시 라인 1개)
# sectors_per_request = 32  → 완전히 비합치됨 (warp당 캐시 라인 32개)
# sectors_per_request = 4   → 4× 대역폭 낭비
```

`l1tex__average_t_sectors_per_request` 비율이 가장 직접적인 측정값입니다: **1.0이 최적, 32가 최악**.

---

## 8. 합치기 심 역할의 공유 메모리

전역 메모리에서 접근 패턴을 합치게 만들 수 없을 때 (예: 열 방향으로 행렬 접근), 공유 메모리를 중간 단계로 사용합니다:

```c
// 공유 메모리에 합치되게 로드한 다음 어떤 패턴으로든 읽기
__shared__ float smem[256];

// 합치된 전역 → 공유 메모리 로드
smem[threadIdx.x] = global_data[coalesced_index];
__syncthreads();

// 공유 메모리에서 비합치된 패턴 접근 (빠름 — DRAM 없음)
float val = smem[shuffled_index(threadIdx.x)];
```

이것이 정확히 L05의 타일 전치입니다 — 핵심 통찰: **전역 메모리는 합치되어야 함; 공유 메모리 접근 패턴은 중요하지 않음 (뱅크 충돌 제외)**.

---

## 핵심 요약

- GPU는 메모리를 **128바이트 (32원소) 트랜잭션** 단위로 발행 — warp 하나, 캐시 라인 하나가 이상적
- **스트라이드-1** (연속) 접근 = warp당 트랜잭션 1개 = 전체 대역폭
- **스트라이드-N** 접근 = warp당 트랜잭션 N개 = 1/N 대역폭
- **SoA가 GPU에서 AoS보다 낫다** — 구조체 필드 접근을 스트라이드-N에서 스트라이드-1로 변환
- 메모리 병목 커널에서 명령 수를 줄이려면 `float4`/`float2` 로드 사용
- `l1tex__average_t_sectors_per_request`로 프로파일링 — 목표는 1.0

---

**다음**: [09. 점유율과 실행 구성](./09_Occupancy_and_Launch_Config.md) — 레지스터 압박과 공유 메모리 한도가 점유율을 어떻게 제한하는지 정량화하고, `__launch_bounds__`로 컴파일러를 안내합니다.
