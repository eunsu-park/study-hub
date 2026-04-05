# 31. Cooperative Groups

**이전**: [Mixed Precision and Tensor Cores](./30_Mixed_Precision_and_Tensor_Cores.md) | **다음**: [GEMM from Scratch](./32_GEMM_from_Scratch.md)

---

## 학습 목표

이 레슨을 마치면 다음을 할 수 있습니다:

1. Cooperative Groups 프로그래밍 모델 이해 및 `__syncthreads()`보다 나은 이유 설명하기
2. 서브-block 동기화를 위해 `cg::thread_block`과 `cg::tile_partition<N>` 사용하기
3. `cg::tile_partition<32>`와 `group.shfl_down()`을 사용한 warp 수준 reduction 구현하기
4. `cg::grid_group`을 위한 `cudaLaunchCooperativeKernel`로 그리드 협력 kernel 실행하기
5. warp 내의 활성 (분기되지 않은) thread를 대상으로 작업하기 위해 `cg::coalesced_group` 사용하기

---

## 1. Cooperative Groups가 필요한 이유

전통적인 CUDA 동기화는 경직되어 있습니다:
- `__syncthreads()`는 항상 block의 모든 thread를 동기화합니다
- Warp 내장 함수 (`__shfl_sync`, `__ballot_sync`)는 명시적인 마스크가 필요합니다
- block의 부분 집합을 동기화하는 이식 가능한 방법이 없습니다

CUDA 9에서 도입된 Cooperative Groups (CG)는 다음을 제공합니다:
```
thread_block     — block의 모든 thread (__syncthreads 대체)
tile_partition<N>— block 내의 N개 thread 그룹 (N은 2의 거듭제곱)
grid_group       — 모든 block의 모든 thread (협력 실행 필요)
coalesced_group  — warp 내의 현재 활성 (수렴된) thread
```

핵심 이점: warp/block에 하드코딩되지 않고 **그룹 크기에 의해 매개변수화된 알고리즘**.

---

## 2. thread_block: Block 수준 동기화

```c
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

__global__ void block_reduce_cg(const float *in, float *out, int N) {
    extern __shared__ float sdata[];

    // 이 thread block에 대한 handle 획득
    cg::thread_block cta = cg::this_thread_block();

    int i   = cta.group_index().x * cta.group_dim().x + cta.thread_index().x;
    int tid = cta.thread_index().x;

    sdata[tid] = (i < N) ? in[i] : 0.f;

    // CG를 통한 block 동기화 (__syncthreads()와 동등)
    cta.sync();   // 또는 cg::sync(cta);

    // block 내의 reduction
    for (unsigned stride = cta.size() / 2; stride > 0; stride >>= 1) {
        if (tid < stride) sdata[tid] += sdata[tid + stride];
        cta.sync();
    }

    if (tid == 0) out[cta.group_index().x] = sdata[0];
}
```

이것은 `__syncthreads()`와 동일하게 보이지만 그룹 객체를 헬퍼 함수에 전달할 수 있어 모듈식 코드를 가능하게 합니다:

```c
// 제네릭 reduce 함수 — .sync()를 가진 모든 그룹 타입과 동작
template <typename Group>
__device__ float reduce_sum(Group g, float *shared, float val) {
    int lane = g.thread_rank();
    shared[lane] = val;
    g.sync();

    for (int stride = g.size() / 2; stride > 0; stride >>= 1) {
        if (lane < stride) shared[lane] += shared[lane + stride];
        g.sync();
    }
    return shared[0];
}
```

---

## 3. tile_partition: 서브-Block 그룹

`tile_partition<N>`은 block을 N개 thread의 고정 크기 tile로 분할합니다 (N은 2의 거듭제곱, ≤32):

```c
__global__ void warp_level_reduce(const float *in, float *out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // block을 32의 tile로 분할 (= warp 크기)
    cg::thread_block  cta  = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(cta);

    float val = (i < N) ? in[i] : 0.f;

    // tile을 사용한 warp 수준 reduction
    for (int offset = warp.size() / 2; offset > 0; offset >>= 1)
        val += warp.shfl_down(val, offset);

    // 각 warp의 lane 0이 부분 합을 보유
    if (warp.thread_rank() == 0)
        atomicAdd(out, val);
}

// 더 세밀한 동기화를 위한 N < 32의 tile_partition
__global__ void group8_example(const int *data, int *out, int N) {
    cg::thread_block cta = cg::this_thread_block();
    cg::thread_block_tile<8> g8 = cg::tiled_partition<8>(cta);

    int i    = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = g8.thread_rank();   // 0..7

    // 8개 thread의 각 그룹이 독립적인 미니 reduction 수행
    int val = (i < N) ? data[i] : 0;
    for (int s = g8.size()/2; s > 0; s >>= 1)
        val += g8.shfl_down(val, s);

    if (lane == 0)
        atomicAdd(out, val);
}
```

---

## 4. group.shfl_down, shfl_up, shfl_xor

CG tile은 CUDA warp 내장 함수와 일치하지만 그룹을 통해 타입화된 `shfl_*` 연산을 제공합니다:

```c
__device__ float warp_reduce_max(cg::thread_block_tile<32> warp, float val) {
    // shfl_down을 사용한 warp 수준 최대 reduction
    for (int offset = 16; offset > 0; offset >>= 1)
        val = fmaxf(val, warp.shfl_down(val, offset));
    return val;   // lane 0만 진정한 최댓값을 가짐
}

__device__ float warp_scan_inclusive(cg::thread_block_tile<32> warp, float val) {
    // 포괄적 scan: 각 lane이 자신의 lane까지 (포함) prefix sum을 가짐
    for (int offset = 1; offset < 32; offset <<= 1) {
        float tmp = warp.shfl_up(val, offset);
        if (warp.thread_rank() >= offset) val += tmp;
    }
    return val;
}

// 버터플라이 reduction (균형 통신을 위해 shfl_xor 사용)
__device__ float warp_reduce_sum_butterfly(cg::thread_block_tile<32> warp, float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val += warp.shfl_xor(val, offset);
    return val;  // 32개 lane 모두 총합을 보유 (브로드캐스트 결과)
}
```

---

## 5. grid_group: 그리드 범위 동기화

`grid_group`은 모든 block의 모든 thread를 동기화합니다. **협력 실행**이 필요합니다:

```c
#include <cooperative_groups.h>

// 전역 장벽을 위한 grid_group을 사용하는 kernel
__global__ void grid_reduce_two_pass(const float *in, float *partial, float *out, int N) {
    cg::grid_group grid = cg::this_grid();

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // --- 패스 1: 로컬 block reduction ---
    extern __shared__ float sdata[];
    sdata[threadIdx.x] = (i < N) ? in[i] : 0.f;
    __syncthreads();

    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) partial[blockIdx.x] = sdata[0];

    // --- 전역 장벽: 모든 block이 패스 1을 완료해야 함 ---
    grid.sync();   // 협력 실행 필요!

    // --- 패스 2: block 0이 부분 합을 reduce ---
    if (blockIdx.x == 0) {
        sdata[threadIdx.x] = (threadIdx.x < gridDim.x)
                             ? partial[threadIdx.x] : 0.f;
        __syncthreads();
        for (int s = blockDim.x/2; s > 0; s >>= 1) {
            if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
            __syncthreads();
        }
        if (threadIdx.x == 0) *out = sdata[0];
    }
}

// 협력 kernel API로 실행
void launch_grid_reduce(const float *d_in, float *d_partial, float *d_out, int N) {
    int block = 256;
    int grid  = (N + block - 1) / block;

    // device가 협력 실행을 지원하는지 확인
    int can_cooperative;
    cudaDeviceGetAttribute(&can_cooperative, cudaDevAttrCooperativeLaunch, 0);
    if (!can_cooperative) { fprintf(stderr, "협력 실행 지원 없음\n"); return; }

    void *args[] = { (void*)&d_in, (void*)&d_partial, (void*)&d_out, (void*)&N };
    size_t shared = block * sizeof(float);

    cudaLaunchCooperativeKernel(
        (void*)grid_reduce_two_pass,
        grid, block, args, shared, nullptr);
}
```

---

## 6. coalesced_group: 활성 Thread만

`cg::coalesced_threads()`는 warp에서 현재 활성 (분기되지 않은) thread만 캡처합니다. 분기가 많은 kernel에 유용합니다:

```c
// 활성 그룹을 사용하여 조건을 만족하는 thread만 처리
__global__ void process_active_only(int *data, int *out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    if (data[i] > 0) {
        // 이 분기를 취한 thread들만으로 그룹 형성
        cg::coalesced_group active = cg::coalesced_threads();

        // 활성 thread만에 대해 reduce (유휴 thread가 작업 낭비 없음)
        float val = (float)data[i];
        for (int offset = active.size()/2; offset > 0; offset >>= 1)
            val += active.shfl_down(val, offset);

        if (active.thread_rank() == 0)
            atomicAdd(out, (int)val);
    }
}

// labeled_partition: 레이블 (예: thread 색상)로 활성 thread 분할
__global__ void labeled_example(int *colors, float *vals, float *out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    int   color = colors[i];
    float val   = vals[i];

    // 이 warp 내에서 동일한 색상의 thread 그룹화
    cg::coalesced_group same_color = cg::labeled_partition(
        cg::coalesced_threads(), color);

    // 동일-색상 그룹 내에서 reduce
    for (int off = same_color.size()/2; off > 0; off >>= 1)
        val += same_color.shfl_down(val, off);

    if (same_color.thread_rank() == 0)
        atomicAdd(&out[color], val);
}
```

---

## 7. 유연한 Warp Reduction 유틸리티

CG와 템플릿을 결합하면 진정으로 유연한 reduction 함수를 제공합니다:

```c
// tile_partition<32>, tile_partition<16>, coalesced_group 등에 동작
template <typename GroupT>
__device__ float group_reduce_sum(GroupT g, float val) {
    for (int offset = g.size() / 2; offset > 0; offset >>= 1)
        val += g.shfl_down(val, offset);
    return val;  // shfl_down의 경우 thread_rank()==0에서만 유효
}

template <typename GroupT>
__device__ float group_reduce_max(GroupT g, float val) {
    for (int offset = g.size() / 2; offset > 0; offset >>= 1)
        val = fmaxf(val, g.shfl_down(val, offset));
    return val;
}

// 예시: 다단계 reduction (warp → block → grid)
__global__ void multi_stage_reduce(const float *in, float *out, int N) {
    cg::thread_block  cta  = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(cta);
    extern __shared__ float sdata[];

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float val = (i < N) ? in[i] : 0.f;

    // 단계 1: warp reduce
    val = group_reduce_sum(warp, val);

    // 단계 2: block reduce (warp당 하나의 값)
    if (warp.thread_rank() == 0)
        sdata[threadIdx.x / 32] = val;
    cta.sync();

    if (threadIdx.x < cta.size() / 32) {
        val = sdata[threadIdx.x];
        cg::thread_block_tile<8> last_group = cg::tiled_partition<8>(cta);
        val = group_reduce_sum(last_group, val);
    }

    if (threadIdx.x == 0) out[blockIdx.x] = val;
}
```

---

## 핵심 요약

- **Cooperative Groups**는 하드코딩된 `__syncthreads()`를 조합 가능한 그룹 객체로 대체합니다: 헬퍼 함수에 전달하여 모듈식 재사용 가능한 코드 작성
- **`cg::this_thread_block()`**은 전체 thread block에 대한 handle을 반환합니다; `.sync()`은 `__syncthreads()`와 동등하지만 그룹 크기 및 rank 메타데이터도 전달
- **`cg::tiled_partition<N>(cta)`**는 N개 thread의 서브-block 그룹을 생성합니다; 서브-warp 및 서브-block 동기화 패턴을 가능하게 합니다
- **`group.shfl_down(val, offset)`**은 tile 내에서 warp shuffle을 수행합니다; 수동 마스크 관리 없이 `__shfl_down_sync`와 동일하게 동작합니다
- **`cg::this_grid()`**와 `.sync()`는 전역 그리드 장벽을 제공하지만 `cudaLaunchCooperativeKernel`과 협력 실행을 지원하는 device가 필요합니다
- **`cg::coalesced_threads()`**는 분기 후 warp에서 활성 thread만 캡처합니다; `labeled_partition()`은 키 값으로 활성 thread를 추가로 그룹화합니다

---

**다음**: [32. GEMM from Scratch](./32_GEMM_from_Scratch.md) — 나이브 기준선부터 register tiling, float4 벡터화 구현까지 단계별로 고성능 행렬 곱셈 kernel을 구축하여 cuBLAS 성능에 근접합니다.
