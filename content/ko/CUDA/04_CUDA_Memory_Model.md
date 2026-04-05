# 04. CUDA 메모리 모델

**이전**: [스레드 인덱싱과 그리드](./03_Thread_Indexing_and_Grids.md) | **다음**: [공유 메모리와 타일링](./05_Shared_Memory_and_Tiling.md)

---

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 모든 GPU 메모리 유형 설명: 전역, 공유, 레지스터, L1/L2, 상수, 텍스처
2. 주어진 접근 패턴에 맞는 올바른 메모리 유형 선택
3. 사용자 정의 커널로 메모리 대역폭 벤치마크
4. 메모리 유형 오용이 성능 절벽을 초래하는 이유 설명
5. 상수 메모리에 `cudaMemcpyToSymbol` 사용

---

## 1. GPU 메모리 계층 구조

GPU 프로그램은 메모리와의 관계에 의해 성능이 결정됩니다. 계층 구조를 이해하는 것이 필수적입니다:

```
가장 빠름 ────────────────────────────────────────── 가장 느림

  레지스터     공유/L1     L2 캐시     전역 (HBM/GDDR)
  ──────     ───────     ───────     ────────────────
  ~0 사이클   1–5 사이클  50 사이클   400–700 사이클
  256 KB/SM  48–228 KB   40–80 MB    총 40–80 GB
  ~19 TB/s   ~19 TB/s    ~5 TB/s     ~2 TB/s (A100)
  스레드별    블록별       장치         장치
```

각 메모리 유형은 서로 다른 **범위**, **수명**, **접근 패턴**을 가집니다.

---

## 2. 전역 메모리

주 GPU DRAM (HBM 또는 GDDR). 모든 스레드가 읽기/쓰기 가능:

```c
__global__ void kernel(float *data, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        data[i] = data[i] * 2.0f;  // 전역 메모리 읽기 + 쓰기
    }
}
```

**특성**:
- `cudaMalloc`으로 할당, `cudaMemcpy`로 전송
- 애플리케이션 수명 동안 유지
- L2에 캐시됨 (Volta+에서 읽기 전용 접근은 L1에도 캐시)
- **합치기(coalescing)가 핵심** — 인접 스레드가 인접 주소에 접근해야 128바이트 트랜잭션으로 결합

**대역폭 벤치마크**:

```c
__global__ void bw_benchmark(float *in, float *out, long n) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    for (; i < n; i += (long)gridDim.x * blockDim.x)
        out[i] = in[i];  // 복사 — 원소당 읽기 1 + 쓰기 1
}

// 기대값: 장치 최대 메모리 대역폭에 근접
// A100: 최대 ~2000 GB/s, 달성 가능: ~1800 GB/s (90%)
```

---

## 3. 공유 메모리

블록별 빠른 스크래치패드, 프로그래머가 명시적으로 관리:

```c
__global__ void reduction_kernel(float *data, float *result, int n) {
    __shared__ float smem[256];  // 커널에서 선언 — 정적 할당

    int tid  = threadIdx.x;
    int i    = blockIdx.x * blockDim.x + tid;

    smem[tid] = (i < n) ? data[i] : 0.0f;  // 전역 → 공유 메모리 로드
    __syncthreads();                         // 모든 스레드 대기

    // 블록 내 리덕션
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }

    if (tid == 0) result[blockIdx.x] = smem[0];  // 블록 결과 쓰기
}
```

**특성**:
- **정적**: `__shared__ float buf[256];` — 컴파일 시 크기 알 수 있음
- **동적**: `extern __shared__ float buf[];` + 세 번째 커널 실행 파라미터

```c
// 동적 공유 메모리 — 실행 시 크기 지정:
reduction_kernel<<<grid, block, sharedBytes>>>(args);
```

- 수명: 커널 실행 하나
- 범위: **같은 블록** 내 모든 스레드
- 속도: L1 캐시와 동일 (~19 TB/s, ~1–5 사이클)
- **뱅크 충돌**은 대역폭을 감소시킵니다 — L05에서 전체 분석

**공유 메모리 / L1 비율 설정** (Ampere+는 통합 풀 보유):

```c
// 공유 메모리 48 KB (기본값) 또는 런타임 API로 최대 228 KB
cudaFuncSetAttribute(myKernel,
    cudaFuncAttributeMaxDynamicSharedMemorySize, 131072);  // 128 KB
```

---

## 4. 레지스터

가장 빠른 저장소 — 0사이클 접근의 스레드 전용 프라이빗 변수:

```c
__global__ void kernel(float *a, float *b, float *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;  // 레지스터
    float x = a[i];   // 레지스터 — 전역에서 한 번 로드
    float y = b[i];   // 레지스터
    float result = x * x + y * y;  // 모두 레지스터 산술
    c[i] = result;    // 전역에 다시 쓰기
}
```

**특성**:
- 각 스레드 전용 — 다른 스레드가 접근 불가
- Ampere에서 SM당 64K 32비트 레지스터
- 스레드당 레지스터가 너무 많으면 → **레지스터 스필링** (지역 메모리, 즉 전역 메모리로 유출)
- 레지스터 사용량 확인: `nvcc -Xptxas -v kernel.cu`는 `Used N registers` 표시

**레지스터 압박**: 점유율의 주요 제약 요인.

```
스레드당 레지스터 = 32 → SM당 최대 2048 스레드 (64K/32) = 64 warp → 100% 점유율
스레드당 레지스터 = 64 → SM당 최대 1024 스레드 (64K/64) = 32 warp → 50% 점유율
스레드당 레지스터 = 128 → 512 스레드/SM = 16 warp → 25% 점유율
```

---

## 5. 상수 메모리

전용 캐시(64 KB)에 캐시된 읽기 전용 메모리로, **브로드캐스트 접근** (warp의 모든 스레드가 같은 주소 읽기)에 빠름:

```c
// 파일 범위에서 선언 (장치 가시적)
__constant__ float filter[64];   // 예: 합성곱 가중치

__global__ void conv_kernel(const float *in, float *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float sum = 0.0f;
        for (int k = 0; k < 64; k++)
            sum += filter[k] * in[i + k];  // 모든 스레드가 filter[k] 읽기 → 브로드캐스트
        out[i] = sum;
    }
}

// CPU에서 초기화:
cudaMemcpyToSymbol(filter, h_filter, 64 * sizeof(float));
```

**사용 시점**: 모든 스레드가 동시에 공유하는 가중치, 룩업 테이블, 상수.

**사용하지 말아야 할 때**: warp의 다른 스레드가 서로 다른 인덱스를 읽는 경우 → 직렬화 (브로드캐스트보다 32배 느림). 대신 전역 메모리 + L2 캐시 사용.

---

## 6. 텍스처 메모리

읽기 전용, 캐시됨, 하드웨어 가속 보간 및 2D/3D 공간 지역성:

```c
// 텍스처 객체 API (CUDA 5.0+)
cudaTextureObject_t tex;

cudaResourceDesc resDesc;
memset(&resDesc, 0, sizeof(resDesc));
resDesc.resType = cudaResourceTypeLinear;
resDesc.res.linear.devPtr = d_data;
resDesc.res.linear.desc.f = cudaChannelFormatKindFloat;
resDesc.res.linear.desc.x = 32;  // 32비트 float
resDesc.res.linear.sizeInBytes = N * sizeof(float);

cudaTextureDesc texDesc;
memset(&texDesc, 0, sizeof(texDesc));
texDesc.readMode = cudaReadModeElementType;

cudaCreateTextureObject(&tex, &resDesc, &texDesc, NULL);

__global__ void texture_kernel(cudaTextureObject_t tex, float *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        out[i] = tex1Dfetch<float>(tex, i);
}
```

**최적 사용처**: 이미지 처리 (공간 지역성, 경계 클램핑, 이중선형 보간), 텍스처 캐시가 L1을 능가하는 불규칙 접근 패턴.

현대 코드 (Volta+)에서는 보통 `__ldg()`(읽기 전용 L1 캐시를 통한 로드)로 충분합니다:

```c
float val = __ldg(&d_data[i]);  // 읽기 전용, L1 읽기 전용 캐시에 캐시됨
```

---

## 7. 지역 메모리

잘못된 이름 — **지역 메모리는 실제로 전역 메모리**이며, 레지스터가 넘칠 때 사용됩니다:

```c
__global__ void large_array_kernel(float *out) {
    float local_arr[1024];  // 레지스터에 너무 큼 → 지역 메모리로 스필
    // ... (느림 — 접근당 400+ 사이클 지연)
}
```

컴파일러가 자동으로 스필을 처리합니다. `nvcc -Xptxas -v`로 감지:
```
ptxas info: myKernel 함수 속성
  스택 프레임 24 바이트, 스필 저장 24 바이트, 스필 로드 24 바이트
```
비영 스필 카운트는 성능 경고입니다.

---

## 8. 메모리 접근 패턴 요약

| 메모리 | 범위 | 지연 | 대역폭 | 크기 | 관리 |
|--------|------|------|--------|------|------|
| **레지스터** | 스레드별 | 0 | ~19 TB/s | 256 KB/SM | 컴파일러 |
| **공유** | 블록별 | 1–5 사이클 | ~19 TB/s | 48–228 KB/SM | 프로그래머 |
| **L1 캐시** | SM별 | 1–5 사이클 | ~19 TB/s | 32 KB (SM의 일부) | 하드웨어 |
| **L2 캐시** | 장치 | 50 사이클 | ~5 TB/s | 40–80 MB | 하드웨어 |
| **전역** | 장치 | 400–700 사이클 | ~2 TB/s | 40–80 GB | 프로그래머 |
| **상수** | 장치 | 1–5 사이클 (브로드캐스트) | 빠름 | 64 KB | 프로그래머 |
| **텍스처** | 장치 | 1–5 사이클 (캐시됨) | 보통 | 48 KB 캐시 | 프로그래머 |
| **지역** | 스레드별 | 400–700 사이클 | ~2 TB/s | 스레드당 512 KB | 컴파일러 |

---

## 9. 고정 (페이지-잠금) 호스트 메모리

표준 `malloc`은 OS가 스왑 가능한 **페이지 가능** 메모리를 할당합니다. 더 빠른 PCIe 전송을 위해 **고정** 메모리 사용:

```c
float *h_pinned;
cudaHostAlloc(&h_pinned, bytes, cudaHostAllocDefault);

// 전송이 ~2배 빠름 (PCIe 4.0에서 3–12 GB/s 대신 6–12 GB/s)
cudaMemcpy(d_data, h_pinned, bytes, cudaMemcpyHostToDevice);

cudaFreeHost(h_pinned);  // free()가 아닌 cudaFreeHost() 사용 필수
```

**트레이드오프**: 고정 메모리는 스왑 불가, 물리 RAM을 소비합니다. 과도하게 고정하지 마세요.

---

## 10. 통합 메모리 (UM)

CUDA 6.0+는 CPU와 GPU 모두에서 접근 가능한 단일 포인터를 허용합니다:

```c
float *data;
cudaMallocManaged(&data, bytes);  // 단일 할당

// CPU가 읽기/쓰기 가능:
for (int i = 0; i < N; i++) data[i] = (float)i;

// GPU가 읽기/쓰기 가능 (커널 호출 후):
myKernel<<<grid, block>>>(data, N);
cudaDeviceSynchronize();

// CPU가 결과 읽기:
printf("data[0] = %f\n", data[0]);

cudaFree(data);
```

**UM은 프로토타이핑에 편리하지만** 자동 페이지 마이그레이션으로 인해 느릴 수 있습니다. NVLink 시스템 (예: 직접 호스트-GPU 메모리 접근이 가능한 A100 SXM)에서는 UM 성능이 우수할 수 있습니다. 프로덕션에서는 명시적 `cudaMemcpy` 선호.

### 통합 메모리 성능 개선: 프리페칭과 어드바이스

자동 페이지 마이그레이션은 첫 번째 접근 시 (페이지 폴트로) 트리거되어 지연이 발생합니다. 두 가지 API로 이를 해소할 수 있습니다:

```c
// 프리페치(Prefetch): 커널 실행 전에 장치로 페이지를 마이그레이션
// 커널 실행 중 페이지 폴트를 방지
cudaMemPrefetchAsync(data, bytes, deviceId, stream);
myKernel<<<grid, block, 0, stream>>>(data, N);

// MemAdvise: 드라이버에 접근 패턴 힌트 제공
// ReadMostly — 드라이버가 여러 프로세서에 읽기 전용 복사본 생성 가능
cudaMemAdvise(data, bytes, cudaMemAdviseSetReadMostly, deviceId);

// PreferredLocation — 명시적으로 마이그레이션하지 않는 한 이 장치에 페이지 유지
cudaMemAdvise(data, bytes, cudaMemAdviseSetPreferredLocation, deviceId);

// AccessedBy — 마이그레이션 없이 장치의 페이지 테이블에 페이지 매핑
// (CPU와 GPU가 데이터를 자주 접근하는 경우 유용)
cudaMemAdvise(data, bytes, cudaMemAdviseSetAccessedBy, deviceId);
```

Pascal+ GPU에서 `cudaMemPrefetchAsync`와 `cudaMemAdvise`를 함께 사용하면 통합 메모리 성능을 명시적 `cudaMemcpy` 워크플로우의 5–10% 이내로 끌어올릴 수 있습니다.

---

## 핵심 요약

- **전역 메모리** (HBM)는 크지만 느림 — 합치된 접근이 성능에 필수
- **공유 메모리**는 프로그래머의 캐시 — 여러 스레드가 재사용하는 데이터를 스테이징하는 데 사용
- **레지스터**는 지연 비용 없음이지만 수가 제한됨; 스필은 성능을 망침
- **상수 메모리**는 소형 브로드캐스트 읽기 데이터 (가중치, 필터 계수)에 탁월
- `cudaHostAlloc`으로 고정 메모리를 사용하여 호스트↔장치 전송 가속
- 메모리 계층 피라미드: 레지스터 > 공유/L1 > L2 > 전역 — 아래에서부터 최적화

---

**다음**: [05. 공유 메모리와 타일링](./05_Shared_Memory_and_Tiling.md) — 공유 메모리를 사용하여 타일된 행렬 곱셈을 구현하고, 뱅크 충돌을 제거하며, Nsight Compute로 프로파일링합니다.
