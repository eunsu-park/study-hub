# 06. Warp 실행과 분기 발산

**이전**: [공유 메모리와 타일링](./05_Shared_Memory_and_Tiling.md) | **다음**: [원자적 연산](./07_Atomic_Operations.md)

---

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. SIMT 실행과 warp 분기 발산이 스레드를 직렬화하는 이유 설명
2. 분기 발산 패턴 식별 및 최소화를 위한 코드 재구성
3. warp 레벨 술어를 위한 `__ballot_sync`, `__any_sync`, `__all_sync` 사용
4. `__shfl_down_sync`로 warp 레벨 리덕션 구현
5. 성능 중요 커널에서 warp 균일 제어 흐름 작성

---

## 1. SIMT: 하나의 명령, 여러 스레드

32개 스레드로 구성된 warp는 단일 단위로 실행됩니다 — 모든 스레드가 매 사이클마다 **같은 명령**을 실행합니다. 이것이 SIMT: 단일 명령 다중 스레드(Single Instruction, Multiple Threads)입니다.

32개 스레드 모두 같은 분기를 택할 때: 성능 손실 없음.
스레드가 발산할 때: warp가 직렬화 — 두 경로를 순차적으로 실행합니다.

```
4개 스레드로 구성된 warp (단순화):

코드:
  if (x > 0) {
      y = x * 2;     // 경로 A
  } else {
      y = -x;        // 경로 B
  }

스레드 0: x =  5  (경로 A 선택)
스레드 1: x = -3  (경로 B 선택)
스레드 2: x =  7  (경로 A 선택)
스레드 3: x = -1  (경로 B 선택)

발산을 통한 실행:
  사이클 1: 스레드 0,2가 y = x*2 실행;  스레드 1,3은 MASKED (유휴)
  사이클 2: 스레드 1,3이 y = -x 실행;   스레드 0,2는 MASKED (유휴)
  합계: 1 사이클 대신 2 사이클 → 50% 효율
```

SIMT 하드웨어는 **활성 마스크**를 사용합니다 — 비활성화된 스레드는 아무것도 하지 않고 기다립니다.

---

## 2. 실제 발산 패턴

### 패턴 1: 데이터 의존 분기 (나쁨)

```c
// 히스토그램 스타일: 서로 다른 스레드가 서로 다른 경로 선택
if (value > threshold) {
    result = compute_heavy(value);    // 일부 스레드만 실행
} else {
    result = 0.0f;
}
```

**해결**: 분기 몸체가 저렴하다면, 산술로 대체:

```c
// 발산 없음 — 모든 스레드에 대해 같은 명령
float mask   = (float)(value > threshold);
result = mask * compute_light(value);  // 건너뛰는 대신 0으로 곱셈
```

이는 `compute_light`가 모든 스레드에서 실행해도 충분히 저렴할 때만 유효합니다.

### 패턴 2: 스레드 ID 기반 발산 (때로는 불가피)

```c
// 리덕션에서 일반적 — 하위 절반 스레드가 작업, 나머지 유휴
if (threadIdx.x < s) {
    smem[threadIdx.x] += smem[threadIdx.x + s];
}
```

`s = 16`일 때: 스레드 0–15가 실행, 16–31은 마스킹됨. warp 경계에서의 2방향 발산만 — 전체 warp 리덕션에서는 비교적 무해합니다.

### 패턴 3: 루프 발산 (위험)

```c
// 스레드들이 서로 다른 횟수로 반복
int count = data[threadIdx.x];  // 스레드별로 다른 값!
for (int i = 0; i < count; i++) {
    process(i);
}
```

warp는 **마지막 스레드가 끝날 때까지** 실행됩니다 — 다른 모든 스레드가 마스킹되어 기다립니다. 횟수 편차가 크면 (예: 0에서 100), 평균 활용률이 ~50%가 될 수 있습니다.

---

## 3. Warp 레벨 내장 함수

Volta (CC 7.0) 이후, CUDA는 명시적 동기화 마스크와 함께 **warp 레벨 기본 연산**을 제공합니다. 마스크는 참여하는 스레드를 지정합니다.

### `__ballot_sync`: 어느 스레드가 조건을 만족하는가?

```c
unsigned mask = __ballot_sync(0xFFFFFFFF, predicate);
// k번째 스레드의 predicate가 true이면 비트 k = 1인 32비트 정수 반환
// 0xFFFFFFFF는 "32개 스레드 모두 참여"를 의미

// 예: warp에서 몇 개의 스레드가 value > 0인지 세기
unsigned active = __ballot_sync(0xFFFFFFFF, value > 0.0f);
int count = __popc(active);  // popcount = 설정된 비트 수
```

### `__any_sync` / `__all_sync`

```c
// 마스크의 어느 스레드라도 조건을 만족하면 true
if (__any_sync(0xFFFFFFFF, has_work)) {
    // 적어도 하나의 스레드가 할 일이 있음
}

// 모든 스레드가 조건을 만족하면 true
if (__all_sync(0xFFFFFFFF, is_valid)) {
    // 안전하게 진행 — 모든 스레드가 유효한 데이터 보유
}
```

---

## 4. Warp 셔플: 공유 메모리 없이 통신

warp 내 스레드는 공유 메모리를 거치지 않고 직접 레지스터 값을 교환할 수 있습니다 — `__shfl_sync` 계열:

```c
// warp 내 특정 레인의 레지스터 값 읽기
int   __shfl_sync     (unsigned mask, int var, int srcLane, int width=32);

// delta 레인만큼 아래로 이동 (리덕션에 유용)
float __shfl_down_sync(unsigned mask, float var, unsigned delta, int width=32);

// delta 레인만큼 위로 이동
float __shfl_up_sync  (unsigned mask, float var, unsigned delta, int width=32);

// XOR 기반 교환 (나비 패턴)
float __shfl_xor_sync (unsigned mask, float var, int laneMask, int width=32);
```

`__syncthreads()`가 필요 없습니다 — warp의 모든 스레드가 동기적으로 실행됩니다.

### 예: 레인 0의 값을 모든 스레드에 브로드캐스트

```c
float leader_val = __shfl_sync(0xFFFFFFFF, my_val, 0);
// 이제 모든 스레드가 스레드 0이 가졌던 값을 보유
```

---

## 5. `__shfl_down_sync`를 이용한 Warp 리덕션

공유 메모리 없이 32개 값을 합산하는 전형적인 warp 레벨 리덕션:

```c
__device__ float warp_reduce_sum(float val) {
    // 각 단계: 스레드 k가 k와 k+offset의 합을 얻음
    val += __shfl_down_sync(0xFFFFFFFF, val, 16);
    val += __shfl_down_sync(0xFFFFFFFF, val, 8);
    val += __shfl_down_sync(0xFFFFFFFF, val, 4);
    val += __shfl_down_sync(0xFFFFFFFF, val, 2);
    val += __shfl_down_sync(0xFFFFFFFF, val, 1);
    // 스레드 0이 이제 32개 값의 합을 보유
    return val;
}
```

4개 스레드 실행 추적 (단순화):

```
초기값:  T0=1, T1=2, T2=3, T3=4
delta=2: T0 += T2 → T0=4,  T1 += T3 → T1=6
delta=1: T0 += T1 → T0=10, T1=6
결과:    T0 = 10 = 1+2+3+4  ✓
```

**성능**: 5개 셔플 명령 vs 5개 공유 메모리 로드+저장. 공유 메모리 버전보다 약 **2배 빠름**이며 동기화가 필요 없습니다.

---

## 6. Warp 리덕션을 사용한 블록 레벨 리덕션

전체 블록을 위해 warp 리덕션과 공유 메모리를 결합합니다:

```c
__device__ float block_reduce_sum(float val) {
    __shared__ float warp_sums[32];  // warp당 슬롯 하나 (최대 32 warp/블록)

    int lane   = threadIdx.x % 32;
    int warpId = threadIdx.x / 32;

    // 1단계: 각 warp 내에서 리덕션
    val = warp_reduce_sum(val);

    // 2단계: 각 warp의 첫 번째 스레드가 공유 메모리에 씀
    if (lane == 0) warp_sums[warpId] = val;
    __syncthreads();

    // 3단계: 첫 번째 warp가 warp 합계를 리덕션
    val = (threadIdx.x < blockDim.x / 32) ? warp_sums[lane] : 0.0f;
    if (warpId == 0) val = warp_reduce_sum(val);

    // 스레드 0이 블록 전체 합계를 보유
    return val;
}
```

---

## 7. 술어 실행: 발산 없이

짧은 분기에서 컴파일러는 **술어 실행(predicated execution)**을 사용할 수 있습니다 — 두 경로 모두 실행되지만 비선택된 경로의 결과가 폐기됩니다:

```ptx
// 어셈블리 등가: if (x > 0) y = 1; else y = -1;
setp.gt.f32  p, x, 0.0
@p  mov.f32  y, 1.0
@!p mov.f32  y, -1.0
```

두 명령 모두 실행됩니다 (발산 없음), 하지만 술어 게이트가 잘못된 결과를 폐기합니다. 매우 짧은 몸체에서 실제 분기보다 빠릅니다. 컴파일러가 간단한 조건문에서 자동으로 처리합니다.

---

## 8. Warp 레벨 리덕션 벤치마크

```c
// 벤치마크: N floats 합산을 위한 공유 메모리 리덕션 vs 셔플 리덕션
// 블록 크기 = 256 (8 warp)

// 공유 메모리 버전: N=1M에서 ~3.2 μs
// 셔플 버전:       N=1M에서 ~1.8 μs   (1.8× 속도 향상)

// 차이 프로파일링: 셔플은 블록당 공유 메모리 읽기-쓰기 쌍 ~8개를 피함
```

---

## 9. 모범 사례 요약

| 패턴 | 나쁨 | 좋음 |
|------|------|------|
| 짧은 분기 | `if (flag) y = a*b; else y = 0;` | `y = flag * a * b;` |
| Warp 리덕션 | 공유 메모리 5단계 | `__shfl_down_sync` 5단계 |
| 레인 특정 작업 | `if (lane == 0) { ... }` | 허용됨 — 스레드 1개 오버헤드 |
| 데이터 의존 루프 | 스레드별 루프 횟수 | 균일한 횟수 사전 계산 |
| 술어 검사 | 비싼 경로 전에 `__any_sync` | ✓ 모두 유휴일 때 건너뛰기 위해 사용 |

---

## 핵심 요약

- **발산은 warp를 직렬화** — 두 경로 모두 실행되고 마스킹된 스레드가 유휴; 2방향 발산 = 50% 효율
- **`__ballot_sync`**는 술어 결과를 warp 레벨 로직을 위한 비트마스크로 변환
- **`__shfl_down_sync`**는 warp 내 레지스터-레지스터 통신 활성화 — 리덕션에서 공유 메모리보다 빠르고 간단
- **Warp 리덕션** 5단계 셔플로 공유 메모리 연산 5개 대체 — ~2배 빠름
- 짧은 몸체의 분기 → 산술 사용 (발산 비용 없음)

---

**다음**: [07. 원자적 연산](./07_Atomic_Operations.md) — 잠금 없는 카운터, 히스토그램 커널 구현, 원자적 충돌의 처리량 비용 측정.
