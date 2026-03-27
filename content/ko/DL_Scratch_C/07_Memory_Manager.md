# 07. 메모리 관리자

**이전**: [Autograd 텐서 연산](./06_Autograd_Tensor_Ops.md) | **다음**: [합성곱 밑바닥 구현](./08_Convolution_from_Scratch.md)

---

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. `malloc`/`free`가 텐서 집약적 워크로드에 왜 문제가 되는지 설명
2. 임시 텐서를 위한 bump-pointer arena allocator 구현
3. 뷰 간에 공유된 텐서를 위한 참조 카운팅 구현
4. 고정 크기 버퍼를 재사용하는 텐서 메모리 풀 구축
5. 할당 오버헤드 프로파일링 및 개선 측정

---

## 1. 할당 문제

Transformer 순방향 패스에서 수백 개의 중간 텐서가 생성되고 소멸됩니다:

```
Q = x @ W_q        → [batch, seq, d_head] × n_heads 할당
K = x @ W_k        → 동일하게 할당
V = x @ W_v        → ...
attn = softmax(Q @ K^T / sqrt(d))  → [batch, n_heads, seq, seq] 할당
out  = attn @ V    → 할당 ...
```

GPT-2(12 레이어, 12 헤드, seq=512)의 경우:
- 순방향 패스당 ~200번의 할당
- 일반적인 `malloc` 비용: **~100ns per call**
- 총 오버헤드: 패스당 ~20 μs — 실시간 추론에서 중요

**해결책**:
1. **Arena allocator**: 대용량 블록을 선행 할당; 각 요청에 포인터를 증가; 한 번에 모두 해제
2. **텐서 풀**: 동일한 크기의 버퍼를 OS로 반환하지 않고 재사용
3. **참조 카운팅**: 여러 뷰가 하나의 버퍼를 안전하게 공유하도록 허용

---

## 2. Arena Allocator

Arena(선형 할당자 또는 리전 할당자라고도 함)는 가장 간단하고 빠른 할당자입니다:

```c
// arena.h
#pragma once
#include <stddef.h>
#include <stdbool.h>

typedef struct {
    char  *base;        // 메모리 블록의 시작
    size_t capacity;    // 바이트 단위 총 크기
    size_t offset;      // 현재 할당 커서
    size_t peak;        // 최고 수위선 (프로파일링용)
    int    n_allocs;    // 할당 횟수 (프로파일링용)
} Arena;

Arena *arena_create(size_t capacity_bytes);
void  *arena_alloc (Arena *arena, size_t size, size_t alignment);
void   arena_reset (Arena *arena);   // 커서를 0으로 재설정 — 메모리를 해제하지 않음
void   arena_destroy(Arena *arena);  // 백킹 블록 해제
void   arena_print_stats(const Arena *arena);
```

```c
// arena.c
#include "arena.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

Arena *arena_create(size_t capacity_bytes) {
    Arena *a = (Arena *)malloc(sizeof(Arena));
    // SIMD를 위해 64바이트에 정렬하여 할당
    if (posix_memalign((void **)&a->base, 64, capacity_bytes) != 0) {
        free(a); return NULL;
    }
    a->capacity = capacity_bytes;
    a->offset   = 0;
    a->peak     = 0;
    a->n_allocs = 0;
    return a;
}

void *arena_alloc(Arena *arena, size_t size, size_t alignment) {
    // 현재 offset을 `alignment`의 배수로 올림
    size_t aligned = (arena->offset + alignment - 1) & ~(alignment - 1);
    assert(aligned + size <= arena->capacity && "Arena 메모리 부족");

    void *ptr      = arena->base + aligned;
    arena->offset  = aligned + size;
    arena->n_allocs++;
    if (arena->offset > arena->peak) arena->peak = arena->offset;

    memset(ptr, 0, size);  // calloc처럼 0으로 초기화
    return ptr;
}

void arena_reset(Arena *arena) {
    arena->offset   = 0;
    arena->n_allocs = 0;
    // 메모리는 지워지지 않음 — arena_alloc의 memset에서 처리
}

void arena_destroy(Arena *arena) {
    free(arena->base);
    free(arena);
}

void arena_print_stats(const Arena *arena) {
    printf("Arena: 사용=%zu KB / %zu KB  peak=%zu KB  allocs=%d\n",
           arena->offset/1024, arena->capacity/1024,
           arena->peak/1024, arena->n_allocs);
}
```

### Arena 기반 텐서 할당

```c
// Arena에서 텐서 할당 (개별 해제 불필요)
Tensor *tensor_arena_alloc(Arena *arena, int ndim, const size_t *shape) {
    Tensor *t = (Tensor *)arena_alloc(arena, sizeof(Tensor), alignof(Tensor));
    t->ndim  = ndim;
    t->numel = 1;
    for (int i = 0; i < ndim; i++) { t->shape[i] = shape[i]; t->numel *= shape[i]; }

    t->strides[ndim-1] = 1;
    for (int i = ndim-2; i >= 0; i--)
        t->strides[i] = t->strides[i+1] * shape[i+1];

    t->data      = (float *)arena_alloc(arena, t->numel * sizeof(float), 64);
    t->owns_data = false;  // Arena가 메모리를 소유
    return t;
}
```

**추론을 위한 사용 패턴**:
```c
// GPT-2 순방향 패스를 위한 500MB arena 선행 할당
Arena *scratch = arena_create(512ULL * 1024 * 1024);

for (int step = 0; step < n_steps; step++) {
    arena_reset(scratch);   // O(1) — 커서만 재설정

    // Arena에서 모든 중간 텐서 할당
    Tensor *Q = tensor_arena_alloc(scratch, 3, (size_t[]){batch, seq, d_head});
    Tensor *K = tensor_arena_alloc(scratch, 3, (size_t[]){batch, seq, d_head});
    // ... 전체 순방향 패스 ...

    // 개별 해제 불필요 — arena_reset이 처리
}

arena_destroy(scratch);
```

---

## 3. 참조 카운팅

텐서가 여러 뷰 간에 공유될 때(예: 전치된 뷰와 원본), 참조 수를 추적하고 마지막 참조가 해제될 때만 기반 데이터를 해제해야 합니다.

```c
typedef struct {
    float *data;
    size_t numel;
    int    refcount;  // 멀티스레드 코드에서는 Atomic; 단일 스레드에서는 int
} TensorData;

TensorData *tensordata_new(size_t numel) {
    TensorData *d = malloc(sizeof(TensorData));
    d->data     = aligned_alloc(64, numel * sizeof(float));
    d->numel    = numel;
    d->refcount = 1;
    memset(d->data, 0, numel * sizeof(float));
    return d;
}

void tensordata_retain(TensorData *d)  { d->refcount++; }

void tensordata_release(TensorData *d) {
    if (--d->refcount == 0) {
        free(d->data);
        free(d);
    }
}

// 뷰 생성: storage 공유, 데이터 복사 없음
Tensor *tensor_view_refcounted(Tensor *src, int ndim, const size_t *shape) {
    Tensor *t  = calloc(1, sizeof(Tensor));
    t->storage = src->storage;
    tensordata_retain(t->storage);   // 참조 카운트 증가
    t->offset  = src->offset;
    // ... shape 복사, 새 strides 계산 ...
    return t;
}

void tensor_free_refcounted(Tensor *t) {
    tensordata_release(t->storage);  // 감소; 0이 되면 데이터 해제
    free(t);
}
```

---

## 4. 텐서 풀 (고정 크기 재사용)

Attention 레이어의 경우, 항상 동일한 shape의 텐서를 할당합니다(모델 설정에 의해 고정됨). 풀은 이러한 버퍼 세트를 선행 할당하고 재사용합니다:

```c
typedef struct {
    float **buffers;
    bool  *in_use;
    int    count;
    int    capacity;
    size_t buf_numel;
} TensorPool;

TensorPool *pool_create(size_t buf_numel, int capacity) {
    TensorPool *p = malloc(sizeof(TensorPool));
    p->buf_numel = buf_numel;
    p->capacity  = capacity;
    p->count     = 0;
    p->buffers   = calloc(capacity, sizeof(float *));
    p->in_use    = calloc(capacity, sizeof(bool));

    for (int i = 0; i < capacity; i++) {
        p->buffers[i] = aligned_alloc(64, buf_numel * sizeof(float));
        p->count++;
    }
    return p;
}

float *pool_acquire(TensorPool *p) {
    for (int i = 0; i < p->count; i++) {
        if (!p->in_use[i]) {
            p->in_use[i] = true;
            memset(p->buffers[i], 0, p->buf_numel * sizeof(float));
            return p->buffers[i];
        }
    }
    fprintf(stderr, "TensorPool: 소진됨 (capacity=%d)\n", p->capacity);
    return aligned_alloc(64, p->buf_numel * sizeof(float));
}

void pool_release(TensorPool *p, float *buf) {
    for (int i = 0; i < p->count; i++) {
        if (p->buffers[i] == buf) { p->in_use[i] = false; return; }
    }
    free(buf);
}
```

---

## 5. KV 캐시를 위한 메모리 레이아웃

추론에 선행 할당된 KV 캐시는 arena 할당의 고전적인 사용 사례입니다:

```c
// KV 캐시 구조체 — 모델 로드 시 한 번만 할당
typedef struct {
    float *K_cache;   // [n_layers, max_seq, n_kv_heads, d_head]
    float *V_cache;
    int    n_layers;
    int    max_seq;
    int    n_kv_heads;
    int    d_head;
    int    cur_pos;
} KVCache;

KVCache *kvcache_create(int n_layers, int max_seq, int n_kv_heads, int d_head) {
    KVCache *kv = calloc(1, sizeof(KVCache));
    kv->n_layers   = n_layers;
    kv->max_seq    = max_seq;
    kv->n_kv_heads = n_kv_heads;
    kv->d_head     = d_head;
    kv->cur_pos    = 0;

    size_t numel = (size_t)n_layers * max_seq * n_kv_heads * d_head;
    kv->K_cache  = aligned_alloc(64, numel * sizeof(float));
    kv->V_cache  = aligned_alloc(64, numel * sizeof(float));
    memset(kv->K_cache, 0, numel * sizeof(float));
    memset(kv->V_cache, 0, numel * sizeof(float));
    return kv;
}
```

이 선행 할당은 추론 중 모든 동적 할당을 제거합니다 — 지연 시간에 중요한 핵심 경로.

---

## 6. 벤치마크: malloc vs Arena

```c
void benchmark_allocation(void) {
    const int N_ALLOCS = 1000;
    const size_t TENSOR_NUMEL = 512 * 768;  // GPT-2 은닉: seq * d_model

    // malloc 기준
    double t0 = get_time_ms();
    for (int i = 0; i < N_ALLOCS; i++) {
        float *p = aligned_alloc(64, TENSOR_NUMEL * sizeof(float));
        memset(p, 0, TENSOR_NUMEL * sizeof(float));
        free(p);
    }
    double t1 = get_time_ms();

    // Arena
    Arena *arena = arena_create(TENSOR_NUMEL * N_ALLOCS * sizeof(float) + 1024*1024);
    double t2 = get_time_ms();
    for (int i = 0; i < N_ALLOCS; i++) {
        arena_reset(arena);
        float *p = arena_alloc(arena, TENSOR_NUMEL * sizeof(float), 64);
        (void)p;
    }
    double t3 = get_time_ms();
    arena_destroy(arena);

    printf("malloc: %.2f ms  (%.0f ns/alloc)\n", t1-t0, (t1-t0)*1e6/N_ALLOCS);
    printf("arena:  %.2f ms  (%.0f ns/alloc)\n", t3-t2, (t3-t2)*1e6/N_ALLOCS);
    printf("속도 향상: %.1fx\n", (t1-t0)/(t3-t2));
}
```

**예상 결과**:
```
malloc: 12.40 ms  (12400 ns/alloc)
arena:   0.08 ms  (   80 ns/alloc)
속도 향상: 155x
```

Arena는 `free()`를 절대 호출하지 않고, 락 경쟁을 피하며, TLB 워밍업의 혜택을 받기 때문에 빠릅니다.

---

## 7. GPT-2 Small(124M 파라미터) 메모리 예산

| 컴포넌트 | 크기 |
|---------|------|
| 모델 가중치 (FP32) | 124M × 4 = **496 MB** |
| 모델 가중치 (FP16) | 124M × 2 = **248 MB** |
| KV 캐시 (seq=2048, 12 레이어, 12 헤드, d=64) | 2×12×2048×12×64×4 = **72 MB** |
| 활성화 (순방향 패스 스크래치) | ~50 MB |
| **총 FP32** | **618 MB** |
| **총 FP16** | **370 MB** |

> 8 GB RAM 기기에서 GPT-2 small은 충분히 맞습니다. 7B 파라미터 모델(Llama-2-7B)의 경우: 7B × 2 = 14 GB in FP16 — 8 GB에 맞추려면 INT4 양자화 필요.

---

## 핵심 요약

- `malloc`/`free` 오버헤드는 호출당 ~100–10,000 ns; arena 할당은 ~1 ns
- **Arena**: bump pointer, O(1) 할당, O(1) 재설정 — 추론 스크래치 공간에 이상적
- **참조 카운팅**: 조기 해제나 이중 해제 없이 안전한 뷰 공유 가능
- **풀**: N개의 고정 크기 버퍼를 선행 할당하고 재사용 — 알려진 shape에 대한 반복 할당 제거
- KV 캐시는 항상 모델 로드 시 선행 할당됨 — 핫 추론 경로에서 할당 없음

---

**다음**: [08. 합성곱 밑바닥 구현](./08_Convolution_from_Scratch.md) — 2D 합성곱 구현, im2col 트릭 학습, conv가 어떻게 GEMM으로 환원되는지 이해합니다.
