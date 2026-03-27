# 07. Memory Manager

**Previous**: [Autograd Tensor Ops](./06_Autograd_Tensor_Ops.md) | **Next**: [Convolution from Scratch](./08_Convolution_from_Scratch.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain why `malloc`/`free` is problematic for tensor-heavy workloads
2. Implement a bump-pointer arena allocator for temporary tensors
3. Implement reference counting for tensors shared across views
4. Build a tensor memory pool to reuse fixed-size buffers
5. Profile allocation overhead and measure the improvement

---

## 1. The Allocation Problem

In a transformer forward pass, hundreds of intermediate tensors are created and destroyed:

```
Q = x @ W_q        → allocate [batch, seq, d_head] × n_heads
K = x @ W_k        → allocate [batch, seq, d_head] × n_heads
V = x @ W_v        → allocate ...
attn = softmax(Q @ K^T / sqrt(d))  → allocate [batch, n_heads, seq, seq]
out  = attn @ V    → allocate ...
```

For GPT-2 (12 layers, 12 heads, seq=512):
- ~200 allocations per forward pass
- Typical `malloc` cost: **~100ns per call**
- Total overhead: ~20 μs per pass — significant for real-time inference

**Solutions**:
1. **Arena allocator**: Allocate a large block upfront; bump a pointer for each request; free all at once
2. **Tensor pool**: Reuse buffers of the same size without returning to the OS
3. **Reference counting**: Allow multiple views to share one buffer safely

---

## 2. Arena Allocator

An arena (also called a linear allocator or region allocator) is the simplest and fastest allocator:

```c
// arena.h
#pragma once
#include <stddef.h>
#include <stdbool.h>

typedef struct {
    char  *base;        // Start of memory block
    size_t capacity;    // Total size in bytes
    size_t offset;      // Current allocation cursor
    size_t peak;        // High-water mark (for profiling)
    int    n_allocs;    // Allocation count (for profiling)
} Arena;

Arena *arena_create(size_t capacity_bytes);
void  *arena_alloc (Arena *arena, size_t size, size_t alignment);
void   arena_reset (Arena *arena);   // Reset cursor to 0 — does not free memory
void   arena_destroy(Arena *arena);  // Free the backing block
void   arena_print_stats(const Arena *arena);
```

```c
// arena.c
#include "arena.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <assert.h>

Arena *arena_create(size_t capacity_bytes) {
    Arena *a = (Arena *)malloc(sizeof(Arena));
    // Allocate aligned to 64 bytes for SIMD
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
    // Align the current offset up to `alignment`
    size_t aligned = (arena->offset + alignment - 1) & ~(alignment - 1);
    assert(aligned + size <= arena->capacity && "Arena out of memory");

    void *ptr      = arena->base + aligned;
    arena->offset  = aligned + size;
    arena->n_allocs++;
    if (arena->offset > arena->peak) arena->peak = arena->offset;

    memset(ptr, 0, size);  // Zero-initialize (like calloc)
    return ptr;
}

void arena_reset(Arena *arena) {
    arena->offset  = 0;
    arena->n_allocs = 0;
    // Memory is NOT cleared — use arena_alloc's memset for safety
}

void arena_destroy(Arena *arena) {
    free(arena->base);
    free(arena);
}

void arena_print_stats(const Arena *arena) {
    printf("Arena: used=%zu KB / %zu KB  peak=%zu KB  allocs=%d\n",
           arena->offset/1024, arena->capacity/1024,
           arena->peak/1024, arena->n_allocs);
}
```

### Arena-Backed Tensor Allocation

```c
// Allocate a tensor from arena (no individual free needed)
Tensor *tensor_arena_alloc(Arena *arena, int ndim, const size_t *shape) {
    Tensor *t = (Tensor *)arena_alloc(arena, sizeof(Tensor), alignof(Tensor));
    t->ndim  = ndim;
    t->numel = 1;
    for (int i = 0; i < ndim; i++) { t->shape[i] = shape[i]; t->numel *= shape[i]; }

    // Row-major strides
    t->strides[ndim-1] = 1;
    for (int i = ndim-2; i >= 0; i--)
        t->strides[i] = t->strides[i+1] * shape[i+1];

    t->data      = (float *)arena_alloc(arena, t->numel * sizeof(float), 64);
    t->owns_data = false;  // Arena owns the memory
    return t;
}
```

**Usage pattern for inference**:
```c
// Pre-allocate 500 MB arena for a GPT-2 forward pass
Arena *scratch = arena_create(512ULL * 1024 * 1024);

for (int step = 0; step < n_steps; step++) {
    arena_reset(scratch);   // O(1) — just reset the cursor

    // Allocate all intermediates from arena
    Tensor *Q = tensor_arena_alloc(scratch, 3, (size_t[]){batch, seq, d_head});
    Tensor *K = tensor_arena_alloc(scratch, 3, (size_t[]){batch, seq, d_head});
    // ... full forward pass ...

    // No individual frees needed — arena_reset handles everything
}

arena_destroy(scratch);
```

---

## 3. Reference Counting

When a tensor is shared between multiple views (e.g., a transposed view and the original), we need to track how many references exist and free the underlying data only when the last reference is released.

```c
typedef struct {
    float *data;
    size_t numel;
    int    refcount;  // Atomic in multi-threaded code; int for single-thread
} TensorData;

typedef struct Tensor {
    TensorData *storage;        // Shared, reference-counted data block
    size_t      shape[TENSOR_MAX_DIMS];
    size_t      strides[TENSOR_MAX_DIMS];
    int         ndim;
    size_t      numel;
    size_t      offset;         // Byte offset into storage->data (for slices)
    // ... autograd fields ...
} Tensor;

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

// Create a view: shares storage, does NOT copy data
Tensor *tensor_view_refcounted(Tensor *src, int ndim, const size_t *shape) {
    Tensor *t  = calloc(1, sizeof(Tensor));
    t->storage = src->storage;
    tensordata_retain(t->storage);   // Increment ref count
    t->offset  = src->offset;
    // ... copy shape, compute new strides ...
    return t;
}

void tensor_free_refcounted(Tensor *t) {
    tensordata_release(t->storage);  // Decrement; frees data if count hits 0
    free(t);
}
```

---

## 4. Tensor Pool (Fixed-Size Reuse)

For attention layers, we always allocate tensors of the same shapes (fixed by model config). A pool pre-allocates a set of such buffers and reuses them:

```c
typedef struct {
    float **buffers;
    bool  *in_use;
    int    count;
    int    capacity;
    size_t buf_numel;   // Each buffer holds this many floats
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
    // Pool exhausted — fall back to malloc (or assert in strict mode)
    fprintf(stderr, "TensorPool: exhausted (capacity=%d)\n", p->capacity);
    return aligned_alloc(64, p->buf_numel * sizeof(float));
}

void pool_release(TensorPool *p, float *buf) {
    for (int i = 0; i < p->count; i++) {
        if (p->buffers[i] == buf) { p->in_use[i] = false; return; }
    }
    free(buf);  // Not from this pool
}
```

---

## 5. Memory Layout for KV Cache

The KV cache pre-allocated for inference is a classic use of arena allocation:

```c
// KV cache structure — allocated once at model load time
typedef struct {
    float *K_cache;   // [n_layers, max_seq, n_kv_heads, d_head]
    float *V_cache;   // Same shape
    int    n_layers;
    int    max_seq;
    int    n_kv_heads;
    int    d_head;
    int    cur_pos;   // Current filled position
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

// Access K[layer][pos][head] = K_cache + (layer * max_seq * n_kv_heads + pos * n_kv_heads + head) * d_head
float *kvcache_k_ptr(KVCache *kv, int layer, int pos, int head) {
    size_t offset = ((size_t)layer * kv->max_seq * kv->n_kv_heads +
                     (size_t)pos   * kv->n_kv_heads +
                     (size_t)head) * kv->d_head;
    return kv->K_cache + offset;
}
```

This pre-allocation eliminates all dynamic allocation during inference — the critical path for latency.

---

## 6. Benchmark: malloc vs Arena

```c
void benchmark_allocation(void) {
    const int N_ALLOCS = 1000;
    const size_t TENSOR_NUMEL = 512 * 768;  // GPT-2 hidden: seq * d_model

    // malloc baseline
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
    printf("Speedup: %.1fx\n", (t1-t0)/(t3-t2));
}
```

**Expected results**:
```
malloc: 12.40 ms  (12400 ns/alloc)
arena:   0.08 ms  (   80 ns/alloc)
Speedup: 155x
```

The arena is fast because it never calls `free()`, avoids lock contention, and benefits from TLB warm-up.

---

## 7. Memory Budget for GPT-2 Small (124M parameters)

| Component | Size |
|-----------|------|
| Model weights (FP32) | 124M × 4 = **496 MB** |
| Model weights (FP16) | 124M × 2 = **248 MB** |
| KV cache (seq=2048, 12 layers, 12 heads, d=64) | 2×12×2048×12×64×4 = **72 MB** |
| Activations (forward pass scratch) | ~50 MB |
| **Total FP32** | **618 MB** |
| **Total FP16** | **370 MB** |

> For devices with 8 GB RAM, GPT-2 small fits comfortably. For a 7B parameter model (Llama-2-7B): 7B × 2 = 14 GB in FP16 — requires INT4 quantization to fit on 8 GB.

---

## Key Takeaways

- `malloc`/`free` overhead is ~100–10,000 ns per call; arena allocation is ~1 ns
- **Arena**: bump pointer, O(1) alloc, O(1) reset — ideal for inference scratch space
- **Reference counting**: enable safe view sharing without premature or double-free
- **Pool**: pre-allocate N fixed-size buffers and reuse them — eliminates repeated allocation for known shapes
- KV cache is always pre-allocated at model load time — zero allocation during the hot inference path

---

**Next**: [08. Convolution from Scratch](./08_Convolution_from_Scratch.md) — Implement 2D convolution, learn the im2col trick, and understand how conv reduces to GEMM.
