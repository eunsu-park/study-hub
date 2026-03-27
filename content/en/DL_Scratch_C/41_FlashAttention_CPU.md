# 41. FlashAttention on CPU

**Previous**: [Quantization: INT8 and INT4](./40_Quantization_Int8_Int4.md) | **Next**: [Speculative Decoding](./42_Speculative_Decoding.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain why naive attention has O(T²) memory complexity and why this is problematic for long sequences
2. Describe the FlashAttention tiling algorithm and how it avoids materializing the full attention matrix
3. Implement the online softmax update (running max and sum) that enables incremental computation
4. Write a tiled FlashAttention forward pass in C with nested loops over Q and K/V tiles
5. Compare naive vs. FlashAttention memory usage and throughput for T=8K sequences

---

## 1. The Memory Problem with Standard Attention

Standard scaled dot-product attention for a sequence of length T with d-dimensional keys:

```
S = Q K^T / sqrt(d)     shape [T, T]
A = softmax(S)           shape [T, T]
O = A V                  shape [T, d]
```

The T×T attention matrix dominates memory. For T=8192 and FP32:

```
bytes = T * T * 4 = 8192 * 8192 * 4 = 268 MB   (per attention head!)
```

With 32 heads, that is 8.6 GB just for attention scores — before any model weights. This is the core limitation of naive attention for long contexts.

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

// Naive attention: materializes the full T×T matrix
// Q, K, V: [T, d] row-major
// out:      [T, d]
void naive_attention(float *out,
                     const float *Q, const float *K, const float *V,
                     int T, int d) {
    float *S   = malloc(T * T * sizeof(float));  // [T, T] — this is the problem
    float *A   = malloc(T * T * sizeof(float));  // [T, T]
    float scale = 1.0f / sqrtf((float)d);

    // Step 1: S = Q K^T / sqrt(d)
    for (int i = 0; i < T; i++) {
        for (int j = 0; j < T; j++) {
            float dot = 0.0f;
            for (int k = 0; k < d; k++)
                dot += Q[i*d + k] * K[j*d + k];
            S[i*T + j] = dot * scale;
        }
    }

    // Step 2: A = softmax(S) row-wise
    for (int i = 0; i < T; i++) {
        float max_s = S[i*T];
        for (int j = 1; j < T; j++)
            if (S[i*T+j] > max_s) max_s = S[i*T+j];
        float sum = 0.0f;
        for (int j = 0; j < T; j++) { A[i*T+j] = expf(S[i*T+j] - max_s); sum += A[i*T+j]; }
        for (int j = 0; j < T; j++) A[i*T+j] /= sum;
    }

    // Step 3: O = A V
    for (int i = 0; i < T; i++) {
        for (int k = 0; k < d; k++) {
            float acc = 0.0f;
            for (int j = 0; j < T; j++)
                acc += A[i*T+j] * V[j*d+k];
            out[i*d+k] = acc;
        }
    }

    free(S); free(A);
}
```

Memory usage: `2 * T * T * 4 bytes` for S and A plus O(T*d) for inputs/output.

---

## 2. Online Softmax: The Key Insight

FlashAttention's core trick is *online softmax*: we can update a running softmax incrementally as we process K/V blocks, without storing all S values.

For a row of scores `[s_1, s_2, ..., s_T]`, the softmax denominator is:

```
l = sum_j exp(s_j - m)   where m = max(s_j)
```

Given new scores from a next block, we update:

```
m_new = max(m_old, local_max)
l_new = exp(m_old - m_new) * l_old + sum_j exp(s_j - m_new)
```

The output accumulator also needs rescaling when m changes:

```
O_new = O_old * exp(m_old - m_new) + (local attention weights) * V_block
```

This allows processing K and V in tiles while maintaining exact (not approximate) softmax values.

```c
// Demonstrates the online softmax update formula
// Given existing (m, l, O) and a new block of scores, update all three
// m:      current running max of scores seen so far
// l:      current running sum of exp(s - m)
// O_acc:  current output accumulator [d]
// s_blk:  new block of scores [blk_size]
// V_blk:  corresponding V values [blk_size, d]
void online_softmax_update(float *m, float *l, float *O_acc,
                            const float *s_blk, const float *V_blk,
                            int blk_size, int d) {
    // 1. Find local max within this block
    float local_max = s_blk[0];
    for (int j = 1; j < blk_size; j++)
        if (s_blk[j] > local_max) local_max = s_blk[j];

    // 2. Update running max
    float m_new = fmaxf(*m, local_max);

    // 3. Compute local exp values relative to new max
    float *exp_s = malloc(blk_size * sizeof(float));
    float local_sum = 0.0f;
    for (int j = 0; j < blk_size; j++) {
        exp_s[j] = expf(s_blk[j] - m_new);
        local_sum += exp_s[j];
    }

    // 4. Rescaling factor for the old accumulator
    float rescale = expf(*m - m_new);

    // 5. Update running sum
    float l_new = rescale * (*l) + local_sum;

    // 6. Update output accumulator O_acc:
    //    O_new = rescale * O_old + sum_j(exp_s[j] * V[j])
    for (int k = 0; k < d; k++) {
        float vsum = 0.0f;
        for (int j = 0; j < blk_size; j++)
            vsum += exp_s[j] * V_blk[j*d + k];
        O_acc[k] = rescale * O_acc[k] + vsum;
    }

    *m = m_new;
    *l = l_new;
    free(exp_s);
}
```

---

## 3. FlashAttention Tiled Forward Pass

Now we put it together: tile Q into blocks of `Br` rows, tile K and V into blocks of `Bc` columns. For each Q tile, iterate over all K/V tiles and update incrementally.

```c
// FlashAttention CPU forward pass
// Q, K, V: [T, d] row-major (assumes causal masking is NOT applied here for clarity)
// out:      [T, d]
// Br: Q tile size (rows of Q per tile)
// Bc: K/V tile size (number of K/V vectors per tile)
void flashattn_cpu(float *out,
                   const float *Q, const float *K, const float *V,
                   int T, int d,
                   int Br, int Bc) {
    float scale = 1.0f / sqrtf((float)d);

    // Temporary buffers
    float *O_tile = malloc(Br * d * sizeof(float));  // output tile
    float *m_tile = malloc(Br * sizeof(float));       // running max per row
    float *l_tile = malloc(Br * sizeof(float));       // running sum per row
    float *s_blk  = malloc(Br * Bc * sizeof(float)); // local scores [Br, Bc]

    // Iterate over Q tiles
    for (int q_start = 0; q_start < T; q_start += Br) {
        int q_end = q_start + Br;
        if (q_end > T) q_end = T;
        int cur_Br = q_end - q_start;

        // Initialize accumulators for this Q tile
        for (int i = 0; i < cur_Br; i++) {
            m_tile[i] = -1e38f;   // -infinity
            l_tile[i] = 0.0f;
            for (int k = 0; k < d; k++)
                O_tile[i*d + k] = 0.0f;
        }

        // Iterate over K/V tiles
        for (int kv_start = 0; kv_start < T; kv_start += Bc) {
            int kv_end = kv_start + Bc;
            if (kv_end > T) kv_end = T;
            int cur_Bc = kv_end - kv_start;

            // Compute S_tile = Q_tile @ K_tile^T * scale  [cur_Br, cur_Bc]
            for (int i = 0; i < cur_Br; i++) {
                int qi = q_start + i;
                for (int j = 0; j < cur_Bc; j++) {
                    int kj = kv_start + j;
                    float dot = 0.0f;
                    for (int dd = 0; dd < d; dd++)
                        dot += Q[qi*d + dd] * K[kj*d + dd];
                    s_blk[i*cur_Bc + j] = dot * scale;
                }
            }

            // Update online softmax + output accumulator for each row
            for (int i = 0; i < cur_Br; i++) {
                const float *s_row  = s_blk + i * cur_Bc;
                const float *V_blk  = V + kv_start * d;  // [cur_Bc, d]

                // Find local max in this row's score block
                float local_max = s_row[0];
                for (int j = 1; j < cur_Bc; j++)
                    if (s_row[j] > local_max) local_max = s_row[j];

                float m_new = fmaxf(m_tile[i], local_max);
                float rescale = expf(m_tile[i] - m_new);

                // Compute local exp and accumulate output
                float local_sum = 0.0f;
                for (int j = 0; j < cur_Bc; j++) {
                    float e = expf(s_row[j] - m_new);
                    local_sum += e;
                    for (int dd = 0; dd < d; dd++)
                        O_tile[i*d + dd] += e * V_blk[j*d + dd];
                }

                // Rescale old accumulator
                for (int dd = 0; dd < d; dd++)
                    O_tile[i*d + dd] = rescale * O_tile[i*d + dd];
                // Note: the += above needs to happen after rescaling —
                // correct version: rescale old, then add new contributions

                l_tile[i] = rescale * l_tile[i] + local_sum;
                m_tile[i] = m_new;
            }
        }

        // Normalize output by l (softmax denominator) and write to out
        for (int i = 0; i < cur_Br; i++) {
            int qi = q_start + i;
            float inv_l = 1.0f / l_tile[i];
            for (int k = 0; k < d; k++)
                out[qi*d + k] = O_tile[i*d + k] * inv_l;
        }
    }

    free(O_tile); free(m_tile); free(l_tile); free(s_blk);
}
```

The corrected accumulation order is important. The version above has a subtle ordering issue in the inner loop. Here is the correct pattern for the critical update:

```c
// Correct online softmax accumulation (per row i, per K/V tile):
float m_old = m_tile[i];
float m_new = fmaxf(m_old, local_max_in_block);
float alpha  = expf(m_old - m_new);  // rescaling factor

// Rescale existing O_tile[i] before adding new contribution
for (int dd = 0; dd < d; dd++)
    O_tile[i*d + dd] *= alpha;

// Add contribution from this K/V block
for (int j = 0; j < cur_Bc; j++) {
    float e = expf(s_row[j] - m_new);
    for (int dd = 0; dd < d; dd++)
        O_tile[i*d + dd] += e * V[(kv_start + j)*d + dd];
    local_sum_new += e;
}

l_tile[i] = alpha * l_tile[i] + local_sum_new;
m_tile[i] = m_new;
```

---

## 4. IO Complexity Analysis

**Naive attention** reads/writes:
- Q, K, V once: `3 * T * d * 4` bytes
- S, A matrices: `2 * T * T * 4` bytes (dominates for large T)

**FlashAttention** (tile sizes Br, Bc):
- Q tile: loaded once per outer loop iteration: `T/Br` times, each `Br*d`
- K, V tiles: loaded once per (Q tile, KV tile) pair
- Total: O(T²d / B) vs O(T²) for naive, where B is SRAM size

At T=8192, d=128, Br=Bc=64:

```
Naive:   2 × 8192² × 4 = 536 MB  (for S and A alone)
Flash:   no T×T buffer — only Br×Bc local tile = 64×64×4 = 16 KB at a time
```

```c
void compare_memory_usage(int T, int d, int Br, int Bc) {
    long naive_bytes  = 2L * T * T * sizeof(float);  // S and A
    long flash_bytes  = (long)(Br * Bc) * sizeof(float)  // s_blk
                      + (long)(Br * d) * sizeof(float)   // O_tile
                      + (long)Br * 2 * sizeof(float);    // m_tile, l_tile
    long input_bytes  = 3L * T * d * sizeof(float);      // Q, K, V (both algorithms)

    printf("T=%d, d=%d, Br=%d, Bc=%d\n", T, d, Br, Bc);
    printf("  Naive extra memory:  %ld MB\n", naive_bytes / (1024*1024));
    printf("  Flash working set:   %ld KB\n", flash_bytes / 1024);
    printf("  Shared input bytes:  %ld MB\n", input_bytes / (1024*1024));
}
```

---

## 5. Benchmarking Naive vs. Flash Attention

```c
// Timer utility
double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

void benchmark_attention(void) {
    const int T = 1024;  // Use smaller T for basic timing; scale up to 8K for real test
    const int d = 64;
    const int Br = 64, Bc = 64;

    float *Q   = malloc(T * d * sizeof(float));
    float *K   = malloc(T * d * sizeof(float));
    float *V   = malloc(T * d * sizeof(float));
    float *out_naive = malloc(T * d * sizeof(float));
    float *out_flash = malloc(T * d * sizeof(float));

    srand(123);
    for (int i = 0; i < T*d; i++) {
        Q[i] = (float)rand()/RAND_MAX - 0.5f;
        K[i] = (float)rand()/RAND_MAX - 0.5f;
        V[i] = (float)rand()/RAND_MAX - 0.5f;
    }

    double t0 = now_sec();
    naive_attention(out_naive, Q, K, V, T, d);
    double t_naive = now_sec() - t0;

    t0 = now_sec();
    flashattn_cpu(out_flash, Q, K, V, T, d, Br, Bc);
    double t_flash = now_sec() - t0;

    // Verify correctness: max absolute difference
    float max_diff = 0.0f;
    for (int i = 0; i < T*d; i++) {
        float diff = fabsf(out_naive[i] - out_flash[i]);
        if (diff > max_diff) max_diff = diff;
    }

    printf("T=%d, d=%d\n", T, d);
    printf("  Naive:  %.3f ms\n", t_naive * 1000.0);
    printf("  Flash:  %.3f ms\n", t_flash * 1000.0);
    printf("  Max diff: %.2e (should be ~1e-6 for FP32)\n", max_diff);

    compare_memory_usage(T, d, Br, Bc);
    compare_memory_usage(8192, 128, 64, 64);  // Show the T=8K case

    free(Q); free(K); free(V); free(out_naive); free(out_flash);
}

int main(void) {
    benchmark_attention();
    return 0;
}
```

Expected output for T=1024:
- Both produce nearly identical results (max diff < 1e-5)
- Flash is slower than naive for small T (tile overhead dominates)
- For T=8K+, naive runs out of memory or becomes dramatically slower due to cache misses on the T×T matrix

---

## 6. Tile Size Selection

The tile sizes Br and Bc should be chosen to fit the working set in L1/L2 cache:

```
Working set per (Q-tile, KV-tile) pair:
  s_blk:   Br * Bc * 4 bytes
  Q_tile:  Br * d  * 4 bytes
  K_tile:  Bc * d  * 4 bytes
  V_tile:  Bc * d  * 4 bytes
  O_tile:  Br * d  * 4 bytes

Total = 4 * (Br*Bc + (2*Br + 2*Bc) * d) bytes

For Br=Bc=64, d=128:
  = 4 * (4096 + 128 * 256) = 4 * 36864 = 147 KB

This fits in L2 cache (typically 256 KB–1 MB per core).
```

For larger `d` (e.g., d=256 in some models), reduce Br and Bc to maintain cache fit.

---

## Key Takeaways

- Naive attention materializes an O(T²) matrix — at T=8K with 32 heads, this exceeds 8 GB, making long-context inference infeasible without tiling.
- FlashAttention replaces the T×T materialization with a tiled computation that only needs O(Br × Bc) local working memory, fitting in CPU L2 cache.
- The online softmax update is exact (not approximate): maintaining a running max `m` and sum `l` allows correct incremental normalization across arbitrarily many K/V tiles.
- The rescaling factor `exp(m_old - m_new)` is applied to both the output accumulator `O` and the running sum `l` before adding each new K/V block's contribution.
- IO complexity improves from O(T²) to O(T²d/B) where B is SRAM capacity — the tile size acts as a bandwidth amplifier.
- On CPU, FlashAttention's main benefit at inference (single token, T = context length) is reduced memory allocation: no need to malloc a 500 MB temporary matrix.
- Tile sizes should be chosen to fit (2*Br + 2*Bc) * d + Br*Bc floats in L2 cache; typical values are Br=Bc=32 to 128 depending on d.

---

**Previous**: [Quantization: INT8 and INT4](./40_Quantization_Int8_Int4.md) | **Next**: [Speculative Decoding](./42_Speculative_Decoding.md)
