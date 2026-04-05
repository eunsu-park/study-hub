# 34. FlashAttention Kernel

**Previous**: [Softmax and LayerNorm Kernels](./33_Softmax_and_LayerNorm_Kernels.md) | **Next**: [Quantized Kernels INT8](./35_Quantized_Kernels_INT8.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain why standard attention requires O(N²) HBM memory and causes OOM for long sequences
2. Describe the FlashAttention tiling strategy and why it enables O(N²/B) HBM traffic
3. Implement the FlashAttention forward kernel loop structure with online softmax accumulation
4. Apply the output rescaling step when the running max changes between K-V tile iterations
5. Implement causal masking within the FlashAttention tile loop

---

## 1. The Standard Attention Memory Problem

Standard multi-head attention computes:

```
Attention(Q, K, V) = softmax(Q·Kᵀ / √d) · V

Q: [N × d]   K: [N × d]   V: [N × d]
N = sequence length, d = head dimension

Standard implementation:
  S = Q·Kᵀ         [N × N]   ← write full N×N matrix to HBM
  P = softmax(S)    [N × N]   ← read/write N×N matrix
  O = P·V           [N × d]   ← read N×N P

HBM memory:
  N² elements = 4096² × 4 bytes = 67 MB  (per head, FP32)
  For N=16384: 67 MB × 16 = ~17 GB  ← OOM on a 40GB GPU
  Read/write ops: O(N²) — bottleneck is HBM bandwidth, not FLOPs
```

---

## 2. FlashAttention Key Insight

FlashAttention (Dao et al. 2022) tiles the computation so that the N×N attention matrix is never materialized in HBM:

```
Tiling strategy:
  Split Q into row-tiles of size Br
  Split K,V into column-tiles of size Bc
  For each Q-tile: iterate over all K,V tiles and accumulate output

  Each iteration fits in SRAM (shared memory):
    Q tile:   Br × d   (stays in SRAM for all K,V tiles)
    K,V tiles: Bc × d   (streamed from HBM one tile at a time)
    S tile:   Br × Bc  (computed on-chip, never written to HBM)

HBM complexity:
  Standard:      O(N²)    reads/writes of S matrix
  FlashAttention: O(N²/B) where B = SRAM size / (d × element_size)
                         → 10-100× less HBM traffic for large N
```

---

## 3. Online Softmax with Output Rescaling

The challenge: softmax over a full row needs the row maximum, but we process the row in tiles. Solution: maintain a running (max, sum) and rescale the output accumulator when the max changes:

```
For each K,V tile t:
  Compute S_t = Q · K_t^T   (Br × Bc raw scores)

  For each row i in the Q tile:
    m_t    = max(S_t[i, :])           (tile max)
    m_new  = max(m_old, m_t)          (updated running max)

    # Rescale old output and sum
    O[i] = O[i] * exp(m_old - m_new) + exp(S_t[i] - m_new) · V_t[i]
                  ↑ rescale old acc         ↑ new tile contribution
    l[i] = l[i] * exp(m_old - m_new) + sum(exp(S_t[i] - m_new))
    m_old = m_new

Final: O[i] = O[i] / l[i]   (normalize by total sum)
```

---

## 4. FlashAttention Forward Kernel

```c
// Simplified FlashAttention-1 forward kernel
// Q, K, V: [N × d], O: [N × d]  (single head, for clarity)
// Br: row tile size, Bc: col tile size
// In practice Br = Bc = 64-128 for typical d=64

#define BR 64   // Q tile rows
#define BC 64   // K,V tile rows
#define D  64   // head dimension

__global__ void flash_attention_fwd(
    const float *Q, const float *K, const float *V, float *O,
    int N, float scale)   // scale = 1/sqrt(d)
{
    // One block handles one Q tile of Br rows
    int q_tile = blockIdx.x;          // which row tile
    int q_start = q_tile * BR;        // first row in Q tile

    if (q_start >= N) return;
    int q_rows = min(BR, N - q_start);

    // Shared memory layout
    __shared__ float sQ[BR][D];      // Q tile (stays for all K,V iterations)
    __shared__ float sK[BC][D];      // K tile (streamed)
    __shared__ float sV[BC][D];      // V tile (streamed)
    __shared__ float sS[BR][BC];     // score tile S = Q · K^T

    int tid = threadIdx.x;

    // Per-row running statistics (registers)
    float m[BR];    // running max
    float l[BR];    // running normalizer (sum of exp)
    float o[BR][D]; // output accumulator

    for (int i = 0; i < q_rows; i++) {
        m[i] = -1e30f;
        l[i] = 0.f;
        for (int d = 0; d < D; d++) o[i][d] = 0.f;
    }

    // Load Q tile into shared memory (cooperative load)
    for (int row = 0; row < q_rows; row++) {
        for (int d = tid; d < D; d += blockDim.x)
            sQ[row][d] = Q[(q_start + row) * D + d];
    }
    __syncthreads();

    // --- Main loop: iterate over K,V tiles ---
    int n_kv_tiles = (N + BC - 1) / BC;
    for (int kv_tile = 0; kv_tile < n_kv_tiles; kv_tile++) {
        int kv_start = kv_tile * BC;
        int kv_rows  = min(BC, N - kv_start);

        // Load K tile and V tile
        for (int row = 0; row < kv_rows; row++) {
            for (int d = tid; d < D; d += blockDim.x) {
                sK[row][d] = K[(kv_start + row) * D + d];
                sV[row][d] = V[(kv_start + row) * D + d];
            }
        }
        __syncthreads();

        // Compute S = Q · K^T (Br × BC)
        // Thread tid computes one column of S (all rows, one kv-index)
        // Simplified: each thread handles one (q_row, kv_col) pair
        for (int qi = 0; qi < q_rows; qi++) {
            for (int ki = tid; ki < kv_rows; ki += blockDim.x) {
                float s = 0.f;
                for (int d = 0; d < D; d++)
                    s += sQ[qi][d] * sK[ki][d];
                sS[qi][ki] = s * scale;
            }
        }
        __syncthreads();

        // --- Online softmax update (one row at a time) ---
        // Only thread 0 does this for simplicity; in practice warp-parallelized
        if (tid == 0) {
            for (int qi = 0; qi < q_rows; qi++) {
                // Causal mask: mask out future positions
                // kv positions kv_start .. kv_start+kv_rows-1
                // Q position: q_start + qi
                // Mask if kv_pos > q_pos

                // Tile max
                float m_tile = -1e30f;
                for (int ki = 0; ki < kv_rows; ki++) {
                    // Apply causal mask
                    if (kv_start + ki > q_start + qi) {
                        sS[qi][ki] = -1e30f;  // mask to -inf
                    }
                    m_tile = fmaxf(m_tile, sS[qi][ki]);
                }

                float m_new = fmaxf(m[qi], m_tile);
                float scale_old = expf(m[qi] - m_new);

                // Rescale old accumulator
                for (int d = 0; d < D; d++)
                    o[qi][d] *= scale_old;
                l[qi] *= scale_old;

                // Accumulate new tile contribution
                float l_tile = 0.f;
                for (int ki = 0; ki < kv_rows; ki++) {
                    float p = expf(sS[qi][ki] - m_new);  // softmax numerator
                    l_tile += p;
                    for (int d = 0; d < D; d++)
                        o[qi][d] += p * sV[ki][d];
                }
                l[qi] += l_tile;
                m[qi]  = m_new;
            }
        }
        __syncthreads();
    }

    // --- Finalize: divide by l ---
    if (tid == 0) {
        for (int qi = 0; qi < q_rows; qi++) {
            float inv_l = 1.f / l[qi];
            for (int d = 0; d < D; d++)
                O[(q_start + qi) * D + d] = o[qi][d] * inv_l;
        }
    }
}
```

---

## 5. FlashAttention-2 Improvements

FlashAttention-2 (Dao 2023) makes several key improvements over FA-1:

```
FA-1 issues:
  1. Inner loop does BR iterations in thread 0 (sequential)
  2. Output rescaling done per K,V tile (expensive expf calls)
  3. Sub-optimal work partitioning across warps

FA-2 improvements:
  1. Parallelism: each warp handles a different row of Q within the tile
  2. Fewer rescalings: accumulate un-normalized O, divide by l only at the end
     (same result because: O_final = Σ_t [ P_t · V_t ] / l_total
                                     = Σ_t [ softmax_t · V_t · l_t ] / l_total)
  3. Causal masking only for boundary tiles (tiles where q_pos and kv_pos overlap)
     → saves ~half the masking overhead

FA-2 online update (simplified):
  // Instead of: o *= exp(m_old - m_new);  l *= exp(m_old - m_new)
  // Track: O_unnormalized and l separately
  // At end: O = O_unnorm / l

  float O_unnorm[D] = {0};
  float l = 0, m = -inf;
  for each tile:
    m_new  = max(m, tile_max)
    l_new  = exp(m - m_new) * l + sum(exp(S_tile - m_new))
    O_unnorm = exp(m - m_new) * O_unnorm + sum(exp(S_tile - m_new) * V_tile)
    l = l_new, m = m_new
  O_final = O_unnorm / l
```

---

## 6. IO Complexity Analysis

```
Standard attention:
  Read Q, K, V:       3 × N × d × 4 bytes
  Write S, P:         2 × N² × 4 bytes
  Read P, write O:    N² × 4 + N × d × 4 bytes
  Total HBM reads:    O(N² + Nd)

FlashAttention:
  Read Q (all K,V iterations): N × d × 4   (Q tile reused)
  Read K, V (per tile):        2 × N × d × 4 bytes total
  Write O:                     N × d × 4 bytes
  Total HBM:                   O(Nd) — no N² term!

Wall-clock speedup (A100, N=2048, d=64):
  Standard attention:     6.5 ms
  FlashAttention-1:       1.8 ms   (3.6× faster)
  FlashAttention-2:       0.9 ms   (7.2× faster)

Memory usage:
  Standard: O(N²) — 4096² × 4 bytes = 67 MB per head
  FA:       O(N)  — only Q,K,V,O tiles in SRAM
```

---

## 7. Tile Size Selection

```
SRAM budget (A100: 192 KB shared memory per SM):

For d=64, FP16:
  sQ: Br × 64 × 2 bytes
  sK: Bc × 64 × 2 bytes
  sV: Bc × 64 × 2 bytes
  sS: Br × Bc × 4 bytes (FP32 accumulation)

With Br = Bc = 64:
  sQ: 64 × 64 × 2 = 8 KB
  sK + sV: 2 × 8 KB = 16 KB
  sS: 64 × 64 × 4 = 16 KB
  Total: 40 KB  (fits in 192 KB, leaves room for other arrays)

With Br = Bc = 128:
  sQ: 32 KB, sK+sV: 64 KB, sS: 64 KB → 160 KB (tight)

Rule: Br × d + 2 × Bc × d + Br × Bc < SRAM budget
Larger tiles → fewer HBM reads per element → better bandwidth utilization
```

---

## Key Takeaways

- **Standard attention** materializes an N×N matrix in HBM, leading to O(N²) memory and bandwidth — prohibitive for N > 4K on a 40GB GPU
- **FlashAttention tiles** the K,V dimensions: for each Q-tile, stream all K,V tiles through SRAM, computing the score sub-matrix on-chip and never writing it to HBM
- **Online softmax** maintains a running (max, sum) pair; when the max increases, rescale the output accumulator and sum by `exp(m_old - m_new)` to preserve correctness
- **Output accumulator**: `O_unnorm += exp(S_tile - m_new) · V_tile`; divide by `l_final` only once at the end (FA-2 approach avoids repeated rescaling)
- **Causal masking**: set future scores to −∞ before the tile-level softmax; for tiles entirely in the past, no masking is needed (skip the branch for ~2× speedup on causal models)
- **IO complexity**: FlashAttention reduces HBM reads from O(N²) to O(Nd); for N=4096, d=64, this is a 64× reduction in HBM traffic

---

**Next**: [35. Quantized Kernels INT8](./35_Quantized_Kernels_INT8.md) — Implement INT8 quantization and dequantization, use the dp4a instruction for efficient integer dot products, and build an INT8 GEMM kernel with fused output rescaling.
