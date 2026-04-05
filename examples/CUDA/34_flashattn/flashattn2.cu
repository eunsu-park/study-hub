/*
 * flashattn2.cu — Lesson 34: FlashAttention Kernel
 *
 * Implements a simplified FlashAttention-2 forward pass for a single head:
 *   Attention(Q, K, V) = softmax(Q·K^T / sqrt(d)) · V
 *
 * Key ideas demonstrated:
 *   - Tiled computation to avoid O(N²) HBM materialisation of the attention matrix
 *   - Online softmax (max + logsumexp running update per tile)
 *   - Rescaling of accumulated output when new max is found
 *
 * For simplicity: single batch, single head, FP32.
 * Production: use FlashAttention-2 library or Triton kernel.
 *
 * Build:  nvcc -O2 -arch=sm_80 flashattn2.cu -o flashattn2
 * Run:    ./flashattn2
 */

#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>

#define CUDA_CHECK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){ \
    fprintf(stderr,"CUDA %s\n",cudaGetErrorString(e)); exit(1); } } while(0)
#define FULL_MASK 0xffffffff

static const int SEQ  = 512;     // sequence length
static const int DIM  = 64;      // head dimension
static const int BQ   = 32;      // query block size
static const int BKV  = 32;      // key/value block size

// ── Naive O(N²) attention (reference) ────────────────────────────────────────
__global__ void naive_attention(const float *Q, const float *K, const float *V,
                                 float *O, int n, int d) {
    extern __shared__ float smem[];
    int qi = blockIdx.x;     // one block per query
    int tid = threadIdx.x;
    if (qi >= n) return;

    float *s_attn = smem;          // [n] attention scores
    float *s_val  = smem + n;      // [d] output accumulator

    // Compute attention scores
    float row_max = -1e30f;
    for (int ki = tid; ki < n; ki += blockDim.x) {
        float dot = 0.f;
        for (int dk = 0; dk < d; dk++)
            dot += Q[qi*d+dk] * K[ki*d+dk];
        s_attn[ki] = dot / sqrtf((float)d);
        row_max = fmaxf(row_max, s_attn[ki]);
    }
    // Simple allreduce for max
    __syncthreads();
    if (tid == 0) {
        float gmax = s_attn[0];
        for (int i = 1; i < n; i++) gmax = fmaxf(gmax, s_attn[i]);
        s_val[0] = gmax;   // reuse s_val[0] temporarily
    }
    __syncthreads();
    float gmax = s_val[0];

    float sum_exp = 0.f;
    for (int ki = tid; ki < n; ki += blockDim.x) {
        s_attn[ki] = expf(s_attn[ki] - gmax);
        sum_exp += s_attn[ki];
    }
    __syncthreads();
    if (tid == 0) { s_val[0] = 0.f; for (int i = 0; i < n; i++) s_val[0] += s_attn[i]; }
    __syncthreads();
    sum_exp = s_val[0];

    // Compute output
    for (int dk = tid; dk < d; dk += blockDim.x) {
        float acc = 0.f;
        for (int ki = 0; ki < n; ki++)
            acc += (s_attn[ki] / sum_exp) * V[ki*d+dk];
        O[qi*d+dk] = acc;
    }
}

// ── FlashAttention-2 forward (tiled, O(N) HBM) ────────────────────────────────
// Each block processes BQ queries. For each query, iterates over KV tiles,
// maintaining running max (mi) and normaliser (li) for online softmax.
__global__ void flash_attention(const float *Q, const float *K, const float *V,
                                  float *O, int n, int d) {
    __shared__ float sK[BKV][64];   // tile of K  (BKV × d, d≤64)
    __shared__ float sV[BKV][64];   // tile of V
    __shared__ float sQ[BQ][64];    // tile of Q (each block handles BQ queries)

    int qi_base = blockIdx.x * BQ;
    int qi      = qi_base + threadIdx.y;   // row this thread handles
    int dk      = threadIdx.x;            // dimension index (0..d-1)

    // Load Q tile into shared memory
    if (qi < n && dk < d) sQ[threadIdx.y][dk] = Q[qi*d+dk];
    __syncthreads();

    float mi = -1e30f;   // running max
    float li = 0.f;      // running sum(exp)
    float oi = 0.f;      // running output accumulator for dimension dk

    for (int kv_start = 0; kv_start < n; kv_start += BKV) {
        // Load KV tile (thread (0,dk) loads one row)
        int kv_idx = kv_start + threadIdx.y;
        if (kv_idx < n && dk < d) {
            sK[threadIdx.y][dk] = K[kv_idx*d+dk];
            sV[threadIdx.y][dk] = V[kv_idx*d+dk];
        }
        __syncthreads();

        if (qi < n) {
            // Compute attention scores for this tile
            float scores[BKV];
            float tile_max = -1e30f;
            #pragma unroll
            for (int j = 0; j < BKV && kv_start+j < n; j++) {
                float dot = 0.f;
                for (int t = 0; t < d; t++) dot += sQ[threadIdx.y][t] * sK[j][t];
                scores[j] = dot / sqrtf((float)d);
                tile_max   = fmaxf(tile_max, scores[j]);
            }

            // Update running max and rescale accumulator
            float new_mi = fmaxf(mi, tile_max);
            float rescale = expf(mi - new_mi);
            oi *= rescale;
            li *= rescale;

            // Accumulate
            for (int j = 0; j < BKV && kv_start+j < n; j++) {
                float e = expf(scores[j] - new_mi);
                oi += e * sV[j][dk];
                li += e;
            }
            mi = new_mi;
        }
        __syncthreads();
    }

    if (qi < n && dk < d)
        O[qi*d+dk] = oi / li;
}

int main(void) {
    const size_t bytes = (size_t)SEQ * DIM * sizeof(float);

    float *h_Q = (float *)malloc(bytes);
    float *h_K = (float *)malloc(bytes);
    float *h_V = (float *)malloc(bytes);
    float *h_O = (float *)malloc(bytes);
    float *h_ref = (float *)malloc(bytes);
    srand(42);
    for (int i = 0; i < SEQ * DIM; i++) {
        h_Q[i] = ((float)rand()/RAND_MAX - 0.5f) * 0.5f;
        h_K[i] = ((float)rand()/RAND_MAX - 0.5f) * 0.5f;
        h_V[i] = ((float)rand()/RAND_MAX - 0.5f) * 0.5f;
    }

    float *d_Q, *d_K, *d_V, *d_O;
    CUDA_CHECK(cudaMalloc(&d_Q, bytes)); CUDA_CHECK(cudaMalloc(&d_K, bytes));
    CUDA_CHECK(cudaMalloc(&d_V, bytes)); CUDA_CHECK(cudaMalloc(&d_O, bytes));
    CUDA_CHECK(cudaMemcpy(d_Q, h_Q, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_K, h_K, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_V, h_V, bytes, cudaMemcpyHostToDevice));

    // Naive reference
    size_t smem_naive = ((size_t)SEQ + DIM) * sizeof(float);
    naive_attention<<<SEQ, 32, smem_naive>>>(d_Q, d_K, d_V, d_O, SEQ, DIM);
    CUDA_CHECK(cudaMemcpy(h_ref, d_O, bytes, cudaMemcpyDeviceToHost));

    // FlashAttention
    dim3 block(DIM, BQ);
    dim3 grid((SEQ + BQ - 1) / BQ);
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0);
    flash_attention<<<grid, block>>>(d_Q, d_K, d_V, d_O, SEQ, DIM);
    cudaEventRecord(t1); cudaEventSynchronize(t1);
    float ms; cudaEventElapsedTime(&ms, t0, t1);

    CUDA_CHECK(cudaMemcpy(h_O, d_O, bytes, cudaMemcpyDeviceToHost));
    float max_err = 0.f;
    for (int i = 0; i < SEQ * DIM; i++)
        max_err = fmaxf(max_err, fabsf(h_O[i] - h_ref[i]));

    printf("FlashAttention-2 (seq=%d, d=%d)\n", SEQ, DIM);
    printf("  Time: %.3f ms  max_err=%.2e  %s\n",
           ms, max_err, max_err < 1e-3f ? "PASS" : "FAIL");

    cudaEventDestroy(t0); cudaEventDestroy(t1);
    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_O);
    free(h_Q); free(h_K); free(h_V); free(h_O); free(h_ref);
    return 0;
}
