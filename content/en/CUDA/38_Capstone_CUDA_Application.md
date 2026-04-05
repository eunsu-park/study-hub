# 38. Capstone — CUDA Application

**Previous**: [Multi-GPU and NCCL](./37_Multi_GPU_and_NCCL.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Integrate kernels from across the course into a complete, runnable application
2. Profile a multi-kernel application using `nsys` and `ncu` and interpret the results
3. Identify the dominant bottleneck (memory bandwidth, compute, or kernel launch overhead)
4. Measure end-to-end performance metrics (steps/sec, tokens/sec, MLUPS)
5. Make informed optimization decisions based on profiling data

---

## Overview

This capstone offers two projects of roughly equal complexity. Choose one based on your interest:

```
Option A: 2D Lid-Driven Cavity — LBM Fluid Simulation
  Topics covered: LBM (L24), stencil patterns (L17), streams (L12), profiling (L11)
  Output: velocity field visualization (PPM), MLUPS benchmark

Option B: LLM Inference Engine — Llama-Style Autoregressive Decoder
  Topics covered: GEMM (L32), Softmax/LayerNorm (L33), FlashAttention (L34),
                  INT8 quantization (L35), fused kernels (L36)
  Output: text generation, tokens/sec benchmark
```

---

## Option A: 2D Lid-Driven Cavity LBM Simulation

### A.1 Project Structure

```
lbm_cavity/
├── main.cu            # entry point, argument parsing
├── lbm.cuh            # kernel declarations
├── lbm_kernels.cu     # collision, streaming, BC kernels
├── macroscopic.cu     # density/velocity extraction
├── io.cu              # PPM output, CSV export
├── Makefile
└── README.md
```

### A.2 Simulation Parameters

```c
// Suggested default configuration
typedef struct {
    int   Nx, Ny;        // grid size (default 512×512)
    int   steps;         // time steps (default 10000)
    float Re;            // Reynolds number (100, 400, 1000, 3200)
    float U_lid;         // lid velocity in lattice units (default 0.1)
    int   save_stride;   // save every N steps for animation (default 500)
    int   warmup;        // warmup steps before timing (default 200)
} LBMConfig;

// Derived parameters
float compute_tau(float Re, float U_lid, int Ny) {
    float nu  = U_lid * (Ny - 1) / Re;
    float tau = 3.f * nu + 0.5f;
    printf("Re=%.0f: tau=%.4f, nu=%.6f, omega=%.4f\n",
           Re, tau, nu, 1.f/tau);
    if (tau < 0.51f) fprintf(stderr, "WARNING: tau near 0.5 — unstable!\n");
    if (tau > 1.9f)  fprintf(stderr, "WARNING: tau > 1.9 — diffusive\n");
    return tau;
}
```

### A.3 Main Simulation Loop

```c
// Complete LBM cavity simulation
int main(int argc, char **argv) {
    LBMConfig cfg = { .Nx=512, .Ny=512, .steps=10000,
                      .Re=400, .U_lid=0.1f, .save_stride=500 };
    parse_args(argc, argv, &cfg);

    float tau   = compute_tau(cfg.Re, cfg.U_lid, cfg.Ny);
    float omega = 1.f / tau;
    int   N     = cfg.Nx * cfg.Ny;

    // Allocate: 9 distributions per node, double buffered
    float *d_f0, *d_f1;
    int   *d_solid;
    float *d_ux, *d_uy, *d_rho;
    cudaMalloc(&d_f0,   9 * N * sizeof(float));
    cudaMalloc(&d_f1,   9 * N * sizeof(float));
    cudaMalloc(&d_solid,    N * sizeof(int));
    cudaMalloc(&d_ux,       N * sizeof(float));
    cudaMalloc(&d_uy,       N * sizeof(float));
    cudaMalloc(&d_rho,      N * sizeof(float));

    // Initialize equilibrium distribution (rho=1, u=0 everywhere)
    dim3 block(16, 16), grid((cfg.Nx+15)/16, (cfg.Ny+15)/16);
    init_equilibrium<<<grid, block>>>(d_f0, cfg.Nx, cfg.Ny);

    // Mark solid walls (boundary nodes)
    mark_cavity_walls<<<grid, block>>>(d_solid, cfg.Nx, cfg.Ny);
    cudaMemcpy(d_f1, d_f0, 9 * N * sizeof(float), cudaMemcpyDeviceToDevice);

    // CUDA streams: collision and BC can overlap with next-step prefetch
    cudaStream_t stream_comp, stream_io;
    cudaStreamCreate(&stream_comp);
    cudaStreamCreate(&stream_io);

    // Warmup
    for (int t = 0; t < cfg.warmup; t++) {
        lbm_collision<<<grid, block, 0, stream_comp>>>(d_f0, d_solid, cfg.Nx, cfg.Ny, omega);
        lbm_streaming<<<grid, block, 0, stream_comp>>>(d_f0, d_f1, d_solid, cfg.Nx, cfg.Ny);
        zou_he_moving_lid<<<(cfg.Nx+255)/256, 256, 0, stream_comp>>>(
            d_f1, cfg.Nx, cfg.Ny, cfg.U_lid);
        float *tmp = d_f0; d_f0 = d_f1; d_f1 = tmp;
    }

    // Timed run
    cudaEvent_t t_start, t_stop;
    cudaEventCreate(&t_start); cudaEventCreate(&t_stop);
    cudaEventRecord(t_start, stream_comp);

    for (int t = 0; t < cfg.steps; t++) {
        lbm_collision<<<grid, block, 0, stream_comp>>>(d_f0, d_solid, cfg.Nx, cfg.Ny, omega);
        lbm_streaming<<<grid, block, 0, stream_comp>>>(d_f0, d_f1, d_solid, cfg.Nx, cfg.Ny);
        zou_he_moving_lid<<<(cfg.Nx+255)/256, 256, 0, stream_comp>>>(
            d_f1, cfg.Nx, cfg.Ny, cfg.U_lid);
        float *tmp = d_f0; d_f0 = d_f1; d_f1 = tmp;

        // Periodic visualization output
        if (t % cfg.save_stride == 0) {
            lbm_macroscopic<<<grid, block, 0, stream_comp>>>(
                d_f0, d_solid, d_rho, d_ux, d_uy, cfg.Nx, cfg.Ny);
            save_ppm_async(d_ux, d_uy, cfg.Nx, cfg.Ny, t, stream_io);
        }
    }
    cudaEventRecord(t_stop, stream_comp);
    cudaDeviceSynchronize();

    float elapsed_ms;
    cudaEventElapsedTime(&elapsed_ms, t_start, t_stop);
    double mlups = (double)cfg.steps * N / (elapsed_ms * 1e3);
    printf("Steps: %d, Grid: %d×%d, Time: %.2f ms\n",
           cfg.steps, cfg.Nx, cfg.Ny, elapsed_ms);
    printf("Performance: %.2f MLUPS (million lattice updates/second)\n", mlups);

    // Final visualization
    lbm_macroscopic<<<grid, block>>>(d_f0, d_solid, d_rho, d_ux, d_uy, cfg.Nx, cfg.Ny);
    save_velocity_csv(d_ux, d_uy, cfg.Nx, cfg.Ny, "velocity_final.csv");

    cudaFree(d_f0); cudaFree(d_f1); cudaFree(d_solid);
    cudaFree(d_ux); cudaFree(d_uy); cudaFree(d_rho);
    return 0;
}
```

### A.4 Performance Targets and Profiling

```
Expected performance (512×512 grid):
  RTX 3090:  ~18 MLUPS
  A100:      ~55 MLUPS
  CPU single thread: ~0.5 MLUPS → GPU speedup: ~100×

Profiling checklist:
  nsys profile ./lbm_cavity
    → Check: are collision and streaming back-to-back or pipelined?
    → Check: is IO stream overlapping compute?

  ncu --metrics gpu__time_duration.sum,
               l1tex__t_bytes.sum,
               sm__throughput.avg.pct_of_peak_sustained_elapsed
      ./lbm_cavity --steps 100

  Expected: lbm_streaming is memory-bandwidth bound (~85% peak bandwidth)
            lbm_collision is compute-bound for small grids

Validation:
  Re=100:  steady-state, single center vortex
  Re=400:  steady-state, asymmetric vortex pattern
  Re=1000: transient to weakly periodic
  Re=3200: unsteady, compare centerline u_x(0.5, y) to Ghia (1982) benchmark data
```

---

## Option B: Llama-Style LLM Inference Engine

### B.1 Project Structure

```
llm_infer/
├── main.cu              # tokenizer, generation loop
├── model.cuh            # weight layout, config
├── kernels/
│   ├── embedding.cu     # token → embedding lookup
│   ├── attention.cu     # FlashAttention forward (from L34)
│   ├── ffn.cu           # FFN: gate_proj, up_proj, down_proj (SwiGLU)
│   ├── layernorm.cu     # RMSNorm (from L33)
│   ├── gemm.cu          # cuBLAS GEMM wrapper
│   └── sampling.cu      # greedy argmax / temperature sampling
├── weights/             # binary weight files
├── tokenizer/           # BPE tokenizer (simplified)
└── Makefile
```

### B.2 Model Configuration (Llama-Tiny for Testing)

```c
typedef struct {
    int vocab_size;    // 32000 (Llama tokenizer)
    int n_layers;      // 8 (tiny) or 32 (7B)
    int d_model;       // 256 (tiny) or 4096 (7B)
    int n_heads;       // 4 (tiny)  or 32 (7B)
    int d_head;        // d_model / n_heads = 64 (7B: 128)
    int d_ffn;         // 4 × d_model (SwiGLU: 2/3 × 4 × d_model)
    int max_seq_len;   // 2048
} LlamaConfig;

// Llama-tiny for development (fits in 500MB GPU memory):
// vocab=32000, layers=8, d=256, heads=4, d_head=64, d_ffn=688
```

### B.3 KV Cache

```c
// Pre-allocate KV cache for all layers: shape [n_layers × max_seq × n_heads × d_head]
// Avoids recomputing past K,V for autoregressive generation

typedef struct {
    float *k_cache;  // [n_layers × max_seq × d_kv]
    float *v_cache;  // [n_layers × max_seq × d_kv]
    int    seq_len;  // current number of cached tokens
    int    max_len;  // maximum cache length
} KVCache;

void init_kv_cache(KVCache *kv, const LlamaConfig *cfg) {
    size_t size = (size_t)cfg->n_layers * cfg->max_seq_len
                * cfg->n_heads * cfg->d_head * sizeof(float);
    cudaMalloc(&kv->k_cache, size);
    cudaMalloc(&kv->v_cache, size);
    kv->seq_len = 0;
    kv->max_len = cfg->max_seq_len;
}
```

### B.4 Transformer Block

```c
// Single transformer decoder block
void transformer_block(
    float *x,           // [1 × d_model] current token state
    const Weights *w,   // weight pointers for this layer
    KVCache *kv,
    cublasHandle_t blas_handle,
    int layer, int pos, const LlamaConfig *cfg)
{
    // 1. RMSNorm before attention
    rmsnorm_forward<<<1, 256>>>(x, w->rms_att, x_norm, cfg->d_model, 1e-5f);

    // 2. QKV projections (3 GEMMs, or one fused)
    cublasSgemm_col_major(blas_handle, x_norm, w->Wq, q, 1, cfg->d_model, cfg->d_model);
    cublasSgemm_col_major(blas_handle, x_norm, w->Wk, k_new, 1, cfg->d_model, cfg->d_head * cfg->n_heads);
    cublasSgemm_col_major(blas_handle, x_norm, w->Wv, v_new, 1, cfg->d_model, cfg->d_head * cfg->n_heads);

    // 3. Apply RoPE (rotary position embedding) to Q and K
    apply_rope<<<1, cfg->n_heads>>>(q, k_new, pos, cfg->d_head);

    // 4. Update KV cache
    float *k_ptr = kv->k_cache + layer * cfg->max_seq_len * cfg->n_heads * cfg->d_head
                               + pos * cfg->n_heads * cfg->d_head;
    cudaMemcpy(k_ptr, k_new, cfg->n_heads * cfg->d_head * sizeof(float),
               cudaMemcpyDeviceToDevice);
    // same for v...

    // 5. FlashAttention (causal, pos+1 tokens)
    flash_attention_decode<<<cfg->n_heads, 256>>>(
        q, kv->k_cache + layer * ..., kv->v_cache + layer * ...,
        att_out, pos + 1, cfg->d_head, 1.f / sqrtf(cfg->d_head));

    // 6. Output projection
    cublasSgemm_col_major(blas_handle, att_out, w->Wo, attn_proj, 1,
                          cfg->n_heads * cfg->d_head, cfg->d_model);

    // 7. Residual add
    vector_add_inplace<<<(cfg->d_model+255)/256, 256>>>(x, attn_proj, cfg->d_model);

    // 8. RMSNorm before FFN
    rmsnorm_forward<<<1, 256>>>(x, w->rms_ffn, x_norm, cfg->d_model, 1e-5f);

    // 9. SwiGLU FFN: out = (gate(x_norm) ⊙ SiLU(gate(x_norm))) * W_down
    //   SwiGLU: gate = x_norm @ W_gate, up = x_norm @ W_up
    //           h = SiLU(gate) * up
    //           out = h @ W_down
    cublasSgemm_col_major(blas_handle, x_norm, w->W_gate, gate, 1, cfg->d_model, cfg->d_ffn);
    cublasSgemm_col_major(blas_handle, x_norm, w->W_up,   up,   1, cfg->d_model, cfg->d_ffn);
    silu_mul_inplace<<<(cfg->d_ffn+255)/256, 256>>>(gate, up, cfg->d_ffn);  // gate *= SiLU(gate) * up
    cublasSgemm_col_major(blas_handle, gate, w->W_down, ffn_out, 1, cfg->d_ffn, cfg->d_model);

    // 10. Residual add
    vector_add_inplace<<<(cfg->d_model+255)/256, 256>>>(x, ffn_out, cfg->d_model);
}
```

### B.5 Generation Loop and Benchmark

```c
void generate(const LlamaModel *model, const int *prompt_tokens,
              int prompt_len, int max_new_tokens) {
    KVCache kv;
    init_kv_cache(&kv, &model->cfg);

    // Prefill: process prompt tokens in parallel (batch GEMM over prompt)
    // Simplified: process token by token
    float *d_logits;
    cudaMalloc(&d_logits, model->cfg.vocab_size * sizeof(float));

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);

    // Prefill phase
    for (int i = 0; i < prompt_len; i++) {
        embed_token<<<1, 256>>>(prompt_tokens[i], model->embedding, d_x, model->cfg.d_model);
        for (int l = 0; l < model->cfg.n_layers; l++)
            transformer_block(d_x, &model->weights[l], &kv, model->blas, l, i, &model->cfg);
        kv.seq_len++;
    }

    // Generation phase (autoregressive)
    int next_token = prompt_tokens[prompt_len - 1];
    cudaEventRecord(t0);

    for (int t = 0; t < max_new_tokens; t++) {
        embed_token<<<1, 256>>>(next_token, model->embedding, d_x, model->cfg.d_model);

        for (int l = 0; l < model->cfg.n_layers; l++)
            transformer_block(d_x, &model->weights[l], &kv, model->blas,
                              l, kv.seq_len, &model->cfg);

        // Final RMSNorm + unembedding
        rmsnorm_forward<<<1, 256>>>(d_x, model->rms_final, d_x_norm,
                                    model->cfg.d_model, 1e-5f);
        cublasSgemm_col_major(model->blas, d_x_norm, model->unembedding,
                              d_logits, 1, model->cfg.d_model, model->cfg.vocab_size);

        // Greedy sampling: argmax over vocab
        next_token = argmax_kernel(d_logits, model->cfg.vocab_size);
        kv.seq_len++;

        // Decode and print token
        const char *text = tokenizer_decode(next_token);
        printf("%s", text); fflush(stdout);

        if (next_token == EOS_TOKEN) break;
    }
    cudaEventRecord(t1);
    cudaDeviceSynchronize();

    float ms;
    cudaEventElapsedTime(&ms, t0, t1);
    printf("\n\nThroughput: %.1f tokens/sec\n", max_new_tokens / (ms * 1e-3f));
}
```

### B.6 Performance Targets

```
Llama-tiny (8 layers, d=256, 4 heads) on RTX 3090:
  Target: > 5000 tokens/sec (single-batch decode)
  Memory: ~50 MB (model) + ~20 MB (KV cache @ 2048 tokens)

Llama-7B (32 layers, d=4096, 32 heads) on A100 80GB FP16:
  Target: ~100 tokens/sec (single-batch greedy)
  Memory: 14 GB (weights) + 2 GB (KV cache @ 2048 tokens)
  Bottleneck: weight loading bandwidth (not FLOPs!) at decode time

INT8 quantized (W8A16):
  Memory: 7 GB (weights) → more KV cache space
  Target: ~180 tokens/sec (+80% from 2× fewer bytes/weight)
```

---

## Profiling Checklist (Both Projects)

```bash
# 1. nsys: high-level timeline
nsys profile --stats=true -o profile_report ./my_app [args]
nsys stats profile_report.nsys-rep  # print summary table

# 2. ncu: detailed kernel metrics
ncu --target-processes all \
    --metrics gpu__time_duration.sum,\
              l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum,\
              sm__throughput.avg.pct_of_peak_sustained_elapsed,\
              sm__inst_executed_pipe_fp32.avg \
    -o ncu_report ./my_app --steps 10

# 3. Key questions to answer:
#   - What fraction of time is in kernels vs memory copies?
#   - Which kernel takes the most time?
#   - Is the bottleneck kernel memory-bound or compute-bound?
#   - Is occupancy limiting performance?
#   - Are there unnecessary kernel launches (< 1 μs)?

# 4. Roofline analysis (ncu --set roofline)
ncu --set roofline -o roofline_report ./my_app --steps 10
ncu-ui roofline_report.ncu-rep   # open GUI with roofline chart
```

---

## Key Takeaways

- **Project A (LBM)**: streaming is the bottleneck kernel — bandwidth-bound; use SoA layout, pull scheme, and overlap IO with compute using separate CUDA streams
- **Project B (LLM)**: decode phase is dominated by weight loading (memory-bandwidth bound) not GEMM compute; INT8 quantization directly improves tokens/sec proportional to bandwidth savings
- **Profile before optimizing**: measure first with nsys for timeline, then ncu for kernel-level metrics; optimize the kernel that actually dominates runtime
- **Correctness first**: verify LBM against Ghia benchmark data (Re=100, 400, 1000); verify LLM against a reference CPU implementation for a fixed prompt and seed
- **Real-world engineering**: most GPU application time is in setup, data loading, and synchronization — not just the core kernel; profile the whole application, not just the hot loop
- **The complete journey**: from thread indexing (L2) → memory hierarchy (L4-L5) → warp execution (L6) → profiling (L11) → reductions (L14) → domain algorithms (L20-L25) → DL kernels (L32-L36) — all concepts build toward efficient, production-quality CUDA applications

---

**Course Complete!** You have now covered the full spectrum of CUDA GPU programming, from architecture fundamentals to production deep learning kernels. The skills in this course map directly to GPU systems engineering, ML infrastructure, and high-performance scientific computing.
