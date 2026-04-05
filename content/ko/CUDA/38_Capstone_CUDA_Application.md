# 38. Capstone — CUDA Application

**이전**: [Multi-GPU and NCCL](./37_Multi_GPU_and_NCCL.md)

---

## 학습 목표

이 레슨을 마치면 다음을 할 수 있습니다:

1. 코스 전반의 kernel을 완전히 실행 가능한 애플리케이션으로 통합하기
2. `nsys`와 `ncu`를 사용하여 멀티 kernel 애플리케이션 프로파일링하고 결과 해석하기
3. 지배적인 병목 (메모리 대역폭, 연산, 또는 kernel 실행 오버헤드) 식별하기
4. 종단간 성능 지표 측정하기 (steps/sec, tokens/sec, MLUPS)
5. 프로파일링 데이터를 기반으로 정보에 입각한 최적화 결정 내리기

---

## 개요

이 캡스톤은 대략 동등한 복잡도의 두 가지 프로젝트를 제공합니다. 관심에 따라 하나를 선택하세요:

```
옵션 A: 2D Lid-Driven Cavity — LBM 유체 시뮬레이션
  다루는 토픽: LBM (L24), stencil 패턴 (L17), stream (L12), 프로파일링 (L11)
  출력: 속도장 시각화 (PPM), MLUPS 벤치마크

옵션 B: LLM 추론 엔진 — Llama 스타일 자기회귀 디코더
  다루는 토픽: GEMM (L32), Softmax/LayerNorm (L33), FlashAttention (L34),
               INT8 양자화 (L35), 융합 kernel (L36)
  출력: 텍스트 생성, tokens/sec 벤치마크
```

---

## 옵션 A: 2D Lid-Driven Cavity LBM 시뮬레이션

### A.1 프로젝트 구조

```
lbm_cavity/
├── main.cu            # 진입점, 인수 파싱
├── lbm.cuh            # kernel 선언
├── lbm_kernels.cu     # collision, streaming, BC kernel
├── macroscopic.cu     # 밀도/속도 추출
├── io.cu              # PPM 출력, CSV 내보내기
├── Makefile
└── README.md
```

### A.2 시뮬레이션 파라미터

```c
// 권장 기본 설정
typedef struct {
    int   Nx, Ny;        // 격자 크기 (기본값 512×512)
    int   steps;         // 시간 단계 수 (기본값 10000)
    float Re;            // Reynolds 수 (100, 400, 1000, 3200)
    float U_lid;         // 격자 단위에서의 뚜껑 속도 (기본값 0.1)
    int   save_stride;   // 애니메이션을 위해 N 단계마다 저장 (기본값 500)
    int   warmup;        // 타이밍 전 워밍업 단계 (기본값 200)
} LBMConfig;

// 유도 파라미터
float compute_tau(float Re, float U_lid, int Ny) {
    float nu  = U_lid * (Ny - 1) / Re;
    float tau = 3.f * nu + 0.5f;
    printf("Re=%.0f: tau=%.4f, nu=%.6f, omega=%.4f\n",
           Re, tau, nu, 1.f/tau);
    if (tau < 0.51f) fprintf(stderr, "경고: tau가 0.5에 근접 — 불안정!\n");
    if (tau > 1.9f)  fprintf(stderr, "경고: tau > 1.9 — 확산성\n");
    return tau;
}
```

### A.3 메인 시뮬레이션 루프

```c
// 완전한 LBM cavity 시뮬레이션
int main(int argc, char **argv) {
    LBMConfig cfg = { .Nx=512, .Ny=512, .steps=10000,
                      .Re=400, .U_lid=0.1f, .save_stride=500 };
    parse_args(argc, argv, &cfg);

    float tau   = compute_tau(cfg.Re, cfg.U_lid, cfg.Ny);
    float omega = 1.f / tau;
    int   N     = cfg.Nx * cfg.Ny;

    // 할당: 노드당 9개 분포, 이중 버퍼링
    float *d_f0, *d_f1;
    int   *d_solid;
    float *d_ux, *d_uy, *d_rho;
    cudaMalloc(&d_f0,   9 * N * sizeof(float));
    cudaMalloc(&d_f1,   9 * N * sizeof(float));
    cudaMalloc(&d_solid,    N * sizeof(int));
    cudaMalloc(&d_ux,       N * sizeof(float));
    cudaMalloc(&d_uy,       N * sizeof(float));
    cudaMalloc(&d_rho,      N * sizeof(float));

    // 평형 분포 초기화 (rho=1, u=0 everywhere)
    dim3 block(16, 16), grid((cfg.Nx+15)/16, (cfg.Ny+15)/16);
    init_equilibrium<<<grid, block>>>(d_f0, cfg.Nx, cfg.Ny);

    // 고체 벽 표시 (경계 노드)
    mark_cavity_walls<<<grid, block>>>(d_solid, cfg.Nx, cfg.Ny);
    cudaMemcpy(d_f1, d_f0, 9 * N * sizeof(float), cudaMemcpyDeviceToDevice);

    // CUDA stream: collision과 BC가 다음 단계 프리페치와 오버랩 가능
    cudaStream_t stream_comp, stream_io;
    cudaStreamCreate(&stream_comp);
    cudaStreamCreate(&stream_io);

    // 워밍업
    for (int t = 0; t < cfg.warmup; t++) {
        lbm_collision<<<grid, block, 0, stream_comp>>>(d_f0, d_solid, cfg.Nx, cfg.Ny, omega);
        lbm_streaming<<<grid, block, 0, stream_comp>>>(d_f0, d_f1, d_solid, cfg.Nx, cfg.Ny);
        zou_he_moving_lid<<<(cfg.Nx+255)/256, 256, 0, stream_comp>>>(
            d_f1, cfg.Nx, cfg.Ny, cfg.U_lid);
        float *tmp = d_f0; d_f0 = d_f1; d_f1 = tmp;
    }

    // 타이밍 실행
    cudaEvent_t t_start, t_stop;
    cudaEventCreate(&t_start); cudaEventCreate(&t_stop);
    cudaEventRecord(t_start, stream_comp);

    for (int t = 0; t < cfg.steps; t++) {
        lbm_collision<<<grid, block, 0, stream_comp>>>(d_f0, d_solid, cfg.Nx, cfg.Ny, omega);
        lbm_streaming<<<grid, block, 0, stream_comp>>>(d_f0, d_f1, d_solid, cfg.Nx, cfg.Ny);
        zou_he_moving_lid<<<(cfg.Nx+255)/256, 256, 0, stream_comp>>>(
            d_f1, cfg.Nx, cfg.Ny, cfg.U_lid);
        float *tmp = d_f0; d_f0 = d_f1; d_f1 = tmp;

        // 주기적 시각화 출력
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
    printf("단계: %d, 격자: %d×%d, 시간: %.2f ms\n",
           cfg.steps, cfg.Nx, cfg.Ny, elapsed_ms);
    printf("성능: %.2f MLUPS (초당 백만 격자 업데이트)\n", mlups);

    // 최종 시각화
    lbm_macroscopic<<<grid, block>>>(d_f0, d_solid, d_rho, d_ux, d_uy, cfg.Nx, cfg.Ny);
    save_velocity_csv(d_ux, d_uy, cfg.Nx, cfg.Ny, "velocity_final.csv");

    cudaFree(d_f0); cudaFree(d_f1); cudaFree(d_solid);
    cudaFree(d_ux); cudaFree(d_uy); cudaFree(d_rho);
    return 0;
}
```

### A.4 성능 목표 및 프로파일링

```
예상 성능 (512×512 격자):
  RTX 3090:  ~18 MLUPS
  A100:      ~55 MLUPS
  CPU 단일 thread: ~0.5 MLUPS → GPU 속도 향상: ~100×

프로파일링 체크리스트:
  nsys profile ./lbm_cavity
    → 확인: collision과 streaming이 연속적인가, 파이프라인화되었는가?
    → 확인: IO stream이 연산과 오버랩되는가?

  ncu --metrics gpu__time_duration.sum,
               l1tex__t_bytes.sum,
               sm__throughput.avg.pct_of_peak_sustained_elapsed
      ./lbm_cavity --steps 100

  예상: lbm_streaming은 메모리 대역폭 병목 (~85% 최대 대역폭)
        lbm_collision은 소형 격자에서 연산 병목

검증:
  Re=100:  정상 상태, 단일 중앙 소용돌이
  Re=400:  정상 상태, 비대칭 소용돌이 패턴
  Re=1000: 약한 주기적으로 전환
  Re=3200: 비정상, Ghia (1982) 벤치마크 데이터와 중심선 u_x(0.5, y) 비교
```

---

## 옵션 B: Llama 스타일 LLM 추론 엔진

### B.1 프로젝트 구조

```
llm_infer/
├── main.cu              # 토크나이저, 생성 루프
├── model.cuh            # 가중치 레이아웃, 설정
├── kernels/
│   ├── embedding.cu     # 토큰 → embedding 조회
│   ├── attention.cu     # FlashAttention 순전파 (L34에서)
│   ├── ffn.cu           # FFN: gate_proj, up_proj, down_proj (SwiGLU)
│   ├── layernorm.cu     # RMSNorm (L33에서)
│   ├── gemm.cu          # cuBLAS GEMM 래퍼
│   └── sampling.cu      # 탐욕적 argmax / 온도 샘플링
├── weights/             # 이진 가중치 파일
├── tokenizer/           # BPE 토크나이저 (단순화)
└── Makefile
```

### B.2 모델 설정 (테스트용 Llama-Tiny)

```c
typedef struct {
    int vocab_size;    // 32000 (Llama 토크나이저)
    int n_layers;      // 8 (tiny) 또는 32 (7B)
    int d_model;       // 256 (tiny) 또는 4096 (7B)
    int n_heads;       // 4 (tiny) 또는 32 (7B)
    int d_head;        // d_model / n_heads = 64 (7B: 128)
    int d_ffn;         // 4 × d_model (SwiGLU: 2/3 × 4 × d_model)
    int max_seq_len;   // 2048
} LlamaConfig;

// 개발용 Llama-tiny (500MB GPU 메모리에 맞음):
// vocab=32000, layers=8, d=256, heads=4, d_head=64, d_ffn=688
```

### B.3 KV 캐시

```c
// 모든 레이어에 대해 KV 캐시 사전 할당: 형태 [n_layers × max_seq × n_heads × d_head]
// 자기회귀 생성 시 이전 K,V 재계산 방지

typedef struct {
    float *k_cache;  // [n_layers × max_seq × d_kv]
    float *v_cache;  // [n_layers × max_seq × d_kv]
    int    seq_len;  // 현재 캐시된 토큰 수
    int    max_len;  // 최대 캐시 길이
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

### B.4 트랜스포머 Block

```c
// 단일 트랜스포머 디코더 block
void transformer_block(
    float *x,           // [1 × d_model] 현재 토큰 상태
    const Weights *w,   // 이 레이어의 가중치 포인터
    KVCache *kv,
    cublasHandle_t blas_handle,
    int layer, int pos, const LlamaConfig *cfg)
{
    // 1. 어텐션 전 RMSNorm
    rmsnorm_forward<<<1, 256>>>(x, w->rms_att, x_norm, cfg->d_model, 1e-5f);

    // 2. QKV 프로젝션 (GEMM 3개, 또는 하나로 융합)
    cublasSgemm_col_major(blas_handle, x_norm, w->Wq, q, 1, cfg->d_model, cfg->d_model);
    cublasSgemm_col_major(blas_handle, x_norm, w->Wk, k_new, 1, cfg->d_model, cfg->d_head * cfg->n_heads);
    cublasSgemm_col_major(blas_handle, x_norm, w->Wv, v_new, 1, cfg->d_model, cfg->d_head * cfg->n_heads);

    // 3. Q와 K에 RoPE (회전 위치 임베딩) 적용
    apply_rope<<<1, cfg->n_heads>>>(q, k_new, pos, cfg->d_head);

    // 4. KV 캐시 업데이트
    float *k_ptr = kv->k_cache + layer * cfg->max_seq_len * cfg->n_heads * cfg->d_head
                               + pos * cfg->n_heads * cfg->d_head;
    cudaMemcpy(k_ptr, k_new, cfg->n_heads * cfg->d_head * sizeof(float),
               cudaMemcpyDeviceToDevice);
    // v에 대해서도 동일하게...

    // 5. FlashAttention (인과적, pos+1 토큰)
    flash_attention_decode<<<cfg->n_heads, 256>>>(
        q, kv->k_cache + layer * ..., kv->v_cache + layer * ...,
        att_out, pos + 1, cfg->d_head, 1.f / sqrtf(cfg->d_head));

    // 6. 출력 프로젝션
    cublasSgemm_col_major(blas_handle, att_out, w->Wo, attn_proj, 1,
                          cfg->n_heads * cfg->d_head, cfg->d_model);

    // 7. Residual 추가
    vector_add_inplace<<<(cfg->d_model+255)/256, 256>>>(x, attn_proj, cfg->d_model);

    // 8. FFN 전 RMSNorm
    rmsnorm_forward<<<1, 256>>>(x, w->rms_ffn, x_norm, cfg->d_model, 1e-5f);

    // 9. SwiGLU FFN: out = (gate(x_norm) ⊙ SiLU(gate(x_norm))) * W_down
    //   SwiGLU: gate = x_norm @ W_gate, up = x_norm @ W_up
    //           h = SiLU(gate) * up
    //           out = h @ W_down
    cublasSgemm_col_major(blas_handle, x_norm, w->W_gate, gate, 1, cfg->d_model, cfg->d_ffn);
    cublasSgemm_col_major(blas_handle, x_norm, w->W_up,   up,   1, cfg->d_model, cfg->d_ffn);
    silu_mul_inplace<<<(cfg->d_ffn+255)/256, 256>>>(gate, up, cfg->d_ffn);  // gate *= SiLU(gate) * up
    cublasSgemm_col_major(blas_handle, gate, w->W_down, ffn_out, 1, cfg->d_ffn, cfg->d_model);

    // 10. Residual 추가
    vector_add_inplace<<<(cfg->d_model+255)/256, 256>>>(x, ffn_out, cfg->d_model);
}
```

### B.5 생성 루프 및 벤치마크

```c
void generate(const LlamaModel *model, const int *prompt_tokens,
              int prompt_len, int max_new_tokens) {
    KVCache kv;
    init_kv_cache(&kv, &model->cfg);

    // 프리필: 프롬프트 토큰을 병렬로 처리 (프롬프트에 대한 배치 GEMM)
    // 단순화: 토큰별로 처리
    float *d_logits;
    cudaMalloc(&d_logits, model->cfg.vocab_size * sizeof(float));

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);

    // 프리필 단계
    for (int i = 0; i < prompt_len; i++) {
        embed_token<<<1, 256>>>(prompt_tokens[i], model->embedding, d_x, model->cfg.d_model);
        for (int l = 0; l < model->cfg.n_layers; l++)
            transformer_block(d_x, &model->weights[l], &kv, model->blas, l, i, &model->cfg);
        kv.seq_len++;
    }

    // 생성 단계 (자기회귀)
    int next_token = prompt_tokens[prompt_len - 1];
    cudaEventRecord(t0);

    for (int t = 0; t < max_new_tokens; t++) {
        embed_token<<<1, 256>>>(next_token, model->embedding, d_x, model->cfg.d_model);

        for (int l = 0; l < model->cfg.n_layers; l++)
            transformer_block(d_x, &model->weights[l], &kv, model->blas,
                              l, kv.seq_len, &model->cfg);

        // 최종 RMSNorm + unembedding
        rmsnorm_forward<<<1, 256>>>(d_x, model->rms_final, d_x_norm,
                                    model->cfg.d_model, 1e-5f);
        cublasSgemm_col_major(model->blas, d_x_norm, model->unembedding,
                              d_logits, 1, model->cfg.d_model, model->cfg.vocab_size);

        // 탐욕적 샘플링: vocab에 대한 argmax
        next_token = argmax_kernel(d_logits, model->cfg.vocab_size);
        kv.seq_len++;

        // 토큰 디코드 및 출력
        const char *text = tokenizer_decode(next_token);
        printf("%s", text); fflush(stdout);

        if (next_token == EOS_TOKEN) break;
    }
    cudaEventRecord(t1);
    cudaDeviceSynchronize();

    float ms;
    cudaEventElapsedTime(&ms, t0, t1);
    printf("\n\n처리량: %.1f tokens/sec\n", max_new_tokens / (ms * 1e-3f));
}
```

### B.6 성능 목표

```
RTX 3090에서 Llama-tiny (8 레이어, d=256, 4 헤드):
  목표: > 5000 tokens/sec (단일 배치 디코드)
  메모리: ~50 MB (모델) + ~20 MB (KV 캐시 @ 2048 토큰)

A100 80GB FP16에서 Llama-7B (32 레이어, d=4096, 32 헤드):
  목표: ~100 tokens/sec (단일 배치 탐욕적)
  메모리: 14 GB (가중치) + 2 GB (KV 캐시 @ 2048 토큰)
  병목: 디코드 시 가중치 로딩 대역폭 (FLOP이 아님!)

INT8 양자화 (W8A16):
  메모리: 7 GB (가중치) → 더 많은 KV 캐시 공간
  목표: ~180 tokens/sec (가중치당 2× 적은 바이트에서 +80%)
```

---

## 프로파일링 체크리스트 (두 프로젝트 모두)

```bash
# 1. nsys: 고수준 타임라인
nsys profile --stats=true -o profile_report ./my_app [args]
nsys stats profile_report.nsys-rep  # 요약 테이블 출력

# 2. ncu: 상세 kernel 지표
ncu --target-processes all \
    --metrics gpu__time_duration.sum,\
              l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum,\
              sm__throughput.avg.pct_of_peak_sustained_elapsed,\
              sm__inst_executed_pipe_fp32.avg \
    -o ncu_report ./my_app --steps 10

# 3. 답해야 할 핵심 질문:
#   - 시간의 어느 정도가 kernel에서 vs 메모리 복사에서 소비되는가?
#   - 어떤 kernel이 가장 많은 시간을 차지하는가?
#   - 병목 kernel은 메모리 병목인가 연산 병목인가?
#   - occupancy가 성능을 제한하고 있는가?
#   - 불필요한 kernel 실행이 있는가 (< 1 μs)?

# 4. 루프라인 분석 (ncu --set roofline)
ncu --set roofline -o roofline_report ./my_app --steps 10
ncu-ui roofline_report.ncu-rep   # 루프라인 차트가 있는 GUI 열기
```

---

## 핵심 요약

- **프로젝트 A (LBM)**: streaming이 병목 kernel — 메모리 대역폭 병목; SoA 레이아웃, pull 방식 사용, 별도의 CUDA stream으로 IO와 연산 오버랩
- **프로젝트 B (LLM)**: 디코드 단계는 GEMM 연산이 아닌 가중치 로딩 (메모리 대역폭 병목)이 지배적; INT8 양자화는 대역폭 절약에 비례하여 tokens/sec를 직접 향상
- **최적화 전 프로파일링**: 타임라인을 위한 nsys, 이후 kernel 수준 지표를 위한 ncu; 실제로 런타임을 지배하는 kernel을 최적화
- **먼저 정확성**: LBM을 Ghia 벤치마크 데이터 (Re=100, 400, 1000)와 검증; 고정된 프롬프트와 시드에 대한 참조 CPU 구현과 LLM 검증
- **실제 엔지니어링**: 대부분의 GPU 애플리케이션 시간은 핵심 kernel이 아닌 설정, 데이터 로딩, 동기화에 있음 — 핫 루프만이 아닌 전체 애플리케이션 프로파일링
- **완전한 여정**: thread 인덱싱 (L2) → 메모리 계층 (L4-L5) → warp 실행 (L6) → 프로파일링 (L11) → reduction (L14) → 도메인 알고리즘 (L20-L25) → DL kernel (L32-L36) — 모든 개념이 효율적이고 프로덕션 품질의 CUDA 애플리케이션을 향해 구축됨

---

**코스 완료!** CUDA GPU 프로그래밍의 전체 스펙트럼을 아키텍처 기초부터 프로덕션 딥러닝 kernel까지 완료했습니다. 이 코스의 기술은 GPU 시스템 엔지니어링, ML 인프라, 고성능 과학 컴퓨팅에 직접 적용됩니다.
