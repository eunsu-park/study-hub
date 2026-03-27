# 45. Capstone: 완전한 추론 엔진

**이전**: [병렬 추론](./44_Parallel_Inference.md) | 코스 완료

---

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 모든 DL_Scratch_C 구성 요소를 작동하는 LLM 추론 엔진으로 조립
2. 완전한 decode 루프 구현: prefill, KV cache 관리, 토큰 생성
3. tokens/sec 측정 및 llama.cpp 기준 성능과 비교
4. 처리량에 영향을 주는 핵심 설계 결정 (메모리 레이아웃, 버퍼 재사용, 스레드 수) 식별
5. 프로덕션 품질 체크리스트에 대한 자신의 구현 자기 평가

---

## 1. 엔진 아키텍처 개요

이 capstone은 레슨 26-44의 모든 기술을 하나의 일관된 프로그램으로 통합합니다:

```
GGUF 파일
    │
    ▼
[43] gguf_load()          ─── 메모리 맵 가중치, tensor 메타데이터
    │
    ▼
[40] 요청 시 dequant     ─── INT4/INT8 가중치 → 계산 시 FP32
    │
    ▼
CLI 입력 → tokenize (BPE, 레슨 21)
    │
    ▼
[Prefill] forward(prompt)  ─── 모든 prompt 토큰에 대한 KV cache 구축
    │
    ▼
[Decode 루프]
    ├── [30] Llama 블록: RMSNorm + GQA attention + SwiGLU FFN
    ├── [26] KV cache 읽기/쓰기
    ├── [23] RoPE 위치 인코딩
    ├── [44] OpenMP 병렬 matmul
    └── [39] sample_token() → 다음 토큰
    │
    ▼
생성되는 대로 토큰 출력
```

---

## 2. 핵심 데이터 구조

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdint.h>

#ifdef _OPENMP
#include <omp.h>
#endif

// ============================================================
// 모델 구성 (GGUF 메타데이터에서 파싱)
// ============================================================
typedef struct {
    int   n_layers;
    int   n_heads;        // query heads
    int   n_kv_heads;     // key/value heads (GQA: n_kv_heads <= n_heads)
    int   d_model;        // embedding 차원
    int   d_ff;           // FFN 중간 차원 (SwiGLU)
    int   vocab_size;
    int   max_seq_len;
    float rope_theta;     // RoPE 기본 주파수 (Llama-3의 경우 기본값 500000.0)
    int   d_head;         // d_model / n_heads
} ModelConfig;

// ============================================================
// KV Cache  (레슨 26, 38)
// ============================================================
typedef struct {
    float  *K;        // [n_layers, max_seq, n_kv_heads, d_head]
    float  *V;        // [n_layers, max_seq, n_kv_heads, d_head]
    int     n_cached; // 현재 캐시에 있는 토큰 수
    int     max_seq;
    int     n_layers;
    int     n_kv_heads;
    int     d_head;
} KVCache;

KVCache *kvcache_create(const ModelConfig *cfg, int max_seq) {
    KVCache *kv = malloc(sizeof(KVCache));
    kv->max_seq    = max_seq;
    kv->n_layers   = cfg->n_layers;
    kv->n_kv_heads = cfg->n_kv_heads;
    kv->d_head     = cfg->d_head;
    kv->n_cached   = 0;
    size_t sz = (size_t)cfg->n_layers * max_seq * cfg->n_kv_heads * cfg->d_head;
    kv->K = calloc(sz, sizeof(float));
    kv->V = calloc(sz, sizeof(float));
    if (!kv->K || !kv->V) { fprintf(stderr, "KV cache alloc failed\n"); exit(1); }
    printf("KV cache: %.2f MB\n",
           2.0 * sz * sizeof(float) / (1024.0 * 1024.0));
    return kv;
}

void kvcache_free(KVCache *kv) {
    free(kv->K); free(kv->V); free(kv);
}

// 위치 pos에 하나의 토큰의 K와 V를 캐시에 쓰기
void kvcache_write(KVCache *kv, int layer, int pos,
                   const float *k_vec, const float *v_vec) {
    int stride = kv->max_seq * kv->n_kv_heads * kv->d_head;
    float *K_layer = kv->K + layer * stride;
    float *V_layer = kv->V + layer * stride;
    int offset = pos * kv->n_kv_heads * kv->d_head;
    memcpy(K_layer + offset, k_vec, kv->n_kv_heads * kv->d_head * sizeof(float));
    memcpy(V_layer + offset, v_vec, kv->n_kv_heads * kv->d_head * sizeof(float));
}

// ============================================================
// 활성화 버퍼 (미리 할당, 각 스텝에 재사용)
// ============================================================
typedef struct {
    float *x;         // 현재 hidden state [d_model]
    float *x_norm;    // RMSNorm 후 [d_model]
    float *q;         // query [n_heads * d_head]
    float *k;         // key   [n_kv_heads * d_head]
    float *v;         // value [n_kv_heads * d_head]
    float *attn_out;  // attention 출력 [n_heads * d_head]
    float *ffn_up;    // FFN gate/up [d_ff]
    float *ffn_gate;  // FFN gate [d_ff]
    float *logits;    // vocabulary logits [vocab_size]
} ActivationBuffers;

ActivationBuffers *buffers_create(const ModelConfig *cfg) {
    ActivationBuffers *b = malloc(sizeof(ActivationBuffers));
    b->x        = malloc(cfg->d_model * sizeof(float));
    b->x_norm   = malloc(cfg->d_model * sizeof(float));
    b->q        = malloc(cfg->n_heads    * cfg->d_head * sizeof(float));
    b->k        = malloc(cfg->n_kv_heads * cfg->d_head * sizeof(float));
    b->v        = malloc(cfg->n_kv_heads * cfg->d_head * sizeof(float));
    b->attn_out = malloc(cfg->n_heads    * cfg->d_head * sizeof(float));
    b->ffn_up   = malloc(cfg->d_ff * sizeof(float));
    b->ffn_gate = malloc(cfg->d_ff * sizeof(float));
    b->logits   = malloc(cfg->vocab_size * sizeof(float));
    return b;
}

void buffers_free(ActivationBuffers *b) {
    free(b->x); free(b->x_norm); free(b->q); free(b->k); free(b->v);
    free(b->attn_out); free(b->ffn_up); free(b->ffn_gate); free(b->logits);
    free(b);
}
```

---

## 3. 핵심 계산 기본 요소

```c
// ============================================================
// RMSNorm  (레슨 24)
// out[i] = x[i] / rms(x) * weight[i]
// ============================================================
void rmsnorm(float *out, const float *x, const float *weight, int n) {
    float ss = 0.0f;
    for (int i = 0; i < n; i++) ss += x[i] * x[i];
    float inv_rms = 1.0f / sqrtf(ss / n + 1e-5f);
    for (int i = 0; i < n; i++) out[i] = x[i] * inv_rms * weight[i];
}

// ============================================================
// RoPE: 제자리에서 rotary 위치 인코딩 적용 (레슨 23)
// 인터리브 쌍에 적용: (x[2i], x[2i+1])
// ============================================================
void rope_apply(float *x, int pos, int d, float theta) {
    for (int i = 0; i < d; i += 2) {
        float freq = 1.0f / powf(theta, (float)i / (float)d);
        float cos_f = cosf(pos * freq);
        float sin_f = sinf(pos * freq);
        float x0 = x[i], x1 = x[i+1];
        x[i]   = x0 * cos_f - x1 * sin_f;
        x[i+1] = x0 * sin_f + x1 * cos_f;
    }
}

// ============================================================
// SwiGLU 활성화: out[i] = gate[i] * silu(up[i])
// silu(x) = x * sigmoid(x)
// ============================================================
static float silu(float x) { return x / (1.0f + expf(-x)); }

void swiglu(float *out, const float *gate, const float *up, int n) {
    for (int i = 0; i < n; i++)
        out[i] = silu(gate[i]) * up[i];
}

// ============================================================
// Matmul (단일 토큰): out[N] = input[K] @ W[N,K]^T
// 선택적으로 병렬 (OpenMP)
// ============================================================
void matmul_vec(float *out, const float *x, const float *W, int N, int K) {
#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int n = 0; n < N; n++) {
        float acc = 0.0f;
        for (int k = 0; k < K; k++) acc += x[k] * W[n*K + k];
        out[n] = acc;
    }
}
```

---

## 4. Transformer 레이어 Forward Pass

```c
// 단순화된 가중치 구조체 (실제 엔진에서는 mmap을 통해 GGUF에서 로드)
typedef struct {
    // Attention 가중치 [모든 형태는 d_head = d_model / n_heads 가정]
    float *wq;       // [n_heads * d_head, d_model]
    float *wk;       // [n_kv_heads * d_head, d_model]
    float *wv;       // [n_kv_heads * d_head, d_model]
    float *wo;       // [d_model, n_heads * d_head]
    float *attn_norm; // [d_model]
    // FFN 가중치 (SwiGLU: 3개 행렬)
    float *w_gate;   // [d_ff, d_model]
    float *w_up;     // [d_ff, d_model]
    float *w_down;   // [d_model, d_ff]
    float *ffn_norm; // [d_model]
} LayerWeights;

void transformer_layer_forward(ActivationBuffers *buf,
                                const LayerWeights *wts,
                                KVCache *kv,
                                const ModelConfig *cfg,
                                int layer, int pos) {
    int dm  = cfg->d_model;
    int dh  = cfg->d_head;
    int nh  = cfg->n_heads;
    int nkv = cfg->n_kv_heads;
    int dff = cfg->d_ff;
    int groups = nh / nkv;  // GQA: 몇 개의 Q head가 하나의 KV head를 공유하는지

    // --- Attention 서브레이어 ---
    rmsnorm(buf->x_norm, buf->x, wts->attn_norm, dm);

    matmul_vec(buf->q, buf->x_norm, wts->wq, nh  * dh, dm);
    matmul_vec(buf->k, buf->x_norm, wts->wk, nkv * dh, dm);
    matmul_vec(buf->v, buf->x_norm, wts->wv, nkv * dh, dm);

    // 각 head의 Q와 K에 RoPE 적용
    for (int h = 0; h < nh;  h++) rope_apply(buf->q + h * dh, pos, dh, cfg->rope_theta);
    for (int h = 0; h < nkv; h++) rope_apply(buf->k + h * dh, pos, dh, cfg->rope_theta);

    // KV cache에 K, V 쓰기
    kvcache_write(kv, layer, pos, buf->k, buf->v);

    // attention 출력 계산: 각 query head에 대해
    float scale = 1.0f / sqrtf((float)dh);
    int stride_kv = kv->n_kv_heads * dh;

    for (int qh = 0; qh < nh; qh++) {
        int kv_head = qh / groups;
        const float *q_h  = buf->q + qh * dh;
        const float *K_h  = kv->K + layer * kv->max_seq * stride_kv + kv_head * dh;
        const float *V_h  = kv->V + layer * kv->max_seq * stride_kv + kv_head * dh;
        float       *o_h  = buf->attn_out + qh * dh;

        // 모든 캐시된 위치에 대한 점수 계산
        float *scores = malloc((pos + 1) * sizeof(float));
        for (int t = 0; t <= pos; t++) {
            float dot = 0.0f;
            const float *k_t = K_h + t * stride_kv;
            for (int d = 0; d < dh; d++) dot += q_h[d] * k_t[d];
            scores[t] = dot * scale;
        }

        // Softmax
        float max_s = scores[0];
        for (int t = 1; t <= pos; t++) if (scores[t] > max_s) max_s = scores[t];
        float sum_e = 0.0f;
        for (int t = 0; t <= pos; t++) { scores[t] = expf(scores[t] - max_s); sum_e += scores[t]; }
        for (int t = 0; t <= pos; t++) scores[t] /= sum_e;

        // V의 가중 합
        memset(o_h, 0, dh * sizeof(float));
        for (int t = 0; t <= pos; t++) {
            const float *v_t = V_h + t * stride_kv;
            for (int d = 0; d < dh; d++) o_h[d] += scores[t] * v_t[d];
        }
        free(scores);
    }

    // attention 출력을 다시 projection: x += Wo @ attn_out
    float *delta = malloc(dm * sizeof(float));
    matmul_vec(delta, buf->attn_out, wts->wo, dm, nh * dh);
    for (int i = 0; i < dm; i++) buf->x[i] += delta[i];
    free(delta);

    // --- FFN 서브레이어 (SwiGLU) ---
    rmsnorm(buf->x_norm, buf->x, wts->ffn_norm, dm);
    matmul_vec(buf->ffn_gate, buf->x_norm, wts->w_gate, dff, dm);
    matmul_vec(buf->ffn_up,   buf->x_norm, wts->w_up,   dff, dm);
    swiglu(buf->ffn_up, buf->ffn_gate, buf->ffn_up, dff);

    delta = malloc(dm * sizeof(float));
    matmul_vec(delta, buf->ffn_up, wts->w_down, dm, dff);
    for (int i = 0; i < dm; i++) buf->x[i] += delta[i];
    free(delta);
}
```

---

## 5. 메인 추론 엔진

```c
// Tokens/sec 측정
typedef struct {
    double t_prefill_start;
    double t_decode_start;
    int    n_prompt;
    int    n_generated;
    double elapsed_decode;
} PerfStats;

double get_time_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

// 최소 샘플러 구성 (temperature + top-p)
typedef struct { float temperature; float top_p; } SamplerCfg;

// Softmax 및 샘플 (단순화 인라인)
static int sample_next_token(float *logits, int vocab, float temp, float top_p) {
    if (temp <= 0.0f) {
        int best = 0;
        for (int i = 1; i < vocab; i++) if (logits[i] > logits[best]) best = i;
        return best;
    }
    for (int i = 0; i < vocab; i++) logits[i] /= temp;
    float max_l = logits[0];
    for (int i = 1; i < vocab; i++) if (logits[i] > max_l) max_l = logits[i];
    float sum = 0.0f;
    for (int i = 0; i < vocab; i++) { logits[i] = expf(logits[i] - max_l); sum += logits[i]; }
    for (int i = 0; i < vocab; i++) logits[i] /= sum;
    float r = (float)rand() / ((float)RAND_MAX + 1.0f);
    float cum = 0.0f;
    for (int i = 0; i < vocab; i++) { cum += logits[i]; if (r < cum) return i; }
    return vocab - 1;
}

// 메인 추론 함수
// 실제 엔진에서: 가중치는 gguf_load (레슨 43)를 통해 로드됨
// 여기서는 제어 흐름 구조를 보여줌
void run_inference(const ModelConfig *cfg,
                   LayerWeights *layer_weights,  // n_layers의 배열
                   float *embed_table,            // [vocab_size, d_model]
                   float *output_norm,            // [d_model] 최종 RMSNorm
                   float *lm_head,                // [vocab_size, d_model]
                   const int *prompt_tokens, int n_prompt,
                   int max_new_tokens,
                   const SamplerCfg *sampler) {

    KVCache          *kv  = kvcache_create(cfg, cfg->max_seq_len);
    ActivationBuffers *buf = buffers_create(cfg);
    PerfStats stats;
    stats.t_prefill_start = get_time_sec();
    stats.n_prompt    = n_prompt;
    stats.n_generated = 0;

    int *output = malloc(max_new_tokens * sizeof(int));
    int cur_token = prompt_tokens[0];
    int pos = 0;

    // ---- Prefill: 모든 prompt 토큰 처리 ----
    for (int pi = 0; pi < n_prompt; pi++) {
        cur_token = prompt_tokens[pi];
        // 토큰 embed
        memcpy(buf->x, embed_table + cur_token * cfg->d_model,
               cfg->d_model * sizeof(float));
        // 모든 transformer 레이어 실행
        for (int l = 0; l < cfg->n_layers; l++)
            transformer_layer_forward(buf, &layer_weights[l], kv, cfg, l, pos);
        pos++;
    }
    stats.t_decode_start = get_time_sec();

    // 생성할 다음 토큰은 마지막 prefill 위치에서 예측됨
    rmsnorm(buf->x_norm, buf->x, output_norm, cfg->d_model);
    matmul_vec(buf->logits, buf->x_norm, lm_head, cfg->vocab_size, cfg->d_model);
    cur_token = sample_next_token(buf->logits, cfg->vocab_size,
                                  sampler->temperature, sampler->top_p);

    // ---- Decode 루프: 한 번에 하나씩 토큰 생성 ----
    double decode_start = get_time_sec();
    for (int gen = 0; gen < max_new_tokens; gen++) {
        output[gen] = cur_token;
        // EOS 확인 (Llama-3의 경우 토큰 2, Llama-3-instruct의 경우 128001)
        if (cur_token == 2 || cur_token == 128001) {
            stats.n_generated = gen + 1;
            break;
        }

        // 현재 토큰에 대한 embed + forward pass
        memcpy(buf->x, embed_table + cur_token * cfg->d_model,
               cfg->d_model * sizeof(float));
        for (int l = 0; l < cfg->n_layers; l++)
            transformer_layer_forward(buf, &layer_weights[l], kv, cfg, l, pos);
        pos++;
        stats.n_generated = gen + 1;

        // 다음 토큰 logits 계산
        rmsnorm(buf->x_norm, buf->x, output_norm, cfg->d_model);
        matmul_vec(buf->logits, buf->x_norm, lm_head, cfg->vocab_size, cfg->d_model);
        cur_token = sample_next_token(buf->logits, cfg->vocab_size,
                                      sampler->temperature, sampler->top_p);

        // 생성되는 대로 토큰 출력 (스트리밍 출력)
        printf(" [tok%d]", cur_token); fflush(stdout);
    }

    stats.elapsed_decode = get_time_sec() - decode_start;
    double tps = stats.n_generated / stats.elapsed_decode;

    printf("\n\n=== Performance ===\n");
    printf("  Prompt tokens:   %d\n", n_prompt);
    printf("  Generated:       %d tokens\n", stats.n_generated);
    printf("  Decode time:     %.2f s\n", stats.elapsed_decode);
    printf("  Throughput:      %.2f tokens/sec\n", tps);
    printf("  Prefill time:    %.2f s (%.1f tok/s)\n",
           stats.t_decode_start - stats.t_prefill_start,
           n_prompt / (stats.t_decode_start - stats.t_prefill_start));

    kvcache_free(kv);
    buffers_free(buf);
    free(output);
}
```

---

## 6. 명령줄 인터페이스

```c
typedef struct {
    char    model_path[512];
    char    prompt[4096];
    int     max_new_tokens;
    float   temperature;
    float   top_p;
    int     n_threads;
    int     context_len;
} CLIArgs;

void print_usage(const char *prog) {
    fprintf(stderr,
        "Usage: %s [OPTIONS]\n"
        "  -m <path>       GGUF 모델 파일 (필수)\n"
        "  -p <prompt>     입력 prompt (기본값: 'Hello')\n"
        "  -n <int>        최대 새 토큰 수 (기본값: 200)\n"
        "  -t <float>      Temperature (기본값: 0.8)\n"
        "  --top-p <float> Top-p nucleus sampling (기본값: 0.9)\n"
        "  -T <int>        스레드 수 (기본값: 4)\n"
        "  -c <int>        Context 길이 (기본값: 2048)\n",
        prog);
}

int parse_args(CLIArgs *args, int argc, char **argv) {
    strcpy(args->model_path, "");
    strcpy(args->prompt, "Hello, world!");
    args->max_new_tokens = 200;
    args->temperature    = 0.8f;
    args->top_p          = 0.9f;
    args->n_threads      = 4;
    args->context_len    = 2048;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-m") == 0 && i+1 < argc)
            strncpy(args->model_path, argv[++i], 511);
        else if (strcmp(argv[i], "-p") == 0 && i+1 < argc)
            strncpy(args->prompt, argv[++i], 4095);
        else if (strcmp(argv[i], "-n") == 0 && i+1 < argc)
            args->max_new_tokens = atoi(argv[++i]);
        else if (strcmp(argv[i], "-t") == 0 && i+1 < argc)
            args->temperature = atof(argv[++i]);
        else if (strcmp(argv[i], "--top-p") == 0 && i+1 < argc)
            args->top_p = atof(argv[++i]);
        else if (strcmp(argv[i], "-T") == 0 && i+1 < argc)
            args->n_threads = atoi(argv[++i]);
        else if (strcmp(argv[i], "-c") == 0 && i+1 < argc)
            args->context_len = atoi(argv[++i]);
        else if (strcmp(argv[i], "-h") == 0) {
            print_usage(argv[0]); return -1;
        }
    }
    if (strlen(args->model_path) == 0) {
        fprintf(stderr, "Error: -m <model_path> is required\n");
        print_usage(argv[0]); return -1;
    }
    return 0;
}

int main(int argc, char **argv) {
    CLIArgs args;
    if (parse_args(&args, argc, argv) < 0) return 1;

#ifdef _OPENMP
    omp_set_num_threads(args.n_threads);
    printf("Using %d OpenMP threads\n", args.n_threads);
#endif

    printf("Model: %s\n", args.model_path);
    printf("Prompt: \"%s\"\n", args.prompt);
    printf("Max tokens: %d, Temperature: %.2f, Top-p: %.2f\n",
           args.max_new_tokens, args.temperature, args.top_p);

    // 완전한 구현에서:
    // 1. GGUFModel model; gguf_load(&model, args.model_path);
    // 2. model.meta에서 ModelConfig 구성
    // 3. mmap된 tensor를 가리키는 LayerWeights (즉시 dequant 포함)
    // 4. BPE tokenizer로 args.prompt 토큰화 (레슨 21)
    // 5. run_inference(...)

    printf("\n[전체 구현: GGUF 로더 (레슨 43),\n"
           " BPE tokenizer (레슨 21), 모든 레이어 가중치 연결]\n");

    // 더미 가중치로 decode 루프 오버헤드 벤치마크
    const int dummy_vocab  = 1000;
    const int dummy_layers = 2;
    const int dummy_dm     = 256;
    const int dummy_heads  = 4;
    const int dummy_dh     = dummy_dm / dummy_heads;
    const int dummy_dff    = dummy_dm * 4;

    ModelConfig cfg = {
        .n_layers   = dummy_layers,
        .n_heads    = dummy_heads,
        .n_kv_heads = dummy_heads,
        .d_model    = dummy_dm,
        .d_ff       = dummy_dff,
        .vocab_size = dummy_vocab,
        .max_seq_len = 512,
        .rope_theta = 500000.0f,
        .d_head     = dummy_dh,
    };

    // 더미 가중치 할당
    LayerWeights *lw = calloc(dummy_layers, sizeof(LayerWeights));
    for (int l = 0; l < dummy_layers; l++) {
        lw[l].wq        = calloc(dummy_heads * dummy_dh * dummy_dm, sizeof(float));
        lw[l].wk        = calloc(dummy_heads * dummy_dh * dummy_dm, sizeof(float));
        lw[l].wv        = calloc(dummy_heads * dummy_dh * dummy_dm, sizeof(float));
        lw[l].wo        = calloc(dummy_dm * dummy_heads * dummy_dh, sizeof(float));
        lw[l].attn_norm = calloc(dummy_dm, sizeof(float));
        lw[l].w_gate    = calloc(dummy_dff * dummy_dm, sizeof(float));
        lw[l].w_up      = calloc(dummy_dff * dummy_dm, sizeof(float));
        lw[l].w_down    = calloc(dummy_dm * dummy_dff, sizeof(float));
        lw[l].ffn_norm  = calloc(dummy_dm, sizeof(float));
    }
    float *embed  = calloc(dummy_vocab * dummy_dm, sizeof(float));
    float *onorm  = calloc(dummy_dm, sizeof(float));
    float *lmhead = calloc(dummy_vocab * dummy_dm, sizeof(float));

    // RMSNorm이 모든 것을 0으로 만들지 않도록 norm을 1.0으로 초기화
    for (int i = 0; i < dummy_dm; i++) { onorm[i] = 1.0f; }
    for (int l = 0; l < dummy_layers; l++) {
        for (int i = 0; i < dummy_dm; i++) {
            lw[l].attn_norm[i] = 1.0f;
            lw[l].ffn_norm[i]  = 1.0f;
        }
    }

    int prompt[] = { 1, 42, 17, 88 };
    SamplerCfg sc = { args.temperature, args.top_p };
    srand(42);

    printf("\n--- 더미 %d-레이어 모델에 대한 추론 실행 ---\n", dummy_layers);
    run_inference(&cfg, lw, embed, onorm, lmhead,
                  prompt, 4, 20, &sc);

    // 정리
    for (int l = 0; l < dummy_layers; l++) {
        free(lw[l].wq); free(lw[l].wk); free(lw[l].wv); free(lw[l].wo);
        free(lw[l].attn_norm); free(lw[l].w_gate); free(lw[l].w_up);
        free(lw[l].w_down); free(lw[l].ffn_norm);
    }
    free(lw); free(embed); free(onorm); free(lmhead);
    return 0;
}
```

---

## 7. 벤치마킹 및 llama.cpp와 비교

```c
// 일반 하드웨어에서 실제 Llama-3-8B Q4_K_M의 예상 성능:
void print_comparison_table(void) {
    printf("=== Tokens/Sec 비교: Llama-3-8B Q4_K_M, Context=512 ===\n");
    printf("%-30s %10s %10s %10s\n", "구현", "스레드", "tok/s", "vs llama.cpp");
    printf("%-30s %10s %10s %10s\n",
           "llama.cpp (최적화)",         "8",  "~30",  "1.0x (기준)");
    printf("%-30s %10s %10s %10s\n",
           "llama.cpp (최적화)",         "4",  "~22",  "0.7x");
    printf("%-30s %10s %10s %10s\n",
           "이 엔진 (OpenMP)",           "8",  "~8-15","0.3-0.5x");
    printf("%-30s %10s %10s %10s\n",
           "이 엔진 (OpenMP)",           "4",  "~5-10","0.2-0.3x");
    printf("%-30s %10s %10s %10s\n",
           "이 엔진 (naive F32)",        "1",  "~1-3", "0.05-0.1x");

    printf("\n격차 설명:\n");
    printf("  llama.cpp 사용: AVX2/AVX-512 SIMD, GGML K-quant 커널,\n");
    printf("  메모리 레이아웃 최적화, numa-aware 할당, metal/CUDA.\n");
    printf("  우리 엔진: 스칼라 FP32만, 이식 가능한 C11, 교육적 명확성.\n");
}
```

---

## 8. 자기 평가 체크리스트

추론 엔진이 완성되었다고 주장하기 전에 다음을 검증하세요:

**정확성**
- [ ] RMSNorm 출력이 참조(PyTorch)와 1e-5 상대 오차 내에서 일치
- [ ] RoPE 회전이 참조와 동일한 attention 패턴 생성
- [ ] Greedy decoding이 llama.cpp와 동일한 토큰 시퀀스 생성 (같은 모델, 같은 prompt)
- [ ] KV cache가 올바르게 성장: prefill 중 위치 0..n_prompt-1 채움, decode 중 pos n_prompt+ 채움
- [ ] GQA head 그룹화: query head `h`가 KV head `h / (n_heads / n_kv_heads)` 읽기

**성능**
- [ ] 버퍼 재사용: decode 루프 내부에 malloc/free 없음 (모든 활성화 버퍼 미리 할당)
- [ ] 중복 softmax 호출 없음 (스텝당 attention head당 하나만)
- [ ] Matmul이 출력 뉴런에 걸쳐 병렬화 (M=1이므로 행이 아닌)
- [ ] KV cache가 attention 점수 계산 중 순차 접근 패턴을 위해 배치됨

**견고성**
- [ ] EOS 토큰이 생성을 올바르게 종료
- [ ] Context 창 오버플로 처리 (오류 또는 슬라이딩 창)
- [ ] Temperature=0.0이 0으로 나누기 없이 greedy로 폴백
- [ ] GGUF 로딩이 tensor를 읽기 전에 magic number와 version 검증

**프로덕션 준비**
- [ ] INT8/INT4 dequantization 통합 (FP32 전용 가중치가 아닌)
- [ ] 메모리 맵 GGUF 로딩 (힙으로 전체 가중치 복사 없음)
- [ ] 텍스트 I/O를 위한 BPE tokenizer (레슨 21) 연결
- [ ] 스트리밍 토큰 출력 (끝에가 아닌 생성되는 대로 각 토큰 출력)

---

## 핵심 요약

- 완전한 LLM 추론 엔진은 embedding 조회, RMSNorm, RoPE, KV cache가 있는 GQA attention, SwiGLU FFN, 샘플링 전략을 통합합니다 — 각각 이전 레슨에서 개발됨.
- 활성화 버퍼 사전 할당이 중요합니다: decode 루프 내부의 malloc/free는 각 밀리초가 중요한 10-30 tokens/sec에서 상당한 오버헤드를 추가합니다.
- KV cache 메모리 레이아웃 (레이어 × 위치 × head × dim)은 캐시된 위치에 대한 attention 점수 계산 루프 중 순차 접근을 최대화하도록 선택해야 합니다.
- 순진한 C11 구현은 일반적으로 llama.cpp 처리량의 0.1-0.3×를 달성합니다; 격차는 llama.cpp의 SIMD intrinsics, quantized 커널, 플랫폼별 메모리 최적화로 설명됩니다 — 알고리즘적 차이가 아닙니다.
- 참조 구현(PyTorch 또는 llama.cpp)에 대한 정확성 검증이 모든 성능 최적화에 선행되어야 합니다: 정확한 것만 벤치마크하세요.
- batch=1에서의 tokens/sec 지표는 거의 전적으로 메모리 대역폭과 가중치 크기에 의해 결정됩니다; 알고리즘적 개선(FlashAttention, speculative decoding)은 경쟁이 아닌 보완적인 최적화입니다.
- 처음부터 이 엔진을 구축하면 프로덕션 시스템의 모든 결정에 대한 깊은 이해를 제공합니다: llama.cpp가 사용하는 데이터 구조의 이유, quantization 선택이 중요한 이유, 진정한 병목이 무엇인지.

---

**이전**: [병렬 추론](./44_Parallel_Inference.md) | 코스 완료
