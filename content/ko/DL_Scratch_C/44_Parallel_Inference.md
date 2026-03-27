# 44. 병렬 추론

**이전**: [GGUF 형식과 로딩](./43_GGUF_and_Loading.md) | **다음**: [Capstone: 추론 엔진](./45_Capstone_Inference_Engine.md)

---

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. OpenMP와 POSIX 스레드를 사용하여 CPU 코어 전반에 걸쳐 행렬 곱셈 병렬화
2. 단일 토큰 LLM 추론이 연산 바운드가 아닌 메모리 대역폭 바운드인 이유 설명
3. CPU 추론의 달성 가능한 처리량 예측을 위한 roofline 모델 적용
4. 멀티헤드 attention을 스레드 전반에 분산하기 위한 attention head 병렬성 구현
5. 메모리 대역폭 측정 및 최대 토큰 생성 속도 예측에 사용

---

## 1. 병렬성이 중요한 이유 — 그리고 중요하지 않을 때

단일 토큰 LLM 추론 (batch=1)은 가중치 행렬 읽기가 지배합니다. `[out, in]` 가중치 행렬을 가진 linear 레이어의 경우:

```
FLOPs:    2 * out * in          (가중치당 하나의 multiply-add)
Bytes:    out * in * bytes_per_weight  (각 가중치를 한 번 읽음)

Arithmetic intensity = FLOPs / Bytes
  INT8의 경우:  2 / 1 = 2.0 FLOP/byte
  F16의 경우:   2 / 2 = 1.0 FLOP/byte
  F32의 경우:   2 / 4 = 0.5 FLOP/byte
```

일반적인 x86 CPU:
- 최대 FP32 GFLOP/s: 40-200 (AVX-512 및 클럭에 따라)
- 최대 메모리 대역폭: 50-100 GB/s (DDR5)
- Ridge point: 1-4 FLOP/byte

단일 토큰 추론은 intensity가 < 2 FLOP/byte이므로, roofline의 ridge point 왼쪽에 위치합니다 — **메모리 대역폭 바운드**. 더 많은 연산 코어를 추가하면 FLOP/s가 아닌 대역폭에 대해서만 선형적으로 도움이 됩니다.

---

## 2. CPU 추론을 위한 Roofline 분석

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <pthread.h>

#ifdef _OPENMP
#include <omp.h>
#endif

// Roofline 모델: 달성 가능한 GFLOP/s = min(peak_flops, bw * intensity)
void roofline_analysis(void) {
    const double peak_flops_gflops = 80.0;   // 일반적인 현대 데스크탑 CPU
    const double bandwidth_GBs     = 60.0;   // DDR5 듀얼 채널

    printf("=== Roofline Analysis: Single-Token LLM Inference ===\n\n");
    printf("%-20s %8s %10s %12s %12s\n",
           "Operation", "FLOP/B", "Peak (GFLOP)", "BW-limit", "Achieved");

    struct { const char *name; double ai; double gflops; } ops[] = {
        // ai = arithmetic intensity (FLOP/byte), gflops = 실제 연산 요구
        { "F32 matmul (1×N×M)",  0.5,  2.0  },
        { "F16 matmul (1×N×M)",  1.0,  2.0  },
        { "INT8 matmul (1×N×M)", 2.0,  2.0  },
        { "INT4 matmul (1×N×M)", 4.0,  2.0  },
        { "Softmax (T=2048)",    0.1,  0.05 },
        { "RMSNorm",             0.1,  0.01 },
    };

    for (int i = 0; i < 6; i++) {
        double ai      = ops[i].ai;
        double gflops  = ops[i].gflops;
        double bw_lim  = bandwidth_GBs * ai;       // 이 AI에서의 대역폭 상한
        double achieved = fmin(peak_flops_gflops, bw_lim);
        const char *bound = (bw_lim < peak_flops_gflops) ? "BW-bound" : "Compute";
        printf("%-20s %8.1f %10.2f %10.2f %10.2f (%s)\n",
               ops[i].name, ai, peak_flops_gflops, bw_lim, achieved, bound);
    }

    printf("\n결론: batch=1의 모든 연산은 메모리 대역폭 바운드입니다.\n");
    printf("최대 token rate ≈ bandwidth / bytes_per_token\n");

    // 7B INT8 모델의 경우: ~7GB 가중치, 토큰당 ~14 GFLOP
    double model_bytes  = 7e9;
    double flops_token  = 14e9;
    double ai_model     = flops_token / model_bytes;
    double tok_per_sec  = bandwidth_GBs * 1e9 / model_bytes;
    printf("7B INT8 model: AI=%.2f, theoretical max = %.1f tokens/sec\n",
           ai_model, tok_per_sec);
}
```

---

## 3. 기준 Matmul (단일 스레드)

```c
// Dense matmul: out[M, N] = input[M, K] @ weight[N, K]^T  (가중치가 전치됨)
// weight은 [N, K] 행 우선으로 저장 (각 행은 하나의 출력 뉴런의 가중치)
void matmul_st(float *out, const float *input, const float *weight,
               int M, int N, int K) {
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float acc = 0.0f;
            for (int k = 0; k < K; k++)
                acc += input[m*K + k] * weight[n*K + k];
            out[m*N + n] = acc;
        }
    }
}
```

---

## 4. OpenMP 병렬성

OpenMP는 외부 루프를 출력 행 전반에 걸쳐 스레드에 분산합니다. 각 출력 행은 독립적이므로 안전합니다.

```c
// OpenMP 병렬 matmul: 출력 행(m)에 걸쳐 병렬화
void matmul_omp(float *out, const float *input, const float *weight,
                int M, int N, int K) {
#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float acc = 0.0f;
            for (int k = 0; k < K; k++)
                acc += input[m*K + k] * weight[n*K + k];
            out[m*N + n] = acc;
        }
    }
}

// 단일 토큰 추론의 경우 M=1이므로 m에 걸쳐 병렬화하면 이점이 없습니다.
// 대신 출력 뉴런(n)에 걸쳐 병렬화합니다:
void matmul_omp_single_token(float *out, const float *input, const float *weight,
                              int N, int K) {
    // M=1 case: input[K], weight[N, K], out[N]
#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int n = 0; n < N; n++) {
        float acc = 0.0f;
        for (int k = 0; k < K; k++)
            acc += input[k] * weight[n*K + k];
        out[n] = acc;
    }
}
```

컴파일: `gcc -O3 -march=native -fopenmp matmul.c -o matmul -lm`

단일 토큰 추론의 경우 작은 N/K에서 스레드 오버헤드가 중요합니다. `OMP_NUM_THREADS`로 조정하세요. 일반적인 LLM 차원(N=4096, K=4096)에서는 4-8 스레드가 거의 선형 속도 향상을 제공합니다.

---

## 5. POSIX 스레드 대안

OpenMP 오버헤드 없이 세밀한 제어를 위해:

```c
typedef struct {
    const float *input;
    const float *weight;
    float       *out;
    int          N, K;
    int          row_start, row_end;  // 이 스레드가 처리하는 출력 뉴런 범위
} MatmulArgs;

void *matmul_thread(void *arg) {
    MatmulArgs *a = (MatmulArgs *)arg;
    for (int n = a->row_start; n < a->row_end; n++) {
        float acc = 0.0f;
        for (int k = 0; k < a->K; k++)
            acc += a->input[k] * a->weight[n * a->K + k];
        a->out[n] = acc;
    }
    return NULL;
}

// POSIX 스레드를 사용한 병렬 matmul (단일 토큰: M=1)
void matmul_pthread(float *out, const float *input, const float *weight,
                    int N, int K, int n_threads) {
    pthread_t   *threads = malloc(n_threads * sizeof(pthread_t));
    MatmulArgs  *args    = malloc(n_threads * sizeof(MatmulArgs));

    int rows_per_thread = (N + n_threads - 1) / n_threads;
    for (int t = 0; t < n_threads; t++) {
        args[t].input     = input;
        args[t].weight    = weight;
        args[t].out       = out;
        args[t].N         = N;
        args[t].K         = K;
        args[t].row_start = t * rows_per_thread;
        args[t].row_end   = args[t].row_start + rows_per_thread;
        if (args[t].row_end > N) args[t].row_end = N;
        pthread_create(&threads[t], NULL, matmul_thread, &args[t]);
    }
    for (int t = 0; t < n_threads; t++)
        pthread_join(threads[t], NULL);

    free(threads); free(args);
}
```

---

## 6. 메모리 대역폭 측정

```c
// 지속적인 메모리 대역폭 측정: 대형 배열을 순차적으로 읽기
double measure_bandwidth_GBs(size_t array_bytes) {
    float *buf = malloc(array_bytes);
    for (size_t i = 0; i < array_bytes / sizeof(float); i++)
        buf[i] = (float)i;

    // 캐시 워밍
    volatile float sink = 0.0f;
    for (size_t i = 0; i < array_bytes / sizeof(float); i += 16)
        sink += buf[i];

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    // 배열을 4번 읽기 (캐싱 효과 방지)
    for (int rep = 0; rep < 4; rep++) {
        for (size_t i = 0; i < array_bytes / sizeof(float); i += 8)
            sink += buf[i];
    }

    clock_gettime(CLOCK_MONOTONIC, &t1);
    double elapsed = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;
    double bytes_read = 4.0 * array_bytes;
    double bw = bytes_read / elapsed / 1e9;

    free(buf);
    (void)sink;
    return bw;
}

void benchmark_bandwidth(void) {
    size_t sizes[] = { 256*1024, 4*1024*1024, 64*1024*1024, 512*1024*1024 };
    const char *labels[] = { "256KB (L2)", "4MB (L3)", "64MB (DRAM)", "512MB (DRAM)" };
    printf("=== Memory Bandwidth ===\n");
    for (int i = 0; i < 4; i++) {
        double bw = measure_bandwidth_GBs(sizes[i]);
        printf("  %-20s %.1f GB/s\n", labels[i], bw);
    }
}
```

---

## 7. Tensor 병렬성: 스레드 전반에 걸쳐 Attention Head 분할

멀티헤드 attention의 경우, 각 head는 forward pass에서 완전히 독립적입니다. 이것이 이상적인 병렬성 대상입니다:

```c
// Attention head 병렬 계산
// 각 스레드는 h attention head의 부분집합을 처리
typedef struct {
    const float *Q;         // [T, h, d_k] 행 우선
    const float *K;         // [T, h, d_k]
    const float *V;         // [T, h, d_k]
    float       *out;       // [T, h, d_k]
    int          T, h, d_k;
    int          head_start, head_end;
} AttnHeadArgs;

// 단일 헤드 attention (독립형, 스레드 관련 없음)
static void single_head_attn(float *out_h, const float *Q_h, const float *K_h,
                              const float *V_h, int T, int d_k) {
    float scale = 1.0f / sqrtf((float)d_k);
    float *scores = malloc(T * sizeof(float));
    float *attn   = malloc(T * sizeof(float));

    // 마지막 query 위치 (추론: 하나의 새 토큰)
    // Q_h: [d_k], K_h: [T, d_k], V_h: [T, d_k]
    for (int j = 0; j < T; j++) {
        float dot = 0.0f;
        for (int k = 0; k < d_k; k++)
            dot += Q_h[k] * K_h[j*d_k + k];
        scores[j] = dot * scale;
    }

    // T개의 key에 대한 softmax
    float max_s = scores[0];
    for (int j = 1; j < T; j++) if (scores[j] > max_s) max_s = scores[j];
    float sum = 0.0f;
    for (int j = 0; j < T; j++) { attn[j] = expf(scores[j] - max_s); sum += attn[j]; }
    for (int j = 0; j < T; j++) attn[j] /= sum;

    // 출력: V의 가중 합
    for (int k = 0; k < d_k; k++) {
        float acc = 0.0f;
        for (int j = 0; j < T; j++) acc += attn[j] * V_h[j*d_k + k];
        out_h[k] = acc;
    }
    free(scores); free(attn);
}

void *attn_head_thread(void *arg) {
    AttnHeadArgs *a = (AttnHeadArgs *)arg;
    int d_k = a->d_k;
    int T   = a->T;
    for (int hi = a->head_start; hi < a->head_end; hi++) {
        const float *Q_h   = a->Q   + hi * d_k;  // 이 head의 query (마지막 토큰)
        const float *K_h   = a->K   + hi * d_k;  // 이 head의 K [T, d_k]
        const float *V_h   = a->V   + hi * d_k;  // 이 head의 V [T, d_k]
        float       *out_h = a->out + hi * d_k;
        single_head_attn(out_h, Q_h, K_h, V_h, T, d_k);
    }
    return NULL;
}

void parallel_multihead_attn(float *out,
                              const float *Q, const float *K, const float *V,
                              int T, int n_heads, int d_k,
                              int n_threads) {
    pthread_t    *threads = malloc(n_threads * sizeof(pthread_t));
    AttnHeadArgs *args    = malloc(n_threads * sizeof(AttnHeadArgs));

    int heads_per_thread = (n_heads + n_threads - 1) / n_threads;
    for (int t = 0; t < n_threads; t++) {
        args[t].Q          = Q;
        args[t].K          = K;
        args[t].V          = V;
        args[t].out        = out;
        args[t].T          = T;
        args[t].h          = n_heads;
        args[t].d_k        = d_k;
        args[t].head_start = t * heads_per_thread;
        args[t].head_end   = args[t].head_start + heads_per_thread;
        if (args[t].head_end > n_heads) args[t].head_end = n_heads;
        pthread_create(&threads[t], NULL, attn_head_thread, &args[t]);
    }
    for (int t = 0; t < n_threads; t++) pthread_join(threads[t], NULL);
    free(threads); free(args);
}
```

---

## 8. 성능 스케일링 벤치마크

```c
double now_sec_mt(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

void benchmark_matmul_scaling(void) {
    // 7B 모델의 일반적인 LLM FFN 차원
    const int N = 4096, K = 4096;  // weight [N, K]
    float *input  = malloc(K * sizeof(float));
    float *weight = malloc((size_t)N * K * sizeof(float));
    float *out    = malloc(N * sizeof(float));

    srand(42);
    for (int i = 0; i < K; i++) input[i] = (float)rand()/RAND_MAX;
    for (size_t i = 0; i < (size_t)N*K; i++) weight[i] = (float)rand()/RAND_MAX - 0.5f;

    printf("=== Matmul Scaling: N=%d, K=%d (single-token M=1) ===\n", N, K);
    printf("%-12s %10s %10s %8s\n", "Threads", "Time(ms)", "GFLOP/s", "Speedup");

    double t_single = 0.0;
    int thread_counts[] = {1, 2, 4, 8, 16};
    for (int ti = 0; ti < 5; ti++) {
        int nt = thread_counts[ti];
        const int REPS = 100;
        double t0 = now_sec_mt();
        for (int r = 0; r < REPS; r++)
            matmul_pthread(out, input, weight, N, K, nt);
        double elapsed = (now_sec_mt() - t0) / REPS;
        double gflops = 2.0 * N * K / elapsed / 1e9;
        if (ti == 0) t_single = elapsed;
        printf("%-12d %10.2f %10.2f %8.2fx\n",
               nt, elapsed * 1000.0, gflops, t_single / elapsed);
    }

    // 이론적 대역폭 제한 예측
    double bw = measure_bandwidth_GBs(64 * 1024 * 1024);
    double bytes_per_token = (double)N * K * 4;  // F32
    double tokens_per_sec  = bw * 1e9 / bytes_per_token;
    printf("\n측정된 DRAM 대역폭: %.1f GB/s\n", bw);
    printf("대역폭 제한 최대: %.1f tokens/sec (F32, 단일 레이어)\n", tokens_per_sec);

    free(input); free(weight); free(out);
}

int main(void) {
    roofline_analysis();
    printf("\n");
    benchmark_bandwidth();
    printf("\n");
    benchmark_matmul_scaling();
    return 0;
}
```

60 GB/s 대역폭을 가진 현대 8코어 데스크탑의 예상 결과:
- 단일 스레드: ~2-5 GFLOP/s (큰 K에서 대역폭 제한)
- 4 스레드: ~3-4× 속도 향상 (거의 선형 — 멀티 채널 DDR에서 코어 수에 따라 대역폭 스케일)
- 8 스레드: ~4-6× 속도 향상 (단일 메모리 컨트롤러 포화로 체감 감소)

---

## 핵심 요약

- 단일 토큰 LLM 추론은 메모리 대역폭 바운드입니다: arithmetic intensity (FLOP/byte)가 모든 quantization 형식에서 roofline ridge point 아래이므로, 병목은 DRAM에서 가중치를 얼마나 빠르게 읽을 수 있는지입니다.
- 최대 tokens/sec은 대략 `DRAM_bandwidth / bytes_per_token`입니다 — 60 GB/s DDR5에서 7B INT8 모델의 경우 대략 60e9 / 7e9 ≈ 8.5 tokens/sec 단일 스레드.
- 출력 뉴런(`n`)에 걸쳐 외부 루프를 병렬화하는 것이 batch=1 matmul에 대한 올바른 접근 방식입니다; M=1일 때 batch 행(`m`)에 걸쳐 병렬화하면 이점이 없습니다.
- OpenMP `#pragma omp parallel for schedule(static)`은 matmul 병렬성에 충분합니다; POSIX 스레드는 더 세밀한 제어를 제공하지만 더 많은 보일러플레이트가 필요합니다.
- Attention head 병렬성은 완전히 병렬 가능합니다 — 각 head의 계산이 완전히 독립적입니다 — 따라서 스레드 수준 병렬성의 이상적인 대상입니다.
- 멀티 채널 DRAM 대역폭은 활성 메모리 채널 수에 따라 스케일됩니다; 다른 가중치 행에 동시에 접근하는 더 많은 스레드가 대역폭 활용을 향상시킬 수 있습니다.
- 스레드 생성 오버헤드는 소형 matmul 시간에 비해 중요합니다; 매우 작은 N×K의 경우 더 적은 스레드를 사용하거나 스레드 실행당 여러 연산을 batch 처리합니다.

---

**이전**: [GGUF 형식과 로딩](./43_GGUF_and_Loading.md) | **다음**: [Capstone: 추론 엔진](./45_Capstone_Inference_Engine.md)
