# 42. 추측적 디코딩

**이전**: [CPU에서의 FlashAttention](./41_FlashAttention_CPU.md) | **다음**: [GGUF 형식과 로딩](./43_GGUF_and_Loading.md)

---

## 학습 목표

이 레슨을 마치면 다음을 할 수 있습니다:

1. 추측적 디코딩이 출력 분포를 변경하지 않고 자기회귀 추론을 가속화할 수 있는 이유를 설명할 수 있다
2. 초안 토큰을 수락할지 결정하는 기각 샘플링 절차를 구현할 수 있다
3. 토큰이 기각될 때 정확한 목표 분포를 유지하는 수정된 재샘플링 단계를 이해할 수 있다
4. 수락률을 추적하고 이를 사용하여 예상 속도 향상을 예측할 수 있다
5. 깔끔한 분리를 위해 함수 포인터를 사용하여 초안/목표 모델 인터페이스를 설계할 수 있다

---

## 1. 자기회귀 병목

표준 LLM 디코딩은 엄격하게 순차적입니다: 토큰 1 생성 → 피드백 → 토큰 2 생성 → ... 각 단계는 목표 모델(예: 70억 파라미터 모델)을 통한 전체 forward pass가 필요합니다. GPU 또는 CPU는 batch=1에서 단일 토큰이 메모리 대역폭에 비해 매우 적은 연산을 제공하기 때문에 충분히 활용되지 못합니다.

추측적 디코딩(Leviathan 등, 2023; Chen 등, 2023)은 이를 활용합니다: 작은 초안 모델이 K개의 후보 토큰을 빠르게 생성하면, 큰 목표 모델이 *단일 병렬 forward pass*로 K개의 토큰 모두를 검증합니다. 목표 모델은 batch=1이 아닌 batch=K로 실행되어 비용을 분산시킵니다.

```
초안 모델 (빠름, 작음):
  x_1, x_2, ..., x_K = draft_generate(context, K)   -- K번의 순차 단계, 빠름

목표 모델 (느림, 큼):
  p(x_1|ctx), p(x_2|ctx,x_1), ..., p(x_K|...) = target_forward(context + draft_tokens)
               -- 하나의 병렬 forward pass!

각 초안 토큰을 수락/기각하고, 기각된 경우 재샘플링.
```

예상 속도 향상 ≈ K × 수락률 (K=4 초안과 80% 수락률: ~3.2× 속도 향상).

---

## 2. 기각 샘플링 수학

`q(x|ctx)`를 초안 모델 분포, `p(x|ctx)`를 목표 분포라고 합시다. 각 초안 토큰 `x_i`에 대해:

```
p(x_i) >= q(x_i)이면:  x_i 수락 (확률 1.0)
그렇지 않으면:           p(x_i) / q(x_i) 확률로 x_i 수락
```

토큰이 기각될 때, 단순히 `p`에서 재샘플링하면 안 됩니다 — `p >> q`인 토큰 쪽으로 편향이 생깁니다. 대신 *수정된 분포*에서 재샘플링합니다:

```
p'(x) = max(0, p(x) - q(x)) / sum_x max(0, p(x) - q(x))
       = normalize(max(0, p - q))
```

이 수정은 수락된 토큰의 주변 분포가 `q`에 관계없이 정확히 `p`임을 보장합니다. 증명은 우아합니다: 예상 출력 분포 = 수락된 부분 (`p`로 분포) + 기각된 부분 (정규화된 `(p-q)_+`에서 재샘플링)이 함께 `p`로 적분됩니다.

---

## 3. 모델 인터페이스 (함수 포인터)

초안 모델과 목표 모델을 추측적 디코딩 로직을 변경하지 않고 교체할 수 있도록 함수 포인터를 사용하여 모델 인터페이스를 정의합니다:

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define VOCAB_SIZE 32000
#define MAX_DRAFT_K 8

// Model forward pass: given token sequence, produce probability distribution
// for the NEXT token. Returns a heap-allocated float[vocab_size].
// Caller must free.
typedef float* (*ModelForwardFn)(const int *tokens, int n_tokens, void *model_state);

typedef struct {
    ModelForwardFn forward;
    void          *state;
    int            vocab_size;
} Model;

// Compute softmax in-place
static void softmax_inplace(float *x, int n) {
    float max_val = x[0];
    for (int i = 1; i < n; i++) if (x[i] > max_val) max_val = x[i];
    float sum = 0.0f;
    for (int i = 0; i < n; i++) { x[i] = expf(x[i] - max_val); sum += x[i]; }
    for (int i = 0; i < n; i++) x[i] /= sum;
}

// Sample one token from probability distribution
static int sample_categorical(const float *probs, int n) {
    float r = (float)rand() / ((float)RAND_MAX + 1.0f);
    float cum = 0.0f;
    for (int i = 0; i < n; i++) {
        cum += probs[i];
        if (r < cum) return i;
    }
    return n - 1;
}
```

---

## 4. 초안 토큰 생성

초안 모델은 자체 컨텍스트를 유지하면서 K개의 토큰을 자기회귀적으로(순차적으로) 생성합니다:

```c
// Generate K draft tokens autoregressively using the draft model
// context:        current token sequence [n_ctx]
// draft_tokens:   output buffer [K]
// draft_probs:    output probabilities q(x_i) for each draft token [K]
//                 (the probability assigned to the chosen draft token)
void draft_generate(int *draft_tokens, float *draft_probs,
                    const int *context, int n_ctx,
                    Model *draft_model, int K) {
    // Build working context (context + draft tokens so far)
    int *ctx = malloc((n_ctx + K) * sizeof(int));
    memcpy(ctx, context, n_ctx * sizeof(int));
    int ctx_len = n_ctx;

    for (int k = 0; k < K; k++) {
        float *logits = draft_model->forward(ctx, ctx_len, draft_model->state);
        softmax_inplace(logits, draft_model->vocab_size);

        int tok = sample_categorical(logits, draft_model->vocab_size);
        draft_tokens[k]  = tok;
        draft_probs[k]   = logits[tok];  // q(x_k | context, x_1..x_{k-1})

        ctx[ctx_len++] = tok;
        free(logits);
    }
    free(ctx);
}
```

---

## 5. 목표 모델 병렬 검증

목표 모델은 하나의 forward pass에서 전체 컨텍스트(K개의 초안 토큰 포함)를 처리하여 K+1개의 확률 분포를 생성합니다(각 위치마다 하나):

```c
// Run target model on context + draft tokens, get K probability distributions
// target_probs_out: [K, vocab_size] — p(x_k | context + draft[0..k-1])
// Returns allocated array; caller must free each row
void target_verify(float **target_probs_out,
                   const int *context, int n_ctx,
                   const int *draft_tokens, int K,
                   Model *target_model) {
    // For simplicity: run K separate forward passes (in real implementation
    // this would be a batched or chunked prefill of the K draft positions)
    int *ctx = malloc((n_ctx + K) * sizeof(int));
    memcpy(ctx, context, n_ctx * sizeof(int));

    for (int k = 0; k < K; k++) {
        // Target evaluates at position n_ctx + k - 1 (predicts draft_tokens[k])
        float *logits = target_model->forward(ctx, n_ctx + k, target_model->state);
        softmax_inplace(logits, target_model->vocab_size);
        target_probs_out[k] = logits;  // caller owns this
        ctx[n_ctx + k] = draft_tokens[k];
    }
    // Also get the distribution at position n_ctx+K (for resampling after full acceptance)
    target_probs_out[K] = target_model->forward(ctx, n_ctx + K, target_model->state);
    softmax_inplace(target_probs_out[K], target_model->vocab_size);

    free(ctx);
}
```

---

## 6. 기각 샘플링과 수정된 재샘플링

```c
// Resample from p'(x) = normalize(max(0, p - q))
// p and q are probability distributions over vocab
static int resample_corrected(const float *p, const float *q, int vocab_size) {
    float *p_prime = malloc(vocab_size * sizeof(float));
    float total = 0.0f;

    for (int i = 0; i < vocab_size; i++) {
        p_prime[i] = fmaxf(0.0f, p[i] - q[i]);
        total += p_prime[i];
    }

    if (total < 1e-10f) {
        // Edge case: p == q everywhere → p' is zero, resample from p
        free(p_prime);
        return sample_categorical(p, vocab_size);
    }

    for (int i = 0; i < vocab_size; i++)
        p_prime[i] /= total;

    int tok = sample_categorical(p_prime, vocab_size);
    free(p_prime);
    return tok;
}

// Speculative decoding: generate one "batch" of tokens
// Returns number of tokens accepted (1 to K+1)
// out_tokens: accepted tokens appended here (caller pre-allocates K+1 slots)
int speculative_decode_step(int *out_tokens,
                             const int *context, int n_ctx,
                             Model *draft_model, Model *target_model,
                             int K,
                             int *accepted_count_accum,
                             int *total_draft_count) {
    int   draft_tokens[MAX_DRAFT_K];
    float draft_probs[MAX_DRAFT_K];
    float *target_probs[MAX_DRAFT_K + 1];  // K+1 distributions from target

    // Step 1: Draft model generates K tokens
    draft_generate(draft_tokens, draft_probs, context, n_ctx, draft_model, K);

    // Step 2: Target model verifies all K positions in parallel
    target_verify(target_probs, context, n_ctx, draft_tokens, K, target_model);

    // Step 3: Rejection sampling
    int n_accepted = 0;
    for (int k = 0; k < K; k++) {
        float p_k = target_probs[k][draft_tokens[k]];
        float q_k = draft_probs[k];

        float accept_prob = fminf(1.0f, p_k / (q_k + 1e-10f));
        float r = (float)rand() / ((float)RAND_MAX + 1.0f);

        if (r < accept_prob) {
            // Accept draft token
            out_tokens[n_accepted++] = draft_tokens[k];
            (*accepted_count_accum)++;
        } else {
            // Reject: resample from corrected distribution
            int resampled = resample_corrected(target_probs[k],
                                               /* q at this position */
                                               NULL,  // simplified
                                               target_model->vocab_size);
            out_tokens[n_accepted++] = resampled;
            // After rejection, stop — subsequent draft tokens are invalid
            for (int kk = k; kk <= K; kk++) free(target_probs[kk]);
            (*total_draft_count) += K;
            return n_accepted;
        }
    }

    // All K draft tokens accepted: also take one token from target at position K
    int bonus = sample_categorical(target_probs[K], target_model->vocab_size);
    out_tokens[n_accepted++] = bonus;

    for (int k = 0; k <= K; k++) free(target_probs[k]);
    (*total_draft_count) += K;
    return n_accepted;
}
```

---

## 7. 수락률 추적이 포함된 전체 추측적 디코딩 루프

```c
typedef struct {
    int   n_accepted;    // total tokens accepted from draft
    int   n_draft;       // total draft tokens proposed
    int   n_target_calls; // number of target model forward passes
    double time_draft;   // seconds spent in draft model
    double time_target;  // seconds spent in target model
} SpecStats;

// Generate max_new_tokens using speculative decoding
// Returns actual tokens generated (stored in out_tokens)
int speculative_decode_full(int *out_tokens,
                             const int *prompt, int prompt_len,
                             Model *draft_model, Model *target_model,
                             int K, int max_new_tokens,
                             SpecStats *stats) {
    memset(stats, 0, sizeof(*stats));

    int *context = malloc((prompt_len + max_new_tokens + K) * sizeof(int));
    memcpy(context, prompt, prompt_len * sizeof(int));
    int ctx_len = prompt_len;
    int total_generated = 0;

    while (total_generated < max_new_tokens) {
        int remaining = max_new_tokens - total_generated;
        int draft_K = (remaining < K) ? remaining : K;

        int step_out[MAX_DRAFT_K + 1];
        int accepted_accum = 0, draft_count = 0;

        int n_step = speculative_decode_step(step_out,
                                              context, ctx_len,
                                              draft_model, target_model,
                                              draft_K,
                                              &accepted_accum, &draft_count);

        // Append accepted tokens to context
        int take = (n_step < remaining) ? n_step : remaining;
        for (int i = 0; i < take; i++) {
            out_tokens[total_generated + i] = step_out[i];
            context[ctx_len + i] = step_out[i];
        }
        ctx_len += take;
        total_generated += take;

        stats->n_accepted += accepted_accum;
        stats->n_draft    += draft_count;
        stats->n_target_calls++;
    }

    free(context);
    return total_generated;
}

void print_spec_stats(const SpecStats *s, int K) {
    float accept_rate = (s->n_draft > 0)
                      ? (float)s->n_accepted / s->n_draft : 0.0f;
    float expected_speedup = (float)K * accept_rate;
    printf("Speculative Decoding Statistics:\n");
    printf("  Draft tokens proposed:  %d\n", s->n_draft);
    printf("  Draft tokens accepted:  %d (%.1f%%)\n",
           s->n_accepted, accept_rate * 100.0f);
    printf("  Target forward passes:  %d\n", s->n_target_calls);
    printf("  Expected speedup vs naive: %.2fx\n", expected_speedup);
    printf("  (Naive would need %d target passes for same output)\n",
           s->n_draft + s->n_target_calls);
}
```

---

## 8. 토이 모델 데모: 수락률 측정

```c
// Toy draft model: returns slightly noisy version of target distribution
// In practice, draft should share tokenizer with target
typedef struct { int vocab_size; float *base_probs; float noise; } ToyModelState;

float *toy_forward(const int *tokens, int n_tokens, void *state) {
    ToyModelState *s = (ToyModelState *)state;
    float *probs = malloc(s->vocab_size * sizeof(float));
    float sum = 0.0f;
    for (int i = 0; i < s->vocab_size; i++) {
        probs[i] = s->base_probs[i] + s->noise * ((float)rand()/RAND_MAX);
        if (probs[i] < 0.0f) probs[i] = 0.0f;
        sum += probs[i];
    }
    for (int i = 0; i < s->vocab_size; i++) probs[i] /= sum;
    return probs;
}

int main(void) {
    srand(42);
    const int V = 100;

    // Target distribution: peaked at token 5
    float target_base[100] = {0};
    target_base[5] = 10.0f; target_base[6] = 5.0f; target_base[7] = 2.0f;
    for (int i = 0; i < V; i++) if (target_base[i] == 0.0f) target_base[i] = 0.1f;

    ToyModelState draft_state  = { V, target_base, 0.5f };  // noisy draft
    ToyModelState target_state = { V, target_base, 0.01f }; // near-exact target

    Model draft_model  = { toy_forward, &draft_state,  V };
    Model target_model = { toy_forward, &target_state, V };

    const int K = 4;
    const int N = 100;  // tokens to generate
    int *out = malloc(N * sizeof(int));
    int prompt[1] = {0};

    SpecStats stats;
    speculative_decode_full(out, prompt, 1, &draft_model, &target_model,
                            K, N, &stats);
    print_spec_stats(&stats, K);

    free(out);
    return 0;
}
```

목표 분포를 밀접하게 근사하는 초안 모델(동일 계열의 소형 대 대형 모델의 일반적인 경우)의 경우, 70-85%의 수락률이 일반적이며, K=4에서 2.8-3.4×의 속도 향상이 발생합니다.

---

## 핵심 요약

- 추측적 디코딩은 K개의 토큰을 병렬로 검증하는 것(batch=K forward pass)이 1개의 토큰을 검증하는 것보다 약간 더 많은 비용이 든다는 사실을 활용합니다 — 목표 모델은 batch=1에서 충분히 활용되지 못합니다.
- 기각 샘플링 기준 `accept_prob = min(1, p(x)/q(x))`는 수락된 토큰이 근사 없이 정확히 목표 분포와 일치함을 보장합니다.
- 토큰이 기각될 때, `normalize(max(0, p - q))`에서 재샘플링하면 단순히 `p`에서 직접 재샘플링할 때 발생하는 편향이 수정됩니다.
- 어떤 기각 후에도 이후의 모든 초안 토큰은 삭제해야 합니다 — 기각된 토큰이 컨텍스트에 있다는 조건하에 생성되었기 때문입니다.
- 예상 속도 향상은 `K × 수락률`입니다; K=4이고 75% 수락률이면 목표 모델 호출이 ~3× 적어집니다.
- 초안 모델과 목표 모델은 동일한 tokenizer를 공유해야 합니다 — 그렇지 않으면 토큰 인덱스의 의미가 달라져 기각 샘플링이 정의되지 않습니다.
- 초안 모델은 속도 향상이 실현되려면 훨씬 더 작아야 합니다(10-100× 더 적은 파라미터); 2× 더 작은 초안은 최소한의 이점을 제공합니다.

---

**이전**: [CPU에서의 FlashAttention](./41_FlashAttention_CPU.md) | **다음**: [GGUF 형식과 로딩](./43_GGUF_and_Loading.md)
