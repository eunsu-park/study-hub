# 39. 샘플링 전략

**이전**: [KV 캐시 최적화](./38_KV_Cache_Optimization.md) | **다음**: [양자화: INT8과 INT4](./40_Quantization_Int8_Int4.md)

---

## 학습 목표

이 레슨을 마치면 다음을 할 수 있습니다:

1. 탐욕적 디코딩(argmax)을 구현하고 다양한 출력을 생성하지 못하는 경우를 설명할 수 있다
2. 출력 무작위성을 제어하기 위해 softmax 전에 logit에 온도 스케일링을 적용할 수 있다
3. 후보 토큰 풀을 제한하는 top-k 필터링을 구현할 수 있다
4. 분포의 꼬리를 적응적으로 차단하는 top-p (nucleus) 샘플링을 구현할 수 있다
5. 출력 품질을 더욱 제어하기 위해 min-p와 반복 패널티를 적용할 수 있다

---

## 1. 토큰 샘플링 문제

각 디코딩 단계에서 모델은 크기 `vocab_size`의 logit 벡터를 생성합니다(예: Llama-3의 경우 32,000). 샘플링 전략은 다음 토큰을 어떻게 선택할지 결정합니다. 이 선택은 출력 품질에 큰 영향을 미칩니다:

- **탐욕적**: 항상 가장 높은 확률의 토큰을 선택합니다. 결정론적이지만 반복적인 루프에 빠질 수 있습니다.
- **무작위 샘플링**: 전체 분포에서 샘플링합니다. 최대한 다양하지만 종종 비일관적입니다.
- **필터링 샘플링**: 실용적인 최적점 — 후보를 제한한 후 샘플링합니다.

모든 전략은 logit(pre-softmax 점수)에서 동작합니다. 일반적인 파이프라인은:

```
logits[vocab] → 필터 적용 → softmax → 샘플
```

softmax는 수치적으로 민감하고 여러 번 계산하지 않으려 하기 때문에 logit 공간에서 작업합니다.

---

## 2. 탐욕적 디코딩 (Argmax)

가장 간단한 전략: 항상 가장 높은 logit을 가진 토큰을 선택합니다.

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <time.h>

// Returns index of maximum value in array
int argmax(const float *logits, int vocab_size) {
    int best = 0;
    float best_val = logits[0];
    for (int i = 1; i < vocab_size; i++) {
        if (logits[i] > best_val) {
            best_val = logits[i];
            best = i;
        }
    }
    return best;
}

// Greedy decode: deterministic, zero temperature
int greedy_sample(const float *logits, int vocab_size) {
    return argmax(logits, vocab_size);
}
```

**탐욕적 방식이 실패하는 경우**: "The cat sat on the ___"에서 다음 단어를 예측하는 모델을 생각해보세요. 탐욕적 방식은 매번 "mat"을 선택합니다. 분포에는 "floor", "chair", "roof"에 의미 있는 확률 질량이 있을 수 있는데, 탐욕적 방식은 이 모두를 무시하고 반복적이며 낮은 엔트로피의 텍스트를 생성합니다.

---

## 3. 온도 샘플링

온도 `T`는 softmax 전에 logit을 스케일링합니다:

```
logits_scaled[i] = logits[i] / T
probs = softmax(logits_scaled)
token = sample(probs)
```

- `T → 0`: 탐욕적에 수렴 (분포가 스파이크로 날카로워짐)
- `T = 1.0`: 표준 softmax (스케일링 없음)
- `T > 1.0`: 분포를 평탄화 (더 무작위)
- `T → ∞`: 모든 토큰에 균일 분포

```c
// Numerically stable softmax (subtract max before exp)
void softmax(float *probs, const float *logits, int n) {
    float max_val = logits[0];
    for (int i = 1; i < n; i++)
        if (logits[i] > max_val) max_val = logits[i];

    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        probs[i] = expf(logits[i] - max_val);
        sum += probs[i];
    }
    for (int i = 0; i < n; i++)
        probs[i] /= sum;
}

// Sample a token index from a probability distribution
int sample_from_probs(const float *probs, int n) {
    float r = (float)rand() / ((float)RAND_MAX + 1.0f);
    float cumsum = 0.0f;
    for (int i = 0; i < n; i++) {
        cumsum += probs[i];
        if (r < cumsum) return i;
    }
    return n - 1;  // fallback for floating-point edge case
}

// Temperature sampling: scale logits, then softmax, then sample
int temperature_sample(const float *logits, int vocab_size, float temperature) {
    if (temperature <= 0.0f) return greedy_sample(logits, vocab_size);

    float *scaled = malloc(vocab_size * sizeof(float));
    float *probs  = malloc(vocab_size * sizeof(float));

    for (int i = 0; i < vocab_size; i++)
        scaled[i] = logits[i] / temperature;

    softmax(probs, scaled, vocab_size);
    int token = sample_from_probs(probs, vocab_size);

    free(scaled);
    free(probs);
    return token;
}
```

일반적인 값: 집중된 창의적 텍스트에는 `T = 0.8`, 브레인스토밍에는 `T = 1.2`.

---

## 4. Top-K 필터링

Top-k는 샘플링을 상위 `k`개의 가장 높은 logit 토큰으로만 제한합니다. 다른 모든 logit은 `-infinity`로 설정됩니다(softmax 후 확률이 ~0이 됨).

```c
// Comparison function for qsort (descending by value)
typedef struct { float val; int idx; } IndexedFloat;

int cmp_desc(const void *a, const void *b) {
    float va = ((IndexedFloat *)a)->val;
    float vb = ((IndexedFloat *)b)->val;
    return (va < vb) - (va > vb);
}

// Set all logits except top-k to -infinity
// Modifies logits_out in place; call before softmax
void top_k_filter(float *logits_out, const float *logits, int vocab_size, int k) {
    if (k <= 0 || k >= vocab_size) {
        memcpy(logits_out, logits, vocab_size * sizeof(float));
        return;
    }

    IndexedFloat *ranked = malloc(vocab_size * sizeof(IndexedFloat));
    for (int i = 0; i < vocab_size; i++) {
        ranked[i].val = logits[i];
        ranked[i].idx = i;
    }
    qsort(ranked, vocab_size, sizeof(IndexedFloat), cmp_desc);

    // Set all to -inf, then restore top-k
    for (int i = 0; i < vocab_size; i++)
        logits_out[i] = -FLT_MAX;
    for (int i = 0; i < k; i++)
        logits_out[ranked[i].idx] = ranked[i].val;

    free(ranked);
}

int top_k_sample(const float *logits, int vocab_size, int k, float temperature) {
    float *filtered = malloc(vocab_size * sizeof(float));
    float *probs    = malloc(vocab_size * sizeof(float));

    top_k_filter(filtered, logits, vocab_size, k);

    // Apply temperature after top-k filter
    if (temperature > 0.0f)
        for (int i = 0; i < vocab_size; i++)
            if (filtered[i] > -FLT_MAX) filtered[i] /= temperature;

    softmax(probs, filtered, vocab_size);
    int token = sample_from_probs(probs, vocab_size);

    free(filtered);
    free(probs);
    return token;
}
```

일반적인 값: `k = 50` 또는 `k = 40`. `k = 1`로 설정하면 탐욕적과 동일합니다.

---

## 5. Top-P (Nucleus) 샘플링

Holtzman 등(2020)이 소개한 top-p는 적응형입니다: 누적 확률이 `p`를 초과하는 최소 토큰 집합을 유지합니다.

왜 적응형인가? Top-k에서는 분포가 얼마나 집중되어 있든 상관없이 항상 고정된 수의 토큰이 고려됩니다. 모델이 매우 확신할 때(한 토큰이 99% 확률), top-k=50은 여전히 50개 토큰에서 샘플링합니다. Top-p는 이 경우 자동으로 소수의 토큰으로 축소됩니다.

```c
// Top-p (nucleus) filter: zero out tokens outside the nucleus
// Returns number of tokens kept
int top_p_filter(float *logits_out, const float *logits, int vocab_size, float p) {
    if (p >= 1.0f) {
        memcpy(logits_out, logits, vocab_size * sizeof(float));
        return vocab_size;
    }

    // Sort by logit value descending
    IndexedFloat *ranked = malloc(vocab_size * sizeof(IndexedFloat));
    for (int i = 0; i < vocab_size; i++) {
        ranked[i].val = logits[i];
        ranked[i].idx = i;
    }
    qsort(ranked, vocab_size, sizeof(IndexedFloat), cmp_desc);

    // Compute cumulative softmax probabilities
    float max_val = ranked[0].val;
    float sum = 0.0f;
    for (int i = 0; i < vocab_size; i++) {
        ranked[i].val = expf(ranked[i].val - max_val);  // softmax numerator
        sum += ranked[i].val;
    }

    // Find cutoff index where cumulative prob > p
    float cumsum = 0.0f;
    int cutoff = vocab_size;
    for (int i = 0; i < vocab_size; i++) {
        cumsum += ranked[i].val / sum;
        if (cumsum >= p) { cutoff = i + 1; break; }
    }

    // Build output: keep only nucleus tokens
    for (int i = 0; i < vocab_size; i++)
        logits_out[i] = -FLT_MAX;
    for (int i = 0; i < cutoff; i++)
        logits_out[ranked[i].idx] = logits[ranked[i].idx];  // restore original logit

    free(ranked);
    return cutoff;
}

int top_p_sample(const float *logits, int vocab_size, float p, float temperature) {
    float *filtered = malloc(vocab_size * sizeof(float));
    float *probs    = malloc(vocab_size * sizeof(float));

    top_p_filter(filtered, logits, vocab_size, p);

    if (temperature > 0.0f)
        for (int i = 0; i < vocab_size; i++)
            if (filtered[i] > -FLT_MAX) filtered[i] /= temperature;

    softmax(probs, filtered, vocab_size);
    int token = sample_from_probs(probs, vocab_size);

    free(filtered);
    free(probs);
    return token;
}
```

일반적인 값: `p = 0.9` 또는 `p = 0.95`. 종종 온도와 결합: 먼저 온도를 적용한 후 top-p를 적용합니다.

---

## 6. Min-P 샘플링

Min-p(2023년 도입)는 top-p보다 단순합니다: 확률이 `min_p × max_prob`보다 작은 토큰을 필터링합니다. 이는 모델의 확신도에 상대적으로 임계값을 스케일링합니다.

```c
// Min-p filter: remove tokens with prob < min_p * max_prob
void min_p_filter(float *logits_out, const float *logits, int vocab_size, float min_p) {
    // Compute softmax probabilities first
    float *probs = malloc(vocab_size * sizeof(float));
    softmax(probs, logits, vocab_size);

    float max_prob = 0.0f;
    for (int i = 0; i < vocab_size; i++)
        if (probs[i] > max_prob) max_prob = probs[i];

    float threshold = min_p * max_prob;
    for (int i = 0; i < vocab_size; i++)
        logits_out[i] = (probs[i] >= threshold) ? logits[i] : -FLT_MAX;

    free(probs);
}
```

Min-p는 정렬을 피하기 때문에 top-p보다 빠르며, 경험적으로 비슷하거나 더 나은 출력 품질을 생성합니다.

---

## 7. 반복 패널티

반복 패널티는 최근 생성된 토큰의 logit을 패널티 계수 `> 1.0`으로 나눔으로써 모델이 반복하는 것을 억제합니다:

```c
// Divide logits of previously seen tokens by penalty (> 1.0)
// tokens_seen: array of previously generated token IDs
// n_seen: number of tokens in history to penalize
void repetition_penalty_apply(float *logits, int vocab_size,
                               const int *tokens_seen, int n_seen,
                               float penalty) {
    if (penalty <= 1.0f) return;
    for (int i = 0; i < n_seen; i++) {
        int tok = tokens_seen[i];
        if (tok < 0 || tok >= vocab_size) continue;
        // Positive logits get reduced; negative logits get pushed more negative
        if (logits[tok] > 0.0f)
            logits[tok] /= penalty;
        else
            logits[tok] *= penalty;
    }
}
```

양수와 음수 logit을 비대칭적으로 처리하는 이유는? 양수 logit은 높은 확률에 기여하는데 나누면 줄어듭니다. 음수 logit은 낮은 확률에 기여하는데 나누면 우연히 증가하게 됩니다. 곱하면 더 음수로 밀어내어 일관되게 확률을 줄입니다.

일반적인 값: `penalty = 1.1`에서 `1.3`. 너무 높으면 모델이 모든 일반적인 단어를 피하게 됩니다.

---

## 8. 토큰 샘플 디스패치 함수

모든 전략을 하나의 구성 가능한 함수로 결합:

```c
typedef enum {
    SAMPLE_GREEDY    = 0,
    SAMPLE_TEMP      = 1,
    SAMPLE_TOP_K     = 2,
    SAMPLE_TOP_P     = 3,
    SAMPLE_MIN_P     = 4,
} SampleStrategy;

typedef struct {
    SampleStrategy strategy;
    float temperature;   // for TEMP, TOP_K, TOP_P, MIN_P
    int   top_k;         // for TOP_K
    float top_p;         // for TOP_P
    float min_p;         // for MIN_P (e.g. 0.05)
    float rep_penalty;   // repetition penalty (1.0 = disabled)
} SamplerConfig;

// Modifies logits buffer (copy before calling if you need originals)
int sample_token(float *logits, int vocab_size,
                 const SamplerConfig *cfg,
                 const int *tokens_seen, int n_seen) {
    // 1. Apply repetition penalty in-place
    if (cfg->rep_penalty > 1.0f)
        repetition_penalty_apply(logits, vocab_size, tokens_seen, n_seen, cfg->rep_penalty);

    float *work = malloc(vocab_size * sizeof(float));
    float *probs = malloc(vocab_size * sizeof(float));
    int token;

    switch (cfg->strategy) {
        case SAMPLE_GREEDY:
            token = greedy_sample(logits, vocab_size);
            break;

        case SAMPLE_TEMP:
            token = temperature_sample(logits, vocab_size, cfg->temperature);
            break;

        case SAMPLE_TOP_K:
            top_k_filter(work, logits, vocab_size, cfg->top_k);
            if (cfg->temperature > 0.0f)
                for (int i = 0; i < vocab_size; i++)
                    if (work[i] > -FLT_MAX) work[i] /= cfg->temperature;
            softmax(probs, work, vocab_size);
            token = sample_from_probs(probs, vocab_size);
            break;

        case SAMPLE_TOP_P:
            top_p_filter(work, logits, vocab_size, cfg->top_p);
            if (cfg->temperature > 0.0f)
                for (int i = 0; i < vocab_size; i++)
                    if (work[i] > -FLT_MAX) work[i] /= cfg->temperature;
            softmax(probs, work, vocab_size);
            token = sample_from_probs(probs, vocab_size);
            break;

        case SAMPLE_MIN_P:
            min_p_filter(work, logits, vocab_size, cfg->min_p);
            if (cfg->temperature > 0.0f)
                for (int i = 0; i < vocab_size; i++)
                    if (work[i] > -FLT_MAX) work[i] /= cfg->temperature;
            softmax(probs, work, vocab_size);
            token = sample_from_probs(probs, vocab_size);
            break;

        default:
            token = greedy_sample(logits, vocab_size);
    }

    free(work);
    free(probs);
    return token;
}
```

---

## 9. 테스트 분포에서 전략 비교

```c
// Simple test: 8-token vocabulary with known logits
// Shows how each strategy affects which tokens get sampled
void test_sampling_strategies(void) {
    srand(42);

    const int V = 8;
    // Logits: token 0 dominates, but tokens 1-3 have meaningful mass
    float logits_orig[8] = { 4.0f, 2.5f, 2.0f, 1.5f, 0.5f, -1.0f, -2.0f, -3.0f };
    const char *names[8] = {"cat","mat","sat","hat","bat","rat","vat","pat"};

    float probs[8];
    softmax(probs, logits_orig, V);

    printf("=== Base distribution ===\n");
    for (int i = 0; i < V; i++)
        printf("  %s: %.3f\n", names[i], probs[i]);

    // Count how often each token is sampled (1000 trials)
    int counts[4][8] = {0};
    SamplerConfig cfgs[4] = {
        { SAMPLE_GREEDY, 1.0f, 0,  0.0f, 0.0f, 1.0f },
        { SAMPLE_TEMP,   1.2f, 0,  0.0f, 0.0f, 1.0f },
        { SAMPLE_TOP_K,  1.0f, 3,  0.0f, 0.0f, 1.0f },
        { SAMPLE_TOP_P,  1.0f, 0,  0.9f, 0.0f, 1.0f },
    };
    const char *cfg_names[4] = {"Greedy", "Temp=1.2", "Top-K=3", "Top-P=0.9"};

    for (int s = 0; s < 4; s++) {
        for (int trial = 0; trial < 1000; trial++) {
            float logits[8];
            memcpy(logits, logits_orig, sizeof(logits));
            int tok = sample_token(logits, V, &cfgs[s], NULL, 0);
            counts[s][tok]++;
        }
    }

    for (int s = 0; s < 4; s++) {
        printf("\n=== %s ===\n", cfg_names[s]);
        for (int i = 0; i < V; i++)
            printf("  %s: %d/1000\n", names[i], counts[s][i]);
    }
}

int main(void) {
    test_sampling_strategies();
    return 0;
}
```

예상 출력 패턴:
- **탐욕적**: "cat" 1000/1000 — 완전히 결정론적
- **Temp=1.2**: 꼬리 토큰으로 질량이 더 퍼짐
- **Top-K=3**: "cat", "mat", "sat"만 등장
- **Top-P=0.9**: 적응형 — "cat"과 "mat"만 등장할 가능성이 높음 (이미 ~90% 합계)

---

## 핵심 요약

- 탐욕적 디코딩은 결정론적이고 빠르지만 반복적이고 낮은 다양성의 출력을 생성하기 쉽습니다; 정확한 재현성이 필요한 작업에만 올바른 선택입니다.
- 온도는 softmax 전에 logit을 스케일링합니다: 1 미만의 값은 분포를 날카롭게 하고(더 집중), 1 초과의 값은 평탄하게 합니다(더 창의적).
- Top-k 필터링은 분포가 얼마나 집중되어 있거나 분산되어 있는지에 관계없이 고정된 수의 후보를 유지합니다 — 모델이 매우 확신할 때 잠재적인 불일치가 생깁니다.
- Top-p (nucleus) 샘플링은 분포 형태에 따라 후보 풀 크기를 조정하여, 모델이 확신할 때 자연스럽게 더 적은 토큰으로 축소됩니다.
- Min-p는 상위 토큰의 확률에 상대적으로 임계값을 스케일링하는 top-p의 더 간단한 O(n) 대안입니다.
- 반복 패널티는 양수와 음수 logit을 비대칭적으로 처리해야 합니다: 양수 logit은 나누고, 음수 logit은 곱하여 일관되게 확률을 줄입니다.
- 실제로, 온도 + top-p 결합(예: T=0.8, p=0.9)이 대부분의 프로덕션 추론 엔진에서 기본값입니다.

---

**이전**: [KV 캐시 최적화](./38_KV_Cache_Optimization.md) | **다음**: [양자화: INT8과 INT4](./40_Quantization_Int8_Int4.md)
