# 34. 언어 모델 학습을 위한 Cross-Entropy 손실

**이전**: [멀티모달 CLIP 스타일 학습](./33_Multimodal_CLIP_Style.md) | **다음**: [옵티마이저](./35_Optimizers.md)

---

## 학습 목표

이 단원을 완료하면 다음을 수행할 수 있습니다:

1. log-softmax + NLL을 cross-entropy의 수치적으로 안정적인 형태로 유도
2. 최댓값 빼기 트릭을 사용한 logsumexp 구현
3. 전체 softmax 행렬 구체화를 피하는 fused softmax-CE backward pass 구현
4. 다음 토큰 예측에 cross-entropy 적용 (이동된 레이블 관례)
5. 지수화된 평균 cross-entropy 손실로서 perplexity 계산

---

## 1. Cross-Entropy: 유도

V 클래스의 분류 작업에서 원시 모델 출력은 logit 벡터 `z ∈ ℝᵛ`입니다. 클래스 `c`에 대한 예측 확률:

```
p_c = exp(z_c) / Σ_k exp(z_k)   (softmax)
```

실제 클래스 `y`에 대한 cross-entropy 손실:

```
L = -log(p_y)
  = -log(exp(z_y) / Σ_k exp(z_k))
  = -z_y + log(Σ_k exp(z_k))
  = -z_y + logsumexp(z)
```

이것이 정확히 `-z_y + logsumexp(z)`입니다. softmax를 명시적으로 계산할 필요 없음 — 올바른 클래스의 원시 logit과 모든 클래스에 대한 logsumexp만 필요합니다.

### 순진한 Softmax가 오버플로우하는 이유

```
exp(1000.0f) = +inf   (IEEE 754 오버플로우)
exp(-1000.0f) = 0.0f  (언더플로우)
```

언어 모델 logit은 특히 학습 초기에 크기가 클 수 있습니다.

---

## 2. Logsumexp: 수치적으로 안정적

```c
#include <math.h>
#include <float.h>

/*
 * logsumexp — log(Σ exp(x[i]))을 안정적으로 계산.
 *
 * 지수화 전에 최댓값 빼기:
 *   log(Σ exp(x_i)) = max_x + log(Σ exp(x_i - max_x))
 *
 * x  : 길이 n의 입력 배열
 * n  : 요소 수
 */
float logsumexp(const float *x, int n) {
    /* 최댓값 찾기 */
    float mx = x[0];
    for (int i = 1; i < n; i++) if (x[i] > mx) mx = x[i];
    /* 안정적인 합 */
    float sum = 0.0f;
    for (int i = 0; i < n; i++) sum += expf(x[i] - mx);
    return mx + logf(sum);
}
```

안정성 증명: `max_x`를 빼면 모든 지수는 ≤ 0이므로 `exp(x_i - max_x) ∈ (0, 1]`. 오버플로우 없음, 언더플로우 없음 (모든 logit이 `-∞`인 퇴화 경우 제외).

---

## 3. Log-Softmax

```c
/*
 * log_softmax — 각 요소에 대해 log(softmax(x)) 계산.
 *
 * 수치적으로 동일:
 *   log_softmax(x)[i] = x[i] - logsumexp(x)
 *
 * x   : 입력 logit  [n]
 * out : 출력        [n]
 */
void log_softmax(const float *x, float *out, int n) {
    float lse = logsumexp(x, n);
    for (int i = 0; i < n; i++) out[i] = x[i] - lse;
}

/*
 * nll_loss — 단일 예시의 음의 로그-우도.
 *
 * log_probs : log_softmax의 출력 [V]
 * target    : 올바른 클래스 인덱스
 */
float nll_loss(const float *log_probs, int target) {
    return -log_probs[target];
}
```

---

## 4. 배치 Cross-Entropy (순진한 방법 — 참고용)

```c
/*
 * cross_entropy_forward_naive — 배치에 걸쳐 평균 CE 손실 계산.
 *
 * logits   : [N, V]  원시 모델 출력
 * targets  : [N]     올바른 클래스 인덱스
 * losses   : [N]     샘플당 CE 손실 (출력)
 * N        : 배치 크기
 * V        : 어휘 크기
 *
 * log_probs [N, V] 구체화 — O(N*V) 메모리.
 */
float cross_entropy_forward_naive(const float *logits, const int *targets,
                                  float *losses, int N, int V)
{
    float total = 0.0f;
    for (int i = 0; i < N; i++) {
        const float *row = logits + i * V;
        float lse = logsumexp(row, V);
        losses[i] = -(row[targets[i]] - lse);
        total += losses[i];
    }
    return total / (float)N;
}
```

문제: V=50257 (GPT-2 어휘)와 N=B×T=16×1024=16384 토큰/배치의 언어 모델에서 logit 행렬은 `16384 × 50257 × 4바이트 ≈ 3.3 GB`. 이는 LLM 학습에서 가장 메모리 집약적인 연산입니다.

---

## 5. Fused Softmax-CE Forward + Backward

핵심 통찰: **logit에 대한 cross-entropy의 backward pass는 단순히 `softmax(z) - one_hot(y)`**이며, `1/N`으로 스케일됩니다. log_probs를 저장할 필요 없음 — forward pass에서 손실을 계산하고 gradient를 직접 계산합니다.

### 5.1 Fused Backward 유도

```
L = (1/N) Σ_i [ -z_{i,y_i} + logsumexp(z_i) ]

∂L/∂z_{i,j} = (1/N) * [ -1_{j=y_i} + exp(z_{i,j}) / Σ_k exp(z_{i,k}) ]
             = (1/N) * [ softmax(z_i)[j] - 1_{j=y_i} ]
```

따라서 `dlogits = (softmax(z) - one_hot(y)) / N`.

backward pass에서 명시적인 log_probs 행렬 불필요 — softmax 확률 (행별로 계산)과 타겟 인덱스만 필요합니다.

### 5.2 Fused 구현

```c
#include <string.h>
#include <stdlib.h>

/*
 * fused_ce_forward_backward — 한 번의 패스에서 CE 손실과 gradient 계산.
 *
 * logits   : [N, V]  입력 logit (읽기 전용)
 * targets  : [N]     올바른 클래스 인덱스
 * dlogits  : [N, V]  출력 gradient (여기에 씀)
 * N        : 배치의 토큰 수
 * V        : 어휘 크기
 *
 * 반환: 스칼라 평균 cross-entropy 손실.
 *
 * 메모리: 한 번에 [V] float 한 행만 할당 — O(N*V)가 아닌 O(V).
 */
float fused_ce_forward_backward(const float *logits, const int *targets,
                                float *dlogits, int N, int V)
{
    float *probs = (float *)malloc((size_t)V * sizeof(float));
    float total_loss = 0.0f;
    float scale = 1.0f / (float)N;

    for (int i = 0; i < N; i++) {
        const float *row   = logits  + i * V;
        float       *drow  = dlogits + i * V;

        /* 안정적인 softmax: 최댓값 찾기 */
        float mx = row[0];
        for (int j = 1; j < V; j++) if (row[j] > mx) mx = row[j];

        /* 정규화되지 않은 exp 계산 */
        float sum = 0.0f;
        for (int j = 0; j < V; j++) {
            probs[j] = expf(row[j] - mx);
            sum += probs[j];
        }
        /* softmax 확률로 정규화 */
        float inv_sum = 1.0f / sum;
        for (int j = 0; j < V; j++) probs[j] *= inv_sum;

        /* 이 토큰의 cross-entropy 손실 */
        total_loss += -logf(probs[targets[i]] + 1e-10f);

        /* Gradient: dlogits = (softmax - one_hot) / N */
        for (int j = 0; j < V; j++) {
            drow[j] = probs[j] * scale;
        }
        drow[targets[i]] -= scale;   /* 올바른 클래스에서 1/N 빼기 */
    }

    free(probs);
    return total_loss / (float)N;
}
```

이 구현은 O(N·V) 시간과 O(V) 메모리 (행당)로 최적입니다. O(N·V) 메모리가 필요한 순진한 버전과 비교하세요.

### 5.3 Fused가 Softmax 행렬 구체화를 피하는 이유

순진한 방법:
1. Forward: log_probs [N, V] 계산 → 메모리에 저장 → 타겟 읽기 → 손실 계산
2. Backward: log_probs 다시 읽기 → gradient 계산

Fused 방법:
1. 행별로 softmax 계산 (한 번에 V 크기의 한 행)
2. 즉시 해당 행의 손실 기여와 gradient 계산
3. dlogits에 gradient 쓰기 → softmax 행 버리기

최대 메모리 사용: O(N·V) 대신 O(V). V=50257과 N=16384에서 이 연산의 메모리가 16384배 감소합니다.

---

## 6. 다음 토큰 예측을 위한 Cross-Entropy

언어 모델은 다음 토큰 예측으로 학습됩니다: 토큰 `[t_0, t_1, ..., t_{T-1}]`이 주어지면 `[t_1, t_2, ..., t_T]` 예측.

### 6.1 레이블 이동

```c
/*
 * shift_labels — 다음 토큰 예측을 위한 (inputs, targets) 추출.
 *
 * tokens  : [B, T+1]  원시 토큰 시퀀스 (최종 타겟 포함)
 * inputs  : [B, T]    토큰 0..T-1 (모델 입력)
 * targets : [B, T]    토큰 1..T   (예측 타겟)
 * B       : 배치 크기
 * T       : 시퀀스 길이 (모델은 T 토큰 보고, T 토큰 예측)
 */
void shift_labels(const int *tokens, int *inputs, int *targets, int B, int T) {
    for (int b = 0; b < B; b++) {
        const int *seq = tokens + b * (T + 1);
        int *inp = inputs  + b * T;
        int *tgt = targets + b * T;
        for (int t = 0; t < T; t++) {
            inp[t] = seq[t];
            tgt[t] = seq[t + 1];
        }
    }
}

/*
 * LM을 위한 학습 cross-entropy:
 *
 * logits  : [B, T, V]   모델 출력
 * targets : [B, T]      다음 토큰 레이블 (1 이동)
 *
 * [B*T, V]와 [B*T]로 평탄화 → fused_ce_forward_backward 호출.
 */
float lm_cross_entropy(const float *logits, const int *targets,
                       float *dlogits, int B, int T, int V)
{
    int N = B * T;
    return fused_ce_forward_backward(logits, targets, dlogits, N, V);
}
```

### 6.2 패딩 토큰 무시

서로 다른 길이의 시퀀스를 패딩과 함께 배치할 때 손실에서 패딩 위치 무시:

```c
/*
 * lm_cross_entropy_masked — target == pad_id인 위치 무시.
 *
 * 반환: 비패딩 위치에 대한 평균 손실.
 */
float lm_cross_entropy_masked(const float *logits, const int *targets,
                               float *dlogits, int B, int T, int V,
                               int pad_id)
{
    float *probs  = (float *)malloc((size_t)V * sizeof(float));
    float total_loss = 0.0f;
    int   count      = 0;

    /* 첫 번째 패스: 정규화를 위해 비패딩 토큰 수 세기 */
    int N = B * T;
    for (int i = 0; i < N; i++) if (targets[i] != pad_id) count++;

    float scale = (count > 0) ? 1.0f / (float)count : 0.0f;

    for (int i = 0; i < N; i++) {
        const float *row  = logits  + i * V;
        float       *drow = dlogits + i * V;

        if (targets[i] == pad_id) {
            /* 패딩 위치에서 gradient 0으로 설정 */
            memset(drow, 0, (size_t)V * sizeof(float));
            continue;
        }

        /* 안정적인 softmax */
        float mx = row[0];
        for (int j = 1; j < V; j++) if (row[j] > mx) mx = row[j];
        float sum = 0.0f;
        for (int j = 0; j < V; j++) { probs[j] = expf(row[j] - mx); sum += probs[j]; }
        float inv = 1.0f / sum;
        for (int j = 0; j < V; j++) probs[j] *= inv;

        total_loss += -logf(probs[targets[i]] + 1e-10f);
        for (int j = 0; j < V; j++) drow[j] = probs[j] * scale;
        drow[targets[i]] -= scale;
    }

    free(probs);
    return (count > 0) ? total_loss / (float)count : 0.0f;
}
```

---

## 7. Perplexity

Perplexity는 언어 모델의 표준 평가 지표입니다:

```
Perplexity = exp(mean_CE_loss)
```

평균 CE 손실이 3.0인 모델의 경우 perplexity = exp(3.0) ≈ 20.1. 이는 모델이 (평균적으로) 20개의 선택지 중에서 균일하게 선택하는 것만큼 혼란스럽다는 의미입니다.

GPT-2는 WikiText-103에서 ~18.3 perplexity를 달성합니다. Shakespeare의 문자 수준 모델은 일반적으로 ~1.4-1.6 bits/char ≈ perplexity 2.6-3.0에 도달합니다.

```c
#include <math.h>

/*
 * compute_perplexity — 평균 CE 손실이 주어지면 perplexity 반환.
 * loss: 평균 cross-entropy (자연수 밑, 밑 e)
 */
float compute_perplexity(float loss) {
    return expf(loss);
}

/*
 * eval_perplexity — 평가 세트에서 모델 실행, perplexity 반환.
 *
 * eval_tokens   : [num_eval_tokens]  토큰화된 검증 세트
 * model_forward : 모델 forward pass를 위한 함수 포인터
 * B, T, V       : 배치, 시퀀스 길이, 어휘 크기
 */
float eval_perplexity(const int *eval_tokens, int num_eval_tokens,
                      /* 모델 forward fn: tokens[B,T] → logits[B,T,V] */
                      void (*model_forward)(const int *tokens, float *logits,
                                           int B, int T),
                      int B, int T, int V)
{
    float *logits  = (float *)malloc((size_t)B * T * V * sizeof(float));
    float *dlogits = (float *)malloc((size_t)B * T * V * sizeof(float));
    int   *inputs  = (int   *)malloc((size_t)B * T     * sizeof(int));
    int   *targets = (int   *)malloc((size_t)B * T     * sizeof(int));

    float total_loss = 0.0f;
    int   num_batches = (num_eval_tokens / (B * (T + 1)));

    for (int b = 0; b < num_batches; b++) {
        const int *batch_tokens = eval_tokens + b * B * (T + 1);
        shift_labels(batch_tokens, inputs, targets, B, T);
        model_forward(inputs, logits, B, T);
        float loss = lm_cross_entropy(logits, targets, dlogits, B, T, V);
        total_loss += loss;
    }

    free(logits); free(dlogits); free(inputs); free(targets);

    float mean_loss = (num_batches > 0) ? total_loss / (float)num_batches : 0.0f;
    return compute_perplexity(mean_loss);
}
```

---

## 핵심 요약

- **Cross-entropy = -z_y + logsumexp(z)**: forward pass에서 softmax를 명시적으로 계산할 필요 없음. 올바른 클래스 logit과 logsumexp만 필요합니다.
- **Logsumexp 안정성**: 지수화 전에 항상 행 최댓값을 빼세요. 이는 잠재적 오버플로우/언더플로우를 무해한 `log(1) = 0` 항으로 변환합니다.
- **Fused backward**: `dlogits[i][j] = (softmax[i][j] - one_hot[i][y_i]) / N`. 전체 [N, V] 로그 확률 행렬 저장 방지 — V=50257이고 N=16384일 때 중요합니다.
- **레이블 이동**: 다음 토큰 예측에서 입력은 `tokens[0..T-1]`이고 타겟은 `tokens[1..T]`. 여기서의 off-by-one 오류는 학습 버그의 일반적인 원인입니다.
- **Perplexity** = `exp(mean_CE_loss)`. 표준 LM 평가 지표입니다. 3.0에서 2.85 nats 감소는 perplexity가 20.1에서 17.3으로 떨어지는 것에 해당 — 의미 있는 향상입니다.
- **패딩 마스킹**: 서로 다른 길이의 시퀀스를 혼합할 때 손실과 gradient 모두에서 패딩 위치를 제외하여 의미 없는 토큰 학습을 피하세요.

---

**이전**: [멀티모달 CLIP 스타일 학습](./33_Multimodal_CLIP_Style.md) | **다음**: [옵티마이저](./35_Optimizers.md)

> 다음 단원에서는 함수 포인터를 사용하여 SGD with momentum, Adam, AdamW, gradient clipping, LR 스케줄링을 구현합니다.
