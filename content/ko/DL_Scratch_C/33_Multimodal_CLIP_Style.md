# 33. 멀티모달 CLIP 스타일 대조 학습

**이전**: [ViT 학습과 파인튜닝](./32_ViT_Training_and_Fine_Tuning.md) | **다음**: [Cross-Entropy 손실](./34_Cross_Entropy_Loss.md)

---

## 학습 목표

이 단원을 완료하면 다음을 수행할 수 있습니다:

1. 코사인 유사도 행렬에서 대칭 cross-entropy로서 InfoNCE 손실 유도 및 구현
2. CLIP 아키텍처 설명: ViT 이미지 인코더 + Transformer 텍스트 인코더
3. N×N 유사도 행렬 구성 및 temperature 스케일링 적용
4. 대조 목표 이해: 쌍을 이룬 (이미지, 텍스트) embedding 당기기, 비쌍 밀어내기
5. 텍스트 프롬프트 embedding을 사용한 제로샷 이미지 분류 구현

---

## 1. CLIP이란?

CLIP (Contrastive Language-Image Pretraining, Radford et al. 2021)은 두 인코더를 공동으로 학습합니다:
- **이미지 인코더**: 이미지 → embedding 벡터로 매핑하는 ViT (또는 ResNet)
- **텍스트 인코더**: 텍스트 → embedding 벡터로 매핑하는 Transformer

학습 신호는 인터넷에서 수집한 4억 개의 (이미지, 캡션) 쌍에서 옵니다. N 쌍의 각 배치에서 CLIP은 N개의 일치하는 (이미지, 텍스트) 쌍의 코사인 유사도를 최대화하면서 N²-N개의 불일치 쌍을 최소화합니다.

이는 `image_embed("a dog") ≈ text_embed("a photo of a dog")`와 같은 embedding을 생성합니다.

### 1.1 아키텍처 요약

```
이미지: [3, 224, 224]
  → 패치 embedding (ViT)
  → 12 Transformer 블록 (d_model=512, heads=8)
  → CLS 토큰 → 선형 투영 → image_embed [512]
  → L2 정규화 → image_feat [512]

텍스트: "a photo of a cat" → BPE 토큰 [T]
  → 토큰 + 위치 embedding [T, 512]
  → 12 Transformer 블록 (d_model=512, heads=8)
  → [EOS] 토큰 → 선형 투영 → text_embed [512]
  → L2 정규화 → text_feat [512]
```

두 인코더 모두 차원 `d_embed`의 **동일한** embedding 공간으로 투영합니다. L2 정규화 후 코사인 유사도는 내적으로 환원됩니다.

---

## 2. InfoNCE 손실

### 2.1 유도

N 개의 (이미지, 텍스트) 쌍의 배치가 주어지면 N×N 유사도 행렬 S를 계산합니다:

```
S[i][j] = dot(image_feat[i], text_feat[j]) / temperature
```

특징이 L2 정규화되어 있으므로 temperature τ (일반적으로 0.07)로 스케일된 코사인 유사도입니다.

손실은 두 개의 대칭 항을 가집니다:

```
L_image = (1/N) Σ_i  -log[ exp(S[i][i]) / Σ_j exp(S[i][j]) ]
L_text  = (1/N) Σ_j  -log[ exp(S[j][j]) / Σ_i exp(S[i][j]) ]
L_total = (L_image + L_text) / 2
```

`L_image`: 각 이미지에 대해 쌍을 이룬 텍스트가 N개의 모든 텍스트 중 가장 유사해야 합니다.
`L_text`: 각 텍스트에 대해 쌍을 이룬 이미지가 N개의 모든 이미지 중 가장 유사해야 합니다.

이는 대각선을 올바른 클래스로 하여 행별 (L_image)과 열별 (L_text) 분포에 적용된 cross-entropy 손실과 정확히 동일합니다.

### 2.2 구현

```c
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/*
 * l2_normalize — [N, D]의 각 행을 단위 노름으로 정규화.
 */
void l2_normalize(float *x, int N, int D) {
    for (int i = 0; i < N; i++) {
        float *row = x + i * D;
        float norm2 = 0.0f;
        for (int d = 0; d < D; d++) norm2 += row[d] * row[d];
        float inv = 1.0f / (sqrtf(norm2) + 1e-8f);
        for (int d = 0; d < D; d++) row[d] *= inv;
    }
}

/*
 * compute_similarity_matrix — S[N, N] = image_feat @ text_feat^T / tau 구성.
 *
 * image_feat : [N, D] L2 정규화됨
 * text_feat  : [N, D] L2 정규화됨
 * S          : [N, N] 출력 (행 = 이미지 인덱스, 열 = 텍스트 인덱스)
 */
void compute_similarity_matrix(const float *image_feat, const float *text_feat,
                               float *S, int N, int D, float temperature)
{
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float dot = 0.0f;
            for (int d = 0; d < D; d++) {
                dot += image_feat[i * D + d] * text_feat[j * D + d];
            }
            S[i * N + j] = dot / temperature;
        }
    }
}

/*
 * log_softmax_row — [N, N] 행렬의 각 행에 대해 log-softmax 계산.
 * 수치적으로 안정적: exp 전에 행 최댓값 빼기.
 * 결과를 인플레이스로 씀.
 */
void log_softmax_rows(float *S, int N) {
    for (int i = 0; i < N; i++) {
        float *row = S + i * N;
        /* 행 최댓값 찾기 */
        float mx = row[0];
        for (int j = 1; j < N; j++) if (row[j] > mx) mx = row[j];
        /* logsumexp 계산 */
        float sum = 0.0f;
        for (int j = 0; j < N; j++) sum += expf(row[j] - mx);
        float lse = mx + logf(sum);
        /* log softmax */
        for (int j = 0; j < N; j++) row[j] -= lse;
    }
}

/*
 * nll_diagonal — 대각선 요소의 평균 음의 로그-우도 계산.
 * 올바른 클래스가 항상 행 i에 대해 인덱스 i인 행렬에 대해.
 */
float nll_diagonal(const float *log_probs, int N) {
    float loss = 0.0f;
    for (int i = 0; i < N; i++) {
        loss -= log_probs[i * N + i];
    }
    return loss / (float)N;
}

/*
 * infonce_loss — 전체 CLIP InfoNCE 손실.
 *
 * image_feat : [N, D] L2 정규화된 이미지 embedding
 * text_feat  : [N, D] L2 정규화된 텍스트 embedding
 * temperature: 스칼라 τ (CLIP에서 학습됨, 여기서는 고정)
 * S_buf      : 스크래치 버퍼 [N, N] (호출자 제공)
 *
 * 스칼라 손실 반환.
 */
float infonce_loss(const float *image_feat, const float *text_feat,
                   int N, int D, float temperature, float *S_buf)
{
    /* 유사도 행렬 구성 */
    compute_similarity_matrix(image_feat, text_feat, S_buf, N, D, temperature);

    /* --- 이미지 측: 행별 softmax (각 이미지 vs 모든 텍스트) --- */
    /* 열별 (텍스트) 측을 위한 복사본 만들기 */
    float *S_T = (float *)malloc((size_t)N * N * sizeof(float));
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            S_T[j * N + i] = S_buf[i * N + j];   /* 전치 */

    log_softmax_rows(S_buf, N);
    float L_image = nll_diagonal(S_buf, N);

    /* --- 텍스트 측: 열별 softmax = S^T의 행별 --- */
    log_softmax_rows(S_T, N);
    float L_text = nll_diagonal(S_T, N);

    free(S_T);
    return (L_image + L_text) * 0.5f;
}
```

### 2.3 Backward Pass (개요)

유사도 행렬 S에 대한 InfoNCE의 gradient:

```
dL/dS[i][j] = (1/(2N)) * (
    (softmax_row[i][j] - 1_{i==j})    /* 이미지 측 gradient */
  + (softmax_col[i][j] - 1_{i==j})   /* 텍스트 측 gradient  */
)
```

여기서 `1_{i==j}`는 대각선에서 1, 그 외는 0입니다. 이는 내적을 통해 역전파하여 두 인코더의 gradient를 생성합니다.

---

## 3. Temperature 파라미터

CLIP은 temperature τ를 0.07로 초기화된 학습 가능한 파라미터로 학습합니다. 원시 파라미터는 `log_tau` (따라서 τ = exp(log_tau))이며 τ > 0을 보장합니다.

```c
typedef struct {
    float log_tau;     /* log of temperature, log(0.07)으로 초기화 */
    float grad_log_tau;
} TemperatureParam;

/* Forward: 학습된 temperature 적용 */
void apply_temperature(float *S, int N, float tau) {
    int total = N * N;
    for (int i = 0; i < total; i++) S[i] /= tau;
}

/*
 * Backward: log_tau에 대한 gradient.
 * S = dot / exp(log_tau)이므로, dL/dlog_tau = -sum(dL/dS * S)
 * (연쇄 법칙: d(dot/tau)/d(log_tau) = -dot/tau = -S)
 */
float grad_temperature(const float *dS, const float *S, int N) {
    float g = 0.0f;
    int total = N * N;
    for (int i = 0; i < total; i++) g += dS[i] * S[i];
    return -g;
}
```

분포 붕괴를 방지하기 위해 τ를 합리적인 범위로 고정: `τ ∈ [0.01, 1.0]`.

---

## 4. 제로샷 분류

학습 후 CLIP은 **레이블이 붙은 파인튜닝 데이터 없이** 이미지를 임의의 카테고리로 분류할 수 있습니다.

### 4.1 메커니즘

K 클래스가 있는 데이터셋의 경우 각 클래스에 대한 텍스트 프롬프트 구성:

```
"a photo of a {class_name}"
```

텍스트 인코더를 통해 K개의 프롬프트 인코딩 → `text_feats [K, D]`.
쿼리 이미지 인코딩 → `image_feat [D]`.
코사인 유사도 계산 → argmax 선택.

### 4.2 구현

```c
/*
 * encode_text_prompts — K개의 클래스 이름 프롬프트를 인코딩하고 text_feats에 저장.
 *
 * 실제 시스템에서는 Transformer 텍스트 인코더를 호출합니다.
 * 여기서는 text_encoder()가 함수 포인터로 사용 가능하다고 가정합니다.
 */
typedef void (*TextEncoderFn)(const int *tokens, int T, float *out, int D);

void encode_class_prompts(const char **class_names, int K,
                          TextEncoderFn encoder,
                          int *token_buf, int max_T,
                          float *text_feats, int D)
{
    for (int k = 0; k < K; k++) {
        /* 실제로는: "a photo of a {class_names[k]}" 토큰화 */
        /* 여기서 토큰화는 스텁으로 남겨둠 */
        int T = bpe_encode(class_names[k], token_buf, max_T); /* 사용자 정의 */
        encoder(token_buf, T, text_feats + k * D, D);
    }
    l2_normalize(text_feats, K, D);
}

/*
 * zero_shot_classify — 하나의 이미지에 대해 예측된 클래스 인덱스 반환.
 *
 * image_feat  : [D] L2 정규화된 이미지 embedding
 * text_feats  : [K, D] L2 정규화된 텍스트 embedding (클래스당 하나)
 */
int zero_shot_classify(const float *image_feat, const float *text_feats,
                       int K, int D)
{
    int best_k = 0;
    float best_sim = -1e30f;
    for (int k = 0; k < K; k++) {
        float sim = 0.0f;
        for (int d = 0; d < D; d++) {
            sim += image_feat[d] * text_feats[k * D + d];
        }
        if (sim > best_sim) { best_sim = sim; best_k = k; }
    }
    return best_k;
}

/*
 * zero_shot_topk — 유사도로 정렬된 상위-k 클래스 인덱스 반환.
 */
void zero_shot_topk(const float *image_feat, const float *text_feats,
                    int K, int D, int topk, int *indices, float *sims)
{
    /* 모든 유사도 계산 */
    float *all_sims = (float *)malloc((size_t)K * sizeof(float));
    for (int k = 0; k < K; k++) {
        float s = 0.0f;
        for (int d = 0; d < D; d++) s += image_feat[d] * text_feats[k * D + d];
        all_sims[k] = s;
        indices[k]  = k;
    }
    /* 상위-k를 위한 부분 선택 정렬 */
    for (int i = 0; i < topk; i++) {
        for (int j = i + 1; j < K; j++) {
            if (all_sims[indices[j]] > all_sims[indices[i]]) {
                int tmp = indices[i]; indices[i] = indices[j]; indices[j] = tmp;
            }
        }
        sims[i] = all_sims[indices[i]];
    }
    free(all_sims);
}
```

### 4.3 프롬프트 엔지니어링

CLIP의 제로샷 정확도는 프롬프트 표현에 크게 의존합니다. 효과적인 전략:

```c
/* 프롬프트 앙상블 — 여러 템플릿에 걸쳐 텍스트 embedding 평균 */
const char *templates[] = {
    "a photo of a %s",
    "a picture of a %s",
    "a photo of the %s",
    "an image of a %s",
    NULL
};

/*
 * encode_ensembled_prompts — 하나의 클래스에 대해 템플릿에 걸쳐 평균.
 * 단일 프롬프트보다 더 강건한 텍스트 embedding 생성.
 */
void encode_ensembled_prompts(const char *class_name,
                              const char **templates,
                              TextEncoderFn encoder,
                              int *token_buf, int max_T,
                              float *out, int D)
{
    memset(out, 0, (size_t)D * sizeof(float));
    int count = 0;
    for (int t = 0; templates[t] != NULL; t++) {
        char prompt[256];
        snprintf(prompt, sizeof(prompt), templates[t], class_name);
        /* 프롬프트 토큰화 → tokens */
        int T = bpe_encode(prompt, token_buf, max_T);
        float *tmp = (float *)malloc((size_t)D * sizeof(float));
        encoder(token_buf, T, tmp, D);
        /* 누적 (정규화 후) */
        for (int d = 0; d < D; d++) out[d] += tmp[d];
        free(tmp);
        count++;
    }
    /* 평균 embedding 정규화 */
    float norm2 = 0.0f;
    for (int d = 0; d < D; d++) norm2 += out[d] * out[d];
    float inv = 1.0f / (sqrtf(norm2) + 1e-8f);
    for (int d = 0; d < D; d++) out[d] *= inv;
    (void)count;
}
```

---

## 5. 규모에서의 CLIP: 효과의 이유

| 요소 | 효과 |
|---|---|
| 4억 개의 쌍 | 대용량 데이터셋이 수동 레이블 필요성 제거 |
| InfoNCE temperature 0.07 | 매우 뾰족한 분포 → 강한 학습 신호 |
| 대형 배치 크기 (32768) | 단계당 더 많은 네거티브 → 더 어려운 대조 작업 |
| 대칭 손실 | 두 인코더가 각 단계에서 공동으로 향상 |
| 별도의 투영 헤드 | 최종 embedding 공간이 인코더 표현과 분리됨 |

### 배치 크기와 InfoNCE

InfoNCE 손실의 품질은 더 큰 N (더 많은 네거티브)으로 향상됩니다. N=32768에서 각 이미지는 32767개의 다른 캡션과 구별되어야 합니다 — N=64보다 훨씬 어려운 작업. 이것이 CLIP이 256개 GPU에 분산된 대형 배치 크기를 사용하는 이유입니다.

우리의 C 구현에서는 사용 가능한 RAM에 의해 제한됩니다. D=512 embedding과 N=256 이미지의 배치: `256 × 512 × 4바이트 = 512KB` — 관리 가능합니다.

---

## 핵심 요약

- **InfoNCE**는 temperature로 스케일된 코사인 유사도 행렬의 대칭 cross-entropy입니다. 대각선에 올바른 (양성) 쌍이 있습니다.
- **Temperature τ**는 분포 날카로움을 제어합니다. 낮은 τ (≈0.07)는 모델이 예측에 매우 확신하게 만들어 강한 gradient 신호를 제공합니다.
- **L2 정규화**는 유사도 계산 전에 코사인 유사도를 단순한 내적으로 환원하여 O(N²·D)로 연산 가능합니다.
- **제로샷 분류**는 이미지 embedding과 텍스트 클래스 설명의 embedding을 비교하여 작동합니다 — 파인튜닝 데이터 불필요.
- **프롬프트 앙상블** (여러 템플릿에 걸쳐 embedding 평균)은 단일 프롬프트 대비 제로샷 정확도를 지속적으로 2-5% 향상시킵니다.
- **배치 크기는 대조 학습의 일급 하이퍼파라미터**: 더 많은 네거티브 = 더 나은 gradient = 더 나은 학습된 표현.
- CLIP 학습 목표는 구현하기 단순하지만 다운스트림 작업에서 지도 학습 기준선을 능가하려면 규모 (데이터 + 연산)가 필요합니다.

---

**이전**: [ViT 학습과 파인튜닝](./32_ViT_Training_and_Fine_Tuning.md) | **다음**: [Cross-Entropy 손실](./34_Cross_Entropy_Loss.md)

> 다음 단원에서는 언어 모델 학습을 위한 수치적으로 안정적인 cross-entropy를 다루며, fused softmax-CE backward pass를 포함합니다.
