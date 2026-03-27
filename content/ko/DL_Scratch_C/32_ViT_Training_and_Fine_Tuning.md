# 32. ViT 학습과 파인튜닝

**이전**: [Vision Transformer (ViT)](./31_Vision_Transformer_ViT.md) | **다음**: [멀티모달 CLIP 스타일 학습](./33_Multimodal_CLIP_Style.md)

---

## 학습 목표

이 단원을 완료하면 다음을 수행할 수 있습니다:

1. C에서 선형 워밍업을 포함한 코사인 LR 스케줄 구현
2. ViT 입력에 대해 패치 수준의 CutMix 증강 적용
3. 정규화를 위해 ViT 블록에 stochastic depth (drop path) 추가
4. 파인튜닝 전략 설계: 분류 헤드 교체, 하위 레이어 동결, 점진적 해동
5. 다운스트림 작업에서 처음부터 학습하는 ViT와 사전 학습된 모델 파인튜닝 비교

---

## 1. ViT 학습이 CNN 학습과 다른 이유

CNN은 이동 등변성과 로컬 연결성 같은 귀납적 편향이 내장되어 있습니다. ViT는 둘 다 없습니다. 모든 토큰이 레이어 1부터 다른 모든 토큰에 attend합니다. 이는 강력하지만 ViT가 일반화하려면 훨씬 더 많은 데이터 또는 강한 정규화가 필요함을 의미합니다.

학습 레시피의 주요 차이점:
- **더 긴 워밍업**: ViT는 느린 LR 증가가 유리 (CNN의 1-5 에포크 대비 5-20 에포크)
- **코사인 감쇠**: 단계적 LR 감소의 갑작스러운 손실 급등 방지
- **강한 증강**: RandAugment, Mixup, CutMix — 임의 크롭/플립만이 아님
- **Stochastic depth**: 학습 중 전체 Transformer 블록을 임의로 제거
- **레이블 스무싱**: 스무스 타겟 (0.1)과의 cross-entropy가 보정에 도움

---

## 2. 워밍업 + 코사인 LR 스케줄

DeiT와 ViT-Base 학습에 사용되는 LR 스케줄:

```
                    lr_max
                   /        \
                  /    코사인 \
                 /     감쇠   \_________ lr_min
     워밍업   /
lr_base ______/
              0   warmup_steps         total_steps
```

```c
#include <math.h>
#include <stddef.h>

typedef struct {
    float lr_min;       /* 최종 LR, 예: 1e-6                         */
    float lr_max;       /* 최고 LR, 예: 1e-3                          */
    int   warmup_steps; /* 선형 증가 기간                         */
    int   total_steps;  /* 전체 학습 단계                         */
} LRSchedule;

/* 주어진 단계 (0-인덱스)의 학습률 반환. */
float cosine_lr_with_warmup(const LRSchedule *s, int step) {
    if (step < s->warmup_steps) {
        /* 선형 워밍업: 0 → lr_max */
        return s->lr_max * (float)(step + 1) / (float)s->warmup_steps;
    }
    /* 코사인 어닐링: lr_max → lr_min */
    int decay_steps = s->total_steps - s->warmup_steps;
    int t = step - s->warmup_steps;
    float progress = (float)t / (float)decay_steps;          /* 0 → 1  */
    float cosine   = 0.5f * (1.0f + cosf((float)M_PI * progress));
    return s->lr_min + (s->lr_max - s->lr_min) * cosine;
}

/* 예: ImageNet의 300에포크 ViT-Base
 *   steps_per_epoch = 1281167 / 1024 ≈ 1251
 *   total_steps = 300 * 1251 = 375300
 *   warmup_steps = 10 * 1251 = 12510  (10에포크 워밍업)
 */
```

각 단계에서 `cosine_lr_with_warmup`을 호출하고 `adam_update()` 전에 옵티마이저의 `lr` 필드를 설정하여 학습 루프에 적용합니다.

---

## 3. CutMix 증강 (패치 수준)

CutMix는 이미지 A의 직사각형 영역을 이미지 B의 해당 영역으로 교체하고 레이블을 비례적으로 혼합합니다. ViT의 경우 자연적인 단위가 픽셀이 아닌 **패치**이므로 패치 수준 CutMix를 구현합니다: 이미지 B의 전체 패치가 이미지 A의 전체 패치를 대체합니다.

### 3.1 CutMix가 ViT에 효과적인 이유

CutMix는 모델이 부분 정보 (마스킹된 패치 시퀀스)에서 결정을 내리도록 강제합니다. 이는 MAE (masked autoencoder) 사전 학습과 밀접하게 관련됨 — 둘 다 모델이 누락되거나 교체된 패치에 강건하도록 요구합니다.

### 3.2 구현

```c
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* [0, 1)의 균일 임의 float. */
static float randf(void) { return (float)rand() / ((float)RAND_MAX + 1.0f); }

/*
 * cutmix_patches — ViT 입력에 대한 패치 수준 CutMix.
 *
 * tokens_a, tokens_b : [num_patches, d_model]  (행 우선)
 * out                 : [num_patches, d_model]  (출력)
 * lambda              : Beta(alpha, alpha)에서 추출된 혼합 비율
 *
 * tokens_b에서 연속된 직사각형 패치 블록을 임의 선택하고
 * tokens_a에 붙여넣습니다. 실제 lambda는 교체된 패치 비율로
 * 재계산되어 레이블 혼합이 정확하게 일치합니다.
 */
float cutmix_patches(const float *tokens_a, const float *tokens_b,
                     float *out,
                     int grid_h, int grid_w, int d_model,
                     float lambda_target)
{
    int num_patches = grid_h * grid_w;

    /* 먼저 이미지 A를 출력에 복사 */
    memcpy(out, tokens_a, (size_t)num_patches * d_model * sizeof(float));

    /* sqrt(1 - lambda)에 비례하는 컷 크기 계산 */
    float ratio = sqrtf(1.0f - lambda_target);
    int cut_h = (int)(grid_h * ratio);
    int cut_w = (int)(grid_w * ratio);
    if (cut_h < 1) cut_h = 1;
    if (cut_w < 1) cut_w = 1;

    /* 컷 영역의 임의 좌상단 모서리 */
    int r0 = (int)(randf() * (grid_h - cut_h + 1));
    int c0 = (int)(randf() * (grid_w - cut_w + 1));

    /* 이미지 B에서 패치 붙여넣기 */
    for (int r = r0; r < r0 + cut_h; r++) {
        for (int c = c0; c < c0 + cut_w; c++) {
            int idx = r * grid_w + c;          /* 패치 인덱스 */
            const float *src = tokens_b + idx * d_model;
            float       *dst = out       + idx * d_model;
            memcpy(dst, src, (size_t)d_model * sizeof(float));
        }
    }

    /* 실제 lambda = 이미지 A에서 온 패치 비율 */
    float actual_lambda = 1.0f - (float)(cut_h * cut_w) / (float)num_patches;
    return actual_lambda;
}

/*
 * CutMix 레이블 혼합:
 *   mixed_loss = lambda * CE(pred, label_a) + (1 - lambda) * CE(pred, label_b)
 */
```

학습 루프에서의 사용:

```c
/* 배치 루프 내부 */
float lambda_target = sample_beta(alpha, alpha);   /* alpha = 1.0 일반적 */
float lambda = cutmix_patches(
    tokens_a, tokens_b, mixed_tokens,
    14, 14, d_model, lambda_target
);
/* mixed_tokens에 대해 forward pass → logits */
/* loss = lambda * CE(logits, label_a) + (1-lambda) * CE(logits, label_b) */
```

---

## 4. Stochastic Depth (Drop Path)

Stochastic depth는 학습 중 Transformer 블록을 임의로 건너뜁니다. 각 블록은 깊이에 따라 선형으로 감소하는 생존 확률 `p_i`를 가집니다 (더 깊은 레이어가 더 자주 제거됨).

```
p_i = 1 - (i / L) * drop_rate
```

여기서 `L`은 전체 블록 수이고 `drop_rate`는 하이퍼파라미터입니다 (ViT-Base의 경우 일반적으로 0.1-0.2).

### 4.1 Drop Path를 포함한 Forward Pass

```c
#include <stdlib.h>

/*
 * drop_path_forward — 잔차 추가에 stochastic depth 적용.
 *
 * 학습 중 확률 (1 - survival_prob)로 잔차를 완전히 건너뜁니다;
 * 그렇지 않으면 기대값을 유지하기 위해 1 / survival_prob로 스케일합니다
 * (베르누이 스케일링).
 *
 * x        : [B, T, d_model]  입력 (bypass 브랜치)
 * residual : [B, T, d_model]  서브레이어의 출력
 * out      : [B, T, d_model]  x + scaled_residual
 * B, T, D  : 배치, 시퀀스 길이, 히든 차원
 * survival : 이 블록의 생존 확률
 * training : 1 = 학습 모드, 0 = 추론
 * mask     : 사전 할당된 [B] 불리언 마스크 (호출자 제공)
 *
 * 샘플당 드롭 마스크를 반환합니다 (1 = 유지, 0 = 제거).
 */
void drop_path_forward(const float *x, const float *residual, float *out,
                       int B, int T, int D,
                       float survival, int training, int *mask)
{
    int total = B * T * D;

    if (!training || survival >= 1.0f) {
        /* 추론: 항상 잔차 추가 (스케일링 불필요) */
        for (int i = 0; i < total; i++) out[i] = x[i] + residual[i];
        return;
    }

    /* 샘플당 베르누이 마스크 샘플링 */
    for (int b = 0; b < B; b++) {
        float u  = (float)rand() / ((float)RAND_MAX + 1.0f);
        mask[b]  = (u < survival) ? 1 : 0;
    }

    float scale = 1.0f / survival;   /* 기대값 보정 */

    for (int b = 0; b < B; b++) {
        float s = mask[b] ? scale : 0.0f;
        for (int t = 0; t < T; t++) {
            for (int d = 0; d < D; d++) {
                int i = (b * T + t) * D + d;
                out[i] = x[i] + s * residual[i];
            }
        }
    }
}

/*
 * drop_path_backward — drop path를 통한 gradient.
 *
 * dout     : 업스트림 gradient [B, T, D]
 * mask     : forward에서 사용된 동일한 마스크  [B]
 * dx       : x에 대한 gradient         [B, T, D]  (누산)
 * dresidual: 잔차에 대한 gradient  [B, T, D]  (누산)
 */
void drop_path_backward(const float *dout, const int *mask,
                        float *dx, float *dresidual,
                        int B, int T, int D, float survival)
{
    float scale = 1.0f / survival;
    for (int b = 0; b < B; b++) {
        float s = mask[b] ? scale : 0.0f;
        for (int t = 0; t < T; t++) {
            for (int d = 0; d < D; d++) {
                int i = (b * T + t) * D + d;
                dx[i]        += dout[i];          /* bypass는 항상 grad 통과 */
                dresidual[i] += s * dout[i];
            }
        }
    }
}

/* L 블록에 걸쳐 선형으로 생존 확률 할당 */
void compute_survival_probs(float *probs, int L, float drop_rate) {
    for (int i = 0; i < L; i++) {
        probs[i] = 1.0f - ((float)i / (float)L) * drop_rate;
    }
}
```

12 블록과 `drop_rate=0.1`인 ViT-Base:
- 블록 0: 생존 = 1.00 (절대 제거 안 됨)
- 블록 6: 생존 = 0.95
- 블록 11: 생존 = 0.90

---

## 5. 파인튜닝 전략

사전 학습된 ViT (예: ImageNet-21k에서 학습)를 더 작은 다운스트림 데이터셋에 파인튜닝하려면 파국적 망각을 피하기 위한 주의가 필요합니다.

### 5.1 표준 파인튜닝 프로토콜

```
단계 1 — 헤드 교체 (1-2 에포크):
  - 사전 학습 가중치 불러오기
  - 분류 헤드 교체 (선형: d_model → num_classes_pretrain)
    → 새 헤드 (선형: d_model → num_classes_target)
  - 새 헤드를 제외한 모든 파라미터 동결
  - 높은 LR (1e-3)로 학습하여 헤드 초기화

단계 2 — 점진적 해동 (나머지 학습):
  - 블록 11 (마지막) 해동, K 단계 학습
  - 블록 10 해동, K 단계 학습
  - ...
  - 패치 embedding + 위치 인코딩 마지막에 해동
  - 코사인 감쇠로 낮은 LR (1e-5 ~ 1e-4) 사용
```

### 5.2 구현

```c
typedef struct {
    int   num_blocks;
    int   freeze_until_block; /* 블록 0..freeze_until_block 동결 */
    float head_lr;            /* 분류 헤드의 LR          */
    float backbone_lr;        /* 해동된 백본 레이어의 LR         */
} FinetuneConfig;

/*
 * param_requires_grad — 파라미터를 갱신해야 하면 1 반환.
 * layer_id로 레이어 소유권 인코딩:
 *   -1  = 패치 embedding / 위치 인코딩
 *    0  = 블록 0
 *   ...
 *   11  = 블록 11
 *   12  = 분류 헤드
 */
int param_requires_grad(int layer_id, const FinetuneConfig *cfg) {
    if (layer_id == 12) return 1;                        /* 헤드: 항상 학습  */
    if (layer_id < 0)   return (cfg->freeze_until_block < 0);
    return (layer_id > cfg->freeze_until_block);
}

/*
 * get_lr_for_layer — 레이어별 LR 감쇠 적용 (선택적이지만 효과적).
 * 각 레이어는 lr * decay^(L - layer_id) 획득.
 */
float get_lr_for_layer(int layer_id, float base_lr, float decay, int L) {
    if (layer_id == L) return base_lr;               /* 헤드는 전체 LR  */
    int depth = L - layer_id;
    float lr  = base_lr;
    for (int i = 0; i < depth; i++) lr *= decay;     /* lr * decay^depth   */
    return lr;
}
```

### 5.3 헤드 해상도 불일치

다른 입력 해상도의 데이터셋에 파인튜닝할 때 (예: 224px에서 사전 학습, 384px에서 파인튜닝), 패치 수가 변하고 위치 embedding을 보간해야 합니다:

```c
#include <math.h>

/*
 * interpolate_pos_embed — 위치 embedding의 이중선형 보간.
 *
 * src      : [N_src + 1, d_model]  소스 위치 embedding (CLS 포함)
 * dst      : [N_dst + 1, d_model]  출력 (호출자 할당)
 * grid_src : 소스 그리드 크기 (예: 224px / 16px 패치의 경우 14)
 * grid_dst : 타겟 그리드 크기 (예: 384px / 16px 패치의 경우 24)
 * d_model  : embedding 차원
 *
 * CLS 토큰 embedding은 보간 없이 그대로 복사.
 */
void interpolate_pos_embed(const float *src, float *dst,
                           int grid_src, int grid_dst, int d_model)
{
    int N_src = grid_src * grid_src;
    int N_dst = grid_dst * grid_dst;

    /* CLS 토큰 (첫 번째 행) 그대로 복사 */
    memcpy(dst, src, (size_t)d_model * sizeof(float));

    const float *src_patches = src + d_model;   /* CLS 건너뜀 */
    float       *dst_patches = dst + d_model;

    /* 이중선형 보간 (위치 embed에 충분) */
    for (int row = 0; row < grid_dst; row++) {
        for (int col = 0; col < grid_dst; col++) {
            /* 목적지 그리드 위치를 소스 그리드 위치로 매핑 */
            float sr = (row + 0.5f) * (float)grid_src / (float)grid_dst - 0.5f;
            float sc = (col + 0.5f) * (float)grid_src / (float)grid_dst - 0.5f;

            int r0 = (int)sr; if (r0 < 0) r0 = 0;
            int c0 = (int)sc; if (c0 < 0) c0 = 0;
            int r1 = r0 + 1;  if (r1 >= grid_src) r1 = grid_src - 1;
            int c1 = c0 + 1;  if (c1 >= grid_src) c1 = grid_src - 1;

            float dr = sr - r0;
            float dc = sc - c0;

            float *out_row = dst_patches + (row * grid_dst + col) * d_model;
            for (int d = 0; d < d_model; d++) {
                float v00 = src_patches[(r0 * grid_src + c0) * d_model + d];
                float v01 = src_patches[(r0 * grid_src + c1) * d_model + d];
                float v10 = src_patches[(r1 * grid_src + c0) * d_model + d];
                float v11 = src_patches[(r1 * grid_src + c1) * d_model + d];
                out_row[d] = v00 * (1-dr)*(1-dc) + v01 * (1-dr)*dc
                           + v10 * dr*(1-dc)      + v11 * dr*dc;
            }
        }
    }
    (void)N_src; (void)N_dst;
}
```

---

## 6. 처음부터 학습 vs. 파인튜닝

| 측면 | 처음부터 학습 | 파인튜닝 |
|---|---|---|
| 필요한 데이터 | >1000만 이미지 (ImageNet-21k 규모) | 수천 장도 가능 |
| 학습 시간 | 300 에포크 (8× A100에서 며칠) | 10-30 에포크 |
| LR 스케줄 | 10에포크 워밍업의 코사인 | 1에포크 워밍업의 코사인 |
| 배치 크기 | 4096 (그래디언트 누산 포함) | 512 |
| 증강 | RandAugment + CutMix + Mixup | RandAugment 또는 단순 크롭/플립 |
| Stochastic depth | drop_rate = 0.1 | drop_rate = 0.0 또는 매우 작음 |
| 레이블 스무싱 | 0.1 | 0.0 또는 0.05 |
| 정규화 | 가중치 감쇠 0.05, 드롭아웃 0.0 | 가중치 감쇠 0.01 |
| 예상 정확도 (ImageNet) | ~81.8% (ViT-Base/16) | ImageNet-21k 초기화로 85%+ |

### 핵심 통찰

ImageNet-1k (128만 이미지)에서 처음부터 학습된 ViT는 약 77-78% top-1만 달성 — 훨씬 적은 파라미터를 가진 ResNet-50 (~76%)보다 열등합니다. 하지만 ImageNet-21k (1400만 이미지)에서 사전 학습한 후 ImageNet-1k에서 파인튜닝한 ViT-Base는 85%+에 도달합니다. 데이터가 충분할 때는 CNN의 귀납적 편향이 덜 중요합니다.

---

## 핵심 요약

- **워밍업이 포함된 코사인 LR**은 ViT의 표준 스케줄입니다. 워밍업은 임의로 초기화된 attention 가중치의 큰 gradient로 인한 초기 불안정성을 방지합니다.
- **패치 수준 CutMix**는 ViT의 토큰화된 표현과 자연스럽게 정렬되며 암묵적 마스킹 정규화기 역할을 합니다.
- **Stochastic depth**는 초기 레이어에 높은 생존 확률을 (데이터에서 학습된 특징 보존), 후기 레이어에 낮은 생존 확률을 부여합니다 (상단에서 더 많은 정규화 허용).
- **파인튜닝 프로토콜**은 명확한 계층을 따릅니다: 먼저 헤드 학습, 상위 블록 해동, 그런 다음 레이어별 LR 감쇠로 더 깊은 레이어를 점진적으로 해동.
- **위치 embedding 보간** (이중선형으로 충분)은 처음부터 재학습 없이 사전 학습된 모델을 더 높은 해상도로 전이할 수 있게 합니다.
- **스케일이 핵심 변수**: ViT는 CNN 귀납적 편향에 필적하려면 10배 더 많은 사전 학습 데이터가 필요하지만 그 이상에서는 더 잘 스케일됩니다.
- **레이블 스무싱 + 가중치 감쇠**는 전역 attention의 높은 용량으로 인해 ViT가 보이는 과신을 함께 처리합니다.

---

**이전**: [Vision Transformer (ViT)](./31_Vision_Transformer_ViT.md) | **다음**: [멀티모달 CLIP 스타일 학습](./33_Multimodal_CLIP_Style.md)

> 다음 단원에서는 InfoNCE 대조 손실과 CLIP 스타일 제로샷 분류를 다룹니다.
