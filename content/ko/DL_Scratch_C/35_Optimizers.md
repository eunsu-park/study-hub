# 35. 옵티마이저: SGD, Adam, AdamW, 그래디언트 클리핑

**이전**: [Cross-Entropy 손실](./34_Cross_Entropy_Loss.md) | **다음**: [학습 루프](./36_Training_Loop.md)

---

## 학습 목표

이 단원을 완료하면 다음을 수행할 수 있습니다:

1. 모멘텀과 Nesterov 변형을 포함한 SGD 구현
2. 1차 및 2차 모멘트 추정과 바이어스 보정을 포함한 Adam 구현
3. AdamW가 가중치 감쇠를 gradient 업데이트에서 분리하는 이유 이해
4. 전역 L2 gradient 노름 클리핑 구현
5. 임의의 옵티마이저와 호환되는 함수 포인터 기반 LR 스케줄 인터페이스 설계

---

## 1. 배경: 옵티마이저가 하는 일

옵티마이저는 원시 파라미터 gradient `g = ∂L/∂θ`를 파라미터 업데이트 `Δθ`로 변환합니다. 핵심 루프:

```
각 학습 단계 t:
    gradient g_t 계산
    필요시 g_t 클리핑
    lr_t 스케줄링
    optimizer_step(params, grads, state, lr_t)
```

다른 옵티마이저들은 gradient 이력 (모멘텀), gradient 크기 (적응적 LR), 정규화 (가중치 감쇠) 사용 방식이 다릅니다.

---

## 2. 모멘텀을 포함한 SGD

### 2.1 표준 모멘텀

```
v_t = β * v_{t-1} + g_t          (속도 업데이트)
θ_t = θ_{t-1} - lr * v_t
```

`β=0.9`에서 속도는 과거 gradient의 지수 이동 평균을 누적합니다. 이는 노이즈가 많은 gradient 방향을 평활화하고 일관된 gradient 방향을 따라 수렴을 가속화합니다.

### 2.2 Nesterov 모멘텀

표준 모멘텀은 현재 위치에서 gradient를 봅니다. Nesterov는 미리 내다봅니다:

```
v_t = β * v_{t-1} + g(θ - β * v_{t-1})
θ_t = θ_{t-1} - lr * v_t
```

실용적으로 재매개변수화를 통해 Nesterov 업데이트는:

```
v_t = β * v_{t-1} + g_t
θ_t = θ_{t-1} - lr * (β * v_t + g_t)
```

이는 동일하지만 선행 지점에서 gradient를 재평가하지 않아도 됩니다.

### 2.3 구현

```c
#include <math.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    float *v;       /* 속도 (모멘텀 버퍼) [n_params] */
    float  momentum;
    float  lr;
    int    nesterov;
    int    n_params;
} SGDState;

SGDState *sgd_new(int n_params, float lr, float momentum, int nesterov) {
    SGDState *s = (SGDState *)calloc(1, sizeof(SGDState));
    s->n_params  = n_params;
    s->lr        = lr;
    s->momentum  = momentum;
    s->nesterov  = nesterov;
    s->v         = (float *)calloc((size_t)n_params, sizeof(float));
    return s;
}

void sgd_free(SGDState *s) { free(s->v); free(s); }

/*
 * sgd_update — 하나의 SGD (선택적 모멘텀 포함) 파라미터 업데이트 적용.
 *
 * params : [n_params]  모델 파라미터 (인플레이스 업데이트)
 * grads  : [n_params]  현재 gradient
 */
void sgd_update(float *params, const float *grads, SGDState *s) {
    float beta = s->momentum;
    float lr   = s->lr;
    int   n    = s->n_params;

    if (beta == 0.0f) {
        /* 기본 SGD */
        for (int i = 0; i < n; i++) params[i] -= lr * grads[i];
        return;
    }

    if (s->nesterov) {
        for (int i = 0; i < n; i++) {
            s->v[i] = beta * s->v[i] + grads[i];
            params[i] -= lr * (beta * s->v[i] + grads[i]);
        }
    } else {
        for (int i = 0; i < n; i++) {
            s->v[i] = beta * s->v[i] + grads[i];
            params[i] -= lr * s->v[i];
        }
    }
}
```

---

## 3. Adam

Adam (Kingma & Ba, 2014)은 다음을 사용하여 파라미터별 적응적 학습률을 유지합니다:
- `m1`: 1차 모멘트 (gradient의 평균) — 방향
- `m2`: 2차 모멘트 (제곱 gradient의 평균) — 크기

```
m1_t = β1 * m1_{t-1} + (1 - β1) * g_t
m2_t = β2 * m2_{t-1} + (1 - β2) * g_t²
m1_hat = m1_t / (1 - β1^t)       (바이어스 보정)
m2_hat = m2_t / (1 - β2^t)
θ_t = θ_{t-1} - lr * m1_hat / (sqrt(m2_hat) + ε)
```

바이어스 보정 항 `1 - β^t`는 모멘트 추정의 0 초기화를 보상합니다: 학습 초기에 `m1`과 `m2`는 0을 향해 편향됩니다.

### 3.1 구현

```c
typedef struct {
    float *m1;      /* 1차 모멘트  [n_params] */
    float *m2;      /* 2차 모멘트 [n_params] */
    float  beta1;   /* 기본값 0.9              */
    float  beta2;   /* 기본값 0.999            */
    float  eps;     /* 기본값 1e-8             */
    float  lr;
    int    step;    /* 현재 단계 (바이어스 보정을 위해 1-인덱스) */
    int    n_params;
} AdamState;

AdamState *adam_new(int n_params, float lr,
                    float beta1, float beta2, float eps) {
    AdamState *s = (AdamState *)calloc(1, sizeof(AdamState));
    s->n_params = n_params;
    s->lr       = lr;
    s->beta1    = beta1;
    s->beta2    = beta2;
    s->eps      = eps;
    s->step     = 0;
    s->m1       = (float *)calloc((size_t)n_params, sizeof(float));
    s->m2       = (float *)calloc((size_t)n_params, sizeof(float));
    return s;
}

void adam_free(AdamState *s) { free(s->m1); free(s->m2); free(s); }

/*
 * adam_update — 하나의 Adam 단계 적용.
 *
 * params : [n_params]  인플레이스 업데이트
 * grads  : [n_params]  현재 gradient
 */
void adam_update(float *params, const float *grads, AdamState *s) {
    s->step++;
    float b1   = s->beta1;
    float b2   = s->beta2;
    float eps  = s->eps;
    float lr   = s->lr;
    int   n    = s->n_params;

    /* 바이어스 보정 인수 */
    float bc1 = 1.0f - powf(b1, (float)s->step);
    float bc2 = 1.0f - powf(b2, (float)s->step);
    /* 유효 lr = lr * sqrt(bc2) / bc1 사전 계산 */
    float lr_corr = lr * sqrtf(bc2) / bc1;

    for (int i = 0; i < n; i++) {
        float g   = grads[i];
        s->m1[i]  = b1 * s->m1[i] + (1.0f - b1) * g;
        s->m2[i]  = b2 * s->m2[i] + (1.0f - b2) * g * g;
        params[i] -= lr_corr * s->m1[i] / (sqrtf(s->m2[i]) + eps);
    }
}
```

---

## 4. AdamW: 분리된 가중치 감쇠

표준 Adam은 가중치 감쇠를 L2 gradient 페널티로 적용합니다:

```c
/* L2 페널티 버전 (Adam이 기본적으로 하는 방식): */
grads[i] += weight_decay * params[i];   /* gradient에 추가 */
/* 그런 다음 Adam 업데이트 적용 — 가중치 감쇠가 m1과 m2에 접힘 */
```

문제: Adam은 `1/sqrt(m2)`로 유효 업데이트를 적응적으로 스케일합니다. gradient에 `λ·θ`를 추가하면 가중치 감쇠 페널티도 스케일됩니다 — gradient 이력이 큰 방향에서는 강한 감쇠, 작은 방향에서는 약한 감쇠. 이는 잘못된 것입니다: 가중치 감쇠는 큰 가중치를 균일하게 페널티해야 합니다.

**AdamW 수정**: 가중치 감쇠를 gradient 업데이트와 별도로 파라미터에 직접 적용:

```c
params[i] -= lr * weight_decay * params[i];   /* 먼저 가중치 감쇠 */
/* 그런 다음: 깨끗한 gradient를 사용한 일반 Adam 업데이트 (L2 항 없음) */
```

### 4.1 구현

```c
typedef struct {
    AdamState base;
    float weight_decay;
} AdamWState;

AdamWState *adamw_new(int n_params, float lr,
                      float beta1, float beta2, float eps,
                      float weight_decay) {
    AdamWState *s = (AdamWState *)calloc(1, sizeof(AdamWState));
    /* 내장된 Adam 상태 초기화 */
    s->base.n_params = n_params;
    s->base.lr       = lr;
    s->base.beta1    = beta1;
    s->base.beta2    = beta2;
    s->base.eps      = eps;
    s->base.step     = 0;
    s->base.m1 = (float *)calloc((size_t)n_params, sizeof(float));
    s->base.m2 = (float *)calloc((size_t)n_params, sizeof(float));
    s->weight_decay  = weight_decay;
    return s;
}

void adamw_free(AdamWState *s) {
    free(s->base.m1); free(s->base.m2); free(s);
}

/*
 * adamw_update — AdamW: gradient 업데이트에서 가중치 감쇠 분리.
 *
 * 가중치 감쇠는 Adam gradient 단계 전에 params에 적용됩니다.
 * Embedding과 LayerNorm 파라미터는 weight_decay = 0이어야 합니다.
 */
void adamw_update(float *params, const float *grads, AdamWState *s) {
    AdamState *a = &s->base;
    a->step++;
    float b1   = a->beta1;
    float b2   = a->beta2;
    float eps  = a->eps;
    float lr   = a->lr;
    float wd   = s->weight_decay;
    int   n    = a->n_params;

    float bc1    = 1.0f - powf(b1, (float)a->step);
    float bc2    = 1.0f - powf(b2, (float)a->step);
    float lr_corr = lr * sqrtf(bc2) / bc1;

    for (int i = 0; i < n; i++) {
        /* 1단계: 가중치 감쇠 (분리됨) */
        params[i] *= (1.0f - lr * wd);

        /* 2단계: Adam gradient 업데이트 (gradient에 L2 항 없음) */
        float g   = grads[i];
        a->m1[i]  = b1 * a->m1[i] + (1.0f - b1) * g;
        a->m2[i]  = b2 * a->m2[i] + (1.0f - b2) * g * g;
        params[i] -= lr_corr * a->m1[i] / (sqrtf(a->m2[i]) + eps);
    }
}
```

**LLM 학습의 표준 하이퍼파라미터** (GPT-2):
- `lr = 6e-4`, `beta1 = 0.9`, `beta2 = 0.95` (0.999가 아님 — LLM에서는 낮은 β2)
- `eps = 1e-8`, `weight_decay = 0.1`
- 가중치 감쇠 적용 대상: 선형 가중치, embedding 테이블
- 가중치 감쇠 **미적용** 대상: LayerNorm 파라미터 (γ, β), bias 항

---

## 5. Gradient 클리핑

큰 gradient 노름은 학습 불안정성을 유발합니다 ("gradient 폭발"). 표준 해결책은 **전역 L2 노름**으로 모든 gradient를 클리핑하는 것입니다:

```
||g|| > clip_value인 경우:
    g ← g * (clip_value / ||g||)
```

이는 모든 gradient를 동일한 인수로 스케일 다운하여 상대적인 크기 (파라미터 공간에서의 방향)를 보존합니다.

### 5.1 구현

```c
/*
 * clip_grad_norm — 전역 L2 노름으로 모든 파라미터 gradient 클리핑.
 *
 * grads      : gradient 포인터 배열 [num_params_groups]
 * sizes      : 각 gradient 배열의 요소 수
 * n_groups   : 파라미터 그룹 수
 * max_norm   : 클리핑 임계값 (예: 1.0)
 *
 * 반환: 클리핑 전 전역 gradient 노름.
 */
float clip_grad_norm(float **grads, const int *sizes, int n_groups,
                     float max_norm)
{
    /* 전역 L2 노름 계산 */
    double norm2 = 0.0;
    for (int g = 0; g < n_groups; g++) {
        int n = sizes[g];
        for (int i = 0; i < n; i++) {
            double v = grads[g][i];
            norm2 += v * v;
        }
    }
    float global_norm = (float)sqrt(norm2);

    if (global_norm > max_norm) {
        float scale = max_norm / (global_norm + 1e-6f);
        for (int g = 0; g < n_groups; g++) {
            int n = sizes[g];
            for (int i = 0; i < n; i++) {
                grads[g][i] *= scale;
            }
        }
    }
    return global_norm;
}

/*
 * 편의를 위한 단일 배열 버전:
 */
float clip_grad_norm_flat(float *grads, int n, float max_norm) {
    float *g = grads;
    return clip_grad_norm(&g, &n, 1, max_norm);
}
```

일반적인 값: Transformer에서 `max_norm = 1.0`. Karpathy의 llm.c는 1.0 사용.

### 5.2 Gradient 노름 모니터링

항상 클리핑 전 gradient 노름을 기록하세요. `max_norm`보다 지속적으로 높은 노름은 학습 문제 (나쁜 LR, 너무 큰 배치 크기, backward pass의 버그)를 나타냅니다.

```c
void training_step(float *params, float *grads, int n, AdamWState *opt,
                   float lr, float max_norm, int step, FILE *log)
{
    /* 1. Gradient 클리핑 */
    float gnorm = clip_grad_norm_flat(grads, n, max_norm);

    /* 2. 스케줄에서 LR 설정 */
    opt->base.lr = lr;

    /* 3. 옵티마이저 단계 */
    adamw_update(params, grads, opt);

    /* 4. 로그 */
    if (log) fprintf(log, "step=%d gnorm=%.4f lr=%.6f\n", step, gnorm, lr);
}
```

---

## 6. 함수 포인터로서의 LR 스케줄

LR 스케줄에 함수 포인터를 사용하면 옵티마이저가 스케줄 유형에 무관해집니다:

```c
/* 스케줄 함수 시그니처: (step, total_steps, config) → lr */
typedef float (*LRScheduleFn)(int step, int total_steps, const void *cfg);

/* 코사인 스케줄 설정 */
typedef struct {
    float lr_max;
    float lr_min;
    int   warmup_steps;
} CosineLRCfg;

float cosine_schedule(int step, int total_steps, const void *cfg_) {
    const CosineLRCfg *cfg = (const CosineLRCfg *)cfg_;
    if (step < cfg->warmup_steps) {
        return cfg->lr_max * (float)(step + 1) / (float)cfg->warmup_steps;
    }
    int t = step - cfg->warmup_steps;
    int d = total_steps - cfg->warmup_steps;
    float progress = (d > 0) ? (float)t / (float)d : 1.0f;
    float cosine   = 0.5f * (1.0f + cosf((float)M_PI * progress));
    return cfg->lr_min + (cfg->lr_max - cfg->lr_min) * cosine;
}

/* 상수 스케줄 (파인튜닝 실험용) */
float constant_schedule(int step, int total_steps, const void *cfg_) {
    (void)step; (void)total_steps;
    return *(const float *)cfg_;
}

/* 학습 루프에서의 사용: */
typedef struct {
    AdamWState    *opt;
    LRScheduleFn   schedule;
    const void    *schedule_cfg;
    int            total_steps;
} Trainer;

void trainer_step(Trainer *t, float *params, float *grads, int n, int step) {
    float lr = t->schedule(step, t->total_steps, t->schedule_cfg);
    training_step(params, grads, n, t->opt, lr, 1.0f, step, stdout);
}
```

이 패턴은 깔끔하고, 테스트 가능하며, 옵티마이저를 특정 LR 스케줄과 결합하지 않습니다.

---

## 핵심 요약

- **모멘텀을 포함한 SGD**는 gradient 방향 이력 (속도)을 누적합니다. Nesterov는 gradient 계산 전에 보정을 적용하여 더 빠른 수렴을 제공합니다.
- **Adam**은 2차 모멘트 `m2`를 통해 파라미터별 gradient 크기를 추적합니다. 이력적으로 큰 gradient를 가진 파라미터는 더 작은 유효 LR을 얻고 그 반대도 마찬가지 — 적응적 학습률.
- **AdamW**는 gradient 단계와 별도로 `param *= (1 - lr * wd)`를 적용하여 Adam의 잘못된 가중치 감쇠를 수정합니다. Adam 사용 시 gradient에 가중치 감쇠를 절대 접어 넣지 마세요.
- **β2 = 0.95** (0.999 아님)는 gradient 크기 변화에 더 빠르게 적응하기 때문에 Transformer 학습에 선호됩니다 — LM 학습 중 gradient 분포는 비정상적입니다.
- **전역 L2 노름에 의한 Gradient 클리핑**은 gradient 방향을 보존합니다. 항상 클리핑 전 노름을 기록하세요 — 지속적으로 클리핑된 노름은 학습 불안정 신호입니다.
- **함수 포인터 LR 스케줄**은 옵티마이저 업데이트에서 LR 스케줄을 깔끔하게 분리합니다. 옵티마이저 코드를 수정하지 않고 코사인/선형/상수 스케줄을 교환하세요.
- **LayerNorm, bias에는 가중치 감쇠 없음**: 정규화 파라미터에 대한 가중치 감쇠는 수렴을 방해할 수 있습니다. 가중치 행렬과 embedding 테이블에만 적용하세요.

---

**이전**: [Cross-Entropy 손실](./34_Cross_Entropy_Loss.md) | **다음**: [학습 루프](./36_Training_Loop.md)

> 다음 단원은 사전 토큰화된 바이너리 파일을 위한 mmap 기반 데이터 로더와 함께 전체 LLM 학습 루프를 구축합니다.
