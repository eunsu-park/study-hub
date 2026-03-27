# 36. mmap 데이터 로더를 이용한 완전한 LLM 학습 루프

**이전**: [옵티마이저](./35_Optimizers.md) | **다음**: [Transformer를 통한 역전파](./37_Backprop_Through_Transformer.md)

---

## 학습 목표

이 단원을 완료하면 다음을 수행할 수 있습니다:

1. 사전 토큰화된 바이너리 토큰 파일을 위한 mmap 기반 데이터 로더 구현
2. RAM 오버헤드 없이 (inputs, targets) 쌍을 생성하는 토큰 배치 샘플러 구축
3. 완전한 학습 루프 구조화: zero_grad → forward → loss → backward → clip → step → log
4. 제한된 하드웨어에서 대형 배치 크기를 시뮬레이션하는 gradient 누산 구현
5. 원시 파라미터 배열에서 `fwrite`/`fread`를 사용하여 학습 체크포인트 저장 및 복원

---

## 1. 데이터 로더에 mmap을 사용하는 이유?

GPT-2 사전 학습 데이터셋 (FineWebEdu, 100억 토큰)은 토큰을 `uint16_t` 값으로 저장합니다:

```
100억 토큰 × 2바이트 = 20 GB
```

20 GB를 RAM에 불러오는 것은 비실용적입니다. 표준 해결책은 **메모리 맵 I/O** (`mmap`)입니다. OS가 파일을 프로세스의 가상 주소 공간에 매핑합니다. 페이지는 접근 시 디스크에서 온디맨드로 로드됩니다 — RAM인 것처럼 정확히 `uint16_t *` 포인터에서 읽고 OS가 나머지를 처리합니다.

이점:
- 명시적인 `fread` 루프 없음 — 포인터 산술만 사용
- OS가 페이지 캐시를 통해 캐싱 관리 (프로세스 간 공유)
- 순차적 접근 패턴이 자동으로 프리페칭 트리거
- 4 GB `malloc` 불필요 — 가상 주소 공간이 저렴함

---

## 2. 데이터 파일 형식

사전 토큰화된 데이터는 플랫 바이너리 파일로 저장됩니다:

```
[uint32_t magic]       4바이트  — 0x20240520 (버전 마커)
[uint32_t n_tokens]    4바이트  — 파일의 토큰 수
[uint16_t tokens[n]]   2*n바이트 — BPE 토큰 ID (GPT-2: 값 0..50256)
```

Python에서 파일 생성 (데이터 준비 단계):

```python
import numpy as np
tokens = tokenizer.encode(text)              # 정수 리스트
arr = np.array(tokens, dtype=np.uint16)
header = np.array([0x20240520, len(arr)], dtype=np.uint32)
with open("train.bin", "wb") as f:
    f.write(header.tobytes())
    f.write(arr.tobytes())
```

---

## 3. mmap 데이터 로더

```c
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <assert.h>

#define DATA_MAGIC 0x20240520u

typedef struct {
    /* mmap 상태 */
    int       fd;
    uint8_t  *data;          /* mmap된 바이트                             */
    size_t    file_size;     /* 바이트 단위 전체 파일 크기                 */

    /* 토큰 뷰 (mmap된 영역 내부를 가리킴) */
    uint16_t *tokens;        /* 첫 번째 토큰에 대한 포인터                   */
    size_t    n_tokens;      /* 파일의 토큰 수                 */

    /* 배치 상태 */
    int       B;             /* 배치 크기                               */
    int       T;             /* 시퀀스 길이                          */
    size_t    pos;           /* 토큰 스트림의 현재 위치 (인덱스) */
} DataLoader;

/*
 * dataloader_init — 사전 토큰화된 바이너리 파일을 열고 mmap.
 */
int dataloader_init(DataLoader *dl, const char *path, int B, int T) {
    dl->B = B;
    dl->T = T;
    dl->pos = 0;

    dl->fd = open(path, O_RDONLY);
    if (dl->fd < 0) { perror("open"); return -1; }

    struct stat st;
    if (fstat(dl->fd, &st) < 0) { perror("fstat"); close(dl->fd); return -1; }
    dl->file_size = (size_t)st.st_size;

    dl->data = (uint8_t *)mmap(NULL, dl->file_size,
                               PROT_READ, MAP_PRIVATE, dl->fd, 0);
    if (dl->data == MAP_FAILED) { perror("mmap"); close(dl->fd); return -1; }

    /* 헤더 검증 */
    uint32_t *header = (uint32_t *)dl->data;
    assert(header[0] == DATA_MAGIC && "bad magic");
    uint32_t n_tokens_u32 = header[1];

    dl->tokens   = (uint16_t *)(dl->data + 2 * sizeof(uint32_t));
    dl->n_tokens = (size_t)n_tokens_u32;

    size_t expected = 2 * sizeof(uint32_t) + dl->n_tokens * sizeof(uint16_t);
    assert(dl->file_size >= expected && "file truncated");

    /* OS 힌트: 순차적 접근 패턴 */
    madvise(dl->data, dl->file_size, MADV_SEQUENTIAL);

    return 0;
}

void dataloader_free(DataLoader *dl) {
    munmap(dl->data, dl->file_size);
    close(dl->fd);
}

/*
 * dataloader_next_batch — 하나의 배치에 대한 inputs와 targets 채우기.
 *
 * inputs  : [B, T]  int32 (토큰 ID로 채워짐)
 * targets : [B, T]  int32 (targets = inputs를 오른쪽으로 1 이동)
 *
 * 로더는 토큰 스트림을 순환 링 버퍼로 취급:
 * 끝에 도달하면 위치 0으로 되돌아감.
 */
void dataloader_next_batch(DataLoader *dl, int *inputs, int *targets) {
    int B = dl->B;
    int T = dl->T;
    size_t n = dl->n_tokens;

    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            size_t idx = (dl->pos + (size_t)(b * T + t)) % n;
            inputs[b * T + t] = (int)dl->tokens[idx];
        }
        for (int t = 0; t < T; t++) {
            size_t idx = (dl->pos + (size_t)(b * T + t) + 1) % n;
            targets[b * T + t] = (int)dl->tokens[idx];
        }
    }

    /* B*T 토큰만큼 위치 전진 */
    dl->pos = (dl->pos + (size_t)(B * T)) % n;
}

/*
 * dataloader_reset — 위치 0에서 재시작 (새 에포크용).
 */
void dataloader_reset(DataLoader *dl) { dl->pos = 0; }
```

---

## 4. 완전한 학습 루프

```c
#include <math.h>
#include <time.h>

/*
 * 학습 설정 구조체 — 모든 하이퍼파라미터를 한 곳에.
 */
typedef struct {
    /* 모델 */
    int vocab_size;   /* 예: 50257 (GPT-2)          */
    int n_layer;      /* 예: 12                      */
    int n_head;       /* 예: 12                      */
    int n_embd;       /* 예: 768                     */
    int block_size;   /* 컨텍스트 길이 T, 예: 1024  */

    /* 데이터 */
    const char *train_path;
    const char *val_path;
    int B;            /* 배치 크기, 예: 16          */

    /* 최적화 */
    float lr_max;         /* 최고 LR, 예: 6e-4        */
    float lr_min;         /* 최종 LR, 예: 6e-5       */
    float weight_decay;   /* 예: 0.1                  */
    float grad_clip;      /* 예: 1.0                  */
    float beta1;          /* 예: 0.9                  */
    float beta2;          /* 예: 0.95                 */
    int   warmup_steps;   /* 예: 715 (100억 토큰의 경우) */
    int   max_steps;      /* 전체 학습 단계       */

    /* Gradient 누산 */
    int   grad_accum_steps; /* effective_batch = B * grad_accum_steps */

    /* 로깅 / 체크포인팅 */
    int   log_interval;    /* N 단계마다 손실 출력   */
    int   val_interval;    /* N 단계마다 평가      */
    int   save_interval;   /* N 단계마다 체크포인트    */
    const char *ckpt_path;
} TrainConfig;

/*
 * zero_grad — 누산 전에 모든 gradient를 0으로 설정.
 */
void zero_grad(float *grads, int n) {
    memset(grads, 0, (size_t)n * sizeof(float));
}

/*
 * train — 완전한 학습 루프.
 *
 * model_forward  : (tokens[B,T], logits[B,T,V]) → void
 * model_backward : (dlogits[B,T,V]) → grads[]를 인플레이스로 업데이트
 * params, grads  : 평탄화된 파라미터 및 gradient 배열
 * n_params       : 전체 파라미터 수
 * cfg            : 학습 설정
 */
void train(void (*model_forward)(const int *, float *, int, int),
           void (*model_backward)(const float *, float *, int, int),
           float *params, float *grads, int n_params,
           const TrainConfig *cfg)
{
    DataLoader train_loader, val_loader;
    dataloader_init(&train_loader, cfg->train_path, cfg->B, cfg->block_size);
    dataloader_init(&val_loader,   cfg->val_path,   cfg->B, cfg->block_size);

    /* 버퍼 할당 */
    int BT = cfg->B * cfg->block_size;
    int BTV = BT * cfg->vocab_size;
    int *inputs   = (int   *)malloc((size_t)BT  * sizeof(int));
    int *targets  = (int   *)malloc((size_t)BT  * sizeof(int));
    float *logits  = (float *)malloc((size_t)BTV * sizeof(float));
    float *dlogits = (float *)malloc((size_t)BTV * sizeof(float));

    /* 옵티마이저 초기화 */
    AdamWState *opt = adamw_new(n_params, cfg->lr_max,
                                cfg->beta1, cfg->beta2, 1e-8f,
                                cfg->weight_decay);

    /* LR 스케줄 */
    CosineLRCfg lr_cfg = {
        .lr_max       = cfg->lr_max,
        .lr_min       = cfg->lr_min,
        .warmup_steps = cfg->warmup_steps
    };

    /* 손실 로깅 */
    FILE *loss_csv = fopen("loss.csv", "w");
    if (loss_csv) fprintf(loss_csv, "step,train_loss,val_loss,lr,gnorm\n");

    for (int step = 0; step < cfg->max_steps; step++) {
        double t0 = (double)clock() / CLOCKS_PER_SEC;

        /* --- 1. Gradient 0으로 초기화 --- */
        zero_grad(grads, n_params);
        float accum_loss = 0.0f;

        /* --- 2. 마이크로-배치에 걸친 gradient 누산 --- */
        for (int micro = 0; micro < cfg->grad_accum_steps; micro++) {
            dataloader_next_batch(&train_loader, inputs, targets);

            /* Forward pass */
            model_forward(inputs, logits, cfg->B, cfg->block_size);

            /* Fused CE 손실 + logit에 대한 backward */
            float loss = fused_ce_forward_backward(
                logits, targets, dlogits,
                BT, cfg->vocab_size
            );
            accum_loss += loss;

            /* 마이크로-배치 평균을 위해 1/grad_accum_steps로 dlogits 스케일 */
            float scale = 1.0f / (float)cfg->grad_accum_steps;
            for (int i = 0; i < BTV; i++) dlogits[i] *= scale;

            /* Backward pass — grads[]에 누산 */
            model_backward(dlogits, grads, cfg->B, cfg->block_size);
        }
        accum_loss /= (float)cfg->grad_accum_steps;

        /* --- 3. Gradient 클리핑 --- */
        float gnorm = clip_grad_norm_flat(grads, n_params, cfg->grad_clip);

        /* --- 4. LR 및 옵티마이저 단계 업데이트 --- */
        float lr = cosine_schedule(step, cfg->max_steps, &lr_cfg);
        opt->base.lr = lr;
        adamw_update(params, grads, opt);

        double t1 = (double)clock() / CLOCKS_PER_SEC;

        /* --- 5. 로깅 --- */
        if (step % cfg->log_interval == 0) {
            float tok_per_sec = (float)(cfg->B * cfg->block_size
                                        * cfg->grad_accum_steps)
                                / (float)(t1 - t0 + 1e-9);
            printf("단계 %5d | 손실 %.4f | lr %.2e | gnorm %.3f | %.0f tok/s\n",
                   step, accum_loss, lr, gnorm, tok_per_sec);
        }

        /* --- 6. 검증 --- */
        if (step % cfg->val_interval == 0) {
            float val_loss = 0.0f;
            int val_steps  = 20;
            for (int v = 0; v < val_steps; v++) {
                dataloader_next_batch(&val_loader, inputs, targets);
                model_forward(inputs, logits, cfg->B, cfg->block_size);
                /* 손실만 계산 — backward 없음 */
                val_loss += cross_entropy_forward_naive(
                    logits, targets, dlogits, BT, cfg->vocab_size
                );
            }
            val_loss /= (float)val_steps;
            printf("val_loss=%.4f (ppl=%.2f)\n",
                   val_loss, expf(val_loss));
            if (loss_csv)
                fprintf(loss_csv, "%d,%.4f,%.4f,%.6f,%.4f\n",
                        step, accum_loss, val_loss, lr, gnorm);
        }

        /* --- 7. 체크포인트 --- */
        if (cfg->save_interval > 0 && step % cfg->save_interval == 0) {
            char path[256];
            snprintf(path, sizeof(path), "%s_%05d.bin", cfg->ckpt_path, step);
            save_checkpoint(params, n_params, opt, step, path);
        }
    }

    if (loss_csv) fclose(loss_csv);
    dataloader_free(&train_loader);
    dataloader_free(&val_loader);
    adamw_free(opt);
    free(inputs); free(targets); free(logits); free(dlogits);
}
```

---

## 5. Gradient 누산

유효 배치 크기 `B_eff = B × grad_accum_steps`가 필요하지만 물리적 메모리가 `B`만 맞을 때:

```
각 단계:
  zero_grad()
  micro in 0..grad_accum_steps-1:
      batch = next_batch()
      logits = forward(batch)
      loss = CE(logits, targets)
      dlogits = CE_backward(loss)
      dlogits *= 1/grad_accum_steps        ← backward 전에 스케일
      grads += backward(dlogits)           ← 누산
  clip_grad_norm(grads)
  optimizer_step()
```

핵심: **backward pass 전에 `grad_accum_steps`로 dlogits를 나누세요** — 누산된 gradient가 전체 유효 배치에 대한 단일 forward/backward와 동일하도록.

예: GPT-2 124M 학습
- 물리적 배치: B=16, T=1024 → 16K 토큰/단계
- Gradient 누산: 8 단계 → 128K 토큰 유효 배치
- 하나의 마이크로-배치에 대한 GPU 메모리: `16 × 1024 × 50257 × 4 ≈ 3.3 GB` (logit만)

---

## 6. 체크포인트 저장 및 불러오기

```c
typedef struct {
    int   step;
    float lr;
    int   adam_step;
} CheckpointHeader;

/*
 * save_checkpoint — params + 옵티마이저 상태를 바이너리 파일에 씀.
 *
 * 형식:
 *   [CheckpointHeader]
 *   [float params[n_params]]
 *   [float m1[n_params]]
 *   [float m2[n_params]]
 */
int save_checkpoint(const float *params, int n_params,
                    const AdamWState *opt, int step,
                    const char *path)
{
    FILE *f = fopen(path, "wb");
    if (!f) { perror("save_checkpoint"); return -1; }

    CheckpointHeader hdr = {
        .step      = step,
        .lr        = opt->base.lr,
        .adam_step = opt->base.step
    };
    fwrite(&hdr,    sizeof(hdr),              1,       f);
    fwrite(params,  sizeof(float), (size_t)n_params,   f);
    fwrite(opt->base.m1, sizeof(float), (size_t)n_params, f);
    fwrite(opt->base.m2, sizeof(float), (size_t)n_params, f);

    fclose(f);
    printf("체크포인트 저장됨: %s (단계 %d)\n", path, step);
    return 0;
}

int load_checkpoint(float *params, int n_params,
                    AdamWState *opt, int *step_out,
                    const char *path)
{
    FILE *f = fopen(path, "rb");
    if (!f) { perror("load_checkpoint"); return -1; }

    CheckpointHeader hdr;
    fread(&hdr,    sizeof(hdr),              1,         f);
    fread(params,  sizeof(float), (size_t)n_params,     f);
    fread(opt->base.m1, sizeof(float), (size_t)n_params, f);
    fread(opt->base.m2, sizeof(float), (size_t)n_params, f);

    opt->base.step = hdr.adam_step;
    opt->base.lr   = hdr.lr;
    *step_out      = hdr.step;

    fclose(f);
    printf("체크포인트 불러옴: %s (단계 %d)\n", path, hdr.step);
    return 0;
}
```

---

## 7. Shakespeare 문자 수준 예시

독립 실행형 학습 예시로 Shakespeare의 문자 수준 GPT는 ~20만 토큰이며 CPU에서 분 단위로 학습됩니다:

```c
/*
 * 문자 수준 데이터: 토큰은 원시 바이트 (0-255), vocab_size=256.
 * 파일 형식: 원시 바이트 (문자 수준에서는 헤더 불필요).
 *
 * 일반적인 결과:
 *   단계    0 | 손실 5.5452 (≈ log(256))  — 임의 초기화
 *   단계  100 | 손실 2.1  — 모델이 문자 빈도 학습
 *   단계  500 | 손실 1.7  — 모델이 일반 n-그램 학습
 *   단계 2000 | 손실 1.4  — Shakespeare 스타일 출력
 */

TrainConfig shakespeare_cfg = {
    .vocab_size       = 256,
    .n_layer          = 6,
    .n_head           = 6,
    .n_embd           = 384,
    .block_size       = 256,

    .train_path       = "shakespeare_train.bin",
    .val_path         = "shakespeare_val.bin",
    .B                = 32,

    .lr_max           = 1e-3f,
    .lr_min           = 1e-4f,
    .weight_decay     = 0.1f,
    .grad_clip        = 1.0f,
    .beta1            = 0.9f,
    .beta2            = 0.95f,
    .warmup_steps     = 100,
    .max_steps        = 5000,
    .grad_accum_steps = 1,    /* 소형 모델에서는 누산 불필요 */

    .log_interval     = 10,
    .val_interval     = 250,
    .save_interval    = 1000,
    .ckpt_path        = "shakespeare_ckpt"
};
```

---

## 핵심 요약

- **mmap**은 파일을 가상 주소 공간에 매핑: 토큰 배열은 일반 포인터로 접근하고 OS가 온디맨드 페이징을 처리합니다. 전체 데이터셋을 `malloc`할 필요 없음.
- **순환 위치 추적** (`pos = (pos + B*T) % n_tokens`)은 특별한 파일 끝 처리 없이 무한한 데이터를 제공합니다.
- **Gradient 누산**은 옵티마이저 단계당 더 많은 forward/backward pass 비용으로 유효 배치 크기를 곱합니다. 누산 후가 아닌 전에 `dlogits`를 스케일하세요.
- **학습 루프 구조는 고정됨**: zero_grad → (마이크로-배치 루프) → clip → lr_schedule → optimizer_step → log. 이 순서를 벗어나면 미묘한 버그가 발생합니다.
- **체크포인트 형식**: 작은 헤더 (step, LR, 옵티마이저 step 수)에 이어 params, m1, m2에 대한 원시 `float` 배열 씀. 직렬화 오버헤드 없는 단순한 `fwrite`/`fread`.
- **CSV로의 손실 로깅**은 학습 곡선의 사후 분석 및 시각화를 가능하게 합니다. 항상 기록하세요: step, train_loss, val_loss, lr, grad_norm.
- **문자 수준 Shakespeare**는 정규적인 빠른 검증입니다: 384 embedding 차원의 6-레이어 GPT가 CPU에서 5분 미만에 perplexity ~4.0으로 학습되어 전체 파이프라인이 엔드-투-엔드로 작동함을 확인합니다.

---

**이전**: [옵티마이저](./35_Optimizers.md) | **다음**: [Transformer를 통한 역전파](./37_Backprop_Through_Transformer.md)

> 다음 단원은 Transformer를 통한 완전한 backward pass를 유도합니다: attention backward (dV, dK, dQ), layernorm backward, 2-레이어 모델에서의 수치적 gradient 검사.
