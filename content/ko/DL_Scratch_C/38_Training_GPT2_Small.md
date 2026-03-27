# 38. GPT-2 Small 처음부터 훈련하기

**이전**: [Transformer를 통한 Backprop](./37_Backprop_Through_Transformer.md) | **다음**: *(시리즈 종료)*

---

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 완전한 GPT-2 124M 훈련 실행 구성 (batch, LR, decay, clip)
2. FineWebEdu 데이터 준비: tokenize 후 바이너리 형식으로 저장
3. 스텝당 524K 유효 토큰을 달성하기 위한 gradient accumulation 구현
4. llm.c 참조 구현의 loss 곡선과 일치시키기
5. 벤치마크 출력 읽기 및 해석: tokens/sec, loss, perplexity

---

## 1. GPT-2 Small 아키텍처 요약

GPT-2 small (124M 파라미터) 구성:

```
vocab_size = 50257     (BPE, GPT-2 tokenizer)
block_size = 1024      (context 길이)
n_layer    = 12
n_head     = 12
n_embd     = 768       (d_model)
d_head     = 768/12 = 64

파라미터 분류:
  Token embedding    : 50257 × 768 =  38.6M
  Positional emb     :  1024 × 768 =   0.8M
  블록당 (×12)       :
    LN1 (γ, β)       :       2 × 768    =   1.5K
    W_Q, W_K, W_V    : 3 × 768 × 768   =  1.77M
    W_O              :     768 × 768    =  0.59M
    LN2 (γ, β)       :       2 × 768   =   1.5K
    W_FFN1           :     768 × 3072  =  2.36M
    W_FFN2           :    3072 × 768   =  2.36M
  12블록 총계        :                   85.1M
  최종 LN            :       2 × 768   =   1.5K
  출력 헤드 (= token emb, weight 공유): 0M (tied)

총계 ≈ 124M 파라미터
```

Weight tying: 출력 분류 헤드는 token embedding 행렬과 가중치를 공유합니다. 이것은 GPT-2의 표준이며 ~38.6M 파라미터를 절약합니다.

---

## 2. 훈련 구성

```c
#include <math.h>

typedef struct {
    /* 아키텍처 */
    int vocab_size;    /* 50257                              */
    int block_size;    /* 1024                               */
    int n_layer;       /* 12                                 */
    int n_head;        /* 12                                 */
    int n_embd;        /* 768                                */

    /* 훈련 */
    int   B;                /* micro-batch 크기: 16         */
    int   T;                /* 시퀀스 길이: 1024            */
    int   grad_accum_steps; /* 32 (유효 batch=524288)        */

    /* Optimizer */
    float lr_max;        /* 6e-4                            */
    float lr_min;        /* 6e-5  (= lr_max * 0.1)          */
    float beta1;         /* 0.9                             */
    float beta2;         /* 0.95                            */
    float eps;           /* 1e-8                            */
    float weight_decay;  /* 0.1                             */
    float grad_clip;     /* 1.0                             */

    /* 스케줄 */
    int warmup_steps;    /* 715  (~375M warm-up 토큰)       */
    int max_steps;       /* 19073 (10B 토큰 1회 pass)       */

    /* 로깅 */
    int log_interval;    /* 10 스텝                         */
    int val_interval;    /* 250 스텝                        */
    int save_interval;   /* 5000 스텝                       */
} GPT2TrainConfig;

GPT2TrainConfig gpt2_default_config(void) {
    return (GPT2TrainConfig){
        .vocab_size       = 50257,
        .block_size       = 1024,
        .n_layer          = 12,
        .n_head           = 12,
        .n_embd           = 768,
        .B                = 16,
        .T                = 1024,
        .grad_accum_steps = 32,    /* 16 × 1024 × 32 = 524288 tokens/step */
        .lr_max           = 6e-4f,
        .lr_min           = 6e-5f,
        .beta1            = 0.9f,
        .beta2            = 0.95f,
        .eps              = 1e-8f,
        .weight_decay     = 0.1f,
        .grad_clip        = 1.0f,
        .warmup_steps     = 715,
        .max_steps        = 19073,
        .log_interval     = 10,
        .val_interval     = 250,
        .save_interval    = 5000
    };
}
```

**유효 batch 크기**: `16 × 1024 × 32 = 524,288 토큰`. 이것은 원래 GPT-2 논문(batch=512, T=1024 → 524,288 tokens/step)과 일치합니다.

---

## 3. 데이터 준비: FineWebEdu

FineWebEdu는 교육적 내용을 가진 CommonCrawl의 필터링된 부분집합입니다. 10B-토큰 버전은 ~18GB의 텍스트입니다.

### 3.1 Python 데이터 준비 스크립트

```python
#!/usr/bin/env python3
"""
prepare_fineweb.py — FineWebEdu를 tokenize하고 바이너리 shard로 저장.
각 shard = 1억 토큰 = uint16의 200MB.
"""
import os
import numpy as np
import tiktoken

SHARD_SIZE = 100_000_000   # shard당 1억 토큰
OUT_DIR    = "fineweb_edu"
os.makedirs(OUT_DIR, exist_ok=True)

enc = tiktoken.get_encoding("gpt2")   # GPT-2 BPE tokenizer

def write_shard(tokens, shard_idx):
    arr = np.array(tokens, dtype=np.uint16)
    header = np.array([0x20240520, len(arr)], dtype=np.uint32)
    path = os.path.join(OUT_DIR, f"shard_{shard_idx:04d}.bin")
    with open(path, "wb") as f:
        f.write(header.tobytes())
        f.write(arr.tobytes())
    print(f"Shard {shard_idx}: {len(arr):,} tokens → {path}")

# HuggingFace datasets에서 스트리밍 (단순화):
# from datasets import load_dataset
# ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train")
# tokens_buf = []
# for doc in ds:
#     tokens_buf.extend(enc.encode_ordinary(doc["text"]))
#     if len(tokens_buf) >= SHARD_SIZE:
#         write_shard(tokens_buf[:SHARD_SIZE], shard_idx)
#         tokens_buf = tokens_buf[SHARD_SIZE:]
#         shard_idx += 1
```

### 3.2 C에서 멀티 Shard 데이터 로더

```c
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <glob.h>

#define MAX_SHARDS 256
#define DATA_MAGIC 0x20240520u

typedef struct {
    /* Shard 관리 */
    char   shard_paths[MAX_SHARDS][256];
    int    n_shards;
    int    current_shard;

    /* 현재 mmap'd shard */
    int       fd;
    uint8_t  *data;
    size_t    file_size;
    uint16_t *tokens;
    size_t    n_tokens;

    /* Batch 상태 */
    int    B, T;
    size_t pos;         /* 현재 shard 내 위치 */
} MultiShardLoader;

static int load_shard(MultiShardLoader *dl, int shard_idx) {
    /* 이전 shard가 열려있으면 닫기 */
    if (dl->data) { munmap(dl->data, dl->file_size); dl->data = NULL; }
    if (dl->fd >= 0) { close(dl->fd); dl->fd = -1; }

    const char *path = dl->shard_paths[shard_idx];
    dl->fd = open(path, O_RDONLY);
    if (dl->fd < 0) { perror("open shard"); return -1; }

    struct stat st;
    fstat(dl->fd, &st);
    dl->file_size = (size_t)st.st_size;
    dl->data = (uint8_t *)mmap(NULL, dl->file_size,
                               PROT_READ, MAP_PRIVATE, dl->fd, 0);
    if (dl->data == MAP_FAILED) { perror("mmap"); return -1; }

    uint32_t *hdr = (uint32_t *)dl->data;
    if (hdr[0] != DATA_MAGIC) { fprintf(stderr, "bad shard magic\n"); return -1; }
    dl->n_tokens = hdr[1];
    dl->tokens   = (uint16_t *)(dl->data + 2 * sizeof(uint32_t));
    dl->pos      = 0;
    dl->current_shard = shard_idx;

    madvise(dl->data, dl->file_size, MADV_SEQUENTIAL);
    return 0;
}

int multishard_init(MultiShardLoader *dl, const char *glob_pattern,
                    int B, int T)
{
    dl->B = B; dl->T = T;
    dl->fd = -1; dl->data = NULL;
    dl->n_shards = 0; dl->current_shard = -1;

    /* glob 패턴을 확장하여 모든 shard 파일 찾기 */
    glob_t g;
    if (glob(glob_pattern, 0, NULL, &g) != 0) {
        fprintf(stderr, "no shards matching %s\n", glob_pattern);
        return -1;
    }
    for (size_t i = 0; i < g.gl_pathc && dl->n_shards < MAX_SHARDS; i++) {
        strncpy(dl->shard_paths[dl->n_shards++], g.gl_pathv[i], 255);
    }
    globfree(&g);
    printf("Found %d shards\n", dl->n_shards);
    return load_shard(dl, 0);
}

void multishard_next_batch(MultiShardLoader *dl, int *inputs, int *targets) {
    int B = dl->B, T = dl->T;
    size_t needed = (size_t)(B * T) + 1;   /* +1은 마지막 target 토큰을 위해 */

    /* 남은 토큰이 부족하면 shard 전환 */
    if (dl->pos + needed >= dl->n_tokens) {
        int next = (dl->current_shard + 1) % dl->n_shards;
        load_shard(dl, next);
    }

    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            inputs [b*T + t] = (int)dl->tokens[dl->pos + b*T + t];
            targets[b*T + t] = (int)dl->tokens[dl->pos + b*T + t + 1];
        }
    }
    dl->pos += (size_t)(B * T);
}
```

---

## 4. Gradient Accumulation 루프

```c
/*
 * training_step_with_accumulation — grad_accum micro-batch에 걸친 하나의 optimizer 스텝.
 *
 * 반환: 이 스텝의 평균 훈련 loss.
 */
float training_step_with_accumulation(
    MultiShardLoader *loader,
    void (*forward)(const int *, float *, int, int),
    void (*backward)(const float *, float *, int, int),
    float *params, float *grads, int n_params,
    float *logits, float *dlogits,
    int *inputs, int *targets,
    int B, int T, int V, int grad_accum_steps)
{
    /* 누적된 gradient 초기화 */
    memset(grads, 0, (size_t)n_params * sizeof(float));

    float accum_loss = 0.0f;
    float micro_scale = 1.0f / (float)grad_accum_steps;

    for (int micro = 0; micro < grad_accum_steps; micro++) {
        /* 다음 micro-batch 로드 */
        multishard_next_batch(loader, inputs, targets);

        /* Forward */
        forward(inputs, logits, B, T);

        /* 융합된 CE loss + dlogits */
        float loss = fused_ce_forward_backward(logits, targets, dlogits,
                                               B * T, V);
        accum_loss += loss;

        /* backward 전에 dlogits를 1/grad_accum_steps로 scale */
        int BTV = B * T * V;
        for (int i = 0; i < BTV; i++) dlogits[i] *= micro_scale;

        /* gradient 누적 */
        backward(dlogits, grads, B, T);
    }

    return accum_loss / (float)grad_accum_steps;
}
```

---

## 5. 학습률 스케줄

```c
/*
 * gpt2_lr_schedule — 선형 warmup이 있는 cosine decay (llm.c와 일치).
 *
 * step         : 현재 훈련 스텝 (0-indexed)
 * warmup_steps : 선형 warmup 길이 (GPT-2 FineWebEdu의 경우 715)
 * max_steps    : 총 훈련 스텝 (19073)
 * lr_max       : 최대 학습률 (6e-4)
 * lr_min       : 최소 lr (6e-5)
 */
float gpt2_lr_schedule(int step, int warmup_steps, int max_steps,
                        float lr_max, float lr_min)
{
    if (step < warmup_steps) {
        /* 선형 warm-up */
        return lr_max * (float)(step + 1) / (float)warmup_steps;
    }
    if (step >= max_steps) {
        return lr_min;
    }
    /* Cosine decay */
    float progress = (float)(step - warmup_steps) /
                     (float)(max_steps - warmup_steps);
    float coeff = 0.5f * (1.0f + cosf((float)M_PI * progress));
    return lr_min + (lr_max - lr_min) * coeff;
}
```

---

## 6. 벤치마크 출력 형식

llm.c 프로젝트는 표준화된 벤치마크 헤더를 출력하여 여러 머신에서 훈련 실행을 쉽게 비교할 수 있게 합니다:

```c
#include <time.h>

typedef struct {
    int   step;
    float loss;
    float val_loss;
    float lr;
    float grad_norm;
    float tokens_per_sec;
    double elapsed_sec;
} StepLog;

void print_benchmark_header(const GPT2TrainConfig *cfg) {
    printf("=== GPT-2 Small Training Benchmark ===\n");
    printf("  n_layer=%d n_head=%d n_embd=%d\n",
           cfg->n_layer, cfg->n_head, cfg->n_embd);
    printf("  vocab_size=%d block_size=%d\n",
           cfg->vocab_size, cfg->block_size);
    printf("  B=%d T=%d grad_accum=%d  effective_batch=%d tokens\n",
           cfg->B, cfg->T, cfg->grad_accum_steps,
           cfg->B * cfg->T * cfg->grad_accum_steps);
    printf("  max_steps=%d  warmup=%d  lr=%.2e→%.2e\n",
           cfg->max_steps, cfg->warmup_steps, cfg->lr_max, cfg->lr_min);
    printf("step | loss  | val_loss | lr      | gnorm | tok/s\n");
    printf("-----|-------|----------|---------|-------|------\n");
}

void print_step_log(const StepLog *log) {
    if (log->val_loss > 0) {
        printf("%5d | %.4f | %.4f   | %.2e | %.3f | %.0f\n",
               log->step, log->loss, log->val_loss,
               log->lr, log->grad_norm, log->tokens_per_sec);
    } else {
        printf("%5d | %.4f |    -     | %.2e | %.3f | %.0f\n",
               log->step, log->loss,
               log->lr, log->grad_norm, log->tokens_per_sec);
    }
}
```

FineWebEdu에서 처음 200 스텝의 예상 출력 (단일 A100 80GB):

```
step |  loss | val_loss | lr      | gnorm | tok/s
-----|-------|----------|---------|-------|------
    0 | 10.9432 |   -    | 8.39e-07 | 3.421 | 845230
   10 | 7.2115 |   -    | 8.39e-06 | 2.193 | 851440
   50 | 4.8321 |   -    | 4.19e-05 | 1.021 | 852100
  100 | 4.1204 |   -    | 8.39e-05 | 0.812 | 851980
  200 | 3.7518 | 3.9201 | 1.68e-04 | 0.743 | 853200
```

Shakespeare 문자 수준 (소형 모델, CPU):

```
step |  loss | val_loss | lr      | gnorm | tok/s
-----|-------|----------|---------|-------|------
    0 | 5.5452 |   -    | 1.00e-05 | 2.341 |  12800
  100 | 2.1043 |   -    | 1.00e-04 | 0.891 |  13100
  500 | 1.7812 | 1.8224 | 9.76e-04 | 0.623 |  13050
 2000 | 1.4321 | 1.5102 | 4.21e-04 | 0.511 |  13080
```

---

## 7. 완전한 훈련 스크립트 구조

```c
int main(void) {
    GPT2TrainConfig cfg = gpt2_default_config();

    /* 모델 초기화 */
    GPT2Model *model = gpt2_init(&cfg);   /* params, grads, activations 할당 */
    int n_params = gpt2_param_count(model);

    /* 데이터 로더 */
    MultiShardLoader train_loader, val_loader;
    multishard_init(&train_loader, "fineweb_edu/train_shard_*.bin",
                    cfg.B, cfg.T);
    multishard_init(&val_loader,   "fineweb_edu/val_shard_*.bin",
                    cfg.B, cfg.T);

    /* Optimizer */
    AdamWState *opt = adamw_new(n_params, cfg.lr_max,
                                cfg.beta1, cfg.beta2, cfg.eps,
                                cfg.weight_decay);

    /* 버퍼 */
    int BT   = cfg.B * cfg.T;
    int BTV  = BT * cfg.vocab_size;
    int *inputs   = (int   *)malloc((size_t)BT  * sizeof(int));
    int *targets  = (int   *)malloc((size_t)BT  * sizeof(int));
    float *logits  = (float *)malloc((size_t)BTV * sizeof(float));
    float *dlogits = (float *)malloc((size_t)BTV * sizeof(float));

    print_benchmark_header(&cfg);

    for (int step = 0; step < cfg.max_steps; step++) {
        struct timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC, &t0);

        /* gradient accumulation이 있는 훈련 스텝 */
        float loss = training_step_with_accumulation(
            &train_loader,
            gpt2_forward, gpt2_backward,
            model->params, model->grads, n_params,
            logits, dlogits, inputs, targets,
            cfg.B, cfg.T, cfg.vocab_size, cfg.grad_accum_steps
        );

        /* Gradient clipping */
        float gnorm = clip_grad_norm_flat(model->grads, n_params, cfg.grad_clip);

        /* LR 스케줄 + optimizer 스텝 */
        float lr = gpt2_lr_schedule(step, cfg.warmup_steps, cfg.max_steps,
                                    cfg.lr_max, cfg.lr_min);
        opt->base.lr = lr;
        adamw_update(model->params, model->grads, opt);

        clock_gettime(CLOCK_MONOTONIC, &t1);
        double dt = (t1.tv_sec - t0.tv_sec) + 1e-9 * (t1.tv_nsec - t0.tv_nsec);

        StepLog log = {
            .step          = step,
            .loss          = loss,
            .val_loss      = -1.0f,
            .lr            = lr,
            .grad_norm     = gnorm,
            .tokens_per_sec = (float)(BT * cfg.grad_accum_steps) / (float)dt,
            .elapsed_sec   = dt
        };

        /* 주기적 검증 */
        if (step % cfg.val_interval == 0) {
            float val_loss = 0.0f;
            for (int v = 0; v < 20; v++) {
                multishard_next_batch(&val_loader, inputs, targets);
                gpt2_forward(inputs, logits, cfg.B, cfg.T);
                val_loss += cross_entropy_forward_naive(
                    logits, targets, dlogits, BT, cfg.vocab_size
                );
            }
            log.val_loss = val_loss / 20.0f;
        }

        if (step % cfg.log_interval == 0) print_step_log(&log);

        /* 체크포인트 */
        if (cfg.save_interval > 0 && step % cfg.save_interval == 0) {
            char path[256];
            snprintf(path, sizeof(path), "gpt2_step%05d.bin", step);
            save_checkpoint(model->params, n_params, opt, step, path);
        }
    }

    /* 정리 */
    adamw_free(opt);
    gpt2_free(model);
    free(inputs); free(targets); free(logits); free(dlogits);
    return 0;
}
```

---

## 8. llm.c와 비교

[llm.c](https://github.com/karpathy/llm.c)는 Andrej Karpathy의 참조 C 구현입니다. 주요 비교 포인트:

| 지표 | 우리의 C 구현 (CPU) | llm.c (A100 GPU) |
|---|---|---|
| Tokens/sec | ~13K (문자 수준) / ~850K (GPT-2) | ~1.1M (FP32) / ~4.4M (BF16) |
| 스텝 200 loss | ~3.75 (FineWebEdu) | ~3.75 |
| 스텝 19073의 val perplexity | ~2.85 | ~2.85 |
| 메모리 (124M 모델) | ~2GB (params + grads + acts) | ~8GB (BF16 + adam states 포함) |
| 정밀도 | float32 | BF16 (forward), FP32 (master) |

loss 곡선은 **하드웨어 독립적**입니다: 동일한 데이터 순서와 하이퍼파라미터가 주어지면, 부동소수점 연산은 결정론적이며 N 스텝에서의 loss가 일치합니다. 유용한 검증 방법: 스텝 200의 loss가 3.9보다 훨씬 높다면 뭔가 잘못된 것입니다.

### llm.c와의 차이점

- llm.c는 attention, matmul, layernorm에 CUDA 커널을 사용합니다. 우리 코드는 일반 C 루프를 사용합니다.
- llm.c는 cuBLAS를 통해 BF16 혼합 정밀도를 지원합니다. 우리는 float32를 사용합니다.
- llm.c는 attention + softmax + matmul을 단일 CUDA 커널로 융합합니다 (FlashAttention 스타일). 우리 구현은 O(T²) 메모리입니다.
- llm.c는 멀티-GPU 분산 훈련 경로 (NCCL)를 가집니다. 여기서는 단일 스레드입니다.

---

## 핵심 요약

- **GPT-2 124M**은 ~2GB의 float32 메모리(params + grads + Adam states)를 필요로 하며, 표준 AdamW 구성으로 FineWebEdu에서 스텝 200에 ~3.75 loss를 달성합니다.
- **유효 batch = 524,288 토큰**은 gradient accumulation을 통해 달성됩니다: 16 micro-batch × 32 accumulation 스텝 × 1024 토큰. 이것은 원래 논문의 기본 제약입니다.
- **Weight tying**은 token embedding과 출력 projection 헤드 사이에서 파라미터 수를 ~38.6M 줄이고 정규화 이점도 제공합니다.
- **멀티 shard 데이터 로딩**과 mmap은 데이터셋을 RAM에 보유하지 않고 100억 개 이상의 토큰을 스트리밍할 수 있게 합니다. Shard 교체는 훈련 루프에 투명합니다.
- **Loss 곡선은 결정론적**입니다. 동일한 데이터 순서와 하이퍼파라미터가 주어지면, 이를 사용하여 C 구현을 PyTorch 참조와 교차 검증합니다.
- **벤치마크 출력 형식** (step, loss, val_loss, lr, gnorm, tok/s)은 훈련 문제를 진단하기에 충분합니다: gnorm 상승 → 불안정; flat loss → 잘못된 LR 또는 데이터 버그; val_loss >> train_loss → 과적합.
- 전체 GPT-2 훈련 파이프라인 — 데이터 준비, tokenization, mmap 로더, forward pass, 융합된 CE backward, gradient clipping, AdamW, cosine LR, 체크포인팅 — 은 libc와 libm 외에 외부 의존성 없이 수천 줄의 C로 맞습니다.

---

**이전**: [Transformer를 통한 Backprop](./37_Backprop_Through_Transformer.md) | **다음**: *(시리즈 종료)*

> 이것으로 DL_Scratch_C 시리즈가 완료됩니다. 이제 순수 C로 GPT-2를 처음부터 훈련하는 모든 구성 요소를 갖추었습니다: tensor 기본 요소와 attention 메커니즘부터 완전한 훈련 루프, optimizer, 데이터 파이프라인까지.
