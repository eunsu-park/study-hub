# 36. Full LLM Training Loop with mmap Data Loader

**Previous**: [Optimizers](./35_Optimizers.md) | **Next**: [Backprop Through the Transformer](./37_Backprop_Through_Transformer.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Implement an mmap-based data loader for pre-tokenized binary token files
2. Build a token batch sampler that produces (inputs, targets) pairs without RAM overhead
3. Structure the full training loop: zero_grad → forward → loss → backward → clip → step → log
4. Implement gradient accumulation to simulate large batch sizes on limited hardware
5. Save and restore training checkpoints using `fwrite`/`fread` on raw parameter arrays

---

## 1. Why mmap for the Data Loader?

A GPT-2 pre-training dataset (FineWebEdu, 10 billion tokens) stores tokens as `uint16_t` values:

```
10B tokens × 2 bytes = 20 GB
```

Loading 20 GB into RAM is impractical. The standard solution is **memory-mapped I/O** (`mmap`). The OS maps the file into the process's virtual address space. Pages are loaded on-demand from disk when accessed — you read from a `uint16_t *` pointer exactly as if it were RAM, and the OS handles the rest.

Benefits:
- No explicit `fread` loop — just pointer arithmetic
- OS manages caching via the page cache (shared across processes)
- Sequential access patterns trigger prefetching automatically
- No 4 GB `malloc` needed — virtual address space is cheap

---

## 2. Data File Format

Pre-tokenized data is stored as a flat binary file:

```
[uint32_t magic]       4 bytes  — 0x20240520 (version marker)
[uint32_t n_tokens]    4 bytes  — number of tokens in file
[uint16_t tokens[n]]   2*n bytes — BPE token ids (GPT-2: values 0..50256)
```

To create such a file from Python (data preparation step):

```python
import numpy as np
tokens = tokenizer.encode(text)              # list of int
arr = np.array(tokens, dtype=np.uint16)
header = np.array([0x20240520, len(arr)], dtype=np.uint32)
with open("train.bin", "wb") as f:
    f.write(header.tobytes())
    f.write(arr.tobytes())
```

---

## 3. mmap Data Loader

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
    /* mmap state */
    int       fd;
    uint8_t  *data;          /* mmap'd bytes                             */
    size_t    file_size;     /* total file size in bytes                 */

    /* Token view (points inside the mmap'd region) */
    uint16_t *tokens;        /* pointer to first token                   */
    size_t    n_tokens;      /* number of tokens in file                 */

    /* Batch state */
    int       B;             /* batch size                               */
    int       T;             /* sequence length                          */
    size_t    pos;           /* current position in token stream (index) */
} DataLoader;

/*
 * dataloader_init — open a pre-tokenized binary file and mmap it.
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

    /* Validate header */
    uint32_t *header = (uint32_t *)dl->data;
    assert(header[0] == DATA_MAGIC && "bad magic");
    uint32_t n_tokens_u32 = header[1];

    dl->tokens   = (uint16_t *)(dl->data + 2 * sizeof(uint32_t));
    dl->n_tokens = (size_t)n_tokens_u32;

    size_t expected = 2 * sizeof(uint32_t) + dl->n_tokens * sizeof(uint16_t);
    assert(dl->file_size >= expected && "file truncated");

    /* Hint to OS: sequential access pattern */
    madvise(dl->data, dl->file_size, MADV_SEQUENTIAL);

    return 0;
}

void dataloader_free(DataLoader *dl) {
    munmap(dl->data, dl->file_size);
    close(dl->fd);
}

/*
 * dataloader_next_batch — fill inputs and targets for one batch.
 *
 * inputs  : [B, T]  int32 (filled with token ids)
 * targets : [B, T]  int32 (targets = inputs shifted right by 1)
 *
 * The loader treats the token stream as a circular ring buffer:
 * when it reaches the end, it wraps around to position 0.
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

    /* Advance position by B*T tokens */
    dl->pos = (dl->pos + (size_t)(B * T)) % n;
}

/*
 * dataloader_reset — restart from position 0 (for new epoch).
 */
void dataloader_reset(DataLoader *dl) { dl->pos = 0; }
```

---

## 4. Full Training Loop

```c
#include <math.h>
#include <time.h>

/*
 * Training configuration struct — all hyperparameters in one place.
 */
typedef struct {
    /* Model */
    int vocab_size;   /* e.g. 50257 (GPT-2)          */
    int n_layer;      /* e.g. 12                      */
    int n_head;       /* e.g. 12                      */
    int n_embd;       /* e.g. 768                     */
    int block_size;   /* context length T, e.g. 1024  */

    /* Data */
    const char *train_path;
    const char *val_path;
    int B;            /* batch size, e.g. 16          */

    /* Optimization */
    float lr_max;         /* peak LR, e.g. 6e-4        */
    float lr_min;         /* final LR, e.g. 6e-5       */
    float weight_decay;   /* e.g. 0.1                  */
    float grad_clip;      /* e.g. 1.0                  */
    float beta1;          /* e.g. 0.9                  */
    float beta2;          /* e.g. 0.95                 */
    int   warmup_steps;   /* e.g. 715 (for 10B tokens) */
    int   max_steps;      /* total training steps       */

    /* Gradient accumulation */
    int   grad_accum_steps; /* effective_batch = B * grad_accum_steps */

    /* Logging / checkpointing */
    int   log_interval;    /* print loss every N steps   */
    int   val_interval;    /* evaluate every N steps      */
    int   save_interval;   /* checkpoint every N steps    */
    const char *ckpt_path;
} TrainConfig;

/*
 * zero_grad — set all gradients to zero before accumulation.
 */
void zero_grad(float *grads, int n) {
    memset(grads, 0, (size_t)n * sizeof(float));
}

/*
 * train — full training loop.
 *
 * model_forward  : (tokens[B,T], logits[B,T,V]) → void
 * model_backward : (dlogits[B,T,V]) → updates grads[] in-place
 * params, grads  : flattened parameter and gradient arrays
 * n_params       : total parameter count
 * cfg            : training configuration
 */
void train(void (*model_forward)(const int *, float *, int, int),
           void (*model_backward)(const float *, float *, int, int),
           float *params, float *grads, int n_params,
           const TrainConfig *cfg)
{
    DataLoader train_loader, val_loader;
    dataloader_init(&train_loader, cfg->train_path, cfg->B, cfg->block_size);
    dataloader_init(&val_loader,   cfg->val_path,   cfg->B, cfg->block_size);

    /* Allocate buffers */
    int BT = cfg->B * cfg->block_size;
    int BTV = BT * cfg->vocab_size;
    int *inputs   = (int   *)malloc((size_t)BT  * sizeof(int));
    int *targets  = (int   *)malloc((size_t)BT  * sizeof(int));
    float *logits  = (float *)malloc((size_t)BTV * sizeof(float));
    float *dlogits = (float *)malloc((size_t)BTV * sizeof(float));

    /* Initialize optimizer */
    AdamWState *opt = adamw_new(n_params, cfg->lr_max,
                                cfg->beta1, cfg->beta2, 1e-8f,
                                cfg->weight_decay);

    /* LR schedule */
    CosineLRCfg lr_cfg = {
        .lr_max       = cfg->lr_max,
        .lr_min       = cfg->lr_min,
        .warmup_steps = cfg->warmup_steps
    };

    /* Loss logging */
    FILE *loss_csv = fopen("loss.csv", "w");
    if (loss_csv) fprintf(loss_csv, "step,train_loss,val_loss,lr,gnorm\n");

    for (int step = 0; step < cfg->max_steps; step++) {
        double t0 = (double)clock() / CLOCKS_PER_SEC;

        /* --- 1. Zero gradients --- */
        zero_grad(grads, n_params);
        float accum_loss = 0.0f;

        /* --- 2. Gradient accumulation over micro-batches --- */
        for (int micro = 0; micro < cfg->grad_accum_steps; micro++) {
            dataloader_next_batch(&train_loader, inputs, targets);

            /* Forward pass */
            model_forward(inputs, logits, cfg->B, cfg->block_size);

            /* Fused CE loss + backward w.r.t. logits */
            float loss = fused_ce_forward_backward(
                logits, targets, dlogits,
                BT, cfg->vocab_size
            );
            accum_loss += loss;

            /* Scale dlogits by 1/grad_accum_steps (to average micro-batches) */
            float scale = 1.0f / (float)cfg->grad_accum_steps;
            for (int i = 0; i < BTV; i++) dlogits[i] *= scale;

            /* Backward pass — accumulates into grads[] */
            model_backward(dlogits, grads, cfg->B, cfg->block_size);
        }
        accum_loss /= (float)cfg->grad_accum_steps;

        /* --- 3. Gradient clipping --- */
        float gnorm = clip_grad_norm_flat(grads, n_params, cfg->grad_clip);

        /* --- 4. Update LR and optimizer step --- */
        float lr = cosine_schedule(step, cfg->max_steps, &lr_cfg);
        opt->base.lr = lr;
        adamw_update(params, grads, opt);

        double t1 = (double)clock() / CLOCKS_PER_SEC;

        /* --- 5. Logging --- */
        if (step % cfg->log_interval == 0) {
            float tok_per_sec = (float)(cfg->B * cfg->block_size
                                        * cfg->grad_accum_steps)
                                / (float)(t1 - t0 + 1e-9);
            printf("step %5d | loss %.4f | lr %.2e | gnorm %.3f | %.0f tok/s\n",
                   step, accum_loss, lr, gnorm, tok_per_sec);
        }

        /* --- 6. Validation --- */
        if (step % cfg->val_interval == 0) {
            float val_loss = 0.0f;
            int val_steps  = 20;
            for (int v = 0; v < val_steps; v++) {
                dataloader_next_batch(&val_loader, inputs, targets);
                model_forward(inputs, logits, cfg->B, cfg->block_size);
                /* Compute loss only — no backward */
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

        /* --- 7. Checkpoint --- */
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

## 5. Gradient Accumulation

When effective batch size `B_eff = B × grad_accum_steps` is needed but physical memory only fits `B`:

```
For each step:
  zero_grad()
  For micro in 0..grad_accum_steps-1:
      batch = next_batch()
      logits = forward(batch)
      loss = CE(logits, targets)
      dlogits = CE_backward(loss)
      dlogits *= 1/grad_accum_steps        ← scale before backward
      grads += backward(dlogits)           ← accumulate
  clip_grad_norm(grads)
  optimizer_step()
```

The key: **divide dlogits by `grad_accum_steps` before the backward pass** so that the accumulated gradients are equivalent to a single forward/backward on the full effective batch.

Example: GPT-2 124M training
- Physical batch: B=16, T=1024 → 16K tokens/step
- Gradient accumulation: 8 steps → 128K tokens effective batch
- GPU memory for one micro-batch: `16 × 1024 × 50257 × 4 ≈ 3.3 GB` (logits alone)

---

## 6. Checkpoint Saving and Loading

```c
typedef struct {
    int   step;
    float lr;
    int   adam_step;
} CheckpointHeader;

/*
 * save_checkpoint — write params + optimizer state to binary file.
 *
 * Format:
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
    printf("Checkpoint saved: %s (step %d)\n", path, step);
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
    printf("Checkpoint loaded: %s (step %d)\n", path, hdr.step);
    return 0;
}
```

---

## 7. Shakespeare Character-Level Example

For a self-contained training example, a character-level GPT on Shakespeare is ~200K tokens and trains in minutes on a CPU:

```c
/*
 * Character-level data: tokens are raw bytes (0-255), vocab_size=256.
 * File format: raw bytes (no header needed for char-level).
 *
 * Typical results:
 *   step    0 | loss 5.5452 (≈ log(256))  — random initialization
 *   step  100 | loss 2.1  — model learns character frequencies
 *   step  500 | loss 1.7  — model learns common n-grams
 *   step 2000 | loss 1.4  — Shakespeare-like output
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
    .grad_accum_steps = 1,    /* no accumulation needed for small model */

    .log_interval     = 10,
    .val_interval     = 250,
    .save_interval    = 1000,
    .ckpt_path        = "shakespeare_ckpt"
};
```

---

## Key Takeaways

- **mmap** maps a file into virtual address space: the token array is accessed via a plain pointer, and the OS handles demand paging. No need to `malloc` the entire dataset.
- **Circular position tracking** (`pos = (pos + B*T) % n_tokens`) provides endless data without special end-of-file handling.
- **Gradient accumulation** multiplies effective batch size at the cost of more forward/backward passes per optimizer step. Scale `dlogits` before accumulating, not after.
- **The training loop structure is fixed**: zero_grad → (micro-batch loop) → clip → lr_schedule → optimizer_step → log. Deviating from this order causes subtle bugs.
- **Checkpoint format**: write a small header (step, LR, optimizer step count) followed by raw `float` arrays for params, m1, m2. Simple `fwrite`/`fread` with no serialization overhead.
- **Loss logging to CSV** enables post-hoc analysis and plotting of training curves. Always log: step, train_loss, val_loss, lr, grad_norm.
- **Character-level Shakespeare** is the canonical quick validation: a 6-layer GPT with 384 embedding dim trains to perplexity ~4.0 in <5 minutes on CPU, confirming the full pipeline works end-to-end.

---

**Previous**: [Optimizers](./35_Optimizers.md) | **Next**: [Backprop Through the Transformer](./37_Backprop_Through_Transformer.md)

> Next lesson derives the complete backward pass through the Transformer: attention backward (dV, dK, dQ), layernorm backward, and a numerical gradient check on a 2-layer model.
