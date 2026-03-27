# 35. Optimizers: SGD, Adam, AdamW, and Gradient Clipping

**Previous**: [Cross-Entropy Loss](./34_Cross_Entropy_Loss.md) | **Next**: [Training Loop](./36_Training_Loop.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Implement SGD with momentum and the Nesterov variant
2. Implement Adam with first- and second-moment estimates and bias correction
3. Understand why AdamW decouples weight decay from the gradient update
4. Implement global L2 gradient norm clipping
5. Design a function-pointer-based LR schedule interface compatible with any optimizer

---

## 1. Background: What Optimizers Do

An optimizer translates raw parameter gradients `g = ∂L/∂θ` into parameter updates `Δθ`. The core loop:

```
for each training step t:
    compute gradients g_t
    clip g_t if needed
    schedule lr_t
    optimizer_step(params, grads, state, lr_t)
```

Different optimizers differ in how they use gradient history (momentum), gradient magnitude (adaptive LR), and regularization (weight decay).

---

## 2. SGD with Momentum

### 2.1 Standard Momentum

```
v_t = β * v_{t-1} + g_t          (velocity update)
θ_t = θ_{t-1} - lr * v_t
```

With `β=0.9`, the velocity accumulates an exponential moving average of past gradients. This smooths out noisy gradient directions and accelerates convergence along consistent gradient directions.

### 2.2 Nesterov Momentum

Standard momentum looks at the gradient at the current position. Nesterov looks ahead:

```
v_t = β * v_{t-1} + g(θ - β * v_{t-1})
θ_t = θ_{t-1} - lr * v_t
```

In practice, with a reparametrization, the Nesterov update is:

```
v_t = β * v_{t-1} + g_t
θ_t = θ_{t-1} - lr * (β * v_t + g_t)
```

This is equivalent but avoids re-evaluating the gradient at the lookahead point.

### 2.3 Implementation

```c
#include <math.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    float *v;       /* velocity (momentum buffer) [n_params] */
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
 * sgd_update — apply one SGD (with optional momentum) parameter update.
 *
 * params : [n_params]  model parameters (updated in-place)
 * grads  : [n_params]  current gradients
 */
void sgd_update(float *params, const float *grads, SGDState *s) {
    float beta = s->momentum;
    float lr   = s->lr;
    int   n    = s->n_params;

    if (beta == 0.0f) {
        /* Vanilla SGD */
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

Adam (Kingma & Ba, 2014) maintains per-parameter adaptive learning rates using:
- `m1`: first moment (mean of gradients) — direction
- `m2`: second moment (mean of squared gradients) — magnitude

```
m1_t = β1 * m1_{t-1} + (1 - β1) * g_t
m2_t = β2 * m2_{t-1} + (1 - β2) * g_t²
m1_hat = m1_t / (1 - β1^t)       (bias correction)
m2_hat = m2_t / (1 - β2^t)
θ_t = θ_{t-1} - lr * m1_hat / (sqrt(m2_hat) + ε)
```

The bias correction terms `1 - β^t` account for the zero-initialization of the moment estimates: early in training, `m1` and `m2` are biased toward zero.

### 3.1 Implementation

```c
typedef struct {
    float *m1;      /* first moment  [n_params] */
    float *m2;      /* second moment [n_params] */
    float  beta1;   /* default 0.9              */
    float  beta2;   /* default 0.999            */
    float  eps;     /* default 1e-8             */
    float  lr;
    int    step;    /* current step (1-indexed for bias correction) */
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
 * adam_update — apply one Adam step.
 *
 * params : [n_params]  updated in-place
 * grads  : [n_params]  current gradients
 */
void adam_update(float *params, const float *grads, AdamState *s) {
    s->step++;
    float b1   = s->beta1;
    float b2   = s->beta2;
    float eps  = s->eps;
    float lr   = s->lr;
    int   n    = s->n_params;

    /* Bias correction factors */
    float bc1 = 1.0f - powf(b1, (float)s->step);
    float bc2 = 1.0f - powf(b2, (float)s->step);
    /* Precompute effective lr = lr * sqrt(bc2) / bc1 */
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

## 4. AdamW: Decoupled Weight Decay

Standard Adam applies weight decay as an L2 gradient penalty:

```c
/* L2 penalty version (what Adam does by default): */
grads[i] += weight_decay * params[i];   /* add to gradient */
/* Then apply Adam update — weight decay is folded into m1 and m2 */
```

The problem: Adam adaptively scales the effective update by `1/sqrt(m2)`. When you add `λ·θ` to the gradient, the weight decay penalty also gets scaled — heavy damping for directions with large gradient history, light damping for directions with small history. This is wrong: weight decay should penalize large weights uniformly.

**AdamW fix**: apply weight decay directly to parameters, separately from the gradient update:

```c
params[i] -= lr * weight_decay * params[i];   /* weight decay FIRST */
/* Then: normal Adam update using clean gradient (no L2 term) */
```

### 4.1 Implementation

```c
typedef struct {
    AdamState base;
    float weight_decay;
} AdamWState;

AdamWState *adamw_new(int n_params, float lr,
                      float beta1, float beta2, float eps,
                      float weight_decay) {
    AdamWState *s = (AdamWState *)calloc(1, sizeof(AdamWState));
    /* Initialize the embedded Adam state */
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
 * adamw_update — AdamW: decouple weight decay from gradient update.
 *
 * Weight decay is applied to params before the Adam gradient step.
 * Embedding and LayerNorm parameters should have weight_decay = 0.
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
        /* Step 1: weight decay (decoupled) */
        params[i] *= (1.0f - lr * wd);

        /* Step 2: Adam gradient update (no L2 term in gradient) */
        float g   = grads[i];
        a->m1[i]  = b1 * a->m1[i] + (1.0f - b1) * g;
        a->m2[i]  = b2 * a->m2[i] + (1.0f - b2) * g * g;
        params[i] -= lr_corr * a->m1[i] / (sqrtf(a->m2[i]) + eps);
    }
}
```

**Standard hyperparameters for LLM training** (GPT-2):
- `lr = 6e-4`, `beta1 = 0.9`, `beta2 = 0.95` (not 0.999 — lower β2 for LLMs)
- `eps = 1e-8`, `weight_decay = 0.1`
- Apply weight decay to: linear weights, embedding table
- **No** weight decay on: LayerNorm parameters (γ, β), bias terms

---

## 5. Gradient Clipping

Large gradient norms cause training instability ("gradient explosion"). The standard fix is to clip all gradients by the **global L2 norm**:

```
If ||g|| > clip_value:
    g ← g * (clip_value / ||g||)
```

This scales all gradients down by the same factor, preserving their relative magnitudes (direction in parameter space).

### 5.1 Implementation

```c
/*
 * clip_grad_norm — clip all parameter gradients by global L2 norm.
 *
 * grads      : array of gradient pointers [num_params_groups]
 * sizes      : number of elements in each gradient array
 * n_groups   : number of parameter groups
 * max_norm   : clipping threshold (e.g., 1.0)
 *
 * Returns: the pre-clipping global gradient norm.
 */
float clip_grad_norm(float **grads, const int *sizes, int n_groups,
                     float max_norm)
{
    /* Compute global L2 norm */
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
 * Simple single-array version for convenience:
 */
float clip_grad_norm_flat(float *grads, int n, float max_norm) {
    float *g = grads;
    return clip_grad_norm(&g, &n, 1, max_norm);
}
```

Typical values: `max_norm = 1.0` for Transformers. Karpathy's llm.c uses 1.0.

### 5.2 Gradient Norm Monitoring

Always log the pre-clipping gradient norm. A norm consistently above `max_norm` indicates a training problem (bad LR, batch size too large, or a bug in the backward pass).

```c
void training_step(float *params, float *grads, int n, AdamWState *opt,
                   float lr, float max_norm, int step, FILE *log)
{
    /* 1. Clip gradients */
    float gnorm = clip_grad_norm_flat(grads, n, max_norm);

    /* 2. Set LR from schedule */
    opt->base.lr = lr;

    /* 3. Optimizer step */
    adamw_update(params, grads, opt);

    /* 4. Log */
    if (log) fprintf(log, "step=%d gnorm=%.4f lr=%.6f\n", step, gnorm, lr);
}
```

---

## 6. LR Schedule as a Function Pointer

Using a function pointer for the LR schedule makes the optimizer agnostic to the schedule type:

```c
/* Schedule function signature: (step, total_steps, config) → lr */
typedef float (*LRScheduleFn)(int step, int total_steps, const void *cfg);

/* Cosine schedule configuration */
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

/* Constant schedule (for fine-tuning experiments) */
float constant_schedule(int step, int total_steps, const void *cfg_) {
    (void)step; (void)total_steps;
    return *(const float *)cfg_;
}

/* Usage in training loop: */
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

This pattern is clean, testable, and avoids coupling the optimizer to any particular LR schedule.

---

## Key Takeaways

- **SGD with momentum** accumulates gradient direction history (velocity). Nesterov applies the correction before computing the gradient, giving faster convergence.
- **Adam** tracks per-parameter gradient magnitude via the second moment `m2`. Parameters with historically large gradients get smaller effective LR, and vice versa — adaptive learning rates.
- **AdamW** fixes Adam's broken weight decay by applying `param *= (1 - lr * wd)` separately from the gradient step. Never fold weight decay into the gradient when using Adam.
- **β2 = 0.95** (not 0.999) is preferred for Transformer training because it adapts faster to gradient magnitude changes — the gradient distribution is non-stationary during LM training.
- **Gradient clipping by global L2 norm** preserves gradient direction. Always log the pre-clip norm — a persistently clipped norm is a training instability signal.
- **Function pointer LR schedule** cleanly separates the optimizer update from the LR schedule. Swap cosine/linear/constant schedules without modifying the optimizer code.
- **No weight decay on LayerNorm, bias**: weight decay on normalization parameters can prevent convergence. Only apply to weight matrices and embedding tables.

---

**Previous**: [Cross-Entropy Loss](./34_Cross_Entropy_Loss.md) | **Next**: [Training Loop](./36_Training_Loop.md)

> Next lesson builds the full LLM training loop with an mmap-based data loader for pre-tokenized binary files.
