# Block 7 — Training Recipes (L34–L38)

Prerequisites: L34 (cross-entropy loss), L35 (optimizers: SGD, Adam, AdamW), L36 (gradient clipping), L37 (attention backward), L38 (data loading with mmap).

Compile: `gcc -std=c11 -Wall -Wextra -O2 -o ex ex.c -lm`

---

## Exercise 7.1 — Numerically Stable Log-Softmax

**Difficulty**: ★★

### Problem

Implement `log_softmax(float *out, const float *x, int n)` using the max-subtraction trick.

The naive formula `log(softmax(x))` computes `exp(x_i) / sum(exp(x_j))` first, which overflows for large `x_i`. The stable version is:

```
m     = max(x)
LSE   = m + log( sum( exp(x_j - m) ) )   (log-sum-exp)
log_softmax(x_i) = x_i - LSE
```

Then verify analytically that `∂(cross_entropy) / ∂x_i = softmax(x_i) - one_hot(y)_i`.

### Starter Code

```c
#include <stdio.h>
#include <math.h>
#include <float.h>
#include <string.h>

void log_softmax(float *out, const float *x, int n) {
    /* Step 1: max */
    float m = -FLT_MAX;
    /* TODO */

    /* Step 2: log-sum-exp */
    float lse = 0.0f;
    /* TODO: lse = m + log(sum(exp(x[j] - m))) */

    /* Step 3: out[i] = x[i] - lse */
    /* TODO */
}

/* Cross-entropy loss: L = -log_softmax(x)[y] */
float cross_entropy(const float *x, int n, int y) {
    float ls[n];
    log_softmax(ls, x, n);
    return -ls[y];
}

/* Gradient: dL/dx[i] = softmax(x)[i] - (i==y ? 1 : 0) */
void cross_entropy_grad(float *grad, const float *x, int n, int y) {
    /* TODO: compute softmax, then subtract 1 at index y */
}

/* Finite difference check */
int main(void) {
    int n=5, y=2;
    float x[5] = {1.0f, 2.0f, 3.0f, 0.5f, -1.0f};

    float ls[5];
    log_softmax(ls, x, n);
    printf("log_softmax: ");
    float s = 0;
    for (int i=0;i<n;i++){printf("%.4f ",ls[i]); s+=expf(ls[i]);}
    printf("\nsum(softmax)=%.6f (expected 1.0)\n", s);

    /* Loss */
    float L = cross_entropy(x, n, y);
    printf("CE loss (y=%d): %.6f\n", y, L);

    /* Analytical gradient */
    float grad[5];
    cross_entropy_grad(grad, x, n, y);
    printf("Analytical grad: ");
    for (int i=0;i<n;i++) printf("%.4f ",grad[i]);
    printf("\n");

    /* Numerical gradient */
    float eps=1e-4f, grad_num[5];
    for (int i=0;i<n;i++){
        float xi = x[i];
        float xp[5]; memcpy(xp,x,sizeof(xp)); xp[i]=xi+eps;
        float xm[5]; memcpy(xm,x,sizeof(xm)); xm[i]=xi-eps;
        grad_num[i] = (cross_entropy(xp,n,y) - cross_entropy(xm,n,y)) / (2*eps);
    }
    printf("Numerical  grad: ");
    float max_err=0;
    for (int i=0;i<n;i++){
        printf("%.4f ",grad_num[i]);
        float e=fabsf(grad[i]-grad_num[i]);
        if(e>max_err) max_err=e;
    }
    printf("\nMax error: %.2e (expected < 1e-4)\n", max_err);

    /* Overflow test */
    float x_big[5] = {1000.0f, 1001.0f, 1002.0f, 999.0f, 998.0f};
    float ls_big[5];
    log_softmax(ls_big, x_big, n);
    printf("log_softmax(large x)[2] = %.4f (should not be NaN)\n", ls_big[2]);
    return 0;
}
```

### Test Cases

| Input x | y | Expected CE loss |
|---------|---|-----------------|
| `[1,2,3,0.5,-1]` | 2 | `≈ 0.4076` |
| `[1000,1001,1002,999,998]` | 2 | `≈ 0.4076` (same, shifted) |
| `[-1000,-999,-998,-1001,-1002]` | 2 | `≈ 0.4076` |

Gradient check: `|analytical - numerical| < 1e-4` for all elements.

### Hints

1. LSE = `m + log(sum(exp(x_j - m)))` where `m = max(x)`.
2. After computing `log_softmax`, the gradient is simply `exp(log_softmax[i])` minus the one-hot vector.
3. The gradient `softmax - one_hot` means: all elements receive their softmax probability, except the true class which receives `softmax[y] - 1`.

### Solution Approach

`log_softmax` is three passes: find max, compute log-sum-exp, subtract. The gradient derivation follows from differentiating `L = -log_softmax(x)[y]` with respect to `x[i]`. The result `softmax(x) - one_hot(y)` is one of the most important formulas in deep learning — it shows that cross-entropy loss with softmax has a clean, bounded gradient.

---

## Exercise 7.2 — AdamW vs Adam on Toy Quadratic

**Difficulty**: ★★

### Problem

Implement both Adam and AdamW, then run them on the toy quadratic loss `L(w) = 0.5 * ||w||^2` with initial `w = [5, 5]`. Observe that AdamW decays `w` to zero while Adam's effective L2 regularization behaves differently.

Adam update:
```
m = β1*m + (1-β1)*g
v = β2*v + (1-β2)*g^2
m̂ = m / (1-β1^t)
v̂ = v / (1-β2^t)
w -= lr * m̂ / (sqrt(v̂) + ε)
```

AdamW update: same as Adam but add weight decay **directly** to the weight, not through the gradient:
```
w -= lr * (m̂ / (sqrt(v̂) + ε) + wd * w)
```

### Starter Code

```c
#include <stdio.h>
#include <math.h>
#include <string.h>

#define D 2

typedef struct {
    float m[D], v[D];  /* first and second moment estimates */
    int t;             /* step counter */
} AdamState;

void adam_step(float *w, float *grad, AdamState *s,
               float lr, float b1, float b2, float eps, float wd) {
    s->t++;
    float bc1 = 1.0f - powf(b1, s->t);
    float bc2 = 1.0f - powf(b2, s->t);
    for (int i = 0; i < D; i++) {
        /* L2 penalty baked into gradient (Adam-style) */
        float g = grad[i] + wd * w[i];
        s->m[i] = b1 * s->m[i] + (1-b1) * g;
        s->v[i] = b2 * s->v[i] + (1-b2) * g * g;
        float m_hat = s->m[i] / bc1;
        float v_hat = s->v[i] / bc2;
        w[i] -= lr * m_hat / (sqrtf(v_hat) + eps);
    }
}

void adamw_step(float *w, float *grad, AdamState *s,
                float lr, float b1, float b2, float eps, float wd) {
    s->t++;
    float bc1 = 1.0f - powf(b1, s->t);
    float bc2 = 1.0f - powf(b2, s->t);
    /* TODO: update m, v using ONLY the true gradient (no wd*w),
             then apply weight decay separately after the Adam step */
}

int main(void) {
    float lr=0.1f, b1=0.9f, b2=0.999f, eps=1e-8f, wd=0.1f;

    float w_adam[D]  = {5.0f, 5.0f};
    float w_adamw[D] = {5.0f, 5.0f};
    AdamState s_adam  = {{0},{0},0};
    AdamState s_adamw = {{0},{0},0};

    printf("%-6s %-12s %-12s %-12s %-12s\n",
           "Step", "w_adam[0]", "w_adamw[0]", "L_adam", "L_adamw");
    for (int t = 1; t <= 200; t++) {
        /* Loss = 0.5*||w||^2, gradient = w */
        float g_adam[D]  = {w_adam[0],  w_adam[1]};
        float g_adamw[D] = {w_adamw[0], w_adamw[1]};

        adam_step (w_adam,  g_adam,  &s_adam,  lr, b1, b2, eps, wd);
        adamw_step(w_adamw, g_adamw, &s_adamw, lr, b1, b2, eps, wd);

        if (t <= 5 || t % 50 == 0) {
            float L_a  = 0.5f*(w_adam[0]*w_adam[0]   + w_adam[1]*w_adam[1]);
            float L_aw = 0.5f*(w_adamw[0]*w_adamw[0] + w_adamw[1]*w_adamw[1]);
            printf("%-6d %-12.6f %-12.6f %-12.6f %-12.6f\n",
                   t, w_adam[0], w_adamw[0], L_a, L_aw);
        }
    }
    /* AdamW should converge faster/closer to 0 than Adam for this loss */
    float final_adam  = w_adam[0]*w_adam[0]  + w_adam[1]*w_adam[1];
    float final_adamw = w_adamw[0]*w_adamw[0] + w_adamw[1]*w_adamw[1];
    printf("\nFinal ||w||^2 — Adam: %.6f  AdamW: %.6f\n", final_adam, final_adamw);
    printf("AdamW converges closer to 0: %s\n", final_adamw < final_adam ? "YES" : "NO");
    return 0;
}
```

### Test Cases

After 200 steps on `L = 0.5*||w||^2` with `lr=0.1, wd=0.1`:
- AdamW should converge to lower `||w||^2` than Adam.
- Both should decrease monotonically.
- The difference arises because Adam applies weight decay through the gradient (which gets scaled by the Adam adaptive step), while AdamW applies it directly.

### Hints

1. AdamW: update `m` and `v` using **raw gradient** only (no `wd*w` term).
2. Then apply: `w[i] -= lr * (m_hat / (sqrt(v_hat) + eps) + wd * w[i])`.
3. For Adam: the gradient passed to the moment updates already includes `wd*w[i]`.
4. The key difference: in AdamW the weight decay is not scaled by the adaptive step size.

### Solution Approach

The code difference is subtle but important. In AdamW, the weight decay `wd * w` is applied after the Adam adaptive update, not folded into the gradient before moment estimation. This means weight decay in AdamW is not dampened by the second moment — it decays the weights at a consistent rate regardless of gradient magnitude. This is why AdamW is preferred over Adam+L2 for large models.

---

## Exercise 7.3 — `global_grad_clip`

**Difficulty**: ★★

### Problem

Implement `global_grad_clip(float **grads, int *sizes, int n_params, float max_norm)` that clips the **global gradient norm** to `max_norm`.

Global norm: `||g|| = sqrt( sum over all parameters of sum(g_ij^2) )`

If `||g|| > max_norm`, scale all gradients by `max_norm / ||g||`. Otherwise, leave them unchanged.

After clipping, verify that the resulting norm equals exactly `max_norm` (when clipping occurred).

### Starter Code

```c
#include <stdio.h>
#include <math.h>

/*
 * Clip global gradient norm in-place.
 *   grads   : array of n_params gradient arrays
 *   sizes   : number of elements in each gradient array
 *   n_params: number of parameter groups
 *   max_norm: target maximum norm
 *
 * Returns the pre-clipping global norm.
 */
float global_grad_clip(float **grads, const int *sizes, int n_params, float max_norm) {
    /* Step 1: compute global norm */
    float norm2 = 0.0f;
    /* TODO */

    float norm = sqrtf(norm2);

    /* Step 2: scale if necessary */
    if (norm > max_norm) {
        float scale = max_norm / norm;
        /* TODO: multiply every gradient element by scale */
    }

    return norm;
}

int main(void) {
    /* Three parameter groups */
    float g0[3] = {3.0f, 0.0f, 0.0f};  /* norm contribution = 9 */
    float g1[2] = {4.0f, 0.0f};        /* norm contribution = 16 */
    float g2[1] = {0.0f};              /* norm contribution = 0  */
    /* Global norm = sqrt(9+16) = 5 */

    float *grads[3] = {g0, g1, g2};
    int    sizes[3] = {3, 2, 1};

    float max_norm = 2.5f;
    float pre_norm = global_grad_clip(grads, sizes, 3, max_norm);
    printf("Pre-clip global norm: %.4f (expected 5.0)\n", pre_norm);

    /* Verify post-clip norm */
    float post_norm2 = 0;
    for (int p=0;p<3;p++) for (int i=0;i<sizes[p];i++) post_norm2+=grads[p][i]*grads[p][i];
    float post_norm = sqrtf(post_norm2);
    printf("Post-clip global norm: %.6f (expected %.6f)\n", post_norm, max_norm);
    printf("Difference: %.2e (expected < 1e-6)\n", fabsf(post_norm - max_norm));

    /* Test: no clip needed when norm < max_norm */
    float g3[2] = {1.0f, 0.5f};
    float *grads2[1] = {g3};
    int sizes2[1] = {2};
    float max_norm2 = 10.0f;
    float pre2 = global_grad_clip(grads2, sizes2, 1, max_norm2);
    printf("\nNo-clip test: pre_norm=%.4f, g3[0]=%.4f (expected 1.0, unchanged)\n",
           pre2, g3[0]);
    return 0;
}
```

### Test Cases

| Pre-clip norm | max_norm | Expected post-clip norm | g0[0] after clip |
|--------------|----------|------------------------|-----------------|
| 5.0 | 2.5 | 2.5 | 3.0 * (2.5/5.0) = 1.5 |
| 1.118 | 10.0 | 1.118 (no clip) | unchanged |
| 0 | 1.0 | 0 (no clip, zero grad) | 0 |

### Hints

1. Accumulate the squared norm across all parameter arrays, take the square root once.
2. Only apply scaling if `norm > max_norm`. When `norm == 0`, skip scaling (division by zero).
3. The scale factor is `max_norm / norm` — apply it uniformly to every element of every gradient.

### Solution Approach

Two passes over all gradients: first to accumulate the squared norm, then (conditionally) to apply the scale. The result is that the gradient direction is preserved but the magnitude is capped. Global clipping is used in LLM training to prevent "gradient explosions" that would destabilize the optimizer, especially early in training.

---

## Exercise 7.4 — Attention Backward (Single Head)

**Difficulty**: ★★★★

### Problem

Implement the backward pass for single-head self-attention (T=4, d_head=8):

Forward:
```
S = Q @ K^T / sqrt(d_head)   [T×T]
A = softmax(S)                [T×T]  (row-wise)
out = A @ V                   [T×d_head]
```

Backward (given d_out `[T×d_head]`):
```
dV  = A^T @ d_out
dA  = d_out @ V^T
dS  = dA * A - A * (sum of dA*A along columns)   (softmax backward)
dQ  = dS @ K / sqrt(d_head)
dK  = dS^T @ Q / sqrt(d_head)
```

Compare analytical gradients to finite differences.

### Starter Code

```c
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <float.h>

#define T      4
#define DHEAD  8

void matmul(const float *A, const float *B, float *C, int M, int K, int N) {
    memset(C, 0, M*N*sizeof(float));
    for(int i=0;i<M;i++) for(int k=0;k<K;k++) for(int j=0;j<N;j++)
        C[i*N+j] += A[i*K+k] * B[k*N+j];
}

void matmul_T2(const float *A, const float *B, float *C, int M, int K, int N) {
    /* C[M][N] = A[M][K] @ B[N][K]^T  (B is transposed) */
    memset(C, 0, M*N*sizeof(float));
    for(int i=0;i<M;i++) for(int j=0;j<N;j++)
        for(int k=0;k<K;k++) C[i*N+j] += A[i*K+k] * B[j*K+k];
}

void matmul_T1(const float *A, const float *B, float *C, int M, int K, int N) {
    /* C[M][N] = A[K][M]^T @ B[K][N] */
    memset(C, 0, M*N*sizeof(float));
    for(int i=0;i<M;i++) for(int k=0;k<K;k++) for(int j=0;j<N;j++)
        C[i*N+j] += A[k*M+i] * B[k*N+j];
}

void softmax_rows(float *A, int T_) {
    for(int i=0;i<T_;i++){
        float m=-FLT_MAX, s=0;
        for(int j=0;j<T_;j++) if(A[i*T_+j]>m) m=A[i*T_+j];
        for(int j=0;j<T_;j++){A[i*T_+j]=expf(A[i*T_+j]-m);s+=A[i*T_+j];}
        for(int j=0;j<T_;j++) A[i*T_+j]/=s;
    }
}

void attention_forward_full(const float *Q, const float *K, const float *V,
                             float *S, float *A, float *out) {
    float scale = 1.0f / sqrtf(DHEAD);
    /* S = Q @ K^T * scale */
    matmul_T2(Q, K, S, T, DHEAD, T);
    for(int i=0;i<T*T;i++) S[i] *= scale;
    /* A = softmax(S) */
    memcpy(A, S, T*T*sizeof(float));
    softmax_rows(A, T);
    /* out = A @ V */
    matmul(A, V, out, T, T, DHEAD);
}

void attention_backward(const float *Q, const float *K, const float *V,
                         const float *A, /* saved from forward */
                         const float *d_out,
                         float *dQ, float *dK, float *dV) {
    float scale = 1.0f / sqrtf(DHEAD);

    /* dV = A^T @ d_out */
    /* TODO */

    /* dA = d_out @ V^T  [T][T] */
    float dA[T*T];
    /* TODO */

    /* Softmax backward: dS[i][j] = A[i][j] * (dA[i][j] - sum_k(dA[i][k]*A[i][k])) */
    float dS[T*T];
    /* TODO */

    /* dQ = dS @ K * scale */
    /* TODO */

    /* dK = dS^T @ Q * scale */
    /* TODO */
}

float sum_all(const float *x, int n){float s=0;for(int i=0;i<n;i++)s+=x[i];return s;}

int main(void) {
    float Q[T*DHEAD], K[T*DHEAD], V[T*DHEAD];
    srand(42);
    for(int i=0;i<T*DHEAD;i++){
        Q[i]=(float)(rand()%20-10)*0.1f;
        K[i]=(float)(rand()%20-10)*0.1f;
        V[i]=(float)(rand()%20-10)*0.1f;
    }

    float S[T*T], A[T*T], out[T*DHEAD];
    attention_forward_full(Q, K, V, S, A, out);

    float d_out[T*DHEAD];
    for(int i=0;i<T*DHEAD;i++) d_out[i]=1.0f;  /* L = sum(out) */

    float dQ[T*DHEAD], dK[T*DHEAD], dV[T*DHEAD];
    memset(dQ,0,sizeof(dQ)); memset(dK,0,sizeof(dK)); memset(dV,0,sizeof(dV));
    attention_backward(Q, K, V, A, d_out, dQ, dK, dV);

    /* Finite difference for dQ */
    float eps=1e-4f;
    int ok=1;
    printf("Checking dQ[0..3] vs finite differences:\n");
    for(int i=0;i<4;i++){
        float Qp[T*DHEAD], Qm[T*DHEAD];
        memcpy(Qp,Q,sizeof(Q)); memcpy(Qm,Q,sizeof(Q));
        Qp[i]+=eps; Qm[i]-=eps;
        float Sp[T*T],Ap[T*T],outp[T*DHEAD];
        float Sm[T*T],Am[T*T],outm[T*DHEAD];
        attention_forward_full(Qp,K,V,Sp,Ap,outp);
        attention_forward_full(Qm,K,V,Sm,Am,outm);
        float num = (sum_all(outp,T*DHEAD)-sum_all(outm,T*DHEAD))/(2*eps);
        float diff = fabsf(dQ[i]-num);
        printf("  dQ[%d] anal=%.5f num=%.5f diff=%.2e\n",i,dQ[i],num,diff);
        if(diff>1e-2f) ok=0;
    }
    printf("%s\n", ok?"PASS":"FAIL");
    return 0;
}
```

### Test Cases

- All `|dQ[i] - num[i]|` < 1e-2 (attention backward is numerically tricky due to softmax chaining).
- Similarly for dK and dV.
- dV is the easiest to verify — it is just `A^T @ d_out`.

### Hints

1. Softmax backward: `dS[i][j] = A[i][j] * (dA[i][j] - dot(dA[i], A[i]))` for row `i`.
2. The `dot(dA[i], A[i])` term is a row-wise dot product — compute it once per row.
3. Start by verifying `dV` (no softmax, just a matmul), then `dA`, then `dS`, then `dQ` and `dK`.
4. Use a small epsilon (1e-4) and loose tolerance (1e-2) — softmax backprop has limited precision in float32.

### Solution Approach

The attention backward is a chain of matrix multiplications and a softmax Jacobian. The softmax Jacobian for row i applied to `dA[i]` is `diag(A[i]) - A[i]*A[i]^T` multiplied by `dA[i]`, which simplifies to `A[i] * (dA[i] - dot(A[i], dA[i]))`. Work through each step symbolically, implement in code, then verify with finite differences.

---

## Exercise 7.5 — mmap DataLoader on Shakespeare

**Difficulty**: ★★★

### Problem

Implement a memory-mapped DataLoader for character-level language modeling on the Shakespeare dataset. Run 10 training steps on a small GPT and report the loss.

The DataLoader:
- Memory-maps a pre-tokenized binary file (uint16 token IDs).
- Returns random mini-batches of shape `[B, T+1]` (T+1 to get input and target from the same window).
- Input: `batch[:, :T]`, Target: `batch[:, 1:]` (next-token prediction).

### Starter Code

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

typedef struct {
    uint16_t *data;   /* mmap pointer */
    size_t    n_tok;  /* total number of tokens */
    int       B, T;   /* batch size, sequence length */
} DataLoader;

int dataloader_init(DataLoader *dl, const char *path, int B, int T) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) { perror("open"); return -1; }
    struct stat st;
    fstat(fd, &st);
    dl->n_tok = st.st_size / sizeof(uint16_t);
    dl->data  = mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (dl->data == MAP_FAILED) { perror("mmap"); return -1; }
    dl->B = B; dl->T = T;
    return 0;
}

/*
 * Fill x[B][T] and y[B][T] from a random offset.
 * y[b][t] = x[b][t+1] (next token targets).
 */
void dataloader_batch(DataLoader *dl, int *x, int *y) {
    int B = dl->B, T = dl->T;
    /* TODO: pick a random starting position (leave T+1 tokens available),
             fill x and y by reading dl->data */
}

void dataloader_free(DataLoader *dl) {
    /* TODO: munmap */
}

int main(void) {
    /*
     * Prepare data: download Shakespeare and tokenize to binary.
     * Quick way using Python:
     *   python3 -c "
     *   import urllib.request, numpy as np
     *   url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
     *   text = urllib.request.urlopen(url).read().decode()
     *   chars = sorted(set(text))
     *   stoi = {c:i for i,c in enumerate(chars)}
     *   ids = np.array([stoi[c] for c in text], dtype=np.uint16)
     *   ids.tofile('shakespeare.bin')
     *   print(f'Vocab size: {len(chars)}, Tokens: {len(ids)}')
     *   "
     */

    DataLoader dl;
    if (dataloader_init(&dl, "shakespeare.bin", 4, 64) != 0) {
        printf("Could not open shakespeare.bin. Please generate it first.\n");
        printf("See the Python snippet above.\n");
        return 1;
    }

    printf("Loaded %zu tokens, B=%d T=%d\n", dl.n_tok, dl.B, dl.T);

    int x[4*64], y[4*64];
    for (int step = 0; step < 10; step++) {
        dataloader_batch(&dl, x, y);
        /* TODO: run forward + backward on your GPT model */
        /* For now, just print token statistics */
        float mean_tok = 0;
        for (int i = 0; i < dl.B * dl.T; i++) mean_tok += x[i];
        mean_tok /= (dl.B * dl.T);
        printf("Step %2d: mean_token=%.2f (first token=%d)\n",
               step+1, mean_tok, x[0]);
    }

    dataloader_free(&dl);
    return 0;
}
```

### Expected Output (once GPT is integrated)

```
Step  1: loss=4.1562
Step  2: loss=3.9812
...
Step 10: loss=3.5234
```

(Initial loss for character-level LM on Shakespeare with vocab≈65 chars should be near `log(65) ≈ 4.17`.)

### Hints

1. `mmap` with `MAP_PRIVATE` gives a read-only view of the file without loading it all into RAM.
2. Random offset: `start = rand() % (dl->n_tok - B*T - 1)`. Make sure the window does not overflow.
3. For input `x[b][t]` use `dl->data[start + b*T + t]`, for target `y[b][t]` use `dl->data[start + b*T + t + 1]`.
4. `munmap(dl->data, n_tok * sizeof(uint16_t))` to free.

### Solution Approach

Memory-mapping avoids loading the entire dataset into RAM — the OS pages in only what is accessed. This is critical for large datasets. The DataLoader here is intentionally minimal; production dataloaders (like karpathy's llm.c) include prefetching, shuffling at the document level, and multi-process workers. Start with the mmap plumbing, verify you can sample batches, then wire in your training loop.
