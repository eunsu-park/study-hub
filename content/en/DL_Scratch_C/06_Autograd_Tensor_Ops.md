# 06. Autograd Tensor Ops

**Previous**: [Autograd Engine](./05_Autograd_Engine.md) | **Next**: [Memory Manager](./07_Memory_Manager.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Derive and implement the backward pass for matrix multiplication
2. Derive and implement the backward pass for softmax
3. Implement the fused softmax-cross-entropy backward pass
4. Verify all gradients with finite difference tests
5. Compose multiple autograd ops to compute a full MLP forward+backward

---

## 1. Matmul Backward

### Forward

```
C = A @ B      (A: [M,K], B: [K,N], C: [M,N])
```

### Backward Derivation

We receive `dL/dC` (same shape as `C`) and need to compute `dL/dA` and `dL/dB`.

Using the chain rule for matrix products:

```
dL/dA = dL/dC @ B^T       [M,N] @ [N,K] = [M,K]  ✓
dL/dB = A^T @ dL/dC       [K,M] @ [M,N] = [K,N]  ✓
```

**Derivation sketch**: For any scalar loss `L`,
```
dL/dA[i,k] = sum_j (dL/dC[i,j] * dC[i,j]/dA[i,k])
            = sum_j (dL/dC[i,j] * B[k,j])
            = (dL/dC @ B^T)[i,k]
```

### Implementation

```c
typedef struct {
    AGNode *A;
    AGNode *B;
} MatmulCtx;

static void matmul_backward(AGNode *node) {
    MatmulCtx *ctx = (MatmulCtx *)node->ctx;
    AGNode *A = ctx->A, *B = ctx->B;
    Tensor *dC = node->grad;  // dL/dC

    // dL/dA = dL/dC @ B^T
    if (A->requires_grad) {
        Tensor *B_T = tensor_transpose(B->tensor, 0, 1);
        tensor_matmul_blas(A->grad, dC, B_T);   // accumulate
        // Note: tensor_matmul_blas should ADD to grad, not overwrite
        tensor_free(B_T);
    }

    // dL/dB = A^T @ dL/dC
    if (B->requires_grad) {
        Tensor *A_T = tensor_transpose(A->tensor, 0, 1);
        tensor_matmul_blas(B->grad, A_T, dC);
        tensor_free(A_T);
    }
}
```

> **Gradient accumulation**: Use `+=` because a tensor may appear in multiple branches of the graph. `tensor_matmul_blas` should call `cblas_sgemm` with `beta=1.0` (not `beta=0.0`) to accumulate.

---

## 2. Softmax Forward and Backward

### Forward

For vector `x` of length `N`:

```
softmax(x)[i] = exp(x[i] - max(x)) / sum_j exp(x[j] - max(x))
```

Subtracting `max(x)` prevents overflow without changing the output.

```c
void softmax_forward(float *out, const float *x, size_t N) {
    // Find max for numerical stability
    float m = x[0];
    for (size_t i = 1; i < N; i++) if (x[i] > m) m = x[i];

    // Compute exp and sum
    float sum = 0.0f;
    for (size_t i = 0; i < N; i++) {
        out[i] = expf(x[i] - m);
        sum += out[i];
    }

    // Normalize
    for (size_t i = 0; i < N; i++) out[i] /= sum;
}
```

### Backward

Given `p = softmax(x)` and upstream gradient `dL/dp`:

```
dL/dx[i] = sum_j (dL/dp[j] * dp[j]/dx[i])

where:
  dp[i]/dx[i] =  p[i] * (1 - p[i])     (diagonal)
  dp[j]/dx[i] = -p[j] * p[i]           (off-diagonal, j ≠ i)

So:
  dL/dx[i] = p[i] * (dL/dp[i] - sum_j dL/dp[j] * p[j])
           = p[i] * (dL/dp[i] - dot(dL/dp, p))
```

```c
void softmax_backward(float *dx, const float *dp, const float *p, size_t N) {
    // dot = sum_j dp[j] * p[j]
    float dot = 0.0f;
    for (size_t j = 0; j < N; j++) dot += dp[j] * p[j];

    for (size_t i = 0; i < N; i++)
        dx[i] += p[i] * (dp[i] - dot);
}
```

---

## 3. Cross-Entropy Loss

### Forward

For a single example with target class `t` and logits `x`:

```
L = -log(softmax(x)[t])
  = -x[t] + log(sum_j exp(x[j] - max(x))) + max(x)
```

Numerically stable implementation:

```c
float cross_entropy_forward(const float *logits, int target, size_t N) {
    float m = logits[0];
    for (size_t i = 1; i < N; i++) if (logits[i] > m) m = logits[i];

    float sum = 0.0f;
    for (size_t i = 0; i < N; i++) sum += expf(logits[i] - m);

    return -(logits[target] - m) + logf(sum);
}
```

### Fused Softmax-Cross-Entropy Backward

Rather than composing softmax backward and cross-entropy backward separately, the fused form is simpler:

```
dL/dx[i] = softmax(x)[i] - 1{i == t}
```

This is because:
```
L = -log(p[t]) where p = softmax(x)

dL/dx[i] = dL/dp[t] * dp[t]/dx[i]
         = (-1/p[t]) * p[t] * (1{i==t} - p[i])
         = -(1{i==t} - p[i])
         = p[i] - 1{i==t}
```

```c
void softmax_crossentropy_backward(float *dx, const float *logits,
                                   int target, size_t N) {
    // Compute softmax(logits) in-place into dx
    float m = logits[0];
    for (size_t i = 1; i < N; i++) if (logits[i] > m) m = logits[i];
    float sum = 0.0f;
    for (size_t i = 0; i < N; i++) { dx[i] = expf(logits[i] - m); sum += dx[i]; }
    for (size_t i = 0; i < N; i++) dx[i] /= sum;

    // Subtract 1 at target
    dx[target] -= 1.0f;

    // Scale by 1/batch_size if averaging over batch
    // (do this at the training loop level)
}
```

---

## 4. Full MLP: Forward + Backward Test

```c
// Test: 2-layer MLP on XOR problem
// y = softmax(relu(x @ W1 + b1) @ W2 + b2)
// loss = cross_entropy(y, target)

static void test_mlp_gradients(void) {
    // Tiny network: input 2 → hidden 4 → output 2
    float W1_data[] = {0.1f, -0.2f, 0.3f, 0.4f,
                       0.5f,  0.1f,-0.3f, 0.2f};
    float b1_data[] = {0.0f, 0.0f, 0.0f, 0.0f};
    float W2_data[] = {0.3f, -0.1f, 0.4f, 0.2f,
                      -0.2f,  0.5f,-0.1f, 0.3f};
    float b2_data[] = {0.0f, 0.0f};
    float x_data[]  = {1.0f, 0.0f};  // XOR input
    int   target    = 1;             // XOR output

    // Create nodes
    AGNode *x  = ag_leaf(x_data,  2, (size_t[]){1,2}, false);
    AGNode *W1 = ag_leaf(W1_data, 2, (size_t[]){2,4}, true);
    AGNode *b1 = ag_leaf(b1_data, 1, (size_t[]){4},   true);
    AGNode *W2 = ag_leaf(W2_data, 2, (size_t[]){4,2}, true);
    AGNode *b2 = ag_leaf(b2_data, 1, (size_t[]){2},   true);

    // Forward
    AGNode *h1    = ag_add(ag_matmul(x, W1), b1);  // [1,4]
    AGNode *h1_r  = ag_relu(h1);                    // [1,4]
    AGNode *logits = ag_add(ag_matmul(h1_r, W2), b2); // [1,2]
    AGNode *loss   = ag_cross_entropy(logits, target);

    printf("loss = %.6f\n", loss->tensor->data[0]);

    // Backward
    ag_backward(loss);

    // Gradient check W1
    gradient_check_node(loss, W1, 1e-4f, 1e-3f, "W1");
    gradient_check_node(loss, W2, 1e-4f, 1e-3f, "W2");
    gradient_check_node(loss, b1, 1e-4f, 1e-3f, "b1");
    gradient_check_node(loss, b2, 1e-4f, 1e-3f, "b2");
}
```

**Expected output**:
```
loss = 0.693147   (log(2) — random initialization near 50/50)
Gradient check W1: PASSED (eps=1.0e-04, rtol=1.0e-03)
Gradient check W2: PASSED
Gradient check b1: PASSED
Gradient check b2: PASSED
```

---

## 5. Gradient Accumulation vs. Gradient Reset

A critical correctness issue: gradients **accumulate** across backward calls. You must zero them between training steps.

```c
void ag_zero_grad(AGNode *node) {
    if (node == NULL || !node->visited) return;
    node->visited = true;

    if (node->grad) {
        memset(node->grad->data, 0, node->grad->numel * sizeof(float));
    }
    for (int i = 0; i < AUTOGRAD_MAX_INPUTS; i++)
        ag_zero_grad(node->inputs[i]);
}

// Reset visited flags before next zero_grad call
void ag_reset_visited(AGNode *node) {
    if (!node->visited) return;
    node->visited = false;
    for (int i = 0; i < AUTOGRAD_MAX_INPUTS; i++)
        ag_reset_visited(node->inputs[i]);
}
```

---

## 6. Summary of Backward Formulas

| Operation | Forward | dL/d(input) |
|-----------|---------|-------------|
| `C = A @ B` | — | `dL/dA = dL/dC @ B^T`, `dL/dB = A^T @ dL/dC` |
| `y = x + b` | — | `dL/dx = dL/dy`, `dL/db = sum(dL/dy, axis=0)` |
| `y = relu(x)` | `x > 0 ? x : 0` | `dL/dx = dL/dy * (x > 0)` |
| `p = softmax(x)` | `exp(x-max)/sum` | `dL/dx[i] = p[i]*(dL/dp[i] - dot(dL/dp, p))` |
| `L = CE(x, t)` | `-log(p[t])` | `dL/dx[i] = p[i] - 1{i==t}` |
| `y = LayerNorm(x)` | `(x - μ)/σ * γ + β` | Derived in L24 |
| `C = attention(Q,K,V)` | `softmax(QK^T/√d) V` | Derived in L37 |

---

## Key Takeaways

- **Matmul backward**: `dA = dC @ B^T`, `dB = A^T @ dC` — transpose the non-differentiated operand
- **Softmax backward**: `dx[i] = p[i] * (dp[i] - dot(dp, p))` — requires the forward output `p`
- **Fused CE backward**: `dx[i] = softmax(x)[i] - 1{i==t}` — simpler than composing softmax+CE
- Always verify with finite differences; even a sign error or a missing transpose breaks training
- Zero gradients before each backward pass

---

**Next**: [07. Memory Manager](./07_Memory_Manager.md) — Build an arena allocator and reference-counted tensor pool to eliminate `malloc`/`free` overhead during inference.
