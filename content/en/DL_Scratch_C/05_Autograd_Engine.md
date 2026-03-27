# 05. Autograd Engine

**Previous**: [Optimized Matmul](./04_Optimized_Matmul.md) | **Next**: [Autograd Tensor Ops](./06_Autograd_Tensor_Ops.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain what a computational graph is and why it enables automatic differentiation
2. Implement a dynamic computation graph in C using linked lists and function pointers
3. Perform a topological sort on the graph in C
4. Call `backward()` to propagate gradients through the graph
5. Verify gradients using the finite difference method

---

## 1. What is Automatic Differentiation?

Given a function `f(x)` computed as a sequence of operations, **autograd** computes `df/dx` without the programmer writing the derivative manually.

There are two modes:
- **Forward mode**: propagate derivatives forward (efficient for `f: R → R^n`)
- **Reverse mode (backpropagation)**: propagate from output to inputs (efficient for `f: R^n → R`, the typical DL case)

We implement **reverse-mode AD**.

### The Chain Rule

If `z = f(y)` and `y = g(x)`, then:
```
dL/dx = dL/dz * dz/dy * dy/dx
```

For a full computation graph `L = f_n(f_{n-1}(...f_1(x)...))`:
```
dL/dx = dL/df_n * df_n/df_{n-1} * ... * df_1/dx
```

Autograd evaluates this right-to-left: starting from `dL/dL = 1`, propagate backward through each operation.

---

## 2. The Computational Graph

Each tensor in the graph is a **node**. Operations create edges between nodes.

```
Forward pass:
  x → [matmul] → y → [relu] → z → [sum] → L

Backward pass (reverse):
  L → dL/dz=1 → [relu_backward] → dL/dy → [matmul_backward] → dL/dx
```

### Node Structure

```c
// autograd.h
#pragma once
#include "tensor.h"
#include <stddef.h>

#define AUTOGRAD_MAX_INPUTS 4

typedef struct AGNode {
    Tensor *tensor;          // The output tensor of this operation
    Tensor *grad;            // Gradient w.r.t. this node (same shape as tensor)

    // Operation that produced this node
    void (*backward_fn)(struct AGNode *node);
    void *ctx;               // Saved context (inputs, other data) for backward

    // Inputs that this node depends on
    struct AGNode *inputs[AUTOGRAD_MAX_INPUTS];
    int n_inputs;

    // Topological sort bookkeeping
    bool visited;
    bool requires_grad;
} AGNode;

// Allocate a node wrapping a tensor
AGNode *ag_node_new(Tensor *tensor, bool requires_grad);
void    ag_node_free(AGNode *node);

// Core autograd operations
void    ag_backward(AGNode *root);          // Run full backward pass
void    ag_zero_grad(AGNode *node);         // Zero gradients recursively
```

---

## 3. Topological Sort

`backward()` must visit nodes in **reverse topological order**: each node's gradient must be computed before it is propagated to its inputs.

```c
// autograd.c
#include "autograd.h"
#include <stdlib.h>
#include <string.h>

// ── Topological sort ─────────────────────────────────────────────────────

typedef struct {
    AGNode **nodes;
    int      count;
    int      capacity;
} NodeList;

static void nodelist_push(NodeList *list, AGNode *node) {
    if (list->count >= list->capacity) {
        list->capacity = list->capacity * 2 + 8;
        list->nodes = realloc(list->nodes, list->capacity * sizeof(AGNode *));
    }
    list->nodes[list->count++] = node;
}

// DFS post-order traversal → topological order (reversed)
static void topo_dfs(AGNode *node, NodeList *order) {
    if (node == NULL || node->visited) return;
    node->visited = true;

    for (int i = 0; i < node->inputs[i] != NULL && i < AUTOGRAD_MAX_INPUTS; i++)
        topo_dfs(node->inputs[i], order);

    nodelist_push(order, node);  // Post-order: node after its inputs
}

static NodeList build_topo(AGNode *root) {
    NodeList order = {NULL, 0, 0};
    topo_dfs(root, &order);
    return order;
}
```

---

## 4. The Backward Pass

```c
void ag_backward(AGNode *root) {
    // Initialize root gradient = 1.0 (scalar loss)
    if (root->grad == NULL) {
        root->grad = tensor_zeros(root->tensor->ndim, root->tensor->shape);
        for (size_t i = 0; i < root->grad->numel; i++)
            root->grad->data[i] = 1.0f;
    }

    // Build topological order
    NodeList order = build_topo(root);

    // Visit in REVERSE topological order (root → leaves)
    for (int i = order.count - 1; i >= 0; i--) {
        AGNode *node = order.nodes[i];
        if (node->backward_fn != NULL && node->requires_grad) {
            // Ensure all input nodes have grad buffers
            for (int j = 0; j < AUTOGRAD_MAX_INPUTS; j++) {
                if (node->inputs[j] && node->inputs[j]->requires_grad &&
                    node->inputs[j]->grad == NULL) {
                    node->inputs[j]->grad = tensor_zeros(
                        node->inputs[j]->tensor->ndim,
                        node->inputs[j]->tensor->shape);
                }
            }
            node->backward_fn(node);
        }
    }

    free(order.nodes);
}
```

---

## 5. Example: Scalar Autograd

Let us implement `add` and `mul` for scalar values to verify the engine:

```c
// Saved context for add backward
typedef struct { AGNode *a; AGNode *b; } AddCtx;

// add forward: out = a + b
AGNode *ag_add(AGNode *a, AGNode *b) {
    assert(a->tensor->numel == b->tensor->numel);

    // Forward computation
    Tensor *out_t = tensor_zeros(a->tensor->ndim, a->tensor->shape);
    for (size_t i = 0; i < a->tensor->numel; i++)
        out_t->data[i] = a->tensor->data[i] + b->tensor->data[i];

    AGNode *out   = ag_node_new(out_t, a->requires_grad || b->requires_grad);
    out->inputs[0] = a;
    out->inputs[1] = b;
    out->n_inputs  = 2;

    // Save context for backward
    AddCtx *ctx = malloc(sizeof(AddCtx));
    ctx->a = a; ctx->b = b;
    out->ctx = ctx;

    // Backward: d(a+b)/da = 1,  d(a+b)/db = 1
    out->backward_fn = [](AGNode *node) {
        AddCtx *ctx = (AddCtx *)node->ctx;
        // dL/da += dL/d(out) * 1
        if (ctx->a->requires_grad)
            for (size_t i = 0; i < node->grad->numel; i++)
                ctx->a->grad->data[i] += node->grad->data[i];
        // dL/db += dL/d(out) * 1
        if (ctx->b->requires_grad)
            for (size_t i = 0; i < node->grad->numel; i++)
                ctx->b->grad->data[i] += node->grad->data[i];
    };
    return out;
}
```

> **Note**: C does not have lambda expressions. In practice, define backward functions as named `static` functions and assign their function pointer. The lambda syntax above is for clarity; actual implementation uses named functions.

### Clean C version:

```c
static void add_backward(AGNode *node) {
    AddCtx *ctx = (AddCtx *)node->ctx;
    for (size_t i = 0; i < node->grad->numel; i++) {
        if (ctx->a->requires_grad)
            ctx->a->grad->data[i] += node->grad->data[i];
        if (ctx->b->requires_grad)
            ctx->b->grad->data[i] += node->grad->data[i];
    }
}

AGNode *ag_add(AGNode *a, AGNode *b) {
    // ... (forward computation) ...
    out->backward_fn = add_backward;  // Named function pointer
    return out;
}
```

---

## 6. Finite Difference Verification

**Always verify gradients numerically** before trusting an autograd implementation.

```c
// Check: df/dx[i] ≈ (f(x + ε*e_i) - f(x - ε*e_i)) / (2ε)
void gradient_check(AGNode *(*forward_fn)(AGNode *), AGNode *x,
                    float eps, float rtol) {
    // Run forward + backward
    AGNode *out = forward_fn(x);
    ag_backward(out);
    float *analytic = x->grad->data;

    // Compute numerical gradient
    float *numeric = calloc(x->tensor->numel, sizeof(float));
    for (size_t i = 0; i < x->tensor->numel; i++) {
        float orig = x->tensor->data[i];

        x->tensor->data[i] = orig + eps;
        AGNode *out_plus = forward_fn(x);
        float f_plus = out_plus->tensor->data[0];  // Scalar output assumed

        x->tensor->data[i] = orig - eps;
        AGNode *out_minus = forward_fn(x);
        float f_minus = out_minus->tensor->data[0];

        numeric[i] = (f_plus - f_minus) / (2.0f * eps);
        x->tensor->data[i] = orig;  // Restore
    }

    // Compare
    bool passed = true;
    for (size_t i = 0; i < x->tensor->numel; i++) {
        float rel_err = fabsf(analytic[i] - numeric[i]) /
                        (fabsf(numeric[i]) + 1e-8f);
        if (rel_err > rtol) {
            printf("FAIL at index %zu: analytic=%.6f numeric=%.6f err=%.4f\n",
                   i, analytic[i], numeric[i], rel_err);
            passed = false;
        }
    }
    printf("Gradient check: %s (eps=%.1e, rtol=%.1e)\n",
           passed ? "PASSED" : "FAILED", eps, rtol);
    free(numeric);
}
```

---

## 7. Full Example: MLP Forward + Backward

```c
int main(void) {
    // Input x [2, 3], weights W [3, 2]
    size_t x_shape[] = {2, 3};
    size_t W_shape[] = {3, 2};
    Tensor *x_t = tensor_zeros(2, x_shape);
    Tensor *W_t = tensor_zeros(2, W_shape);

    // Fill with test values
    float x_vals[] = {1,2,3, 4,5,6};
    float W_vals[] = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f};
    memcpy(x_t->data, x_vals, sizeof(x_vals));
    memcpy(W_t->data, W_vals, sizeof(W_vals));

    AGNode *x = ag_node_new(x_t, true);   // requires_grad = true
    AGNode *W = ag_node_new(W_t, true);

    // Forward: h = x @ W  → [2, 2]
    AGNode *h = ag_matmul(x, W);

    // Forward: loss = sum(h)
    AGNode *loss = ag_sum(h);

    printf("loss = %.4f\n", loss->tensor->data[0]);

    // Backward
    ag_backward(loss);

    printf("dL/dW:\n");
    tensor_print(W->grad, "W.grad");
    // Expected: W.grad[i][j] = sum over rows of x (since d/dW[sum(xW)] = x^T)

    return 0;
}
```

---

## Key Takeaways

- The computational graph records *which operations produced which tensors* during the forward pass
- `backward_fn` is a function pointer stored in each node; it accumulates `grad` into the node's inputs
- Topological sort ensures each node's gradient is fully accumulated before propagating to its inputs
- **Gradient accumulation** (`+=`, not `=`): multiple paths through the graph must sum their contributions
- Always run gradient checks before trusting a new backward implementation

---

**Next**: [06. Autograd Tensor Ops](./06_Autograd_Tensor_Ops.md) — Implement backward passes for matmul, softmax, and cross-entropy, and verify with finite differences.
