# Lesson 5 — Autograd Engine (per-lesson exercise)

Prerequisites: L02 (memory layout), basic calculus chain rule.

Compile: `gcc -std=c11 -Wall -Wextra -O2 -o ex ex.c -lm`

Reverse-mode automatic differentiation (autograd) is the algorithm behind `loss.backward()` in every deep learning framework. The idea: build a graph of operations during the forward pass; in the backward pass, walk the graph in reverse, applying the chain rule at each node.

Implementing 100 lines of autograd in C demystifies the whole field.

---

## Exercise 5.1 — Scalar Tape

**Difficulty**: ★★★

### Problem

Define a `Value` struct that represents a scalar plus its gradient and the operation that produced it:

```c
typedef enum { OP_LEAF, OP_ADD, OP_MUL, OP_TANH } Op;

typedef struct Value {
    float data;
    float grad;
    Op    op;
    struct Value *parents[2];
} Value;
```

Implement four constructors:

```c
Value *value_leaf(float x);
Value *value_add(Value *a, Value *b);   /* result = a + b */
Value *value_mul(Value *a, Value *b);   /* result = a * b */
Value *value_tanh(Value *a);
```

Each non-leaf constructor allocates a new `Value`, sets its `op` and `parents`, and computes its `data` from the parents.

---

## Exercise 5.2 — Backward Pass

**Difficulty**: ★★★

For a single output `loss`, set `loss.grad = 1` and walk the graph in reverse topological order, applying:

| Op | Gradient rule |
|----|---------------|
| ADD | `parents[0].grad += result.grad`; `parents[1].grad += result.grad` |
| MUL | `parents[0].grad += result.grad * parents[1].data`; `parents[1].grad += result.grad * parents[0].data` |
| TANH | `parents[0].grad += result.grad * (1 - result.data^2)` |

### Starter

```c
#include <math.h>
#include <stdlib.h>

void backward(Value *root) {
    /* Naive: assume root is the loss; perform DFS-postorder; accumulate grads */
    if (!root) return;
    switch (root->op) {
        case OP_LEAF:
            return;
        case OP_ADD:
            root->parents[0]->grad += root->grad;
            root->parents[1]->grad += root->grad;
            break;
        case OP_MUL:
            root->parents[0]->grad += root->grad * root->parents[1]->data;
            root->parents[1]->grad += root->grad * root->parents[0]->data;
            break;
        case OP_TANH:
            root->parents[0]->grad += root->grad * (1.0f - root->data * root->data);
            break;
    }
    backward(root->parents[0]);
    if (root->op == OP_ADD || root->op == OP_MUL) backward(root->parents[1]);
}
```

**Caveat**: this naive recursion visits a node's parents through every path — DAGs with shared subgraphs get exponential blowup. The fix is a topological-order traversal that visits each node ONCE; do that for any non-toy graph.

---

## Exercise 5.3 — Tiny Neural Net Trained with the Tape

**Difficulty**: ★★★★

Build a 2-input, 4-hidden, 1-output MLP with `tanh` activations. Forward pass constructs the graph; backward computes per-weight gradients; SGD updates weights.

Train on XOR:

```
inputs = [(0,0,0), (0,1,1), (1,0,1), (1,1,0)]
```

After 1000 SGD steps with `lr = 0.1`, all four predictions should be within 0.05 of the target. If not, the most likely cause is that you re-used `Value` nodes across forward passes — each forward pass needs a fresh graph, otherwise old gradients accumulate.

---

## Exercise 5.4 — Tensor Tape Sketch — Bonus

**Difficulty**: ★★★★

Generalize `Value` to hold a tensor (pointer + shape + strides from L02). Each op now produces a tensor, and the gradient is also a tensor of the same shape. The chain rule for tensor ops is more involved (e.g., `matmul(A, B).grad → A.grad += grad @ B^T; B.grad += A^T @ grad`).

Implement just `OP_MATMUL` and verify on a tiny 2×3 × 3×2 example by finite differences.
