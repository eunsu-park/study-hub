# 01. Micro Autograd

[Next: Tiny GAN](./02_Tiny_GAN.md)

---

> **Related Topics**: Deep_Learning, Math_for_AI, Calculus_and_Differential_Equations
>
> **Implementation**: `micro_autograd.py` (~300 lines, NumPy only)

## Learning Objectives

- Understand how computation graphs represent mathematical expressions as directed acyclic graphs (DAGs)
- Implement reverse-mode automatic differentiation (backpropagation) from first principles
- Apply the chain rule systematically to propagate gradients through composite functions
- Build a minimal `Value` class that supports arithmetic operations with gradient tracking
- Construct and train a simple multi-layer neural network using the custom autograd engine

---

## 1. Theory: Computation Graphs and Automatic Differentiation

### 1.1 Computation Graphs

Every mathematical expression can be decomposed into a graph of elementary operations. Consider `L = (a * b + c) ** 2`. This becomes:

```
a ──┐
    ├── (*) ── d ──┐
b ──┘              ├── (+) ── e ── (**2) ── L
               c ──┘
```

Each node stores a value and, during backpropagation, accumulates a gradient. The edges encode which operation produced which intermediate result.

### 1.2 Reverse-Mode Autodiff

Forward-mode differentiation computes `dout/din` for one input at a time — expensive when you have millions of parameters. Reverse-mode starts from the output and propagates gradients backward through the graph in a single pass, computing `dL/dx` for *every* node `x` simultaneously.

The algorithm:

1. **Forward pass**: Evaluate the expression, building the graph.
2. **Topological sort**: Order nodes so that every node appears before its consumers.
3. **Backward pass**: Walk the sorted list in reverse. At each node, apply the local gradient rule and accumulate into children.

### 1.3 The Chain Rule

For a composition `L = f(g(x))`, the chain rule gives:

```
dL/dx = (dL/df) * (df/dg) * (dg/dx)
```

In code, each operation stores a `_backward` function that multiplies the incoming gradient (`self.grad`) by the local derivative and adds it to each child's `.grad`.

---

## 2. Implementation Walkthrough

### 2.1 The `Value` Class

The core abstraction is a `Value` object that wraps a scalar float and tracks the graph:

```python
class Value:
    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
```

Key fields:
- `data` — the forward-pass numerical value
- `grad` — accumulated gradient (initialized to 0.0, set during backward)
- `_backward` — closure that implements the local chain-rule step
- `_prev` — set of parent `Value` nodes (children in the graph)

### 2.2 Operator Overloading

Each arithmetic operation returns a new `Value` and attaches a `_backward` closure:

```python
def __add__(self, other):
    other = other if isinstance(other, Value) else Value(other)
    out = Value(self.data + other.data, (self, other), '+')

    def _backward():
        self.grad += out.grad       # d(a+b)/da = 1
        other.grad += out.grad      # d(a+b)/db = 1
    out._backward = _backward
    return out
```

For multiplication:

```python
def __mul__(self, other):
    other = other if isinstance(other, Value) else Value(other)
    out = Value(self.data * other.data, (self, other), '*')

    def _backward():
        self.grad += other.data * out.grad   # d(a*b)/da = b
        other.grad += self.data * out.grad   # d(a*b)/db = a
    out._backward = _backward
    return out
```

The pattern repeats for `__pow__`, `__neg__`, `__sub__`, `__truediv__`, `tanh()`, `relu()`, and `exp()`.

### 2.3 Topological Sort and Backward

The `backward()` method on `Value` performs a full reverse pass:

```python
def backward(self):
    topo = []
    visited = set()
    def build_topo(v):
        if v not in visited:
            visited.add(v)
            for child in v._prev:
                build_topo(child)
            topo.append(v)
    build_topo(self)

    self.grad = 1.0
    for v in reversed(topo):
        v._backward()
```

The seed gradient `self.grad = 1.0` starts the chain: "the derivative of L with respect to L is 1."

### 2.4 Neural Network Primitives

On top of `Value`, the implementation builds three abstractions:

| Class | Purpose |
|-------|---------|
| `Neuron` | Single neuron: `w * x + b` followed by an optional activation |
| `Layer` | Collection of neurons with the same input dimension |
| `MLP` | Stack of layers with configurable sizes |

```python
class Neuron:
    def __init__(self, nin, nonlin=True):
        self.w = [Value(np.random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(0.0)
        self.nonlin = nonlin

    def __call__(self, x):
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return act.tanh() if self.nonlin else act
```

The `MLP` class chains layers and exposes a `parameters()` method that collects every `Value` used as a weight or bias — the list you iterate over during gradient descent.

---

## 3. Key Design Decisions

1. **Scalar-level granularity**: Operating on individual floats (not tensors) keeps the implementation trivially simple. Every gradient rule is just high-school calculus.
2. **Gradient accumulation with `+=`**: A node that is used in multiple downstream operations receives gradient contributions from each. Using `+=` (not `=`) is essential for correctness.
3. **Closure-based backward**: Each op captures `self`, `other`, and `out` in a closure. This avoids a class hierarchy for operations and keeps the code compact.
4. **Zero-grad before each pass**: Gradients must be reset to 0.0 before each backward call, otherwise they accumulate across training steps.

---

## 4. Training Example

The `main()` block demonstrates training a small MLP on a toy dataset:

```python
model = MLP(3, [4, 4, 1])     # 3 inputs -> 4 -> 4 -> 1 output
for epoch in range(100):
    # Forward
    preds = [model(x) for x in xs]
    loss = sum((p - y) ** 2 for p, y in zip(preds, ys))

    # Backward
    for p in model.parameters():
        p.grad = 0.0
    loss.backward()

    # Update
    for p in model.parameters():
        p.data -= 0.05 * p.grad
```

This is the complete training loop: forward, zero-grad, backward, update. No optimizer class, no loss function library — just the raw algorithm.

---

## Exercises

1. **Add `sigmoid` activation**: Implement `def sigmoid(self)` on `Value` using the formula `1 / (1 + exp(-x))`. Verify gradients numerically by comparing with finite differences: `(f(x+h) - f(x-h)) / 2h`.

2. **Implement `__matmul__`**: Add a dot-product operation between two lists of `Value` objects. Use it to simplify the `Neuron.__call__` method.

3. **Numerical gradient checking**: Write a function that computes `dL/dw` numerically for every parameter and compares with the analytical gradient from `backward()`. Print the maximum absolute difference. It should be less than `1e-5`.

4. **L2 regularization**: Modify the training loop to add `lambda * sum(p.data**2 for p in model.parameters())` to the loss. Observe how it affects final weights.

5. **Visualize the graph**: Write a function that outputs the computation graph in DOT format (Graphviz). Each node shows its value and gradient; each edge shows the operation.

---

## References

- Karpathy, A. (2022). *micrograd* — A tiny autograd engine. [GitHub](https://github.com/karpathy/micrograd)
- Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986). "Learning representations by back-propagating errors." *Nature*, 323(6088), 533-536.
- Baydin, A. G., Pearlmutter, B. A., Radul, A. A., & Siskind, J. M. (2018). "Automatic Differentiation in Machine Learning: a Survey." *JMLR*, 18(153), 1-43.
- Griewank, A., & Walther, A. (2008). *Evaluating Derivatives: Principles and Techniques of Algorithmic Differentiation*. SIAM.
