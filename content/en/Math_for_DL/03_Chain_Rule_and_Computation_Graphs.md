# Lesson 3: Chain Rule and Computation Graphs

## Learning Objectives

- State and apply the multivariate chain rule for composite functions
- Represent computations as directed acyclic graphs (DAGs) with nodes and edges
- Distinguish between forward-mode and reverse-mode automatic differentiation
- Derive backpropagation as an application of the reverse-mode chain rule
- Implement a simple computation graph and backpropagation from scratch in NumPy
- Compute gradients through common DL operations (linear, ReLU, sigmoid, loss)
- Understand why reverse mode is $O(1)$ in the number of outputs for computing gradients

---

## 1. The Multivariate Chain Rule

### 1.1 Single-Variable Review

For $y = f(g(x))$, the chain rule gives:

$$\frac{dy}{dx} = \frac{dy}{dg} \cdot \frac{dg}{dx} = f'(g(x)) \cdot g'(x)$$

### 1.2 Multivariate Chain Rule

Now consider $f: \mathbb{R}^n \to \mathbb{R}$ composed with $\mathbf{g}: \mathbb{R}^m \to \mathbb{R}^n$. If $L = f(\mathbf{g}(\mathbf{x}))$, then:

$$\frac{\partial L}{\partial x_j} = \sum_{i=1}^{n} \frac{\partial L}{\partial g_i} \frac{\partial g_i}{\partial x_j}$$

In matrix form, using the Jacobian $\mathbf{J}_\mathbf{g} \in \mathbb{R}^{n \times m}$:

$$\nabla_\mathbf{x} L = \mathbf{J}_\mathbf{g}^\top \nabla_\mathbf{g} L$$

**This is the fundamental equation of backpropagation**: multiply the upstream gradient by the transpose of the local Jacobian.

### 1.3 Chain of Compositions

For a chain $\mathbf{x} \xrightarrow{f_1} \mathbf{h}_1 \xrightarrow{f_2} \mathbf{h}_2 \xrightarrow{f_3} L$:

$$\nabla_\mathbf{x} L = \mathbf{J}_{f_1}^\top \mathbf{J}_{f_2}^\top \nabla_{\mathbf{h}_2} L$$

We can evaluate this product either left-to-right (forward mode) or right-to-left (reverse mode).

---

## 2. Computation Graphs

### 2.1 What Is a Computation Graph?

A **computation graph** is a directed acyclic graph (DAG) where:
- **Nodes** represent values (tensors)
- **Edges** represent operations (functions)
- **Leaf nodes** are inputs and parameters
- The **root node** is the output (typically the loss scalar)

### 2.2 Example: A Simple Neural Network

Consider $L = (y - \sigma(\mathbf{w}^\top \mathbf{x} + b))^2$ where $\sigma$ is the sigmoid function.

```
   x  w   b
   │  │   │
   └──┼───┘
      │
   z = w^T x + b    (linear)
      │
   a = σ(z)         (sigmoid)
      │
   y  │
   │  │
   └──┘
   e = y - a         (subtract)
      │
   L = e^2           (square)
```

Each node stores:
1. Its **value** (from the forward pass)
2. Its **gradient** $\frac{\partial L}{\partial \text{node}}$ (from the backward pass)

### 2.3 Implementation: A Minimal Computation Graph

```python
import numpy as np

class Value:
    """A node in a computation graph that supports automatic differentiation."""

    def __init__(self, data, children=(), op=''):
        self.data = data
        self.grad = 0.0
        self._children = set(children)
        self._op = op
        self._backward = lambda: None

    def __repr__(self):
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += out.grad   # dL/da = dL/dout * 1
            other.grad += out.grad  # dL/db = dL/dout * 1
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad  # dL/da = dL/dout * b
            other.grad += self.data * out.grad  # dL/db = dL/dout * a
        out._backward = _backward
        return out

    def sigmoid(self):
        s = 1 / (1 + np.exp(-self.data))
        out = Value(s, (self,), 'sigmoid')

        def _backward():
            self.grad += s * (1 - s) * out.grad
        out._backward = _backward
        return out

    def backward(self):
        """Topological sort then reverse-order backward pass."""
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._children:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        self.grad = 1.0
        for v in reversed(topo):
            v._backward()

# Build computation graph
x_val, w_val, b_val, y_val = 2.0, -3.0, 1.0, 0.0

x = Value(x_val)
w = Value(w_val)
b = Value(b_val)
y = Value(y_val)

# Forward pass
z = x * w + b        # linear: z = wx + b
a = z.sigmoid()       # activation: a = sigmoid(z)
e = y + a * Value(-1) # error: e = y - a
L = e * e             # loss: L = e^2

# Backward pass
L.backward()

print(f"Forward: z={z.data:.4f}, a={a.data:.4f}, L={L.data:.4f}")
print(f"Gradients: dL/dw={w.grad:.4f}, dL/db={b.grad:.4f}, dL/dx={x.grad:.4f}")
```

---

## 3. Forward Mode vs. Reverse Mode

### 3.1 Forward Mode Automatic Differentiation

In forward mode, we propagate derivatives **alongside** the computation:

For each input $x_j$, we track **dual numbers** $(v_i, \dot{v}_i)$ where $\dot{v}_i = \frac{\partial v_i}{\partial x_j}$.

| Step | Value | Derivative $\frac{\partial}{\partial x}$ |
|------|-------|----------------------------------------|
| $z = wx + b$ | $z = (-3)(2) + 1 = -5$ | $\dot{z} = w = -3$ |
| $a = \sigma(z)$ | $a = \sigma(-5) \approx 0.0067$ | $\dot{a} = a(1-a) \cdot \dot{z}$ |
| $e = y - a$ | $e \approx -0.0067$ | $\dot{e} = -\dot{a}$ |
| $L = e^2$ | $L \approx 4.5 \times 10^{-5}$ | $\dot{L} = 2e \cdot \dot{e}$ |

**Cost**: One forward pass per input variable. For $n$ inputs and 1 output: $O(n)$ passes.

### 3.2 Reverse Mode Automatic Differentiation

In reverse mode, we first do a full forward pass, then propagate gradients **backward** from the output:

| Step (reverse) | Adjoint $\bar{v}_i = \frac{\partial L}{\partial v_i}$ |
|----------------|------------------------------------------------------|
| $\bar{L} = 1$ | Seed |
| $\bar{e} = 2e \cdot \bar{L}$ | $\frac{\partial L}{\partial e} = 2e$ |
| $\bar{a} = -\bar{e}$ | $\frac{\partial L}{\partial a} = -\bar{e}$ |
| $\bar{z} = a(1-a) \cdot \bar{a}$ | $\frac{\partial L}{\partial z} = \sigma'(z) \cdot \bar{a}$ |
| $\bar{w} = x \cdot \bar{z}$ | $\frac{\partial L}{\partial w} = x \cdot \bar{z}$ |
| $\bar{x} = w \cdot \bar{z}$ | $\frac{\partial L}{\partial x} = w \cdot \bar{z}$ |
| $\bar{b} = \bar{z}$ | $\frac{\partial L}{\partial b} = \bar{z}$ |

**Cost**: One forward pass + one backward pass. For $n$ inputs and 1 output: $O(1)$ passes.

### 3.3 Why DL Uses Reverse Mode

| Property | Forward Mode | Reverse Mode |
|----------|-------------|-------------|
| Passes needed | One per input | One per output |
| Best for | Few inputs, many outputs | Many inputs, few outputs |
| DL scenario | $n \sim 10^9$ params, 1 loss | **Reverse mode wins** |
| Memory | Low (stream) | High (store forward values) |

Deep learning has one scalar output (the loss) and millions of inputs (parameters), making reverse mode the clear winner.

---

## 4. Backpropagation Through Common Layers

### 4.1 Linear Layer

**Forward**: $\mathbf{z} = \mathbf{W}\mathbf{x} + \mathbf{b}$

**Backward**: Given $\frac{\partial L}{\partial \mathbf{z}}$ (upstream gradient):
- $\frac{\partial L}{\partial \mathbf{W}} = \frac{\partial L}{\partial \mathbf{z}} \mathbf{x}^\top$
- $\frac{\partial L}{\partial \mathbf{x}} = \mathbf{W}^\top \frac{\partial L}{\partial \mathbf{z}}$
- $\frac{\partial L}{\partial \mathbf{b}} = \frac{\partial L}{\partial \mathbf{z}}$

### 4.2 Element-wise Activation

For any element-wise function $\mathbf{a} = \phi(\mathbf{z})$:

$$\frac{\partial L}{\partial z_i} = \frac{\partial L}{\partial a_i} \cdot \phi'(z_i)$$

In vector form: $\frac{\partial L}{\partial \mathbf{z}} = \frac{\partial L}{\partial \mathbf{a}} \odot \phi'(\mathbf{z})$ (Hadamard product).

| Activation | $\phi(z)$ | $\phi'(z)$ |
|-----------|----------|-----------|
| ReLU | $\max(0, z)$ | $\mathbf{1}[z > 0]$ |
| Sigmoid | $\sigma(z)$ | $\sigma(z)(1 - \sigma(z))$ |
| Tanh | $\tanh(z)$ | $1 - \tanh^2(z)$ |

### 4.3 MSE Loss

**Forward**: $L = \frac{1}{n}\|\mathbf{y} - \hat{\mathbf{y}}\|^2$

**Backward**: $\frac{\partial L}{\partial \hat{\mathbf{y}}} = \frac{2}{n}(\hat{\mathbf{y}} - \mathbf{y})$

---

## 5. Full Backpropagation Example

Let's implement a complete forward and backward pass through a two-layer network:

$$\mathbf{z}_1 = \mathbf{W}_1 \mathbf{x} + \mathbf{b}_1$$
$$\mathbf{a}_1 = \text{ReLU}(\mathbf{z}_1)$$
$$\mathbf{z}_2 = \mathbf{W}_2 \mathbf{a}_1 + \mathbf{b}_2$$
$$L = \frac{1}{2}\|\mathbf{z}_2 - \mathbf{y}\|^2$$

```python
import numpy as np

np.random.seed(42)

# Network architecture
n_in, n_hidden, n_out = 3, 4, 2
x = np.random.randn(n_in)
y = np.random.randn(n_out)

# Initialize weights
W1 = np.random.randn(n_hidden, n_in) * 0.5
b1 = np.zeros(n_hidden)
W2 = np.random.randn(n_out, n_hidden) * 0.5
b2 = np.zeros(n_out)

# === Forward Pass ===
z1 = W1 @ x + b1          # (n_hidden,)
a1 = np.maximum(z1, 0)     # ReLU
z2 = W2 @ a1 + b2          # (n_out,)
L = 0.5 * np.sum((z2 - y)**2)  # MSE loss

print(f"Forward pass:")
print(f"  z1 = {z1}")
print(f"  a1 = {a1}")
print(f"  z2 = {z2}")
print(f"  L  = {L:.6f}")

# === Backward Pass ===
# Step 1: dL/dz2
dL_dz2 = z2 - y                          # (n_out,)

# Step 2: dL/dW2, dL/db2, dL/da1
dL_dW2 = np.outer(dL_dz2, a1)            # (n_out, n_hidden)
dL_db2 = dL_dz2                           # (n_out,)
dL_da1 = W2.T @ dL_dz2                    # (n_hidden,)

# Step 3: dL/dz1 (through ReLU)
dL_dz1 = dL_da1 * (z1 > 0).astype(float) # (n_hidden,)

# Step 4: dL/dW1, dL/db1
dL_dW1 = np.outer(dL_dz1, x)             # (n_hidden, n_in)
dL_db1 = dL_dz1                           # (n_hidden,)

print(f"\nBackward pass:")
print(f"  dL/dW2 shape: {dL_dW2.shape}")
print(f"  dL/dW1 shape: {dL_dW1.shape}")

# === Numerical Gradient Check ===
eps = 1e-5
dL_dW1_num = np.zeros_like(W1)
for i in range(W1.shape[0]):
    for j in range(W1.shape[1]):
        W1_plus = W1.copy(); W1_plus[i, j] += eps
        z1_p = W1_plus @ x + b1
        a1_p = np.maximum(z1_p, 0)
        z2_p = W2 @ a1_p + b2
        L_plus = 0.5 * np.sum((z2_p - y)**2)

        W1_minus = W1.copy(); W1_minus[i, j] -= eps
        z1_m = W1_minus @ x + b1
        a1_m = np.maximum(z1_m, 0)
        z2_m = W2 @ a1_m + b2
        L_minus = 0.5 * np.sum((z2_m - y)**2)

        dL_dW1_num[i, j] = (L_plus - L_minus) / (2 * eps)

print(f"\nGradient check dL/dW1:")
print(f"  Analytical:\n{dL_dW1}")
print(f"  Numerical:\n{dL_dW1_num}")
print(f"  Max error: {np.max(np.abs(dL_dW1 - dL_dW1_num)):.2e}")
```

---

## 6. The Jacobian View of Backpropagation

### 6.1 Vector-Jacobian Products (VJPs)

Backpropagation never explicitly forms the full Jacobian matrix. Instead, it computes **vector-Jacobian products** (VJPs):

$$\bar{\mathbf{x}} = \bar{\mathbf{y}}^\top \mathbf{J}$$

where $\bar{\mathbf{y}} = \frac{\partial L}{\partial \mathbf{y}}$ is the upstream gradient and $\mathbf{J} = \frac{\partial \mathbf{y}}{\partial \mathbf{x}}$ is the local Jacobian.

For a linear layer $\mathbf{y} = \mathbf{W}\mathbf{x}$, the Jacobian w.r.t. $\mathbf{x}$ is $\mathbf{J} = \mathbf{W}$, so:

$$\bar{\mathbf{x}} = \bar{\mathbf{y}}^\top \mathbf{W} = \mathbf{W}^\top \bar{\mathbf{y}}$$

This is exactly the formula we derived in Section 4.1.

### 6.2 Jacobian-Vector Products (JVPs)

Forward mode computes **Jacobian-vector products** (JVPs):

$$\dot{\mathbf{y}} = \mathbf{J} \dot{\mathbf{x}}$$

where $\dot{\mathbf{x}}$ is the tangent vector (perturbation direction).

### 6.3 Computational Complexity

For a composition $f_L \circ f_{L-1} \circ \cdots \circ f_1$ with Jacobians $\mathbf{J}_1, \ldots, \mathbf{J}_L$:

**Reverse mode** (right-to-left, VJPs):
$$\bar{\mathbf{x}} = \bar{L} \cdot \mathbf{J}_L \cdot \mathbf{J}_{L-1} \cdots \mathbf{J}_1$$
Each VJP produces a vector, so the entire chain costs $O(L)$ vector-matrix products.

**Forward mode** (left-to-right, JVPs):
$$\dot{L} = \mathbf{J}_L \cdot \mathbf{J}_{L-1} \cdots \mathbf{J}_1 \cdot \dot{\mathbf{x}}$$
Also $O(L)$, but gives the derivative w.r.t. one input direction per pass.

---

## 7. Gradient Flow Pathologies

### 7.1 Vanishing Gradients

When activation derivatives are $< 1$ (sigmoid outputs in $(0, 0.25)$), multiplying many such factors together causes the gradient to shrink exponentially:

$$\|\bar{\mathbf{x}}\| \leq \prod_{l=1}^{L} \|\mathbf{J}_l\| \cdot |\bar{L}|$$

If $\|\mathbf{J}_l\| < 1$ for all $l$, the gradient vanishes exponentially in depth $L$.

### 7.2 Exploding Gradients

Conversely, if $\|\mathbf{J}_l\| > 1$ for all $l$, gradients grow exponentially.

### 7.3 Mitigations

| Problem | Solution | Why It Works |
|---------|----------|-------------|
| Vanishing | ReLU activation | $\text{ReLU}'(z) = 1$ for $z > 0$ (no shrinkage) |
| Vanishing | Residual connections | Gradient has a direct path: $\frac{\partial}{\partial \mathbf{x}}(\mathbf{x} + f(\mathbf{x})) = \mathbf{I} + \mathbf{J}_f$ |
| Exploding | Gradient clipping | Cap $\|\nabla L\|$ at a threshold |
| Both | Batch normalization | Normalizes pre-activations, stabilizes Jacobian norms |
| Both | Proper initialization | Xavier/He init sets $\text{Var}(\text{output}) \approx \text{Var}(\text{input})$ |

```python
# Demonstrate gradient vanishing with sigmoid vs ReLU
np.random.seed(0)
depth = 20
dim = 50

def simulate_gradient_flow(activation='relu'):
    """Simulate gradient magnitude through a deep network."""
    grad = np.ones(dim)
    norms = [np.linalg.norm(grad)]

    for _ in range(depth):
        W = np.random.randn(dim, dim) * np.sqrt(2.0 / dim)  # He init
        z = np.random.randn(dim)

        if activation == 'relu':
            mask = (z > 0).astype(float)
            J_diag = mask
        elif activation == 'sigmoid':
            sig = 1 / (1 + np.exp(-z))
            J_diag = sig * (1 - sig)

        # VJP: grad = grad @ diag(J_diag) @ W.T = (W @ (J_diag * grad))
        grad = W.T @ (J_diag * grad)
        norms.append(np.linalg.norm(grad))

    return norms

import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(8, 4))
ax.semilogy(simulate_gradient_flow('relu'), 'b-o', markersize=3, label='ReLU')
ax.semilogy(simulate_gradient_flow('sigmoid'), 'r-s', markersize=3, label='Sigmoid')
ax.set_xlabel('Layer (from output)')
ax.set_ylabel('||gradient||')
ax.set_title('Gradient magnitude through layers')
ax.legend()
ax.grid(True, alpha=0.3)
plt.show()
```

---

## Summary

| Concept | Key Takeaway |
|---------|-------------|
| Multivariate chain rule | $\nabla_\mathbf{x} L = \mathbf{J}^\top \nabla_\mathbf{y} L$ |
| Computation graph | DAG of operations; forward stores values, backward propagates gradients |
| Forward mode | One pass per input; $O(n)$ for $n$ parameters |
| Reverse mode | One pass per output; $O(1)$ for scalar loss -- this is backpropagation |
| VJP | $\bar{\mathbf{x}} = \bar{\mathbf{y}}^\top \mathbf{J}$: how backprop avoids forming full Jacobians |
| Gradient pathology | Vanishing/exploding gradients from repeated Jacobian multiplication |

---

## Exercises

1. Extend the `Value` class to support `__sub__`, `__pow__`, and `tanh` with correct backward methods.
2. Implement backpropagation for a 3-layer network (2 hidden layers) and verify with finite differences.
3. Compute the gradient of the cross-entropy loss $L = -\sum y_i \log \hat{y}_i$ w.r.t. logits (before softmax).
4. Compare gradient norms through a 50-layer network with sigmoid vs. ReLU vs. tanh activations.
5. Implement a residual block $\mathbf{y} = \mathbf{x} + f(\mathbf{x})$ and show that it alleviates vanishing gradients.

---

**Next**: [04. Jacobian and Hessian](04_Jacobian_and_Hessian.md)
