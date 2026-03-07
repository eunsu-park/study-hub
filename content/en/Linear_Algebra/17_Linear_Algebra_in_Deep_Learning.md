# Lesson 17: Linear Algebra in Deep Learning

[Previous: Lesson 16](./16_Linear_Algebra_in_Machine_Learning.md) | [Overview](./00_Overview.md) | [Next: Lesson 18](./18_Linear_Algebra_in_Computer_Graphics.md)

---

## Learning Objectives

- Understand backpropagation as the chain rule applied to matrix computations
- Express forward and backward passes of neural network layers as matrix operations
- Explain how batch matrix operations enable efficient GPU computation
- Derive the attention mechanism in Transformers as a sequence of matrix operations
- Analyze weight initialization strategies through the lens of eigenvalue distributions
- Connect GPU acceleration to the parallelism inherent in linear algebra

---

## 1. Neural Networks as Matrix Computations

### 1.1 A Single Layer

A fully connected (dense) layer computes:

$$\mathbf{h} = \sigma(W\mathbf{x} + \mathbf{b})$$

where $W \in \mathbb{R}^{m \times n}$ is the weight matrix, $\mathbf{b} \in \mathbb{R}^m$ is the bias, and $\sigma$ is a nonlinear activation function applied element-wise.

Without the activation function, stacking layers would be pointless: $W_2(W_1 \mathbf{x}) = (W_2 W_1) \mathbf{x}$, which is just another linear map. The nonlinearity is what gives depth its power.

```python
import numpy as np

# Forward pass of a single dense layer
def dense_forward(x, W, b):
    """Forward pass: z = Wx + b."""
    z = W @ x + b
    return z

def relu(z):
    return np.maximum(0, z)

def relu_grad(z):
    return (z > 0).astype(float)

# Example: input dim=4, output dim=3
np.random.seed(42)
x = np.array([1.0, 2.0, 3.0, 4.0])
W = np.random.randn(3, 4) * 0.5
b = np.zeros(3)

z = dense_forward(x, W, b)
h = relu(z)
print(f"Input: {x}")
print(f"Pre-activation z = Wx + b: {z}")
print(f"Output h = ReLU(z): {h}")
```

### 1.2 Batched Computation

In practice, inputs come in batches. For a batch $X \in \mathbb{R}^{B \times n}$, the layer computes:

$$H = \sigma(XW^T + \mathbf{1}_B \mathbf{b}^T)$$

where each row of $X$ is one sample. The entire batch is processed with a single matrix multiplication.

```python
# Batched forward pass
batch_size = 32
n_in, n_out = 128, 64

X = np.random.randn(batch_size, n_in)
W = np.random.randn(n_out, n_in) * np.sqrt(2.0 / n_in)  # He initialization
b = np.zeros(n_out)

# Forward: Z = X @ W^T + b (broadcasting handles the bias)
Z = X @ W.T + b
H = relu(Z)

print(f"Input batch: {X.shape}")
print(f"Weight matrix: {W.shape}")
print(f"Output batch: {H.shape}")
print(f"FLOPs for matrix multiply: {2 * batch_size * n_in * n_out:,}")
```

---

## 2. Backpropagation as the Matrix Chain Rule

### 2.1 Scalar Chain Rule to Matrix Chain Rule

For a scalar composition $f(g(x))$, the chain rule gives $\frac{df}{dx} = \frac{df}{dg} \cdot \frac{dg}{dx}$.

For matrices, if the loss $L$ depends on $Z = XW^T + b$:

$$\frac{\partial L}{\partial W} = \left(\frac{\partial L}{\partial Z}\right)^T X, \quad \frac{\partial L}{\partial X} = \frac{\partial L}{\partial Z} \cdot W$$

```python
def dense_backward(dL_dZ, x, W):
    """Backward pass for z = Wx + b.

    Args:
        dL_dZ: gradient of loss w.r.t. z (upstream gradient), shape (m,)
        x: input vector, shape (n,)
        W: weight matrix, shape (m, n)

    Returns:
        dL_dW: gradient w.r.t. W, shape (m, n)
        dL_db: gradient w.r.t. b, shape (m,)
        dL_dx: gradient w.r.t. x, shape (n,)
    """
    dL_dW = np.outer(dL_dZ, x)     # (m, n)
    dL_db = dL_dZ                    # (m,)
    dL_dx = W.T @ dL_dZ             # (n,)
    return dL_dW, dL_db, dL_dx

# Example: two-layer network forward and backward
np.random.seed(42)
n0, n1, n2 = 4, 3, 2

x = np.random.randn(n0)
W1 = np.random.randn(n1, n0) * 0.5
b1 = np.zeros(n1)
W2 = np.random.randn(n2, n1) * 0.5
b2 = np.zeros(n2)
target = np.array([1.0, 0.0])

# Forward pass
z1 = W1 @ x + b1
h1 = relu(z1)
z2 = W2 @ h1 + b2
y = z2  # No activation on output (regression)

# Loss: MSE
loss = 0.5 * np.sum((y - target)**2)
print(f"Loss: {loss:.4f}")

# Backward pass
dL_dy = y - target                          # (n2,)
dL_dz2 = dL_dy                             # No activation gradient
dL_dW2, dL_db2, dL_dh1 = dense_backward(dL_dz2, h1, W2)

dL_dz1 = dL_dh1 * relu_grad(z1)            # Element-wise (ReLU gradient)
dL_dW1, dL_db1, dL_dx = dense_backward(dL_dz1, x, W1)

print(f"dL/dW1 shape: {dL_dW1.shape}")
print(f"dL/dW2 shape: {dL_dW2.shape}")
```

### 2.2 Batched Backpropagation

```python
def dense_forward_batch(X, W, b):
    """Batched forward: Z = X @ W^T + b."""
    return X @ W.T + b

def dense_backward_batch(dL_dZ, X, W):
    """Batched backward for Z = X @ W^T + b.

    Args:
        dL_dZ: (batch_size, m) upstream gradient
        X: (batch_size, n) input
        W: (m, n) weight matrix

    Returns:
        dL_dW: (m, n)
        dL_db: (m,)
        dL_dX: (batch_size, n)
    """
    batch_size = X.shape[0]
    dL_dW = dL_dZ.T @ X / batch_size     # (m, n)
    dL_db = dL_dZ.mean(axis=0)           # (m,)
    dL_dX = dL_dZ @ W                    # (batch_size, n)
    return dL_dW, dL_db, dL_dX

# Verify with numerical gradient
np.random.seed(42)
B, n_in, n_out = 16, 5, 3
X = np.random.randn(B, n_in)
W = np.random.randn(n_out, n_in) * 0.5
b = np.zeros(n_out)
target = np.random.randn(B, n_out)

Z = dense_forward_batch(X, W, b)
loss = 0.5 * np.mean(np.sum((Z - target)**2, axis=1))
dL_dZ = (Z - target) / B

dL_dW, dL_db, dL_dX = dense_backward_batch(dL_dZ, X, W)

# Numerical gradient check
eps = 1e-5
dW_numerical = np.zeros_like(W)
for i in range(W.shape[0]):
    for j in range(W.shape[1]):
        W_plus = W.copy(); W_plus[i, j] += eps
        W_minus = W.copy(); W_minus[i, j] -= eps
        loss_plus = 0.5 * np.mean(np.sum((X @ W_plus.T + b - target)**2, axis=1))
        loss_minus = 0.5 * np.mean(np.sum((X @ W_minus.T + b - target)**2, axis=1))
        dW_numerical[i, j] = (loss_plus - loss_minus) / (2 * eps)

print(f"Gradient check (max diff): {np.max(np.abs(dL_dW - dW_numerical)):.6e}")
```

### 2.3 Computational Graph as Matrix Chain

A deep network's forward pass is a chain of matrix multiplications interleaved with nonlinearities. Backpropagation computes the product of Jacobian matrices in reverse order.

For a network $y = f_L \circ f_{L-1} \circ \cdots \circ f_1(x)$:

$$\frac{\partial L}{\partial \theta_k} = \frac{\partial L}{\partial y} \cdot \frac{\partial f_L}{\partial f_{L-1}} \cdots \frac{\partial f_{k+1}}{\partial f_k} \cdot \frac{\partial f_k}{\partial \theta_k}$$

```python
# Full multi-layer network (forward + backward)
class SimpleNetwork:
    def __init__(self, layer_sizes):
        self.weights = []
        self.biases = []
        for i in range(len(layer_sizes) - 1):
            # He initialization
            W = np.random.randn(layer_sizes[i+1], layer_sizes[i]) * \
                np.sqrt(2.0 / layer_sizes[i])
            b = np.zeros(layer_sizes[i+1])
            self.weights.append(W)
            self.biases.append(b)

    def forward(self, X):
        """Forward pass, storing intermediate values for backprop."""
        self.activations = [X]
        self.pre_activations = []
        H = X
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            Z = H @ W.T + b
            self.pre_activations.append(Z)
            if i < len(self.weights) - 1:
                H = relu(Z)
            else:
                H = Z  # No activation on last layer
            self.activations.append(H)
        return H

    def backward(self, dL_dout):
        """Backward pass, returning gradients for all parameters."""
        grads_W = []
        grads_b = []
        dL_dH = dL_dout
        B = dL_dH.shape[0]

        for i in reversed(range(len(self.weights))):
            if i < len(self.weights) - 1:
                dL_dZ = dL_dH * relu_grad(self.pre_activations[i])
            else:
                dL_dZ = dL_dH

            dL_dW = dL_dZ.T @ self.activations[i] / B
            dL_db = dL_dZ.mean(axis=0)
            dL_dH = dL_dZ @ self.weights[i]

            grads_W.insert(0, dL_dW)
            grads_b.insert(0, dL_db)

        return grads_W, grads_b

# Test
net = SimpleNetwork([10, 64, 32, 1])
X_test = np.random.randn(32, 10)
y_test = np.random.randn(32, 1)

y_pred = net.forward(X_test)
loss = 0.5 * np.mean((y_pred - y_test)**2)
dL_dy = (y_pred - y_test) / 32

grads_W, grads_b = net.backward(dL_dy)
for i, gW in enumerate(grads_W):
    print(f"Layer {i}: W shape {net.weights[i].shape}, grad shape {gW.shape}")
```

---

## 3. Attention as Matrix Multiplication

### 3.1 Scaled Dot-Product Attention

The attention mechanism in Transformers is built entirely from matrix operations:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

where:
- $Q \in \mathbb{R}^{n \times d_k}$: queries
- $K \in \mathbb{R}^{m \times d_k}$: keys
- $V \in \mathbb{R}^{m \times d_v}$: values

```python
def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def scaled_dot_product_attention(Q, K, V, mask=None):
    """Scaled dot-product attention.

    Args:
        Q: (seq_q, d_k) or (batch, seq_q, d_k)
        K: (seq_k, d_k) or (batch, seq_k, d_k)
        V: (seq_k, d_v) or (batch, seq_k, d_v)
        mask: optional boolean mask

    Returns:
        output: (seq_q, d_v), attention weights: (seq_q, seq_k)
    """
    d_k = Q.shape[-1]

    # Step 1: Q K^T / sqrt(d_k)
    scores = Q @ K.swapaxes(-2, -1) / np.sqrt(d_k)

    # Step 2: Apply mask (for causal / padding)
    if mask is not None:
        scores = np.where(mask, scores, -1e9)

    # Step 3: Softmax
    weights = softmax(scores, axis=-1)

    # Step 4: Weighted sum of values
    output = weights @ V

    return output, weights

# Example
seq_len, d_k, d_v = 6, 8, 8
Q = np.random.randn(seq_len, d_k)
K = np.random.randn(seq_len, d_k)
V = np.random.randn(seq_len, d_v)

output, weights = scaled_dot_product_attention(Q, K, V)
print(f"Attention output shape: {output.shape}")
print(f"Attention weights shape: {weights.shape}")
print(f"Weights sum per row: {weights.sum(axis=-1)}")

import matplotlib.pyplot as plt

plt.figure(figsize=(6, 5))
plt.imshow(weights, cmap='viridis', aspect='auto')
plt.colorbar()
plt.xlabel('Key position')
plt.ylabel('Query position')
plt.title('Attention Weights')
plt.tight_layout()
plt.show()
```

### 3.2 Multi-Head Attention

Multi-head attention runs multiple attention heads in parallel and concatenates the results:

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) W^O$$

where $\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$.

```python
class MultiHeadAttention:
    def __init__(self, d_model, n_heads):
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.d_model = d_model

        # Projection matrices
        scale = np.sqrt(2.0 / d_model)
        self.W_Q = np.random.randn(d_model, d_model) * scale
        self.W_K = np.random.randn(d_model, d_model) * scale
        self.W_V = np.random.randn(d_model, d_model) * scale
        self.W_O = np.random.randn(d_model, d_model) * scale

    def forward(self, Q, K, V, mask=None):
        """Multi-head attention forward pass.

        Q, K, V: (batch_size, seq_len, d_model)
        """
        batch_size = Q.shape[0]

        # Linear projections
        Q_proj = Q @ self.W_Q.T  # (batch, seq, d_model)
        K_proj = K @ self.W_K.T
        V_proj = V @ self.W_V.T

        # Reshape into heads: (batch, n_heads, seq, d_k)
        Q_heads = Q_proj.reshape(batch_size, -1, self.n_heads, self.d_k).transpose(0, 2, 1, 3)
        K_heads = K_proj.reshape(batch_size, -1, self.n_heads, self.d_k).transpose(0, 2, 1, 3)
        V_heads = V_proj.reshape(batch_size, -1, self.n_heads, self.d_k).transpose(0, 2, 1, 3)

        # Scaled dot-product attention per head
        scores = np.einsum('bhqd,bhkd->bhqk', Q_heads, K_heads) / np.sqrt(self.d_k)
        if mask is not None:
            scores = np.where(mask, scores, -1e9)
        weights = softmax(scores, axis=-1)
        heads_out = np.einsum('bhqk,bhkd->bhqd', weights, V_heads)

        # Concatenate heads: (batch, seq, d_model)
        concat = heads_out.transpose(0, 2, 1, 3).reshape(batch_size, -1, self.d_model)

        # Final projection
        output = concat @ self.W_O.T
        return output, weights

# Test
batch_size, seq_len, d_model, n_heads = 4, 16, 64, 8

Q = np.random.randn(batch_size, seq_len, d_model)
K = np.random.randn(batch_size, seq_len, d_model)
V = np.random.randn(batch_size, seq_len, d_model)

mha = MultiHeadAttention(d_model, n_heads)
output, weights = mha.forward(Q, K, V)

print(f"Input shape: {Q.shape}")
print(f"Output shape: {output.shape}")
print(f"Attention weights shape: {weights.shape}")
```

### 3.3 Causal (Autoregressive) Masking

In decoder models (GPT), each position can only attend to previous positions. This is implemented with a triangular mask:

```python
# Causal mask: lower triangular
seq_len = 8
causal_mask = np.tril(np.ones((seq_len, seq_len), dtype=bool))
print("Causal mask:")
print(causal_mask.astype(int))

# Apply to attention
Q = np.random.randn(seq_len, 16)
K = np.random.randn(seq_len, 16)
V = np.random.randn(seq_len, 16)

output_causal, weights_causal = scaled_dot_product_attention(Q, K, V, mask=causal_mask)

# Verify: position i has zero weight on positions > i
for i in range(seq_len):
    assert np.allclose(weights_causal[i, i+1:], 0, atol=1e-6)
print("Causal masking verified: no future attention leakage")
```

---

## 4. Weight Initialization

### 4.1 Why Initialization Matters

Poor initialization can cause **vanishing** or **exploding gradients**, where the signal either decays to zero or grows without bound as it propagates through layers. Both phenomena are understood through the eigenvalues of the weight matrices.

```python
# Demonstrate vanishing/exploding gradients
def simulate_forward(n_layers, n_dim, init_scale):
    """Simulate forward pass through many linear layers."""
    x = np.random.randn(n_dim)
    norms = [np.linalg.norm(x)]

    for _ in range(n_layers):
        W = np.random.randn(n_dim, n_dim) * init_scale
        x = W @ x
        norms.append(np.linalg.norm(x))

    return norms

n_layers, n_dim = 50, 100

plt.figure(figsize=(12, 5))
for scale, label in [(0.5, 'Small init (vanishing)'),
                      (1.0 / np.sqrt(n_dim), 'Xavier/He (stable)'),
                      (2.0, 'Large init (exploding)')]:
    norms = simulate_forward(n_layers, n_dim, scale)
    plt.semilogy(norms, label=f'{label} (scale={scale:.4f})')

plt.xlabel('Layer')
plt.ylabel('Activation norm')
plt.title('Effect of Weight Initialization on Signal Propagation')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

### 4.2 Xavier and He Initialization

**Xavier (Glorot) initialization** (for tanh/sigmoid):

$$W_{ij} \sim \mathcal{N}\left(0, \frac{2}{n_{\text{in}} + n_{\text{out}}}\right)$$

**He (Kaiming) initialization** (for ReLU):

$$W_{ij} \sim \mathcal{N}\left(0, \frac{2}{n_{\text{in}}}\right)$$

The derivation relies on keeping the variance of activations constant across layers.

```python
def analyze_initialization(init_fn, n_layers=20, n_dim=256, n_trials=100):
    """Analyze how activations propagate with a given initialization."""
    layer_means = np.zeros(n_layers)
    layer_stds = np.zeros(n_layers)

    for _ in range(n_trials):
        x = np.random.randn(n_dim)
        for layer in range(n_layers):
            W = init_fn(n_dim, n_dim)
            z = W @ x
            x = relu(z)

        # Record stats at each layer (simplified: just final)
        pass

    # More detailed: track layer by layer
    x = np.random.randn(32, n_dim)  # Batch
    stats = []
    for layer in range(n_layers):
        W = init_fn(n_dim, n_dim)
        z = x @ W.T
        x = relu(z)
        stats.append({
            'mean': x.mean(),
            'std': x.std(),
            'dead_ratio': (x == 0).mean()
        })

    return stats

# Compare initializations
def xavier_init(n_in, n_out):
    return np.random.randn(n_out, n_in) * np.sqrt(2.0 / (n_in + n_out))

def he_init(n_in, n_out):
    return np.random.randn(n_out, n_in) * np.sqrt(2.0 / n_in)

def naive_init(n_in, n_out):
    return np.random.randn(n_out, n_in) * 0.01

stats_xavier = analyze_initialization(xavier_init)
stats_he = analyze_initialization(he_init)
stats_naive = analyze_initialization(naive_init)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for stats, name in [(stats_xavier, 'Xavier'), (stats_he, 'He'), (stats_naive, 'Naive (0.01)')]:
    stds = [s['std'] for s in stats]
    deads = [s['dead_ratio'] for s in stats]
    axes[0].plot(stds, label=name)
    axes[1].plot(deads, label=name)

axes[0].set_xlabel('Layer')
axes[0].set_ylabel('Activation std')
axes[0].set_title('Activation Scale Across Layers')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].set_xlabel('Layer')
axes[1].set_ylabel('Dead neuron ratio')
axes[1].set_title('Dead Neurons (ReLU)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### 4.3 Orthogonal Initialization

Initializing weights as random orthogonal matrices perfectly preserves norms during forward propagation (for linear networks):

$$\|W\mathbf{x}\|_2 = \|\mathbf{x}\|_2 \quad \text{when } W^TW = I$$

```python
from scipy.stats import ortho_group

def orthogonal_init(n_in, n_out):
    """Orthogonal initialization."""
    if n_in == n_out:
        return ortho_group.rvs(n_in)
    else:
        flat = np.random.randn(n_out, n_in)
        U, _, Vt = np.linalg.svd(flat, full_matrices=False)
        return U if n_out <= n_in else Vt

# Compare: signal propagation with orthogonal init (linear network)
n_layers, n_dim = 100, 64

x_orth = np.random.randn(n_dim)
x_he = np.random.randn(n_dim)
norms_orth = [np.linalg.norm(x_orth)]
norms_he = [np.linalg.norm(x_he)]

for _ in range(n_layers):
    W_orth = orthogonal_init(n_dim, n_dim)
    W_he = he_init(n_dim, n_dim)
    x_orth = W_orth @ x_orth  # Linear (no activation)
    x_he = W_he @ x_he
    norms_orth.append(np.linalg.norm(x_orth))
    norms_he.append(np.linalg.norm(x_he))

plt.figure(figsize=(10, 5))
plt.semilogy(norms_orth, label='Orthogonal')
plt.semilogy(norms_he, label='He')
plt.xlabel('Layer')
plt.ylabel('Activation norm')
plt.title('Signal Propagation: Orthogonal vs He (Linear Network)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

---

## 5. GPU Acceleration and Parallelism

### 5.1 Why Linear Algebra Maps to GPUs

GPUs excel at linear algebra because matrix operations are **embarrassingly parallel**:

- Matrix-vector product: each output element is an independent dot product
- Matrix multiplication: $C_{ij} = \sum_k A_{ik} B_{kj}$ -- all $(i, j)$ pairs are independent
- Element-wise operations: perfectly parallel

```python
# Demonstrate parallelism in matrix operations
import time

# CPU baseline
sizes = [100, 500, 1000, 2000, 4000]
times_np = []

for n in sizes:
    A = np.random.randn(n, n).astype(np.float32)
    B = np.random.randn(n, n).astype(np.float32)

    start = time.time()
    C = A @ B
    elapsed = time.time() - start
    times_np.append(elapsed)

    flops = 2 * n**3
    gflops = flops / elapsed / 1e9
    print(f"n={n:5d}: time={elapsed:.4f}s, {gflops:.1f} GFLOPS")

# Theoretical: matrix multiply is O(n^3), but GPU can achieve near-peak throughput
# A modern GPU does 10-100 TFLOPS for float32
```

### 5.2 Memory Layout and BLAS

The performance of matrix operations depends heavily on memory layout (row-major vs column-major) and the use of optimized BLAS (Basic Linear Algebra Subprograms).

```python
# Row-major (C-order) vs column-major (Fortran-order)
n = 2000
A_c = np.ascontiguousarray(np.random.randn(n, n))  # Row-major
A_f = np.asfortranarray(np.random.randn(n, n))      # Column-major

# Row slicing: faster for C-order
start = time.time()
for i in range(n):
    _ = A_c[i, :].sum()
t_c_row = time.time() - start

start = time.time()
for i in range(n):
    _ = A_f[i, :].sum()
t_f_row = time.time() - start

# Column slicing: faster for Fortran-order
start = time.time()
for j in range(n):
    _ = A_c[:, j].sum()
t_c_col = time.time() - start

start = time.time()
for j in range(n):
    _ = A_f[:, j].sum()
t_f_col = time.time() - start

print(f"Row slicing:  C-order={t_c_row:.4f}s, F-order={t_f_row:.4f}s")
print(f"Col slicing:  C-order={t_c_col:.4f}s, F-order={t_f_col:.4f}s")
print(f"\nNumPy BLAS config:")
np.show_config()
```

### 5.3 Mixed Precision Training

Modern GPUs have dedicated hardware for lower-precision math (float16, bfloat16). Mixed precision training uses float16 for forward/backward passes but float32 for weight updates:

```python
# Simulate mixed precision effects
np.random.seed(42)
n = 500

# Generate a matrix operation in different precisions
A = np.random.randn(n, n)
B = np.random.randn(n, n)

# Full precision
C_64 = A.astype(np.float64) @ B.astype(np.float64)
C_32 = A.astype(np.float32) @ B.astype(np.float32)
C_16 = A.astype(np.float16) @ B.astype(np.float16)

# Relative errors
err_32 = np.linalg.norm(C_32 - C_64) / np.linalg.norm(C_64)
err_16 = np.linalg.norm(C_16.astype(np.float64) - C_64) / np.linalg.norm(C_64)

print(f"float32 relative error: {err_32:.2e}")
print(f"float16 relative error: {err_16:.2e}")
print(f"float16 max value: {np.finfo(np.float16).max}")
print(f"Overflow risk: max(C) = {np.max(np.abs(C_64)):.1f}")
```

---

## 6. Jacobian and Hessian in Deep Learning

### 6.1 The Jacobian Matrix

For a vector-valued function $\mathbf{f}: \mathbb{R}^n \to \mathbb{R}^m$, the Jacobian is:

$$J_{ij} = \frac{\partial f_i}{\partial x_j}$$

In neural networks, the Jacobian of a layer maps input perturbations to output perturbations.

```python
def numerical_jacobian(f, x, eps=1e-5):
    """Compute Jacobian numerically."""
    n = len(x)
    f0 = f(x)
    m = len(f0)
    J = np.zeros((m, n))
    for j in range(n):
        x_plus = x.copy()
        x_plus[j] += eps
        J[:, j] = (f(x_plus) - f0) / eps
    return J

# Jacobian of a simple neural network layer
W = np.random.randn(3, 4) * 0.5
b = np.zeros(3)

def layer_fn(x):
    return relu(W @ x + b)

x = np.random.randn(4)
J = numerical_jacobian(layer_fn, x)
print(f"Jacobian shape: {J.shape}")
print(f"Jacobian:\n{np.round(J, 3)}")

# The Jacobian of ReLU(Wx + b) is diag(relu'(z)) @ W
z = W @ x + b
J_analytic = np.diag(relu_grad(z)) @ W
print(f"\nAnalytic Jacobian:\n{np.round(J_analytic, 3)}")
print(f"Match: {np.allclose(J, J_analytic, atol=1e-4)}")

# Singular values of Jacobian indicate sensitivity
sv = np.linalg.svd(J, compute_uv=False)
print(f"\nJacobian singular values: {sv}")
print(f"Condition number: {sv[0] / sv[-1] if sv[-1] > 0 else np.inf:.2f}")
```

### 6.2 Gradient Flow and Singular Values

The effectiveness of backpropagation depends on the singular values of the Jacobian matrices through the network. If they are consistently > 1, gradients explode. If < 1, they vanish.

```python
# Track Jacobian singular values through layers
def analyze_gradient_flow(layer_sizes, init_fn, n_trials=10):
    """Analyze singular values of per-layer Jacobians."""
    all_sv = []

    for _ in range(n_trials):
        x = np.random.randn(layer_sizes[0])
        layer_svs = []

        for i in range(len(layer_sizes) - 1):
            W = init_fn(layer_sizes[i], layer_sizes[i+1])
            z = W @ x
            # Jacobian = diag(relu'(z)) @ W
            relu_mask = (z > 0).astype(float)
            J = np.diag(relu_mask) @ W
            sv = np.linalg.svd(J, compute_uv=False)
            layer_svs.append(sv)
            x = relu(z)

        all_sv.append(layer_svs)

    return all_sv

layer_sizes = [64] * 11  # 10 layers, all 64 dim

for init_fn, name in [(he_init, 'He'), (xavier_init, 'Xavier')]:
    sv_data = analyze_gradient_flow(layer_sizes, init_fn, n_trials=20)

    # Average max singular value per layer
    avg_max_sv = [np.mean([trial[l][0] for trial in sv_data]) for l in range(10)]
    avg_min_sv = [np.mean([trial[l][-1] for trial in sv_data if len(trial[l]) > 0])
                  for l in range(10)]

    print(f"\n{name} initialization:")
    print(f"  Avg max sv: {[f'{s:.3f}' for s in avg_max_sv]}")
```

---

## 7. Low-Rank Adaptation (LoRA)

LoRA fine-tunes large models by decomposing weight updates into low-rank matrices, dramatically reducing the number of trainable parameters:

$$W' = W + \Delta W = W + BA$$

where $B \in \mathbb{R}^{d \times r}$ and $A \in \mathbb{R}^{r \times k}$, with $r \ll \min(d, k)$.

```python
class LoRALayer:
    def __init__(self, W_pretrained, rank=4, alpha=1.0):
        """LoRA: Low-Rank Adaptation of a pretrained weight matrix.

        Args:
            W_pretrained: (d_out, d_in) frozen weight matrix
            rank: rank of the low-rank update
            alpha: scaling factor
        """
        self.W = W_pretrained  # Frozen
        d_out, d_in = W_pretrained.shape
        self.rank = rank
        self.alpha = alpha

        # Trainable low-rank factors
        self.B = np.zeros((d_out, rank))         # Zero-initialized
        self.A = np.random.randn(rank, d_in) * 0.01  # Small random init

    def forward(self, x):
        """Forward: y = (W + (alpha/rank) * B @ A) @ x."""
        scale = self.alpha / self.rank
        return (self.W + scale * self.B @ self.A) @ x

    @property
    def n_params(self):
        return self.B.size + self.A.size

# Example: adapt a large pretrained weight matrix
d_in, d_out = 4096, 4096
W_pretrained = np.random.randn(d_out, d_in) * np.sqrt(1.0 / d_in)

for rank in [1, 4, 16, 64]:
    lora = LoRALayer(W_pretrained, rank=rank)
    original_params = d_out * d_in
    lora_params = lora.n_params
    print(f"Rank {rank:3d}: trainable params = {lora_params:8,d} "
          f"({100 * lora_params / original_params:.3f}% of original)")

# Test forward pass
x = np.random.randn(d_in)
y_original = W_pretrained @ x
y_lora = lora.forward(x)
print(f"\nOutput difference norm: {np.linalg.norm(y_original - y_lora):.6f}")
print(f"(Should be small since B is zero-initialized)")
```

---

## Exercises

### Exercise 1: Backpropagation by Hand

For a two-layer network with shapes $[3, 4, 2]$, ReLU activation, and MSE loss:

1. Perform a forward pass for a single input
2. Compute all gradients analytically
3. Verify against numerical gradients (finite differences)
4. Update weights with one step of gradient descent

### Exercise 2: Attention Implementation

Implement multi-head attention from scratch:

1. Project Q, K, V into multiple heads
2. Compute scaled dot-product attention per head
3. Concatenate and project the output
4. Apply causal masking for autoregressive generation
5. Verify output shapes and that attention weights sum to 1

### Exercise 3: Initialization Comparison

For a 20-layer ReLU network with hidden size 256:

1. Initialize with naive (0.01), Xavier, He, and orthogonal initialization
2. Pass 1000 random inputs through the network
3. Plot the mean and standard deviation of activations at each layer
4. Which initialization best preserves the activation scale?

### Exercise 4: Gradient Flow Analysis

Build a 10-layer network and compute the Jacobian at each layer:

1. Compute singular values of each layer's Jacobian
2. Compute the product of max singular values across all layers
3. How does this relate to the gradient magnitude at the first layer?

### Exercise 5: LoRA Simulation

For a simulated language model with 4 weight matrices of size $1024 \times 1024$:

1. Implement LoRA with ranks $r \in \{1, 4, 16, 64\}$
2. Compare total trainable parameters vs. full fine-tuning
3. Simulate a fine-tuning task: given target weight changes $\Delta W$, find optimal $B, A$ such that $BA \approx \Delta W$ (hint: use truncated SVD)

---

[Previous: Lesson 16](./16_Linear_Algebra_in_Machine_Learning.md) | [Overview](./00_Overview.md) | [Next: Lesson 18](./18_Linear_Algebra_in_Computer_Graphics.md)

**License**: CC BY-NC 4.0
