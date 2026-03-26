# 13. Linear Algebra for Deep Learning

## Learning Objectives

- Understand and utilize tensor concepts, dimensions, and axis operations
- Master efficient tensor operations using Einstein notation and einsum
- Understand the principles of automatic differentiation (forward/reverse mode) and computational graphs
- Recognize numerical stability issues in deep learning and their solutions
- Learn the mathematical theory of weight initialization and its practical application
- Understand the mathematical background of batch/layer normalization, residual connections, etc.

---

## 1. Tensor Operations

### 1.1 Tensor Hierarchy

**Scalar (0-tensor)**: Single number
```python
import numpy as np
import torch

s = 3.14
```

**Vector (1-tensor)**: 1D array
```python
v = np.array([1, 2, 3])  # shape: (3,)
```

**Matrix (2-tensor)**: 2D array
```python
M = np.array([[1, 2], [3, 4]])  # shape: (2, 2)
```

**Tensor (n-tensor)**: n-dimensional array
```python
T = np.random.randn(3, 4, 5)  # shape: (3, 4, 5)
```

### 1.2 Tensor Axes and Dimensions

Typical tensor shapes in deep learning:
- **Images**: (batch, channels, height, width) or (batch, height, width, channels)
- **Sequences**: (batch, sequence_length, features)
- **Weights**: (output_features, input_features)

```python
# Understanding axes
batch_images = np.random.randn(32, 3, 224, 224)  # NCHW format
print(f"Shape: {batch_images.shape}")
print(f"Batch size: {batch_images.shape[0]}")
print(f"Number of channels: {batch_images.shape[1]}")
print(f"Height: {batch_images.shape[2]}")
print(f"Width: {batch_images.shape[3]}")

# Operations along axes
batch_mean = batch_images.mean(axis=0)  # Batch mean: (3, 224, 224)
spatial_mean = batch_images.mean(axis=(2, 3))  # Spatial mean: (32, 3)
global_mean = batch_images.mean()  # Global mean: scalar

print(f"Batch mean shape: {batch_mean.shape}")
print(f"Spatial mean shape: {spatial_mean.shape}")
print(f"Global mean: {global_mean}")
```

### 1.3 Broadcasting

NumPy and PyTorch automatically expand arrays of different sizes for operations.

**Broadcasting rules**:
1. If shapes have different numbers of dimensions, prepend 1s to the smaller shape
2. Compatible if dimensions are equal or one of them is 1
3. Dimensions of size 1 are stretched to match the other size

```python
# Broadcasting example
A = np.random.randn(32, 3, 224, 224)  # Batch images
mean = np.array([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)  # Per-channel mean
std = np.array([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)   # Per-channel std

# Normalization: broadcasting applied
A_normalized = (A - mean) / std
print(f"Normalized shape: {A_normalized.shape}")  # Still (32, 3, 224, 224)

# Broadcasting visualization
print("\nBroadcasting process:")
print(f"A: {A.shape}")
print(f"mean: {mean.shape} -> broadcast -> {A.shape}")
print(f"Result: {A_normalized.shape}")
```

### 1.4 Tensor Shape Transformations

```python
import torch

# Various shape transformations
x = torch.randn(32, 3, 224, 224)

# 1. reshape: requires contiguous memory
x_flat = x.reshape(32, -1)  # (32, 3*224*224)
print(f"Flatten: {x_flat.shape}")

# 2. view vs reshape
x_view = x.view(32, 3, -1)  # (32, 3, 224*224)
x_reshaped = x.reshape(32, 3, -1)  # same

# 3. transpose: swap axes
x_transposed = x.transpose(1, 2)  # (32, 224, 3, 224)
print(f"Transpose: {x_transposed.shape}")

# 4. permute: reorder axes in arbitrary order
x_permuted = x.permute(0, 2, 3, 1)  # NCHW -> NHWC
print(f"Permute: {x_permuted.shape}")

# 5. squeeze/unsqueeze: add/remove dimensions
x_squeezed = torch.randn(32, 1, 224, 224).squeeze(1)  # (32, 224, 224)
x_unsqueezed = x_squeezed.unsqueeze(1)  # (32, 1, 224, 224)
print(f"Squeeze: {x_squeezed.shape}, Unsqueeze: {x_unsqueezed.shape}")
```

---

## 2. Einstein Notation / einsum

### 2.1 Einstein Summation Convention

**Core idea**: Repeated indices are automatically summed.

**Traditional notation**:
$$
C_{ik} = \sum_{j} A_{ij} B_{jk}
$$

**Einstein notation**:
$$
C_{ik} = A_{ij} B_{jk}
$$

Since $j$ appears on both sides, it's implicitly summed.

### 2.2 Basic einsum Usage

```python
# 1. Matrix-vector product: y = Ax
A = np.random.randn(3, 4)
x = np.random.randn(4)
y1 = A @ x                      # Traditional method
y2 = np.einsum('ij,j->i', A, x)  # einsum
print(f"Matrix-vector product matches: {np.allclose(y1, y2)}")

# 2. Matrix multiplication: C = AB
A = np.random.randn(3, 4)
B = np.random.randn(4, 5)
C1 = A @ B
C2 = np.einsum('ij,jk->ik', A, B)
print(f"Matrix multiplication matches: {np.allclose(C1, C2)}")

# 3. Trace: tr(A) = Σ A_ii
A = np.random.randn(4, 4)
trace1 = np.trace(A)
trace2 = np.einsum('ii->', A)
print(f"Trace matches: {np.allclose(trace1, trace2)}")

# 4. Outer product: C_ij = a_i b_j
a = np.array([1, 2, 3])
b = np.array([4, 5])
C1 = np.outer(a, b)
C2 = np.einsum('i,j->ij', a, b)
print(f"Outer product matches: {np.allclose(C1, C2)}")
```

### 2.3 Batch Operations

Most useful for deep learning.

```python
# Batch matrix multiplication (bmm)
# A: (batch, n, m), B: (batch, m, p) -> C: (batch, n, p)
batch_size, n, m, p = 32, 10, 20, 15
A = np.random.randn(batch_size, n, m)
B = np.random.randn(batch_size, m, p)

C1 = np.einsum('bij,bjk->bik', A, B)
# Compare with torch
A_torch = torch.from_numpy(A)
B_torch = torch.from_numpy(B)
C2 = torch.bmm(A_torch, B_torch).numpy()
print(f"Batch matrix multiplication matches: {np.allclose(C1, C2)}")

print(f"Input shapes: A={A.shape}, B={B.shape}")
print(f"Output shape: C={C1.shape}")
```

### 2.4 Attention Mechanism

Transformer's core operations are very concise with einsum.

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V
$$

```python
def attention_einsum(Q, K, V):
    """
    Q: (batch, query_len, d_k)
    K: (batch, key_len, d_k)
    V: (batch, key_len, d_v)
    """
    d_k = Q.shape[-1]

    # QK^T: (batch, query_len, key_len)
    scores = np.einsum('bqd,bkd->bqk', Q, K) / np.sqrt(d_k)

    # Softmax
    attention_weights = np.exp(scores - scores.max(axis=-1, keepdims=True))
    attention_weights /= attention_weights.sum(axis=-1, keepdims=True)

    # Attention weights × V: (batch, query_len, d_v)
    output = np.einsum('bqk,bkv->bqv', attention_weights, V)

    return output, attention_weights

# Test
batch, seq_len, d_model = 32, 10, 64
Q = np.random.randn(batch, seq_len, d_model)
K = np.random.randn(batch, seq_len, d_model)
V = np.random.randn(batch, seq_len, d_model)

output, weights = attention_einsum(Q, K, V)
print(f"Attention output shape: {output.shape}")
print(f"Attention weights shape: {weights.shape}")
```

### 2.5 Complex Tensor Operations

```python
# 1. Bilinear form: x^T A y
A = np.random.randn(5, 5)
x = np.random.randn(5)
y = np.random.randn(5)

result1 = x @ A @ y
result2 = np.einsum('i,ij,j->', x, A, y)
print(f"Bilinear form matches: {np.allclose(result1, result2)}")

# 2. Batch bilinear form
batch_size = 32
x_batch = np.random.randn(batch_size, 5)
y_batch = np.random.randn(batch_size, 5)
result = np.einsum('bi,ij,bj->b', x_batch, A, y_batch)
print(f"Batch bilinear form shape: {result.shape}")

# 3. Tensor contraction
# A: (i,j,k), B: (k,l,m) -> C: (i,j,l,m)
A = np.random.randn(3, 4, 5)
B = np.random.randn(5, 6, 7)
C = np.einsum('ijk,klm->ijlm', A, B)
print(f"Tensor contraction result shape: {C.shape}")
```

---

## 3. Automatic Differentiation

### 3.1 Computational Graph

All computations are represented as directed acyclic graphs (DAGs).

```python
import torch

# Computational graph example
x = torch.tensor(2.0, requires_grad=True)
y = torch.tensor(3.0, requires_grad=True)

# Forward computation
z = x**2 + x*y + y**2
z.backward()  # Backpropagation

print(f"z = {z.item()}")
print(f"∂z/∂x = {x.grad.item()}")  # 2x + y = 2*2 + 3 = 7
print(f"∂z/∂y = {y.grad.item()}")  # x + 2y = 2 + 2*3 = 8

# Computational graph visualization (conceptual)
"""
      x=2         y=3
       ↓           ↓
    x²=4    →  x*y=6  ←  y²=9
       ↓           ↓        ↓
       └─────→  z = 4+6+9 = 19
"""
```

### 3.2 Forward Mode

**Direction**: Input → Output (Jacobian-vector product, JVP)

**Computes**: $\frac{\partial y}{\partial x} \cdot v$ (directional derivative in direction $v$)

**Advantage**: Efficient when inputs are few ($n_{\text{in}} \ll n_{\text{out}}$)

**Example**: $f: \mathbb{R} \to \mathbb{R}^{1000}$

```python
# Forward mode concept (PyTorch uses reverse mode by default)
def forward_mode_example():
    """Conceptual implementation of forward mode"""
    # f(x) = x^2 + 2x + 1
    # df/dx = 2x + 2

    x = 3.0
    dx = 1.0  # Direction vector (usually 1)

    # Compute value and derivative simultaneously
    y = x**2 + 2*x + 1      # y = 16
    dy = 2*x*dx + 2*dx      # dy/dx = 8

    return y, dy

y, dy = forward_mode_example()
print(f"f(3) = {y}, f'(3) = {dy}")
```

### 3.3 Reverse Mode / Backpropagation

**Direction**: Output → Input (vector-Jacobian product, VJP)

**Computes**: $v^T \cdot \frac{\partial y}{\partial x}$ (gradient with respect to output)

**Advantage**: Efficient when outputs are few ($n_{\text{out}} \ll n_{\text{in}}$)

**Deep Learning**: Loss function is scalar ($n_{\text{out}} = 1$) → reverse mode is optimal

```python
# Reverse mode (backpropagation)
def reverse_mode_demo():
    """Demonstration of reverse mode automatic differentiation"""
    x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    w = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], requires_grad=True)
    b = torch.tensor([1.0, 1.0], requires_grad=True)

    # Forward pass: y = w^T x + b
    y = x @ w + b  # (2,)

    # Loss: L = ||y||^2
    loss = (y ** 2).sum()

    # Backpropagation
    loss.backward()

    print(f"x.grad shape: {x.grad.shape}")  # (3,)
    print(f"w.grad shape: {w.grad.shape}")  # (3, 2)
    print(f"b.grad shape: {b.grad.shape}")  # (2,)

    return loss.item()

loss = reverse_mode_demo()
print(f"Loss: {loss}")
```

### 3.4 Why Reverse Mode Fits Deep Learning

**Neural networks**: $f: \mathbb{R}^n \to \mathbb{R}$ (parameter space → loss)

- **Input dimension**: $n \sim 10^6$ ~ $10^9$ (number of parameters)
- **Output dimension**: 1 (loss function)

**Cost comparison**:
- Forward mode: $O(n)$ passes needed (for each input)
- Reverse mode: $O(1)$ pass (one backpropagation)

**Conclusion**: Reverse mode is $O(n)$ times more efficient!

```python
# Efficiency comparison simulation
import time

def forward_pass(params):
    """Simple neural network forward pass"""
    x = torch.randn(1000, 1000)
    for p in params:
        x = torch.relu(x @ p)
    return x.sum()

# Simulate large neural network
n_params = 10
param_list = [torch.randn(1000, 1000, requires_grad=True) for _ in range(n_params)]

# Reverse mode (standard backpropagation)
start = time.time()
loss = forward_pass(param_list)
loss.backward()
reverse_time = time.time() - start

print(f"Reverse mode time: {reverse_time:.4f}s")
print(f"Total parameter count: {sum(p.numel() for p in param_list):,}")
print(f"All gradients computed in a single backward pass")
```

---

## 4. Numerical Stability

### 4.1 Floating-Point Arithmetic Limitations

IEEE 754 standard:
- **Float32**: Approximately $10^{-38}$ ~ $10^{38}$, precision $\sim 10^{-7}$
- **Underflow**: Numbers too small → 0
- **Overflow**: Numbers too large → inf

```python
# Floating-point limits demonstration
print(f"Float32 minimum: {np.finfo(np.float32).min}")
print(f"Float32 maximum: {np.finfo(np.float32).max}")
print(f"Float32 epsilon: {np.finfo(np.float32).eps}")

# Underflow
x = np.float32(1e-40)
print(f"\n1e-40 (float32): {x}")  # 0.0

# Overflow
x = np.float32(1e40)
print(f"1e40 (float32): {x}")  # inf

# Precision loss
x = np.float32(1.0 + 1e-8)
print(f"1.0 + 1e-8 (float32): {x}")  # 1.0 (lost)
```

### 4.2 Log-Sum-Exp Trick

Problem: Computing $\log \sum_{i=1}^{n} e^{x_i}$ causes overflow

**Naive computation**:
```python
x = np.array([1000, 1001, 1002], dtype=np.float32)
result_naive = np.log(np.sum(np.exp(x)))
print(f"Direct computation: {result_naive}")  # inf (overflow!)
```

**Log-Sum-Exp trick**:
$$
\log \sum_{i} e^{x_i} = a + \log \sum_{i} e^{x_i - a}
$$
where $a = \max_i x_i$

```python
def log_sum_exp(x):
    """Numerically stable log-sum-exp"""
    a = np.max(x)
    return a + np.log(np.sum(np.exp(x - a)))

result_stable = log_sum_exp(x)
print(f"LSE trick: {result_stable}")  # 1002.407 (accurate)

# Compare with scipy
from scipy.special import logsumexp
result_scipy = logsumexp(x)
print(f"Scipy: {result_scipy}")
print(f"Matches: {np.isclose(result_stable, result_scipy)}")
```

### 4.3 Numerically Stable Softmax

**Standard softmax**:
$$
\text{softmax}(x)_i = \frac{e^{x_i}}{\sum_j e^{x_j}}
$$

**Problem**: Overflow with large $x_i$

**Stable version**:
$$
\text{softmax}(x)_i = \frac{e^{x_i - \max_j x_j}}{\sum_j e^{x_j - \max_j x_j}}
$$

```python
def softmax_naive(x):
    """Unstable softmax"""
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x)

def softmax_stable(x):
    """Numerically stable softmax"""
    x_max = np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

# Test
x = np.array([1000, 1001, 1002], dtype=np.float32)

print("Unstable version:")
result_naive = softmax_naive(x)
print(f"  Result: {result_naive}")
print(f"  Sum: {result_naive.sum()}")

print("\nStable version:")
result_stable = softmax_stable(x)
print(f"  Result: {result_stable}")
print(f"  Sum: {result_stable.sum()}")

# Batch processing
batch_logits = np.random.randn(32, 1000).astype(np.float32)
probs = softmax_stable(batch_logits)
print(f"\nBatch softmax: {probs.shape}, sum: {probs.sum(axis=1)[:5]}")
```

### 4.4 Gradient Clipping

Prevents exploding gradients.

```python
def clip_gradients(gradients, max_norm=1.0):
    """Gradient norm clipping"""
    total_norm = np.sqrt(sum(np.sum(g**2) for g in gradients))
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for g in gradients:
            g *= clip_coef
    return gradients, total_norm

# PyTorch example
model = torch.nn.Linear(100, 10)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Inside training loop
for batch in range(10):
    loss = (model(torch.randn(32, 100)) ** 2).sum()
    loss.backward()

    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    optimizer.step()
    optimizer.zero_grad()
```

---

## 5. Weight Initialization Theory

### 5.1 Problem: Signal Vanishing/Explosion

Without proper initialization:
- **Signal vanishing**: Activations converge to 0 → gradient vanishing
- **Signal explosion**: Activations diverge to infinity → gradient explosion

**Goal**: Maintain variance of activations and gradients across layers

### 5.2 Xavier/Glorot Initialization

**Setting**: Linear layer $y = Wx + b$, no activation

**Assumptions**:
- $x_i$ has mean 0, variance $\text{Var}(x)$
- $W_{ij}$ are independent, mean 0

**Variance propagation**:
$$
\text{Var}(y_i) = n_{\text{in}} \cdot \text{Var}(W) \cdot \text{Var}(x)
$$

**Forward**: To maintain $\text{Var}(y) = \text{Var}(x)$
$$
\text{Var}(W) = \frac{1}{n_{\text{in}}}
$$

**Backward**: To maintain gradient variance
$$
\text{Var}(W) = \frac{1}{n_{\text{out}}}
$$

**Xavier initialization**: Compromise
$$
\text{Var}(W) = \frac{2}{n_{\text{in}} + n_{\text{out}}}
$$

```python
def xavier_uniform(n_in, n_out):
    """Xavier uniform initialization"""
    limit = np.sqrt(6 / (n_in + n_out))
    return np.random.uniform(-limit, limit, (n_in, n_out))

def xavier_normal(n_in, n_out):
    """Xavier normal initialization"""
    std = np.sqrt(2 / (n_in + n_out))
    return np.random.normal(0, std, (n_in, n_out))

# Verify variance propagation
n_layers = 10
layer_sizes = [100] * (n_layers + 1)
x = np.random.randn(1000, layer_sizes[0])

activations = [x]
for i in range(n_layers):
    W = xavier_uniform(layer_sizes[i], layer_sizes[i+1])
    x = x @ W  # Linear transformation (no activation)
    activations.append(x)

# Check variance at each layer
variances = [np.var(a) for a in activations]
print("Xavier initialization - variance per layer:")
for i, var in enumerate(variances):
    print(f"  Layer {i}: {var:.4f}")
```

### 5.3 He Initialization (for ReLU)

ReLU zeroes out negative values, reducing variance by half.

**He initialization**:
$$
\text{Var}(W) = \frac{2}{n_{\text{in}}}
$$

```python
def he_normal(n_in, n_out):
    """He normal initialization (for ReLU)"""
    std = np.sqrt(2 / n_in)
    return np.random.normal(0, std, (n_in, n_out))

# Simulate ReLU network
x = np.random.randn(1000, 100)
activations_he = [x]

for i in range(10):
    W = he_normal(100, 100)
    x = x @ W
    x = np.maximum(0, x)  # ReLU
    activations_he.append(x)

variances_he = [np.var(a) for a in activations_he]
print("\nHe initialization + ReLU - variance per layer:")
for i, var in enumerate(variances_he):
    print(f"  Layer {i}: {var:.4f}")

# Xavier vs He comparison
x_xavier = np.random.randn(1000, 100)
activations_xavier = [x_xavier]
for i in range(10):
    W = xavier_uniform(100, 100)
    x_xavier = x_xavier @ W
    x_xavier = np.maximum(0, x_xavier)
    activations_xavier.append(x_xavier)

print("\nXavier initialization + ReLU (inappropriate):")
for i, a in enumerate(activations_xavier):
    print(f"  Layer {i}: variance {np.var(a):.4f}, zero fraction {(a==0).mean():.2%}")
```

### 5.4 PyTorch Initialization

```python
import torch.nn as nn

# Default initialization
layer = nn.Linear(100, 50)
print(f"Default initialization: mean {layer.weight.mean():.4f}, std {layer.weight.std():.4f}")

# Xavier initialization
nn.init.xavier_uniform_(layer.weight)
print(f"Xavier: mean {layer.weight.mean():.4f}, std {layer.weight.std():.4f}")

# He initialization
nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
print(f"He: mean {layer.weight.mean():.4f}, std {layer.weight.std():.4f}")

# Full model initialization
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

model = nn.Sequential(
    nn.Linear(100, 50),
    nn.ReLU(),
    nn.Linear(50, 10)
)
model.apply(init_weights)
```

---

## 6. Normalization and Residual Connections

### 6.1 Batch Normalization

**Formula**:
$$
\hat{x} = \frac{x - \mathbb{E}[x]}{\sqrt{\text{Var}[x] + \epsilon}}
$$

$$
y = \gamma \hat{x} + \beta
$$

where $\gamma, \beta$ are learnable parameters.

```python
class BatchNorm1d:
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        self.eps = eps
        self.momentum = momentum
        self.gamma = np.ones(num_features)
        self.beta = np.zeros(num_features)
        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)

    def forward(self, x, training=True):
        if training:
            # Compute batch statistics
            batch_mean = x.mean(axis=0)
            batch_var = x.var(axis=0)

            # Update running statistics (exponential moving average)
            self.running_mean = (1 - self.momentum) * self.running_mean + \
                                self.momentum * batch_mean
            self.running_var = (1 - self.momentum) * self.running_var + \
                               self.momentum * batch_var

            # Normalize
            x_norm = (x - batch_mean) / np.sqrt(batch_var + self.eps)
        else:
            # At inference: use running statistics
            x_norm = (x - self.running_mean) / np.sqrt(self.running_var + self.eps)

        # Scale/shift
        return self.gamma * x_norm + self.beta

# Test
bn = BatchNorm1d(10)
x_train = np.random.randn(32, 10) * 10 + 5  # mean 5, std 10
x_normed = bn.forward(x_train, training=True)

print(f"Input: mean {x_train.mean(axis=0).mean():.2f}, std {x_train.std(axis=0).mean():.2f}")
print(f"After normalization: mean {x_normed.mean(axis=0).mean():.4f}, std {x_normed.std(axis=0).mean():.4f}")
```

### 6.2 Layer Normalization

Normalizes along feature dimension instead of batch dimension (used in Transformers).

```python
def layer_norm(x, gamma, beta, eps=1e-5):
    """Layer normalization"""
    # x: (batch, features)
    mean = x.mean(axis=1, keepdims=True)
    var = x.var(axis=1, keepdims=True)
    x_norm = (x - mean) / np.sqrt(var + eps)
    return gamma * x_norm + beta

x = np.random.randn(32, 128) * 5 + 10
gamma = np.ones(128)
beta = np.zeros(128)
x_ln = layer_norm(x, gamma, beta)

print(f"Layer normalization: per-sample mean {x_ln.mean(axis=1)[:5]}")
```

### 6.3 Gradient Flow in Residual Connections

ResNet's key insight: $y = F(x) + x$

**Gradient**:
$$
\frac{\partial \mathcal{L}}{\partial x} = \frac{\partial \mathcal{L}}{\partial y} \left(1 + \frac{\partial F}{\partial x}\right)
$$

The identity path ($+1$) directly propagates gradients.

```python
# Residual connection effect simulation
def plain_network(x, depth=50):
    """Plain network"""
    for _ in range(depth):
        W = np.random.randn(100, 100) * 0.01
        x = np.tanh(x @ W)
    return x

def residual_network(x, depth=50):
    """Residual network"""
    for _ in range(depth):
        W = np.random.randn(100, 100) * 0.01
        F_x = np.tanh(x @ W)
        x = F_x + x  # Residual connection
    return x

x = np.random.randn(10, 100)

y_plain = plain_network(x, depth=50)
y_residual = residual_network(x, depth=50)

print(f"Plain network: mean {y_plain.mean():.6f}, std {y_plain.std():.6f}")
print(f"Residual network: mean {y_residual.mean():.6f}, std {y_residual.std():.6f}")
print(f"\nThe plain network signal has vanished!")
```

### 6.4 Vanishing/Exploding Gradient Analysis

**Chain rule**:
$$
\frac{\partial \mathcal{L}}{\partial W_1} = \frac{\partial \mathcal{L}}{\partial y_L} \prod_{i=2}^{L} \frac{\partial y_i}{\partial y_{i-1}}
$$

**Problem**: Product of Jacobians
- $\|\frac{\partial y_i}{\partial y_{i-1}}\| < 1$ → gradient vanishing
- $\|\frac{\partial y_i}{\partial y_{i-1}}\| > 1$ → gradient explosion

**Solutions**:
1. Proper initialization (Xavier, He)
2. Batch normalization
3. Residual connections
4. Gradient clipping

---

## Practice Problems

### Problem 1: Tensor Manipulation
Implement the following in PyTorch:
(a) Transform (32, 3, 64, 64) image batch to (32, 64, 64, 3)
(b) Compute spatial mean for each image to create (32, 3) tensor
(c) Calculate per-channel standard deviation and normalize

### Problem 2: einsum Mastery
Implement the following using einsum:
(a) Diagonal sum of 3D batch matrices: (batch, n, n) → (batch,)
(b) Core operation of multi-head attention (Q, K multiplication)
(c) 4D tensor contraction: (a,b,c,d) × (c,d,e,f) → (a,b,e,f)

### Problem 3: Numerical Stability
(a) Implement stable log-softmax function (using log-sum-exp)
(b) Test with very large logit values [1000, 2000, 3000]
(c) Compare results with PyTorch's F.log_softmax

### Problem 4: Initialization Experiments
(a) Create 10-layer neural network with Xavier, He, and random(0.01) initialization
(b) Run forward pass with ReLU activation
(c) Plot activation variance at each layer and compare

### Problem 5: Batch Normalization Implementation
(a) Implement 2D batch normalization from scratch (forward + backward)
(b) Compare results with PyTorch's nn.BatchNorm2d
(c) Verify difference between training/inference modes

---

## References

### Books
- **Deep Learning** (Goodfellow et al., 2016) - Chapter 6 (Deep Networks), Chapter 8 (Optimization)
- **Dive into Deep Learning** (Zhang et al.) - Interactive book with code

### Papers
- Glorot & Bengio (2010), "Understanding the difficulty of training deep feedforward neural networks" - Xavier initialization
- He et al. (2015), "Delving Deep into Rectifiers" - He initialization
- Ioffe & Szegedy (2015), "Batch Normalization"
- He et al. (2016), "Deep Residual Learning for Image Recognition" - ResNet

### Online Resources
- [PyTorch einsum tutorial](https://pytorch.org/docs/stable/generated/torch.einsum.html)
- [Efficient Attention with einsum](https://rockt.github.io/2018/04/30/einsum)
- [Numerical Stability in Deep Learning](https://towardsdatascience.com)
- [Weight Initialization Guide](https://pytorch.org/docs/stable/nn.init.html)

### Tools
- **PyTorch**: Automatic differentiation and tensor operations
- **NumPy**: Numerical computation
- **TensorBoard**: Activation/gradient visualization
