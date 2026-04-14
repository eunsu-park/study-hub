# Lesson 1: Vectors and Matrices for Deep Learning

## Learning Objectives

- Distinguish between vectors, matrices, and higher-order tensors using consistent index notation
- Apply Einstein summation convention to express matrix operations compactly
- Compute matrix-vector and matrix-matrix products using batched operations
- Understand the layout conventions (numerator vs. denominator) for matrix calculus
- Differentiate scalar functions with respect to vectors and matrices
- Recognize how weight matrices, bias vectors, and activation tensors flow through a neural network layer
- Translate between mathematical notation and NumPy/einsum code

---

## 1. Tensors: The Language of Deep Learning

Deep learning operates on **tensors** -- multi-dimensional arrays of numbers. Before diving into calculus, we need a precise vocabulary for these objects.

### 1.1 Scalars, Vectors, Matrices, and Beyond

| Object | Order | Notation | Example in DL |
|--------|-------|----------|---------------|
| Scalar | 0 | $x$, $\alpha$ | Learning rate, loss value |
| Vector | 1 | $\mathbf{x} \in \mathbb{R}^n$ | Feature vector, bias |
| Matrix | 2 | $\mathbf{W} \in \mathbb{R}^{m \times n}$ | Weight matrix |
| 3-Tensor | 3 | $\mathcal{X} \in \mathbb{R}^{B \times T \times d}$ | Batch of sequences |

A **tensor of order $k$** (also called a $k$-tensor or rank-$k$ tensor in the CS sense) is an element of $\mathbb{R}^{n_1 \times n_2 \times \cdots \times n_k}$. Each dimension $n_i$ is called an **axis**.

```python
import numpy as np

scalar = 0.001          # shape: ()
vector = np.array([1.0, 2.0, 3.0])  # shape: (3,)
matrix = np.random.randn(4, 3)       # shape: (4, 3)
tensor3 = np.random.randn(8, 10, 64) # shape: (8, 10, 64) -- batch of sequences
```

### 1.2 Index Notation

Index notation writes a tensor element by listing its indices. For a matrix $\mathbf{A} \in \mathbb{R}^{m \times n}$, the element in row $i$, column $j$ is $A_{ij}$ (or $a_{ij}$).

Matrix-vector product $\mathbf{y} = \mathbf{A}\mathbf{x}$:

$$y_i = \sum_{j=1}^{n} A_{ij} x_j, \quad i = 1, \ldots, m$$

Matrix-matrix product $\mathbf{C} = \mathbf{A}\mathbf{B}$ where $\mathbf{A} \in \mathbb{R}^{m \times k}$, $\mathbf{B} \in \mathbb{R}^{k \times n}$:

$$C_{ij} = \sum_{p=1}^{k} A_{ip} B_{pj}$$

### 1.3 Einstein Summation Convention

The **Einstein convention** drops the $\sum$ symbol: any index that appears exactly twice in a product term is implicitly summed over.

$$y_i = A_{ij} x_j \quad \text{(sum over } j \text{)}$$
$$C_{ij} = A_{ip} B_{pj} \quad \text{(sum over } p \text{)}$$

This maps directly to NumPy's `np.einsum`:

```python
A = np.random.randn(4, 3)
x = np.random.randn(3)
B = np.random.randn(3, 5)

# Matrix-vector product
y = np.einsum('ij,j->i', A, x)
assert np.allclose(y, A @ x)

# Matrix-matrix product
C = np.einsum('ip,pj->ij', A, B)
assert np.allclose(C, A @ B)

# Trace: sum of diagonal elements
M = np.random.randn(4, 4)
tr = np.einsum('ii->', M)
assert np.isclose(tr, np.trace(M))

# Outer product
u = np.array([1, 2, 3])
v = np.array([4, 5])
outer = np.einsum('i,j->ij', u, v)
assert np.allclose(outer, np.outer(u, v))
```

---

## 2. Batched Operations

In deep learning, we rarely process single samples. A **batch** of $B$ input vectors $\mathbf{x}^{(1)}, \ldots, \mathbf{x}^{(B)} \in \mathbb{R}^n$ is stacked into a matrix $\mathbf{X} \in \mathbb{R}^{B \times n}$.

### 2.1 Batched Linear Transformation

A single linear layer computes $\mathbf{y} = \mathbf{W}\mathbf{x} + \mathbf{b}$ for one sample. For a batch:

$$\mathbf{Y} = \mathbf{X} \mathbf{W}^\top + \mathbf{1}_B \mathbf{b}^\top$$

where $\mathbf{X} \in \mathbb{R}^{B \times n_{\text{in}}}$, $\mathbf{W} \in \mathbb{R}^{n_{\text{out}} \times n_{\text{in}}}$, $\mathbf{b} \in \mathbb{R}^{n_{\text{out}}}$, and $\mathbf{Y} \in \mathbb{R}^{B \times n_{\text{out}}}$.

In practice, NumPy broadcasting handles the bias addition:

```python
B, n_in, n_out = 32, 784, 256
X = np.random.randn(B, n_in)
W = np.random.randn(n_out, n_in)
b = np.random.randn(n_out)

Y = X @ W.T + b  # Broadcasting adds b to each row
print(f"Input: {X.shape}, Output: {Y.shape}")  # (32, 784), (32, 256)
```

### 2.2 Batched Matrix Multiplication with einsum

For attention mechanisms, we need batched matrix multiplications:

```python
# Batch of query and key matrices
B, T, d = 4, 10, 64  # batch, sequence length, dimension
Q = np.random.randn(B, T, d)
K = np.random.randn(B, T, d)

# Batched Q @ K^T for each sample in the batch
scores = np.einsum('btd,bsd->bts', Q, K)
print(f"Attention scores: {scores.shape}")  # (4, 10, 10)
```

---

## 3. Matrix Calculus Conventions

When we take derivatives involving vectors and matrices, we need to choose a **layout convention**. There are two competing standards, and confusion between them causes endless bugs.

### 3.1 Numerator Layout vs. Denominator Layout

Consider $\mathbf{y} \in \mathbb{R}^m$ as a function of $\mathbf{x} \in \mathbb{R}^n$.

**Numerator layout** (Jacobian convention):

$$\frac{\partial \mathbf{y}}{\partial \mathbf{x}} \in \mathbb{R}^{m \times n}, \quad \left(\frac{\partial \mathbf{y}}{\partial \mathbf{x}}\right)_{ij} = \frac{\partial y_i}{\partial x_j}$$

**Denominator layout** (gradient convention):

$$\frac{\partial \mathbf{y}}{\partial \mathbf{x}} \in \mathbb{R}^{n \times m}, \quad \left(\frac{\partial \mathbf{y}}{\partial \mathbf{x}}\right)_{ij} = \frac{\partial y_j}{\partial x_i}$$

> **Convention used in this course**: We use the **numerator layout** (Jacobian convention), which is standard in deep learning and in the *Matrix Cookbook*. The Jacobian of $\mathbf{y}$ w.r.t. $\mathbf{x}$ has shape $m \times n$ -- rows indexed by outputs, columns by inputs.

### 3.2 The Gradient of a Scalar

When the output is a scalar $L \in \mathbb{R}$ (e.g., a loss function), the derivative with respect to a vector $\mathbf{x} \in \mathbb{R}^n$ is the **gradient**:

$$\nabla_{\mathbf{x}} L = \frac{\partial L}{\partial \mathbf{x}} = \begin{bmatrix} \frac{\partial L}{\partial x_1} \\ \frac{\partial L}{\partial x_2} \\ \vdots \\ \frac{\partial L}{\partial x_n} \end{bmatrix} \in \mathbb{R}^n$$

In the numerator layout this is a row vector $1 \times n$; in practice (and in PyTorch) we treat it as a column vector matching the shape of $\mathbf{x}$.

### 3.3 Derivative of a Scalar w.r.t. a Matrix

For a loss $L$ depending on a weight matrix $\mathbf{W} \in \mathbb{R}^{m \times n}$:

$$\frac{\partial L}{\partial \mathbf{W}} \in \mathbb{R}^{m \times n}, \quad \left(\frac{\partial L}{\partial \mathbf{W}}\right)_{ij} = \frac{\partial L}{\partial W_{ij}}$$

The gradient has the **same shape** as the parameter. This is the fundamental rule that makes SGD updates $\mathbf{W} \leftarrow \mathbf{W} - \eta \frac{\partial L}{\partial \mathbf{W}}$ dimensionally consistent.

---

## 4. Essential Matrix Calculus Identities

Here are the matrix calculus identities you will use repeatedly in deep learning.

### 4.1 Vector-by-Vector Derivatives

Let $\mathbf{x}, \mathbf{a} \in \mathbb{R}^n$ and $\mathbf{A} \in \mathbb{R}^{m \times n}$.

| Function | Derivative w.r.t. $\mathbf{x}$ | Shape |
|----------|-------------------------------|-------|
| $\mathbf{a}^\top \mathbf{x}$ | $\mathbf{a}^\top$ | $1 \times n$ |
| $\mathbf{x}^\top \mathbf{A} \mathbf{x}$ | $\mathbf{x}^\top (\mathbf{A} + \mathbf{A}^\top)$ | $1 \times n$ |
| $\|\mathbf{x}\|^2 = \mathbf{x}^\top \mathbf{x}$ | $2\mathbf{x}^\top$ | $1 \times n$ |
| $\mathbf{A}\mathbf{x}$ | $\mathbf{A}$ (Jacobian) | $m \times n$ |

### 4.2 Scalar-by-Matrix Derivatives

Let $L$ be a scalar, $\mathbf{W} \in \mathbb{R}^{m \times n}$, $\mathbf{x} \in \mathbb{R}^n$, $\mathbf{y} \in \mathbb{R}^m$.

| Function | $\frac{\partial L}{\partial \mathbf{W}}$ |
|----------|----------------------------------------|
| $L = \mathbf{y}^\top \mathbf{W} \mathbf{x}$ | $\mathbf{y} \mathbf{x}^\top$ |
| $L = \text{tr}(\mathbf{A}^\top \mathbf{W})$ | $\mathbf{A}$ |
| $L = \text{tr}(\mathbf{W}^\top \mathbf{A} \mathbf{W})$ | $(\mathbf{A} + \mathbf{A}^\top)\mathbf{W}$ |

### 4.3 Derivation Example: Linear Layer Gradient

Consider one sample through a linear layer:

$$\mathbf{y} = \mathbf{W}\mathbf{x} + \mathbf{b}$$

Given a downstream scalar loss $L$ and the gradient $\frac{\partial L}{\partial \mathbf{y}} \in \mathbb{R}^m$, we want $\frac{\partial L}{\partial \mathbf{W}}$ and $\frac{\partial L}{\partial \mathbf{x}}$.

**Step 1**: By the chain rule,

$$\frac{\partial L}{\partial W_{ij}} = \sum_k \frac{\partial L}{\partial y_k} \frac{\partial y_k}{\partial W_{ij}}$$

Since $y_k = \sum_l W_{kl} x_l + b_k$, we have $\frac{\partial y_k}{\partial W_{ij}} = \delta_{ki} x_j$.

$$\frac{\partial L}{\partial W_{ij}} = \frac{\partial L}{\partial y_i} x_j$$

In matrix form:

$$\boxed{\frac{\partial L}{\partial \mathbf{W}} = \frac{\partial L}{\partial \mathbf{y}} \mathbf{x}^\top} \in \mathbb{R}^{m \times n}$$

**Step 2**: For the input gradient,

$$\frac{\partial L}{\partial x_j} = \sum_k \frac{\partial L}{\partial y_k} \frac{\partial y_k}{\partial x_j} = \sum_k \frac{\partial L}{\partial y_k} W_{kj}$$

$$\boxed{\frac{\partial L}{\partial \mathbf{x}} = \mathbf{W}^\top \frac{\partial L}{\partial \mathbf{y}}} \in \mathbb{R}^n$$

```python
# Numerical verification
n_in, n_out = 3, 2
W = np.random.randn(n_out, n_in)
x = np.random.randn(n_in)
b = np.random.randn(n_out)

# Forward pass
y = W @ x + b

# Suppose dL/dy is given (from downstream)
dL_dy = np.random.randn(n_out)

# Analytical gradients
dL_dW = np.outer(dL_dy, x)     # (n_out, n_in)
dL_dx = W.T @ dL_dy            # (n_in,)
dL_db = dL_dy                  # (n_out,)

# Numerical verification via finite differences
eps = 1e-5
dL_dW_num = np.zeros_like(W)
for i in range(n_out):
    for j in range(n_in):
        W_plus = W.copy(); W_plus[i, j] += eps
        W_minus = W.copy(); W_minus[i, j] -= eps
        y_plus = W_plus @ x + b
        y_minus = W_minus @ x + b
        # Using dL/dy as the "loss gradient" direction
        dL_dW_num[i, j] = dL_dy @ (y_plus - y_minus) / (2 * eps)

print(f"dL/dW analytical:\n{dL_dW}")
print(f"dL/dW numerical:\n{dL_dW_num}")
print(f"Match: {np.allclose(dL_dW, dL_dW_num, atol=1e-4)}")
```

---

## 5. Special Matrices in Deep Learning

### 5.1 Diagonal Matrices

A diagonal matrix $\mathbf{D} = \text{diag}(d_1, \ldots, d_n)$ has $D_{ij} = d_i \delta_{ij}$. Multiplication $\mathbf{D}\mathbf{x}$ scales each element: $(D\mathbf{x})_i = d_i x_i$.

**DL usage**: Element-wise scaling in batch normalization, diagonal approximations to the Hessian (Adam optimizer).

### 5.2 Orthogonal Matrices

$\mathbf{Q} \in \mathbb{R}^{n \times n}$ is orthogonal if $\mathbf{Q}^\top \mathbf{Q} = \mathbf{I}$. Key property: orthogonal matrices preserve norms, $\|\mathbf{Q}\mathbf{x}\| = \|\mathbf{x}\|$.

**DL usage**: Orthogonal weight initialization prevents gradient vanishing/exploding.

### 5.3 Symmetric and Positive Definite Matrices

A symmetric matrix satisfies $\mathbf{A} = \mathbf{A}^\top$. A symmetric matrix is **positive definite** (PD) if $\mathbf{x}^\top \mathbf{A} \mathbf{x} > 0$ for all $\mathbf{x} \neq \mathbf{0}$.

**DL usage**: The Hessian of the loss is symmetric. PD Hessians indicate local convexity (a local minimum).

### 5.4 Low-Rank Matrices

A matrix $\mathbf{W} \in \mathbb{R}^{m \times n}$ with rank $r \ll \min(m, n)$ can be factored as $\mathbf{W} = \mathbf{U}\mathbf{V}^\top$ where $\mathbf{U} \in \mathbb{R}^{m \times r}$, $\mathbf{V} \in \mathbb{R}^{n \times r}$.

**DL usage**: LoRA (Low-Rank Adaptation) fine-tunes large language models by learning low-rank updates $\Delta \mathbf{W} = \mathbf{B}\mathbf{A}$ with $r \ll d$.

---

## 6. Tensor Reshaping and Transposition

Reshaping is a zero-cost operation that reinterprets memory layout. Understanding reshaping is critical for implementing multi-head attention, convolutions, and other DL operations.

### 6.1 Reshape, Transpose, and Permute

```python
# A batch of images: (batch, channels, height, width)
images = np.random.randn(8, 3, 32, 32)

# Flatten spatial dimensions for a fully-connected layer
flat = images.reshape(8, -1)  # (8, 3072)
print(f"Flattened: {flat.shape}")

# Multi-head attention reshape
B, T, d_model, n_heads = 4, 10, 512, 8
d_head = d_model // n_heads  # 64

x = np.random.randn(B, T, d_model)
x_heads = x.reshape(B, T, n_heads, d_head)  # (4, 10, 8, 64)
x_heads = x_heads.transpose(0, 2, 1, 3)     # (4, 8, 10, 64) -- heads become axis 1
print(f"Multi-head shape: {x_heads.shape}")
```

### 6.2 einsum for Complex Tensor Contractions

```python
# Bilinear form: x^T A y for each sample in a batch
B, m, n = 16, 5, 7
x = np.random.randn(B, m)
y = np.random.randn(B, n)
A = np.random.randn(m, n)

# Result is a scalar per sample
result = np.einsum('bi,ij,bj->b', x, A, y)
print(f"Bilinear form: {result.shape}")  # (16,)

# Verify with loop
for b in range(3):
    manual = x[b] @ A @ y[b]
    assert np.isclose(result[b], manual)
```

---

## 7. Norms in Deep Learning

### 7.1 Vector Norms

The $L^p$ norm of $\mathbf{x} \in \mathbb{R}^n$:

$$\|\mathbf{x}\|_p = \left(\sum_{i=1}^{n} |x_i|^p\right)^{1/p}$$

| Norm | Formula | DL Usage |
|------|---------|----------|
| $L^1$ | $\sum \|x_i\|$ | Sparsity regularization (Lasso) |
| $L^2$ | $\sqrt{\sum x_i^2}$ | Weight decay, gradient clipping |
| $L^\infty$ | $\max \|x_i\|$ | Adversarial robustness |

### 7.2 Matrix Norms

The **Frobenius norm** treats a matrix as a long vector:

$$\|\mathbf{W}\|_F = \sqrt{\sum_{i,j} W_{ij}^2} = \sqrt{\text{tr}(\mathbf{W}^\top \mathbf{W})}$$

The **spectral norm** is the largest singular value:

$$\|\mathbf{W}\|_2 = \sigma_{\max}(\mathbf{W})$$

**DL usage**: Spectral normalization constrains the Lipschitz constant of discriminator networks in GANs.

```python
W = np.random.randn(4, 3)

# Frobenius norm
frob = np.linalg.norm(W, 'fro')
frob_manual = np.sqrt(np.sum(W**2))
assert np.isclose(frob, frob_manual)

# Spectral norm
spectral = np.linalg.norm(W, 2)  # largest singular value
svd_vals = np.linalg.svd(W, compute_uv=False)
assert np.isclose(spectral, svd_vals[0])

print(f"Frobenius norm: {frob:.4f}")
print(f"Spectral norm: {spectral:.4f}")
```

---

## 8. Putting It in DL Context: A Complete Forward Pass

Let's trace the math through a two-layer network with ReLU activation:

$$\mathbf{h} = \text{ReLU}(\mathbf{W}_1 \mathbf{x} + \mathbf{b}_1)$$
$$\hat{y} = \mathbf{w}_2^\top \mathbf{h} + b_2$$

```python
# Network dimensions
n_in, n_hidden, n_out = 4, 8, 1

# Initialize weights
W1 = np.random.randn(n_hidden, n_in) * 0.01
b1 = np.zeros(n_hidden)
w2 = np.random.randn(n_hidden) * 0.01
b2 = 0.0

# Input
x = np.random.randn(n_in)

# Forward pass with explicit math
z1 = W1 @ x + b1         # Pre-activation: z_i = sum_j W1_{ij} x_j + b1_i
h = np.maximum(z1, 0)     # ReLU: h_i = max(0, z_i)
y_hat = w2 @ h + b2       # Output: y_hat = sum_i w2_i * h_i + b2

print(f"Input x: {x.shape}")
print(f"Hidden z1: {z1.shape}")
print(f"Hidden h (after ReLU): {h.shape}")
print(f"Output y_hat: {y_hat:.4f}")

# Shape analysis
print(f"\nW1: {W1.shape} (n_hidden x n_in)")
print(f"W1 @ x: {(W1 @ x).shape} -> same as b1: {b1.shape}")
print(f"w2: {w2.shape} (n_hidden,)")
print(f"w2 @ h: scalar -> same as b2: scalar")
```

---

## Summary

| Concept | Key Takeaway |
|---------|-------------|
| Tensor notation | Indices label axes; Einstein convention drops $\sum$ for repeated indices |
| Batched operations | Stack samples into batch dimension; use `@` or `einsum` for batch matmul |
| Layout convention | Numerator layout: gradient shape matches parameter shape |
| Matrix calculus | $\partial L / \partial \mathbf{W} = (\partial L / \partial \mathbf{y})\mathbf{x}^\top$ for a linear layer |
| Special matrices | Diagonal, orthogonal, PD, low-rank -- each has DL applications |
| Norms | Frobenius for regularization, spectral for Lipschitz constraints |

---

## Exercises

1. Use `np.einsum` to compute the trace of $\mathbf{A}\mathbf{B}$ without forming the product matrix.
2. Derive $\frac{\partial L}{\partial \mathbf{b}}$ for the linear layer $\mathbf{y} = \mathbf{W}\mathbf{x} + \mathbf{b}$.
3. Verify numerically that $\|\mathbf{Q}\mathbf{x}\|_2 = \|\mathbf{x}\|_2$ for a random orthogonal matrix $\mathbf{Q}$.
4. Implement a batched linear layer forward pass using `einsum`.
5. Compute the Frobenius norm of the gradient $\frac{\partial L}{\partial \mathbf{W}}$ and explain when this quantity is useful for monitoring training.

---

**Next**: [02. Partial Derivatives and Gradients](02_Partial_Derivatives_and_Gradients.md)
