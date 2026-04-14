# Backpropagation

**Previous**: [Loss Functions](./05_Loss_Functions.md) | **Next**: [Gradient Descent Variants](./07_Gradient_Descent_Variants.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Derive backpropagation from the chain rule of calculus
2. Draw and traverse a computational graph to compute gradients
3. Implement backpropagation for a 2-layer MLP from scratch
4. Compute gradients for each layer's weights and biases
5. Verify analytical gradients using numerical gradient checking
6. Explain the forward pass / backward pass symmetry
7. Identify gradient flow patterns and potential issues
8. Apply backpropagation to networks with different activation functions

---

Backpropagation is the algorithm that makes neural network training possible. It efficiently computes the gradient of the loss with respect to every weight in the network by applying the chain rule of calculus layer by layer, from output to input. While the math can seem intimidating at first, the core idea is surprisingly simple: work backward through the computational graph, multiplying local gradients along the way.

---

## 1. The Core Idea

### 1.1 What Problem Does Backpropagation Solve?

To update weights using gradient descent, we need:

```
∂J/∂W^(l)   for every layer l
∂J/∂b^(l)   for every layer l

These gradients tell us: "How much does the loss change if I slightly 
adjust each weight?"
```

For a network with millions of weights, computing each gradient separately would be impossibly slow. Backpropagation computes ALL gradients in a single backward pass.

### 1.2 Key Insight: The Chain Rule

```
If y = f(g(x)), then dy/dx = dy/dg · dg/dx

For a neural network:
  x → z^(1) → a^(1) → z^(2) → a^(2) → L

  ∂L/∂W^(1) = ∂L/∂a^(2) · ∂a^(2)/∂z^(2) · ∂z^(2)/∂a^(1) · ∂a^(1)/∂z^(1) · ∂z^(1)/∂W^(1)
               └────────────────────────────────────────────────────────────────────────────┘
                                    chain rule, right to left
```

---

## 2. Computational Graphs

### 2.1 What Is a Computational Graph?

A computational graph breaks a complex expression into simple operations at each node:

```
Example: L = (σ(w·x + b) - y)^2

    x ──┐
        ├──► × ──► + ──► σ ──► - ──► ()^2 ──► L
    w ──┘         │            │
                  b            y

Each node:
  - Forward: computes its output from inputs
  - Backward: computes local gradient and passes it upstream
```

### 2.2 Local Gradients

Each operation has a simple local gradient:

```
Addition:     f = a + b     →  ∂f/∂a = 1,  ∂f/∂b = 1
Multiplication: f = a · b  →  ∂f/∂a = b,  ∂f/∂b = a
Sigmoid:      f = σ(a)     →  ∂f/∂a = σ(a)(1 - σ(a))
Square:       f = a^2      →  ∂f/∂a = 2a
ReLU:         f = max(0,a) →  ∂f/∂a = 1 if a > 0, else 0
MatMul:       f = W·a      →  ∂f/∂W = a^T,  ∂f/∂a = W^T
```

### 2.3 Backward Pass Example

```
Forward:  x=2, w=3, b=1
  
  m = w·x = 6
  z = m + b = 7
  a = σ(z) = σ(7) ≈ 0.999
  
  y = 1 (target)
  d = a - y = -0.001
  L = d^2 = 0.000001

Backward: Starting from ∂L/∂L = 1

  ∂L/∂d = 2d = -0.002
  ∂L/∂a = ∂L/∂d · ∂d/∂a = -0.002 · 1 = -0.002
  ∂L/∂z = ∂L/∂a · ∂a/∂z = -0.002 · σ(z)(1-σ(z)) = -0.002 · 0.0009 ≈ -1.8e-6
  ∂L/∂w = ∂L/∂z · ∂z/∂m · ∂m/∂w = -1.8e-6 · 1 · x = -3.6e-6
  ∂L/∂b = ∂L/∂z · ∂z/∂b = -1.8e-6 · 1 = -1.8e-6
  ∂L/∂x = ∂L/∂z · ∂z/∂m · ∂m/∂x = -1.8e-6 · 1 · w = -5.4e-6
```

---

## 3. Backpropagation for a 2-Layer MLP

### 3.1 Network Setup

```
Input:    x ∈ R^(n0)
Layer 1:  z^(1) = W^(1)·x + b^(1),     a^(1) = ReLU(z^(1))
Layer 2:  z^(2) = W^(2)·a^(1) + b^(2),  a^(2) = softmax(z^(2))
Loss:     L = CCE(a^(2), y)

Parameters: W^(1), b^(1), W^(2), b^(2)
```

### 3.2 Forward Pass (Review)

```
Step 1:  z^(1) = W^(1) · x + b^(1)          shape: (n1, 1)
Step 2:  a^(1) = ReLU(z^(1))                 shape: (n1, 1)
Step 3:  z^(2) = W^(2) · a^(1) + b^(2)      shape: (n2, 1)
Step 4:  a^(2) = softmax(z^(2))              shape: (n2, 1)
Step 5:  L = -Σ y_k · log(a^(2)_k)           scalar
```

### 3.3 Backward Pass (Derivation)

We work backward, computing the gradient of L with respect to each variable.

**Step 5 → 4: Output layer gradient (softmax + CCE)**

```
δ^(2) = ∂L/∂z^(2) = a^(2) - y               shape: (n2, 1)

(This elegant result comes from combining softmax and CCE derivatives)
```

**Step 4 → 3: Gradients for W^(2) and b^(2)**

Since z^(2) = W^(2) · a^(1) + b^(2):

```
∂L/∂W^(2) = δ^(2) · (a^(1))^T               shape: (n2, n1)
∂L/∂b^(2) = δ^(2)                             shape: (n2, 1)
```

**Step 3 → 2: Propagate gradient to hidden layer**

```
∂L/∂a^(1) = (W^(2))^T · δ^(2)                shape: (n1, 1)
```

**Step 2 → 1: Apply ReLU derivative**

```
δ^(1) = ∂L/∂z^(1) = ∂L/∂a^(1) ⊙ ReLU'(z^(1))   shape: (n1, 1)
       = (W^(2))^T · δ^(2) ⊙ (z^(1) > 0)

(⊙ denotes element-wise multiplication)
```

**Step 1 → 0: Gradients for W^(1) and b^(1)**

```
∂L/∂W^(1) = δ^(1) · x^T                      shape: (n1, n0)
∂L/∂b^(1) = δ^(1)                              shape: (n1, 1)
```

### 3.4 Summary of Backward Pass

```
δ^(L) = a^(L) - y                              (output error)
∂L/∂W^(L) = δ^(L) · (a^(L-1))^T               (weight gradient)
∂L/∂b^(L) = δ^(L)                              (bias gradient)

For l = L-1, L-2, ..., 1:
  δ^(l) = (W^(l+1))^T · δ^(l+1) ⊙ σ'(z^(l))  (propagate error backward)
  ∂L/∂W^(l) = δ^(l) · (a^(l-1))^T             (weight gradient)
  ∂L/∂b^(l) = δ^(l)                            (bias gradient)
```

---

## 4. Batch Backpropagation

For a batch of m samples (columns of X):

```
Forward:
  Z^(l) = W^(l) · A^(l-1) + b^(l)    (b broadcast across columns)
  A^(l) = σ(Z^(l))

Backward:
  dZ^(L) = A^(L) - Y                              shape: (nL, m)
  dW^(L) = (1/m) · dZ^(L) · (A^(L-1))^T           shape: (nL, n_{L-1})
  db^(L) = (1/m) · Σ_{columns} dZ^(L)              shape: (nL, 1)

  dA^(l) = (W^(l+1))^T · dZ^(l+1)                 shape: (n_l, m)
  dZ^(l) = dA^(l) ⊙ σ'(Z^(l))                     shape: (n_l, m)
  dW^(l) = (1/m) · dZ^(l) · (A^(l-1))^T            shape: (n_l, n_{l-1})
  db^(l) = (1/m) · Σ_{columns} dZ^(l)              shape: (n_l, 1)
```

---

## 5. Implementation

### 5.1 Complete Forward + Backward

```python
import numpy as np

def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return (z > 0).astype(float)

def softmax(z):
    z_shifted = z - np.max(z, axis=0, keepdims=True)
    exp_z = np.exp(z_shifted)
    return exp_z / np.sum(exp_z, axis=0, keepdims=True)

def cross_entropy_loss(y_pred, y_true):
    m = y_true.shape[1]
    return -np.sum(y_true * np.log(y_pred + 1e-15)) / m

def forward_pass(X, params):
    """Forward pass, returns activations cache."""
    caches = [{'a': X}]  # a^(0) = X
    a = X
    for i, (W, b) in enumerate(params):
        z = W @ a + b
        if i < len(params) - 1:
            a = relu(z)
        else:
            a = softmax(z)
        caches.append({'z': z, 'a': a})
    return a, caches

def backward_pass(Y, params, caches):
    """Backward pass, returns gradients."""
    m = Y.shape[1]
    L = len(params)
    grads = []

    # Output layer: softmax + CCE
    dz = caches[L]['a'] - Y  # (nL, m)

    for l in range(L, 0, -1):
        a_prev = caches[l-1]['a']
        dW = (1/m) * dz @ a_prev.T
        db = (1/m) * np.sum(dz, axis=1, keepdims=True)
        grads.insert(0, (dW, db))

        if l > 1:  # propagate to previous layer
            da = params[l-1][0].T @ dz
            dz = da * relu_derivative(caches[l-1]['z'])

    return grads

# Demo: 2-layer network [3, 4, 2]
np.random.seed(42)
W1 = np.random.randn(4, 3) * 0.1
b1 = np.zeros((4, 1))
W2 = np.random.randn(2, 4) * 0.1
b2 = np.zeros((2, 1))
params = [(W1, b1), (W2, b2)]

# Dummy data: 5 samples, 3 features, 2 classes
X = np.random.randn(3, 5)
Y = np.array([[1, 0, 1, 0, 1],
              [0, 1, 0, 1, 0]])  # one-hot

# Forward
y_pred, caches = forward_pass(X, params)
loss = cross_entropy_loss(y_pred, Y)
print(f"Loss: {loss:.4f}")

# Backward
grads = backward_pass(Y, params, caches)
for i, (dW, db) in enumerate(grads):
    print(f"Layer {i+1}: dW shape {dW.shape}, db shape {db.shape}")
```

---

## 6. Gradient Checking

### 6.1 Why Check Gradients?

Backpropagation is error-prone to implement. A single sign error or transposition mistake can produce plausible-looking but incorrect gradients. Numerical gradient checking verifies your implementation.

### 6.2 The Method

For each parameter θ_i, compare the analytical gradient with the numerical approximation:

```
Numerical:   ∂L/∂θ_i ≈ [L(θ_i + ε) - L(θ_i - ε)] / (2ε)

Relative error = |grad_analytical - grad_numerical| / max(|grad_analytical|, |grad_numerical|)

If relative error < 1e-5 → likely correct
If relative error > 1e-3 → likely a bug
```

### 6.3 Implementation

```python
def gradient_check(X, Y, params, epsilon=1e-7):
    """Verify analytical gradients against numerical gradients."""
    # Analytical gradients
    y_pred, caches = forward_pass(X, params)
    grads = backward_pass(Y, params, caches)

    for l, (W, b) in enumerate(params):
        dW_analytical = grads[l][0]

        # Numerical gradient for W
        dW_numerical = np.zeros_like(W)
        for i in range(W.shape[0]):
            for j in range(W.shape[1]):
                W[i, j] += epsilon
                y_plus, _ = forward_pass(X, params)
                loss_plus = cross_entropy_loss(y_plus, Y)

                W[i, j] -= 2 * epsilon
                y_minus, _ = forward_pass(X, params)
                loss_minus = cross_entropy_loss(y_minus, Y)

                dW_numerical[i, j] = (loss_plus - loss_minus) / (2 * epsilon)
                W[i, j] += epsilon  # restore

        # Compare
        diff = np.linalg.norm(dW_analytical - dW_numerical)
        norm = np.linalg.norm(dW_analytical) + np.linalg.norm(dW_numerical)
        relative_error = diff / (norm + 1e-15)
        status = "OK" if relative_error < 1e-5 else "FAIL"
        print(f"Layer {l+1} W: relative error = {relative_error:.2e} [{status}]")
```

---

## 7. Backpropagation Intuition

### 7.1 Credit Assignment

Backpropagation answers: "How much did each weight contribute to the error?"

```
                    Layer 1      Layer 2      Output
                    ┌─────┐      ┌─────┐
    x ──────────────┤     ├──────┤     ├──────► ŷ ──► Loss = 3.2
                    └─────┘      └─────┘
                      w1           w2

    Backprop tells us:
    - w2 contributed 2.1 to the error  → gets larger update
    - w1 contributed 0.3 to the error  → gets smaller update
    
    This is "credit assignment": distributing blame for the error.
```

### 7.2 Forward vs. Backward Symmetry

```
Forward Pass (left → right):
  a^(0) → W^(1) → z^(1) → σ → a^(1) → W^(2) → z^(2) → σ → a^(2) → L

Backward Pass (right → left):
  dL/dL → dL/da^(2) → dL/dz^(2) → dL/da^(1) → dL/dz^(1) → dL/dW^(1)

Each operation in the forward pass has a corresponding gradient in the backward pass.
```

### 7.3 Computational Complexity

```
Forward pass:   O(N)   where N = total number of weights
Backward pass:  O(N)   same order as forward pass!

Backpropagation is efficient: computing ALL gradients costs about
the same as a single forward pass. This is much better than the
O(N^2) cost of computing each gradient separately.
```

---

## 8. Common Pitfalls

### 8.1 Forgetting to Cache

Backpropagation needs the activations computed during the forward pass. Always store z^(l) and a^(l) during forward.

### 8.2 In-Place Operations

```python
# WRONG: modifies the cached value
a = relu(z)
z *= 2  # This changes the cached z used in backward!

# RIGHT: keep z unchanged
a = relu(z)
z_scaled = z * 2
```

### 8.3 Shape Mismatches

Always verify matrix dimensions match:
```
dW = dz @ a_prev.T    # (n_l, m) @ (n_{l-1}, m)^T = (n_l, n_{l-1}) ✓
db = sum(dz, axis=1)   # (n_l, m) → (n_l, 1) ✓
```

---

## 9. Summary

```
Key Takeaways
═══════════════════════════════════════════════════════
1. Backprop = chain rule applied layer by layer, output → input
2. Each layer: δ^(l) = (W^(l+1))^T · δ^(l+1) ⊙ σ'(z^(l))
3. Weight gradients: dW^(l) = δ^(l) · (a^(l-1))^T / m
4. Bias gradients: db^(l) = mean of δ^(l) across samples
5. Softmax + CCE gives: δ^(L) = a^(L) - y (elegant!)
6. Same computational cost as forward pass: O(N)
7. Always verify with gradient checking (ε = 1e-7)
8. Cache forward pass values; avoid in-place modifications
═══════════════════════════════════════════════════════
```

---

## Exercises

1. Derive backpropagation for a 3-layer MLP with tanh activations
2. Implement gradient checking and verify your backward pass
3. Draw the computational graph for L = ||σ(Wx + b) - y||^2
4. Implement backpropagation for a network with batch normalization

---

**Previous**: [Loss Functions](./05_Loss_Functions.md) | **Next**: [Gradient Descent Variants](./07_Gradient_Descent_Variants.md)
