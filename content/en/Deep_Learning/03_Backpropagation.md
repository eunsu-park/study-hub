# 03. Understanding Backpropagation

[Previous: Neural Network Basics](./02_Neural_Network_Basics.md) | [Next: Training Techniques](./04_Training_Techniques.md)

---

## Learning Objectives

- Understand the principles of the backpropagation algorithm
- Learn gradient calculation using the chain rule
- Implement backpropagation directly with NumPy

---

## Theory & Principles

Backpropagation is often introduced as "apply the chain rule layer by layer," but this hides the structure that makes it efficient. The key idea is the **vector-Jacobian product (VJP)**: at every layer, you never form the full Jacobian matrix; you propagate a single cotangent vector that always has the shape of one tensor, not the product of two. This is why training a 100-million-parameter network is feasible at all.

This section covers:

- **A.** The chain rule in Jacobian formalism
- **B.** Vector-Jacobian product and why we never materialize the full Jacobian
- **C.** Backprop as repeated VJP application on the DAG
- **D.** Forward-mode vs reverse-mode trade-offs in deep nets

### A. Chain Rule in Jacobian Form

Let `f = g \circ h` with `h: R^n -> R^m` and `g: R^m -> R^k`. The Jacobian of the composition is the matrix product of Jacobians:

```
J_f(x) = J_g(h(x)) * J_h(x)            [k x n]  =  [k x m] * [m x n]
```

A neural network is a long composition `f = f_L \circ f_{L-1} \circ ... \circ f_1`. By induction, its Jacobian is:

```
J_f = J_{f_L} * J_{f_{L-1}} * ... * J_{f_1}
```

If we computed this product naively from the right (forward mode) or from the left (reverse mode), the intermediate matrices would be enormous — `J_{f_l}` for a fully-connected layer is `m x n` with both dimensions in the thousands or millions. Storing one such matrix is already gigabytes; storing the chain is impossible.

### B. Vector-Jacobian Products: The Key Saving

For a scalar loss `L`, we only need a *single row* of the final Jacobian — `dL/dx`, which has shape `(1, n)` or equivalently the same shape as `x`. Reverse-mode AD exploits this by propagating a row vector `v^T` from left to right:

```
v_L^T  =  dL/dy_L                                     [shape: same as y_L]
v_{l-1}^T = v_l^T * J_{f_l}(x_{l-1})                  [shape: same as x_{l-1}]
```

The product `v^T * J` is a **vector-Jacobian product (VJP)**. Each VJP can be computed *without ever materializing J*, because the structure of the layer tells us how to multiply a vector by `J` directly. Examples:

- **Linear layer** `y = W x + b`: `J = W`, so VJP is `v^T W`. Cost: O(m * n), no Jacobian stored.
- **ReLU** `y_i = max(0, x_i)`: `J` is diagonal with `J_{ii} = [x_i > 0]`. VJP is element-wise mask: `v_i * [x_i > 0]`. Cost: O(n).
- **Softmax + cross-entropy**: VJP simplifies to `softmax(x) - one_hot(y)`. Cost: O(n).

Every primitive op in PyTorch ships with a hand-written VJP rule — that is what `grad_fn` actually stores.

### C. Backprop = Repeated VJP on the DAG

The full backprop algorithm is:

1. Run forward; remember intermediate activations needed by the VJP rules.
2. Initialize `v = 1` (the seed `dL/dL`) at the loss node.
3. Walk the DAG in reverse topological order. At each node, look up its VJP rule, multiply the incoming `v` by the local Jacobian, and pass the result to the parents. If a node has multiple consumers, sum the cotangents flowing back into it (the chain rule's sum-over-paths becomes a literal sum at fan-out points).
4. Accumulate the result into `.grad` for leaf tensors with `requires_grad=True`.

The total cost is `O(forward cost)` because each VJP costs roughly the same as the corresponding forward op.

### D. Forward vs Reverse Mode in Deep Nets

For a network with `n` parameters and a scalar loss `L`:

- **Reverse mode**: one backward pass = O(forward) regardless of `n`. This is what `loss.backward()` does.
- **Forward mode**: one tangent propagation = O(forward), but you need *one pass per input direction*. To get `dL/dx_i` for all `n` parameters takes `n` forward passes — completely infeasible.

Forward mode wins only when `n << m` (few parameters, many outputs), which is the opposite of deep learning. PyTorch 2.x exposes both via `torch.func.jacrev` and `torch.func.jacfwd`; the right choice always depends on the shape ratio.

### From Theory to the Code Below

| Theory concept | Code construct in this lesson |
|----------------|-------------------------------|
| Chain rule (matrix form) | The hand-written `dL_dW = X^T @ dL_dy` lines |
| VJP for linear layer | `dL/dW`, `dL/dx` derivations in NumPy |
| VJP for ReLU | `(x > 0)` mask used in the backward pass |
| Backprop = reverse traversal | The order `dL_dy -> dL_dW2 -> dL_dh -> dL_dW1` |

---


## 1. What is Backpropagation?

Backpropagation is an algorithm for training neural network weights.

```
Forward Pass:  Input ──▶ Hidden Layer ──▶ Output ──▶ Loss
Backward Pass: Input ◀── Hidden Layer ◀── Output ◀── Loss
```

### Intuition: Credit Assignment

Think of a neural network as an assembly line with multiple stations. When the final product (prediction) is defective (high loss), we need to figure out **which station was responsible and by how much** — this is the *credit assignment problem*. Backpropagation solves it by tracing the error backwards through the chain: each station (layer) reports its local sensitivity to inputs ("if my input changed by ε, my output would change by ∂output/∂input × ε"), and these local sensitivities multiply along the chain to give each parameter's contribution to the final error.

### Core Ideas

1. **Forward Pass**: Compute values from input to output
2. **Loss Calculation**: Difference between prediction and ground truth
3. **Backward Pass**: Propagate gradients from loss towards input — each layer multiplies the incoming gradient by its local Jacobian
4. **Weight Update**: Adjust weights using gradients

---

## 2. Chain Rule

Why does the chain rule matter for neural networks? A neural network is a deeply nested function composition: `L(softmax(W2 * relu(W1 * x + b1) + b2), y)`. To compute `dL/dW1`, we cannot differentiate directly — we must decompose the derivative into a product of local derivatives at each layer. The chain rule provides exactly this decomposition, making it possible to compute gradients layer by layer without ever expanding the full expression.

The differentiation rule for composite functions.

### Formula

```
y = f(g(x))

dy/dx = (dy/dg) × (dg/dx)
```

### Example

```
z = x²
y = sin(z)
L = y²

dL/dx = (dL/dy) × (dy/dz) × (dz/dx)
      = 2y × cos(z) × 2x
```

---

## 3. Backpropagation for a Single Neuron

### Forward Pass

```python
z = w*x + b      # Linear transformation
a = sigmoid(z)    # Activation
L = (a - y)²     # Loss (MSE)
```

### Backward Pass (Gradient Calculation)

```python
dL/da = 2(a - y)                    # Gradient of loss w.r.t. activation
da/dz = sigmoid(z) * (1 - sigmoid(z))  # Sigmoid derivative
dz/dw = x                           # Gradient of linear transform w.r.t. weight
dz/db = 1                           # Gradient of linear transform w.r.t. bias

# Apply chain rule
dL/dw = (dL/da) × (da/dz) × (dz/dw)
dL/db = (dL/da) × (da/dz) × (dz/db)
```

---

## 4. Loss Functions

### MSE (Mean Squared Error)

```python
L = (1/n) × Σ(y_pred - y_true)²
dL/dy_pred = (2/n) × (y_pred - y_true)
```

### Cross-Entropy (Classification)

```python
L = -Σ y_true × log(y_pred)
dL/dy_pred = -y_true / y_pred  # Simplified when combined with softmax
```

### Softmax + Cross-Entropy Combined

```python
# Amazing result: becomes very simple
dL/dz = y_pred - y_true  # Gradient w.r.t. softmax input
```

---

## 5. MLP Backpropagation

Backpropagation process for a 2-layer MLP.

### Architecture

```
Input(x) → [W1, b1] → ReLU → [W2, b2] → Output(y)
```

### Forward Pass

```python
# Why save z1 and a1?  The backward pass needs these intermediate values
# to compute gradients — this is the memory-compute tradeoff of backprop.
z1 = x @ W1 + b1      # Linear transform: project input into hidden space
a1 = relu(z1)          # Non-linearity: enable learning of non-linear patterns
z2 = a1 @ W2 + b2      # Linear transform: project hidden representation to output
y_pred = z2            # Or softmax(z2) for classification
```

### Backward Pass

```python
# Output layer
dL/dz2 = y_pred - y_true  # (for softmax + CE)

# Why transpose?  In the forward pass, z2 = a1 @ W2, where a1 is (batch, H)
# and W2 is (H, out).  To get dL/dW2, we need to "undo" the matmul so that
# each element W2[i,j] gets credited for how much it contributed to the loss.
# By the chain rule: dL/dW2[i,j] = Σ_batch a1[:,i] × dL/dz2[:,j].
# In matrix form that's exactly a1.T @ dL/dz2  (transpose aligns the
# batch dimension for the dot product).
dL/dW2 = a1.T @ dL/dz2     # (H, batch) @ (batch, out) → (H, out)
dL/db2 = sum(dL/dz2, axis=0)

# Hidden layer — propagating the gradient backwards through W2:
# In the forward pass, z2 = a1 @ W2.  To compute dL/da1, we need the
# gradient w.r.t. a1 (the *input* to this matmul).  Again by chain rule:
# dL/da1[:,i] = Σ_j dL/dz2[:,j] × W2[i,j], which in matrix form is
# dL/dz2 @ W2.T — the transpose of W2 "routes" each output gradient
# back to the input dimension that produced it.
dL/da1 = dL/dz2 @ W2.T     # (batch, out) @ (out, H) → (batch, H)
dL/dz1 = dL/da1 * relu_derivative(z1)  # element-wise: ReLU passes gradient where z1 > 0, blocks where z1 ≤ 0
dL/dW1 = x.T @ dL/dz1
dL/db1 = sum(dL/dz1, axis=0)
```

---

## 6. NumPy Implementation Core

```python
class MLP:
    def backward(self, x, y_true, y_pred, cache):
        """Backpropagation: compute gradients"""
        # Why unpack cache?  Forward pass saved intermediate activations (a1)
        # and pre-activation values (z1) — we need them here to compute gradients.
        # Without caching, we'd have to recompute the forward pass.
        a1, z1 = cache

        # --- Output layer gradients ---
        # Why start here?  Backprop works from the loss backward.
        # For softmax + cross-entropy, the combined gradient simplifies to (y_pred - y_true).
        dz2 = y_pred - y_true
        # Why a1.T?  dL/dW2[i,j] = sum_over_batch(a1[:,i] * dz2[:,j]),
        # which is the matrix product a1.T @ dz2.
        dW2 = a1.T @ dz2
        db2 = np.sum(dz2, axis=0)

        # --- Hidden layer gradients (chain rule) ---
        da1 = dz2 @ self.W2.T  # Route output-layer error back through W2
        # Why (z1 > 0)?  This is ReLU's derivative: gradient flows where neurons
        # were active (z1 > 0), and is blocked where they were off (z1 <= 0).
        dz1 = da1 * (z1 > 0)
        dW1 = x.T @ dz1
        db1 = np.sum(dz1, axis=0)

        # Why return a dict?  The optimizer needs each parameter's gradient separately
        # to apply the update rule (e.g., W -= lr * grad).
        return {'W1': dW1, 'b1': db1, 'W2': dW2, 'b2': db2}
```

---

## 7. PyTorch's Automatic Differentiation

In PyTorch, all of this is automatic.

```python
# Forward pass
y_pred = model(x)
loss = criterion(y_pred, y_true)

# Backward pass (automatic!)
loss.backward()

# Access gradients
print(model.fc1.weight.grad)
```

### Computational Graph

```python
x = torch.tensor([2.0], requires_grad=True)
y = x ** 2
z = y * 3
z.backward()

# x.grad = dz/dx = dz/dy × dy/dx = 3 × 2x = 12
```

---

## 8. Vanishing/Exploding Gradient Problems

### Vanishing Gradient

- Cause: Derivatives of sigmoid/tanh close to 0
- Solution: ReLU, Residual Connections

### Exploding Gradient

- Cause: Gradient accumulation in deep networks
- Solution: Gradient Clipping, Batch Normalization

```python
# Gradient Clipping in PyTorch
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

---

## 9. Numerical Gradient Verification

A method to verify if backpropagation implementation is correct.

```python
def numerical_gradient(f, x, h=1e-5):
    """Compute gradient using numerical differentiation"""
    grad = np.zeros_like(x)
    for i in range(x.size):
        x_plus = x.copy()
        x_plus.flat[i] += h
        x_minus = x.copy()
        x_minus.flat[i] -= h
        grad.flat[i] = (f(x_plus) - f(x_minus)) / (2 * h)
    return grad

# Verification
analytical_grad = backward(...)  # Analytical gradient
numerical_grad = numerical_gradient(loss_fn, weights)
diff = np.linalg.norm(analytical_grad - numerical_grad)
assert diff < 1e-5, "Gradient check failed!"
```

---

## Summary

### Core of Backpropagation

1. **Chain Rule**: Core of composite function differentiation
2. **Local Computation**: Gradients computed independently at each layer
3. **Gradient Propagation**: Propagate from output towards input

### What You Learn from NumPy

- Meaning of matrix transpose and multiplication
- Role of activation function derivatives
- Gradient summation in batch processing

### Moving to PyTorch

- All gradients computed in one line with `loss.backward()`
- Automatic computational graph construction
- GPU acceleration

---

## Exercises

### Exercise 1: Chain Rule by Hand

Given the following composite function, compute the gradient `dL/dx` analytically using the chain rule:

```
x → z = 3x + 1 → a = relu(z) → L = a²
```

1. Write out each intermediate derivative: `dL/da`, `da/dz`, `dz/dx`.
2. Apply the chain rule to get `dL/dx`.
3. Verify your result numerically using the finite-difference formula `(f(x+h) - f(x-h)) / (2h)` with `h=1e-5` and `x=2.0`.

### Exercise 2: Single Neuron Backprop

Manually implement forward and backward passes for a single neuron with sigmoid activation and MSE loss.

1. Use `w=0.5`, `b=0.1`, `x=1.0`, `y_true=0.8`.
2. Compute the forward pass: `z = w*x + b`, `a = sigmoid(z)`, `L = (a - y_true)²`.
3. Compute gradients `dL/dw` and `dL/db` by hand, step by step.
4. Implement this in NumPy and verify your hand-computed values match.

### Exercise 3: Gradient Check for a 2-Layer MLP

Use numerical gradient verification to confirm your backprop implementation is correct.

1. Build a 2-layer MLP (input=4, hidden=8, output=2) in NumPy with random weights.
2. Implement the analytical backward pass.
3. Implement `numerical_gradient` using the central difference formula.
4. Compute the relative error: `||analytical - numerical|| / (||analytical|| + ||numerical||)`.
5. Confirm the relative error is below `1e-5`.

### Exercise 4: Vanishing Gradients with Sigmoid vs ReLU

Empirically observe the vanishing gradient problem.

1. Build a deep MLP with 10 hidden layers in PyTorch, using sigmoid activations.
2. Initialize weights using `torch.nn.init.normal_(std=1.0)`.
3. Run a forward pass and call `loss.backward()`.
4. Print the gradient norm for each layer's weight.
5. Repeat steps 1-4 using ReLU activations and compare the gradient norms across layers. Explain the observed difference.

### Exercise 5: Computational Graph and `retain_graph`

Explore PyTorch's dynamic computational graph.

1. Create a computation `z = x**2 + y**2` where `x` and `y` require gradients.
2. Call `z.backward()` and print `x.grad` and `y.grad`.
3. Try calling `z.backward()` again and observe the error.
4. Re-create the computation and call `z.backward(retain_graph=True)` twice. Verify that gradients accumulate (double the expected values), and explain why `optimizer.zero_grad()` is necessary in training loops.

---

## Next Steps

In [04_Training_Techniques.md](./04_Training_Techniques.md), we'll learn methods for weight updates using gradients.
