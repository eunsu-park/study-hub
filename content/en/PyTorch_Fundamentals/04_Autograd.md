# Autograd

**Previous**: [Tensor Operations](./03_Tensor_Operations.md) | **Next**: [nn.Module](./05_nn_Module.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain how PyTorch's autograd engine builds and traverses computational graphs
2. Use `requires_grad` to mark tensors for gradient computation
3. Compute gradients with `backward()` and access them via `.grad`
4. Understand the difference between leaf and non-leaf tensors in the graph
5. Use `torch.no_grad()` and `detach()` to control gradient computation
6. Compute higher-order gradients and Jacobian-vector products
7. Debug gradient issues (None gradients, accumulation bugs, in-place errors)
8. Apply `retain_graph` and `create_graph` for advanced autograd scenarios

---

Automatic differentiation (autograd) is the engine that makes neural network training possible. Instead of computing gradients by hand, PyTorch records every operation on tensors with `requires_grad=True`, building a computational graph that can be traversed in reverse to compute all gradients simultaneously.

---

## 1. The Computational Graph

### 1.1 How It Works

When you perform operations on tensors with `requires_grad=True`, PyTorch records each operation in a **directed acyclic graph (DAG)**:

```python
import torch

x = torch.tensor(2.0, requires_grad=True)
y = torch.tensor(3.0, requires_grad=True)

# Each operation creates a node in the graph
z = x * y          # MulBackward0
w = z + x          # AddBackward0
loss = w ** 2       # PowBackward0

print(loss)              # tensor(64., grad_fn=<PowBackward0>)
print(loss.grad_fn)      # <PowBackward0 object>
```

The graph for this computation:

```
  x (leaf)     y (leaf)
    │            │
    ├────┐       │
    │    ▼       ▼
    │   z = x * y  (MulBackward0)
    │       │
    ▼       ▼
  w = z + x        (AddBackward0)
        │
        ▼
 loss = w ** 2     (PowBackward0)
```

### 1.2 Forward and Backward Pass

```python
x = torch.tensor(2.0, requires_grad=True)

# Forward pass: compute output
y = x ** 3 + 2 * x ** 2 + x
# y = x^3 + 2x^2 + x
# dy/dx = 3x^2 + 4x + 1

# Backward pass: compute gradients
y.backward()

print(x.grad)  # tensor(21.)  (3*4 + 4*2 + 1 = 21)
```

---

## 2. Leaf Tensors and grad

### 2.1 Leaf vs Non-Leaf

```python
# Leaf tensors: created directly by the user
a = torch.tensor(1.0, requires_grad=True)   # leaf
b = torch.randn(3, requires_grad=True)      # leaf
c = torch.tensor(5.0)                        # leaf (no grad)

# Non-leaf tensors: result of operations
d = a * 2     # non-leaf (has grad_fn)
e = b + 1     # non-leaf

print(a.is_leaf)  # True
print(d.is_leaf)  # False
```

### 2.2 Accessing Gradients

```python
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = (x ** 2).sum()  # scalar output

y.backward()
print(x.grad)  # tensor([2., 4., 6.])  (dy/dx = 2x)

# Non-leaf tensors do NOT retain gradients by default
z = x * 2
w = (z ** 2).sum()

# z.grad is None after backward -- non-leaf gradients are freed
# To keep them, use retain_grad():
z = x * 2
z.retain_grad()
w = (z ** 2).sum()
w.backward()
print(z.grad)  # tensor([ 4.,  8., 12.])
```

### 2.3 Gradient Accumulation

**Gradients accumulate by default** -- they are added, not replaced:

```python
x = torch.tensor(2.0, requires_grad=True)

# First backward
y1 = x ** 2
y1.backward()
print(x.grad)  # tensor(4.)

# Second backward WITHOUT zeroing gradients
y2 = x ** 3
y2.backward()
print(x.grad)  # tensor(16.)  NOT 12! (4 + 12 = 16, accumulated!)

# ALWAYS zero gradients before each backward pass
x.grad.zero_()
y3 = x ** 3
y3.backward()
print(x.grad)  # tensor(12.)  correct!
```

> **Critical Rule**: Always call `optimizer.zero_grad()` (or manually zero gradients) before `loss.backward()` in training loops. Forgetting this is one of the most common PyTorch bugs.

---

## 3. Controlling Gradient Computation

### 3.1 torch.no_grad()

Disables gradient tracking -- used during inference and parameter updates:

```python
x = torch.tensor(2.0, requires_grad=True)

# Inside no_grad, operations don't build a graph
with torch.no_grad():
    y = x * 3
    print(y.requires_grad)  # False
    print(y.grad_fn)        # None

# Common uses:
# 1. Evaluation/inference
model.eval()
with torch.no_grad():
    predictions = model(test_data)

# 2. Manual parameter updates
with torch.no_grad():
    param -= learning_rate * param.grad
```

### 3.2 detach()

Creates a tensor that shares data but is detached from the graph:

```python
x = torch.tensor(2.0, requires_grad=True)
y = x ** 2

# Detach: breaks the graph connection
y_detached = y.detach()
print(y_detached.requires_grad)  # False

# Use case: using a model's output as fixed input to another computation
features = encoder(x)
features_fixed = features.detach()  # stop gradients from flowing to encoder
output = decoder(features_fixed)
```

### 3.3 torch.enable_grad() and torch.set_grad_enabled()

```python
# Re-enable gradients inside no_grad
with torch.no_grad():
    # gradients disabled
    with torch.enable_grad():
        # gradients enabled again
        pass

# Toggle based on a flag
is_training = True
with torch.set_grad_enabled(is_training):
    output = model(x)
```

---

## 4. backward() In Detail

### 4.1 Scalar Output

When the output is a scalar, `backward()` needs no arguments:

```python
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
loss = (x ** 2).sum()  # scalar
loss.backward()
print(x.grad)  # tensor([2., 4., 6.])
```

### 4.2 Non-Scalar Output (Jacobian-Vector Product)

When the output is not a scalar, you must provide a `gradient` argument:

```python
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = x ** 2  # NOT a scalar -- shape [3]

# Must provide the "vector" in Jacobian-vector product
# Typically use ones for element-wise gradient
y.backward(torch.ones_like(y))
print(x.grad)  # tensor([2., 4., 6.])

# This is equivalent to:
# (y * torch.ones_like(y)).sum().backward()
```

### 4.3 retain_graph

By default, the computational graph is freed after `backward()`:

```python
x = torch.tensor(2.0, requires_grad=True)
y = x ** 2

y.backward()       # graph is freed
# y.backward()     # ERROR: graph already freed

# To backward multiple times, use retain_graph
x = torch.tensor(2.0, requires_grad=True)
y = x ** 2
y.backward(retain_graph=True)  # keep the graph
x.grad.zero_()
y.backward()  # works now (graph retained from previous call)
```

---

## 5. Higher-Order Gradients

### 5.1 Second Derivative

```python
x = torch.tensor(2.0, requires_grad=True)
y = x ** 3  # y = x^3, dy/dx = 3x^2, d2y/dx2 = 6x

# First derivative
grad1 = torch.autograd.grad(y, x, create_graph=True)[0]
print(grad1)  # tensor(12.)  (3 * 4 = 12)

# Second derivative
grad2 = torch.autograd.grad(grad1, x)[0]
print(grad2)  # tensor(12.)  (6 * 2 = 12)
```

### 5.2 torch.autograd.grad

A functional interface to compute gradients:

```python
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = (x ** 2).sum()

# Equivalent to y.backward(); x.grad
grad = torch.autograd.grad(y, x)[0]
print(grad)  # tensor([2., 4., 6.])

# Advantage: doesn't modify .grad attribute
# Useful for computing gradients without side effects
```

---

## 6. Practical Autograd Patterns

### 6.1 Gradient of a Loss Function

```python
# Typical training step
x = torch.randn(5, 3)                          # input
y_true = torch.tensor([0, 1, 2, 1, 0])         # labels
model = torch.nn.Linear(3, 3)                   # simple model

y_pred = model(x)                               # forward
loss = torch.nn.functional.cross_entropy(y_pred, y_true)

loss.backward()                                  # backward

# Inspect gradients
for name, param in model.named_parameters():
    print(f"{name}: grad shape = {param.grad.shape}")
    print(f"  grad norm = {param.grad.norm():.4f}")
```

### 6.2 Gradient Clipping

```python
# Prevent exploding gradients
loss.backward()

# Clip by norm
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# Clip by value
torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=0.5)
```

### 6.3 Freezing Parameters

```python
# Freeze a layer (no gradient computation)
for param in model.layer1.parameters():
    param.requires_grad = False

# Only trainable parameters
trainable = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.Adam(trainable, lr=0.001)
```

---

## 7. Common Autograd Errors

### 7.1 None Gradients

```python
x = torch.tensor(2.0)  # requires_grad=False by default
y = x ** 2
# y.backward()  -- x.grad will be None because requires_grad=False

# Fix:
x = torch.tensor(2.0, requires_grad=True)
```

### 7.2 In-Place Operations on Leaf Variables

```python
x = torch.tensor([1.0, 2.0], requires_grad=True)
# x.add_(1)  # ERROR: in-place operation on a leaf Variable

# Fix: use out-of-place operation
y = x + 1
```

### 7.3 Graph Already Freed

```python
x = torch.tensor(2.0, requires_grad=True)
y = x ** 2
y.backward()
# y.backward()  # ERROR: graph freed

# Fix: use retain_graph=True
```

### 7.4 Gradient Not Zeroed

```python
# Symptom: loss doesn't decrease as expected
# Cause: gradients accumulate across iterations

# Fix: zero gradients before backward
optimizer.zero_grad()  # or param.grad.zero_()
loss.backward()
optimizer.step()
```

---

## 8. Autograd Internals

### 8.1 grad_fn Chain

```python
x = torch.tensor(2.0, requires_grad=True)
y = x * 3       # MulBackward0
z = y + 1       # AddBackward0
w = z ** 2       # PowBackward0

# Walk the graph
print(w.grad_fn)                           # PowBackward0
print(w.grad_fn.next_functions)            # ((AddBackward0, 0),)
print(w.grad_fn.next_functions[0][0].next_functions)  # ((MulBackward0, 0),)
```

### 8.2 Hooks

Register functions that execute during backward:

```python
x = torch.tensor(2.0, requires_grad=True)

# Register a hook on the tensor
def print_grad(grad):
    print(f"Gradient: {grad}")

x.register_hook(print_grad)

y = x ** 2
y.backward()  # prints: "Gradient: 4.0"
```

---

## Summary

| Concept | Key Takeaway |
|---------|-------------|
| Computational graph | Built dynamically during forward pass; traversed in reverse during backward |
| requires_grad | Set to True for parameters you want to optimize |
| backward() | Computes gradients and stores them in `.grad` attributes of leaf tensors |
| Gradient accumulation | Gradients add up by default; always zero before backward |
| torch.no_grad() | Disables graph construction; use for inference and parameter updates |
| detach() | Breaks gradient flow; creates a tensor independent of the graph |
| retain_graph | Keep the graph after backward for multiple backward passes |
| create_graph | Enable higher-order gradient computation |

---

**Next**: [nn.Module](./05_nn_Module.md) -- Building neural network architectures with PyTorch's module system.
