# Custom Layers and Functions

**Previous**: [Debugging PyTorch](./11_Debugging_PyTorch.md) | **Next**: [TorchScript and Deployment](./13_TorchScript_and_Deployment.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Implement custom autograd functions with `torch.autograd.Function`
2. Define `forward()` and `backward()` (setup_context + backward in modern API)
3. Validate custom gradients with `torch.autograd.gradcheck`
4. Build reusable custom layers as `nn.Module` subclasses
5. Combine custom functions with standard modules in a model
6. Implement operations that PyTorch doesn't provide natively
7. Understand when to use custom Functions vs custom Modules

---

While PyTorch provides hundreds of built-in operations, sometimes you need to implement your own -- whether for a novel activation function, a custom loss, or an operation with a more numerically stable gradient formula.

---

## 1. Custom autograd.Function

### 1.1 Basic Structure

```python
import torch
from torch.autograd import Function

class MyReLU(Function):
    @staticmethod
    def forward(ctx, input):
        # ctx: context object for saving tensors needed in backward
        ctx.save_for_backward(input)
        return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        # grad_output: gradient of the loss w.r.t. the output of forward
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input < 0] = 0
        return grad_input

# Usage
x = torch.randn(5, requires_grad=True)
y = MyReLU.apply(x)  # use .apply(), not direct call
loss = y.sum()
loss.backward()
print(x.grad)  # gradient is 1 where x > 0, 0 where x < 0
```

### 1.2 Why Custom Functions?

1. **Numerical stability**: Implement a mathematically equivalent but numerically stabler backward
2. **Memory efficiency**: Avoid storing unnecessary intermediates
3. **Novel operations**: Implement operations not in PyTorch
4. **Interfacing with C/CUDA**: Wrap external code with autograd support

---

## 2. Multi-Input, Multi-Output Functions

### 2.1 Function with Multiple Inputs

```python
class LinearFunction(Function):
    @staticmethod
    def forward(ctx, input, weight, bias=None):
        ctx.save_for_backward(input, weight, bias)
        output = input @ weight.T
        if bias is not None:
            output += bias
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors

        grad_input = grad_output @ weight           # dL/dinput
        grad_weight = grad_output.T @ input          # dL/dweight
        grad_bias = None
        if bias is not None:
            grad_bias = grad_output.sum(dim=0)       # dL/dbias

        # Return one gradient per forward input (in same order)
        return grad_input, grad_weight, grad_bias

# Usage
x = torch.randn(4, 3, requires_grad=True)
w = torch.randn(5, 3, requires_grad=True)
b = torch.randn(5, requires_grad=True)

output = LinearFunction.apply(x, w, b)
output.sum().backward()
print(x.grad.shape)  # [4, 3]
print(w.grad.shape)  # [5, 3]
print(b.grad.shape)  # [5]
```

### 2.2 Non-Differentiable Inputs

```python
class MaskedSoftmax(Function):
    @staticmethod
    def forward(ctx, logits, mask):
        # mask is non-differentiable (integer/bool tensor)
        logits = logits.masked_fill(~mask, float('-inf'))
        probs = torch.softmax(logits, dim=-1)
        ctx.save_for_backward(probs, mask)
        return probs

    @staticmethod
    def backward(ctx, grad_output):
        probs, mask = ctx.saved_tensors
        # Standard softmax backward
        grad_logits = probs * (grad_output - (grad_output * probs).sum(
            dim=-1, keepdim=True))
        grad_logits = grad_logits.masked_fill(~mask, 0)
        return grad_logits, None  # None for mask (non-differentiable)
```

---

## 3. Gradient Checking

### 3.1 torch.autograd.gradcheck

```python
from torch.autograd import gradcheck

# Test with double precision for numerical accuracy
x = torch.randn(3, 4, dtype=torch.double, requires_grad=True)

# gradcheck compares analytical gradients with numerical (finite-difference)
result = gradcheck(MyReLU.apply, (x,), eps=1e-6, atol=1e-4)
print(f"Gradient check passed: {result}")

# For multiple inputs
x = torch.randn(4, 3, dtype=torch.double, requires_grad=True)
w = torch.randn(5, 3, dtype=torch.double, requires_grad=True)
b = torch.randn(5, dtype=torch.double, requires_grad=True)

result = gradcheck(LinearFunction.apply, (x, w, b))
print(f"Linear gradient check passed: {result}")
```

### 3.2 gradgradcheck (Second Derivatives)

```python
from torch.autograd import gradgradcheck

x = torch.randn(3, dtype=torch.double, requires_grad=True)
result = gradgradcheck(MyReLU.apply, (x,))
print(f"Second-order gradient check passed: {result}")
```

---

## 4. Practical Custom Functions

### 4.1 Straight-Through Estimator

Used for quantization and discrete operations where the true gradient is zero:

```python
class StraightThroughEstimator(Function):
    @staticmethod
    def forward(ctx, input):
        # Quantize to nearest integer
        return input.round()

    @staticmethod
    def backward(ctx, grad_output):
        # Pass gradient straight through (identity)
        return grad_output

ste = StraightThroughEstimator.apply

x = torch.tensor([0.3, 0.7, 1.2, 1.8], requires_grad=True)
y = ste(x)
print(y)  # tensor([0., 1., 1., 2.])
y.sum().backward()
print(x.grad)  # tensor([1., 1., 1., 1.])  -- gradient passes through
```

### 4.2 Numerically Stable Log-Sum-Exp

```python
class StableLogSumExp(Function):
    @staticmethod
    def forward(ctx, input, dim):
        max_val = input.max(dim=dim, keepdim=True).values
        exp_shifted = (input - max_val).exp()
        sum_exp = exp_shifted.sum(dim=dim, keepdim=True)
        output = max_val + sum_exp.log()
        ctx.save_for_backward(input, output)
        ctx.dim = dim
        return output.squeeze(dim)

    @staticmethod
    def backward(ctx, grad_output):
        input, output = ctx.saved_tensors
        # softmax is the gradient of log-sum-exp
        grad_input = torch.softmax(input, dim=ctx.dim)
        grad_input = grad_input * grad_output.unsqueeze(ctx.dim)
        return grad_input, None  # None for dim
```

### 4.3 Custom Activation: Swish with Custom Backward

```python
class Swish(Function):
    @staticmethod
    def forward(ctx, input):
        sigmoid = torch.sigmoid(input)
        output = input * sigmoid
        ctx.save_for_backward(input, sigmoid)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, sigmoid = ctx.saved_tensors
        # d(x * sigmoid(x))/dx = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
        grad = sigmoid + input * sigmoid * (1 - sigmoid)
        return grad_output * grad
```

---

## 5. Custom Layers (nn.Module)

### 5.1 Wrapping a Custom Function in a Module

```python
import torch.nn as nn

class SwishLayer(nn.Module):
    def forward(self, x):
        return Swish.apply(x)

# Use in a model
model = nn.Sequential(
    nn.Linear(784, 256),
    SwishLayer(),
    nn.Linear(256, 10),
)
```

### 5.2 Custom Layer with Parameters

```python
class ScaleShift(nn.Module):
    """Learnable element-wise scale and shift: y = gamma * x + beta."""

    def __init__(self, num_features):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        return self.gamma * x + self.beta

    def extra_repr(self):
        return f"num_features={self.gamma.shape[0]}"

layer = ScaleShift(64)
print(layer)  # ScaleShift(num_features=64)
```

### 5.3 Custom Attention Layer

```python
class SimpleSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.scale = self.head_dim ** -0.5

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, H, T, D]
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = torch.softmax(attn, dim=-1)

        out = (attn @ v).transpose(1, 2).reshape(B, T, C)
        return self.out_proj(out)
```

---

## 6. When to Use What

| Scenario | Use |
|----------|-----|
| New layer with learnable parameters | `nn.Module` subclass |
| Custom forward + standard backward | `nn.Module` with built-in ops |
| Custom backward (stability, efficiency) | `autograd.Function` |
| Non-differentiable operation with STE | `autograd.Function` |
| Simple activation function | Function in `forward()` or `autograd.Function` |
| Wrapping external C/CUDA code | `autograd.Function` + C extension |

---

## Summary

| Concept | Key Takeaway |
|---------|-------------|
| autograd.Function | Custom forward and backward; use `.apply()` to call |
| ctx.save_for_backward | Store tensors needed in backward pass |
| Gradient return order | One gradient per forward input; None for non-differentiable |
| gradcheck | Verify custom gradients against finite differences; use float64 |
| Custom Module | nn.Module with nn.Parameter for learnable weights |
| extra_repr | Override for informative `print(module)` |
| STE | Pass gradients through non-differentiable operations |

---

**Next**: [TorchScript and Deployment](./13_TorchScript_and_Deployment.md) -- Compiling and deploying PyTorch models.
