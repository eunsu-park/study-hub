# Lesson 10: Numerical Stability

## Learning Objectives

- Understand IEEE 754 floating-point representation and its limitations
- Identify overflow and underflow in common DL operations
- Derive and apply the log-sum-exp trick for numerically stable softmax and log-softmax
- Implement stable versions of sigmoid, binary cross-entropy, and softmax cross-entropy
- Understand catastrophic cancellation and how to avoid it
- Apply mixed-precision training concepts and understand when fp16 is safe
- Diagnose NaN and Inf values in training pipelines
- Implement numerically stable gradient computations

---

## 1. Floating-Point Arithmetic

### 1.1 IEEE 754 Representation

A floating-point number is stored as:

$$x = (-1)^s \times m \times 2^e$$

where $s$ is the sign bit, $m$ is the mantissa (significand), and $e$ is the exponent.

| Format | Total bits | Mantissa bits | Exponent bits | Range | Precision |
|--------|-----------|--------------|--------------|-------|-----------|
| float16 (half) | 16 | 10 | 5 | $\pm 6.5 \times 10^4$ | $\sim 3$ decimal digits |
| bfloat16 | 16 | 7 | 8 | $\pm 3.4 \times 10^{38}$ | $\sim 2$ decimal digits |
| float32 (single) | 32 | 23 | 8 | $\pm 3.4 \times 10^{38}$ | $\sim 7$ decimal digits |
| float64 (double) | 64 | 52 | 11 | $\pm 1.8 \times 10^{308}$ | $\sim 15$ decimal digits |

### 1.2 Machine Epsilon

The smallest number $\epsilon$ such that $1 + \epsilon \neq 1$ in floating-point:

```python
import numpy as np

for dtype in [np.float16, np.float32, np.float64]:
    info = np.finfo(dtype)
    print(f"{dtype.__name__:10s}: eps = {info.eps:.2e}, "
          f"max = {info.max:.2e}, min (normal) = {info.tiny:.2e}")
```

### 1.3 Special Values

- **Inf**: Result of overflow (e.g., `np.float32(1e38) * 10`)
- **-Inf**: Result of negative overflow
- **NaN** (Not a Number): Result of undefined operations (`0/0`, `inf - inf`, `sqrt(-1)`)
- **NaN propagation**: Any arithmetic with NaN produces NaN

```python
# Demonstrating special values
print(f"Overflow: {np.float32(1e38) * np.float32(10)}")     # inf
print(f"0/0: {np.float32(0) / np.float32(0)}")               # nan
print(f"inf - inf: {np.inf - np.inf}")                         # nan
print(f"nan + 1: {np.nan + 1}")                                # nan
print(f"nan > 0: {np.nan > 0}")                                # False
print(f"nan == nan: {np.nan == np.nan}")                        # False!
```

---

## 2. Overflow and Underflow

### 2.1 Overflow in Exponentials

The exponential function grows extremely fast. For float32:

$$e^{88} \approx 1.65 \times 10^{38} \approx \text{float32 max}$$
$$e^{89} = \text{Inf}$$

This is a constant threat in softmax, sigmoid, and probability computations.

### 2.2 Underflow in Probabilities

Products of many small probabilities underflow to zero:

$$\prod_{i=1}^{1000} 0.01 = 10^{-2000} \to 0 \text{ (underflow)}$$

Taking logarithms converts products to sums, avoiding underflow:

$$\log \prod p_i = \sum \log p_i$$

```python
# Underflow in probability products
probs = np.full(1000, 0.01, dtype=np.float64)

product = np.prod(probs)
log_sum = np.sum(np.log(probs))

print(f"Direct product: {product}")         # 0.0 (underflow!)
print(f"Log of product: {log_sum:.2f}")      # -4605.17 (works fine)
```

---

## 3. The Log-Sum-Exp Trick

### 3.1 The Problem

Computing $\log \sum_i e^{z_i}$ naively fails when any $z_i$ is large (overflow in $e^{z_i}$) or when all $z_i$ are very negative (underflow to $\log(0)$).

### 3.2 The Solution

$$\log \sum_{i} e^{z_i} = c + \log \sum_{i} e^{z_i - c}$$

where $c = \max_i z_i$.

**Why this works**:
- After subtracting $c$, all exponents are $\leq 0$, so $e^{z_i - c} \leq 1$ (no overflow)
- At least one exponent equals 0 (where $z_i = c$), so the sum is $\geq 1$ (no underflow in log)

```python
def log_sum_exp_naive(z):
    """Naive implementation -- WILL overflow."""
    return np.log(np.sum(np.exp(z)))

def log_sum_exp_stable(z):
    """Numerically stable log-sum-exp."""
    c = np.max(z)
    return c + np.log(np.sum(np.exp(z - c)))

# Test cases
z_normal = np.array([1.0, 2.0, 3.0])
z_large = np.array([1000.0, 1001.0, 1002.0])
z_small = np.array([-1000.0, -999.0, -998.0])

for name, z in [('Normal', z_normal), ('Large', z_large), ('Small', z_small)]:
    naive = log_sum_exp_naive(z)
    stable = log_sum_exp_stable(z)
    scipy_ref = np.logaddexp.reduce(z)  # NumPy's stable implementation
    print(f"{name:6s}: naive={naive:12.4f}, stable={stable:12.4f}, ref={scipy_ref:12.4f}")
```

---

## 4. Stable Softmax and Log-Softmax

### 4.1 Stable Softmax

$$\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_j e^{z_j}} = \frac{e^{z_i - c}}{\sum_j e^{z_j - c}}$$

where $c = \max_j z_j$.

### 4.2 Stable Log-Softmax

$$\log \text{softmax}(z_i) = z_i - \log \sum_j e^{z_j} = z_i - c - \log \sum_j e^{z_j - c}$$

The log-softmax is more numerically stable than taking $\log(\text{softmax}(z))$, because softmax outputs can be extremely small (underflow to zero, then $\log(0) = -\text{Inf}$).

```python
def softmax_naive(z):
    """Naive softmax -- overflows for large z."""
    e = np.exp(z)
    return e / e.sum()

def softmax_stable(z):
    """Numerically stable softmax."""
    e = np.exp(z - np.max(z))
    return e / e.sum()

def log_softmax_stable(z):
    """Numerically stable log-softmax."""
    c = np.max(z)
    return z - c - np.log(np.sum(np.exp(z - c)))

# Test
z = np.array([1000.0, 1000.5, 999.0])

print("Naive softmax:", softmax_naive(z))    # [nan, nan, nan]
print("Stable softmax:", softmax_stable(z))  # [0.269, 0.442, 0.289] or similar
print("Log-softmax:", log_softmax_stable(z))
```

---

## 5. Stable Sigmoid and Cross-Entropy

### 5.1 Stable Sigmoid

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

For $z \gg 0$: $\sigma(z) \approx 1$ (fine)
For $z \ll 0$: $e^{-z}$ overflows, but $\sigma(z) = \frac{e^z}{e^z + 1} \approx e^z$ (alternative form)

```python
def sigmoid_stable(z):
    """Numerically stable sigmoid."""
    return np.where(z >= 0,
                    1 / (1 + np.exp(-z)),
                    np.exp(z) / (1 + np.exp(z)))

# Test
z_test = np.array([-1000, -100, -1, 0, 1, 100, 1000], dtype=np.float64)
print("Naive sigmoid:", 1 / (1 + np.exp(-z_test)))  # Overflow for z=-1000
print("Stable sigmoid:", sigmoid_stable(z_test))
```

### 5.2 Stable Binary Cross-Entropy

$$L = -[y \log \sigma(z) + (1-y) \log(1 - \sigma(z))]$$

Computing $\log \sigma(z)$ via $\log(1/(1+e^{-z})) = -\log(1+e^{-z})$ and $\log(1 - \sigma(z)) = -z - \log(1+e^{-z})$:

$$L = -yz + \log(1 + e^z) \quad \text{if } z \geq 0$$
$$L = (1-y)z + \log(1 + e^{-z}) \quad \text{if } z < 0$$

Combined using the `softplus` function:

$$L = \max(z, 0) - yz + \log(1 + e^{-|z|})$$

```python
def bce_stable(z, y):
    """Numerically stable binary cross-entropy from logits."""
    return np.maximum(z, 0) - z * y + np.log1p(np.exp(-np.abs(z)))

def bce_naive(z, y):
    """Naive BCE -- can produce nan."""
    p = 1 / (1 + np.exp(-z))
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))

# Test with extreme logits
z_extreme = np.array([-100, -10, 0, 10, 100], dtype=np.float64)
y = np.array([0, 0, 1, 1, 1], dtype=np.float64)

print("Naive BCE:", bce_naive(z_extreme, y))
print("Stable BCE:", bce_stable(z_extreme, y))
```

### 5.3 Stable Softmax Cross-Entropy

For multiclass classification with logits $\mathbf{z}$ and one-hot label $\mathbf{y}$ (true class $c$):

$$L = -z_c + \log \sum_k e^{z_k} = -z_c + c' + \log \sum_k e^{z_k - c'}$$

where $c' = \max_k z_k$. This avoids both overflow and underflow.

```python
def softmax_ce_stable(z, y_onehot):
    """Numerically stable softmax cross-entropy from logits."""
    c = np.max(z)
    log_sum_exp = c + np.log(np.sum(np.exp(z - c)))
    true_logit = np.sum(z * y_onehot)
    return log_sum_exp - true_logit

# Test
z = np.array([100, 200, 300])  # Extreme logits
y = np.zeros(3); y[1] = 1

loss = softmax_ce_stable(z, y)
print(f"Stable softmax CE: {loss:.4f}")  # Should be 100 (= 200 - 300 ... wait, 300 - 200 = 100)
```

---

## 6. Catastrophic Cancellation

### 6.1 The Problem

When subtracting two nearly equal floating-point numbers, relative error can explode:

$$\frac{|(a - b) - (a_\text{true} - b_\text{true})|}{|a_\text{true} - b_\text{true}|}$$

If $a$ and $b$ agree in their first $k$ digits but differ in the $(k+1)$-th, the subtraction loses $k$ digits of precision.

### 6.2 Examples in DL

```python
# Catastrophic cancellation
a = 1.0
b = 1e-16

# Mathematically: (a + b) - a = b = 1e-16
result = (a + b) - a
print(f"(1 + 1e-16) - 1 = {result}")  # Should be 1e-16
print(f"Correct answer: {b}")
print(f"Relative error: {abs(result - b) / b:.2e}")

# Variance computation: two-pass vs naive
np.random.seed(42)
data = np.random.randn(10000) * 0.001 + 1e6  # Mean ~1e6, var ~1e-6

# Naive (catastrophic cancellation): Var = E[X^2] - (E[X])^2
mean_x = np.mean(data)
var_naive = np.mean(data**2) - mean_x**2

# Two-pass (stable)
var_stable = np.var(data)

print(f"\nVariance (naive):  {var_naive:.10e}")
print(f"Variance (stable): {var_stable:.10e}")
print(f"Relative error: {abs(var_naive - var_stable) / var_stable:.2e}")
```

### 6.3 Safe Alternatives

| Dangerous | Safe Alternative |
|-----------|-----------------|
| `np.exp(x) - 1` for small $x$ | `np.expm1(x)` |
| `np.log(1 + x)` for small $x$ | `np.log1p(x)` |
| `E[X^2] - (E[X])^2` | Two-pass: `mean((x - mean(x))^2)` |
| `sqrt(x^2 + y^2)` | `np.hypot(x, y)` |

---

## 7. Mixed Precision Training

### 7.1 Why Use Lower Precision?

- **Memory**: fp16 uses half the memory of fp32, allowing larger batches and models
- **Speed**: Modern GPUs have dedicated fp16/bfloat16 tensor cores (2-4x throughput)
- **Communication**: Reduced bandwidth for distributed training

### 7.2 The Challenge

fp16 has limited range ($\pm 65504$) and precision ($\sim 3$ digits). This causes:
- **Overflow**: Gradient values > 65504 become Inf
- **Underflow**: Small gradients (< $6 \times 10^{-8}$) become zero
- **Precision loss**: Weight updates $w \leftarrow w - \eta g$ may round to $w$ if $\eta g \ll w$

### 7.3 Loss Scaling

**Loss scaling** multiplies the loss by a large factor $S$ before backprop:
1. Compute $L' = S \cdot L$ (in fp16)
2. Backprop to get $S \cdot \nabla L$ (larger gradients, less underflow)
3. Divide gradients by $S$ before the optimizer step (in fp32)

```python
# Simulate fp16 underflow
grad_fp32 = np.float32(1e-6)
grad_fp16 = np.float16(grad_fp32)
print(f"Gradient fp32: {grad_fp32}")
print(f"Gradient fp16: {grad_fp16}")  # Likely underflows to 0

# With loss scaling S = 1024
S = 1024
scaled_grad_fp16 = np.float16(grad_fp32 * S)
unscaled = np.float32(scaled_grad_fp16) / S
print(f"Scaled fp16: {scaled_grad_fp16}")
print(f"Unscaled: {unscaled}")

# Dynamic loss scaling simulation
print("\n--- Dynamic Loss Scaling ---")
S = 2**15  # Initial scale
scale_factor = 2
n_steps_ok = 0
scale_window = 200

for step in range(10):
    # Simulate gradient
    grad = np.random.randn() * 1e-5
    scaled = grad * S

    if np.isinf(np.float16(scaled)):
        S /= scale_factor
        print(f"Step {step}: Overflow detected, S -> {S:.0f}")
    else:
        n_steps_ok += 1
        if n_steps_ok >= scale_window:
            S *= scale_factor
            n_steps_ok = 0
            print(f"Step {step}: Scale up, S -> {S:.0f}")
        else:
            print(f"Step {step}: OK, S = {S:.0f}")
```

---

## 8. Debugging NaN and Inf

### 8.1 Common Causes

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| Loss = NaN | Division by zero, log(0), 0/0 | Add epsilon, use stable implementations |
| Loss = Inf | Overflow in exp | Use log-sum-exp, gradient clipping |
| Gradients = NaN | sqrt(0) gradient, log(negative) | Clamp inputs, add epsilon |
| Gradients = Inf | Division by near-zero | Add epsilon to denominator |
| Loss suddenly NaN | Learning rate too high | Reduce lr, use warmup |

### 8.2 Defensive Programming

```python
def safe_log(x, eps=1e-7):
    """Log that never sees zero."""
    return np.log(np.maximum(x, eps))

def safe_divide(a, b, eps=1e-8):
    """Division that never divides by zero."""
    return a / (b + eps * np.sign(b + 1e-30))

def safe_sqrt(x, eps=1e-8):
    """Square root with epsilon for gradient stability."""
    return np.sqrt(np.maximum(x, eps))

# Check for NaN/Inf in a training step
def check_tensor(x, name="tensor"):
    """Check for NaN/Inf and report."""
    if np.any(np.isnan(x)):
        print(f"WARNING: NaN in {name}")
        print(f"  Count: {np.sum(np.isnan(x))}/{x.size}")
        return False
    if np.any(np.isinf(x)):
        print(f"WARNING: Inf in {name}")
        print(f"  Count: {np.sum(np.isinf(x))}/{x.size}")
        return False
    return True

# Example
x = np.array([1, 0, -1, np.nan, np.inf])
check_tensor(x, "test_tensor")
```

---

## Summary

| Concept | Key Takeaway |
|---------|-------------|
| Float representation | fp32 has $\sim 7$ decimal digits; fp16 has $\sim 3$ and range $\pm 65504$ |
| Log-sum-exp | $\log \sum e^{z_i} = c + \log \sum e^{z_i - c}$ with $c = \max z_i$ |
| Stable softmax | Subtract max before exponentiating |
| Stable BCE | $\max(z,0) - yz + \log(1 + e^{-|z|})$ from logits |
| Catastrophic cancellation | Avoid subtracting nearly equal numbers; use `expm1`, `log1p` |
| Mixed precision | fp16 for speed; loss scaling to prevent underflow |
| Debug NaN/Inf | Add epsilon, clamp inputs, check tensors after each step |

---

## Exercises

1. Implement a numerically stable version of the log-sigmoid function $\log \sigma(z)$ that works for $z \in [-1000, 1000]$.
2. Show that the naive variance formula $E[X^2] - (E[X])^2$ fails for data with large mean and small variance, and implement the two-pass algorithm.
3. Implement the full softmax cross-entropy loss with gradient, from logits, in a numerically stable way. Verify with finite differences.
4. Write a function that detects the first step where NaN/Inf appears during a simulated training loop and diagnoses the cause.
5. Implement dynamic loss scaling: start with $S = 2^{15}$, halve on overflow, double every 200 clean steps.

---

**Next**: [11. Attention and Softmax Math](11_Attention_and_Softmax_Math.md)
