# 05. Linear & Logistic Regression

[Previous: Training Techniques](./04_Training_Techniques.md) | [Next: Multi-Layer Perceptron (MLP)](./06_Impl_MLP.md)

---

## Overview

Linear regression and logistic regression are the most fundamental building blocks of deep learning. Each layer of a neural network is essentially a combination of linear transformation + nonlinear activation.

## Learning Objectives

1. **Mathematical Understanding**
   - Gradient Descent principles
   - Loss Functions (MSE, Cross-Entropy)
   - Matrix differentiation

2. **Implementation Skills**
   - Direct implementation of Forward/Backward pass
   - Weight initialization
   - Writing training loops

3. **Practice**
   - MNIST binary classification
   - Overfitting/regularization experiments

---

## Theory & Principles

Linear and logistic regression are the simplest models in this course, and that simplicity is exactly why they deserve a careful theoretical treatment. Both have closed-form or near-closed-form solutions, both are convex, and the way you implement them by hand is exactly the template that scales up to every later layer of every later network. Understanding the math here makes everything that follows feel familiar.

This section covers:

- **A.** Linear regression: closed-form solution vs gradient descent
- **B.** Logistic regression as maximum likelihood under a Bernoulli model
- **C.** Why MSE for regression and cross-entropy for classification, not the reverse
- **D.** Convexity, uniqueness, and what changes when you add hidden layers

### A. Linear Regression: Closed Form vs Gradient Descent

Linear regression minimizes the mean squared error:

```
L(w) = (1 / 2N) * ||X w - y||^2
```

Setting `\nabla_w L = 0` yields the **normal equations** and a closed-form solution:

```
\nabla_w L = (1 / N) X^T (X w - y) = 0
=>  w* = (X^T X)^{-1} X^T y                  (when X^T X is invertible)
```

Why use gradient descent at all if a closed form exists? Three reasons:

1. **Cost.** `(X^T X)^{-1}` is `O(d^3)` for `d` features. With `d = 10^6`, this is impossible. Gradient descent costs `O(N * d)` per step.
2. **Memory.** Forming `X^T X` requires `O(d^2)` memory; for high-dimensional features (images, text), this exceeds RAM.
3. **Generality.** Gradient descent extends unchanged to non-linear models. The closed form does not.

So the closed form is the *target* of gradient descent: as steps go to zero and iterations to infinity, `w_t -> w*`.

### B. Logistic Regression as Maximum Likelihood

Logistic regression assumes the label `y \in {0, 1}` is Bernoulli-distributed conditional on `x`:

```
p(y=1 | x) = \sigma(w^T x + b),       \sigma(z) = 1 / (1 + e^{-z})
```

The negative log-likelihood (NLL) over a dataset is:

```
NLL(w) = - sum_i [ y_i log p_i + (1 - y_i) log(1 - p_i) ]
```

This is exactly **binary cross-entropy**. So minimizing cross-entropy *is* maximum-likelihood estimation under the Bernoulli model. The gradient has a remarkably clean form:

```
\nabla_w NLL = sum_i (p_i - y_i) x_i
```

Notice the term `(p_i - y_i)`: it is the prediction error in probability space. The sigmoid's derivative cancels neatly with the log's derivative — this is why no `\sigma'` term appears in the gradient.

### C. Loss Function Choice: MSE vs Cross-Entropy

Why MSE for continuous targets and cross-entropy for class labels?

**MSE for regression** comes from assuming Gaussian noise: `y = w^T x + \epsilon` with `\epsilon ~ N(0, \sigma^2)`. The NLL is `(1 / 2 \sigma^2) (y - w^T x)^2 + const`, which is MSE up to a constant.

**Cross-entropy for classification** comes from the Bernoulli/categorical NLL above. If you use MSE on classification with sigmoid output, the gradient is:

```
\nabla_w MSE = (p - y) * \sigma'(z) * x = (p - y) * p (1 - p) * x
```

The extra factor `p (1 - p)` vanishes when `p \to 0` or `p \to 1`, even when the prediction is *wrong* (e.g., `p = 0.99` but `y = 0`). The gradient flatlines exactly when you most want it to push back hard. Cross-entropy avoids this — it has no such damping factor — and is therefore strictly preferred.

### D. Convexity and What Hidden Layers Change

Both linear regression's MSE and logistic regression's NLL are **convex** in `w`. Convexity guarantees that any local minimum is the global minimum, so gradient descent (with appropriate step size) converges to the optimum from any starting point. There is essentially nothing to tune.

Adding even one hidden layer with a nonlinearity destroys convexity. The loss surface becomes a non-convex landscape with many local minima, saddle points, and plateaus. This is why deep learning needs:

- Careful initialization (so you start in a "good" region),
- Adaptive optimizers (Adam) instead of vanilla GD,
- Regularization (dropout, BN) to flatten the landscape,
- Tricks like learning-rate warmup that have no analogue in convex optimization.

The simplicity of this lesson's models is the last time everything is guaranteed to "just work."

### From Theory to the Code Below

| Theory concept | Code construct in this lesson |
|----------------|-------------------------------|
| MSE gradient `(1/N) X^T (X w - y)` | `dL_dw = X.T @ (pred - y) / N` |
| Closed form `w* = (X^T X)^{-1} X^T y` | Optional check via `np.linalg.lstsq` |
| Sigmoid + BCE clean gradient | `dL_dw = X.T @ (p - y)` (no `\sigma'` factor) |
| Convexity guarantee | The fact that both models converge from any init |

---


## Mathematical Background

### 1. Linear Regression

```
Model:    ŷ = Xw + b
Loss:     L = (1/2n) Σ(y - ŷ)²  (MSE)

Gradients:
∂L/∂w = (1/n) X^T (ŷ - y)
∂L/∂b = (1/n) Σ(ŷ - y)

Update:
w ← w - η × ∂L/∂w
b ← b - η × ∂L/∂b
```

### 2. Logistic Regression

```
Model:    z = Xw + b
          ŷ = σ(z) = 1/(1 + e^(-z))

Loss:     L = -(1/n) Σ[y·log(ŷ) + (1-y)·log(1-ŷ)]  (BCE)

Gradients:
∂L/∂w = (1/n) X^T (ŷ - y)  ← Surprisingly, same form as Linear!
∂L/∂b = (1/n) Σ(ŷ - y)
```

---

## File Structure

```
01_Linear_Logistic/
├── README.md                 # This file
├── theory.md                 # Detailed theory (mathematical derivations)
├── numpy/
│   ├── linear_numpy.py       # Linear Regression (NumPy)
│   ├── logistic_numpy.py     # Logistic Regression (NumPy)
│   └── test_numpy.py         # Unit tests
├── pytorch_lowlevel/
│   ├── linear_lowlevel.py    # Using PyTorch basic ops
│   └── logistic_lowlevel.py
├── paper/
│   └── linear_paper.py       # Clean nn.Module implementation
└── exercises/
    ├── 01_regularization.md  # Add L1/L2 regularization
    └── 02_softmax.md         # Extend to Softmax
```

---

## Quick Start

### Running NumPy Implementation

```bash
cd numpy/
python linear_numpy.py      # Train linear regression
python logistic_numpy.py    # Train logistic regression
python test_numpy.py        # Run tests
```

### Running PyTorch Implementation

```bash
cd pytorch_lowlevel/
python linear_lowlevel.py
```

---

## Core Concepts

### 1. Gradient Descent

```python
# Basic algorithm
for epoch in range(n_epochs):
    # Forward
    y_pred = model.forward(X)

    # Loss
    loss = compute_loss(y, y_pred)

    # Backward (compute gradients)
    gradients = compute_gradients(y, y_pred)

    # Update
    model.weights -= learning_rate * gradients
```

### 2. Matrix Differentiation (Important!)

```
∂(Xw)/∂w = X^T
∂(w^T X^T)/∂w = X
∂(||Xw - y||²)/∂w = 2 X^T (Xw - y)
```

### 3. Sigmoid and Its Derivative

```python
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    s = sigmoid(z)
    return s * (1 - s)  # σ(z)(1 - σ(z))
```

---

## Practice Problems

### Basic
1. Implement Linear Regression without bias
2. Observe convergence speed with different learning rates (lr)
3. Compare Batch vs Stochastic Gradient Descent

### Intermediate
1. Add L2 regularization (Ridge)
2. Add L1 regularization (Lasso)
3. Implement Mini-batch GD

### Advanced
1. Implement Momentum, Adam optimizers
2. Implement Early Stopping
3. Extend to Softmax Regression (multi-class)

---

## References

- CS229 (Stanford) Lecture Notes
- Deep Learning Book Chapter 5, 6
- [Coursera ML - Andrew Ng](https://www.coursera.org/learn/machine-learning)
