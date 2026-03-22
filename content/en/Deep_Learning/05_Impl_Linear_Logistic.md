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
