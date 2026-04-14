# Perceptron and Linear Classifiers

**Previous**: [Biological to Artificial Neurons](./01_Biological_to_Artificial_Neurons.md) | **Next**: [Activation Functions](./03_Activation_Functions.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Implement the perceptron learning algorithm from scratch
2. Prove why the perceptron converges for linearly separable data
3. Explain the XOR problem and why it broke single-layer perceptrons
4. Visualize the decision boundary learned by a perceptron
5. Distinguish between the perceptron and Adaline (adaptive linear neuron)
6. Connect the perceptron learning rule to gradient descent
7. Identify when a dataset is linearly separable
8. Explain why multi-layer networks solve the XOR problem

---

The perceptron, invented by Frank Rosenblatt in 1957, was the first algorithm that could genuinely *learn* from data. Unlike the McCulloch-Pitts neuron where you had to set the threshold by hand, the perceptron automatically adjusts its weights to classify inputs correctly. Understanding the perceptron -- its power and its limitations -- is essential for grasping why we need multi-layer neural networks.

---

## 1. The Perceptron Model

### 1.1 Architecture

```
    x1 ──w1──┐
             │
    x2 ──w2──┼──► z = Σ(wi·xi) + b ──► step(z) ──► ŷ ∈ {0, 1}
             │
    x3 ──w3──┘

    step(z) = 1 if z ≥ 0
              0 if z < 0
```

The perceptron is a single artificial neuron with:
- **Learnable weights** w1, w2, ..., wn
- **Learnable bias** b
- **Step function** as the activation (hard threshold)

### 1.2 Mathematical Formulation

```
z = w · x + b = Σ(wi · xi) + b

ŷ = step(z) = { 1  if z ≥ 0
               { 0  if z < 0
```

The decision boundary is the hyperplane where z = 0:

```
w · x + b = 0

In 2D: w1·x1 + w2·x2 + b = 0
       → x2 = -(w1/w2)·x1 - b/w2
       (a straight line!)
```

---

## 2. The Perceptron Learning Rule

### 2.1 The Algorithm

Rosenblatt's insight: update weights only when the perceptron makes an error.

```
Algorithm: Perceptron Learning
─────────────────────────────
Initialize weights w = 0, bias b = 0
For each epoch:
    For each training sample (x, y):
        1. Compute prediction: ŷ = step(w · x + b)
        2. Compute error: e = y - ŷ
        3. Update weights: w ← w + η · e · x
        4. Update bias:    b ← b + η · e

Where η is the learning rate (typically η = 1 for perceptron)
```

### 2.2 Understanding the Update Rule

The update `w ← w + η · e · x` is intuitive:

- **Correct prediction** (e = 0): No update. Leave weights alone.
- **Predicted 0, actual 1** (e = +1): Add x to w. Pull decision boundary toward x.
- **Predicted 1, actual 0** (e = -1): Subtract x from w. Push decision boundary away from x.

```
Case 1: y=1, ŷ=0 (missed positive)
    w_new = w + x    → moves boundary to include x

Case 2: y=0, ŷ=1 (false positive)
    w_new = w - x    → moves boundary to exclude x

Case 3: y=ŷ (correct)
    w_new = w        → no change
```

### 2.3 Python Implementation

```python
import numpy as np

class Perceptron:
    """Single-layer perceptron classifier."""

    def __init__(self, n_features, learning_rate=1.0):
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        self.lr = learning_rate

    def predict(self, x):
        z = np.dot(self.weights, x) + self.bias
        return 1 if z >= 0 else 0

    def train(self, X, y, n_epochs=100):
        """Train perceptron using the learning rule."""
        errors_per_epoch = []
        for epoch in range(n_epochs):
            errors = 0
            for xi, yi in zip(X, y):
                prediction = self.predict(xi)
                error = yi - prediction
                if error != 0:
                    self.weights += self.lr * error * xi
                    self.bias += self.lr * error
                    errors += 1
            errors_per_epoch.append(errors)
            if errors == 0:
                print(f"Converged at epoch {epoch + 1}")
                break
        return errors_per_epoch

# AND gate data
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 0, 0, 1])

p = Perceptron(n_features=2)
p.train(X, y)
print(f"Weights: {p.weights}, Bias: {p.bias}")
for xi, yi in zip(X, y):
    print(f"  {xi} -> {p.predict(xi)} (expected {yi})")
```

---

## 3. The Perceptron Convergence Theorem

### 3.1 Statement

**Theorem** (Rosenblatt, 1962): If the training data is linearly separable, the perceptron learning algorithm will converge in a finite number of steps.

### 3.2 Proof Sketch

Let **w*** be a weight vector that correctly separates the data with margin γ:
- For all positive examples: w* · x ≥ γ > 0
- For all negative examples: w* · x < 0

After k misclassification updates starting from w = 0:

**Lower bound on alignment**: w_k · w* grows linearly:
```
w_k · w* ≥ k · γ    (each update adds at least γ alignment)
```

**Upper bound on norm**: ||w_k||^2 grows linearly:
```
||w_k||^2 ≤ k · R^2    (where R = max ||x||)
```

By Cauchy-Schwarz: w_k · w* ≤ ||w_k|| · ||w*||

Combining:
```
k · γ ≤ ||w_k|| · ||w*|| ≤ √(k · R^2) · ||w*||

k ≤ (R · ||w*|| / γ)^2
```

The number of mistakes is bounded. The algorithm must converge.

### 3.3 What If Data Is Not Linearly Separable?

If the data cannot be separated by a hyperplane, the perceptron will **never converge**. It will oscillate indefinitely, never finding a perfect boundary. This is the fundamental limitation that led to the XOR crisis.

---

## 4. The XOR Problem

### 4.1 Why XOR Broke Everything

```
XOR Truth Table:          Scatter Plot:
x1  x2  y                x2
0   0   0                 1 ┤  ●(0,1)        ○(1,1)
0   1   1                   │    class=1        class=0
1   0   1                   │
1   1   0                 0 ┤  ○(0,0)        ●(1,0)
                             └──┤──────────┤──► x1
                                0          1

    ● = class 1,  ○ = class 0
    
    No single line can separate ● from ○ !
```

### 4.2 Proof of Non-Linear Separability

For a line w1·x1 + w2·x2 + b = 0 to classify XOR correctly, we need:

```
(0,0) → 0:   b < 0            ... (1)
(0,1) → 1:   w2 + b ≥ 0       ... (2)
(1,0) → 1:   w1 + b ≥ 0       ... (3)
(1,1) → 0:   w1 + w2 + b < 0  ... (4)
```

From (2) and (3): w1 + w2 + 2b ≥ 0
From (4): w1 + w2 < -b
Therefore: -b + 2b ≤ w1 + w2 + 2b and w1 + w2 < -b
So: b < -b → 2b < 0 → b < 0 (consistent with (1))
But from (2): w2 ≥ -b > 0, and from (3): w1 ≥ -b > 0
Then w1 + w2 ≥ -2b > 0, but (4) requires w1 + w2 + b < 0, i.e., w1 + w2 < -b
So w1 + w2 ≥ -2b and w1 + w2 < -b → -2b ≤ w1 + w2 < -b → -2b < -b → -b < 0 → b > 0

**Contradiction!** We derived both b < 0 and b > 0. No single perceptron can learn XOR.

### 4.3 The Multi-Layer Solution

XOR can be solved by combining two perceptrons in a hidden layer:

```
    x1 ──┬──► h1 = step(x1 + x2 - 0.5)    (OR gate)
         │
         ├──► h2 = step(-x1 - x2 + 1.5)   (NAND gate)
         │
    x2 ──┘
         
    h1, h2 ──► y = step(h1 + h2 - 1.5)     (AND gate)

    Result: y = XOR(x1, x2) = AND(OR(x1,x2), NAND(x1,x2))
```

```python
def xor_network(x1, x2):
    """XOR using two-layer perceptron network."""
    # Hidden layer
    h1 = 1 if (x1 + x2 - 0.5) >= 0 else 0   # OR
    h2 = 1 if (-x1 - x2 + 1.5) >= 0 else 0   # NAND
    # Output layer
    y = 1 if (h1 + h2 - 1.5) >= 0 else 0      # AND
    return y

for x1, x2 in [(0,0), (0,1), (1,0), (1,1)]:
    print(f"XOR({x1}, {x2}) = {xor_network(x1, x2)}")
```

This simple example demonstrates the power of **multi-layer networks** -- the foundation of deep learning.

---

## 5. Adaline: Adaptive Linear Neuron

### 5.1 From Perceptron to Adaline

Adaline (Widrow & Hoff, 1960) improves on the perceptron by using a **continuous loss function** instead of discrete error counting:

```
Perceptron:   update based on step(z) error       (discontinuous)
Adaline:      update based on z (before step)      (continuous, differentiable)
```

```
    x ──w──► z = w·x + b ──┬──► step(z) ──► ŷ   (classification)
                            │
                            └──► z  (used for weight update)

    Loss = (1/2)(y - z)^2    (MSE on pre-activation)
    ∂L/∂w = -(y - z) · x    (gradient)
```

### 5.2 Adaline Update Rule

```
w ← w + η · (y - z) · x     (gradient descent on MSE)
b ← b + η · (y - z)
```

The key difference from the perceptron: Adaline uses the **continuous** pre-activation z in the update, not the discrete prediction ŷ. This makes the updates smoother and connects directly to gradient descent.

### 5.3 Implementation

```python
class Adaline:
    """Adaptive Linear Neuron (Widrow-Hoff)."""

    def __init__(self, n_features, learning_rate=0.01):
        self.weights = np.random.randn(n_features) * 0.01
        self.bias = 0.0
        self.lr = learning_rate

    def net_input(self, x):
        return np.dot(self.weights, x) + self.bias

    def predict(self, x):
        return 1 if self.net_input(x) >= 0.0 else 0

    def train(self, X, y, n_epochs=100):
        losses = []
        for epoch in range(n_epochs):
            total_loss = 0.0
            for xi, yi in zip(X, y):
                z = self.net_input(xi)
                error = yi - z
                self.weights += self.lr * error * xi
                self.bias += self.lr * error
                total_loss += 0.5 * error ** 2
            losses.append(total_loss / len(y))
        return losses
```

---

## 6. Decision Boundaries Visualized

### 6.1 Linear Decision Boundary

For a 2D perceptron with weights w = [w1, w2] and bias b:

```
Decision boundary:  w1·x1 + w2·x2 + b = 0
Solving for x2:     x2 = -(w1/w2)·x1 - b/w2

Slope:      -w1/w2
Intercept:  -b/w2

The weight vector w is PERPENDICULAR to the decision boundary.

          ↑ x2
          │
          │  w = [w1, w2]
          │  ↗ (perpendicular to boundary)
    ──────┼──────────────── decision boundary
          │       ↗
          │      /
          └──────────────→ x1
```

### 6.2 Regions

```
w · x + b > 0   →  class 1 region
w · x + b = 0   →  decision boundary
w · x + b < 0   →  class 0 region
```

The weight vector **w** points toward the class-1 region. The bias **b** shifts the boundary away from the origin.

---

## 7. Connection to Linear Classifiers

The perceptron is the simplest member of a broader family of **linear classifiers**:

| Classifier | Loss Function | Activation | Update Rule |
|-----------|---------------|------------|-------------|
| Perceptron | 0-1 loss (implicit) | Step function | Error-driven |
| Adaline | MSE: (y-z)^2 | Identity → Step | Gradient descent |
| Logistic Regression | Cross-entropy | Sigmoid | Gradient descent |
| SVM | Hinge loss | Sign | Subgradient |

All of these share the same linear model z = w·x + b but differ in how they define "error" and how they update weights.

The logistic regression you learned in Machine Learning is essentially a perceptron with a **sigmoid activation** and a **cross-entropy loss** -- making it differentiable and probabilistic.

---

## 8. Summary

```
Key Takeaways
═══════════════════════════════════════════════════════
1. Perceptron: step(w·x + b), learns by correcting errors
2. Learning rule: w ← w + η·(y - ŷ)·x (update on mistakes)
3. Convergence: guaranteed for linearly separable data
4. XOR problem: single perceptron cannot learn XOR
5. Multi-layer networks solve XOR (foundation of deep learning)
6. Adaline: uses continuous loss → connects to gradient descent
7. Decision boundary: hyperplane w·x + b = 0
═══════════════════════════════════════════════════════
```

---

## Exercises

1. Train a perceptron on OR gate data and plot the decision boundary
2. Verify that the perceptron fails on XOR by running it for 1000 epochs
3. Build a two-layer network to solve XOR with manually chosen weights
4. Implement Adaline and compare its convergence curve with the perceptron

---

**Previous**: [Biological to Artificial Neurons](./01_Biological_to_Artificial_Neurons.md) | **Next**: [Activation Functions](./03_Activation_Functions.md)
