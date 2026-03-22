# Logistic Regression

**Previous**: [Linear Regression](./02_Linear_Regression.md) | **Next**: [Model Evaluation](./04_Model_Evaluation.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain why logistic regression is a classification algorithm despite its name
2. Describe the sigmoid function and how it maps linear outputs to probabilities
3. Implement binary logistic regression using scikit-learn and from scratch with gradient descent
4. Compare L1, L2, and Elastic Net regularization in the context of logistic regression
5. Distinguish between One-vs-Rest and Softmax approaches for multi-class classification
6. Apply threshold adjustment to balance precision and recall for a given problem
7. Demonstrate how class weighting addresses imbalanced datasets

---

Despite its name, logistic regression is a classification algorithm -- and one of the most important ones to understand. It predicts class probabilities by passing a linear combination of features through the sigmoid function, giving you an interpretable model that serves as the foundation for understanding neural networks and more advanced classifiers.

---

## 1. Binary Classification

### 1.1 Sigmoid Function

```python
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Visualize sigmoid function
z = np.linspace(-10, 10, 100)
plt.figure(figsize=(10, 5))
plt.plot(z, sigmoid(z), 'b-', linewidth=2)
plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.5)
plt.axvline(x=0, color='r', linestyle='--', alpha=0.5)
plt.xlabel('z')
plt.ylabel('σ(z)')
plt.title('Sigmoid Function')
plt.grid(True, alpha=0.3)
plt.ylim(-0.1, 1.1)
plt.show()

# Properties:
# - Output range: (0, 1) → interpretable as probability
# - 0.5 when z=0
# - 1 when z → ∞, 0 when z → -∞
```

### 1.2 Logistic Regression Model

```
P(y=1|X) = σ(θᵀX) = 1 / (1 + e^(-θᵀX))

Decision boundary:
- P(y=1|X) >= 0.5 → Predict class 1
- P(y=1|X) < 0.5 → Predict class 0
```

### 1.3 sklearn Implementation

```python
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# Breast cancer dataset (binary classification)
cancer = load_breast_cancer()
X, y = cancer.data, cancer.target
print(f"Classes: {cancer.target_names}")
print(f"Number of features: {X.shape[1]}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

# Predict
y_pred = model.predict(X_test_scaled)
y_proba = model.predict_proba(X_test_scaled)

# Evaluate
print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=cancer.target_names))

# Example prediction probabilities
print(f"\nFirst 5 sample prediction probabilities:")
for i in range(5):
    print(f"  Sample {i}: {cancer.target_names[0]}={y_proba[i][0]:.3f}, "
          f"{cancer.target_names[1]}={y_proba[i][1]:.3f} → Prediction: {cancer.target_names[y_pred[i]]}")
```

---

## 2. Cost Function and Optimization

### 2.1 Log Loss (Binary Cross-Entropy)

```python
# Cost function:
# J(θ) = -1/m * Σ[yᵢlog(ŷᵢ) + (1-yᵢ)log(1-ŷᵢ)]

from sklearn.metrics import log_loss

# Example
y_true = [0, 0, 1, 1]
y_proba = [0.1, 0.4, 0.35, 0.8]

loss = log_loss(y_true, y_proba)
print(f"Log Loss: {loss:.4f}")

# Perfect prediction
y_proba_perfect = [0.0, 0.0, 1.0, 1.0]
loss_perfect = log_loss(y_true, y_proba_perfect)
print(f"Perfect prediction Log Loss: {loss_perfect:.4f}")
```

### 2.2 Gradient Descent

```python
def logistic_regression_gd(X, y, learning_rate=0.1, n_iterations=1000):
    m, n = X.shape
    X_b = np.c_[np.ones((m, 1)), X]  # Add bias
    theta = np.zeros(n + 1)

    for _ in range(n_iterations):
        z = X_b @ theta
        h = sigmoid(z)
        gradient = (1/m) * X_b.T @ (h - y)
        theta = theta - learning_rate * gradient

    return theta

# Test
from sklearn.datasets import make_classification
X, y = make_classification(n_samples=1000, n_features=2, n_redundant=0,
                           n_informative=2, random_state=42)

theta = logistic_regression_gd(X, y)
print(f"Learned coefficients: {theta}")
```

---

## 3. Regularization

### 3.1 L2 Regularization (default)

```python
# penalty='l2' (default)
# C = 1/λ (smaller values mean stronger regularization)

Cs = [0.001, 0.01, 0.1, 1, 10, 100]

for C in Cs:
    model = LogisticRegression(C=C, max_iter=1000)
    model.fit(X_train_scaled, y_train)
    train_acc = model.score(X_train_scaled, y_train)
    test_acc = model.score(X_test_scaled, y_test)
    print(f"C={C:6}: Train={train_acc:.4f}, Test={test_acc:.4f}")
```

### 3.2 L1 Regularization (Lasso)

```python
# Feature selection effect
model_l1 = LogisticRegression(penalty='l1', solver='saga', C=0.1, max_iter=1000)
model_l1.fit(X_train_scaled, y_train)

# Number of non-zero coefficients
non_zero = np.sum(model_l1.coef_ != 0)
print(f"L1 regularization: non-zero coefficients = {non_zero}/{X.shape[1]}")
print(f"Accuracy: {model_l1.score(X_test_scaled, y_test):.4f}")
```

### 3.3 Elastic Net

```python
model_en = LogisticRegression(penalty='elasticnet', solver='saga',
                              l1_ratio=0.5, C=1, max_iter=1000)
model_en.fit(X_train_scaled, y_train)
print(f"Elastic Net accuracy: {model_en.score(X_test_scaled, y_test):.4f}")
```

---

## 4. Multi-class Classification

### 4.1 One-vs-Rest (OvR)

```python
from sklearn.datasets import load_iris
from sklearn.multiclass import OneVsRestClassifier

iris = load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# OvR (default for multi_class='ovr')
model_ovr = LogisticRegression(multi_class='ovr', max_iter=1000)
model_ovr.fit(X_train, y_train)

print(f"OvR accuracy: {model_ovr.score(X_test, y_test):.4f}")
print(f"Coefficient shape: {model_ovr.coef_.shape}")  # (3, 4) = num_classes x num_features
```

### 4.2 Softmax (Multinomial)

```python
# Softmax function: outputs probability for each class
# P(y=k|X) = exp(θₖᵀX) / Σexp(θⱼᵀX)

model_softmax = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
model_softmax.fit(X_train, y_train)

print(f"Softmax accuracy: {model_softmax.score(X_test, y_test):.4f}")

# Prediction probabilities
y_proba = model_softmax.predict_proba(X_test[:3])
print("\nPrediction probabilities (first 3 samples):")
for i, proba in enumerate(y_proba):
    print(f"  Sample {i}: {proba} → Prediction: {iris.target_names[np.argmax(proba)]}")
```

### 4.3 Comparison

```python
from sklearn.model_selection import cross_val_score

models = {
    'OvR': LogisticRegression(multi_class='ovr', max_iter=1000),
    'Multinomial': LogisticRegression(multi_class='multinomial', max_iter=1000)
}

for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=5)
    print(f"{name}: {scores.mean():.4f} (+/- {scores.std():.4f})")
```

---

## 5. Decision Boundary Visualization

```python
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt

# Generate 2D data
X, y = make_classification(n_samples=200, n_features=2, n_redundant=0,
                           n_informative=2, n_clusters_per_class=1,
                           random_state=42)

# Train model
model = LogisticRegression()
model.fit(X, y)

# Visualize decision boundary
def plot_decision_boundary(model, X, y):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(10, 6))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu')
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='black', cmap='RdYlBu')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Logistic Regression Decision Boundary')
    plt.show()

plot_decision_boundary(model, X, y)

# Visualize probability boundary
def plot_probability_boundary(model, X, y):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))

    Z = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(10, 6))
    plt.contourf(xx, yy, Z, levels=20, alpha=0.8, cmap='RdYlBu')
    plt.colorbar(label='P(y=1)')
    plt.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=2)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='black', cmap='RdYlBu')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Prediction Probability and Decision Boundary (0.5)')
    plt.show()

plot_probability_boundary(model, X, y)
```

---

## 6. Threshold Adjustment

```python
from sklearn.metrics import precision_recall_curve, roc_curve

# Prepare data
cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_s, y_train)

y_proba = model.predict_proba(X_test_s)[:, 1]

# Predict with various thresholds
thresholds = [0.3, 0.5, 0.7]

print("Performance by threshold:")
for thresh in thresholds:
    y_pred_thresh = (y_proba >= thresh).astype(int)
    from sklearn.metrics import precision_score, recall_score
    prec = precision_score(y_test, y_pred_thresh)
    rec = recall_score(y_test, y_pred_thresh)
    print(f"  threshold={thresh}: Precision={prec:.3f}, Recall={rec:.3f}")

# Precision-Recall curve
precision, recall, thresholds_pr = precision_recall_curve(y_test, y_proba)

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(thresholds_pr, precision[:-1], 'b-', label='Precision')
plt.plot(thresholds_pr, recall[:-1], 'r-', label='Recall')
plt.xlabel('Threshold')
plt.ylabel('Score')
plt.title('Precision/Recall vs Threshold')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(recall, precision)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.grid(True)

plt.tight_layout()
plt.show()
```

---

## 7. Handling Imbalanced Data

```python
from sklearn.datasets import make_classification

# Generate imbalanced data
X, y = make_classification(n_samples=1000, n_classes=2, weights=[0.9, 0.1],
                           n_features=10, random_state=42)

print(f"Class distribution: {np.bincount(y)}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Default model
model_default = LogisticRegression(max_iter=1000)
model_default.fit(X_train, y_train)

# class_weight='balanced'
model_balanced = LogisticRegression(class_weight='balanced', max_iter=1000)
model_balanced.fit(X_train, y_train)

# Compare
from sklearn.metrics import classification_report

print("=== Default Model ===")
print(classification_report(y_test, model_default.predict(X_test)))

print("=== class_weight='balanced' ===")
print(classification_report(y_test, model_balanced.predict(X_test)))
```

---

## Practice Problems

### Problem 1: Binary Classification
Train a logistic regression model on breast cancer data and compute the F1-score.

```python
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import f1_score

cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, test_size=0.2, random_state=42
)

# Solution
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_s, y_train)
y_pred = model.predict(X_test_s)

print(f"F1-score: {f1_score(y_test, y_pred):.4f}")
```

### Problem 2: Multi-class Classification
Perform 3-class classification on Iris data.

```python
from sklearn.datasets import load_iris

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42
)

# Solution
model = LogisticRegression(multi_class='multinomial', max_iter=1000)
model.fit(X_train, y_train)
print(f"Accuracy: {model.score(X_test, y_test):.4f}")
print(f"\nPrediction probability (first sample): {model.predict_proba(X_test[:1])}")
```

---

## Summary

| Concept | Description |
|---------|-------------|
| Sigmoid | Probability output (0~1) |
| Log Loss | Cost function (Binary Cross-Entropy) |
| OvR | Multi-class (One-vs-Rest) |
| Softmax | Multi-class (Multinomial) |
| C | Regularization strength (1/λ) |
| class_weight | Handle imbalanced data |
