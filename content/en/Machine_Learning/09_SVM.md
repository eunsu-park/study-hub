# Support Vector Machine (SVM)

**Previous**: [Ensemble Learning - Boosting](./08_Ensemble_Boosting.md) | **Next**: [kNN and Naive Bayes](./10_kNN_and_Naive_Bayes.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain the maximum margin principle and the role of support vectors in defining the decision boundary
2. Distinguish between hard margin and soft margin SVM and describe the effect of the C parameter
3. Describe the kernel trick and explain how it enables nonlinear classification without explicit feature transformation
4. Compare linear, polynomial, RBF, and sigmoid kernels and identify when each is appropriate
5. Apply feature scaling as a preprocessing step and explain why it is critical for SVM
6. Implement SVM for multiclass classification using One-vs-One and One-vs-Rest strategies
7. Demonstrate Support Vector Regression (SVR) and explain the epsilon-insensitive tube concept

---

Support Vector Machines find the decision boundary that maximally separates classes -- not just any boundary that works, but the one with the widest safety margin. This elegant geometric idea, combined with the kernel trick that handles nonlinear data, makes SVM one of the most theoretically grounded and practically effective algorithms for small to medium-sized datasets.

---

## 1. Core Concepts of SVM

### 1.1 Hyperplane and Margin

**Hyperplane**
- Decision boundary that separates different classes
- In 2D: Line, in 3D: Plane, in N-D: Hyperplane
- Equation: w·x + b = 0 (where w is weight vector, b is bias)

**Margin**
- Distance between hyperplane and nearest data points
- SVM finds the hyperplane with the **maximum margin**
- Larger margin → better generalization

### 1.2 Support Vectors

**Definition**
- Data points closest to the decision boundary
- Critical points that define the hyperplane
- Only support vectors affect the decision boundary

**Characteristics**
- Removing non-support vectors doesn't change the model
- SVM is robust to outliers far from the boundary
- Memory efficient (only stores support vectors)

### 1.3 Hard Margin vs Soft Margin

| Type | Description | Use Case |
|------|-------------|----------|
| **Hard Margin** | Strictly separates all training data | Linearly separable data only |
| **Soft Margin** | Allows some misclassification | Real-world data (with noise/outliers) |

**Soft Margin SVM** introduces:
- Slack variables ξ (xi): Allows some points to violate the margin
- C parameter: Controls trade-off between margin size and misclassification

---

## 2. Linear SVM

### 2.1 sklearn Implementation

```python
from sklearn.svm import SVC
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

# Generate linearly separable data
X, y = make_classification(n_samples=100, n_features=2, n_redundant=0,
                           n_informative=2, n_clusters_per_class=1, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear SVM
svm_clf = SVC(kernel='linear', C=1.0, random_state=42)
svm_clf.fit(X_train, y_train)

print(f"Train Accuracy: {svm_clf.score(X_train, y_train):.4f}")
print(f"Test Accuracy: {svm_clf.score(X_test, y_test):.4f}")
print(f"Number of Support Vectors: {len(svm_clf.support_vectors_)}")
```

### 2.2 Visualizing Decision Boundary

```python
def plot_svm_decision_boundary(svm_clf, X, y):
    # Create mesh grid
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))

    # Predict on mesh grid
    Z = svm_clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot
    plt.figure(figsize=(10, 6))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', edgecolors='k')

    # Plot support vectors
    plt.scatter(svm_clf.support_vectors_[:, 0],
                svm_clf.support_vectors_[:, 1],
                s=200, linewidth=1.5, facecolors='none', edgecolors='black',
                label='Support Vectors')

    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('SVM Decision Boundary')
    plt.legend()
    plt.show()

plot_svm_decision_boundary(svm_clf, X_train, y_train)
```

### 2.3 Effect of C Parameter

```python
# Compare different C values
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
C_values = [0.1, 1.0, 10.0]

for ax, C in zip(axes, C_values):
    svm = SVC(kernel='linear', C=C, random_state=42)
    svm.fit(X_train, y_train)

    # Plot decision boundary
    x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
    y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))
    Z = svm.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    ax.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='coolwarm', edgecolors='k')
    ax.scatter(svm.support_vectors_[:, 0], svm.support_vectors_[:, 1],
               s=200, facecolors='none', edgecolors='black')
    ax.set_title(f'C={C}, Support Vectors={len(svm.support_vectors_)}')
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')

plt.tight_layout()
plt.show()
```

**C Parameter Interpretation**:
- **Small C (e.g., 0.1)**: Wide margin, more misclassification allowed (high bias, low variance)
- **Large C (e.g., 10)**: Narrow margin, fewer misclassifications (low bias, high variance)

---

## 3. Kernel Trick

> **Lifting to a higher dimension.** Imagine data points on a flat table -- red and blue dots hopelessly intermingled with no straight line that separates them. Now imagine lifting some dots upward (based on their distance from the center, for example). In this 3D space, a flat plane can cleanly separate the colors. The kernel trick does exactly this -- it implicitly maps data to a higher-dimensional space where a linear separator exists, without ever computing the coordinates in that space. This is why SVMs can find non-linear decision boundaries while still solving a linear optimization problem.

### 3.1 Why Kernels?

**Problem**: Real-world data is often not linearly separable

**Solution**: Kernel trick
- Map data to higher-dimensional space where it becomes linearly separable
- No explicit transformation needed (computed via kernel function)

**Kernel Function**: K(x, x') = φ(x) · φ(x')
- Computes inner product in high-dimensional space
- Computationally efficient

### 3.2 Common Kernels

| Kernel | Formula | Use Case |
|--------|---------|----------|
| **Linear** | K(x, x') = x · x' | Linearly separable data |
| **Polynomial** | K(x, x') = (γx · x' + r)^d | Moderately non-linear data |
| **RBF (Radial Basis Function)** | K(x, x') = exp(-γ\|\|x - x'\|\|²) | Most common, highly non-linear data |
| **Sigmoid** | K(x, x') = tanh(γx · x' + r) | Neural network-like behavior |

### 3.3 RBF Kernel Example

```python
from sklearn.datasets import make_circles

# Generate non-linearly separable data (two concentric circles)
X, y = make_circles(n_samples=200, factor=0.5, noise=0.1, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# RBF SVM
rbf_svm = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
rbf_svm.fit(X_train, y_train)

print(f"Train Accuracy: {rbf_svm.score(X_train, y_train):.4f}")
print(f"Test Accuracy: {rbf_svm.score(X_test, y_test):.4f}")

# Visualize
plot_svm_decision_boundary(rbf_svm, X_train, y_train)
```

### 3.4 Gamma Parameter (RBF Kernel)

**Gamma (γ)**: Controls the influence of a single training example
- **Small gamma**: Far reach → smooth decision boundary (high bias)
- **Large gamma**: Close reach → complex decision boundary (high variance)

```python
# Compare different gamma values
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
gamma_values = [0.1, 1.0, 10.0]

for ax, gamma in zip(axes, gamma_values):
    svm = SVC(kernel='rbf', C=1.0, gamma=gamma, random_state=42)
    svm.fit(X_train, y_train)

    # Plot decision boundary
    x_min, x_max = X_train[:, 0].min() - 0.5, X_train[:, 0].max() + 0.5
    y_min, y_max = X_train[:, 1].min() - 0.5, X_train[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))
    Z = svm.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    ax.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='coolwarm', edgecolors='k')
    ax.set_title(f'gamma={gamma}, Accuracy={svm.score(X_test, y_test):.3f}')
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')

plt.tight_layout()
plt.show()
```

### 3.5 Polynomial Kernel Example

```python
# Polynomial kernel
poly_svm = SVC(kernel='poly', degree=3, C=1.0, gamma='scale', random_state=42)
poly_svm.fit(X_train, y_train)

print(f"Polynomial SVM Accuracy: {poly_svm.score(X_test, y_test):.4f}")
```

---

## 4. Multiclass Classification with SVM

SVM is originally a binary classifier. For multiclass problems:

### 4.1 Strategies

| Strategy | Description |
|----------|-------------|
| **One-vs-Rest (OvR)** | Train N classifiers (one per class vs all others) |
| **One-vs-One (OvO)** | Train N(N-1)/2 classifiers (one for each pair of classes) |

sklearn's `SVC` uses **One-vs-One** by default.

### 4.2 Example

```python
from sklearn.datasets import load_iris

# Load iris dataset (3 classes)
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42
)

# Multiclass SVM
multi_svm = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
multi_svm.fit(X_train, y_train)

print(f"Train Accuracy: {multi_svm.score(X_train, y_train):.4f}")
print(f"Test Accuracy: {multi_svm.score(X_test, y_test):.4f}")

# Confusion matrix
from sklearn.metrics import confusion_matrix, classification_report
y_pred = multi_svm.predict(X_test)
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))
```

---

## 5. Feature Scaling and Preprocessing

### 5.1 Why Scaling is Critical for SVM

SVM is **distance-based** and **sensitive to feature scales**.
- Features with large scales dominate the decision boundary
- Always standardize or normalize features before SVM

### 5.2 Standardization Example

```python
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Create pipeline with scaling
svm_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42))
])

# Without scaling
svm_no_scale = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
svm_no_scale.fit(X_train, y_train)

# With scaling
svm_pipeline.fit(X_train, y_train)

print("Without Scaling:")
print(f"  Train: {svm_no_scale.score(X_train, y_train):.4f}")
print(f"  Test:  {svm_no_scale.score(X_test, y_test):.4f}")

print("\nWith Scaling:")
print(f"  Train: {svm_pipeline.score(X_train, y_train):.4f}")
print(f"  Test:  {svm_pipeline.score(X_test, y_test):.4f}")
```

---

## 6. Hyperparameter Tuning

### 6.1 Grid Search for RBF SVM

```python
from sklearn.model_selection import GridSearchCV

# Define parameter grid
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [0.001, 0.01, 0.1, 1, 'scale', 'auto'],
    'kernel': ['rbf']
}

# Grid search
grid_search = GridSearchCV(
    SVC(random_state=42),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
print(f"Test score: {grid_search.score(X_test, y_test):.4f}")
```

### 6.2 Kernel Selection

```python
# Compare different kernels
kernels = ['linear', 'poly', 'rbf', 'sigmoid']
results = {}

for kernel in kernels:
    svm = SVC(kernel=kernel, C=1.0, gamma='scale', random_state=42)
    svm.fit(X_train, y_train)
    results[kernel] = {
        'train': svm.score(X_train, y_train),
        'test': svm.score(X_test, y_test)
    }

import pandas as pd
df_results = pd.DataFrame(results).T
print(df_results)
```

---

## 7. SVM for Regression (SVR)

### 7.1 SVR Concept

**Support Vector Regression (SVR)**: Finds a function that deviates from actual targets by at most ε (epsilon)
- Instead of maximizing margin between classes, maximizes margin around regression line
- Points within ε-tube don't contribute to loss

### 7.2 SVR Example

```python
from sklearn.svm import SVR
from sklearn.datasets import make_regression

# Generate regression data
X, y = make_regression(n_samples=100, n_features=1, noise=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train SVR
svr = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)
svr.fit(X_train, y_train)

# Predict
y_pred = svr.predict(X_test)

# Evaluate
from sklearn.metrics import mean_squared_error, r2_score
print(f"MSE: {mean_squared_error(y_test, y_pred):.4f}")
print(f"R²: {r2_score(y_test, y_pred):.4f}")

# Visualize
plt.figure(figsize=(10, 6))
plt.scatter(X_train, y_train, label='Train', alpha=0.6)
plt.scatter(X_test, y_test, label='Test', alpha=0.6)
X_plot = np.linspace(X.min(), X.max(), 300).reshape(-1, 1)
y_plot = svr.predict(X_plot)
plt.plot(X_plot, y_plot, 'r-', label='SVR prediction', linewidth=2)
plt.xlabel('X')
plt.ylabel('y')
plt.title('Support Vector Regression')
plt.legend()
plt.show()
```

---

## 8. Advantages and Disadvantages of SVM

### 8.1 Advantages

1. **Effective in High Dimensions**: Works well with many features
2. **Memory Efficient**: Only stores support vectors (not all training data)
3. **Versatile**: Different kernels for different data patterns
4. **Robust to Overfitting**: Especially in high-dimensional space (with proper C and gamma)
5. **Works with Small Datasets**: Effective even with limited training samples

### 8.2 Disadvantages

1. **Slow Training**: O(N²) to O(N³) time complexity for large datasets (N > 10,000)
2. **Sensitive to Feature Scaling**: Requires normalization/standardization
3. **No Probability Estimates**: Requires additional computation (`probability=True`)
4. **Difficult Hyperparameter Tuning**: C, gamma, kernel selection requires experimentation
5. **Black Box with Non-linear Kernels**: Hard to interpret decision process

---

## 9. Practical Tips

### 9.1 When to Use SVM

| Scenario | Recommendation |
|----------|----------------|
| Small to medium dataset (<10K samples) | SVM is a good choice |
| High-dimensional data (text, genomics) | SVM works well |
| Clear margin of separation | Linear SVM is efficient |
| Non-linear relationships | RBF or polynomial kernel |
| Need probability estimates | Use `probability=True` or consider Logistic Regression |
| Very large dataset (>100K samples) | Consider faster alternatives (Logistic Regression, SGDClassifier) |

### 9.2 Hyperparameter Tuning Guidelines

```
1. Always scale features (StandardScaler)
2. Start with RBF kernel
3. Use GridSearchCV with cross-validation
4. Tune C and gamma together:
   - Start: C=1, gamma='scale'
   - Try: C=[0.1, 1, 10, 100], gamma=[0.001, 0.01, 0.1, 1]
5. If RBF doesn't work well, try:
   - Linear kernel (for linearly separable data)
   - Polynomial kernel (for moderate non-linearity)
```

### 9.3 Common Mistakes

1. **Not scaling features**: Always use StandardScaler or MinMaxScaler
2. **Using default C=1 without tuning**: C depends on your data scale
3. **Ignoring computational cost**: SVM is slow on large datasets
4. **Forgetting to set `probability=True`**: If you need class probabilities

---

## 10. Exercises

### Exercise 1: Linear vs RBF Kernel
Load the wine dataset and compare linear and RBF SVM. Which performs better?

```python
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler

# Your code here
```

### Exercise 2: Hyperparameter Tuning
Use GridSearchCV to find optimal C and gamma for the breast cancer dataset.

```python
from sklearn.datasets import load_breast_cancer

# Your code here
```

### Exercise 3: SVM Regression
Create a polynomial regression problem and compare SVR with different kernels.

```python
# Your code here
```

### Exercise 4: Effect of Scaling
Train SVM on the iris dataset with and without scaling. Compare the results.

```python
# Your code here
```

---

## Summary

| Topic | Key Points |
|-------|------------|
| **Core Concept** | Maximum margin classifier, support vectors define boundary |
| **Linear SVM** | C parameter controls margin vs misclassification trade-off |
| **Kernel Trick** | Maps to high-dimensional space without explicit transformation |
| **RBF Kernel** | Most common, gamma controls complexity |
| **Multiclass** | One-vs-One (OvO) strategy by default in sklearn |
| **Scaling** | CRITICAL - always scale features before SVM |
| **Hyperparameters** | C (regularization), gamma (RBF influence), kernel type |
| **Use Cases** | Small to medium datasets, high-dimensional data, clear margins |
| **Limitations** | Slow on large datasets, requires scaling, black box with kernels |

**Key Takeaway**: SVM is powerful for small to medium-sized datasets with complex decision boundaries. Always scale features and tune C and gamma for best results.
