# Decision Tree

**Previous**: [Cross-Validation and Hyperparameters](./05_Cross_Validation_Hyperparameters.md) | **Next**: [Ensemble Learning - Bagging](./07_Ensemble_Bagging.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Describe the structure of a decision tree (root, internal nodes, leaves) and how predictions are made
2. Calculate entropy, Gini impurity, and information gain for a candidate split
3. Explain the CART algorithm and its greedy split-search process
4. Apply pre-pruning hyperparameters (max_depth, min_samples_split, min_samples_leaf) to control overfitting
5. Implement cost-complexity pruning (CCP) and select the optimal alpha via cross-validation
6. Interpret feature importance scores from a trained decision tree
7. Compare classification trees and regression trees and identify appropriate use cases for each

---

Decision trees are among the most intuitive machine learning algorithms -- they mirror the way humans make decisions by asking a sequence of yes/no questions. Their transparency makes them invaluable when you need to explain predictions to stakeholders, and they serve as the building blocks for powerful ensemble methods like Random Forest and Gradient Boosting.

---

## 1. Basic Concepts of Decision Trees

### 1.1 Tree Structure

```python
"""
Decision tree components:
1. Root Node: First split point
2. Internal Node: Intermediate split points
3. Leaf Node: Final prediction value
4. Split: Data splitting based on features
5. Depth: Distance from root to node

Example: Titanic survival prediction
          [Gender]
         /      \
      Male      Female
       |          |
    [Age]      Survived
    /    \
  <10   >=10
   |      |
 Survived  Died
"""
```

### 1.2 Basic Usage

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.tree import plot_tree, export_text
from sklearn.datasets import load_iris, load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Load data
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42
)

# Create and train model
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Predict
y_pred = clf.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

# Print tree structure
print("\nTree Structure:")
print(export_text(clf, feature_names=iris.feature_names))
```

### 1.3 Tree Visualization

```python
# Visualization
plt.figure(figsize=(20, 10))
plot_tree(
    clf,
    feature_names=iris.feature_names,
    class_names=iris.target_names,
    filled=True,
    rounded=True,
    fontsize=10
)
plt.title('Decision Tree - Iris Classification')
plt.tight_layout()
plt.show()

# Feature importance
print("\nFeature Importance:")
for name, importance in zip(iris.feature_names, clf.feature_importances_):
    print(f"  {name}: {importance:.4f}")
```

---

## 2. Split Criteria

### 2.1 Entropy

```python
import numpy as np

def entropy(y):
    """Calculate information entropy.

    Entropy measures disorder/uncertainty in the class distribution.
    A node with all samples from one class has entropy=0 (perfectly pure).
    A node with evenly mixed classes has maximum entropy — the worst possible
    split because we gain no information about which class a sample belongs to.
    Decision trees choose splits that maximally reduce entropy (information gain).
    """
    _, counts = np.unique(y, return_counts=True)
    probabilities = counts / len(y)  # fraction of samples belonging to each class
    # log2(p) ≤ 0 for p ∈ (0,1], so the negation makes H(y) ≥ 0 —
    # entropy is by convention a non-negative quantity
    # + 1e-10 prevents log(0) which is undefined — zero-probability classes
    # (absent from this node) would produce -inf and crash the computation
    return -np.sum(probabilities * np.log2(probabilities + 1e-10))

# Examples
y_pure = [0, 0, 0, 0, 0]  # Pure node
y_mixed = [0, 0, 1, 1, 1]  # Mixed node
y_balanced = [0, 0, 1, 1]  # Balanced node

print("Entropy examples:")
print(f"  Pure node: {entropy(y_pure):.4f}")  # 0
print(f"  Mixed node [2:3]: {entropy(y_mixed):.4f}")
print(f"  Balanced node [2:2]: {entropy(y_balanced):.4f}")  # 1 (maximum)
```

### 2.2 Gini Impurity

```python
def gini_impurity(y):
    """Calculate Gini impurity"""
    _, counts = np.unique(y, return_counts=True)
    probabilities = counts / len(y)  # each element = fraction of samples in that class
    # 1 - Σp² equals the probability that two randomly drawn samples belong
    # to *different* classes — pure nodes score 0, maximally mixed nodes score ~0.5
    return 1 - np.sum(probabilities ** 2)

print("\nGini impurity examples:")
print(f"  Pure node: {gini_impurity(y_pure):.4f}")  # 0
print(f"  Mixed node: {gini_impurity(y_mixed):.4f}")
print(f"  Balanced node: {gini_impurity(y_balanced):.4f}")  # 0.5 (maximum)

# Comparison: Entropy vs Gini
"""
- Gini: Faster computation, default
- Entropy: Tends toward more balanced trees
- In practice, little difference
"""
```

### 2.3 Information Gain

```python
def information_gain(parent, left_child, right_child, criterion='gini'):
    """Calculate information gain"""
    if criterion == 'gini':
        impurity_func = gini_impurity
    else:
        impurity_func = entropy

    # Weighted average impurity: child impurities are weighted by their sample
    # counts to account for uneven splits — a large pure child is worth more
    # than a small pure child, so the split quality is measured proportionally
    n = len(left_child) + len(right_child)
    n_left, n_right = len(left_child), len(right_child)

    weighted_impurity = (n_left / n) * impurity_func(left_child) + \
                       (n_right / n) * impurity_func(right_child)

    return impurity_func(parent) - weighted_impurity

# Example: Compare splits
parent = [0, 0, 0, 1, 1, 1]

# Split A: Good split
left_a = [0, 0, 0]
right_a = [1, 1, 1]

# Split B: Bad split
left_b = [0, 0, 1]
right_b = [0, 1, 1]

print("\nInformation gain comparison:")
print(f"  Split A (perfect): {information_gain(parent, left_a, right_a):.4f}")
print(f"  Split B (mixed): {information_gain(parent, left_b, right_b):.4f}")
```

---

## 3. CART Algorithm

### 3.1 Classification Tree

```python
from sklearn.tree import DecisionTreeClassifier

# Compare different criteria
criteria = ['gini', 'entropy', 'log_loss']

print("Classification Tree - Criterion Comparison:")
for criterion in criteria:
    clf = DecisionTreeClassifier(criterion=criterion, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"  {criterion}: Accuracy = {accuracy:.4f}, Depth = {clf.get_depth()}")
```

### 3.2 Regression Tree

```python
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load data
diabetes = load_diabetes()
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    diabetes.data, diabetes.target, test_size=0.2, random_state=42
)

# Regression tree (MSE criterion)
reg = DecisionTreeRegressor(criterion='squared_error', random_state=42)
reg.fit(X_train_r, y_train_r)
y_pred_r = reg.predict(X_test_r)

print("\nRegression Tree Results:")
print(f"  MSE: {mean_squared_error(y_test_r, y_pred_r):.4f}")
print(f"  R²: {r2_score(y_test_r, y_pred_r):.4f}")

# Other criteria
criteria_reg = ['squared_error', 'friedman_mse', 'absolute_error']

print("\nRegression Tree - Criterion Comparison:")
for criterion in criteria_reg:
    reg = DecisionTreeRegressor(criterion=criterion, random_state=42)
    reg.fit(X_train_r, y_train_r)
    y_pred = reg.predict(X_test_r)
    mse = mean_squared_error(y_test_r, y_pred)
    print(f"  {criterion}: MSE = {mse:.4f}")
```

### 3.3 Split Search Process

```python
"""
CART algorithm split process:

1. For all features:
   - Consider all possible split points
   - Calculate impurity reduction for each split

2. Select optimal split:
   - Choose (feature, split point) with maximum impurity reduction

3. Recursive splitting:
   - Repeat steps 1-2 for each child node
   - Stop when termination condition met

Termination conditions:
- Maximum depth reached
- Node sample count below minimum threshold
- Pure node reached (impurity = 0)
"""

# Simulate split process
def find_best_split(X, y, feature_idx):
    """Find optimal split point for single feature"""
    feature = X[:, feature_idx]
    sorted_indices = np.argsort(feature)

    best_gain = -1
    best_threshold = None

    for i in range(1, len(feature)):
        if feature[sorted_indices[i-1]] == feature[sorted_indices[i]]:
            continue

        threshold = (feature[sorted_indices[i-1]] + feature[sorted_indices[i]]) / 2
        left_mask = feature <= threshold

        if np.sum(left_mask) == 0 or np.sum(~left_mask) == 0:
            continue

        gain = information_gain(y, y[left_mask], y[~left_mask])

        if gain > best_gain:
            best_gain = gain
            best_threshold = threshold

    return best_threshold, best_gain

# Test
print("\nOptimal split point search:")
for i, name in enumerate(iris.feature_names):
    threshold, gain = find_best_split(iris.data, iris.target, i)
    print(f"  {name}: threshold={threshold:.2f}, gain={gain:.4f}")
```

---

## 4. Pruning

### 4.1 Pre-pruning

```python
# Limit tree growth with hyperparameters
clf_pruned = DecisionTreeClassifier(
    max_depth=3,              # Maximum depth
    min_samples_split=10,     # Minimum samples required to split
    min_samples_leaf=5,       # Minimum samples in leaf node
    max_features='sqrt',      # Maximum features to consider
    max_leaf_nodes=10,        # Maximum leaf nodes
    random_state=42
)
clf_pruned.fit(X_train, y_train)

print("Pre-pruning results:")
print(f"  Depth: {clf_pruned.get_depth()}")
print(f"  Leaf nodes: {clf_pruned.get_n_leaves()}")
print(f"  Accuracy: {accuracy_score(y_test, clf_pruned.predict(X_test)):.4f}")
```

### 4.2 Post-pruning - Cost Complexity Pruning

```python
# CCP (Cost Complexity Pruning)
path = clf.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas = path.ccp_alphas
impurities = path.impurities

print("CCP Alpha path:")
print(f"  Number of alpha values: {len(ccp_alphas)}")

# Generate tree for each alpha
clfs = []
for ccp_alpha in ccp_alphas:
    clf_ccp = DecisionTreeClassifier(ccp_alpha=ccp_alpha, random_state=42)
    clf_ccp.fit(X_train, y_train)
    clfs.append(clf_ccp)

# Node count and depth changes
train_scores = [clf.score(X_train, y_train) for clf in clfs]
test_scores = [clf.score(X_test, y_test) for clf in clfs]
n_leaves = [clf.get_n_leaves() for clf in clfs]
depths = [clf.get_depth() for clf in clfs]

# Visualization
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Alpha vs Accuracy
axes[0].plot(ccp_alphas, train_scores, marker='o', label='Train', drawstyle='steps-post')
axes[0].plot(ccp_alphas, test_scores, marker='o', label='Test', drawstyle='steps-post')
axes[0].set_xlabel('Alpha')
axes[0].set_ylabel('Accuracy')
axes[0].set_title('Alpha vs Accuracy')
axes[0].legend()

# Alpha vs Leaf nodes
axes[1].plot(ccp_alphas, n_leaves, marker='o', drawstyle='steps-post')
axes[1].set_xlabel('Alpha')
axes[1].set_ylabel('Number of Leaves')
axes[1].set_title('Alpha vs Number of Leaves')

# Alpha vs Depth
axes[2].plot(ccp_alphas, depths, marker='o', drawstyle='steps-post')
axes[2].set_xlabel('Alpha')
axes[2].set_ylabel('Depth')
axes[2].set_title('Alpha vs Depth')

plt.tight_layout()
plt.show()

# Select optimal alpha (cross-validation)
from sklearn.model_selection import cross_val_score

cv_scores = []
for ccp_alpha in ccp_alphas:
    clf_ccp = DecisionTreeClassifier(ccp_alpha=ccp_alpha, random_state=42)
    scores = cross_val_score(clf_ccp, X_train, y_train, cv=5)
    cv_scores.append(scores.mean())

best_idx = np.argmax(cv_scores)
best_alpha = ccp_alphas[best_idx]
print(f"\nOptimal Alpha: {best_alpha:.6f}")
print(f"Optimal CV score: {cv_scores[best_idx]:.4f}")
```

---

## 5. Decision Boundary Visualization

```python
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
import numpy as np

# Generate 2D data
X_2d, y_2d = make_classification(
    n_samples=200, n_features=2, n_redundant=0,
    n_informative=2, n_clusters_per_class=1, random_state=42
)

# Compare trees with different depths
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
depths = [1, 2, 3, 5, 10, None]

for ax, depth in zip(axes.flatten(), depths):
    clf = DecisionTreeClassifier(max_depth=depth, random_state=42)
    clf.fit(X_2d, y_2d)

    # Decision boundary
    x_min, x_max = X_2d[:, 0].min() - 0.5, X_2d[:, 0].max() + 0.5
    y_min, y_max = X_2d[:, 1].min() - 0.5, X_2d[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    ax.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu')
    ax.scatter(X_2d[:, 0], X_2d[:, 1], c=y_2d, edgecolors='black', cmap='RdYlBu')

    depth_str = depth if depth else 'None'
    ax.set_title(f'Max Depth = {depth_str}\nAccuracy = {clf.score(X_2d, y_2d):.3f}')

plt.tight_layout()
plt.show()
```

---

## 6. Advantages and Disadvantages of Decision Trees

### 6.1 Pros and Cons

```python
"""
Advantages:
1. Easy to interpret: Visualize decision-making process
2. Minimal preprocessing: No scaling or normalization needed
3. Nonlinear relationships: Can learn complex nonlinear patterns
4. Various data types: Handle both numerical and categorical
5. Fast prediction: O(log n) time complexity

Disadvantages:
1. Overfitting tendency: Deep trees easily overfit
2. Instability: Sensitive to small data changes
3. Optimization limitations: No global optimum guarantee (greedy)
4. Cannot extrapolate: Difficult to predict outside training range
5. Bias: Sensitive to class imbalance
"""
```

---

## 7. Hyperparameter Tuning

```python
from sklearn.model_selection import GridSearchCV

# Hyperparameter grid
param_grid = {
    'max_depth': [3, 5, 7, 10, None],
    'min_samples_split': [2, 5, 10, 20],
    'min_samples_leaf': [1, 2, 5, 10],
    'criterion': ['gini', 'entropy']
}

# Grid Search
grid_search = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)
grid_search.fit(X_train, y_train)

print("Hyperparameter tuning results:")
print(f"  Best parameters: {grid_search.best_params_}")
print(f"  Best CV score: {grid_search.best_score_:.4f}")
print(f"  Test score: {grid_search.score(X_test, y_test):.4f}")
```

---

## 8. Feature Importance

```python
# Train with full tree
clf = DecisionTreeClassifier(random_state=42)
clf.fit(iris.data, iris.target)

# Feature importance
importances = clf.feature_importances_
indices = np.argsort(importances)[::-1]

# Visualization
plt.figure(figsize=(10, 6))
plt.bar(range(len(importances)), importances[indices])
plt.xticks(range(len(importances)),
           [iris.feature_names[i] for i in indices], rotation=45)
plt.ylabel('Feature Importance')
plt.title('Decision Tree Feature Importance')
plt.tight_layout()
plt.show()

print("\nFeature importance ranking:")
for i, idx in enumerate(indices):
    print(f"  {i+1}. {iris.feature_names[idx]}: {importances[idx]:.4f}")
```

---

## Practice Problems

### Problem 1: Basic Classification
Train a decision tree on breast cancer data and evaluate.

```python
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, test_size=0.2, random_state=42
)

# Solution
clf = DecisionTreeClassifier(max_depth=5, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=cancer.target_names))
```

### Problem 2: Pruning
Find optimal alpha using CCP and prune the tree.

```python
# Solution
from sklearn.model_selection import cross_val_score

# Calculate CCP path
clf_full = DecisionTreeClassifier(random_state=42)
clf_full.fit(X_train, y_train)
path = clf_full.cost_complexity_pruning_path(X_train, y_train)

# Find optimal alpha with cross-validation
best_alpha = 0
best_score = 0
for alpha in path.ccp_alphas[::5]:  # Sample for efficiency
    clf = DecisionTreeClassifier(ccp_alpha=alpha, random_state=42)
    scores = cross_val_score(clf, X_train, y_train, cv=5)
    if scores.mean() > best_score:
        best_score = scores.mean()
        best_alpha = alpha

print(f"Optimal Alpha: {best_alpha:.6f}")
print(f"Optimal CV score: {best_score:.4f}")

clf_pruned = DecisionTreeClassifier(ccp_alpha=best_alpha, random_state=42)
clf_pruned.fit(X_train, y_train)
print(f"Test accuracy: {clf_pruned.score(X_test, y_test):.4f}")
```

### Problem 3: Regression Tree
Train a regression tree on diabetes data.

```python
from sklearn.datasets import load_diabetes
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

diabetes = load_diabetes()
X_train, X_test, y_train, y_test = train_test_split(
    diabetes.data, diabetes.target, test_size=0.2, random_state=42
)

# Solution
reg = DecisionTreeRegressor(max_depth=4, min_samples_leaf=10, random_state=42)
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)

print(f"MSE: {mean_squared_error(y_test, y_pred):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
print(f"R²: {r2_score(y_test, y_pred):.4f}")
```

---

## Summary

| Concept | Description | Use |
|---------|-------------|-----|
| Entropy | Measure of information uncertainty | Split criterion (criterion='entropy') |
| Gini Impurity | Probability of misclassification | Split criterion (criterion='gini') |
| Information Gain | Impurity reduction after split | Select optimal split |
| max_depth | Maximum tree depth | Prevent overfitting |
| min_samples_split | Minimum samples for split | Prevent overfitting |
| min_samples_leaf | Minimum samples in leaf node | Prevent overfitting |
| ccp_alpha | Cost-complexity pruning | Post-pruning |
| feature_importances_ | Feature importance | Feature selection |

### Decision Tree Checklist

1. Apply pruning to prevent overfitting
2. Use feature importance for interpretability
3. Consider ensemble methods (Random Forest) for instability
4. No scaling needed for numerical features
5. Categorical features require encoding (for sklearn)
