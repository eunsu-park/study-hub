# Ensemble Learning - Boosting

**Previous**: [Ensemble Learning - Bagging](./07_Ensemble_Bagging.md) | **Next**: [SVM](./09_SVM.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain how boosting differs from bagging in terms of training strategy and error reduction
2. Describe the AdaBoost algorithm including sample weight updates and model weighting
3. Implement Gradient Boosting and explain how it performs gradient descent in function space
4. Compare XGBoost, LightGBM, and CatBoost in terms of tree growth strategy, speed, and categorical handling
5. Apply early stopping and regularization techniques to prevent overfitting in boosting models
6. Demonstrate a systematic hyperparameter tuning order for boosting algorithms
7. Identify which boosting algorithm to choose based on dataset size, feature types, and speed requirements

---

Where bagging reduces variance by averaging independent models, boosting takes a fundamentally different approach: it reduces bias by building models sequentially, with each new model specifically targeting the mistakes of its predecessors. This strategy has produced some of the most accurate off-the-shelf algorithms in machine learning, including XGBoost and LightGBM, which dominate tabular data competitions.

---

> **Serial correction, not parallel voting.** If bagging is a committee where everyone votes independently, boosting is more like a relay team where each runner compensates for the previous one's weakness. The first model makes predictions; the second model focuses specifically on the cases the first model got wrong; the third focuses on what's still wrong after two attempts. Each new model is a "specialist" in the predecessor's failures, and the final prediction combines all specialists' input with appropriate weights.

## 1. Boosting Concepts

### 1.1 Key Principles

**Sequential Training**
- Train weak learners sequentially
- Each model corrects the errors of previous models
- Final prediction combines all models

**Sample Weighting**
- Increase weights on incorrectly classified samples
- Subsequent models focus on difficult cases
- Achieve high accuracy progressively

### 1.2 Differences from Bagging

| Feature | Bagging | Boosting |
|---------|---------|----------|
| Training | Parallel | Sequential |
| Sample Weighting | Equal | Increases for errors |
| Primary Goal | Reduce variance | Reduce bias |
| Overfitting Risk | Low | Higher (requires careful tuning) |
| Example | Random Forest | XGBoost, AdaBoost |

---

## 2. AdaBoost (Adaptive Boosting)

### 2.1 Algorithm Process

```
1. Initialize sample weights (1/N for all)
2. For each iteration t:
   a. Train weak learner h_t on weighted samples
   b. Calculate error rate ε_t
   c. Calculate model weight α_t = 0.5 * ln((1-ε_t) / ε_t)
   d. Update sample weights:
      - Increase weight for misclassified samples
      - Decrease weight for correctly classified samples
   e. Normalize weights
3. Final prediction: weighted vote of all weak learners
```

### 2.2 Weight Update Formula

```
New weight = Old weight × exp(α_t × prediction_error)

Where:
- prediction_error = 1 (incorrect), -1 (correct)
- α_t = model weight (higher when error rate is lower)
```

### 2.3 Implementation with sklearn

```python
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Generate data
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# AdaBoost Classifier
ada_clf = AdaBoostClassifier(
    base_estimator=DecisionTreeClassifier(max_depth=1),  # Weak learner (stump)
    n_estimators=50,          # Number of weak learners
    learning_rate=1.0,        # Weight update rate
    algorithm='SAMME.R',      # Algorithm ('SAMME', 'SAMME.R')
    random_state=42
)

ada_clf.fit(X_train, y_train)
print(f"Train Accuracy: {ada_clf.score(X_train, y_train):.4f}")
print(f"Test Accuracy: {ada_clf.score(X_test, y_test):.4f}")

# Feature importance
import matplotlib.pyplot as plt
import numpy as np

importances = ada_clf.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
plt.title("Feature Importances (AdaBoost)")
plt.bar(range(X.shape[1]), importances[indices])
plt.xlabel("Feature Index")
plt.ylabel("Importance")
plt.show()
```

### 2.4 AdaBoost Hyperparameters

- `n_estimators`: Number of weak learners (default: 50)
- `learning_rate`: Contribution weight of each weak learner (default: 1.0)
- `base_estimator`: Weak learner model (default: Decision Tree with depth 1)
- `algorithm`: 'SAMME' (discrete) or 'SAMME.R' (real, recommended)

---

## 3. Gradient Boosting

### 3.1 Core Concepts

**Gradient Descent in Function Space**
- Each model predicts the residuals (errors) of previous models
- Uses gradient descent to minimize loss function
- Powerful for regression and classification

**Process**
```
1. Initialize with a simple model (e.g., mean)
2. For each iteration t:
   a. Calculate residuals (negative gradient of loss)
   b. Train weak learner h_t to predict residuals
   c. Add h_t to ensemble with learning rate η
3. Final prediction = initial model + Σ(η × h_t)
```

### 3.2 sklearn GradientBoostingClassifier

```python
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

# Load data
data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42
)

# Gradient Boosting
gb_clf = GradientBoostingClassifier(
    n_estimators=100,         # Number of boosting stages
    learning_rate=0.1,        # Shrinkage rate
    max_depth=3,              # Max depth of trees
    subsample=0.8,            # Fraction of samples for training each tree
    min_samples_split=2,      # Minimum samples to split a node
    min_samples_leaf=1,       # Minimum samples in a leaf
    max_features='sqrt',      # Number of features to consider
    random_state=42
)

gb_clf.fit(X_train, y_train)
print(f"Train Accuracy: {gb_clf.score(X_train, y_train):.4f}")
print(f"Test Accuracy: {gb_clf.score(X_test, y_test):.4f}")
```

### 3.3 Key Hyperparameters

| Parameter | Description | Tuning Tips |
|-----------|-------------|-------------|
| `n_estimators` | Number of boosting stages | More is better, but watch for overfitting |
| `learning_rate` | Shrinkage rate for each tree | Lower values require more trees (trade-off) |
| `max_depth` | Maximum depth of trees | 3-5 typically works well |
| `subsample` | Fraction of samples for training | 0.5-0.8 reduces overfitting |
| `min_samples_split` | Minimum samples to split | Increase to prevent overfitting |
| `min_samples_leaf` | Minimum samples in leaf | Increase to prevent overfitting |
| `max_features` | Features to consider | 'sqrt' or 'log2' for high-dimensional data |

---

## 4. XGBoost (Extreme Gradient Boosting)

### 4.1 Advantages of XGBoost

1. **High Performance**: Parallel processing, cache optimization
2. **Regularization**: L1 (Lasso) and L2 (Ridge) regularization to prevent overfitting
3. **Tree Pruning**: Depth-first pruning with max_depth
4. **Missing Value Handling**: Automatically learns best direction for missing values
5. **Early Stopping**: Stops training when validation performance doesn't improve

### 4.2 Installation and Basic Usage

```bash
# Install XGBoost
pip install xgboost
```

```python
import xgboost as xgb
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load data
data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42
)

# XGBoost Classifier
xgb_clf = xgb.XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    subsample=0.8,
    colsample_bytree=0.8,    # Fraction of features to use per tree
    gamma=0,                 # Minimum loss reduction for split
    reg_alpha=0,             # L1 regularization
    reg_lambda=1,            # L2 regularization
    random_state=42
)

xgb_clf.fit(X_train, y_train)
y_pred = xgb_clf.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
```

### 4.3 Early Stopping

```python
# Early stopping with validation set
xgb_clf = xgb.XGBClassifier(
    n_estimators=1000,
    learning_rate=0.1,
    max_depth=3,
    early_stopping_rounds=10,  # Stop if no improvement for 10 rounds
    random_state=42
)

xgb_clf.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=True
)

print(f"Best iteration: {xgb_clf.best_iteration}")
print(f"Best score: {xgb_clf.best_score:.4f}")
```

### 4.4 Feature Importance Visualization

```python
import matplotlib.pyplot as plt

# Plot feature importance
xgb.plot_importance(xgb_clf, max_num_features=10)
plt.title("Feature Importance (XGBoost)")
plt.show()

# Get feature importance as array
importances = xgb_clf.feature_importances_
print("Top 5 features:")
for idx in importances.argsort()[::-1][:5]:
    print(f"Feature {idx}: {importances[idx]:.4f}")
```

---

## 5. LightGBM

### 5.1 Features of LightGBM

1. **Leaf-wise Growth**: Grows tree leaf-wise (not level-wise) for better accuracy
2. **Histogram-based Learning**: Bins continuous features for faster training
3. **GOSS (Gradient-based One-Side Sampling)**: Samples based on gradients
4. **EFB (Exclusive Feature Bundling)**: Bundles mutually exclusive features
5. **Categorical Feature Support**: Handles categorical features directly

### 5.2 Installation and Usage

```bash
# Install LightGBM
pip install lightgbm
```

```python
import lightgbm as lgb
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load data
data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42
)

# LightGBM Classifier
lgb_clf = lgb.LGBMClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=-1,            # No limit (use num_leaves instead)
    num_leaves=31,           # Maximum number of leaves
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0,             # L1 regularization
    reg_lambda=1,            # L2 regularization
    random_state=42
)

lgb_clf.fit(X_train, y_train)
y_pred = lgb_clf.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
```

### 5.3 LightGBM with Categorical Features

```python
import pandas as pd
import lightgbm as lgb

# Example with categorical features
df = pd.DataFrame({
    'cat_feature': ['A', 'B', 'A', 'C', 'B'],
    'num_feature': [1.0, 2.0, 3.0, 4.0, 5.0],
    'target': [0, 1, 0, 1, 1]
})

# Specify categorical features
lgb_clf = lgb.LGBMClassifier(random_state=42)
lgb_clf.fit(
    df[['cat_feature', 'num_feature']],
    df['target'],
    categorical_feature=['cat_feature']  # Specify categorical features
)
```

---

## 6. Comparison: XGBoost vs LightGBM vs CatBoost

| Feature | XGBoost | LightGBM | CatBoost |
|---------|---------|----------|----------|
| Tree Growth | Level-wise | Leaf-wise | Symmetric (level-wise) |
| Speed | Fast | Fastest | Moderate |
| Memory Usage | Moderate | Low | Moderate |
| Categorical Handling | Manual encoding | Supported | Best support |
| Overfitting Risk | Moderate | Higher (leaf-wise) | Lower |
| Tuning Difficulty | Moderate | Moderate | Easier (good defaults) |
| Use Case | General purpose | Large datasets, speed critical | Categorical features, ease of use |

### 6.1 CatBoost Example

```bash
# Install CatBoost
pip install catboost
```

```python
from catboost import CatBoostClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42
)

# CatBoost Classifier
cat_clf = CatBoostClassifier(
    iterations=100,
    learning_rate=0.1,
    depth=3,
    verbose=False,
    random_state=42
)

cat_clf.fit(X_train, y_train)
print(f"Accuracy: {cat_clf.score(X_test, y_test):.4f}")
```

---

## 7. Hyperparameter Tuning for Boosting Models

### 7.1 Grid Search for XGBoost

```python
from sklearn.model_selection import GridSearchCV
import xgboost as xgb

param_grid = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.3],
    'max_depth': [3, 5, 7],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0]
}

xgb_clf = xgb.XGBClassifier(random_state=42)
grid_search = GridSearchCV(
    xgb_clf, param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best score: {grid_search.best_score_:.4f}")
```

### 7.2 Randomized Search for LightGBM

```python
from sklearn.model_selection import RandomizedSearchCV
import lightgbm as lgb
from scipy.stats import randint, uniform

param_dist = {
    'n_estimators': randint(50, 300),
    'learning_rate': uniform(0.01, 0.3),
    'num_leaves': randint(20, 100),
    'max_depth': randint(3, 10),
    'subsample': uniform(0.6, 0.4),
    'colsample_bytree': uniform(0.6, 0.4)
}

lgb_clf = lgb.LGBMClassifier(random_state=42)
random_search = RandomizedSearchCV(
    lgb_clf, param_dist,
    n_iter=50,           # Number of parameter combinations to try
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1,
    random_state=42
)

random_search.fit(X_train, y_train)
print(f"Best parameters: {random_search.best_params_}")
print(f"Best score: {random_search.best_score_:.4f}")
```

---

## 8. Preventing Overfitting in Boosting

### 8.1 Regularization Techniques

```python
# Example with multiple regularization techniques
xgb_clf = xgb.XGBClassifier(
    n_estimators=100,
    learning_rate=0.05,      # Lower learning rate
    max_depth=3,             # Limit tree depth
    min_child_weight=3,      # Minimum sum of weights in child node
    gamma=0.1,               # Minimum loss reduction for split
    subsample=0.8,           # Row sampling
    colsample_bytree=0.8,    # Column sampling
    reg_alpha=0.1,           # L1 regularization
    reg_lambda=1.0,          # L2 regularization
    random_state=42
)
```

### 8.2 Early Stopping

```python
# Early stopping to prevent overfitting
xgb_clf = xgb.XGBClassifier(
    n_estimators=1000,
    learning_rate=0.1,
    early_stopping_rounds=50,  # Stop if no improvement for 50 rounds
    random_state=42
)

xgb_clf.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_test, y_test)],
    eval_metric='logloss',
    verbose=10
)
```

---

## 9. Practical Tips

### 9.1 When to Use Which Algorithm

| Scenario | Recommended Algorithm |
|----------|----------------------|
| Small dataset (<10K rows) | Gradient Boosting, AdaBoost |
| Large dataset (>100K rows) | LightGBM |
| Many categorical features | CatBoost |
| Need feature importance | XGBoost, LightGBM |
| Need high interpretability | Gradient Boosting (fewer trees) |
| Speed is critical | LightGBM |
| Balanced performance | XGBoost (most versatile) |

### 9.2 Hyperparameter Tuning Order

```
1. Fix n_estimators to a high value (e.g., 1000)
2. Tune learning_rate (start with 0.1)
3. Tune tree-specific parameters (max_depth, num_leaves, min_child_weight)
4. Tune sampling parameters (subsample, colsample_bytree)
5. Tune regularization parameters (gamma, reg_alpha, reg_lambda)
6. Lower learning_rate and increase n_estimators for final model
```

### 9.3 Common Mistakes to Avoid

1. **Not using early stopping**: Always use validation set with early stopping
2. **Ignoring feature scaling**: While tree-based models don't require scaling, it can help with convergence
3. **Default hyperparameters**: Always tune for your specific dataset
4. **Overfitting on small datasets**: Use stronger regularization
5. **Not handling imbalanced data**: Use `scale_pos_weight` or `class_weight`

---

## 10. Exercises

### Exercise 1: AdaBoost vs Gradient Boosting
Compare AdaBoost and Gradient Boosting on the iris dataset. Which performs better?

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier

# Your code here
```

### Exercise 2: XGBoost Hyperparameter Tuning
Load the wine dataset and use GridSearchCV to find optimal hyperparameters for XGBoost.

```python
from sklearn.datasets import load_wine
from sklearn.model_selection import GridSearchCV
import xgboost as xgb

# Your code here
```

### Exercise 3: LightGBM with Early Stopping
Train a LightGBM model on the digits dataset with early stopping. Plot training and validation curves.

```python
from sklearn.datasets import load_digits
import lightgbm as lgb
import matplotlib.pyplot as plt

# Your code here
```

### Exercise 4: Feature Importance Comparison
Compare feature importances from XGBoost, LightGBM, and Random Forest on the same dataset.

```python
# Your code here
```

---

## Summary

| Topic | Key Points |
|-------|------------|
| **Boosting Basics** | Sequential training, error correction, sample weighting |
| **AdaBoost** | Adaptive weighting, weak learners, SAMME algorithm |
| **Gradient Boosting** | Gradient descent in function space, residual prediction |
| **XGBoost** | Regularization, parallel processing, early stopping |
| **LightGBM** | Leaf-wise growth, histogram-based, fastest training |
| **CatBoost** | Best categorical handling, symmetric trees, easy to use |
| **Tuning** | learning_rate ↔ n_estimators trade-off, regularization |
| **Overfitting** | Early stopping, regularization, sampling, tree depth |

**Key Takeaway**: Boosting models are powerful but require careful tuning. Start with XGBoost for general use, use LightGBM for large datasets, and CatBoost for categorical features.
