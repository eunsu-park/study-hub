# Ensemble Learning - Bagging

**Previous**: [Decision Trees](./06_Decision_Trees.md) | **Next**: [Ensemble Learning - Boosting](./08_Ensemble_Boosting.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain the ensemble learning principle and why combining multiple weak learners produces a strong learner
2. Describe how bootstrap sampling works and calculate the expected out-of-bag (OOB) ratio
3. Implement a bagging classifier from scratch using bootstrap sampling and majority voting
4. Compare Random Forest with standard bagging and explain how random feature selection reduces tree correlation
5. Interpret both impurity-based and permutation-based feature importance scores
6. Apply OOB error as a built-in validation estimate without a separate holdout set
7. Distinguish between Random Forest, Extra Trees, and Voting classifiers and identify appropriate use cases

---

A single decision tree is fast and interpretable but fragile -- small changes in the data can produce a completely different tree. Bagging solves this instability by training many trees on slightly different random samples and averaging their predictions, dramatically reducing variance while preserving the ability to capture complex patterns.

---

> **The wisdom of a committee.** Imagine asking 100 people to guess the number of jelly beans in a jar. Each person's guess may be wildly off, but the *average* of all guesses is remarkably close to the truth -- this is the "wisdom of crowds" effect. Bagging (Bootstrap Aggregating) applies the same principle: train many diverse models on slightly different random samples of the data, then average their predictions. Each individual model may overfit to its sample, but the ensemble cancels out the individual errors.

## 1. Basic Concepts of Ensemble Learning

### 1.1 What is Ensemble?

```python
"""
Ensemble Learning:
- Combine multiple weak learners to create a strong learner
- "Wisdom of Crowds"

Main types of ensembles:
1. Bagging: Parallel learning, reduce variance
   - Random Forest
   - Bagging Classifier/Regressor

2. Boosting: Sequential learning, reduce bias
   - AdaBoost
   - Gradient Boosting
   - XGBoost, LightGBM

3. Stacking: Meta model learning
   - Use predictions from various models as input

4. Voting: Simple voting
   - Hard Voting, Soft Voting
"""
```

### 1.2 Principle of Bagging

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

# Bootstrap sampling visualization
np.random.seed(42)
original_data = np.arange(10)

print("Bootstrap sampling example:")
print(f"Original data: {original_data}")

for i in range(3):
    bootstrap_sample = np.random.choice(original_data, size=len(original_data), replace=True)
    oob = set(original_data) - set(bootstrap_sample)
    print(f"Sample {i+1}: {bootstrap_sample} (OOB: {oob})")

# OOB ratio in bootstrap samples
"""
Expected OOB ratio:
- Probability each sample is not selected = (1 - 1/n)^n
- As n increases → e^(-1) ≈ 0.368 (about 37%)
- Each model uses only about 63% of original data
"""

n = 1000
selected = np.zeros(n)
for _ in range(n):
    idx = np.random.randint(0, n)
    selected[idx] = 1
oob_ratio = 1 - np.mean(selected)
print(f"\nExperimental OOB ratio: {oob_ratio:.4f}")
print(f"Theoretical OOB ratio: {1/np.e:.4f}")
```

---

## 2. Implementing Bagging from Scratch

```python
from sklearn.base import clone

class SimpleBagging:
    """Simple bagging implementation"""

    def __init__(self, base_estimator, n_estimators=10, random_state=None):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.estimators_ = []
        self.oob_indices_ = []

    def fit(self, X, y):
        np.random.seed(self.random_state)
        n_samples = len(X)
        self.estimators_ = []
        self.oob_indices_ = []

        for _ in range(self.n_estimators):
            # Bootstrap sampling
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            oob_indices = list(set(range(n_samples)) - set(indices))

            X_bootstrap = X[indices]
            y_bootstrap = y[indices]

            # Train model
            estimator = clone(self.base_estimator)
            estimator.fit(X_bootstrap, y_bootstrap)

            self.estimators_.append(estimator)
            self.oob_indices_.append(oob_indices)

        return self

    def predict(self, X):
        # Collect predictions from each model
        predictions = np.array([est.predict(X) for est in self.estimators_])
        # Majority voting
        return np.apply_along_axis(
            lambda x: np.bincount(x.astype(int)).argmax(),
            axis=0,
            arr=predictions
        )

    def predict_proba(self, X):
        # Average probabilities
        probas = np.array([est.predict_proba(X) for est in self.estimators_])
        return np.mean(probas, axis=0)

# Test
X, y = make_classification(n_samples=500, n_features=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Single tree vs bagging
single_tree = DecisionTreeClassifier(random_state=42)
single_tree.fit(X_train, y_train)

bagging = SimpleBagging(DecisionTreeClassifier(), n_estimators=10, random_state=42)
bagging.fit(X_train, y_train)

print("Bagging effect comparison:")
print(f"  Single decision tree: {single_tree.score(X_test, y_test):.4f}")
print(f"  Bagging (10 trees): {np.mean(bagging.predict(X_test) == y_test):.4f}")
```

---

## 3. sklearn's BaggingClassifier

```python
from sklearn.ensemble import BaggingClassifier, BaggingRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Use BaggingClassifier
bagging_clf = BaggingClassifier(
    estimator=DecisionTreeClassifier(),
    n_estimators=100,
    max_samples=1.0,        # Bootstrap sample size (ratio)
    max_features=1.0,       # Feature ratio to use per model
    bootstrap=True,         # Use bootstrap sampling
    bootstrap_features=False,  # Feature bootstrap
    oob_score=True,         # Calculate OOB score
    n_jobs=-1,              # Parallel processing
    random_state=42
)

bagging_clf.fit(X_train, y_train)
y_pred = bagging_clf.predict(X_test)

print("BaggingClassifier results:")
print(f"  Training accuracy: {bagging_clf.score(X_train, y_train):.4f}")
print(f"  Test accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"  OOB score: {bagging_clf.oob_score_:.4f}")
```

### 3.1 Performance vs Number of Models

```python
# Performance change with increasing number of models
n_estimators_range = [1, 5, 10, 20, 50, 100, 200]
train_scores = []
test_scores = []
oob_scores = []

for n_est in n_estimators_range:
    clf = BaggingClassifier(
        estimator=DecisionTreeClassifier(),
        n_estimators=n_est,
        oob_score=True,
        random_state=42,
        n_jobs=-1
    )
    clf.fit(X_train, y_train)

    train_scores.append(clf.score(X_train, y_train))
    test_scores.append(clf.score(X_test, y_test))
    oob_scores.append(clf.oob_score_)

# Visualization
plt.figure(figsize=(10, 6))
plt.plot(n_estimators_range, train_scores, 'o-', label='Train')
plt.plot(n_estimators_range, test_scores, 's-', label='Test')
plt.plot(n_estimators_range, oob_scores, '^-', label='OOB')
plt.xlabel('Number of Estimators')
plt.ylabel('Accuracy')
plt.title('Bagging: Performance vs Number of Estimators')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

---

## 4. Random Forest

### 4.1 Basic Usage

```python
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.datasets import load_iris

# Load data
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42
)

# Random Forest classifier
rf_clf = RandomForestClassifier(
    n_estimators=100,       # Number of trees
    max_depth=None,         # Maximum depth
    min_samples_split=2,    # Minimum samples to split
    min_samples_leaf=1,     # Minimum samples in leaf
    max_features='sqrt',    # Number of features to consider
    bootstrap=True,         # Bootstrap sampling
    oob_score=True,         # OOB score
    n_jobs=-1,              # Parallel processing
    random_state=42
)

rf_clf.fit(X_train, y_train)

print("Random Forest results:")
print(f"  Training accuracy: {rf_clf.score(X_train, y_train):.4f}")
print(f"  Test accuracy: {rf_clf.score(X_test, y_test):.4f}")
print(f"  OOB score: {rf_clf.oob_score_:.4f}")
```

### 4.2 Random Forest vs Regular Bagging

```python
"""
Difference between Random Forest and Bagging:

1. Random feature selection:
   - Bagging: Use all features (max_features=1.0)
   - Random Forest: Use sqrt(n_features) or log2(n_features)

2. Tree correlation:
   - Bagging: High correlation between trees
   - Random Forest: Low correlation between trees (increased diversity)

3. Variance reduction:
   - Var(average) = Var(single) / n + (n-1)/n * Cov
   - Lower correlation (Cov) → greater variance reduction
"""

# Comparison experiment
bagging = BaggingClassifier(
    estimator=DecisionTreeClassifier(),
    n_estimators=100,
    max_features=1.0,  # Use all features
    random_state=42,
    n_jobs=-1
)

rf = RandomForestClassifier(
    n_estimators=100,
    max_features='sqrt',  # Use sqrt(n_features)
    random_state=42,
    n_jobs=-1
)

bagging.fit(X_train, y_train)
rf.fit(X_train, y_train)

print("Bagging vs Random Forest:")
print(f"  Bagging accuracy: {bagging.score(X_test, y_test):.4f}")
print(f"  Random Forest accuracy: {rf.score(X_test, y_test):.4f}")
```

### 4.3 max_features Parameter

```python
# Performance change with max_features
from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, test_size=0.2, random_state=42
)

n_features = X_train.shape[1]
max_features_options = [1, 'sqrt', 'log2', 0.5, n_features]

print("Performance by max_features:")
for max_feat in max_features_options:
    rf = RandomForestClassifier(
        n_estimators=100,
        max_features=max_feat,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    print(f"  max_features={max_feat}: {rf.score(X_test, y_test):.4f}")
```

---

## 5. Feature Importance

### 5.1 Basic Feature Importance

```python
# Train Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Feature importance
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]

# Visualization
plt.figure(figsize=(12, 6))
plt.bar(range(len(importances)), importances[indices])
plt.xticks(range(len(importances)),
           [cancer.feature_names[i] for i in indices],
           rotation=90)
plt.ylabel('Feature Importance')
plt.title('Random Forest Feature Importance')
plt.tight_layout()
plt.show()

# Top 10 features
print("\nTop 10 features:")
for i in range(10):
    print(f"  {i+1}. {cancer.feature_names[indices[i]]}: {importances[indices[i]]:.4f}")
```

### 5.2 Feature Importance Interpretation Methods

```python
"""
Feature importance calculation methods:

1. Impurity-based importance (Mean Decrease in Impurity, MDI):
   - Average impurity reduction when each feature is used for splitting
   - feature_importances_ default
   - Drawback: Biased toward high cardinality features

2. Permutation Importance:
   - Measure performance decrease when feature values are randomly shuffled
   - More reliable importance
"""

from sklearn.inspection import permutation_importance

# Calculate permutation importance
perm_importance = permutation_importance(
    rf, X_test, y_test,
    n_repeats=30,
    random_state=42,
    n_jobs=-1
)

# Compare visualization
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# MDI (impurity-based)
sorted_idx_mdi = rf.feature_importances_.argsort()[-10:]
axes[0].barh(range(10), rf.feature_importances_[sorted_idx_mdi])
axes[0].set_yticks(range(10))
axes[0].set_yticklabels([cancer.feature_names[i] for i in sorted_idx_mdi])
axes[0].set_title('MDI (Impurity-based) Feature Importance')

# Permutation importance
sorted_idx_perm = perm_importance.importances_mean.argsort()[-10:]
axes[1].barh(range(10), perm_importance.importances_mean[sorted_idx_perm])
axes[1].set_yticks(range(10))
axes[1].set_yticklabels([cancer.feature_names[i] for i in sorted_idx_perm])
axes[1].set_title('Permutation Feature Importance')

plt.tight_layout()
plt.show()
```

### 5.3 Use for Feature Selection

```python
from sklearn.feature_selection import SelectFromModel

# Importance-based feature selection
selector = SelectFromModel(
    RandomForestClassifier(n_estimators=100, random_state=42),
    threshold='median'  # Select only features above median importance
)
selector.fit(X_train, y_train)

# Selected features
selected_features = cancer.feature_names[selector.get_support()]
print(f"Number of selected features: {len(selected_features)}")
print(f"Selected features: {list(selected_features)}")

# Train with selected features
X_train_selected = selector.transform(X_train)
X_test_selected = selector.transform(X_test)

rf_selected = RandomForestClassifier(n_estimators=100, random_state=42)
rf_selected.fit(X_train_selected, y_train)

print(f"\nAll features accuracy: {rf.score(X_test, y_test):.4f}")
print(f"Selected features accuracy: {rf_selected.score(X_test_selected, y_test):.4f}")
```

---

## 6. OOB (Out-of-Bag) Error

### 6.1 Understanding OOB Score

```python
"""
OOB (Out-of-Bag) error:
- Each tree is trained on bootstrap sample
- Each sample is OOB in about 37% of trees (not used for training)
- Validation with OOB samples → no need for separate validation set

Advantages:
1. No need for additional data splitting
2. Similar effect to cross-validation
3. Validation possible during training
"""

# Use OOB score
rf = RandomForestClassifier(
    n_estimators=100,
    oob_score=True,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)

print("OOB score analysis:")
print(f"  OOB score: {rf.oob_score_:.4f}")
print(f"  Test score: {rf.score(X_test, y_test):.4f}")

# OOB prediction probabilities
print(f"\nOOB prediction probabilities (first 5 samples):")
print(rf.oob_decision_function_[:5])
```

### 6.2 OOB vs Cross-validation Comparison

```python
from sklearn.model_selection import cross_val_score

# Cross-validation
cv_scores = cross_val_score(
    RandomForestClassifier(n_estimators=100, random_state=42),
    X_train, y_train, cv=5
)

# OOB
rf_oob = RandomForestClassifier(n_estimators=100, oob_score=True, random_state=42)
rf_oob.fit(X_train, y_train)

print("OOB vs Cross-validation comparison:")
print(f"  OOB score: {rf_oob.oob_score_:.4f}")
print(f"  CV average score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
```

---

## 7. Hyperparameter Tuning

```python
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from scipy.stats import randint, uniform

# Grid Search
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', 0.5]
}

# More efficient Randomized Search
param_distributions = {
    'n_estimators': randint(50, 300),
    'max_depth': [None] + list(range(5, 31)),
    'min_samples_split': randint(2, 21),
    'min_samples_leaf': randint(1, 11),
    'max_features': uniform(0.1, 0.9)
}

random_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    param_distributions,
    n_iter=50,
    cv=5,
    scoring='accuracy',
    random_state=42,
    n_jobs=-1
)

random_search.fit(X_train, y_train)

print("Hyperparameter tuning results:")
print(f"  Best parameters: {random_search.best_params_}")
print(f"  Best CV score: {random_search.best_score_:.4f}")
print(f"  Test score: {random_search.score(X_test, y_test):.4f}")
```

---

## 8. Random Forest Regression

```python
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load data
diabetes = load_diabetes()
X_train, X_test, y_train, y_test = train_test_split(
    diabetes.data, diabetes.target, test_size=0.2, random_state=42
)

# Random Forest regression
rf_reg = RandomForestRegressor(
    n_estimators=100,
    max_depth=None,
    min_samples_split=2,
    random_state=42,
    n_jobs=-1
)
rf_reg.fit(X_train, y_train)
y_pred = rf_reg.predict(X_test)

print("Random Forest regression results:")
print(f"  MSE: {mean_squared_error(y_test, y_pred):.4f}")
print(f"  RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
print(f"  R²: {r2_score(y_test, y_pred):.4f}")

# Actual vs Predicted
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title(f'Random Forest Regression (R² = {r2_score(y_test, y_pred):.4f})')
plt.grid(True, alpha=0.3)
plt.show()
```

---

## 9. Extra Trees (Extremely Randomized Trees)

```python
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor

"""
Extra Trees vs Random Forest:

1. Split point selection:
   - Random Forest: Select optimal split point for each feature
   - Extra Trees: Select random split point for each feature

2. Bootstrap:
   - Random Forest: Use bootstrap by default
   - Extra Trees: Use full data by default

3. Characteristics:
   - Extra Trees: Faster, more randomness
   - Random Forest: Generally better performance
"""

# Comparison
rf = RandomForestClassifier(n_estimators=100, random_state=42)
et = ExtraTreesClassifier(n_estimators=100, random_state=42)

rf.fit(X_train, y_train)
et.fit(X_train, y_train)

print("Random Forest vs Extra Trees:")
print(f"  Random Forest: {rf.score(X_test, y_test):.4f}")
print(f"  Extra Trees: {et.score(X_test, y_test):.4f}")
```

---

## 10. Voting Classifier

```python
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# Define various models
clf1 = LogisticRegression(random_state=42, max_iter=1000)
clf2 = RandomForestClassifier(n_estimators=50, random_state=42)
clf3 = SVC(probability=True, random_state=42)

# Hard Voting (majority vote)
hard_voting = VotingClassifier(
    estimators=[
        ('lr', clf1),
        ('rf', clf2),
        ('svc', clf3)
    ],
    voting='hard'
)

# Soft Voting (average probabilities)
soft_voting = VotingClassifier(
    estimators=[
        ('lr', clf1),
        ('rf', clf2),
        ('svc', clf3)
    ],
    voting='soft'
)

# Train and compare
print("Voting Classifier comparison:")
for clf, label in [(clf1, 'Logistic'), (clf2, 'RF'), (clf3, 'SVC'),
                   (hard_voting, 'Hard Voting'), (soft_voting, 'Soft Voting')]:
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    print(f"  {label}: {score:.4f}")
```

---

## Practice Problems

### Problem 1: Random Forest Classification
Train Random Forest on breast cancer data and analyze feature importance.

```python
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier

cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, test_size=0.2, random_state=42
)

# Solution
rf = RandomForestClassifier(n_estimators=100, oob_score=True, random_state=42)
rf.fit(X_train, y_train)

print(f"Test accuracy: {rf.score(X_test, y_test):.4f}")
print(f"OOB score: {rf.oob_score_:.4f}")

print("\nTop 5 features:")
indices = np.argsort(rf.feature_importances_)[::-1][:5]
for i, idx in enumerate(indices):
    print(f"  {i+1}. {cancer.feature_names[idx]}: {rf.feature_importances_[idx]:.4f}")
```

### Problem 2: Hyperparameter Tuning
Find optimal Random Forest parameters using Grid Search.

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [5, 10, None],
    'min_samples_leaf': [1, 2, 5]
}

# Solution
grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)
grid_search.fit(X_train, y_train)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best CV score: {grid_search.best_score_:.4f}")
print(f"Test score: {grid_search.score(X_test, y_test):.4f}")
```

### Problem 3: Voting Ensemble
Create a Voting Classifier combining multiple models.

```python
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

# Solution
voting_clf = VotingClassifier(
    estimators=[
        ('lr', LogisticRegression(max_iter=1000)),
        ('rf', RandomForestClassifier(n_estimators=50)),
        ('dt', DecisionTreeClassifier(max_depth=5))
    ],
    voting='soft'
)
voting_clf.fit(X_train, y_train)
print(f"Voting accuracy: {voting_clf.score(X_test, y_test):.4f}")
```

---

## Summary

| Model | Features | Advantages | Disadvantages |
|-------|----------|------------|---------------|
| Bagging | Bootstrap + averaging | Reduce variance, prevent overfitting | Hard to interpret |
| Random Forest | Bagging + random features | High performance, feature importance | High computation |
| Extra Trees | Fully random splits | Fast training | Possibly lower than RF |
| Voting | Combine diverse models | Leverage diversity | Need tuning individual models |

### Random Forest Hyperparameter Guide

| Parameter | Default | Recommended Range | Effect |
|-----------|---------|-------------------|--------|
| n_estimators | 100 | 100-500 | More stable with higher values |
| max_depth | None | 10-30 | Control overfitting |
| min_samples_split | 2 | 2-20 | Control overfitting |
| min_samples_leaf | 1 | 1-10 | Control overfitting |
| max_features | 'sqrt' | 'sqrt', 'log2', 0.3-0.7 | Tree diversity |
