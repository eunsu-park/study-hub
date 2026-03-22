# Cross-Validation and Hyperparameter Tuning

**Previous**: [Model Evaluation](./04_Model_Evaluation.md) | **Next**: [Decision Trees](./06_Decision_Trees.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain why a single train/test split is insufficient for reliable model evaluation
2. Implement K-Fold, Stratified K-Fold, and Time Series cross-validation strategies
3. Compare cross_val_score and cross_validate for multi-metric evaluation
4. Apply Grid Search and Randomized Search to tune hyperparameters systematically
5. Describe how nested cross-validation separates model selection from model evaluation
6. Demonstrate hyperparameter tuning within a scikit-learn Pipeline
7. Evaluate the trade-offs between Grid Search, Randomized Search, and Bayesian optimization

---

A model evaluated on a single train/test split can give misleading results -- you might just be lucky (or unlucky) with how the data was divided. Cross-validation provides a more robust estimate of generalization performance, while hyperparameter tuning ensures you are getting the most out of your chosen algorithm rather than settling for default settings.

---

## 1. Cross-Validation

### 1.1 K-Fold Cross-Validation

```python
import numpy as np
from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris

# Load data
iris = load_iris()
X, y = iris.data, iris.target

# Create model
model = LogisticRegression(max_iter=1000)

# K-Fold cross-validation
scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')

print("K-Fold Cross-Validation (K=5)")
print(f"Fold scores: {scores}")
print(f"Mean accuracy: {scores.mean():.4f}")
print(f"Standard deviation: {scores.std():.4f}")
print(f"95% confidence interval: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
```

### 1.2 Stratified K-Fold

```python
from sklearn.model_selection import StratifiedKFold

# Preserve class ratios
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

scores = cross_val_score(model, X, y, cv=skf, scoring='accuracy')

print("\nStratified K-Fold")
print(f"Mean accuracy: {scores.mean():.4f}")

# Check class distribution in each fold
print("\nClass distribution per fold:")
for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
    train_classes = np.bincount(y[train_idx])
    val_classes = np.bincount(y[val_idx])
    print(f"  Fold {fold}: Train={train_classes}, Val={val_classes}")
```

### 1.3 Various Cross-Validation Methods

```python
from sklearn.model_selection import (
    LeaveOneOut,
    LeavePOut,
    ShuffleSplit,
    RepeatedKFold,
    RepeatedStratifiedKFold
)

# Leave-One-Out (LOO)
loo = LeaveOneOut()
print(f"LOO splits: {loo.get_n_splits(X)}")  # Equal to number of samples

# Shuffle Split (random split)
ss = ShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
scores = cross_val_score(model, X, y, cv=ss)
print(f"\nShuffle Split mean: {scores.mean():.4f}")

# Repeated K-Fold
rkf = RepeatedKFold(n_splits=5, n_repeats=10, random_state=42)
scores = cross_val_score(model, X, y, cv=rkf)
print(f"Repeated K-Fold mean: {scores.mean():.4f}")
print(f"Repeated K-Fold total splits: {len(scores)}")  # 5 * 10 = 50
```

### 1.4 Time Series Cross-Validation

```python
from sklearn.model_selection import TimeSeriesSplit

# For time series data (past → future prediction)
tscv = TimeSeriesSplit(n_splits=5)

print("Time Series Split:")
for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
    print(f"  Fold {fold}: Train=[{train_idx[0]}:{train_idx[-1]}], Test=[{test_idx[0]}:{test_idx[-1]}]")
```

---

## 2. cross_val_score vs cross_validate

```python
from sklearn.model_selection import cross_validate

# Evaluate multiple metrics simultaneously
scoring = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']

cv_results = cross_validate(
    model, X, y,
    cv=5,
    scoring=scoring,
    return_train_score=True
)

print("cross_validate results:")
for metric in scoring:
    train_key = f'train_{metric}'
    test_key = f'test_{metric}'
    print(f"\n{metric}:")
    print(f"  Train: {cv_results[train_key].mean():.4f} (+/- {cv_results[train_key].std():.4f})")
    print(f"  Test:  {cv_results[test_key].mean():.4f} (+/- {cv_results[test_key].std():.4f})")

# Training time information
print(f"\nAverage training time: {cv_results['fit_time'].mean():.4f}s")
print(f"Average prediction time: {cv_results['score_time'].mean():.4f}s")
```

---

## 3. Hyperparameter Tuning

### 3.1 Grid Search

```python
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler

# Prepare data
cancer = load_breast_cancer()
X, y = cancer.data, cancer.target

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Hyperparameter grid
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [1, 0.1, 0.01, 0.001],
    'kernel': ['rbf', 'linear']
}

# Grid Search
grid_search = GridSearchCV(
    SVC(),
    param_grid,
    cv=5,
    scoring='accuracy',
    verbose=1,
    n_jobs=-1  # Use all CPUs
)

grid_search.fit(X_scaled, y)

print("\nGrid Search results:")
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best score: {grid_search.best_score_:.4f}")

# View all results
import pandas as pd
results = pd.DataFrame(grid_search.cv_results_)
print(f"\nTop 5 combinations:")
print(results.nsmallest(5, 'rank_test_score')[['params', 'mean_test_score', 'std_test_score', 'rank_test_score']])
```

### 3.2 Randomized Search

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint

# Hyperparameter distributions
param_distributions = {
    'C': uniform(0.1, 100),  # Uniform distribution from 0.1 to 100.1
    'gamma': uniform(0.001, 1),
    'kernel': ['rbf', 'linear', 'poly']
}

# Randomized Search
random_search = RandomizedSearchCV(
    SVC(),
    param_distributions,
    n_iter=50,  # Try 50 combinations
    cv=5,
    scoring='accuracy',
    random_state=42,
    n_jobs=-1
)

random_search.fit(X_scaled, y)

print("Randomized Search results:")
print(f"Best parameters: {random_search.best_params_}")
print(f"Best score: {random_search.best_score_:.4f}")
```

### 3.3 Grid Search vs Randomized Search

```python
"""
Grid Search:
- Pros: Exhaustive search, optimal solution guaranteed (within grid)
- Cons: Exponential growth in combinations

Randomized Search:
- Pros: Computationally efficient, can explore continuous distributions
- Cons: No guarantee of optimal solution

Selection criteria:
- Few parameters with clear range → Grid Search
- Many parameters or uncertain range → Randomized Search
"""
```

---

## 4. Advanced Tuning Techniques

### 4.1 Halving Search

```python
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV

# Progressively allocate resources during search
halving_search = HalvingGridSearchCV(
    SVC(),
    param_grid,
    cv=5,
    factor=3,  # Reduce candidates to 1/3 each round
    resource='n_samples',
    random_state=42
)

halving_search.fit(X_scaled, y)

print("Halving Grid Search results:")
print(f"Best parameters: {halving_search.best_params_}")
print(f"Best score: {halving_search.best_score_:.4f}")
```

### 4.2 Bayesian Optimization (Optuna)

```python
# pip install optuna

import optuna
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

def objective(trial):
    # Suggest hyperparameters
    n_estimators = trial.suggest_int('n_estimators', 10, 200)
    max_depth = trial.suggest_int('max_depth', 2, 32)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=42
    )

    scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    return scores.mean()

# Run optimization
# study = optuna.create_study(direction='maximize')
# study.optimize(objective, n_trials=100)

# print(f"Best parameters: {study.best_params}")
# print(f"Best score: {study.best_value:.4f}")
```

---

## 5. Nested Cross-Validation

```python
from sklearn.model_selection import cross_val_score, GridSearchCV

# Outer loop: Model evaluation
# Inner loop: Hyperparameter tuning

# Inner CV (hyperparameter tuning)
param_grid = {'C': [0.1, 1, 10], 'gamma': [0.1, 0.01]}
inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
grid_search = GridSearchCV(SVC(), param_grid, cv=inner_cv, scoring='accuracy')

# Outer CV (model evaluation)
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
nested_scores = cross_val_score(grid_search, X_scaled, y, cv=outer_cv, scoring='accuracy')

print("Nested cross-validation results:")
print(f"Outer fold scores: {nested_scores}")
print(f"Mean score: {nested_scores.mean():.4f} (+/- {nested_scores.std():.4f})")

# Compare: Regular CV vs Nested CV
grid_search.fit(X_scaled, y)
print(f"\nRegular CV best score: {grid_search.best_score_:.4f}")
print(f"Nested CV mean score: {nested_scores.mean():.4f}")
# Nested CV provides more realistic generalization performance estimate
```

---

## 6. Using with Pipelines

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Define pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC())
])

# Parameter names: step__parameter
param_grid = {
    'svm__C': [0.1, 1, 10],
    'svm__gamma': [0.1, 0.01, 0.001],
    'svm__kernel': ['rbf', 'linear']
}

grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X, y)

print("Pipeline Grid Search results:")
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best score: {grid_search.best_score_:.4f}")
```

---

## 7. Practical Tips

### 7.1 Scoring Functions

```python
from sklearn.metrics import make_scorer, f1_score, mean_squared_error

# Built-in scoring
# Classification: 'accuracy', 'precision', 'recall', 'f1', 'roc_auc'
# Regression: 'r2', 'neg_mean_squared_error', 'neg_mean_absolute_error'

# Custom scoring function
def custom_score(y_true, y_pred):
    return f1_score(y_true, y_pred, average='weighted')

custom_scorer = make_scorer(custom_score)

scores = cross_val_score(model, X, y, cv=5, scoring=custom_scorer)
print(f"Custom score: {scores.mean():.4f}")
```

### 7.2 Early Stopping Callbacks

```python
# Early stopping in Optuna
# import optuna

# def objective(trial):
#     # ...
#     for epoch in range(100):
#         accuracy = train_epoch()
#         trial.report(accuracy, epoch)
#         if trial.should_prune():
#             raise optuna.TrialPruned()
#     return accuracy

# study = optuna.create_study(direction='maximize',
#                            pruner=optuna.pruners.MedianPruner())
```

### 7.3 Saving Results

```python
import joblib
import json

# Save best model
best_model = grid_search.best_estimator_
joblib.dump(best_model, 'best_model.pkl')

# Save results
results = {
    'best_params': grid_search.best_params_,
    'best_score': grid_search.best_score_,
    'cv_results': {k: v.tolist() if isinstance(v, np.ndarray) else v
                   for k, v in grid_search.cv_results_.items()}
}

with open('tuning_results.json', 'w') as f:
    json.dump(results, f, indent=2)
```

---

## Exercises

### Exercise 1: K-Fold Cross-Validation
Perform 10-Fold cross-validation on the Iris dataset.

```python
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

iris = load_iris()
model = LogisticRegression(max_iter=1000)

# Solution
scores = cross_val_score(model, iris.data, iris.target, cv=10)
print(f"Mean accuracy: {scores.mean():.4f} (+/- {scores.std():.4f})")
```

### Exercise 2: Grid Search
Tune the C parameter of logistic regression.

```python
from sklearn.model_selection import GridSearchCV

param_grid = {'C': [0.01, 0.1, 1, 10, 100]}

# Solution
grid = GridSearchCV(LogisticRegression(max_iter=1000), param_grid, cv=5)
grid.fit(iris.data, iris.target)
print(f"Best C: {grid.best_params_['C']}")
print(f"Best score: {grid.best_score_:.4f}")
```

---

## Summary

| Technique | Purpose | Features |
|-----------|---------|----------|
| K-Fold | Model evaluation | Split data into K parts |
| Stratified K-Fold | Imbalanced data | Preserve class ratios |
| Time Series Split | Time series | Maintain temporal order |
| Grid Search | Parameter tuning | Exhaustive search |
| Randomized Search | Parameter tuning | Random sampling |
| Nested CV | Reliable evaluation | Separate tuning and evaluation |
