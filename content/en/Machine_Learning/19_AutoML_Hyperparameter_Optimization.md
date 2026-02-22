# AutoML and Hyperparameter Optimization

## Overview

AutoML automates the process of selecting algorithms, engineering features, and tuning hyperparameters. This lesson covers Optuna for hyperparameter optimization, Auto-sklearn, FLAML, and H2O AutoML — tools that can significantly reduce the time from data to deployed model.

---

## 1. The AutoML Landscape

### 1.1 What AutoML Automates

```python
"""
The CASH Problem: Combined Algorithm Selection and Hyperparameter optimization

Manual ML workflow:
  Data → Preprocessing → Feature Engineering → Model Selection → Hyperparameter Tuning → Evaluation
         ↑ AutoML automates some or all of these steps

AutoML levels:
1. Hyperparameter Optimization (HPO): Tune one model's parameters
   → Optuna, Hyperopt, Ray Tune
2. Model Selection + HPO: Try multiple algorithms with tuning
   → Auto-sklearn, FLAML, H2O AutoML
3. Full Pipeline AutoML: Include preprocessing and feature engineering
   → Auto-sklearn (with preprocessing), TPOT, AutoGluon

Trade-offs:
  + Saves time, finds good models quickly
  + Reduces human bias in model selection
  - Computational cost can be high
  - "Black box" pipeline can be hard to debug
  - May overfit to validation set with many trials
"""
```

### 1.2 HPO Methods Overview

| Method | Description | Pros | Cons |
|--------|-------------|------|------|
| **Grid Search** | Try all combinations | Exhaustive | Exponential cost |
| **Random Search** | Random parameter sampling | Better than grid for high-D | No learning |
| **Bayesian (TPE)** | Learn from past trials | Efficient, finds good params | Overhead per trial |
| **Successive Halving** | Early stopping of bad configs | Fast | Needs many configs |
| **Hyperband** | SH + adaptive allocation | Very fast | Complex |
| **Population-Based** | Evolutionary approach | Good for large spaces | High parallelism needed |

---

## 2. Optuna Deep Dive

### 2.1 Basic Optuna Usage

```python
# pip install optuna
import optuna
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.ensemble import GradientBoostingRegressor

# Load data
housing = fetch_california_housing(as_frame=True)
X, y = housing.data, housing.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def objective(trial):
    """Optuna objective function: returns the value to minimize."""
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'max_depth': trial.suggest_int('max_depth', 2, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
    }

    model = GradientBoostingRegressor(**params, random_state=42)
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    return -scores.mean()  # Optuna minimizes by default

# Create and run study
study = optuna.create_study(direction='minimize', study_name='gb_tuning')
study.optimize(objective, n_trials=50, show_progress_bar=True)

print(f"\nBest MSE: {study.best_value:.4f}")
print(f"Best params: {study.best_params}")
```

### 2.2 Optuna Pruning (Early Stopping)

```python
from sklearn.model_selection import StratifiedKFold

def objective_with_pruning(trial):
    """Prune unpromising trials early using intermediate values."""
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'max_depth': trial.suggest_int('max_depth', 2, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
    }

    model = GradientBoostingRegressor(**params, random_state=42)

    # Report intermediate values for each CV fold
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42) if False else None
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    fold_scores = []
    for step, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        X_fold_train = X_train.iloc[train_idx]
        y_fold_train = y_train.iloc[train_idx]
        X_fold_val = X_train.iloc[val_idx]
        y_fold_val = y_train.iloc[val_idx]

        model.fit(X_fold_train, y_fold_train)
        score = model.score(X_fold_val, y_fold_val)
        fold_scores.append(score)

        # Report intermediate value (running mean of R²)
        trial.report(np.mean(fold_scores), step)

        # Prune if this trial is not promising
        if trial.should_prune():
            raise optuna.TrialPruned()

    return -np.mean(fold_scores)

# Pruning study (MedianPruner stops trials worse than median)
pruned_study = optuna.create_study(
    direction='minimize',
    pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2),
)
pruned_study.optimize(objective_with_pruning, n_trials=50, show_progress_bar=True)

# Check how many were pruned
pruned = len([t for t in pruned_study.trials if t.state == optuna.trial.TrialState.PRUNED])
print(f"\nPruned {pruned} out of {len(pruned_study.trials)} trials")
print(f"Best R²: {-pruned_study.best_value:.4f}")
```

### 2.3 Optuna Visualization

```python
import optuna.visualization as vis

# 1. Optimization History
fig = vis.plot_optimization_history(study)
fig.show()

# 2. Parameter Importance
fig = vis.plot_param_importances(study)
fig.show()

# 3. Parallel Coordinate Plot
fig = vis.plot_parallel_coordinate(study, params=['n_estimators', 'max_depth', 'learning_rate'])
fig.show()

# 4. Contour Plot (2D parameter interactions)
fig = vis.plot_contour(study, params=['learning_rate', 'max_depth'])
fig.show()

# 5. Slice Plot
fig = vis.plot_slice(study, params=['n_estimators', 'learning_rate', 'max_depth'])
fig.show()

print("Optuna provides rich interactive visualizations via Plotly.")
print("Use these to understand which parameters matter most.")
```

### 2.4 Multi-Objective Optimization

```python
def multi_objective(trial):
    """Optimize for both accuracy and model size (inference speed)."""
    n_estimators = trial.suggest_int('n_estimators', 10, 500)
    max_depth = trial.suggest_int('max_depth', 2, 10)
    learning_rate = trial.suggest_float('learning_rate', 0.01, 0.3, log=True)

    model = GradientBoostingRegressor(
        n_estimators=n_estimators, max_depth=max_depth,
        learning_rate=learning_rate, random_state=42
    )
    scores = cross_val_score(model, X_train, y_train, cv=3, scoring='neg_mean_squared_error')

    mse = -scores.mean()
    complexity = n_estimators * (2 ** max_depth)  # Proxy for model size

    return mse, complexity  # Minimize both

# Multi-objective study
mo_study = optuna.create_study(
    directions=['minimize', 'minimize'],
    study_name='multi_objective',
)
mo_study.optimize(multi_objective, n_trials=50, show_progress_bar=True)

# Pareto front
pareto_trials = mo_study.best_trials
print(f"\nPareto-optimal solutions: {len(pareto_trials)}")
for t in pareto_trials[:5]:
    print(f"  MSE={t.values[0]:.4f}, Complexity={t.values[1]:.0f}, "
          f"n_estimators={t.params['n_estimators']}, max_depth={t.params['max_depth']}")
```

---

## 3. Auto-sklearn

### 3.1 Automated Model Selection

```python
"""
Auto-sklearn automatically searches over:
  - 15 classifiers / 14 regressors
  - 18 feature preprocessing methods
  - Hyperparameters for each

# pip install auto-sklearn  (Linux only, uses SMAC for Bayesian optimization)

import autosklearn.regression
import autosklearn.classification

# Regression
automl = autosklearn.regression.AutoSklearnRegressor(
    time_left_for_this_task=300,     # Total time budget (seconds)
    per_run_time_limit=30,           # Max time per model
    n_jobs=-1,
    memory_limit=4096,               # MB
    ensemble_size=5,                 # Final ensemble of top models
    seed=42,
)
automl.fit(X_train, y_train)

# Results
print(automl.leaderboard())
print(f"Test R²: {automl.score(X_test, y_test):.4f}")

# Show the final ensemble
print(automl.show_models())

# Get the final pipeline
for weight, pipeline in automl.get_models_with_weights():
    print(f"Weight: {weight:.3f}")
    print(f"Pipeline: {pipeline}")
"""
print("Auto-sklearn: Best for Linux environments, uses meta-learning to warm-start search.")
print("Key features: automatic ensemble, meta-learning, SMAC Bayesian optimization.")
```

---

## 4. FLAML (Fast and Lightweight AutoML)

### 4.1 Quick Start

```python
"""
FLAML is designed for speed and low computational cost.
Key features:
  - Cost-effective search (tries cheap models first)
  - Learns optimal sample size
  - Much faster than Auto-sklearn

# pip install flaml

from flaml import AutoML

automl = AutoML()
automl.fit(
    X_train, y_train,
    task='regression',
    time_budget=60,           # Total time (seconds)
    metric='rmse',
    estimator_list=['lgbm', 'xgboost', 'rf', 'extra_tree', 'lrl1'],
    eval_method='cv',
    n_splits=5,
    seed=42,
    verbose=1,
)

# Results
print(f"Best model: {automl.best_estimator}")
print(f"Best config: {automl.best_config}")
print(f"Best RMSE: {automl.best_loss:.4f}")
print(f"Test R²: {automl.score(X_test, y_test, metric='r2'):.4f}")

# Feature importance (if tree-based)
if hasattr(automl.model, 'feature_importances_'):
    import pandas as pd
    importances = pd.Series(
        automl.model.feature_importances_,
        index=X_train.columns
    ).sort_values(ascending=False)
    print(importances.head(10))
"""
print("FLAML: Fastest AutoML library, great for time-constrained experiments.")
print("Excels at finding good LightGBM configs quickly.")
```

---

## 5. H2O AutoML

### 5.1 H2O AutoML Framework

```python
"""
H2O AutoML trains a large set of models and creates stacked ensembles.

# pip install h2o

import h2o
from h2o.automl import H2OAutoML

h2o.init()

# Convert to H2O Frame
train = h2o.H2OFrame(pd.concat([X_train, y_train], axis=1))
test = h2o.H2OFrame(pd.concat([X_test, y_test], axis=1))

# Run AutoML
aml = H2OAutoML(
    max_runtime_secs=300,
    max_models=20,
    seed=42,
    sort_metric='RMSE',
    exclude_algos=['DeepLearning'],  # Exclude slow algorithms
)
aml.train(x=X_train.columns.tolist(), y='MedHouseVal', training_frame=train)

# Leaderboard
lb = aml.leaderboard
print(lb.head(10))

# Best model
print(f"Best model: {aml.leader.model_id}")
print(f"Test performance:")
perf = aml.leader.model_performance(test)
print(perf)

h2o.shutdown()
"""
print("H2O AutoML: Enterprise-grade, excellent stacked ensembles.")
print("Supports distributed computing for large datasets.")
```

---

## 6. Comparing AutoML Frameworks

### 6.1 Framework Comparison

| Feature | Optuna | Auto-sklearn | FLAML | H2O AutoML |
|---------|--------|-------------|-------|------------|
| **Type** | HPO framework | Full AutoML | Full AutoML | Full AutoML |
| **Speed** | Depends on objective | Slow | Very fast | Medium |
| **Model Selection** | Manual (you choose) | Automatic | Automatic | Automatic |
| **Ensemble** | Manual | Built-in | Manual | Stacking |
| **Platform** | Any | Linux only | Any | Any (JVM) |
| **Learning Curve** | Medium | Easy | Easy | Easy |
| **Best For** | Custom HPO, fine control | Linux research | Quick experiments | Production, large data |
| **GPU Support** | Via user code | No | LightGBM/XGBoost | Yes |

### 6.2 When to Use What

```python
"""
Decision Guide:

1. Need full control, custom objective?
   → Optuna (most flexible HPO framework)

2. Quick baseline on any platform?
   → FLAML (fastest, works everywhere)

3. Research on Linux, want meta-learning?
   → Auto-sklearn (best meta-learning, robust ensembles)

4. Large dataset, need production deployment?
   → H2O AutoML (distributed, enterprise features)

5. Deep learning HPO?
   → Optuna + PyTorch/TensorFlow
   → Ray Tune (distributed HPO)

6. Kaggle competition?
   → Optuna for tuning XGBoost/LightGBM (most common winner)
"""
```

---

## 7. Best Practices

### 7.1 Avoiding Overfitting in HPO

```python
"""
HPO overfitting: The more trials you run, the more likely you
"overfit" to the validation set by finding a configuration that
is lucky on that specific split.

Mitigations:
1. Use robust CV (5-fold or more)
2. Hold out a final test set that HPO never sees
3. Limit the number of trials (diminishing returns after ~100-200)
4. Use early stopping (Optuna pruning)
5. Report confidence intervals, not just best score
"""

# Example: checking for HPO overfitting
import optuna

# Run a study
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100, show_progress_bar=True)

# Train best model on full training set
best_params = study.best_params
best_model = GradientBoostingRegressor(**best_params, random_state=42)
best_model.fit(X_train, y_train)

# Compare validation score vs test score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error

cv_mse = -cross_val_score(best_model, X_train, y_train, cv=5, scoring='neg_mean_squared_error').mean()
test_mse = mean_squared_error(y_test, best_model.predict(X_test))

print(f"CV MSE (seen during HPO):    {cv_mse:.4f}")
print(f"Test MSE (never seen):       {test_mse:.4f}")
print(f"Gap (potential overfitting):  {abs(cv_mse - test_mse):.4f}")
```

### 7.2 Efficient Search Spaces

```python
"""
Tips for defining good search spaces:

1. Use log scale for learning rates and regularization:
   trial.suggest_float('lr', 1e-5, 1e-1, log=True)

2. Use integer for discrete choices:
   trial.suggest_int('n_estimators', 50, 500, step=50)

3. Use categorical for algorithm selection:
   trial.suggest_categorical('model', ['rf', 'xgb', 'lgbm'])

4. Conditional parameters:
   model_name = trial.suggest_categorical('model', ['rf', 'svm'])
   if model_name == 'rf':
       n_estimators = trial.suggest_int('rf_n_estimators', 50, 500)
   elif model_name == 'svm':
       C = trial.suggest_float('svm_C', 0.01, 100, log=True)

5. Start with wide ranges, then narrow based on results.

6. Use Optuna's parameter importance to drop unimportant params.
"""
```

---

## 8. Practice Problems

### Exercise 1: Optuna for Classification

```python
"""
1. Load the breast cancer dataset.
2. Define an Optuna objective that searches over:
   - Algorithm: RandomForest vs GradientBoosting vs SVM
   - Conditional hyperparameters for each
3. Use F1-score as the metric.
4. Run 100 trials with pruning.
5. Visualize: optimization history, parameter importance, contour plot.
6. Compare the best Optuna model with a default RandomForest.
"""
```

### Exercise 2: AutoML Comparison

```python
"""
1. Load California Housing dataset.
2. Run FLAML with a 60-second budget.
3. Run Optuna (GradientBoosting only) with the same time budget.
4. Compare:
   a) Best model performance (R², RMSE)
   b) Number of models evaluated
   c) Final model complexity
5. Which approach found a better model? Why?
"""
```

### Exercise 3: Multi-Objective HPO

```python
"""
1. Use Optuna multi-objective to optimize:
   - Objective 1: Minimize prediction error (RMSE)
   - Objective 2: Minimize training time
   - Objective 3: Minimize model size (number of parameters)
2. Plot the Pareto front.
3. Select a model from the Pareto front that balances accuracy and speed.
4. Justify your choice for a real-world deployment scenario.
"""
```

---

## 9. Summary

### Key Takeaways

| Concept | Description |
|---------|-------------|
| **Optuna** | Flexible HPO with TPE, pruning, multi-objective |
| **Auto-sklearn** | Full pipeline AutoML with meta-learning (Linux) |
| **FLAML** | Fastest AutoML, cost-effective search |
| **H2O AutoML** | Production-grade with stacked ensembles |
| **Pruning** | Stop bad trials early to save compute |
| **Multi-objective** | Optimize accuracy vs. speed vs. complexity |
| **HPO overfitting** | More trials ≠ better generalization |

### Best Practices

1. **Always hold out a test set** that HPO never sees
2. **Start with FLAML** for a quick baseline (minutes)
3. **Use Optuna** when you need fine-grained control
4. **Limit trial count** — diminishing returns after 100-200 trials
5. **Log-scale** for learning rates and regularization parameters
6. **Use pruning** to eliminate bad configs early
7. **Check for overfitting** by comparing CV and test performance

### Next Steps

- **L20**: Anomaly Detection — detect outliers and unusual patterns
- **MLOps L03-04**: MLflow for experiment tracking — log Optuna studies systematically
