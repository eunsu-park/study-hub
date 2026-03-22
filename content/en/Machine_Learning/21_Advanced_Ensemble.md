# Advanced Ensemble Methods — Stacking and Blending

[← Previous: 20. Anomaly Detection](20_Anomaly_Detection.md) | [Next: 22. Production ML Serving →](22_Production_ML_Serving.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain why stacking works from the perspective of bias-variance tradeoff and diverse learner combination
2. Design a stacking architecture with level-0 base learners and a level-1 meta-learner, understanding each layer's role
3. Implement stacking using scikit-learn's `StackingClassifier` and `StackingRegressor` with cross-validated training
4. Distinguish between stacking (CV-based) and blending (holdout-based) and choose the right approach for your data size
5. Measure prediction diversity among base learners using a correlation matrix and use it to select complementary models
6. Apply hyperparameter optimization strategies specifically tailored for stacked ensembles
7. Recognize when stacking yields diminishing returns and when simpler ensembles are more appropriate

---

You have already learned two major ensemble strategies: bagging (L07), which reduces variance by averaging independent models trained on bootstrap samples, and boosting (L08), which reduces bias by sequentially correcting errors. Stacking takes a fundamentally different approach — it learns *how to combine* diverse models. Instead of using a fixed rule like averaging or weighted voting, stacking trains a second-level model (the meta-learner) to discover the optimal combination of base model predictions. This lesson teaches you to build, tune, and critically evaluate stacked ensembles, including the closely related technique of blending. By the end, you will understand not only how to stack models, but — equally important — when not to.

---

> **Analogy**: Stacking is like a panel of expert judges — each judge (base model) scores independently, then a chief judge (meta-learner) weighs all opinions to make the final decision. A good chief judge knows that Judge A is excellent at spotting technical merit but overly generous on artistic impression, while Judge B is the opposite. The chief judge doesn't simply average the scores; instead, they learn each judge's strengths and weaknesses to produce a more accurate final score than any single judge could.

---

## 1. Why Stacking Works

### 1.1 The Bias-Variance Perspective

Different model families have different bias-variance profiles:

```python
"""
Model Bias-Variance Profiles:

┌─────────────────────┬──────────┬──────────┐
│ Model               │ Bias     │ Variance │
├─────────────────────┼──────────┼──────────┤
│ Linear Regression   │ High     │ Low      │
│ k-NN (small k)      │ Low      │ High     │
│ Random Forest       │ Low      │ Medium   │
│ SVM (RBF kernel)    │ Low      │ Medium   │
│ Gradient Boosting   │ Low      │ Med-High │
└─────────────────────┴──────────┴──────────┘

Bagging reduces variance (averaging independent predictions).
Boosting reduces bias (sequential error correction).
Stacking reduces BOTH: it combines models with different bias-variance
profiles, and the meta-learner learns the optimal weighting.
"""
```

### 1.2 Diversity Is the Key

Stacking works because different models make *different errors*. If all base models made identical predictions, combining them adds no value. The power comes from **diversity** — each model captures different patterns in the data.

Three sources of diversity:

1. **Algorithm diversity**: Different model families (linear, tree-based, instance-based)
2. **Data diversity**: Different feature subsets or different data views
3. **Hyperparameter diversity**: Same algorithm with different settings

```python
"""
Suppose three models predict a binary outcome for 10 samples.

Model A: [1, 1, 0, 0, 1, 1, 0, 0, 1, 1]  (errors on samples 3, 7)
Model B: [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]  (errors on samples 2, 6)
Model C: [1, 1, 1, 0, 0, 1, 0, 0, 1, 1]  (errors on samples 5, 8)
Truth:   [1, 1, 1, 0, 1, 1, 0, 0, 1, 1]

Majority vote: [1, 1, 1, 0, 1, 1, 0, 0, 1, 1] → perfect!

Each model has 20% error, but their errors don't overlap.
A meta-learner can learn to trust each model where it excels.
"""
```

### 1.3 Ambiguity Decomposition

The expected error of a combined predictor (Krogh & Vedelsby, 1995):

```
E[error_combined] ≤ (1/N) × Σ E[error_i] - diversity_term

Where:
- N = number of base models
- diversity_term = average pairwise disagreement among models

Key insight: Higher diversity → larger subtracted term → lower combined error
```

---

## 2. Stacking Architecture

### 2.1 Two-Level Structure

```
Level 0 (Base Learners):
┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐
│ Model 1  │  │ Model 2  │  │ Model 3  │  │ Model N  │
│ (e.g.LR) │  │ (e.g.RF) │  │ (e.g.SVM)│  │ (e.g.KNN)│
└────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘
     │              │              │              │
     ▼              ▼              ▼              ▼
   pred_1         pred_2         pred_3         pred_N
     │              │              │              │
     └──────────────┴──────┬───────┴──────────────┘
                           │
                    ┌──────▼──────┐
Level 1             │ Meta-Learner │
(Meta-Learner):     │ (e.g. LR)   │
                    └──────┬──────┘
                           │
                     Final Prediction
```

**Level 0 (Base Learners)**: Diverse models that each learn different aspects of the data.

**Level 1 (Meta-Learner)**: A model that takes base learners' predictions as input features and learns the best way to combine them.

### 2.2 Why Use a Simple Meta-Learner?

The meta-learner operates on a very small feature space (one feature per base model). A regularized linear model is the standard choice because:

```python
"""
1. Small feature space: With 4 base models, the meta-learner has only 4 inputs.
   A linear model with 4 coefficients is appropriate for this dimensionality.

2. Regularization is natural: Logistic/Ridge regression automatically
   regularizes the combination weights, preventing overfitting.

3. Interpretable: The meta-learner coefficients directly tell you
   how much each base model contributes:
   Meta-LR weights: [0.35, 0.30, 0.25, 0.10]
     → RF 35%, XGBoost 30%, SVM 25%, KNN 10%

4. Exception: For competitions with many base models (50+),
   a non-linear meta-learner can capture interactions between predictions.
"""
```

---

## 3. Cross-Validated Stacking (Preventing Data Leakage)

### 3.1 The Data Leakage Problem

A naive approach trains base models on the entire training set and uses their predictions on that same data as meta-features. This causes **data leakage**: predictions are overly optimistic because the models have already seen the data.

### 3.2 The Cross-Validation Solution

The standard approach uses K-fold CV to generate "out-of-fold" (OOF) predictions:

```
Training data: [Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5]

For each base model:
  Iteration 1: Train on [2,3,4,5] → predict Fold 1
  Iteration 2: Train on [1,3,4,5] → predict Fold 2
  ...
  Iteration 5: Train on [1,2,3,4] → predict Fold 5

Concatenate OOF predictions → meta-features for ALL training samples
Each prediction was made without seeing the corresponding training sample!
```

### 3.3 Implementation with sklearn

```python
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score

X, y = make_classification(
    n_samples=2000, n_features=20, n_informative=15,
    n_redundant=3, random_state=42
)

# Why: We choose models with fundamentally different learning strategies.
# LR learns linear boundaries, RF partitions feature space with trees,
# SVM maps to high-dimensional space, KNN uses local neighborhoods.
estimators = [
    ('lr', LogisticRegression(max_iter=1000, random_state=42)),
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
    ('svm', SVC(probability=True, random_state=42)),
    ('knn', KNeighborsClassifier(n_neighbors=10))
]

# Why: LogisticRegression as meta-learner — we only have 4 meta-features.
# cv=5 ensures out-of-fold predictions prevent data leakage.
stacking_clf = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(max_iter=1000),
    cv=5,
    stack_method='auto',     # uses predict_proba if available
    passthrough=False        # don't include original features
)

scores = cross_val_score(stacking_clf, X, y, cv=5, scoring='accuracy')
print(f"Stacking CV Accuracy: {scores.mean():.4f} (+/- {scores.std():.4f})")
```

### 3.4 StackingRegressor

```python
from sklearn.ensemble import StackingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.datasets import make_regression

X_reg, y_reg = make_regression(
    n_samples=2000, n_features=20, n_informative=15, noise=20, random_state=42
)

# Why: Ridge as meta-learner adds L2 regularization, preventing any single
# base model from dominating when base models are correlated.
stacking_reg = StackingRegressor(
    estimators=[
        ('ridge', Ridge(alpha=1.0)),
        ('rf', RandomForestRegressor(n_estimators=100, random_state=42)),
        ('svr', SVR(kernel='rbf')),
        ('knn', KNeighborsRegressor(n_neighbors=10))
    ],
    final_estimator=Ridge(alpha=1.0),
    cv=5
)

reg_scores = cross_val_score(stacking_reg, X_reg, y_reg, cv=5, scoring='r2')
print(f"Stacking CV R²: {reg_scores.mean():.4f} (+/- {reg_scores.std():.4f})")
```

### 3.5 The `passthrough` Option

```python
"""
passthrough=False (default):
  Meta-features = [pred_model1, pred_model2, ..., pred_modelN]

passthrough=True:
  Meta-features = [pred_model1, ..., pred_modelN, feature_1, ..., feature_20]

Use passthrough=True when:
  ✓ Base models are weak and miss important features
  ✓ You have enough data to support the larger feature space

Keep passthrough=False when:
  ✓ Base models already capture feature information well
  ✓ Small dataset (risk of overfitting with more meta-features)
"""
```

---

## 4. Blending vs. Stacking

### 4.1 What Is Blending?

Blending uses a single holdout set instead of cross-validation:

```
Training Data → split → Training (80%) + Blend Holdout (20%)
                            │                    │
                     Train base models     Predict on holdout
                            │                    │
                            │           Train meta-learner on
                            │           holdout predictions
                            │                    │
                     For test data: base models predict →
                     meta-learner combines → final prediction
```

### 4.2 Comparison

| Aspect | Stacking (CV-based) | Blending (Holdout-based) |
|--------|-------------------|------------------------|
| **Data efficiency** | High — uses all data | Lower — loses 20-30% |
| **Computation** | K times more expensive | Faster (single train) |
| **Variance** | Lower (averaged over folds) | Higher (single split) |
| **Best for** | Small-to-medium datasets | Large datasets |

### 4.3 Implementing Blending from Scratch

```python
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def blend_models(X_train, y_train, X_test, y_test, base_models, meta_model,
                 blend_ratio=0.2, random_state=42):
    """Manual blending: simpler than stacking, trades data for speed."""
    # Step 1: Split into train and blend sets
    X_tr, X_blend, y_tr, y_blend = train_test_split(
        X_train, y_train, test_size=blend_ratio, random_state=random_state,
        stratify=y_train  # Why: preserves class balance in both splits
    )

    # Step 2: Train base models, predict on blend + test
    blend_meta = np.zeros((len(X_blend), len(base_models)))
    test_meta = np.zeros((len(X_test), len(base_models)))

    for i, (name, model) in enumerate(base_models):
        model.fit(X_tr, y_tr)
        # Why: predict_proba[:, 1] carries more information than hard labels.
        # The meta-learner can learn "when SVM says 0.51, it's uncertain."
        if hasattr(model, 'predict_proba'):
            blend_meta[:, i] = model.predict_proba(X_blend)[:, 1]
            test_meta[:, i] = model.predict_proba(X_test)[:, 1]
        else:
            blend_meta[:, i] = model.predict(X_blend)
            test_meta[:, i] = model.predict(X_test)

    # Step 3: Train meta-learner on blend predictions
    meta_model.fit(blend_meta, y_blend)

    # Step 4: Final prediction
    return meta_model.predict(test_meta), accuracy_score(y_test, meta_model.predict(test_meta))
```

---

## 5. Multi-Level Stacking and Feature-Weighted Ensembles

### 5.1 Beyond Two Levels

Multi-level stacking adds additional meta-learner layers:

```
Level 0: [LR, RF, SVM, KNN, XGBoost]   ← Diverse base models
Level 1: [Ridge, RF_small]              ← First meta-layer
Level 2: [LogisticRegression]           ← Final meta-learner
```

**Practical guidelines**:
- 2 levels is the sweet spot (1-3% gain over single models)
- 3 levels: small further improvement (0.1-0.5%), only for competitions
- 4+ levels: almost never worth the complexity — overfitting risk grows

### 5.2 Meta-Feature Engineering

Beyond raw predictions, you can create richer meta-features:

```python
"""
Advanced meta-features per base model:
  - predict_proba[:, 1]          → probability of positive class
  - max(predict_proba)           → model confidence
  - -Σ p_i log(p_i)             → prediction entropy (uncertainty)
  - pred_model_A - pred_model_B  → pairwise disagreement signal

Why this helps: The meta-learner can learn "when Model A is confident
but Model B disagrees, trust Model B" — which simple probability
features alone cannot express.
"""
```

---

## 6. Choosing Diverse Base Learners

### 6.1 Measuring Prediction Diversity

The most practical diversity measure is the **correlation matrix of OOF predictions**:

```python
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_predict

def plot_prediction_correlation(models, X, y, cv=5):
    """
    Why this matters: High correlation means models make similar errors.
    Adding a highly correlated model adds redundancy, not diversity.
    Aim for pairwise correlation < 0.7.
    """
    predictions = {}
    for name, model in models:
        predictions[name] = cross_val_predict(model, X, y, cv=cv)

    corr = np.corrcoef(np.array(list(predictions.values())))
    sns.heatmap(corr, xticklabels=list(predictions.keys()),
                yticklabels=list(predictions.keys()),
                annot=True, fmt='.2f', cmap='RdYlGn_r', vmin=-1, vmax=1)
    plt.title('Base Model Prediction Correlation')
    plt.tight_layout()
    plt.show()
    return corr
```

### 6.2 Model Selection Strategy

```python
"""
Step 1: Start with canonical diverse models
  ┌────────────────────┬────────────────────────────────┐
  │ Model Family       │ Recommended Models             │
  ├────────────────────┼────────────────────────────────┤
  │ Linear             │ LogisticRegression, Ridge       │
  │ Tree-based         │ RandomForest, ExtraTrees        │
  │ Boosting           │ XGBoost, LightGBM, CatBoost     │
  │ Instance-based     │ KNN (k=5, 10, 20)              │
  │ Kernel-based       │ SVM (RBF, polynomial)           │
  │ Probabilistic      │ GaussianNB                      │
  └────────────────────┴────────────────────────────────┘

Step 2: Check prediction correlations
  - Correlation > 0.9 → remove one
  - Correlation 0.7-0.9 → keep but monitor
  - Correlation < 0.7 → excellent diversity

Step 3: Test incrementally
  - Start with 3 models, add one at a time
  - Stop adding when stacked performance plateaus

Anti-patterns:
  ✗ 5 Random Forests with different seeds → high correlation
  ✗ All tree-based models → similar decision boundaries
  ✗ Too many models (15+) → diminishing returns
"""
```

### 6.3 The Q-Statistic

```python
"""
Q_ij = (N11 × N00 - N01 × N10) / (N11 × N00 + N01 × N10)

N11 = both correct, N00 = both wrong, N01 = i correct/j wrong, N10 = i wrong/j correct

Q = 1 → always agree (no diversity)
Q = 0 → independent (good diversity)
Q < 0 → negatively correlated (excellent, rare in practice)

For stacking, aim for Q < 0.5 between all pairs.
"""
```

---

## 7. Hyperparameter Optimization for Stacked Ensembles

### 7.1 Staged Optimization Strategy

Optimizing everything jointly is computationally prohibitive. Use a staged approach:

```python
"""
Phase 1: Tune base models individually (high impact, 60-70% of HPO budget)
  - Tune each model independently using Optuna or GridSearchCV
  - Use the same CV split as the stacking CV

Phase 2: Select base models based on diversity (medium cost)
  - Compute prediction correlation matrix
  - Remove redundant models (correlation > 0.9)

Phase 3: Tune meta-learner (cheap, 1-2 hyperparameters)
  - LogisticRegression: tune C
  - Ridge: tune alpha

Phase 4: Tune structural choices (small impact)
  - passthrough: True vs False
  - stack_method: 'predict_proba' vs 'predict'

Why staged? Joint HPO over 50+ dimensions is intractable.
Staged: Σ(individual spaces) instead of Π(individual spaces).
"""
```

### 7.2 Optuna Example

```python
"""
import optuna

def objective(trial):
    rf_n = trial.suggest_int('rf_n_estimators', 50, 300, step=50)
    rf_depth = trial.suggest_int('rf_max_depth', 3, 15)
    svm_C = trial.suggest_float('svm_C', 0.01, 100, log=True)
    knn_k = trial.suggest_int('knn_k', 3, 30)
    meta_C = trial.suggest_float('meta_C', 0.01, 10, log=True)
    passthrough = trial.suggest_categorical('passthrough', [True, False])

    stack = StackingClassifier(
        estimators=[
            ('lr', LogisticRegression(max_iter=1000)),
            ('rf', RandomForestClassifier(n_estimators=rf_n, max_depth=rf_depth)),
            ('svm', SVC(probability=True, C=svm_C)),
            ('knn', KNeighborsClassifier(n_neighbors=knn_k)),
        ],
        final_estimator=LogisticRegression(C=meta_C, max_iter=1000),
        cv=5, passthrough=passthrough
    )
    return cross_val_score(stack, X, y, cv=5, scoring='accuracy').mean()

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)
"""
```

---

## 8. When NOT to Use Stacking

### 8.1 When It Helps vs. Doesn't

```python
"""
✓ Stacking helps when:
  1. Base models have comparable accuracy (within 2-3% of each other)
  2. Base models make different types of errors (correlation < 0.7)
  3. Dataset is large enough (1000+ samples)
  4. You need the last 0.5-1% of performance

✗ Stacking doesn't help when:
  1. One model dominates all others by > 3%
  2. All models are from the same family (low diversity)
  3. Dataset is very small (< 500 samples)
  4. Interpretability is required
  5. Inference latency is critical (all base models run at inference)
  6. The problem is easy (single model > 98%)
"""
```

### 8.2 Computational Cost

```python
"""
Stacking with 4 base models, 5-fold CV:
  Training: 4 × 5 = 20 fits (OOF) + 4 final fits + 1 meta-learner = 25 fits
  Inference: 4 + 1 = 5 predictions

Cost multiplier: ~25x training, ~5x inference vs single model

Production alternatives:
  - Model distillation: train a single model to mimic the stack
  - Use top 2-3 base models instead of all 5
  - Switch to weighted averaging (no meta-learner training)
"""
```

### 8.3 Decision Framework

```
Should I use stacking?
│
├── One model > 3% better than all others? → Use that model alone
├── < 1000 training samples? → Use simple averaging or best single model
├── Base model predictions diverse (corr < 0.7)? (if no → try different families)
├── Inference latency critical? → Consider distillation or weighted averaging
└── Otherwise → Use StackingClassifier with 3-5 diverse models
```

---

## 9. Competition Winning Strategies

### 9.1 Kaggle Stacking Patterns

```python
"""
Pattern 1: Standard 2-Level Stack (most common)
  Level 0: 5-10 models (GBM variants + NN + linear + KNN)
  Level 1: Linear model (LogisticRegression or Ridge)

Pattern 2: Blend of Stacks
  Stack A: [XGBoost, LightGBM, CatBoost] → Ridge
  Stack B: [NN1, NN2, NN3] → LR
  Final: Weighted average of Stack A, B predictions

Pattern 3: Feature Diversity Stacking
  Models trained on different feature subsets (original, PCA, target-encoded)
  Meta-learner combines all predictions
"""
```

### 9.2 Practical Tips

```python
"""
1. Always use OOF predictions — in-sample predictions cause leakage
2. Save OOF and test predictions as .npy files for fast experimentation
3. Average test predictions across K folds for each base model
4. Use the SAME CV split (StratifiedKFold, fixed seed) everywhere
5. Greedy ensemble selection often beats using all models:
   - Start with best model, greedily add models that improve CV score
   - 3-5 well-chosen models often beat 20+ random models
"""
```

---

## 10. Complete Stacking Workflow

```python
"""
1. SPLIT DATA → train_test_split + define StratifiedKFold
2. CHOOSE BASE MODELS → maximize algorithm diversity
3. EVALUATE individually → cross_val_score, remove models < 55% accuracy
4. MEASURE DIVERSITY → correlation matrix, remove pairs > 0.9
5. BUILD STACK → StackingClassifier(estimators, final_estimator, cv)
6. EVALUATE → cross_val_score, compare to best single model
7. TUNE → base models first, then meta-learner, then passthrough
8. PREDICT → stack.fit(X_train, y_train); stack.predict(X_test)
9. DEPLOY → full stack or distill to single model
"""
```

---

## Exercises

### Exercise 1: Build and Compare Ensemble Strategies

```python
"""
Using the Breast Cancer dataset (sklearn.datasets):

1. Train 5 individual models: LR, RF, GradientBoosting, SVM, KNN
2. Evaluate each with 5-fold cross-validation
3. Build three ensemble strategies:
   a. Simple averaging of predicted probabilities
   b. Blending with a 20% holdout
   c. Stacking with StackingClassifier (cv=5)
4. Compare all 8 approaches in a bar chart
5. Which strategy works best? Why?
Bonus: Does passthrough=True help?
"""
```

### Exercise 2: Diversity Analysis

```python
"""
Using make_classification (n_samples=3000, n_features=30):

1. Train 6 base models, generate OOF predictions
2. Plot the prediction correlation heatmap
3. Build two stacks:
   a. Stack A: 3 most diverse models (lowest correlation)
   b. Stack B: 3 least diverse models (highest correlation)
4. Compare performance — does diversity predict improvement?
"""
```

### Exercise 3: Stacking vs. Single Best Model

```python
"""
Experiment: When does stacking beat the best single model?

1. Generate datasets of sizes: 200, 500, 1000, 5000, 10000
2. Compare best single model vs StackingClassifier at each size
3. Plot performance gap vs dataset size
4. Plot training time ratio (stacking / single model)
5. Conclusion: At what size does stacking consistently win?
"""
```

### Exercise 4: Competition-Style Ensemble

```python
"""
1. Load sklearn.datasets.load_digits
2. Implement full stacking pipeline:
   a. Fixed StratifiedKFold, OOF predictions for 5+ models
   b. Save predictions as .npy arrays
   c. Train meta-learner, generate final predictions
3. Implement greedy ensemble selection
4. Compare: greedy selection vs. using all models
"""
```

---

## 11. Summary

### Key Takeaways

| Concept | Description |
|---------|-------------|
| **Stacking** | Trains a meta-learner to combine base model predictions |
| **Blending** | Simpler variant using holdout set instead of CV |
| **Diversity** | Different model errors = better ensemble; measure with correlation matrix |
| **Cross-validated OOF** | Prevents data leakage in meta-feature generation |
| **Meta-learner** | Keep it simple (linear model) for 3-5 base models |
| **Multi-level** | 2 levels is the sweet spot; 3+ rarely improves enough |
| **When to skip** | Small data, one dominant model, latency-critical, interpretability needed |

### Best Practices

1. **Start simple**: Try weighted averaging before building a full stack
2. **Maximize diversity**: Choose base models from different algorithm families
3. **Check correlation**: Remove redundant models (correlation > 0.9)
4. **Use OOF predictions**: Never use in-sample predictions as meta-features
5. **Keep the meta-learner simple**: Logistic/Ridge regression is almost always sufficient
6. **Tune in stages**: Base models first, then meta-learner, then structural choices
7. **Measure the gain**: If stacking improves < 0.2%, the complexity may not be worth it

### Connections to Other Lessons

- **L07 (Bagging)**: Stacking often uses Random Forest as a base learner
- **L08 (Boosting)**: XGBoost/LightGBM are among the strongest base models for stacking
- **L05 (Cross-Validation)**: OOF predictions rely on the same CV principles
- **L19 (AutoML)**: Some AutoML tools (Auto-sklearn, H2O) automatically build stacked ensembles
