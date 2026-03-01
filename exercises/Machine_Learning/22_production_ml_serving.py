"""
Exercises: Production ML — Model Serving Patterns
===================================================

Practice problems for preparing ML models for production deployment.

Requirements:
    pip install scikit-learn numpy scipy
"""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Shared dataset for all exercises
X, y = make_classification(
    n_samples=10000, n_features=30, n_informative=20,
    n_redundant=5, random_state=42
)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# ============================================================
# Exercise 1: Model Optimization Benchmark
# ============================================================
"""
1. Train a GradientBoostingClassifier(n_estimators=300, max_depth=6)
2. Measure: accuracy, single-sample latency (p50, p99), model size on disk
3. Create optimized variants:
   a. Fewer trees (n_estimators=50)
   b. Shallower trees (max_depth=3)
   c. Knowledge distillation to LogisticRegression
   d. joblib compression (compress=5)
4. Print a comparison table: accuracy, latency, size for all variants
5. Which variant offers the best trade-off for a 50ms latency budget?
"""

# Your code here:


# ============================================================
# Exercise 2: Training-Serving Skew Prevention
# ============================================================
"""
1. Inject 5% missing values randomly into the dataset
2. Build TWO approaches:
   a. CORRECT: sklearn Pipeline (SimpleImputer + StandardScaler + RF)
   b. WRONG: Separate preprocessing with different imputation at serve time
3. Save the pipeline with joblib, reload it, and verify predictions match
4. Measure the accuracy difference between correct and skewed approaches
5. Print: pipeline accuracy, skewed accuracy, accuracy drop (%)
"""

# Your code here:


# ============================================================
# Exercise 3: Drift Detection System
# ============================================================
"""
1. Generate reference data: 2000 samples, 10 features, standard normal
2. Generate 4 production scenarios:
   a. No drift (same distribution)
   b. Mean shift (+0.5) in features 0, 3
   c. Variance change (×2) in features 1, 4, 7
   d. Combined shift in features 2, 5 (mean +1.0, variance ×1.5)
3. For each scenario, run KS test on every feature
4. Report: which features are correctly flagged as drifted?
5. Bonus: Implement PSI and compare sensitivity with KS test
"""

# Your code here:


# ============================================================
# Exercise 4: Latency-Accuracy Pareto Frontier
# ============================================================
"""
1. Train 8+ models of varying complexity:
   - LogisticRegression, DecisionTree (depth=3, 10, 20),
   - RandomForest (10, 50, 200 trees), GBM (50, 200 trees)
2. For each model, measure accuracy and single-sample latency (p50)
3. Identify Pareto-optimal models:
   A model is Pareto-optimal if no other model is both faster AND
   more accurate.
4. Print results table with Pareto-optimal models marked
5. Given a 30ms latency budget, which model should you choose?
"""

# Your code here:


# ============================================================
# Exercise 5: Model Artifact Packaging
# ============================================================
"""
1. Train a RandomForestClassifier with a full Pipeline (imputer + scaler + model)
2. Create a model metadata dictionary containing:
   - model_version, training_date, n_training_samples
   - feature_names, target_classes, performance_metrics
   - serialization_format, expected_latency_ms
3. Save both the pipeline (joblib) and metadata (JSON) to /tmp/
4. Load them back and verify:
   - Predictions match the original
   - Metadata is valid JSON with all required fields
5. Implement an input validation function that checks:
   - Correct number of features
   - No NaN or Inf values (if imputer is not in pipeline)
   - Data type is float
"""

# Your code here:
