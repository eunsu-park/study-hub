"""
Pipelines and Practice - Exercise Solutions
=============================================
Lesson 13: Pipelines and Practice

Exercises cover:
  1. Basic Pipeline: scaling + PCA + logistic regression
  2. ColumnTransformer: handle numeric and categorical features
  3. Model saving and loading with joblib
"""

import numpy as np
import pandas as pd
import joblib
import os
from sklearn.datasets import load_iris
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import cross_val_score


# ============================================================
# Exercise 1: Basic Pipeline
# Scaling + PCA + Logistic Regression for Iris data.
# ============================================================
def exercise_1_basic_pipeline():
    """Build a pipeline that chains scaling, PCA, and classification.

    Pipelines ensure that preprocessing and modeling happen together:
    1. Prevents data leakage: scaler.fit() only sees training data in each
       CV fold, never the validation fold
    2. Simplifies code: single .fit()/.predict() call for the whole chain
    3. Enables GridSearchCV on the entire chain (e.g., tune PCA n_components
       and classifier C simultaneously)
    """
    print("=" * 60)
    print("Exercise 1: Basic Pipeline (Scale + PCA + LR)")
    print("=" * 60)

    iris = load_iris()
    X, y = iris.data, iris.target

    # Method 1: Explicit Pipeline with named steps
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=2)),
        ("classifier", LogisticRegression(max_iter=1000, random_state=42))
    ])

    scores = cross_val_score(pipeline, X, y, cv=5, scoring="accuracy")
    print(f"Named Pipeline CV: {scores.mean():.4f} +/- {scores.std():.4f}")

    # Method 2: make_pipeline auto-generates step names
    pipe2 = make_pipeline(
        StandardScaler(),
        PCA(n_components=2),
        LogisticRegression(max_iter=1000, random_state=42)
    )
    scores2 = cross_val_score(pipe2, X, y, cv=5, scoring="accuracy")
    print(f"make_pipeline CV:  {scores2.mean():.4f} +/- {scores2.std():.4f}")

    # GridSearchCV with pipeline -- search over PCA components and C
    from sklearn.model_selection import GridSearchCV
    param_grid = {
        "pca__n_components": [2, 3, 4],
        "classifier__C": [0.1, 1, 10],
    }
    grid = GridSearchCV(pipeline, param_grid, cv=5, scoring="accuracy")
    grid.fit(X, y)
    print(f"\nBest params: {grid.best_params_}")
    print(f"Best CV score: {grid.best_score_:.4f}")


# ============================================================
# Exercise 2: ColumnTransformer
# Handle numeric and categorical features differently.
# ============================================================
def exercise_2_column_transformer():
    """ColumnTransformer applies different preprocessing per feature type.

    Real datasets have mixed types: numeric features need scaling,
    categorical features need encoding. ColumnTransformer lets you
    define separate sub-pipelines for each group and concatenates
    the results into a single feature matrix.

    This is essential for production ML -- it ensures the same
    transformations are applied consistently during training and inference.
    """
    print("\n" + "=" * 60)
    print("Exercise 2: ColumnTransformer")
    print("=" * 60)

    # Sample dataset with mixed types
    data = pd.DataFrame({
        "age": [25, 30, 35, 40, 28, 55, 22, 45],
        "income": [50000, 60000, 70000, 80000, 55000, 95000, 42000, 85000],
        "city": ["NYC", "LA", "NYC", "Chicago", "LA", "NYC", "Chicago", "LA"],
        "plan": ["basic", "premium", "basic", "premium", "basic", "premium",
                 "basic", "premium"],
    })

    print("Input data:")
    print(data)

    numeric_features = ["age", "income"]
    categorical_features = ["city", "plan"]

    # Define per-column preprocessing
    preprocessor = ColumnTransformer([
        # Numeric: scale to zero mean, unit variance
        ("num", StandardScaler(), numeric_features),
        # Categorical: one-hot encode (sparse=False for dense output)
        ("cat", OneHotEncoder(sparse_output=False, drop="first"),
         categorical_features),
    ])

    X_transformed = preprocessor.fit_transform(data)

    # Get feature names for the transformed output
    cat_names = preprocessor.named_transformers_["cat"].get_feature_names_out(
        categorical_features
    )
    all_names = list(numeric_features) + list(cat_names)

    print(f"\nTransformed shape: {X_transformed.shape}")
    print(f"Feature names: {all_names}")
    print(f"\nTransformed data (first 3 rows):")
    for i in range(3):
        vals = ", ".join(f"{v:.3f}" for v in X_transformed[i])
        print(f"  [{vals}]")

    # Full pipeline: ColumnTransformer + Classifier
    full_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000, random_state=42))
    ])
    print(f"\nFull pipeline steps: {[s[0] for s in full_pipeline.steps]}")


# ============================================================
# Exercise 3: Model Saving and Loading
# Save and load a trained pipeline with joblib.
# ============================================================
def exercise_3_model_saving():
    """Save and load a trained pipeline with joblib.

    joblib is preferred over pickle for sklearn models because:
    1. More efficient serialization of large NumPy arrays (common in models)
    2. Optional compression to reduce file size

    Best practice: save the entire pipeline (preprocessing + model) so that
    prediction at inference time requires only raw input, not manual
    preprocessing.
    """
    print("\n" + "=" * 60)
    print("Exercise 3: Model Saving and Loading")
    print("=" * 60)

    iris = load_iris()
    X, y = iris.data, iris.target

    # Build and train pipeline
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=2)),
        ("classifier", LogisticRegression(max_iter=1000, random_state=42))
    ])
    pipeline.fit(X, y)

    original_score = pipeline.score(X, y)
    print(f"Original model accuracy: {original_score:.4f}")

    # Save
    model_path = "/tmp/iris_pipeline.joblib"
    joblib.dump(pipeline, model_path)
    file_size = os.path.getsize(model_path)
    print(f"Model saved to: {model_path}")
    print(f"File size: {file_size / 1024:.1f} KB")

    # Load and verify
    loaded_pipeline = joblib.load(model_path)
    loaded_score = loaded_pipeline.score(X, y)
    print(f"Loaded model accuracy: {loaded_score:.4f}")
    assert abs(original_score - loaded_score) < 1e-10, "Scores should match exactly"

    # Predict with loaded model
    sample = X[:3]
    predictions = loaded_pipeline.predict(sample)
    probas = loaded_pipeline.predict_proba(sample)
    print(f"\nSample predictions:")
    for i in range(3):
        probs = ", ".join(f"{p:.3f}" for p in probas[i])
        print(f"  Sample {i}: predicted={iris.target_names[predictions[i]]}, "
              f"probabilities=[{probs}]")

    # Clean up
    os.remove(model_path)
    print(f"\nCleaned up: {model_path}")


if __name__ == "__main__":
    exercise_1_basic_pipeline()
    exercise_2_column_transformer()
    exercise_3_model_saving()
