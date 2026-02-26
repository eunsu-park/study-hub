# Pipelines and Practice

**Previous**: [Dimensionality Reduction](./12_Dimensionality_Reduction.md) | **Next**: [Practical Projects](./14_Practical_Projects.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain why sklearn Pipelines prevent data leakage and simplify ML workflows
2. Build pipelines that chain preprocessing, feature extraction, and model training into a single object
3. Apply ColumnTransformer to handle numeric and categorical features with different preprocessing steps
4. Integrate Pipelines with cross-validation and GridSearchCV for correct hyperparameter tuning
5. Implement custom transformers using BaseEstimator and TransformerMixin
6. Save and load trained pipelines using joblib, including version metadata
7. Design reusable pipeline templates for classification and regression problems

---

Knowing how to train a model is only half the story -- in production, the entire preprocessing-to-prediction chain must be reproducible, leak-free, and portable. Sklearn's Pipeline and ColumnTransformer solve this by encapsulating every transformation step into a single object that can be cross-validated, serialized, and deployed as a unit. This lesson bridges the gap between notebook experiments and production-grade ML code.

---

## 1. Pipeline Basics

### 1.1 Why Use Pipelines?

```python
"""
Problems when coding without pipelines:

1. Data Leakage:
   - Test data information leaks into training
   - Example: Scaling entire dataset before splitting

2. Code Complexity:
   - Manually managing multiple steps
   - High risk of errors

3. Reproducibility Issues:
   - Order mistakes
   - Parameter inconsistencies

Pipeline advantages:
1. Code simplification
2. Prevent data leakage
3. Perfect integration with cross-validation
4. Easy hyperparameter tuning
5. Convenient model saving/deployment
"""

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.datasets import load_iris
```

### 1.2 Creating a Basic Pipeline

```python
# Load data
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42
)

# Create pipeline (explicit names)
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=2)),
    ('classifier', LogisticRegression())
])

# Train and predict
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
score = pipeline.score(X_test, y_test)

print(f"Pipeline accuracy: {score:.4f}")

# make_pipeline (automatic names)
pipeline_auto = make_pipeline(
    StandardScaler(),
    PCA(n_components=2),
    LogisticRegression()
)

pipeline_auto.fit(X_train, y_train)
print(f"make_pipeline accuracy: {pipeline_auto.score(X_test, y_test):.4f}")
```

### 1.3 Accessing Pipeline Steps

```python
# Check step names
print("Pipeline steps:")
for name, step in pipeline.named_steps.items():
    print(f"  {name}: {type(step).__name__}")

# Access specific steps
print(f"\nPCA explained variance: {pipeline.named_steps['pca'].explained_variance_ratio_}")
print(f"Logistic regression coefficient shape: {pipeline.named_steps['classifier'].coef_.shape}")

# Get intermediate step results
X_scaled = pipeline.named_steps['scaler'].transform(X_test)
X_pca = pipeline.named_steps['pca'].transform(X_scaled)
print(f"\nShape after scaling: {X_scaled.shape}")
print(f"Shape after PCA: {X_pca.shape}")
```

---

## 2. ColumnTransformer

### 2.1 Handling Different Feature Types

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

"""
ColumnTransformer:
- Apply different preprocessing to different feature types
- Numeric: Scaling
- Categorical: Encoding
"""

# Sample data
data = {
    'age': [25, 32, 47, 51, 62],
    'income': [50000, 60000, 80000, 120000, 95000],
    'gender': ['M', 'F', 'M', 'F', 'M'],
    'education': ['Bachelor', 'Master', 'PhD', 'Bachelor', 'Master'],
    'purchased': [0, 1, 1, 1, 0]
}
df = pd.DataFrame(data)

X = df.drop('purchased', axis=1)
y = df['purchased']

print("Data types:")
print(X.dtypes)
```

### 2.2 Creating ColumnTransformer

```python
# Classify features
numeric_features = ['age', 'income']
categorical_features = ['gender', 'education']

# Define ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_features)
    ],
    remainder='passthrough'  # Handle remaining features: 'drop', 'passthrough'
)

# Transform
X_transformed = preprocessor.fit_transform(X)

print(f"Original shape: {X.shape}")
print(f"Transformed shape: {X_transformed.shape}")

# Transformed feature names
feature_names = (
    numeric_features +
    list(preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features))
)
print(f"Feature names: {feature_names}")
```

### 2.3 Pipeline + ColumnTransformer

```python
from sklearn.ensemble import RandomForestClassifier

# Complete pipeline
full_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Train (using entire dataset due to small size)
full_pipeline.fit(X, y)

# Predict
new_data = pd.DataFrame({
    'age': [30],
    'income': [70000],
    'gender': ['F'],
    'education': ['Master']
})
prediction = full_pipeline.predict(new_data)
print(f"Prediction: {prediction[0]}")
```

---

## 3. Complex Preprocessing Pipelines

### 3.1 Including Missing Value Handling

```python
from sklearn.impute import SimpleImputer

# Data with missing values
data_missing = {
    'age': [25, np.nan, 47, 51, 62],
    'income': [50000, 60000, np.nan, 120000, 95000],
    'gender': ['M', 'F', 'M', None, 'M'],
    'education': ['Bachelor', 'Master', 'PhD', 'Bachelor', None],
    'purchased': [0, 1, 1, 1, 0]
}
df_missing = pd.DataFrame(data_missing)
X_missing = df_missing.drop('purchased', axis=1)
y_missing = df_missing['purchased']

# Numeric pipeline
numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Categorical pipeline
categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
])

# ColumnTransformer
preprocessor_full = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# Complete pipeline
complete_pipeline = Pipeline([
    ('preprocessor', preprocessor_full),
    ('classifier', RandomForestClassifier(random_state=42))
])

complete_pipeline.fit(X_missing, y_missing)
print("Pipeline with missing values trained successfully")
```

### 3.2 Including Feature Selection

```python
from sklearn.feature_selection import SelectKBest, f_classif

# Pipeline with feature selection
pipeline_with_selection = Pipeline([
    ('preprocessor', preprocessor_full),
    ('feature_selection', SelectKBest(score_func=f_classif, k='all')),
    ('classifier', RandomForestClassifier(random_state=42))
])

pipeline_with_selection.fit(X_missing, y_missing)
print("Pipeline with feature selection trained successfully")
```

---

## 4. Pipeline with Cross-Validation

### 4.1 Correct Cross-Validation

```python
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.datasets import load_breast_cancer

# Load data
cancer = load_breast_cancer()
X, y = cancer.data, cancer.target

# Define pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression(max_iter=1000))
])

# Cross-validation (correct way)
# Scaler is fit only on training data in each fold
scores = cross_val_score(pipeline, X, y, cv=5, scoring='accuracy')

print("Cross-validation results:")
print(f"  Each fold: {scores}")
print(f"  Mean: {scores.mean():.4f} (+/- {scores.std():.4f})")
```

### 4.2 Pipeline Hyperparameter Tuning

```python
# Parameter names: step__parameter
param_grid = {
    'scaler': [StandardScaler(), MinMaxScaler()],
    'classifier__C': [0.1, 1, 10],
    'classifier__penalty': ['l1', 'l2'],
    'classifier__solver': ['liblinear']
}

# Grid Search
grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

grid_search.fit(X, y)

print("Grid Search results:")
print(f"  Best parameters: {grid_search.best_params_}")
print(f"  Best score: {grid_search.best_score_:.4f}")
```

### 4.3 Complex Parameter Grid

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# Multi-model comparison pipeline
pipeline_multi = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression())  # placeholder
])

# Different parameters per model
param_grid_multi = [
    {
        'classifier': [LogisticRegression(max_iter=1000)],
        'classifier__C': [0.1, 1, 10]
    },
    {
        'classifier': [RandomForestClassifier(random_state=42)],
        'classifier__n_estimators': [50, 100],
        'classifier__max_depth': [None, 5, 10]
    },
    {
        'classifier': [SVC()],
        'classifier__C': [0.1, 1],
        'classifier__kernel': ['rbf', 'linear']
    }
]

grid_search_multi = GridSearchCV(
    pipeline_multi,
    param_grid_multi,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

grid_search_multi.fit(X, y)

print("Multi-model comparison results:")
print(f"  Best model: {type(grid_search_multi.best_params_['classifier']).__name__}")
print(f"  Best parameters: {grid_search_multi.best_params_}")
print(f"  Best score: {grid_search_multi.best_score_:.4f}")
```

---

## 5. Model Saving and Loading

### 5.1 Using joblib

```python
import joblib

# Train best model
best_pipeline = grid_search.best_estimator_

# Save model
joblib.dump(best_pipeline, 'best_model.joblib')
print("Model saved: best_model.joblib")

# Load model
loaded_model = joblib.load('best_model.joblib')

# Test
X_test_sample = X[:5]
predictions = loaded_model.predict(X_test_sample)
print(f"Loaded model predictions: {predictions}")
```

### 5.2 Using pickle

```python
import pickle

# Save with pickle
with open('model.pkl', 'wb') as f:
    pickle.dump(best_pipeline, f)

# Load with pickle
with open('model.pkl', 'rb') as f:
    loaded_model_pkl = pickle.load(f)

print("Pickle model predictions:", loaded_model_pkl.predict(X[:3]))
```

### 5.3 Version Control

```python
import sklearn
from datetime import datetime

# Save with metadata
model_metadata = {
    'model': best_pipeline,
    'sklearn_version': sklearn.__version__,
    'training_date': datetime.now().isoformat(),
    'feature_names': list(cancer.feature_names),
    'target_names': list(cancer.target_names),
    'cv_score': grid_search.best_score_
}

joblib.dump(model_metadata, 'model_with_metadata.joblib')

# Load and verify
loaded_metadata = joblib.load('model_with_metadata.joblib')
print(f"Training date: {loaded_metadata['training_date']}")
print(f"sklearn version: {loaded_metadata['sklearn_version']}")
print(f"CV score: {loaded_metadata['cv_score']:.4f}")
```

---

## 6. FunctionTransformer

### 6.1 Custom Transformation Functions

```python
from sklearn.preprocessing import FunctionTransformer

# Custom transformation functions
def log_transform(X):
    return np.log1p(X)  # log(1 + x)

def add_polynomial_features(X):
    return np.c_[X, X ** 2, X ** 3]

# Create FunctionTransformer
log_transformer = FunctionTransformer(log_transform, validate=True)
poly_transformer = FunctionTransformer(add_polynomial_features, validate=True)

# Use in pipeline
pipeline_custom = Pipeline([
    ('log', log_transformer),
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression(max_iter=1000))
])

# Test
X_positive = np.abs(X) + 1  # Convert to positive for log
scores = cross_val_score(pipeline_custom, X_positive, y, cv=5)
print(f"Custom transformation pipeline CV score: {scores.mean():.4f}")
```

### 6.2 Feature Addition Function

```python
# Domain-specific feature addition
def create_ratio_features(X):
    """Create ratio features"""
    X = np.array(X)
    if X.shape[1] >= 2:
        ratio = (X[:, 0] / (X[:, 1] + 1e-10)).reshape(-1, 1)
        return np.c_[X, ratio]
    return X

ratio_transformer = FunctionTransformer(create_ratio_features)

# Pipeline
pipeline_ratio = Pipeline([
    ('ratio_features', ratio_transformer),
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression(max_iter=1000))
])

scores = cross_val_score(pipeline_ratio, X, y, cv=5)
print(f"Ratio feature addition CV score: {scores.mean():.4f}")
```

---

## 7. Custom Transformers

```python
from sklearn.base import BaseEstimator, TransformerMixin

class OutlierRemover(BaseEstimator, TransformerMixin):
    """Outlier removal transformer"""

    def __init__(self, threshold=3):
        self.threshold = threshold
        self.mean_ = None
        self.std_ = None

    def fit(self, X, y=None):
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)
        return self

    def transform(self, X):
        X = np.array(X)
        z_scores = np.abs((X - self.mean_) / (self.std_ + 1e-10))
        # Replace outliers with boundary values
        X_clipped = np.where(z_scores > self.threshold,
                             self.mean_ + self.threshold * self.std_ * np.sign(X - self.mean_),
                             X)
        return X_clipped


class FeatureSelector(BaseEstimator, TransformerMixin):
    """Feature selection transformer"""

    def __init__(self, feature_indices=None):
        self.feature_indices = feature_indices

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = np.array(X)
        if self.feature_indices is not None:
            return X[:, self.feature_indices]
        return X


# Use custom transformers
custom_pipeline = Pipeline([
    ('outlier', OutlierRemover(threshold=3)),
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression(max_iter=1000))
])

scores = cross_val_score(custom_pipeline, X, y, cv=5)
print(f"Custom transformer CV score: {scores.mean():.4f}")
```

---

## 8. Practical Preprocessing Templates

### 8.1 Classification Problem Template

```python
from sklearn.compose import make_column_selector

def create_classification_pipeline(model, numeric_features=None, categorical_features=None):
    """Create pipeline for classification problems"""

    # Numeric feature pipeline
    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Categorical feature pipeline
    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
    ])

    # ColumnTransformer
    if numeric_features is None and categorical_features is None:
        # Auto-detect
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, make_column_selector(dtype_include=np.number)),
                ('cat', categorical_transformer, make_column_selector(dtype_include=object))
            ]
        )
    else:
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features or []),
                ('cat', categorical_transformer, categorical_features or [])
            ]
        )

    # Complete pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])

    return pipeline


# Usage example
from sklearn.ensemble import GradientBoostingClassifier

pipeline = create_classification_pipeline(
    GradientBoostingClassifier(random_state=42),
    numeric_features=['age', 'income'],
    categorical_features=['gender', 'education']
)
```

### 8.2 Regression Problem Template

```python
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score

def create_regression_pipeline(model, numeric_features=None, categorical_features=None):
    """Create pipeline for regression problems"""

    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))
    ])

    if numeric_features is None and categorical_features is None:
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, make_column_selector(dtype_include=np.number)),
                ('cat', categorical_transformer, make_column_selector(dtype_include=object))
            ]
        )
    else:
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features or []),
                ('cat', categorical_transformer, categorical_features or [])
            ]
        )

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', model)
    ])

    return pipeline
```

---

## 9. Deployment Considerations

### 9.1 Wrapping Prediction Function

```python
class ModelWrapper:
    """Model wrapper for deployment"""

    def __init__(self, model_path):
        self.model = joblib.load(model_path)
        self.feature_names = None

    def set_feature_names(self, names):
        self.feature_names = names

    def predict(self, input_data):
        """Handle dictionary or DataFrame input"""
        if isinstance(input_data, dict):
            input_data = pd.DataFrame([input_data])

        if self.feature_names:
            input_data = input_data[self.feature_names]

        return self.model.predict(input_data)

    def predict_proba(self, input_data):
        if isinstance(input_data, dict):
            input_data = pd.DataFrame([input_data])

        if self.feature_names:
            input_data = input_data[self.feature_names]

        return self.model.predict_proba(input_data)


# Usage example
# wrapper = ModelWrapper('best_model.joblib')
# wrapper.set_feature_names(['age', 'income', 'gender', 'education'])
# prediction = wrapper.predict({'age': 30, 'income': 70000, 'gender': 'M', 'education': 'Bachelor'})
```

### 9.2 Input Validation

```python
def validate_input(data, expected_columns, expected_dtypes=None):
    """Validate input data"""
    errors = []

    # Check required columns
    missing_cols = set(expected_columns) - set(data.columns)
    if missing_cols:
        errors.append(f"Missing columns: {missing_cols}")

    # Check data types
    if expected_dtypes:
        for col, dtype in expected_dtypes.items():
            if col in data.columns and not np.issubdtype(data[col].dtype, dtype):
                errors.append(f"Wrong type - {col}: {data[col].dtype} (expected: {dtype})")

    # Check missing values
    null_counts = data[expected_columns].isnull().sum()
    null_cols = null_counts[null_counts > 0]
    if len(null_cols) > 0:
        print(f"Warning: Missing values found - {dict(null_cols)}")

    if errors:
        raise ValueError("\n".join(errors))

    return True
```

---

## 10. Practical Checklist

```python
"""
ML Project Checklist:

1. Data Preparation
   [ ] Load and explore data
   [ ] Define target variable
   [ ] Split into train/validation/test

2. Exploratory Data Analysis (EDA)
   [ ] Check missing values
   [ ] Check outliers
   [ ] Check feature distributions
   [ ] Correlation with target

3. Preprocessing Pipeline
   [ ] Handle numeric features (scaling, missing values)
   [ ] Handle categorical features (encoding, missing values)
   [ ] Feature selection/creation

4. Modeling
   [ ] Set baseline model
   [ ] Compare multiple models
   [ ] Hyperparameter tuning
   [ ] Cross-validation

5. Evaluation
   [ ] Choose appropriate metrics
   [ ] Check for overfitting/underfitting
   [ ] Error analysis

6. Deployment
   [ ] Save model
   [ ] Input validation
   [ ] Wrap prediction function
   [ ] Monitoring plan
"""
```

---

## Exercises

### Exercise 1: Basic Pipeline
Create a pipeline with scaling + PCA + logistic regression for Iris data.

```python
from sklearn.datasets import load_iris
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

iris = load_iris()
X, y = iris.data, iris.target

# Solution
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=2)),
    ('classifier', LogisticRegression())
])

scores = cross_val_score(pipeline, X, y, cv=5)
print(f"CV score: {scores.mean():.4f}")
```

### Exercise 2: ColumnTransformer
Create a pipeline that handles numeric and categorical features differently.

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

# Sample data
data = pd.DataFrame({
    'age': [25, 30, 35, 40],
    'income': [50000, 60000, 70000, 80000],
    'city': ['A', 'B', 'A', 'C']
})

# Solution
numeric_features = ['age', 'income']
categorical_features = ['city']

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric_features),
    ('cat', OneHotEncoder(), categorical_features)
])

X_transformed = preprocessor.fit_transform(data)
print(f"Transformed shape: {X_transformed.shape}")
```

### Exercise 3: Model Saving and Loading
Save and load a trained pipeline.

```python
import joblib

# Train
pipeline.fit(X, y)

# Save
joblib.dump(pipeline, 'iris_pipeline.joblib')

# Load
loaded_pipeline = joblib.load('iris_pipeline.joblib')

# Test
print(f"Loaded model accuracy: {loaded_pipeline.score(X, y):.4f}")
```

---

## Summary

| Component | Purpose | Example |
|-----------|---------|---------|
| Pipeline | Sequential step connection | Scaling → PCA → Model |
| ColumnTransformer | Different processing per feature | Separate numeric/categorical |
| FunctionTransformer | Custom functions | Log transform |
| make_pipeline | Automatic naming | Simple pipelines |

### Pipeline Hyperparameter Naming Convention

```
step_name__parameter_name

Examples:
- classifier__C: Classifier's C parameter
- preprocessor__num__scaler__with_mean: Nested parameter
```

### Model Saving Comparison

| Method | Pros | Cons |
|--------|------|------|
| joblib | Efficient for large NumPy arrays | sklearn-specific |
| pickle | Standard library | Slow for large datasets |
| ONNX | Framework-independent | Requires conversion |

### Practical Tips

1. Always use Pipeline to prevent data leakage
2. Clearly separate preprocessing with ColumnTransformer
3. Include metadata when saving models
4. Write input validation functions
5. Maintain thorough version control
