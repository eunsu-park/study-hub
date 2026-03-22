# Machine Learning Overview

**Next**: [Linear Regression](./02_Linear_Regression.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain what machine learning is and how it differs from traditional programming
2. Distinguish between supervised, unsupervised, and reinforcement learning
3. Describe the end-to-end machine learning workflow from problem definition to deployment
4. Identify overfitting and underfitting and explain the bias-variance tradeoff
5. Apply train/validation/test splitting strategies correctly
6. Demonstrate the use of scikit-learn's estimator and transformer interfaces
7. Implement feature scaling using StandardScaler and MinMaxScaler

---

Every day, you interact with machine learning systems -- from email spam filters to product recommendations to voice assistants. What makes these systems powerful is their ability to learn patterns from data rather than following hand-coded rules. Understanding the core concepts, workflow, and pitfalls of ML is the essential first step before diving into any specific algorithm.

---

## 1. What is Machine Learning?

Machine Learning is an algorithm that learns from data to perform predictions or decisions without explicit programming.

```python
# Traditional Programming vs Machine Learning
# Traditional: Rules + Data → Results
# Machine Learning: Data + Results → Rules (Model)
```

---

## 2. Types of Machine Learning

### 2.1 Supervised Learning

Learning with labeled data (input X and target y).

```python
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Example: Predict house price from size
X = np.array([[50], [60], [70], [80], [90], [100]])  # Size (pyeong)
y = np.array([1.5, 1.8, 2.1, 2.5, 2.8, 3.2])  # Price (100M KRW)

# Train model
model = LinearRegression()
model.fit(X, y)

# Predict
new_house = [[75]]
predicted_price = model.predict(new_house)
print(f"Predicted price for 75 pyeong house: {predicted_price[0]:.2f} 100M KRW")
```

**Main Algorithms:**
- **Regression**: Predict continuous values
  - Linear regression, polynomial regression, ridge, lasso
- **Classification**: Predict categories
  - Logistic regression, SVM, decision trees, random forest

### 2.2 Unsupervised Learning

Learning patterns or structures in data without labels.

```python
from sklearn.cluster import KMeans
import numpy as np

# Customer data (age, purchase amount)
X = np.array([[25, 100], [30, 150], [35, 120],
              [50, 300], [55, 350], [60, 400]])

# K-Means clustering
kmeans = KMeans(n_clusters=2, random_state=42)
labels = kmeans.fit_predict(X)

print(f"Cluster labels: {labels}")
print(f"Cluster centers:\n{kmeans.cluster_centers_}")
```

**Main Algorithms:**
- **Clustering**: K-Means, DBSCAN, hierarchical clustering
- **Dimensionality Reduction**: PCA, t-SNE
- **Anomaly Detection**: Isolation Forest

### 2.3 Reinforcement Learning

Learning by interacting with environment to maximize rewards.

- Agent selects actions
- Receives rewards or penalties from environment
- Maximizes cumulative reward

**Applications:** Game AI, robot control, autonomous driving

---

## 3. Machine Learning Workflow

```
1. Problem Definition → 2. Data Collection → 3. Data Exploration (EDA)
                                        ↓
        7. Deployment/Monitoring ← 6. Model Selection ← 5. Model Training ← 4. Data Preprocessing
```

### 3.1 Basic Workflow Example

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# 1. Load data
from sklearn.datasets import load_iris
iris = load_iris()
X, y = iris.data, iris.target

# 2. Split data BEFORE scaling to prevent data leakage — if we scale on the
# full dataset, test set statistics (mean, std) leak into training and the
# reported accuracy becomes overly optimistic
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42  # Fixes random seed for reproducible train/test splits across runs
)

# 3. Data preprocessing (scaling)
# Standardization (zero mean, unit variance) ensures gradient-based optimizers
# converge efficiently; features with larger scales would dominate the loss
# landscape otherwise, causing slow or biased training
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # transform only — no refitting on test data

# 4. Train model
model = LogisticRegression(max_iter=200)
model.fit(X_train_scaled, y_train)

# 5. Predict
y_pred = model.predict(X_test_scaled)

# 6. Evaluate
# Accuracy alone is misleading for imbalanced classes — classification_report
# shows per-class precision/recall to reveal if the model ignores minority classes
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))
```

---

## 4. Core Concepts

### 4.1 Train/Validation/Test Split

```python
from sklearn.model_selection import train_test_split

# Split data into train (60%), validation (20%), test (20%)
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, random_state=42  # 0.25 * 0.8 = 0.2
)

print(f"Train: {len(X_train)}, Validation: {len(X_val)}, Test: {len(X_test)}")
```

- **Training data**: Used for model training
- **Validation data**: Used for hyperparameter tuning
- **Test data**: Used for final performance evaluation (only once)

### 4.2 Overfitting and Underfitting

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Generate data
np.random.seed(42)
X = np.sort(np.random.rand(20, 1) * 6, axis=0)
y = np.sin(X).ravel() + np.random.randn(20) * 0.1

# Models with different complexity
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
degrees = [1, 4, 15]
titles = ['Underfitting', 'Good Fit', 'Overfitting']

for ax, degree, title in zip(axes, degrees, titles):
    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X)
    model = LinearRegression()
    model.fit(X_poly, y)

    X_plot = np.linspace(0, 6, 100).reshape(-1, 1)
    X_plot_poly = poly.transform(X_plot)
    y_plot = model.predict(X_plot_poly)

    ax.scatter(X, y, color='blue', alpha=0.7)
    ax.plot(X_plot, y_plot, color='red', linewidth=2)
    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('y')

plt.tight_layout()
plt.show()
```

- **Underfitting**: Model is too simple to learn even training data well
- **Overfitting**: Model is too fitted to training data, fails to generalize to new data

### 4.3 Bias-Variance Tradeoff

```
Total Error = Bias² + Variance + Noise

Bias: Error due to model simplicity
Variance: Sensitivity of model to data changes

High bias → Underfitting   (model too rigid to capture the underlying signal)
High variance → Overfitting  (model memorizes noise in the training data)
```

### 4.4 Feature Scaling

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Example data
X = np.array([[100, 0.001], [200, 0.002], [300, 0.003]])

# StandardScaler (Z-score normalization)
scaler_std = StandardScaler()
X_std = scaler_std.fit_transform(X)
print("StandardScaler result:")
print(X_std)

# MinMaxScaler (0-1 normalization)
scaler_minmax = MinMaxScaler()
X_minmax = scaler_minmax.fit_transform(X)
print("\nMinMaxScaler result:")
print(X_minmax)
```

---

## 5. sklearn Basic API

### 5.1 Estimator Interface

```python
# All sklearn models follow the same interface
from sklearn.ensemble import RandomForestClassifier

# 1. Create model
model = RandomForestClassifier(n_estimators=100, random_state=42)

# 2. Train (fit)
model.fit(X_train, y_train)

# 3. Predict (predict)
y_pred = model.predict(X_test)

# 4. Predict probability (predict_proba) - classification models
y_proba = model.predict_proba(X_test)

# 5. Score (score)
accuracy = model.score(X_test, y_test)
```

### 5.2 Transformer Interface

```python
from sklearn.preprocessing import StandardScaler

# 1. Create transformer
scaler = StandardScaler()

# 2. Fit (fit)
scaler.fit(X_train)

# 3. Transform (transform)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# fit + transform together
X_train_scaled = scaler.fit_transform(X_train)
# Warning: only transform on test data!
X_test_scaled = scaler.transform(X_test)
```

---

## 6. Datasets

### 6.1 sklearn Built-in Datasets

```python
from sklearn.datasets import (
    load_iris,        # Classification (3 classes)
    load_digits,      # Classification (10 classes)
    load_breast_cancer,  # Binary classification
    load_boston,      # Regression (deprecated)
    load_diabetes,    # Regression
    make_classification,  # Synthetic classification data
    make_regression,      # Synthetic regression data
)

# Iris dataset
iris = load_iris()
print(f"Features: {iris.feature_names}")
print(f"Targets: {iris.target_names}")
print(f"Data shape: {iris.data.shape}")

# Generate synthetic data
X, y = make_classification(
    n_samples=1000,
    n_features=20,
    n_informative=10,
    n_redundant=5,
    n_classes=2,
    random_state=42
)
print(f"Synthetic data shape: {X.shape}")
```

### 6.2 Load External Data

```python
import pandas as pd

# CSV
df = pd.read_csv('data.csv')

# Separate features and target
X = df.drop('target', axis=1)
y = df['target']

# Kaggle data (example)
# !pip install kaggle
# !kaggle datasets download -d username/dataset-name
```

---

## 7. Machine Learning Project Template

```python
"""
Basic Machine Learning Project Template
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

# 1. Load data
# df = pd.read_csv('data.csv')
# X = df.drop('target', axis=1)
# y = df['target']

# 2. Exploratory Data Analysis (EDA)
# print(df.info())
# print(df.describe())
# print(df['target'].value_counts())

# 3. Data preprocessing
# - Handle missing values
# - Encoding
# - Scaling

# 4. Split data
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42, stratify=y
# )

# 5. Model selection and training
# from sklearn.ensemble import RandomForestClassifier
# model = RandomForestClassifier(random_state=42)
# model.fit(X_train, y_train)

# 6. Cross-validation
# cv_scores = cross_val_score(model, X_train, y_train, cv=5)
# print(f"CV score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# 7. Hyperparameter tuning
# from sklearn.model_selection import GridSearchCV
# param_grid = {'n_estimators': [50, 100, 200]}
# grid_search = GridSearchCV(model, param_grid, cv=5)
# grid_search.fit(X_train, y_train)

# 8. Final evaluation
# y_pred = model.predict(X_test)
# print(classification_report(y_test, y_pred))

# 9. Save model
# import joblib
# joblib.dump(model, 'model.pkl')
```

---

## Practice Problems

### Problem 1: Data Splitting
Split iris data into 80:20 while maintaining class proportions.

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris = load_iris()
X, y = iris.data, iris.target

# Solution
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training data: {len(X_train)}")
print(f"Test data: {len(X_test)}")
print(f"Test class distribution: {np.bincount(y_test)}")
```

### Problem 2: Basic Model Training
Train a logistic regression model and compute accuracy.

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Solution
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
```

---

## Summary

| Concept | Description |
|---------|-------------|
| Supervised Learning | Learn from labeled data (regression, classification) |
| Unsupervised Learning | Learn patterns without labels (clustering, dimensionality reduction) |
| Reinforcement Learning | Learn by interaction to maximize rewards |
| Overfitting | Too fitted to training data |
| Underfitting | Model too simple |
| Bias-Variance | Tradeoff between model complexity and generalization |
