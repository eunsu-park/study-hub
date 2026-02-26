# Real-World Projects

**Previous**: [Pipelines and Practice](./13_Pipelines_and_Practice.md) | **Next**: [Feature Engineering](./15_Feature_Engineering.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Describe the end-to-end machine learning project workflow from problem definition to deployment
2. Implement a complete classification project (Titanic survival prediction) including EDA, feature engineering, model comparison, and hyperparameter tuning
3. Implement a complete regression project (housing price prediction) with residual analysis and feature importance
4. Apply cross-validation strategies including K-Fold and OOF (out-of-fold) prediction generation
5. Build ensemble predictions using blending and stacking techniques
6. Evaluate models with appropriate metrics for classification (F1, AUC) and regression (RMSE, R2, MAE)
7. Identify and analyze prediction errors to guide iterative model improvement

---

Algorithms and theory only matter if you can apply them to real problems end-to-end. This lesson puts everything together -- EDA, preprocessing, model selection, tuning, evaluation, and ensemble -- through two complete projects that mirror the workflow you would follow in a Kaggle competition or a real business setting. The patterns you practice here form the foundation of every applied ML project.

---

## 1. Machine Learning Project Workflow

### 1.1 Overall Process

```python
"""
Machine Learning Project Stages:

1. Problem Definition
   - Understand business objectives
   - Define success metrics
   - Decide: classification/regression/clustering

2. Data Collection and Exploration
   - Load data
   - EDA (Exploratory Data Analysis)
   - Check data quality

3. Data Preprocessing
   - Handle missing values
   - Handle outliers
   - Feature engineering
   - Encoding and scaling

4. Modeling
   - Baseline model
   - Model selection and comparison
   - Hyperparameter tuning

5. Evaluation and Interpretation
   - Performance evaluation
   - Error analysis
   - Feature importance

6. Deployment and Monitoring
   - Model saving
   - Prediction API
   - Performance monitoring
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    mean_squared_error, r2_score, mean_absolute_error
)
import warnings
warnings.filterwarnings('ignore')
```

---

## 2. Project 1: Titanic Survival Prediction (Classification)

### 2.1 Data Loading and Exploration

```python
# Load data (actual Kaggle data or seaborn built-in data)
# df = pd.read_csv('titanic.csv')
df = sns.load_dataset('titanic')

print("=== Basic Data Information ===")
print(f"Data shape: {df.shape}")
print(f"\nColumn info:")
print(df.info())

print(f"\nFirst 5 rows:")
print(df.head())

print(f"\nDescriptive statistics:")
print(df.describe())

print(f"\nTarget distribution:")
print(df['survived'].value_counts(normalize=True))
```

### 2.2 Exploratory Data Analysis (EDA)

```python
# Check missing values
print("=== Missing Value Analysis ===")
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
missing_df = pd.DataFrame({'Missing Count': missing, 'Missing Percentage (%)': missing_pct})
print(missing_df[missing_df['Missing Count'] > 0])

# Visualization
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Survival rate
sns.countplot(data=df, x='survived', ax=axes[0, 0])
axes[0, 0].set_title('Survival Distribution')

# Survival by gender
sns.countplot(data=df, x='sex', hue='survived', ax=axes[0, 1])
axes[0, 1].set_title('Survival by Sex')

# Survival by class
sns.countplot(data=df, x='pclass', hue='survived', ax=axes[0, 2])
axes[0, 2].set_title('Survival by Class')

# Age distribution
sns.histplot(data=df, x='age', hue='survived', kde=True, ax=axes[1, 0])
axes[1, 0].set_title('Age Distribution by Survival')

# Fare distribution
sns.histplot(data=df, x='fare', hue='survived', kde=True, ax=axes[1, 1])
axes[1, 1].set_title('Fare Distribution by Survival')

# Survival by port of embarkation
sns.countplot(data=df, x='embarked', hue='survived', ax=axes[1, 2])
axes[1, 2].set_title('Survival by Embarked')

plt.tight_layout()
plt.show()

# Correlation
print("\n=== Correlation with Numeric Variables ===")
numeric_cols = df.select_dtypes(include=[np.number]).columns
print(df[numeric_cols].corr()['survived'].sort_values(ascending=False))
```

### 2.3 Data Preprocessing

```python
# Working copy
df_clean = df.copy()

# Remove unnecessary columns
drop_cols = ['deck', 'embark_town', 'alive', 'who', 'adult_male', 'class']
df_clean = df_clean.drop(columns=drop_cols, errors='ignore')

# Handle missing values
# Age: Replace with median
df_clean['age'] = df_clean['age'].fillna(df_clean['age'].median())

# Port of embarkation: Replace with mode
df_clean['embarked'] = df_clean['embarked'].fillna(df_clean['embarked'].mode()[0])

# Feature engineering
# Family size
df_clean['family_size'] = df_clean['sibsp'] + df_clean['parch'] + 1

# Traveling alone
df_clean['is_alone'] = (df_clean['family_size'] == 1).astype(int)

# Age groups
df_clean['age_group'] = pd.cut(df_clean['age'],
                                bins=[0, 12, 18, 35, 60, 100],
                                labels=['Child', 'Teen', 'Young', 'Middle', 'Senior'])

# Categorical encoding
df_clean['sex'] = LabelEncoder().fit_transform(df_clean['sex'])
df_clean['embarked'] = LabelEncoder().fit_transform(df_clean['embarked'])
df_clean['age_group'] = LabelEncoder().fit_transform(df_clean['age_group'])

# Final feature selection
features = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare',
            'embarked', 'family_size', 'is_alone', 'age_group']
X = df_clean[features]
y = df_clean['survived']

print(f"Final features: {features}")
print(f"X shape: {X.shape}")
```

### 2.4 Modeling

```python
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# Data split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model definitions
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'XGBoost': XGBClassifier(eval_metric='logloss', random_state=42),
    'LightGBM': LGBMClassifier(random_state=42, verbose=-1)
}

# Model comparison
print("=== Model Comparison ===")
results = []

for name, model in models.items():
    # Use scaled data for SVM and Logistic Regression
    if name in ['Logistic Regression']:
        X_tr, X_te = X_train_scaled, X_test_scaled
    else:
        X_tr, X_te = X_train, X_test

    # Cross-validation
    cv_scores = cross_val_score(model, X_tr, y_train, cv=5, scoring='accuracy')

    # Train and test
    model.fit(X_tr, y_train)
    test_score = model.score(X_te, y_test)

    results.append({
        'Model': name,
        'CV Mean': cv_scores.mean(),
        'CV Std': cv_scores.std(),
        'Test Score': test_score
    })

    print(f"{name}: CV={cv_scores.mean():.4f}(+/-{cv_scores.std():.4f}), Test={test_score:.4f}")

results_df = pd.DataFrame(results)
print(f"\nBest CV score: {results_df.loc[results_df['CV Mean'].idxmax(), 'Model']}")
```

### 2.5 Hyperparameter Tuning

```python
# Tune best performing model (e.g., Random Forest)
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)

print("\n=== Hyperparameter Tuning Results ===")
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best CV score: {grid_search.best_score_:.4f}")
print(f"Test score: {grid_search.score(X_test, y_test):.4f}")

best_model = grid_search.best_estimator_
```

### 2.6 Final Evaluation

```python
# Predictions
y_pred = best_model.predict(X_test)

# Confusion matrix
print("=== Final Evaluation ===")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Not Survived', 'Survived']))

# Visualize confusion matrix
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Not Survived', 'Survived'],
            yticklabels=['Not Survived', 'Survived'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Feature importance
importances = best_model.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
plt.bar(range(len(importances)), importances[indices])
plt.xticks(range(len(importances)), [features[i] for i in indices], rotation=45)
plt.title('Feature Importance')
plt.tight_layout()
plt.show()

print("\nFeature importance:")
for i in indices:
    print(f"  {features[i]}: {importances[i]:.4f}")
```

---

## 3. Project 2: Housing Price Prediction (Regression)

### 3.1 Data Loading and Exploration

```python
from sklearn.datasets import fetch_california_housing

# California housing price data
housing = fetch_california_housing()
df_house = pd.DataFrame(housing.data, columns=housing.feature_names)
df_house['MedHouseVal'] = housing.target

print("=== Housing Price Data ===")
print(f"Data shape: {df_house.shape}")
print(f"\nColumns: {list(df_house.columns)}")
print(f"\nDescriptive statistics:")
print(df_house.describe())

# Target distribution
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.hist(df_house['MedHouseVal'], bins=50, edgecolor='black')
plt.xlabel('Median House Value')
plt.ylabel('Count')
plt.title('Target Distribution')

plt.subplot(1, 2, 2)
plt.hist(np.log1p(df_house['MedHouseVal']), bins=50, edgecolor='black')
plt.xlabel('Log(Median House Value)')
plt.ylabel('Count')
plt.title('Log-Transformed Target')

plt.tight_layout()
plt.show()
```

### 3.2 Exploratory Data Analysis

```python
# Correlation heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(df_house.corr(), annot=True, fmt='.2f', cmap='coolwarm', center=0)
plt.title('Correlation Matrix')
plt.tight_layout()
plt.show()

# Correlation with target
print("\nCorrelation with target:")
correlations = df_house.corr()['MedHouseVal'].drop('MedHouseVal').sort_values(ascending=False)
print(correlations)

# Main features vs target relationship
fig, axes = plt.subplots(2, 4, figsize=(16, 8))

for ax, col in zip(axes.flatten(), housing.feature_names):
    ax.scatter(df_house[col], df_house['MedHouseVal'], alpha=0.1)
    ax.set_xlabel(col)
    ax.set_ylabel('MedHouseVal')
    ax.set_title(f'Corr: {df_house[col].corr(df_house["MedHouseVal"]):.3f}')

plt.tight_layout()
plt.show()
```

### 3.3 Data Preprocessing

```python
# Separate features and target
X = df_house.drop('MedHouseVal', axis=1)
y = df_house['MedHouseVal']

# Feature engineering
X_eng = X.copy()

# Rooms per person
X_eng['RoomsPerPerson'] = X_eng['AveRooms'] / X_eng['AveOccup']

# Bedroom ratio
X_eng['BedroomRatio'] = X_eng['AveBedrms'] / X_eng['AveRooms']

# Population density (approximate)
X_eng['PopDensity'] = X_eng['Population'] / X_eng['AveOccup']

# Handle inf/NaN
X_eng = X_eng.replace([np.inf, -np.inf], np.nan)
X_eng = X_eng.fillna(X_eng.median())

# Data split
X_train, X_test, y_train, y_test = train_test_split(
    X_eng, y, test_size=0.2, random_state=42
)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Training data: {X_train.shape}")
print(f"Test data: {X_test.shape}")
```

### 3.4 Modeling

```python
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# Model definitions
reg_models = {
    'Linear Regression': LinearRegression(),
    'Ridge': Ridge(alpha=1.0),
    'Lasso': Lasso(alpha=0.1),
    'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5),
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingRegressor(random_state=42),
    'XGBoost': XGBRegressor(random_state=42),
    'LightGBM': LGBMRegressor(random_state=42, verbose=-1)
}

# Model comparison
print("=== Regression Model Comparison ===")
reg_results = []

for name, model in reg_models.items():
    # Use scaled data for linear models
    if name in ['Linear Regression', 'Ridge', 'Lasso', 'ElasticNet']:
        X_tr, X_te = X_train_scaled, X_test_scaled
    else:
        X_tr, X_te = X_train, X_test

    # Cross-validation
    cv_scores = cross_val_score(model, X_tr, y_train, cv=5, scoring='r2')

    # Train and predict
    model.fit(X_tr, y_train)
    y_pred = model.predict(X_te)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    reg_results.append({
        'Model': name,
        'CV R2 Mean': cv_scores.mean(),
        'CV R2 Std': cv_scores.std(),
        'Test RMSE': rmse,
        'Test R2': r2
    })

    print(f"{name}: CV R2={cv_scores.mean():.4f}(+/-{cv_scores.std():.4f}), RMSE={rmse:.4f}, R2={r2:.4f}")

reg_results_df = pd.DataFrame(reg_results)
print(f"\nBest test R2: {reg_results_df.loc[reg_results_df['Test R2'].idxmax(), 'Model']}")
```

### 3.5 Hyperparameter Tuning

```python
# Tune LightGBM
lgbm_params = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 10, 15, -1],
    'learning_rate': [0.01, 0.05, 0.1],
    'num_leaves': [31, 63, 127]
}

lgbm = LGBMRegressor(random_state=42, verbose=-1)
grid_lgbm = GridSearchCV(lgbm, lgbm_params, cv=5, scoring='r2', n_jobs=-1, verbose=1)
grid_lgbm.fit(X_train, y_train)

print("\n=== LightGBM Tuning Results ===")
print(f"Best parameters: {grid_lgbm.best_params_}")
print(f"Best CV R2: {grid_lgbm.best_score_:.4f}")

y_pred_best = grid_lgbm.predict(X_test)
print(f"Test RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_best)):.4f}")
print(f"Test R2: {r2_score(y_test, y_pred_best):.4f}")
```

### 3.6 Final Evaluation

```python
# Final model
best_reg = grid_lgbm.best_estimator_
y_pred_final = best_reg.predict(X_test)

# Evaluation metrics
print("=== Final Regression Evaluation ===")
print(f"MAE: {mean_absolute_error(y_test, y_pred_final):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_final)):.4f}")
print(f"R2: {r2_score(y_test, y_pred_final):.4f}")

# Actual vs Predicted
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Scatter plot
axes[0].scatter(y_test, y_pred_final, alpha=0.3)
axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
axes[0].set_xlabel('Actual')
axes[0].set_ylabel('Predicted')
axes[0].set_title('Actual vs Predicted')

# Residual distribution
residuals = y_test - y_pred_final
axes[1].hist(residuals, bins=50, edgecolor='black')
axes[1].set_xlabel('Residual')
axes[1].set_ylabel('Count')
axes[1].set_title(f'Residual Distribution\nMean: {residuals.mean():.4f}')

# Residuals vs predicted
axes[2].scatter(y_pred_final, residuals, alpha=0.3)
axes[2].axhline(y=0, color='r', linestyle='--')
axes[2].set_xlabel('Predicted')
axes[2].set_ylabel('Residual')
axes[2].set_title('Residuals vs Predicted')

plt.tight_layout()
plt.show()

# Feature importance
importances = best_reg.feature_importances_
feature_names = X_eng.columns
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(12, 6))
plt.bar(range(len(importances)), importances[indices])
plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45)
plt.title('Feature Importance (LightGBM)')
plt.tight_layout()
plt.show()
```

---

## 4. Kaggle Competition Strategy

### 4.1 Basic Strategy

```python
"""
Kaggle competition strategy:

1. Quick Start
   - Run provided baseline code
   - Make first submission with simple model
   - Check leaderboard position

2. Focus on EDA
   - Data understanding is key
   - Identify missing values, outliers, distributions
   - Analyze relationship with target

3. Feature Engineering
   - Leverage domain knowledge
   - Create interaction features
   - Group statistics

4. Try Various Models
   - Linear models → Tree-based → Ensemble
   - Hyperparameter tuning

5. Ensemble
   - Combine predictions from different models
   - Blending, stacking

6. Validation Strategy
   - Ensure local CV matches leaderboard scores
   - Prevent overfitting
"""
```

### 4.2 Cross-Validation Strategy

```python
from sklearn.model_selection import KFold, StratifiedKFold

def cross_validate_model(model, X, y, n_splits=5, stratified=False, return_preds=False):
    """
    Cross-validation and OOF prediction generation
    """
    if stratified:
        kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        split_func = kf.split(X, y)
    else:
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        split_func = kf.split(X)

    scores = []
    oof_preds = np.zeros(len(X))

    for fold, (train_idx, val_idx) in enumerate(split_func):
        X_train_fold = X.iloc[train_idx] if hasattr(X, 'iloc') else X[train_idx]
        X_val_fold = X.iloc[val_idx] if hasattr(X, 'iloc') else X[val_idx]
        y_train_fold = y.iloc[train_idx] if hasattr(y, 'iloc') else y[train_idx]
        y_val_fold = y.iloc[val_idx] if hasattr(y, 'iloc') else y[val_idx]

        model.fit(X_train_fold, y_train_fold)
        val_pred = model.predict(X_val_fold)

        oof_preds[val_idx] = val_pred
        score = r2_score(y_val_fold, val_pred)
        scores.append(score)

        print(f"Fold {fold+1}: {score:.4f}")

    print(f"Mean: {np.mean(scores):.4f} (+/- {np.std(scores):.4f})")

    if return_preds:
        return oof_preds
    return np.mean(scores)

# Usage example
# oof_preds = cross_validate_model(model, X_train, y_train, n_splits=5, return_preds=True)
```

### 4.3 Ensemble Techniques

```python
def simple_blend(predictions_list, weights=None):
    """Simple blending"""
    if weights is None:
        weights = [1/len(predictions_list)] * len(predictions_list)

    blended = np.zeros(len(predictions_list[0]))
    for pred, weight in zip(predictions_list, weights):
        blended += weight * pred

    return blended


def stacking_ensemble(models, X_train, y_train, X_test, n_folds=5):
    """Stacking ensemble"""
    n_models = len(models)
    n_train = len(X_train)
    n_test = len(X_test)

    # OOF predictions
    oof_train = np.zeros((n_train, n_models))
    oof_test = np.zeros((n_test, n_models))

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    for i, model in enumerate(models):
        print(f"Training model {i+1}/{n_models}")
        oof_test_fold = np.zeros((n_test, n_folds))

        for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
            X_tr = X_train.iloc[train_idx] if hasattr(X_train, 'iloc') else X_train[train_idx]
            X_val = X_train.iloc[val_idx] if hasattr(X_train, 'iloc') else X_train[val_idx]
            y_tr = y_train.iloc[train_idx] if hasattr(y_train, 'iloc') else y_train[train_idx]

            model.fit(X_tr, y_tr)

            oof_train[val_idx, i] = model.predict(X_val)
            oof_test_fold[:, fold] = model.predict(X_test)

        oof_test[:, i] = oof_test_fold.mean(axis=1)

    return oof_train, oof_test

# Usage example
# models = [RandomForestRegressor(), XGBRegressor(), LGBMRegressor()]
# oof_train, oof_test = stacking_ensemble(models, X_train, y_train, X_test)
# meta_model = Ridge()
# meta_model.fit(oof_train, y_train)
# final_preds = meta_model.predict(oof_test)
```

---

## 5. Practical Tips Collection

### 5.1 Quick Experiment Template

```python
def quick_experiment(X_train, y_train, X_test, y_test, task='classification'):
    """Quick model comparison"""
    if task == 'classification':
        models = {
            'LR': LogisticRegression(max_iter=1000),
            'RF': RandomForestClassifier(n_estimators=100, random_state=42),
            'XGB': XGBClassifier(eval_metric='logloss', random_state=42),
            'LGBM': LGBMClassifier(random_state=42, verbose=-1)
        }
        scoring = 'accuracy'
    else:
        models = {
            'Ridge': Ridge(),
            'RF': RandomForestRegressor(n_estimators=100, random_state=42),
            'XGB': XGBRegressor(random_state=42),
            'LGBM': LGBMRegressor(random_state=42, verbose=-1)
        }
        scoring = 'r2'

    results = {}
    for name, model in models.items():
        cv_score = cross_val_score(model, X_train, y_train, cv=5, scoring=scoring).mean()
        model.fit(X_train, y_train)
        test_score = model.score(X_test, y_test)
        results[name] = {'CV': cv_score, 'Test': test_score}
        print(f"{name}: CV={cv_score:.4f}, Test={test_score:.4f}")

    return results
```

### 5.2 Memory Optimization

```python
def reduce_memory_usage(df):
    """Reduce DataFrame memory usage"""
    start_mem = df.memory_usage().sum() / 1024**2

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()

            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)

            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)

    end_mem = df.memory_usage().sum() / 1024**2
    print(f'Memory: {start_mem:.2f} MB → {end_mem:.2f} MB ({100*(start_mem-end_mem)/start_mem:.1f}% reduction)')

    return df
```

### 5.3 Error Analysis

```python
def analyze_errors(y_true, y_pred, X, feature_names, top_n=10):
    """Error analysis"""
    errors = np.abs(y_true - y_pred)

    # Largest errors
    top_errors_idx = np.argsort(errors)[-top_n:]

    print(f"=== Top {top_n} Error Analysis ===")
    for idx in top_errors_idx[::-1]:
        print(f"\nIndex {idx}:")
        print(f"  Actual: {y_true.iloc[idx] if hasattr(y_true, 'iloc') else y_true[idx]:.4f}")
        print(f"  Predicted: {y_pred[idx]:.4f}")
        print(f"  Error: {errors[idx]:.4f}")

    # Error correlation with features
    X_arr = X.values if hasattr(X, 'values') else X
    error_corr = []
    for i, name in enumerate(feature_names):
        corr = np.corrcoef(errors, X_arr[:, i])[0, 1]
        error_corr.append((name, corr))

    error_corr.sort(key=lambda x: abs(x[1]), reverse=True)

    print("\n=== Error Correlation with Features ===")
    for name, corr in error_corr[:5]:
        print(f"  {name}: {corr:.4f}")

# Usage example
# analyze_errors(y_test, y_pred, X_test, feature_names)
```

---

## Exercises

### Exercise 1: Complete Pipeline
Build a complete ML pipeline with Iris data.

```python
from sklearn.datasets import load_iris
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split

# Solution
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42
)

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5)
print(f"CV score: {cv_scores.mean():.4f}")

pipeline.fit(X_train, y_train)
print(f"Test score: {pipeline.score(X_test, y_test):.4f}")
```

### Exercise 2: Feature Engineering
Add new features to given data.

```python
import pandas as pd
import numpy as np

df = pd.DataFrame({
    'A': [1, 2, 3, 4, 5],
    'B': [10, 20, 30, 40, 50],
    'C': [100, 200, 300, 400, 500]
})

# Solution: Add ratio features, sum features
df['A_B_ratio'] = df['A'] / df['B']
df['AB_sum'] = df['A'] + df['B']
df['log_C'] = np.log1p(df['C'])
df['A_squared'] = df['A'] ** 2

print(df)
```

### Exercise 3: Ensemble
Blend multiple models.

```python
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Solution
models = [
    LogisticRegression(max_iter=1000),
    RandomForestClassifier(n_estimators=100, random_state=42)
]

predictions = []
for model in models:
    model.fit(X_train, y_train)
    pred = model.predict_proba(X_test)[:, 1]
    predictions.append(pred)

# Average blending
blended = np.mean(predictions, axis=0)
blended_labels = (blended > 0.5).astype(int)

print(f"Blending accuracy: {accuracy_score(y_test, blended_labels):.4f}")
```

---

## Summary

### Classification vs Regression Checklist

| Stage | Classification | Regression |
|-------|----------------|------------|
| Evaluation Metrics | Accuracy, F1, AUC | RMSE, MAE, R2 |
| Target Handling | Encoding | Check outliers, log transform |
| Imbalance | SMOTE, class weights | N/A |
| Error Analysis | Confusion matrix | Residual analysis |

### Model Selection Guide

| Situation | Recommended Model |
|-----------|-------------------|
| Quick baseline | Logistic/Linear regression |
| General performance | Random Forest |
| Best performance | XGBoost, LightGBM |
| Interpretation needed | Decision tree, linear model |
| Large dataset | LightGBM |

### Kaggle Essential Tips

1. Always compare local CV and leaderboard scores
2. Beware of overfitting - don't tune to public leaderboard
3. Ensemble with diverse models
4. Feature engineering is key
5. Reference kernels/notebooks but understand and apply
