# Model Evaluation

**Previous**: [Logistic Regression](./03_Logistic_Regression.md) | **Next**: [Cross-Validation and Hyperparameters](./05_Cross_Validation_Hyperparameters.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Interpret a confusion matrix and identify Type I and Type II errors
2. Calculate precision, recall, F1-score, and accuracy from a confusion matrix
3. Explain when accuracy alone is insufficient and which metric to prioritize instead
4. Compare ROC-AUC and PR-AUC curves and describe when each is more appropriate
5. Evaluate regression models using MAE, MSE, RMSE, R-squared, and MAPE
6. Apply learning curves and validation curves to diagnose underfitting and overfitting
7. Select the appropriate evaluation metric based on the problem context (balanced vs. imbalanced, classification vs. regression)

---

Training a model is only half the story -- you need to know how well it actually performs on unseen data. Choosing the wrong evaluation metric can lead you to deploy a model that looks great on paper but fails in production, especially when classes are imbalanced or when different types of errors carry different costs.

---

## 1. Classification Evaluation Metrics

### 1.1 Confusion Matrix

```python
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Example data
y_true = [1, 0, 1, 1, 0, 1, 0, 0, 1, 1]
y_pred = [1, 0, 1, 0, 0, 1, 1, 0, 1, 1]

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:")
print(cm)

# Visualization
fig, ax = plt.subplots(figsize=(6, 5))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Negative', 'Positive'])
disp.plot(ax=ax, cmap='Blues')
plt.title('Confusion Matrix')
plt.show()

# Confusion matrix elements
tn, fp, fn, tp = cm.ravel()
print(f"\nTN (True Negative): {tn}")
print(f"FP (False Positive): {fp} - Type I Error")
print(f"FN (False Negative): {fn} - Type II Error")
print(f"TP (True Positive): {tp}")
```

### 1.2 Accuracy

```python
from sklearn.metrics import accuracy_score

# Accuracy = (TP + TN) / (TP + TN + FP + FN)
accuracy = accuracy_score(y_true, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# Manual calculation
accuracy_manual = (tp + tn) / (tp + tn + fp + fn)
print(f"Accuracy (manual): {accuracy_manual:.4f}")

# Warning: accuracy alone is inadequate for imbalanced data
# Example: 99% negative → predicting all negative gives 99% accuracy
```

### 1.3 Precision, Recall, F1-score

```python
from sklearn.metrics import precision_score, recall_score, f1_score

# Precision = TP / (TP + FP)
# "Proportion of actual positives among predicted positives"
precision = precision_score(y_true, y_pred)
print(f"Precision: {precision:.4f}")

# Recall (Sensitivity) = TP / (TP + FN)
# "Proportion of predicted positives among actual positives"
recall = recall_score(y_true, y_pred)
print(f"Recall: {recall:.4f}")

# F1-Score = 2 * (Precision * Recall) / (Precision + Recall)
# Harmonic mean of precision and recall
f1 = f1_score(y_true, y_pred)
print(f"F1-Score: {f1:.4f}")

# Manual calculation
precision_manual = tp / (tp + fp)
recall_manual = tp / (tp + fn)
f1_manual = 2 * precision_manual * recall_manual / (precision_manual + recall_manual)
print(f"\nManual calculation:")
print(f"Precision: {precision_manual:.4f}")
print(f"Recall: {recall_manual:.4f}")
print(f"F1: {f1_manual:.4f}")
```

### 1.4 Classification Report

```python
from sklearn.metrics import classification_report

y_true = [0, 0, 0, 1, 1, 1, 2, 2, 2]
y_pred = [0, 0, 1, 1, 1, 2, 2, 2, 0]

report = classification_report(y_true, y_pred, target_names=['Class A', 'Class B', 'Class C'])
print("Classification Report:")
print(report)

# Return as dictionary
report_dict = classification_report(y_true, y_pred, output_dict=True)
print(f"\nClass B F1-score: {report_dict['Class B']['f1-score']:.4f}")
```

### 1.5 ROC Curve and AUC

```python
from sklearn.metrics import roc_curve, roc_auc_score, auc
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Prepare data
cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, test_size=0.2, random_state=42
)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Prediction probabilities
y_proba = model.predict_proba(X_test)[:, 1]

# ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)

# Visualization
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC Curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], 'r--', linewidth=1, label='Random Classifier')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.grid(True, alpha=0.3)
plt.show()

print(f"AUC Score: {roc_auc:.4f}")
print(f"AUC Score (sklearn): {roc_auc_score(y_test, y_proba):.4f}")
```

### 1.6 PR Curve (Precision-Recall)

```python
from sklearn.metrics import precision_recall_curve, average_precision_score

# PR curve
precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
ap = average_precision_score(y_test, y_proba)

# Visualization
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, 'b-', linewidth=2, label=f'PR Curve (AP = {ap:.4f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

print(f"Average Precision: {ap:.4f}")

# ROC vs PR
# - ROC: Stable on imbalanced data, but FPR may look low when positive class is rare
# - PR: More sensitive on imbalanced data, focuses on positive class prediction performance
```

---

## 2. Multi-class Classification Evaluation

### 2.1 Multi-class Metrics

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Accuracy
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

# F1-score (various averaging methods)
print(f"\nF1-Score (macro): {f1_score(y_test, y_pred, average='macro'):.4f}")
print(f"F1-Score (weighted): {f1_score(y_test, y_pred, average='weighted'):.4f}")
print(f"F1-Score (micro): {f1_score(y_test, y_pred, average='micro'):.4f}")

# macro: Simple average of F1 for each class
# weighted: Weighted average by number of samples per class
# micro: Calculate by summing all TP, FP, FN
```

### 2.2 Multi-class ROC

```python
from sklearn.preprocessing import label_binarize

# Binarize labels
y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
y_proba = model.predict_proba(X_test)

# ROC for each class
plt.figure(figsize=(10, 6))
colors = ['blue', 'red', 'green']

for i, (color, name) in enumerate(zip(colors, iris.target_names)):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color=color, linewidth=2,
             label=f'{name} (AUC = {roc_auc:.4f})')

plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Multi-class ROC Curves')
plt.legend(loc='lower right')
plt.grid(True, alpha=0.3)
plt.show()
```

---

## 3. Regression Evaluation Metrics

```python
import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error
)

# Example data
y_true = np.array([3, -0.5, 2, 7, 4.5])
y_pred = np.array([2.5, 0.0, 2, 8, 4.0])

# MAE (Mean Absolute Error)
mae = mean_absolute_error(y_true, y_pred)
print(f"MAE: {mae:.4f}")
# Interpretation: On average, predictions deviate from actual values by {mae}

# MSE (Mean Squared Error)
mse = mean_squared_error(y_true, y_pred)
print(f"MSE: {mse:.4f}")
# Feature: Larger penalties for larger errors

# RMSE (Root Mean Squared Error)
rmse = np.sqrt(mse)
print(f"RMSE: {rmse:.4f}")
# Interpretation: Same units as target, interpretable

# R² (Coefficient of Determination)
r2 = r2_score(y_true, y_pred)
print(f"R²: {r2:.4f}")
# Interpretation: 0~1, closer to 1 is better, what % of variance is explained by model

# MAPE (Mean Absolute Percentage Error)
mape = mean_absolute_percentage_error(y_true, y_pred)
print(f"MAPE: {mape:.4f}")
# Warning: Unstable when y_true is close to 0

# Manual calculation
print("\n=== Manual Calculation ===")
print(f"MAE: {np.mean(np.abs(y_true - y_pred)):.4f}")
print(f"MSE: {np.mean((y_true - y_pred)**2):.4f}")
print(f"R²: {1 - np.sum((y_true - y_pred)**2) / np.sum((y_true - np.mean(y_true))**2):.4f}")
```

### 3.1 R² Score Interpretation

```python
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression

diabetes = load_diabetes()
X_train, X_test, y_train, y_test = train_test_split(
    diabetes.data, diabetes.target, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
print(f"R² Score: {r2:.4f}")
print(f"Interpretation: The model explains {r2*100:.1f}% of target variance.")

# Visualization
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title(f'Actual vs Predicted (R² = {r2:.4f})')
plt.grid(True, alpha=0.3)
plt.show()
```

---

## 4. Evaluation Metric Selection Guide

```python
"""
Classification Problems:

1. Balanced data
   - Accuracy, F1-score

2. Imbalanced data
   - Precision, Recall, F1-score, PR-AUC
   - Positive class important: Emphasize Recall
   - False positive cost: Emphasize Precision

3. Probability prediction quality
   - ROC-AUC, PR-AUC, Log Loss

4. Multi-class
   - Macro F1: Equal importance per class
   - Weighted F1: Importance proportional to sample count
   - Micro F1: Similar to overall accuracy


Regression Problems:

1. Basic
   - MSE, RMSE, MAE

2. Outlier sensitivity
   - MAE (robust), MSE (sensitive)

3. Relative error
   - MAPE, R²

4. Model comparison
   - R² (normalized to 0~1 range)
"""

# Evaluation metric comparison functions
def evaluate_classification(y_true, y_pred, y_proba=None):
    """Comprehensive classification evaluation"""
    print("=== Classification Evaluation Results ===")
    print(f"Accuracy:  {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred, average='weighted'):.4f}")
    print(f"Recall:    {recall_score(y_true, y_pred, average='weighted'):.4f}")
    print(f"F1-Score:  {f1_score(y_true, y_pred, average='weighted'):.4f}")
    if y_proba is not None:
        print(f"ROC-AUC:   {roc_auc_score(y_true, y_proba):.4f}")

def evaluate_regression(y_true, y_pred):
    """Comprehensive regression evaluation"""
    print("=== Regression Evaluation Results ===")
    print(f"MAE:  {mean_absolute_error(y_true, y_pred):.4f}")
    print(f"MSE:  {mean_squared_error(y_true, y_pred):.4f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_true, y_pred)):.4f}")
    print(f"R²:   {r2_score(y_true, y_pred):.4f}")
```

---

## 5. Learning Curves and Validation Curves

### 5.1 Learning Curve

```python
from sklearn.model_selection import learning_curve

# Prepare data
cancer = load_breast_cancer()
X, y = cancer.data, cancer.target

# Calculate learning curve
train_sizes, train_scores, val_scores = learning_curve(
    LogisticRegression(max_iter=1000),
    X, y,
    train_sizes=np.linspace(0.1, 1.0, 10),
    cv=5,
    scoring='accuracy'
)

# Mean and standard deviation
train_mean = train_scores.mean(axis=1)
train_std = train_scores.std(axis=1)
val_mean = val_scores.mean(axis=1)
val_std = val_scores.std(axis=1)

# Visualization
plt.figure(figsize=(10, 6))
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color='orange')
plt.plot(train_sizes, train_mean, 'o-', color='blue', label='Training Score')
plt.plot(train_sizes, val_mean, 'o-', color='orange', label='Validation Score')
plt.xlabel('Training Set Size')
plt.ylabel('Accuracy')
plt.title('Learning Curve')
plt.legend(loc='best')
plt.grid(True, alpha=0.3)
plt.show()

# Interpretation:
# - Both curves low → Underfitting
# - Training curve high, validation curve low → Overfitting
# - Curves converge → Good fit
```

### 5.2 Validation Curve

```python
from sklearn.model_selection import validation_curve

# Hyperparameter range
param_range = np.logspace(-4, 2, 10)

# Calculate validation curve
train_scores, val_scores = validation_curve(
    LogisticRegression(max_iter=1000),
    X, y,
    param_name='C',
    param_range=param_range,
    cv=5,
    scoring='accuracy'
)

train_mean = train_scores.mean(axis=1)
train_std = train_scores.std(axis=1)
val_mean = val_scores.mean(axis=1)
val_std = val_scores.std(axis=1)

# Visualization
plt.figure(figsize=(10, 6))
plt.semilogx(param_range, train_mean, 'o-', color='blue', label='Training Score')
plt.semilogx(param_range, val_mean, 'o-', color='orange', label='Validation Score')
plt.fill_between(param_range, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
plt.fill_between(param_range, val_mean - val_std, val_mean + val_std, alpha=0.1, color='orange')
plt.xlabel('C (Regularization Parameter)')
plt.ylabel('Accuracy')
plt.title('Validation Curve')
plt.legend(loc='best')
plt.grid(True, alpha=0.3)
plt.show()
```

---

## Practice Problems

### Problem 1: Classification Evaluation
Calculate Precision, Recall, and F1-score from the confusion matrix.

```python
# TN=50, FP=10, FN=5, TP=35

# Solution
tn, fp, fn, tp = 50, 10, 5, 35
precision = tp / (tp + fp)
recall = tp / (tp + fn)
f1 = 2 * precision * recall / (precision + recall)

print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
```

### Problem 2: Regression Evaluation
Calculate R² from predictions and actual values.

```python
y_true = [100, 150, 200, 250, 300]
y_pred = [110, 140, 210, 240, 290]

# Solution
from sklearn.metrics import r2_score
print(f"R² Score: {r2_score(y_true, y_pred):.4f}")
```

---

## Summary

| Metric | Classification/Regression | Range | Description |
|--------|---------------------------|-------|-------------|
| Accuracy | Classification | 0-1 | Overall correct proportion |
| Precision | Classification | 0-1 | Actual positives among positive predictions |
| Recall | Classification | 0-1 | Positive predictions among actual positives |
| F1-Score | Classification | 0-1 | Harmonic mean of Precision/Recall |
| ROC-AUC | Classification | 0-1 | Overall classifier performance |
| MAE | Regression | 0-∞ | Mean absolute error |
| MSE | Regression | 0-∞ | Mean squared error |
| R² | Regression | -∞-1 | Proportion of explained variance |
