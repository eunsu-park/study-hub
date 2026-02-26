# 모델 평가 (Model Evaluation)

**이전**: [로지스틱 회귀](./03_Logistic_Regression.md) | **다음**: [교차검증과 하이퍼파라미터 튜닝](./05_Cross_Validation_Hyperparameters.md)

---

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 혼동 행렬(confusion matrix)을 해석하고 1종 오류(Type I Error)와 2종 오류(Type II Error)를 식별할 수 있습니다
2. 혼동 행렬로부터 정밀도(precision), 재현율(recall), F1-점수(F1-score), 정확도(accuracy)를 계산할 수 있습니다
3. 정확도만으로는 불충분한 경우와 대신 우선시해야 할 지표가 무엇인지 설명할 수 있습니다
4. ROC-AUC와 PR-AUC 곡선을 비교하고 각각이 더 적합한 상황을 설명할 수 있습니다
5. MAE, MSE, RMSE, R-제곱(R-squared), MAPE를 사용하여 회귀 모델을 평가할 수 있습니다
6. 학습 곡선(learning curve)과 검증 곡선(validation curve)을 적용하여 과소적합(underfitting)과 과대적합(overfitting)을 진단할 수 있습니다
7. 문제 맥락(균형/불균형, 분류/회귀)에 따라 적절한 평가 지표를 선택할 수 있습니다

---

모델을 학습시키는 것은 절반에 불과합니다 — 모델이 본 적 없는 데이터에서 얼마나 잘 동작하는지 알아야 합니다. 잘못된 평가 지표를 선택하면, 특히 클래스가 불균형하거나 서로 다른 유형의 오류가 다른 비용을 갖는 경우에, 서류상으로는 훌륭해 보이지만 실제 운영에서는 실패하는 모델을 배포하게 될 수 있습니다.

---

## 1. 분류 평가 지표

### 1.1 혼동 행렬 (Confusion Matrix)

```python
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# 예시 데이터
y_true = [1, 0, 1, 1, 0, 1, 0, 0, 1, 1]
y_pred = [1, 0, 1, 0, 0, 1, 1, 0, 1, 1]

# 혼동 행렬
cm = confusion_matrix(y_true, y_pred)
print("혼동 행렬:")
print(cm)

# 시각화
fig, ax = plt.subplots(figsize=(6, 5))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Negative', 'Positive'])
disp.plot(ax=ax, cmap='Blues')
plt.title('Confusion Matrix')
plt.show()

# 혼동 행렬 요소
tn, fp, fn, tp = cm.ravel()
print(f"\nTN (True Negative): {tn}")
print(f"FP (False Positive): {fp} - Type I Error")
print(f"FN (False Negative): {fn} - Type II Error")
print(f"TP (True Positive): {tp}")
```

### 1.2 정확도 (Accuracy)

```python
from sklearn.metrics import accuracy_score

# Accuracy = (TP + TN) / (TP + TN + FP + FN)
accuracy = accuracy_score(y_true, y_pred)
print(f"정확도: {accuracy:.4f}")

# 수동 계산
accuracy_manual = (tp + tn) / (tp + tn + fp + fn)
print(f"정확도 (수동): {accuracy_manual:.4f}")

# 주의: 불균형 데이터에서는 정확도만으로 평가 부적절
# 예: 99% negative → 모두 negative 예측해도 99% 정확도
```

### 1.3 정밀도, 재현율, F1-score

```python
from sklearn.metrics import precision_score, recall_score, f1_score

# Precision = TP / (TP + FP)
# "양성으로 예측한 것 중 실제 양성의 비율"
precision = precision_score(y_true, y_pred)
print(f"정밀도 (Precision): {precision:.4f}")

# Recall (Sensitivity) = TP / (TP + FN)
# "실제 양성 중 양성으로 예측한 비율"
recall = recall_score(y_true, y_pred)
print(f"재현율 (Recall): {recall:.4f}")

# F1-Score = 2 * (Precision * Recall) / (Precision + Recall)
# 정밀도와 재현율의 조화 평균
f1 = f1_score(y_true, y_pred)
print(f"F1-Score: {f1:.4f}")

# 수동 계산
precision_manual = tp / (tp + fp)
recall_manual = tp / (tp + fn)
f1_manual = 2 * precision_manual * recall_manual / (precision_manual + recall_manual)
print(f"\n수동 계산:")
print(f"Precision: {precision_manual:.4f}")
print(f"Recall: {recall_manual:.4f}")
print(f"F1: {f1_manual:.4f}")
```

### 1.4 분류 리포트

```python
from sklearn.metrics import classification_report

y_true = [0, 0, 0, 1, 1, 1, 2, 2, 2]
y_pred = [0, 0, 1, 1, 1, 2, 2, 2, 0]

report = classification_report(y_true, y_pred, target_names=['Class A', 'Class B', 'Class C'])
print("분류 리포트:")
print(report)

# 딕셔너리로 반환
report_dict = classification_report(y_true, y_pred, output_dict=True)
print(f"\nClass B의 F1-score: {report_dict['Class B']['f1-score']:.4f}")
```

### 1.5 ROC 곡선과 AUC

```python
from sklearn.metrics import roc_curve, roc_auc_score, auc
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# 데이터 준비
cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, test_size=0.2, random_state=42
)

# 모델 학습
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 예측 확률
y_proba = model.predict_proba(X_test)[:, 1]

# ROC 곡선
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)

# 시각화
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

### 1.6 PR 곡선 (Precision-Recall)

```python
from sklearn.metrics import precision_recall_curve, average_precision_score

# PR 곡선
precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
ap = average_precision_score(y_test, y_proba)

# 시각화
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
# - ROC: 불균형 데이터에서도 안정적이지만, 긍정 클래스가 적으면 FPR이 낮아 보일 수 있음
# - PR: 불균형 데이터에서 더 민감, 긍정 클래스 예측 성능에 집중
```

---

## 2. 다중 분류 평가

### 2.1 다중 분류 지표

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

# 정확도
print(f"정확도: {accuracy_score(y_test, y_pred):.4f}")

# F1-score (다양한 평균 방법)
print(f"\nF1-Score (macro): {f1_score(y_test, y_pred, average='macro'):.4f}")
print(f"F1-Score (weighted): {f1_score(y_test, y_pred, average='weighted'):.4f}")
print(f"F1-Score (micro): {f1_score(y_test, y_pred, average='micro'):.4f}")

# macro: 각 클래스의 F1을 단순 평균
# weighted: 각 클래스의 샘플 수로 가중 평균
# micro: 전체 TP, FP, FN을 합산하여 계산
```

### 2.2 다중 클래스 ROC

```python
from sklearn.preprocessing import label_binarize

# 레이블 이진화
y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
y_proba = model.predict_proba(X_test)

# 각 클래스별 ROC
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

## 3. 회귀 평가 지표

```python
import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error
)

# 예시 데이터
y_true = np.array([3, -0.5, 2, 7, 4.5])
y_pred = np.array([2.5, 0.0, 2, 8, 4.0])

# MAE (Mean Absolute Error)
mae = mean_absolute_error(y_true, y_pred)
print(f"MAE: {mae:.4f}")
# 해석: 평균적으로 예측이 실제값에서 {mae} 만큼 벗어남

# MSE (Mean Squared Error)
mse = mean_squared_error(y_true, y_pred)
print(f"MSE: {mse:.4f}")
# 특징: 큰 오차에 더 큰 패널티

# RMSE (Root Mean Squared Error)
rmse = np.sqrt(mse)
print(f"RMSE: {rmse:.4f}")
# 해석: 타겟과 같은 단위로 해석 가능

# R² (결정계수)
r2 = r2_score(y_true, y_pred)
print(f"R²: {r2:.4f}")
# 해석: 0~1, 1에 가까울수록 좋음, 모델이 분산의 몇 %를 설명하는지

# MAPE (Mean Absolute Percentage Error)
mape = mean_absolute_percentage_error(y_true, y_pred)
print(f"MAPE: {mape:.4f}")
# 주의: y_true가 0에 가까우면 불안정

# 수동 계산
print("\n=== 수동 계산 ===")
print(f"MAE: {np.mean(np.abs(y_true - y_pred)):.4f}")
print(f"MSE: {np.mean((y_true - y_pred)**2):.4f}")
print(f"R²: {1 - np.sum((y_true - y_pred)**2) / np.sum((y_true - np.mean(y_true))**2):.4f}")
```

### 3.1 R² Score 해석

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
print(f"해석: 모델이 타겟 분산의 {r2*100:.1f}%를 설명합니다.")

# 시각화
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2)
plt.xlabel('실제값')
plt.ylabel('예측값')
plt.title(f'실제값 vs 예측값 (R² = {r2:.4f})')
plt.grid(True, alpha=0.3)
plt.show()
```

---

## 4. 평가 지표 선택 가이드

```python
"""
분류 문제:

1. 균형 데이터
   - Accuracy, F1-score

2. 불균형 데이터
   - Precision, Recall, F1-score, PR-AUC
   - 양성 클래스가 중요: Recall 중시
   - 오탐이 비용: Precision 중시

3. 확률 예측 품질
   - ROC-AUC, PR-AUC, Log Loss

4. 다중 분류
   - Macro F1: 클래스 균등 중요
   - Weighted F1: 샘플 수 비례 중요
   - Micro F1: 전체 정확도와 유사


회귀 문제:

1. 기본
   - MSE, RMSE, MAE

2. 이상치 민감
   - MAE (robust), MSE (sensitive)

3. 상대적 오차
   - MAPE, R²

4. 모델 비교
   - R² (0~1 범위로 정규화됨)
"""

# 평가 지표 비교 함수
def evaluate_classification(y_true, y_pred, y_proba=None):
    """분류 모델 종합 평가"""
    print("=== 분류 평가 결과 ===")
    print(f"Accuracy:  {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred, average='weighted'):.4f}")
    print(f"Recall:    {recall_score(y_true, y_pred, average='weighted'):.4f}")
    print(f"F1-Score:  {f1_score(y_true, y_pred, average='weighted'):.4f}")
    if y_proba is not None:
        print(f"ROC-AUC:   {roc_auc_score(y_true, y_proba):.4f}")

def evaluate_regression(y_true, y_pred):
    """회귀 모델 종합 평가"""
    print("=== 회귀 평가 결과 ===")
    print(f"MAE:  {mean_absolute_error(y_true, y_pred):.4f}")
    print(f"MSE:  {mean_squared_error(y_true, y_pred):.4f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_true, y_pred)):.4f}")
    print(f"R²:   {r2_score(y_true, y_pred):.4f}")
```

---

## 5. 학습 곡선과 검증 곡선

### 5.1 학습 곡선 (Learning Curve)

```python
from sklearn.model_selection import learning_curve

# 데이터 준비
cancer = load_breast_cancer()
X, y = cancer.data, cancer.target

# 학습 곡선 계산
train_sizes, train_scores, val_scores = learning_curve(
    LogisticRegression(max_iter=1000),
    X, y,
    train_sizes=np.linspace(0.1, 1.0, 10),
    cv=5,
    scoring='accuracy'
)

# 평균 및 표준편차
train_mean = train_scores.mean(axis=1)
train_std = train_scores.std(axis=1)
val_mean = val_scores.mean(axis=1)
val_std = val_scores.std(axis=1)

# 시각화
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

# 해석:
# - 두 곡선이 모두 낮음 → 과소적합
# - 훈련 곡선 높고 검증 곡선 낮음 → 과적합
# - 두 곡선이 수렴 → 적절한 적합
```

### 5.2 검증 곡선 (Validation Curve)

```python
from sklearn.model_selection import validation_curve

# 하이퍼파라미터 범위
param_range = np.logspace(-4, 2, 10)

# 검증 곡선 계산
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

# 시각화
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

## 연습 문제

### 문제 1: 분류 평가
혼동 행렬에서 Precision, Recall, F1-score를 계산하세요.

```python
# TN=50, FP=10, FN=5, TP=35

# 풀이
tn, fp, fn, tp = 50, 10, 5, 35
precision = tp / (tp + fp)
recall = tp / (tp + fn)
f1 = 2 * precision * recall / (precision + recall)

print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
```

### 문제 2: 회귀 평가
예측값과 실제값으로 R²를 계산하세요.

```python
y_true = [100, 150, 200, 250, 300]
y_pred = [110, 140, 210, 240, 290]

# 풀이
from sklearn.metrics import r2_score
print(f"R² Score: {r2_score(y_true, y_pred):.4f}")
```

---

## 요약

| 지표 | 분류/회귀 | 범위 | 설명 |
|------|----------|------|------|
| Accuracy | 분류 | 0-1 | 전체 정답 비율 |
| Precision | 분류 | 0-1 | 양성 예측 중 실제 양성 |
| Recall | 분류 | 0-1 | 실제 양성 중 양성 예측 |
| F1-Score | 분류 | 0-1 | Precision/Recall 조화평균 |
| ROC-AUC | 분류 | 0-1 | 분류기 전반적 성능 |
| MAE | 회귀 | 0-∞ | 평균 절대 오차 |
| MSE | 회귀 | 0-∞ | 평균 제곱 오차 |
| R² | 회귀 | -∞-1 | 설명 분산 비율 |
