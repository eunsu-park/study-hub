# Machine Learning (머신러닝) 학습 가이드

## 개요

머신러닝은 데이터로부터 패턴을 학습하여 예측이나 결정을 수행하는 알고리즘의 집합입니다. 이 학습 자료는 머신러닝의 기초 개념부터 주요 알고리즘, 실전 적용까지 체계적으로 다룹니다.

---

## 학습 로드맵

```
ML 개요 → 선형회귀 → 로지스틱 회귀 → 모델 평가 → 교차검증/하이퍼파라미터
                                              ↓
                    실전 프로젝트 ← 파이프라인 ← 차원축소 ← 클러스터링 ← k-NN/나이브베이즈
                        ↓                                                      ↑
                특성 공학 → 설명 가능성 → 불균형 데이터             결정트리 → 앙상블(배깅)
                        ↓                                            → 앙상블(부스팅) → SVM ──┘
                시계열 ML → AutoML → 이상 탐지
```

---

## 파일 목록

| 파일 | 주제 | 핵심 내용 |
|------|------|----------|
| [01_ML_Overview.md](./01_ML_Overview.md) | ML 개요 | 지도/비지도/강화학습, ML 워크플로우, 편향-분산 트레이드오프 |
| [02_Linear_Regression.md](./02_Linear_Regression.md) | 선형회귀 | 단순/다중 회귀, 경사하강법, 정규화(Ridge/Lasso) |
| [03_Logistic_Regression.md](./03_Logistic_Regression.md) | 로지스틱 회귀 | 이진 분류, 시그모이드 함수, 다중 분류(Softmax) |
| [04_Model_Evaluation.md](./04_Model_Evaluation.md) | 모델 평가 | 정확도, 정밀도, 재현율, F1-score, ROC-AUC |
| [05_Cross_Validation_Hyperparameters.md](./05_Cross_Validation_Hyperparameters.md) | 교차검증과 하이퍼파라미터 | K-Fold CV, GridSearchCV, RandomizedSearchCV |
| [06_Decision_Trees.md](./06_Decision_Trees.md) | 결정트리 | CART, 엔트로피, 지니 불순도, 가지치기 |
| [07_Ensemble_Bagging.md](./07_Ensemble_Bagging.md) | 앙상블 - 배깅 | Random Forest, 특성 중요도, OOB 에러 |
| [08_Ensemble_Boosting.md](./08_Ensemble_Boosting.md) | 앙상블 - 부스팅 | AdaBoost, Gradient Boosting, XGBoost, LightGBM |
| [09_SVM.md](./09_SVM.md) | SVM | 서포트 벡터, 마진, 커널 트릭 |
| [10_kNN_and_Naive_Bayes.md](./10_kNN_and_Naive_Bayes.md) | k-NN과 나이브베이즈 | 거리 기반 분류, 확률 기반 분류 |
| [11_Clustering.md](./11_Clustering.md) | 클러스터링 | K-Means, DBSCAN, 계층적 군집화 |
| [12_Dimensionality_Reduction.md](./12_Dimensionality_Reduction.md) | 차원축소 | PCA, t-SNE, 특성 선택 |
| [13_Pipelines_and_Practice.md](./13_Pipelines_and_Practice.md) | 파이프라인과 실무 | sklearn Pipeline, ColumnTransformer, 모델 저장 |
| [14_Practical_Projects.md](./14_Practical_Projects.md) | 실전 프로젝트 | Kaggle 문제 해결, 분류/회귀 실습 |
| [15_Feature_Engineering.md](./15_Feature_Engineering.md) | 특성 공학 | 수치형/범주형/시간형 변환, 특성 선택, Featuretools |
| [16_Model_Explainability.md](./16_Model_Explainability.md) | 모델 설명 가능성 | SHAP, LIME, PDP/ICE, 공정성 지표 |
| [17_Imbalanced_Data.md](./17_Imbalanced_Data.md) | 불균형 데이터 | SMOTE/ADASYN, 비용 민감 학습, 임계값 최적화 |
| [18_Time_Series_ML.md](./18_Time_Series_ML.md) | 시계열 ML | 지연/롤링 특성, TimeSeriesSplit, Prophet, 트리 기반 예측 |
| [19_AutoML_Hyperparameter_Optimization.md](./19_AutoML_Hyperparameter_Optimization.md) | AutoML과 하이퍼파라미터 최적화 | Optuna, Auto-sklearn, FLAML, H2O AutoML |
| [20_Anomaly_Detection.md](./20_Anomaly_Detection.md) | 이상 탐지 | Isolation Forest, LOF, One-Class SVM, PyOD |

---

## 환경 설정

### 필수 라이브러리 설치

```bash
# pip 사용
pip install numpy pandas matplotlib seaborn scikit-learn

# 추가 라이브러리 (부스팅)
pip install xgboost lightgbm catboost

# Jupyter Notebook (권장)
pip install jupyter
jupyter notebook
```

### 버전 확인

```python
import sklearn
import xgboost
import lightgbm

print(f"scikit-learn: {sklearn.__version__}")
print(f"XGBoost: {xgboost.__version__}")
print(f"LightGBM: {lightgbm.__version__}")
```

### 권장 버전
- Python: 3.9+
- scikit-learn: 1.2+
- XGBoost: 1.7+
- LightGBM: 3.3+

---

## 학습 순서 권장

### 1단계: 기초 이론 (01-04)
- 머신러닝 개념 이해
- 회귀와 분류의 기본
- 모델 평가 방법

### 2단계: 모델 튜닝 (05)
- 교차검증
- 하이퍼파라미터 최적화

### 3단계: 트리 기반 모델 (06-08)
- 결정트리
- 앙상블 기법

### 4단계: 기타 알고리즘 (09-10)
- SVM
- k-NN, 나이브베이즈

### 5단계: 비지도 학습 (11-12)
- 클러스터링
- 차원축소

### 6단계: 실무와 프로젝트 (13-14)
- 파이프라인 구축
- 실전 문제 해결

### 7단계: 심화 주제 (15-20)
- 특성 공학과 모델 설명 가능성
- 불균형 데이터 처리와 시계열
- AutoML, 하이퍼파라미터 최적화, 이상 탐지

---

## 알고리즘 선택 가이드

```
문제 유형 파악
    │
    ├── 정답이 있음 (지도학습)
    │       ├── 연속형 타겟 → 회귀
    │       │       ├── 선형 관계 → 선형회귀
    │       │       ├── 비선형 → 트리, 앙상블
    │       │       └── 해석력 중요 → 선형회귀, 결정트리
    │       │
    │       └── 범주형 타겟 → 분류
    │               ├── 이진 분류 → 로지스틱, SVM, 트리
    │               ├── 다중 분류 → 로지스틱(softmax), 트리
    │               └── 확률 필요 → 로지스틱, 나이브베이즈
    │
    └── 정답이 없음 (비지도학습)
            ├── 그룹화 → 클러스터링
            │       ├── 구형 클러스터 → K-Means
            │       └── 임의 형태 → DBSCAN
            │
            └── 차원축소 → PCA, t-SNE
```

---

## 참고 자료

### 공식 문서
- [scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [LightGBM Documentation](https://lightgbm.readthedocs.io/)

### 추천 데이터셋
- [Kaggle Datasets](https://www.kaggle.com/datasets)
- [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/)
- [sklearn.datasets](https://scikit-learn.org/stable/datasets.html)

### 추천 도서
- "Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow" - Aurélien Géron
- "An Introduction to Statistical Learning" - James et al.
