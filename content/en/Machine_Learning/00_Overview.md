# Machine Learning Learning Guide

## Overview

Machine learning is a collection of algorithms that learn patterns from data to make predictions or decisions. This learning material systematically covers from basic concepts of machine learning to key algorithms and practical applications.

---

## Learning Roadmap

```
ML Overview → Linear Regression → Logistic Regression → Model Evaluation → Cross-Validation/Hyperparameters
                                                ↓
                Practical Projects ← Pipelines ← Dimensionality Reduction ← Clustering ← k-NN/Naive Bayes
                        ↓                                                                  ↑
                Feature Engineering → Explainability → Imbalanced Data        Decision Trees → Ensemble(Bagging)
                        ↓                                                         → Ensemble(Boosting) → SVM ──┘
                Time Series ML → AutoML → Anomaly Detection → Advanced Ensemble
                        ↓
                Production ML Serving → A/B Testing for ML
```

---

## File List

| File | Topic | Key Content |
|------|-------|-------------|
| [01_ML_Overview.md](./01_ML_Overview.md) | ML Overview | Supervised/Unsupervised/Reinforcement Learning, ML Workflow, Bias-Variance Tradeoff |
| [02_Linear_Regression.md](./02_Linear_Regression.md) | Linear Regression | Simple/Multiple Regression, Gradient Descent, Regularization (Ridge/Lasso) |
| [03_Logistic_Regression.md](./03_Logistic_Regression.md) | Logistic Regression | Binary Classification, Sigmoid Function, Multiclass (Softmax) |
| [04_Model_Evaluation.md](./04_Model_Evaluation.md) | Model Evaluation | Accuracy, Precision, Recall, F1-score, ROC-AUC |
| [05_Cross_Validation_Hyperparameters.md](./05_Cross_Validation_Hyperparameters.md) | Cross-Validation & Hyperparameters | K-Fold CV, GridSearchCV, RandomizedSearchCV |
| [06_Decision_Trees.md](./06_Decision_Trees.md) | Decision Trees | CART, Entropy, Gini Impurity, Pruning |
| [07_Ensemble_Bagging.md](./07_Ensemble_Bagging.md) | Ensemble - Bagging | Random Forest, Feature Importance, OOB Error |
| [08_Ensemble_Boosting.md](./08_Ensemble_Boosting.md) | Ensemble - Boosting | AdaBoost, Gradient Boosting, XGBoost, LightGBM |
| [09_SVM.md](./09_SVM.md) | SVM | Support Vectors, Margin, Kernel Trick |
| [10_kNN_and_Naive_Bayes.md](./10_kNN_and_Naive_Bayes.md) | k-NN & Naive Bayes | Distance-based Classification, Probability-based Classification |
| [11_Clustering.md](./11_Clustering.md) | Clustering | K-Means, DBSCAN, Hierarchical Clustering |
| [12_Dimensionality_Reduction.md](./12_Dimensionality_Reduction.md) | Dimensionality Reduction | PCA, t-SNE, Feature Selection |
| [13_Pipelines_and_Practice.md](./13_Pipelines_and_Practice.md) | Pipelines & Practice | sklearn Pipeline, ColumnTransformer, Model Saving |
| [14_Practical_Projects.md](./14_Practical_Projects.md) | Practical Projects | Kaggle Problem Solving, Classification/Regression Practice |
| [15_Feature_Engineering.md](./15_Feature_Engineering.md) | Feature Engineering | Numerical/Categorical/Temporal Transforms, Feature Selection, Featuretools |
| [16_Model_Explainability.md](./16_Model_Explainability.md) | Model Explainability | SHAP, LIME, PDP/ICE, Fairness Metrics |
| [17_Imbalanced_Data.md](./17_Imbalanced_Data.md) | Imbalanced Data | SMOTE/ADASYN, Cost-sensitive Learning, Threshold Optimization |
| [18_Time_Series_ML.md](./18_Time_Series_ML.md) | Time Series ML | Lag/Rolling Features, TimeSeriesSplit, Prophet, Tree-based Forecasting |
| [19_AutoML_Hyperparameter_Optimization.md](./19_AutoML_Hyperparameter_Optimization.md) | AutoML & Hyperparameter Optimization | Optuna, Auto-sklearn, FLAML, H2O AutoML |
| [20_Anomaly_Detection.md](./20_Anomaly_Detection.md) | Anomaly Detection | Isolation Forest, LOF, One-Class SVM, PyOD |
| [21_Advanced_Ensemble.md](./21_Advanced_Ensemble.md) | Advanced Ensemble | Stacking, Blending, Meta-Learner, Diverse Base Learners, Competition Strategies |
| [22_Production_ML_Serving.md](./22_Production_ML_Serving.md) | Production ML Serving | Model Optimization, Serving Patterns, Training-Serving Skew, Drift Detection |
| [23_AB_Testing_for_ML.md](./23_AB_Testing_for_ML.md) | A/B Testing for ML | Power Analysis, Hypothesis Testing, Sequential Testing, Multi-Armed Bandits, Interleaving |

---

## Environment Setup

### Install Required Libraries

```bash
# Using pip
pip install numpy pandas matplotlib seaborn scikit-learn

# Additional libraries (boosting)
pip install xgboost lightgbm catboost

# Jupyter Notebook (recommended)
pip install jupyter
jupyter notebook
```

### Version Check

```python
import sklearn
import xgboost
import lightgbm

print(f"scikit-learn: {sklearn.__version__}")
print(f"XGBoost: {xgboost.__version__}")
print(f"LightGBM: {lightgbm.__version__}")
```

### Recommended Versions
- Python: 3.9+
- scikit-learn: 1.2+
- XGBoost: 1.7+
- LightGBM: 3.3+

---

## Recommended Learning Order

### Stage 1: Basic Theory (01-04)
- Understand machine learning concepts
- Basics of regression and classification
- Model evaluation methods

### Stage 2: Model Tuning (05)
- Cross-validation
- Hyperparameter optimization

### Stage 3: Tree-based Models (06-08)
- Decision trees
- Ensemble techniques

### Stage 4: Other Algorithms (09-10)
- SVM
- k-NN, Naive Bayes

### Stage 5: Unsupervised Learning (11-12)
- Clustering
- Dimensionality reduction

### Stage 6: Practice & Projects (13-14)
- Building pipelines
- Real-world problem solving

### Stage 7: Advanced Topics (15-21)
- Feature engineering and model explainability
- Handling imbalanced data and time series
- AutoML, hyperparameter optimization, anomaly detection
- Advanced ensemble methods (stacking, blending)

### Stage 8: Production (22-23)
- Model optimization and serving patterns
- A/B testing and online experimentation

---

## Algorithm Selection Guide

```
Identify Problem Type
    │
    ├── Has Labels (Supervised Learning)
    │       ├── Continuous Target → Regression
    │       │       ├── Linear Relationship → Linear Regression
    │       │       ├── Non-linear → Trees, Ensemble
    │       │       └── Interpretability Important → Linear Regression, Decision Trees
    │       │
    │       └── Categorical Target → Classification
    │               ├── Binary Classification → Logistic, SVM, Trees
    │               ├── Multiclass → Logistic (softmax), Trees
    │               └── Need Probabilities → Logistic, Naive Bayes
    │
    └── No Labels (Unsupervised Learning)
            ├── Grouping → Clustering
            │       ├── Spherical Clusters → K-Means
            │       └── Arbitrary Shapes → DBSCAN
            │
            └── Dimensionality Reduction → PCA, t-SNE
```

---

## References

### Official Documentation
- [scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [LightGBM Documentation](https://lightgbm.readthedocs.io/)

### Recommended Datasets
- [Kaggle Datasets](https://www.kaggle.com/datasets)
- [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/)
- [sklearn.datasets](https://scikit-learn.org/stable/datasets.html)

### Recommended Books
- "Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow" - Aurélien Géron
- "An Introduction to Statistical Learning" - James et al.
