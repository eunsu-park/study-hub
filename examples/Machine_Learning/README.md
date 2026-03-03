# Machine Learning Examples

Runnable Jupyter Notebook examples corresponding to the 14 lessons in the Machine_Learning folder.

## Folder Structure

```
examples/
├── 01_linear_regression.ipynb      # Linear Regression
├── 02_logistic_regression.ipynb    # Logistic Regression
├── 03_model_evaluation.ipynb       # Model Evaluation Metrics
├── 04_cross_validation.ipynb       # Cross Validation
├── 05_preprocessing.ipynb          # Data Preprocessing
├── 06_decision_tree.ipynb          # Decision Tree
├── 07_random_forest.ipynb          # Random Forest
├── 08_xgboost_lightgbm.ipynb       # XGBoost, LightGBM
├── 09_svm.ipynb                    # SVM (Support Vector Machine)
├── 10_knn_naive_bayes.ipynb        # k-NN, Naive Bayes
├── 11_clustering.ipynb             # K-Means, DBSCAN
├── 12_pca.ipynb                    # PCA, t-SNE Dimensionality Reduction
├── 13_pipeline.ipynb               # sklearn Pipeline
├── 14_kaggle_project.ipynb         # Hands-on Kaggle Project
├── datasets/                       # Example Datasets
└── README.md
```

## How to Run

### Environment Setup

```bash
# Create a virtual environment (recommended)
python -m venv ml-env
source ml-env/bin/activate  # Windows: ml-env\Scripts\activate

# Install required packages
pip install numpy pandas matplotlib seaborn scikit-learn jupyter

# XGBoost, LightGBM (for Lesson 08)
pip install xgboost lightgbm
```

### Running Jupyter Notebook

```bash
cd Machine_Learning/examples
jupyter notebook

# Or JupyterLab
jupyter lab
```

## Lesson-by-Lesson Example List

| Lesson | Topic | Key Content |
|--------|-------|-------------|
| 01 | Linear Regression | Simple/Multiple Regression, MSE, R² |
| 02 | Logistic Regression | Binary/Multiclass Classification, ROC-AUC |
| 03 | Model Evaluation | Accuracy, Precision, Recall, F1 |
| 04 | Cross Validation | K-Fold, Stratified, GridSearchCV |
| 05 | Preprocessing | Scaling, Encoding, Missing Values |
| 06 | Decision Tree | Tree Visualization, Overfitting Prevention |
| 07 | Random Forest | Bagging, OOB, Feature Importance |
| 08 | XGBoost/LightGBM | Gradient Boosting, Early Stopping |
| 09 | SVM | Kernel Trick, Hyperplane |
| 10 | k-NN/Naive Bayes | Distance-based, Probability-based Classification |
| 11 | Clustering | K-Means, DBSCAN, Silhouette |
| 12 | Dimensionality Reduction | PCA, t-SNE, Explained Variance |
| 13 | Pipeline | Pipeline, ColumnTransformer |
| 14 | Kaggle Project | Titanic, Feature Engineering |
| 22 | Production ML Serving | Model Optimization, Serving Patterns, Drift Detection |
| 23 | A/B Testing | Power Analysis, Sequential Testing, Multi-Armed Bandit |
| 24 | Symbolic Regression | Expression Trees, Genetic Programming, Pareto Front |

## Learning Order

1. **Fundamentals**: 01 → 02 → 03 → 04 → 05
2. **Tree Models**: 06 → 07 → 08
3. **Other Algorithms**: 09 → 10
4. **Unsupervised Learning**: 11 → 12
5. **Hands-on Practice**: 13 → 14
6. **Production**: 22 → 23

## Datasets

Datasets used in the examples:

| Dataset | Source | Usage |
|---------|--------|-------|
| Iris | sklearn | Classification (Multiclass) |
| Wine | sklearn | Classification (Multiclass) |
| California Housing | sklearn | Regression |
| Digits | sklearn | Classification (Image) |
| Titanic | Kaggle | Classification (Real-world) |

## Required Packages

```
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=1.0.0
jupyter>=1.0.0
xgboost>=1.5.0      # Lesson 08
lightgbm>=3.3.0     # Lesson 08
```

## References

- [scikit-learn Official Documentation](https://scikit-learn.org/stable/)
- [Kaggle](https://www.kaggle.com/)
- [Machine Learning Mastery](https://machinelearningmastery.com/)
