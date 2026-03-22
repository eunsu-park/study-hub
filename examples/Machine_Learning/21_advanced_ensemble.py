"""
Advanced Ensemble Methods: Stacking and Blending
=================================================
Demonstrates:
  1. StackingClassifier with diverse base learners (LR, RF, SVM, KNN)
  2. Blending (manual holdout-based ensemble)
  3. Prediction correlation heatmap for diversity analysis
  4. Performance comparison: single models vs. stacked vs. blended
  5. Hyperparameter tuning for stacked ensembles (Optuna)

Requirements:
  pip install scikit-learn numpy matplotlib seaborn optuna
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.model_selection import (
    train_test_split, cross_val_score, cross_val_predict, StratifiedKFold
)
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

np.random.seed(42)

# ============================================================
# 1. Generate Dataset
# ============================================================
# Why make_classification: Synthetic data lets us control complexity and reproduce
# results. flip_y=0.03 adds realistic label noise.
X, y = make_classification(
    n_samples=3000, n_features=20, n_informative=12, n_redundant=4,
    n_clusters_per_class=2, flip_y=0.03, random_state=42
)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# ============================================================
# 2. Define Diverse Base Learners
# ============================================================
# Why these four? Each learns boundaries in a fundamentally different way:
#   LR = linear boundary | RF = axis-aligned tree splits
#   SVM = kernel-space boundary | KNN = local distance vote
base_models = [
    ('LR', make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=42))),
    ('RF', RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)),
    ('SVM', make_pipeline(StandardScaler(), SVC(probability=True, C=1.0, random_state=42))),
    ('KNN', make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=15))),
]

# ============================================================
# 3. Evaluate Individual Models
# ============================================================
print("\n" + "=" * 60)
print("Individual Model Performance (5-Fold CV)")
print("=" * 60)

# Why StratifiedKFold: Preserves class balance. Same folds for all models
# ensures fair meta-feature generation.
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
individual_scores = {}
for name, model in base_models:
    scores = cross_val_score(model, X_train, y_train, cv=kf, scoring='accuracy')
    individual_scores[name] = scores.mean()
    print(f"  {name:5s}: {scores.mean():.4f} (+/- {scores.std():.4f})")

# ============================================================
# 4. Prediction Correlation Heatmap (Diversity Analysis)
# ============================================================
# Why: If two models make the same errors, combining them adds nothing.
# We want low correlation = high diversity.
print("\n" + "=" * 60)
print("Prediction Diversity Analysis")
print("=" * 60)

oof_predictions = {}
for name, model in base_models:
    oof_predictions[name] = cross_val_predict(model, X_train, y_train, cv=kf)

pred_matrix = np.array(list(oof_predictions.values()))
corr = np.corrcoef(pred_matrix)
names_list = list(oof_predictions.keys())

for i in range(len(names_list)):
    for j in range(i + 1, len(names_list)):
        print(f"  Corr({names_list[i]}, {names_list[j]}): {corr[i, j]:.3f}")

fig, ax = plt.subplots(figsize=(7, 5))
sns.heatmap(corr, xticklabels=names_list, yticklabels=names_list,
            annot=True, fmt='.2f', cmap='RdYlGn_r', vmin=0, vmax=1, ax=ax)
ax.set_title('Base Model Prediction Correlation\n(Lower = More Diverse)')
plt.tight_layout()
plt.savefig('correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# 5. Stacking with StackingClassifier
# ============================================================
print("\n" + "=" * 60)
print("Stacking (sklearn StackingClassifier)")
print("=" * 60)

# Why LogisticRegression as meta-learner: Only 4 meta-features — a linear
# model is the right complexity. cv=kf prevents data leakage via OOF predictions.
stacking_clf = StackingClassifier(
    estimators=base_models,
    final_estimator=LogisticRegression(max_iter=1000, C=1.0),
    cv=kf, stack_method='auto', passthrough=False, n_jobs=-1
)
stacking_scores = cross_val_score(stacking_clf, X_train, y_train, cv=kf, scoring='accuracy')
print(f"  Stacking CV: {stacking_scores.mean():.4f} (+/- {stacking_scores.std():.4f})")

stacking_clf.fit(X_train, y_train)
stack_test_acc = accuracy_score(y_test, stacking_clf.predict(X_test))
print(f"  Stacking Test: {stack_test_acc:.4f}")

# ============================================================
# 6. Blending (Holdout-based Ensemble)
# ============================================================
# Why blending? Simpler and faster than stacking — one train/predict cycle
# per model. Tradeoff: loses some training data to the blend holdout.
print("\n" + "=" * 60)
print("Blending (Holdout-based)")
print("=" * 60)

X_tr, X_blend, y_tr, y_blend = train_test_split(
    X_train, y_train, test_size=0.2, stratify=y_train, random_state=42
)

blend_meta = np.zeros((X_blend.shape[0], len(base_models)))
test_meta = np.zeros((X_test.shape[0], len(base_models)))

for i, (name, model) in enumerate(base_models):
    model.fit(X_tr, y_tr)
    # Why predict_proba[:, 1]: Probabilities carry confidence info — richer
    # than hard 0/1 labels for the meta-learner.
    blend_meta[:, i] = model.predict_proba(X_blend)[:, 1]
    test_meta[:, i] = model.predict_proba(X_test)[:, 1]

meta_lr = LogisticRegression(max_iter=1000, C=1.0)
meta_lr.fit(blend_meta, y_blend)
blend_test_acc = accuracy_score(y_test, meta_lr.predict(test_meta))
print(f"  Blending Test: {blend_test_acc:.4f}")

print("  Meta-learner coefficients:")
for name, coef in zip([n for n, _ in base_models], meta_lr.coef_[0]):
    print(f"    {name:5s}: {coef:+.4f}")

# ============================================================
# 7. Performance Comparison
# ============================================================
print("\n" + "=" * 60)
print("Performance Comparison (Test Set)")
print("=" * 60)

all_results = {}
for name, model in base_models:
    model.fit(X_train, y_train)
    all_results[name] = accuracy_score(y_test, model.predict(X_test))
all_results['Stacking'] = stack_test_acc
all_results['Blending'] = blend_test_acc

for name, acc in all_results.items():
    print(f"  {name:10s}: {acc:.4f}")

fig, ax = plt.subplots(figsize=(10, 5))
bar_names = list(all_results.keys())
accs = list(all_results.values())
colors = ['#4C72B0'] * len(base_models) + ['#DD5050', '#55A868']
bars = ax.bar(bar_names, accs, color=colors, edgecolor='white')
ax.set_ylabel('Test Accuracy')
ax.set_title('Single Models vs. Ensemble Methods')
ax.set_ylim(min(accs) - 0.02, max(accs) + 0.01)
for bar, acc in zip(bars, accs):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
            f'{acc:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
ax.axhline(y=max(individual_scores.values()), color='gray', linestyle='--',
           linewidth=0.8, label=f'Best Single ({max(individual_scores.values()):.4f})')
ax.legend()
plt.tight_layout()
plt.savefig('ensemble_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# 8. Hyperparameter Tuning (Optuna)
# ============================================================
# Why Optuna? Bayesian optimization (TPE) explores the HP space efficiently,
# spending more trials on promising regions.
print("\n" + "=" * 60)
print("Hyperparameter Tuning with Optuna")
print("=" * 60)

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        rf_n = trial.suggest_int('rf_n_estimators', 50, 300, step=50)
        rf_depth = trial.suggest_int('rf_max_depth', 3, 15)
        svm_C = trial.suggest_float('svm_C', 0.01, 100.0, log=True)
        knn_k = trial.suggest_int('knn_k', 3, 30)
        meta_C = trial.suggest_float('meta_C', 0.01, 10.0, log=True)
        passthrough = trial.suggest_categorical('passthrough', [True, False])

        stack = StackingClassifier(
            estimators=[
                ('lr', make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=42))),
                ('rf', RandomForestClassifier(n_estimators=rf_n, max_depth=rf_depth, random_state=42)),
                ('svm', make_pipeline(StandardScaler(), SVC(probability=True, C=svm_C, random_state=42))),
                ('knn', make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=knn_k))),
            ],
            final_estimator=LogisticRegression(C=meta_C, max_iter=1000),
            cv=3, passthrough=passthrough, n_jobs=-1  # cv=3 for speed during HPO
        )
        return cross_val_score(stack, X_train, y_train, cv=3, scoring='accuracy').mean()

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=20, show_progress_bar=False)

    print(f"  Best CV Accuracy: {study.best_value:.4f}")
    print(f"  Best Params: {study.best_params}")

    # Retrain best config with full cv=5
    bp = study.best_params
    tuned_stack = StackingClassifier(
        estimators=[
            ('lr', make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=42))),
            ('rf', RandomForestClassifier(n_estimators=bp['rf_n_estimators'], max_depth=bp['rf_max_depth'], random_state=42)),
            ('svm', make_pipeline(StandardScaler(), SVC(probability=True, C=bp['svm_C'], random_state=42))),
            ('knn', make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=bp['knn_k']))),
        ],
        final_estimator=LogisticRegression(C=bp['meta_C'], max_iter=1000),
        cv=5, passthrough=bp['passthrough'], n_jobs=-1
    )
    tuned_stack.fit(X_train, y_train)
    tuned_acc = accuracy_score(y_test, tuned_stack.predict(X_test))
    print(f"  Tuned Test Accuracy: {tuned_acc:.4f} (delta: {tuned_acc - stack_test_acc:+.4f})")

except ImportError:
    print("  Optuna not installed — pip install optuna")

# ============================================================
# 9. Summary
# ============================================================
print("\n" + "=" * 60)
print("Summary")
print("=" * 60)
best_single = max(individual_scores, key=individual_scores.get)
print(f"  Best single model:  {best_single} ({individual_scores[best_single]:.4f})")
print(f"  Stacking accuracy:  {stack_test_acc:.4f} (gain: {stack_test_acc - individual_scores[best_single]:+.4f})")
print(f"  Blending accuracy:  {blend_test_acc:.4f} (gain: {blend_test_acc - individual_scores[best_single]:+.4f})")
print("\nKey takeaways:")
print("  - Stacking combines diverse models via a learned meta-learner")
print("  - Blending is simpler (holdout-based) but wastes some training data")
print("  - Diversity (low prediction correlation) is the key to ensemble gains")
print("  - A simple meta-learner (LogisticRegression) usually works best")
