"""
k-NN and Naive Bayes - Exercise Solutions
===========================================
Lesson 10: k-NN and Naive Bayes

Exercises cover:
  1. k-NN classification: find optimal k on wine dataset
  2. k-NN regression: sine wave with different k values
  3. Text classification with Naive Bayes (synthetic data)
  4. Feature scaling impact on k-NN
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine, load_breast_cancer
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report


# ============================================================
# Exercise 1: k-NN Classification
# Find the optimal k value on the wine dataset using CV.
# ============================================================
def exercise_1_knn_classification():
    """Find optimal k for k-NN using cross-validation.

    k is the most critical hyperparameter:
    - Small k (e.g., 1): complex, jagged decision boundary; captures noise
    - Large k: smooth boundary; may miss local patterns
    The optimal k balances bias (too smooth) and variance (too noisy).

    We sweep k from 1 to 30 and pick the one with highest CV accuracy.
    """
    print("=" * 60)
    print("Exercise 1: k-NN Classification (Wine Dataset)")
    print("=" * 60)

    wine = load_wine()
    X, y = wine.data, wine.target

    # Scaling is critical for k-NN since it's distance-based
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier())
    ])

    k_range = range(1, 31)
    cv_scores = []

    for k in k_range:
        pipe.set_params(knn__n_neighbors=k)
        scores = cross_val_score(pipe, X, y, cv=5, scoring="accuracy")
        cv_scores.append(scores.mean())

    best_k = k_range[np.argmax(cv_scores)]
    print(f"Optimal k: {best_k}")
    print(f"Best CV accuracy: {max(cv_scores):.4f}")

    # Plot k vs accuracy
    plt.figure(figsize=(9, 5))
    plt.plot(k_range, cv_scores, "o-", markersize=4)
    plt.axvline(x=best_k, color="red", linestyle="--",
                label=f"Best k={best_k}")
    plt.xlabel("k (Number of Neighbors)")
    plt.ylabel("CV Accuracy")
    plt.title("k-NN: Optimal k Selection")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("10_ex1_knn_k_selection.png", dpi=100)
    plt.close()
    print("Plot saved: 10_ex1_knn_k_selection.png")


# ============================================================
# Exercise 2: k-NN Regression
# Sine wave regression with different k values.
# ============================================================
def exercise_2_knn_regression():
    """k-NN regression on a sine wave to visualize the bias-variance trade-off.

    k-NN regression predicts the average of k nearest neighbors' targets.
    - k=1: interpolates exactly through training points (high variance)
    - k=20: produces a very smooth curve (high bias, may miss the sine shape)
    - k=5: reasonable balance for this dataset size
    """
    print("\n" + "=" * 60)
    print("Exercise 2: k-NN Regression (Sine Wave)")
    print("=" * 60)

    # Generate sine wave data with noise
    np.random.seed(42)
    X = np.sort(np.random.uniform(0, 2 * np.pi, 100)).reshape(-1, 1)
    y = np.sin(X.ravel()) + np.random.randn(100) * 0.2

    X_plot = np.linspace(0, 2 * np.pi, 200).reshape(-1, 1)

    k_values = [1, 3, 5, 10, 20]
    fig, axes = plt.subplots(1, len(k_values), figsize=(18, 4))

    for ax, k in zip(axes, k_values):
        knn = KNeighborsRegressor(n_neighbors=k)
        knn.fit(X, y)
        y_pred = knn.predict(X_plot)

        # Cross-validated score
        cv_score = cross_val_score(
            KNeighborsRegressor(n_neighbors=k), X, y, cv=5,
            scoring="neg_mean_squared_error"
        ).mean()

        ax.scatter(X, y, s=15, alpha=0.5, label="Data")
        ax.plot(X_plot, np.sin(X_plot.ravel()), "g--", alpha=0.5, label="True")
        ax.plot(X_plot, y_pred, "r-", linewidth=2, label="k-NN")
        ax.set_title(f"k={k}\nCV MSE={-cv_score:.3f}")
        ax.legend(fontsize=7)

    plt.suptitle("k-NN Regression: Effect of k", fontsize=13)
    plt.tight_layout()
    plt.savefig("10_ex2_knn_regression.png", dpi=100)
    plt.close()
    print("Plot saved: 10_ex2_knn_regression.png")

    # Print summary
    print(f"{'k':<5} {'CV MSE':>10}")
    print("-" * 17)
    for k in k_values:
        cv = cross_val_score(KNeighborsRegressor(n_neighbors=k),
                             X, y, cv=5, scoring="neg_mean_squared_error")
        print(f"{k:<5} {-cv.mean():>10.4f}")


# ============================================================
# Exercise 3: Text Classification
# Use Naive Bayes for sentiment classification (synthetic data).
# ============================================================
def exercise_3_text_classification():
    """Naive Bayes text classification with synthetic movie reviews.

    MultinomialNB is ideal for text because:
    1. It models word count distributions (matching bag-of-words features)
    2. Training is extremely fast (just counting and dividing)
    3. The naive independence assumption works surprisingly well for text
       because correlated words still provide discriminative signal

    Alpha (smoothing) prevents zero probabilities for unseen words:
    P(word|class) = (count + alpha) / (total + alpha * vocab_size)
    """
    print("\n" + "=" * 60)
    print("Exercise 3: Text Classification with Naive Bayes")
    print("=" * 60)

    # Synthetic movie review data (since we don't need external datasets)
    positive_reviews = [
        "This movie was excellent and truly amazing",
        "I loved every minute of this wonderful film",
        "Great acting brilliant story highly recommend",
        "Fantastic movie best I have seen this year",
        "Beautiful cinematography outstanding performances",
        "A masterpiece of filmmaking truly exceptional",
        "Heartwarming story with superb acting talent",
        "Incredible movie exceeded all my expectations",
        "Perfect blend of humor and drama loved it",
        "One of the best movies I have ever watched",
        "Thoroughly enjoyed this amazing experience",
        "Brilliant direction and wonderful screenplay",
    ]
    negative_reviews = [
        "This movie was terrible waste of time",
        "I hated this boring and pointless film",
        "Awful acting horrible story would not recommend",
        "Worst movie I have ever seen really bad",
        "Disappointing waste of money and time",
        "Dull boring predictable complete garbage",
        "Terrible direction and awful screenplay disaster",
        "Unwatchable I walked out of the theater",
        "Painfully bad acting and a nonsensical plot",
        "A total failure on every level avoid it",
        "Incredibly boring and uninspired filmmaking",
        "Could not finish this terrible excuse for a movie",
    ]

    texts = positive_reviews + negative_reviews
    labels = [1] * len(positive_reviews) + [0] * len(negative_reviews)

    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.25, random_state=42, stratify=labels
    )

    # Pipeline: TF-IDF vectorization + Naive Bayes
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("nb", MultinomialNB(alpha=1.0))
    ])
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred,
                                target_names=["Negative", "Positive"]))

    # Test with new reviews
    new_reviews = [
        "An absolutely wonderful and brilliant movie",
        "Terrible film I regret watching it",
        "Not bad but not great either",
    ]
    predictions = pipe.predict(new_reviews)
    probas = pipe.predict_proba(new_reviews)
    print("New review predictions:")
    for review, pred, prob in zip(new_reviews, predictions, probas):
        label = "Positive" if pred == 1 else "Negative"
        print(f"  '{review[:50]}...' -> {label} (confidence: {max(prob):.2f})")


# ============================================================
# Exercise 4: Feature Scaling Impact
# Compare k-NN on breast cancer with and without scaling.
# ============================================================
def exercise_4_scaling_impact():
    """Demonstrate the critical impact of feature scaling on k-NN.

    k-NN relies on distance metrics (default: Euclidean). Without scaling,
    features with larger numeric ranges dominate the distance calculation.

    Example: 'mean area' ranges ~100-2500, while 'mean smoothness' ranges
    ~0.05-0.16. Without scaling, k-NN essentially ignores smoothness
    because its contribution to Euclidean distance is negligible.
    """
    print("\n" + "=" * 60)
    print("Exercise 4: Feature Scaling Impact on k-NN")
    print("=" * 60)

    cancer = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        cancer.data, cancer.target, test_size=0.2, random_state=42
    )

    # Without scaling
    knn_raw = KNeighborsClassifier(n_neighbors=5)
    knn_raw.fit(X_train, y_train)
    acc_raw = knn_raw.score(X_test, y_test)

    # With scaling
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    knn_scaled = KNeighborsClassifier(n_neighbors=5)
    knn_scaled.fit(X_train_s, y_train)
    acc_scaled = knn_scaled.score(X_test_s, y_test)

    print(f"Without scaling: {acc_raw:.4f}")
    print(f"With scaling:    {acc_scaled:.4f}")
    print(f"Improvement:     {acc_scaled - acc_raw:+.4f}")

    # Show why: feature ranges differ by orders of magnitude
    ranges = X_train.max(axis=0) - X_train.min(axis=0)
    sorted_idx = np.argsort(ranges)[::-1]

    print(f"\nTop 5 features by range (dominate unscaled distance):")
    for i in range(5):
        idx = sorted_idx[i]
        print(f"  {cancer.feature_names[idx]}: range = {ranges[idx]:.2f}")

    print(f"\nBottom 5 features by range (ignored without scaling):")
    for i in range(-5, 0):
        idx = sorted_idx[i]
        print(f"  {cancer.feature_names[idx]}: range = {ranges[idx]:.6f}")


if __name__ == "__main__":
    exercise_1_knn_classification()
    exercise_2_knn_regression()
    exercise_3_text_classification()
    exercise_4_scaling_impact()
