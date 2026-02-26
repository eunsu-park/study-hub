# k-Nearest Neighbors (k-NN) and Naive Bayes

**Previous**: [SVM](./09_SVM.md) | **Next**: [Clustering](./11_Clustering.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain the k-NN algorithm and how it performs lazy (instance-based) learning
2. Compare distance metrics (Euclidean, Manhattan, Cosine) and identify when each is appropriate
3. Apply cross-validation to select the optimal k value and evaluate uniform vs. distance-weighted voting
4. Describe Bayes' theorem and the "naive" conditional independence assumption
5. Distinguish between Gaussian, Multinomial, and Bernoulli Naive Bayes and match each to its data type
6. Implement a text classification pipeline using TF-IDF and Multinomial Naive Bayes
7. Evaluate the trade-offs between k-NN and Naive Bayes in terms of speed, memory, and dimensionality

---

This lesson covers two algorithms that sit at opposite ends of the machine learning spectrum. k-NN makes predictions by finding similar examples in the training data -- simple and assumption-free, but slow at prediction time. Naive Bayes takes a probabilistic approach using Bayes' theorem with a bold independence assumption that rarely holds in practice yet yields surprisingly competitive results, especially for text classification.

---

## Part 1: k-Nearest Neighbors (k-NN)

## 1. k-NN Concepts

### 1.1 Basic Principle

**"You are the average of your k nearest neighbors"**

- **Non-parametric algorithm**: Doesn't assume any data distribution
- **Instance-based learning (lazy learning)**: Stores all training data, no explicit training phase
- **Prediction**: Based on majority vote (classification) or average (regression) of k nearest neighbors

### 1.2 How k-NN Works

```
1. Store all training data
2. For a new data point:
   a. Calculate distance to all training points
   b. Find k nearest neighbors
   c. Classification: Majority vote among k neighbors
      Regression: Average of k neighbors' values
```

### 1.3 Distance Metrics

| Metric | Formula | Use Case |
|--------|---------|----------|
| **Euclidean** | √(Σ(xi - yi)²) | Most common, continuous features |
| **Manhattan** | Σ\|xi - yi\| | High-dimensional data, grid-like paths |
| **Minkowski** | (Σ\|xi - yi\|^p)^(1/p) | Generalization (p=1: Manhattan, p=2: Euclidean) |
| **Hamming** | Count of differing positions | Categorical/binary features |
| **Cosine** | 1 - (x·y)/(||x|| ||y||) | Text data, high-dimensional sparse data |

---

## 2. k-NN Classification

### 2.1 Basic Implementation

```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Load data
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42
)

# Train k-NN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Predict
y_pred = knn.predict(X_test)

# Evaluate
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))
```

### 2.2 Choosing Optimal k

```python
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import numpy as np

# Try different k values
k_values = range(1, 31)
cv_scores = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train, y_train, cv=5, scoring='accuracy')
    cv_scores.append(scores.mean())

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(k_values, cv_scores, marker='o')
plt.xlabel('k (Number of Neighbors)')
plt.ylabel('Cross-Validation Accuracy')
plt.title('Optimal k Selection')
plt.grid(True)
plt.show()

# Best k
optimal_k = k_values[np.argmax(cv_scores)]
print(f"Optimal k: {optimal_k}, Accuracy: {max(cv_scores):.4f}")
```

**Guidelines for choosing k**:
- **Small k (e.g., 1-5)**: More complex decision boundary, sensitive to noise
- **Large k**: Smoother decision boundary, may underfit
- **Rule of thumb**: Start with k = √N (N = number of training samples)
- **Odd k**: Avoids ties in binary classification

### 2.3 Distance Metrics Comparison

```python
metrics = ['euclidean', 'manhattan', 'minkowski', 'cosine']
results = {}

for metric in metrics:
    knn = KNeighborsClassifier(n_neighbors=5, metric=metric)
    knn.fit(X_train, y_train)
    results[metric] = knn.score(X_test, y_test)

import pandas as pd
df_results = pd.DataFrame(list(results.items()), columns=['Metric', 'Accuracy'])
print(df_results.sort_values('Accuracy', ascending=False))
```

### 2.4 Weighted k-NN

Give closer neighbors more weight:

```python
# Uniform weights (all neighbors equal)
knn_uniform = KNeighborsClassifier(n_neighbors=5, weights='uniform')
knn_uniform.fit(X_train, y_train)

# Distance-based weights (closer neighbors have more influence)
knn_distance = KNeighborsClassifier(n_neighbors=5, weights='distance')
knn_distance.fit(X_train, y_train)

print(f"Uniform weights: {knn_uniform.score(X_test, y_test):.4f}")
print(f"Distance weights: {knn_distance.score(X_test, y_test):.4f}")
```

---

## 3. k-NN Regression

### 3.1 Basic Implementation

```python
from sklearn.neighbors import KNeighborsRegressor
from sklearn.datasets import load_diabetes
from sklearn.metrics import mean_squared_error, r2_score

# Load data
diabetes = load_diabetes()
X_train, X_test, y_train, y_test = train_test_split(
    diabetes.data, diabetes.target, test_size=0.2, random_state=42
)

# Train k-NN Regressor
knn_reg = KNeighborsRegressor(n_neighbors=5)
knn_reg.fit(X_train, y_train)

# Predict
y_pred = knn_reg.predict(X_test)

# Evaluate
print(f"MSE: {mean_squared_error(y_test, y_pred):.2f}")
print(f"R²: {r2_score(y_test, y_pred):.4f}")
```

### 3.2 Visualizing k-NN Regression

```python
from sklearn.datasets import make_regression
import numpy as np

# Generate 1D regression data
X, y = make_regression(n_samples=100, n_features=1, noise=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train with different k values
k_values = [1, 5, 20]
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for ax, k in zip(axes, k_values):
    knn = KNeighborsRegressor(n_neighbors=k)
    knn.fit(X_train, y_train)

    # Predict on smooth curve
    X_plot = np.linspace(X.min(), X.max(), 300).reshape(-1, 1)
    y_plot = knn.predict(X_plot)

    # Plot
    ax.scatter(X_train, y_train, alpha=0.6, label='Train')
    ax.scatter(X_test, y_test, alpha=0.6, label='Test')
    ax.plot(X_plot, y_plot, 'r-', linewidth=2, label='k-NN prediction')
    ax.set_title(f'k={k}')
    ax.set_xlabel('X')
    ax.set_ylabel('y')
    ax.legend()

plt.tight_layout()
plt.show()
```

---

## 4. Feature Scaling for k-NN

### 4.1 Why Scaling is Critical

k-NN is **distance-based**, so features with large scales dominate the distance calculation.

```python
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Without scaling
knn_no_scale = KNeighborsClassifier(n_neighbors=5)
knn_no_scale.fit(X_train, y_train)

# With scaling (Pipeline)
knn_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('knn', KNeighborsClassifier(n_neighbors=5))
])
knn_pipeline.fit(X_train, y_train)

print("Without Scaling:")
print(f"  Train: {knn_no_scale.score(X_train, y_train):.4f}")
print(f"  Test:  {knn_no_scale.score(X_test, y_test):.4f}")

print("\nWith Scaling:")
print(f"  Train: {knn_pipeline.score(X_train, y_train):.4f}")
print(f"  Test:  {knn_pipeline.score(X_test, y_test):.4f}")
```

---

## 5. Advantages and Disadvantages of k-NN

### 5.1 Advantages

1. **Simple and Intuitive**: Easy to understand and implement
2. **No Training Phase**: Fast training (just stores data)
3. **No Assumptions**: Works with any data distribution
4. **Versatile**: Works for classification and regression

### 5.2 Disadvantages

1. **Slow Prediction**: O(N) per prediction (N = training set size)
2. **Memory Intensive**: Stores all training data
3. **Sensitive to Feature Scaling**: Requires normalization
4. **Curse of Dimensionality**: Performance degrades in high dimensions
5. **Sensitive to Irrelevant Features**: Noisy features hurt performance

---

## Part 2: Naive Bayes

## 6. Naive Bayes Concepts

### 6.1 Bayes' Theorem

```
P(Class|Features) = P(Features|Class) × P(Class) / P(Features)

Where:
- P(Class|Features): Posterior probability (what we want)
- P(Features|Class): Likelihood (probability of features given class)
- P(Class): Prior probability (class frequency in training data)
- P(Features): Evidence (normalizing constant)
```

### 6.2 "Naive" Assumption

**Assumption**: All features are independent given the class

```
P(x1, x2, ..., xn | Class) = P(x1|Class) × P(x2|Class) × ... × P(xn|Class)
```

This assumption is rarely true in practice, but Naive Bayes works well anyway!

---

## 7. Types of Naive Bayes

### 7.1 Gaussian Naive Bayes

**Assumption**: Features follow a normal (Gaussian) distribution

**Use Case**: Continuous features

```python
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import load_iris

# Load data
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42
)

# Train Gaussian Naive Bayes
gnb = GaussianNB()
gnb.fit(X_train, y_train)

# Predict
y_pred = gnb.predict(X_test)

# Evaluate
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))
```

### 7.2 Multinomial Naive Bayes

**Assumption**: Features represent counts or frequencies

**Use Case**: Text classification (word counts, TF-IDF)

```python
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.datasets import fetch_20newsgroups

# Load text data
categories = ['alt.atheism', 'talk.religion.misc']
newsgroups_train = fetch_20newsgroups(subset='train', categories=categories, random_state=42)
newsgroups_test = fetch_20newsgroups(subset='test', categories=categories, random_state=42)

# Convert text to word counts
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(newsgroups_train.data)
X_test = vectorizer.transform(newsgroups_test.data)
y_train = newsgroups_train.target
y_test = newsgroups_test.target

# Train Multinomial Naive Bayes
mnb = MultinomialNB()
mnb.fit(X_train, y_train)

# Predict
y_pred = mnb.predict(X_test)

# Evaluate
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=newsgroups_test.target_names))
```

### 7.3 Bernoulli Naive Bayes

**Assumption**: Features are binary (0 or 1)

**Use Case**: Binary features (document contains word or not)

```python
from sklearn.naive_bayes import BernoulliNB

# Convert to binary (word present or not)
X_train_binary = (X_train > 0).astype(int)
X_test_binary = (X_test > 0).astype(int)

# Train Bernoulli Naive Bayes
bnb = BernoulliNB()
bnb.fit(X_train_binary, y_train)

# Predict
y_pred = bnb.predict(X_test_binary)

# Evaluate
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
```

---

## 8. Text Classification with Naive Bayes

### 8.1 Complete Pipeline

```python
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

# Create pipeline
text_clf = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=1000, stop_words='english')),
    ('clf', MultinomialNB(alpha=1.0))
])

# Train
text_clf.fit(newsgroups_train.data, newsgroups_train.target)

# Predict
predicted = text_clf.predict(newsgroups_test.data)

# Evaluate
print(f"Accuracy: {accuracy_score(newsgroups_test.target, predicted):.4f}")

# Test on custom text
docs_new = [
    'God is love',
    'OpenGL on the GPU is fast'
]
predicted_new = text_clf.predict(docs_new)
for doc, category in zip(docs_new, predicted_new):
    print(f'{doc} => {newsgroups_test.target_names[category]}')
```

### 8.2 Alpha Parameter (Laplace Smoothing)

**Problem**: What if a word never appears in a class during training?
- P(word|class) = 0 → Entire probability becomes 0

**Solution**: Add-one smoothing (Laplace smoothing)
- Add α (alpha) to all counts

```python
# Compare different alpha values
alphas = [0.1, 1.0, 10.0]

for alpha in alphas:
    mnb = MultinomialNB(alpha=alpha)
    mnb.fit(X_train, y_train)
    accuracy = mnb.score(X_test, y_test)
    print(f"alpha={alpha}: Accuracy={accuracy:.4f}")
```

---

## 9. Comparison: k-NN vs Naive Bayes

| Feature | k-NN | Naive Bayes |
|---------|------|-------------|
| **Type** | Instance-based | Probabilistic |
| **Training Speed** | Fast (no training) | Fast |
| **Prediction Speed** | Slow (O(N)) | Fast |
| **Memory Usage** | High (stores all data) | Low (stores parameters) |
| **Feature Scaling** | Required | Not required |
| **High Dimensions** | Suffers (curse of dimensionality) | Works well (especially text) |
| **Interpretability** | Hard | Easy (probability-based) |
| **Best Use Case** | Small datasets, low dimensions | Text classification, categorical data |

---

## 10. Practical Tips

### 10.1 When to Use k-NN

- Small to medium datasets (<10K samples)
- Low-dimensional data (<20 features)
- Non-linear decision boundaries
- Need for interpretability (show nearest neighbors)

### 10.2 When to Use Naive Bayes

- Text classification (spam detection, sentiment analysis)
- Large datasets (fast training and prediction)
- Categorical features
- Need for probability estimates
- When independence assumption is reasonable

### 10.3 Hyperparameter Tuning

```python
from sklearn.model_selection import GridSearchCV

# k-NN Grid Search
knn_params = {
    'n_neighbors': [3, 5, 7, 9, 11],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}

knn_grid = GridSearchCV(KNeighborsClassifier(), knn_params, cv=5, scoring='accuracy')
knn_grid.fit(X_train, y_train)
print(f"Best k-NN params: {knn_grid.best_params_}")

# Naive Bayes Grid Search
nb_params = {
    'alpha': [0.1, 0.5, 1.0, 2.0, 10.0]
}

nb_grid = GridSearchCV(MultinomialNB(), nb_params, cv=5, scoring='accuracy')
nb_grid.fit(X_train, y_train)
print(f"Best NB params: {nb_grid.best_params_}")
```

---

## 11. Exercises

### Exercise 1: k-NN Classification
Load the wine dataset and find the optimal k value using cross-validation.

```python
from sklearn.datasets import load_wine

# Your code here
```

### Exercise 2: k-NN Regression
Create a sine wave dataset and compare k-NN regression with different k values.

```python
# Your code here
```

### Exercise 3: Text Classification
Use Naive Bayes to classify movie reviews as positive or negative.

```python
from sklearn.datasets import load_files

# Your code here
```

### Exercise 4: Feature Scaling Impact
Compare k-NN performance on the breast cancer dataset with and without feature scaling.

```python
from sklearn.datasets import load_breast_cancer

# Your code here
```

---

## Summary

| Topic | Key Points |
|-------|------------|
| **k-NN Principle** | Classification/regression based on k nearest neighbors |
| **k Selection** | Small k = complex boundary, large k = smooth boundary |
| **Distance Metrics** | Euclidean (most common), Manhattan, Cosine (text) |
| **Feature Scaling** | CRITICAL for k-NN (distance-based algorithm) |
| **k-NN Limitations** | Slow prediction, high memory, curse of dimensionality |
| **Naive Bayes** | Probabilistic classifier based on Bayes' theorem |
| **Independence Assumption** | Assumes features are independent (naive but works well) |
| **NB Types** | Gaussian (continuous), Multinomial (counts), Bernoulli (binary) |
| **Text Classification** | Naive Bayes excels at text data (spam, sentiment) |
| **Alpha (Smoothing)** | Prevents zero probabilities for unseen words |

**Key Takeaway**: k-NN is simple and works well for small, low-dimensional datasets. Naive Bayes is fast and excellent for text classification despite its naive independence assumption.
