# Anomaly Detection

## Overview

Anomaly detection identifies data points, events, or patterns that deviate significantly from expected behavior. Unlike classification, anomaly detection often works with little or no labeled data for the anomalous class — making it essential for fraud detection, system monitoring, manufacturing quality control, and cybersecurity.

---

## 1. Types of Anomalies

### 1.1 Anomaly Taxonomy

```python
"""
Types of anomalies:

1. Point Anomalies: A single data point is anomalous
   Example: A credit card transaction of $50,000 when typical is $50-$500

2. Contextual Anomalies: Anomalous in a specific context, normal otherwise
   Example: Temperature of 35°C is normal in summer, anomalous in winter

3. Collective Anomalies: A group of data points is anomalous together
   Example: A sequence of rapid small transactions (card testing pattern)

Approaches:

┌──────────────────────┬──────────────────────────────────────┐
│ Supervised           │ Labeled normal + anomaly data        │
│                      │ → Classification (see L17)           │
├──────────────────────┼──────────────────────────────────────┤
│ Semi-supervised      │ Labeled normal data only             │
│                      │ → One-Class SVM, Autoencoders        │
├──────────────────────┼──────────────────────────────────────┤
│ Unsupervised         │ No labels at all                     │
│                      │ → Isolation Forest, LOF, DBSCAN      │
└──────────────────────┴──────────────────────────────────────┘
"""
```

---

## 2. Statistical Methods

### 2.1 Z-Score and IQR

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(42)

# Generate data with outliers
normal_data = np.random.normal(50, 10, 1000)
outliers = np.array([120, 130, -20, -30, 150])
data = np.concatenate([normal_data, outliers])

df = pd.DataFrame({'value': data})

# Method 1: Z-Score
df['z_score'] = (df['value'] - df['value'].mean()) / df['value'].std()
df['is_anomaly_zscore'] = df['z_score'].abs() > 3  # |z| > 3

# Method 2: IQR (Interquartile Range)
Q1 = df['value'].quantile(0.25)
Q3 = df['value'].quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR
df['is_anomaly_iqr'] = (df['value'] < lower) | (df['value'] > upper)

# Method 3: Modified Z-Score (robust, uses median)
median = df['value'].median()
mad = np.median(np.abs(df['value'] - median))  # Median Absolute Deviation
df['modified_z'] = 0.6745 * (df['value'] - median) / mad
df['is_anomaly_modified_z'] = df['modified_z'].abs() > 3.5

print(f"Z-Score anomalies:    {df['is_anomaly_zscore'].sum()}")
print(f"IQR anomalies:        {df['is_anomaly_iqr'].sum()}")
print(f"Modified Z anomalies: {df['is_anomaly_modified_z'].sum()}")

# Visualization
fig, axes = plt.subplots(1, 3, figsize=(18, 4))
for ax, method in zip(axes, ['is_anomaly_zscore', 'is_anomaly_iqr', 'is_anomaly_modified_z']):
    colors = df[method].map({True: 'red', False: 'blue'})
    ax.scatter(range(len(df)), df['value'], c=colors, alpha=0.5, s=10)
    ax.set_title(method.replace('is_anomaly_', '').replace('_', ' ').title())
    ax.set_xlabel('Index')
    ax.set_ylabel('Value')
plt.tight_layout()
plt.savefig('statistical_anomalies.png', dpi=150)
plt.show()
```

### 2.2 Mahalanobis Distance (Multivariate)

```python
from scipy.spatial.distance import mahalanobis
from scipy.stats import chi2

np.random.seed(42)

# Multivariate normal data with correlation
mean = [50, 100]
cov = [[100, 80], [80, 150]]
normal_2d = np.random.multivariate_normal(mean, cov, 500)
outliers_2d = np.array([[100, 30], [10, 200], [120, 250]])
X = np.vstack([normal_2d, outliers_2d])

# Mahalanobis distance
cov_inv = np.linalg.inv(np.cov(X.T))
center = X.mean(axis=0)
mahal_distances = np.array([mahalanobis(x, center, cov_inv) for x in X])

# Threshold: chi2 distribution with df=2, p=0.001
threshold = np.sqrt(chi2.ppf(0.999, df=2))
is_anomaly = mahal_distances > threshold

print(f"Mahalanobis threshold (p=0.001): {threshold:.2f}")
print(f"Anomalies detected: {is_anomaly.sum()}")

# Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Euclidean (misses correlated outliers)
eucl_dist = np.sqrt(((X - center)**2).sum(axis=1))
axes[0].scatter(X[:, 0], X[:, 1], c=eucl_dist, cmap='coolwarm', alpha=0.7, s=20)
axes[0].set_title('Euclidean Distance')
axes[0].set_xlabel('Feature 1')
axes[0].set_ylabel('Feature 2')

# Mahalanobis (accounts for correlation)
colors = ['red' if a else 'blue' for a in is_anomaly]
axes[1].scatter(X[:, 0], X[:, 1], c=colors, alpha=0.7, s=20)
axes[1].set_title(f'Mahalanobis (threshold={threshold:.1f})')
axes[1].set_xlabel('Feature 1')
axes[1].set_ylabel('Feature 2')

plt.tight_layout()
plt.savefig('mahalanobis.png', dpi=150)
plt.show()
```

---

## 3. Isolation Forest

### 3.1 Algorithm and Implementation

```python
"""
Isolation Forest (Liu et al., 2008):
  - Key insight: Anomalies are FEW and DIFFERENT → easy to isolate
  - Build random binary trees by randomly selecting features and split values
  - Anomalies require fewer splits to isolate (shorter path length)
  - Normal points require more splits (longer path length)

How it works:
  1. Randomly select a feature
  2. Randomly select a split value between min and max
  3. Split data into left/right
  4. Repeat until each point is isolated
  5. Anomaly score = average path length across all trees
  6. Short path → anomaly, Long path → normal
"""

from sklearn.ensemble import IsolationForest
from sklearn.datasets import make_blobs

# Generate data with anomalies
X_normal, _ = make_blobs(n_samples=1000, centers=2, cluster_std=1.0, random_state=42)
X_anomaly = np.random.uniform(-8, 8, (50, 2))  # 50 random outliers
X_all = np.vstack([X_normal, X_anomaly])
y_true = np.concatenate([np.ones(1000), -np.ones(50)])  # 1=normal, -1=anomaly

# Train Isolation Forest
iso_forest = IsolationForest(
    n_estimators=200,
    contamination=0.05,  # Expected proportion of anomalies
    random_state=42,
    n_jobs=-1,
)
y_pred = iso_forest.fit_predict(X_all)  # 1=normal, -1=anomaly
scores = iso_forest.decision_function(X_all)  # Lower = more anomalous

print(f"Detected anomalies: {(y_pred == -1).sum()}")
print(f"Score range: [{scores.min():.3f}, {scores.max():.3f}]")

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Predicted
colors_pred = ['red' if p == -1 else 'blue' for p in y_pred]
axes[0].scatter(X_all[:, 0], X_all[:, 1], c=colors_pred, alpha=0.5, s=15)
axes[0].set_title(f'Isolation Forest Predictions ({(y_pred==-1).sum()} anomalies)')

# Anomaly scores heatmap
xx, yy = np.meshgrid(np.linspace(-10, 10, 100), np.linspace(-10, 10, 100))
Z = iso_forest.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
axes[1].contourf(xx, yy, Z, levels=20, cmap='RdBu')
axes[1].scatter(X_all[:, 0], X_all[:, 1], c=colors_pred, alpha=0.5, s=15, edgecolors='k', linewidths=0.3)
axes[1].set_title('Anomaly Score Contour (red=anomalous)')

plt.tight_layout()
plt.savefig('isolation_forest.png', dpi=150)
plt.show()
```

### 3.2 Tuning Contamination

```python
from sklearn.metrics import precision_score, recall_score, f1_score

# Try different contamination values
contaminations = [0.01, 0.02, 0.05, 0.1, 0.15, 0.2]
results = []

for cont in contaminations:
    iso = IsolationForest(n_estimators=200, contamination=cont, random_state=42)
    y_pred = iso.fit_predict(X_all)

    # Convert to binary: anomaly=1, normal=0
    y_pred_binary = (y_pred == -1).astype(int)
    y_true_binary = (y_true == -1).astype(int)

    results.append({
        'contamination': cont,
        'n_detected': (y_pred == -1).sum(),
        'precision': precision_score(y_true_binary, y_pred_binary),
        'recall': recall_score(y_true_binary, y_pred_binary),
        'f1': f1_score(y_true_binary, y_pred_binary),
    })

results_df = pd.DataFrame(results)
print("Contamination tuning:")
print(results_df.round(3).to_string(index=False))
```

---

## 4. Local Outlier Factor (LOF)

### 4.1 Density-Based Anomaly Detection

```python
"""
LOF (Breunig et al., 2000):
  - Measures local density deviation of a point relative to its neighbors
  - Points in low-density regions relative to neighbors → outliers
  - Key advantage: handles clusters of different densities

How it works:
  1. For each point, find k nearest neighbors
  2. Compute local reachability density (LRD)
  3. Compare LRD to neighbors' LRDs → LOF score
  4. LOF ≈ 1: similar density to neighbors (normal)
  5. LOF >> 1: much lower density than neighbors (anomaly)
"""

from sklearn.neighbors import LocalOutlierFactor

# LOF
lof = LocalOutlierFactor(
    n_neighbors=20,
    contamination=0.05,
    novelty=False,  # False for outlier detection (unsupervised)
)
y_pred_lof = lof.fit_predict(X_all)
lof_scores = -lof.negative_outlier_factor_  # Negate for intuitive interpretation

print(f"LOF detected anomalies: {(y_pred_lof == -1).sum()}")
print(f"LOF score range: [{lof_scores.min():.3f}, {lof_scores.max():.3f}]")

# Compare with Isolation Forest
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

iso_colors = ['red' if p == -1 else 'blue' for p in iso_forest.fit_predict(X_all)]
axes[0].scatter(X_all[:, 0], X_all[:, 1], c=iso_colors, alpha=0.5, s=15)
axes[0].set_title('Isolation Forest')

lof_colors = ['red' if p == -1 else 'blue' for p in y_pred_lof]
axes[1].scatter(X_all[:, 0], X_all[:, 1], c=lof_colors, alpha=0.5, s=15)
axes[1].set_title('Local Outlier Factor')

plt.tight_layout()
plt.savefig('iso_vs_lof.png', dpi=150)
plt.show()
```

### 4.2 LOF for Novelty Detection

```python
# Novelty detection: train on clean data, detect new anomalies
lof_novelty = LocalOutlierFactor(n_neighbors=20, novelty=True)
lof_novelty.fit(X_normal)  # Train on normal data only

# Test on mix of normal and anomalous
X_new_normal = np.random.multivariate_normal([0, 0], [[1, 0], [0, 1]], 20) + X_normal.mean(axis=0)
X_new_anomaly = np.random.uniform(-8, 8, (10, 2))
X_new = np.vstack([X_new_normal, X_new_anomaly])
y_new_true = np.concatenate([np.ones(20), -np.ones(10)])

y_new_pred = lof_novelty.predict(X_new)
print(f"New normals detected as anomaly: {(y_new_pred[:20] == -1).sum()} / 20")
print(f"New anomalies detected: {(y_new_pred[20:] == -1).sum()} / 10")
```

---

## 5. One-Class SVM

### 5.1 Support Vector Approach

```python
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler

# Scale features (important for SVM)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_all)

# One-Class SVM
ocsvm = OneClassSVM(
    kernel='rbf',
    gamma='scale',
    nu=0.05,  # Upper bound on fraction of anomalies
)
y_pred_svm = ocsvm.fit_predict(X_scaled)

print(f"One-Class SVM anomalies: {(y_pred_svm == -1).sum()}")

# Decision boundary visualization
xx, yy = np.meshgrid(
    np.linspace(X_scaled[:, 0].min() - 1, X_scaled[:, 0].max() + 1, 100),
    np.linspace(X_scaled[:, 1].min() - 1, X_scaled[:, 1].max() + 1, 100),
)
Z = ocsvm.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

fig, ax = plt.subplots(figsize=(8, 6))
ax.contourf(xx, yy, Z, levels=20, cmap='RdBu')
ax.contour(xx, yy, Z, levels=[0], linewidths=2, colors='black')
svm_colors = ['red' if p == -1 else 'blue' for p in y_pred_svm]
ax.scatter(X_scaled[:, 0], X_scaled[:, 1], c=svm_colors, alpha=0.5, s=15, edgecolors='k', linewidths=0.3)
ax.set_title('One-Class SVM Decision Boundary')
plt.tight_layout()
plt.savefig('ocsvm.png', dpi=150)
plt.show()
```

---

## 6. Ensemble and PyOD

### 6.1 Combining Multiple Detectors

```python
"""
Ensemble approach: Combine multiple anomaly detectors for robustness.

# pip install pyod

from pyod.models.iforest import IForest
from pyod.models.lof import LOF
from pyod.models.ocsvm import OCSVM
from pyod.models.knn import KNN
from pyod.models.combination import average, maximization

# Initialize detectors
detectors = {
    'IForest': IForest(contamination=0.05, random_state=42),
    'LOF': LOF(contamination=0.05),
    'OCSVM': OCSVM(contamination=0.05),
    'KNN': KNN(contamination=0.05),
}

# Fit and collect scores
all_scores = []
for name, detector in detectors.items():
    detector.fit(X_all)
    scores = detector.decision_scores_
    # Normalize scores to [0, 1]
    scores_norm = (scores - scores.min()) / (scores.max() - scores.min())
    all_scores.append(scores_norm)

all_scores = np.array(all_scores)

# Combine: average of normalized scores
ensemble_scores = all_scores.mean(axis=0)
threshold = np.percentile(ensemble_scores, 95)  # Top 5% are anomalies
ensemble_pred = (ensemble_scores > threshold).astype(int)

print(f"Ensemble detected: {ensemble_pred.sum()} anomalies")
"""

# Manual ensemble without pyod
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

iso = IsolationForest(n_estimators=200, contamination=0.05, random_state=42)
iso.fit(X_all)
iso_scores = -iso.decision_function(X_all)  # Higher = more anomalous

lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05)
lof.fit_predict(X_all)
lof_scores = -lof.negative_outlier_factor_

# Normalize and combine
def normalize(scores):
    return (scores - scores.min()) / (scores.max() - scores.min())

combined = (normalize(iso_scores) + normalize(lof_scores)) / 2
threshold = np.percentile(combined, 95)
ensemble_pred = (combined > threshold).astype(int)
y_true_binary = (y_true == -1).astype(int)

print(f"Ensemble anomalies: {ensemble_pred.sum()}")
print(f"Precision: {precision_score(y_true_binary, ensemble_pred):.3f}")
print(f"Recall:    {recall_score(y_true_binary, ensemble_pred):.3f}")
print(f"F1:        {f1_score(y_true_binary, ensemble_pred):.3f}")
```

---

## 7. Time Series Anomaly Detection

### 7.1 Statistical Process Control

```python
np.random.seed(42)

# Generate time series with anomalies
n = 500
t = np.arange(n)
normal_ts = 50 + 5 * np.sin(2 * np.pi * t / 100) + np.random.normal(0, 2, n)

# Inject anomalies
anomaly_indices = [100, 200, 300, 400]
for idx in anomaly_indices:
    normal_ts[idx] += np.random.choice([-1, 1]) * np.random.uniform(15, 25)

ts_df = pd.DataFrame({'value': normal_ts})

# Method 1: Rolling Z-Score
window = 30
ts_df['rolling_mean'] = ts_df['value'].rolling(window, center=False).mean()
ts_df['rolling_std'] = ts_df['value'].rolling(window, center=False).std()
ts_df['rolling_z'] = (ts_df['value'] - ts_df['rolling_mean']) / ts_df['rolling_std']
ts_df['anomaly_z'] = ts_df['rolling_z'].abs() > 3

# Method 2: Exponentially Weighted Moving Average (EWMA)
span = 20
ts_df['ewma'] = ts_df['value'].ewm(span=span).mean()
ts_df['ewma_std'] = ts_df['value'].ewm(span=span).std()
ts_df['ewma_z'] = (ts_df['value'] - ts_df['ewma']) / ts_df['ewma_std']
ts_df['anomaly_ewma'] = ts_df['ewma_z'].abs() > 3

# Plot
fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

axes[0].plot(ts_df['value'], 'b-', alpha=0.7, linewidth=0.8)
axes[0].plot(ts_df['rolling_mean'], 'g-', label='Rolling Mean')
upper = ts_df['rolling_mean'] + 3 * ts_df['rolling_std']
lower = ts_df['rolling_mean'] - 3 * ts_df['rolling_std']
axes[0].fill_between(range(n), upper, lower, alpha=0.2, color='green')
anomalies_z = ts_df[ts_df['anomaly_z']]
axes[0].scatter(anomalies_z.index, anomalies_z['value'], c='red', s=50, zorder=5, label='Anomalies')
axes[0].set_title('Rolling Z-Score Anomaly Detection')
axes[0].legend()

axes[1].plot(ts_df['value'], 'b-', alpha=0.7, linewidth=0.8)
axes[1].plot(ts_df['ewma'], 'orange', label='EWMA')
upper_ewma = ts_df['ewma'] + 3 * ts_df['ewma_std']
lower_ewma = ts_df['ewma'] - 3 * ts_df['ewma_std']
axes[1].fill_between(range(n), upper_ewma, lower_ewma, alpha=0.2, color='orange')
anomalies_ewma = ts_df[ts_df['anomaly_ewma']]
axes[1].scatter(anomalies_ewma.index, anomalies_ewma['value'], c='red', s=50, zorder=5, label='Anomalies')
axes[1].set_title('EWMA Anomaly Detection')
axes[1].legend()

plt.tight_layout()
plt.savefig('ts_anomaly.png', dpi=150)
plt.show()
```

### 7.2 Seasonal Decomposition for Anomalies

```python
"""
For seasonal time series:
1. Decompose into trend + seasonal + residual (STL decomposition)
2. Apply anomaly detection to the residuals
3. Large residuals = anomalies

from statsmodels.tsa.seasonal import STL

stl = STL(ts_df['value'], period=100, robust=True)
result = stl.fit()
residuals = result.resid

# Detect anomalies in residuals
z_scores = (residuals - residuals.mean()) / residuals.std()
anomalies = z_scores.abs() > 3
"""
print("STL decomposition removes trend and seasonality,")
print("making anomaly detection more sensitive to true deviations.")
```

---

## 8. Evaluation Without Labels

### 8.1 When You Don't Have Ground Truth

```python
"""
In practice, labeled anomalies are rare. Evaluation strategies:

1. Visual Inspection:
   - Plot detected anomalies and check with domain experts
   - Most practical for initial model development

2. Stability Analysis:
   - Run detector with different parameters
   - Points consistently flagged = likely true anomalies
   - Points flagged only with extreme settings = borderline

3. Internal Metrics:
   - Silhouette-like scores: How different are detected anomalies from normal?
   - Anomaly score distribution: Should be bimodal (normal peak + anomaly tail)

4. Proxy Metrics:
   - If anomalies correlate with known events (system outages, fraud reports)
   - Temporal correlation with external signals

5. A/B Testing in Production:
   - Measure business impact of acting on detected anomalies
   - True positive → prevented fraud, caught defect
   - False positive → wasted investigation time
"""

# Stability analysis example
from sklearn.ensemble import IsolationForest

# Run with multiple contamination values
stability_matrix = np.zeros((len(X_all), 5))
for i, cont in enumerate([0.01, 0.03, 0.05, 0.08, 0.1]):
    iso = IsolationForest(contamination=cont, random_state=42)
    preds = iso.fit_predict(X_all)
    stability_matrix[:, i] = (preds == -1).astype(int)

# Points flagged in all settings = high confidence anomalies
stability_score = stability_matrix.mean(axis=1)
high_conf_anomalies = stability_score >= 0.8  # Flagged in 80%+ of runs

print(f"High-confidence anomalies (flagged in 4+/5 runs): {high_conf_anomalies.sum()}")
print(f"Medium-confidence (flagged in 2-3/5): {((stability_score >= 0.4) & (stability_score < 0.8)).sum()}")
print(f"Low-confidence (flagged in 1/5): {((stability_score > 0) & (stability_score < 0.4)).sum()}")
```

---

## 9. Method Selection Guide

### 9.1 Choosing the Right Method

| Method | Data Type | Labeled Data | Scalability | Interpretability |
|--------|-----------|-------------|-------------|-----------------|
| Z-Score / IQR | Univariate | Not needed | Excellent | Very high |
| Mahalanobis | Multivariate | Not needed | Good | High |
| **Isolation Forest** | Tabular | Not needed | Excellent | Medium |
| **LOF** | Tabular | Not needed | Medium | Medium |
| One-Class SVM | Tabular | Normal only | Poor (large data) | Low |
| DBSCAN | Tabular | Not needed | Good | Medium |
| Autoencoder | Any (high-D) | Normal only | Good | Low |
| Rolling Z-Score | Time series | Not needed | Excellent | Very high |
| STL + Residual | Seasonal TS | Not needed | Good | High |

### 9.2 Decision Framework

```python
"""
              ┌── Univariate?
              │   └── Yes → Z-Score or IQR (start here)
              │   └── No ─┐
              │            │
              │   ┌── High dimensional (>50 features)?
              │   │   └── Yes → Autoencoder or Isolation Forest
              │   │   └── No ─┐
              │   │            │
              │   │   ┌── Need real-time / streaming?
              │   │   │   └── Yes → Rolling Z-Score or EWMA
              │   │   │   └── No ─┐
              │   │   │            │
              │   │   │   ┌── Have normal-only training data?
              │   │   │   │   └── Yes → One-Class SVM or LOF (novelty mode)
              │   │   │   │   └── No → Isolation Forest (most robust default)
"""
```

---

## 10. Practice Problems

### Exercise 1: Network Intrusion Detection

```python
"""
1. Generate network traffic data with these features:
   - bytes_sent, bytes_received, duration, n_packets, protocol_type
   - Normal traffic: moderate values, correlated bytes/packets
   - Anomalies: DDoS (high packets, low bytes), exfiltration (high bytes_sent)
2. Apply Isolation Forest, LOF, and One-Class SVM.
3. Compare detection rates.
4. Build an ensemble of all three methods.
5. Use stability analysis to identify high-confidence anomalies.
"""
```

### Exercise 2: Manufacturing Quality Control

```python
"""
1. Generate sensor data (temperature, pressure, vibration) for 1000 products.
2. Normal: correlated features, product is good.
3. Anomaly types:
   - Overheating: high temperature, normal pressure
   - Pressure spike: high pressure, normal temperature
   - Combined: both high (machine failure)
4. Detect each anomaly type using Mahalanobis distance.
5. Compute precision and recall for each anomaly type.
6. How well does Isolation Forest handle mixed anomaly types?
"""
```

### Exercise 3: Time Series Monitoring

```python
"""
1. Generate 1 year of daily server response time data:
   - Normal: mean=200ms, weekly pattern, slight upward trend
   - Anomalies: 5 sudden spikes (server issues), 2 gradual degradations
2. Detect point anomalies using rolling Z-score.
3. Detect gradual degradation using CUSUM or change-point detection.
4. Compare: Which method catches each anomaly type?
5. Design an alerting system with configurable sensitivity.
"""
```

---

## 11. Summary

### Key Takeaways

| Concept | Description |
|---------|-------------|
| **Point/Contextual/Collective** | Three types of anomalies require different approaches |
| **Z-Score / IQR** | Simple, fast — start here for univariate data |
| **Mahalanobis** | Multivariate distance accounting for correlation |
| **Isolation Forest** | Best general-purpose method, scales well |
| **LOF** | Density-based, handles varying cluster densities |
| **One-Class SVM** | Semi-supervised (train on normal data only) |
| **Ensemble** | Combine methods for robustness |
| **Time Series** | Rolling statistics, STL decomposition for seasonal data |
| **Evaluation** | Stability analysis when labels are unavailable |

### Best Practices

1. **Start with Isolation Forest** — robust default for most tabular data
2. **Contamination parameter** is critical — estimate from domain knowledge
3. **Ensemble methods** reduce false positives
4. **Stability analysis** provides confidence levels without labels
5. **Preprocess carefully** — scale features for SVM and distance-based methods
6. **Domain expertise is essential** — validated anomalies > automated scores

### Next Steps

- **L17**: Imbalanced Data — when you have some labeled anomalies, treat as classification
- **Deep_Learning**: Autoencoders for high-dimensional anomaly detection
- **Data_Science L23**: Nonparametric Statistics — bootstrap and permutation tests for anomalies
