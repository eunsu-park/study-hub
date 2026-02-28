"""
Anomaly Detection - Exercise Solutions
========================================
Lesson 20: Anomaly Detection

Exercises cover:
  1. Network intrusion detection with ensemble of methods
  2. Manufacturing quality control with Mahalanobis distance
  3. Time series monitoring with rolling Z-score and CUSUM
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score


# ============================================================
# Exercise 1: Network Intrusion Detection
# Apply multiple anomaly detection methods and build an ensemble.
# ============================================================
def exercise_1_network_intrusion():
    """Network intrusion detection with an ensemble of 3 methods.

    Ensemble anomaly detection is more robust because different methods
    capture different aspects of "abnormal":
    - Isolation Forest: isolates anomalies by random splits (global)
    - LOF: compares local density to neighbors (local)
    - One-Class SVM: learns a boundary around normal data (kernel-based)

    High-confidence anomalies are flagged by all methods (consensus).
    """
    print("=" * 60)
    print("Exercise 1: Network Intrusion Detection")
    print("=" * 60)

    np.random.seed(42)

    # Generate normal network traffic (correlated bytes/packets)
    n_normal = 2000
    n_packets_normal = np.random.poisson(100, n_normal).astype(float)
    bytes_sent_normal = n_packets_normal * np.random.uniform(40, 60, n_normal)
    bytes_recv_normal = n_packets_normal * np.random.uniform(30, 50, n_normal)
    duration_normal = np.random.exponential(5, n_normal)

    # DDoS anomalies: extremely high packets, low bytes per packet
    n_ddos = 30
    n_packets_ddos = np.random.poisson(5000, n_ddos).astype(float)
    bytes_sent_ddos = n_packets_ddos * np.random.uniform(1, 5, n_ddos)
    bytes_recv_ddos = np.random.uniform(10, 50, n_ddos)
    duration_ddos = np.random.uniform(0.1, 1, n_ddos)

    # Exfiltration anomalies: very high bytes_sent, normal packets
    n_exfil = 20
    n_packets_exfil = np.random.poisson(100, n_exfil).astype(float)
    bytes_sent_exfil = np.random.uniform(50000, 200000, n_exfil)
    bytes_recv_exfil = np.random.uniform(100, 500, n_exfil)
    duration_exfil = np.random.uniform(10, 60, n_exfil)

    # Combine
    X = np.column_stack([
        np.concatenate([bytes_sent_normal, bytes_sent_ddos, bytes_sent_exfil]),
        np.concatenate([bytes_recv_normal, bytes_recv_ddos, bytes_recv_exfil]),
        np.concatenate([duration_normal, duration_ddos, duration_exfil]),
        np.concatenate([n_packets_normal, n_packets_ddos, n_packets_exfil]),
    ])
    y_true = np.array([0]*n_normal + [1]*n_ddos + [1]*n_exfil)
    feature_names = ["bytes_sent", "bytes_recv", "duration", "n_packets"]

    contamination = (n_ddos + n_exfil) / len(X)
    print(f"Total samples: {len(X)}")
    print(f"Anomalies: {n_ddos + n_exfil} ({contamination:.2%})")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Apply 3 methods
    methods = {
        "Isolation Forest": IsolationForest(
            contamination=contamination, random_state=42, n_jobs=-1
        ),
        "LOF": LocalOutlierFactor(
            contamination=contamination, n_neighbors=20, novelty=False
        ),
        "One-Class SVM": OneClassSVM(
            nu=contamination, kernel="rbf", gamma="scale"
        ),
    }

    predictions = {}
    print(f"\n{'Method':<20} {'Precision':>10} {'Recall':>8} {'F1':>8}")
    print("-" * 50)

    for name, model in methods.items():
        if name == "LOF":
            y_pred_raw = model.fit_predict(X_scaled)
        else:
            model.fit(X_scaled)
            y_pred_raw = model.predict(X_scaled)

        # Convert: -1 = anomaly -> 1, 1 = normal -> 0
        y_pred = (y_pred_raw == -1).astype(int)
        predictions[name] = y_pred

        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        print(f"{name:<20} {prec:>10.4f} {rec:>8.4f} {f1:>8.4f}")

    # Ensemble: flag as anomaly if majority (>=2) methods agree
    ensemble = sum(predictions.values())
    y_ensemble = (ensemble >= 2).astype(int)

    prec = precision_score(y_true, y_ensemble, zero_division=0)
    rec = recall_score(y_true, y_ensemble)
    f1 = f1_score(y_true, y_ensemble)
    print(f"{'Ensemble (>=2/3)':<20} {prec:>10.4f} {rec:>8.4f} {f1:>8.4f}")

    # High-confidence: all 3 agree
    y_high_conf = (ensemble == 3).astype(int)
    print(f"\nHigh-confidence anomalies (all 3 agree): {y_high_conf.sum()}")
    print(f"  True anomalies among high-conf: "
          f"{np.sum(y_high_conf & y_true)}/{y_high_conf.sum()}")


# ============================================================
# Exercise 2: Manufacturing Quality Control
# Detect anomalies using Mahalanobis distance.
# ============================================================
def exercise_2_manufacturing_qc():
    """Manufacturing QC using Mahalanobis distance.

    Mahalanobis distance accounts for feature correlations -- unlike
    Euclidean distance, it recognizes that a point is unusual even if
    each individual feature is within range, as long as the feature
    *combination* is unlikely given the correlation structure.

    Example: temperature=80 is normal, pressure=200 is normal, but
    temperature=80 AND pressure=200 together might be abnormal if
    they are normally positively correlated.
    """
    print("\n" + "=" * 60)
    print("Exercise 2: Manufacturing Quality Control")
    print("=" * 60)

    np.random.seed(42)

    # Normal products: correlated temperature-pressure-vibration
    n_normal = 1000
    mean_normal = [50, 100, 5]  # temp, pressure, vibration
    cov_normal = [[9, 6, 0.5],
                  [6, 16, 1],
                  [0.5, 1, 0.5]]
    X_normal = np.random.multivariate_normal(mean_normal, cov_normal, n_normal)

    # Anomaly type 1: Overheating (high temp, normal pressure)
    n_overheat = 20
    X_overheat = np.column_stack([
        np.random.normal(75, 3, n_overheat),   # high temp
        np.random.normal(100, 4, n_overheat),   # normal pressure
        np.random.normal(8, 1, n_overheat),      # elevated vibration
    ])

    # Anomaly type 2: Pressure spike
    n_pressure = 15
    X_pressure = np.column_stack([
        np.random.normal(50, 3, n_pressure),     # normal temp
        np.random.normal(145, 5, n_pressure),    # high pressure
        np.random.normal(7, 1, n_pressure),      # elevated vibration
    ])

    # Anomaly type 3: Combined failure
    n_combined = 10
    X_combined = np.column_stack([
        np.random.normal(72, 3, n_combined),     # high temp
        np.random.normal(140, 5, n_combined),    # high pressure
        np.random.normal(12, 2, n_combined),     # high vibration
    ])

    X = np.vstack([X_normal, X_overheat, X_pressure, X_combined])
    y_true = np.array([0]*n_normal + [1]*n_overheat + [2]*n_pressure + [3]*n_combined)
    y_binary = (y_true > 0).astype(int)

    anomaly_names = {0: "Normal", 1: "Overheat", 2: "Pressure Spike", 3: "Combined"}

    # Mahalanobis distance (using only normal data statistics)
    mean = X_normal.mean(axis=0)
    cov = np.cov(X_normal, rowvar=False)
    cov_inv = np.linalg.inv(cov)

    def mahalanobis(x, mean, cov_inv):
        diff = x - mean
        return np.sqrt(diff @ cov_inv @ diff)

    distances = np.array([mahalanobis(x, mean, cov_inv) for x in X])

    # Threshold: chi-squared distribution with 3 DOF, 99.5th percentile
    threshold = np.sqrt(12.84)  # chi2(3, 0.995) ~ 12.84
    y_pred = (distances > threshold).astype(int)

    print(f"Mahalanobis threshold: {threshold:.2f}")
    print(f"\nOverall: Precision={precision_score(y_binary, y_pred):.4f}, "
          f"Recall={recall_score(y_binary, y_pred):.4f}")

    # Per anomaly type
    print(f"\nDetection rate by anomaly type:")
    for atype in [1, 2, 3]:
        mask = y_true == atype
        detected = y_pred[mask].sum()
        total = mask.sum()
        print(f"  {anomaly_names[atype]}: {detected}/{total} detected "
              f"({detected/total*100:.0f}%)")

    # Compare with Isolation Forest
    iso = IsolationForest(contamination=0.05, random_state=42)
    iso_pred = (iso.fit_predict(X) == -1).astype(int)
    print(f"\nIsolation Forest comparison:")
    for atype in [1, 2, 3]:
        mask = y_true == atype
        detected = iso_pred[mask].sum()
        total = mask.sum()
        print(f"  {anomaly_names[atype]}: {detected}/{total} detected "
              f"({detected/total*100:.0f}%)")


# ============================================================
# Exercise 3: Time Series Monitoring
# Rolling Z-score for spikes, CUSUM for gradual degradation.
# ============================================================
def exercise_3_time_series_monitoring():
    """Time series anomaly detection: Z-score + CUSUM.

    Two complementary methods for different anomaly types:
    - Rolling Z-score: detects sudden spikes/drops (point anomalies)
      by measuring how many standard deviations a point is from
      the recent rolling mean.
    - CUSUM (Cumulative Sum): detects gradual shifts in the mean
      by accumulating positive deviations. A drift that is too small
      to trigger Z-score but persists over many timesteps will
      accumulate in CUSUM and eventually trigger an alert.
    """
    print("\n" + "=" * 60)
    print("Exercise 3: Time Series Monitoring")
    print("=" * 60)

    np.random.seed(42)

    # Generate 1 year of daily server response time
    n_days = 365
    t = np.arange(n_days)

    # Normal pattern: slight weekly seasonality + trend
    baseline = 200 + 0.05 * t  # slight upward trend
    weekly = 10 * np.sin(2 * np.pi * t / 7)
    noise = np.random.randn(n_days) * 8

    response_time = baseline + weekly + noise

    # Inject anomalies
    # 5 sudden spikes (server issues)
    spike_days = [45, 120, 200, 280, 330]
    for d in spike_days:
        response_time[d] += np.random.uniform(80, 150)

    # 2 gradual degradations (memory leak pattern)
    # Degradation 1: days 150-170
    response_time[150:170] += np.linspace(0, 40, 20)
    # Degradation 2: days 250-275
    response_time[250:275] += np.linspace(0, 35, 25)

    # Mark true anomalies
    y_true_spike = np.zeros(n_days)
    for d in spike_days:
        y_true_spike[d] = 1

    y_true_degrade = np.zeros(n_days)
    y_true_degrade[150:170] = 1
    y_true_degrade[250:275] = 1

    # --- Method 1: Rolling Z-Score (for spikes) ---
    window = 14
    rolling_mean = np.convolve(response_time, np.ones(window)/window, mode="same")
    rolling_std = np.array([
        np.std(response_time[max(0, i-window):i+1])
        for i in range(n_days)
    ])
    rolling_std = np.clip(rolling_std, 1, None)  # avoid division by zero

    z_scores = (response_time - rolling_mean) / rolling_std
    z_threshold = 3.0
    z_anomalies = (z_scores > z_threshold).astype(int)

    spike_detected = sum(z_anomalies[d] for d in spike_days)
    print(f"Rolling Z-Score (threshold={z_threshold}):")
    print(f"  Spikes detected: {spike_detected}/{len(spike_days)}")
    print(f"  Total alerts: {z_anomalies.sum()}")

    # --- Method 2: CUSUM (for gradual degradation) ---
    target = np.median(response_time[:30])  # baseline from first month
    slack = 5  # allowable deviation before accumulating
    cusum_pos = np.zeros(n_days)
    cusum_neg = np.zeros(n_days)

    for i in range(1, n_days):
        cusum_pos[i] = max(0, cusum_pos[i-1] + response_time[i] - target - slack)
        cusum_neg[i] = max(0, cusum_neg[i-1] - response_time[i] + target - slack)

    cusum_threshold = 100
    cusum_anomalies = ((cusum_pos > cusum_threshold) | (cusum_neg > cusum_threshold)).astype(int)

    # Check degradation detection
    degrade1_detected = cusum_anomalies[150:170].any()
    degrade2_detected = cusum_anomalies[250:275].any()
    print(f"\nCUSUM (threshold={cusum_threshold}, slack={slack}):")
    print(f"  Degradation 1 (days 150-170): {'Detected' if degrade1_detected else 'Missed'}")
    print(f"  Degradation 2 (days 250-275): {'Detected' if degrade2_detected else 'Missed'}")
    print(f"  Total anomaly days: {cusum_anomalies.sum()}")

    # Combined detection
    combined = (z_anomalies | cusum_anomalies).astype(int)
    print(f"\nCombined (Z-score OR CUSUM):")
    print(f"  Total anomaly days: {combined.sum()}")
    print(f"  Spike coverage:       {spike_detected}/{len(spike_days)}")
    print(f"  Degradation coverage: {int(degrade1_detected) + int(degrade2_detected)}/2")

    # Visualization
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    # Response time with anomalies
    axes[0].plot(response_time, "b-", alpha=0.6, label="Response time")
    axes[0].plot(rolling_mean, "orange", label="Rolling mean")
    spike_idx = np.where(z_anomalies)[0]
    axes[0].scatter(spike_idx, response_time[spike_idx], color="red", s=50,
                    zorder=5, label="Z-score alert")
    axes[0].set_ylabel("Response Time (ms)")
    axes[0].set_title("Server Response Time Monitoring")
    axes[0].legend(fontsize=8)

    # Z-scores
    axes[1].plot(z_scores, "g-", alpha=0.6)
    axes[1].axhline(y=z_threshold, color="red", linestyle="--",
                     label=f"Threshold={z_threshold}")
    axes[1].set_ylabel("Z-Score")
    axes[1].legend(fontsize=8)

    # CUSUM
    axes[2].plot(cusum_pos, "purple", label="CUSUM+")
    axes[2].axhline(y=cusum_threshold, color="red", linestyle="--",
                     label=f"Threshold={cusum_threshold}")
    axes[2].set_ylabel("CUSUM")
    axes[2].set_xlabel("Day")
    axes[2].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig("20_ex3_ts_monitoring.png", dpi=100)
    plt.close()
    print("\nPlot saved: 20_ex3_ts_monitoring.png")


if __name__ == "__main__":
    exercise_1_network_intrusion()
    exercise_2_manufacturing_qc()
    exercise_3_time_series_monitoring()
