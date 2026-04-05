"""
Exercise Solutions: Drift Detection and Monitoring
===========================================
Lesson 10 from MLOps topic.

Exercises
---------
1. Drift Detection — Implement KS test, PSI, and Wasserstein distance
   for detecting data drift between reference and production distributions.
2. Evidently Reports — Simulate generating an Evidently-style data quality
   and drift report with visualizations (text-based).
3. Alert System — Build an alerting system that monitors drift metrics
   and triggers actions based on configurable thresholds.
"""

import math
import random
import statistics
from datetime import datetime, timedelta


# ============================================================
# Exercise 1: Drift Detection
# ============================================================

def exercise_1_drift_detection():
    """Implement drift detection using KS test, PSI, and Wasserstein distance.

    Data drift occurs when production data distribution shifts from training data.
    Three complementary approaches:

    - KS Test: Measures maximum distance between CDFs (sensitive to any shift)
    - PSI (Population Stability Index): Measures distribution shift via binned comparison
    - Wasserstein Distance: "Earth mover's distance" — minimum cost to transform one
      distribution into another
    """

    # --- Generate reference (training) and production data ---
    random.seed(42)
    n_ref = 1000
    n_prod = 1000

    # Reference: standard normal
    reference = [random.gauss(0, 1) for _ in range(n_ref)]

    # Production with drift: mean shifted and variance changed
    production_no_drift = [random.gauss(0, 1) for _ in range(n_prod)]
    production_mild_drift = [random.gauss(0.3, 1.1) for _ in range(n_prod)]
    production_severe_drift = [random.gauss(1.0, 1.5) for _ in range(n_prod)]

    # --- KS Test (Kolmogorov-Smirnov) ---
    def ks_test(sample1, sample2):
        """Two-sample KS test: maximum absolute difference between CDFs.

        The KS statistic ranges from 0 (identical distributions) to 1 (completely
        different). The p-value estimates whether the difference is statistically
        significant.
        """
        combined = sorted(set(sample1 + sample2))
        n1, n2 = len(sample1), len(sample2)
        s1_sorted = sorted(sample1)
        s2_sorted = sorted(sample2)

        max_diff = 0
        for x in combined:
            # CDF of sample 1 at x
            cdf1 = sum(1 for v in s1_sorted if v <= x) / n1
            # CDF of sample 2 at x
            cdf2 = sum(1 for v in s2_sorted if v <= x) / n2
            diff = abs(cdf1 - cdf2)
            if diff > max_diff:
                max_diff = diff

        # Approximate p-value using asymptotic distribution
        n_eff = (n1 * n2) / (n1 + n2)
        lambda_val = (math.sqrt(n_eff) + 0.12 + 0.11 / math.sqrt(n_eff)) * max_diff
        # Kolmogorov distribution approximation
        p_value = 2 * math.exp(-2 * lambda_val * lambda_val)
        p_value = max(0, min(1, p_value))

        return {"statistic": round(max_diff, 4), "p_value": round(p_value, 6)}

    # --- PSI (Population Stability Index) ---
    def calculate_psi(reference, production, n_bins=10):
        """Population Stability Index.

        PSI = SUM( (actual% - expected%) * ln(actual% / expected%) )

        Thresholds:
          PSI < 0.1:  No significant shift
          0.1 <= PSI < 0.25: Moderate shift — investigate
          PSI >= 0.25: Significant shift — action required
        """
        # Create bins from reference data
        ref_sorted = sorted(reference)
        bin_edges = [ref_sorted[int(i * len(ref_sorted) / n_bins)]
                     for i in range(n_bins)]
        bin_edges = [float('-inf')] + bin_edges + [float('inf')]

        def get_bin_proportions(data, edges):
            proportions = []
            for i in range(len(edges) - 1):
                count = sum(1 for x in data if edges[i] <= x < edges[i + 1])
                proportions.append(max(count / len(data), 0.0001))  # Avoid log(0)
            return proportions

        ref_proportions = get_bin_proportions(reference, bin_edges)
        prod_proportions = get_bin_proportions(production, bin_edges)

        psi = sum(
            (p - r) * math.log(p / r)
            for p, r in zip(prod_proportions, ref_proportions)
        )

        return round(psi, 4)

    # --- Wasserstein Distance ---
    def wasserstein_distance(sample1, sample2):
        """1D Wasserstein distance (earth mover's distance).

        Computed as the integral of |CDF1 - CDF2|.
        Intuition: minimum cost to transform one pile of dirt (distribution)
        into another.
        """
        s1 = sorted(sample1)
        s2 = sorted(sample2)
        all_values = sorted(set(s1 + s2))

        n1, n2 = len(s1), len(s2)
        distance = 0
        prev_x = all_values[0]

        for x in all_values[1:]:
            cdf1 = sum(1 for v in s1 if v <= prev_x) / n1
            cdf2 = sum(1 for v in s2 if v <= prev_x) / n2
            distance += abs(cdf1 - cdf2) * (x - prev_x)
            prev_x = x

        return round(distance, 4)

    # --- Run all tests ---
    print("Drift Detection Methods")
    print("=" * 60)

    datasets = [
        ("No drift", production_no_drift),
        ("Mild drift (mean=0.3, std=1.1)", production_mild_drift),
        ("Severe drift (mean=1.0, std=1.5)", production_severe_drift),
    ]

    for name, prod_data in datasets:
        print(f"\n  Production: {name}")
        print(f"  {'-'*50}")

        # KS Test
        ks = ks_test(reference, prod_data)
        ks_interpretation = (
            "No drift" if ks["p_value"] > 0.05
            else "Drift detected (p < 0.05)"
        )

        # PSI
        psi = calculate_psi(reference, prod_data)
        psi_interpretation = (
            "No significant shift" if psi < 0.1
            else "Moderate shift" if psi < 0.25
            else "Significant shift"
        )

        # Wasserstein
        wd = wasserstein_distance(reference, prod_data)

        print(f"    KS Test:     statistic={ks['statistic']:.4f}, "
              f"p-value={ks['p_value']:.6f} -> {ks_interpretation}")
        print(f"    PSI:         {psi:.4f} -> {psi_interpretation}")
        print(f"    Wasserstein: {wd:.4f}")

        # Distribution summary
        ref_mean = statistics.mean(reference)
        ref_std = statistics.stdev(reference)
        prod_mean = statistics.mean(prod_data)
        prod_std = statistics.stdev(prod_data)
        print(f"    Reference:   mean={ref_mean:.3f}, std={ref_std:.3f}")
        print(f"    Production:  mean={prod_mean:.3f}, std={prod_std:.3f}")

    return {"ks_test": ks_test, "psi": calculate_psi, "wasserstein": wasserstein_distance}


# ============================================================
# Exercise 2: Evidently Reports
# ============================================================

def exercise_2_evidently_reports():
    """Simulate generating an Evidently-style drift and data quality report.

    Evidently AI generates HTML reports with:
    - Dataset summary statistics
    - Per-feature drift detection
    - Data quality checks (nulls, duplicates, out-of-range)
    - Prediction drift analysis
    """

    random.seed(42)

    # Feature definitions with expected ranges
    features = {
        "age": {"type": "numerical", "min": 18, "max": 90, "mean": 35, "std": 12},
        "income": {"type": "numerical", "min": 0, "max": 500000, "mean": 55000, "std": 25000},
        "tenure_months": {"type": "numerical", "min": 0, "max": 120, "mean": 24, "std": 18},
        "monthly_charges": {"type": "numerical", "min": 10, "max": 200, "mean": 65, "std": 30},
        "contract_type": {"type": "categorical", "values": ["month-to-month", "one-year", "two-year"],
                          "distribution": [0.5, 0.3, 0.2]},
    }

    def generate_data(features, n_samples, drift_config=None):
        """Generate simulated data with optional drift."""
        data = []
        for _ in range(n_samples):
            row = {}
            for fname, fconfig in features.items():
                if fconfig["type"] == "numerical":
                    mean = fconfig["mean"]
                    std = fconfig["std"]
                    if drift_config and fname in drift_config:
                        mean += drift_config[fname].get("mean_shift", 0)
                        std *= drift_config[fname].get("std_factor", 1)
                    value = random.gauss(mean, std)
                    # Inject some nulls
                    if random.random() < 0.02:
                        value = None
                    row[fname] = value
                else:
                    dist = fconfig["distribution"]
                    if drift_config and fname in drift_config:
                        dist = drift_config[fname].get("distribution", dist)
                    r = random.random()
                    cumsum = 0
                    for val, prob in zip(fconfig["values"], dist):
                        cumsum += prob
                        if r <= cumsum:
                            row[fname] = val
                            break
            data.append(row)
        return data

    # Generate reference and production data
    reference_data = generate_data(features, 1000)
    production_data = generate_data(features, 1000, drift_config={
        "age": {"mean_shift": 5},       # Age drift
        "income": {"std_factor": 1.5},   # Income variance increase
        "contract_type": {"distribution": [0.7, 0.2, 0.1]},  # Contract shift
    })

    # --- Generate Report ---
    print("Evidently-Style Data Report")
    print("=" * 60)

    # Section 1: Dataset Summary
    print("\n  SECTION 1: Dataset Summary")
    print(f"  {'-'*50}")
    print(f"  {'':20s} {'Reference':>12s} {'Production':>12s}")
    print(f"  {'Samples':<20s} {len(reference_data):>12d} {len(production_data):>12d}")

    for fname in features:
        ref_vals = [r[fname] for r in reference_data if r[fname] is not None]
        prod_vals = [r[fname] for r in production_data if r[fname] is not None]

        ref_nulls = sum(1 for r in reference_data if r[fname] is None)
        prod_nulls = sum(1 for r in production_data if r[fname] is None)

        if features[fname]["type"] == "numerical":
            ref_mean = statistics.mean(ref_vals)
            prod_mean = statistics.mean(prod_vals)
            print(f"  {fname + ' (mean)':<20s} {ref_mean:>12.1f} {prod_mean:>12.1f}")
            print(f"  {fname + ' (nulls)':<20s} {ref_nulls:>12d} {prod_nulls:>12d}")

    # Section 2: Feature Drift
    print(f"\n  SECTION 2: Feature Drift Detection")
    print(f"  {'-'*50}")
    print(f"  {'Feature':<20s} {'Type':<12s} {'Drift Score':>12s} {'Status':>10s}")
    print(f"  {'-'*54}")

    for fname, fconfig in features.items():
        ref_vals = [r[fname] for r in reference_data if r[fname] is not None]
        prod_vals = [r[fname] for r in production_data if r[fname] is not None]

        if fconfig["type"] == "numerical":
            # KS test for numerical
            n1, n2 = len(ref_vals), len(prod_vals)
            combined = sorted(set(ref_vals + prod_vals))
            # Subsample for speed
            combined = combined[::max(1, len(combined) // 200)]
            s1, s2 = sorted(ref_vals), sorted(prod_vals)
            max_diff = 0
            for x in combined:
                cdf1 = sum(1 for v in s1 if v <= x) / n1
                cdf2 = sum(1 for v in s2 if v <= x) / n2
                max_diff = max(max_diff, abs(cdf1 - cdf2))
            drift_score = max_diff
        else:
            # Chi-square-like for categorical
            ref_counts = {}
            prod_counts = {}
            for v in ref_vals:
                ref_counts[v] = ref_counts.get(v, 0) + 1
            for v in prod_vals:
                prod_counts[v] = prod_counts.get(v, 0) + 1
            all_cats = set(ref_counts.keys()) | set(prod_counts.keys())
            drift_score = sum(
                abs(ref_counts.get(c, 0) / len(ref_vals) -
                    prod_counts.get(c, 0) / len(prod_vals))
                for c in all_cats
            ) / 2

        status = "OK" if drift_score < 0.1 else "DRIFT" if drift_score < 0.2 else "ALERT"
        print(f"  {fname:<20s} {fconfig['type']:<12s} {drift_score:>12.4f} {status:>10s}")

    # Section 3: Data Quality
    print(f"\n  SECTION 3: Data Quality Checks")
    print(f"  {'-'*50}")

    quality_checks = []
    for fname, fconfig in features.items():
        prod_vals = [r[fname] for r in production_data]
        null_count = sum(1 for v in prod_vals if v is None)
        null_rate = null_count / len(prod_vals)

        quality_checks.append({
            "feature": fname,
            "null_rate": null_rate,
            "null_status": "OK" if null_rate < 0.05 else "WARN",
        })

        if fconfig["type"] == "numerical":
            valid_vals = [v for v in prod_vals if v is not None]
            out_of_range = sum(1 for v in valid_vals
                               if v < fconfig["min"] or v > fconfig["max"])
            oor_rate = out_of_range / len(valid_vals) if valid_vals else 0
            quality_checks[-1]["oor_rate"] = oor_rate
            quality_checks[-1]["oor_status"] = "OK" if oor_rate < 0.01 else "WARN"

    print(f"  {'Feature':<20s} {'Null Rate':>10s} {'Status':>8s} {'OOR Rate':>10s} {'Status':>8s}")
    print(f"  {'-'*56}")
    for qc in quality_checks:
        oor_rate = qc.get('oor_rate', 'N/A')
        oor_status = qc.get('oor_status', 'N/A')
        oor_str = f"{oor_rate:.4f}" if isinstance(oor_rate, float) else oor_rate
        print(f"  {qc['feature']:<20s} {qc['null_rate']:>10.4f} {qc['null_status']:>8s} "
              f"{oor_str:>10s} {str(oor_status):>8s}")

    return quality_checks


# ============================================================
# Exercise 3: Alert System
# ============================================================

def exercise_3_alert_system():
    """Build an alerting system for drift monitoring.

    Features:
    - Configurable thresholds per metric
    - Multiple severity levels (info, warning, critical)
    - Cooldown periods to avoid alert fatigue
    - Action triggers (retrain, notify, rollback)
    """

    class AlertRule:
        def __init__(self, name, metric, threshold, severity, action, cooldown_minutes=60):
            self.name = name
            self.metric = metric
            self.threshold = threshold
            self.severity = severity
            self.action = action
            self.cooldown_minutes = cooldown_minutes
            self.last_fired = None

        def evaluate(self, current_value, current_time):
            if current_value > self.threshold:
                if self.last_fired:
                    elapsed = (current_time - self.last_fired).total_seconds() / 60
                    if elapsed < self.cooldown_minutes:
                        return None  # In cooldown
                self.last_fired = current_time
                return {
                    "rule": self.name,
                    "metric": self.metric,
                    "value": round(current_value, 4),
                    "threshold": self.threshold,
                    "severity": self.severity,
                    "action": self.action,
                    "time": current_time.isoformat(),
                }
            return None

    class AlertManager:
        def __init__(self):
            self.rules = []
            self.alert_history = []
            self.actions_taken = []

        def add_rule(self, rule):
            self.rules.append(rule)

        def evaluate(self, metrics, current_time):
            """Evaluate all rules against current metrics."""
            alerts = []
            for rule in self.rules:
                value = metrics.get(rule.metric)
                if value is not None:
                    alert = rule.evaluate(value, current_time)
                    if alert:
                        alerts.append(alert)
                        self.alert_history.append(alert)
                        self.actions_taken.append(alert["action"])
            return alerts

        def get_summary(self):
            severity_counts = {"info": 0, "warning": 0, "critical": 0}
            for alert in self.alert_history:
                severity_counts[alert["severity"]] += 1
            return {
                "total_alerts": len(self.alert_history),
                "by_severity": severity_counts,
                "actions_taken": self.actions_taken,
            }

    # --- Configure alert rules ---
    manager = AlertManager()

    manager.add_rule(AlertRule(
        "data_drift_warning", "ks_statistic", 0.1, "warning",
        "notify_data_team", cooldown_minutes=120,
    ))
    manager.add_rule(AlertRule(
        "data_drift_critical", "ks_statistic", 0.2, "critical",
        "trigger_retraining", cooldown_minutes=240,
    ))
    manager.add_rule(AlertRule(
        "psi_warning", "psi", 0.15, "warning",
        "notify_ml_engineer", cooldown_minutes=120,
    ))
    manager.add_rule(AlertRule(
        "psi_critical", "psi", 0.25, "critical",
        "trigger_retraining", cooldown_minutes=240,
    ))
    manager.add_rule(AlertRule(
        "accuracy_drop", "accuracy_delta", 0.05, "warning",
        "notify_ml_lead", cooldown_minutes=60,
    ))
    manager.add_rule(AlertRule(
        "accuracy_severe_drop", "accuracy_delta", 0.10, "critical",
        "rollback_model", cooldown_minutes=30,
    ))
    manager.add_rule(AlertRule(
        "latency_spike", "p99_latency_ms", 100, "warning",
        "notify_infra_team", cooldown_minutes=30,
    ))
    manager.add_rule(AlertRule(
        "null_rate_high", "null_rate", 0.05, "info",
        "log_data_quality_issue", cooldown_minutes=360,
    ))

    # --- Simulate 48 hours of monitoring ---
    print("Alert System Simulation")
    print("=" * 60)

    print("\nConfigured Rules:")
    for rule in manager.rules:
        print(f"  [{rule.severity:>8s}] {rule.name}: "
              f"{rule.metric} > {rule.threshold} -> {rule.action} "
              f"(cooldown: {rule.cooldown_minutes}min)")

    print("\n48-Hour Monitoring Simulation:")
    print("-" * 60)

    random.seed(42)
    base_time = datetime(2025, 3, 1)

    for hour in range(48):
        current_time = base_time + timedelta(hours=hour)

        # Simulate metrics that gradually drift
        drift_factor = hour / 48  # Increases over time
        metrics = {
            "ks_statistic": 0.05 + drift_factor * 0.25 + random.gauss(0, 0.03),
            "psi": 0.08 + drift_factor * 0.3 + random.gauss(0, 0.04),
            "accuracy_delta": drift_factor * 0.15 + random.gauss(0, 0.02),
            "p99_latency_ms": 40 + drift_factor * 80 + random.gauss(0, 10),
            "null_rate": 0.01 + drift_factor * 0.06 + random.gauss(0, 0.01),
        }
        # Ensure non-negative
        metrics = {k: max(0, v) for k, v in metrics.items()}

        alerts = manager.evaluate(metrics, current_time)

        if alerts:
            print(f"\n  Hour {hour:2d} ({current_time.strftime('%Y-%m-%d %H:%M')}):")
            for alert in alerts:
                icon = {"info": "INFO", "warning": "WARN", "critical": "CRIT"}
                print(f"    [{icon[alert['severity']]}] {alert['rule']}: "
                      f"{alert['metric']}={alert['value']:.4f} "
                      f"(threshold={alert['threshold']}) -> {alert['action']}")

    # Summary
    summary = manager.get_summary()
    print(f"\nAlert Summary:")
    print(f"-" * 40)
    print(f"  Total alerts: {summary['total_alerts']}")
    print(f"  By severity:")
    for sev, count in summary["by_severity"].items():
        print(f"    {sev}: {count}")
    print(f"  Actions taken:")
    action_counts = {}
    for action in summary["actions_taken"]:
        action_counts[action] = action_counts.get(action, 0) + 1
    for action, count in sorted(action_counts.items(), key=lambda x: -x[1]):
        print(f"    {action}: {count}")

    return manager


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    print("Exercise 1: Drift Detection")
    print("=" * 60)
    exercise_1_drift_detection()

    print("\n\n")
    print("Exercise 2: Evidently Reports")
    print("=" * 60)
    exercise_2_evidently_reports()

    print("\n\n")
    print("Exercise 3: Alert System")
    print("=" * 60)
    exercise_3_alert_system()
