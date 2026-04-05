"""
Exercises for Lesson 14: Production Interpretability
Topic: Interpretable_AI

Solutions to practice problems from the lesson.
"""

import numpy as np
import hashlib
import time
import json
from collections import defaultdict


# === Exercise 1: Explanation Caching Strategy ===
# Problem: Design and implement an explanation caching system that stores
# precomputed explanations and handles cache invalidation when the model
# or input distribution changes.

def exercise_1():
    """Design an explanation caching strategy with TTL and invalidation."""
    np.random.seed(42)

    class ExplanationCache:
        """LRU-style cache for model explanations with TTL and model versioning."""

        def __init__(self, max_size=1000, ttl_seconds=3600, model_version="v1.0"):
            self.cache = {}
            self.access_order = []  # For LRU eviction
            self.max_size = max_size
            self.ttl_seconds = ttl_seconds
            self.model_version = model_version
            self.stats = {"hits": 0, "misses": 0, "evictions": 0, "invalidations": 0}

        def _make_key(self, input_data):
            """Create a deterministic cache key from input features."""
            data_str = json.dumps(input_data, sort_keys=True)
            return hashlib.md5(
                f"{self.model_version}:{data_str}".encode()
            ).hexdigest()

        def get(self, input_data):
            """Retrieve cached explanation if valid."""
            key = self._make_key(input_data)
            if key in self.cache:
                entry = self.cache[key]
                # Check TTL
                if time.time() - entry["timestamp"] < self.ttl_seconds:
                    self.stats["hits"] += 1
                    # Update access order
                    if key in self.access_order:
                        self.access_order.remove(key)
                    self.access_order.append(key)
                    return entry["explanation"]
                else:
                    # Expired
                    del self.cache[key]
                    self.stats["invalidations"] += 1
            self.stats["misses"] += 1
            return None

        def put(self, input_data, explanation):
            """Store explanation in cache."""
            key = self._make_key(input_data)
            # Evict if at capacity
            while len(self.cache) >= self.max_size and self.access_order:
                oldest_key = self.access_order.pop(0)
                if oldest_key in self.cache:
                    del self.cache[oldest_key]
                    self.stats["evictions"] += 1

            self.cache[key] = {
                "explanation": explanation,
                "timestamp": time.time(),
            }
            self.access_order.append(key)

        def invalidate_all(self, new_model_version):
            """Invalidate entire cache on model update."""
            old_size = len(self.cache)
            self.cache.clear()
            self.access_order.clear()
            self.model_version = new_model_version
            self.stats["invalidations"] += old_size
            return old_size

        def hit_rate(self):
            total = self.stats["hits"] + self.stats["misses"]
            return self.stats["hits"] / total if total > 0 else 0.0

    # Simulate cache usage
    cache = ExplanationCache(max_size=50, ttl_seconds=300, model_version="v1.0")

    # Generate synthetic inputs and explanations
    def compute_explanation(input_data):
        """Simulate expensive explanation computation."""
        features = input_data["features"]
        return {"feature_importances": {f"f{i}": round(v * 0.1, 4)
                                        for i, v in enumerate(features)},
                "prediction": round(sum(features) * 0.1, 4)}

    # Phase 1: Cold cache - all misses
    print("  Phase 1: Cold cache (first 20 requests)")
    for i in range(20):
        input_data = {"features": list(np.random.randn(5).round(2))}
        result = cache.get(input_data)
        if result is None:
            explanation = compute_explanation(input_data)
            cache.put(input_data, explanation)
    print(f"    Hit rate: {cache.hit_rate():.2%}")
    print(f"    Cache size: {len(cache.cache)}")

    # Phase 2: Repeat some requests - should get hits
    print("\n  Phase 2: Repeated requests (same 20 inputs)")
    cached_inputs = [{"features": list(np.random.RandomState(42 + i).randn(5).round(2))}
                     for i in range(20)]
    # First fill
    for inp in cached_inputs:
        result = cache.get(inp)
        if result is None:
            cache.put(inp, compute_explanation(inp))
    # Then repeat
    for inp in cached_inputs:
        cache.get(inp)
    print(f"    Hit rate: {cache.hit_rate():.2%}")
    print(f"    Stats: {cache.stats}")

    # Phase 3: Model update invalidation
    print("\n  Phase 3: Model update (cache invalidation)")
    invalidated = cache.invalidate_all("v2.0")
    print(f"    Invalidated {invalidated} entries")
    print(f"    Cache size after invalidation: {len(cache.cache)}")
    print(f"    New model version: {cache.model_version}")

    # Phase 4: Cache size strategy analysis
    print("\n  Phase 4: Cache sizing analysis")
    for max_sz in [10, 50, 100, 500]:
        test_cache = ExplanationCache(max_size=max_sz)
        # Simulate workload: 80% popular inputs (from pool of 20), 20% unique
        popular_pool = [{"features": list(np.random.randn(5).round(2))} for _ in range(20)]
        for _ in range(200):
            if np.random.rand() < 0.8:
                inp = popular_pool[np.random.randint(20)]
            else:
                inp = {"features": list(np.random.randn(5).round(2))}
            if test_cache.get(inp) is None:
                test_cache.put(inp, {"importance": "dummy"})
        print(f"    max_size={max_sz:>4}: hit_rate={test_cache.hit_rate():.2%}, "
              f"evictions={test_cache.stats['evictions']}")


# === Exercise 2: Explanation Drift Statistics ===
# Problem: Compute statistics to detect when the distribution of model
# explanations has shifted, indicating potential concept drift.

def exercise_2():
    """Compute explanation drift statistics over time windows."""
    np.random.seed(42)

    # Simulate explanation feature importances over time
    n_features = 5
    feature_names = [f"feature_{i}" for i in range(n_features)]

    # Baseline period: stable importances
    baseline_importances = []
    for _ in range(200):
        imp = np.array([0.35, 0.25, 0.20, 0.12, 0.08])
        imp += np.random.normal(0, 0.03, n_features)
        imp = np.abs(imp)
        imp /= imp.sum()
        baseline_importances.append(imp)

    baseline_importances = np.array(baseline_importances)
    baseline_mean = baseline_importances.mean(axis=0)
    baseline_std = baseline_importances.std(axis=0)

    # Production period: gradual drift in feature importance
    production_windows = {
        "Week 1 (stable)": np.array([0.35, 0.25, 0.20, 0.12, 0.08]),
        "Week 2 (mild drift)": np.array([0.30, 0.28, 0.22, 0.12, 0.08]),
        "Week 3 (moderate drift)": np.array([0.22, 0.30, 0.25, 0.13, 0.10]),
        "Week 4 (severe drift)": np.array([0.10, 0.15, 0.40, 0.20, 0.15]),
    }

    def compute_drift_stats(window_importances, baseline_mean, baseline_std):
        """Compute drift statistics for a window of explanations."""
        window_mean = window_importances.mean(axis=0)

        # Population Stability Index (PSI)
        psi = 0.0
        for i in range(len(baseline_mean)):
            p = max(baseline_mean[i], 1e-8)
            q = max(window_mean[i], 1e-8)
            psi += (p - q) * np.log(p / q)

        # Jensen-Shannon divergence
        m = 0.5 * (baseline_mean + window_mean)
        kl_pm = np.sum(baseline_mean * np.log(baseline_mean / (m + 1e-10) + 1e-10))
        kl_qm = np.sum(window_mean * np.log(window_mean / (m + 1e-10) + 1e-10))
        jsd = 0.5 * (kl_pm + kl_qm)

        # Z-score of mean shift
        z_scores = (window_mean - baseline_mean) / (baseline_std + 1e-10)

        # Rank correlation (Spearman-like)
        baseline_rank = np.argsort(-baseline_mean)
        window_rank = np.argsort(-window_mean)
        rank_agreement = np.mean(baseline_rank == window_rank)

        return {
            "psi": psi,
            "jsd": jsd,
            "z_scores": z_scores,
            "max_abs_z": np.max(np.abs(z_scores)),
            "rank_agreement": rank_agreement,
            "window_mean": window_mean,
        }

    print("  Explanation Drift Detection:")
    print(f"  Baseline mean importance: {baseline_mean.round(4)}")
    print(f"  Baseline std:             {baseline_std.round(4)}")
    print()

    for window_name, center in production_windows.items():
        # Generate samples for window
        window_imps = []
        for _ in range(50):
            imp = center + np.random.normal(0, 0.02, n_features)
            imp = np.abs(imp)
            imp /= imp.sum()
            window_imps.append(imp)
        window_imps = np.array(window_imps)

        stats = compute_drift_stats(window_imps, baseline_mean, baseline_std)

        # Alert levels
        if stats["psi"] > 0.25:
            alert = "CRITICAL"
        elif stats["psi"] > 0.1:
            alert = "WARNING"
        else:
            alert = "NORMAL"

        print(f"  {window_name}:")
        print(f"    Mean importance:  {stats['window_mean'].round(4)}")
        print(f"    PSI:              {stats['psi']:.6f}  [{alert}]")
        print(f"    JSD:              {stats['jsd']:.6f}")
        print(f"    Max |Z-score|:    {stats['max_abs_z']:.4f}")
        print(f"    Rank agreement:   {stats['rank_agreement']:.2%}")
        print()

    print("  Thresholds: PSI < 0.1 = stable, 0.1-0.25 = warning, > 0.25 = critical")
    print("  Explanation drift often precedes performance degradation.")


# === Exercise 3: Choosing Explanation Methods Given Latency Constraints ===
# Problem: Given different latency budgets and model types, recommend the
# most appropriate explanation method and estimate its computational cost.

def exercise_3():
    """Select explanation methods based on latency constraints."""
    np.random.seed(42)

    # Define explanation methods with their characteristics
    explanation_methods = {
        "SHAP (KernelSHAP)": {
            "complexity": "O(2^n) approx, uses sampling",
            "base_latency_ms": 500,
            "scales_with": "features * samples",
            "model_agnostic": True,
            "faithfulness": "high",
            "supports_batch": False,
        },
        "SHAP (TreeSHAP)": {
            "complexity": "O(T * L * D)",
            "base_latency_ms": 5,
            "scales_with": "tree_depth * n_trees",
            "model_agnostic": False,
            "faithfulness": "exact",
            "supports_batch": True,
        },
        "LIME": {
            "complexity": "O(n_samples * n_features)",
            "base_latency_ms": 200,
            "scales_with": "n_perturbation_samples",
            "model_agnostic": True,
            "faithfulness": "medium",
            "supports_batch": False,
        },
        "Gradient-based (Saliency)": {
            "complexity": "O(1 forward + 1 backward)",
            "base_latency_ms": 10,
            "scales_with": "model_size",
            "model_agnostic": False,
            "faithfulness": "medium",
            "supports_batch": True,
        },
        "Integrated Gradients": {
            "complexity": "O(n_steps * (forward + backward))",
            "base_latency_ms": 100,
            "scales_with": "n_integration_steps",
            "model_agnostic": False,
            "faithfulness": "high",
            "supports_batch": True,
        },
        "Coefficients (Linear Model)": {
            "complexity": "O(1)",
            "base_latency_ms": 0.1,
            "scales_with": "constant",
            "model_agnostic": False,
            "faithfulness": "exact",
            "supports_batch": True,
        },
    }

    # Deployment scenarios
    scenarios = [
        {
            "name": "Real-time API (e-commerce)",
            "latency_budget_ms": 50,
            "model_type": "gradient_boosting",
            "n_features": 30,
            "throughput_rps": 1000,
        },
        {
            "name": "Batch processing (credit scoring)",
            "latency_budget_ms": 5000,
            "model_type": "neural_network",
            "n_features": 50,
            "throughput_rps": 10,
        },
        {
            "name": "Interactive dashboard (analytics)",
            "latency_budget_ms": 500,
            "model_type": "linear",
            "n_features": 15,
            "throughput_rps": 50,
        },
        {
            "name": "Mobile app (on-device)",
            "latency_budget_ms": 100,
            "model_type": "neural_network",
            "n_features": 20,
            "throughput_rps": 5,
        },
    ]

    # Model type to compatible methods mapping
    model_compatibility = {
        "linear": ["Coefficients (Linear Model)", "SHAP (KernelSHAP)", "LIME"],
        "gradient_boosting": ["SHAP (TreeSHAP)", "SHAP (KernelSHAP)", "LIME"],
        "neural_network": ["Gradient-based (Saliency)", "Integrated Gradients",
                           "SHAP (KernelSHAP)", "LIME"],
    }

    print("  Explanation Method Selection by Deployment Scenario:")
    print("  " + "=" * 70)

    for scenario in scenarios:
        print(f"\n  Scenario: {scenario['name']}")
        print(f"    Model type: {scenario['model_type']}")
        print(f"    Latency budget: {scenario['latency_budget_ms']} ms")
        print(f"    Features: {scenario['n_features']}")
        print(f"    Throughput: {scenario['throughput_rps']} req/s")

        compatible = model_compatibility.get(scenario["model_type"], [])
        print(f"\n    Compatible methods:")

        best_method = None
        best_score = -1

        for method_name in compatible:
            method = explanation_methods[method_name]
            est_latency = method["base_latency_ms"]

            # Rough scaling adjustment
            if method["scales_with"] == "features * samples":
                est_latency *= (scenario["n_features"] / 10)
            elif method["scales_with"] == "n_perturbation_samples":
                est_latency *= (scenario["n_features"] / 10)

            fits = est_latency <= scenario["latency_budget_ms"]
            faith = {"exact": 3, "high": 2, "medium": 1}.get(method["faithfulness"], 0)

            # Score: faithfulness * 10 + (budget_remaining)
            if fits:
                score = faith * 10 + (scenario["latency_budget_ms"] - est_latency) / 100
            else:
                score = -1

            status = "FITS" if fits else "TOO SLOW"
            print(f"      {method_name:<35} ~{est_latency:>6.0f} ms [{status}]"
                  f"  faithfulness={method['faithfulness']}")

            if score > best_score:
                best_score = score
                best_method = method_name

        if best_method:
            print(f"    RECOMMENDATION: {best_method}")
        else:
            print(f"    RECOMMENDATION: Use precomputed explanations (caching)")


# === Exercise 4: Designing an Explanation Monitoring Alert ===
# Problem: Design alert rules that trigger when explanation patterns indicate
# potential model issues (drift, bias emergence, feature dependency changes).

def exercise_4():
    """Design and simulate an explanation monitoring alert system."""
    np.random.seed(42)

    class ExplanationMonitor:
        """Monitors explanation statistics and fires alerts."""

        def __init__(self, feature_names, baseline_stats):
            self.feature_names = feature_names
            self.baseline = baseline_stats
            self.alert_rules = []
            self.alert_history = []

        def add_rule(self, name, check_fn, severity="WARNING"):
            self.alert_rules.append({
                "name": name,
                "check": check_fn,
                "severity": severity,
            })

        def evaluate(self, window_stats, window_name):
            """Evaluate all rules against current window statistics."""
            alerts = []
            for rule in self.alert_rules:
                triggered, details = rule["check"](self.baseline, window_stats)
                if triggered:
                    alert = {
                        "rule": rule["name"],
                        "severity": rule["severity"],
                        "window": window_name,
                        "details": details,
                    }
                    alerts.append(alert)
                    self.alert_history.append(alert)
            return alerts

    # Define alert rules
    def importance_shift_rule(baseline, current, threshold=0.15):
        shifts = {}
        for feat in baseline["mean_importance"]:
            base_val = baseline["mean_importance"][feat]
            curr_val = current["mean_importance"].get(feat, 0)
            if abs(curr_val - base_val) > threshold:
                shifts[feat] = {"baseline": base_val, "current": curr_val,
                                "shift": curr_val - base_val}
        return len(shifts) > 0, shifts

    def top_feature_change_rule(baseline, current):
        base_top = max(baseline["mean_importance"], key=baseline["mean_importance"].get)
        curr_top = max(current["mean_importance"], key=current["mean_importance"].get)
        changed = base_top != curr_top
        return changed, {"baseline_top": base_top, "current_top": curr_top}

    def explanation_variance_rule(baseline, current, multiplier=2.0):
        high_var_features = {}
        for feat in baseline["std_importance"]:
            base_std = baseline["std_importance"][feat]
            curr_std = current["std_importance"].get(feat, 0)
            if curr_std > base_std * multiplier:
                high_var_features[feat] = {
                    "baseline_std": base_std, "current_std": curr_std
                }
        return len(high_var_features) > 0, high_var_features

    def concentration_rule(baseline, current, threshold=0.6):
        """Alert if a single feature dominates explanations."""
        max_imp = max(current["mean_importance"].values())
        concentrated = max_imp > threshold
        return concentrated, {"max_importance": max_imp, "threshold": threshold}

    # Setup
    features = ["credit_score", "income", "debt_ratio", "employment_years", "age"]
    baseline = {
        "mean_importance": {"credit_score": 0.35, "income": 0.25, "debt_ratio": 0.20,
                            "employment_years": 0.12, "age": 0.08},
        "std_importance": {"credit_score": 0.03, "income": 0.02, "debt_ratio": 0.02,
                           "employment_years": 0.01, "age": 0.01},
    }

    monitor = ExplanationMonitor(features, baseline)
    monitor.add_rule("Feature Importance Shift", importance_shift_rule, "WARNING")
    monitor.add_rule("Top Feature Changed", top_feature_change_rule, "CRITICAL")
    monitor.add_rule("Explanation Variance Spike", explanation_variance_rule, "WARNING")
    monitor.add_rule("Feature Concentration", concentration_rule, "CRITICAL")

    # Simulate production windows
    windows = {
        "2025-W01 (normal)": {
            "mean_importance": {"credit_score": 0.34, "income": 0.26, "debt_ratio": 0.21,
                                "employment_years": 0.11, "age": 0.08},
            "std_importance": {"credit_score": 0.03, "income": 0.02, "debt_ratio": 0.02,
                               "employment_years": 0.01, "age": 0.01},
        },
        "2025-W02 (mild shift)": {
            "mean_importance": {"credit_score": 0.28, "income": 0.30, "debt_ratio": 0.22,
                                "employment_years": 0.12, "age": 0.08},
            "std_importance": {"credit_score": 0.04, "income": 0.03, "debt_ratio": 0.02,
                               "employment_years": 0.01, "age": 0.01},
        },
        "2025-W03 (severe drift)": {
            "mean_importance": {"credit_score": 0.10, "income": 0.15, "debt_ratio": 0.45,
                                "employment_years": 0.20, "age": 0.10},
            "std_importance": {"credit_score": 0.08, "income": 0.06, "debt_ratio": 0.05,
                               "employment_years": 0.03, "age": 0.02},
        },
        "2025-W04 (concentration)": {
            "mean_importance": {"credit_score": 0.70, "income": 0.10, "debt_ratio": 0.10,
                                "employment_years": 0.05, "age": 0.05},
            "std_importance": {"credit_score": 0.02, "income": 0.01, "debt_ratio": 0.01,
                               "employment_years": 0.01, "age": 0.01},
        },
    }

    print("  Explanation Monitoring Dashboard:")
    print("  " + "=" * 65)

    for window_name, window_stats in windows.items():
        alerts = monitor.evaluate(window_stats, window_name)
        if alerts:
            for alert in alerts:
                print(f"\n  [{alert['severity']}] {window_name}")
                print(f"    Rule: {alert['rule']}")
                details = alert["details"]
                if isinstance(details, dict):
                    for k, v in details.items():
                        print(f"    {k}: {v}")
        else:
            print(f"\n  [OK] {window_name} - All checks passed")

    print(f"\n  --- Alert Summary ---")
    severity_counts = defaultdict(int)
    for alert in monitor.alert_history:
        severity_counts[alert["severity"]] += 1
    for sev, count in sorted(severity_counts.items()):
        print(f"    {sev}: {count} alerts")
    print(f"    Total alerts: {len(monitor.alert_history)}")

    print(f"\n  Explanation monitoring catches model issues early,")
    print(f"  often before performance metrics degrade.")


if __name__ == "__main__":
    print("=== Exercise 1: Explanation Caching Strategy ===")
    exercise_1()
    print("\n=== Exercise 2: Explanation Drift Statistics ===")
    exercise_2()
    print("\n=== Exercise 3: Choosing Explanation Methods by Latency ===")
    exercise_3()
    print("\n=== Exercise 4: Explanation Monitoring Alerts ===")
    exercise_4()
    print("\nAll exercises completed!")
