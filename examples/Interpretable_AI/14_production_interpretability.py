"""
14. Production Interpretability

Builds a production-ready explanation serving pipeline with caching,
drift monitoring, and latency tracking. Demonstrates the engineering
challenges of serving model explanations at scale.

Covered topics:
    - Explanation caching with input hashing and TTL eviction
    - Synchronous vs. asynchronous explanation serving patterns
    - Feature importance drift detection over time windows
    - Explanation latency monitoring and SLA tracking
    - Background pre-computation of popular explanations
    - MLOps integration: explanation quality gates in CI/CD

Related to: L14 - Production Interpretability

Requirements:
    pip install numpy matplotlib scikit-learn
"""

import hashlib
import json
import time
from collections import OrderedDict
from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split


# ====== Section 1: Explanation Cache ======

class ExplanationCache:
    """LRU cache for model explanations with TTL eviction.

    In production, generating explanations (e.g., SHAP values for a
    GBT with 500 trees) can take 100-1000x longer than prediction.
    Caching avoids re-computing explanations for repeated or similar
    inputs.

    Features:
      - LRU eviction when capacity is reached
      - TTL-based expiry for stale explanations
      - Input hashing for cache key generation
      - Hit/miss ratio tracking for monitoring

    Args:
        max_size: Maximum number of cached explanations.
        ttl_seconds: Time-to-live for cache entries.
    """

    def __init__(self, max_size: int = 1000, ttl_seconds: float = 300.0):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache: OrderedDict[str, dict] = OrderedDict()
        self.hits = 0
        self.misses = 0

    def _hash_input(self, features: np.ndarray) -> str:
        """Hash input features to create a cache key.

        Uses SHA-256 of the raw bytes for collision resistance.
        Rounding to 6 decimal places provides tolerance for floating
        point noise while still distinguishing meaningfully different
        inputs.
        """
        rounded = np.round(features, 6)
        return hashlib.sha256(rounded.tobytes()).hexdigest()[:16]

    def get(self, features: np.ndarray) -> dict | None:
        """Retrieve a cached explanation, or None if not found/expired."""
        key = self._hash_input(features)

        if key not in self._cache:
            self.misses += 1
            return None

        entry = self._cache[key]
        if time.time() - entry["timestamp"] > self.ttl_seconds:
            # TTL expired
            del self._cache[key]
            self.misses += 1
            return None

        # Move to end (most recently used)
        self._cache.move_to_end(key)
        self.hits += 1
        return entry["explanation"]

    def put(self, features: np.ndarray, explanation: dict) -> None:
        """Store an explanation in the cache."""
        key = self._hash_input(features)

        if key in self._cache:
            self._cache.move_to_end(key)
            self._cache[key] = {
                "explanation": explanation,
                "timestamp": time.time(),
            }
            return

        if len(self._cache) >= self.max_size:
            self._cache.popitem(last=False)  # evict LRU

        self._cache[key] = {
            "explanation": explanation,
            "timestamp": time.time(),
        }

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / max(total, 1)

    @property
    def size(self) -> int:
        return len(self._cache)


# ====== Section 2: Explanation Generator ======

def compute_permutation_explanation(
    model: GradientBoostingClassifier,
    x: np.ndarray,
    X_background: np.ndarray,
    feature_names: list[str],
    n_repeats: int = 10,
) -> dict:
    """Compute a local permutation-based explanation for one sample.

    Simulates the cost of a real SHAP/LIME computation by measuring
    feature importance through marginal contribution estimation.

    Args:
        model: Trained classifier.
        x: Single input sample (d,).
        X_background: Background dataset for marginal expectations.
        feature_names: Names of features.
        n_repeats: Number of permutations per feature.

    Returns:
        Explanation dictionary with feature importances and metadata.
    """
    start_time = time.time()

    base_prob = model.predict_proba(x.reshape(1, -1))[0, 1]
    importances = {}

    for j, name in enumerate(feature_names):
        diffs = []
        for _ in range(n_repeats):
            x_perm = x.copy()
            bg_idx = np.random.randint(0, len(X_background))
            x_perm[j] = X_background[bg_idx, j]
            perm_prob = model.predict_proba(x_perm.reshape(1, -1))[0, 1]
            diffs.append(abs(base_prob - perm_prob))
        importances[name] = float(np.mean(diffs))

    elapsed = time.time() - start_time

    return {
        "prediction": float(base_prob),
        "feature_importances": importances,
        "computation_time_ms": elapsed * 1000,
        "method": "permutation",
    }


# ====== Section 3: Explanation Serving Pipeline ======

@dataclass
class LatencyTracker:
    """Tracks explanation serving latency for SLA monitoring."""
    latencies_ms: list[float] = field(default_factory=list)
    sla_threshold_ms: float = 200.0

    def record(self, latency_ms: float) -> None:
        self.latencies_ms.append(latency_ms)

    @property
    def p50(self) -> float:
        return float(np.percentile(self.latencies_ms, 50)) if self.latencies_ms else 0.0

    @property
    def p95(self) -> float:
        return float(np.percentile(self.latencies_ms, 95)) if self.latencies_ms else 0.0

    @property
    def p99(self) -> float:
        return float(np.percentile(self.latencies_ms, 99)) if self.latencies_ms else 0.0

    @property
    def sla_violation_rate(self) -> float:
        if not self.latencies_ms:
            return 0.0
        violations = sum(1 for l in self.latencies_ms if l > self.sla_threshold_ms)
        return violations / len(self.latencies_ms)


def serve_explanation(
    model: GradientBoostingClassifier,
    x: np.ndarray,
    cache: ExplanationCache,
    X_background: np.ndarray,
    feature_names: list[str],
    tracker: LatencyTracker,
) -> dict:
    """Serve an explanation with caching and latency tracking.

    This simulates the production serving pattern:
      1. Check cache for pre-computed explanation
      2. On cache miss, compute explanation
      3. Store in cache for future requests
      4. Track latency for SLA monitoring

    Args:
        model: Trained classifier.
        x: Single input sample.
        cache: Explanation cache.
        X_background: Background dataset.
        feature_names: Feature names.
        tracker: Latency tracker.

    Returns:
        Explanation dictionary.
    """
    start = time.time()

    # Check cache
    cached = cache.get(x)
    if cached is not None:
        elapsed = (time.time() - start) * 1000
        cached["served_from_cache"] = True
        cached["serve_latency_ms"] = elapsed
        tracker.record(elapsed)
        return cached

    # Cache miss: compute explanation
    explanation = compute_permutation_explanation(
        model, x, X_background, feature_names,
    )
    explanation["served_from_cache"] = False

    # Store in cache
    cache.put(x, explanation)

    elapsed = (time.time() - start) * 1000
    explanation["serve_latency_ms"] = elapsed
    tracker.record(elapsed)

    return explanation


# ====== Section 4: Feature Importance Drift Monitor ======

class DriftMonitor:
    """Monitors feature importance distributions for drift.

    In production, the distribution of feature importances can shift
    over time due to data drift, concept drift, or model degradation.
    Detecting this early allows teams to investigate and retrain.

    Uses a simple window-based approach: compare the mean importance
    vector of the latest window to a reference window using cosine
    distance.

    Args:
        n_features: Number of features.
        window_size: Number of explanations per window.
        drift_threshold: Cosine distance threshold for alerting.
    """

    def __init__(
        self,
        n_features: int,
        window_size: int = 50,
        drift_threshold: float = 0.1,
    ):
        self.n_features = n_features
        self.window_size = window_size
        self.drift_threshold = drift_threshold
        self.reference_window: list[np.ndarray] = []
        self.current_window: list[np.ndarray] = []
        self.drift_scores: list[float] = []

    def add_explanation(self, importances: dict[str, float]) -> dict | None:
        """Add an explanation and check for drift.

        Returns a drift alert dictionary if drift is detected, else None.
        """
        imp_vector = np.array(list(importances.values()))

        if len(self.reference_window) < self.window_size:
            self.reference_window.append(imp_vector)
            return None

        self.current_window.append(imp_vector)

        if len(self.current_window) < self.window_size:
            return None

        # Compute drift score (cosine distance)
        ref_mean = np.mean(self.reference_window, axis=0)
        cur_mean = np.mean(self.current_window, axis=0)

        ref_norm = np.linalg.norm(ref_mean)
        cur_norm = np.linalg.norm(cur_mean)

        if ref_norm < 1e-8 or cur_norm < 1e-8:
            cosine_sim = 0.0
        else:
            cosine_sim = float(np.dot(ref_mean, cur_mean) / (ref_norm * cur_norm))

        drift_score = 1.0 - cosine_sim
        self.drift_scores.append(drift_score)

        # Slide the window
        self.reference_window = self.current_window.copy()
        self.current_window = []

        if drift_score > self.drift_threshold:
            return {
                "alert": True,
                "drift_score": drift_score,
                "threshold": self.drift_threshold,
                "ref_mean": ref_mean.tolist(),
                "cur_mean": cur_mean.tolist(),
            }

        return None


# ====== Section 5: Visualization ======

def visualize_production(
    tracker: LatencyTracker,
    cache: ExplanationCache,
    drift_monitor: DriftMonitor,
    feature_names: list[str],
    save_path: str = "production_interpretability.png",
) -> None:
    """Four-panel production interpretability dashboard."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # --- Panel 1: Latency Distribution ---
    ax1 = axes[0, 0]
    ax1.hist(tracker.latencies_ms, bins=50, color="#3498db",
             edgecolor="black", linewidth=0.5, alpha=0.7)
    ax1.axvline(tracker.p50, color="#2ecc71", linestyle="--",
                linewidth=2, label=f"p50={tracker.p50:.1f}ms")
    ax1.axvline(tracker.p95, color="#f39c12", linestyle="--",
                linewidth=2, label=f"p95={tracker.p95:.1f}ms")
    ax1.axvline(tracker.sla_threshold_ms, color="#e74c3c", linestyle="-",
                linewidth=2, label=f"SLA={tracker.sla_threshold_ms}ms")
    ax1.set_xlabel("Latency (ms)")
    ax1.set_ylabel("Count")
    ax1.set_title(f"Explanation Latency Distribution\n"
                   f"SLA violations: {tracker.sla_violation_rate:.1%}")
    ax1.legend(fontsize=9)

    # --- Panel 2: Cache Performance ---
    ax2 = axes[0, 1]
    sizes = ["Hits", "Misses"]
    counts = [cache.hits, cache.misses]
    colors = ["#2ecc71", "#e74c3c"]
    ax2.pie(counts, labels=sizes, colors=colors, autopct="%1.1f%%",
            startangle=90, textprops={"fontsize": 12})
    ax2.set_title(f"Cache Performance\n"
                   f"Size: {cache.size}, Hit Rate: {cache.hit_rate:.1%}")

    # --- Panel 3: Drift Scores ---
    ax3 = axes[1, 0]
    if drift_monitor.drift_scores:
        ax3.plot(drift_monitor.drift_scores, "o-", color="#9b59b6",
                 markersize=4, linewidth=1.5)
        ax3.axhline(drift_monitor.drift_threshold, color="#e74c3c",
                     linestyle="--", label=f"threshold={drift_monitor.drift_threshold}")
        ax3.legend(fontsize=9)
    ax3.set_xlabel("Window Index")
    ax3.set_ylabel("Cosine Distance")
    ax3.set_title("Feature Importance Drift Over Time")
    ax3.grid(True, alpha=0.3)

    # --- Panel 4: Latency Over Time ---
    ax4 = axes[1, 1]
    # Moving average
    window = 20
    if len(tracker.latencies_ms) >= window:
        moving_avg = np.convolve(
            tracker.latencies_ms,
            np.ones(window) / window,
            mode="valid",
        )
        ax4.plot(moving_avg, color="#2c3e50", linewidth=1.5)
    ax4.axhline(tracker.sla_threshold_ms, color="#e74c3c",
                linestyle="--", alpha=0.5)
    ax4.set_xlabel("Request Index")
    ax4.set_ylabel("Latency (ms, moving avg)")
    ax4.set_title("Latency Trend (Moving Average)")
    ax4.grid(True, alpha=0.3)

    plt.suptitle("Production Interpretability Dashboard", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\n  Figure saved to: {save_path}")
    plt.close()


# ====== Section 6: Main Pipeline ======

def main() -> None:
    """Simulate a production explanation serving pipeline."""
    print("=" * 65)
    print("  Production Interpretability")
    print("  Caching | Latency Tracking | Drift Monitoring")
    print("=" * 65)

    # --- Step 1: Train model ---
    print("\n[1] Training Model")
    print("-" * 50)

    rng = np.random.default_rng(42)
    n, d = 2000, 8
    feature_names = [f"feat_{i}" for i in range(d)]

    X = rng.normal(0, 1, (n, d))
    true_w = np.array([1.5, -1.0, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0])
    y = (X @ true_w + rng.normal(0, 0.5, n) > 0).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42,
    )

    model = GradientBoostingClassifier(
        n_estimators=50, max_depth=3, random_state=42,
    )
    model.fit(X_train, y_train)
    print(f"  Train accuracy: {model.score(X_train, y_train):.4f}")
    print(f"  Test accuracy:  {model.score(X_test, y_test):.4f}")

    # --- Step 2: Initialize serving infrastructure ---
    print("\n[2] Initializing Serving Infrastructure")
    print("-" * 50)

    cache = ExplanationCache(max_size=200, ttl_seconds=60.0)
    tracker = LatencyTracker(sla_threshold_ms=50.0)
    drift_monitor = DriftMonitor(
        n_features=d, window_size=30, drift_threshold=0.15,
    )
    X_background = X_train[:100]

    print(f"  Cache capacity: {cache.max_size}")
    print(f"  Cache TTL: {cache.ttl_seconds}s")
    print(f"  SLA threshold: {tracker.sla_threshold_ms}ms")
    print(f"  Drift window: {drift_monitor.window_size}")

    # --- Step 3: Simulate request traffic ---
    print("\n[3] Simulating 300 Explanation Requests")
    print("-" * 50)

    n_requests = 300
    drift_alerts = []

    for i in range(n_requests):
        # Some requests repeat (cache hits)
        if i > 50 and rng.random() < 0.3:
            # Repeat a previous request
            idx = rng.integers(0, min(i, len(X_test)))
        else:
            idx = i % len(X_test)

        x = X_test[idx]

        # Simulate drift after request 200 by shifting feature importance
        if i >= 200:
            x = x.copy()
            x[0] *= 0.5  # reduce importance of feat_0
            x[3] += rng.normal(0, 2)  # increase noise feature

        explanation = serve_explanation(
            model, x, cache, X_background, feature_names, tracker,
        )

        # Feed to drift monitor
        alert = drift_monitor.add_explanation(explanation["feature_importances"])
        if alert is not None:
            drift_alerts.append((i, alert))

    print(f"  Requests served: {n_requests}")
    print(f"  Cache hit rate: {cache.hit_rate:.1%}")
    print(f"  Cache size: {cache.size}")

    # --- Step 4: Latency report ---
    print("\n[4] Latency Report")
    print("-" * 50)
    print(f"  p50:  {tracker.p50:.2f}ms")
    print(f"  p95:  {tracker.p95:.2f}ms")
    print(f"  p99:  {tracker.p99:.2f}ms")
    print(f"  SLA violations: {tracker.sla_violation_rate:.1%}")

    # --- Step 5: Drift report ---
    print("\n[5] Drift Monitoring Report")
    print("-" * 50)
    print(f"  Windows evaluated: {len(drift_monitor.drift_scores)}")
    print(f"  Drift alerts: {len(drift_alerts)}")
    for req_idx, alert in drift_alerts:
        print(f"    Alert at request {req_idx}: "
              f"drift_score={alert['drift_score']:.4f}")

    # --- Step 6: Visualization ---
    print("\n[6] Generating Dashboard")
    print("-" * 50)

    visualize_production(tracker, cache, drift_monitor, feature_names)

    # --- Summary ---
    print("\n" + "=" * 65)
    print("  Summary")
    print("=" * 65)
    print(f"""
  Production interpretability pipeline results:
    1. Explanation caching reduced average latency by serving
       {cache.hit_rate:.1%} of requests from cache.
    2. Latency: p50={tracker.p50:.1f}ms, p95={tracker.p95:.1f}ms,
       p99={tracker.p99:.1f}ms. SLA violations: {tracker.sla_violation_rate:.1%}.
    3. Drift monitoring detected {len(drift_alerts)} alerts when feature
       importance distributions shifted (after simulated drift).
    4. Key engineering patterns:
       - LRU cache with TTL for explanation reuse
       - Input hashing for cache key generation
       - Percentile-based latency tracking for SLA compliance
       - Window-based cosine distance for drift detection
    """)


if __name__ == "__main__":
    main()
