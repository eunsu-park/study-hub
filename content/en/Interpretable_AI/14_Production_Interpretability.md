# Lesson 14: Production Interpretability

[Previous: AI Regulation and Governance](./13_AI_Regulation_and_Governance.md) | [Next: Domain-Specific Interpretability](./15_Domain_Specific_Interpretability.md)

---

## Learning Objectives

- Design explanation serving architectures (synchronous, asynchronous, hybrid) that meet production SLA requirements while providing meaningful interpretability
- Implement explanation caching strategies with input hashing and model-version-aware cache invalidation to reduce computation costs
- Build explanation drift monitoring systems that detect changes in feature importance distributions over time and trigger alerts
- Integrate interpretability artifacts into MLOps pipelines using MLflow, CI/CD validation gates, and automated explanation quality checks
- Construct a complete FastAPI explanation service with Redis caching, background pre-computation, and real-time monitoring dashboards

---

## 1. Architecture Patterns for Explanation Serving

### 1.1 The Explanation Serving Problem

In production, generating explanations is often MORE expensive than generating
predictions. A SHAP explanation for a single prediction can take 100x-1000x
longer than the prediction itself. This creates an engineering challenge:
how do we provide explanations without violating latency SLAs?

```python
"""
The Explanation Latency Problem

Consider a credit scoring API with a 200ms SLA:
  - Model inference:     5ms   (fast — GPU-accelerated)
  - SHAP explanation:   500ms  (slow — requires many forward passes)
  - LIME explanation:  2000ms  (very slow — trains surrogate model)

We CANNOT simply compute explanations synchronously for every request.
We need architectural patterns that decouple explanation generation
from prediction serving.

Three primary patterns exist:
1. SYNCHRONOUS:  Explanation computed with every prediction
2. ASYNCHRONOUS: Explanation computed in background, retrieved later
3. HYBRID:       Fast explanation synchronous, detailed explanation async
"""

import time
from dataclasses import dataclass, field
from typing import Optional, Callable, Any
from enum import Enum


class ExplanationPattern(Enum):
    """The three primary explanation serving patterns."""
    SYNCHRONOUS = "synchronous"
    ASYNCHRONOUS = "asynchronous"
    HYBRID = "hybrid"


@dataclass
class LatencyBudget:
    """Defines the latency budget for a prediction + explanation service.

    The budget determines which explanation methods are viable.
    Methods that exceed the budget must be served asynchronously.
    """
    total_sla_ms: float           # Total allowed end-to-end latency
    prediction_ms: float          # Time allocated to prediction
    explanation_ms: float         # Time allocated to explanation
    network_overhead_ms: float    # Time for serialization, network, etc.

    @property
    def available_for_explanation(self) -> float:
        """How much time is left for explanation computation."""
        return self.total_sla_ms - self.prediction_ms - self.network_overhead_ms

    def can_serve_synchronously(self, method_latency_ms: float) -> bool:
        """Check if an explanation method fits within the SLA.

        WHY this check matters: choosing the wrong method can cause
        SLA violations, timeouts, and degraded user experience.
        """
        return method_latency_ms <= self.available_for_explanation

    def recommend_pattern(self, method_latency_ms: float) -> ExplanationPattern:
        """Recommend an architecture pattern based on latency budget.

        Decision logic:
        - If method fits in SLA: synchronous (simplest, best UX)
        - If method is 2-5x over budget: hybrid (fast approximation sync)
        - If method is >5x over budget: fully asynchronous
        """
        budget = self.available_for_explanation
        if method_latency_ms <= budget:
            return ExplanationPattern.SYNCHRONOUS
        elif method_latency_ms <= budget * 5:
            return ExplanationPattern.HYBRID
        else:
            return ExplanationPattern.ASYNCHRONOUS


# Example: evaluate explanation methods for different SLAs
EXPLANATION_METHODS = {
    "Feature importance (tree-based)": 2,        # Built-in, ~instant
    "Coefficient-based (linear model)": 1,       # Direct extraction
    "SHAP (TreeExplainer)": 50,                  # Optimized for trees
    "SHAP (KernelExplainer)": 500,               # Model-agnostic, slow
    "LIME (tabular)": 2000,                      # Trains surrogate
    "Integrated Gradients (neural net)": 100,    # Multiple forward passes
    "GradCAM (CNN)": 15,                         # Single backward pass
    "Counterfactual (DiCE)": 3000,               # Optimization loop
}

SLA_PROFILES = {
    "Real-time API (200ms)": LatencyBudget(200, 10, 0, 20),
    "Interactive dashboard (1s)": LatencyBudget(1000, 10, 0, 50),
    "Batch processing (30s)": LatencyBudget(30000, 10, 0, 100),
}

print("EXPLANATION METHOD vs. SLA COMPATIBILITY")
print("=" * 80)

for sla_name, budget in SLA_PROFILES.items():
    print(f"\n{sla_name}")
    print(f"  Available for explanation: {budget.available_for_explanation:.0f}ms")
    print(f"  {'Method':40s} {'Latency':>10s} {'Pattern':>15s}")
    print(f"  {'-' * 65}")

    for method, latency in EXPLANATION_METHODS.items():
        pattern = budget.recommend_pattern(latency)
        print(f"  {method:40s} {latency:>8.0f}ms {pattern.value:>15s}")
```

### 1.2 Synchronous Explanation Serving

The simplest pattern: compute the explanation alongside every prediction.
Only viable for fast explanation methods.

```python
"""
Synchronous Explanation Serving

USE WHEN:
  - Explanation method is fast enough to fit within SLA
  - Every prediction MUST have an explanation (regulatory requirement)
  - User needs explanation immediately (interactive UX)

ADVANTAGES:
  - Simplest architecture — no additional infrastructure
  - Guaranteed consistency — explanation always matches prediction
  - No eventual consistency issues

DISADVANTAGES:
  - Limited to fast explanation methods
  - Prediction latency increases
  - CPU/memory cost per request increases
"""

import numpy as np
from dataclasses import dataclass
from typing import Any


@dataclass
class PredictionWithExplanation:
    """Response object containing both prediction and explanation.

    Every synchronous response includes the explanation inline.
    This guarantees the explanation is always available and
    always corresponds to the exact prediction served.
    """
    prediction: float
    prediction_label: str
    confidence: float
    explanation: dict[str, float]  # feature -> importance
    model_version: str
    latency_ms: float


class SynchronousExplainer:
    """Serves predictions with inline explanations.

    This is the simplest production pattern. The explanation
    is computed as part of the same request processing pipeline.
    """

    def __init__(self, model, feature_names: list[str], model_version: str):
        self.model = model
        self.feature_names = feature_names
        self.model_version = model_version

    def predict_and_explain(self, features: np.ndarray) -> PredictionWithExplanation:
        """Generate prediction and explanation in a single call.

        For tree-based models, we use built-in feature importance
        combined with individual prediction contribution (via
        predict with pred_contribs if available, or approximation).

        WHY we use feature importance here instead of SHAP:
        Built-in importance is O(1) per prediction for tree models,
        while SHAP is O(n_features * n_samples) — too slow for sync.
        """
        start_time = time.time()

        # Step 1: Generate prediction
        prediction_proba = float(self.model.predict_proba(
            features.reshape(1, -1)
        )[0, 1])
        prediction_label = "approved" if prediction_proba >= 0.5 else "denied"

        # Step 2: Generate explanation (fast method)
        # Use feature importance as a proxy for per-prediction explanation.
        # In production, you'd use TreeExplainer for per-prediction SHAP,
        # which is still fast enough for tree models.
        importances = self.model.feature_importances_
        feature_contributions = {}
        for name, importance in zip(self.feature_names, importances):
            feature_contributions[name] = round(float(importance), 4)

        # Sort by absolute importance
        feature_contributions = dict(
            sorted(feature_contributions.items(),
                   key=lambda x: abs(x[1]), reverse=True)
        )

        elapsed_ms = (time.time() - start_time) * 1000

        return PredictionWithExplanation(
            prediction=prediction_proba,
            prediction_label=prediction_label,
            confidence=abs(prediction_proba - 0.5) * 2,  # 0-1 scale
            explanation=feature_contributions,
            model_version=self.model_version,
            latency_ms=elapsed_ms,
        )


# Example usage with the credit model from Lesson 13
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Create and train model
X, y = make_classification(n_samples=2000, n_features=8, random_state=42)
feature_names = [
    "income", "employment_years", "debt_ratio", "credit_history",
    "loan_amount", "savings", "housing_status", "num_dependents",
]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=42)
model.fit(X_train, y_train)

# Serve with synchronous explanations
explainer = SynchronousExplainer(model, feature_names, model_version="v2.3.1")

# Simulate serving 5 requests
print("SYNCHRONOUS EXPLANATION SERVING")
print("=" * 60)
for i in range(5):
    result = explainer.predict_and_explain(X_test[i])
    print(f"\nRequest {i+1}:")
    print(f"  Prediction: {result.prediction_label} ({result.prediction:.3f})")
    print(f"  Confidence: {result.confidence:.3f}")
    print(f"  Latency: {result.latency_ms:.2f}ms")
    print(f"  Top 3 factors:")
    for j, (feature, importance) in enumerate(result.explanation.items()):
        if j >= 3:
            break
        print(f"    {feature}: {importance:.4f}")
```

### 1.3 Asynchronous Explanation Serving

When explanations are too expensive for real-time serving, decouple them
from the prediction path using a message queue or background worker.

```python
"""
Asynchronous Explanation Serving

USE WHEN:
  - Explanation method is too slow for real-time SLA
  - Explanations are not needed for every prediction
  - Users can tolerate a delay (seconds to minutes)

ARCHITECTURE:
  1. Prediction API returns prediction immediately
  2. Prediction ID is enqueued for explanation generation
  3. Background worker computes explanation and stores it
  4. Client polls or receives webhook when explanation is ready

ADVANTAGES:
  - No impact on prediction latency
  - Can use expensive methods (LIME, counterfactuals)
  - Worker pool can be scaled independently

DISADVANTAGES:
  - More complex infrastructure (queue, workers, storage)
  - Eventual consistency — explanation may not be ready immediately
  - Must handle case where explanation generation fails
"""

import uuid
import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum
from collections import deque
from threading import Lock


class ExplanationStatus(Enum):
    """Status of an asynchronous explanation request."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class ExplanationRequest:
    """A request for asynchronous explanation generation."""
    request_id: str
    prediction_id: str
    input_features: list[float]
    model_version: str
    method: str  # "shap", "lime", "counterfactual"
    status: ExplanationStatus = ExplanationStatus.PENDING
    result: Optional[dict] = None
    error: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None


class AsyncExplanationService:
    """Asynchronous explanation generation service.

    In production, this would use:
    - Redis or RabbitMQ for the task queue
    - Celery or similar for background workers
    - PostgreSQL or S3 for explanation storage

    This simplified version uses in-memory structures
    to demonstrate the pattern.
    """

    def __init__(self, model, feature_names: list[str]):
        self.model = model
        self.feature_names = feature_names
        self._queue: deque = deque()
        self._results: dict[str, ExplanationRequest] = {}
        self._lock = Lock()

    def submit_explanation_request(
        self,
        prediction_id: str,
        input_features: list[float],
        model_version: str,
        method: str = "shap",
    ) -> str:
        """Submit a request for asynchronous explanation.

        Returns a request_id that can be used to poll for results.

        WHY we return immediately:
        The prediction API should not block on explanation generation.
        The client can poll for the explanation or receive a webhook.
        """
        request_id = str(uuid.uuid4())
        request = ExplanationRequest(
            request_id=request_id,
            prediction_id=prediction_id,
            input_features=input_features,
            model_version=model_version,
            method=method,
        )

        with self._lock:
            self._queue.append(request)
            self._results[request_id] = request

        return request_id

    def get_explanation_status(self, request_id: str) -> Optional[ExplanationRequest]:
        """Check the status of an explanation request.

        Clients poll this endpoint to check if their explanation is ready.
        In production, consider websockets or webhooks instead of polling.
        """
        return self._results.get(request_id)

    def process_next(self) -> Optional[str]:
        """Process the next explanation request in the queue.

        This simulates a background worker processing explanation requests.
        In production, this would be a Celery task or similar.
        """
        with self._lock:
            if not self._queue:
                return None
            request = self._queue.popleft()
            request.status = ExplanationStatus.PROCESSING

        try:
            # Simulate expensive explanation computation
            features = np.array(request.input_features).reshape(1, -1)

            # In production, this would be SHAP/LIME computation
            # Here we simulate with feature importance
            importances = self.model.feature_importances_
            explanation = {
                name: round(float(imp), 4)
                for name, imp in zip(self.feature_names, importances)
            }

            # Simulate computation time
            time.sleep(0.01)  # In reality: 500ms-3000ms

            request.result = {
                "method": request.method,
                "feature_contributions": explanation,
                "base_value": float(self.model.predict_proba(features)[0, 1]),
            }
            request.status = ExplanationStatus.COMPLETED
            request.completed_at = time.time()

        except Exception as e:
            request.status = ExplanationStatus.FAILED
            request.error = str(e)

        return request.request_id


# Demonstrate async flow
async_service = AsyncExplanationService(model, feature_names)

print("ASYNCHRONOUS EXPLANATION SERVING")
print("=" * 60)

# Step 1: Submit requests
request_ids = []
for i in range(3):
    rid = async_service.submit_explanation_request(
        prediction_id=f"pred_{i}",
        input_features=X_test[i].tolist(),
        model_version="v2.3.1",
        method="shap",
    )
    request_ids.append(rid)
    print(f"Submitted request {rid[:8]}... for prediction pred_{i}")

# Step 2: Process requests (simulating background worker)
print("\nProcessing requests...")
for _ in range(3):
    processed_id = async_service.process_next()
    if processed_id:
        print(f"  Processed: {processed_id[:8]}...")

# Step 3: Retrieve results
print("\nRetrieving results:")
for rid in request_ids:
    result = async_service.get_explanation_status(rid)
    if result and result.status == ExplanationStatus.COMPLETED:
        elapsed = result.completed_at - result.created_at
        print(f"\n  Request {rid[:8]}...: {result.status.value}")
        print(f"  Processing time: {elapsed:.3f}s")
        top_features = sorted(
            result.result["feature_contributions"].items(),
            key=lambda x: abs(x[1]),
            reverse=True,
        )[:3]
        for name, importance in top_features:
            print(f"    {name}: {importance:.4f}")
```

---

## 2. Caching Strategies

### 2.1 Input-Hash-Based Explanation Caching

Explanation computation is deterministic for a given (input, model_version) pair.
Caching exploits this to avoid redundant computation.

```python
"""
Explanation Caching Strategy

KEY INSIGHT: For a given model version and input features,
the explanation is DETERMINISTIC. If we've computed it before,
we can serve it from cache instead of recomputing.

CACHE KEY: hash(model_version + input_features)

INVALIDATION TRIGGERS:
1. Model version changes (new deployment)
2. Explanation method changes (e.g., switching from SHAP to LIME)
3. TTL expiration (configurable, typically 24-48 hours)
4. Manual invalidation (admin override)

WHY input hashing:
  We hash the input features rather than using a request ID because
  different users with identical features should get the same
  explanation. This maximizes cache hit rate.
"""

import hashlib
import json
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Optional, Any
from threading import Lock


@dataclass
class CacheEntry:
    """A single cache entry for an explanation."""
    key: str
    explanation: dict
    model_version: str
    created_at: float
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)


class ExplanationCache:
    """LRU cache for explanations with model-version-aware invalidation.

    This cache is designed for explanation serving where:
    - Cache key = hash(model_version + input_features)
    - Entries are automatically invalidated when model version changes
    - LRU eviction prevents unbounded memory growth
    - TTL ensures stale explanations are not served indefinitely

    In production, this would be backed by Redis or Memcached.
    This in-memory implementation demonstrates the logic.
    """

    def __init__(self, max_size: int = 10000, ttl_seconds: float = 86400):
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._max_size = max_size
        self._ttl_seconds = ttl_seconds
        self._lock = Lock()
        self._stats = {"hits": 0, "misses": 0, "evictions": 0, "invalidations": 0}

    @staticmethod
    def _compute_cache_key(model_version: str, input_features: list[float]) -> str:
        """Compute a deterministic cache key from model version and features.

        WHY we include model_version in the key:
        Different model versions produce different explanations for the
        same input. Including version in the key ensures we never serve
        a stale explanation from a previous model.

        WHY we use SHA-256:
        We need a fixed-length key that is collision-resistant.
        SHA-256 provides both properties.
        """
        # Normalize features to avoid floating-point representation issues
        # Round to 6 decimal places — sufficient for most ML applications
        normalized = [round(f, 6) for f in input_features]
        payload = json.dumps({
            "model_version": model_version,
            "features": normalized,
        }, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()

    def get(self, model_version: str, input_features: list[float]) -> Optional[dict]:
        """Retrieve an explanation from cache.

        Returns None on cache miss. On cache hit, moves entry to
        end of LRU queue (most recently used).
        """
        key = self._compute_cache_key(model_version, input_features)

        with self._lock:
            if key not in self._cache:
                self._stats["misses"] += 1
                return None

            entry = self._cache[key]

            # Check TTL
            if time.time() - entry.created_at > self._ttl_seconds:
                del self._cache[key]
                self._stats["misses"] += 1
                return None

            # Check model version (defense-in-depth)
            if entry.model_version != model_version:
                del self._cache[key]
                self._stats["invalidations"] += 1
                self._stats["misses"] += 1
                return None

            # Cache hit — update LRU position
            self._cache.move_to_end(key)
            entry.access_count += 1
            entry.last_accessed = time.time()
            self._stats["hits"] += 1
            return entry.explanation

    def put(self, model_version: str, input_features: list[float],
            explanation: dict) -> None:
        """Store an explanation in cache.

        If cache is full, evicts the least recently used entry.
        """
        key = self._compute_cache_key(model_version, input_features)

        with self._lock:
            if key in self._cache:
                # Update existing entry
                self._cache.move_to_end(key)
                self._cache[key].explanation = explanation
                self._cache[key].created_at = time.time()
                return

            # Evict LRU if at capacity
            while len(self._cache) >= self._max_size:
                self._cache.popitem(last=False)
                self._stats["evictions"] += 1

            self._cache[key] = CacheEntry(
                key=key,
                explanation=explanation,
                model_version=model_version,
                created_at=time.time(),
            )

    def invalidate_model(self, model_version: str) -> int:
        """Invalidate ALL cache entries for a specific model version.

        Called when a new model is deployed. This ensures no stale
        explanations are served from the previous model.

        WHY bulk invalidation:
        When a model is retrained, ALL explanations change because
        the underlying feature importances change. Serving cached
        explanations from the old model would be misleading.
        """
        count = 0
        with self._lock:
            keys_to_remove = [
                k for k, v in self._cache.items()
                if v.model_version == model_version
            ]
            for key in keys_to_remove:
                del self._cache[key]
                count += 1
            self._stats["invalidations"] += count
        return count

    @property
    def hit_rate(self) -> float:
        """Cache hit rate as a fraction."""
        total = self._stats["hits"] + self._stats["misses"]
        if total == 0:
            return 0.0
        return self._stats["hits"] / total

    def stats(self) -> dict:
        """Return cache statistics."""
        return {
            **self._stats,
            "size": len(self._cache),
            "max_size": self._max_size,
            "hit_rate": f"{self.hit_rate:.1%}",
        }


# Demonstrate caching behavior
cache = ExplanationCache(max_size=100, ttl_seconds=3600)

print("EXPLANATION CACHE DEMONSTRATION")
print("=" * 60)

# Simulate a sequence of requests
model_v1 = "v2.3.1"
sample_features = X_test[:10].tolist()

# First pass: all misses (cold cache)
print("\nPass 1: Cold cache (all misses)")
for features in sample_features:
    result = cache.get(model_v1, features)
    if result is None:
        # Compute explanation and cache it
        explanation = {"feature_1": 0.35, "feature_2": 0.25}
        cache.put(model_v1, features, explanation)

print(f"  Cache stats: {cache.stats()}")

# Second pass: all hits (warm cache)
print("\nPass 2: Warm cache (all hits)")
for features in sample_features:
    result = cache.get(model_v1, features)
    assert result is not None

print(f"  Cache stats: {cache.stats()}")

# Model update: invalidate old model cache
print("\nModel update: deploying v2.4.0")
invalidated = cache.invalidate_model(model_v1)
print(f"  Invalidated {invalidated} entries")
print(f"  Cache stats: {cache.stats()}")

# Post-update: misses again (new model)
print("\nPass 3: After model update (all misses)")
model_v2 = "v2.4.0"
for features in sample_features:
    result = cache.get(model_v2, features)
    if result is None:
        explanation = {"feature_1": 0.30, "feature_2": 0.28}
        cache.put(model_v2, features, explanation)

print(f"  Cache stats: {cache.stats()}")
```

### 2.2 Pre-computation for Common Inputs

For frequently queried inputs or representative examples, pre-compute
explanations offline and serve them instantly.

```python
"""
Pre-computation Strategy for Explanations

Instead of computing explanations on-demand, pre-compute them
for inputs that are likely to be queried.

CANDIDATES FOR PRE-COMPUTATION:
1. Cluster centroids — representative examples for each prediction group
2. Boundary cases — inputs near the decision boundary
3. High-volume inputs — frequently queried inputs (from access logs)
4. Regulatory examples — inputs that regulators might query

WHY pre-computation:
  - Enables expensive methods (LIME, counterfactuals) without latency hit
  - Provides consistent, pre-reviewed explanations
  - Reduces real-time compute costs
"""

import numpy as np
from sklearn.cluster import KMeans
from dataclasses import dataclass


@dataclass
class PrecomputedExplanation:
    """A pre-computed explanation with its associated cluster."""
    cluster_id: int
    centroid: np.ndarray
    prediction: float
    prediction_label: str
    explanation: dict[str, float]
    coverage: int  # How many real inputs this cluster covers


class ExplanationPrecomputer:
    """Pre-compute explanations for representative inputs.

    Strategy:
    1. Cluster recent inputs into k groups
    2. Compute expensive explanations for each cluster centroid
    3. At serving time, find nearest cluster and return pre-computed explanation
    4. Re-compute periodically (e.g., weekly) as input distribution shifts

    TRADEOFF:
    - More clusters = more accurate explanations but more computation
    - Fewer clusters = less computation but explanations are less specific
    - Typical: 50-200 clusters per prediction class
    """

    def __init__(self, model, feature_names: list[str], n_clusters: int = 20):
        self.model = model
        self.feature_names = feature_names
        self.n_clusters = n_clusters
        self.kmeans = None
        self.precomputed: dict[int, PrecomputedExplanation] = {}

    def fit(self, X: np.ndarray) -> None:
        """Cluster inputs and pre-compute explanations.

        This is run OFFLINE (batch job), not during request serving.
        Typical schedule: after each model deployment, then weekly.
        """
        # Step 1: Cluster inputs
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        labels = self.kmeans.fit_predict(X)

        # Step 2: Pre-compute explanation for each cluster centroid
        centroids = self.kmeans.cluster_centers_

        for cluster_id in range(self.n_clusters):
            centroid = centroids[cluster_id]

            # Generate prediction
            pred = float(self.model.predict_proba(centroid.reshape(1, -1))[0, 1])
            label = "approved" if pred >= 0.5 else "denied"

            # Compute explanation (can be expensive here — it's offline)
            # In production, this would be full SHAP/LIME
            importances = self.model.feature_importances_
            explanation = {
                name: round(float(imp * centroid[i]), 4)
                for i, (name, imp) in enumerate(
                    zip(self.feature_names, importances)
                )
            }

            coverage = int(np.sum(labels == cluster_id))

            self.precomputed[cluster_id] = PrecomputedExplanation(
                cluster_id=cluster_id,
                centroid=centroid,
                prediction=pred,
                prediction_label=label,
                explanation=explanation,
                coverage=coverage,
            )

    def get_nearest_explanation(self, features: np.ndarray) -> PrecomputedExplanation:
        """Find the nearest pre-computed explanation.

        This is O(k) where k is the number of clusters — very fast.
        In production with many clusters, use approximate nearest
        neighbor (e.g., FAISS) for O(log k) lookup.
        """
        cluster_id = int(self.kmeans.predict(features.reshape(1, -1))[0])
        return self.precomputed[cluster_id]

    def coverage_report(self) -> dict:
        """Report on pre-computation coverage.

        A good pre-computation should cover >90% of inputs
        with clusters of reasonable size.
        """
        coverages = [pc.coverage for pc in self.precomputed.values()]
        return {
            "total_clusters": self.n_clusters,
            "total_coverage": sum(coverages),
            "min_cluster_size": min(coverages),
            "max_cluster_size": max(coverages),
            "median_cluster_size": int(np.median(coverages)),
        }


# Demonstrate pre-computation
precomputer = ExplanationPrecomputer(model, feature_names, n_clusters=10)
precomputer.fit(X_train)

print("EXPLANATION PRE-COMPUTATION")
print("=" * 60)

# Coverage report
report = precomputer.coverage_report()
print(f"\nCoverage Report:")
for key, value in report.items():
    print(f"  {key}: {value}")

# Serve pre-computed explanations
print(f"\nServing pre-computed explanations:")
for i in range(5):
    start = time.time()
    explanation = precomputer.get_nearest_explanation(X_test[i])
    elapsed = (time.time() - start) * 1000

    print(f"\n  Input {i}: cluster {explanation.cluster_id}")
    print(f"  Prediction: {explanation.prediction_label} ({explanation.prediction:.3f})")
    print(f"  Lookup time: {elapsed:.3f}ms")  # Should be <1ms
    print(f"  Cluster coverage: {explanation.coverage} inputs")
```

---

## 3. Explanation Drift Monitoring

### 3.1 Detecting Changes in Feature Importance

When the distribution of feature importances changes over time, it may indicate
concept drift, data drift, or model degradation. Monitoring explanation drift
provides an early warning system.

```python
"""
Explanation Drift Monitoring

WHAT IS EXPLANATION DRIFT?
  When the distribution of feature importance values changes over time.
  This can indicate:
  1. DATA DRIFT: Input distribution has shifted
  2. CONCEPT DRIFT: Relationship between features and target has changed
  3. MODEL DEGRADATION: Model is becoming less relevant
  4. DATA QUALITY ISSUES: Upstream data pipeline problems

WHY monitor explanation drift (not just prediction drift):
  Prediction accuracy can remain stable while explanations change
  dramatically. This happens when the model compensates for one
  factor by increasing reliance on another. Explanation drift
  catches this BEFORE accuracy degrades.

HOW TO DETECT:
  1. Maintain a baseline distribution of feature importances
  2. Periodically compute feature importances on recent data
  3. Use statistical tests to detect significant changes
  4. Alert when drift exceeds a threshold
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from scipy import stats as scipy_stats


@dataclass
class DriftWindow:
    """A time window of feature importance observations."""
    window_id: str
    start_time: float
    end_time: float
    importance_distributions: dict[str, list[float]]  # feature -> list of importances
    n_samples: int


@dataclass
class DriftAlert:
    """An alert triggered by explanation drift."""
    feature_name: str
    drift_score: float         # KL divergence or similar
    baseline_mean: float
    current_mean: float
    test_statistic: float
    p_value: float
    alert_level: str           # "warning" or "critical"


class ExplanationDriftMonitor:
    """Monitors changes in feature importance distributions over time.

    Architecture:
    1. Establish baseline from initial deployment period
    2. Collect explanation data in rolling windows
    3. Compare each window to baseline using statistical tests
    4. Generate alerts when significant drift is detected

    In production, this integrates with monitoring systems
    like Prometheus/Grafana or Datadog.
    """

    def __init__(
        self,
        feature_names: list[str],
        warning_threshold: float = 0.05,   # p-value for warning
        critical_threshold: float = 0.01,  # p-value for critical
    ):
        self.feature_names = feature_names
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        self.baseline: Optional[DriftWindow] = None
        self.windows: list[DriftWindow] = []

    def set_baseline(self, importance_matrix: np.ndarray) -> None:
        """Set the baseline distribution from initial deployment data.

        importance_matrix: shape (n_samples, n_features)
        Each row is the feature importance vector for one prediction.

        WHY we need a baseline:
        Drift is RELATIVE — we need a reference point to compare against.
        The baseline should be from the model's initial deployment period
        when performance was verified and approved.
        """
        distributions = {}
        for i, name in enumerate(self.feature_names):
            distributions[name] = importance_matrix[:, i].tolist()

        self.baseline = DriftWindow(
            window_id="baseline",
            start_time=time.time(),
            end_time=time.time(),
            importance_distributions=distributions,
            n_samples=importance_matrix.shape[0],
        )

    def add_window(self, window_id: str, importance_matrix: np.ndarray) -> list[DriftAlert]:
        """Add a new monitoring window and check for drift.

        Returns a list of DriftAlerts for any features that show
        statistically significant changes from baseline.

        WHY Kolmogorov-Smirnov test:
        KS test is non-parametric — it doesn't assume a specific
        distribution shape. This is important because feature
        importance distributions are often non-normal.
        """
        if self.baseline is None:
            raise ValueError("Must set baseline before monitoring")

        # Build current window distributions
        distributions = {}
        for i, name in enumerate(self.feature_names):
            distributions[name] = importance_matrix[:, i].tolist()

        window = DriftWindow(
            window_id=window_id,
            start_time=time.time(),
            end_time=time.time(),
            importance_distributions=distributions,
            n_samples=importance_matrix.shape[0],
        )
        self.windows.append(window)

        # Compare each feature to baseline using KS test
        alerts = []
        for name in self.feature_names:
            baseline_dist = self.baseline.importance_distributions[name]
            current_dist = distributions[name]

            # Two-sample Kolmogorov-Smirnov test
            ks_stat, p_value = scipy_stats.ks_2samp(baseline_dist, current_dist)

            baseline_mean = float(np.mean(baseline_dist))
            current_mean = float(np.mean(current_dist))

            if p_value < self.critical_threshold:
                alert_level = "critical"
            elif p_value < self.warning_threshold:
                alert_level = "warning"
            else:
                continue  # No alert

            alerts.append(DriftAlert(
                feature_name=name,
                drift_score=ks_stat,
                baseline_mean=baseline_mean,
                current_mean=current_mean,
                test_statistic=ks_stat,
                p_value=p_value,
                alert_level=alert_level,
            ))

        return sorted(alerts, key=lambda a: a.p_value)


# Demonstrate drift monitoring
np.random.seed(42)

# Create baseline importance distributions
# Shape: (500, 8) — 500 samples, 8 features
baseline_importances = np.random.dirichlet(
    [3, 2, 2, 1.5, 1, 1, 0.5, 0.5], size=500
)

monitor = ExplanationDriftMonitor(feature_names)
monitor.set_baseline(baseline_importances)

# Window 1: No drift (same distribution)
print("EXPLANATION DRIFT MONITORING")
print("=" * 60)

window1 = np.random.dirichlet([3, 2, 2, 1.5, 1, 1, 0.5, 0.5], size=300)
alerts1 = monitor.add_window("week_1", window1)
print(f"\nWeek 1: {len(alerts1)} alerts")

# Window 2: Moderate drift (shift in one feature)
window2 = np.random.dirichlet([1.5, 4, 2, 1.5, 1, 1, 0.5, 0.5], size=300)
alerts2 = monitor.add_window("week_2", window2)
print(f"\nWeek 2: {len(alerts2)} alerts")
for alert in alerts2:
    print(f"  [{alert.alert_level.upper()}] {alert.feature_name}")
    print(f"    Baseline mean: {alert.baseline_mean:.4f}")
    print(f"    Current mean:  {alert.current_mean:.4f}")
    print(f"    KS statistic:  {alert.test_statistic:.4f}")
    print(f"    p-value:       {alert.p_value:.6f}")

# Window 3: Severe drift (major distribution change)
window3 = np.random.dirichlet([0.5, 0.5, 5, 3, 2, 1, 0.5, 0.5], size=300)
alerts3 = monitor.add_window("week_3", window3)
print(f"\nWeek 3: {len(alerts3)} alerts")
for alert in alerts3:
    print(f"  [{alert.alert_level.upper()}] {alert.feature_name}")
    print(f"    Drift score: {alert.drift_score:.4f}, p-value: {alert.p_value:.6f}")
```

---

## 4. MLOps Integration

### 4.1 Explanation Artifacts in MLflow

Treating explanations as first-class MLOps artifacts ensures traceability,
versioning, and reproducibility.

```python
"""
Explanation Artifacts in MLOps Pipelines

Explanations should be treated as FIRST-CLASS ARTIFACTS alongside
model weights, metrics, and data snapshots. This means:

1. VERSIONED: Each explanation is tied to a specific model version
2. STORED: Explanations are persisted in the model registry
3. VALIDATED: Explanation quality is checked before deployment
4. TRACKED: Changes in explanations across versions are logged

Integration points:
- MLflow: log explanations as artifacts alongside model
- CI/CD: validate explanation quality in deployment pipeline
- Model Registry: attach Model Card to registered model
- Monitoring: track explanation metrics in production

WHY this matters:
  Without MLOps integration, explanations become ad-hoc, inconsistent,
  and impossible to audit. Regulators may ask: "What explanation did
  user X receive on date Y?" Without versioning, you cannot answer.
"""

import json
import os
from dataclasses import dataclass, field
from typing import Any, Optional
from datetime import datetime


@dataclass
class ExplanationArtifact:
    """An explanation artifact to be logged in MLflow.

    This class packages explanation data in a format suitable
    for MLflow artifact logging.
    """
    model_version: str
    method: str
    feature_names: list[str]
    global_importance: dict[str, float]
    sample_explanations: list[dict]
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Serialize to dictionary for JSON storage."""
        return {
            "model_version": self.model_version,
            "method": self.method,
            "feature_names": self.feature_names,
            "global_importance": self.global_importance,
            "sample_explanations": self.sample_explanations,
            "metadata": self.metadata,
            "generated_at": datetime.now().isoformat(),
        }

    def save(self, path: str) -> None:
        """Save artifact to a JSON file.

        In production, this would be mlflow.log_artifact(path).
        """
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


def create_explanation_artifact(model, X_sample, feature_names, model_version):
    """Create an explanation artifact for MLflow logging.

    This function is called during the model training pipeline
    to generate explanation artifacts that are logged alongside
    the model.

    Steps:
    1. Compute global feature importance
    2. Generate sample explanations for representative inputs
    3. Package as ExplanationArtifact
    4. Return for logging to MLflow
    """
    # Global feature importance
    importances = model.feature_importances_
    global_importance = {
        name: round(float(imp), 6)
        for name, imp in zip(feature_names, importances)
    }

    # Sample explanations (top 5 from each prediction class)
    predictions = model.predict(X_sample)
    probas = model.predict_proba(X_sample)[:, 1]

    sample_explanations = []
    for pred_class in [0, 1]:
        mask = predictions == pred_class
        indices = np.where(mask)[0][:5]

        for idx in indices:
            sample_explanations.append({
                "sample_index": int(idx),
                "prediction": int(predictions[idx]),
                "probability": round(float(probas[idx]), 4),
                "feature_values": {
                    name: round(float(X_sample[idx, i]), 4)
                    for i, name in enumerate(feature_names)
                },
                "feature_importance": global_importance,
            })

    artifact = ExplanationArtifact(
        model_version=model_version,
        method="tree_feature_importance",
        feature_names=feature_names,
        global_importance=global_importance,
        sample_explanations=sample_explanations,
        metadata={
            "n_samples": len(X_sample),
            "n_features": len(feature_names),
            "model_type": type(model).__name__,
        },
    )

    return artifact


# Create artifact
artifact = create_explanation_artifact(
    model, X_test[:100], feature_names, "v2.3.1"
)

print("EXPLANATION ARTIFACT FOR MLFLOW")
print("=" * 60)
print(f"Model Version: {artifact.model_version}")
print(f"Method: {artifact.method}")
print(f"Sample Explanations: {len(artifact.sample_explanations)}")
print(f"\nGlobal Feature Importance:")
for name, imp in sorted(
    artifact.global_importance.items(), key=lambda x: -x[1]
):
    bar = "#" * int(imp * 100)
    print(f"  {name:20s}: {imp:.4f} {bar}")
```

### 4.2 CI/CD Validation Gates

```python
"""
CI/CD Validation for Explanations

Before a new model is deployed, automated checks should verify
that explanations meet quality standards. These checks serve as
DEPLOYMENT GATES — the pipeline blocks if they fail.

VALIDATION CHECKS:
1. CONSISTENCY: Do explanations agree with known ground truth?
2. STABILITY: Do similar inputs get similar explanations?
3. COMPLETENESS: Are all required explanation fields populated?
4. FAIRNESS: Do explanation patterns differ across demographic groups?
5. REGRESSION: Has explanation quality degraded from previous version?

WHY CI/CD gates:
  Manual explanation review doesn't scale. Automated gates catch
  regressions before they reach production, reducing risk and
  compliance overhead.
"""

from dataclasses import dataclass
from typing import Callable
from enum import Enum


class ValidationResult(Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    WARNING = "WARNING"


@dataclass
class ValidationCheck:
    """A single validation check in the CI/CD pipeline."""
    name: str
    description: str
    result: ValidationResult
    details: str
    is_blocking: bool  # If True, deployment is blocked on failure


class ExplanationValidator:
    """Validates explanation quality as a CI/CD deployment gate.

    This runs as part of the model deployment pipeline,
    after training and before production rollout.
    """

    def __init__(self, model, feature_names: list[str], X_test: np.ndarray,
                 y_test: np.ndarray):
        self.model = model
        self.feature_names = feature_names
        self.X_test = X_test
        self.y_test = y_test

    def check_explanation_stability(self, n_perturbations: int = 10,
                                     noise_scale: float = 0.01) -> ValidationCheck:
        """Check that explanations are stable under small input perturbations.

        WHY stability matters:
        If adding tiny noise to the input drastically changes the
        explanation, users cannot trust the explanation. Unstable
        explanations are a sign of unreliable interpretability.
        """
        importances_base = self.model.feature_importances_

        # Perturb a subset of test inputs and check explanation stability
        stabilities = []
        for i in range(min(50, len(self.X_test))):
            base_input = self.X_test[i:i+1]
            base_pred = self.model.predict_proba(base_input)[0, 1]

            # Generate perturbed predictions
            perturbed_preds = []
            for _ in range(n_perturbations):
                noise = np.random.normal(0, noise_scale, base_input.shape)
                perturbed = base_input + noise
                pred = self.model.predict_proba(perturbed)[0, 1]
                perturbed_preds.append(pred)

            # Stability = 1 - coefficient of variation of predictions
            pred_std = np.std(perturbed_preds)
            stability = 1.0 - min(pred_std / max(abs(base_pred), 1e-10), 1.0)
            stabilities.append(stability)

        mean_stability = np.mean(stabilities)

        if mean_stability >= 0.95:
            result = ValidationResult.PASS
        elif mean_stability >= 0.85:
            result = ValidationResult.WARNING
        else:
            result = ValidationResult.FAIL

        return ValidationCheck(
            name="Explanation Stability",
            description="Explanations should be stable under small input perturbations",
            result=result,
            details=f"Mean stability: {mean_stability:.3f} (threshold: 0.95)",
            is_blocking=True,
        )

    def check_feature_importance_coverage(self) -> ValidationCheck:
        """Check that no single feature dominates the explanation.

        WHY coverage matters:
        If one feature accounts for >80% of importance, the
        explanation is essentially "it's all about X" — which
        may indicate overfitting or a trivial model.
        """
        importances = self.model.feature_importances_
        max_importance = np.max(importances)
        total_importance = np.sum(importances)
        dominance_ratio = max_importance / total_importance

        if dominance_ratio < 0.5:
            result = ValidationResult.PASS
        elif dominance_ratio < 0.7:
            result = ValidationResult.WARNING
        else:
            result = ValidationResult.FAIL

        dominant_feature = self.feature_names[np.argmax(importances)]

        return ValidationCheck(
            name="Feature Importance Coverage",
            description="No single feature should dominate the explanation",
            result=result,
            details=(
                f"Max dominance ratio: {dominance_ratio:.3f} "
                f"(feature: {dominant_feature})"
            ),
            is_blocking=False,
        )

    def check_explanation_completeness(self) -> ValidationCheck:
        """Check that all features have non-zero importance.

        WHY completeness matters:
        Features with exactly zero importance may indicate:
        - Feature engineering bugs (feature is constant)
        - Data pipeline issues (feature not flowing correctly)
        - Model not using all available information
        """
        importances = self.model.feature_importances_
        zero_features = [
            name for name, imp in zip(self.feature_names, importances)
            if imp < 1e-10
        ]

        if len(zero_features) == 0:
            result = ValidationResult.PASS
            details = "All features have non-zero importance"
        elif len(zero_features) <= 2:
            result = ValidationResult.WARNING
            details = f"Zero-importance features: {zero_features}"
        else:
            result = ValidationResult.FAIL
            details = f"{len(zero_features)} features with zero importance: {zero_features}"

        return ValidationCheck(
            name="Explanation Completeness",
            description="All features should contribute to explanations",
            result=result,
            details=details,
            is_blocking=False,
        )

    def run_all_checks(self) -> list[ValidationCheck]:
        """Run all validation checks and return results.

        This is called by the CI/CD pipeline.
        """
        return [
            self.check_explanation_stability(),
            self.check_feature_importance_coverage(),
            self.check_explanation_completeness(),
        ]


# Run validation
validator = ExplanationValidator(model, feature_names, X_test, y_test)
checks = validator.run_all_checks()

print("CI/CD EXPLANATION VALIDATION")
print("=" * 60)

all_pass = True
for check in checks:
    blocking_str = "[BLOCKING]" if check.is_blocking else "[advisory]"
    print(f"\n{check.result.value:8s} {blocking_str} {check.name}")
    print(f"         {check.description}")
    print(f"         {check.details}")
    if check.result == ValidationResult.FAIL and check.is_blocking:
        all_pass = False

print(f"\n{'=' * 60}")
deployment_decision = "APPROVED" if all_pass else "BLOCKED"
print(f"Deployment Decision: {deployment_decision}")
```

---

## 5. A/B Testing Explanations

### 5.1 Measuring Explanation Effectiveness

```python
"""
A/B Testing Explanations

Explanations are user-facing outputs. Like any UX element,
they should be A/B tested to determine which format best
serves users' needs.

WHAT TO TEST:
1. TRUST: Do explanations increase user trust in the AI system?
2. ACTION: Do explanations help users take appropriate action?
3. COMPREHENSION: Do users correctly understand what the explanation means?
4. SATISFACTION: How satisfied are users with the explanation?

HOW TO TEST:
- Randomly assign users to explanation variants
- Measure outcome metrics per variant
- Use statistical tests to determine significance
- Iterate on the winning variant

COMMON VARIANTS:
- Feature importance (bar chart) vs. natural language
- Top-3 features vs. full feature ranking
- With counterfactual vs. without
- Technical explanation vs. plain language
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from scipy import stats as scipy_stats


@dataclass
class ExplanationVariant:
    """A variant in an A/B test of explanations."""
    variant_id: str
    name: str
    description: str


@dataclass
class UserInteraction:
    """A single user interaction with an explanation."""
    user_id: str
    variant_id: str
    trust_score: float          # 1-5 Likert scale
    comprehension_correct: bool  # Did they correctly interpret?
    took_action: bool            # Did they take the suggested action?
    time_spent_seconds: float    # How long they engaged with explanation


class ExplanationABTest:
    """A/B test framework for comparing explanation formats.

    This framework collects user interaction data and performs
    statistical analysis to determine which explanation format
    is most effective.
    """

    def __init__(self, test_name: str, variants: list[ExplanationVariant]):
        self.test_name = test_name
        self.variants = {v.variant_id: v for v in variants}
        self.interactions: list[UserInteraction] = []

    def record_interaction(self, interaction: UserInteraction) -> None:
        """Record a user interaction with an explanation."""
        self.interactions.append(interaction)

    def analyze(self) -> dict:
        """Analyze A/B test results.

        For each metric, compare variants using appropriate statistical tests:
        - Continuous metrics (trust, time): Mann-Whitney U test
        - Binary metrics (comprehension, action): Chi-squared test

        WHY Mann-Whitney instead of t-test:
        Likert scale data is ordinal, not interval. Mann-Whitney is
        non-parametric and doesn't assume normality.
        """
        results = {}

        variant_ids = list(self.variants.keys())
        if len(variant_ids) != 2:
            raise ValueError("A/B test requires exactly 2 variants")

        # Split interactions by variant
        group_a = [i for i in self.interactions if i.variant_id == variant_ids[0]]
        group_b = [i for i in self.interactions if i.variant_id == variant_ids[1]]

        # Trust score comparison (Mann-Whitney U)
        trust_a = [i.trust_score for i in group_a]
        trust_b = [i.trust_score for i in group_b]
        if trust_a and trust_b:
            u_stat, p_value = scipy_stats.mannwhitneyu(
                trust_a, trust_b, alternative="two-sided"
            )
            results["trust"] = {
                "variant_a_mean": np.mean(trust_a),
                "variant_b_mean": np.mean(trust_b),
                "u_statistic": u_stat,
                "p_value": p_value,
                "significant": p_value < 0.05,
                "winner": variant_ids[0] if np.mean(trust_a) > np.mean(trust_b)
                         else variant_ids[1],
            }

        # Comprehension comparison (proportions z-test)
        comp_a = sum(1 for i in group_a if i.comprehension_correct)
        comp_b = sum(1 for i in group_b if i.comprehension_correct)
        n_a, n_b = len(group_a), len(group_b)

        if n_a > 0 and n_b > 0:
            p_a = comp_a / n_a
            p_b = comp_b / n_b
            p_pooled = (comp_a + comp_b) / (n_a + n_b)
            se = np.sqrt(p_pooled * (1 - p_pooled) * (1/n_a + 1/n_b))
            if se > 0:
                z_stat = (p_a - p_b) / se
                p_value = 2 * (1 - scipy_stats.norm.cdf(abs(z_stat)))
            else:
                z_stat, p_value = 0.0, 1.0

            results["comprehension"] = {
                "variant_a_rate": p_a,
                "variant_b_rate": p_b,
                "z_statistic": z_stat,
                "p_value": p_value,
                "significant": p_value < 0.05,
            }

        # Action rate comparison
        action_a = sum(1 for i in group_a if i.took_action)
        action_b = sum(1 for i in group_b if i.took_action)

        if n_a > 0 and n_b > 0:
            results["action_rate"] = {
                "variant_a_rate": action_a / n_a,
                "variant_b_rate": action_b / n_b,
            }

        return results


# Simulate A/B test data
np.random.seed(42)

variants = [
    ExplanationVariant("bar_chart", "Bar Chart", "Feature importance as horizontal bar chart"),
    ExplanationVariant("narrative", "Narrative", "Natural language explanation paragraph"),
]

test = ExplanationABTest("Explanation Format Test", variants)

# Simulate 200 users, 100 per variant
for i in range(100):
    # Variant A: bar chart — higher comprehension, lower trust
    test.record_interaction(UserInteraction(
        user_id=f"user_a_{i}",
        variant_id="bar_chart",
        trust_score=np.random.choice([3, 4, 4, 4, 5]),
        comprehension_correct=np.random.random() < 0.78,
        took_action=np.random.random() < 0.45,
        time_spent_seconds=np.random.exponential(15),
    ))
    # Variant B: narrative — higher trust, lower comprehension
    test.record_interaction(UserInteraction(
        user_id=f"user_b_{i}",
        variant_id="narrative",
        trust_score=np.random.choice([3, 4, 4, 5, 5]),
        comprehension_correct=np.random.random() < 0.65,
        took_action=np.random.random() < 0.52,
        time_spent_seconds=np.random.exponential(25),
    ))

results = test.analyze()

print("A/B TEST RESULTS: Explanation Format")
print("=" * 60)
print(f"Variant A: Bar Chart (n=100)")
print(f"Variant B: Narrative (n=100)")

for metric, data in results.items():
    print(f"\n{metric.upper()}:")
    for key, value in data.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
```

---

## 6. Building a FastAPI Explanation Service

### 6.1 Complete Service Implementation

This section builds a production-ready explanation service combining all the
patterns discussed: synchronous serving, caching, pre-computation, and monitoring.

```python
"""
FastAPI Explanation Service with Redis Caching

This is a complete, production-ready explanation service that
integrates all the patterns discussed in this lesson:

1. Synchronous explanations for fast methods (feature importance)
2. Asynchronous explanations for slow methods (SHAP, LIME)
3. Redis caching with model-version-aware invalidation
4. Pre-computed explanations for common inputs
5. Drift monitoring endpoints
6. Health checks and metrics

ARCHITECTURE:
  Client -> FastAPI -> [Cache Check] -> [Model + Explainer] -> Response
                          |                     |
                          v                     v
                        Redis              Background Worker
                                               |
                                               v
                                          Explanation Store

DEPENDENCIES:
  pip install fastapi uvicorn redis numpy scikit-learn
"""

# Note: This code demonstrates the service structure.
# In production, you would run it with: uvicorn service:app --port 8000

from dataclasses import dataclass, field, asdict
from typing import Optional, Any
import json
import hashlib
import time
import uuid
import numpy as np


# ============================================================
# Data Models (Pydantic-style, using dataclasses for portability)
# ============================================================

@dataclass
class PredictionRequest:
    """Input to the prediction endpoint."""
    features: list[float]
    explain: bool = True          # Whether to include explanation
    method: str = "fast"          # "fast" (sync) or "detailed" (async)
    request_id: Optional[str] = None

    def __post_init__(self):
        if self.request_id is None:
            self.request_id = str(uuid.uuid4())


@dataclass
class PredictionResponse:
    """Output from the prediction endpoint."""
    request_id: str
    prediction: float
    prediction_label: str
    confidence: float
    model_version: str
    explanation: Optional[dict] = None
    explanation_id: Optional[str] = None  # For async retrieval
    cached: bool = False
    latency_ms: float = 0.0


@dataclass
class ExplanationResponse:
    """Output from the async explanation retrieval endpoint."""
    explanation_id: str
    status: str  # "pending", "completed", "failed"
    explanation: Optional[dict] = None
    method: str = ""
    computed_at: Optional[str] = None


@dataclass
class HealthResponse:
    """Output from the health check endpoint."""
    status: str
    model_version: str
    cache_hit_rate: float
    uptime_seconds: float
    total_predictions: int
    total_explanations: int


# ============================================================
# Explanation Service (combines all patterns)
# ============================================================

class ExplanationService:
    """Production explanation service.

    This class encapsulates the entire explanation serving logic:
    - Model inference
    - Fast explanations (synchronous)
    - Detailed explanations (asynchronous)
    - Caching with model-version invalidation
    - Pre-computation for common inputs
    - Monitoring and metrics
    """

    def __init__(self, model, feature_names: list[str], model_version: str):
        self.model = model
        self.feature_names = feature_names
        self.model_version = model_version
        self.start_time = time.time()

        # Cache (in production: Redis)
        self._cache: dict[str, dict] = {}
        self._cache_ttl = 3600  # 1 hour

        # Async explanation store
        self._async_store: dict[str, dict] = {}

        # Metrics
        self._metrics = {
            "total_predictions": 0,
            "total_explanations": 0,
            "cache_hits": 0,
            "cache_misses": 0,
        }

        # Drift monitoring data
        self._importance_history: list[dict] = []

    def _cache_key(self, features: list[float]) -> str:
        """Generate cache key from model version + features."""
        payload = json.dumps({
            "v": self.model_version,
            "f": [round(f, 6) for f in features],
        }, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()

    def _get_cached(self, features: list[float]) -> Optional[dict]:
        """Check cache for existing explanation."""
        key = self._cache_key(features)
        entry = self._cache.get(key)
        if entry and time.time() - entry["time"] < self._cache_ttl:
            self._metrics["cache_hits"] += 1
            return entry["explanation"]
        self._metrics["cache_misses"] += 1
        return None

    def _set_cached(self, features: list[float], explanation: dict) -> None:
        """Store explanation in cache."""
        key = self._cache_key(features)
        self._cache[key] = {"explanation": explanation, "time": time.time()}

    def predict(self, request: PredictionRequest) -> PredictionResponse:
        """Handle a prediction request with optional explanation.

        Flow:
        1. Generate prediction (always synchronous)
        2. If explain=True and method="fast": compute inline explanation
        3. If explain=True and method="detailed": queue async explanation
        4. Check cache before computing
        5. Update metrics
        """
        start = time.time()
        self._metrics["total_predictions"] += 1

        # Step 1: Prediction
        features_array = np.array(request.features).reshape(1, -1)
        proba = float(self.model.predict_proba(features_array)[0, 1])
        label = "approved" if proba >= 0.5 else "denied"
        confidence = abs(proba - 0.5) * 2

        response = PredictionResponse(
            request_id=request.request_id,
            prediction=round(proba, 4),
            prediction_label=label,
            confidence=round(confidence, 4),
            model_version=self.model_version,
        )

        # Step 2: Explanation
        if request.explain:
            self._metrics["total_explanations"] += 1

            if request.method == "fast":
                # Check cache first
                cached = self._get_cached(request.features)
                if cached:
                    response.explanation = cached
                    response.cached = True
                else:
                    # Compute fast explanation
                    explanation = self._compute_fast_explanation(features_array)
                    self._set_cached(request.features, explanation)
                    response.explanation = explanation

            elif request.method == "detailed":
                # Queue for async processing
                explanation_id = str(uuid.uuid4())
                self._async_store[explanation_id] = {
                    "status": "pending",
                    "features": request.features,
                    "request_id": request.request_id,
                }
                response.explanation_id = explanation_id

        response.latency_ms = round((time.time() - start) * 1000, 2)
        return response

    def _compute_fast_explanation(self, features: np.ndarray) -> dict:
        """Compute a fast (synchronous) explanation.

        Uses tree feature importance — O(1) per prediction.
        """
        importances = self.model.feature_importances_
        contributions = {}
        for name, imp in zip(self.feature_names, importances):
            contributions[name] = round(float(imp), 4)

        # Sort by importance
        contributions = dict(
            sorted(contributions.items(), key=lambda x: -abs(x[1]))
        )

        # Record for drift monitoring
        self._importance_history.append(contributions)

        return {
            "method": "feature_importance",
            "contributions": contributions,
            "top_factors": dict(list(contributions.items())[:3]),
        }

    def get_async_explanation(self, explanation_id: str) -> ExplanationResponse:
        """Retrieve an asynchronous explanation result."""
        entry = self._async_store.get(explanation_id)
        if entry is None:
            return ExplanationResponse(
                explanation_id=explanation_id,
                status="not_found",
            )
        return ExplanationResponse(
            explanation_id=explanation_id,
            status=entry["status"],
            explanation=entry.get("result"),
        )

    def health(self) -> HealthResponse:
        """Return service health status."""
        total_cache = self._metrics["cache_hits"] + self._metrics["cache_misses"]
        hit_rate = (
            self._metrics["cache_hits"] / total_cache
            if total_cache > 0 else 0.0
        )

        return HealthResponse(
            status="healthy",
            model_version=self.model_version,
            cache_hit_rate=round(hit_rate, 3),
            uptime_seconds=round(time.time() - self.start_time, 1),
            total_predictions=self._metrics["total_predictions"],
            total_explanations=self._metrics["total_explanations"],
        )

    def invalidate_cache(self) -> int:
        """Invalidate entire cache (called on model update)."""
        count = len(self._cache)
        self._cache.clear()
        return count


# ============================================================
# FastAPI Application Structure
# ============================================================

FASTAPI_APP_CODE = '''
"""
FastAPI application for the explanation service.

Run with: uvicorn app:app --host 0.0.0.0 --port 8000

Endpoints:
  POST /predict          - Get prediction with optional explanation
  GET  /explanation/{id} - Retrieve async explanation
  GET  /health           - Service health check
  POST /admin/invalidate - Invalidate explanation cache
  GET  /metrics          - Prometheus-compatible metrics
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import redis
import joblib

app = FastAPI(title="Explanation Service", version="1.0.0")

# Initialize service (in production, load from model registry)
# model = joblib.load("model.pkl")
# service = ExplanationService(model, feature_names, model_version)

@app.post("/predict")
async def predict(request: PredictionRequest):
    """Generate prediction with optional explanation.

    Query parameters:
    - explain: bool (default True) - include explanation
    - method: str (default "fast") - "fast" or "detailed"
    """
    return service.predict(request)

@app.get("/explanation/{explanation_id}")
async def get_explanation(explanation_id: str):
    """Retrieve an asynchronous explanation by ID."""
    result = service.get_async_explanation(explanation_id)
    if result.status == "not_found":
        raise HTTPException(status_code=404, detail="Explanation not found")
    return result

@app.get("/health")
async def health():
    """Service health check."""
    return service.health()

@app.post("/admin/invalidate")
async def invalidate_cache():
    """Invalidate the explanation cache (admin only)."""
    count = service.invalidate_cache()
    return {"invalidated": count}
'''


# ============================================================
# Demonstrate the service
# ============================================================

service = ExplanationService(model, feature_names, "v2.3.1")

print("FASTAPI EXPLANATION SERVICE DEMO")
print("=" * 60)

# Simulate API requests
for i in range(8):
    request = PredictionRequest(
        features=X_test[i % 4].tolist(),  # Repeat to demonstrate caching
        explain=True,
        method="fast",
    )
    response = service.predict(request)
    print(f"\nRequest {i+1}:")
    print(f"  Prediction: {response.prediction_label} ({response.prediction})")
    print(f"  Cached: {response.cached}")
    print(f"  Latency: {response.latency_ms}ms")
    if response.explanation:
        top = response.explanation["top_factors"]
        print(f"  Top factors: {top}")

# Health check
health = service.health()
print(f"\nHealth Check:")
print(f"  Status: {health.status}")
print(f"  Model: {health.model_version}")
print(f"  Cache hit rate: {health.cache_hit_rate:.1%}")
print(f"  Total predictions: {health.total_predictions}")
print(f"  Total explanations: {health.total_explanations}")
```

---

## Summary

- **Explanation serving** requires careful architecture because explanations are often 100-1000x more expensive than predictions — three patterns (synchronous, asynchronous, hybrid) address different latency requirements
- **Caching** is essential for production efficiency: hash the (model_version, input_features) pair as the cache key, and invalidate the entire cache when the model is updated to prevent serving stale explanations
- **Pre-computation** of explanations for cluster centroids and common inputs enables expensive methods (SHAP, LIME, counterfactuals) without real-time latency penalties
- **Explanation drift monitoring** uses statistical tests (KS test) to detect changes in feature importance distributions over time, providing early warning of data drift, concept drift, or model degradation
- **MLOps integration** treats explanations as first-class artifacts: versioned, stored, validated in CI/CD pipelines, and tracked alongside model metrics in registries like MLflow
- **A/B testing explanations** applies UX experimentation to determine which explanation format (bar chart vs. narrative, technical vs. plain language) best serves users' trust, comprehension, and action needs
- **Alert systems** should trigger on both prediction drift AND explanation drift, as explanation changes can precede accuracy degradation
- A **complete explanation service** (demonstrated with FastAPI + Redis) combines all these patterns into a production-ready system with health checks, metrics, and administrative endpoints

---

## Exercises

### Exercise 1: Latency Budget Analysis

Given three ML applications with different SLAs, design the optimal explanation architecture:

1. **Real-time fraud detection** (50ms SLA, 10,000 requests/second): Which explanation method and pattern would you use? What caching strategy?
2. **Loan approval system** (5-second SLA, 100 requests/minute, regulatory requirement for explanation): How would you ensure every decision has a compliant explanation?
3. **Medical image analysis** (30-second SLA, 10 requests/day, high-stakes): What explanation depth is appropriate? How would you handle the clinician's need for detailed explanations?

For each, specify: architecture pattern, explanation method, caching strategy, and expected hit rate.

### Exercise 2: Build a Cache Invalidation System

Implement a more sophisticated cache invalidation system that handles:

1. **Model version changes**: Invalidate all entries for the old version
2. **Feature schema changes**: Detect when feature names or order change
3. **Partial invalidation**: Invalidate entries matching a feature value range (e.g., all entries where `income > 100000`)
4. **TTL with jitter**: Add random jitter to TTL to prevent thundering herd on expiration
5. **Cache warming**: After invalidation, pre-populate cache with explanations for the most frequently accessed inputs

### Exercise 3: Drift Detection Pipeline

Build a complete drift detection pipeline that:

1. Collects feature importance vectors from the explanation service
2. Computes rolling statistics (mean, std, percentiles) over 1-hour windows
3. Applies multiple statistical tests (KS, Chi-squared, PSI) to detect drift
4. Generates alerts with severity levels (info, warning, critical)
5. Produces a weekly drift report with visualizations (matplotlib)
6. Implements automatic model retraining trigger when drift exceeds a threshold

### Exercise 4: Explanation A/B Test Design

Design and implement a complete A/B test comparing three explanation formats for a credit scoring system:

1. **Variant A**: Top-3 feature importance bar chart
2. **Variant B**: Natural language paragraph explaining the decision
3. **Variant C**: Counterfactual explanation ("If X were Y, the decision would change")

Define metrics, calculate required sample size for statistical power, implement random assignment, and build the analysis pipeline. Run the test on simulated user data and report results.

### Exercise 5: Production Service Extension

Extend the FastAPI explanation service to include:

1. **Rate limiting**: Maximum 100 explanation requests per user per minute
2. **Explanation versioning**: Store and retrieve historical explanations by timestamp
3. **Webhook callbacks**: Notify clients when async explanations complete
4. **Batch endpoint**: Accept multiple prediction requests and return explanations in batch
5. **Prometheus metrics**: Export request count, latency histogram, cache hit rate, and error rate as Prometheus-compatible metrics

---

[Previous: AI Regulation and Governance](./13_AI_Regulation_and_Governance.md) | [Overview](./00_Overview.md) | [Next: Domain-Specific Interpretability](./15_Domain_Specific_Interpretability.md)

**License**: CC BY-NC 4.0
