"""
Exercise Solutions: Feature Stores
===========================================
Lesson 11 from MLOps topic.

Exercises
---------
1. Feature Store Setup — Build a simulated feature store with entity
   definitions, feature views, and offline/online stores.
2. Training Data Generation — Generate point-in-time correct training
   datasets from the feature store.
3. Online Serving — Implement low-latency online feature retrieval
   with caching and freshness guarantees.
"""

import random
import time
import json
import hashlib
from datetime import datetime, timedelta


# ============================================================
# Exercise 1: Feature Store Setup
# ============================================================

def exercise_1_feature_store_setup():
    """Build a simulated feature store with core concepts.

    A Feature Store provides:
    - Entity: The primary key for looking up features (e.g., customer_id)
    - Feature View: A logical grouping of features from a data source
    - Offline Store: Historical features for training (batch retrieval)
    - Online Store: Latest features for serving (low-latency retrieval)
    - Materialization: Process of copying features from offline to online store
    """

    class Entity:
        """Defines the primary key for feature lookups."""
        def __init__(self, name, join_key, value_type="INT64"):
            self.name = name
            self.join_key = join_key
            self.value_type = value_type

    class Feature:
        """A single feature with name and type."""
        def __init__(self, name, dtype="FLOAT64", description=""):
            self.name = name
            self.dtype = dtype
            self.description = description

    class FeatureView:
        """Logical grouping of features from a data source."""
        def __init__(self, name, entity, features, ttl_hours=24, source=""):
            self.name = name
            self.entity = entity
            self.features = features
            self.ttl_hours = ttl_hours
            self.source = source

    class OfflineStore:
        """Historical feature storage (simulates Parquet/BigQuery)."""
        def __init__(self):
            self.data = {}  # (entity_id, timestamp) -> {features}

        def write(self, entity_id, features, timestamp):
            key = (entity_id, timestamp.isoformat())
            self.data[key] = features

        def get_historical(self, entity_id, start_time=None, end_time=None):
            results = []
            for (eid, ts_str), features in self.data.items():
                if eid != entity_id:
                    continue
                ts = datetime.fromisoformat(ts_str)
                if start_time and ts < start_time:
                    continue
                if end_time and ts > end_time:
                    continue
                results.append({"timestamp": ts, "features": features})
            return sorted(results, key=lambda x: x["timestamp"])

        def get_point_in_time(self, entity_id, event_time):
            """Get the latest features at or before event_time."""
            candidates = []
            for (eid, ts_str), features in self.data.items():
                if eid != entity_id:
                    continue
                ts = datetime.fromisoformat(ts_str)
                if ts <= event_time:
                    candidates.append({"timestamp": ts, "features": features})
            if not candidates:
                return None
            return max(candidates, key=lambda x: x["timestamp"])

    class OnlineStore:
        """Latest feature storage (simulates Redis/DynamoDB)."""
        def __init__(self):
            self.data = {}  # entity_id -> {features, updated_at}

        def write(self, entity_id, features, timestamp):
            current = self.data.get(entity_id)
            if current is None or timestamp > current["updated_at"]:
                self.data[entity_id] = {
                    "features": features,
                    "updated_at": timestamp,
                }

        def get(self, entity_id):
            entry = self.data.get(entity_id)
            if entry:
                return {
                    "features": entry["features"],
                    "updated_at": entry["updated_at"],
                    "freshness_seconds": (datetime.now() - entry["updated_at"]).total_seconds(),
                }
            return None

    class FeatureStore:
        """Complete feature store with offline + online stores."""
        def __init__(self, name):
            self.name = name
            self.entities = {}
            self.feature_views = {}
            self.offline_store = OfflineStore()
            self.online_store = OnlineStore()
            self.materialization_log = []

        def register_entity(self, entity):
            self.entities[entity.name] = entity

        def register_feature_view(self, feature_view):
            self.feature_views[feature_view.name] = feature_view

        def ingest(self, feature_view_name, entity_id, features, timestamp):
            """Ingest features into offline store."""
            self.offline_store.write(entity_id, features, timestamp)

        def materialize(self, feature_view_name, start_time, end_time):
            """Copy features from offline to online store (latest values)."""
            fv = self.feature_views[feature_view_name]
            materialized = 0

            for (eid, ts_str), features in self.offline_store.data.items():
                ts = datetime.fromisoformat(ts_str)
                if start_time <= ts <= end_time:
                    self.online_store.write(eid, features, ts)
                    materialized += 1

            self.materialization_log.append({
                "feature_view": feature_view_name,
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "records_materialized": materialized,
                "materialized_at": datetime.now().isoformat(),
            })
            return materialized

    # --- Build the feature store ---
    fs = FeatureStore("customer_churn_features")

    # Define entities
    customer = Entity("customer", "customer_id", "INT64")
    fs.register_entity(customer)

    # Define feature views
    profile_features = FeatureView(
        name="customer_profile",
        entity=customer,
        features=[
            Feature("age", "INT64", "Customer age"),
            Feature("tenure_months", "INT64", "Months since signup"),
            Feature("contract_type", "STRING", "Contract: month-to-month/1yr/2yr"),
        ],
        ttl_hours=168,  # Weekly update
        source="customer_db.profiles",
    )

    usage_features = FeatureView(
        name="customer_usage",
        entity=customer,
        features=[
            Feature("monthly_charges", "FLOAT64", "Monthly bill amount"),
            Feature("total_charges", "FLOAT64", "Cumulative charges"),
            Feature("avg_daily_minutes", "FLOAT64", "Avg daily usage minutes"),
            Feature("support_calls_30d", "INT64", "Support calls in last 30 days"),
        ],
        ttl_hours=24,  # Daily update
        source="usage_events_stream",
    )

    fs.register_feature_view(profile_features)
    fs.register_feature_view(usage_features)

    # --- Populate with sample data ---
    random.seed(42)
    base_time = datetime(2025, 1, 1)

    for customer_id in range(1, 51):
        for day in range(0, 90, 7):  # Weekly profile snapshots
            ts = base_time + timedelta(days=day)
            fs.ingest("customer_profile", customer_id, {
                "age": 25 + customer_id % 40,
                "tenure_months": 6 + day // 30,
                "contract_type": random.choice(["month-to-month", "one-year", "two-year"]),
            }, ts)

        for day in range(90):  # Daily usage
            ts = base_time + timedelta(days=day)
            fs.ingest("customer_usage", customer_id, {
                "monthly_charges": 50 + random.gauss(20, 10),
                "total_charges": 500 + day * 2 + random.gauss(0, 5),
                "avg_daily_minutes": 30 + random.gauss(10, 5),
                "support_calls_30d": random.randint(0, 5),
            }, ts)

    # Materialize to online store
    materialized = fs.materialize(
        "customer_usage",
        base_time + timedelta(days=85),
        base_time + timedelta(days=90),
    )

    # --- Display results ---
    print("Feature Store Setup")
    print("=" * 60)

    print(f"\n  Store: {fs.name}")
    print(f"  Entities: {list(fs.entities.keys())}")
    print(f"  Feature Views:")
    for fv_name, fv in fs.feature_views.items():
        feature_list = [f.name for f in fv.features]
        print(f"    {fv_name}: {feature_list} (TTL: {fv.ttl_hours}h)")
    print(f"\n  Offline records: {len(fs.offline_store.data)}")
    print(f"  Online records: {len(fs.online_store.data)}")
    print(f"  Materialized: {materialized} records")

    # Example lookups
    print(f"\n  Online lookup (customer 5):")
    result = fs.online_store.get(5)
    if result:
        print(f"    Features: {json.dumps(result['features'], indent=4)}")
        print(f"    Updated: {result['updated_at']}")

    print(f"\n  Historical lookup (customer 5, last 7 days):")
    hist = fs.offline_store.get_historical(
        5,
        base_time + timedelta(days=83),
        base_time + timedelta(days=90),
    )
    for h in hist[:3]:
        print(f"    {h['timestamp']}: charges={h['features'].get('monthly_charges', 'N/A'):.1f}")

    return fs


# ============================================================
# Exercise 2: Training Data Generation
# ============================================================

def exercise_2_training_data():
    """Generate point-in-time correct training datasets.

    Point-in-time correctness means: for each training example, features
    are as they were at the time the label was generated — not as they
    are now. This prevents data leakage.

    Example: If a customer churned on March 15, their features should
    be from March 14 or earlier, not from March 16.
    """

    class TrainingDataGenerator:
        """Generate point-in-time correct training datasets."""

        def __init__(self, feature_store_data):
            self.feature_store = feature_store_data

        def get_features_at_time(self, entity_id, event_time, feature_names):
            """Get features as of event_time (no future leakage)."""
            entity_history = self.feature_store.get(entity_id, [])
            # Find the most recent record at or before event_time
            candidates = [r for r in entity_history if r["timestamp"] <= event_time]
            if not candidates:
                return None
            latest = max(candidates, key=lambda r: r["timestamp"])
            return {k: latest["features"].get(k) for k in feature_names}

        def generate_dataset(self, label_events, feature_names, lookback_buffer_hours=0):
            """Generate a training dataset from label events and feature store.

            Args:
                label_events: List of {"entity_id": ..., "timestamp": ..., "label": ...}
                feature_names: Features to retrieve
                lookback_buffer_hours: Additional buffer before event time

            Returns:
                Training dataset with point-in-time correct features
            """
            dataset = []
            skipped = 0

            for event in label_events:
                event_time = event["timestamp"] - timedelta(hours=lookback_buffer_hours)
                features = self.get_features_at_time(
                    event["entity_id"], event_time, feature_names
                )
                if features is None:
                    skipped += 1
                    continue

                row = {
                    "entity_id": event["entity_id"],
                    "event_timestamp": event["timestamp"].isoformat(),
                    "feature_timestamp": event_time.isoformat(),
                    **features,
                    "label": event["label"],
                }
                dataset.append(row)

            return dataset, skipped

    # --- Build simulated feature store data ---
    random.seed(42)
    base_time = datetime(2025, 1, 1)

    feature_store_data = {}
    for cid in range(1, 101):
        history = []
        for day in range(90):
            ts = base_time + timedelta(days=day)
            history.append({
                "timestamp": ts,
                "features": {
                    "tenure_months": 6 + day // 30,
                    "monthly_charges": 50 + random.gauss(20, 10),
                    "total_charges": 500 + day * 2,
                    "support_calls_30d": random.randint(0, 5),
                    "avg_daily_minutes": 30 + random.gauss(10, 5),
                    "contract_type_encoded": random.choice([0, 1, 2]),
                },
            })
        feature_store_data[cid] = history

    # --- Generate label events (churn observations) ---
    label_events = []
    for cid in range(1, 101):
        # Observation at day 60
        event_time = base_time + timedelta(days=60)
        # Churn probability influenced by support calls
        last_features = feature_store_data[cid][-30]["features"]
        churn_prob = 0.1 + 0.15 * last_features["support_calls_30d"] / 5
        churned = 1 if random.random() < churn_prob else 0

        label_events.append({
            "entity_id": cid,
            "timestamp": event_time,
            "label": churned,
        })

    # --- Generate training data ---
    generator = TrainingDataGenerator(feature_store_data)
    feature_names = [
        "tenure_months", "monthly_charges", "total_charges",
        "support_calls_30d", "avg_daily_minutes", "contract_type_encoded",
    ]

    print("Training Data Generation")
    print("=" * 60)

    # Without buffer (features at exact event time)
    dataset_no_buffer, skipped_no = generator.generate_dataset(
        label_events, feature_names, lookback_buffer_hours=0
    )

    # With 24h buffer (features from 24h before event)
    dataset_with_buffer, skipped_buf = generator.generate_dataset(
        label_events, feature_names, lookback_buffer_hours=24
    )

    print(f"\n  Point-in-Time Dataset (no buffer):")
    print(f"    Samples: {len(dataset_no_buffer)}")
    print(f"    Skipped (no features): {skipped_no}")
    print(f"    Churn rate: {sum(r['label'] for r in dataset_no_buffer) / len(dataset_no_buffer):.2%}")

    print(f"\n  Point-in-Time Dataset (24h buffer):")
    print(f"    Samples: {len(dataset_with_buffer)}")
    print(f"    Skipped: {skipped_buf}")
    print(f"    Churn rate: {sum(r['label'] for r in dataset_with_buffer) / len(dataset_with_buffer):.2%}")

    # Show sample rows
    print(f"\n  Sample Rows (first 5):")
    print(f"  {'CID':>5s} {'Label':>6s} {'Tenure':>7s} {'Charges':>9s} {'Support':>8s}")
    print(f"  {'-'*40}")
    for row in dataset_with_buffer[:5]:
        print(f"  {row['entity_id']:>5d} {row['label']:>6d} "
              f"{row['tenure_months']:>7d} {row['monthly_charges']:>9.1f} "
              f"{row['support_calls_30d']:>8d}")

    # Demonstrate point-in-time correctness
    print(f"\n  Point-in-Time Correctness Check:")
    sample = dataset_with_buffer[0]
    print(f"    Entity: {sample['entity_id']}")
    print(f"    Event time: {sample['event_timestamp']}")
    print(f"    Feature time: {sample['feature_timestamp']}")
    print(f"    -> Features are from BEFORE the label event (no leakage)")

    return dataset_with_buffer


# ============================================================
# Exercise 3: Online Serving
# ============================================================

def exercise_3_online_serving():
    """Implement low-latency online feature retrieval.

    Online serving requirements:
    - p99 latency < 10ms
    - Feature freshness < TTL
    - Graceful degradation when features are stale or missing
    - Caching for frequently accessed entities
    """

    class LRUCache:
        """Simple LRU cache for feature lookups."""
        def __init__(self, max_size=1000):
            self.max_size = max_size
            self.cache = {}
            self.access_order = []
            self.hits = 0
            self.misses = 0

        def get(self, key):
            if key in self.cache:
                self.hits += 1
                self.access_order.remove(key)
                self.access_order.append(key)
                return self.cache[key]
            self.misses += 1
            return None

        def put(self, key, value):
            if key in self.cache:
                self.access_order.remove(key)
            elif len(self.cache) >= self.max_size:
                oldest = self.access_order.pop(0)
                del self.cache[oldest]
            self.cache[key] = value
            self.access_order.append(key)

        @property
        def hit_rate(self):
            total = self.hits + self.misses
            return self.hits / total if total > 0 else 0

    class OnlineFeatureServer:
        """Low-latency feature serving with caching and freshness checks."""

        def __init__(self, feature_store, cache_size=500, max_ttl_hours=24):
            self.feature_store = feature_store
            self.cache = LRUCache(max_size=cache_size)
            self.max_ttl_hours = max_ttl_hours
            self.total_requests = 0
            self.stale_features_served = 0
            self.missing_features = 0
            self.latencies = []

        def get_features(self, entity_id, feature_names, allow_stale=True):
            """Retrieve features for online serving.

            Args:
                entity_id: Entity to look up
                feature_names: Required features
                allow_stale: If True, serve stale features with a warning
            """
            self.total_requests += 1
            start = time.time()

            # Check cache first
            cache_key = f"{entity_id}:{','.join(sorted(feature_names))}"
            cached = self.cache.get(cache_key)
            if cached and not self._is_stale(cached["updated_at"]):
                latency_ms = (time.time() - start) * 1000
                self.latencies.append(latency_ms)
                return {
                    "features": cached["features"],
                    "source": "cache",
                    "latency_ms": round(latency_ms, 3),
                    "fresh": True,
                }

            # Cache miss or stale — look up from store
            entry = self.feature_store.get(entity_id)
            if entry is None:
                self.missing_features += 1
                latency_ms = (time.time() - start) * 1000
                self.latencies.append(latency_ms)
                return {
                    "features": {name: None for name in feature_names},
                    "source": "default",
                    "latency_ms": round(latency_ms, 3),
                    "fresh": False,
                    "warning": "Entity not found, serving defaults",
                }

            features = {k: entry["features"].get(k) for k in feature_names}
            is_fresh = not self._is_stale(entry["updated_at"])

            if not is_fresh:
                self.stale_features_served += 1
                if not allow_stale:
                    latency_ms = (time.time() - start) * 1000
                    self.latencies.append(latency_ms)
                    return {
                        "features": {name: None for name in feature_names},
                        "source": "rejected",
                        "latency_ms": round(latency_ms, 3),
                        "fresh": False,
                        "warning": "Features too stale, serving defaults",
                    }

            # Update cache
            self.cache.put(cache_key, {
                "features": features,
                "updated_at": entry["updated_at"],
            })

            latency_ms = (time.time() - start) * 1000
            self.latencies.append(latency_ms)

            result = {
                "features": features,
                "source": "store",
                "latency_ms": round(latency_ms, 3),
                "fresh": is_fresh,
            }
            if not is_fresh:
                staleness = (datetime.now() - entry["updated_at"]).total_seconds() / 3600
                result["warning"] = f"Features are {staleness:.1f}h old (TTL={self.max_ttl_hours}h)"

            return result

        def _is_stale(self, updated_at):
            age_hours = (datetime.now() - updated_at).total_seconds() / 3600
            return age_hours > self.max_ttl_hours

        def get_stats(self):
            latencies_sorted = sorted(self.latencies) if self.latencies else [0]
            n = len(latencies_sorted)
            return {
                "total_requests": self.total_requests,
                "cache_hit_rate": round(self.cache.hit_rate, 4),
                "stale_features_served": self.stale_features_served,
                "missing_features": self.missing_features,
                "latency_p50_ms": round(latencies_sorted[int(n * 0.5)], 3) if n > 0 else 0,
                "latency_p99_ms": round(latencies_sorted[min(int(n * 0.99), n - 1)], 3) if n > 0 else 0,
                "latency_mean_ms": round(sum(latencies_sorted) / n, 3) if n > 0 else 0,
            }

    # --- Build simulated online store ---
    random.seed(42)

    online_store = {}
    now = datetime.now()
    for cid in range(1, 201):
        # Some features are fresh, some are stale
        hours_ago = random.choice([1, 2, 6, 12, 24, 48, 72])
        online_store[cid] = {
            "features": {
                "tenure_months": random.randint(1, 60),
                "monthly_charges": round(50 + random.gauss(20, 10), 2),
                "total_charges": round(random.uniform(100, 5000), 2),
                "support_calls_30d": random.randint(0, 5),
                "contract_type_encoded": random.choice([0, 1, 2]),
            },
            "updated_at": now - timedelta(hours=hours_ago),
        }

    class SimpleOnlineStore:
        def get(self, entity_id):
            return online_store.get(entity_id)

    server = OnlineFeatureServer(
        SimpleOnlineStore(), cache_size=100, max_ttl_hours=24
    )

    print("Online Feature Serving")
    print("=" * 60)

    feature_names = ["tenure_months", "monthly_charges", "support_calls_30d"]

    # Simulate serving requests
    print("\n  Sample Requests:")
    print("-" * 60)

    for cid in [1, 5, 10, 999]:  # Mix of existing and non-existing
        result = server.get_features(cid, feature_names)
        print(f"\n  Customer {cid}:")
        print(f"    Source: {result['source']}")
        print(f"    Fresh: {result['fresh']}")
        print(f"    Latency: {result['latency_ms']:.3f}ms")
        if "warning" in result:
            print(f"    Warning: {result['warning']}")
        print(f"    Features: {json.dumps(result['features'])}")

    # Repeat request to test cache
    print("\n  Cache Test (repeat customer 5):")
    result = server.get_features(5, feature_names)
    print(f"    Source: {result['source']}")
    print(f"    Latency: {result['latency_ms']:.3f}ms")

    # High-volume simulation
    print("\n  High-Volume Simulation (1000 requests):")
    print("-" * 40)
    random.seed(42)
    for _ in range(1000):
        # Zipf-like distribution: some entities accessed much more than others
        cid = random.choice([1, 1, 1, 2, 2, 3, 5, 10, 20, random.randint(1, 200)])
        server.get_features(cid, feature_names)

    stats = server.get_stats()
    print(f"  Total requests: {stats['total_requests']}")
    print(f"  Cache hit rate: {stats['cache_hit_rate']:.1%}")
    print(f"  Latency p50: {stats['latency_p50_ms']:.3f}ms")
    print(f"  Latency p99: {stats['latency_p99_ms']:.3f}ms")
    print(f"  Stale features served: {stats['stale_features_served']}")
    print(f"  Missing features: {stats['missing_features']}")

    return server


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    print("Exercise 1: Feature Store Setup")
    print("=" * 60)
    exercise_1_feature_store_setup()

    print("\n\n")
    print("Exercise 2: Training Data Generation")
    print("=" * 60)
    exercise_2_training_data()

    print("\n\n")
    print("Exercise 3: Online Serving")
    print("=" * 60)
    exercise_3_online_serving()
