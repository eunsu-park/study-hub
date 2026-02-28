"""
Exercise Solutions: Lesson 16 - Kafka Streams & ksqlDB

Covers:
  - Exercise 1: Real-Time Transaction Aggregator (tumbling/hopping/session windows)
  - Exercise 2: KStream-KTable Enrichment Pipeline
  - Exercise 3: ksqlDB Fraud Detection Pipeline (SQL)
  - Exercise 4: Dead Letter Queue and Error Handling
  - Exercise 5: Multi-Source Join and Session Analytics

Note: Simulates Faust/Kafka Streams concepts in pure Python.
"""

import time
import random
from collections import defaultdict
from datetime import datetime, timedelta
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Simulated Faust Records and Tables
# ---------------------------------------------------------------------------

@dataclass
class Transaction:
    """Faust Record: faust.Record equivalent."""
    user_id: str
    merchant: str
    amount: float
    category: str
    timestamp: datetime


@dataclass
class Order:
    order_id: str
    user_id: str
    product_id: str
    amount: float


@dataclass
class UserProfile:
    user_id: str
    name: str
    tier: str
    credit_limit: float


class WindowedTable:
    """Simulates a Faust windowed table (KTable with time windows).

    expires must be set to prevent unbounded state growth. Without it,
    RocksDB (Faust's state backend) accumulates windows indefinitely,
    eventually consuming all disk space and causing OOM.
    """
    def __init__(self, name: str, window_size_sec: int, expires_sec: int):
        self.name = name
        self.window_size_sec = window_size_sec
        self.expires_sec = expires_sec
        # key -> {window_key -> value}
        self.data: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))

    def _window_key(self, ts: datetime) -> str:
        epoch = int(ts.timestamp())
        window_start = epoch - (epoch % self.window_size_sec)
        return datetime.fromtimestamp(window_start).strftime("%H:%M:%S")

    def update(self, key: str, value: float, ts: datetime) -> float:
        wk = self._window_key(ts)
        self.data[key][wk] += value
        return self.data[key][wk]

    def get(self, key: str, ts: datetime) -> float:
        wk = self._window_key(ts)
        return self.data[key].get(wk, 0.0)


class KTable:
    """Simulates a Faust KTable (changelog-backed lookup table).

    Why KTable for lookups (not KStream):
    - A KTable represents the LATEST state per key (upsert semantics).
    - Reading from a KTable gives the current value, not a stream of all changes.
    - This is the correct abstraction for enrichment lookups (user profiles, etc.).
    - A KStream would require windowed joins, which are more complex and
      don't give point-in-time lookups.
    """
    def __init__(self, name: str):
        self.name = name
        self.data: dict[str, dict] = {}

    def update(self, key: str, value: dict) -> None:
        self.data[key] = value

    def get(self, key: str) -> dict | None:
        return self.data.get(key)


# ---------------------------------------------------------------------------
# Exercise 1: Real-Time Transaction Aggregator
# ---------------------------------------------------------------------------

def exercise1_transaction_aggregator():
    """Three simultaneous windowed aggregations on the same stream:
    1. Tumbling 5-min: per-user total spending
    2. Hopping 10-min/2-min step: per-merchant transaction count
    3. Session 30-sec gap: transactions per user session
    """
    # Create windowed tables
    # expires prevents unbounded state: old windows are discarded after this duration
    tumbling_user = WindowedTable("user_spending_5min", window_size_sec=300, expires_sec=3600)
    hopping_merchant = WindowedTable("merchant_count_10min", window_size_sec=120, expires_sec=3600)
    session_user: dict[str, dict] = defaultdict(lambda: {"count": 0, "last_ts": None})

    # Generate transactions
    base_time = datetime(2024, 11, 15, 14, 0, 0)
    merchants = ["Amazon", "Walmart", "BestBuy", "Target"]
    categories = ["electronics", "groceries", "clothing"]

    print("\n  Processing transactions:")
    alerts = []

    for i in range(30):
        ts = base_time + timedelta(seconds=random.randint(0, 600))
        txn = Transaction(
            user_id=f"user_{random.randint(1, 5)}",
            merchant=random.choice(merchants),
            amount=round(random.uniform(10, 200), 2),
            category=random.choice(categories),
            timestamp=ts,
        )

        # 1. Tumbling 5-min: per-user spending
        total = tumbling_user.update(txn.user_id, txn.amount, txn.timestamp)
        if total > 500:
            alert = f"ALERT: {txn.user_id} spent ${total:.2f} in 5-min window"
            alerts.append(alert)

        # 2. Hopping: per-merchant count (simplified as tumbling with smaller window)
        hopping_merchant.update(txn.merchant, 1, txn.timestamp)

        # 3. Session: count per user session (30-sec gap)
        sess = session_user[txn.user_id]
        if sess["last_ts"] and (txn.timestamp - sess["last_ts"]).seconds > 30:
            sess["count"] = 1  # New session
        else:
            sess["count"] += 1
        sess["last_ts"] = txn.timestamp

    # Display results
    print(f"\n  Tumbling 5-min: Per-User Total Spending")
    for user_id in sorted(tumbling_user.data.keys()):
        windows = tumbling_user.data[user_id]
        for wk, total in sorted(windows.items()):
            flag = " ** ALERT" if total > 500 else ""
            print(f"    {user_id} [{wk}]: ${total:.2f}{flag}")

    print(f"\n  Hopping 2-min: Per-Merchant Transaction Count")
    for merchant in sorted(hopping_merchant.data.keys()):
        windows = hopping_merchant.data[merchant]
        for wk, count in sorted(windows.items()):
            print(f"    {merchant} [{wk}]: {int(count)} txns")

    print(f"\n  Session (30s gap): Per-User Session Counts")
    for uid, sess in sorted(session_user.items()):
        print(f"    {uid}: {sess['count']} txns in current session")

    if alerts:
        print(f"\n  Spending Alerts ({len(alerts)}):")
        for a in alerts:
            print(f"    {a}")

    # HTTP endpoint simulation
    print(f"\n  HTTP endpoint /spending/ would return:")
    print(f"    {dict(tumbling_user.data)}")

    return tumbling_user, hopping_merchant


# ---------------------------------------------------------------------------
# Exercise 2: KStream-KTable Enrichment Pipeline
# ---------------------------------------------------------------------------

def exercise2_enrichment_pipeline():
    """Order enrichment with user profile KTable + dead letter queue."""
    # Setup KTable with user profiles
    profiles = KTable("user_profiles")
    profiles.update("u1", {"name": "Alice", "tier": "gold", "credit_limit": 5000})
    profiles.update("u2", {"name": "Bob", "tier": "silver", "credit_limit": 2000})
    profiles.update("u3", {"name": "Carol", "tier": "bronze", "credit_limit": 500})

    # Order events (KStream)
    orders = [
        Order("ord-1", "u1", "p101", 450.00),
        Order("ord-2", "u2", "p102", 2500.00),  # Exceeds credit limit
        Order("ord-3", "u99", "p103", 100.00),  # Missing profile
        Order("ord-4", "u3", "p104", 300.00),
        Order("ord-5", "u1", "p105", 6000.00),  # Exceeds credit limit
        Order("ord-6", "u88", "p106", 50.00),   # Missing profile
    ]

    enriched_orders = []
    missing_profile_dlq = []  # Dead letter queue for missing profiles
    credit_alerts = []

    # Per-tier credit alert counts (tumbling 1-hour table)
    tier_alert_counts: dict[str, int] = defaultdict(int)

    print("\n  Order Enrichment Pipeline:")
    for order in orders:
        profile = profiles.get(order.user_id)

        if profile is None:
            # Route to dead letter queue
            missing_profile_dlq.append({
                "order_id": order.order_id,
                "user_id": order.user_id,
                "reason": "missing_profile",
            })
            print(f"    {order.order_id}: user={order.user_id} -> MISSING PROFILE (DLQ)")
            continue

        # Enrich with profile data
        enriched = {
            "order_id": order.order_id,
            "user_id": order.user_id,
            "user_name": profile["name"],
            "tier": profile["tier"],
            "amount": order.amount,
        }

        if order.amount > profile["credit_limit"]:
            credit_alerts.append({
                "order_id": order.order_id,
                "user_id": order.user_id,
                "amount": order.amount,
                "credit_limit": profile["credit_limit"],
            })
            tier_alert_counts[profile["tier"]] += 1
            print(f"    {order.order_id}: {profile['name']} ${order.amount} "
                  f"> credit_limit ${profile['credit_limit']} -> CREDIT ALERT")
        else:
            enriched_orders.append(enriched)
            print(f"    {order.order_id}: {profile['name']} (tier={profile['tier']}) "
                  f"${order.amount} -> enriched")

    print(f"\n  Results:")
    print(f"    Enriched orders: {len(enriched_orders)}")
    print(f"    Missing profile (DLQ): {len(missing_profile_dlq)}")
    print(f"    Credit alerts: {len(credit_alerts)}")
    print(f"    Alert counts by tier: {dict(tier_alert_counts)}")

    return enriched_orders, missing_profile_dlq, credit_alerts


# ---------------------------------------------------------------------------
# Exercise 3: ksqlDB Fraud Detection Pipeline
# ---------------------------------------------------------------------------

def exercise3_ksqldb_fraud():
    """ksqlDB pipeline for detecting suspicious transactions.

    Each step is annotated with the ksqlDB SQL equivalent.
    """
    print("\n  ksqlDB Fraud Detection Pipeline (SQL equivalents):\n")

    # Step 1: Create source STREAM with event-time semantics
    # STREAM = unbounded, append-only, ordered by event time
    sql_step1 = """
    -- Step 1: STREAM because transactions are immutable events (insert-only).
    -- TIMESTAMP on txn_time enables event-time windowed operations.
    CREATE STREAM raw_transactions (
        txn_id VARCHAR KEY,
        user_id VARCHAR,
        amount DOUBLE,
        merchant VARCHAR,
        country VARCHAR,
        txn_time TIMESTAMP
    ) WITH (
        KAFKA_TOPIC = 'raw_transactions',
        VALUE_FORMAT = 'JSON',
        TIMESTAMP = 'txn_time'  -- Event-time semantics
    );"""
    print(f"  {sql_step1}")

    # Step 2: Create TABLE of per-user rolling 1-hour spending
    # TABLE = materialized view, latest value per key (upsert semantics)
    sql_step2 = """
    -- Step 2: TABLE because we want the LATEST rolling total per user (upsert).
    -- EMIT CHANGES makes this a continuous, incrementally updated materialized view.
    CREATE TABLE user_hourly_spending AS
    SELECT
        user_id,
        SUM(amount) AS total_1h,
        COUNT(*) AS txn_count_1h
    FROM raw_transactions
    WINDOW TUMBLING (SIZE 1 HOUR)
    GROUP BY user_id
    EMIT CHANGES;"""
    print(f"  {sql_step2}")

    # Step 3: STREAM of suspicious transactions
    sql_step3 = """
    -- Step 3: STREAM because alerts are events we want to capture.
    -- Combines single-txn and rolling-total conditions.
    CREATE STREAM fraud_alerts AS
    SELECT
        t.txn_id,
        t.user_id,
        t.amount,
        t.merchant,
        s.total_1h,
        CASE
            WHEN t.amount > 2000 THEN 'HIGH_SINGLE_TXN'
            WHEN s.total_1h > 5000 THEN 'HIGH_ROLLING_TOTAL'
        END AS alert_reason
    FROM raw_transactions t
    LEFT JOIN user_hourly_spending s
        ON t.user_id = s.user_id
    WHERE t.amount > 2000 OR s.total_1h > 5000
    EMIT CHANGES;"""
    print(f"  {sql_step3}")

    # Step 4: Join with user profile TABLE
    sql_step4 = """
    -- Step 4: LEFT JOIN to TABLE (point-in-time lookup for email).
    -- user_profiles is a TABLE (latest state per user).
    CREATE STREAM enriched_alerts AS
    SELECT
        a.txn_id,
        a.user_id,
        a.amount,
        a.alert_reason,
        p.email
    FROM fraud_alerts a
    LEFT JOIN user_profiles p
        ON a.user_id = p.user_id
    EMIT CHANGES;"""
    print(f"  {sql_step4}")

    # Step 5: Push query (continuous monitoring)
    sql_step5 = """
    -- Step 5: Push query — EMIT CHANGES streams results continuously.
    -- This query never terminates; new alerts appear as they occur.
    SELECT * FROM enriched_alerts EMIT CHANGES;"""
    print(f"  {sql_step5}")

    # Step 6: Pull query (point-in-time lookup)
    sql_step6 = """
    -- Step 6: Pull query — point-in-time lookup (like a REST GET).
    -- Returns the CURRENT value in the materialized table, not a stream.
    SELECT * FROM user_hourly_spending
    WHERE user_id = 'user_42';"""
    print(f"  {sql_step6}")

    # Simulate execution
    print("\n  Simulated Execution:")
    transactions = [
        {"txn_id": "T001", "user_id": "u1", "amount": 500, "merchant": "Amazon", "country": "US"},
        {"txn_id": "T002", "user_id": "u1", "amount": 2500, "merchant": "Luxury", "country": "US"},
        {"txn_id": "T003", "user_id": "u2", "amount": 100, "merchant": "Grocery", "country": "UK"},
        {"txn_id": "T004", "user_id": "u1", "amount": 3000, "merchant": "Jewelry", "country": "US"},
    ]

    user_totals: dict[str, float] = defaultdict(float)
    alerts = []
    for txn in transactions:
        user_totals[txn["user_id"]] += txn["amount"]
        reasons = []
        if txn["amount"] > 2000:
            reasons.append("HIGH_SINGLE_TXN")
        if user_totals[txn["user_id"]] > 5000:
            reasons.append("HIGH_ROLLING_TOTAL")
        if reasons:
            alert = {**txn, "alert_reasons": reasons, "rolling_total": user_totals[txn["user_id"]]}
            alerts.append(alert)
            print(f"    ALERT: {txn['txn_id']} user={txn['user_id']} amount=${txn['amount']} "
                  f"rolling=${user_totals[txn['user_id']]:.0f} reasons={reasons}")
        else:
            print(f"    OK:    {txn['txn_id']} user={txn['user_id']} amount=${txn['amount']} "
                  f"rolling=${user_totals[txn['user_id']]:.0f}")

    return alerts


# ---------------------------------------------------------------------------
# Exercise 4: Dead Letter Queue and Error Handling
# ---------------------------------------------------------------------------

def exercise4_dead_letter_queue():
    """Production-grade error handling with DLQ and exponential backoff.

    Exactly-once semantics interaction with retries:
    - Kafka transactions ensure that messages are produced exactly once.
    - If a retry succeeds, the original (failed) attempt was never committed.
    - If all retries fail, the event goes to the DLQ exactly once.
    - Idempotent writes to the DLQ topic prevent duplicates even if the
      producer retries the DLQ send itself.
    """
    profiles = KTable("user_profiles")
    profiles.update("u1", {"name": "Alice", "tier": "gold", "credit_limit": 5000})
    profiles.update("u2", {"name": "Bob", "tier": "silver", "credit_limit": 2000})
    # u3 will be added after a delay (simulating eventual consistency)

    dlq: list[dict] = []
    dlq_stats: dict[str, int] = defaultdict(int)
    enriched: list[dict] = []

    orders = [
        Order("ord-1", "u1", "p101", 450.00),
        Order("ord-2", "u3", "p102", 100.00),  # Profile not yet available
        Order("ord-3", "u2", "p103", 300.00),
        Order("ord-4", "u99", "p104", 50.00),   # Profile will never exist
    ]

    print("\n  Processing orders with exponential backoff retry:")
    for order in orders:
        success = False
        retry_delays = [0.1, 0.2, 0.4]  # 100ms, 200ms, 400ms

        for attempt in range(1 + len(retry_delays)):
            try:
                profile = profiles.get(order.user_id)
                if profile is None:
                    if attempt < len(retry_delays):
                        delay = retry_delays[attempt]
                        print(f"    {order.order_id}: user={order.user_id} not found, "
                              f"retry {attempt + 1}/{len(retry_delays)} in {delay}s")
                        # Simulate: u3 becomes available after first retry
                        if order.user_id == "u3" and attempt == 1:
                            profiles.update("u3", {"name": "Carol", "tier": "bronze", "credit_limit": 500})
                        time.sleep(delay / 100)  # Shortened for demo
                        continue
                    raise KeyError(f"Profile not found for {order.user_id}")

                enriched.append({
                    "order_id": order.order_id,
                    "user_id": order.user_id,
                    "user_name": profile["name"],
                    "amount": order.amount,
                })
                print(f"    {order.order_id}: enriched with {profile['name']} "
                      f"(attempt {attempt + 1})")
                success = True
                break

            except Exception as e:
                if attempt == len(retry_delays):
                    error_reason = str(e)
                    dlq.append({
                        "order_id": order.order_id,
                        "user_id": order.user_id,
                        "error_reason": error_reason,
                        "attempts": attempt + 1,
                    })
                    dlq_stats[error_reason] += 1
                    print(f"    {order.order_id}: -> DLQ ({error_reason})")

    # DLQ stats endpoint
    print(f"\n  DLQ Stats (HTTP /dlq/stats/):")
    print(f"    {json.dumps(dict(dlq_stats), indent=4)}" if dlq_stats else "    (empty)")
    print(f"\n  Summary: {len(enriched)} enriched, {len(dlq)} in DLQ")

    return enriched, dlq


# Need json for exercise 4
import json


# ---------------------------------------------------------------------------
# Exercise 5: Multi-Source Join and Session Analytics
# ---------------------------------------------------------------------------

def exercise5_session_analytics():
    """Correlate clickstream and purchase events for conversion analysis."""
    base_time = datetime(2024, 11, 15, 14, 0, 0)

    # Generate page views with sessions
    page_views = []
    for i in range(40):
        uid = f"user_{random.randint(1, 5)}"
        session_offset = random.choice([0, 0, 0, 120, 240, 2400])  # Some same session, some new
        ts = base_time + timedelta(seconds=i * 15 + session_offset)
        page_views.append({
            "user_id": uid,
            "page": random.choice(["/home", "/product/1", "/product/2", "/cart", "/checkout"]),
            "session_id": f"s_{uid}_{int(ts.timestamp()) // 1800}",  # 30-min sessions
            "timestamp": ts,
        })

    # Generate purchases
    purchases = [
        {"user_id": "user_1", "order_id": "o1", "amount": 99.99,
         "timestamp": base_time + timedelta(minutes=10)},
        {"user_id": "user_2", "order_id": "o2", "amount": 249.99,
         "timestamp": base_time + timedelta(minutes=20)},
        {"user_id": "user_3", "order_id": "o3", "amount": 49.99,
         "timestamp": base_time + timedelta(minutes=15)},
    ]

    # Session aggregation: pages per session
    session_pages: dict[str, set] = defaultdict(set)
    for pv in page_views:
        session_pages[pv["session_id"]].add(pv["page"])

    session_counts = {sid: len(pages) for sid, pages in session_pages.items()}

    print(f"\n  Session Page Counts (30-min session window):")
    for sid, count in sorted(session_counts.items()):
        uid = sid.split("_")[1] + "_" + sid.split("_")[2]
        print(f"    {sid}: {count} unique pages (user={uid})")

    # Join purchases with session data
    print(f"\n  Purchase-Session Enrichment:")
    conversion_events = []
    for purchase in purchases:
        # Find sessions for this user that ended within 30 min before purchase
        user_sessions = [
            (sid, count) for sid, count in session_counts.items()
            if purchase["user_id"] in sid
        ]
        pages_in_session = max((c for _, c in user_sessions), default=0)

        event = {
            "user_id": purchase["user_id"],
            "order_id": purchase["order_id"],
            "amount": purchase["amount"],
            "pages_in_session": pages_in_session,
        }
        conversion_events.append(event)
        print(f"    {purchase['order_id']}: {purchase['user_id']} "
              f"${purchase['amount']} | pages_in_session={pages_in_session}")

    # Average pages by amount bucket
    buckets = {"$0-50": [], "$50-200": [], "$200+": []}
    for ev in conversion_events:
        if ev["amount"] < 50:
            buckets["$0-50"].append(ev["pages_in_session"])
        elif ev["amount"] < 200:
            buckets["$50-200"].append(ev["pages_in_session"])
        else:
            buckets["$200+"].append(ev["pages_in_session"])

    print(f"\n  Average Pages per Session by Amount Bucket:")
    print(f"    {'Bucket':<12} {'Avg Pages':>10} {'Purchases':>10}")
    print(f"    {'-'*12} {'-'*10} {'-'*10}")
    for bucket, pages in buckets.items():
        avg = sum(pages) / len(pages) if pages else 0
        print(f"    {bucket:<12} {avg:>10.1f} {len(pages):>10}")

    return conversion_events


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 70)
    print("Exercise 1: Real-Time Transaction Aggregator")
    print("=" * 70)
    exercise1_transaction_aggregator()

    print()
    print("=" * 70)
    print("Exercise 2: KStream-KTable Enrichment Pipeline")
    print("=" * 70)
    exercise2_enrichment_pipeline()

    print()
    print("=" * 70)
    print("Exercise 3: ksqlDB Fraud Detection Pipeline")
    print("=" * 70)
    exercise3_ksqldb_fraud()

    print()
    print("=" * 70)
    print("Exercise 4: Dead Letter Queue and Error Handling")
    print("=" * 70)
    exercise4_dead_letter_queue()

    print()
    print("=" * 70)
    print("Exercise 5: Multi-Source Join and Session Analytics")
    print("=" * 70)
    exercise5_session_analytics()
