"""
Exercise Solutions: Lesson 17 - Spark Structured Streaming

Covers:
  - Exercise 1: Kafka-to-Parquet Streaming Pipeline
  - Exercise 2: Stateful Deduplication with Watermark
  - Exercise 3: Stream-Stream Join for Order Matching
  - Exercise 4: foreachBatch Multi-Sink Writer
  - Exercise 5: End-to-End Streaming Analytics System

Note: Pure Python simulation of Spark Structured Streaming concepts.
"""

import random
import json
import os
import tempfile
from datetime import datetime, timedelta
from collections import defaultdict
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Simulated Streaming Engine
# ---------------------------------------------------------------------------

@dataclass
class StreamEvent:
    """A single event in a stream."""
    data: dict
    event_time: datetime
    processing_time: datetime = None

    def __post_init__(self):
        if self.processing_time is None:
            self.processing_time = datetime.now()


class MicroBatch:
    """Represents one micro-batch in Structured Streaming."""
    def __init__(self, batch_id: int, events: list[StreamEvent]):
        self.batch_id = batch_id
        self.events = events
        self.input_rows = len(events)

    @property
    def is_empty(self) -> bool:
        return len(self.events) == 0


class StreamingQuery:
    """Simulates a Spark Structured Streaming query."""
    def __init__(self, name: str, watermark_seconds: int = 0):
        self.name = name
        self.watermark_seconds = watermark_seconds
        self.checkpoint: dict = {"last_offset": 0, "state": {}}
        self.batches_processed = 0


# ---------------------------------------------------------------------------
# Exercise 1: Kafka-to-Parquet Streaming Pipeline
# ---------------------------------------------------------------------------

def exercise1_kafka_to_parquet():
    """
    Spark Structured Streaming equivalent:

        events_df = (
            spark.readStream
            .format("kafka")
            .option("kafka.bootstrap.servers", "localhost:9092")
            .option("subscribe", "user_events")
            .option("maxOffsetsPerTrigger", 5000)
            .load()
        )

        parsed = events_df.select(
            from_json(col("value").cast("string"), schema).alias("data")
        ).select("data.*")

        windowed = (
            parsed
            .withWatermark("event_time", "3 minutes")
            .groupBy(
                col("action"),
                window("event_time", "10 minutes")
            )
            .agg(
                count("*").alias("event_count"),
                sum("amount").alias("total_amount"),
                approx_count_distinct("user_id").alias("unique_users"),
            )
        )

        query = (
            windowed.writeStream
            .format("parquet")
            .outputMode("update")
            .option("checkpointLocation", "/checkpoints/event_summary")
            .start("/data/output/event_summary/")
        )
    """
    # Generate sample events
    actions = ["click", "purchase", "view", "add_to_cart"]
    base_time = datetime(2024, 11, 15, 14, 0, 0)

    events = []
    for i in range(100):
        event_time = base_time + timedelta(seconds=random.randint(0, 1800))
        events.append(StreamEvent(
            data={
                "event_id": f"evt_{i+1:04d}",
                "user_id": f"user_{random.randint(1, 30)}",
                "action": random.choice(actions),
                "amount": round(random.uniform(5, 200), 2),
            },
            event_time=event_time,
        ))

    # Apply 3-minute watermark
    watermark = timedelta(minutes=3)

    # Tumbling 10-minute window aggregation
    window_size = timedelta(minutes=10)
    window_agg: dict[tuple[str, str], dict] = {}

    for evt in events:
        # Determine window
        epoch = int(evt.event_time.timestamp())
        window_start_epoch = epoch - (epoch % int(window_size.total_seconds()))
        window_start = datetime.fromtimestamp(window_start_epoch)
        window_end = window_start + window_size
        window_key = f"{window_start.strftime('%H:%M')}-{window_end.strftime('%H:%M')}"

        action = evt.data["action"]
        key = (action, window_key)

        if key not in window_agg:
            window_agg[key] = {"event_count": 0, "total_amount": 0.0, "users": set()}
        window_agg[key]["event_count"] += 1
        window_agg[key]["total_amount"] += evt.data["amount"]
        window_agg[key]["users"].add(evt.data["user_id"])

    # Display results
    print(f"\n  10-Minute Windowed Aggregation (watermark=3min):")
    print(f"  {'Window':<14} {'Action':<14} {'Count':>6} {'Amount':>10} {'Users':>6}")
    print(f"  {'-'*14} {'-'*14} {'-'*6} {'-'*10} {'-'*6}")
    for (action, window), stats in sorted(window_agg.items(), key=lambda x: (x[0][1], x[0][0])):
        print(f"  {window:<14} {action:<14} {stats['event_count']:>6} "
              f"${stats['total_amount']:>9.2f} {len(stats['users']):>6}")

    # Monitoring: inputRowsPerSecond
    processing_time_seconds = 30  # simulated
    rows_per_second = len(events) / processing_time_seconds
    print(f"\n  Query Monitoring:")
    print(f"    inputRowsPerSecond: {rows_per_second:.1f}")
    print(f"    maxOffsetsPerTrigger: 5000")
    print(f"    checkpoint: /checkpoints/event_summary/")

    return window_agg


# ---------------------------------------------------------------------------
# Exercise 2: Stateful Deduplication with Watermark
# ---------------------------------------------------------------------------

def exercise2_deduplication():
    """
    Spark equivalent:

        deduped = (
            raw_events
            .withWatermark("event_ts", "15 minutes")
            .dropDuplicates(["event_id", "event_ts"])
        )

    Why event_ts must be included in dropDuplicates alongside event_id:
    - Spark uses the watermark column to determine WHEN to expire state.
    - Without event_ts in the dedup key, Spark cannot use the watermark
      to clean up the state store.
    - The state would grow unboundedly because Spark wouldn't know when
      it's safe to forget old event_ids.
    - Including event_ts ties the dedup state lifecycle to the watermark,
      ensuring state is cleaned up after the watermark advances past it.
    """
    base_time = datetime(2024, 11, 15, 14, 0, 0)

    # Generate events with duplicates (producer retries)
    raw_events = []
    for i in range(50):
        event_time = base_time + timedelta(seconds=random.randint(0, 900))
        event = {
            "event_id": f"evt_{i+1:04d}",
            "user_id": f"user_{random.randint(1, 10)}",
            "amount": round(random.uniform(10, 200), 2),
            "event_ts": event_time,
        }
        raw_events.append(event)
        # Add duplicates for ~20% of events
        if random.random() < 0.2:
            raw_events.append(dict(event))  # Exact duplicate

    print(f"\n  Raw events: {len(raw_events)} (includes duplicates)")

    # Deduplication with watermark
    seen: dict[tuple[str, str], dict] = {}
    watermark_duration = timedelta(minutes=15)
    current_max_time = base_time

    deduped_events = []
    duplicates_removed = 0

    for evt in raw_events:
        event_time = evt["event_ts"]
        current_max_time = max(current_max_time, event_time)
        watermark_threshold = current_max_time - watermark_duration

        # Drop if event_ts is before watermark (too late)
        if event_time < watermark_threshold:
            duplicates_removed += 1
            continue

        dedup_key = (evt["event_id"], evt["event_ts"].isoformat())
        if dedup_key in seen:
            duplicates_removed += 1
            continue

        seen[dedup_key] = evt
        deduped_events.append(evt)

    print(f"  After deduplication: {len(deduped_events)} events")
    print(f"  Duplicates removed: {duplicates_removed}")
    print(f"  Deduplication rate: {duplicates_removed / len(raw_events) * 100:.1f}%")

    # foreachBatch: write to Kafka and PostgreSQL simultaneously
    print(f"\n  foreachBatch output sinks:")
    print(f"    -> Kafka topic 'clean_events': {len(deduped_events)} events")
    print(f"    -> PostgreSQL 'clean_events_log': {len(deduped_events)} rows")

    return deduped_events


# ---------------------------------------------------------------------------
# Exercise 3: Stream-Stream Join for Order Matching
# ---------------------------------------------------------------------------

def exercise3_stream_stream_join():
    """
    Spark equivalent:

        orders_with_wm = orders.withWatermark("order_ts", "1 hour")
        payments_with_wm = payments.withWatermark("pay_ts", "2 hours")

        matched = orders_with_wm.join(
            payments_with_wm,
            expr('''
                orders.order_id = payments.order_id AND
                pay_ts >= order_ts AND
                pay_ts <= order_ts + INTERVAL 4 HOURS
            '''),
            "inner"
        )

    Why both streams need watermarks:
    - Each stream's watermark tells Spark how late events can arrive.
    - Spark uses BOTH watermarks to determine when it can safely emit
      results and clean up state.
    - The state retention period = max(order watermark, payment watermark)
      + time range condition (4 hours).
    - Without watermarks on both sides, Spark would keep ALL unmatched
      state indefinitely (unbounded memory).
    """
    base_time = datetime(2024, 11, 15, 10, 0, 0)

    # Generate orders
    orders = []
    for i in range(15):
        order_ts = base_time + timedelta(minutes=random.randint(0, 120))
        orders.append({
            "order_id": f"ord_{i+1:03d}",
            "user_id": f"user_{random.randint(1, 5)}",
            "amount": round(random.uniform(50, 500), 2),
            "order_ts": order_ts,
        })

    # Generate payments (some match orders, some don't)
    payments = []
    for order in orders[:10]:  # Only 10 out of 15 orders get paid
        delay_minutes = random.randint(5, 300)  # Some may exceed 4-hour window
        pay_ts = order["order_ts"] + timedelta(minutes=delay_minutes)
        payments.append({
            "payment_id": f"pay_{order['order_id']}",
            "order_id": order["order_id"],
            "payment_method": random.choice(["credit_card", "paypal", "bank_transfer"]),
            "pay_ts": pay_ts,
        })

    # Stream-stream join with time condition
    JOIN_WINDOW_HOURS = 4
    matched = []
    unmatched_orders = []

    for order in orders:
        payment = next((p for p in payments if p["order_id"] == order["order_id"]), None)
        if payment:
            time_diff = (payment["pay_ts"] - order["order_ts"]).total_seconds() / 3600
            if time_diff <= JOIN_WINDOW_HOURS:
                matched.append({
                    "order_id": order["order_id"],
                    "user_id": order["user_id"],
                    "amount": order["amount"],
                    "payment_method": payment["payment_method"],
                    "order_ts": order["order_ts"].isoformat(),
                    "pay_ts": payment["pay_ts"].isoformat(),
                    "delay_hours": round(time_diff, 2),
                })
            else:
                unmatched_orders.append({
                    "order_id": order["order_id"],
                    "reason": f"Payment received {time_diff:.1f}h after order (>{JOIN_WINDOW_HOURS}h window)",
                })
        else:
            unmatched_orders.append({
                "order_id": order["order_id"],
                "reason": "No payment received",
            })

    print(f"\n  Stream-Stream Join: Orders <-> Payments (within {JOIN_WINDOW_HOURS}h)")
    print(f"\n  Matched Orders ({len(matched)}):")
    print(f"  {'Order':<10} {'User':<8} {'Amount':>8} {'Payment':>15} {'Delay':>7}")
    print(f"  {'-'*10} {'-'*8} {'-'*8} {'-'*15} {'-'*7}")
    for m in matched[:10]:
        print(f"  {m['order_id']:<10} {m['user_id']:<8} ${m['amount']:>7.2f} "
              f"{m['payment_method']:>15} {m['delay_hours']:>5.1f}h")

    print(f"\n  Unmatched Orders ({len(unmatched_orders)}):")
    for u in unmatched_orders:
        print(f"    {u['order_id']}: {u['reason']}")

    return matched, unmatched_orders


# ---------------------------------------------------------------------------
# Exercise 4: foreachBatch Multi-Sink Writer
# ---------------------------------------------------------------------------

def exercise4_foreach_batch():
    """
    Spark equivalent:

        def process_batch(batch_df, batch_id):
            if batch_df.isEmpty():
                return
            # 1. Write raw to Parquet
            batch_df.write.mode("append").partitionBy("date").parquet(...)
            # 2. Upsert to Delta Lake
            DeltaTable.merge(...).whenMatchedUpdate(...).whenNotMatchedInsert(...)
            # 3. Write summary to PostgreSQL
            summary.write.jdbc(url, table, mode="append")

        query = stream.writeStream.foreachBatch(process_batch).start()

    Why batch_id is critical for idempotent foreachBatch:
    - Spark may re-execute a micro-batch on failure (at-least-once delivery).
    - Without batch_id in the merge condition, the same rows could be
      inserted multiple times (duplicates).
    - Using batch_id in the merge: UPDATE WHERE target._batch_id < source._batch_id
      ensures that re-processing the same batch is a no-op.
    """
    base_time = datetime(2024, 11, 15, 14, 0, 0)
    tmpdir = tempfile.mkdtemp(prefix="streaming_sinks_")

    # Simulate 3 micro-batches
    batches = []
    for batch_id in range(3):
        events = []
        for i in range(random.randint(5, 15)):
            event_time = base_time + timedelta(minutes=batch_id * 10 + random.randint(0, 9))
            events.append(StreamEvent(
                data={
                    "event_id": f"evt_{batch_id}_{i}",
                    "user_id": f"user_{random.randint(1, 5)}",
                    "action": random.choice(["click", "purchase"]),
                    "amount": round(random.uniform(10, 200), 2),
                    "date": event_time.strftime("%Y-%m-%d"),
                },
                event_time=event_time,
            ))
        batches.append(MicroBatch(batch_id, events))

    # Delta Lake table (simulated)
    delta_table: dict[str, dict] = {}
    batch_log: list[dict] = []

    print("\n  foreachBatch Processing:")
    for batch in batches:
        print(f"\n  --- Batch {batch.batch_id} ({batch.input_rows} rows) ---")

        if batch.is_empty:
            print(f"    Skipping empty batch")
            continue

        # Sink 1: Parquet (partitioned by date)
        parquet_dir = os.path.join(tmpdir, "raw_events")
        dates = set(e.data["date"] for e in batch.events)
        print(f"    [Parquet] Written to {len(dates)} date partitions")

        # Sink 2: Delta Lake upsert (per-user totals)
        for evt in batch.events:
            uid = evt.data["user_id"]
            if uid in delta_table:
                if delta_table[uid]["_batch_id"] < batch.batch_id:
                    delta_table[uid]["total_amount"] += evt.data["amount"]
                    delta_table[uid]["event_count"] += 1
                    delta_table[uid]["_batch_id"] = batch.batch_id
                # else: skip (idempotent - already processed this or newer batch)
            else:
                delta_table[uid] = {
                    "user_id": uid,
                    "total_amount": evt.data["amount"],
                    "event_count": 1,
                    "_batch_id": batch.batch_id,
                }
        print(f"    [Delta] Upserted {len(set(e.data['user_id'] for e in batch.events))} users")

        # Sink 3: PostgreSQL batch log
        total_amount = sum(e.data["amount"] for e in batch.events)
        log_entry = {
            "batch_id": batch.batch_id,
            "row_count": batch.input_rows,
            "total_amount": round(total_amount, 2),
            "timestamp": datetime.now().isoformat(),
        }
        batch_log.append(log_entry)
        print(f"    [Postgres] Logged: batch_id={batch.batch_id}, "
              f"rows={batch.input_rows}, total=${total_amount:.2f}")

    # Cleanup
    try:
        os.rmdir(tmpdir)
    except OSError:
        pass

    print(f"\n  Delta Lake User Totals:")
    for uid, data in sorted(delta_table.items()):
        print(f"    {uid}: {data['event_count']} events, ${data['total_amount']:.2f}")

    return delta_table, batch_log


# ---------------------------------------------------------------------------
# Exercise 5: End-to-End Streaming Analytics System
# ---------------------------------------------------------------------------

def exercise5_streaming_analytics():
    """
    E-commerce streaming analytics: clicks + cart_events + purchases.

    Why checkpoint directories must never be shared between queries:
    - Each query maintains its own offset tracking and state.
    - Sharing checkpoints would corrupt offsets: query A's state overwrites
      query B's progress, causing data loss or duplication.
    - Each query has independent failure/recovery semantics.
    """
    base_time = datetime(2024, 11, 15, 14, 0, 0)
    WINDOW_MINUTES = 15

    # Generate events
    clicks = [
        {"user_id": f"user_{random.randint(1, 10)}", "page": f"/product/{random.randint(1, 20)}",
         "timestamp": base_time + timedelta(seconds=random.randint(0, 900))}
        for _ in range(80)
    ]
    cart_events = [
        {"user_id": f"user_{random.randint(1, 10)}", "product_id": f"prod_{random.randint(1, 20)}",
         "action": "add", "timestamp": base_time + timedelta(seconds=random.randint(0, 900))}
        for _ in range(30)
    ]
    purchases = [
        {"user_id": f"user_{random.randint(1, 10)}", "product_id": f"prod_{random.randint(1, 5)}",
         "amount": round(random.uniform(20, 300), 2),
         "timestamp": base_time + timedelta(seconds=random.randint(0, 900))}
        for _ in range(15)
    ]

    # Product catalog (static DataFrame for enrichment)
    catalog = {
        f"prod_{i}": {"category": random.choice(["electronics", "clothing", "books", "food"]),
                       "price": round(random.uniform(10, 300), 2)}
        for i in range(1, 21)
    }

    # Funnel computation (15-minute tumbling window)
    def window_key(ts):
        epoch = int(ts.timestamp())
        ws = epoch - (epoch % (WINDOW_MINUTES * 60))
        return datetime.fromtimestamp(ws).strftime("%H:%M")

    funnel: dict[str, dict] = defaultdict(lambda: {"clicks": 0, "carts": 0, "purchases": 0})

    for c in clicks:
        funnel[window_key(c["timestamp"])]["clicks"] += 1
    for c in cart_events:
        funnel[window_key(c["timestamp"])]["carts"] += 1
    for p in purchases:
        funnel[window_key(p["timestamp"])]["purchases"] += 1

    print(f"\n  E-Commerce Funnel (15-min tumbling windows):")
    print(f"  {'Window':<10} {'Clicks':>8} {'Carts':>8} {'Purchases':>10} {'Conv Rate':>10}")
    print(f"  {'-'*10} {'-'*8} {'-'*8} {'-'*10} {'-'*10}")

    for wk in sorted(funnel.keys()):
        stats = funnel[wk]
        conv_rate = stats["purchases"] / stats["clicks"] if stats["clicks"] > 0 else 0
        print(f"  {wk:<10} {stats['clicks']:>8} {stats['carts']:>8} "
              f"{stats['purchases']:>10} {conv_rate:>9.2%}")

    # Static enrichment: join purchases with product catalog
    print(f"\n  Enriched Purchases (joined with product catalog):")
    enriched = []
    for p in purchases[:8]:
        product_info = catalog.get(p["product_id"], {"category": "unknown", "price": 0})
        enriched.append({
            "user_id": p["user_id"],
            "product_id": p["product_id"],
            "amount": p["amount"],
            "category": product_info["category"],
            "catalog_price": product_info["price"],
        })
        print(f"    {p['user_id']} bought {p['product_id']} "
              f"(${p['amount']:.2f}, category={product_info['category']})")

    # Checkpoint configuration
    print(f"\n  Checkpoint Configuration:")
    print(f"    Query 1 (funnel):   /checkpoints/funnel_metrics/")
    print(f"    Query 2 (enriched): /checkpoints/enriched_purchases/")
    print(f"    IMPORTANT: Never share checkpoint directories between queries!")
    print(f"    Each query has independent offsets and state that would corrupt if shared.")

    return funnel, enriched


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 70)
    print("Exercise 1: Kafka-to-Parquet Streaming Pipeline")
    print("=" * 70)
    exercise1_kafka_to_parquet()

    print()
    print("=" * 70)
    print("Exercise 2: Stateful Deduplication with Watermark")
    print("=" * 70)
    exercise2_deduplication()

    print()
    print("=" * 70)
    print("Exercise 3: Stream-Stream Join for Order Matching")
    print("=" * 70)
    exercise3_stream_stream_join()

    print()
    print("=" * 70)
    print("Exercise 4: foreachBatch Multi-Sink Writer")
    print("=" * 70)
    exercise4_foreach_batch()

    print()
    print("=" * 70)
    print("Exercise 5: End-to-End Streaming Analytics System")
    print("=" * 70)
    exercise5_streaming_analytics()
