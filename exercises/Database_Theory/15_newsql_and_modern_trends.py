"""
Exercises for Lesson 15: NewSQL and Modern Trends
Topic: Database_Theory

Solutions to practice problems from the lesson.
Covers NewSQL comparison, TrueTime analysis, vector database design,
time-series schema, HTAP analysis, and database selection.
"""


# === Exercise 1: NewSQL Comparison ===
# Problem: Compare Spanner, CockroachDB, and TiDB.

def exercise_1():
    """Compare NewSQL databases across dimensions."""
    comparison = {
        "Clock mechanism": {
            "Spanner": "TrueTime (GPS + atomic clocks, epsilon 1-7ms)",
            "CockroachDB": "Hybrid Logical Clocks (HLC) + NTP",
            "TiDB": "Timestamp Oracle (TSO) -- centralized timestamp server"
        },
        "Default isolation level": {
            "Spanner": "External consistency (stronger than serializable)",
            "CockroachDB": "Serializable (SSI-based)",
            "TiDB": "Snapshot Isolation (SI), configurable to Serializable"
        },
        "Replication protocol": {
            "Spanner": "Paxos (per shard/split)",
            "CockroachDB": "Raft (per Range)",
            "TiDB": "Raft (via TiKV storage engine)"
        },
        "SQL compatibility": {
            "Spanner": "Google SQL (proprietary, not fully ANSI SQL)",
            "CockroachDB": "PostgreSQL wire protocol compatible",
            "TiDB": "MySQL wire protocol compatible"
        },
        "HTAP support": {
            "Spanner": "Limited (primarily OLTP, BigQuery for analytics)",
            "CockroachDB": "Limited (OLTP-focused, improving analytics)",
            "TiDB": "YES (TiKV for OLTP + TiFlash columnar for OLAP)"
        },
        "Open source?": {
            "Spanner": "No (Google Cloud service only)",
            "CockroachDB": "BSL license (source-available, not truly open source since 2024)",
            "TiDB": "Yes (Apache 2.0)"
        },
        "Typical deployment": {
            "Spanner": "Google Cloud only (managed service)",
            "CockroachDB": "Self-hosted, CockroachDB Cloud, any cloud",
            "TiDB": "Self-hosted, TiDB Cloud, any cloud"
        }
    }

    # Print as table
    dims = list(comparison.keys())
    systems = ["Spanner", "CockroachDB", "TiDB"]

    print(f"  {'Dimension':<25}", end="")
    for sys in systems:
        print(f" | {sys:<35}", end="")
    print()
    print(f"  {'-'*25}", end="")
    for _ in systems:
        print(f"-+-{'-'*35}", end="")
    print()

    for dim in dims:
        print(f"  {dim:<25}", end="")
        for sys in systems:
            val = comparison[dim][sys][:35]
            print(f" | {val:<35}", end="")
        print()


# === Exercise 2: TrueTime and Commit Wait ===
# Problem: Analyze TrueTime's commit wait mechanism.

def exercise_2():
    """TrueTime commit wait analysis."""
    print("TrueTime: reports clock uncertainty [T-epsilon, T+epsilon]")
    print()

    epsilon = 5  # ms

    print(f"1. Why Spanner must 'wait out' the uncertainty:")
    print(f"   When T1 commits at real time t, TrueTime says: t is in [t-{epsilon}ms, t+{epsilon}ms].")
    print(f"   If T2 starts after T1 commits (in real time), we need T2's timestamp > T1's.")
    print(f"   Without commit wait: T1's timestamp might be t+{epsilon}ms (upper bound),")
    print(f"   and T2's timestamp might be t+1ms-{epsilon}ms (lower bound). T2 < T1!")
    print(f"   Commit wait: T1 waits until TrueTime.now().earliest > T1's timestamp,")
    print(f"   guaranteeing no future transaction can get an earlier timestamp.")
    print()

    print(f"2. Minimum commit latency with epsilon = {epsilon}ms:")
    print(f"   Commit wait = 2 x epsilon = {2 * epsilon}ms")
    print(f"   (Wait from commit decision until uncertainty interval passes)")
    print(f"   Total commit latency: ~{2 * epsilon}ms + network round-trip")
    print()

    print(f"3. If epsilon = 0 (perfect clock):")
    print(f"   No commit wait needed! Every timestamp is exact.")
    print(f"   Commit is instant after Paxos consensus.")
    print(f"   The protocol degenerates to simple timestamp assignment.")
    print()

    print(f"4. Why CockroachDB can't use TrueTime:")
    print(f"   TrueTime requires GPS receivers and atomic clocks in every datacenter.")
    print(f"   CockroachDB runs on commodity hardware without specialized clocks.")
    print(f"   Instead, CockroachDB uses Hybrid Logical Clocks (HLC):")
    print(f"     - Physical component (NTP-synchronized, ~50ms uncertainty)")
    print(f"     - Logical component (Lamport-style counter)")
    print(f"     - Causality tracking without bounded uncertainty")
    print()

    print(f"5. CockroachDB 'read restart' scenario:")
    print(f"   T1 reads key K at timestamp ts=100.")
    print(f"   T2 writes K at timestamp ts=95 (NTP clock skew -- T2's clock is slow).")
    print(f"   T1 later discovers T2's write at ts=95 < ts=100.")
    print(f"   T1 must RESTART at a higher timestamp to see T2's write.")
    print(f"   User impact: increased latency (retry), but correctness is maintained.")
    print(f"   This is transparent to the application -- the driver retries automatically.")


# === Exercise 3: Vector Database Design ===
# Problem: Design vector database for RAG system.

def exercise_3():
    """Vector database design for RAG chatbot."""
    print("RAG System: Customer Support Chatbot")
    print(f"  Articles: 100,000 (avg 500 words each)")
    print(f"  Updates: weekly")
    print(f"  Query load: 1,000 queries/minute")
    print(f"  Latency: < 100ms")
    print()

    print("1. Vector database choice: Qdrant")
    print("   Justification:")
    print("   - Open-source, can self-host (data privacy for customer support)")
    print("   - Native HNSW with quantization (fast, memory-efficient)")
    print("   - Built-in payload filtering (metadata search alongside vectors)")
    print("   - REST + gRPC API (easy integration)")
    print("   - Alternatives: Pinecone (managed, simpler), Weaviate (hybrid search)")
    print()

    print("2. Embedding model: OpenAI text-embedding-3-small (1536 dimensions)")
    print("   - Good quality for English text at reasonable cost")
    print("   - 1536 dimensions balances accuracy vs. storage")
    print("   - Alternative: Cohere embed-v3 (1024 dims), local: sentence-transformers")
    print()

    print("3. Indexing strategy: HNSW")
    print("   - Best for recall and latency at this scale (100K articles)")
    print("   - Parameters: M=16, efConstruction=200, efSearch=100")
    print("   - With 100K vectors x 1536 dims x 4 bytes = ~600 MB (fits in RAM)")
    print("   - Quantization (scalar/PQ) if memory is constrained")
    print()

    print("4. Handling article updates:")
    print("   Strategy: Chunk-level updates with versioning")
    print("   - Split articles into ~200-word chunks with overlap")
    print("   - Each chunk has metadata: article_id, chunk_index, version")
    print("   - On update: re-embed only changed chunks")
    print("   - Delete old version chunks, insert new version")
    print("   - Weekly batch: re-embed all updated articles overnight")
    print()

    print("5. Hybrid search implementation:")
    print("   - Vector similarity: HNSW index for semantic search")
    print("   - Keyword matching: BM25 full-text search (Qdrant payload index)")
    print("   - Combination: Reciprocal Rank Fusion (RRF)")
    print("     score = 1/(k + rank_vector) + 1/(k + rank_keyword)")
    print("   - Benefits: catches exact terms (product names, error codes)")
    print("     that pure vector search might miss")

    # Demonstrate RRF scoring
    print()
    print("   RRF Example (k=60):")
    results_vector = ["doc_A", "doc_B", "doc_C", "doc_D"]
    results_keyword = ["doc_C", "doc_A", "doc_E", "doc_B"]
    k = 60

    scores = {}
    for rank, doc in enumerate(results_vector, 1):
        scores[doc] = scores.get(doc, 0) + 1 / (k + rank)
    for rank, doc in enumerate(results_keyword, 1):
        scores[doc] = scores.get(doc, 0) + 1 / (k + rank)

    sorted_results = sorted(scores.items(), key=lambda x: -x[1])
    for doc, score in sorted_results:
        print(f"     {doc}: {score:.6f}")


# === Exercise 4: Time-Series Schema Design ===
# Problem: Design TimescaleDB schema for smart building monitoring.

def exercise_4():
    """TimescaleDB schema for smart building monitoring."""
    print("Smart Building Monitoring System")
    print(f"  Sensors: 500 across 50 floors")
    print(f"  Readings: temperature, humidity, CO2, occupancy every 10 seconds")
    print(f"  Retention: 90 days raw, 2 years hourly aggregates")
    print()

    # 1. Table creation
    print("1. Table creation with hypertable:")
    sql_create = """
    CREATE TABLE sensor_readings (
        time        TIMESTAMPTZ NOT NULL,
        sensor_id   INTEGER NOT NULL,
        floor       INTEGER NOT NULL,
        temperature DOUBLE PRECISION,
        humidity    DOUBLE PRECISION,
        co2_ppm     DOUBLE PRECISION,
        occupancy   INTEGER
    );

    -- Convert to hypertable (partition by time, chunk = 1 day)
    SELECT create_hypertable('sensor_readings', 'time',
        chunk_time_interval => INTERVAL '1 day');

    -- Add index for sensor-specific queries
    CREATE INDEX idx_sensor_time ON sensor_readings (sensor_id, time DESC);
    CREATE INDEX idx_floor_time ON sensor_readings (floor, time DESC);
    """
    print(sql_create)

    # 2. Continuous aggregate
    print("2. Continuous aggregate for hourly averages:")
    sql_cagg = """
    CREATE MATERIALIZED VIEW hourly_averages
    WITH (timescaledb.continuous) AS
    SELECT
        time_bucket('1 hour', time) AS hour,
        floor,
        AVG(temperature) AS avg_temp,
        AVG(humidity) AS avg_humidity,
        AVG(co2_ppm) AS avg_co2,
        AVG(occupancy) AS avg_occupancy,
        MAX(temperature) AS max_temp,
        MIN(temperature) AS min_temp,
        COUNT(*) AS reading_count
    FROM sensor_readings
    GROUP BY time_bucket('1 hour', time), floor
    WITH DATA;

    -- Refresh policy: update every hour, look back 2 hours
    SELECT add_continuous_aggregate_policy('hourly_averages',
        start_offset => INTERVAL '2 hours',
        end_offset   => INTERVAL '1 hour',
        schedule_interval => INTERVAL '1 hour');
    """
    print(sql_cagg)

    # 3. Retention policy
    print("3. Retention policy:")
    sql_retention = """
    -- Drop raw data older than 90 days
    SELECT add_retention_policy('sensor_readings', INTERVAL '90 days');

    -- Keep hourly aggregates for 2 years (separate retention)
    SELECT add_retention_policy('hourly_averages', INTERVAL '2 years');
    """
    print(sql_retention)

    # 4. Compression policy
    print("4. Compression policy:")
    sql_compression = """
    -- Enable compression on raw data
    ALTER TABLE sensor_readings SET (
        timescaledb.compress,
        timescaledb.compress_segmentby = 'sensor_id',
        timescaledb.compress_orderby = 'time DESC'
    );

    -- Compress chunks older than 7 days
    SELECT add_compression_policy('sensor_readings', INTERVAL '7 days');
    """
    print(sql_compression)

    # 5. Query
    print("5. Floors where avg temperature exceeded 28C in last 6 hours:")
    sql_query = """
    SELECT
        floor,
        AVG(temperature) AS avg_temp,
        MAX(temperature) AS max_temp,
        COUNT(DISTINCT sensor_id) AS sensor_count
    FROM sensor_readings
    WHERE time > NOW() - INTERVAL '6 hours'
    GROUP BY floor
    HAVING AVG(temperature) > 28.0
    ORDER BY avg_temp DESC;
    """
    print(sql_query)


# === Exercise 6: Database Selection ===
# Problem: Select appropriate database for each application.

def exercise_6():
    """Database selection for various applications."""
    applications = [
        {
            "app": "1. Genomics: DNA sequence embeddings, 10B sequences",
            "choice": "Vector database: Milvus (or Zilliz Cloud)",
            "justification": [
                "Purpose-built for billion-scale vector similarity search",
                "Supports IVF+PQ indexing for 10B vectors (distributed)",
                "GPU acceleration for distance computation",
                "Not Pinecone (10B exceeds their typical managed tier)"
            ]
        },
        {
            "app": "2. Factory monitoring: 10K machines, 50 metrics/sec each",
            "choice": "Time-series database: InfluxDB or TimescaleDB",
            "justification": [
                "500K data points per second (10K x 50)",
                "Time-series optimized: columnar compression, downsampling",
                "Built-in retention policies and continuous aggregates",
                "TimescaleDB if SQL ecosystem important; InfluxDB for pure time-series"
            ]
        },
        {
            "app": "3. Global banking: ACID transactions in 30 countries",
            "choice": "NewSQL: Google Spanner (or CockroachDB)",
            "justification": [
                "Distributed ACID transactions across continents",
                "Spanner: externally consistent with TrueTime",
                "CockroachDB: if avoiding Google Cloud lock-in",
                "Both provide automatic sharding and replication"
            ]
        },
        {
            "app": "4. Collaborative document editor (10K users)",
            "choice": "PostgreSQL + Redis + CRDT library",
            "justification": [
                "PostgreSQL: document metadata, user accounts, permissions",
                "Redis: real-time presence, cursor positions, pub/sub",
                "CRDT (e.g., Yjs, Automerge): conflict-free real-time editing",
                "10K users is small enough for a well-tuned PostgreSQL"
            ]
        },
        {
            "app": "5. Analytics: SQL over 100 TB in S3",
            "choice": "Lakehouse: Apache Iceberg + Trino (or DuckDB for smaller queries)",
            "justification": [
                "Data lives in S3 (object storage) -- no data movement",
                "Iceberg: table format with ACID, schema evolution, time travel",
                "Trino: distributed SQL engine, reads Iceberg tables directly",
                "Alternative: Databricks with Delta Lake, or AWS Athena"
            ]
        }
    ]

    for app in applications:
        print(f"{app['app']}")
        print(f"  Choice: {app['choice']}")
        for j in app["justification"]:
            print(f"    - {j}")
        print()


if __name__ == "__main__":
    print("=" * 70)
    print("=== Exercise 1: NewSQL Comparison ===")
    print("=" * 70)
    exercise_1()

    print("=" * 70)
    print("=== Exercise 2: TrueTime Analysis ===")
    print("=" * 70)
    exercise_2()

    print("=" * 70)
    print("=== Exercise 3: Vector Database Design ===")
    print("=" * 70)
    exercise_3()

    print("=" * 70)
    print("=== Exercise 4: Time-Series Schema ===")
    print("=" * 70)
    exercise_4()

    print("=" * 70)
    print("=== Exercise 6: Database Selection ===")
    print("=" * 70)
    exercise_6()

    print("\nAll exercises completed!")
