"""
Dagster Asset Pipeline Example
==============================
A complete Dagster asset pipeline demonstrating:
- Software-defined assets (raw_orders, cleaned_orders, order_metrics)
- Custom IO Managers for Parquet/CSV persistence
- Partitioned assets with daily partitions
- Dagster resources configuration
- Asset dependency graph with bronze/silver/gold layers

Requirements:
    pip install dagster dagster-webserver pandas pyarrow

Usage:
    # Launch the Dagster UI (development)
    dagster dev -f asset_pipeline.py

    # Materialize all assets via CLI
    dagster asset materialize --select '*' -f asset_pipeline.py
"""

import dagster as dg
import pandas as pd
import json
from datetime import datetime, timedelta
from pathlib import Path


# =============================================================================
# Resources — Injectable dependencies for different environments
# =============================================================================

class FileStorageResource(dg.ConfigurableResource):
    """Configurable file storage resource.

    Why a resource instead of hardcoded paths?
      - Dev: store in /tmp/dagster_dev/
      - Prod: store in /data/dagster_prod/ (or S3 via a different resource)
      - Tests: use an in-memory mock
      - The asset code stays IDENTICAL across environments
    """
    base_path: str = "/tmp/dagster_assets"

    def ensure_dir(self, subpath: str) -> Path:
        """Create directory if it doesn't exist and return full path."""
        path = Path(self.base_path) / subpath
        path.mkdir(parents=True, exist_ok=True)
        return path

    def read_json(self, subpath: str) -> list[dict]:
        """Read JSON data from storage (simulates external API/DB read)."""
        filepath = Path(self.base_path) / subpath
        if filepath.exists():
            with open(filepath) as f:
                return json.load(f)
        return []


# =============================================================================
# IO Managers — Control HOW assets are stored and loaded
# =============================================================================

class ParquetIOManager(dg.ConfigurableIOManager):
    """Store and load assets as Parquet files.

    Why a custom IO Manager?
      - Separates asset LOGIC (what to compute) from PERSISTENCE (where to store)
      - The same asset function works with Parquet locally, S3 Parquet in prod,
        or in-memory during tests — just swap the IO Manager
      - Metadata (row count, columns, file path) is automatically tracked in Dagster UI
    """
    base_path: str = "/tmp/dagster_parquet"

    def _get_path(self, context) -> Path:
        """Determine file path from the asset key and optional partition."""
        asset_name = context.asset_key.to_python_identifier()
        base = Path(self.base_path)

        # Why check for partition_key?
        # Partitioned assets get a subdirectory per partition (e.g., 2024-01-15/)
        # Non-partitioned assets get a single file
        if context.has_partition_key:
            return base / asset_name / f"{context.partition_key}.parquet"
        return base / f"{asset_name}.parquet"

    def handle_output(self, context: dg.OutputContext, obj: pd.DataFrame) -> None:
        """Persist the asset output as a Parquet file."""
        path = self._get_path(context)
        path.parent.mkdir(parents=True, exist_ok=True)
        obj.to_parquet(path, index=False)

        # Why add metadata?
        # - Visible in the Dagster UI's asset detail page
        # - Useful for debugging: "How many rows? Which columns? Where stored?"
        context.add_output_metadata({
            "num_rows": dg.MetadataValue.int(len(obj)),
            "columns": dg.MetadataValue.text(", ".join(obj.columns)),
            "file_path": dg.MetadataValue.path(str(path)),
            "file_size_bytes": dg.MetadataValue.int(path.stat().st_size),
        })
        context.log.info(f"Wrote {len(obj)} rows to {path}")

    def load_input(self, context: dg.InputContext) -> pd.DataFrame:
        """Load an upstream asset from its Parquet file."""
        path = self._get_path(context)
        if not path.exists():
            raise FileNotFoundError(
                f"Asset file not found: {path}. "
                f"Has the upstream asset been materialized?"
            )
        df = pd.read_parquet(path)
        context.log.info(f"Loaded {len(df)} rows from {path}")
        return df


class CsvIOManager(dg.ConfigurableIOManager):
    """Store and load assets as CSV files.

    Why CSV alongside Parquet?
      - CSV is human-readable (useful for debugging and small datasets)
      - Some downstream tools (Excel, legacy systems) prefer CSV
      - Demonstrates that IO Managers are swappable per asset
    """
    base_path: str = "/tmp/dagster_csv"

    def _get_path(self, context) -> Path:
        asset_name = context.asset_key.to_python_identifier()
        return Path(self.base_path) / f"{asset_name}.csv"

    def handle_output(self, context: dg.OutputContext, obj: pd.DataFrame) -> None:
        path = self._get_path(context)
        path.parent.mkdir(parents=True, exist_ok=True)
        obj.to_csv(path, index=False)
        context.add_output_metadata({
            "num_rows": dg.MetadataValue.int(len(obj)),
            "file_path": dg.MetadataValue.path(str(path)),
        })

    def load_input(self, context: dg.InputContext) -> pd.DataFrame:
        path = self._get_path(context)
        return pd.read_csv(path)


# =============================================================================
# Assets — Software-Defined Data Assets (Bronze → Silver → Gold)
# =============================================================================

# ── Bronze Layer: Raw Ingestion ─────────────────────────────────────

@dg.asset(
    group_name="bronze",
    description="Raw order data ingested from the e-commerce transactional database",
    metadata={
        "source": "postgres://orders_db/public.orders",
        "owner": "data-platform-team",
        "refresh_frequency": "daily",
    },
    # Why compute_kind? It shows as a badge in the Dagster UI
    # so you can quickly see what technology each asset uses
    compute_kind="pandas",
)
def raw_orders(
    context: dg.AssetExecutionContext,
    storage: FileStorageResource,
) -> pd.DataFrame:
    """Ingest raw order data from the transactional database.

    In production, this would query a database replica or read from an API.
    Here we generate mock data for demonstration.

    Design decisions:
      - Append-only: raw data is never modified, only new batches added
      - Metadata columns (_ingested_at, _source): enable data lineage tracking
      - No business logic: bronze layer preserves the source exactly as-is
    """
    context.log.info("Ingesting raw orders from source system")

    # Generate mock order data (in production: query database or API)
    orders = pd.DataFrame({
        "order_id": range(1001, 1021),
        "customer_id": [101, 102, 103, 104, 105] * 4,
        "product_id": [1, 2, 3, 1, 2, 3, 1, 2, 3, 1,
                       2, 3, 1, 2, 3, 1, 2, 3, 1, 2],
        "quantity": [1, 2, 1, 3, 1, 2, 1, 1, 2, 1,
                     3, 1, 2, 1, 1, 2, 3, 1, 2, 1],
        "unit_price": [29.99, 49.99, 99.99, 29.99, 49.99,
                       99.99, 29.99, 49.99, 99.99, 29.99,
                       49.99, 99.99, 29.99, 49.99, 99.99,
                       29.99, 49.99, 99.99, 29.99, 49.99],
        "discount": [0.0, 0.1, 0.0, 0.05, 0.0,
                     0.15, 0.0, 0.0, 0.1, 0.0,
                     0.05, 0.0, 0.2, 0.0, 0.0,
                     0.1, 0.0, 0.05, 0.0, 0.15],
        "status": ["completed", "completed", "pending", "completed", "refunded",
                   "completed", "completed", "pending", "completed", "completed",
                   "refunded", "completed", "completed", "completed", "pending",
                   "completed", "completed", "completed", "refunded", "completed"],
        "created_at": pd.date_range("2024-01-01", periods=20, freq="4h"),
    })

    # Why add ingestion metadata?
    # - Track WHEN data was ingested (not just when the order was created)
    # - Track WHERE it came from (important when you have multiple sources)
    orders["_ingested_at"] = datetime.now()
    orders["_source"] = "orders_db_replica"

    context.log.info(f"Ingested {len(orders)} raw orders")
    return orders


@dg.asset(
    group_name="bronze",
    description="Product catalog data from the product management system",
    compute_kind="pandas",
)
def raw_products(context: dg.AssetExecutionContext) -> pd.DataFrame:
    """Ingest product catalog (refreshed weekly, not daily).

    Why a separate asset for products?
      - Different refresh cadence: products change weekly, orders change daily
      - Different source system: product API vs orders database
      - Independent materialization: can refresh products without re-running orders
    """
    products = pd.DataFrame({
        "product_id": [1, 2, 3],
        "product_name": ["Basic Widget", "Pro Widget", "Enterprise Widget"],
        "category": ["basic", "professional", "enterprise"],
        "cost_price": [10.0, 20.0, 40.0],
        "weight_kg": [0.5, 1.0, 2.5],
    })

    context.log.info(f"Ingested {len(products)} products from catalog")
    return products


# ── Silver Layer: Cleaned and Enriched ──────────────────────────────

@dg.asset(
    group_name="silver",
    description="Orders enriched with product details, validated and deduplicated",
    compute_kind="pandas",
)
def enriched_orders(
    context: dg.AssetExecutionContext,
    raw_orders: pd.DataFrame,
    raw_products: pd.DataFrame,
) -> pd.DataFrame:
    """Clean, validate, and enrich order data.

    This is where we transform raw data into a reliable, analysis-ready format.

    Silver layer responsibilities:
      1. Deduplication: remove duplicate order records
      2. Validation: filter out invalid records (negative amounts, etc.)
      3. Enrichment: join with reference data (products)
      4. Derivation: compute line_total, margin, etc.

    Why join at silver (not gold)?
      - Multiple gold assets may need product info
      - Compute the join ONCE, reuse in all downstream assets
      - Silver = single source of truth for enriched order data
    """
    context.log.info(f"Processing {len(raw_orders)} raw orders")

    # Step 1: Deduplication
    # Why drop duplicates? Source systems may send duplicates (retries, CDC duplicates)
    df = raw_orders.drop_duplicates(subset=["order_id"], keep="last")
    dupes_removed = len(raw_orders) - len(df)
    if dupes_removed > 0:
        context.log.warning(f"Removed {dupes_removed} duplicate orders")

    # Step 2: Validation — remove invalid records
    # Why validate here? Bronze preserves everything; Silver enforces data quality
    valid_mask = (
        (df["quantity"] > 0) &
        (df["unit_price"] > 0) &
        (df["discount"] >= 0) & (df["discount"] <= 1.0) &
        (df["customer_id"] > 0)
    )
    invalid_count = (~valid_mask).sum()
    if invalid_count > 0:
        context.log.warning(f"Filtered out {invalid_count} invalid orders")
    df = df[valid_mask]

    # Step 3: Enrichment — join with product data
    df = df.merge(raw_products, on="product_id", how="left")

    # Step 4: Derived fields
    # Why compute here? These are stable business calculations reused by all gold assets
    df["line_total"] = df["quantity"] * df["unit_price"] * (1 - df["discount"])
    df["margin"] = df["line_total"] - (df["quantity"] * df["cost_price"])
    df["margin_pct"] = (df["margin"] / df["line_total"] * 100).round(2)

    context.log.info(
        f"Enriched orders: {len(df)} valid records, "
        f"{invalid_count} filtered, {dupes_removed} deduped"
    )
    return df


# ── Gold Layer: Business Aggregations ───────────────────────────────

@dg.asset(
    group_name="gold",
    description="Revenue and margin metrics aggregated by product category",
    compute_kind="pandas",
)
def category_metrics(
    context: dg.AssetExecutionContext,
    enriched_orders: pd.DataFrame,
) -> pd.DataFrame:
    """Compute business metrics per product category.

    Why aggregate at the gold layer?
      - BI tools query gold tables directly (no complex SQL needed)
      - Pre-computation avoids redundant work by multiple analysts
      - Clear ownership: data team guarantees metric definitions
    """
    # Only count completed orders toward revenue metrics
    # Why? Pending/refunded orders are not realized revenue
    completed = enriched_orders[enriched_orders["status"] == "completed"]

    metrics = completed.groupby("category").agg(
        total_revenue=("line_total", "sum"),
        total_margin=("margin", "sum"),
        order_count=("order_id", "nunique"),
        avg_order_value=("line_total", "mean"),
        avg_discount=("discount", "mean"),
        total_units_sold=("quantity", "sum"),
    ).reset_index()

    # Derived metrics
    metrics["avg_margin_pct"] = (
        (metrics["total_margin"] / metrics["total_revenue"] * 100).round(2)
    )
    metrics["revenue_per_unit"] = (
        metrics["total_revenue"] / metrics["total_units_sold"]
    ).round(2)

    context.log.info(f"Computed metrics for {len(metrics)} categories")
    return metrics


@dg.asset(
    group_name="gold",
    description="Customer-level purchase behavior and segmentation",
    compute_kind="pandas",
)
def customer_summary(
    context: dg.AssetExecutionContext,
    enriched_orders: pd.DataFrame,
) -> pd.DataFrame:
    """Compute per-customer purchase summaries and assign segments.

    This powers the customer analytics dashboard and feeds into
    the ML team's churn prediction model.
    """
    completed = enriched_orders[enriched_orders["status"] == "completed"]

    summary = completed.groupby("customer_id").agg(
        total_spent=("line_total", "sum"),
        total_orders=("order_id", "nunique"),
        total_units=("quantity", "sum"),
        avg_order_value=("line_total", "mean"),
        first_order_date=("created_at", "min"),
        last_order_date=("created_at", "max"),
        favorite_category=("category", lambda x: x.mode().iloc[0] if len(x) > 0 else "unknown"),
    ).reset_index()

    # Customer segmentation based on total spend
    # Why segment? Different marketing strategies for different value tiers
    def assign_segment(spent: float) -> str:
        if spent >= 500:
            return "platinum"
        elif spent >= 200:
            return "gold"
        elif spent >= 50:
            return "silver"
        return "bronze"

    summary["segment"] = summary["total_spent"].apply(assign_segment)

    context.log.info(
        f"Computed summaries for {len(summary)} customers. "
        f"Segments: {summary['segment'].value_counts().to_dict()}"
    )
    return summary


# =============================================================================
# Partitioned Assets — Daily Incremental Processing
# =============================================================================

# Why partitions? In production, you process data incrementally (daily batches)
# instead of reprocessing the entire dataset every run.
# Benefits: O(delta) processing, targeted backfills, parallel execution.

daily_partitions = dg.DailyPartitionsDefinition(
    start_date="2024-01-01",
    end_date="2024-01-31",  # Bounded for this example
)


@dg.asset(
    partitions_def=daily_partitions,
    group_name="bronze_partitioned",
    description="Daily partitioned order ingestion",
    compute_kind="pandas",
)
def daily_orders(context: dg.AssetExecutionContext) -> pd.DataFrame:
    """Ingest orders for a specific date partition.

    Each run processes exactly ONE day of data.
    Dagster tracks which partitions have been materialized.

    Why partition at ingestion?
      - Source query is bounded: WHERE order_date = '2024-01-15'
      - Backfill a bad day = re-materialize that single partition
      - No risk of reprocessing unrelated data
    """
    partition_date = context.partition_key
    context.log.info(f"Ingesting orders for partition: {partition_date}")

    # Generate mock data for this partition
    import random
    random.seed(hash(partition_date))  # Deterministic per date
    n_orders = random.randint(5, 15)

    orders = pd.DataFrame({
        "order_id": range(n_orders),
        "amount": [round(random.uniform(10, 500), 2) for _ in range(n_orders)],
        "status": [random.choice(["completed", "pending", "refunded"])
                   for _ in range(n_orders)],
        "order_date": [partition_date] * n_orders,
    })

    context.log.info(f"Ingested {len(orders)} orders for {partition_date}")
    return orders


@dg.asset(
    partitions_def=daily_partitions,
    group_name="silver_partitioned",
    description="Daily cleaned orders (partitioned)",
    compute_kind="pandas",
)
def daily_cleaned_orders(
    context: dg.AssetExecutionContext,
    daily_orders: pd.DataFrame,
) -> pd.DataFrame:
    """Clean daily order partition.

    Key insight: When Dagster materializes partition "2024-01-15" for this asset,
    it automatically loads partition "2024-01-15" from daily_orders.
    Partition alignment is automatic — no manual wiring needed.
    """
    partition_date = context.partition_key
    df = daily_orders[daily_orders["status"] == "completed"].copy()
    df = df[df["amount"] > 0]

    context.log.info(
        f"Partition {partition_date}: "
        f"{len(df)}/{len(daily_orders)} orders after cleaning"
    )
    return df


# =============================================================================
# Definitions — The Central Registry
# =============================================================================

# Why a single Definitions object?
# 1. Dagster discovers ALL assets, resources, schedules from this one object
# 2. Configuration is validated at startup, not at runtime
# 3. The Dagster UI reads this to render the full asset graph

defs = dg.Definitions(
    assets=[
        # Non-partitioned pipeline (full refresh)
        raw_orders,
        raw_products,
        enriched_orders,
        category_metrics,
        customer_summary,
        # Partitioned pipeline (incremental)
        daily_orders,
        daily_cleaned_orders,
    ],
    resources={
        # Why configure resources here (not inside assets)?
        # - Swap between dev/staging/prod by changing resource config
        # - Asset code remains unchanged across environments
        "storage": FileStorageResource(base_path="/tmp/dagster_example"),
        "io_manager": ParquetIOManager(base_path="/tmp/dagster_parquet"),
    },
)


# =============================================================================
# Entry Point — For direct execution (testing/debugging)
# =============================================================================

if __name__ == "__main__":
    # Why materialize() for local testing?
    # - Runs the full asset graph without the Dagster UI/daemon
    # - Useful for quick smoke tests during development
    # - Uses in-memory IO manager for speed
    print("Materializing all non-partitioned assets...")
    result = dg.materialize(
        assets=[raw_orders, raw_products, enriched_orders,
                category_metrics, customer_summary],
        resources={
            "storage": FileStorageResource(base_path="/tmp/dagster_test"),
            "io_manager": dg.mem_io_manager,  # In-memory for testing
        },
    )

    if result.success:
        print("\nAll assets materialized successfully!")
        # Inspect outputs
        metrics = result.output_for_node("category_metrics")
        print(f"\nCategory Metrics:\n{metrics.to_string()}")

        customers = result.output_for_node("customer_summary")
        print(f"\nCustomer Summary:\n{customers.to_string()}")
    else:
        print("Materialization failed!")
        for event in result.all_events:
            if event.is_failure:
                print(f"  FAILED: {event.step_key} - {event.event_specific_data}")
