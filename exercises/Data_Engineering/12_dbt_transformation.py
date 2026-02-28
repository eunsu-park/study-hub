"""
Exercise Solutions: Lesson 12 - dbt Transformation

Covers:
  - Problem 1: Staging Model (stg_products)
  - Problem 2: Incremental Model (daily sales aggregates)
  - Problem 3: Write Tests (fct_sales model)

Note: Simulates dbt models and tests in pure Python.
      dbt SQL models are represented as Python transformation functions.
"""

from datetime import datetime, date, timedelta
import random


# ---------------------------------------------------------------------------
# Simulated dbt primitives
# ---------------------------------------------------------------------------

class SimulatedSource:
    """Represents a dbt source (raw table in the warehouse).

    In dbt:
        sources:
          - name: raw
            tables:
              - name: products
              - name: sales
    """
    def __init__(self, name: str, data: list[dict]):
        self.name = name
        self.data = data

    def ref(self) -> list[dict]:
        """Equivalent to {{ source('raw', 'products') }} in dbt SQL."""
        return self.data


class SimulatedModel:
    """Represents a dbt model (a SQL transformation that creates a table/view).

    In dbt, each model is a .sql file in the models/ directory.
    The materialization (view, table, incremental, ephemeral) determines
    how the results are stored.
    """
    def __init__(self, name: str, materialization: str = "table"):
        self.name = name
        self.materialization = materialization
        self.data: list[dict] = []
        self.run_count = 0

    def display(self, max_rows: int = 15) -> None:
        if not self.data:
            print(f"  (empty model: {self.name})")
            return
        cols = list(self.data[0].keys())
        widths = {c: max(len(c), max(len(str(r.get(c, ""))) for r in self.data[:max_rows])) + 2
                  for c in cols}
        header = "".join(c.ljust(widths[c]) for c in cols)
        print(f"  {header}")
        print(f"  {''.join('-' * widths[c] for c in cols)}")
        for i, row in enumerate(self.data[:max_rows]):
            line = "".join(str(row.get(c, "")).ljust(widths[c]) for c in cols)
            print(f"  {line}")
        if len(self.data) > max_rows:
            print(f"  ... ({len(self.data) - max_rows} more rows)")


# ---------------------------------------------------------------------------
# Problem 1: Staging Model
# Create a stg_products model from the raw products table.
# Convert prices to dollars and handle NULL values.
# ---------------------------------------------------------------------------

def create_raw_products() -> SimulatedSource:
    """Create sample raw products data with dirty values."""
    raw_data = [
        {"product_id": 1, "product_name": "Laptop Pro", "price_cents": 129999, "category": "Electronics", "brand": "TechCo"},
        {"product_id": 2, "product_name": "  Running Shoes  ", "price_cents": 8999, "category": "CLOTHING", "brand": "FitWear"},
        {"product_id": 3, "product_name": "Python Book", "price_cents": 4500, "category": "Books", "brand": None},
        {"product_id": 4, "product_name": None, "price_cents": 1299, "category": "food", "brand": "FreshCo"},
        {"product_id": 5, "product_name": "Wireless Mouse", "price_cents": None, "category": "Electronics", "brand": "TechCo"},
        {"product_id": 6, "product_name": "Coffee Maker", "price_cents": 7999, "category": "Home", "brand": "HomeBrew"},
        {"product_id": 7, "product_name": "Headphones", "price_cents": -500, "category": "Electronics", "brand": "SoundMax"},
    ]
    return SimulatedSource("raw_products", raw_data)


def stg_products(source: SimulatedSource) -> SimulatedModel:
    """
    dbt SQL equivalent (models/staging/stg_products.sql):

        {{
            config(
                materialized='view',
                description='Cleaned and standardized products from raw source'
            )
        }}

        WITH source AS (
            SELECT * FROM {{ source('raw', 'products') }}
        ),

        cleaned AS (
            SELECT
                product_id,
                COALESCE(TRIM(product_name), 'Unknown') AS product_name,
                CASE
                    WHEN price_cents IS NULL OR price_cents < 0 THEN 0.00
                    ELSE ROUND(price_cents / 100.0, 2)
                END AS price_dollars,
                UPPER(COALESCE(category, 'UNCATEGORIZED')) AS category,
                COALESCE(brand, 'Unknown') AS brand,
                CURRENT_TIMESTAMP AS _loaded_at
            FROM source
        )

        SELECT * FROM cleaned
    """
    model = SimulatedModel("stg_products", materialization="view")
    now = datetime.now().isoformat()

    for raw in source.ref():
        # Clean product_name: trim whitespace, handle NULL
        name = raw.get("product_name")
        clean_name = name.strip() if name else "Unknown"

        # Convert cents to dollars, handle NULL and negative values
        price_cents = raw.get("price_cents")
        if price_cents is None or price_cents < 0:
            price_dollars = 0.00
        else:
            price_dollars = round(price_cents / 100.0, 2)

        # Standardize category to uppercase, handle NULL
        category = raw.get("category")
        clean_category = category.upper() if category else "UNCATEGORIZED"

        # Handle NULL brand
        brand = raw.get("brand") or "Unknown"

        model.data.append({
            "product_id": raw["product_id"],
            "product_name": clean_name,
            "price_dollars": price_dollars,
            "category": clean_category,
            "brand": brand,
            "_loaded_at": now,
        })

    model.run_count += 1
    return model


def problem1_staging_model():
    """Build and display the stg_products model."""
    source = create_raw_products()
    print("\n  Raw Products (source):")
    raw_model = SimulatedModel("raw_products")
    raw_model.data = source.data
    raw_model.display()

    model = stg_products(source)
    print(f"\n  stg_products (materialized={model.materialization}):")
    model.display()

    print("\n  Transformations applied:")
    print("    - NULL product_name -> 'Unknown'")
    print("    - Whitespace trimmed from product_name")
    print("    - price_cents / 100 -> price_dollars (NULL/negative -> 0.00)")
    print("    - category -> UPPER case, NULL -> 'UNCATEGORIZED'")
    print("    - NULL brand -> 'Unknown'")
    return model


# ---------------------------------------------------------------------------
# Problem 2: Incremental Model
# Write a model that incrementally processes daily sales aggregates.
# ---------------------------------------------------------------------------

def create_raw_sales(days: int = 10, per_day: int = 20) -> SimulatedSource:
    """Create sample sales data across multiple days."""
    categories = ["Electronics", "Clothing", "Books", "Food"]
    base_date = date(2024, 11, 1)
    data = []
    for d in range(days):
        sale_date = base_date + timedelta(days=d)
        for i in range(per_day):
            data.append({
                "sale_id": d * per_day + i + 1,
                "sale_date": sale_date.isoformat(),
                "category": random.choice(categories),
                "amount": round(random.uniform(10, 500), 2),
                "updated_at": datetime(2024, 11, 1 + d, 23, 0).isoformat(),
            })
    return SimulatedSource("raw_sales", data)


class IncrementalModel(SimulatedModel):
    """Simulates a dbt incremental model.

    In dbt:
        {{
            config(
                materialized='incremental',
                unique_key='sale_date || category',
                incremental_strategy='merge',
            )
        }}

        SELECT
            sale_date,
            category,
            COUNT(*) as sale_count,
            SUM(amount) as total_revenue,
            AVG(amount) as avg_sale
        FROM {{ source('raw', 'sales') }}

        {% if is_incremental() %}
        WHERE updated_at > (SELECT MAX(updated_at) FROM {{ this }})
        {% endif %}

        GROUP BY sale_date, category
    """

    def __init__(self):
        super().__init__("daily_sales_agg", materialization="incremental")
        self.last_updated_at: str | None = None

    def run(self, source: SimulatedSource) -> dict:
        """Execute the incremental model.

        First run: full refresh (process all data).
        Subsequent runs: only process new/updated data (is_incremental=True).
        """
        is_incremental = self.run_count > 0
        stats = {"mode": "incremental" if is_incremental else "full_refresh", "rows_processed": 0}

        # Filter source data (incremental mode)
        if is_incremental and self.last_updated_at:
            source_data = [
                r for r in source.ref()
                if r["updated_at"] > self.last_updated_at
            ]
            print(f"  [incremental] Processing rows where updated_at > {self.last_updated_at}")
        else:
            source_data = source.ref()
            print(f"  [full_refresh] Processing all {len(source_data)} rows")

        stats["rows_processed"] = len(source_data)

        if not source_data:
            print("  [incremental] No new data to process")
            self.run_count += 1
            return stats

        # Aggregate
        agg: dict[tuple[str, str], dict] = {}
        for r in source_data:
            key = (r["sale_date"], r["category"])
            if key not in agg:
                agg[key] = {"sale_count": 0, "total_revenue": 0.0, "amounts": []}
            agg[key]["sale_count"] += 1
            agg[key]["total_revenue"] += r["amount"]
            agg[key]["amounts"].append(r["amount"])

        new_rows = []
        for (sale_date, category), stats_agg in agg.items():
            new_rows.append({
                "sale_date": sale_date,
                "category": category,
                "sale_count": stats_agg["sale_count"],
                "total_revenue": round(stats_agg["total_revenue"], 2),
                "avg_sale": round(stats_agg["total_revenue"] / stats_agg["sale_count"], 2),
            })

        # MERGE: upsert into existing data
        existing_keys = {(r["sale_date"], r["category"]) for r in self.data}
        updated, inserted = 0, 0
        for row in new_rows:
            key = (row["sale_date"], row["category"])
            if key in existing_keys:
                # Update existing row
                for i, existing in enumerate(self.data):
                    if (existing["sale_date"], existing["category"]) == key:
                        self.data[i] = row
                        updated += 1
                        break
            else:
                self.data.append(row)
                inserted += 1

        # Update watermark
        self.last_updated_at = max(r["updated_at"] for r in source_data)
        self.run_count += 1

        stats["updated"] = updated
        stats["inserted"] = inserted
        print(f"  Result: {inserted} inserted, {updated} updated")
        return stats


def problem2_incremental_model():
    """Demonstrate incremental processing with two runs."""
    source = create_raw_sales(days=5, per_day=15)
    model = IncrementalModel()

    # Run 1: Full refresh
    print("\n  === Run 1: Full Refresh ===")
    stats1 = model.run(source)
    print(f"  Stats: {stats1}")
    model.display(10)

    # Add new data (days 6-7)
    new_sales = create_raw_sales(days=7, per_day=15)
    source = new_sales  # Source now has 7 days

    # Run 2: Incremental
    print("\n  === Run 2: Incremental ===")
    stats2 = model.run(source)
    print(f"  Stats: {stats2}")
    model.display(15)

    # Run 3: No new data
    print("\n  === Run 3: No New Data ===")
    stats3 = model.run(source)
    print(f"  Stats: {stats3}")

    return model


# ---------------------------------------------------------------------------
# Problem 3: Write Tests
# Write tests for the fct_sales model.
# ---------------------------------------------------------------------------

class DbtTest:
    """Simulates dbt generic and custom tests.

    dbt tests in schema.yml:
        models:
          - name: fct_sales
            columns:
              - name: sale_id
                tests:
                  - unique
                  - not_null
              - name: amount
                tests:
                  - not_null
                  - dbt_utils.accepted_range:
                      min_value: 0
                      inclusive: true
    """

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.results: list[dict] = []

    def test_not_null(self, data: list[dict], column: str) -> bool:
        """Test that no values in the column are NULL.

        dbt equivalent: tests: [not_null]
        SQL: SELECT COUNT(*) FROM model WHERE column IS NULL
        """
        nulls = [i for i, row in enumerate(data) if row.get(column) is None]
        passed = len(nulls) == 0
        self.results.append({
            "test": f"not_null({column})",
            "passed": passed,
            "failures": len(nulls),
            "detail": f"Found {len(nulls)} NULL values" if not passed else "OK",
        })
        return passed

    def test_unique(self, data: list[dict], column: str) -> bool:
        """Test that all values in the column are unique.

        dbt equivalent: tests: [unique]
        SQL: SELECT column, COUNT(*) FROM model GROUP BY column HAVING COUNT(*) > 1
        """
        values = [row.get(column) for row in data]
        seen = set()
        duplicates = set()
        for v in values:
            if v in seen:
                duplicates.add(v)
            seen.add(v)
        passed = len(duplicates) == 0
        self.results.append({
            "test": f"unique({column})",
            "passed": passed,
            "failures": len(duplicates),
            "detail": f"Duplicate values: {duplicates}" if not passed else "OK",
        })
        return passed

    def test_accepted_range(self, data: list[dict], column: str,
                            min_value: float | None = None,
                            max_value: float | None = None) -> bool:
        """Test that values are within an accepted range.

        dbt equivalent: dbt_utils.accepted_range
        """
        failures = []
        for i, row in enumerate(data):
            val = row.get(column)
            if val is None:
                continue
            if min_value is not None and val < min_value:
                failures.append((i, val))
            if max_value is not None and val > max_value:
                failures.append((i, val))
        passed = len(failures) == 0
        range_str = f"[{min_value}, {max_value}]"
        self.results.append({
            "test": f"accepted_range({column}, {range_str})",
            "passed": passed,
            "failures": len(failures),
            "detail": f"Out-of-range values: {failures[:5]}" if not passed else "OK",
        })
        return passed

    def test_referential_integrity(self, data: list[dict], column: str,
                                    reference_values: set) -> bool:
        """Test referential integrity (foreign key check).

        dbt equivalent: relationships test
        """
        orphans = [row[column] for row in data
                    if row.get(column) is not None and row[column] not in reference_values]
        passed = len(orphans) == 0
        self.results.append({
            "test": f"referential_integrity({column})",
            "passed": passed,
            "failures": len(orphans),
            "detail": f"Orphan values: {set(orphans)}" if not passed else "OK",
        })
        return passed

    def report(self) -> None:
        """Display test results summary."""
        print(f"\n  dbt Test Results for '{self.model_name}':")
        print(f"  {'Test':<45} {'Status':<8} {'Failures':>8} {'Detail'}")
        print(f"  {'-'*45} {'-'*8} {'-'*8} {'-'*25}")
        for r in self.results:
            status = "PASS" if r["passed"] else "FAIL"
            print(f"  {r['test']:<45} {status:<8} {r['failures']:>8} {r['detail']}")
        total = len(self.results)
        passed = sum(1 for r in self.results if r["passed"])
        print(f"\n  Summary: {passed}/{total} passed")


def problem3_dbt_tests():
    """Run dbt-style tests on a fct_sales model."""
    # Sample fct_sales data (including some intentional issues for testing)
    fct_sales = [
        {"sale_id": 1, "product_id": 101, "customer_id": 1, "amount": 150.00, "sale_date": "2024-11-01"},
        {"sale_id": 2, "product_id": 102, "customer_id": 2, "amount": 89.99, "sale_date": "2024-11-01"},
        {"sale_id": 3, "product_id": 101, "customer_id": 3, "amount": 250.00, "sale_date": "2024-11-02"},
        {"sale_id": 4, "product_id": 103, "customer_id": 1, "amount": 45.00, "sale_date": "2024-11-02"},
        {"sale_id": 5, "product_id": 999, "customer_id": 2, "amount": 0.01, "sale_date": "2024-11-03"},  # orphan product_id
        {"sale_id": 6, "product_id": 102, "customer_id": 4, "amount": -10.00, "sale_date": "2024-11-03"},  # negative amount
        {"sale_id": 6, "product_id": 101, "customer_id": 3, "amount": 75.00, "sale_date": "2024-11-03"},  # duplicate sale_id
        {"sale_id": 7, "product_id": 103, "customer_id": None, "amount": 120.00, "sale_date": "2024-11-04"},  # NULL customer_id
    ]

    valid_product_ids = {101, 102, 103, 104}

    tester = DbtTest("fct_sales")

    print("\n  Running tests on fct_sales model:")

    # Test 1: sale_id should be unique
    tester.test_unique(fct_sales, "sale_id")

    # Test 2: sale_id should not be NULL
    tester.test_not_null(fct_sales, "sale_id")

    # Test 3: amount should not be NULL
    tester.test_not_null(fct_sales, "amount")

    # Test 4: amount should be positive (>= 0)
    tester.test_accepted_range(fct_sales, "amount", min_value=0.0)

    # Test 5: customer_id should not be NULL
    tester.test_not_null(fct_sales, "customer_id")

    # Test 6: product_id referential integrity
    tester.test_referential_integrity(fct_sales, "product_id", valid_product_ids)

    tester.report()

    print("\n  Schema YAML equivalent:")
    print("""
    models:
      - name: fct_sales
        description: "Fact table for sales transactions"
        columns:
          - name: sale_id
            tests: [unique, not_null]
          - name: amount
            tests:
              - not_null
              - dbt_utils.accepted_range:
                  min_value: 0
          - name: customer_id
            tests: [not_null]
          - name: product_id
            tests:
              - relationships:
                  to: ref('dim_products')
                  field: product_id
    """)

    return tester


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 70)
    print("Problem 1: Staging Model (stg_products)")
    print("=" * 70)
    problem1_staging_model()

    print()
    print("=" * 70)
    print("Problem 2: Incremental Model (Daily Sales Aggregates)")
    print("=" * 70)
    problem2_incremental_model()

    print()
    print("=" * 70)
    print("Problem 3: dbt Tests for fct_sales")
    print("=" * 70)
    problem3_dbt_tests()
