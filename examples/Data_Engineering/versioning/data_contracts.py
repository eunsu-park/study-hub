"""
Data Contract Implementation
============================
Demonstrates data contract patterns for reliable data pipelines:
- Schema contract definition using Pydantic
- JSON Schema-based contracts (language-agnostic)
- Contract validation at pipeline boundaries
- Schema evolution detection and classification
- Producer/consumer contract testing

Requirements:
    pip install pydantic pandas jsonschema

Usage:
    python data_contracts.py
"""

import json
import pandas as pd
from datetime import datetime
from enum import Enum
from typing import Optional, Any
from dataclasses import dataclass, field


# =============================================================================
# 1. Pydantic-Based Schema Contracts
# =============================================================================

# Why Pydantic for contracts?
# - Type-safe: catches type errors at validation time, not in production
# - Self-documenting: field descriptions ARE the contract documentation
# - Extensible: custom validators encode business rules
# - Fast: Pydantic v2 uses Rust-based validation (10-50x faster than v1)

from pydantic import BaseModel, Field, field_validator, model_validator


class OrderStatus(str, Enum):
    """Valid order statuses — the contract guarantees only these values.

    Why an Enum instead of a plain string?
    - Catches typos at validation time ("compelted" vs "completed")
    - Self-documenting: you can see all valid values in one place
    - IDE support: autocomplete and type checking
    """
    PENDING = "pending"
    COMPLETED = "completed"
    REFUNDED = "refunded"
    CANCELLED = "cancelled"


class OrderContract(BaseModel):
    """Schema contract for the orders dataset.

    This Pydantic model IS the contract. Every field declaration specifies:
    - Name and type (the schema)
    - Constraints (min/max, patterns, enums)
    - Description (documentation for consumers)
    - Nullability (Optional vs required)

    Contract Version: 2.0
    Owner: data-platform-team (@data-platform)
    Consumers: analytics-team, ml-team, finance-team
    SLA: Freshness < 1 hour, Completeness > 99%, Availability 99.9%
    """

    order_id: int = Field(
        ...,  # ... means required (no default)
        gt=0,
        description="Unique order identifier (auto-incrementing)",
    )
    customer_id: int = Field(
        ...,
        gt=0,
        description="Customer foreign key (references customers.customer_id)",
    )
    amount: float = Field(
        ...,
        ge=0,
        le=1_000_000,
        description="Order total in USD, after discounts, before tax",
    )
    currency: str = Field(
        default="USD",
        pattern=r"^[A-Z]{3}$",
        description="ISO 4217 currency code (3 uppercase letters)",
    )
    status: OrderStatus = Field(
        ...,
        description="Current order status",
    )
    created_at: datetime = Field(
        ...,
        description="Timestamp when the order was placed (UTC)",
    )
    updated_at: Optional[datetime] = Field(
        default=None,
        description="Timestamp of last status change (UTC, null if never updated)",
    )

    # ── Business Rule Validators ──────────────────────────────────

    @field_validator("amount")
    @classmethod
    def amount_precision(cls, v: float) -> float:
        """Ensure amount has at most 2 decimal places.

        Why? Financial data must have consistent precision.
        $99.999 stored as a float can cause rounding errors downstream.
        """
        return round(v, 2)

    @model_validator(mode="after")
    def updated_after_created(self) -> "OrderContract":
        """Business rule: updated_at must be >= created_at if set.

        Why? An order can't be updated before it was created.
        This catches data pipeline bugs that mix up timestamps.
        """
        if self.updated_at and self.updated_at < self.created_at:
            raise ValueError(
                f"updated_at ({self.updated_at}) is before "
                f"created_at ({self.created_at})"
            )
        return self


# =============================================================================
# 2. DataFrame Contract Validation
# =============================================================================

@dataclass
class ValidationResult:
    """Result of validating a DataFrame against a contract."""
    is_valid: bool
    total_rows: int
    valid_rows: int
    error_rows: int
    error_rate: float
    errors: list[dict] = field(default_factory=list)
    contract_version: str = "2.0"

    def summary(self) -> str:
        status = "PASSED" if self.is_valid else "FAILED"
        return (
            f"Contract Validation [{status}]\n"
            f"  Total rows:  {self.total_rows}\n"
            f"  Valid rows:  {self.valid_rows}\n"
            f"  Error rows:  {self.error_rows}\n"
            f"  Error rate:  {self.error_rate:.2%}\n"
            f"  Contract:    v{self.contract_version}"
        )


def validate_dataframe(
    df: pd.DataFrame,
    contract_class: type[BaseModel],
    error_threshold: float = 0.05,
    sample_errors: int = 5,
) -> ValidationResult:
    """Validate every row of a DataFrame against a Pydantic contract.

    Why validate at pipeline boundaries?
      - Fail fast: catch problems at ingestion, not in dashboards
      - Clear errors: which row, which field, what's wrong
      - Configurable threshold: allow some bad records (real data is messy)
        but reject the entire batch if too many fail

    Args:
        df: The DataFrame to validate
        contract_class: Pydantic model defining the contract
        error_threshold: Maximum allowed error rate (0.05 = 5%)
        sample_errors: Number of error examples to include in the result

    Returns:
        ValidationResult with pass/fail status and error details
    """
    errors = []
    valid_count = 0

    for idx, row in df.iterrows():
        try:
            contract_class(**row.to_dict())
            valid_count += 1
        except Exception as e:
            errors.append({
                "row_index": idx,
                "error": str(e),
                "data": {k: str(v) for k, v in row.to_dict().items()},
            })

    total = len(df)
    error_count = len(errors)
    error_rate = error_count / total if total > 0 else 0

    # Why a threshold instead of zero-tolerance?
    # Real-world data always has some quality issues. Zero-tolerance
    # means your pipeline never runs. A 1-5% threshold is practical.
    is_valid = error_rate <= error_threshold

    return ValidationResult(
        is_valid=is_valid,
        total_rows=total,
        valid_rows=valid_count,
        error_rows=error_count,
        error_rate=error_rate,
        errors=errors[:sample_errors],  # Only include a sample
    )


# =============================================================================
# 3. JSON Schema Contracts (Language-Agnostic)
# =============================================================================

# Why JSON Schema alongside Pydantic?
# - Language-agnostic: works with Python, Java, Go, JavaScript, etc.
# - Standard format: understood by schema registries (Confluent, Glue)
# - API boundaries: define contracts for REST APIs and event streams
# - Interop: share contracts between teams using different languages

ORDER_JSON_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "OrderRecord",
    "description": "Data contract for orders dataset (v2.0)",
    "type": "object",
    "required": ["order_id", "customer_id", "amount", "status", "created_at"],
    "properties": {
        "order_id": {
            "type": "integer",
            "minimum": 1,
            "description": "Unique order identifier",
        },
        "customer_id": {
            "type": "integer",
            "minimum": 1,
            "description": "Customer foreign key",
        },
        "amount": {
            "type": "number",
            "minimum": 0,
            "maximum": 1000000,
            "description": "Order total in USD",
        },
        "currency": {
            "type": "string",
            "pattern": "^[A-Z]{3}$",
            "default": "USD",
            "description": "ISO 4217 currency code",
        },
        "status": {
            "type": "string",
            "enum": ["pending", "completed", "refunded", "cancelled"],
            "description": "Current order status",
        },
        "created_at": {
            "type": "string",
            "format": "date-time",
            "description": "Order creation timestamp (ISO 8601)",
        },
        "updated_at": {
            "type": ["string", "null"],
            "format": "date-time",
            "description": "Last update timestamp (nullable)",
        },
    },
    # Why additionalProperties: false?
    # STRICT mode: rejects any columns not in the contract.
    # This prevents "schema drift" — new columns appearing silently.
    "additionalProperties": False,
}


def validate_json_schema(record: dict, schema: dict) -> tuple[bool, list[str]]:
    """Validate a single record against a JSON Schema contract.

    Returns (is_valid, list_of_errors).
    """
    try:
        from jsonschema import validate, ValidationError, Draft202012Validator

        validator = Draft202012Validator(schema)
        errors = list(validator.iter_errors(record))
        if errors:
            return False, [
                f"{'.'.join(str(p) for p in e.absolute_path)}: {e.message}"
                for e in errors
            ]
        return True, []
    except ImportError:
        # Fallback: basic type checking without jsonschema library
        errors = []
        for field_name in schema.get("required", []):
            if field_name not in record:
                errors.append(f"Missing required field: {field_name}")
        return len(errors) == 0, errors


# =============================================================================
# 4. Schema Evolution Detection
# =============================================================================

class ChangeType(Enum):
    """Types of schema changes."""
    ADD_COLUMN = "add_column"
    REMOVE_COLUMN = "remove_column"
    RENAME_COLUMN = "rename_column"
    CHANGE_TYPE = "change_type"
    CHANGE_CONSTRAINT = "change_constraint"
    CHANGE_NULLABILITY = "change_nullability"


@dataclass
class SchemaChange:
    """A detected change between two schema versions."""
    change_type: ChangeType
    column_name: str
    is_breaking: bool
    details: dict = field(default_factory=dict)
    migration_plan: str = ""


def detect_schema_changes(
    old_schema: dict,
    new_schema: dict,
) -> list[SchemaChange]:
    """Compare two JSON Schema versions and classify each change.

    Why automated change detection?
      - Humans miss subtle breaking changes in large schemas
      - Consistent classification across all contract updates
      - Feeds into CI/CD: block breaking changes without explicit approval
      - Documents exactly what changed between versions

    Classification rules:
      NON-BREAKING: Add optional column, widen type, relax constraint
      BREAKING: Remove column, narrow type, add required column, rename column
    """
    changes: list[SchemaChange] = []

    old_props = old_schema.get("properties", {})
    new_props = new_schema.get("properties", {})
    old_required = set(old_schema.get("required", []))
    new_required = set(new_schema.get("required", []))

    # ── Detect removed columns ─────────────────────────────────────
    for col in old_props:
        if col not in new_props:
            changes.append(SchemaChange(
                change_type=ChangeType.REMOVE_COLUMN,
                column_name=col,
                is_breaking=True,
                details={"old_definition": old_props[col]},
                migration_plan=(
                    f"1. Mark '{col}' as deprecated in contract v{new_schema.get('title', '?')}\n"
                    f"2. Notify consumers: analytics, ML, finance\n"
                    f"3. Allow 30-day transition period\n"
                    f"4. Remove column after all consumers have migrated"
                ),
            ))

    # ── Detect added columns ───────────────────────────────────────
    for col in new_props:
        if col not in old_props:
            is_required_and_no_default = (
                col in new_required and
                "default" not in new_props[col]
            )
            changes.append(SchemaChange(
                change_type=ChangeType.ADD_COLUMN,
                column_name=col,
                # Adding a required column without default = BREAKING
                # because existing producers don't produce this field
                is_breaking=is_required_and_no_default,
                details={
                    "new_definition": new_props[col],
                    "required": col in new_required,
                },
                migration_plan=(
                    f"Add '{col}' as optional with default first, "
                    f"then make required after all producers adopt."
                    if is_required_and_no_default else ""
                ),
            ))

    # ── Detect type changes ────────────────────────────────────────
    for col in old_props:
        if col in new_props:
            old_type = old_props[col].get("type")
            new_type = new_props[col].get("type")
            if old_type != new_type:
                # Type widening (int → number) is safe; narrowing is breaking
                safe_widenings = {
                    ("integer", "number"),  # int → float is safe
                }
                is_widening = (old_type, new_type) in safe_widenings
                changes.append(SchemaChange(
                    change_type=ChangeType.CHANGE_TYPE,
                    column_name=col,
                    is_breaking=not is_widening,
                    details={"old_type": old_type, "new_type": new_type},
                    migration_plan=(
                        "" if is_widening else
                        f"Add '{col}_v2' with new type alongside '{col}'. "
                        f"Deprecate '{col}' after consumer migration."
                    ),
                ))

            # Check constraint changes (e.g., max value changed)
            old_max = old_props[col].get("maximum")
            new_max = new_props[col].get("maximum")
            if old_max != new_max and old_max is not None and new_max is not None:
                changes.append(SchemaChange(
                    change_type=ChangeType.CHANGE_CONSTRAINT,
                    column_name=col,
                    # Relaxing a constraint is safe; tightening is breaking
                    is_breaking=new_max < old_max,
                    details={"old_max": old_max, "new_max": new_max},
                ))

    # ── Detect nullability changes ─────────────────────────────────
    for col in old_props:
        if col in new_props:
            old_nullable = col not in old_required
            new_nullable = col not in new_required
            if old_nullable != new_nullable:
                changes.append(SchemaChange(
                    change_type=ChangeType.CHANGE_NULLABILITY,
                    column_name=col,
                    # Making a nullable field required is breaking
                    # Making a required field nullable is safe
                    is_breaking=not new_nullable,  # nullable → required = breaking
                    details={
                        "old_nullable": old_nullable,
                        "new_nullable": new_nullable,
                    },
                ))

    return changes


# =============================================================================
# 5. Producer/Consumer Contract Test
# =============================================================================

def producer_test(
    producer_fn,
    contract_class: type[BaseModel],
    error_threshold: float = 0.01,
) -> bool:
    """Test that a producer function satisfies the contract.

    Why test the producer?
      - Catches contract violations in CI/CD, before deployment
      - The producer team runs this as part of their test suite
      - Prevents breaking consumers by deploying bad pipeline code

    Args:
        producer_fn: A callable that returns a DataFrame
        contract_class: The Pydantic contract model
        error_threshold: Max allowed error rate

    Returns:
        True if the producer output satisfies the contract
    """
    print("Running producer contract test...")
    df = producer_fn()
    result = validate_dataframe(df, contract_class, error_threshold)
    print(result.summary())

    if not result.is_valid:
        print("\nSample errors:")
        for err in result.errors:
            print(f"  Row {err['row_index']}: {err['error'][:100]}")
    return result.is_valid


def consumer_test(
    consumer_fn,
    contract_class: type[BaseModel],
) -> bool:
    """Test that a consumer function handles contract-compliant data correctly.

    Why test the consumer?
      - Ensures the consumer correctly uses the contracted fields
      - Catches issues when the consumer assumes fields not in the contract
      - Validates that the consumer produces expected output from known input

    Args:
        consumer_fn: A callable(DataFrame) -> Any
        contract_class: The Pydantic contract model (used to generate test data)

    Returns:
        True if the consumer handles contract data without errors
    """
    print("Running consumer contract test...")

    # Generate minimal valid test data from the contract
    # Why generate from the contract? Ensures the test data is always valid.
    test_records = [
        {
            "order_id": 1,
            "customer_id": 100,
            "amount": 99.99,
            "currency": "USD",
            "status": "completed",
            "created_at": datetime(2024, 1, 15, 10, 30),
            "updated_at": None,
        },
        {
            "order_id": 2,
            "customer_id": 101,
            "amount": 250.00,
            "currency": "USD",
            "status": "completed",
            "created_at": datetime(2024, 1, 16, 14, 0),
            "updated_at": datetime(2024, 1, 16, 15, 0),
        },
    ]

    # Validate test data against contract (should always pass)
    for record in test_records:
        contract_class(**record)

    test_df = pd.DataFrame(test_records)

    try:
        result = consumer_fn(test_df)
        print(f"  Consumer produced result: {type(result).__name__}")
        print(f"  Consumer contract test: PASSED")
        return True
    except Exception as e:
        print(f"  Consumer contract test: FAILED")
        print(f"  Error: {e}")
        return False


# =============================================================================
# Main Demo
# =============================================================================

def main():
    print("=" * 60)
    print("Data Contract Implementation Demo")
    print("=" * 60)

    # ── Demo 1: Pydantic Contract Validation ───────────────────────
    print("\n--- Demo 1: Pydantic Contract Validation ---\n")

    # Create test data with some intentional contract violations
    test_data = pd.DataFrame({
        "order_id": [1, 2, 3, -4, 5],          # -4 violates gt=0
        "customer_id": [101, 102, 103, 104, 0], # 0 violates gt=0
        "amount": [99.99, 149.50, 25.00, 299.99, -10.0],  # -10 violates ge=0
        "currency": ["USD", "USD", "EUR", "USD", "usd"],   # "usd" violates pattern
        "status": ["completed", "completed", "pending", "completed", "invalid"],
        "created_at": pd.date_range("2024-01-01", periods=5, freq="D"),
        "updated_at": [None, None, None, None, None],
    })

    result = validate_dataframe(test_data, OrderContract, error_threshold=0.5)
    print(result.summary())
    if result.errors:
        print(f"\nSample errors ({len(result.errors)} shown):")
        for err in result.errors:
            print(f"  Row {err['row_index']}: {err['error'][:120]}...")

    # ── Demo 2: JSON Schema Validation ─────────────────────────────
    print("\n\n--- Demo 2: JSON Schema Validation ---\n")

    valid_record = {
        "order_id": 42,
        "customer_id": 101,
        "amount": 199.99,
        "currency": "USD",
        "status": "completed",
        "created_at": "2024-01-15T10:30:00Z",
    }
    is_valid, errors = validate_json_schema(valid_record, ORDER_JSON_SCHEMA)
    print(f"Valid record:   is_valid={is_valid}, errors={errors}")

    invalid_record = {
        "order_id": -1,           # Violates minimum: 1
        "customer_id": 101,
        "amount": 2_000_000,      # Violates maximum: 1,000,000
        "status": "shipped",      # Not in enum
        "created_at": "2024-01-15T10:30:00Z",
        "extra_field": "oops",    # Violates additionalProperties: false
    }
    is_valid, errors = validate_json_schema(invalid_record, ORDER_JSON_SCHEMA)
    print(f"Invalid record: is_valid={is_valid}")
    for e in errors:
        print(f"  - {e}")

    # ── Demo 3: Schema Evolution Detection ─────────────────────────
    print("\n\n--- Demo 3: Schema Evolution Detection ---\n")

    # Simulate a schema evolution: v2.0 → v3.0
    order_schema_v3 = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "OrderRecord_v3",
        "type": "object",
        "required": ["order_id", "customer_id", "amount", "status",
                      "created_at", "tax_amount"],  # NEW required field!
        "properties": {
            "order_id": {"type": "integer", "minimum": 1},
            "customer_id": {"type": "integer", "minimum": 1},
            "amount": {
                "type": "number",
                "minimum": 0,
                "maximum": 2000000,     # CHANGED: relaxed from 1M to 2M
            },
            # "currency" REMOVED (breaking!)
            "status": {
                "type": "string",
                "enum": ["pending", "completed", "refunded", "cancelled", "processing"],
            },
            "created_at": {"type": "string", "format": "date-time"},
            "updated_at": {"type": ["string", "null"], "format": "date-time"},
            "tax_amount": {                     # ADDED: required without default (breaking!)
                "type": "number",
                "minimum": 0,
                "description": "Tax amount in USD",
            },
            "shipping_method": {                # ADDED: optional (non-breaking)
                "type": "string",
                "description": "Shipping method selected",
            },
        },
        "additionalProperties": False,
    }

    changes = detect_schema_changes(ORDER_JSON_SCHEMA, order_schema_v3)

    breaking = [c for c in changes if c.is_breaking]
    non_breaking = [c for c in changes if not c.is_breaking]

    print(f"Total changes: {len(changes)}")
    print(f"  Breaking:     {len(breaking)}")
    print(f"  Non-breaking: {len(non_breaking)}")

    print(f"\nBreaking changes (require consumer coordination):")
    for c in breaking:
        print(f"  [{c.change_type.value}] {c.column_name}")
        print(f"    Details: {c.details}")
        if c.migration_plan:
            print(f"    Migration: {c.migration_plan}")

    print(f"\nNon-breaking changes (safe to deploy):")
    for c in non_breaking:
        print(f"  [{c.change_type.value}] {c.column_name}")
        print(f"    Details: {c.details}")

    # ── Demo 4: Producer/Consumer Contract Tests ───────────────────
    print("\n\n--- Demo 4: Producer/Consumer Contract Tests ---\n")

    def mock_producer() -> pd.DataFrame:
        """Simulate a data producer (e.g., ETL pipeline output)."""
        return pd.DataFrame({
            "order_id": [1, 2, 3],
            "customer_id": [101, 102, 103],
            "amount": [99.99, 149.50, 25.00],
            "currency": ["USD", "USD", "USD"],
            "status": ["completed", "completed", "pending"],
            "created_at": pd.date_range("2024-01-01", periods=3, freq="D"),
            "updated_at": [None, None, None],
        })

    def mock_consumer(df: pd.DataFrame) -> dict:
        """Simulate a data consumer (e.g., analytics aggregation)."""
        completed = df[df["status"] == "completed"]
        return {
            "total_revenue": completed["amount"].sum(),
            "order_count": len(completed),
            "avg_order_value": completed["amount"].mean(),
        }

    # Run producer test
    producer_ok = producer_test(mock_producer, OrderContract)

    # Run consumer test
    consumer_ok = consumer_test(mock_consumer, OrderContract)

    print(f"\nOverall contract test result:")
    print(f"  Producer: {'PASS' if producer_ok else 'FAIL'}")
    print(f"  Consumer: {'PASS' if consumer_ok else 'FAIL'}")

    # ── Summary ────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Summary of Contract Patterns Demonstrated")
    print("=" * 60)
    print("""
    1. Pydantic Contracts
       - Type-safe, self-documenting schema definitions
       - Field validators for business rules
       - DataFrame-level validation with error thresholds

    2. JSON Schema Contracts
       - Language-agnostic (Python, Java, Go, JS)
       - Strict mode (additionalProperties: false)
       - Standard format for schema registries

    3. Schema Evolution Detection
       - Automated breaking/non-breaking classification
       - Migration plan generation for breaking changes
       - CI/CD integration: block unsafe changes

    4. Producer/Consumer Testing
       - Producer test: does my pipeline satisfy the contract?
       - Consumer test: does my code handle contract data correctly?
       - Run in CI/CD to prevent contract violations

    Key Principle:
       Validate at the BOUNDARY between producer and consumer.
       Fail FAST with clear error messages.
       Allow configurable thresholds (real data is never perfect).
    """)


if __name__ == "__main__":
    main()
