"""
Feast FeatureView definitions — multiple views + on-demand transformation.

Adapted from MLOps Lesson 11 §3.1-3.2.
Demonstrates:
  - Basic FeatureView with FileSource
  - On-demand (derived) features computed at retrieval time
  - Tags for team ownership

See practical_project/features/ for a simpler single-view version.
"""

from datetime import timedelta

import pandas as pd
from feast import FeatureView, Field, FileSource, on_demand_feature_view
from feast.types import Float32, Float64, Int64, String

from entities import product, user

# ── Data sources ──────────────────────────────────────────────────────────

user_source = FileSource(
    path="data/churn.parquet",
    timestamp_field="event_timestamp",
)

# ── User features ─────────────────────────────────────────────────────────

user_features = FeatureView(
    name="user_features",
    entities=[user],
    ttl=timedelta(days=365),
    schema=[
        Field(name="age", dtype=Int64),
        Field(name="tenure_months", dtype=Int64),
        Field(name="total_purchases", dtype=Int64),
        Field(name="avg_purchase_amount", dtype=Float64),
        Field(name="customer_segment", dtype=String),
        Field(name="days_since_last_purchase", dtype=Int64),
        Field(name="support_tickets", dtype=Int64),
    ],
    source=user_source,
    online=True,
    tags={"team": "ml-platform", "owner": "data-science"},
)

# ── On-demand (derived) features ──────────────────────────────────────────

@on_demand_feature_view(
    sources=[user_features],
    schema=[
        Field(name="purchase_frequency", dtype=Float32),
        Field(name="is_high_value", dtype=Int64),
    ],
)
def user_derived_features(inputs: pd.DataFrame) -> pd.DataFrame:
    """Compute derived features at retrieval time."""
    df = pd.DataFrame()
    df["purchase_frequency"] = (
        inputs["total_purchases"] / inputs["tenure_months"].clip(lower=1)
    )
    df["is_high_value"] = (inputs["avg_purchase_amount"] > 100).astype(int)
    return df
