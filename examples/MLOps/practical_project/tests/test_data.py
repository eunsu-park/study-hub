"""
Data pipeline unit tests.

Tests data generation and ingestion validation logic.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.generate_churn_data import generate_churn_data


class TestDataGeneration:
    """Tests for synthetic data generator."""

    def test_row_count(self):
        df = generate_churn_data(n_rows=100)
        assert len(df) == 100

    def test_required_columns(self):
        df = generate_churn_data(n_rows=50)
        required = [
            "user_id", "age", "tenure_months", "total_purchases",
            "avg_purchase_amount", "customer_segment", "target",
            "event_timestamp",
        ]
        for col in required:
            assert col in df.columns, f"Missing column: {col}"

    def test_target_is_binary(self):
        df = generate_churn_data(n_rows=500)
        assert set(df["target"].unique()).issubset({0, 1})

    def test_age_range(self):
        df = generate_churn_data(n_rows=500)
        assert df["age"].min() >= 18
        assert df["age"].max() <= 120

    def test_segments(self):
        df = generate_churn_data(n_rows=1000)
        expected = {"Premium", "Standard", "Basic"}
        assert set(df["customer_segment"].unique()) == expected

    def test_no_nulls_in_key_columns(self):
        df = generate_churn_data(n_rows=200)
        assert df["user_id"].notna().all()
        assert df["target"].notna().all()

    def test_reproducibility(self):
        df1 = generate_churn_data(n_rows=100, seed=42)
        df2 = generate_churn_data(n_rows=100, seed=42)
        pd.testing.assert_frame_equal(df1, df2)
