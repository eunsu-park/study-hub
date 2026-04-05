"""
Data ingestion and validation module.

Adapted from MLOps Lesson 12 §3.1.
Reads local Parquet, validates with Great Expectations, saves processed output.

Usage:
    python src/ingest.py [--source data/churn.parquet] [--output data/validated.parquet]
"""

import argparse
from datetime import datetime

import great_expectations as ge
import pandas as pd


class DataIngestion:
    """Data ingestion with validation."""

    def __init__(self, source_path: str):
        self.source_path = source_path

    def ingest(self) -> pd.DataFrame:
        """Load data and add ingestion metadata."""
        df = pd.read_parquet(self.source_path)
        df["ingestion_timestamp"] = datetime.now()
        return df

    def validate(self, df: pd.DataFrame) -> bool:
        """Run Great Expectations validation suite."""
        ge_df = ge.from_pandas(df)

        results = [
            ge_df.expect_column_to_exist("user_id"),
            ge_df.expect_column_to_exist("target"),
            ge_df.expect_column_values_to_not_be_null("user_id"),
            ge_df.expect_column_values_to_be_between("age", min_value=18, max_value=120),
            ge_df.expect_table_row_count_to_be_between(min_value=1000),
        ]

        passed = all(r.success for r in results)
        print(f"Validation: {'PASSED' if passed else 'FAILED'} ({sum(r.success for r in results)}/{len(results)} checks)")
        return passed

    def save(self, df: pd.DataFrame, output_path: str):
        """Save validated data."""
        df.to_parquet(output_path, index=False)
        print(f"Saved {len(df)} rows -> {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Ingest and validate data")
    parser.add_argument("--source", default="data/churn.parquet")
    parser.add_argument("--output", default="data/validated.parquet")
    args = parser.parse_args()

    pipeline = DataIngestion(args.source)
    df = pipeline.ingest()

    if pipeline.validate(df):
        pipeline.save(df, args.output)
    else:
        raise ValueError("Data validation failed — aborting ingestion.")


if __name__ == "__main__":
    main()
