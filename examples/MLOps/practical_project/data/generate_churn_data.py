"""
Generate synthetic customer churn data for the MLOps practical project.

Creates ~5000 rows with realistic distributions for a binary classification task.
Columns: user_id, age, tenure_months, total_purchases, avg_purchase_amount,
         customer_segment, days_since_last_purchase, support_tickets, target (churn).

Usage:
    python data/generate_churn_data.py [--rows 5000] [--output data/churn.parquet]
"""

import argparse
from datetime import datetime, timedelta

import numpy as np
import pandas as pd


def generate_churn_data(n_rows: int = 5000, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic churn dataset."""
    rng = np.random.default_rng(seed)

    # User demographics
    user_ids = list(range(1, n_rows + 1))
    ages = rng.integers(18, 75, size=n_rows)
    tenure_months = rng.exponential(scale=24, size=n_rows).astype(int).clip(1, 120)
    segments = rng.choice(
        ["Premium", "Standard", "Basic"],
        size=n_rows,
        p=[0.15, 0.50, 0.35],
    )

    # Behavioral features
    total_purchases = rng.poisson(lam=15, size=n_rows).clip(0, 200)
    avg_purchase_amount = rng.lognormal(mean=3.5, sigma=0.8, size=n_rows).round(2)
    days_since_last = rng.exponential(scale=30, size=n_rows).astype(int).clip(0, 365)
    support_tickets = rng.poisson(lam=1.5, size=n_rows).clip(0, 20)

    # Churn target: higher churn probability for low tenure, high days_since_last,
    # many support tickets, and low purchase frequency.
    churn_score = (
        -0.02 * tenure_months
        + 0.01 * days_since_last
        + 0.15 * support_tickets
        - 0.03 * total_purchases
        + rng.normal(0, 0.5, size=n_rows)
    )
    churn_prob = 1 / (1 + np.exp(-churn_score))
    target = (rng.random(n_rows) < churn_prob).astype(int)

    # Timestamp for Feast entity_df
    event_timestamps = [
        datetime(2024, 1, 1) + timedelta(days=int(rng.integers(0, 180)))
        for _ in range(n_rows)
    ]

    df = pd.DataFrame({
        "user_id": user_ids,
        "age": ages,
        "tenure_months": tenure_months,
        "total_purchases": total_purchases,
        "avg_purchase_amount": avg_purchase_amount,
        "customer_segment": segments,
        "days_since_last_purchase": days_since_last,
        "support_tickets": support_tickets,
        "event_timestamp": event_timestamps,
        "target": target,
    })

    return df


def main():
    parser = argparse.ArgumentParser(description="Generate churn data")
    parser.add_argument("--rows", type=int, default=5000)
    parser.add_argument("--output", default="data/churn.parquet")
    parser.add_argument("--csv", action="store_true", help="Also save CSV")
    args = parser.parse_args()

    df = generate_churn_data(n_rows=args.rows)
    df.to_parquet(args.output, index=False)
    print(f"Generated {len(df)} rows -> {args.output}")
    print(f"  Churn rate: {df['target'].mean():.1%}")
    print(f"  Segments: {df['customer_segment'].value_counts().to_dict()}")

    if args.csv:
        csv_path = args.output.replace(".parquet", ".csv")
        df.to_csv(csv_path, index=False)
        print(f"  CSV copy -> {csv_path}")


if __name__ == "__main__":
    main()
