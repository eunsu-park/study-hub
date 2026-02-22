"""
Materialize features from offline store to online store (Redis).

Adapted from MLOps Lesson 11 §4.3.
This syncs the local Parquet data into Redis for low-latency serving.

Usage:
    python materialize.py [--full]
"""

import argparse
from datetime import datetime, timedelta

from feast import FeatureStore


def main():
    parser = argparse.ArgumentParser(description="Feast materialization")
    parser.add_argument(
        "--full", action="store_true",
        help="Full re-materialization (instead of incremental)",
    )
    args = parser.parse_args()

    store = FeatureStore(repo_path=".")

    if args.full:
        # Full materialization: sync everything from start to now
        start = datetime(2024, 1, 1)
        end = datetime.now()
        print(f"Full materialization: {start} -> {end}")
        store.materialize(start_date=start, end_date=end)
    else:
        # Incremental: only new data since last materialization
        end = datetime.now()
        print(f"Incremental materialization up to {end}")
        store.materialize_incremental(end_date=end)

    print("Materialization complete.")

    # Verify: list registered feature views
    for fv in store.list_feature_views():
        print(f"  Feature view: {fv.name} (entities: {[e.name for e in fv.entities]})")


if __name__ == "__main__":
    main()
