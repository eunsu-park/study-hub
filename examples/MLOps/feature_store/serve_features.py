"""
Demonstrate offline (historical) and online feature retrieval.

Adapted from MLOps Lesson 11 §4.2-4.3.
Run after materialize.py to show both retrieval paths.

Usage:
    python serve_features.py
"""

from datetime import datetime

import pandas as pd
from feast import FeatureStore


def demo_historical_features(store: FeatureStore):
    """Retrieve historical features for training (offline store)."""
    print("=" * 60)
    print("OFFLINE STORE: Historical Feature Retrieval")
    print("=" * 60)

    # Entity dataframe with point-in-time timestamps
    entity_df = pd.DataFrame({
        "user_id": [1, 2, 3, 4, 5],
        "event_timestamp": pd.to_datetime([
            "2024-01-15 10:00:00",
            "2024-01-15 11:00:00",
            "2024-01-15 12:00:00",
            "2024-02-01 09:00:00",
            "2024-03-01 15:00:00",
        ]),
    })

    features = [
        "user_features:total_purchases",
        "user_features:avg_purchase_amount",
        "user_features:tenure_months",
        "user_features:days_since_last_purchase",
        "user_derived_features:purchase_frequency",
        "user_derived_features:is_high_value",
    ]

    training_df = store.get_historical_features(
        entity_df=entity_df,
        features=features,
    ).to_df()

    print(f"\nTraining data shape: {training_df.shape}")
    print(training_df.to_string(index=False))


def demo_online_features(store: FeatureStore):
    """Retrieve features for real-time inference (online store)."""
    print("\n" + "=" * 60)
    print("ONLINE STORE: Real-time Feature Retrieval")
    print("=" * 60)

    features = [
        "user_features:total_purchases",
        "user_features:avg_purchase_amount",
        "user_features:days_since_last_purchase",
        "user_derived_features:purchase_frequency",
    ]

    result = store.get_online_features(
        features=features,
        entity_rows=[
            {"user_id": 1},
            {"user_id": 2},
        ],
    ).to_dict()

    print("\nOnline features:")
    for key, values in result.items():
        print(f"  {key}: {values}")


def main():
    store = FeatureStore(repo_path=".")

    demo_historical_features(store)
    demo_online_features(store)


if __name__ == "__main__":
    main()
