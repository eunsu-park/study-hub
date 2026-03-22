"""
Feature Engineering - Exercise Solutions
=========================================
Lesson 15: Feature Engineering

Exercises cover:
  1. Customer churn feature engineering (synthetic telecom data)
  2. Time series feature extraction (synthetic website traffic)
  3. Text + numeric combined features (synthetic product data)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, mean_squared_error


# ============================================================
# Exercise 1: Customer Churn Feature Engineering
# Create meaningful features from synthetic telecom data.
# ============================================================
def exercise_1_customer_churn():
    """Feature engineering for telecom customer churn prediction.

    Feature engineering is often the single most impactful step in ML.
    Domain knowledge guides feature creation:
    - Ratio features capture relative behavior (avg monthly cost)
    - Binned features capture non-linear effects (tenure groups)
    - Interaction features capture combined effects
    """
    print("=" * 60)
    print("Exercise 1: Customer Churn Feature Engineering")
    print("=" * 60)

    # Generate synthetic telecom data
    np.random.seed(42)
    n = 2000

    tenure = np.random.randint(1, 73, n)
    monthly_charges = np.random.uniform(20, 120, n)
    total_charges = tenure * monthly_charges * np.random.uniform(0.9, 1.1, n)
    contract = np.random.choice(
        ["Month-to-month", "One year", "Two year"], n, p=[0.5, 0.3, 0.2]
    )
    internet = np.random.choice(
        ["DSL", "Fiber optic", "No"], n, p=[0.35, 0.45, 0.2]
    )
    payment = np.random.choice(
        ["Electronic check", "Mailed check", "Bank transfer", "Credit card"], n
    )

    # Churn probability depends on features (realistic pattern)
    churn_prob = (
        0.1
        + 0.3 * (contract == "Month-to-month")
        + 0.2 * (internet == "Fiber optic")
        + 0.15 * (payment == "Electronic check")
        - 0.01 * tenure
        + 0.002 * monthly_charges
    )
    churn_prob = np.clip(churn_prob, 0.05, 0.95)
    churn = (np.random.random(n) < churn_prob).astype(int)

    df = pd.DataFrame({
        "tenure": tenure,
        "monthly_charges": monthly_charges,
        "total_charges": total_charges,
        "contract_type": contract,
        "internet_service": internet,
        "payment_method": payment,
        "churn": churn,
    })

    print(f"Dataset: {len(df)} rows, churn rate: {churn.mean():.2%}")

    # --- Feature Engineering ---
    # 1. Ratio features: reveal spending patterns relative to tenure
    df["avg_monthly"] = df["total_charges"] / df["tenure"].clip(lower=1)
    df["charge_ratio"] = df["monthly_charges"] / df["avg_monthly"].clip(lower=1)

    # 2. Binned features: capture non-linear tenure effects
    df["tenure_group"] = pd.cut(df["tenure"], bins=[0, 12, 24, 48, 72],
                                 labels=["0-12", "12-24", "24-48", "48-72"])

    # 3. Binary flags
    df["is_month_to_month"] = (df["contract_type"] == "Month-to-month").astype(int)
    df["is_fiber"] = (df["internet_service"] == "Fiber optic").astype(int)
    df["is_echeck"] = (df["payment_method"] == "Electronic check").astype(int)

    # 4. Interaction features: combined risk factors
    df["month_fiber"] = df["is_month_to_month"] * df["is_fiber"]
    df["high_charge_short_tenure"] = (
        (df["monthly_charges"] > df["monthly_charges"].median()) &
        (df["tenure"] < 12)
    ).astype(int)

    # 5. Encoding: label encode categorical for tree-based models
    for col in ["contract_type", "internet_service", "payment_method", "tenure_group"]:
        df[col + "_enc"] = LabelEncoder().fit_transform(df[col].astype(str))

    # Prepare features
    feature_cols = [
        "tenure", "monthly_charges", "total_charges", "avg_monthly",
        "charge_ratio", "is_month_to_month", "is_fiber", "is_echeck",
        "month_fiber", "high_charge_short_tenure",
        "contract_type_enc", "internet_service_enc",
        "payment_method_enc", "tenure_group_enc",
    ]

    X = df[feature_cols].values
    y = df["churn"].values

    # Compare baseline (raw 3 features) vs engineered features
    raw_cols = ["tenure", "monthly_charges", "total_charges"]
    X_raw = df[raw_cols].values

    clf = GradientBoostingClassifier(n_estimators=100, random_state=42)

    cv_raw = cross_val_score(clf, X_raw, y, cv=5, scoring="f1")
    cv_eng = cross_val_score(clf, X, y, cv=5, scoring="f1")

    print(f"\nBaseline ({len(raw_cols)} features) F1: "
          f"{cv_raw.mean():.4f} +/- {cv_raw.std():.4f}")
    print(f"Engineered ({len(feature_cols)} features) F1: "
          f"{cv_eng.mean():.4f} +/- {cv_eng.std():.4f}")
    print(f"Improvement: {cv_eng.mean() - cv_raw.mean():+.4f}")

    # Feature importance
    clf.fit(X, y)
    importances = clf.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    print(f"\nTop 5 features:")
    for i in range(5):
        print(f"  {feature_cols[sorted_idx[i]]}: {importances[sorted_idx[i]]:.4f}")


# ============================================================
# Exercise 2: Time Series Feature Extraction
# Create temporal and rolling features from website traffic data.
# ============================================================
def exercise_2_time_series_features():
    """Time series feature engineering for website traffic prediction.

    Key feature types for time series:
    - Calendar features: capture weekly/monthly/seasonal patterns
    - Lag features: past values as predictors (autoregressive nature)
    - Rolling statistics: smooth noise and capture trends
    - Cyclical encoding: sin/cos transform for periodic features
      (so day 0 and day 6 are close, unlike raw integer encoding)
    """
    print("\n" + "=" * 60)
    print("Exercise 2: Time Series Feature Extraction")
    print("=" * 60)

    # Generate 1 year of daily website traffic
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=365, freq="D")
    trend = np.linspace(1000, 1500, 365)
    weekly = 200 * np.sin(2 * np.pi * np.arange(365) / 7)
    monthly = 100 * np.sin(2 * np.pi * np.arange(365) / 30)
    noise = np.random.randn(365) * 50

    page_views = trend + weekly + monthly + noise
    page_views = np.clip(page_views, 200, None)

    df = pd.DataFrame({"date": dates, "page_views": page_views})

    # --- Feature Engineering ---
    # 1. Calendar features
    df["day_of_week"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

    # 2. Cyclical encoding -- sin/cos pairs so the model knows
    # that Sunday (6) and Monday (0) are adjacent, not 6 units apart
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    # 3. Lag features -- shift(1) prevents leakage by only using
    # values known at prediction time (yesterday's traffic)
    for lag in [1, 7, 14, 28]:
        df[f"lag_{lag}"] = df["page_views"].shift(lag)

    # 4. Rolling statistics
    for window in [7, 14, 30]:
        df[f"rolling_mean_{window}"] = (
            df["page_views"].shift(1).rolling(window).mean()
        )
        df[f"rolling_std_{window}"] = (
            df["page_views"].shift(1).rolling(window).std()
        )

    # Drop rows with NaN from lag/rolling
    df = df.dropna().reset_index(drop=True)

    # Target: next-day page views
    df["target"] = df["page_views"]

    feature_cols = [c for c in df.columns
                    if c not in ["date", "page_views", "target"]]

    # Temporal split: last 30 days for test
    train = df.iloc[:-30]
    test = df.iloc[-30:]

    X_train, y_train = train[feature_cols].values, train["target"].values
    X_test, y_test = test[feature_cols].values, test["target"].values

    # Model
    gb = GradientBoostingRegressor(n_estimators=100, random_state=42)
    gb.fit(X_train, y_train)
    y_pred = gb.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"Features created: {len(feature_cols)}")
    print(f"Test RMSE: {rmse:.2f}")

    # Naive baseline: predict lag_1
    naive_pred = test["lag_1"].values
    naive_rmse = np.sqrt(mean_squared_error(y_test, naive_pred))
    print(f"Naive (lag_1) RMSE: {naive_rmse:.2f}")
    print(f"Improvement: {(naive_rmse - rmse) / naive_rmse * 100:.1f}%")

    # Top features
    importances = gb.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    print(f"\nTop 5 features:")
    for i in range(5):
        print(f"  {feature_cols[sorted_idx[i]]}: {importances[sorted_idx[i]]:.4f}")


# ============================================================
# Exercise 3: Text + Numeric Combined Features
# Build features from synthetic product data.
# ============================================================
def exercise_3_text_numeric():
    """Combine text-derived and numeric features for product rating prediction.

    Multi-modal feature engineering combines signals from different data types:
    - Text: lengths, keyword presence, TF-IDF statistics
    - Numeric: raw values, ratios, log transforms
    - Categorical: frequency encoding
    Each modality captures different aspects of the underlying phenomenon.
    """
    print("\n" + "=" * 60)
    print("Exercise 3: Text + Numeric Combined Features")
    print("=" * 60)

    np.random.seed(42)
    n = 500

    # Generate synthetic product data
    categories = np.random.choice(
        ["Electronics", "Books", "Clothing", "Home", "Sports"], n
    )
    prices = np.random.lognormal(3.5, 1.0, n).clip(5, 2000)
    num_reviews = np.random.poisson(50, n)

    descriptions = []
    for i in range(n):
        length = np.random.randint(10, 50)
        words = np.random.choice(
            ["great", "product", "quality", "poor", "excellent", "fast",
             "shipping", "love", "terrible", "good", "bad", "average",
             "recommend", "broken", "perfect", "waste", "amazing", "okay",
             "decent", "value", "price", "expensive", "cheap", "durable"],
            length
        )
        descriptions.append(" ".join(words))

    # Rating depends on description sentiment + price + reviews
    positive_words = {"great", "excellent", "love", "perfect", "amazing",
                      "good", "recommend", "durable"}
    negative_words = {"terrible", "poor", "bad", "broken", "waste"}

    ratings = []
    for i in range(n):
        words = set(descriptions[i].split())
        pos_count = len(words & positive_words)
        neg_count = len(words & negative_words)
        rating = 3.0 + 0.3 * pos_count - 0.4 * neg_count
        rating += np.random.randn() * 0.3
        rating -= 0.0005 * prices[i]
        rating += 0.005 * num_reviews[i]
        ratings.append(np.clip(rating, 1, 5))

    df = pd.DataFrame({
        "description": descriptions,
        "price": prices,
        "num_reviews": num_reviews,
        "category": categories,
        "rating": ratings,
    })

    # --- Feature Engineering ---
    # Text features
    df["desc_length"] = df["description"].apply(len)
    df["desc_word_count"] = df["description"].apply(lambda x: len(x.split()))
    df["avg_word_length"] = df["desc_length"] / df["desc_word_count"]
    df["has_positive"] = df["description"].apply(
        lambda x: int(bool(set(x.split()) & positive_words))
    )
    df["has_negative"] = df["description"].apply(
        lambda x: int(bool(set(x.split()) & negative_words))
    )
    df["sentiment_ratio"] = (df["has_positive"] - df["has_negative"] + 1) / 2

    # Numeric features
    df["log_price"] = np.log1p(df["price"])
    df["log_reviews"] = np.log1p(df["num_reviews"])
    df["price_per_review"] = df["price"] / (df["num_reviews"] + 1)

    # Category features: frequency encoding
    cat_freq = df["category"].value_counts(normalize=True).to_dict()
    df["category_freq"] = df["category"].map(cat_freq)

    # Category mean price (target-free statistic)
    cat_price = df.groupby("category")["price"].transform("mean")
    df["category_avg_price"] = cat_price
    df["price_vs_category"] = df["price"] / df["category_avg_price"]

    # Select features
    feature_cols = [
        "desc_length", "desc_word_count", "avg_word_length",
        "has_positive", "has_negative", "sentiment_ratio",
        "log_price", "log_reviews", "price_per_review",
        "category_freq", "price_vs_category",
    ]

    X = df[feature_cols].values
    y = df["rating"].values

    # Baseline: just raw numeric features
    X_raw = df[["price", "num_reviews"]].values

    gb = GradientBoostingRegressor(n_estimators=100, random_state=42)

    cv_raw = cross_val_score(gb, X_raw, y, cv=5,
                              scoring="neg_root_mean_squared_error")
    cv_eng = cross_val_score(gb, X, y, cv=5,
                              scoring="neg_root_mean_squared_error")

    print(f"Baseline (2 features) RMSE: {-cv_raw.mean():.4f}")
    print(f"Engineered ({len(feature_cols)} features) RMSE: {-cv_eng.mean():.4f}")
    print(f"Improvement: {(-cv_raw.mean() - (-cv_eng.mean())) / (-cv_raw.mean()) * 100:.1f}%")

    # Feature importance
    gb.fit(X, y)
    importances = gb.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    print(f"\nFeature importance ranking:")
    for i in range(len(feature_cols)):
        print(f"  {feature_cols[sorted_idx[i]]:<25s} {importances[sorted_idx[i]]:.4f}")


if __name__ == "__main__":
    exercise_1_customer_churn()
    exercise_2_time_series_features()
    exercise_3_text_numeric()
