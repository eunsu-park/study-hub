# ML Project Lifecycle

## Learning Objectives

After completing this lesson, you will be able to:

1. Describe each phase of the ML project lifecycle — problem definition, data collection, feature engineering, model training, validation, deployment, and monitoring — and explain how they connect
2. Translate business objectives into well-scoped ML problems by defining target variables, success metrics, and operational constraints
3. Implement data validation and quality checks at each stage of the ML pipeline to ensure data integrity
4. Apply best practices for feature engineering and model validation to build reproducible, reliable ML systems
5. Design a retraining strategy based on monitoring signals such as data drift and performance degradation

---

## 1. ML Project Phases Overview

Machine learning projects require managing the entire lifecycle from data collection to monitoring, beyond just training models.

```
┌─────────────────────────────────────────────────────────────────────┐
│                    ML Project Lifecycle                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐    │
│   │ Problem  │───▶│  Data    │───▶│ Feature  │───▶│  Model   │    │
│   │ Definition│    │Collection│    │Engineering│    │ Training │    │
│   └──────────┘    └──────────┘    └──────────┘    └────┬─────┘    │
│                                                         │          │
│   ┌──────────┐    ┌──────────┐    ┌──────────┐         │          │
│   │Monitoring│◀───│Deployment│◀───│Validation│◀────────┘          │
│   │          │    │          │    │          │                     │
│   └────┬─────┘    └──────────┘    └──────────┘                     │
│        │                                                            │
│        └──────────────── Retraining ─────────────────────────▶     │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 2. Problem Definition and Scope

### 2.1 Defining Business Objectives

```python
"""
ML Project Problem Definition Template
"""

project_definition = {
    # Business objective
    "business_objective": "Reduce customer churn rate by 30%",

    # ML problem definition
    "ml_problem": {
        "type": "binary_classification",
        "target": "is_churned",
        "success_metric": "precision_at_recall_80",
        "baseline": 0.65
    },

    # Constraints
    "constraints": {
        "latency": "< 100ms",
        "throughput": "1000 req/s",
        "model_size": "< 500MB",
        "interpretability": "high"  # Regulatory requirement
    },

    # Data requirements
    "data_requirements": {
        "historical_period": "2 years",
        "minimum_samples": 100000,
        "features": ["usage_patterns", "demographics", "support_tickets"]
    }
}
```

### 2.2 Defining Success Criteria

```python
"""
Defining Model Performance Criteria
"""

success_criteria = {
    # Offline metrics (model quality)
    "offline_metrics": {
        "accuracy": {"min": 0.85, "target": 0.90},
        "precision": {"min": 0.80, "target": 0.85},
        "recall": {"min": 0.75, "target": 0.80},
        "auc_roc": {"min": 0.85, "target": 0.90}
    },

    # Online metrics (business impact)
    "online_metrics": {
        "churn_rate_reduction": {"target": "30%"},
        "false_positive_cost": {"max": "$10K/month"}
    },

    # System metrics
    "system_metrics": {
        "p99_latency": {"max": "100ms"},
        "availability": {"min": "99.9%"},
        "throughput": {"min": "1000 req/s"}
    }
}
```

---

## 3. Data Collection and Preparation

### 3.1 Data Pipeline

```python
"""
Data Collection Pipeline Example
"""

from typing import Dict, Any
import pandas as pd
from datetime import datetime, timedelta

class DataPipeline:
    """Data collection and preparation pipeline"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.data_sources = config["data_sources"]

    def extract(self) -> Dict[str, pd.DataFrame]:
        """Extract data from various sources"""
        # Separate extraction per source — allows independent retry and schema validation
        data = {}

        # Extract from database
        data["transactions"] = self.query_database(
            query="SELECT * FROM transactions WHERE date > ?",
            params=[self.config["start_date"]]
        )

        # Extract from S3
        data["user_events"] = self.read_from_s3(
            bucket="data-lake",
            prefix=f"events/{self.config['date_partition']}/"
        )

        # Extract from API
        data["external_features"] = self.fetch_from_api(
            endpoint=self.config["external_api"]
        )

        return data

    def transform(self, raw_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Data transformation and preprocessing"""

        # Left join preserves all transactions even if event tracking was incomplete
        df = raw_data["transactions"].merge(
            raw_data["user_events"],
            on="user_id",
            how="left"
        )

        # Order matters: handle missing before outliers, since nulls can skew outlier detection
        df = self.handle_missing(df)

        # Handle outliers
        df = self.handle_outliers(df)

        # Convert data types
        df = self.convert_types(df)

        return df

    def validate(self, df: pd.DataFrame) -> bool:
        """Data quality validation"""
        # Validation as a separate step (not inside transform) — enables fail-fast
        # before expensive downstream processing
        validations = {
            "row_count": len(df) > self.config["min_rows"],
            "null_ratio": df.isnull().mean().max() < 0.1,
            "schema_match": self.check_schema(df),
            "value_ranges": self.check_value_ranges(df)
        }

        return all(validations.values())

    def load(self, df: pd.DataFrame, destination: str):
        """Save processed data"""
        # Embed version metadata in the data itself — enables lineage tracing
        # without relying on filename conventions alone
        df["_data_version"] = self.config["version"]
        df["_processed_at"] = datetime.now()

        # Parquet over CSV — columnar format gives 3-10x compression and schema enforcement
        df.to_parquet(
            f"{destination}/data_v{self.config['version']}.parquet",
            index=False
        )
```

### 3.2 Data Version Control (DVC)

```yaml
# dvc.yaml - DVC pipeline definition
# Declaring deps/outs lets DVC skip unchanged stages — saves hours on large pipelines
stages:
  prepare_data:
    cmd: python src/data/prepare.py
    deps:
      - src/data/prepare.py
      - data/raw/
    outs:
      - data/processed/train.parquet
      - data/processed/test.parquet

  train:
    cmd: python src/train.py
    deps:
      - src/train.py
      - data/processed/train.parquet
    # params tracked separately so DVC detects hyperparameter changes even if code is unchanged
    params:
      - train.epochs
      - train.learning_rate
    outs:
      - models/model.pkl
    metrics:
      # cache: false keeps metrics in git (not DVC storage) for easy diff across commits
      - metrics/train_metrics.json:
          cache: false
```

```bash
# DVC basic commands
# Start tracking data
dvc add data/raw/dataset.csv

# Run pipeline
dvc repro

# Check differences between versions
dvc diff

# Pull data (from remote storage)
dvc pull
```

---

## 4. Feature Engineering

### 4.1 Feature Definition and Calculation

```python
"""
Feature Engineering Pipeline
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

class FeatureEngineer:
    """Feature engineering class"""

    def __init__(self, feature_config: dict):
        self.config = feature_config
        self.encoders = {}
        self.scalers = {}

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features"""
        # Build into a new DataFrame to avoid mutating the original and to track feature lineage
        features = pd.DataFrame()

        # Time-based features — capture cyclical behavioral patterns (e.g., weekend vs weekday usage)
        features["hour"] = df["timestamp"].dt.hour
        features["day_of_week"] = df["timestamp"].dt.dayofweek
        features["is_weekend"] = features["day_of_week"].isin([5, 6]).astype(int)

        # Rolling aggregates capture recent behavioral trends, more predictive than lifetime totals
        features["total_purchases_30d"] = self.rolling_aggregate(
            df, "purchase_amount", window=30, agg="sum"
        )
        features["avg_session_duration_7d"] = self.rolling_aggregate(
            df, "session_duration", window=7, agg="mean"
        )

        # Ratio features — normalize by time to make users with different tenure comparable
        features["purchase_frequency"] = (
            df["purchase_count"] / df["days_since_signup"]
        ).fillna(0)

        # Interaction features — cross-feature ratios often reveal non-linear relationships
        features["value_per_session"] = (
            df["total_purchase_value"] / df["session_count"]
        ).fillna(0)

        return features

    def encode_categoricals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical variables"""
        for col in self.config["categorical_features"]:
            # fit_transform only on first call; transform-only on inference
            # to prevent data leakage from test/production data
            if col not in self.encoders:
                self.encoders[col] = LabelEncoder()
                df[col] = self.encoders[col].fit_transform(df[col])
            else:
                df[col] = self.encoders[col].transform(df[col])
        return df

    def scale_numericals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Scale numerical variables"""
        numerical_cols = self.config["numerical_features"]

        if "standard" not in self.scalers:
            self.scalers["standard"] = StandardScaler()
            df[numerical_cols] = self.scalers["standard"].fit_transform(
                df[numerical_cols]
            )
        else:
            df[numerical_cols] = self.scalers["standard"].transform(
                df[numerical_cols]
            )
        return df

    def save_transformers(self, path: str):
        """Save encoders/scalers"""
        # Persist transformers alongside the model — required for consistent
        # encoding/scaling during inference without re-fitting
        import joblib
        joblib.dump({
            "encoders": self.encoders,
            "scalers": self.scalers
        }, path)
```

### 4.2 Feature Store Integration

```python
"""
Feature Store Usage Example (Feast)
"""

from feast import FeatureStore

# Initialize Feature Store
fs = FeatureStore(repo_path="./feature_repo")

# Offline store for training — uses point-in-time joins to prevent future data leakage
training_df = fs.get_historical_features(
    entity_df=entity_df,  # entity_id, event_timestamp
    features=[
        "user_features:total_purchases",
        "user_features:avg_session_duration",
        "product_features:category",
        "product_features:price_range"
    ]
).to_df()

# Online store for inference — pre-materialized key-value lookup for low-latency (<10ms) serving
feature_vector = fs.get_online_features(
    features=[
        "user_features:total_purchases",
        "user_features:avg_session_duration"
    ],
    entity_rows=[{"user_id": 12345}]
).to_dict()
```

---

## 5. Model Training

### 5.1 Experiment Management

```python
"""
Model Training with Experiment Management
"""

import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import optuna

class ModelTrainer:
    """Model training class"""

    def __init__(self, experiment_name: str):
        mlflow.set_experiment(experiment_name)

    def train_with_tracking(
        self,
        X_train, y_train,
        X_val, y_val,
        params: dict
    ):
        """Training with MLflow tracking"""
        # Context manager ensures the run is closed even if training crashes mid-way
        with mlflow.start_run():
            # Log params first — if training fails, you can still diagnose which config caused it
            mlflow.log_params(params)

            # Record dataset sizes to detect silent data pipeline issues across runs
            mlflow.log_param("train_size", len(X_train))
            mlflow.log_param("val_size", len(X_val))

            # Train model
            model = RandomForestClassifier(**params)
            model.fit(X_train, y_train)

            # Validation
            val_predictions = model.predict(X_val)
            val_proba = model.predict_proba(X_val)[:, 1]

            # Calculate and log metrics
            metrics = self.calculate_metrics(y_val, val_predictions, val_proba)
            mlflow.log_metrics(metrics)

            # Log model with signature — signature enforces input schema at serving time,
            # catching schema mismatches before they cause silent prediction errors
            mlflow.sklearn.log_model(
                model, "model",
                signature=mlflow.models.infer_signature(X_train, val_predictions)
            )

            # Feature importance aids debugging and builds trust with stakeholders
            self.log_feature_importance(model, X_train.columns)

            return model, metrics

    def hyperparameter_tuning(self, X, y, n_trials: int = 100):
        """Hyperparameter tuning using Optuna"""
        # Optuna over GridSearch — uses Bayesian optimization to find good params
        # in far fewer trials than exhaustive search
        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                "max_depth": trial.suggest_int("max_depth", 3, 15),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10)
            }

            model = RandomForestClassifier(**params, random_state=42)
            scores = cross_val_score(model, X, y, cv=5, scoring="roc_auc")
            return scores.mean()

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)

        # Log best parameters
        with mlflow.start_run(run_name="best_params"):
            mlflow.log_params(study.best_params)
            mlflow.log_metric("best_auc", study.best_value)

        return study.best_params
```

### 5.2 Training Pipeline

```yaml
# training_pipeline.yaml
pipeline:
  name: "churn-prediction-training"
  # Schedule during off-peak hours to avoid competing with serving workloads for GPU/CPU
  schedule: "0 2 * * *"  # Daily at 2 AM

  stages:
    # Validation separate from feature engineering — fail fast on bad data before
    # spending compute on feature computation
    - name: data_validation
      script: src/validate_data.py
      inputs:
        - data/raw/
      outputs:
        - reports/data_validation.html

    - name: feature_engineering
      script: src/feature_engineering.py
      inputs:
        - data/raw/
      outputs:
        - data/features/

    - name: train
      script: src/train.py
      inputs:
        - data/features/
      params:
        - config/train_config.yaml
      outputs:
        - models/

    - name: evaluate
      script: src/evaluate.py
      inputs:
        - models/
        - data/features/test.parquet
      outputs:
        - reports/evaluation.html
```

---

## 6. Model Validation and Testing

### 6.1 Model Quality Gates

```python
"""
Model Quality Validation
"""

from typing import Dict, Any
import numpy as np

class ModelValidator:
    """Model validation class"""

    def __init__(self, quality_gates: Dict[str, float]):
        self.quality_gates = quality_gates

    def validate(self, metrics: Dict[str, float]) -> Dict[str, Any]:
        """Validate quality gates"""
        results = {
            "passed": True,
            "details": {}
        }

        for metric_name, threshold in self.quality_gates.items():
            actual_value = metrics.get(metric_name, 0)
            passed = actual_value >= threshold

            results["details"][metric_name] = {
                "threshold": threshold,
                "actual": actual_value,
                "passed": passed
            }

            if not passed:
                results["passed"] = False

        return results

    def compare_with_baseline(
        self,
        new_metrics: Dict[str, float],
        baseline_metrics: Dict[str, float],
        min_improvement: float = 0.01
    ) -> Dict[str, Any]:
        """Compare with baseline model"""
        # Always compare against production baseline — prevents deploying a model
        # that passes absolute thresholds but is worse than what's already serving
        results = {"improved": True, "details": {}}

        for metric_name in new_metrics:
            new_val = new_metrics[metric_name]
            baseline_val = baseline_metrics.get(metric_name, 0)
            improvement = (new_val - baseline_val) / baseline_val if baseline_val else 0

            results["details"][metric_name] = {
                "new": new_val,
                "baseline": baseline_val,
                "improvement": f"{improvement:.2%}"
            }

            # Check for performance degradation
            if new_val < baseline_val * (1 - min_improvement):
                results["improved"] = False

        return results

# Usage example
validator = ModelValidator({
    "accuracy": 0.85,
    "precision": 0.80,
    "recall": 0.75,
    "auc_roc": 0.85
})

validation_result = validator.validate(model_metrics)
if not validation_result["passed"]:
    raise ValueError(f"Model failed quality gates: {validation_result}")
```

### 6.2 A/B Test Preparation

```python
"""
A/B Test Configuration
"""

ab_test_config = {
    "experiment_name": "churn_model_v2",
    "variants": {
        "control": {
            "model_version": "v1.2.3",
            "traffic_percentage": 50
        },
        "treatment": {
            "model_version": "v2.0.0",
            "traffic_percentage": 50
        }
    },
    "metrics": {
        "primary": "conversion_rate",
        "secondary": ["latency_p99", "error_rate"]
    },
    "duration_days": 14,
    "min_sample_size": 10000
}
```

---

## 7. Deployment

### 7.1 Deployment Strategies

```python
"""
Model Deployment Strategies
"""

deployment_strategies = {
    "blue_green": {
        "description": "Deploy new version to separate environment then switch traffic",
        "rollback": "Immediate (switch traffic to previous environment)",
        "use_case": "When minimizing downtime is required"
    },
    "canary": {
        "description": "Gradually shift small portion of traffic to new version",
        "rollback": "Possible through traffic ratio adjustment",
        "use_case": "Risk minimization, A/B testing"
    },
    "shadow": {
        "description": "Replicate real traffic to test new model (results not used)",
        "rollback": "Not needed (no production impact)",
        "use_case": "New model validation"
    }
}
```

### 7.2 Deployment Code

```python
"""
Automated Model Deployment
"""

import mlflow
from mlflow.tracking import MlflowClient

class ModelDeployer:
    """Model deployment class"""

    def __init__(self, registry_uri: str):
        self.client = MlflowClient(registry_uri)

    def promote_to_production(
        self,
        model_name: str,
        version: str,
        archive_current: bool = True
    ):
        """Promote model to production"""
        # Archive before promoting — ensures exactly one production version exists
        # and preserves rollback history
        if archive_current:
            current_prod = self.get_production_model(model_name)
            if current_prod:
                self.client.transition_model_version_stage(
                    name=model_name,
                    version=current_prod.version,
                    stage="Archived"
                )

        # Promote new version to production
        self.client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage="Production"
        )

        print(f"Model {model_name} v{version} promoted to Production")

    def rollback(self, model_name: str):
        """Rollback to previous version"""
        # Pick the most recent archived version — most likely to be compatible
        # with current feature schema and data distribution
        versions = self.client.search_model_versions(
            f"name='{model_name}'"
        )

        archived = [v for v in versions if v.current_stage == "Archived"]
        if not archived:
            raise ValueError("No archived version available for rollback")

        latest_archived = max(archived, key=lambda x: int(x.version))

        # Archive current production
        current_prod = self.get_production_model(model_name)
        if current_prod:
            self.client.transition_model_version_stage(
                name=model_name,
                version=current_prod.version,
                stage="Archived"
            )

        # Execute rollback
        self.client.transition_model_version_stage(
            name=model_name,
            version=latest_archived.version,
            stage="Production"
        )

        print(f"Rolled back to v{latest_archived.version}")
```

---

## 8. Monitoring

### 8.1 Monitoring Metrics

```python
"""
Model Monitoring Configuration
"""

monitoring_config = {
    # Model performance metrics
    "model_metrics": {
        "accuracy": {"threshold": 0.85, "alert_on": "below"},
        "latency_p99": {"threshold": 100, "alert_on": "above", "unit": "ms"},
        "error_rate": {"threshold": 0.01, "alert_on": "above"}
    },

    # Data drift metrics
    "drift_metrics": {
        "psi": {"threshold": 0.1, "alert_on": "above"},  # Population Stability Index
        "ks_statistic": {"threshold": 0.1, "alert_on": "above"}
    },

    # System metrics
    "system_metrics": {
        "cpu_usage": {"threshold": 80, "alert_on": "above", "unit": "%"},
        "memory_usage": {"threshold": 80, "alert_on": "above", "unit": "%"},
        "gpu_utilization": {"threshold": 90, "alert_on": "above", "unit": "%"}
    }
}
```

### 8.2 Retraining Triggers

```python
"""
Automatic Retraining Triggers
"""

class RetrainingTrigger:
    """Retraining trigger class"""

    def __init__(self, config: dict):
        self.config = config

    def check_triggers(self, metrics: dict) -> dict:
        """Check if retraining is needed"""
        triggers = {
            "should_retrain": False,
            "reasons": []
        }

        # 1. Performance degradation — the most direct signal; catches concept drift
        if metrics.get("accuracy", 1.0) < self.config["min_accuracy"]:
            triggers["should_retrain"] = True
            triggers["reasons"].append("accuracy_degradation")

        # 2. Data drift (PSI) — an early warning before performance drops;
        # detects distribution shift even when labels aren't available yet
        if metrics.get("psi", 0) > self.config["max_psi"]:
            triggers["should_retrain"] = True
            triggers["reasons"].append("data_drift")

        # 3. Time-based retraining — safety net for gradual drift that metrics may miss
        days_since_training = metrics.get("days_since_training", 0)
        if days_since_training > self.config["max_days_without_training"]:
            triggers["should_retrain"] = True
            triggers["reasons"].append("scheduled_retrain")

        # 4. New data threshold
        if metrics.get("new_data_count", 0) > self.config["new_data_threshold"]:
            triggers["should_retrain"] = True
            triggers["reasons"].append("new_data_available")

        return triggers

# Configuration example
retrain_config = {
    "min_accuracy": 0.85,
    "max_psi": 0.1,
    "max_days_without_training": 30,
    "new_data_threshold": 100000
}

trigger = RetrainingTrigger(retrain_config)
result = trigger.check_triggers(current_metrics)

if result["should_retrain"]:
    print(f"Triggering retrain due to: {result['reasons']}")
    # trigger_training_pipeline()
```

---

## 9. Version Control Strategy

### 9.1 Comprehensive Version Control

```yaml
# version_management.yaml
versioning:
  # Data versioning
  data:
    strategy: "semantic"  # v1.0.0
    storage: "dvc"
    format: "parquet"

  # Code versioning
  code:
    strategy: "git"
    branching: "git-flow"

  # Model versioning
  model:
    strategy: "semantic"
    registry: "mlflow"
    stages: ["None", "Staging", "Production", "Archived"]

  # Feature versioning
  features:
    strategy: "semantic"
    store: "feast"

  # Track lineage relationships
  lineage:
    data_version -> code_version -> model_version
    features_version -> model_version
```

### 9.2 Semantic Versioning

```python
"""
Model Semantic Versioning
"""

# Version format: MAJOR.MINOR.PATCH
# MAJOR: Incompatible changes (new architecture, feature schema changes)
# MINOR: Feature additions (new features, hyperparameter changes)
# PATCH: Bug fixes, retraining

version_examples = {
    "1.0.0": "Initial production release",
    "1.0.1": "Retrained with same data/features",
    "1.1.0": "Added new features",
    "1.2.0": "Hyperparameter optimization",
    "2.0.0": "Model architecture change (RF -> XGBoost)"
}
```

---

## Exercises

### Exercise 1: Pipeline Design
Design an ML pipeline for an e-commerce recommendation system. Define each stage from data collection to monitoring.

### Exercise 2: Retraining Policy
Design a retraining policy for the following situation:
- 100,000 new orders daily
- Strong seasonality in product sales
- Model inference latency requirement < 50ms

---

## Summary

| Phase | Main Activities | Key Deliverables |
|------|----------|------------|
| Problem Definition | Business goals, ML problem definition | Project documentation |
| Data Preparation | Collection, validation, version control | Validated datasets |
| Feature Engineering | Feature creation, transformation | Feature pipeline |
| Model Training | Training, experiment management | Trained models, metrics |
| Validation | Quality gates, A/B testing | Validation reports |
| Deployment | Blue/Green, Canary | Serving endpoints |
| Monitoring | Performance, drift detection | Dashboards, alerts |

---

## References

- [MLOps Principles - ML System Design](https://ml-ops.org/)
- [Google MLOps Maturity Model](https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning)
- [Data Version Control (DVC)](https://dvc.org/doc)
