# 07. Model Registry

## Learning Objectives

After completing this lesson, you will be able to:

1. Define the role of a Model Registry in the ML lifecycle and explain how it enables collaboration between data scientists and ML engineers
2. Implement model versioning using semantic versioning conventions and register models with MLflow's Model Registry API
3. Manage model stage transitions (Staging → Production → Archived) and implement approval workflows for production promotion
4. Compare cloud-managed model registries (AWS SageMaker, Google Vertex AI, Azure ML) against self-hosted MLflow and evaluate the trade-offs
5. Design a governance framework for a model registry that includes access control, audit logging, and rollback procedures

---

## Overview

A **Model Registry** is a central storage system that manages the entire lifecycle of machine learning models. It provides systematic version control, metadata management, and deployment stage management for trained models. The model registry is a critical component that ensures collaboration between data scientists and ML engineers, enabling safe deployment and rollback of models.

## Core Concepts

### 1. Model Version Management

**Semantic Versioning**
```
MAJOR.MINOR.PATCH

MAJOR: Breaking changes (model architecture change, input/output change)
MINOR: Backward compatible feature additions (new features, performance improvements)
PATCH: Bug fixes, minor parameter adjustments
```

**Version Management Example (MLflow)**
```python
import mlflow
from mlflow.tracking import MlflowClient

client = MlflowClient()
model_name = "ChurnPredictionModel"

# Register model
result = mlflow.register_model(
    model_uri=f"runs:/{run_id}/model",
    name=model_name
)

print(f"Model registered: {result.name}")
print(f"Version: {result.version}")

# Set model description
client.update_model_version(
    name=model_name,
    version=result.version,
    description="XGBoost model with feature engineering v2"
)

# Add tags
client.set_model_version_tag(
    name=model_name,
    version=result.version,
    key="validation_auc",
    value="0.85"
)
```

### 2. Stage Management

Models progress through defined stages:

```
None → Staging → Production → Archived
```

**Stage Transition**
```python
from mlflow.tracking import MlflowClient

client = MlflowClient()

# Transition to Staging
client.transition_model_version_stage(
    name="ChurnPredictionModel",
    version="2",
    stage="Staging"
)

# Transition to Production (archive existing Production version)
client.transition_model_version_stage(
    name="ChurnPredictionModel",
    version="2",
    stage="Production",
    archive_existing_versions=True  # Move existing Production to Archived
)

# Rollback (move previous version back to Production)
client.transition_model_version_stage(
    name="ChurnPredictionModel",
    version="1",
    stage="Production"
)
```

### 3. Metadata Management

**Metadata Schema Example**
```python
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

@dataclass
class ModelMetadata:
    """Model metadata schema"""
    # Structured metadata as a dataclass — enforces a consistent schema across teams;
    # without this, teams use ad-hoc tags and crucial information gets lost

    # Basic information
    model_name: str
    version: str
    stage: str

    # Training information
    training_date: datetime
    training_duration_minutes: float
    dataset_version: str
    data_size: int

    # Performance metrics
    metrics: Dict[str, float]  # {"accuracy": 0.92, "f1": 0.88}

    # Model information
    framework: str  # "sklearn", "pytorch", "tensorflow"
    algorithm: str  # "RandomForest", "XGBoost"
    hyperparameters: Dict[str, any]

    # Deployment information
    deployed_by: str
    deployment_date: Optional[datetime]
    serving_endpoint: Optional[str]

    # Business information
    use_case: str
    owner: str
    tags: List[str]

    # Compliance — required for regulated industries (finance, healthcare);
    # audit trail proves which model was approved, by whom, and when
    approval_status: str  # "pending", "approved", "rejected"
    approver: Optional[str]
    approval_date: Optional[datetime]

# Example usage
metadata = ModelMetadata(
    model_name="ChurnPredictionModel",
    version="2.1.0",
    stage="Production",
    training_date=datetime.now(),
    training_duration_minutes=45.3,
    dataset_version="v2023.01",
    data_size=1_000_000,
    metrics={
        "accuracy": 0.92,
        "precision": 0.89,
        "recall": 0.87,
        "f1": 0.88,
        "auc": 0.94
    },
    framework="sklearn",
    algorithm="RandomForestClassifier",
    hyperparameters={
        "n_estimators": 100,
        "max_depth": 10,
        "min_samples_split": 5
    },
    deployed_by="ml-team@company.com",
    deployment_date=None,
    serving_endpoint=None,
    use_case="Customer churn prediction",
    owner="data-science-team",
    tags=["churn", "classification", "customer"],
    approval_status="approved",
    approver="ml-lead@company.com",
    approval_date=datetime.now()
)
```

## MLflow Model Registry Implementation

### 1. Basic Registry Operations

**Model Registration**
```python
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Set tracking URI
mlflow.set_tracking_uri("http://localhost:5000")

# Load data
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Start MLflow run
with mlflow.start_run() as run:
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Log parameters
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("random_state", 42)

    # Log metrics
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    mlflow.log_metric("train_accuracy", train_score)
    mlflow.log_metric("test_accuracy", test_score)

    # Log model
    mlflow.sklearn.log_model(
        model,
        "model",
        registered_model_name="IrisClassifier"
    )

    print(f"Run ID: {run.info.run_id}")
```

**Query Registry**
```python
from mlflow.tracking import MlflowClient

client = MlflowClient()

# List all registered models
for rm in client.search_registered_models():
    print(f"Name: {rm.name}")
    print(f"Latest Versions: {rm.latest_versions}")

# Get specific model details
model_name = "IrisClassifier"
model_versions = client.search_model_versions(f"name='{model_name}'")

for mv in model_versions:
    print(f"Version: {mv.version}")
    print(f"Stage: {mv.current_stage}")
    print(f"Run ID: {mv.run_id}")
    print(f"Status: {mv.status}")
```

### 2. Stage Transition Workflow

**Approval Workflow Implementation**
```python
from typing import Optional
from datetime import datetime
import mlflow
from mlflow.tracking import MlflowClient

class ModelApprovalWorkflow:
    """Model approval and stage transition workflow"""

    def __init__(self, tracking_uri: str):
        mlflow.set_tracking_uri(tracking_uri)
        self.client = MlflowClient()

    def request_staging(
        self,
        model_name: str,
        version: str,
        requester: str,
        reason: str
    ):
        """Request transition to Staging"""
        # Tags serve as a lightweight approval audit trail — stored alongside the model
        # so governance info is never separated from the artifact it governs
        self.client.set_model_version_tag(
            name=model_name,
            version=version,
            key="approval_status",
            value="staging_requested"
        )

        self.client.set_model_version_tag(
            name=model_name,
            version=version,
            key="requester",
            value=requester
        )

        self.client.set_model_version_tag(
            name=model_name,
            version=version,
            key="request_reason",
            value=reason
        )

        self.client.set_model_version_tag(
            name=model_name,
            version=version,
            key="request_date",
            value=datetime.now().isoformat()
        )

        print(f"Staging request submitted for {model_name} v{version}")

    def approve_staging(
        self,
        model_name: str,
        version: str,
        approver: str
    ):
        """Approve and transition to Staging"""

        # Transition to Staging
        self.client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage="Staging"
        )

        # Add approval tags
        self.client.set_model_version_tag(
            name=model_name,
            version=version,
            key="approval_status",
            value="approved_staging"
        )

        self.client.set_model_version_tag(
            name=model_name,
            version=version,
            key="approver",
            value=approver
        )

        self.client.set_model_version_tag(
            name=model_name,
            version=version,
            key="approval_date",
            value=datetime.now().isoformat()
        )

        print(f"{model_name} v{version} approved and moved to Staging")

    def request_production(
        self,
        model_name: str,
        version: str,
        requester: str,
        validation_results: dict
    ):
        """Request transition to Production"""
        # Attach validation results as evidence — approver can verify
        # that the model passed quality gates without re-running tests
        for metric, value in validation_results.items():
            self.client.set_model_version_tag(
                name=model_name,
                version=version,
                key=f"validation_{metric}",
                value=str(value)
            )

        # Production approval request
        self.client.set_model_version_tag(
            name=model_name,
            version=version,
            key="approval_status",
            value="production_requested"
        )

        self.client.set_model_version_tag(
            name=model_name,
            version=version,
            key="production_requester",
            value=requester
        )

        print(f"Production request submitted for {model_name} v{version}")

    def approve_production(
        self,
        model_name: str,
        version: str,
        approver: str,
        archive_existing: bool = True
    ):
        """Approve and transition to Production"""

        # Transition to Production
        self.client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage="Production",
            archive_existing_versions=archive_existing
        )

        # Add approval tags
        self.client.set_model_version_tag(
            name=model_name,
            version=version,
            key="approval_status",
            value="approved_production"
        )

        self.client.set_model_version_tag(
            name=model_name,
            version=version,
            key="production_approver",
            value=approver
        )

        self.client.set_model_version_tag(
            name=model_name,
            version=version,
            key="production_date",
            value=datetime.now().isoformat()
        )

        print(f"{model_name} v{version} deployed to Production")

# Example usage
workflow = ModelApprovalWorkflow("http://localhost:5000")

# Request Staging
workflow.request_staging(
    model_name="ChurnPredictionModel",
    version="2",
    requester="data-scientist@company.com",
    reason="Improved accuracy by 5%"
)

# Approve Staging
workflow.approve_staging(
    model_name="ChurnPredictionModel",
    version="2",
    approver="ml-lead@company.com"
)

# Request Production
workflow.request_production(
    model_name="ChurnPredictionModel",
    version="2",
    requester="ml-engineer@company.com",
    validation_results={
        "accuracy": 0.92,
        "precision": 0.89,
        "recall": 0.87,
        "latency_ms": 15
    }
)

# Approve Production
workflow.approve_production(
    model_name="ChurnPredictionModel",
    version="2",
    approver="ml-director@company.com",
    archive_existing=True
)
```

### 3. Model Comparison and Selection

**Model Comparison Tool**
```python
import pandas as pd
from mlflow.tracking import MlflowClient

class ModelComparator:
    """Compare models in registry"""

    def __init__(self, tracking_uri: str):
        self.client = MlflowClient(tracking_uri=tracking_uri)

    def compare_versions(
        self,
        model_name: str,
        metric_names: list
    ) -> pd.DataFrame:
        """Compare all versions of a model"""

        versions = self.client.search_model_versions(f"name='{model_name}'")

        comparison_data = []
        for version in versions:
            run = self.client.get_run(version.run_id)

            row = {
                "version": version.version,
                "stage": version.current_stage,
                "run_id": version.run_id,
                "creation_date": version.creation_timestamp
            }

            # Add metrics
            for metric in metric_names:
                row[metric] = run.data.metrics.get(metric, None)

            comparison_data.append(row)

        df = pd.DataFrame(comparison_data)
        df = df.sort_values("version", ascending=False)

        return df

    def get_best_model(
        self,
        model_name: str,
        metric_name: str,
        higher_is_better: bool = True
    ) -> dict:
        """Get best performing model version"""

        df = self.compare_versions(model_name, [metric_name])

        if higher_is_better:
            best_row = df.loc[df[metric_name].idxmax()]
        else:
            best_row = df.loc[df[metric_name].idxmin()]

        return {
            "version": best_row["version"],
            "stage": best_row["stage"],
            "metric_value": best_row[metric_name],
            "run_id": best_row["run_id"]
        }

# Example usage
comparator = ModelComparator("http://localhost:5000")

# Compare all versions
df = comparator.compare_versions(
    model_name="ChurnPredictionModel",
    metric_names=["accuracy", "precision", "recall", "f1"]
)
print(df)

# Find best model
best = comparator.get_best_model(
    model_name="ChurnPredictionModel",
    metric_name="f1",
    higher_is_better=True
)
print(f"Best model: version {best['version']} with F1={best['metric_value']}")
```

## CI/CD Integration

### 1. GitHub Actions Workflow

**Model Training and Registration Pipeline**
```yaml
# .github/workflows/train_and_register.yml
name: Train and Register Model

# Trigger on model code or data changes — avoids unnecessary retraining
# when only docs or infra changes are pushed
on:
  push:
    branches: [ main ]
    paths:
      - 'src/model/**'
      - 'data/**'
  workflow_dispatch:

env:
  MLFLOW_TRACKING_URI: ${{ secrets.MLFLOW_TRACKING_URI }}
  AWS_ACCESS_KEY_ID: ${{ secrets.AWS_ACCESS_KEY_ID }}
  AWS_SECRET_ACCESS_KEY: ${{ secrets.AWS_SECRET_ACCESS_KEY }}

jobs:
  train:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v3

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.9'

    - name: Install dependencies
      run: |
        pip install -r requirements.txt

    - name: Run training
      run: |
        python src/train.py \
          --data-path data/train.csv \
          --model-name ChurnPredictionModel

    - name: Get run ID
      id: get_run_id
      run: |
        RUN_ID=$(cat run_id.txt)
        echo "run_id=$RUN_ID" >> $GITHUB_OUTPUT

    - name: Register model
      run: |
        python src/register_model.py \
          --run-id ${{ steps.get_run_id.outputs.run_id }} \
          --model-name ChurnPredictionModel

    - name: Run validation
      id: validate
      run: |
        python src/validate_model.py \
          --model-name ChurnPredictionModel \
          --version latest

    - name: Comment PR with results
      if: github.event_name == 'pull_request'
      uses: actions/github-script@v6
      with:
        script: |
          const fs = require('fs');
          const results = JSON.parse(fs.readFileSync('validation_results.json'));

          const comment = `
          ## Model Training Results

          - **Accuracy**: ${results.accuracy}
          - **Precision**: ${results.precision}
          - **Recall**: ${results.recall}
          - **F1 Score**: ${results.f1}

          Model registered as version ${results.version}
          `;

          github.rest.issues.createComment({
            issue_number: context.issue.number,
            owner: context.repo.owner,
            repo: context.repo.repo,
            body: comment
          });
```

**Model Promotion Pipeline**
```yaml
# .github/workflows/promote_model.yml
name: Promote Model to Production

on:
  workflow_dispatch:
    inputs:
      model_name:
        description: 'Model name'
        required: true
        default: 'ChurnPredictionModel'
      version:
        description: 'Model version'
        required: true
      skip_validation:
        description: 'Skip validation tests'
        required: false
        default: 'false'

env:
  MLFLOW_TRACKING_URI: ${{ secrets.MLFLOW_TRACKING_URI }}

jobs:
  validate:
    if: ${{ github.event.inputs.skip_validation != 'true' }}
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v3

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.9'

    - name: Install dependencies
      run: pip install -r requirements.txt

    - name: Run validation tests
      run: |
        python tests/validate_production.py \
          --model-name ${{ github.event.inputs.model_name }} \
          --version ${{ github.event.inputs.version }}

    - name: Performance benchmark
      run: |
        python tests/benchmark.py \
          --model-name ${{ github.event.inputs.model_name }} \
          --version ${{ github.event.inputs.version }}

  promote:
    needs: validate
    if: always() && (needs.validate.result == 'success' || needs.validate.result == 'skipped')
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v3

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.9'

    - name: Install MLflow
      run: pip install mlflow boto3

    - name: Promote to Production
      run: |
        python scripts/promote_to_production.py \
          --model-name ${{ github.event.inputs.model_name }} \
          --version ${{ github.event.inputs.version }} \
          --approver ${{ github.actor }}

    - name: Create GitHub Release
      uses: actions/create-release@v1
      env:
        GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
      with:
        tag_name: ${{ github.event.inputs.model_name }}-v${{ github.event.inputs.version }}
        release_name: ${{ github.event.inputs.model_name }} v${{ github.event.inputs.version }}
        body: |
          Model promoted to Production
          - Model: ${{ github.event.inputs.model_name }}
          - Version: ${{ github.event.inputs.version }}
          - Approver: ${{ github.actor }}
        draft: false
        prerelease: false
```

### 2. Promotion Script

**promote_to_production.py**
```python
import argparse
from datetime import datetime
import mlflow
from mlflow.tracking import MlflowClient

def promote_to_production(
    model_name: str,
    version: str,
    approver: str,
    archive_existing: bool = True
):
    """Promote model to Production stage"""

    client = MlflowClient()

    # Get current Production model
    production_versions = client.get_latest_versions(
        model_name,
        stages=["Production"]
    )

    if production_versions:
        current_prod = production_versions[0]
        print(f"Current Production version: {current_prod.version}")

    # Transition new version to Production
    client.transition_model_version_stage(
        name=model_name,
        version=version,
        stage="Production",
        archive_existing_versions=archive_existing
    )

    # Add metadata
    client.set_model_version_tag(
        name=model_name,
        version=version,
        key="promoted_by",
        value=approver
    )

    client.set_model_version_tag(
        name=model_name,
        version=version,
        key="promotion_date",
        value=datetime.now().isoformat()
    )

    client.update_model_version(
        name=model_name,
        version=version,
        description=f"Promoted to Production by {approver} on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )

    print(f"✅ {model_name} v{version} promoted to Production")

    if production_versions and archive_existing:
        print(f"📦 Previous version {current_prod.version} archived")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--approver", required=True)
    parser.add_argument("--keep-existing", action="store_true")

    args = parser.parse_args()

    promote_to_production(
        model_name=args.model_name,
        version=args.version,
        approver=args.approver,
        archive_existing=not args.keep_existing
    )
```

## Rollback Strategy

### 1. Rollback Implementation

**Model Rollback Manager**
```python
from typing import Optional
from datetime import datetime
import mlflow
from mlflow.tracking import MlflowClient

class ModelRollback:
    """Model rollback management"""

    def __init__(self, tracking_uri: str):
        mlflow.set_tracking_uri(tracking_uri)
        self.client = MlflowClient()

    def get_production_history(self, model_name: str) -> list:
        """Get Production stage history"""

        versions = self.client.search_model_versions(f"name='{model_name}'")

        production_history = []
        for version in versions:
            # Check if version was in Production
            if version.current_stage == "Production" or \
               self.client.get_model_version(model_name, version.version).current_stage == "Archived":

                tags = {tag.key: tag.value for tag in version.tags}

                if "promotion_date" in tags:
                    production_history.append({
                        "version": version.version,
                        "current_stage": version.current_stage,
                        "promotion_date": tags.get("promotion_date"),
                        "promoted_by": tags.get("promoted_by"),
                        "run_id": version.run_id
                    })

        # Sort by promotion date (descending)
        production_history.sort(
            key=lambda x: x["promotion_date"],
            reverse=True
        )

        return production_history

    def rollback_to_previous(
        self,
        model_name: str,
        reason: str,
        rollback_by: str
    ) -> Optional[str]:
        """Rollback to previous Production version"""
        # Use production history (not just "latest Archived") — ensures we restore
        # the version that was actually serving, not an unrelated archived experiment
        history = self.get_production_history(model_name)

        if len(history) < 2:
            print("No previous Production version available for rollback")
            return None

        current_version = history[0]["version"]
        previous_version = history[1]["version"]

        print(f"Rolling back from v{current_version} to v{previous_version}")

        # Archive (not delete) current version — preserves the artifact for post-mortem
        # analysis of what went wrong
        self.client.transition_model_version_stage(
            name=model_name,
            version=current_version,
            stage="Archived"
        )

        # Record rollback reason as tag — creates an audit trail explaining why
        # this version was removed from production
        self.client.set_model_version_tag(
            name=model_name,
            version=current_version,
            key="rollback_reason",
            value=reason
        )

        self.client.set_model_version_tag(
            name=model_name,
            version=current_version,
            key="rolled_back_by",
            value=rollback_by
        )

        self.client.set_model_version_tag(
            name=model_name,
            version=current_version,
            key="rollback_date",
            value=datetime.now().isoformat()
        )

        # Restore previous version to Production
        self.client.transition_model_version_stage(
            name=model_name,
            version=previous_version,
            stage="Production"
        )

        # Add restoration tags
        self.client.set_model_version_tag(
            name=model_name,
            version=previous_version,
            key="restored_to_production",
            value="true"
        )

        self.client.set_model_version_tag(
            name=model_name,
            version=previous_version,
            key="restoration_date",
            value=datetime.now().isoformat()
        )

        print(f"✅ Rollback complete: v{previous_version} restored to Production")

        return previous_version

    def rollback_to_specific(
        self,
        model_name: str,
        target_version: str,
        reason: str,
        rollback_by: str
    ):
        """Rollback to specific version"""

        # Get current Production version
        production_versions = self.client.get_latest_versions(
            model_name,
            stages=["Production"]
        )

        if not production_versions:
            print("No Production version found")
            return

        current_version = production_versions[0].version

        if current_version == target_version:
            print(f"v{target_version} is already in Production")
            return

        print(f"Rolling back from v{current_version} to v{target_version}")

        # Archive current version
        self.client.transition_model_version_stage(
            name=model_name,
            version=current_version,
            stage="Archived"
        )

        # Add rollback metadata
        self.client.set_model_version_tag(
            name=model_name,
            version=current_version,
            key="rollback_reason",
            value=reason
        )

        # Restore target version
        self.client.transition_model_version_stage(
            name=model_name,
            version=target_version,
            stage="Production"
        )

        self.client.set_model_version_tag(
            name=model_name,
            version=target_version,
            key="restored_by",
            value=rollback_by
        )

        print(f"✅ Rollback complete: v{target_version} restored to Production")

# Example usage
rollback = ModelRollback("http://localhost:5000")

# View Production history
history = rollback.get_production_history("ChurnPredictionModel")
print("Production History:")
for item in history:
    print(f"  v{item['version']}: {item['promotion_date']} by {item['promoted_by']}")

# Rollback to previous version
rollback.rollback_to_previous(
    model_name="ChurnPredictionModel",
    reason="Performance degradation detected in production",
    rollback_by="ops-team@company.com"
)

# Rollback to specific version
rollback.rollback_to_specific(
    model_name="ChurnPredictionModel",
    target_version="3",
    reason="Critical bug found in v5",
    rollback_by="ml-engineer@company.com"
)
```

### 2. Automated Rollback Triggers

**Performance Monitor with Auto-Rollback**
```python
import time
from typing import Callable
import mlflow
from mlflow.tracking import MlflowClient

class ProductionMonitor:
    """Production model monitor with auto-rollback"""

    def __init__(
        self,
        tracking_uri: str,
        model_name: str,
        check_interval_seconds: int = 300
    ):
        mlflow.set_tracking_uri(tracking_uri)
        self.client = MlflowClient()
        self.model_name = model_name
        self.check_interval = check_interval_seconds
        self.rollback_manager = ModelRollback(tracking_uri)

    def check_metric_threshold(
        self,
        metric_name: str,
        threshold: float,
        comparison: str = "greater_than"
    ) -> bool:
        """Check if metric meets threshold"""

        # Get current Production model
        production_versions = self.client.get_latest_versions(
            self.model_name,
            stages=["Production"]
        )

        if not production_versions:
            return True

        version = production_versions[0]
        run = self.client.get_run(version.run_id)

        metric_value = run.data.metrics.get(metric_name)

        if metric_value is None:
            return True

        if comparison == "greater_than":
            return metric_value > threshold
        elif comparison == "less_than":
            return metric_value < threshold
        else:
            return True

    def monitor_with_rollback(
        self,
        metric_checks: list,
        rollback_on_failure: bool = True
    ):
        """Monitor metrics and auto-rollback on failure"""

        print(f"Starting production monitoring for {self.model_name}")

        while True:
            try:
                all_passed = True
                failed_checks = []

                for check in metric_checks:
                    passed = self.check_metric_threshold(
                        metric_name=check["metric"],
                        threshold=check["threshold"],
                        comparison=check["comparison"]
                    )

                    if not passed:
                        all_passed = False
                        failed_checks.append(check["metric"])

                if not all_passed:
                    print(f"❌ Metric checks failed: {failed_checks}")

                    if rollback_on_failure:
                        print("Initiating automatic rollback...")
                        self.rollback_manager.rollback_to_previous(
                            model_name=self.model_name,
                            reason=f"Auto-rollback: Failed checks {failed_checks}",
                            rollback_by="automated-monitor"
                        )
                        print("✅ Automatic rollback completed")
                        break
                else:
                    print("✅ All metric checks passed")

                time.sleep(self.check_interval)

            except Exception as e:
                print(f"Error in monitoring: {e}")
                time.sleep(self.check_interval)

# Example usage
monitor = ProductionMonitor(
    tracking_uri="http://localhost:5000",
    model_name="ChurnPredictionModel",
    check_interval_seconds=300  # 5 minutes
)

# Define metric thresholds
metric_checks = [
    {
        "metric": "accuracy",
        "threshold": 0.85,
        "comparison": "greater_than"
    },
    {
        "metric": "latency_ms",
        "threshold": 100,
        "comparison": "less_than"
    }
]

# Start monitoring with auto-rollback
monitor.monitor_with_rollback(
    metric_checks=metric_checks,
    rollback_on_failure=True
)
```

## Best Practices

### 1. Version Management
- Use semantic versioning consistently
- Document all changes in model descriptions
- Tag models with business-relevant metadata
- Keep training scripts version controlled

### 2. Stage Transitions
- Require approval for Production deployments
- Validate models thoroughly in Staging
- Archive old Production versions (don't delete)
- Document rollback procedures

### 3. Metadata
- Record all relevant training parameters
- Log data version used for training
- Track model dependencies and frameworks
- Include business context in tags

### 4. CI/CD Integration
- Automate training and registration
- Run validation tests before promotion
- Use pull requests for model changes
- Create releases for Production deployments

### 5. Rollback Strategy
- Keep previous Production versions accessible
- Monitor Production models continuously
- Define clear rollback criteria
- Automate rollback for critical failures

## Practice Exercise

**Task**: Implement a complete model registry workflow

1. Train multiple model versions with different hyperparameters
2. Register all models in MLflow Registry
3. Implement an approval workflow for stage transitions
4. Create a CI/CD pipeline for model promotion
5. Implement automated monitoring with rollback triggers
6. Compare model versions and select the best performer
7. Promote to Production and validate deployment

**Deliverables**:
- Training script with MLflow tracking
- Model registration code
- Approval workflow implementation
- GitHub Actions workflow files
- Monitoring script with auto-rollback
- Documentation of the entire process

## Summary

The model registry is the central hub for ML model lifecycle management:

- **Version Management**: Track all model versions with semantic versioning
- **Stage Management**: Control model progression through None → Staging → Production → Archived
- **Metadata**: Store comprehensive information about training, performance, and deployment
- **CI/CD Integration**: Automate training, validation, and promotion workflows
- **Rollback Capability**: Quickly restore previous versions when issues arise
- **Governance**: Implement approval workflows and audit trails

A well-implemented model registry enables safe, reproducible, and collaborative ML operations at scale.
