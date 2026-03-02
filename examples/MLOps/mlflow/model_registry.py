"""
MLflow Model Registry Example
=============================

Example of model version management using MLflow Model Registry.

How to run:
    # First run tracking_example.py to train/save models
    python model_registry.py
"""

import mlflow
from mlflow.tracking import MlflowClient
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import os

# MLflow configuration
TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
MODEL_NAME = "iris-classifier"


def setup():
    """MLflow setup"""
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment("model-registry-demo")
    return MlflowClient()


def train_and_register_model(client, version_tag: str):
    """Train and register a model"""
    # Prepare data
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.2, random_state=42
    )

    with mlflow.start_run(run_name=f"training-{version_tag}") as run:
        # Train model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Evaluate
        accuracy = accuracy_score(y_test, model.predict(X_test))
        mlflow.log_metric("accuracy", accuracy)

        # Save and register model
        mlflow.sklearn.log_model(
            model,
            "model",
            registered_model_name=MODEL_NAME
        )

        print(f"\nModel registered: {MODEL_NAME}")
        print(f"  Run ID: {run.info.run_id}")
        print(f"  Accuracy: {accuracy:.4f}")

        return run.info.run_id


def get_model_versions(client):
    """Retrieve registered model versions"""
    print(f"\n{'='*50}")
    print(f"Version list for model '{MODEL_NAME}':")
    print("="*50)

    try:
        versions = client.search_model_versions(f"name='{MODEL_NAME}'")
        for v in versions:
            print(f"\nVersion {v.version}:")
            print(f"  Stage: {v.current_stage}")
            print(f"  Run ID: {v.run_id}")
            print(f"  Created: {v.creation_timestamp}")
            if v.description:
                print(f"  Description: {v.description}")
        return versions
    except Exception as e:
        print(f"Model not found: {e}")
        return []


def transition_to_staging(client, version: str):
    """Transition model to Staging"""
    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=version,
        stage="Staging",
        archive_existing_versions=False
    )
    print(f"\nTransitioned model v{version} to Staging.")


def transition_to_production(client, version: str):
    """Transition model to Production"""
    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=version,
        stage="Production",
        archive_existing_versions=True
    )
    print(f"\nTransitioned model v{version} to Production.")


def update_model_description(client, version: str, description: str):
    """Update model description"""
    client.update_model_version(
        name=MODEL_NAME,
        version=version,
        description=description
    )
    print(f"\nUpdated description for model v{version}.")


def add_model_tag(client, version: str, key: str, value: str):
    """Add model tag"""
    client.set_model_version_tag(
        name=MODEL_NAME,
        version=version,
        key=key,
        value=value
    )
    print(f"\nAdded tag to model v{version}: {key}={value}")


def load_model_by_stage(stage: str):
    """Load model by stage"""
    try:
        model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/{stage}")
        print(f"\nSuccessfully loaded {stage} model!")
        return model
    except Exception as e:
        print(f"\nFailed to load {stage} model: {e}")
        return None


def demo_workflow(client):
    """Full workflow demo"""
    print("\n" + "="*60)
    print("MLflow Model Registry Workflow Demo")
    print("="*60)

    # 1. Register first model
    print("\n[1] Training and registering the first model...")
    train_and_register_model(client, "v1")

    # 2. Query versions
    versions = get_model_versions(client)
    if not versions:
        return

    latest_version = max(v.version for v in versions)

    # 3. Add description
    print("\n[2] Adding model description...")
    update_model_description(
        client, latest_version,
        "Initial model trained on Iris dataset with Random Forest"
    )

    # 4. Add tags
    print("\n[3] Adding model tags...")
    add_model_tag(client, latest_version, "validated", "true")
    add_model_tag(client, latest_version, "dataset", "iris")

    # 5. Transition to Staging
    print("\n[4] Transitioning to Staging...")
    transition_to_staging(client, latest_version)

    # 6. Register second model
    print("\n[5] Training and registering the second model (improved version)...")
    train_and_register_model(client, "v2")

    # 7. Re-query versions
    versions = get_model_versions(client)
    new_latest = max(v.version for v in versions)

    # 8. Move new version to Staging
    print("\n[6] Moving new version to Staging...")
    transition_to_staging(client, new_latest)

    # 9. Promote to Production
    print("\n[7] Promoting to Production...")
    transition_to_production(client, new_latest)

    # 10. Check final state
    print("\n[8] Final model state:")
    get_model_versions(client)

    # 11. Test loading Production model
    print("\n[9] Testing Production model load...")
    model = load_model_by_stage("Production")
    if model:
        # Simple prediction test
        iris = load_iris()
        sample = iris.data[:3]
        predictions = model.predict(sample)
        print(f"  Sample predictions: {predictions}")
        print(f"  Actual labels: {iris.target[:3]}")


def main():
    """Main function"""
    client = setup()

    print("\nMLflow Model Registry Example")
    print("="*50)
    print("\nOptions:")
    print("1. Train and register a new model")
    print("2. Query registered models")
    print("3. Full workflow demo")

    choice = input("\nSelect (1/2/3): ").strip()

    if choice == "1":
        train_and_register_model(client, "manual")
    elif choice == "2":
        get_model_versions(client)
    elif choice == "3":
        demo_workflow(client)
    else:
        print("Invalid selection.")


if __name__ == "__main__":
    main()
