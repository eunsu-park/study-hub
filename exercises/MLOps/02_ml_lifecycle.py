"""
Exercise Solutions: ML Lifecycle
===========================================
Lesson 02 from MLOps topic.

Exercises
---------
1. Pipeline Design — Design an end-to-end ML pipeline for a given use case,
   identifying all stages and their inputs/outputs.
2. Retraining Policy — Define a retraining policy for a model in production,
   specifying triggers, frequency, and validation criteria.
"""

import random
import time
from datetime import datetime, timedelta


# ============================================================
# Exercise 1: Pipeline Design
# ============================================================

def exercise_1_pipeline_design():
    """Design an end-to-end ML pipeline for a fraud detection use case.

    We model the pipeline as a DAG (directed acyclic graph) of stages,
    each with defined inputs, outputs, and dependencies. This simulates
    how tools like Kubeflow or Airflow represent ML pipelines.
    """

    class PipelineStage:
        """Represents a single stage in an ML pipeline."""

        def __init__(self, name, description, inputs, outputs, dependencies=None):
            self.name = name
            self.description = description
            self.inputs = inputs
            self.outputs = outputs
            self.dependencies = dependencies or []
            self.status = "pending"
            self.duration_seconds = 0

        def execute(self):
            """Simulate stage execution."""
            self.status = "running"
            # Simulate work with a small delay
            duration = random.uniform(0.01, 0.05)
            time.sleep(duration)
            self.duration_seconds = duration
            self.status = "completed"

    class MLPipeline:
        """End-to-end ML pipeline with dependency management."""

        def __init__(self, name, use_case):
            self.name = name
            self.use_case = use_case
            self.stages = []

        def add_stage(self, stage):
            self.stages.append(stage)

        def validate_dag(self):
            """Validate that the pipeline forms a valid DAG (no cycles)."""
            visited = set()
            in_progress = set()
            stage_map = {s.name: s for s in self.stages}

            def dfs(name):
                if name in in_progress:
                    return False  # Cycle detected
                if name in visited:
                    return True
                in_progress.add(name)
                for dep in stage_map[name].dependencies:
                    if not dfs(dep):
                        return False
                in_progress.remove(name)
                visited.add(name)
                return True

            return all(dfs(s.name) for s in self.stages)

        def topological_sort(self):
            """Return stages in execution order (topological sort)."""
            stage_map = {s.name: s for s in self.stages}
            visited = set()
            order = []

            def dfs(name):
                if name in visited:
                    return
                visited.add(name)
                for dep in stage_map[name].dependencies:
                    dfs(dep)
                order.append(stage_map[name])

            for s in self.stages:
                dfs(s.name)
            return order

        def run(self):
            """Execute the pipeline in topological order."""
            if not self.validate_dag():
                raise ValueError("Pipeline contains cycles!")

            execution_order = self.topological_sort()
            print(f"Pipeline: {self.name}")
            print(f"Use Case: {self.use_case}")
            print(f"Stages: {len(execution_order)}")
            print("-" * 60)

            total_time = 0
            for i, stage in enumerate(execution_order, 1):
                deps_str = ", ".join(stage.dependencies) if stage.dependencies else "none"
                print(f"\n  Stage {i}: {stage.name}")
                print(f"    Description: {stage.description}")
                print(f"    Inputs:      {', '.join(stage.inputs)}")
                print(f"    Outputs:     {', '.join(stage.outputs)}")
                print(f"    Depends on:  {deps_str}")

                stage.execute()
                total_time += stage.duration_seconds
                print(f"    Status:      {stage.status} ({stage.duration_seconds:.3f}s)")

            print(f"\nTotal pipeline execution: {total_time:.3f}s")
            return execution_order

    # --- Build a Fraud Detection pipeline ---
    pipeline = MLPipeline(
        name="Fraud Detection Pipeline",
        use_case="Real-time credit card fraud detection with daily retraining",
    )

    pipeline.add_stage(PipelineStage(
        name="data_ingestion",
        description="Ingest transaction data from payment gateway and data warehouse",
        inputs=["raw_transactions_csv", "customer_profiles_db"],
        outputs=["raw_dataset"],
    ))

    pipeline.add_stage(PipelineStage(
        name="data_validation",
        description="Validate schema, check for nulls, verify value ranges",
        inputs=["raw_dataset"],
        outputs=["validation_report", "validated_dataset"],
        dependencies=["data_ingestion"],
    ))

    pipeline.add_stage(PipelineStage(
        name="feature_engineering",
        description="Compute transaction velocity, amount z-scores, time features",
        inputs=["validated_dataset"],
        outputs=["feature_matrix"],
        dependencies=["data_validation"],
    ))

    pipeline.add_stage(PipelineStage(
        name="data_splitting",
        description="Stratified split (70/15/15) preserving fraud ratio",
        inputs=["feature_matrix"],
        outputs=["train_set", "validation_set", "test_set"],
        dependencies=["feature_engineering"],
    ))

    pipeline.add_stage(PipelineStage(
        name="model_training",
        description="Train XGBoost with class weights for imbalanced data",
        inputs=["train_set", "validation_set"],
        outputs=["trained_model", "training_metrics"],
        dependencies=["data_splitting"],
    ))

    pipeline.add_stage(PipelineStage(
        name="model_evaluation",
        description="Evaluate on test set: precision, recall, F1, AUC-PR",
        inputs=["trained_model", "test_set"],
        outputs=["evaluation_report", "confusion_matrix"],
        dependencies=["model_training"],
    ))

    pipeline.add_stage(PipelineStage(
        name="model_validation",
        description="Check quality gates: recall > 0.95, false positive rate < 0.05",
        inputs=["evaluation_report"],
        outputs=["validation_decision"],
        dependencies=["model_evaluation"],
    ))

    pipeline.add_stage(PipelineStage(
        name="model_registration",
        description="Register model in registry with metadata and lineage",
        inputs=["trained_model", "validation_decision"],
        outputs=["registered_model_version"],
        dependencies=["model_validation"],
    ))

    pipeline.add_stage(PipelineStage(
        name="deployment",
        description="Deploy to staging, run shadow mode, then promote to production",
        inputs=["registered_model_version"],
        outputs=["deployment_endpoint"],
        dependencies=["model_registration"],
    ))

    pipeline.add_stage(PipelineStage(
        name="monitoring_setup",
        description="Configure drift detection, latency alerts, and feedback collection",
        inputs=["deployment_endpoint"],
        outputs=["monitoring_dashboard"],
        dependencies=["deployment"],
    ))

    pipeline.run()
    return pipeline


# ============================================================
# Exercise 2: Retraining Policy
# ============================================================

def exercise_2_retraining_policy():
    """Define and simulate a retraining policy for a production model.

    A retraining policy specifies:
    - Triggers: What conditions initiate retraining?
    - Frequency: How often do we check / retrain?
    - Validation: What criteria must the new model meet?
    - Rollback: What happens if the new model is worse?

    We simulate a monitoring loop that checks triggers and decides
    whether to retrain.
    """

    class RetrainingPolicy:
        """Defines when and how to retrain a production model."""

        def __init__(self, model_name):
            self.model_name = model_name
            self.triggers = {}
            self.validation_criteria = {}
            self.max_staleness_days = 30
            self.min_data_points_since_last_train = 10000
            self.retraining_log = []

        def add_trigger(self, name, check_fn, description):
            """Add a retraining trigger with a check function."""
            self.triggers[name] = {
                "check": check_fn,
                "description": description,
            }

        def add_validation_criterion(self, name, check_fn, description):
            """Add a validation criterion the new model must pass."""
            self.validation_criteria[name] = {
                "check": check_fn,
                "description": description,
            }

        def evaluate_triggers(self, context):
            """Check all triggers and return which ones fired."""
            fired = []
            for name, trigger in self.triggers.items():
                if trigger["check"](context):
                    fired.append((name, trigger["description"]))
            return fired

        def validate_new_model(self, new_metrics, old_metrics):
            """Validate whether the new model meets all criteria."""
            context = {"new": new_metrics, "old": old_metrics}
            results = {}
            for name, criterion in self.validation_criteria.items():
                passed = criterion["check"](context)
                results[name] = {
                    "passed": passed,
                    "description": criterion["description"],
                }
            return results

    # --- Build a retraining policy for fraud detection ---
    policy = RetrainingPolicy("fraud_detection_xgboost_v2")

    # Trigger 1: Performance degradation
    policy.add_trigger(
        "performance_drop",
        lambda ctx: ctx["current_recall"] < 0.93,
        "Recall dropped below 0.93 threshold (production minimum: 0.95)",
    )

    # Trigger 2: Data drift detected
    policy.add_trigger(
        "data_drift",
        lambda ctx: ctx["drift_score"] > 0.15,
        "Feature drift score exceeded 0.15 (KS test on top 5 features)",
    )

    # Trigger 3: Model staleness
    policy.add_trigger(
        "model_staleness",
        lambda ctx: ctx["days_since_training"] > 30,
        "Model has not been retrained in over 30 days",
    )

    # Trigger 4: Sufficient new data
    policy.add_trigger(
        "new_data_volume",
        lambda ctx: ctx["new_data_points"] > 50000,
        "Over 50,000 new labeled data points available since last training",
    )

    # Trigger 5: Concept drift (label distribution shift)
    policy.add_trigger(
        "concept_drift",
        lambda ctx: abs(ctx["current_fraud_rate"] - ctx["training_fraud_rate"]) > 0.02,
        "Fraud rate shifted by more than 2% from training distribution",
    )

    # Validation criteria for new model
    policy.add_validation_criterion(
        "recall_threshold",
        lambda ctx: ctx["new"]["recall"] >= 0.95,
        "New model recall must be >= 0.95",
    )

    policy.add_validation_criterion(
        "precision_floor",
        lambda ctx: ctx["new"]["precision"] >= 0.50,
        "New model precision must be >= 0.50 (avoid excessive false positives)",
    )

    policy.add_validation_criterion(
        "improvement_check",
        lambda ctx: ctx["new"]["f1"] >= ctx["old"]["f1"] * 0.98,
        "New model F1 must be within 2% of old model F1 (no regression)",
    )

    policy.add_validation_criterion(
        "latency_check",
        lambda ctx: ctx["new"]["p99_latency_ms"] <= 50,
        "New model p99 latency must be <= 50ms for real-time serving",
    )

    # --- Simulate monitoring over 90 days ---
    print("Retraining Policy Simulation")
    print("=" * 60)
    print(f"Model: {policy.model_name}")
    print(f"Monitoring Period: 90 days")
    print()

    print("Triggers:")
    for name, trigger in policy.triggers.items():
        print(f"  - {name}: {trigger['description']}")
    print()

    print("Validation Criteria:")
    for name, criterion in policy.validation_criteria.items():
        print(f"  - {name}: {criterion['description']}")
    print()

    random.seed(42)
    base_date = datetime(2025, 1, 1)
    last_training_date = base_date
    retrain_count = 0

    for day in range(1, 91):
        current_date = base_date + timedelta(days=day)

        # Simulate gradually degrading metrics
        days_since = (current_date - last_training_date).days
        degradation = min(days_since * 0.002, 0.1)  # Slow degradation
        noise = random.gauss(0, 0.01)

        context = {
            "current_recall": max(0.80, 0.97 - degradation + noise),
            "drift_score": min(0.5, 0.05 + days_since * 0.004 + random.gauss(0, 0.02)),
            "days_since_training": days_since,
            "new_data_points": days_since * 2000,
            "current_fraud_rate": 0.015 + random.gauss(0, 0.005),
            "training_fraud_rate": 0.012,
        }

        fired_triggers = policy.evaluate_triggers(context)

        if fired_triggers:
            # Check if any critical trigger fired
            critical = any(t[0] in ["performance_drop", "data_drift", "concept_drift"]
                          for t in fired_triggers)

            if critical or len(fired_triggers) >= 2:
                retrain_count += 1
                print(f"Day {day:3d} ({current_date.strftime('%Y-%m-%d')}) — "
                      f"RETRAINING TRIGGERED")
                for trigger_name, trigger_desc in fired_triggers:
                    print(f"    Trigger: {trigger_name}")

                # Simulate retraining result
                new_metrics = {
                    "recall": 0.96 + random.gauss(0, 0.01),
                    "precision": 0.65 + random.gauss(0, 0.05),
                    "f1": 0.78 + random.gauss(0, 0.02),
                    "p99_latency_ms": 30 + random.gauss(0, 5),
                }
                old_metrics = {
                    "recall": context["current_recall"],
                    "precision": 0.60,
                    "f1": 0.73,
                    "p99_latency_ms": 35,
                }

                validation_results = policy.validate_new_model(new_metrics, old_metrics)
                all_passed = all(r["passed"] for r in validation_results.values())

                print(f"    Validation: {'PASSED' if all_passed else 'FAILED'}")
                for vname, vresult in validation_results.items():
                    status = "PASS" if vresult["passed"] else "FAIL"
                    print(f"      [{status}] {vname}: {vresult['description']}")

                if all_passed:
                    last_training_date = current_date
                    print(f"    Action: New model deployed to production")
                else:
                    print(f"    Action: Kept old model; new model rejected")

                policy.retraining_log.append({
                    "date": current_date,
                    "triggers": [t[0] for t in fired_triggers],
                    "deployed": all_passed,
                })
                print()

    # Summary
    print("-" * 60)
    print("Retraining Summary:")
    print(f"  Total retraining attempts: {retrain_count}")
    print(f"  Successful deployments: "
          f"{sum(1 for r in policy.retraining_log if r['deployed'])}")
    print(f"  Rejected models: "
          f"{sum(1 for r in policy.retraining_log if not r['deployed'])}")

    return policy


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    print("Exercise 1: Pipeline Design")
    print("=" * 60)
    exercise_1_pipeline_design()

    print("\n\n")
    print("Exercise 2: Retraining Policy")
    print("=" * 60)
    exercise_2_retraining_policy()
