"""
Exercise Solutions: Kubeflow Pipelines
===========================================
Lesson 06 from MLOps topic.

Exercises
---------
1. Basic Pipeline — Create a 3-step ML pipeline (preprocess, train, evaluate)
   with defined inputs/outputs and artifact passing.
2. Hyperparameter Search with ParallelFor — Implement parallel hyperparameter
   search over a grid of configurations.
3. Scheduling — Design a scheduled pipeline with cron-like triggers and
   conditional execution based on data freshness.
"""

import random
import math
import json
import time
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor


# ============================================================
# Simulated Kubeflow Pipeline SDK (pure Python)
# ============================================================

class Component:
    """Simulates a Kubeflow Pipeline component."""

    def __init__(self, name, func, base_image="python:3.11", packages=None):
        self.name = name
        self.func = func
        self.base_image = base_image
        self.packages = packages or []

    def __call__(self, **kwargs):
        """Execute the component."""
        return self.func(**kwargs)


class Artifact:
    """Simulates a Kubeflow Pipeline artifact (dataset, model, metrics)."""

    def __init__(self, name, artifact_type="dataset"):
        self.name = name
        self.type = artifact_type
        self.data = None
        self.metadata = {}

    def __repr__(self):
        return f"Artifact(name={self.name}, type={self.type})"


class Pipeline:
    """Simulates a Kubeflow Pipeline."""

    def __init__(self, name, description=""):
        self.name = name
        self.description = description
        self.steps = []
        self.status = "pending"
        self.start_time = None
        self.end_time = None

    def add_step(self, name, component, inputs=None, outputs=None, depends_on=None):
        step = {
            "name": name,
            "component": component,
            "inputs": inputs or {},
            "outputs": outputs or {},
            "depends_on": depends_on or [],
            "status": "pending",
            "result": None,
        }
        self.steps.append(step)
        return step

    def run(self):
        """Execute pipeline respecting dependencies."""
        self.start_time = datetime.now()
        self.status = "running"
        completed = set()

        print(f"Pipeline: {self.name}")
        print(f"Steps: {len(self.steps)}")
        print("-" * 50)

        while len(completed) < len(self.steps):
            for step in self.steps:
                if step["name"] in completed:
                    continue
                # Check dependencies
                deps_met = all(d in completed for d in step["depends_on"])
                if not deps_met:
                    continue

                print(f"\n  Running: {step['name']}")
                step["status"] = "running"
                try:
                    result = step["component"](**step["inputs"])
                    step["result"] = result
                    step["status"] = "completed"
                    completed.add(step["name"])
                    print(f"  Status: completed")
                except Exception as e:
                    step["status"] = "failed"
                    print(f"  Status: FAILED ({e})")
                    self.status = "failed"
                    return

        self.status = "completed"
        self.end_time = datetime.now()
        duration = (self.end_time - self.start_time).total_seconds()
        print(f"\nPipeline completed in {duration:.2f}s")


# ============================================================
# Exercise 1: Basic Pipeline
# ============================================================

def exercise_1_basic_pipeline():
    """Create a 3-step ML pipeline: preprocess -> train -> evaluate.

    Demonstrates:
    - Defining pipeline components with typed inputs/outputs
    - Passing artifacts between steps
    - Pipeline execution with dependency ordering
    """

    # --- Component definitions ---

    def preprocess(n_samples=500, test_ratio=0.2, seed=42):
        """Preprocess data: generate, clean, split."""
        random.seed(seed)
        print(f"    Generating {n_samples} samples...")

        data = []
        for _ in range(n_samples):
            features = [random.gauss(0, 1) for _ in range(5)]
            # Target: simple linear combination + noise
            target = sum(f * w for f, w in zip(features, [0.5, -0.3, 0.8, 0.1, -0.6]))
            target = 1 if target + random.gauss(0, 0.3) > 0 else 0
            data.append({"features": features, "label": target})

        # Split
        random.shuffle(data)
        split = int(len(data) * (1 - test_ratio))
        train_data = data[:split]
        test_data = data[split:]

        # Compute statistics
        pos_train = sum(1 for d in train_data if d["label"] == 1)
        pos_test = sum(1 for d in test_data if d["label"] == 1)

        print(f"    Train: {len(train_data)} samples ({pos_train} positive)")
        print(f"    Test:  {len(test_data)} samples ({pos_test} positive)")

        return {
            "train_data": train_data,
            "test_data": test_data,
            "stats": {
                "n_train": len(train_data),
                "n_test": len(test_data),
                "n_features": 5,
                "pos_ratio_train": pos_train / len(train_data),
            },
        }

    def train(train_data, lr=0.01, epochs=50, regularization=0.001):
        """Train a logistic regression model."""
        n_features = len(train_data[0]["features"])
        weights = [0.0] * n_features
        bias = 0.0

        print(f"    Training logistic regression (lr={lr}, epochs={epochs})...")
        for epoch in range(epochs):
            total_loss = 0
            for sample in train_data:
                x = sample["features"]
                y = sample["label"]
                z = sum(w * xi for w, xi in zip(weights, x)) + bias
                z = max(-500, min(500, z))
                pred = 1 / (1 + math.exp(-z))
                error = pred - y
                for j in range(n_features):
                    weights[j] -= lr * (error * x[j] + regularization * weights[j])
                bias -= lr * error
                eps = 1e-7
                total_loss += -(y * math.log(pred + eps) + (1 - y) * math.log(1 - pred + eps))

        final_loss = total_loss / len(train_data)
        print(f"    Final training loss: {final_loss:.4f}")

        return {
            "model": {"weights": weights, "bias": bias},
            "training_metrics": {"final_loss": round(final_loss, 4), "epochs": epochs},
        }

    def evaluate(model, test_data, threshold=0.5):
        """Evaluate model on test set."""
        weights = model["weights"]
        bias = model["bias"]

        tp = fp = tn = fn = 0
        for sample in test_data:
            x = sample["features"]
            y = sample["label"]
            z = sum(w * xi for w, xi in zip(weights, x)) + bias
            z = max(-500, min(500, z))
            pred_prob = 1 / (1 + math.exp(-z))
            pred = 1 if pred_prob >= threshold else 0

            if pred == 1 and y == 1: tp += 1
            elif pred == 1 and y == 0: fp += 1
            elif pred == 0 and y == 0: tn += 1
            else: fn += 1

        total = tp + fp + tn + fn
        accuracy = (tp + tn) / total if total > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        metrics = {
            "accuracy": round(accuracy, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        }
        print(f"    Evaluation: accuracy={accuracy:.4f}, f1={f1:.4f}")
        return metrics

    # --- Build and run pipeline ---
    pipeline = Pipeline("basic-ml-pipeline", "3-step ML training pipeline")

    # Step 1: Preprocess
    preprocess_comp = Component("preprocess", preprocess)
    pipeline.add_step("preprocess", preprocess_comp, inputs={"n_samples": 500})

    # Step 2: Train (depends on preprocess)
    def train_step(**kwargs):
        preprocess_result = pipeline.steps[0]["result"]
        return train(preprocess_result["train_data"], lr=0.01, epochs=50)

    train_comp = Component("train", train_step)
    pipeline.add_step("train", train_comp, depends_on=["preprocess"])

    # Step 3: Evaluate (depends on train and preprocess)
    def evaluate_step(**kwargs):
        model = pipeline.steps[1]["result"]["model"]
        test_data = pipeline.steps[0]["result"]["test_data"]
        return evaluate(model, test_data)

    eval_comp = Component("evaluate", evaluate_step)
    pipeline.add_step("evaluate", eval_comp, depends_on=["train"])

    pipeline.run()

    return pipeline


# ============================================================
# Exercise 2: Hyperparameter Search with ParallelFor
# ============================================================

def exercise_2_parallel_search():
    """Implement parallel hyperparameter search over a grid of configurations.

    Kubeflow's ParallelFor allows running the same component with different
    parameters in parallel. We simulate this with ThreadPoolExecutor.
    """

    def train_and_evaluate(config):
        """Train and evaluate a single configuration."""
        random.seed(config.get("seed", 42))
        n_samples = 300
        data = []
        for _ in range(n_samples):
            features = [random.gauss(0, 1) for _ in range(5)]
            target = sum(f * w for f, w in zip(features, [0.5, -0.3, 0.8, 0.1, -0.6]))
            target = 1 if target + random.gauss(0, 0.3) > 0 else 0
            data.append({"features": features, "label": target})

        split = int(n_samples * 0.8)
        train_data, test_data = data[:split], data[split:]

        # Train
        n_features = 5
        weights = [0.0] * n_features
        bias = 0.0
        lr = config["learning_rate"]
        reg = config["regularization"]

        for _ in range(config["epochs"]):
            for sample in train_data:
                x = sample["features"]
                y = sample["label"]
                z = sum(w * xi for w, xi in zip(weights, x)) + bias
                z = max(-500, min(500, z))
                pred = 1 / (1 + math.exp(-z))
                error = pred - y
                for j in range(n_features):
                    weights[j] -= lr * (error * x[j] + reg * weights[j])
                bias -= lr * error

        # Evaluate
        correct = 0
        tp = fp = fn = 0
        for sample in test_data:
            x = sample["features"]
            y = sample["label"]
            z = sum(w * xi for w, xi in zip(weights, x)) + bias
            z = max(-500, min(500, z))
            pred = 1 if (1 / (1 + math.exp(-z))) >= 0.5 else 0
            if pred == y: correct += 1
            if pred == 1 and y == 1: tp += 1
            elif pred == 1 and y == 0: fp += 1
            elif pred == 0 and y == 1: fn += 1

        accuracy = correct / len(test_data)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        return {
            "config": config,
            "accuracy": round(accuracy, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        }

    # --- Define hyperparameter grid ---
    configs = []
    for lr in [0.001, 0.01, 0.05]:
        for reg in [0.0001, 0.001, 0.01]:
            for epochs in [50, 100]:
                configs.append({
                    "learning_rate": lr,
                    "regularization": reg,
                    "epochs": epochs,
                    "seed": 42,
                })

    print(f"ParallelFor Hyperparameter Search")
    print(f"=" * 60)
    print(f"Total configurations: {len(configs)}")
    print(f"Simulating parallel execution with max_workers=4...")
    print()

    # --- Execute in parallel (simulates Kubeflow ParallelFor) ---
    start_time = time.time()
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(train_and_evaluate, configs))
    elapsed = time.time() - start_time

    # --- Sort and display results ---
    results.sort(key=lambda r: -r["f1"])

    print(f"{'LR':>8s} {'Reg':>8s} {'Epochs':>7s} "
          f"{'Accuracy':>9s} {'Precision':>10s} {'Recall':>7s} {'F1':>7s}")
    print("-" * 60)
    for r in results:
        c = r["config"]
        print(f"{c['learning_rate']:>8.4f} {c['regularization']:>8.4f} {c['epochs']:>7d} "
              f"{r['accuracy']:>9.4f} {r['precision']:>10.4f} {r['recall']:>7.4f} {r['f1']:>7.4f}")

    print(f"\nParallel execution time: {elapsed:.2f}s")
    print(f"\nBest configuration (by F1):")
    best = results[0]
    print(f"  Learning rate: {best['config']['learning_rate']}")
    print(f"  Regularization: {best['config']['regularization']}")
    print(f"  Epochs: {best['config']['epochs']}")
    print(f"  F1: {best['f1']:.4f}")

    return results


# ============================================================
# Exercise 3: Scheduling
# ============================================================

def exercise_3_scheduling():
    """Design a scheduled pipeline with cron-like triggers and conditional execution.

    Demonstrates:
    - Cron schedule parsing and simulation
    - Data freshness checks before execution
    - Conditional pipeline stages
    - Execution history logging
    """

    class PipelineScheduler:
        """Simulates a Kubeflow recurring run scheduler."""

        def __init__(self, pipeline_name):
            self.pipeline_name = pipeline_name
            self.schedules = []
            self.execution_history = []
            self.conditions = {}

        def add_schedule(self, name, cron_expression, description=""):
            """Add a cron-like schedule.

            Simplified cron: (minute, hour, day_of_month, month, day_of_week)
            '*' means every, specific numbers match exactly.
            """
            self.schedules.append({
                "name": name,
                "cron": cron_expression,
                "description": description,
                "enabled": True,
            })

        def add_condition(self, name, check_fn, description=""):
            """Add a pre-execution condition."""
            self.conditions[name] = {
                "check": check_fn,
                "description": description,
            }

        def _match_cron(self, cron_str, current_time):
            """Check if current time matches a cron expression (simplified)."""
            parts = cron_str.split()
            minute, hour, dom, month, dow = parts

            checks = [
                (minute, current_time.minute),
                (hour, current_time.hour),
                (dom, current_time.day),
                (month, current_time.month),
                (dow, current_time.weekday()),
            ]

            for pattern, value in checks:
                if pattern == "*":
                    continue
                if "/" in pattern:
                    _, step = pattern.split("/")
                    if value % int(step) != 0:
                        return False
                elif "," in pattern:
                    if str(value) not in pattern.split(","):
                        return False
                elif int(pattern) != value:
                    return False
            return True

        def check_conditions(self, context):
            """Evaluate all pre-execution conditions."""
            results = {}
            for name, condition in self.conditions.items():
                passed = condition["check"](context)
                results[name] = {
                    "passed": passed,
                    "description": condition["description"],
                }
            return results

        def simulate_day(self, date, context):
            """Simulate scheduled checks for a full day (checking every hour)."""
            triggers = []
            for hour in range(24):
                check_time = date.replace(hour=hour, minute=0, second=0)
                for schedule in self.schedules:
                    if not schedule["enabled"]:
                        continue
                    if self._match_cron(schedule["cron"], check_time):
                        triggers.append({
                            "schedule": schedule["name"],
                            "time": check_time,
                        })
            return triggers

    # --- Build a scheduled retraining pipeline ---
    scheduler = PipelineScheduler("fraud-detection-retrain")

    # Schedule 1: Daily retraining at 2 AM
    scheduler.add_schedule(
        "daily-retrain",
        "0 2 * * *",
        "Run retraining pipeline daily at 2:00 AM",
    )

    # Schedule 2: Weekly full evaluation on Sundays
    scheduler.add_schedule(
        "weekly-evaluation",
        "0 6 * * 0",
        "Run full model evaluation every Sunday at 6:00 AM",
    )

    # Schedule 3: Monthly data quality audit on the 1st
    scheduler.add_schedule(
        "monthly-audit",
        "0 0 1 * *",
        "Run data quality audit on the 1st of each month",
    )

    # Pre-execution conditions
    random.seed(42)

    scheduler.add_condition(
        "data_freshness",
        lambda ctx: ctx.get("hours_since_last_data", 0) < 24,
        "Data must be less than 24 hours old",
    )

    scheduler.add_condition(
        "minimum_samples",
        lambda ctx: ctx.get("new_samples_count", 0) >= 1000,
        "At least 1,000 new samples since last training",
    )

    scheduler.add_condition(
        "no_active_incident",
        lambda ctx: not ctx.get("active_incident", False),
        "No active system incident",
    )

    # --- Simulate 7 days ---
    print("Pipeline Scheduling Simulation")
    print("=" * 60)

    print("\nSchedules:")
    for s in scheduler.schedules:
        print(f"  {s['name']}: {s['cron']} — {s['description']}")

    print("\nPre-execution Conditions:")
    for name, cond in scheduler.conditions.items():
        print(f"  {name}: {cond['description']}")

    print("\n7-Day Simulation:")
    print("-" * 60)

    base_date = datetime(2025, 3, 1)  # Saturday
    day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

    for day_offset in range(7):
        current_date = base_date + timedelta(days=day_offset)
        day_name = day_names[current_date.weekday()]

        # Simulate context that changes each day
        context = {
            "hours_since_last_data": random.randint(1, 30),
            "new_samples_count": random.randint(500, 3000),
            "active_incident": random.random() < 0.1,  # 10% chance
        }

        triggers = scheduler.simulate_day(current_date, context)

        if triggers:
            print(f"\n  {current_date.strftime('%Y-%m-%d')} ({day_name}):")
            for trigger in triggers:
                print(f"    Triggered: {trigger['schedule']} "
                      f"at {trigger['time'].strftime('%H:%M')}")

                # Check conditions
                conditions = scheduler.check_conditions(context)
                all_passed = all(c["passed"] for c in conditions.values())

                for cname, cresult in conditions.items():
                    status = "PASS" if cresult["passed"] else "FAIL"
                    print(f"      [{status}] {cname}")

                if all_passed:
                    print(f"      => Pipeline EXECUTED")
                    scheduler.execution_history.append({
                        "date": current_date.isoformat(),
                        "schedule": trigger["schedule"],
                        "status": "executed",
                    })
                else:
                    print(f"      => Pipeline SKIPPED (conditions not met)")
                    scheduler.execution_history.append({
                        "date": current_date.isoformat(),
                        "schedule": trigger["schedule"],
                        "status": "skipped",
                    })
        else:
            print(f"\n  {current_date.strftime('%Y-%m-%d')} ({day_name}): No triggers")

    # Summary
    executed = sum(1 for e in scheduler.execution_history if e["status"] == "executed")
    skipped = sum(1 for e in scheduler.execution_history if e["status"] == "skipped")
    print(f"\nExecution Summary:")
    print(f"  Total triggers: {len(scheduler.execution_history)}")
    print(f"  Executed: {executed}")
    print(f"  Skipped: {skipped}")

    return scheduler


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    print("Exercise 1: Basic Pipeline")
    print("=" * 60)
    exercise_1_basic_pipeline()

    print("\n\n")
    print("Exercise 2: Hyperparameter Search with ParallelFor")
    print("=" * 60)
    exercise_2_parallel_search()

    print("\n\n")
    print("Exercise 3: Scheduling")
    print("=" * 60)
    exercise_3_scheduling()
