"""
Exercise Solutions: Weights & Biases
===========================================
Lesson 05 from MLOps topic.

Exercises
---------
1. Basic Experiment Tracking — Track a CNN training experiment on MNIST
   (simulated) with per-epoch metrics, hyperparameters, and system metrics.
2. Run Sweeps — Implement a hyperparameter sweep with Bayesian-like
   optimization (simulated without real W&B).
3. Artifacts — Log, version, and retrieve model artifacts and datasets
   through a simulated artifact store.
"""

import math
import random
import hashlib
import json
from datetime import datetime


# ============================================================
# Simulated W&B API (pure Python)
# ============================================================

class SimulatedWandb:
    """Minimal simulation of the Weights & Biases API."""

    def __init__(self, project="default"):
        self.project = project
        self.runs = []
        self.artifacts = {}
        self._current_run = None

    def init(self, project=None, config=None, name=None, tags=None):
        """Initialize a new run."""
        run = {
            "id": f"run_{len(self.runs) + 1:04d}",
            "name": name or f"run-{random.randint(1000,9999)}",
            "project": project or self.project,
            "config": config or {},
            "tags": tags or [],
            "history": [],
            "summary": {},
            "system_metrics": [],
            "start_time": datetime.now().isoformat(),
            "status": "running",
        }
        self._current_run = run
        self.runs.append(run)
        return run

    def log(self, data, step=None):
        """Log metrics for the current step."""
        entry = {"_step": step or len(self._current_run["history"]), **data}
        self._current_run["history"].append(entry)
        # Update summary with latest values
        self._current_run["summary"].update(data)

    def log_artifact(self, name, artifact_type, metadata=None):
        """Log an artifact (dataset, model, etc.)."""
        version = len(self.artifacts.get(name, [])) + 1
        artifact = {
            "name": name,
            "type": artifact_type,
            "version": f"v{version}",
            "metadata": metadata or {},
            "digest": hashlib.md5(f"{name}-v{version}".encode()).hexdigest()[:8],
            "created_at": datetime.now().isoformat(),
        }
        if name not in self.artifacts:
            self.artifacts[name] = []
        self.artifacts[name].append(artifact)
        return artifact

    def finish(self):
        """End the current run."""
        self._current_run["status"] = "finished"
        self._current_run["end_time"] = datetime.now().isoformat()
        run = self._current_run
        self._current_run = None
        return run


# ============================================================
# Simulated CNN Training
# ============================================================

def simulate_cnn_training(config, epochs=10, seed=42):
    """Simulate CNN training on MNIST with realistic learning curves.

    Instead of actual training, we generate plausible metrics based on
    the hyperparameters. This captures the relationships:
    - Higher learning rate -> faster initial convergence but potential instability
    - More filters -> better accuracy but slower training
    - Dropout -> less overfitting
    - Batch size -> affects convergence smoothness
    """
    random.seed(seed)

    lr = config.get("learning_rate", 0.001)
    filters = config.get("filters", 32)
    dropout = config.get("dropout", 0.5)
    batch_size = config.get("batch_size", 64)

    # Base accuracy influenced by architecture
    base_accuracy = 0.85 + 0.05 * math.log2(filters / 16)
    # Learning rate effect
    lr_factor = 1.0 - abs(math.log10(lr) + 3) * 0.05  # Optimal around 0.001
    # Dropout effect (reduces overfitting gap)
    overfitting_gap = 0.05 * (1 - dropout)

    history = []
    for epoch in range(epochs):
        # Training metrics
        progress = 1 - math.exp(-0.5 * (epoch + 1))  # Diminishing returns curve
        train_acc = min(0.999, base_accuracy * lr_factor * progress + random.gauss(0, 0.005))
        train_loss = max(0.01, (1 - train_acc) * 2.5 + random.gauss(0, 0.02))

        # Validation metrics (slightly worse due to generalization gap)
        val_acc = min(0.999, train_acc - overfitting_gap * progress + random.gauss(0, 0.008))
        val_loss = max(0.01, (1 - val_acc) * 2.8 + random.gauss(0, 0.03))

        # Simulated system metrics
        gpu_utilization = 70 + filters / 4 + random.gauss(0, 3)
        gpu_memory_mb = 1024 + filters * 8 + batch_size * 2
        epoch_time_s = 10 + filters * 0.3 + batch_size * 0.05

        history.append({
            "epoch": epoch + 1,
            "train_accuracy": round(train_acc, 4),
            "train_loss": round(train_loss, 4),
            "val_accuracy": round(val_acc, 4),
            "val_loss": round(val_loss, 4),
            "gpu_utilization": round(gpu_utilization, 1),
            "gpu_memory_mb": round(gpu_memory_mb),
            "epoch_time_s": round(epoch_time_s, 1),
        })

    return history


# ============================================================
# Exercise 1: Basic Experiment Tracking
# ============================================================

def exercise_1_basic_tracking():
    """Track a CNN training experiment on MNIST with W&B-style logging.

    Demonstrates:
    - Configuring a run with hyperparameters
    - Logging per-epoch training and validation metrics
    - Logging system metrics (GPU utilization, memory)
    - Summarizing final results
    """
    wandb = SimulatedWandb(project="mnist-cnn")

    config = {
        "model": "SimpleCNN",
        "dataset": "MNIST",
        "epochs": 10,
        "batch_size": 64,
        "learning_rate": 0.001,
        "optimizer": "Adam",
        "filters": 32,
        "dropout": 0.5,
        "weight_decay": 1e-4,
        "scheduler": "cosine",
    }

    run = wandb.init(
        project="mnist-cnn",
        config=config,
        name="cnn-baseline-v1",
        tags=["baseline", "mnist", "cnn"],
    )

    print("MNIST CNN Experiment Tracking")
    print("=" * 60)
    print(f"Run: {run['name']} ({run['id']})")
    print(f"Config: {json.dumps(config, indent=2)}")
    print()

    # --- Simulate training ---
    history = simulate_cnn_training(config, epochs=config["epochs"])

    print(f"{'Epoch':>5s} {'Train Acc':>10s} {'Train Loss':>11s} "
          f"{'Val Acc':>10s} {'Val Loss':>10s} {'GPU %':>6s}")
    print("-" * 60)

    best_val_acc = 0
    for entry in history:
        wandb.log({
            "train/accuracy": entry["train_accuracy"],
            "train/loss": entry["train_loss"],
            "val/accuracy": entry["val_accuracy"],
            "val/loss": entry["val_loss"],
            "system/gpu_utilization": entry["gpu_utilization"],
            "system/gpu_memory_mb": entry["gpu_memory_mb"],
            "epoch_time_s": entry["epoch_time_s"],
        }, step=entry["epoch"])

        if entry["val_accuracy"] > best_val_acc:
            best_val_acc = entry["val_accuracy"]

        print(f"{entry['epoch']:>5d} {entry['train_accuracy']:>10.4f} "
              f"{entry['train_loss']:>11.4f} {entry['val_accuracy']:>10.4f} "
              f"{entry['val_loss']:>10.4f} {entry['gpu_utilization']:>5.1f}%")

    # Log summary metrics
    wandb.log({"best_val_accuracy": best_val_acc})

    run = wandb.finish()

    print(f"\nRun Summary:")
    print(f"  Best Validation Accuracy: {best_val_acc:.4f}")
    print(f"  Final Train Loss: {history[-1]['train_loss']:.4f}")
    print(f"  Status: {run['status']}")
    print(f"  Total logged steps: {len(run['history'])}")

    return run


# ============================================================
# Exercise 2: Run Sweeps
# ============================================================

def exercise_2_sweeps():
    """Implement a hyperparameter sweep with Bayesian-like optimization.

    W&B Sweeps automate hyperparameter search. This simulates:
    - Defining a sweep configuration (search space)
    - Running multiple agents with different configs
    - Bayesian optimization: using past results to guide future choices
    - Identifying the best configuration

    Our "Bayesian" approach: After initial random exploration, we sample
    near the best-performing configurations with some noise.
    """
    wandb = SimulatedWandb(project="mnist-sweep")

    # --- Sweep configuration ---
    sweep_config = {
        "method": "bayes",  # random, grid, or bayes
        "metric": {"name": "val/accuracy", "goal": "maximize"},
        "parameters": {
            "learning_rate": {"min": 0.0001, "max": 0.01, "distribution": "log_uniform"},
            "batch_size": {"values": [32, 64, 128, 256]},
            "filters": {"values": [16, 32, 64, 128]},
            "dropout": {"min": 0.1, "max": 0.7, "distribution": "uniform"},
        },
        "early_terminate": {
            "type": "hyperband",
            "min_iter": 3,
            "eta": 3,
        },
    }

    print("Hyperparameter Sweep")
    print("=" * 60)
    print(f"Method: {sweep_config['method']}")
    print(f"Metric: {sweep_config['metric']['name']} ({sweep_config['metric']['goal']})")
    print(f"Parameters: {list(sweep_config['parameters'].keys())}")
    print()

    n_runs = 20
    random.seed(42)
    best_configs = []  # Track good configs for Bayesian-like sampling

    def sample_config(run_idx):
        """Sample a configuration — random for first 5, then Bayesian-like."""
        params = sweep_config["parameters"]

        if run_idx < 5 or not best_configs:
            # Random exploration phase
            lr = math.exp(random.uniform(
                math.log(params["learning_rate"]["min"]),
                math.log(params["learning_rate"]["max"]),
            ))
            batch_size = random.choice(params["batch_size"]["values"])
            filters = random.choice(params["filters"]["values"])
            dropout = random.uniform(params["dropout"]["min"], params["dropout"]["max"])
        else:
            # Bayesian-like: perturb the best config
            best = random.choice(best_configs[:3])  # Sample from top 3
            lr = best["learning_rate"] * math.exp(random.gauss(0, 0.3))
            lr = max(params["learning_rate"]["min"],
                     min(params["learning_rate"]["max"], lr))
            batch_size = random.choice(params["batch_size"]["values"])
            filters = random.choice([
                best["filters"],
                random.choice(params["filters"]["values"]),
            ])
            dropout = best["dropout"] + random.gauss(0, 0.1)
            dropout = max(params["dropout"]["min"],
                          min(params["dropout"]["max"], dropout))

        return {
            "learning_rate": round(lr, 6),
            "batch_size": batch_size,
            "filters": filters,
            "dropout": round(dropout, 3),
            "epochs": 10,
        }

    # --- Run sweep ---
    all_results = []

    print(f"{'Run':>4s} {'LR':>10s} {'Batch':>6s} {'Filters':>8s} "
          f"{'Dropout':>8s} {'Val Acc':>8s} {'Phase':>8s}")
    print("-" * 60)

    for i in range(n_runs):
        config = sample_config(i)
        phase = "explore" if i < 5 else "exploit"

        run = wandb.init(
            config=config,
            name=f"sweep-{i+1:03d}",
            tags=["sweep", phase],
        )

        history = simulate_cnn_training(config, epochs=config["epochs"], seed=42 + i)
        best_val_acc = max(h["val_accuracy"] for h in history)

        for entry in history:
            wandb.log({
                "val/accuracy": entry["val_accuracy"],
                "val/loss": entry["val_loss"],
            })

        wandb.finish()

        result = {**config, "val_accuracy": best_val_acc}
        all_results.append(result)

        # Update best configs for Bayesian sampling
        best_configs = sorted(all_results, key=lambda x: -x["val_accuracy"])

        print(f"{i+1:>4d} {config['learning_rate']:>10.6f} {config['batch_size']:>6d} "
              f"{config['filters']:>8d} {config['dropout']:>8.3f} "
              f"{best_val_acc:>8.4f} {phase:>8s}")

    # --- Summary ---
    best = best_configs[0]
    print(f"\nSweep Summary:")
    print(f"  Total runs: {n_runs}")
    print(f"  Best Validation Accuracy: {best['val_accuracy']:.4f}")
    print(f"  Best Config:")
    for k, v in best.items():
        if k != "val_accuracy":
            print(f"    {k}: {v}")

    # Show top 5
    print(f"\nTop 5 Configurations:")
    for i, cfg in enumerate(best_configs[:5], 1):
        print(f"  {i}. val_acc={cfg['val_accuracy']:.4f} | "
              f"lr={cfg['learning_rate']:.6f} batch={cfg['batch_size']} "
              f"filters={cfg['filters']} dropout={cfg['dropout']:.3f}")

    return best_configs


# ============================================================
# Exercise 3: Artifacts
# ============================================================

def exercise_3_artifacts():
    """Log, version, and retrieve model artifacts and datasets.

    W&B Artifacts provide:
    - Versioned storage for datasets, models, and other files
    - Lineage tracking (which run produced/consumed an artifact)
    - Deduplication via content hashing
    - Aliases (e.g., "latest", "production")

    We simulate the full artifact lifecycle.
    """
    wandb = SimulatedWandb(project="artifact-demo")

    print("W&B Artifacts Workflow")
    print("=" * 60)

    # --- Step 1: Log a dataset artifact ---
    print("\n1. Dataset Artifact")
    print("-" * 40)

    run = wandb.init(name="data-preparation", tags=["data"])

    dataset_v1 = wandb.log_artifact(
        name="mnist-processed",
        artifact_type="dataset",
        metadata={
            "source": "torchvision.datasets.MNIST",
            "n_train": 60000,
            "n_test": 10000,
            "preprocessing": "normalize(0.1307, 0.3081)",
            "format": "torch tensors",
            "size_mb": 45.2,
        },
    )
    print(f"  Logged: {dataset_v1['name']}:{dataset_v1['version']}")
    print(f"  Type: {dataset_v1['type']}")
    print(f"  Digest: {dataset_v1['digest']}")
    print(f"  Metadata: {json.dumps(dataset_v1['metadata'], indent=4)}")

    wandb.finish()

    # --- Step 2: Log a model artifact ---
    print("\n2. Model Artifact")
    print("-" * 40)

    run = wandb.init(
        name="training-v1",
        config={"model": "SimpleCNN", "epochs": 10, "lr": 0.001},
        tags=["training"],
    )

    model_v1 = wandb.log_artifact(
        name="mnist-cnn-model",
        artifact_type="model",
        metadata={
            "architecture": "SimpleCNN(Conv2d->ReLU->MaxPool->Conv2d->ReLU->MaxPool->FC)",
            "parameters": 62006,
            "val_accuracy": 0.9845,
            "framework": "pytorch",
            "input_shape": "(1, 28, 28)",
            "output_classes": 10,
            "training_dataset": "mnist-processed:v1",
        },
    )
    print(f"  Logged: {model_v1['name']}:{model_v1['version']}")
    print(f"  Metadata: {json.dumps(model_v1['metadata'], indent=4)}")

    wandb.finish()

    # --- Step 3: Log updated versions ---
    print("\n3. Artifact Versioning")
    print("-" * 40)

    # Dataset v2 with augmentation
    run = wandb.init(name="data-augmentation", tags=["data"])
    dataset_v2 = wandb.log_artifact(
        name="mnist-processed",
        artifact_type="dataset",
        metadata={
            "source": "torchvision.datasets.MNIST",
            "n_train": 120000,  # 2x with augmentation
            "n_test": 10000,
            "preprocessing": "normalize(0.1307, 0.3081) + augment(rotate=15, translate=0.1)",
            "format": "torch tensors",
            "size_mb": 89.7,
            "changes": "Added rotation and translation augmentation",
        },
    )
    wandb.finish()

    # Model v2 trained on augmented data
    run = wandb.init(
        name="training-v2",
        config={"model": "ImprovedCNN", "epochs": 20, "lr": 0.001},
        tags=["training"],
    )
    model_v2 = wandb.log_artifact(
        name="mnist-cnn-model",
        artifact_type="model",
        metadata={
            "architecture": "ImprovedCNN(Conv2d*3->BatchNorm->Dropout->FC*2)",
            "parameters": 134590,
            "val_accuracy": 0.9923,
            "framework": "pytorch",
            "input_shape": "(1, 28, 28)",
            "output_classes": 10,
            "training_dataset": "mnist-processed:v2",
            "changes": "Deeper architecture + batch norm + augmented data",
        },
    )
    wandb.finish()

    print(f"  Dataset versions: {len(wandb.artifacts.get('mnist-processed', []))}")
    for v in wandb.artifacts.get("mnist-processed", []):
        print(f"    {v['name']}:{v['version']} - {v['metadata'].get('n_train', 'N/A')} samples")

    print(f"\n  Model versions: {len(wandb.artifacts.get('mnist-cnn-model', []))}")
    for v in wandb.artifacts.get("mnist-cnn-model", []):
        print(f"    {v['name']}:{v['version']} - "
              f"val_acc={v['metadata'].get('val_accuracy', 'N/A')} "
              f"params={v['metadata'].get('parameters', 'N/A')}")

    # --- Step 4: Artifact lineage ---
    print("\n4. Artifact Lineage")
    print("-" * 40)

    lineage = {
        "mnist-processed:v1": {
            "produced_by": "data-preparation",
            "consumed_by": ["training-v1"],
        },
        "mnist-processed:v2": {
            "produced_by": "data-augmentation",
            "consumed_by": ["training-v2"],
        },
        "mnist-cnn-model:v1": {
            "produced_by": "training-v1",
            "consumed_by": [],
            "input_artifacts": ["mnist-processed:v1"],
        },
        "mnist-cnn-model:v2": {
            "produced_by": "training-v2",
            "consumed_by": [],
            "input_artifacts": ["mnist-processed:v2"],
        },
    }

    print("  Lineage Graph:")
    print("  mnist-processed:v1 -- (data-preparation) --> training-v1 --> mnist-cnn-model:v1")
    print("  mnist-processed:v2 -- (data-augmentation) --> training-v2 --> mnist-cnn-model:v2")
    print()

    for artifact_name, info in lineage.items():
        print(f"  {artifact_name}:")
        print(f"    Produced by: {info['produced_by']}")
        print(f"    Consumed by: {info['consumed_by'] or 'none'}")
        if "input_artifacts" in info:
            print(f"    Depends on: {info['input_artifacts']}")

    # --- Step 5: Artifact comparison ---
    print("\n5. Version Comparison (Model v1 vs v2)")
    print("-" * 40)
    v1_meta = wandb.artifacts["mnist-cnn-model"][0]["metadata"]
    v2_meta = wandb.artifacts["mnist-cnn-model"][1]["metadata"]
    print(f"  {'Property':<25s} {'v1':>15s} {'v2':>15s}")
    print(f"  {'-'*55}")
    print(f"  {'Parameters':<25s} {v1_meta['parameters']:>15d} {v2_meta['parameters']:>15d}")
    print(f"  {'Val Accuracy':<25s} {v1_meta['val_accuracy']:>15.4f} {v2_meta['val_accuracy']:>15.4f}")
    print(f"  {'Training Dataset':<25s} {v1_meta['training_dataset']:>15s} {v2_meta['training_dataset']:>15s}")

    return wandb.artifacts


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    print("Exercise 1: Basic Experiment Tracking")
    print("=" * 60)
    exercise_1_basic_tracking()

    print("\n\n")
    print("Exercise 2: Run Sweeps")
    print("=" * 60)
    exercise_2_sweeps()

    print("\n\n")
    print("Exercise 3: Artifacts")
    print("=" * 60)
    exercise_3_artifacts()
