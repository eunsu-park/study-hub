# 12. 온디바이스 훈련

**이전**: [엣지 하드웨어](./11_Edge_Hardware.md) | **다음**: [실시간 추론](./13_Real_Time_Inference.md)

## 학습 목표

이 레슨을 마치면 다음을 할 수 있습니다:

1. Federated Averaging (FedAvg) 알고리즘과 통신 효율성 트레이드오프를 설명할 수 있다
2. 고정된 특징 추출기와 학습 가능한 헤드를 사용하여 온디바이스 파인튜닝을 구현할 수 있다
3. 제한된 데이터와 컴퓨팅 자원을 가진 엣지 디바이스에서 전이 학습을 적용할 수 있다
4. federated learning과 secure aggregation을 활용하여 프라이버시 보호 ML 파이프라인을 설계할 수 있다
5. 보정된 노이즈를 사용하여 모델 업데이트에 differential privacy 보장을 추가할 수 있다
6. 디바이스에서 개별 사용자에 맞게 모델을 적응시키는 개인화 시스템을 구축할 수 있다

---

대부분의 엣지 AI는 추론, 즉 사전 훈련된 모델을 디바이스에서 실행하는 데 초점을 맞춥니다. 그러나 많은 실제 애플리케이션에서는 배포 후에도 모델이 개선되어야 합니다. 키보드가 사용자의 타이핑 패턴을 학습하고, 음성 비서가 사용자의 억양에 적응하며, 제조 검사기가 새로운 제품 변형에 맞춰 조정되는 것이 그 예입니다. 온디바이스 훈련은 사용자 데이터를 로컬에 유지하고, 클라우드 의존도를 줄이며, 실시간 적응을 가능하게 합니다. 이 레슨에서는 대규모 federated learning부터 단일 디바이스에서 단일 모델을 파인튜닝하는 것까지, 자원이 제한된 디바이스에서 학습을 실용적으로 만드는 기법들을 다룹니다.

---

## 1. Federated Learning

### 1.1 Federated Learning 아키텍처

```
+-----------------------------------------------------------------+
|              Federated Learning Overview                          |
+-----------------------------------------------------------------+
|                                                                   |
|   Central Server                                                 |
|   +----------------------------+                                 |
|   |  Global Model (w_global)   |                                 |
|   +----------------------------+                                 |
|        |        |        |                                       |
|   Distribute  Distribute  Distribute                             |
|   model       model       model                                  |
|        |        |        |                                       |
|        v        v        v                                       |
|   +--------+ +--------+ +--------+                               |
|   |Device 1| |Device 2| |Device 3|                               |
|   |        | |        | |        |                               |
|   |Local   | |Local   | |Local   |                               |
|   |training| |training| |training|                               |
|   |on local| |on local| |on local|                               |
|   |data    | |data    | |data    |                               |
|   +--------+ +--------+ +--------+                               |
|        |        |        |                                       |
|   Send        Send       Send                                    |
|   updates    updates    updates                                  |
|   (not data) (not data) (not data)                               |
|        |        |        |                                       |
|        v        v        v                                       |
|   +----------------------------+                                 |
|   |  Aggregate updates         |                                 |
|   |  w_global = avg(updates)   |                                 |
|   +----------------------------+                                 |
|                                                                   |
|   Key principle: data never leaves the device.                   |
|   Only model updates (gradients or weights) are communicated.    |
|                                                                   |
+-----------------------------------------------------------------+
```

### 1.2 FedAvg 알고리즘

```python
#!/usr/bin/env python3
"""Federated Averaging (FedAvg) -- McMahan et al., 2017."""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List
import copy


@dataclass
class ModelUpdate:
    """Represents a client's model update."""
    weights: Dict[str, np.ndarray]
    num_samples: int
    client_id: str


class FedAvgServer:
    """Federated Averaging server.

    FedAvg works in rounds:
    1. Server sends global model to selected clients
    2. Each client trains locally for E epochs on local data
    3. Clients send updated weights (not data) back
    4. Server averages weights, weighted by dataset size
    """

    def __init__(self, initial_weights: Dict[str, np.ndarray]):
        self.global_weights = copy.deepcopy(initial_weights)
        self.round_number = 0

    def select_clients(self, all_clients: list,
                       fraction: float = 0.1) -> list:
        """Randomly select a fraction of clients per round.

        Selecting a subset (typically 10-20%) reduces communication
        cost and allows the system to scale to millions of devices.
        """
        num_selected = max(1, int(len(all_clients) * fraction))
        return list(np.random.choice(all_clients, num_selected, replace=False))

    def aggregate(self, updates: List[ModelUpdate]) -> None:
        """Weighted average of client model updates.

        Weights are proportional to dataset size: clients with
        more training samples have proportionally more influence
        on the global model.
        """
        total_samples = sum(u.num_samples for u in updates)

        new_weights = {}
        for key in self.global_weights:
            weighted_sum = np.zeros_like(self.global_weights[key])
            for update in updates:
                weight = update.num_samples / total_samples
                weighted_sum += weight * update.weights[key]
            new_weights[key] = weighted_sum

        self.global_weights = new_weights
        self.round_number += 1

    def get_global_model(self) -> Dict[str, np.ndarray]:
        """Return current global weights for distribution."""
        return copy.deepcopy(self.global_weights)


class FedAvgClient:
    """Federated learning client (runs on edge device)."""

    def __init__(self, client_id: str, local_data: tuple,
                 learning_rate: float = 0.01):
        self.client_id = client_id
        self.x_train, self.y_train = local_data
        self.lr = learning_rate

    def local_train(self, global_weights: Dict[str, np.ndarray],
                    local_epochs: int = 5,
                    batch_size: int = 32) -> ModelUpdate:
        """Train locally for E epochs starting from global weights.

        The number of local epochs (E) controls the trade-off between
        communication efficiency and convergence:
        - More epochs = fewer communication rounds, but may diverge
        - Fewer epochs = more stable, but higher communication cost
        """
        weights = copy.deepcopy(global_weights)
        num_samples = len(self.x_train)

        for epoch in range(local_epochs):
            # Shuffle data each epoch
            indices = np.random.permutation(num_samples)

            for start in range(0, num_samples, batch_size):
                batch_idx = indices[start:start + batch_size]
                x_batch = self.x_train[batch_idx]
                y_batch = self.y_train[batch_idx]

                # Forward + backward (simplified for demonstration)
                gradients = self._compute_gradients(weights, x_batch, y_batch)

                # SGD update
                for key in weights:
                    weights[key] -= self.lr * gradients[key]

        return ModelUpdate(
            weights=weights,
            num_samples=num_samples,
            client_id=self.client_id
        )

    def _compute_gradients(self, weights, x, y):
        """Compute gradients (placeholder -- replace with real model)."""
        gradients = {}
        for key in weights:
            gradients[key] = np.random.randn(*weights[key].shape) * 0.01
        return gradients


# --- Simulation ---
def run_federated_simulation(num_clients: int = 100,
                             num_rounds: int = 20,
                             clients_per_round: float = 0.1):
    """Simulate federated learning across multiple clients."""
    # Initialize global model weights
    initial_weights = {
        "layer1": np.random.randn(784, 128) * 0.01,
        "layer2": np.random.randn(128, 10) * 0.01,
    }

    server = FedAvgServer(initial_weights)

    # Create clients with non-IID data partitions
    clients = []
    for i in range(num_clients):
        n = np.random.randint(50, 500)  # Varying dataset sizes
        local_data = (
            np.random.randn(n, 784).astype(np.float32),
            np.random.randint(0, 10, size=n)
        )
        clients.append(FedAvgClient(f"client_{i}", local_data))

    # Training rounds
    for round_num in range(num_rounds):
        # Select clients
        selected = server.select_clients(clients, fraction=clients_per_round)

        # Distribute global model and train locally
        global_weights = server.get_global_model()
        updates = []
        for client in selected:
            update = client.local_train(global_weights, local_epochs=5)
            updates.append(update)

        # Aggregate
        server.aggregate(updates)
        print(f"Round {round_num + 1}/{num_rounds}: "
              f"{len(selected)} clients, "
              f"total samples: {sum(u.num_samples for u in updates)}")


if __name__ == "__main__":
    run_federated_simulation()
```

### 1.3 통신 효율성

```python
#!/usr/bin/env python3
"""Techniques to reduce communication in federated learning."""

import numpy as np
from typing import Dict


def gradient_compression(gradients: Dict[str, np.ndarray],
                         top_k_fraction: float = 0.01) -> Dict[str, tuple]:
    """Top-K sparsification: only send the largest gradient values.

    Instead of sending the full gradient vector (e.g., 10M parameters),
    send only the top 1% of values by magnitude. This reduces
    upload bandwidth by ~100x with minimal accuracy loss.
    """
    compressed = {}
    for key, grad in gradients.items():
        flat = grad.ravel()
        k = max(1, int(len(flat) * top_k_fraction))

        # Select top-k by magnitude
        top_indices = np.argsort(np.abs(flat))[-k:]
        top_values = flat[top_indices]

        compressed[key] = (top_indices, top_values, grad.shape)

    return compressed


def quantize_gradients(gradients: Dict[str, np.ndarray],
                       num_bits: int = 8) -> Dict[str, tuple]:
    """Quantize gradient values to reduce communication size.

    Instead of 32-bit floats, compress each gradient tensor to
    num_bits per value. For 8-bit quantization, this is a 4x reduction.
    """
    quantized = {}
    for key, grad in gradients.items():
        min_val = grad.min()
        max_val = grad.max()

        # Scale to [0, 2^bits - 1]
        num_levels = (1 << num_bits) - 1
        if max_val - min_val > 0:
            scaled = ((grad - min_val) / (max_val - min_val) * num_levels)
            quantized_grad = np.round(scaled).astype(np.uint8)
        else:
            quantized_grad = np.zeros_like(grad, dtype=np.uint8)

        quantized[key] = (quantized_grad, min_val, max_val, grad.shape)

    return quantized


def federated_dropout(weights: Dict[str, np.ndarray],
                      dropout_rate: float = 0.5) -> Dict[str, np.ndarray]:
    """Structured federated dropout: each client trains a random submodel.

    By training different subsets of neurons per client, we reduce both
    computation and communication while maintaining global model diversity.
    """
    masked_weights = {}
    for key, w in weights.items():
        mask = np.random.binomial(1, 1 - dropout_rate, size=w.shape)
        masked_weights[key] = w * mask / (1 - dropout_rate)
    return masked_weights


# Measure compression ratios
def compare_compression():
    """Compare communication cost of different strategies."""
    # Simulate a model with ~10M parameters
    gradients = {
        "conv1": np.random.randn(64, 3, 3, 3).astype(np.float32),
        "conv2": np.random.randn(128, 64, 3, 3).astype(np.float32),
        "fc1": np.random.randn(1024, 512).astype(np.float32),
        "fc2": np.random.randn(512, 10).astype(np.float32),
    }

    full_size = sum(g.nbytes for g in gradients.values())
    print(f"Full gradient size: {full_size / 1024:.1f} KB")

    # Top-K (1%)
    compressed = gradient_compression(gradients, top_k_fraction=0.01)
    topk_size = sum(
        idx.nbytes + val.nbytes
        for idx, val, _ in compressed.values()
    )
    print(f"Top-1% compressed:  {topk_size / 1024:.1f} KB "
          f"({topk_size / full_size * 100:.1f}%)")

    # 8-bit quantization
    quantized = quantize_gradients(gradients, num_bits=8)
    quant_size = sum(q.nbytes + 8 for q, _, _, _ in quantized.values())
    print(f"8-bit quantized:    {quant_size / 1024:.1f} KB "
          f"({quant_size / full_size * 100:.1f}%)")


if __name__ == "__main__":
    compare_compression()
```

---

## 2. 온디바이스 파인튜닝

### 2.1 파인튜닝 전략

```
+-----------------------------------------------------------------+
|             On-Device Fine-Tuning Strategy                       |
+-----------------------------------------------------------------+
|                                                                   |
|   Pre-trained Model (from cloud)                                 |
|   +----------------------------------------------------------+  |
|   | Feature Extractor         |  Classification Head          |  |
|   | (Conv layers)             |  (FC layers)                  |  |
|   |                           |                               |  |
|   | [FROZEN]                  |  [TRAINABLE]                  |  |
|   | - No gradient computation |  - Fine-tune with local data  |  |
|   | - Saves memory + compute  |  - Small parameter count      |  |
|   | - Preserves learned       |  - Quick adaptation           |  |
|   |   features                |                               |  |
|   +----------------------------------------------------------+  |
|                                                                   |
|   Why freeze the backbone?                                       |
|   - Edge devices have limited RAM (can't store all gradients)    |
|   - The backbone already knows good features (ImageNet, etc.)    |
|   - Only the head (last 1-2 layers) needs task-specific tuning   |
|   - Reduces training time from hours to minutes                  |
|                                                                   |
+-----------------------------------------------------------------+
```

### 2.2 PyTorch 온디바이스 파인튜닝

```python
#!/usr/bin/env python3
"""On-device fine-tuning with frozen backbone."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import time


class OnDeviceFineTuner:
    """Fine-tune a pre-trained model on edge device.

    Strategy: freeze the feature extractor and only train the
    classification head. This reduces memory usage by ~10x and
    training time by ~5-20x compared to full fine-tuning.
    """

    def __init__(self, model: nn.Module, num_classes: int,
                 device: str = "cpu"):
        self.device = torch.device(device)
        self.model = model.to(self.device)

        # Freeze all backbone parameters
        for param in self.model.parameters():
            param.requires_grad = False

        # Replace and unfreeze classification head
        if hasattr(self.model, "classifier"):
            in_features = self._get_head_in_features(self.model.classifier)
            self.model.classifier = nn.Linear(in_features, num_classes)
        elif hasattr(self.model, "fc"):
            in_features = self.model.fc.in_features
            self.model.fc = nn.Linear(in_features, num_classes)

        self.model.to(self.device)

        # Only optimize unfrozen parameters
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.Adam(trainable_params, lr=1e-3)
        self.criterion = nn.CrossEntropyLoss()

        # Report parameter counts
        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Total params:     {total:,}")
        print(f"Trainable params: {trainable:,} ({trainable/total*100:.1f}%)")

    def _get_head_in_features(self, head):
        """Extract input features from a Sequential or Linear head."""
        if isinstance(head, nn.Linear):
            return head.in_features
        for module in head.modules():
            if isinstance(module, nn.Linear):
                return module.in_features
        raise ValueError("Could not determine head input features")

    def fine_tune(self, train_data: tuple,
                  epochs: int = 10,
                  batch_size: int = 16) -> dict:
        """Fine-tune on local data."""
        x_train, y_train = train_data
        dataset = TensorDataset(
            torch.tensor(x_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.long)
        )
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.model.train()
        history = {"loss": [], "accuracy": []}

        for epoch in range(epochs):
            epoch_loss = 0.0
            correct = 0
            total = 0

            start = time.perf_counter()

            for x_batch, y_batch in loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(x_batch)
                loss = self.criterion(outputs, y_batch)
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item() * x_batch.size(0)
                correct += (outputs.argmax(1) == y_batch).sum().item()
                total += x_batch.size(0)

            elapsed = time.perf_counter() - start
            avg_loss = epoch_loss / total
            accuracy = correct / total

            history["loss"].append(avg_loss)
            history["accuracy"].append(accuracy)

            print(f"Epoch {epoch+1}/{epochs}: "
                  f"loss={avg_loss:.4f}, acc={accuracy:.4f}, "
                  f"time={elapsed:.1f}s")

        return history


if __name__ == "__main__":
    from torchvision.models import mobilenet_v2

    # Simulate on-device scenario
    model = mobilenet_v2(weights="IMAGENET1K_V1")

    tuner = OnDeviceFineTuner(model, num_classes=5, device="cpu")

    # Simulate local data (50 images, 5 classes)
    x_train = torch.randn(50, 3, 224, 224).numpy()
    y_train = torch.randint(0, 5, (50,)).numpy()

    history = tuner.fine_tune(
        (x_train, y_train),
        epochs=5,
        batch_size=8
    )
```

---

## 3. 엣지에서의 전이 학습

### 3.1 특징 추출 파이프라인

```python
#!/usr/bin/env python3
"""Efficient transfer learning: extract features once, train head repeatedly."""

import torch
import torch.nn as nn
import numpy as np
import time


class EdgeTransferLearner:
    """Two-stage transfer learning optimized for edge.

    Stage 1: Extract features from frozen backbone (one-time cost)
    Stage 2: Train lightweight head on extracted features (fast, repeatable)

    This is more efficient than fine-tuning because:
    - Feature extraction runs once per image (not once per epoch)
    - The head trains on small feature vectors, not full images
    - Stage 2 fits in minimal RAM (no backbone in memory during training)
    """

    def __init__(self, backbone: nn.Module, feature_dim: int):
        self.backbone = backbone
        self.backbone.eval()
        self.feature_dim = feature_dim

        # Remove classification head
        if hasattr(self.backbone, "classifier"):
            self.backbone.classifier = nn.Identity()
        elif hasattr(self.backbone, "fc"):
            self.backbone.fc = nn.Identity()

        for param in self.backbone.parameters():
            param.requires_grad = False

    def extract_features(self, images: torch.Tensor,
                         batch_size: int = 8) -> np.ndarray:
        """Extract features from all images (one-time cost)."""
        features = []

        start = time.perf_counter()
        with torch.no_grad():
            for i in range(0, len(images), batch_size):
                batch = images[i:i + batch_size]
                feat = self.backbone(batch)
                if feat.dim() > 2:
                    feat = feat.mean(dim=[2, 3])  # Global average pool
                features.append(feat.numpy())

        all_features = np.concatenate(features, axis=0)
        elapsed = time.perf_counter() - start
        print(f"Extracted {len(images)} features in {elapsed:.1f}s "
              f"({len(images)/elapsed:.0f} img/s)")

        return all_features

    def train_head(self, features: np.ndarray, labels: np.ndarray,
                   num_classes: int, epochs: int = 50,
                   lr: float = 0.01) -> nn.Linear:
        """Train a linear head on extracted features.

        This is extremely fast because we are training a single
        linear layer on pre-computed feature vectors.
        """
        head = nn.Linear(self.feature_dim, num_classes)
        optimizer = torch.optim.SGD(head.parameters(), lr=lr, momentum=0.9)
        criterion = nn.CrossEntropyLoss()

        x = torch.tensor(features, dtype=torch.float32)
        y = torch.tensor(labels, dtype=torch.long)

        head.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            output = head(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 10 == 0:
                acc = (output.argmax(1) == y).float().mean()
                print(f"  Head epoch {epoch+1}: loss={loss:.4f}, acc={acc:.4f}")

        return head


if __name__ == "__main__":
    from torchvision.models import mobilenet_v2

    backbone = mobilenet_v2(weights="IMAGENET1K_V1")
    learner = EdgeTransferLearner(backbone, feature_dim=1280)

    # Simulate local data
    images = torch.randn(100, 3, 224, 224)
    labels = np.random.randint(0, 3, size=100)

    features = learner.extract_features(images)
    head = learner.train_head(features, labels, num_classes=3, epochs=50)
```

---

## 4. 프라이버시 보호 ML

### 4.1 프라이버시 기법 개요

```
+-----------------------------------------------------------------+
|           Privacy-Preserving Techniques for Edge AI              |
+-----------------------------------------------------------------+
|                                                                   |
|   Technique              What It Protects    Overhead             |
|   +-----------------------------------------------------------+ |
|   | Federated Learning   | Raw data stays    | Communication    | |
|   |                      | on device         | rounds           | |
|   +-----------------------------------------------------------+ |
|   | Differential Privacy | Individual data   | Accuracy loss    | |
|   |                      | points in updates | (noise added)    | |
|   +-----------------------------------------------------------+ |
|   | Secure Aggregation   | Individual model  | Crypto overhead  | |
|   |                      | updates           | (MPC protocol)   | |
|   +-----------------------------------------------------------+ |
|   | Homomorphic          | Encrypted data    | Very high        | |
|   | Encryption           | during compute    | compute cost     | |
|   +-----------------------------------------------------------+ |
|   | Trusted Execution    | Data in memory    | Hardware TEE     | |
|   | Environments (TEE)   | during processing | required         | |
|   +-----------------------------------------------------------+ |
|                                                                   |
+-----------------------------------------------------------------+
```

### 4.2 Secure Aggregation

```python
#!/usr/bin/env python3
"""Simplified secure aggregation for federated learning."""

import numpy as np
from typing import Dict, List
import hashlib


class SecureAggregation:
    """Secure aggregation prevents the server from seeing individual updates.

    Protocol (simplified Bonawitz et al., 2017):
    1. Each pair of clients agrees on a random mask (via key exchange)
    2. Client i adds mask_ij to its update, client j subtracts mask_ij
    3. When server sums all masked updates, masks cancel out
    4. Server gets the aggregate but not individual contributions

    This implementation demonstrates the concept with pairwise masking.
    """

    @staticmethod
    def generate_pairwise_mask(client_i: str, client_j: str,
                               shape: tuple,
                               seed_base: str = "round_1") -> np.ndarray:
        """Generate a deterministic mask shared between two clients."""
        # Both clients can compute the same mask from their shared secret
        pair_key = "".join(sorted([client_i, client_j])) + seed_base
        seed = int(hashlib.sha256(pair_key.encode()).hexdigest()[:8], 16)
        rng = np.random.RandomState(seed)
        return rng.randn(*shape).astype(np.float32)

    @staticmethod
    def mask_update(update: np.ndarray,
                    client_id: str,
                    all_client_ids: list,
                    shape: tuple) -> np.ndarray:
        """Apply pairwise masks to an update."""
        masked = update.copy()

        for other_id in all_client_ids:
            if other_id == client_id:
                continue

            mask = SecureAggregation.generate_pairwise_mask(
                client_id, other_id, shape
            )

            if client_id < other_id:
                masked += mask   # Client i adds
            else:
                masked -= mask   # Client j subtracts

        return masked


def demonstrate_secure_aggregation():
    """Show that masks cancel during aggregation."""
    clients = ["alice", "bob", "charlie"]
    shape = (4,)

    # Each client has a real update
    real_updates = {
        "alice": np.array([1.0, 2.0, 3.0, 4.0]),
        "bob": np.array([0.5, 1.5, 2.5, 3.5]),
        "charlie": np.array([0.1, 0.2, 0.3, 0.4]),
    }

    # Mask each update
    masked_updates = {}
    for client_id, update in real_updates.items():
        masked = SecureAggregation.mask_update(
            update, client_id, clients, shape
        )
        masked_updates[client_id] = masked
        print(f"{client_id} masked update: {masked}")

    # Server sums masked updates (masks cancel out)
    aggregate = sum(masked_updates.values())
    true_aggregate = sum(real_updates.values())

    print(f"\nServer aggregate:     {aggregate}")
    print(f"True aggregate:       {true_aggregate}")
    print(f"Difference (should be ~0): {np.abs(aggregate - true_aggregate).max():.2e}")


if __name__ == "__main__":
    demonstrate_secure_aggregation()
```

---

## 5. Differential Privacy

### 5.1 DP 기초

```
+-----------------------------------------------------------------+
|              Differential Privacy Basics                          |
+-----------------------------------------------------------------+
|                                                                   |
|   Definition: A mechanism M satisfies (epsilon, delta)-DP if     |
|   for any two datasets D, D' differing in one record:            |
|                                                                   |
|       P[M(D) in S] <= e^epsilon * P[M(D') in S] + delta          |
|                                                                   |
|   Intuition: removing or adding one person's data barely         |
|   changes the output distribution. An adversary cannot tell      |
|   whether any individual participated.                           |
|                                                                   |
|   epsilon (privacy budget):                                      |
|   - Smaller = stronger privacy, more noise, less accuracy        |
|   - Typical range: 1.0 - 10.0 for practical systems             |
|   - epsilon < 1: strong privacy                                  |
|   - epsilon > 10: weak privacy                                   |
|                                                                   |
|   delta:                                                         |
|   - Probability of privacy breach                                |
|   - Typically 1/N^2 where N is dataset size                      |
|                                                                   |
+-----------------------------------------------------------------+
```

### 5.2 온디바이스 훈련을 위한 DP-SGD

```python
#!/usr/bin/env python3
"""Differentially Private Stochastic Gradient Descent (DP-SGD)."""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class DPSGD:
    """DP-SGD optimizer for privacy-preserving training.

    DP-SGD modifies standard SGD in two ways:
    1. Clip per-sample gradients to bound sensitivity
    2. Add calibrated Gaussian noise to the clipped gradient sum

    The noise scale is determined by the privacy budget (epsilon).
    """

    def __init__(self, params, lr: float = 0.01,
                 max_grad_norm: float = 1.0,
                 noise_multiplier: float = 1.0):
        self.params = list(params)
        self.lr = lr
        self.max_grad_norm = max_grad_norm
        self.noise_multiplier = noise_multiplier

    def clip_and_noise_step(self, per_sample_grads: list) -> None:
        """Apply gradient clipping and noise addition.

        Args:
            per_sample_grads: List of gradient dicts, one per sample
        """
        batch_size = len(per_sample_grads)

        for param_idx, param in enumerate(self.params):
            # Collect per-sample gradients for this parameter
            sample_grads = []
            for sample in per_sample_grads:
                sample_grads.append(sample[param_idx])

            # Step 1: Clip each per-sample gradient
            clipped_grads = []
            for grad in sample_grads:
                grad_norm = torch.norm(grad)
                clip_factor = min(1.0, self.max_grad_norm / (grad_norm + 1e-8))
                clipped_grads.append(grad * clip_factor)

            # Step 2: Sum clipped gradients
            summed = torch.stack(clipped_grads).sum(dim=0)

            # Step 3: Add Gaussian noise
            noise_std = self.max_grad_norm * self.noise_multiplier
            noise = torch.randn_like(summed) * noise_std
            noisy_grad = (summed + noise) / batch_size

            # Step 4: Update parameter
            with torch.no_grad():
                param -= self.lr * noisy_grad


def compute_per_sample_gradients(model: nn.Module,
                                 criterion: nn.Module,
                                 x_batch: torch.Tensor,
                                 y_batch: torch.Tensor) -> list:
    """Compute gradients for each sample individually.

    Standard backprop computes the sum of gradients across the batch.
    For DP-SGD, we need per-sample gradients to clip each one
    independently before summing.
    """
    per_sample_grads = []

    for i in range(len(x_batch)):
        model.zero_grad()
        output = model(x_batch[i:i+1])
        loss = criterion(output, y_batch[i:i+1])
        loss.backward()

        grads = [p.grad.clone() for p in model.parameters() if p.requires_grad]
        per_sample_grads.append(grads)

    return per_sample_grads


def train_with_dp(model: nn.Module,
                  train_data: tuple,
                  epochs: int = 10,
                  epsilon: float = 8.0,
                  delta: float = 1e-5,
                  max_grad_norm: float = 1.0,
                  batch_size: int = 32):
    """Train a model with DP-SGD guarantees."""
    x_train, y_train = train_data
    dataset = TensorDataset(
        torch.tensor(x_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long)
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Compute noise multiplier from epsilon, delta, and number of steps
    num_steps = len(loader) * epochs
    # Simplified: in practice, use privacy accounting (e.g., Renyi DP)
    noise_multiplier = np.sqrt(2 * np.log(1.25 / delta)) / epsilon

    optimizer = DPSGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=0.01,
        max_grad_norm=max_grad_norm,
        noise_multiplier=noise_multiplier
    )
    criterion = nn.CrossEntropyLoss()

    print(f"Training with (epsilon={epsilon}, delta={delta})-DP")
    print(f"Noise multiplier: {noise_multiplier:.4f}")

    for epoch in range(epochs):
        total_loss = 0
        for x_batch, y_batch in loader:
            per_sample_grads = compute_per_sample_gradients(
                model, criterion, x_batch, y_batch
            )
            optimizer.clip_and_noise_step(per_sample_grads)

            with torch.no_grad():
                output = model(x_batch)
                total_loss += criterion(output, y_batch).item()

        print(f"Epoch {epoch+1}/{epochs}: loss={total_loss/len(loader):.4f}")
```

---

## 6. 개인화

### 6.1 개인화 전략

```
+-----------------------------------------------------------------+
|          Edge AI Personalization Strategies                       |
+-----------------------------------------------------------------+
|                                                                   |
|   Strategy 1: Local Fine-Tuning                                  |
|   - Fine-tune last layers on user data                           |
|   - Simple, effective for few classes                            |
|   - Risk: catastrophic forgetting on rare events                 |
|                                                                   |
|   Strategy 2: Per-User Adapter Layers                            |
|   - Add small adapter modules per user                           |
|   - Shared backbone + user-specific adapters                     |
|   - Low memory per user (<1% of model)                           |
|                                                                   |
|   Strategy 3: Meta-Learning (MAML-style)                         |
|   - Train a model that adapts fast to new tasks                  |
|   - Few-shot personalization (5-10 examples)                     |
|   - Higher initial training cost                                 |
|                                                                   |
|   Strategy 4: Mixture of Experts                                 |
|   - Multiple specialist sub-models                               |
|   - Router learns which expert to use per input                  |
|   - Natural personalization through routing                      |
|                                                                   |
+-----------------------------------------------------------------+
```

### 6.2 어댑터 기반 개인화

```python
#!/usr/bin/env python3
"""Lightweight per-user adapters for on-device personalization."""

import torch
import torch.nn as nn
from pathlib import Path
import json


class PersonalAdapter(nn.Module):
    """Small bottleneck adapter injected into a frozen backbone.

    Architecture: input -> down_proj -> ReLU -> up_proj -> output + input
    The adapter adds <1% parameters to the model but allows
    user-specific customization. Each user's adapter is stored
    as a small file (~10-100 KB).
    """

    def __init__(self, feature_dim: int, bottleneck: int = 16):
        super().__init__()
        self.down = nn.Linear(feature_dim, bottleneck)
        self.up = nn.Linear(bottleneck, feature_dim)
        self.relu = nn.ReLU()

        # Initialize near-identity (adapter starts as no-op)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.relu(self.down(x))
        x = self.up(x)
        return x + residual


class PersonalizedModel(nn.Module):
    """Model with per-user adapters."""

    def __init__(self, backbone: nn.Module, feature_dim: int,
                 num_classes: int, adapter_bottleneck: int = 16):
        super().__init__()
        self.backbone = backbone
        self.adapter = PersonalAdapter(feature_dim, adapter_bottleneck)
        self.head = nn.Linear(feature_dim, num_classes)

        # Freeze backbone
        for param in self.backbone.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            features = self.backbone(x)
            if features.dim() > 2:
                features = features.mean(dim=[2, 3])
        adapted = self.adapter(features)
        return self.head(adapted)

    def save_personal_weights(self, user_id: str, save_dir: str = "./users"):
        """Save only the adapter and head weights (tiny file)."""
        path = Path(save_dir)
        path.mkdir(exist_ok=True)

        state = {
            "adapter": self.adapter.state_dict(),
            "head": self.head.state_dict(),
        }
        torch.save(state, path / f"{user_id}_adapter.pt")

        size_kb = (path / f"{user_id}_adapter.pt").stat().st_size / 1024
        print(f"Saved {user_id} adapter: {size_kb:.1f} KB")

    def load_personal_weights(self, user_id: str, save_dir: str = "./users"):
        """Load a specific user's adapter weights."""
        path = Path(save_dir) / f"{user_id}_adapter.pt"
        state = torch.load(path, map_location="cpu")
        self.adapter.load_state_dict(state["adapter"])
        self.head.load_state_dict(state["head"])
        print(f"Loaded adapter for user: {user_id}")


class PersonalizationManager:
    """Manage personalized models for multiple users."""

    def __init__(self, base_model: PersonalizedModel):
        self.base_model = base_model
        self.user_metadata = {}

    def personalize_for_user(self, user_id: str,
                             user_data: tuple,
                             epochs: int = 20,
                             lr: float = 1e-3):
        """Fine-tune adapter for a specific user."""
        x_data, y_data = user_data

        # Only optimize adapter + head
        trainable = list(self.base_model.adapter.parameters()) + \
                    list(self.base_model.head.parameters())
        optimizer = torch.optim.Adam(trainable, lr=lr)
        criterion = nn.CrossEntropyLoss()

        x = torch.tensor(x_data, dtype=torch.float32)
        y = torch.tensor(y_data, dtype=torch.long)

        self.base_model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            output = self.base_model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

        acc = (self.base_model(x).argmax(1) == y).float().mean().item()
        print(f"User {user_id}: accuracy={acc:.4f} after {epochs} epochs")

        self.base_model.save_personal_weights(user_id)
        self.user_metadata[user_id] = {
            "samples": len(x_data),
            "accuracy": acc,
        }

    def switch_user(self, user_id: str):
        """Hot-swap to a different user's personalization."""
        self.base_model.load_personal_weights(user_id)
```

---

## 연습 문제

### 연습 1: Federated Learning 시뮬레이션
1. 각 클라이언트가 MNIST 숫자의 서로 다른 부분 집합을 보유하는 non-IID 설정에서 10개의 클라이언트로 FedAvg를 구현하십시오
2. 50 라운드 동안 훈련하고 라운드별 글로벌 모델 정확도를 플로팅하십시오
3. IID 데이터 분할과 non-IID 데이터 분할의 수렴을 비교하십시오

### 연습 2: 온디바이스 파인튜닝
1. 사전 훈련된 MobileNetV2를 로드하고 분류기를 제외한 모든 레이어를 동결하십시오
2. 3개의 새로운 클래스에서 20개의 이미지로 파인튜닝하십시오 (사용자 정의 사용 사례 시뮬레이션)
3. CPU에서 훈련 시간과 메모리 사용량을 측정하십시오

### 연습 3: Differential Privacy
1. 소규모 데이터셋에서 DP-SGD를 사용한 경우와 사용하지 않은 경우 모두 간단한 MLP를 훈련하십시오
2. epsilon 값 1, 5, 10에서 정확도를 비교하십시오
3. 프라이버시-유틸리티 트레이드오프 곡선을 플로팅하십시오

---

**이전**: [엣지 하드웨어](./11_Edge_Hardware.md) | **다음**: [실시간 추론](./13_Real_Time_Inference.md)
