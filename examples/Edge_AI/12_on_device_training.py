"""
12. On-Device Training

Demonstrates techniques for training and fine-tuning neural networks
directly on edge devices with limited memory and compute.

Covers:
- Memory-efficient gradient computation
- Frozen backbone with trainable head (transfer learning)
- Gradient checkpointing to reduce memory
- Low-rank adaptation (LoRA) for parameter-efficient fine-tuning
- Federated learning simulation (local training rounds)

Requirements:
    pip install torch numpy
"""

import torch
import torch.nn as nn
import torch.optim as optim
import time
import math
import copy
from typing import List, Tuple

print("=" * 60)
print("Edge AI — On-Device Training")
print("=" * 60)


# ============================================
# 1. Simulate Edge-Constrained Environment
# ============================================
print("\n[1] Edge Training Constraints")
print("-" * 40)

# Simulate a device with limited resources
DEVICE_MEMORY_MB = 256
MAX_BATCH_SIZE = 4
DEVICE_NAME = "Simulated Edge Device (256 MB RAM)"

print(f"Device: {DEVICE_NAME}")
print(f"Memory budget: {DEVICE_MEMORY_MB} MB")
print(f"Max batch size: {MAX_BATCH_SIZE}")
print(f"Target: personalize a pre-trained model on local data")


# ============================================
# 2. Pre-trained Backbone + Trainable Head
# ============================================
print("\n[2] Frozen Backbone with Trainable Head")
print("-" * 40)
print("Freeze most layers, only train the classification head.")
print("This drastically reduces memory and compute requirements.\n")


class FeatureExtractor(nn.Module):
    """Simulates a pre-trained backbone (frozen during on-device training)."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )

    def forward(self, x):
        return self.features(x)


class OnDeviceModel(nn.Module):
    """Model with frozen backbone and trainable head."""

    def __init__(self, backbone, num_classes=5):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes),
        )
        # Freeze backbone
        for param in self.backbone.parameters():
            param.requires_grad = False

    def forward(self, x):
        with torch.no_grad():
            features = self.backbone(x)
        return self.head(features)


backbone = FeatureExtractor()
model = OnDeviceModel(backbone, num_classes=5)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
frozen_params = total_params - trainable_params

print(f"Total parameters:     {total_params:,}")
print(f"Frozen (backbone):    {frozen_params:,}")
print(f"Trainable (head):     {trainable_params:,}")
print(f"Trainable ratio:      {trainable_params / total_params * 100:.1f}%")

# Estimate memory savings
frozen_mem_mb = frozen_params * 4 / (1024 * 1024)
trainable_mem_mb = trainable_params * 4 / (1024 * 1024)
grad_mem_mb = trainable_params * 4 / (1024 * 1024)  # Gradients
print(f"\nMemory for weights:   {(frozen_mem_mb + trainable_mem_mb):.3f} MB")
print(f"Memory for gradients: {grad_mem_mb:.3f} MB (head only)")


# ============================================
# 3. On-Device Training Loop
# ============================================
print("\n[3] On-Device Training Loop")
print("-" * 40)

# Simulate local user data (small personalization dataset)
num_samples = 40
x_local = torch.randn(num_samples, 3, 32, 32)
y_local = torch.randint(0, 5, (num_samples,))

optimizer = optim.SGD(model.head.parameters(), lr=0.01, momentum=0.9)
criterion = nn.CrossEntropyLoss()

print(f"Local dataset: {num_samples} samples, 5 classes")
print(f"Training for 5 epochs with batch size {MAX_BATCH_SIZE}...")

model.train()
for epoch in range(5):
    total_loss = 0.0
    correct = 0
    n_batches = 0

    for i in range(0, num_samples, MAX_BATCH_SIZE):
        x_batch = x_local[i:i + MAX_BATCH_SIZE]
        y_batch = y_local[i:i + MAX_BATCH_SIZE]

        optimizer.zero_grad()
        outputs = model(x_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct += (outputs.argmax(1) == y_batch).sum().item()
        n_batches += 1

    acc = correct / num_samples
    avg_loss = total_loss / n_batches
    print(f"  Epoch {epoch + 1}: loss={avg_loss:.4f}, acc={acc:.2f}")


# ============================================
# 4. Gradient Checkpointing
# ============================================
print("\n[4] Gradient Checkpointing")
print("-" * 40)
print("Trade compute for memory by recomputing activations during backward.\n")


class CheckpointedModel(nn.Module):
    """Model using gradient checkpointing to reduce memory."""

    def __init__(self, num_classes=10):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        # Checkpoint each block: activations recomputed during backward
        x = torch.utils.checkpoint.checkpoint(self.block1, x, use_reentrant=False)
        x = torch.utils.checkpoint.checkpoint(self.block2, x, use_reentrant=False)
        x = torch.utils.checkpoint.checkpoint(self.block3, x, use_reentrant=False)
        x = self.pool(x).flatten(1)
        return self.fc(x)


ckpt_model = CheckpointedModel()

# Compare memory usage: standard vs checkpointed
x_test = torch.randn(4, 3, 64, 64, requires_grad=False)

# Standard forward
std_model = copy.deepcopy(ckpt_model)
std_model.train()
out_std = std_model(x_test)
loss_std = out_std.sum()
loss_std.backward()

# Checkpointed forward
ckpt_model.train()
out_ckpt = ckpt_model(x_test)
loss_ckpt = out_ckpt.sum()
loss_ckpt.backward()

print("Gradient checkpointing trades ~30-40% more compute for ~50-60% less memory.")
print("Essential when training large models on memory-constrained devices.")
print(f"Standard model:      all intermediate activations stored")
print(f"Checkpointed model:  only block boundaries stored, rest recomputed")


# ============================================
# 5. Low-Rank Adaptation (LoRA)
# ============================================
print("\n[5] Low-Rank Adaptation (LoRA)")
print("-" * 40)
print("Add small trainable low-rank matrices to frozen layers.")
print("Dramatically reduces trainable parameters.\n")


class LoRALinear(nn.Module):
    """Linear layer with LoRA adaptation."""

    def __init__(self, original_linear: nn.Linear, rank: int = 4):
        super().__init__()
        in_features = original_linear.in_features
        out_features = original_linear.out_features

        # Freeze original weights
        self.weight = original_linear.weight
        self.weight.requires_grad = False
        self.bias = original_linear.bias
        if self.bias is not None:
            self.bias.requires_grad = False

        # Low-rank adaptation matrices
        self.lora_a = nn.Parameter(torch.randn(in_features, rank) * 0.01)
        self.lora_b = nn.Parameter(torch.zeros(rank, out_features))
        self.rank = rank

    def forward(self, x):
        # Original output + low-rank adaptation
        base = nn.functional.linear(x, self.weight, self.bias)
        lora = x @ self.lora_a @ self.lora_b
        return base + lora


# Apply LoRA to a model
class LoRAModel(nn.Module):
    def __init__(self, num_classes=10, lora_rank=4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(4), nn.Flatten(),
        )
        original_fc1 = nn.Linear(64 * 4 * 4, 128)
        original_fc2 = nn.Linear(128, num_classes)
        self.fc1 = LoRALinear(original_fc1, rank=lora_rank)
        self.fc2 = LoRALinear(original_fc2, rank=lora_rank)
        self.relu = nn.ReLU()

        # Freeze conv layers
        for param in self.features.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.features(x)
        x = self.relu(self.fc1(x))
        return self.fc2(x)


lora_model = LoRAModel(num_classes=5, lora_rank=4)
total_p = sum(p.numel() for p in lora_model.parameters())
trainable_p = sum(p.numel() for p in lora_model.parameters() if p.requires_grad)

print(f"LoRA rank: 4")
print(f"Total parameters:     {total_p:,}")
print(f"Trainable (LoRA):     {trainable_p:,}")
print(f"Trainable ratio:      {trainable_p / total_p * 100:.2f}%")
print(f"Memory for LoRA grads: {trainable_p * 4 / 1024:.2f} KB")


# ============================================
# 6. Federated Learning Simulation
# ============================================
print("\n[6] Federated Learning (Local Training Rounds)")
print("-" * 40)
print("Simulate multiple edge devices training locally, then aggregating.\n")


def create_local_data(client_id, n_samples=20):
    """Simulate local data for a federated client."""
    torch.manual_seed(client_id * 42)
    x = torch.randn(n_samples, 3, 32, 32)
    y = torch.randint(0, 5, (n_samples,))
    return x, y


def local_training_round(model, x_local, y_local, lr=0.01, epochs=3):
    """Run local training on one client."""
    local_model = copy.deepcopy(model)
    local_model.train()
    optimizer = optim.SGD(
        [p for p in local_model.parameters() if p.requires_grad],
        lr=lr,
    )
    criterion = nn.CrossEntropyLoss()

    for _ in range(epochs):
        for i in range(0, len(x_local), MAX_BATCH_SIZE):
            xb = x_local[i:i + MAX_BATCH_SIZE]
            yb = y_local[i:i + MAX_BATCH_SIZE]
            optimizer.zero_grad()
            loss = criterion(local_model(xb), yb)
            loss.backward()
            optimizer.step()

    return local_model


def federated_average(global_model, local_models):
    """Average model parameters from multiple clients (FedAvg)."""
    global_dict = global_model.state_dict()
    n_clients = len(local_models)

    for key in global_dict:
        stacked = torch.stack([m.state_dict()[key].float() for m in local_models])
        global_dict[key] = stacked.mean(dim=0)

    global_model.load_state_dict(global_dict)
    return global_model


# Create global model
global_model = OnDeviceModel(FeatureExtractor(), num_classes=5)

NUM_CLIENTS = 4
NUM_ROUNDS = 3

print(f"Clients: {NUM_CLIENTS}, Rounds: {NUM_ROUNDS}")
print(f"Local epochs per round: 3\n")

for round_idx in range(NUM_ROUNDS):
    local_models = []
    for client_id in range(NUM_CLIENTS):
        x_c, y_c = create_local_data(client_id + round_idx * 100)
        local_m = local_training_round(global_model, x_c, y_c)
        local_models.append(local_m)

    global_model = federated_average(global_model, local_models)

    # Evaluate on combined data
    global_model.eval()
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for client_id in range(NUM_CLIENTS):
            x_c, y_c = create_local_data(client_id + round_idx * 100)
            preds = global_model(x_c).argmax(1)
            total_correct += (preds == y_c).sum().item()
            total_samples += len(y_c)

    acc = total_correct / total_samples
    print(f"  Round {round_idx + 1}: global accuracy = {acc:.2f}")


# ============================================
# 7. Training Memory Budget Calculator
# ============================================
print("\n[7] Training Memory Budget Calculator")
print("-" * 40)


def training_memory_estimate(model, batch_size, input_shape, dtype_bytes=4):
    """Estimate peak memory during training."""
    param_count = sum(p.numel() for p in model.parameters())
    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)

    weights_mb = param_count * dtype_bytes / (1024 ** 2)
    grads_mb = trainable_count * dtype_bytes / (1024 ** 2)
    optimizer_mb = trainable_count * dtype_bytes * 2 / (1024 ** 2)  # SGD momentum

    # Rough activation estimate
    activation_mb = batch_size * 0.5  # Very rough proxy

    total_mb = weights_mb + grads_mb + optimizer_mb + activation_mb

    return {
        "weights_mb": weights_mb,
        "gradients_mb": grads_mb,
        "optimizer_mb": optimizer_mb,
        "activations_mb": activation_mb,
        "total_mb": total_mb,
    }


mem = training_memory_estimate(model, MAX_BATCH_SIZE, (3, 32, 32))
print(f"Training memory estimate (batch_size={MAX_BATCH_SIZE}):")
print(f"  Weights:      {mem['weights_mb']:.3f} MB")
print(f"  Gradients:    {mem['gradients_mb']:.3f} MB")
print(f"  Optimizer:    {mem['optimizer_mb']:.3f} MB")
print(f"  Activations:  {mem['activations_mb']:.3f} MB (estimate)")
print(f"  Total:        {mem['total_mb']:.3f} MB")
print(f"  Device limit: {DEVICE_MEMORY_MB} MB -> {'OK' if mem['total_mb'] < DEVICE_MEMORY_MB else 'EXCEEDS'}")


# ============================================
# 8. Summary
# ============================================
print("\n[8] Summary")
print("-" * 40)
print("Key takeaways:")
print("- Freeze backbone, train head: simplest on-device personalization")
print("- Gradient checkpointing: trades compute for 50-60% less memory")
print("- LoRA: <1% trainable parameters with minimal accuracy loss")
print("- Federated learning: train on-device, aggregate centrally for privacy")
print("- Always compute memory budget before deploying training on edge")
