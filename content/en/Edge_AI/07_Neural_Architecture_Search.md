# Lesson 7: Neural Architecture Search

[Previous: Efficient Architectures](./06_Efficient_Architectures.md) | [Next: ONNX and Model Export](./08_ONNX_and_Model_Export.md)

---

## Learning Objectives

- Understand the NAS problem formulation: search space, search strategy, and evaluation
- Design search spaces with appropriate operations, connectivity patterns, and constraints
- Compare search strategies: reinforcement learning, evolutionary, one-shot, and gradient-based
- Apply hardware-aware NAS with latency and energy constraints
- Implement cost proxies to reduce NAS search time from GPU-months to GPU-hours
- Analyze landmark NAS methods: NASNet, DARTS, MnasNet, and EfficientNet-NAS

---

## 1. NAS Fundamentals

Neural Architecture Search (NAS) automates the process of designing neural network architectures. Instead of relying on human intuition to choose layer types, connections, and dimensions, NAS searches over a space of possible architectures to find optimal designs.

```
                    NAS Framework
                         │
          ┌──────────────┼──────────────┐
          │              │              │
    ┌─────▼─────┐  ┌────▼────┐   ┌────▼─────┐
    │  Search   │  │ Search  │   │Evaluation│
    │  Space    │  │Strategy │   │ Strategy │
    └───────────┘  └─────────┘   └──────────┘
    What to search  How to search  How to measure
    - Operations    - RL           - Full training
    - Connections   - Evolutionary - Weight sharing
    - Dimensions    - Gradient     - Cost proxies
                    - Random
```

### 1.1 The Three Components

```python
class NASFramework:
    """
    Conceptual NAS framework with three key components.
    """

    def __init__(self, search_space, search_strategy, evaluator):
        self.search_space = search_space       # Defines possible architectures
        self.search_strategy = search_strategy  # Explores the search space
        self.evaluator = evaluator             # Measures architecture quality

    def search(self, num_iterations=1000):
        """
        Main NAS loop:
        1. Search strategy proposes an architecture
        2. Evaluator measures its quality (accuracy, latency, etc.)
        3. Search strategy updates based on feedback
        4. Repeat until budget exhausted
        """
        best_arch = None
        best_score = 0

        for i in range(num_iterations):
            # Step 1: Sample an architecture
            architecture = self.search_strategy.propose(self.search_space)

            # Step 2: Evaluate it
            metrics = self.evaluator.evaluate(architecture)

            # Step 3: Update search strategy
            self.search_strategy.update(architecture, metrics)

            # Track best
            if metrics["score"] > best_score:
                best_score = metrics["score"]
                best_arch = architecture
                print(f"  Iteration {i}: New best score = {best_score:.4f}")

        return best_arch, best_score
```

---

## 2. Search Space Design

The search space defines what architectures can be discovered. A well-designed search space is large enough to contain good architectures but small enough to be searchable.

### 2.1 Cell-Based Search Space

Most modern NAS methods search for a **cell** (a small subgraph) that is then stacked to form the full network. This dramatically reduces the search space.

```
Cell-Based Search Space:

Search for one cell:          Stack cells to build network:
┌─────────────────┐          ┌──────────┐
│  Node 0 (input) │          │  Cell    │ ← Normal Cell
│       ↓         │          │ (stride 1)│
│  Node 1         │          ├──────────┤
│    ↓   ↓        │          │  Cell    │ ← Reduction Cell (stride 2)
│  Node 2  Node 3 │          │ (stride 2)│
│    ↓   ↓        │          ├──────────┤
│  Node 4 (output)│          │  Cell    │ ← Normal Cell
└─────────────────┘          │ (stride 1)│
                             ├──────────┤
  Edges = operations           ... × N
  (conv3x3, conv5x5,        ├──────────┤
   sep_conv, pool, skip,     │ Classifier│
   zero/none)                └──────────┘
```

```python
import torch
import torch.nn as nn
from enum import Enum
from itertools import product


class Operation(Enum):
    """Available operations in the search space."""
    CONV_3x3 = "conv_3x3"
    CONV_5x5 = "conv_5x5"
    SEP_CONV_3x3 = "sep_conv_3x3"
    SEP_CONV_5x5 = "sep_conv_5x5"
    DIL_CONV_3x3 = "dil_conv_3x3"
    MAX_POOL_3x3 = "max_pool_3x3"
    AVG_POOL_3x3 = "avg_pool_3x3"
    SKIP_CONNECT = "skip_connect"
    NONE = "none"  # No connection


def build_operation(op, channels):
    """Build a PyTorch module for a given operation."""
    if op == Operation.CONV_3x3:
        return nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels), nn.ReLU(inplace=True))
    elif op == Operation.CONV_5x5:
        return nn.Sequential(
            nn.Conv2d(channels, channels, 5, padding=2, bias=False),
            nn.BatchNorm2d(channels), nn.ReLU(inplace=True))
    elif op == Operation.SEP_CONV_3x3:
        return nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels), nn.ReLU(inplace=True))
    elif op == Operation.SEP_CONV_5x5:
        return nn.Sequential(
            nn.Conv2d(channels, channels, 5, padding=2, groups=channels, bias=False),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels), nn.ReLU(inplace=True))
    elif op == Operation.DIL_CONV_3x3:
        return nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(channels), nn.ReLU(inplace=True))
    elif op == Operation.MAX_POOL_3x3:
        return nn.MaxPool2d(3, stride=1, padding=1)
    elif op == Operation.AVG_POOL_3x3:
        return nn.AvgPool2d(3, stride=1, padding=1)
    elif op == Operation.SKIP_CONNECT:
        return nn.Identity()
    elif op == Operation.NONE:
        return None  # Zero operation
    else:
        raise ValueError(f"Unknown operation: {op}")


class SearchSpace:
    """
    Cell-based search space.

    A cell has `num_nodes` intermediate nodes. Each node receives input
    from 2 previous nodes via 2 operations (one per edge).
    """

    def __init__(self, num_nodes=4, operations=None):
        self.num_nodes = num_nodes
        self.operations = operations or list(Operation)

        # Calculate search space size
        num_ops = len(self.operations)
        # Each of the num_nodes intermediate nodes selects:
        #   - 2 input nodes (from all previous nodes)
        #   - 1 operation per input edge
        total_choices = 1
        for node in range(num_nodes):
            num_prev = node + 2  # node 0 has 2 inputs (cell inputs)
            # Choose 2 inputs from num_prev options, with operations
            choices_per_node = (num_prev * num_ops) ** 2
            total_choices *= choices_per_node

        print(f"Search space size: ~{total_choices:.1e} architectures")
        print(f"  Nodes: {num_nodes}")
        print(f"  Operations: {num_ops}")

    def random_architecture(self):
        """Sample a random architecture from the search space."""
        import random
        arch = []
        for node_idx in range(self.num_nodes):
            num_prev = node_idx + 2
            # Select 2 input indices and 2 operations
            input1 = random.randint(0, num_prev - 1)
            input2 = random.randint(0, num_prev - 1)
            op1 = random.choice(self.operations)
            op2 = random.choice(self.operations)
            arch.append((input1, op1, input2, op2))
        return arch


# Example
space = SearchSpace(num_nodes=4)
arch = space.random_architecture()
print(f"\nSampled architecture:")
for i, (in1, op1, in2, op2) in enumerate(arch):
    print(f"  Node {i+2}: input_{in1}→{op1.value}, input_{in2}→{op2.value}")
```

---

## 3. Search Strategies

### 3.1 Reinforcement Learning (NASNet)

The original NAS (Zoph & Le, 2017) uses an RNN controller trained with RL to generate architecture descriptions. The controller is rewarded based on the architecture's validation accuracy.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import random


class RLController(nn.Module):
    """
    Simplified RL-based NAS controller.

    An LSTM generates architecture decisions sequentially.
    Trained with REINFORCE policy gradient.
    """

    def __init__(self, num_operations, num_nodes=4, hidden_size=64):
        super().__init__()
        self.num_operations = num_operations
        self.num_nodes = num_nodes
        self.hidden_size = hidden_size

        self.lstm = nn.LSTMCell(hidden_size, hidden_size)

        # Separate output heads for:
        # 1. Selecting input node index
        # 2. Selecting operation
        self.node_selector = nn.Linear(hidden_size, 10)  # max 10 nodes
        self.op_selector = nn.Linear(hidden_size, num_operations)

        # Learnable initial input
        self.init_input = nn.Parameter(torch.randn(hidden_size))

    def forward(self):
        """
        Generate one architecture.
        Returns: list of decisions and their log probabilities.
        """
        decisions = []
        log_probs = []

        h = torch.zeros(1, self.hidden_size)
        c = torch.zeros(1, self.hidden_size)
        x = self.init_input.unsqueeze(0)

        for node_idx in range(self.num_nodes):
            num_prev = node_idx + 2  # Available input nodes

            # Decision 1: Select input node 1
            h, c = self.lstm(x, (h, c))
            node_logits = self.node_selector(h)[:, :num_prev]
            node_probs = F.softmax(node_logits, dim=1)
            node_dist = torch.distributions.Categorical(node_probs)
            input1 = node_dist.sample()
            log_probs.append(node_dist.log_prob(input1))

            # Decision 2: Select operation 1
            h, c = self.lstm(x, (h, c))
            op_logits = self.op_selector(h)
            op_probs = F.softmax(op_logits, dim=1)
            op_dist = torch.distributions.Categorical(op_probs)
            op1 = op_dist.sample()
            log_probs.append(op_dist.log_prob(op1))

            # Decision 3: Select input node 2
            h, c = self.lstm(x, (h, c))
            node_logits = self.node_selector(h)[:, :num_prev]
            node_probs = F.softmax(node_logits, dim=1)
            node_dist = torch.distributions.Categorical(node_probs)
            input2 = node_dist.sample()
            log_probs.append(node_dist.log_prob(input2))

            # Decision 4: Select operation 2
            h, c = self.lstm(x, (h, c))
            op_logits = self.op_selector(h)
            op_probs = F.softmax(op_logits, dim=1)
            op_dist = torch.distributions.Categorical(op_probs)
            op2 = op_dist.sample()
            log_probs.append(op_dist.log_prob(op2))

            decisions.append((input1.item(), op1.item(), input2.item(), op2.item()))

        total_log_prob = torch.stack(log_probs).sum()
        return decisions, total_log_prob


def train_controller_reinforce(controller, evaluate_fn, num_episodes=100, lr=1e-3):
    """
    Train the RL controller using REINFORCE with baseline.

    Args:
        controller: RLController instance
        evaluate_fn: Function that takes architecture → returns reward (accuracy)
        num_episodes: Number of architectures to sample and evaluate
        lr: Learning rate for the controller
    """
    optimizer = torch.optim.Adam(controller.parameters(), lr=lr)
    baseline = 0  # Moving average reward (baseline for variance reduction)
    decay = 0.95

    for episode in range(num_episodes):
        # Sample architecture
        architecture, log_prob = controller()

        # Evaluate architecture (this is the expensive part!)
        reward = evaluate_fn(architecture)

        # Update baseline
        baseline = decay * baseline + (1 - decay) * reward

        # REINFORCE loss
        advantage = reward - baseline
        loss = -log_prob * advantage  # Negative because we maximize reward

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (episode + 1) % 10 == 0:
            print(f"Episode {episode+1}: Reward={reward:.4f}, "
                  f"Baseline={baseline:.4f}")


# Example with dummy evaluation
controller = RLController(num_operations=9, num_nodes=4)
# train_controller_reinforce(controller, dummy_evaluate, num_episodes=100)
```

### 3.2 Evolutionary Search

Evolutionary NAS maintains a population of architectures and iteratively mutates the best ones.

```python
import random
import copy
from typing import List, Tuple


class EvolutionaryNAS:
    """
    Evolutionary NAS using tournament selection and mutation.

    Algorithm:
    1. Initialize population with random architectures
    2. Evaluate all architectures
    3. Select parents via tournament
    4. Mutate parents to create children
    5. Add children, remove worst
    6. Repeat
    """

    def __init__(self, search_space, population_size=50,
                 tournament_size=5, mutation_prob=0.3):
        self.search_space = search_space
        self.population_size = population_size
        self.tournament_size = tournament_size
        self.mutation_prob = mutation_prob

    def initialize_population(self):
        """Create initial random population."""
        return [
            {"arch": self.search_space.random_architecture(), "fitness": 0.0}
            for _ in range(self.population_size)
        ]

    def mutate(self, architecture):
        """
        Mutate an architecture by randomly changing one decision.
        """
        arch = copy.deepcopy(architecture)
        # Pick a random node to mutate
        node_idx = random.randint(0, len(arch) - 1)
        in1, op1, in2, op2 = arch[node_idx]
        num_prev = node_idx + 2

        # Randomly mutate one of the four decisions
        choice = random.randint(0, 3)
        if choice == 0:
            in1 = random.randint(0, num_prev - 1)
        elif choice == 1:
            op1 = random.choice(self.search_space.operations)
        elif choice == 2:
            in2 = random.randint(0, num_prev - 1)
        else:
            op2 = random.choice(self.search_space.operations)

        arch[node_idx] = (in1, op1, in2, op2)
        return arch

    def tournament_select(self, population):
        """Select the best individual from a random tournament."""
        tournament = random.sample(population, self.tournament_size)
        return max(tournament, key=lambda x: x["fitness"])

    def search(self, evaluate_fn, num_generations=100):
        """
        Run evolutionary search.

        Args:
            evaluate_fn: architecture → fitness score
            num_generations: number of evolution cycles
        """
        # Initialize
        population = self.initialize_population()
        for ind in population:
            ind["fitness"] = evaluate_fn(ind["arch"])

        best_ever = max(population, key=lambda x: x["fitness"])
        history = [best_ever["fitness"]]

        for gen in range(num_generations):
            # Select parent via tournament
            parent = self.tournament_select(population)

            # Mutate to create child
            child_arch = self.mutate(parent["arch"])
            child_fitness = evaluate_fn(child_arch)
            child = {"arch": child_arch, "fitness": child_fitness}

            # Add child, remove worst
            population.append(child)
            population.sort(key=lambda x: x["fitness"])
            population.pop(0)  # Remove worst

            # Track best
            current_best = max(population, key=lambda x: x["fitness"])
            if current_best["fitness"] > best_ever["fitness"]:
                best_ever = copy.deepcopy(current_best)

            history.append(best_ever["fitness"])

            if (gen + 1) % 20 == 0:
                print(f"Generation {gen+1}: Best fitness = {best_ever['fitness']:.4f}")

        return best_ever, history
```

### 3.3 Gradient-Based Search (DARTS)

DARTS (Liu et al., 2019) makes the search space continuous by placing a mixture of all operations on each edge, weighted by architecture parameters. Both architecture and weight parameters are optimized simultaneously.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class MixedOperation(nn.Module):
    """
    DARTS mixed operation: a weighted sum of all candidate operations.

    During search, all operations run in parallel and their outputs
    are combined using softmax-weighted architecture parameters.

    After search, the operation with the highest weight is selected.
    """

    def __init__(self, channels, operations_list):
        super().__init__()
        self.ops = nn.ModuleList()
        for op in operations_list:
            self.ops.append(build_operation(op, channels) or ZeroOp())

        # Architecture parameters (learnable)
        self.alpha = nn.Parameter(torch.zeros(len(operations_list)))

    def forward(self, x):
        """Weighted sum of all operation outputs."""
        weights = F.softmax(self.alpha, dim=0)
        out = sum(w * op(x) for w, op in zip(weights, self.ops) if op is not None)
        return out

    def selected_operation(self):
        """Return the operation with highest weight (for final architecture)."""
        idx = self.alpha.argmax().item()
        return idx


class ZeroOp(nn.Module):
    """Zero operation: outputs zeros."""
    def forward(self, x):
        return torch.zeros_like(x)


class DARTSCell(nn.Module):
    """
    DARTS differentiable cell.

    Each edge has a MixedOperation. The architecture parameters
    (alpha for each edge) are optimized jointly with model weights.
    """

    def __init__(self, channels, num_nodes=4, operations=None):
        super().__init__()
        self.num_nodes = num_nodes
        ops = operations or [op for op in Operation if op != Operation.NONE]

        # Create mixed operations for all possible edges
        self.edges = nn.ModuleDict()
        for node_idx in range(num_nodes):
            for input_idx in range(node_idx + 2):  # +2 for cell inputs
                key = f"node{node_idx}_input{input_idx}"
                self.edges[key] = MixedOperation(channels, ops)

    def forward(self, x0, x1):
        """
        Forward pass through the cell.
        x0, x1 are the two cell inputs.
        """
        states = [x0, x1]

        for node_idx in range(self.num_nodes):
            node_input = 0
            for input_idx in range(node_idx + 2):
                key = f"node{node_idx}_input{input_idx}"
                node_input = node_input + self.edges[key](states[input_idx])
            states.append(node_input)

        # Concatenate all intermediate node outputs
        return torch.cat(states[2:], dim=1)

    def arch_parameters(self):
        """Return architecture parameters (alpha values)."""
        return [edge.alpha for edge in self.edges.values()]


def darts_bilevel_optimization(cell, train_loader, val_loader,
                                channels, epochs=50, lr_w=0.025, lr_a=3e-4):
    """
    DARTS bi-level optimization.

    Alternates between:
    1. Update weights w on training data (fixing architecture alpha)
    2. Update architecture alpha on validation data (fixing weights w)

    This is a simplified version — full DARTS uses second-order approximation.
    """
    # Weight optimizer
    weight_params = [p for n, p in cell.named_parameters() if "alpha" not in n]
    w_optimizer = torch.optim.SGD(weight_params, lr=lr_w, momentum=0.9,
                                   weight_decay=3e-4)

    # Architecture optimizer
    arch_params = cell.arch_parameters()
    a_optimizer = torch.optim.Adam(arch_params, lr=lr_a, weight_decay=1e-3)

    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        cell.train()

        for (train_input, train_target), (val_input, val_target) in zip(
            train_loader, val_loader
        ):
            # Step 1: Update architecture on validation data
            a_optimizer.zero_grad()
            # (Simplified: in full DARTS, this uses a second-order approximation)
            val_pred = cell(val_input, val_input)
            # In practice, add a classifier head here
            # val_loss = criterion(classifier(val_pred), val_target)
            # val_loss.backward()
            # a_optimizer.step()

            # Step 2: Update weights on training data
            w_optimizer.zero_grad()
            train_pred = cell(train_input, train_input)
            # train_loss = criterion(classifier(train_pred), train_target)
            # train_loss.backward()
            # w_optimizer.step()

    # Extract final architecture
    print("\n=== Discovered Architecture ===")
    for key, edge in cell.edges.items():
        weights = F.softmax(edge.alpha, dim=0)
        best_idx = edge.alpha.argmax().item()
        ops = [op for op in Operation if op != Operation.NONE]
        print(f"  {key}: {ops[best_idx].value} (weight: {weights[best_idx]:.3f})")

    return cell
```

---

## 4. Hardware-Aware NAS

Standard NAS optimizes for accuracy only. **Hardware-aware NAS** adds hardware constraints (latency, energy, memory) directly into the search objective.

### 4.1 Multi-Objective Search

```python
import torch


class HardwareAwareNAS:
    """
    Hardware-aware NAS with multi-objective optimization.

    Objective: maximize accuracy while satisfying hardware constraints.

    Approach 1: Constrained optimization
      maximize accuracy(arch) subject to latency(arch) ≤ target

    Approach 2: Weighted objective
      maximize accuracy(arch) - lambda * latency(arch)

    Approach 3: Pareto optimization
      Find the Pareto frontier of accuracy vs latency
    """

    def __init__(self, search_space, latency_model, accuracy_predictor):
        self.search_space = search_space
        self.latency_model = latency_model
        self.accuracy_predictor = accuracy_predictor

    def weighted_objective(self, architecture, lambda_lat=0.1):
        """
        Combined objective: accuracy - lambda * log(latency).

        Using log(latency) ensures the penalty is proportional
        rather than absolute.
        """
        import math
        accuracy = self.accuracy_predictor(architecture)
        latency = self.latency_model(architecture)
        return accuracy - lambda_lat * math.log(latency)

    def constrained_search(self, target_latency_ms, num_samples=1000):
        """
        Sample architectures and keep only those meeting the latency constraint.
        Among valid architectures, return the most accurate.
        """
        valid_architectures = []

        for _ in range(num_samples):
            arch = self.search_space.random_architecture()
            latency = self.latency_model(arch)

            if latency <= target_latency_ms:
                accuracy = self.accuracy_predictor(arch)
                valid_architectures.append({
                    "arch": arch,
                    "accuracy": accuracy,
                    "latency": latency,
                })

        if not valid_architectures:
            print(f"No architecture found with latency ≤ {target_latency_ms}ms")
            return None

        # Return the best valid architecture
        best = max(valid_architectures, key=lambda x: x["accuracy"])
        print(f"Best architecture: accuracy={best['accuracy']:.2f}%, "
              f"latency={best['latency']:.1f}ms")
        print(f"  ({len(valid_architectures)} valid out of {num_samples} sampled)")
        return best
```

### 4.2 Latency Prediction Models

Training models to completion for every candidate is too expensive. Instead, build a **latency predictor** that estimates inference time from the architecture description.

```python
import torch
import torch.nn as nn
import time


class LatencyLookupTable:
    """
    Build a lookup table mapping operations to measured latency on target hardware.

    This is the approach used by MnasNet and FBNet:
    1. Measure latency of each operation type at each feature map size
    2. Total model latency ≈ sum of per-operation latencies
    """

    def __init__(self, device="cpu"):
        self.device = device
        self.table = {}

    def build_table(self, channels_list, spatial_sizes, operations):
        """
        Measure latency of each operation at different sizes.
        """
        print("Building latency lookup table...")

        for channels in channels_list:
            for spatial in spatial_sizes:
                for op in operations:
                    module = build_operation(op, channels)
                    if module is None:
                        self.table[(channels, spatial, op)] = 0.0
                        continue

                    module = module.to(self.device).eval()
                    x = torch.randn(1, channels, spatial, spatial, device=self.device)

                    # Warmup
                    with torch.no_grad():
                        for _ in range(10):
                            module(x)

                    # Measure
                    times = []
                    with torch.no_grad():
                        for _ in range(50):
                            start = time.perf_counter()
                            module(x)
                            times.append((time.perf_counter() - start) * 1000)

                    avg_ms = sum(times) / len(times)
                    self.table[(channels, spatial, op)] = avg_ms

        print(f"Measured {len(self.table)} operation-size combinations")

    def predict_latency(self, architecture, layer_configs):
        """
        Predict total model latency by summing per-layer latencies.

        Args:
            architecture: list of (input, op, input, op) tuples
            layer_configs: list of (channels, spatial_size) for each layer

        Returns:
            Estimated total latency in milliseconds
        """
        total_ms = 0
        for (in1, op1, in2, op2), (channels, spatial) in zip(architecture, layer_configs):
            key1 = (channels, spatial, op1)
            key2 = (channels, spatial, op2)
            total_ms += self.table.get(key1, 0.0) + self.table.get(key2, 0.0)
        return total_ms


class LearnedLatencyPredictor(nn.Module):
    """
    Neural network that predicts architecture latency from its encoding.

    Trained on a dataset of (architecture_encoding, measured_latency) pairs.
    More flexible than lookup tables — can capture cross-layer interactions.
    """

    def __init__(self, encoding_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(encoding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Softplus(),  # Ensure positive output
        )

    def forward(self, arch_encoding):
        """
        Args:
            arch_encoding: tensor of shape (batch, encoding_dim)
        Returns:
            predicted latency (batch, 1) in milliseconds
        """
        return self.net(arch_encoding)

    def train_predictor(self, encodings, latencies, epochs=100, lr=1e-3):
        """Train the latency predictor on measured data."""
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        criterion = nn.MSELoss()

        for epoch in range(epochs):
            pred = self.forward(encodings)
            loss = criterion(pred.squeeze(), latencies)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 20 == 0:
                mape = ((pred.squeeze() - latencies).abs() / latencies).mean() * 100
                print(f"Epoch {epoch+1}: MSE={loss.item():.4f}, MAPE={mape:.1f}%")
```

---

## 5. Cost Proxies and Efficient Evaluation

The most expensive part of NAS is evaluating each candidate architecture. **Cost proxies** dramatically reduce evaluation time.

### 5.1 Common Cost Proxies

| Proxy | Speed | Accuracy | Description |
|-------|-------|----------|-------------|
| **Full training** | Slowest (hours) | Best | Train to convergence, evaluate |
| **Reduced training** | Fast (minutes) | Good | Train for fewer epochs, smaller dataset |
| **Weight sharing** | Very fast (seconds) | Moderate | All architectures share weights in a supernet |
| **Zero-shot proxies** | Instant | Rough | Score based on initialization (no training) |

```python
import torch
import torch.nn as nn


def zero_shot_proxy_synflow(model, input_shape=(1, 3, 32, 32)):
    """
    SynFlow (Tanaka et al., 2020): a training-free proxy for architecture quality.

    Computes the sum of products of all parameters (measures network "flow").
    Higher SynFlow score correlates with better final accuracy.

    Key insight: the ease with which signal flows through the network
    (at initialization) predicts trainability.
    """
    # Set all parameters to their absolute values
    signs = {}
    for name, param in model.named_parameters():
        signs[name] = torch.sign(param.data)
        param.data.abs_()

    # Forward pass with all-ones input
    model.eval()
    x = torch.ones(*input_shape)
    output = model(x)

    # Backward pass
    loss = output.sum()
    loss.backward()

    # SynFlow score = sum of (|param| * |gradient|)
    score = 0
    for name, param in model.named_parameters():
        if param.grad is not None:
            score += (param.data * param.grad.abs()).sum().item()

    # Restore signs
    for name, param in model.named_parameters():
        param.data *= signs[name]

    return score


def zero_shot_proxy_naswot(model, input_shape=(1, 3, 32, 32), num_samples=64):
    """
    NASWOT (Mellor et al., 2021): Neural Architecture Search Without Training.

    Measures the overlap of linear regions in the ReLU network.
    Less overlap (more unique activations) = better architecture.
    Uses the log-determinant of the kernel matrix as a score.
    """
    model.eval()

    # Collect binary activation patterns (which ReLUs are active)
    activation_patterns = []

    def hook_fn(module, input, output):
        activation_patterns.append((output > 0).float().view(output.size(0), -1))

    hooks = []
    for module in model.modules():
        if isinstance(module, nn.ReLU):
            hooks.append(module.register_forward_hook(hook_fn))

    with torch.no_grad():
        x = torch.randn(num_samples, *input_shape[1:])
        model(x)

    for h in hooks:
        h.remove()

    if not activation_patterns:
        return 0

    # Concatenate all activation patterns
    patterns = torch.cat(activation_patterns, dim=1)  # (num_samples, total_neurons)

    # Compute kernel matrix K = patterns @ patterns.T
    K = patterns @ patterns.T
    K = K / K.max()  # Normalize

    # Score = log-determinant of K (higher = more diverse patterns)
    # Use eigenvalue decomposition for numerical stability
    eigenvalues = torch.linalg.eigvalsh(K)
    eigenvalues = eigenvalues.clamp(min=1e-6)
    score = eigenvalues.log().sum().item()

    return score


# Example: compare two architectures using zero-shot proxies
model_a = nn.Sequential(
    nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
    nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
    nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, 10),
)

model_b = nn.Sequential(
    nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(),
    nn.Conv2d(16, 16, 3, padding=1), nn.ReLU(),
    nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(16, 10),
)

score_a = zero_shot_proxy_naswot(model_a)
score_b = zero_shot_proxy_naswot(model_b)
print(f"Model A NASWOT score: {score_a:.2f}")
print(f"Model B NASWOT score: {score_b:.2f}")
print(f"Model A is {'better' if score_a > score_b else 'worse'} (higher = better)")
```

---

## 6. Landmark NAS Methods

### 6.1 MnasNet: Platform-Aware NAS

MnasNet (Tan et al., 2019) was the first NAS method to directly optimize for mobile latency, producing the basis for EfficientNet.

```python
def mnasnet_objective(accuracy, latency, target_latency, w=-0.07):
    """
    MnasNet's multi-objective reward function.

    R(arch) = accuracy(arch) × [latency(arch) / target]^w

    When w < 0:
    - If latency < target: bonus (ratio < 1, raised to negative power > 1)
    - If latency > target: penalty (ratio > 1, raised to negative power < 1)

    w = -0.07 was found to give a good accuracy-latency tradeoff.
    """
    ratio = latency / target_latency
    return accuracy * (ratio ** w)


# Compare architectures with the MnasNet objective
target = 75  # Target: 75ms latency

print(f"{'Model':<20} {'Acc':>6} {'Lat':>6} {'MnasNet Reward':>15}")
print("-" * 50)
for name, acc, lat in [
    ("Accurate but slow", 78.0, 120),
    ("Balanced",          76.5,  75),
    ("Fast but less acc", 74.0,  45),
    ("Very fast",         70.0,  30),
]:
    reward = mnasnet_objective(acc, lat, target)
    print(f"{name:<20} {acc:>5.1f}% {lat:>5}ms {reward:>14.2f}")
```

### 6.2 Evolution of NAS Methods

| Method | Year | Search Cost | Strategy | Key Innovation |
|--------|------|-------------|----------|----------------|
| **NASNet** | 2017 | 48,000 GPU-hours | RL | Cell-based search, transferable cells |
| **AmoebaNet** | 2019 | 7,200 GPU-hours | Evolutionary | Tournament selection, regularized evolution |
| **DARTS** | 2019 | 4 GPU-days | Gradient | Continuous relaxation, bi-level optimization |
| **ProxylessNAS** | 2019 | 8 GPU-hours | Gradient + HW | Direct latency optimization, path binarization |
| **MnasNet** | 2019 | 40,000 GPU-hours | RL + HW | Latency in reward, factorized search space |
| **FBNet** | 2019 | 9 GPU-hours | Gradient + HW | Differentiable + lookup table latency |
| **OFA** | 2020 | Train once | Supernet | Once-for-all elastic network |
| **TF-NAS** | 2021 | <1 GPU-hour | Zero-shot | Training-free proxies |

---

## 7. Practical NAS: Once-for-All (OFA)

OFA (Cai et al., 2020) trains a single **supernet** once, then extracts specialized subnets for different hardware targets without any additional training.

```python
import torch
import torch.nn as nn
import random


class ElasticConv(nn.Module):
    """
    Elastic convolution that supports multiple kernel sizes.

    The largest kernel contains all smaller kernels as subsets.
    At search time, different kernel sizes can be selected.
    """

    def __init__(self, in_channels, out_channels, max_kernel=7):
        super().__init__()
        self.max_kernel = max_kernel
        self.conv = nn.Conv2d(
            in_channels, out_channels, max_kernel,
            padding=max_kernel // 2, bias=False,
        )

    def forward(self, x, kernel_size=None):
        if kernel_size is None or kernel_size == self.max_kernel:
            return self.conv(x)

        # Extract the center kernel_size × kernel_size portion
        center = self.max_kernel // 2
        start = center - kernel_size // 2
        end = start + kernel_size

        weight = self.conv.weight[:, :, start:end, start:end]
        padding = kernel_size // 2
        return nn.functional.conv2d(x, weight, padding=padding)


class ElasticBlock(nn.Module):
    """
    Elastic block supporting variable depth, width, and kernel size.

    During supernet training, randomly sample configurations.
    During search, evaluate specific configurations.
    """

    def __init__(self, max_channels, min_channels=None, max_kernel=7):
        super().__init__()
        self.max_channels = max_channels
        self.min_channels = min_channels or max_channels // 4
        self.elastic_conv = ElasticConv(max_channels, max_channels, max_kernel)
        self.bn = nn.BatchNorm2d(max_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, config=None):
        """
        Args:
            x: input tensor
            config: dict with 'channels' and 'kernel_size'
                    If None, use full (max) configuration
        """
        kernel_size = config.get("kernel_size", 7) if config else 7
        channels = config.get("channels", self.max_channels) if config else self.max_channels

        out = self.elastic_conv(x, kernel_size=kernel_size)
        out = self.bn(out)
        out = self.relu(out)

        # Channel slicing: use only first `channels` channels
        if channels < self.max_channels:
            out = out[:, :channels, :, :]

        return out

    def sample_config(self):
        """Sample a random sub-network configuration."""
        channel_choices = list(range(self.min_channels, self.max_channels + 1, 8))
        kernel_choices = [3, 5, 7]
        return {
            "channels": random.choice(channel_choices),
            "kernel_size": random.choice(kernel_choices),
        }


class OFATrainer:
    """
    Once-for-All training strategy.

    Progressive shrinking:
    1. Train the largest network first
    2. Progressively shrink kernel size (7→5→3)
    3. Progressively shrink depth (remove layers)
    4. Progressively shrink width (fewer channels)

    After training, any subnet can be extracted without retraining.
    """

    def __init__(self, supernet):
        self.supernet = supernet

    def train_phase(self, train_loader, phase, epochs=25):
        """
        Train one phase of progressive shrinking.

        Args:
            phase: "elastic_kernel", "elastic_depth", or "elastic_width"
        """
        print(f"\n=== Training Phase: {phase} ({epochs} epochs) ===")

        if phase == "elastic_kernel":
            sample_fn = lambda: {"kernel_size": random.choice([3, 5, 7])}
        elif phase == "elastic_depth":
            sample_fn = lambda: {
                "kernel_size": random.choice([3, 5, 7]),
                "depth": random.randint(2, 4),
            }
        elif phase == "elastic_width":
            sample_fn = lambda: {
                "kernel_size": random.choice([3, 5, 7]),
                "depth": random.randint(2, 4),
                "channels": random.choice([16, 24, 32, 48, 64]),
            }
        else:
            raise ValueError(f"Unknown phase: {phase}")

        # During each training step, sample a random subnet and train it
        # The supernet weights are shared across all subnets
        for epoch in range(epochs):
            config = sample_fn()
            # self.supernet.set_active_subnet(config)
            # ... standard training step ...
            pass

        print(f"Phase {phase} complete")

    def search_subnet(self, target_latency, latency_predictor, num_samples=1000):
        """
        Search for the best subnet meeting the latency constraint.
        No training needed — just evaluate accuracy with shared weights.
        """
        best = None
        for _ in range(num_samples):
            config = self._random_config()
            latency = latency_predictor(config)
            if latency <= target_latency:
                accuracy = self._evaluate_subnet(config)
                if best is None or accuracy > best["accuracy"]:
                    best = {"config": config, "accuracy": accuracy, "latency": latency}
        return best

    def _random_config(self):
        return {
            "kernel_size": random.choice([3, 5, 7]),
            "depth": random.randint(2, 4),
            "channels": random.choice([16, 24, 32, 48, 64]),
        }

    def _evaluate_subnet(self, config):
        # In practice, extract and evaluate the subnet
        return random.uniform(70, 80)  # Placeholder
```

---

## Summary

| Concept | Key Takeaway |
|---------|-------------|
| **NAS Framework** | Search space + search strategy + evaluation strategy |
| **Cell-based search** | Search for a small cell, stack it to form the network |
| **RL-based NAS** | LSTM controller generates architectures, trained with REINFORCE |
| **Evolutionary NAS** | Population of architectures, mutate best, remove worst |
| **DARTS** | Continuous relaxation with softmax weights, gradient-based |
| **Hardware-aware** | Add latency/energy/memory constraints to the search objective |
| **Latency models** | Lookup tables or learned predictors for fast latency estimation |
| **Cost proxies** | Zero-shot, weight sharing, reduced training to speed up evaluation |
| **OFA** | Train one supernet, extract subnets for any hardware target |
| **MnasNet** | Pioneered latency-aware NAS, foundation for EfficientNet |

---

## Exercises

### Exercise 1: Search Space Analysis

1. Implement a cell-based search space with 4 nodes and 8 operation types
2. Calculate the total number of possible architectures
3. Sample 100 random architectures and compute their parameter counts
4. Plot the distribution of parameter counts. Is it uniform?

### Exercise 2: Evolutionary NAS on a Toy Problem

1. Define a simple search space: 5 layers, each choosing between Conv3x3, Conv5x5, SepConv3x3
2. Implement evolutionary search with population=20, tournament_size=3
3. Use a cheap evaluation proxy (e.g., train for 5 epochs on CIFAR-10)
4. Run for 50 generations and plot best fitness over time
5. Compare with random search (same number of evaluations)

### Exercise 3: DARTS Implementation

1. Implement a simplified DARTS cell with 2 nodes and 4 operation types
2. Train on CIFAR-10 using the bi-level optimization (alternate weight and architecture updates)
3. After training, extract the discrete architecture (argmax of architecture parameters)
4. Retrain the discrete architecture from scratch. Does it match the search-phase accuracy?

### Exercise 4: Hardware-Aware Objective

1. Build a latency lookup table for Conv3x3, Conv5x5, SepConv3x3 at various channel counts
2. Define the MnasNet reward function with target latency = 50ms
3. Run random search (1000 samples) with: (a) accuracy only, (b) MnasNet reward
4. Compare the Pareto frontiers of both approaches

### Exercise 5: Zero-Shot NAS Proxy

1. Implement the NASWOT (activation pattern diversity) proxy
2. Generate 50 random architectures and compute their NASWOT scores
3. Train the top-5 and bottom-5 architectures on CIFAR-10 for 50 epochs
4. Plot NASWOT score vs final accuracy. How well does the proxy correlate?

---

[Previous: Efficient Architectures](./06_Efficient_Architectures.md) | [Overview](./00_Overview.md) | [Next: ONNX and Model Export](./08_ONNX_and_Model_Export.md)

**License**: CC BY-NC 4.0
