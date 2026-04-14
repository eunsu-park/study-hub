"""
nn.Module - Examples
====================
Lesson 05: nn.Module

Demonstrates:
  1. Defining custom modules
  2. Parameters and buffers
  3. nn.Sequential, ModuleList, ModuleDict
  4. Model inspection (parameters, state_dict)
  5. Weight initialization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def example_1_custom_module():
    """Define and use a custom nn.Module."""
    print("=" * 60)
    print("Example 1: Custom Module")
    print("=" * 60)

    class MLP(nn.Module):
        def __init__(self, input_dim, hidden_dim, output_dim):
            super().__init__()
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, output_dim)

        def forward(self, x):
            x = F.relu(self.fc1(x))
            return self.fc2(x)

    model = MLP(784, 128, 10)
    x = torch.randn(4, 784)
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Model:\n{model}")


def example_2_parameters_buffers():
    """Parameters vs buffers vs plain attributes."""
    print("\n" + "=" * 60)
    print("Example 2: Parameters and Buffers")
    print("=" * 60)

    class MyLayer(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(dim))
            self.register_buffer('running_mean', torch.zeros(dim))
            self.config = {'dim': dim}  # plain attribute

        def forward(self, x):
            return x * self.weight + self.running_mean

    layer = MyLayer(5)
    print(f"Parameters: {list(layer.named_parameters())}")
    print(f"Buffers: {list(layer.named_buffers())}")
    print(f"State dict keys: {list(layer.state_dict().keys())}")


def example_3_composition():
    """Module composition with Sequential, ModuleList, ModuleDict."""
    print("\n" + "=" * 60)
    print("Example 3: Module Composition")
    print("=" * 60)

    # Sequential
    seq_model = nn.Sequential(
        nn.Linear(784, 256), nn.ReLU(), nn.Dropout(0.3),
        nn.Linear(256, 10),
    )
    print(f"Sequential output: {seq_model(torch.randn(2, 784)).shape}")

    # ModuleList
    class MultiHead(nn.Module):
        def __init__(self, n_heads):
            super().__init__()
            self.shared = nn.Linear(784, 128)
            self.heads = nn.ModuleList([nn.Linear(128, 10)
                                        for _ in range(n_heads)])

        def forward(self, x):
            x = F.relu(self.shared(x))
            return [head(x) for head in self.heads]

    mh = MultiHead(3)
    outputs = mh(torch.randn(2, 784))
    print(f"MultiHead outputs: {len(outputs)} heads, "
          f"each shape {outputs[0].shape}")

    # All parameters are tracked
    total = sum(p.numel() for p in mh.parameters())
    print(f"Total parameters: {total:,}")


def example_4_inspection():
    """Inspect model parameters and architecture."""
    print("\n" + "=" * 60)
    print("Example 4: Model Inspection")
    print("=" * 60)

    class ResBlock(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim),
            )

        def forward(self, x):
            return x + self.net(x)

    model = nn.Sequential(
        nn.Linear(100, 64),
        ResBlock(64),
        ResBlock(64),
        nn.Linear(64, 10),
    )

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters()
                    if p.requires_grad)
    print(f"Total params: {total:,}")
    print(f"Trainable params: {trainable:,}")

    print("\nNamed parameters:")
    for name, p in model.named_parameters():
        print(f"  {name}: {p.shape}")


def example_5_init():
    """Custom weight initialization."""
    print("\n" + "=" * 60)
    print("Example 5: Weight Initialization")
    print("=" * 60)

    model = nn.Sequential(nn.Linear(10, 20), nn.Linear(20, 5))

    # Before init
    print(f"Before init - fc1 weight mean: "
          f"{model[0].weight.data.mean():.4f}")

    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

    model.apply(init_weights)

    print(f"After Xavier init - fc1 weight mean: "
          f"{model[0].weight.data.mean():.4f}")
    print(f"After Xavier init - fc1 bias: {model[0].bias.data}")


if __name__ == "__main__":
    example_1_custom_module()
    example_2_parameters_buffers()
    example_3_composition()
    example_4_inspection()
    example_5_init()
