"""
TorchScript and Deployment - Examples
=====================================
Lesson 13: TorchScript and Deployment

Demonstrates:
  1. torch.jit.trace for simple models
  2. torch.jit.script for models with control flow
  3. torch.compile basics
  4. Dynamic quantization
"""

import torch
import torch.nn as nn
import tempfile
import os


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


class DynamicModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 5)

    def forward(self, x):
        if x.sum() > 0:
            return torch.relu(self.fc(x))
        else:
            return torch.sigmoid(self.fc(x))


def example_1_tracing():
    """Convert model to TorchScript via tracing."""
    print("=" * 60)
    print("Example 1: TorchScript Tracing")
    print("=" * 60)

    model = SimpleModel()
    model.eval()

    example_input = torch.randn(1, 784)
    traced = torch.jit.trace(model, example_input)

    # Verify outputs match
    with torch.no_grad():
        orig_out = model(example_input)
        traced_out = traced(example_input)
        print(f"Outputs match: {torch.allclose(orig_out, traced_out)}")

    # Save and reload
    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
        traced.save(f.name)
        size = os.path.getsize(f.name) / 1024
        print(f"Saved traced model: {size:.1f} KB")

        loaded = torch.jit.load(f.name)
        loaded_out = loaded(example_input)
        print(f"Reloaded outputs match: "
              f"{torch.allclose(orig_out, loaded_out)}")
        os.unlink(f.name)


def example_2_scripting():
    """Convert model with control flow via scripting."""
    print("\n" + "=" * 60)
    print("Example 2: TorchScript Scripting")
    print("=" * 60)

    model = DynamicModel()
    model.eval()

    scripted = torch.jit.script(model)

    pos_input = torch.ones(1, 10)
    neg_input = -torch.ones(1, 10)

    with torch.no_grad():
        pos_out = scripted(pos_input)
        neg_out = scripted(neg_input)

    print(f"Positive input -> uses relu: min={pos_out.min():.4f} >= 0")
    print(f"Negative input -> uses sigmoid: "
          f"range=[{neg_out.min():.4f}, {neg_out.max():.4f}] in (0,1)")


def example_3_compile():
    """torch.compile for runtime optimization."""
    print("\n" + "=" * 60)
    print("Example 3: torch.compile")
    print("=" * 60)

    model = SimpleModel()
    model.eval()

    try:
        compiled = torch.compile(model)
        x = torch.randn(32, 784)
        with torch.no_grad():
            output = compiled(x)
        print(f"torch.compile output shape: {output.shape}")
        print("torch.compile available and working.")
    except Exception as e:
        print(f"torch.compile not available: {e}")


def example_4_quantization():
    """Dynamic quantization to reduce model size."""
    print("\n" + "=" * 60)
    print("Example 4: Dynamic Quantization")
    print("=" * 60)

    model = nn.Sequential(
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 10),
    )
    model.eval()

    quantized = torch.ao.quantization.quantize_dynamic(
        model, {nn.Linear}, dtype=torch.qint8
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        fp32_path = os.path.join(tmpdir, 'fp32.pt')
        int8_path = os.path.join(tmpdir, 'int8.pt')

        torch.save(model.state_dict(), fp32_path)
        torch.save(quantized.state_dict(), int8_path)

        fp32_size = os.path.getsize(fp32_path) / 1024
        int8_size = os.path.getsize(int8_path) / 1024

        print(f"FP32 model: {fp32_size:.1f} KB")
        print(f"INT8 model: {int8_size:.1f} KB")
        print(f"Compression: {fp32_size/int8_size:.1f}x")

    # Verify outputs are close
    x = torch.randn(1, 784)
    with torch.no_grad():
        fp32_out = model(x)
        int8_out = quantized(x)
        diff = (fp32_out - int8_out).abs().max().item()
        print(f"Max output difference: {diff:.6f}")


if __name__ == "__main__":
    example_1_tracing()
    example_2_scripting()
    example_3_compile()
    example_4_quantization()
