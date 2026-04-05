"""
10. PyTorch Mobile and ExecuTorch

Demonstrates exporting PyTorch models for mobile and edge deployment
using TorchScript and the ExecuTorch-style workflow.

Covers:
- TorchScript tracing and scripting
- Optimizing models for mobile (torch.utils.mobile_optimizer)
- Simulated ExecuTorch export pipeline
- Model size and compatibility checks
- Operator compatibility analysis

Requirements:
    pip install torch torchvision
"""

import torch
import torch.nn as nn
import os
import tempfile
import time

print("=" * 60)
print("Edge AI — PyTorch Mobile and ExecuTorch")
print("=" * 60)


# ============================================
# 1. Define a MobileNet-style Model
# ============================================
print("\n[1] Define Model for Mobile Deployment")
print("-" * 40)


class DepthwiseSeparableConv(nn.Module):
    """Depthwise separable convolution block."""

    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, 3, stride=stride,
                                   padding=1, groups=in_ch, bias=False)
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.pointwise = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU6(inplace=True)

    def forward(self, x):
        x = self.relu(self.bn1(self.depthwise(x)))
        x = self.relu(self.bn2(self.pointwise(x)))
        return x


class MiniMobileNet(nn.Module):
    """Minimal MobileNet-like model for demonstration."""

    def __init__(self, num_classes=10):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU6(inplace=True),
        )
        self.features = nn.Sequential(
            DepthwiseSeparableConv(16, 32, stride=1),
            DepthwiseSeparableConv(32, 64, stride=2),
            DepthwiseSeparableConv(64, 64, stride=1),
            DepthwiseSeparableConv(64, 128, stride=2),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.features(x)
        x = self.head(x)
        return x


model = MiniMobileNet()
model.eval()
params = sum(p.numel() for p in model.parameters())
print(f"MiniMobileNet parameters: {params:,}")

dummy_input = torch.randn(1, 3, 224, 224)
with torch.no_grad():
    output = model(dummy_input)
print(f"Input shape:  {dummy_input.shape}")
print(f"Output shape: {output.shape}")


# ============================================
# 2. TorchScript Tracing
# ============================================
print("\n[2] TorchScript Tracing")
print("-" * 40)
print("Tracing records operations by running the model with sample input.")
print("Good for models with no data-dependent control flow.\n")

traced_model = torch.jit.trace(model, dummy_input)

# Verify traced output matches original
with torch.no_grad():
    traced_out = traced_model(dummy_input)
    diff = (output - traced_out).abs().max().item()
print(f"Max difference (original vs traced): {diff:.8f}")

# Save traced model
traced_path = os.path.join(tempfile.gettempdir(), "mini_mobilenet_traced.pt")
traced_model.save(traced_path)
traced_size = os.path.getsize(traced_path) / (1024 * 1024)
print(f"Traced model size: {traced_size:.2f} MB")


# ============================================
# 3. TorchScript Scripting
# ============================================
print("\n[3] TorchScript Scripting")
print("-" * 40)
print("Scripting compiles the Python code to TorchScript IR.")
print("Supports data-dependent control flow (if/for/while).\n")


class ConditionalModel(nn.Module):
    """Model with data-dependent branching (requires scripting, not tracing)."""

    def __init__(self):
        super().__init__()
        self.conv_small = nn.Conv2d(3, 16, 3, padding=1)
        self.conv_large = nn.Conv2d(3, 16, 5, padding=2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(16, 10)

    def forward(self, x):
        # Data-dependent branch
        if x.mean() > 0:
            x = self.conv_small(x)
        else:
            x = self.conv_large(x)
        x = torch.relu(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)


cond_model = ConditionalModel()
cond_model.eval()

scripted_model = torch.jit.script(cond_model)
print("Scripted ConditionalModel successfully")
print(f"Graph has {len(list(scripted_model.graph.nodes()))} nodes")

scripted_path = os.path.join(tempfile.gettempdir(), "conditional_scripted.pt")
scripted_model.save(scripted_path)
scripted_size = os.path.getsize(scripted_path) / (1024 * 1024)
print(f"Scripted model size: {scripted_size:.2f} MB")


# ============================================
# 4. Mobile Optimization
# ============================================
print("\n[4] Mobile Optimization")
print("-" * 40)
print("torch.utils.mobile_optimizer fuses ops and removes dropout")
print("for efficient on-device inference.\n")

try:
    from torch.utils.mobile_optimizer import optimize_for_mobile

    optimized = optimize_for_mobile(traced_model)
    opt_path = os.path.join(tempfile.gettempdir(), "mini_mobilenet_optimized.ptl")
    optimized._save_for_lite_interpreter(opt_path)
    opt_size = os.path.getsize(opt_path) / (1024 * 1024)
    print(f"Original traced size:  {traced_size:.2f} MB")
    print(f"Mobile-optimized size: {opt_size:.2f} MB")
    print(f"Size reduction: {(1 - opt_size / traced_size) * 100:.1f}%")

    # Verify output
    with torch.no_grad():
        opt_out = optimized(dummy_input)
        diff = (output - opt_out).abs().max().item()
    print(f"Max difference (original vs optimized): {diff:.8f}")
except Exception as e:
    print(f"Mobile optimization not available: {e}")
    print("(Requires PyTorch built with mobile support)")


# ============================================
# 5. Operator Compatibility Check
# ============================================
print("\n[5] Operator Compatibility Analysis")
print("-" * 40)
print("Check which operators are used and whether they are supported")
print("on target mobile/edge runtimes.\n")

# Extract operators from traced model
ops_used = set()
for node in traced_model.graph.nodes():
    if node.kind().startswith("aten::") or node.kind().startswith("prim::"):
        ops_used.add(node.kind())

# Simulated mobile-compatible operator list
MOBILE_SUPPORTED_OPS = {
    "aten::conv2d", "aten::batch_norm", "aten::relu_", "aten::relu",
    "aten::max_pool2d", "aten::adaptive_avg_pool2d", "aten::linear",
    "aten::flatten", "aten::add", "aten::mul", "aten::hardtanh_",
    "aten::size", "aten::reshape", "aten::contiguous",
    "prim::Constant", "prim::ListConstruct", "prim::TupleConstruct",
    "aten::_convolution", "aten::t",
}

aten_ops = {op for op in ops_used if op.startswith("aten::")}
supported = aten_ops & MOBILE_SUPPORTED_OPS
unsupported = aten_ops - MOBILE_SUPPORTED_OPS

print(f"Total ATen operators used: {len(aten_ops)}")
print(f"Supported on mobile:      {len(supported)}")
print(f"Potentially unsupported:  {len(unsupported)}")

if unsupported:
    print("\nOperators needing custom kernels or replacement:")
    for op in sorted(unsupported):
        print(f"  - {op}")


# ============================================
# 6. ExecuTorch-Style Export Simulation
# ============================================
print("\n[6] ExecuTorch-Style Export Pipeline (Simulated)")
print("-" * 40)
print("ExecuTorch is PyTorch's next-gen on-device runtime.")
print("Pipeline: torch.export -> edge compile -> save\n")


def simulate_executorch_export(model, sample_input):
    """Simulate the ExecuTorch export pipeline."""
    steps = []

    # Step 1: torch.export (captures the full graph)
    try:
        exported = torch.export.export(model, (sample_input,))
        steps.append(("torch.export", "OK"))
        graph_nodes = len(exported.graph.nodes)
    except Exception as e:
        # Fallback to tracing if export is unavailable
        exported = torch.jit.trace(model, sample_input)
        steps.append(("torch.export", f"fallback to trace: {e}"))
        graph_nodes = len(list(exported.graph.nodes()))

    # Step 2: Edge-specific lowering (simulated)
    steps.append(("edge_compile", "OK (simulated)"))

    # Step 3: Operator partitioning (simulated)
    steps.append(("partition (CPU delegate)", "OK (simulated)"))

    # Step 4: Serialize
    export_path = os.path.join(tempfile.gettempdir(), "model_executorch.pte")
    if hasattr(exported, 'save'):
        exported.save(export_path)
    else:
        torch.save(exported, export_path)
    export_size = os.path.getsize(export_path) / (1024 * 1024)
    steps.append(("serialize", f"OK ({export_size:.2f} MB)"))

    return steps, graph_nodes


steps, n_nodes = simulate_executorch_export(model, dummy_input)

print(f"{'Step':<30} {'Status'}")
print("-" * 55)
for step_name, status in steps:
    print(f"{step_name:<30} {status}")
print(f"\nGraph nodes: {n_nodes}")


# ============================================
# 7. Inference Benchmark
# ============================================
print("\n[7] Inference Latency Benchmark")
print("-" * 40)

n_warmup = 20
n_runs = 100

# Warmup
for _ in range(n_warmup):
    with torch.no_grad():
        model(dummy_input)

# Benchmark original
start = time.perf_counter()
for _ in range(n_runs):
    with torch.no_grad():
        model(dummy_input)
orig_ms = (time.perf_counter() - start) / n_runs * 1000

# Benchmark traced
for _ in range(n_warmup):
    with torch.no_grad():
        traced_model(dummy_input)

start = time.perf_counter()
for _ in range(n_runs):
    with torch.no_grad():
        traced_model(dummy_input)
traced_ms = (time.perf_counter() - start) / n_runs * 1000

print(f"{'Model':<25} {'Latency (ms)':<15}")
print("-" * 40)
print(f"{'Original (eager)':<25} {orig_ms:<15.2f}")
print(f"{'TorchScript (traced)':<25} {traced_ms:<15.2f}")
print(f"{'Speedup':<25} {orig_ms / traced_ms:<15.2f}x")


# ============================================
# 8. Summary
# ============================================
print("\n[8] Summary")
print("-" * 40)
print("Key takeaways:")
print("- TorchScript tracing: fast, but no data-dependent control flow")
print("- TorchScript scripting: handles branches/loops, full Python subset")
print("- Mobile optimizer: fuses ops, drops dropout, reduces .ptl size")
print("- ExecuTorch: next-gen runtime with delegated execution on NPUs/DSPs")
print("- Always verify numerical equivalence after export/optimization")

# Cleanup temp files
for f in ["mini_mobilenet_traced.pt", "conditional_scripted.pt",
          "mini_mobilenet_optimized.ptl", "model_executorch.pte"]:
    path = os.path.join(tempfile.gettempdir(), f)
    if os.path.exists(path):
        os.remove(path)
