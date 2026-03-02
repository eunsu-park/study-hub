"""
12. Model Saving and Deployment

Implements PyTorch model saving, TorchScript, and ONNX conversion.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import tempfile

print("=" * 60)
print("PyTorch Model Saving and Deployment")
print("=" * 60)


# ============================================
# 1. Sample Model
# ============================================
print("\n[1] Sample Model")
print("-" * 40)

class SimpleClassifier(nn.Module):
    def __init__(self, input_size=784, hidden_size=256, num_classes=10):
        super().__init__()
        self.config = {
            'input_size': input_size,
            'hidden_size': hidden_size,
            'num_classes': num_classes
        }
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.fc2(x)
        return x

model = SimpleClassifier()
print(f"Model structure:\n{model}")
print(f"Parameter count: {sum(p.numel() for p in model.parameters()):,}")


# ============================================
# 2. state_dict Saving
# ============================================
print("\n[2] state_dict Saving")
print("-" * 40)

# Use temporary directory
save_dir = tempfile.mkdtemp()

# Save
weights_path = os.path.join(save_dir, 'model_weights.pth')
torch.save(model.state_dict(), weights_path)
print(f"Saved: {weights_path}")
print(f"File size: {os.path.getsize(weights_path) / 1024:.2f} KB")

# Load
loaded_model = SimpleClassifier()
loaded_model.load_state_dict(torch.load(weights_path, weights_only=True))
loaded_model.eval()

# Verify
x = torch.randn(2, 1, 28, 28)
model.eval()
with torch.no_grad():
    original_out = model(x)
    loaded_out = loaded_model(x)
    diff = (original_out - loaded_out).abs().max().item()
    print(f"Output difference: {diff:.10f}")


# ============================================
# 3. Checkpoint Saving
# ============================================
print("\n[3] Checkpoint Saving")
print("-" * 40)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Simulated training state
epoch = 10
loss = 0.123
best_acc = 0.95

# Save checkpoint
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
    'best_acc': best_acc,
    'model_config': model.config
}

checkpoint_path = os.path.join(save_dir, 'checkpoint.pth')
torch.save(checkpoint, checkpoint_path)
print(f"Checkpoint saved: {checkpoint_path}")

# Load checkpoint
loaded_checkpoint = torch.load(checkpoint_path, weights_only=False)
print(f"Loaded epoch: {loaded_checkpoint['epoch']}")
print(f"Loaded best_acc: {loaded_checkpoint['best_acc']}")
print(f"Model config: {loaded_checkpoint['model_config']}")


# ============================================
# 4. TorchScript - Tracing
# ============================================
print("\n[4] TorchScript - Tracing")
print("-" * 40)

model.eval()
example_input = torch.randn(1, 1, 28, 28)

# Trace
traced_model = torch.jit.trace(model, example_input)

# Save
traced_path = os.path.join(save_dir, 'model_traced.pt')
traced_model.save(traced_path)
print(f"TorchScript saved: {traced_path}")
print(f"File size: {os.path.getsize(traced_path) / 1024:.2f} KB")

# Load and verify
loaded_traced = torch.jit.load(traced_path)
with torch.no_grad():
    traced_out = loaded_traced(example_input)
    original_out = model(example_input)
    diff = (traced_out - original_out).abs().max().item()
    print(f"Output difference: {diff:.10f}")


# ============================================
# 5. TorchScript - Scripting
# ============================================
print("\n[5] TorchScript - Scripting")
print("-" * 40)

class ConditionalModel(nn.Module):
    """Model with conditional logic"""
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 5)

    def forward(self, x, use_relu: bool = True):
        x = self.fc(x)
        if use_relu:
            x = F.relu(x)
        return x

cond_model = ConditionalModel()
scripted_model = torch.jit.script(cond_model)

scripted_path = os.path.join(save_dir, 'model_scripted.pt')
scripted_model.save(scripted_path)
print(f"Scripted model saved: {scripted_path}")

# Conditional execution test
x = torch.randn(2, 10)
out_relu = scripted_model(x, True)
out_no_relu = scripted_model(x, False)
print(f"With ReLU: min={out_relu.min():.4f}")
print(f"Without ReLU: min={out_no_relu.min():.4f}")


# ============================================
# 6. ONNX Conversion
# ============================================
print("\n[6] ONNX Conversion")
print("-" * 40)

try:
    import onnx

    model.eval()
    dummy_input = torch.randn(1, 1, 28, 28)

    onnx_path = os.path.join(save_dir, 'model.onnx')

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        },
        opset_version=11
    )

    print(f"ONNX saved: {onnx_path}")
    print(f"File size: {os.path.getsize(onnx_path) / 1024:.2f} KB")

    # Verify
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX model verification passed")

except ImportError:
    print("onnx not installed - skipping")


# ============================================
# 7. ONNX Runtime Inference
# ============================================
print("\n[7] ONNX Runtime Inference")
print("-" * 40)

try:
    import onnxruntime as ort
    import numpy as np

    session = ort.InferenceSession(onnx_path)

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    # Inference
    input_data = np.random.randn(2, 1, 28, 28).astype(np.float32)
    result = session.run([output_name], {input_name: input_data})

    print(f"ONNX Runtime output: {result[0].shape}")

    # Compare with PyTorch result
    model.eval()
    with torch.no_grad():
        torch_out = model(torch.from_numpy(input_data))
        diff = np.abs(result[0] - torch_out.numpy()).max()
        print(f"PyTorch vs ONNX difference: {diff:.6f}")

except ImportError:
    print("onnxruntime not installed - skipping")


# ============================================
# 8. Quantization
# ============================================
print("\n[8] Quantization")
print("-" * 40)

# Dynamic quantization
quantized_model = torch.quantization.quantize_dynamic(
    model, {nn.Linear}, dtype=torch.qint8
)

# Size comparison
original_size = sum(p.numel() * p.element_size() for p in model.parameters())
quantized_size = sum(
    p.numel() * p.element_size() for p in quantized_model.parameters()
    if p.dtype != torch.qint8
)

print(f"Original model size: {original_size / 1024:.2f} KB")
print(f"Quantized model (partial layers): ~{original_size / 1024 * 0.25:.2f} KB (estimated)")

# Inference comparison
x = torch.randn(100, 1, 28, 28)

model.eval()
quantized_model.eval()

import time

# Original model
start = time.time()
for _ in range(10):
    with torch.no_grad():
        _ = model(x)
original_time = time.time() - start

# Quantized model
start = time.time()
for _ in range(10):
    with torch.no_grad():
        _ = quantized_model(x)
quantized_time = time.time() - start

print(f"Original inference time: {original_time*1000:.2f} ms")
print(f"Quantized inference time: {quantized_time*1000:.2f} ms")


# ============================================
# 9. Inference Optimization
# ============================================
print("\n[9] Inference Optimization")
print("-" * 40)

model.eval()
x = torch.randn(100, 1, 28, 28)

# no_grad
start = time.time()
for _ in range(100):
    with torch.no_grad():
        _ = model(x)
no_grad_time = time.time() - start

# inference_mode (faster)
start = time.time()
for _ in range(100):
    with torch.inference_mode():
        _ = model(x)
inference_time = time.time() - start

print(f"no_grad time: {no_grad_time*1000:.2f} ms")
print(f"inference_mode time: {inference_time*1000:.2f} ms")
print(f"Improvement: {(no_grad_time - inference_time) / no_grad_time * 100:.1f}%")


# ============================================
# 10. Mobile Optimization
# ============================================
print("\n[10] Mobile Optimization")
print("-" * 40)

try:
    # Optimize for mobile
    traced_model = torch.jit.trace(model.eval(), example_input)
    optimized_model = torch.utils.mobile_optimizer.optimize_for_mobile(traced_model)

    mobile_path = os.path.join(save_dir, 'model_mobile.ptl')
    optimized_model._save_for_lite_interpreter(mobile_path)

    print(f"Mobile model saved: {mobile_path}")
    print(f"File size: {os.path.getsize(mobile_path) / 1024:.2f} KB")
except Exception as e:
    print(f"Mobile optimization skipped: {e}")


# ============================================
# 11. Saved Files List
# ============================================
print("\n[11] Saved Files List")
print("-" * 40)

print(f"Save directory: {save_dir}")
for f in os.listdir(save_dir):
    path = os.path.join(save_dir, f)
    size = os.path.getsize(path) / 1024
    print(f"  {f}: {size:.2f} KB")


# ============================================
# Summary
# ============================================
print("\n" + "=" * 60)
print("Model Saving and Deployment Summary")
print("=" * 60)

summary = """
Saving methods:

1. state_dict (recommended)
   torch.save(model.state_dict(), 'model.pth')
   model.load_state_dict(torch.load('model.pth'))

2. Checkpoint
   checkpoint = {'model': model.state_dict(), 'optimizer': ...}
   torch.save(checkpoint, 'checkpoint.pth')

3. TorchScript
   traced = torch.jit.trace(model, example_input)
   traced.save('model.pt')

4. ONNX
   torch.onnx.export(model, input, 'model.onnx')

Inference optimization:
   - model.eval()
   - torch.inference_mode()
   - Quantization (quantize_dynamic)

Deployment options:
   - FastAPI/Flask: Web API
   - ONNX Runtime: Universal inference
   - TorchScript: C++ deployment
   - PyTorch Mobile: Mobile apps
"""
print(summary)
print("=" * 60)

# Temporary file cleanup notice
print(f"\nTemporary file location: {save_dir}")
print("(Not automatically deleted - delete manually if needed)")
