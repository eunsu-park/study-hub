"""
Exercises for Lesson 41: Model Saving and Deployment
Topic: Deep_Learning

Solutions to practice problems from the lesson.
"""

import os
import tempfile
import torch
import torch.nn as nn


# A simple model used throughout the exercises
class SimpleClassifier(nn.Module):
    """Lightweight 3-layer MLP for MNIST-like tasks."""

    def __init__(self, input_size=784, hidden_size=256, num_classes=10):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


# === Exercise 1: state_dict Save and Load ===
# Problem: Save a model's state_dict to disk, create a fresh model instance,
# load the weights, and verify outputs are identical before and after loading.

def exercise_1():
    """Save and load model weights via state_dict."""
    torch.manual_seed(42)
    model = SimpleClassifier()
    model.eval()

    x = torch.randn(4, 784)
    with torch.no_grad():
        out_before = model(x)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "model_weights.pth")

        # Save
        torch.save(model.state_dict(), path)
        print("  Saved state_dict to: model_weights.pth")
        print("  File size: {:.1f} KB".format(os.path.getsize(path) / 1024))

        # Load into a fresh model
        model2 = SimpleClassifier()
        model2.load_state_dict(torch.load(path, weights_only=True))
        model2.eval()

        with torch.no_grad():
            out_after = model2(x)

    print("  Outputs identical after reload: {}".format(
        torch.allclose(out_before, out_after)))

    # Inspect state_dict keys
    keys = list(model.state_dict().keys())
    print("  state_dict keys: {}".format(keys))


# === Exercise 2: Checkpoint Saving ===
# Problem: Simulate a training loop that saves a full checkpoint
# (model weights + optimizer state + epoch + best accuracy).
# Then restore from the checkpoint and verify training can resume.

def exercise_2():
    """Save and restore a training checkpoint."""
    torch.manual_seed(0)
    model = SimpleClassifier()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Simulate a few training steps to advance optimizer state
    for _ in range(5):
        dummy_loss = model(torch.randn(8, 784)).mean()
        optimizer.zero_grad()
        dummy_loss.backward()
        optimizer.step()

    checkpoint = {
        'epoch': 10,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': 0.2345,
        'best_acc': 0.932,
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, "checkpoint_epoch10.pth")
        torch.save(checkpoint, ckpt_path)
        print("  Checkpoint saved ({:.1f} KB)".format(os.path.getsize(ckpt_path) / 1024))

        # Restore
        loaded_ckpt = torch.load(ckpt_path, weights_only=False)
        model2 = SimpleClassifier()
        optimizer2 = torch.optim.Adam(model2.parameters(), lr=1e-3)
        model2.load_state_dict(loaded_ckpt['model_state_dict'])
        optimizer2.load_state_dict(loaded_ckpt['optimizer_state_dict'])

    print("  Restored epoch: {}".format(loaded_ckpt['epoch']))
    print("  Restored best_acc: {}".format(loaded_ckpt['best_acc']))
    print("  Optimizer lr restored: {}".format(
        optimizer2.param_groups[0]['lr']))


# === Exercise 3: TorchScript Tracing ===
# Problem: Export a model via torch.jit.trace, save it to disk, reload it,
# and confirm that traced and original outputs match.

def exercise_3():
    """Trace a model with TorchScript and verify correctness."""
    torch.manual_seed(7)
    model = SimpleClassifier()
    model.eval()

    example_input = torch.randn(1, 784)

    with torch.no_grad():
        out_original = model(example_input)

    # Trace
    traced = torch.jit.trace(model, example_input)

    with torch.no_grad():
        out_traced = traced(example_input)

    print("  Original output shape: {}".format(out_original.shape))
    print("  Traced output shape:   {}".format(out_traced.shape))
    print("  Outputs match: {}".format(torch.allclose(out_original, out_traced)))

    with tempfile.TemporaryDirectory() as tmpdir:
        pt_path = os.path.join(tmpdir, "model_traced.pt")
        traced.save(pt_path)
        print("  TorchScript file size: {:.1f} KB".format(os.path.getsize(pt_path) / 1024))

        # Reload (can be done from C++ as well)
        loaded_traced = torch.jit.load(pt_path)
        with torch.no_grad():
            out_loaded = loaded_traced(example_input)

    print("  Reloaded output matches: {}".format(
        torch.allclose(out_original, out_loaded)))


# === Exercise 4: ONNX Export ===
# Problem: Export the model to ONNX with dynamic batch size, validate the
# ONNX model, and run inference with ONNX Runtime.

def exercise_4():
    """Export model to ONNX and run inference with onnxruntime."""
    try:
        import onnx
        import onnxruntime as ort
        import numpy as np
    except ImportError:
        print("  Skipped: onnx and onnxruntime not installed.")
        print("  Install with: pip install onnx onnxruntime")
        return

    torch.manual_seed(3)
    model = SimpleClassifier()
    model.eval()

    dummy_input = torch.randn(1, 784)

    with tempfile.TemporaryDirectory() as tmpdir:
        onnx_path = os.path.join(tmpdir, "model.onnx")

        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
            opset_version=17,
        )

        # Validate
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        print("  ONNX export and validation: OK")

        # Inference with ONNX Runtime
        session = ort.InferenceSession(onnx_path)
        test_input = np.random.randn(4, 784).astype(np.float32)
        ort_out = session.run(['output'], {'input': test_input})[0]

        # Compare with PyTorch
        with torch.no_grad():
            pt_out = model(torch.from_numpy(test_input)).numpy()

        print("  ONNX Runtime output shape: {}".format(ort_out.shape))
        print("  Outputs match (atol=1e-5): {}".format(
            np.allclose(ort_out, pt_out, atol=1e-5)))


# === Exercise 5: Inference Optimization ===
# Problem: Compare inference speed/correctness across three modes:
# (1) model.train(), (2) model.eval() + no_grad, (3) model.eval() + inference_mode.
# Show that eval() + inference_mode is preferred for deployment.

def exercise_5():
    """Compare inference modes and quantization."""
    import time

    torch.manual_seed(1)
    model = SimpleClassifier()
    x = torch.randn(64, 784)
    N = 200

    # Mode 1: train mode (includes dropout, builds gradient graph)
    model.train()
    t0 = time.perf_counter()
    for _ in range(N):
        _ = model(x)
    t_train = time.perf_counter() - t0

    # Mode 2: eval + no_grad
    model.eval()
    t0 = time.perf_counter()
    for _ in range(N):
        with torch.no_grad():
            _ = model(x)
    t_no_grad = time.perf_counter() - t0

    # Mode 3: eval + inference_mode (PyTorch 1.9+)
    t0 = time.perf_counter()
    for _ in range(N):
        with torch.inference_mode():
            _ = model(x)
    t_infer = time.perf_counter() - t0

    print("  train mode:          {:.4f}s ({} iters)".format(t_train, N))
    print("  eval + no_grad:      {:.4f}s ({} iters)".format(t_no_grad, N))
    print("  eval + infer_mode:   {:.4f}s ({} iters)".format(t_infer, N))

    # Dynamic quantization (CPU only, int8)
    model.eval()
    q_model = torch.quantization.quantize_dynamic(
        model, {nn.Linear}, dtype=torch.qint8
    )
    with torch.inference_mode():
        out_fp32 = model(x)
        out_int8 = q_model(x)

    print("  Quantization output diff (max): {:.6f}".format(
        (out_fp32 - out_int8).abs().max().item()))

    q_params = sum(p.numel() for p in q_model.parameters())
    fp32_params = sum(p.numel() for p in model.parameters())
    print("  FP32 params: {:,}, INT8 params: {:,}".format(fp32_params, q_params))


if __name__ == "__main__":
    print("=== Exercise 1: state_dict Save and Load ===")
    exercise_1()
    print("\n=== Exercise 2: Checkpoint Saving and Restoration ===")
    exercise_2()
    print("\n=== Exercise 3: TorchScript Tracing ===")
    exercise_3()
    print("\n=== Exercise 4: ONNX Export ===")
    exercise_4()
    print("\n=== Exercise 5: Inference Optimization and Quantization ===")
    exercise_5()
    print("\nAll exercises completed!")
