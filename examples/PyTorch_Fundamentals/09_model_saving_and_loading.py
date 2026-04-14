"""
Model Saving and Loading - Examples
===================================
Lesson 09: Model Saving and Loading

Demonstrates:
  1. Saving/loading state_dict
  2. Complete checkpoint save/load
  3. Partial loading for transfer learning
  4. ONNX export (conceptual)
"""

import torch
import torch.nn as nn
import os
import tempfile


def example_1_state_dict():
    """Save and load model weights via state_dict."""
    print("=" * 60)
    print("Example 1: state_dict Save/Load")
    print("=" * 60)

    model = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 5))

    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
        path = f.name
        torch.save(model.state_dict(), path)
        size_kb = os.path.getsize(path) / 1024
        print(f"Saved to {path} ({size_kb:.1f} KB)")

    model2 = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 5))
    model2.load_state_dict(torch.load(path, weights_only=True))
    model2.eval()

    x = torch.randn(1, 10)
    assert torch.equal(model(x), model2(x)), "Outputs should match"
    print("Loaded model produces identical output.")
    os.unlink(path)


def example_2_checkpoint():
    """Save and load a complete training checkpoint."""
    print("\n" + "=" * 60)
    print("Example 2: Training Checkpoint")
    print("=" * 60)

    model = nn.Linear(10, 5)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Simulate a few training steps
    for _ in range(3):
        x = torch.randn(4, 10)
        loss = model(x).sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    checkpoint = {
        'epoch': 10,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_val_loss': 0.42,
    }

    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
        path = f.name
        torch.save(checkpoint, path)
        print(f"Checkpoint saved ({os.path.getsize(path)/1024:.1f} KB)")

    loaded = torch.load(path, weights_only=False)
    print(f"Resumed from epoch {loaded['epoch']}")
    print(f"Best val loss: {loaded['best_val_loss']}")
    os.unlink(path)


def example_3_partial_load():
    """Partial loading for transfer learning."""
    print("\n" + "=" * 60)
    print("Example 3: Partial Loading (Transfer Learning)")
    print("=" * 60)

    # Old model: 5 classes
    old_model = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 5))
    old_state = old_model.state_dict()

    # New model: 10 classes (different output layer)
    new_model = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 10))
    new_state = new_model.state_dict()

    # Load matching keys only
    pretrained = {k: v for k, v in old_state.items()
                  if k in new_state and v.shape == new_state[k].shape}
    new_state.update(pretrained)
    new_model.load_state_dict(new_state)

    print(f"Loaded {len(pretrained)}/{len(new_state)} parameters")
    for k in new_state:
        status = "loaded" if k in pretrained else "random"
        print(f"  {k}: {new_state[k].shape} [{status}]")


def example_4_model_checkpoint_class():
    """Automatic best-model saving utility."""
    print("\n" + "=" * 60)
    print("Example 4: ModelCheckpoint Utility")
    print("=" * 60)

    class ModelCheckpoint:
        def __init__(self, path, mode='min'):
            self.path = path
            self.best = float('inf') if mode == 'min' else float('-inf')
            self.mode = mode

        def __call__(self, metric, model):
            improved = (metric < self.best if self.mode == 'min'
                        else metric > self.best)
            if improved:
                self.best = metric
                torch.save(model.state_dict(), self.path)
                return True
            return False

    model = nn.Linear(10, 5)

    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
        ckpt = ModelCheckpoint(f.name)

        fake_losses = [1.0, 0.8, 0.9, 0.7, 0.75, 0.6]
        for epoch, loss in enumerate(fake_losses):
            saved = ckpt(loss, model)
            status = "SAVED" if saved else "skip"
            print(f"Epoch {epoch}: loss={loss:.2f} [{status}] "
                  f"best={ckpt.best:.2f}")

        os.unlink(f.name)


if __name__ == "__main__":
    example_1_state_dict()
    example_2_checkpoint()
    example_3_partial_load()
    example_4_model_checkpoint_class()
