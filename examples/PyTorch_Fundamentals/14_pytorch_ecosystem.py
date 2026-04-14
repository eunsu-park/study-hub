"""
PyTorch Ecosystem - Examples
============================
Lesson 14: PyTorch Ecosystem

Demonstrates:
  1. torchvision models and transforms
  2. torchmetrics usage
  3. PyTorch Lightning module structure
  4. Model inspection utilities
"""

import torch
import torch.nn as nn


def example_1_torchvision_models():
    """Load and modify pretrained torchvision models."""
    print("=" * 60)
    print("Example 1: torchvision Models")
    print("=" * 60)

    try:
        import torchvision.models as models

        # List a few available models
        all_models = models.list_models()
        print(f"Total available models: {len(all_models)}")
        print(f"Some models: {all_models[:5]}")

        # Load ResNet18 (without downloading weights for speed)
        resnet = models.resnet18(weights=None)
        print(f"\nResNet18 output: {resnet.fc.out_features} classes")

        # Modify for 5 classes
        resnet.fc = nn.Linear(resnet.fc.in_features, 5)
        x = torch.randn(2, 3, 224, 224)
        out = resnet(x)
        print(f"Modified output shape: {out.shape}")

    except ImportError:
        print("torchvision not installed. pip install torchvision")


def example_2_transforms():
    """Data transforms for image preprocessing."""
    print("\n" + "=" * 60)
    print("Example 2: Transforms")
    print("=" * 60)

    try:
        from torchvision import transforms

        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        val_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        print(f"Train transform: {len(train_transform.transforms)} steps")
        print(f"Val transform: {len(val_transform.transforms)} steps")

        # Apply to a random tensor (simulating an image)
        fake_image = torch.randint(0, 256, (3, 256, 256), dtype=torch.uint8)
        normalized = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )(fake_image.float() / 255.0)
        print(f"Normalized range: [{normalized.min():.2f}, "
              f"{normalized.max():.2f}]")

    except ImportError:
        print("torchvision not installed.")


def example_3_lightning_structure():
    """Show PyTorch Lightning module structure (no Lightning required)."""
    print("\n" + "=" * 60)
    print("Example 3: Lightning-Style Module (Pure PyTorch)")
    print("=" * 60)

    class LightningStyleModel(nn.Module):
        """Mimics Lightning's structure using pure PyTorch."""

        def __init__(self, input_dim, hidden_dim, output_dim, lr=1e-3):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, output_dim),
            )
            self.lr = lr

        def forward(self, x):
            return self.net(x)

        def training_step(self, batch):
            x, y = batch
            logits = self(x)
            loss = nn.functional.cross_entropy(logits, y)
            acc = (logits.argmax(1) == y).float().mean()
            return {'loss': loss, 'acc': acc}

        def configure_optimizers(self):
            return torch.optim.Adam(self.parameters(), lr=self.lr)

    model = LightningStyleModel(20, 64, 5)
    optimizer = model.configure_optimizers()

    # Simulated training step
    batch = (torch.randn(32, 20), torch.randint(0, 5, (32,)))
    result = model.training_step(batch)
    print(f"Loss: {result['loss'].item():.4f}")
    print(f"Accuracy: {result['acc'].item():.2%}")


def example_4_model_summary():
    """Custom model summary utility."""
    print("\n" + "=" * 60)
    print("Example 4: Model Summary Utility")
    print("=" * 60)

    def model_summary(model, input_size=None):
        total_params = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters()
                        if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable: {trainable:,}")
        print(f"Non-trainable: {total_params - trainable:,}")

        # Size estimate
        param_size = sum(p.numel() * p.element_size()
                         for p in model.parameters())
        buffer_size = sum(b.numel() * b.element_size()
                          for b in model.buffers())
        print(f"Parameter size: {param_size/1024:.1f} KB")
        print(f"Buffer size: {buffer_size/1024:.1f} KB")

        print("\nLayers:")
        for name, module in model.named_modules():
            if name:  # skip root
                n_params = sum(p.numel() for p in module.parameters(
                    recurse=False))
                if n_params > 0:
                    print(f"  {name}: {type(module).__name__} "
                          f"({n_params:,} params)")

    model = nn.Sequential(
        nn.Linear(784, 256), nn.ReLU(), nn.BatchNorm1d(256),
        nn.Linear(256, 128), nn.ReLU(),
        nn.Linear(128, 10),
    )
    model_summary(model)


if __name__ == "__main__":
    example_1_torchvision_models()
    example_2_transforms()
    example_3_lightning_structure()
    example_4_model_summary()
