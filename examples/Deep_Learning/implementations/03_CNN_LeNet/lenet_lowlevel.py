"""
PyTorch Low-Level LeNet-5 Implementation

Uses F.conv2d and torch.matmul instead of nn.Conv2d and nn.Linear.
Parameters are managed manually.
"""

import torch
import torch.nn.functional as F
import math
from typing import Tuple, List


class LeNetLowLevel:
    """
    LeNet-5 Low-Level Implementation

    Does not use nn.Module; uses only basic operations like F.conv2d.
    """

    def __init__(self, num_classes: int = 10):
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Conv1: 1 -> 6 channels, 5x5 kernel
        self.conv1_weight = self._init_conv_weight(1, 6, 5)
        self.conv1_bias = torch.zeros(6, requires_grad=True, device=self.device)

        # Conv2: 6 -> 16 channels, 5x5 kernel
        self.conv2_weight = self._init_conv_weight(6, 16, 5)
        self.conv2_bias = torch.zeros(16, requires_grad=True, device=self.device)

        # Conv3: 16 -> 120 channels, 5x5 kernel
        self.conv3_weight = self._init_conv_weight(16, 120, 5)
        self.conv3_bias = torch.zeros(120, requires_grad=True, device=self.device)

        # FC1: 120 -> 84
        self.fc1_weight = self._init_linear_weight(120, 84)
        self.fc1_bias = torch.zeros(84, requires_grad=True, device=self.device)

        # FC2: 84 -> num_classes
        self.fc2_weight = self._init_linear_weight(84, num_classes)
        self.fc2_bias = torch.zeros(num_classes, requires_grad=True, device=self.device)

    def _init_conv_weight(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int
    ) -> torch.Tensor:
        """Kaiming initialization"""
        fan_in = in_channels * kernel_size * kernel_size
        std = math.sqrt(2.0 / fan_in)
        weight = torch.randn(
            out_channels, in_channels, kernel_size, kernel_size,
            requires_grad=True, device=self.device
        ) * std
        return weight

    def _init_linear_weight(
        self,
        in_features: int,
        out_features: int
    ) -> torch.Tensor:
        """Xavier initialization"""
        std = math.sqrt(2.0 / (in_features + out_features))
        weight = torch.randn(
            out_features, in_features,
            requires_grad=True, device=self.device
        ) * std
        return weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: (N, 1, 32, 32) input image

        Returns:
            logits: (N, num_classes)
        """
        # Layer 1: Conv -> ReLU -> AvgPool
        # (N, 1, 32, 32) -> (N, 6, 28, 28) -> (N, 6, 14, 14)
        x = F.conv2d(x, self.conv1_weight, self.conv1_bias, stride=1, padding=0)
        x = F.relu(x)
        x = F.avg_pool2d(x, kernel_size=2, stride=2)

        # Layer 2: Conv -> ReLU -> AvgPool
        # (N, 6, 14, 14) -> (N, 16, 10, 10) -> (N, 16, 5, 5)
        x = F.conv2d(x, self.conv2_weight, self.conv2_bias, stride=1, padding=0)
        x = F.relu(x)
        x = F.avg_pool2d(x, kernel_size=2, stride=2)

        # Layer 3: Conv -> ReLU
        # (N, 16, 5, 5) -> (N, 120, 1, 1)
        x = F.conv2d(x, self.conv3_weight, self.conv3_bias, stride=1, padding=0)
        x = F.relu(x)

        # Flatten: (N, 120, 1, 1) -> (N, 120)
        x = x.view(x.size(0), -1)

        # FC1: (N, 120) -> (N, 84)
        x = torch.matmul(x, self.fc1_weight.t()) + self.fc1_bias
        x = F.relu(x)

        # FC2: (N, 84) -> (N, num_classes)
        x = torch.matmul(x, self.fc2_weight.t()) + self.fc2_bias

        return x

    def parameters(self) -> List[torch.Tensor]:
        """Return trainable parameters"""
        return [
            self.conv1_weight, self.conv1_bias,
            self.conv2_weight, self.conv2_bias,
            self.conv3_weight, self.conv3_bias,
            self.fc1_weight, self.fc1_bias,
            self.fc2_weight, self.fc2_bias,
        ]

    def zero_grad(self):
        """Reset gradients"""
        for param in self.parameters():
            if param.grad is not None:
                param.grad.zero_()

    def to(self, device):
        """Move to device"""
        self.device = device
        for param in self.parameters():
            param.data = param.data.to(device)
            if param.grad is not None:
                param.grad = param.grad.to(device)
        return self


def sgd_step(params: List[torch.Tensor], lr: float):
    """Manual SGD update"""
    with torch.no_grad():
        for param in params:
            if param.grad is not None:
                param -= lr * param.grad


def sgd_step_with_momentum(
    params: List[torch.Tensor],
    velocities: List[torch.Tensor],
    lr: float,
    momentum: float = 0.9
):
    """Momentum SGD"""
    with torch.no_grad():
        for param, velocity in zip(params, velocities):
            if param.grad is not None:
                velocity.mul_(momentum).add_(param.grad)
                param -= lr * velocity


def train_epoch(
    model: LeNetLowLevel,
    dataloader,
    lr: float = 0.01
) -> Tuple[float, float]:
    """Train for one epoch"""
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in dataloader:
        images = images.to(model.device)
        labels = labels.to(model.device)

        # Forward
        logits = model.forward(images)

        # Loss (compute Cross Entropy directly)
        # log_softmax + nll_loss = cross_entropy
        log_probs = F.log_softmax(logits, dim=1)
        loss = F.nll_loss(log_probs, labels)

        # Backward
        model.zero_grad()
        loss.backward()

        # Update
        sgd_step(model.parameters(), lr)

        # Metrics
        total_loss += loss.item() * images.size(0)
        predictions = logits.argmax(dim=1)
        total_correct += (predictions == labels).sum().item()
        total_samples += images.size(0)

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples

    return avg_loss, accuracy


@torch.no_grad()
def evaluate(
    model: LeNetLowLevel,
    dataloader
) -> Tuple[float, float]:
    """Evaluate"""
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in dataloader:
        images = images.to(model.device)
        labels = labels.to(model.device)

        # Forward
        logits = model.forward(images)

        # Loss
        loss = F.cross_entropy(logits, labels)

        # Metrics
        total_loss += loss.item() * images.size(0)
        predictions = logits.argmax(dim=1)
        total_correct += (predictions == labels).sum().item()
        total_samples += images.size(0)

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples

    return avg_loss, accuracy


def main():
    """Training script"""
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader

    print("=== LeNet-5 Low-Level Training ===\n")

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Dataset (MNIST resized to 32x32)
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST(
        root='./data', train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        root='./data', train=False, download=True, transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    print(f"Train samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}\n")

    # Model
    model = LeNetLowLevel(num_classes=10)
    model.to(device)

    # Parameter count
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}\n")

    # Training
    epochs = 10
    lr = 0.01

    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, train_loader, lr)
        test_loss, test_acc = evaluate(model, test_loader)

        print(f"Epoch {epoch+1}/{epochs}")
        print(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
        print(f"  Test  - Loss: {test_loss:.4f}, Acc: {test_acc:.4f}")

        # Learning rate decay
        if (epoch + 1) % 5 == 0:
            lr *= 0.5
            print(f"  LR -> {lr}")

        print()

    print("Training complete!")

    # Final results
    final_loss, final_acc = evaluate(model, test_loader)
    print(f"\nFinal Test Accuracy: {final_acc:.4f}")


# Convolution operation visualization
def visualize_conv_operation():
    """Visualize the convolution operation process"""
    import matplotlib.pyplot as plt

    # Simple input
    input_img = torch.zeros(1, 1, 5, 5)
    input_img[0, 0, 1:4, 1:4] = 1.0  # 3x3 square in the center

    # Edge detection filters
    filters = {
        'Horizontal': torch.tensor([
            [-1, -1, -1],
            [ 0,  0,  0],
            [ 1,  1,  1]
        ]).float().view(1, 1, 3, 3),

        'Vertical': torch.tensor([
            [-1, 0, 1],
            [-1, 0, 1],
            [-1, 0, 1]
        ]).float().view(1, 1, 3, 3),

        'Identity': torch.tensor([
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0]
        ]).float().view(1, 1, 3, 3),
    }

    fig, axes = plt.subplots(2, len(filters) + 1, figsize=(12, 6))

    # Input image
    axes[0, 0].imshow(input_img[0, 0], cmap='gray')
    axes[0, 0].set_title('Input')
    axes[0, 0].axis('off')

    axes[1, 0].axis('off')

    # Apply each filter
    for i, (name, kernel) in enumerate(filters.items()):
        output = F.conv2d(input_img, kernel, padding=1)

        # Filter
        axes[0, i+1].imshow(kernel[0, 0], cmap='RdBu', vmin=-1, vmax=1)
        axes[0, i+1].set_title(f'{name} Filter')
        axes[0, i+1].axis('off')

        # Output
        axes[1, i+1].imshow(output[0, 0].detach(), cmap='gray')
        axes[1, i+1].set_title(f'Output')
        axes[1, i+1].axis('off')

    plt.tight_layout()
    plt.savefig('conv_visualization.png', dpi=150)
    print("Saved conv_visualization.png")


if __name__ == "__main__":
    main()
