"""
PyTorch Low-Level VGG Implementation

Uses F.conv2d and torch.matmul instead of nn.Conv2d and nn.Linear.
Parameters are managed manually with block-based organization.
"""

import torch
import torch.nn.functional as F
import math
from typing import Tuple, List, Dict, Optional


# VGG configuration: numbers = output channels, 'M' = MaxPool
VGG_CONFIGS = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGGLowLevel:
    """
    VGG Low-Level Implementation

    Does not use nn.Module; uses only basic operations like F.conv2d.
    """

    def __init__(
        self,
        config_name: str = 'VGG16',
        num_classes: int = 1000,
        input_channels: int = 3,
        use_bn: bool = False
    ):
        """
        Args:
            config_name: VGG variant ('VGG11', 'VGG13', 'VGG16', 'VGG19')
            num_classes: Number of output classes
            input_channels: Number of input channels (RGB=3)
            use_bn: Whether to use Batch Normalization
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.config = VGG_CONFIGS[config_name]
        self.use_bn = use_bn

        # Feature extractor parameters
        self.conv_params = []
        self.bn_params = [] if use_bn else None
        self._build_features(input_channels)

        # Classifier parameters
        self._build_classifier(num_classes)

    def _init_conv_weight(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create Conv weights with Kaiming initialization"""
        fan_in = in_channels * kernel_size * kernel_size
        std = math.sqrt(2.0 / fan_in)

        weight = torch.randn(
            out_channels, in_channels, kernel_size, kernel_size,
            requires_grad=True, device=self.device
        ) * std
        bias = torch.zeros(out_channels, requires_grad=True, device=self.device)

        return weight, bias

    def _init_bn_params(self, num_features: int) -> Dict[str, torch.Tensor]:
        """Initialize BatchNorm parameters"""
        return {
            'gamma': torch.ones(num_features, requires_grad=True, device=self.device),
            'beta': torch.zeros(num_features, requires_grad=True, device=self.device),
            'running_mean': torch.zeros(num_features, device=self.device),
            'running_var': torch.ones(num_features, device=self.device),
        }

    def _init_linear_weight(
        self,
        in_features: int,
        out_features: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create Linear weights with Xavier initialization"""
        std = math.sqrt(2.0 / (in_features + out_features))

        weight = torch.randn(
            out_features, in_features,
            requires_grad=True, device=self.device
        ) * std
        bias = torch.zeros(out_features, requires_grad=True, device=self.device)

        return weight, bias

    def _build_features(self, input_channels: int):
        """Build feature extractor (Conv layers)"""
        in_channels = input_channels

        for v in self.config:
            if v == 'M':
                # MaxPool has no parameters
                self.conv_params.append('M')
                if self.use_bn:
                    self.bn_params.append(None)
            else:
                out_channels = v
                weight, bias = self._init_conv_weight(in_channels, out_channels, 3)
                self.conv_params.append({'weight': weight, 'bias': bias})

                if self.use_bn:
                    bn = self._init_bn_params(out_channels)
                    self.bn_params.append(bn)

                in_channels = out_channels

    def _build_classifier(self, num_classes: int):
        """Build classifier (FC layers)"""
        # 7x7x512 = 25088 (for 224x224 input)
        # For CIFAR-10 (32x32): 1x1x512 = 512

        # FC1: 25088 -> 4096
        self.fc1_weight, self.fc1_bias = self._init_linear_weight(512 * 7 * 7, 4096)

        # FC2: 4096 -> 4096
        self.fc2_weight, self.fc2_bias = self._init_linear_weight(4096, 4096)

        # FC3: 4096 -> num_classes
        self.fc3_weight, self.fc3_bias = self._init_linear_weight(4096, num_classes)

    def _batch_norm(
        self,
        x: torch.Tensor,
        bn_params: Dict[str, torch.Tensor],
        training: bool = True,
        momentum: float = 0.1,
        eps: float = 1e-5
    ) -> torch.Tensor:
        """Manual Batch Normalization"""
        if training:
            # Compute mean and var of current batch
            mean = x.mean(dim=(0, 2, 3), keepdim=True)
            var = x.var(dim=(0, 2, 3), unbiased=False, keepdim=True)

            # Update running statistics
            with torch.no_grad():
                bn_params['running_mean'] = (
                    (1 - momentum) * bn_params['running_mean'] +
                    momentum * mean.squeeze()
                )
                bn_params['running_var'] = (
                    (1 - momentum) * bn_params['running_var'] +
                    momentum * var.squeeze()
                )
        else:
            mean = bn_params['running_mean'].view(1, -1, 1, 1)
            var = bn_params['running_var'].view(1, -1, 1, 1)

        # Normalize
        x_norm = (x - mean) / torch.sqrt(var + eps)

        # Scale and shift
        gamma = bn_params['gamma'].view(1, -1, 1, 1)
        beta = bn_params['beta'].view(1, -1, 1, 1)

        return gamma * x_norm + beta

    def forward(
        self,
        x: torch.Tensor,
        training: bool = True
    ) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: (N, C, H, W) input image
            training: Training mode (affects BN and Dropout)

        Returns:
            logits: (N, num_classes)
        """
        # Feature extraction
        for i, params in enumerate(self.conv_params):
            if params == 'M':
                x = F.max_pool2d(x, kernel_size=2, stride=2)
            else:
                x = F.conv2d(x, params['weight'], params['bias'],
                            stride=1, padding=1)

                if self.use_bn and self.bn_params[i] is not None:
                    x = self._batch_norm(x, self.bn_params[i], training)

                x = F.relu(x)

        # Flatten
        x = x.view(x.size(0), -1)

        # Classifier
        # FC1
        x = torch.matmul(x, self.fc1_weight.t()) + self.fc1_bias
        x = F.relu(x)
        if training:
            x = F.dropout(x, p=0.5, training=True)

        # FC2
        x = torch.matmul(x, self.fc2_weight.t()) + self.fc2_bias
        x = F.relu(x)
        if training:
            x = F.dropout(x, p=0.5, training=True)

        # FC3
        x = torch.matmul(x, self.fc3_weight.t()) + self.fc3_bias

        return x

    def parameters(self) -> List[torch.Tensor]:
        """Return trainable parameters"""
        params = []

        # Conv parameters
        for p in self.conv_params:
            if p != 'M':
                params.extend([p['weight'], p['bias']])

        # BN parameters
        if self.use_bn:
            for bn in self.bn_params:
                if bn is not None:
                    params.extend([bn['gamma'], bn['beta']])

        # FC parameters
        params.extend([
            self.fc1_weight, self.fc1_bias,
            self.fc2_weight, self.fc2_bias,
            self.fc3_weight, self.fc3_bias,
        ])

        return params

    def zero_grad(self):
        """Reset gradients"""
        for param in self.parameters():
            if param.grad is not None:
                param.grad.zero_()

    def to(self, device):
        """Move to device"""
        self.device = device

        # Conv parameters
        for p in self.conv_params:
            if p != 'M':
                p['weight'] = p['weight'].to(device)
                p['bias'] = p['bias'].to(device)

        # BN parameters
        if self.use_bn:
            for bn in self.bn_params:
                if bn is not None:
                    for key in bn:
                        bn[key] = bn[key].to(device)

        # FC parameters
        for attr in ['fc1_weight', 'fc1_bias', 'fc2_weight',
                     'fc2_bias', 'fc3_weight', 'fc3_bias']:
            tensor = getattr(self, attr)
            setattr(self, attr, tensor.to(device))

        return self

    def count_parameters(self) -> int:
        """Count number of parameters"""
        return sum(p.numel() for p in self.parameters())


class VGGSmall(VGGLowLevel):
    """
    Small VGG for CIFAR-10

    Input: 32x32 -> Output feature map: 1x1x512
    """

    def _build_classifier(self, num_classes: int):
        """Classifier adapted for small input"""
        # 32x32 input -> 5 pooling layers -> 1x1x512
        self.fc1_weight, self.fc1_bias = self._init_linear_weight(512, 512)
        self.fc2_weight, self.fc2_bias = self._init_linear_weight(512, 512)
        self.fc3_weight, self.fc3_bias = self._init_linear_weight(512, num_classes)


def sgd_step_with_momentum(
    params: List[torch.Tensor],
    velocities: List[torch.Tensor],
    lr: float,
    momentum: float = 0.9,
    weight_decay: float = 5e-4
):
    """Momentum SGD with Weight Decay"""
    with torch.no_grad():
        for param, velocity in zip(params, velocities):
            if param.grad is not None:
                # Weight decay
                param.grad.add_(param, alpha=weight_decay)

                # Momentum update
                velocity.mul_(momentum).add_(param.grad)
                param.sub_(velocity, alpha=lr)


def train_epoch(
    model: VGGLowLevel,
    dataloader,
    lr: float = 0.01,
    momentum: float = 0.9,
    weight_decay: float = 5e-4
) -> Tuple[float, float]:
    """Train for one epoch"""
    # Initialize velocity (first epoch)
    if not hasattr(train_epoch, 'velocities') or len(train_epoch.velocities) != len(model.parameters()):
        train_epoch.velocities = [torch.zeros_like(p) for p in model.parameters()]

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in dataloader:
        images = images.to(model.device)
        labels = labels.to(model.device)

        # Forward
        logits = model.forward(images, training=True)

        # Loss
        loss = F.cross_entropy(logits, labels)

        # Backward
        model.zero_grad()
        loss.backward()

        # Update
        sgd_step_with_momentum(
            model.parameters(),
            train_epoch.velocities,
            lr, momentum, weight_decay
        )

        # Metrics
        total_loss += loss.item() * images.size(0)
        predictions = logits.argmax(dim=1)
        total_correct += (predictions == labels).sum().item()
        total_samples += images.size(0)

    return total_loss / total_samples, total_correct / total_samples


@torch.no_grad()
def evaluate(model: VGGLowLevel, dataloader) -> Tuple[float, float]:
    """Evaluate"""
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in dataloader:
        images = images.to(model.device)
        labels = labels.to(model.device)

        logits = model.forward(images, training=False)
        loss = F.cross_entropy(logits, labels)

        total_loss += loss.item() * images.size(0)
        predictions = logits.argmax(dim=1)
        total_correct += (predictions == labels).sum().item()
        total_samples += images.size(0)

    return total_loss / total_samples, total_correct / total_samples


def visualize_features(model: VGGLowLevel, image: torch.Tensor) -> List[torch.Tensor]:
    """
    Extract feature maps from each block

    Returns:
        List of feature maps after each conv block (before pooling)
    """
    features = []
    x = image.to(model.device)

    for i, params in enumerate(model.conv_params):
        if params == 'M':
            features.append(x.clone())  # Save before pooling
            x = F.max_pool2d(x, kernel_size=2, stride=2)
        else:
            x = F.conv2d(x, params['weight'], params['bias'],
                        stride=1, padding=1)
            x = F.relu(x)

    features.append(x)  # Last block
    return features


def main():
    """VGG training demo with CIFAR-10"""
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader

    print("=== VGG Low-Level Training (CIFAR-10) ===\n")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Data preprocessing
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])

    # CIFAR-10 dataset
    train_dataset = datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train
    )
    test_dataset = datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test
    )

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=2)

    print(f"Train samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}\n")

    # Model (small VGG for CIFAR)
    model = VGGSmall(config_name='VGG16', num_classes=10, use_bn=True)
    model.to(device)

    print(f"VGG16-BN for CIFAR-10")
    print(f"Total parameters: {model.count_parameters():,}\n")

    # Training
    epochs = 100
    lr = 0.1

    for epoch in range(epochs):
        # Learning rate schedule
        if epoch in [30, 60, 80]:
            lr *= 0.1
            print(f"LR -> {lr}")

        train_loss, train_acc = train_epoch(model, train_loader, lr)

        # Evaluate every 10 epochs
        if (epoch + 1) % 10 == 0:
            test_loss, test_acc = evaluate(model, test_loader)
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
            print(f"  Test  - Loss: {test_loss:.4f}, Acc: {test_acc:.4f}\n")

    # Final evaluation
    final_loss, final_acc = evaluate(model, test_loader)
    print(f"Final Test Accuracy: {final_acc:.4f}")


if __name__ == "__main__":
    main()
