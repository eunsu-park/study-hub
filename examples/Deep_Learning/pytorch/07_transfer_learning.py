"""
07. Transfer Learning

Implements transfer learning using pretrained models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

print("=" * 60)
print("PyTorch Transfer Learning")
print("=" * 60)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device in use: {device}")


# ============================================
# 1. Loading Pretrained Models
# ============================================
print("\n[1] Loading Pretrained Models")
print("-" * 40)

try:
    import torchvision.models as models

    # Various pretrained models
    print("Available pretrained models:")
    pretrained_models = {
        'ResNet-18': lambda: models.resnet18(weights='IMAGENET1K_V1'),
        'ResNet-50': lambda: models.resnet50(weights='IMAGENET1K_V2'),
        'EfficientNet-B0': lambda: models.efficientnet_b0(weights='IMAGENET1K_V1'),
        'MobileNet-V2': lambda: models.mobilenet_v2(weights='IMAGENET1K_V1'),
    }

    for name, loader in pretrained_models.items():
        model = loader()
        params = sum(p.numel() for p in model.parameters())
        print(f"  {name}: {params:,} parameters")

    TORCHVISION_AVAILABLE = True
except ImportError:
    print("torchvision is not installed. Proceeding in demo mode.")
    TORCHVISION_AVAILABLE = False


# ============================================
# 2. Feature Extraction
# ============================================
print("\n[2] Feature Extraction")
print("-" * 40)

if TORCHVISION_AVAILABLE:
    # Load ResNet-18
    model = models.resnet18(weights='IMAGENET1K_V1')

    # Check original classifier
    print(f"Original FC layer: {model.fc}")

    # Freeze all weights
    for param in model.parameters():
        param.requires_grad = False

    # Replace last layer
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_features, 10)  # 10 classes
    )

    print(f"New FC layer: {model.fc}")

    # Check trainable parameters
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")


# ============================================
# 3. Fine-tuning
# ============================================
print("\n[3] Fine-tuning")
print("-" * 40)

if TORCHVISION_AVAILABLE:
    # Load new model
    model = models.resnet18(weights='IMAGENET1K_V1')

    # Replace last layer
    model.fc = nn.Linear(model.fc.in_features, 10)

    # All parameters trainable (default)
    print("Full fine-tuning:")
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {trainable:,}")


# ============================================
# 4. Gradual Unfreezing
# ============================================
print("\n[4] Gradual Unfreezing")
print("-" * 40)

if TORCHVISION_AVAILABLE:
    model = models.resnet18(weights='IMAGENET1K_V1')

    # Stage 1: Freeze all layers
    for param in model.parameters():
        param.requires_grad = False

    # Only last layer trainable
    model.fc = nn.Linear(model.fc.in_features, 10)

    def count_trainable(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Gradual unfreezing process:")
    print(f"  Stage 1 (FC only): {count_trainable(model):,} parameters")

    # Stage 2: Unfreeze layer4
    for param in model.layer4.parameters():
        param.requires_grad = True
    print(f"  Stage 2 (FC + layer4): {count_trainable(model):,} parameters")

    # Stage 3: Unfreeze layer3
    for param in model.layer3.parameters():
        param.requires_grad = True
    print(f"  Stage 3 (FC + layer4 + layer3): {count_trainable(model):,} parameters")

    # Stage 4: Unfreeze all
    for param in model.parameters():
        param.requires_grad = True
    print(f"  Stage 4 (all): {count_trainable(model):,} parameters")


# ============================================
# 5. Discriminative Learning Rates
# ============================================
print("\n[5] Discriminative Learning Rates")
print("-" * 40)

if TORCHVISION_AVAILABLE:
    model = models.resnet18(weights='IMAGENET1K_V1')
    model.fc = nn.Linear(model.fc.in_features, 10)

    # Different learning rates per layer
    optimizer = torch.optim.Adam([
        {'params': model.conv1.parameters(), 'lr': 1e-5},
        {'params': model.layer1.parameters(), 'lr': 2e-5},
        {'params': model.layer2.parameters(), 'lr': 5e-5},
        {'params': model.layer3.parameters(), 'lr': 1e-4},
        {'params': model.layer4.parameters(), 'lr': 2e-4},
        {'params': model.fc.parameters(), 'lr': 1e-3},
    ])

    print("Learning rates per layer:")
    for i, group in enumerate(optimizer.param_groups):
        print(f"  Group {i}: lr = {group['lr']}")


# ============================================
# 6. Data Preprocessing (ImageNet Normalization)
# ============================================
print("\n[6] ImageNet Normalization")
print("-" * 40)

try:
    from torchvision import transforms

    # ImageNet normalization values
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std)
    ])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std)
    ])

    print(f"ImageNet Mean: {imagenet_mean}")
    print(f"ImageNet Std: {imagenet_std}")
    print("Train transforms: RandomResizedCrop, Flip, Normalize")
    print("Val transforms: Resize, CenterCrop, Normalize")
except:
    print("transforms load failed")


# ============================================
# 7. Full Transfer Learning Pipeline
# ============================================
print("\n[7] Full Transfer Learning Pipeline")
print("-" * 40)

class TransferLearningPipeline:
    """Transfer Learning Pipeline"""

    def __init__(self, backbone='resnet18', num_classes=10, strategy='finetune'):
        self.strategy = strategy

        if TORCHVISION_AVAILABLE:
            # Load backbone
            if backbone == 'resnet18':
                self.model = models.resnet18(weights='IMAGENET1K_V1')
                in_features = self.model.fc.in_features
                self.model.fc = nn.Linear(in_features, num_classes)
            elif backbone == 'resnet50':
                self.model = models.resnet50(weights='IMAGENET1K_V2')
                in_features = self.model.fc.in_features
                self.model.fc = nn.Linear(in_features, num_classes)
            else:
                raise ValueError(f"Unknown backbone: {backbone}")

            # Freeze weights based on strategy
            if strategy == 'feature_extract':
                self._freeze_backbone()
            elif strategy == 'finetune':
                pass  # All trainable
            elif strategy == 'gradual':
                self._freeze_backbone()
        else:
            # Simple model for demo
            self.model = nn.Sequential(
                nn.Conv2d(3, 64, 3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(64, num_classes)
            )

    def _freeze_backbone(self):
        """Freeze all layers except FC"""
        for name, param in self.model.named_parameters():
            if 'fc' not in name:
                param.requires_grad = False

    def unfreeze_layer(self, layer_name):
        """Unfreeze a specific layer"""
        layer = getattr(self.model, layer_name, None)
        if layer:
            for param in layer.parameters():
                param.requires_grad = True

    def get_optimizer(self, lr=1e-4):
        """Create optimizer"""
        if self.strategy == 'feature_extract':
            # Only trainable parameters
            params = filter(lambda p: p.requires_grad, self.model.parameters())
            return torch.optim.Adam(params, lr=lr)
        else:
            return torch.optim.Adam(self.model.parameters(), lr=lr)

    def summary(self):
        """Model summary"""
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        print(f"Strategy: {self.strategy}")
        print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

# Test
print("\nStrategy comparison:")
for strategy in ['feature_extract', 'finetune']:
    print(f"\n{strategy}:")
    pipeline = TransferLearningPipeline('resnet18', 10, strategy)
    pipeline.summary()


# ============================================
# 8. Training Example (Dummy Data)
# ============================================
print("\n[8] Training Example (Dummy Data)")
print("-" * 40)

# Generate dummy data
X_train = torch.randn(100, 3, 224, 224)
y_train = torch.randint(0, 10, (100,))

train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# Pipeline setup
pipeline = TransferLearningPipeline('resnet18', 10, 'feature_extract')
model = pipeline.model.to(device)
optimizer = pipeline.get_optimizer(lr=1e-3)
criterion = nn.CrossEntropyLoss()

# Simple training
model.train()
for epoch in range(2):
    epoch_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f"Epoch {epoch+1}: Loss = {epoch_loss/len(train_loader):.4f}")


# ============================================
# 9. Transfer Learning Checklist
# ============================================
print("\n[9] Transfer Learning Checklist")
print("-" * 40)

checklist = """
- Pretrained model selection
  - Choose a model trained on data similar to your task
  - ImageNet models work well in most cases

- Preprocessing
  - Use ImageNet normalization
  - Match model input size (usually 224x224)

- Strategy selection
  - Little data: Feature extraction (train FC only)
  - Enough data: Fine-tuning (train everything)
  - In between: Gradual unfreezing

- Learning rate
  - Feature extraction: 1e-3 ~ 1e-2
  - Fine-tuning: 1e-5 ~ 1e-4
  - Consider discriminative learning rates

- Regularization
  - Dropout, Weight Decay
  - Data augmentation
  - Early stopping

- Mode switching
  - Training: model.train()
  - Evaluation: model.eval()
"""
print(checklist)


# ============================================
# Summary
# ============================================
print("\n" + "=" * 60)
print("Transfer Learning Summary")
print("=" * 60)

summary = """
Transfer Learning Strategies:

1. Feature Extraction
   - Freeze pretrained weights
   - Train only the last layer
   - Suitable when data is scarce

2. Fine-tuning
   - Train the entire network
   - Use low learning rates
   - When data is sufficient

3. Gradual Unfreezing
   - Sequentially unfreeze from later layers
   - A balanced approach

Key Code:
    # Freeze weights
    for param in model.parameters():
        param.requires_grad = False

    # Replace last layer
    model.fc = nn.Linear(in_features, num_classes)

    # ImageNet normalization
    transforms.Normalize([0.485, 0.456, 0.406],
                        [0.229, 0.224, 0.225])
"""
print(summary)
print("=" * 60)
