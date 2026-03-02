"""
13. Practical Image Classification Project (CIFAR-10)

Implements a full training pipeline for CIFAR-10 classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import time

print("=" * 60)
print("CIFAR-10 Image Classification Project")
print("=" * 60)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")


# ============================================
# 1. Data Preparation
# ============================================
print("\n[1] Data Preparation")
print("-" * 40)

try:
    from torchvision import datasets, transforms

    # CIFAR-10 normalization values
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)

    # Training transforms (data augmentation)
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    # Test transforms
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    # Load datasets
    train_data = datasets.CIFAR10('data', train=True, download=True,
                                   transform=train_transform)
    test_data = datasets.CIFAR10('data', train=False,
                                  transform=test_transform)

    train_loader = DataLoader(train_data, batch_size=128, shuffle=True,
                              num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_data, batch_size=256)

    classes = ('airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')

    print(f"Training data: {len(train_data)}")
    print(f"Test data: {len(test_data)}")
    print(f"Classes: {classes}")

    DATA_AVAILABLE = True

except Exception as e:
    print(f"Data loading failed: {e}")
    print("Proceeding with dummy data.")
    DATA_AVAILABLE = False


# ============================================
# 2. Model Definition
# ============================================
print("\n[2] Model Definition")
print("-" * 40)

class CIFAR10CNN(nn.Module):
    """CNN for CIFAR-10"""
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: 32 -> 16
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),

            # Block 2: 16 -> 8
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),

            # Block 3: 8 -> 4
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

class ResBlock(nn.Module):
    """Residual Block"""
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)

class ResNetCIFAR(nn.Module):
    """ResNet for CIFAR"""
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256, num_classes)

    def _make_layer(self, in_ch, out_ch, num_blocks, stride):
        layers = [ResBlock(in_ch, out_ch, stride)]
        for _ in range(1, num_blocks):
            layers.append(ResBlock(out_ch, out_ch))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# Create models
model = CIFAR10CNN().to(device)
print(f"CIFAR10CNN parameters: {sum(p.numel() for p in model.parameters()):,}")

resnet = ResNetCIFAR().to(device)
print(f"ResNetCIFAR parameters: {sum(p.numel() for p in resnet.parameters()):,}")


# ============================================
# 3. Mixup Data Augmentation
# ============================================
print("\n[3] Mixup Data Augmentation")
print("-" * 40)

def mixup_data(x, y, alpha=0.2):
    """Mixup data augmentation"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Mixup loss"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# Test
x = torch.randn(4, 3, 32, 32)
y = torch.tensor([0, 1, 2, 3])
mixed_x, y_a, y_b, lam = mixup_data(x, y, alpha=0.2)
print(f"Mixup lambda: {lam:.4f}")


# ============================================
# 4. Training Functions
# ============================================
print("\n[4] Training Functions")
print("-" * 40)

def train_epoch(model, loader, optimizer, criterion, use_mixup=False):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for data, target in loader:
        data, target = data.to(device), target.to(device)

        if use_mixup:
            data, target_a, target_b, lam = mixup_data(data, target)

        optimizer.zero_grad()
        output = model(data)

        if use_mixup:
            loss = mixup_criterion(criterion, output, target_a, target_b, lam)
        else:
            loss = criterion(output, target)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if not use_mixup:
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)

    avg_loss = total_loss / len(loader)
    accuracy = 100. * correct / total if total > 0 else 0
    return avg_loss, accuracy

def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)

            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)

    avg_loss = total_loss / len(loader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy


# ============================================
# 5. Full Training Pipeline
# ============================================
print("\n[5] Training Run")
print("-" * 40)

def train_model(model, train_loader, test_loader, epochs=10, use_mixup=False):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.1,
        momentum=0.9,
        weight_decay=5e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []}
    best_acc = 0

    for epoch in range(epochs):
        start_time = time.time()

        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, use_mixup
        )
        test_loss, test_acc = evaluate(model, test_loader, criterion)

        scheduler.step()

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)

        elapsed = time.time() - start_time

        if test_acc > best_acc:
            best_acc = test_acc

        print(f"Epoch {epoch+1:3d}: Train Acc={train_acc:5.2f}%, "
              f"Test Acc={test_acc:5.2f}%, Time={elapsed:.1f}s")

    print(f"\nBest test accuracy: {best_acc:.2f}%")
    return history

if DATA_AVAILABLE:
    # Short training (demo)
    model = CIFAR10CNN().to(device)
    history = train_model(model, train_loader, test_loader, epochs=5)
else:
    print("No data available - skipping training")
    history = None


# ============================================
# 6. Result Visualization
# ============================================
print("\n[6] Result Visualization")
print("-" * 40)

if history:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history['train_loss'], label='Train')
    axes[0].plot(history['test_loss'], label='Test')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history['train_acc'], label='Train')
    axes[1].plot(history['test_acc'], label='Test')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('cifar10_training.png', dpi=100)
    plt.close()
    print("Plot saved: cifar10_training.png")


# ============================================
# 7. Per-Class Accuracy
# ============================================
print("\n[7] Per-Class Analysis")
print("-" * 40)

if DATA_AVAILABLE:
    def per_class_accuracy(model, loader, classes):
        model.eval()
        class_correct = [0] * len(classes)
        class_total = [0] * len(classes)

        with torch.no_grad():
            for data, target in loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                pred = output.argmax(dim=1)

                for i in range(len(target)):
                    label = target[i].item()
                    class_total[label] += 1
                    if pred[i] == label:
                        class_correct[label] += 1

        print("Per-class accuracy:")
        for i, cls in enumerate(classes):
            if class_total[i] > 0:
                acc = 100 * class_correct[i] / class_total[i]
                print(f"  {cls:12s}: {acc:5.2f}%")

    per_class_accuracy(model, test_loader, classes)


# ============================================
# 8. Prediction Visualization
# ============================================
print("\n[8] Prediction Visualization")
print("-" * 40)

if DATA_AVAILABLE:
    def visualize_predictions(model, loader, classes, n=8):
        model.eval()
        data, target = next(iter(loader))
        data, target = data[:n].to(device), target[:n]

        with torch.no_grad():
            output = model(data)
            pred = output.argmax(dim=1)

        fig, axes = plt.subplots(2, 4, figsize=(12, 6))
        for i, ax in enumerate(axes.flat):
            if i < n:
                img = data[i].cpu().numpy().transpose(1, 2, 0)
                # Inverse normalization
                img = img * np.array(std) + np.array(mean)
                img = np.clip(img, 0, 1)

                ax.imshow(img)
                color = 'green' if pred[i] == target[i] else 'red'
                ax.set_title(f"Pred: {classes[pred[i]]}\nTrue: {classes[target[i]]}",
                            color=color)
                ax.axis('off')

        plt.tight_layout()
        plt.savefig('cifar10_predictions.png', dpi=100)
        plt.close()
        print("Prediction visualization saved: cifar10_predictions.png")

    visualize_predictions(model, test_loader, classes)


# ============================================
# Summary
# ============================================
print("\n" + "=" * 60)
print("CIFAR-10 Project Summary")
print("=" * 60)

summary = """
Key techniques:

1. Data Augmentation
   - RandomCrop, HorizontalFlip
   - ColorJitter
   - Mixup/CutMix

2. Model Architecture
   - Conv-BN-ReLU blocks
   - Dropout2d, Dropout
   - ResNet blocks (Skip Connection)

3. Training Settings
   - SGD + Momentum + Weight Decay
   - Cosine Annealing LR
   - Label Smoothing

Expected accuracy:
   - Basic CNN: 75-80%
   - + Data augmentation: 80-85%
   - + Mixup: 85-88%
   - ResNet + Transfer learning: 90%+

Next steps:
   - Deeper models (ResNet-50)
   - AutoAugment
   - Knowledge Distillation
"""
print(summary)
print("=" * 60)
