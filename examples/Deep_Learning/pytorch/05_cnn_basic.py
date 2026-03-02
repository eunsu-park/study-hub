"""
05. CNN Basics - PyTorch Version

Implements Convolutional Neural Networks (CNNs) in PyTorch.
Performs MNIST and CIFAR-10 classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

print("=" * 60)
print("PyTorch CNN Basics")
print("=" * 60)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device in use: {device}")


# ============================================
# 1. Understanding Convolution Operations
# ============================================
print("\n[1] Understanding Convolution Operations")
print("-" * 40)

# Conv2d basics
conv = nn.Conv2d(
    in_channels=1,    # Input channels
    out_channels=3,   # Number of filters (output channels)
    kernel_size=3,    # Filter size
    stride=1,         # Step size
    padding=1         # Padding
)

print(f"Conv2d parameters:")
print(f"  weight shape: {conv.weight.shape}")  # (out, in, H, W)
print(f"  bias shape: {conv.bias.shape}")       # (out,)

# Input/output check
x = torch.randn(1, 1, 8, 8)  # (batch, channel, H, W)
out = conv(x)
print(f"\nInput: {x.shape} -> Output: {out.shape}")


# Output size calculation
def calc_output_size(input_size, kernel_size, stride=1, padding=0):
    return (input_size - kernel_size + 2 * padding) // stride + 1

print("\nOutput size formula: (input - kernel + 2*padding) / stride + 1")
for k, s, p in [(3, 1, 0), (3, 1, 1), (3, 2, 0), (5, 1, 2)]:
    out_size = calc_output_size(32, k, s, p)
    print(f"  input=32, kernel={k}, stride={s}, pad={p} -> output={out_size}")


# ============================================
# 2. Pooling Operations
# ============================================
print("\n[2] Pooling Operations")
print("-" * 40)

# MaxPool2d
pool = nn.MaxPool2d(kernel_size=2, stride=2)

x = torch.tensor([[[[1, 2, 3, 4],
                    [5, 6, 7, 8],
                    [9, 10, 11, 12],
                    [13, 14, 15, 16]]]], dtype=torch.float32)

print(f"Input:\n{x.squeeze()}")
print(f"\nMaxPool2d(2,2) output:\n{pool(x).squeeze()}")

# AvgPool2d
avg_pool = nn.AvgPool2d(2, 2)
print(f"\nAvgPool2d(2,2) output:\n{avg_pool(x).squeeze()}")


# ============================================
# 3. MNIST CNN
# ============================================
print("\n[3] MNIST CNN")
print("-" * 40)

class MNISTNet(nn.Module):
    """Simple CNN for MNIST"""
    def __init__(self):
        super().__init__()
        # Conv block 1: 1->32 channels, 28->14
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)

        # Conv block 2: 32->64 channels, 14->7
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)

        # FC block
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

model = MNISTNet()
print(model)

# Parameter count
total = sum(p.numel() for p in model.parameters())
print(f"\nTotal parameters: {total:,}")


# ============================================
# 4. MNIST Training
# ============================================
print("\n[4] MNIST Training")
print("-" * 40)

# Load data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

try:
    train_data = datasets.MNIST('data', train=True, download=True, transform=transform)
    test_data = datasets.MNIST('data', train=False, transform=transform)

    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=1000)

    print(f"Training data: {len(train_data)} samples")
    print(f"Test data: {len(test_data)} samples")

    # Model, loss, optimizer
    model = MNISTNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training
    epochs = 3
    train_losses = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        acc = 100. * correct / total
        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)
        print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Acc={acc:.2f}%")

    # Test
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    print(f"\nTest accuracy: {100. * correct / total:.2f}%")

except Exception as e:
    print(f"MNIST load failed (offline?): {e}")
    print("Proceeding in demo mode.")

    # Test with dummy data
    x_dummy = torch.randn(4, 1, 28, 28)
    model = MNISTNet()
    out = model(x_dummy)
    print(f"Dummy input: {x_dummy.shape} -> Output: {out.shape}")


# ============================================
# 5. Feature Map Visualization
# ============================================
print("\n[5] Feature Map Visualization")
print("-" * 40)

def visualize_feature_maps(model, image, layer_name='conv1'):
    """Visualize feature maps"""
    model.eval()

    # Capture intermediate output using hook
    activations = {}
    def hook_fn(module, input, output):
        activations['output'] = output.detach()

    hook = getattr(model, layer_name).register_forward_hook(hook_fn)

    with torch.no_grad():
        model(image)

    hook.remove()
    feature_maps = activations['output']

    # Visualization
    n_maps = min(16, feature_maps.shape[1])
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))

    for i, ax in enumerate(axes.flat):
        if i < n_maps:
            ax.imshow(feature_maps[0, i].cpu().numpy(), cmap='viridis')
        ax.axis('off')

    plt.suptitle(f'{layer_name} Feature Maps')
    plt.tight_layout()
    plt.savefig('cnn_feature_maps.png', dpi=100)
    plt.close()
    print(f"Feature maps saved: cnn_feature_maps.png")

# Visualize (if trained model is available)
try:
    sample_image = train_data[0][0].unsqueeze(0).to(device)
    visualize_feature_maps(model, sample_image, 'conv1')
except:
    print("Visualization skipped (no data)")


# ============================================
# 6. Filter Visualization
# ============================================
print("\n[6] Filter Visualization")
print("-" * 40)

def visualize_filters(model, layer_name='conv1'):
    """Visualize Conv filters"""
    filters = getattr(model, layer_name).weight.detach().cpu()

    # First 16 filters
    n_filters = min(16, filters.shape[0])
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))

    for i, ax in enumerate(axes.flat):
        if i < n_filters:
            # Filter of the first input channel
            ax.imshow(filters[i, 0].numpy(), cmap='gray')
        ax.axis('off')

    plt.suptitle(f'{layer_name} Filters')
    plt.tight_layout()
    plt.savefig('cnn_filters.png', dpi=100)
    plt.close()
    print(f"Filters saved: cnn_filters.png")

try:
    visualize_filters(model, 'conv1')
except:
    print("Filter visualization skipped")


# ============================================
# 7. CIFAR-10 CNN
# ============================================
print("\n[7] CIFAR-10 CNN")
print("-" * 40)

class CIFAR10Net(nn.Module):
    """CNN for CIFAR-10"""
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: 3->64, 32->16
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),

            # Block 2: 64->128, 16->8
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),

            # Block 3: 128->256, 8->4
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

cifar_model = CIFAR10Net()
print(cifar_model)

# Parameter count
total = sum(p.numel() for p in cifar_model.parameters())
print(f"\nTotal parameters: {total:,}")

# Test
x_test = torch.randn(2, 3, 32, 32)
out = cifar_model(x_test)
print(f"Input: {x_test.shape} -> Output: {out.shape}")


# ============================================
# 8. Data Augmentation
# ============================================
print("\n[8] Data Augmentation")
print("-" * 40)

train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                        (0.2470, 0.2435, 0.2616))
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                        (0.2470, 0.2435, 0.2616))
])

print("Train transforms: RandomCrop, Flip, ColorJitter, Normalize")
print("Test transforms: ToTensor, Normalize")


# ============================================
# 9. Model Save/Load
# ============================================
print("\n[9] Model Save/Load")
print("-" * 40)

# Save
torch.save(cifar_model.state_dict(), 'cifar_cnn.pth')
print("Model saved: cifar_cnn.pth")

# Load
loaded_model = CIFAR10Net()
loaded_model.load_state_dict(torch.load('cifar_cnn.pth', weights_only=True))
loaded_model.eval()
print("Model loaded")


# ============================================
# Summary
# ============================================
print("\n" + "=" * 60)
print("CNN Basics Summary")
print("=" * 60)

summary = """
CNN Components:
1. Conv2d: Local pattern extraction
2. BatchNorm2d: Training stabilization
3. ReLU: Non-linearity
4. MaxPool2d: Spatial reduction
5. Dropout2d: Overfitting prevention
6. Flatten + Linear: Classification

Output size formula:
    output = (input - kernel + 2*padding) / stride + 1

Common pattern:
    Conv -> BN -> ReLU -> Pool (repeat) -> Flatten -> FC

Recommended settings:
- kernel_size=3, padding=1 (same padding)
- Channel increase: 64 -> 128 -> 256
- Pool for spatial reduction
- Dropout before FC
"""
print(summary)
print("=" * 60)
