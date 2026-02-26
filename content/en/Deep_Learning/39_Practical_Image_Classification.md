[Previous: Object Detection](./38_Object_Detection.md) | [Next: Practical Text Classification Project](./40_Practical_Text_Classification.md)

---

# 39. Practical Image Classification Project

## Learning Objectives

- Complete CIFAR-10 classification project
- Data augmentation strategies
- Build training pipeline
- Apply performance improvement techniques

---

## 1. Project Overview

### CIFAR-10 Dataset

```
- 60,000 images (32×32 RGB)
- 10 classes: airplane, automobile, bird, cat, deer,
             dog, frog, horse, ship, truck
- Training: 50,000 / Testing: 10,000
```

### Target Accuracy

| Model | Target Accuracy |
|-------|----------------|
| Simple CNN | 70-75% |
| ResNet-like | 85-90% |
| Transfer Learning | 90%+ |

---

## 2. Data Preparation

### Loading and Preprocessing

```python
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Normalization values (CIFAR-10)
mean = (0.4914, 0.4822, 0.4465)
std = (0.2470, 0.2435, 0.2616)

# Training transforms (with augmentation)
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

# Test transforms (no augmentation)
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

# Datasets
train_data = datasets.CIFAR10('data', train=True, download=True,
                               transform=train_transform)
test_data = datasets.CIFAR10('data', train=False,
                              transform=test_transform)

# Loaders
train_loader = DataLoader(train_data, batch_size=128, shuffle=True,
                          num_workers=4, pin_memory=True)
test_loader = DataLoader(test_data, batch_size=256)
```

---

## 3. Model Definition

### Basic CNN

```python
class CIFAR10CNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: 32 → 16
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),

            # Block 2: 16 → 8
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),

            # Block 3: 8 → 4
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
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
```

### ResNet Block

```python
class ResBlock(nn.Module):
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
```

---

## 4. Training Pipeline

### Complete Code

```python
def train_cifar10():
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    epochs = 100
    lr = 0.1

    # Model
    model = CIFAR10CNN().to(device)

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Optimizer
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=0.9,
        weight_decay=5e-4
    )

    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs
    )

    # Training
    best_acc = 0
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0
        correct = 0
        total = 0

        for data, target in train_loader:
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)

        train_acc = 100. * correct / total

        # Test
        model.eval()
        test_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += criterion(output, target).item()
                pred = output.argmax(dim=1)
                correct += (pred == target).sum().item()
                total += target.size(0)

        test_acc = 100. * correct / total

        scheduler.step()

        print(f"Epoch {epoch+1}: Train Acc={train_acc:.2f}%, Test Acc={test_acc:.2f}%")

        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), 'best_model.pth')

    print(f"\nBest test accuracy: {best_acc:.2f}%")
```

---

## 5. Performance Improvement Techniques

### Mixup

```python
def mixup_data(x, y, alpha=0.2):
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# In the training loop
mixed_x, y_a, y_b, lam = mixup_data(data, target)
output = model(mixed_x)
loss = mixup_criterion(criterion, output, y_a, y_b, lam)
```

### CutMix

```python
def cutmix_data(x, y, alpha=1.0):
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    # Random box
    W, H = x.size(3), x.size(2)
    cut_w = int(W * np.sqrt(1 - lam))
    cut_h = int(H * np.sqrt(1 - lam))
    cx, cy = np.random.randint(W), np.random.randint(H)

    x1 = np.clip(cx - cut_w // 2, 0, W)
    x2 = np.clip(cx + cut_w // 2, 0, W)
    y1 = np.clip(cy - cut_h // 2, 0, H)
    y2 = np.clip(cy + cut_h // 2, 0, H)

    x[:, :, y1:y2, x1:x2] = x[index, :, y1:y2, x1:x2]
    lam = 1 - (x2 - x1) * (y2 - y1) / (W * H)

    return x, y, y[index], lam
```

### Label Smoothing

```python
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
```

---

## 6. Result Analysis

### Confusion Matrix

```python
from sklearn.metrics import confusion_matrix
import seaborn as sns

def plot_confusion_matrix(model, test_loader, classes):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for data, target in test_loader:
            output = model(data.to(device))
            preds = output.argmax(dim=1).cpu()
            all_preds.extend(preds.tolist())
            all_labels.extend(target.tolist())

    cm = confusion_matrix(all_labels, all_preds)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
```

### Per-Class Accuracy

```python
def per_class_accuracy(model, test_loader, classes):
    model.eval()
    class_correct = [0] * 10
    class_total = [0] * 10

    with torch.no_grad():
        for data, target in test_loader:
            output = model(data.to(device))
            preds = output.argmax(dim=1).cpu()
            for i in range(len(target)):
                label = target[i].item()
                class_total[label] += 1
                if preds[i] == label:
                    class_correct[label] += 1

    for i, cls in enumerate(classes):
        acc = 100 * class_correct[i] / class_total[i]
        print(f"{cls}: {acc:.2f}%")
```

---

## 7. Transfer Learning Application

```python
import torchvision.models as models

# Pre-trained model
model = models.resnet18(weights='IMAGENET1K_V1')

# Modify last layer
model.fc = nn.Linear(model.fc.in_features, 10)

# Modify first Conv (CIFAR: 32×32)
model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
model.maxpool = nn.Identity()  # Remove pooling

model = model.to(device)
```

---

## Summary

### Checklist

- [ ] Apply data augmentation
- [ ] Use BatchNorm + Dropout
- [ ] Appropriate scheduler (Cosine Annealing)
- [ ] Use Weight Decay
- [ ] Consider Mixup/CutMix
- [ ] Save and analyze model

### Expected Results

| Technique | Test Accuracy |
|-----------|--------------|
| Basic CNN | 75-80% |
| + Data augmentation | 80-85% |
| + Mixup | 85-88% |
| ResNet + Transfer learning | 90%+ |

---

## Exercises

### Exercise 1: Train a Baseline CNN and Analyze Errors

1. Train `CIFAR10CNN` for 50 epochs using the standard pipeline with data augmentation.
2. After training, generate a confusion matrix using `plot_confusion_matrix`. Which class pairs are most frequently confused?
3. Use `per_class_accuracy` to identify the two hardest classes to classify. Display 5 misclassified examples from each hard class.
4. Hypothesize why those classes are difficult: is it due to visual similarity, intra-class variation, or another reason?

### Exercise 2: Compare Data Augmentation Strategies

Run three training experiments on CIFAR-10 for 30 epochs each:
1. **No augmentation**: `transforms.ToTensor()` and normalize only.
2. **Basic augmentation**: random crop and horizontal flip only.
3. **Full augmentation**: the complete `train_transform` with `ColorJitter` included.
4. Plot the test accuracy curves for all three experiments on the same graph. Quantify the improvement from each augmentation step. Which augmentation has the biggest impact and why?

### Exercise 3: Implement and Test Mixup Training

1. Implement the full Mixup training loop using `mixup_data` and `mixup_criterion`.
2. Train `CIFAR10CNN` for 50 epochs with `alpha=0.2`, `alpha=0.5`, and `alpha=1.0`.
3. Compare final test accuracy for each alpha value.
4. Visualize 4 mixed training examples for `alpha=0.2`. Explain conceptually why Mixup acts as a regularizer: what does it prevent the model from doing near training examples?

### Exercise 4: Transfer Learning vs Training from Scratch

Using the ResNet-18 transfer learning setup:
1. Train ResNet-18 with ImageNet pre-trained weights for 30 epochs.
2. Train ResNet-18 from scratch (random initialization) for the same 30 epochs.
3. Compare test accuracy and training loss curves. Which converges faster?
4. Experiment: freeze all ResNet layers except the final `fc` layer for 10 epochs (feature extraction), then unfreeze and fine-tune for 20 more epochs. How does this staged approach compare to fine-tuning all layers from the start?

### Exercise 5: Combine All Techniques for Maximum Accuracy

Build the best CIFAR-10 classifier you can by combining:
1. A ResNet-like architecture with residual blocks (`ResBlock`).
2. Full data augmentation (`RandomCrop`, `RandomHorizontalFlip`, `ColorJitter`).
3. CutMix or Mixup augmentation during training.
4. Label smoothing (`CrossEntropyLoss(label_smoothing=0.1)`).
5. Cosine annealing learning rate scheduler.

Train for 100 epochs and report your final test accuracy. Create a table comparing all techniques you tried in this lesson and their contribution to the final result.

---

## Next Steps

Proceed with text classification project in [40_Practical_Text_Classification.md](./40_Practical_Text_Classification.md).
