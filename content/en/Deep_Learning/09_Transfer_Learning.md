# 09. Transfer Learning

[Previous: CNN Advanced](./08_CNN_Advanced.md) | [Next: CNN (LeNet)](./10_Impl_CNN_LeNet.md)

---

## Learning Objectives

- Understand the concept and benefits of transfer learning
- Utilize pretrained models
- Learn fine-tuning strategies
- Practical image classification project

---

## 1. What is Transfer Learning?

### Concept

Early CNN layers learn universal features (edges, textures, colors) that are useful regardless of task. Only the final layers specialize to the original dataset. By freezing early layers and retraining final layers, you leverage these universal features even with a small dataset. This works because visual features are hierarchical: edges combine into textures, textures into parts, parts into objects — and the lower levels of this hierarchy are shared across virtually all image recognition tasks.

```
Model trained on ImageNet
        ↓
    Low-level features (edges, textures) → Reuse
        ↓
    High-level features → Adapt to new data
        ↓
    New classification task
```

### Benefits

- High performance with limited data
- Faster training
- Better generalization

---

## 2. Transfer Learning Strategies

**When to fine-tune vs feature-extract?** Small dataset + similar domain (e.g., classifying dog breeds using an ImageNet model) -- freeze most layers and only train the head. Large dataset + different domain (e.g., medical X-rays) -- fine-tune all layers with a small learning rate so the network can adapt its low-level features to the new image distribution. In between, gradual unfreezing offers a safe middle ground.

### Strategy 1: Feature Extraction

```python
# Freeze: prevents gradient updates to pretrained weights, preserving
# the learned universal features (edges, textures, shapes)
for param in model.parameters():
    param.requires_grad = False

# Replace the classification head to match our number of classes;
# pretrained weights for everything else remain intact
model.fc = nn.Linear(2048, num_classes)
```

- Use pretrained features as-is
- Train only the final classification layer
- Suitable when data is limited

### Strategy 2: Fine-tuning

```python
# Unfreeze all layers for fine-tuning
for param in model.parameters():
    param.requires_grad = True

# Use a very low learning rate (1e-5) — large updates would destroy
# the pretrained features. The goal is to gently nudge the weights
# toward the new domain, not to learn from scratch.
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
```

- Start from pretrained weights
- Fine-tune the entire network
- Suitable when sufficient data available

### Strategy 3: Gradual Unfreezing

```python
# Step 1: Last layer only
for param in model.parameters():
    param.requires_grad = False
model.fc.requires_grad_(True)
train_for_epochs(5)

# Step 2: Last block too
model.layer4.requires_grad_(True)
train_for_epochs(5)

# Step 3: Entire network
model.requires_grad_(True)
train_for_epochs(10)
```

---

## 3. PyTorch Implementation

### Basic Transfer Learning

```python
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms, datasets

# 1. Load pretrained model — weights were learned from 1.2M ImageNet images
model = models.resnet50(weights='IMAGENET1K_V2')

# 2. Freeze all backbone weights — only the new head will be trained,
# so we optimize far fewer parameters (much faster, less overfitting)
for param in model.parameters():
    param.requires_grad = False

# 3. Replace the classification head to match our number of classes;
# the new layers are randomly initialized and will be the only trainable params
num_features = model.fc.in_features
model.fc = nn.Sequential(
    nn.Dropout(0.5),                    # Regularize the high-dimensional backbone output
    nn.Linear(num_features, 256),       # Reduce from 2048 to 256 — bottleneck prevents overfitting
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256, num_classes)
)
```

### Data Preprocessing

```python
# Use ImageNet normalization
normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize
])
```

---

## 4. Training Strategies

### Discriminative Learning Rates

```python
# Different learning rates for each layer group — earlier layers learn
# universal features that need minimal adjustment (tiny LR), while later
# layers and the FC head are more task-specific and need larger updates
optimizer = torch.optim.Adam([
    {'params': model.layer1.parameters(), 'lr': 1e-5},   # Edges, textures — nearly universal
    {'params': model.layer2.parameters(), 'lr': 5e-5},   # Low-level combinations
    {'params': model.layer3.parameters(), 'lr': 1e-4},   # Mid-level features
    {'params': model.layer4.parameters(), 'lr': 5e-4},   # High-level, task-specific features
    {'params': model.fc.parameters(), 'lr': 1e-3},       # New head — learns from scratch
])
```

### Learning Rate Scheduling

```python
# Warmup + Cosine Decay
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=1e-3,
    epochs=epochs,
    steps_per_epoch=len(train_loader),
    pct_start=0.1  # 10% warmup
)
```

---

## 5. Various Pretrained Models

### torchvision Models

```python
# Classification
resnet50 = models.resnet50(weights='IMAGENET1K_V2')
efficientnet = models.efficientnet_b0(weights='IMAGENET1K_V1')
vit = models.vit_b_16(weights='IMAGENET1K_V1')

# Object detection
fasterrcnn = models.detection.fasterrcnn_resnet50_fpn(weights='DEFAULT')

# Segmentation
deeplabv3 = models.segmentation.deeplabv3_resnet50(weights='DEFAULT')
```

### timm Library

```python
import timm

# Check available models
print(timm.list_models('*efficientnet*'))

# Load model
model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=10)
```

---

## 6. Practical Project: Flower Classification

### Data Preparation

```python
# Flowers102 dataset
from torchvision.datasets import Flowers102

train_data = Flowers102(
    root='data',
    split='train',
    transform=train_transform,
    download=True
)

test_data = Flowers102(
    root='data',
    split='test',
    transform=val_transform
)
```

### Model and Training

```python
class FlowerClassifier(nn.Module):
    def __init__(self, num_classes=102):
        super().__init__()
        self.backbone = models.efficientnet_b0(weights='IMAGENET1K_V1')

        # Replace last layer
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)

# Training
model = FlowerClassifier()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
```

---

## 7. Considerations

### Strategy by Data Size

| Data Size | Strategy | Description |
|-----------|---------|-------------|
| Very Small (<1000) | Feature Extraction | Train only last layer |
| Small (1000-10000) | Gradual Unfreezing | Unfreeze from later layers |
| Medium (10000+) | Full Fine-tuning | Train all with low LR |

### Domain Similarity

```
Similar to ImageNet (animals, objects):
    → Can use shallow layers as-is

Different from ImageNet (medical, satellite):
    → Need to fine-tune deeper layers
```

### Common Mistakes

1. Missing ImageNet normalization
2. Learning rate too high
3. Forgetting to switch train/eval mode
4. Including frozen weights in optimizer

---

## 8. Performance Improvement Tips

### Data Augmentation

```python
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    normalize
])
```

### Label Smoothing

```python
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
```

### Mixup / CutMix

```python
def mixup(x, y, alpha=0.2):
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0))
    mixed_x = lam * x + (1 - lam) * x[idx]
    y_a, y_b = y, y[idx]
    return mixed_x, y_a, y_b, lam
```

---

## Summary

### Core Concepts

1. **Feature Extraction**: Reuse pretrained features
2. **Fine-tuning**: Adjust entire network with low LR
3. **Gradual Unfreezing**: Sequential training from later layers

### Checklist

- [ ] Use ImageNet normalization
- [ ] Choose appropriate learning rate (1e-4 ~ 1e-5)
- [ ] Switch model.train() / model.eval()
- [ ] Apply data augmentation
- [ ] Set up early stopping

---

## Exercises

### Exercise 1: Feature Extraction vs Fine-tuning Comparison

Compare feature extraction and full fine-tuning strategies on a small dataset.

1. Use the Flowers102 dataset and limit training to 500 samples.
2. Strategy A: Freeze all layers of ResNet-18, train only the final FC layer for 10 epochs.
3. Strategy B: Unfreeze all layers and fine-tune with `lr=1e-5` for 10 epochs.
4. Record final validation accuracy for both strategies.
5. Explain the trade-off: when does feature extraction win, and when does fine-tuning win?

### Exercise 2: Gradual Unfreezing Schedule

Implement the three-stage gradual unfreezing strategy from the lesson.

1. Load a pretrained ResNet-18.
2. Stage 1 (epochs 1-5): Train only the final FC layer.
3. Stage 2 (epochs 6-10): Also unfreeze `layer4`.
4. Stage 3 (epochs 11-20): Unfreeze all layers with `lr=1e-5`.
5. Plot validation accuracy across all 20 epochs, marking the stage boundaries with vertical lines.

### Exercise 3: Discriminative Learning Rates

Apply discriminative learning rates (different LR per layer group) and observe the effect.

1. Load a pretrained ResNet-18.
2. Set up an optimizer with 5 parameter groups: `layer1` (lr=1e-5), `layer2` (lr=3e-5), `layer3` (lr=1e-4), `layer4` (lr=3e-4), `fc` (lr=1e-3).
3. Train for 15 epochs on CIFAR-10 (use a small subset of 2000 samples).
4. Compare against training with a uniform lr=1e-4 for the same epochs.
5. Explain the intuition: why should earlier layers use smaller learning rates?

### Exercise 4: Domain Gap Investigation

Explore how domain similarity between source (ImageNet) and target affects transfer learning quality.

1. Choose two target datasets: CIFAR-10 (natural images, similar to ImageNet) and a medical or texture dataset from `torchvision.datasets` or a custom folder.
2. For each dataset, compare performance of: (a) training from scratch, (b) feature extraction only, (c) full fine-tuning.
3. Present results in a table.
4. Explain why the domain gap causes different strategies to win for different datasets.

### Exercise 5: ImageNet Normalization — What Happens Without It

Empirically verify that ImageNet normalization is critical for transfer learning.

1. Load a pretrained EfficientNet-B0.
2. Run two experiments on a 100-sample evaluation set: (a) with the correct ImageNet normalization `mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]`, (b) with no normalization (or wrong normalization like `mean=0.5, std=0.5`).
3. Compare the top-1 accuracy for both.
4. Visualize a sample image under both preprocessing schemes and explain why the shifted input distribution breaks the pretrained model's feature detectors.

---

## Next Steps

In [13_RNN_Basics.md](./13_RNN_Basics.md), we'll learn recurrent neural networks.
