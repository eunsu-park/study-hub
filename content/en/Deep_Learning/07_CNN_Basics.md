# 07. CNN Basics (Convolutional Neural Networks)

[Previous: Multi-Layer Perceptron (MLP)](./06_Impl_MLP.md) | [Next: CNN Advanced](./08_CNN_Advanced.md)

---

## Learning Objectives

- Understand the principles of convolution operations
- Learn pooling, padding, and stride concepts
- Implement CNNs with PyTorch
- Classify MNIST/CIFAR-10 datasets

---

## 1. Convolution Operation

### Concept

Images have local structure — nearby pixels are correlated. A fully connected layer ignores this by treating the image as a flat vector. Convolutions exploit locality by using small filters that slide across the image, dramatically reducing parameters while capturing spatial patterns. This is why CNNs build a **spatial hierarchy**: early layers detect edges and textures in small neighborhoods, and deeper layers compose these into complex shapes and objects.

Detects local patterns in images (edges, textures).

```
Input Image     Filter(Kernel)     Output
[1 2 3 4]       [1 0]              [?]
[5 6 7 8]  *    [0 1]   =
[9 0 1 2]
```

### Formula

```
Output[i,j] = Σ Σ Input[i+m, j+n] × Filter[m, n]
```

**Why does multiply-and-sum detect patterns?** The operation is a *dot product* between the filter and a local image patch. A dot product measures **similarity**: when the patch's pixel pattern aligns with the filter weights (both positive in the same places, both negative in the same places), the sum is large. When they don't match, positive and negative terms cancel out and the output is near zero. So each filter is essentially a *template* — the output map lights up wherever the image locally resembles that template.

### Dimension Calculation

```
Output size = (Input - Kernel + 2×Padding) / Stride + 1

Example: Input 32×32, Kernel 3×3, Padding 1, Stride 1
         = (32 - 3 + 2) / 1 + 1 = 32
```

---

## 2. Key Concepts

### Padding

```
Add zeros to input borders to maintain output size

padding='same': Output = Input size
padding='valid': No padding (Output < Input)
```

### Stride

```
Filter movement interval

stride=1: Move one pixel at a time (default)
stride=2: Move two pixels at a time → Output size halved
```

### Pooling

Progressive downsampling provides translation invariance (a shifted cat is still a cat) and reduces computation for deeper layers. By discarding precise positional information within each local region, the network focuses on *whether* a feature was detected rather than *exactly where*.

```
Reduce spatial size, increase invariance

Max Pooling: Maximum value in region
Avg Pooling: Average value in region
```

Max pooling selects the strongest activation in each region — you can think of it as asking "was this feature detected *anywhere* in this neighborhood?" Mathematically, for a 2x2 pool window: `y = max(x_{i,j}, x_{i+1,j}, x_{i,j+1}, x_{i+1,j+1})`. The max operation is a piecewise-linear function whose gradient is 1 for the winning element and 0 for the rest — only the most activated position gets updated during backprop.

---

## 3. CNN Architecture

### Basic Structure

```
Input → [Conv → ReLU → Pool] × N → Flatten → FC → Output
```

### LeNet-5 (1998)

```
Input (32×32×1)
  ↓
Conv1 (5×5, 6 channels) → 28×28×6
  ↓
MaxPool (2×2) → 14×14×6
  ↓
Conv2 (5×5, 16 channels) → 10×10×16
  ↓
MaxPool (2×2) → 5×5×16
  ↓
Flatten → 400
  ↓
FC → 120 → 84 → 10
```

---

## 4. PyTorch Conv2d

### Basic Usage

```python
import torch.nn as nn

# Conv2d(in_channels, out_channels, kernel_size, stride, padding)
conv = nn.Conv2d(
    in_channels=3,      # RGB image
    out_channels=64,    # 64 filters
    kernel_size=3,      # 3×3 kernel
    stride=1,
    padding=1           # same padding
)

# Input: (batch, channels, height, width)
x = torch.randn(1, 3, 32, 32)
out = conv(x)  # (1, 64, 32, 32)
```

### MaxPool2d

```python
pool = nn.MaxPool2d(kernel_size=2, stride=2)
# 32×32 → 16×16
```

---

## 5. MNIST CNN Implementation

### Model Definition

```python
class MNISTNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Conv block 1
        # kernel_size=3: 3×3 is the smallest kernel that captures all 8 neighbors
        # — the sweet spot between expressiveness and efficiency (VGG principle)
        # padding=1: "Same" padding so output spatial size equals input,
        # preventing information loss at borders
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)

        # Conv block 2
        # Double the channels (32→64): deeper layers need more filters to
        # represent the combinatorial explosion of higher-level features
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)

        # FC block
        # 64*7*7 = 3136: after two 2×2 pools, 28→14→7 spatially, with 64 channels
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # x: (batch, 1, 28, 28)
        x = F.relu(self.conv1(x))  # (batch, 32, 28, 28)
        x = self.pool1(x)          # (batch, 32, 14, 14)

        x = F.relu(self.conv2(x))  # (batch, 64, 14, 14)
        x = self.pool2(x)          # (batch, 64, 7, 7)

        x = x.view(-1, 64 * 7 * 7) # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
```

### Training Code

```python
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Load data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_data = datasets.MNIST('data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)

# Model, loss, optimizer
model = MNISTNet()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training
for epoch in range(5):
    for images, labels in train_loader:
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

---

## 6. Feature Map Visualization

```python
def visualize_feature_maps(model, image):
    """Visualize feature maps from the first Conv layer"""
    model.eval()
    with torch.no_grad():
        # First Conv output
        x = model.conv1(image)
        x = F.relu(x)

    # Display in grid
    fig, axes = plt.subplots(4, 8, figsize=(12, 6))
    for i, ax in enumerate(axes.flat):
        if i < x.shape[1]:
            ax.imshow(x[0, i].cpu().numpy(), cmap='viridis')
        ax.axis('off')
    plt.tight_layout()
    plt.savefig('feature_maps.png')
```

---

## 7. Understanding Convolution with NumPy (Reference)

```python
def conv2d_numpy(image, kernel):
    """2D convolution implementation with NumPy (educational)"""
    h, w = image.shape
    kh, kw = kernel.shape
    oh, ow = h - kh + 1, w - kw + 1

    output = np.zeros((oh, ow))

    for i in range(oh):
        for j in range(ow):
            # Extract region
            region = image[i:i+kh, j:j+kw]
            # Element-wise multiplication and sum
            output[i, j] = np.sum(region * kernel)

    return output

# Sobel edge detection example
# Why these specific values?  The Sobel-x kernel computes a weighted
# horizontal difference: the right column is positive (+1, +2, +1) and
# the left column is negative (-1, -2, -1).  When convolved with a
# region that transitions from dark (left) to bright (right), the
# positives dominate → large positive output.  In a uniform region,
# left and right cancel → output ≈ 0.  The center row has double
# weight (±2) to emphasize the pixel directly adjacent to the edge.
sobel_x = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]])

edges = conv2d_numpy(image, sobel_x)
```

> **Note**: In actual CNNs, use PyTorch's optimized implementation. The key insight is that in a trained CNN, the network *learns* filter values via backpropagation — just as Sobel's hand-crafted weights detect edges, learned filters automatically discover the patterns most useful for the task.

---

## 8. Batch Normalization and Dropout

### Usage in CNNs

```python
class CNNWithBN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)  # BN for Conv
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout2d(0.25)  # 2D Dropout

        self.fc1 = nn.Linear(32 * 14 * 14, 128)
        self.bn_fc = nn.BatchNorm1d(128)  # BN for FC
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.dropout(x)

        x = x.view(-1, 32 * 14 * 14)
        x = self.fc1(x)
        x = self.bn_fc(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x
```

---

## 9. CIFAR-10 Classification

### Data

- 32×32 RGB images
- 10 classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

### Model

```python
class CIFAR10Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            # Two consecutive 3×3 convs give a 5×5 effective receptive field
            # with fewer parameters: 2×(3×3)=18 vs 1×(5×5)=25 weights per channel
            nn.Conv2d(3, 64, 3, padding=1),   # padding=1: same padding preserves 32×32
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 32→16: halves spatial dims, doubles receptive field

            # Channel doubling (64→128): as spatial resolution decreases,
            # we increase channels to maintain representational capacity
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 16→8
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),     # 50% dropout: aggressive regularization before FC
                                 # layers which have the most parameters
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 128 * 8 * 8)
        x = self.classifier(x)
        return x
```

---

## 10. Summary

### Core Concepts

1. **Convolution**: Local pattern extraction, parameter sharing
2. **Pooling**: Spatial reduction, increased invariance
3. **Channels**: Learn diverse features
4. **Hierarchical Learning**: Low-level → High-level features

### CNN vs MLP

| Item | MLP | CNN |
|------|-----|-----|
| Connectivity | Fully connected | Local connections |
| Parameters | Many | Few (shared) |
| Spatial Information | Ignored | Preserved |
| Images | Inefficient | Efficient |

**Why parameter sharing matters**: An MLP treating a 224×224×3 image as a flat vector would need 224×224×3 = 150,528 weights *per neuron* in the first layer. A 3×3 conv filter uses only 3×3×3 = 27 weights and slides across the entire image — the same 27 weights detect the same pattern regardless of position. This makes CNNs both parameter-efficient and translation-invariant.

### Next Steps

In [08_CNN_Advanced.md](./08_CNN_Advanced.md), we'll learn famous architectures like ResNet and VGG.

---

## Exercises

### Exercise 1: Output Dimension Calculation

Without running any code, compute the output spatial dimensions for each scenario:

1. Input: 28×28, Conv2d(kernel=5, stride=1, padding=0). What is the output size?
2. Input: 64×64, Conv2d(kernel=3, stride=2, padding=1). What is the output size?
3. Input: 32×32, three sequential Conv2d(kernel=3, stride=1, padding=1) followed by MaxPool2d(2,2). What is the final size?
4. Verify each answer by running `torch.randn(1, C, H, W)` through the corresponding layers.

### Exercise 2: Manual 2D Convolution

Implement convolution manually in NumPy and apply it to an image.

1. Use the `conv2d_numpy` function from the lesson.
2. Create a 5×5 test image with values of your choice.
3. Apply the Sobel-x filter and a Sobel-y filter (`[[-1,-2,-1],[0,0,0],[1,2,1]]`).
4. Visualize the original image and both edge-filtered outputs with matplotlib.
5. Describe what patterns each filter detects.

### Exercise 3: Train MNISTNet and Visualize Feature Maps

Build and train the `MNISTNet` model, then inspect what the first convolutional layer learns.

1. Train `MNISTNet` for 3 epochs on MNIST.
2. After training, pick one test image and extract the output of `conv1` (before ReLU) using a forward hook.
3. Visualize all 32 feature maps in a grid.
4. Identify which filters respond strongly to horizontal edges, vertical edges, or other patterns.

### Exercise 4: CIFAR-10 with BatchNorm vs Without

Compare the effect of Batch Normalization on training stability.

1. Define two versions of `CIFAR10Net`: one with `nn.BatchNorm2d` after each conv layer, one without.
2. Train both for 15 epochs using identical hyperparameters.
3. Plot training loss and test accuracy curves side by side.
4. Explain in 2-3 sentences why Batch Normalization helps training converge faster and more stably.

### Exercise 5: Parameter Count Analysis

Analytically derive and verify the parameter counts for CNNs vs MLPs.

1. Count the parameters in `MNISTNet` manually (conv layers and FC layers separately).
2. Calculate how many parameters a fully connected network with the same input (28×28=784) and output (10) and one hidden layer of size 128 would have.
3. Use `sum(p.numel() for p in model.parameters())` to verify both counts.
4. Explain in your own words why CNNs use far fewer parameters than MLPs for image tasks.
