"""
06. CNN Advanced - Famous Architectures

Implements famous architectures like VGG, ResNet, and EfficientNet in PyTorch.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

print("=" * 60)
print("PyTorch CNN Advanced - Famous Architectures")
print("=" * 60)


# ============================================
# 1. VGG Block and Model
# ============================================
print("\n[1] VGG16 Implementation")
print("-" * 40)

def make_vgg_block(in_channels, out_channels, num_convs):
    """Create a VGG block"""
    layers = []
    for i in range(num_convs):
        layers.append(nn.Conv2d(
            in_channels if i == 0 else out_channels,
            out_channels, kernel_size=3, padding=1
        ))
        layers.append(nn.ReLU(inplace=True))
    layers.append(nn.MaxPool2d(2, 2))
    return nn.Sequential(*layers)


class VGG16(nn.Module):
    """VGG16 Implementation"""
    def __init__(self, num_classes=1000):
        super().__init__()
        # Feature extractor
        self.features = nn.Sequential(
            make_vgg_block(3, 64, 2),    # 224->112
            make_vgg_block(64, 128, 2),  # 112->56
            make_vgg_block(128, 256, 3), # 56->28
            make_vgg_block(256, 512, 3), # 28->14
            make_vgg_block(512, 512, 3), # 14->7
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

vgg = VGG16(num_classes=10)
print(f"VGG16 parameters: {sum(p.numel() for p in vgg.parameters()):,}")

# Test
x = torch.randn(1, 3, 224, 224)
out = vgg(x)
print(f"Input: {x.shape} -> Output: {out.shape}")


# ============================================
# 2. ResNet Basic Block
# ============================================
print("\n[2] ResNet Implementation")
print("-" * 40)

class BasicBlock(nn.Module):
    """ResNet Basic Block (for ResNet-18, 34)"""
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity  # Skip connection!
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    """ResNet Bottleneck Block (for ResNet-50, 101, 152)"""
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * 4, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * 4)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    """ResNet Implementation"""
    def __init__(self, block, layers, num_classes=1000):
        super().__init__()
        self.in_channels = 64

        # Initial layers
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        # ResNet layers
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # Classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion,
                         1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion)
            )

        layers = [block(self.in_channels, out_channels, stride, downsample)]
        self.in_channels = out_channels * block.expansion

        for _ in range(1, num_blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def resnet18(num_classes=1000):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes)

def resnet34(num_classes=1000):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes)

def resnet50(num_classes=1000):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes)

# Test
resnet = resnet18(num_classes=10)
print(f"ResNet-18 parameters: {sum(p.numel() for p in resnet.parameters()):,}")

x = torch.randn(1, 3, 224, 224)
out = resnet(x)
print(f"Input: {x.shape} -> Output: {out.shape}")


# ============================================
# 3. SE Block (Squeeze-and-Excitation)
# ============================================
print("\n[3] SE Block")
print("-" * 40)

class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        # Squeeze
        y = self.squeeze(x).view(b, c)
        # Excitation
        y = self.excitation(y).view(b, c, 1, 1)
        # Scale
        return x * y.expand_as(x)

# Test
se = SEBlock(64)
x = torch.randn(2, 64, 32, 32)
out = se(x)
print(f"SE Block: {x.shape} -> {out.shape}")


# ============================================
# 4. MBConv (EfficientNet Block)
# ============================================
print("\n[4] MBConv Block (EfficientNet)")
print("-" * 40)

class MBConv(nn.Module):
    """Mobile Inverted Bottleneck Convolution"""
    def __init__(self, in_channels, out_channels, expand_ratio=6,
                 stride=1, se_ratio=0.25):
        super().__init__()
        hidden_dim = in_channels * expand_ratio
        self.use_skip = stride == 1 and in_channels == out_channels

        layers = []

        # Expand
        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(inplace=True)
            ])

        # Depthwise
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride=stride,
                     padding=1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(inplace=True)
        ])

        self.conv = nn.Sequential(*layers)

        # SE
        se_channels = max(1, int(in_channels * se_ratio))
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(hidden_dim, se_channels, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(se_channels, hidden_dim, 1),
            nn.Sigmoid()
        )

        # Project
        self.project = nn.Sequential(
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        identity = x
        out = self.conv(x)
        out = out * self.se(out)
        out = self.project(out)

        if self.use_skip:
            out = out + identity
        return out

# Test
mbconv = MBConv(32, 32, expand_ratio=6)
x = torch.randn(2, 32, 28, 28)
out = mbconv(x)
print(f"MBConv: {x.shape} -> {out.shape}")


# ============================================
# 5. Pretrained Models
# ============================================
print("\n[5] torchvision Pretrained Models")
print("-" * 40)

try:
    import torchvision.models as models

    # Various pretrained models
    model_names = ['resnet18', 'resnet50', 'vgg16', 'mobilenet_v2']

    for name in model_names:
        model = getattr(models, name)(weights=None)  # Structure only, no weights
        params = sum(p.numel() for p in model.parameters())
        print(f"{name}: {params:,} parameters")

    # Load pretrained ResNet50
    print("\nLoading pretrained ResNet50:")
    resnet50_pretrained = models.resnet50(weights='IMAGENET1K_V2')
    print(f"  Last layer: {resnet50_pretrained.fc}")

    # Modify for transfer learning
    resnet50_pretrained.fc = nn.Linear(2048, 10)  # Change to 10 classes
    print(f"  Modified last layer: {resnet50_pretrained.fc}")

except ImportError:
    print("torchvision is not installed.")


# ============================================
# 6. Model Comparison
# ============================================
print("\n[6] Model Comparison")
print("-" * 40)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def measure_forward_time(model, input_shape, iterations=100):
    import time
    model.eval()
    x = torch.randn(*input_shape)
    with torch.no_grad():
        # Warmup
        for _ in range(10):
            _ = model(x)
        # Measure
        start = time.time()
        for _ in range(iterations):
            _ = model(x)
        end = time.time()
    return (end - start) / iterations * 1000  # ms

# Compare simple models
models_to_compare = {
    'VGG16 (simple)': VGG16(num_classes=10),
    'ResNet-18': resnet18(num_classes=10),
    'ResNet-50': resnet50(num_classes=10),
}

print(f"{'Model':<20} {'Params':>12} {'Time (ms)':>12}")
print("-" * 46)

for name, model in models_to_compare.items():
    params = count_parameters(model)
    try:
        time_ms = measure_forward_time(model, (1, 3, 224, 224), iterations=10)
        print(f"{name:<20} {params:>12,} {time_ms:>12.2f}")
    except:
        print(f"{name:<20} {params:>12,} {'N/A':>12}")


# ============================================
# 7. Skip Connection Effect Experiment
# ============================================
print("\n[7] Skip Connection Effect Experiment")
print("-" * 40)

class ResBlockWithoutSkip(nn.Module):
    """Block without Skip Connection"""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out)  # No skip!

class ResBlockWithSkip(nn.Module):
    """Block with Skip Connection"""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + identity)  # With skip!

# Compare deep networks
def make_deep_net(block_class, num_blocks, channels=64):
    layers = [nn.Conv2d(3, channels, 3, padding=1), nn.ReLU()]
    for _ in range(num_blocks):
        layers.append(block_class(channels))
    layers.append(nn.AdaptiveAvgPool2d(1))
    layers.append(nn.Flatten())
    layers.append(nn.Linear(channels, 10))
    return nn.Sequential(*layers)

# Check gradients
def check_gradient_flow(model, depth):
    model.train()
    x = torch.randn(1, 3, 32, 32, requires_grad=True)
    out = model(x)
    loss = out.sum()
    loss.backward()

    # Check first Conv layer gradient
    first_conv_grad = None
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            if module.weight.grad is not None:
                first_conv_grad = module.weight.grad.abs().mean().item()
                break

    return first_conv_grad

print("Gradient flow comparison (deep networks):")
for depth in [5, 10, 20]:
    net_no_skip = make_deep_net(ResBlockWithoutSkip, depth)
    net_with_skip = make_deep_net(ResBlockWithSkip, depth)

    grad_no_skip = check_gradient_flow(net_no_skip, depth)
    grad_with_skip = check_gradient_flow(net_with_skip, depth)

    print(f"  Depth {depth:2d}: No skip = {grad_no_skip:.6f}, With skip = {grad_with_skip:.6f}")


# ============================================
# Summary
# ============================================
print("\n" + "=" * 60)
print("CNN Architecture Summary")
print("=" * 60)

summary = """
Key Architectures:

1. VGG (2014)
   - Uses only 3x3 Conv
   - Depth = Performance (simple but many parameters)

2. ResNet (2015)
   - Skip Connection solves vanishing gradient
   - Can train 100+ layers
   - Most widely used

3. EfficientNet (2019)
   - Compound Scaling
   - MBConv (Depthwise Separable + SE)
   - Efficient parameter usage

Key Techniques:
- Batch Normalization
- Skip Connection (Residual)
- Depthwise Separable Conv
- Squeeze-and-Excitation

Practical Selection:
- Fast inference: MobileNet, EfficientNet-B0
- High accuracy: EfficientNet-B4~B7
- Balanced: ResNet-50
"""
print(summary)
print("=" * 60)
