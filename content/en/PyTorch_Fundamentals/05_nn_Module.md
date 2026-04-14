# nn.Module

**Previous**: [Autograd](./04_Autograd.md) | **Next**: [Loss Functions and Optimizers](./06_Loss_Functions_and_Optimizers.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Define custom neural network classes by subclassing `nn.Module`
2. Implement the `forward()` method and understand why `__call__` wraps it
3. Register parameters and buffers correctly for serialization and device management
4. Use built-in layers (`nn.Linear`, `nn.Conv2d`, `nn.ReLU`, `nn.Dropout`, etc.)
5. Compose modules using `nn.Sequential`, `nn.ModuleList`, and `nn.ModuleDict`
6. Inspect a model's parameters with `parameters()`, `named_parameters()`, and `state_dict()`
7. Move entire models between devices and dtypes with `.to()`
8. Apply custom weight initialization strategies

---

`nn.Module` is the base class for all neural network components in PyTorch. Every layer, loss function, and model is an `nn.Module`. Learning to define, compose, and inspect modules is the gateway to building any architecture.

---

## 1. Defining a Module

### 1.1 Basic Structure

```python
import torch
import torch.nn as nn

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()  # MUST call parent __init__
        self.linear1 = nn.Linear(784, 128)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x

# Create model
model = SimpleNet()

# Forward pass (use the model as a callable)
x = torch.randn(32, 784)  # batch of 32 images, 784 pixels each
output = model(x)          # calls model.forward(x) with hooks
print(output.shape)        # torch.Size([32, 10])
```

### 1.2 Why `model(x)` Instead of `model.forward(x)`

Always call `model(x)`, never `model.forward(x)` directly:

```python
# model(x) does:
# 1. Runs registered forward pre-hooks
# 2. Calls self.forward(x)
# 3. Runs registered forward hooks
# 4. Returns the result

# Calling model.forward(x) skips hooks!
output = model(x)          # CORRECT
output = model.forward(x)  # WRONG (bypasses hooks)
```

---

## 2. Parameters and Buffers

### 2.1 nn.Parameter

Parameters are tensors that are automatically registered for gradient computation:

```python
class CustomLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        # nn.Parameter: automatically registered, requires_grad=True
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x):
        return x @ self.weight.T + self.bias

layer = CustomLayer(4, 3)

# Parameters are tracked
for name, param in layer.named_parameters():
    print(f"{name}: shape={param.shape}, requires_grad={param.requires_grad}")
# weight: shape=torch.Size([3, 4]), requires_grad=True
# bias: shape=torch.Size([3]), requires_grad=True
```

### 2.2 Buffers

Buffers are tensors that should be saved with the model but are NOT parameters (no gradients):

```python
class BatchNormManual(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        # Parameters (learnable)
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))
        # Buffers (not learnable, but saved with model)
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, x):
        if self.training:
            mean = x.mean(dim=0)
            var = x.var(dim=0)
            # Update running statistics
            self.running_mean = 0.9 * self.running_mean + 0.1 * mean
            self.running_var = 0.9 * self.running_var + 0.1 * var
        else:
            mean = self.running_mean
            var = self.running_var
        x_norm = (x - mean) / torch.sqrt(var + 1e-5)
        return self.gamma * x_norm + self.beta

bn = BatchNormManual(10)

# Buffers appear in state_dict but NOT in parameters()
print(list(bn.named_parameters()))  # gamma, beta only
print(list(bn.named_buffers()))     # running_mean, running_var
print(bn.state_dict().keys())       # gamma, beta, running_mean, running_var
```

### 2.3 Plain Attributes vs Parameters

```python
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 5)          # registered as submodule
        self.w = nn.Parameter(torch.randn(5))    # registered as parameter
        self.register_buffer('mask', torch.ones(5))  # registered as buffer
        self.config = {'lr': 0.01}               # plain attribute (NOT tracked)
        self.scale = torch.tensor(2.0)           # plain tensor (NOT tracked!)

model = Model()
model.to('cpu')  # linear, w, mask all move; scale does NOT move
```

> **Rule**: Use `nn.Parameter` for learnable tensors, `register_buffer` for non-learnable tensors that need to be saved/moved, and plain attributes for everything else.

---

## 3. Built-in Layers

### 3.1 Linear Layers

```python
# Fully connected layer: y = xW^T + b
linear = nn.Linear(in_features=128, out_features=64)
print(linear.weight.shape)  # [64, 128]
print(linear.bias.shape)    # [64]

# Without bias
linear_no_bias = nn.Linear(128, 64, bias=False)
```

### 3.2 Activation Functions

```python
# As modules (for use in nn.Sequential)
relu = nn.ReLU()
leaky_relu = nn.LeakyReLU(negative_slope=0.01)
sigmoid = nn.Sigmoid()
tanh = nn.Tanh()
gelu = nn.GELU()
softmax = nn.Softmax(dim=-1)

# As functions (for use in forward())
import torch.nn.functional as F
x = torch.randn(3, 4)
y = F.relu(x)
y = F.gelu(x)
y = F.softmax(x, dim=-1)
```

### 3.3 Normalization

```python
# Batch normalization
bn = nn.BatchNorm1d(num_features=128)  # for 1D: [batch, features]
bn2d = nn.BatchNorm2d(num_features=64) # for 2D: [batch, channels, H, W]

# Layer normalization
ln = nn.LayerNorm(normalized_shape=128)
ln_2d = nn.LayerNorm([64, 32, 32])  # normalize over last N dims

# Group normalization
gn = nn.GroupNorm(num_groups=8, num_channels=64)
```

### 3.4 Dropout

```python
dropout = nn.Dropout(p=0.5)           # randomly zero 50% of elements
dropout2d = nn.Dropout2d(p=0.25)      # drop entire channels

x = torch.randn(4, 128)

# Dropout is active in train mode, inactive in eval mode
model.train()
y_train = dropout(x)  # some elements zeroed
model.eval()
y_eval = dropout(x)   # all elements pass through
```

### 3.5 Convolutional Layers (Preview)

```python
# 2D convolution (for images)
conv = nn.Conv2d(
    in_channels=3,      # e.g., RGB input
    out_channels=64,     # number of filters
    kernel_size=3,       # 3x3 filter
    stride=1,
    padding=1            # same padding
)

x = torch.randn(1, 3, 32, 32)  # [batch, channels, height, width]
y = conv(x)
print(y.shape)  # [1, 64, 32, 32]
```

---

## 4. Module Composition

### 4.1 nn.Sequential

```python
# Chain layers in order
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(128, 10)
)

x = torch.randn(32, 784)
output = model(x)  # passes through all layers in order
print(output.shape)  # [32, 10]

# Named sequential
model = nn.Sequential(
    ('flatten', nn.Flatten()),
    ('fc1', nn.Linear(784, 256)),
    ('relu1', nn.ReLU()),
    ('fc2', nn.Linear(256, 10)),
)
```

### 4.2 nn.ModuleList

For dynamic collections of modules:

```python
class MultiHeadModel(nn.Module):
    def __init__(self, n_heads):
        super().__init__()
        self.shared = nn.Linear(784, 256)
        # ModuleList: properly registered as submodules
        self.heads = nn.ModuleList([
            nn.Linear(256, 10) for _ in range(n_heads)
        ])

    def forward(self, x):
        x = F.relu(self.shared(x))
        return [head(x) for head in self.heads]

model = MultiHeadModel(3)
# All head parameters are tracked:
print(sum(p.numel() for p in model.parameters()))
```

> **Warning**: A plain Python list (`self.heads = [...]`) would NOT register the modules. Always use `nn.ModuleList`.

### 4.3 nn.ModuleDict

```python
class FlexibleModel(nn.Module):
    def __init__(self, activations):
        super().__init__()
        self.layers = nn.ModuleDict({
            'linear1': nn.Linear(784, 256),
            'linear2': nn.Linear(256, 10),
        })
        self.activations = nn.ModuleDict({
            'relu': nn.ReLU(),
            'gelu': nn.GELU(),
        })

    def forward(self, x, activation='relu'):
        x = self.layers['linear1'](x)
        x = self.activations[activation](x)
        x = self.layers['linear2'](x)
        return x
```

### 4.4 Nested Modules

```python
class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, x):
        return x + self.net(x)  # skip connection

class DeepNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_blocks, output_dim):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.Sequential(
            *[ResidualBlock(hidden_dim) for _ in range(n_blocks)]
        )
        self.output_proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.input_proj(x))
        x = self.blocks(x)
        return self.output_proj(x)

model = DeepNet(784, 256, 4, 10)
print(model)  # prints the full architecture tree
```

---

## 5. Inspecting Models

### 5.1 Parameters

```python
model = DeepNet(784, 256, 4, 10)

# All parameters
for name, param in model.named_parameters():
    print(f"{name}: {param.shape}")

# Count parameters
total = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total: {total:,}  Trainable: {trainable:,}")
```

### 5.2 State Dict

```python
# state_dict: OrderedDict of parameter name -> tensor
sd = model.state_dict()
for key, value in sd.items():
    print(f"{key}: {value.shape}")

# Load state dict
model2 = DeepNet(784, 256, 4, 10)
model2.load_state_dict(sd)
```

### 5.3 Model Summary

```python
# Print architecture
print(model)

# Children (immediate submodules only)
for name, child in model.named_children():
    print(f"{name}: {type(child).__name__}")

# All modules (recursive)
for name, module in model.named_modules():
    print(f"{name}: {type(module).__name__}")
```

---

## 6. Device and dtype Management

### 6.1 Moving Models

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = SimpleNet()
model = model.to(device)  # moves ALL parameters and buffers

# Check
for param in model.parameters():
    print(param.device)  # cuda:0 (or cpu)

# dtype conversion
model = model.to(torch.float16)  # half precision
model = model.float()            # back to float32
model = model.double()           # float64
```

### 6.2 Device-Agnostic Code

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = SimpleNet().to(device)
x = torch.randn(32, 784).to(device)  # input must be on same device
output = model(x)
```

---

## 7. Weight Initialization

### 7.1 Default Initialization

PyTorch initializes `nn.Linear` weights using Kaiming uniform by default. But you may want custom initialization:

```python
def init_weights(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)

model = SimpleNet()
model.apply(init_weights)  # recursively applies to all submodules
```

### 7.2 Common Initializers

```python
# Xavier (Glorot)
nn.init.xavier_uniform_(tensor)
nn.init.xavier_normal_(tensor)

# Kaiming (He)
nn.init.kaiming_uniform_(tensor, nonlinearity='relu')
nn.init.kaiming_normal_(tensor, nonlinearity='relu')

# Others
nn.init.zeros_(tensor)
nn.init.ones_(tensor)
nn.init.constant_(tensor, val=0.5)
nn.init.normal_(tensor, mean=0, std=0.01)
nn.init.uniform_(tensor, a=-0.1, b=0.1)
nn.init.orthogonal_(tensor)
```

---

## 8. train() vs eval()

```python
model.train()  # training mode: dropout active, BatchNorm uses batch stats
model.eval()   # eval mode: dropout inactive, BatchNorm uses running stats

# Check current mode
print(model.training)  # True or False

# Common pattern
model.eval()
with torch.no_grad():
    predictions = model(test_input)
model.train()
```

> **Important**: `model.eval()` does NOT disable gradient computation. You still need `torch.no_grad()` for that. `eval()` only affects layers like Dropout and BatchNorm.

---

## Summary

| Concept | Key Takeaway |
|---------|-------------|
| nn.Module | Base class for all neural network components |
| forward() | Define computation; call via `model(x)`, never `model.forward(x)` |
| nn.Parameter | Learnable tensor, auto-registered with the module |
| register_buffer | Non-learnable tensor saved with the model |
| nn.Sequential | Chain layers in order |
| nn.ModuleList/Dict | Dynamic collections of submodules (properly registered) |
| state_dict | Serializable snapshot of all parameters and buffers |
| train()/eval() | Toggle behavior of Dropout, BatchNorm, etc. |
| apply() | Recursively apply a function (e.g., weight init) to all submodules |

---

**Next**: [Loss Functions and Optimizers](./06_Loss_Functions_and_Optimizers.md) -- Choosing and configuring loss functions and optimization algorithms.
