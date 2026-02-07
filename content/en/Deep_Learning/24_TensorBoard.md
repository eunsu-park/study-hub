# 24. TensorBoard Visualization

## Learning Objectives

- Understand TensorBoard's core features and use cases
- Learn how to integrate TensorBoard with PyTorch
- Visualize training metrics, model graphs, and embeddings
- Compare and analyze hyperparameter tuning results

---

## 1. Introduction to TensorBoard

### 1.1 What is TensorBoard?

TensorBoard is a tool for visualizing and analyzing machine learning experiments.

```
┌─────────────────────────────────────────────────────────────┐
│                      TensorBoard                             │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐        │
│  │ Scalars │  │ Images  │  │ Graphs  │  │Histograms│        │
│  │ (loss,  │  │ (samples,│  │ (model  │  │ (weight │        │
│  │ accuracy)│  │ outputs) │  │ structure)│ │ distribution)│  │
│  └─────────┘  └─────────┘  └─────────┘  └─────────┘        │
│                                                              │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐        │
│  │Embeddings│ │  Text   │  │ Audio   │  │ HParams │        │
│  │(t-SNE,  │  │ (logs,  │  │ (audio  │  │(hyper-  │        │
│  │ PCA)    │  │ samples)│  │ samples)│  │parameters)│       │
│  └─────────┘  └─────────┘  └─────────┘  └─────────┘        │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 Installation and Execution

```bash
# Installation
pip install tensorboard

# Execution
tensorboard --logdir=runs --port=6006

# Access via browser: http://localhost:6006
```

---

## 2. Integrating PyTorch with TensorBoard

### 2.1 Basic Usage of SummaryWriter

```python
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn

# Create SummaryWriter
writer = SummaryWriter('runs/experiment_1')

# Log scalar values
for step in range(100):
    loss = 1.0 / (step + 1)  # Example loss value
    accuracy = step / 100.0   # Example accuracy

    writer.add_scalar('Loss/train', loss, step)
    writer.add_scalar('Accuracy/train', accuracy, step)

# Close
writer.close()
```

### 2.2 Organizing Log Directories per Experiment

```python
from datetime import datetime
import os

def create_writer(experiment_name: str, extra: str = None) -> SummaryWriter:
    """Create unique log directory per experiment"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    if extra:
        log_dir = f'runs/{experiment_name}/{extra}/{timestamp}'
    else:
        log_dir = f'runs/{experiment_name}/{timestamp}'

    os.makedirs(log_dir, exist_ok=True)
    return SummaryWriter(log_dir)

# Usage example
writer = create_writer('mnist_cnn', 'lr_0.001_batch_32')
```

---

## 3. Scalar Logging

### 3.1 Recording Training/Validation Metrics

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

class Trainer:
    def __init__(self, model, train_loader, val_loader, device='cuda'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=0.001)
        self.writer = SummaryWriter()
        self.global_step = 0

    def train_epoch(self, epoch: int):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

            # Per-batch logging (optional)
            if batch_idx % 100 == 0:
                self.writer.add_scalar('Loss/train_step', loss.item(), self.global_step)
            self.global_step += 1

        # Per-epoch logging
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total

        self.writer.add_scalar('Loss/train', epoch_loss, epoch)
        self.writer.add_scalar('Accuracy/train', epoch_acc, epoch)

        return epoch_loss, epoch_acc

    def validate(self, epoch: int):
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                val_loss += self.criterion(output, target).item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()

        val_loss /= len(self.val_loader)
        val_acc = 100. * correct / total

        self.writer.add_scalar('Loss/val', val_loss, epoch)
        self.writer.add_scalar('Accuracy/val', val_acc, epoch)

        return val_loss, val_acc
```

### 3.2 Displaying Multiple Scalars on One Graph

```python
# Method 1: Using add_scalars
writer.add_scalars('Loss', {
    'train': train_loss,
    'val': val_loss
}, epoch)

writer.add_scalars('Accuracy', {
    'train': train_acc,
    'val': val_acc
}, epoch)

# Method 2: Using same tag path
# Loss/train and Loss/val are automatically grouped in TensorBoard
```

### 3.3 Logging Learning Rate Scheduler

```python
from torch.optim.lr_scheduler import CosineAnnealingLR

scheduler = CosineAnnealingLR(optimizer, T_max=100)

for epoch in range(100):
    train_one_epoch()
    scheduler.step()

    # Record current learning rate
    current_lr = scheduler.get_last_lr()[0]
    writer.add_scalar('Learning_Rate', current_lr, epoch)
```

---

## 4. Image Logging

### 4.1 Visualizing Input Images

```python
import torchvision
from torchvision import transforms

def log_images(writer, images, tag, step, normalize=True):
    """Visualize image batch as a grid"""
    # Restore normalized images to original range (optional)
    if normalize:
        # Reverse ImageNet normalization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        images = images * std + mean
        images = torch.clamp(images, 0, 1)

    # Create grid
    grid = torchvision.utils.make_grid(images, nrow=8, padding=2)
    writer.add_image(tag, grid, step)

# Usage example
for batch_idx, (images, labels) in enumerate(train_loader):
    if batch_idx == 0:  # Log only first batch
        log_images(writer, images[:32], 'Input/samples', epoch)
        break
```

### 4.2 Visualizing Generative Model Outputs

```python
class GANTrainer:
    def __init__(self, generator, discriminator, writer):
        self.G = generator
        self.D = discriminator
        self.writer = writer
        self.fixed_noise = torch.randn(64, 100, 1, 1)  # Fixed noise

    def log_generated_images(self, epoch):
        """Visualize generated images (for tracking training progress)"""
        self.G.eval()
        with torch.no_grad():
            fake_images = self.G(self.fixed_noise.to(self.G.device))
            fake_images = (fake_images + 1) / 2  # [-1, 1] -> [0, 1]

        grid = torchvision.utils.make_grid(fake_images, nrow=8)
        self.writer.add_image('Generated/samples', grid, epoch)
        self.G.train()
```

### 4.3 Visualizing Feature Maps

```python
def visualize_feature_maps(model, image, writer, layer_name, step):
    """Visualize intermediate layer feature maps of CNN"""
    activation = {}

    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook

    # Register hook
    layer = dict(model.named_modules())[layer_name]
    handle = layer.register_forward_hook(get_activation(layer_name))

    # Forward pass
    model.eval()
    with torch.no_grad():
        _ = model(image.unsqueeze(0))

    # Extract feature maps
    feat = activation[layer_name].squeeze(0)  # [C, H, W]

    # Visualize by channel (first 16)
    feat = feat[:16].unsqueeze(1)  # [16, 1, H, W]
    grid = torchvision.utils.make_grid(feat, nrow=4, normalize=True)
    writer.add_image(f'Features/{layer_name}', grid, step)

    handle.remove()
```

### 4.4 Grad-CAM Visualization

```python
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Register hooks
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def __call__(self, x, class_idx=None):
        self.model.eval()
        output = self.model(x)

        if class_idx is None:
            class_idx = output.argmax(dim=1)

        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, class_idx] = 1
        output.backward(gradient=one_hot)

        # Compute Grad-CAM
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=x.shape[2:], mode='bilinear', align_corners=False)
        cam = cam - cam.min()
        cam = cam / cam.max()

        return cam.squeeze().cpu().numpy()

def log_gradcam(writer, model, image, target_layer, step):
    """Log Grad-CAM results to TensorBoard"""
    gradcam = GradCAM(model, target_layer)
    cam = gradcam(image.unsqueeze(0))

    # Apply colormap
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # Original image
    img_np = image.permute(1, 2, 0).cpu().numpy()
    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
    axes[0].imshow(img_np)
    axes[0].set_title('Original')
    axes[0].axis('off')

    # Grad-CAM
    axes[1].imshow(cam, cmap='jet')
    axes[1].set_title('Grad-CAM')
    axes[1].axis('off')

    # Overlay
    axes[2].imshow(img_np)
    axes[2].imshow(cam, cmap='jet', alpha=0.5)
    axes[2].set_title('Overlay')
    axes[2].axis('off')

    plt.tight_layout()
    writer.add_figure('GradCAM', fig, step)
    plt.close(fig)
```

---

## 5. Histograms

### 5.1 Visualizing Weight Distributions

```python
def log_weights_histograms(writer, model, epoch):
    """Visualize model weight distributions as histograms"""
    for name, param in model.named_parameters():
        if param.requires_grad:
            # Weight values
            writer.add_histogram(f'Weights/{name}', param.data, epoch)

            # Gradient values (if available)
            if param.grad is not None:
                writer.add_histogram(f'Gradients/{name}', param.grad, epoch)

# Use in training loop
for epoch in range(num_epochs):
    train_one_epoch()

    # Log histograms every 10 epochs
    if epoch % 10 == 0:
        log_weights_histograms(writer, model, epoch)
```

### 5.2 Tracking Activation Value Distributions

```python
class ActivationLogger:
    """Track activation value distributions per layer"""

    def __init__(self, model, writer):
        self.writer = writer
        self.activations = {}
        self.hooks = []

        for name, module in model.named_modules():
            if isinstance(module, (nn.ReLU, nn.GELU, nn.SiLU)):
                hook = module.register_forward_hook(
                    self._make_hook(name)
                )
                self.hooks.append(hook)

    def _make_hook(self, name):
        def hook(module, input, output):
            self.activations[name] = output.detach()
        return hook

    def log(self, step):
        for name, activation in self.activations.items():
            self.writer.add_histogram(f'Activations/{name}', activation, step)
        self.activations.clear()

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
```

---

## 6. Model Graphs

### 6.1 Visualizing Model Structure

```python
import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# Log model graph
model = SimpleCNN()
dummy_input = torch.randn(1, 3, 32, 32)

writer = SummaryWriter('runs/model_graph')
writer.add_graph(model, dummy_input)
writer.close()
```

### 6.2 Complex Model Graphs

```python
# Transformer model graph
from torchvision.models import vit_b_16

model = vit_b_16(pretrained=False)
dummy_input = torch.randn(1, 3, 224, 224)

writer.add_graph(model, dummy_input)
```

---

## 7. Embedding Visualization

### 7.1 Visualizing Embeddings with t-SNE/PCA

```python
import torch
import torchvision
from torchvision import datasets, transforms

def extract_embeddings(model, dataloader, device):
    """Extract embeddings from the layer before the last layer"""
    model.eval()
    embeddings = []
    labels = []
    images = []

    with torch.no_grad():
        for data, target in dataloader:
            data = data.to(device)

            # Forward pass up to before the last FC layer
            # Modify according to model structure
            x = model.features(data)
            x = model.avgpool(x)
            emb = x.view(x.size(0), -1)

            embeddings.append(emb.cpu())
            labels.append(target)
            images.append(data.cpu())

    return (
        torch.cat(embeddings),
        torch.cat(labels),
        torch.cat(images)
    )

# Usage example
embeddings, labels, images = extract_embeddings(model, test_loader, device)

# Log embeddings to TensorBoard
writer.add_embedding(
    embeddings,
    metadata=labels,
    label_img=images,
    global_step=epoch,
    tag='Embeddings/test_set'
)
```

### 7.2 Visualizing Word Embeddings (NLP)

```python
import torch.nn as nn

# Word embedding example
vocab = ['king', 'queen', 'man', 'woman', 'prince', 'princess',
         'dog', 'cat', 'puppy', 'kitten']
embedding_dim = 128

embedding_layer = nn.Embedding(len(vocab), embedding_dim)

# Extract embedding vectors
indices = torch.arange(len(vocab))
embeddings = embedding_layer(indices)

# Log to TensorBoard
writer.add_embedding(
    embeddings,
    metadata=vocab,
    tag='Word_Embeddings'
)
```

---

## 8. Hyperparameter Tuning (HParams)

### 8.1 Logging Hyperparameter Experiments

```python
from torch.utils.tensorboard.summary import hparams

def train_with_hparams(lr, batch_size, optimizer_name, epochs=10):
    """Run experiment with specific hyperparameters"""

    # Unique experiment directory
    run_name = f'lr_{lr}_bs_{batch_size}_{optimizer_name}'
    writer = SummaryWriter(f'runs/hparam_search/{run_name}')

    # Model and data setup
    model = SimpleCNN().to(device)

    if optimizer_name == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Training
    best_accuracy = 0
    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer)
        val_loss, val_acc = validate(model, val_loader)

        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)

        best_accuracy = max(best_accuracy, val_acc)

    # Record hyperparameters and final metrics
    hparam_dict = {
        'lr': lr,
        'batch_size': batch_size,
        'optimizer': optimizer_name
    }
    metric_dict = {
        'hparam/best_accuracy': best_accuracy,
        'hparam/final_loss': val_loss
    }

    writer.add_hparams(hparam_dict, metric_dict)
    writer.close()

    return best_accuracy

# Execute grid search
learning_rates = [0.001, 0.01, 0.1]
batch_sizes = [32, 64, 128]
optimizers = ['adam', 'sgd']

for lr in learning_rates:
    for bs in batch_sizes:
        for opt in optimizers:
            acc = train_with_hparams(lr, bs, opt)
            print(f'LR={lr}, BS={bs}, OPT={opt} -> Acc={acc:.2f}%')
```

### 8.2 Integrating Optuna with TensorBoard

```python
import optuna
from optuna.integration import TensorBoardCallback

def objective(trial):
    # Hyperparameter sampling
    lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
    n_layers = trial.suggest_int('n_layers', 1, 3)
    hidden_dim = trial.suggest_int('hidden_dim', 64, 256, step=64)
    dropout = trial.suggest_float('dropout', 0.1, 0.5)

    # Model creation
    model = create_model(n_layers, hidden_dim, dropout)

    # Training and evaluation
    accuracy = train_and_evaluate(model, lr, batch_size)

    return accuracy

# Optimize with TensorBoard callback
study = optuna.create_study(direction='maximize')
tensorboard_callback = TensorBoardCallback('runs/optuna/', metric_name='accuracy')

study.optimize(
    objective,
    n_trials=100,
    callbacks=[tensorboard_callback]
)

print(f'Best trial: {study.best_trial.params}')
print(f'Best accuracy: {study.best_value:.2f}%')
```

---

## 9. Custom Scalar Layouts

### 9.1 Defining Dashboard Layouts

```python
from torch.utils.tensorboard import SummaryWriter
from torch.utils.tensorboard.summary import custom_scalars

# Define custom layout
layout = {
    'Training Metrics': {
        'loss': ['Multiline', ['Loss/train', 'Loss/val']],
        'accuracy': ['Multiline', ['Accuracy/train', 'Accuracy/val']],
    },
    'Learning Rate': {
        'lr': ['Multiline', ['Learning_Rate']],
    },
    'Per-Class Accuracy': {
        'classes': ['Multiline', [f'Accuracy/class_{i}' for i in range(10)]],
    },
}

writer = SummaryWriter('runs/custom_layout')
writer.add_custom_scalars(layout)

# Continue with normal logging
for epoch in range(100):
    writer.add_scalar('Loss/train', train_loss, epoch)
    writer.add_scalar('Loss/val', val_loss, epoch)
    writer.add_scalar('Accuracy/train', train_acc, epoch)
    writer.add_scalar('Accuracy/val', val_acc, epoch)
    writer.add_scalar('Learning_Rate', lr, epoch)

    for i in range(10):
        writer.add_scalar(f'Accuracy/class_{i}', class_acc[i], epoch)
```

---

## 10. Text and Audio Logging

### 10.1 Text Logging

```python
# Log training logs
writer.add_text('Hyperparameters', f'''
- Learning Rate: {lr}
- Batch Size: {batch_size}
- Optimizer: {optimizer_name}
- Epochs: {num_epochs}
''', 0)

# Log model summary
from torchinfo import summary

model_summary = str(summary(model, input_size=(1, 3, 224, 224), verbose=0))
writer.add_text('Model/summary', f'```\n{model_summary}\n```', 0)

# Log NLP samples
writer.add_text('Samples/input', 'The quick brown fox jumps over the lazy dog', 0)
writer.add_text('Samples/prediction', 'The fast brown fox jumps over the lazy dog', 0)
```

### 10.2 Audio Logging

```python
import torchaudio

# Log audio file
waveform, sample_rate = torchaudio.load('audio.wav')
writer.add_audio('Audio/input', waveform, 0, sample_rate=sample_rate)

# Log generated audio (e.g., TTS, music generation)
generated_audio = model.generate(text_input)
writer.add_audio('Audio/generated', generated_audio, step, sample_rate=22050)
```

---

## 11. Profiling

### 11.1 PyTorch Profiler with TensorBoard

```python
import torch
from torch.profiler import profile, record_function, ProfilerActivity

# Profiling setup
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=torch.profiler.schedule(
        wait=1,      # Warmup
        warmup=1,    # Prepare profiling
        active=3,    # Actual profiling
        repeat=2     # Repeat
    ),
    on_trace_ready=torch.profiler.tensorboard_trace_handler('runs/profiler'),
    record_shapes=True,
    profile_memory=True,
    with_stack=True
) as prof:
    for step, (data, target) in enumerate(train_loader):
        if step >= (1 + 1 + 3) * 2:
            break

        with record_function("data_loading"):
            data, target = data.to(device), target.to(device)

        with record_function("forward"):
            output = model(data)
            loss = criterion(output, target)

        with record_function("backward"):
            optimizer.zero_grad()
            loss.backward()

        with record_function("optimizer_step"):
            optimizer.step()

        prof.step()

# Check PYTORCH_PROFILER tab in TensorBoard
```

### 11.2 Memory Profiling

```python
def profile_memory(model, input_size, device='cuda'):
    """Analyze GPU memory usage"""
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    model = model.to(device)
    x = torch.randn(input_size).to(device)

    # Forward pass
    torch.cuda.synchronize()
    output = model(x)
    forward_memory = torch.cuda.max_memory_allocated() / 1e9

    # Backward pass
    loss = output.sum()
    loss.backward()
    torch.cuda.synchronize()
    total_memory = torch.cuda.max_memory_allocated() / 1e9

    print(f'Forward memory: {forward_memory:.2f} GB')
    print(f'Total memory (forward + backward): {total_memory:.2f} GB')

    return forward_memory, total_memory

# Logging
fwd_mem, total_mem = profile_memory(model, (32, 3, 224, 224))
writer.add_scalar('Memory/forward_GB', fwd_mem, 0)
writer.add_scalar('Memory/total_GB', total_mem, 0)
```

---

## 12. TensorBoard in Distributed Training

### 12.1 Logging in DDP Environment

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def setup_distributed():
    dist.init_process_group(backend='nccl')
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    return rank, world_size

def train_ddp():
    rank, world_size = setup_distributed()

    # Log TensorBoard only from rank 0
    writer = SummaryWriter() if rank == 0 else None

    model = MyModel().to(rank)
    model = DDP(model, device_ids=[rank])

    for epoch in range(num_epochs):
        # Calculate local metrics
        local_loss = train_one_epoch(model, train_loader)

        # Average loss across all processes
        loss_tensor = torch.tensor([local_loss]).to(rank)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        avg_loss = loss_tensor.item() / world_size

        # Log only from rank 0
        if writer is not None:
            writer.add_scalar('Loss/train', avg_loss, epoch)

    if writer is not None:
        writer.close()

    dist.destroy_process_group()
```

---

## 13. Practical Example: Complete Training Pipeline

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms, models
from datetime import datetime
import os

class TensorBoardTrainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: optim.Optimizer,
        scheduler=None,
        device: str = 'cuda',
        experiment_name: str = 'default'
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.criterion = nn.CrossEntropyLoss()

        # TensorBoard setup
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_dir = f'runs/{experiment_name}/{timestamp}'
        self.writer = SummaryWriter(log_dir)

        # Log model graph
        dummy_input = next(iter(train_loader))[0][:1].to(device)
        self.writer.add_graph(model, dummy_input)

        # Log hyperparameters
        self._log_hyperparameters()

        self.global_step = 0
        self.best_val_acc = 0

    def _log_hyperparameters(self):
        hparams = {
            'lr': self.optimizer.param_groups[0]['lr'],
            'batch_size': self.train_loader.batch_size,
            'optimizer': self.optimizer.__class__.__name__,
            'model': self.model.__class__.__name__,
        }
        self.writer.add_text('Hyperparameters', str(hparams), 0)

    def train_epoch(self, epoch: int):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

            # Per-batch loss logging
            self.writer.add_scalar('Loss/train_step', loss.item(), self.global_step)
            self.global_step += 1

            # Log first batch images
            if batch_idx == 0 and epoch % 10 == 0:
                grid = torchvision.utils.make_grid(data[:16])
                self.writer.add_image('Input/train_samples', grid, epoch)

        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total

        self.writer.add_scalar('Loss/train', epoch_loss, epoch)
        self.writer.add_scalar('Accuracy/train', epoch_acc, epoch)

        # Weight histograms (every 10 epochs)
        if epoch % 10 == 0:
            for name, param in self.model.named_parameters():
                self.writer.add_histogram(f'Weights/{name}', param, epoch)
                if param.grad is not None:
                    self.writer.add_histogram(f'Gradients/{name}', param.grad, epoch)

        return epoch_loss, epoch_acc

    def validate(self, epoch: int):
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                val_loss += self.criterion(output, target).item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(target.cpu().numpy())

        val_loss /= len(self.val_loader)
        val_acc = 100. * correct / total

        self.writer.add_scalar('Loss/val', val_loss, epoch)
        self.writer.add_scalar('Accuracy/val', val_acc, epoch)

        # Update best performance
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            self.writer.add_scalar('Best/val_accuracy', val_acc, epoch)

        return val_loss, val_acc

    def train(self, num_epochs: int):
        for epoch in range(num_epochs):
            train_loss, train_acc = self.train_epoch(epoch)
            val_loss, val_acc = self.validate(epoch)

            if self.scheduler:
                self.scheduler.step()
                self.writer.add_scalar(
                    'Learning_Rate',
                    self.scheduler.get_last_lr()[0],
                    epoch
                )

            print(f'Epoch {epoch+1}/{num_epochs}:')
            print(f'  Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%')
            print(f'  Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%')

        # Log final metrics
        self.writer.add_hparams(
            {'lr': self.optimizer.param_groups[0]['lr']},
            {'hparam/best_accuracy': self.best_val_acc}
        )

        self.writer.close()
        print(f'\nTraining complete. Best Val Accuracy: {self.best_val_acc:.2f}%')


# Usage example
if __name__ == '__main__':
    # Prepare data
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
    val_dataset = datasets.CIFAR10('./data', train=False, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4)

    # Model setup
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 10)

    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

    # Training
    trainer = TensorBoardTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device='cuda',
        experiment_name='cifar10_resnet18'
    )

    trainer.train(num_epochs=50)
```

---

## 14. Tips and Best Practices

### 14.1 Optimizing Logging Frequency

```python
# Logging per batch too frequently can degrade performance
# Recommended: batch loss every 100-500 steps, epoch metrics every epoch

LOG_INTERVAL = 100

for batch_idx, (data, target) in enumerate(train_loader):
    # ... training code ...

    if batch_idx % LOG_INTERVAL == 0:
        writer.add_scalar('Loss/train_step', loss.item(), global_step)
```

### 14.2 Managing Log Files

```bash
# Clean up old logs
find runs/ -type d -mtime +30 -exec rm -rf {} +

# Keep only specific experiments
tensorboard --logdir=runs/experiment_final --port=6006
```

### 14.3 Remote TensorBoard

```bash
# Run TensorBoard on server
tensorboard --logdir=runs --host=0.0.0.0 --port=6006

# SSH tunneling from local machine
ssh -L 6006:localhost:6006 user@server

# Or use ngrok
ngrok http 6006
```

---

## Exercises

### Exercise 1: Implement Basic Logging
While training an MNIST classification model, log the following to TensorBoard:
- Training/validation loss and accuracy
- Learning rate changes
- Sample input images

### Exercise 2: Model Analysis
For a trained CNN model:
- Visualize weight histograms
- Visualize feature maps
- Apply Grad-CAM

### Exercise 3: Hyperparameter Tuning
For learning rate, batch size, and dropout ratio:
- Execute grid search
- Compare results in HParams dashboard
- Find optimal combination

---

## References

- [TensorBoard Official Documentation](https://www.tensorflow.org/tensorboard)
- [PyTorch TensorBoard Tutorial](https://pytorch.org/tutorials/intermediate/tensorboard_tutorial.html)
- [torch.utils.tensorboard API](https://pytorch.org/docs/stable/tensorboard.html)
