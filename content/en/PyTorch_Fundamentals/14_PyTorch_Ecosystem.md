# PyTorch Ecosystem

**Previous**: [TorchScript and Deployment](./13_TorchScript_and_Deployment.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Use `torchvision` for computer vision tasks (models, datasets, transforms)
2. Use `torchaudio` for audio processing tasks
3. Use `torchtext` for NLP data processing
4. Accelerate research with PyTorch Lightning
5. Load and fine-tune models from HuggingFace Transformers
6. Choose the right ecosystem library for your task
7. Understand the relationship between PyTorch and its ecosystem libraries

---

PyTorch's power is amplified by a rich ecosystem of domain-specific libraries. This lesson surveys the most important ones, showing you how they fit together and when to use each.

---

## 1. torchvision

### 1.1 Pretrained Models

```python
import torch
import torchvision.models as models

# Load a pretrained model
resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
resnet.eval()

# Modern weights API (recommended)
weights = models.ResNet50_Weights.DEFAULT
resnet50 = models.resnet50(weights=weights)
preprocess = weights.transforms()

# List available models
print(models.list_models())  # all available model names

# Fine-tuning: replace the final layer
resnet.fc = torch.nn.Linear(512, 5)  # 5 classes instead of 1000
```

### 1.2 Datasets

```python
from torchvision import datasets, transforms

# Built-in datasets
mnist = datasets.MNIST('./data', train=True, download=True,
                        transform=transforms.ToTensor())

cifar10 = datasets.CIFAR10('./data', train=True, download=True,
                            transform=transforms.Compose([
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5),
                                                     (0.5, 0.5, 0.5)),
                            ]))

# ImageFolder for custom data
# Expects: root/class_name/image.jpg
custom = datasets.ImageFolder('./my_data/train',
                               transform=transforms.Compose([
                                   transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                               ]))
print(custom.classes)       # ['cat', 'dog']
print(custom.class_to_idx)  # {'cat': 0, 'dog': 1}
```

### 1.3 Transforms v2

```python
from torchvision.transforms import v2

transform = v2.Compose([
    v2.RandomResizedCrop(224),
    v2.RandomHorizontalFlip(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225]),
])

# v2 transforms work on tensors, PIL images, and bounding boxes
```

### 1.4 Utility Functions

```python
from torchvision.utils import make_grid, save_image
import matplotlib.pyplot as plt

# Create a grid of images for visualization
images = torch.randn(16, 3, 64, 64)  # 16 RGB images
grid = make_grid(images, nrow=4, normalize=True)
plt.imshow(grid.permute(1, 2, 0))
plt.savefig('grid.png')

# Save a single image
save_image(images[0], 'single_image.png')
```

---

## 2. torchaudio

### 2.1 Loading and Processing Audio

```python
import torchaudio

# Load audio file
waveform, sample_rate = torchaudio.load('audio.wav')
print(f"Shape: {waveform.shape}")  # [channels, samples]
print(f"Sample rate: {sample_rate}")

# Resample
resampler = torchaudio.transforms.Resample(
    orig_freq=sample_rate, new_freq=16000
)
waveform_16k = resampler(waveform)

# Mel spectrogram
mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=16000,
    n_fft=400,
    n_mels=128,
)
mel_spec = mel_transform(waveform_16k)
print(f"Mel spectrogram shape: {mel_spec.shape}")
```

### 2.2 Pretrained Models

```python
import torchaudio

# Speech recognition with wav2vec2
bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
model = bundle.get_model()
sample_rate = bundle.sample_rate
```

---

## 3. torchtext (Legacy Note)

```python
# Note: torchtext has been deprecated in favor of:
# 1. HuggingFace datasets + tokenizers for NLP
# 2. torchdata for general data loading

# Modern NLP pipeline uses HuggingFace (see Section 5)
```

---

## 4. PyTorch Lightning

### 4.1 What is Lightning?

PyTorch Lightning removes boilerplate from PyTorch training. You define the model and training logic; Lightning handles the rest (training loop, logging, checkpointing, distributed training).

### 4.2 LightningModule

```python
import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


class LitModel(L.LightningModule):
    def __init__(self, input_dim, hidden_dim, output_dim, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
        self.lr = lr

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
```

### 4.3 Training with Lightning

```python
# Data
X = torch.randn(1000, 20)
y = torch.randint(0, 5, (1000,))
train_set = TensorDataset(X[:800], y[:800])
val_set = TensorDataset(X[800:], y[800:])

train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
val_loader = DataLoader(val_set, batch_size=32)

# Model
model = LitModel(input_dim=20, hidden_dim=64, output_dim=5)

# Trainer
trainer = L.Trainer(
    max_epochs=20,
    accelerator='auto',          # CPU, GPU, or TPU
    devices='auto',              # number of devices
    callbacks=[
        L.callbacks.EarlyStopping(monitor='val_loss', patience=5),
        L.callbacks.ModelCheckpoint(monitor='val_loss'),
    ],
)

# Train
trainer.fit(model, train_loader, val_loader)

# Test
trainer.test(model, val_loader)
```

### 4.4 Lightning vs Vanilla PyTorch

| Feature | Vanilla PyTorch | Lightning |
|---------|----------------|-----------|
| Training loop | Write yourself | Automated |
| Multi-GPU | Wrap with DDP manually | `accelerator='gpu', devices=4` |
| Mixed precision | Manual GradScaler | `precision='16-mixed'` |
| Logging | Manual | Built-in (TensorBoard, W&B, etc.) |
| Checkpointing | Manual | `ModelCheckpoint` callback |
| Early stopping | Manual | `EarlyStopping` callback |
| Flexibility | Maximum | Very high (override any method) |
| Learning value | High (understand everything) | Lower (abstracts details) |

---

## 5. HuggingFace Ecosystem

### 5.1 Transformers Library

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load pretrained model and tokenizer
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name, num_labels=2
)

# Tokenize input
inputs = tokenizer("This movie was fantastic!", return_tensors="pt",
                    padding=True, truncation=True)

# Inference
model.eval()
with torch.no_grad():
    outputs = model(**inputs)
    predictions = torch.softmax(outputs.logits, dim=-1)
    print(f"Positive: {predictions[0][1]:.4f}")
```

### 5.2 Fine-Tuning with HuggingFace Trainer

```python
from transformers import Trainer, TrainingArguments
from datasets import load_dataset

# Load dataset
dataset = load_dataset("imdb")

# Tokenize
def tokenize_fn(examples):
    return tokenizer(examples["text"], padding="max_length",
                      truncation=True, max_length=512)

tokenized = dataset.map(tokenize_fn, batched=True)

# Training arguments
args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    weight_decay=0.01,
)

# Train
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["test"],
)
trainer.train()
```

### 5.3 Using HuggingFace Hub

```python
from huggingface_hub import hf_hub_download

# Download a specific file
path = hf_hub_download("bert-base-uncased", "config.json")

# Load model weights from Hub
from transformers import AutoModel
model = AutoModel.from_pretrained("bert-base-uncased")

# Push your model to Hub
# model.push_to_hub("my-username/my-model")
```

---

## 6. Other Notable Libraries

### 6.1 Overview

| Library | Domain | Key Feature |
|---------|--------|-------------|
| **timm** | Vision | 700+ pretrained image models |
| **Weights & Biases** | Experiment tracking | Dashboard, sweeps, artifacts |
| **Optuna** | Hyperparameter tuning | Bayesian optimization |
| **ONNX Runtime** | Inference | Cross-platform optimized runtime |
| **DeepSpeed** | Distributed training | ZeRO optimizer, model parallelism |
| **FSDP** | Distributed training | Built-in PyTorch, sharded data parallel |
| **torchmetrics** | Metrics | 100+ metrics (accuracy, F1, BLEU, etc.) |
| **Kornia** | Vision | Differentiable CV ops |

### 6.2 timm (PyTorch Image Models)

```python
import timm

# List available models
print(timm.list_models('efficientnet*'))

# Load pretrained model
model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=10)

# Get model-specific transforms
data_config = timm.data.resolve_data_config(model.pretrained_cfg)
transform = timm.data.create_transform(**data_config)
```

### 6.3 torchmetrics

```python
import torchmetrics

# Accuracy
accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=10)

# Update with predictions and targets
preds = torch.randn(32, 10)
target = torch.randint(0, 10, (32,))
accuracy.update(preds, target)

# Compute
print(f"Accuracy: {accuracy.compute():.4f}")

# Reset for next epoch
accuracy.reset()
```

---

## 7. Choosing the Right Tool

```
Start here:
    │
    ├── Need pretrained vision models?
    │   ├── Standard models (ResNet, ViT) → torchvision.models
    │   └── 700+ models with configs → timm
    │
    ├── Need pretrained NLP models?
    │   └── HuggingFace Transformers
    │
    ├── Want to reduce training boilerplate?
    │   └── PyTorch Lightning
    │
    ├── Need experiment tracking?
    │   └── Weights & Biases or TensorBoard
    │
    ├── Need to deploy models?
    │   ├── Python server → TorchServe or FastAPI
    │   ├── Cross-platform → ONNX + ONNX Runtime
    │   └── Mobile → ExecuTorch
    │
    └── Need multi-GPU training?
        ├── Simple → torch.nn.DataParallel
        ├── Recommended → DistributedDataParallel
        └── Large models → DeepSpeed or FSDP
```

---

## Summary

| Library | What It Does | When to Use |
|---------|-------------|-------------|
| torchvision | Vision models, datasets, transforms | Image classification, detection, segmentation |
| torchaudio | Audio processing, speech models | Speech recognition, audio classification |
| Lightning | Training framework abstraction | Reduce boilerplate, multi-GPU, logging |
| HuggingFace | NLP models, datasets, tokenizers | Text classification, generation, fine-tuning |
| timm | Extensive vision model zoo | When torchvision doesn't have your model |
| ONNX Runtime | Cross-platform inference | Production deployment |
| torchmetrics | Evaluation metrics | Clean metric tracking in training |

---

## What's Next?

Congratulations on completing PyTorch Fundamentals! You now have the practical skills to build, train, debug, and deploy neural networks with PyTorch. Here are recommended next steps:

- **[Deep_Learning](../Deep_Learning/00_Overview.md)**: Build CNNs, Transformers, GANs, and Diffusion models
- **[Machine_Learning](../Machine_Learning/00_Overview.md)**: Strengthen your ML theory foundations
- **[CUDA](../CUDA/00_Overview.md)**: Understand GPU programming at a lower level
- **[MLOps](../MLOps/00_Overview.md)**: Production ML pipelines, experiment tracking, CI/CD
