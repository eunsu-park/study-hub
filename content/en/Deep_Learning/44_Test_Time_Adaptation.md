[Previous: Few-Shot Learning](./43_Few_Shot_Learning.md)

---

# 44. Test-Time Adaptation

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain the domain shift problem and why pretrained models degrade on shifted distributions
2. Describe batch normalization adaptation as the simplest form of test-time adaptation
3. Implement TENT (Test-time Entropy Minimization) for adapting models during inference
4. Compare TTA approaches (BN adaptation, TENT, TTT, CoTTA) and their trade-offs
5. Apply test-time adaptation strategies to improve model robustness in deployment

---

## Table of Contents

1. [The Domain Shift Problem](#1-the-domain-shift-problem)
2. [Batch Normalization Adaptation](#2-batch-normalization-adaptation)
3. [TENT: Test-Time Entropy Minimization](#3-tent-test-time-entropy-minimization)
4. [Test-Time Training (TTT)](#4-test-time-training-ttt)
5. [Continual Test-Time Adaptation (CoTTA)](#5-continual-test-time-adaptation-cotta)
6. [Practical Deployment](#6-practical-deployment)
7. [Exercises](#7-exercises)

---

## 1. The Domain Shift Problem

### 1.1 What Is Domain Shift?

A model trained on source data (e.g., photos from cameras) often performs poorly on target data from a different distribution (e.g., sketches, corrupted images, different hospitals):

```
Source domain (training):              Target domain (deployment):

┌──────────────┐                      ┌──────────────┐
│  Clean photos│  Model achieves      │  Corrupted   │  Same model drops
│  from Lab    │  95% accuracy        │  images from │  to 60% accuracy
│  ImageNet    │  ──────────────►     │  real world  │
└──────────────┘                      └──────────────┘

Examples of domain shift:
  • Weather: sunny → foggy/rainy
  • Equipment: camera A → camera B
  • Style: photo → painting/sketch
  • Corruption: clean → noisy/blurred/compressed
  • Institution: hospital A → hospital B
```

### 1.2 Standard Solutions and Their Limits

| Approach | When Possible | Limitation |
|----------|---------------|------------|
| Domain adaptation | Before deployment (need target data) | Target data may not be available ahead of time |
| Data augmentation | During training | Cannot anticipate all possible shifts |
| Robust architectures | During training | Helps but doesn't eliminate the gap |
| **Test-time adaptation** | **During inference** | **Adapts on the fly — no target labels needed** |

TTA is unique because it requires **no labeled target data** and **no access to source data** during adaptation. It works with the model and the incoming test batch alone.

---

## 2. Batch Normalization Adaptation

### 2.1 Why BN Statistics Matter

Batch Normalization stores running mean (μ) and variance (σ²) computed during training. At test time, these stored statistics normalize the input. But if the test distribution differs from training, these statistics are wrong.

```python
# Standard BatchNorm at test time:
# Uses stored running_mean and running_var from training
output = (input - running_mean) / sqrt(running_var + eps) * gamma + beta

# Problem: running_mean and running_var represent the SOURCE domain
# When input comes from a DIFFERENT domain, normalization is off
```

### 2.2 BN Adapt: Replace with Test Statistics

The simplest TTA method: replace stored BN statistics with statistics from the current test batch.

```python
import torch
import torch.nn as nn


def adapt_bn(model, test_loader, num_batches=10):
    """Adapt BatchNorm statistics to test distribution.

    Simply runs forward passes on test data to update
    running mean/var, then freezes them for inference.
    """
    # Reset BN statistics
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
            module.reset_running_stats()
            module.train()  # Use batch statistics
            module.momentum = None  # Use cumulative moving average

    # Collect test-time statistics
    model.eval()
    with torch.no_grad():
        for i, (inputs, _) in enumerate(test_loader):
            if i >= num_batches:
                break
            # Forward pass updates BN running stats
            for module in model.modules():
                if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                    module.train()
            model(inputs)

    # Freeze BN back to eval mode
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
            module.eval()

    return model
```

### 2.3 Effectiveness

| Method | Clean Accuracy | Corrupted Accuracy (avg) |
|--------|---------------|--------------------------|
| Standard (no adapt) | 76.1% | 43.5% |
| BN Adapt | 76.1% | 57.2% |

BN adaptation is free (no extra training), requires no labels, and recovers a large portion of the accuracy lost to corruption. Its limitation: it only fixes normalization statistics, not the model weights.

---

## 3. TENT: Test-Time Entropy Minimization

### 3.1 The Idea

TENT (Wang et al., 2021) goes beyond BN statistics by actually updating model parameters at test time. The key: minimize the **entropy** of the model's predictions on test data.

```
High entropy prediction (uncertain):     Low entropy prediction (confident):
  cat:    0.25                             cat:    0.85
  dog:    0.25                             dog:    0.10
  bird:   0.25                             bird:    0.03
  fish:   0.25                             fish:    0.02

TENT minimizes entropy → pushes model toward confident predictions
If the model's architecture is good, confident = correct (usually)
```

### 3.2 Algorithm

```
For each test batch:
  1. Forward pass → compute predicted probabilities p
  2. Compute entropy: H(p) = -Σ p_i log(p_i)
  3. Backpropagate entropy loss
  4. Update ONLY the affine parameters (γ, β) of BatchNorm layers
  5. Use the updated model for prediction
```

### 3.3 Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD


class TENT:
    """Test-Time Entropy Minimization (TENT).

    Adapts a pretrained model at test time by minimizing the entropy
    of its predictions, updating only BatchNorm affine parameters.
    """

    def __init__(self, model, lr=0.001, steps=1):
        self.model = model
        self.steps = steps

        # Collect only BN affine parameters (γ and β)
        self.params = []
        for module in model.modules():
            if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                module.requires_grad_(True)
                self.params.extend([module.weight, module.bias])
            else:
                # Freeze all other parameters
                for param in module.parameters(recurse=False):
                    param.requires_grad_(False)

        self.optimizer = SGD(self.params, lr=lr, momentum=0.9)

    def adapt_and_predict(self, inputs):
        """Adapt model on the input batch, then predict."""
        self.model.train()  # BN uses batch statistics

        for _ in range(self.steps):
            outputs = self.model(inputs)
            loss = self._entropy_loss(outputs)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

        # Final prediction with adapted parameters
        self.model.eval()
        with torch.no_grad():
            return self.model(inputs)

    @staticmethod
    def _entropy_loss(logits):
        """Shannon entropy of softmax probabilities, averaged over batch."""
        probs = F.softmax(logits, dim=1)
        log_probs = F.log_softmax(logits, dim=1)
        entropy = -(probs * log_probs).sum(dim=1)
        return entropy.mean()
```

### 3.4 Why Only BN Parameters?

Updating all parameters would overfit to the test batch (especially with no labels). BN affine parameters (γ, β) are a minimal, low-risk set:
- They control the scale and shift of normalized features
- They can compensate for distribution shift without changing what features are detected
- Only 2 parameters per BN layer (vs. thousands per conv layer)

### 3.5 Results on ImageNet-C

| Method | Mean Corruption Error (%) |
|--------|--------------------------|
| Standard | 57.2 |
| BN Adapt | 40.5 |
| TENT (1 step) | 38.0 |
| TENT (10 steps) | 36.2 |

---

## 4. Test-Time Training (TTT)

### 4.1 The Approach

TTT (Sun et al., 2020) adds a self-supervised auxiliary task during training (e.g., predicting image rotation). At test time, it fine-tunes the shared encoder on this auxiliary task using the test input.

```
Training:
  ┌─────────┐     ┌────────────┐     ┌───────────────┐
  │  Input   │────►│  Shared    │────►│  Main task    │──► Classification
  │  image   │     │  Encoder   │     │  head         │     loss
  └─────────┘     └─────┬──────┘     └───────────────┘
                        │
                        └────────────►┌───────────────┐
                                      │  Rotation     │──► Self-supervised
                                      │  prediction   │     loss
                                      └───────────────┘

Test time:
  1. Receive test image
  2. Create rotated versions (0°, 90°, 180°, 270°)
  3. Fine-tune encoder on rotation prediction task
  4. Use adapted encoder for main classification
```

### 4.2 Key Insight

The rotation prediction task does not require labels but provides a training signal that adapts the feature extractor to the test data distribution. The assumption: if the encoder can correctly predict rotations on test data, its features are well-calibrated for that distribution.

---

## 5. Continual Test-Time Adaptation (CoTTA)

### 5.1 Challenge: Continual Shifts

In deployment, the distribution may shift continuously (e.g., weather changing throughout the day). Naively applying TENT to a sequence of different distributions leads to:
- **Error accumulation**: Wrong pseudo-labels compound over time
- **Catastrophic forgetting**: The model forgets source knowledge

### 5.2 CoTTA Solutions

CoTTA (Wang et al., 2022) addresses these with two mechanisms:

1. **Weight-averaged pseudo-labels**: Use an exponential moving average (EMA) teacher model for more stable pseudo-labels
2. **Stochastic restore**: Randomly restore a fraction of parameters to their source values each step, preventing drift

```python
class CoTTA:
    """Continual Test-Time Adaptation.

    Uses EMA teacher for stable pseudo-labels and stochastic
    restoration to prevent catastrophic forgetting.
    """

    def __init__(self, model, lr=0.001, ema_decay=0.999, restore_prob=0.01):
        self.student = model
        self.teacher = self._copy_model(model)  # EMA teacher
        self.source_params = {n: p.clone() for n, p in model.named_parameters()}
        self.ema_decay = ema_decay
        self.restore_prob = restore_prob

        # Only adapt BN params
        params = [p for n, p in model.named_parameters()
                  if 'bn' in n or 'norm' in n]
        self.optimizer = torch.optim.Adam(params, lr=lr)

    def adapt(self, inputs):
        # Get pseudo-labels from EMA teacher
        self.teacher.eval()
        with torch.no_grad():
            teacher_logits = self.teacher(inputs)
            pseudo_labels = teacher_logits.argmax(dim=1)
            confidence = F.softmax(teacher_logits, dim=1).max(dim=1)[0]

        # Train student on confident pseudo-labels
        self.student.train()
        mask = confidence > 0.9  # Only use confident predictions
        if mask.sum() > 0:
            student_logits = self.student(inputs[mask])
            loss = F.cross_entropy(student_logits, pseudo_labels[mask])
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

        # Update EMA teacher
        self._update_teacher()

        # Stochastic restore
        self._stochastic_restore()

        return self.student(inputs)

    def _update_teacher(self):
        for t_param, s_param in zip(self.teacher.parameters(),
                                     self.student.parameters()):
            t_param.data = self.ema_decay * t_param.data + \
                          (1 - self.ema_decay) * s_param.data

    def _stochastic_restore(self):
        for name, param in self.student.named_parameters():
            if name in self.source_params:
                mask = torch.rand_like(param) < self.restore_prob
                param.data[mask] = self.source_params[name][mask]

    @staticmethod
    def _copy_model(model):
        import copy
        teacher = copy.deepcopy(model)
        for p in teacher.parameters():
            p.requires_grad_(False)
        return teacher
```

---

## 6. Practical Deployment

### 6.1 When to Use TTA

| Situation | Recommended Approach |
|-----------|---------------------|
| Known, fixed shift (new camera) | BN Adapt (simplest, reliable) |
| Unknown corruption (weather, noise) | TENT (adapts on the fly) |
| Gradually changing distribution | CoTTA (handles continual shift) |
| Single test image (no batch) | TTT or instance-level BN adapt |
| Safety-critical application | BN Adapt only (most conservative) |

### 6.2 Batch Size Sensitivity

TTA methods depend on batch statistics. Very small batches (1-8) make BN statistics unreliable:

| Batch Size | BN Adapt Improvement | TENT Improvement |
|------------|---------------------|------------------|
| 1 | May hurt performance | Unreliable |
| 16 | Moderate | Good |
| 64 | Good | Best |
| 200 | Excellent | Excellent |

For single-image inference, consider instance normalization or feature-level adaptation instead.

### 6.3 Computational Cost

| Method | Extra Forward | Extra Backward | Memory Overhead |
|--------|--------------|----------------|----------------|
| BN Adapt | N batches × 1 | 0 | Negligible |
| TENT (1 step) | 1 per batch | 1 per batch | +BN params grad |
| TTT | K rotations | K rotations | +Aux head |
| CoTTA | 1 (teacher) + 1 (student) | 1 | +Teacher model |

---

## 7. Exercises

### Exercise 1: Distribution Shift Analysis

Take a pretrained ResNet-50 and evaluate it on ImageNet validation set with:
1. No corruption (baseline)
2. Gaussian noise (σ = 0.1, 0.3, 0.5)
3. Motion blur (kernel sizes 5, 15, 25)
4. JPEG compression (quality 10, 30, 50)

Plot accuracy vs. corruption severity. Which corruption type causes the largest accuracy drop?

### Exercise 2: BN Adaptation

Implement BN adaptation and test it on the corruptions from Exercise 1:
1. Adapt using 10, 50, and 100 batches of corrupted data
2. Compare adapted vs. non-adapted accuracy
3. What is the minimum number of batches needed for reliable adaptation?

### Exercise 3: TENT Implementation

Implement TENT from scratch:
1. Set up a pretrained model with gradient tracking only for BN affine parameters
2. Implement the entropy loss
3. Test with 1, 3, and 10 adaptation steps
4. Compare with BN Adapt: when does TENT provide additional benefit?

### Exercise 4: Batch Size Study

Study how batch size affects TTA quality:
1. Run TENT with batch sizes: 1, 4, 16, 64, 256
2. For batch size 1, try replacing BatchNorm with Instance Normalization
3. Plot accuracy vs. batch size and explain the trend

### Exercise 5: Continual Adaptation

Simulate a continual shift scenario:
1. Create a sequence of 10 different corruption types, each applied to 100 batches
2. Apply TENT naively (accumulates errors) and measure accuracy over time
3. Apply CoTTA (with EMA teacher and stochastic restore) and compare
4. Show that CoTTA prevents the performance degradation seen with naive TENT

---

*End of Lesson 44*
