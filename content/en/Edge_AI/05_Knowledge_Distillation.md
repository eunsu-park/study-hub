# Lesson 5: Knowledge Distillation

[Previous: Pruning](./04_Pruning.md) | [Next: Efficient Architectures](./06_Efficient_Architectures.md)

---

## Learning Objectives

- Understand the teacher-student framework and why soft targets carry more information than hard labels
- Implement knowledge distillation with temperature scaling and combined loss functions
- Apply feature matching and attention transfer for deeper knowledge transfer
- Explore self-distillation and online distillation as alternatives to the standard approach
- Analyze the DistilBERT case study as a real-world application of distillation
- Design distillation strategies for different model compression scenarios

---

## 1. The Teacher-Student Framework

Knowledge distillation (Hinton et al., 2015) trains a small **student** model to mimic the behavior of a larger, more accurate **teacher** model. The key insight is that the teacher's **soft probability outputs** contain richer information than the original hard labels.

```
Traditional Training:              Knowledge Distillation:
                                   ┌───────────────┐
                                   │  Teacher       │
Input ──▶ Student ──▶ Loss         │  (Large, Pre-  │
              │         ▲          │   trained)     │
              ▼         │          └───────┬───────┘
          Prediction  Hard Label           │ Soft Labels
           [0, 1, 0]  [0, 1, 0]           │ [0.02, 0.90, 0.08]
                                           │
                                   Input ──▶ Student ──▶ Loss
                                                │         ▲  ▲
                                                ▼         │  │
                                            Prediction  Hard  Soft
                                                       Label  Label
```

### 1.1 Why Soft Targets?

Consider a cat-vs-dog classifier. A hard label says "this is a cat" (100% cat, 0% dog). But the teacher's soft output might say "90% cat, 8% dog, 2% other" — revealing that this particular image has some dog-like features. This inter-class relationship information is **dark knowledge** that hard labels cannot capture.

```python
import torch
import torch.nn.functional as F


def demonstrate_soft_targets():
    """
    Show how soft targets carry more information than hard labels.
    """
    # Teacher logits for 3 images
    teacher_logits = torch.tensor([
        [5.0, 1.0, 0.5],   # Clearly class 0 (cat)
        [3.0, 2.5, 0.2],   # Class 0 (cat), but similar to class 1 (dog)
        [0.1, 0.2, 4.0],   # Clearly class 2 (bird)
    ])

    # Hard labels (argmax)
    hard_labels = teacher_logits.argmax(dim=1)
    print("Hard labels:", hard_labels.tolist())
    # [0, 0, 2] — Image 1 and 2 both labeled "cat", no nuance

    # Soft targets at different temperatures
    for T in [1.0, 2.0, 4.0, 10.0]:
        soft = F.softmax(teacher_logits / T, dim=1)
        print(f"\nTemperature T={T}:")
        for i, probs in enumerate(soft):
            print(f"  Image {i}: {probs.tolist()}")

    # At T=1: [0.93, 0.02, 0.01] — very peaked, little information
    # At T=4: [0.56, 0.24, 0.20] — softer, reveals class similarities
    # Higher T → softer distribution → more inter-class information


demonstrate_soft_targets()
```

---

## 2. Temperature Scaling

Temperature `T` controls the softness of probability distributions. Higher temperatures produce softer (more uniform) distributions that reveal more about the relationships between classes.

### 2.1 The Math

```
Standard softmax:           p_i = exp(z_i) / Σ_j exp(z_j)
Softmax with temperature:   p_i = exp(z_i / T) / Σ_j exp(z_j / T)

T = 1:  Standard softmax (peaked)
T > 1:  Softer distribution (reveals dark knowledge)
T → ∞:  Uniform distribution (no information)
T < 1:  Sharper distribution (amplifies differences)
```

```python
import torch
import torch.nn.functional as F


def temperature_softmax(logits, temperature):
    """Apply softmax with temperature scaling."""
    return F.softmax(logits / temperature, dim=-1)


def visualize_temperature_effect(logits):
    """Show how temperature affects the output distribution."""
    print(f"Logits: {logits.tolist()}")
    print(f"\n{'T':>6} {'Probs':>40} {'Entropy':>10}")
    print("-" * 60)

    for T in [0.5, 1.0, 2.0, 4.0, 8.0, 20.0]:
        probs = temperature_softmax(logits, T)
        entropy = -(probs * probs.log()).sum().item()
        probs_str = ", ".join(f"{p:.4f}" for p in probs.tolist())
        print(f"{T:>6.1f} [{probs_str}] {entropy:>10.4f}")


logits = torch.tensor([3.0, 1.0, 0.5, -1.0])
visualize_temperature_effect(logits)
```

### 2.2 Choosing Temperature

| Temperature | Effect | Use Case |
|------------|--------|----------|
| T = 1 | Standard softmax | No distillation |
| T = 2-5 | Moderate softening | General distillation (recommended start) |
| T = 5-10 | Very soft | When teacher is very confident |
| T = 10-20 | Near uniform | When classes are very similar |

The **optimal temperature** depends on the task and the teacher's confidence. A good starting point is T=4, then tune via validation accuracy.

---

## 3. Distillation Loss Function

The distillation loss combines two components:

1. **Soft loss (KL divergence)**: Measures how well the student matches the teacher's soft predictions
2. **Hard loss (cross-entropy)**: Standard supervised loss against ground-truth labels

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class DistillationLoss(nn.Module):
    """
    Combined knowledge distillation loss.

    L = alpha * T^2 * KL(soft_student || soft_teacher) + (1 - alpha) * CE(student, labels)

    The T^2 scaling compensates for the gradient magnitude change
    when using temperature > 1 (Hinton et al., 2015).
    """

    def __init__(self, temperature=4.0, alpha=0.7):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, student_logits, teacher_logits, true_labels):
        """
        Args:
            student_logits: Raw output from student (batch_size, num_classes)
            teacher_logits: Raw output from teacher (batch_size, num_classes)
            true_labels: Ground-truth class indices (batch_size,)

        Returns:
            Combined distillation loss (scalar)
        """
        T = self.temperature

        # Soft loss: KL divergence between softened distributions
        soft_student = F.log_softmax(student_logits / T, dim=1)
        soft_teacher = F.softmax(teacher_logits / T, dim=1)
        soft_loss = F.kl_div(
            soft_student,
            soft_teacher,
            reduction="batchmean",
        ) * (T ** 2)  # Scale by T^2 to match gradient magnitudes

        # Hard loss: standard cross-entropy
        hard_loss = self.ce_loss(student_logits, true_labels)

        # Combined
        total_loss = self.alpha * soft_loss + (1 - self.alpha) * hard_loss
        return total_loss


# Example
batch_size, num_classes = 4, 10
teacher_logits = torch.randn(batch_size, num_classes)
student_logits = torch.randn(batch_size, num_classes)
labels = torch.randint(0, num_classes, (batch_size,))

criterion = DistillationLoss(temperature=4.0, alpha=0.7)
loss = criterion(student_logits, teacher_logits, labels)
print(f"Distillation loss: {loss.item():.4f}")
```

### 3.1 Why T^2 Scaling?

When temperature T is used, the gradients of the KL divergence are scaled by 1/T^2. Multiplying by T^2 restores the gradient magnitude to be comparable with the hard loss, ensuring balanced training.

```python
import torch
import torch.nn.functional as F


def demonstrate_t_squared():
    """Show that T^2 scaling balances soft and hard loss gradients."""
    student_logits = torch.randn(4, 10, requires_grad=True)
    teacher_logits = torch.randn(4, 10)
    labels = torch.randint(0, 10, (4,))

    for T in [1.0, 4.0, 10.0]:
        student_logits.grad = None

        # Soft loss WITHOUT T^2 scaling
        soft_s = F.log_softmax(student_logits / T, dim=1)
        soft_t = F.softmax(teacher_logits / T, dim=1)
        loss_no_scale = F.kl_div(soft_s, soft_t, reduction="batchmean")
        loss_no_scale.backward(retain_graph=True)
        grad_no_scale = student_logits.grad.norm().item()

        student_logits.grad = None

        # Soft loss WITH T^2 scaling
        loss_scaled = F.kl_div(soft_s, soft_t, reduction="batchmean") * (T ** 2)
        loss_scaled.backward(retain_graph=True)
        grad_scaled = student_logits.grad.norm().item()

        # Hard loss gradient
        student_logits.grad = None
        hard_loss = F.cross_entropy(student_logits, labels)
        hard_loss.backward()
        grad_hard = student_logits.grad.norm().item()

        print(f"T={T:>4.0f}  "
              f"Grad(soft, no T^2): {grad_no_scale:.4f}  "
              f"Grad(soft, T^2): {grad_scaled:.4f}  "
              f"Grad(hard): {grad_hard:.4f}")


demonstrate_t_squared()
```

---

## 4. Complete Distillation Training Loop

```python
import torch
import torch.nn as nn
import torch.optim as optim


class KnowledgeDistiller:
    """
    Complete knowledge distillation trainer.

    Trains a student model to mimic a pretrained teacher.
    """

    def __init__(self, teacher, student, temperature=4.0, alpha=0.7,
                 lr=1e-3, device="cpu"):
        self.teacher = teacher.to(device).eval()
        self.student = student.to(device)
        self.device = device
        self.criterion = DistillationLoss(temperature, alpha)
        self.optimizer = optim.Adam(student.parameters(), lr=lr)

        # Freeze teacher weights
        for param in self.teacher.parameters():
            param.requires_grad = False

        # Count parameters
        teacher_params = sum(p.numel() for p in teacher.parameters())
        student_params = sum(p.numel() for p in student.parameters())
        print(f"Teacher parameters: {teacher_params:,}")
        print(f"Student parameters: {student_params:,}")
        print(f"Compression ratio:  {teacher_params / student_params:.1f}x")

    def train_epoch(self, train_loader):
        """Train student for one epoch."""
        self.student.train()
        total_loss = 0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            # Get teacher predictions (no gradient needed)
            with torch.no_grad():
                teacher_logits = self.teacher(inputs)

            # Get student predictions
            student_logits = self.student(inputs)

            # Compute distillation loss
            loss = self.criterion(student_logits, teacher_logits, labels)

            # Backpropagate
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            _, predicted = student_logits.max(1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        return {
            "loss": total_loss / len(train_loader),
            "accuracy": 100.0 * correct / total,
        }

    def evaluate(self, test_loader):
        """Evaluate student on test set."""
        self.student.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                outputs = self.student(inputs)
                _, predicted = outputs.max(1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        return 100.0 * correct / total

    def distill(self, train_loader, test_loader, epochs=20):
        """Full distillation training loop."""
        # Evaluate teacher baseline
        self.teacher.eval()
        teacher_acc = self._evaluate_model(self.teacher, test_loader)
        print(f"\nTeacher accuracy: {teacher_acc:.2f}%")

        # Evaluate student before distillation
        student_baseline = self.evaluate(test_loader)
        print(f"Student accuracy (before distillation): {student_baseline:.2f}%\n")

        best_acc = 0
        for epoch in range(epochs):
            metrics = self.train_epoch(train_loader)
            test_acc = self.evaluate(test_loader)

            if test_acc > best_acc:
                best_acc = test_acc
                best_state = {k: v.clone() for k, v in self.student.state_dict().items()}

            print(f"Epoch {epoch+1:>3}/{epochs} — "
                  f"Loss: {metrics['loss']:.4f}, "
                  f"Train: {metrics['accuracy']:.1f}%, "
                  f"Test: {test_acc:.2f}%"
                  f"{'  *best*' if test_acc == best_acc else ''}")

        # Load best model
        self.student.load_state_dict(best_state)
        print(f"\nDistillation complete!")
        print(f"  Teacher accuracy: {teacher_acc:.2f}%")
        print(f"  Student accuracy: {best_acc:.2f}%")
        print(f"  Gap: {teacher_acc - best_acc:.2f}%")

        return self.student

    def _evaluate_model(self, model, test_loader):
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
        return 100.0 * correct / total


# Usage example (with real models and data):
# teacher = torchvision.models.resnet50(weights="IMAGENET1K_V1")
# student = torchvision.models.mobilenet_v2()
# distiller = KnowledgeDistiller(teacher, student, temperature=4.0, alpha=0.7)
# distilled_student = distiller.distill(train_loader, test_loader, epochs=50)
```

---

## 5. Feature Matching

Beyond matching final outputs, **feature matching** (FitNets, Romero et al., 2015) forces the student's intermediate representations to match the teacher's.

```
Teacher:                          Student:
┌─────────┐                      ┌─────────┐
│ Layer 1  │──────────────────────│ Layer 1  │
│ (64 ch)  │    Feature Match    │ (16 ch)  │
└────┬─────┘    ←───────────     └────┬─────┘
     │          L2 loss on            │
┌────▼─────┐    intermediate     ┌────▼─────┐
│ Layer 2  │    features         │ Layer 2  │
│ (128 ch) │──────────────────────│ (32 ch)  │
└────┬─────┘                     └────┬─────┘
     │                                │
┌────▼─────┐                     ┌────▼─────┐
│ Output   │    KD Loss          │ Output   │
└──────────┘    ←───────────     └──────────┘
```

```python
import torch
import torch.nn as nn


class FeatureMatchingLoss(nn.Module):
    """
    Feature matching loss that aligns intermediate representations
    between teacher and student networks.

    Since teacher and student may have different channel dimensions,
    a learnable projection layer is used to match dimensions.
    """

    def __init__(self, teacher_channels, student_channels):
        """
        Args:
            teacher_channels: list of channel dims at matching points
            student_channels: list of channel dims at matching points
        """
        super().__init__()
        assert len(teacher_channels) == len(student_channels)

        # Projection layers to match dimensions
        self.projectors = nn.ModuleList()
        for t_ch, s_ch in zip(teacher_channels, student_channels):
            if t_ch != s_ch:
                self.projectors.append(
                    nn.Conv2d(s_ch, t_ch, kernel_size=1, bias=False)
                )
            else:
                self.projectors.append(nn.Identity())

    def forward(self, teacher_features, student_features):
        """
        Args:
            teacher_features: list of tensors from teacher intermediate layers
            student_features: list of tensors from student intermediate layers

        Returns:
            feature matching loss (scalar)
        """
        total_loss = 0
        for t_feat, s_feat, proj in zip(teacher_features, student_features,
                                         self.projectors):
            # Project student features to teacher's dimension
            s_projected = proj(s_feat)

            # Normalize features (important for stable training)
            t_norm = F.normalize(t_feat.view(t_feat.size(0), -1), dim=1)
            s_norm = F.normalize(s_projected.view(s_projected.size(0), -1), dim=1)

            # L2 loss
            total_loss += F.mse_loss(s_norm, t_norm)

        return total_loss / len(teacher_features)


class FeatureExtractor(nn.Module):
    """
    Wraps a model to extract intermediate features at specified layers.
    """

    def __init__(self, model, layer_names):
        super().__init__()
        self.model = model
        self.layer_names = layer_names
        self.features = {}

        # Register hooks
        for name, module in model.named_modules():
            if name in layer_names:
                module.register_forward_hook(self._make_hook(name))

    def _make_hook(self, name):
        def hook(module, input, output):
            self.features[name] = output
        return hook

    def forward(self, x):
        self.features = {}
        output = self.model(x)
        feature_list = [self.features[name] for name in self.layer_names]
        return output, feature_list


# Example usage
# teacher_extractor = FeatureExtractor(teacher, ["layer1", "layer2", "layer3"])
# student_extractor = FeatureExtractor(student, ["features.3", "features.6", "features.12"])
# feature_loss = FeatureMatchingLoss([64, 128, 256], [24, 32, 64])
```

---

## 6. Attention Transfer

Attention transfer (Zagoruyko & Komodakis, 2017) transfers the spatial attention maps from teacher to student. Attention maps indicate which spatial regions the network focuses on.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_attention_map(feature_map, mode="mean"):
    """
    Compute spatial attention map from a feature map.

    Args:
        feature_map: tensor of shape (B, C, H, W)
        mode: "mean" (average across channels) or "max"

    Returns:
        attention map of shape (B, H*W) — normalized
    """
    if mode == "mean":
        # Average absolute activation across channels
        attention = feature_map.abs().mean(dim=1)  # (B, H, W)
    elif mode == "max":
        attention = feature_map.abs().max(dim=1)[0]  # (B, H, W)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Flatten spatial dimensions and L2-normalize
    B = attention.size(0)
    attention = attention.view(B, -1)  # (B, H*W)
    attention = F.normalize(attention, p=2, dim=1)
    return attention


class AttentionTransferLoss(nn.Module):
    """
    Attention Transfer loss: minimize the difference between
    teacher and student attention maps at multiple layers.

    L_AT = Σ_l || A_teacher^l / ||A_teacher^l||_2
               - A_student^l / ||A_student^l||_2 ||^2
    """

    def __init__(self, beta=1000.0):
        super().__init__()
        self.beta = beta  # Weight for attention loss

    def forward(self, teacher_features, student_features):
        """
        Args:
            teacher_features: list of (B, C, H, W) tensors
            student_features: list of (B, C, H, W) tensors
        """
        loss = 0
        for t_feat, s_feat in zip(teacher_features, student_features):
            t_attention = compute_attention_map(t_feat)
            s_attention = compute_attention_map(s_feat)

            # If spatial dimensions differ, resize student to match teacher
            if t_attention.shape != s_attention.shape:
                # Reshape back to 2D, interpolate, then flatten
                t_h = int(t_attention.shape[1] ** 0.5)
                s_h = int(s_attention.shape[1] ** 0.5)
                s_2d = s_attention.view(-1, 1, s_h, s_h)
                s_resized = F.interpolate(s_2d, size=(t_h, t_h), mode="bilinear")
                s_attention = s_resized.view(s_attention.size(0), -1)
                s_attention = F.normalize(s_attention, p=2, dim=1)

            loss += F.mse_loss(s_attention, t_attention)

        return self.beta * loss / len(teacher_features)


# Example
teacher_feats = [torch.randn(4, 64, 32, 32), torch.randn(4, 128, 16, 16)]
student_feats = [torch.randn(4, 32, 32, 32), torch.randn(4, 64, 16, 16)]

at_loss = AttentionTransferLoss(beta=1000.0)
loss = at_loss(teacher_feats, student_feats)
print(f"Attention transfer loss: {loss.item():.4f}")
```

---

## 7. Self-Distillation and Online Distillation

### 7.1 Self-Distillation

In self-distillation, the model distills knowledge from itself — typically from deeper layers to shallower layers, or from a later training checkpoint to an earlier one.

```
Self-Distillation (Born-Again Networks):

Round 1:  Train Model_1 from scratch
Round 2:  Train Model_2 (same architecture) with Model_1 as teacher
Round 3:  Train Model_3 with Model_2 as teacher
...

Surprisingly, Model_2 often outperforms Model_1,
even though they have identical architectures.
```

```python
import torch
import torch.nn as nn
import copy


def born_again_distillation(model_class, model_kwargs, train_loader,
                             test_loader, num_generations=3, epochs=20):
    """
    Born-Again Networks: iterative self-distillation.

    Each generation trains a new model of the SAME architecture
    using the previous generation as teacher.
    """
    results = []

    # Generation 0: train from scratch
    print("=== Generation 0: Training from scratch ===")
    teacher = model_class(**model_kwargs)
    teacher = train_from_scratch(teacher, train_loader, epochs)
    acc = evaluate(teacher, test_loader)
    results.append({"generation": 0, "accuracy": acc})
    print(f"Generation 0 accuracy: {acc:.2f}%\n")

    for gen in range(1, num_generations + 1):
        print(f"=== Generation {gen}: Distilling from Generation {gen-1} ===")
        student = model_class(**model_kwargs)

        # Distill from previous generation
        distiller = KnowledgeDistiller(
            teacher=teacher,
            student=student,
            temperature=4.0,
            alpha=0.7,
        )
        student = distiller.distill(train_loader, test_loader, epochs)
        acc = evaluate(student, test_loader)

        results.append({"generation": gen, "accuracy": acc})
        print(f"Generation {gen} accuracy: {acc:.2f}%\n")

        # This generation becomes the teacher for the next
        teacher = student

    # Summary
    print("=== Self-Distillation Results ===")
    for r in results:
        print(f"  Generation {r['generation']}: {r['accuracy']:.2f}%")

    return teacher, results
```

### 7.2 Online Distillation

In online distillation, teacher and student train simultaneously. There is no pretrained teacher.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class OnlineDistillation(nn.Module):
    """
    Deep Mutual Learning (Zhang et al., 2018).

    Two networks train simultaneously, each teaching the other.
    Both networks improve beyond what either could achieve alone.

    Net1 learns from: hard labels + soft labels from Net2
    Net2 learns from: hard labels + soft labels from Net1
    """

    def __init__(self, net1, net2, temperature=3.0, alpha=0.5):
        super().__init__()
        self.net1 = net1
        self.net2 = net2
        self.temperature = temperature
        self.alpha = alpha

    def mutual_loss(self, logits_a, logits_b, labels):
        """Compute mutual learning loss for network A."""
        T = self.temperature

        # Hard loss
        ce_loss = F.cross_entropy(logits_a, labels)

        # Soft loss (learn from peer)
        soft_a = F.log_softmax(logits_a / T, dim=1)
        soft_b = F.softmax(logits_b.detach() / T, dim=1)  # Detach peer
        kl_loss = F.kl_div(soft_a, soft_b, reduction="batchmean") * (T ** 2)

        return (1 - self.alpha) * ce_loss + self.alpha * kl_loss

    def train_step(self, inputs, labels, optimizer1, optimizer2):
        """One training step for both networks."""
        # Forward pass for both
        logits1 = self.net1(inputs)
        logits2 = self.net2(inputs)

        # Compute losses
        loss1 = self.mutual_loss(logits1, logits2, labels)
        loss2 = self.mutual_loss(logits2, logits1, labels)

        # Update net1
        optimizer1.zero_grad()
        loss1.backward(retain_graph=True)
        optimizer1.step()

        # Update net2
        optimizer2.zero_grad()
        loss2.backward()
        optimizer2.step()

        return loss1.item(), loss2.item()


# Usage:
# net1 = SmallModel()
# net2 = SmallModel()
# dml = OnlineDistillation(net1, net2)
# opt1 = torch.optim.Adam(net1.parameters(), lr=1e-3)
# opt2 = torch.optim.Adam(net2.parameters(), lr=1e-3)
#
# for inputs, labels in train_loader:
#     loss1, loss2 = dml.train_step(inputs, labels, opt1, opt2)
```

---

## 8. Case Study: DistilBERT

DistilBERT (Sanh et al., 2019) is one of the most successful applications of knowledge distillation, compressing BERT-Base (110M parameters) to DistilBERT (66M parameters) while retaining 97% of BERT's language understanding capability.

### 8.1 Architecture

```
BERT-Base (Teacher):           DistilBERT (Student):
┌────────────────────┐        ┌────────────────────┐
│ Embedding Layer    │        │ Embedding Layer    │
│ (same)             │        │ (same)             │
├────────────────────┤        ├────────────────────┤
│ Transformer × 12   │        │ Transformer × 6    │  ← Half the layers
│ (hidden_dim=768)   │        │ (hidden_dim=768)   │
├────────────────────┤        ├────────────────────┤
│ Output Head        │        │ Output Head        │
└────────────────────┘        └────────────────────┘

Parameters: 110M                Parameters: 66M (40% smaller)
Inference:  ~45ms               Inference:  ~25ms (60% faster)
GLUE score: 79.5                GLUE score: 77.0 (97% of BERT)
```

### 8.2 DistilBERT's Triple Loss

DistilBERT uses three loss components:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class DistilBERTLoss(nn.Module):
    """
    DistilBERT training loss with three components:

    1. Distillation loss (KL divergence with temperature)
    2. Masked language modeling loss (hard labels)
    3. Cosine embedding loss (align hidden states)
    """

    def __init__(self, temperature=2.0, alpha_ce=0.5, alpha_mlm=0.5, alpha_cos=1.0):
        super().__init__()
        self.temperature = temperature
        self.alpha_ce = alpha_ce    # Weight for distillation loss
        self.alpha_mlm = alpha_mlm  # Weight for MLM loss
        self.alpha_cos = alpha_cos  # Weight for cosine loss
        self.ce = nn.CrossEntropyLoss()

    def forward(self, student_logits, teacher_logits,
                student_hidden, teacher_hidden,
                mlm_labels=None):
        """
        Args:
            student_logits: Student MLM output (B, seq_len, vocab_size)
            teacher_logits: Teacher MLM output (B, seq_len, vocab_size)
            student_hidden: Student hidden states (B, seq_len, hidden_dim)
            teacher_hidden: Teacher hidden states (B, seq_len, hidden_dim)
            mlm_labels: Ground truth MLM labels (B, seq_len), -100 for non-masked
        """
        T = self.temperature

        # 1. Distillation loss (soft targets)
        s_flat = student_logits.view(-1, student_logits.size(-1))
        t_flat = teacher_logits.view(-1, teacher_logits.size(-1))
        soft_loss = F.kl_div(
            F.log_softmax(s_flat / T, dim=1),
            F.softmax(t_flat / T, dim=1),
            reduction="batchmean",
        ) * (T ** 2)

        # 2. MLM loss (hard labels)
        mlm_loss = torch.tensor(0.0)
        if mlm_labels is not None:
            mlm_loss = self.ce(
                student_logits.view(-1, student_logits.size(-1)),
                mlm_labels.view(-1),
            )

        # 3. Cosine embedding loss (hidden state alignment)
        # Aligns the direction of student and teacher hidden states
        cos_loss = 1.0 - F.cosine_similarity(
            student_hidden.view(-1, student_hidden.size(-1)),
            teacher_hidden.view(-1, teacher_hidden.size(-1)),
            dim=1,
        ).mean()

        total = (self.alpha_ce * soft_loss
                 + self.alpha_mlm * mlm_loss
                 + self.alpha_cos * cos_loss)

        return total, {
            "distill": soft_loss.item(),
            "mlm": mlm_loss.item(),
            "cosine": cos_loss.item(),
        }
```

### 8.3 Layer Selection Strategy

DistilBERT uses 6 out of BERT's 12 layers. The student initializes from every other teacher layer:

```python
def initialize_distilbert_from_bert(bert_model, distilbert_model):
    """
    Initialize DistilBERT layers from every other BERT layer.

    BERT layers:     0  1  2  3  4  5  6  7  8  9  10  11
    DistilBERT:      ↑     ↑     ↑     ↑     ↑       ↑
    maps to:         0     1     2     3     4       5
    """
    bert_layers = bert_model.encoder.layer
    distil_layers = distilbert_model.transformer.layer

    # Map: DistilBERT layer i ← BERT layer 2*i
    for i in range(len(distil_layers)):
        teacher_idx = 2 * i
        print(f"DistilBERT layer {i} ← BERT layer {teacher_idx}")

        # Copy self-attention weights
        distil_layers[i].attention.load_state_dict(
            bert_layers[teacher_idx].attention.state_dict()
        )
        # Copy feed-forward weights
        distil_layers[i].ffn.load_state_dict(
            bert_layers[teacher_idx].output.state_dict()
        )

    # Copy embeddings directly
    distilbert_model.embeddings.load_state_dict(
        bert_model.embeddings.state_dict()
    )

    print(f"\nInitialized {len(distil_layers)} DistilBERT layers "
          f"from {len(bert_layers)} BERT layers")
```

### 8.4 Results Summary

| Model | Params | Size | GLUE Score | Inference | Retained |
|-------|--------|------|-----------|-----------|----------|
| BERT-Base | 110M | 440 MB | 79.5 | 45ms | 100% |
| DistilBERT | 66M | 264 MB | 77.0 | 25ms | 97% |
| TinyBERT-6L | 66M | 264 MB | 79.5 | 25ms | 100% |
| MiniLM-6L | 66M | 264 MB | 80.4 | 25ms | 101% |

---

## 9. Distillation Design Choices

### 9.1 Choosing the Student Architecture

```python
def student_architecture_guidelines(teacher_params, target_compression):
    """
    Rules of thumb for choosing student architecture.
    """
    student_params = teacher_params / target_compression

    guidelines = {
        "Width reduction": "Reduce hidden dimensions by sqrt(compression_ratio)",
        "Depth reduction": "Remove every other layer (DistilBERT approach)",
        "Combined": "Reduce both width and depth for larger compressions",
    }

    print(f"Teacher: {teacher_params/1e6:.0f}M parameters")
    print(f"Target:  {student_params/1e6:.0f}M parameters ({target_compression}x)")
    print(f"\nStrategies:")

    if target_compression <= 2:
        print(f"  Recommended: Depth reduction")
        print(f"  Remove every other layer (keep {int(12/target_compression)} of 12)")
    elif target_compression <= 4:
        print(f"  Recommended: Combined width + depth")
        print(f"  Reduce layers by 2x AND hidden dim by {target_compression/2:.1f}x")
    else:
        print(f"  Recommended: Different architecture family")
        print(f"  Consider MobileNet, EfficientNet, or custom tiny model")

    return guidelines


student_architecture_guidelines(110e6, 2)   # BERT → DistilBERT
student_architecture_guidelines(110e6, 10)  # BERT → tiny model
```

### 9.2 Alpha and Temperature Tuning

```python
def hyperparameter_search_distillation():
    """
    Recommended hyperparameter search grid for distillation.
    """
    grid = {
        "temperature": [1, 2, 4, 8, 16],
        "alpha": [0.1, 0.3, 0.5, 0.7, 0.9],
        "learning_rate": [1e-4, 5e-4, 1e-3],
    }

    print("Distillation Hyperparameter Search Grid:")
    print(f"  Temperature: {grid['temperature']}")
    print(f"  Alpha (soft weight): {grid['alpha']}")
    print(f"  Learning rate: {grid['learning_rate']}")
    print(f"\n  Total configurations: "
          f"{len(grid['temperature']) * len(grid['alpha']) * len(grid['learning_rate'])}")

    print("\nGuidelines:")
    print("  - Temperature: Start at 4. Increase if teacher is very confident.")
    print("  - Alpha: Start at 0.7 (70% soft, 30% hard). "
          "Decrease if labels are noisy.")
    print("  - LR: Usually lower than training from scratch (1e-4 to 1e-3).")
    print("  - Epochs: Distillation typically needs fewer epochs than "
          "training from scratch.")


hyperparameter_search_distillation()
```

---

## Summary

| Concept | Key Takeaway |
|---------|-------------|
| **Knowledge Distillation** | Train a small student to mimic a large teacher |
| **Soft targets** | Teacher's probability outputs carry inter-class relationship information |
| **Temperature** | Higher T produces softer distributions with more dark knowledge |
| **Distillation loss** | alpha * KL(soft) + (1-alpha) * CE(hard), scaled by T^2 |
| **Feature matching** | Align intermediate representations for deeper knowledge transfer |
| **Attention transfer** | Transfer spatial attention maps from teacher to student |
| **Self-distillation** | Distill from the model itself (Born-Again Networks) |
| **Online distillation** | Two models teach each other simultaneously |
| **DistilBERT** | 40% smaller, 60% faster, retains 97% of BERT's accuracy |

---

## Exercises

### Exercise 1: Basic Knowledge Distillation

1. Train a large teacher (ResNet-18) on CIFAR-10 to >93% accuracy
2. Define a small student (3-layer CNN with ~50K parameters)
3. Train the student normally (without distillation) and record accuracy
4. Train the student with distillation (T=4, alpha=0.7) and record accuracy
5. How much does distillation improve the student?

### Exercise 2: Temperature Exploration

Using the teacher-student pair from Exercise 1:
1. Distill with temperatures T = 1, 2, 4, 8, 16, 32
2. Plot student accuracy vs temperature
3. At what temperature is accuracy maximized?
4. Inspect the teacher's average confidence (max probability). Is there a relationship between teacher confidence and optimal temperature?

### Exercise 3: Feature Matching

1. Extend the distillation setup from Exercise 1 with feature matching
2. Match features at 2 intermediate layers (after conv blocks)
3. Compare distillation with: (a) output-only KD, (b) feature matching only, (c) both combined
4. Which approach works best?

### Exercise 4: Self-Distillation

1. Train a ResNet-18 on CIFAR-10 (Generation 0)
2. Distill into a new ResNet-18 of the same architecture (Generation 1)
3. Repeat for Generation 2 and 3
4. Plot accuracy across generations. Does accuracy keep improving?

### Exercise 5: DistilBERT-Style Compression

Using Hugging Face Transformers:
1. Load a pretrained BERT-Base model
2. Create a 6-layer version by taking every other layer
3. Distill on a text classification task (e.g., SST-2)
4. Compare: (a) BERT-Base (teacher), (b) 6-layer BERT trained from scratch, (c) 6-layer distilled
5. How much of the teacher's accuracy does the distilled model retain?

---

[Previous: Pruning](./04_Pruning.md) | [Overview](./00_Overview.md) | [Next: Efficient Architectures](./06_Efficient_Architectures.md)

**License**: CC BY-NC 4.0
