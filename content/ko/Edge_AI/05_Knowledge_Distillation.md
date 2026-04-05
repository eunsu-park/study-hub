# 레슨 5: Knowledge Distillation

[이전: Pruning](./04_Pruning.md) | [다음: Efficient Architectures](./06_Efficient_Architectures.md)

---

## 학습 목표

- teacher-student 프레임워크를 이해하고, soft target이 hard label보다 더 많은 정보를 담는 이유를 파악합니다
- temperature scaling과 결합 손실 함수를 사용하여 knowledge distillation을 구현합니다
- feature matching과 attention transfer를 활용하여 더 깊은 지식 전이를 수행합니다
- 표준 접근법의 대안으로서 self-distillation과 online distillation을 탐구합니다
- 실제 적용 사례로서 DistilBERT 케이스 스터디를 분석합니다
- 다양한 모델 압축 시나리오에 적합한 distillation 전략을 설계합니다

---

## 1. Teacher-Student 프레임워크

Knowledge distillation(Hinton et al., 2015)은 크고 정확한 **teacher** 모델의 행동을 작은 **student** 모델이 모방하도록 학습시키는 기법입니다. 핵심 통찰은 teacher의 **소프트 확률 출력**이 원본 hard label보다 더 풍부한 정보를 담고 있다는 것입니다.

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

### 1.1 Soft Target의 필요성

고양이-개 분류기를 예로 들어보겠습니다. Hard label은 "이것은 고양이입니다"라고만 말합니다(고양이 100%, 개 0%). 그러나 teacher의 soft 출력은 "고양이 90%, 개 8%, 기타 2%"라고 할 수 있으며, 이 특정 이미지에 개와 유사한 특징이 있음을 드러냅니다. 이러한 클래스 간 관계 정보가 바로 hard label로는 포착할 수 없는 **dark knowledge**입니다.

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

Temperature `T`는 확률 분포의 부드러움을 제어합니다. 높은 temperature는 더 부드러운(더 균일한) 분포를 생성하여 클래스 간 관계에 대한 더 많은 정보를 드러냅니다.

### 2.1 수학적 배경

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

### 2.2 Temperature 선택 가이드

| Temperature | 효과 | 사용 사례 |
|------------|------|----------|
| T = 1 | 표준 softmax | distillation 미적용 |
| T = 2-5 | 적당한 부드러움 | 일반적인 distillation (권장 시작점) |
| T = 5-10 | 매우 부드러움 | teacher의 확신도가 매우 높을 때 |
| T = 10-20 | 거의 균등 분포 | 클래스 간 유사도가 매우 높을 때 |

**최적 temperature**는 태스크와 teacher의 확신도에 따라 달라집니다. T=4를 시작점으로 하여 검증 정확도를 기준으로 튜닝하는 것이 좋습니다.

---

## 3. Distillation 손실 함수

Distillation 손실은 두 가지 구성 요소를 결합합니다:

1. **Soft loss (KL divergence)**: student가 teacher의 soft 예측을 얼마나 잘 일치시키는지 측정합니다
2. **Hard loss (cross-entropy)**: 실제 정답 레이블에 대한 표준 지도학습 손실입니다

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

### 3.1 T^2 스케일링의 이유

Temperature T를 사용하면 KL divergence의 기울기가 1/T^2으로 스케일링됩니다. T^2을 곱하면 기울기 크기가 hard loss와 비교 가능한 수준으로 복원되어, 균형 잡힌 학습이 보장됩니다.

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

## 4. 완전한 Distillation 학습 루프

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

최종 출력을 일치시키는 것 외에도, **feature matching**(FitNets, Romero et al., 2015)은 student의 중간 표현을 teacher의 것과 일치하도록 강제합니다.

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

Attention transfer(Zagoruyko & Komodakis, 2017)는 teacher의 공간 어텐션 맵을 student에게 전이합니다. 어텐션 맵은 네트워크가 어떤 공간적 영역에 집중하는지를 나타냅니다.

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

## 7. Self-Distillation과 Online Distillation

### 7.1 Self-Distillation

Self-distillation에서는 모델이 자기 자신으로부터 지식을 추출합니다. 일반적으로 더 깊은 레이어에서 더 얕은 레이어로, 또는 후기 학습 체크포인트에서 초기 체크포인트로 지식을 전이합니다.

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

Online distillation에서는 teacher와 student가 동시에 학습합니다. 사전 학습된 teacher가 필요하지 않습니다.

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

## 8. 케이스 스터디: DistilBERT

DistilBERT(Sanh et al., 2019)는 knowledge distillation의 가장 성공적인 적용 사례 중 하나로, BERT-Base(1억 1천만 파라미터)를 DistilBERT(6,600만 파라미터)로 압축하면서 BERT 언어 이해 능력의 97%를 유지합니다.

### 8.1 아키텍처

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

### 8.2 DistilBERT의 삼중 손실

DistilBERT는 세 가지 손실 구성 요소를 사용합니다:

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

### 8.3 레이어 선택 전략

DistilBERT는 BERT의 12개 레이어 중 6개를 사용합니다. Student는 teacher의 격번 레이어로부터 초기화됩니다:

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

### 8.4 결과 요약

| 모델 | 파라미터 | 크기 | GLUE 점수 | 추론 시간 | 성능 유지율 |
|------|---------|------|----------|----------|-----------|
| BERT-Base | 110M | 440 MB | 79.5 | 45ms | 100% |
| DistilBERT | 66M | 264 MB | 77.0 | 25ms | 97% |
| TinyBERT-6L | 66M | 264 MB | 79.5 | 25ms | 100% |
| MiniLM-6L | 66M | 264 MB | 80.4 | 25ms | 101% |

---

## 9. Distillation 설계 선택

### 9.1 Student 아키텍처 선택

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

### 9.2 Alpha와 Temperature 튜닝

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

## 요약

| 개념 | 핵심 요점 |
|------|----------|
| **Knowledge Distillation** | 작은 student를 큰 teacher를 모방하도록 학습시킵니다 |
| **Soft target** | Teacher의 확률 출력은 클래스 간 관계 정보를 담고 있습니다 |
| **Temperature** | 높은 T는 더 많은 dark knowledge가 포함된 부드러운 분포를 생성합니다 |
| **Distillation loss** | alpha * KL(soft) + (1-alpha) * CE(hard), T^2으로 스케일링 |
| **Feature matching** | 더 깊은 지식 전이를 위해 중간 표현을 정렬합니다 |
| **Attention transfer** | teacher의 공간 어텐션 맵을 student에게 전이합니다 |
| **Self-distillation** | 모델 자체로부터 distillation을 수행합니다 (Born-Again Networks) |
| **Online distillation** | 두 모델이 동시에 서로를 가르칩니다 |
| **DistilBERT** | 40% 작고, 60% 빠르며, BERT 정확도의 97%를 유지합니다 |

---

## 연습 문제

### 연습 문제 1: 기본 Knowledge Distillation

1. 대형 teacher(ResNet-18)를 CIFAR-10에서 93% 이상 정확도로 학습시킵니다
2. 소형 student(약 50K 파라미터의 3층 CNN)를 정의합니다
3. Student를 distillation 없이 일반적으로 학습시키고 정확도를 기록합니다
4. Student를 distillation(T=4, alpha=0.7)으로 학습시키고 정확도를 기록합니다
5. Distillation이 student를 얼마나 개선시켰습니까?

### 연습 문제 2: Temperature 탐구

연습 문제 1의 teacher-student 쌍을 사용하여:
1. Temperature T = 1, 2, 4, 8, 16, 32로 distillation을 수행합니다
2. Student 정확도 대 temperature 그래프를 그립니다
3. 어떤 temperature에서 정확도가 최대가 됩니까?
4. Teacher의 평균 확신도(최대 확률)를 확인합니다. Teacher 확신도와 최적 temperature 사이에 관계가 있습니까?

### 연습 문제 3: Feature Matching

1. 연습 문제 1의 distillation 설정을 feature matching으로 확장합니다
2. 2개의 중간 레이어(conv 블록 이후)에서 feature를 매칭합니다
3. 다음을 비교합니다: (a) 출력만 KD, (b) feature matching만, (c) 둘 다 결합
4. 어떤 접근법이 가장 효과적입니까?

### 연습 문제 4: Self-Distillation

1. ResNet-18을 CIFAR-10에서 학습시킵니다 (Generation 0)
2. 동일한 아키텍처의 새 ResNet-18에 distillation합니다 (Generation 1)
3. Generation 2와 3까지 반복합니다
4. 세대별 정확도를 그래프로 그립니다. 정확도가 계속 향상됩니까?

### 연습 문제 5: DistilBERT 스타일 압축

Hugging Face Transformers를 사용하여:
1. 사전 학습된 BERT-Base 모델을 로드합니다
2. 격번 레이어를 취하여 6층 버전을 생성합니다
3. 텍스트 분류 태스크(예: SST-2)에서 distillation합니다
4. 비교합니다: (a) BERT-Base (teacher), (b) 처음부터 학습된 6층 BERT, (c) distillation된 6층 모델
5. Distillation된 모델이 teacher 정확도의 몇 퍼센트를 유지합니까?

---

[이전: Pruning](./04_Pruning.md) | [Overview](./00_Overview.md) | [다음: Efficient Architectures](./06_Efficient_Architectures.md)

**License**: CC BY-NC 4.0
