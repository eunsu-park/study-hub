"""
03. Knowledge Distillation

Demonstrates teacher-student training with temperature scaling
for compressing large models into smaller, edge-deployable ones.

Covers:
- Teacher and student model architectures
- Soft target generation with temperature scaling
- Combined KD loss (soft + hard targets)
- Training loop with distillation
- Accuracy comparison

Requirements:
    pip install torch torchvision
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

print("=" * 60)
print("Edge AI — Knowledge Distillation")
print("=" * 60)


# ============================================
# 1. Teacher and Student Models
# ============================================
print("\n[1] Define Teacher and Student Models")
print("-" * 40)


class TeacherModel(nn.Module):
    """Large model with high capacity."""

    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        features = self.features(x)
        logits = self.classifier(features)
        return logits


class StudentModel(nn.Module):
    """Small model designed for edge deployment."""

    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Linear(784, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(32, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        features = self.features(x)
        logits = self.classifier(features)
        return logits


teacher = TeacherModel()
student = StudentModel()

teacher_params = sum(p.numel() for p in teacher.parameters())
student_params = sum(p.numel() for p in student.parameters())

print(f"Teacher parameters: {teacher_params:,}")
print(f"Student parameters: {student_params:,}")
print(f"Compression ratio:  {teacher_params / student_params:.1f}x")


# ============================================
# 2. Temperature Scaling and Soft Targets
# ============================================
print("\n[2] Temperature Scaling for Soft Targets")
print("-" * 40)
print("High temperature 'softens' the probability distribution,")
print("revealing the teacher's dark knowledge about class similarities.\n")

# Example logits from a teacher
logits = torch.tensor([[5.0, 2.0, 0.5, -1.0, 0.1, -0.5, 0.2, -2.0, 3.0, 1.0]])

for T in [1.0, 3.0, 5.0, 10.0, 20.0]:
    soft_probs = F.softmax(logits / T, dim=1)
    entropy = -(soft_probs * soft_probs.log()).sum().item()
    top_prob = soft_probs.max().item()
    print(f"T={T:5.1f}  top_prob={top_prob:.4f}  entropy={entropy:.4f}  "
          f"probs={soft_probs.squeeze()[:5].tolist()}")

print()
print("As T increases:")
print("- Distribution becomes softer (more uniform)")
print("- Entropy increases")
print("- Dark knowledge (inter-class relationships) becomes visible")


# ============================================
# 3. Knowledge Distillation Loss
# ============================================
print("\n[3] Knowledge Distillation Loss Function")
print("-" * 40)


def distillation_loss(student_logits, teacher_logits, labels, T, alpha):
    """
    Combined knowledge distillation loss.

    Args:
        student_logits: Raw logits from student model
        teacher_logits: Raw logits from teacher model (detached)
        labels: Ground truth labels
        T: Temperature for softening distributions
        alpha: Weight for KD loss (1 - alpha for hard loss)

    Returns:
        Combined loss = alpha * KD_loss + (1 - alpha) * CE_loss
    """
    # Soft targets: KL divergence between soft student and soft teacher
    soft_student = F.log_softmax(student_logits / T, dim=1)
    soft_teacher = F.softmax(teacher_logits / T, dim=1)
    kd_loss = F.kl_div(soft_student, soft_teacher, reduction="batchmean")
    # Scale by T^2 as gradients are scaled down by 1/T^2
    kd_loss = kd_loss * (T * T)

    # Hard targets: standard cross-entropy with ground truth
    hard_loss = F.cross_entropy(student_logits, labels)

    # Combined loss
    total_loss = alpha * kd_loss + (1 - alpha) * hard_loss
    return total_loss, kd_loss.item(), hard_loss.item()


# Demonstrate with dummy data
dummy_student_logits = torch.randn(4, 10)
dummy_teacher_logits = torch.randn(4, 10)
dummy_labels = torch.tensor([3, 7, 1, 5])

for alpha in [0.0, 0.3, 0.5, 0.7, 1.0]:
    total, kd, hard = distillation_loss(
        dummy_student_logits, dummy_teacher_logits,
        dummy_labels, T=4.0, alpha=alpha
    )
    print(f"alpha={alpha:.1f}: total={total.item():.4f}, "
          f"kd={kd:.4f}, hard={hard:.4f}")

print()
print("alpha=0: pure hard-label training (no distillation)")
print("alpha=1: pure soft-label training (no ground truth)")
print("alpha=0.5-0.7: typical sweet spot for distillation")


# ============================================
# 4. Training Loop with Distillation
# ============================================
print("\n[4] Knowledge Distillation Training Loop")
print("-" * 40)

torch.manual_seed(42)

# Generate synthetic dataset (simulating MNIST-like data)
n_train = 2000
X_train = torch.randn(n_train, 1, 28, 28)
y_train = torch.randint(0, 10, (n_train,))
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Train teacher first (simplified)
print("Training teacher model...")
teacher = TeacherModel()
teacher_opt = optim.Adam(teacher.parameters(), lr=1e-3)
teacher.train()
for epoch in range(5):
    epoch_loss = 0
    for X_batch, y_batch in train_loader:
        logits = teacher(X_batch)
        loss = F.cross_entropy(logits, y_batch)
        teacher_opt.zero_grad()
        loss.backward()
        teacher_opt.step()
        epoch_loss += loss.item()
    print(f"  Teacher epoch {epoch + 1}: loss={epoch_loss / len(train_loader):.4f}")

teacher.eval()


# Train student WITHOUT distillation (baseline)
print("\nTraining student WITHOUT distillation...")
student_baseline = StudentModel()
student_opt = optim.Adam(student_baseline.parameters(), lr=1e-3)
student_baseline.train()
for epoch in range(10):
    epoch_loss = 0
    for X_batch, y_batch in train_loader:
        logits = student_baseline(X_batch)
        loss = F.cross_entropy(logits, y_batch)
        student_opt.zero_grad()
        loss.backward()
        student_opt.step()
        epoch_loss += loss.item()
    if (epoch + 1) % 5 == 0:
        print(f"  Student (no KD) epoch {epoch + 1}: "
              f"loss={epoch_loss / len(train_loader):.4f}")


# Train student WITH distillation
print("\nTraining student WITH knowledge distillation...")
student_kd = StudentModel()
student_kd_opt = optim.Adam(student_kd.parameters(), lr=1e-3)
T = 4.0
alpha = 0.7

student_kd.train()
for epoch in range(10):
    epoch_loss = 0
    epoch_kd_loss = 0
    epoch_hard_loss = 0

    for X_batch, y_batch in train_loader:
        student_logits = student_kd(X_batch)

        with torch.no_grad():
            teacher_logits = teacher(X_batch)

        loss, kd_l, hard_l = distillation_loss(
            student_logits, teacher_logits, y_batch, T=T, alpha=alpha
        )

        student_kd_opt.zero_grad()
        loss.backward()
        student_kd_opt.step()

        epoch_loss += loss.item()
        epoch_kd_loss += kd_l
        epoch_hard_loss += hard_l

    n_batches = len(train_loader)
    if (epoch + 1) % 5 == 0:
        print(f"  Student (KD) epoch {epoch + 1}: "
              f"total={epoch_loss / n_batches:.4f}, "
              f"kd={epoch_kd_loss / n_batches:.4f}, "
              f"hard={epoch_hard_loss / n_batches:.4f}")


# ============================================
# 5. Evaluation Comparison
# ============================================
print("\n[5] Evaluation Comparison")
print("-" * 40)

# Generate test data
n_test = 500
X_test = torch.randn(n_test, 1, 28, 28)
y_test = torch.randint(0, 10, (n_test,))

teacher.eval()
student_baseline.eval()
student_kd.eval()

with torch.no_grad():
    teacher_preds = teacher(X_test).argmax(dim=1)
    baseline_preds = student_baseline(X_test).argmax(dim=1)
    kd_preds = student_kd(X_test).argmax(dim=1)

teacher_acc = (teacher_preds == y_test).float().mean().item()
baseline_acc = (baseline_preds == y_test).float().mean().item()
kd_acc = (kd_preds == y_test).float().mean().item()

# Agreement with teacher
teacher_agreement_baseline = (baseline_preds == teacher_preds).float().mean().item()
teacher_agreement_kd = (kd_preds == teacher_preds).float().mean().item()

print(f"{'Model':<25} {'Params':<12} {'Accuracy':<12} {'Teacher Agreement'}")
print("-" * 65)
print(f"{'Teacher':<25} {teacher_params:<12,} {teacher_acc:<12.1%} {'—'}")
print(f"{'Student (no KD)':<25} {student_params:<12,} {baseline_acc:<12.1%} "
      f"{teacher_agreement_baseline:.1%}")
print(f"{'Student (with KD)':<25} {student_params:<12,} {kd_acc:<12.1%} "
      f"{teacher_agreement_kd:.1%}")

print()
print("Note: On synthetic random data, accuracy is ~10% (random chance).")
print("On real tasks (e.g., MNIST), KD typically improves student accuracy")
print("by 1-5% over training without distillation, with the student")
print(f"achieving this at {teacher_params / student_params:.0f}x fewer parameters.")
