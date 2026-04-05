"""
Exercises for Lesson 05: Knowledge Distillation
Topic: Edge_AI

Solutions to practice problems from the lesson.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# === Exercise 1: Temperature Scaling and Dark Knowledge ===
# Problem: Given teacher logits, analyze how temperature reveals
# inter-class relationships (dark knowledge).

def exercise_1():
    """Analyze dark knowledge at different temperatures."""
    # Teacher logits for an image classification (10 classes)
    # True class: cat (index 3)
    class_names = ["airplane", "car", "bird", "cat", "deer",
                   "dog", "frog", "horse", "ship", "truck"]
    teacher_logits = torch.tensor([
        [0.2, -0.5, 2.1, 8.5, 0.3, 3.2, -0.1, 1.5, -0.3, -0.8]
    ])

    print("  Teacher logits (true class: cat):")
    for name, logit in zip(class_names, teacher_logits[0]):
        print(f"    {name:<10} {logit.item():>6.1f}")

    print("\n  Soft probabilities at different temperatures:\n")
    print(f"  {'Class':<10}", end="")
    for T in [1, 3, 5, 10, 20]:
        print(f"  {'T=' + str(T):>8}", end="")
    print()
    print("  " + "-" * 55)

    for i, name in enumerate(class_names):
        print(f"  {name:<10}", end="")
        for T in [1, 3, 5, 10, 20]:
            prob = F.softmax(teacher_logits / T, dim=1)[0, i].item()
            print(f"  {prob:>8.4f}", end="")
        print()

    # Dark knowledge: at T=1, cat dominates (>99.9%)
    # At higher T, we see that dog is the most similar class to cat
    print("\n  Dark knowledge revealed at high T:")
    T = 10
    probs = F.softmax(teacher_logits / T, dim=1)[0]
    sorted_idx = probs.argsort(descending=True)
    for rank, idx in enumerate(sorted_idx):
        if rank < 5:
            print(f"    {rank+1}. {class_names[idx]:<10} {probs[idx].item():.4f}")

    print("\n  The teacher learned that 'cat' looks most like 'dog',")
    print("  then 'bird', then 'horse' — this inter-class structure is")
    print("  the 'dark knowledge' that helps the student learn better.")


# === Exercise 2: KD Loss Implementation and Analysis ===
# Problem: Implement knowledge distillation loss and verify the gradient
# scaling effect of T^2.

def exercise_2():
    """Implement and analyze the KD loss gradient scaling."""
    # Student and teacher logits
    student_logits = torch.tensor([[2.0, 1.0, 0.5]], requires_grad=True)
    teacher_logits = torch.tensor([[5.0, 3.0, 1.0]])

    print("  KD Loss analysis with T^2 scaling:\n")
    print(f"  {'T':>4} {'KD Loss':>10} {'KD*T^2':>10} {'Grad Norm':>12} {'Scaled Grad':>12}")
    print("  " + "-" * 52)

    for T in [1, 2, 5, 10, 20]:
        # Fresh computation graph
        s = student_logits.detach().clone().requires_grad_(True)

        soft_s = F.log_softmax(s / T, dim=1)
        soft_t = F.softmax(teacher_logits / T, dim=1)

        kd_loss = F.kl_div(soft_s, soft_t, reduction="batchmean")
        kd_loss_scaled = kd_loss * (T * T)

        # Compute gradient of scaled loss
        kd_loss_scaled.backward()
        grad_norm = s.grad.norm().item()

        # Reset for unscaled gradient
        s2 = student_logits.detach().clone().requires_grad_(True)
        soft_s2 = F.log_softmax(s2 / T, dim=1)
        kd_loss2 = F.kl_div(soft_s2, soft_t.detach(), reduction="batchmean")
        kd_loss2.backward()
        unscaled_grad = s2.grad.norm().item()

        print(f"  {T:>4} {kd_loss.item():>10.6f} {kd_loss_scaled.item():>10.6f} "
              f"{unscaled_grad:>12.6f} {grad_norm:>12.6f}")

    print("\n  Without T^2 scaling, gradients shrink as O(1/T^2)")
    print("  The T^2 factor keeps gradient magnitudes roughly constant")
    print("  across different temperatures, enabling consistent learning.")


# === Exercise 3: Student Architecture Design ===
# Problem: Design student architectures of different sizes and compare
# their distillation performance.

def exercise_3():
    """Compare student architectures for distillation."""
    torch.manual_seed(42)

    class Teacher(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(100, 256), nn.ReLU(),
                nn.Linear(256, 128), nn.ReLU(),
                nn.Linear(128, 64), nn.ReLU(),
                nn.Linear(64, 10),
            )
        def forward(self, x):
            return self.net(x)

    def make_student(width):
        return nn.Sequential(
            nn.Linear(100, width), nn.ReLU(),
            nn.Linear(width, 10),
        )

    # Train teacher
    X = torch.randn(500, 100)
    y = torch.randint(0, 10, (500,))

    teacher = Teacher()
    opt = torch.optim.Adam(teacher.parameters(), lr=1e-3)
    teacher.train()
    for _ in range(50):
        loss = nn.CrossEntropyLoss()(teacher(X), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    teacher.eval()

    with torch.no_grad():
        teacher_acc = (teacher(X).argmax(1) == y).float().mean().item()

    print(f"  Teacher: {sum(p.numel() for p in teacher.parameters()):,} params, "
          f"accuracy={teacher_acc:.1%}\n")

    # Train students of different sizes
    results = []
    for width in [8, 16, 32, 64, 128]:
        # Without distillation
        student_plain = make_student(width)
        opt = torch.optim.Adam(student_plain.parameters(), lr=1e-3)
        student_plain.train()
        for _ in range(100):
            loss = nn.CrossEntropyLoss()(student_plain(X), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        student_plain.eval()
        with torch.no_grad():
            plain_acc = (student_plain(X).argmax(1) == y).float().mean().item()

        # With distillation
        student_kd = make_student(width)
        opt = torch.optim.Adam(student_kd.parameters(), lr=1e-3)
        student_kd.train()
        T = 4.0
        alpha = 0.7
        for _ in range(100):
            s_logits = student_kd(X)
            with torch.no_grad():
                t_logits = teacher(X)

            soft_loss = F.kl_div(
                F.log_softmax(s_logits / T, dim=1),
                F.softmax(t_logits / T, dim=1),
                reduction="batchmean"
            ) * (T * T)
            hard_loss = nn.CrossEntropyLoss()(s_logits, y)
            loss = alpha * soft_loss + (1 - alpha) * hard_loss

            opt.zero_grad()
            loss.backward()
            opt.step()

        student_kd.eval()
        with torch.no_grad():
            kd_acc = (student_kd(X).argmax(1) == y).float().mean().item()

        params = sum(p.numel() for p in student_kd.parameters())
        results.append((width, params, plain_acc, kd_acc))

    print(f"  {'Width':>6} {'Params':>8} {'Plain':>8} {'KD':>8} {'KD Gain':>8} {'Compression':>12}")
    print("  " + "-" * 55)
    teacher_params = sum(p.numel() for p in teacher.parameters())
    for width, params, plain, kd in results:
        gain = kd - plain
        ratio = teacher_params / params
        print(f"  {width:>6} {params:>8,} {plain:>8.1%} {kd:>8.1%} "
              f"{gain:>+7.1%} {ratio:>11.1f}x")

    print("\n  KD consistently improves student accuracy, especially")
    print("  for smaller students where the accuracy gap is largest.")


# === Exercise 4: Feature-Based Distillation ===
# Problem: Implement hint-based distillation where intermediate features
# are matched between teacher and student.

def exercise_4():
    """Feature-based (hint) distillation implementation."""
    torch.manual_seed(42)

    class TeacherWithHints(nn.Module):
        def __init__(self):
            super().__init__()
            self.block1 = nn.Sequential(nn.Linear(50, 128), nn.ReLU())
            self.block2 = nn.Sequential(nn.Linear(128, 64), nn.ReLU())
            self.head = nn.Linear(64, 5)

        def forward(self, x, return_hints=False):
            h1 = self.block1(x)
            h2 = self.block2(h1)
            out = self.head(h2)
            if return_hints:
                return out, [h1, h2]
            return out

    class StudentWithHints(nn.Module):
        def __init__(self):
            super().__init__()
            self.block1 = nn.Sequential(nn.Linear(50, 32), nn.ReLU())
            self.block2 = nn.Sequential(nn.Linear(32, 16), nn.ReLU())
            self.head = nn.Linear(16, 5)
            # Adaptation layers: match student hint dims to teacher hint dims
            self.adapt1 = nn.Linear(32, 128)
            self.adapt2 = nn.Linear(16, 64)

        def forward(self, x, return_hints=False):
            h1 = self.block1(x)
            h2 = self.block2(h1)
            out = self.head(h2)
            if return_hints:
                return out, [self.adapt1(h1), self.adapt2(h2)]
            return out

    X = torch.randn(200, 50)
    y = torch.randint(0, 5, (200,))

    # Train teacher
    teacher = TeacherWithHints()
    opt = torch.optim.Adam(teacher.parameters(), lr=1e-3)
    teacher.train()
    for _ in range(50):
        loss = nn.CrossEntropyLoss()(teacher(X), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    teacher.eval()

    # Feature-based distillation
    student = StudentWithHints()
    opt = torch.optim.Adam(student.parameters(), lr=1e-3)
    student.train()

    T = 4.0
    alpha_kd = 0.5      # Weight for logit KD loss
    beta_hint = 0.3      # Weight for hint loss
    alpha_hard = 0.2     # Weight for hard label loss

    for epoch in range(100):
        s_logits, s_hints = student(X, return_hints=True)
        with torch.no_grad():
            t_logits, t_hints = teacher(X, return_hints=True)

        # Logit distillation loss
        kd_loss = F.kl_div(
            F.log_softmax(s_logits / T, dim=1),
            F.softmax(t_logits / T, dim=1),
            reduction="batchmean"
        ) * (T * T)

        # Feature hint loss (MSE between adapted student and teacher features)
        hint_loss = sum(
            F.mse_loss(sh, th) for sh, th in zip(s_hints, t_hints)
        ) / len(s_hints)

        # Hard label loss
        hard_loss = nn.CrossEntropyLoss()(s_logits, y)

        total_loss = alpha_kd * kd_loss + beta_hint * hint_loss + alpha_hard * hard_loss

        opt.zero_grad()
        total_loss.backward()
        opt.step()

    student.eval()
    with torch.no_grad():
        t_acc = (teacher(X).argmax(1) == y).float().mean().item()
        s_acc = (student(X).argmax(1) == y).float().mean().item()

    t_params = sum(p.numel() for p in teacher.parameters())
    s_params = sum(p.numel() for p in student.parameters())

    print(f"  Teacher: {t_params:,} params, accuracy={t_acc:.1%}")
    print(f"  Student: {s_params:,} params, accuracy={s_acc:.1%}")
    print(f"  Compression: {t_params / s_params:.1f}x")
    print(f"\n  Feature-based distillation transfers knowledge at multiple")
    print(f"  levels (intermediate representations), not just final logits.")
    print(f"  Adaptation layers bridge dimension mismatches between")
    print(f"  teacher and student feature spaces.")


if __name__ == "__main__":
    print("=== Exercise 1: Temperature Scaling and Dark Knowledge ===")
    exercise_1()
    print("\n=== Exercise 2: KD Loss Implementation and Analysis ===")
    exercise_2()
    print("\n=== Exercise 3: Student Architecture Design ===")
    exercise_3()
    print("\n=== Exercise 4: Feature-Based Distillation ===")
    exercise_4()
    print("\nAll exercises completed!")
