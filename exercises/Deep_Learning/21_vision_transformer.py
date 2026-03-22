"""
Exercises for Lesson 21: Vision Transformer
Topic: Deep_Learning

Solutions to practice problems from the lesson.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


# === Exercise 1: Patch Embedding Shape Verification ===
# Problem: Verify patch embedding produces expected tensor shapes.

def exercise_1():
    """Verify PatchEmbedding output shapes for different patch sizes."""

    class PatchEmbedding(nn.Module):
        def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
            super().__init__()
            self.num_patches = (img_size // patch_size) ** 2
            self.proj = nn.Conv2d(in_channels, embed_dim,
                                  kernel_size=patch_size, stride=patch_size)

        def forward(self, x):
            x = self.proj(x)  # (B, embed_dim, H/P, W/P)
            x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
            return x

    x = torch.randn(2, 3, 224, 224)

    for patch_size in [16, 32, 8]:
        pe = PatchEmbedding(img_size=224, patch_size=patch_size, embed_dim=768)
        out = pe(x)
        num_patches = (224 // patch_size) ** 2
        print(f"  patch_size={patch_size}: output={out.shape}, "
              f"num_patches={num_patches} (expected {num_patches})")

    print("\n  Smaller patches -> more tokens -> better resolution but O(n^2) cost.")
    print("  patch_size=8 gives 784 tokens vs 196 for patch_size=16.")


# === Exercise 2: CLS Token Role Verification ===
# Problem: Verify CLS token aggregates global info via cosine similarity.

def exercise_2():
    """CLS token cosine similarity with patch tokens."""
    torch.manual_seed(42)

    class MiniViT(nn.Module):
        def __init__(self, img_size=32, patch_size=8, d_model=64, nhead=4, depth=2):
            super().__init__()
            n_patches = (img_size // patch_size) ** 2
            self.patch_embed = nn.Conv2d(3, d_model, patch_size, stride=patch_size)
            self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
            self.pos_embed = nn.Parameter(torch.randn(1, n_patches + 1, d_model) * 0.02)
            layer = nn.TransformerEncoderLayer(d_model, nhead, 128,
                                               batch_first=True, dropout=0.1)
            self.encoder = nn.TransformerEncoder(layer, depth)

        def forward(self, x):
            B = x.size(0)
            patches = self.patch_embed(x).flatten(2).transpose(1, 2)
            cls = self.cls_token.expand(B, -1, -1)
            x = torch.cat([cls, patches], dim=1) + self.pos_embed
            return self.encoder(x)

    model = MiniViT(img_size=32, patch_size=8)
    x = torch.randn(2, 3, 32, 32)

    model.eval()
    with torch.no_grad():
        out = model(x)  # (2, 17, 64) -> 1 CLS + 16 patches

    cls_token = out[:, 0]      # (2, 64)
    patch_tokens = out[:, 1:]  # (2, 16, 64)

    # Cosine similarity between CLS and each patch
    cls_norm = F.normalize(cls_token.unsqueeze(1), dim=-1)
    patches_norm = F.normalize(patch_tokens, dim=-1)
    sim = (cls_norm * patches_norm).sum(dim=-1)  # (2, 16)

    # Reshape to spatial grid
    grid_size = 4  # 32/8 = 4
    sim_map = sim[0].view(grid_size, grid_size)

    print(f"  Output shape: {out.shape} (batch=2, 1+16 tokens, d_model=64)")
    print(f"\n  CLS-to-patch cosine similarity map (4x4 grid):")
    for i in range(grid_size):
        row = " ".join(f"{sim_map[i, j].item():.3f}" for j in range(grid_size))
        print(f"    [{row}]")

    print("\n  CLS aggregates global info; mean-pooling is an alternative but")
    print("  CLS is standard for classification tasks in ViT.")


# === Exercise 3: ViT Fine-tuning on Synthetic CIFAR-10 ===
# Problem: Compare fine-tuning with differential LR vs from scratch.

def exercise_3():
    """ViT fine-tuning with differential learning rates on synthetic data."""
    torch.manual_seed(42)

    class SimpleViT(nn.Module):
        def __init__(self, img_size=32, patch_size=8, d_model=64, nhead=4,
                     depth=2, num_classes=10):
            super().__init__()
            n_patches = (img_size // patch_size) ** 2
            self.patch_embed = nn.Conv2d(3, d_model, patch_size, stride=patch_size)
            self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
            self.pos_embed = nn.Parameter(torch.randn(1, n_patches + 1, d_model) * 0.02)
            layer = nn.TransformerEncoderLayer(d_model, nhead, 128,
                                               batch_first=True, dropout=0.1)
            self.backbone = nn.TransformerEncoder(layer, depth)
            self.head = nn.Linear(d_model, num_classes)

        def forward(self, x):
            B = x.size(0)
            patches = self.patch_embed(x).flatten(2).transpose(1, 2)
            cls = self.cls_token.expand(B, -1, -1)
            x = torch.cat([cls, patches], dim=1) + self.pos_embed
            x = self.backbone(x)
            return self.head(x[:, 0])

    X = torch.randn(512, 3, 32, 32)
    y = torch.randint(0, 10, (512,))
    X_test = torch.randn(128, 3, 32, 32)
    y_test = torch.randint(0, 10, (128,))
    loader = DataLoader(TensorDataset(X, y), batch_size=64, shuffle=True)

    for mode in ["Differential LR", "From scratch"]:
        torch.manual_seed(42)
        model = SimpleViT()

        if mode == "Differential LR":
            optimizer = torch.optim.Adam([
                {"params": model.backbone.parameters(), "lr": 1e-5},
                {"params": model.head.parameters(), "lr": 1e-3},
                {"params": [model.patch_embed.weight, model.patch_embed.bias,
                            model.cls_token, model.pos_embed], "lr": 1e-5},
            ])
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

        for epoch in range(10):
            model.train()
            for xb, yb in loader:
                loss = nn.CrossEntropyLoss()(model(xb), yb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            scheduler.step()

        model.eval()
        with torch.no_grad():
            acc = (model(X_test).argmax(1) == y_test).float().mean().item()

        print(f"  {mode}: test_acc={acc:.4f}")


# === Exercise 4: Window Attention Complexity Analysis ===
# Problem: Quantify computational advantage of Swin's window attention.

def exercise_4():
    """Window attention complexity analysis for Swin Transformer."""
    H, W = 56, 56
    d_model = 96
    window_size = 7

    N = H * W  # Total tokens

    # Full self-attention: O(N^2)
    full_ops = N * N  # Each token attends to all others

    # Window attention: each window has W_s^2 tokens
    tokens_per_window = window_size ** 2
    n_windows = (H // window_size) * (W // window_size)
    window_ops = n_windows * tokens_per_window * tokens_per_window

    ratio = full_ops / window_ops

    print(f"  Image: {H}x{W}, d_model={d_model}, window_size={window_size}")
    print(f"  Total tokens N = {N}")
    print(f"  Full attention operations: {full_ops:,} (N^2)")
    print(f"  Window attention: {n_windows} windows x {tokens_per_window}^2 = {window_ops:,}")
    print(f"  Speedup ratio: {ratio:.1f}x")

    # General formula: ratio = (H*W)^2 / ((H/W_s)*(W/W_s) * W_s^4) = (H*W)/(W_s^2)
    print(f"\n  General formula: ratio = (H*W) / W_s^2")

    # At what image size is window attention 100x cheaper?
    # (H*W) / W_s^2 > 100 -> H*W > 100 * W_s^2 = 100 * 49 = 4900
    # For square images: H^2 > 4900, H > 70
    threshold = math.sqrt(100 * window_size ** 2)
    print(f"  Window attention is 100x cheaper when image size > {threshold:.0f}x{threshold:.0f}")


# === Exercise 5: DeiT Knowledge Distillation ===
# Problem: Train DeiT-style model with distillation from CNN teacher.

def exercise_5():
    """DeiT knowledge distillation from a CNN teacher."""
    torch.manual_seed(42)

    # Teacher: simple CNN
    class TeacherCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, 10),
            )

        def forward(self, x):
            return self.net(x)

    # Student: small DeiT-style ViT with distillation token
    class DeiTStudent(nn.Module):
        def __init__(self, img_size=32, patch_size=8, d_model=64, nhead=4,
                     depth=2, num_classes=10, use_dist_token=True):
            super().__init__()
            n_patches = (img_size // patch_size) ** 2
            self.use_dist_token = use_dist_token
            self.patch_embed = nn.Conv2d(3, d_model, patch_size, stride=patch_size)
            self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
            n_tokens = n_patches + 1
            if use_dist_token:
                self.dist_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
                n_tokens += 1
            self.pos_embed = nn.Parameter(torch.randn(1, n_tokens, d_model) * 0.02)
            layer = nn.TransformerEncoderLayer(d_model, nhead, 128,
                                               batch_first=True, dropout=0.1)
            self.encoder = nn.TransformerEncoder(layer, depth)
            self.cls_head = nn.Linear(d_model, num_classes)
            if use_dist_token:
                self.dist_head = nn.Linear(d_model, num_classes)

        def forward(self, x):
            B = x.size(0)
            patches = self.patch_embed(x).flatten(2).transpose(1, 2)
            tokens = [self.cls_token.expand(B, -1, -1)]
            if self.use_dist_token:
                tokens.append(self.dist_token.expand(B, -1, -1))
            tokens.append(patches)
            x = torch.cat(tokens, dim=1) + self.pos_embed
            x = self.encoder(x)
            cls_out = self.cls_head(x[:, 0])
            if self.use_dist_token:
                dist_out = self.dist_head(x[:, 1])
                return cls_out, dist_out
            return cls_out

    # Synthetic data
    X = torch.randn(512, 3, 32, 32)
    y = torch.randint(0, 10, (512,))
    X_test = torch.randn(128, 3, 32, 32)
    y_test = torch.randint(0, 10, (128,))
    loader = DataLoader(TensorDataset(X, y), batch_size=64, shuffle=True)

    # Train teacher
    teacher = TeacherCNN()
    opt_t = torch.optim.Adam(teacher.parameters(), lr=0.001)
    for epoch in range(20):
        teacher.train()
        for xb, yb in loader:
            loss = nn.CrossEntropyLoss()(teacher(xb), yb)
            opt_t.zero_grad()
            loss.backward()
            opt_t.step()
    teacher.eval()

    # Train student WITH distillation
    torch.manual_seed(42)
    student_dist = DeiTStudent(use_dist_token=True)
    opt_s = torch.optim.Adam(student_dist.parameters(), lr=0.001)

    for epoch in range(20):
        student_dist.train()
        for xb, yb in loader:
            cls_out, dist_out = student_dist(xb)
            with torch.no_grad():
                teacher_pred = teacher(xb).argmax(dim=1)
            loss_cls = nn.CrossEntropyLoss()(cls_out, yb)
            loss_dist = nn.CrossEntropyLoss()(dist_out, teacher_pred)
            loss = 0.5 * loss_cls + 0.5 * loss_dist
            opt_s.zero_grad()
            loss.backward()
            opt_s.step()

    student_dist.eval()
    with torch.no_grad():
        cls_out, _ = student_dist(X_test)
        acc_dist = (cls_out.argmax(1) == y_test).float().mean().item()

    # Train student WITHOUT distillation
    torch.manual_seed(42)
    student_no_dist = DeiTStudent(use_dist_token=False)
    opt_s2 = torch.optim.Adam(student_no_dist.parameters(), lr=0.001)

    for epoch in range(20):
        student_no_dist.train()
        for xb, yb in loader:
            out = student_no_dist(xb)
            loss = nn.CrossEntropyLoss()(out, yb)
            opt_s2.zero_grad()
            loss.backward()
            opt_s2.step()

    student_no_dist.eval()
    with torch.no_grad():
        acc_no_dist = (student_no_dist(X_test).argmax(1) == y_test).float().mean().item()

    with torch.no_grad():
        teacher_acc = (teacher(X_test).argmax(1) == y_test).float().mean().item()

    print(f"  Teacher CNN:             test_acc={teacher_acc:.4f}")
    print(f"  Student + distillation:  test_acc={acc_dist:.4f}")
    print(f"  Student (no distill):    test_acc={acc_no_dist:.4f}")


if __name__ == "__main__":
    print("=== Exercise 1: Patch Embedding Shapes ===")
    exercise_1()
    print("\n=== Exercise 2: CLS Token Role ===")
    exercise_2()
    print("\n=== Exercise 3: ViT Fine-tuning ===")
    exercise_3()
    print("\n=== Exercise 4: Window Attention Complexity ===")
    exercise_4()
    print("\n=== Exercise 5: DeiT Distillation ===")
    exercise_5()
    print("\nAll exercises completed!")
