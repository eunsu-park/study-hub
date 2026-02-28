"""
Exercises for Lesson 36: Self-Supervised Learning
Topic: Deep_Learning

Solutions to practice problems from the lesson.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import torchvision.transforms as T


# === Exercise 1: Understand Contrastive Learning Augmentations ===
# Problem: Analyze SimCLR augmentation pipeline.

def exercise_1():
    """Analyze augmentation effects on contrastive learning."""
    torch.manual_seed(42)

    # SimCLR augmentation pipeline
    augmentation = T.Compose([
        T.RandomResizedCrop(32, scale=(0.2, 1.0)),
        T.RandomHorizontalFlip(p=0.5),
        T.ColorJitter(0.4, 0.4, 0.4, 0.1),
        T.RandomGrayscale(p=0.2),
    ])

    # Generate a synthetic "image" (3x32x32)
    original = torch.rand(3, 32, 32)

    # Apply augmentations 8 times
    print("  8 augmented views of the same image:")
    for i in range(8):
        aug = augmentation(original)
        pixel_diff = (aug - original).abs().mean().item()
        print(f"    View {i}: mean_pixel_change={pixel_diff:.4f}, "
              f"mean={aug.mean():.3f}, std={aug.std():.3f}")

    print("\n  Key augmentations for contrastive learning:")
    print("  - ColorJitter: Most visually drastic. Without it, model can use")
    print("    color as a shortcut (same color = same image, ignoring structure).")
    print("  - GaussianBlur: Encourages invariance to texture/high-frequency detail,")
    print("    forcing the model to learn more semantic features.")
    print("  - RandomCrop: Teaches spatial invariance and multi-scale understanding.")


# === Exercise 2: SimCLR Training ===
# Problem: Train SimCLR on synthetic data with linear evaluation.

def exercise_2():
    """SimCLR training and linear evaluation."""
    torch.manual_seed(42)

    class SimCLR(nn.Module):
        def __init__(self, feature_dim=64, proj_dim=32):
            super().__init__()
            # Backbone (simplified)
            self.backbone = nn.Sequential(
                nn.Linear(100, 128), nn.ReLU(),
                nn.Linear(128, feature_dim),
            )
            # Projection head
            self.projector = nn.Sequential(
                nn.Linear(feature_dim, 64), nn.ReLU(),
                nn.Linear(64, proj_dim),
            )

        def forward(self, x):
            features = self.backbone(x)
            projections = self.projector(features)
            return F.normalize(projections, dim=-1)

        def encode(self, x):
            return self.backbone(x)

    def nt_xent_loss(z1, z2, temperature=0.5):
        """NT-Xent (InfoNCE) loss for SimCLR."""
        batch_size = z1.size(0)
        z = torch.cat([z1, z2], dim=0)  # 2N
        sim = z @ z.T / temperature
        # Mask out self-similarity
        mask = torch.eye(2 * batch_size, dtype=torch.bool)
        sim.masked_fill_(mask, -1e9)
        # Positive pairs: (i, i+N) and (i+N, i)
        labels = torch.cat([torch.arange(batch_size) + batch_size,
                            torch.arange(batch_size)])
        return F.cross_entropy(sim, labels)

    # Synthetic data with class structure
    n_classes = 5
    n_per_class = 200
    X_all = torch.zeros(n_classes * n_per_class, 100)
    y_all = torch.zeros(n_classes * n_per_class, dtype=torch.long)

    for c in range(n_classes):
        center = torch.randn(100) * 2
        X_all[c * n_per_class:(c + 1) * n_per_class] = center + torch.randn(n_per_class, 100) * 0.5
        y_all[c * n_per_class:(c + 1) * n_per_class] = c

    # SimCLR pretraining (create two "augmented" views with noise)
    model = SimCLR()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(20):
        # Create two augmented views
        noise1 = torch.randn_like(X_all) * 0.3
        noise2 = torch.randn_like(X_all) * 0.3
        view1 = X_all + noise1
        view2 = X_all + noise2

        # Mini-batch training
        for i in range(0, len(X_all), 256):
            v1 = view1[i:i + 256]
            v2 = view2[i:i + 256]
            z1 = model(v1)
            z2 = model(v2)
            loss = nt_xent_loss(z1, z2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Linear evaluation
    model.eval()
    with torch.no_grad():
        features = model.encode(X_all)

    # Split
    perm = torch.randperm(len(features))
    train_idx = perm[:800]
    test_idx = perm[800:]

    linear = nn.Linear(64, n_classes)
    opt_linear = torch.optim.Adam(linear.parameters(), lr=0.01)
    loader = DataLoader(TensorDataset(features[train_idx], y_all[train_idx]),
                        batch_size=64, shuffle=True)

    for epoch in range(30):
        for feat, label in loader:
            loss = F.cross_entropy(linear(feat), label)
            opt_linear.zero_grad()
            loss.backward()
            opt_linear.step()

    with torch.no_grad():
        linear_acc = (linear(features[test_idx]).argmax(1) == y_all[test_idx]).float().mean().item()

    print(f"  SimCLR linear probe accuracy: {linear_acc:.4f}")


# === Exercise 3: MoCo Momentum Encoder Analysis ===
# Problem: Analyze the effect of momentum update parameter m.

def exercise_3():
    """Analyze MoCo's momentum encoder behavior."""
    torch.manual_seed(42)

    # Small network
    query_net = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 10))
    key_net = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 10))

    # Copy initial weights
    for q_param, k_param in zip(query_net.parameters(), key_net.parameters()):
        k_param.data.copy_(q_param.data)

    # Simulate SGD updates on query network
    optimizer = torch.optim.SGD(query_net.parameters(), lr=0.01)

    m_values = [0.0, 0.9, 0.999, 1.0]

    print(f"  {'Step':>4}", end="")
    for m in m_values:
        print(f"  {'m='+str(m):>10}", end="")
    print()

    for step in range(100):
        # Update query network with SGD
        x = torch.randn(4, 10)
        loss = query_net(x).sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 20 == 0:
            print(f"  {step:4d}", end="")
            for m in m_values:
                # Reset key network for this m
                temp_key = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 10))
                for kp, qp in zip(temp_key.parameters(), query_net.parameters()):
                    kp.data.copy_(qp.data)

                # Apply momentum update
                for kp, qp in zip(temp_key.parameters(), query_net.parameters()):
                    kp.data = m * kp.data + (1 - m) * qp.data

                # Measure divergence
                diff = sum((kp - qp).norm().item()
                           for kp, qp in zip(temp_key.parameters(), query_net.parameters()))
                print(f"  {diff:10.4f}", end="")
            print()

    print("\n  m=0.0: key = query (no momentum, just copies weights)")
    print("  m=0.999: key slowly tracks query (stable, consistent representations)")
    print("  m=1.0: key never updates (frozen from initialization)")


# === Exercise 4: MAE Masking Visualization ===
# Problem: Visualize different masking ratios for MAE.

def exercise_4():
    """MAE random masking at different ratios."""
    torch.manual_seed(42)

    def random_masking(num_patches, mask_ratio):
        """Return indices of visible patches."""
        n_keep = int(num_patches * (1 - mask_ratio))
        noise = torch.rand(num_patches)
        ids_shuffle = torch.argsort(noise)
        ids_keep = ids_shuffle[:n_keep]
        ids_mask = ids_shuffle[n_keep:]
        return sorted(ids_keep.tolist()), sorted(ids_mask.tolist())

    num_patches = 196  # 14x14 grid for 224x224 image with patch_size=16
    grid_size = 14

    print(f"  Image: 224x224, patch_size=16, num_patches={num_patches}")
    print()

    for mask_ratio in [0.25, 0.50, 0.75, 0.90]:
        visible, masked = random_masking(num_patches, mask_ratio)

        # Create grid visualization (1=visible, 0=masked)
        grid = torch.zeros(grid_size, grid_size)
        for idx in visible:
            r, c = idx // grid_size, idx % grid_size
            grid[r, c] = 1

        n_visible = len(visible)
        n_masked = len(masked)

        print(f"  mask_ratio={mask_ratio}: visible={n_visible}, masked={n_masked}")
        # Show top 4 rows of grid
        for r in range(4):
            row = "".join(["#" if grid[r, c] == 1 else "." for c in range(grid_size)])
            print(f"    {row}")
        print(f"    ... ({grid_size - 4} more rows)")
        print()

    print("  75% masking forces the model to understand global structure,")
    print("  not just copy nearby texture patterns.")
    print("  MAE (reconstruction) is better for dense tasks (detection, segmentation);")
    print("  SimCLR (contrastive) is better for classification tasks.")


# === Exercise 5: Evaluate and Compare SSL Methods ===
# Problem: Compare SSL pre-training vs supervised in low-label regime.

def exercise_5():
    """Compare SSL vs supervised learning in low-label regime."""
    torch.manual_seed(42)

    feature_dim = 64
    n_classes = 5

    # Simulate data
    n_total = 1000
    X_all = torch.randn(n_total, 100)
    y_all = torch.randint(0, n_classes, (n_total,))

    # Add class structure
    for c in range(n_classes):
        mask = y_all == c
        X_all[mask, c * 20:(c + 1) * 20] += 2.0

    # "Pre-trained" SSL encoder (trained on unlabeled data)
    ssl_encoder = nn.Sequential(
        nn.Linear(100, 128), nn.ReLU(), nn.Linear(128, feature_dim)
    )

    # Pretrain with a simple contrastive objective
    ssl_optimizer = torch.optim.Adam(ssl_encoder.parameters(), lr=0.001)
    for epoch in range(30):
        v1 = X_all + torch.randn_like(X_all) * 0.2
        v2 = X_all + torch.randn_like(X_all) * 0.2
        z1 = F.normalize(ssl_encoder(v1), dim=-1)
        z2 = F.normalize(ssl_encoder(v2), dim=-1)
        sim = z1 @ z2.T / 0.5
        labels = torch.arange(n_total)
        loss = F.cross_entropy(sim, labels)
        ssl_optimizer.zero_grad()
        loss.backward()
        ssl_optimizer.step()

    # Evaluate with different label fractions
    label_fractions = [0.01, 0.05, 0.10, 0.50]

    print(f"  {'Labels':>7} {'SSL+Linear':>12} {'SSL+FT':>10} {'Supervised':>12}")
    print(f"  {'-'*7} {'-'*12} {'-'*10} {'-'*12}")

    X_test = X_all[800:]
    y_test = y_all[800:]

    for frac in label_fractions:
        n_labeled = max(n_classes, int(800 * frac))  # At least 1 per class

        # SSL + Linear probe
        ssl_encoder.eval()
        with torch.no_grad():
            train_feat = ssl_encoder(X_all[:n_labeled])
            test_feat = ssl_encoder(X_test)

        probe = nn.Linear(feature_dim, n_classes)
        opt_p = torch.optim.Adam(probe.parameters(), lr=0.01)
        for _ in range(100):
            loss = F.cross_entropy(probe(train_feat), y_all[:n_labeled])
            opt_p.zero_grad()
            loss.backward()
            opt_p.step()

        with torch.no_grad():
            ssl_linear_acc = (probe(test_feat).argmax(1) == y_test).float().mean().item()

        # SSL + Fine-tune
        torch.manual_seed(42)
        ft_enc = nn.Sequential(
            nn.Linear(100, 128), nn.ReLU(), nn.Linear(128, feature_dim)
        )
        ft_enc.load_state_dict(ssl_encoder.state_dict())
        ft_head = nn.Linear(feature_dim, n_classes)
        ft_opt = torch.optim.Adam(list(ft_enc.parameters()) + list(ft_head.parameters()), lr=1e-4)

        for _ in range(100):
            feat = ft_enc(X_all[:n_labeled])
            loss = F.cross_entropy(ft_head(feat), y_all[:n_labeled])
            ft_opt.zero_grad()
            loss.backward()
            ft_opt.step()

        with torch.no_grad():
            ssl_ft_acc = (ft_head(ft_enc(X_test)).argmax(1) == y_test).float().mean().item()

        # Supervised from scratch
        torch.manual_seed(42)
        sup_model = nn.Sequential(
            nn.Linear(100, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, n_classes),
        )
        sup_opt = torch.optim.Adam(sup_model.parameters(), lr=0.001)

        for _ in range(100):
            loss = F.cross_entropy(sup_model(X_all[:n_labeled]), y_all[:n_labeled])
            sup_opt.zero_grad()
            loss.backward()
            sup_opt.step()

        with torch.no_grad():
            sup_acc = (sup_model(X_test).argmax(1) == y_test).float().mean().item()

        pct = f"{frac*100:.0f}%"
        print(f"  {pct:>7} {ssl_linear_acc:12.4f} {ssl_ft_acc:10.4f} {sup_acc:12.4f}")

    print("\n  SSL pre-training shines in low-label regimes because the encoder")
    print("  already learned good representations from unlabeled data.")


if __name__ == "__main__":
    print("=== Exercise 1: Contrastive Augmentations ===")
    exercise_1()
    print("\n=== Exercise 2: SimCLR Training ===")
    exercise_2()
    print("\n=== Exercise 3: MoCo Momentum Encoder ===")
    exercise_3()
    print("\n=== Exercise 4: MAE Masking ===")
    exercise_4()
    print("\n=== Exercise 5: SSL vs Supervised ===")
    exercise_5()
    print("\nAll exercises completed!")
