"""
Exercises for Lesson 26: Normalization Layers
Topic: Deep_Learning

Solutions to practice problems from the lesson.
"""

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


# === Exercise 1: Compare Normalization Methods on CIFAR-10-like Data ===
# Problem: Compare BatchNorm, GroupNorm, LayerNorm, and no normalization.

def exercise_1():
    """Compare normalization methods on synthetic CIFAR-10 data."""
    torch.manual_seed(42)

    class CIFAR10Net(nn.Module):
        def __init__(self, norm_type='batch'):
            super().__init__()
            self.norm_type = norm_type

            def conv_block(in_c, out_c):
                layers = [nn.Conv2d(in_c, out_c, 3, padding=1)]
                if norm_type == 'batch':
                    layers.append(nn.BatchNorm2d(out_c))
                elif norm_type == 'group':
                    layers.append(nn.GroupNorm(min(32, out_c), out_c))
                elif norm_type == 'layer':
                    layers.append(nn.GroupNorm(1, out_c))  # G=1 = LayerNorm
                layers.append(nn.ReLU(inplace=True))
                return nn.Sequential(*layers)

            self.features = nn.Sequential(
                conv_block(3, 64), conv_block(64, 64), nn.MaxPool2d(2),
                conv_block(64, 128), conv_block(128, 128), nn.MaxPool2d(2),
                conv_block(128, 256), conv_block(256, 256), nn.MaxPool2d(2),
            )
            self.classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(256, 10)
            )

        def forward(self, x):
            return self.classifier(self.features(x))

    # Synthetic data
    X = torch.randn(512, 3, 32, 32)
    y = torch.randint(0, 10, (512,))
    X_test = torch.randn(128, 3, 32, 32)
    y_test = torch.randint(0, 10, (128,))

    print(f"  {'Norm Type':<12} {'BS=64 Acc':>10} {'BS=4 Acc':>10}")
    print(f"  {'-'*12} {'-'*10} {'-'*10}")

    for norm_type in ['none', 'batch', 'group', 'layer']:
        accs = {}
        for batch_size in [64, 4]:
            torch.manual_seed(42)
            model = CIFAR10Net(norm_type)
            loader = DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=True)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

            for epoch in range(10):
                model.train()
                for xb, yb in loader:
                    loss = nn.CrossEntropyLoss()(model(xb), yb)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            model.eval()
            with torch.no_grad():
                acc = (model(X_test).argmax(1) == y_test).float().mean().item()
            accs[batch_size] = acc

        print(f"  {norm_type:<12} {accs[64]:10.4f} {accs[4]:10.4f}")

    print("\n  BatchNorm degrades with small batch sizes (noisy statistics).")
    print("  GroupNorm is more robust across batch sizes.")


# === Exercise 2: RMSNorm vs LayerNorm in Transformers ===
# Problem: Compare RMSNorm and LayerNorm speed and performance.

def exercise_2():
    """RMSNorm vs LayerNorm speed and quality comparison."""
    torch.manual_seed(42)

    class RMSNorm(nn.Module):
        def __init__(self, d_model, eps=1e-6):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(d_model))
            self.eps = eps

        def forward(self, x):
            rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
            return self.weight * x / rms

    class TransformerBlock(nn.Module):
        def __init__(self, d_model, num_heads, norm_cls):
            super().__init__()
            self.attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
            self.norm1 = norm_cls()
            self.norm2 = norm_cls()
            self.ff = nn.Sequential(
                nn.Linear(d_model, d_model * 4), nn.GELU(), nn.Linear(d_model * 4, d_model)
            )

        def forward(self, x):
            x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
            x = x + self.ff(self.norm2(x))
            return x

    class TransformerLM(nn.Module):
        def __init__(self, vocab_size, d_model=128, num_heads=4,
                     num_layers=2, norm_type='layer'):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, d_model)
            self.pos = nn.Parameter(torch.randn(1, 200, d_model) * 0.02)

            if norm_type == 'layer':
                norm_cls = lambda: nn.LayerNorm(d_model)
            else:
                norm_cls = lambda: RMSNorm(d_model)

            self.layers = nn.ModuleList([
                TransformerBlock(d_model, num_heads, norm_cls)
                for _ in range(num_layers)
            ])
            self.output = nn.Linear(d_model, vocab_size)

        def forward(self, x):
            x = self.embedding(x) + self.pos[:, :x.size(1)]
            for layer in self.layers:
                x = layer(x)
            return self.output(x)

    # Character-level data from a simple pattern
    text = "abcdefghij" * 500
    chars = sorted(set(text))
    c2i = {c: i for i, c in enumerate(chars)}
    vocab_size = len(chars)

    seq_len = 30
    data = [c2i[c] for c in text]
    X_seqs = [data[i:i + seq_len] for i in range(len(data) - seq_len)][:1000]
    y_seqs = [data[i + seq_len] for i in range(len(data) - seq_len)][:1000]

    X_t = torch.tensor(X_seqs[:800], dtype=torch.long)
    y_t = torch.tensor(y_seqs[:800], dtype=torch.long)
    loader = DataLoader(TensorDataset(X_t, y_t), batch_size=64, shuffle=True)

    for norm_type in ['layer', 'rms']:
        torch.manual_seed(42)
        model = TransformerLM(vocab_size, norm_type=norm_type)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        t0 = time.time()
        for epoch in range(10):
            model.train()
            for xb, yb in loader:
                logits = model(xb)
                # Use last token prediction
                loss = nn.CrossEntropyLoss()(logits[:, -1, :], yb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        train_time = time.time() - t0

        model.eval()
        with torch.no_grad():
            test_logits = model(X_t[:100])
            test_loss = nn.CrossEntropyLoss()(test_logits[:, -1, :], y_t[:100]).item()

        norm_label = "LayerNorm" if norm_type == 'layer' else "RMSNorm "
        print(f"  {norm_label}: train_time={train_time:.2f}s, test_loss={test_loss:.4f}")

    print("  RMSNorm is faster (no mean subtraction) and matches LayerNorm quality.")
    print("  Used in LLaMA, Gemma, and other modern LLMs.")


# === Exercise 3: Adaptive Instance Normalization (AdaIN) for Style Transfer ===
# Problem: Implement AdaIN and demonstrate style statistics transfer.

def exercise_3():
    """AdaIN implementation for style transfer."""
    torch.manual_seed(42)

    class AdaIN(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, content, style):
            # Calculate mean and std per channel
            c_mean = content.mean(dim=[2, 3], keepdim=True)
            c_std = content.std(dim=[2, 3], keepdim=True) + 1e-6
            s_mean = style.mean(dim=[2, 3], keepdim=True)
            s_std = style.std(dim=[2, 3], keepdim=True) + 1e-6

            # Normalize content, apply style statistics
            normalized = (content - c_mean) / c_std
            return normalized * s_std + s_mean

    class SimpleEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            )
            for param in self.parameters():
                param.requires_grad = False

        def forward(self, x):
            return self.net(x)

    class SimpleDecoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(128, 64, 3, padding=1), nn.ReLU(),
                nn.Upsample(scale_factor=2),
                nn.Conv2d(64, 3, 3, padding=1),
            )

        def forward(self, x):
            return self.net(x)

    class StyleTransferNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = SimpleEncoder()
            self.adain = AdaIN()
            self.decoder = SimpleDecoder()

        def forward(self, content, style, alpha=1.0):
            c_feat = self.encoder(content)
            s_feat = self.encoder(style)
            t = self.adain(c_feat, s_feat)
            # Controllable style transfer
            t = alpha * t + (1 - alpha) * c_feat
            return self.decoder(t)

    model = StyleTransferNet()

    # Synthetic content and style images
    content = torch.randn(1, 3, 32, 32)
    style = torch.randn(1, 3, 32, 32) * 2 + 1  # Different distribution

    # Test AdaIN
    adain = AdaIN()
    c_feat = torch.randn(1, 128, 16, 16)
    s_feat = torch.randn(1, 128, 16, 16) * 3 + 2

    transferred = adain(c_feat, s_feat)

    print(f"  Content features - mean: {c_feat.mean(dim=[2,3]).mean():.4f}, "
          f"std: {c_feat.std(dim=[2,3]).mean():.4f}")
    print(f"  Style features   - mean: {s_feat.mean(dim=[2,3]).mean():.4f}, "
          f"std: {s_feat.std(dim=[2,3]).mean():.4f}")
    print(f"  Transferred      - mean: {transferred.mean(dim=[2,3]).mean():.4f}, "
          f"std: {transferred.std(dim=[2,3]).mean():.4f}")

    print("  AdaIN matches style statistics while preserving content structure.")

    # Test controllable alpha
    for alpha in [0.0, 0.5, 1.0]:
        out = model(content, style, alpha=alpha)
        print(f"  alpha={alpha}: output_mean={out.mean().item():.4f}")

    print("  alpha=0 preserves content, alpha=1 fully applies style.")


if __name__ == "__main__":
    print("=== Exercise 1: Compare Normalization Methods ===")
    exercise_1()
    print("\n=== Exercise 2: RMSNorm vs LayerNorm ===")
    exercise_2()
    print("\n=== Exercise 3: AdaIN for Style Transfer ===")
    exercise_3()
    print("\nAll exercises completed!")
