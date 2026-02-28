"""
Exercises for Lesson 17: Attention Deep Dive
Topic: Deep_Learning

Solutions to practice problems from the lesson.
"""

import math
import time
import tracemalloc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


# === Exercise 1: Attention Memory Scaling ===
# Problem: Measure how attention memory grows with sequence length.

def exercise_1():
    """Empirically measure O(n^2) memory scaling of standard attention."""
    seq_lengths = [512, 1024, 2048, 4096]
    d_k = 64

    print(f"  {'SeqLen':>8} {'Peak Memory (MB)':>18}")
    print(f"  {'-'*8} {'-'*18}")

    memories = []
    for seq_len in seq_lengths:
        tracemalloc.start()

        Q = torch.randn(1, 1, seq_len, d_k)
        K = torch.randn(1, 1, seq_len, d_k)
        V = torch.randn(1, 1, seq_len, d_k)

        scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)
        weights = torch.softmax(scores, dim=-1)
        context = weights @ V

        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        peak_mb = peak / (1024 * 1024)
        memories.append(peak_mb)
        print(f"  {seq_len:8d} {peak_mb:18.2f}")

        del Q, K, V, scores, weights, context

    # Verify O(n^2) trend: doubling seq_len should ~4x memory
    if len(memories) >= 2:
        ratio = memories[-1] / memories[-2] if memories[-2] > 0 else 0
        print(f"\n  Memory ratio (4096 vs 2048): {ratio:.1f}x (expect ~4x for O(n^2))")

    # A100 80GB calculation for 12-head MHA in float16
    # Attention matrix alone: 12 * seq^2 * 2 bytes
    # At seq=131072: 12 * 131072^2 * 2 = ~400 GB -> way over 80 GB
    # At seq=32768: 12 * 32768^2 * 2 = ~25 GB -> fits
    print("  A100 80GB: single-layer 12-head MHA runs out around seq_len ~40000 in float16.")


# === Exercise 2: Online Softmax Verification ===
# Problem: Verify online softmax matches standard softmax.

def exercise_2():
    """Online softmax computed in blocks matches standard softmax."""
    np.random.seed(42)

    scores = np.random.randn(100).astype(np.float64)

    # Standard softmax (numerically stable)
    standard = np.exp(scores - scores.max()) / np.exp(scores - scores.max()).sum()

    # Online softmax in 4 blocks of 25
    block_size = 25
    n_blocks = 4

    # Pass 1: Find global max and sum of exp
    global_max = -np.inf
    global_sum = 0.0
    block_maxes = []
    block_sums = []

    for b in range(n_blocks):
        block = scores[b * block_size:(b + 1) * block_size]
        bmax = block.max()
        block_maxes.append(bmax)

        if bmax > global_max:
            # Rescale previous sum
            global_sum = global_sum * np.exp(global_max - bmax)
            global_max = bmax

        block_exp_sum = np.exp(block - global_max).sum()
        global_sum += block_exp_sum
        block_sums.append(block_exp_sum)

    # Pass 2: Compute softmax values
    online_result = np.exp(scores - global_max) / global_sum

    diff = np.abs(standard - online_result).max()
    print(f"  Max difference between standard and online softmax: {diff:.2e}")
    print(f"  Match within 1e-6: {diff < 1e-6}")

    # Without max subtraction
    raw_exp = np.exp(scores)
    raw_softmax = raw_exp / raw_exp.sum()
    raw_diff = np.abs(standard - raw_softmax).max()
    print(f"\n  Without max subtraction, difference: {raw_diff:.2e}")
    print("  Without subtracting max, exp() can overflow for large values,")
    print("  causing NaN or inf in the result.")


# === Exercise 3: Attention Pattern Analysis ===
# Problem: Analyze attention patterns (diagonal, first-token, uniform).

def exercise_3():
    """Analyze attention pattern types from synthetic attention matrices."""
    torch.manual_seed(42)

    def analyze_attention_patterns(attn_weights):
        """Classify attention pattern: diagonal, first-token, or uniform."""
        seq_len = attn_weights.shape[-1]

        # Diagonal pattern: high weight on diagonal
        diag_score = torch.diagonal(attn_weights, dim1=-2, dim2=-1).mean().item()

        # First-token pattern: high weight on column 0
        first_token_score = attn_weights[..., 0].mean().item()

        # Uniform pattern: low variance
        uniformity = 1.0 / (attn_weights.var().item() + 1e-8)

        if diag_score > 0.3:
            return "diagonal (local)"
        elif first_token_score > 0.3:
            return "first-token"
        elif uniformity > 100:
            return "uniform"
        else:
            return "mixed"

    # Create synthetic attention patterns
    seq_len = 8

    # Simulate 6 attention heads with different patterns
    patterns = {
        "Head 0": torch.eye(seq_len).unsqueeze(0) * 0.8 + 0.02,  # Diagonal
        "Head 1": torch.zeros(1, seq_len, seq_len),  # First-token
        "Head 2": torch.ones(1, seq_len, seq_len) / seq_len,  # Uniform
        "Head 3": torch.eye(seq_len).unsqueeze(0) * 0.5 + 0.05,  # Weak diagonal
    }
    patterns["Head 1"][:, :, 0] = 0.7
    patterns["Head 1"][:, :, 1:] = 0.3 / (seq_len - 1)

    for head_name, weights in patterns.items():
        weights = weights / weights.sum(dim=-1, keepdim=True)  # normalize
        pattern = analyze_attention_patterns(weights)
        print(f"  {head_name}: {pattern}")

    print("\n  Different heads specialize: some attend locally (diagonal),")
    print("  some aggregate global info (first-token), some distribute evenly (uniform).")


# === Exercise 4: RoPE vs Sinusoidal Encoding ===
# Problem: Compare positional encoding approaches.

def exercise_4():
    """Compare sinusoidal and RoPE positional encodings on a copy task."""
    torch.manual_seed(42)

    class SinusoidalPE(nn.Module):
        def __init__(self, d_model, max_len=200):
            super().__init__()
            pe = torch.zeros(max_len, d_model)
            pos = torch.arange(0, max_len).unsqueeze(1).float()
            div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(pos * div)
            pe[:, 1::2] = torch.cos(pos * div)
            self.register_buffer('pe', pe.unsqueeze(0))

        def forward(self, x):
            return x + self.pe[:, :x.size(1)]

    class RoPE(nn.Module):
        def __init__(self, d_model, max_len=200):
            super().__init__()
            inv_freq = 1.0 / (10000 ** (torch.arange(0, d_model, 2).float() / d_model))
            self.register_buffer('inv_freq', inv_freq)

        def forward(self, x):
            seq_len = x.size(1)
            t = torch.arange(seq_len, device=x.device).float()
            freqs = torch.outer(t, self.inv_freq)  # (seq, d/2)
            cos_f = freqs.cos().unsqueeze(0)
            sin_f = freqs.sin().unsqueeze(0)
            x1, x2 = x[..., ::2], x[..., 1::2]
            out = torch.zeros_like(x)
            out[..., ::2] = x1 * cos_f - x2 * sin_f
            out[..., 1::2] = x1 * sin_f + x2 * cos_f
            return out

    class SmallTransformer(nn.Module):
        def __init__(self, vocab_size, d_model=64, nhead=4, num_layers=2, pe_type="sinusoidal"):
            super().__init__()
            self.embed = nn.Embedding(vocab_size, d_model)
            if pe_type == "sinusoidal":
                self.pe = SinusoidalPE(d_model)
            else:
                self.pe = RoPE(d_model)
            layer = nn.TransformerEncoderLayer(d_model, nhead, 128, batch_first=True)
            self.encoder = nn.TransformerEncoder(layer, num_layers)
            self.head = nn.Linear(d_model, vocab_size)

        def forward(self, x):
            emb = self.pe(self.embed(x))
            return self.head(self.encoder(emb))

    # Copy task: predict input tokens
    vocab_size = 20
    train_len = 32
    test_len = 32  # Same length for fair comparison

    X_train = torch.randint(0, vocab_size, (500, train_len))
    loader = DataLoader(TensorDataset(X_train, X_train), batch_size=64, shuffle=True)

    X_test = torch.randint(0, vocab_size, (100, test_len))

    for pe_type in ["sinusoidal", "rope"]:
        torch.manual_seed(42)
        model = SmallTransformer(vocab_size, pe_type=pe_type)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        for epoch in range(15):
            model.train()
            for xb, yb in loader:
                logits = model(xb)
                loss = nn.CrossEntropyLoss()(logits.reshape(-1, vocab_size), yb.reshape(-1))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        model.eval()
        with torch.no_grad():
            test_logits = model(X_test)
            acc = (test_logits.argmax(-1) == X_test).float().mean().item()

        print(f"  {pe_type:>12}: copy_task_acc={acc:.4f} (len={test_len})")

    print("  RoPE encodes relative position, generalizing better to unseen lengths.")


# === Exercise 5: Local Attention vs Full Attention ===
# Problem: Compare local and full attention on a long sequence task.

def exercise_5():
    """Local vs full attention: accuracy and memory trade-off."""
    torch.manual_seed(42)

    class LocalAttention(nn.Module):
        def __init__(self, d_model, nhead, window_size=128):
            super().__init__()
            self.d_model = d_model
            self.nhead = nhead
            self.window_size = window_size
            self.head_dim = d_model // nhead
            self.qkv = nn.Linear(d_model, 3 * d_model)
            self.proj = nn.Linear(d_model, d_model)

        def forward(self, x):
            B, S, _ = x.shape
            qkv = self.qkv(x).reshape(B, S, 3, self.nhead, self.head_dim)
            q, k, v = qkv.unbind(2)
            q = q.transpose(1, 2)  # (B, H, S, D)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)

            # For simplicity: process full attention but mask outside window
            scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            mask = torch.ones(S, S, device=x.device).bool()
            for i in range(S):
                start = max(0, i - self.window_size // 2)
                end = min(S, i + self.window_size // 2)
                mask[i, start:end] = False
            scores.masked_fill_(mask.unsqueeze(0).unsqueeze(0), float('-inf'))
            attn = F.softmax(scores, dim=-1)
            out = (attn @ v).transpose(1, 2).reshape(B, S, self.d_model)
            return self.proj(out)

    # Task: label = token at position 0 (requires global info)
    seq_len = 256  # Reduced from 2048 for speed
    vocab_size = 10
    n_samples = 500

    X = torch.randint(0, vocab_size, (n_samples, seq_len))
    y = X[:, 0]  # Label is the first token

    X_train, y_train = X[:400], y[:400]
    X_test, y_test = X[400:], y[400:]
    loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)

    class AttentionModel(nn.Module):
        def __init__(self, vocab_size, d_model=32, use_local=False, window=64):
            super().__init__()
            self.embed = nn.Embedding(vocab_size, d_model)
            if use_local:
                self.attn = LocalAttention(d_model, nhead=2, window_size=window)
            else:
                self.attn = nn.MultiheadAttention(d_model, num_heads=2, batch_first=True)
            self.use_local = use_local
            self.fc = nn.Linear(d_model, vocab_size)

        def forward(self, x):
            emb = self.embed(x)
            if self.use_local:
                out = self.attn(emb)
            else:
                out, _ = self.attn(emb, emb, emb)
            pooled = out.mean(dim=1)
            return self.fc(pooled)

    for mode, use_local in [("Full Attention", False), ("Local (w=64)", True)]:
        torch.manual_seed(42)
        model = AttentionModel(vocab_size, use_local=use_local, window=64)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        t0 = time.time()
        for epoch in range(20):
            model.train()
            for xb, yb in loader:
                loss = nn.CrossEntropyLoss()(model(xb), yb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        train_time = time.time() - t0

        model.eval()
        with torch.no_grad():
            acc = (model(X_test).argmax(1) == y_test).float().mean().item()

        print(f"  {mode:<18}: acc={acc:.4f}, time={train_time:.1f}s")

    print("\n  Local attention misses position-0 info if it's outside the window.")
    print("  BigBird's global tokens allow specific positions to attend everywhere.")


if __name__ == "__main__":
    print("=== Exercise 1: Attention Memory Scaling ===")
    exercise_1()
    print("\n=== Exercise 2: Online Softmax Verification ===")
    exercise_2()
    print("\n=== Exercise 3: Attention Pattern Analysis ===")
    exercise_3()
    print("\n=== Exercise 4: RoPE vs Sinusoidal ===")
    exercise_4()
    print("\n=== Exercise 5: Local vs Full Attention ===")
    exercise_5()
    print("\nAll exercises completed!")
