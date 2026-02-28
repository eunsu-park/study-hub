"""
Exercises for Lesson 16: Attention and Transformer
Topic: Deep_Learning

Solutions to practice problems from the lesson.
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


# === Exercise 1: Scaled Dot-Product Attention by Hand ===
# Problem: Implement attention in pure NumPy and verify against PyTorch.

def exercise_1():
    """Scaled dot-product attention in NumPy vs PyTorch."""
    np.random.seed(42)

    seq_len, d_k = 4, 8
    Q = np.random.randn(seq_len, d_k).astype(np.float32)
    K = np.random.randn(seq_len, d_k).astype(np.float32)
    V = np.random.randn(seq_len, d_k).astype(np.float32)

    # NumPy implementation
    scores = Q @ K.T / np.sqrt(d_k)
    # Softmax (numerically stable)
    scores_exp = np.exp(scores - scores.max(axis=-1, keepdims=True))
    weights = scores_exp / scores_exp.sum(axis=-1, keepdims=True)
    context_np = weights @ V

    print(f"  Attention weights (NumPy):\n{weights.round(4)}")

    # PyTorch verification
    Q_t = torch.tensor(Q).unsqueeze(0).unsqueeze(0)  # (1, 1, seq, d_k)
    K_t = torch.tensor(K).unsqueeze(0).unsqueeze(0)
    V_t = torch.tensor(V).unsqueeze(0).unsqueeze(0)

    context_pt = F.scaled_dot_product_attention(Q_t, K_t, V_t)
    context_pt = context_pt.squeeze().numpy()

    diff = np.abs(context_np - context_pt).max()
    print(f"\n  Max difference NumPy vs PyTorch: {diff:.8f}")
    print(f"  Match: {diff < 1e-5}")
    print("\n  Without sqrt(d_k) scaling, dot products grow with d_k, making")
    print("  softmax saturate (nearly one-hot), leading to vanishing gradients.")


# === Exercise 2: Self-Attention Visualization ===
# Problem: Visualize attention weights on a short sentence.

def exercise_2():
    """Self-attention weights for a short sentence."""
    torch.manual_seed(42)

    words = ["The", "cat", "sat", "on", "the", "mat"]
    vocab = {w: i for i, w in enumerate(words)}
    seq_len = len(words)
    d_model = 32
    nhead = 1

    embedding = nn.Embedding(len(vocab), d_model)
    mha = nn.MultiheadAttention(d_model, nhead, batch_first=True)

    ids = torch.tensor([[vocab[w] for w in words]])
    emb = embedding(ids)  # (1, 6, 32)

    attn_output, attn_weights = mha(emb, emb, emb)
    # attn_weights: (1, 6, 6)

    weights = attn_weights[0].detach().numpy()
    print(f"  Attention weight matrix ({seq_len}x{seq_len}):")
    print(f"  {'':>6}", end="")
    for w in words:
        print(f"{w:>6}", end="")
    print()
    for i, w in enumerate(words):
        print(f"  {w:>6}", end="")
        for j in range(seq_len):
            print(f"{weights[i, j]:6.3f}", end="")
        print()

    # Find what "sat" attends to most
    sat_idx = vocab["sat"]
    max_attn_idx = weights[sat_idx].argmax()
    print(f"\n  'sat' attends most strongly to '{words[max_attn_idx]}' "
          f"(weight={weights[sat_idx, max_attn_idx]:.3f})")


# === Exercise 3: Transformer Encoder for Classification ===
# Problem: Classify sequences using nn.TransformerEncoder.

def exercise_3():
    """TransformerEncoder for synthetic sequence classification."""
    torch.manual_seed(42)

    # Synthetic data: sequences of 20 integers, label = mean > 50
    n_samples = 2000
    X = torch.randint(0, 100, (n_samples, 20)).float()
    y = (X.mean(dim=1) > 50).long()

    X_train, y_train = X[:1600], y[:1600]
    X_test, y_test = X[1600:], y[1600:]
    loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)

    class TransformerClassifier(nn.Module):
        def __init__(self, d_model=64, nhead=4, num_layers=2, num_classes=2):
            super().__init__()
            self.input_proj = nn.Linear(1, d_model)
            self.pos_encoding = nn.Parameter(torch.randn(1, 20, d_model) * 0.1)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead, dim_feedforward=128,
                batch_first=True, dropout=0.1
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.fc = nn.Linear(d_model, num_classes)

        def forward(self, x):
            x = self.input_proj(x.unsqueeze(-1))  # (batch, 20, d_model)
            x = x + self.pos_encoding
            x = self.encoder(x)
            x = x.mean(dim=1)  # Mean pooling
            return self.fc(x)

    model = TransformerClassifier()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(30):
        model.train()
        for xb, yb in loader:
            loss = nn.CrossEntropyLoss()(model(xb), yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        acc = (model(X_test).argmax(1) == y_test).float().mean().item()

    print(f"  Transformer classifier test accuracy: {acc:.4f}")

    # Compare with LSTM
    class LSTMClassifier(nn.Module):
        def __init__(self, hidden=64, num_classes=2):
            super().__init__()
            self.lstm = nn.LSTM(1, hidden, batch_first=True)
            self.fc = nn.Linear(hidden, num_classes)

        def forward(self, x):
            _, (h_n, _) = self.lstm(x.unsqueeze(-1))
            return self.fc(h_n[-1])

    torch.manual_seed(42)
    lstm_model = LSTMClassifier()
    optimizer2 = torch.optim.Adam(lstm_model.parameters(), lr=0.001)

    for epoch in range(30):
        lstm_model.train()
        for xb, yb in loader:
            loss = nn.CrossEntropyLoss()(lstm_model(xb), yb)
            optimizer2.zero_grad()
            loss.backward()
            optimizer2.step()

    lstm_model.eval()
    with torch.no_grad():
        lstm_acc = (lstm_model(X_test).argmax(1) == y_test).float().mean().item()

    print(f"  LSTM classifier test accuracy:        {lstm_acc:.4f}")


# === Exercise 4: GQA vs MHA Parameter and Memory Analysis ===
# Problem: Compare Grouped-Query Attention parameter counts.

def exercise_4():
    """Compare MHA, GQA, and MQA parameter counts and KV cache sizes."""

    class GroupedQueryAttention(nn.Module):
        def __init__(self, d_model, n_heads, n_kv_heads):
            super().__init__()
            self.d_model = d_model
            self.n_heads = n_heads
            self.n_kv_heads = n_kv_heads
            self.head_dim = d_model // n_heads
            self.n_groups = n_heads // n_kv_heads

            self.W_q = nn.Linear(d_model, d_model, bias=False)
            self.W_k = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
            self.W_v = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
            self.W_o = nn.Linear(d_model, d_model, bias=False)

        def forward(self, x):
            B, S, _ = x.shape
            Q = self.W_q(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
            K = self.W_k(x).view(B, S, self.n_kv_heads, self.head_dim).transpose(1, 2)
            V = self.W_v(x).view(B, S, self.n_kv_heads, self.head_dim).transpose(1, 2)

            # Repeat K,V for each group
            K = K.repeat_interleave(self.n_groups, dim=1)
            V = V.repeat_interleave(self.n_groups, dim=1)

            attn = (Q @ K.transpose(-2, -1)) / math.sqrt(self.head_dim)
            attn = F.softmax(attn, dim=-1)
            out = (attn @ V).transpose(1, 2).reshape(B, S, self.d_model)
            return self.W_o(out)

    configs = [
        ("MHA (8/8)", 512, 8, 8),
        ("GQA (8/4)", 512, 8, 4),
        ("MQA (8/1)", 512, 8, 1),
    ]

    seq_len = 2048
    print(f"  {'Config':<12} {'KV Params':>10} {'KV Cache (MB)':>14} {'Output Shape':>15}")
    print(f"  {'-'*12} {'-'*10} {'-'*14} {'-'*15}")

    for name, d_model, n_heads, n_kv_heads in configs:
        gqa = GroupedQueryAttention(d_model, n_heads, n_kv_heads)

        kv_params = sum(p.numel() for p in [gqa.W_k.weight, gqa.W_v.weight])

        # KV cache size per layer: 2 * seq_len * n_kv_heads * head_dim * 2 bytes (float16)
        head_dim = d_model // n_heads
        kv_cache_bytes = 2 * seq_len * n_kv_heads * head_dim * 2
        kv_cache_mb = kv_cache_bytes / (1024 * 1024)

        # Forward pass
        x = torch.randn(2, 32, d_model)
        out = gqa(x)

        print(f"  {name:<12} {kv_params:>10,} {kv_cache_mb:>13.2f}  {str(tuple(out.shape)):>15}")

    print("\n  MQA uses fewest KV parameters but may sacrifice quality;")
    print("  GQA is a practical middle ground used in LLaMA 2/3.")


# === Exercise 5: Causal (Autoregressive) Attention Mask ===
# Problem: Implement causal mask and verify future positions get zero weight.

def exercise_5():
    """Causal attention mask implementation and verification."""
    torch.manual_seed(42)

    seq_len = 6
    d_k = 8

    # Create causal mask: upper triangle = -inf
    causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1) * float('-inf')
    print(f"  Causal mask:\n{causal_mask}")

    Q = torch.randn(1, seq_len, d_k)
    K = torch.randn(1, seq_len, d_k)
    V = torch.randn(1, seq_len, d_k)

    # Scaled dot-product attention with mask
    scores = (Q @ K.transpose(-2, -1)) / math.sqrt(d_k)
    scores = scores + causal_mask.unsqueeze(0)
    weights = F.softmax(scores, dim=-1)

    print(f"\n  Attention weights with causal mask:")
    w = weights[0].detach().numpy()
    for i in range(seq_len):
        row = " ".join(f"{w[i, j]:.3f}" for j in range(seq_len))
        print(f"    Position {i}: [{row}]")

    # Verify future weights are zero
    future_weights = weights[0].triu(diagonal=1)
    all_zero = (future_weights == 0).all().item()
    print(f"\n  All future attention weights are zero: {all_zero}")

    # Build small causal Transformer for next-token prediction
    class CausalTransformer(nn.Module):
        def __init__(self, vocab_size=50, d_model=64, nhead=4, num_layers=2):
            super().__init__()
            self.embed = nn.Embedding(vocab_size, d_model)
            self.pos = nn.Parameter(torch.randn(1, 100, d_model) * 0.1)
            layer = nn.TransformerEncoderLayer(d_model, nhead, 128,
                                               batch_first=True, dropout=0.1)
            self.encoder = nn.TransformerEncoder(layer, num_layers)
            self.head = nn.Linear(d_model, vocab_size)

        def forward(self, x):
            S = x.size(1)
            mask = torch.triu(torch.ones(S, S, device=x.device), diagonal=1).bool()
            emb = self.embed(x) + self.pos[:, :S]
            out = self.encoder(emb, mask=mask)
            return self.head(out)

    # Discretized sine wave: 50 bins
    n_bins = 50
    n_samples = 500
    sl = 30
    data = []
    for _ in range(n_samples):
        phase = np.random.uniform(0, 2 * np.pi)
        wave = np.sin(np.linspace(phase, phase + 4 * np.pi, sl))
        binned = ((wave + 1) / 2 * (n_bins - 1)).astype(int)
        data.append(binned)

    data_t = torch.tensor(np.array(data), dtype=torch.long)
    X_tok = data_t[:, :-1]
    y_tok = data_t[:, 1:]

    split = 450
    loader = DataLoader(TensorDataset(X_tok[:split], y_tok[:split]),
                        batch_size=64, shuffle=True)

    model = CausalTransformer(vocab_size=n_bins)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(20):
        model.train()
        for xb, yb in loader:
            logits = model(xb)
            loss = nn.CrossEntropyLoss()(logits.reshape(-1, n_bins), yb.reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        test_logits = model(X_tok[split:])
        test_preds = test_logits.argmax(-1)
        acc = (test_preds == y_tok[split:]).float().mean().item()

    print(f"\n  Causal Transformer next-token accuracy: {acc:.4f}")


if __name__ == "__main__":
    print("=== Exercise 1: Scaled Dot-Product Attention ===")
    exercise_1()
    print("\n=== Exercise 2: Self-Attention Visualization ===")
    exercise_2()
    print("\n=== Exercise 3: Transformer Encoder Classification ===")
    exercise_3()
    print("\n=== Exercise 4: GQA vs MHA Analysis ===")
    exercise_4()
    print("\n=== Exercise 5: Causal Attention Mask ===")
    exercise_5()
    print("\nAll exercises completed!")
